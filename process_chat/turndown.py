"""
Turndown / Operating Envelope Analysis — map plant performance across feed range.

Sweeps feed flow from minimum to maximum and records equipment utilization,
compressor operating point (vs surge/stonewall), separator levels, power/duty
at each step. Identifies:
  - Minimum stable flow (turndown limit)
  - Maximum throughput
  - Operating envelope boundaries
  - Which equipment limits turndown vs. max capacity

Returns a ``TurndownResult`` for UI display (envelope chart, utilization table).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .process_model import NeqSimProcessModel, KPI

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class EnvelopePoint:
    """One operating point in the envelope sweep."""
    flow_pct: float = 0.0           # % of design flow
    flow_kg_hr: float = 0.0
    feasible: bool = True
    equipment_utilizations: Dict[str, float] = field(default_factory=dict)  # name → 0-1+
    limiting_equipment: str = ""
    limiting_constraint: str = ""
    max_utilization: float = 0.0
    kpis: Dict[str, float] = field(default_factory=dict)  # key KPIs at this point
    detail: str = ""


@dataclass
class EnvelopeBoundary:
    """A boundary condition in the operating envelope."""
    equipment: str
    constraint: str
    limit_type: str  # "TURNDOWN" or "MAX_CAPACITY"
    flow_pct: float = 0.0
    flow_kg_hr: float = 0.0
    detail: str = ""


@dataclass
class TurndownResult:
    """Complete turndown / operating envelope result."""
    design_flow_kg_hr: float = 0.0
    min_stable_flow_kg_hr: float = 0.0
    min_stable_flow_pct: float = 0.0
    max_flow_kg_hr: float = 0.0
    max_flow_pct: float = 0.0
    envelope_points: List[EnvelopePoint] = field(default_factory=list)
    boundaries: List[EnvelopeBoundary] = field(default_factory=list)
    turndown_limit_equipment: str = ""
    max_capacity_equipment: str = ""
    safe_operating_range_pct: str = ""  # e.g. "45% – 115%"
    feed_stream: str = ""
    method: str = "sweep_and_evaluate"
    message: str = ""


# ---------------------------------------------------------------------------
# Equipment utilization extraction (shared with optimizer)
# ---------------------------------------------------------------------------

_COMPRESSOR_TYPES = {"Compressor"}
_SEPARATOR_TYPES = {"Separator", "TwoPhaseSeparator", "ThreePhaseSeparator",
                    "GasScrubber", "GasScrubberSimple"}


def _get_equipment_utilization(model: NeqSimProcessModel) -> Dict[str, Tuple[float, str]]:
    """Get utilization ratio and constraint for each equipment."""
    utilizations: Dict[str, Tuple[float, str]] = {}
    units = model.list_units()

    for u_info in units:
        try:
            unit = model.get_unit(u_info.name)
            if unit is None:
                continue
            java_class = str(unit.getClass().getSimpleName())
        except Exception:
            continue

        try:
            if java_class in _COMPRESSOR_TYPES:
                util = 0.0
                constraint = "power"
                # Check surge distance
                try:
                    surge_dist = float(unit.getDistanceToSurge())
                    if surge_dist >= 0:
                        # Surge utilization = 1 - distance_to_surge (closer = higher)
                        surge_util = max(0, 1.0 - surge_dist)
                        if surge_util > util:
                            util = surge_util
                            constraint = "surge_margin"
                except Exception:
                    pass
                # Check max utilization
                try:
                    max_util = float(unit.getMaxUtilization())
                    if max_util > util:
                        util = max_util
                        constraint = "capacity"
                except Exception:
                    pass
                if util > 0:
                    utilizations[u_info.name] = (util, constraint)

            elif java_class in _SEPARATOR_TYPES:
                try:
                    max_util = float(unit.getMaxUtilization())
                    utilizations[u_info.name] = (max_util, "gas_load_factor")
                except Exception:
                    pass

        except Exception:
            continue

    return utilizations


def _extract_key_kpis(model: NeqSimProcessModel) -> Dict[str, float]:
    """Extract key KPIs from the current model state."""
    kpis: Dict[str, float] = {}
    units = model.list_units()

    for u_info in units:
        try:
            unit = model.get_unit(u_info.name)
            if unit is None:
                continue
            java_class = str(unit.getClass().getSimpleName())
        except Exception:
            continue

        try:
            if java_class in _COMPRESSOR_TYPES:
                try:
                    kpis[f"{u_info.name}.power_kW"] = float(unit.getPower("kW"))
                except Exception:
                    pass
            if java_class in {"Cooler", "Heater", "AirCooler", "WaterCooler"}:
                try:
                    kpis[f"{u_info.name}.duty_kW"] = float(unit.getDuty()) / 1000.0
                except Exception:
                    pass
        except Exception:
            continue

    return kpis


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_turndown_analysis(
    model: NeqSimProcessModel,
    feed_stream: Optional[str] = None,
    min_pct: float = 30.0,
    max_pct: float = 130.0,
    n_points: int = 11,
    utilization_limit: float = 1.0,
) -> TurndownResult:
    """
    Sweep feed flow and map the operating envelope.

    Args:
        model: The NeqSim process model.
        feed_stream: Name of feed stream (auto-detected if None).
        min_pct: Minimum flow as % of current (default 30%).
        max_pct: Maximum flow as % of current (default 130%).
        n_points: Number of sweep points.
        utilization_limit: Max allowed utilization (default 1.0 = 100%).

    Returns:
        TurndownResult with envelope data.
    """
    result = TurndownResult(method="sweep_and_evaluate")

    proc = model.get_process()
    if proc is None:
        result.message = "No process available."
        return result

    # Find feed stream
    streams = model.list_streams()
    target_stream = None
    target_name = ""

    if feed_stream:
        for s in streams:
            if feed_stream.lower() in s.name.lower():
                target_name = s.name
                break
    if not target_name and streams:
        target_name = streams[0].name

    if not target_name:
        result.message = "No streams found in the process."
        return result

    result.feed_stream = target_name

    # Get current flow rate
    try:
        stream_obj = model.get_stream(target_name)
        design_flow = float(stream_obj.getFlowRate("kg/hr"))
    except Exception:
        result.message = f"Could not read flow rate from '{target_name}'."
        return result

    result.design_flow_kg_hr = design_flow

    # Sweep
    flow_pcts = []
    step = (max_pct - min_pct) / max(n_points - 1, 1)
    for i in range(n_points):
        flow_pcts.append(min_pct + i * step)

    min_feasible_pct = max_pct
    max_feasible_pct = min_pct
    turndown_equip = ""
    maxcap_equip = ""

    for pct in flow_pcts:
        try:
            clone = model.clone()
            if clone is None:
                continue

            test_flow = design_flow * pct / 100.0
            # Set flow on the feed stream
            try:
                s = clone.get_stream(target_name)
                s.setFlowRate(test_flow, "kg/hr")
            except Exception:
                continue

            run_result = clone.run()

            # Get utilizations
            utils = _get_equipment_utilization(clone)
            kpis = _extract_key_kpis(clone)

            max_util = 0.0
            lim_equip = ""
            lim_constraint = ""
            for name, (u, c) in utils.items():
                if u > max_util:
                    max_util = u
                    lim_equip = name
                    lim_constraint = c

            feasible = max_util <= utilization_limit

            point = EnvelopePoint(
                flow_pct=pct,
                flow_kg_hr=test_flow,
                feasible=feasible,
                equipment_utilizations={n: v[0] for n, v in utils.items()},
                limiting_equipment=lim_equip,
                limiting_constraint=lim_constraint,
                max_utilization=max_util,
                kpis=kpis,
            )
            result.envelope_points.append(point)

            if feasible:
                if pct < min_feasible_pct:
                    min_feasible_pct = pct
                if pct > max_feasible_pct:
                    max_feasible_pct = pct

            # Track boundaries
            if not feasible and pct < 100:
                turndown_equip = lim_equip
            if not feasible and pct > 100:
                maxcap_equip = lim_equip

        except Exception as e:
            logger.warning(f"Turndown sweep at {pct:.0f}%: {e}")
            continue

    # Restore original flow
    try:
        s_orig = model.get_stream(target_name)
        s_orig.setFlowRate(design_flow, "kg/hr")
        model.run()
    except Exception:
        pass

    # Determine boundaries
    result.min_stable_flow_pct = min_feasible_pct
    result.min_stable_flow_kg_hr = design_flow * min_feasible_pct / 100.0
    result.max_flow_pct = max_feasible_pct
    result.max_flow_kg_hr = design_flow * max_feasible_pct / 100.0
    result.turndown_limit_equipment = turndown_equip
    result.max_capacity_equipment = maxcap_equip
    result.safe_operating_range_pct = f"{min_feasible_pct:.0f}% – {max_feasible_pct:.0f}%"

    # Build boundary records
    if turndown_equip:
        result.boundaries.append(EnvelopeBoundary(
            equipment=turndown_equip,
            constraint="turndown",
            limit_type="TURNDOWN",
            flow_pct=min_feasible_pct,
            flow_kg_hr=result.min_stable_flow_kg_hr,
            detail=f"Minimum stable flow limited by {turndown_equip}",
        ))
    if maxcap_equip:
        result.boundaries.append(EnvelopeBoundary(
            equipment=maxcap_equip,
            constraint="capacity",
            limit_type="MAX_CAPACITY",
            flow_pct=max_feasible_pct,
            flow_kg_hr=result.max_flow_kg_hr,
            detail=f"Maximum capacity limited by {maxcap_equip}",
        ))

    result.message = (
        f"Operating envelope: {result.safe_operating_range_pct} of design flow "
        f"({result.min_stable_flow_kg_hr:,.0f} – {result.max_flow_kg_hr:,.0f} kg/hr)"
    )

    return result


# ---------------------------------------------------------------------------
# Formatter for LLM
# ---------------------------------------------------------------------------

def format_turndown_result(result: TurndownResult) -> str:
    """Format turndown result for LLM consumption."""
    lines = ["=== TURNDOWN / OPERATING ENVELOPE ANALYSIS ==="]
    lines.append(f"Feed Stream: {result.feed_stream}")
    lines.append(f"Design Flow: {result.design_flow_kg_hr:,.0f} kg/hr")
    lines.append(f"Safe Operating Range: {result.safe_operating_range_pct}")
    lines.append(f"Min Stable Flow: {result.min_stable_flow_kg_hr:,.0f} kg/hr ({result.min_stable_flow_pct:.0f}%)")
    lines.append(f"Max Flow: {result.max_flow_kg_hr:,.0f} kg/hr ({result.max_flow_pct:.0f}%)")

    if result.turndown_limit_equipment:
        lines.append(f"Turndown Limited By: {result.turndown_limit_equipment}")
    if result.max_capacity_equipment:
        lines.append(f"Max Capacity Limited By: {result.max_capacity_equipment}")

    if result.envelope_points:
        lines.append("\nENVELOPE SWEEP:")
        lines.append(f"{'Flow%':>6} {'Flow(kg/hr)':>12} {'Feasible':>8} {'MaxUtil%':>8} {'Limiting':>20}")
        for p in result.envelope_points:
            lines.append(
                f"{p.flow_pct:>6.0f} {p.flow_kg_hr:>12,.0f} "
                f"{'YES' if p.feasible else 'NO':>8} "
                f"{p.max_utilization*100:>7.1f}% "
                f"{p.limiting_equipment:>20}"
            )

    if result.boundaries:
        lines.append("\nBOUNDARIES:")
        for b in result.boundaries:
            lines.append(f"  {b.limit_type}: {b.detail} at {b.flow_pct:.0f}%")

    if result.message:
        lines.append(f"\n{result.message}")

    return "\n".join(lines)
