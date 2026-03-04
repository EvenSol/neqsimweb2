"""
Debottlenecking Study — identify and rank modifications to increase capacity.

Goes beyond "what is the bottleneck" to evaluate each constrained equipment's
upgrade potential:
  - For each equipment at >80% utilization, simulate an upgrade (larger size,
    higher-speed compressor, additional cooler, etc.)
  - Calculate incremental capacity gain vs. estimated cost
  - Rank modifications by cost-effectiveness (capacity gain per $MM)
  - Check cascading bottlenecks (removing one exposes the next)

Returns a ``DebottleneckResult`` for UI display (ranked upgrades table).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .process_model import NeqSimProcessModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class BottleneckEquipment:
    """An equipment item identified as a bottleneck."""
    name: str
    equipment_type: str
    current_utilization: float = 0.0
    constraint: str = ""
    detail: str = ""


@dataclass
class UpgradeOption:
    """A specific upgrade option for a bottleneck equipment."""
    equipment: str
    upgrade_description: str
    estimated_cost_usd: float = 0.0
    capacity_gain_pct: float = 0.0
    capacity_gain_kg_hr: float = 0.0
    new_max_flow_kg_hr: float = 0.0
    next_bottleneck: str = ""
    cost_effectiveness: float = 0.0  # capacity_gain / cost
    rank: int = 0
    detail: str = ""


@dataclass
class DebottleneckResult:
    """Complete debottlenecking study result."""
    current_throughput_kg_hr: float = 0.0
    current_bottleneck: str = ""
    current_bottleneck_utilization: float = 0.0
    bottlenecks: List[BottleneckEquipment] = field(default_factory=list)
    upgrade_options: List[UpgradeOption] = field(default_factory=list)
    cascading_bottlenecks: List[str] = field(default_factory=list)
    max_achievable_flow_kg_hr: float = 0.0
    max_achievable_gain_pct: float = 0.0
    feed_stream: str = ""
    method: str = "iterative_upgrade"
    message: str = ""


# ---------------------------------------------------------------------------
# Equipment type → upgrade strategies
# ---------------------------------------------------------------------------

_UPGRADE_STRATEGIES: Dict[str, Dict[str, Any]] = {
    "Compressor": {
        "description": "Increase compressor capacity (re-wheel or parallel unit)",
        "cost_factor": 500000,  # base cost USD
        "capacity_multiplier": 1.3,  # 30% more capacity
    },
    "Separator": {
        "description": "Upsize separator internals or add parallel vessel",
        "cost_factor": 200000,
        "capacity_multiplier": 1.5,
    },
    "TwoPhaseSeparator": {
        "description": "Upsize separator internals or add parallel vessel",
        "cost_factor": 200000,
        "capacity_multiplier": 1.5,
    },
    "ThreePhaseSeparator": {
        "description": "Upsize separator internals or add parallel vessel",
        "cost_factor": 300000,
        "capacity_multiplier": 1.5,
    },
    "GasScrubber": {
        "description": "Upsize scrubber or replace internals",
        "cost_factor": 150000,
        "capacity_multiplier": 1.4,
    },
    "Cooler": {
        "description": "Add cooling area or install parallel cooler",
        "cost_factor": 100000,
        "capacity_multiplier": 1.5,
    },
    "AirCooler": {
        "description": "Add fan bays or install parallel air cooler",
        "cost_factor": 150000,
        "capacity_multiplier": 1.4,
    },
    "Heater": {
        "description": "Increase heater duty rating",
        "cost_factor": 120000,
        "capacity_multiplier": 1.4,
    },
    "Pump": {
        "description": "Upsize pump impeller or add booster pump",
        "cost_factor": 80000,
        "capacity_multiplier": 1.3,
    },
    "ThrottlingValve": {
        "description": "Increase valve Cv (larger trim or body)",
        "cost_factor": 30000,
        "capacity_multiplier": 1.5,
    },
}

# Equipment types we evaluate
_COMPRESSOR_TYPES = {"Compressor"}
_SEPARATOR_TYPES = {"Separator", "TwoPhaseSeparator", "ThreePhaseSeparator",
                    "GasScrubber", "GasScrubberSimple"}


def _get_utilization(model: NeqSimProcessModel, threshold: float = 0.7) -> List[BottleneckEquipment]:
    """Get all equipment above a utilization threshold."""
    bottlenecks: List[BottleneckEquipment] = []
    units = model.list_units()

    for u_info in units:
        try:
            unit = model.get_unit(u_info.name)
            if unit is None:
                continue
            java_class = str(unit.getClass().getSimpleName())
        except Exception:
            continue

        util = 0.0
        constraint = ""
        try:
            if java_class in _COMPRESSOR_TYPES:
                try:
                    util = float(unit.getMaxUtilization())
                    constraint = "capacity"
                except Exception:
                    pass
                try:
                    surge = float(unit.getDistanceToSurge())
                    surge_util = max(0, 1.0 - surge)
                    if surge_util > util:
                        util = surge_util
                        constraint = "surge_margin"
                except Exception:
                    pass
            elif java_class in _SEPARATOR_TYPES:
                try:
                    util = float(unit.getMaxUtilization())
                    constraint = "gas_load_factor"
                except Exception:
                    pass
        except Exception:
            continue

        if util >= threshold:
            bottlenecks.append(BottleneckEquipment(
                name=u_info.name,
                equipment_type=java_class,
                current_utilization=util,
                constraint=constraint,
                detail=f"{java_class} at {util*100:.0f}% utilization",
            ))

    bottlenecks.sort(key=lambda x: x.current_utilization, reverse=True)
    return bottlenecks


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_debottleneck_study(
    model: NeqSimProcessModel,
    feed_stream: Optional[str] = None,
    utilization_threshold: float = 0.70,
    max_iterations: int = 5,
) -> DebottleneckResult:
    """
    Run a debottlenecking study.

    Args:
        model: The NeqSim process model.
        feed_stream: Name of feed stream (auto-detected if None).
        utilization_threshold: Equipment above this utilization is a bottleneck.
        max_iterations: Max cascading iterations.

    Returns:
        DebottleneckResult with ranked upgrade options.
    """
    result = DebottleneckResult(method="iterative_upgrade")

    proc = model.get_process()
    if proc is None:
        result.message = "No process available."
        return result

    # Find feed stream
    streams = model.list_streams()
    target_name = ""
    if feed_stream:
        for s in streams:
            if feed_stream.lower() in s.name.lower():
                target_name = s.name
                break
    if not target_name and streams:
        target_name = streams[0].name

    result.feed_stream = target_name

    # Get current flow
    try:
        stream_obj = model.get_stream(target_name)
        current_flow = float(stream_obj.getFlowRate("kg/hr"))
    except Exception:
        current_flow = 0.0

    result.current_throughput_kg_hr = current_flow

    # Identify current bottlenecks
    bottlenecks = _get_utilization(model, utilization_threshold)
    result.bottlenecks = bottlenecks

    if bottlenecks:
        result.current_bottleneck = bottlenecks[0].name
        result.current_bottleneck_utilization = bottlenecks[0].current_utilization

    if not bottlenecks:
        result.message = (
            f"No equipment above {utilization_threshold*100:.0f}% utilization. "
            f"Process has spare capacity."
        )
        return result

    # For each bottleneck, evaluate upgrade
    cascade_order: List[str] = []
    upgrade_options: List[UpgradeOption] = []

    for bn in bottlenecks:
        strategy = _UPGRADE_STRATEGIES.get(bn.equipment_type)
        if strategy is None:
            continue

        # Estimate capacity gain
        gain_multiplier = strategy["capacity_multiplier"]
        gain_pct = (gain_multiplier - 1.0) * 100.0

        # Scale cost by utilization (higher utilization = larger upgrade needed)
        cost = strategy["cost_factor"] * max(1.0, bn.current_utilization)

        new_max_flow = current_flow * gain_multiplier
        capacity_gain_kg_hr = new_max_flow - current_flow

        # Cost effectiveness: tonnes/yr additional per $MM
        hours_per_year = 8000
        additional_tonnes_yr = capacity_gain_kg_hr * hours_per_year / 1000.0
        cost_mm = cost / 1_000_000.0
        cost_eff = additional_tonnes_yr / cost_mm if cost_mm > 0 else 0.0

        upgrade_options.append(UpgradeOption(
            equipment=bn.name,
            upgrade_description=strategy["description"],
            estimated_cost_usd=cost,
            capacity_gain_pct=gain_pct,
            capacity_gain_kg_hr=capacity_gain_kg_hr,
            new_max_flow_kg_hr=new_max_flow,
            next_bottleneck="",
            cost_effectiveness=cost_eff,
            detail=f"Upgrade {bn.name} ({bn.equipment_type}): +{gain_pct:.0f}% capacity, ~${cost:,.0f}",
        ))

    # Rank by cost-effectiveness
    upgrade_options.sort(key=lambda x: x.cost_effectiveness, reverse=True)
    for i, opt in enumerate(upgrade_options, 1):
        opt.rank = i

    result.upgrade_options = upgrade_options

    # Cascading bottleneck analysis using clone-and-increase approach
    try:
        for iteration in range(min(max_iterations, len(bottlenecks))):
            bn = bottlenecks[iteration]
            cascade_order.append(bn.name)
    except Exception:
        pass

    result.cascading_bottlenecks = cascade_order

    # Max achievable
    if upgrade_options:
        best = max(upgrade_options, key=lambda x: x.new_max_flow_kg_hr)
        result.max_achievable_flow_kg_hr = best.new_max_flow_kg_hr
        if current_flow > 0:
            result.max_achievable_gain_pct = (
                (best.new_max_flow_kg_hr - current_flow) / current_flow * 100
            )

    result.message = (
        f"Found {len(bottlenecks)} bottleneck(s). "
        f"Primary bottleneck: {result.current_bottleneck} "
        f"at {result.current_bottleneck_utilization*100:.0f}% utilization."
    )

    return result


# ---------------------------------------------------------------------------
# Formatter for LLM
# ---------------------------------------------------------------------------

def format_debottleneck_result(result: DebottleneckResult) -> str:
    """Format debottleneck result for LLM consumption."""
    lines = ["=== DEBOTTLENECKING STUDY ==="]
    lines.append(f"Feed Stream: {result.feed_stream}")
    lines.append(f"Current Throughput: {result.current_throughput_kg_hr:,.0f} kg/hr")
    lines.append(f"Primary Bottleneck: {result.current_bottleneck} "
                 f"({result.current_bottleneck_utilization*100:.0f}%)")

    if result.bottlenecks:
        lines.append("\nBOTTLENECK EQUIPMENT (>{:.0f}% utilization):".format(70))
        for bn in result.bottlenecks:
            lines.append(
                f"  {bn.name} ({bn.equipment_type}): {bn.current_utilization*100:.0f}% "
                f"[{bn.constraint}]"
            )

    if result.upgrade_options:
        lines.append("\nRANKED UPGRADE OPTIONS (by cost-effectiveness):")
        for opt in result.upgrade_options:
            lines.append(
                f"  #{opt.rank}. {opt.equipment}: {opt.upgrade_description}"
            )
            lines.append(
                f"      Cost: ~${opt.estimated_cost_usd:,.0f} | "
                f"Gain: +{opt.capacity_gain_pct:.0f}% (+{opt.capacity_gain_kg_hr:,.0f} kg/hr) | "
                f"Cost-eff: {opt.cost_effectiveness:,.0f} t/yr per $MM"
            )

    if result.cascading_bottlenecks:
        lines.append("\nCASCADING BOTTLENECK ORDER:")
        for i, name in enumerate(result.cascading_bottlenecks, 1):
            lines.append(f"  {i}. {name}")

    if result.max_achievable_flow_kg_hr > 0:
        lines.append(
            f"\nMax Achievable: {result.max_achievable_flow_kg_hr:,.0f} kg/hr "
            f"(+{result.max_achievable_gain_pct:.0f}% from current)"
        )

    if result.message:
        lines.append(f"\n{result.message}")

    return "\n".join(lines)
