"""
Energy Integration & Pinch Analysis — identify heat recovery opportunities.

Extracts hot and cold streams from the process, calculates:
  - Composite curves (hot/cold)
  - Grand composite curve
  - Pinch temperature
  - Minimum hot/cold utility requirements
  - Heat recovery potential vs. current configuration

Returns an ``EnergyIntegrationResult`` for UI display (composite curves, tables).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .process_model import NeqSimProcessModel


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class HeatStream:
    """A hot or cold process stream for pinch analysis."""
    name: str
    stream_type: str  # "HOT" or "COLD"
    supply_T_C: float = 0.0
    target_T_C: float = 0.0
    duty_kW: float = 0.0
    mCp_kW_K: float = 0.0  # mass flow × Cp
    equipment_name: str = ""
    equipment_type: str = ""


@dataclass
class CompositePoint:
    """One point on a composite curve (cumulative duty vs temperature)."""
    temperature_C: float = 0.0
    duty_kW: float = 0.0


@dataclass
class HeatRecoverySuggestion:
    """A suggested heat integration match."""
    hot_stream: str
    cold_stream: str
    recoverable_duty_kW: float = 0.0
    hot_T_range: str = ""
    cold_T_range: str = ""
    detail: str = ""


@dataclass
class EnergyIntegrationResult:
    """Complete energy integration / pinch analysis result."""
    hot_streams: List[HeatStream] = field(default_factory=list)
    cold_streams: List[HeatStream] = field(default_factory=list)
    hot_composite: List[CompositePoint] = field(default_factory=list)
    cold_composite: List[CompositePoint] = field(default_factory=list)
    grand_composite: List[CompositePoint] = field(default_factory=list)
    pinch_temperature_C: Optional[float] = None
    min_hot_utility_kW: float = 0.0
    min_cold_utility_kW: float = 0.0
    current_hot_utility_kW: float = 0.0
    current_cold_utility_kW: float = 0.0
    max_heat_recovery_kW: float = 0.0
    current_heat_recovery_kW: float = 0.0
    recovery_potential_kW: float = 0.0
    suggestions: List[HeatRecoverySuggestion] = field(default_factory=list)
    delta_t_min_C: float = 10.0
    method: str = "interval_analysis"
    message: str = ""


# ---------------------------------------------------------------------------
# Stream extraction from process model
# ---------------------------------------------------------------------------

_COOLER_TYPES = {"Cooler", "AirCooler", "WaterCooler"}
_HEATER_TYPES = {"Heater"}
_HX_TYPES = {"HeatExchanger", "MultiStreamHeatExchanger"}


def _extract_heat_streams(model: NeqSimProcessModel) -> Tuple[List[HeatStream], List[HeatStream]]:
    """Extract hot and cold streams from all heat-exchange equipment."""
    hot_streams: List[HeatStream] = []
    cold_streams: List[HeatStream] = []

    proc = model.get_process()
    if proc is None:
        return hot_streams, cold_streams

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
            # Coolers → hot streams needing cooling
            if java_class in _COOLER_TYPES:
                inlet_T = _safe_float(unit, "getInletStream().getTemperature", "C")
                outlet_T = _safe_float(unit, "getOutletStream().getTemperature", "C")
                duty = _safe_duty(unit)
                if inlet_T is not None and outlet_T is not None and inlet_T > outlet_T:
                    dt = inlet_T - outlet_T
                    mCp = abs(duty) / dt if dt > 0.01 else 0.0
                    hot_streams.append(HeatStream(
                        name=f"{u_info.name} (process side)",
                        stream_type="HOT",
                        supply_T_C=inlet_T,
                        target_T_C=outlet_T,
                        duty_kW=abs(duty),
                        mCp_kW_K=mCp,
                        equipment_name=u_info.name,
                        equipment_type=java_class,
                    ))

            # Heaters → cold streams needing heating
            elif java_class in _HEATER_TYPES:
                inlet_T = _safe_float(unit, "getInletStream().getTemperature", "C")
                outlet_T = _safe_float(unit, "getOutletStream().getTemperature", "C")
                duty = _safe_duty(unit)
                if inlet_T is not None and outlet_T is not None and outlet_T > inlet_T:
                    dt = outlet_T - inlet_T
                    mCp = abs(duty) / dt if dt > 0.01 else 0.0
                    cold_streams.append(HeatStream(
                        name=f"{u_info.name} (process side)",
                        stream_type="COLD",
                        supply_T_C=inlet_T,
                        target_T_C=outlet_T,
                        duty_kW=abs(duty),
                        mCp_kW_K=mCp,
                        equipment_name=u_info.name,
                        equipment_type=java_class,
                    ))

            # Compressors → hot gas needing aftercooling (if no cooler exists downstream)
            elif java_class == "Compressor":
                inlet_T = _safe_float(unit, "getInletStream().getTemperature", "C")
                outlet_T = _safe_float(unit, "getOutletStream().getTemperature", "C")
                if inlet_T is not None and outlet_T is not None and outlet_T > inlet_T:
                    # Estimate cooling duty from enthalpy difference or power
                    power_kW = 0.0
                    try:
                        power_kW = abs(float(unit.getPower("kW")))
                    except Exception:
                        pass
                    if power_kW > 0:
                        hot_streams.append(HeatStream(
                            name=f"{u_info.name} discharge",
                            stream_type="HOT",
                            supply_T_C=outlet_T,
                            target_T_C=max(inlet_T, 35.0),  # assume needs cooling to ~35°C
                            duty_kW=power_kW * 0.8,  # approximate
                            mCp_kW_K=power_kW * 0.8 / max(outlet_T - 35.0, 1.0),
                            equipment_name=u_info.name,
                            equipment_type=java_class,
                        ))

        except Exception:
            continue

    return hot_streams, cold_streams


def _safe_float(unit, method_chain: str, *args) -> Optional[float]:
    """Safely call a chained method and return float, or None."""
    try:
        obj = unit
        for m in method_chain.split("."):
            if m.endswith("()"):
                obj = getattr(obj, m[:-2])()
            else:
                obj = getattr(obj, m)
        if args:
            return float(obj(*args))
        return float(obj())
    except Exception:
        return None


def _safe_duty(unit) -> float:
    """Extract duty from a heat exchanger unit."""
    for method in ("getDuty", "getEnergyInput"):
        try:
            val = getattr(unit, method)()
            if val is not None:
                # NeqSim returns duty in W, convert to kW
                return float(val) / 1000.0
        except Exception:
            pass
    return 0.0


# ---------------------------------------------------------------------------
# Pinch analysis — problem table / interval analysis
# ---------------------------------------------------------------------------

def _interval_analysis(
    hot_streams: List[HeatStream],
    cold_streams: List[HeatStream],
    delta_t_min: float = 10.0,
) -> Tuple[Optional[float], float, float, List[CompositePoint], List[CompositePoint], List[CompositePoint]]:
    """
    Perform interval (problem table) analysis.

    Returns:
      (pinch_T, min_hot_utility, min_cold_utility,
       hot_composite, cold_composite, grand_composite)
    """
    if not hot_streams and not cold_streams:
        return None, 0.0, 0.0, [], [], []

    # Shifted temperatures
    # Hot streams: shifted down by ΔTmin/2
    # Cold streams: shifted up by ΔTmin/2
    half_dt = delta_t_min / 2.0

    temp_set: set = set()
    for h in hot_streams:
        temp_set.add(h.supply_T_C - half_dt)
        temp_set.add(h.target_T_C - half_dt)
    for c in cold_streams:
        temp_set.add(c.supply_T_C + half_dt)
        temp_set.add(c.target_T_C + half_dt)

    if len(temp_set) < 2:
        return None, 0.0, 0.0, [], [], []

    intervals = sorted(temp_set, reverse=True)

    # Calculate net heat surplus/deficit in each interval
    cascade: List[float] = []
    for i in range(len(intervals) - 1):
        t_high = intervals[i]
        t_low = intervals[i + 1]
        dt = t_high - t_low

        surplus = 0.0
        for h in hot_streams:
            h_shifted_supply = h.supply_T_C - half_dt
            h_shifted_target = h.target_T_C - half_dt
            if h_shifted_supply >= t_high and h_shifted_target <= t_low:
                surplus += h.mCp_kW_K * dt

        for c in cold_streams:
            c_shifted_supply = c.supply_T_C + half_dt
            c_shifted_target = c.target_T_C + half_dt
            if c_shifted_target >= t_high and c_shifted_supply <= t_low:
                surplus -= c.mCp_kW_K * dt

        cascade.append(surplus)

    # Cascade: cumulative heat flow
    cumulative = [0.0]
    for s in cascade:
        cumulative.append(cumulative[-1] + s)

    min_cascade = min(cumulative)
    hot_utility = -min_cascade if min_cascade < 0 else 0.0

    # Adjusted cascade
    adjusted = [c + hot_utility for c in cumulative]
    cold_utility = adjusted[-1]

    # Pinch location
    pinch_T = None
    for i, val in enumerate(adjusted):
        if abs(val) < 0.01:
            pinch_T = intervals[i] if i < len(intervals) else None
            break

    # Composite curves
    hot_composite = _build_composite(hot_streams, is_hot=True)
    cold_composite = _build_composite(cold_streams, is_hot=False)

    # Grand composite
    grand_composite = []
    for i, t in enumerate(intervals):
        grand_composite.append(CompositePoint(
            temperature_C=t + half_dt,  # un-shift
            duty_kW=adjusted[i],
        ))

    return pinch_T, hot_utility, cold_utility, hot_composite, cold_composite, grand_composite


def _build_composite(streams: List[HeatStream], is_hot: bool) -> List[CompositePoint]:
    """Build a composite curve from hot or cold streams."""
    if not streams:
        return []

    # Collect all temperatures
    temps: set = set()
    for s in streams:
        temps.add(s.supply_T_C)
        temps.add(s.target_T_C)

    sorted_temps = sorted(temps, reverse=is_hot)

    composite: List[CompositePoint] = []
    cumulative_duty = 0.0
    composite.append(CompositePoint(temperature_C=sorted_temps[0], duty_kW=0.0))

    for i in range(len(sorted_temps) - 1):
        t1 = sorted_temps[i]
        t2 = sorted_temps[i + 1]
        dt = abs(t2 - t1)

        segment_mCp = 0.0
        for s in streams:
            s_high = max(s.supply_T_C, s.target_T_C)
            s_low = min(s.supply_T_C, s.target_T_C)
            i_high = max(t1, t2)
            i_low = min(t1, t2)
            if s_high >= i_high and s_low <= i_low:
                segment_mCp += s.mCp_kW_K

        cumulative_duty += segment_mCp * dt
        composite.append(CompositePoint(temperature_C=t2, duty_kW=cumulative_duty))

    return composite


# ---------------------------------------------------------------------------
# Suggestion generator
# ---------------------------------------------------------------------------

def _generate_suggestions(
    hot_streams: List[HeatStream],
    cold_streams: List[HeatStream],
    delta_t_min: float,
) -> List[HeatRecoverySuggestion]:
    """Generate practical heat recovery match suggestions."""
    suggestions: List[HeatRecoverySuggestion] = []

    for h in sorted(hot_streams, key=lambda x: x.duty_kW, reverse=True):
        for c in sorted(cold_streams, key=lambda x: x.duty_kW, reverse=True):
            # Check temperature feasibility
            if h.supply_T_C - c.target_T_C < delta_t_min:
                continue
            if h.target_T_C > c.supply_T_C - delta_t_min:
                continue

            # Recoverable duty = min of available duties, limited by temperature
            recoverable = min(h.duty_kW, c.duty_kW)
            if recoverable < 1.0:
                continue

            suggestions.append(HeatRecoverySuggestion(
                hot_stream=h.name,
                cold_stream=c.name,
                recoverable_duty_kW=recoverable,
                hot_T_range=f"{h.supply_T_C:.0f}→{h.target_T_C:.0f} °C",
                cold_T_range=f"{c.supply_T_C:.0f}→{c.target_T_C:.0f} °C",
                detail=(
                    f"Recover {recoverable:.0f} kW from {h.name} "
                    f"to preheat {c.name}"
                ),
            ))

    # De-duplicate and keep best matches
    suggestions.sort(key=lambda x: x.recoverable_duty_kW, reverse=True)
    return suggestions[:10]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_energy_integration(
    model: NeqSimProcessModel,
    delta_t_min_C: float = 10.0,
) -> EnergyIntegrationResult:
    """
    Run energy integration / pinch analysis on the process.

    Args:
        model: The NeqSim process model to analyze.
        delta_t_min_C: Minimum approach temperature for heat exchange (°C).

    Returns:
        EnergyIntegrationResult with composite curves, pinch T, and suggestions.
    """
    result = EnergyIntegrationResult(delta_t_min_C=delta_t_min_C)

    proc = model.get_process()
    if proc is None:
        result.message = "No process available."
        return result

    # Extract hot and cold streams
    hot_streams, cold_streams = _extract_heat_streams(model)
    result.hot_streams = hot_streams
    result.cold_streams = cold_streams

    if not hot_streams and not cold_streams:
        result.message = (
            "No heating or cooling equipment found in the process. "
            "Energy integration requires at least one heater and one cooler."
        )
        return result

    # Current utility usage
    result.current_hot_utility_kW = sum(c.duty_kW for c in cold_streams)
    result.current_cold_utility_kW = sum(h.duty_kW for h in hot_streams)

    # Pinch analysis
    pinch_T, min_hot, min_cold, hot_comp, cold_comp, grand_comp = _interval_analysis(
        hot_streams, cold_streams, delta_t_min_C,
    )

    result.pinch_temperature_C = pinch_T
    result.min_hot_utility_kW = min_hot
    result.min_cold_utility_kW = min_cold
    result.hot_composite = hot_comp
    result.cold_composite = cold_comp
    result.grand_composite = grand_comp

    # Heat recovery potential
    result.max_heat_recovery_kW = (
        result.current_hot_utility_kW + result.current_cold_utility_kW
        - min_hot - min_cold
    ) / 2.0 if (result.current_hot_utility_kW + result.current_cold_utility_kW) > 0 else 0.0
    result.recovery_potential_kW = max(0.0, result.max_heat_recovery_kW - result.current_heat_recovery_kW)

    # Generate suggestions
    result.suggestions = _generate_suggestions(hot_streams, cold_streams, delta_t_min_C)

    if pinch_T is not None:
        result.message = f"Pinch temperature: {pinch_T:.1f} °C (ΔTmin = {delta_t_min_C:.0f} °C)"
    else:
        result.message = "Pinch analysis completed."

    return result


# ---------------------------------------------------------------------------
# Formatter for LLM
# ---------------------------------------------------------------------------

def format_energy_integration_result(result: EnergyIntegrationResult) -> str:
    """Format the result for LLM consumption."""
    lines = ["=== ENERGY INTEGRATION / PINCH ANALYSIS ==="]
    lines.append(f"Method: {result.method}")
    lines.append(f"ΔTmin: {result.delta_t_min_C:.0f} °C")

    if result.pinch_temperature_C is not None:
        lines.append(f"Pinch Temperature: {result.pinch_temperature_C:.1f} °C")

    lines.append(f"\nCurrent Hot Utility (heaters): {result.current_hot_utility_kW:.1f} kW")
    lines.append(f"Current Cold Utility (coolers): {result.current_cold_utility_kW:.1f} kW")
    lines.append(f"Minimum Hot Utility Required: {result.min_hot_utility_kW:.1f} kW")
    lines.append(f"Minimum Cold Utility Required: {result.min_cold_utility_kW:.1f} kW")
    lines.append(f"Maximum Heat Recovery Potential: {result.max_heat_recovery_kW:.1f} kW")
    lines.append(f"Additional Recoverable: {result.recovery_potential_kW:.1f} kW")

    if result.hot_streams:
        lines.append("\nHOT STREAMS (need cooling):")
        for h in result.hot_streams:
            lines.append(
                f"  {h.name}: {h.supply_T_C:.1f}→{h.target_T_C:.1f} °C, "
                f"Duty={h.duty_kW:.1f} kW ({h.equipment_name})"
            )

    if result.cold_streams:
        lines.append("\nCOLD STREAMS (need heating):")
        for c in result.cold_streams:
            lines.append(
                f"  {c.name}: {c.supply_T_C:.1f}→{c.target_T_C:.1f} °C, "
                f"Duty={c.duty_kW:.1f} kW ({c.equipment_name})"
            )

    if result.suggestions:
        lines.append("\nHEAT RECOVERY SUGGESTIONS:")
        for i, s in enumerate(result.suggestions, 1):
            lines.append(
                f"  {i}. {s.detail} "
                f"(Hot: {s.hot_T_range}, Cold: {s.cold_T_range})"
            )

    if result.message:
        lines.append(f"\n{result.message}")

    return "\n".join(lines)
