"""
Performance Monitoring — compare simulation predictions vs actual plant data.

Accepts actual measurements (temperatures, pressures, flows, power) and
compares them against model predictions at the same boundary conditions.
Detects performance degradation:
  - Compressor efficiency drop (fouling, seal wear)
  - Heat exchanger UA decline (fouling)
  - Pressure drop increase (restriction)
  - Separator carryover (internals damage)

Returns a ``PerformanceMonitorResult`` for UI display (residual table, alerts).
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
class Measurement:
    """A single actual plant measurement."""
    tag: str                       # e.g. "compressor.outlet_temperature_C"
    actual_value: float = 0.0
    predicted_value: float = 0.0
    residual: float = 0.0          # actual - predicted
    residual_pct: float = 0.0      # % deviation
    unit: str = ""
    status: str = "NORMAL"         # NORMAL, WARNING, ALARM
    diagnosis: str = ""


@dataclass
class DegradationAlert:
    """A performance degradation alert."""
    equipment: str
    alert_type: str               # EFFICIENCY_DROP, FOULING, RESTRICTION, CARRYOVER
    severity: str = "WARNING"     # INFO, WARNING, ALARM
    metric: str = ""
    expected_value: float = 0.0
    actual_value: float = 0.0
    deviation_pct: float = 0.0
    diagnosis: str = ""
    recommendation: str = ""


@dataclass
class PerformanceMonitorResult:
    """Complete performance monitoring result."""
    measurements: List[Measurement] = field(default_factory=list)
    alerts: List[DegradationAlert] = field(default_factory=list)
    overall_status: str = "NORMAL"  # NORMAL, WARNING, ALARM
    total_measurements: int = 0
    warnings_count: int = 0
    alarms_count: int = 0
    boundary_conditions_applied: Dict[str, float] = field(default_factory=dict)
    method: str = "residual_analysis"
    message: str = ""


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_THRESHOLDS = {
    "temperature_C": {"warning": 5.0, "alarm": 15.0},        # °C deviation
    "pressure_bara": {"warning": 2.0, "alarm": 5.0},         # bara
    "flow_kg_hr": {"warning": 10.0, "alarm": 25.0},          # % deviation
    "power_kW": {"warning": 10.0, "alarm": 25.0},            # %
    "duty_kW": {"warning": 10.0, "alarm": 25.0},             # %
    "efficiency": {"warning": 3.0, "alarm": 8.0},            # percentage points
    "default": {"warning": 10.0, "alarm": 25.0},             # %
}

_DIAGNOSIS_MAP = {
    ("Compressor", "power_kW", "high"): (
        "EFFICIENCY_DROP",
        "Compressor consuming more power than expected.",
        "Check for fouling, seal wear, or internal recirculation. "
        "Consider performance test and maintenance inspection.",
    ),
    ("Compressor", "outlet_temperature", "high"): (
        "EFFICIENCY_DROP",
        "Higher discharge temperature indicates reduced efficiency.",
        "Inspect compressor internals, check valve clearances, "
        "verify suction filter condition.",
    ),
    ("Cooler", "outlet_temperature", "high"): (
        "FOULING",
        "Cooler not achieving design outlet temperature — possible fouling.",
        "Check cooling water flow/temperature. Inspect tube bundle. "
        "Consider chemical or mechanical cleaning.",
    ),
    ("Heater", "duty_kW", "high"): (
        "FOULING",
        "Heater duty higher than expected — possible fouling reducing efficiency.",
        "Inspect heater tubes and combustion efficiency.",
    ),
    ("Separator", "pressure_drop", "high"): (
        "RESTRICTION",
        "Higher than expected pressure drop across separator.",
        "Check for internals fouling, blocked demisters, or liquid buildup.",
    ),
}


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------

def _classify_measurement(
    tag: str, actual: float, predicted: float, equipment_type: str
) -> Measurement:
    """Compare one measurement and classify status."""
    residual = actual - predicted
    base = abs(predicted) if abs(predicted) > 1e-6 else 1.0

    # Determine threshold type from tag
    threshold_key = "default"
    unit = ""
    for key in _THRESHOLDS:
        if key in tag:
            threshold_key = key
            break

    if "temperature" in tag:
        unit = "°C"
        residual_pct = residual  # absolute for temperature
    elif "pressure" in tag:
        unit = "bara"
        residual_pct = residual  # absolute
    else:
        unit = _guess_unit(tag)
        residual_pct = (residual / base) * 100.0

    thresholds = _THRESHOLDS.get(threshold_key, _THRESHOLDS["default"])

    abs_dev = abs(residual_pct)
    if abs_dev >= thresholds["alarm"]:
        status = "ALARM"
    elif abs_dev >= thresholds["warning"]:
        status = "WARNING"
    else:
        status = "NORMAL"

    # Diagnosis
    diagnosis = ""
    direction = "high" if residual > 0 else "low"
    if status != "NORMAL":
        for (eq_type, metric_key, dir_key), (_, diag, _) in _DIAGNOSIS_MAP.items():
            if eq_type in equipment_type and metric_key in tag and dir_key == direction:
                diagnosis = diag
                break
        if not diagnosis:
            diagnosis = f"{'Higher' if residual > 0 else 'Lower'} than predicted by {abs_dev:.1f}%"

    return Measurement(
        tag=tag,
        actual_value=actual,
        predicted_value=predicted,
        residual=residual,
        residual_pct=residual_pct,
        unit=unit,
        status=status,
        diagnosis=diagnosis,
    )


def _guess_unit(tag: str) -> str:
    """Guess the engineering unit from tag name."""
    if "kW" in tag or "power" in tag or "duty" in tag:
        return "kW"
    if "kg" in tag or "flow" in tag:
        return "kg/hr"
    if "bar" in tag or "pressure" in tag:
        return "bara"
    if "temperature" in tag or "_C" in tag or "_K" in tag:
        return "°C"
    return ""


def _generate_alerts(measurements: List[Measurement]) -> List[DegradationAlert]:
    """Generate degradation alerts from measurements."""
    alerts: List[DegradationAlert] = []

    # Group by equipment
    equip_issues: Dict[str, List[Measurement]] = {}
    for m in measurements:
        if m.status in ("WARNING", "ALARM"):
            parts = m.tag.split(".", 1)
            equip = parts[0] if len(parts) > 1 else m.tag
            equip_issues.setdefault(equip, []).append(m)

    for equip, issues in equip_issues.items():
        worst = max(issues, key=lambda x: abs(x.residual_pct))

        # Determine alert type and recommendation
        alert_type = "DEVIATION"
        recommendation = "Investigate deviation from expected performance."
        for (eq_type, metric_key, dir_key), (a_type, _, rec) in _DIAGNOSIS_MAP.items():
            for issue in issues:
                direction = "high" if issue.residual > 0 else "low"
                if metric_key in issue.tag and dir_key == direction:
                    alert_type = a_type
                    recommendation = rec
                    break

        severity = "ALARM" if any(m.status == "ALARM" for m in issues) else "WARNING"

        alerts.append(DegradationAlert(
            equipment=equip,
            alert_type=alert_type,
            severity=severity,
            metric=worst.tag,
            expected_value=worst.predicted_value,
            actual_value=worst.actual_value,
            deviation_pct=worst.residual_pct,
            diagnosis=worst.diagnosis,
            recommendation=recommendation,
        ))

    return alerts


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_performance_monitor(
    model: NeqSimProcessModel,
    actual_data: Dict[str, float] = None,
    boundary_conditions: Dict[str, float] = None,
) -> PerformanceMonitorResult:
    """
    Compare model predictions vs actual plant data.

    Args:
        model: The NeqSim process model.
        actual_data: Dict of tag→value actual measurements.
            Tags follow the pattern "equipment_name.property",
            e.g. {"compressor.power_kW": 2500, "cooler_outlet.temperature_C": 42}
        boundary_conditions: Optional dict of boundary conditions to set
            on the model before comparison (e.g. feed T, P, flow).

    Returns:
        PerformanceMonitorResult with residuals and alerts.
    """
    result = PerformanceMonitorResult()

    if not actual_data:
        result.message = (
            "No actual plant data provided. Please supply measurements as a dict, "
            "e.g. {\"compressor.power_kW\": 2500, \"cooler_outlet.temperature_C\": 42}"
        )
        return result

    proc = model.get_process()
    if proc is None:
        result.message = "No process available."
        return result

    # Apply boundary conditions if provided
    if boundary_conditions:
        try:
            from .scenario_engine import apply_patch_to_model
            from .patch_schema import InputPatch
            patch = InputPatch(changes=boundary_conditions)
            apply_patch_to_model(model, patch)
            model.run()
            result.boundary_conditions_applied = boundary_conditions
        except Exception as e:
            result.message = f"Warning: could not apply boundary conditions: {e}"

    # Extract predicted values and compare
    units = model.list_units()
    unit_types = {}
    for u_info in units:
        try:
            unit = model.get_unit(u_info.name)
            if unit:
                unit_types[u_info.name] = str(unit.getClass().getSimpleName())
        except Exception:
            unit_types[u_info.name] = "Unknown"

    for tag, actual_value in actual_data.items():
        # Parse tag: "equipment.property"
        parts = tag.rsplit(".", 1)
        if len(parts) < 2:
            continue

        equip_name = parts[0]
        prop = parts[1]

        # Get predicted value from model
        predicted = _get_predicted_value(model, equip_name, prop)
        if predicted is None:
            continue

        equipment_type = unit_types.get(equip_name, "Unknown")
        measurement = _classify_measurement(tag, actual_value, predicted, equipment_type)
        result.measurements.append(measurement)

    # Generate alerts
    result.alerts = _generate_alerts(result.measurements)

    # Summary
    result.total_measurements = len(result.measurements)
    result.warnings_count = sum(1 for m in result.measurements if m.status == "WARNING")
    result.alarms_count = sum(1 for m in result.measurements if m.status == "ALARM")

    if result.alarms_count > 0:
        result.overall_status = "ALARM"
    elif result.warnings_count > 0:
        result.overall_status = "WARNING"
    else:
        result.overall_status = "NORMAL"

    result.message = (
        f"Compared {result.total_measurements} measurements: "
        f"{result.warnings_count} warnings, {result.alarms_count} alarms."
    )

    return result


def _get_predicted_value(
    model: NeqSimProcessModel, equip_name: str, prop: str
) -> Optional[float]:
    """Extract a predicted value from the model for a given equipment + property."""
    try:
        unit = model.get_unit(equip_name)
        if unit is None:
            # Try as stream
            try:
                stream = model.get_stream(equip_name)
                return _extract_stream_prop(stream, prop)
            except Exception:
                return None

        return _extract_unit_prop(unit, prop)
    except Exception:
        return None


def _extract_unit_prop(unit, prop: str) -> Optional[float]:
    """Extract a property value from a unit operation."""
    prop_lower = prop.lower()

    mappings = {
        "power_kw": lambda u: float(u.getPower("kW")),
        "duty_kw": lambda u: float(u.getDuty()) / 1000.0,
        "outlet_temperature_c": lambda u: float(u.getOutletStream().getTemperature("C")),
        "outlet_temperature": lambda u: float(u.getOutletStream().getTemperature("C")),
        "inlet_temperature_c": lambda u: float(u.getInletStream().getTemperature("C")),
        "inlet_temperature": lambda u: float(u.getInletStream().getTemperature("C")),
        "outlet_pressure_bara": lambda u: float(u.getOutletStream().getPressure("bara")),
        "outlet_pressure": lambda u: float(u.getOutletStream().getPressure("bara")),
        "inlet_pressure_bara": lambda u: float(u.getInletStream().getPressure("bara")),
        "inlet_pressure": lambda u: float(u.getInletStream().getPressure("bara")),
        "temperature_c": lambda u: float(u.getOutletStream().getTemperature("C")),
        "pressure_bara": lambda u: float(u.getOutletStream().getPressure("bara")),
    }

    # Try inlet methods with fallback
    inlet_methods = ("getInletStream", "getInStream", "getFeed", "getFeedStream")

    for key, getter in mappings.items():
        if key == prop_lower:
            try:
                return getter(unit)
            except Exception:
                # Try with inlet fallback
                if "inlet" in key:
                    for m in inlet_methods:
                        try:
                            inlet = getattr(unit, m)()
                            if "temperature" in key:
                                return float(inlet.getTemperature("C"))
                            elif "pressure" in key:
                                return float(inlet.getPressure("bara"))
                        except Exception:
                            continue
                return None

    return None


def _extract_stream_prop(stream, prop: str) -> Optional[float]:
    """Extract a property from a stream object."""
    prop_lower = prop.lower()
    try:
        if "temperature" in prop_lower:
            return float(stream.getTemperature("C"))
        elif "pressure" in prop_lower:
            return float(stream.getPressure("bara"))
        elif "flow" in prop_lower:
            return float(stream.getFlowRate("kg/hr"))
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Formatter for LLM
# ---------------------------------------------------------------------------

def format_performance_monitor_result(result: PerformanceMonitorResult) -> str:
    """Format the result for LLM consumption."""
    lines = ["=== PERFORMANCE MONITORING RESULTS ==="]
    lines.append(f"Overall Status: {result.overall_status}")
    lines.append(f"Measurements: {result.total_measurements} "
                 f"({result.warnings_count} warnings, {result.alarms_count} alarms)")

    if result.boundary_conditions_applied:
        lines.append("\nBoundary Conditions Applied:")
        for k, v in result.boundary_conditions_applied.items():
            lines.append(f"  {k}: {v}")

    if result.measurements:
        lines.append("\nMEASUREMENTS:")
        lines.append(f"{'Tag':<40} {'Actual':>10} {'Predicted':>10} {'Residual':>10} {'Status':>8}")
        for m in result.measurements:
            lines.append(
                f"{m.tag:<40} {m.actual_value:>10.2f} {m.predicted_value:>10.2f} "
                f"{m.residual:>+10.2f} {m.status:>8}"
            )

    if result.alerts:
        lines.append("\nDEGRADATION ALERTS:")
        for a in result.alerts:
            lines.append(
                f"  [{a.severity}] {a.equipment}: {a.alert_type} — {a.diagnosis}"
            )
            lines.append(f"    Recommendation: {a.recommendation}")

    if result.message:
        lines.append(f"\n{result.message}")

    return "\n".join(lines)
