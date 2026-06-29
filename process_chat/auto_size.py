"""
Auto-Size — automatically size all process equipment and extract utilization.

Uses NeqSim's ``AutoSizeable`` interface to calculate equipment dimensions,
extract sizing reports, and compute utilization ratios for capacity tracking
and production optimization.

Supported equipment:
  - Separator, ThreePhaseSeparator, GasScrubber — K-factor sizing
  - ThrottlingValve — Cv calculation (IEC 60534)
  - Heater, Cooler — duty-based sizing
  - HeatExchanger — UA/LMTD sizing
  - Compressor — chart generation + design limits
  - Pipeline — velocity-based sizing

Features:
  - ``auto_size_all()`` — size all equipment with optional safety factor
  - ``get_utilization_report()`` — utilization ratios for all equipment
  - ``get_sizing_report()`` — dimensional sizing data
  - ``get_bottleneck()`` — identify the process bottleneck
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .process_model import NeqSimProcessModel


# ---------------------------------------------------------------------------
# Equipment types that support autoSize
# ---------------------------------------------------------------------------

_AUTOSIZEABLE_TYPES = {
    "Separator", "TwoPhaseSeparator", "ThreePhaseSeparator",
    "GasScrubber", "GasScrubberSimple",
    "ThrottlingValve", "ControlValve",
    "Heater", "Cooler", "AirCooler", "WaterCooler",
    "HeatExchanger",
    "Compressor",
    "Pump", "ESPPump",
    "Pipeline", "PipeBeggsAndBrills", "AdiabaticPipe",
    "Manifold",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SizingInfo:
    """Sizing data for one piece of equipment."""
    name: str
    equipment_type: str
    auto_sized: bool
    sizing_data: Dict[str, Any] = field(default_factory=dict)
    message: str = ""


@dataclass
class UtilizationInfo:
    """Utilization data for one piece of equipment."""
    name: str
    equipment_type: str
    utilization_pct: float       # 0–100+
    constraint_name: str = ""    # which constraint is limiting
    constraint_value: float = 0.0
    design_value: float = 0.0
    unit: str = ""
    detail: str = ""
    is_bottleneck: bool = False


@dataclass
class AutoSizeResult:
    """Result of auto-sizing the process."""
    equipment_sized: List[SizingInfo]
    utilization: List[UtilizationInfo]
    bottleneck_name: str = ""
    bottleneck_constraint: str = ""
    bottleneck_utilization_pct: float = 0.0
    total_equipment: int = 0
    sized_count: int = 0
    message: str = ""


# ---------------------------------------------------------------------------
# Auto-sizing
# ---------------------------------------------------------------------------

def auto_size_all(
    model: NeqSimProcessModel,
    safety_factor: float = 1.2,
    generate_compressor_charts: bool = True,
    chart_template: str = "CENTRIFUGAL_STANDARD",
    chart_num_speeds: int = 5,
    skip_already_sized: Optional[set] = None,
) -> AutoSizeResult:
    """
    Auto-size all equipment in the process and extract utilization.

    Parameters
    ----------
    model : NeqSimProcessModel
        The process model.
    safety_factor : float
        Design safety factor (e.g. 1.2 = 20% margin). Default 1.2.
    generate_compressor_charts : bool
        Whether to generate compressor charts during auto-sizing. Default True.
    chart_template : str
        Compressor chart template to use. Default 'CENTRIFUGAL_STANDARD'.
    chart_num_speeds : int
        Number of speed curves for generated charts.
    skip_already_sized : set, optional
        Set of equipment names that have already been sized. These will be
        skipped (only utilization is extracted). Pass ``None`` to size everything.

    Returns
    -------
    AutoSizeResult
        Sizing data, utilization, and bottleneck identification.
    """
    # Ensure process is run first
    try:
        model.run()
    except Exception:
        pass

    equipment_sized: List[SizingInfo] = []
    total = 0

    for name, unit in model._units.items():
        try:
            java_class = str(unit.getClass().getSimpleName())
        except Exception:
            continue

        total += 1

        # Check if equipment supports autoSize
        if java_class not in _AUTOSIZEABLE_TYPES:
            continue

        # Skip equipment that has already been sized (unless force_resize)
        if skip_already_sized and name in skip_already_sized:
            # Still extract current sizing data without re-sizing
            sizing_data = _extract_sizing_data(unit, java_class)
            equipment_sized.append(SizingInfo(
                name=name, equipment_type=java_class,
                auto_sized=True, sizing_data=sizing_data,
                message="Already sized (skipped)",
            ))
            continue

        sizing = _auto_size_single(
            unit, name, java_class, safety_factor,
            generate_compressor_charts, chart_template, chart_num_speeds,
        )
        equipment_sized.append(sizing)

    # Re-run process after sizing to get updated conditions
    try:
        model.run()
    except Exception:
        pass

    # Extract utilization for all equipment
    utilization = _get_all_utilization(model)

    # Find bottleneck
    bottleneck_name = ""
    bottleneck_constraint = ""
    bottleneck_util = 0.0
    if utilization:
        worst = max(utilization, key=lambda u: u.utilization_pct)
        worst.is_bottleneck = True
        bottleneck_name = worst.name
        bottleneck_constraint = worst.constraint_name
        bottleneck_util = worst.utilization_pct

    sized_count = sum(1 for s in equipment_sized if s.auto_sized)

    return AutoSizeResult(
        equipment_sized=equipment_sized,
        utilization=utilization,
        bottleneck_name=bottleneck_name,
        bottleneck_constraint=bottleneck_constraint,
        bottleneck_utilization_pct=bottleneck_util,
        total_equipment=total,
        sized_count=sized_count,
        message=(
            f"Auto-sized {sized_count}/{total} equipment items "
            f"(safety factor: {safety_factor:.0%}). "
            f"Bottleneck: {bottleneck_name} ({bottleneck_constraint} "
            f"at {bottleneck_util:.1f}%)."
            if bottleneck_name else
            f"Auto-sized {sized_count}/{total} equipment items."
        ),
    )


def _auto_size_single(
    unit: Any,
    name: str,
    java_class: str,
    safety_factor: float,
    gen_charts: bool,
    chart_template: str,
    chart_num_speeds: int,
) -> SizingInfo:
    """Auto-size a single equipment unit."""
    sizing_data: Dict[str, Any] = {}
    auto_sized = False

    try:
        # Try autoSize with safety factor
        if hasattr(unit, "autoSize"):
            try:
                unit.autoSize(safety_factor)
                auto_sized = True
            except Exception:
                try:
                    unit.autoSize()
                    auto_sized = True
                except Exception:
                    pass

        # For compressors: optionally generate chart
        if java_class == "Compressor" and gen_charts:
            try:
                from neqsim import jneqsim
                CompressorChartGenerator = (
                    jneqsim.process.equipment.compressor.CompressorChartGenerator
                )
                generator = CompressorChartGenerator(unit)
                chart = generator.generateFromTemplate(chart_template, chart_num_speeds)
                unit.setCompressorChartType('interpolate and extrapolate')
                unit.setCompressorChart(chart)
                unit.getCompressorChart().setHeadUnit('kJ/kg')
                unit.setSolveSpeed(True)
                unit.setUsePolytropicCalc(True)

                # Set speed limits to match the chart curve range so the
                # solver stays within the generated performance map.
                try:
                    cc = unit.getCompressorChart()
                    unit.setMaximumSpeed(cc.getMaxSpeedCurve())
                    unit.setMinimumSpeed(cc.getMinSpeedCurve())
                except Exception:
                    pass

                unit.run()
                sizing_data["chart_generated"] = True
                sizing_data["chart_template"] = chart_template
                auto_sized = True
            except Exception:
                sizing_data["chart_generated"] = False

        # Extract sizing report
        sizing_data.update(_extract_sizing_data(unit, java_class))

    except Exception as e:
        return SizingInfo(
            name=name, equipment_type=java_class,
            auto_sized=False, message=f"Auto-size failed: {e}",
        )

    return SizingInfo(
        name=name, equipment_type=java_class,
        auto_sized=auto_sized, sizing_data=sizing_data,
        message="Sized" if auto_sized else "Not sizeable",
    )


def _extract_sizing_data(unit: Any, java_class: str) -> Dict[str, Any]:
    """Extract dimensional/sizing data from an equipment unit."""
    data: Dict[str, Any] = {}

    # Try JSON sizing report
    if hasattr(unit, "getSizingReportJson"):
        try:
            import json
            report = str(unit.getSizingReportJson())
            data["sizing_report"] = json.loads(report)
        except Exception:
            pass

    # Separator-specific
    if java_class in ("Separator", "TwoPhaseSeparator", "ThreePhaseSeparator",
                       "GasScrubber", "GasScrubberSimple"):
        for attr, key in [
            ("getInternalDiameter", "diameter_m"),
            ("getSeparatorLength", "length_m"),
            ("getGasLoadFactor", "gas_load_factor"),
            ("getDesignGasLoadFactor", "design_gas_load_factor"),
            ("getGasSuperficialVelocity", "gas_velocity_m_s"),
            ("getMaxAllowableGasVelocity", "max_gas_velocity_m_s"),
        ]:
            if hasattr(unit, attr):
                try:
                    data[key] = round(float(getattr(unit, attr)()), 4)
                except Exception:
                    pass

    # Compressor-specific
    if java_class == "Compressor":
        for attr, key in [
            ("getSpeed", "speed_rpm"),
            ("getPolytropicEfficiency", "polytropic_efficiency"),
            ("getPolytropicHead", "polytropic_head_kJ_kg"),
            ("getPower", "power_W"),
        ]:
            if hasattr(unit, attr):
                try:
                    val = float(getattr(unit, attr)())
                    if key == "polytropic_head_kJ_kg":
                        try:
                            val = float(unit.getPolytropicHead("kJ/kg"))
                        except Exception:
                            pass
                    data[key] = round(val, 2)
                except Exception:
                    pass

        # Design limits
        for attr, key in [
            ("getMaxSpeed", "max_speed_rpm"),
            ("getMinimumSpeed", "min_speed_rpm"),
            ("getMaxDesignPower", "max_design_power_W"),
        ]:
            if hasattr(unit, attr):
                try:
                    data[key] = round(float(getattr(unit, attr)()), 2)
                except Exception:
                    pass

        # Chart status
        if hasattr(unit, "getCompressorChart"):
            try:
                chart = unit.getCompressorChart()
                data["chart_active"] = bool(chart.isUseCompressorChart())
            except Exception:
                pass

    # Valve-specific
    if java_class in ("ThrottlingValve", "ControlValve"):
        for attr, key in [
            ("getCv", "cv"),
            ("getPercentValveOpening", "opening_pct"),
        ]:
            if hasattr(unit, attr):
                try:
                    data[key] = round(float(getattr(unit, attr)()), 2)
                except Exception:
                    pass

    # Heater/Cooler-specific
    if java_class in ("Heater", "Cooler", "AirCooler", "WaterCooler"):
        for attr, key in [
            ("getDuty", "duty_W"),
            ("getEnergyInput", "energy_input_W"),
        ]:
            if hasattr(unit, attr):
                try:
                    data[key] = round(float(getattr(unit, attr)()), 2)
                except Exception:
                    pass

    return data


# ---------------------------------------------------------------------------
# Utilization extraction
# ---------------------------------------------------------------------------

def _get_all_utilization(model: NeqSimProcessModel) -> List[UtilizationInfo]:
    """Extract utilization ratios for all equipment."""
    results = []

    for name, unit in model._units.items():
        try:
            java_class = str(unit.getClass().getSimpleName())
        except Exception:
            continue

        util = _get_utilization(unit, name, java_class)
        if util is not None:
            results.append(util)

    return results


def _get_utilization(unit: Any, name: str, java_class: str) -> Optional[UtilizationInfo]:
    """Extract utilization for a single equipment unit."""

    # Try generic NeqSim API first
    if hasattr(unit, "getMaxUtilization"):
        try:
            util = float(unit.getMaxUtilization())
            if util > 0:
                constraint = ""
                if hasattr(unit, "getBottleneckConstraintName"):
                    try:
                        constraint = str(unit.getBottleneckConstraintName())
                    except Exception:
                        pass
                return UtilizationInfo(
                    name=name, equipment_type=java_class,
                    utilization_pct=round(util * 100, 1),
                    constraint_name=constraint or "capacity",
                    detail=f"Utilization: {util*100:.1f}%",
                )
        except Exception:
            pass

    # Compressor checks
    if java_class == "Compressor":
        utils = []

        # Surge margin
        if hasattr(unit, "getDistanceToSurge"):
            try:
                dist = float(unit.getDistanceToSurge())
                # dist > 0 means flow is above surge. 
                # Utilization from surge perspective: 1/(1+dist)
                if dist >= 0:
                    surge_util = 1.0 / (1.0 + dist) if (1 + dist) > 0 else 1.0
                    utils.append(UtilizationInfo(
                        name=name, equipment_type=java_class,
                        utilization_pct=round(surge_util * 100, 1),
                        constraint_name="surge_margin",
                        detail=f"Surge margin: {dist*100:.1f}%",
                    ))
            except Exception:
                pass

        # Power limit
        if hasattr(unit, "getPower"):
            try:
                power = abs(float(unit.getPower()))
                max_power = None
                if hasattr(unit, "getMaxDesignPower"):
                    try:
                        max_power = float(unit.getMaxDesignPower())
                    except Exception:
                        pass
                if max_power and max_power > 0:
                    putil = power / max_power
                    utils.append(UtilizationInfo(
                        name=name, equipment_type=java_class,
                        utilization_pct=round(putil * 100, 1),
                        constraint_name="power",
                        constraint_value=round(power / 1000, 1),
                        design_value=round(max_power / 1000, 1),
                        unit="kW",
                        detail=f"Power: {power/1000:.0f}/{max_power/1000:.0f} kW",
                    ))
            except Exception:
                pass

        # Speed limit
        if hasattr(unit, "getSpeed") and hasattr(unit, "getMaxSpeed"):
            try:
                speed = float(unit.getSpeed())
                max_speed = float(unit.getMaxSpeed())
                if max_speed > 0 and speed > 0:
                    sutil = speed / max_speed
                    utils.append(UtilizationInfo(
                        name=name, equipment_type=java_class,
                        utilization_pct=round(sutil * 100, 1),
                        constraint_name="speed",
                        constraint_value=round(speed, 0),
                        design_value=round(max_speed, 0),
                        unit="RPM",
                        detail=f"Speed: {speed:.0f}/{max_speed:.0f} RPM",
                    ))
            except Exception:
                pass

        # Stonewall margin
        if hasattr(unit, "getDistanceToStoneWall"):
            try:
                dist = float(unit.getDistanceToStoneWall())
                if dist >= 0:
                    sw_util = 1.0 / (1.0 + dist) if (1 + dist) > 0 else 1.0
                    utils.append(UtilizationInfo(
                        name=name, equipment_type=java_class,
                        utilization_pct=round(sw_util * 100, 1),
                        constraint_name="stonewall_margin",
                        detail=f"Stonewall margin: {dist*100:.1f}%",
                    ))
            except Exception:
                pass

        if utils:
            worst = max(utils, key=lambda u: u.utilization_pct)
            return worst

    # Separator checks
    if java_class in ("Separator", "TwoPhaseSeparator", "ThreePhaseSeparator",
                       "GasScrubber", "GasScrubberSimple"):
        utils = []

        if hasattr(unit, "getGasLoadFactor") and hasattr(unit, "getDesignGasLoadFactor"):
            try:
                glf = float(unit.getGasLoadFactor())
                design_glf = float(unit.getDesignGasLoadFactor())
                if design_glf > 0:
                    util = glf / design_glf
                    utils.append(UtilizationInfo(
                        name=name, equipment_type=java_class,
                        utilization_pct=round(util * 100, 1),
                        constraint_name="gas_load_factor",
                        constraint_value=round(glf, 4),
                        design_value=round(design_glf, 4),
                        unit="m/s",
                        detail=f"K-factor: {glf:.4f}/{design_glf:.4f}",
                    ))
            except Exception:
                pass

        if hasattr(unit, "getGasSuperficialVelocity") and hasattr(unit, "getMaxAllowableGasVelocity"):
            try:
                vel = float(unit.getGasSuperficialVelocity())
                max_vel = float(unit.getMaxAllowableGasVelocity())
                if max_vel > 0:
                    vutil = vel / max_vel
                    utils.append(UtilizationInfo(
                        name=name, equipment_type=java_class,
                        utilization_pct=round(vutil * 100, 1),
                        constraint_name="gas_velocity",
                        constraint_value=round(vel, 2),
                        design_value=round(max_vel, 2),
                        unit="m/s",
                        detail=f"Gas velocity: {vel:.2f}/{max_vel:.2f} m/s",
                    ))
            except Exception:
                pass

        if utils:
            worst = max(utils, key=lambda u: u.utilization_pct)
            return worst

    # Valve checks
    if java_class in ("ThrottlingValve", "ControlValve"):
        if hasattr(unit, "getPercentValveOpening"):
            try:
                opening = float(unit.getPercentValveOpening())
                # Design limit is typically 90% opening
                design_opening = 90.0
                util = opening / design_opening
                return UtilizationInfo(
                    name=name, equipment_type=java_class,
                    utilization_pct=round(util * 100, 1),
                    constraint_name="valve_opening",
                    constraint_value=round(opening, 1),
                    design_value=design_opening,
                    unit="%",
                    detail=f"Opening: {opening:.1f}/{design_opening:.0f}%",
                )
            except Exception:
                pass

    # Heater/Cooler checks
    if java_class in ("Heater", "Cooler", "AirCooler", "WaterCooler"):
        if hasattr(unit, "getDuty"):
            try:
                duty = abs(float(unit.getDuty()))
                max_duty = None
                if hasattr(unit, "getMaxDesignDuty"):
                    try:
                        max_duty = float(unit.getMaxDesignDuty())
                    except Exception:
                        pass
                if max_duty and max_duty > 0:
                    dutil = duty / max_duty
                    return UtilizationInfo(
                        name=name, equipment_type=java_class,
                        utilization_pct=round(dutil * 100, 1),
                        constraint_name="duty",
                        constraint_value=round(duty / 1000, 1),
                        design_value=round(max_duty / 1000, 1),
                        unit="kW",
                        detail=f"Duty: {duty/1000:.0f}/{max_duty/1000:.0f} kW",
                    )
            except Exception:
                pass

    return None


# ---------------------------------------------------------------------------
# Format for LLM
# ---------------------------------------------------------------------------

def format_autosize_result(result: AutoSizeResult) -> str:
    """Format auto-size result as text for the LLM to interpret."""
    lines = ["=== AUTO-SIZE RESULT ==="]
    lines.append(f"  Equipment in process: {result.total_equipment}")
    lines.append(f"  Successfully sized: {result.sized_count}")
    lines.append("")

    if result.bottleneck_name:
        lines.append(f"  BOTTLENECK: {result.bottleneck_name}")
        lines.append(f"    Constraint: {result.bottleneck_constraint}")
        lines.append(f"    Utilization: {result.bottleneck_utilization_pct:.1f}%")
        lines.append("")

    # Sizing data
    if result.equipment_sized:
        lines.append("  SIZING RESULTS:")
        for si in result.equipment_sized:
            status = "✓ Sized" if si.auto_sized else "- Not sized"
            lines.append(f"    {status} {si.name} ({si.equipment_type})")
            for k, v in si.sizing_data.items():
                if k != "sizing_report":
                    lines.append(f"      {k}: {v}")

    # Utilization
    if result.utilization:
        lines.append("\n  EQUIPMENT UTILIZATION:")
        for ui in sorted(result.utilization, key=lambda u: u.utilization_pct, reverse=True):
            marker = "★" if ui.is_bottleneck else " "
            bar = "█" * min(int(ui.utilization_pct / 5), 20) + "░" * max(20 - int(ui.utilization_pct / 5), 0)
            lines.append(
                f"  {marker} {ui.name:30s} [{bar}] {ui.utilization_pct:5.1f}%  "
                f"({ui.constraint_name}: {ui.detail})"
            )

    lines.append(f"\n  {result.message}")
    return "\n".join(lines)
