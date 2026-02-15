"""
Compressor Chart — generate and extract compressor performance curves.

Uses NeqSim's ``CompressorChartGenerator`` and ``CompressorCurveTemplate``
to automatically create compressor performance maps based on operating
conditions and application type.

Features:
  - Generate multi-speed performance curves from 12 predefined templates
  - Extract chart data for Plotly/Streamlit plotting
  - Surge line and stone wall extraction
  - Operating point overlay
  - Speed-based utilization metrics
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .process_model import NeqSimProcessModel


# ---------------------------------------------------------------------------
# Available templates
# ---------------------------------------------------------------------------

CHART_TEMPLATES = {
    "CENTRIFUGAL_STANDARD": "General purpose centrifugal compressor (~78% eff)",
    "CENTRIFUGAL_HIGH_FLOW": "High throughput, low pressure ratio (~78% eff)",
    "CENTRIFUGAL_HIGH_HEAD": "High pressure ratio, narrow range (~78% eff)",
    "PIPELINE": "Gas transmission, flat curves, wide turndown (82-85% eff)",
    "EXPORT": "Offshore gas export, high pressure (~80% eff)",
    "INJECTION": "Gas injection/EOR, very high PR (~77% eff)",
    "GAS_LIFT": "Artificial lift, wide surge margin (~75% eff)",
    "REFRIGERATION": "LNG/process cooling, wide range (~78% eff)",
    "BOOSTER": "Process plant, moderate PR (~76% eff)",
    "SINGLE_STAGE": "Simple, wide flow range (~75% eff)",
    "MULTISTAGE_INLINE": "Barrel type, 4-8 stages (~78% eff)",
    "INTEGRALLY_GEARED": "Multiple pinions, highest efficiency (82% eff)",
    "OVERHUNG": "Cantilever, simple maintenance (~74% eff)",
}

DEFAULT_TEMPLATE = "CENTRIFUGAL_STANDARD"
DEFAULT_NUM_SPEEDS = 5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SpeedCurve:
    """Data for one speed curve."""
    speed_rpm: float
    flow_m3_hr: List[float]
    head_kJ_kg: List[float]
    efficiency_pct: List[float]


@dataclass
class OperatingPoint:
    """Current operating point on the chart."""
    flow_m3_hr: float
    head_kJ_kg: float
    efficiency_pct: float
    speed_rpm: float
    distance_to_surge: float  # ratio (>0 = above surge)
    distance_to_stonewall: float  # ratio (>0 = below stonewall)
    in_surge: bool
    at_stonewall: bool
    speed_in_range: bool
    power_kw: float


@dataclass
class CompressorChartData:
    """Complete chart data for plotting."""
    compressor_name: str
    template_used: str
    speed_curves: List[SpeedCurve]
    surge_flow: List[float]
    surge_head: List[float]
    stonewall_flow: List[float]
    stonewall_head: List[float]
    operating_point: Optional[OperatingPoint]
    head_unit: str = "kJ/kg"
    flow_unit: str = "m³/hr"
    min_speed: float = 0.0
    max_speed: float = 0.0
    message: str = ""


@dataclass
class CompressorChartResult:
    """Result of chart generation for multiple compressors."""
    charts: List[CompressorChartData]
    message: str = ""


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def generate_compressor_chart(
    model: NeqSimProcessModel,
    compressor_name: Optional[str] = None,
    template: str = DEFAULT_TEMPLATE,
    num_speeds: int = DEFAULT_NUM_SPEEDS,
) -> CompressorChartResult:
    """
    Generate compressor performance charts for one or all compressors.

    Parameters
    ----------
    model : NeqSimProcessModel
        The process model containing compressors.
    compressor_name : str, optional
        Name of specific compressor. If None, generates for all compressors.
    template : str
        Template name (e.g. 'PIPELINE', 'EXPORT', 'CENTRIFUGAL_STANDARD').
    num_speeds : int
        Number of speed curves to generate (default 5).

    Returns
    -------
    CompressorChartResult
        Chart data for plotting.
    """
    template = template.upper().replace(" ", "_").replace("-", "_")
    if template not in CHART_TEMPLATES:
        template = DEFAULT_TEMPLATE

    charts: List[CompressorChartData] = []

    # Find compressor(s)
    compressors = _find_compressors(model, compressor_name)
    if not compressors:
        return CompressorChartResult(
            charts=[],
            message=f"No compressors found"
            + (f" matching '{compressor_name}'" if compressor_name else "")
            + ".",
        )

    for name, unit in compressors:
        try:
            chart_data = _generate_single_chart(unit, name, template, num_speeds)
            charts.append(chart_data)
        except Exception as e:
            charts.append(CompressorChartData(
                compressor_name=name,
                template_used=template,
                speed_curves=[],
                surge_flow=[], surge_head=[],
                stonewall_flow=[], stonewall_head=[],
                operating_point=None,
                message=f"Chart generation failed: {e}",
            ))

    return CompressorChartResult(
        charts=charts,
        message=f"Generated {len(charts)} compressor chart(s) using {template} template.",
    )


def _find_compressors(
    model: NeqSimProcessModel,
    name: Optional[str] = None,
) -> List[Tuple[str, Any]]:
    """Find compressor units in the model."""
    results = []
    for uname, unit in model._units.items():
        try:
            java_class = str(unit.getClass().getSimpleName())
        except Exception:
            continue
        if java_class == "Compressor":
            if name is None or uname.lower() == name.lower():
                results.append((uname, unit))
    return results


def _generate_single_chart(
    unit: Any,
    name: str,
    template: str,
    num_speeds: int,
) -> CompressorChartData:
    """Generate chart for a single compressor unit."""
    from neqsim import jneqsim

    CompressorChartGenerator = jneqsim.process.equipment.compressor.CompressorChartGenerator

    # Ensure compressor has been run
    try:
        unit.run()
    except Exception:
        pass

    # Generate chart using NeqSim's CompressorChartGenerator
    generator = CompressorChartGenerator(unit)
    chart = generator.generateFromTemplate(template, num_speeds)

    # Apply chart to compressor with interpolation mode
    unit.setCompressorChartType('interpolate and extrapolate')
    unit.setCompressorChart(chart)
    unit.getCompressorChart().setHeadUnit('kJ/kg')

    # Enable chart-based speed solving: the chart will determine the
    # required speed to reach the target outlet pressure.
    unit.setSolveSpeed(True)
    unit.setUsePolytropicCalc(True)
    unit.run()

    # Extract speed curves using the generator (gives access to real curve data)
    speed_curves = _extract_speed_curves(unit, generator)

    # Extract surge and stonewall curves from the chart
    surge_flow, surge_head = _extract_surge_curve(unit)
    sw_flow, sw_head = _extract_stonewall_curve(unit)

    # Extract operating point
    op = _extract_operating_point(unit)

    # Speed range
    min_speed = 0.0
    max_speed = 0.0
    if speed_curves:
        min_speed = min(sc.speed_rpm for sc in speed_curves)
        max_speed = max(sc.speed_rpm for sc in speed_curves)

    return CompressorChartData(
        compressor_name=name,
        template_used=template,
        speed_curves=speed_curves,
        surge_flow=surge_flow,
        surge_head=surge_head,
        stonewall_flow=sw_flow,
        stonewall_head=sw_head,
        operating_point=op,
        min_speed=min_speed,
        max_speed=max_speed,
        message=f"Chart generated: {len(speed_curves)} speed curves, template={template}",
    )


def _extract_speed_curves(unit: Any, generator: Any = None) -> List[SpeedCurve]:
    """Extract speed curve data points from a compressor and its chart generator."""
    curves = []

    try:
        chart = unit.getCompressorChart()
        if chart is None:
            return curves

        # Method 1: try to get curves from the chart's getCurves()/getRealCurves()
        real_curves = None
        try:
            real_curves = chart.getCurves()
        except Exception:
            pass

        if real_curves is not None:
            try:
                for i, curve in enumerate(real_curves):
                    try:
                        spd = float(curve.speed) if hasattr(curve, 'speed') else float(curve.getSpeed())
                        flow_arr = list(curve.flow) if hasattr(curve, 'flow') else list(curve.getFlow())
                        head_arr = list(curve.head) if hasattr(curve, 'head') else list(curve.getHead())
                        eff_arr = []
                        try:
                            eff_arr = list(curve.polytropicEfficiency) if hasattr(curve, 'polytropicEfficiency') else list(curve.getPolytropicEfficiency())
                        except Exception:
                            eff_arr = [0.0] * len(flow_arr)

                        flow_list = [round(float(f), 1) for f in flow_arr]
                        head_list = [round(float(h), 2) for h in head_arr]
                        eff_list = [round(float(e), 1) if e > 1 else round(float(e) * 100, 1) for e in eff_arr]

                        if flow_list and head_list:
                            curves.append(SpeedCurve(
                                speed_rpm=round(spd, 0),
                                flow_m3_hr=flow_list,
                                head_kJ_kg=head_list,
                                efficiency_pct=eff_list,
                            ))
                    except Exception:
                        continue
            except Exception:
                pass

        # Method 2: try sampling at different speeds via chart methods
        if not curves:
            try:
                speeds = []
                try:
                    # Try to get speed range from chart
                    speed_arr = chart.getSpeed()
                    if speed_arr is not None:
                        speeds = [float(s) for s in speed_arr]
                except Exception:
                    pass

                if not speeds:
                    # Estimate from compressor's current speed
                    try:
                        cur_speed = float(unit.getSpeed())
                        if cur_speed > 0:
                            for frac in [0.7, 0.8, 0.9, 1.0, 1.1]:
                                speeds.append(cur_speed * frac)
                    except Exception:
                        pass

                for speed in speeds:
                    flows = []
                    heads = []
                    effs = []
                    try:
                        n_points = 10
                        cur_flow = float(unit.getInletStream().getFlowRate("m3/hr"))
                        if cur_flow <= 0:
                            continue
                        for j in range(n_points):
                            f = cur_flow * (0.5 + 1.0 * j / max(n_points - 1, 1))
                            try:
                                h = float(chart.getPolytropicHead(f, speed))
                                e = float(chart.getPolytropicEfficiency(f, speed))
                                if h > 0 and 0 < e <= 1.0:
                                    flows.append(round(f, 1))
                                    heads.append(round(h, 2))
                                    effs.append(round(e * 100, 1))
                            except Exception:
                                continue
                    except Exception:
                        continue

                    if flows:
                        curves.append(SpeedCurve(
                            speed_rpm=round(speed, 0),
                            flow_m3_hr=flows,
                            head_kJ_kg=heads,
                            efficiency_pct=effs,
                        ))
            except Exception:
                pass

    except Exception:
        pass

    return curves


def _extract_surge_curve(unit: Any) -> Tuple[List[float], List[float]]:
    """Extract surge curve data from compressor chart."""
    flows, heads = [], []
    try:
        chart = unit.getCompressorChart()
        if chart is None:
            return flows, heads

        surge = chart.getSurgeCurve()
        if surge is None:
            return flows, heads

        # Try getFlow() / getHead() — returns Java arrays (standard API)
        try:
            surge_flow_arr = surge.getFlow()
            surge_head_arr = surge.getHead()
            if surge_flow_arr is not None and surge_head_arr is not None:
                flows = [round(float(f), 1) for f in surge_flow_arr]
                heads = [round(float(h), 2) for h in surge_head_arr]
                if flows and heads:
                    return flows, heads
        except Exception:
            pass

        # Fallback: try to read surge from setCurve data
        try:
            surge_flow_arr = surge.getSurgeFlow()
            surge_head_arr = surge.getSurgeHead()
            if surge_flow_arr is not None and surge_head_arr is not None:
                flows = [round(float(f), 1) for f in surge_flow_arr]
                heads = [round(float(h), 2) for h in surge_head_arr]
        except Exception:
            pass

    except Exception:
        pass

    return flows, heads


def _extract_stonewall_curve(unit: Any) -> Tuple[List[float], List[float]]:
    """Extract stonewall (choke) curve from compressor chart."""
    flows, heads = [], []
    try:
        chart = unit.getCompressorChart()
        if chart is None:
            return flows, heads

        sw = None
        try:
            sw = chart.getStoneWallCurve()
        except Exception:
            return flows, heads

        if sw is None:
            return flows, heads

        # Try getFlow() / getHead() — returns Java arrays (standard API)
        try:
            sw_flow_arr = sw.getFlow()
            sw_head_arr = sw.getHead()
            if sw_flow_arr is not None and sw_head_arr is not None:
                flows = [round(float(f), 1) for f in sw_flow_arr]
                heads = [round(float(h), 2) for h in sw_head_arr]
                if flows and heads:
                    return flows, heads
        except Exception:
            pass

        # Fallback: try alternative access
        try:
            sw_flow_arr = sw.getStoneWallFlow()
            sw_head_arr = sw.getStoneWallHead()
            if sw_flow_arr is not None and sw_head_arr is not None:
                flows = [round(float(f), 1) for f in sw_flow_arr]
                heads = [round(float(h), 2) for h in sw_head_arr]
        except Exception:
            pass

    except Exception:
        pass

    return flows, heads


def _extract_operating_point(unit: Any) -> Optional[OperatingPoint]:
    """Extract the current operating point from a compressor."""
    try:
        flow = 0.0
        try:
            inlet = unit.getInletStream() if hasattr(unit, "getInletStream") else unit.getInStream()
            flow = float(inlet.getFlowRate("m3/hr"))
        except Exception:
            pass

        head = 0.0
        try:
            head = float(unit.getPolytropicHead())
        except Exception:
            try:
                head = float(unit.getPolytropicFluidHead())
            except Exception:
                pass

        eff = 0.0
        try:
            eff = float(unit.getPolytropicEfficiency()) * 100
        except Exception:
            pass

        speed = 0.0
        try:
            speed = float(unit.getSpeed())
        except Exception:
            pass

        dist_surge = 0.0
        try:
            dist_surge = float(unit.getDistanceToSurge())
        except Exception:
            pass

        dist_sw = 0.0
        try:
            dist_sw = float(unit.getDistanceToStoneWall())
        except Exception:
            pass

        in_surge = False
        try:
            in_surge = bool(unit.isSurge())
        except Exception:
            pass

        at_sw = False
        try:
            at_sw = bool(unit.isStoneWall())
        except Exception:
            pass

        speed_ok = True
        try:
            speed_ok = bool(unit.isSpeedWithinRange())
        except Exception:
            pass

        power = 0.0
        try:
            power = abs(float(unit.getPower("MW"))) * 1000  # MW -> kW
        except Exception:
            try:
                power = abs(float(unit.getPower())) / 1000
            except Exception:
                pass

        return OperatingPoint(
            flow_m3_hr=round(flow, 1),
            head_kJ_kg=round(head, 2),
            efficiency_pct=round(eff, 1),
            speed_rpm=round(speed, 0),
            distance_to_surge=round(dist_surge, 4),
            distance_to_stonewall=round(dist_sw, 4),
            in_surge=in_surge,
            at_stonewall=at_sw,
            speed_in_range=speed_ok,
            power_kw=round(power, 1),
        )

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Format for LLM
# ---------------------------------------------------------------------------

def format_chart_result(result: CompressorChartResult) -> str:
    """Format chart result as text for the LLM to interpret."""
    lines = ["=== COMPRESSOR CHART GENERATION RESULT ==="]

    for cd in result.charts:
        lines.append(f"\n--- {cd.compressor_name} ---")
        lines.append(f"  Template: {cd.template_used}")
        lines.append(f"  Speed curves: {len(cd.speed_curves)}")
        if cd.min_speed > 0:
            lines.append(f"  Speed range: {cd.min_speed:.0f} – {cd.max_speed:.0f} RPM")

        if cd.operating_point:
            op = cd.operating_point
            lines.append("  Operating Point:")
            lines.append(f"    Flow: {op.flow_m3_hr:.1f} m³/hr")
            lines.append(f"    Head: {op.head_kJ_kg:.2f} kJ/kg")
            lines.append(f"    Efficiency: {op.efficiency_pct:.1f}%")
            lines.append(f"    Speed: {op.speed_rpm:.0f} RPM")
            lines.append(f"    Power: {op.power_kw:.1f} kW")
            lines.append(f"    Distance to surge: {op.distance_to_surge*100:.1f}%")
            lines.append(f"    Distance to stonewall: {op.distance_to_stonewall*100:.1f}%")
            if op.in_surge:
                lines.append("    ⚠️ OPERATING IN SURGE!")
            if op.at_stonewall:
                lines.append("    ⚠️ AT STONE WALL / CHOKE!")
            if not op.speed_in_range:
                lines.append("    ⚠️ Speed outside curve range!")

        for sc in cd.speed_curves:
            lines.append(f"  Speed {sc.speed_rpm:.0f} RPM: "
                         f"flow [{sc.flow_m3_hr[0]:.0f}–{sc.flow_m3_hr[-1]:.0f}] m³/hr, "
                         f"head [{sc.head_kJ_kg[0]:.1f}–{sc.head_kJ_kg[-1]:.1f}] kJ/kg")

        if cd.surge_flow:
            lines.append(f"  Surge line: {len(cd.surge_flow)} points")
        if cd.stonewall_flow:
            lines.append(f"  Stone wall: {len(cd.stonewall_flow)} points")

        if cd.message:
            lines.append(f"  Note: {cd.message}")

    lines.append(f"\n{result.message}")
    return "\n".join(lines)
