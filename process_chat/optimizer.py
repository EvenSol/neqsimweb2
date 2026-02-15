"""
Process Optimizer — find maximum production / throughput for a NeqSim process.

Two approaches:
  1. **Java-native**: Uses NeqSim's ``ProcessOptimizationEngine`` if available
     (``findMaximumThroughput()``, equipment capacity strategy plugins).
  2. **Python fallback**: Golden-section search on the feed-stream flow rate,
     checking equipment utilization at each step until the bottleneck is found.

Both return an ``OptimizationResult`` dataclass with optimal flow, bottleneck
equipment, utilization breakdown, and the iteration history.

Optional pre-optimization setup:
  - ``auto_size_equipment()`` — sizes separators, valves, etc. so that design
    limits are available for utilization checks.
  - ``generate_compressor_charts()`` — creates compressor performance curves
    so surge / stonewall / speed limits are enforced during the search.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .process_model import NeqSimProcessModel, ModelRunResult, KPI


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class EquipmentUtilization:
    """Utilization snapshot for a single piece of equipment."""
    name: str
    equipment_type: str
    utilization: float          # 0.0–1.0+ (>1.0 = constraint violation)
    constraint_name: str = ""   # which constraint is limiting
    detail: str = ""            # human-readable detail


@dataclass
class OptimizationIteration:
    """Record of one search iteration."""
    iteration: int
    flow_rate_kg_hr: float
    feasible: bool
    max_utilization: float
    bottleneck: str
    detail: str = ""


@dataclass
class OptimizationResult:
    """Final result of a process optimization run."""
    optimal_flow_kg_hr: float
    original_flow_kg_hr: float
    max_increase_pct: float
    bottleneck_equipment: str
    bottleneck_type: str
    bottleneck_utilization: float
    utilization_breakdown: List[EquipmentUtilization]
    iterations: List[OptimizationIteration]
    kpis_at_optimum: Dict[str, KPI] = field(default_factory=dict)
    search_algorithm: str = "golden_section"
    converged: bool = True
    message: str = ""


# ---------------------------------------------------------------------------
# Equipment utilization extraction
# ---------------------------------------------------------------------------

# Java simple-class names that are compressors
_COMPRESSOR_TYPES = {"Compressor"}
_SEPARATOR_TYPES = {"Separator", "TwoPhaseSeparator", "ThreePhaseSeparator",
                    "GasScrubber", "GasScrubberSimple"}
_PUMP_TYPES = {"Pump", "ESPPump"}
_EXPANDER_TYPES = {"Expander"}
_VALVE_TYPES = {"ThrottlingValve", "ControlValve", "CheckValve"}

# Maximum utilization threshold — equipment above this is considered infeasible
MAX_UTILIZATION_THRESHOLD = 1.0


def _get_equipment_utilization(unit, name: str, java_class: str) -> Optional[EquipmentUtilization]:
    """
    Extract the utilization ratio for a single equipment unit.

    Tries, in order:
      1. ``getMaxUtilization()`` (new NeqSim API — returns 0–1+ ratio)
      2. Equipment-type-specific heuristics (surge margin, gas load factor, etc.)
      3. Returns None if no utilization can be determined.
    """
    # --- Try the generic NeqSim API first ---
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
                return EquipmentUtilization(
                    name=name,
                    equipment_type=java_class,
                    utilization=util,
                    constraint_name=constraint or "max_utilization",
                    detail=f"Utilization: {util*100:.1f}%",
                )
        except Exception:
            pass

    # --- Compressor-specific checks ---
    if java_class in _COMPRESSOR_TYPES:
        utils = []

        # Surge margin check
        if hasattr(unit, "getDistanceToSurge"):
            try:
                surge_dist = float(unit.getDistanceToSurge())
                # distanceToSurge > 0 means above surge, < 0 means below
                # Utilization from surge: as flow decreases toward surge, utilization → 1
                if hasattr(unit, "getSurgeFlowRate"):
                    surge_flow = float(unit.getSurgeFlowRate())
                    if surge_flow > 0:
                        actual_flow = None
                        for m in ("getInletStream", "getInStream"):
                            if hasattr(unit, m):
                                try:
                                    s = getattr(unit, m)()
                                    actual_flow = float(s.getFlowRate("kg/hr"))
                                    break
                                except Exception:
                                    pass
                        if actual_flow and actual_flow > 0:
                            # If flow == surge_flow → utilization = 1.0
                            surge_util = surge_flow / actual_flow
                            utils.append(("surge_margin", surge_util,
                                         f"Surge margin: flow={actual_flow:.0f}, surge={surge_flow:.0f} kg/hr"))
            except Exception:
                pass

        # Power limit check
        if hasattr(unit, "getPower"):
            try:
                power = abs(float(unit.getPower()))  # W
                max_power = None
                if hasattr(unit, "getMaxDesignPower"):
                    try:
                        max_power = float(unit.getMaxDesignPower())
                    except Exception:
                        pass
                if max_power and max_power > 0:
                    power_util = power / max_power
                    utils.append(("power_limit", power_util,
                                 f"Power: {power/1000:.0f}/{max_power/1000:.0f} kW ({power_util*100:.1f}%)"))
            except Exception:
                pass

        # Speed check
        if hasattr(unit, "getSpeed") and hasattr(unit, "getMaxSpeed"):
            try:
                speed = float(unit.getSpeed())
                max_speed = float(unit.getMaxSpeed())
                if max_speed > 0 and speed > 0:
                    speed_util = speed / max_speed
                    utils.append(("speed_limit", speed_util,
                                 f"Speed: {speed:.0f}/{max_speed:.0f} rpm ({speed_util*100:.1f}%)"))
            except Exception:
                pass

        # Polytropic head check — use as general capacity metric
        if hasattr(unit, "getPolytropicHead"):
            try:
                head = float(unit.getPolytropicHead())
                if hasattr(unit, "getMaxDesignHead"):
                    try:
                        max_head = float(unit.getMaxDesignHead())
                        if max_head > 0:
                            head_util = head / max_head
                            utils.append(("head_limit", head_util,
                                         f"Head: {head:.0f}/{max_head:.0f} kJ/kg ({head_util*100:.1f}%)"))
                    except Exception:
                        pass
            except Exception:
                pass

        if utils:
            # Take the worst (highest utilization) constraint
            worst = max(utils, key=lambda x: x[1])
            return EquipmentUtilization(
                name=name, equipment_type=java_class,
                utilization=worst[1], constraint_name=worst[0], detail=worst[2],
            )

    # --- Separator-specific checks ---
    if java_class in _SEPARATOR_TYPES:
        utils = []

        # Gas load factor
        if hasattr(unit, "getGasLoadFactor"):
            try:
                glf = float(unit.getGasLoadFactor())
                design_glf = None
                if hasattr(unit, "getDesignGasLoadFactor"):
                    try:
                        design_glf = float(unit.getDesignGasLoadFactor())
                    except Exception:
                        pass
                if design_glf and design_glf > 0:
                    util = glf / design_glf
                    utils.append(("gas_load_factor", util,
                                 f"Gas load factor: {glf:.4f}/{design_glf:.4f} ({util*100:.1f}%)"))
            except Exception:
                pass

        # Gas velocity check
        if hasattr(unit, "getGasSuperficialVelocity") and hasattr(unit, "getMaxAllowableGasVelocity"):
            try:
                vel = float(unit.getGasSuperficialVelocity())
                max_vel = float(unit.getMaxAllowableGasVelocity())
                if max_vel > 0:
                    vel_util = vel / max_vel
                    utils.append(("gas_velocity", vel_util,
                                 f"Gas velocity: {vel:.2f}/{max_vel:.2f} m/s ({vel_util*100:.1f}%)"))
            except Exception:
                pass

        if utils:
            worst = max(utils, key=lambda x: x[1])
            return EquipmentUtilization(
                name=name, equipment_type=java_class,
                utilization=worst[1], constraint_name=worst[0], detail=worst[2],
            )

    # --- Pump-specific checks ---
    if java_class in _PUMP_TYPES:
        if hasattr(unit, "getPower"):
            try:
                power = abs(float(unit.getPower()))
                if hasattr(unit, "getMaxDesignPower"):
                    max_power = float(unit.getMaxDesignPower())
                    if max_power > 0:
                        util = power / max_power
                        return EquipmentUtilization(
                            name=name, equipment_type=java_class,
                            utilization=util, constraint_name="power_limit",
                            detail=f"Power: {power/1000:.0f}/{max_power/1000:.0f} kW",
                        )
            except Exception:
                pass

    return None


def get_all_utilizations(model: NeqSimProcessModel) -> List[EquipmentUtilization]:
    """
    Extract utilization ratios for all equipment in the process.

    Equipment without utilization data is omitted.
    """
    results = []
    for name, u in model._units.items():
        try:
            java_class = str(u.getClass().getSimpleName())
        except Exception:
            continue
        util = _get_equipment_utilization(u, name, java_class)
        if util is not None:
            results.append(util)
    return results


# ---------------------------------------------------------------------------
# Feasibility check — is the process running within equipment limits?
# ---------------------------------------------------------------------------

def check_feasibility(
    model: NeqSimProcessModel,
    max_utilization: float = MAX_UTILIZATION_THRESHOLD,
) -> Tuple[bool, float, str, List[EquipmentUtilization]]:
    """
    Check whether a solved process is feasible (all equipment within limits).

    Returns:
        (feasible, max_util, bottleneck_name, utilization_list)
    """
    utils = get_all_utilizations(model)
    if not utils:
        # No utilization data available — assume feasible
        return True, 0.0, "", utils

    worst = max(utils, key=lambda u: u.utilization)
    feasible = worst.utilization <= max_utilization
    return feasible, worst.utilization, worst.name, utils


# ---------------------------------------------------------------------------
# Pre-optimization setup — auto-size + compressor charts
# ---------------------------------------------------------------------------

def prepare_model_for_optimization(
    model: NeqSimProcessModel,
    safety_factor: float = 1.2,
    auto_size: bool = True,
    generate_charts: bool = True,
    chart_template: str = "CENTRIFUGAL_STANDARD",
    chart_num_speeds: int = 5,
) -> Dict[str, Any]:
    """
    Prepare a process model for optimization by auto-sizing equipment and
    generating compressor charts.

    This ensures that design limits are set on equipment so that the optimizer
    can compute meaningful utilization ratios.

    Parameters
    ----------
    model : NeqSimProcessModel
        The process model.
    safety_factor : float
        Design safety factor for auto-sizing (default 1.2).
    auto_size : bool
        Whether to auto-size equipment (default True).
    generate_charts : bool
        Whether to generate compressor charts (default True).
    chart_template : str
        Compressor chart template name.
    chart_num_speeds : int
        Number of speed curves for charts.

    Returns
    -------
    dict
        Summary of preparation steps performed.
    """
    summary: Dict[str, Any] = {"auto_sized": [], "charts_generated": []}

    if auto_size:
        from .auto_size import _AUTOSIZEABLE_TYPES
        for name, unit in model._units.items():
            try:
                java_class = str(unit.getClass().getSimpleName())
            except Exception:
                continue
            if java_class not in _AUTOSIZEABLE_TYPES:
                continue
            try:
                if hasattr(unit, "autoSize"):
                    try:
                        unit.autoSize(safety_factor)
                    except Exception:
                        try:
                            unit.autoSize()
                        except Exception:
                            continue
                    summary["auto_sized"].append(name)
            except Exception:
                pass

    if generate_charts:
        for name, unit in model._units.items():
            try:
                java_class = str(unit.getClass().getSimpleName())
            except Exception:
                continue
            if java_class != "Compressor":
                continue
            try:
                # Check if chart already active
                has_chart = False
                if hasattr(unit, "getCompressorChart"):
                    try:
                        chart = unit.getCompressorChart()
                        has_chart = bool(chart.isUseCompressorChart())
                    except Exception:
                        pass
                if not has_chart:
                    from neqsim import jneqsim
                    gen = jneqsim.process.equipment.compressor.CompressorChartGenerator(unit)
                    chart = gen.generateFromTemplate(chart_template, chart_num_speeds)
                    unit.setCompressorChartType('interpolate and extrapolate')
                    unit.setCompressorChart(chart)
                    unit.getCompressorChart().setHeadUnit('kJ/kg')
                    unit.setSolveSpeed(True)
                    unit.setUsePolytropicCalc(True)
                    summary["charts_generated"].append(name)
            except Exception:
                pass

    # Re-run to apply sizing
    try:
        model.run()
    except Exception:
        pass

    return summary


# ---------------------------------------------------------------------------
# Java-native optimization (ProcessOptimizationEngine)
# ---------------------------------------------------------------------------

def _try_java_optimizer(
    model: NeqSimProcessModel,
    feed_stream_name: str,
    min_flow: float,
    max_flow: float,
    inlet_pressure: Optional[float] = None,
    outlet_pressure: Optional[float] = None,
) -> Optional[OptimizationResult]:
    """
    Attempt to use NeqSim's Java ``ProcessOptimizationEngine``.
    Returns None if the class is not available.
    """
    try:
        from neqsim import jneqsim

        ProcessOptimizationEngine = jneqsim.process.util.optimizer.ProcessOptimizationEngine
        SearchAlgorithm = ProcessOptimizationEngine.SearchAlgorithm

        proc = model.get_process()
        engine = ProcessOptimizationEngine(proc)
        engine.setFeedStreamName(feed_stream_name)
        engine.setSearchAlgorithm(SearchAlgorithm.GOLDEN_SECTION)

        # Determine pressures if not specified
        if inlet_pressure is None or outlet_pressure is None:
            feed = model.get_stream(feed_stream_name)
            if inlet_pressure is None:
                inlet_pressure = float(feed.getPressure("bara"))
            if outlet_pressure is None:
                # Use last stream's pressure
                streams = model.list_streams()
                if streams:
                    outlet_pressure = streams[-1].pressure_bara or inlet_pressure

        result = engine.findMaximumThroughput(
            inlet_pressure, outlet_pressure, min_flow, max_flow
        )

        optimal_flow = float(result.getOptimalValue())
        bottleneck = str(result.getBottleneck()) if result.getBottleneck() else "unknown"

        # Get original flow
        feed = model.get_stream(feed_stream_name)
        original_flow = float(feed.getFlowRate("kg/hr"))
        increase_pct = ((optimal_flow - original_flow) / original_flow * 100
                        if original_flow > 0 else 0)

        return OptimizationResult(
            optimal_flow_kg_hr=optimal_flow,
            original_flow_kg_hr=original_flow,
            max_increase_pct=increase_pct,
            bottleneck_equipment=bottleneck,
            bottleneck_type="",
            bottleneck_utilization=1.0,
            utilization_breakdown=[],
            iterations=[],
            search_algorithm="ProcessOptimizationEngine (Java)",
            converged=True,
            message=f"Maximum throughput: {optimal_flow:.0f} kg/hr (limited by {bottleneck})",
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Python fallback — golden-section search on feed flow rate
# ---------------------------------------------------------------------------

_PHI = (1 + math.sqrt(5)) / 2        # golden ratio ≈ 1.618
_RESPHI = 2 - _PHI                     # ≈ 0.382


def _evaluate_at_flow(
    model: NeqSimProcessModel,
    feed_stream_name: str,
    target_flow: float,
    timeout_ms: int = 120000,
    utilization_limit: float = MAX_UTILIZATION_THRESHOLD,
) -> Tuple[bool, float, str, List[EquipmentUtilization], ModelRunResult]:
    """
    Clone model, set feed flow, run, and check feasibility.
    Returns (feasible, max_utilization, bottleneck, utils, run_result).
    """
    clone = model.clone()
    # Set the feed stream flow rate
    stream = clone.get_stream(feed_stream_name)
    stream.setFlowRate(target_flow, "kg/hr")

    # Run the process
    result = clone.run(timeout_ms=timeout_ms)

    # Check equipment feasibility
    feasible, max_util, bottleneck, utils = check_feasibility(clone, utilization_limit)

    return feasible, max_util, bottleneck, utils, result


def _find_feed_stream(model: NeqSimProcessModel) -> Optional[str]:
    """Auto-detect the feed stream name (first stream in the process)."""
    proc = model.get_process()
    try:
        units = list(proc.getUnitOperations())
        if units:
            first = units[0]
            name = str(first.getName())
            java_class = str(first.getClass().getSimpleName())
            if "Stream" in java_class:
                return name
            # If first unit isn't a stream, look for its inlet
            for m in ("getInletStream", "getInStream", "getFeed"):
                if hasattr(first, m):
                    try:
                        s = getattr(first, m)()
                        if s is not None:
                            sname = str(s.getName())
                            return sname
                    except Exception:
                        pass
    except Exception:
        pass

    # Fall back to first stream in model's stream dict
    streams = model.list_streams()
    if streams:
        return streams[0].name
    return None


def optimize_production(
    model: NeqSimProcessModel,
    feed_stream_name: Optional[str] = None,
    min_flow_kg_hr: Optional[float] = None,
    max_flow_kg_hr: Optional[float] = None,
    utilization_limit: float = MAX_UTILIZATION_THRESHOLD,
    tolerance_pct: float = 1.0,
    max_iterations: int = 25,
    timeout_ms: int = 120000,
    try_java_engine: bool = True,
    auto_size_first: bool = True,
    safety_factor: float = 1.2,
    generate_compressor_charts: bool = True,
) -> OptimizationResult:
    """
    Find the maximum production rate for a process by scaling the feed flow.

    Uses golden-section search (or the Java ProcessOptimizationEngine if
    available) to find the highest feed flow rate where all equipment
    utilization stays below ``utilization_limit``.

    If no equipment utilization data is available (no design limits set),
    the optimizer works in "maximum convergence" mode — it finds the highest
    flow rate at which the process simulation still converges successfully.

    Parameters
    ----------
    model : NeqSimProcessModel
        The process model to optimize.
    feed_stream_name : str, optional
        Name of the feed stream to manipulate. Auto-detected if None.
    min_flow_kg_hr : float, optional
        Lower bound for flow rate search. Defaults to 10% of current flow.
    max_flow_kg_hr : float, optional
        Upper bound for flow rate search. Defaults to 500% of current flow.
    utilization_limit : float
        Maximum allowed equipment utilization (default 1.0 = 100%).
    tolerance_pct : float
        Convergence tolerance as % of flow range (default 1%).
    max_iterations : int
        Maximum search iterations (default 25).
    timeout_ms : int
        Timeout per simulation run in ms.
    try_java_engine : bool
        Whether to try the Java ProcessOptimizationEngine first.
    auto_size_first : bool
        Whether to auto-size equipment before optimizing (default True).
        This ensures design limits are available for utilization checks.
    safety_factor : float
        Safety factor for auto-sizing (default 1.2 = 20% margin).
    generate_compressor_charts : bool
        Whether to generate compressor charts before optimizing (default True).

    Returns
    -------
    OptimizationResult
        Detailed optimization result with optimal flow, bottleneck, and history.
    """
    # --- Pre-optimization setup (auto-size + compressor charts) ---
    if auto_size_first:
        prepare_model_for_optimization(
            model,
            safety_factor=safety_factor,
            auto_size=True,
            generate_charts=generate_compressor_charts,
        )

    # --- Auto-detect feed stream ---
    if feed_stream_name is None:
        feed_stream_name = _find_feed_stream(model)
        if feed_stream_name is None:
            return OptimizationResult(
                optimal_flow_kg_hr=0, original_flow_kg_hr=0,
                max_increase_pct=0, bottleneck_equipment="",
                bottleneck_type="", bottleneck_utilization=0,
                utilization_breakdown=[], iterations=[],
                converged=False,
                message="Could not auto-detect feed stream. Please specify feed_stream_name.",
            )

    # --- Get current (original) flow rate ---
    try:
        feed = model.get_stream(feed_stream_name)
        original_flow = float(feed.getFlowRate("kg/hr"))
    except Exception as e:
        return OptimizationResult(
            optimal_flow_kg_hr=0, original_flow_kg_hr=0,
            max_increase_pct=0, bottleneck_equipment="",
            bottleneck_type="", bottleneck_utilization=0,
            utilization_breakdown=[], iterations=[],
            converged=False,
            message=f"Could not read feed stream '{feed_stream_name}': {e}",
        )

    if original_flow <= 0:
        return OptimizationResult(
            optimal_flow_kg_hr=0, original_flow_kg_hr=0,
            max_increase_pct=0, bottleneck_equipment="",
            bottleneck_type="", bottleneck_utilization=0,
            utilization_breakdown=[], iterations=[],
            converged=False,
            message=f"Feed flow rate is zero or negative ({original_flow}).",
        )

    # --- Set default bounds ---
    if min_flow_kg_hr is None:
        min_flow_kg_hr = original_flow * 0.1
    if max_flow_kg_hr is None:
        max_flow_kg_hr = original_flow * 5.0

    # --- Try Java optimizer first ---
    if try_java_engine:
        java_result = _try_java_optimizer(
            model, feed_stream_name, min_flow_kg_hr, max_flow_kg_hr
        )
        if java_result is not None:
            return java_result

    # --- Python golden-section search ---
    iterations: List[OptimizationIteration] = []

    # First, check if current flow is feasible
    has_utilization_data = False

    try:
        base_feasible, base_util, base_bottleneck, base_utils, base_result = \
            _evaluate_at_flow(model, feed_stream_name, original_flow, timeout_ms, utilization_limit)
        has_utilization_data = len(base_utils) > 0

        iterations.append(OptimizationIteration(
            iteration=0, flow_rate_kg_hr=original_flow,
            feasible=base_feasible, max_utilization=base_util,
            bottleneck=base_bottleneck,
            detail=f"Base case: util={base_util*100:.1f}%",
        ))
    except Exception as e:
        return OptimizationResult(
            optimal_flow_kg_hr=original_flow,
            original_flow_kg_hr=original_flow,
            max_increase_pct=0,
            bottleneck_equipment="simulation_error",
            bottleneck_type="error",
            bottleneck_utilization=0,
            utilization_breakdown=[],
            iterations=iterations,
            converged=False,
            message=f"Base case simulation failed: {e}",
        )

    # --- Search strategy depends on whether we have utilization data ---
    if has_utilization_data:
        # Golden-section search: find max flow where utilization <= limit
        result = _golden_section_utilization(
            model, feed_stream_name,
            lo=min_flow_kg_hr, hi=max_flow_kg_hr,
            utilization_limit=utilization_limit,
            tolerance_pct=tolerance_pct,
            max_iterations=max_iterations,
            timeout_ms=timeout_ms,
            iterations=iterations,
        )
    else:
        # No utilization data → find max flow where simulation converges
        # Use binary search: increase flow until simulation fails, then bracket
        result = _binary_search_convergence(
            model, feed_stream_name,
            lo=original_flow, hi=max_flow_kg_hr,
            tolerance_pct=tolerance_pct,
            max_iterations=max_iterations,
            timeout_ms=timeout_ms,
            iterations=iterations,
        )

    # Compute increase percentage
    if original_flow > 0 and result.optimal_flow_kg_hr > 0:
        result.max_increase_pct = (
            (result.optimal_flow_kg_hr - original_flow) / original_flow * 100
        )
    result.original_flow_kg_hr = original_flow

    return result


def _golden_section_utilization(
    model: NeqSimProcessModel,
    feed_stream_name: str,
    lo: float,
    hi: float,
    utilization_limit: float,
    tolerance_pct: float,
    max_iterations: int,
    timeout_ms: int,
    iterations: List[OptimizationIteration],
) -> OptimizationResult:
    """
    Golden-section search for maximum flow where equipment utilization <= limit.

    The search maintains a bracket [lo, hi] where:
    - At lo, the process is feasible
    - At hi, the process is infeasible (or max flow bound)
    """
    best_flow = lo
    best_util = 0.0
    best_bottleneck = ""
    best_type = ""
    best_utils: List[EquipmentUtilization] = []
    best_kpis: Dict[str, KPI] = {}

    for i in range(1, max_iterations + 1):
        # Check convergence
        flow_range = hi - lo
        if flow_range <= 0:
            break
        if (flow_range / max(hi, 1.0)) * 100 <= tolerance_pct:
            break

        # Golden-section test point — test the upper probe
        # We want to find the largest feasible flow
        mid = lo + _RESPHI * (hi - lo)

        try:
            feasible, max_util, bottleneck, utils, result = \
                _evaluate_at_flow(model, feed_stream_name, mid, timeout_ms, utilization_limit)

            iterations.append(OptimizationIteration(
                iteration=i, flow_rate_kg_hr=mid,
                feasible=feasible, max_utilization=max_util,
                bottleneck=bottleneck,
                detail=f"util={max_util*100:.1f}%, {'OK' if feasible else 'EXCEEDED'}",
            ))

            if feasible:
                # This flow rate works — try higher
                best_flow = mid
                best_util = max_util
                best_bottleneck = bottleneck
                best_utils = utils
                best_kpis = result.kpis
                if utils:
                    worst = max(utils, key=lambda u: u.utilization)
                    best_type = worst.equipment_type
                lo = mid
            else:
                # Too high — reduce
                hi = mid

        except Exception:
            # Simulation failed — treat as infeasible
            iterations.append(OptimizationIteration(
                iteration=i, flow_rate_kg_hr=mid,
                feasible=False, max_utilization=999.0,
                bottleneck="simulation_error",
                detail="Simulation failed at this flow rate",
            ))
            hi = mid

    return OptimizationResult(
        optimal_flow_kg_hr=best_flow,
        original_flow_kg_hr=0,  # filled by caller
        max_increase_pct=0,     # filled by caller
        bottleneck_equipment=best_bottleneck,
        bottleneck_type=best_type,
        bottleneck_utilization=best_util,
        utilization_breakdown=best_utils,
        iterations=iterations,
        kpis_at_optimum=best_kpis,
        search_algorithm="golden_section (utilization)",
        converged=True,
        message=(
            f"Maximum feasible flow: {best_flow:.0f} kg/hr "
            f"(bottleneck: {best_bottleneck}, utilization: {best_util*100:.1f}%)"
        ),
    )


def _binary_search_convergence(
    model: NeqSimProcessModel,
    feed_stream_name: str,
    lo: float,
    hi: float,
    tolerance_pct: float,
    max_iterations: int,
    timeout_ms: int,
    iterations: List[OptimizationIteration],
) -> OptimizationResult:
    """
    Binary search for maximum flow where the simulation still converges.

    Used when no equipment utilization data is available (no design limits set).
    Increases flow until simulation fails, then narrows the bracket.
    """
    best_flow = lo
    best_kpis: Dict[str, KPI] = {}

    for i in range(1, max_iterations + 1):
        flow_range = hi - lo
        if flow_range <= 0:
            break
        if (flow_range / max(hi, 1.0)) * 100 <= tolerance_pct:
            break

        mid = (lo + hi) / 2.0

        try:
            _, max_util, bottleneck, utils, result = \
                _evaluate_at_flow(model, feed_stream_name, mid, timeout_ms)

            # Check if simulation produced reasonable results
            # (sometimes NeqSim "converges" but produces NaN or zero flow)
            out_ok = True
            for kpi_name, kpi in result.kpis.items():
                if math.isnan(kpi.value) or math.isinf(kpi.value):
                    out_ok = False
                    break

            feasible = out_ok

            iterations.append(OptimizationIteration(
                iteration=i, flow_rate_kg_hr=mid,
                feasible=feasible, max_utilization=max_util,
                bottleneck=bottleneck or ("convergence" if not feasible else ""),
                detail=f"{'Converged' if feasible else 'Failed/NaN'}",
            ))

            if feasible:
                best_flow = mid
                best_kpis = result.kpis
                lo = mid
            else:
                hi = mid

        except Exception as e:
            iterations.append(OptimizationIteration(
                iteration=i, flow_rate_kg_hr=mid,
                feasible=False, max_utilization=0,
                bottleneck="simulation_error",
                detail=f"Error: {str(e)[:80]}",
            ))
            hi = mid

    return OptimizationResult(
        optimal_flow_kg_hr=best_flow,
        original_flow_kg_hr=0,
        max_increase_pct=0,
        bottleneck_equipment="convergence_limit",
        bottleneck_type="simulation",
        bottleneck_utilization=0,
        utilization_breakdown=[],
        iterations=iterations,
        kpis_at_optimum=best_kpis,
        search_algorithm="binary_search (convergence)",
        converged=True,
        message=(
            f"Maximum convergent flow: {best_flow:.0f} kg/hr "
            f"(no equipment utilization data — based on simulation convergence only)"
        ),
    )


# ---------------------------------------------------------------------------
# Format results for LLM and chat display
# ---------------------------------------------------------------------------

def format_optimization_result(result: OptimizationResult) -> str:
    """Format an OptimizationResult into text for the LLM to interpret."""
    lines = []

    lines.append("=== PROCESS OPTIMIZATION RESULT ===")
    lines.append(f"  Algorithm: {result.search_algorithm}")
    lines.append(f"  Converged: {'Yes' if result.converged else 'No'}")
    lines.append(f"  Original feed flow: {result.original_flow_kg_hr:.0f} kg/hr")
    lines.append(f"  Optimal feed flow:  {result.optimal_flow_kg_hr:.0f} kg/hr")
    lines.append(f"  Max increase:       {result.max_increase_pct:+.1f}%")
    lines.append("")

    if result.bottleneck_equipment:
        lines.append(f"  Bottleneck: {result.bottleneck_equipment} ({result.bottleneck_type})")
        lines.append(f"  Bottleneck utilization: {result.bottleneck_utilization*100:.1f}%")
        lines.append("")

    if result.utilization_breakdown:
        lines.append("  Equipment Utilization at Optimum:")
        # Sort by utilization descending
        sorted_utils = sorted(result.utilization_breakdown,
                              key=lambda u: u.utilization, reverse=True)
        for u in sorted_utils:
            bar = "█" * int(u.utilization * 20) + "░" * (20 - int(u.utilization * 20))
            lines.append(f"    {u.name:30s} [{bar}] {u.utilization*100:5.1f}%  ({u.constraint_name})")
        lines.append("")

    if result.kpis_at_optimum:
        lines.append("  Key KPIs at Optimum:")
        # Show power, duty, and important stream conditions
        for k, kpi in sorted(result.kpis_at_optimum.items()):
            if any(s in k for s in ('.power_kW', '.duty_kW', 'total_', 'mass_balance',
                                     '.temperature_C', '.pressure_bara', '.flow_kg_hr')):
                lines.append(f"    {k}: {kpi.value:.2f} {kpi.unit}")
        lines.append("")

    if result.iterations:
        lines.append("  Search Iterations:")
        for it in result.iterations:
            icon = "✓" if it.feasible else "✗"
            lines.append(
                f"    {icon} iter {it.iteration}: flow={it.flow_rate_kg_hr:.0f} kg/hr, "
                f"util={it.max_utilization*100:.1f}%, {it.detail}"
            )
        lines.append("")

    lines.append(f"  Summary: {result.message}")

    return "\n".join(lines)
