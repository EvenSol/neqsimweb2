"""
Scenario Engine — clone → patch → validate → run → compare.

Works with uploaded .neqsim process models via NeqSimProcessModel adapter.

Two modes:
  1. Direct model manipulation (set unit/stream values directly on Java objects)
  2. Input-contract mode (for processes with a structured input like ProcessInput)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from .patch_schema import (
    InputPatch,
    Scenario,
    AddUnitOp,
    AddComponentOp,
    AddStreamOp,
    TargetSpec,
    AddProcessOp,
)
from .process_model import NeqSimProcessModel, ModelRunResult, KPI, ConstraintStatus


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    scenario: Scenario
    result: ModelRunResult
    success: bool = True
    error: Optional[str] = None


@dataclass
class Comparison:
    base: ScenarioResult
    cases: List[ScenarioResult]
    delta_kpis: List[Dict[str, Any]]
    constraint_summary: List[Dict[str, Any]]
    patch_log: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Equipment insertion — add new units to the process topology
# ---------------------------------------------------------------------------

# Map template names to Java class paths — covers all major NeqSim equipment
_EQUIPMENT_CLASSES = {
    # Heat exchangers
    "cooler": "neqsim.process.equipment.heatexchanger.Cooler",
    "heater": "neqsim.process.equipment.heatexchanger.Heater",
    "heat_exchanger": "neqsim.process.equipment.heatexchanger.HeatExchanger",
    "air_cooler": "neqsim.process.equipment.heatexchanger.AirCooler",
    "water_cooler": "neqsim.process.equipment.heatexchanger.WaterCooler",
    "multi_stream_heat_exchanger": "neqsim.process.equipment.heatexchanger.MultiStreamHeatExchanger",
    # Compressors
    "compressor": "neqsim.process.equipment.compressor.Compressor",
    # Expanders
    "expander": "neqsim.process.equipment.expander.Expander",
    "turbo_expander_compressor": "neqsim.process.equipment.expander.TurboExpanderCompressor",
    # Separators
    "separator": "neqsim.process.equipment.separator.Separator",
    "two_phase_separator": "neqsim.process.equipment.separator.TwoPhaseSeparator",
    "three_phase_separator": "neqsim.process.equipment.separator.ThreePhaseSeparator",
    "gas_scrubber": "neqsim.process.equipment.separator.GasScrubber",
    "gas_scrubber_simple": "neqsim.process.equipment.separator.GasScrubberSimple",
    "hydrocyclone": "neqsim.process.equipment.separator.Hydrocyclone",
    # Valves
    "valve": "neqsim.process.equipment.valve.ThrottlingValve",
    "control_valve": "neqsim.process.equipment.valve.ControlValve",
    "check_valve": "neqsim.process.equipment.valve.CheckValve",
    # Pumps
    "pump": "neqsim.process.equipment.pump.Pump",
    "esp_pump": "neqsim.process.equipment.pump.ESPPump",
    # Mixers / Splitters
    "mixer": "neqsim.process.equipment.mixer.Mixer",
    "splitter": "neqsim.process.equipment.splitter.Splitter",
    "component_splitter": "neqsim.process.equipment.splitter.ComponentSplitter",
    # Absorbers / columns
    "simple_absorber": "neqsim.process.equipment.absorber.SimpleAbsorber",
    "simple_teg_absorber": "neqsim.process.equipment.absorber.SimpleTEGAbsorber",
    "water_stripper_column": "neqsim.process.equipment.absorber.WaterStripperColumn",
    "distillation_column": "neqsim.process.equipment.distillation.DistillationColumn",
    # Pipeline / pipe
    "pipeline": "neqsim.process.equipment.pipeline.Pipeline",
    "adiabatic_pipe": "neqsim.process.equipment.pipeline.AdiabaticPipe",
    "adiabatic_two_phase_pipe": "neqsim.process.equipment.pipeline.AdiabaticTwoPhasePipe",
    "simple_tp_out_pipeline": "neqsim.process.equipment.pipeline.SimpleTPoutPipeline",
    # Reactors
    "gibbs_reactor": "neqsim.process.equipment.reactor.GibbsReactor",
    # Ejectors
    "ejector": "neqsim.process.equipment.ejector.Ejector",
    # Flare
    "flare": "neqsim.process.equipment.flare.Flare",
    # Filter
    "filter": "neqsim.process.equipment.filter.Filter",
    # Membrane
    "membrane_separator": "neqsim.process.equipment.membrane.MembraneSeparator",
    # Power generation
    "gas_turbine": "neqsim.process.equipment.powergeneration.GasTurbine",
    # Reservoir / well
    "well_flow": "neqsim.process.equipment.reservoir.WellFlow",
    "simple_reservoir": "neqsim.process.equipment.reservoir.SimpleReservoir",
    # Tank
    "tank": "neqsim.process.equipment.tank.Tank",
    # Subsea
    "simple_flow_line": "neqsim.process.equipment.subsea.SimpleFlowLine",
    # Streams
    "stream": "neqsim.process.equipment.stream.Stream",
    "equilibrium_stream": "neqsim.process.equipment.stream.EquilibriumStream",
    # Utilities (adjuster, recycle, setter)
    "adjuster": "neqsim.process.equipment.util.Adjuster",
    "recycle": "neqsim.process.equipment.util.Recycle",
    "calculator": "neqsim.process.equipment.util.Calculator",
    "set_point": "neqsim.process.equipment.util.SetPoint",
    # Electrolyzer
    "electrolyzer": "neqsim.process.equipment.electrolyzer.Electrolyzer",
    "co2_electrolyzer": "neqsim.process.equipment.electrolyzer.CO2Electrolyzer",
    # Adsorber
    "adsorber": "neqsim.process.equipment.adsorber.SimpleAdsorber",
}


def _find_outlet_stream(unit):
    """Get the outlet stream of a unit by trying common getter methods."""
    for method in ("getOutletStream", "getOutStream", "getGasOutStream"):
        if hasattr(unit, method):
            try:
                s = getattr(unit, method)()
                if s is not None:
                    return s
            except Exception:
                pass
    return None


def _set_inlet_stream(unit, stream):
    """Set the inlet stream of a unit by trying common setter methods."""
    for method in ("setInletStream", "setInStream", "setFeed"):
        if hasattr(unit, method):
            try:
                getattr(unit, method)(stream)
                return True
            except Exception:
                pass
    return False


def _get_java_class(java_class_path: str):
    """Import and return a Java class from its dotted path."""
    from neqsim import jneqsim
    parts = java_class_path.rsplit(".", 1)
    pkg_path, cls_name = parts[0], parts[1]
    java_pkg = jneqsim
    for p in pkg_path.split(".")[1:]:
        java_pkg = getattr(java_pkg, p)
    return getattr(java_pkg, cls_name)


# Reverse map: Java class simple names -> equipment class paths
_JAVA_CLASS_TO_PATH = {}
for eq_name, path in _EQUIPMENT_CLASSES.items():
    cls = path.rsplit(".", 1)[1]
    _JAVA_CLASS_TO_PATH[cls] = path
# Add Stream mapping
_JAVA_CLASS_TO_PATH["Stream"] = "neqsim.process.equipment.stream.Stream"


def _recreate_unit(original_unit, inlet_stream):
    """
    Create a fresh copy of a unit operation using a new inlet stream.
    
    This is necessary because NeqSim units create internal stream references
    during construction. Simply calling setInletStream() doesn't propagate
    to gas/liquid outlet streams in separators etc.
    
    Returns the new unit with parameters copied from the original.
    """
    from neqsim import jneqsim
    
    name = str(original_unit.getName())
    java_class = str(original_unit.getClass().getSimpleName())
    
    # Special case: Stream objects just need their fluid updated
    if java_class == "Stream":
        StreamClass = jneqsim.process.equipment.stream.Stream
        new_stream = StreamClass(name, inlet_stream)
        return new_stream
    
    # Get the Java class path
    class_path = _JAVA_CLASS_TO_PATH.get(java_class)
    if class_path is None:
        # Unknown type — try using setInletStream as fallback
        _set_inlet_stream(original_unit, inlet_stream)
        return original_unit
    
    # Create new instance
    JavaClass = _get_java_class(class_path)
    
    # Most equipment constructors take (name, inletStream)
    try:
        new_unit = JavaClass(name, inlet_stream)
    except Exception:
        # Some might need different constructor args
        try:
            new_unit = JavaClass(inlet_stream)
            new_unit.setName(name)
        except Exception:
            # Last resort: reuse original with setInletStream
            _set_inlet_stream(original_unit, inlet_stream)
            return original_unit
    
    # Copy configurable parameters from original
    param_getters_setters = [
        ("getOutletPressure", "setOutletPressure", None),
        ("getIsentropicEfficiency", "setIsentropicEfficiency", None),
        ("getPolytropicEfficiency", "setPolytropicEfficiency", None),
    ]
    
    for getter, setter, unit_arg in param_getters_setters:
        if hasattr(original_unit, getter) and hasattr(new_unit, setter):
            try:
                val = getattr(original_unit, getter)()
                if val is not None and float(val) != 0.0:
                    if unit_arg:
                        getattr(new_unit, setter)(float(val), unit_arg)
                    else:
                        getattr(new_unit, setter)(float(val))
            except Exception:
                pass
    
    # Copy outlet temperature if cooler/heater
    if java_class in ("Cooler", "Heater"):
        try:
            # Get outlet temperature spec from original
            out_t = float(original_unit.getOutletStream().getTemperature("C"))
            # Check if it was actually specified (not just inlet T passing through)
            in_t = float(original_unit.getInletStream().getTemperature("C"))
            if abs(out_t - in_t) > 0.1:  # Temperature was actually changed
                new_unit.setOutTemperature(out_t, "C")
        except Exception:
            pass
    
    # Copy outlet pressure settings
    if hasattr(original_unit, "getOutPressure") and hasattr(new_unit, "setOutPressure"):
        try:
            val = float(original_unit.getOutPressure())
            if val > 0:
                new_unit.setOutPressure(val, "bara")
        except Exception:
            pass

    # -------------------------------------------------------------------
    # Compressor-specific: preserve chart, speed, polytropic settings
    # -------------------------------------------------------------------
    if java_class == "Compressor":
        # Copy compressor chart (expensive to regenerate)
        try:
            if hasattr(original_unit, "getCompressorChart"):
                chart = original_unit.getCompressorChart()
                if chart is not None:
                    new_unit.setCompressorChartType('interpolate and extrapolate')
                    new_unit.setCompressorChart(chart)
                    try:
                        new_unit.getCompressorChart().setHeadUnit('kJ/kg')
                    except Exception:
                        pass
                    if hasattr(new_unit, "setSolveSpeed"):
                        new_unit.setSolveSpeed(True)
        except Exception:
            pass

        # Copy speed setting
        try:
            if hasattr(original_unit, "getSpeed") and hasattr(new_unit, "setSpeed"):
                speed = float(original_unit.getSpeed())
                if speed > 0:
                    new_unit.setSpeed(speed)
        except Exception:
            pass

        # Copy polytropic calc flag
        try:
            if hasattr(original_unit, "getUsePolytropicCalc"):
                new_unit.setUsePolytropicCalc(original_unit.getUsePolytropicCalc())
        except Exception:
            pass

        # Copy compression ratio if set
        try:
            if hasattr(original_unit, "getCompressionRatio") and hasattr(new_unit, "setCompressionRatio"):
                cr = float(original_unit.getCompressionRatio())
                if cr > 1.0:
                    new_unit.setCompressionRatio(cr)
        except Exception:
            pass

    # -------------------------------------------------------------------
    # Separator / valve / pump: preserve autoSize dimensions
    # -------------------------------------------------------------------
    if java_class in ("Separator", "TwoPhaseSeparator", "ThreePhaseSeparator",
                       "GasScrubber", "GasScrubberSimple"):
        for getter, setter in [
            ("getInternalDiameter", "setInternalDiameter"),
            ("getSeparatorLength", "setSeparatorLength"),
        ]:
            if hasattr(original_unit, getter) and hasattr(new_unit, setter):
                try:
                    val = float(getattr(original_unit, getter)())
                    if val > 0:
                        getattr(new_unit, setter)(val)
                except Exception:
                    pass

    return new_unit


# ---------------------------------------------------------------------------
# Adding chemical components to a stream's fluid
# ---------------------------------------------------------------------------

def apply_add_components(
    model: NeqSimProcessModel,
    add_components: List[AddComponentOp],
    scale_factor: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Add chemical components to stream fluids.
    
    Each AddComponentOp specifies a stream and a dict of component→flow_rate.
    The scale_factor multiplies all flow rates (used by iterative solver).
    
    Returns a log of operations.
    """
    from neqsim import jneqsim
    log = []

    for ac in add_components:
        try:
            # Find the stream (try qualified and unqualified names)
            stream_obj = None
            try:
                stream_obj = model.get_stream(ac.stream_name)
            except KeyError:
                # Try to find via units
                for uname, u in model._units.items():
                    if uname == ac.stream_name:
                        stream_obj = u
                        break
                    # Try outlet
                    out = _find_outlet_stream(u)
                    if out is not None:
                        try:
                            if str(out.getName()) == ac.stream_name:
                                stream_obj = out
                                break
                        except Exception:
                            pass
            
            if stream_obj is None:
                log.append({
                    "key": f"add_components.{ac.stream_name}",
                    "status": "FAILED",
                    "error": f"Stream '{ac.stream_name}' not found",
                })
                continue

            # Get the fluid (thermoSystem) from the stream
            fluid = None
            for getter in ("getFluid", "getThermoSystem"):
                if hasattr(stream_obj, getter):
                    try:
                        fluid = getattr(stream_obj, getter)()
                        if fluid is not None:
                            break
                    except Exception:
                        pass

            if fluid is None:
                log.append({
                    "key": f"add_components.{ac.stream_name}",
                    "status": "FAILED",
                    "error": "Cannot access fluid from stream",
                })
                continue

            # Add each component
            added = []
            for comp_name, base_flow in ac.components.items():
                scaled_flow = base_flow * scale_factor
                try:
                    fluid.addComponent(comp_name, scaled_flow, ac.flow_unit)
                    added.append(f"{comp_name}={scaled_flow:.2f} {ac.flow_unit}")
                except Exception as e:
                    log.append({
                        "key": f"add_components.{ac.stream_name}.{comp_name}",
                        "status": "FAILED",
                        "error": f"Failed to add {comp_name}: {e}",
                    })

            # Re-apply mixing rule after adding components
            try:
                fluid.setMixingRule(2)
            except Exception:
                pass

            if added:
                log.append({
                    "key": f"add_components.{ac.stream_name}",
                    "value": "; ".join(added),
                    "status": "OK",
                    "scale_factor": scale_factor,
                })

        except Exception as e:
            log.append({
                "key": f"add_components.{ac.stream_name}",
                "status": "FAILED",
                "error": str(e),
            })

    return log


# ---------------------------------------------------------------------------
# Adding a new inlet stream and mixing into the process
# ---------------------------------------------------------------------------

def apply_add_streams(
    model: NeqSimProcessModel,
    add_streams: List[AddStreamOp],
) -> List[Dict[str, Any]]:
    """
    Add new inlet streams and mix them into the process using a Mixer.

    Each AddStreamOp creates a new Stream with the specified composition and
    inserts a Mixer after the specified unit to combine the upstream outlet
    with the new stream.
    """
    from neqsim import jneqsim

    log = []

    for ast in add_streams:
        try:
            proc = model.get_process()
            original_units = list(proc.getUnitOperations())

            after_idx = None
            for i, u in enumerate(original_units):
                try:
                    if str(u.getName()) == ast.insert_after:
                        after_idx = i
                        break
                except Exception:
                    pass

            if after_idx is None:
                log.append({
                    "key": f"add_stream.{ast.name}",
                    "status": "FAILED",
                    "error": f"Unit '{ast.insert_after}' not found in process",
                })
                continue

            upstream_unit = original_units[after_idx]
            upstream_outlet = _find_outlet_stream(upstream_unit)
            if upstream_outlet is None:
                log.append({
                    "key": f"add_stream.{ast.name}",
                    "status": "FAILED",
                    "error": f"Cannot find outlet stream for '{ast.insert_after}'",
                })
                continue

            # Resolve a base stream for cloning the thermo system
            base_stream = None
            if ast.base_stream:
                try:
                    base_stream = model.get_stream(ast.base_stream)
                except KeyError:
                    log.append({
                        "key": f"add_stream.{ast.name}.base_stream",
                        "status": "WARN",
                        "error": f"Base stream '{ast.base_stream}' not found; using upstream outlet",
                    })
            if base_stream is None:
                base_stream = upstream_outlet

            # Clone thermo system
            fluid = None
            for getter in ("getFluid", "getThermoSystem"):
                if hasattr(base_stream, getter):
                    try:
                        fluid = getattr(base_stream, getter)()
                        if fluid is not None:
                            break
                    except Exception:
                        pass

            if fluid is None:
                log.append({
                    "key": f"add_stream.{ast.name}",
                    "status": "FAILED",
                    "error": "Cannot access thermo system from base stream",
                })
                continue

            new_fluid = fluid.clone()
            # Reset total flow before adding components
            try:
                new_fluid.setTotalFlowRate(0.0, ast.flow_unit)
            except Exception:
                try:
                    new_fluid.setTotalFlowRate(0.0, "kg/hr")
                except Exception:
                    pass

            if ast.temperature_C is not None:
                try:
                    new_fluid.setTemperature(float(ast.temperature_C), "C")
                except Exception:
                    pass
            if ast.pressure_bara is not None:
                try:
                    new_fluid.setPressure(float(ast.pressure_bara), "bara")
                except Exception:
                    pass

            added = []
            for comp_name, flow in ast.components.items():
                try:
                    new_fluid.addComponent(comp_name, float(flow), ast.flow_unit)
                    added.append(f"{comp_name}={float(flow):.2f} {ast.flow_unit}")
                except Exception as e:
                    log.append({
                        "key": f"add_stream.{ast.name}.{comp_name}",
                        "status": "FAILED",
                        "error": f"Failed to add {comp_name}: {e}",
                    })

            try:
                new_fluid.setMixingRule(2)
            except Exception:
                pass

            # Create new Stream unit
            StreamClass = jneqsim.process.equipment.stream.Stream
            new_stream = StreamClass(ast.name, new_fluid)

            # Create Mixer and connect streams
            MixerClass = jneqsim.process.equipment.mixer.Mixer
            mixer_name = ast.mixer_name or f"mixer_{ast.name}"
            try:
                mixer = MixerClass(mixer_name, upstream_outlet)
            except Exception:
                mixer = MixerClass(mixer_name)
                try:
                    mixer.addStream(upstream_outlet)
                except Exception:
                    pass

            try:
                mixer.addStream(new_stream)
            except Exception as e:
                log.append({
                    "key": f"add_stream.{ast.name}.mixer",
                    "status": "FAILED",
                    "error": f"Failed to connect new stream to mixer: {e}",
                })
                continue

            # Build the new unit sequence: upstream -> new stream -> mixer -> downstream
            new_sequence = list(original_units)
            insert_pos = after_idx + 1
            new_sequence.insert(insert_pos, new_stream)
            new_sequence.insert(insert_pos + 1, mixer)

            # Recreate all downstream units after the mixer
            downstream_start = insert_pos + 2
            for i in range(downstream_start, len(new_sequence)):
                prev_unit = new_sequence[i - 1]
                prev_outlet = _find_outlet_stream(prev_unit)
                if prev_outlet is None:
                    log.append({
                        "key": f"add_stream.{ast.name}.rebuild.{i}",
                        "status": "WARN",
                        "error": f"Cannot find outlet of unit {i-1} for downstream reconnect",
                    })
                    continue

                unit = new_sequence[i]
                if unit is new_stream or unit is mixer:
                    continue
                new_sequence[i] = _recreate_unit(unit, prev_outlet)

            # Rebuild process
            unit_ops = proc.getUnitOperations()
            unit_ops.clear()
            for unit in new_sequence:
                proc.add(unit)

            model._index_model_objects()

            log.append({
                "key": f"add_stream.{ast.name}",
                "value": f"stream + mixer after '{ast.insert_after}'",
                "status": "OK",
                "components": "; ".join(added) if added else "",
                "mixer": mixer_name,
            })

        except Exception as e:
            log.append({
                "key": f"add_stream.{ast.name}",
                "status": "FAILED",
                "error": str(e),
            })

    return log


# ---------------------------------------------------------------------------
# Iterative target-seeking solver
# ---------------------------------------------------------------------------

def _resolve_kpi_name(kpis: dict, target_kpi: str) -> Optional[str]:
    """Find exact or partial-match KPI key, return the resolved name or None."""
    if target_kpi in kpis:
        return target_kpi
    for k in kpis:
        if target_kpi in k:
            return k
    # Try case-insensitive match
    lower = target_kpi.lower()
    for k in kpis:
        if lower in k.lower():
            return k
    return None


def _scale_stream_flow(model: NeqSimProcessModel, stream_name: str, scale: float):
    """Scale a stream's flow rate by *scale* relative to its current value."""
    stream = model.get_stream(stream_name)       # raises KeyError if not found
    current = float(stream.getFlowRate("kg/hr"))
    stream.setFlowRate(current * scale, "kg/hr")


def solve_for_target(
    model: NeqSimProcessModel,
    scenario: Scenario,
    timeout_ms: int = 120000,
) -> 'Comparison':
    """
    Iteratively adjust inputs until a target KPI is reached.
    
    Supports three modes via ``target.variable``:
      * ``"component_scale"`` — scale add_component flows by the bisection factor
      * ``"stream_scale"`` — multiply the named stream's flow rate by the factor
      * ``"unit_param"`` — set a unit parameter directly to the bisection mid-point
    
    Uses bisection on the scale factor (or parameter value for unit_param).
    The target is specified in scenario.patch.targets[0].
    
    Returns a Comparison with the converged result.
    """
    target = scenario.patch.targets[0]
    
    lo = target.min_value
    hi = target.max_value
    best_scale = target.initial_guess
    best_result = None
    best_error = float('inf')
    
    use_stream_scale = (target.variable or "").lower() == "stream_scale"
    use_unit_param = (target.variable or "").lower() == "unit_param"
    stream_name = target.stream_name or ""
    unit_name = target.unit_name or ""
    unit_param = target.unit_param or ""
    # Strip "streams." prefix the LLM sometimes adds
    for prefix in ("streams.",):
        if stream_name.startswith(prefix):
            stream_name = stream_name[len(prefix):]
    
    iteration_log = []
    
    # First: run base case
    base_clone = model.clone()
    base_result = base_clone.run(timeout_ms=timeout_ms)
    base_scenario = Scenario(
        name="BASE", description="Base case (no changes)",
        patch=InputPatch(changes={})
    )
    base_sr = ScenarioResult(scenario=base_scenario, result=base_result)
    
    # Resolve the target KPI name against the base result
    resolved = _resolve_kpi_name(base_result.kpis, target.target_kpi)
    if resolved:
        target.target_kpi = resolved
    
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3   # bail if the same error repeats
    
    # Bisection iterations
    for iteration in range(target.max_iterations):
        mid = (lo + hi) / 2.0 if iteration > 0 else best_scale
        
        try:
            clone = model.clone()
            
            # ---- Apply the manipulated variable ----
            if use_stream_scale and stream_name:
                try:
                    _scale_stream_flow(clone, stream_name, mid)
                except KeyError as e:
                    iteration_log.append({
                        "iteration": iteration, "scale": mid,
                        "status": "FAILED", "error": f"Stream not found: {e}"
                    })
                    break
            
            if use_unit_param and unit_name and unit_param:
                try:
                    u = clone.get_unit(unit_name)
                    _set_unit_value(u, unit_param, mid)
                except KeyError as e:
                    iteration_log.append({
                        "iteration": iteration, "value": mid,
                        "status": "FAILED", "error": f"Unit not found: {e}"
                    })
                    break
                except Exception as e:
                    iteration_log.append({
                        "iteration": iteration, "value": mid,
                        "status": "FAILED", "error": f"Cannot set {unit_param}: {e}"
                    })
                    break
            
            # Apply add_components with current scale (component_scale mode)
            if scenario.patch.add_components:
                sf = mid if not (use_stream_scale or use_unit_param) else 1.0
                comp_log = apply_add_components(clone, scenario.patch.add_components, scale_factor=sf)
                failed = [e for e in comp_log if e.get("status") == "FAILED"]
                if failed:
                    consecutive_failures += 1
                    iteration_log.append({
                        "iteration": iteration, "scale": mid,
                        "status": "FAILED", "error": str(failed)
                    })
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        iteration_log[-1]["status"] = "ABORT"
                        iteration_log[-1]["error"] += " (aborting — repeated failures)"
                        break
                    hi = mid
                    continue
            
            # Apply add_units if any
            if scenario.patch.add_units:
                apply_add_units(clone, scenario.patch.add_units)

            # Apply add_streams if any
            if scenario.patch.add_streams:
                apply_add_streams(clone, scenario.patch.add_streams)
            
            # Apply parameter changes if any
            if scenario.patch.changes:
                apply_patch_to_model(clone, scenario.patch)
            
            # Run
            result = clone.run(timeout_ms=timeout_ms)
            
            # Read the target KPI
            resolved_key = _resolve_kpi_name(result.kpis, target.target_kpi)
            if resolved_key:
                target.target_kpi = resolved_key
                kpi = result.kpis[resolved_key]
            else:
                kpi = None
            
            if kpi is None:
                iteration_log.append({
                    "iteration": iteration, "scale": mid,
                    "status": "FAILED",
                    "error": f"KPI '{target.target_kpi}' not found. Available: {list(result.kpis.keys())}"
                })
                break
            
            consecutive_failures = 0  # success — reset counter
            current_val = kpi.value
            error_pct = abs(current_val - target.target_value) / max(abs(target.target_value), 1e-10) * 100
            
            iteration_log.append({
                "iteration": iteration, "scale": round(mid, 4),
                "kpi_value": round(current_val, 2),
                "target": target.target_value,
                "error_pct": round(error_pct, 2),
                "status": "OK",
            })
            
            # Track best result
            if error_pct < best_error:
                best_error = error_pct
                best_scale = mid
                best_result = result
            
            # Converged?
            if error_pct <= target.tolerance_pct:
                iteration_log[-1]["status"] = "CONVERGED"
                break
            
            # Bisection: adjust bounds
            if current_val < target.target_value:
                lo = mid  # need more
            else:
                hi = mid  # need less
                
        except Exception as e:
            iteration_log.append({
                "iteration": iteration, "scale": mid,
                "status": "FAILED", "error": str(e)
            })
            hi = mid
    
    # Build the Comparison result
    if best_result is None:
        case_sr = ScenarioResult(
            scenario=scenario,
            result=ModelRunResult(kpis={}, constraints=[], raw={}),
            success=False,
            error=f"Iterative solver failed to converge. Log: {iteration_log}"
        )
        return Comparison(
            base=base_sr, cases=[case_sr],
            delta_kpis=[], constraint_summary=[],
            patch_log=[{"scenario": scenario.name, "key": "solver", "status": "FAILED",
                       "iterations": iteration_log}],
        )
    
    # Successful convergence (or best effort)
    case_sr = ScenarioResult(scenario=scenario, result=best_result)
    deltas = compare_results(base_result, best_result, scenario.name)
    
    constraints_summary = []
    for c in best_result.constraints:
        constraints_summary.append({
            "scenario": scenario.name,
            "constraint": c.name,
            "status": c.status,
            "detail": c.detail,
        })
    
    # Build detailed patch log
    if use_unit_param:
        solver_value = (
            f"{unit_name}.{unit_param}={best_scale:.4f}, "
            f"target={target.target_kpi}={target.target_value}, "
            f"achieved_error={best_error:.2f}%"
        )
    else:
        solver_value = (
            f"scale={best_scale:.4f}, target={target.target_kpi}={target.target_value}, "
            f"achieved_error={best_error:.2f}%"
        )

    patch_log = [
        {
            "scenario": scenario.name,
            "key": "iterative_solver",
            "status": "CONVERGED" if best_error <= target.tolerance_pct else "BEST_EFFORT",
            "value": solver_value,
            "iterations": len(iteration_log),
            "iteration_log": iteration_log,
        }
    ]
    
    # Add what components were added at the final scale
    if scenario.patch.add_components:
        for ac in scenario.patch.add_components:
            comp_detail = [f"{comp}: {flow * best_scale:.2f} {ac.flow_unit}" 
                          for comp, flow in ac.components.items()]
            patch_log.append({
                "scenario": scenario.name,
                "key": f"add_components.{ac.stream_name}",
                "value": "; ".join(sorted(comp_detail)),
                "status": "OK",
                "scale_factor": round(best_scale, 4),
            })

    # Log stream scaling info
    if use_stream_scale and stream_name:
        patch_log.append({
            "scenario": scenario.name,
            "key": f"stream_scale.{stream_name}",
            "value": f"flow scaled by {best_scale:.4f}x",
            "status": "OK",
            "scale_factor": round(best_scale, 4),
        })
    
    # Log unit_param adjustment info
    if use_unit_param and unit_name and unit_param:
        patch_log.append({
            "scenario": scenario.name,
            "key": f"unit_param.{unit_name}.{unit_param}",
            "value": f"{unit_param} set to {best_scale:.4f}",
            "status": "OK",
            "final_value": round(best_scale, 4),
        })
    
    return Comparison(
        base=base_sr, cases=[case_sr],
        delta_kpis=deltas, constraint_summary=constraints_summary,
        patch_log=patch_log,
    )


def apply_add_units(model: NeqSimProcessModel, add_units: List[AddUnitOp]) -> List[Dict[str, Any]]:
    """
    Insert new equipment into the process topology by rebuilding process.
    
    NeqSim's ProcessSystem requires units to be added in order during construction
    for proper stream wiring. Simply inserting a unit mid-process doesn't work
    because the process runner doesn't initialize newly inserted units.
    
    Strategy: rebuild the process from scratch with new units in the right positions.
    
    For each AddUnitOp:
      1. Find the 'insert_after' unit position
      2. Extract the ordered unit list
      3. Create new equipment using the upstream unit's outlet stream
      4. Reconnect downstream units to the new equipment's outlet
      5. Rebuild the process in the correct order
    
    Returns a log of operations.
    """
    from neqsim import jneqsim
    from neqsim.process import clearProcess, getProcess

    log = []
    proc = model.get_process()

    # Collect all current units in order
    original_units = list(proc.getUnitOperations())
    
    # Build an insertion plan: list of (position, AddUnitOp) 
    # Process all add_units first to determine where each goes
    insertions = []  # (after_index, add_op)
    
    for add_op in add_units:
        # Find the 'insert_after' unit by name
        after_idx = None
        for i, u in enumerate(original_units):
            try:
                if str(u.getName()) == add_op.insert_after:
                    after_idx = i
                    break
            except Exception:
                pass
        
        if after_idx is None:
            log.append({
                "key": f"add_unit.{add_op.name}",
                "status": "FAILED",
                "error": f"Unit '{add_op.insert_after}' not found in process"
            })
            continue
        
        insertions.append((after_idx, add_op))
    
    if not insertions:
        return log
    
    # Sort insertions by position (descending so indices don't shift)
    insertions.sort(key=lambda x: x[0], reverse=True)
    
    # Build the new unit sequence
    new_sequence = list(original_units)  # copy of references
    
    # Track which new units we create (to configure and log)
    created_units = []
    
    for after_idx, add_op in insertions:
        try:
            eq_type = add_op.equipment_type.lower().replace(" ", "_")
            java_class_path = _EQUIPMENT_CLASSES.get(eq_type)
            
            if java_class_path is None:
                log.append({
                    "key": f"add_unit.{add_op.name}",
                    "status": "FAILED",
                    "error": f"Unknown equipment type: '{add_op.equipment_type}'. "
                             f"Available: {', '.join(_EQUIPMENT_CLASSES.keys())}"
                })
                continue
            
            # Get outlet stream from upstream unit
            upstream_unit = new_sequence[after_idx]
            outlet_stream = _find_outlet_stream(upstream_unit)
            if outlet_stream is None:
                log.append({
                    "key": f"add_unit.{add_op.name}",
                    "status": "FAILED",
                    "error": f"Cannot find outlet stream for '{add_op.insert_after}'"
                })
                continue
            
            # Create the Java class
            parts = java_class_path.rsplit(".", 1)
            pkg = parts[0]
            cls_name = parts[1]
            java_pkg = jneqsim
            for p in pkg.split(".")[1:]:
                java_pkg = getattr(java_pkg, p)
            JavaClass = getattr(java_pkg, cls_name)
            
            # Create new unit with upstream outlet as inlet
            new_unit = JavaClass(add_op.name, outlet_stream)
            
            # Configure params
            params = dict(add_op.params)
            for pkey, pval in params.items():
                pk = pkey.lower()
                try:
                    if "outlet_temperature" in pk or "out_temperature" in pk or "outtemperature" in pk:
                        new_unit.setOutTemperature(float(pval), "C")
                    elif "outlet_pressure" in pk or "outletpressure" in pk:
                        new_unit.setOutletPressure(float(pval))
                    elif "out_pressure" in pk or "outpressure" in pk:
                        new_unit.setOutPressure(float(pval), "bara")
                    elif "pressure_drop" in pk or "pressuredrop" in pk:
                        inlet_p = float(outlet_stream.getPressure("bara"))
                        new_unit.setOutPressure(inlet_p - float(pval), "bara")
                    elif "isentropic_efficiency" in pk or "efficiency" in pk:
                        if hasattr(new_unit, "setIsentropicEfficiency"):
                            new_unit.setIsentropicEfficiency(float(pval))
                    elif "polytropic_efficiency" in pk:
                        if hasattr(new_unit, "setPolytropicEfficiency"):
                            new_unit.setPolytropicEfficiency(float(pval))
                    else:
                        setter = f"set{pkey[0].upper()}{pkey[1:]}"
                        if hasattr(new_unit, setter):
                            getattr(new_unit, setter)(pval)
                except Exception as e:
                    log.append({
                        "key": f"add_unit.{add_op.name}.param.{pkey}",
                        "status": "WARN",
                        "error": f"Failed to set param {pkey}={pval}: {e}"
                    })
            
            # Insert into sequence at the right position
            insert_pos = after_idx + 1
            new_sequence.insert(insert_pos, new_unit)
            created_units.append((add_op, new_unit, params))
            
            log.append({
                "key": f"add_unit.{add_op.name}",
                "value": f"{add_op.equipment_type} after '{add_op.insert_after}'",
                "status": "OK",
                "params": params,
            })
            
        except Exception as e:
            log.append({
                "key": f"add_unit.{add_op.name}",
                "status": "FAILED",
                "error": str(e)
            })
    
    # -----------------------------------------------------------------------
    # CRITICAL: Rebuild ALL downstream units with fresh stream connections.
    #
    # NeqSim units create internal outlet stream objects during construction.
    # Calling setInletStream() on an existing unit does NOT rewire its
    # gas/liquid outlet streams. The ONLY way to propagate a change (e.g. a
    # new cooler) downstream is to create FRESH instances of every unit after
    # the insertion point, each constructed with the previous unit's outlet.
    # -----------------------------------------------------------------------
    
    # Find the earliest insertion point (all downstream units must be recreated)
    earliest_insert = min(after_idx + 1 for after_idx, _ in insertions)
    
    # Walk the sequence from earliest insertion onward and recreate each unit
    for i in range(earliest_insert, len(new_sequence)):
        unit = new_sequence[i]
        # Get outlet stream of previous unit as the new inlet
        prev_unit = new_sequence[i - 1]
        prev_outlet = _find_outlet_stream(prev_unit)
        if prev_outlet is None:
            log.append({
                "key": f"rebuild.{i}",
                "status": "WARN",
                "error": f"Cannot find outlet of unit {i-1} for downstream reconnect"
            })
            continue
        
        # Skip units we just created (they already have correct inlet)
        is_newly_created = any(u is unit for _, u, _ in created_units)
        if is_newly_created:
            continue
        
        # Recreate this existing unit with the new inlet stream
        recreated = _recreate_unit(unit, prev_outlet)
        new_sequence[i] = recreated
    
    # Clear the process and re-add all units (new + recreated)
    unit_ops = proc.getUnitOperations()
    unit_ops.clear()
    
    for unit in new_sequence:
        proc.add(unit)

    # Re-index the model objects so introspection sees the new units/streams
    model._index_model_objects()

    return log


# ---------------------------------------------------------------------------
# Adding a sub-process (multiple units as a group)
# ---------------------------------------------------------------------------

def apply_add_process(
    model: NeqSimProcessModel,
    add_process: List[AddProcessOp],
) -> List[Dict[str, Any]]:
    """
    Add a sub-process system (a sequence of connected units) into the model.
    
    Each AddProcessOp defines a list of units to create and insert as a group
    after a specified existing unit. This is equivalent to calling apply_add_units
    for each unit in sequence, but the units are treated as a logical group.
    
    Returns a log of operations.
    """
    log = []
    
    for proc_op in add_process:
        try:
            # Convert each unit definition to an AddUnitOp, chaining them together.
            # Process units ONE AT A TIME so each newly-added unit is visible
            # to the next unit's insert_after lookup.
            prev_unit_name = proc_op.insert_after
            sub_unit_names = []
            
            for i, unit_def in enumerate(proc_op.units):
                add_unit = AddUnitOp(
                    name=unit_def.get("name", f"{proc_op.name}_{i}"),
                    equipment_type=unit_def["equipment_type"],
                    insert_after=prev_unit_name,
                    params=unit_def.get("params", {}),
                )
                sub_unit_names.append(add_unit.name)
                
                # Apply this single unit immediately so the next unit can find it
                sub_log = apply_add_units(model, [add_unit])
                for entry in sub_log:
                    entry["process_group"] = proc_op.name
                log.extend(sub_log)
                
                # Only advance prev_unit_name if the insertion succeeded
                if sub_log and sub_log[0].get("status") == "OK":
                    prev_unit_name = add_unit.name
                else:
                    # Stop chaining if a unit failed
                    break
            
            log.append({
                "key": f"add_process.{proc_op.name}",
                "status": "OK",
                "value": f"Added {len(sub_unit_names)} units: {sub_unit_names}",
            })
            
        except Exception as e:
            log.append({
                "key": f"add_process.{proc_op.name}",
                "status": "FAILED",
                "error": str(e),
            })
    
    return log


# ---------------------------------------------------------------------------
# Applying patches to a loaded model
# ---------------------------------------------------------------------------

def apply_patch_to_model(model: NeqSimProcessModel, patch: InputPatch) -> List[Dict[str, Any]]:
    """
    Apply an InputPatch directly to the process model's units/streams.
    
    The patch.changes keys are interpreted as:
      - "units.<name>.<setter_method>": calls unit.setter_method(value)
      - "streams.<name>.pressure_bara": calls stream.setPressure(value, "bara")
      - "streams.<name>.temperature_C": calls stream.setTemperature(value, "C")
      - "streams.<name>.flow_kg_hr": calls stream.setFlowRate(value, "kg/hr")
      - "<unit_name>.<method>": tries process.getUnit(unit_name).method(value)
    
    Stream/unit names may contain dots (qualified names like "unit.stream"),
    so we split from the *end* to find the property suffix.
    
    Returns a log of applied operations.
    """
    log = []
    proc = model.get_process()

    def _split_key(key: str, prefix: str):
        """Split 'prefix.objectName.property' where objectName may contain dots.

        Strategy: strip the prefix, then split from the *right* on '.' to
        get the property suffix.  If the resulting object name is not found,
        progressively move dots from the property into the name (handles
        multi-segment property names like rare edge cases).
        """
        remainder = key[len(prefix):]  # drop "streams." or "units."
        # Try splitting from the right
        parts = remainder.rsplit(".", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return remainder, ""

    for key, raw_value in patch.changes.items():
        # Resolve relative operations
        if isinstance(raw_value, dict) and "op" in raw_value:
            op = raw_value["op"]
            v = raw_value["value"]
            try:
                if key.startswith("streams."):
                    obj_name, prop = _split_key(key, "streams.")
                    s = model.get_stream(obj_name)
                    current = _get_stream_value(s, prop)
                    if op == "add":
                        value = current + v
                    elif op == "scale":
                        value = current * v
                    else:
                        raise ValueError(f"Unknown op: {op}")
                elif key.startswith("units."):
                    obj_name, prop = _split_key(key, "units.")
                    u = model.get_unit(obj_name)
                    current = _get_unit_value(u, prop)
                    if op == "add":
                        value = current + v
                    elif op == "scale":
                        value = current * v
                    else:
                        raise ValueError(f"Unknown op: {op}")
                else:
                    # Try via getUnit
                    parts = key.rsplit(".", 1)
                    if len(parts) == 2:
                        unit_name, prop = parts
                        u = proc.getUnit(unit_name)
                        current = _get_unit_value(u, prop)
                        if op == "add":
                            value = current + v
                        elif op == "scale":
                            value = current * v
                        else:
                            raise ValueError(f"Unknown op: {op}")
                    else:
                        raise KeyError(f"Cannot resolve relative patch for: {key}")
            except Exception as e:
                log.append({"key": key, "status": "FAILED", "error": str(e)})
                continue
        else:
            value = raw_value

        # Apply the value
        try:
            if key.startswith("streams."):
                obj_name, prop = _split_key(key, "streams.")
                s = model.get_stream(obj_name)
                _set_stream_value(s, prop, value)
                log.append({"key": key, "value": value, "status": "OK"})

            elif key.startswith("units."):
                obj_name, prop = _split_key(key, "units.")
                u = model.get_unit(obj_name)
                _set_unit_value(u, prop, value)
                log.append({"key": key, "value": value, "status": "OK"})

            else:
                # Try direct unit access: "<unit_tag>.<method_hint>"
                parts = key.rsplit(".", 1)
                if len(parts) == 2:
                    unit_name, prop = parts
                    try:
                        u = proc.getUnit(unit_name)
                        _set_unit_value(u, prop, value)
                        log.append({"key": key, "value": value, "status": "OK"})
                    except Exception:
                        log.append({"key": key, "status": "FAILED", "error": f"Unit '{unit_name}' not found"})
                else:
                    log.append({"key": key, "status": "FAILED", "error": "Cannot parse key"})

        except Exception as e:
            log.append({"key": key, "status": "FAILED", "error": str(e)})

    return log


def _get_stream_value(stream, prop: str) -> float:
    """Get a stream property value."""
    prop_lower = prop.lower()
    if "pressure" in prop_lower:
        if "barg" in prop_lower:
            return float(stream.getPressure("barg"))
        return float(stream.getPressure("bara"))
    elif "temperature" in prop_lower:
        return float(stream.getTemperature("C"))
    elif "flow" in prop_lower:
        if "mol" in prop_lower and "sec" in prop_lower:
            return float(stream.getFlowRate("mol/sec"))
        elif "m3" in prop_lower or "am3" in prop_lower:
            return float(stream.getFlowRate("Am3/hr"))
        elif "sm3" in prop_lower or "std" in prop_lower:
            return float(stream.getFlowRate("Sm3/day"))
        return float(stream.getFlowRate("kg/hr"))
    else:
        raise KeyError(f"Unknown stream property: {prop}")


def _set_stream_value(stream, prop: str, value: Any):
    """Set a stream property value."""
    prop_lower = prop.lower()
    if "pressure" in prop_lower:
        if "barg" in prop_lower:
            stream.setPressure(float(value), "barg")
        else:
            stream.setPressure(float(value), "bara")
    elif "temperature" in prop_lower:
        stream.setTemperature(float(value), "C")
    elif "flow" in prop_lower:
        if "mol" in prop_lower and "sec" in prop_lower:
            stream.setFlowRate(float(value), "mol/sec")
        elif "m3" in prop_lower and ("am3" in prop_lower or "actual" in prop_lower):
            stream.setFlowRate(float(value), "Am3/hr")
        elif "m3" in prop_lower and ("sm3" in prop_lower or "std" in prop_lower or "standard" in prop_lower):
            stream.setFlowRate(float(value), "Sm3/day")
        elif "kg" in prop_lower:
            stream.setFlowRate(float(value), "kg/hr")
        else:
            stream.setFlowRate(float(value), "kg/hr")  # default to kg/hr
    else:
        raise KeyError(f"Unknown stream property: {prop}")


def _get_unit_value(unit, prop: str) -> float:
    """Get a unit property value by trying common getter patterns."""
    prop_lower = prop.lower()
    
    # ---- Power / duty ----
    if "power" in prop_lower and hasattr(unit, "getPower"):
        return float(unit.getPower()) / 1000.0  # W -> kW
    if "duty" in prop_lower and hasattr(unit, "getDuty"):
        return float(unit.getDuty()) / 1000.0

    # ---- Efficiency ----
    if "polytropic" in prop_lower and "efficiency" in prop_lower and hasattr(unit, "getPolytropicEfficiency"):
        return float(unit.getPolytropicEfficiency())
    if ("isentropic" in prop_lower and "efficiency" in prop_lower) or prop_lower == "efficiency":
        if hasattr(unit, "getIsentropicEfficiency"):
            return float(unit.getIsentropicEfficiency())
    if prop_lower == "efficiency" and hasattr(unit, "getEfficiency"):
        return float(unit.getEfficiency())

    # ---- Pressure ----
    if "outletpressure" in prop_lower or "outlet_pressure" in prop_lower:
        if hasattr(unit, "getOutletPressure"):
            return float(unit.getOutletPressure())
    if "outpressure" in prop_lower or "out_pressure" in prop_lower:
        if hasattr(unit, "getOutPressure"):
            return float(unit.getOutPressure())
    if "pressure" in prop_lower and hasattr(unit, "getPressure"):
        return float(unit.getPressure())

    # ---- Temperature ----
    if "outtemperature" in prop_lower or "out_temperature" in prop_lower:
        # Try reading from outlet stream
        for m in ("getOutletStream", "getOutStream", "getGasOutStream"):
            if hasattr(unit, m):
                try:
                    return float(getattr(unit, m)().getTemperature("C"))
                except Exception:
                    pass
    if "temperature" in prop_lower:
        for m in ("getTemperature",):
            if hasattr(unit, m):
                try:
                    return float(getattr(unit, m)("C"))
                except Exception:
                    try:
                        return float(getattr(unit, m)())
                    except Exception:
                        pass

    # ---- Compressor specific ----
    if "speed" in prop_lower and hasattr(unit, "getSpeed"):
        return float(unit.getSpeed())
    if ("compressionratio" in prop_lower or "compression_ratio" in prop_lower):
        if hasattr(unit, "getCompressionRatio"):
            return float(unit.getCompressionRatio())
    if "polytropichead" in prop_lower or "polytropic_head" in prop_lower:
        if hasattr(unit, "getPolytropicHead"):
            return float(unit.getPolytropicHead())

    # ---- Valve specific ----
    if prop_lower in ("cv", "valve_cv") and hasattr(unit, "getCv"):
        return float(unit.getCv())
    if "opening" in prop_lower and hasattr(unit, "getPercentValveOpening"):
        return float(unit.getPercentValveOpening())

    # ---- Heat exchanger specific ----
    if "uavalue" in prop_lower or "ua_value" in prop_lower:
        if hasattr(unit, "getUAvalue"):
            return float(unit.getUAvalue())

    # ---- Flow rate ----
    if "flow" in prop_lower:
        for m in ("getOutletStream", "getOutStream", "getGasOutStream"):
            if hasattr(unit, m):
                try:
                    return float(getattr(unit, m)().getFlowRate("kg/hr"))
                except Exception:
                    pass

    # ---- Pipeline specific ----
    if "length" in prop_lower and hasattr(unit, "getLength"):
        return float(unit.getLength())
    if "diameter" in prop_lower and hasattr(unit, "getDiameter"):
        return float(unit.getDiameter())

    # ---- Generic getter ----
    getter = f"get{prop[0].upper()}{prop[1:]}" if prop else None
    if getter and hasattr(unit, getter):
        return float(getattr(unit, getter)())
    
    # Try snake_case to camelCase
    if "_" in prop:
        parts = prop.split("_")
        camel = parts[0] + "".join(p.capitalize() for p in parts[1:])
        getter = f"get{camel[0].upper()}{camel[1:]}"
        if hasattr(unit, getter):
            return float(getattr(unit, getter)())
    
    raise KeyError(f"Cannot get property '{prop}' on unit")


def _set_unit_value(unit, prop: str, value: Any):
    """Set a unit property value by trying common setter patterns."""
    prop_lower = prop.lower()

    # ---- Temperature setters ----
    if "outtemperature" in prop_lower or "out_temperature" in prop_lower or prop_lower == "outtemp_c":
        unit.setOutTemperature(float(value), "C")
        return
    if prop_lower in ("temperature", "temperature_c") and hasattr(unit, "setTemperature"):
        unit.setTemperature(float(value), "C")
        return

    # ---- Pressure setters ----
    if "outpressure" in prop_lower or "out_pressure" in prop_lower:
        if "barg" in prop_lower:
            unit.setOutPressure(float(value), "barg")
        else:
            unit.setOutPressure(float(value), "bara")
        return
    if "outletpressure" in prop_lower or "outlet_pressure" in prop_lower:
        if "barg" in prop_lower:
            unit.setOutletPressure(float(value), "barg")
        else:
            unit.setOutletPressure(float(value))
        return
    if prop_lower in ("pressure", "pressure_bara") and hasattr(unit, "setPressure"):
        unit.setPressure(float(value), "bara")
        return
    if prop_lower == "pressure_barg" and hasattr(unit, "setPressure"):
        unit.setPressure(float(value), "barg")
        return
    if "pressure_drop" in prop_lower or "pressuredrop" in prop_lower or prop_lower == "dp_bar":
        # Apply pressure drop by reducing outlet pressure relative to inlet
        inlet_p = None
        for m in ("getInletStream", "getInStream", "getFeed"):
            if hasattr(unit, m):
                try:
                    inlet_p = float(getattr(unit, m)().getPressure("bara"))
                    break
                except Exception:
                    pass
        if inlet_p is not None and hasattr(unit, "setOutPressure"):
            unit.setOutPressure(inlet_p - float(value), "bara")
            return
        elif inlet_p is not None and hasattr(unit, "setOutletPressure"):
            unit.setOutletPressure(inlet_p - float(value))
            return
        raise KeyError(f"Cannot apply pressure_drop: no inlet stream found")

    # ---- Efficiency setters ----
    if ("isentropic" in prop_lower and "efficiency" in prop_lower) or prop_lower == "efficiency":
        if hasattr(unit, "setIsentropicEfficiency"):
            unit.setIsentropicEfficiency(float(value))
            return
    if "polytropic" in prop_lower and "efficiency" in prop_lower:
        if hasattr(unit, "setPolytropicEfficiency"):
            unit.setPolytropicEfficiency(float(value))
            return
    if prop_lower in ("efficiency",) and hasattr(unit, "setEfficiency"):
        unit.setEfficiency(float(value))
        return

    # ---- Compressor-specific ----
    if "speed" in prop_lower and hasattr(unit, "setSpeed"):
        unit.setSpeed(float(value))
        return
    if ("compressionratio" in prop_lower or "compression_ratio" in prop_lower) and hasattr(unit, "setCompressionRatio"):
        unit.setCompressionRatio(float(value))
        return
    if "usepolytropic" in prop_lower or "use_polytropic" in prop_lower:
        if hasattr(unit, "setUsePolytropicCalc"):
            unit.setUsePolytropicCalc(bool(value))
            return
    if "powersetpoint" in prop_lower or "power_setpoint" in prop_lower or "power_kw" in prop_lower:
        if hasattr(unit, "setPower"):
            unit.setPower(float(value) * 1000.0)  # kW -> W
            return

    # ---- Valve-specific ----
    if prop_lower in ("cv", "valve_cv") and hasattr(unit, "setCv"):
        unit.setCv(float(value))
        return
    if prop_lower in ("cgv", "valve_cgv") or ("opening" in prop_lower and "percent" in prop_lower):
        if hasattr(unit, "setPercentValveOpening"):
            unit.setPercentValveOpening(float(value))
            return

    # ---- Heat exchanger specific ----
    if ("duty" in prop_lower or "energyinput" in prop_lower or "energy_input" in prop_lower):
        if hasattr(unit, "setEnergyInput"):
            unit.setEnergyInput(float(value) * 1000.0)  # kW -> W
            return
        if hasattr(unit, "setDuty"):
            unit.setDuty(float(value) * 1000.0)  # kW -> W
            return
    if "uavalue" in prop_lower or "ua_value" in prop_lower:
        if hasattr(unit, "setUAvalue"):
            unit.setUAvalue(float(value))
            return

    # ---- Separator specific ----
    if "internalmaterial" in prop_lower or "internal_material" in prop_lower:
        if hasattr(unit, "setInternalDiameter"):
            unit.setInternalDiameter(float(value))
            return

    # ---- Pump specific ----
    if "head" in prop_lower and hasattr(unit, "setHead"):
        unit.setHead(float(value))
        return

    # ---- Pipeline specific ----
    if ("length" in prop_lower or "pipe_length" in prop_lower) and hasattr(unit, "setLength"):
        unit.setLength(float(value))
        return
    if ("diameter" in prop_lower or "pipe_diameter" in prop_lower) and hasattr(unit, "setDiameter"):
        unit.setDiameter(float(value))
        return
    if "roughness" in prop_lower and hasattr(unit, "setRoughness"):
        unit.setRoughness(float(value))
        return

    # ---- Splitter specific ----
    if "splitfactor" in prop_lower or "split_factor" in prop_lower:
        if hasattr(unit, "setSplitFactors"):
            if isinstance(value, list):
                unit.setSplitFactors(value)
            else:
                unit.setSplitFactors([float(value), 1.0 - float(value)])
            return

    # ---- Column/absorber specific ----
    if "numberofstages" in prop_lower or "number_of_stages" in prop_lower:
        if hasattr(unit, "setNumberOfStages"):
            unit.setNumberOfStages(int(value))
            return

    # ---- Flow rate setters ----
    if "flowrate" in prop_lower or "flow_rate" in prop_lower:
        if "kg" in prop_lower:
            if hasattr(unit, "setFlowRate"):
                unit.setFlowRate(float(value), "kg/hr")
                return
        elif "mol" in prop_lower:
            if hasattr(unit, "setFlowRate"):
                unit.setFlowRate(float(value), "mol/sec")
                return
        elif hasattr(unit, "setFlowRate"):
            unit.setFlowRate(float(value), "kg/hr")  # default to kg/hr
            return

    # ---- Generic setter fallback ----
    # Try camelCase setter: "outletPressure" → "setOutletPressure"
    setter = f"set{prop[0].upper()}{prop[1:]}" if prop else None
    if setter and hasattr(unit, setter):
        try:
            getattr(unit, setter)(float(value))
            return
        except Exception:
            try:
                getattr(unit, setter)(value)
                return
            except Exception:
                pass

    # Try snake_case to camelCase: "outlet_pressure" -> "setOutletPressure"
    if "_" in prop:
        parts = prop.split("_")
        camel = parts[0] + "".join(p.capitalize() for p in parts[1:])
        setter = f"set{camel[0].upper()}{camel[1:]}"
        if hasattr(unit, setter):
            try:
                getattr(unit, setter)(float(value))
                return
            except Exception:
                try:
                    getattr(unit, setter)(value)
                    return
                except Exception:
                    pass

    raise KeyError(f"Cannot set property '{prop}' on unit '{unit.getName() if hasattr(unit, 'getName') else unit}'")


# ---------------------------------------------------------------------------
# Scenario runner: base + cases → comparison dataframe
# ---------------------------------------------------------------------------

def compare_results(base_result: ModelRunResult, case_result: ModelRunResult, case_name: str) -> List[Dict[str, Any]]:
    """Compare KPIs between base and a scenario case."""
    rows = []
    for k, base_kpi in base_result.kpis.items():
        if k in case_result.kpis:
            v0 = base_kpi.value
            v1 = case_result.kpis[k].value
            delta = v1 - v0 if isinstance(v0, (int, float)) and isinstance(v1, (int, float)) else None
            pct = (delta / v0 * 100) if delta is not None and v0 != 0 else None
            rows.append({
                "scenario": case_name,
                "kpi": k,
                "base": v0,
                "case": v1,
                "delta": delta,
                "delta_pct": pct,
                "unit": base_kpi.unit,
            })
    return rows


def run_scenarios(
    model: NeqSimProcessModel,
    scenarios: List[Scenario],
    timeout_ms: int = 120000,
) -> Comparison:
    """
    Run the base case + all scenarios, returning a structured comparison.
    
    For each scenario:
      1. Clone the model (re-deserialize from bytes)
      2. Apply the patch
      3. Run the simulation
      4. Compare KPIs against base
    """
    # --- Base case ---
    base_clone = model.clone()
    base_result = base_clone.run(timeout_ms=timeout_ms)
    base_scenario = Scenario(name="BASE", description="Base case (no changes)", patch=InputPatch(changes={}))
    base_sr = ScenarioResult(scenario=base_scenario, result=base_result)

    # --- Scenario cases ---
    case_results: List[ScenarioResult] = []
    all_delta_kpis: List[Dict[str, Any]] = []
    all_constraints: List[Dict[str, Any]] = []
    patch_log: List[Dict[str, Any]] = []

    for sc in scenarios:
        try:
            # If the scenario has targets, dispatch to iterative solver
            if sc.patch.targets:
                target_comparison = solve_for_target(model, sc, timeout_ms=timeout_ms)
                # Merge results (use the target solver's base if this is the first scenario)
                case_results.extend(target_comparison.cases)
                all_delta_kpis.extend(target_comparison.delta_kpis)
                all_constraints.extend(target_comparison.constraint_summary)
                patch_log.extend(target_comparison.patch_log)
                continue

            clone = model.clone()

            # Track whether we have partial failures (warn but continue)
            has_warnings = False

            # First: add any new equipment units (topology changes — abort on failure)
            if sc.patch.add_units:
                add_log = apply_add_units(clone, sc.patch.add_units)
                patch_log.extend([{"scenario": sc.name, **entry} for entry in add_log])
                add_failed = [e for e in add_log if e.get("status") == "FAILED"]
                if add_failed:
                    case_results.append(ScenarioResult(
                        scenario=sc,
                        result=ModelRunResult(kpis={}, constraints=[], raw={}),
                        success=False,
                        error=f"Add unit errors: {add_failed}"
                    ))
                    continue

            # Add sub-process systems (topology changes — abort on failure)
            if sc.patch.add_process:
                proc_log = apply_add_process(clone, sc.patch.add_process)
                patch_log.extend([{"scenario": sc.name, **entry} for entry in proc_log])
                proc_failed = [e for e in proc_log if e.get("status") == "FAILED"]
                if proc_failed:
                    case_results.append(ScenarioResult(
                        scenario=sc,
                        result=ModelRunResult(kpis={}, constraints=[], raw={}),
                        success=False,
                        error=f"Add process errors: {proc_failed}"
                    ))
                    continue

            # Add new inlet streams (topology changes — abort on failure)
            if sc.patch.add_streams:
                stream_log = apply_add_streams(clone, sc.patch.add_streams)
                patch_log.extend([{"scenario": sc.name, **entry} for entry in stream_log])
                stream_failed = [e for e in stream_log if e.get("status") == "FAILED"]
                if stream_failed:
                    case_results.append(ScenarioResult(
                        scenario=sc,
                        result=ModelRunResult(kpis={}, constraints=[], raw={}),
                        success=False,
                        error=f"Add stream errors: {stream_failed}"
                    ))
                    continue

            # Second: add chemical components (non-critical — warn but continue)
            if sc.patch.add_components:
                comp_log = apply_add_components(clone, sc.patch.add_components)
                patch_log.extend([{"scenario": sc.name, **entry} for entry in comp_log])
                comp_failed = [e for e in comp_log if e.get("status") == "FAILED"]
                if comp_failed:
                    has_warnings = True
                    # Only abort if ALL components failed
                    comp_ok = [e for e in comp_log if e.get("status") == "OK"]
                    if not comp_ok:
                        case_results.append(ScenarioResult(
                            scenario=sc,
                            result=ModelRunResult(kpis={}, constraints=[], raw={}),
                            success=False,
                            error=f"All add_component operations failed: {comp_failed}"
                        ))
                        continue

            # Then: apply parameter changes (non-critical — warn but continue)
            if sc.patch.changes:
                ops_log = apply_patch_to_model(clone, sc.patch)
                patch_log.extend([{"scenario": sc.name, **entry} for entry in ops_log])

                # Only abort if ALL patches failed
                failed = [e for e in ops_log if e.get("status") == "FAILED"]
                succeeded = [e for e in ops_log if e.get("status") == "OK"]
                if failed:
                    has_warnings = True
                    if not succeeded:
                        case_results.append(ScenarioResult(
                            scenario=sc,
                            result=ModelRunResult(kpis={}, constraints=[], raw={}),
                            success=False,
                            error=f"All patch operations failed: {failed}"
                        ))
                        continue

            result = clone.run(timeout_ms=timeout_ms)
            case_results.append(ScenarioResult(scenario=sc, result=result))

            # Compare
            deltas = compare_results(base_result, result, sc.name)
            all_delta_kpis.extend(deltas)

            # Constraints
            for c in result.constraints:
                all_constraints.append({
                    "scenario": sc.name,
                    "constraint": c.name,
                    "status": c.status,
                    "detail": c.detail,
                })

        except Exception as e:
            case_results.append(ScenarioResult(
                scenario=sc,
                result=ModelRunResult(kpis={}, constraints=[], raw={}),
                success=False,
                error=str(e)
            ))

    return Comparison(
        base=base_sr,
        cases=case_results,
        delta_kpis=all_delta_kpis,
        constraint_summary=all_constraints,
        patch_log=patch_log,
    )


def comparison_to_dataframe(comparison: Comparison) -> pd.DataFrame:
    """Convert a Comparison to a pandas DataFrame suitable for display."""
    if not comparison.delta_kpis:
        return pd.DataFrame()

    df = pd.DataFrame(comparison.delta_kpis)
    
    # Format numeric columns
    for col in ["base", "case", "delta"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: round(x, 4) if isinstance(x, float) else x)
    if "delta_pct" in df.columns:
        df["delta_pct"] = df["delta_pct"].apply(lambda x: round(x, 2) if isinstance(x, float) else x)

    return df


def results_summary_table(comparison: Comparison) -> pd.DataFrame:
    """
    Build a summary table: rows = KPIs, columns = BASE + each scenario.
    Easier to read than the delta format.
    """
    kpi_names = list(comparison.base.result.kpis.keys())
    
    data = {"KPI": kpi_names, "Unit": [], "BASE": []}
    for k in kpi_names:
        kpi = comparison.base.result.kpis[k]
        data["Unit"].append(kpi.unit)
        data["BASE"].append(round(kpi.value, 4))

    for case in comparison.cases:
        col_name = case.scenario.name
        vals = []
        for k in kpi_names:
            if k in case.result.kpis:
                vals.append(round(case.result.kpis[k].value, 4))
            else:
                vals.append(None)
        data[col_name] = vals

    return pd.DataFrame(data)
