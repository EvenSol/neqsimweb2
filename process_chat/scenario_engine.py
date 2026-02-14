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

from .patch_schema import InputPatch, Scenario, AddUnitOp
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

# Map template names to Java class paths
_EQUIPMENT_CLASSES = {
    "cooler": "neqsim.process.equipment.heatexchanger.Cooler",
    "heater": "neqsim.process.equipment.heatexchanger.Heater",
    "compressor": "neqsim.process.equipment.compressor.Compressor",
    "separator": "neqsim.process.equipment.separator.Separator",
    "valve": "neqsim.process.equipment.valve.ThrottlingValve",
    "expander": "neqsim.process.equipment.expander.Expander",
    "pump": "neqsim.process.equipment.pump.Pump",
    "heat_exchanger": "neqsim.process.equipment.heatexchanger.HeatExchanger",
    "three_phase_separator": "neqsim.process.equipment.separator.ThreePhaseSeparator",
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
    
    return new_unit


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
    
    Also supports the direct unit access pattern from the user's code:
      - "<unit_tag>.setOutTemperature(value, 'C')" style via simplified keys
    
    Returns a log of applied operations.
    """
    log = []
    proc = model.get_process()

    for key, raw_value in patch.changes.items():
        # Resolve relative operations
        if isinstance(raw_value, dict) and "op" in raw_value:
            op = raw_value["op"]
            v = raw_value["value"]
            try:
                if key.startswith("streams."):
                    parts = key.split(".", 2)
                    stream_name, prop = parts[1], parts[2]
                    s = model.get_stream(stream_name)
                    current = _get_stream_value(s, prop)
                    if op == "add":
                        value = current + v
                    elif op == "scale":
                        value = current * v
                    else:
                        raise ValueError(f"Unknown op: {op}")
                elif key.startswith("units."):
                    parts = key.split(".", 2)
                    unit_name, prop = parts[1], parts[2]
                    u = model.get_unit(unit_name)
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
                parts = key.split(".", 2)
                stream_name, prop = parts[1], parts[2]
                s = model.get_stream(stream_name)
                _set_stream_value(s, prop, value)
                log.append({"key": key, "value": value, "status": "OK"})

            elif key.startswith("units."):
                parts = key.split(".", 2)
                unit_name, prop = parts[1], parts[2]
                u = model.get_unit(unit_name)
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
        return float(stream.getPressure("bara"))
    elif "temperature" in prop_lower:
        return float(stream.getTemperature("C"))
    elif "flow" in prop_lower:
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
    elif "flow" in prop_lower and "kg" in prop_lower:
        stream.setFlowRate(float(value), "kg/hr")
    elif "flow" in prop_lower and "mol" in prop_lower:
        stream.setFlowRate(float(value), "mol/sec")
    else:
        raise KeyError(f"Unknown stream property: {prop}")


def _get_unit_value(unit, prop: str) -> float:
    """Get a unit property value by trying common getter patterns."""
    prop_lower = prop.lower()
    
    if "power" in prop_lower and hasattr(unit, "getPower"):
        return float(unit.getPower()) / 1000.0  # W -> kW
    if "duty" in prop_lower and hasattr(unit, "getDuty"):
        return float(unit.getDuty()) / 1000.0
    if "efficiency" in prop_lower and hasattr(unit, "getIsentropicEfficiency"):
        return float(unit.getIsentropicEfficiency())
    
    # Try generic getter
    getter = f"get{prop[0].upper()}{prop[1:]}" if prop else None
    if getter and hasattr(unit, getter):
        return float(getattr(unit, getter)())
    
    raise KeyError(f"Cannot get property '{prop}' on unit")


def _set_unit_value(unit, prop: str, value: Any):
    """Set a unit property value by trying common setter patterns."""
    prop_lower = prop.lower()

    # Common setters
    if "outtemperature" in prop_lower or "out_temperature" in prop_lower or prop_lower == "outtemp_c":
        unit.setOutTemperature(float(value), "C")
        return
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
    if "efficiency" in prop_lower and hasattr(unit, "setIsentropicEfficiency"):
        unit.setIsentropicEfficiency(float(value))
        return
    if "pressure" in prop_lower and hasattr(unit, "setPressure"):
        if "barg" in prop_lower:
            unit.setPressure(float(value), "barg")
        else:
            unit.setPressure(float(value), "bara")
        return
    if "temperature" in prop_lower and hasattr(unit, "setTemperature"):
        unit.setTemperature(float(value), "C")
        return

    # Try generic setter
    setter = f"set{prop[0].upper()}{prop[1:]}" if prop else None
    if setter and hasattr(unit, setter):
        getattr(unit, setter)(value)
        return

    raise KeyError(f"Cannot set property '{prop}' on unit")


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
            clone = model.clone()

            # First: add any new equipment units
            if sc.patch.add_units:
                add_log = apply_add_units(clone, sc.patch.add_units)
                patch_log.extend([{"scenario": sc.name, **entry} for entry in add_log])
                # Check for add failures
                add_failed = [e for e in add_log if e.get("status") == "FAILED"]
                if add_failed:
                    case_results.append(ScenarioResult(
                        scenario=sc,
                        result=ModelRunResult(kpis={}, constraints=[], raw={}),
                        success=False,
                        error=f"Add unit errors: {add_failed}"
                    ))
                    continue

            # Then: apply parameter changes
            ops_log = apply_patch_to_model(clone, sc.patch)
            patch_log.extend([{"scenario": sc.name, **entry} for entry in ops_log])

            # Check for failed patches
            failed = [e for e in ops_log if e.get("status") == "FAILED"]
            if failed:
                case_results.append(ScenarioResult(
                    scenario=sc,
                    result=ModelRunResult(kpis={}, constraints=[], raw={}),
                    success=False,
                    error=f"Patch errors: {failed}"
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
