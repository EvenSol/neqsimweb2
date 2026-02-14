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


def apply_add_units(model: NeqSimProcessModel, add_units: List[AddUnitOp]) -> List[Dict[str, Any]]:
    """
    Insert new equipment into the process topology.
    
    For each AddUnitOp:
      1. Find the 'insert_after' unit and its position
      2. Get its outlet stream
      3. Create the new equipment with that stream as inlet
      4. Configure the new equipment with params
      5. Find the downstream unit and reconnect its inlet
      6. Insert the new unit at the right position
      7. Re-index the model objects
    
    Returns a log of operations.
    """
    from neqsim import jneqsim

    log = []
    proc = model.get_process()

    for add_op in add_units:
        try:
            # 1. Find the 'insert_after' unit
            after_unit = proc.getUnit(add_op.insert_after)
            if after_unit is None:
                log.append({
                    "key": f"add_unit.{add_op.name}",
                    "status": "FAILED",
                    "error": f"Unit '{add_op.insert_after}' not found in process"
                })
                continue

            after_idx = int(proc.getUnitNumber(add_op.insert_after))

            # 2. Get the outlet stream of the 'insert_after' unit
            outlet_stream = _find_outlet_stream(after_unit)
            if outlet_stream is None:
                log.append({
                    "key": f"add_unit.{add_op.name}",
                    "status": "FAILED",
                    "error": f"Cannot find outlet stream for '{add_op.insert_after}'"
                })
                continue

            # 3. Create the new equipment
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

            # Import the Java class and create instance with inlet stream
            parts = java_class_path.rsplit(".", 1)
            pkg = parts[0]
            cls_name = parts[1]

            # Navigate to the Java package
            java_pkg = jneqsim
            for p in pkg.split(".")[1:]:  # skip 'neqsim' since jneqsim is already neqsim
                java_pkg = getattr(java_pkg, p)
            JavaClass = getattr(java_pkg, cls_name)

            new_unit = JavaClass(add_op.name, outlet_stream)

            # 4. Configure the new equipment with params
            params = dict(add_op.params)  # copy
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
                        # For coolers/heaters: set outlet pressure = inlet - drop
                        inlet_p = float(outlet_stream.getPressure("bara"))
                        new_unit.setOutPressure(inlet_p - float(pval), "bara")
                    elif "isentropic_efficiency" in pk or "efficiency" in pk:
                        if hasattr(new_unit, "setIsentropicEfficiency"):
                            new_unit.setIsentropicEfficiency(float(pval))
                    elif "polytropic_efficiency" in pk:
                        if hasattr(new_unit, "setPolytropicEfficiency"):
                            new_unit.setPolytropicEfficiency(float(pval))
                    else:
                        # Try generic setter
                        setter = f"set{pkey[0].upper()}{pkey[1:]}"
                        if hasattr(new_unit, setter):
                            getattr(new_unit, setter)(pval)
                except Exception as e:
                    log.append({
                        "key": f"add_unit.{add_op.name}.param.{pkey}",
                        "status": "WARN",
                        "error": f"Failed to set param {pkey}={pval}: {e}"
                    })

            # 5. Find the downstream unit and reconnect its inlet
            units_list = list(proc.getUnitOperations())
            insert_pos = after_idx + 1

            if insert_pos < len(units_list):
                downstream_unit = units_list[insert_pos]
                new_outlet = _find_outlet_stream(new_unit)
                if new_outlet is not None:
                    reconnected = _set_inlet_stream(downstream_unit, new_outlet)
                    if not reconnected:
                        log.append({
                            "key": f"add_unit.{add_op.name}",
                            "status": "WARN",
                            "error": f"Could not reconnect downstream unit '{downstream_unit.getName()}' inlet"
                        })

            # 6. Insert the new unit at the right position
            proc.add(insert_pos, new_unit)

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

    # 7. Re-index the model objects so introspection sees the new units/streams
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
