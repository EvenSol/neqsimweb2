"""Shared, UI-independent projection of current Studio engineering results.

The Process Flowsheet Studio remains the owner of model construction and native
NeqSim execution.  This module only validates the session handoff and presents
the already-solved model through a small, stable application-service boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, MutableMapping


ACTIVE_CASE_STATE_KEY = "neqsim_studio_case_context"
FLOWSHEET_CASE_STATE_KEY = "flowsheet_studio_case"
FLOWSHEET_RESULT_STATE_KEY = "flowsheet_studio_result"
FLOWSHEET_CASE_HISTORY_STATE_KEY = "flowsheet_studio_case_history"
CURRENT_RESULT_STATUSES = frozenset({"solved", "warning"})


class StudioResultsUnavailable(RuntimeError):
    """Raised when no exact, current solved result can be published."""


@dataclass(frozen=True)
class StudioResultContext:
    """References to one validated current result without copying native objects."""

    active_case: Mapping[str, Any]
    spec: Mapping[str, Any]
    model: Any
    result: Any
    run_record: Mapping[str, Any]
    warnings: tuple[str, ...]
    signature: str


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_current_result_context(
    session_state: Mapping[str, Any],
) -> StudioResultContext:
    """Return the exact active solved result or fail closed with a useful reason."""

    active_case = session_state.get(ACTIVE_CASE_STATE_KEY)
    if not isinstance(active_case, Mapping):
        raise StudioResultsUnavailable(
            "No active Studio case is available. Open or create a process case first."
        )

    status = str(active_case.get("status", "")).strip().lower()
    if status not in CURRENT_RESULT_STATUSES:
        label = status.replace("-", " ") or "not solved"
        raise StudioResultsUnavailable(
            f"The active case is {label}. Run the current inputs before reviewing results."
        )

    state = session_state.get(FLOWSHEET_CASE_STATE_KEY)
    if not session_state.get(FLOWSHEET_RESULT_STATE_KEY) or not isinstance(
        state, Mapping
    ):
        raise StudioResultsUnavailable(
            "The active case has no solved result in this session. Run it again to "
            "restore native NeqSim evidence."
        )

    runtime = _mapping(active_case.get("runtime"))
    solved_signature = str(runtime.get("solved_signature") or "").strip()
    stored_signature = str(state.get("signature") or "").strip()
    if not solved_signature or solved_signature != stored_signature:
        raise StudioResultsUnavailable(
            "The stored result does not match the active case inputs. Run the current "
            "case before reviewing results."
        )
    if runtime.get("model_available") is not True:
        raise StudioResultsUnavailable(
            "The solved case metadata is present, but its native model is not available "
            "in this session. Run the case again."
        )

    active_spec = active_case.get("case_spec")
    stored_spec = state.get("spec")
    if not isinstance(active_spec, Mapping) or active_spec != stored_spec:
        raise StudioResultsUnavailable(
            "The portable case and stored result are out of sync. Run the active case "
            "before reviewing results."
        )

    model = state.get("model")
    result = state.get("result")
    if model is None or result is None:
        raise StudioResultsUnavailable(
            "The current native model or result is unavailable. Run the active case again."
        )

    warnings = tuple(
        str(item).strip()
        for item in state.get("warnings", ())
        if str(item).strip()
    )
    return StudioResultContext(
        active_case=active_case,
        spec=stored_spec,
        model=model,
        result=result,
        run_record=_mapping(state.get("run_record")),
        warnings=warnings,
        signature=stored_signature,
    )


def _kpi(result: Any, name: str) -> dict[str, Any] | None:
    kpis = getattr(result, "kpis", {})
    if not isinstance(kpis, Mapping):
        return None
    item = kpis.get(name)
    if item is None:
        return None
    value = _finite_number(getattr(item, "value", None))
    if value is None:
        return None
    return {
        "name": str(getattr(item, "name", name)),
        "value": value,
        "unit": str(getattr(item, "unit", "")),
    }


def build_result_summary(context: StudioResultContext) -> dict[str, Any]:
    """Build professional status and KPI evidence from shared diagnostics."""

    from process_chat.solver_diagnostics import (
        aggregate_convergence,
        aggregate_energy_balance,
        aggregate_unit_balances,
        aggregate_validation_status,
        solved_feed_flow_kg_hr,
    )

    result = context.result
    constraints = getattr(result, "constraints", ())
    validation_status = aggregate_validation_status(
        getattr(item, "status", "UNKNOWN") for item in constraints
    )
    convergence = aggregate_convergence(result)
    energy_balance = aggregate_energy_balance(result)
    unit_balances = aggregate_unit_balances(result)

    power = _kpi(result, "total_power_kW")
    duty = _kpi(result, "total_duty_kW")
    mass = _kpi(result, "mass_balance_pct")
    component = _kpi(result, "component_balance_max_pct")
    fluid = _mapping(context.spec.get("fluid"))
    fallback_flow = _finite_number(fluid.get("total_flow")) or 0.0
    feed_flow = solved_feed_flow_kg_hr(result, fallback_flow)
    specific_energy = None
    if power and feed_flow > 0.0:
        specific_energy = {
            "name": "specific_compression_energy",
            "value": power["value"] / (feed_flow / 1000.0),
            "unit": "kWh/t",
        }

    attention_required = validation_status == "VIOLATION" or (
        convergence.get("applicable") is True
        and convergence.get("converged") is False
    )
    warnings_present = bool(context.warnings) or validation_status in {
        "WARN",
        "UNKNOWN",
    }
    if attention_required:
        engineering_state = "Attention required"
    elif warnings_present:
        engineering_state = "Solved with warnings"
    else:
        engineering_state = "Solved"

    return {
        "engineering_state": engineering_state,
        "validation_status": validation_status,
        "metrics": {
            "total_power": power,
            "total_duty": duty,
            "specific_energy": specific_energy,
            "mass_imbalance": mass,
            "component_imbalance": component,
        },
        "convergence": convergence,
        "energy_balance": energy_balance,
        "unit_balances": unit_balances,
        "stream_count": len(stream_rows(context)),
        "equipment_count": len(equipment_rows(context)),
    }


def stream_rows(context: StudioResultContext) -> list[dict[str, Any]]:
    """Return normalized, explicit-unit rows from the existing model adapter."""

    rows = []
    for stream in context.model.list_streams():
        rows.append(
            {
                "Stream": stream.name,
                "Temperature [degC]": stream.temperature_C,
                "Pressure [bara]": stream.pressure_bara,
                "Mass flow [kg/hr]": stream.flow_rate_kg_hr,
                "Molar flow [mol/s]": stream.flow_rate_mol_sec,
                "Process system": getattr(stream, "process_system", ""),
                "Owner": getattr(stream, "owner_name", ""),
            }
        )
    return rows


def equipment_rows(context: StudioResultContext) -> list[dict[str, Any]]:
    """Return solved equipment rows without changing model semantics."""

    rows = []
    for unit in context.model.list_units():
        row = {
            "Equipment": unit.name,
            "Type": unit.unit_type,
            "Process system": unit.process_system,
        }
        properties = getattr(unit, "properties", {})
        if isinstance(properties, Mapping):
            row.update(properties)
        rows.append(row)
    return rows


def constraint_rows(context: StudioResultContext) -> list[dict[str, str]]:
    """Return the existing solved engineering constraint evidence."""

    rows = [
        {
            "name": str(getattr(item, "name", "simulation")),
            "status": str(getattr(item, "status", "UNKNOWN")),
            "detail": str(getattr(item, "detail", "No detail reported.")),
        }
        for item in getattr(context.result, "constraints", ())
    ]
    return rows or [
        {
            "name": "simulation",
            "status": "OK",
            "detail": "The process completed without reported constraint warnings.",
        }
    ]


_DESIGN_DEFINITIONS = (
    ("Pump flow", "designFlowCapacity_m3_per_hr", "flowMargin_m3_per_hr", "flowUtilization_pct", "m3/hr", "pump_design"),
    ("Pump head", "designHeadCapacity_m", "headMargin_m", "headUtilization_pct", "m", "pump_design"),
    ("Pump motor", "motorRating_kW", "motorMargin_kW", "motorUtilization_pct", "kW", "pump_design"),
    ("Heat-exchanger duty", "designDutyCapacity_kW", "dutyMargin_kW", "dutyUtilization_pct", "kW", "heat_exchanger_design"),
    ("Heat-exchanger UA", "designUACapacity_W_K", "uaMargin_W_K", "uaUtilization_pct", "W/K", "heat_exchanger_design"),
    ("Valve Cv", "designCvCapacity_US", "cvMargin_US", "cvUtilization_pct", "US Cv", "valve_design"),
    ("Pipeline pressure drop", "designPressureDropCapacity_bar", "pressureDropMargin_bar", "pressureDropUtilization_pct", "bar", "pipeline_design"),
    ("Pipeline velocity", "designVelocityCapacity_m_s", "velocityMargin_m_s", "velocityUtilization_pct", "m/s", "pipeline_design"),
)


def equipment_design_rows(context: StudioResultContext) -> list[dict[str, Any]]:
    """Project existing operating/design properties into reviewable rows."""

    lookup = {
        row["name"].casefold(): (row["status"], row["detail"])
        for row in constraint_rows(context)
    }
    rows: list[dict[str, Any]] = []
    for equipment in equipment_rows(context):
        equipment_name = str(equipment["Equipment"])
        equipment_type = str(equipment.get("Type", ""))
        critical_segment = _finite_number(
            equipment.get("velocityCriticalSegment_index")
        )
        critical_length = _finite_number(
            equipment.get("velocityCriticalLength_m")
        )
        for label, capacity_key, margin_key, utilization_key, unit, prefix in _DESIGN_DEFINITIONS:
            capacity = _finite_number(equipment.get(capacity_key))
            if capacity is None:
                continue
            margin = _finite_number(equipment.get(margin_key))
            utilization = _finite_number(equipment.get(utilization_key))
            status, detail = lookup.get(
                f"{prefix}.{equipment_name}".casefold(),
                ("UNKNOWN", "No matching solved design constraint was reported."),
            )
            rows.append(
                {
                    "Equipment": equipment_name,
                    "Type": equipment_type,
                    "Design check": label,
                    "Operating value": capacity - margin if margin is not None else None,
                    "Design capacity": capacity,
                    "Margin": margin,
                    "Utilization [%]": utilization,
                    "Unit": unit,
                    "Status": status,
                    "Constraint detail": detail,
                    "Critical segment [-]": critical_segment if prefix == "pipeline_design" else None,
                    "Critical length [m]": critical_length if prefix == "pipeline_design" else None,
                }
            )
    return rows


def validation_tables(context: StudioResultContext) -> dict[str, list[dict[str, Any]]]:
    """Return shared diagnostic rows for professional validation tables."""

    from process_chat.solver_diagnostics import (
        component_balance_rows,
        convergence_rows,
        energy_transfer_rows,
        material_boundary_rows,
        unit_balance_rows,
    )

    return {
        "material_boundaries": material_boundary_rows(context.result),
        "component_balances": component_balance_rows(context.result),
        "energy_transfers": energy_transfer_rows(context.result),
        "convergence": convergence_rows(context.result),
        "unit_balances": unit_balance_rows(context.result),
    }


def case_history_rows(session_state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return display-safe scalar evidence from the existing case history."""

    history = session_state.get(FLOWSHEET_CASE_HISTORY_STATE_KEY, ())
    if not isinstance(history, (list, tuple)):
        return []
    rows = []
    for record in history:
        if not isinstance(record, Mapping):
            continue
        row = {
            str(key): value
            for key, value in record.items()
            if not str(key).startswith("_")
            and (value is None or isinstance(value, (str, int, float, bool)))
        }
        if row:
            rows.append(row)
    return rows


def remember_result_destination(
    session_state: MutableMapping[str, Any], destination: str
) -> None:
    """Record why the shared results workspace was opened."""

    session_state["neqsim_studio_results_destination"] = str(destination)
