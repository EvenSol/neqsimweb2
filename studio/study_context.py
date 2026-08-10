"""Bounded Studio projection of deterministic Process Chat study evidence."""

from __future__ import annotations

import math
from typing import Any, Mapping

from studio.results_context import (
    StudioResultContext,
    StudioResultsUnavailable,
    load_current_result_context,
)


CHAT_SESSION_STATE_KEY = "chat_session"
CHAT_MESSAGES_STATE_KEY = "chat_messages"


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def engineering_unit(name: Any) -> str:
    """Return an explicit unit from established NeqSim KPI/variable names."""

    normalized = str(name).strip().lower()
    suffixes = (
        ("_kg_per_hr", "kg/hr"),
        ("_kg_hr", "kg/hr"),
        ("_mol_sec", "mol/s"),
        ("_m3_per_hr", "m3/hr"),
        ("_m3_hr", "m3/hr"),
        ("_kwh_t", "kWh/t"),
        ("_w_k", "W/K"),
        ("_bara", "bara"),
        ("_barg", "barg"),
        ("_bar", "bar"),
        ("_kw", "kW"),
        ("_mw", "MW"),
        ("_degc", "degC"),
        ("_temperature_c", "degC"),
        ("_c", "degC"),
        ("_pct", "%"),
        ("_percent", "%"),
        ("_m_s", "m/s"),
        ("_m", "m"),
    )
    for suffix, unit in suffixes:
        if normalized.endswith(suffix):
            return unit
    if "imbalance" in normalized or "utilization" in normalized:
        return "%"
    if "efficiency" in normalized or normalized.endswith("_fraction"):
        return "fraction"
    return "not reported"


def _getter(session: Any, name: str) -> Any:
    getter = getattr(session, name, None)
    return getter() if callable(getter) else None


def _latest_message_result(
    session_state: Mapping[str, Any],
    key: str,
    model: Any,
    solved_signature: str,
) -> Any:
    """Return the latest retained chat attachment for the current model.

    ProcessChatSession's ``get_last_*`` values intentionally describe only the
    most recent chat turn.  The page retains completed result objects on
    assistant messages, so Studio can recover the latest applicable study
    after an ordinary follow-up or after another study type has run.
    """

    messages = session_state.get(CHAT_MESSAGES_STATE_KEY, ())
    if not isinstance(messages, (list, tuple)):
        return None
    for message in reversed(messages):
        if not isinstance(message, Mapping):
            continue
        result = message.get(key)
        if result is None:
            continue
        tagged_model = message.get("_study_model")
        tagged_signature = message.get("_study_signature")
        model_matches = tagged_model is None or tagged_model is model
        signature_matches = (
            tagged_signature is None or tagged_signature == solved_signature
        )
        if model_matches and signature_matches:
            return result
    return None


def _sensitivity_evidence(result: Any) -> dict[str, Any] | None:
    if result is None:
        return None
    response_kpis = [str(item) for item in getattr(result, "response_kpis", ())]
    point_rows: list[dict[str, Any]] = []
    for index, point in enumerate(getattr(result, "sweep_points", ()), start=1):
        inputs = getattr(point, "input_values", {})
        outputs = getattr(point, "output_values", {})
        if not isinstance(inputs, Mapping):
            inputs = {}
        if not isinstance(outputs, Mapping):
            outputs = {}
        input_text = "; ".join(
            f"{key}={value:g} {engineering_unit(key)}"
            for key, raw_value in inputs.items()
            if (value := _finite_number(raw_value)) is not None
        )
        feasible = bool(getattr(point, "feasible", False))
        error = str(getattr(point, "error", "") or "")
        if outputs:
            for kpi_name, raw_value in outputs.items():
                point_rows.append(
                    {
                        "Point": index,
                        "Feasible": feasible,
                        "Inputs": input_text,
                        "Response KPI": str(kpi_name),
                        "Value": _finite_number(raw_value),
                        "Unit": engineering_unit(kpi_name),
                        "Error": error,
                    }
                )
        else:
            point_rows.append(
                {
                    "Point": index,
                    "Feasible": feasible,
                    "Inputs": input_text,
                    "Response KPI": None,
                    "Value": None,
                    "Unit": "not reported",
                    "Error": error or "No response value was reported.",
                }
            )

    response_name = response_kpis[0] if response_kpis else "response"
    tornado_rows = []
    for bar in getattr(result, "tornado_bars", ()):
        variable = str(getattr(bar, "variable", ""))
        tornado_rows.append(
            {
                "Variable": variable,
                "Input unit": engineering_unit(variable),
                "Low input": _finite_number(getattr(bar, "low_value", None)),
                "High input": _finite_number(getattr(bar, "high_value", None)),
                "Response KPI": response_name,
                "Response unit": engineering_unit(response_name),
                "Base response": _finite_number(getattr(bar, "kpi_base", None)),
                "Response at low": _finite_number(getattr(bar, "kpi_at_low", None)),
                "Response at high": _finite_number(getattr(bar, "kpi_at_high", None)),
                "Low delta": _finite_number(getattr(bar, "delta_low", None)),
                "High delta": _finite_number(getattr(bar, "delta_high", None)),
            }
        )
    return {
        "analysis_type": str(getattr(result, "analysis_type", "unknown")),
        "method": str(getattr(result, "method", "not reported")),
        "n_points": int(getattr(result, "n_points", len(point_rows)) or 0),
        "response_kpis": response_kpis,
        "message": str(getattr(result, "message", "") or ""),
        "point_rows": point_rows,
        "tornado_rows": tornado_rows,
    }


def _optimization_evidence(result: Any) -> dict[str, Any] | None:
    if result is None:
        return None
    utilization_rows = []
    for item in getattr(result, "utilization_breakdown", ()):
        utilization = _finite_number(getattr(item, "utilization", None))
        utilization_rows.append(
            {
                "Equipment": str(getattr(item, "name", "")),
                "Type": str(getattr(item, "equipment_type", "")),
                "Utilization [%]": (
                    utilization * 100.0 if utilization is not None else None
                ),
                "Constraint": str(getattr(item, "constraint_name", "")),
                "Detail": str(getattr(item, "detail", "")),
            }
        )
    iteration_rows = []
    for item in getattr(result, "iterations", ()):
        utilization = _finite_number(getattr(item, "max_utilization", None))
        iteration_rows.append(
            {
                "Iteration": getattr(item, "iteration", None),
                "Feed flow [kg/hr]": _finite_number(
                    getattr(item, "flow_rate_kg_hr", None)
                ),
                "Feasible": bool(getattr(item, "feasible", False)),
                "Maximum utilization [%]": (
                    utilization * 100.0 if utilization is not None else None
                ),
                "Bottleneck": str(getattr(item, "bottleneck", "")),
                "Detail": str(getattr(item, "detail", "")),
            }
        )
    kpi_rows = []
    kpis = getattr(result, "kpis_at_optimum", {})
    if isinstance(kpis, Mapping):
        for name, item in kpis.items():
            value = _finite_number(getattr(item, "value", None))
            if value is None:
                continue
            kpi_rows.append(
                {
                    "KPI": str(getattr(item, "name", name)),
                    "Value": value,
                    "Unit": str(getattr(item, "unit", "")) or engineering_unit(name),
                }
            )
    bottleneck_utilization = _finite_number(
        getattr(result, "bottleneck_utilization", None)
    )
    return {
        "converged": bool(getattr(result, "converged", False)),
        "search_algorithm": str(
            getattr(result, "search_algorithm", "not reported")
        ),
        "original_flow_kg_hr": _finite_number(
            getattr(result, "original_flow_kg_hr", None)
        ),
        "optimal_flow_kg_hr": _finite_number(
            getattr(result, "optimal_flow_kg_hr", None)
        ),
        "max_increase_pct": _finite_number(
            getattr(result, "max_increase_pct", None)
        ),
        "bottleneck_equipment": str(
            getattr(result, "bottleneck_equipment", "")
        ),
        "bottleneck_type": str(getattr(result, "bottleneck_type", "")),
        "bottleneck_utilization_pct": (
            bottleneck_utilization * 100.0
            if bottleneck_utilization is not None
            else None
        ),
        "message": str(getattr(result, "message", "") or ""),
        "utilization_rows": utilization_rows,
        "iteration_rows": iteration_rows,
        "kpi_rows": kpi_rows,
    }


def _scenario_evidence(comparison: Any) -> dict[str, Any] | None:
    """Project an inherited scenario comparison without rerunning it."""

    if comparison is None:
        return None
    case_rows: list[dict[str, Any]] = []
    kpi_rows: list[dict[str, Any]] = []
    scenario_results = [getattr(comparison, "base", None)]
    scenario_results.extend(getattr(comparison, "cases", ()) or ())
    for index, scenario_result in enumerate(scenario_results):
        if scenario_result is None:
            continue
        scenario = getattr(scenario_result, "scenario", None)
        scenario_name = str(
            getattr(scenario, "name", "") or ("BASE" if index == 0 else f"Case {index}")
        )
        success = bool(getattr(scenario_result, "success", False))
        error = str(getattr(scenario_result, "error", "") or "")
        run_result = getattr(scenario_result, "result", None)
        kpis = getattr(run_result, "kpis", {})
        if not isinstance(kpis, Mapping):
            kpis = {}
        case_rows.append(
            {
                "Scenario": scenario_name,
                "Status": "Solved" if success else "Failed",
                "KPI count": len(kpis),
                "Error": error,
            }
        )
        for key, item in kpis.items():
            value = _finite_number(getattr(item, "value", None))
            if value is None:
                continue
            name = str(getattr(item, "name", key))
            kpi_rows.append(
                {
                    "Scenario": scenario_name,
                    "KPI": name,
                    "Value": value,
                    "Unit": str(getattr(item, "unit", "")) or engineering_unit(name),
                }
            )

    delta_rows = []
    for item in getattr(comparison, "delta_kpis", ()) or ():
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("kpi", ""))
        delta_rows.append(
            {
                "Scenario": str(item.get("scenario", item.get("case_name", ""))),
                "KPI": name,
                "Base": _finite_number(item.get("base")),
                "Case": _finite_number(item.get("case")),
                "Delta": _finite_number(item.get("delta")),
                "Delta [%]": _finite_number(item.get("delta_pct")),
                "Unit": str(item.get("unit", "")) or engineering_unit(name),
            }
        )

    constraint_rows = []
    for item in getattr(comparison, "constraint_summary", ()) or ():
        if not isinstance(item, Mapping):
            continue
        constraint_rows.append(
            {
                "Scenario": str(item.get("scenario", "")),
                "Constraint": str(item.get("constraint", item.get("name", ""))),
                "Status": str(item.get("status", "UNKNOWN")),
                "Detail": str(item.get("detail", "")),
            }
        )

    patch_rows = []
    for item in getattr(comparison, "patch_log", ()) or ():
        if not isinstance(item, Mapping):
            continue
        patch_rows.append(
            {
                "Scenario": str(item.get("scenario", "")),
                "Change": str(item.get("key", "")),
                "Value": str(item.get("value", "")),
                "Status": str(item.get("status", "UNKNOWN")),
                "Detail": str(item.get("error", item.get("detail", ""))),
            }
        )
    return {
        "case_rows": case_rows,
        "kpi_rows": kpi_rows,
        "delta_rows": delta_rows,
        "constraint_rows": constraint_rows,
        "patch_rows": patch_rows,
    }


def _emissions_evidence(result: Any) -> dict[str, Any] | None:
    """Project deterministic emissions evidence with explicit engineering units."""

    if result is None:
        return None
    metrics = [
        ("CO2", "total_co2_kg_hr", "kg/hr"),
        ("CO2e", "total_co2e_kg_hr", "kg/hr"),
        ("Annual CO2", "total_co2_tonnes_yr", "tonne/yr"),
        ("Annual CO2e", "total_co2e_tonnes_yr", "tonne/yr"),
        ("CH4", "total_ch4_kg_hr", "kg/hr"),
        ("NOx", "total_nox_kg_hr", "kg/hr"),
        ("Emission intensity", "emission_intensity_kg_per_tonne", "kg CO2e/tonne"),
        ("Product rate", "product_rate_kg_hr", "kg/hr"),
    ]
    metric_rows = [
        {"Metric": label, "Value": value, "Unit": unit}
        for label, attribute, unit in metrics
        if (value := _finite_number(getattr(result, attribute, None))) is not None
    ]
    source_rows = []
    for source in getattr(result, "sources", ()) or ():
        source_rows.append(
            {
                "Source": str(getattr(source, "name", "")),
                "Type": str(getattr(source, "source_type", "")),
                "CO2 [kg/hr]": _finite_number(getattr(source, "co2_kg_hr", None)),
                "CO2e [kg/hr]": _finite_number(getattr(source, "co2e_kg_hr", None)),
                "CH4 [kg/hr]": _finite_number(getattr(source, "ch4_kg_hr", None)),
                "NOx [kg/hr]": _finite_number(getattr(source, "nox_kg_hr", None)),
                "Fuel [kg/hr]": _finite_number(getattr(source, "fuel_rate_kg_hr", None)),
                "Detail": str(getattr(source, "detail", "")),
            }
        )
    return {
        "method": str(getattr(result, "method", "not reported")),
        "message": str(getattr(result, "message", "") or ""),
        "metric_rows": metric_rows,
        "source_rows": source_rows,
    }


def _energy_audit_evidence(result: Any) -> dict[str, Any] | None:
    """Project the inherited energy audit and its screening recommendations."""

    if result is None:
        return None
    metrics = [
        ("Power", "total_power_kW", "kW"),
        ("Cooling", "total_cooling_kW", "kW"),
        ("Heating", "total_heating_kW", "kW"),
        ("Net energy", "net_energy_kW", "kW"),
        ("Specific energy", "specific_energy_kWh_per_tonne", "kWh/tonne"),
        ("Product rate", "product_rate_kg_hr", "kg/hr"),
        ("Fuel gas", "fuel_gas_rate_kg_hr", "kg/hr"),
        ("Fuel gas cost", "fuel_gas_cost_usd_hr", "USD/hr"),
    ]
    metric_rows = [
        {"Metric": label, "Value": value, "Unit": unit}
        for label, attribute, unit in metrics
        if (value := _finite_number(getattr(result, attribute, None))) is not None
    ]
    consumer_rows = [
        {
            "Equipment": str(getattr(item, "name", "")),
            "Type": str(getattr(item, "equipment_type", "")),
            "Service": str(getattr(item, "energy_type", "")),
            "Consumption [kW]": _finite_number(getattr(item, "consumption_kW", None)),
            "Share [%]": _finite_number(getattr(item, "share_pct", None)),
            "Detail": str(getattr(item, "detail", "")),
        }
        for item in (getattr(result, "consumers", ()) or ())
    ]
    benchmark_rows = [
        {
            "Metric": str(getattr(item, "metric", "")),
            "Actual": _finite_number(getattr(item, "actual_value", None)),
            "Benchmark": _finite_number(getattr(item, "benchmark_value", None)),
            "Unit": str(getattr(item, "unit", "")) or "not reported",
            "Status": str(getattr(item, "status", "UNKNOWN")),
            "Detail": str(getattr(item, "detail", "")),
        }
        for item in (getattr(result, "benchmarks", ()) or ())
    ]
    suggestion_rows = [
        {
            "Equipment": str(getattr(item, "equipment", "")),
            "Suggestion": str(getattr(item, "suggestion", "")),
            "Potential saving [kW]": _finite_number(
                getattr(item, "potential_saving_kW", None)
            ),
            "Potential saving [%]": _finite_number(
                getattr(item, "potential_saving_pct", None)
            ),
            "Detail": str(getattr(item, "detail", "")),
        }
        for item in (getattr(result, "suggestions", ()) or ())
    ]
    return {
        "method": str(getattr(result, "method", "not reported")),
        "message": str(getattr(result, "message", "") or ""),
        "product_stream": str(getattr(result, "product_stream", "") or ""),
        "metric_rows": metric_rows,
        "consumer_rows": consumer_rows,
        "benchmark_rows": benchmark_rows,
        "suggestion_rows": suggestion_rows,
    }


def current_study_evidence(session_state: Mapping[str, Any]) -> dict[str, Any]:
    """Return studies tied to the exact active native model and solved signature."""

    try:
        context: StudioResultContext = load_current_result_context(session_state)
    except StudioResultsUnavailable as exc:
        return {"available": False, "reason": str(exc)}

    chat_session = session_state.get(CHAT_SESSION_STATE_KEY)
    if chat_session is None:
        return {
            "available": True,
            "reason": "No Process Chat studies have been run in this session.",
            "provenance": {
                "case_id": context.active_case.get("case_id"),
                "solved_signature": context.signature,
                "source": "Process Chat deterministic NeqSim studies",
            },
            "sensitivity": None,
            "optimization": None,
            "scenario_comparison": None,
            "emissions": None,
            "energy_audit": None,
        }
    if getattr(chat_session, "model", None) is not context.model:
        return {
            "available": False,
            "reason": (
                "Process Chat study evidence belongs to a different runtime model. "
                "Reopen Process Chat from the active solved case and rerun the study."
            ),
        }
    chat_case_context = getattr(chat_session, "_studio_case_context", None)
    chat_runtime = (
        chat_case_context.get("runtime")
        if isinstance(chat_case_context, Mapping)
        else None
    )
    chat_signature = (
        chat_runtime.get("solved_signature")
        if isinstance(chat_runtime, Mapping)
        else None
    )
    if chat_signature != context.signature:
        return {
            "available": False,
            "reason": (
                "Process Chat evidence belongs to a different solved signature. "
                "Reopen Process Chat from the active solved case and rerun the study."
            ),
        }

    sensitivity_result = _getter(chat_session, "get_last_sensitivity")
    if sensitivity_result is None:
        sensitivity_result = _latest_message_result(
            session_state,
            "sensitivity",
            context.model,
            context.signature,
        )
    optimization_result = _getter(chat_session, "get_last_optimization")
    if optimization_result is None:
        optimization_result = _latest_message_result(
            session_state,
            "optimization",
            context.model,
            context.signature,
        )
    comparison_result = _getter(chat_session, "get_last_comparison")
    if comparison_result is None:
        comparison_result = _latest_message_result(
            session_state,
            "comparison",
            context.model,
            context.signature,
        )
    emissions_result = _getter(chat_session, "get_last_emissions")
    if emissions_result is None:
        emissions_result = _latest_message_result(
            session_state,
            "emissions",
            context.model,
            context.signature,
        )
    energy_audit_result = _getter(chat_session, "get_last_energy_audit")
    if energy_audit_result is None:
        energy_audit_result = _latest_message_result(
            session_state,
            "energy_audit",
            context.model,
            context.signature,
        )
    sensitivity = _sensitivity_evidence(sensitivity_result)
    optimization = _optimization_evidence(optimization_result)
    scenario_comparison = _scenario_evidence(comparison_result)
    emissions = _emissions_evidence(emissions_result)
    energy_audit = _energy_audit_evidence(energy_audit_result)
    reason = ""
    if not any(
        (sensitivity, optimization, scenario_comparison, emissions, energy_audit)
    ):
        reason = "No Process Chat engineering evidence is available yet."
    return {
        "available": True,
        "reason": reason,
        "provenance": {
            "case_id": context.active_case.get("case_id"),
            "solved_signature": context.signature,
            "source": "Process Chat deterministic NeqSim studies",
            "model_name": context.active_case.get("runtime", {}).get("model_name"),
        },
        "sensitivity": sensitivity,
        "optimization": optimization,
        "scenario_comparison": scenario_comparison,
        "emissions": emissions,
        "energy_audit": energy_audit,
    }
