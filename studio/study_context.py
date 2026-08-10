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
        }
    if getattr(chat_session, "model", None) is not context.model:
        return {
            "available": False,
            "reason": (
                "Process Chat study evidence belongs to a different runtime model. "
                "Reopen Process Chat from the active solved case and rerun the study."
            ),
        }

    sensitivity = _sensitivity_evidence(
        _getter(chat_session, "get_last_sensitivity")
    )
    optimization = _optimization_evidence(
        _getter(chat_session, "get_last_optimization")
    )
    reason = ""
    if sensitivity is None and optimization is None:
        reason = "No sensitivity or optimization result is available yet."
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
    }
