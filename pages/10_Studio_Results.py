"""Professional active-case results workspace for NeqSim Studio."""

from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from studio.results_context import (  # noqa: E402
    RESULT_SECTIONS,
    StudioResultsUnavailable,
    build_result_summary,
    case_history_rows,
    constraint_rows,
    equipment_design_rows,
    equipment_rows,
    load_current_result_context,
    selected_result_section,
    stream_rows,
    validation_tables,
)
from theme import apply_theme, theme_toggle  # noqa: E402


st.set_page_config(
    page_title="Studio Engineering Results",
    page_icon="images/neqsimlogocircleflat.png",
    layout="wide",
)
apply_theme()
theme_toggle()

with st.sidebar:
    st.markdown("### Studio workspace")
    if st.button("← Studio home", use_container_width=True):
        st.switch_page("pages/00_NeqSim_Studio.py")
    if st.button("Edit or run flowsheet", use_container_width=True):
        st.switch_page("pages/35_Process_Flowsheet_Studio.py")
    if st.button("NeqSim Classic", use_container_width=True):
        st.switch_page("welcome.py")
    st.caption(
        "Results are read from the existing solved Process Flowsheet Studio model. "
        "No calculation is repeated in this page."
    )

st.title("Engineering Results")
st.caption(
    "Review the exact active solved case, its units, conservation evidence, "
    "equipment limits, provenance and session comparisons."
)

try:
    context = load_current_result_context(st.session_state)
except StudioResultsUnavailable as unavailable:
    st.info(str(unavailable))
    if st.button("Open Process Flowsheet", type="primary"):
        st.switch_page("pages/35_Process_Flowsheet_Studio.py")
    st.stop()

summary = build_result_summary(context)
case = context.active_case
state = summary["engineering_state"]
if state == "Attention required":
    st.error(f"{case['name']} · {state}")
elif state == "Solved with warnings":
    st.warning(f"{case['name']} · {state}")
else:
    st.success(f"{case['name']} · {state}")

for warning in context.warnings:
    st.warning(warning)


def _metric_value(metric: dict | None, digits: int = 2) -> str:
    if not metric:
        return "—"
    return f"{metric['value']:,.{digits}f} {metric['unit']}".strip()


metrics = summary["metrics"]
metric_cols = st.columns(5)
metric_cols[0].metric("Compressor power", _metric_value(metrics["total_power"]))
metric_cols[1].metric("|Cooling duty|", _metric_value(metrics["total_duty"]))
metric_cols[2].metric("Specific energy", _metric_value(metrics["specific_energy"]))
metric_cols[3].metric("Mass imbalance", _metric_value(metrics["mass_imbalance"], 6))
metric_cols[4].metric("Validation", summary["validation_status"])

default_section = selected_result_section(st.session_state)
selected_section = st.radio(
    "Results view",
    RESULT_SECTIONS,
    index=RESULT_SECTIONS.index(default_section),
    horizontal=True,
)

if selected_section == "Overview":
    left, right = st.columns(2)
    with left:
        st.subheader("Case basis")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Property": "Case ID", "Value": case["case_id"]},
                    {"Property": "Schema", "Value": f"v{case['case_schema_version']}"},
                    {"Property": "Thermodynamic model", "Value": case["thermodynamics"]["eos_model"]},
                    {"Property": "Mixing rule", "Value": case["thermodynamics"].get("mixing_rule") or "Not reported"},
                    {"Property": "Unit system", "Value": case["units"]["system"]},
                    {"Property": "Solved signature", "Value": context.signature[:16]},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.subheader("Provenance")
        provenance_rows = [
            {"Property": key.replace("_", " ").title(), "Value": value}
            for key, value in case["provenance"].items()
        ]
        provenance_rows.extend(
            {"Property": key.replace("_", " ").title(), "Value": value}
            for key, value in context.run_record.items()
        )
        st.dataframe(
            pd.DataFrame(provenance_rows),
            use_container_width=True,
            hide_index=True,
        )
    st.caption(
        "Assumptions and limitations: values come from the current native NeqSim "
        "solve; pressure is absolute bara; temperatures are degC; design rows are "
        "screening evidence unless project design data and accountable review exist."
    )

if selected_section == "Streams":
    rows = stream_rows(context)
    st.subheader(f"Solved streams · {len(rows)}")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

if selected_section == "Equipment & design":
    rows = equipment_rows(context)
    st.subheader(f"Solved equipment · {len(rows)}")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.subheader("Operating versus design basis")
    design_rows = equipment_design_rows(context)
    if design_rows:
        st.dataframe(
            pd.DataFrame(design_rows), use_container_width=True, hide_index=True
        )
    else:
        st.info(
            "No active equipment design capacities were reported. Configure a design "
            "basis in the Process Flowsheet equipment properties to create this evidence."
        )
    st.subheader("Engineering constraints")
    st.dataframe(
        pd.DataFrame(constraint_rows(context)),
        use_container_width=True,
        hide_index=True,
    )

if selected_section == "Validation":
    tables = validation_tables(context)
    energy = summary["energy_balance"]
    st.subheader("Convergence")
    convergence = summary["convergence"]
    if convergence["applicable"] is None:
        st.info("This legacy result did not record native convergence diagnostics.")
    elif convergence["applicable"] is False:
        st.success("Feed-forward solve; no recycle or adjuster loops are present.")
    elif convergence["converged"]:
        st.success("Every native recycle and adjuster reports convergence.")
    else:
        st.error("One or more native recycle or adjuster units did not converge.")
    if tables["convergence"]:
        st.dataframe(
            pd.DataFrame(tables["convergence"]),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("System energy balance")
    if energy["applicable"] is True:
        st.dataframe(
            pd.DataFrame(
                [
                    {"Term": "Feed enthalpy", "Value": energy["feed_enthalpy_kW"], "Unit": "kW"},
                    {"Term": "Product enthalpy", "Value": energy["product_enthalpy_kW"], "Unit": "kW"},
                    {"Term": "External transfer", "Value": energy["external_energy_transfer_kW"], "Unit": "kW"},
                    {"Term": "Residual", "Value": energy["residual_kW"], "Unit": "kW"},
                    {"Term": "Relative imbalance", "Value": energy["imbalance_pct"], "Unit": "%"},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("System energy closure is unavailable for this result.")

    for title, table_key in (
        ("Material boundaries", "material_boundaries"),
        ("Component balance", "component_balances"),
        ("Per-unit closure", "unit_balances"),
        ("Audited energy transfers", "energy_transfers"),
    ):
        st.subheader(title)
        rows = tables[table_key]
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info(f"{title} evidence is unavailable for this result.")

if selected_section == "Case studies":
    st.subheader("Solved session case comparison")
    history = case_history_rows(st.session_state)
    if history:
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
    else:
        st.info(
            "Solve more than one case to build comparison evidence in this session."
        )
    st.caption(
        "Sensitivity, adjust/specification and bounded optimization continue to run "
        "through the inherited Process Flowsheet Studio tools and native NeqSim model."
    )
    if st.button("Open studies in Process Flowsheet", type="primary"):
        st.switch_page("pages/35_Process_Flowsheet_Studio.py")

st.divider()
st.caption(
    "This page is a presentation adapter over the existing solved model. Edit, solve, "
    "export and download reproducible deliverables in Process Flowsheet Studio."
)
