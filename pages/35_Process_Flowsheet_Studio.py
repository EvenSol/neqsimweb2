"""Process Flowsheet Studio for structured, reproducible NeqSim studies."""

from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import asdict
from typing import Any

import pandas as pd
import streamlit as st


# Keep JVM serialization compatible with the Process Chat model adapter.
_JVM_OPENS = (
    "--add-opens=java.base/java.util=ALL-UNNAMED "
    "--add-opens=java.base/java.lang=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
    "--add-opens=java.base/java.io=ALL-UNNAMED"
)
if "add-opens" not in os.environ.get("JAVA_TOOL_OPTIONS", ""):
    os.environ["JAVA_TOOL_OPTIONS"] = _JVM_OPENS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from process_chat.process_builder import ProcessBuilder  # noqa: E402
from theme import apply_theme, theme_toggle  # noqa: E402


PAGE_TITLE = "Process Flowsheet Studio"
TEMPLATE_NAME = "Inlet separation and two-stage gas compression"
CASE_STATE_KEY = "flowsheet_studio_case"
RESULT_STATE_KEY = "flowsheet_studio_result"

DEFAULT_COMPOSITION = pd.DataFrame(
    {
        "component": [
            "nitrogen",
            "CO2",
            "methane",
            "ethane",
            "propane",
            "i-butane",
            "n-butane",
            "i-pentane",
            "n-pentane",
            "n-hexane",
        ],
        "mole_fraction": [
            0.010,
            0.020,
            0.850,
            0.060,
            0.030,
            0.008,
            0.012,
            0.004,
            0.003,
            0.003,
        ],
    }
)


def _clean_composition(table: pd.DataFrame) -> tuple[dict[str, float], float]:
    """Validate and normalize an editable composition table."""
    required = {"component", "mole_fraction"}
    if not required.issubset(table.columns):
        raise ValueError("The composition table must contain component and mole_fraction.")

    cleaned: dict[str, float] = {}
    for _, row in table.iterrows():
        component = str(row["component"]).strip()
        if not component or component.lower() == "nan":
            continue
        try:
            fraction = float(row["mole_fraction"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid mole fraction for {component}.") from exc
        if fraction < 0.0:
            raise ValueError(f"Mole fraction for {component} cannot be negative.")
        if fraction > 0.0:
            cleaned[component] = cleaned.get(component, 0.0) + fraction

    total = sum(cleaned.values())
    if not cleaned or total <= 0.0:
        raise ValueError("Enter at least one component with a positive mole fraction.")
    normalized = {name: value / total for name, value in cleaned.items()}
    return normalized, total


def _build_case_spec(
    case_name: str,
    composition: dict[str, float],
    eos_model: str,
    feed_temperature_c: float,
    feed_pressure_bara: float,
    feed_flow_kg_hr: float,
    stage_1_pressure_bara: float,
    stage_2_pressure_bara: float,
    intercooler_temperature_c: float,
    export_temperature_c: float,
    isentropic_efficiency: float,
) -> dict[str, Any]:
    """Create the ProcessBuilder specification for the first Studio template."""
    return {
        "schema_version": 1,
        "name": case_name,
        "description": (
            "Gas-rich feed through an inlet scrubber, two compressor stages, "
            "intercooling, interstage separation, and export cooling."
        ),
        "assumptions": [
            "Steady-state simulation.",
            "Pressures are absolute (bara).",
            "Feed flow is mass flow in kg/hr.",
            "Compressors use the specified isentropic efficiency.",
            "Coolers impose outlet temperature without an explicit pressure drop.",
        ],
        "fluid": {
            "eos_model": eos_model,
            "mixing_rule": 2,
            "components": composition,
            "composition_basis": "mole_fraction",
            "temperature_C": feed_temperature_c,
            "pressure_bara": feed_pressure_bara,
            "total_flow": feed_flow_kg_hr,
            "flow_unit": "kg/hr",
        },
        "process": [
            {
                "name": "feed gas",
                "type": "stream",
                "params": {
                    "temperature_C": feed_temperature_c,
                    "pressure_bara": feed_pressure_bara,
                    "flow_rate": feed_flow_kg_hr,
                    "flow_unit": "kg/hr",
                },
            },
            {
                "name": "inlet scrubber",
                "type": "separator",
                "outlet": "gas",
            },
            {
                "name": "compressor stage 1",
                "type": "compressor",
                "params": {
                    "outlet_pressure_bara": stage_1_pressure_bara,
                    "isentropic_efficiency": isentropic_efficiency,
                },
            },
            {
                "name": "intercooler",
                "type": "cooler",
                "params": {
                    "outlet_temperature_C": intercooler_temperature_c,
                },
            },
            {
                "name": "interstage scrubber",
                "type": "separator",
                "outlet": "gas",
            },
            {
                "name": "compressor stage 2",
                "type": "compressor",
                "params": {
                    "outlet_pressure_bara": stage_2_pressure_bara,
                    "isentropic_efficiency": isentropic_efficiency,
                },
            },
            {
                "name": "export cooler",
                "type": "cooler",
                "params": {
                    "outlet_temperature_C": export_temperature_c,
                },
            },
        ],
    }


def _validate_case(spec: dict[str, Any], composition_total: float) -> list[str]:
    """Return non-blocking engineering warnings after hard validation."""
    warnings: list[str] = []
    fluid = spec["fluid"]
    process = spec["process"]

    stage_1_pressure = process[2]["params"]["outlet_pressure_bara"]
    stage_2_pressure = process[5]["params"]["outlet_pressure_bara"]
    feed_pressure = fluid["pressure_bara"]
    efficiency = process[2]["params"]["isentropic_efficiency"]

    if feed_pressure <= 0.0:
        raise ValueError("Feed pressure must be greater than zero bara.")
    if fluid["total_flow"] <= 0.0:
        raise ValueError("Feed flow must be greater than zero kg/hr.")
    if not feed_pressure < stage_1_pressure < stage_2_pressure:
        raise ValueError(
            "Pressure ordering must be feed pressure < stage 1 pressure "
            "< stage 2 pressure."
        )
    if not 0.5 <= efficiency <= 1.0:
        raise ValueError("Isentropic efficiency must be between 0.50 and 1.00.")
    if abs(composition_total - 1.0) > 1.0e-6:
        warnings.append(
            f"Composition summed to {composition_total:.6f} and was normalized to 1.0."
        )
    if stage_1_pressure / feed_pressure > 3.0:
        warnings.append("Stage 1 pressure ratio exceeds 3.0; check compressor feasibility.")
    if stage_2_pressure / stage_1_pressure > 3.0:
        warnings.append("Stage 2 pressure ratio exceeds 3.0; check compressor feasibility.")
    if fluid["eos_model"] == "gerg2008":
        warnings.append(
            "GERG-2008 is intended for gas-phase property calculations; "
            "use SRK/PR/CPA if liquid dropout is important."
        )
    return warnings


def _stream_dataframe(model: Any) -> pd.DataFrame:
    """Create a compact stream table without duplicate short aliases."""
    records = []
    for stream in model.list_streams():
        if "." not in stream.name and "/" not in stream.name and stream.name != "feed gas":
            continue
        records.append(
            {
                "Stream": stream.name,
                "Temperature [°C]": stream.temperature_C,
                "Pressure [bara]": stream.pressure_bara,
                "Mass flow [kg/hr]": stream.flow_rate_kg_hr,
                "Molar flow [mol/s]": stream.flow_rate_mol_sec,
            }
        )
    return pd.DataFrame(records).drop_duplicates()


def _equipment_dataframe(model: Any) -> pd.DataFrame:
    """Create an equipment performance table from the shared model adapter."""
    records = []
    for unit in model.list_units():
        row = {
            "Equipment": unit.name,
            "Type": unit.unit_type,
            "Process system": unit.process_system,
        }
        row.update(unit.properties)
        records.append(row)
    return pd.DataFrame(records)


def _constraint_dataframe(result: Any) -> pd.DataFrame:
    records = [asdict(item) for item in result.constraints]
    if not records:
        records.append(
            {
                "name": "simulation",
                "status": "OK",
                "detail": "The process completed without reported constraint warnings.",
            }
        )
    return pd.DataFrame(records)


def _kpi_value(result: Any, name: str) -> float | None:
    kpi = result.kpis.get(name)
    return float(kpi.value) if kpi is not None else None


def _format_metric(value: float | None, unit: str, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.{digits}f} {unit}"


st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="images/neqsimlogocircleflat.png",
    layout="wide",
)
apply_theme()
theme_toggle()

st.title("🏭 Process Flowsheet Studio")
st.markdown(
    """
Build and solve a reproducible NeqSim process case using structured engineering
inputs. Version 1 provides an inlet-separation and two-stage gas-compression
template. The solved process is shared with **Process Chat** for further
what-if studies.
"""
)

solver_status = "Solved" if st.session_state.get(RESULT_STATE_KEY) else "Not run"
with st.sidebar:
    st.divider()
    st.subheader("🏭 Flowsheet case")
    st.caption(TEMPLATE_NAME)
    st.write("**Mode:** Steady state")
    solver_status_placeholder = st.empty()
    solver_status_placeholder.write(f"**Solver:** {solver_status}")
    st.write("**Workspace:** Setup → Flowsheet → Workbook → Validation")

with st.expander("Model scope and assumptions", expanded=False):
    st.markdown(
        """
- Steady-state thermodynamic and process simulation in NeqSim.
- Pressure inputs are absolute (`bara`); temperature inputs are degrees Celsius.
- Composition is molar and is normalized before calculation.
- Cooling duties and compressor powers come directly from the solved NeqSim model.
- Results are suitable for screening and engineering studies, not design certification.
"""
    )

st.subheader("1. Case setup")
case_name = st.text_input(
    "Case name",
    value="Gas Compression Case",
    help="A reusable engineering case name stored with the downloaded model.",
)
st.caption(f"Template: {TEMPLATE_NAME}")

st.markdown("#### Fluid and operating basis")
fluid_col, process_col = st.columns(2)

with fluid_col:
    eos_model = st.selectbox(
        "Equation of state",
        options=["srk", "pr", "cpa", "gerg2008"],
        index=0,
        format_func=lambda value: value.upper(),
        help="Mixing rule 2 is used for cubic/association equations of state.",
    )
    feed_temperature_c = st.number_input(
        "Feed temperature [°C]",
        min_value=-100.0,
        max_value=200.0,
        value=30.0,
        step=1.0,
    )
    feed_pressure_bara = st.number_input(
        "Feed pressure [bara]",
        min_value=1.0,
        max_value=500.0,
        value=50.0,
        step=1.0,
    )
    feed_flow_kg_hr = st.number_input(
        "Feed mass flow [kg/hr]",
        min_value=1.0,
        max_value=10_000_000.0,
        value=100_000.0,
        step=1_000.0,
    )

with process_col:
    stage_1_pressure_bara = st.number_input(
        "Stage 1 discharge pressure [bara]",
        min_value=1.0,
        max_value=500.0,
        value=80.0,
        step=1.0,
    )
    stage_2_pressure_bara = st.number_input(
        "Stage 2 discharge pressure [bara]",
        min_value=1.0,
        max_value=500.0,
        value=130.0,
        step=1.0,
    )
    isentropic_efficiency = st.slider(
        "Compressor isentropic efficiency",
        min_value=0.50,
        max_value=0.95,
        value=0.78,
        step=0.01,
    )
    intercooler_temperature_c = st.number_input(
        "Intercooler outlet temperature [°C]",
        min_value=-50.0,
        max_value=150.0,
        value=35.0,
        step=1.0,
    )
    export_temperature_c = st.number_input(
        "Export cooler outlet temperature [°C]",
        min_value=-50.0,
        max_value=150.0,
        value=40.0,
        step=1.0,
    )

st.markdown("**Feed composition**")
composition_table = st.data_editor(
    DEFAULT_COMPOSITION,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "component": st.column_config.TextColumn("NeqSim component"),
        "mole_fraction": st.column_config.NumberColumn(
            "Mole fraction",
            min_value=0.0,
            max_value=1.0,
            format="%.6f",
        ),
    },
    key="flowsheet_composition_editor",
)

try:
    preview_composition, preview_total = _clean_composition(composition_table)
    total_col, count_col = st.columns(2)
    total_col.metric("Entered mole-fraction sum", f"{preview_total:.6f}")
    count_col.metric("Active components", len(preview_composition))
except ValueError as preview_error:
    preview_composition = {}
    preview_total = 0.0
    st.warning(str(preview_error))

run_case = st.button(
    "▶ Run NeqSim flowsheet",
    type="primary",
    use_container_width=True,
)

if run_case:
    try:
        composition, composition_total = _clean_composition(composition_table)
        case_spec = _build_case_spec(
            case_name=case_name.strip() or "Gas Compression Case",
            composition=composition,
            eos_model=eos_model,
            feed_temperature_c=feed_temperature_c,
            feed_pressure_bara=feed_pressure_bara,
            feed_flow_kg_hr=feed_flow_kg_hr,
            stage_1_pressure_bara=stage_1_pressure_bara,
            stage_2_pressure_bara=stage_2_pressure_bara,
            intercooler_temperature_c=intercooler_temperature_c,
            export_temperature_c=export_temperature_c,
            isentropic_efficiency=isentropic_efficiency,
        )
        case_warnings = _validate_case(case_spec, composition_total)

        with st.spinner("Building and solving the NeqSim process..."):
            builder = ProcessBuilder()
            model = builder.build_from_spec(case_spec)
            result = model.run()
            model_bytes = builder.save_neqsim_bytes()

        state = {
            "spec": case_spec,
            "warnings": case_warnings,
            "builder": builder,
            "model": model,
            "result": result,
            "model_bytes": model_bytes,
        }
        st.session_state[CASE_STATE_KEY] = state
        st.session_state[RESULT_STATE_KEY] = True

        # Shared state used by the existing Process Chat page.
        st.session_state["process_model"] = model
        st.session_state["process_model_name"] = "process_flowsheet_studio.neqsim"
        if model_bytes:
            st.session_state["process_model_bytes"] = model_bytes

        solver_status_placeholder.write("**Solver:** Solved")
        st.success("The NeqSim flowsheet solved and is ready for review.")
    except Exception as exc:
        st.session_state.pop(RESULT_STATE_KEY, None)
        solver_status_placeholder.write("**Solver:** Failed")
        st.error(f"Flowsheet calculation failed: {exc}")
        with st.expander("Technical error details", expanded=False):
            st.code(traceback.format_exc())

if st.session_state.get(RESULT_STATE_KEY) and CASE_STATE_KEY in st.session_state:
    state = st.session_state[CASE_STATE_KEY]
    spec = state["spec"]
    builder = state["builder"]
    model = state["model"]
    result = state["result"]
    model_bytes = state["model_bytes"]

    st.divider()
    st.subheader("2. Engineering results")

    for warning in state["warnings"]:
        st.warning(warning)

    total_power_kw = _kpi_value(result, "total_power_kW")
    total_duty_kw = _kpi_value(result, "total_duty_kW")
    mass_balance_pct = _kpi_value(result, "mass_balance_pct")
    feed_tonnes_per_hour = spec["fluid"]["total_flow"] / 1000.0
    specific_energy = None
    if total_power_kw is not None and feed_tonnes_per_hour > 0.0:
        specific_energy = total_power_kw / feed_tonnes_per_hour

    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Total compressor power",
        _format_metric(total_power_kw, "kW"),
    )
    metric_cols[1].metric(
        "Total |cooling duty|",
        _format_metric(total_duty_kw, "kW"),
    )
    metric_cols[2].metric(
        "Specific compression energy",
        _format_metric(specific_energy, "kWh/t"),
    )
    metric_cols[3].metric(
        "Mass imbalance",
        _format_metric(mass_balance_pct, "%", digits=3),
    )

    diagram_tab, streams_tab, equipment_tab, validation_tab = st.tabs(
        [
            "Flowsheet",
            "Workbook · Streams",
            "Workbook · Equipment",
            "Solver & Validation",
        ]
    )

    with diagram_tab:
        try:
            dot_source = model.get_diagram_dot(
                style="HYSYS",
                detail_level="ENGINEERING",
                show_stream_values=True,
                title=spec["name"],
            )
            st.graphviz_chart(dot_source, use_container_width=True)
        except Exception as diagram_error:
            st.warning(f"Diagram rendering was unavailable: {diagram_error}")
        with st.expander("Build log", expanded=False):
            st.code("\n".join(builder.build_log))

    stream_table = _stream_dataframe(model)
    with streams_tab:
        if stream_table.empty:
            st.info("No stream rows were returned by the model adapter.")
        else:
            st.dataframe(
                stream_table.style.format(
                    {
                        "Temperature [°C]": "{:.2f}",
                        "Pressure [bara]": "{:.3f}",
                        "Mass flow [kg/hr]": "{:,.2f}",
                        "Molar flow [mol/s]": "{:,.4f}",
                    },
                    na_rep="—",
                ),
                use_container_width=True,
                hide_index=True,
            )

    equipment_table = _equipment_dataframe(model)
    with equipment_tab:
        if equipment_table.empty:
            st.info("No equipment rows were returned by the model adapter.")
        else:
            st.dataframe(
                equipment_table,
                use_container_width=True,
                hide_index=True,
            )

    constraint_table = _constraint_dataframe(result)
    with validation_tab:
        status_counts = constraint_table["status"].value_counts()
        if status_counts.get("VIOLATION", 0) > 0:
            st.error("One or more engineering validation checks reported a violation.")
        elif status_counts.get("WARN", 0) > 0:
            st.warning("The calculation completed with engineering warnings.")
        else:
            st.success("All reported engineering validation checks passed.")
        st.dataframe(
            constraint_table,
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Validation level: NeqSim convergence evidence, pressure ordering, "
            "composition normalization, mass balance, and engineering bounds."
        )

    st.subheader("3. Reproducible deliverables")
    case_json = json.dumps(spec, indent=2)
    python_script = builder.to_python_script()
    download_cols = st.columns(3)
    download_cols[0].download_button(
        "Download case JSON",
        data=case_json,
        file_name="process_flowsheet_case.json",
        mime="application/json",
        use_container_width=True,
    )
    download_cols[1].download_button(
        "Download Python model",
        data=python_script,
        file_name="process_flowsheet_model.py",
        mime="text/x-python",
        use_container_width=True,
    )
    if model_bytes:
        download_cols[2].download_button(
            "Download .neqsim model",
            data=model_bytes,
            file_name="process_flowsheet_studio.neqsim",
            mime="application/zip",
            use_container_width=True,
        )
    else:
        download_cols[2].info("Serialized NeqSim model was unavailable.")

    st.info(
        "This solved process is also available in the current session under "
        "Process Chat for natural-language what-if analysis."
    )
