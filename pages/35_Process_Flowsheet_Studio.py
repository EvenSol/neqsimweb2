"""Process Flowsheet Studio for structured, reproducible NeqSim studies."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import traceback
from dataclasses import asdict
from io import BytesIO
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
FAILURE_SIGNATURE_STATE_KEY = "flowsheet_studio_failure_signature"
CASE_HISTORY_STATE_KEY = "flowsheet_studio_case_history"
CASE_SCHEMA_VERSION = 1
MAX_CASE_FILE_BYTES = 1_000_000
MAX_CASE_HISTORY = 20
SUPPORTED_EOS_MODELS = ("srk", "pr", "cpa", "gerg2008")
EXPECTED_TEMPLATE_TOPOLOGY = (
    ("feed gas", "stream"),
    ("inlet scrubber", "separator"),
    ("compressor stage 1", "compressor"),
    ("intercooler", "cooler"),
    ("interstage scrubber", "separator"),
    ("compressor stage 2", "compressor"),
    ("export cooler", "cooler"),
)
TEMPLATE_OBJECTS = {
    "feed gas": ("Feed gas", "Material stream"),
    "inlet scrubber": ("Inlet scrubber", "Separator"),
    "compressor stage 1": ("Compressor stage 1", "Compressor"),
    "intercooler": ("Intercooler", "Cooler"),
    "interstage scrubber": ("Interstage scrubber", "Separator"),
    "compressor stage 2": ("Compressor stage 2", "Compressor"),
    "export cooler": ("Export cooler", "Cooler"),
}

CONTROL_DEFAULTS = {
    "flowsheet_case_name": "Gas Compression Case",
    "flowsheet_eos_model": "srk",
    "flowsheet_feed_temperature_c": 30.0,
    "flowsheet_feed_pressure_bara": 50.0,
    "flowsheet_feed_flow_kg_hr": 100_000.0,
    "flowsheet_stage_1_pressure_bara": 80.0,
    "flowsheet_stage_2_pressure_bara": 130.0,
    "flowsheet_isentropic_efficiency": 0.78,
    "flowsheet_intercooler_temperature_c": 35.0,
    "flowsheet_export_temperature_c": 40.0,
}

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


def _initialize_case_controls() -> None:
    """Initialize stable widget state used by new and imported cases."""
    for key, value in CONTROL_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value
        else:
            # Preserve controls whose selected-object widget is temporarily hidden.
            st.session_state[key] = st.session_state[key]
    if "flowsheet_composition_source" not in st.session_state:
        st.session_state["flowsheet_composition_source"] = DEFAULT_COMPOSITION.copy()
    if "flowsheet_composition_revision" not in st.session_state:
        st.session_state["flowsheet_composition_revision"] = 0
    if CASE_HISTORY_STATE_KEY not in st.session_state:
        st.session_state[CASE_HISTORY_STATE_KEY] = []


def _finite_float(value: Any, field_name: str) -> float:
    """Convert a JSON value to a finite float with a field-specific error."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite.")
    return result


def _bounded_float(
    value: Any,
    field_name: str,
    minimum: float,
    maximum: float,
) -> float:
    """Return a finite float inside the range supported by its UI control."""
    result = _finite_float(value, field_name)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"{field_name} must be between {minimum:g} and {maximum:g}."
        )
    return result


def _load_case_controls(case_data: Any) -> tuple[dict[str, Any], pd.DataFrame, list[str]]:
    """Validate an exported Studio case and map it back to UI controls."""
    if not isinstance(case_data, dict):
        raise ValueError("The case JSON root must be an object.")
    schema_version = case_data.get("schema_version")
    if type(schema_version) is not int or schema_version != CASE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version. Expected {CASE_SCHEMA_VERSION}."
        )

    case_name = str(case_data.get("name", "")).strip()
    if not case_name:
        raise ValueError("The case must have a non-empty name.")
    if len(case_name) > 120:
        raise ValueError("The case name cannot exceed 120 characters.")

    fluid = case_data.get("fluid")
    if not isinstance(fluid, dict):
        raise ValueError("The case must contain a fluid object.")

    eos_model = str(fluid.get("eos_model", "")).lower().strip()
    if eos_model not in SUPPORTED_EOS_MODELS:
        raise ValueError(
            "Unsupported equation of state. Use SRK, PR, CPA, or GERG2008."
        )
    if fluid.get("composition_basis") != "mole_fraction":
        raise ValueError("Only mole_fraction composition basis is supported.")
    if fluid.get("flow_unit") != "kg/hr":
        raise ValueError("Only kg/hr feed flow is supported.")
    mixing_rule = _finite_float(fluid.get("mixing_rule"), "fluid.mixing_rule")
    if not mixing_rule.is_integer() or int(mixing_rule) != 2:
        raise ValueError("This Studio version requires mixing rule 2.")

    components = fluid.get("components")
    if not isinstance(components, dict):
        raise ValueError("fluid.components must be an object.")
    composition_rows = []
    for component, value in components.items():
        name = str(component).strip()
        if not name:
            raise ValueError("Component names cannot be empty.")
        composition_rows.append(
            {
                "component": name,
                "mole_fraction": _finite_float(
                    value,
                    f"fluid.components.{name}",
                ),
            }
        )
    composition_table = pd.DataFrame(composition_rows)
    normalized_composition, composition_total = _clean_composition(composition_table)
    composition_table = pd.DataFrame(
        {
            "component": list(normalized_composition.keys()),
            "mole_fraction": list(normalized_composition.values()),
        }
    )

    process = case_data.get("process")
    if not isinstance(process, list):
        raise ValueError("The case must contain a process array.")
    topology = tuple(
        (
            str(step.get("name", "")),
            str(step.get("type", "")),
        )
        for step in process
        if isinstance(step, dict)
    )
    if topology != EXPECTED_TEMPLATE_TOPOLOGY:
        raise ValueError(
            "The imported process does not match the supported two-stage "
            "gas-compression template."
        )

    steps = {step["name"]: step for step in process}
    if any(
        steps[name].get("outlet") != "gas"
        for name in ("inlet scrubber", "interstage scrubber")
    ):
        raise ValueError("Both scrubbers must use their gas outlet.")
    feed_params = steps["feed gas"].get("params", {})
    compressor_1_params = steps["compressor stage 1"].get("params", {})
    compressor_2_params = steps["compressor stage 2"].get("params", {})
    intercooler_params = steps["intercooler"].get("params", {})
    export_cooler_params = steps["export cooler"].get("params", {})
    parameter_objects = (
        feed_params,
        compressor_1_params,
        compressor_2_params,
        intercooler_params,
        export_cooler_params,
    )
    if not all(isinstance(params, dict) for params in parameter_objects):
        raise ValueError("Every configurable process step must have a params object.")

    feed_temperature_c = _bounded_float(
        fluid.get("temperature_C"),
        "fluid.temperature_C",
        -100.0,
        200.0,
    )
    feed_pressure_bara = _bounded_float(
        fluid.get("pressure_bara"),
        "fluid.pressure_bara",
        1.0,
        500.0,
    )
    feed_flow_kg_hr = _bounded_float(
        fluid.get("total_flow"),
        "fluid.total_flow",
        1.0,
        10_000_000.0,
    )
    feed_param_temperature = _finite_float(
        feed_params.get("temperature_C"),
        "feed gas temperature_C",
    )
    feed_param_pressure = _finite_float(
        feed_params.get("pressure_bara"),
        "feed gas pressure_bara",
    )
    feed_param_flow = _finite_float(
        feed_params.get("flow_rate"),
        "feed gas flow_rate",
    )
    if feed_params.get("flow_unit") != "kg/hr":
        raise ValueError("The feed stream flow_unit must be kg/hr.")
    if not math.isclose(feed_temperature_c, feed_param_temperature):
        raise ValueError("Fluid and feed-stream temperatures are inconsistent.")
    if not math.isclose(feed_pressure_bara, feed_param_pressure):
        raise ValueError("Fluid and feed-stream pressures are inconsistent.")
    if not math.isclose(feed_flow_kg_hr, feed_param_flow):
        raise ValueError("Fluid and feed-stream flow rates are inconsistent.")

    stage_1_pressure_bara = _bounded_float(
        compressor_1_params.get("outlet_pressure_bara"),
        "compressor stage 1 outlet pressure",
        1.0,
        500.0,
    )
    stage_2_pressure_bara = _bounded_float(
        compressor_2_params.get("outlet_pressure_bara"),
        "compressor stage 2 outlet pressure",
        1.0,
        500.0,
    )
    efficiency_1 = _bounded_float(
        compressor_1_params.get("isentropic_efficiency"),
        "compressor stage 1 isentropic efficiency",
        0.50,
        0.95,
    )
    efficiency_2 = _bounded_float(
        compressor_2_params.get("isentropic_efficiency"),
        "compressor stage 2 isentropic efficiency",
        0.50,
        0.95,
    )
    if not math.isclose(efficiency_1, efficiency_2):
        raise ValueError(
            "Both compressor stages must use the same isentropic efficiency."
        )
    intercooler_temperature_c = _bounded_float(
        intercooler_params.get("outlet_temperature_C"),
        "intercooler outlet temperature",
        -50.0,
        150.0,
    )
    export_temperature_c = _bounded_float(
        export_cooler_params.get("outlet_temperature_C"),
        "export cooler outlet temperature",
        -50.0,
        150.0,
    )

    canonical_spec = _build_case_spec(
        case_name=case_name,
        composition=normalized_composition,
        eos_model=eos_model,
        feed_temperature_c=feed_temperature_c,
        feed_pressure_bara=feed_pressure_bara,
        feed_flow_kg_hr=feed_flow_kg_hr,
        stage_1_pressure_bara=stage_1_pressure_bara,
        stage_2_pressure_bara=stage_2_pressure_bara,
        intercooler_temperature_c=intercooler_temperature_c,
        export_temperature_c=export_temperature_c,
        isentropic_efficiency=efficiency_1,
    )
    warnings = _validate_case(canonical_spec, composition_total)
    controls = {
        "flowsheet_case_name": case_name,
        "flowsheet_eos_model": eos_model,
        "flowsheet_feed_temperature_c": feed_temperature_c,
        "flowsheet_feed_pressure_bara": feed_pressure_bara,
        "flowsheet_feed_flow_kg_hr": feed_flow_kg_hr,
        "flowsheet_stage_1_pressure_bara": stage_1_pressure_bara,
        "flowsheet_stage_2_pressure_bara": stage_2_pressure_bara,
        "flowsheet_isentropic_efficiency": efficiency_1,
        "flowsheet_intercooler_temperature_c": intercooler_temperature_c,
        "flowsheet_export_temperature_c": export_temperature_c,
    }
    return controls, composition_table, warnings


def _apply_imported_case(
    controls: dict[str, Any],
    composition_table: pd.DataFrame,
    warnings: list[str],
) -> None:
    """Replace the current controls and invalidate any previously solved model."""
    for key, value in controls.items():
        st.session_state[key] = value
    st.session_state["flowsheet_composition_source"] = composition_table
    st.session_state["flowsheet_composition_revision"] += 1
    st.session_state.pop(CASE_STATE_KEY, None)
    st.session_state.pop(RESULT_STATE_KEY, None)
    st.session_state.pop(FAILURE_SIGNATURE_STATE_KEY, None)
    if st.session_state.get("process_model_name") == "process_flowsheet_studio.neqsim":
        st.session_state.pop("process_model", None)
        st.session_state.pop("process_model_name", None)
        st.session_state.pop("process_model_bytes", None)
    notice = "Case loaded. Review the inputs and run the NeqSim flowsheet."
    if warnings:
        notice += " " + " ".join(warnings)
    st.session_state["flowsheet_import_notice"] = notice


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
        fraction = _finite_float(
            row["mole_fraction"],
            f"Mole fraction for {component}",
        )
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
        "schema_version": CASE_SCHEMA_VERSION,
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


def _case_signature(spec: dict[str, Any], composition_total: float) -> str:
    """Return a deterministic identity for the inputs and their normalization."""
    signature_payload = {
        "spec": spec,
        "entered_composition_total": composition_total,
    }
    encoded_payload = json.dumps(
        signature_payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded_payload).hexdigest()


def _solver_status(
    current_signature: str | None,
    stored_state: Any,
    has_result: bool,
    failure_signature: str | None,
) -> tuple[str, bool]:
    """Classify solver state and whether the stored result matches the inputs."""
    stored_signature = (
        stored_state.get("signature") if isinstance(stored_state, dict) else None
    )
    has_stored_result = bool(has_result and isinstance(stored_state, dict))
    results_are_current = bool(
        has_stored_result
        and current_signature is not None
        and stored_signature == current_signature
    )
    if current_signature is None:
        return "Invalid inputs", False
    if results_are_current:
        return "Solved", True
    if failure_signature == current_signature:
        return "Failed", False
    if has_stored_result:
        return "Needs rerun", False
    return "Not run", False


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


def _workbook_cell(value: Any) -> Any:
    """Return an Excel-safe scalar while preserving ordinary numeric cells."""
    if value is None:
        return ""
    if isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return ""
        return value
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    try:
        return json.dumps(value, allow_nan=False, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _engineering_workbook_bytes(
    spec: dict[str, Any],
    result: Any,
    stream_table: pd.DataFrame,
    equipment_table: pd.DataFrame,
    constraint_table: pd.DataFrame,
) -> bytes:
    """Build a review-ready Excel workbook from one solved NeqSim case."""
    fluid = spec["fluid"]
    process_steps = {step["name"]: step for step in spec["process"]}
    total_power_kw = _kpi_value(result, "total_power_kW")
    total_duty_kw = _kpi_value(result, "total_duty_kW")
    mass_balance_pct = _kpi_value(result, "mass_balance_pct")
    feed_tonnes_per_hour = float(fluid["total_flow"]) / 1000.0
    specific_energy = None
    if total_power_kw is not None and feed_tonnes_per_hour > 0.0:
        specific_energy = total_power_kw / feed_tonnes_per_hour

    case_summary = pd.DataFrame(
        [
            ("Case", "Name", spec["name"], ""),
            ("Case", "Template", TEMPLATE_NAME, ""),
            ("Case", "Simulation mode", "Steady state", ""),
            ("Thermodynamics", "Equation of state", str(fluid["eos_model"]).upper(), ""),
            ("Thermodynamics", "Mixing rule", fluid["mixing_rule"], "NeqSim rule"),
            ("Fluid", "Composition basis", fluid["composition_basis"], "mole fraction"),
            ("Feed", "Temperature", fluid["temperature_C"], "°C"),
            ("Feed", "Pressure", fluid["pressure_bara"], "bara absolute"),
            ("Feed", "Mass flow", fluid["total_flow"], fluid["flow_unit"]),
            (
                "Compressor stage 1",
                "Discharge pressure",
                process_steps["compressor stage 1"]["params"][
                    "outlet_pressure_bara"
                ],
                "bara absolute",
            ),
            (
                "Compressor stage 2",
                "Discharge pressure",
                process_steps["compressor stage 2"]["params"][
                    "outlet_pressure_bara"
                ],
                "bara absolute",
            ),
            (
                "Compressors",
                "Isentropic efficiency",
                process_steps["compressor stage 1"]["params"][
                    "isentropic_efficiency"
                ],
                "fraction",
            ),
            (
                "Intercooler",
                "Outlet temperature",
                process_steps["intercooler"]["params"]["outlet_temperature_C"],
                "°C",
            ),
            (
                "Export cooler",
                "Outlet temperature",
                process_steps["export cooler"]["params"]["outlet_temperature_C"],
                "°C",
            ),
            ("Solver", "State represented by workbook", "Solved", ""),
        ],
        columns=["Section", "Parameter", "Value", "Unit / basis"],
    )
    kpi_table = pd.DataFrame(
        [
            ("Total compressor power", total_power_kw, "kW"),
            ("Total cooling duty magnitude", total_duty_kw, "kW"),
            ("Specific compression energy", specific_energy, "kWh/t feed"),
            ("Total mass imbalance", mass_balance_pct, "%"),
        ],
        columns=["KPI", "Value", "Unit"],
    )
    composition_table = pd.DataFrame(
        {
            "Component": list(fluid["components"]),
            "Mole fraction [-]": list(fluid["components"].values()),
        }
    )
    assumptions = list(spec.get("assumptions", []))
    assumptions_table = pd.DataFrame(
        {
            "Type": ["Assumption"] * len(assumptions) + ["Limitation"],
            "Statement": assumptions
            + [
                "Results support screening and engineering studies; "
                "they are not design certification."
            ],
        }
    )
    sheet_frames = {
        "Case Summary": case_summary,
        "KPIs": kpi_table,
        "Composition": composition_table,
        "Streams": stream_table,
        "Equipment": equipment_table,
        "Validation": constraint_table,
        "Assumptions": assumptions_table,
    }

    output = BytesIO()
    with pd.ExcelWriter(
        output,
        engine="xlsxwriter",
        engine_kwargs={
            "options": {
                "strings_to_formulas": False,
                "strings_to_urls": False,
                "nan_inf_to_errors": True,
            }
        },
    ) as writer:
        workbook = writer.book
        workbook.set_properties(
            {
                "title": f"{spec['name']} · NeqSim engineering workbook",
                "subject": "Process Flowsheet Studio solved-case results",
                "author": "NeqSim Process Flowsheet Studio",
            }
        )
        header_format = workbook.add_format(
            {
                "bold": True,
                "font_color": "white",
                "bg_color": "#1F4E78",
                "border": 1,
            }
        )
        for sheet_name, source_frame in sheet_frames.items():
            frame = source_frame.copy()
            for column in frame.columns:
                frame[column] = frame[column].map(_workbook_cell)
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes(1, 0)
            if len(frame.columns) > 0:
                worksheet.autofilter(0, 0, len(frame), len(frame.columns) - 1)
            for column_index, column_name in enumerate(frame.columns):
                worksheet.write(0, column_index, column_name, header_format)
                values = [str(column_name)] + [
                    str(value) for value in frame[column_name].tolist()
                ]
                width = min(max(len(value) for value in values) + 2, 60)
                worksheet.set_column(column_index, column_index, width)

    return output.getvalue()


def _case_history_record(
    spec: dict[str, Any],
    result: Any,
    signature: str,
) -> dict[str, Any]:
    """Create a compact comparison record from one successful NeqSim solve."""
    total_power_kw = _kpi_value(result, "total_power_kW")
    total_duty_kw = _kpi_value(result, "total_duty_kW")
    mass_balance_pct = _kpi_value(result, "mass_balance_pct")
    feed_flow_kg_hr = float(spec["fluid"]["total_flow"])
    feed_tonnes_per_hour = feed_flow_kg_hr / 1000.0
    specific_energy_kwh_t = None
    if total_power_kw is not None and feed_tonnes_per_hour > 0.0:
        specific_energy_kwh_t = total_power_kw / feed_tonnes_per_hour

    constraint_statuses = {
        str(getattr(constraint, "status", "")).upper()
        for constraint in result.constraints
    }
    if "VIOLATION" in constraint_statuses:
        validation_status = "VIOLATION"
    elif "WARN" in constraint_statuses:
        validation_status = "WARN"
    else:
        validation_status = "OK"

    process = spec["process"]
    return {
        "_signature": signature,
        "_spec": json.loads(json.dumps(spec, allow_nan=False)),
        "Case ID": signature[:8],
        "Case": spec["name"],
        "EOS": str(spec["fluid"]["eos_model"]).upper(),
        "Components": len(spec["fluid"]["components"]),
        "Feed temperature [°C]": float(spec["fluid"]["temperature_C"]),
        "Feed pressure [bara]": float(spec["fluid"]["pressure_bara"]),
        "Feed flow [kg/hr]": feed_flow_kg_hr,
        "Stage 1 pressure [bara]": float(
            process[2]["params"]["outlet_pressure_bara"]
        ),
        "Stage 2 pressure [bara]": float(
            process[5]["params"]["outlet_pressure_bara"]
        ),
        "Isentropic efficiency [-]": float(
            process[2]["params"]["isentropic_efficiency"]
        ),
        "Compressor power [kW]": total_power_kw,
        "Cooling duty magnitude [kW]": total_duty_kw,
        "Specific energy [kWh/t]": specific_energy_kwh_t,
        "Mass imbalance [%]": mass_balance_pct,
        "Validation": validation_status,
    }


def _upsert_case_history(
    history: Any,
    record: dict[str, Any],
    max_cases: int = MAX_CASE_HISTORY,
) -> list[dict[str, Any]]:
    """Store one unique solved case while bounding session memory."""
    if max_cases < 1:
        raise ValueError("max_cases must be at least one.")
    signature = record.get("_signature")
    if not isinstance(signature, str) or not signature:
        raise ValueError("A solved case record must have a signature.")

    history_items = history if isinstance(history, list) else []
    cleaned_history = [
        dict(item)
        for item in history_items
        if isinstance(item, dict)
        and isinstance(item.get("_signature"), str)
        and item.get("_signature") != signature
    ]
    cleaned_history.append(dict(record))
    return cleaned_history[-max_cases:]


def _percent_delta(value: Any, baseline: Any) -> float | None:
    """Return a finite percentage delta, or None for an unusable baseline."""
    if value is None or baseline is None:
        return None
    value_float = _finite_float(value, "Case result")
    baseline_float = _finite_float(baseline, "Baseline result")
    if abs(baseline_float) <= 1.0e-12:
        return None
    return 100.0 * (value_float - baseline_float) / abs(baseline_float)


def _case_comparison_dataframe(
    history: Any,
    baseline_signature: str,
) -> pd.DataFrame:
    """Build a workbook-style solved-case table with baseline KPI deltas."""
    history_items = history if isinstance(history, list) else []
    records = [
        dict(item)
        for item in history_items
        if isinstance(item, dict)
        and isinstance(item.get("_signature"), str)
    ]
    if not records:
        return pd.DataFrame()

    baseline = next(
        (
            record
            for record in records
            if record["_signature"] == baseline_signature
        ),
        records[0],
    )
    comparison_rows = []
    for record in records:
        row = {
            key: value
            for key, value in record.items()
            if not key.startswith("_")
        }
        row["Baseline"] = (
            "Yes" if record["_signature"] == baseline["_signature"] else ""
        )
        row["Power Δ vs baseline [%]"] = _percent_delta(
            record.get("Compressor power [kW]"),
            baseline.get("Compressor power [kW]"),
        )
        row["Duty Δ vs baseline [%]"] = _percent_delta(
            record.get("Cooling duty magnitude [kW]"),
            baseline.get("Cooling duty magnitude [kW]"),
        )
        row["Specific energy Δ vs baseline [%]"] = _percent_delta(
            record.get("Specific energy [kWh/t]"),
            baseline.get("Specific energy [kWh/t]"),
        )
        comparison_rows.append(row)
    return pd.DataFrame(comparison_rows)


def _case_history_label(record: dict[str, Any]) -> str:
    """Return a safe selector label for a retained solved case."""
    case_name = str(record.get("Case") or "Unnamed case")
    signature = str(record.get("_signature") or "")[:8]
    try:
        feed_flow = _finite_float(
            record.get("Feed flow [kg/hr]"),
            "Feed flow",
        )
        feed_flow_text = f"{feed_flow:,.0f} kg/hr"
    except ValueError:
        feed_flow_text = "unknown flow"
    return f"{case_name} · {feed_flow_text} · {signature}"


def _load_case_history_record(
    record: Any,
) -> tuple[dict[str, Any], pd.DataFrame, list[str]]:
    """Validate a retained solved-case specification for control restoration."""
    if not isinstance(record, dict) or not isinstance(record.get("_spec"), dict):
        raise ValueError(
            "This retained result predates reusable case restoration. "
            "Solve it again before restoring it."
        )
    return _load_case_controls(record["_spec"])


def _template_object_label(object_name: str) -> str:
    """Return a searchable palette label with the object's engineering type."""
    display_name, object_type = TEMPLATE_OBJECTS[object_name]
    return f"{display_name} · {object_type}"


def _render_object_property_editor() -> None:
    """Render supported properties for one selected template object."""
    st.markdown("#### Selected-object properties")
    st.caption(
        "Search the current flowsheet, select one object, and edit its supported "
        "steady-state properties. Units are shown on every editable value."
    )
    selected_object = st.selectbox(
        "Find flowsheet object",
        options=list(TEMPLATE_OBJECTS),
        format_func=_template_object_label,
        key="flowsheet_selected_object",
        help="Type while the list is open to search by object name or type.",
    )
    display_name, object_type = TEMPLATE_OBJECTS[selected_object]
    st.write(f"**Selected:** {display_name}")
    st.caption(f"Object type: {object_type}")

    if selected_object == "compressor stage 1":
        st.number_input(
            "Discharge pressure [bara]",
            min_value=1.0,
            max_value=500.0,
            step=1.0,
            key="flowsheet_stage_1_pressure_bara",
        )
        st.slider(
            "Shared isentropic efficiency [-]",
            min_value=0.50,
            max_value=0.95,
            step=0.01,
            key="flowsheet_isentropic_efficiency",
            help="The current template applies one efficiency to both stages.",
        )
    elif selected_object == "compressor stage 2":
        st.number_input(
            "Discharge pressure [bara]",
            min_value=1.0,
            max_value=500.0,
            step=1.0,
            key="flowsheet_stage_2_pressure_bara",
        )
        st.slider(
            "Shared isentropic efficiency [-]",
            min_value=0.50,
            max_value=0.95,
            step=0.01,
            key="flowsheet_isentropic_efficiency",
            help="The current template applies one efficiency to both stages.",
        )
    elif selected_object == "intercooler":
        st.number_input(
            "Outlet temperature [°C]",
            min_value=-50.0,
            max_value=150.0,
            step=1.0,
            key="flowsheet_intercooler_temperature_c",
        )
    elif selected_object == "export cooler":
        st.number_input(
            "Outlet temperature [°C]",
            min_value=-50.0,
            max_value=150.0,
            step=1.0,
            key="flowsheet_export_temperature_c",
        )
    elif selected_object == "feed gas":
        st.info(
            "Feed temperature, absolute pressure, mass flow, equation of state, "
            "and molar composition are edited in the fluid basis."
        )
    else:
        st.info(
            "This separator performs an equilibrium split at its inlet conditions. "
            "The current template has no independent separator set point."
        )

    st.caption(
        "Property edits update the structured case specification. Run NeqSim to "
        "solve the edited case and refresh Process Chat."
    )


st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="images/neqsimlogocircleflat.png",
    layout="wide",
)
apply_theme()
theme_toggle()
_initialize_case_controls()

st.title("🏭 Process Flowsheet Studio")
st.markdown(
    """
Build and solve a reproducible NeqSim process case using structured engineering
inputs. Version 1 provides an inlet-separation and two-stage gas-compression
template. The solved process is shared with **Process Chat** for further
what-if studies.
"""
)

with st.sidebar:
    st.divider()
    st.subheader("🏭 Flowsheet case")
    st.caption(TEMPLATE_NAME)
    st.write("**Mode:** Steady state")
    solver_status_placeholder = st.empty()
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

with st.expander("Open a reusable Studio case", expanded=False):
    st.caption(
        "Import a JSON case previously downloaded from this page. "
        "The file is validated before it can replace the active controls."
    )
    uploaded_case = st.file_uploader(
        "Studio case JSON",
        type=["json"],
        key="flowsheet_case_upload",
    )
    load_uploaded_case = st.button(
        "Open case",
        disabled=uploaded_case is None,
        use_container_width=True,
        key="flowsheet_open_case",
    )
    if load_uploaded_case:
        try:
            if uploaded_case.size > MAX_CASE_FILE_BYTES:
                raise ValueError("The case file cannot exceed 1 MB.")
            case_data = json.loads(uploaded_case.getvalue().decode("utf-8-sig"))
            imported_controls, imported_composition, import_warnings = (
                _load_case_controls(case_data)
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as import_error:
            st.error(f"Case import failed: {import_error}")
        else:
            _apply_imported_case(
                imported_controls,
                imported_composition,
                import_warnings,
            )
            st.rerun()

import_notice = st.session_state.pop("flowsheet_import_notice", None)
if import_notice:
    st.success(import_notice)

st.subheader("1. Case setup")
case_name = st.text_input(
    "Case name",
    help="A reusable engineering case name stored with the downloaded model.",
    key="flowsheet_case_name",
)
st.caption(f"Template: {TEMPLATE_NAME}")

st.markdown("#### Fluid and operating basis")
fluid_col, object_col = st.columns(2)

with fluid_col:
    eos_model = st.selectbox(
        "Equation of state",
        options=SUPPORTED_EOS_MODELS,
        format_func=lambda value: value.upper(),
        help="Mixing rule 2 is used for cubic/association equations of state.",
        key="flowsheet_eos_model",
    )
    feed_temperature_c = st.number_input(
        "Feed temperature [°C]",
        min_value=-100.0,
        max_value=200.0,
        step=1.0,
        key="flowsheet_feed_temperature_c",
    )
    feed_pressure_bara = st.number_input(
        "Feed pressure [bara]",
        min_value=1.0,
        max_value=500.0,
        step=1.0,
        key="flowsheet_feed_pressure_bara",
    )
    feed_flow_kg_hr = st.number_input(
        "Feed mass flow [kg/hr]",
        min_value=1.0,
        max_value=10_000_000.0,
        step=1_000.0,
        key="flowsheet_feed_flow_kg_hr",
    )

with object_col:
    _render_object_property_editor()

stage_1_pressure_bara = float(
    st.session_state["flowsheet_stage_1_pressure_bara"]
)
stage_2_pressure_bara = float(
    st.session_state["flowsheet_stage_2_pressure_bara"]
)
isentropic_efficiency = float(
    st.session_state["flowsheet_isentropic_efficiency"]
)
intercooler_temperature_c = float(
    st.session_state["flowsheet_intercooler_temperature_c"]
)
export_temperature_c = float(
    st.session_state["flowsheet_export_temperature_c"]
)

st.markdown("**Feed composition**")
composition_table = st.data_editor(
    st.session_state["flowsheet_composition_source"],
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
    key=(
        "flowsheet_composition_editor_"
        f"{st.session_state['flowsheet_composition_revision']}"
    ),
)

preview_composition: dict[str, float] = {}
preview_total = 0.0
draft_case_spec: dict[str, Any] | None = None
draft_warnings: list[str] = []
current_case_signature: str | None = None
draft_error: str | None = None

try:
    preview_composition, preview_total = _clean_composition(composition_table)
except ValueError as preview_error:
    draft_error = str(preview_error)
    st.warning(draft_error)
else:
    total_col, count_col = st.columns(2)
    total_col.metric("Entered mole-fraction sum", f"{preview_total:.6f}")
    count_col.metric("Active components", len(preview_composition))
    try:
        draft_case_spec = _build_case_spec(
            case_name=case_name.strip() or "Gas Compression Case",
            composition=preview_composition,
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
        draft_warnings = _validate_case(draft_case_spec, preview_total)
        current_case_signature = _case_signature(draft_case_spec, preview_total)
    except ValueError as validation_error:
        draft_case_spec = None
        draft_error = str(validation_error)
        st.warning(draft_error)

stored_state = st.session_state.get(CASE_STATE_KEY)
solver_status, results_are_current = _solver_status(
    current_signature=current_case_signature,
    stored_state=stored_state,
    has_result=bool(st.session_state.get(RESULT_STATE_KEY)),
    failure_signature=st.session_state.get(FAILURE_SIGNATURE_STATE_KEY),
)
solver_status_placeholder.write(f"**Solver:** {solver_status}")

run_case = st.button(
    "▶ Run NeqSim flowsheet",
    type="primary",
    use_container_width=True,
)

if run_case:
    try:
        if draft_case_spec is None or current_case_signature is None:
            raise ValueError(draft_error or "The current case inputs are invalid.")
        case_spec = draft_case_spec
        case_warnings = draft_warnings

        solver_status_placeholder.write("**Solver:** Solving")
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
            "signature": current_case_signature,
        }
        st.session_state[CASE_STATE_KEY] = state
        st.session_state[RESULT_STATE_KEY] = True
        st.session_state.pop(FAILURE_SIGNATURE_STATE_KEY, None)
        solved_case_record = _case_history_record(
            case_spec,
            result,
            current_case_signature,
        )
        st.session_state[CASE_HISTORY_STATE_KEY] = _upsert_case_history(
            st.session_state.get(CASE_HISTORY_STATE_KEY),
            solved_case_record,
        )
        results_are_current = True
        solver_status = "Solved"

        # Shared state used by the existing Process Chat page.
        st.session_state["process_model"] = model
        st.session_state["process_model_name"] = "process_flowsheet_studio.neqsim"
        if model_bytes:
            st.session_state["process_model_bytes"] = model_bytes

        solver_status_placeholder.write("**Solver:** Solved")
        st.success("The NeqSim flowsheet solved and is ready for review.")
    except Exception as exc:
        if current_case_signature is not None:
            st.session_state[FAILURE_SIGNATURE_STATE_KEY] = current_case_signature
        results_are_current = False
        solver_status = "Failed"
        solver_status_placeholder.write("**Solver:** Failed")
        st.error(f"Flowsheet calculation failed: {exc}")
        with st.expander("Technical error details", expanded=False):
            st.code(traceback.format_exc())

has_stored_result = bool(
    st.session_state.get(RESULT_STATE_KEY)
    and isinstance(st.session_state.get(CASE_STATE_KEY), dict)
)
if has_stored_result and not results_are_current:
    if solver_status == "Failed":
        stale_reason = "The current calculation failed."
    elif solver_status == "Invalid inputs":
        stale_reason = "The current inputs are invalid."
    else:
        stale_reason = "The inputs changed after the last successful calculation."
    st.info(
        f"{stale_reason} The last solved results are retained but hidden until "
        "their exact inputs are restored or the current case solves successfully. "
        "Process Chat continues to reference the last solved model."
    )

if results_are_current and has_stored_result:
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

    case_history = st.session_state.get(CASE_HISTORY_STATE_KEY, [])
    history_records = [
        record
        for record in case_history
        if isinstance(record, dict)
        and isinstance(record.get("_signature"), str)
    ]
    if history_records:
        st.markdown("#### What-if case comparison")
        st.caption(
            "Each unique, successfully solved NeqSim case is retained in this "
            f"session (up to {MAX_CASE_HISTORY}). Select a baseline to compare "
            "power, cooling duty, and specific energy."
        )
        record_by_signature = {
            record["_signature"]: record for record in history_records
        }
        history_signatures = list(record_by_signature)
        baseline_state_key = "flowsheet_case_history_baseline"
        if st.session_state.get(baseline_state_key) not in history_signatures:
            st.session_state[baseline_state_key] = history_signatures[0]
        baseline_signature = st.selectbox(
            "Comparison baseline",
            options=history_signatures,
            format_func=lambda signature: _case_history_label(
                record_by_signature[signature]
            ),
            key=baseline_state_key,
        )
        comparison_table = _case_comparison_dataframe(
            history_records,
            baseline_signature,
        )
        comparison_formats = {
            "Feed temperature [°C]": "{:.2f}",
            "Feed pressure [bara]": "{:.2f}",
            "Feed flow [kg/hr]": "{:,.2f}",
            "Stage 1 pressure [bara]": "{:.2f}",
            "Stage 2 pressure [bara]": "{:.2f}",
            "Isentropic efficiency [-]": "{:.3f}",
            "Compressor power [kW]": "{:,.2f}",
            "Cooling duty magnitude [kW]": "{:,.2f}",
            "Specific energy [kWh/t]": "{:.3f}",
            "Mass imbalance [%]": "{:.6f}",
            "Power Δ vs baseline [%]": "{:+.3f}",
            "Duty Δ vs baseline [%]": "{:+.3f}",
            "Specific energy Δ vs baseline [%]": "{:+.3f}",
        }
        st.dataframe(
            comparison_table.style.format(
                comparison_formats,
                na_rep="—",
            ),
            use_container_width=True,
            hide_index=True,
        )
        action_cols = st.columns(2)
        selected_history_record = record_by_signature[baseline_signature]
        restore_available = isinstance(
            selected_history_record.get("_spec"),
            dict,
        )
        restore_case = action_cols[0].button(
            "Restore selected case inputs",
            disabled=not restore_available,
            help=(
                "Load this solved case into the editable controls. "
                "Run NeqSim again to rebuild its process model."
            ),
            use_container_width=True,
        )
        action_cols[1].download_button(
            "Download case comparison CSV",
            data=comparison_table.to_csv(index=False),
            file_name="process_flowsheet_case_comparison.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if restore_case:
            try:
                restored_controls, restored_composition, restored_warnings = (
                    _load_case_history_record(selected_history_record)
                )
            except ValueError as restore_error:
                st.error(f"Case restoration failed: {restore_error}")
            else:
                _apply_imported_case(
                    restored_controls,
                    restored_composition,
                    restored_warnings,
                )
                st.rerun()

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
    workbook_bytes = None
    workbook_error = None
    try:
        workbook_bytes = _engineering_workbook_bytes(
            spec,
            result,
            stream_table,
            equipment_table,
            constraint_table,
        )
    except Exception as export_error:
        workbook_error = str(export_error)

    download_cols = st.columns(4)
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
    if workbook_bytes:
        download_cols[3].download_button(
            "Download engineering workbook",
            data=workbook_bytes,
            file_name="process_flowsheet_engineering_workbook.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True,
        )
    else:
        download_cols[3].info(
            "Engineering workbook was unavailable."
            + (f" {workbook_error}" if workbook_error else "")
        )

    st.info(
        "This solved process is also available in the current session under "
        "Process Chat for natural-language what-if analysis."
    )
