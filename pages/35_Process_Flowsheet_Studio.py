"""Process Flowsheet Studio for structured, reproducible NeqSim studies."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from io import BytesIO
from time import perf_counter
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

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from process_chat.runtime_imports import import_local_symbols  # noqa: E402
from process_chat.process_builder import ProcessBuilder  # noqa: E402
from theme import apply_theme, theme_toggle  # noqa: E402


_SOLVER_DIAGNOSTIC_SYMBOL_NAMES = (
    "aggregate_convergence",
    "aggregate_energy_balance",
    "aggregate_unit_balances",
    "aggregate_validation_status",
    "component_balance_rows",
    "convergence_rows",
    "energy_transfer_rows",
    "material_boundary_rows",
    "solved_feed_flow_kg_hr",
    "unit_balance_rows",
)
globals().update(
    import_local_symbols(
        "process_chat.solver_diagnostics",
        _SOLVER_DIAGNOSTIC_SYMBOL_NAMES,
        project_root=_PROJECT_ROOT,
    )
)


_EDITOR_SYMBOL_NAMES = (
    "add_catalog_unit",
    "apply_graph_draft",
    "build_graph_draft_dot",
    "clone_material_inlet",
    "connect_graph_ports",
    "create_graph_draft",
    "create_graph_history",
    "disconnect_graph_connection",
    "extend_material_path",
    "graph_connection_rows",
    "graph_history_status",
    "graph_port_rows",
    "inlet_composition_property_rows",
    "inlet_condition_property_rows",
    "inline_unit_catalog",
    "inline_unit_catalog_rows",
    "inline_unit_property_rows",
    "insert_inline_unit_on_connection",
    "insert_mixer_on_connection",
    "material_connection_rows",
    "process_unit_property_rows",
    "record_graph_history",
    "redo_graph_history",
    "remove_material_inlet",
    "remove_inline_unit",
    "replace_inline_unit",
    "reroute_graph_connection",
    "rename_material_inlet",
    "rename_inline_unit",
    "undo_graph_history",
    "update_inlet_composition",
    "update_inlet_conditions",
    "update_inline_unit_properties",
    "validate_catalog_unit",
    "validate_starter_unit_projection",
)
globals().update(
    import_local_symbols(
        "process_chat.flowsheet_editor",
        _EDITOR_SYMBOL_NAMES,
        project_root=_PROJECT_ROOT,
    )
)


PAGE_TITLE = "Process Flowsheet Studio"
TEMPLATE_NAME = "Inlet separation and two-stage gas compression"
CASE_STATE_KEY = "flowsheet_studio_case"
RESULT_STATE_KEY = "flowsheet_studio_result"
FAILURE_SIGNATURE_STATE_KEY = "flowsheet_studio_failure_signature"
CASE_HISTORY_STATE_KEY = "flowsheet_studio_case_history"
CASE_HISTORY_BASELINE_STATE_KEY = "flowsheet_case_history_baseline"
CASE_NOTICE_STATE_KEY = "flowsheet_case_notice"
GRAPH_DRAFT_STATE_KEY = "flowsheet_studio_graph_draft"
GRAPH_HISTORY_STATE_KEY = "flowsheet_studio_graph_history"
STUDIO_PROCESS_MODEL_NAME = "process_flowsheet_studio.neqsim"
LEGACY_CASE_SCHEMA_VERSION = 1
SHARED_FLUID_CASE_SCHEMA_VERSION = 2
CASE_SCHEMA_VERSION = 3
BASE_FLUID_PACKAGE_ID = "base-fluid"
PRIMARY_INLET_ID = "feed-gas"
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
TEMPLATE_UNIT_IDS = {
    "inlet scrubber": "inlet-scrubber",
    "compressor stage 1": "compressor-stage-1",
    "intercooler": "intercooler",
    "interstage scrubber": "interstage-scrubber",
    "compressor stage 2": "compressor-stage-2",
    "export cooler": "export-cooler",
}
TEMPLATE_PROPERTY_CONTROLS = {
    "compressor stage 1": {
        "outlet_pressure_bara": {
            "state_key": "flowsheet_stage_1_pressure_bara",
            "minimum": 1.0,
            "maximum": 500.0,
        },
        "isentropic_efficiency": {
            "state_key": "flowsheet_stage_1_isentropic_efficiency",
            "minimum": 0.50,
            "maximum": 0.95,
        },
    },
    "compressor stage 2": {
        "outlet_pressure_bara": {
            "state_key": "flowsheet_stage_2_pressure_bara",
            "minimum": 1.0,
            "maximum": 500.0,
        },
        "isentropic_efficiency": {
            "state_key": "flowsheet_stage_2_isentropic_efficiency",
            "minimum": 0.50,
            "maximum": 0.95,
        },
    },
    "intercooler": {
        "outlet_temperature_C": {
            "state_key": "flowsheet_intercooler_temperature_c",
            "minimum": -50.0,
            "maximum": 150.0,
        },
        "pressure_drop_bar": {
            "state_key": "flowsheet_intercooler_pressure_drop_bar",
            "minimum": 0.0,
            "maximum": 50.0,
        },
    },
    "export cooler": {
        "outlet_temperature_C": {
            "state_key": "flowsheet_export_temperature_c",
            "minimum": -50.0,
            "maximum": 150.0,
        },
        "pressure_drop_bar": {
            "state_key": "flowsheet_export_pressure_drop_bar",
            "minimum": 0.0,
            "maximum": 50.0,
        },
    },
}

CONTROL_DEFAULTS = {
    "flowsheet_case_name": "Gas Compression Case",
    "flowsheet_eos_model": "srk",
    "flowsheet_feed_temperature_c": 30.0,
    "flowsheet_feed_pressure_bara": 50.0,
    "flowsheet_feed_flow_kg_hr": 100_000.0,
    "flowsheet_stage_1_pressure_bara": 80.0,
    "flowsheet_stage_2_pressure_bara": 130.0,
    "flowsheet_stage_1_isentropic_efficiency": 0.78,
    "flowsheet_stage_2_isentropic_efficiency": 0.78,
    "flowsheet_intercooler_temperature_c": 35.0,
    "flowsheet_intercooler_pressure_drop_bar": 0.0,
    "flowsheet_export_temperature_c": 40.0,
    "flowsheet_export_pressure_drop_bar": 0.0,
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


def _clear_studio_runtime(clear_history: bool) -> None:
    """Clear calculated Studio state without deleting an unrelated Chat model."""
    for key in (
        CASE_STATE_KEY,
        RESULT_STATE_KEY,
        FAILURE_SIGNATURE_STATE_KEY,
    ):
        st.session_state.pop(key, None)
    if clear_history:
        st.session_state[CASE_HISTORY_STATE_KEY] = []
        st.session_state.pop(CASE_HISTORY_BASELINE_STATE_KEY, None)
    if st.session_state.get("process_model_name") == STUDIO_PROCESS_MODEL_NAME:
        st.session_state.pop("process_model", None)
        st.session_state.pop("process_model_name", None)
        st.session_state.pop("process_model_bytes", None)


def _start_new_case() -> None:
    """Start a clean default case after the user confirms destructive reset."""
    current_revision = st.session_state.get(
        "flowsheet_composition_revision",
        0,
    )
    try:
        next_revision = int(current_revision) + 1
    except (TypeError, ValueError):
        next_revision = 1

    for key, value in CONTROL_DEFAULTS.items():
        st.session_state[key] = value
    st.session_state["flowsheet_composition_source"] = DEFAULT_COMPOSITION.copy()
    st.session_state["flowsheet_composition_revision"] = next_revision
    st.session_state["flowsheet_selected_object"] = "feed gas"
    st.session_state.pop(GRAPH_DRAFT_STATE_KEY, None)
    st.session_state.pop(GRAPH_HISTORY_STATE_KEY, None)
    _clear_studio_runtime(clear_history=True)

    for key in (
        "flowsheet_case_upload",
        "flowsheet_confirm_new_case",
        "flowsheet_import_notice",
    ):
        st.session_state.pop(key, None)
    st.session_state[CASE_NOTICE_STATE_KEY] = (
        "New case started from the validated gas-compression template. "
        "Review the inputs and run the NeqSim flowsheet."
    )


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


def _graph_object_name(record: dict[str, Any], fallback_id: str) -> str:
    """Return a safe UI label for a graph object with legacy metadata."""
    clean_fallback = str(fallback_id).strip()
    raw_name = record.get("name")
    if raw_name is None:
        return clean_fallback
    return str(raw_name).strip() or clean_fallback


def _material_output_selection_label(unit: dict[str, Any]) -> str:
    """Describe declared material outlets for a graph-edit success notice."""
    unit_id = str(unit.get("id", "")).strip()
    ports = unit.get("ports")
    material_outputs = (
        ports.get("material_out")
        if isinstance(ports, dict)
        else None
    )
    output_labels = [
        f"'{unit_id}:{str(port).strip()}'"
        for port in material_outputs or []
        if str(port).strip()
    ]
    if not output_labels:
        return f"an available outlet on '{unit_id}'"
    if len(output_labels) == 1:
        return output_labels[0]
    return "one of " + ", ".join(output_labels)


def _required_identifier(value: Any, field_label: str) -> str:
    """Return a non-null, non-blank persisted object identifier."""
    if value is None:
        raise ValueError(f"{field_label} cannot be empty.")
    result = str(value).strip()
    if not result:
        raise ValueError(f"{field_label} cannot be empty.")
    return result


def _graph_name_set(
    records: list[Any],
    *,
    casefold: bool = False,
) -> set[str]:
    """Return non-null, non-blank graph names for collision checks."""
    if not isinstance(records, list):
        return set()
    result: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        raw_name = record.get("name")
        if raw_name is None:
            continue
        name = str(raw_name).strip()
        if name:
            result.add(name.casefold() if casefold else name)
    return result


def _terminal_material_stream_names(
    units: list[Any],
    connections: list[Any],
) -> set[str]:
    """Return names the native builder will assign to product boundaries."""
    connected_outputs: set[tuple[str, str]] = set()
    for connection in connections:
        if not isinstance(connection, dict):
            continue
        if str(connection.get("type", "")).strip().lower() != "material":
            continue
        source = connection.get("source")
        if not isinstance(source, dict):
            continue
        if str(source.get("kind", "")).strip().lower() != "unit":
            continue
        connected_outputs.add(
            (
                str(source.get("id", "")).strip(),
                str(source.get("port", "")).strip().lower(),
            )
        )

    result: set[str] = set()
    for unit in units:
        if not isinstance(unit, dict):
            continue
        unit_id = str(unit.get("id", "")).strip()
        raw_name = unit.get("name")
        unit_name = "" if raw_name is None else str(raw_name).strip()
        ports = unit.get("ports")
        if not unit_id or not unit_name or not isinstance(ports, dict):
            continue
        material_outputs = ports.get("material_out")
        if not isinstance(material_outputs, list):
            continue
        for raw_port in material_outputs:
            output_port = str(raw_port).strip().lower()
            if (
                output_port
                and (unit_id, output_port) not in connected_outputs
            ):
                result.add(f"{unit_name} [{output_port}] product")
    return result


def _terminal_name_conflicts(
    records: list[Any],
    terminal_names: set[str],
) -> list[str]:
    """Return graph object names reserved by synthesized product streams."""
    terminal_name_keys = {
        str(name).strip().casefold()
        for name in terminal_names
        if str(name).strip()
    }
    conflicts: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        raw_name = record.get("name")
        if raw_name is None:
            continue
        name = str(raw_name).strip()
        name_key = name.casefold()
        if name and name_key in terminal_name_keys:
            conflicts.setdefault(name_key, name)
    return [conflicts[name_key] for name_key in sorted(conflicts)]


def _validate_graph_solve_readiness(case_spec: dict[str, Any]) -> None:
    """Reject graph drafts whose independent feeds are not yet consumed."""
    inlets = case_spec.get("inlets")
    connections = case_spec.get("connections")
    if not isinstance(inlets, list) or not isinstance(connections, list):
        raise ValueError("The graph requires inlet and connection arrays.")

    connected_inlets: set[str] = set()
    for connection in connections:
        if not isinstance(connection, dict):
            continue
        source = connection.get("source")
        if not isinstance(source, dict):
            continue
        if (
            str(source.get("kind", "")).strip().lower() == "inlet"
            and str(source.get("port", "")).strip().lower() == "out"
        ):
            connected_inlets.add(str(source.get("id", "")).strip())

    disconnected = [
        str(inlet.get("id", "")).strip()
        for inlet in inlets
        if isinstance(inlet, dict)
        and str(inlet.get("id", "")).strip()
        and str(inlet.get("id", "")).strip() not in connected_inlets
    ]
    if disconnected:
        raise ValueError(
            "Connect every independent feed before solving; disconnected "
            f"inlet(s): {', '.join(disconnected)}."
        )


def _secondary_inlet_map(
    inlets: list[Any],
    primary_inlet_id: str,
) -> dict[str, dict[str, Any]]:
    """Return addressable secondary inlets without trusting imported IDs."""
    if not isinstance(inlets, list):
        return {}

    clean_primary_id = str(primary_inlet_id).strip()
    result: dict[str, dict[str, Any]] = {}
    for inlet in inlets:
        if not isinstance(inlet, dict):
            continue
        raw_inlet_id = inlet.get("id")
        if raw_inlet_id is None:
            continue
        inlet_id = str(raw_inlet_id).strip()
        if not inlet_id or inlet_id == clean_primary_id:
            continue
        result[inlet_id] = inlet
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


def _validate_fluid_package_integrity(
    fluid_packages: list[Any],
    inlets: list[Any],
) -> None:
    """Validate shared characterization and any number of inlet conditions."""
    if not fluid_packages:
        raise ValueError("At least one fluid package is required.")
    if not inlets:
        raise ValueError("At least one inlet stream is required.")

    packages_by_id: dict[str, dict[str, Any]] = {}
    registry_by_package: dict[str, set[str]] = {}
    component_definitions: dict[str, dict[str, Any]] = {}

    for package_index, package in enumerate(fluid_packages):
        if not isinstance(package, dict):
            raise ValueError(f"fluid_packages[{package_index}] must be an object.")
        package_id = _required_identifier(
            package.get("id"),
            f"fluid_packages[{package_index}].id",
        )
        if package_id in packages_by_id:
            raise ValueError("Fluid-package ids must be unique.")
        packages_by_id[package_id] = package

        eos_model = str(package.get("eos_model", "")).lower().strip()
        if eos_model not in SUPPORTED_EOS_MODELS:
            raise ValueError(
                f"Fluid package '{package_id}' has an unsupported equation of state."
            )
        mixing_rule = _finite_float(
            package.get("mixing_rule"),
            f"fluid package '{package_id}' mixing_rule",
        )
        if not mixing_rule.is_integer() or mixing_rule < 0.0:
            raise ValueError(
                f"Fluid package '{package_id}' mixing_rule must be a non-negative "
                "integer."
            )

        interaction_parameters = package.get("binary_interaction_parameters")
        if not isinstance(interaction_parameters, dict):
            raise ValueError(
                f"Fluid package '{package_id}' interaction parameters must be an "
                "object."
            )
        if not str(interaction_parameters.get("source", "")).strip():
            raise ValueError(
                f"Fluid package '{package_id}' needs an interaction-data source."
            )
        if not isinstance(interaction_parameters.get("overrides", {}), dict):
            raise ValueError(
                f"Fluid package '{package_id}' interaction overrides must be an "
                "object."
            )

        registry = package.get("component_registry")
        if not isinstance(registry, list) or not registry:
            raise ValueError(
                f"Fluid package '{package_id}' needs a component registry."
            )
        registry_names: set[str] = set()
        registry_keys: set[str] = set()
        for component_index, component in enumerate(registry):
            if not isinstance(component, dict):
                raise ValueError(
                    f"Fluid package '{package_id}' component "
                    f"{component_index} must be an object."
                )
            component_name = str(component.get("name", "")).strip()
            if not component_name:
                raise ValueError(
                    f"Fluid package '{package_id}' has an unnamed component."
                )
            component_key = component_name.casefold()
            if component_key in registry_keys:
                raise ValueError(
                    f"Fluid package '{package_id}' component names must be unique."
                )
            registry_names.add(component_name)
            registry_keys.add(component_key)

            component_kind = str(component.get("kind", "")).lower().strip()
            if component_kind not in ("standard", "pseudo"):
                raise ValueError(
                    f"Component '{component_name}' kind must be standard or pseudo."
                )
            definition: dict[str, Any] = {"kind": component_kind}
            if component_kind == "pseudo":
                molar_mass = _finite_float(
                    component.get("molar_mass_kg_per_mol"),
                    f"{component_name} molar_mass_kg_per_mol",
                )
                density = _finite_float(
                    component.get("normal_liquid_density_kg_per_m3"),
                    f"{component_name} normal_liquid_density_kg_per_m3",
                )
                if molar_mass <= 0.0 or density <= 0.0:
                    raise ValueError(
                        f"Pseudo-component '{component_name}' properties must be "
                        "positive."
                    )
                definition.update(
                    {
                        "molar_mass_kg_per_mol": molar_mass,
                        "normal_liquid_density_kg_per_m3": density,
                    }
                )

            previous_definition = component_definitions.get(component_key)
            if previous_definition is not None:
                if previous_definition["kind"] != definition["kind"]:
                    raise ValueError(
                        f"Component '{component_name}' cannot be both standard and "
                        "pseudo."
                    )
                if component_kind == "pseudo":
                    for property_name in (
                        "molar_mass_kg_per_mol",
                        "normal_liquid_density_kg_per_m3",
                    ):
                        if not math.isclose(
                            previous_definition[property_name],
                            definition[property_name],
                            rel_tol=1.0e-9,
                            abs_tol=1.0e-12,
                        ):
                            raise ValueError(
                                f"Pseudo-component '{component_name}' has conflicting "
                                f"{property_name} characterization."
                            )
            else:
                component_definitions[component_key] = definition
        registry_by_package[package_id] = registry_names

    inlet_ids: set[str] = set()
    inlet_names: set[str] = set()
    for inlet_index, inlet in enumerate(inlets):
        if not isinstance(inlet, dict):
            raise ValueError(f"inlets[{inlet_index}] must be an object.")
        inlet_id = _required_identifier(
            inlet.get("id"),
            f"inlets[{inlet_index}].id",
        )
        if inlet_id in inlet_ids:
            raise ValueError("Inlet ids must be unique.")
        inlet_ids.add(inlet_id)

        raw_inlet_name = inlet.get("name")
        inlet_name = (
            "" if raw_inlet_name is None else str(raw_inlet_name).strip()
        )
        if not inlet_name:
            raise ValueError(f"Inlet '{inlet_id}' requires a stream name.")
        if inlet_name in inlet_names:
            raise ValueError(f"Inlet stream name '{inlet_name}' is duplicated.")
        inlet_names.add(inlet_name)

        package_id = str(inlet.get("fluid_package_id", "")).strip()
        if package_id not in packages_by_id:
            raise ValueError(
                f"Inlet '{inlet_id}' references unknown fluid package "
                f"'{package_id}'."
            )
        if inlet.get("composition_basis") != "mole_fraction":
            raise ValueError(f"Inlet '{inlet_id}' requires mole fractions.")
        if inlet.get("flow_unit") != "kg/hr":
            raise ValueError(f"Inlet '{inlet_id}' requires kg/hr mass flow.")

        composition = inlet.get("composition")
        if not isinstance(composition, dict):
            raise ValueError(f"Inlet '{inlet_id}' composition must be an object.")
        if set(composition) != registry_by_package[package_id]:
            raise ValueError(
                f"Inlet '{inlet_id}' composition must match the shared component "
                "registry exactly."
            )
        composition_total = 0.0
        for component_name, fraction_value in composition.items():
            fraction = _finite_float(
                fraction_value,
                f"Inlet '{inlet_id}' {component_name} mole fraction",
            )
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(
                    f"Inlet '{inlet_id}' mole fractions must be between 0 and 1."
                )
            composition_total += fraction
        if not math.isclose(
            composition_total,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError(
                f"Inlet '{inlet_id}' mole fractions must sum to 1.0."
            )

        temperature = _finite_float(
            inlet.get("temperature_C"),
            f"Inlet '{inlet_id}' temperature_C",
        )
        pressure = _finite_float(
            inlet.get("pressure_bara"),
            f"Inlet '{inlet_id}' pressure_bara",
        )
        flow = _finite_float(
            inlet.get("total_flow"),
            f"Inlet '{inlet_id}' total_flow",
        )
        if temperature <= -273.15:
            raise ValueError(
                f"Inlet '{inlet_id}' temperature must be above absolute zero."
            )
        if pressure <= 0.0:
            raise ValueError(f"Inlet '{inlet_id}' pressure must be positive.")
        if flow <= 0.0:
            raise ValueError(f"Inlet '{inlet_id}' flow must be positive.")


def _validate_case_architecture(
    case_data: dict[str, Any],
    expected_fluid: dict[str, Any],
) -> None:
    """Validate shared fluids and the primary-inlet builder projection."""
    if case_data["schema_version"] == LEGACY_CASE_SCHEMA_VERSION:
        return

    fluid_packages = case_data.get("fluid_packages")
    inlets = case_data.get("inlets")
    if not isinstance(fluid_packages, list):
        raise ValueError("The case requires a fluid_packages array.")
    if not isinstance(inlets, list):
        raise ValueError("The case requires an inlets array.")
    _validate_fluid_package_integrity(fluid_packages, inlets)

    if len(fluid_packages) != 1:
        raise ValueError("The current starter template requires one fluid package.")
    package = fluid_packages[0]
    package_id = str(package["id"]).strip()
    if package_id != BASE_FLUID_PACKAGE_ID:
        raise ValueError(
            f"The starter template fluid package id must be "
            f"'{BASE_FLUID_PACKAGE_ID}'."
        )
    if str(package["eos_model"]).lower() != expected_fluid["eos_model"]:
        raise ValueError("Fluid-package and builder EOS definitions are inconsistent.")
    package_mixing_rule = int(float(package["mixing_rule"]))
    if package_mixing_rule != expected_fluid["mixing_rule"]:
        raise ValueError(
            "Fluid-package and builder mixing-rule definitions are inconsistent."
        )

    registry = package["component_registry"]
    registry_names = [str(component["name"]).strip() for component in registry]
    if any(component["kind"] != "standard" for component in registry):
        raise ValueError(
            "The current builder projection supports standard components only."
        )
    if set(registry_names) != set(expected_fluid["components"]):
        raise ValueError(
            "The fluid-package registry and builder components are inconsistent."
        )

    interaction_parameters = package["binary_interaction_parameters"]
    if interaction_parameters["source"] != "NeqSim database":
        raise ValueError(
            "The current builder projection requires NeqSim database interactions."
        )
    if interaction_parameters.get("overrides"):
        raise ValueError(
            "Binary-interaction overrides are not yet supported by ProcessBuilder."
        )

    primary_inlets = [
        inlet
        for inlet in inlets
        if str(inlet["id"]).strip() == PRIMARY_INLET_ID
    ]
    if len(primary_inlets) != 1:
        raise ValueError(
            f"The case requires exactly one primary inlet '{PRIMARY_INLET_ID}'."
        )
    inlet = primary_inlets[0]
    if inlet["fluid_package_id"] != package_id:
        raise ValueError("The inlet references an unknown fluid package.")

    inlet_composition = inlet["composition"]
    for component, expected_fraction in expected_fluid["components"].items():
        inlet_fraction = float(inlet_composition[component])
        if not math.isclose(
            inlet_fraction,
            expected_fraction,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                f"Inlet and builder compositions differ for {component}."
            )

    comparisons = (
        ("temperature_C", "temperature_C", "inlet temperature"),
        ("pressure_bara", "pressure_bara", "inlet pressure"),
        ("total_flow", "total_flow", "inlet flow"),
    )
    for inlet_key, expected_key, label in comparisons:
        inlet_value = float(inlet[inlet_key])
        if not math.isclose(
            inlet_value,
            expected_fluid[expected_key],
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"The {label} conflicts with the builder projection.")

def _index_graph_objects(
    objects: list[Any],
    label: str,
) -> dict[str, dict[str, Any]]:
    """Index graph objects by stable id and reject malformed or duplicate ids."""
    indexed: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(objects):
        if not isinstance(item, dict):
            raise ValueError(f"{label}[{index}] must be an object.")
        item_id = _required_identifier(
            item.get("id"),
            f"{label}[{index}].id",
        )
        if item_id in indexed:
            raise ValueError(f"{label} ids must be unique.")
        indexed[item_id] = item
    return indexed


def _validate_graph_integrity(
    inlets: list[Any],
    units: list[Any],
    connections: list[Any],
) -> None:
    """Validate reusable node, port, and connection invariants."""
    indexed_inlets = _index_graph_objects(inlets, "inlets")
    indexed_units = _index_graph_objects(units, "units")
    overlapping_ids = set(indexed_inlets).intersection(indexed_units)
    if overlapping_ids:
        duplicate_id = sorted(overlapping_ids)[0]
        raise ValueError(
            f"Graph id '{duplicate_id}' is used by both an inlet and a unit."
        )

    for unit_id, unit in indexed_units.items():
        ports = unit.get("ports")
        if not isinstance(ports, dict):
            raise ValueError(f"Unit '{unit_id}' requires a ports object.")
        for connection_type in ("material", "energy"):
            input_key = f"{connection_type}_in"
            output_key = f"{connection_type}_out"
            input_ports = ports.get(input_key, [])
            output_ports = ports.get(output_key, [])
            for key, port_names in (
                (input_key, input_ports),
                (output_key, output_ports),
            ):
                if not isinstance(port_names, list):
                    raise ValueError(f"Unit '{unit_id}' {key} must be an array.")
                cleaned_ports = [str(port).strip() for port in port_names]
                if any(not port for port in cleaned_ports):
                    raise ValueError(f"Unit '{unit_id}' {key} has an empty port.")
                if len(cleaned_ports) != len(set(cleaned_ports)):
                    raise ValueError(f"Unit '{unit_id}' {key} ports must be unique.")
            ambiguous_ports = set(input_ports).intersection(output_ports)
            if ambiguous_ports:
                port = sorted(ambiguous_ports)[0]
                raise ValueError(
                    f"Unit '{unit_id}' port '{port}' cannot be both input and output."
                )

    indexed_connections = _index_graph_objects(connections, "connections")
    used_sources: set[tuple[str, str, str, str]] = set()
    used_targets: set[tuple[str, str, str, str]] = set()
    used_routes: set[
        tuple[str, str, str, str, str, str, str]
    ] = set()

    for connection_id, connection in indexed_connections.items():
        connection_type = str(connection.get("type", "")).strip()
        if connection_type not in ("material", "energy"):
            raise ValueError(
                f"Connection '{connection_id}' type must be material or energy."
            )
        endpoints: dict[str, tuple[str, str, str]] = {}
        for endpoint_name in ("source", "target"):
            endpoint = connection.get(endpoint_name)
            if not isinstance(endpoint, dict):
                raise ValueError(
                    f"Connection '{connection_id}' {endpoint_name} must be an object."
                )
            endpoint_kind = str(endpoint.get("kind", "")).strip()
            endpoint_id = str(endpoint.get("id", "")).strip()
            endpoint_port = str(endpoint.get("port", "")).strip()
            if not endpoint_id or not endpoint_port:
                raise ValueError(
                    f"Connection '{connection_id}' {endpoint_name} needs id and port."
                )
            if endpoint_kind == "inlet":
                if endpoint_id not in indexed_inlets:
                    raise ValueError(
                        f"Connection '{connection_id}' references unknown inlet "
                        f"'{endpoint_id}'."
                    )
                if endpoint_name != "source":
                    raise ValueError(
                        f"Inlet '{endpoint_id}' can only be a connection source."
                    )
                if connection_type != "material" or endpoint_port != "out":
                    raise ValueError(
                        f"Inlet '{endpoint_id}' exposes only material output port 'out'."
                    )
            elif endpoint_kind == "unit":
                if endpoint_id not in indexed_units:
                    raise ValueError(
                        f"Connection '{connection_id}' references unknown unit "
                        f"'{endpoint_id}'."
                    )
                direction = "out" if endpoint_name == "source" else "in"
                port_key = f"{connection_type}_{direction}"
                declared_ports = indexed_units[endpoint_id]["ports"].get(port_key, [])
                if endpoint_port not in declared_ports:
                    raise ValueError(
                        f"Connection '{connection_id}' uses undeclared {port_key} "
                        f"port '{endpoint_port}' on unit '{endpoint_id}'."
                    )
            else:
                raise ValueError(
                    f"Connection '{connection_id}' has unsupported endpoint kind "
                    f"'{endpoint_kind}'."
                )
            endpoints[endpoint_name] = (
                endpoint_kind,
                endpoint_id,
                endpoint_port,
            )

        source = endpoints["source"]
        target = endpoints["target"]
        if source[:2] == target[:2]:
            raise ValueError(
                f"Connection '{connection_id}' cannot connect a node to itself."
            )
        source_key = (connection_type, *source)
        target_key = (connection_type, *target)
        route_key = (connection_type, *source, *target)
        if source_key in used_sources:
            raise ValueError(
                f"Graph output port {source[1]}:{source[2]} has multiple connections."
            )
        if target_key in used_targets:
            raise ValueError(
                f"Graph input port {target[1]}:{target[2]} has multiple connections."
            )
        if route_key in used_routes:
            raise ValueError(f"Connection route '{connection_id}' is duplicated.")
        used_sources.add(source_key)
        used_targets.add(target_key)
        used_routes.add(route_key)


def _build_inlet_fluid_specs(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Compile shared characterization and inlet conditions for ProcessBuilder."""
    fluid_packages = spec.get("fluid_packages")
    inlets = spec.get("inlets")
    if not isinstance(fluid_packages, list):
        raise ValueError("Inlet fluid construction requires fluid_packages.")
    if not isinstance(inlets, list):
        raise ValueError("Inlet fluid construction requires inlets.")

    _validate_fluid_package_integrity(fluid_packages, inlets)
    packages_by_id = {
        str(package["id"]).strip(): package for package in fluid_packages
    }
    inlet_specs: list[dict[str, Any]] = []
    for inlet in inlets:
        package_id = str(inlet["fluid_package_id"]).strip()
        package = packages_by_id[package_id]
        fluid_spec = {
            "eos_model": str(package["eos_model"]).lower().strip(),
            "mixing_rule": int(float(package["mixing_rule"])),
            "components": dict(inlet["composition"]),
            "composition_basis": inlet["composition_basis"],
            "temperature_C": float(inlet["temperature_C"]),
            "pressure_bara": float(inlet["pressure_bara"]),
            "total_flow": float(inlet["total_flow"]),
            "flow_unit": inlet["flow_unit"],
        }
        characterization = json.loads(
            json.dumps(
                {
                    "component_registry": package["component_registry"],
                    "binary_interaction_parameters": package[
                        "binary_interaction_parameters"
                    ],
                },
                allow_nan=False,
            )
        )
        inlet_specs.append(
            {
                "inlet_id": str(inlet["id"]).strip(),
                "name": str(inlet.get("name", inlet["id"])).strip(),
                "fluid_package_id": package_id,
                "fluid_spec": fluid_spec,
                "characterization": characterization,
            }
        )
    return inlet_specs


def _build_execution_plan(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Compile a validated acyclic process graph into deterministic steps."""
    fluid_packages = spec.get("fluid_packages")
    inlets = spec.get("inlets")
    units = spec.get("units")
    connections = spec.get("connections")
    if not isinstance(fluid_packages, list):
        raise ValueError("Execution planning requires a fluid_packages array.")
    if not isinstance(inlets, list):
        raise ValueError("Execution planning requires an inlets array.")
    if not isinstance(units, list):
        raise ValueError("Execution planning requires a units array.")
    if not isinstance(connections, list):
        raise ValueError("Execution planning requires a connections array.")

    _validate_fluid_package_integrity(fluid_packages, inlets)
    _validate_graph_integrity(inlets, units, connections)

    indexed_inlets = _index_graph_objects(inlets, "inlets")
    indexed_units = _index_graph_objects(units, "units")
    node_order = [*indexed_inlets, *indexed_units]
    order_index = {
        node_id: node_index for node_index, node_id in enumerate(node_order)
    }
    incoming: dict[str, list[str]] = {node_id: [] for node_id in node_order}
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in node_order}
    dependencies: dict[str, set[str]] = {
        node_id: set() for node_id in node_order
    }
    dependents: dict[str, set[str]] = {
        node_id: set() for node_id in node_order
    }

    for connection in connections:
        connection_id = str(connection["id"]).strip()
        source_id = str(connection["source"]["id"]).strip()
        target_id = str(connection["target"]["id"]).strip()
        outgoing[source_id].append(connection_id)
        incoming[target_id].append(connection_id)
        dependencies[target_id].add(source_id)
        dependents[source_id].add(target_id)

    pending_dependencies = {
        node_id: len(node_dependencies)
        for node_id, node_dependencies in dependencies.items()
    }
    ready = [
        node_id
        for node_id in node_order
        if pending_dependencies[node_id] == 0
    ]
    ordered_nodes: list[str] = []
    while ready:
        node_id = ready.pop(0)
        ordered_nodes.append(node_id)
        for dependent_id in sorted(
            dependents[node_id],
            key=order_index.__getitem__,
        ):
            pending_dependencies[dependent_id] -= 1
            if pending_dependencies[dependent_id] == 0:
                ready.append(dependent_id)
                ready.sort(key=order_index.__getitem__)

    if len(ordered_nodes) != len(node_order):
        cyclic_nodes = [
            node_id
            for node_id in node_order
            if pending_dependencies[node_id] > 0
        ]
        raise ValueError(
            "The acyclic executor cannot schedule graph cycles involving: "
            f"{', '.join(cyclic_nodes)}. Recycles require tear-stream solving."
        )

    plan: list[dict[str, Any]] = []
    for step_number, node_id in enumerate(ordered_nodes, start=1):
        is_inlet = node_id in indexed_inlets
        node = indexed_inlets[node_id] if is_inlet else indexed_units[node_id]
        node_dependencies = sorted(
            dependencies[node_id],
            key=order_index.__getitem__,
        )
        plan.append(
            {
                "Step": step_number,
                "Kind": "inlet" if is_inlet else "unit",
                "Object ID": node_id,
                "Name": str(node.get("name", node_id)).strip() or node_id,
                "Type": "stream" if is_inlet else str(node.get("type", "")),
                "Dependencies": ", ".join(node_dependencies),
                "Incoming connections": ", ".join(incoming[node_id]),
                "Outgoing connections": ", ".join(outgoing[node_id]),
            }
        )
    return plan


def _build_graph_solver_inputs(
    spec: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Compile one validated Studio case for generic graph execution."""
    execution_plan = _build_execution_plan(spec)
    inlet_specs = _build_inlet_fluid_specs(spec)
    graph_spec = {
        "name": str(spec.get("name", "Graph Process")).strip()
        or "Graph Process",
        "units": json.loads(
            json.dumps(spec["units"], allow_nan=False)
        ),
        "connections": json.loads(
            json.dumps(spec["connections"], allow_nan=False)
        ),
    }
    execution_order = [
        str(step["Object ID"]).strip()
        for step in execution_plan
    ]
    return graph_spec, inlet_specs, execution_order


def _validate_case_graph(
    case_data: dict[str, Any],
    process: list[dict[str, Any]],
) -> None:
    """Validate schema-v3 graph integrity and starter-template compatibility."""
    if case_data["schema_version"] < CASE_SCHEMA_VERSION:
        return

    inlets = case_data.get("inlets")
    units = case_data.get("units")
    connections = case_data.get("connections")
    if not isinstance(inlets, list) or not inlets:
        raise ValueError("Schema v3 requires a non-empty inlets array.")
    if not isinstance(units, list):
        raise ValueError("Schema v3 requires a units array.")
    if not isinstance(connections, list):
        raise ValueError("Schema v3 requires a connections array.")

    _validate_graph_integrity(inlets, units, connections)
    conflicting_graph_names = _terminal_name_conflicts(
        [*inlets, *units],
        _terminal_material_stream_names(units, connections),
    )
    if conflicting_graph_names:
        raise ValueError(
            "Inlet or unit names conflict with generated terminal product "
            "streams: "
            + ", ".join(conflicting_graph_names)
            + "."
        )
    indexed_units = _index_graph_objects(units, "units")
    expected_units, _ = _build_template_graph(process)
    expected_unit_map = {unit["id"]: unit for unit in expected_units}
    validate_starter_unit_projection(units, expected_units)
    for unit_id, unit in indexed_units.items():
        if unit_id in expected_unit_map:
            continue
        if str(unit.get("type", "")).strip().lower() == "mixer":
            if not str(unit.get("name", "")).strip():
                raise ValueError(f"Graph mixer '{unit_id}' requires a name.")
            ports = unit.get("ports")
            if not isinstance(ports, dict):
                raise ValueError(f"Graph mixer '{unit_id}' requires ports.")
            material_inputs = ports.get("material_in")
            material_outputs = ports.get("material_out")
            if (
                not isinstance(material_inputs, list)
                or len(material_inputs) < 2
                or len(set(material_inputs)) != len(material_inputs)
            ):
                raise ValueError(
                    f"Graph mixer '{unit_id}' requires at least two unique "
                    "material input ports."
                )
            if material_outputs != ["out"]:
                raise ValueError(
                    f"Graph mixer '{unit_id}' requires material output port 'out'."
                )
            if unit.get("params", {}) != {}:
                raise ValueError(
                    f"Graph mixer '{unit_id}' does not accept operating params."
                )
        else:
            validate_catalog_unit(unit)
    _build_execution_plan(case_data)

def _load_case_controls(case_data: Any) -> tuple[dict[str, Any], pd.DataFrame, list[str]]:
    """Validate an exported Studio case and map it back to UI controls."""
    if not isinstance(case_data, dict):
        raise ValueError("The case JSON root must be an object.")
    schema_version = case_data.get("schema_version")
    supported_schema_versions = (
        LEGACY_CASE_SCHEMA_VERSION,
        SHARED_FLUID_CASE_SCHEMA_VERSION,
        CASE_SCHEMA_VERSION,
    )
    if type(schema_version) is not int or schema_version not in (
        supported_schema_versions
    ):
        raise ValueError(
            "Unsupported schema_version. Expected version 1, 2, or 3."
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

    _validate_case_architecture(
        case_data,
        {
            "eos_model": eos_model,
            "mixing_rule": int(mixing_rule),
            "components": normalized_composition,
            "temperature_C": feed_temperature_c,
            "pressure_bara": feed_pressure_bara,
            "total_flow": feed_flow_kg_hr,
        },
    )

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
    intercooler_pressure_drop_bar = _bounded_float(
        intercooler_params.get("pressure_drop_bar", 0.0),
        "intercooler pressure drop",
        0.0,
        50.0,
    )
    export_pressure_drop_bar = _bounded_float(
        export_cooler_params.get("pressure_drop_bar", 0.0),
        "export cooler pressure drop",
        0.0,
        50.0,
    )

    _validate_case_graph(case_data, process)

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
        intercooler_pressure_drop_bar=intercooler_pressure_drop_bar,
        export_temperature_c=export_temperature_c,
        export_pressure_drop_bar=export_pressure_drop_bar,
        stage_1_isentropic_efficiency=efficiency_1,
        stage_2_isentropic_efficiency=efficiency_2,
    )
    graph_draft = None
    if schema_version >= CASE_SCHEMA_VERSION:
        imported_draft = create_graph_draft(
            case_data["units"],
            case_data["connections"],
            case_data["inlets"],
        )
        if (
            imported_draft["inlets"] != canonical_spec["inlets"]
            or imported_draft["units"] != canonical_spec["units"]
            or imported_draft["connections"] != canonical_spec["connections"]
        ):
            graph_draft = imported_draft
            canonical_spec = _apply_studio_graph_draft(
                canonical_spec,
                imported_draft,
            )
    warnings = _validate_case(canonical_spec, composition_total)
    if schema_version < CASE_SCHEMA_VERSION:
        warnings.insert(
            0,
            f"Schema-v{schema_version} case migrated to graph schema v3.",
        )
    controls = {
        "flowsheet_case_name": case_name,
        "flowsheet_eos_model": eos_model,
        "flowsheet_feed_temperature_c": feed_temperature_c,
        "flowsheet_feed_pressure_bara": feed_pressure_bara,
        "flowsheet_feed_flow_kg_hr": feed_flow_kg_hr,
        "flowsheet_stage_1_pressure_bara": stage_1_pressure_bara,
        "flowsheet_stage_2_pressure_bara": stage_2_pressure_bara,
        "flowsheet_stage_1_isentropic_efficiency": efficiency_1,
        "flowsheet_stage_2_isentropic_efficiency": efficiency_2,
        "flowsheet_intercooler_temperature_c": intercooler_temperature_c,
        "flowsheet_intercooler_pressure_drop_bar": intercooler_pressure_drop_bar,
        "flowsheet_export_temperature_c": export_temperature_c,
        "flowsheet_export_pressure_drop_bar": export_pressure_drop_bar,
        GRAPH_DRAFT_STATE_KEY: graph_draft,
    }
    return controls, composition_table, warnings


def _apply_imported_case(
    controls: dict[str, Any],
    composition_table: pd.DataFrame,
    warnings: list[str],
) -> None:
    """Replace the current controls and invalidate any previously solved model."""
    for key, value in controls.items():
        if key == GRAPH_DRAFT_STATE_KEY:
            continue
        st.session_state[key] = value
    graph_draft = controls.get(GRAPH_DRAFT_STATE_KEY)
    if graph_draft is None:
        st.session_state.pop(GRAPH_DRAFT_STATE_KEY, None)
    else:
        st.session_state[GRAPH_DRAFT_STATE_KEY] = graph_draft
    st.session_state.pop(GRAPH_HISTORY_STATE_KEY, None)
    st.session_state["flowsheet_composition_source"] = composition_table
    st.session_state["flowsheet_composition_revision"] += 1
    _clear_studio_runtime(clear_history=False)
    notice = "Case loaded. Review the inputs and run the NeqSim flowsheet."
    if warnings:
        notice += " " + " ".join(warnings)
    st.session_state[CASE_NOTICE_STATE_KEY] = notice


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


def _build_template_graph(
    process: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Project the starter process into stable nodes and explicit material edges."""
    units: list[dict[str, Any]] = []
    for step in process[1:]:
        unit = {
            "id": TEMPLATE_UNIT_IDS[step["name"]],
            "name": step["name"],
            "type": step["type"],
            "ports": {
                "material_in": ["in"],
                "material_out": (
                    ["gas", "liquid"] if step["type"] == "separator" else ["out"]
                ),
            },
        }
        if "params" in step:
            unit["params"] = dict(step["params"])
        if "outlet" in step:
            unit["builder_outlet"] = step["outlet"]
        units.append(unit)

    path = (
        (
            "feed-gas-to-inlet-scrubber",
            ("inlet", PRIMARY_INLET_ID, "out"),
            ("unit", "inlet-scrubber", "in"),
        ),
        (
            "inlet-scrubber-gas-to-compressor-stage-1",
            ("unit", "inlet-scrubber", "gas"),
            ("unit", "compressor-stage-1", "in"),
        ),
        (
            "compressor-stage-1-to-intercooler",
            ("unit", "compressor-stage-1", "out"),
            ("unit", "intercooler", "in"),
        ),
        (
            "intercooler-to-interstage-scrubber",
            ("unit", "intercooler", "out"),
            ("unit", "interstage-scrubber", "in"),
        ),
        (
            "interstage-scrubber-gas-to-compressor-stage-2",
            ("unit", "interstage-scrubber", "gas"),
            ("unit", "compressor-stage-2", "in"),
        ),
        (
            "compressor-stage-2-to-export-cooler",
            ("unit", "compressor-stage-2", "out"),
            ("unit", "export-cooler", "in"),
        ),
    )
    connections = [
        {
            "id": connection_id,
            "type": "material",
            "source": {"kind": source[0], "id": source[1], "port": source[2]},
            "target": {"kind": target[0], "id": target[1], "port": target[2]},
        }
        for connection_id, source, target in path
    ]
    return units, connections


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
    intercooler_pressure_drop_bar: float,
    export_temperature_c: float,
    export_pressure_drop_bar: float,
    stage_1_isentropic_efficiency: float,
    stage_2_isentropic_efficiency: float,
) -> dict[str, Any]:
    """Create schema-v3 case data plus the current ProcessBuilder projection."""
    fluid_spec = {
        "eos_model": eos_model,
        "mixing_rule": 2,
        "components": dict(composition),
        "composition_basis": "mole_fraction",
        "temperature_C": feed_temperature_c,
        "pressure_bara": feed_pressure_bara,
        "total_flow": feed_flow_kg_hr,
        "flow_unit": "kg/hr",
    }
    fluid_package = {
        "id": BASE_FLUID_PACKAGE_ID,
        "name": "Base fluid characterization",
        "eos_model": eos_model,
        "mixing_rule": 2,
        "component_registry": [
            {
                "name": component,
                "kind": "standard",
                "source": "NeqSim component database",
            }
            for component in composition
        ],
        "binary_interaction_parameters": {
            "source": "NeqSim database",
            "overrides": {},
        },
    }
    inlet = {
        "id": PRIMARY_INLET_ID,
        "name": "feed gas",
        "fluid_package_id": BASE_FLUID_PACKAGE_ID,
        "composition": dict(composition),
        "composition_basis": "mole_fraction",
        "temperature_C": feed_temperature_c,
        "pressure_bara": feed_pressure_bara,
        "total_flow": feed_flow_kg_hr,
        "flow_unit": "kg/hr",
    }
    process = [
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
                "isentropic_efficiency": stage_1_isentropic_efficiency,
            },
        },
        {
            "name": "intercooler",
            "type": "cooler",
            "params": {
                "outlet_temperature_C": intercooler_temperature_c,
                "pressure_drop_bar": intercooler_pressure_drop_bar,
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
                "isentropic_efficiency": stage_2_isentropic_efficiency,
            },
        },
        {
            "name": "export cooler",
            "type": "cooler",
            "params": {
                "outlet_temperature_C": export_temperature_c,
                "pressure_drop_bar": export_pressure_drop_bar,
            },
        },
    ]
    units, connections = _build_template_graph(process)
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
            "Cooler pressure drops are fixed values in bar.",
            "All inlet streams reference a shared base fluid characterization.",
            "Material paths use explicit source and target ports.",
        ],
        "fluid_packages": [fluid_package],
        "inlets": [inlet],
        "units": units,
        "connections": connections,
        "fluid": fluid_spec,
        "process": process,
    }

def _validate_case(spec: dict[str, Any], composition_total: float) -> list[str]:
    """Return non-blocking engineering warnings after hard validation."""
    _build_execution_plan(spec)
    inlet_fluid_specs = _build_inlet_fluid_specs(spec)
    warnings: list[str] = []
    fluid = spec["fluid"]
    primary_inlet_specs = [
        inlet_spec
        for inlet_spec in inlet_fluid_specs
        if inlet_spec["inlet_id"] == PRIMARY_INLET_ID
    ]
    if len(primary_inlet_specs) != 1:
        raise ValueError(
            f"The case requires exactly one primary inlet '{PRIMARY_INLET_ID}'."
        )
    if primary_inlet_specs[0]["fluid_spec"] != fluid:
        raise ValueError(
            "The primary inlet conflicts with the ProcessBuilder fluid projection."
        )
    process = spec["process"]

    stage_1_pressure = process[2]["params"]["outlet_pressure_bara"]
    stage_2_pressure = process[5]["params"]["outlet_pressure_bara"]
    feed_pressure = fluid["pressure_bara"]
    stage_1_efficiency = process[2]["params"]["isentropic_efficiency"]
    stage_2_efficiency = process[5]["params"]["isentropic_efficiency"]
    intercooler_pressure_drop = process[3]["params"]["pressure_drop_bar"]
    export_pressure_drop = process[6]["params"]["pressure_drop_bar"]

    if feed_pressure <= 0.0:
        raise ValueError("Feed pressure must be greater than zero bara.")
    if fluid["total_flow"] <= 0.0:
        raise ValueError("Feed flow must be greater than zero kg/hr.")
    if not feed_pressure < stage_1_pressure < stage_2_pressure:
        raise ValueError(
            "Pressure ordering must be feed pressure < stage 1 pressure "
            "< stage 2 pressure."
        )
    for stage_number, efficiency in (
        (1, stage_1_efficiency),
        (2, stage_2_efficiency),
    ):
        if not 0.50 <= efficiency <= 0.95:
            raise ValueError(
                f"Compressor stage {stage_number} isentropic efficiency must be "
                "between 0.50 and 0.95."
            )
    for cooler_name, pressure_drop, inlet_pressure in (
        ("Intercooler", intercooler_pressure_drop, stage_1_pressure),
        ("Export cooler", export_pressure_drop, stage_2_pressure),
    ):
        if not 0.0 <= pressure_drop <= 50.0:
            raise ValueError(
                f"{cooler_name} pressure drop must be between 0 and 50 bar."
            )
        if pressure_drop >= inlet_pressure:
            raise ValueError(
                f"{cooler_name} pressure drop must be lower than its inlet pressure."
            )
        if pressure_drop / inlet_pressure > 0.10:
            warnings.append(
                f"{cooler_name} pressure drop exceeds 10% of its inlet pressure."
            )
    if abs(composition_total - 1.0) > 1.0e-6:
        warnings.append(
            f"Composition summed to {composition_total:.6f} and was normalized to 1.0."
        )
    if stage_1_pressure / feed_pressure > 3.0:
        warnings.append("Stage 1 pressure ratio exceeds 3.0; check compressor feasibility.")
    stage_2_inlet_pressure = stage_1_pressure - intercooler_pressure_drop
    if stage_2_pressure / stage_2_inlet_pressure > 3.0:
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


def _selected_object_result_tables(
    selected_object: str,
    stream_table: pd.DataFrame,
    equipment_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter current solved results for one object in the template palette."""
    if selected_object not in TEMPLATE_OBJECTS:
        raise ValueError(f"Unknown flowsheet object: {selected_object}")

    selected_key = selected_object.casefold()

    def equipment_matches(value: Any) -> bool:
        name = str(value).casefold()
        return name == selected_key or name.endswith(f"/{selected_key}")

    def stream_matches(value: Any) -> bool:
        name = str(value).casefold()
        if name == selected_key:
            return True
        owner = name.split(".", 1)[0]
        return owner == selected_key or owner.endswith(f"/{selected_key}")

    if "Equipment" in equipment_table.columns:
        selected_equipment = equipment_table[
            equipment_table["Equipment"].map(equipment_matches)
        ].copy()
    else:
        selected_equipment = equipment_table.iloc[0:0].copy()

    if "Stream" in stream_table.columns:
        selected_streams = stream_table[
            stream_table["Stream"].map(stream_matches)
        ].copy()
    else:
        selected_streams = stream_table.iloc[0:0].copy()

    return (
        selected_equipment.reset_index(drop=True),
        selected_streams.reset_index(drop=True),
    )


def _pressure_profile_dataframe(
    spec: dict[str, Any],
    equipment_table: pd.DataFrame,
) -> pd.DataFrame:
    """Compare solved outlet pressures with the current case specifications."""
    process_steps = {step["name"]: step for step in spec["process"]}
    stage_1_pressure = float(
        process_steps["compressor stage 1"]["params"]["outlet_pressure_bara"]
    )
    stage_2_pressure = float(
        process_steps["compressor stage 2"]["params"]["outlet_pressure_bara"]
    )
    intercooler_drop = float(
        process_steps["intercooler"]["params"]["pressure_drop_bar"]
    )
    export_drop = float(
        process_steps["export cooler"]["params"]["pressure_drop_bar"]
    )
    expected_pressures = (
        ("Compressor stage 1", "compressor stage 1", stage_1_pressure),
        ("Intercooler", "intercooler", stage_1_pressure - intercooler_drop),
        ("Compressor stage 2", "compressor stage 2", stage_2_pressure),
        ("Export cooler", "export cooler", stage_2_pressure - export_drop),
    )

    records: list[dict[str, Any]] = []
    for display_name, object_name, expected in expected_pressures:
        actual = None
        if {
            "Equipment",
            "outletPressure_bara",
        }.issubset(equipment_table.columns):
            object_key = object_name.casefold()
            names = equipment_table["Equipment"].astype(str).str.casefold()
            matches = equipment_table[
                (names == object_key) | names.str.endswith(f"/{object_key}")
            ]
            if not matches.empty:
                candidate = pd.to_numeric(
                    matches.iloc[0]["outletPressure_bara"],
                    errors="coerce",
                )
                if pd.notna(candidate) and math.isfinite(float(candidate)):
                    actual = float(candidate)

        deviation = None if actual is None else actual - expected
        if deviation is None:
            status = "WARN"
            detail = "Solved outlet pressure was not reported by the model adapter."
        elif abs(deviation) <= 0.05:
            status = "OK"
            detail = "Calculated pressure matches the current specification."
        elif abs(deviation) <= 0.50:
            status = "WARN"
            detail = "Calculated pressure is outside the 0.05 bar pass tolerance."
        else:
            status = "VIOLATION"
            detail = "Calculated pressure differs by more than 0.50 bar."

        records.append(
            {
                "Operation": display_name,
                "Expected outlet [bara]": expected,
                "Calculated outlet [bara]": actual,
                "Deviation [bar]": deviation,
                "Pass tolerance [bar]": 0.05,
                "Status": status,
                "Detail": detail,
            }
        )
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


def _material_boundary_dataframe(result: Any) -> pd.DataFrame:
    """Build an explicit-unit feed/product table from solved boundaries."""
    columns = [
        "Role",
        "Stream",
        "Mass flow [kg/hr]",
        "Temperature [°C]",
        "Pressure [bara]",
        "Molar flow [mol/s]",
        "Enthalpy flow [kW]",
    ]
    records = [
        {
            "Role": row["role"].title(),
            "Stream": row["stream_name"],
            "Mass flow [kg/hr]": row["mass_flow_kg_hr"],
            "Temperature [°C]": row["temperature_C"],
            "Pressure [bara]": row["pressure_bara"],
            "Molar flow [mol/s]": row["molar_flow_mol_sec"],
            "Enthalpy flow [kW]": row["enthalpy_flow_kW"],
        }
        for row in material_boundary_rows(result)
    ]
    return pd.DataFrame(records, columns=columns)


def _component_balance_dataframe(result: Any) -> pd.DataFrame:
    """Build an explicit-unit component closure table."""
    columns = [
        "Component",
        "Feed molar flow [mol/s]",
        "Product molar flow [mol/s]",
        "Residual [mol/s]",
        "Imbalance [%]",
    ]
    records = [
        {
            "Component": row["component"],
            "Feed molar flow [mol/s]": row[
                "feed_molar_flow_mol_sec"
            ],
            "Product molar flow [mol/s]": row[
                "product_molar_flow_mol_sec"
            ],
            "Residual [mol/s]": row[
                "residual_molar_flow_mol_sec"
            ],
            "Imbalance [%]": row["imbalance_pct"],
        }
        for row in component_balance_rows(result)
    ]
    return pd.DataFrame(records, columns=columns)


def _energy_balance_dataframe(result: Any) -> pd.DataFrame:
    """Build an explicit-unit system energy-closure summary."""
    columns = ["Term", "Value", "Unit"]
    summary = aggregate_energy_balance(result)
    if summary["applicable"] is not True:
        return pd.DataFrame(columns=columns)
    records = [
        {
            "Term": "Feed material enthalpy",
            "Value": summary["feed_enthalpy_kW"],
            "Unit": "kW",
        },
        {
            "Term": "Product material enthalpy",
            "Value": summary["product_enthalpy_kW"],
            "Unit": "kW",
        },
        {
            "Term": "Signed external energy transfer",
            "Value": summary["external_energy_transfer_kW"],
            "Unit": "kW",
        },
        {
            "Term": "Closure residual",
            "Value": summary["residual_kW"],
            "Unit": "kW",
        },
        {
            "Term": "Relative imbalance",
            "Value": summary["imbalance_pct"],
            "Unit": "%",
        },
    ]
    return pd.DataFrame(records, columns=columns)


def _energy_transfer_dataframe(result: Any) -> pd.DataFrame:
    """Build a signed shaft-work and heat-transfer table."""
    columns = ["Unit", "Type", "Transfer", "Energy transfer [kW]"]
    records = [
        {
            "Unit": row["unit_name"],
            "Type": row["unit_type"],
            "Transfer": (
                "Shaft work"
                if row["transfer_kind"] == "shaft_work"
                else "Heat"
            ),
            "Energy transfer [kW]": row["energy_transfer_kW"],
        }
        for row in energy_transfer_rows(result)
    ]
    return pd.DataFrame(records, columns=columns)


def _convergence_dataframe(result: Any) -> pd.DataFrame:
    """Build an explicit-unit native solver convergence table."""
    columns = [
        "Process system",
        "Unit",
        "Type",
        "State",
        "Iterations",
        "Maximum iterations",
        "Dominant residual",
        "Acceleration",
        "Flow error",
        "Flow tolerance",
        "Temperature error",
        "Temperature tolerance",
        "Pressure error",
        "Pressure tolerance",
        "Composition error",
        "Composition tolerance",
        "Target error",
        "Target tolerance",
    ]
    records = [
        {
            "Process system": row["process_system"],
            "Unit": row["unit_name"],
            "Type": row["unit_type"].title(),
            "State": (
                "Converged" if row["converged"] else "Not converged"
            ),
            "Iterations": row["iterations"],
            "Maximum iterations": row["max_iterations"],
            "Dominant residual": row["dominant_error"],
            "Acceleration": row["acceleration_method"],
            "Flow error": row["flow_error"],
            "Flow tolerance": row["flow_tolerance"],
            "Temperature error": row["temperature_error"],
            "Temperature tolerance": row["temperature_tolerance"],
            "Pressure error": row["pressure_error"],
            "Pressure tolerance": row["pressure_tolerance"],
            "Composition error": row["composition_error"],
            "Composition tolerance": row["composition_tolerance"],
            "Target error": row["error"],
            "Target tolerance": row["tolerance"],
        }
        for row in convergence_rows(result)
    ]
    return pd.DataFrame(records, columns=columns)


def _convergence_state_label(summary: dict[str, Any]) -> str:
    """Return a concise convergence label for reports and comparisons."""
    if summary["applicable"] is None:
        return "Not recorded"
    if summary["applicable"] is False:
        return "Feed-forward"
    return "Converged" if summary["converged"] else "Not converged"


def _unit_balance_dataframe(result: Any) -> pd.DataFrame:
    """Build an explicit-unit material and energy closure table."""
    columns = [
        "Process system",
        "Unit",
        "Type",
        "Inlets",
        "Outlets",
        "Inlet mass flow [kg/hr]",
        "Outlet mass flow [kg/hr]",
        "Mass residual [kg/hr]",
        "Mass imbalance [%]",
        "Inlet enthalpy flow [kW]",
        "Outlet enthalpy flow [kW]",
        "External energy transfer [kW]",
        "Energy residual [kW]",
        "Energy imbalance [%]",
    ]
    records = [
        {
            "Process system": row["process_system"],
            "Unit": row["unit_name"],
            "Type": row["unit_type"],
            "Inlets": row["inlet_count"],
            "Outlets": row["outlet_count"],
            "Inlet mass flow [kg/hr]": row["inlet_mass_flow_kg_hr"],
            "Outlet mass flow [kg/hr]": row["outlet_mass_flow_kg_hr"],
            "Mass residual [kg/hr]": row["mass_residual_kg_hr"],
            "Mass imbalance [%]": row["mass_imbalance_pct"],
            "Inlet enthalpy flow [kW]": row["inlet_enthalpy_kW"],
            "Outlet enthalpy flow [kW]": row["outlet_enthalpy_kW"],
            "External energy transfer [kW]": row[
                "external_energy_transfer_kW"
            ],
            "Energy residual [kW]": row["energy_residual_kW"],
            "Energy imbalance [%]": row["energy_imbalance_pct"],
        }
        for row in unit_balance_rows(result)
    ]
    return pd.DataFrame(records, columns=columns)


def _unit_identity_label(identity: dict[str, str] | None) -> str:
    """Format one process-system-scoped unit identity for reporting."""
    if identity is None:
        return "n/a"
    return (
        f"{identity['process_system']} / {identity['unit_name']} "
        f"({identity['unit_type']})"
    )


def _unit_balance_coverage_label(summary: dict[str, Any]) -> str:
    """Return a concise per-unit closure coverage label."""
    if summary["applicable"] is None:
        return "Not recorded"
    if summary["applicable"] is False:
        return (
            "Not applicable"
            if summary["coverage_complete"]
            else "Material unavailable; energy not audited"
        )
    material_label = (
        "Material complete"
        if summary["coverage_complete"]
        else "Material partial"
    )
    if summary["energy_unit_count"] == 0:
        energy_label = "energy not audited"
    elif summary["energy_coverage_complete"]:
        energy_label = "energy complete"
    else:
        energy_label = "energy partial"
    return f"{material_label}; {energy_label}"


def _kpi_value(result: Any, name: str) -> float | None:
    kpi = result.kpis.get(name)
    return float(kpi.value) if kpi is not None else None


def _format_metric(value: float | None, unit: str, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.{digits}f} {unit}"


def _neqsim_package_version() -> str:
    """Return the installed NeqSim Python package version when available."""
    try:
        return metadata.version("neqsim")
    except metadata.PackageNotFoundError:
        return "not reported"


def _solver_run_record(
    result: Any,
    model: Any,
    signature: str,
    execution_seconds: float,
    completed_at_utc: str | None = None,
    neqsim_version: str | None = None,
) -> dict[str, Any]:
    """Create deterministic provenance for one successful Studio execution."""
    elapsed_seconds = _finite_float(execution_seconds, "Execution wall time")
    if elapsed_seconds < 0.0:
        raise ValueError("Execution wall time cannot be negative.")

    validation_statuses = [
        str(getattr(item, "status", "UNKNOWN")).upper()
        for item in result.constraints
    ]
    validation_summary = aggregate_validation_status(validation_statuses)
    convergence_summary = aggregate_convergence(result)
    unit_balance_summary = aggregate_unit_balances(result)
    try:
        unit_count = len(model.list_units())
    except Exception:
        unit_count = None
    try:
        stream_count = len(model.list_streams())
    except Exception:
        stream_count = None

    return {
        "Execution status": "Solved",
        "Completed at [UTC]": completed_at_utc
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "Execution wall time [s]": round(elapsed_seconds, 3),
        "NeqSim package version": neqsim_version or _neqsim_package_version(),
        "Case fingerprint": signature[:12],
        "Unit operations": unit_count,
        "Indexed stream references": stream_count,
        "Validation checks": len(validation_statuses),
        "Validation summary": validation_summary,
        "Validation warnings / violations": sum(
            status in {"WARN", "VIOLATION"} for status in validation_statuses
        ),
        "Validation incomplete checks": sum(
            status == "UNKNOWN" for status in validation_statuses
        ),
        "Iterative convergence": _convergence_state_label(
            convergence_summary
        ),
        "Iterative solver units": convergence_summary["unit_count"],
        "Unconverged solver units": convergence_summary[
            "unconverged_count"
        ],
        "Maximum convergence iterations": convergence_summary[
            "max_iterations"
        ],
        "Per-unit closure coverage": _unit_balance_coverage_label(
            unit_balance_summary
        ),
        "Per-unit mass audits": unit_balance_summary["unit_count"],
        "Maximum unit mass imbalance [%]": unit_balance_summary[
            "max_mass_imbalance_pct"
        ],
        "Limiting mass-balance unit": _unit_identity_label(
            unit_balance_summary["max_mass_imbalance_unit"]
        ),
        "Per-unit energy audits": unit_balance_summary[
            "energy_unit_count"
        ],
        "Maximum unit energy imbalance [%]": unit_balance_summary[
            "max_energy_imbalance_pct"
        ],
        "Limiting energy-balance unit": _unit_identity_label(
            unit_balance_summary["max_energy_imbalance_unit"]
        ),
        "Units excluded from closure": ", ".join(
            unit_balance_summary["excluded_units"]
        ),
    }


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
    pressure_profile_table: pd.DataFrame,
    run_record: dict[str, Any],
) -> bytes:
    """Build a review-ready Excel workbook from one solved NeqSim case."""
    fluid = spec["fluid"]
    process_steps = {step["name"]: step for step in spec["process"]}
    total_power_kw = _kpi_value(result, "total_power_kW")
    total_duty_kw = _kpi_value(result, "total_duty_kW")
    mass_balance_pct = _kpi_value(result, "mass_balance_pct")
    convergence_summary = aggregate_convergence(result)
    unit_balance_summary = aggregate_unit_balances(result)
    feed_flow_kg_hr = solved_feed_flow_kg_hr(
        result,
        float(fluid["total_flow"]),
    )
    feed_tonnes_per_hour = feed_flow_kg_hr / 1000.0
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
                "Compressor stage 1",
                "Isentropic efficiency",
                process_steps["compressor stage 1"]["params"][
                    "isentropic_efficiency"
                ],
                "fraction",
            ),
            (
                "Compressor stage 2",
                "Isentropic efficiency",
                process_steps["compressor stage 2"]["params"][
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
                "Intercooler",
                "Pressure drop",
                process_steps["intercooler"]["params"]["pressure_drop_bar"],
                "bar",
            ),
            (
                "Export cooler",
                "Outlet temperature",
                process_steps["export cooler"]["params"]["outlet_temperature_C"],
                "°C",
            ),
            (
                "Export cooler",
                "Pressure drop",
                process_steps["export cooler"]["params"]["pressure_drop_bar"],
                "bar",
            ),
            (
                "Solver",
                "State represented by workbook",
                run_record.get("Execution status", "Not recorded"),
                "",
            ),
            (
                "Solver",
                "Completed",
                run_record.get("Completed at [UTC]", "Not recorded"),
                "UTC",
            ),
            (
                "Solver",
                "Build, solve, and serialization wall time",
                run_record.get("Execution wall time [s]"),
                "s",
            ),
            (
                "Solver",
                "Iterative convergence",
                _convergence_state_label(convergence_summary),
                "",
            ),
            (
                "Solver",
                "Iterative solver units",
                convergence_summary["unit_count"],
                "count",
            ),
            (
                "Solver",
                "Unconverged solver units",
                convergence_summary["unconverged_count"],
                "count",
            ),
            (
                "Solver",
                "Maximum convergence iterations",
                convergence_summary["max_iterations"],
                "iterations",
            ),
            (
                "Solver",
                "Per-unit closure coverage",
                _unit_balance_coverage_label(unit_balance_summary),
                "",
            ),
            (
                "Solver",
                "Per-unit mass audits",
                unit_balance_summary["unit_count"],
                "count",
            ),
            (
                "Solver",
                "Maximum unit mass imbalance",
                unit_balance_summary["max_mass_imbalance_pct"],
                "%",
            ),
            (
                "Solver",
                "Limiting mass-balance unit",
                _unit_identity_label(
                    unit_balance_summary["max_mass_imbalance_unit"]
                ),
                "",
            ),
            (
                "Solver",
                "Per-unit energy audits",
                unit_balance_summary["energy_unit_count"],
                "count",
            ),
            (
                "Solver",
                "Maximum unit energy imbalance",
                unit_balance_summary["max_energy_imbalance_pct"],
                "%",
            ),
            (
                "Solver",
                "Limiting energy-balance unit",
                _unit_identity_label(
                    unit_balance_summary["max_energy_imbalance_unit"]
                ),
                "",
            ),
            (
                "Solver",
                "Units excluded from per-unit closure",
                ", ".join(unit_balance_summary["excluded_units"]),
                "",
            ),
            (
                "Software",
                "NeqSim Python package version",
                run_record.get("NeqSim package version", "Not reported"),
                "",
            ),
            (
                "Reproducibility",
                "Case fingerprint",
                run_record.get("Case fingerprint", "Not recorded"),
                "SHA-256 prefix",
            ),
        ],
        columns=["Section", "Parameter", "Value", "Unit / basis"],
    )
    kpi_table = pd.DataFrame(
        [
            ("Total compressor power", total_power_kw, "kW"),
            ("Total cooling duty magnitude", total_duty_kw, "kW"),
            ("Solved aggregate feed flow", feed_flow_kg_hr, "kg/hr"),
            ("Specific compression energy", specific_energy, "kWh/t feed"),
            (
                "Maximum unit mass imbalance",
                unit_balance_summary["max_mass_imbalance_pct"],
                "%",
            ),
            (
                "Maximum unit energy imbalance",
                unit_balance_summary["max_energy_imbalance_pct"],
                "%",
            ),
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
    fluid_package_table = pd.DataFrame(
        [
            {
                "Package ID": package["id"],
                "Name": package["name"],
                "EOS": str(package["eos_model"]).upper(),
                "Mixing rule": package["mixing_rule"],
                "Registered components": len(package["component_registry"]),
                "Interaction source": package[
                    "binary_interaction_parameters"
                ]["source"],
            }
            for package in spec["fluid_packages"]
        ]
    )
    inlet_table = pd.DataFrame(
        [
            {
                "Inlet ID": inlet["id"],
                "Name": inlet["name"],
                "Fluid package": inlet["fluid_package_id"],
                "Temperature [°C]": inlet["temperature_C"],
                "Pressure [bara]": inlet["pressure_bara"],
                "Flow": inlet["total_flow"],
                "Flow unit": inlet["flow_unit"],
                "Composition basis": inlet["composition_basis"],
                "Components": len(inlet["composition"]),
            }
            for inlet in spec["inlets"]
        ]
    )
    unit_table = pd.DataFrame(
        [
            {
                "Unit ID": unit["id"],
                "Name": unit["name"],
                "Type": unit["type"],
                "Material inlet ports": ", ".join(
                    unit["ports"]["material_in"]
                ),
                "Material outlet ports": ", ".join(
                    unit["ports"]["material_out"]
                ),
                "Builder settings": json.dumps(
                    unit.get("params", {}),
                    sort_keys=True,
                ),
                "Builder outlet": unit.get("builder_outlet", ""),
            }
            for unit in spec["units"]
        ]
    )
    connection_table = pd.DataFrame(
        [
            {
                "Connection ID": connection["id"],
                "Type": connection["type"],
                "Source kind": connection["source"]["kind"],
                "Source ID": connection["source"]["id"],
                "Source port": connection["source"]["port"],
                "Target kind": connection["target"]["kind"],
                "Target ID": connection["target"]["id"],
                "Target port": connection["target"]["port"],
            }
            for connection in spec["connections"]
        ]
    )
    execution_plan_table = pd.DataFrame(_build_execution_plan(spec))
    inlet_fluid_spec_table = pd.DataFrame(
        [
            {
                "Inlet ID": inlet_spec["inlet_id"],
                "Name": inlet_spec["name"],
                "Fluid package": inlet_spec["fluid_package_id"],
                "EOS": inlet_spec["fluid_spec"]["eos_model"].upper(),
                "Mixing rule": inlet_spec["fluid_spec"]["mixing_rule"],
                "Composition basis": inlet_spec["fluid_spec"][
                    "composition_basis"
                ],
                "Composition": json.dumps(
                    inlet_spec["fluid_spec"]["components"],
                    sort_keys=True,
                ),
                "Temperature [°C]": inlet_spec["fluid_spec"]["temperature_C"],
                "Pressure [bara]": inlet_spec["fluid_spec"]["pressure_bara"],
                "Mass flow [kg/hr]": inlet_spec["fluid_spec"]["total_flow"],
                "Interaction source": inlet_spec["characterization"][
                    "binary_interaction_parameters"
                ]["source"],
            }
            for inlet_spec in _build_inlet_fluid_specs(spec)
        ]
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
    material_boundary_table = _material_boundary_dataframe(result)
    component_balance_table = _component_balance_dataframe(result)
    energy_balance_table = _energy_balance_dataframe(result)
    energy_transfer_table = _energy_transfer_dataframe(result)
    convergence_table = _convergence_dataframe(result)
    unit_balance_table = _unit_balance_dataframe(result)
    sheet_frames = {
        "Case Summary": case_summary,
        "KPIs": kpi_table,
        "Composition": composition_table,
        "Fluid Package": fluid_package_table,
        "Inlets": inlet_table,
        "Units": unit_table,
        "Connections": connection_table,
        "Execution Plan": execution_plan_table,
        "Inlet Build Specs": inlet_fluid_spec_table,
        "Material Balance": material_boundary_table,
        "Component Balance": component_balance_table,
        "Energy Balance": energy_balance_table,
        "Energy Transfers": energy_transfer_table,
        "Convergence": convergence_table,
        "Unit Balances": unit_balance_table,
        "Streams": stream_table,
        "Equipment": equipment_table,
        "Validation": constraint_table,
        "Pressure Profile": pressure_profile_table,
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
    component_balance_pct = _kpi_value(
        result,
        "component_balance_max_pct",
    )
    feed_flow_kg_hr = solved_feed_flow_kg_hr(
        result,
        float(spec["fluid"]["total_flow"]),
    )
    feed_tonnes_per_hour = feed_flow_kg_hr / 1000.0
    specific_energy_kwh_t = None
    if total_power_kw is not None and feed_tonnes_per_hour > 0.0:
        specific_energy_kwh_t = total_power_kw / feed_tonnes_per_hour

    constraint_statuses = [
        str(getattr(constraint, "status", "")).upper()
        for constraint in result.constraints
    ]
    validation_status = aggregate_validation_status(constraint_statuses)
    convergence_summary = aggregate_convergence(result)

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
        "Stage 1 efficiency [-]": float(
            process[2]["params"]["isentropic_efficiency"]
        ),
        "Stage 2 efficiency [-]": float(
            process[5]["params"]["isentropic_efficiency"]
        ),
        "Intercooler pressure drop [bar]": float(
            process[3]["params"]["pressure_drop_bar"]
        ),
        "Export cooler pressure drop [bar]": float(
            process[6]["params"]["pressure_drop_bar"]
        ),
        "Compressor power [kW]": total_power_kw,
        "Cooling duty magnitude [kW]": total_duty_kw,
        "Specific energy [kWh/t]": specific_energy_kwh_t,
        "Mass imbalance [%]": mass_balance_pct,
        "Max component imbalance [%]": component_balance_pct,
        "Iterative convergence": _convergence_state_label(
            convergence_summary
        ),
        "Max convergence iterations": convergence_summary[
            "max_iterations"
        ],
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


def _render_object_property_editor() -> str:
    """Render supported properties and return the selected template object."""
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

    if selected_object == "feed gas":
        st.info(
            "Feed temperature, absolute pressure, mass flow, equation of state, "
            "and molar composition are edited in the fluid basis."
        )
    else:
        property_controls = TEMPLATE_PROPERTY_CONTROLS.get(
            selected_object,
            {},
        )
        unit_type = object_type.casefold()
        params = {
            property_name: st.session_state[definition["state_key"]]
            for property_name, definition in property_controls.items()
        }
        property_rows = process_unit_property_rows(unit_type, params)
        if not property_rows:
            st.info(
                "This separator performs an equilibrium split at its inlet "
                "conditions. The native unit has no independent steady-state "
                "property in the current schema."
            )
        for row in property_rows:
            control = property_controls.get(row["key"])
            if control is None:
                raise ValueError(
                    f"Template object '{selected_object}' is missing the "
                    f"'{row['key']}' control binding."
                )
            minimum = max(float(row["minimum"]), control["minimum"])
            maximum = min(float(row["maximum"]), control["maximum"])
            st.number_input(
                f"{row['label']} [{row['unit']}]",
                min_value=minimum,
                max_value=maximum,
                step=float(row["step"]),
                format=row["format"],
                key=control["state_key"],
            )

    st.caption(
        "Property edits update the structured case specification. Run NeqSim to "
        "solve the edited case and refresh Process Chat."
    )
    return selected_object


def _graph_history_for_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Return current history, bootstrapping an imported graph when needed."""
    existing_history = st.session_state.get(GRAPH_HISTORY_STATE_KEY)
    if existing_history is not None:
        graph_history_status(existing_history)
        return existing_history

    starter_units, starter_connections = _build_template_graph(
        spec["process"]
    )
    history = create_graph_history(
        starter_units,
        starter_connections,
        spec["inlets"],
    )
    if (
        spec["units"] != starter_units
        or spec["connections"] != starter_connections
    ):
        history = record_graph_history(
            history,
            spec["units"],
            spec["connections"],
            spec["inlets"],
        )
        st.session_state[GRAPH_HISTORY_STATE_KEY] = history
    return history


def _reconcile_inlet_composition(
    composition: dict[str, Any],
    registry_composition: dict[str, Any],
) -> dict[str, float]:
    """Rebase an inlet composition onto a changed shared component registry."""
    if not isinstance(composition, dict):
        raise ValueError("Inlet composition must be an object.")
    if not isinstance(registry_composition, dict) or not registry_composition:
        raise ValueError("Shared registry composition must be non-empty.")

    values: dict[str, float] = {}
    fallback_values: dict[str, float] = {}
    for component_name, raw_fallback in registry_composition.items():
        if isinstance(raw_fallback, bool):
            raise ValueError("Shared registry mole fractions must be numeric.")
        try:
            fallback = float(raw_fallback)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Shared registry mole fractions must be numeric."
            ) from error
        if not math.isfinite(fallback) or fallback < 0.0:
            raise ValueError(
                "Shared registry mole fractions must be finite and non-negative."
            )
        fallback_values[component_name] = fallback

        raw_value = composition.get(component_name, fallback)
        if isinstance(raw_value, bool):
            raise ValueError("Inlet mole fractions must be numeric.")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError("Inlet mole fractions must be numeric.") from error
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "Inlet mole fractions must be finite and non-negative."
            )
        values[component_name] = value

    total = sum(values.values())
    if total <= 0.0:
        values = fallback_values
        total = sum(values.values())
    if total <= 0.0:
        raise ValueError("Reconciled inlet mole fractions must have a positive sum.")
    return {
        component_name: value / total
        for component_name, value in values.items()
    }


def _apply_studio_graph_draft(
    case_spec: dict[str, Any],
    draft: dict[str, Any],
) -> dict[str, Any]:
    """Apply graph edits while keeping the primary inlet tied to case controls."""
    if not isinstance(case_spec, dict) or not isinstance(draft, dict):
        return apply_graph_draft(case_spec, draft)
    if "inlets" not in draft:
        return apply_graph_draft(case_spec, draft)

    refreshed_draft = json.loads(json.dumps(draft, allow_nan=False))
    case_process = case_spec.get("process")
    draft_units = refreshed_draft.get("units")
    if isinstance(case_process, list) and isinstance(draft_units, list):
        current_template_units, _ = _build_template_graph(case_process)
        current_template_by_id = {
            str(unit["id"]).strip(): unit
            for unit in current_template_units
        }
        refreshed_draft["units"] = [
            json.loads(
                json.dumps(
                    current_template_by_id.get(
                        str(unit.get("id", "")).strip(),
                        unit,
                    ),
                    allow_nan=False,
                )
            )
            if isinstance(unit, dict)
            else unit
            for unit in draft_units
        ]
    case_inlets = case_spec.get("inlets")
    draft_inlets = refreshed_draft.get("inlets")
    if not isinstance(case_inlets, list) or not isinstance(draft_inlets, list):
        return apply_graph_draft(case_spec, refreshed_draft)
    primary_case_inlets = [
        inlet
        for inlet in case_inlets
        if isinstance(inlet, dict)
        and str(inlet.get("id", "")).strip() == PRIMARY_INLET_ID
    ]
    primary_draft_indices = [
        index
        for index, inlet in enumerate(draft_inlets)
        if isinstance(inlet, dict)
        and str(inlet.get("id", "")).strip() == PRIMARY_INLET_ID
    ]
    if len(primary_case_inlets) != 1 or len(primary_draft_indices) != 1:
        raise ValueError(
            f"Graph drafts require exactly one primary inlet '{PRIMARY_INLET_ID}'."
        )
    refreshed_draft["inlets"][primary_draft_indices[0]] = json.loads(
        json.dumps(primary_case_inlets[0], allow_nan=False)
    )
    primary_inlet = primary_case_inlets[0]
    primary_package_id = str(
        primary_inlet.get("fluid_package_id", "")
    ).strip()
    primary_composition = primary_inlet.get("composition")
    if primary_package_id and isinstance(primary_composition, dict):
        for index, inlet in enumerate(refreshed_draft["inlets"]):
            if index == primary_draft_indices[0] or not isinstance(inlet, dict):
                continue
            if (
                str(inlet.get("fluid_package_id", "")).strip()
                != primary_package_id
            ):
                continue
            composition = inlet.get("composition")
            if (
                isinstance(composition, dict)
                and set(composition) != set(primary_composition)
            ):
                inlet["composition"] = _reconcile_inlet_composition(
                    composition,
                    primary_composition,
                )
    return apply_graph_draft(case_spec, refreshed_draft)


def _activate_graph_revision(
    spec: dict[str, Any],
    history: dict[str, Any],
    draft: dict[str, Any],
    notice: str,
) -> None:
    """Validate and activate one history revision without stale solve state."""
    candidate_case = _apply_studio_graph_draft(spec, draft)
    _validate_case_graph(candidate_case, candidate_case["process"])
    starter_units, starter_connections = _build_template_graph(
        candidate_case["process"]
    )
    draft_inlets = draft.get("inlets")
    has_secondary_inlets = isinstance(draft_inlets, list) and any(
        isinstance(inlet, dict)
        and str(inlet.get("id", "")).strip() != PRIMARY_INLET_ID
        for inlet in draft_inlets
    )
    if (
        draft["units"] == starter_units
        and draft["connections"] == starter_connections
        and not has_secondary_inlets
    ):
        st.session_state.pop(GRAPH_DRAFT_STATE_KEY, None)
    else:
        st.session_state[GRAPH_DRAFT_STATE_KEY] = draft
    st.session_state[GRAPH_HISTORY_STATE_KEY] = history
    _clear_studio_runtime(clear_history=False)
    st.session_state[CASE_NOTICE_STATE_KEY] = notice


def _record_graph_revision(
    spec: dict[str, Any],
    draft: dict[str, Any],
    notice: str,
) -> None:
    """Record and activate one accepted graph edit."""
    history = _graph_history_for_spec(spec)
    history = record_graph_history(
        history,
        draft["units"],
        draft["connections"],
        draft.get("inlets"),
    )
    _activate_graph_revision(spec, history, draft, notice)


def _render_graph_palette(spec: dict[str, Any]) -> None:
    """Render safe lifecycle controls for the active unsolved graph."""
    catalog = inline_unit_catalog()
    catalog_rows = inline_unit_catalog_rows()
    connection_rows = material_connection_rows(spec["connections"])
    connection_labels = {
        row["id"]: row["label"] for row in connection_rows
    }
    all_connection_rows = graph_connection_rows(
        spec["inlets"],
        spec["units"],
        spec["connections"],
    )
    all_connection_labels = {
        row["id"]: row["label"] for row in all_connection_rows
    }
    graph_widget_revision = hashlib.sha256(
        json.dumps(
            {
                "inlets": spec["inlets"],
                "units": spec["units"],
                "connections": spec["connections"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:12]
    protected_unit_ids = set(TEMPLATE_UNIT_IDS.values())
    protected_unit_names = _graph_name_set(
        _build_template_graph(spec["process"])[0]
    )
    added_units = [
        unit
        for unit in spec["units"]
        if isinstance(unit, dict)
        and str(unit.get("id", "")).strip() not in protected_unit_ids
    ]
    added_unit_map = {
        str(unit["id"]).strip(): unit for unit in added_units
    }

    with st.expander("Edit flowsheet graph", expanded=False):
        st.markdown("#### Draft flowsheet")
        st.caption(
            "Automatically laid out from the active inlet, unit, port, and "
            "connection schema before NeqSim execution."
        )
        try:
            draft_dot = build_graph_draft_dot(
                spec["inlets"],
                spec["units"],
                spec["connections"],
            )
        except ValueError as preview_error:
            st.error(f"Draft flowsheet preview failed: {preview_error}")
        else:
            st.graphviz_chart(
                draft_dot,
                use_container_width=True,
            )
            st.caption(
                "Blue solid paths are material streams; amber dashed paths "
                "are energy links. Oval nodes mark inlet and product boundaries."
            )

        st.divider()
        graph_history = _graph_history_for_spec(spec)
        history_status = graph_history_status(graph_history)
        st.markdown("#### Edit history")
        history_cols = st.columns(3)
        history_cols[0].caption(
            "Graph revision "
            f"{history_status['position']} of {history_status['total']}"
        )
        undo_edit = history_cols[1].button(
            "Undo graph edit",
            disabled=not history_status["can_undo"],
            use_container_width=True,
            key="flowsheet_undo_graph_edit",
        )
        redo_edit = history_cols[2].button(
            "Redo graph edit",
            disabled=not history_status["can_redo"],
            use_container_width=True,
            key="flowsheet_redo_graph_edit",
        )
        if undo_edit or redo_edit:
            try:
                if undo_edit:
                    graph_history, selected_draft = undo_graph_history(
                        graph_history
                    )
                    navigation_notice = (
                        "Undid the latest graph edit. "
                        "Run NeqSim to solve this revision."
                    )
                else:
                    graph_history, selected_draft = redo_graph_history(
                        graph_history
                    )
                    navigation_notice = (
                        "Redid the next graph edit. "
                        "Run NeqSim to solve this revision."
                    )
                _activate_graph_revision(
                    spec,
                    graph_history,
                    selected_draft,
                    navigation_notice,
                )
            except ValueError as history_error:
                st.error(f"Graph history navigation failed: {history_error}")
            else:
                st.rerun()

        st.divider()
        st.markdown("#### Reorganize an existing process")
        st.caption(
            "Select an existing material path and change it in one undoable "
            "transaction. You do not need to disconnect the process first."
        )
        if connection_rows:
            reorganize_connection_id = st.selectbox(
                "Material path to reorganize",
                options=list(connection_labels),
                format_func=connection_labels.__getitem__,
                key=(
                    "flowsheet_reorganize_connection_"
                    f"{graph_widget_revision}"
                ),
            )
            reorganize_connection = next(
                connection
                for connection in spec["connections"]
                if (
                    isinstance(connection, dict)
                    and str(connection.get("id", "")).strip()
                    == reorganize_connection_id
                )
            )
            current_source = reorganize_connection["source"]
            current_target = reorganize_connection["target"]

            all_material_sources = graph_port_rows(
                spec["inlets"],
                spec["units"],
                spec["connections"],
                "material",
                "source",
            )
            all_material_targets = graph_port_rows(
                spec["inlets"],
                spec["units"],
                spec["connections"],
                "material",
                "target",
            )

            def candidate_port_rows(
                rows: list[dict[str, Any]],
                current_endpoint: dict[str, Any],
            ) -> list[dict[str, Any]]:
                current_key = tuple(
                    str(current_endpoint.get(field, "")).strip()
                    for field in ("kind", "id", "port")
                )
                candidates = [
                    row
                    for row in rows
                    if (
                        not row["connected"]
                        or tuple(
                            str(row["endpoint"].get(field, "")).strip()
                            for field in ("kind", "id", "port")
                        )
                        == current_key
                    )
                ]
                return sorted(
                    candidates,
                    key=lambda row: (
                        tuple(
                            str(row["endpoint"].get(field, "")).strip()
                            for field in ("kind", "id", "port")
                        )
                        != current_key,
                        row["label"].casefold(),
                    ),
                )

            source_candidates = candidate_port_rows(
                all_material_sources,
                current_source,
            )
            target_candidates = candidate_port_rows(
                all_material_targets,
                current_target,
            )
            current_source_label = source_candidates[0]["label"]
            current_target_label = target_candidates[0]["label"]
            st.info(
                f"Current route: {current_source_label} → "
                f"{current_target_label}"
            )

            (
                insert_mixer_tab,
                equipment_tab,
                reconnect_tab,
                disconnect_tab,
            ) = st.tabs(
                [
                    "Insert mixer",
                    "Equipment",
                    "Reconnect",
                    "Disconnect",
                ]
            )
            with insert_mixer_tab:
                st.caption(
                    "The current source will enter mixer port in_0. The mixer "
                    "outlet will reconnect to the current downstream target, "
                    "while in_1 remains available for another feed."
                )
                mixer_name = st.text_input(
                    "Mixer name",
                    value="Feed mixer",
                    max_chars=80,
                    key=(
                        "flowsheet_reorganize_mixer_name_"
                        f"{graph_widget_revision}"
                    ),
                )
                resolved_mixer_name = mixer_name.strip() or "Feed mixer"
                secondary_source_candidates = [
                    row
                    for row in all_material_sources
                    if not row["connected"]
                ]
                selected_secondary_source = None
                if secondary_source_candidates:
                    secondary_source_index = st.selectbox(
                        "Second mixer inlet",
                        options=[
                            None,
                            *range(len(secondary_source_candidates)),
                        ],
                        format_func=lambda index: (
                            "Leave in_1 unconnected for now"
                            if index is None
                            else secondary_source_candidates[index]["label"]
                        ),
                        key=(
                            "flowsheet_reorganize_mixer_second_source_"
                            f"{graph_widget_revision}"
                        ),
                        help=(
                            "Choose another feed or material outlet to make "
                            "the inserted mixer immediately solve-ready."
                        ),
                    )
                    if secondary_source_index is not None:
                        selected_secondary_source = (
                            secondary_source_candidates[
                                secondary_source_index
                            ]
                        )
                else:
                    st.caption(
                        "No free material source is currently available for "
                        "mixer in_1."
                    )
                secondary_preview = (
                    f"{selected_secondary_source['label']} → "
                    f"{resolved_mixer_name}:in_1"
                    if selected_secondary_source is not None
                    else (
                        f"{resolved_mixer_name}:in_1 remains free and must be "
                        "connected before solving"
                    )
                )
                st.markdown(
                    f"**Preview:** {current_source_label} → "
                    f"{resolved_mixer_name}:in_0 → "
                    f"{resolved_mixer_name}:out → "
                    f"{current_target_label}; {secondary_preview}."
                )
                insert_mixer = st.button(
                    "Insert mixer and preserve downstream path",
                    use_container_width=True,
                    key=(
                        "flowsheet_reorganize_insert_mixer_"
                        f"{graph_widget_revision}"
                    ),
                )
                if insert_mixer:
                    try:
                        (
                            units,
                            connections,
                            mixer_id,
                            downstream_connection_id,
                        ) = insert_mixer_on_connection(
                            spec["inlets"],
                            spec["units"],
                            spec["connections"],
                            reorganize_connection_id,
                            resolved_mixer_name,
                            protected_unit_ids,
                        )
                        secondary_connection_id = None
                        if selected_secondary_source is not None:
                            (
                                connections,
                                secondary_connection_id,
                            ) = connect_graph_ports(
                                spec["inlets"],
                                units,
                                connections,
                                "material",
                                selected_secondary_source["endpoint"],
                                {
                                    "kind": "unit",
                                    "id": mixer_id,
                                    "port": "in_1",
                                },
                            )
                        candidate_draft = create_graph_draft(
                            units,
                            connections,
                            spec["inlets"],
                        )
                        candidate_case = _apply_studio_graph_draft(
                            spec,
                            candidate_draft,
                        )
                        _validate_case_graph(
                            candidate_case,
                            candidate_case["process"],
                        )
                    except ValueError as edit_error:
                        st.error(
                            "Mixer insertion failed without changing the "
                            f"draft: {edit_error}"
                        )
                    else:
                        if secondary_connection_id is None:
                            solve_notice = (
                                f"Left {mixer_id}:in_1 available. Connect it "
                                "before running NeqSim."
                            )
                        else:
                            solve_notice = (
                                "Connected the second source with "
                                f"'{secondary_connection_id}'. The reorganized "
                                "graph is ready to run in NeqSim."
                            )
                        _record_graph_revision(
                            spec,
                            candidate_draft,
                            (
                                f"Inserted mixer '{mixer_id}' in "
                                f"'{reorganize_connection_id}', preserved the "
                                "downstream process with "
                                f"'{downstream_connection_id}'. "
                                f"{solve_notice}"
                            ),
                        )
                        st.rerun()

            with equipment_tab:
                st.caption(
                    "Insert new equipment into this path or replace its "
                    "downstream unit without manually rebuilding adjacent "
                    "connections."
                )
                inline_insert_types = [
                    unit_type
                    for unit_type, definition in catalog.items()
                    if definition["ports"].get("material_in") == ["in"]
                    and definition["ports"].get("material_out") == ["out"]
                ]
                equipment_action = st.radio(
                    "Equipment action",
                    options=[
                        "Insert equipment in this path",
                        "Replace downstream equipment",
                    ],
                    horizontal=True,
                    key=(
                        "flowsheet_reorganize_equipment_action_"
                        f"{graph_widget_revision}"
                    ),
                )
                reorganize_unit_type = st.selectbox(
                    "Equipment type",
                    options=inline_insert_types,
                    format_func=lambda value: (
                        f"{catalog[value]['label']} · "
                        f"{catalog[value]['category']}"
                    ),
                    key=(
                        "flowsheet_reorganize_equipment_type_"
                        f"{graph_widget_revision}"
                    ),
                )
                reorganize_definition = catalog[reorganize_unit_type]
                reorganize_unit_name = st.text_input(
                    "Equipment name",
                    value=f"New {reorganize_definition['label']}",
                    max_chars=80,
                    key=(
                        "flowsheet_reorganize_equipment_name_"
                        f"{graph_widget_revision}"
                    ),
                )
                requested_name = (
                    reorganize_unit_name.strip()
                    or f"New {reorganize_definition['label']}"
                )
                downstream_unit = None
                if str(current_target.get("kind", "")).strip() == "unit":
                    downstream_unit = next(
                        (
                            unit
                            for unit in spec["units"]
                            if (
                                isinstance(unit, dict)
                                and str(unit.get("id", "")).strip()
                                == str(current_target.get("id", "")).strip()
                            )
                        ),
                        None,
                    )
                if equipment_action == "Replace downstream equipment":
                    retained_name_records = [
                        unit
                        for unit in spec["units"]
                        if (
                            isinstance(unit, dict)
                            and (
                                not isinstance(downstream_unit, dict)
                                or str(unit.get("id", "")).strip()
                                != str(
                                    downstream_unit.get("id", "")
                                ).strip()
                            )
                        )
                    ]
                else:
                    retained_name_records = spec["units"]
                existing_names = _graph_name_set(
                    retained_name_records,
                    casefold=True,
                )
                existing_names.update(
                    _graph_name_set(spec["inlets"], casefold=True)
                )
                resolved_name = requested_name
                name_suffix = 2
                while resolved_name.casefold() in existing_names:
                    resolved_name = f"{requested_name} {name_suffix}"
                    name_suffix += 1

                if equipment_action == "Insert equipment in this path":
                    st.markdown(
                        f"**Preview:** {current_source_label} → "
                        f"{resolved_name}:in → {resolved_name}:out → "
                        f"{current_target_label}"
                    )
                    equipment_button_label = (
                        "Insert equipment and preserve downstream path"
                    )
                    equipment_action_ready = True
                else:
                    if downstream_unit is None:
                        st.warning(
                            "This path does not terminate at equipment. Select "
                            "a path whose downstream endpoint is a unit."
                        )
                        equipment_action_ready = False
                    else:
                        try:
                            replace_inline_unit(
                                spec["units"],
                                spec["connections"],
                                str(downstream_unit["id"]).strip(),
                                reorganize_unit_type,
                                resolved_name,
                                {
                                    *protected_unit_ids,
                                    *(
                                        str(inlet.get("id", "")).strip()
                                        for inlet in spec["inlets"]
                                        if isinstance(inlet, dict)
                                    ),
                                },
                                _graph_name_set(spec["inlets"]),
                            )
                        except ValueError as replacement_blocker:
                            st.warning(
                                "This downstream unit cannot be replaced as "
                                "one continuous path: "
                                f"{replacement_blocker}"
                            )
                            equipment_action_ready = False
                        else:
                            downstream_name = _graph_object_name(
                                downstream_unit,
                                str(downstream_unit.get("id", "")).strip(),
                            )
                            st.markdown(
                                f"**Preview:** replace {downstream_name} with "
                                f"{resolved_name}; retain its upstream and "
                                "downstream material paths."
                            )
                            equipment_action_ready = True
                    equipment_button_label = (
                        "Replace equipment and preserve surrounding path"
                    )

                st.caption("Initial properties and engineering units")
                default_property_rows = inline_unit_property_rows(
                    reorganize_unit_type
                )
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Property": row["label"],
                                "Value": row["value"],
                                "Unit": row["unit"],
                            }
                            for row in default_property_rows
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                change_equipment = st.button(
                    equipment_button_label,
                    disabled=not equipment_action_ready,
                    use_container_width=True,
                    key=(
                        "flowsheet_reorganize_change_equipment_"
                        f"{graph_widget_revision}"
                    ),
                )
                if change_equipment:
                    try:
                        if (
                            equipment_action
                            == "Insert equipment in this path"
                        ):
                            units, connections, changed_unit_id = (
                                insert_inline_unit_on_connection(
                                    spec["units"],
                                    spec["connections"],
                                    reorganize_connection_id,
                                    reorganize_unit_type,
                                    resolved_name,
                                    {
                                        *protected_unit_ids,
                                        *(
                                            str(inlet.get("id", "")).strip()
                                            for inlet in spec["inlets"]
                                            if isinstance(inlet, dict)
                                        ),
                                    },
                                )
                            )
                            action_notice = (
                                f"Inserted '{resolved_name}' "
                                f"({changed_unit_id}) in "
                                f"'{reorganize_connection_id}' while "
                                "preserving the downstream process."
                            )
                        else:
                            units, connections, changed_unit_id = (
                                replace_inline_unit(
                                    spec["units"],
                                    spec["connections"],
                                    str(downstream_unit["id"]).strip(),
                                    reorganize_unit_type,
                                    resolved_name,
                                    {
                                        *protected_unit_ids,
                                        *(
                                            str(inlet.get("id", "")).strip()
                                            for inlet in spec["inlets"]
                                            if isinstance(inlet, dict)
                                        ),
                                    },
                                    _graph_name_set(spec["inlets"]),
                                )
                            )
                            replaced_label = _graph_object_name(
                                downstream_unit,
                                str(
                                    downstream_unit.get("id", "")
                                ).strip(),
                            )
                            action_notice = (
                                f"Replaced '{replaced_label}' with "
                                f"'{resolved_name}' ({changed_unit_id}) while "
                                "preserving its surrounding material path."
                            )
                        candidate_draft = create_graph_draft(
                            units,
                            connections,
                            spec["inlets"],
                        )
                        candidate_case = _apply_studio_graph_draft(
                            spec,
                            candidate_draft,
                        )
                        _validate_case_graph(
                            candidate_case,
                            candidate_case["process"],
                        )
                    except ValueError as edit_error:
                        st.error(
                            "Equipment change failed without changing the "
                            f"draft: {edit_error}"
                        )
                    else:
                        _record_graph_revision(
                            spec,
                            candidate_draft,
                            action_notice + " Run NeqSim to solve this revision.",
                        )
                        st.rerun()

            with reconnect_tab:
                st.caption(
                    "Replace the selected path's source and target together. "
                    "The old endpoints are released only after the new route "
                    "passes port, occupancy, and cycle validation."
                )
                reconnect_cols = st.columns(2)
                replacement_source_index = reconnect_cols[0].selectbox(
                    "New source",
                    options=list(range(len(source_candidates))),
                    format_func=lambda index: source_candidates[index]["label"],
                    key=(
                        "flowsheet_reorganize_source_"
                        f"{graph_widget_revision}"
                    ),
                )
                replacement_target_index = reconnect_cols[1].selectbox(
                    "New target",
                    options=list(range(len(target_candidates))),
                    format_func=lambda index: target_candidates[index]["label"],
                    key=(
                        "flowsheet_reorganize_target_"
                        f"{graph_widget_revision}"
                    ),
                )
                replacement_source = source_candidates[
                    replacement_source_index
                ]
                replacement_target = target_candidates[
                    replacement_target_index
                ]
                route_changed = (
                    replacement_source["endpoint"] != current_source
                    or replacement_target["endpoint"] != current_target
                )
                st.markdown(
                    f"**Preview:** {replacement_source['label']} → "
                    f"{replacement_target['label']}"
                )
                reconnect_path = st.button(
                    "Replace selected path",
                    disabled=not route_changed,
                    use_container_width=True,
                    key=(
                        "flowsheet_reorganize_replace_"
                        f"{graph_widget_revision}"
                    ),
                )
                if reconnect_path:
                    try:
                        connections = reroute_graph_connection(
                            spec["inlets"],
                            spec["units"],
                            spec["connections"],
                            reorganize_connection_id,
                            replacement_source["endpoint"],
                            replacement_target["endpoint"],
                        )
                        candidate_draft = create_graph_draft(
                            spec["units"],
                            connections,
                            spec["inlets"],
                        )
                        candidate_case = _apply_studio_graph_draft(
                            spec,
                            candidate_draft,
                        )
                        _validate_case_graph(
                            candidate_case,
                            candidate_case["process"],
                        )
                    except ValueError as edit_error:
                        st.error(
                            "Path replacement failed without changing the "
                            f"draft: {edit_error}"
                        )
                    else:
                        _record_graph_revision(
                            spec,
                            candidate_draft,
                            (
                                f"Reconnected '{reorganize_connection_id}' as "
                                f"{replacement_source['label']} → "
                                f"{replacement_target['label']}. Run NeqSim "
                                "to solve the reorganized graph."
                            ),
                        )
                        st.rerun()

            with disconnect_tab:
                st.caption(
                    "Remove only this explicit path. Both endpoints remain "
                    "available for a new connection, and undo restores the "
                    "original route."
                )
                st.markdown(
                    f"**Preview:** remove {current_source_label} → "
                    f"{current_target_label}"
                )
                confirm_disconnect = st.checkbox(
                    "I understand this leaves an unsolved graph draft",
                    key=(
                        "flowsheet_reorganize_confirm_disconnect_"
                        f"{graph_widget_revision}"
                    ),
                )
                disconnect_path = st.button(
                    "Disconnect this path",
                    disabled=not confirm_disconnect,
                    use_container_width=True,
                    key=(
                        "flowsheet_reorganize_disconnect_"
                        f"{graph_widget_revision}"
                    ),
                )
                if disconnect_path:
                    try:
                        connections = disconnect_graph_connection(
                            spec["inlets"],
                            spec["units"],
                            spec["connections"],
                            reorganize_connection_id,
                        )
                        candidate_draft = create_graph_draft(
                            spec["units"],
                            connections,
                            spec["inlets"],
                        )
                        candidate_case = _apply_studio_graph_draft(
                            spec,
                            candidate_draft,
                        )
                        _validate_case_graph(
                            candidate_case,
                            candidate_case["process"],
                        )
                    except ValueError as edit_error:
                        st.error(
                            "Path disconnection failed without changing the "
                            f"draft: {edit_error}"
                        )
                    else:
                        _record_graph_revision(
                            spec,
                            candidate_draft,
                            (
                                f"Disconnected '{reorganize_connection_id}'. "
                                "Reconnect the available endpoints before "
                                "running NeqSim, or use undo to restore it."
                            ),
                        )
                        st.rerun()
        else:
            st.info(
                "Add or connect a material path before reorganizing the "
                "process."
            )

        st.divider()
        st.markdown("#### Add an independent feed stream")
        st.caption(
            "Clone a compatible inlet from the shared fluid package, then edit "
            "its own temperature, pressure, flow, and molar composition below. "
            "The new feed starts with an available material outlet port."
        )
        inlet_map = {
            str(inlet["id"]).strip(): inlet
            for inlet in spec["inlets"]
            if isinstance(inlet, dict)
            and str(inlet.get("id", "")).strip()
        }
        feed_cols = st.columns(2)
        source_inlet_id = feed_cols[0].selectbox(
            "Copy fluid basis and conditions from",
            options=list(inlet_map),
            format_func=lambda value: (
                f"{_graph_object_name(inlet_map[value], value)} · {value}"
            ),
            key=f"flowsheet_new_feed_source_{graph_widget_revision}",
        )
        new_feed_name = feed_cols[1].text_input(
            "New feed name",
            value="New feed",
            max_chars=80,
            key=f"flowsheet_new_feed_name_{graph_widget_revision}",
        )
        add_feed = st.button(
            "Add feed stream",
            use_container_width=True,
            key=f"flowsheet_add_feed_{graph_widget_revision}",
        )
        if add_feed:
            try:
                inlets, new_inlet_id = clone_material_inlet(
                    spec["inlets"],
                    source_inlet_id,
                    new_feed_name,
                    {
                        *protected_unit_ids,
                        *(
                            str(unit.get("id", "")).strip()
                            for unit in spec["units"]
                            if isinstance(unit, dict)
                            and str(unit.get("id", "")).strip()
                        ),
                    },
                    protected_unit_names.union(
                        _graph_name_set(spec["units"])
                    ).union(
                        _terminal_material_stream_names(
                            spec["units"],
                            spec["connections"],
                        )
                    ),
                )
                candidate_draft = create_graph_draft(
                    spec["units"],
                    spec["connections"],
                    inlets,
                )
                candidate_case = _apply_studio_graph_draft(
                    spec,
                    candidate_draft,
                )
                _validate_case_graph(
                    candidate_case,
                    candidate_case["process"],
                )
            except ValueError as edit_error:
                st.error(f"Feed creation failed: {edit_error}")
            else:
                _record_graph_revision(
                    spec,
                    candidate_draft,
                    (
                        f"Added independent feed '{new_feed_name.strip()}' "
                        f"with id '{new_inlet_id}'. Edit its conditions, then "
                        "connect its available outlet port."
                    ),
                )
                st.rerun()

        secondary_inlet_map = _secondary_inlet_map(
            spec["inlets"],
            PRIMARY_INLET_ID,
        )
        if secondary_inlet_map:
            st.divider()
            st.markdown("#### Manage secondary inlets")
            st.caption(
                "Each inlet reuses the shared EOS and component registry while "
                "retaining independent conditions and molar composition. The "
                "primary inlet remains available in the fluid basis above."
            )
            selected_inlet_id = st.selectbox(
                "Material inlet",
                options=list(secondary_inlet_map),
                format_func=lambda value: (
                    f"{_graph_object_name(secondary_inlet_map[value], value)}"
                    f" · {value}"
                ),
                key=(
                    "flowsheet_secondary_inlet_"
                    f"{graph_widget_revision}"
                ),
            )
            selected_inlet = secondary_inlet_map[selected_inlet_id]
            selected_inlet_name = _graph_object_name(
                selected_inlet,
                selected_inlet_id,
            )
            st.markdown("##### Operating conditions")
            try:
                condition_rows = inlet_condition_property_rows(
                    selected_inlet
                )
            except ValueError as condition_metadata_error:
                condition_rows = []
                st.warning(
                    "Stored inlet conditions are outside the current editor "
                    "limits and have been preserved. The case remains runnable; "
                    f"edit its composition here or revise the conditions in "
                    f"case JSON. Details: {condition_metadata_error}"
                )
            condition_updates: dict[str, float] = {}
            condition_columns = (
                st.columns(len(condition_rows)) if condition_rows else []
            )
            for column, row in zip(condition_columns, condition_rows):
                condition_updates[row["key"]] = column.number_input(
                    f"{row['label']} [{row['unit']}]",
                    min_value=float(row["minimum"]),
                    max_value=float(row["maximum"]),
                    value=float(row["value"]),
                    step=float(row["step"]),
                    format=row["format"],
                    key=(
                        "flowsheet_inlet_condition_"
                        f"{selected_inlet_id}_{row['key']}_"
                        f"{graph_widget_revision}"
                    ),
                )

            st.markdown("##### Molar composition")
            st.caption(
                "Component identities belong to the shared fluid package. "
                "Entered fractions are normalized when saved."
            )
            composition_rows = inlet_composition_property_rows(
                selected_inlet
            )
            composition_table = st.data_editor(
                pd.DataFrame(composition_rows)[
                    ["component", "mole_fraction"]
                ],
                num_rows="fixed",
                use_container_width=True,
                hide_index=True,
                disabled=["component"],
                column_config={
                    "component": st.column_config.TextColumn(
                        "Shared component",
                    ),
                    "mole_fraction": st.column_config.NumberColumn(
                        "Mole fraction [mol/mol]",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.6f",
                    ),
                },
                key=(
                    "flowsheet_inlet_composition_"
                    f"{selected_inlet_id}_{graph_widget_revision}"
                ),
            )
            entered_total = pd.to_numeric(
                composition_table["mole_fraction"],
                errors="coerce",
            ).sum()
            st.caption(f"Entered mole-fraction sum: {entered_total:.6f}")
            save_inlet_properties = st.button(
                "Save inlet properties",
                use_container_width=True,
                key=(
                    "flowsheet_save_inlet_properties_"
                    f"{selected_inlet_id}_{graph_widget_revision}"
                ),
            )
            if save_inlet_properties:
                try:
                    composition_updates = {
                        str(row["component"]): row["mole_fraction"]
                        for row in composition_table.to_dict("records")
                    }
                    inlets = spec["inlets"]
                    if condition_rows:
                        inlets = update_inlet_conditions(
                            inlets,
                            selected_inlet_id,
                            condition_updates,
                        )
                    inlets = update_inlet_composition(
                        inlets,
                        selected_inlet_id,
                        composition_updates,
                    )
                    candidate_draft = create_graph_draft(
                        spec["units"],
                        spec["connections"],
                        inlets,
                    )
                    candidate_case = _apply_studio_graph_draft(
                        spec,
                        candidate_draft,
                    )
                    _validate_case_graph(
                        candidate_case,
                        candidate_case["process"],
                    )
                except (KeyError, TypeError, ValueError) as edit_error:
                    st.error(f"Inlet property update failed: {edit_error}")
                else:
                    _record_graph_revision(
                        spec,
                        candidate_draft,
                        (
                            f"Updated conditions and composition for "
                            f"'{selected_inlet_name}'. Run NeqSim to solve "
                            "the revised graph."
                        ),
                    )
                    st.rerun()

            st.markdown("##### Feed lifecycle")
            lifecycle_cols = st.columns(2)
            renamed_inlet = lifecycle_cols[0].text_input(
                "Rename selected feed",
                value=selected_inlet_name,
                max_chars=80,
                key=(
                    "flowsheet_rename_inlet_"
                    f"{selected_inlet_id}_{graph_widget_revision}"
                ),
            )
            rename_inlet = lifecycle_cols[0].button(
                "Rename feed",
                use_container_width=True,
                key=(
                    "flowsheet_rename_inlet_button_"
                    f"{selected_inlet_id}_{graph_widget_revision}"
                ),
            )
            confirm_feed_removal = lifecycle_cols[1].checkbox(
                "Confirm unconnected feed removal",
                key=(
                    "flowsheet_confirm_remove_inlet_"
                    f"{selected_inlet_id}_{graph_widget_revision}"
                ),
            )
            remove_inlet = lifecycle_cols[1].button(
                "Remove feed",
                disabled=not confirm_feed_removal,
                use_container_width=True,
                key=(
                    "flowsheet_remove_inlet_"
                    f"{selected_inlet_id}_{graph_widget_revision}"
                ),
            )
            if rename_inlet or remove_inlet:
                try:
                    if rename_inlet:
                        inlets = rename_material_inlet(
                            spec["inlets"],
                            selected_inlet_id,
                            renamed_inlet,
                            protected_unit_names.union(
                                _graph_name_set(spec["units"])
                            ).union(
                                _terminal_material_stream_names(
                                    spec["units"],
                                    spec["connections"],
                                )
                            ),
                        )
                        lifecycle_notice = (
                            f"Renamed feed '{selected_inlet_name}' to "
                            f"'{renamed_inlet.strip()}'."
                        )
                    else:
                        inlets = remove_material_inlet(
                            spec["inlets"],
                            spec["connections"],
                            selected_inlet_id,
                            {PRIMARY_INLET_ID},
                        )
                        lifecycle_notice = (
                            f"Removed unconnected feed "
                            f"'{selected_inlet_name}'."
                        )
                    candidate_draft = create_graph_draft(
                        spec["units"],
                        spec["connections"],
                        inlets,
                    )
                    candidate_case = _apply_studio_graph_draft(
                        spec,
                        candidate_draft,
                    )
                    _validate_case_graph(
                        candidate_case,
                        candidate_case["process"],
                    )
                except ValueError as edit_error:
                    st.error(f"Feed lifecycle update failed: {edit_error}")
                else:
                    _record_graph_revision(
                        spec,
                        candidate_draft,
                        lifecycle_notice
                        + " Run NeqSim to solve the revised graph.",
                    )
                    st.rerun()

        st.divider()
        st.markdown("#### Extend a material path")
        st.caption(
            "Choose any unconnected feed or equipment outlet, then add and "
            "connect the next native unit in one step. Separator gas and "
            "liquid outlets appear independently."
        )
        extend_source_rows = graph_port_rows(
            spec["inlets"],
            spec["units"],
            spec["connections"],
            "material",
            "source",
            available_only=True,
        )
        extendable_types = [
            unit_type
            for unit_type, definition in catalog.items()
            if len(definition["ports"].get("material_in", [])) == 1
        ]
        extend_type_key = (
            f"flowsheet_extend_type_{graph_widget_revision}"
        )
        if (
            extend_type_key in st.session_state
            and st.session_state[extend_type_key] not in extendable_types
        ):
            st.session_state.pop(extend_type_key)
        extend_cols = st.columns(3)
        if extend_source_rows:
            extend_source_index = extend_cols[0].selectbox(
                "Source outlet",
                options=list(range(len(extend_source_rows))),
                format_func=lambda index: extend_source_rows[index]["label"],
                key=f"flowsheet_extend_source_{graph_widget_revision}",
            )
        else:
            extend_cols[0].info("No unconnected material outlets.")
            extend_source_index = None
        extend_type = extend_cols[1].selectbox(
            "Next equipment",
            options=extendable_types,
            format_func=lambda value: catalog[value]["label"],
            key=extend_type_key,
        )
        extend_name = extend_cols[2].text_input(
            "Equipment name",
            value=f"New {catalog[extend_type]['label']}",
            max_chars=80,
            key=f"flowsheet_extend_name_{graph_widget_revision}",
        )
        extend_path = st.button(
            "Add and connect equipment",
            disabled=extend_source_index is None,
            use_container_width=True,
            key=f"flowsheet_extend_path_{graph_widget_revision}",
        )
        if extend_path:
            try:
                units, connections, new_unit_id, new_connection_id = (
                    extend_material_path(
                        spec["inlets"],
                        spec["units"],
                        spec["connections"],
                        extend_source_rows[extend_source_index]["endpoint"],
                        extend_type,
                        extend_name,
                        protected_unit_ids,
                    )
                )
                candidate_draft = create_graph_draft(
                    units,
                    connections,
                    spec["inlets"],
                )
                candidate_case = _apply_studio_graph_draft(
                    spec,
                    candidate_draft,
                )
                _validate_case_graph(
                    candidate_case,
                    candidate_case["process"],
                )
            except ValueError as edit_error:
                st.error(f"Material path extension failed: {edit_error}")
            else:
                new_unit = next(
                    unit
                    for unit in units
                    if str(unit.get("id", "")).strip() == new_unit_id
                )
                next_output_label = _material_output_selection_label(
                    new_unit
                )
                _record_graph_revision(
                    spec,
                    candidate_draft,
                    (
                        f"Added {extend_type} '{extend_name.strip()}' and "
                        f"connected it with '{new_connection_id}'. Select "
                        f"{next_output_label} to extend the path again."
                    ),
                )
                st.rerun()

        st.divider()
        st.markdown("#### Add standalone equipment")
        st.caption(
            "Create an unconnected native equipment node, then route any "
            "available feed or phase outlet into it using the port controls "
            "below. Add mixers here, then connect independent feeds to their "
            "separate inlet ports. This also supports branches such as "
            "separator liquid to pump."
        )
        standalone_cols = st.columns(2)
        standalone_type = standalone_cols[0].selectbox(
            "Standalone equipment type",
            options=list(catalog),
            format_func=lambda value: (
                f"{catalog[value]['label']} · "
                f"{catalog[value]['category']}"
            ),
            key=f"flowsheet_standalone_type_{graph_widget_revision}",
        )
        standalone_name = standalone_cols[1].text_input(
            "Standalone equipment name",
            value=f"New {catalog[standalone_type]['label']}",
            max_chars=80,
            key=f"flowsheet_standalone_name_{graph_widget_revision}",
        )
        add_standalone = st.button(
            "Add equipment node",
            use_container_width=True,
            key=f"flowsheet_add_standalone_{graph_widget_revision}",
        )
        if add_standalone:
            try:
                reserved_ids = {
                    *protected_unit_ids,
                    *(
                        str(inlet.get("id", "")).strip()
                        for inlet in spec["inlets"]
                        if isinstance(inlet, dict)
                        and str(inlet.get("id", "")).strip()
                    ),
                }
                reserved_names = _graph_name_set(spec["inlets"])
                units, new_unit_id = add_catalog_unit(
                    spec["units"],
                    standalone_type,
                    standalone_name,
                    reserved_ids,
                    reserved_names,
                )
                candidate_draft = create_graph_draft(
                    units,
                    spec["connections"],
                    spec["inlets"],
                )
                candidate_case = _apply_studio_graph_draft(
                    spec,
                    candidate_draft,
                )
                _validate_case_graph(
                    candidate_case,
                    candidate_case["process"],
                )
            except ValueError as edit_error:
                st.error(f"Equipment creation failed: {edit_error}")
            else:
                _record_graph_revision(
                    spec,
                    candidate_draft,
                    (
                        f"Added standalone {standalone_type} "
                        f"'{standalone_name.strip()}' with id '{new_unit_id}'. "
                        "Connect an available source port to its inlet before "
                        "running NeqSim."
                    ),
                )
                st.rerun()

        st.divider()
        st.markdown("#### Connect and disconnect ports")
        st.caption(
            "Create explicit material or energy paths between declared ports. "
            "One connection is allowed per input or output port; splitters and "
            "mixers expose distinct branch ports."
        )
        connection_type = st.selectbox(
            "Connection type",
            options=["material", "energy"],
            format_func=str.title,
            key="flowsheet_connection_type",
        )
        available_source_rows = graph_port_rows(
            spec["inlets"],
            spec["units"],
            spec["connections"],
            connection_type,
            "source",
            available_only=True,
        )
        available_target_rows = graph_port_rows(
            spec["inlets"],
            spec["units"],
            spec["connections"],
            connection_type,
            "target",
            available_only=True,
        )
        port_cols = st.columns(2)
        source_index = port_cols[0].selectbox(
            "Available source port",
            options=list(range(len(available_source_rows))),
            format_func=lambda index: available_source_rows[index]["label"],
            key=f"flowsheet_source_port_{graph_widget_revision}",
        )
        target_index = port_cols[1].selectbox(
            "Available target port",
            options=list(range(len(available_target_rows))),
            format_func=lambda index: available_target_rows[index]["label"],
            key=f"flowsheet_target_port_{graph_widget_revision}",
        )
        connect_ports = st.button(
            "Connect selected ports",
            disabled=(
                source_index is None
                or target_index is None
            ),
            use_container_width=True,
            key=f"flowsheet_connect_ports_{graph_widget_revision}",
        )
        if connect_ports:
            try:
                connections, new_connection_id = connect_graph_ports(
                    spec["inlets"],
                    spec["units"],
                    spec["connections"],
                    connection_type,
                    available_source_rows[source_index]["endpoint"],
                    available_target_rows[target_index]["endpoint"],
                )
                candidate_draft = create_graph_draft(
                    spec["units"],
                    connections,
                    spec["inlets"],
                )
                candidate_case = _apply_studio_graph_draft(
                    spec,
                    candidate_draft,
                )
                _validate_case_graph(
                    candidate_case,
                    candidate_case["process"],
                )
            except ValueError as edit_error:
                st.error(f"Port connection failed: {edit_error}")
            else:
                _record_graph_revision(
                    spec,
                    candidate_draft,
                    (
                        f"Created {connection_type} connection "
                        f"'{new_connection_id}'. Run NeqSim to solve "
                        "the updated graph."
                    ),
                )
                st.rerun()

        if all_connection_rows:
            disconnect_id = st.selectbox(
                "Existing connection",
                options=list(all_connection_labels),
                format_func=all_connection_labels.__getitem__,
                key=(
                    "flowsheet_disconnect_connection_"
                    f"{graph_widget_revision}"
                ),
            )
            confirm_disconnect = st.checkbox(
                "Confirm disconnection",
                key=(
                    "flowsheet_confirm_disconnect_"
                    f"{graph_widget_revision}"
                ),
                help=(
                    "The selected path is removed from the unsolved draft. "
                    "Undo restores it."
                ),
            )
            disconnect_ports = st.button(
                "Disconnect selected path",
                disabled=not confirm_disconnect,
                use_container_width=True,
                key=(
                    "flowsheet_disconnect_ports_"
                    f"{graph_widget_revision}"
                ),
            )
            if disconnect_ports:
                try:
                    connections = disconnect_graph_connection(
                        spec["inlets"],
                        spec["units"],
                        spec["connections"],
                        disconnect_id,
                    )
                    candidate_draft = create_graph_draft(
                        spec["units"],
                        connections,
                        spec["inlets"],
                    )
                    candidate_case = _apply_studio_graph_draft(
                        spec,
                        candidate_draft,
                    )
                    _validate_case_graph(
                        candidate_case,
                        candidate_case["process"],
                    )
                except ValueError as edit_error:
                    st.error(f"Port disconnection failed: {edit_error}")
                else:
                    _record_graph_revision(
                        spec,
                        candidate_draft,
                        (
                            f"Disconnected '{disconnect_id}'. "
                            "Use undo to restore the path or reconnect "
                            "available ports before solving."
                        ),
                    )
                    st.rerun()

        st.divider()
        st.markdown("#### Add equipment from palette")
        st.caption(
            "Insert a native NeqSim unit into one existing material path. "
            "The draft is validated before it replaces the active graph."
        )
        st.dataframe(
            pd.DataFrame(catalog_rows),
            use_container_width=True,
            hide_index=True,
        )
        inline_insert_types = [
            unit_type
            for unit_type, definition in catalog.items()
            if definition["ports"].get("material_in") == ["in"]
            and definition["ports"].get("material_out") == ["out"]
        ]
        palette_unit_type_key = "flowsheet_palette_unit_type"
        if (
            palette_unit_type_key in st.session_state
            and st.session_state[palette_unit_type_key]
            not in inline_insert_types
        ):
            st.session_state.pop(palette_unit_type_key)
        unit_type = st.selectbox(
            "Equipment type",
            options=inline_insert_types,
            format_func=lambda value: (
                f"{catalog[value]['label']} · "
                f"{catalog[value]['category']}"
            ),
            key=palette_unit_type_key,
        )
        selected_definition = catalog[unit_type]
        unit_name = st.text_input(
            "Equipment name",
            value="",
            placeholder=f"New {selected_definition['label']}",
            key="flowsheet_palette_unit_name",
        )
        connection_id = st.selectbox(
            "Material path",
            options=list(connection_labels),
            format_func=connection_labels.__getitem__,
            key=(
                "flowsheet_palette_connection_"
                f"{graph_widget_revision}"
            ),
        )
        st.caption("Initial properties and engineering units")
        default_property_rows = inline_unit_property_rows(unit_type)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Property": row["label"],
                        "Value": row["value"],
                        "Unit": row["unit"],
                    }
                    for row in default_property_rows
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

        insert_unit = st.button(
            "Insert equipment in selected path",
            disabled=not connection_rows,
            use_container_width=True,
            key="flowsheet_insert_palette_unit",
        )
        if insert_unit:
            try:
                requested_name = (
                    unit_name.strip()
                    or f"New {selected_definition['label']}"
                )
                existing_names = _graph_name_set(
                    spec["units"],
                    casefold=True,
                )
                existing_names.update(
                    _graph_name_set(spec["inlets"], casefold=True)
                )
                resolved_name = requested_name
                name_suffix = 2
                while resolved_name.casefold() in existing_names:
                    resolved_name = f"{requested_name} {name_suffix}"
                    name_suffix += 1
                units, connections, new_unit_id = (
                    insert_inline_unit_on_connection(
                        spec["units"],
                        spec["connections"],
                        connection_id,
                        unit_type,
                        resolved_name,
                        {
                            *protected_unit_ids,
                            *(
                                str(inlet.get("id", "")).strip()
                                for inlet in spec["inlets"]
                                if isinstance(inlet, dict)
                                and str(inlet.get("id", "")).strip()
                            ),
                        },
                    )
                )
                candidate_draft = create_graph_draft(
                    units,
                    connections,
                    spec["inlets"],
                )
                candidate_case = _apply_studio_graph_draft(
                    spec,
                    candidate_draft,
                )
                _validate_case_graph(
                    candidate_case,
                    candidate_case["process"],
                )
            except ValueError as edit_error:
                st.error(f"Equipment insertion failed: {edit_error}")
            else:
                _record_graph_revision(
                    spec,
                    candidate_draft,
                    (
                        f"Added '{resolved_name}' ({new_unit_id}). "
                        "Review its initial properties and run NeqSim."
                    ),
                )
                st.rerun()

        if added_unit_map:
            st.divider()
            st.markdown("#### Manage added equipment")
            selected_unit_id = st.selectbox(
                "Added equipment",
                options=list(added_unit_map),
                format_func=lambda value: (
                    f"{added_unit_map[value]['name']} · "
                    f"{added_unit_map[value]['type']} · {value}"
                ),
                key="flowsheet_added_unit",
            )
            selected_unit = added_unit_map[selected_unit_id]
            renamed_unit_name = st.text_input(
                "Equipment display name",
                value=str(selected_unit["name"]),
                key=f"flowsheet_added_unit_name_{selected_unit_id}",
                help=(
                    "The stable graph ID and all port connections remain "
                    "unchanged."
                ),
            )
            st.markdown("##### Operating properties")
            st.caption(
                "Values use the executable graph schema and explicit "
                "engineering units."
            )
            property_rows = inline_unit_property_rows(
                selected_unit["type"],
                selected_unit["params"],
            )
            property_updates: dict[str, float] = {}
            for row in property_rows:
                property_updates[row["key"]] = st.number_input(
                    f"{row['label']} [{row['unit']}]",
                    min_value=float(row["minimum"]),
                    max_value=float(row["maximum"]),
                    value=float(row["value"]),
                    step=float(row["step"]),
                    format=row["format"],
                    key=(
                        "flowsheet_added_unit_property_"
                        f"{selected_unit_id}_{row['key']}_"
                        f"{graph_widget_revision}"
                    ),
                )
            properties_changed = any(
                property_updates[row["key"]] != row["value"]
                for row in property_rows
            )
            save_properties = st.button(
                "Save equipment properties",
                disabled=not properties_changed,
                use_container_width=True,
                key=(
                    "flowsheet_save_unit_properties_"
                    f"{selected_unit_id}_{graph_widget_revision}"
                ),
            )
            lifecycle_cols = st.columns(2)
            rename_unit = lifecycle_cols[0].button(
                "Rename equipment",
                use_container_width=True,
                key=f"flowsheet_rename_unit_{selected_unit_id}",
            )
            confirm_removal = lifecycle_cols[1].checkbox(
                "Confirm removal",
                key=f"flowsheet_confirm_remove_{selected_unit_id}",
                help=(
                    "Removal deletes an unconnected node, releases the source "
                    "feeding a terminal node, or reconnects one simple incoming "
                    "and outgoing material path. Branches and energy links are "
                    "protected."
                ),
            )
            remove_unit = lifecycle_cols[1].button(
                "Remove equipment",
                disabled=not confirm_removal,
                use_container_width=True,
                key=f"flowsheet_remove_unit_{selected_unit_id}",
            )

            if save_properties:
                try:
                    units = update_inline_unit_properties(
                        spec["units"],
                        selected_unit_id,
                        property_updates,
                    )
                    candidate_draft = create_graph_draft(
                        units,
                        spec["connections"],
                        spec["inlets"],
                    )
                    candidate_case = _apply_studio_graph_draft(
                        spec,
                        candidate_draft,
                    )
                    _validate_case_graph(
                        candidate_case,
                        candidate_case["process"],
                    )
                except ValueError as edit_error:
                    st.error(f"Equipment property update failed: {edit_error}")
                else:
                    _record_graph_revision(
                        spec,
                        candidate_draft,
                        (
                            f"Updated operating properties for "
                            f"'{selected_unit['name']}'. Run NeqSim to solve "
                            "the revised graph."
                        ),
                    )
                    st.rerun()

            if rename_unit:
                try:
                    units = rename_inline_unit(
                        spec["units"],
                        selected_unit_id,
                        renamed_unit_name,
                        _graph_name_set(spec["inlets"]),
                    )
                    candidate_draft = create_graph_draft(
                        units,
                        spec["connections"],
                        spec["inlets"],
                    )
                    candidate_case = _apply_studio_graph_draft(
                        spec,
                        candidate_draft,
                    )
                    _validate_case_graph(
                        candidate_case,
                        candidate_case["process"],
                    )
                except ValueError as edit_error:
                    st.error(f"Equipment rename failed: {edit_error}")
                else:
                    _record_graph_revision(
                        spec,
                        candidate_draft,
                        (
                            f"Renamed '{selected_unit['name']}' to "
                            f"'{renamed_unit_name.strip()}'. "
                            "The stable graph ID and connections were retained."
                        ),
                    )
                    st.rerun()

            if remove_unit:
                try:
                    units, connections = remove_inline_unit(
                        spec["units"],
                        spec["connections"],
                        selected_unit_id,
                    )
                    candidate_draft = create_graph_draft(
                        units,
                        connections,
                        spec["inlets"],
                    )
                    candidate_case = _apply_studio_graph_draft(
                        spec,
                        candidate_draft,
                    )
                    _validate_case_graph(
                        candidate_case,
                        candidate_case["process"],
                    )
                except ValueError as edit_error:
                    st.error(f"Equipment removal failed: {edit_error}")
                else:
                    _record_graph_revision(
                        spec,
                        candidate_draft,
                        (
                            f"Removed '{selected_unit['name']}' and updated "
                            "its explicit material routes. Run NeqSim to solve "
                            "the updated graph."
                        ),
                    )
                    st.rerun()

        if st.session_state.get(GRAPH_DRAFT_STATE_KEY) is not None:
            st.divider()
            st.caption(
                "This unsolved graph draft persists in the current session "
                "and is included in the case JSON."
            )
            action_cols = st.columns(2)
            action_cols[0].download_button(
                "Download unsolved case JSON",
                data=json.dumps(spec, indent=2),
                file_name="process_flowsheet_unsolved_case.json",
                mime="application/json",
                use_container_width=True,
            )
            reset_draft = action_cols[1].button(
                "Restore starter graph",
                use_container_width=True,
                key="flowsheet_restore_starter_graph",
            )
            if reset_draft:
                starter_units, starter_connections = _build_template_graph(
                    spec["process"]
                )
                starter_draft = create_graph_draft(
                    starter_units,
                    starter_connections,
                    spec["inlets"],
                )
                _record_graph_revision(
                    spec,
                    starter_draft,
                    (
                        "Restored the validated starter graph. "
                        "Fluid and equipment-property inputs were retained."
                    ),
                )
                st.rerun()


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
    with st.expander("Start a new case", expanded=False):
        st.caption(
            "Reset the editable case, solver state, retained comparisons, and "
            "the Studio-owned Process Chat model."
        )
        confirm_new_case = st.checkbox(
            "I understand that unsaved case changes and comparisons will be cleared.",
            key="flowsheet_confirm_new_case",
        )
        st.button(
            "Start new case",
            disabled=not confirm_new_case,
            on_click=_start_new_case,
            use_container_width=True,
            key="flowsheet_start_new_case",
        )

with st.expander("Model scope and assumptions", expanded=False):
    st.markdown(
        """
- Steady-state thermodynamic and process simulation in NeqSim.
- Pressure inputs are absolute (`bara`); temperature inputs are degrees Celsius.
- Composition is molar and is normalized before calculation.
- Schema v3 separates shared fluid/inlet data and records an explicit process graph.
- Unit nodes expose material ports; connections identify source and target ports.
- Material and energy ports can be connected or disconnected in the draft editor.
- Graph validation enforces declared ports, direction, and single port occupancy.
- A deterministic execution plan orders acyclic multi-inlet graphs and
  dependencies.
- Each inlet compiles to an independent ProcessBuilder fluid definition with
  explicit units.
- Inlet temperature, absolute pressure, and mass flow use shared property metadata.
- Inline equipment edits persist as an unsolved graph draft and are included in
  JSON cases.
- Added equipment properties are metadata-driven, bounded, and stored with
  explicit units.
- Starter-template equipment uses the same property metadata; separators expose
  their native equilibrium behavior without invented set points.
- The active draft graph is automatically laid out before NeqSim execution.
- Cyclic graphs remain blocked until recycle and tear-stream solving is available.
- Fluid validation supports multiple compatible inlets with independent conditions.
- Pseudo-component names cannot carry conflicting molar mass or density data.
- The starter graph still projects one inlet into the Process Chat builder.
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

case_notice = st.session_state.pop(CASE_NOTICE_STATE_KEY, None)
if case_notice:
    st.success(case_notice)

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
    feed_condition_rows = inlet_condition_property_rows(
        {
            "id": PRIMARY_INLET_ID,
            "temperature_C": st.session_state[
                "flowsheet_feed_temperature_c"
            ],
            "pressure_bara": st.session_state[
                "flowsheet_feed_pressure_bara"
            ],
            "total_flow": st.session_state["flowsheet_feed_flow_kg_hr"],
            "flow_unit": "kg/hr",
        }
    )
    feed_condition_state_keys = {
        "temperature_C": "flowsheet_feed_temperature_c",
        "pressure_bara": "flowsheet_feed_pressure_bara",
        "total_flow": "flowsheet_feed_flow_kg_hr",
    }
    feed_condition_values: dict[str, float] = {}
    for row in feed_condition_rows:
        feed_condition_values[row["key"]] = st.number_input(
            f"Feed {row['label'].casefold()} [{row['unit']}]",
            min_value=float(row["minimum"]),
            max_value=float(row["maximum"]),
            step=float(row["step"]),
            format=row["format"],
            key=feed_condition_state_keys[row["key"]],
        )
    feed_temperature_c = feed_condition_values["temperature_C"]
    feed_pressure_bara = feed_condition_values["pressure_bara"]
    feed_flow_kg_hr = feed_condition_values["total_flow"]
    st.caption(
        "Inlet conditions are independent of the shared EOS, component "
        "registry, and characterization."
    )

with object_col:
    selected_object = _render_object_property_editor()

stage_1_pressure_bara = float(
    st.session_state["flowsheet_stage_1_pressure_bara"]
)
stage_2_pressure_bara = float(
    st.session_state["flowsheet_stage_2_pressure_bara"]
)
stage_1_isentropic_efficiency = float(
    st.session_state["flowsheet_stage_1_isentropic_efficiency"]
)
stage_2_isentropic_efficiency = float(
    st.session_state["flowsheet_stage_2_isentropic_efficiency"]
)
intercooler_temperature_c = float(
    st.session_state["flowsheet_intercooler_temperature_c"]
)
intercooler_pressure_drop_bar = float(
    st.session_state["flowsheet_intercooler_pressure_drop_bar"]
)
export_temperature_c = float(
    st.session_state["flowsheet_export_temperature_c"]
)
export_pressure_drop_bar = float(
    st.session_state["flowsheet_export_pressure_drop_bar"]
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
            intercooler_pressure_drop_bar=intercooler_pressure_drop_bar,
            export_temperature_c=export_temperature_c,
            export_pressure_drop_bar=export_pressure_drop_bar,
            stage_1_isentropic_efficiency=stage_1_isentropic_efficiency,
            stage_2_isentropic_efficiency=stage_2_isentropic_efficiency,
        )
        active_graph_draft = st.session_state.get(GRAPH_DRAFT_STATE_KEY)
        if active_graph_draft is not None:
            draft_case_spec = _apply_studio_graph_draft(
                draft_case_spec,
                active_graph_draft,
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

if draft_case_spec is not None:
    _render_graph_palette(draft_case_spec)
    with st.expander("Graph execution plan", expanded=False):
        st.caption(
            "Derived from inlet, unit, port, and connection definitions. "
            "Steps are deterministic; cyclic graphs require the later recycle solver."
        )
        st.dataframe(
            pd.DataFrame(_build_execution_plan(draft_case_spec)),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("#### Independent inlet fluid definitions")
        st.caption(
            "All inlets reuse validated characterization while retaining their "
            "own molar composition, temperature, absolute pressure, and mass flow."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Inlet ID": inlet_spec["inlet_id"],
                        "Fluid package": inlet_spec["fluid_package_id"],
                        "EOS": inlet_spec["fluid_spec"]["eos_model"].upper(),
                        "Mixing rule": inlet_spec["fluid_spec"]["mixing_rule"],
                        "Components": len(
                            inlet_spec["fluid_spec"]["components"]
                        ),
                        "Temperature [°C]": inlet_spec["fluid_spec"][
                            "temperature_C"
                        ],
                        "Pressure [bara]": inlet_spec["fluid_spec"][
                            "pressure_bara"
                        ],
                        "Mass flow [kg/hr]": inlet_spec["fluid_spec"][
                            "total_flow"
                        ],
                    }
                    for inlet_spec in _build_inlet_fluid_specs(
                        draft_case_spec
                    )
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

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
        _validate_graph_solve_readiness(case_spec)

        solver_status_placeholder.write("**Solver:** Solving")
        execution_started = perf_counter()
        with st.spinner("Building and solving the NeqSim process..."):
            builder = ProcessBuilder()
            graph_spec, inlet_specs, execution_order = (
                _build_graph_solver_inputs(case_spec)
            )
            model = builder.build_acyclic_graph(
                graph_spec,
                inlet_specs,
                execution_order,
            )
            result = model.run()
            model_bytes = builder.save_neqsim_bytes()
        execution_seconds = perf_counter() - execution_started
        run_record = _solver_run_record(
            result,
            model,
            current_case_signature,
            execution_seconds,
        )

        state = {
            "spec": case_spec,
            "warnings": case_warnings,
            "builder": builder,
            "model": model,
            "result": result,
            "model_bytes": model_bytes,
            "signature": current_case_signature,
            "run_record": run_record,
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
        st.session_state["process_model_name"] = STUDIO_PROCESS_MODEL_NAME
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
    run_record = state.get("run_record", {})

    st.divider()
    st.subheader("2. Engineering results")

    for warning in state["warnings"]:
        st.warning(warning)

    total_power_kw = _kpi_value(result, "total_power_kW")
    total_duty_kw = _kpi_value(result, "total_duty_kW")
    mass_balance_pct = _kpi_value(result, "mass_balance_pct")
    feed_flow_kg_hr = solved_feed_flow_kg_hr(
        result,
        float(spec["fluid"]["total_flow"]),
    )
    feed_tonnes_per_hour = feed_flow_kg_hr / 1000.0
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
        baseline_state_key = CASE_HISTORY_BASELINE_STATE_KEY
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
            "Stage 1 efficiency [-]": "{:.3f}",
            "Stage 2 efficiency [-]": "{:.3f}",
            "Intercooler pressure drop [bar]": "{:.3f}",
            "Export cooler pressure drop [bar]": "{:.3f}",
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

    stream_table = _stream_dataframe(model)
    equipment_table = _equipment_dataframe(model)
    selected_equipment_table, selected_stream_table = (
        _selected_object_result_tables(
            selected_object,
            stream_table,
            equipment_table,
        )
    )

    pressure_profile_table = _pressure_profile_dataframe(spec, equipment_table)

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

        selected_display_name = TEMPLATE_OBJECTS[selected_object][0]
        st.markdown(f"#### {selected_display_name} · solved results")
        if selected_equipment_table.empty and selected_stream_table.empty:
            st.info("The model adapter returned no solved rows for this object.")
        if not selected_equipment_table.empty:
            st.caption("Equipment performance")
            st.dataframe(
                selected_equipment_table,
                use_container_width=True,
                hide_index=True,
            )
        if not selected_stream_table.empty:
            st.caption("Outlet stream conditions")
            st.dataframe(
                selected_stream_table.style.format(
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

        with st.expander("Build log", expanded=False):
            st.code("\n".join(builder.build_log))

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
    material_boundary_table = _material_boundary_dataframe(result)
    component_balance_table = _component_balance_dataframe(result)
    energy_balance_table = _energy_balance_dataframe(result)
    energy_transfer_table = _energy_transfer_dataframe(result)
    convergence_table = _convergence_dataframe(result)
    convergence_summary = aggregate_convergence(result)
    unit_balance_table = _unit_balance_dataframe(result)
    unit_balance_summary = aggregate_unit_balances(result)
    with validation_tab:
        status_counts = constraint_table["status"].value_counts()
        profile_counts = pressure_profile_table["Status"].value_counts()
        violation_count = status_counts.get("VIOLATION", 0) + profile_counts.get(
            "VIOLATION",
            0,
        )
        warning_count = status_counts.get("WARN", 0) + profile_counts.get(
            "WARN",
            0,
        )
        unknown_count = status_counts.get("UNKNOWN", 0) + profile_counts.get(
            "UNKNOWN",
            0,
        )
        if violation_count > 0:
            st.error("One or more engineering validation checks reported a violation.")
        elif warning_count > 0:
            st.warning("The calculation completed with engineering warnings.")
        elif unknown_count > 0:
            st.warning(
                "The calculation completed, but one or more engineering "
                "validation checks are unavailable or not applicable."
            )
        else:
            st.success("All reported engineering validation checks passed.")
        st.dataframe(
            constraint_table,
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("#### Iterative convergence")
        if convergence_summary["applicable"] is None:
            st.info(
                "This legacy result did not record native recycle or "
                "adjuster convergence diagnostics."
            )
        elif convergence_summary["applicable"] is False:
            st.success(
                "Feed-forward solve: no recycle or adjuster convergence "
                "loops are present."
            )
        else:
            if convergence_summary["converged"]:
                st.success(
                    "Every native recycle and adjuster reports convergence."
                )
            else:
                st.error(
                    "One or more native recycle or adjuster units did not "
                    "converge."
                )
            st.dataframe(
                convergence_table.style.format(
                    {
                        "Flow error": "{:.6g}",
                        "Flow tolerance": "{:.6g}",
                        "Temperature error": "{:.6g}",
                        "Temperature tolerance": "{:.6g}",
                        "Pressure error": "{:.6g}",
                        "Pressure tolerance": "{:.6g}",
                        "Composition error": "{:.6g}",
                        "Composition tolerance": "{:.6g}",
                        "Target error": "{:.6g}",
                        "Target tolerance": "{:.6g}",
                    },
                    na_rep="—",
                ),
                use_container_width=True,
                hide_index=True,
            )
            suggestions = convergence_summary["suggestions"]
            if suggestions:
                st.caption("Native solver guidance")
                for suggestion in suggestions:
                    st.caption(f"• {suggestion}")
        st.markdown("#### Per-unit material and energy closure")
        if unit_balance_summary["applicable"] is None:
            st.info(
                "This legacy result did not record explicit-port "
                "per-unit closure diagnostics."
            )
        elif unit_balance_summary["applicable"] is False:
            if unit_balance_summary["coverage_complete"]:
                st.info(
                    "No supported unit operation exposed both inlet and "
                    "outlet material ports for a closure audit."
                )
            else:
                st.warning(
                    "Per-unit closure evidence is unavailable because "
                    "every candidate unit was excluded from the material "
                    "balance audit."
                )
                if unit_balance_summary["excluded_units"]:
                    st.caption(
                        "Unaudited equipment: "
                        + ", ".join(
                            unit_balance_summary["excluded_units"]
                        )
                        + "."
                    )
        else:
            if unit_balance_summary["coverage_complete"]:
                st.success(
                    "Every supported explicit-port unit was included "
                    "in the material-balance audit."
                )
            else:
                st.warning(
                    "Per-unit material-balance coverage is incomplete; "
                    "excluded equipment is listed below."
                )
            if unit_balance_summary["energy_unit_count"] == 0:
                st.warning("No per-unit energy balance was audited.")
            elif not unit_balance_summary["energy_coverage_complete"]:
                st.warning(
                    "Per-unit energy balance was audited for "
                    f"{int(unit_balance_summary['energy_unit_count'])} "
                    "of "
                    f"{int(unit_balance_summary['unit_count'])} "
                    "included units."
                )
            closure_cols = st.columns(2)
            closure_cols[0].metric(
                "Maximum unit mass imbalance",
                _format_metric(
                    unit_balance_summary["max_mass_imbalance_pct"],
                    "%",
                    6,
                ),
                help=_unit_identity_label(
                    unit_balance_summary["max_mass_imbalance_unit"]
                ),
            )
            closure_cols[1].metric(
                "Maximum unit energy imbalance",
                _format_metric(
                    unit_balance_summary["max_energy_imbalance_pct"],
                    "%",
                    6,
                ),
                help=_unit_identity_label(
                    unit_balance_summary["max_energy_imbalance_unit"]
                ),
            )
            st.dataframe(
                unit_balance_table.style.format(
                    {
                        "Inlet mass flow [kg/hr]": "{:,.6f}",
                        "Outlet mass flow [kg/hr]": "{:,.6f}",
                        "Mass residual [kg/hr]": "{:+.6e}",
                        "Mass imbalance [%]": "{:.6g}",
                        "Inlet enthalpy flow [kW]": "{:+,.6f}",
                        "Outlet enthalpy flow [kW]": "{:+,.6f}",
                        "External energy transfer [kW]": "{:+,.6f}",
                        "Energy residual [kW]": "{:+.6e}",
                        "Energy imbalance [%]": "{:.6g}",
                    },
                    na_rep="—",
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Mass residual is outlet mass flow minus inlet mass flow. "
                "Energy residual is outlet enthalpy flow minus inlet "
                "enthalpy flow minus signed external energy transfer. "
                "Positive external transfer adds energy to the material "
                "system."
            )
            if unit_balance_summary["excluded_units"]:
                st.caption(
                    "Unaudited equipment: "
                    + ", ".join(unit_balance_summary["excluded_units"])
                    + "."
                )
        st.markdown("#### Solved material boundaries")
        if material_boundary_table.empty:
            st.info(
                "This legacy result did not expose explicit material "
                "boundary diagnostics."
            )
        else:
            st.dataframe(
                material_boundary_table.style.format(
                    {
                        "Mass flow [kg/hr]": "{:,.3f}",
                        "Temperature [°C]": "{:.3f}",
                        "Pressure [bara]": "{:.4f}",
                        "Molar flow [mol/s]": "{:,.6f}",
                        "Enthalpy flow [kW]": "{:+,.6f}",
                    },
                    na_rep="—",
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Feed and product rows are native solved streams. "
                "Mass and enthalpy flow are aggregated across every "
                "listed boundary."
            )
        st.markdown("#### Component balance")
        if component_balance_table.empty:
            st.info(
                "Component-level boundary diagnostics are unavailable "
                "for this result."
            )
        else:
            st.dataframe(
                component_balance_table.style.format(
                    {
                        "Feed molar flow [mol/s]": "{:,.9g}",
                        "Product molar flow [mol/s]": "{:,.9g}",
                        "Residual [mol/s]": "{:+.6e}",
                        "Imbalance [%]": "{:.6g}",
                    },
                    na_rep="—",
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Component flows are aggregated across every solved "
                "feed and product boundary. Relative imbalance uses a "
                "1e-9 mol/s absolute scale floor for trace components."
            )
        st.markdown("#### System energy balance")
        if energy_balance_table.empty:
            st.info(
                "System energy closure is unavailable for this result. "
                "The validation table identifies unaudited or unreadable "
                "equipment when applicable."
            )
        else:
            st.dataframe(
                energy_balance_table.style.format(
                    {"Value": "{:+,.9g}"},
                    na_rep="—",
                ),
                use_container_width=True,
                hide_index=True,
            )
            if not energy_transfer_table.empty:
                st.caption("Audited external energy transfers")
                st.dataframe(
                    energy_transfer_table.style.format(
                        {"Energy transfer [kW]": "{:+,.6f}"},
                        na_rep="—",
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            st.caption(
                "Closure uses product enthalpy minus feed enthalpy minus "
                "signed external transfer. Positive shaft work or heat "
                "adds energy to the material system; negative values "
                "remove energy."
            )
        st.markdown("#### Solved pressure profile")
        st.dataframe(
            pressure_profile_table.style.format(
                {
                    "Expected outlet [bara]": "{:.3f}",
                    "Calculated outlet [bara]": "{:.3f}",
                    "Deviation [bar]": "{:+.3f}",
                    "Pass tolerance [bar]": "{:.2f}",
                },
                na_rep="—",
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Pressures are absolute. OK requires agreement within 0.05 bar; "
            "a deviation above 0.50 bar is a violation."
        )
        st.markdown("#### Solver run record")
        if run_record:
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Property": key, "Value": value}
                        for key, value in run_record.items()
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Execution wall time includes process construction, native "
                "NeqSim solving, and model serialization."
            )
        else:
            st.info(
                "Run provenance was not recorded for this legacy session result. "
                "Run the case again to create it."
            )
        st.caption(
            "Validation level: per-unit NeqSim recycle and adjuster "
            "convergence, pressure ordering, composition normalization, "
            "material, component and audited energy closure, and "
            "engineering bounds."
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
            pressure_profile_table,
            run_record,
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
