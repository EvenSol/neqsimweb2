"""Pure schema helpers for editing Process Flowsheet Studio graphs."""

from __future__ import annotations

import copy
import heapq
import json
import math
import re
from typing import Any

from .graph_schema import (
    canonical_material_output_port,
    material_connection_name,
)


GRAPH_DRAFT_SCHEMA_VERSION = 1
GRAPH_HISTORY_SCHEMA_VERSION = 1
MAX_GRAPH_HISTORY_ENTRIES = 50
MAX_MULTI_INLET_PORTS = 64
MAX_SPLITTER_OUTLET_PORTS = 64


def _number_property(
    label: str,
    unit: str,
    minimum: float,
    maximum: float,
    step: float,
    display_format: str,
) -> dict[str, Any]:
    """Define one explicit-unit numeric property for editor presentation."""
    return {
        "kind": "number",
        "label": label,
        "unit": unit,
        "minimum": minimum,
        "maximum": maximum,
        "step": step,
        "format": display_format,
    }


def _boolean_property(label: str, description: str) -> dict[str, Any]:
    """Define one boolean property for editor presentation."""
    return {
        "kind": "boolean",
        "label": label,
        "description": description,
    }


def _integer_property(
    label: str,
    unit: str,
    minimum: int,
    maximum: int,
    step: int,
    description: str,
) -> dict[str, Any]:
    """Define one bounded integer property for editor presentation."""
    return {
        "kind": "integer",
        "label": label,
        "unit": unit,
        "minimum": minimum,
        "maximum": maximum,
        "step": step,
        "format": "%d",
        "description": description,
    }


def _choice_property(
    label: str,
    choices: tuple[str, ...],
    description: str,
) -> dict[str, Any]:
    """Define one closed-set string property for editor presentation."""
    return {
        "kind": "choice",
        "label": label,
        "unit": "",
        "choices": choices,
        "description": description,
    }


_COMPRESSOR_CHART_TEMPLATES = (
    "CENTRIFUGAL_STANDARD",
    "CENTRIFUGAL_HIGH_FLOW",
    "CENTRIFUGAL_HIGH_HEAD",
    "PIPELINE",
    "EXPORT",
    "INJECTION",
    "GAS_LIFT",
    "REFRIGERATION",
    "BOOSTER",
    "SINGLE_STAGE",
    "MULTISTAGE_INLINE",
    "INTEGRALLY_GEARED",
    "OVERHUNG",
)


_GRAPH_NODE_STYLES = {
    "compressor": ("#dbeafe", "#2563eb"),
    "cooler": ("#cffafe", "#0891b2"),
    "heater": ("#ffedd5", "#ea580c"),
    "heat_exchanger": ("#fee2e2", "#dc2626"),
    "valve": ("#f3e8ff", "#9333ea"),
    "pump": ("#dcfce7", "#16a34a"),
    "expander": ("#fef3c7", "#d97706"),
    "pipeline": ("#f1f5f9", "#475569"),
    "separator": ("#e0e7ff", "#4f46e5"),
    "mixer": ("#fae8ff", "#c026d3"),
    "splitter": ("#fce7f3", "#db2777"),
}


_INLINE_UNIT_CATALOG: dict[str, dict[str, Any]] = {
    "compressor": {
        "label": "Compressor",
        "category": "Pressure change",
        "description": "Raise gas pressure with an explicit efficiency.",
        "ports": {
            "material_in": ["in"],
            "material_out": ["out"],
        },
        "default_params": {
            "outlet_pressure_bara": 80.0,
            "isentropic_efficiency": 0.78,
            "use_compressor_chart": False,
            "chart_template": "CENTRIFUGAL_STANDARD",
            "chart_num_speeds": 5,
        },
        "properties": {
            "outlet_pressure_bara": _number_property(
                "Outlet pressure",
                "bara (absolute)",
                1.0,
                500.0,
                1.0,
                "%.2f",
            ),
            "isentropic_efficiency": _number_property(
                "Isentropic efficiency",
                "-",
                0.01,
                1.0,
                0.01,
                "%.3f",
            ),
            "use_compressor_chart": _boolean_property(
                "Use native compressor map",
                (
                    "Generate a NeqSim screening map from the solved design "
                    "point and solve the compressor against that map."
                ),
            ),
            "chart_template": _choice_property(
                "Compressor map template",
                _COMPRESSOR_CHART_TEMPLATES,
                (
                    "Select a supported native NeqSim map family. The map is "
                    "synthetic screening data, not a vendor guarantee."
                ),
            ),
            "chart_num_speeds": _integer_property(
                "Map speed curves",
                "curves",
                3,
                12,
                1,
                "Number of corrected-speed curves in the generated map.",
            ),
        },
    },
    "cooler": {
        "label": "Cooler",
        "category": "Heat transfer",
        "description": "Cool a material stream with an optional pressure loss.",
        "ports": {
            "material_in": ["in"],
            "material_out": ["out"],
        },
        "default_params": {
            "outlet_temperature_C": 35.0,
            "pressure_drop_bar": 0.0,
        },
        "properties": {
            "outlet_temperature_C": _number_property(
                "Outlet temperature",
                "°C",
                -100.0,
                300.0,
                1.0,
                "%.2f",
            ),
            "pressure_drop_bar": _number_property(
                "Pressure drop",
                "bar",
                0.0,
                200.0,
                0.1,
                "%.3f",
            ),
        },
    },
    "heater": {
        "label": "Heater",
        "category": "Heat transfer",
        "description": "Heat a material stream to a specified temperature.",
        "ports": {
            "material_in": ["in"],
            "material_out": ["out"],
        },
        "default_params": {
            "outlet_temperature_C": 50.0,
            "pressure_drop_bar": 0.0,
        },
        "properties": {
            "outlet_temperature_C": _number_property(
                "Outlet temperature",
                "°C",
                -100.0,
                500.0,
                1.0,
                "%.2f",
            ),
            "pressure_drop_bar": _number_property(
                "Pressure drop",
                "bar",
                0.0,
                200.0,
                0.1,
                "%.3f",
            ),
        },
    },
    "heat_exchanger": {
        "label": "Heat Exchanger",
        "category": "Heat transfer",
        "description": (
            "Exchange heat between explicit hot and cold material sides "
            "using a specified overall UA."
        ),
        "ports": {
            "material_in": ["hot_in", "cold_in"],
            "material_out": ["hot_out", "cold_out"],
        },
        "default_params": {
            "ua_w_per_k": 100_000.0,
            "use_design_basis": False,
            "design_duty_capacity_kw": 2_500.0,
            "design_ua_capacity_w_per_k": 125_000.0,
        },
        "properties": {
            "ua_w_per_k": _number_property(
                "Overall conductance UA",
                "W/K",
                1.0,
                1_000_000_000.0,
                1_000.0,
                "%.2f",
            ),
            "use_design_basis": _boolean_property(
                "Evaluate exchanger design limits",
                (
                    "Compare the trusted solved heat-transfer duty and "
                    "native UA with explicit screening capacities."
                ),
            ),
            "design_duty_capacity_kw": _number_property(
                "Design duty capacity",
                "kW (absolute heat-transfer duty)",
                0.001,
                100_000_000.0,
                10.0,
                "%.3f",
            ),
            "design_ua_capacity_w_per_k": _number_property(
                "Design UA capacity",
                "W/K",
                1.0,
                1_000_000_000.0,
                1_000.0,
                "%.2f",
            ),
        },
    },
    "valve": {
        "label": "Valve",
        "category": "Pressure change",
        "description": (
            "Reduce pressure through a throttling valve with an explicit "
            "steady-state opening."
        ),
        "ports": {
            "material_in": ["in"],
            "material_out": ["out"],
        },
        "default_params": {
            "outlet_pressure_bara": 40.0,
            "percent_valve_opening": 100.0,
            "use_design_basis": False,
            "design_cv_capacity_us": 100.0,
        },
        "properties": {
            "outlet_pressure_bara": _number_property(
                "Outlet pressure",
                "bara (absolute)",
                0.01,
                500.0,
                1.0,
                "%.2f",
            ),
            "percent_valve_opening": _number_property(
                "Valve opening",
                "%",
                1.0,
                100.0,
                1.0,
                "%.2f",
            ),
            "use_design_basis": _boolean_property(
                "Evaluate valve Cv limit",
                (
                    "Compare the solved native required Cv with an explicit "
                    "rated US Cv capacity."
                ),
            ),
            "design_cv_capacity_us": _number_property(
                "Rated Cv capacity",
                "US Cv",
                0.001,
                100_000_000.0,
                1.0,
                "%.3f",
            ),
        },
    },
    "pump": {
        "label": "Pump",
        "category": "Pressure change",
        "description": "Raise liquid pressure with an explicit efficiency.",
        "ports": {
            "material_in": ["in"],
            "material_out": ["out"],
        },
        "default_params": {
            "outlet_pressure_bara": 80.0,
            "efficiency": 0.75,
            "use_design_basis": False,
            "design_flow_capacity_m3_per_hr": 100.0,
            "design_head_capacity_m": 600.0,
            "motor_rating_kw": 100.0,
        },
        "properties": {
            "outlet_pressure_bara": _number_property(
                "Outlet pressure",
                "bara (absolute)",
                0.01,
                1000.0,
                1.0,
                "%.2f",
            ),
            "efficiency": _number_property(
                "Efficiency",
                "-",
                0.01,
                1.0,
                0.01,
                "%.3f",
            ),
            "use_design_basis": _boolean_property(
                "Evaluate pump design limits",
                (
                    "Compare the solved native operating point with explicit "
                    "flow, head, and motor capacities."
                ),
            ),
            "design_flow_capacity_m3_per_hr": _number_property(
                "Design flow capacity",
                "m3/hr (actual at pump inlet)",
                0.001,
                1_000_000.0,
                1.0,
                "%.3f",
            ),
            "design_head_capacity_m": _number_property(
                "Design head capacity",
                "m liquid",
                0.1,
                20_000.0,
                1.0,
                "%.3f",
            ),
            "motor_rating_kw": _number_property(
                "Motor rating",
                "kW",
                0.001,
                1_000_000.0,
                1.0,
                "%.3f",
            ),
        },
    },
    "expander": {
        "label": "Expander",
        "category": "Pressure change",
        "description": "Recover work while reducing stream pressure.",
        "ports": {
            "material_in": ["in"],
            "material_out": ["out"],
        },
        "default_params": {
            "outlet_pressure_bara": 30.0,
            "isentropic_efficiency": 0.80,
        },
        "properties": {
            "outlet_pressure_bara": _number_property(
                "Outlet pressure",
                "bara (absolute)",
                0.01,
                500.0,
                1.0,
                "%.2f",
            ),
            "isentropic_efficiency": _number_property(
                "Isentropic efficiency",
                "-",
                0.01,
                1.0,
                0.01,
                "%.3f",
            ),
        },
    },
    "pipeline": {
        "label": "Pipeline",
        "category": "Transport",
        "description": "Transport a stream through a specified pipe geometry.",
        "ports": {
            "material_in": ["in"],
            "material_out": ["out"],
        },
        "default_params": {
            "length": 1000.0,
            "diameter": 0.30,
            "roughness": 1.0e-5,
            "use_design_basis": False,
            "design_pressure_drop_capacity_bar": 1.0,
            "design_velocity_capacity_m_per_s": 20.0,
        },
        "properties": {
            "length": _number_property(
                "Length",
                "m",
                0.01,
                10_000_000.0,
                10.0,
                "%.2f",
            ),
            "diameter": _number_property(
                "Internal diameter",
                "m",
                0.001,
                10.0,
                0.01,
                "%.4f",
            ),
            "roughness": _number_property(
                "Absolute roughness",
                "m",
                0.0,
                0.1,
                1.0e-6,
                "%.7f",
            ),
            "use_design_basis": _boolean_property(
                "Evaluate pipeline hydraulic limits",
                (
                    "Compare solved native pressure drop and mixture velocity "
                    "with explicit screening capacities."
                ),
            ),
            "design_pressure_drop_capacity_bar": _number_property(
                "Design pressure-drop capacity",
                "bar",
                0.000001,
                1_000.0,
                0.01,
                "%.6f",
            ),
            "design_velocity_capacity_m_per_s": _number_property(
                "Design velocity capacity",
                "m/s",
                0.001,
                100.0,
                0.1,
                "%.3f",
            ),
        },
    },
    "separator": {
        "label": "Separator",
        "category": "Separation",
        "description": (
            "Separate one feed into explicit gas and liquid outlet ports."
        ),
        "ports": {
            "material_in": ["in"],
            "material_out": ["gas", "liquid"],
        },
        "default_params": {
            "auto_size": False,
            "design_gas_load_factor_m_per_s": 0.11,
        },
        "properties": {
            "auto_size": _boolean_property(
                "Run native mechanical sizing",
                (
                    "Size the separator from the solved flow before the "
                    "final closed rerun. This is a screening design, not "
                    "design certification."
                ),
            ),
            "design_gas_load_factor_m_per_s": _number_property(
                "Design gas-load factor",
                "m/s",
                0.01,
                1.0,
                0.01,
                "%.3f",
            ),
        },
    },
    "mixer": {
        "label": "Mixer",
        "category": "Flow routing",
        "description": (
            "Combine two independently defined material inlet streams."
        ),
        "ports": {
            "material_in": ["in_0", "in_1"],
            "material_out": ["out"],
        },
        "default_params": {},
        "properties": {},
    },
    "splitter": {
        "label": "Splitter",
        "category": "Flow routing",
        "description": (
            "Divide one material stream between branch outlets with an "
            "editable out_0 flow fraction or normalized multi-outlet "
            "allocations."
        ),
        "ports": {
            "material_in": ["in"],
            "material_out": ["out_0", "out_1"],
        },
        "default_params": {
            "split_factor": 0.5,
        },
        "properties": {
            "split_factor": _number_property(
                "Outlet out_0 flow fraction",
                "-",
                0.0,
                1.0,
                0.01,
                "%.3f",
            ),
        },
    },
}

_PROCESS_UNIT_PROPERTY_DEFINITIONS: dict[str, dict[str, Any]] = {
    unit_type: {
        "default_params": definition["default_params"],
        "properties": definition["properties"],
    }
    for unit_type, definition in _INLINE_UNIT_CATALOG.items()
}
_INLET_CONDITION_PROPERTIES = {
    "temperature_C": _number_property(
        "Temperature",
        "°C",
        -100.0,
        200.0,
        1.0,
        "%.2f",
    ),
    "pressure_bara": _number_property(
        "Pressure",
        "bara (absolute)",
        1.0,
        500.0,
        1.0,
        "%.2f",
    ),
    "total_flow": _number_property(
        "Mass flow",
        "kg/hr",
        1.0,
        10_000_000.0,
        1_000.0,
        "%.2f",
    ),
}


def inline_unit_catalog() -> dict[str, dict[str, Any]]:
    """Return an isolated copy of units safe for inline graph insertion."""
    return copy.deepcopy(_INLINE_UNIT_CATALOG)


def inline_unit_catalog_rows() -> list[dict[str, Any]]:
    """Return deterministic searchable palette rows for presentation layers."""
    return [
        {
            "Type": unit_type,
            "Equipment": definition["label"],
            "Category": definition["category"],
            "Description": definition["description"],
        }
        for unit_type, definition in _INLINE_UNIT_CATALOG.items()
    ]


def process_unit_property_rows(
    unit_type: str,
    params: Any = None,
) -> list[dict[str, Any]]:
    """Return deterministic explicit-unit property rows for one process unit."""
    cleaned_type = str(unit_type).strip().lower()
    definition = _PROCESS_UNIT_PROPERTY_DEFINITIONS.get(cleaned_type)
    if definition is None:
        raise ValueError(f"Unsupported process unit type '{cleaned_type}'.")
    if params is None:
        selected_params = definition["default_params"]
    elif isinstance(params, dict):
        selected_params = copy.deepcopy(params)
    else:
        raise ValueError("Process unit params must be an object.")

    if cleaned_type == "splitter" and not selected_params:
        # Schema-v3 palette splitters were persisted with empty params before
        # their equal allocation became an editor-backed property.
        selected_params = copy.deepcopy(definition["default_params"])

    if cleaned_type == "separator":
        # Schema-v3 separators were parameterless before mechanical sizing
        # became opt-in. Preserve their operating behavior while presenting
        # the complete current property contract.
        for key, value in definition["default_params"].items():
            selected_params.setdefault(key, copy.deepcopy(value))

    if cleaned_type == "compressor":
        # Earlier graph schemas stored only pressure and efficiency. Preserve
        # that operating behavior while making native maps explicitly opt-in.
        for key in (
            "use_compressor_chart",
            "chart_template",
            "chart_num_speeds",
        ):
            selected_params.setdefault(
                key,
                copy.deepcopy(definition["default_params"][key]),
            )

    if cleaned_type == "pump":
        # Earlier graph schemas stored only outlet pressure and efficiency.
        # Preserve their operating behavior while making design-limit
        # evaluation explicitly opt-in.
        for key in (
            "use_design_basis",
            "design_flow_capacity_m3_per_hr",
            "design_head_capacity_m",
            "motor_rating_kw",
        ):
            selected_params.setdefault(
                key,
                copy.deepcopy(definition["default_params"][key]),
            )

    if cleaned_type == "heat_exchanger":
        # Earlier graph schemas stored only the operating UA. Preserve that
        # solve while making screening capacities explicitly opt-in.
        for key in (
            "use_design_basis",
            "design_duty_capacity_kw",
            "design_ua_capacity_w_per_k",
        ):
            selected_params.setdefault(
                key,
                copy.deepcopy(definition["default_params"][key]),
            )

    if cleaned_type == "pipeline":
        # Earlier graph schemas stored only geometry. Preserve that solve and
        # make hydraulic-capacity screening explicitly opt-in.
        for key in (
            "use_design_basis",
            "design_pressure_drop_capacity_bar",
            "design_velocity_capacity_m_per_s",
        ):
            selected_params.setdefault(
                key,
                copy.deepcopy(definition["default_params"][key]),
            )

    if cleaned_type == "splitter" and "split_factors" in selected_params:
        if "split_factor" in selected_params:
            raise ValueError(
                "Process splitter has conflicting split_factor and "
                "split_factors properties."
            )
        raw_factors = selected_params.pop("split_factors")
        if not isinstance(raw_factors, list) or len(raw_factors) != 2:
            raise ValueError(
                "Process splitter split_factors must contain exactly two values."
            )
        factors: list[float] = []
        for raw_factor in raw_factors:
            if isinstance(raw_factor, bool):
                raise ValueError(
                    "Process splitter split_factors must be numeric."
                )
            try:
                factor = float(raw_factor)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Process splitter split_factors must be numeric."
                ) from error
            if not math.isfinite(factor) or factor < 0.0:
                raise ValueError(
                    "Process splitter split_factors must be finite and "
                    "non-negative."
                )
            factors.append(factor)
        factor_scale = max(factors)
        if factor_scale <= 0.0:
            raise ValueError(
                "Process splitter split_factors must have a positive sum."
            )
        scaled_factors = [factor / factor_scale for factor in factors]
        selected_params["split_factor"] = (
            scaled_factors[0] / sum(scaled_factors)
        )

    if cleaned_type == "valve":
        # Earlier graph schemas stored only the specified outlet pressure or
        # pressure plus opening. Preserve that solve and keep rated-Cv
        # screening explicitly opt-in.
        for key in (
            "percent_valve_opening",
            "use_design_basis",
            "design_cv_capacity_us",
        ):
            selected_params.setdefault(
                key,
                copy.deepcopy(definition["default_params"][key]),
            )

    property_keys = set(definition["properties"])
    parameter_keys = set(selected_params)
    missing = sorted(property_keys - parameter_keys)
    unknown = sorted(parameter_keys - property_keys)
    if missing:
        raise ValueError(
            f"Process unit '{cleaned_type}' is missing property '{missing[0]}'."
        )
    if unknown:
        raise ValueError(
            f"Process unit '{cleaned_type}' has unsupported property "
            f"'{unknown[0]}'."
        )

    rows: list[dict[str, Any]] = []
    for key, metadata in definition["properties"].items():
        raw_value = selected_params[key]
        if metadata["kind"] == "boolean":
            if type(raw_value) is not bool:
                raise ValueError(
                    f"Process unit property '{key}' must be boolean."
                )
            rows.append(
                {
                    "key": key,
                    "kind": "boolean",
                    "label": metadata["label"],
                    "description": metadata["description"],
                    "unit": "",
                    "value": raw_value,
                }
            )
            continue
        if metadata["kind"] == "choice":
            if not isinstance(raw_value, str) or raw_value not in metadata[
                "choices"
            ]:
                raise ValueError(
                    f"Process unit property '{key}' must be one of: "
                    + ", ".join(metadata["choices"])
                    + "."
                )
            rows.append(
                {
                    "key": key,
                    "kind": "choice",
                    "label": metadata["label"],
                    "description": metadata["description"],
                    "unit": metadata["unit"],
                    "value": raw_value,
                    "choices": list(metadata["choices"]),
                }
            )
            continue
        if metadata["kind"] == "integer":
            if isinstance(raw_value, bool):
                raise ValueError(
                    f"Process unit property '{key}' must be an integer."
                )
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Process unit property '{key}' must be an integer."
                ) from error
            if not math.isfinite(numeric_value) or not numeric_value.is_integer():
                raise ValueError(
                    f"Process unit property '{key}' must be an integer."
                )
            value = int(numeric_value)
            if value < metadata["minimum"] or value > metadata["maximum"]:
                raise ValueError(
                    f"Process unit property '{key}' must be between "
                    f"{metadata['minimum']} and {metadata['maximum']} "
                    f"{metadata['unit']}."
                )
            rows.append(
                {
                    "key": key,
                    "kind": "integer",
                    "label": metadata["label"],
                    "description": metadata["description"],
                    "unit": metadata["unit"],
                    "value": value,
                    "minimum": metadata["minimum"],
                    "maximum": metadata["maximum"],
                    "step": metadata["step"],
                    "format": metadata["format"],
                }
            )
            continue
        if isinstance(raw_value, bool):
            raise ValueError(f"Process unit property '{key}' must be numeric.")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Process unit property '{key}' must be numeric."
            ) from error
        if not math.isfinite(value):
            raise ValueError(f"Process unit property '{key}' must be finite.")
        if value < metadata["minimum"] or value > metadata["maximum"]:
            raise ValueError(
                f"Process unit property '{key}' must be between "
                f"{metadata['minimum']} and {metadata['maximum']} "
                f"{metadata['unit']}."
            )
        rows.append(
            {
                "key": key,
                "kind": "number",
                "label": metadata["label"],
                "unit": metadata["unit"],
                "value": value,
                "minimum": metadata["minimum"],
                "maximum": metadata["maximum"],
                "step": metadata["step"],
                "format": metadata["format"],
            }
        )
    return rows


def inline_unit_property_rows(
    unit_type: str,
    params: Any = None,
) -> list[dict[str, Any]]:
    """Return explicit-unit rows for a unit allowed in the inline palette."""
    cleaned_type = str(unit_type).strip().lower()
    if cleaned_type not in _INLINE_UNIT_CATALOG:
        raise ValueError(f"Unsupported inline unit type '{cleaned_type}'.")
    return process_unit_property_rows(cleaned_type, params)


def inlet_condition_property_rows(inlet: Any) -> list[dict[str, Any]]:
    """Return explicit-unit rows for one material-inlet operating condition."""
    if not isinstance(inlet, dict):
        raise ValueError("Material inlet must be an object.")
    inlet_id = str(inlet.get("id", "")).strip()
    if not inlet_id:
        raise ValueError("Material inlet requires a non-empty id.")
    if inlet.get("flow_unit") != "kg/hr":
        raise ValueError(
            f"Material inlet '{inlet_id}' requires mass flow in kg/hr."
        )

    condition_keys = set(_INLET_CONDITION_PROPERTIES)
    missing = sorted(condition_keys - set(inlet))
    if missing:
        raise ValueError(
            f"Material inlet '{inlet_id}' is missing condition '{missing[0]}'."
        )

    rows: list[dict[str, Any]] = []
    for key, metadata in _INLET_CONDITION_PROPERTIES.items():
        raw_value = inlet[key]
        if isinstance(raw_value, bool):
            raise ValueError(f"Material inlet condition '{key}' must be numeric.")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Material inlet condition '{key}' must be numeric."
            ) from error
        if not math.isfinite(value):
            raise ValueError(f"Material inlet condition '{key}' must be finite.")
        if value < metadata["minimum"] or value > metadata["maximum"]:
            raise ValueError(
                f"Material inlet condition '{key}' must be between "
                f"{metadata['minimum']} and {metadata['maximum']} "
                f"{metadata['unit']}."
            )
        rows.append(
            {
                "key": key,
                "label": metadata["label"],
                "unit": metadata["unit"],
                "value": value,
                "minimum": metadata["minimum"],
                "maximum": metadata["maximum"],
                "step": metadata["step"],
                "format": metadata["format"],
            }
        )
    return rows


def inlet_composition_property_rows(inlet: Any) -> list[dict[str, Any]]:
    """Return validated mole-fraction rows for one material inlet."""
    if not isinstance(inlet, dict):
        raise ValueError("Material inlet must be an object.")
    inlet_id = str(inlet.get("id", "")).strip()
    if not inlet_id:
        raise ValueError("Material inlet requires a non-empty id.")
    if inlet.get("composition_basis") != "mole_fraction":
        raise ValueError(
            f"Material inlet '{inlet_id}' requires mole-fraction composition."
        )

    composition = inlet.get("composition")
    if not isinstance(composition, dict) or not composition:
        raise ValueError(
            f"Material inlet '{inlet_id}' requires a non-empty composition."
        )

    rows: list[dict[str, Any]] = []
    component_keys: set[str] = set()
    composition_total = 0.0
    for component_name, raw_value in composition.items():
        cleaned_component = str(component_name).strip()
        if not cleaned_component:
            raise ValueError(
                f"Material inlet '{inlet_id}' has an empty component name."
            )
        component_key = cleaned_component.casefold()
        if component_key in component_keys:
            raise ValueError(
                f"Material inlet '{inlet_id}' has duplicate component "
                f"'{cleaned_component}'."
            )
        component_keys.add(component_key)

        if isinstance(raw_value, bool):
            raise ValueError(
                f"Material inlet component '{cleaned_component}' must be numeric."
            )
        try:
            mole_fraction = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Material inlet component '{cleaned_component}' must be numeric."
            ) from error
        if not math.isfinite(mole_fraction):
            raise ValueError(
                f"Material inlet component '{cleaned_component}' must be finite."
            )
        if not 0.0 <= mole_fraction <= 1.0:
            raise ValueError(
                f"Material inlet component '{cleaned_component}' must be "
                "between 0 and 1 mol/mol."
            )
        composition_total += mole_fraction
        rows.append(
            {
                "component": cleaned_component,
                "mole_fraction": mole_fraction,
                "unit": "mol/mol",
                "minimum": 0.0,
                "maximum": 1.0,
                "format": "%.6f",
            }
        )

    if not math.isclose(
        composition_total,
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise ValueError(
            f"Material inlet '{inlet_id}' mole fractions must sum to 1.0."
        )
    return rows


def update_inlet_composition(
    inlets: list[Any],
    inlet_id: str,
    composition: Any,
) -> list[dict[str, Any]]:
    """Transactionally replace one inlet's normalized mole fractions."""
    if not isinstance(inlets, list):
        raise ValueError("Graph inlets must be an array.")
    if not isinstance(composition, dict):
        raise ValueError("Material inlet composition must be an object.")

    cleaned_inlet_id = str(inlet_id).strip()
    copied_inlets = copy.deepcopy(inlets)
    matches = [
        index
        for index, inlet in enumerate(copied_inlets)
        if isinstance(inlet, dict)
        and str(inlet.get("id", "")).strip() == cleaned_inlet_id
    ]
    if not matches:
        raise ValueError(f"Unknown material inlet '{cleaned_inlet_id}'.")
    if len(matches) > 1:
        raise ValueError(f"Material inlet id '{cleaned_inlet_id}' is duplicated.")

    selected_inlet = copied_inlets[matches[0]]
    current_rows = inlet_composition_property_rows(selected_inlet)
    component_order = [row["component"] for row in current_rows]
    if set(composition) != set(component_order):
        raise ValueError(
            f"Material inlet '{cleaned_inlet_id}' composition must match its "
            "shared component registry exactly."
        )

    entered_values: dict[str, float] = {}
    entered_total = 0.0
    for component_name in component_order:
        raw_value = composition[component_name]
        if isinstance(raw_value, bool):
            raise ValueError(
                f"Material inlet component '{component_name}' must be numeric."
            )
        try:
            mole_fraction = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Material inlet component '{component_name}' must be numeric."
            ) from error
        if not math.isfinite(mole_fraction):
            raise ValueError(
                f"Material inlet component '{component_name}' must be finite."
            )
        if not 0.0 <= mole_fraction <= 1.0:
            raise ValueError(
                f"Material inlet component '{component_name}' must be "
                "between 0 and 1 mol/mol."
            )
        entered_values[component_name] = mole_fraction
        entered_total += mole_fraction

    if entered_total <= 0.0:
        raise ValueError(
            f"Material inlet '{cleaned_inlet_id}' composition total must be "
            "positive."
        )
    normalized_composition = {
        component_name: entered_values[component_name] / entered_total
        for component_name in component_order
    }
    updated_inlet = {
        **selected_inlet,
        "composition": normalized_composition,
    }
    inlet_composition_property_rows(updated_inlet)
    copied_inlets[matches[0]] = updated_inlet
    return copied_inlets


def update_inlet_conditions(
    inlets: list[Any],
    inlet_id: str,
    condition_updates: Any,
) -> list[dict[str, Any]]:
    """Transactionally update one inlet's independent operating conditions."""
    if not isinstance(inlets, list):
        raise ValueError("Graph inlets must be an array.")
    if not isinstance(condition_updates, dict):
        raise ValueError("Material inlet condition updates must be an object.")

    cleaned_inlet_id = str(inlet_id).strip()
    copied_inlets = copy.deepcopy(inlets)
    matches = [
        index
        for index, inlet in enumerate(copied_inlets)
        if isinstance(inlet, dict)
        and str(inlet.get("id", "")).strip() == cleaned_inlet_id
    ]
    if not matches:
        raise ValueError(f"Unknown material inlet '{cleaned_inlet_id}'.")
    if len(matches) > 1:
        raise ValueError(f"Material inlet id '{cleaned_inlet_id}' is duplicated.")

    unknown_updates = sorted(
        set(condition_updates) - set(_INLET_CONDITION_PROPERTIES)
    )
    if unknown_updates:
        raise ValueError(
            f"Material inlet '{cleaned_inlet_id}' has unsupported condition "
            f"'{unknown_updates[0]}'."
        )

    selected_inlet = copied_inlets[matches[0]]
    updated_inlet = {
        **selected_inlet,
        **copy.deepcopy(condition_updates),
    }
    property_rows = inlet_condition_property_rows(updated_inlet)
    for row in property_rows:
        updated_inlet[row["key"]] = row["value"]
    copied_inlets[matches[0]] = updated_inlet
    return copied_inlets


def _slugify(value: str) -> str:
    """Convert a user-facing name to a stable graph-id stem."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or "unit"


def _normalized_name_keys(values: Any) -> set[str]:
    """Return case-insensitive non-null, non-blank names."""
    if values is None:
        return set()
    result: set[str] = set()
    for value in values:
        if value is None:
            continue
        cleaned = str(value).strip()
        if cleaned:
            result.add(cleaned.casefold())
    return result


def clone_material_inlet(
    inlets: list[Any],
    source_inlet_id: str,
    name: str,
    reserved_ids: set[str] | None = None,
    reserved_names: set[str] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Clone one compatible feed into a new independently editable inlet."""
    if not isinstance(inlets, list) or not inlets:
        raise ValueError("Graph inlets must be a non-empty array.")

    copied_inlets = copy.deepcopy(inlets)
    cleaned_source_id = str(source_inlet_id).strip()
    source_matches = [
        inlet
        for inlet in copied_inlets
        if isinstance(inlet, dict)
        and str(inlet.get("id", "")).strip() == cleaned_source_id
    ]
    if not source_matches:
        raise ValueError(f"Unknown material inlet '{cleaned_source_id}'.")
    if len(source_matches) > 1:
        raise ValueError(f"Material inlet id '{cleaned_source_id}' is duplicated.")

    cleaned_name = str(name).strip()
    if not cleaned_name:
        raise ValueError("Material inlet name cannot be empty.")
    if len(cleaned_name) > 80:
        raise ValueError("Material inlet name cannot exceed 80 characters.")

    inlet_names = _normalized_name_keys(
        inlet.get("name")
        for inlet in copied_inlets
        if isinstance(inlet, dict)
    )
    inlet_names.update(_normalized_name_keys(reserved_names))
    if cleaned_name.casefold() in inlet_names:
        raise ValueError(f"Material inlet name '{cleaned_name}' is duplicated.")

    existing_ids = {
        str(inlet.get("id", "")).strip()
        for inlet in copied_inlets
        if isinstance(inlet, dict)
    }
    existing_ids.update(
        str(reserved_id).strip() for reserved_id in (reserved_ids or set())
    )
    id_stem = _slugify(cleaned_name)
    inlet_id = id_stem
    suffix = 2
    while inlet_id in existing_ids:
        inlet_id = f"{id_stem}-{suffix}"
        suffix += 1

    cloned_inlet = copy.deepcopy(source_matches[0])
    cloned_inlet.update(
        {
            "id": inlet_id,
            "name": cleaned_name,
        }
    )
    inlet_condition_property_rows(cloned_inlet)
    inlet_composition_property_rows(cloned_inlet)
    copied_inlets.append(cloned_inlet)
    return copied_inlets, inlet_id


def rename_material_inlet(
    inlets: list[Any],
    inlet_id: str,
    name: str,
    reserved_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Rename one inlet without changing its stable graph identity."""
    if not isinstance(inlets, list):
        raise ValueError("Graph inlets must be an array.")

    copied_inlets = copy.deepcopy(inlets)
    cleaned_inlet_id = str(inlet_id).strip()
    matches = [
        index
        for index, inlet in enumerate(copied_inlets)
        if isinstance(inlet, dict)
        and str(inlet.get("id", "")).strip() == cleaned_inlet_id
    ]
    if not matches:
        raise ValueError(f"Unknown material inlet '{cleaned_inlet_id}'.")
    if len(matches) > 1:
        raise ValueError(f"Material inlet id '{cleaned_inlet_id}' is duplicated.")

    cleaned_name = str(name).strip()
    if not cleaned_name:
        raise ValueError("Material inlet name cannot be empty.")
    if len(cleaned_name) > 80:
        raise ValueError("Material inlet name cannot exceed 80 characters.")
    peer_names = _normalized_name_keys(
        inlet.get("name")
        for index, inlet in enumerate(copied_inlets)
        if index != matches[0] and isinstance(inlet, dict)
    )
    peer_names.update(_normalized_name_keys(reserved_names))
    if cleaned_name.casefold() in peer_names:
        raise ValueError(f"Material inlet name '{cleaned_name}' is duplicated.")

    copied_inlets[matches[0]]["name"] = cleaned_name
    return copied_inlets


def remove_material_inlet(
    inlets: list[Any],
    connections: list[Any],
    inlet_id: str,
    protected_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Remove one unconnected inlet while preserving a valid feed boundary."""
    if not isinstance(inlets, list):
        raise ValueError("Graph inlets must be an array.")
    if not isinstance(connections, list):
        raise ValueError("Graph connections must be an array.")

    cleaned_inlet_id = str(inlet_id).strip()
    protected = {
        str(protected_id).strip() for protected_id in (protected_ids or set())
    }
    if cleaned_inlet_id in protected:
        raise ValueError(f"Material inlet '{cleaned_inlet_id}' is protected.")

    copied_inlets = copy.deepcopy(inlets)
    matches = [
        index
        for index, inlet in enumerate(copied_inlets)
        if isinstance(inlet, dict)
        and str(inlet.get("id", "")).strip() == cleaned_inlet_id
    ]
    if not matches:
        raise ValueError(f"Unknown material inlet '{cleaned_inlet_id}'.")
    if len(matches) > 1:
        raise ValueError(f"Material inlet id '{cleaned_inlet_id}' is duplicated.")
    if len(copied_inlets) == 1:
        raise ValueError("A flowsheet requires at least one material inlet.")

    for connection in connections:
        if not isinstance(connection, dict):
            raise ValueError("Graph connections must contain objects.")
        for endpoint in (connection.get("source"), connection.get("target")):
            if (
                isinstance(endpoint, dict)
                and str(endpoint.get("kind", "")).strip().lower() == "inlet"
                and str(endpoint.get("id", "")).strip() == cleaned_inlet_id
            ):
                raise ValueError(
                    f"Material inlet '{cleaned_inlet_id}' is still connected."
                )

    copied_inlets.pop(matches[0])
    return copied_inlets


def create_inline_unit_spec(
    unit_type: str,
    name: str,
    existing_ids: set[str],
) -> dict[str, Any]:
    """Create a catalog-backed unit with a collision-free stable graph id."""
    cleaned_type = str(unit_type).strip().lower()
    definition = _INLINE_UNIT_CATALOG.get(cleaned_type)
    if definition is None:
        raise ValueError(f"Unsupported inline unit type '{cleaned_type}'.")

    cleaned_name = str(name).strip()
    if not cleaned_name:
        raise ValueError("Equipment name cannot be empty.")
    if len(cleaned_name) > 80:
        raise ValueError("Equipment name cannot exceed 80 characters.")

    normalized_existing_ids = {
        str(existing_id).strip() for existing_id in existing_ids
    }
    id_stem = _slugify(cleaned_name)
    unit_id = id_stem
    suffix = 2
    while unit_id in normalized_existing_ids:
        unit_id = f"{id_stem}-{suffix}"
        suffix += 1

    return {
        "id": unit_id,
        "name": cleaned_name,
        "type": cleaned_type,
        "ports": copy.deepcopy(definition["ports"]),
        "params": copy.deepcopy(definition["default_params"]),
    }


def add_catalog_unit(
    units: list[Any],
    unit_type: str,
    name: str,
    reserved_ids: set[str] | None = None,
    reserved_names: set[str] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Add one unconnected catalog unit for subsequent explicit port routing."""
    if not isinstance(units, list):
        raise ValueError("Graph units must be an array.")

    copied_units = copy.deepcopy(units)
    existing_ids = {
        str(unit.get("id", "")).strip()
        for unit in copied_units
        if isinstance(unit, dict)
    }
    existing_ids.update(
        str(reserved_id).strip() for reserved_id in (reserved_ids or set())
    )
    cleaned_name = str(name).strip()
    existing_names = _normalized_name_keys(
        unit.get("name")
        for unit in copied_units
        if isinstance(unit, dict)
    )
    existing_names.update(_normalized_name_keys(reserved_names))
    if cleaned_name.casefold() in existing_names:
        raise ValueError(f"Equipment name '{cleaned_name}' is duplicated.")

    unit = create_inline_unit_spec(
        unit_type,
        cleaned_name,
        existing_ids,
    )
    validate_catalog_unit(unit)
    copied_units.append(unit)
    return copied_units, unit["id"]


def _validated_mixer_inlet_ports(
    unit_id: str,
    ports: Any,
) -> list[str]:
    """Return a mixer's contiguous, explicitly indexed material inlet ports."""
    if not isinstance(ports, dict):
        raise ValueError(f"Inline mixer '{unit_id}' requires ports.")
    material_inputs = ports.get("material_in")
    if not isinstance(material_inputs, list) or len(material_inputs) < 2:
        raise ValueError(
            f"Inline mixer '{unit_id}' requires at least two material inlet ports."
        )
    if len(material_inputs) > MAX_MULTI_INLET_PORTS:
        raise ValueError(
            f"Inline mixer '{unit_id}' cannot exceed "
            f"{MAX_MULTI_INLET_PORTS} material inlet ports."
        )
    expected_inputs = [
        f"in_{index}" for index in range(len(material_inputs))
    ]
    if material_inputs != expected_inputs:
        raise ValueError(
            f"Inline mixer '{unit_id}' material inlet ports must be contiguous "
            "from 'in_0'."
        )
    if ports.get("material_out") != ["out"]:
        raise ValueError(
            f"Inline mixer '{unit_id}' requires the material outlet port 'out'."
        )
    unexpected_port_groups = sorted(
        key
        for key, value in ports.items()
        if key not in {"material_in", "material_out"} and value
    )
    if unexpected_port_groups:
        raise ValueError(
            f"Inline mixer '{unit_id}' has unsupported port group "
            f"'{unexpected_port_groups[0]}'."
        )
    return list(material_inputs)


def _validated_separator_inlet_ports(
    unit_id: str,
    ports: Any,
) -> list[str]:
    """Return a separator's backward-compatible explicit material inlets."""
    if not isinstance(ports, dict):
        raise ValueError(f"Inline separator '{unit_id}' requires ports.")
    material_inputs = ports.get("material_in")
    if not isinstance(material_inputs, list) or not material_inputs:
        raise ValueError(
            f"Inline separator '{unit_id}' requires a material inlet port."
        )
    if len(material_inputs) > MAX_MULTI_INLET_PORTS:
        raise ValueError(
            f"Inline separator '{unit_id}' cannot exceed "
            f"{MAX_MULTI_INLET_PORTS} material inlet ports."
        )
    expected_inputs = [
        "in",
        *[
            f"in_{index}"
            for index in range(1, len(material_inputs))
        ],
    ]
    if material_inputs != expected_inputs:
        raise ValueError(
            f"Inline separator '{unit_id}' material inlet ports must start "
            "with 'in' and continue contiguously from 'in_1'."
        )
    if ports.get("material_out") != ["gas", "liquid"]:
        raise ValueError(
            f"Inline separator '{unit_id}' requires material outlet ports "
            "'gas' and 'liquid'."
        )
    unexpected_port_groups = sorted(
        key
        for key, value in ports.items()
        if key not in {"material_in", "material_out"} and value
    )
    if unexpected_port_groups:
        raise ValueError(
            f"Inline separator '{unit_id}' has unsupported port group "
            f"'{unexpected_port_groups[0]}'."
        )
    return list(material_inputs)


def _validated_splitter_outlet_ports(
    unit_id: str,
    ports: Any,
) -> list[str]:
    """Return a splitter's contiguous, explicitly indexed outlet ports."""
    if not isinstance(ports, dict):
        raise ValueError(f"Inline splitter '{unit_id}' requires ports.")
    if ports.get("material_in") != ["in"]:
        raise ValueError(
            f"Inline splitter '{unit_id}' requires the material inlet port 'in'."
        )
    material_outputs = ports.get("material_out")
    if not isinstance(material_outputs, list) or len(material_outputs) < 2:
        raise ValueError(
            f"Inline splitter '{unit_id}' requires at least two material "
            "outlet ports."
        )
    if len(material_outputs) > MAX_SPLITTER_OUTLET_PORTS:
        raise ValueError(
            f"Inline splitter '{unit_id}' cannot exceed "
            f"{MAX_SPLITTER_OUTLET_PORTS} material outlet ports."
        )
    expected_outputs = [
        f"out_{index}" for index in range(len(material_outputs))
    ]
    if material_outputs != expected_outputs:
        raise ValueError(
            f"Inline splitter '{unit_id}' material outlet ports must be "
            "contiguous from 'out_0'."
        )
    unexpected_port_groups = sorted(
        key
        for key, value in ports.items()
        if key not in {"material_in", "material_out"} and value
    )
    if unexpected_port_groups:
        raise ValueError(
            f"Inline splitter '{unit_id}' has unsupported port group "
            f"'{unexpected_port_groups[0]}'."
        )
    return list(material_outputs)


def _normalized_splitter_factors(
    unit_id: str,
    params: Any,
    outlet_count: int,
) -> list[float]:
    """Return finite normalized allocations for every declared outlet."""
    if not isinstance(params, dict):
        raise ValueError(f"Inline splitter '{unit_id}' params must be an object.")
    parameter_keys = set(params)
    unknown = sorted(parameter_keys - {"split_factor", "split_factors"})
    if unknown:
        raise ValueError(
            f"Process splitter has unsupported property '{unknown[0]}'."
        )
    if {"split_factor", "split_factors"} <= parameter_keys:
        raise ValueError(
            "Process splitter has conflicting split_factor and split_factors "
            "properties."
        )

    if "split_factor" in params:
        if outlet_count != 2:
            raise ValueError(
                "Process splitter legacy split_factor requires exactly two "
                "material outlet ports."
            )
        raw_factor = params["split_factor"]
        if isinstance(raw_factor, bool):
            raise ValueError("Process splitter split_factor must be numeric.")
        try:
            split_factor = float(raw_factor)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Process splitter split_factor must be numeric."
            ) from error
        if not math.isfinite(split_factor):
            raise ValueError("Process splitter split_factor must be finite.")
        if split_factor < 0.0 or split_factor > 1.0:
            raise ValueError(
                "Process splitter split_factor must be between 0.0 and 1.0."
            )
        return [split_factor, 1.0 - split_factor]

    if "split_factors" in params:
        raw_factors = params["split_factors"]
        if not isinstance(raw_factors, list):
            raise ValueError(
                "Process splitter split_factors must be an array."
            )
        if len(raw_factors) != outlet_count:
            raise ValueError(
                "Process splitter split_factors must match the declared "
                f"{outlet_count} material outlet ports."
            )
        factors: list[float] = []
        for raw_factor in raw_factors:
            if isinstance(raw_factor, bool):
                raise ValueError(
                    "Process splitter split_factors must be numeric."
                )
            try:
                factor = float(raw_factor)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Process splitter split_factors must be numeric."
                ) from error
            if not math.isfinite(factor) or factor < 0.0:
                raise ValueError(
                    "Process splitter split_factors must be finite and "
                    "non-negative."
                )
            factors.append(factor)
        factor_scale = max(factors)
        if factor_scale <= 0.0:
            raise ValueError(
                "Process splitter split_factors must have a positive sum."
            )
        scaled_factors = [factor / factor_scale for factor in factors]
        scaled_sum = sum(scaled_factors)
        return [factor / scaled_sum for factor in scaled_factors]

    return [1.0 / outlet_count for _ in range(outlet_count)]


def splitter_allocation_rows(unit: Any) -> list[dict[str, Any]]:
    """Return one normalized dimensionless allocation row per splitter outlet."""
    if not isinstance(unit, dict):
        raise ValueError("Splitter unit must be an object.")
    unit_id = str(unit.get("id", "")).strip()
    if str(unit.get("type", "")).strip().lower() != "splitter":
        raise ValueError(f"Graph unit '{unit_id}' is not a splitter.")
    outlets = _validated_splitter_outlet_ports(unit_id, unit.get("ports"))
    factors = _normalized_splitter_factors(
        unit_id,
        unit.get("params"),
        len(outlets),
    )
    return [
        {
            "port": port,
            "label": f"Outlet {port} allocation",
            "unit": "-",
            "value": factor,
        }
        for port, factor in zip(outlets, factors)
    ]


def validate_catalog_unit(unit: Any) -> None:
    """Validate that a unit matches the editor catalog's executable shape."""
    if not isinstance(unit, dict):
        raise ValueError("Catalog unit must be an object.")

    unit_id = str(unit.get("id", "")).strip()
    unit_name = str(unit.get("name", "")).strip()
    unit_type = str(unit.get("type", "")).strip().lower()
    if not unit_id or not unit_name:
        raise ValueError("Catalog unit requires a non-empty id and name.")

    definition = _INLINE_UNIT_CATALOG.get(unit_type)
    if definition is None:
        raise ValueError(f"Unsupported inline unit type '{unit_type}'.")
    if unit_type == "mixer":
        _validated_mixer_inlet_ports(unit_id, unit.get("ports"))
    elif unit_type == "separator":
        _validated_separator_inlet_ports(unit_id, unit.get("ports"))
    elif unit_type == "splitter":
        outlets = _validated_splitter_outlet_ports(
            unit_id,
            unit.get("ports"),
        )
        _normalized_splitter_factors(
            unit_id,
            unit.get("params"),
            len(outlets),
        )
    elif unit.get("ports") != definition["ports"]:
        raise ValueError(
            f"Inline unit '{unit_id}' ports do not match the '{unit_type}' catalog."
        )
    if not isinstance(unit.get("params"), dict):
        raise ValueError(f"Inline unit '{unit_id}' params must be an object.")
    if unit_type != "splitter":
        inline_unit_property_rows(unit_type, unit["params"])


def resize_multi_inlet_unit_ports(
    units: list[Any],
    connections: list[Any],
    unit_id: str,
    inlet_count: Any,
) -> list[dict[str, Any]]:
    """Resize a mixer or separator inlet-port array without dropping routes.

    Expansion appends deterministic explicit ports. Reduction removes only
    trailing ports and is rejected when a removed port still has a material
    connection, preserving graph identity and every existing route. Separator
    port ``in`` remains stable for backward compatibility.
    """
    if not isinstance(units, list):
        raise ValueError("Graph units must be an array.")
    if not isinstance(connections, list):
        raise ValueError("Graph connections must be an array.")
    if isinstance(inlet_count, bool):
        raise ValueError("Material inlet count must be an integer.")
    try:
        normalized_count = int(inlet_count)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Material inlet count must be an integer.") from error
    try:
        numeric_count = float(inlet_count)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Material inlet count must be an integer.") from error
    if not math.isfinite(numeric_count) or numeric_count != normalized_count:
        raise ValueError("Material inlet count must be an integer.")
    if normalized_count > MAX_MULTI_INLET_PORTS:
        raise ValueError(
            "Material inlet count cannot exceed "
            f"{MAX_MULTI_INLET_PORTS}."
        )

    cleaned_unit_id = str(unit_id).strip()
    copied_units = copy.deepcopy(units)
    matches = [
        unit
        for unit in copied_units
        if isinstance(unit, dict)
        and str(unit.get("id", "")).strip() == cleaned_unit_id
    ]
    if not matches:
        raise ValueError(f"Unknown graph unit '{cleaned_unit_id}'.")
    if len(matches) > 1:
        raise ValueError(f"Graph unit id '{cleaned_unit_id}' is duplicated.")
    unit = matches[0]
    unit_type = str(unit.get("type", "")).strip().lower()
    if unit_type == "mixer":
        minimum_count = 2
        current_ports = _validated_mixer_inlet_ports(
            cleaned_unit_id,
            unit.get("ports"),
        )
        requested_ports = [
            f"in_{index}" for index in range(normalized_count)
        ]
    elif unit_type == "separator":
        minimum_count = 1
        current_ports = _validated_separator_inlet_ports(
            cleaned_unit_id,
            unit.get("ports"),
        )
        requested_ports = [
            "in",
            *[
                f"in_{index}"
                for index in range(1, normalized_count)
            ],
        ]
    else:
        raise ValueError(
            f"Graph unit '{cleaned_unit_id}' does not support multiple "
            "material inlet ports."
        )
    if normalized_count < minimum_count:
        raise ValueError(
            f"{unit_type.capitalize()} inlet count must be at least "
            f"{minimum_count}."
        )

    retained_ports = set(requested_ports)
    removed_ports = set(current_ports) - retained_ports
    for index, connection in enumerate(connections):
        if not isinstance(connection, dict):
            raise ValueError(f"Graph connection {index} must be an object.")
        if str(connection.get("type", "")).strip().lower() != "material":
            continue
        target = connection.get("target")
        if not isinstance(target, dict):
            raise ValueError(
                f"Graph connection {index} requires a target object."
            )
        if (
            str(target.get("kind", "")).strip().lower() == "unit"
            and str(target.get("id", "")).strip() == cleaned_unit_id
            and str(target.get("port", "")).strip() in removed_ports
        ):
            connection_id = str(connection.get("id", "")).strip() or str(index)
            raise ValueError(
                f"Disconnect {unit_type} connection '{connection_id}' before "
                f"removing port '{str(target.get('port', '')).strip()}'."
            )

    unit["ports"]["material_in"] = requested_ports
    validate_catalog_unit(unit)
    return copied_units


def resize_mixer_inlet_ports(
    units: list[Any],
    connections: list[Any],
    mixer_id: str,
    inlet_count: Any,
) -> list[dict[str, Any]]:
    """Backward-compatible wrapper for resizing mixer material inlets."""
    return resize_multi_inlet_unit_ports(
        units,
        connections,
        mixer_id,
        inlet_count,
    )


def resize_separator_inlet_ports(
    units: list[Any],
    connections: list[Any],
    separator_id: str,
    inlet_count: Any,
) -> list[dict[str, Any]]:
    """Resize one separator's explicit material inlet ports."""
    return resize_multi_inlet_unit_ports(
        units,
        connections,
        separator_id,
        inlet_count,
    )


def resize_splitter_outlet_ports(
    units: list[Any],
    connections: list[Any],
    splitter_id: str,
    outlet_count: Any,
) -> list[dict[str, Any]]:
    """Resize a splitter's branch outlets without dropping connected routes."""
    if not isinstance(units, list):
        raise ValueError("Graph units must be an array.")
    if not isinstance(connections, list):
        raise ValueError("Graph connections must be an array.")
    if isinstance(outlet_count, bool):
        raise ValueError("Material outlet count must be an integer.")
    try:
        normalized_count = int(outlet_count)
        numeric_count = float(outlet_count)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Material outlet count must be an integer.") from error
    if not math.isfinite(numeric_count) or numeric_count != normalized_count:
        raise ValueError("Material outlet count must be an integer.")
    if normalized_count < 2:
        raise ValueError("Splitter outlet count must be at least 2.")
    if normalized_count > MAX_SPLITTER_OUTLET_PORTS:
        raise ValueError(
            "Material outlet count cannot exceed "
            f"{MAX_SPLITTER_OUTLET_PORTS}."
        )

    cleaned_unit_id = str(splitter_id).strip()
    copied_units = copy.deepcopy(units)
    matches = [
        unit
        for unit in copied_units
        if isinstance(unit, dict)
        and str(unit.get("id", "")).strip() == cleaned_unit_id
    ]
    if not matches:
        raise ValueError(f"Unknown graph unit '{cleaned_unit_id}'.")
    if len(matches) > 1:
        raise ValueError(f"Graph unit id '{cleaned_unit_id}' is duplicated.")
    splitter = matches[0]
    if str(splitter.get("type", "")).strip().lower() != "splitter":
        raise ValueError(
            f"Graph unit '{cleaned_unit_id}' is not a splitter."
        )
    current_ports = _validated_splitter_outlet_ports(
        cleaned_unit_id,
        splitter.get("ports"),
    )
    current_factors = _normalized_splitter_factors(
        cleaned_unit_id,
        splitter.get("params"),
        len(current_ports),
    )
    requested_ports = [
        f"out_{index}" for index in range(normalized_count)
    ]
    removed_ports = set(current_ports) - set(requested_ports)
    for index, connection in enumerate(connections):
        if not isinstance(connection, dict):
            raise ValueError(f"Graph connection {index} must be an object.")
        if str(connection.get("type", "")).strip().lower() != "material":
            continue
        source = connection.get("source")
        if not isinstance(source, dict):
            raise ValueError(
                f"Graph connection {index} requires a source object."
            )
        if (
            str(source.get("kind", "")).strip().lower() == "unit"
            and str(source.get("id", "")).strip() == cleaned_unit_id
            and str(source.get("port", "")).strip() in removed_ports
        ):
            connection_id = str(connection.get("id", "")).strip() or str(index)
            raise ValueError(
                f"Disconnect splitter connection '{connection_id}' before "
                f"removing port '{str(source.get('port', '')).strip()}'."
            )

    if normalized_count > len(current_factors):
        appended_weight = 1.0 / len(current_factors)
        resized_factors = [
            *current_factors,
            *[
                appended_weight
                for _ in range(normalized_count - len(current_factors))
            ],
        ]
    else:
        resized_factors = current_factors[:normalized_count]
    factor_sum = sum(resized_factors)
    if factor_sum <= 0.0:
        resized_factors = [
            1.0 / normalized_count for _ in range(normalized_count)
        ]
    else:
        resized_factors = [
            factor / factor_sum for factor in resized_factors
        ]

    splitter["ports"]["material_out"] = requested_ports
    if normalized_count == 2:
        splitter["params"] = {"split_factor": resized_factors[0]}
    else:
        splitter["params"] = {"split_factors": resized_factors}
    validate_catalog_unit(splitter)
    return copied_units


def update_splitter_allocations(
    units: list[Any],
    splitter_id: str,
    allocation_weights: Any,
) -> list[dict[str, Any]]:
    """Transactionally normalize and store one allocation per outlet."""
    if not isinstance(units, list):
        raise ValueError("Graph units must be an array.")
    cleaned_unit_id = str(splitter_id).strip()
    copied_units = copy.deepcopy(units)
    matches = [
        unit
        for unit in copied_units
        if isinstance(unit, dict)
        and str(unit.get("id", "")).strip() == cleaned_unit_id
    ]
    if not matches:
        raise ValueError(f"Unknown graph unit '{cleaned_unit_id}'.")
    if len(matches) > 1:
        raise ValueError(f"Graph unit id '{cleaned_unit_id}' is duplicated.")
    splitter = matches[0]
    if str(splitter.get("type", "")).strip().lower() != "splitter":
        raise ValueError(
            f"Graph unit '{cleaned_unit_id}' is not a splitter."
        )
    outlets = _validated_splitter_outlet_ports(
        cleaned_unit_id,
        splitter.get("ports"),
    )
    factors = _normalized_splitter_factors(
        cleaned_unit_id,
        {"split_factors": allocation_weights},
        len(outlets),
    )
    if len(outlets) == 2:
        splitter["params"] = {"split_factor": factors[0]}
    else:
        splitter["params"] = {"split_factors": factors}
    validate_catalog_unit(splitter)
    return copied_units


def validate_starter_unit_projection(
    units: list[Any],
    expected_units: list[Any],
    inlets: list[Any] | None = None,
) -> None:
    """Keep retained starter nodes canonical while allowing their removal.

    The backward-compatible ``process`` array still projects the starter
    template. A graph draft may omit any of those nodes when the user
    reorganizes the process, but it may not silently redefine a retained
    starter identity. Replacement equipment must therefore use a new graph id.
    """
    if not isinstance(units, list):
        raise ValueError("Graph units must be an array.")
    if not isinstance(expected_units, list):
        raise ValueError("Expected starter units must be an array.")

    indexed_units: dict[str, dict[str, Any]] = {}
    for index, unit in enumerate(units):
        if not isinstance(unit, dict):
            raise ValueError(f"Graph unit {index} must be an object.")
        unit_id = str(unit.get("id", "")).strip()
        if not unit_id:
            raise ValueError(f"Graph unit {index} requires an id.")
        if unit_id in indexed_units:
            raise ValueError(f"Graph unit id '{unit_id}' is duplicated.")
        indexed_units[unit_id] = unit

    expected_ids: set[str] = set()
    expected_name_keys: set[str] = set()
    for index, expected_unit in enumerate(expected_units):
        if not isinstance(expected_unit, dict):
            raise ValueError(f"Starter unit {index} must be an object.")
        expected_unit_id = str(expected_unit.get("id", "")).strip()
        expected_unit_name = str(expected_unit.get("name", "")).strip()
        if not expected_unit_id:
            raise ValueError(f"Starter unit {index} requires an id.")
        if not expected_unit_name:
            raise ValueError(f"Starter unit {index} requires a name.")
        if expected_unit_id in expected_ids:
            raise ValueError(
                f"Starter unit id '{expected_unit_id}' is duplicated."
            )
        expected_name_key = expected_unit_name.casefold()
        if expected_name_key in expected_name_keys:
            raise ValueError(
                f"Starter unit name '{expected_unit_name}' is duplicated."
            )
        expected_ids.add(expected_unit_id)
        expected_name_keys.add(expected_name_key)
        retained_unit = indexed_units.get(expected_unit_id)
        if retained_unit is not None and retained_unit != expected_unit:
            raise ValueError(
                "Graph units conflict with the starter-template projection at "
                f"'{expected_unit_id}'."
            )

    for unit_id, unit in indexed_units.items():
        if unit_id in expected_ids:
            continue
        unit_name = str(unit.get("name", "")).strip()
        if unit_name.casefold() in expected_name_keys:
            raise ValueError(
                f"Graph unit name '{unit_name}' conflicts with a "
                "starter-template unit identity."
            )

    if inlets is not None:
        if not isinstance(inlets, list):
            raise ValueError("Graph inlets must be an array.")
        for index, inlet in enumerate(inlets):
            if not isinstance(inlet, dict):
                raise ValueError(f"Graph inlet {index} must be an object.")
            inlet_id = str(inlet.get("id", "")).strip()
            if not inlet_id:
                raise ValueError(f"Graph inlet {index} requires an id.")
            if inlet_id in expected_ids:
                raise ValueError(
                    f"Graph inlet id '{inlet_id}' conflicts with a "
                    "starter-template unit identity."
                )
            inlet_name = str(inlet.get("name", "")).strip()
            if inlet_name.casefold() in expected_name_keys:
                raise ValueError(
                    f"Graph inlet name '{inlet_name}' conflicts with a "
                    "starter-template unit identity."
                )


def _connection_index(
    connections: list[Any],
    connection_id: str,
) -> int:
    """Return one unique connection index or fail with an explicit message."""
    matches = [
        index
        for index, connection in enumerate(connections)
        if isinstance(connection, dict)
        and str(connection.get("id", "")).strip() == connection_id
    ]
    if not matches:
        raise ValueError(f"Unknown graph connection '{connection_id}'.")
    if len(matches) > 1:
        raise ValueError(f"Graph connection id '{connection_id}' is duplicated.")
    return matches[0]


def _connection_identity_keys(connections: list[Any]) -> set[str]:
    """Return case-insensitive IDs and effective material-stream names."""
    keys = {
        str(connection.get("id", "")).strip().casefold()
        for connection in connections
        if isinstance(connection, dict)
        and str(connection.get("id", "")).strip()
    }
    keys.update(
        material_connection_name(connection).casefold()
        for connection in connections
        if isinstance(connection, dict)
        and str(connection.get("type", "")).strip().lower() == "material"
    )
    return keys


def _reserved_material_stream_name_keys(
    inlets: list[Any],
    units: list[Any],
    connections: list[Any],
    new_source: dict[str, str],
) -> set[str]:
    """Return process and surviving terminal names unavailable to a new stream."""
    units_by_id = {
        str(unit.get("id", "")).strip(): unit
        for unit in units
        if isinstance(unit, dict)
    }
    reserved = _normalized_name_keys(
        record.get("name")
        for record in [*inlets, *units]
        if isinstance(record, dict)
    )
    connected_outputs = {
        (
            source_id,
            canonical_material_output_port(
                connection.get("source", {}).get("port", ""),
                units_by_id.get(source_id, {}).get("type"),
            ),
        )
        for connection in connections
        if isinstance(connection, dict)
        and str(connection.get("type", "")).strip().lower() == "material"
        and isinstance(connection.get("source"), dict)
        and str(connection["source"].get("kind", "")).strip().lower()
        == "unit"
        for source_id in [
            str(connection["source"].get("id", "")).strip()
        ]
    }
    if str(new_source.get("kind", "")).strip().lower() == "unit":
        source_id = str(new_source.get("id", "")).strip()
        connected_outputs.add(
            (
                source_id,
                canonical_material_output_port(
                    new_source.get("port", ""),
                    units_by_id.get(source_id, {}).get("type"),
                ),
            )
        )

    for unit in units:
        if not isinstance(unit, dict):
            continue
        unit_id = str(unit.get("id", "")).strip()
        unit_name = str(unit.get("name", "")).strip()
        ports = unit.get("ports")
        material_outputs = (
            ports.get("material_out")
            if isinstance(ports, dict)
            else None
        )
        if (
            not unit_id
            or not unit_name
            or not isinstance(material_outputs, list)
        ):
            continue
        for raw_port in material_outputs:
            output_port = str(raw_port).strip()
            if (
                output_port
                and (
                    unit_id,
                    canonical_material_output_port(
                        output_port,
                        unit.get("type"),
                    ),
                )
                not in connected_outputs
            ):
                reserved.add(
                    f"{unit_name} [{output_port}] product".casefold()
                )
    return reserved


def _unique_connection_id(stem: str, reserved_keys: set[str]) -> str:
    """Return a stable ID that cannot shadow an existing material stream."""
    connection_id = _slugify(stem)
    suffix = 2
    while connection_id.casefold() in reserved_keys:
        connection_id = f"{_slugify(stem)}-{suffix}"
        suffix += 1
    return connection_id


def rename_material_connection(
    connections: list[Any],
    connection_id: str,
    stream_name: str,
    reserved_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Rename one material path without changing its stable graph identity."""
    if not isinstance(connections, list):
        raise ValueError("Graph connections must be an array.")
    cleaned_connection_id = str(connection_id).strip()
    if not cleaned_connection_id:
        raise ValueError("Material connection id cannot be empty.")
    cleaned_name = str(stream_name).strip()
    if not cleaned_name:
        raise ValueError("Material stream name cannot be empty.")

    copied_connections = copy.deepcopy(connections)
    connection_index = _connection_index(
        copied_connections,
        cleaned_connection_id,
    )
    selected = copied_connections[connection_index]
    if str(selected.get("type", "")).strip().lower() != "material":
        raise ValueError(
            f"Connection '{cleaned_connection_id}' is not a material stream."
        )

    peer_name_keys = {
        material_connection_name(connection).casefold()
        for index, connection in enumerate(copied_connections)
        if index != connection_index
        and isinstance(connection, dict)
        and str(connection.get("type", "")).strip().lower() == "material"
    }
    peer_name_keys.update(_normalized_name_keys(reserved_names))
    if cleaned_name.casefold() in peer_name_keys:
        raise ValueError(
            f"Material stream name '{cleaned_name}' is already in use."
        )

    selected["name"] = cleaned_name
    return copied_connections


def _graph_port_inventory(
    inlets: list[Any],
    units: list[Any],
    connections: list[Any],
) -> dict[str, Any]:
    """Validate graph ports and report their current connection occupancy."""
    if not isinstance(inlets, list):
        raise ValueError("Graph inlets must be an array.")
    draft = create_graph_draft(units, connections)
    copied_inlets = copy.deepcopy(inlets)
    copied_units = draft["units"]
    copied_connections = draft["connections"]

    object_ids: set[str] = set()
    sources: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    targets: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    def add_port(
        connection_type: str,
        direction: str,
        kind: str,
        object_id: str,
        object_name: str,
        port: str,
    ) -> None:
        record = {
            "connection_type": connection_type,
            "direction": direction,
            "kind": kind,
            "id": object_id,
            "name": object_name,
            "port": port,
        }
        key = (connection_type, kind, object_id, port)
        records = sources if direction == "source" else targets
        if key in records:
            raise ValueError(
                f"Graph {connection_type} {direction} port "
                f"'{object_id}:{port}' is duplicated."
            )
        records[key] = record

    for index, inlet in enumerate(copied_inlets):
        if not isinstance(inlet, dict):
            raise ValueError(f"Graph inlet {index} must be an object.")
        inlet_id = str(inlet.get("id", "")).strip()
        if not inlet_id:
            raise ValueError(f"Graph inlet {index} requires an id.")
        if inlet_id in object_ids:
            raise ValueError(f"Graph object id '{inlet_id}' is duplicated.")
        object_ids.add(inlet_id)
        inlet_name = str(inlet.get("name", inlet_id)).strip() or inlet_id
        add_port(
            "material",
            "source",
            "inlet",
            inlet_id,
            inlet_name,
            "out",
        )

    for unit in copied_units:
        unit_id = str(unit["id"]).strip()
        if unit_id in object_ids:
            raise ValueError(f"Graph object id '{unit_id}' is duplicated.")
        object_ids.add(unit_id)
        unit_name = str(unit.get("name", unit_id)).strip() or unit_id
        ports = unit.get("ports")
        if not isinstance(ports, dict):
            raise ValueError(f"Graph unit '{unit_id}' requires ports.")
        for connection_type in ("material", "energy"):
            input_ports = ports.get(f"{connection_type}_in", [])
            output_ports = ports.get(f"{connection_type}_out", [])
            cleaned_by_direction: dict[str, list[str]] = {}
            for direction, port_names in (
                ("target", input_ports),
                ("source", output_ports),
            ):
                if not isinstance(port_names, list):
                    port_key = (
                        f"{connection_type}_"
                        f"{'in' if direction == 'target' else 'out'}"
                    )
                    raise ValueError(
                        f"Graph unit '{unit_id}' {port_key} must be an array."
                    )
                cleaned_ports = [str(port).strip() for port in port_names]
                if any(not port for port in cleaned_ports):
                    raise ValueError(
                        f"Graph unit '{unit_id}' has an empty "
                        f"{connection_type} port."
                    )
                cleaned_by_direction[direction] = cleaned_ports
                if connection_type == "material" and direction == "source":
                    canonical_ports = [
                        canonical_material_output_port(
                            port,
                            unit.get("type"),
                        )
                        for port in cleaned_ports
                    ]
                    if len(canonical_ports) != len(set(canonical_ports)):
                        raise ValueError(
                            f"Graph unit '{unit_id}' material output ports "
                            "alias the same native outlet."
                        )
                for port in cleaned_ports:
                    add_port(
                        connection_type,
                        direction,
                        "unit",
                        unit_id,
                        unit_name,
                        port,
                    )
            ambiguous_ports = set(cleaned_by_direction["target"]).intersection(
                cleaned_by_direction["source"]
            )
            if ambiguous_ports:
                ambiguous_port = sorted(ambiguous_ports)[0]
                raise ValueError(
                    f"Graph unit '{unit_id}' {connection_type} port "
                    f"'{ambiguous_port}' cannot be both input and output."
                )

    used_sources: set[tuple[str, str, str, str]] = set()
    used_targets: set[tuple[str, str, str, str]] = set()
    used_routes: set[
        tuple[
            tuple[str, str, str, str],
            tuple[str, str, str, str],
        ]
    ] = set()
    for connection in copied_connections:
        connection_id = str(connection["id"]).strip()
        connection_type = str(connection["type"]).strip()
        source = connection["source"]
        target = connection["target"]
        source_key = (
            connection_type,
            str(source["kind"]).strip(),
            str(source["id"]).strip(),
            str(source["port"]).strip(),
        )
        target_key = (
            connection_type,
            str(target["kind"]).strip(),
            str(target["id"]).strip(),
            str(target["port"]).strip(),
        )
        if source_key not in sources:
            raise ValueError(
                f"Connection '{connection_id}' uses an undeclared "
                f"{connection_type} output port."
            )
        if target_key not in targets:
            raise ValueError(
                f"Connection '{connection_id}' uses an undeclared "
                f"{connection_type} input port."
            )
        if source_key[1:3] == target_key[1:3]:
            raise ValueError(
                f"Connection '{connection_id}' cannot connect a node to itself."
            )
        if source_key in used_sources:
            raise ValueError(
                f"Graph output port {source_key[2]}:{source_key[3]} "
                "already has a connection."
            )
        if target_key in used_targets:
            raise ValueError(
                f"Graph input port {target_key[2]}:{target_key[3]} "
                "already has a connection."
            )
        route_key = (source_key, target_key)
        if route_key in used_routes:
            raise ValueError(f"Connection route '{connection_id}' is duplicated.")
        used_sources.add(source_key)
        used_targets.add(target_key)
        used_routes.add(route_key)

    return {
        "sources": sources,
        "targets": targets,
        "used_sources": used_sources,
        "used_targets": used_targets,
        "connections": copied_connections,
    }


def graph_port_rows(
    inlets: list[Any],
    units: list[Any],
    connections: list[Any],
    connection_type: str,
    direction: str,
    available_only: bool = False,
) -> list[dict[str, Any]]:
    """Return deterministic source or target port rows for graph editors."""
    cleaned_type = str(connection_type).strip().lower()
    if cleaned_type not in ("material", "energy"):
        raise ValueError("Connection type must be material or energy.")
    cleaned_direction = str(direction).strip().lower()
    if cleaned_direction not in ("source", "target"):
        raise ValueError("Port direction must be source or target.")
    if not isinstance(available_only, bool):
        raise ValueError("available_only must be a boolean.")

    inventory = _graph_port_inventory(inlets, units, connections)
    records = inventory[
        "sources" if cleaned_direction == "source" else "targets"
    ]
    occupied = inventory[
        "used_sources"
        if cleaned_direction == "source"
        else "used_targets"
    ]
    rows: list[dict[str, Any]] = []
    for key, record in records.items():
        if key[0] != cleaned_type:
            continue
        is_connected = key in occupied
        if available_only and is_connected:
            continue
        row = copy.deepcopy(record)
        row["connected"] = is_connected
        row["endpoint"] = {
            "kind": row["kind"],
            "id": row["id"],
            "port": row["port"],
        }
        row["label"] = (
            f"{row['name']} · {row['kind']} {row['id']}:{row['port']}"
        )
        rows.append(row)
    return rows


def connect_graph_ports(
    inlets: list[Any],
    units: list[Any],
    connections: list[Any],
    connection_type: str,
    source: Any,
    target: Any,
) -> tuple[list[dict[str, Any]], str]:
    """Transactionally connect one available output port to one input port."""
    cleaned_type = str(connection_type).strip().lower()
    if cleaned_type not in ("material", "energy"):
        raise ValueError("Connection type must be material or energy.")
    inventory = _graph_port_inventory(inlets, units, connections)

    normalized_endpoints: dict[str, dict[str, str]] = {}
    endpoint_keys: dict[str, tuple[str, str, str, str]] = {}
    for endpoint_name, endpoint in (("source", source), ("target", target)):
        if not isinstance(endpoint, dict):
            raise ValueError(f"Connection {endpoint_name} must be an object.")
        normalized = {
            field: str(endpoint.get(field, "")).strip()
            for field in ("kind", "id", "port")
        }
        if not all(normalized.values()):
            raise ValueError(
                f"Connection {endpoint_name} requires kind, id, and port."
            )
        normalized_endpoints[endpoint_name] = normalized
        endpoint_keys[endpoint_name] = (
            cleaned_type,
            normalized["kind"],
            normalized["id"],
            normalized["port"],
        )

    source_key = endpoint_keys["source"]
    target_key = endpoint_keys["target"]
    if source_key not in inventory["sources"]:
        raise ValueError("Selected graph source is not a declared output port.")
    if target_key not in inventory["targets"]:
        raise ValueError("Selected graph target is not a declared input port.")
    if source_key[1:3] == target_key[1:3]:
        raise ValueError("A graph connection cannot connect a node to itself.")
    if source_key in inventory["used_sources"]:
        raise ValueError(
            f"Graph output port {source_key[2]}:{source_key[3]} "
            "already has a connection."
        )
    if target_key in inventory["used_targets"]:
        raise ValueError(
            f"Graph input port {target_key[2]}:{target_key[3]} "
            "already has a connection."
        )

    copied_connections = inventory["connections"]
    reserved_connection_keys = _connection_identity_keys(
        copied_connections
    )
    if cleaned_type == "material":
        reserved_connection_keys.update(
            _reserved_material_stream_name_keys(
                inlets,
                units,
                copied_connections,
                normalized_endpoints["source"],
            )
        )
    connection_id = _unique_connection_id(
        (
            f"{cleaned_type}-{source_key[2]}-{source_key[3]}-to-"
            f"{target_key[2]}-{target_key[3]}"
        ),
        reserved_connection_keys,
    )
    copied_connections.append(
        {
            "id": connection_id,
            "name": connection_id,
            "type": cleaned_type,
            "source": normalized_endpoints["source"],
            "target": normalized_endpoints["target"],
        }
    )
    _graph_port_inventory(inlets, units, copied_connections)
    return copied_connections, connection_id


def extend_material_path(
    inlets: list[Any],
    units: list[Any],
    connections: list[Any],
    source: Any,
    unit_type: str,
    unit_name: str,
    reserved_ids: set[str] | None = None,
    reserved_names: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str]:
    """Add one catalog unit and connect an available material source to it.

    The operation is transactional: validation completes against isolated
    copies before the new unit and connection are returned. Inlet identities
    and names are always reserved so an equipment node cannot shadow a feed.
    """
    inlet_ids = {
        str(inlet.get("id", "")).strip()
        for inlet in inlets
        if isinstance(inlet, dict) and str(inlet.get("id", "")).strip()
    }
    inlet_names = {
        str(inlet.get("name", "")).strip()
        for inlet in inlets
        if isinstance(inlet, dict) and str(inlet.get("name", "")).strip()
    }
    combined_ids = inlet_ids.union(
        str(reserved_id).strip()
        for reserved_id in (reserved_ids or set())
        if str(reserved_id).strip()
    )
    combined_names = inlet_names.union(
        str(reserved_name).strip()
        for reserved_name in (reserved_names or set())
        if str(reserved_name).strip()
    )
    copied_units, unit_id = add_catalog_unit(
        units,
        unit_type,
        unit_name,
        combined_ids,
        combined_names,
    )
    new_unit = next(
        unit
        for unit in copied_units
        if str(unit.get("id", "")).strip() == unit_id
    )
    inlet_ports = new_unit["ports"].get("material_in", [])
    if len(inlet_ports) != 1:
        raise ValueError(
            f"Equipment '{unit_id}' must expose exactly one material inlet "
            "for path extension."
        )
    copied_connections, connection_id = connect_graph_ports(
        inlets,
        copied_units,
        connections,
        "material",
        source,
        {
            "kind": "unit",
            "id": unit_id,
            "port": str(inlet_ports[0]).strip(),
        },
    )
    return copied_units, copied_connections, unit_id, connection_id


def disconnect_graph_connection(
    inlets: list[Any],
    units: list[Any],
    connections: list[Any],
    connection_id: str,
) -> list[dict[str, Any]]:
    """Transactionally remove one explicit material or energy connection."""
    inventory = _graph_port_inventory(inlets, units, connections)
    copied_connections = inventory["connections"]
    cleaned_connection_id = str(connection_id).strip()
    selected_index = _connection_index(
        copied_connections,
        cleaned_connection_id,
    )
    copied_connections.pop(selected_index)
    _graph_port_inventory(inlets, units, copied_connections)
    return copied_connections


def _validate_acyclic_material_connections(
    connections: list[Any],
) -> None:
    """Reject material cycles before a draft reaches the acyclic executor."""
    adjacency: dict[tuple[str, str], set[tuple[str, str]]] = {}
    indegree: dict[tuple[str, str], int] = {}
    for connection in connections:
        if (
            not isinstance(connection, dict)
            or str(connection.get("type", "")).strip().lower() != "material"
        ):
            continue
        source = connection.get("source")
        target = connection.get("target")
        if not isinstance(source, dict) or not isinstance(target, dict):
            continue
        source_node = (
            str(source.get("kind", "")).strip(),
            str(source.get("id", "")).strip(),
        )
        target_node = (
            str(target.get("kind", "")).strip(),
            str(target.get("id", "")).strip(),
        )
        adjacency.setdefault(source_node, set())
        adjacency.setdefault(target_node, set())
        indegree.setdefault(source_node, 0)
        indegree.setdefault(target_node, 0)
        if target_node not in adjacency[source_node]:
            adjacency[source_node].add(target_node)
            indegree[target_node] += 1

    available = [
        node for node, dependency_count in indegree.items()
        if dependency_count == 0
    ]
    heapq.heapify(available)
    visited = 0
    while available:
        node = heapq.heappop(available)
        visited += 1
        for downstream in sorted(adjacency[node]):
            indegree[downstream] -= 1
            if indegree[downstream] == 0:
                heapq.heappush(available, downstream)

    if visited != len(indegree):
        next_index = 0
        indices: dict[tuple[str, str], int] = {}
        lowlinks: dict[tuple[str, str], int] = {}
        stack: list[tuple[str, str]] = []
        on_stack: set[tuple[str, str]] = set()
        cyclic_nodes: set[tuple[str, str]] = set()

        for start in sorted(adjacency):
            if start in indices:
                continue
            indices[start] = next_index
            lowlinks[start] = next_index
            next_index += 1
            stack.append(start)
            on_stack.add(start)
            parents: dict[tuple[str, str], tuple[str, str]] = {}
            traversal = [(start, iter(sorted(adjacency[start])))]
            while traversal:
                node, downstream_nodes = traversal[-1]
                try:
                    downstream = next(downstream_nodes)
                except StopIteration:
                    traversal.pop()
                    parent = parents.get(node)
                    if parent is not None:
                        lowlinks[parent] = min(
                            lowlinks[parent],
                            lowlinks[node],
                        )
                    if lowlinks[node] != indices[node]:
                        continue
                    component: list[tuple[str, str]] = []
                    while stack:
                        member = stack.pop()
                        on_stack.remove(member)
                        component.append(member)
                        if member == node:
                            break
                    if len(component) > 1 or node in adjacency[node]:
                        cyclic_nodes.update(component)
                    continue
                if downstream not in indices:
                    parents[downstream] = node
                    indices[downstream] = next_index
                    lowlinks[downstream] = next_index
                    next_index += 1
                    stack.append(downstream)
                    on_stack.add(downstream)
                    traversal.append(
                        (
                            downstream,
                            iter(sorted(adjacency[downstream])),
                        )
                    )
                elif downstream in on_stack:
                    lowlinks[node] = min(
                        lowlinks[node],
                        indices[downstream],
                    )
        cyclic_units = sorted(
            object_id
            for kind, object_id in cyclic_nodes
            if kind == "unit"
        )
        cycle_label = ", ".join(cyclic_units) or "unknown units"
        raise ValueError(
            "Material reroute would create a cycle involving: "
            f"{cycle_label}."
        )


def reroute_graph_connection(
    inlets: list[Any],
    units: list[Any],
    connections: list[Any],
    connection_id: str,
    source: Any,
    target: Any,
) -> list[dict[str, Any]]:
    """Atomically replace both endpoints while preserving connection identity.

    The selected edge is removed from the occupancy inventory before the new
    route is validated, so either original endpoint may be retained. Invalid
    ports, occupied endpoints, self-links, and material cycles fail without
    mutating the caller's graph.
    """
    inventory = _graph_port_inventory(inlets, units, connections)
    copied_connections = inventory["connections"]
    cleaned_connection_id = str(connection_id).strip()
    selected_index = _connection_index(
        copied_connections,
        cleaned_connection_id,
    )
    connection_type = str(
        copied_connections[selected_index]["type"]
    ).strip().lower()
    selected_stream_name = (
        material_connection_name(copied_connections[selected_index])
        if connection_type == "material"
        else None
    )
    remaining_connections = [
        connection
        for index, connection in enumerate(copied_connections)
        if index != selected_index
    ]
    connected, generated_id = connect_graph_ports(
        inlets,
        units,
        remaining_connections,
        connection_type,
        source,
        target,
    )
    replacement = next(
        connection
        for connection in connected
        if str(connection.get("id", "")).strip() == generated_id
    )
    connected.remove(replacement)
    replacement["id"] = cleaned_connection_id
    if selected_stream_name is not None:
        replacement["name"] = selected_stream_name
    connected.insert(selected_index, replacement)
    _graph_port_inventory(inlets, units, connected)
    _validate_acyclic_material_connections(connected)
    return connected


def graph_connection_rows(
    inlets: list[Any],
    units: list[Any],
    connections: list[Any],
) -> list[dict[str, str]]:
    """Return deterministic labels for all removable graph connections."""
    inventory = _graph_port_inventory(inlets, units, connections)
    rows: list[dict[str, str]] = []
    for connection in inventory["connections"]:
        connection_id = str(connection["id"]).strip()
        connection_type = str(connection["type"]).strip()
        source = connection["source"]
        target = connection["target"]
        source_label = (
            f"{str(source['id']).strip()}:{str(source['port']).strip()}"
        )
        target_label = (
            f"{str(target['id']).strip()}:{str(target['port']).strip()}"
        )
        rows.append(
            {
                "id": connection_id,
                "type": connection_type,
                "label": (
                    f"{connection_type.upper()} · "
                    f"{source_label} → {target_label}"
                ),
            }
        )
    return rows


def insert_inline_unit_on_connection(
    units: list[Any],
    connections: list[Any],
    connection_id: str,
    unit_type: str,
    unit_name: str,
    reserved_ids: set[str] | None = None,
    reserved_names: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Transactionally insert one catalog unit into a material connection.

    The existing edge keeps its id and source but terminates at the new unit's
    ``in`` port. A new edge connects the new unit's ``out`` port to the
    original target. Inputs are never mutated, so callers can safely retain an
    undo snapshot before accepting the returned graph.
    """
    if not isinstance(units, list):
        raise ValueError("Graph units must be an array.")
    if not isinstance(connections, list):
        raise ValueError("Graph connections must be an array.")

    copied_units = copy.deepcopy(units)
    copied_connections = copy.deepcopy(connections)
    cleaned_connection_id = str(connection_id).strip()
    selected_index = _connection_index(
        copied_connections,
        cleaned_connection_id,
    )
    selected_connection = copied_connections[selected_index]
    if str(selected_connection.get("type", "")).strip().lower() != "material":
        raise ValueError("Inline equipment can only be inserted in material paths.")

    source = selected_connection.get("source")
    target = selected_connection.get("target")
    if not isinstance(source, dict) or not isinstance(target, dict):
        raise ValueError(
            f"Connection '{cleaned_connection_id}' requires source and target."
        )
    for endpoint_name, endpoint in (("source", source), ("target", target)):
        if not str(endpoint.get("kind", "")).strip():
            raise ValueError(
                f"Connection '{cleaned_connection_id}' {endpoint_name} needs kind."
            )
        if not str(endpoint.get("id", "")).strip():
            raise ValueError(
                f"Connection '{cleaned_connection_id}' {endpoint_name} needs id."
            )
        if not str(endpoint.get("port", "")).strip():
            raise ValueError(
                f"Connection '{cleaned_connection_id}' {endpoint_name} needs port."
            )

    existing_object_ids = {
        str(unit.get("id", "")).strip()
        for unit in copied_units
        if isinstance(unit, dict)
    }
    existing_object_ids.update(
        str(endpoint.get("id", "")).strip()
        for connection in copied_connections
        if isinstance(connection, dict)
        for endpoint in (
            connection.get("source"),
            connection.get("target"),
        )
        if isinstance(endpoint, dict)
    )
    existing_object_ids.update(
        str(reserved_id).strip() for reserved_id in (reserved_ids or set())
    )
    cleaned_name = str(unit_name).strip()
    existing_name_keys = _normalized_name_keys(
        unit.get("name")
        for unit in copied_units
        if isinstance(unit, dict)
    )
    existing_name_keys.update(_normalized_name_keys(reserved_names))
    if cleaned_name.casefold() in existing_name_keys:
        raise ValueError(f"Equipment name '{cleaned_name}' is duplicated.")
    new_unit = create_inline_unit_spec(
        unit_type,
        cleaned_name,
        existing_object_ids,
    )
    validate_catalog_unit(new_unit)

    target_id = str(target["id"]).strip()
    target_index = next(
        (
            index
            for index, unit in enumerate(copied_units)
            if isinstance(unit, dict)
            and str(unit.get("id", "")).strip() == target_id
        ),
        len(copied_units),
    )
    copied_units.insert(target_index, new_unit)

    selected_connection["target"] = {
        "kind": "unit",
        "id": new_unit["id"],
        "port": "in",
    }
    downstream_connection_id = _unique_connection_id(
        f"{new_unit['id']}-to-{target_id}",
        _connection_identity_keys(copied_connections).union(
            existing_name_keys,
            {cleaned_name.casefold()},
        ),
    )
    copied_connections.insert(
        selected_index + 1,
        {
            "id": downstream_connection_id,
            "type": "material",
            "source": {
                "kind": "unit",
                "id": new_unit["id"],
                "port": "out",
            },
            "target": target,
        },
    )
    return copied_units, copied_connections, new_unit["id"]


def insert_mixer_on_connection(
    inlets: list[Any],
    units: list[Any],
    connections: list[Any],
    connection_id: str,
    unit_name: str,
    reserved_ids: set[str] | None = None,
    reserved_names: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str]:
    """Insert a mixer without disconnecting the selected downstream process.

    The selected material source is connected to ``in_0`` and the mixer
    outlet is connected to the original target in one validated transaction.
    ``in_1`` remains available for another feed or process outlet.
    """
    if not isinstance(connections, list):
        raise ValueError("Graph connections must be an array.")
    copied_connections = copy.deepcopy(connections)
    cleaned_connection_id = str(connection_id).strip()
    selected_index = _connection_index(
        copied_connections,
        cleaned_connection_id,
    )
    selected_connection = copied_connections[selected_index]
    if str(selected_connection.get("type", "")).strip().lower() != "material":
        raise ValueError("A mixer can only be inserted in a material path.")
    inventory = _graph_port_inventory(inlets, units, copied_connections)
    copied_connections = inventory["connections"]
    selected_connection = copied_connections[selected_index]

    copied_units, mixer_id = add_catalog_unit(
        units,
        "mixer",
        unit_name,
        {
            str(inlet.get("id", "")).strip()
            for inlet in inlets
            if isinstance(inlet, dict) and str(inlet.get("id", "")).strip()
        }.union(
            str(reserved_id).strip()
            for reserved_id in (reserved_ids or set())
            if str(reserved_id).strip()
        ),
        {
            str(inlet.get("name", "")).strip()
            for inlet in inlets
            if isinstance(inlet, dict) and str(inlet.get("name", "")).strip()
        }.union(
            str(reserved_name).strip()
            for reserved_name in (reserved_names or set())
            if str(reserved_name).strip()
        ),
    )
    mixer = next(
        unit
        for unit in copied_units
        if str(unit.get("id", "")).strip() == mixer_id
    )
    material_inputs = mixer["ports"].get("material_in", [])
    material_outputs = mixer["ports"].get("material_out", [])
    if material_inputs != ["in_0", "in_1"] or material_outputs != ["out"]:
        raise ValueError(
            "Mixer catalog ports must provide in_0, in_1, and out."
        )

    copied_units.remove(mixer)
    original_target = copy.deepcopy(selected_connection["target"])
    target_id = str(original_target["id"]).strip()
    target_index = next(
        (
            index
            for index, unit in enumerate(copied_units)
            if isinstance(unit, dict)
            and str(unit.get("id", "")).strip() == target_id
        ),
        len(copied_units),
    )
    copied_units.insert(target_index, mixer)

    selected_connection["target"] = {
        "kind": "unit",
        "id": mixer_id,
        "port": "in_0",
    }
    downstream_reserved_keys = _connection_identity_keys(
        copied_connections
    )
    downstream_reserved_keys.update(
        _normalized_name_keys(
            record.get("name")
            for record in [*inlets, *copied_units]
            if isinstance(record, dict)
        )
    )
    downstream_reserved_keys.update(_normalized_name_keys(reserved_names))
    downstream_connection_id = _unique_connection_id(
        f"{mixer_id}-out-to-{target_id}",
        downstream_reserved_keys,
    )
    copied_connections.insert(
        selected_index + 1,
        {
            "id": downstream_connection_id,
            "type": "material",
            "source": {
                "kind": "unit",
                "id": mixer_id,
                "port": "out",
            },
            "target": original_target,
        },
    )
    _graph_port_inventory(inlets, copied_units, copied_connections)
    _validate_acyclic_material_connections(copied_connections)
    return (
        copied_units,
        copied_connections,
        mixer_id,
        downstream_connection_id,
    )


def replace_inline_unit(
    units: list[Any],
    connections: list[Any],
    unit_id: str,
    replacement_type: str,
    replacement_name: str,
    reserved_ids: set[str] | None = None,
    reserved_names: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Replace one simple material-path unit without breaking its neighbours.

    The selected unit must have exactly one material ``in`` connection and one
    material ``out`` connection, with no energy or branch references. Removal
    first reconnects the surrounding path on isolated copies; the replacement
    is then inserted into that same connection. Callers therefore receive
    either a complete replacement or the original graph remains untouched.
    """
    if not isinstance(units, list):
        raise ValueError("Graph units must be an array.")
    if not isinstance(connections, list):
        raise ValueError("Graph connections must be an array.")

    inspected_units = units
    inspected_connections = connections
    cleaned_unit_id = str(unit_id).strip()
    unit_matches = [
        unit
        for unit in inspected_units
        if isinstance(unit, dict)
        and str(unit.get("id", "")).strip() == cleaned_unit_id
    ]
    if not unit_matches:
        raise ValueError(f"Unknown graph unit '{cleaned_unit_id}'.")
    if len(unit_matches) > 1:
        raise ValueError(f"Graph unit id '{cleaned_unit_id}' is duplicated.")
    validate_catalog_unit(unit_matches[0])

    cleaned_replacement_type = str(replacement_type).strip().lower()
    replacement_definition = _INLINE_UNIT_CATALOG.get(cleaned_replacement_type)
    if replacement_definition is None:
        raise ValueError(
            f"Unsupported inline unit type '{cleaned_replacement_type}'."
        )
    replacement_ports = replacement_definition["ports"]
    if (
        replacement_ports.get("material_in") != ["in"]
        or replacement_ports.get("material_out") != ["out"]
    ):
        raise ValueError(
            f"Replacement equipment '{cleaned_replacement_type}' must expose "
            "exactly the material ports 'in' and 'out'."
        )

    peer_names = _normalized_name_keys(
        unit.get("name")
        for unit in inspected_units
        if (
            isinstance(unit, dict)
            and str(unit.get("id", "")).strip() != cleaned_unit_id
        )
    )
    peer_names.update(_normalized_name_keys(reserved_names))
    cleaned_replacement_name = str(replacement_name).strip()
    if cleaned_replacement_name.casefold() in peer_names:
        raise ValueError(
            f"Equipment name '{cleaned_replacement_name}' is duplicated."
        )

    incoming_connection_ids = [
        str(connection.get("id", "")).strip()
        for connection in inspected_connections
        if (
            isinstance(connection, dict)
            and str(connection.get("type", "")).strip().lower() == "material"
            and isinstance(connection.get("target"), dict)
            and str(connection["target"].get("kind", "")).strip() == "unit"
            and str(connection["target"].get("id", "")).strip()
            == cleaned_unit_id
            and str(connection["target"].get("port", "")).strip() == "in"
        )
    ]
    if len(incoming_connection_ids) != 1:
        raise ValueError(
            f"Graph unit '{cleaned_unit_id}' requires exactly one material "
            "input before it can be replaced."
        )
    outgoing_connections = [
        connection
        for connection in inspected_connections
        if (
            isinstance(connection, dict)
            and str(connection.get("type", "")).strip().lower() == "material"
            and isinstance(connection.get("source"), dict)
            and str(connection["source"].get("kind", "")).strip() == "unit"
            and str(connection["source"].get("id", "")).strip()
            == cleaned_unit_id
            and str(connection["source"].get("port", "")).strip() == "out"
        )
    ]
    if len(outgoing_connections) != 1:
        raise ValueError(
            f"Graph unit '{cleaned_unit_id}' requires exactly one material "
            "output before it can be replaced."
        )
    replacement_connection_id = incoming_connection_ids[0]

    reduced_units, reduced_connections = remove_inline_unit(
        inspected_units,
        inspected_connections,
        cleaned_unit_id,
    )
    replaced_units, replaced_connections, replacement_id = (
        insert_inline_unit_on_connection(
            reduced_units,
            reduced_connections,
            replacement_connection_id,
            replacement_type,
            cleaned_replacement_name,
            {cleaned_unit_id}.union(
                {
                    str(reserved_id).strip()
                    for reserved_id in (reserved_ids or set())
                }
            ),
        )
    )
    return replaced_units, replaced_connections, replacement_id


def rename_inline_unit(
    units: list[Any],
    unit_id: str,
    new_name: str,
    reserved_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Rename one catalog unit without changing its stable graph identity."""
    if not isinstance(units, list):
        raise ValueError("Graph units must be an array.")

    copied_units = copy.deepcopy(units)
    cleaned_unit_id = str(unit_id).strip()
    matches = [
        index
        for index, unit in enumerate(copied_units)
        if isinstance(unit, dict)
        and str(unit.get("id", "")).strip() == cleaned_unit_id
    ]
    if not matches:
        raise ValueError(f"Unknown graph unit '{cleaned_unit_id}'.")
    if len(matches) > 1:
        raise ValueError(f"Graph unit id '{cleaned_unit_id}' is duplicated.")

    cleaned_name = str(new_name).strip()
    if not cleaned_name:
        raise ValueError("Equipment name cannot be empty.")
    if len(cleaned_name) > 80:
        raise ValueError("Equipment name cannot exceed 80 characters.")
    for unit in copied_units:
        if not isinstance(unit, dict):
            continue
        existing_id = str(unit.get("id", "")).strip()
        raw_existing_name = unit.get("name")
        existing_name = (
            "" if raw_existing_name is None else str(raw_existing_name).strip()
        )
        if (
            existing_id != cleaned_unit_id
            and existing_name.casefold() == cleaned_name.casefold()
        ):
            raise ValueError(
                f"Equipment name '{cleaned_name}' is already in use."
            )
    if cleaned_name.casefold() in _normalized_name_keys(reserved_names):
        raise ValueError(f"Equipment name '{cleaned_name}' is already in use.")

    selected_unit = copied_units[matches[0]]
    validate_catalog_unit(selected_unit)
    selected_unit["name"] = cleaned_name
    validate_catalog_unit(selected_unit)
    return copied_units


def replace_inline_unit_type(
    units: list[Any],
    unit_id: str,
    replacement_type: str,
) -> list[dict[str, Any]]:
    """Replace one catalog unit while preserving its graph identity and routes.

    Replacement is intentionally limited to equipment with an identical port
    contract. The stable id, display name, and non-execution metadata are
    retained; type-specific parameters reset to the replacement defaults.
    Inputs are never mutated.
    """
    if not isinstance(units, list):
        raise ValueError("Graph units must be an array.")

    copied_units = copy.deepcopy(units)
    cleaned_unit_id = str(unit_id).strip()
    matches = [
        index
        for index, unit in enumerate(copied_units)
        if isinstance(unit, dict)
        and str(unit.get("id", "")).strip() == cleaned_unit_id
    ]
    if not matches:
        raise ValueError(f"Unknown graph unit '{cleaned_unit_id}'.")
    if len(matches) > 1:
        raise ValueError(f"Graph unit id '{cleaned_unit_id}' is duplicated.")

    selected_unit = copied_units[matches[0]]
    validate_catalog_unit(selected_unit)
    current_type = str(selected_unit.get("type", "")).strip().lower()
    cleaned_replacement_type = str(replacement_type).strip().lower()
    replacement = _INLINE_UNIT_CATALOG.get(cleaned_replacement_type)
    if replacement is None:
        raise ValueError(
            f"Unsupported inline unit type '{cleaned_replacement_type}'."
        )
    if cleaned_replacement_type == current_type:
        raise ValueError(
            f"Inline unit '{cleaned_unit_id}' is already type "
            f"'{current_type}'."
        )
    if replacement["ports"] != selected_unit["ports"]:
        raise ValueError(
            f"Cannot replace '{current_type}' with "
            f"'{cleaned_replacement_type}': material and energy ports differ."
        )

    selected_unit["type"] = cleaned_replacement_type
    selected_unit["ports"] = copy.deepcopy(replacement["ports"])
    selected_unit["params"] = copy.deepcopy(replacement["default_params"])
    validate_catalog_unit(selected_unit)
    return copied_units


def update_process_unit_properties(
    units: list[Any],
    unit_id: str,
    property_updates: Any,
) -> list[dict[str, Any]]:
    """Transactionally update metadata-backed properties for one process unit."""
    if not isinstance(units, list):
        raise ValueError("Graph units must be an array.")
    if not isinstance(property_updates, dict):
        raise ValueError("Process unit property updates must be an object.")

    copied_units = copy.deepcopy(units)
    cleaned_unit_id = str(unit_id).strip()
    matches = [
        index
        for index, unit in enumerate(copied_units)
        if isinstance(unit, dict)
        and str(unit.get("id", "")).strip() == cleaned_unit_id
    ]
    if not matches:
        raise ValueError(f"Unknown graph unit '{cleaned_unit_id}'.")
    if len(matches) > 1:
        raise ValueError(f"Graph unit id '{cleaned_unit_id}' is duplicated.")

    selected_unit = copied_units[matches[0]]
    if (
        str(selected_unit.get("type", "")).strip().lower() == "separator"
        and not property_updates
        and "params" not in selected_unit
    ):
        # A no-op must not rewrite imported parameterless separators merely
        # because the current editor can offer opt-in design properties.
        return copied_units
    current_params = selected_unit.get("params", {})
    if not isinstance(current_params, dict):
        raise ValueError(
            f"Process unit '{cleaned_unit_id}' params must be an object."
        )
    retained_params = copy.deepcopy(current_params)
    if (
        str(selected_unit.get("type", "")).strip().lower() == "splitter"
        and "split_factor" in property_updates
    ):
        retained_params.pop("split_factors", None)
    updated_params = {
        **retained_params,
        **copy.deepcopy(property_updates),
    }
    property_rows = process_unit_property_rows(
        selected_unit["type"],
        updated_params,
    )
    normalized_params = {
        row["key"]: row["value"] for row in property_rows
    }
    if normalized_params or "params" in selected_unit:
        selected_unit["params"] = normalized_params
    return copied_units


def update_inline_unit_properties(
    units: list[Any],
    unit_id: str,
    property_updates: Any,
) -> list[dict[str, Any]]:
    """Update one inline-palette unit while preserving its catalog contract."""
    updated_units = update_process_unit_properties(
        units,
        unit_id,
        property_updates,
    )
    cleaned_unit_id = str(unit_id).strip()
    selected_unit = next(
        unit
        for unit in updated_units
        if str(unit.get("id", "")).strip() == cleaned_unit_id
    )
    validate_catalog_unit(selected_unit)
    return updated_units


def remove_inline_unit(
    units: list[Any],
    connections: list[Any],
    unit_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove one added catalog unit and update its material path safely.

    An unconnected unit is deleted directly. A terminal unit with one incoming
    ``in`` connection is deleted with that connection. A unit with one incoming
    ``in`` and one outgoing ``out`` connection is removed by reconnecting the
    surrounding path. Energy links, branches, or any other references must be
    removed explicitly. Inputs are never mutated.
    """
    if not isinstance(units, list):
        raise ValueError("Graph units must be an array.")
    if not isinstance(connections, list):
        raise ValueError("Graph connections must be an array.")

    copied_units = copy.deepcopy(units)
    copied_connections = copy.deepcopy(connections)
    cleaned_unit_id = str(unit_id).strip()
    unit_matches = [
        index
        for index, unit in enumerate(copied_units)
        if isinstance(unit, dict)
        and str(unit.get("id", "")).strip() == cleaned_unit_id
    ]
    if not unit_matches:
        raise ValueError(f"Unknown graph unit '{cleaned_unit_id}'.")
    if len(unit_matches) > 1:
        raise ValueError(f"Graph unit id '{cleaned_unit_id}' is duplicated.")

    incoming_indices: list[int] = []
    outgoing_indices: list[int] = []
    unsupported_references: list[str] = []
    for index, connection in enumerate(copied_connections):
        if not isinstance(connection, dict):
            continue
        connection_id = str(connection.get("id", "")).strip() or str(index)
        connection_type = str(connection.get("type", "")).strip().lower()
        source = connection.get("source")
        target = connection.get("target")
        source_matches = (
            isinstance(source, dict)
            and str(source.get("kind", "")).strip() == "unit"
            and str(source.get("id", "")).strip() == cleaned_unit_id
        )
        target_matches = (
            isinstance(target, dict)
            and str(target.get("kind", "")).strip() == "unit"
            and str(target.get("id", "")).strip() == cleaned_unit_id
        )
        if not source_matches and not target_matches:
            continue

        if (
            connection_type == "material"
            and target_matches
            and str(target.get("port", "")).strip() == "in"
            and not source_matches
        ):
            incoming_indices.append(index)
        elif (
            connection_type == "material"
            and source_matches
            and str(source.get("port", "")).strip() == "out"
            and not target_matches
        ):
            outgoing_indices.append(index)
        else:
            unsupported_references.append(connection_id)

    if unsupported_references:
        raise ValueError(
            f"Graph unit '{cleaned_unit_id}' has unsupported connections: "
            + ", ".join(unsupported_references)
            + "."
        )
    if not incoming_indices and not outgoing_indices:
        copied_units.pop(unit_matches[0])
        return copied_units, copied_connections
    validate_catalog_unit(copied_units[unit_matches[0]])
    if len(incoming_indices) == 1 and not outgoing_indices:
        copied_connections.pop(incoming_indices[0])
        copied_units.pop(unit_matches[0])
        return copied_units, copied_connections
    if len(incoming_indices) != 1 or len(outgoing_indices) != 1:
        raise ValueError(
            f"Graph unit '{cleaned_unit_id}' requires no connections, one "
            "terminal incoming connection, or exactly one incoming and one "
            "outgoing material connection."
        )

    incoming_index = incoming_indices[0]
    outgoing_index = outgoing_indices[0]
    outgoing_target = copied_connections[outgoing_index].get("target")
    if not isinstance(outgoing_target, dict):
        raise ValueError(
            f"Graph unit '{cleaned_unit_id}' outgoing target must be an object."
        )
    if str(outgoing_target.get("id", "")).strip() == cleaned_unit_id:
        raise ValueError(
            f"Graph unit '{cleaned_unit_id}' cannot reconnect to itself."
        )

    copied_connections[incoming_index]["target"] = copy.deepcopy(
        outgoing_target
    )
    copied_connections.pop(outgoing_index)
    copied_units.pop(unit_matches[0])
    return copied_units, copied_connections


def create_graph_draft(
    units: list[Any],
    connections: list[Any],
    inlets: list[Any] | None = None,
) -> dict[str, Any]:
    """Create an isolated, versioned draft from case graph arrays."""
    if not isinstance(units, list):
        raise ValueError("Graph draft units must be an array.")
    if not isinstance(connections, list):
        raise ValueError("Graph draft connections must be an array.")

    copied_units = copy.deepcopy(units)
    copied_connections = copy.deepcopy(connections)
    copied_inlets = copy.deepcopy(inlets)
    inlet_ids: set[str] = set()
    if copied_inlets is not None:
        if not isinstance(copied_inlets, list):
            raise ValueError("Graph draft inlets must be an array.")
        for index, inlet in enumerate(copied_inlets):
            if not isinstance(inlet, dict):
                raise ValueError(
                    f"Graph draft inlet {index} must be an object."
                )
            inlet_id = str(inlet.get("id", "")).strip()
            if not inlet_id:
                raise ValueError(
                    f"Graph draft inlet {index} requires an id."
                )
            if inlet_id in inlet_ids:
                raise ValueError(
                    f"Graph draft inlet id '{inlet_id}' is duplicated."
                )
            inlet_ids.add(inlet_id)

    unit_ids: set[str] = set()
    for index, unit in enumerate(copied_units):
        if not isinstance(unit, dict):
            raise ValueError(f"Graph draft unit {index} must be an object.")
        unit_id = str(unit.get("id", "")).strip()
        if not unit_id:
            raise ValueError(f"Graph draft unit {index} requires an id.")
        if unit_id in unit_ids:
            raise ValueError(f"Graph draft unit id '{unit_id}' is duplicated.")
        unit_ids.add(unit_id)

    conflicting_object_ids = inlet_ids.intersection(unit_ids)
    if conflicting_object_ids:
        conflicting_id = sorted(conflicting_object_ids)[0]
        raise ValueError(
            f"Graph object id '{conflicting_id}' is duplicated between an "
            "inlet and a unit; it is both an inlet and a unit."
        )

    connection_ids: set[str] = set()
    for index, connection in enumerate(copied_connections):
        if not isinstance(connection, dict):
            raise ValueError(
                f"Graph draft connection {index} must be an object."
            )
        connection_id = str(connection.get("id", "")).strip()
        if not connection_id:
            raise ValueError(f"Graph draft connection {index} requires an id.")
        if connection_id in connection_ids:
            raise ValueError(
                f"Graph draft connection id '{connection_id}' is duplicated."
            )
        connection_ids.add(connection_id)
        if str(connection.get("type", "")).strip() not in (
            "material",
            "energy",
        ):
            raise ValueError(
                f"Graph draft connection '{connection_id}' has invalid type."
            )
        for endpoint_name in ("source", "target"):
            endpoint = connection.get(endpoint_name)
            if not isinstance(endpoint, dict):
                raise ValueError(
                    f"Graph draft connection '{connection_id}' "
                    f"{endpoint_name} must be an object."
                )
            for field_name in ("kind", "id", "port"):
                if not str(endpoint.get(field_name, "")).strip():
                    raise ValueError(
                        f"Graph draft connection '{connection_id}' "
                        f"{endpoint_name} requires {field_name}."
                    )

    draft = {
        "schema_version": GRAPH_DRAFT_SCHEMA_VERSION,
        "units": copied_units,
        "connections": copied_connections,
    }
    if copied_inlets is not None:
        draft["inlets"] = copied_inlets
    return draft


def apply_graph_draft(
    case_spec: dict[str, Any],
    draft: Any,
) -> dict[str, Any]:
    """Apply a validated draft to an isolated copy of a complete case."""
    if not isinstance(case_spec, dict):
        raise ValueError("Case specification must be an object.")
    if not isinstance(draft, dict):
        raise ValueError("Graph draft must be an object.")
    if draft.get("schema_version") != GRAPH_DRAFT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported graph draft schema version. Expected version 1."
        )
    if "inlets" in draft and draft["inlets"] is None:
        raise ValueError("Graph draft inlets must be an array.")
    retained_or_draft_inlets = (
        draft.get("inlets")
        if "inlets" in draft
        else case_spec.get("inlets")
    )
    validated = create_graph_draft(
        draft.get("units"),
        draft.get("connections"),
        retained_or_draft_inlets,
    )
    updated_case = copy.deepcopy(case_spec)
    updated_case["units"] = validated["units"]
    updated_case["connections"] = validated["connections"]
    if "inlets" in validated:
        updated_case["inlets"] = validated["inlets"]
    return updated_case


def _validated_graph_history(history: Any) -> dict[str, Any]:
    """Return an isolated validated graph-history timeline."""
    if not isinstance(history, dict):
        raise ValueError("Graph history must be an object.")
    if history.get("schema_version") != GRAPH_HISTORY_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported graph history schema version. Expected version 1."
        )

    entries = history.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Graph history entries must be a non-empty array.")
    validated_entries: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Graph history entry {index} must be an object.")
        if entry.get("schema_version") != GRAPH_DRAFT_SCHEMA_VERSION:
            raise ValueError(
                f"Graph history entry {index} has an unsupported draft version."
            )
        validated_entries.append(
            create_graph_draft(
                entry.get("units"),
                entry.get("connections"),
                entry.get("inlets") if "inlets" in entry else None,
            )
        )

    cursor = history.get("cursor")
    if isinstance(cursor, bool) or not isinstance(cursor, int):
        raise ValueError("Graph history cursor must be an integer.")
    if cursor < 0 or cursor >= len(validated_entries):
        raise ValueError("Graph history cursor is outside the entry range.")
    return {
        "schema_version": GRAPH_HISTORY_SCHEMA_VERSION,
        "entries": validated_entries,
        "cursor": cursor,
    }


def create_graph_history(
    units: list[Any],
    connections: list[Any],
    inlets: list[Any] | None = None,
) -> dict[str, Any]:
    """Create a history timeline containing one isolated graph revision."""
    return {
        "schema_version": GRAPH_HISTORY_SCHEMA_VERSION,
        "entries": [create_graph_draft(units, connections, inlets)],
        "cursor": 0,
    }


def record_graph_history(
    history: Any,
    units: list[Any],
    connections: list[Any],
    inlets: list[Any] | None = None,
    max_entries: int = MAX_GRAPH_HISTORY_ENTRIES,
) -> dict[str, Any]:
    """Append one graph revision and discard any abandoned redo branch."""
    if (
        isinstance(max_entries, bool)
        or not isinstance(max_entries, int)
        or max_entries < 2
    ):
        raise ValueError("Graph history limit must be an integer of at least 2.")

    updated = _validated_graph_history(history)
    candidate = create_graph_draft(units, connections, inlets)
    if updated["entries"][updated["cursor"]] == candidate:
        return updated

    entries = updated["entries"][: updated["cursor"] + 1]
    entries.append(candidate)
    if len(entries) > max_entries:
        entries = entries[-max_entries:]
    return {
        "schema_version": GRAPH_HISTORY_SCHEMA_VERSION,
        "entries": entries,
        "cursor": len(entries) - 1,
    }


def graph_history_status(history: Any) -> dict[str, Any]:
    """Return presentation-ready position and navigation availability."""
    validated = _validated_graph_history(history)
    cursor = validated["cursor"]
    total = len(validated["entries"])
    return {
        "position": cursor + 1,
        "total": total,
        "can_undo": cursor > 0,
        "can_redo": cursor < total - 1,
    }


def undo_graph_history(
    history: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Move back one revision and return isolated history and draft copies."""
    updated = _validated_graph_history(history)
    if updated["cursor"] == 0:
        raise ValueError("Graph history has no earlier revision to undo.")
    updated["cursor"] -= 1
    return updated, copy.deepcopy(updated["entries"][updated["cursor"]])


def redo_graph_history(
    history: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Move forward one revision and return isolated history and draft copies."""
    updated = _validated_graph_history(history)
    if updated["cursor"] >= len(updated["entries"]) - 1:
        raise ValueError("Graph history has no later revision to redo.")
    updated["cursor"] += 1
    return updated, copy.deepcopy(updated["entries"][updated["cursor"]])


def _dot_text(value: Any) -> str:
    """Quote untrusted graph text for a Graphviz string attribute."""
    return json.dumps(str(value), ensure_ascii=False)


def build_graph_draft_dot(
    inlets: list[Any],
    units: list[Any],
    connections: list[Any],
) -> str:
    """Render a validated draft graph as deterministic auto-layout DOT.

    Material connections are solid blue paths, energy connections are dashed
    amber paths, and each unconnected material output is shown as an explicit
    product boundary. Internal DOT node ids never contain user-provided text.
    """
    if not isinstance(inlets, list):
        raise ValueError("Graph preview inlets must be an array.")
    validated_draft = create_graph_draft(units, connections, inlets)
    validated_inlets = validated_draft["inlets"]
    validated_units = validated_draft["units"]
    validated_connections = validated_draft["connections"]

    node_ids: dict[tuple[str, str], str] = {}
    inlet_records: list[tuple[str, dict[str, Any], str]] = []
    for index, inlet in enumerate(validated_inlets):
        inlet_id = str(inlet["id"]).strip()
        key = ("inlet", inlet_id)
        if key in node_ids:
            raise ValueError(f"Graph preview inlet id '{inlet_id}' is duplicated.")
        dot_id = f"inlet_{index}"
        node_ids[key] = dot_id
        inlet_records.append((inlet_id, inlet, dot_id))

    unit_records: list[tuple[str, dict[str, Any], str]] = []
    units_by_id: dict[str, dict[str, Any]] = {}
    for index, unit in enumerate(validated_units):
        unit_id = str(unit["id"]).strip()
        if not isinstance(unit.get("ports"), dict):
            raise ValueError(f"Graph preview unit '{unit_id}' requires ports.")
        if ("inlet", unit_id) in node_ids:
            raise ValueError(
                f"Graph preview id '{unit_id}' is both an inlet and a unit."
            )
        key = ("unit", unit_id)
        dot_id = f"unit_{index}"
        node_ids[key] = dot_id
        units_by_id[unit_id] = unit
        unit_records.append((unit_id, unit, dot_id))

    connected_outputs: set[tuple[str, str]] = set()
    rendered_connections: list[
        tuple[dict[str, Any], str, str, str, str]
    ] = []
    for connection in sorted(
        validated_connections,
        key=lambda item: str(item["id"]),
    ):
        connection_id = str(connection["id"]).strip()
        source = connection["source"]
        target = connection["target"]
        source_kind = str(source["kind"]).strip()
        source_id = str(source["id"]).strip()
        source_port = str(source["port"]).strip()
        target_kind = str(target["kind"]).strip()
        target_id = str(target["id"]).strip()
        target_port = str(target["port"]).strip()
        source_dot_id = node_ids.get((source_kind, source_id))
        target_dot_id = node_ids.get((target_kind, target_id))
        if source_dot_id is None:
            raise ValueError(
                f"Graph preview connection '{connection_id}' has unknown source "
                f"{source_kind} '{source_id}'."
            )
        if target_dot_id is None:
            raise ValueError(
                f"Graph preview connection '{connection_id}' has unknown target "
                f"{target_kind} '{target_id}'."
            )
        connection_type = str(connection["type"])
        if source_kind == "inlet":
            if connection_type != "material" or source_port != "out":
                raise ValueError(
                    f"Graph preview inlet '{source_id}' exposes material port 'out'."
                )
        else:
            output_ports = units_by_id[source_id]["ports"].get(
                f"{connection_type}_out",
                [],
            )
            if source_port not in output_ports:
                raise ValueError(
                    f"Graph preview connection '{connection_id}' uses undeclared "
                    f"{connection_type}_out port '{source_port}'."
                )
        if target_kind != "unit":
            raise ValueError(
                f"Graph preview connection '{connection_id}' target must be a unit."
            )
        input_ports = units_by_id[target_id]["ports"].get(
            f"{connection_type}_in",
            [],
        )
        if target_port not in input_ports:
            raise ValueError(
                f"Graph preview connection '{connection_id}' uses undeclared "
                f"{connection_type}_in port '{target_port}'."
            )
        if connection_type == "material":
            source_unit_type = (
                units_by_id[source_id].get("type")
                if source_kind == "unit"
                else None
            )
            connected_outputs.add(
                (
                    source_id,
                    canonical_material_output_port(
                        source_port,
                        source_unit_type,
                    ),
                )
            )
        rendered_connections.append(
            (
                connection,
                source_dot_id,
                target_dot_id,
                source_port,
                target_port,
            )
        )

    lines = [
        "digraph flowsheet {",
        '  graph [rankdir="LR", bgcolor="transparent", pad="0.2", '
        'nodesep="0.45", ranksep="0.75", splines="ortho"];',
        '  node [shape="box", style="rounded,filled", fontname="Helvetica", '
        'fontsize="10", margin="0.12,0.08", penwidth="1.4"];',
        '  edge [fontname="Helvetica", fontsize="8", arrowsize="0.7", '
        'penwidth="1.4"];',
    ]
    for inlet_id, inlet, dot_id in inlet_records:
        inlet_name = str(inlet.get("name", inlet_id)).strip() or inlet_id
        label = f"{inlet_name}\nINLET"
        lines.append(
            f"  {dot_id} [label={_dot_text(label)}, shape=\"oval\", "
            'fillcolor="#dcfce7", color="#16a34a"];'
        )

    for unit_id, unit, dot_id in unit_records:
        unit_name = str(unit.get("name", unit_id)).strip() or unit_id
        unit_type = str(unit.get("type", "unit")).strip().lower() or "unit"
        fill_color, line_color = _GRAPH_NODE_STYLES.get(
            unit_type,
            ("#f8fafc", "#64748b"),
        )
        label = f"{unit_name}\n{unit_type.upper()}"
        lines.append(
            f"  {dot_id} [label={_dot_text(label)}, "
            f"fillcolor=\"{fill_color}\", color=\"{line_color}\"];"
        )

    for (
        connection,
        source_dot_id,
        target_dot_id,
        source_port,
        target_port,
    ) in rendered_connections:
        connection_type = str(connection["type"])
        if connection_type == "energy":
            edge_style = 'color="#d97706", fontcolor="#92400e", style="dashed"'
        else:
            edge_style = 'color="#2563eb", fontcolor="#1e40af"'
        if connection_type == "material":
            edge_label = (
                f"{material_connection_name(connection)}\n"
                f"{source_port} \u2192 {target_port}"
            )
        else:
            edge_label = f"{source_port} \u2192 {target_port}"
        lines.append(
            f"  {source_dot_id} -> {target_dot_id} "
            f"[label={_dot_text(edge_label)}, {edge_style}];"
        )

    product_index = 0
    for unit_id, unit, dot_id in unit_records:
        ports = unit.get("ports")
        if not isinstance(ports, dict):
            raise ValueError(f"Graph preview unit '{unit_id}' requires ports.")
        output_ports = ports.get("material_out", [])
        if not isinstance(output_ports, list):
            raise ValueError(
                f"Graph preview unit '{unit_id}' material_out must be an array."
            )
        for port in output_ports:
            port_name = str(port).strip()
            if not port_name:
                raise ValueError(
                    f"Graph preview unit '{unit_id}' has an empty output port."
                )
            if (
                unit_id,
                canonical_material_output_port(
                    port_name,
                    unit.get("type"),
                ),
            ) in connected_outputs:
                continue
            product_dot_id = f"product_{product_index}"
            product_index += 1
            product_label = f"{unit_id}:{port_name}\nPRODUCT"
            lines.append(
                f"  {product_dot_id} [label={_dot_text(product_label)}, "
                'shape="oval", fillcolor="#f1f5f9", color="#64748b"];'
            )
            lines.append(
                f"  {dot_id} -> {product_dot_id} "
                f"[label={_dot_text(port_name)}, color=\"#2563eb\", "
                'fontcolor="#1e40af"];'
            )
    lines.append("}")
    return "\n".join(lines)


def material_connection_rows(
    connections: list[Any],
) -> list[dict[str, str]]:
    """Return deterministic palette labels for editable material paths."""
    if not isinstance(connections, list):
        raise ValueError("Graph connections must be an array.")

    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for index, connection in enumerate(connections):
        if not isinstance(connection, dict):
            raise ValueError(f"Graph connection {index} must be an object.")
        connection_id = str(connection.get("id", "")).strip()
        if not connection_id:
            raise ValueError(f"Graph connection {index} requires an id.")
        if connection_id in seen_ids:
            raise ValueError(f"Graph connection id '{connection_id}' is duplicated.")
        seen_ids.add(connection_id)
        if str(connection.get("type", "")).strip().lower() != "material":
            continue
        stream_name = material_connection_name(connection)
        stream_name_key = stream_name.casefold()
        if stream_name_key in seen_names:
            raise ValueError(
                f"Material stream name '{stream_name}' is duplicated."
            )
        seen_names.add(stream_name_key)

        source = connection.get("source")
        target = connection.get("target")
        if not isinstance(source, dict) or not isinstance(target, dict):
            raise ValueError(
                f"Connection '{connection_id}' requires source and target."
            )
        source_id = str(source.get("id", "")).strip()
        source_port = str(source.get("port", "")).strip()
        target_id = str(target.get("id", "")).strip()
        target_port = str(target.get("port", "")).strip()
        if not all((source_id, source_port, target_id, target_port)):
            raise ValueError(
                f"Connection '{connection_id}' has an incomplete material route."
            )
        rows.append(
            {
                "id": connection_id,
                "name": stream_name,
                "label": (
                    f"{stream_name} · {source_id}:{source_port} → "
                    f"{target_id}:{target_port}"
                ),
            }
        )
    return rows
