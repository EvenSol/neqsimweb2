"""Pure schema helpers for editing Process Flowsheet Studio graphs."""

from __future__ import annotations

import copy
import json
import math
import re
from typing import Any


GRAPH_DRAFT_SCHEMA_VERSION = 1
GRAPH_HISTORY_SCHEMA_VERSION = 1
MAX_GRAPH_HISTORY_ENTRIES = 50


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
        "label": label,
        "unit": unit,
        "minimum": minimum,
        "maximum": maximum,
        "step": step,
        "format": display_format,
    }


_GRAPH_NODE_STYLES = {
    "compressor": ("#dbeafe", "#2563eb"),
    "cooler": ("#cffafe", "#0891b2"),
    "heater": ("#ffedd5", "#ea580c"),
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
    "valve": {
        "label": "Valve",
        "category": "Pressure change",
        "description": "Reduce pressure through a throttling valve.",
        "ports": {
            "material_in": ["in"],
            "material_out": ["out"],
        },
        "default_params": {
            "outlet_pressure_bara": 40.0,
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
        "default_params": {},
        "properties": {},
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
        "label": "Splitter (equal default)",
        "category": "Flow routing",
        "description": (
            "Divide one material stream between two branch outlets; "
            "new nodes default to an equal allocation."
        ),
        "ports": {
            "material_in": ["in"],
            "material_out": ["out_0", "out_1"],
        },
        "default_params": {},
        "properties": {},
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
        selected_params = params
    else:
        raise ValueError("Process unit params must be an object.")

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
    if unit.get("ports") != definition["ports"]:
        raise ValueError(
            f"Inline unit '{unit_id}' ports do not match the '{unit_type}' catalog."
        )
    if not isinstance(unit.get("params"), dict):
        raise ValueError(f"Inline unit '{unit_id}' params must be an object.")
    inline_unit_property_rows(unit_type, unit["params"])


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


def _unique_connection_id(stem: str, existing_ids: set[str]) -> str:
    """Return a stable connection id without overwriting an existing edge."""
    connection_id = _slugify(stem)
    suffix = 2
    while connection_id in existing_ids:
        connection_id = f"{_slugify(stem)}-{suffix}"
        suffix += 1
    return connection_id


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
    existing_ids = {
        str(connection["id"]).strip()
        for connection in copied_connections
    }
    connection_id = _unique_connection_id(
        (
            f"{cleaned_type}-{source_key[2]}-{source_key[3]}-to-"
            f"{target_key[2]}-{target_key[3]}"
        ),
        existing_ids,
    )
    copied_connections.append(
        {
            "id": connection_id,
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
    new_unit = create_inline_unit_spec(
        unit_type,
        unit_name,
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
    existing_connection_ids = {
        str(connection.get("id", "")).strip()
        for connection in copied_connections
        if isinstance(connection, dict)
    }
    downstream_connection_id = _unique_connection_id(
        f"{new_unit['id']}-to-{target_id}",
        existing_connection_ids,
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
    current_params = selected_unit.get("params", {})
    if not isinstance(current_params, dict):
        raise ValueError(
            f"Process unit '{cleaned_unit_id}' params must be an object."
        )
    updated_params = {
        **current_params,
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
    validate_catalog_unit(copied_units[unit_matches[0]])

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
