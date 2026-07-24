"""Pure schema helpers for editing Process Flowsheet Studio graphs."""

from __future__ import annotations

import copy
import re
from typing import Any


GRAPH_DRAFT_SCHEMA_VERSION = 1


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
    },
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


def _slugify(value: str) -> str:
    """Convert a user-facing name to a stable graph-id stem."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or "unit"


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


def insert_inline_unit_on_connection(
    units: list[Any],
    connections: list[Any],
    connection_id: str,
    unit_type: str,
    unit_name: str,
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


def create_graph_draft(
    units: list[Any],
    connections: list[Any],
) -> dict[str, Any]:
    """Create an isolated, versioned draft from case graph arrays."""
    if not isinstance(units, list):
        raise ValueError("Graph draft units must be an array.")
    if not isinstance(connections, list):
        raise ValueError("Graph draft connections must be an array.")

    copied_units = copy.deepcopy(units)
    copied_connections = copy.deepcopy(connections)
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

    return {
        "schema_version": GRAPH_DRAFT_SCHEMA_VERSION,
        "units": copied_units,
        "connections": copied_connections,
    }


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
    validated = create_graph_draft(
        draft.get("units"),
        draft.get("connections"),
    )
    updated_case = copy.deepcopy(case_spec)
    updated_case["units"] = validated["units"]
    updated_case["connections"] = validated["connections"]
    return updated_case


def material_connection_rows(
    connections: list[Any],
) -> list[dict[str, str]]:
    """Return deterministic palette labels for editable material paths."""
    if not isinstance(connections, list):
        raise ValueError("Graph connections must be an array.")

    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
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
                "label": (
                    f"{source_id}:{source_port} → "
                    f"{target_id}:{target_port}"
                ),
            }
        )
    return rows
