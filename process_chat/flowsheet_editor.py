"""Pure schema helpers for editing Process Flowsheet Studio graphs."""

from __future__ import annotations

import copy
import re
from typing import Any


GRAPH_DRAFT_SCHEMA_VERSION = 1
GRAPH_HISTORY_SCHEMA_VERSION = 1
MAX_GRAPH_HISTORY_ENTRIES = 50


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


def rename_inline_unit(
    units: list[Any],
    unit_id: str,
    new_name: str,
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
        existing_name = str(unit.get("name", "")).strip()
        if (
            existing_id != cleaned_unit_id
            and existing_name.casefold() == cleaned_name.casefold()
        ):
            raise ValueError(
                f"Equipment name '{cleaned_name}' is already in use."
            )

    selected_unit = copied_units[matches[0]]
    validate_catalog_unit(selected_unit)
    selected_unit["name"] = cleaned_name
    validate_catalog_unit(selected_unit)
    return copied_units


def remove_inline_unit(
    units: list[Any],
    connections: list[Any],
    unit_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove one inline catalog unit and reconnect its material path.

    Removal is deliberately limited to a unit with exactly one ``in`` material
    connection and one ``out`` material connection. Energy links, branches, or
    any other references must be removed explicitly in a later graph editor.
    Inputs are never mutated.
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
            f"Inline unit '{cleaned_unit_id}' has unsupported connections: "
            + ", ".join(unsupported_references)
            + "."
        )
    if len(incoming_indices) != 1 or len(outgoing_indices) != 1:
        raise ValueError(
            f"Inline unit '{cleaned_unit_id}' requires exactly one incoming "
            "and one outgoing material connection."
        )

    incoming_index = incoming_indices[0]
    outgoing_index = outgoing_indices[0]
    outgoing_target = copied_connections[outgoing_index].get("target")
    if not isinstance(outgoing_target, dict):
        raise ValueError(
            f"Inline unit '{cleaned_unit_id}' outgoing target must be an object."
        )
    if str(outgoing_target.get("id", "")).strip() == cleaned_unit_id:
        raise ValueError(
            f"Inline unit '{cleaned_unit_id}' cannot reconnect to itself."
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
) -> dict[str, Any]:
    """Create a history timeline containing one isolated graph revision."""
    return {
        "schema_version": GRAPH_HISTORY_SCHEMA_VERSION,
        "entries": [create_graph_draft(units, connections)],
        "cursor": 0,
    }


def record_graph_history(
    history: Any,
    units: list[Any],
    connections: list[Any],
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
    candidate = create_graph_draft(units, connections)
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
