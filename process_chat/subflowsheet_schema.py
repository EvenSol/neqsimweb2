"""Validation helpers for hierarchical Process Flowsheet Studio graphs.

Subflowsheets are execution-neutral graph groups.  Their boundary ports expose
the exact unit port used by a connection that crosses the group boundary.  The
flat unit and connection graph remains the authoritative execution model, so
schema-v3 cases and ProcessBuilder integrations remain backward compatible.
"""

from __future__ import annotations

from typing import Any

from .graph_schema import canonical_material_output_port


SUPPORTED_BOUNDARY_TYPES = ("material", "energy")
SUPPORTED_BOUNDARY_DIRECTIONS = ("inlet", "outlet")


def _required_text(value: Any, label: str) -> str:
    """Return a trimmed non-empty string."""
    if value is None:
        raise ValueError(f"{label} is required.")
    result = str(value).strip()
    if not result:
        raise ValueError(f"{label} is required.")
    return result


def _indexed_objects(records: list[Any], label: str) -> dict[str, dict[str, Any]]:
    """Index graph records by id while rejecting malformed duplicates."""
    if not isinstance(records, list):
        raise ValueError(f"{label} must be an array.")
    indexed: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{label}[{index}] must be an object.")
        record_id = _required_text(record.get("id"), f"{label}[{index}].id")
        if record_id in indexed:
            raise ValueError(f"{label} id '{record_id}' is duplicated.")
        indexed[record_id] = record
    return indexed


def _declared_unit_port(
    unit: dict[str, Any],
    connection_type: str,
    direction: str,
    port: str,
) -> bool:
    """Return whether a boundary endpoint names one declared unit port."""
    ports = unit.get("ports")
    if not isinstance(ports, dict):
        return False
    key = f"{connection_type}_{'in' if direction == 'inlet' else 'out'}"
    declared = ports.get(key)
    if not isinstance(declared, list):
        return False
    if connection_type == "material" and direction == "outlet":
        canonical_port = canonical_material_output_port(port, unit.get("type"))
        return canonical_port in {
            canonical_material_output_port(item, unit.get("type"))
            for item in declared
        }
    return port in {str(item).strip() for item in declared}


def validate_subflowsheets(
    subflowsheets: list[Any],
    units: list[Any],
    connections: list[Any],
) -> None:
    """Validate explicit subflowsheet membership and crossing boundary ports.

    A unit may belong to at most one subflowsheet in schema v4.  Every material
    or energy connection that crosses a group boundary must use an explicitly
    declared boundary port.  Unconnected material outlets may also be declared
    as product boundary ports.
    """
    indexed_units = _indexed_objects(units, "units")
    indexed_connections = _indexed_objects(connections, "connections")
    if not isinstance(subflowsheets, list):
        raise ValueError("subflowsheets must be an array.")

    subflowsheet_ids: set[str] = set()
    subflowsheet_names: set[str] = set()
    unit_owner: dict[str, str] = {}
    boundary_by_group: dict[
        str,
        set[tuple[str, str, str, str]],
    ] = {}

    for group_index, group in enumerate(subflowsheets):
        if not isinstance(group, dict):
            raise ValueError(f"subflowsheets[{group_index}] must be an object.")
        group_id = _required_text(
            group.get("id"),
            f"subflowsheets[{group_index}].id",
        )
        if group_id in subflowsheet_ids:
            raise ValueError(f"Subflowsheet id '{group_id}' is duplicated.")
        subflowsheet_ids.add(group_id)
        group_name = _required_text(
            group.get("name"),
            f"Subflowsheet '{group_id}' name",
        )
        group_name_key = group_name.casefold()
        if group_name_key in subflowsheet_names:
            raise ValueError(f"Subflowsheet name '{group_name}' is duplicated.")
        subflowsheet_names.add(group_name_key)

        raw_unit_ids = group.get("unit_ids")
        if not isinstance(raw_unit_ids, list) or not raw_unit_ids:
            raise ValueError(
                f"Subflowsheet '{group_id}' requires a non-empty unit_ids array."
            )
        member_ids: set[str] = set()
        for unit_index, raw_unit_id in enumerate(raw_unit_ids):
            unit_id = _required_text(
                raw_unit_id,
                f"Subflowsheet '{group_id}' unit_ids[{unit_index}]",
            )
            if unit_id in member_ids:
                raise ValueError(
                    f"Subflowsheet '{group_id}' repeats unit '{unit_id}'."
                )
            if unit_id not in indexed_units:
                raise ValueError(
                    f"Subflowsheet '{group_id}' references unknown unit "
                    f"'{unit_id}'."
                )
            previous_owner = unit_owner.get(unit_id)
            if previous_owner is not None:
                raise ValueError(
                    f"Unit '{unit_id}' belongs to both subflowsheet "
                    f"'{previous_owner}' and '{group_id}'."
                )
            member_ids.add(unit_id)
            unit_owner[unit_id] = group_id

        raw_boundaries = group.get("boundary_ports")
        if not isinstance(raw_boundaries, list) or not raw_boundaries:
            raise ValueError(
                f"Subflowsheet '{group_id}' requires boundary_ports."
            )
        boundary_ids: set[str] = set()
        boundary_names: set[str] = set()
        boundary_endpoints: set[tuple[str, str, str, str]] = set()
        for boundary_index, boundary in enumerate(raw_boundaries):
            if not isinstance(boundary, dict):
                raise ValueError(
                    f"Subflowsheet '{group_id}' boundary_ports"
                    f"[{boundary_index}] must be an object."
                )
            boundary_id = _required_text(
                boundary.get("id"),
                f"Subflowsheet '{group_id}' boundary port id",
            )
            if boundary_id in boundary_ids:
                raise ValueError(
                    f"Subflowsheet '{group_id}' boundary port id "
                    f"'{boundary_id}' is duplicated."
                )
            boundary_ids.add(boundary_id)
            boundary_name = _required_text(
                boundary.get("name"),
                f"Subflowsheet '{group_id}' boundary '{boundary_id}' name",
            )
            boundary_name_key = boundary_name.casefold()
            if boundary_name_key in boundary_names:
                raise ValueError(
                    f"Subflowsheet '{group_id}' boundary port name "
                    f"'{boundary_name}' is duplicated."
                )
            boundary_names.add(boundary_name_key)

            connection_type = str(boundary.get("type", "")).strip().lower()
            if connection_type not in SUPPORTED_BOUNDARY_TYPES:
                raise ValueError(
                    f"Subflowsheet '{group_id}' boundary '{boundary_id}' type "
                    "must be material or energy."
                )
            direction = str(boundary.get("direction", "")).strip().lower()
            if direction not in SUPPORTED_BOUNDARY_DIRECTIONS:
                raise ValueError(
                    f"Subflowsheet '{group_id}' boundary '{boundary_id}' "
                    "direction must be inlet or outlet."
                )
            endpoint = boundary.get("endpoint")
            if not isinstance(endpoint, dict):
                raise ValueError(
                    f"Subflowsheet '{group_id}' boundary '{boundary_id}' "
                    "requires an endpoint object."
                )
            if str(endpoint.get("kind", "")).strip().lower() != "unit":
                raise ValueError(
                    f"Subflowsheet '{group_id}' boundary '{boundary_id}' "
                    "must expose a member unit port."
                )
            unit_id = _required_text(
                endpoint.get("id"),
                f"Subflowsheet '{group_id}' boundary '{boundary_id}' unit id",
            )
            port = _required_text(
                endpoint.get("port"),
                f"Subflowsheet '{group_id}' boundary '{boundary_id}' port",
            )
            if unit_id not in member_ids:
                raise ValueError(
                    f"Subflowsheet '{group_id}' boundary '{boundary_id}' "
                    f"references non-member unit '{unit_id}'."
                )
            if not _declared_unit_port(
                indexed_units[unit_id],
                connection_type,
                direction,
                port,
            ):
                port_key = (
                    f"{connection_type}_"
                    f"{'in' if direction == 'inlet' else 'out'}"
                )
                raise ValueError(
                    f"Subflowsheet '{group_id}' boundary '{boundary_id}' uses "
                    f"undeclared {port_key} port '{port}' on unit '{unit_id}'."
                )
            endpoint_key = (connection_type, direction, unit_id, port)
            if endpoint_key in boundary_endpoints:
                raise ValueError(
                    f"Subflowsheet '{group_id}' exposes unit port "
                    f"'{unit_id}:{port}' more than once."
                )
            boundary_endpoints.add(endpoint_key)
        boundary_by_group[group_id] = boundary_endpoints

    crossing_endpoints: dict[str, set[tuple[str, str, str, str]]] = {
        group_id: set() for group_id in subflowsheet_ids
    }
    connected_material_outputs: set[tuple[str, str]] = set()
    for connection_id, connection in indexed_connections.items():
        connection_type = str(connection.get("type", "")).strip().lower()
        if connection_type not in SUPPORTED_BOUNDARY_TYPES:
            continue
        source = connection.get("source")
        target = connection.get("target")
        if not isinstance(source, dict) or not isinstance(target, dict):
            continue
        source_id = str(source.get("id", "")).strip()
        target_id = str(target.get("id", "")).strip()
        source_group = unit_owner.get(source_id)
        target_group = unit_owner.get(target_id)
        if connection_type == "material" and str(
            source.get("kind", "")
        ).strip().lower() == "unit":
            connected_material_outputs.add(
                (
                    source_id,
                    canonical_material_output_port(
                        source.get("port", ""),
                        indexed_units.get(source_id, {}).get("type"),
                    ),
                )
            )
        if source_group == target_group:
            continue
        if source_group is not None:
            crossing_endpoints[source_group].add(
                (
                    connection_type,
                    "outlet",
                    source_id,
                    str(source.get("port", "")).strip(),
                )
            )
        if target_group is not None:
            crossing_endpoints[target_group].add(
                (
                    connection_type,
                    "inlet",
                    target_id,
                    str(target.get("port", "")).strip(),
                )
            )

    for group_id, endpoints in boundary_by_group.items():
        for endpoint in endpoints:
            connection_type, direction, unit_id, port = endpoint
            if endpoint in crossing_endpoints[group_id]:
                continue
            if connection_type == "material" and direction == "outlet":
                canonical_port = canonical_material_output_port(
                    port,
                    indexed_units[unit_id].get("type"),
                )
                if (unit_id, canonical_port) not in connected_material_outputs:
                    continue
            raise ValueError(
                f"Subflowsheet '{group_id}' boundary endpoint "
                f"'{unit_id}:{port}' is not an active graph boundary."
            )

        missing = crossing_endpoints[group_id].difference(endpoints)
        if missing:
            connection_type, direction, unit_id, port = sorted(missing)[0]
            raise ValueError(
                f"Subflowsheet '{group_id}' must declare its {connection_type} "
                f"{direction} boundary at '{unit_id}:{port}'."
            )


def subflowsheet_membership(
    subflowsheets: list[Any],
) -> dict[str, str]:
    """Return unit-id to subflowsheet-id membership after basic validation."""
    if not isinstance(subflowsheets, list):
        raise ValueError("subflowsheets must be an array.")
    membership: dict[str, str] = {}
    for group_index, group in enumerate(subflowsheets):
        if not isinstance(group, dict):
            raise ValueError(f"subflowsheets[{group_index}] must be an object.")
        group_id = _required_text(
            group.get("id"),
            f"subflowsheets[{group_index}].id",
        )
        unit_ids = group.get("unit_ids")
        if not isinstance(unit_ids, list):
            raise ValueError(
                f"Subflowsheet '{group_id}' unit_ids must be an array."
            )
        for raw_unit_id in unit_ids:
            unit_id = _required_text(
                raw_unit_id,
                f"Subflowsheet '{group_id}' unit id",
            )
            if unit_id in membership:
                raise ValueError(
                    f"Unit '{unit_id}' belongs to multiple subflowsheets."
                )
            membership[unit_id] = group_id
    return membership
