"""Shared validation helpers for Process Flowsheet Studio graph schemas."""

from __future__ import annotations

import re
from typing import Any


_SEPARATOR_UNIT_TYPES = {
    "separator",
    "two_phase_separator",
    "three_phase_separator",
    "gas_scrubber",
    "membrane_separator",
}
_TWO_SIDED_HEAT_EXCHANGER_UNIT_TYPES = {"heat_exchanger"}


def canonical_material_output_port(
    raw_port: Any,
    unit_type: Any = None,
) -> str:
    """Map graph aliases that resolve to one native material outlet.

    Generic separator ``out``/``main`` ports resolve to the native gas outlet,
    so they share identity with ``gas``/``vapor`` for graph validation.
    """
    output_port = str(raw_port).strip().lower()
    normalized_unit_type = str(unit_type).strip().lower()
    indexed_port = re.fullmatch(
        r"(?:out|split)[_-]?(\d+)",
        output_port,
    )
    if indexed_port:
        output_index = int(indexed_port.group(1))
        if normalized_unit_type in _TWO_SIDED_HEAT_EXCHANGER_UNIT_TYPES:
            return {
                0: "hot_out",
                1: "cold_out",
            }.get(output_index, f"heat_exchanger_out_{output_index}")
        return f"split_{output_index}"
    canonical_port = {
        "main": "out",
        "vapor": "gas",
        "oil": "liquid",
        "aqueous": "water",
    }.get(output_port, output_port)
    if (
        normalized_unit_type in _SEPARATOR_UNIT_TYPES
        and canonical_port == "out"
    ):
        return "gas"
    return canonical_port


def material_connection_name(connection: Any) -> str:
    """Return one explicit material-stream name with a stable ID fallback."""
    if not isinstance(connection, dict):
        raise ValueError("Material connection must be an object.")
    connection_id = str(connection.get("id", "")).strip()
    if not connection_id:
        raise ValueError("Material connection requires an id.")
    if str(connection.get("type", "")).strip().lower() != "material":
        raise ValueError(
            f"Connection '{connection_id}' is not a material stream."
        )
    raw_name = connection.get("name")
    if raw_name is None:
        return connection_id
    stream_name = str(raw_name).strip()
    if not stream_name:
        raise ValueError(
            f"Material connection '{connection_id}' requires a stream name."
        )
    return stream_name
