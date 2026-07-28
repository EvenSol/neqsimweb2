"""Shared validation helpers for Process Flowsheet Studio graph schemas."""

from __future__ import annotations

from typing import Any


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
