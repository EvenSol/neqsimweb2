"""Pure adapters for solved Process Chat and Studio diagnostics."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


_BOUNDARY_NUMERIC_FIELDS = (
    "mass_flow_kg_hr",
    "temperature_C",
    "pressure_bara",
    "molar_flow_mol_sec",
)


def material_boundary_rows(result: Any) -> List[Dict[str, Any]]:
    """Return validated, isolated material-boundary rows from a solve result."""
    raw = getattr(result, "raw", {})
    if not isinstance(raw, dict):
        raise ValueError("Solver result raw diagnostics must be an object.")
    source_rows = raw.get("material_boundaries", [])
    if source_rows is None:
        return []
    if not isinstance(source_rows, list):
        raise ValueError("Material boundary diagnostics must be an array.")

    rows: List[Dict[str, Any]] = []
    for index, source_row in enumerate(source_rows):
        if not isinstance(source_row, dict):
            raise ValueError(
                f"Material boundary row {index} must be an object."
            )
        role = str(source_row.get("role", "")).strip().lower()
        stream_name = str(source_row.get("stream_name", "")).strip()
        if role not in {"feed", "product"}:
            raise ValueError(
                f"Material boundary row {index} has an invalid role."
            )
        if not stream_name:
            raise ValueError(
                f"Material boundary row {index} requires a stream name."
            )

        row: Dict[str, Any] = {
            "role": role,
            "stream_name": stream_name,
        }
        for field_name in _BOUNDARY_NUMERIC_FIELDS:
            value = source_row.get(field_name)
            if value is None and field_name != "mass_flow_kg_hr":
                row[field_name] = None
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Material boundary row {index} field "
                    f"'{field_name}' must be numeric."
                ) from exc
            if not math.isfinite(numeric_value):
                raise ValueError(
                    f"Material boundary row {index} field "
                    f"'{field_name}' must be finite."
                )
            row[field_name] = numeric_value
        rows.append(row)
    return rows


def _kpi_value(result: Any, name: str) -> Optional[float]:
    kpis = getattr(result, "kpis", {})
    if not isinstance(kpis, dict):
        return None
    kpi = kpis.get(name)
    try:
        value = float(kpi.value)
    except (AttributeError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def aggregate_material_balance(result: Any) -> Dict[str, Optional[float]]:
    """Aggregate solved feed/product rows with KPI compatibility fallback."""
    rows = material_boundary_rows(result)
    feed_rows = [row for row in rows if row["role"] == "feed"]
    product_rows = [row for row in rows if row["role"] == "product"]

    feed_flow = (
        sum(row["mass_flow_kg_hr"] for row in feed_rows)
        if feed_rows
        else _kpi_value(result, "material_feed_flow_kg_hr")
    )
    product_flow = (
        sum(row["mass_flow_kg_hr"] for row in product_rows)
        if product_rows
        else _kpi_value(result, "material_product_flow_kg_hr")
    )
    imbalance_pct = _kpi_value(result, "mass_balance_pct")
    if (
        imbalance_pct is None
        and feed_flow is not None
        and product_flow is not None
        and feed_flow > 0.0
    ):
        imbalance_pct = abs(feed_flow - product_flow) / feed_flow * 100.0

    return {
        "feed_count": float(len(feed_rows)) if feed_rows else (
            _kpi_value(result, "material_feed_count")
        ),
        "product_count": float(len(product_rows)) if product_rows else (
            _kpi_value(result, "material_product_count")
        ),
        "feed_flow_kg_hr": feed_flow,
        "product_flow_kg_hr": product_flow,
        "imbalance_pct": imbalance_pct,
    }


def solved_feed_flow_kg_hr(
    result: Any,
    fallback_flow_kg_hr: float,
) -> float:
    """Return the aggregate solved feed flow or a validated legacy fallback."""
    summary = aggregate_material_balance(result)
    feed_flow = summary["feed_flow_kg_hr"]
    if feed_flow is not None and feed_flow > 0.0:
        return feed_flow
    try:
        fallback = float(fallback_flow_kg_hr)
    except (TypeError, ValueError) as exc:
        raise ValueError("Fallback feed flow must be numeric.") from exc
    if not math.isfinite(fallback) or fallback <= 0.0:
        raise ValueError("Fallback feed flow must be finite and positive.")
    return fallback
