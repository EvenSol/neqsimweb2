"""Pure adapters for solved Process Chat and Studio diagnostics."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


_BOUNDARY_NUMERIC_FIELDS = (
    "mass_flow_kg_hr",
    "temperature_C",
    "pressure_bara",
    "molar_flow_mol_sec",
    "enthalpy_flow_kW",
)
_COMPONENT_BALANCE_ABSOLUTE_TOL_MOL_SEC = 1.0e-9
_ENERGY_BALANCE_SCALE_FLOOR_KW = 1.0e-9
_ENERGY_TRANSFER_KINDS = {"shaft_work", "heat"}
_VALIDATION_STATUSES = {"OK", "WARN", "VIOLATION", "UNKNOWN"}


def aggregate_validation_status(statuses: Any) -> str:
    """Return the most severe reported validation status."""
    normalized = {
        str(status).strip().upper()
        for status in statuses
    }
    if not normalized:
        return "OK"
    if not normalized.issubset(_VALIDATION_STATUSES):
        normalized.add("UNKNOWN")
    for status in ("VIOLATION", "WARN", "UNKNOWN"):
        if status in normalized:
            return status
    return "OK"


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
        component_source = source_row.get(
            "component_molar_flows_mol_sec"
        )
        if component_source is None:
            row["component_molar_flows_mol_sec"] = None
        else:
            if not isinstance(component_source, dict):
                raise ValueError(
                    f"Material boundary row {index} component flows "
                    "must be an object."
                )
            component_flows: Dict[str, float] = {}
            for source_name, source_value in component_source.items():
                component_name = str(source_name).strip()
                if not component_name:
                    raise ValueError(
                        f"Material boundary row {index} has an empty "
                        "component name."
                    )
                try:
                    component_flow = float(source_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Material boundary row {index} component "
                        f"'{component_name}' flow must be numeric."
                    ) from exc
                if not math.isfinite(component_flow) or component_flow < 0.0:
                    raise ValueError(
                        f"Material boundary row {index} component "
                        f"'{component_name}' flow must be finite and "
                        "non-negative."
                    )
                component_flows[component_name] = component_flow
            row["component_molar_flows_mol_sec"] = component_flows
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
    raw = getattr(result, "raw", {})
    material_balance_applicable = (
        raw.get("material_balance_applicable")
        if isinstance(raw, dict)
        else None
    )
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
    imbalance_pct = (
        None
        if material_balance_applicable is False
        else _kpi_value(result, "mass_balance_pct")
    )
    if (
        material_balance_applicable is not False
        and imbalance_pct is None
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


def component_balance_rows(result: Any) -> List[Dict[str, float | str]]:
    """Return component feed/product closure rows from solved boundaries."""
    raw = getattr(result, "raw", {})
    component_balance_applicable = (
        raw.get("component_balance_applicable")
        if isinstance(raw, dict)
        else None
    )
    if component_balance_applicable is False:
        return []
    rows = material_boundary_rows(result)
    if not rows:
        if component_balance_applicable is True:
            raise ValueError(
                "Component boundary diagnostics are incomplete: "
                "no solved material boundaries are available."
            )
        return []
    component_maps = [
        row["component_molar_flows_mol_sec"]
        for row in rows
    ]
    if all(component_map is None for component_map in component_maps):
        if component_balance_applicable is True:
            raise ValueError(
                "Component boundary diagnostics are incomplete: "
                "component flows are unavailable."
            )
        return []
    positive_roles = {
        row["role"]
        for row in rows
        if row["mass_flow_kg_hr"] > 0.0
    }
    if positive_roles != {"feed", "product"}:
        raise ValueError(
            "Component boundary diagnostics require positive-flow "
            "feed and product boundaries."
        )
    for index, (row, component_map) in enumerate(
        zip(rows, component_maps)
    ):
        if (
            row["mass_flow_kg_hr"] > 0.0
            and not component_map
        ):
            raise ValueError(
                "Component boundary diagnostics are incomplete for "
                f"row {index}."
            )

    component_names = sorted(
        {
            component_name
            for component_map in component_maps
            if component_map
            for component_name in component_map
        }
    )
    balance_rows: List[Dict[str, float | str]] = []
    for component_name in component_names:
        feed_flow = sum(
            (row["component_molar_flows_mol_sec"] or {}).get(
                component_name,
                0.0,
            )
            for row in rows
            if row["role"] == "feed"
        )
        product_flow = sum(
            (row["component_molar_flows_mol_sec"] or {}).get(
                component_name,
                0.0,
            )
            for row in rows
            if row["role"] == "product"
        )
        residual = product_flow - feed_flow
        component_scale = max(
            feed_flow,
            product_flow,
            _COMPONENT_BALANCE_ABSOLUTE_TOL_MOL_SEC,
        )
        imbalance_pct = abs(residual) / component_scale * 100.0
        balance_rows.append(
            {
                "component": component_name,
                "feed_molar_flow_mol_sec": feed_flow,
                "product_molar_flow_mol_sec": product_flow,
                "residual_molar_flow_mol_sec": residual,
                "imbalance_pct": imbalance_pct,
            }
        )
    return balance_rows


def energy_transfer_rows(result: Any) -> List[Dict[str, float | str]]:
    """Return validated signed external-energy transfers from a solve."""
    raw = getattr(result, "raw", {})
    if not isinstance(raw, dict):
        raise ValueError("Solver result raw diagnostics must be an object.")
    source_rows = raw.get("energy_transfers", [])
    if source_rows is None:
        return []
    if not isinstance(source_rows, list):
        raise ValueError("Energy transfer diagnostics must be an array.")

    rows: List[Dict[str, float | str]] = []
    for index, source_row in enumerate(source_rows):
        if not isinstance(source_row, dict):
            raise ValueError(
                f"Energy transfer row {index} must be an object."
            )
        unit_name = str(source_row.get("unit_name", "")).strip()
        unit_type = str(source_row.get("unit_type", "")).strip()
        transfer_kind = str(
            source_row.get("transfer_kind", "")
        ).strip().lower()
        if not unit_name or not unit_type:
            raise ValueError(
                f"Energy transfer row {index} requires unit name and type."
            )
        if transfer_kind not in _ENERGY_TRANSFER_KINDS:
            raise ValueError(
                f"Energy transfer row {index} has an invalid kind."
            )
        try:
            energy_transfer_kW = float(
                source_row.get("energy_transfer_kW")
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Energy transfer row {index} must be numeric."
            ) from exc
        if not math.isfinite(energy_transfer_kW):
            raise ValueError(
                f"Energy transfer row {index} must be finite."
            )
        rows.append(
            {
                "unit_name": unit_name,
                "unit_type": unit_type,
                "transfer_kind": transfer_kind,
                "energy_transfer_kW": energy_transfer_kW,
            }
        )
    return rows


def aggregate_energy_balance(result: Any) -> Dict[str, Any]:
    """Aggregate material enthalpy and signed external-energy closure.

    Positive external transfer adds energy to the material system. The
    residual is ``products - feeds - external transfer``.
    """
    raw = getattr(result, "raw", {})
    if not isinstance(raw, dict):
        raise ValueError("Solver result raw diagnostics must be an object.")
    applicable = raw.get("energy_balance_applicable")
    if applicable not in {None, True, False}:
        raise ValueError(
            "Energy balance applicability must be true, false, or null."
        )

    rows = material_boundary_rows(result)
    transfers = energy_transfer_rows(result)
    if applicable is False:
        return {
            "applicable": False,
            "feed_enthalpy_kW": None,
            "product_enthalpy_kW": None,
            "external_energy_transfer_kW": None,
            "residual_kW": None,
            "imbalance_pct": None,
            "transfer_count": float(len(transfers)),
        }
    if not rows:
        if applicable is True:
            raise ValueError(
                "Energy boundary diagnostics are incomplete: "
                "no solved material boundaries are available."
            )
        return {
            "applicable": None,
            "feed_enthalpy_kW": None,
            "product_enthalpy_kW": None,
            "external_energy_transfer_kW": None,
            "residual_kW": None,
            "imbalance_pct": None,
            "transfer_count": float(len(transfers)),
        }

    positive_rows = [
        row for row in rows if row["mass_flow_kg_hr"] > 0.0
    ]
    positive_roles = {row["role"] for row in positive_rows}
    if positive_roles != {"feed", "product"}:
        raise ValueError(
            "Energy boundary diagnostics require positive-flow "
            "feed and product boundaries."
        )
    for index, row in enumerate(positive_rows):
        if row["enthalpy_flow_kW"] is None:
            raise ValueError(
                "Energy boundary diagnostics are incomplete for "
                f"positive-flow row {index}."
            )

    feed_enthalpy = sum(
        float(row["enthalpy_flow_kW"])
        for row in positive_rows
        if row["role"] == "feed"
    )
    product_enthalpy = sum(
        float(row["enthalpy_flow_kW"])
        for row in positive_rows
        if row["role"] == "product"
    )
    external_transfer = sum(
        float(row["energy_transfer_kW"])
        for row in transfers
    )
    residual = product_enthalpy - feed_enthalpy - external_transfer
    scale = max(
        abs(feed_enthalpy),
        abs(product_enthalpy),
        abs(external_transfer),
        _ENERGY_BALANCE_SCALE_FLOOR_KW,
    )
    imbalance_pct = abs(residual) / scale * 100.0
    return {
        "applicable": True if applicable is True else applicable,
        "feed_enthalpy_kW": feed_enthalpy,
        "product_enthalpy_kW": product_enthalpy,
        "external_energy_transfer_kW": external_transfer,
        "residual_kW": residual,
        "imbalance_pct": imbalance_pct,
        "transfer_count": float(len(transfers)),
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
