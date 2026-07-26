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
_CONVERGENCE_UNIT_TYPES = {"recycle", "adjuster"}
_CONVERGENCE_OPTIONAL_NUMERIC_FIELDS = (
    "flow_error",
    "temperature_error",
    "pressure_error",
    "composition_error",
    "error",
    "flow_tolerance",
    "temperature_tolerance",
    "pressure_tolerance",
    "composition_tolerance",
    "tolerance",
)
_UNIT_BALANCE_MASS_FIELDS = (
    "inlet_mass_flow_kg_hr",
    "outlet_mass_flow_kg_hr",
    "mass_residual_kg_hr",
    "mass_imbalance_pct",
)
_UNIT_BALANCE_ENERGY_FIELDS = (
    "inlet_enthalpy_kW",
    "outlet_enthalpy_kW",
    "external_energy_transfer_kW",
    "energy_residual_kW",
    "energy_imbalance_pct",
)
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


def convergence_rows(result: Any) -> List[Dict[str, Any]]:
    """Return validated per-unit recycle and adjuster convergence rows."""
    raw = getattr(result, "raw", {})
    if not isinstance(raw, dict):
        raise ValueError("Solver result raw diagnostics must be an object.")
    diagnostics = raw.get("convergence_diagnostics")
    if diagnostics is None:
        return []
    if not isinstance(diagnostics, dict):
        raise ValueError("Convergence diagnostics must be an object.")
    source_rows = diagnostics.get("rows", [])
    if not isinstance(source_rows, list):
        raise ValueError("Convergence diagnostic rows must be an array.")

    rows: List[Dict[str, Any]] = []
    identities = set()
    for index, source_row in enumerate(source_rows):
        if not isinstance(source_row, dict):
            raise ValueError(
                f"Convergence diagnostic row {index} must be an object."
            )
        process_system = str(
            source_row.get("process_system", "")
        ).strip()
        unit_name = str(source_row.get("unit_name", "")).strip()
        unit_type = str(source_row.get("unit_type", "")).strip().lower()
        converged = source_row.get("converged")
        if not process_system or not unit_name:
            raise ValueError(
                f"Convergence diagnostic row {index} requires process "
                "system and unit name."
            )
        if unit_type not in _CONVERGENCE_UNIT_TYPES:
            raise ValueError(
                f"Convergence diagnostic row {index} has an invalid "
                "unit type."
            )
        if not isinstance(converged, bool):
            raise ValueError(
                f"Convergence diagnostic row {index} requires a boolean "
                "converged state."
            )
        identity = (process_system, unit_name, unit_type)
        if identity in identities:
            raise ValueError(
                f"Convergence diagnostic row {index} duplicates a unit."
            )
        identities.add(identity)

        iterations = source_row.get("iterations")
        if iterations is not None:
            if isinstance(iterations, bool):
                raise ValueError(
                    f"Convergence diagnostic row {index} iterations "
                    "must be an integer."
                )
            try:
                iterations_value = int(iterations)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Convergence diagnostic row {index} iterations "
                    "must be an integer."
                ) from exc
            if iterations_value < 0 or float(iterations_value) != float(
                iterations
            ):
                raise ValueError(
                    f"Convergence diagnostic row {index} iterations "
                    "must be a non-negative integer."
                )
            iterations = iterations_value

        max_iterations = source_row.get("max_iterations")
        if max_iterations is not None:
            if isinstance(max_iterations, bool):
                raise ValueError(
                    f"Convergence diagnostic row {index} maximum "
                    "iterations must be an integer."
                )
            try:
                max_iterations_value = int(max_iterations)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Convergence diagnostic row {index} maximum "
                    "iterations must be an integer."
                ) from exc
            if (
                max_iterations_value < 1
                or float(max_iterations_value) != float(max_iterations)
            ):
                raise ValueError(
                    f"Convergence diagnostic row {index} maximum "
                    "iterations must be a positive integer."
                )
            max_iterations = max_iterations_value

        row: Dict[str, Any] = {
            "process_system": process_system,
            "unit_name": unit_name,
            "unit_type": unit_type,
            "converged": converged,
            "iterations": iterations,
            "max_iterations": max_iterations,
            "dominant_error": (
                str(source_row.get("dominant_error", "")).strip()
                or None
            ),
            "acceleration_method": (
                str(source_row.get("acceleration_method", "")).strip()
                or None
            ),
        }
        for field_name in _CONVERGENCE_OPTIONAL_NUMERIC_FIELDS:
            value = source_row.get(field_name)
            if value is None:
                row[field_name] = None
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Convergence diagnostic row {index} field "
                    f"'{field_name}' must be numeric."
                ) from exc
            if not math.isfinite(numeric_value) or numeric_value < 0.0:
                raise ValueError(
                    f"Convergence diagnostic row {index} field "
                    f"'{field_name}' must be finite and non-negative."
                )
            row[field_name] = numeric_value
        rows.append(row)
    return rows


def aggregate_convergence(result: Any) -> Dict[str, Any]:
    """Aggregate strict convergence state with legacy-result compatibility."""
    raw = getattr(result, "raw", {})
    if not isinstance(raw, dict):
        raise ValueError("Solver result raw diagnostics must be an object.")
    diagnostics = raw.get("convergence_diagnostics")
    if diagnostics is None:
        return {
            "applicable": None,
            "converged": None,
            "unit_count": None,
            "unconverged_count": None,
            "max_iterations": None,
            "suggestions": [],
        }
    if not isinstance(diagnostics, dict):
        raise ValueError("Convergence diagnostics must be an object.")
    applicable = diagnostics.get("applicable")
    converged = diagnostics.get("converged")
    if not isinstance(applicable, bool):
        raise ValueError(
            "Convergence applicability must be a boolean."
        )
    if converged is not None and not isinstance(converged, bool):
        raise ValueError(
            "Aggregate convergence state must be boolean or null."
        )
    source_suggestions = diagnostics.get("suggestions", [])
    if not isinstance(source_suggestions, list):
        raise ValueError("Convergence suggestions must be an array.")
    suggestions = []
    for index, suggestion in enumerate(source_suggestions):
        if not isinstance(suggestion, str) or not suggestion.strip():
            raise ValueError(
                f"Convergence suggestion {index} must be non-empty text."
            )
        suggestions.append(suggestion.strip())

    rows = convergence_rows(result)
    if not applicable:
        if rows or converged is not None:
            raise ValueError(
                "Feed-forward convergence diagnostics cannot contain "
                "iterative-unit state."
            )
        return {
            "applicable": False,
            "converged": None,
            "unit_count": 0.0,
            "unconverged_count": 0.0,
            "max_iterations": None,
            "suggestions": suggestions,
        }
    if not rows:
        raise ValueError(
            "Applicable convergence diagnostics require unit rows."
        )
    computed_converged = all(row["converged"] for row in rows)
    if converged is None or converged is not computed_converged:
        raise ValueError(
            "Aggregate convergence state conflicts with its unit rows."
        )
    iteration_values = [
        int(row["iterations"])
        for row in rows
        if row["iterations"] is not None
    ]
    return {
        "applicable": True,
        "converged": computed_converged,
        "unit_count": float(len(rows)),
        "unconverged_count": float(
            sum(not row["converged"] for row in rows)
        ),
        "max_iterations": (
            float(max(iteration_values)) if iteration_values else None
        ),
        "suggestions": suggestions,
    }


def unit_balance_rows(result: Any) -> List[Dict[str, Any]]:
    """Return validated per-unit material and energy closure rows."""
    raw = getattr(result, "raw", {})
    if not isinstance(raw, dict):
        raise ValueError("Solver result raw diagnostics must be an object.")
    diagnostics = raw.get("unit_balance_diagnostics")
    if diagnostics is None:
        return []
    if not isinstance(diagnostics, dict):
        raise ValueError("Unit balance diagnostics must be an object.")
    source_rows = diagnostics.get("rows", [])
    if not isinstance(source_rows, list):
        raise ValueError("Unit balance diagnostic rows must be an array.")

    rows: List[Dict[str, Any]] = []
    identities = set()
    for index, source_row in enumerate(source_rows):
        if not isinstance(source_row, dict):
            raise ValueError(
                f"Unit balance row {index} must be an object."
            )
        process_system = str(
            source_row.get("process_system", "")
        ).strip()
        unit_name = str(source_row.get("unit_name", "")).strip()
        unit_type = str(source_row.get("unit_type", "")).strip()
        if not process_system or not unit_name or not unit_type:
            raise ValueError(
                f"Unit balance row {index} requires process system, "
                "unit name, and unit type."
            )
        identity = (process_system, unit_name, unit_type)
        if identity in identities:
            raise ValueError(
                f"Unit balance row {index} duplicates a unit."
            )
        identities.add(identity)

        row: Dict[str, Any] = {
            "process_system": process_system,
            "unit_name": unit_name,
            "unit_type": unit_type,
        }
        for field_name in ("inlet_count", "outlet_count"):
            value = source_row.get(field_name)
            if isinstance(value, bool):
                raise ValueError(
                    f"Unit balance row {index} field '{field_name}' "
                    "must be a positive integer."
                )
            try:
                integer_value = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Unit balance row {index} field '{field_name}' "
                    "must be a positive integer."
                ) from exc
            if integer_value < 1 or float(integer_value) != float(value):
                raise ValueError(
                    f"Unit balance row {index} field '{field_name}' "
                    "must be a positive integer."
                )
            row[field_name] = integer_value

        for field_name in _UNIT_BALANCE_MASS_FIELDS:
            value = source_row.get(field_name)
            if isinstance(value, bool):
                raise ValueError(
                    f"Unit balance row {index} field '{field_name}' "
                    "must be numeric."
                )
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Unit balance row {index} field '{field_name}' "
                    "must be numeric."
                ) from exc
            if not math.isfinite(numeric_value):
                raise ValueError(
                    f"Unit balance row {index} field '{field_name}' "
                    "must be finite."
                )
            if (
                field_name != "mass_residual_kg_hr"
                and numeric_value < 0.0
            ):
                raise ValueError(
                    f"Unit balance row {index} field '{field_name}' "
                    "must be non-negative."
                )
            row[field_name] = numeric_value

        energy_values = [
            source_row.get(field_name)
            for field_name in _UNIT_BALANCE_ENERGY_FIELDS
        ]
        energy_available = any(
            value is not None for value in energy_values
        )
        if energy_available and any(
            value is None for value in energy_values
        ):
            raise ValueError(
                f"Unit balance row {index} has incomplete energy closure."
            )
        for field_name, value in zip(
            _UNIT_BALANCE_ENERGY_FIELDS,
            energy_values,
        ):
            if value is None:
                row[field_name] = None
                continue
            if isinstance(value, bool):
                raise ValueError(
                    f"Unit balance row {index} field '{field_name}' "
                    "must be numeric."
                )
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Unit balance row {index} field '{field_name}' "
                    "must be numeric."
                ) from exc
            if not math.isfinite(numeric_value):
                raise ValueError(
                    f"Unit balance row {index} field '{field_name}' "
                    "must be finite."
                )
            if (
                field_name == "energy_imbalance_pct"
                and numeric_value < 0.0
            ):
                raise ValueError(
                    f"Unit balance row {index} energy imbalance must "
                    "be non-negative."
                )
            row[field_name] = numeric_value
        rows.append(row)
    return rows


def aggregate_unit_balances(result: Any) -> Dict[str, Any]:
    """Aggregate strict per-unit closure with legacy compatibility."""
    raw = getattr(result, "raw", {})
    if not isinstance(raw, dict):
        raise ValueError("Solver result raw diagnostics must be an object.")
    diagnostics = raw.get("unit_balance_diagnostics")
    if diagnostics is None:
        return {
            "applicable": None,
            "coverage_complete": None,
            "energy_coverage_complete": None,
            "unit_count": None,
            "energy_unit_count": None,
            "max_mass_imbalance_pct": None,
            "max_energy_imbalance_pct": None,
            "max_mass_imbalance_unit": None,
            "max_energy_imbalance_unit": None,
            "excluded_units": [],
        }
    if not isinstance(diagnostics, dict):
        raise ValueError("Unit balance diagnostics must be an object.")

    applicable = diagnostics.get("applicable")
    coverage_complete = diagnostics.get("coverage_complete")
    if not isinstance(applicable, bool):
        raise ValueError("Unit balance applicability must be a boolean.")
    if not isinstance(coverage_complete, bool):
        raise ValueError(
            "Unit balance coverage state must be a boolean."
        )
    source_excluded = diagnostics.get("excluded_units", [])
    if not isinstance(source_excluded, list):
        raise ValueError("Excluded unit balances must be an array.")
    excluded_units = []
    for index, source_unit in enumerate(source_excluded):
        if not isinstance(source_unit, str) or not source_unit.strip():
            raise ValueError(
                f"Excluded unit balance {index} must be non-empty text."
            )
        excluded_units.append(source_unit.strip())
    if coverage_complete is not (not excluded_units):
        raise ValueError(
            "Unit balance coverage state conflicts with excluded units."
        )

    rows = unit_balance_rows(result)
    if applicable is not bool(rows):
        raise ValueError(
            "Unit balance applicability conflicts with its rows."
        )
    energy_rows = [
        row
        for row in rows
        if row["energy_imbalance_pct"] is not None
    ]
    mass_limiting_row = (
        min(
            rows,
            key=lambda row: (
                -row["mass_imbalance_pct"],
                row["process_system"],
                row["unit_name"],
                row["unit_type"],
            ),
        )
        if rows
        else None
    )
    energy_limiting_row = (
        min(
            energy_rows,
            key=lambda row: (
                -row["energy_imbalance_pct"],
                row["process_system"],
                row["unit_name"],
                row["unit_type"],
            ),
        )
        if energy_rows
        else None
    )

    def unit_identity(row: Dict[str, Any] | None) -> Dict[str, str] | None:
        if row is None:
            return None
        return {
            "process_system": row["process_system"],
            "unit_name": row["unit_name"],
            "unit_type": row["unit_type"],
        }

    return {
        "applicable": applicable,
        "coverage_complete": coverage_complete,
        "energy_coverage_complete": len(energy_rows) == len(rows),
        "unit_count": float(len(rows)),
        "energy_unit_count": float(len(energy_rows)),
        "max_mass_imbalance_pct": (
            max(row["mass_imbalance_pct"] for row in rows)
            if rows
            else None
        ),
        "max_energy_imbalance_pct": (
            max(row["energy_imbalance_pct"] for row in energy_rows)
            if energy_rows
            else None
        ),
        "max_mass_imbalance_unit": unit_identity(mass_limiting_row),
        "max_energy_imbalance_unit": unit_identity(energy_limiting_row),
        "excluded_units": excluded_units,
    }


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
    missing_positive_enthalpy = any(
        row["enthalpy_flow_kW"] is None for row in positive_rows
    )
    if applicable is None and (
        positive_roles != {"feed", "product"}
        or missing_positive_enthalpy
    ):
        return {
            "applicable": None,
            "feed_enthalpy_kW": None,
            "product_enthalpy_kW": None,
            "external_energy_transfer_kW": None,
            "residual_kW": None,
            "imbalance_pct": None,
            "transfer_count": float(len(transfers)),
        }
    if positive_roles != {"feed", "product"}:
        raise ValueError(
            "Energy boundary diagnostics require positive-flow "
            "feed and product boundaries."
        )
    for index, row in enumerate(positive_rows):
        if missing_positive_enthalpy and row["enthalpy_flow_kW"] is None:
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
