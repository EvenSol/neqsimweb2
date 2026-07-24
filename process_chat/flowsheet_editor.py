"""Pure schema helpers for editing Process Flowsheet Studio graphs."""

from __future__ import annotations

import copy
import re
from typing import Any


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
