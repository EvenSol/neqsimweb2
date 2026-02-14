"""
Equipment Templates — pre-defined parameter sets for planning scenarios.

When an engineer says "install a cooler" or "add a compressor stage",
the LLM maps to one of these templates and fills in default parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TemplateDef:
    """Definition of an equipment template for planning scenarios."""
    name: str
    display_name: str
    description: str
    required_params: List[str]
    optional_params: List[str]
    default_values: Dict[str, Any]
    constraints: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Template library
# ---------------------------------------------------------------------------

TEMPLATES: Dict[str, TemplateDef] = {
    "cooler": TemplateDef(
        name="cooler",
        display_name="Cooler / After-cooler",
        description="Air or water cooler to reduce gas temperature",
        required_params=["outlet_temperature_C"],
        optional_params=["pressure_drop_bar", "duty_kW"],
        default_values={
            "outlet_temperature_C": 35.0,
            "pressure_drop_bar": 0.2,
        },
        constraints={
            "min_approach_C": "5°C minimum approach temperature",
            "max_outlet_T_C": "Must be above hydrate formation temperature",
        },
    ),

    "heater": TemplateDef(
        name="heater",
        display_name="Heater",
        description="Process heater to increase temperature",
        required_params=["outlet_temperature_C"],
        optional_params=["pressure_drop_bar", "duty_kW"],
        default_values={
            "outlet_temperature_C": 60.0,
            "pressure_drop_bar": 0.1,
        },
        constraints={
            "max_temperature_C": "Material temperature limit",
        },
    ),

    "compressor_stage": TemplateDef(
        name="compressor_stage",
        display_name="Compressor Stage",
        description="Centrifugal or reciprocating compressor stage",
        required_params=["outlet_pressure_bara"],
        optional_params=["isentropic_efficiency", "polytropic_efficiency"],
        default_values={
            "isentropic_efficiency": 0.75,
            "polytropic_efficiency": 0.80,
        },
        constraints={
            "max_pressure_ratio": "Typically 3-4 per stage for centrifugal",
            "max_discharge_T_C": "Usually 150°C for centrifugal",
            "surge_margin": "Minimum 10% surge margin",
        },
    ),

    "separator": TemplateDef(
        name="separator",
        display_name="Separator (2-phase or 3-phase)",
        description="Vessel for phase separation",
        required_params=["separator_type"],
        optional_params=["pressure_bara", "temperature_C"],
        default_values={
            "separator_type": "two_phase",
        },
        constraints={
            "liquid_residence_time": "Minimum 2 minutes for 2-phase, 5 for 3-phase",
        },
    ),

    "valve": TemplateDef(
        name="valve",
        display_name="Throttling Valve / JT Valve",
        description="Pressure reduction via throttling",
        required_params=["outlet_pressure_bara"],
        optional_params=[],
        default_values={},
        constraints={
            "hydrate_risk": "Check hydrate formation at outlet conditions",
        },
    ),

    "heat_exchanger": TemplateDef(
        name="heat_exchanger",
        display_name="Heat Exchanger",
        description="Shell-and-tube or plate heat exchanger",
        required_params=["UA_value"],
        optional_params=["outlet_temperature_C", "pressure_drop_bar"],
        default_values={
            "pressure_drop_bar": 0.5,
        },
        constraints={
            "min_approach_C": "5°C minimum temperature approach",
            "max_pressure_drop": "Design limit",
        },
    ),

    "pump": TemplateDef(
        name="pump",
        display_name="Pump",
        description="Liquid pump for pressure boosting",
        required_params=["outlet_pressure_bara"],
        optional_params=["efficiency"],
        default_values={
            "efficiency": 0.75,
        },
        constraints={
            "NPSH": "Net positive suction head must be maintained",
        },
    ),

    "expander": TemplateDef(
        name="expander",
        display_name="Expander / Turbo-expander",
        description="Gas expander for power recovery or refrigeration",
        required_params=["outlet_pressure_bara"],
        optional_params=["isentropic_efficiency"],
        default_values={
            "isentropic_efficiency": 0.80,
        },
        constraints={
            "liquid_formation": "Check for liquid at outlet",
        },
    ),
}


def get_template(name: str) -> TemplateDef:
    """Get a template by name."""
    if name not in TEMPLATES:
        available = ", ".join(TEMPLATES.keys())
        raise KeyError(f"Unknown template: {name}. Available: {available}")
    return TEMPLATES[name]


def list_templates() -> List[Dict[str, Any]]:
    """List all available templates for display."""
    return [
        {
            "name": t.name,
            "display_name": t.display_name,
            "description": t.description,
            "required_params": t.required_params,
            "optional_params": t.optional_params,
            "default_values": t.default_values,
        }
        for t in TEMPLATES.values()
    ]


def template_help_text() -> str:
    """Generate help text about available templates for the LLM system prompt."""
    lines = ["Available equipment templates for planning scenarios:"]
    for t in TEMPLATES.values():
        lines.append(f"\n  {t.display_name} (template: '{t.name}')")
        lines.append(f"    {t.description}")
        lines.append(f"    Required: {', '.join(t.required_params)}")
        if t.optional_params:
            lines.append(f"    Optional: {', '.join(t.optional_params)}")
        if t.default_values:
            defaults = ", ".join(f"{k}={v}" for k, v in t.default_values.items())
            lines.append(f"    Defaults: {defaults}")
    return "\n".join(lines)
