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

    "air_cooler": TemplateDef(
        name="air_cooler",
        display_name="Air Cooler (Fin-fan)",
        description="Air-cooled heat exchanger (fin-fan cooler)",
        required_params=["outlet_temperature_C"],
        optional_params=["pressure_drop_bar"],
        default_values={
            "outlet_temperature_C": 40.0,
            "pressure_drop_bar": 0.3,
        },
        constraints={
            "min_approach_C": "Typically 10-15°C above ambient",
        },
    ),

    "water_cooler": TemplateDef(
        name="water_cooler",
        display_name="Water Cooler",
        description="Water-cooled heat exchanger",
        required_params=["outlet_temperature_C"],
        optional_params=["pressure_drop_bar"],
        default_values={
            "outlet_temperature_C": 30.0,
            "pressure_drop_bar": 0.2,
        },
        constraints={
            "min_approach_C": "5°C above cooling water temperature",
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
        display_name="Separator (2-phase)",
        description="Two-phase gas/liquid separation vessel",
        required_params=[],
        optional_params=["pressure_bara", "temperature_C"],
        default_values={},
        constraints={
            "liquid_residence_time": "Minimum 2 minutes",
        },
    ),

    "three_phase_separator": TemplateDef(
        name="three_phase_separator",
        display_name="Three-phase Separator",
        description="Vessel for gas/oil/water separation",
        required_params=[],
        optional_params=["pressure_bara", "temperature_C"],
        default_values={},
        constraints={
            "liquid_residence_time": "Minimum 5 minutes for three-phase",
        },
    ),

    "gas_scrubber": TemplateDef(
        name="gas_scrubber",
        display_name="Gas Scrubber",
        description="Knock-out drum / scrubber for removing liquid from gas",
        required_params=[],
        optional_params=["pressure_bara", "temperature_C"],
        default_values={},
        constraints={
            "gas_load_factor": "Max gas load per separator internals",
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

    "control_valve": TemplateDef(
        name="control_valve",
        display_name="Control Valve",
        description="Automated control valve for pressure/flow regulation",
        required_params=["outlet_pressure_bara"],
        optional_params=["Cv"],
        default_values={},
        constraints={},
    ),

    "heat_exchanger": TemplateDef(
        name="heat_exchanger",
        display_name="Heat Exchanger",
        description="Shell-and-tube or plate heat exchanger (2-stream)",
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

    "mixer": TemplateDef(
        name="mixer",
        display_name="Mixer",
        description="Combines multiple streams into one",
        required_params=[],
        optional_params=[],
        default_values={},
        constraints={},
    ),

    "splitter": TemplateDef(
        name="splitter",
        display_name="Splitter",
        description="Splits a stream into multiple streams (by fraction)",
        required_params=["split_fractions"],
        optional_params=[],
        default_values={
            "split_fractions": [0.5, 0.5],
        },
        constraints={
            "sum_to_1": "Split fractions must sum to 1.0",
        },
    ),

    "simple_absorber": TemplateDef(
        name="simple_absorber",
        display_name="Absorber Column",
        description="Simple absorption column (e.g. TEG dehydration, amine scrubbing)",
        required_params=[],
        optional_params=["number_of_stages"],
        default_values={
            "number_of_stages": 5,
        },
        constraints={},
    ),

    "simple_teg_absorber": TemplateDef(
        name="simple_teg_absorber",
        display_name="TEG Absorber",
        description="Triethylene glycol (TEG) dehydration absorber",
        required_params=[],
        optional_params=["number_of_stages", "teg_flow_rate_kg_hr"],
        default_values={
            "number_of_stages": 5,
        },
        constraints={},
    ),

    "pipeline": TemplateDef(
        name="pipeline",
        display_name="Pipeline",
        description="Pipeline segment for flow simulation",
        required_params=["length_m", "diameter_m"],
        optional_params=["roughness_m", "elevation_m"],
        default_values={
            "roughness_m": 0.00005,
            "elevation_m": 0.0,
        },
        constraints={},
    ),

    "adiabatic_pipe": TemplateDef(
        name="adiabatic_pipe",
        display_name="Adiabatic Pipe",
        description="Adiabatic pipe segment (no heat loss)",
        required_params=["length_m", "diameter_m"],
        optional_params=["roughness_m"],
        default_values={
            "roughness_m": 0.00005,
        },
        constraints={},
    ),

    "ejector": TemplateDef(
        name="ejector",
        display_name="Ejector",
        description="Gas ejector for compression using motive gas",
        required_params=[],
        optional_params=[],
        default_values={},
        constraints={},
    ),

    "flare": TemplateDef(
        name="flare",
        display_name="Flare",
        description="Flare for combustion of waste gases",
        required_params=[],
        optional_params=[],
        default_values={},
        constraints={},
    ),

    "gas_turbine": TemplateDef(
        name="gas_turbine",
        display_name="Gas Turbine",
        description="Gas turbine for power generation",
        required_params=[],
        optional_params=["efficiency"],
        default_values={
            "efficiency": 0.35,
        },
        constraints={},
    ),

    "membrane_separator": TemplateDef(
        name="membrane_separator",
        display_name="Membrane Separator",
        description="Membrane-based gas separation unit",
        required_params=[],
        optional_params=[],
        default_values={},
        constraints={},
    ),

    "gibbs_reactor": TemplateDef(
        name="gibbs_reactor",
        display_name="Gibbs Reactor",
        description="Chemical reactor using Gibbs energy minimization",
        required_params=[],
        optional_params=["temperature_C", "pressure_bara"],
        default_values={},
        constraints={},
    ),

    "well_flow": TemplateDef(
        name="well_flow",
        display_name="Well Flow",
        description="Well flow / tubing performance model",
        required_params=["wellhead_pressure_bara"],
        optional_params=["reservoir_pressure_bara", "depth_m"],
        default_values={},
        constraints={},
    ),

    "recycle": TemplateDef(
        name="recycle",
        display_name="Recycle",
        description="Recycle stream for iterative convergence of recycle loops",
        required_params=[],
        optional_params=["tolerance"],
        default_values={
            "tolerance": 1e-4,
        },
        constraints={},
    ),

    "adjuster": TemplateDef(
        name="adjuster",
        display_name="Adjuster / Controller",
        description="Adjusts a variable to meet a target specification",
        required_params=["target_variable", "target_value"],
        optional_params=["tolerance"],
        default_values={},
        constraints={},
    ),

    "electrolyzer": TemplateDef(
        name="electrolyzer",
        display_name="Electrolyzer",
        description="Water electrolyzer for hydrogen production",
        required_params=[],
        optional_params=["efficiency"],
        default_values={},
        constraints={},
    ),

    "adsorber": TemplateDef(
        name="adsorber",
        display_name="Adsorber",
        description="Adsorption unit (e.g. molecular sieve, activated carbon)",
        required_params=[],
        optional_params=[],
        default_values={},
        constraints={},
    ),

    "tank": TemplateDef(
        name="tank",
        display_name="Storage Tank",
        description="Atmospheric or pressurized storage tank",
        required_params=[],
        optional_params=["pressure_bara"],
        default_values={},
        constraints={},
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
