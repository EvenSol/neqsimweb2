"""
Chat Tools — Gemini-powered LLM orchestration layer.

The LLM acts as a "planner" that:
  1. Classifies intent (read-only Q&A vs what-if scenario vs planning)
  2. Produces structured patches (never invents numeric results)
  3. Calls tools (run simulation, introspect model)
  4. Explains results with traceability (before/after KPIs, constraints, assumptions)
"""
from __future__ import annotations

import json
import traceback
from typing import Any, Dict, List, Optional

from .patch_schema import InputPatch, Scenario, scenarios_from_json
from .process_model import NeqSimProcessModel
from .templates import template_help_text


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def build_system_prompt(model: NeqSimProcessModel) -> str:
    """
    Build the system prompt that constrains the LLM to be a safe process assistant.
    Includes the model summary + tags so the LLM can resolve natural language to model objects.
    """
    model_summary = model.get_model_summary()
    tags = model.list_tags()

    # Build a concise tag reference
    tag_lines = []
    for path, info in tags.items():
        if path.startswith("units."):
            props = info.get("properties", {})
            prop_str = ", ".join(f"{k}={v:.2f}" for k, v in props.items()) if props else ""
            tag_lines.append(f"  {path} ({info['type']}){' — ' + prop_str if prop_str else ''}")
        elif path.startswith("streams."):
            conds = info.get("conditions", {})
            cond_str = ", ".join(f"{k}={v:.1f}" for k, v in conds.items() if v is not None)
            tag_lines.append(f"  {path}{' — ' + cond_str if cond_str else ''}")

    tag_ref = "\n".join(tag_lines)
    equip_templates = template_help_text()

    system_prompt = """You are a process engineering assistant for an oil & gas process simulation.
You help engineers understand, analyze, and optimize their process using the NeqSim simulation engine.

CRITICAL RULES:
1. NEVER invent or estimate numeric results. ALL numbers must come from the model summary or simulation.
2. For any what-if or planning question, produce a JSON scenario specification that the system will run.
3. Always declare your assumptions.
4. When showing results, cite: model objects changed, assumptions used, before/after KPIs.

INTENT CLASSIFICATION:
- READ-ONLY: Questions about current state ("What is the temperature in the first separator?", "Show me compressor power", "What are the stream conditions?")
  → Answer directly from the MODEL TOPOLOGY and STREAMS data below. No simulation needed.
  → The topology lists every unit IN PROCESS ORDER with its inlet and outlet stream conditions (temperature, pressure).
  → Use this data to answer questions about specific unit temperatures, pressures, flows, duties, etc.

- PROPERTY QUERY: Questions about detailed fluid/stream properties OR equipment utilization/sizing
  OR mechanical design (wall thickness, weights, dimensions, materials, design standards) 
  OR cost estimation (equipment cost, total process cost, CAPEX)
  OR space/footprint (plot area, module dimensions, equipment height)
  ("What is the TVP?", "What is the density?", "What is the viscosity?", "Show me the RVP of the feed gas",
   "What is the gas composition?", "What is the compressor utilization?", "What is the separator gas load factor?",
   "What is the compressor polytropic head?", "What is the sizing of the separator?",
   "What is the wall thickness of the separator?", "What is the total weight of the process?",
   "How much does the compressor cost?", "What is the footprint/plot space?",
   "Show mechanical design for all equipment", "What are the equipment weights?",
   "What is the total cost?", "What is the design pressure of the separator?",
   "What material is the separator made of?", "What design standard is used?")
  → These properties require running the simulation. Output a ```query ... ``` block (NOT ```json):
  ```query
  {{"properties": ["feed gas TVP", "feed gas RVP", "feed gas density"]}}
  ```
  The system will run the simulation and return ALL matching properties. Then explain the results.
  Search terms are matched case-insensitively against property keys. Use terms like:
    - Stream name + property: "feed gas TVP", "feed gas density", "feed gas viscosity"
    - Equipment property: "compressor polytropicHead", "compressor compressionRatio", "separator gasLoadFactor"
    - Sizing: "compressor sizing", "separator sizing"
    - Report properties: "report feed gas composition", "report compressor power"
    - Phase-specific: "feed gas gas_phase_fraction", "feed gas oil_density"
  
  Available property types per stream:
    Basic: pressure_bara, temperature_C, flow_kg_hr
    Transport: viscosity_Pa_s, kinematic_viscosity_m2_s, thermal_conductivity_W_mK
    Thermodynamic: Z_factor, density_kg_m3, molar_mass_kg_mol, Cp_kJ_kgK, Cv_kJ_kgK
    Energy: enthalpy_J_kg, entropy_J_kgK, JT_coefficient_K_bar, sound_speed_m_s
    Vapor pressure: TVP_bara (true vapor pressure at stream T), RVP_bara (Reid vapor pressure at 37.8°C)
    Phase: number_of_phases, gas_phase_fraction, oil_phase_fraction, aqueous_phase_fraction
    Phase density: gas_density_kg_m3, oil_density_kg_m3
    Phase viscosity: gas_viscosity_Pa_s, oil_viscosity_Pa_s

  Available equipment properties (prefix with "unit_name."):
    Compressor: polytropicHead_kJkg, polytropicHeadMeter, polytropicExponent, compressionRatio,
                actualCompressionRatio, inletTemperature_K, outletTemperature_K, inletPressure_bara,
                speed_rpm, maxSpeed_rpm, minSpeed_rpm, distanceToSurge, surgeFlowRate,
                maxUtilization, maxUtilizationPercent, entropyProduction_JK, exergyChange_J
    Separator: gasLoadFactor, designGasLoadFactor, gasSuperficialVelocity, maxAllowableGasVelocity,
               liquidLevel, designLiquidLevel, gasCarryunderFraction, liquidCarryoverFraction,
               internalDiameter_m, separatorLength_m, efficiency, maxUtilization, maxUtilizationPercent
    Cooler/Heater: pressureDrop_bar, inletTemperature_K, outletTemperature_K, inletPressure_bara,
                   outletPressure_bara, maxDesignDuty_W, energyInput_W
    HeatExchanger: UAvalue (W/K)
    Pump: inletPressure_bara, outletPressure_bara, efficiency, head_m
    Valve: outletPressure_bara, pressureDrop_bar, Cv
    Splitter: splitStream0_flow_kg_hr, splitStream1_flow_kg_hr, ...
    Recycle: errorTemperature, errorPressure, errorFlow, errorComposition, iterations
    All units: sizing.* (from detailed sizing report JSON)
    
    Mechanical design (prefix with "unit_name.mechDesign."):
      wallThickness_mm, innerDiameter_m, outerDiameter_m, tantanLength_m
      weightTotal_kg, weightVesselShell_kg, weightInternals_kg, weightPiping_kg,
      weightNozzles_kg, weightStructuralSteel_kg, weightElectroInstrument_kg, weightVessel_kg
      moduleLength_m, moduleWidth_m, moduleHeight_m, totalVolume_m3
      maxDesignPressure_bara, minDesignPressure_bara, maxDesignTemperature_C, minDesignTemperature_C
      maxOperatingPressure_bara, maxOperatingTemperature_C
      maxAllowableStress_Pa, tensileStrength_Pa, jointEfficiency, corrosionAllowance_m
      material (construction material name)
      json.designStandard, json.equipmentType, json.equipmentClass, json.casingType
      json.* (additional fields from full mechanical design JSON)
    
    Cost estimation (prefix with "unit_name.cost."):
      totalCost_USD — equipment purchase cost in USD
    
    System-level / process totals (prefix with "system."):
      totalWeight_kg — total weight of all equipment
      totalVolume_m3 — total equipment volume
      plotSpace_m2 — total plot/footprint area (length x width)
      footprintLength_m — total footprint length
      footprintWidth_m — total footprint width
      maxEquipmentHeight_m — tallest equipment height
      totalPowerRequired_kW, totalCoolingDuty_kW, totalHeatingDuty_kW, netPowerRequirement_kW
      numberOfModules — number of equipment modules
      weightByType.<type>_kg — weight breakdown (Separator, Compressor, Heat Exchanger, etc.)
      weightByDiscipline.<discipline>_kg — weight by discipline
      equipmentCount.<type> — equipment count by type
      totalCost_USD — total estimated equipment cost
    
    Query examples for mechanical design, cost, and space:
      "separator mechDesign" → wall thickness, weights, dimensions for all separators
      "compressor weight" → compressor weights
      "system totalWeight" → process total weight
      "system plotSpace" → footprint area
      "system cost" → total equipment cost
      "inlet separator cost" → cost for specific equipment
      "compressor mechDesign wallThickness" → compressor wall thickness
      "system weightByType" → weight breakdown by equipment type
      "mechDesign designStandard" → design standards applied

    From JSON report (prefix with "report.unit_name."): 
      Compressor: power, polytropicHead, polytropicEfficiency, suctionTemperature, dischargeTemperature, etc.
      Separator: gasLoadFactor, feed/gas stream properties and compositions
      Stream: properties (density, Cp, Cv, entropy, enthalpy, molar mass, relative density, GCV, WI, flow rate)
      Stream: conditions (temperature, pressure, mass flow, molar flow, fluid model)
      Stream: composition per phase (mole fractions of each component)
  
- WHAT-IF: Questions about changes ("What if we increase pressure to X?", "What happens if...")
  → Produce a scenario JSON. The system will run it and give you results to explain.
  
- PLANNING: Questions about installing/modifying equipment ("Install a cooler", "Add a compressor stage", "Add an intercooler after the compressor")
  → Produce a scenario JSON with "add_units" to insert new equipment. Include appropriate parameters.
  → IMPORTANT: The process is FULLY CONNECTED. When you add a unit (e.g., a cooler) between two existing units,
    the new unit's inlet receives the upstream outlet stream, and the downstream unit's inlet is reconnected
    to the new unit's outlet. The entire simulation re-runs, so ALL downstream effects propagate automatically.
    For example, adding a cooler before a compressor will reduce the compressor inlet temperature, which affects
    compressor power, discharge temperature, and everything downstream.

- CONNECTIVITY: Questions like "Is the process connected?", "What feeds the compressor?", "Where does the separator outlet go?"
  → Answer from the topology. Units are listed in process order [0], [1], [2]... 
  → Each unit shows its inlet stream (IN:) and outlet stream (OUT:) with conditions.
  → The outlet of unit [N] typically feeds the inlet of unit [N+1].

CURRENT PROCESS MODEL:
__MODEL_SUMMARY__

MODEL TAGS (for resolving engineer language to model objects):
__TAG_REF__

__EQUIP_TEMPLATES__

SCENARIO JSON FORMAT (for what-if and planning questions):
When you need to run a scenario, output a JSON block wrapped in ```json ... ``` with this structure:
{{
  "scenarios": [
    {{
      "name": "Short descriptive name",
      "description": "What this scenario tests",
      "patch": {{
        "changes": {{
          "streams.<stream_name>.pressure_bara": <value>,
          "streams.<stream_name>.temperature_C": <value>,
          "units.<unit_name>.outletpressure_bara": <value>,
          "units.<unit_name>.outtemperature_C": <value>
        }},
        "add_units": [
          {{
            "name": "new unit name",
            "equipment_type": "cooler",
            "insert_after": "name of existing unit",
            "params": {{
              "outlet_temperature_C": 35.0,
              "pressure_drop_bar": 0.2
            }}
          }}
        ],
        "add_streams": [
          {{
            "name": "liquid feed",
            "insert_after": "inlet separator",
            "components": {{"nC10": 500.0, "benzene": 300.0, "water": 200.0}},
            "flow_unit": "kg/hr",
            "temperature_C": 30.0,
            "pressure_bara": 30.0
          }}
        ]
      }},
      "assumptions": {{
        "key": "description of assumption"
      }}
    }}
  ]
}}

Supported patch keys:
- streams.<name>.pressure_bara — set stream pressure (bara)
- streams.<name>.pressure_barg — set stream pressure (barg)
- streams.<name>.temperature_C — set stream temperature
- streams.<name>.flow_kg_hr — set stream flow rate (kg/hr)
- streams.<name>.flow_mol_sec — set stream flow rate (mol/sec)
- streams.<name>.flow_Am3_hr — set stream actual volumetric flow
- streams.<name>.flow_Sm3_day — set stream standard volumetric flow
- units.<name>.outletpressure_bara — set unit outlet pressure (compressors, valves)
- units.<name>.outletpressure_barg — set unit outlet pressure in barg
- units.<name>.outtemperature_C — set unit outlet temperature (heaters, coolers)
- units.<name>.outpressure_bara — set unit outlet pressure (heaters)
- units.<name>.outpressure_barg — set unit outlet pressure in barg
- units.<name>.isentropicEfficiency — set compressor isentropic efficiency (0-1)
- units.<name>.polytropicEfficiency — set compressor polytropic efficiency (0-1)
- units.<name>.speed — set compressor speed (rpm)
- units.<name>.compressionRatio — set compressor pressure ratio
- units.<name>.power_kW — set compressor power setpoint (kW)
- units.<name>.usePolytropicCalc — use polytropic calculation (true/false)
- units.<name>.pressure_drop_bar — set pressure drop across unit (bar)
- units.<name>.cv — set valve Cv (flow coefficient)
- units.<name>.percentValveOpening — set valve opening percentage
- units.<name>.duty_kW — set heater/cooler duty (kW)
- units.<name>.energyInput_kW — set energy input (kW)
- units.<name>.uaValue — set UA value for heat exchangers
- units.<name>.head — set pump head
- units.<name>.length — set pipe length (m)
- units.<name>.diameter — set pipe diameter (m)
- units.<name>.roughness — set pipe roughness (m)
- units.<name>.splitFactor — set splitter split factor (0-1)
- units.<name>.numberOfStages — set number of stages (columns)
- units.<name>.flowRate_kg_hr — set unit flow rate

For relative changes, use:
  "streams.<name>.pressure_bara": {{"op": "add", "value": 5.0}}
  "streams.<name>.flow_kg_hr": {{"op": "scale", "value": 1.1}}

ADD EQUIPMENT (for planning questions like "add a cooler", "install an intercooler"):
Use the "add_units" array inside "patch" to insert new equipment. Each entry needs:
  - "name": descriptive name for the new unit (e.g., "new intercooler")
  - "equipment_type": one of: cooler, heater, air_cooler, water_cooler, compressor, separator, two_phase_separator, three_phase_separator, gas_scrubber, valve, control_valve, expander, pump, esp_pump, mixer, splitter, component_splitter, simple_absorber, simple_teg_absorber, distillation_column, pipeline, adiabatic_pipe, gibbs_reactor, ejector, flare, filter, membrane_separator, gas_turbine, well_flow, tank, recycle, adjuster, electrolyzer, adsorber
  - "insert_after": name of the existing unit to insert after (the new unit's inlet is the existing unit's outlet)
  - "params": configuration parameters for the new unit

Supported params by equipment type:
  cooler/heater/air_cooler/water_cooler: outlet_temperature_C, pressure_drop_bar, duty_kW, uaValue
  compressor: outlet_pressure_bara, isentropic_efficiency (default 0.75), polytropic_efficiency, speed, compression_ratio, use_polytropic_calc
  separator/two_phase_separator/three_phase_separator/gas_scrubber: (no special params needed)
  valve/control_valve: outlet_pressure_bara, cv (Cv flow coefficient), percent_valve_opening
  expander: outlet_pressure_bara, isentropic_efficiency
  pump/esp_pump: outlet_pressure_bara, efficiency, head
  mixer: (no params needed, combines streams)
  splitter: split_factor (0-1 fraction to first outlet)
  pipeline/adiabatic_pipe: length (m), diameter (m), roughness (m)
  simple_absorber/simple_teg_absorber: number_of_stages
  ejector, flare, filter, membrane_separator, gas_turbine, tank, adsorber: (type-specific defaults)
  recycle: tolerance
  adjuster: target_variable, target_value

You can combine "add_units" with "changes" in the same scenario (e.g., add a cooler AND change a compressor's pressure).
When the user asks to "add" or "install" equipment, use "add_units". When they ask to "change" or "modify", use "changes".

ADD STREAMS (for adding new inlet streams and mixing them into the process):
Use "add_streams" to create a new stream and insert a mixer after an existing unit.
The mixer combines the upstream outlet with the new stream, then reconnects downstream units.
{{
  "patch": {{
    "add_streams": [
      {{
        "name": "liquid feed",
        "insert_after": "inlet separator",
        "components": {{"nC10": 500.0, "benzene": 300.0, "water": 200.0}},
        "flow_unit": "kg/hr",
        "temperature_C": 30.0,
        "pressure_bara": 30.0,
        "base_stream": "feed gas",
        "mixer_name": "liquid feed mixer"
      }}
    ]
  }}
}}
Notes:
  - "base_stream" is used to clone the thermodynamic model (EOS/mixing rule).
  - If temperature/pressure are not set, the base stream conditions are used.

ADD PROCESS SYSTEM (for adding a group of connected units as a sub-process):
Use "add_process" to insert a whole sub-process (multiple connected units) at once.
This is useful when the engineer asks to "add a dew point control module" or "add a compression train".
{{
  "patch": {{
    "add_process": [
      {{
        "name": "dew point control",
        "insert_after": "inlet separator",
        "units": [
          {{"name": "JT valve", "equipment_type": "valve", "params": {{"outlet_pressure_bara": 45.0}}}},
          {{"name": "LP separator", "equipment_type": "separator", "params": {{}}}},
          {{"name": "recompressor", "equipment_type": "compressor", "params": {{"outlet_pressure_bara": 60.0}}}}
        ]
      }}
    ]
  }}
}}
Each unit in the sub-process is connected in sequence: the first unit's inlet is the outlet of "insert_after",
and subsequent units chain together. The last unit's outlet feeds the unit that previously received from "insert_after".

ADD COMPONENTS (for adding new chemicals to a stream):
Use the "add_components" array inside "patch" to add chemical substances to a stream's fluid.
{{
  "patch": {{
    "add_components": [
      {{
        "stream_name": "feed gas",
        "components": {{
          "nC10": 500.0,
          "benzene": 300.0,
          "water": 200.0
        }},
        "flow_unit": "kg/hr"
      }}
    ]
  }}
}}
Component names must match NeqSim database naming (case-sensitive, lowercase):
  Alkanes: methane, ethane, propane, i-butane, n-butane, i-pentane, n-pentane, 
           n-hexane, n-heptane, n-octane, n-nonane, nC10, nC11, nC12
  Aromatics: benzene, toluene
  Gases: nitrogen, CO2, H2S, oxygen, hydrogen, helium
  Water/glycols: water, MEG, TEG, DEG
  Alcohols: methanol, ethanol
  Plus fractions: C7, C8, C9 (pseudo-components)
IMPORTANT: Use "nC10" NOT "n-decane", use "water" NOT "Water", use "benzene" NOT "Benzene".

ITERATIVE TARGET-SEEKING (for "add X until Y = Z" questions):
When the user wants to achieve a specific output value by adjusting inputs, use "targets" combined 
with "add_components" (or "changes"). The system will automatically iterate (bisection method) to 
converge on the target.
{{
  "patch": {{
    "add_components": [
      {{
        "stream_name": "feed gas",
        "components": {{"nC10": 500.0, "benzene": 300.0, "water": 200.0}},
        "flow_unit": "kg/hr"
      }}
    ],
    "targets": [
      {{
        "target_kpi": "inlet separator.liquidOutStream.flow_kg_hr",
        "target_value": 1000.0,
        "tolerance_pct": 2.0,
        "variable": "component_scale",
        "initial_guess": 1.0,
        "min_value": 0.01,
        "max_value": 50.0,
        "max_iterations": 20
      }}
    ]
  }}
}}
The solver scales ALL component flow rates by the same factor until the target KPI converges.
For the target_kpi, use qualified stream names like "unitName.streamName.property" 
(e.g., "inlet separator.liquidOutStream.flow_kg_hr").
Available KPI suffixes: .flow_kg_hr, .temperature_C, .pressure_bara, .power_kW, .duty_kW

STREAM FLOW SCALING (for "increase/decrease feed flow until Y = Z" questions):
When the user wants to adjust an existing stream's flow rate to hit a target, use
variable="stream_scale" with stream_name. The solver scales the named stream's total
flow rate each iteration (no add_components needed).
{{
  "patch": {{
    "targets": [
      {{
        "target_kpi": "1st stage compressor.power_kW",
        "target_value": 2000.0,
        "tolerance_pct": 1.0,
        "variable": "stream_scale",
        "stream_name": "feed gas",
        "initial_guess": 1.2,
        "min_value": 0.5,
        "max_value": 5.0,
        "max_iterations": 15
      }}
    ]
  }}
}}

UNIT PARAMETER ADJUSTMENT (for "adjust X until Y = Z" questions):
When the user wants to find the right setting of a unit parameter (e.g., compressor outlet
pressure, cooler outlet temperature, valve pressure drop) to achieve a specific target KPI,
use variable="unit_param" with unit_name and unit_param. The solver uses bisection where
min_value/max_value are the parameter bounds (NOT scale factors) and initial_guess is the
starting parameter value.
{{
  "patch": {{
    "targets": [
      {{
        "target_kpi": "1st stage compressor.gasOutStream.temperature_C",
        "target_value": 150.0,
        "tolerance_pct": 1.0,
        "variable": "unit_param",
        "unit_name": "1st stage compressor",
        "unit_param": "outletpressure_bara",
        "initial_guess": 80.0,
        "min_value": 30.0,
        "max_value": 200.0,
        "max_iterations": 20
      }}
    ]
  }}
}}
Use "unit_param" when the user asks things like:
  - "What compressor discharge pressure gives 150°C outlet temperature?"
  - "Adjust the cooler outlet temperature until the compressor power is minimized"
  - "Find the valve outlet pressure that gives 50 kg/hr liquid in the separator"
The unit_param field can be any of the supported patch keys for units (see above).

When you produce a scenario JSON, wait for the simulation results before explaining the impact.
Be concise but thorough in your explanations. Always mention any constraint violations.

RESULTS INTERPRETATION:
When you receive simulation results, ALWAYS:
1. Look at the KEY CHANGES section first — these are KPIs with significant deltas between base and scenario.
2. Report the before→after values for the most important KPIs (power, duty, temperatures, pressures).
3. Check thermodynamic consistency: e.g., lowering compressor inlet temperature MUST reduce compressor power
   (for the same outlet pressure and flow). If results seem inconsistent, say so.
4. Never claim values are "the same" unless the delta is truly zero or negligible (<0.1%).
5. Mention downstream propagation effects — changes to upstream units affect everything downstream.
"""
    system_prompt = system_prompt.replace("__MODEL_SUMMARY__", model_summary)
    system_prompt = system_prompt.replace("__TAG_REF__", tag_ref)
    system_prompt = system_prompt.replace("__EQUIP_TEMPLATES__", equip_templates)
    return system_prompt


# ---------------------------------------------------------------------------
# Builder system prompt (no model loaded — build from scratch)
# ---------------------------------------------------------------------------

def build_builder_system_prompt() -> str:
    """System prompt used when no process model is loaded.

    Teaches the LLM how to output a ``build`` JSON spec that the system
    will use to create a NeqSim process from scratch.
    """
    return r"""You are a process engineering assistant for the NeqSim simulation engine.
No process model is currently loaded. You can help the user BUILD a new process from scratch.

CRITICAL RULES:
1. NEVER invent or estimate numeric results. ALL simulation numbers come from NeqSim.
2. When the user asks to create/build a process, output a ```build ... ``` JSON block (see format below).
3. When the user asks to see the Python script, output: ```build {"action": "show_script"}```
4. When the user asks to save the process, output: ```build {"action": "save"}```
5. Always declare your assumptions (e.g., EOS model choice, default efficiencies).

INTENT CLASSIFICATION:
- BUILD: "Create a gas compression process", "Build a separation train", "Make a process with..."
  → Output a ```build``` block with the full process specification.
- SHOW SCRIPT: "Show me the Python script", "Generate Python code", "Give me the code"
  → Output: ```build {"action": "show_script"}```
- SAVE: "Save the process", "Download the .neqsim file", "Export the model"
  → Output: ```build {"action": "save"}```
- MODIFY: "Add a cooler after the compressor", "Change the pressure to 80 bara"
  → If a process has been built, output a ```build``` block with an "add" array.
- QUESTION: General process engineering questions
  → Answer directly from your knowledge. Do NOT make up simulation numbers.

BUILD SPECIFICATION FORMAT:
```build
{
  "name": "Descriptive Process Name",
  "fluid": {
    "eos_model": "srk",
    "components": {"methane": 0.85, "ethane": 0.07, "propane": 0.03, "CO2": 0.02, "nitrogen": 0.03},
    "composition_basis": "mole_fraction",
    "temperature_C": 25.0,
    "pressure_bara": 50.0,
    "total_flow": 10000,
    "flow_unit": "kg/hr"
  },
  "process": [
    {"name": "feed gas", "type": "stream"},
    {"name": "inlet separator", "type": "separator"},
    {"name": "1st stage compressor", "type": "compressor", "params": {"outlet_pressure_bara": 100.0, "isentropic_efficiency": 0.75}},
    {"name": "intercooler", "type": "cooler", "params": {"outlet_temperature_C": 35.0, "pressure_drop_bar": 0.5}},
    {"name": "export scrubber", "type": "gas_scrubber"}
  ]
}
```

SPECIFICATION RULES:
- The first entry in "process" should always be a stream (the feed).
- Each subsequent unit automatically receives the previous unit's outlet stream as its inlet.
- For separators, the gas outlet feeds the next unit by default.
  Add "outlet": "liquid" to follow the liquid branch instead.
  Add "outlet": "water" for the water outlet of 3-phase separators.
- Use "inlet": "unit_name" to chain from a specific earlier unit instead of the previous one.
  Use "inlet": "unit_name.getLiquidOutStream" for a specific outlet.

AVAILABLE EOS MODELS:
- "srk" — SRK equation of state (default, good for gas processing)
- "pr" or "pr78" — Peng-Robinson (alternative cubic EOS)
- "cpa" or "cpa-srk" — CPA-SRK (required for polar: water, MEG, methanol)
- "cpa-pr" — CPA-Peng-Robinson
- "umr-pru" — UMR-PRU (accurate phase envelopes)
- "gerg2008" — GERG-2008 (fiscal metering, custody transfer)
- "pcsaft" — PC-SAFT
- "ideal" — Ideal gas (testing only)
Choose CPA if the fluid contains water, MEG, TEG, or methanol at significant amounts.
Choose SRK or PR for standard hydrocarbon processing.

AVAILABLE EQUIPMENT TYPES:
- stream — Feed or intermediate stream
- separator, two_phase_separator, three_phase_separator, gas_scrubber — Phase separation
- compressor — Gas compression
- cooler, heater, air_cooler, water_cooler — Temperature control
- heat_exchanger — Heat recovery
- valve, control_valve — Pressure reduction (JT effect)
- expander — Turbo-expander (power recovery)
- pump — Liquid pressurization
- mixer — Combine streams
- splitter — Split stream (use split_factor param)
- pipeline, adiabatic_pipe — Pipeline pressure drop
- simple_absorber, simple_teg_absorber — Gas dehydration/sweetening
- gibbs_reactor — Chemical equilibrium reactor
- recycle — Recycle stream (convergence loop)

PARAMETER REFERENCE:
  compressor: outlet_pressure_bara, isentropic_efficiency (0-1, default 0.75),
              polytropic_efficiency, speed, compression_ratio, use_polytropic_calc
  cooler/heater: outlet_temperature_C, pressure_drop_bar, duty_kW
  valve: outlet_pressure_bara, cv, percent_valve_opening
  expander: outlet_pressure_bara, isentropic_efficiency
  pump: outlet_pressure_bara, efficiency, head
  splitter: split_factor (0-1)
  pipeline: length, diameter, roughness
  separator: (no required params — sizing is automatic)

COMPONENT NAMES (NeqSim database — use exactly these):
  Alkanes: methane, ethane, propane, i-butane, n-butane, i-pentane, n-pentane,
           n-hexane, n-heptane, n-octane, n-nonane, nC10, nC11
  Aromatics: benzene, toluene
  Gases: nitrogen, CO2, H2S, oxygen, hydrogen, helium
  Water/glycols: water, MEG, TEG, DEG
  Alcohols: methanol, ethanol
  Plus fractions: C7, C8, C9

INCREMENTAL ADDITIONS (after a process has been built):
To add equipment to an existing built process:
```build
{
  "add": [
    {"name": "new cooler", "type": "cooler", "insert_after": "compressor 1",
     "params": {"outlet_temperature_C": 40.0}}
  ]
}
```

DESIGN GUIDELINES (use these defaults unless the user specifies otherwise):
- Compressor isentropic efficiency: 0.75 (centrifugal), 0.80 (reciprocating)
- Compressor max pressure ratio per stage: ~3-4 for centrifugal
- Compressor max discharge temperature: 150°C (add intercooling if exceeded)
- Cooler outlet temperature: 35°C (tropical), 25°C (cold climate)
- Pressure drop across coolers/heaters: 0.2-0.5 bar
- Separator: no params needed — NeqSim auto-sizes
- For multi-stage compression: Add intercoolers between stages
- For gas processing: inlet separator → compression → cooling → export scrubber

When the user describes a process, design it with appropriate engineering defaults and explain your choices.
After the process is built, explain the key results (power, duty, temperatures, pressures).
"""


# ---------------------------------------------------------------------------
# Chat message processing
# ---------------------------------------------------------------------------

def extract_scenario_json(text: str) -> Optional[dict]:
    """
    Extract a JSON scenario specification from LLM output text.
    Looks for ```json ... ``` blocks.
    """
    import re
    pattern = r'```json\s*\n(.*?)\n\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            data = json.loads(match)
            if "scenarios" in data:
                return data
        except json.JSONDecodeError:
            continue
    
    # Also try the entire text as JSON
    try:
        data = json.loads(text)
        if "scenarios" in data:
            return data
    except json.JSONDecodeError:
        pass

    return None


def extract_property_query(text: str) -> Optional[dict]:
    """
    Extract a property query specification from LLM output text.
    Looks for ```query ... ``` blocks with JSON containing {"properties": ...}.
    """
    import re
    pattern = r'```query\s*\n(.*?)\n\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            data = json.loads(match)
            if "properties" in data:
                return data
        except json.JSONDecodeError:
            continue
    
    return None


def extract_build_spec(text: str) -> Optional[dict]:
    """Extract a process build specification from LLM output.

    Looks for ````build ... ```` blocks with JSON.
    Returns the parsed dict or ``None``.
    """
    import re
    pattern = r'```build\s*\n(.*?)\n\s*```'
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        try:
            data = json.loads(match)
            return data
        except json.JSONDecodeError:
            continue

    return None


def format_comparison_for_llm(comparison) -> str:
    """
    Format a scenario comparison result into text the LLM can use
    to explain results to the engineer.
    
    Filters to only significant KPI changes (|delta| > 0.01 or delta_pct > 0.1%)
    to keep the text concise and avoid wasting LLM tokens.
    """
    from .scenario_engine import results_summary_table
    
    lines = []

    # Key changes — highlight KPIs that changed significantly
    if comparison.delta_kpis:
        sig_changes = [d for d in comparison.delta_kpis 
                       if d.get('delta') is not None and abs(d['delta']) > 0.01]
        if sig_changes:
            lines.append("=== KEY CHANGES (base → scenario) ===")
            # Sort by magnitude of change and limit to top 50 most significant
            sig_changes.sort(key=lambda x: abs(x.get('delta_pct', 0) or 0), reverse=True)
            shown = sig_changes[:50]
            for d in shown:
                pct_str = f" ({d['delta_pct']:+.1f}%)" if d.get('delta_pct') is not None else ""
                lines.append(f"  {d['kpi']}: {d['base']:.2f} → {d['case']:.2f} [{d['unit']}] "
                             f"(delta={d['delta']:+.2f}{pct_str})")
            if len(sig_changes) > 50:
                lines.append(f"  ... and {len(sig_changes) - 50} more changes")
            lines.append("")
    
    # Summary table — only include KPIs that changed or are key process indicators
    summary_df = results_summary_table(comparison)
    if not summary_df.empty:
        # Filter to important KPIs: those with changes, total_power/duty, mass_balance, 
        # equipment power/duty, and a few key stream properties
        important_prefixes = ('total_', 'mass_balance')
        important_suffixes = ('.power_kW', '.duty_kW')
        
        def is_important_kpi(row):
            kpi = row.get('KPI', '')
            # Always include summary KPIs
            if kpi.startswith(important_prefixes):
                return True
            if kpi.endswith(important_suffixes):
                return True
            # Include KPIs that changed between base and any case column
            base_val = row.get('BASE')
            for col in summary_df.columns:
                if col not in ('KPI', 'Unit', 'BASE'):
                    case_val = row.get(col)
                    if base_val is not None and case_val is not None:
                        try:
                            if abs(float(base_val) - float(case_val)) > 0.01:
                                return True
                        except (ValueError, TypeError):
                            pass
            return False
        
        filtered_df = summary_df[summary_df.apply(is_important_kpi, axis=1)]
        if not filtered_df.empty:
            lines.append("=== SIMULATION RESULTS (changed KPIs) ===")
            lines.append(filtered_df.to_string(index=False))
    
    # Constraint check
    if comparison.constraint_summary:
        lines.append("\n=== CONSTRAINTS ===")
        for c in comparison.constraint_summary:
            status_icon = {"OK": "✓", "WARN": "⚠", "VIOLATION": "✗"}.get(c["status"], "?")
            lines.append(f"  {status_icon} [{c['status']}] {c['constraint']}: {c['detail']}")
    
    # Patch log
    if comparison.patch_log:
        lines.append("\n=== APPLIED CHANGES ===")
        for entry in comparison.patch_log:
            status = entry.get("status", "?")
            if status in ("CONVERGED", "BEST_EFFORT"):
                # Iterative solver result
                lines.append(f"  🔄 {entry['key']}: {entry.get('value', '?')}")
                lines.append(f"     Iterations: {entry.get('iterations', '?')}")
                # Show iteration log summary
                iter_log = entry.get("iteration_log", [])
                if iter_log:
                    for il in iter_log:
                        icon = "✓" if il.get("status") == "CONVERGED" else "→"
                        kpi_val = il.get("kpi_value", "?")
                        err = il.get("error_pct", "?")
                        lines.append(f"     {icon} iter {il.get('iteration', '?')}: "
                                     f"scale={il.get('scale', '?')}, "
                                     f"value={kpi_val}, error={err}%")
            elif status == "OK":
                lines.append(f"  ✓ {entry['key']} = {entry.get('value', '?')}")
                if "scale_factor" in entry:
                    lines.append(f"    (scale factor: {entry['scale_factor']})")
            else:
                lines.append(f"  ✗ {entry['key']}: {entry.get('error', 'unknown error')}")
    
    # Failed scenarios
    for case in comparison.cases:
        if not case.success:
            lines.append(f"\n⚠ Scenario '{case.scenario.name}' FAILED: {case.error}")
    
    return "\n".join(lines)


class ProcessChatSession:
    """
    Manages a chat session with a NeqSim process model.
    
    Uses Gemini to interpret engineer questions and orchestrate
    simulation runs via the scenario engine.

    Supports two modes:
      - **Model mode**: A process model is loaded (uploaded or built).
      - **Builder mode**: No model loaded; the LLM can build one from scratch.
    """

    def __init__(
        self,
        model: Optional[NeqSimProcessModel] = None,
        api_key: str = "",
        ai_model: str = "gemini-2.0-flash",
    ):
        self.model = model
        self.api_key = api_key
        self.ai_model = ai_model
        self.history: List[Dict[str, str]] = []
        self._last_comparison = None
        self._builder = None       # ProcessBuilder instance (when building)
        self._last_script = None   # Last generated Python script
        self._last_save_bytes = None  # Last generated .neqsim bytes

        # Build appropriate system prompt
        if model is not None:
            self._system_prompt = build_system_prompt(model)
        else:
            self._system_prompt = build_builder_system_prompt()

    # -- Main chat entry point ----------------------------------------------

    def chat(self, user_message: str) -> str:
        """
        Process a user message and return the assistant response.
        
        Handles:
          - Build specs (```build```) → build process from scratch
          - Scenario JSON (```json```) → run what-if scenarios
          - Property queries (```query```) → extract simulation data
          - Plain Q&A → direct LLM response
        """
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)

        # Build conversation
        self.history.append({"role": "user", "content": user_message})

        # First LLM call: classify intent
        contents = self._build_contents()
        
        response = client.models.generate_content(
            model=self.ai_model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=self._system_prompt,
                temperature=0.3,
            ),
        )

        assistant_text = response.text

        # --- Check for build spec ---
        build_spec = extract_build_spec(assistant_text)
        if build_spec:
            return self._handle_build(assistant_text, build_spec, client, types)

        # --- Check for scenario JSON ---
        scenario_data = extract_scenario_json(assistant_text)
        if scenario_data and self.model:
            return self._handle_scenario(assistant_text, scenario_data, client, types)

        # --- Check for property query ---
        property_query = extract_property_query(assistant_text)
        if property_query and self.model:
            return self._handle_property_query(assistant_text, property_query, client, types)

        # --- Pure Q&A ---
        self.history.append({"role": "assistant", "content": assistant_text})
        self._last_comparison = None
        return assistant_text

    # -- Build handling -----------------------------------------------------

    def _handle_build(self, assistant_text: str, build_spec: dict, client, types) -> str:
        """Process a build specification from the LLM."""
        from .process_builder import ProcessBuilder

        action = build_spec.get("action")

        # --- Action: show_script ---
        if action == "show_script":
            if self._builder and self._builder.spec:
                script = self._builder.to_python_script()
                self._last_script = script
                self.history.append({"role": "assistant", "content": assistant_text})
                self.history.append({
                    "role": "user",
                    "content": f"[SYSTEM: Python script generated. Show it to the engineer.]\n\n```python\n{script}\n```"
                })
                return self._llm_followup(client, types)
            else:
                self.history.append({"role": "assistant", "content": assistant_text})
                self.history.append({
                    "role": "user",
                    "content": "[SYSTEM: No process has been built yet. Ask the user to build a process first.]"
                })
                return self._llm_followup(client, types)

        # --- Action: save ---
        if action == "save":
            if self._builder and self._builder.model:
                raw = self._builder.save_neqsim_bytes()
                self._last_save_bytes = raw
                self.history.append({"role": "assistant", "content": assistant_text})
                self.history.append({
                    "role": "user",
                    "content": "[SYSTEM: The .neqsim file has been prepared. A download button is now available in the sidebar. Tell the engineer they can download it.]"
                })
                return self._llm_followup(client, types)
            elif self.model:
                # Model was uploaded — save current state
                import neqsim, tempfile, os
                proc = self.model.get_process()
                with tempfile.NamedTemporaryFile(suffix=".neqsim", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    neqsim.save_neqsim(proc, tmp_path)
                    with open(tmp_path, "rb") as f:
                        self._last_save_bytes = f.read()
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                self.history.append({"role": "assistant", "content": assistant_text})
                self.history.append({
                    "role": "user",
                    "content": "[SYSTEM: The .neqsim file has been prepared from the current model. A download button is now available. Tell the engineer.]"
                })
                return self._llm_followup(client, types)
            else:
                self.history.append({"role": "assistant", "content": assistant_text})
                self.history.append({
                    "role": "user",
                    "content": "[SYSTEM: No process to save. Ask the user to build or upload one first.]"
                })
                return self._llm_followup(client, types)

        # --- Build a new process ---
        if "fluid" in build_spec and "process" in build_spec:
            try:
                builder = ProcessBuilder()
                model = builder.build_from_spec(build_spec)

                self._builder = builder
                self.model = model
                # Rebuild system prompt now that we have a model
                self._system_prompt = build_system_prompt(model)

                # Prepare summary for LLM
                summary = model.get_model_summary()
                build_log = "\n".join(builder.build_log)

                self.history.append({"role": "assistant", "content": assistant_text})
                self.history.append({
                    "role": "user",
                    "content": (
                        f"[SYSTEM: Process built successfully! Build log:\n{build_log}\n\n"
                        f"Model summary:\n{summary}\n\n"
                        "Explain the built process and key results to the engineer. "
                        "Mention the equipment, key temperatures/pressures, power consumption, and duties. "
                        "Tell them they can now ask what-if questions, request the Python script, or save the .neqsim file.]"
                    )
                })
                return self._llm_followup(client, types)

            except Exception as e:
                self.history.append({"role": "assistant", "content": assistant_text})
                self.history.append({
                    "role": "user",
                    "content": (
                        f"[SYSTEM: Process build FAILED: {str(e)}\n"
                        f"{traceback.format_exc()}\n"
                        "Inform the engineer and suggest corrections.]"
                    )
                })
                return self._llm_followup(client, types)

        # --- Incremental additions ---
        if "add" in build_spec and self.model:
            try:
                from .scenario_engine import apply_add_units
                from .patch_schema import AddUnitOp

                add_list = build_spec["add"]
                add_ops = [
                    AddUnitOp(
                        name=a["name"],
                        equipment_type=a["type"],
                        insert_after=a["insert_after"],
                        params=a.get("params", {}),
                    )
                    for a in add_list
                ]

                log = apply_add_units(self.model, add_ops)
                failed = [e for e in log if e.get("status") == "FAILED"]
                if failed:
                    raise RuntimeError(f"Add unit errors: {failed}")

                # Re-run, re-index, refresh source bytes
                NeqSimProcessModel._run_until_converged(self.model.get_process())
                self.model._index_model_objects()
                self.model.refresh_source_bytes()

                # Update system prompt
                self._system_prompt = build_system_prompt(self.model)

                summary = self.model.get_model_summary()
                log_str = "\n".join(str(e) for e in log)

                self.history.append({"role": "assistant", "content": assistant_text})
                self.history.append({
                    "role": "user",
                    "content": (
                        f"[SYSTEM: Equipment added successfully. Log:\n{log_str}\n\n"
                        f"Updated model summary:\n{summary}\n\n"
                        "Explain the changes and updated results to the engineer.]"
                    )
                })
                return self._llm_followup(client, types)

            except Exception as e:
                self.history.append({"role": "assistant", "content": assistant_text})
                self.history.append({
                    "role": "user",
                    "content": f"[SYSTEM: Add equipment failed: {str(e)}. Inform the engineer.]"
                })
                return self._llm_followup(client, types)

        # Unknown build spec — just pass through
        self.history.append({"role": "assistant", "content": assistant_text})
        self._last_comparison = None
        return assistant_text

    # -- Scenario handling --------------------------------------------------

    def _handle_scenario(self, assistant_text: str, scenario_data: dict, client, types) -> str:
        """Execute scenario JSON and feed results back to LLM.

        If the scenario contains structural additions (add_units, add_streams,
        add_process), those changes are also applied to the base model so they
        persist for subsequent interactions.
        """
        try:
            scenarios = scenarios_from_json(scenario_data)

            from .scenario_engine import run_scenarios
            comparison = run_scenarios(self.model, scenarios)

            results_text = format_comparison_for_llm(comparison)

            # --- Persist structural additions to the base model ---
            structural_applied = False
            for sc in scenarios:
                has_structural = (
                    sc.patch.add_units or sc.patch.add_streams or sc.patch.add_process
                )
                if not has_structural:
                    continue
                # Only persist if the scenario actually succeeded
                case_ok = any(
                    c.success and c.scenario.name == sc.name
                    for c in comparison.cases
                )
                if not case_ok:
                    continue
                try:
                    from .scenario_engine import (
                        apply_add_units, apply_add_streams, apply_add_process,
                        apply_patch_to_model,
                    )
                    if sc.patch.add_units:
                        apply_add_units(self.model, sc.patch.add_units)
                    if sc.patch.add_process:
                        apply_add_process(self.model, sc.patch.add_process)
                    if sc.patch.add_streams:
                        apply_add_streams(self.model, sc.patch.add_streams)
                    if sc.patch.changes:
                        apply_patch_to_model(self.model, sc.patch)
                    # Re-run, re-index, and refresh source bytes so clones
                    # see the updated topology
                    NeqSimProcessModel._run_until_converged(
                        self.model.get_process()
                    )
                    self.model._index_model_objects()
                    self.model.refresh_source_bytes()
                    self._system_prompt = build_system_prompt(self.model)
                    structural_applied = True
                except Exception:
                    pass  # comparison still valid even if persistence fails

            self.history.append({"role": "assistant", "content": assistant_text})

            persist_note = ""
            if structural_applied:
                persist_note = (
                    "\n\n[SYSTEM NOTE: The structural additions (new equipment/streams) "
                    "have been applied to the base model. Future questions will see "
                    "the updated topology.]\n\nUpdated model summary:\n"
                    + self.model.get_model_summary()
                )

            self.history.append({
                "role": "user",
                "content": (
                    f"[SYSTEM: Simulation completed. Results below. "
                    f"Explain these results to the engineer concisely.]\n\n"
                    f"{results_text}{persist_note}"
                )
            })

            final_text = self._llm_followup(client, types)
            self._last_comparison = comparison
            return final_text

        except Exception as e:
            self.history.append({"role": "assistant", "content": assistant_text})
            self.history.append({
                "role": "user",
                "content": f"[SYSTEM: Simulation failed with error: {str(e)}. Please inform the engineer and suggest corrections.]"
            })
            return self._llm_followup(client, types)

    # -- Property query handling --------------------------------------------

    def _handle_property_query(self, assistant_text: str, property_query: dict, client, types) -> str:
        """Run model and extract matching properties."""
        try:
            queries = property_query.get("properties", [])
            all_results = []
            cached_result = self.model.run()
            for q in queries:
                result_text = self.model.query_properties(q, _cached_result=cached_result)
                all_results.append(result_text)

            properties_text = "\n\n".join(all_results)

            self.history.append({"role": "assistant", "content": assistant_text})
            self.history.append({
                "role": "user",
                "content": f"[SYSTEM: Property query completed. Results below. Present these results clearly to the engineer.]\n\n{properties_text}"
            })

            final_text = self._llm_followup(client, types)
            self._last_comparison = None
            return final_text

        except Exception as e:
            self.history.append({"role": "assistant", "content": assistant_text})
            self.history.append({
                "role": "user",
                "content": f"[SYSTEM: Property query failed: {str(e)}. Inform the engineer.]"
            })
            return self._llm_followup(client, types)

    # -- Helpers ------------------------------------------------------------

    def _llm_followup(self, client, types) -> str:
        """Make a follow-up LLM call and record the response."""
        contents = self._build_contents()
        response = client.models.generate_content(
            model=self.ai_model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=self._system_prompt,
                temperature=0.3,
            ),
        )
        final_text = response.text
        self.history.append({"role": "assistant", "content": final_text})
        return final_text

    def get_last_comparison(self):
        """Get the last scenario comparison result (for UI display)."""
        return getattr(self, "_last_comparison", None)

    def get_last_script(self) -> Optional[str]:
        """Get the last generated Python script."""
        return self._last_script

    def get_last_save_bytes(self) -> Optional[bytes]:
        """Get the last generated .neqsim file bytes."""
        return self._last_save_bytes

    def get_builder(self):
        """Get the ProcessBuilder instance if process was built from scratch."""
        return self._builder

    def _build_contents(self) -> list:
        """Build Gemini API contents from chat history."""
        contents = []
        for msg in self.history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        return contents

    def reset(self):
        """Clear chat history."""
        self.history.clear()
        self._last_comparison = None
        self._last_script = None
        self._last_save_bytes = None

