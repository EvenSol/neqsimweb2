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
from .optimizer import optimize_production, format_optimization_result, OptimizationResult
from .risk_analysis import run_risk_analysis, format_risk_result, RiskAnalysisResult
from .compressor_chart import (
    generate_compressor_chart,
    format_chart_result,
    CompressorChartResult,
    refresh_operating_point,
)
from .auto_size import auto_size_all, format_autosize_result, AutoSizeResult
from .emissions import calculate_emissions, format_emissions_result, EmissionsResult
from .dynamic_sim import run_dynamic_simulation, format_dynamic_result, DynamicSimResult
from .sensitivity import run_sensitivity_analysis, format_sensitivity_result, SensitivityResult
from .pvt_simulation import run_pvt_simulation, format_pvt_result, PVTResult
from .safety_systems import run_safety_analysis, format_safety_result, SafetyReport
from .flow_assurance import run_flow_assurance, format_flow_assurance_result, FlowAssuranceResult


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

IMPORTANT: Do NOT use "changes" to modify stream composition. "streams.<name>.components" is NOT a valid
patch key. To add or replace components in a stream, ALWAYS use "add_components" inside "patch":
{{
  "patch": {{
    "add_components": [
      {{
        "stream_name": "feed gas",
        "components": {{"water": 500.0}},
        "flow_unit": "kg/hr"
      }}
    ]
  }}
}}
This adds new components (or increases the flow of existing ones) by absolute flow rate.
You CANNOT set mole fractions directly — only absolute flow rates (e.g., kg/hr, mol/sec).
To set 50 mol% water in a feed that is 1000 kg/hr methane, calculate the equivalent water mass flow.

SCRIPT AND SAVE COMMANDS:
When the user asks to see the Python script, output a ```build``` block:
```build
{{"action": "show_script"}}
```
When the user asks to save/download the process, output:
```build
{{"action": "save"}}
```
Do NOT write Python scripts yourself — ALWAYS use the show_script action so the system generates the correct script.

IMPORTANT — AVOID UNNECESSARY REBUILDS:
Compressor charts, mechanical design, and auto-sizing data are expensive to compute and
are lost when the process is rebuilt from scratch. To preserve them:
  - To UPDATE PROPERTIES on existing equipment: use scenario JSON with "changes" (above).
  - To ADD EQUIPMENT at the start/end: use ```build {{"add": [...]}}```.
  - Do NOT regenerate compressor charts unless the user explicitly asks to generate a new chart.
    When the user just updates flow or conditions, the existing chart is automatically used to
    recalculate the operating point.
  - Only emit a full build spec (with "fluid" and "process") when building a brand-new process.

PROCESS OPTIMIZATION (for "find maximum production", "maximize throughput", "what is the max flow?"):
When the user asks to optimize, find maximum production, or maximize throughput, output an ```optimize ... ``` block:
```optimize
{{
  "objective": "maximize_throughput",
  "feed_stream": "feed gas",
  "min_flow_kg_hr": 1000,
  "max_flow_kg_hr": 500000,
  "utilization_limit": 1.0,
  "tolerance_pct": 1.0,
  "max_iterations": 25
}}
```
Parameters:
  - objective: "maximize_throughput" (find max feed flow before equipment limits are hit)
  - feed_stream: name of the feed stream to scale (auto-detected if omitted)
  - min_flow_kg_hr: lower bound for flow search (default: 10% of current flow)
  - max_flow_kg_hr: upper bound for flow search (default: 500% of current flow)
  - utilization_limit: max equipment utilization ratio, 0-1 (default: 1.0 = 100%)
  - tolerance_pct: convergence tolerance as % of range (default: 1.0)
  - max_iterations: max search iterations (default: 25)

The optimizer will:
  1. Scale the feed flow rate using golden-section search
  2. At each step, run the full process simulation
  3. Check equipment utilization (compressor surge/power, separator capacity, etc.)
  4. Find the maximum flow where all equipment stays within limits
  5. Report the optimal flow, bottleneck equipment, and utilization breakdown

Use this for questions like:
  - "Find maximum production for this process"
  - "What is the maximum throughput?"
  - "How much can we increase the feed flow?"
  - "Optimize the flow rate"
  - "What is the bottleneck equipment?"
  - "Find max flow before equipment limits are exceeded"

RISK ANALYSIS (for "risk matrix", "equipment criticality", "availability analysis", "what if equipment fails?"):
When the user asks about risk, reliability, availability, failure analysis, or equipment criticality, output a ```risk ... ``` block:
```risk
{{
  "analysis": "full",
  "product_stream": "export gas",
  "feed_stream": "feed gas",
  "mc_iterations": 1000,
  "mc_days": 365,
  "include_degraded": true
}}
```
Parameters:
  - analysis: "full" (complete risk analysis with criticality + risk matrix + Monte Carlo)
  - product_stream: name of the product/export stream (auto-detected if omitted)
  - feed_stream: name of the feed stream (auto-detected if omitted)
  - mc_iterations: number of Monte Carlo iterations (default: 1000)
  - mc_days: simulation horizon in days (default: 365)
  - include_degraded: whether to include degraded failure modes (default: true)

The risk analysis will:
  1. Assign OREDA-based reliability data (MTTF, MTTR) to each equipment
  2. Simulate equipment trips (shutdown each unit one-by-one, measure production loss)
  3. Simulate degraded operation (reduced capacity, measure impact)
  4. Build a 5×5 risk matrix (probability from failure rate, consequence from production loss)
  5. Run Monte Carlo availability simulation (stochastic failures over 1 year)
  6. Rank equipment by criticality index

Use this for questions like:
  - "Show the risk matrix for this process"
  - "What is the equipment criticality ranking?"
  - "How reliable is this process?"
  - "What is the system availability?"
  - "What happens if the compressor fails?"
  - "Run a failure analysis"
  - "What are the most critical equipment items?"
  - "Run Monte Carlo availability simulation"

COMPRESSOR CHART (for "generate compressor map", "show compressor performance curve", "compressor chart"):
When the user asks to generate or show a compressor performance chart/map/curve, output a ```chart ... ``` block:
```chart
{{
  "compressor": "1st stage compressor",
  "template": "CENTRIFUGAL_STANDARD",
  "num_speeds": 5,
  "show_only": false
}}
```
Parameters:
  - compressor: name of the compressor (or "all" for all compressors). Auto-detected if omitted.
  - template: chart template (default: "CENTRIFUGAL_STANDARD"). Options:
      Basic: CENTRIFUGAL_STANDARD, CENTRIFUGAL_HIGH_FLOW, CENTRIFUGAL_HIGH_HEAD
      Application: PIPELINE, EXPORT, INJECTION, GAS_LIFT, REFRIGERATION, BOOSTER
      Type: SINGLE_STAGE, MULTISTAGE_INLINE, INTEGRALLY_GEARED, OVERHUNG
  - num_speeds: number of speed curves to generate (default: 5)
  - show_only: true = reuse existing chart (update operating point only), false = generate new chart.
    IMPORTANT: If a chart has already been generated for this compressor (e.g. via auto-size or a previous
    "generate chart" request), set "show_only": true to reuse it. Only set false when the user explicitly
    asks to GENERATE or CREATE a NEW chart, or requests a different template.

This generates a CompressorChart, applies it to the compressor, and re-runs the simulation.
The chart enforces surge/stonewall/speed limits during all subsequent calculations.

Use this for questions like:
  - "Generate a compressor chart for the 1st stage compressor" → show_only: false
  - "Show the compressor performance map" → show_only: true (reuse existing)
  - "Show me the chart for 23-KA-01" → show_only: true (reuse existing)
  - "Create a compressor curve using pipeline template" → show_only: false (new template)
  - "Generate compressor charts for all compressors" → show_only: false

AUTO-SIZE (for "auto size equipment", "equipment sizing", "utilization report", "bottleneck analysis"):
When the user asks to auto-size equipment, get a sizing report, check utilization, or find bottlenecks, output an ```autosize ... ``` block:
```autosize
{{
  "safety_factor": 1.2,
  "generate_charts": true,
  "chart_template": "CENTRIFUGAL_STANDARD",
  "force_resize": false
}}
```
Parameters:
  - safety_factor: design safety factor (default: 1.2 = 20% margin)
  - generate_charts: whether to generate compressor charts (default: true)
  - chart_template: compressor chart template (default: "CENTRIFUGAL_STANDARD")
  - force_resize: ALWAYS false unless the user explicitly says "re-size", "redo sizing", or
    "recalculate sizing". Phrases like "auto size", "auto size all", "size equipment", or
    "generate sizing" do NOT mean re-size — they mean use existing sizing if available.
    Only set true when the user literally uses words like RE-size, REDO, or wants a DIFFERENT
    safety factor than before.

The auto-sizing will:
  1. Apply autoSize() to all sizeable equipment (separators, valves, heaters, coolers, etc.)
     — skipping equipment already sized unless force_resize is true
  2. Optionally generate compressor performance charts
  3. Extract sizing data (dimensions, capacities, design values)
  4. Calculate utilization ratios for all equipment
  5. Identify the process bottleneck

Use this for questions like:
  - "Auto-size all equipment" → force_resize: false
  - "Auto size all" → force_resize: false
  - "Size equipment" → force_resize: false
  - "What is the equipment sizing?" → force_resize: false
  - "Show equipment utilization" → force_resize: false
  - "What is the process bottleneck?" → force_resize: false
  - "RE-size all equipment" → force_resize: true (only when "re-size" or "redo sizing" is used)
  - "Redo sizing with 30% safety factor" → force_resize: true
  - "Generate a sizing report"
  - "What is the utilization of each piece of equipment?"

EMISSIONS ANALYSIS (for "CO2 emissions", "carbon footprint", "environmental impact", "fuel gas consumption"):
When the user asks about emissions, carbon footprint, environmental impact, or fuel consumption, output an ```emissions ... ``` block:
```emissions
{{
  "product_stream": "export gas",
  "include_fugitives": true,
  "flare_streams": [],
  "power_source": "gas_turbine"
}}
```
Parameters:
  - product_stream: name of the product stream for intensity calculation (auto-detected if omitted)
  - include_fugitives: include fugitive emissions estimates (default: true)
  - flare_streams: list of stream names routed to flare (default: [])
  - power_source: "gas_turbine" or "electric" (default: "gas_turbine")

Use this for questions like:
  - "What are the CO2 emissions?"
  - "Calculate the carbon footprint"
  - "What is the emission intensity?"
  - "How much fuel gas is consumed?"
  - "Show the environmental impact"

DYNAMIC SIMULATION (for "blowdown", "startup", "shutdown", "transient", "time response"):
When the user asks about transient behavior, blowdown, startup/shutdown sequences, output a ```dynamic ... ``` block:

For TRUE TRANSIENT simulations (valve closures, step changes, upset conditions) — uses NeqSim's native runTransient() with dynamic mode on separators and valves:
```dynamic
{{
  "scenario_type": "transient",
  "changes": [
    {{"unit": "VLV-100", "property": "percentValveOpening", "value": 10}},
    {{"unit": "VLV-102", "property": "percentValveOpening", "value": 10}}
  ],
  "duration_s": 50,
  "n_steps": 10,
  "dt": 5.0
}}
```
The "changes" array specifies what to change before stepping. Supported properties:
  - percentValveOpening (0-100%)
  - cv (valve Cv)
  - outletPressure (bara)
  - flow (kg/hr, for streams)
  - speed (RPM, for compressors)
  - temperature (°C)
  - pressure (bara)

For blowdown:
```dynamic
{{
  "scenario_type": "blowdown",
  "vessel_name": "inlet separator",
  "initial_pressure_bara": 50.0,
  "final_pressure_bara": 1.013,
  "orifice_diameter_mm": 50.0,
  "duration_s": 600,
  "n_steps": 50
}}
```
For startup/ramp scenarios:
```dynamic
{{
  "scenario_type": "startup",
  "stream_name": "feed gas",
  "start_value": 1000,
  "end_value": 50000,
  "duration_s": 3600,
  "n_steps": 20
}}
```
Parameters:
  - scenario_type: "transient", "blowdown", "startup", "shutdown", or "ramp"
  - changes: (transient only) array of unit/property/value changes to apply
  - dt: (transient only) time step in seconds
  - vessel_name: for blowdown — name of vessel (auto-detected if omitted)
  - stream_name: for ramp/startup — name of stream to ramp
  - duration_s: simulation duration in seconds
  - n_steps: number of time steps

IMPORTANT: For "transient" scenario_type, separators, tanks, and valves are automatically set to dynamic mode
(setCalculateSteadyState(false)) so they track holdup, levels, and Cv-based flow. Equipment sizes are NOT
recalculated — only the process response (flows, pressures, levels) changes over time.

Use this for: "simulate blowdown", "startup sequence", "what happens during shutdown?", "transient response",
"close valves to 10%", "run N transient steps", "step change", "upset scenario"

SENSITIVITY ANALYSIS (for "sensitivity study", "parameter sweep", "tornado chart", "what-if matrix"):
When the user asks for parameter sensitivity, sweep studies, or tornado charts, output a ```sensitivity ... ``` block:

For single-variable sweep:
```sensitivity
{{
  "analysis_type": "single_sweep",
  "variable": "streams.feed gas.pressure_bara",
  "min_value": 30.0,
  "max_value": 80.0,
  "n_points": 10,
  "response_kpis": ["power", "temperature"]
}}
```

For tornado analysis:
```sensitivity
{{
  "analysis_type": "tornado",
  "variables": [
    {{"name": "streams.feed gas.pressure_bara", "low": 30, "high": 80}},
    {{"name": "streams.feed gas.temperature_C", "low": 10, "high": 50}},
    {{"name": "units.compressor.isentropicEfficiency", "low": 0.65, "high": 0.85}}
  ],
  "response_kpi": "compressor.power_kW"
}}
```

For two-variable surface:
```sensitivity
{{
  "analysis_type": "two_variable",
  "variable": "streams.feed gas.pressure_bara",
  "min_value": 30, "max_value": 80, "n_points": 5,
  "variable_2": "streams.feed gas.temperature_C",
  "min_2": 10, "max_2": 50, "n_2": 5,
  "response_kpis": ["power"]
}}
```

Use this for: "run sensitivity analysis", "how sensitive is power to pressure?", "tornado chart for compressor power",
"sweep feed pressure from 30 to 80 bara", "parameter study"

PVT SIMULATION (for "PVT study", "CME", "differential liberation", "saturation point", "bubble point"):
When the user asks about PVT experiments, output a ```pvt ... ``` block:
```pvt
{{
  "experiment": "CME",
  "stream_name": "feed gas",
  "temperature_C": 100.0,
  "p_start_bara": 400.0,
  "p_end_bara": 10.0,
  "n_steps": 20
}}
```
Available experiments: "CME" (Constant Mass Expansion), "CVD" (Constant Volume Depletion),
"DL" (Differential Liberation), "separator_test", "saturation_point"

For separator test:
```pvt
{{
  "experiment": "separator_test",
  "stream_name": "feed gas",
  "stages": [
    {{"temperature_C": 40.0, "pressure_bara": 50.0}},
    {{"temperature_C": 30.0, "pressure_bara": 10.0}},
    {{"temperature_C": 15.0, "pressure_bara": 1.013}}
  ]
}}
```

Use this for: "run CME experiment", "CVD experiment", "constant volume depletion",
"what is the bubble point?", "differential liberation study",
"separator test", "PVT analysis", "saturation pressure"

SAFETY ANALYSIS (for "PSV sizing", "relief valve", "safety system", "API 520", "flare load"):
When the user asks about pressure safety valves, relief systems, or safety sizing, output a ```safety ... ``` block:
```safety
{{
  "design_pressure_factor": 1.1,
  "include_fire": true,
  "include_blocked_outlet": true
}}
```
Parameters:
  - design_pressure_factor: ratio of design pressure to operating pressure (default: 1.1)
  - include_fire: include fire case relief scenarios (default: true)
  - include_blocked_outlet: include blocked outlet scenarios (default: true)

Use this for: "size the PSVs", "what is the flare load?", "relief valve analysis",
"safety system design", "API 520 sizing"

FLOW ASSURANCE (for "hydrate risk", "wax", "corrosion", "MEG dosing", "flow assurance"):
When the user asks about hydrates, wax, corrosion, or flow assurance, output a ```flowassurance ... ``` block:
```flowassurance
{{
  "check_hydrates": true,
  "check_wax": true,
  "check_corrosion": true,
  "inhibitor_type": "MEG"
}}
```
Parameters:
  - check_hydrates: check for gas hydrate formation risk (default: true)
  - check_wax: check for wax deposition risk (default: true)
  - check_corrosion: check for CO₂/H₂S corrosion (default: true)
  - inhibitor_type: "MEG" or "MeOH" for hydrate inhibitor (default: "MEG")

Use this for: "check for hydrate risk", "flow assurance assessment", "is there wax risk?",
"corrosion analysis", "MEG injection rate", "what is the hydrate temperature?"

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

    # The prompt was originally written with {{/}} escaping for Python
    # .format().  Since we now use .replace() for substitution, the
    # double-braces are passed literally to the LLM which then mimics
    # them in its output — breaking json.loads().  Normalise to single.
    system_prompt = system_prompt.replace("{{", "{").replace("}}", "}")

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

PROPERTY UPDATES (after a process has been built — PREFERRED over full rebuild):
When the user wants to change parameters on existing equipment (pressure, temperature,
flow rate, efficiency, etc.) WITHOUT changing the process structure, use a scenario JSON
with ``changes``. This preserves compressor charts, auto-sizing data, and mechanical design.

Example — change compressor outlet pressure:
```json
{
  "scenarios": [
    {
      "name": "Update pressure",
      "description": "Update compressor outlet pressure",
      "patch": {"changes": {"units.compressor 1.outletPressure": 120.0}}
    }
  ]
}
```

Example — update feed flow rate:
```json
{
  "scenarios": [
    {
      "name": "Increase flow",
      "description": "Increase feed flow rate",
      "patch": {"changes": {"streams.feed gas.flow_kg_hr": 15000}}
    }
  ]
}
```

IMPORTANT — AVOID UNNECESSARY REBUILDS:
If a process has already been built, do NOT emit a new full build spec (with "fluid" and
"process") just to update parameters or add equipment at the start/end. A full rebuild
destroys compressor charts, mechanical design, and auto-sizing data. Instead:
  - To UPDATE PROPERTIES: use scenario JSON changes (above)
  - To ADD EQUIPMENT AT END: use incremental additions (```build {"add": [...]}```)
  - To GENERATE COMPRESSOR CHARTS: use ```chart``` only when explicitly requested
Only emit a full build spec when:
  - Building a completely new process from scratch (no existing process)
  - The user explicitly asks to "rebuild" or "start over"
  - Equipment is inserted in the MIDDLE (not start/end) of the process

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

IMPORTANT: Do NOT write Python scripts yourself. ALWAYS use ```build {"action": "show_script"}``` so
the system generates the correct script with proper NeqSim Java API imports.
Do NOT use "from neqsim.process import ..." or "from neqsim.thermo.system import ..." — these are WRONG.
The correct imports are handled automatically by the show_script action.

PROCESS OPTIMIZATION (after a process has been built):
When the user asks to optimize, find maximum production, or maximize throughput, output:
```optimize
{
  "objective": "maximize_throughput",
  "feed_stream": "feed gas",
  "min_flow_kg_hr": 1000,
  "max_flow_kg_hr": 500000,
  "utilization_limit": 1.0,
  "tolerance_pct": 1.0,
  "max_iterations": 25
}
```
Use this for: "find max production", "maximize throughput", "what is the bottleneck?",
"how much can we increase flow?", "optimize the process".

RISK ANALYSIS (after a process has been built):
When the user asks about risk, reliability, availability, or equipment criticality, output:
```risk
{
  "analysis": "full",
  "mc_iterations": 1000,
  "mc_days": 365,
  "include_degraded": true
}
```
Use this for: "show risk matrix", "equipment criticality", "system availability",
"failure analysis", "Monte Carlo simulation", "which equipment is most critical?".

COMPRESSOR CHART (after a process has been built):
When the user asks to generate or show a compressor chart/map/performance curve, output:
```chart
{
  "compressor": "1st stage compressor",
  "template": "CENTRIFUGAL_STANDARD",
  "num_speeds": 5,
  "show_only": false
}
```
Templates: CENTRIFUGAL_STANDARD, CENTRIFUGAL_HIGH_FLOW, CENTRIFUGAL_HIGH_HEAD,
PIPELINE, EXPORT, INJECTION, GAS_LIFT, REFRIGERATION, BOOSTER,
SINGLE_STAGE, MULTISTAGE_INLINE, INTEGRALLY_GEARED, OVERHUNG.
  - show_only: true = reuse existing chart (update operating point only), false = generate new chart.
    If a chart was already generated (e.g. via auto-size), set true to reuse it.
    Only set false when the user explicitly asks to GENERATE/CREATE a NEW chart or a different template.
Use this for: "generate compressor chart" (show_only: false), "show compressor map" (show_only: true),
"compressor performance curve" (show_only: true if chart exists, false otherwise).

AUTO-SIZE (after a process has been built):
When the user asks to auto-size, get sizing report, check utilization, or find bottleneck, output:
```autosize
{
  "safety_factor": 1.2,
  "generate_charts": true,
  "chart_template": "CENTRIFUGAL_STANDARD",
  "force_resize": false
}
```
  - force_resize: false = skip already-sized equipment (default), true = re-size everything.
    Only set true when user explicitly asks to RE-SIZE or wants a different safety factor.
Use this for: "auto-size equipment", "sizing report", "equipment utilization",
"what is the bottleneck?", "size all equipment", "re-size" (force_resize: true).

EMISSIONS ANALYSIS (after a process has been built):
When the user asks about emissions, CO2, carbon footprint, output:
```emissions
{
  "product_stream": "export gas",
  "include_fugitives": true,
  "power_source": "gas_turbine"
}
```
Use this for: "CO2 emissions", "carbon footprint", "emission intensity", "fuel gas consumption".

DYNAMIC SIMULATION (after a process has been built):
When the user asks about blowdown, startup, shutdown, or transient behavior, output:
For true transient (valve closures, step changes, upset conditions):
```dynamic
{
  "scenario_type": "transient",
  "changes": [{"unit": "VLV-100", "property": "percentValveOpening", "value": 10}],
  "duration_s": 50,
  "n_steps": 10,
  "dt": 5.0
}
```
For blowdown:
```dynamic
{
  "scenario_type": "blowdown",
  "duration_s": 600,
  "n_steps": 50
}
```
Use this for: "simulate blowdown", "startup sequence", "shutdown simulation", "transient",
"close valves", "run transient steps", "step change".

SENSITIVITY ANALYSIS (after a process has been built):
When the user asks for sensitivity studies or parameter sweeps, output:
```sensitivity
{
  "analysis_type": "single_sweep",
  "variable": "streams.feed gas.pressure_bara",
  "min_value": 30, "max_value": 80, "n_points": 10,
  "response_kpis": ["power"]
}
```
Use this for: "sensitivity analysis", "parameter sweep", "tornado chart".

PVT SIMULATION (after a process has been built):
When the user asks about PVT experiments, output:
```pvt
{
  "experiment": "CME",
  "stream_name": "feed gas",
  "temperature_C": 100, "p_start_bara": 400, "p_end_bara": 10, "n_steps": 20
}
```
Use this for: "CME experiment", "CVD experiment", "constant volume depletion",
"bubble point", "differential liberation", "PVT study".

SAFETY ANALYSIS (after a process has been built):
When the user asks about PSV sizing, relief valves, or safety systems, output:
```safety
{
  "design_pressure_factor": 1.1,
  "include_fire": true,
  "include_blocked_outlet": true
}
```
Use this for: "PSV sizing", "relief valve analysis", "flare load", "safety system design".

FLOW ASSURANCE (after a process has been built):
When the user asks about hydrates, wax, corrosion, or flow assurance, output:
```flowassurance
{
  "check_hydrates": true,
  "check_wax": true,
  "check_corrosion": true,
  "inhibitor_type": "MEG"
}
```
Use this for: "hydrate risk", "wax risk", "corrosion analysis", "MEG dosing", "flow assurance".
"""


# ---------------------------------------------------------------------------
# Chat message processing
# ---------------------------------------------------------------------------

def _normalise_json(text: str) -> str:
    """Normalise doubled braces ``{{ }}`` → ``{ }`` so ``json.loads`` succeeds.

    LLMs sometimes mimic the Python format-string escaping they see in
    the system prompt and emit ``{{ ... }}`` instead of ``{ ... }``.
    """
    return text.replace("{{", "{").replace("}}", "}")


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
            data = json.loads(_normalise_json(match))
            if "scenarios" in data:
                return data
        except json.JSONDecodeError:
            continue
    
    # Also try the entire text as JSON
    try:
        data = json.loads(_normalise_json(text))
        if "scenarios" in data:
            return data
    except json.JSONDecodeError:
        pass

    return None


def extract_optimize_spec(text: str) -> Optional[dict]:
    """
    Extract an optimization specification from LLM output text.
    Looks for ```optimize ... ``` blocks with JSON.
    """
    import re
    pattern = r'```optimize\s*\n(.*?)\n\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            data = json.loads(_normalise_json(match))
            if "objective" in data:
                return data
        except json.JSONDecodeError:
            continue
    
    return None


def extract_risk_spec(text: str) -> Optional[dict]:
    """
    Extract a risk analysis specification from LLM output text.
    Looks for ```risk ... ``` blocks with JSON.
    """
    import re
    pattern = r'```risk\s*\n(.*?)\n\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            data = json.loads(_normalise_json(match))
            if "analysis" in data:
                return data
        except json.JSONDecodeError:
            continue
    
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
            data = json.loads(_normalise_json(match))
            if "properties" in data and isinstance(data["properties"], list):
                return data
        except json.JSONDecodeError:
            continue
    
    return None


# ---------------------------------------------------------------------------
# Tool-block stripping (safety net for followup LLM calls)
# ---------------------------------------------------------------------------

# All tool block types that the LLM might accidentally emit in a followup
_TOOL_BLOCK_TYPES = (
    "json", "chart", "autosize", "optimize", "risk", "emissions",
    "dynamic", "sensitivity", "pvt", "safety", "flowassurance",
    "flow_assurance", "query", "build",
)

def _strip_tool_blocks(text: str) -> str:
    """Remove any ```<tool> ... ``` code blocks from LLM output.

    Followup LLM calls should only return natural-language explanations.
    If the LLM accidentally emits a tool block, strip it so the user
    doesn't see raw JSON.
    """
    import re
    for block_type in _TOOL_BLOCK_TYPES:
        pattern = rf'```{block_type}\s*\n.*?\n\s*```'
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    # Also strip any remaining unlabelled ```{ ... }``` JSON blocks
    text = re.sub(r'```\s*\n\s*\{.*?\}\s*\n\s*```', '', text, flags=re.DOTALL)
    return text.strip()


def _is_build_spec(data: dict) -> bool:
    """Return True if *data* looks like a process build specification.

    Build specs are distinguished from scenario JSON by having either
    ``fluid`` + ``process`` keys (full build), an ``action`` key
    (show_script / save), or an ``add`` key (incremental addition).
    """
    if "action" in data:
        return True
    if "add" in data:
        return True
    if "fluid" in data and "process" in data:
        return True
    return False


def extract_build_spec(text: str) -> Optional[dict]:
    """Extract a process build specification from LLM output.

    Checks in order:
    1. Explicit ````build ... ```` blocks (always treated as build spec).
    2. ````json ... ```` blocks whose content looks like a build spec.
    3. Unmarked ```` ``` ... ``` ```` blocks whose content is a build spec.
    4. Raw JSON objects in the text (e.g. user pasted JSON directly).

    Returns the parsed dict or ``None``.
    """
    import re

    # 1. Explicit ```build blocks
    pattern = r'```build\s*\n(.*?)\n\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(_normalise_json(match))
            return data
        except json.JSONDecodeError:
            continue

    # 2. ```json blocks that look like build specs
    pattern_json = r'```json\s*\n(.*?)\n\s*```'
    matches_json = re.findall(pattern_json, text, re.DOTALL)
    for match in matches_json:
        try:
            data = json.loads(_normalise_json(match))
            if _is_build_spec(data):
                return data
        except json.JSONDecodeError:
            continue

    # 3. Unmarked ``` blocks (no language tag) that look like build specs
    pattern_plain = r'```\n(.*?)\n```'
    matches_plain = re.findall(pattern_plain, text, re.DOTALL)
    for match in matches_plain:
        try:
            data = json.loads(_normalise_json(match))
            if isinstance(data, dict) and _is_build_spec(data):
                return data
        except json.JSONDecodeError:
            continue

    # 4. Raw JSON in text (user may paste JSON directly without code fences)
    normalised = _normalise_json(text)
    brace_start = normalised.find('{')
    if brace_start >= 0:
        try:
            decoder = json.JSONDecoder()
            data, _ = decoder.raw_decode(normalised, brace_start)
            if isinstance(data, dict) and _is_build_spec(data):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def extract_chart_spec(text: str) -> Optional[dict]:
    """Extract a compressor chart specification from LLM output.

    Looks for ````chart ... ```` blocks with JSON.
    """
    import re
    pattern = r'```chart\s*\n(.*?)\n\s*```'
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        try:
            data = json.loads(_normalise_json(match))
            return data
        except json.JSONDecodeError:
            continue

    return None


def extract_autosize_spec(text: str) -> Optional[dict]:
    """Extract an auto-size specification from LLM output.

    Looks for ````autosize ... ```` blocks with JSON.
    """
    import re
    pattern = r'```autosize\s*\n(.*?)\n\s*```'
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        try:
            data = json.loads(_normalise_json(match))
            return data
        except json.JSONDecodeError:
            continue

    return None


def extract_emissions_spec(text: str) -> Optional[dict]:
    """Extract an emissions specification from LLM output."""
    import re
    pattern = r'```emissions\s*\n(.*?)\n\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(_normalise_json(match))
        except json.JSONDecodeError:
            continue
    return None


def extract_dynamic_spec(text: str) -> Optional[dict]:
    """Extract a dynamic simulation specification from LLM output."""
    import re
    pattern = r'```dynamic\s*\n(.*?)\n\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(_normalise_json(match))
            if "scenario_type" in data:
                return data
        except json.JSONDecodeError:
            continue
    return None


def extract_sensitivity_spec(text: str) -> Optional[dict]:
    """Extract a sensitivity analysis specification from LLM output."""
    import re
    pattern = r'```sensitivity\s*\n(.*?)\n\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(_normalise_json(match))
        except json.JSONDecodeError:
            continue
    return None


def extract_pvt_spec(text: str) -> Optional[dict]:
    """Extract a PVT simulation specification from LLM output."""
    import re
    pattern = r'```pvt\s*\n(.*?)\n\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(_normalise_json(match))
        except json.JSONDecodeError:
            continue
    return None


def extract_safety_spec(text: str) -> Optional[dict]:
    """Extract a safety analysis specification from LLM output."""
    import re
    pattern = r'```safety\s*\n(.*?)\n\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(_normalise_json(match))
        except json.JSONDecodeError:
            continue
    return None


def extract_flow_assurance_spec(text: str) -> Optional[dict]:
    """Extract a flow assurance specification from LLM output."""
    import re
    pattern = r'```flowassurance\s*\n(.*?)\n\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(_normalise_json(match))
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


# ---------------------------------------------------------------------------
# Smart rebuild classification
# ---------------------------------------------------------------------------

def _classify_build_change(old_spec: dict, new_spec: dict):
    """
    Compare old and new build specs to decide whether a full rebuild is needed.

    Returns ``(change_type, param_changes, fluid_changes, new_steps)`` where:

    * **change_type**: one of ``"no_change"``, ``"property_update"``,
      ``"append_end"``, ``"prepend_start"``, ``"append_both"``,
      ``"full_rebuild"``.
    * **param_changes**: ``{unit_name: {"type": ..., "old_params": ...,
      "new_params": ...}}`` for units whose parameters differ.
    * **fluid_changes**: ``{prop: new_value}`` for changed feed-stream
      conditions (T, P, flow).
    * **new_steps**: list of step dicts that were appended / prepended.
    """
    old_steps = old_spec.get("process", [])
    new_steps = new_spec.get("process", [])

    old_names = [s["name"] for s in old_steps]
    new_names = [s["name"] for s in new_steps]

    # --- Fluid component / EOS changes require a full rebuild ---
    old_fluid = old_spec.get("fluid", {})
    new_fluid = new_spec.get("fluid", {})
    if (old_fluid.get("components") != new_fluid.get("components")
            or old_fluid.get("eos_model", "srk") != new_fluid.get("eos_model", "srk")
            or old_fluid.get("mixing_rule", 2) != new_fluid.get("mixing_rule", 2)
            or old_fluid.get("composition_basis") != new_fluid.get("composition_basis")):
        return "full_rebuild", {}, {}, []

    # --- Fluid condition changes (T, P, flow) ---
    fluid_changes: dict = {}
    for prop in ("temperature_C", "pressure_bara", "total_flow", "flow_unit"):
        old_val = old_fluid.get(prop)
        new_val = new_fluid.get(prop)
        if old_val != new_val and new_val is not None:
            fluid_changes[prop] = new_val

    if not old_names:
        return "full_rebuild", {}, {}, []

    # --- Find where old_names appear as a contiguous sub-sequence ---
    start_idx = None
    for i in range(max(len(new_names) - len(old_names) + 1, 1)):
        if new_names[i:i + len(old_names)] == old_names:
            start_idx = i
            break

    if start_idx is None:
        return "full_rebuild", {}, {}, []

    prepended = new_steps[:start_idx]
    appended = new_steps[start_idx + len(old_names):]

    # --- Param changes on the common (existing) steps ---
    param_changes: dict = {}
    for i, old_step in enumerate(old_steps):
        new_step = new_steps[start_idx + i]
        old_params = old_step.get("params", {})
        new_params = new_step.get("params", {})
        if old_params != new_params:
            param_changes[old_step["name"]] = {
                "type": old_step["type"],
                "old_params": old_params,
                "new_params": new_params,
            }

    if not prepended and not appended:
        if not param_changes and not fluid_changes:
            return "no_change", {}, {}, []
        return "property_update", param_changes, fluid_changes, []

    if not prepended and appended:
        return "append_end", param_changes, fluid_changes, appended

    if prepended and not appended:
        return "prepend_start", param_changes, fluid_changes, prepended

    return "append_both", param_changes, fluid_changes, prepended + appended


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
        self._last_optimization = None
        self._last_risk_analysis = None
        self._last_chart = None
        self._chart_cache: Dict[str, "CompressorChartResult"] = {}  # persists across messages
        self._sized_equipment: set = set()  # tracks equipment already auto-sized
        self._last_autosize = None
        self._last_emissions = None
        self._last_dynamic = None
        self._last_sensitivity = None
        self._last_pvt = None
        self._last_safety = None
        self._last_flow_assurance = None
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
          - Optimize specs (```optimize```) → find max production
          - Plain Q&A → direct LLM response
        """
        from google import genai
        from google.genai import types

        # Clear per-message results so stale data doesn't leak across messages
        self._last_comparison = None
        self._last_optimization = None
        self._last_risk_analysis = None
        self._last_chart = None
        self._last_autosize = None
        self._last_emissions = None
        self._last_dynamic = None
        self._last_sensitivity = None
        self._last_pvt = None
        self._last_safety = None
        self._last_flow_assurance = None

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
        if not build_spec:
            # LLM didn't emit a build block — check if the user pasted
            # a JSON build spec directly in their message.
            build_spec = extract_build_spec(user_message)
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

        # --- Check for optimization spec ---
        optimize_spec = extract_optimize_spec(assistant_text)
        if optimize_spec and self.model:
            return self._handle_optimization(assistant_text, optimize_spec, client, types)

        # --- Check for risk analysis spec ---
        risk_spec = extract_risk_spec(assistant_text)
        if risk_spec and self.model:
            return self._handle_risk_analysis(assistant_text, risk_spec, client, types)

        # --- Check for compressor chart spec ---
        chart_spec = extract_chart_spec(assistant_text)
        if chart_spec and self.model:
            return self._handle_chart(assistant_text, chart_spec, client, types)

        # --- Check for auto-size spec ---
        autosize_spec = extract_autosize_spec(assistant_text)
        if autosize_spec and self.model:
            return self._handle_autosize(assistant_text, autosize_spec, client, types)

        # --- Check for emissions spec ---
        emissions_spec = extract_emissions_spec(assistant_text)
        if emissions_spec and self.model:
            return self._handle_emissions(assistant_text, emissions_spec, client, types)

        # --- Check for dynamic simulation spec ---
        dynamic_spec = extract_dynamic_spec(assistant_text)
        if dynamic_spec and self.model:
            return self._handle_dynamic(assistant_text, dynamic_spec, client, types)

        # --- Check for sensitivity analysis spec ---
        sensitivity_spec = extract_sensitivity_spec(assistant_text)
        if sensitivity_spec and self.model:
            return self._handle_sensitivity(assistant_text, sensitivity_spec, client, types)

        # --- Check for PVT simulation spec ---
        pvt_spec = extract_pvt_spec(assistant_text)
        if pvt_spec and self.model:
            return self._handle_pvt(assistant_text, pvt_spec, client, types)

        # --- Check for safety analysis spec ---
        safety_spec = extract_safety_spec(assistant_text)
        if safety_spec and self.model:
            return self._handle_safety(assistant_text, safety_spec, client, types)

        # --- Check for flow assurance spec ---
        fa_spec = extract_flow_assurance_spec(assistant_text)
        if fa_spec and self.model:
            return self._handle_flow_assurance(assistant_text, fa_spec, client, types)

        # --- Pure Q&A ---
        cleaned = _strip_tool_blocks(assistant_text)
        self.history.append({"role": "assistant", "content": cleaned})
        self._last_comparison = None
        return cleaned

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
                # Return the script directly — do NOT rely on _llm_followup
                # because the LLM tends to summarize instead of reproducing
                # the full code verbatim.
                script_response = (
                    "Here is the Python script for your process:\n\n"
                    f"```python\n{script}\n```\n\n"
                    "You can copy this script and run it in any Python environment "
                    "with the NeqSim library installed."
                )
                self.history.append({"role": "assistant", "content": script_response})
                return script_response
            elif self.model:
                # Model exists but no builder (uploaded model) — provide model summary
                summary = self.model.get_model_summary()
                self.history.append({"role": "assistant", "content": assistant_text})
                self.history.append({
                    "role": "user",
                    "content": (
                        "[SYSTEM: This model was uploaded (not built from scratch), so an auto-generated "
                        "Python script is not available. Provide the model summary instead and suggest "
                        "the user can build a new process from scratch to get a Python script.]\n\n"
                        f"Model summary:\n{summary}"
                    )
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
            # ── Smart rebuild detection ──────────────────────────────
            # If we already have a built model, check whether we can
            # apply the change incrementally (preserving compressor
            # charts, mechanical design, and auto-sizing data).
            if self._builder and self._builder.spec and self.model:
                change_type, param_changes, fluid_changes, extra_steps = (
                    _classify_build_change(self._builder.spec, build_spec)
                )
                if change_type in (
                    "no_change", "property_update", "append_end",
                    "prepend_start", "append_both",
                ):
                    return self._handle_incremental_update(
                        assistant_text, build_spec, change_type,
                        param_changes, fluid_changes, extra_steps,
                        client, types,
                    )

            # ── Full rebuild (new process or fundamental change) ─────
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

                # Keep builder spec in sync so to_python_script() is accurate
                if self._builder and self._builder.spec:
                    proc_steps = self._builder.spec.setdefault("process", [])
                    for a in add_list:
                        proc_steps.append({
                            "name": a["name"],
                            "type": a["type"],
                            "params": a.get("params", {}),
                        })

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
        cleaned = _strip_tool_blocks(assistant_text)
        self.history.append({"role": "assistant", "content": cleaned})
        self._last_comparison = None
        return cleaned

    # -- Incremental update (avoids full rebuild) ---------------------------

    def _handle_incremental_update(
        self,
        assistant_text: str,
        build_spec: dict,
        change_type: str,
        param_changes: dict,
        fluid_changes: dict,
        extra_steps: list,
        client,
        types,
    ) -> str:
        """Apply changes to the existing model without a full rebuild.

        Preserves compressor charts, auto-sizing data, and mechanical
        design by modifying the live Java objects in place (or appending
        new units at the ends).
        """
        from .process_builder import _apply_param

        try:
            changes_applied: list = []

            # 1. Apply fluid condition changes to the feed stream(s)
            if fluid_changes:
                proc = self.model.get_process()
                for u in proc.getUnitOperations():
                    try:
                        java_class = str(u.getClass().getSimpleName())
                        if java_class != "Stream":
                            continue
                        fluid = None
                        for getter in ("getFluid", "getThermoSystem"):
                            if hasattr(u, getter):
                                fluid = getattr(u, getter)()
                                if fluid is not None:
                                    break
                        if fluid is None:
                            continue
                        if "temperature_C" in fluid_changes:
                            fluid.setTemperature(float(fluid_changes["temperature_C"]), "C")
                            changes_applied.append(
                                f"Feed temperature → {fluid_changes['temperature_C']}°C"
                            )
                        if "pressure_bara" in fluid_changes:
                            fluid.setPressure(float(fluid_changes["pressure_bara"]), "bara")
                            changes_applied.append(
                                f"Feed pressure → {fluid_changes['pressure_bara']} bara"
                            )
                        if "total_flow" in fluid_changes:
                            flow_unit = fluid_changes.get(
                                "flow_unit",
                                build_spec.get("fluid", {}).get("flow_unit", "kg/hr"),
                            )
                            fluid.setTotalFlowRate(
                                float(fluid_changes["total_flow"]), flow_unit
                            )
                            changes_applied.append(
                                f"Feed flow → {fluid_changes['total_flow']} {flow_unit}"
                            )
                        break  # only update the first (feed) stream
                    except Exception:
                        continue

            # 2. Apply parameter changes to existing equipment
            if param_changes:
                for unit_name, change_info in param_changes.items():
                    try:
                        unit = self.model.get_unit(unit_name)
                        new_params = change_info["new_params"]
                        for k, v in new_params.items():
                            try:
                                _apply_param(unit, k, v)
                                changes_applied.append(f"{unit_name}: {k}={v}")
                            except Exception:
                                pass
                    except Exception as e:
                        changes_applied.append(f"{unit_name}: FAILED — {e}")

            # 3. Append / prepend new equipment at the ends
            if extra_steps and change_type in (
                "append_end", "prepend_start", "append_both",
            ):
                from .scenario_engine import apply_add_units
                from .patch_schema import AddUnitOp

                old_steps = self._builder.spec.get("process", [])
                old_names = [s["name"] for s in old_steps]

                new_names = [s["name"] for s in build_spec.get("process", [])]
                # Determine which are appended vs prepended
                start_idx = None
                search_range = max(len(new_names) - len(old_names) + 1, 0)
                for i in range(search_range):
                    if new_names[i:i + len(old_names)] == old_names:
                        start_idx = i
                        break
                if start_idx is None:
                    # Old steps not found as contiguous subsequence —
                    # fall back to full rebuild.
                    if self._builder is not None:
                        self._builder._spec = None
                    return self._handle_build(assistant_text, build_spec, client, types)

                prepended = build_spec["process"][:start_idx]
                appended = build_spec["process"][start_idx + len(old_names):]

                # Handle appended (after last existing unit)
                if appended:
                    last_existing = old_names[-1] if old_names else None
                    add_ops = []
                    prev_name = last_existing
                    for step in appended:
                        if step["type"].lower() == "stream":
                            continue  # skip raw streams – they're feeds
                        if prev_name is None:
                            continue
                        add_ops.append(AddUnitOp(
                            name=step["name"],
                            equipment_type=step["type"],
                            insert_after=prev_name,
                            params=step.get("params", {}),
                        ))
                        prev_name = step["name"]
                    if add_ops:
                        log = apply_add_units(self.model, add_ops)
                        for entry in log:
                            if entry.get("status") == "OK":
                                changes_applied.append(
                                    f"Added {entry.get('value', entry.get('key', '?'))} (end)"
                                )

                # Handle prepended (before first existing unit)
                if prepended:
                    first_existing = old_names[0] if old_names else None
                    # Prepended units need to go before the first unit which
                    # is difficult without a rebuild. Use apply_add_units
                    # with a chain: insert each prepended unit after the
                    # previous prepended unit, then reconnect.
                    # For now, we insert them and let NeqSim handle the
                    # stream wiring via apply_add_units.
                    if first_existing:
                        add_ops = []
                        # We need to insert in reverse order so that each
                        # unit is placed before the first existing.
                        # Actually, insert the first prepended unit after
                        # the feed stream (or as a new feed), then chain.
                        # Get the feed stream name (first unit in process)
                        proc = self.model.get_process()
                        units = list(proc.getUnitOperations())
                        feed_name = None
                        for u in units:
                            try:
                                jc = str(u.getClass().getSimpleName())
                                if jc == "Stream":
                                    feed_name = str(u.getName())
                                    break
                            except Exception:
                                pass

                        prev_name = feed_name
                        for step in prepended:
                            if step["type"].lower() == "stream":
                                continue
                            if prev_name is None:
                                continue
                            add_ops.append(AddUnitOp(
                                name=step["name"],
                                equipment_type=step["type"],
                                insert_after=prev_name,
                                params=step.get("params", {}),
                            ))
                            prev_name = step["name"]
                        if add_ops:
                            log = apply_add_units(self.model, add_ops)
                            for entry in log:
                                if entry.get("status") == "OK":
                                    changes_applied.append(
                                        f"Added {entry.get('value', entry.get('key', '?'))} (start)"
                                    )

            # 4. Re-run the process
            NeqSimProcessModel._run_until_converged(self.model.get_process())
            self.model._index_model_objects()
            self.model.refresh_source_bytes()

            # 5. Update the builder spec (keep in sync)
            self._builder._spec = build_spec
            self._builder._process_name = build_spec.get(
                "name", self._builder._process_name
            )

            # 6. Update system prompt
            self._system_prompt = build_system_prompt(self.model)

            summary = self.model.get_model_summary()
            changes_str = (
                "\n".join(f"  • {c}" for c in changes_applied)
                if changes_applied
                else "  No effective changes detected."
            )

            self.history.append({"role": "assistant", "content": assistant_text})
            self.history.append({
                "role": "user",
                "content": (
                    f"[SYSTEM: Process updated incrementally (without full rebuild — "
                    f"compressor charts, auto-sizing, and mechanical design are preserved). "
                    f"Changes applied:\n{changes_str}\n\n"
                    f"Updated model summary:\n{summary}\n\n"
                    "Explain the updated results to the engineer. "
                    "Mention the key changes and how they affected the process.]"
                ),
            })
            return self._llm_followup(client, types)

        except Exception as e:
            # Incremental update failed — fall back to full rebuild
            if self._builder is not None:
                self._builder._spec = None  # force full rebuild path
            return self._handle_build(assistant_text, build_spec, client, types)

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

            # --- Persist changes to the base model ---
            # Both structural additions AND property-only changes are
            # persisted so the model stays up-to-date for subsequent queries.
            structural_applied = False
            for sc in scenarios:
                has_structural = (
                    sc.patch.add_units or sc.patch.add_streams or sc.patch.add_process
                )
                has_changes = bool(sc.patch.changes)
                has_components = bool(getattr(sc.patch, 'add_components', None))
                has_targets = bool(getattr(sc.patch, 'targets', None))
                if not has_structural and not has_changes and not has_components and not has_targets:
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
                        apply_patch_to_model, apply_add_components,
                    )
                    if sc.patch.add_units:
                        apply_add_units(self.model, sc.patch.add_units)
                    if sc.patch.add_process:
                        apply_add_process(self.model, sc.patch.add_process)
                    if sc.patch.add_streams:
                        apply_add_streams(self.model, sc.patch.add_streams)
                    if getattr(sc.patch, 'add_components', None):
                        apply_add_components(self.model, sc.patch.add_components)
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

                    # Keep builder spec in sync so to_python_script() is accurate
                    if self._builder and self._builder.spec:
                        proc_steps = self._builder.spec.setdefault("process", [])
                        for au in (sc.patch.add_units or []):
                            proc_steps.append({
                                "name": au.name,
                                "type": au.equipment_type,
                                "params": dict(au.params) if au.params else {},
                            })
                        for ap in (sc.patch.add_process or []):
                            for u in (ap.units or []):
                                proc_steps.append({
                                    "name": u.get("name", "unit"),
                                    "type": u.get("equipment_type", "unknown"),
                                    "params": u.get("params", {}),
                                })
                except Exception:
                    pass  # comparison still valid even if persistence fails

            # --- Refresh cached chart operating points ---
            # After any scenario changes the model state, cached charts
            # hold stale operating-point data.  Refresh them now so that
            # subsequent "show_only" requests (and history replays in the
            # page) reflect the updated conditions.
            if self._chart_cache:
                try:
                    refreshed_cache: dict = {}
                    for ck, cr in self._chart_cache.items():
                        new_charts = [
                            refresh_operating_point(self.model, cd)
                            for cd in cr.charts
                        ]
                        refreshed_cache[ck] = CompressorChartResult(
                            charts=new_charts,
                            message=cr.message,
                        )
                    self._chart_cache.update(refreshed_cache)
                except Exception:
                    pass

            self.history.append({"role": "assistant", "content": assistant_text})

            persist_note = ""
            if structural_applied:
                persist_note = (
                    "\n\n[SYSTEM NOTE: Changes have been applied to the base model "
                    "(compressor charts and mechanical design preserved). "
                    "Future questions will see the updated state.]\n\n"
                    "Updated model summary:\n"
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

    # -- Optimization handling -----------------------------------------------

    def _handle_optimization(self, assistant_text: str, optimize_spec: dict, client, types) -> str:
        """Run process optimization and feed results back to LLM."""
        try:
            feed_stream = optimize_spec.get("feed_stream")
            min_flow = optimize_spec.get("min_flow_kg_hr")
            max_flow = optimize_spec.get("max_flow_kg_hr")
            util_limit = float(optimize_spec.get("utilization_limit", 1.0))
            tolerance = float(optimize_spec.get("tolerance_pct", 1.0))
            max_iter = int(optimize_spec.get("max_iterations", 25))

            result = optimize_production(
                model=self.model,
                feed_stream_name=feed_stream,
                min_flow_kg_hr=min_flow,
                max_flow_kg_hr=max_flow,
                utilization_limit=util_limit,
                tolerance_pct=tolerance,
                max_iterations=max_iter,
            )

            self._last_optimization = result
            results_text = format_optimization_result(result)

            self.history.append({"role": "assistant", "content": assistant_text})
            self.history.append({
                "role": "user",
                "content": (
                    f"[SYSTEM: Process optimization completed. Results below. "
                    f"Explain these results to the engineer clearly. "
                    f"Highlight the optimal flow rate, bottleneck equipment, "
                    f"and how much production can be increased. "
                    f"If utilization data is available, mention the utilization "
                    f"breakdown for each equipment.]\n\n{results_text}"
                )
            })

            final_text = self._llm_followup(client, types)
            self._last_comparison = None
            return final_text

        except Exception as e:
            self.history.append({"role": "assistant", "content": assistant_text})
            self.history.append({
                "role": "user",
                "content": (
                    f"[SYSTEM: Optimization failed with error: {str(e)}. "
                    f"Inform the engineer and suggest possible reasons.]"
                )
            })
            return self._llm_followup(client, types)

    def get_last_optimization(self) -> Optional[OptimizationResult]:
        """Get the last optimization result (for UI display)."""
        return getattr(self, "_last_optimization", None)

    # -- Risk analysis handling ----------------------------------------------

    def _handle_risk_analysis(self, assistant_text: str, risk_spec: dict, client, types) -> str:
        """Run risk analysis and feed results back to LLM."""
        try:
            product_stream = risk_spec.get("product_stream")
            feed_stream = risk_spec.get("feed_stream")
            mc_iterations = int(risk_spec.get("mc_iterations", 1_000))
            mc_days = int(risk_spec.get("mc_days", 365))
            include_degraded = bool(risk_spec.get("include_degraded", True))

            result = run_risk_analysis(
                model=self.model,
                product_stream=product_stream,
                feed_stream=feed_stream,
                mc_iterations=mc_iterations,
                mc_days=mc_days,
                include_degraded=include_degraded,
            )

            self._last_risk_analysis = result
            results_text = format_risk_result(result)

            self.history.append({"role": "assistant", "content": assistant_text})
            self.history.append({
                "role": "user",
                "content": (
                    f"[SYSTEM: Risk analysis completed. Results below. "
                    f"Explain these results to the engineer clearly. "
                    f"Highlight the most critical equipment, the risk levels, "
                    f"system availability, and Monte Carlo production estimates. "
                    f"Present the risk matrix and equipment criticality ranking. "
                    f"Mention any HIGH or EXTREME risk items that need attention.]\n\n{results_text}"
                )
            })

            final_text = self._llm_followup(client, types)
            self._last_comparison = None
            return final_text

        except Exception as e:
            self.history.append({"role": "assistant", "content": assistant_text})
            self.history.append({
                "role": "user",
                "content": (
                    f"[SYSTEM: Risk analysis failed with error: {str(e)}. "
                    f"Inform the engineer and suggest possible reasons.]"
                )
            })
            return self._llm_followup(client, types)

    def get_last_risk_analysis(self) -> Optional[RiskAnalysisResult]:
        """Get the last risk analysis result (for UI display)."""
        return getattr(self, "_last_risk_analysis", None)

    # -- Compressor chart handling -------------------------------------------

    def _handle_chart(self, assistant_text: str, chart_spec: dict, client, types) -> str:
        """Generate or show compressor chart(s) and feed results back to LLM.

        When ``show_only`` is true in the chart spec, reuse a previously
        generated chart (updating only the operating point) instead of
        regenerating the full performance map.
        """
        try:
            compressor_name = chart_spec.get("compressor")
            template = chart_spec.get("template", "CENTRIFUGAL_STANDARD")
            num_speeds = int(chart_spec.get("num_speeds", 5))
            show_only = bool(chart_spec.get("show_only", False))

            # --- Try to reuse cached chart when show_only is requested ---
            cache_key = (compressor_name or "__all__").lower()
            if show_only and cache_key in self._chart_cache:
                cached = self._chart_cache[cache_key]
                # Re-run the model to get updated operating conditions.
                # If run() fails the model still has _units from the last
                # successful run (e.g. after a scenario), so refreshing the
                # operating point will still pick up current values.
                try:
                    self.model.run()
                except Exception:
                    # Model already ran during scenario persistence — units
                    # are valid; just ensure _units dict is populated.
                    try:
                        self.model._index_model_objects()
                    except Exception:
                        pass
                # Refresh the operating point(s) from current model state
                refreshed_charts = []
                for cd in cached.charts:
                    refreshed_charts.append(
                        refresh_operating_point(self.model, cd)
                    )
                result = CompressorChartResult(
                    charts=refreshed_charts,
                    message=cached.message,
                )
            else:
                # Full chart generation
                result = generate_compressor_chart(
                    model=self.model,
                    compressor_name=compressor_name,
                    template=template,
                    num_speeds=num_speeds,
                )

                # Persist chart state so cloned models retain it
                try:
                    self.model.refresh_source_bytes()
                except Exception:
                    pass

                # Update system prompt (chart may change compressor behaviour)
                self._system_prompt = build_system_prompt(self.model)

            # Store in both per-message result and persistent cache
            self._last_chart = result
            self._chart_cache[cache_key] = result

            # Also cache individual compressor entries so that a later
            # "show chart for X" finds them even when they were generated
            # as part of an "all compressors" request.
            if cache_key == "__all__":
                for cd in result.charts:
                    individual_key = cd.compressor_name.lower()
                    self._chart_cache[individual_key] = CompressorChartResult(
                        charts=[cd],
                        message=result.message,
                    )

            results_text = format_chart_result(result)

            self.history.append({"role": "assistant", "content": assistant_text})
            self.history.append({
                "role": "user",
                "content": (
                    f"[SYSTEM: Compressor chart generation completed. Results below. "
                    f"Explain the chart data to the engineer. Mention the speed curves, "
                    f"surge and stonewall limits, and the current operating point. "
                    f"Note any operating point concerns (near surge or stonewall).]\n\n"
                    f"{results_text}"
                )
            })

            final_text = self._llm_followup(client, types)
            self._last_comparison = None
            return final_text

        except Exception as e:
            self.history.append({"role": "assistant", "content": assistant_text})
            self.history.append({
                "role": "user",
                "content": (
                    f"[SYSTEM: Compressor chart generation failed: {str(e)}. "
                    f"Inform the engineer and suggest possible reasons.]"
                )
            })
            return self._llm_followup(client, types)

    def get_last_chart(self) -> Optional[CompressorChartResult]:
        """Get the last compressor chart result (for UI display)."""
        return getattr(self, "_last_chart", None)

    # -- Auto-size handling --------------------------------------------------

    def _handle_autosize(self, assistant_text: str, autosize_spec: dict, client, types) -> str:
        """Auto-size all equipment and feed results back to LLM."""
        try:
            safety_factor = float(autosize_spec.get("safety_factor", 1.2))
            gen_charts = bool(autosize_spec.get("generate_charts", True))
            chart_template = autosize_spec.get("chart_template", "CENTRIFUGAL_STANDARD")
            force_resize = bool(autosize_spec.get("force_resize", False))

            # If force_resize, clear the tracking set so everything is re-sized
            if force_resize:
                self._sized_equipment.clear()
                self._chart_cache.clear()

            result = auto_size_all(
                model=self.model,
                safety_factor=safety_factor,
                generate_compressor_charts=gen_charts,
                chart_template=chart_template,
                skip_already_sized=self._sized_equipment if self._sized_equipment else None,
            )

            self._last_autosize = result
            results_text = format_autosize_result(result)

            # Track which equipment was successfully sized
            for s in result.equipment_sized:
                if s.auto_sized:
                    self._sized_equipment.add(s.name)

            # Cache compressor charts generated during auto-sizing so that
            # subsequent "show chart" requests can reuse them.
            if gen_charts and self.model:
                try:
                    from .compressor_chart import _find_compressors, _extract_speed_curves, \
                        _extract_surge_curve, _extract_stonewall_curve, _extract_operating_point, \
                        CompressorChartData, SpeedCurve
                    for cname, cunit in _find_compressors(self.model):
                        try:
                            if cunit.getCompressorChart() is None:
                                continue
                            speed_curves = _extract_speed_curves(cunit)
                            surge_f, surge_h = _extract_surge_curve(cunit)
                            sw_f, sw_h = _extract_stonewall_curve(cunit)
                            op = _extract_operating_point(cunit)
                            min_spd = min((sc.speed_rpm for sc in speed_curves), default=0.0)
                            max_spd = max((sc.speed_rpm for sc in speed_curves), default=0.0)
                            cd = CompressorChartData(
                                compressor_name=cname,
                                template_used=chart_template,
                                speed_curves=speed_curves,
                                surge_flow=surge_f, surge_head=surge_h,
                                stonewall_flow=sw_f, stonewall_head=sw_h,
                                operating_point=op,
                                min_speed=min_spd, max_speed=max_spd,
                                message=f"Chart from auto-size, template={chart_template}",
                            )
                            cache_key = cname.lower()
                            self._chart_cache[cache_key] = CompressorChartResult(
                                charts=[cd],
                                message=f"Cached chart for {cname}",
                            )
                        except Exception:
                            pass
                except Exception:
                    pass

            # Persist auto-sized state so cloned models retain it
            try:
                self.model.refresh_source_bytes()
            except Exception:
                pass

            # Update system prompt (sizing changes model state)
            self._system_prompt = build_system_prompt(self.model)

            self.history.append({"role": "assistant", "content": assistant_text})
            self.history.append({
                "role": "user",
                "content": (
                    f"[SYSTEM: Auto-sizing completed. Results below. "
                    f"Explain the sizing results and utilization to the engineer. "
                    f"Highlight the bottleneck equipment, utilization percentages, "
                    f"and any equipment near capacity limits. "
                    f"Mention the sizing data (dimensions, capacities) for key equipment.]\n\n"
                    f"{results_text}"
                )
            })

            final_text = self._llm_followup(client, types)
            self._last_comparison = None
            return final_text

        except Exception as e:
            self.history.append({"role": "assistant", "content": assistant_text})
            self.history.append({
                "role": "user",
                "content": (
                    f"[SYSTEM: Auto-sizing failed: {str(e)}. "
                    f"Inform the engineer and suggest possible reasons.]"
                )
            })
            return self._llm_followup(client, types)

    def get_last_autosize(self) -> Optional[AutoSizeResult]:
        """Get the last auto-size result (for UI display)."""
        return getattr(self, "_last_autosize", None)

    # -- Emissions handling --------------------------------------------------

    def _handle_emissions(self, assistant_text: str, emissions_spec: dict, client, types) -> str:
        """Run emissions analysis and feed results back to LLM."""
        try:
            product_stream = emissions_spec.get("product_stream")
            include_fugitives = bool(emissions_spec.get("include_fugitives", True))
            flare_streams = emissions_spec.get("flare_streams", [])
            power_source = emissions_spec.get("power_source", "gas_turbine")

            result = calculate_emissions(
                model=self.model,
                product_stream=product_stream,
                include_fugitives=include_fugitives,
                flare_streams=flare_streams,
                power_source=power_source,
            )

            self._last_emissions = result
            results_text = format_emissions_result(result)

            self.history.append({"role": "assistant", "content": assistant_text})
            self.history.append({
                "role": "user",
                "content": (
                    f"[SYSTEM: Emissions analysis completed. Results below. "
                    f"Explain the emissions to the engineer. Highlight total CO₂, "
                    f"CO₂e, emission intensity, and the largest emission sources. "
                    f"Suggest opportunities for emissions reduction.]\n\n{results_text}"
                )
            })

            final_text = self._llm_followup(client, types)
            return final_text

        except Exception as e:
            self.history.append({"role": "assistant", "content": assistant_text})
            self.history.append({
                "role": "user",
                "content": f"[SYSTEM: Emissions analysis failed: {str(e)}. Inform the engineer.]"
            })
            return self._llm_followup(client, types)

    def get_last_emissions(self) -> Optional[EmissionsResult]:
        """Get the last emissions result (for UI display)."""
        return getattr(self, "_last_emissions", None)

    # -- Dynamic simulation handling -----------------------------------------

    def _handle_dynamic(self, assistant_text: str, dynamic_spec: dict, client, types) -> str:
        """Run dynamic simulation and feed results back to LLM."""
        try:
            _init_p = dynamic_spec.get("initial_pressure_bara")
            _start_v = dynamic_spec.get("start_value")
            _end_v = dynamic_spec.get("end_value")
            _dt_v = dynamic_spec.get("dt")
            result = run_dynamic_simulation(
                model=self.model,
                scenario_type=dynamic_spec.get("scenario_type", "blowdown"),
                vessel_name=dynamic_spec.get("vessel_name"),
                stream_name=dynamic_spec.get("stream_name"),
                initial_pressure_bara=float(_init_p) if _init_p is not None else None,
                final_pressure_bara=float(dynamic_spec.get("final_pressure_bara", 1.013)),
                orifice_diameter_mm=float(dynamic_spec.get("orifice_diameter_mm", 50)),
                duration_s=float(dynamic_spec.get("duration_s", 600)),
                n_steps=int(dynamic_spec.get("n_steps", 50)),
                start_value=float(_start_v) if _start_v is not None else None,
                end_value=float(_end_v) if _end_v is not None else None,
                changes=dynamic_spec.get("changes"),
                dt=float(_dt_v) if _dt_v is not None else None,
            )

            self._last_dynamic = result
            results_text = format_dynamic_result(result)

            self.history.append({"role": "assistant", "content": assistant_text})
            self.history.append({
                "role": "user",
                "content": (
                    f"[SYSTEM: Dynamic simulation completed. Results below. "
                    f"Explain the transient behavior to the engineer. "
                    f"Highlight the initial and final states, key time points, "
                    f"and any safety concerns (e.g., low temperatures during blowdown).]\n\n"
                    f"{results_text}"
                )
            })

            final_text = self._llm_followup(client, types)
            return final_text

        except Exception as e:
            self.history.append({"role": "assistant", "content": assistant_text})
            self.history.append({
                "role": "user",
                "content": f"[SYSTEM: Dynamic simulation failed: {str(e)}. Inform the engineer.]"
            })
            return self._llm_followup(client, types)

    def get_last_dynamic(self) -> Optional[DynamicSimResult]:
        """Get the last dynamic simulation result (for UI display)."""
        return getattr(self, "_last_dynamic", None)

    # -- Sensitivity analysis handling ---------------------------------------

    def _handle_sensitivity(self, assistant_text: str, sensitivity_spec: dict, client, types) -> str:
        """Run sensitivity analysis and feed results back to LLM."""
        try:
            result = run_sensitivity_analysis(
                model=self.model,
                analysis_type=sensitivity_spec.get("analysis_type", "single_sweep"),
                variable=sensitivity_spec.get("variable"),
                min_value=sensitivity_spec.get("min_value"),
                max_value=sensitivity_spec.get("max_value"),
                n_points=int(sensitivity_spec.get("n_points", 10)),
                response_kpis=sensitivity_spec.get("response_kpis"),
                variables=sensitivity_spec.get("variables"),
                response_kpi=sensitivity_spec.get("response_kpi"),
                variable_2=sensitivity_spec.get("variable_2"),
                min_2=sensitivity_spec.get("min_2"),
                max_2=sensitivity_spec.get("max_2"),
                n_2=sensitivity_spec.get("n_2"),
            )

            self._last_sensitivity = result
            results_text = format_sensitivity_result(result)

            self.history.append({"role": "assistant", "content": assistant_text})
            self.history.append({
                "role": "user",
                "content": (
                    f"[SYSTEM: Sensitivity analysis completed. Results below. "
                    f"Explain the sensitivity to the engineer. "
                    f"Highlight which variables have the most impact and the trends.]\n\n"
                    f"{results_text}"
                )
            })

            final_text = self._llm_followup(client, types)
            return final_text

        except Exception as e:
            self.history.append({"role": "assistant", "content": assistant_text})
            self.history.append({
                "role": "user",
                "content": f"[SYSTEM: Sensitivity analysis failed: {str(e)}. Inform the engineer.]"
            })
            return self._llm_followup(client, types)

    def get_last_sensitivity(self) -> Optional[SensitivityResult]:
        """Get the last sensitivity analysis result (for UI display)."""
        return getattr(self, "_last_sensitivity", None)

    # -- PVT simulation handling ---------------------------------------------

    def _handle_pvt(self, assistant_text: str, pvt_spec: dict, client, types) -> str:
        """Run PVT simulation and feed results back to LLM."""
        try:
            result = run_pvt_simulation(
                model=self.model,
                experiment=pvt_spec.get("experiment", "CME"),
                stream_name=pvt_spec.get("stream_name"),
                temperature_C=float(pvt_spec.get("temperature_C", 100)),
                p_start_bara=float(pvt_spec.get("p_start_bara", 400)),
                p_end_bara=float(pvt_spec.get("p_end_bara", 10)),
                n_steps=int(pvt_spec.get("n_steps", 20)),
                stages=pvt_spec.get("stages"),
            )

            self._last_pvt = result
            results_text = format_pvt_result(result)

            self.history.append({"role": "assistant", "content": assistant_text})
            self.history.append({
                "role": "user",
                "content": (
                    f"[SYSTEM: PVT simulation completed. Results below. "
                    f"Explain the PVT data to the engineer. "
                    f"Highlight the saturation point, key trends, and any notable observations.]\n\n"
                    f"{results_text}"
                )
            })

            final_text = self._llm_followup(client, types)
            return final_text

        except Exception as e:
            self.history.append({"role": "assistant", "content": assistant_text})
            self.history.append({
                "role": "user",
                "content": f"[SYSTEM: PVT simulation failed: {str(e)}. Inform the engineer.]"
            })
            return self._llm_followup(client, types)

    def get_last_pvt(self) -> Optional[PVTResult]:
        """Get the last PVT simulation result (for UI display)."""
        return getattr(self, "_last_pvt", None)

    # -- Safety analysis handling --------------------------------------------

    def _handle_safety(self, assistant_text: str, safety_spec: dict, client, types) -> str:
        """Run safety analysis and feed results back to LLM."""
        try:
            result = run_safety_analysis(
                model=self.model,
                design_pressure_factor=float(safety_spec.get("design_pressure_factor", 1.1)),
                include_fire=bool(safety_spec.get("include_fire", True)),
                include_blocked_outlet=bool(safety_spec.get("include_blocked_outlet", True)),
            )

            self._last_safety = result
            results_text = format_safety_result(result)

            self.history.append({"role": "assistant", "content": assistant_text})
            self.history.append({
                "role": "user",
                "content": (
                    f"[SYSTEM: Safety analysis completed. Results below. "
                    f"Explain the PSV sizing and relief scenarios to the engineer. "
                    f"Highlight the controlling case, required orifice sizes, "
                    f"and flare system design load.]\n\n{results_text}"
                )
            })

            final_text = self._llm_followup(client, types)
            return final_text

        except Exception as e:
            self.history.append({"role": "assistant", "content": assistant_text})
            self.history.append({
                "role": "user",
                "content": f"[SYSTEM: Safety analysis failed: {str(e)}. Inform the engineer.]"
            })
            return self._llm_followup(client, types)

    def get_last_safety(self) -> Optional[SafetyReport]:
        """Get the last safety analysis result (for UI display)."""
        return getattr(self, "_last_safety", None)

    # -- Flow assurance handling ---------------------------------------------

    def _handle_flow_assurance(self, assistant_text: str, fa_spec: dict, client, types) -> str:
        """Run flow assurance assessment and feed results back to LLM."""
        try:
            result = run_flow_assurance(
                model=self.model,
                check_hydrates=bool(fa_spec.get("check_hydrates", True)),
                check_wax=bool(fa_spec.get("check_wax", True)),
                check_corrosion=bool(fa_spec.get("check_corrosion", True)),
                inhibitor_type=fa_spec.get("inhibitor_type", "MEG"),
            )

            self._last_flow_assurance = result
            results_text = format_flow_assurance_result(result)

            self.history.append({"role": "assistant", "content": assistant_text})
            self.history.append({
                "role": "user",
                "content": (
                    f"[SYSTEM: Flow assurance assessment completed. Results below. "
                    f"Explain the risks to the engineer. Highlight any HIGH risk items, "
                    f"hydrate formation temperatures, corrosion rates, and recommended "
                    f"mitigation measures.]\n\n{results_text}"
                )
            })

            final_text = self._llm_followup(client, types)
            return final_text

        except Exception as e:
            self.history.append({"role": "assistant", "content": assistant_text})
            self.history.append({
                "role": "user",
                "content": f"[SYSTEM: Flow assurance assessment failed: {str(e)}. Inform the engineer.]"
            })
            return self._llm_followup(client, types)

    def get_last_flow_assurance(self) -> Optional[FlowAssuranceResult]:
        """Get the last flow assurance result (for UI display)."""
        return getattr(self, "_last_flow_assurance", None)

    # -- Helpers ------------------------------------------------------------

    def _llm_followup(self, client, types) -> str:
        """Make a follow-up LLM call and record the response.

        Strips any accidental tool/code blocks from the response so that
        the user never sees raw JSON specs.
        """
        contents = self._build_contents()
        response = client.models.generate_content(
            model=self.ai_model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=self._system_prompt,
                temperature=0.3,
            ),
        )
        final_text = _strip_tool_blocks(response.text)
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
        """Clear chat history and all cached results."""
        self.history.clear()
        self._last_comparison = None
        self._last_optimization = None
        self._last_risk_analysis = None
        self._last_chart = None
        self._chart_cache.clear()
        self._sized_equipment.clear()
        self._last_autosize = None
        self._last_script = None
        self._last_save_bytes = None
        self._last_emissions = None
        self._last_dynamic = None
        self._last_sensitivity = None
        self._last_pvt = None
        self._last_safety = None
        self._last_flow_assurance = None
        self._builder = None

