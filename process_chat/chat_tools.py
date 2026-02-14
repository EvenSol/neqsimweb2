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

- PROPERTY QUERY: Questions about detailed fluid/stream properties ("What is the TVP?", "What is the density?", "What is the viscosity?", "Show me the RVP of the feed gas", "What is the gas composition?")
  → These properties require running the simulation. Output a ```query ... ``` block (NOT ```json):
  ```query
  {{"properties": ["feed gas TVP", "feed gas RVP", "feed gas density"]}}
  ```
  The system will run the simulation and return ALL matching properties. Then explain the results.
  Search terms are matched case-insensitively against property keys. Use terms like:
    - Stream name + property: "feed gas TVP", "feed gas density", "feed gas viscosity"
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
- streams.<name>.pressure_bara — set stream pressure
- streams.<name>.temperature_C — set stream temperature
- streams.<name>.flow_kg_hr — set stream flow rate
- units.<name>.outletpressure_bara — set unit outlet pressure (compressors, valves)
- units.<name>.outletpressure_barg — set unit outlet pressure in barg
- units.<name>.outtemperature_C — set unit outlet temperature (heaters, coolers)
- units.<name>.outpressure_bara — set unit outlet pressure (heaters)
- units.<name>.outpressure_barg — set unit outlet pressure in barg
- units.<name>.isentropicEfficiency — set compressor efficiency

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
  cooler/heater/air_cooler/water_cooler: outlet_temperature_C, pressure_drop_bar
  compressor: outlet_pressure_bara, isentropic_efficiency (default 0.75)
  separator/two_phase_separator/three_phase_separator/gas_scrubber: (no special params needed)
  valve/control_valve: outlet_pressure_bara
  expander: outlet_pressure_bara, isentropic_efficiency
  pump/esp_pump: outlet_pressure_bara, efficiency
  mixer: (no params needed, combines streams)
  splitter: split_fractions (list of fractions summing to 1.0)
  pipeline/adiabatic_pipe: length_m, diameter_m, roughness_m
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


def format_comparison_for_llm(comparison) -> str:
    """
    Format a scenario comparison result into text the LLM can use
    to explain results to the engineer.
    """
    from .scenario_engine import results_summary_table
    
    lines = []

    # Key changes — highlight KPIs that changed significantly
    if comparison.delta_kpis:
        sig_changes = [d for d in comparison.delta_kpis 
                       if d.get('delta') is not None and abs(d['delta']) > 0.01]
        if sig_changes:
            lines.append("=== KEY CHANGES (base → scenario) ===")
            for d in sorted(sig_changes, key=lambda x: abs(x.get('delta_pct', 0) or 0), reverse=True):
                pct_str = f" ({d['delta_pct']:+.1f}%)" if d.get('delta_pct') is not None else ""
                lines.append(f"  {d['kpi']}: {d['base']:.2f} → {d['case']:.2f} [{d['unit']}] "
                             f"(delta={d['delta']:+.2f}{pct_str})")
            lines.append("")
    
    # Summary table
    summary_df = results_summary_table(comparison)
    if not summary_df.empty:
        lines.append("=== FULL SIMULATION RESULTS ===")
        lines.append(summary_df.to_string(index=False))
    
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
    """

    def __init__(self, model: NeqSimProcessModel, api_key: str, ai_model: str = "gemini-2.0-flash"):
        self.model = model
        self.api_key = api_key
        self.ai_model = ai_model
        self.history: List[Dict[str, str]] = []
        self._system_prompt = build_system_prompt(model)

    def chat(self, user_message: str) -> str:
        """
        Process a user message and return the assistant response.
        
        If the LLM produces a scenario JSON, it's automatically executed
        and the results are fed back to the LLM for explanation.
        """
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)

        # Build conversation
        self.history.append({"role": "user", "content": user_message})

        # First LLM call: classify intent and possibly produce scenario JSON
        contents = self._build_contents()
        
        response = client.models.generate_content(
            model=self.ai_model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=self._system_prompt,
                temperature=0.3,  # Low temperature for reliable structured output
            ),
        )

        assistant_text = response.text

        # Check if the LLM produced a scenario to run
        scenario_data = extract_scenario_json(assistant_text)
        property_query = extract_property_query(assistant_text)
        
        if scenario_data:
            # Execute the scenarios
            try:
                scenarios = scenarios_from_json(scenario_data)
                
                from .scenario_engine import run_scenarios
                comparison = run_scenarios(self.model, scenarios)
                
                results_text = format_comparison_for_llm(comparison)
                
                # Feed results back to LLM for explanation
                self.history.append({"role": "assistant", "content": assistant_text})
                self.history.append({
                    "role": "user",
                    "content": f"[SYSTEM: Simulation completed. Results below. Explain these results to the engineer concisely.]\n\n{results_text}"
                })
                
                contents = self._build_contents()
                response2 = client.models.generate_content(
                    model=self.ai_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=self._system_prompt,
                        temperature=0.3,
                    ),
                )
                
                final_text = response2.text
                self.history.append({"role": "assistant", "content": final_text})
                
                # Store comparison for UI access
                self._last_comparison = comparison
                return final_text

            except Exception as e:
                error_msg = f"Simulation failed: {str(e)}\n{traceback.format_exc()}"
                self.history.append({"role": "assistant", "content": assistant_text})
                self.history.append({
                    "role": "user",
                    "content": f"[SYSTEM: Simulation failed with error: {str(e)}. Please inform the engineer and suggest corrections.]"
                })
                
                contents = self._build_contents()
                response2 = client.models.generate_content(
                    model=self.ai_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=self._system_prompt,
                        temperature=0.3,
                    ),
                )
                
                final_text = response2.text
                self.history.append({"role": "assistant", "content": final_text})
                return final_text
        elif property_query:
            # Property query — run model and extract matching properties
            try:
                queries = property_query.get("properties", [])
                all_results = []
                for q in queries:
                    result_text = self.model.query_properties(q)
                    all_results.append(result_text)
                
                properties_text = "\n\n".join(all_results)
                
                # Feed results back to LLM for explanation
                self.history.append({"role": "assistant", "content": assistant_text})
                self.history.append({
                    "role": "user",
                    "content": f"[SYSTEM: Property query completed. Results below. Present these results clearly to the engineer.]\n\n{properties_text}"
                })
                
                contents = self._build_contents()
                response2 = client.models.generate_content(
                    model=self.ai_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=self._system_prompt,
                        temperature=0.3,
                    ),
                )
                
                final_text = response2.text
                self.history.append({"role": "assistant", "content": final_text})
                self._last_comparison = None
                return final_text
                
            except Exception as e:
                self.history.append({"role": "assistant", "content": assistant_text})
                self.history.append({
                    "role": "user",
                    "content": f"[SYSTEM: Property query failed: {str(e)}. Inform the engineer.]"
                })
                contents = self._build_contents()
                response2 = client.models.generate_content(
                    model=self.ai_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=self._system_prompt,
                        temperature=0.3,
                    ),
                )
                final_text = response2.text
                self.history.append({"role": "assistant", "content": final_text})
                return final_text
        else:
            # Pure Q&A response (no scenario needed)
            self.history.append({"role": "assistant", "content": assistant_text})
            self._last_comparison = None
            return assistant_text

    def get_last_comparison(self):
        """Get the last scenario comparison result (for UI display)."""
        return getattr(self, "_last_comparison", None)

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
