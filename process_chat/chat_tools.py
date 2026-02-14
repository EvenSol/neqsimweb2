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

    return f"""You are a process engineering assistant for an oil & gas process simulation.
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
{model_summary}

MODEL TAGS (for resolving engineer language to model objects):
{tag_ref}

{equip_templates}

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
  - "equipment_type": one of: cooler, heater, compressor, separator, valve, expander, pump, heat_exchanger, three_phase_separator
  - "insert_after": name of the existing unit to insert after (the new unit's inlet is the existing unit's outlet)
  - "params": configuration parameters for the new unit

Supported params by equipment type:
  cooler/heater: outlet_temperature_C, pressure_drop_bar
  compressor: outlet_pressure_bara, isentropic_efficiency (default 0.75)
  separator: (no special params needed)
  valve/expander: outlet_pressure_bara
  pump: outlet_pressure_bara, efficiency

You can combine "add_units" with "changes" in the same scenario (e.g., add a cooler AND change a compressor's pressure).
When the user asks to "add" or "install" equipment, use "add_units". When they ask to "change" or "modify", use "changes".

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
            if status == "OK":
                lines.append(f"  ✓ {entry['key']} = {entry.get('value', '?')}")
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
