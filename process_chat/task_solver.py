"""
Task Solver — Hosted wrapper around the NeqSim task-solving workflow.

Implements the 3-step workflow from the Task Solving Guide:
  Step 1: Scope & Research  →  task_spec (LLM-generated)
  Step 2: Analysis           →  LLM generates Python code → exec → fix loop
  Step 3: Report             →  structured results + downloadable report

The key idea: the LLM writes real Python code that calls the NeqSim API,
the system executes it, and if there's an error the LLM sees the traceback
and fixes the code — up to MAX_FIX_ATTEMPTS retries per step.

Designed for long-running tasks (5-10 min) with live progress callbacks.
"""

from __future__ import annotations

import json
import io
import re
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Task types matching the Task Solving Guide (A–G)
# ---------------------------------------------------------------------------
TASK_TYPES = {
    "A": "Property Calculation",
    "B": "Process Simulation",
    "C": "PVT Study",
    "D": "Standards Calculation",
    "E": "Phase Envelope / Equilibrium",
    "F": "Flow Assurance",
    "G": "Workflow (Multi-Step)",
}

TASK_TEMPLATES = [
    {
        "label": "Property lookup",
        "description": "Calculate density, viscosity, Cp, JT coefficient, etc.",
        "type": "A",
        "example": "What is the density of natural gas (methane 0.85, ethane 0.07, propane 0.05, CO2 0.03) at 100 bara and 40°C?",
    },
    {
        "label": "Process simulation",
        "description": "Size equipment, run a process flowsheet, compare scenarios.",
        "type": "B",
        "example": "Simulate 2-stage compression of natural gas from 30 to 200 bara with intercooling to 40°C. Report power consumption and outlet temperatures.",
    },
    {
        "label": "PVT study",
        "description": "CME, CVD, differential liberation, saturation pressure.",
        "type": "C",
        "example": "Run a constant mass expansion (CME) for a North Sea oil at reservoir temperature 100°C from 400 to 50 bara.",
    },
    {
        "label": "Gas quality / standards",
        "description": "Wobbe index, GCV, hydrocarbon dew point per ISO/GPA.",
        "type": "D",
        "example": "Calculate GCV, Wobbe index or hydrocarbon dew point and water dew point for a pipeline-quality gas.",
    },
    {
        "label": "Phase envelope",
        "description": "PT phase envelope, cricondenbar, cricondentherm.",
        "type": "E",
        "example": "Generate the phase envelope for a rich gas and find cricondenbar and cricondentherm.",
    },
    {
        "label": "Hydrate / flow assurance",
        "description": "Hydrate formation temperature, MEG dosing, wax onset.",
        "type": "F",
        "example": "What is the hydrate formation temperature for this gas at 100 bara? How much MEG is needed to inhibit?",
    },
    {
        "label": "Multi-step workflow",
        "description": "Combine multiple calculation types into a single report.",
        "type": "G",
        "example": "Full fluid characterization: phase envelope, key properties at pipeline conditions, hydrate check, and dew point analysis.",
    },
]

# Default lean gas composition
DEFAULT_COMPOSITION = {
    "methane": 0.85,
    "ethane": 0.07,
    "propane": 0.03,
    "i-butane": 0.005,
    "n-butane": 0.005,
    "CO2": 0.02,
    "nitrogen": 0.02,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TaskStep:
    """One step in the task execution plan."""
    number: int
    title: str
    description: str
    status: str = "pending"       # pending | running | done | error
    result_text: str = ""
    result_data: Optional[dict] = None
    error: str = ""
    elapsed_seconds: float = 0.0
    code: str = ""                # the Python code that was executed
    attempts: int = 0             # number of code-gen attempts (including fixes)


@dataclass
class TaskResult:
    """Complete result from running a task."""
    task_description: str = ""
    task_type: str = ""
    task_spec: str = ""
    steps: List[TaskStep] = field(default_factory=list)
    results_json: dict = field(default_factory=dict)
    figures: Dict[str, bytes] = field(default_factory=dict)
    report_html: str = ""
    report_text: str = ""
    total_elapsed: float = 0.0
    success: bool = False
    all_code: str = ""            # concatenated working code for download


# Progress callback type: (step_number, step_title, status_message)
ProgressCallback = Callable[[int, str, str], None]

# Maximum retries when code fails
MAX_FIX_ATTEMPTS = 4


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _call_llm(api_key: str, system_prompt: str, user_message: str,
              ai_model: str = "gemini-2.0-flash", temperature: float = 0.3) -> str:
    """Single LLM call, returns text."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=ai_model,
        contents=[{"role": "user", "parts": [{"text": user_message}]}],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
        ),
    )
    return response.text or ""


def _call_llm_multi(api_key: str, system_prompt: str,
                    messages: List[Dict[str, str]],
                    ai_model: str = "gemini-2.0-flash",
                    temperature: float = 0.3) -> str:
    """Multi-turn LLM call for the fix loop."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    response = client.models.generate_content(
        model=ai_model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
        ),
    )
    return response.text or ""


def _extract_json_block(text: str) -> Optional[dict]:
    """Extract the first JSON block from LLM text."""
    m = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _extract_python_block(text: str) -> Optional[str]:
    """Extract the first Python code block from LLM text."""
    m = re.search(r'```python\s*\n(.*?)\n```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'```\s*\n(.*?)\n```', text, re.DOTALL)
    if m:
        code = m.group(1).strip()
        if any(kw in code for kw in ("import ", "from ", "def ", "neqsim", "fluid")):
            return code
    return None


# ---------------------------------------------------------------------------
# NeqSim API Reference (embedded in system prompt for code generation)
# ---------------------------------------------------------------------------

NEQSIM_API_REFERENCE = r"""
## NeqSim Python API Reference

### Imports
```python
from neqsim.thermo import fluid_df, TPflash, phaseenvelope, dataFrame, fluid
from neqsim.thermo.thermoTools import fluidcreator, hydt
from neqsim import jneqsim
import pandas as pd
```

### 1. Creating Fluids

#### Method A: From DataFrame (recommended)
```python
df = pd.DataFrame({
    'ComponentName': ["methane", "ethane", "propane", "CO2", "nitrogen"],
    'MolarComposition[-]': [0.85, 0.07, 0.03, 0.03, 0.02],
    'MolarMass[kg/mol]': [None, None, None, None, None],
    'RelativeDensity[-]': [None, None, None, None, None],
})
neqsim_fluid = fluid_df(df, lastIsPlusFraction=False, add_all_components=False)
neqsim_fluid.autoSelectModel()  # auto-selects SRK/CPA based on components
```

#### Method B: From fluid() helper
```python
neqsim_fluid = fluid("srk")  # "srk", "pr", "cpa-srk", "UMR-PRU-EoS"
neqsim_fluid.addComponent("methane", 0.85)
neqsim_fluid.addComponent("ethane", 0.07)
neqsim_fluid.setMixingRule("classic")  # or mixing rule number: 2, 10 for CPA
```

#### Method C: Direct Java API
```python
neqsim_fluid = jneqsim.thermo.system.SystemSrkEos(273.15 + 25.0, 60.0)
# or SystemPrEos, SystemSrkCPAstatoil (for CPA), SystemUMRPRUMCEos (for UMR-PRU)
neqsim_fluid.addComponent("methane", 0.85)
neqsim_fluid.addComponent("ethane", 0.07)
neqsim_fluid.setMixingRule("classic")
neqsim_fluid.setMultiPhaseCheck(True)  # REQUIRED for water/heavy systems
```

#### Setting model by name
```python
neqsim_fluid.setModel("UMR-PRU-EoS")  # or "SRK-EoS", "PrEos", "CPAs-SRK-EOS"
```

### 2. Setting Conditions
```python
neqsim_fluid.setTemperature(40.0, "C")       # Celsius
neqsim_fluid.setTemperature(313.15)           # Kelvin (default unit)
neqsim_fluid.setPressure(100.0, "bara")       # bar absolute
neqsim_fluid.setPressure(100.0)               # bar (default)
```

### 3. Flash Calculations
```python
TPflash(neqsim_fluid)                        # T-P flash (most common)
# or via Java:
thermoOps = jneqsim.thermodynamicoperations.ThermodynamicOperations(neqsim_fluid)
thermoOps.TPflash()
thermoOps.PHflash(enthalpy_value)             # pressure-enthalpy flash
thermoOps.PSflash(entropy_value)              # pressure-entropy flash
thermoOps.bubblePointPressureFlash(False)      # bubble point
thermoOps.dewPointPressureFlash()              # dew point pressure
thermoOps.dewPointTemperatureFlash()           # dew point temperature
thermoOps.hydrateFormationTemperature()        # hydrate formation T
```

### 4. Reading Properties (AFTER flash + fluid.initProperties())
```python
# Always call initProperties() after flash if you need transport properties
neqsim_fluid.initProperties()

neqsim_fluid.getDensity("kg/m3")
neqsim_fluid.getMolarMass("kg/mol")
neqsim_fluid.getZ()                          # compressibility factor
neqsim_fluid.getEnthalpy("J/mol")            # or "kJ/kg"
neqsim_fluid.getEntropy("J/molK")
neqsim_fluid.getCp("J/molK")                 # or "kJ/kgK"
neqsim_fluid.getCv("J/molK")
neqsim_fluid.getSoundSpeed("m/s")
neqsim_fluid.getViscosity("kg/msec")         # dynamic viscosity Pa·s
neqsim_fluid.getThermalConductivity("W/mK")
neqsim_fluid.getTemperature("C")             # returns Celsius
neqsim_fluid.getPressure("bara")
neqsim_fluid.getNumberOfPhases()
neqsim_fluid.getVolume("m3")

# Phase-level access
gas_phase = neqsim_fluid.getPhase("gas")     # or getPhase(0)
liq_phase = neqsim_fluid.getPhase("oil")     # or getPhase(1)
aq_phase  = neqsim_fluid.getPhase("aqueous") # if present
phase.getDensity("kg/m3")
phase.getViscosity("kg/msec")
phase.getZ()
phase.getBeta()                              # phase fraction (mole basis)

# Get full results as DataFrame
results_df = dataFrame(neqsim_fluid)         # returns pandas DataFrame
```

### 5. Phase Envelope
```python
thermoOps = jneqsim.thermodynamicoperations.ThermodynamicOperations(neqsim_fluid)
thermoOps.calcPTphaseEnvelope2()

op = thermoOps.getOperation()
dewT  = [t - 273.15 for t in list(op.get("dewT"))]   # dew temps in C
dewP  = list(op.get("dewP"))                           # dew pressures in bara
bubT  = [t - 273.15 for t in list(op.get("bubT"))]
bubP  = list(op.get("bubP"))

# Cricondenbar = max pressure on envelope
# Cricondentherm = max temperature on envelope
```

### 6. Hydrate Calculations
```python
# MUST include water in composition, use CPA EOS
neqsim_fluid.setMultiPhaseCheck(True)
TPflash(neqsim_fluid)
thermoOps = jneqsim.thermodynamicoperations.ThermodynamicOperations(neqsim_fluid)
neqsim_fluid.setPressure(100.0, "bara")
thermoOps.hydrateFormationTemperature()
hydrate_T_C = neqsim_fluid.getTemperature("C")

# Or use hydrate helper:
from neqsim.thermo.thermoTools import hydt
hydrate_T = hydt(neqsim_fluid)  # returns hydrate formation temperature in C

# For hydrate equilibrium curve at multiple pressures:
pressures = [20, 50, 100, 150, 200]
for p in pressures:
    fl = neqsim_fluid.clone()
    fl.setPressure(p, "bara")
    ops = jneqsim.thermodynamicoperations.ThermodynamicOperations(fl)
    ops.hydrateFormationTemperature()
    print(f"P={p} bara -> T_hyd={fl.getTemperature('C'):.2f} C")
```

### 7. Gas Quality Standards (ISO 6976)
```python
standard = jneqsim.standards.gasquality.Standard_ISO6976(neqsim_fluid)
standard.calculate()
gcv = standard.getValue("GCV", "MJ/Sm3")
ncv = standard.getValue("NCV", "MJ/Sm3")
wobbe = standard.getValue("SuperiorWobbeIndex", "MJ/Sm3")
rel_density = standard.getValue("RelativeDensity")
molar_mass = standard.getValue("MolarMass", "kg/kmol")
```

### 8. Process Simulation
```python
from neqsim import jneqsim
eq = jneqsim.process.equipment

# Create fluid
fluid = jneqsim.thermo.system.SystemSrkEos(273.15 + 25.0, 50.0)
fluid.addComponent("methane", 0.85)
fluid.addComponent("ethane", 0.10)
fluid.addComponent("propane", 0.05)
fluid.setMixingRule("classic")

# Create process system
process = jneqsim.process.processmodel.ProcessSystem()

# Stream
feed = eq.stream.Stream("feed", fluid)
feed.setFlowRate(10000, "kg/hr")          # MUST set flow rate
process.add(feed)

# Separator
separator = eq.separator.Separator("inlet sep", feed)
process.add(separator)

# Compressor (outlet connected to separator gas out)
comp = eq.compressor.Compressor("compressor", separator.getGasOutStream())
comp.setOutletPressure(100.0, "bara")
comp.setIsentropicEfficiency(0.75)
process.add(comp)

# Cooler
cooler = eq.heatexchanger.Cooler("aftercooler", comp.getOutletStream())
cooler.setOutTemperature(35.0, "C")
process.add(cooler)

# Heater
heater = eq.heatexchanger.Heater("preheater", cooler.getOutletStream())
heater.setOutTemperature(60.0, "C")
process.add(heater)

# Valve (throttling / JT)
valve = eq.valve.ThrottlingValve("JT valve", heater.getOutletStream())
valve.setOutletPressure(30.0, "bara")
process.add(valve)

# Pump
pump = eq.pump.Pump("pump", separator.getLiquidOutStream())
pump.setOutletPressure(80.0, "bara")
process.add(pump)

# Heat exchanger (shell-and-tube, two sides)
hx = eq.heatexchanger.HeatExchanger("HX", feed)
hx.setFeedStream(1, other_stream)  # second feed
process.add(hx)

# Pipeline (Beggs & Brill)
pipe = eq.pipeline.PipeBeggsAndBrills("pipeline", feed)
pipe.setPipeWallRoughness(1.5e-5)
pipe.setLength(50000.0)     # meters
pipe.setDiameter(0.3)       # meters
pipe.setInletElevation(0.0)
pipe.setOutletElevation(0.0)
process.add(pipe)

# Run
process.run()

# Read results
comp.getOutletStream().getTemperature("C")
comp.getOutletStream().getPressure("bara")
comp.getPower("kW")                      # or "MW"
comp.getIsentropicEfficiency()
comp.getPolytropicEfficiency()
cooler.getDuty()                         # heat duty in watts
separator.getGasOutStream().getFlowRate("kg/hr")
separator.getLiquidOutStream().getFlowRate("kg/hr")
valve.getOutletStream().getTemperature("C")  # JT cooling effect
pump.getPower("kW")
```

#### Available equipment types:
- stream.Stream
- separator.Separator, separator.TwoPhaseSeparator, separator.ThreePhaseSeparator, separator.GasScrubber
- compressor.Compressor
- heatexchanger.Cooler, heatexchanger.Heater, heatexchanger.HeatExchanger
- valve.ThrottlingValve
- pump.Pump
- expander.Expander
- mixer.Mixer (use m.addStream(s) to add streams)
- splitter.Splitter
- pipeline.Pipeline, pipeline.AdiabaticPipe, pipeline.PipeBeggsAndBrills
- absorber.SimpleAbsorber, absorber.SimpleTEGAbsorber
- reactor.GibbsReactor
- util.Recycle, util.Adjuster

### 9. PVT Simulations
```python
# Saturation pressure
thermoOps = jneqsim.thermodynamicoperations.ThermodynamicOperations(neqsim_fluid)
thermoOps.bubblePointPressureFlash(False)
p_sat = neqsim_fluid.getPressure("bara")

# CME (Constant Mass Expansion) — manual stepping
for p in pressures:
    fl = neqsim_fluid.clone()
    fl.setPressure(p, "bara")
    TPflash(fl)
    fl.initProperties()
    density = fl.getDensity("kg/m3")
    z = fl.getZ()
```

### 10. Component Names (use standard NeqSim names)
Common: "methane", "ethane", "propane", "i-butane", "n-butane",
  "i-pentane", "n-pentane", "n-hexane", "nitrogen", "CO2", "H2S",
  "water", "MEG", "TEG", "methanol", "oxygen", "hydrogen", "helium"
Plus fractions: "C7", "C8", "C9", "C10", … (need MolarMass + RelativeDensity)
Ions: "Na+", "Cl-", "K+", "Ca++", "Mg++", "Ba++", "Sr++", "Fe++", "SO4--", "HCO3-"

### 11. Key Rules
- Temperature constructor args are in KELVIN: 273.15 + T_celsius
- getTemperature() returns KELVIN unless you pass "C"
- Always call setMixingRule() after adding all components
- Call setMultiPhaseCheck(True) for systems with water or multiple liquid phases
- For CPA EOS (water, MEG, methanol): use setMixingRule(10) or setMixingRule("cpa_mix")
- Call initProperties() after flash before reading transport properties (viscosity, conductivity)
- Set flow rate on streams before running process
- Use fluid.clone() when you need independent copies (e.g., parameter sweeps)

### 12. IMPORTANT: Output Format
Your code MUST end by building a `results` dict containing all computed values:
```python
results = {
    "key_results": { ... },
    "tables": { ... },      # optional: DataFrames as dicts
    "description": "...",   # brief text summary
}
```
The `results` variable will be read after execution.
Print important values with print() so they appear in the output log.
"""


# ---------------------------------------------------------------------------
# Safe code execution
# ---------------------------------------------------------------------------

_ALLOWED_MODULES = frozenset({
    "neqsim", "neqsim.thermo", "neqsim.thermo.thermoTools",
    "jneqsim", "pandas", "numpy", "math", "json",
    "scipy", "scipy.optimize",
})


def _execute_code(code: str, timeout_seconds: int = 300) -> dict:
    """Execute generated Python code in a restricted namespace.

    Returns {"success": bool, "results": dict|None, "output": str, "error": str}.
    """
    import io as _io
    import sys as _sys
    import contextlib

    # Build a namespace with common imports pre-loaded
    namespace = {"__builtins__": __builtins__}

    # Capture stdout
    stdout_buf = _io.StringIO()
    result = {"success": False, "results": None, "output": "", "error": ""}

    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(code, namespace)  # noqa: S102 — intentional exec of LLM code

        result["output"] = stdout_buf.getvalue()

        # Extract the results dict
        if "results" in namespace:
            raw = namespace["results"]
            # Convert to JSON-serializable form
            result["results"] = _to_serializable(raw)
        else:
            result["results"] = {"note": "Code ran successfully but did not define a 'results' dict."}

        result["success"] = True

    except Exception as e:
        result["output"] = stdout_buf.getvalue()
        result["error"] = traceback.format_exc()

    return result


def _to_serializable(obj, depth=0):
    """Convert an object to a JSON-serializable form (handles DataFrames, Java objects, etc.)."""
    if depth > 10:
        return str(obj)
    if obj is None:
        return None
    if isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v, depth + 1) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    try:
        return float(obj)
    except (TypeError, ValueError):
        pass
    return str(obj)


# ---------------------------------------------------------------------------
# Step 1: Scope & classify
# ---------------------------------------------------------------------------

_SCOPE_SYSTEM_PROMPT = """You are a thermodynamic / process engineering task planner for NeqSim.

Given a user's engineering question, produce a JSON task specification with these fields:

{
  "task_type": "A" | "B" | "C" | "D" | "E" | "F" | "G",
  "title": "short title",
  "description": "what the user wants",
  "complexity": "quick" | "standard" | "comprehensive",
  "composition": {"methane": 0.85, ...} or null if user didn't specify,
  "conditions": {"temperature_C": 40, "pressure_bara": 100, ...} or {},
  "parameters": { ... any extra parameters for the task ... },
  "steps": [
    {"title": "step title", "description": "detailed description — what to calculate, what API to use"},
    ...
  ],
  "deliverables": ["list of expected outputs"],
  "eos_model": "srk" | "pr" | "cpa" | "umr" | null
}

Rules:
- If composition is not specified, use a standard lean natural gas.
- If conditions are not specified, use typical pipeline conditions (40°C, 100 bara).
- For complex tasks (type G), break into 3-6 sub-steps. Each step gets its own code.
- For simple tasks (type A/D), use 1-2 steps.
- Choose CPA EOS when water, MEG, methanol, or TEG is present.
- Choose UMR-PRU for accurate phase envelopes.
- Each step description should be specific enough to write code from.
- Return ONLY the JSON (in a ```json block). No other text.
"""


def classify_and_scope(api_key: str, user_request: str,
                       user_composition: Optional[dict] = None,
                       ai_model: str = "gemini-2.0-flash") -> dict:
    """Step 1: Use LLM to classify the task and create a scope/spec."""
    extra = ""
    if user_composition:
        extra = f"\n\nUser-provided composition: {json.dumps(user_composition)}"

    response = _call_llm(
        api_key=api_key,
        system_prompt=_SCOPE_SYSTEM_PROMPT,
        user_message=user_request + extra,
        ai_model=ai_model,
    )

    spec = _extract_json_block(response)
    if spec is None:
        spec = {
            "task_type": "A",
            "title": user_request[:80],
            "description": user_request,
            "complexity": "quick",
            "composition": user_composition or DEFAULT_COMPOSITION,
            "conditions": {"temperature_C": 40, "pressure_bara": 100},
            "parameters": {},
            "steps": [
                {"title": "Calculate properties", "description": user_request},
            ],
            "deliverables": ["Results summary"],
            "eos_model": None,
        }

    if not spec.get("composition"):
        spec["composition"] = user_composition or DEFAULT_COMPOSITION

    return spec


# ---------------------------------------------------------------------------
# Step 2: Code generation + execution with fix loop
# ---------------------------------------------------------------------------

_CODE_GEN_SYSTEM_PROMPT = (
    "You are a NeqSim simulation engineer. You write Python code that uses the "
    "NeqSim Python library to solve thermodynamic and process engineering tasks.\n\n"
    "CRITICAL RULES:\n"
    "1. Write COMPLETE, RUNNABLE Python code — all imports at the top.\n"
    "2. Your code MUST define a `results` dict at the end containing all computed values.\n"
    "3. Use print() to show intermediate results so the user can follow progress.\n"
    "4. Use try/except around individual operations to be robust.\n"
    "5. Do NOT use matplotlib or any GUI. No plotting. Just compute and store numbers.\n"
    "6. Return the code inside a ```python ... ``` block.\n"
    "7. If you need to iterate over many conditions, do so in a loop.\n"
    "8. For process simulation: always set flow rate on streams.\n"
    "9. Always call initProperties() after flash before reading transport props.\n"
    "10. When fixing code: show the COMPLETE fixed code, not just the changed part.\n"
    "\n"
    + NEQSIM_API_REFERENCE
)


def generate_and_run_code(
    api_key: str,
    spec: dict,
    step_info: dict,
    step_number: int,
    ai_model: str = "gemini-2.0-flash",
    progress_cb: Optional[ProgressCallback] = None,
) -> TaskStep:
    """Generate Python code for a step, execute it, and fix errors if needed.

    Returns a TaskStep with status, code, results, and attempt count.
    """
    step_title = step_info.get("title", f"Step {step_number}")
    step_desc = step_info.get("description", "")

    task_step = TaskStep(
        number=step_number,
        title=step_title,
        description=step_desc,
        status="running",
    )

    def _progress(msg: str):
        if progress_cb:
            progress_cb(step_number, step_title, msg)

    # Build the initial prompt
    user_msg = (
        f"Task: {spec.get('description', '')}\n\n"
        f"Step {step_number}: {step_title}\n"
        f"Description: {step_desc}\n\n"
        f"Fluid composition: {json.dumps(spec.get('composition', {}))}\n"
        f"Conditions: {json.dumps(spec.get('conditions', {}))}\n"
        f"EOS model: {spec.get('eos_model', 'auto')}\n"
        f"Extra parameters: {json.dumps(spec.get('parameters', {}))}\n\n"
        f"Write complete Python code to perform this step. "
        f"The code must define a `results` dict at the end."
    )

    conversation = [{"role": "user", "content": user_msg}]

    t_start = time.time()

    for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
        task_step.attempts = attempt
        _progress(f"Generating code (attempt {attempt}/{MAX_FIX_ATTEMPTS})…")

        # Get code from LLM
        llm_response = _call_llm_multi(
            api_key=api_key,
            system_prompt=_CODE_GEN_SYSTEM_PROMPT,
            messages=conversation,
            ai_model=ai_model,
        )

        code = _extract_python_block(llm_response)
        if code is None:
            # LLM didn't return code — try the whole response as code
            if "import " in llm_response and "results" in llm_response:
                code = llm_response.strip()
            else:
                conversation.append({"role": "model", "content": llm_response})
                conversation.append({
                    "role": "user",
                    "content": (
                        "You did not return Python code in a ```python block. "
                        "Please write the complete Python code now."
                    ),
                })
                continue

        task_step.code = code

        # Execute the code
        _progress(f"Running code (attempt {attempt})…")
        exec_result = _execute_code(code)

        if exec_result["success"]:
            # Code worked
            task_step.status = "done"
            task_step.result_data = exec_result["results"]
            task_step.elapsed_seconds = time.time() - t_start

            # Build summary text
            output = exec_result["output"]
            if output:
                # Show last 500 chars of output
                task_step.result_text = output[-500:] if len(output) > 500 else output
            else:
                task_step.result_text = "Code executed successfully."

            _progress(f"✓ Code worked on attempt {attempt} ({task_step.elapsed_seconds:.1f}s)")
            return task_step

        # Code failed — send error to LLM for fixing
        error_text = exec_result["error"]
        output_text = exec_result["output"]

        _progress(f"Code failed (attempt {attempt}), sending error to LLM for fixing…")

        conversation.append({"role": "model", "content": f"```python\n{code}\n```"})

        fix_msg = (
            f"The code failed with this error:\n\n"
            f"```\n{error_text[-1500:]}\n```\n\n"
        )
        if output_text:
            fix_msg += f"Partial output before error:\n```\n{output_text[-500:]}\n```\n\n"

        fix_msg += (
            "Please fix the code. Common issues:\n"
            "- Wrong import path (use 'from neqsim import jneqsim' not 'import jneqsim')\n"
            "- Wrong package name: use 'thermodynamicoperations' (lowercase 'o')\n"
            "- Missing setMixingRule() after addComponent()\n"
            "- Temperature in Kelvin for constructors: 273.15 + T_celsius\n"
            "- Missing initProperties() before transport property access\n"
            "- Missing setMultiPhaseCheck(True) for water systems\n"
            "- Use getTemperature('C') not getTemperature() for Celsius\n"
            "- For CPA: use setMixingRule(10)\n"
            "\n"
            "Return the COMPLETE fixed Python code in a ```python block."
        )
        conversation.append({"role": "user", "content": fix_msg})

    # All attempts exhausted
    task_step.status = "error"
    task_step.error = f"Failed after {MAX_FIX_ATTEMPTS} attempts. Last error:\n{error_text[-500:]}"
    task_step.elapsed_seconds = time.time() - t_start
    _progress(f"✗ Failed after {MAX_FIX_ATTEMPTS} attempts")

    return task_step


# ---------------------------------------------------------------------------
# Step 3: Report generation
# ---------------------------------------------------------------------------

def _generate_report_text(api_key: str, spec: dict, all_results: dict,
                          all_code: str = "",
                          ai_model: str = "gemini-2.0-flash") -> str:
    """Use LLM to generate a readable engineering report from results."""
    report_prompt = """You are writing an engineering report based on NeqSim simulation results.

Write a clear, professional report with these sections:
1. **Executive Summary** — 2-3 sentence overview of key findings
2. **Problem Description** — what was asked
3. **Approach** — method, EOS model, assumptions
4. **Results** — present ALL key findings with numbers and units. Use markdown tables.
5. **Validation Notes** — physics checks, reasonableness observations
6. **Conclusions** — key takeaways and recommendations

Guidelines:
- Use proper engineering units (bara, °C, kg/m³, kJ/kg, etc.)
- Flag any unusual or unexpected results
- Note key assumptions
- Keep it concise but complete — include ALL numerical results
- Use markdown formatting with headers, tables, and bullet points
- Include relevant equations using LaTeX ($...$) where helpful
- If results include error data, note what failed and why
"""

    user_msg = f"""Task specification:
```json
{json.dumps(spec, indent=2, default=str)}
```

Simulation results:
```json
{json.dumps(all_results, indent=2, default=str)}
```

Write the engineering report now."""

    return _call_llm(
        api_key=api_key,
        system_prompt=report_prompt,
        user_message=user_msg,
        ai_model=ai_model,
    )


def _generate_html_report(title: str, report_md: str, spec: dict,
                           all_results: dict) -> str:
    """Wrap the markdown report in a styled HTML document."""
    import html as html_mod

    body_html = html_mod.escape(report_md)
    body_html = body_html.replace("\n", "<br>\n")
    body_html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', body_html)
    body_html = re.sub(r'#{3}\s*(.+?)(?:<br>)', r'<h3>\1</h3>', body_html)
    body_html = re.sub(r'#{2}\s*(.+?)(?:<br>)', r'<h2>\1</h2>', body_html)
    body_html = re.sub(r'#{1}\s*(.+?)(?:<br>)', r'<h1>\1</h1>', body_html)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{html_mod.escape(title)}</title>
<style>
    body {{ font-family: 'Segoe UI', Tahoma, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; line-height: 1.6; color: #333; }}
    h1 {{ color: #1a5276; border-bottom: 2px solid #1a5276; padding-bottom: 10px; }}
    h2 {{ color: #2e86c1; margin-top: 30px; }}
    h3 {{ color: #2874a6; }}
    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background-color: #2e86c1; color: white; }}
    tr:nth-child(even) {{ background-color: #f2f2f2; }}
    .meta {{ color: #888; font-size: 0.9em; }}
    pre {{ background: #f5f5f5; padding: 12px; border-radius: 4px; overflow-x: auto; }}
    code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 0.95em; }}
    .footer {{ margin-top: 40px; padding-top: 10px; border-top: 1px solid #ddd; color: #888; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>{html_mod.escape(title)}</h1>
<p class="meta">Generated by NeqSim Task Solver | Task type: {html_mod.escape(spec.get('task_type', 'N/A'))}</p>
<hr>
{body_html}
<div class="footer">
    <p>Report generated by NeqSim Web Task Solver. Results are based on thermodynamic simulations
    and should be validated against experimental data for engineering decisions.</p>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_task(
    api_key: str,
    user_request: str,
    user_composition: Optional[dict] = None,
    user_conditions: Optional[dict] = None,
    report_level: str = "standard",
    ai_model: str = "gemini-2.0-flash",
    progress_cb: Optional[ProgressCallback] = None,
) -> TaskResult:
    """
    Run a complete task: scope → generate code → execute → fix → report.

    The LLM generates Python code for each step, the system executes it,
    and if it fails the error is sent back to the LLM for fixing (up to
    MAX_FIX_ATTEMPTS retries per step).

    Parameters
    ----------
    api_key : str
        Gemini API key.
    user_request : str
        Natural language engineering question.
    user_composition : dict, optional
        Fluid composition {component: mole_fraction}.
    user_conditions : dict, optional
        Override conditions {temperature_C, pressure_bara, ...}.
    report_level : str
        "quick" | "standard" | "comprehensive"
    ai_model : str
        Gemini model name.
    progress_cb : callable, optional
        Called with (step_number, step_title, status_message).

    Returns
    -------
    TaskResult
    """
    result = TaskResult(task_description=user_request)
    t_start = time.time()

    def _progress(step_num: int, title: str, msg: str):
        if progress_cb:
            progress_cb(step_num, title, msg)

    # ── Step 1: Scope & Classify ──────────────────────────────────────────
    _progress(0, "Scope & Research", "Classifying task and building specification…")

    try:
        extra_ctx = ""
        if report_level == "comprehensive":
            extra_ctx = "\nUser wants a comprehensive / detailed report with full analysis."
        elif report_level == "quick":
            extra_ctx = "\nUser wants a quick answer. Minimize steps."

        spec = classify_and_scope(
            api_key=api_key,
            user_request=user_request + extra_ctx,
            user_composition=user_composition,
            ai_model=ai_model,
        )
    except Exception as e:
        result.steps.append(TaskStep(
            number=0, title="Scope & Research",
            description="Classify task", status="error",
            error=f"Failed to classify task: {e}",
        ))
        result.total_elapsed = time.time() - t_start
        return result

    # Apply user overrides
    if user_composition:
        spec["composition"] = user_composition
    if user_conditions:
        spec["conditions"] = {**spec.get("conditions", {}), **user_conditions}

    result.task_type = spec.get("task_type", "A")
    result.task_spec = json.dumps(spec, indent=2, default=str)

    scope_step = TaskStep(
        number=0, title="Scope & Research",
        description=f"Task type: {result.task_type} — {TASK_TYPES.get(result.task_type, '')}",
        status="done",
        result_text=(
            f"Classified as **Type {result.task_type}** "
            f"({TASK_TYPES.get(result.task_type, '')}).\n"
            f"Complexity: **{spec.get('complexity', 'standard')}**.\n"
            f"EOS: **{spec.get('eos_model') or 'auto-select'}**.\n"
            f"Steps planned: **{len(spec.get('steps', []))}**."
        ),
    )
    result.steps.append(scope_step)
    _progress(0, "Scope & Research", "✓ Task classified")

    # ── Step 2: Generate, execute, and fix code for each step ─────────────
    steps = spec.get("steps", [])
    all_results = {}
    all_code_parts = []

    for i, step_info in enumerate(steps):
        step_num = i + 1

        task_step = generate_and_run_code(
            api_key=api_key,
            spec=spec,
            step_info=step_info,
            step_number=step_num,
            ai_model=ai_model,
            progress_cb=progress_cb,
        )

        result.steps.append(task_step)

        if task_step.result_data:
            all_results[f"step_{step_num}"] = task_step.result_data

        if task_step.code:
            all_code_parts.append(
                f"# === Step {step_num}: {task_step.title} ===\n"
                f"# Attempts: {task_step.attempts}, Status: {task_step.status}\n\n"
                f"{task_step.code}\n"
            )

    result.results_json = all_results
    result.all_code = "\n\n".join(all_code_parts)

    # ── Step 3: Generate report ───────────────────────────────────────────
    report_step_num = len(steps) + 1
    _progress(report_step_num, "Generate Report", "Writing engineering report…")

    try:
        report_text = _generate_report_text(
            api_key=api_key,
            spec=spec,
            all_results=all_results,
            all_code=result.all_code,
            ai_model=ai_model,
        )
        result.report_text = report_text
        result.report_html = _generate_html_report(
            title=spec.get("title", user_request[:80]),
            report_md=report_text,
            spec=spec,
            all_results=all_results,
        )

        report_step = TaskStep(
            number=report_step_num,
            title="Generate Report",
            description="Create engineering report from results",
            status="done",
            result_text="Report generated successfully.",
        )
        result.steps.append(report_step)
        _progress(report_step_num, "Generate Report", "✓ Report ready")

    except Exception as e:
        report_step = TaskStep(
            number=report_step_num,
            title="Generate Report",
            description="Create engineering report",
            status="error",
            error=str(e),
        )
        result.steps.append(report_step)

    result.total_elapsed = time.time() - t_start
    result.success = all(s.status == "done" for s in result.steps)

    return result
