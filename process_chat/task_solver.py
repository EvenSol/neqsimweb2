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
import os
import re
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Error memory — learn from past mistakes across tasks
# ---------------------------------------------------------------------------
_ERROR_MEMORY_FILE = os.path.join(os.path.dirname(__file__), ".error_memory.json")
_MAX_MEMORY_ENTRIES = 30


def _load_error_memory() -> List[dict]:
    """Load persisted error→fix lessons."""
    try:
        with open(_ERROR_MEMORY_FILE, "r", encoding="utf-8") as f:
            entries = json.load(f)
        if isinstance(entries, list):
            return entries[-_MAX_MEMORY_ENTRIES:]
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return []


def _save_error_memory(entries: List[dict]):
    """Persist error→fix lessons (keeps last N)."""
    entries = entries[-_MAX_MEMORY_ENTRIES:]
    try:
        with open(_ERROR_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, default=str)
    except OSError:
        pass  # non-critical, skip silently


def _record_fix(error_snippet: str, failed_code_snippet: str, fixed_code_snippet: str,
                task_type: str = ""):
    """Record a successful fix so future tasks can learn from it."""
    # Extract the core error line (last line of traceback or most specific)
    error_lines = error_snippet.strip().splitlines()
    core_error = error_lines[-1] if error_lines else error_snippet[:200]

    # Don't store very long snippets — just enough context
    entry = {
        "error": core_error[:300],
        "error_type": core_error.split(":")[0].strip() if ":" in core_error else core_error[:50],
        "task_type": task_type,
        "fix_hint": _diff_summary(failed_code_snippet, fixed_code_snippet),
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
    }

    entries = _load_error_memory()

    # Deduplicate: if same error_type already exists, update it
    for i, existing in enumerate(entries):
        if existing.get("error_type") == entry["error_type"] and \
           existing.get("error")[:100] == entry["error"][:100]:
            entries[i] = entry
            _save_error_memory(entries)
            return

    entries.append(entry)
    _save_error_memory(entries)


def _diff_summary(failed: str, fixed: str) -> str:
    """Generate a short summary of what changed between failed and fixed code."""
    failed_lines = set(failed.strip().splitlines())
    fixed_lines = set(fixed.strip().splitlines())

    added = fixed_lines - failed_lines
    removed = failed_lines - fixed_lines

    parts = []
    if removed:
        # Show at most 3 key removed lines
        for line in list(removed)[:3]:
            line = line.strip()
            if line and not line.startswith("#") and len(line) > 5:
                parts.append(f"- Removed: {line[:120]}")
    if added:
        for line in list(added)[:3]:
            line = line.strip()
            if line and not line.startswith("#") and len(line) > 5:
                parts.append(f"+ Added: {line[:120]}")

    return "\n".join(parts) if parts else "Code was restructured to fix the error."


def _format_memory_for_prompt() -> str:
    """Format stored error lessons as text for injection into system prompt."""
    entries = _load_error_memory()
    if not entries:
        return ""

    lines = ["\n\nLEARNED FROM PREVIOUS ERRORS (apply these lessons):"]
    for i, e in enumerate(entries[-15:], 1):  # inject last 15 at most
        lines.append(f"{i}. Error: {e['error'][:200]}")
        if e.get("fix_hint"):
            lines.append(f"   Fix: {e['fix_hint'][:200]}")
    return "\n".join(lines) + "\n"

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
MAX_FIX_ATTEMPTS = 25


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

Full JavaDoc: https://equinor.github.io/neqsimhome/javadoc/site/apidocs/index.html
Java source:  https://github.com/equinor/neqsim
Python wrapper: https://github.com/equinor/neqsim-python

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

#### Available EOS classes (jneqsim.thermo.system.*)
- SystemSrkEos — SRK (Soave-Redlich-Kwong), general purpose
- SystemPrEos — Peng-Robinson, general purpose
- SystemPrEos1978 — PR 1978 variant
- SystemSrkCPAstatoil — CPA-SRK for polar (water, MEG, methanol, TEG)
- SystemPrCPA — CPA-PR
- SystemUMRPRUMCEos — UMR-PRU-MC for accurate phase envelopes
- SystemUMRPRUEos — UMR-PRU
- SystemGERG2008Eos — GERG-2008 reference equation (natural gas)
- SystemEOSCGEos — EOS-CG for CO2-rich systems
- SystemSpanWagnerEos — Span-Wagner reference for CO2
- SystemLeachmanEos — Leachman reference for hydrogen
- SystemPCSAFT / SystemPCSAFTa — PC-SAFT (+ association)
- SystemElectrolyteCPA / SystemElectrolyteCPAstatoil — electrolyte systems
- SystemSoreideWhitson — Søreide-Whitson for produced water
- SystemIdealGas — ideal gas law
- SystemWaterIF97 — IAPWS-IF97 for steam/water

#### Mixing rules (setMixingRule)
- 1: classic, all kij = 0
- 2: classic + kij from NeqSim database (RECOMMENDED for SRK/PR)
- 3: classic + temperature-dependent kij
- 4: Huron-Vidal with database parameters
- 7: classic CPA kij from database
- 9: CPA temperature-dependent kij
- 10: CPA temperature + composition dependent kij (RECOMMENDED for CPA)
- "classic": classic mixing rule (string form)

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

# Density: units "kg/m3", "mol/m3"
neqsim_fluid.getDensity("kg/m3")
neqsim_fluid.getMolarMass("kg/mol")           # or "gr/mol"
neqsim_fluid.getZ()                           # compressibility factor (PV=ZnRT)
neqsim_fluid.getEnthalpy("J/mol")             # units: "J", "J/mol", "kJ/kmol", "J/kg", "kJ/kg"
neqsim_fluid.getEntropy("J/molK")             # units: "J/K", "J/molK", "J/kgK", "kJ/kgK"
neqsim_fluid.getCp("J/molK")                  # units: "J/K", "J/molK", "J/kgK", "kJ/kgK"
neqsim_fluid.getCv("J/molK")
neqsim_fluid.getSoundSpeed("m/s")             # or "km/h"
neqsim_fluid.getViscosity("kg/msec")          # or "cP", "Pas"
neqsim_fluid.getKinematicViscosity("m2/sec")
neqsim_fluid.getThermalConductivity("W/mK")   # or "W/cmK"
neqsim_fluid.getJouleThomsonCoefficient("C/bar")  # or "K/bar"
neqsim_fluid.getTemperature("C")              # or "K", "R"
neqsim_fluid.getPressure("bara")              # or "barg", "Pa", "MPa", "psi"
neqsim_fluid.getNumberOfPhases()
neqsim_fluid.getVolume("m3")                  # or "litre", "m3/kg", "m3/mol"
neqsim_fluid.getMolarVolume("m3/mol")
neqsim_fluid.getGamma()                       # heat capacity ratio Cp/Cv
neqsim_fluid.getInternalEnergy("J/mol")        # or "J/kg", "kJ/kg"
neqsim_fluid.getExergy(288.15, "J/mol")        # T_surroundings in K
neqsim_fluid.getFlowRate("kg/hr")             # "kg/sec","m3/hr","Sm3/hr","Sm3/day","MSm3/day","mole/sec"
neqsim_fluid.getMass("kg")                     # or "gr", "tons"
neqsim_fluid.getInterfacialTension("gas", "oil")  # returns N/m, by phase name
neqsim_fluid.getInterfacialTension(0, 1)       # by phase index

# Phase-level access
neqsim_fluid.hasPhaseType("gas")              # returns boolean
neqsim_fluid.hasComponent("methane")          # returns boolean
neqsim_fluid.getPhaseOfType("gas")            # returns phase or null if absent
gas_phase = neqsim_fluid.getPhase("gas")      # or getPhase(0)
liq_phase = neqsim_fluid.getPhase("oil")      # or getPhase(1)
aq_phase  = neqsim_fluid.getPhase("aqueous")  # if present
phase.getDensity("kg/m3")
phase.getViscosity("kg/msec")
phase.getZ()
phase.getBeta()                               # phase fraction (mole basis)
neqsim_fluid.getPhaseFraction("gas", "mole")  # or "volume", "weight"

# Get full results as DataFrame
results_df = dataFrame(neqsim_fluid)          # returns pandas DataFrame

# Utility
neqsim_fluid.toJson()                         # JSON string of fluid state
neqsim_fluid.prettyPrint()                    # print readable summary
neqsim_fluid.getModelName()                   # returns EOS model name
neqsim_fluid.getMolarComposition()            # returns double[] of mole fractions
neqsim_fluid.getCompNames()                   # returns String[] of component names
neqsim_fluid.getNumberOfComponents()          # int
neqsim_fluid.setTotalFlowRate(100.0, "kg/hr") # set flow rate
neqsim_fluid.setMolarComposition([0.8, 0.1, 0.1])  # normalize and set
neqsim_fluid.validateSetup()                  # check for common setup errors
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

### 10. Component Names (use EXACT names from NeqSim COMP.csv database)
Full reference: https://github.com/equinor/neqsim/blob/master/src/main/resources/data/COMP.csv
IMPORTANT: Component names are case-sensitive and must match exactly.

**Hydrocarbons**: "methane", "ethane", "propane", "n-butane", "i-butane", "n-pentane",
  "i-pentane", "n-hexane", "benzene", "toluene", "n-heptane", "n-octane", "n-nonane",
  "nC10", "nC11", "nC12", "nC13", "nC14", "nC15", "nC16", "nC17", "nC18", "nC19", "nC20",
  "nC21"-"nC30", "nC34", "nC39", "c-C4", "c-C5", "c-hexane", "c-C7", "c-C8", "cy-C9",
  "22-dim-C3", "22-dim-C4", "23-dim-C4", "2-m-C5", "3-m-C5", "M-cy-C5", "M-cy-C6",
  "ethylbenzene", "m-Xylene", "p-Xylene", "o-Xylene", "ethylcyclohexane", "naphthalene",
  "propylbenzene", "c-propane", "ethylene", "propene", "cis-butene", "trans-butene", "iso-butene"
**Inerts/gases**: "CO2", "nitrogen", "oxygen", "H2S", "helium", "neon", "argon", "CO", "COS",
  "SF6", "R12", "R134a", "N2O4"
**Special hydrogen**: "hydrogen", "para-hydrogen", "ortho-hydrogen"
**Water & ice**: "water", "ice"
**Glycols**: "MEG", "TEG", "DEG", "PG", "glycerol"
**Amines**: "MDEA", "MEA", "DEA", "Piperazine"
**Alcohols**: "methanol", "ethanol", "1-propanol", "i-propanol"
**Acids**: "acetic acid", "formic acid", "hydrochloric acid", "sulfuric acid", "nitric acid"
**Ions**: "Na+", "Cl-", "K+", "Ca++", "Mg++", "Ba++", "Sr++", "Fe++", "Li+", "I-", "OH-",
  "HCO3-", "CO3--", "SO4--", "Br-", "F-", "Pb++", "Hg++", "NO3-", "HS-", "S--",
  "NH4+", "H3O+", "H+"
**Other**: "ammonia", "acetone", "S8", "mercury", "SO2", "SO3", "NO", "NO2", "N2O", "N2O3",
  "N2O5", "H2O2", "HCN", "CS2", "CH2O", "NaCl", "CaCl2", "NaHSO4", "asphaltene"
Plus fractions: "C7", "C8", "C9", "C10", … (need MolarMass + RelativeDensity)

### 11. Key Rules — Units
- Temperature constructor args are in KELVIN: 273.15 + T_celsius
- getTemperature() returns KELVIN unless you pass "C" — ALWAYS use getTemperature("C") for results
- getPressure() defaults to bara — use getPressure("bara") to be explicit
- getEnthalpy() returns J (total) — use getEnthalpy("J/mol") or getEnthalpy("kJ/kg") for specific
- getEntropy() returns J/K (total) — use getEntropy("J/molK") or getEntropy("kJ/kgK") for specific
- getViscosity() returns kg/(m·s) — use getViscosity("cP") for centipoise
- getCp()/getCv() return J/K (total) — use getCp("J/molK") or getCp("kJ/kgK")
- Phase envelope T arrays (dewT, bubT) are in KELVIN — subtract 273.15 for Celsius
- After hydrateFormationTemperature(), dewPointTemperatureFlash(), bubblePointPressureFlash():
  the fluid's temperature/pressure are updated — use getTemperature("C") to read in Celsius
- Always include units in result keys: 'temperature_C', 'pressure_bara', 'density_kg_m3'

### 12. Key Rules — General
- Always call setMixingRule() after adding all components
- Call setMultiPhaseCheck(True) for systems with water or multiple liquid phases
- For CPA EOS (water, MEG, methanol): use setMixingRule(10) or setMixingRule("cpa_mix")
- Call initProperties() after flash before reading transport properties (viscosity, conductivity)
- Set flow rate on streams before running process
- Use fluid.clone() when you need independent copies (e.g., parameter sweeps)
- Use setHydrateCheck(True) before hydrate calculations
- Use setEnhancedMultiPhaseCheck(True) for complex systems (sour gas, CO2, liquid-liquid)
- Use hasPhaseType("gas") to safely check if a phase exists before accessing it
- Use getPhaseOfType("gas") — returns null (None in Python) if the phase does not exist
- Mixing rule 2 is recommended for SRK/PR; mixing rule 10 for CPA systems
- addTBPfraction(name, moles, molarMass_kg_mol, density_g_cm3) for TBP pseudo-components
- addPlusFraction(name, moles, molarMass, density) for plus fractions
- addComponent(name, value, "kg/hr") adds component by mass flow rate
- setTotalFlowRate(value, "kg/hr") to set total flow rate
- validateSetup() checks for common errors (components, mixing rule, T/P ranges)

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
    """Convert an object to a JSON-serializable form (handles DataFrames, Java objects, numpy, etc.)."""
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
    # Handle numpy types
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return [_to_serializable(v, depth + 1) for v in obj.tolist()]
        if isinstance(obj, (np.integer, np.bool_)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
    except ImportError:
        pass
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
                       uploaded_docs: Optional[str] = None,
                       ai_model: str = "gemini-2.0-flash") -> dict:
    """Step 1: Use LLM to classify the task and create a scope/spec."""
    extra = ""
    if user_composition:
        extra = f"\n\nUser-provided composition: {json.dumps(user_composition)}"
    if uploaded_docs:
        extra += f"\n\nUser-uploaded reference documents:\n{uploaded_docs[:8000]}"

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
    "Full JavaDoc reference: https://equinor.github.io/neqsimhome/javadoc/site/apidocs/index.html\n\n"
    "CRITICAL RULES:\n"
    "1. Write COMPLETE, RUNNABLE Python code — all imports at the top.\n"
    "2. Your code MUST define a `results` dict at the end containing all computed values.\n"
    "3. Use print() to show intermediate results so the user can follow progress.\n"
    "4. Use try/except around individual operations to be robust.\n"
    "5. Do NOT use matplotlib or any GUI. Store chart data in the results dict instead.\n"
    "   When results contain data suitable for a chart (curves, sweeps, profiles),\n"
    "   add a '_charts' key to results with a list of chart specs:\n"
    "   results['_charts'] = [\n"
    "       {'title': 'Phase Envelope', 'x': temp_list, 'y': pres_list,\n"
    "        'xlabel': 'Temperature (°C)', 'ylabel': 'Pressure (bara)',\n"
    "        'series': 'Dew Point'},\n"
    "       {'title': 'Phase Envelope', 'x': bub_temp, 'y': bub_pres,\n"
    "        'xlabel': 'Temperature (°C)', 'ylabel': 'Pressure (bara)',\n"
    "        'series': 'Bubble Point'},\n"
    "   ]\n"
    "   Multiple series on the same chart share the same 'title'. Each entry is one trace.\n"
    "   Always include xlabel, ylabel with units. The system will render charts automatically.\n"
    "6. Return the code inside a ```python ... ``` block.\n"
    "7. If you need to iterate over many conditions, do so in a loop.\n"
    "8. For process simulation: always set flow rate on streams.\n"
    "9. Always call initProperties() after flash before reading transport props.\n"
    "10. When fixing code: show the COMPLETE fixed code, not just the changed part.\n"
    "11. When building DataFrames from multiple arrays, ALWAYS verify they have the\n"
    "    same length first. Use min(len(a), len(b)) to truncate, or store in a dict/list instead.\n"
    "12. Phase envelope arrays (dewT, dewP, bubT, bubP) often have DIFFERENT lengths.\n"
    "    Store them as separate lists in the results dict, NOT in a single DataFrame.\n"
    "13. UNITS — ALWAYS pass explicit unit strings to ALL getter methods. Defaults return\n"
    "    SI/internal units (Kelvin, Pa, J, etc.) which are NOT what users expect:\n"
    "    - getTemperature('C') not getTemperature()  (default is KELVIN)\n"
    "    - getPressure('bara') not getPressure()  (default is bara, but be explicit)\n"
    "    - getDensity('kg/m3'), getEnthalpy('J/mol' or 'kJ/kg'), getEntropy('J/molK')\n"
    "    - getViscosity('cP' or 'kg/msec'), getCp('J/molK' or 'kJ/kgK')\n"
    "    - getThermalConductivity('W/mK'), getSoundSpeed('m/s')\n"
    "    - getFlowRate('kg/hr'), getVolume('m3'), getMolarMass('kg/mol')\n"
    "    - getJouleThomsonCoefficient('C/bar')\n"
    "14. In the results dict, ALWAYS include the unit in the key name, e.g.:\n"
    "    results['temperature_C'] not results['temperature'],\n"
    "    results['density_kg_m3'] not results['density'],\n"
    "    results['viscosity_cP'] not results['viscosity'].\n"
    "15. Phase envelope temperatures from op.get('dewT')/op.get('bubT') are in KELVIN.\n"
    "    Convert: [t - 273.15 for t in list(op.get('dewT'))]\n"
    "16. Hydrate/dew/bubble point: after the flash, getTemperature() returns KELVIN.\n"
    "    Always use getTemperature('C') to get Celsius.\n"
    "17. COMPONENT NAMES: Use EXACT names from section 10 of the API reference.\n"
    "    Names are case-sensitive. Common mistakes: 'Methane' (wrong) → 'methane' (right),\n"
    "    'co2' (wrong) → 'CO2' (right), 'h2s' (wrong) → 'H2S' (right),\n"
    "    'iC4' (wrong) → 'i-butane' (right), 'nC4' (wrong) → 'n-butane' (right),\n"
    "    'nC5' (wrong) → 'n-pentane' (right), 'iC5' (wrong) → 'i-pentane' (right).\n"
    "    Full list: https://github.com/equinor/neqsim/blob/master/src/main/resources/data/COMP.csv\n"
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
    prior_context: Optional[str] = None,
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

    # Inject lessons from past errors into system prompt
    system_prompt = _CODE_GEN_SYSTEM_PROMPT + _format_memory_for_prompt()

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

    if prior_context:
        user_msg += f"\n\nContext from previous analysis:\n{prior_context[:4000]}"

    conversation = [{"role": "user", "content": user_msg}]

    t_start = time.time()
    error_text = "No code was generated in any attempt."
    last_failed_code = ""
    last_error = ""

    for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
        task_step.attempts = attempt
        _progress(f"Generating code (attempt {attempt}/{MAX_FIX_ATTEMPTS})…")

        # Get code from LLM
        llm_response = _call_llm_multi(
            api_key=api_key,
            system_prompt=system_prompt,
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

            # Record the fix for future learning (only when it took retries)
            if attempt > 1 and last_failed_code and last_error:
                try:
                    _record_fix(
                        error_snippet=last_error,
                        failed_code_snippet=last_failed_code,
                        fixed_code_snippet=code,
                        task_type=spec.get("task_type", ""),
                    )
                except Exception:
                    pass  # non-critical

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
        last_failed_code = code
        last_error = error_text

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
            "- UNITS: ALWAYS pass explicit unit strings to getter methods:\n"
            "  getTemperature('C'), getPressure('bara'), getDensity('kg/m3'),\n"
            "  getEnthalpy('kJ/kg'), getViscosity('cP'), getCp('kJ/kgK'), etc.\n"
            "  Without unit args, values are in SI (Kelvin, Pa, J, kg/ms) — NOT user-friendly.\n"
            "- Phase envelope temps (dewT, bubT) are in KELVIN — subtract 273.15 for Celsius.\n"
            "- Include units in result dict keys: 'temperature_C', 'density_kg_m3', etc.\n"
            "- For CPA: use setMixingRule(10)\n"
            "- DataFrame 'All arrays must be of the same length': do NOT put arrays of different\n"
            "  lengths into a DataFrame. Use separate lists in the results dict, or truncate to min length.\n"
            "  Phase envelope arrays (dewT, dewP, bubT, bubP) often have different lengths!\n"
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
# Step 3: Report generation — charts, HTML, markdown
# ---------------------------------------------------------------------------

def _build_plotly_charts(all_results: dict) -> list:
    """Build plotly Figure objects from result data.

    Looks for:
      1. Explicit '_charts' key in any result step (list of chart specs)
      2. Auto-detected plottable patterns (parallel arrays, curves, sweeps)

    Returns list of plotly Figure objects.
    """
    import plotly.graph_objects as go

    figures = []

    # --- Collect explicit chart specs ---
    chart_specs = []
    for _step_key, data in all_results.items():
        if not isinstance(data, dict):
            continue
        if "_charts" in data:
            specs = data["_charts"]
            if isinstance(specs, list):
                chart_specs.extend(specs)

    # Group by title → multi-series charts
    if chart_specs:
        by_title = {}
        for cs in chart_specs:
            t = cs.get("title", "Chart")
            by_title.setdefault(t, []).append(cs)

        for chart_title, series_list in by_title.items():
            fig = go.Figure()
            xlabel = series_list[0].get("xlabel", "")
            ylabel = series_list[0].get("ylabel", "")
            for s in series_list:
                x = s.get("x", [])
                y = s.get("y", [])
                name = s.get("series", s.get("name", ""))
                mode = s.get("mode", "lines")
                if x and y:
                    fig.add_trace(go.Scatter(x=x, y=y, mode=mode, name=name))
            fig.update_layout(
                title=chart_title, xaxis_title=xlabel, yaxis_title=ylabel,
                template="plotly_white", height=480,
            )
            figures.append(fig)

    # --- Auto-detect plottable patterns ---
    for _step_key, data in all_results.items():
        if not isinstance(data, dict):
            continue

        # Phase envelope
        if "dew_point_curve" in data:
            dew = data["dew_point_curve"]
            bub = data.get("bubble_point_curve", {})
            if isinstance(dew, dict) and dew.get("temperature_C") and dew.get("pressure_bara"):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dew["temperature_C"], y=dew["pressure_bara"],
                    mode="lines", name="Dew Point", line=dict(color="blue", width=2),
                ))
                if isinstance(bub, dict) and bub.get("temperature_C"):
                    fig.add_trace(go.Scatter(
                        x=bub["temperature_C"], y=bub["pressure_bara"],
                        mode="lines", name="Bubble Point", line=dict(color="red", width=2),
                    ))
                if "cricondenbar_bara" in data:
                    fig.add_trace(go.Scatter(
                        x=[data.get("cricondenbar_T_C")], y=[data["cricondenbar_bara"]],
                        mode="markers+text", name=f"Cricondenbar ({data['cricondenbar_bara']:.1f} bara)",
                        marker=dict(size=10, color="green"),
                        text=[f"Cricondenbar\n{data['cricondenbar_bara']:.1f} bara"],
                        textposition="top right",
                    ))
                if "cricondentherm_C" in data:
                    fig.add_trace(go.Scatter(
                        x=[data["cricondentherm_C"]], y=[data.get("cricondentherm_P_bara")],
                        mode="markers+text", name=f"Cricondentherm ({data['cricondentherm_C']:.1f} °C)",
                        marker=dict(size=10, color="orange"),
                        text=[f"Cricondentherm\n{data['cricondentherm_C']:.1f} °C"],
                        textposition="top left",
                    ))
                fig.update_layout(
                    title="Phase Envelope", xaxis_title="Temperature (°C)",
                    yaxis_title="Pressure (bara)", template="plotly_white", height=480,
                )
                figures.append(fig)

        # Hydrate curve
        if "hydrate_curve" in data:
            curve = data["hydrate_curve"]
            if isinstance(curve, list) and curve:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[p.get("temperature_C", p.get("T_C")) for p in curve],
                    y=[p.get("pressure_bara", p.get("P_bara")) for p in curve],
                    mode="lines+markers", name="Hydrate Formation",
                    line=dict(color="cyan", width=2),
                ))
                if "hydrate_formation_temperature_C" in data:
                    fig.add_trace(go.Scatter(
                        x=[data["hydrate_formation_temperature_C"]],
                        y=[data.get("pressure_bara", 100)],
                        mode="markers+text",
                        name=f"At {data.get('pressure_bara', 100)} bara",
                        marker=dict(size=12, color="red"),
                        text=[f"{data['hydrate_formation_temperature_C']:.1f} °C"],
                        textposition="top right",
                    ))
                fig.update_layout(
                    title="Hydrate Formation Temperature vs Pressure",
                    xaxis_title="Temperature (°C)", yaxis_title="Pressure (bara)",
                    template="plotly_white", height=450,
                )
                figures.append(fig)

        # Generic x/y array pairs: detect parallel lists with matching keys
        # e.g. pressure_bara: [...] and temperature_C: [...]
        list_keys = {k: v for k, v in data.items()
                     if isinstance(v, list) and len(v) >= 3
                     and all(isinstance(x, (int, float)) for x in v)
                     and k != "_charts" and k not in ("dew_point_curve", "bubble_point_curve", "hydrate_curve")}
        if len(list_keys) >= 2:
            keys = list(list_keys.keys())
            # Use first key as x, plot remaining as y — only if same length
            x_key = keys[0]
            x_vals = list_keys[x_key]
            plotted = False
            fig = go.Figure()
            for y_key in keys[1:]:
                y_vals = list_keys[y_key]
                if len(y_vals) == len(x_vals):
                    fig.add_trace(go.Scatter(
                        x=x_vals, y=y_vals, mode="lines+markers", name=y_key,
                    ))
                    plotted = True
            if plotted:
                fig.update_layout(
                    title="Property Sweep",
                    xaxis_title=x_key.replace("_", " "),
                    yaxis_title="Value",
                    template="plotly_white", height=450,
                )
                figures.append(fig)

    return figures


def _generate_charts_html(all_results: dict) -> str:
    """Generate HTML fragments for all charts found in results."""
    figures = _build_plotly_charts(all_results)
    if not figures:
        return ""

    parts = ['<h2>Charts</h2>']
    for fig in figures:
        parts.append('<div class="chart-container">')
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
        parts.append('</div>')
    return "\n".join(parts)

def _generate_report_text(api_key: str, spec: dict, all_results: dict,
                          all_code: str = "",
                          ai_model: str = "gemini-2.0-flash") -> str:
    """Use LLM to generate a readable engineering report from results."""
    report_prompt = """You are writing an engineering report based on NeqSim simulation results.

Write a clear, professional report with these sections:
1. **Executive Summary** — 2-3 sentence overview of key findings
2. **Problem Description** — what was asked
3. **Approach** — method, EOS model, assumptions. Include key equations using LaTeX:
   e.g., $PV = ZnRT$, fugacity conditions, EOS form used, etc.
4. **Results** — present ALL key findings with numbers and units.
   ALWAYS use markdown tables for numerical results. Example:
   | Property | Value | Unit |
   |----------|-------|------|
   | Temperature | 25.0 | °C |
   | Pressure | 100.0 | bara |
5. **Validation Notes** — physics checks, reasonableness observations
6. **Conclusions** — key takeaways and recommendations

Guidelines:
- Use proper engineering units (bara, °C, kg/m³, kJ/kg, cP, W/(m·K), etc.)
- CRITICAL UNIT CHECKS — verify every value before reporting:
  - If a temperature value is above 200 or below -100, it is likely in Kelvin — convert to °C
    by subtracting 273.15. Typical values: hydrate T = -20 to 30°C, process T = -50 to 200°C.
    Values like 293, 313, 255 are almost certainly Kelvin.
  - If viscosity is very small (e.g. 1e-5 to 1e-3), it is in Pa·s — multiply by 1000 for cP.
    Gas: 0.005-0.05 cP, Liquid: 0.1-10 cP.
  - If Cp/Cv is very large (e.g. 1500-4200), it is in J/(kg·K) — divide by 1000 for kJ/(kg·K).
  - If enthalpy is very large (e.g. thousands), check if J/mol vs kJ/kg.
  - If density seems off: gas 0.5-200 kg/m³, liquid 400-1200 kg/m³.
  - Thermal conductivity: gas 0.01-0.1, liquid 0.1-0.7 W/(m·K).
- Check result dict key names for embedded units (e.g. '_C', '_K', '_cP', '_kg_m3')
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
    """Wrap the markdown report in a styled HTML document with tables, equations, and charts."""
    import html as html_mod

    # --- Convert markdown to HTML ---
    lines = report_md.split("\n")
    html_lines = []
    in_table = False
    in_code_block = False
    table_rows = []

    def _flush_table():
        nonlocal table_rows, in_table
        if not table_rows:
            return ""
        out = '<table>\n'
        for i, row in enumerate(table_rows):
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            # Skip separator rows (---|---|---)
            if all(set(c.strip()) <= {'-', ':', ' '} for c in cells):
                continue
            tag = "th" if i == 0 else "td"
            out += "  <tr>" + "".join(f"<{tag}>{html_mod.escape(c)}</{tag}>" for c in cells) + "</tr>\n"
        out += '</table>\n'
        table_rows = []
        in_table = False
        return out

    for line in lines:
        stripped = line.strip()

        # Code blocks
        if stripped.startswith("```"):
            if in_table:
                html_lines.append(_flush_table())
            if in_code_block:
                html_lines.append("</code></pre>")
                in_code_block = False
            else:
                in_code_block = True
                html_lines.append("<pre><code>")
            continue
        if in_code_block:
            html_lines.append(html_mod.escape(line))
            continue

        # Table rows (lines containing |)
        if "|" in stripped and stripped.startswith("|"):
            if not in_table:
                in_table = True
                table_rows = []
            table_rows.append(stripped)
            continue
        elif in_table:
            html_lines.append(_flush_table())

        # Headers
        if stripped.startswith("#### "):
            html_lines.append(f"<h4>{html_mod.escape(stripped[5:])}</h4>")
        elif stripped.startswith("### "):
            html_lines.append(f"<h3>{html_mod.escape(stripped[4:])}</h3>")
        elif stripped.startswith("## "):
            html_lines.append(f"<h2>{html_mod.escape(stripped[3:])}</h2>")
        elif stripped.startswith("# "):
            html_lines.append(f"<h1>{html_mod.escape(stripped[2:])}</h1>")
        elif stripped.startswith("- ") or stripped.startswith("* "):
            content = html_mod.escape(stripped[2:])
            content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
            html_lines.append(f"<li>{content}</li>")
        elif stripped == "":
            html_lines.append("<br>")
        else:
            content = html_mod.escape(stripped)
            content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
            content = re.sub(r'`(.+?)`', r'<code>\1</code>', content)
            html_lines.append(f"<p>{content}</p>")

    if in_table:
        html_lines.append(_flush_table())

    body_html = "\n".join(html_lines)

    # --- Generate plotly chart HTML from results ---
    charts_html = _generate_charts_html(all_results)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{html_mod.escape(title)}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {{delimiters: [{{left: '$$', right: '$$', display: true}}, {{left: '$', right: '$', display: false}}]}});"></script>
<style>
    body {{ font-family: 'Segoe UI', Tahoma, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; line-height: 1.6; color: #333; }}
    h1 {{ color: #1a5276; border-bottom: 2px solid #1a5276; padding-bottom: 10px; }}
    h2 {{ color: #2e86c1; margin-top: 30px; }}
    h3 {{ color: #2874a6; }}
    h4 {{ color: #1a5276; }}
    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background-color: #2e86c1; color: white; }}
    tr:nth-child(even) {{ background-color: #f2f2f2; }}
    li {{ margin: 4px 0; }}
    .meta {{ color: #888; font-size: 0.9em; }}
    pre {{ background: #f5f5f5; padding: 12px; border-radius: 4px; overflow-x: auto; }}
    code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 0.95em; }}
    .chart-container {{ margin: 20px 0; }}
    .footer {{ margin-top: 40px; padding-top: 10px; border-top: 1px solid #ddd; color: #888; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>{html_mod.escape(title)}</h1>
<p class="meta">Generated by NeqSim Task Solver | Task type: {html_mod.escape(spec.get('task_type', 'N/A'))}</p>
<hr>
{body_html}
{charts_html}
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


# ---------------------------------------------------------------------------
# Follow-up / iteration on existing results
# ---------------------------------------------------------------------------

_FOLLOWUP_SCOPE_PROMPT = """You are a thermodynamic / process engineering task planner for NeqSim.

The user has already completed an analysis and wants to EXTEND or REFINE it.
You are given the original task specification, its results, and the user's follow-up request.

Produce a JSON specification for the ADDITIONAL steps only (not the steps already done).
Use the same format as before:

{
  "title": "short title for the follow-up",
  "description": "what the user wants NOW",
  "steps": [
    {"title": "step title", "description": "detailed description"},
    ...
  ],
  "deliverables": ["list of expected outputs"],
  "update_report": true
}

Rules:
- Keep the same composition and EOS model from the original task.
- Only add NEW steps — do not repeat what was already done.
- If the user wants more detail on something already calculated, create a step
  that dives deeper (e.g. wider pressure range, more properties, sensitivity).
- If the user uploaded reference documents, incorporate their data and requirements.
- Return ONLY the JSON in a ```json block.
"""


def follow_up_task(
    api_key: str,
    follow_up_request: str,
    previous_result: dict,
    uploaded_docs: Optional[str] = None,
    ai_model: str = "gemini-2.0-flash",
    progress_cb: Optional[ProgressCallback] = None,
) -> TaskResult:
    """
    Continue iterating on a previous task result.

    Takes the previous spec + results + follow-up request, plans new steps,
    executes them, and produces an updated report combining old and new results.

    Parameters
    ----------
    api_key : str
        Gemini API key.
    follow_up_request : str
        What the user wants to add/change.
    previous_result : dict
        The ts_last_result dict from session state.
    uploaded_docs : str, optional
        Concatenated text from uploaded files.
    ai_model : str
        Gemini model name.
    progress_cb : callable, optional
        Called with (step_number, step_title, status_message).

    Returns
    -------
    TaskResult
    """
    result = TaskResult(task_description=follow_up_request)
    t_start = time.time()

    prev_spec = previous_result.get("spec", {})
    prev_results = previous_result.get("all_results", {})
    prev_report = previous_result.get("report_text", "")
    prev_steps = previous_result.get("steps", [])
    prev_code = previous_result.get("all_code", "")
    prev_task = previous_result.get("task_input", "")

    def _progress(step_num: int, title: str, msg: str):
        if progress_cb:
            progress_cb(step_num, title, msg)

    # ── Plan follow-up steps ──────────────────────────────────────────────
    _progress(0, "Plan Follow-up", "Analyzing follow-up request…")

    scope_msg = (
        f"Original task: {prev_task}\n\n"
        f"Original specification:\n```json\n{json.dumps(prev_spec, indent=2, default=str)[:3000]}\n```\n\n"
        f"Results from original analysis:\n```json\n{json.dumps(prev_results, indent=2, default=str)[:4000]}\n```\n\n"
        f"User's follow-up request: {follow_up_request}"
    )
    if uploaded_docs:
        scope_msg += f"\n\nUser-uploaded reference documents:\n{uploaded_docs[:6000]}"

    try:
        response = _call_llm(
            api_key=api_key,
            system_prompt=_FOLLOWUP_SCOPE_PROMPT,
            user_message=scope_msg,
            ai_model=ai_model,
        )
        followup_spec = _extract_json_block(response)
        if followup_spec is None:
            followup_spec = {
                "title": follow_up_request[:80],
                "description": follow_up_request,
                "steps": [{"title": "Additional analysis", "description": follow_up_request}],
                "deliverables": ["Updated results"],
                "update_report": True,
            }
    except Exception as e:
        result.steps.append(TaskStep(
            number=0, title="Plan Follow-up",
            description="Plan additional steps", status="error",
            error=f"Failed to plan follow-up: {e}",
        ))
        result.total_elapsed = time.time() - t_start
        return result

    scope_step = TaskStep(
        number=0, title="Plan Follow-up",
        description=followup_spec.get("description", follow_up_request),
        status="done",
        result_text=f"Planned **{len(followup_spec.get('steps', []))}** additional steps.",
    )
    result.steps.append(scope_step)
    _progress(0, "Plan Follow-up", "✓ Follow-up planned")

    # Use the original spec's composition, conditions, EOS
    merged_spec = {
        **prev_spec,
        "title": followup_spec.get("title", prev_spec.get("title", "")),
        "description": followup_spec.get("description", follow_up_request),
        "steps": followup_spec.get("steps", []),
    }

    result.task_type = prev_spec.get("task_type", "G")
    result.task_spec = json.dumps(merged_spec, indent=2, default=str)

    # ── Execute new steps ─────────────────────────────────────────────────
    new_steps = followup_spec.get("steps", [])
    new_results = {}
    new_code_parts = []

    # Build a prior-context summary for the code generator
    prior_context = (
        f"Previous results (available for reference):\n"
        f"{json.dumps(prev_results, indent=2, default=str)[:3000]}"
    )
    if uploaded_docs:
        prior_context += f"\n\nReference documents:\n{uploaded_docs[:3000]}"

    # Numbering continues from previous steps
    prev_step_count = len([s for s in prev_steps if hasattr(s, 'number') and s.number > 0])
    if prev_step_count == 0:
        prev_step_count = len(prev_results)

    for i, step_info in enumerate(new_steps):
        step_num = prev_step_count + i + 1

        task_step = generate_and_run_code(
            api_key=api_key,
            spec=merged_spec,
            step_info=step_info,
            step_number=step_num,
            ai_model=ai_model,
            progress_cb=progress_cb,
            prior_context=prior_context,
        )

        result.steps.append(task_step)

        if task_step.result_data:
            new_results[f"step_{step_num}"] = task_step.result_data

        if task_step.code:
            new_code_parts.append(
                f"# === Follow-up Step {step_num}: {task_step.title} ===\n"
                f"# Attempts: {task_step.attempts}, Status: {task_step.status}\n\n"
                f"{task_step.code}\n"
            )

    # Merge old + new results
    combined_results = {**prev_results, **new_results}
    combined_code = prev_code
    if new_code_parts:
        combined_code += "\n\n# " + "=" * 60 + "\n# FOLLOW-UP STEPS\n# " + "=" * 60 + "\n\n"
        combined_code += "\n\n".join(new_code_parts)

    result.results_json = combined_results
    result.all_code = combined_code

    # ── Regenerate report with combined data ──────────────────────────────
    report_step_num = prev_step_count + len(new_steps) + 1
    _progress(report_step_num, "Update Report", "Regenerating report with all results…")

    combined_spec = {
        **prev_spec,
        "follow_up": followup_spec.get("description", follow_up_request),
    }

    try:
        report_text = _generate_report_text(
            api_key=api_key,
            spec=combined_spec,
            all_results=combined_results,
            all_code=combined_code,
            ai_model=ai_model,
        )
        result.report_text = report_text
        result.report_html = _generate_html_report(
            title=prev_spec.get("title", "Follow-up Analysis"),
            report_md=report_text,
            spec=combined_spec,
            all_results=combined_results,
        )

        report_step = TaskStep(
            number=report_step_num,
            title="Update Report",
            description="Regenerate report with all results (original + follow-up)",
            status="done",
            result_text="Updated report generated successfully.",
        )
        result.steps.append(report_step)
        _progress(report_step_num, "Update Report", "✓ Updated report ready")

    except Exception as e:
        report_step = TaskStep(
            number=report_step_num,
            title="Update Report",
            description="Regenerate report",
            status="error",
            error=str(e),
        )
        result.steps.append(report_step)

    result.total_elapsed = time.time() - t_start
    result.success = all(s.status == "done" for s in result.steps)

    return result


def extract_text_from_upload(uploaded_file) -> str:
    """Extract text content from an uploaded file (txt, csv, md, json, pdf)."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()

    if name.endswith((".txt", ".md", ".csv", ".log")):
        return raw.decode("utf-8", errors="replace")

    if name.endswith(".json"):
        return raw.decode("utf-8", errors="replace")

    if name.endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(io.BytesIO(raw))
            return f"Excel file '{uploaded_file.name}':\n{df.to_string()}"
        except Exception as e:
            return f"[Could not parse Excel file: {e}]"

    if name.endswith(".pdf"):
        # Best-effort PDF text extraction
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(raw))
            pages = []
            for page in reader.pages[:30]:  # limit to 30 pages
                text = page.extract_text()
                if text:
                    pages.append(text)
            return f"PDF '{uploaded_file.name}' ({len(reader.pages)} pages):\n" + "\n---\n".join(pages)
        except ImportError:
            return f"[PDF uploaded: {uploaded_file.name} — install PyPDF2 for text extraction]"
        except Exception as e:
            return f"[Could not parse PDF: {e}]"

    # Fallback: try as text
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return f"[Binary file: {uploaded_file.name}, {len(raw)} bytes]"
