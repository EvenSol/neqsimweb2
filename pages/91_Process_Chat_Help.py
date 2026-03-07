"""
Process Chat — Help & Documentation

Comprehensive guide to all capabilities of the Process Chat module.
"""
import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from theme import apply_theme, theme_toggle

st.set_page_config(
    page_title="Process Chat Help",
    page_icon="images/neqsimlogocircleflat.png",
    layout="wide",
)
apply_theme()
theme_toggle()

st.title("📖 Process Chat — Documentation")
st.markdown("Complete guide to all features, tools, and capabilities of the Process Chat module.")

# ─────────────────────────────────────────────
# Table of Contents (sidebar)
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("📑 Contents")
    st.markdown("""
- [Getting Started](#getting-started)
- [Building a Process](#building-a-process)
- [What-If Scenarios](#what-if-scenarios)
- [Analysis Tools (28)](#analysis-tools)
- [Equipment Types (36)](#equipment-types)
- [EOS Models (8)](#equation-of-state-models)
- [Sensitivity Analysis](#sensitivity-analysis)
- [PVT Simulation](#pvt-simulation)
- [Dynamic Simulation](#dynamic-simulation)
- [Safety & Risk](#safety-risk-analysis)
- [Emissions & Energy](#emissions-energy)
- [Flow Assurance](#flow-assurance)
- [Compressor Charts](#compressor-performance-charts)
- [DEXPI P&ID Integration](#dexpi-p-id-integration)
- [Custom Python Code](#custom-python-code-neqsim-code)
- [External Data](#external-data-integration)
- [Saving & Loading](#saving-loading-models)
- [Example Prompts](#example-prompts-by-category)
""")

st.divider()

# ═══════════════════════════════════════════════
#  GETTING STARTED
# ═══════════════════════════════════════════════
st.header("Getting Started", anchor="getting-started")
st.markdown("""
Process Chat is an AI-powered interface for the **NeqSim** thermodynamic process simulation engine.
You interact with your process model through natural language — the AI interprets your request,
selects the right tool, runs the calculation, and presents the results.

**Three ways to start:**

| Method | How | Best For |
|--------|-----|----------|
| **Upload a model** | Drag a `.neqsim` file to the sidebar uploader | Analyzing existing designs |
| **Load test process** | Click **📂 Load Test Process** in the sidebar | Quick exploration |
| **Build from scratch** | Click **🔨 Start New Process**, then describe what you want | New designs |

**Requirements:**
- A **Google Gemini API key** (free at [aistudio.google.com](https://aistudio.google.com/))
- Enter the key in the sidebar under **AI Settings**
""")

st.divider()

# ═══════════════════════════════════════════════
#  BUILDING A PROCESS
# ═══════════════════════════════════════════════
st.header("Building a Process", anchor="building-a-process")
st.markdown("""
Describe any process in natural language. The AI will design the flowsheet, select equipment,
configure the fluid, and run the simulation.

**Example prompts:**
```
Build a gas compression process with 80% methane, 10% ethane, 5% propane, 3% CO2, 2% N2
at 30°C and 50 bara. Compress to 150 bara in two stages with intercooling.
```
```
Create a gas dehydration process with TEG absorber for natural gas at 70 bara
```
```
Build a refrigeration cycle with propane as refrigerant
```

**What happens behind the scenes:**
1. The AI creates a **build specification** (JSON) with fluid composition, EOS model, and equipment list
2. **ProcessBuilder** converts this into a NeqSim `ProcessSystem`
3. The simulation runs to convergence
4. Results are presented with a process flow diagram

**After building, you can:**
- Ask what-if questions to explore alternatives
- View and download the Python script
- Save the model as a `.neqsim` file
- Run any of the 28 analysis tools
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Incremental Building")
    st.markdown("""
    You can add equipment to an existing process:
    ```
    Add a cooler after the compressor to cool to 35°C
    ```
    ```
    Add a second compression stage to 200 bara
    ```
    ```
    Add MEG injection upstream of the pipeline
    ```
    """)

with col2:
    st.subheader("Script & Export")
    st.markdown("""
    ```
    Show the Python script
    ```
    Generates a standalone NeqSim Python script that reproduces
    the entire process. You can run it outside of the web app.

    ```
    Save the process
    ```
    Creates a `.neqsim` file download link in the sidebar.
    """)

st.divider()

# ═══════════════════════════════════════════════
#  WHAT-IF SCENARIOS
# ═══════════════════════════════════════════════
st.header("What-If Scenarios", anchor="what-if-scenarios")
st.markdown("""
Ask any "what if" question and the AI will patch the model, re-run the simulation,
and show a before/after comparison.

**Supported changes:**

| Parameter | Example Prompt |
|-----------|---------------|
| Stream pressure | *"What if the feed pressure drops to 40 bara?"* |
| Stream temperature | *"What if inlet temperature increases to 45°C?"* |
| Stream flow rate | *"What if we increase feed flow by 20%?"* |
| Equipment outlet pressure | *"What if we increase compressor discharge to 120 bara?"* |
| Equipment outlet temperature | *"What if we cool to 25°C instead of 35°C?"* |
| Compressor efficiency | *"What if compressor efficiency drops to 70%?"* |
| Compressor speed | *"What if we increase compressor speed by 5%?"* |
| Valve Cv | *"What if we install a larger control valve?"* |
| Multiple changes | *"What if feed pressure drops 5 bar AND temperature rises 10°C?"* |

**Relative changes:**
- *"Increase feed flow by 15%"* → applies a `scale` factor of 1.15
- *"Add 5 bara to export pressure"* → applies an `add` offset

**Target-seeking (solver):**
- *"What feed rate gives 50 MW total compressor power?"* → bisection solver finds the answer

**Multi-scenario comparison:**
- *"Compare performance at 40, 50, and 60 bara feed pressure"*
""")

st.divider()

# ═══════════════════════════════════════════════
#  ANALYSIS TOOLS
# ═══════════════════════════════════════════════
st.header("Analysis Tools", anchor="analysis-tools")
st.markdown("Process Chat includes **28 specialized analysis tools**. Simply ask in natural language — the AI selects the right tool automatically.")

tools_data = [
    ("🔧 Build", "build", "Create or modify a process from scratch", "Build a gas compression process with natural gas at 50 bara"),
    ("🔀 What-If Scenario", "json", "Run what-if changes, compare before/after", "What if we increase export pressure to 120 bara?"),
    ("🔍 Property Query", "query", "Fetch stream/equipment properties", "What is the gas density at the export?"),
    ("📈 Optimization", "optimize", "Find maximum throughput before equipment limits", "Find maximum production for this process"),
    ("⚠️ Risk Analysis", "risk", "Equipment criticality, risk matrix, Monte Carlo", "Show the risk matrix for this process"),
    ("📊 Compressor Chart", "chart", "Performance curves, surge limits, stonewall", "Show compressor performance map with surge line"),
    ("📐 Auto-Size", "autosize", "Equipment sizing report, utilization ratios", "Size all equipment and show utilization"),
    ("🌍 Emissions", "emissions", "CO₂/CH₄ footprint, scope 1/2/3", "Calculate CO₂ emissions for this process"),
    ("⚡ Dynamic Simulation", "dynamic", "Blowdown, startup, shutdown, transient", "Run a blowdown simulation on the separator"),
    ("📉 Sensitivity", "sensitivity", "Parameter sweeps, tornado charts, 2D surface", "Sweep feed pressure from 30 to 70 bara"),
    ("🧪 PVT Simulation", "pvt", "CME, CVD, differential liberation", "Run a CME test at 100°C"),
    ("🛡️ Safety Analysis", "safety", "PSV sizing, relief scenarios (API 520/521)", "Size the PSVs for all vessels"),
    ("🌊 Flow Assurance", "flowassurance", "Hydrate, wax, corrosion risk", "Check for hydrate risk in the export pipeline"),
    ("♻️ Energy Integration", "energy_integration", "Pinch analysis, composite curves", "Run pinch analysis with ΔTmin = 10°C"),
    ("📉 Turndown", "turndown", "Operating envelope, min/max stable flow", "What is the turndown range for this process?"),
    ("📊 Performance Monitor", "performance_monitor", "Actual vs simulated, degradation alerts", "Compare these actual readings with the model"),
    ("🔓 Debottleneck", "debottleneck", "Identify bottleneck, upgrade options", "Where is the bottleneck in this process?"),
    ("🎓 Training Scenarios", "training", "Operator upset simulations with Q&A", "Generate a cooling water failure training scenario"),
    ("⚡ Energy Audit", "energy_audit", "Utility balance, specific energy, benchmarks", "Run an energy audit for this process"),
    ("🔥 Flare Analysis", "flare_analysis", "Flare sources, recovery, carbon tax exposure", "Analyze flare gas and recovery options"),
    ("📅 Multi-Period", "multi_period", "Seasonal/annual production planning", "Compare summer vs winter performance"),
    ("🌤️ Weather", "weather", "Live weather, forecast impact on coolers", "What is the weather in Stavanger? Impact on coolers?"),
    ("🔬 Lab Import", "lab_import", "Update feed from lab data (CSV/JSON/inline)", "Update feed composition: 85% methane, 7% ethane..."),
    ("🛢️ Production Scenarios", "production", "GOR sweep, water cut, well blending", "Sweep GOR from 500 to 2000"),
    ("📋 Report", "report", "Structured JSON report export", "Generate a process report"),
    ("📌 Signal Tracker", "tracker", "Track KPIs across chat turns", "Track compressor power and export temperature"),
    ("📐 DEXPI Import", "dexpi", "Parse & import DEXPI P&ID XML", "Analyze the uploaded DEXPI P&ID"),
    ("🐍 Custom Code", "neqsim_code", "Run arbitrary NeqSim Python code", "Calculate the phase envelope for the feed gas"),
]

# Display as a nice table
st.markdown("| | Tool | Code | Description | Example Prompt |")
st.markdown("|---|------|------|-------------|----------------|")
for icon_name, code, desc, example in tools_data:
    st.markdown(f"| {icon_name} | `{code}` | {desc} | *\"{example}\"* |")

st.divider()

# ═══════════════════════════════════════════════
#  EQUIPMENT TYPES
# ═══════════════════════════════════════════════
st.header("Equipment Types", anchor="equipment-types")
st.markdown("**36 equipment types** are available for process building and scenario patching.")

equip_col1, equip_col2 = st.columns(2)

with equip_col1:
    st.subheader("Heat Transfer")
    st.markdown("""
| Type | Description | Key Parameters |
|------|-------------|----------------|
| `cooler` | Process cooler | outlet_temperature_C, pressure_drop_bar |
| `heater` | Process heater | outlet_temperature_C |
| `air_cooler` | Air-cooled (fin-fan) | outlet_temperature_C |
| `water_cooler` | Water-cooled | outlet_temperature_C |
| `heat_exchanger` | Shell & tube (2 streams) | UA_value |
    """)

    st.subheader("Rotating Equipment")
    st.markdown("""
| Type | Description | Key Parameters |
|------|-------------|----------------|
| `compressor` | Centrifugal/reciprocating | outlet_pressure_bara, isentropic_efficiency |
| `expander` | Gas expander | outlet_pressure_bara, isentropic_efficiency |
| `pump` | Liquid pump | outlet_pressure_bara, efficiency |
| `esp_pump` | Electric submersible pump | outlet_pressure_bara |
| `gas_turbine` | Gas turbine driver | efficiency |
    """)

    st.subheader("Separation")
    st.markdown("""
| Type | Description | Key Parameters |
|------|-------------|----------------|
| `separator` | 2-phase gas/liquid | *(auto-sized)* |
| `two_phase_separator` | Explicit 2-phase separator | *(auto-sized)* |
| `three_phase_separator` | 3-phase G/O/W | *(auto-sized)* |
| `gas_scrubber` | Knock-out drum | *(auto-sized)* |
| `membrane_separator` | Gas membrane separator | *(auto-sized)* |
    """)

    st.subheader("Valves")
    st.markdown("""
| Type | Description | Key Parameters |
|------|-------------|----------------|
| `valve` | Throttling valve (JT) | outlet_pressure_bara |
| `control_valve` | Automated control valve | outlet_pressure_bara, Cv |
    """)

with equip_col2:
    st.subheader("Columns & Absorption")
    st.markdown("""
| Type | Description | Key Parameters |
|------|-------------|----------------|
| `simple_absorber` | Absorption column | number_of_stages |
| `simple_teg_absorber` | TEG dehydration | number_of_stages |
| `distillation_column` | Distillation column | number_of_stages |
| `adsorber` | Molecular sieve / activated carbon | — |
    """)

    st.subheader("Piping & Flow")
    st.markdown("""
| Type | Description | Key Parameters |
|------|-------------|----------------|
| `pipeline` | Pipeline with friction | length_m, diameter_m, roughness |
| `adiabatic_pipe` | Adiabatic pipe | length_m, diameter_m |
| `well_flow` | Well tubing (Beggs & Brills) | wellhead_pressure_bara |
    """)

    st.subheader("Mixing & Splitting")
    st.markdown("""
| Type | Description | Key Parameters |
|------|-------------|----------------|
| `mixer` | Combine streams | — |
| `splitter` | Split by fraction | split_fractions |
| `component_splitter` | Split by component | recovery_fractions |
    """)

    st.subheader("Special Equipment")
    st.markdown("""
| Type | Description | Key Parameters |
|------|-------------|----------------|
| `gibbs_reactor` | Gibbs equilibrium reactor | — |
| `ejector` | Gas ejector | — |
| `flare` | Flare stack | — |
| `filter` | Inline filter | — |
| `tank` | Storage tank | pressure_bara |
| `electrolyzer` | Water electrolyzer (H₂) | — |
    """)

    st.subheader("Control & Convergence")
    st.markdown("""
| Type | Description | Key Parameters |
|------|-------------|----------------|
| `recycle` | Recycle loop convergence | tolerance |
| `adjuster` | Spec-adjust controller | target_variable, target_value |
    """)

st.divider()

# ═══════════════════════════════════════════════
#  EOS MODELS
# ═══════════════════════════════════════════════
st.header("Equation of State Models", anchor="equation-of-state-models")
st.markdown("""
The thermodynamic model determines how fluid properties are calculated.
Choose based on your fluid composition and accuracy requirements.

| Model | Code | Best For | Notes |
|-------|------|----------|-------|
| **SRK** | `srk` | Standard hydrocarbon processing | Default for most natural gas |
| **Peng-Robinson** | `pr` | Alternative cubic EOS | Slightly better liquid densities |
| **PR-1978** | `pr78` | Modified Peng-Robinson | Updated α-function |
| **CPA-SRK** | `cpa` or `cpa-srk` | **Water, MEG, TEG, methanol** | **Required** for polar components |
| **CPA-PR** | `cpa-pr` | Polar components (PR-based) | Alternative CPA formulation |
| **UMR-PRU** | `umr-pru` | Accurate phase envelopes | Best for cricondenbar/therm |
| **GERG-2008** | `gerg2008` | **Fiscal metering, custody transfer** | Reference EOS, 90–450 K / ≤ 350 bar |
| **PC-SAFT** | `pcsaft` | Advanced molecular modeling | Polymers, associating fluids |
| **Ideal Gas** | `ideal` | Testing only | No interactions |

**Selection guidance:**
- Dry natural gas → `srk` or `pr`
- Gas with water/glycol/methanol → `cpa` (mandatory!)
- Phase envelope calculations → `umr-pru`
- Custody transfer / fiscal metering → `gerg2008`
- Hydrogen systems → `srk` with NeqSim H₂ parameters
""")

st.divider()

# ═══════════════════════════════════════════════
#  SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════
st.header("Sensitivity Analysis", anchor="sensitivity-analysis")
st.markdown("""
Sweep one or two variables across a range and observe the effect on key performance indicators (KPIs).

**Three analysis types:**

| Type | Description | Example |
|------|-------------|---------|
| **Single Sweep** | Vary one parameter, plot KPI response | *"Sweep feed pressure from 30 to 70 bara"* |
| **Tornado Chart** | Multiple variables at low/high bounds | *"Which variable has the most impact on compressor power?"* |
| **Two-Variable Surface** | 2D heatmap of two swept parameters | *"Sweep pressure 30–70 and temperature 20–50, show power"* |

**Sweepable parameters:**
- `streams.<name>.pressure_bara` — stream pressure
- `streams.<name>.temperature_C` — stream temperature
- `streams.<name>.flow_kg_hr` — mass flow rate
- `units.<name>.outletpressure_bara` — equipment discharge pressure
- `units.<name>.outtemperature_C` — equipment outlet temperature
- `units.<name>.isentropicefficiency` — compressor/expander efficiency
- `units.<name>.speed` — compressor speed (RPM)

**KPI responses (automatically matched):**
- Power (kW), duty (kW), temperature (°C), pressure (bara)
- Flow rates, compression ratio, efficiency
- Any named KPI from the process model
""")

st.divider()

# ═══════════════════════════════════════════════
#  PVT SIMULATION
# ═══════════════════════════════════════════════
st.header("PVT Simulation", anchor="pvt-simulation")
st.markdown("""
Run standard PVT laboratory experiments on any stream in the process.

| Experiment | Description | Example Prompt |
|------------|-------------|----------------|
| **CME** (Constant Mass Expansion) | Expand fluid at constant temperature, measure relative volume | *"Run a CME test at 100°C on the feed"* |
| **CVD** (Constant Volume Depletion) | Deplete gas at constant volume, measure liquid dropout | *"Run CVD at reservoir temperature"* |
| **Differential Liberation** | Stage-wise gas release at constant temperature | *"Run differential liberation at 80°C"* |
| **Separator Test** | Multi-stage separator flash | *"Run a separator test at 50 bara and 30°C"* |

**Parameters:**
- `temperature_C` — experiment temperature
- `p_start_bara`, `p_end_bara` — pressure range
- `n_steps` — number of pressure steps
- `stream_name` — which stream to use (defaults to feed)
""")

st.divider()

# ═══════════════════════════════════════════════
#  DYNAMIC SIMULATION
# ═══════════════════════════════════════════════
st.header("Dynamic Simulation", anchor="dynamic-simulation")
st.markdown("""
Run transient (time-dependent) simulations to study process response to upsets and changes.

| Scenario | Description | Example Prompt |
|----------|-------------|----------------|
| **Transient** | General time-varying changes | *"What happens if feed flow drops 50% over 60 seconds?"* |
| **Blowdown** | Pressure vessel depressurization | *"Run a blowdown simulation on the separator"* |
| **Startup** | Cold-start sequence | *"Simulate a process startup sequence"* |
| **Shutdown** | Controlled shutdown | *"Simulate an emergency shutdown"* |

**Parameters:**
- `duration_s` — simulation duration in seconds
- `n_steps` — number of time steps
- `changes[]` — list of time-varying parameter changes

**Output:**
- Time profiles of pressure, temperature, flow, composition
- Equipment response curves
- Safety limit violations
""")

st.divider()

# ═══════════════════════════════════════════════
#  SAFETY & RISK
# ═══════════════════════════════════════════════
st.header("Safety & Risk Analysis", anchor="safety-risk-analysis")

safety_col1, safety_col2 = st.columns(2)

with safety_col1:
    st.subheader("⚠️ Risk Analysis")
    st.markdown("""
    Comprehensive equipment risk assessment:

    - **Criticality ranking** — OREDA failure rate database
    - **5×5 risk matrix** — consequence × likelihood
    - **Monte Carlo availability** — plant uptime simulation
    - **Degraded state analysis** — performance with equipment down

    **Example prompts:**
    - *"Show the risk matrix for this process"*
    - *"Run Monte Carlo with 10000 iterations over 365 days"*
    - *"What is the plant availability?"*
    """)

with safety_col2:
    st.subheader("🛡️ Safety Analysis (PSV Sizing)")
    st.markdown("""
    Pressure safety valve sizing per **API 520/521**:

    - **Relief scenarios** — blocked outlet, fire case, control valve failure
    - **PSV orifice sizing** — required area, selected orifice letter
    - **Blowdown analysis** — depressurization curves

    **Example prompts:**
    - *"Size PSVs for all vessels in the process"*
    - *"Include fire case relief scenarios"*
    - *"What is the required relief rate for the separator?"*
    """)

st.divider()

# ═══════════════════════════════════════════════
#  EMISSIONS & ENERGY
# ═══════════════════════════════════════════════
st.header("Emissions & Energy", anchor="emissions-energy")

em_col1, em_col2, em_col3 = st.columns(3)

with em_col1:
    st.subheader("🌍 Emissions")
    st.markdown("""
    - CO₂ combustion emissions
    - CH₄ fugitive emissions
    - Flare stack emissions
    - Scope 1/2/3 breakdown
    - Emission intensity (kg CO₂/unit product)

    *"Calculate CO₂ emissions"*
    """)

with em_col2:
    st.subheader("⚡ Energy Audit")
    st.markdown("""
    - Utility balance sheet
    - Specific energy consumption
    - Equipment-level power breakdown
    - Benchmark vs industry average
    - Fuel gas cost analysis

    *"Run an energy audit"*
    """)

with em_col3:
    st.subheader("♻️ Energy Integration")
    st.markdown("""
    - Composite curves (hot/cold)
    - Pinch point identification
    - Heat recovery potential (kW)
    - ΔTmin optimization
    - Heat exchanger network

    *"Run pinch analysis with ΔTmin = 10°C"*
    """)

st.divider()

# ═══════════════════════════════════════════════
#  FLOW ASSURANCE
# ═══════════════════════════════════════════════
st.header("Flow Assurance", anchor="flow-assurance")
st.markdown("""
Assess hydrate, wax, and corrosion risks across the process.

| Check | What It Does | Mitigation |
|-------|-------------|------------|
| **Hydrate formation** | Calculates hydrate equilibrium T at each point | MEG, methanol, ethanol dosing rates |
| **Wax deposition** | Wax appearance temperature (WAT) screening | Pour point depressants |
| **CO₂ corrosion** | de Waard-Milliams / Norsok M-506 correlation | Corrosion rate (mm/yr), inhibitor |
| **H₂S corrosion** | Sour service assessment | Material selection guidance |

**Example prompts:**
- *"Check for hydrate risk in the export pipeline"*
- *"What is the MEG injection rate needed?"*
- *"Assess corrosion risk for the wet gas pipeline"*
- *"Run full flow assurance check"*
""")

st.divider()

# ═══════════════════════════════════════════════
#  COMPRESSOR CHARTS
# ═══════════════════════════════════════════════
st.header("Compressor Performance Charts", anchor="compressor-performance-charts")
st.markdown("""
Generate detailed compressor performance maps with surge and stonewall limits.
Each chart shows head-vs-flow curves at multiple speeds, with surge line, stonewall,
and the current operating point.

**13 compressor templates available:**

| Template | Description |
|----------|-------------|
| `CENTRIFUGAL_STANDARD` | General purpose centrifugal (~78% eff) — **default** |
| `CENTRIFUGAL_HIGH_FLOW` | High throughput, low pressure ratio (~78% eff) |
| `CENTRIFUGAL_HIGH_HEAD` | High pressure ratio, narrow range (~78% eff) |
| `PIPELINE` | Gas transmission, flat curves, wide turndown (82–85% eff) |
| `EXPORT` | Offshore gas export, high pressure (~80% eff) |
| `INJECTION` | Gas injection/EOR, very high pressure ratio (~77% eff) |
| `GAS_LIFT` | Artificial lift, wide surge margin (~75% eff) |
| `REFRIGERATION` | LNG/process cooling, wide range (~78% eff) |
| `BOOSTER` | Process plant, moderate pressure ratio (~76% eff) |
| `SINGLE_STAGE` | Simple, wide flow range (~75% eff) |
| `MULTISTAGE_INLINE` | Barrel type, 4–8 stages (~78% eff) |
| `INTEGRALLY_GEARED` | Multiple pinions, highest efficiency (82% eff) |
| `OVERHUNG` | Cantilever, simple maintenance (~74% eff) |

**Example prompts:**
- *"Generate compressor chart for the 1st stage compressor"*
- *"Show compressor performance map with surge line"*
- *"Generate chart using PIPELINE template"*
- *"Show the operating envelope for the export compressor"*
""")

st.divider()

# ═══════════════════════════════════════════════
#  DEXPI P&ID
# ═══════════════════════════════════════════════
st.header("DEXPI P&ID Integration", anchor="dexpi-p-id-integration")
st.markdown("""
Import and analyze DEXPI Proteus XML P&ID files — the international standard for engineering diagram exchange.

**Capabilities:**
1. **Parse P&ID** — extract equipment, piping, instruments, connectivity
2. **View in app** — equipment table, piping table, connectivity graph, instrumentation
3. **Import to NeqSim** — create a simulation model from the P&ID topology
4. **Connectivity analysis** — trace flow paths through the process
5. **Design data extraction** — use P&ID design conditions in the simulation

**How to use:**
1. Upload a `.xml` DEXPI file via the sidebar, or click **Load Test DEXPI**
2. The DEXPI P&ID Viewer expander shows the parsed data
3. Ask the chat to analyze: *"Analyze the uploaded DEXPI P&ID"*
4. To import into NeqSim: *"Import the DEXPI P&ID into a NeqSim model with SRK EOS"*

**Supported DEXPI equipment classes:**
- Pumps (centrifugal, reciprocating)
- Heat exchangers (plate, tubular, air-cooled)
- Vessels (tanks, pressure vessels, columns)
- Compressors (centrifugal, reciprocating)
- Reactors, filters, dryers, mixers, separators

**Extracted data:**
- Equipment tag names, types, nozzles, design attributes
- Piping line numbers, fluid codes, nominal diameters
- Instrumentation tags, measured variables, signal types
- Connectivity (which equipment connects to which via which pipe)
""")

st.divider()

# ═══════════════════════════════════════════════
#  CUSTOM PYTHON CODE
# ═══════════════════════════════════════════════
st.header("Custom Python Code (neqsim_code)", anchor="custom-python-code-neqsim-code")
st.markdown("""
For calculations not covered by the built-in tools, the AI can write and execute
custom Python code using the full NeqSim API.

**Allowed imports:**
```python
from neqsim.thermo import fluid, TPflash, dataFrame, phaseenvelope, fluid_df
from neqsim import jneqsim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import math, json
```

**Blocked (for security):** `os`, `subprocess`, `sys`, `open()`, network calls

**Auto-retry:** If the code fails, the AI automatically analyzes the error and
rewrites the code — up to 5 attempts until it succeeds.

**Output capture:**
- `print()` statements → displayed as text
- Pandas DataFrames → rendered as interactive tables
- Matplotlib figures → displayed as images
- Plotly figures → displayed as interactive charts

**Example prompts:**
- *"Calculate the phase envelope for 90% methane, 5% ethane, 3% propane, 2% CO2"*
- *"Compare viscosity predictions between SRK and PR for the feed gas"*
- *"Plot bubble point pressure vs temperature for this composition"*
- *"Calculate the Joule-Thomson coefficient at 50 bara and 30°C"*
- *"What is the water dew point for the export gas?"*
""")

st.divider()

# ═══════════════════════════════════════════════
#  EXTERNAL DATA
# ═══════════════════════════════════════════════
st.header("External Data Integration", anchor="external-data-integration")

ext_col1, ext_col2, ext_col3 = st.columns(3)

with ext_col1:
    st.subheader("🌤️ Weather API")
    st.markdown("""
    Fetch live weather data and analyze
    impact on process performance.

    - Current temperature, humidity, wind
    - 7-day forecast
    - Cooler/air-cooler capacity impact
    - Seasonal de-rating

    *"Weather in Stavanger? Impact on coolers?"*
    *"What is the design ambient for Hammerfest?"*
    """)

with ext_col2:
    st.subheader("🔬 Lab/LIMS Import")
    st.markdown("""
    Update feed composition from lab data.

    **Formats supported:**
    - **Inline** — directly in chat message
    - **CSV** — upload via sidebar
    - **JSON** — upload via sidebar

    *"Update feed: 85% methane, 7% ethane,
    3% propane, 2% CO2, 3% N2"*
    """)

with ext_col3:
    st.subheader("🛢️ Production Scenarios")
    st.markdown("""
    Simulate changing well conditions:

    - **GOR sweep** — varying gas/oil ratio
    - **Water cut sweep** — increasing water
    - **Well blending** — mix multiple wells
    - **Composition sweep** — vary a component

    *"Sweep GOR from 500 to 2000"*
    *"What if water cut increases to 40%?"*
    """)

st.divider()

# ═══════════════════════════════════════════════
#  ADDITIONAL TOOLS
# ═══════════════════════════════════════════════
st.header("Additional Operations Tools", anchor="additional-tools")

add_col1, add_col2 = st.columns(2)

with add_col1:
    st.subheader("📈 Optimization")
    st.markdown("""
    Find the maximum throughput before equipment limits are hit.

    Uses golden-section search to increase feed flow until
    any equipment reaches its utilization limit.

    **Output:** optimal flow rate, limiting equipment, utilization breakdown

    *"Find maximum production"*
    *"What is the bottleneck?"*
    """)

    st.subheader("📉 Turndown Analysis")
    st.markdown("""
    Map the stable operating envelope from minimum to maximum flow.

    - Minimum stable flow (surge, liquid loading)
    - Maximum capacity (choke, flooding)
    - Turndown ratio

    *"What is the turndown range?"*
    *"Can we operate at 40% of design?"*
    """)

    st.subheader("📌 Signal Tracker")
    st.markdown("""
    Track KPIs across multiple chat interactions.

    - Add signals: *"Track compressor power and export temperature"*
    - Take snapshots: *"Snapshot current signals as 'base case'"*
    - Show history: *"Show tracked signal trends"*
    - Compare: see how KPIs evolved across scenarios
    """)

with add_col2:
    st.subheader("🔓 Debottleneck Study")
    st.markdown("""
    Identify process bottlenecks and evaluate upgrade options.

    - Equipment utilization ranking
    - Cost-effectiveness of upgrades
    - Capacity increase potential

    *"Where is the bottleneck?"*
    *"Evaluate debottleneck options"*
    """)

    st.subheader("🎓 Training Scenarios")
    st.markdown("""
    Generate operator training upset scenarios with quiz Q&A.

    **Predefined scenarios:**
    - Cooling water failure
    - Compressor trip
    - Feed composition change
    - Instrument failure
    - Power dip

    *"Generate a cooling water failure training scenario"*
    """)

    st.subheader("📋 Report Generation")
    st.markdown("""
    Export structured JSON reports for documentation.

    **Scopes:**
    - `process` — full process report
    - `unit` — single unit operation detail
    - `stream` — stream conditions & composition
    - `module` — subsystem report

    *"Generate a process report"*
    """)

st.divider()

# ═══════════════════════════════════════════════
#  SAVING & LOADING
# ═══════════════════════════════════════════════
st.header("Saving & Loading Models", anchor="saving-loading-models")
st.markdown("""
**Save your process:**
1. In chat: *"Save the process"* → download button appears in sidebar
2. From Python: `neqsim.save_neqsim(process, "my_process.neqsim")`

**Load a process:**
1. Upload a `.neqsim` file via the sidebar file uploader
2. Or click **📂 Load Test Process** for the sample model

**The `.neqsim` file contains:**
- Full process topology (equipment + connections)
- Fluid compositions and thermodynamic model
- Equipment settings (pressures, temperatures, efficiencies)
- Everything needed to reproduce the simulation

**Python script export:**
- Ask *"Show the Python script"* to get standalone NeqSim code
- The script can be run outside the web app
- Useful for integration with other tools or batch runs
""")

st.divider()

# ═══════════════════════════════════════════════
#  EXAMPLE PROMPTS BY CATEGORY
# ═══════════════════════════════════════════════
st.header("Example Prompts by Category", anchor="example-prompts-by-category")

with st.expander("🔧 Building & Modifying Processes", expanded=True):
    st.markdown("""
    ```
    Build a gas compression process with 80% methane, 10% ethane, 5% propane,
    3% CO2, 2% N2 at 30°C and 50 bara. Compress to 150 bara in two stages.
    ```
    ```
    Create a gas dehydration process with TEG absorber at 70 bara
    ```
    ```
    Build a 3-stage compression train with intercooling for natural gas from 30 to 200 bara
    ```
    ```
    Add a cooler after the compressor to bring the temperature down to 35°C
    ```
    ```
    Add an expander to recover energy from the high-pressure gas
    ```
    """)

with st.expander("🔀 What-If Analysis"):
    st.markdown("""
    ```
    What if we increase the export pressure by 10 bara?
    ```
    ```
    What happens if feed temperature increases to 45°C?
    ```
    ```
    What if we increase feed flow by 20%?
    ```
    ```
    Compare performance at 40, 50, and 60 bara feed pressure
    ```
    ```
    What feed rate gives 50 MW total compressor power?
    ```
    """)

with st.expander("📈 Optimization & Operations"):
    st.markdown("""
    ```
    Find maximum production for this process
    ```
    ```
    What is the turndown range?
    ```
    ```
    Where is the bottleneck?
    ```
    ```
    Run an energy audit
    ```
    ```
    Compare summer vs winter performance
    ```
    """)

with st.expander("⚠️ Safety & Risk"):
    st.markdown("""
    ```
    Show the risk matrix for this process
    ```
    ```
    Run Monte Carlo availability with 10000 iterations
    ```
    ```
    Size PSVs for all vessels including fire case
    ```
    ```
    Run a blowdown simulation on the separator
    ```
    """)

with st.expander("🌍 Environmental"):
    st.markdown("""
    ```
    Calculate CO₂ emissions for this process
    ```
    ```
    Run pinch analysis with ΔTmin = 10°C
    ```
    ```
    Analyze flare gas and recovery options
    ```
    ```
    What is the emission intensity?
    ```
    """)

with st.expander("🌊 Flow Assurance & PVT"):
    st.markdown("""
    ```
    Check for hydrate risk in the export pipeline
    ```
    ```
    What MEG injection rate is needed?
    ```
    ```
    Run a CME test at 100°C
    ```
    ```
    Run differential liberation at 80°C
    ```
    """)

with st.expander("🐍 Custom Calculations"):
    st.markdown("""
    ```
    Calculate the phase envelope for 90% methane, 5% ethane, 3% propane, 2% CO2
    ```
    ```
    Compare viscosity between SRK and PR for the feed gas
    ```
    ```
    Plot bubble point pressure vs temperature
    ```
    ```
    What is the Joule-Thomson coefficient at 50 bara and 30°C?
    ```
    ```
    Calculate the water content of the gas at export conditions
    ```
    """)

with st.expander("📊 Charts & Monitoring"):
    st.markdown("""
    ```
    Show compressor performance map with surge line
    ```
    ```
    Sweep feed pressure from 30 to 70 bara, show compressor power
    ```
    ```
    Track compressor power and export temperature
    ```
    ```
    Size all equipment and show utilization
    ```
    ```
    Generate a process report
    ```
    """)

st.divider()

# ═══════════════════════════════════════════════
#  QUICK REFERENCE CARD
# ═══════════════════════════════════════════════
st.header("Quick Reference Card")
st.markdown("""
| Want to... | Say... |
|------------|--------|
| Build a process | *"Build a gas compression process with..."* |
| Change something | *"What if pressure increases to 100 bara?"* |
| Find limits | *"Find maximum production"* |
| Check safety | *"Size the PSVs"* or *"Show risk matrix"* |
| Environmental | *"Calculate CO₂ emissions"* |
| Sweep parameter | *"Sweep feed pressure from 30 to 70 bara"* |
| PVT test | *"Run a CME test at 100°C"* |
| Dynamic event | *"Run a blowdown simulation"* |
| Flow assurance | *"Check for hydrate risk"* |
| Heat recovery | *"Run pinch analysis"* |
| Custom calc | *"Calculate the phase envelope for..."* |
| Get code | *"Show the Python script"* |
| Save | *"Save the process"* |
| Weather | *"Weather in Stavanger? Impact on coolers?"* |
| Update feed | *"Update feed: 85% methane, 7% ethane..."* |

**Tips:**
- You don't need to know tool names — describe what you want in plain English
- The AI automatically selects the right tool
- Results include interactive charts, tables, and diagrams
- Ask follow-up questions to refine results
- All changes are reversible — original model is preserved
""")
