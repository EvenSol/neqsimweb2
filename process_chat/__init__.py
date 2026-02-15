"""
NeqSim Process Chat — "Chat with your plant" MVP

Modules:
    patch_schema     — Structured model-edit definitions (the only way the LLM changes the model)
    process_model    — NeqSim process builder + adapter (wraps the oil/gas separation process)
    scenario_engine  — Clone → patch → validate → run → compare
    templates        — Equipment templates for planning ("install a cooler…")
    chat_tools       — Gemini tool-calling layer (LLM as planner, NeqSim as calculator)
    optimizer        — Process optimization (find max throughput, bottleneck detection)
    risk_analysis    — Risk framework (equipment criticality, risk matrix, Monte Carlo availability)
    compressor_chart — Compressor performance chart generation and extraction
    auto_size        — Auto-sizing of equipment and utilization tracking
    emissions        — CO₂/CH₄/fugitive emissions calculation and reporting
    dynamic_sim      — Transient simulation (blowdown, ramp-up/down, startup/shutdown)
    sensitivity      — Parameter sweeps, tornado charts, two-variable surface analysis
    pvt_simulation   — PVT experiments (CME, differential liberation, separator test)
    safety_systems   — PSV sizing per API 520/521, relief scenario analysis
    flow_assurance   — Hydrate, wax, corrosion prediction and inhibitor dosing
"""
