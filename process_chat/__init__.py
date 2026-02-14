"""
NeqSim Process Chat — "Chat with your plant" MVP

Modules:
    patch_schema   — Structured model-edit definitions (the only way the LLM changes the model)
    process_model  — NeqSim process builder + adapter (wraps the oil/gas separation process)
    scenario_engine — Clone → patch → validate → run → compare
    templates      — Equipment templates for planning ("install a cooler…")
    chat_tools     — Gemini tool-calling layer (LLM as planner, NeqSim as calculator)
"""
