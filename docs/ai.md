# AI-assisted workflows

This project now ships with lightweight AI scaffolding to help configure and interpret
NeqSim simulations. The helpers now include deterministic “offline mode” templates so the
Streamlit UI remains fully functional without API credentials. When a valid
`st.make_request` integration is available the helpers automatically forward prompts to the
configured LLM provider.

## Available assistants

* **TP Flash** – supports natural-language scenario planning, validation guidance, result
  summaries, knowledge lookups, and a conversational what-if exploration widget.
* **Gas Hydrate, LNG Ageing, Property Generator** – each page exposes a contextual
  assistant that reuses the shared helper framework defined in `components/assistant.py`.

Both the shared helper and the what-if widget maintain per-session conversation history so
follow-up prompts automatically include recent context, whether the app is online or using
offline fallbacks.

## Extending the system

1. Add metadata for the new page to `configs/ai_pages.json`.
2. Import and call `render_ai_helper` with any runtime context you want to surface.
3. If the feature requires model interaction, build a helper in `services/ai.py` or
   `services/retrieval.py` and keep API access logic centralized in `services/ai_core.py`.

This structure keeps prompts, retrieval logic, and UI wiring modular so additional
Streamlit pages can opt in with only a few lines of code.
