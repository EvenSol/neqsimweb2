"""Shared assistant helpers for Streamlit pages."""
from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from services.ai_core import call_llm, load_page_config


def _offline_assistant_reply(
    question: str,
    context: Dict[str, Any],
    config: Dict[str, Any],
    history: List[Dict[str, str]],
) -> str:
    context_lines = [f"- {key}: {value}" for key, value in context.items()] if context else []
    description = config.get(
        "description",
        "Review the provided context details and adjust your simulation inputs accordingly.",
    )
    history_lines: List[str] = []
    if history:
        history_lines.append("Recent conversation:")
        for turn in history[-4:]:
            speaker = "You" if turn["role"] == "user" else "Assistant"
            history_lines.append(f"{speaker}: {turn['content']}")
    message_parts = [
        "Assistant response (offline mode)",
        description,
    ]
    if context_lines:
        message_parts.append("Context snapshot:")
        message_parts.extend(context_lines)
    if history_lines:
        message_parts.extend(history_lines)
    message_parts.append(f"Echoing your question for reference: {question}")
    message_parts.append("Consult the NeqSim documentation or in-app tooltips for deeper guidance.")
    return "\n".join(message_parts)


def render_ai_helper(page_id: str, context: Dict[str, str]) -> None:
    """Render a consistent assistant helper based on page configuration."""

    config = load_page_config(page_id)
    if not config:
        return

    title = config.get("title", "AI helper")
    description = config.get(
        "description",
        "Ask the assistant for guidance on configuring the simulation.",
    )
    history_key = f"{page_id}_assistant_history"
    if history_key not in st.session_state:
        st.session_state[history_key] = []
    history: List[Dict[str, str]] = st.session_state[history_key]

    with st.expander(title, expanded=False):
        st.write(description)
        if context:
            st.json(context)
        if history:
            st.markdown("### Conversation history")
            for turn in history[-6:]:
                speaker = "You" if turn["role"] == "user" else "Assistant"
                st.markdown(f"**{speaker}:** {turn['content']}")
        question = st.text_area("Ask a question", key=f"{page_id}_assistant")
        if st.button("Get assistant reply", key=f"{page_id}_assistant_button") and question:
            history.append({"role": "user", "content": question})
            conversation_lines = [
                f"{('User' if turn['role'] == 'user' else 'Assistant')}: {turn['content']}"
                for turn in history[-6:]
            ]
            prompt_parts = [
                "Use the JSON context to assist with NeqSim simulations.",
                f"Context: {context}",
            ]
            if conversation_lines:
                prompt_parts.append("Prior conversation:\n" + "\n".join(conversation_lines))
            prompt_parts.append(f"Current question: {question}")
            prompt = "\n\n".join(prompt_parts)
            system_prompt = config.get(
                "system_prompt", "Provide helpful tips based on the context JSON and conversation history."
            )

            response = call_llm(
                prompt,
                system_prompt=system_prompt,
                offline_fallback=lambda *_: _offline_assistant_reply(
                    question, context, config, history
                ),
            )
            history.append({"role": "assistant", "content": response.text})
            st.session_state[history_key] = history
            st.session_state[f"{page_id}_assistant"] = ""
            st.markdown(response.text)
