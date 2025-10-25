"""Reusable chat widget for conversational what-if analysis."""
from __future__ import annotations

import re
from typing import Callable, Tuple

import pandas as pd
import streamlit as st

from services.ai import summarize_flash_results

_TEMPERATURE_DIRECTIVE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:c|°c|degc)", re.IGNORECASE)
_PRESSURE_DIRECTIVE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:bar|bara)", re.IGNORECASE)


def _apply_adjustments(schedule: pd.DataFrame, message: str) -> pd.DataFrame:
    updated = schedule.copy(deep=True)
    temp_matches = list(_TEMPERATURE_DIRECTIVE.finditer(message))
    pres_matches = list(_PRESSURE_DIRECTIVE.finditer(message))
    if temp_matches:
        target_temp = float(temp_matches[-1].group(1))
        updated['Temperature (C)'] = target_temp
    if pres_matches:
        target_pres = float(pres_matches[-1].group(1))
        updated['Pressure (bara)'] = target_pres
    return updated


def render_what_if_chat(
    fluid_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    is_plus_fluid: bool,
    run_flash: Callable[[pd.DataFrame, pd.DataFrame, bool], Tuple[pd.DataFrame, float, float]],
) -> None:
    """Render a sidebar chat widget that can iterate on TP flash scenarios."""

    chat_key = "what_if_chat_history"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []
    history = st.session_state[chat_key]
    if history and isinstance(history[0], tuple):
        st.session_state[chat_key] = [
            {"role": entry[0], "content": entry[1]} for entry in history if len(entry) >= 2
        ]
        history = st.session_state[chat_key]

    with st.sidebar.expander("What-if assistant", expanded=False):
        st.markdown("Enter adjustments like 'try 40 °C at 15 bar' to explore new flashes.")
        user_message = st.text_input("Ask the assistant", key="what_if_prompt")
        if st.button("Send", key="what_if_send") and user_message:
            history.append({"role": "user", "content": user_message})
            updated_schedule = _apply_adjustments(schedule_df, user_message)
            results_df, last_temp, last_pres = run_flash(fluid_df, updated_schedule, is_plus_fluid)
            prior_turns = [
                f"{('User' if turn['role'] == 'user' else 'Assistant')}: {turn['content']}"
                for turn in history[-6:]
            ]
            scenario_context = {
                "temperature": f"{last_temp:.2f} °C" if not pd.isna(last_temp) else "n/a",
                "pressure": f"{last_pres:.2f} bara" if not pd.isna(last_pres) else "n/a",
                "latest_request": user_message,
            }
            if prior_turns:
                scenario_context["recent_conversation"] = "\n".join(prior_turns)
            summary = summarize_flash_results(
                results_df,
                scenario_context,
            )
            history.append(
                {
                    "role": "assistant",
                    "content": summary,
                    "temperature": last_temp,
                    "pressure": last_pres,
                }
            )
            st.session_state[chat_key] = history
            st.session_state["what_if_prompt"] = ""
            st.experimental_rerun()

        for turn in history[-8:]:
            speaker = "You" if turn["role"] == "user" else "Assistant"
            st.markdown(f"**{speaker}:** {turn['content']}")
