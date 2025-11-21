"""Core utilities for interacting with AI backends used across the app."""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Union

try:
    import streamlit as st
except ImportError:  # pragma: no cover - Streamlit not available in some environments
    st = None  # type: ignore


CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "configs" / "ai_pages.json"


@dataclass
class AIResponse:
    """Container for AI responses to keep structure consistent."""

    text: str
    raw: Optional[Dict[str, Any]] = None


def _compose_prompt(prompt: str, system_prompt: Optional[str]) -> str:
    if not system_prompt:
        return prompt
    return f"System:\n{system_prompt.strip()}\n\nUser:\n{prompt.strip()}"


OfflineFallback = Union[str, Callable[[str, Optional[str]], str]]


def _resolve_offline_fallback(
    fallback: OfflineFallback,
    prompt: str,
    system_prompt: Optional[str],
) -> str:
    if callable(fallback):
        try:
            return fallback(prompt, system_prompt)
        except Exception:  # pragma: no cover - guardrail for fallback utilities
            pass
    if isinstance(fallback, str):
        return fallback
    return (
        "(Simulated AI response) Set up a real API key via Streamlit secrets to replace this placeholder."
    )


def call_llm(
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
    offline_fallback: Optional[OfflineFallback] = None,
) -> AIResponse:
    """Attempt to call the configured LLM via Streamlit's ``make_request`` hook.

    Falls back to a simulated response when the hook is unavailable so that the
    rest of the UI remains functional in offline development environments.
    """

    request_payload = _compose_prompt(prompt, system_prompt)
    if st is not None and hasattr(st, "make_request"):
        try:
            response = st.make_request(request_payload)
            text = response if isinstance(response, str) else json.dumps(response)
            return AIResponse(text=text, raw={"response": response, "temperature": temperature})
        except Exception as exc:  # pragma: no cover - runtime guard
            if offline_fallback is not None:
                fallback_text = _resolve_offline_fallback(offline_fallback, prompt, system_prompt)
                return AIResponse(
                    text=fallback_text,
                    raw={"error": str(exc), "temperature": temperature, "offline": True},
                )
            fallback_text = (
                "AI request failed with error "
                f"{exc}. Returning a diagnostic message so the UI can continue."
            )
            return AIResponse(text=fallback_text)
    if offline_fallback is not None:
        fallback_text = _resolve_offline_fallback(offline_fallback, prompt, system_prompt)
        return AIResponse(text=fallback_text, raw={"offline": True, "temperature": temperature})
    simulated = (
        "(Simulated AI response) "
        "Set up a real API key via Streamlit secrets to replace this placeholder."
    )
    return AIResponse(text=simulated, raw={"offline": True, "temperature": temperature})


@lru_cache(maxsize=None)
def load_page_config(page_id: str) -> Dict[str, Any]:
    """Load assistant configuration for a given page.

    The configuration file lives in ``configs/ai_pages.json`` and provides
    metadata about inputs/outputs that the assistant can use to tailor prompts.
    """

    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return config.get(page_id, {})
