"""Lightweight retrieval helper that simulates knowledge-base lookups."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .ai_core import call_llm


@dataclass
class RetrievalResult:
    title: str
    summary: str
    source: str
    confidence: float


def query_equilibrium_knowledge(components: List[str], temperature: float, pressure: float) -> List[Dict[str, object]]:
    """Return stubbed knowledge results filtered by component keywords."""

    unique_components = sorted({component.strip() for component in components if component.strip()})
    component_list = ", ".join(unique_components)
    prompt = (
        "Provide a short literature-style summary for equilibrium data that "
        "matches the following mixture and operating conditions."
        f"\nComponents: {component_list or 'default mixture'}\nTemperature: {temperature} °C\nPressure: {pressure} bara."
    )

    def _offline_summary(_: str, __: Optional[str]) -> str:
        summary_lines = [
            "Knowledge base lookup (offline mode)",
            f"Mixture components: {component_list or 'default composition'}.",
            f"Operating window ≈ {temperature:.1f} °C and {pressure:.1f} bara.",
            "Consult internal lab reports or literature tables for phase-equilibrium references at comparable conditions.",
            "Use these insights as qualitative guidance until the live retrieval service is configured.",
        ]
        return "\n".join(summary_lines)

    response = call_llm(prompt, offline_fallback=_offline_summary)
    simulated_result = RetrievalResult(
        title="Thermodynamic insights",
        summary=response.text,
        source="internal-knowledge-base",
        confidence=0.4,
    )
    return [simulated_result.__dict__]
