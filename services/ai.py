"""High level AI helpers reused by multiple Streamlit pages."""
from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from fluids import default_fluid

from .ai_core import call_llm

_COMPONENT_PATTERN = re.compile(
    r"(?P<amount>[-+]?\d+(?:\.\d+)?)\s*(?:%|percent|mole\s*%|mol\s*%)\s*(?:of\s+)?(?P<name>[A-Za-z0-9\-\s]+)",
    flags=re.IGNORECASE,
)
_PRESSURE_PATTERN = re.compile(r"(\-?\d+(?:\.\d+)?)\s*(?:bar|bara|barg)", flags=re.IGNORECASE)
_TEMPERATURE_PATTERN = re.compile(r"(\-?\d+(?:\.\d+)?)\s*(?:c|°c|degc|deg\s*c|celsius)", flags=re.IGNORECASE)


def _empty_fluid_frame() -> pd.DataFrame:
    return pd.DataFrame(default_fluid)


def _match_component(name: str, candidates: Iterable[str]) -> str:
    target = name.strip().lower()
    for candidate in candidates:
        if candidate.lower() == target:
            return candidate
    # fallback: partial match
    for candidate in candidates:
        if target in candidate.lower() or candidate.lower() in target:
            return candidate
    return name.strip()


def plan_scenario(prompt: str) -> Dict[str, pd.DataFrame]:
    """Translate a natural-language description into fluid and TP schedules."""

    base_df = _empty_fluid_frame()
    fluid_df = base_df.copy(deep=True)

    matches = list(_COMPONENT_PATTERN.finditer(prompt))
    if matches:
        fluid_df['MolarComposition[-]'] = 0.0
        component_names = fluid_df['ComponentName'].tolist()
        for match in matches:
            amount = float(match.group('amount'))
            name = _match_component(match.group('name'), component_names)
            fluid_df.loc[fluid_df['ComponentName'] == name, 'MolarComposition[-]'] = amount
        total = fluid_df['MolarComposition[-]'].sum()
        if total > 0:
            fluid_df['MolarComposition[-]'] = fluid_df['MolarComposition[-]'] / total
    # If no explicit matches were found, return the defaults so the UI remains usable

    pressures = [float(match.group(1)) for match in _PRESSURE_PATTERN.finditer(prompt)]
    temperatures = [float(match.group(1)) for match in _TEMPERATURE_PATTERN.finditer(prompt)]
    schedule_rows: List[Tuple[float, float]] = []
    max_len = max(len(pressures), len(temperatures), 1)
    for idx in range(max_len):
        temp = temperatures[idx] if idx < len(temperatures) else temperatures[0] if temperatures else 20.0
        pres = pressures[idx] if idx < len(pressures) else pressures[0] if pressures else 1.0
        schedule_rows.append((temp, pres))
    schedule_df = pd.DataFrame(schedule_rows, columns=['Temperature (C)', 'Pressure (bara)'])
    return {"fluid": fluid_df, "schedule": schedule_df}


def _compose_validation_guidance(issues: List[Dict[str, str]]) -> str:
    if not issues:
        return "Inputs look good—no validation issues detected."

    bullet_lines = [f"• {issue['message']}" for issue in issues]
    suggestions: List[str] = []
    for issue in issues:
        code = issue.get("code")
        if code == "composition_zero":
            suggestions.append(
                "Assign a positive molar composition to at least one component before running the flash."
            )
        elif code == "composition_normalize":
            suggestions.append(
                "Use the normalization helper or scale the composition values so they sum to 1.0."
            )
        elif code == "schedule_empty":
            suggestions.append(
                "Add at least one temperature/pressure pair to the schedule table."
            )
        elif code == "temperature_range":
            suggestions.append(
                "Review extreme temperatures and confirm they are intentional for the model setup."
            )
        elif code == "pressure_positive":
            suggestions.append("Ensure all pressures are positive values in bara.")

    message_parts = ["Validation checks highlighted the following issues:"]
    message_parts.extend(bullet_lines)
    if suggestions:
        message_parts.append("Suggested fixes:")
        message_parts.extend(f"- {tip}" for tip in suggestions)
    return "\n".join(message_parts)


def explain_validation_issue(issues: List[Dict[str, str]], fluid: pd.DataFrame, schedule: pd.DataFrame) -> str:
    """Compose an AI explanation for validation warnings."""

    issue_lines = "\n".join(f"- {issue['message']}" for issue in issues)
    prompt = (
        "You are assisting with thermodynamic simulations. The user provided "
        "the following validation feedback:\n"
        f"{issue_lines}\n\nSummarize the problems and propose concrete fixes."
    )
    fallback_message = _compose_validation_guidance(issues)
    response = call_llm(prompt, offline_fallback=lambda *_: fallback_message)
    return response.text


def summarize_flash_results(results_df: pd.DataFrame, scenario_context: Dict[str, str]) -> str:
    """Create a high-level summary of TP flash outputs."""

    if results_df.empty:
        return "No results available to summarize."
    phase_counts = results_df['phase'].value_counts() if 'phase' in results_df.columns else pd.Series()
    compositions = []
    for component_col in [col for col in results_df.columns if col.lower().startswith('x[') or col.lower().startswith('y[')]:
        compositions.append(f"{component_col}: mean={results_df[component_col].mean():.3f}")
    summary_parts = [
        "### Flash overview",
        f"Simulations evaluated {len(results_df)} state points.",
    ]
    if scenario_context:
        summary_parts.append(
            "Key scenario inputs: " + ", ".join(f"{k}={v}" for k, v in scenario_context.items())
        )
    if not phase_counts.empty:
        phase_text = ", ".join(f"{phase}: {count}" for phase, count in phase_counts.items())
        summary_parts.append(f"Phase occurrences: {phase_text}.")
    if compositions:
        summary_parts.append("Representative compositions:\n" + "\n".join(f"* {line}" for line in compositions[:6]))
    prompt = "\n\n".join(summary_parts)

    def _fallback_summary(_: str, __: Optional[str]) -> str:
        lines = summary_parts.copy()
        if not results_df.empty:
            numeric_cols = [col for col in results_df.columns if pd.api.types.is_numeric_dtype(results_df[col])]
            if numeric_cols:
                lines.append("Key numeric ranges:")
                for column in numeric_cols[:6]:
                    series = results_df[column]
                    lines.append(
                        f"- {column}: min={series.min():.3g}, mean={series.mean():.3g}, max={series.max():.3g}"
                    )
        return "\n".join(lines)

    ai_response = call_llm(
        prompt,
        system_prompt=(
            "Provide an approachable explanation for process engineers based on the "
            "structured summary of TP flash simulation results."
        ),
        offline_fallback=_fallback_summary,
    )
    return ai_response.text


def create_issue_messages(fluid: pd.DataFrame, schedule: pd.DataFrame) -> List[Dict[str, str]]:
    issues: List[Dict[str, str]] = []
    total = fluid['MolarComposition[-]'].sum()
    if total <= 0:
        issues.append({
            "level": "error",
            "code": "composition_zero",
            "message": "Total molar composition must be greater than zero.",
        })
    elif not math.isclose(total, 1.0, rel_tol=0.05):
        issues.append({
            "level": "warning",
            "code": "composition_normalize",
            "message": (
                f"Molar composition sums to {total:.3f}, consider normalizing to 1.0."
            ),
        })
    if schedule.empty:
        issues.append({
            "level": "error",
            "code": "schedule_empty",
            "message": "At least one temperature/pressure pair is required.",
        })
    else:
        for idx, row in schedule.iterrows():
            if row['Temperature (C)'] < -271 or row['Temperature (C)'] > 1000:
                issues.append({
                    "level": "warning",
                    "code": "temperature_range",
                    "message": (
                        f"Row {idx + 1} temperature {row['Temperature (C)']}°C is outside recommended limits."
                    ),
                })
            if row['Pressure (bara)'] <= 0:
                issues.append({
                    "level": "error",
                    "code": "pressure_positive",
                    "message": (
                        f"Row {idx + 1} pressure {row['Pressure (bara)']} bara must be positive."
                    ),
                })
    return issues
