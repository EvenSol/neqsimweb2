"""Phase Equilibrium Lab — a native NeqSim thermodynamics game."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
import sys

import pandas as pd
import streamlit as st


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from process_chat.phase_equilibrium_game import (  # noqa: E402
    CHALLENGE_NAME,
    CHALLENGE_TIMEOUT_MS,
    MAX_PRESSURE_BARA,
    MAX_TEMPERATURE_C,
    MIN_PRESSURE_BARA,
    MIN_TEMPERATURE_C,
    RICH_GAS_COMPOSITION,
    PhaseChallengeRun,
    PhaseControls,
    run_phase_challenge,
)
from theme import apply_theme, theme_toggle  # noqa: E402


LAST_RUN_KEY = "phase_lab_last_run"
ATTEMPT_COUNT_KEY = "phase_lab_attempt_count"
BEST_SCORE_KEY = "phase_lab_best_score"
TEMPERATURE_KEY = "phase_lab_temperature_c"
PRESSURE_KEY = "phase_lab_pressure_bara"


def _initialize_state() -> None:
    st.session_state.setdefault(TEMPERATURE_KEY, 20.0)
    st.session_state.setdefault(PRESSURE_KEY, 50.0)
    st.session_state.setdefault(ATTEMPT_COUNT_KEY, 0)
    st.session_state.setdefault(BEST_SCORE_KEY, 0)


def _reset_lab() -> None:
    st.session_state[TEMPERATURE_KEY] = 20.0
    st.session_state[PRESSURE_KEY] = 50.0
    st.session_state[ATTEMPT_COUNT_KEY] = 0
    st.session_state[BEST_SCORE_KEY] = 0
    st.session_state.pop(LAST_RUN_KEY, None)


def _display(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "Unavailable"
    return f"{value:,.{digits}f}"


def _property_rows(run: PhaseChallengeRun) -> list[dict[str, object]]:
    evidence = run.evidence
    return [
        {"Property": "Gas phase fraction", "Value": evidence.gas_fraction_mol_pct, "Unit": "mol%"},
        {
            "Property": "Liquid phase fraction",
            "Value": evidence.liquid_fraction_mol_pct,
            "Unit": "mol%",
        },
        {"Property": "Gas density", "Value": evidence.gas_density_kg_m3, "Unit": "kg/m³"},
        {"Property": "Liquid density", "Value": evidence.liquid_density_kg_m3, "Unit": "kg/m³"},
        {"Property": "Gas compressibility factor", "Value": evidence.gas_z_factor, "Unit": "–"},
        {"Property": "Gas viscosity", "Value": evidence.gas_viscosity_cp, "Unit": "cP"},
        {"Property": "Liquid viscosity", "Value": evidence.liquid_viscosity_cp, "Unit": "cP"},
        {"Property": "Mixture enthalpy", "Value": evidence.mixture_enthalpy_kj_kg, "Unit": "kJ/kg"},
        {"Property": "Mixture Cp", "Value": evidence.mixture_cp_kj_kgk, "Unit": "kJ/(kg·K)"},
        {
            "Property": "Phase-fraction closure error",
            "Value": evidence.phase_fraction_closure_error,
            "Unit": "–",
        },
    ]


st.set_page_config(
    page_title="NeqSim Phase Equilibrium Lab",
    page_icon="images/neqsimlogocircleflat.png",
    layout="wide",
)
apply_theme()
theme_toggle()
_initialize_state()

st.markdown(
    """
<style>
    .block-container { max-width: 1500px; padding-top: 1.1rem; padding-bottom: 3rem; }
    .phase-hero {
        padding: 1.55rem 1.75rem;
        border-radius: 17px;
        color: #f5fcff;
        background: radial-gradient(circle at 90% 15%, rgba(77, 211, 193, 0.28), transparent 34%), linear-gradient(125deg, #102d4a, #165e75);
        box-shadow: 0 15px 38px rgba(9, 42, 65, 0.17);
        margin-bottom: 1rem;
    }
    .phase-hero h1 { color: #ffffff !important; margin: 0; letter-spacing: -0.035em; }
    .phase-hero p { color: #d9eef4 !important; max-width: 900px; margin: 0.65rem 0 0; line-height: 1.55; }
    .phase-mission {
        padding: 1rem 1.15rem;
        border: 1px solid rgba(23, 111, 120, 0.25);
        border-left: 5px solid #1b8f88;
        border-radius: 11px;
        background: rgba(239, 250, 248, 0.94);
        margin-bottom: 1rem;
    }
    @media (max-width: 600px) {
        .block-container { padding-left: 0.85rem; padding-right: 0.85rem; }
        .phase-hero { padding: 1.2rem; }
    }
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Phase Equilibrium Lab")
    st.caption("Native SRK TP-flash challenge")
    if st.button("← NeqSim Games", use_container_width=True):
        st.switch_page("pages/34_NeqSim_Games.py")
    if st.button("NeqSim Studio", use_container_width=True):
        st.switch_page("pages/00_NeqSim_Studio.py")
    if st.button("Reset lab", use_container_width=True):
        _reset_lab()
        st.rerun()
    st.divider()
    attempt_metric_placeholder = st.empty()
    best_score_metric_placeholder = st.empty()
    st.caption(
        "Synthetic educational fluid. Results are equilibrium-training "
        "evidence, not a certified fluid characterization."
    )

st.markdown(
    f"""
<section class="phase-hero" aria-labelledby="phase-lab-title">
  <h1 id="phase-lab-title">{CHALLENGE_NAME}</h1>
  <p>
    You have a fixed rich-gas sample. Move temperature and pressure until a
    native NeqSim TP flash lands inside a narrow gas-condensate property window.
  </p>
</section>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<section class="phase-mission" aria-label="Phase equilibrium mission">
  <strong>Win condition:</strong> form exactly gas + hydrocarbon liquid, recover
  16–20 mol% condensate, hold gas density at 78–92 kg/m³ and Z at 0.80–0.83,
  hold liquid density at 480–510 kg/m³, keep liquid viscosity below 0.12 cP,
  and close the native phase fractions.
</section>
""",
    unsafe_allow_html=True,
)

control_panel, result_panel = st.columns([0.86, 1.35], gap="large")
with control_panel:
    st.subheader("Flash conditions")
    st.slider(
        "Temperature [°C]",
        min_value=float(MIN_TEMPERATURE_C),
        max_value=float(MAX_TEMPERATURE_C),
        step=1.0,
        key=TEMPERATURE_KEY,
        help=(
            "Cooling generally promotes liquid dropout, but gas-condensate "
            "phase behavior is non-linear."
        ),
    )
    st.slider(
        "Pressure [bara]",
        min_value=float(MIN_PRESSURE_BARA),
        max_value=float(MAX_PRESSURE_BARA),
        step=1.0,
        key=PRESSURE_KEY,
        help="Pressure changes phase split, density, and real-gas compressibility together.",
    )
    current_controls = PhaseControls(
        temperature_c=float(st.session_state[TEMPERATURE_KEY]),
        pressure_bara=float(st.session_state[PRESSURE_KEY]),
    )
    run_clicked = st.button(
        "▶ Run native TP flash",
        type="primary",
        use_container_width=True,
    )
    st.caption(
        "Model: SRK equation of state · mixing rule 2 · multiphase stability enabled."
    )
    with st.expander("Fixed rich-gas composition"):
        st.dataframe(
            pd.DataFrame(
                [
                    {"Component": component, "Feed mole %": fraction * 100.0}
                    for component, fraction in RICH_GAS_COMPOSITION.items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

if run_clicked:
    try:
        with st.spinner("NeqSim is solving phase equilibrium and properties…"):
            completed_run = run_phase_challenge(
                current_controls,
                timeout_ms=CHALLENGE_TIMEOUT_MS,
            )
    except TimeoutError:
        st.session_state.pop(LAST_RUN_KEY, None)
        st.error(
            "The TP flash exceeded its execution budget. The partial native "
            "fluid state was discarded."
        )
    except Exception as error:
        st.session_state.pop(LAST_RUN_KEY, None)
        st.error(f"The NeqSim TP flash failed: {error}")
    else:
        st.session_state[LAST_RUN_KEY] = completed_run
        st.session_state[ATTEMPT_COUNT_KEY] += 1
        st.session_state[BEST_SCORE_KEY] = max(
            int(st.session_state[BEST_SCORE_KEY]),
            completed_run.assessment.score,
        )

attempt_metric_placeholder.metric(
    "Attempts",
    int(st.session_state[ATTEMPT_COUNT_KEY]),
)
best_score_metric_placeholder.metric(
    "Best score",
    f"{int(st.session_state[BEST_SCORE_KEY])}/1000",
)

last_run = st.session_state.get(LAST_RUN_KEY)
run_is_current = (
    isinstance(last_run, PhaseChallengeRun)
    and last_run.controls == current_controls
)

with result_panel:
    st.subheader("Equilibrium result")
    if not isinstance(last_run, PhaseChallengeRun):
        st.info(
            "The sample starts outside the target property window. Choose "
            "temperature and pressure, then run the first TP flash."
        )
    elif not run_is_current:
        st.warning(
            "Flash conditions changed after the last solve. Run again before "
            "using the score or property evidence."
        )
    else:
        assessment = last_run.assessment
        if assessment.won:
            st.success(assessment.grade)
        else:
            st.warning(assessment.grade)
        score_column, phase_column, liquid_column, time_column = st.columns(4)
        score_column.metric("Score", f"{assessment.score}/1000")
        phase_column.metric("Phases", " + ".join(last_run.evidence.phase_types))
        liquid_column.metric(
            "Condensate",
            f"{_display(last_run.evidence.liquid_fraction_mol_pct, 2)} mol%",
        )
        time_column.metric("Flash time", f"{last_run.elapsed_seconds:.2f} s")
        st.progress(assessment.score / 1000.0)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Status": "PASS" if check.passed else "FAIL",
                        "Target": check.name,
                        "Solved value": check.actual,
                        "Requirement": check.requirement,
                    }
                    for check in assessment.checks
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("#### Thermodynamic feedback")
        for guidance in assessment.guidance:
            st.write(f"• {guidance}")

if run_is_current:
    st.divider()
    properties_tab, equilibrium_tab, replay_tab = st.tabs(
        ["Fluid properties", "Phase compositions & K-values", "Evidence export"]
    )
    with properties_tab:
        properties = pd.DataFrame(_property_rows(last_run))
        st.dataframe(
            properties.style.format(
                {"Value": lambda value: _display(value, 6)}
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Properties are read from the converged native phases; unavailable "
            "values remain unavailable and cannot pass a target."
        )
    with equilibrium_tab:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Component": row.component,
                        "Feed x": row.feed_mole_fraction,
                        "Gas y": row.gas_mole_fraction,
                        "Liquid x": row.liquid_mole_fraction,
                        "K = y/x": row.k_value,
                    }
                    for row in last_run.evidence.components
                ]
            ).style.format(
                {
                    "Feed x": "{:.6f}",
                    "Gas y": "{:.6f}",
                    "Liquid x": "{:.6f}",
                    "K = y/x": "{:.4f}",
                },
                na_rep="Unavailable",
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.info(
            "K > 1 means the component favors the gas phase; K < 1 means it "
            "favors the hydrocarbon liquid at this equilibrium point."
        )
    with replay_tab:
        st.download_button(
            "Download solved phase evidence JSON",
            data=json.dumps(asdict(last_run), indent=2, allow_nan=False),
            file_name="neqsim_phase_equilibrium_attempt.json",
            mime="application/json",
            use_container_width=True,
        )
        st.caption(
            "The export records controls, native properties, phase compositions, "
            "K-values, checks, score, and solve time."
        )

st.divider()
st.caption(
    "Phase Equilibrium Lab v1 · synthetic rich gas · SRK + mixing rule 2 · "
    "one bounded native TP flash per attempt."
)
