"""NeqSim Plant Operator — an engineering training game."""

from __future__ import annotations

import json
import os
import sys

import pandas as pd
import streamlit as st


_JVM_OPENS = (
    "--add-opens=java.base/java.util=ALL-UNNAMED",
    "--add-opens=java.base/java.lang=ALL-UNNAMED",
    "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
    "--add-opens=java.base/java.io=ALL-UNNAMED",
)
_existing_java_tool_options = os.environ.get("JAVA_TOOL_OPTIONS", "").strip()
_existing_java_tokens = set(_existing_java_tool_options.split())
_missing_jvm_opens = tuple(
    option for option in _JVM_OPENS if option not in _existing_java_tokens
)
if _missing_jvm_opens:
    os.environ["JAVA_TOOL_OPTIONS"] = " ".join(
        part
        for part in (_existing_java_tool_options, *_missing_jvm_opens)
        if part
    )

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from process_chat.plant_operator_game import (  # noqa: E402
    BASELINE_FLOW_KG_HR,
    CHALLENGE_NAME,
    CHALLENGE_TIMEOUT_MS,
    ChallengeControls,
    ChallengeRun,
    run_challenge,
)
from theme import apply_theme, theme_toggle  # noqa: E402


LAST_RUN_STATE_KEY = "plant_operator_last_run"
ATTEMPT_COUNT_STATE_KEY = "plant_operator_attempt_count"
BEST_SCORE_STATE_KEY = "plant_operator_best_score"
CONTROL_DEFAULTS = {
    "plant_operator_feed_flow": BASELINE_FLOW_KG_HR,
    "plant_operator_stage_1_pressure": 80.0,
    "plant_operator_stage_2_pressure": 125.0,
    "plant_operator_intercooler_temperature": 40.0,
    "plant_operator_export_temperature": 45.0,
}


def _initialize_state() -> None:
    """Create stable widget and progress state for this session."""
    for key, value in CONTROL_DEFAULTS.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault(ATTEMPT_COUNT_STATE_KEY, 0)
    st.session_state.setdefault(BEST_SCORE_STATE_KEY, 0)


def _reset_challenge() -> None:
    """Reset controls and session-local progress to the original plant state."""
    for key, value in CONTROL_DEFAULTS.items():
        st.session_state[key] = value
    st.session_state.pop(LAST_RUN_STATE_KEY, None)
    st.session_state[ATTEMPT_COUNT_STATE_KEY] = 0
    st.session_state[BEST_SCORE_STATE_KEY] = 0


def _controls_from_widgets() -> ChallengeControls:
    """Return the current player decisions as the engine input contract."""
    return ChallengeControls(
        feed_flow_kg_hr=float(st.session_state["plant_operator_feed_flow"]),
        stage_1_pressure_bara=float(
            st.session_state["plant_operator_stage_1_pressure"]
        ),
        stage_2_pressure_bara=float(
            st.session_state["plant_operator_stage_2_pressure"]
        ),
        intercooler_temperature_c=float(
            st.session_state["plant_operator_intercooler_temperature"]
        ),
        export_temperature_c=float(
            st.session_state["plant_operator_export_temperature"]
        ),
    )


def _evidence_dataframe(run: ChallengeRun) -> pd.DataFrame:
    """Format the score evidence with explicit units."""
    evidence = run.evidence
    return pd.DataFrame(
        [
            {
                "Metric": "Feed throughput",
                "Value": evidence.feed_flow_kg_hr,
                "Unit": "kg/hr",
            },
            {
                "Metric": "Export pressure",
                "Value": evidence.export_pressure_bara,
                "Unit": "bara",
            },
            {
                "Metric": "Export temperature",
                "Value": evidence.export_temperature_c,
                "Unit": "°C",
            },
            {
                "Metric": "Stage 1 discharge temperature",
                "Value": evidence.stage_1_discharge_temperature_c,
                "Unit": "°C",
            },
            {
                "Metric": "Stage 2 discharge temperature",
                "Value": evidence.stage_2_discharge_temperature_c,
                "Unit": "°C",
            },
            {
                "Metric": "Compression power",
                "Value": evidence.total_power_kw,
                "Unit": "kW",
            },
            {
                "Metric": "Specific compression energy",
                "Value": evidence.specific_power_kwh_per_tonne,
                "Unit": "kWh/tonne",
            },
            {
                "Metric": "Cooling duty",
                "Value": evidence.total_cooling_duty_kw,
                "Unit": "kW",
            },
            {
                "Metric": "Mass-balance error",
                "Value": evidence.mass_balance_error_pct,
                "Unit": "%",
            },
            {
                "Metric": "Energy-balance error",
                "Value": evidence.energy_balance_error_pct,
                "Unit": "%",
            },
        ]
    )


st.set_page_config(
    page_title="NeqSim Plant Operator",
    page_icon="images/neqsimlogocircleflat.png",
    layout="wide",
)
apply_theme()
theme_toggle()
_initialize_state()

st.markdown(
    """
<style>
    .block-container {
        max-width: 1500px;
        padding-top: 1.1rem;
        padding-bottom: 3rem;
    }
    .operator-hero {
        padding: 1.5rem 1.7rem;
        border: 1px solid rgba(23, 71, 116, 0.30);
        border-radius: 16px;
        background:
            linear-gradient(120deg, rgba(8, 31, 52, 0.98), rgba(18, 72, 102, 0.94));
        color: #f4fbff;
        box-shadow: 0 14px 38px rgba(8, 31, 52, 0.16);
        margin-bottom: 1rem;
    }
    .operator-kicker {
        color: #72d7d0 !important;
        font-size: 0.76rem;
        font-weight: 750;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin: 0 0 0.35rem 0;
    }
    .operator-title {
        color: #ffffff !important;
        font-size: clamp(2rem, 4vw, 3.15rem);
        letter-spacing: -0.035em;
        line-height: 1.02;
        margin: 0;
    }
    .operator-lead {
        color: #d4e8f0 !important;
        max-width: 900px;
        line-height: 1.55;
        margin: 0.7rem 0 0 0;
    }
    .process-train {
        display: grid;
        grid-template-columns: repeat(7, minmax(105px, 1fr));
        gap: 0.45rem;
        align-items: stretch;
        margin: 0.8rem 0 1.1rem 0;
    }
    .process-node {
        padding: 0.72rem 0.5rem;
        border-radius: 10px;
        border: 1px solid rgba(29, 91, 126, 0.24);
        background: rgba(239, 248, 250, 0.92);
        color: #163a52 !important;
        text-align: center;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .process-node span {
        display: block;
        color: #58778a !important;
        font-size: 0.70rem;
        font-weight: 500;
        margin-top: 0.18rem;
    }
    .mission-card {
        padding: 1rem 1.1rem;
        border-left: 4px solid #d5a92f;
        border-radius: 10px;
        background: rgba(255, 249, 229, 0.86);
        margin-bottom: 0.8rem;
    }
    @media (max-width: 900px) {
        .process-train {
            grid-template-columns: repeat(2, minmax(120px, 1fr));
        }
    }
    @media (max-width: 600px) {
        .block-container {
            padding-left: 0.85rem;
            padding-right: 0.85rem;
        }
        .operator-hero {
            padding: 1.2rem;
        }
        .process-train {
            grid-template-columns: 1fr;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Plant Operator")
    st.caption("Native NeqSim engineering challenge")
    if st.button("← NeqSim Studio", use_container_width=True):
        st.switch_page("pages/00_NeqSim_Studio.py")
    if st.button("Reset challenge", use_container_width=True):
        _reset_challenge()
        st.rerun()
    st.divider()
    attempt_metric_placeholder = st.empty()
    best_score_metric_placeholder = st.empty()
    st.caption(
        "Synthetic educational case. Results are engineering-screening "
        "evidence, not design certification."
    )

st.markdown(
    f"""
<section class="operator-hero" aria-labelledby="operator-page-title">
    <p class="operator-kicker">NeqSim Plant Operator · Challenge 1</p>
    <h1 class="operator-title" id="operator-page-title">{CHALLENGE_NAME}</h1>
    <p class="operator-lead">
        Raise production from 100,000 to 110,000 kg/hr. Balance the two-stage
        compressor train, satisfy export conditions, and keep the native NeqSim
        conservation checks trustworthy.
    </p>
</section>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="process-train" role="img" aria-label="Gas compression process train">
    <div class="process-node">Feed<span>50 bara · 30 °C</span></div>
    <div class="process-node">Inlet scrubber<span>liquid removal</span></div>
    <div class="process-node">Compressor 1<span>pressure split</span></div>
    <div class="process-node">Intercooler<span>temperature control</span></div>
    <div class="process-node">Scrubber 2<span>condensate removal</span></div>
    <div class="process-node">Compressor 2<span>export pressure</span></div>
    <div class="process-node">Export cooler<span>product temperature</span></div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<section class="mission-card" aria-label="Mission constraints">
    <strong>Win condition:</strong> achieve at least 110,000 kg/hr and 128 bara,
    keep export gas at or below 45 °C, compressor discharges at or below 120 °C,
    compression below 4.2 MW, specific energy below 41 kWh/tonne, cooling below
    5.5 MW, and pass the native mass, energy, and solver checks.
</section>
""",
    unsafe_allow_html=True,
)

control_panel, result_panel = st.columns([0.92, 1.35], gap="large")
with control_panel:
    st.subheader("Operating decisions")
    st.slider(
        "Feed throughput [kg/hr]",
        min_value=90_000.0,
        max_value=125_000.0,
        step=1_000.0,
        key="plant_operator_feed_flow",
        help="The original plant operates at 100,000 kg/hr.",
    )
    st.slider(
        "Stage 1 discharge pressure [bara]",
        min_value=60.0,
        max_value=110.0,
        step=1.0,
        key="plant_operator_stage_1_pressure",
    )
    st.slider(
        "Stage 2 discharge pressure [bara]",
        min_value=115.0,
        max_value=145.0,
        step=1.0,
        key="plant_operator_stage_2_pressure",
    )
    st.slider(
        "Intercooler outlet temperature [°C]",
        min_value=20.0,
        max_value=60.0,
        step=1.0,
        key="plant_operator_intercooler_temperature",
    )
    st.slider(
        "Export cooler outlet temperature [°C]",
        min_value=20.0,
        max_value=55.0,
        step=1.0,
        key="plant_operator_export_temperature",
    )
    run_attempt = st.button(
        "▶ Run operating strategy",
        type="primary",
        use_container_width=True,
    )
    with st.expander("Operator hint", expanded=False):
        st.write(
            "For ideal compression with good intercooling, an approximately "
            "equal pressure ratio in each stage tends to reduce power. Real "
            "phase behaviour and liquid removal are still calculated by NeqSim."
        )

current_controls = _controls_from_widgets()
if run_attempt:
    try:
        with st.spinner("NeqSim is building, solving, and validating the plant..."):
            completed_run = run_challenge(
                current_controls,
                timeout_ms=CHALLENGE_TIMEOUT_MS,
            )
    except TimeoutError:
        st.session_state.pop(LAST_RUN_STATE_KEY, None)
        st.error(
            "The attempt exceeded the 180-second execution budget. The partial "
            "native model was discarded; change the settings and try again."
        )
    except Exception as error:
        st.session_state.pop(LAST_RUN_STATE_KEY, None)
        st.error(f"The NeqSim attempt failed: {error}")
    else:
        st.session_state[LAST_RUN_STATE_KEY] = completed_run
        st.session_state[ATTEMPT_COUNT_STATE_KEY] += 1
        st.session_state[BEST_SCORE_STATE_KEY] = max(
            int(st.session_state[BEST_SCORE_STATE_KEY]),
            completed_run.assessment.score,
        )

attempt_metric_placeholder.metric(
    "Attempts",
    int(st.session_state[ATTEMPT_COUNT_STATE_KEY]),
)
best_score_metric_placeholder.metric(
    "Best score",
    f"{int(st.session_state[BEST_SCORE_STATE_KEY])}/1000",
)

last_run = st.session_state.get(LAST_RUN_STATE_KEY)
run_is_current = isinstance(last_run, ChallengeRun) and last_run.controls == current_controls

with result_panel:
    st.subheader("Control-room result")
    if not isinstance(last_run, ChallengeRun):
        st.info(
            "The plant is at its original operating point. Adjust the controls "
            "and run your first native NeqSim attempt."
        )
    elif not run_is_current:
        st.warning(
            "Controls changed after the last solve. Run the strategy again before "
            "using its score or engineering evidence."
        )
    else:
        assessment = last_run.assessment
        if assessment.won:
            st.success(f"{assessment.grade}.")
        else:
            st.warning(assessment.grade)

        score_col, power_col, production_col, time_col = st.columns(4)
        score_col.metric("Score", f"{assessment.score}/1000")
        power_col.metric("Compression", f"{last_run.evidence.total_power_kw:,.0f} kW")
        production_col.metric(
            "Throughput",
            f"{last_run.evidence.feed_flow_kg_hr:,.0f} kg/hr",
            delta=(
                f"{(last_run.evidence.feed_flow_kg_hr / BASELINE_FLOW_KG_HR - 1.0) * 100:.1f}%"
            ),
        )
        time_col.metric("Solve time", f"{last_run.elapsed_seconds:.2f} s")
        st.progress(assessment.score / 1000.0)

        check_rows = [
            {
                "Status": "PASS" if check.passed else "FAIL",
                "Constraint": check.name,
                "Solved value": check.actual,
                "Requirement": check.requirement,
            }
            for check in assessment.checks
        ]
        st.dataframe(
            pd.DataFrame(check_rows),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("#### Engineering feedback")
        for guidance in assessment.guidance:
            st.write(f"• {guidance}")

if run_is_current:
    st.divider()
    evidence_tab, validation_tab, replay_tab = st.tabs(
        ["Solved evidence", "Native validation", "Replay & Process Chat"]
    )
    with evidence_tab:
        st.dataframe(
            _evidence_dataframe(last_run).style.format({"Value": "{:,.6g}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "All score inputs are solved NeqSim values. Standard, mass, and energy "
            "units are kept explicit."
        )
    with validation_tab:
        native_rows = [
            {
                "Check": constraint.name,
                "Status": constraint.status,
                "Detail": constraint.detail,
            }
            for constraint in last_run.result.constraints
        ]
        st.dataframe(
            pd.DataFrame(native_rows),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "The game never converts a failed or unavailable native check into a pass."
        )
    with replay_tab:
        st.download_button(
            "Download reproducible challenge JSON",
            data=json.dumps(last_run.spec, indent=2),
            file_name="neqsim_10_percent_throughput_challenge.json",
            mime="application/json",
            use_container_width=True,
        )
        if st.button("Analyze solved attempt in Process Chat", use_container_width=True):
            st.session_state["process_model"] = last_run.model
            st.session_state["process_model_name"] = "plant_operator_challenge.neqsim"
            st.switch_page("pages/90_Process_Chat.py")
        st.caption(
            "Process Chat receives the exact solved NeqSim model from this attempt."
        )

st.divider()
st.caption(
    "NeqSim Plant Operator v1 · synthetic public training fluid · steady-state "
    "SRK model · one shared 180-second build/solve/evidence budget."
)
