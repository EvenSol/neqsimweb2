"""Professional entry page for the new NeqSim Studio workspace."""

from __future__ import annotations

import os
import sys

import streamlit as st


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from studio.navigation import (  # noqa: E402
    STATUS_AVAILABLE,
    STATUS_CORE_IN_PROGRESS,
    STUDIO_DESTINATIONS,
)
from theme import apply_theme, theme_toggle  # noqa: E402


st.set_page_config(
    page_title="NeqSim Studio",
    page_icon="images/neqsimlogocircleflat.png",
    layout="wide",
)
apply_theme()
theme_toggle()

st.markdown(
    """
<style>
    .block-container {
        max-width: 1500px;
        padding-top: 1.3rem;
        padding-bottom: 3rem;
    }
    .studio-hero {
        padding: 2.1rem 2.2rem;
        border: 1px solid rgba(49, 91, 160, 0.22);
        border-radius: 18px;
        background:
            radial-gradient(circle at 85% 20%, rgba(49, 91, 160, 0.16), transparent 32%),
            linear-gradient(135deg, rgba(247, 250, 255, 0.98), rgba(236, 244, 255, 0.88));
        margin-bottom: 1.2rem;
    }
    .studio-kicker {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: #315ba0;
        margin-bottom: 0.35rem;
    }
    .studio-title {
        font-size: clamp(2.0rem, 4vw, 3.25rem);
        line-height: 1.03;
        font-weight: 760;
        letter-spacing: -0.035em;
        color: #13213c;
        margin: 0;
    }
    .studio-lead {
        max-width: 850px;
        margin-top: 0.85rem;
        color: #526078;
        font-size: 1.04rem;
        line-height: 1.62;
    }
    .studio-status {
        display: inline-block;
        padding: 0.26rem 0.6rem;
        border-radius: 999px;
        background: rgba(49, 91, 160, 0.10);
        color: #315ba0;
        font-size: 0.76rem;
        font-weight: 700;
        margin-right: 0.35rem;
        margin-top: 0.45rem;
    }
    .workflow-card {
        min-height: 185px;
        padding: 1.15rem 1.2rem;
        margin: 0.35rem 0 0.45rem 0;
        border: 1px solid rgba(91, 110, 140, 0.20);
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.78);
        box-shadow: 0 8px 24px rgba(25, 45, 75, 0.055);
    }
    .workflow-icon {
        font-size: 1.45rem;
        margin-bottom: 0.55rem;
    }
    .workflow-title {
        color: #172641;
        font-size: 1.03rem;
        font-weight: 720;
        margin-bottom: 0.42rem;
    }
    .workflow-description {
        color: #66738a;
        line-height: 1.45;
        font-size: 0.89rem;
    }
    .workflow-state {
        margin-top: 0.75rem;
        color: #315ba0;
        font-size: 0.76rem;
        font-weight: 700;
    }
    .studio-section {
        margin-top: 1.8rem;
        margin-bottom: 0.35rem;
    }
    @media (max-width: 720px) {
        .studio-hero {
            padding: 1.35rem 1.25rem;
        }
        .workflow-card {
            min-height: auto;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Workspace")
    st.success("NeqSim Studio · Beta")
    if st.button("← NeqSim Classic", use_container_width=True):
        st.switch_page("welcome.py")
    st.caption(
        "Studio is being built alongside the existing application. "
        "Classic calculations and workflows remain available."
    )

st.markdown(
    """
<div class="studio-hero">
    <div class="studio-kicker">NeqSim Studio · Beta</div>
    <div class="studio-title">Engineering simulation, in one workspace.</div>
    <div class="studio-lead">
        Build process models, inspect engineering evidence, run studies and move
        toward professional drawings and case-aware engineering assistance — all
        backed by the existing NeqSim calculation core.
    </div>
    <div>
        <span class="studio-status">Classic preserved</span>
        <span class="studio-status">NeqSim core</span>
        <span class="studio-status">Case-based workspace</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

primary, secondary, spacer = st.columns([1.15, 1.0, 3.2])
with primary:
    if st.button("⚙️ Open Process Flowsheet", type="primary", use_container_width=True):
        st.switch_page("pages/35_Process_Flowsheet_Studio.py")
with secondary:
    if st.button("Open Classic", use_container_width=True):
        st.switch_page("welcome.py")

st.markdown('<div class="studio-section"></div>', unsafe_allow_html=True)
st.subheader("Start an engineering workflow")
st.caption(
    "Available workflows open the existing validated implementation. Planned items "
    "stay visible so the Studio roadmap is clear without overstating capability."
)

for row_start in range(0, len(STUDIO_DESTINATIONS), 4):
    columns = st.columns(4)
    for column, destination in zip(
        columns,
        STUDIO_DESTINATIONS[row_start : row_start + 4],
    ):
        if destination.status == STATUS_AVAILABLE:
            state_label = "Available now"
        elif destination.status == STATUS_CORE_IN_PROGRESS:
            state_label = "NeqSim core integration in progress"
        else:
            state_label = "Planned for Studio"

        with column:
            st.markdown(
                f"""
<div class="workflow-card">
    <div class="workflow-icon">{destination.icon}</div>
    <div class="workflow-title">{destination.title}</div>
    <div class="workflow-description">{destination.description}</div>
    <div class="workflow-state">{state_label}</div>
</div>
""",
                unsafe_allow_html=True,
            )
            if destination.available:
                if st.button(
                    f"Open {destination.title}",
                    key=f"open_{destination.key}",
                    use_container_width=True,
                ):
                    st.switch_page(destination.page)
            else:
                st.button(
                    "Coming soon",
                    key=f"planned_{destination.key}",
                    disabled=True,
                    use_container_width=True,
                )

st.markdown('<div class="studio-section"></div>', unsafe_allow_html=True)
left, right = st.columns([1.35, 1.0])

with left:
    st.subheader("Built on the existing Flowsheet Studio")
    st.write(
        "The new workspace does not restart process simulation from zero. It will "
        "reuse the mature generic graph, explicit streams and ports, multi-inlet "
        "execution, convergence diagnostics, workbooks, equipment-design evidence, "
        "subflowsheets, persistence and Process Chat handoff already present in "
        "NeqSim Web."
    )
    with st.expander("Inherited engineering foundation"):
        st.markdown(
            """
- Generic process graph with named material and energy connections
- Multiple independent feeds, mixers, splitters and phase outlets
- Searchable equipment palette and editable engineering properties
- Mass, component and energy closure with fail-loud convergence state
- Equipment design screening and normalized engineering workbooks
- Sensitivity, adjust/specification and bounded study tools
- Native `.neqsim` persistence, schema migration and Process Chat handoff
- Subflowsheets, boundary-port contracts and deployment hardening
"""
        )

with right:
    st.subheader("Engineering drawings")
    st.info(
        "Studio will consume the canonical PFD/P&ID/DEXPI capabilities developed in "
        "the NeqSim core. The web application will focus on viewing, navigation, "
        "selection and engineering workflow rather than creating a second diagram engine."
    )
    st.caption(
        "PFD generation may be simulation-driven. P&ID output remains a proposal "
        "until the required discipline data and accountable review are present."
    )

st.divider()
st.caption(
    "NeqSim Studio is an evolving engineering workspace. Existing NeqSim Classic "
    "pages remain available throughout development."
)
