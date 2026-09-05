"""Professional entry page for the new NeqSim Studio workspace."""

from __future__ import annotations

from html import escape
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
from studio.case_context import (  # noqa: E402
    clear_active_case,
    decode_portable_case,
    encode_portable_case,
    get_active_case,
    queue_new_case,
    queue_open_case,
    queue_recent_case,
    recent_cases,
    save_case_as,
)
from studio.results_context import remember_result_destination  # noqa: E402
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
        margin: 0 0 0.35rem 0;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: #315ba0;
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
        margin: 0.85rem 0 0 0;
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
        margin: 0 0 0.42rem 0;
    }
    .workflow-description {
        color: #66738a;
        line-height: 1.45;
        font-size: 0.89rem;
        margin: 0;
    }
    .workflow-state {
        margin: 0.75rem 0 0 0;
        color: #315ba0;
        font-size: 0.76rem;
        font-weight: 700;
    }
    .studio-section {
        margin-top: 1.8rem;
        margin-bottom: 0.35rem;
    }
    .case-summary {
        padding: 1rem 1.15rem;
        border: 1px solid rgba(49, 91, 160, 0.20);
        border-radius: 14px;
        background: rgba(247, 250, 255, 0.90);
    }
    .case-title {
        font-size: 1.05rem;
        margin: 0 0 0.25rem 0;
    }
    @media (max-width: 720px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .studio-hero {
            padding: 1.35rem 1.25rem;
        }
        .studio-title {
            font-size: 2rem;
        }
        .studio-lead {
            font-size: 0.98rem;
        }
        .workflow-card {
            min-height: auto;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)

active_case = get_active_case(st.session_state)
active_case_name_html = escape(active_case["name"]) if active_case else ""

with st.sidebar:
    st.markdown("### Workspace")
    st.success("NeqSim Studio · Beta")
    if st.button("← NeqSim Classic", use_container_width=True):
        st.switch_page("home.py")
    st.caption(
        "Studio is being built alongside the existing application. "
        "Classic calculations and workflows remain available."
    )
    if active_case:
        st.divider()
        st.markdown("**Active case**")
        st.write(active_case["name"])
        st.caption(
            f"{active_case['status'].replace('-', ' ').title()} · "
            f"{active_case['thermodynamics']['eos_model']} · "
            f"schema v{active_case['case_schema_version']}"
        )

st.markdown(
    """
<section class="studio-hero" aria-labelledby="studio-page-title">
    <p class="studio-kicker">NeqSim Studio · Beta</p>
    <h1 class="studio-title" id="studio-page-title">
        Engineering simulation, in one workspace.
    </h1>
    <p class="studio-lead">
        Build process models, inspect engineering evidence, run studies and move
        toward professional drawings and case-aware engineering assistance — all
        backed by the existing NeqSim calculation core.
    </p>
    <div aria-label="Workspace status">
        <span class="studio-status">Classic preserved</span>
        <span class="studio-status">NeqSim core</span>
        <span class="studio-status">Case-based workspace</span>
    </div>
</section>
""",
    unsafe_allow_html=True,
)

st.subheader("Case workspace")
st.caption(
    "The active case follows you across Studio pages. Downloads remain the same "
    "portable Process Flowsheet Studio JSON format used today."
)

if active_case:
    summary, actions = st.columns([1.55, 1.0])
    with summary:
        st.markdown(
            f"""
<section class="case-summary" aria-label="Active case summary">
    <h3 class="case-title">{active_case_name_html}</h3>
    <span>{active_case['status'].replace('-', ' ').title()} ·
    {active_case['thermodynamics']['eos_model']} ·
    {active_case['units']['system']} units</span><br/>
    <small>Updated {active_case['provenance']['modified_at']}</small>
</section>
""",
            unsafe_allow_html=True,
        )
        if active_case.get("error"):
            st.error(active_case["error"])
        for warning in active_case.get("warnings", []):
            st.warning(warning)
    with actions:
        st.download_button(
            "Download active case",
            data=encode_portable_case(active_case["case_spec"]),
            file_name="neqsim_studio_case.json",
            mime="application/json",
            use_container_width=True,
        )
        if st.button("Continue active case", type="primary", use_container_width=True):
            queue_open_case(
                st.session_state,
                active_case["case_spec"],
                preserve_identity=True,
            )
            st.switch_page("pages/35_Process_Flowsheet_Studio.py")

    with st.expander("Save As or reset active case"):
        save_as_name = st.text_input(
            "New case name",
            value=f"{active_case['name']} copy",
            key="studio_save_as_name",
        )
        save_as_col, reset_col = st.columns(2)
        if save_as_col.button("Save As", use_container_width=True):
            try:
                cloned_case = save_case_as(st.session_state, save_as_name)
            except ValueError as save_error:
                st.error(str(save_error))
            else:
                queue_open_case(
                    st.session_state,
                    cloned_case["case_spec"],
                    preserve_identity=True,
                )
                st.switch_page("pages/35_Process_Flowsheet_Studio.py")
        confirm_reset = st.checkbox(
            "Confirm reset",
            help=(
                "Clears the active Studio context; Classic session data is not "
                "changed."
            ),
        )
        if reset_col.button(
            "Reset active case",
            disabled=not confirm_reset,
            use_container_width=True,
        ):
            clear_active_case(st.session_state)
            st.rerun()
else:
    st.info(
        "No active Studio case yet. Start the validated flowsheet template or open "
        "an existing portable case."
    )

new_col, open_col = st.columns(2)
with new_col:
    if st.button("＋ New process case", type="primary", use_container_width=True):
        queue_new_case(st.session_state)
        st.switch_page("pages/35_Process_Flowsheet_Studio.py")
with open_col:
    uploaded_case = st.file_uploader(
        "Open portable case JSON",
        type=["json"],
        key="studio_workspace_case_upload",
    )
    if st.button(
        "Open uploaded case",
        disabled=uploaded_case is None,
        use_container_width=True,
    ):
        try:
            portable_case = decode_portable_case(uploaded_case.getvalue())
        except ValueError as open_error:
            st.error(str(open_error))
        else:
            queue_open_case(st.session_state, portable_case)
            st.switch_page("pages/35_Process_Flowsheet_Studio.py")

available_recent_cases = recent_cases(st.session_state)
if available_recent_cases:
    with st.expander("Recent cases", expanded=False):
        for recent_case in available_recent_cases[:5]:
            label_col, open_recent_col = st.columns([3.0, 1.0])
            label_col.write(
                f"**{recent_case['name']}**  \n"
                f"{recent_case['status'].replace('-', ' ').title()} · "
                f"{recent_case['thermodynamics']['eos_model']}"
            )
            if open_recent_col.button(
                f"Open recent case · {recent_case['name']}",
                key=f"open_recent_{recent_case['case_id']}",
                use_container_width=True,
            ):
                queue_recent_case(st.session_state, recent_case["case_id"])
                st.switch_page("pages/35_Process_Flowsheet_Studio.py")

primary, secondary, spacer = st.columns([1.15, 1.0, 3.2])
with primary:
    if st.button("⚙️ Open Process Flowsheet", type="primary", use_container_width=True):
        st.switch_page("pages/35_Process_Flowsheet_Studio.py")
with secondary:
    if st.button("Open Classic", use_container_width=True):
        st.switch_page("home.py")

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

        workflow_title_id = f"workflow-title-{destination.key}"
        destination_icon_html = escape(destination.icon)
        destination_title_html = escape(destination.title)
        destination_description_html = escape(destination.description)
        state_label_html = escape(state_label)

        with column:
            st.markdown(
                f"""
<article class="workflow-card" aria-labelledby="{workflow_title_id}">
    <div class="workflow-icon" aria-hidden="true">{destination_icon_html}</div>
    <h3 class="workflow-title" id="{workflow_title_id}">
        {destination_title_html}
    </h3>
    <p class="workflow-description">{destination_description_html}</p>
    <p class="workflow-state">{state_label_html}</p>
</article>
""",
                unsafe_allow_html=True,
            )
            if destination.available:
                if st.button(
                    f"Open {destination.title}",
                    key=f"open_{destination.key}",
                    use_container_width=True,
                ):
                    if destination.page == "pages/10_Studio_Results.py":
                        remember_result_destination(
                            st.session_state,
                            destination.key,
                        )
                    st.switch_page(destination.page)
            else:
                st.button(
                    f"Coming soon · {destination.title}",
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
