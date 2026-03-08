"""
Task Solver — Ask an engineering question, get a full report.

Wraps the NeqSim Task Solving Guide 3-step workflow in a hosted UI:
  1. Describe what you want (natural language or template)
  2. System scopes, simulates, and validates — with live progress
  3. Download the report + results

Handles long-running tasks (5-10 min) with step-by-step progress display.
"""

import streamlit as st
import pandas as pd
import json
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from theme import apply_theme, theme_toggle

st.set_page_config(
    page_title="Task Solver",
    page_icon="images/neqsimlogocircleflat.png",
    layout="wide",
)
apply_theme()
theme_toggle()

st.title("🔧 Task Solver")
st.markdown("""
Describe an engineering task and get a full report — **no local Java, Maven, or Python setup required**.

This wraps the [NeqSim Task Solving Guide](https://equinor.github.io/neqsim/development/TASK_SOLVING_GUIDE.html)
3-step workflow (Scope → Analysis → Report) in a hosted interface.
""")

# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar: API key + settings
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # Gemini API key
    api_key_from_secrets = False
    try:
        if "GEMINI_API_KEY" in st.secrets:
            api_key_from_secrets = True
            st.session_state["ts_gemini_api_key"] = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    if not api_key_from_secrets:
        st.text_input(
            "Gemini API Key",
            type="password",
            key="ts_gemini_api_key",
            help="Get a free key from https://aistudio.google.com/",
        )
    else:
        st.success("✓ API key loaded from secrets")

    st.divider()

    report_level = st.radio(
        "Report depth",
        ["quick", "standard", "comprehensive"],
        index=1,
        help=(
            "**Quick**: Brief answer, minimal analysis. "
            "**Standard**: Full report with validation. "
            "**Comprehensive**: Detailed multi-section report."
        ),
    )

    st.divider()
    st.caption(
        "Based on the [Task Solving Guide]"
        "(https://equinor.github.io/neqsim/development/TASK_SOLVING_GUIDE.html). "
        "Task types: Property (A), Process (B), PVT (C), Standards (D), "
        "Phase Envelope (E), Flow Assurance (F), Multi-Step (G)."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Task templates (quick-start buttons)
# ─────────────────────────────────────────────────────────────────────────────
from process_chat.task_solver import TASK_TEMPLATES, DEFAULT_COMPOSITION, TASK_TYPES

st.subheader("Choose a task type or describe your own")

# Template selector
template_cols = st.columns(4)
for i, tmpl in enumerate(TASK_TEMPLATES):
    col = template_cols[i % 4]
    with col:
        if st.button(
            f"**{tmpl['label']}**\n\n{tmpl['type']}",
            key=f"tmpl_{i}",
            help=tmpl["description"],
            use_container_width=True,
        ):
            st.session_state["ts_task_text"] = tmpl["example"]
            st.session_state["ts_selected_template"] = i
            st.rerun()

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
#  Main input area
# ─────────────────────────────────────────────────────────────────────────────
task_input = st.text_area(
    "Describe your engineering task",
    height=120,
    placeholder=(
        "Example: Calculate the JT cooling for a rich gas expanding from 200 to 50 bara.\n"
        "Example: What is the hydrate formation temperature for natural gas at 100 bara?\n"
        "Example: Generate phase envelope and key properties for pipeline gas at 40°C."
    ),
    key="ts_task_text",
)

# Optional: custom composition
with st.expander("📋 Custom Fluid Composition (optional)", expanded=False):
    st.markdown("Leave empty to use a default lean natural gas, or enter your own composition.")

    use_custom = st.checkbox("Use custom composition", key="ts_use_custom")

    if use_custom:
        default_df = pd.DataFrame([
            {"Component": k, "MoleFraction": v}
            for k, v in DEFAULT_COMPOSITION.items()
        ])

        if "ts_comp_df" not in st.session_state:
            st.session_state["ts_comp_df"] = default_df

        edited_comp = st.data_editor(
            st.session_state["ts_comp_df"],
            column_config={
                "Component": st.column_config.TextColumn("Component Name"),
                "MoleFraction": st.column_config.NumberColumn(
                    "Mole Fraction", min_value=0.0, max_value=1.0, format="%.6f",
                ),
            },
            num_rows="dynamic",
            key="ts_comp_editor",
        )
        st.session_state["ts_comp_df"] = edited_comp
        total = edited_comp["MoleFraction"].sum()
        if total > 0:
            st.caption(f"Total: {total:.6f}" + (" *(will be normalized)*" if abs(total - 1.0) > 0.001 else ""))

# Optional: conditions override
with st.expander("🌡️ Override Conditions (optional)", expanded=False):
    cond_cols = st.columns(2)
    with cond_cols[0]:
        cond_temp = st.number_input("Temperature (°C)", value=40.0, min_value=-273.15, max_value=1000.0, key="ts_temp")
    with cond_cols[1]:
        cond_pres = st.number_input("Pressure (bara)", value=100.0, min_value=0.01, max_value=10000.0, key="ts_pres")
    use_conditions = st.checkbox("Apply these conditions", key="ts_use_conditions")

# Optional: upload reference documents
with st.expander("📎 Upload Reference Documents (optional)", expanded=False):
    st.markdown(
        "Upload literature, data sheets, specs, or task descriptions. "
        "Supported: `.txt`, `.md`, `.csv`, `.json`, `.xlsx`, `.pdf`"
    )
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["txt", "md", "csv", "json", "xlsx", "xls", "pdf", "log"],
        accept_multiple_files=True,
        key="ts_uploads",
    )

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
#  Run task
# ─────────────────────────────────────────────────────────────────────────────

def _get_api_key() -> str:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return st.session_state.get("ts_gemini_api_key", "")


def _get_composition() -> dict | None:
    if st.session_state.get("ts_use_custom") and "ts_comp_df" in st.session_state:
        df = st.session_state["ts_comp_df"]
        df = df[df["MoleFraction"] > 0]
        if not df.empty:
            total = df["MoleFraction"].sum()
            return {
                row["Component"]: round(row["MoleFraction"] / total, 8)
                for _, row in df.iterrows()
            }
    return None


def _get_conditions() -> dict | None:
    if st.session_state.get("ts_use_conditions"):
        return {
            "temperature_C": st.session_state.get("ts_temp", 40.0),
            "pressure_bara": st.session_state.get("ts_pres", 100.0),
        }
    return None


def _get_uploaded_text() -> str | None:
    """Extract text from all uploaded files."""
    files = st.session_state.get("ts_uploads")
    if not files:
        return None
    from process_chat.task_solver import extract_text_from_upload
    parts = []
    for f in files:
        f.seek(0)
        parts.append(extract_text_from_upload(f))
    combined = "\n\n---\n\n".join(parts)
    return combined if combined.strip() else None


# Run button
run_col1, run_col2 = st.columns([1, 3])
with run_col1:
    run_clicked = st.button(
        "🚀 Solve Task",
        type="primary",
        use_container_width=True,
        disabled=not task_input.strip(),
    )

if run_clicked:
    api_key = _get_api_key()
    if not api_key:
        st.error("No Gemini API key found. Enter one in the sidebar or set it in Streamlit secrets.")
        st.stop()

    if not task_input.strip():
        st.warning("Please describe your engineering task.")
        st.stop()

    user_comp = _get_composition()
    user_cond = _get_conditions()
    uploaded_text = _get_uploaded_text()

    # ── Live progress execution ───────────────────────────────────────────
    # Create status containers upfront — one for the overall workflow
    st.subheader("📊 Task Execution")

    # We use a placeholder for each phase so we can update in real time
    progress_bar = st.progress(0, text="Starting task…")
    status_container = st.status("Running task…", expanded=True)

    # Shared state for progress callback
    _progress_log = []

    def progress_callback(step_num, step_title, status_msg):
        _progress_log.append((step_num, step_title, status_msg))

    # Run the task (this is the potentially long-running part)
    with status_container:
        st.markdown(f"**Task:** {task_input.strip()}")
        st.markdown(f"**Report level:** {report_level}")
        if user_comp:
            st.markdown(f"**Custom composition:** {len(user_comp)} components")
        if uploaded_text:
            st.markdown(f"**Uploaded documents:** {len(st.session_state.get('ts_uploads', []))} file(s)")
        st.divider()

        # Step 1: Scope
        step1_ph = st.empty()
        step1_ph.info("⏳ **Step 1: Scope & Research** — Classifying task…")

        try:
            from process_chat.task_solver import classify_and_scope
            extra_ctx = ""
            if report_level == "comprehensive":
                extra_ctx = "\nUser wants a comprehensive / detailed report with full analysis."
            elif report_level == "quick":
                extra_ctx = "\nUser wants a quick answer. Minimize steps."

            spec = classify_and_scope(
                api_key=api_key,
                user_request=task_input.strip() + extra_ctx,
                user_composition=user_comp,
                uploaded_docs=uploaded_text,
                ai_model="gemini-2.0-flash",
            )

            if user_comp:
                spec["composition"] = user_comp
            if user_cond:
                spec["conditions"] = {**spec.get("conditions", {}), **user_cond}

            task_type = spec.get("task_type", "A")
            step1_ph.success(
                f"✅ **Step 1: Scope & Research** — "
                f"Type **{task_type}** ({TASK_TYPES.get(task_type, '')}), "
                f"Complexity: **{spec.get('complexity', 'standard')}**, "
                f"EOS: **{spec.get('eos_model') or 'auto'}**"
            )

            progress_bar.progress(15, text="Task classified. Running analysis…")

        except Exception as e:
            step1_ph.error(f"❌ **Step 1 failed:** {e}")
            status_container.update(label="Task failed", state="error")
            st.stop()

        # Show the plan
        steps = spec.get("steps", [])
        with st.expander("📋 Task Plan", expanded=False):
            st.json(spec, expanded=False)
            for i, s in enumerate(steps):
                st.markdown(
                    f"**{i+1}.** {s.get('title', '')} — {s.get('description', '')}"
                )

        st.divider()

        # Step 2: Generate, execute, and auto-fix code for each step
        from process_chat.task_solver import generate_and_run_code, TaskStep

        all_results = {}
        total_steps = len(steps)
        results_steps = []
        all_code_parts = []

        for i, step_info in enumerate(steps):
            step_title = step_info.get("title", f"Step {i+1}")
            step_desc = step_info.get("description", "")
            step_num = i + 1

            ph = st.empty()
            ph.info(f"⏳ **Step {step_num}: {step_title}** — Generating code…")

            pct_base = 15 + int(70 * i / max(total_steps, 1))
            pct_done = 15 + int(70 * (i + 1) / max(total_steps, 1))
            progress_bar.progress(pct_base, text=f"Running: {step_title}…")

            # Use the agentic code-gen + exec + fix loop
            def _step_progress(sn, title, msg, _ph=ph, _pb=progress_bar, _base=pct_base):
                _ph.info(f"⏳ **Step {sn}: {title}** — {msg}")

            task_step = generate_and_run_code(
                api_key=api_key,
                spec=spec,
                step_info=step_info,
                step_number=step_num,
                ai_model="gemini-2.0-flash",
                progress_cb=_step_progress,
            )

            results_steps.append(task_step)

            if task_step.status == "done":
                # Build quick summary from result_data
                summary_parts = []
                if task_step.result_data and isinstance(task_step.result_data, dict):
                    key_results = task_step.result_data.get("key_results", task_step.result_data)
                    if isinstance(key_results, dict):
                        for k, v in list(key_results.items())[:8]:
                            if isinstance(v, (int, float)):
                                summary_parts.append(f"`{k}` = **{v}**")
                            elif isinstance(v, str) and len(v) < 100:
                                summary_parts.append(f"`{k}` = {v}")
                summary_line = " · ".join(summary_parts) if summary_parts else "Done"
                attempt_info = f" ({task_step.attempts} attempt{'s' if task_step.attempts > 1 else ''})" if task_step.attempts > 1 else ""
                ph.success(
                    f"✅ **Step {step_num}: {step_title}** ({task_step.elapsed_seconds:.1f}s{attempt_info})\n\n{summary_line}"
                )
                all_results[f"step_{step_num}"] = task_step.result_data
            else:
                ph.error(f"❌ **Step {step_num}: {step_title}** — Failed after {task_step.attempts} attempts\n\n{task_step.error[:200]}")
                all_results[f"step_{step_num}"] = {"error": task_step.error}

            if task_step.code:
                all_code_parts.append(
                    f"# === Step {step_num}: {step_title} ===\n\n{task_step.code}\n"
                )

            progress_bar.progress(pct_done, text=f"Completed: {step_title}")

        all_generated_code = "\n\n".join(all_code_parts)

        st.divider()

        # Step 3: Generate report
        step3_ph = st.empty()
        step3_ph.info("⏳ **Report Generation** — Writing engineering report…")
        progress_bar.progress(85, text="Generating report…")

        try:
            from process_chat.task_solver import _generate_report_text, _generate_html_report

            report_text = _generate_report_text(
                api_key=api_key,
                spec=spec,
                all_results=all_results,
                all_code=all_generated_code,
                ai_model="gemini-2.0-flash",
            )
            report_html = _generate_html_report(
                title=spec.get("title", task_input.strip()[:80]),
                report_md=report_text,
                spec=spec,
                all_results=all_results,
            )

            step3_ph.success("✅ **Report Generation** — Complete!")
            progress_bar.progress(100, text="Task complete!")

        except Exception as e:
            step3_ph.error(f"❌ **Report Generation failed:** {e}")
            report_text = f"Report generation error: {e}"
            report_html = ""

    # Update the status container
    any_error = any(s.status == "error" for s in results_steps)
    if any_error:
        status_container.update(label="Task completed with errors", state="error", expanded=False)
    else:
        status_container.update(label="Task completed successfully!", state="complete", expanded=False)

    # ── Store results in session state ────────────────────────────────────
    st.session_state["ts_last_result"] = {
        "spec": spec,
        "all_results": all_results,
        "report_text": report_text,
        "report_html": report_html,
        "steps": results_steps,
        "task_input": task_input.strip(),
        "all_code": all_generated_code,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Display results (persisted across reruns)
# ─────────────────────────────────────────────────────────────────────────────
if "ts_last_result" in st.session_state:
    lr = st.session_state["ts_last_result"]

    st.divider()
    st.subheader("📄 Engineering Report")

    # Report content
    st.markdown(lr["report_text"])

    # Download buttons
    st.divider()
    dl_cols = st.columns(4)

    with dl_cols[0]:
        st.download_button(
            "📥 Download HTML Report",
            data=lr["report_html"],
            file_name="neqsim_task_report.html",
            mime="text/html",
            use_container_width=True,
        )

    with dl_cols[1]:
        results_json_str = json.dumps(lr["all_results"], indent=2, default=str)
        st.download_button(
            "📥 Download results.json",
            data=results_json_str,
            file_name="results.json",
            mime="application/json",
            use_container_width=True,
        )

    with dl_cols[2]:
        spec_json_str = json.dumps(lr["spec"], indent=2, default=str)
        st.download_button(
            "📥 Download task_spec.json",
            data=spec_json_str,
            file_name="task_spec.json",
            mime="application/json",
            use_container_width=True,
        )

    with dl_cols[3]:
        if lr.get("all_code"):
            st.download_button(
                "📥 Download Python Code",
                data=lr["all_code"],
                file_name="neqsim_task_code.py",
                mime="text/x-python",
                use_container_width=True,
            )

    # Generated code (expandable)
    if lr.get("all_code"):
        with st.expander("🐍 Generated Python Code (reproducible)", expanded=False):
            st.code(lr["all_code"], language="python")

    # Detailed results expandable
    with st.expander("🔍 Detailed Step Results", expanded=False):
        for step in lr["steps"]:
            icon = "✅" if step.status == "done" else "❌"
            attempt_info = f", {step.attempts} attempt{'s' if step.attempts > 1 else ''}" if step.attempts > 0 else ""
            st.markdown(f"**{icon} Step {step.number}: {step.title}** ({step.elapsed_seconds:.1f}s{attempt_info})")
            if step.code:
                with st.expander(f"Code — Step {step.number}", expanded=False):
                    st.code(step.code, language="python")
            if step.result_data:
                with st.expander(f"Raw data — Step {step.number}", expanded=False):
                    st.json(step.result_data)
            if step.error:
                st.error(step.error)

    # ── Charts (auto-detected + explicit) ─────────────────────────────────
    try:
        from process_chat.task_solver import _build_plotly_charts
        figures = _build_plotly_charts(lr["all_results"])
        if figures:
            st.subheader("📈 Charts")
            for fig in figures:
                st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass  # chart rendering is optional, don't block page

    # Properties table if available
    for key, data in lr["all_results"].items():
        if not isinstance(data, dict):
            continue
        if any(k.startswith("density") or k.startswith("z_factor") or k.startswith("Cp")
               for k in data.keys()):
            st.subheader("📊 Fluid Properties")
            props = {
                k: v for k, v in data.items()
                if isinstance(v, (int, float)) and k not in ("number_of_phases",)
                and not k.startswith("_")
            }
            if props:
                prop_df = pd.DataFrame(
                    [{"Property": k.replace("_", " "), "Value": v} for k, v in props.items()]
                )
                st.dataframe(prop_df, hide_index=True, use_container_width=True)
            break  # only show once

    # Standards results if available
    for key, data in lr["all_results"].items():
        if not isinstance(data, dict):
            continue
        if "GCV_MJ_Sm3" in data or "Wobbe_index_MJ_Sm3" in data:
            st.subheader("📐 Gas Quality / Standards")
            std_df = pd.DataFrame([
                {"Parameter": k.replace("_", " "), "Value": v}
                for k, v in data.items()
                if isinstance(v, (int, float))
            ])
            if not std_df.empty:
                st.dataframe(std_df, hide_index=True, use_container_width=True)
            break

    # Clear button
    if st.button("🗑️ Clear Results"):
        st.session_state.pop("ts_last_result", None)
        st.session_state.pop("ts_task_text", None)
        st.session_state.pop("ts_iteration_count", None)
        st.rerun()

    # ─────────────────────────────────────────────────────────────────────
    #  Follow-up / iterate on the report
    # ─────────────────────────────────────────────────────────────────────
    st.divider()
    iteration_n = st.session_state.get("ts_iteration_count", 0)
    st.subheader(f"🔄 Continue / Refine Report" + (f" (iteration {iteration_n})" if iteration_n else ""))
    st.markdown(
        "Ask for more detail, additional calculations, or different conditions. "
        "You can also upload reference documents below."
    )

    followup_input = st.text_area(
        "What would you like to add or change?",
        height=100,
        placeholder=(
            "Example: Also calculate properties at 200 bara.\n"
            "Example: Add MEG inhibition analysis for the hydrate case.\n"
            "Example: Compare with Peng-Robinson EOS.\n"
            "Example: Include the data from the uploaded PDF in the report."
        ),
        key="ts_followup_text",
    )

    # Upload docs for follow-up
    followup_files = st.file_uploader(
        "Upload additional reference documents (optional)",
        type=["txt", "md", "csv", "json", "xlsx", "xls", "pdf", "log"],
        accept_multiple_files=True,
        key="ts_followup_uploads",
    )

    followup_clicked = st.button(
        "🔄 Continue Analysis",
        type="primary",
        disabled=not followup_input.strip(),
    )

    if followup_clicked:
        api_key = _get_api_key()
        if not api_key:
            st.error("No Gemini API key found.")
            st.stop()

        from process_chat.task_solver import follow_up_task, extract_text_from_upload

        # Extract text from follow-up uploads
        followup_doc_text = None
        if followup_files:
            parts = []
            for f in followup_files:
                f.seek(0)
                parts.append(extract_text_from_upload(f))
            followup_doc_text = "\n\n---\n\n".join(parts)
            if not followup_doc_text.strip():
                followup_doc_text = None

        # Also include any docs from the initial upload
        initial_doc_text = _get_uploaded_text()
        if initial_doc_text and followup_doc_text:
            combined_docs = initial_doc_text + "\n\n---\n\n" + followup_doc_text
        else:
            combined_docs = followup_doc_text or initial_doc_text

        st.subheader("📊 Follow-up Execution")
        fu_progress = st.progress(0, text="Starting follow-up…")
        fu_status = st.status("Running follow-up…", expanded=True)

        with fu_status:
            st.markdown(f"**Follow-up:** {followup_input.strip()}")
            if combined_docs:
                st.markdown(f"**Reference documents:** included")
            st.divider()

            fu_step_ph = st.empty()
            fu_step_ph.info("⏳ Planning follow-up steps…")

            def fu_progress_cb(step_num, step_title, status_msg):
                fu_step_ph.info(f"⏳ **Step {step_num}: {step_title}** — {status_msg}")

            fu_result = follow_up_task(
                api_key=api_key,
                follow_up_request=followup_input.strip(),
                previous_result=lr,
                uploaded_docs=combined_docs,
                ai_model="gemini-2.0-flash",
                progress_cb=fu_progress_cb,
            )

            if fu_result.success:
                fu_step_ph.success("✅ Follow-up complete!")
                fu_progress.progress(100, text="Follow-up complete!")
                fu_status.update(label="Follow-up completed!", state="complete", expanded=False)
            else:
                fu_step_ph.warning("⚠️ Follow-up completed with some errors")
                fu_progress.progress(100, text="Follow-up done (with errors)")
                fu_status.update(label="Follow-up completed with errors", state="error", expanded=False)

        # Merge into session state — replace with updated results
        st.session_state["ts_last_result"] = {
            "spec": lr["spec"],
            "all_results": fu_result.results_json,
            "report_text": fu_result.report_text or lr["report_text"],
            "report_html": fu_result.report_html or lr["report_html"],
            "steps": lr["steps"] + [s for s in fu_result.steps],
            "task_input": lr["task_input"],
            "all_code": fu_result.all_code or lr.get("all_code", ""),
        }
        st.session_state["ts_iteration_count"] = iteration_n + 1
        st.rerun()
