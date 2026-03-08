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
import threading
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
            st.session_state["ts_task_input"] = tmpl["example"]
            st.session_state["ts_selected_template"] = i

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
#  Main input area
# ─────────────────────────────────────────────────────────────────────────────
task_input = st.text_area(
    "Describe your engineering task",
    value=st.session_state.get("ts_task_input", ""),
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

    from process_chat.task_solver import run_task, TaskResult

    user_comp = _get_composition()
    user_cond = _get_conditions()

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

    # Phase envelope plot if available
    for key, data in lr["all_results"].items():
        if "dew_point_curve" in data:
            st.subheader("📈 Phase Envelope")
            import plotly.graph_objects as go

            fig = go.Figure()
            dew = data["dew_point_curve"]
            bub = data.get("bubble_point_curve", {})

            fig.add_trace(go.Scatter(
                x=dew.get("temperature_C", []),
                y=dew.get("pressure_bara", []),
                mode="lines", name="Dew Point",
                line=dict(color="blue", width=2),
            ))
            if bub.get("temperature_C"):
                fig.add_trace(go.Scatter(
                    x=bub["temperature_C"],
                    y=bub["pressure_bara"],
                    mode="lines", name="Bubble Point",
                    line=dict(color="red", width=2),
                ))

            # Mark cricondenbar / cricondentherm
            if "cricondenbar_bara" in data:
                fig.add_trace(go.Scatter(
                    x=[data["cricondenbar_T_C"]],
                    y=[data["cricondenbar_bara"]],
                    mode="markers+text",
                    name=f"Cricondenbar ({data['cricondenbar_bara']:.1f} bara)",
                    marker=dict(size=10, color="green"),
                    text=[f"Cricondenbar\n{data['cricondenbar_bara']:.1f} bara"],
                    textposition="top right",
                ))
            if "cricondentherm_C" in data:
                fig.add_trace(go.Scatter(
                    x=[data["cricondentherm_C"]],
                    y=[data["cricondentherm_P_bara"]],
                    mode="markers+text",
                    name=f"Cricondentherm ({data['cricondentherm_C']:.1f} °C)",
                    marker=dict(size=10, color="orange"),
                    text=[f"Cricondentherm\n{data['cricondentherm_C']:.1f} °C"],
                    textposition="top left",
                ))

            fig.update_layout(
                title="Phase Envelope",
                xaxis_title="Temperature (°C)",
                yaxis_title="Pressure (bara)",
                template="plotly_white",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Hydrate curve plot if available
    for key, data in lr["all_results"].items():
        if "hydrate_curve" in data:
            st.subheader("🧊 Hydrate Formation Curve")
            import plotly.graph_objects as go

            curve = data["hydrate_curve"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[p["temperature_C"] for p in curve],
                y=[p["pressure_bara"] for p in curve],
                mode="lines+markers",
                name="Hydrate Formation",
                line=dict(color="cyan", width=2),
            ))
            if "hydrate_formation_temperature_C" in data:
                fig.add_trace(go.Scatter(
                    x=[data["hydrate_formation_temperature_C"]],
                    y=[data.get("pressure_bara", 100)],
                    mode="markers+text",
                    name=f"At {data.get('pressure_bara', 100)} bara",
                    marker=dict(size=12, color="red"),
                    text=[f"{data['hydrate_formation_temperature_C']:.1f} °C"],
                    textposition="top right",
                ))

            fig.update_layout(
                title="Hydrate Formation Temperature vs Pressure",
                xaxis_title="Temperature (°C)",
                yaxis_title="Pressure (bara)",
                template="plotly_white",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Properties table if available
    for key, data in lr["all_results"].items():
        if any(k.startswith("density") or k.startswith("z_factor") or k.startswith("Cp")
               for k in data.keys()):
            st.subheader("📊 Fluid Properties")
            props = {
                k: v for k, v in data.items()
                if isinstance(v, (int, float)) and k not in ("number_of_phases",)
            }
            if props:
                prop_df = pd.DataFrame(
                    [{"Property": k, "Value": v} for k, v in props.items()]
                )
                st.dataframe(prop_df, hide_index=True, use_container_width=True)
            break  # only show once

    # Standards results if available
    for key, data in lr["all_results"].items():
        if "GCV_MJ_Sm3" in data or "Wobbe_index_MJ_Sm3" in data:
            st.subheader("📐 Gas Quality / Standards")
            std_df = pd.DataFrame([
                {"Parameter": k, "Value": v}
                for k, v in data.items()
                if isinstance(v, (int, float))
            ])
            if not std_df.empty:
                st.dataframe(std_df, hide_index=True, use_container_width=True)
            break

    # Clear button
    if st.button("🗑️ Clear Results"):
        st.session_state.pop("ts_last_result", None)
        st.session_state.pop("ts_task_input", None)
        st.rerun()
