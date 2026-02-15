"""
Process Chat — Chat with your NeqSim process model.

Upload a .neqsim process file → introspect → ask questions → run what-if scenarios.
"""
import streamlit as st
import pandas as pd
import json
import os
import sys
import traceback

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from theme import apply_theme, theme_toggle

st.set_page_config(
    page_title="Process Chat",
    page_icon="images/neqsimlogocircleflat.png",
    layout="wide",
)
apply_theme()
theme_toggle()

st.title("💬 Process Chat")
st.markdown("""
Chat with your NeqSim process model. Upload a `.neqsim` process file **or build a new process from scratch**.
Ask questions, run what-if scenarios, and explore planning options.
""")

# ─────────────────────────────────────────────
# Sidebar: Model Upload & AI Settings
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Process Model")

    uploaded_file = st.file_uploader(
        "Upload .neqsim process file",
        type=["neqsim", "xml", "zip"],
        help="Upload a NeqSim process model file (.neqsim compressed XML)",
    )

    if uploaded_file is not None:
        # Only reload if file changed
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("_loaded_file_key") != file_key:
            with st.spinner("Loading process model..."):
                try:
                    from process_chat.process_model import NeqSimProcessModel

                    file_bytes = uploaded_file.read()
                    model = NeqSimProcessModel.from_bytes(file_bytes, uploaded_file.name)
                    st.session_state["process_model"] = model
                    st.session_state["process_model_bytes"] = file_bytes
                    st.session_state["process_model_name"] = uploaded_file.name
                    st.session_state["_loaded_file_key"] = file_key
                    # Reset chat session when model changes
                    st.session_state.pop("chat_session", None)
                    st.session_state["chat_messages"] = []
                    st.success(f"✓ Model loaded: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Failed to load model: {str(e)}")
                    with st.expander("📋 Error Details"):
                        st.code(traceback.format_exc())
                    st.session_state.pop("process_model", None)
        else:
            st.success(f"✓ {uploaded_file.name}")

    # --- Start New Process button ---
    if st.button("🔨 Start New Process", use_container_width=True,
                 help="Build a process from scratch by describing it in chat"):
        # Enter builder mode — clear any stale state
        st.session_state.pop("process_model", None)
        st.session_state.pop("_loaded_file_key", None)
        st.session_state.pop("chat_session", None)
        st.session_state["chat_messages"] = []
        st.session_state["_builder_mode"] = True
        st.rerun()
    
    st.divider()

    # API key fallback — if not in secrets, let user enter here too
    api_key_from_secrets = False
    try:
        if 'GEMINI_API_KEY' in st.secrets:
            api_key_from_secrets = True
            st.session_state['gemini_api_key'] = st.secrets['GEMINI_API_KEY']
    except Exception:
        pass
    if not api_key_from_secrets and not st.session_state.get('gemini_api_key', ''):
        st.sidebar.text_input("Gemini API Key", type="password", key="gemini_api_key",
                              help="Get a free key from https://aistudio.google.com/")

    st.divider()

    # --- Download buttons (Python script + .neqsim file) ---
    chat_session = st.session_state.get("chat_session")
    if chat_session:
        last_script = chat_session.get_last_script()
        last_save = chat_session.get_last_save_bytes()
        if last_script or last_save:
            st.subheader("📥 Downloads")
            if last_script:
                proc_name = ""
                builder = chat_session.get_builder()
                if builder:
                    proc_name = builder.process_name
                script_fname = (proc_name.replace(" ", "_").lower() or "process") + ".py"
                st.download_button(
                    "📜 Python Script",
                    data=last_script,
                    file_name=script_fname,
                    mime="text/x-python",
                    use_container_width=True,
                )
            if last_save:
                save_fname = ""
                builder = chat_session.get_builder()
                if builder:
                    save_fname = builder.process_name.replace(" ", "_").lower()
                save_fname = (save_fname or "process") + ".neqsim"
                st.download_button(
                    "💾 .neqsim File",
                    data=last_save,
                    file_name=save_fname,
                    mime="application/octet-stream",
                    use_container_width=True,
                )
            st.divider()

    # --- Diagram Settings ---
    if st.session_state.get("process_model") is not None:
        st.subheader("📐 Diagram Settings")
        diagram_style = st.selectbox(
            "Style",
            ["HYSYS", "NEQSIM", "PROII", "ASPEN_PLUS"],
            index=0,
            key="diagram_style",
            help="Visual style for the process flow diagram",
        )
        diagram_detail = st.selectbox(
            "Detail Level",
            ["ENGINEERING", "CONCEPTUAL", "DEBUG"],
            index=0,
            key="diagram_detail",
            help="Amount of information shown on the diagram",
        )
        st.divider()

    # Example questions — adapt to mode
    st.subheader("💡 Example Questions")
    model = st.session_state.get("process_model")
    if model is not None:
        example_questions = [
            "What equipment is in this process?",
            "What are the current stream conditions?",
            "What is the total compressor power?",
            "What if we increase the export pressure by 10 bara?",
            "What if we reduce the cooler outlet temperature to 30°C?",
            "What happens if we increase feed flow by 10%?",
            "Show me the Python script",
            "Save the process",
        ]
    else:
        example_questions = [
            "Build a simple gas compression process",
            "Create a 3-stage compression train with intercooling for natural gas from 30 to 200 bara",
            "Build a gas dehydration process with TEG absorber",
            "Create a separation process with inlet separator, compressor, and export cooler",
            "Build a gas processing plant with separation and compression",
        ]
    for q in example_questions:
        if st.button(q, key=f"ex_{hash(q)}", use_container_width=True):
            st.session_state["_pending_question"] = q
            # If no model loaded, activate builder mode so the page
            # doesn't hit st.stop() before the chat input is processed.
            if model is None:
                st.session_state["_builder_mode"] = True


# ─────────────────────────────────────────────
# Check model state and show overview
# ─────────────────────────────────────────────
model = st.session_state.get("process_model")
builder_mode = st.session_state.get("_builder_mode", False)

if model is None and not builder_mode:
    st.info("👆 Upload a `.neqsim` process model file in the sidebar, or click **Start New Process** to build one from scratch.")
    st.markdown("""
    ### Getting Started
    
    **Option 1: Upload an existing model**
    Upload a `.neqsim` file to analyze, query, and run what-if scenarios.

    **Option 2: Build from scratch**
    Click **🔨 Start New Process** in the sidebar, then describe the process you want to build:
    - *"Build a gas compression process with methane and ethane at 50 bara"*
    - *"Create a 3-stage compression train with intercooling"*
    - *"Build a separation and dehydration process"*
    
    The AI will design the process, run the simulation, and you can then:
    - Ask what-if questions
    - Download the Python script
    - Save as a `.neqsim` file
    """)
    st.stop()


# ─────────────────────────────────────────────
# Model Introspection Panel (only when model loaded)
# ─────────────────────────────────────────────
if model is not None:
    with st.expander("📊 Process Model Overview", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Unit Operations")
            units = model.list_units()
            if units:
                unit_data = []
                for u in units:
                    row = {"Name": u.name, "Type": u.unit_type}
                    row.update(u.properties)
                    unit_data.append(row)
                st.dataframe(pd.DataFrame(unit_data), use_container_width=True, hide_index=True)
            else:
                st.info("No unit operations found.")

        with col2:
            st.markdown("#### Streams")
            streams = model.list_streams()
            if streams:
                stream_data = []
                for s in streams:
                    stream_data.append({
                        "Name": s.name,
                        "T (°C)": f"{s.temperature_C:.1f}" if s.temperature_C is not None else "—",
                        "P (bara)": f"{s.pressure_bara:.2f}" if s.pressure_bara is not None else "—",
                        "Flow (kg/hr)": f"{s.flow_rate_kg_hr:.1f}" if s.flow_rate_kg_hr is not None else "—",
                    })
                st.dataframe(pd.DataFrame(stream_data), use_container_width=True, hide_index=True)
            else:
                st.info("No streams found.")
elif builder_mode:
    st.info("🔨 **Builder mode** — Describe the process you want to create in the chat below.")

# ─────────────────────────────────────────────
# Process Flow Diagram (PFD)
# ─────────────────────────────────────────────
if model is not None:
    with st.expander("📐 Process Flow Diagram", expanded=True):
        try:
            pfd_style = st.session_state.get("diagram_style", "HYSYS")
            pfd_detail = st.session_state.get("diagram_detail", "ENGINEERING")
            dot_source = model.get_diagram_dot(
                style=pfd_style,
                detail_level=pfd_detail,
                show_stream_values=(pfd_detail != "CONCEPTUAL"),
            )
            st.graphviz_chart(dot_source, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render process flow diagram: {e}")


# ─────────────────────────────────────────────
# Chat Interface
# ─────────────────────────────────────────────

def _show_comparison(comparison):
    """Display scenario comparison results inline."""
    from process_chat.scenario_engine import results_summary_table, comparison_to_dataframe

    st.markdown("---")
    st.markdown("**📊 Scenario Comparison**")
    
    summary_df = results_summary_table(comparison)
    if not summary_df.empty:
        # Filter to important KPIs: changed values, power/duty, mass balance
        important_suffixes = ('.power_kW', '.duty_kW')
        
        def is_display_worthy(row):
            kpi = row.get('KPI', '')
            # Always show summary & equipment KPIs
            if kpi.startswith(('total_', 'mass_balance')):
                return True
            if kpi.endswith(important_suffixes):
                return True
            # Show rows where values changed between columns
            base_val = row.get('BASE')
            for col in summary_df.columns:
                if col not in ('KPI', 'Unit', 'BASE'):
                    case_val = row.get(col)
                    if base_val is not None and case_val is not None:
                        try:
                            if abs(float(base_val) - float(case_val)) > 0.01:
                                return True
                        except (ValueError, TypeError):
                            pass
            return False
        
        display_df = summary_df[summary_df.apply(is_display_worthy, axis=1)].copy()
        if not display_df.empty:
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No significant KPI changes detected.")
    
    # Constraints
    if comparison.constraint_summary:
        st.markdown("**Constraint Check:**")
        for c in comparison.constraint_summary:
            icon = {"OK": "✅", "WARN": "⚠️", "VIOLATION": "❌"}.get(c["status"], "❓")
            st.markdown(f"{icon} **{c['constraint']}** ({c['status']}): {c['detail']}")


# Initialize chat history
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

# Display chat messages
for msg in st.session_state["chat_messages"]:
    role = msg["role"]
    if role == "system":
        continue  # Don't display system messages
    with st.chat_message(role):
        st.markdown(msg["content"])

        # If there's a comparison result attached, show it
        if "comparison" in msg and msg["comparison"] is not None:
            _show_comparison(msg["comparison"])


# Handle pending question from sidebar buttons
if "_pending_question" in st.session_state:
    pending = st.session_state.pop("_pending_question")
    st.session_state["_chat_input"] = pending


# Chat input
user_input = st.chat_input("Ask about your process model...")

# Also check for pending question
if not user_input and "_chat_input" in st.session_state:
    user_input = st.session_state.pop("_chat_input")

if user_input:
    # Get API key from secrets or session state
    api_key_val = ""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            api_key_val = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    if not api_key_val:
        api_key_val = st.session_state.get("gemini_api_key", "")

    if not api_key_val:
        st.error("No Gemini API key found. Set one on the front page or in Streamlit secrets.")
        st.stop()

    # Display user message
    st.session_state["chat_messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get or create chat session
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                from process_chat.chat_tools import ProcessChatSession

                # Create session if needed (or if model changed)
                if "chat_session" not in st.session_state:
                    st.session_state["chat_session"] = ProcessChatSession(
                        model=model,     # None in builder mode
                        api_key=api_key_val,
                        ai_model="gemini-2.0-flash",
                    )

                session = st.session_state["chat_session"]
                response = session.chat(user_input)
                comparison = session.get_last_comparison()

                # --- Sync model if builder created one ---
                if session.model is not None and st.session_state.get("process_model") is None:
                    st.session_state["process_model"] = session.model
                    st.session_state["_builder_mode"] = False
                    st.session_state["process_model_name"] = "Built Process"

                st.markdown(response)

                # Store message with optional comparison
                msg_data = {"role": "assistant", "content": response}
                if comparison is not None:
                    msg_data["comparison"] = comparison
                    _show_comparison(comparison)

                st.session_state["chat_messages"].append(msg_data)

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state["chat_messages"].append(
                    {"role": "assistant", "content": f"⚠️ {error_msg}"}
                )

    st.rerun()


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🗑️ Clear Chat"):
        st.session_state["chat_messages"] = []
        st.session_state.pop("chat_session", None)
        st.rerun()
with col2:
    if st.button("🔄 Reset All"):
        st.session_state.pop("process_model", None)
        st.session_state.pop("_loaded_file_key", None)
        st.session_state.pop("_builder_mode", None)
        st.session_state.pop("chat_session", None)
        st.session_state["chat_messages"] = []
        st.rerun()
with col3:
    if model:
        st.download_button(
            "📥 Download Model Summary",
            data=model.get_model_summary(),
            file_name="model_summary.txt",
            mime="text/plain",
        )
