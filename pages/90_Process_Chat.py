"""
Process Chat — Chat with your NeqSim process model.

Upload a .neqsim process file → introspect → ask questions → run what-if scenarios.
"""
import streamlit as st
import pandas as pd
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
        help="Upload a NeqSim process model file (.neqsim) or a DEXPI P&ID XML file",
    )

    if uploaded_file is not None:
        # Only reload if file changed
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("_loaded_file_key") != file_key:
            file_bytes = uploaded_file.read()

            # Detect DEXPI Proteus XML by checking for <PlantModel> root
            _is_dexpi = False
            if uploaded_file.name.lower().endswith(".xml"):
                try:
                    _snippet = file_bytes[:2000].decode("utf-8", errors="ignore")
                    if "<PlantModel" in _snippet or "PlantInformation" in _snippet:
                        _is_dexpi = True
                except Exception:
                    pass

            if _is_dexpi:
                # DEXPI P&ID file — store XML for chat analysis
                st.session_state["dexpi_xml"] = file_bytes
                st.session_state["dexpi_filename"] = uploaded_file.name
                st.session_state["_loaded_file_key"] = file_key
                # Reset chat session to inject DEXPI context
                st.session_state.pop("chat_session", None)
                st.session_state["chat_messages"] = []
                # Keep any existing process_model — DEXPI & .neqsim coexist
                if st.session_state.get("process_model") is None:
                    st.session_state["_builder_mode"] = True
                st.session_state["_pending_question"] = "Analyze the DEXPI P&ID and summarize the equipment, piping, and instrumentation."
                st.success(f"✓ DEXPI P&ID loaded: {uploaded_file.name}")
                st.rerun()
            else:
                # Standard .neqsim model file
                with st.spinner("Loading process model..."):
                    try:
                        from process_chat.process_model import NeqSimProcessModel

                        model = NeqSimProcessModel.from_bytes(file_bytes, uploaded_file.name)
                        st.session_state["process_model"] = model
                        st.session_state["process_model_bytes"] = file_bytes
                        st.session_state["process_model_name"] = uploaded_file.name
                        st.session_state["_loaded_file_key"] = file_key
                        # Keep DEXPI state — .neqsim & DEXPI coexist
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
            if st.session_state.get("dexpi_xml"):
                st.success(f"✓ DEXPI: {uploaded_file.name}")
            else:
                st.success(f"✓ {uploaded_file.name}")

    # --- Start New Process button ---
    if st.button("🔨 Start New Process", use_container_width=True,
                 help="Build a process from scratch by describing it in chat"):
        # Enter builder mode — clear any stale state
        st.session_state.pop("process_model", None)
        st.session_state.pop("process_model_bytes", None)
        st.session_state.pop("process_model_name", None)
        st.session_state.pop("_loaded_file_key", None)
        st.session_state.pop("dexpi_xml", None)
        st.session_state.pop("dexpi_filename", None)
        st.session_state.pop("chat_session", None)
        st.session_state["chat_messages"] = []
        st.session_state["_builder_mode"] = True
        st.rerun()

    # --- Load Test Process button ---
    _TEST_PROCESS_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test_process.neqsim",
    )
    if os.path.exists(_TEST_PROCESS_PATH):
        if st.button("📂 Load Test Process", use_container_width=True,
                     help="Load a sample gas processing model for testing"):
            with st.spinner("Loading test process model..."):
                try:
                    from process_chat.process_model import NeqSimProcessModel

                    with open(_TEST_PROCESS_PATH, "rb") as f:
                        file_bytes = f.read()
                    model = NeqSimProcessModel.from_bytes(file_bytes, "test_process.neqsim")
                    st.session_state["process_model"] = model
                    st.session_state["process_model_bytes"] = file_bytes
                    st.session_state["process_model_name"] = "test_process.neqsim"
                    st.session_state["_loaded_file_key"] = "test_process_builtin"
                    st.session_state.pop("_builder_mode", None)
                    st.session_state.pop("chat_session", None)
                    st.session_state["chat_messages"] = []
                    st.success("✓ Test process loaded")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load test process: {str(e)}")
        st.caption(
            "[View process description on Colab](https://colab.research.google.com/github/EvenSol/NeqSim-Colab/blob/master/notebooks/process/comparesimulations.ipynb)"
        )

    # --- Load Test DEXPI P&ID button ---
    _TEST_DEXPI_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test_dexpi_pid.xml",
    )
    if os.path.exists(_TEST_DEXPI_PATH):
        if st.button("📐 Load Test DEXPI P&ID", use_container_width=True,
                     help="Load a sample DEXPI P&ID (gas processing unit) for testing"):
            with open(_TEST_DEXPI_PATH, "rb") as f:
                dexpi_bytes = f.read()
            st.session_state["dexpi_xml"] = dexpi_bytes
            st.session_state["dexpi_filename"] = "test_dexpi_pid.xml"
            st.session_state["_loaded_file_key"] = "test_dexpi_builtin"
            # Keep any existing process_model — DEXPI & .neqsim coexist
            st.session_state.pop("chat_session", None)
            st.session_state["chat_messages"] = []
            if st.session_state.get("process_model") is None:
                st.session_state["_builder_mode"] = True
            st.session_state["_pending_question"] = "Analyze the DEXPI P&ID and summarize the equipment, piping, and instrumentation."
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
        st.text_input("Gemini API Key", type="password", key="gemini_api_key",
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

    # --- Lab Data Upload ---
    if st.session_state.get("process_model") is not None:
        st.subheader("🧪 Lab Data Import")
        lab_file = st.file_uploader(
            "Upload CSV/JSON composition",
            type=["csv", "json"],
            key="lab_file_upload",
            help="Upload lab/LIMS data with component names and mole fractions",
        )
        if lab_file is not None:
            lab_text = lab_file.read().decode("utf-8")
            ext = lab_file.name.rsplit(".", 1)[-1].lower()
            if ext == "csv":
                prompt = f"Import this lab CSV data and preview the composition changes:\n```csv\n{lab_text}\n```"
            else:
                prompt = f"Import this lab JSON data and preview the composition changes:\n```json\n{lab_text}\n```"
            st.session_state["_pending_question"] = prompt
            st.session_state["_lab_upload_data"] = lab_text
            st.session_state["_lab_upload_format"] = ext
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
            "What happens if we increase feed flow by 10%?",
            "Find maximum production for this process",
            "Show the risk matrix for this process",
            "Calculate the CO₂ emissions for this process",
            "Run a blowdown simulation on the separator",
            "Sweep the inlet temperature from 20 to 60°C",
            "Run a flow assurance assessment (hydrates, corrosion)",
            "Size the relief valves for all vessels",
            "Run a CME PVT experiment on the feed stream",
            "Run pinch analysis / heat integration study",
            "Turndown analysis — what is the operating envelope?",
            "Run a debottleneck study",
            "Run energy audit / utility balance",
            "Analyse flare sources and recovery options",
            "Seasonal planning — summer vs winter performance",
            "Generate operator training upset scenarios",
            "What is the weather at Stavanger? Impact on coolers?",
            "Update feed composition: 85% methane, 7% ethane, 3% propane, 2% CO2, 3% N2",
            "Show me the Python script",
        ]

        with st.expander("📚 Available Capabilities", expanded=False):
            st.markdown("""
**Analysis & Simulation:**
- **What-if scenarios** — change pressures, temperatures, flows
- **Sensitivity analysis** — single sweep, tornado, 2D surface
- **Dynamic simulation** — blowdown, transient, startup/shutdown
- **PVT simulation** — CME, CVD, differential liberation

**Equipment & Sizing:**
- **Auto-size equipment** — sizing report, utilization, bottleneck
- **Compressor charts** — performance maps, surge limits
- **Optimization** — find max throughput / bottleneck

**Safety & Environmental:**
- **Risk analysis** — criticality ranking, Monte Carlo availability
- **Safety analysis** — PSV sizing, relief scenarios (API 520)
- **Emissions analysis** — CO₂, fugitives, emission intensity
- **Flare analysis** — flare sources, recovery options, carbon tax

**Plant Operations:**
- **Energy integration** — pinch analysis, composite curves, heat recovery
- **Turndown analysis** — operating envelope, min/max stable flow
- **Performance monitoring** — actual vs predicted, degradation alerts
- **Debottleneck study** — upgrade options, cost-effectiveness ranking
- **Energy audit** — utility balance, benchmarking, fuel cost
- **Training scenarios** — upset simulations with quiz Q&A
- **Seasonal planning** — multi-period production comparison

**External Data Integration:**
- **Weather API** — live ambient conditions, 7-day forecast, cooler impact
- **Lab/LIMS import** — update feed composition from lab data (CSV/JSON/inline)

**Build & Export:**
- **Build process** — create from natural language description
- **Python script** — generate NeqSim Python code
- **Save/download** — export .neqsim file
- **Custom code** — run arbitrary NeqSim Python calculations
- **DEXPI P&ID** — import and analyze engineering diagrams

📖 **[Full Documentation →](Process_Chat_Help)**
            """)
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
    st.info("👆 Upload a `.neqsim` process model or a **DEXPI P&ID XML** file in the sidebar, click **Load Test Process** to try a sample, or click **Start New Process** to build one from scratch.")
    st.markdown("""
    ### Getting Started
    
    **Option 1: Upload an existing model**
    Upload a `.neqsim` file to analyze, query, and run what-if scenarios.
    
    **Option 1b: Upload a DEXPI P&ID**
    Upload a DEXPI Proteus XML (`.xml`) file to extract equipment, piping,
    instrumentation, and connectivity. The chat can analyze the P&ID topology
    and optionally import it into a NeqSim simulation model.
    
    **Option 2: Load the test process**
    Click **📂 Load Test Process** in the sidebar to load a pre-built gas processing model.
    The process is described in this [Colab notebook](https://colab.research.google.com/github/EvenSol/NeqSim-Colab/blob/master/notebooks/process/comparesimulations.ipynb).

    **Option 3: Build from scratch**
    Click **🔨 Start New Process** in the sidebar, then describe the process you want to build:
    - *"Build a gas compression process with methane and ethane at 50 bara"*
    - *"Create a 3-stage compression train with intercooling"*
    - *"Build a separation and dehydration process"*
    
    The AI will design the process, run the simulation, and you can then:
    - Ask what-if questions
    - **Find maximum production** (optimize feed flow)
    - **Run risk analysis** (equipment criticality, risk matrix, availability)
    - **Calculate emissions** (CO₂, CH₄, emission intensity)
    - **Flow assurance** (hydrate, wax, corrosion assessment)
    - **Dynamic simulation** (blowdown, startup/shutdown transients)
    - **Sensitivity analysis** (parameter sweeps, tornado charts)
    - **PVT experiments** (CME, differential liberation, separator test)
    - **Safety sizing** (PSV sizing per API 520/521)
    - Download the Python script
    - Save as a `.neqsim` file
    
    ### 💾 Saving & Loading Process Models
    
    You can save your process to a `.neqsim` file for later use:
    
    1. **In chat**: Type *"Save the process"* — a download button will appear in the sidebar.
    2. **From Python**: Use `neqsim.save_neqsim(process, "my_process.neqsim")` to save, and `neqsim.load_neqsim("my_process.neqsim")` to reload.
    3. **Reload**: Upload the saved `.neqsim` file via the sidebar file uploader to continue working with it.
    
    The `.neqsim` file contains the full process topology, fluid compositions, and equipment settings — everything needed to reproduce the simulation.
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
                    row = {}
                    if model.is_process_model:
                        row["System"] = u.process_system
                    row["Name"] = u.name
                    row["Type"] = u.unit_type
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
                    row = {}
                    if model.is_process_model:
                        row["System"] = s.process_system
                    row["Name"] = s.name
                    row["T (°C)"] = f"{s.temperature_C:.1f}" if s.temperature_C is not None else "—"
                    row["P (bara)"] = f"{s.pressure_bara:.2f}" if s.pressure_bara is not None else "—"
                    row["Flow (kg/hr)"] = f"{s.flow_rate_kg_hr:.1f}" if s.flow_rate_kg_hr is not None else "—"
                    stream_data.append(row)
                st.dataframe(pd.DataFrame(stream_data), use_container_width=True, hide_index=True)
            else:
                st.info("No streams found.")
elif builder_mode:
    st.info("🔨 **Builder mode** — Describe the process you want to create in the chat below.")

# DEXPI P&ID status banner (shown regardless of model state)
if st.session_state.get("dexpi_xml"):
    _dexpi_fname = st.session_state.get("dexpi_filename", "DEXPI file")
    _dexpi_info = f"📐 **DEXPI P&ID loaded:** {_dexpi_fname}"
    if model is not None:
        _dexpi_info += "  ·  Combined with NeqSim process model"
    _dexpi_info += '  —  Ask *"analyze the P&ID"* to run full analysis.'
    st.info(_dexpi_info)

    # ─────────────────────────────────────────────
    # DEXPI P&ID Viewer
    # ─────────────────────────────────────────────
    with st.expander("📐 DEXPI P&ID Viewer", expanded=False):
        try:
            from process_chat.dexpi_integration import parse_dexpi_xml
            _dexpi_pid = parse_dexpi_xml(st.session_state["dexpi_xml"])

            _title = _dexpi_pid.title or _dexpi_pid.drawing_number or _dexpi_fname
            st.markdown(f"**{_title}**")
            if _dexpi_pid.drawing_number or _dexpi_pid.revision:
                st.caption(f"Drawing: {_dexpi_pid.drawing_number}  Rev: {_dexpi_pid.revision}  Schema: {_dexpi_pid.schema_version}")

            # Summary metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Equipment", len(_dexpi_pid.equipment))
            c2.metric("Piping Lines", len(_dexpi_pid.piping))
            c3.metric("Instruments", len(_dexpi_pid.instruments))
            c4.metric("Connections", _dexpi_pid.connection_count)

            # Equipment table
            if _dexpi_pid.equipment:
                st.markdown("**Equipment:**")
                _eq_rows = []
                for eq in _dexpi_pid.equipment:
                    _attrs = []
                    for k, v in list(eq.attributes.items())[:3]:
                        if v:
                            _attrs.append(f"{k}: {v}")
                    _eq_rows.append({
                        "Tag": eq.tag_name,
                        "Type": eq.component_class,
                        "Nozzles": len(eq.nozzles),
                        "Attributes": "; ".join(_attrs) if _attrs else "—",
                    })
                st.dataframe(pd.DataFrame(_eq_rows), use_container_width=True, hide_index=True)

            # Piping table
            if _dexpi_pid.piping:
                st.markdown("**Piping Networks:**")
                _pip_rows = []
                for p in _dexpi_pid.piping:
                    _pip_rows.append({
                        "Line": p.line_number,
                        "Fluid": p.fluid_code,
                        "Diameter": p.nominal_diameter,
                        "Class": p.piping_class,
                        "Segments": p.segments,
                        "Valves": len(p.valves),
                    })
                st.dataframe(pd.DataFrame(_pip_rows), use_container_width=True, hide_index=True)

            # Connectivity graph (from piping connections)
            if _dexpi_pid.piping:
                _connections = []
                for p in _dexpi_pid.piping:
                    for conn in p.connections:
                        _from = conn.get("from", "")
                        _to = conn.get("to", "")
                        if _from or _to:
                            _connections.append(f"  {_from or '?'} → **{p.line_number}** ({p.fluid_code}) → {_to or '?'}")
                if _connections:
                    st.markdown("**Connectivity:**")
                    st.markdown("\n".join(_connections))

            # Instrumentation
            if _dexpi_pid.instruments:
                st.markdown("**Instrumentation:**")
                _inst_rows = []
                for inst in _dexpi_pid.instruments:
                    _inst_rows.append({
                        "Tag": inst.tag,
                        "Class": inst.component_class,
                        "Type": inst.function_type or "—",
                    })
                st.dataframe(pd.DataFrame(_inst_rows), use_container_width=True, hide_index=True)

            # Raw XML viewer
            with st.expander("📄 View raw XML", expanded=False):
                _xml_text = st.session_state["dexpi_xml"].decode("utf-8", errors="replace")
                st.code(_xml_text[:50000], language="xml")

        except Exception as e:
            st.warning(f"Could not parse DEXPI for preview: {e}")
            # Fall back to raw XML
            _xml_text = st.session_state["dexpi_xml"].decode("utf-8", errors="replace")
            st.code(_xml_text[:50000], language="xml")

# ─────────────────────────────────────────────
# Process Flow Diagram (PFD)
# ─────────────────────────────────────────────
if model is not None:
    with st.expander("📐 Process Flow Diagram", expanded=True):
        try:
            pfd_style = st.session_state.get("diagram_style", "HYSYS")
            pfd_detail = st.session_state.get("diagram_detail", "ENGINEERING")
            show_values = pfd_detail != "CONCEPTUAL"

            if model.is_process_model:
                # ProcessModel: show each ProcessSystem in its own tab
                dot_pairs = model.get_diagram_dots(
                    style=pfd_style,
                    detail_level=pfd_detail,
                    show_stream_values=show_values,
                )
                # Also add a combined view
                combined_dot = model.get_diagram_dot(
                    style=pfd_style,
                    detail_level=pfd_detail,
                    show_stream_values=show_values,
                )
                tab_names = [name or f"System {i+1}" for i, (name, _) in enumerate(dot_pairs)]
                tab_names.insert(0, "🗂️ Combined")
                tabs = st.tabs(tab_names)
                with tabs[0]:
                    st.graphviz_chart(combined_dot, use_container_width=True)
                for i, (ps_name, dot_src) in enumerate(dot_pairs):
                    with tabs[i + 1]:
                        st.graphviz_chart(dot_src, use_container_width=True)
            else:
                # Single ProcessSystem
                dot_source = model.get_diagram_dot(
                    style=pfd_style,
                    detail_level=pfd_detail,
                    show_stream_values=show_values,
                )
                st.graphviz_chart(dot_source, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render process flow diagram: {e}")


# ─────────────────────────────────────────────
# Chat Interface
# ─────────────────────────────────────────────

def _show_comparison(comparison):
    """Display scenario comparison results inline."""
    from process_chat.scenario_engine import results_summary_table

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


def _show_optimization(opt_result):
    """Display optimization results inline with visual utilization bars."""
    st.markdown("---")
    st.markdown("**🎯 Process Optimization Result**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Original Flow",
            f"{opt_result.original_flow_kg_hr:,.0f} kg/hr",
        )
    with col2:
        st.metric(
            "Optimal Flow",
            f"{opt_result.optimal_flow_kg_hr:,.0f} kg/hr",
            delta=f"{opt_result.max_increase_pct:+.1f}%",
        )
    with col3:
        st.metric(
            "Bottleneck",
            opt_result.bottleneck_equipment or "None",
            delta=f"{opt_result.bottleneck_utilization*100:.0f}% util" if opt_result.bottleneck_utilization is not None else None,
            delta_color="inverse",
        )

    # Utilization breakdown
    if opt_result.utilization_breakdown:
        st.markdown("**Equipment Utilization at Optimum:**")
        util_data = []
        for u in sorted(opt_result.utilization_breakdown, key=lambda x: x.utilization, reverse=True):
            util_data.append({
                "Equipment": u.name,
                "Type": u.equipment_type,
                "Utilization %": round(u.utilization * 100, 1),
                "Constraint": u.constraint_name,
                "Detail": u.detail,
            })
        st.dataframe(pd.DataFrame(util_data), use_container_width=True, hide_index=True)

    # Search iteration chart
    if opt_result.iterations and len(opt_result.iterations) > 1:
        with st.expander("📈 Optimization Search History", expanded=False):
            iter_data = []
            for it in opt_result.iterations:
                iter_data.append({
                    "Iteration": it.iteration,
                    "Flow (kg/hr)": round(it.flow_rate_kg_hr, 0),
                    "Max Utilization %": round(it.max_utilization * 100, 1),
                    "Feasible": "✓" if it.feasible else "✗",
                    "Bottleneck": it.bottleneck,
                })
            st.dataframe(pd.DataFrame(iter_data), use_container_width=True, hide_index=True)

    st.markdown(f"**Algorithm:** {opt_result.search_algorithm}")
    if not opt_result.converged:
        st.warning(f"Optimization did not fully converge: {opt_result.message}")


def _show_risk_analysis(risk_result):
    """Display risk analysis results inline with risk matrix, criticality, and Monte Carlo."""
    st.markdown("---")
    st.markdown("**⚠️ Risk Analysis Result**")

    # Top metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "System Availability",
            f"{risk_result.system_availability_pct:.2f}%",
        )
    with col2:
        st.metric(
            "Most Critical Equipment",
            risk_result.most_critical_equipment or "None",
        )
    with col3:
        mc = risk_result.monte_carlo
        if mc:
            st.metric(
                "Expected Production",
                f"{mc.expected_production_pct:.1f}%",
                delta=f"P90: {mc.p90_production_pct:.1f}%",
                delta_color="off",
            )

    # Risk matrix table
    if risk_result.risk_matrix:
        st.markdown("**Risk Matrix:**")
        risk_data = []
        for ri in sorted(risk_result.risk_matrix, key=lambda x: x.risk_score, reverse=True):
            level_icon = {
                "LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠",
                "VERY_HIGH": "🔴", "EXTREME": "⛔",
            }.get(ri.risk_level.value, "⚪")
            risk_data.append({
                "Equipment": ri.equipment_name,
                "Probability": ri.probability.value,
                "Consequence": ri.consequence.value,
                "Score": ri.risk_score,
                "Risk Level": f"{level_icon} {ri.risk_level.value}",
                "Failure Rate (/yr)": round(ri.failure_rate_per_year, 2),
                "Production Loss %": round(ri.production_loss_pct, 1),
            })
        st.dataframe(pd.DataFrame(risk_data), use_container_width=True, hide_index=True)

    # Equipment criticality
    trip_impacts = [fi for fi in risk_result.failure_impacts if fi.failure_type == "TRIP"]
    if trip_impacts:
        with st.expander("🔍 Equipment Criticality (Trip Impact)", expanded=False):
            crit_data = []
            for fi in sorted(trip_impacts, key=lambda x: x.criticality_index, reverse=True):
                crit_data.append({
                    "Equipment": fi.equipment_name,
                    "Type": fi.equipment_type,
                    "Production Loss %": round(fi.production_loss_pct, 1),
                    "Criticality Index": round(fi.criticality_index, 2),
                    "Failed Flow (kg/hr)": round(fi.failed_production_kg_hr, 0),
                })
            st.dataframe(pd.DataFrame(crit_data), use_container_width=True, hide_index=True)

    # Degraded impacts
    deg_impacts = [fi for fi in risk_result.failure_impacts if fi.failure_type == "DEGRADED"]
    if deg_impacts:
        with st.expander("📉 Degraded Operation Impact", expanded=False):
            deg_data = []
            for fi in sorted(deg_impacts, key=lambda x: x.production_loss_pct, reverse=True):
                deg_data.append({
                    "Equipment": fi.equipment_name,
                    "Type": fi.equipment_type,
                    "Production Loss %": round(fi.production_loss_pct, 1),
                    "Failed Flow (kg/hr)": round(fi.failed_production_kg_hr, 0),
                })
            st.dataframe(pd.DataFrame(deg_data), use_container_width=True, hide_index=True)

    # Equipment reliability
    if risk_result.equipment_reliability:
        with st.expander("📊 Equipment Reliability (OREDA)", expanded=False):
            rel_data = []
            for r in risk_result.equipment_reliability:
                rel_data.append({
                    "Equipment": r.name,
                    "Type": r.equipment_type,
                    "MTTF (hours)": round(r.mttf_hours, 0),
                    "MTTR (hours)": round(r.mttr_hours, 0),
                    "Failure Rate (/yr)": round(r.failure_rate_per_year, 2),
                    "Availability %": round(r.availability * 100, 2),
                })
            st.dataframe(pd.DataFrame(rel_data), use_container_width=True, hide_index=True)

    # Monte Carlo details
    mc = risk_result.monte_carlo
    if mc:
        with st.expander("🎲 Monte Carlo Simulation Details", expanded=False):
            mc_col1, mc_col2 = st.columns(2)
            with mc_col1:
                st.markdown("**Simulation Parameters**")
                st.write(f"- Iterations: {mc.iterations:,d}")
                st.write(f"- Horizon: {mc.horizon_days} days")
                st.write(f"- Expected Availability: {mc.expected_availability_pct:.1f}%")
            with mc_col2:
                st.markdown("**Production Statistics**")
                st.write(f"- P10: {mc.p10_production_pct:.1f}%")
                st.write(f"- P50: {mc.p50_production_pct:.1f}%")
                st.write(f"- P90: {mc.p90_production_pct:.1f}%")
                st.write(f"- Expected Downtime: {mc.expected_downtime_hours_year:.0f} hrs/year")
                st.write(f"- Expected Events: {mc.expected_failure_events_year:.1f} /year")

            if mc.equipment_downtime_contribution:
                st.markdown("**Downtime Contribution:**")
                dt_data = []
                for name, pct in sorted(mc.equipment_downtime_contribution.items(),
                                        key=lambda x: x[1], reverse=True):
                    if pct > 0.1:
                        dt_data.append({"Equipment": name, "Contribution %": round(pct, 1)})
                if dt_data:
                    st.dataframe(pd.DataFrame(dt_data), use_container_width=True, hide_index=True)


def _show_compressor_chart(chart_result):
    """Display compressor chart results inline with Plotly performance map."""
    import plotly.graph_objects as go

    st.markdown("---")
    st.markdown("**📈 Compressor Performance Chart**")

    import uuid as _uuid

    for chart_data in chart_result.charts:
        st.markdown(f"**{chart_data.compressor_name}** — Template: {chart_data.template_used}")

        fig = go.Figure()

        # Speed curves
        for sc in chart_data.speed_curves:
            if sc.flow_m3_hr and sc.head_kJ_kg:
                fig.add_trace(go.Scatter(
                    x=sc.flow_m3_hr,
                    y=sc.head_kJ_kg,
                    mode="lines",
                    name=f"{sc.speed_rpm:.0f} RPM",
                    line=dict(width=2),
                ))

        # Surge curve
        if chart_data.surge_flow and chart_data.surge_head:
            fig.add_trace(go.Scatter(
                x=chart_data.surge_flow,
                y=chart_data.surge_head,
                mode="lines",
                name="Surge Line",
                line=dict(color="red", width=3, dash="dash"),
            ))

        # Stonewall curve
        if chart_data.stonewall_flow and chart_data.stonewall_head:
            fig.add_trace(go.Scatter(
                x=chart_data.stonewall_flow,
                y=chart_data.stonewall_head,
                mode="lines",
                name="Stonewall",
                line=dict(color="orange", width=3, dash="dot"),
            ))

        # Operating point
        op = chart_data.operating_point
        if op and op.flow_m3_hr is not None and op.flow_m3_hr > 0 and op.head_kJ_kg is not None and op.head_kJ_kg > 0:
            fig.add_trace(go.Scatter(
                x=[op.flow_m3_hr],
                y=[op.head_kJ_kg],
                mode="markers+text",
                name="Operating Point",
                marker=dict(size=12, color="green", symbol="star"),
                text=[f"OP: {op.flow_m3_hr:.0f}, {op.head_kJ_kg:.0f}"],
                textposition="top right",
            ))

        fig.update_layout(
            title=f"Compressor Map — {chart_data.compressor_name}",
            xaxis_title="Actual Volume Flow (m³/hr)",
            yaxis_title="Polytropic Head (kJ/kg)",
            hovermode="closest",
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"comp_chart_{chart_data.compressor_name}_{_uuid.uuid4().hex[:8]}")

        # Operating point details
        if op:
            op_cols = st.columns(4)
            with op_cols[0]:
                st.metric("Flow", f"{op.flow_m3_hr:.0f} m³/hr" if op.flow_m3_hr else "N/A")
            with op_cols[1]:
                st.metric("Head", f"{op.head_kJ_kg:.1f} kJ/kg" if op.head_kJ_kg else "N/A")
            with op_cols[2]:
                st.metric("Speed", f"{op.speed_rpm:.0f} RPM" if op.speed_rpm else "N/A")
            with op_cols[3]:
                st.metric("Efficiency", f"{op.efficiency_pct:.1f}%" if op.efficiency_pct else "N/A")

            if op.distance_to_surge is not None:
                st.info(f"Surge margin: {op.distance_to_surge*100:.1f}%")
            if op.distance_to_stonewall is not None:
                st.info(f"Stonewall margin: {op.distance_to_stonewall*100:.1f}%")


def _show_auto_size(autosize_result):
    """Display auto-size results inline with sizing data and utilization."""
    st.markdown("---")
    st.markdown("**📐 Auto-Size & Utilization Report**")

    # Top metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Equipment Sized",
            f"{autosize_result.sized_count}/{autosize_result.total_equipment}",
        )
    with col2:
        st.metric(
            "Bottleneck",
            autosize_result.bottleneck_name or "None detected",
        )
    with col3:
        if autosize_result.bottleneck_utilization_pct > 0:
            st.metric(
                "Bottleneck Utilization",
                f"{autosize_result.bottleneck_utilization_pct:.1f}%",
                delta=autosize_result.bottleneck_constraint,
                delta_color="inverse",
            )

    # Utilization breakdown
    if autosize_result.utilization:
        st.markdown("**Equipment Utilization:**")
        util_data = []
        for u in sorted(autosize_result.utilization, key=lambda x: x.utilization_pct, reverse=True):
            marker = "★" if u.is_bottleneck else ""
            util_data.append({
                "": marker,
                "Equipment": u.name,
                "Type": u.equipment_type,
                "Utilization %": round(u.utilization_pct, 1),
                "Constraint": u.constraint_name,
                "Detail": u.detail,
            })
        st.dataframe(pd.DataFrame(util_data), use_container_width=True, hide_index=True)

    # Sizing details
    sized_items = [s for s in autosize_result.equipment_sized if s.auto_sized and s.sizing_data]
    if sized_items:
        with st.expander("📋 Equipment Sizing Details", expanded=False):
            for si in sized_items:
                st.markdown(f"**{si.name}** ({si.equipment_type})")
                sizing_display = {k: v for k, v in si.sizing_data.items() if k != "sizing_report"}
                if sizing_display:
                    sizing_df = pd.DataFrame([sizing_display])
                    st.dataframe(sizing_df, use_container_width=True, hide_index=True)
                # JSON report
                if "sizing_report" in si.sizing_data:
                    with st.expander(f"Full sizing report — {si.name}", expanded=False):
                        st.json(si.sizing_data["sizing_report"])


def _show_emissions(emissions_result):
    """Display emissions analysis results inline."""
    st.markdown("---")
    st.markdown("**🏭 Emissions Analysis**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total CO₂", f"{emissions_result.total_co2_kg_hr:,.1f} kg/hr")
    with col2:
        st.metric("Total CO₂e", f"{emissions_result.total_co2e_kg_hr:,.1f} kg/hr")
    with col3:
        if emissions_result.emission_intensity_kg_per_tonne is not None:
            st.metric("Emission Intensity", f"{emissions_result.emission_intensity_kg_per_tonne:,.2f} kg CO₂e/tonne")
        else:
            st.metric("Emission Intensity", "N/A")

    if emissions_result.sources:
        st.markdown("**Emission Sources:**")
        src_data = []
        for s in sorted(emissions_result.sources, key=lambda x: x.co2_kg_hr, reverse=True):
            src_data.append({
                "Source": s.name,
                "Category": s.source_type,
                "CO₂ (kg/hr)": round(s.co2_kg_hr, 2),
                "CH₄ (kg/hr)": round(s.ch4_kg_hr, 4),
                "CO₂e (kg/hr)": round(s.co2e_kg_hr, 2),
            })
        st.dataframe(pd.DataFrame(src_data), use_container_width=True, hide_index=True)

    if getattr(emissions_result, "message", ""):
        st.info(emissions_result.message)


def _show_dynamic(dynamic_result):
    """Display dynamic simulation results inline with time-series charts."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.markdown("---")
    st.markdown(f"**⏱️ Dynamic Simulation — {dynamic_result.scenario_type}**")

    # --- Top-level metrics ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Duration", f"{dynamic_result.duration_s:.0f} s")
    with col2:
        st.metric("Time Steps", f"{len(dynamic_result.time_series)}")
    with col3:
        st.metric("Method", getattr(dynamic_result, "method", "—"))

    ts = dynamic_result.time_series
    var_names = getattr(dynamic_result, "variable_names", []) or []
    var_units = getattr(dynamic_result, "variable_units", {}) or {}
    available_vars = [v for v in var_names if v not in ("time_frac",)]

    if not ts or not available_vars:
        if getattr(dynamic_result, "message", ""):
            st.info(dynamic_result.message)
        return

    # --- Initial → Final summary ---
    first_vals = ts[0].values if ts else {}
    last_vals = ts[-1].values if ts else {}
    changed_vars = []
    for v in available_vars:
        v0 = first_vals.get(v)
        vf = last_vals.get(v)
        if v0 is not None and vf is not None:
            try:
                if abs(float(v0) - float(vf)) > 1e-6:
                    changed_vars.append(v)
            except (ValueError, TypeError):
                pass

    if changed_vars:
        with st.expander("📊 Initial → Final State", expanded=True):
            delta_rows = []
            for v in changed_vars:
                v0 = first_vals.get(v, 0)
                vf = last_vals.get(v, 0)
                unit = var_units.get(v, "")
                # Friendly label: "V-100.liquid_level_m" → "V-100 liquid level"
                label = v.replace("_", " ").replace(".", " — ")
                try:
                    pct = ((float(vf) - float(v0)) / abs(float(v0))) * 100.0 if float(v0) != 0 else 0.0
                except (ValueError, TypeError, ZeroDivisionError):
                    pct = 0.0
                delta_rows.append({
                    "Variable": label,
                    "Initial": round(float(v0), 4),
                    "Final": round(float(vf), 4),
                    "Change %": round(pct, 1),
                    "Unit": unit,
                })
            st.dataframe(pd.DataFrame(delta_rows), use_container_width=True, hide_index=True)

    # --- Variable selector ---
    # Group variables by equipment/category for better UX
    # "V-100.liquid_level_m" → category "V-100"
    categories: dict = {}
    for v in available_vars:
        parts = v.split(".", 1)
        cat = parts[0] if len(parts) > 1 else "General"
        categories.setdefault(cat, []).append(v)

    # Smart defaults: pick variables that actually changed, up to 6
    default_selection = changed_vars[:6] if changed_vars else available_vars[:4]

    selected_vars = st.multiselect(
        "Select variables to plot",
        options=available_vars,
        default=default_selection,
        key=f"dynamic_var_select_{id(dynamic_result)}",
        help="Choose which variables to trace over time",
    )

    if not selected_vars:
        selected_vars = available_vars[:2]

    # --- Chart: one subplot per unit group for selected vars ---
    # Group selected vars by their unit (for axis scaling)
    unit_groups: dict = {}  # unit_str -> [var_name, ...]
    for v in selected_vars:
        u = var_units.get(v, "")
        unit_groups.setdefault(u, []).append(v)

    n_axes = min(len(unit_groups), 3)  # max 3 y-axes
    if n_axes <= 1:
        # Simple single-axis chart
        fig = go.Figure()
        for v in selected_vars:
            unit = var_units.get(v, "")
            label = v.replace("_", " ")
            if unit:
                label += f" ({unit})"
            fig.add_trace(go.Scatter(
                x=[p.time_s for p in ts],
                y=[p.values.get(v, 0) for p in ts],
                name=label,
                mode="lines+markers" if len(ts) <= 30 else "lines",
            ))
        fig.update_layout(
            title=f"{dynamic_result.scenario_type.capitalize()} Simulation",
            xaxis_title="Time (s)",
            hovermode="x unified",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Multi-axis: subplots stacked vertically, one per unit group
        group_keys = list(unit_groups.keys())[:3]
        fig = make_subplots(
            rows=len(group_keys), cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=[grp or "[-]" for grp in group_keys],
        )
        for row_idx, grp in enumerate(group_keys, 1):
            for v in unit_groups[grp]:
                label = v.replace("_", " ")
                fig.add_trace(
                    go.Scatter(
                        x=[p.time_s for p in ts],
                        y=[p.values.get(v, 0) for p in ts],
                        name=label,
                        mode="lines+markers" if len(ts) <= 30 else "lines",
                    ),
                    row=row_idx, col=1,
                )
            fig.update_yaxes(title_text=grp or "[-]", row=row_idx, col=1)
        fig.update_xaxes(title_text="Time (s)", row=len(group_keys), col=1)
        fig.update_layout(
            title=f"{dynamic_result.scenario_type.capitalize()} Simulation",
            hovermode="x unified",
            height=300 * len(group_keys),
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Data table ---
    with st.expander("📋 Time-Series Data", expanded=False):
        table_rows = []
        for p in ts:
            row = {"Time (s)": p.time_s}
            for v in selected_vars:
                label = v.replace("_", " ")
                unit = var_units.get(v, "")
                col_name = f"{label} ({unit})" if unit else label
                row[col_name] = round(p.values.get(v, 0), 4)
            table_rows.append(row)
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    if getattr(dynamic_result, "message", ""):
        st.info(dynamic_result.message)


def _show_sensitivity(sensitivity_result):
    """Display sensitivity analysis results inline with charts."""
    import plotly.graph_objects as go

    st.markdown("---")
    st.markdown(f"**📊 Sensitivity Analysis — {sensitivity_result.analysis_type}**")

    if sensitivity_result.analysis_type == "tornado":
        # Tornado chart
        if sensitivity_result.tornado_bars:
            def _swing(b):
                return abs(b.kpi_at_high - b.kpi_at_low)

            bars = sorted(sensitivity_result.tornado_bars, key=_swing)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=[b.variable for b in bars],
                x=[b.kpi_at_high - b.kpi_base for b in bars],
                base=[b.kpi_base for b in bars],
                name="High",
                orientation="h",
                marker_color="indianred",
            ))
            fig.add_trace(go.Bar(
                y=[b.variable for b in bars],
                x=[b.kpi_at_low - b.kpi_base for b in bars],
                base=[b.kpi_base for b in bars],
                name="Low",
                orientation="h",
                marker_color="steelblue",
            ))
            # Use response_kpis from the result if available
            kpi_label = ""
            if sensitivity_result.response_kpis:
                kpi_label = sensitivity_result.response_kpis[0]
            fig.update_layout(
                title="Tornado Chart",
                xaxis_title=kpi_label or "KPI",
                barmode="overlay",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Data table
            tornado_data = []
            for b in sorted(sensitivity_result.tornado_bars, key=_swing, reverse=True):
                tornado_data.append({
                    "Variable": b.variable,
                    "Input Low": round(b.low_value, 4),
                    "Input High": round(b.high_value, 4),
                    "KPI at Low": round(b.kpi_at_low, 4),
                    "KPI at Base": round(b.kpi_base, 4),
                    "KPI at High": round(b.kpi_at_high, 4),
                    "Swing": round(_swing(b), 4),
                })
            st.dataframe(pd.DataFrame(tornado_data), use_container_width=True, hide_index=True)

    elif sensitivity_result.analysis_type == "two_variable":
        if sensitivity_result.sweep_points:
            pts = sensitivity_result.sweep_points
            # Extract consistent key names from the first point with 2+ inputs
            key_names = []
            for p in pts:
                if len(p.input_values) >= 2:
                    key_names = list(p.input_values.keys())[:2]
                    break
            if len(key_names) == 2:
                x_vals = sorted(set(p.input_values.get(key_names[0], 0) for p in pts if p.input_values))
                y_vals = sorted(set(p.input_values.get(key_names[1], 0) for p in pts if len(p.input_values) > 1))

                if x_vals and y_vals:
                    # Build surface for first output value
                    first_kpi = list(pts[0].output_values.keys())[0] if pts[0].output_values else "KPI"
                    z_map = {}
                    for p in pts:
                        if len(p.input_values) >= 2:
                            z_map[(p.input_values[key_names[0]], p.input_values[key_names[1]])] = p.output_values.get(first_kpi, 0)
                    z = [[z_map.get((x, y), 0) for x in x_vals] for y in y_vals]

                    fig = go.Figure(data=go.Heatmap(x=x_vals, y=y_vals, z=z, colorscale="Viridis"))
                    fig.update_layout(
                        title=f"Sensitivity Surface \u2014 {first_kpi}",
                        xaxis_title=key_names[0],
                        yaxis_title=key_names[1],
                    )
                    st.plotly_chart(fig, use_container_width=True)

    else:  # single_sweep
        if sensitivity_result.sweep_points:
            # Filter to feasible points that have output data
            pts = [p for p in sensitivity_result.sweep_points if p.feasible and p.output_values]
            if not pts:
                st.info("No feasible sweep points to display.")
            else:
                var_name = sensitivity_result.sweep_variable or (
                    list(pts[0].input_values.keys())[0] if pts[0].input_values else "Input"
                )
                x_data = [p.input_values.get(var_name, 0) for p in pts]

                # Skip noise keywords and report.* duplicates (these exist
                # as direct KPIs with proper units already)
                _SKIP = {'mechdesign', 'sizing', 'composition', 'weight fraction',
                         'mole fraction', 'maxdesign', 'maxoperating', 'json.',
                         'molar_volume', 'molar_mass_kg', 'jointefficiency',
                         'tensilestrength', 'maxallowablestress', 'mindesign',
                         'report.'}

                # Collect KPI series, filtering noise and constants
                all_kpi_keys = sorted({k for p in pts for k in p.output_values})
                filtered = {}
                for k in all_kpi_keys:
                    kl = k.lower()
                    if any(s in kl for s in _SKIP):
                        continue
                    ys = [p.output_values.get(k, 0) for p in pts]
                    rng = max(ys) - min(ys)
                    if rng < 1e-9:
                        continue  # constant across sweep
                    if all(abs(v) < 1e-12 for v in ys):
                        continue  # all zero
                    filtered[k] = ys

                if not filtered:
                    st.info("All KPI values are constant across the sweep range.")
                else:
                    # Group traces by response_kpi keywords → one chart per keyword
                    response_kpis = sensitivity_result.response_kpis or []
                    used_keys = set()

                    for resp_kpi in response_kpis:
                        rkl = resp_kpi.lower()
                        matching = {k: ys for k, ys in filtered.items()
                                    if rkl in k.lower() and k not in used_keys}
                        if not matching:
                            continue
                        # Sort by name length (shorter = more primary), limit 5
                        sorted_keys = sorted(matching, key=len)[:5]
                        used_keys.update(sorted_keys)
                        fig = go.Figure()
                        for mk in sorted_keys:
                            fig.add_trace(go.Scatter(
                                x=x_data, y=matching[mk],
                                name=mk, mode="lines+markers"))
                        fig.update_layout(
                            title=f"Sweep — {resp_kpi}",
                            xaxis_title=var_name,
                            yaxis_title=resp_kpi,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Show remaining (unmatched) KPIs if any
                    remaining = {k: ys for k, ys in filtered.items() if k not in used_keys}
                    if remaining:
                        sorted_rem = sorted(remaining, key=len)[:8]
                        fig = go.Figure()
                        for k in sorted_rem:
                            fig.add_trace(go.Scatter(
                                x=x_data, y=remaining[k],
                                name=k, mode="lines+markers"))
                        fig.update_layout(
                            title="Sweep — Other KPIs",
                            xaxis_title=var_name,
                            yaxis_title="Value",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Data table
                    table_data = {var_name: x_data}
                    for k, ys in filtered.items():
                        table_data[k] = [round(v, 4) for v in ys]
                    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)


def _show_pvt(pvt_result):
    """Display PVT simulation results inline."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.markdown("---")
    exp_name = getattr(pvt_result, "experiment_type", "") or "PVT"
    st.markdown(f"**🧪 PVT Simulation — {exp_name}**")

    col1, col2 = st.columns(2)
    with col1:
        if pvt_result.saturation_pressure_bara:
            st.metric("Saturation Pressure", f"{pvt_result.saturation_pressure_bara:.2f} bara")
    with col2:
        temp_c = getattr(pvt_result, "saturation_temperature_C", None)
        if temp_c is not None and pvt_result.saturation_pressure_bara:
            st.metric("Temperature", f"{temp_c:.1f} °C")

    if pvt_result.data_points:
        pts = pvt_result.data_points

        # --- Experiment-specific plots ---
        exp_upper = exp_name.upper().replace(" ", "")

        if exp_upper in ("CME", "DIFFERENTIALLIBERATION"):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[p.pressure_bara for p in pts],
                y=[p.values.get("relative_volume", 0) for p in pts],
                name="Relative Volume",
                mode="lines+markers",
            ))
            fig.update_layout(
                title=exp_name,
                xaxis_title="Pressure (bara)",
                yaxis_title="Relative Volume",
            )
            st.plotly_chart(fig, use_container_width=True)

        elif exp_upper == "CVD":
            # Two-axis plot: liquid dropout + cumulative gas produced
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(
                x=[p.pressure_bara for p in pts],
                y=[p.values.get("liquid_dropout_pct", 0) for p in pts],
                name="Liquid Dropout (%)",
                mode="lines+markers",
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=[p.pressure_bara for p in pts],
                y=[p.values.get("cumulative_gas_produced_mol_pct", 0) for p in pts],
                name="Cum. Gas Produced (mol%)",
                mode="lines+markers",
                line=dict(dash="dash"),
            ), secondary_y=True)
            fig.update_layout(title="Constant Volume Depletion")
            fig.update_xaxes(title_text="Pressure (bara)")
            fig.update_yaxes(title_text="Liquid Dropout (%)", secondary_y=False)
            fig.update_yaxes(title_text="Cum. Gas Produced (mol%)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

            # Second chart: gas Z-factor
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=[p.pressure_bara for p in pts],
                y=[p.values.get("gas_Z_factor", 0) for p in pts],
                name="Gas Z-factor",
                mode="lines+markers",
            ))
            fig2.update_layout(
                title="Gas Z-factor",
                xaxis_title="Pressure (bara)",
                yaxis_title="Z",
            )
            st.plotly_chart(fig2, use_container_width=True)

        elif exp_upper == "SEPARATORTEST":
            fig = go.Figure()
            stages = [p.values.get("stage", i + 1) for i, p in enumerate(pts)]
            fig.add_trace(go.Bar(
                x=[f"Stage {int(s)}" for s in stages],
                y=[p.values.get("GOR_Sm3_Sm3", 0) for p in pts],
                name="GOR (Sm³/Sm³)",
            ))
            fig.update_layout(
                title="Separator Test — GOR per Stage",
                xaxis_title="Stage",
                yaxis_title="GOR (Sm³/Sm³)",
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            # Generic fallback: plot the first numeric value column found
            fig = go.Figure()
            col_names = getattr(pvt_result, "column_names", [])
            plotted = False
            for cname in col_names:
                yvals = [p.values.get(cname) for p in pts]
                if any(v is not None and v != 0 for v in yvals):
                    fig.add_trace(go.Scatter(
                        x=[p.pressure_bara for p in pts],
                        y=[v or 0 for v in yvals],
                        name=cname.replace("_", " "),
                        mode="lines+markers",
                    ))
                    plotted = True
            if plotted:
                fig.update_layout(
                    title=exp_name,
                    xaxis_title="Pressure (bara)",
                )
                st.plotly_chart(fig, use_container_width=True)

        # --- Generic data table using column_names or all available keys ---
        col_names = getattr(pvt_result, "column_names", None)
        col_units = getattr(pvt_result, "column_units", {}) or {}
        pvt_data = []
        for p in pts:
            row = {"Pressure (bara)": round(p.pressure_bara, 2)}
            if p.temperature_C is not None:
                row["Temperature (°C)"] = round(p.temperature_C, 2)
            keys = col_names if col_names else sorted(p.values.keys())
            for k in keys:
                v = p.values.get(k)
                if v is not None:
                    unit = col_units.get(k, "")
                    label = k.replace("_", " ")
                    if unit:
                        label = f"{label} ({unit})"
                    row[label] = round(v, 4) if isinstance(v, float) else v
            pvt_data.append(row)
        st.dataframe(pd.DataFrame(pvt_data), use_container_width=True, hide_index=True)


def _show_safety(safety_result):
    """Display safety/PSV sizing analysis results inline."""
    st.markdown("---")
    st.markdown("**🛡️ Safety & Relief Systems Analysis**")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total PSV Count", f"{safety_result.total_psv_count}")
    with col2:
        st.metric("Max Relief Rate", f"{safety_result.max_relief_rate_kg_hr:,.0f} kg/hr")

    # Show controlling (highest relief rate) scenario if available
    if safety_result.scenarios:
        ctrl = max(safety_result.scenarios, key=lambda s: s.required_relief_rate_kg_hr)
        st.info(f"**Controlling Scenario:** {ctrl.scenario} on {ctrl.equipment_name} — "
                f"Relief rate {ctrl.required_relief_rate_kg_hr:,.0f} kg/hr, "
                f"Required orifice {ctrl.api_orifice_letter}")

    if safety_result.scenarios:
        st.markdown("**Relief Scenarios:**")
        scen_data = []
        for s in safety_result.scenarios:
            scen_data.append({
                "Equipment": s.equipment_name,
                "Scenario": s.scenario,
                "Set Pressure (bara)": round(s.set_pressure_bara, 1),
                "Relief Rate (kg/hr)": round(s.required_relief_rate_kg_hr, 0),
                "Required Area (mm²)": round(s.required_orifice_area_mm2, 1),
                "Orifice": s.api_orifice_letter,
                "Phase": s.fluid_phase,
            })
        st.dataframe(pd.DataFrame(scen_data), use_container_width=True, hide_index=True)


def _show_flow_assurance(fa_result):
    """Display flow assurance assessment results inline."""
    st.markdown("---")
    st.markdown("**🌊 Flow Assurance Assessment**")

    overall_icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(fa_result.overall_risk, "⚪")
    st.markdown(f"**Overall Risk Level:** {overall_icon} {fa_result.overall_risk}")

    # Hydrate risks
    if fa_result.hydrate_risks:
        st.markdown("**Hydrate Risks:**")
        hyd_data = []
        for h in fa_result.hydrate_risks:
            risk_icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(h.risk_level, "⚪")
            hyd_data.append({
                "Stream": h.location,
                "Risk": f"{risk_icon} {h.risk_level}",
                "Hydrate T (°C)": round(h.hydrate_T_C, 1) if h.hydrate_T_C is not None else "N/A",
                "Stream T (°C)": round(h.operating_T_C, 1) if h.operating_T_C is not None else "N/A",
                "Subcooling (°C)": round(h.subcooling_C, 1) if h.subcooling_C is not None else "N/A",
                "Inhibitor": h.inhibitor_type or "—",
                "Inhibitor Rate (kg/hr)": round(h.inhibitor_rate_kg_hr, 1) if h.inhibitor_rate_kg_hr else "—",
            })
        st.dataframe(pd.DataFrame(hyd_data), use_container_width=True, hide_index=True)

    # Corrosion risks
    if fa_result.corrosion_risks:
        st.markdown("**Corrosion Risks:**")
        cor_data = []
        for c in fa_result.corrosion_risks:
            risk_icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(c.risk_level, "⚪")
            cor_data.append({
                "Location": c.location,
                "Risk": f"{risk_icon} {c.risk_level}",
                "Corr. Rate (mm/yr)": round(c.corrosion_rate_mm_yr, 2) if c.corrosion_rate_mm_yr else "—",
                "Mechanism": c.mechanism or "—",
            })
        st.dataframe(pd.DataFrame(cor_data), use_container_width=True, hide_index=True)

    # Wax risks
    if fa_result.wax_risks:
        st.markdown("**Wax Risks:**")
        wax_data = []
        for w in fa_result.wax_risks:
            risk_icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(w.risk_level, "⚪")
            wax_data.append({
                "Stream": w.stream_name,
                "Risk": f"{risk_icon} {w.risk_level}",
                "WAT (°C)": round(w.wax_appearance_T_C, 1),
                "Stream T (°C)": round(w.operating_T_C, 1),
                "Margin (°C)": round(w.margin_C, 1),
            })
        st.dataframe(pd.DataFrame(wax_data), use_container_width=True, hide_index=True)

    if fa_result.recommendations:
        with st.expander("💡 Recommendations", expanded=False):
            for rec in fa_result.recommendations:
                st.markdown(f"- {rec}")


def _show_energy_integration(ei_result):
    """Display energy integration / pinch analysis results inline."""
    import plotly.graph_objects as go

    st.markdown("---")
    st.markdown("**🔥 Energy Integration / Pinch Analysis**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Pinch Temperature", f"{ei_result.pinch_temperature_C:.1f} °C")
    with col2:
        st.metric("Min Hot Utility", f"{ei_result.min_hot_utility_kW:,.0f} kW")
    with col3:
        st.metric("Min Cold Utility", f"{ei_result.min_cold_utility_kW:,.0f} kW")

    if ei_result.max_heat_recovery_kW > 0:
        st.metric("Max Heat Recovery", f"{ei_result.max_heat_recovery_kW:,.0f} kW")

    # Composite curves
    if ei_result.hot_composite or ei_result.cold_composite:
        fig = go.Figure()
        if ei_result.hot_composite:
            fig.add_trace(go.Scatter(
                x=[p.duty_kW for p in ei_result.hot_composite],
                y=[p.temperature_C for p in ei_result.hot_composite],
                name="Hot Composite", line=dict(color="red"),
            ))
        if ei_result.cold_composite:
            fig.add_trace(go.Scatter(
                x=[p.duty_kW for p in ei_result.cold_composite],
                y=[p.temperature_C for p in ei_result.cold_composite],
                name="Cold Composite", line=dict(color="blue"),
            ))
        fig.update_layout(
            title="Composite Curves",
            xaxis_title="Duty (kW)",
            yaxis_title="Temperature (°C)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Heat streams table
    if ei_result.hot_streams or ei_result.cold_streams:
        with st.expander("🔍 Heat Stream Details", expanded=False):
            all_streams = []
            for h in ei_result.hot_streams:
                all_streams.append({
                    "Name": h.name, "Type": "HOT",
                    "Tin (°C)": round(h.t_in_C, 1),
                    "Tout (°C)": round(h.t_out_C, 1),
                    "Duty (kW)": round(h.duty_kW, 0),
                })
            for c in ei_result.cold_streams:
                all_streams.append({
                    "Name": c.name, "Type": "COLD",
                    "Tin (°C)": round(c.t_in_C, 1),
                    "Tout (°C)": round(c.t_out_C, 1),
                    "Duty (kW)": round(c.duty_kW, 0),
                })
            st.dataframe(pd.DataFrame(all_streams), use_container_width=True, hide_index=True)

    # Suggestions
    if ei_result.suggestions:
        with st.expander("💡 Heat Recovery Suggestions", expanded=False):
            for s in ei_result.suggestions:
                st.markdown(f"- **{s.hot_stream}** → **{s.cold_stream}**: "
                           f"{s.recoverable_kW:.0f} kW ({s.detail})")


def _show_turndown(td_result):
    """Display turndown / operating envelope results inline."""
    import plotly.graph_objects as go

    st.markdown("---")
    st.markdown("**📉 Turndown / Operating Envelope**")

    col1, col2, col3 = st.columns(3)
    if td_result.min_stable:
        with col1:
            st.metric("Min Stable Flow",
                      f"{td_result.min_stable.flow_pct:.0f}%",
                      delta=f"Limit: {td_result.min_stable.limiting_equipment}")
    if td_result.max_capacity:
        with col2:
            st.metric("Max Capacity",
                      f"{td_result.max_capacity.flow_pct:.0f}%",
                      delta=f"Limit: {td_result.max_capacity.limiting_equipment}")
    with col3:
        st.metric("Design Flow", f"{td_result.design_flow_kg_hr:,.0f} kg/hr")

    # Envelope chart
    if td_result.envelope:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[p.flow_pct for p in td_result.envelope],
            y=[p.max_utilization for p in td_result.envelope],
            name="Max Utilization", line=dict(color="blue"),
        ))
        fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                      annotation_text="Equipment Limit")
        fig.update_layout(
            title="Operating Envelope",
            xaxis_title="Flow (% of design)",
            yaxis_title="Max Equipment Utilization",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Data table
        with st.expander("📊 Envelope Data", expanded=False):
            env_data = [{
                "Flow (%)": p.flow_pct,
                "Product (kg/hr)": round(p.product_rate_kg_hr, 0),
                "Power (kW)": round(p.total_power_kW, 0),
                "Max Util": round(p.max_utilization, 3),
                "Limiting Equipment": p.limiting_equipment,
            } for p in td_result.envelope]
            st.dataframe(pd.DataFrame(env_data), use_container_width=True, hide_index=True)


def _show_performance_monitor(pm_result):
    """Display performance monitoring results inline."""
    st.markdown("---")
    st.markdown("**📊 Performance Monitoring**")

    # Overall health
    health_icon = {"HEALTHY": "🟢", "DEGRADED": "🟡", "CRITICAL": "🔴"}.get(
        pm_result.overall_health, "⚪")
    st.markdown(f"**Overall Health:** {health_icon} {pm_result.overall_health}")

    # Alerts
    if pm_result.alerts:
        st.markdown("**Degradation Alerts:**")
        alert_data = []
        for a in pm_result.alerts:
            icon = {"NORMAL": "🟢", "WARNING": "🟡", "ALARM": "🔴"}.get(a.severity, "⚪")
            alert_data.append({
                "Equipment": a.equipment,
                "Measurement": a.measurement,
                "Status": f"{icon} {a.severity}",
                "Actual": round(a.actual_value, 2),
                "Predicted": round(a.predicted_value, 2),
                "Residual": round(a.residual, 2),
                "Diagnosis": a.diagnosis,
            })
        st.dataframe(pd.DataFrame(alert_data), use_container_width=True, hide_index=True)

    # Details
    if pm_result.measurements:
        with st.expander("🔍 All Measurements", expanded=False):
            m_data = [{
                "Path": m.path,
                "Actual": round(m.actual_value, 2),
                "Predicted": round(m.predicted_value, 2),
                "Residual": round(m.residual, 2),
                "Unit": m.unit,
                "Status": m.status,
            } for m in pm_result.measurements]
            st.dataframe(pd.DataFrame(m_data), use_container_width=True, hide_index=True)


def _show_debottleneck(db_result):
    """Display debottleneck study results inline."""
    st.markdown("---")
    st.markdown("**🔧 Debottleneck Study**")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Production", f"{db_result.current_production_kg_hr:,.0f} kg/hr")
    with col2:
        if db_result.potential_increase_pct > 0:
            st.metric("Potential Increase", f"{db_result.potential_increase_pct:.0f}%")

    # Bottleneck equipment
    if db_result.bottlenecks:
        st.markdown("**High-Utilization Equipment:**")
        bn_data = [{
            "Equipment": b.name,
            "Type": b.equipment_type,
            "Utilization": f"{b.utilization_pct:.0f}%",
            "Limiting Parameter": b.limiting_parameter,
        } for b in db_result.bottlenecks]
        st.dataframe(pd.DataFrame(bn_data), use_container_width=True, hide_index=True)

    # Upgrade options
    if db_result.upgrade_options:
        st.markdown("**Upgrade Options (ranked by cost-effectiveness):**")
        up_data = [{
            "Equipment": u.equipment,
            "Strategy": u.strategy,
            "Capacity Gain": f"{u.capacity_gain_pct:.0f}%",
            "Cost (USD)": f"${u.estimated_cost_usd:,.0f}",
            "Extra Flow (kg/hr)": f"{u.extra_throughput_kg_hr:,.0f}",
            "Cost-Eff (t/yr/$MM)": f"{u.cost_effectiveness:,.0f}",
        } for u in db_result.upgrade_options]
        st.dataframe(pd.DataFrame(up_data), use_container_width=True, hide_index=True)


def _show_training(tr_result):
    """Display training scenario results inline."""
    st.markdown("---")
    st.markdown("**🎓 Operator Training Scenarios**")

    for scenario in tr_result.scenarios:
        severity_icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "🔴"}.get(
            scenario.severity, "⚪")

        with st.expander(f"{severity_icon} {scenario.name} — {scenario.severity}", expanded=False):
            st.markdown(f"**Description:** {scenario.description}")
            st.markdown(f"**Recommended Response:** {scenario.recommended_response}")

            if scenario.impacts:
                st.markdown("**Impacts:**")
                imp_data = [{
                    "KPI": i.kpi_name,
                    "Before": round(i.base_value, 2),
                    "After": round(i.upset_value, 2),
                    "Change (%)": round(i.change_pct, 1),
                } for i in scenario.impacts]
                st.dataframe(pd.DataFrame(imp_data), use_container_width=True, hide_index=True)

            if scenario.recovery_actions:
                st.markdown("**Recovery Actions:**")
                for idx, action in enumerate(scenario.recovery_actions, 1):
                    st.markdown(f"  {idx}. {action}")

            if scenario.quiz_question:
                st.markdown(f"**Quiz:** {scenario.quiz_question}")
                st.markdown(f"**Answer:** ||{scenario.quiz_answer}||")


def _show_energy_audit(ea_result):
    """Display energy audit results inline."""
    import plotly.graph_objects as go

    st.markdown("---")
    st.markdown("**⚡ Energy Audit / Utility Balance**")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Power", f"{ea_result.total_power_kW:,.0f} kW")
    with col2:
        st.metric("Total Cooling", f"{ea_result.total_cooling_kW:,.0f} kW")
    with col3:
        st.metric("Total Heating", f"{ea_result.total_heating_kW:,.0f} kW")
    with col4:
        st.metric("Specific Energy", f"{ea_result.specific_energy_kWh_per_tonne:.1f} kWh/t")

    # Pie chart of energy consumers
    if ea_result.consumers:
        power_consumers = [c for c in ea_result.consumers if c.energy_type == "POWER" and c.consumption_kW > 0]
        if power_consumers:
            fig = go.Figure(data=[go.Pie(
                labels=[c.name for c in power_consumers],
                values=[c.consumption_kW for c in power_consumers],
                hole=0.3,
            )])
            fig.update_layout(title="Power Consumption Breakdown")
            st.plotly_chart(fig, use_container_width=True)

    # Benchmarks
    if ea_result.benchmarks:
        st.markdown("**Benchmark Comparison:**")
        bm_data = [{
            "Metric": b.metric,
            "Actual": round(b.actual_value, 1),
            "Benchmark": round(b.benchmark_value, 1),
            "Unit": b.unit,
            "Status": {"GOOD": "🟢 Good", "NORMAL": "🟡 Normal", "POOR": "🔴 Poor"}.get(b.status, b.status),
        } for b in ea_result.benchmarks]
        st.dataframe(pd.DataFrame(bm_data), use_container_width=True, hide_index=True)

    # Suggestions
    if ea_result.suggestions:
        with st.expander("💡 Improvement Suggestions", expanded=False):
            for s in ea_result.suggestions:
                st.markdown(f"- **{s.equipment}**: {s.suggestion} "
                           f"(~{s.potential_saving_kW:.0f} kW, {s.potential_saving_pct:.0f}%)")


def _show_flare_analysis(fl_result):
    """Display flare analysis results inline."""
    st.markdown("---")
    st.markdown("**🔥 Flare Minimization Analysis**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Flare Rate", f"{fl_result.total_flare_rate_kg_hr:,.0f} kg/hr")
    with col2:
        st.metric("CO₂ Equivalent", f"{fl_result.total_co2_equiv_tonnes_yr:,.0f} t/yr")
    with col3:
        st.metric("Carbon Tax Exposure", f"${fl_result.carbon_tax_exposure_usd_yr:,.0f}/yr")

    # Flare sources
    if fl_result.sources:
        st.markdown("**Flare Sources:**")
        src_data = [{
            "Source": s.name,
            "Type": s.source_type,
            "Flow (kg/hr)": round(s.flow_rate_kg_hr, 0),
            "CO₂e (t/yr)": round(s.co2_equiv_tonnes_yr, 0),
            "Detail": s.detail,
        } for s in fl_result.sources]
        st.dataframe(pd.DataFrame(src_data), use_container_width=True, hide_index=True)

    # Recovery options
    if fl_result.recovery_options:
        st.markdown("**Recovery Options:**")
        opt_data = [{
            "Option": o.name,
            "Recovery": f"{o.recovery_pct:.0f}%",
            "CAPEX": f"${o.capex_usd:,.0f}",
            "Revenue ($/yr)": f"${o.revenue_usd_yr:,.0f}",
            "Payback (yr)": round(o.payback_years, 1),
            "CO₂ Reduction (t/yr)": round(o.co2_reduction_tonnes_yr, 0),
        } for o in fl_result.recovery_options]
        st.dataframe(pd.DataFrame(opt_data), use_container_width=True, hide_index=True)

    if fl_result.best_option:
        st.success(f"**Recommended:** {fl_result.best_option}")


def _show_multi_period(mp_result):
    """Display multi-period / seasonal planning results inline."""
    import plotly.graph_objects as go

    st.markdown("---")
    st.markdown("**📅 Multi-Period / Seasonal Planning**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Production", f"{mp_result.total_production_tonnes_yr:,.0f} t/yr")
    with col2:
        st.metric("Total Energy", f"{mp_result.total_energy_MWh_yr:,.0f} MWh/yr")
    with col3:
        st.metric("Avg Specific Energy", f"{mp_result.avg_specific_energy:.1f} kWh/t")

    if mp_result.best_scenario:
        st.markdown(f"**Best Efficiency:** {mp_result.best_scenario} | "
                   f"**Worst Efficiency:** {mp_result.worst_scenario}")

    # Comparison chart
    if mp_result.scenarios:
        names = [s.name for s in mp_result.scenarios]
        fig = go.Figure(data=[
            go.Bar(name="Product (t/period)", x=names,
                   y=[s.production_tonnes_period for s in mp_result.scenarios]),
        ])
        fig.update_layout(title="Production by Scenario")
        st.plotly_chart(fig, use_container_width=True)

        # Data table
        sc_data = [{
            "Scenario": s.name,
            "Product (kg/hr)": round(s.product_rate_kg_hr, 0),
            "Power (kW)": round(s.total_power_kW, 0),
            "Cooling (kW)": round(s.total_cooling_kW, 0),
            "Spec. Energy (kWh/t)": round(s.specific_energy_kWh_tonne, 1),
            "CO₂ (t/yr)": round(s.co2_equiv_tonnes_yr, 0),
        } for s in mp_result.scenarios]
        st.dataframe(pd.DataFrame(sc_data), use_container_width=True, hide_index=True)

        # Warnings
        for s in mp_result.scenarios:
            if s.warnings:
                for w in s.warnings:
                    st.warning(f"{s.name}: {w}")


def _show_weather(wx_result):
    """Display weather analysis results inline."""
    import plotly.graph_objects as go

    st.markdown("---")
    st.markdown(f"**🌤️ Weather — {wx_result.location_name}**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Temperature", f"{wx_result.current.temperature_C:.1f}°C")
    with col2:
        st.metric("Humidity", f"{wx_result.current.relative_humidity_pct:.0f}%")
    with col3:
        st.metric("Wind", f"{wx_result.current.wind_speed_m_s:.1f} m/s")

    # Cooler impact
    if wx_result.cooler_impact:
        ci = wx_result.cooler_impact
        status_color = {"OK": "🟢", "WARNING": "🟡", "CRITICAL": "🔴"}.get(ci.status, "⚪")
        st.markdown(f"{status_color} **Cooler Status:** {ci.status} — "
                    f"Capacity {ci.capacity_factor:.0%} "
                    f"(design {ci.design_ambient_C:.0f}°C, actual {ci.current_ambient_C:.1f}°C, "
                    f"Δ{ci.delta_C:+.1f}°C)")

    # 7-day forecast chart
    if wx_result.forecast:
        dates = [d.date for d in wx_result.forecast]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=[d.temp_max_C for d in wx_result.forecast],
                                 name="Max", mode="lines+markers", line=dict(color="red")))
        fig.add_trace(go.Scatter(x=dates, y=[d.temp_avg_C for d in wx_result.forecast],
                                 name="Avg", mode="lines+markers", line=dict(color="orange")))
        fig.add_trace(go.Scatter(x=dates, y=[d.temp_min_C for d in wx_result.forecast],
                                 name="Min", mode="lines+markers", line=dict(color="blue")))
        if wx_result.cooler_impact:
            fig.add_hline(y=wx_result.cooler_impact.design_ambient_C,
                          line_dash="dash", line_color="gray",
                          annotation_text="Design Basis")
        fig.update_layout(title="7-Day Temperature Forecast",
                          xaxis_title="Date", yaxis_title="Temperature (°C)")
        st.plotly_chart(fig, use_container_width=True)

    # Recommendations
    if wx_result.recommendations:
        with st.expander("Recommendations", expanded=True):
            for r in wx_result.recommendations:
                st.markdown(f"• {r}")


def _show_report(report_data):
    """Display a JSON report (full process, module, unit, or stream) inline."""
    st.markdown("---")

    # Determine scope label
    if isinstance(report_data, dict):
        scope = "Process Report"
        if "equipmentName" in report_data:
            scope = f"Unit Report — {report_data['equipmentName']}"
        elif "streamName" in report_data:
            scope = f"Stream Report — {report_data['streamName']}"
        elif "name" in report_data:
            scope = f"Report — {report_data['name']}"
        # Check for multi-system (module) structure: keys like "system_name/unit"
        top_keys = list(report_data.keys())
        has_slash = any("/" in k for k in top_keys[:20])
    else:
        scope = "Report"
        has_slash = False

    with st.expander(f"📋 {scope}", expanded=False):
        if isinstance(report_data, dict) and has_slash:
            # Multi-module report — group by module prefix
            modules: dict = {}
            for k, v in report_data.items():
                if "/" in k:
                    mod, rest = k.split("/", 1)
                    modules.setdefault(mod, {})[rest] = v
                else:
                    modules.setdefault("_root", {})[k] = v
            for mod_name, mod_data in modules.items():
                label = mod_name if mod_name != "_root" else "General"
                st.subheader(label)
                st.json(mod_data)
        else:
            st.json(report_data)


def _show_lab_import(lab_result):
    """Display lab composition import results inline."""
    st.markdown("---")
    st.markdown(f"**🧪 Lab Composition Import — {lab_result.sample.sample_id}**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Components", len(lab_result.sample.components))
    with col2:
        st.metric("Stream", lab_result.stream_name)
    with col3:
        status = "✅ Applied" if lab_result.applied else "👁️ Preview"
        st.metric("Status", status)

    # Composition table
    if lab_result.composition_df is not None and not lab_result.composition_df.empty:
        st.dataframe(lab_result.composition_df, use_container_width=True, hide_index=True)

    # Warnings
    for w in lab_result.warnings:
        st.warning(w)

    # Unmapped components
    if lab_result.sample.unmapped:
        st.info(f"Unmapped components (used as-is): {', '.join(lab_result.sample.unmapped)}")


def _show_dexpi(dexpi_result):
    """Display DEXPI P&ID analysis results inline."""
    st.markdown("---")
    pid = dexpi_result.pid_summary
    title = pid.title or pid.drawing_number or "DEXPI P&ID"
    st.markdown(f"**📐 DEXPI P&ID Analysis — {title}**")

    if pid.drawing_number or pid.revision:
        st.caption(f"Drawing: {pid.drawing_number}  Rev: {pid.revision}  Schema: {pid.schema_version}")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Equipment", len(pid.equipment))
    with col2:
        st.metric("Piping Lines", len(pid.piping))
    with col3:
        st.metric("Instruments", len(pid.instruments))
    with col4:
        st.metric("Connections", pid.connection_count)

    # Equipment table
    if pid.equipment:
        st.markdown("**Equipment:**")
        eq_data = []
        for eq in pid.equipment:
            eq_data.append({
                "Tag": eq.tag_name,
                "Type": eq.component_class,
                "Nozzles": len(eq.nozzles),
            })
        st.dataframe(pd.DataFrame(eq_data), use_container_width=True, hide_index=True)

    # Equipment type counts
    if dexpi_result.equipment_type_counts:
        with st.expander("Equipment Type Counts", expanded=False):
            for cls, count in sorted(dexpi_result.equipment_type_counts.items()):
                st.markdown(f"- **{cls}**: {count}")

    # Piping table
    if pid.piping:
        st.markdown("**Piping Networks:**")
        pip_data = []
        for p in pid.piping:
            pip_data.append({
                "Line": p.line_number,
                "Fluid": p.fluid_code,
                "Diameter": p.nominal_diameter,
                "Class": p.piping_class,
                "Segments": p.segments,
                "Valves": len(p.valves),
            })
        st.dataframe(pd.DataFrame(pip_data), use_container_width=True, hide_index=True)

    # Instrumentation
    if pid.instruments:
        with st.expander(f"Instrumentation ({len(pid.instruments)} items)", expanded=False):
            inst_data = []
            for inst in pid.instruments:
                inst_data.append({
                    "Tag": inst.tag,
                    "Class": inst.component_class,
                    "Type": inst.function_type or "—",
                })
            st.dataframe(pd.DataFrame(inst_data), use_container_width=True, hide_index=True)

    # NeqSim import status
    if dexpi_result.neqsim_model_loaded:
        st.success(f"✅ NeqSim model imported: {dexpi_result.neqsim_units} units, {dexpi_result.neqsim_streams} streams")

        # DEXPI export / download button
        if dexpi_result.neqsim_model is not None:
            try:
                from process_chat.dexpi_integration import export_to_dexpi
                dexpi_bytes = export_to_dexpi(dexpi_result.neqsim_model)
                if dexpi_bytes:
                    st.download_button(
                        "📥 Download as DEXPI XML",
                        data=dexpi_bytes,
                        file_name="neqsim_export.xml",
                        mime="application/xml",
                    )
            except Exception:
                pass

    # Warnings
    for w in dexpi_result.warnings:
        st.warning(w)


def _show_neqsim_code(code_result):
    """Display NeqSim code execution results inline."""
    import plotly.graph_objects as go

    st.markdown("---")
    st.markdown("**🐍 NeqSim Code Execution**")

    # Show the code
    code = code_result.get("code", "")
    if code:
        with st.expander("View executed code", expanded=False):
            st.code(code, language="python")

    # Show retry info
    attempts = code_result.get("attempts", [])
    if len(attempts) > 1:
        successes = [a for a in attempts if not a.get("error")]
        failures = [a for a in attempts if a.get("error")]
        if successes:
            st.info(f"✅ Succeeded on attempt {successes[0].get('attempt', '?')}/{len(attempts)} "
                     f"({len(failures)} auto-fix retries)")
        with st.expander(f"Retry history ({len(attempts)} attempts)", expanded=False):
            for a in attempts:
                attempt_num = a.get("attempt", "?")
                if a.get("error"):
                    st.error(f"Attempt {attempt_num}: {a['error']}")
                else:
                    st.success(f"Attempt {attempt_num}: Success")

    # Show error if final result failed
    if code_result.get("error"):
        st.error(f"Execution error: {code_result['error']}")

    # Show stdout
    stdout = code_result.get("stdout", "")
    if stdout:
        st.text(stdout[:10000])

    # Show tables (DataFrames)
    tables = code_result.get("tables", [])
    for tbl in tables:
        name = tbl.get("name", "Result")
        df = tbl.get("dataframe")
        if df is not None:
            st.markdown(f"**{name}**")
            st.dataframe(df, use_container_width=True)
        else:
            csv = tbl.get("csv", "")
            if csv:
                st.markdown(f"**{name}**")
                st.text(csv[:5000])

    # Show figures
    figures = code_result.get("figures", [])
    for fig_info in figures:
        fig = fig_info.get("figure")
        fig_type = fig_info.get("type", "")
        if fig_type == "plotly" and fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        elif fig_type == "matplotlib" and fig is not None:
            st.pyplot(fig, use_container_width=True)

    # Download button for the code
    if code:
        st.download_button(
            "📥 Download Python script",
            data=code,
            file_name="neqsim_script.py",
            mime="text/x-python",
        )


def _show_model_built(model_result):
    """Display model overview inline in chat after a build or update."""
    st.markdown("---")
    st.markdown("**📊 Process Model Overview**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Unit Operations**")
        if model_result.units:
            st.dataframe(pd.DataFrame(model_result.units), use_container_width=True, hide_index=True)
        else:
            st.info("No unit operations found.")

    with col2:
        st.markdown("**Streams**")
        if model_result.streams:
            st.dataframe(pd.DataFrame(model_result.streams), use_container_width=True, hide_index=True)
        else:
            st.info("No streams found.")

    if model_result.pfd_dot:
        st.markdown("**Process Flow Diagram**")
        try:
            st.graphviz_chart(model_result.pfd_dot, use_container_width=True)
        except Exception:
            pass


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

        # Render attached result objects with error protection
        _RESULT_RENDERERS = [
            ("comparison", _show_comparison),
            ("optimization", _show_optimization),
            ("risk_analysis", _show_risk_analysis),
            ("chart", _show_compressor_chart),
            ("autosize", _show_auto_size),
            ("emissions", _show_emissions),
            ("dynamic", _show_dynamic),
            ("sensitivity", _show_sensitivity),
            ("pvt", _show_pvt),
            ("safety", _show_safety),
            ("flow_assurance", _show_flow_assurance),
            ("energy_integration", _show_energy_integration),
            ("turndown", _show_turndown),
            ("performance_monitor", _show_performance_monitor),
            ("debottleneck", _show_debottleneck),
            ("training", _show_training),
            ("energy_audit", _show_energy_audit),
            ("flare_analysis", _show_flare_analysis),
            ("multi_period", _show_multi_period),
            ("weather", _show_weather),
            ("lab_import", _show_lab_import),
            ("report", _show_report),
            ("dexpi", _show_dexpi),
            ("neqsim_code", _show_neqsim_code),
            ("model_built", _show_model_built),
        ]
        for key, renderer in _RESULT_RENDERERS:
            data = msg.get(key)
            if data is not None:
                try:
                    renderer(data)
                except Exception as e:
                    st.warning(f"Could not render {key} result: {e}")


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
                # Sync API key in case the user changed it
                session.api_key = api_key_val
                # Inject DEXPI XML if available
                if st.session_state.get("dexpi_xml"):
                    session.set_dexpi_xml(
                        st.session_state["dexpi_xml"],
                        st.session_state.get("dexpi_filename", "dexpi.xml"),
                    )
                response = session.chat(user_input)
                comparison = session.get_last_comparison()
                optimization = session.get_last_optimization()
                risk_analysis = session.get_last_risk_analysis()
                chart = session.get_last_chart()
                autosize = session.get_last_autosize()
                emissions = session.get_last_emissions()
                dynamic = session.get_last_dynamic()
                sensitivity = session.get_last_sensitivity()
                pvt = session.get_last_pvt()
                safety = session.get_last_safety()
                flow_assurance = session.get_last_flow_assurance()
                energy_integration = session.get_last_energy_integration()
                turndown = session.get_last_turndown()
                performance_monitor = session.get_last_performance_monitor()
                debottleneck = session.get_last_debottleneck()
                training = session.get_last_training()
                energy_audit = session.get_last_energy_audit()
                flare_analysis = session.get_last_flare_analysis()
                multi_period = session.get_last_multi_period()
                weather = session.get_last_weather()
                lab_import = session.get_last_lab_import()
                report = session.get_last_report()
                dexpi = session.get_last_dexpi()
                neqsim_code = session.get_last_neqsim_code()
                model_built = session.get_last_model_built()

                # --- Sync model from session back to session_state ---
                if session.model is not None:
                    if st.session_state.get("process_model") is not session.model:
                        st.session_state["process_model"] = session.model
                        st.session_state["_builder_mode"] = False
                        st.session_state["process_model_name"] = (
                            st.session_state.get("process_model_name") or "Built Process"
                        )

                # --- Refresh operating points in old chart messages ---
                # After any model change (scenario, build, etc.), old chart
                # widgets in the chat history would show stale operating
                # points.  Refresh them so the history always reflects the
                # current model state.
                if session.model is not None:
                    try:
                        from process_chat.compressor_chart import (
                            refresh_operating_point,
                            CompressorChartResult,
                        )
                        for old_msg in st.session_state["chat_messages"]:
                            old_chart = old_msg.get("chart")
                            if old_chart is not None:
                                refreshed = [
                                    refresh_operating_point(session.model, cd)
                                    for cd in old_chart.charts
                                ]
                                old_msg["chart"] = CompressorChartResult(
                                    charts=refreshed,
                                    message=old_chart.message,
                                )
                    except Exception:
                        pass

                st.markdown(response)

                # Store message and render attached results
                msg_data = {"role": "assistant", "content": response}
                _result_pairs = [
                    ("comparison", comparison, _show_comparison),
                    ("optimization", optimization, _show_optimization),
                    ("risk_analysis", risk_analysis, _show_risk_analysis),
                    ("chart", chart, _show_compressor_chart),
                    ("autosize", autosize, _show_auto_size),
                    ("emissions", emissions, _show_emissions),
                    ("dynamic", dynamic, _show_dynamic),
                    ("sensitivity", sensitivity, _show_sensitivity),
                    ("pvt", pvt, _show_pvt),
                    ("safety", safety, _show_safety),
                    ("flow_assurance", flow_assurance, _show_flow_assurance),
                    ("energy_integration", energy_integration, _show_energy_integration),
                    ("turndown", turndown, _show_turndown),
                    ("performance_monitor", performance_monitor, _show_performance_monitor),
                    ("debottleneck", debottleneck, _show_debottleneck),
                    ("training", training, _show_training),
                    ("energy_audit", energy_audit, _show_energy_audit),
                    ("flare_analysis", flare_analysis, _show_flare_analysis),
                    ("multi_period", multi_period, _show_multi_period),
                    ("weather", weather, _show_weather),
                    ("lab_import", lab_import, _show_lab_import),
                    ("report", report, _show_report),
                    ("dexpi", dexpi, _show_dexpi),
                    ("neqsim_code", neqsim_code, _show_neqsim_code),
                    ("model_built", model_built, _show_model_built),
                ]
                for key, result_obj, renderer in _result_pairs:
                    if result_obj is not None:
                        msg_data[key] = result_obj
                        try:
                            renderer(result_obj)
                        except Exception as e:
                            st.warning(f"Could not render {key}: {e}")

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
        st.session_state.pop("process_model_bytes", None)
        st.session_state.pop("process_model_name", None)
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
