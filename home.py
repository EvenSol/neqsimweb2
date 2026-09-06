"""NeqSim Web front page."""

import streamlit as st

from theme import apply_theme, theme_toggle


GAS_PROCESSING_URL = (
    "https://colab.research.google.com/github/EvenSol/NeqSim-Colab/blob/master/"
    "notebooks/examples_of_NeqSim_in_Colab.ipynb"
)


st.set_page_config(
    page_title="NeqSim",
    page_icon="images/neqsimlogocircleflat.png",
)
apply_theme()
theme_toggle()

st.markdown(
    """
    <style>
        div[data-testid="column"] {
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        div[data-testid="column"] h1 {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        @media (max-width: 640px) {
            .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

col_logo, col_title = st.columns([1, 5], vertical_alignment="center")
with col_logo:
    st.image("images/neqsimlogocircleflat.png", width=90)
with col_title:
    st.markdown("# NeqSim")
    st.caption("Process Simulation Tool")

if st.session_state.get("experimental_mode", False):
    st.write("## Choose your workspace")

    workspace_studio, workspace_classic = st.columns(2)
    with workspace_studio:
        st.markdown("### 🧭 NeqSim Studio")
        st.caption(
            "New professional engineering workspace. Studio is in beta and reuses "
            "the validated NeqSim process-simulation foundation."
        )
        if st.button(
            "Open NeqSim Studio",
            type="primary",
            use_container_width=True,
            key="open_neqsim_studio",
        ):
            st.switch_page("pages/00_NeqSim_Studio.py")

    with workspace_classic:
        st.markdown("### 🧰 NeqSim Classic")
        st.caption(
            "Continue with the existing NeqSim Web pages, calculators and workflows. "
            "Nothing below has moved."
        )
        st.info("You are already in Classic. Use the existing sidebar as before.")

    st.divider()
else:
    st.caption(
        "Stable mode is active. Enable Experimental mode below the sidebar menu "
        "to show developing models and interfaces."
    )

st.write("## Welcome! 👋")

st.markdown(
    f"""
### About NeqSim
NeqSim (Non-equilibrium Simulator) is a library for the simulation of fluid
behavior, phase equilibrium, and process systems. Explore the stable models and
simulations through this easy-to-use Streamlit interface.

### Getting Started
Use the left-hand menu to select a calculation. Normal mode contains TP Flash,
Phase Envelope, Gas Hydrate, Hydrogen, and EOS-CG. Enable **Experimental mode**
below the menu to add the wider set of developing tools.

### Documentation & Tutorials
- [NeqSim Documentation](https://equinor.github.io/neqsim/)
- [Introduction to Gas Processing Using NeqSim]({GAS_PROCESSING_URL})

### GitHub Repository
NeqSim is developed in Java and available as an open-source project:

- [NeqSim Home](https://equinor.github.io/neqsimhome/)

### Community & Feedback
- [NeqSim GitHub Discussions](https://github.com/equinor/neqsim/discussions)

### Request New Features
- Open a feature request in [GitHub Issues](https://github.com/equinor/neqsim/issues)
- Start a discussion in [GitHub Discussions](https://github.com/equinor/neqsim/discussions)

### Extend the App Yourself
This web application is open source and built with Python.

**Quick start:**
1. Clone the repository: `git clone https://github.com/EvenSol/neqsimweb2.git`
2. Create a virtual environment: `python -m venv .venv`
3. Activate it and install dependencies: `pip install -r requirements.txt`
4. Run locally: `streamlit run welcome.py`

**Resources:**
- [NeqSim Web App Repository](https://github.com/EvenSol/neqsimweb2)
- [NeqSim Python Package](https://github.com/equinor/neqsim-python)
- [Streamlit Documentation](https://docs.streamlit.io/)
    """
)

if "ai_enabled" not in st.session_state:
    st.session_state["ai_enabled"] = False

st.sidebar.divider()
st.sidebar.subheader("🤖 AI Assistant")

ai_enabled = st.sidebar.toggle(
    "Enable AI Features",
    value=st.session_state["ai_enabled"],
    help="Enable AI-powered analysis and recommendations",
)
st.session_state["ai_enabled"] = ai_enabled
st.session_state["ai_model"] = "gemini-2.5-flash"

if ai_enabled:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            st.session_state["gemini_api_key"] = st.secrets["GEMINI_API_KEY"]
            st.sidebar.success("✓ AI ready (gemini-2.5-flash)")
        else:
            st.sidebar.warning("No GEMINI_API_KEY in Streamlit secrets.")
    except Exception:
        st.sidebar.warning("No GEMINI_API_KEY in Streamlit secrets.")
