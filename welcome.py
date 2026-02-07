import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from theme import apply_theme, theme_toggle

def get_gemini_api_key():
    """Get Gemini API key from secrets or session state."""
    # First check Streamlit secrets (for deployed app)
    try:
        if 'GEMINI_API_KEY' in st.secrets:
            return st.secrets['GEMINI_API_KEY']
    except Exception:
        pass
    # Fall back to session state (for user-provided key)
    return st.session_state.get('gemini_api_key', '')

def make_request(question_input: str):
    # Only attempt request if AI is enabled and API key is available
    if not st.session_state.get('ai_enabled', False):
        return ""
    api_key = get_gemini_api_key()
    if not api_key or api_key.strip() == "":
        return ""
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=question_input
        )
        return response.text
    except Exception:
        return ""

st.set_page_config(page_title="NeqSim", page_icon='images/neqsimlogocircleflat.png')
apply_theme()
theme_toggle()

# Custom CSS for responsive header layout with vertical centering
st.markdown("""
<style>
    /* Vertically center content in columns */
    div[data-testid="column"] {
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    /* Remove extra spacing around headers */
    div[data-testid="column"] h1 {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    /* Mobile adjustments */
    @media (max-width: 640px) {
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Logo and title with proper vertical alignment
col_logo, col_title = st.columns([1, 5], vertical_alignment="center")
with col_logo:
    st.image('images/neqsimlogocircleflat.png', width=90)
with col_title:
    st.markdown("# NeqSim")
    st.caption("Process Simulation Tool")

st.write("## Welcome! 👋")

"""
### About NeqSim
NeqSim (Non-equilibrium Simulator) is a library for the simulation of fluid behavior, phase equilibrium, and process systems.
Explore the various models and simulations NeqSim offer through this easy-to-use Streamlit interface.

### Getting Started
Use the left-hand menu to select the desired simulation or process. Enter any required inputs, and NeqSim will handle the calculations.

### Documentation & Tutorials
For comprehensive documentation on how to use NeqSim for processing and fluid simulations, please refer to our resources:  
- [NeqSim Documentation](https://equinor.github.io/neqsim/)
- [Introduction to Gas Processing Using NeqSim](https://colab.research.google.com/github/EvenSol/NeqSim-Colab/blob/master/notebooks/examples_of_NeqSim_in_Colab.ipynb)

### GitHub Repository
NeqSim is developed in the Java programming language and is available as an open-source project via GitHub. You can access the complete source code and contribute to the project via the home page:

- [NeqSim Home](https://equinor.github.io/neqsimhome/)

### Community & Feedback
We welcome any feedback, questions, or suggestions for further development. Join the conversation or contribute to discussions on our GitHub page:

- [NeqSim GitHub Discussions](https://github.com/equinor/neqsim/discussions)

### Request New Features
Have an idea for a new simulation or feature? You can:
- Open a feature request in [GitHub Issues](https://github.com/equinor/neqsim/issues)
- Start a discussion in [GitHub Discussions](https://github.com/equinor/neqsim/discussions)

### Extend the App Yourself
This web application is open source and built with Python. To develop and extend it locally:

**Tools needed:**
- Python 3.10+
- Git
- A code editor (e.g., VS Code)

**Quick start:**
1. Clone the repository: `git clone https://github.com/equinor/neqsimweb2.git`
2. Create a virtual environment: `python -m venv .venv`
3. Activate it and install dependencies: `pip install -r requirements.txt`
4. Run locally: `streamlit run welcome.py`

**Resources:**
- [NeqSim Web App Repository](https://github.com/equinor/neqsimweb2)
- [NeqSim Python Package](https://github.com/equinor/neqsim-python)
- [Streamlit Documentation](https://docs.streamlit.io/)
"""

# Initialize AI enabled state (default OFF)
if 'ai_enabled' not in st.session_state:
    st.session_state['ai_enabled'] = False

# AI Settings in sidebar
st.sidebar.divider()
st.sidebar.subheader("🤖 AI Assistant")

# Toggle to enable/disable AI
ai_enabled = st.sidebar.toggle(
    "Enable AI Features",
    value=st.session_state['ai_enabled'],
    help="Enable AI-powered analysis and recommendations"
)
st.session_state['ai_enabled'] = ai_enabled

if ai_enabled:
    # Check if API key is configured in secrets
    api_key_from_secrets = False
    try:
        if 'GEMINI_API_KEY' in st.secrets:
            api_key_from_secrets = True
            st.session_state['gemini_api_key'] = st.secrets['GEMINI_API_KEY']
            st.sidebar.success("✓ AI ready")
    except Exception:
        pass

    # If no secrets, show manual input option (for local development)
    if not api_key_from_secrets:
        gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password", 
                                                help="Get a free key from https://aistudio.google.com/")
        if gemini_api_key:
            st.session_state['gemini_api_key'] = gemini_api_key
            st.sidebar.success("✓ API key saved")
        else:
            st.sidebar.info("Enter API key to use AI features")
    
    # Model selection (only shown when AI is enabled)
    ai_model = st.sidebar.selectbox(
        "AI Model",
        options=["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-8b"],
        index=0,
        help="Select the AI model. gemini-2.0-flash is recommended."
    )
    st.session_state['ai_model'] = ai_model

st.make_request = make_request
