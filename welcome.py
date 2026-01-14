import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import google.generativeai as genai
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
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(question_input)
        return response.text
    except Exception:
        return ""

st.set_page_config(page_title="NeqSim", page_icon='images/neqsimlogocircleflat.png')
apply_theme()
theme_toggle()

st.image('images/neqsimlogocircleflat.png', width=150)

st.write("# Welcome to the NeqSim Process Simulation Tool! 👋")

"""
### About NeqSim
NeqSim (Non-equilibrium Simulator) is a library for the simulation of fluid behavior, phase equilibrium, and process systems.
Explore the various models and simulations NeqSim offer through this easy-to-use Streamlit interface.

### Documentation & Tutorials
For comprehensive documentation on how to use NeqSim for processing and fluid simulations, please refer to our detailed tutorial:  
[Introduction to Gas Processing Using NeqSim](https://colab.research.google.com/github/EvenSol/NeqSim-Colab/blob/master/notebooks/examples_of_NeqSim_in_Colab.ipynb)

### GitHub Repository
NeqSim is developed in the Java programming language and is available as an open-source project via GitHub. You can access the complete source code and contribute to the project via the home page:

- [NeqSim Home](https://equinor.github.io/neqsimhome/)

### Community & Feedback
We welcome any feedback, questions, or suggestions for further development. Join the conversation or contribute to discussions on our GitHub page:

- [NeqSim GitHub Discussions](https://github.com/equinor/neqsim/discussions)

### Getting Started
Use the left-hand menu to select the desired simulation or process. Enter any required inputs, and NeqSim will handle the calculations.
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
