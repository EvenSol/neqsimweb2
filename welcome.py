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
    # Only attempt request if API key is available
    api_key = get_gemini_api_key()
    if not api_key or api_key.strip() == "":
        return ""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
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

### NeqSim AI Assistant
NeqSim is integrated with Google Gemini AI for enhanced simulation support and analysis.
"""

# Check if API key is configured in secrets
api_key_from_secrets = False
try:
    if 'GEMINI_API_KEY' in st.secrets:
        api_key_from_secrets = True
        st.session_state['gemini_api_key'] = st.secrets['GEMINI_API_KEY']
        st.sidebar.success("✓ AI features enabled")
except Exception:
    pass

# If no secrets, show manual input option (for local development)
if not api_key_from_secrets:
    gemini_api_key = st.sidebar.text_input("Gemini API Key (optional)", type="password", 
                                            help="Get a free key from https://aistudio.google.com/")
    if gemini_api_key:
        st.session_state['gemini_api_key'] = gemini_api_key
        st.sidebar.success("✓ API key saved for all pages")

st.make_request = make_request
