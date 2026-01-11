import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import openai
from openai import OpenAI

def make_request(question_input: str):
    # Only attempt request if API key is provided
    if not openai_api_key or openai_api_key.strip() == "":
        return ""
    try:
        client = OpenAI(api_key=openai_api_key)
        completion = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=question_input,
            max_tokens=500,
            temperature=0
        )
        return completion.choices[0].text
    except Exception:
        return ""

st.set_page_config(page_title="NeqSim", page_icon='images/neqsimlogocircleflat.png')

# Initialize dark mode in session state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Dark mode CSS
dark_mode_css = """
<style>
/* Dark mode styles */
[data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"] {
    background-color: #1a1a2e !important;
    color: #eaeaea !important;
}
[data-testid="stSidebar"] > div:first-child {
    background-color: #16213e !important;
}
.stMarkdown, .stText, p, span, label, .stCaption {
    color: #eaeaea !important;
}
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}
[data-testid="stDataEditor"], .stDataFrame {
    background-color: #0f3460 !important;
}
.stButton > button {
    background-color: #e94560 !important;
    color: white !important;
    border: none !important;
}
.stButton > button:hover {
    background-color: #ff6b6b !important;
}
a { color: #4fc3f7 !important; }
.stExpander { border-color: #0f3460 !important; }
[data-testid="stExpander"] > div:first-child {
    background-color: #16213e !important;
}
</style>
"""

# Light mode (default) with mobile-friendly CSS
light_mode_css = """
<style>
/* Responsive adjustments for mobile */
@media (max-width: 768px) {
    .stDataEditor > div { font-size: 14px; }
    .stButton > button { width: 100%; padding: 0.75rem; font-size: 16px; }
    h1 { font-size: 1.75rem !important; }
    h2, h3 { font-size: 1.25rem !important; }
    .block-container { padding: 1rem !important; }
}
/* Touch-friendly buttons */
.stButton > button { min-height: 44px; }
</style>
"""

# Apply the appropriate theme
if st.session_state.dark_mode:
    st.markdown(dark_mode_css + light_mode_css, unsafe_allow_html=True)
else:
    st.markdown(light_mode_css, unsafe_allow_html=True)

# Theme toggle in sidebar
with st.sidebar:
    theme_icon = "🌙" if not st.session_state.dark_mode else "☀️"
    theme_label = "Dark Mode" if not st.session_state.dark_mode else "Light Mode"
    if st.button(f"{theme_icon} {theme_label}", key="theme_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

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
NeqSim is integrated with OpenAI for enhanced simulation support. Enter your OpenAI API key in the sidebar to interact with the AI assistant for insights and guidance related to your simulations.
"""

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
st.make_request = make_request
