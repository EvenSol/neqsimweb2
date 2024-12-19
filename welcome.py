import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import openai
from openai import OpenAI

def make_request(question_input: str):
    try:
        OpenAI.api_key = openai_api_key
        client = OpenAI(api_key=openai_api_key)
    except:
        st.write('OpenAI key not provided...')
    try:
        completion = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=question_input,
            max_tokens=500,
            temperature=0
        )
        return completion.choices[0].text
    except:
        return ""

st.set_page_config(page_title="NeqSim", page_icon='images/neqsimlogocircleflat.png')

st.image('images/neqsimlogocircleflat.png', width=150)

st.write("# Welcome to the NeqSim Simulation Platform! 👋")

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
