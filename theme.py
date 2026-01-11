"""
Shared theme utilities for NeqSim Web Application
"""
import streamlit as st

def apply_theme():
    """Apply the current theme (dark/light) based on session state."""
    
    # Initialize dark mode if not set
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
        [data-testid="column"] { width: 100% !important; flex: 100% !important; }
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


def theme_toggle():
    """Add a theme toggle button to the sidebar."""
    with st.sidebar:
        theme_icon = "🌙" if not st.session_state.dark_mode else "☀️"
        theme_label = "Dark Mode" if not st.session_state.dark_mode else "Light Mode"
        if st.button(f"{theme_icon} {theme_label}", key="theme_toggle"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
