"""Entrypoint and dynamic page router for NeqSim Web."""

import os as _os

# JVM module-access flags required by XStream on Java 17+.
# Must be set before *any* import triggers jpype.startJVM().
if "add-opens" not in _os.environ.get("JAVA_TOOL_OPTIONS", ""):
    _os.environ["JAVA_TOOL_OPTIONS"] = (
        "--add-opens=java.base/java.util=ALL-UNNAMED "
        "--add-opens=java.base/java.lang=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
        "--add-opens=java.base/java.io=ALL-UNNAMED"
    )

import streamlit as st

from app_navigation import (
    experimental_page_specs,
    stable_page_specs,
)


EXPERIMENTAL_MODE_KEY = "experimental_mode"
EXPERIMENTAL_MODE_WIDGET_KEY = "_experimental_mode_toggle"


def get_gemini_api_key():
    """Get the Gemini API key from secrets or session state."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return st.session_state.get("gemini_api_key", "")


def make_request(question_input: str):
    """Run optional AI interpretation when it is configured and enabled."""
    if not st.session_state.get("ai_enabled", False):
        return ""
    api_key = get_gemini_api_key()
    if not api_key or not api_key.strip():
        return ""
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=question_input,
        )
        return response.text
    except Exception:
        return ""


def create_page(spec):
    """Create a Streamlit page from a navigation specification."""
    return st.Page(spec.path, title=spec.title)


def persist_experimental_mode():
    """Keep the mode selection stable while navigating between pages."""
    st.session_state[EXPERIMENTAL_MODE_KEY] = bool(
        st.session_state[EXPERIMENTAL_MODE_WIDGET_KEY]
    )


if EXPERIMENTAL_MODE_KEY not in st.session_state:
    st.session_state[EXPERIMENTAL_MODE_KEY] = False
if EXPERIMENTAL_MODE_WIDGET_KEY not in st.session_state:
    st.session_state[EXPERIMENTAL_MODE_WIDGET_KEY] = st.session_state[
        EXPERIMENTAL_MODE_KEY
    ]

experimental_mode = bool(st.session_state.get(EXPERIMENTAL_MODE_KEY, False))
navigation_pages = {
    "": [st.Page("home.py", title="Home", icon="🏠", default=True)],
    "Stable tools": [create_page(spec) for spec in stable_page_specs()],
}
if experimental_mode:
    navigation_pages["Experimental"] = [
        create_page(spec) for spec in experimental_page_specs()
    ]

selected_page = st.navigation(navigation_pages)

# Streamlit renders the navigation menu at the top of the sidebar. Creating the
# mode control afterwards keeps it directly below the menu on every page.
with st.sidebar:
    st.divider()
    st.toggle(
        "Experimental mode",
        key=EXPERIMENTAL_MODE_WIDGET_KEY,
        on_change=persist_experimental_mode,
        help=(
            "Show developing tools. Experimental models and interfaces may "
            "change and require additional validation."
        ),
    )
    if experimental_mode:
        st.caption("Experimental tools are enabled for this session.")

# Classic pages offering optional AI interpretation call this shared function.
st.make_request = make_request
selected_page.run()
