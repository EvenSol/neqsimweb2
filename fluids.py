# Sample data for the DataFrame
import pandas as pd

# =============================================================================
# Shared Fluid Registry
# =============================================================================
# Pages can save/load fluid compositions to/from a shared in-session registry.
# This allows users to define a fluid once and reuse it across pages.

def _get_registry():
    """Return the shared fluid registry dict (lazy-init)."""
    import streamlit as st
    if 'saved_fluids' not in st.session_state:
        st.session_state.saved_fluids = {}
    return st.session_state.saved_fluids


def save_fluid(name: str, df):
    """Save a fluid composition DataFrame to the shared registry."""
    registry = _get_registry()
    registry[name] = pd.DataFrame(df).copy()


def load_fluid(name: str):
    """Load a fluid composition DataFrame from the shared registry. Returns None if not found."""
    return _get_registry().get(name)


def get_saved_fluid_names():
    """Return a list of saved fluid names."""
    return list(_get_registry().keys())


def fluid_library_selector(page_key: str, session_key: str):
    """Render a 'Load from Library' selectbox + 'Save to Library' controls.

    Parameters
    ----------
    page_key : str
        Unique prefix for widget keys on this page (e.g. 'tpflash').
    session_key : str
        The session_state key that holds the page's fluid DataFrame.
        When the user loads a fluid, this key is overwritten.

    Returns
    -------
    bool
        True if a fluid was loaded (caller should st.rerun if needed).
    """
    import streamlit as st

    loaded = False
    names = get_saved_fluid_names()
    if names:
        pick = st.selectbox(
            "📂 Load from Fluid Library",
            ["— none —"] + names,
            key=f"_lib_load_{page_key}",
        )
        if pick != "— none —":
            df = load_fluid(pick)
            if df is not None:
                st.session_state[session_key] = df.copy()
                loaded = True
                st.success(f"Loaded '{pick}' from fluid library")

    col_name, col_btn = st.columns([3, 1])
    save_name = col_name.text_input(
        "Fluid name",
        value="",
        key=f"_lib_save_name_{page_key}",
        placeholder="Enter name to save…",
    )
    if col_btn.button("💾 Save", key=f"_lib_save_btn_{page_key}"):
        if save_name.strip():
            if session_key in st.session_state:
                save_fluid(save_name.strip(), st.session_state[session_key])
                st.success(f"Saved '{save_name.strip()}' to fluid library")
            else:
                st.warning("No fluid data to save — edit the composition first.")
        else:
            st.warning("Enter a name before saving.")

    return loaded


default_fluid = {
    'ComponentName':  ["water", "methanol", "MEG", "TEG", "oxygen", "hydrogen", "nitrogen", "CO2", "H2S", "methane", "ethane", "propane", "i-butane", "n-butane", "i-pentane", "n-pentane", "n-hexane", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20"],
    'MolarComposition[-]':  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'MolarMass[kg/mol]': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.0913, 0.1041, 0.1188, 0.136, 0.150, 0.164, 0.179, 0.188, 0.204, 0.216, 0.236, 0.253, 0.27, 0.391],
    'RelativeDensity[-]': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.746, 0.768, 0.79, 0.787, 0.793, 0.804, 0.817, 0.83, 0.835, 0.843, 0.837, 0.84, 0.85, 0.877]
}

detailedHC_data = {
    'ComponentName':  ["nitrogen", "CO2", "methane", "ethane", "propane", "i-butane", "n-butane", "i-pentane", "n-pentane", "2-m-C5", "3-m-C5", "n-hexane", "benzene", "c-hexane", "n-heptane", "c-C7", "toluene","n-octane", "m-Xylene","c-C8","n-nonane","nC10","nC11","nC12", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20"],
    'MolarComposition[-]':  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'MolarMass[kg/mol]': [None, None, None, None, None, None, None, None, None, None, None, None, None,None, None, None, None,None, None, None, None, None,None, None, 0.0913, 0.1041, 0.1188, 0.136, 0.150, 0.164, 0.179, 0.188, 0.204, 0.216, 0.236, 0.253, 0.27, 0.391],
    'RelativeDensity[-]': [None, None, None, None, None, None, None, None, None, None, None, None, None,None, None, None, None,None, None, None, None, None,None, None, 0.746, 0.768, 0.79, 0.787, 0.793, 0.804, 0.817, 0.83, 0.835, 0.843, 0.837, 0.84, 0.85, 0.877]
}

lng_fluid = {
    'ComponentName':  ["nitrogen", "methane", "ethane", "propane", "i-butane", "n-butane", "i-pentane", "n-pentane", "n-hexane"],
    'MolarComposition[-]':  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}


