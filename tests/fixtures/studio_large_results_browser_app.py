"""Browser-only large solved-workspace fixture for Studio performance profiling."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from studio.results_context import (  # noqa: E402
    equipment_design_rows,
    equipment_rows,
    load_current_result_context,
    stream_rows,
)


def _load_large_session_fixture():
    """Reuse the deterministic scale fixture guarded by the Python regression."""

    fixture_path = PROJECT_ROOT / "tests" / "test_studio_workspace_performance.py"
    module_spec = importlib.util.spec_from_file_location(
        "studio_workspace_performance_fixture",
        fixture_path,
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Unable to load large Studio fixture from {fixture_path}")
    fixture_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(fixture_module)
    return fixture_module._large_page_session()


st.set_page_config(
    page_title="Studio browser performance fixture",
    layout="wide",
)
st.session_state.update(_load_large_session_fixture())
context = load_current_result_context(st.session_state)

st.title("Large Studio browser profile")
st.caption(
    "Test-only presentation of the deterministic solved-workspace fixture. "
    "It exercises shared Studio projections and Streamlit dataframe rendering "
    "without running or replacing NeqSim calculations."
)
view = st.radio(
    "Results view",
    ("Streams", "Equipment & design"),
    horizontal=True,
)

if view == "Streams":
    rows = stream_rows(context)
    st.subheader(f"Solved streams · {len(rows)}")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    rows = equipment_rows(context)
    st.subheader(f"Solved equipment · {len(rows)}")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.subheader("Operating versus design basis")
    st.dataframe(
        pd.DataFrame(equipment_design_rows(context)),
        use_container_width=True,
        hide_index=True,
    )
    st.subheader("Engineering constraints")
    st.dataframe(pd.DataFrame([{"Status": "No active constraint"}]), hide_index=True)
