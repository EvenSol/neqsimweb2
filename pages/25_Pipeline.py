# -*- coding: utf-8 -*-
"""Native Beggs-Brill and two-fluid pipeline simulations with NeqSim."""

from __future__ import annotations

import math
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from fluids import default_fluid
from pipeline_hydraulics import (
    PIPELINE_FLUID_PRESETS,
    PipelineInputError,
    build_beggs_brill_pipe,
    build_two_fluid_pipe,
    fluid_from_preset,
    normalize_elevation_profile,
    normalize_fluid_composition,
    read_beggs_brill_profiles,
    read_two_fluid_profiles,
    solve_inlet_pressure,
    standard_mass_flow_kg_s,
)
from theme import apply_theme


st.set_page_config(
    page_title="Pipeline Hydraulics",
    page_icon="images/neqsimlogocircleflat.png",
    layout="wide",
)
apply_theme()

st.title("🔧 Pipeline Hydraulics")
st.markdown(
    """
Run NeqSim's native **PipeBeggsAndBrills** steady-state correlation or its
**TwoFluidPipe** transient model. Both paths use the same validated fluid,
geometry, units, and guarded inlet-pressure solver.
"""
)
st.caption(
    "Results are engineering-screening calculations. Confirm model validity, "
    "time-step sensitivity, and design assumptions before project use."
)
st.divider()


def _show_exception(title: str, error: Exception) -> None:
    """Show a concise error with optional diagnostics for reproducibility."""

    st.error(f"{title}: {error}")
    with st.expander("Technical details"):
        st.code(traceback.format_exc())


with st.sidebar:
    st.header("🧪 Fluid Composition")
    fluid_preset = st.selectbox(
        "Pipeline fluid preset",
        list(PIPELINE_FLUID_PRESETS),
        key="pipe_fluid_preset",
    )
    thermodynamic_model_label = st.selectbox(
        "Thermodynamic model",
        ["SRK", "PR", "CPA", "Auto-select"],
        key="pipe_thermodynamic_model",
        help=(
            "SRK is the default for hydrocarbon pipeline screening. Select CPA "
            "for strongly associating fluids such as water/glycol systems."
        ),
    )
    thermodynamic_model = {
        "SRK": "srk",
        "PR": "pr",
        "CPA": "cpa",
        "Auto-select": "auto",
    }[thermodynamic_model_label]
    if "pipe_fluid_df" not in st.session_state:
        st.session_state.pipe_fluid_df = fluid_from_preset(
            default_fluid,
            "Lean natural gas",
        )
    if "pipe_fluid_editor_revision" not in st.session_state:
        st.session_state.pipe_fluid_editor_revision = 0
    if st.button("Load selected preset", key="pipe_load_preset"):
        st.session_state.pipe_fluid_df = fluid_from_preset(
            default_fluid,
            fluid_preset,
        )
        st.session_state.pipe_fluid_editor_revision += 1

    edited_fluid = st.data_editor(
        st.session_state.pipe_fluid_df,
        column_config={
            "ComponentName": st.column_config.TextColumn("Component"),
            "MolarComposition[-]": st.column_config.NumberColumn(
                "Molar Comp.",
                min_value=0.0,
                max_value=1.0,
                format="%.6f",
            ),
            "MolarMass[kg/mol]": st.column_config.NumberColumn(
                "MW [kg/mol]",
                min_value=0.0,
                format="%.4f",
            ),
            "RelativeDensity[-]": st.column_config.NumberColumn(
                "Rel. density",
                min_value=0.0,
                format="%.4f",
            ),
        },
        num_rows="dynamic",
        width="stretch",
        key=(
            "pipe_fluid_editor_"
            f"{st.session_state.pipe_fluid_editor_revision}"
        ),
    )
    st.session_state.pipe_fluid_df = edited_fluid
    composition_sum = pd.to_numeric(
        edited_fluid["MolarComposition[-]"],
        errors="coerce",
    ).fillna(0.0).sum()
    st.caption(
        f"Entered molar-fraction sum: **{composition_sum:.6f}**. "
        "Positive compositions are normalized before simulation."
    )


tab_ss, tab_dyn = st.tabs(
    [
        "📐 Steady-State (Beggs-Brill)",
        "🌊 Dynamic Simulation (Two-Fluid Model)",
    ]
)


with tab_ss:
    st.markdown(
        """
The steady-state calculation uses NeqSim's native Beggs-Brill liquid-holdup,
flow-regime, friction, hydrostatic, thermodynamic, and heat-transfer models.
The specified outlet pressure is matched by solving the inlet pressure.
"""
    )
    st.subheader("⚙️ Pipeline Configuration")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Geometry**")
        ss_length = st.number_input(
            "Pipe Length (m)",
            value=10_000.0,
            min_value=1.0,
            step=100.0,
            key="ss_len",
        )
        ss_diameter = st.number_input(
            "Inner Diameter (mm)",
            value=200.0,
            min_value=10.0,
            step=10.0,
            key="ss_dia",
        )
        ss_roughness = st.number_input(
            "Roughness (μm)",
            value=50.0,
            min_value=0.0,
            step=5.0,
            key="ss_rough",
        )
        ss_elevation = st.number_input(
            "Total Elevation Change (m)",
            value=0.0,
            step=10.0,
            key="ss_elev",
            help="Positive is uphill; negative is downhill.",
        )
        ss_segments = st.number_input(
            "Number of Increments",
            value=20,
            min_value=5,
            max_value=200,
            key="ss_seg",
        )

    with col2:
        st.markdown("**Boundary Conditions**")
        ss_outlet_pressure = st.number_input(
            "Outlet Pressure (bara)",
            value=50.0,
            min_value=1.0,
            step=5.0,
            key="ss_outP",
        )
        ss_inlet_temperature = st.number_input(
            "Inlet Temperature (°C)",
            value=40.0,
            step=5.0,
            key="ss_inT",
        )
        ss_flow = st.number_input(
            "Flow Rate",
            value=10.0,
            min_value=0.001,
            step=1.0,
            key="ss_flow",
        )
        ss_flow_unit = st.selectbox(
            "Flow Unit",
            ["MSm3/day", "kg/s", "m3/hr"],
            key="ss_funit",
        )

    with col3:
        st.markdown("**Heat Transfer**")
        ss_ambient_temperature = st.number_input(
            "Ambient Temperature (°C)",
            value=5.0,
            step=1.0,
            key="ss_amb",
        )
        ss_htc = st.number_input(
            "Overall HTC (W/m²·K)",
            value=5.0,
            min_value=0.0,
            step=1.0,
            key="ss_htc",
            help="Zero selects NeqSim's adiabatic mode.",
        )

    st.divider()
    if st.button("🔧 Calculate Hydraulics", type="primary", key="ss_run"):
        try:
            normalized_fluid = normalize_fluid_composition(edited_fluid)
            mass_flow_kg_s = standard_mass_flow_kg_s(
                normalized_fluid,
                ss_flow,
                ss_flow_unit,
                ss_outlet_pressure,
                ss_inlet_temperature,
                thermodynamic_model,
            )

            def build_steady_pipe(inlet_pressure_bara: float):
                return build_beggs_brill_pipe(
                    normalized_fluid,
                    inlet_pressure_bara,
                    ss_inlet_temperature,
                    mass_flow_kg_s,
                    ss_length,
                    ss_diameter / 1_000.0,
                    ss_roughness / 1.0e6,
                    ss_elevation,
                    int(ss_segments),
                    ss_htc,
                    ss_ambient_temperature,
                    thermodynamic_model,
                )

            with st.spinner("Solving native Beggs-Brill pipeline..."):
                solution = solve_inlet_pressure(
                    build_steady_pipe,
                    ss_outlet_pressure,
                    tolerance_bar=0.01,
                )
                profiles = read_beggs_brill_profiles(solution.pipe)

            pressure_drop = (
                solution.inlet_pressure_bara - solution.outlet_pressure_bara
            )
            average_velocity = float(np.mean(profiles.mixture_velocity_m_s))
            st.success(
                "Native Beggs-Brill calculation converged in "
                f"{solution.iterations} pressure iterations."
            )
            st.subheader("📊 Results Summary")
            metric_columns = st.columns(4)
            metric_columns[0].metric(
                "Inlet Pressure (bara)",
                f"{solution.inlet_pressure_bara:.2f}",
            )
            metric_columns[1].metric("Pressure Drop (bar)", f"{pressure_drop:.2f}")
            metric_columns[2].metric(
                "Outlet Temperature (°C)",
                f"{profiles.temperature_c[-1]:.2f}",
            )
            metric_columns[3].metric(
                "Average Mixture Velocity (m/s)",
                f"{average_velocity:.2f}",
            )
            metric_columns = st.columns(4)
            metric_columns[0].metric(
                "Outlet Pressure (bara)",
                f"{solution.outlet_pressure_bara:.2f}",
            )
            metric_columns[1].metric(
                "Outlet Flow Regime",
                profiles.flow_regime[-1],
            )
            metric_columns[2].metric(
                "Outlet Liquid Holdup",
                f"{profiles.liquid_holdup[-1]:.4f}",
            )
            metric_columns[3].metric(
                "Mass Flow (kg/s)",
                f"{mass_flow_kg_s:.2f}",
            )

            figure = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Pressure",
                    "Temperature",
                    "Mixture Velocity",
                    "Liquid Holdup",
                ),
            )
            figure.add_trace(
                go.Scatter(
                    x=profiles.position_km,
                    y=profiles.pressure_bara,
                    mode="lines",
                    line=dict(color="#2196F3", width=2),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=profiles.position_km,
                    y=profiles.temperature_c,
                    mode="lines",
                    line=dict(color="#F44336", width=2),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
            figure.add_trace(
                go.Scatter(
                    x=profiles.position_km,
                    y=profiles.mixture_velocity_m_s,
                    mode="lines",
                    line=dict(color="#4CAF50", width=2),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=profiles.position_km,
                    y=profiles.liquid_holdup,
                    mode="lines",
                    fill="tozeroy",
                    line=dict(color="#FF9800", width=2),
                    showlegend=False,
                ),
                row=2,
                col=2,
            )
            figure.update_xaxes(title_text="Distance (km)")
            figure.update_yaxes(title_text="bara", row=1, col=1)
            figure.update_yaxes(title_text="°C", row=1, col=2)
            figure.update_yaxes(title_text="m/s", row=2, col=1)
            figure.update_yaxes(title_text="Holdup (-)", row=2, col=2)
            figure.update_layout(height=650)
            st.plotly_chart(figure, width="stretch")

            segment_table = pd.DataFrame(
                {
                    "Distance (km)": profiles.position_km,
                    "Pressure (bara)": profiles.pressure_bara,
                    "Temperature (°C)": profiles.temperature_c,
                    "Mixture Velocity (m/s)": profiles.mixture_velocity_m_s,
                    "Vsg (m/s)": profiles.gas_velocity_m_s,
                    "Vsl (m/s)": profiles.liquid_velocity_m_s,
                    "Density (kg/m³)": profiles.mixture_density_kg_m3,
                    "Reynolds Number": profiles.reynolds_number,
                    "Liquid Holdup": profiles.liquid_holdup,
                    "Flow Regime": profiles.flow_regime,
                }
            )
            st.subheader("📋 Native Increment Data")
            st.dataframe(segment_table, width="stretch", hide_index=True)

            erosion_ratios = []
            for velocity, density in zip(
                profiles.mixture_velocity_m_s,
                profiles.mixture_density_kg_m3,
            ):
                erosional_velocity = (
                    122.0 / math.sqrt(density) if density > 0.0 else math.inf
                )
                erosion_ratios.append(velocity / erosional_velocity)
            critical_index = int(np.argmax(erosion_ratios))
            critical_ratio = erosion_ratios[critical_index]
            critical_position = profiles.position_km[critical_index]
            if critical_ratio > 1.0:
                st.warning(
                    "API RP 14E screening velocity is exceeded at the critical "
                    f"increment ({critical_ratio:.2f} × limit at "
                    f"{critical_position:.2f} km)."
                )
            else:
                st.info(
                    "API RP 14E screening velocity remains below the selected "
                    f"C=122 limit; maximum utilization is {critical_ratio:.1%} "
                    f"at {critical_position:.2f} km."
                )
        except PipelineInputError as error:
            st.error(str(error))
        except Exception as error:
            _show_exception("Beggs-Brill calculation failed", error)


with tab_dyn:
    st.markdown(
        """
NeqSim's **TwoFluidPipe** solves separate gas and liquid conservation
equations with slip, terrain, heat transfer, flow-regime transitions, and
optional Lagrangian slug tracking. The steady state initializes the requested
transient before time stepping begins.
"""
    )
    st.subheader("🗺️ Pipeline Elevation Profile")
    profile_presets = {
        "Flat": {
            "Distance (m)": [0.0, 5_000.0, 10_000.0],
            "Elevation (m)": [0.0, 0.0, 0.0],
        },
        "Uphill": {
            "Distance (m)": [0.0, 5_000.0, 10_000.0],
            "Elevation (m)": [0.0, 50.0, 100.0],
        },
        "Downhill": {
            "Distance (m)": [0.0, 5_000.0, 10_000.0],
            "Elevation (m)": [0.0, -50.0, -100.0],
        },
        "Undulating": {
            "Distance (m)": [0.0, 2_000.0, 4_000.0, 6_000.0, 8_000.0, 10_000.0],
            "Elevation (m)": [0.0, -30.0, 10.0, -50.0, -20.0, 0.0],
        },
        "V-shape (Dip)": {
            "Distance (m)": [0.0, 3_000.0, 5_000.0, 7_000.0, 10_000.0],
            "Elevation (m)": [0.0, -80.0, -100.0, -60.0, 0.0],
        },
        "Riser": {
            "Distance (m)": [0.0, 7_000.0, 8_000.0, 10_000.0],
            "Elevation (m)": [0.0, -200.0, -200.0, 0.0],
        },
    }
    profile_preset = st.selectbox(
        "Profile Preset",
        list(profile_presets),
        key="dyn_preset",
    )
    if (
        "dyn_profile_df" not in st.session_state
        or st.session_state.get("_dyn_last_preset") != profile_preset
    ):
        st.session_state.dyn_profile_df = pd.DataFrame(
            profile_presets[profile_preset]
        )
        st.session_state._dyn_last_preset = profile_preset
    profile_df = st.data_editor(
        st.session_state.dyn_profile_df,
        column_config={
            "Distance (m)": st.column_config.NumberColumn(
                "Distance (m)",
                min_value=0.0,
                format="%.0f",
            ),
            "Elevation (m)": st.column_config.NumberColumn(
                "Elevation (m)",
                format="%.1f",
            ),
        },
        num_rows="dynamic",
        width="stretch",
        key="dyn_profile_editor",
    )
    st.session_state.dyn_profile_df = profile_df

    if len(profile_df) >= 2:
        profile_figure = go.Figure()
        profile_figure.add_trace(
            go.Scatter(
                x=[value / 1_000.0 for value in profile_df["Distance (m)"]],
                y=profile_df["Elevation (m)"],
                mode="lines+markers",
                fill="tozeroy",
                line=dict(color="#1976D2", width=3),
                marker=dict(size=8),
                name="Elevation",
            )
        )
        profile_figure.update_layout(
            title="Pipeline Elevation Profile",
            xaxis_title="Distance (km)",
            yaxis_title="Elevation (m)",
            height=280,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(profile_figure, width="stretch")

    st.divider()
    st.subheader("⚙️ Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Pipe Geometry**")
        dyn_diameter_mm = st.number_input(
            "Inner Diameter (mm)",
            value=200.0,
            min_value=10.0,
            step=10.0,
            key="dyn_diam",
        )
        dyn_roughness_um = st.number_input(
            "Roughness (μm)",
            value=50.0,
            min_value=0.0,
            step=5.0,
            key="dyn_rough",
        )
        dyn_sections = st.number_input(
            "Number of Sections",
            value=50,
            min_value=10,
            max_value=500,
            key="dyn_nsec",
        )
    with col2:
        st.markdown("**Boundary Conditions**")
        dyn_outlet_pressure = st.number_input(
            "Outlet Pressure (bara)",
            value=50.0,
            min_value=1.0,
            step=5.0,
            key="dyn_outP",
        )
        dyn_inlet_temperature = st.number_input(
            "Inlet Temperature (°C)",
            value=40.0,
            step=5.0,
            key="dyn_inT",
        )
        dyn_base_flow = st.number_input(
            "Base Flow Rate (kg/hr)",
            value=5_000.0,
            min_value=1.0,
            step=100.0,
            key="dyn_base_flow",
        )
    with col3:
        st.markdown("**Heat Transfer**")
        dyn_ambient_temperature = st.number_input(
            "Ambient / Surface Temp (°C)",
            value=5.0,
            step=1.0,
            key="dyn_ambT",
        )
        dyn_htc = st.number_input(
            "Overall HTC (W/m²·K)",
            value=5.0,
            min_value=0.0,
            step=1.0,
            key="dyn_htc",
            help="Zero disables wall heat transfer.",
        )

    st.divider()
    st.subheader("🎛️ Flow Rate & Simulation")
    col1, col2 = st.columns([2, 1])
    with col1:
        dyn_flow_percent = st.slider(
            "Feed Flow Rate Adjustment",
            min_value=5,
            max_value=300,
            value=100,
            step=5,
            format="%d %%",
            key="dyn_flow_pct",
            help="Zero flow is excluded because native J/kg properties require mass flow.",
        )
        effective_flow_kg_hr = dyn_base_flow * dyn_flow_percent / 100.0
        st.caption(
            f"Effective flow rate: **{effective_flow_kg_hr:,.0f} kg/hr** "
            f"({effective_flow_kg_hr / 3_600.0:.3f} kg/s)"
        )
    with col2:
        dyn_simulation_time = st.number_input(
            "Simulation Time (s)",
            value=600.0,
            min_value=1.0,
            step=60.0,
            key="dyn_simtime",
        )
        dyn_steps = st.number_input(
            "Transient Steps",
            value=10,
            min_value=2,
            max_value=100,
            key="dyn_steps",
        )
        dyn_slug_tracking = st.checkbox(
            "Enable Slug Tracking",
            value=True,
            key="dyn_slug",
        )

    st.divider()
    if st.button("🌊 Run Dynamic Simulation", type="primary", key="dyn_run"):
        try:
            normalized_fluid = normalize_fluid_composition(edited_fluid)
            normalized_distances, normalized_elevations = normalize_elevation_profile(
                profile_df["Distance (m)"].tolist(),
                profile_df["Elevation (m)"].tolist(),
            )

            def build_dynamic_pipe(inlet_pressure_bara: float):
                return build_two_fluid_pipe(
                    normalized_fluid,
                    inlet_pressure_bara,
                    dyn_inlet_temperature,
                    effective_flow_kg_hr,
                    dyn_diameter_mm / 1_000.0,
                    dyn_roughness_um / 1.0e6,
                    normalized_distances,
                    normalized_elevations,
                    int(dyn_sections),
                    dyn_htc,
                    dyn_ambient_temperature,
                    dyn_slug_tracking,
                    thermodynamic_model,
                )

            with st.spinner("Initializing native two-fluid steady state..."):
                solution = solve_inlet_pressure(
                    build_dynamic_pipe,
                    dyn_outlet_pressure,
                    tolerance_bar=0.02,
                )
                pipe = solution.pipe
                inlet_stream = solution.inlet_stream
                steady_profiles = read_two_fluid_profiles(pipe)

            st.success(
                "Two-fluid steady state converged in "
                f"{solution.iterations} pressure iterations. Inlet pressure "
                f"is {solution.inlet_pressure_bara:.2f} bara."
            )
            inlet_phase_count = int(inlet_stream.getFluid().getNumberOfPhases())
            if inlet_phase_count < 2 and dyn_slug_tracking:
                st.warning(
                    "The inlet is single phase. Slug tracking remains enabled, "
                    "but slug statistics are meaningful only if liquid forms downstream."
                )

            st.subheader("📊 Initialized Steady State")
            metric_columns = st.columns(4)
            metric_columns[0].metric(
                "Inlet P (bara)",
                f"{steady_profiles.pressure_bara[0]:.2f}",
            )
            metric_columns[1].metric(
                "Outlet P (bara)",
                f"{steady_profiles.pressure_bara[-1]:.2f}",
            )
            metric_columns[2].metric(
                "Outlet T (°C)",
                f"{steady_profiles.temperature_c[-1]:.2f}",
            )
            metric_columns[3].metric(
                "Flow Rate (kg/hr)",
                f"{effective_flow_kg_hr:,.0f}",
            )
            metric_columns = st.columns(4)
            metric_columns[0].metric(
                "ΔP (bar)",
                f"{solution.inlet_pressure_bara - solution.outlet_pressure_bara:.2f}",
            )
            metric_columns[1].metric(
                "Average Holdup",
                f"{np.mean(steady_profiles.liquid_holdup):.4f}",
            )
            try:
                liquid_inventory = float(pipe.getLiquidInventory("m3"))
            except Exception:
                liquid_inventory = math.nan
            metric_columns[2].metric(
                "Liquid Inventory (m³)",
                f"{liquid_inventory:.3f}" if math.isfinite(liquid_inventory) else "N/A",
            )
            metric_columns[3].metric(
                "Mid-pipe Regime",
                steady_profiles.flow_regime[len(steady_profiles.flow_regime) // 2],
            )

            steady_figure = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Pressure",
                    "Temperature",
                    "Liquid Holdup",
                    "Phase Velocities",
                ),
            )
            steady_figure.add_trace(
                go.Scatter(
                    x=steady_profiles.position_km,
                    y=steady_profiles.pressure_bara,
                    mode="lines",
                    line=dict(color="#2196F3", width=2),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            steady_figure.add_trace(
                go.Scatter(
                    x=steady_profiles.position_km,
                    y=steady_profiles.temperature_c,
                    mode="lines",
                    line=dict(color="#F44336", width=2),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
            steady_figure.add_trace(
                go.Scatter(
                    x=steady_profiles.position_km,
                    y=steady_profiles.liquid_holdup,
                    mode="lines",
                    fill="tozeroy",
                    line=dict(color="#4CAF50", width=2),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            steady_figure.add_trace(
                go.Scatter(
                    x=steady_profiles.position_km,
                    y=steady_profiles.gas_velocity_m_s,
                    mode="lines",
                    name="Gas",
                    line=dict(color="#FF9800", width=2),
                ),
                row=2,
                col=2,
            )
            steady_figure.add_trace(
                go.Scatter(
                    x=steady_profiles.position_km,
                    y=steady_profiles.liquid_velocity_m_s,
                    mode="lines",
                    name="Liquid",
                    line=dict(color="#2196F3", width=2),
                ),
                row=2,
                col=2,
            )
            steady_figure.update_xaxes(title_text="Distance (km)")
            steady_figure.update_yaxes(title_text="bara", row=1, col=1)
            steady_figure.update_yaxes(title_text="°C", row=1, col=2)
            steady_figure.update_yaxes(title_text="Holdup (-)", row=2, col=1)
            steady_figure.update_yaxes(title_text="m/s", row=2, col=2)
            steady_figure.update_layout(height=650)
            st.plotly_chart(steady_figure, width="stretch")

            with st.expander("📋 Initialized Section Data"):
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Distance (km)": steady_profiles.position_km,
                            "Pressure (bara)": steady_profiles.pressure_bara,
                            "Temperature (°C)": steady_profiles.temperature_c,
                            "Liquid Holdup": steady_profiles.liquid_holdup,
                            "Gas Velocity (m/s)": steady_profiles.gas_velocity_m_s,
                            "Liquid Velocity (m/s)": (
                                steady_profiles.liquid_velocity_m_s
                            ),
                            "Flow Regime": steady_profiles.flow_regime,
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )

            st.divider()
            st.subheader("🌊 Transient Simulation")
            time_step = dyn_simulation_time / int(dyn_steps)
            st.caption(
                f"Advancing {dyn_simulation_time:.1f} s in {int(dyn_steps)} "
                f"native steps of {time_step:.3f} s. Repeat with a smaller step "
                "to check time-step sensitivity."
            )
            progress = st.progress(0)
            time_label = st.empty()
            live_chart = st.empty()
            history_chart = st.empty()
            slug_box = st.empty()

            time_history = [0.0]
            outlet_pressure_history = [steady_profiles.pressure_bara[-1]]
            outlet_temperature_history = [steady_profiles.temperature_c[-1]]
            inventory_history = [
                liquid_inventory if math.isfinite(liquid_inventory) else 0.0
            ]

            for step in range(int(dyn_steps)):
                pipe.runTransient(time_step)
                transient_profiles = read_two_fluid_profiles(pipe)
                current_time = (step + 1) * time_step
                time_history.append(current_time)
                outlet_pressure_history.append(
                    transient_profiles.pressure_bara[-1]
                )
                outlet_temperature_history.append(
                    transient_profiles.temperature_c[-1]
                )
                try:
                    current_inventory = float(pipe.getLiquidInventory("m3"))
                    if not math.isfinite(current_inventory):
                        raise ValueError("non-finite inventory")
                except Exception:
                    current_inventory = inventory_history[-1]
                inventory_history.append(current_inventory)

                progress.progress((step + 1) / int(dyn_steps))
                time_label.markdown(
                    f"**t = {current_time:.1f} s** / {dyn_simulation_time:.1f} s"
                )
                live_figure = make_subplots(
                    rows=1,
                    cols=2,
                    subplot_titles=(
                        f"Pressure at t={current_time:.1f} s",
                        f"Liquid holdup at t={current_time:.1f} s",
                    ),
                )
                live_figure.add_trace(
                    go.Scatter(
                        x=transient_profiles.position_km,
                        y=transient_profiles.pressure_bara,
                        mode="lines",
                        line=dict(color="#2196F3", width=2),
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )
                live_figure.add_trace(
                    go.Scatter(
                        x=transient_profiles.position_km,
                        y=transient_profiles.liquid_holdup,
                        mode="lines",
                        fill="tozeroy",
                        line=dict(color="#4CAF50", width=2),
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )
                live_figure.update_xaxes(title_text="Distance (km)")
                live_figure.update_yaxes(title_text="bara", row=1, col=1)
                live_figure.update_yaxes(title_text="Holdup (-)", row=1, col=2)
                live_figure.update_layout(height=350)
                live_chart.plotly_chart(live_figure, width="stretch")

                history_figure = make_subplots(
                    rows=1,
                    cols=3,
                    subplot_titles=(
                        "Liquid Inventory",
                        "Outlet Pressure",
                        "Outlet Temperature",
                    ),
                )
                history_figure.add_trace(
                    go.Scatter(
                        x=time_history,
                        y=inventory_history,
                        mode="lines+markers",
                        line=dict(color="#FF9800", width=2),
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )
                history_figure.add_trace(
                    go.Scatter(
                        x=time_history,
                        y=outlet_pressure_history,
                        mode="lines+markers",
                        line=dict(color="#2196F3", width=2),
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )
                history_figure.add_trace(
                    go.Scatter(
                        x=time_history,
                        y=outlet_temperature_history,
                        mode="lines+markers",
                        line=dict(color="#F44336", width=2),
                        showlegend=False,
                    ),
                    row=1,
                    col=3,
                )
                history_figure.update_xaxes(title_text="Time (s)")
                history_figure.update_yaxes(title_text="m³", row=1, col=1)
                history_figure.update_yaxes(title_text="bara", row=1, col=2)
                history_figure.update_yaxes(title_text="°C", row=1, col=3)
                history_figure.update_layout(height=300)
                history_chart.plotly_chart(history_figure, width="stretch")

                if dyn_slug_tracking:
                    try:
                        slug_box.text(str(pipe.getSlugStatisticsSummary()))
                    except Exception:
                        slug_box.caption("Slug statistics are not available for this state.")

            final_profiles = read_two_fluid_profiles(pipe)
            st.success(f"Transient simulation complete — {dyn_simulation_time:.1f} s")
            st.subheader("📋 Final State")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Distance (km)": final_profiles.position_km,
                        "Pressure (bara)": final_profiles.pressure_bara,
                        "Temperature (°C)": final_profiles.temperature_c,
                        "Liquid Holdup": final_profiles.liquid_holdup,
                        "Gas Velocity (m/s)": final_profiles.gas_velocity_m_s,
                        "Liquid Velocity (m/s)": final_profiles.liquid_velocity_m_s,
                        "Flow Regime": final_profiles.flow_regime,
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            st.subheader("💧 Liquid Accumulation Summary")
            metric_columns = st.columns(3)
            metric_columns[0].metric(
                "Final Liquid Inventory (m³)",
                f"{inventory_history[-1]:.3f}",
            )
            metric_columns[1].metric(
                "Maximum Liquid Inventory (m³)",
                f"{max(inventory_history):.3f}",
            )
            metric_columns[2].metric(
                "Inventory Change (m³)",
                f"{inventory_history[-1] - inventory_history[0]:+.3f}",
            )
            maximum_holdup_index = int(np.argmax(final_profiles.liquid_holdup))
            st.info(
                "Maximum final liquid holdup is "
                f"{final_profiles.liquid_holdup[maximum_holdup_index]:.4f} at "
                f"{final_profiles.position_km[maximum_holdup_index]:.2f} km; "
                f"regime {final_profiles.flow_regime[maximum_holdup_index]}."
            )
        except PipelineInputError as error:
            st.error(str(error))
        except Exception as error:
            _show_exception("Two-fluid simulation failed", error)
