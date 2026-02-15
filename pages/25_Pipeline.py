# -*- coding: utf-8 -*-
"""
Pipeline Hydraulics Page
========================
Steady-state (Beggs-Brill) and dynamic two-fluid model (OLGA-style) pipeline
simulations. Supports terrain profiles, slug tracking, and live transient
visualisation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from theme import apply_theme
from fluids import default_fluid

st.set_page_config(
    page_title="Pipeline Hydraulics",
    page_icon='images/neqsimlogocircleflat.png',
    layout="wide",
)
apply_theme()

st.title('🔧 Pipeline Hydraulics')
st.markdown("""
Steady-state and **dynamic two-fluid model** pipeline simulations.
Supports single-phase and multiphase flow, terrain profiles, slug tracking,
and live transient visualisation.
""")
st.divider()

# =============================================================================
# Sidebar — Fluid Composition
# =============================================================================
with st.sidebar:
    st.header("🧪 Fluid Composition")
    if 'pipe_fluid_df' not in st.session_state:
        st.session_state.pipe_fluid_df = pd.DataFrame(default_fluid)

    edited_fluid = st.data_editor(
        st.session_state.pipe_fluid_df,
        column_config={
            "ComponentName": st.column_config.TextColumn("Component"),
            "MolarComposition[-]": st.column_config.NumberColumn(
                "Molar Comp.", min_value=0.0, max_value=1.0, format="%.6f"
            ),
            "MolarMass[kg/mol]": st.column_config.NumberColumn(
                "MW [kg/mol]", min_value=0.0, format="%.4f"
            ),
            "RelativeDensity[-]": st.column_config.NumberColumn(
                "Rel.Density", min_value=0.0, format="%.4f"
            ),
        },
        num_rows="dynamic",
        use_container_width=True,
    )

# =============================================================================
# Tabs
# =============================================================================
tab_ss, tab_dyn = st.tabs([
    "📐 Steady-State (Beggs-Brill)",
    "🌊 Dynamic Simulation (Two-Fluid Model)",
])


# #####################################################################
#  TAB 1 — STEADY-STATE  (Beggs-Brill correlations)
# #####################################################################
with tab_ss:
    st.subheader("⚙️ Pipeline Configuration")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Geometry**")
        ss_length = st.number_input("Pipe Length (m)", value=10000.0, min_value=1.0, step=100.0, key="ss_len")
        ss_diameter = st.number_input("Inner Diameter (mm)", value=200.0, min_value=10.0, step=10.0, key="ss_dia")
        ss_roughness = st.number_input("Roughness (μm)", value=50.0, min_value=0.0, step=5.0, key="ss_rough")
        ss_elevation = st.number_input("Total Elevation Change (m)", value=0.0, step=10.0, key="ss_elev",
                                       help="Positive = uphill, Negative = downhill")
        ss_segments = st.number_input("Number of Segments", value=20, min_value=5, max_value=200, key="ss_seg")

    with col2:
        st.markdown("**Conditions**")
        ss_outlet_P = st.number_input("Outlet Pressure (bara)", value=50.0, min_value=1.0, step=5.0, key="ss_outP")
        ss_inlet_T = st.number_input("Inlet Temperature (°C)", value=40.0, step=5.0, key="ss_inT")
        ss_flow = st.number_input("Flow Rate", value=10.0, min_value=0.001, step=1.0, key="ss_flow")
        ss_flow_unit = st.selectbox("Flow Unit", ["MSm3/day", "kg/s", "m3/hr"], key="ss_funit")

    with col3:
        st.markdown("**Environment / Heat Transfer**")
        ss_ambient = st.number_input("Ambient Temperature (°C)", value=5.0, step=1.0, key="ss_amb")
        ss_htc = st.number_input("Overall HTC (W/m²·K)", value=5.0, min_value=0.0, step=1.0, key="ss_htc",
                                 help="0 = adiabatic")
        ss_material = st.selectbox("Pipe Material",
                                   ["Carbon Steel", "Stainless Steel", "Duplex SS", "CRA Lined"], key="ss_mat")

    st.divider()

    # -----------------------------------------------------------------
    # Helper: march forward from a given inlet pressure and return all
    # segment data including the calculated outlet pressure.
    # -----------------------------------------------------------------
    def _march_forward(inlet_P, edited_fluid, inlet_T, mass_flow_kgs,
                       id_m, roughness_m, n_segments, seg_length, seg_elevation,
                       htc, ambient_T):
        """Forward march along the pipeline. Returns (outlet_P, result_dict)."""
        from neqsim.thermo import fluid_df, TPflash

        positions = [0]
        pressures = [inlet_P]
        temperatures = [inlet_T]
        velocities, densities_list, viscosities_list = [], [], []
        flow_regimes, reynolds_numbers = [], []
        phase_counts, liquid_holdups = [], []
        gas_velocities, liq_velocities = [], []
        area = np.pi / 4 * id_m ** 2

        current_P, current_T = inlet_P, inlet_T

        for seg in range(n_segments):
            seg_fluid = fluid_df(edited_fluid, lastIsPlusFraction=False, add_all_components=False)
            seg_fluid.setPressure(current_P, "bara")
            seg_fluid.setTemperature(current_T + 273.15, "K")
            TPflash(seg_fluid)
            seg_fluid.initProperties()

            seg_rho = seg_fluid.getDensity("kg/m3")
            try:
                seg_mu = seg_fluid.getViscosity()
            except Exception:
                seg_mu = 1e-5

            vel = mass_flow_kgs / (seg_rho * area) if seg_rho > 0 else 0
            Re = seg_rho * vel * id_m / seg_mu if seg_mu > 0 else 1e6

            rel_rough = roughness_m / id_m
            if Re > 2300:
                ff = (-1.8 * np.log10((rel_rough / 3.7) ** 1.11 + 6.9 / Re)) ** (-2)
            elif Re > 0:
                ff = 64 / Re
            else:
                ff = 0.02

            seg_n_phases = seg_fluid.getNumberOfPhases()
            lambda_l, vsg, vsl = 0.0, 0.0, 0.0
            if seg_n_phases > 1:
                try:
                    gas_vol, liq_vol, gas_mass, liq_mass = 0.0, 0.0, 0.0, 0.0
                    for pi in range(seg_n_phases):
                        phase = seg_fluid.getPhase(pi)
                        pvol = phase.getVolume("m3")
                        prho = phase.getDensity("kg/m3")
                        ptype = str(phase.getPhaseTypeName()).lower()
                        if "gas" in ptype:
                            gas_vol += pvol; gas_mass += pvol * prho
                        else:
                            liq_vol += pvol; liq_mass += pvol * prho
                    total_vol = gas_vol + liq_vol
                    total_mass = gas_mass + liq_mass
                    lambda_l = liq_vol / total_vol if total_vol > 0 else 0.5
                    if total_mass > 0 and area > 0:
                        gmf = gas_mass / total_mass
                        lmf = liq_mass / total_mass
                        grho = gas_mass / gas_vol if gas_vol > 0 else seg_rho
                        lrho = liq_mass / liq_vol if liq_vol > 0 else seg_rho
                        vsg = (mass_flow_kgs * gmf / grho) / area if grho > 0 else 0
                        vsl = (mass_flow_kgs * lmf / lrho) / area if lrho > 0 else 0

                    Vm = vel
                    NFr = Vm ** 2 / (9.81 * id_m) if id_m > 0 else 0
                    if 0.001 < lambda_l < 0.999:
                        L1 = 316.0 * lambda_l ** 0.302
                        L2 = 0.0009252 * lambda_l ** (-2.4684)
                        L3 = 0.10 * lambda_l ** (-1.4516)
                        L4 = 0.5 * lambda_l ** (-6.738)
                        if (lambda_l < 0.01 and NFr < L1) or (lambda_l >= 0.01 and NFr < L2):
                            regime = "Segregated"
                        elif lambda_l >= 0.01 and L2 <= NFr <= L3:
                            regime = "Transition"
                        elif (0.01 <= lambda_l < 0.4 and L3 < NFr <= L1) or \
                                (lambda_l >= 0.4 and L3 < NFr <= L4):
                            regime = "Intermittent"
                        elif (lambda_l < 0.4 and NFr >= L1) or \
                                (lambda_l >= 0.4 and NFr > L4):
                            regime = "Distributed"
                        else:
                            regime = "Intermittent"
                    else:
                        regime = "Turbulent" if Re > 2300 else "Laminar"
                except Exception:
                    regime = "Multiphase"
            else:
                ptype = str(seg_fluid.getPhase(0).getPhaseTypeName()).lower()
                if "gas" in ptype:
                    vsg = vel; vsl = 0.0
                else:
                    vsg = 0.0; vsl = vel
                if Re > 2300:
                    regime = "Turbulent"
                elif Re > 0:
                    regime = "Laminar"
                else:
                    regime = "Unknown"

            dP_friction = ff * (seg_length / id_m) * 0.5 * seg_rho * vel ** 2
            dP_gravity = seg_rho * 9.81 * seg_elevation
            dP_total_bara = (dP_friction + dP_gravity) / 1e5

            cp = 2000.0
            try:
                cp = seg_fluid.getCp("J/kg/K")
            except Exception:
                pass
            if htc > 0 and mass_flow_kgs > 0:
                perimeter = np.pi * id_m
                Q_heat = htc * perimeter * seg_length * (current_T - ambient_T)
                dT = Q_heat / (mass_flow_kgs * cp) if cp > 0 else 0
            else:
                dT = 0

            current_P = max(current_P - dP_total_bara, 1.0)
            current_T = current_T - dT

            positions.append((seg + 1) * seg_length)
            pressures.append(current_P)
            temperatures.append(current_T)
            velocities.append(vel)
            densities_list.append(seg_rho)
            viscosities_list.append(seg_mu)
            flow_regimes.append(regime)
            reynolds_numbers.append(Re)
            phase_counts.append(seg_n_phases)
            liquid_holdups.append(round(lambda_l, 4))
            gas_velocities.append(round(vsg, 3))
            liq_velocities.append(round(vsl, 3))

        return {
            "positions": positions,
            "pressures": pressures,
            "temperatures": temperatures,
            "velocities": velocities,
            "densities": densities_list,
            "viscosities": viscosities_list,
            "flow_regimes": flow_regimes,
            "reynolds_numbers": reynolds_numbers,
            "phase_counts": phase_counts,
            "liquid_holdups": liquid_holdups,
            "gas_velocities": gas_velocities,
            "liq_velocities": liq_velocities,
        }

    # -----------------------------------------------------------------
    # Steady-state calculation
    # -----------------------------------------------------------------
    if st.button("🔧 Calculate Hydraulics", type="primary", key="ss_run"):
        if edited_fluid["MolarComposition[-]"].sum() <= 0:
            st.error("Please enter a valid fluid composition.")
        else:
            with st.spinner("Running pipeline hydraulics calculation..."):
                try:
                    from neqsim.thermo import fluid_df, TPflash, dataFrame
                    from neqsim import jneqsim

                    id_m = ss_diameter / 1000.0
                    roughness_m = ss_roughness / 1e6
                    seg_length = ss_length / ss_segments
                    seg_elevation = ss_elevation / ss_segments

                    # Calculate mass flow rate
                    neqsim_fluid = fluid_df(edited_fluid, lastIsPlusFraction=False, add_all_components=False)
                    neqsim_fluid.setPressure(ss_outlet_P, "bara")
                    neqsim_fluid.setTemperature(ss_inlet_T + 273.15, "K")
                    TPflash(neqsim_fluid)
                    neqsim_fluid.initProperties()
                    rho = neqsim_fluid.getDensity("kg/m3")

                    if ss_flow_unit == "MSm3/day":
                        std_fluid = fluid_df(edited_fluid, lastIsPlusFraction=False, add_all_components=False)
                        std_fluid.setPressure(1.01325, "bara")
                        std_fluid.setTemperature(15.0 + 273.15, "K")
                        TPflash(std_fluid)
                        std_fluid.initProperties()
                        rho_std = std_fluid.getDensity("kg/m3")
                        vol_flow_std = ss_flow * 1e6 / 86400
                        mass_flow_kgs = vol_flow_std * rho_std
                    elif ss_flow_unit == "kg/s":
                        mass_flow_kgs = ss_flow
                    else:
                        mass_flow_kgs = ss_flow / 3600.0 * rho

                    # -----------------------------------------------------------
                    # Bisection: find inlet P such that outlet P ≈ ss_outlet_P
                    # -----------------------------------------------------------
                    P_lo = ss_outlet_P            # minimum possible inlet P
                    P_hi = ss_outlet_P + 500.0    # generous upper bound
                    target_P = ss_outlet_P
                    tol = 0.01                    # convergence tolerance (bar)
                    max_iter = 40

                    # Quick check: even with the upper bound the dP may be too large
                    res_hi = _march_forward(
                        P_hi, edited_fluid, ss_inlet_T, mass_flow_kgs,
                        id_m, roughness_m, ss_segments, seg_length, seg_elevation,
                        ss_htc, ss_ambient)
                    if res_hi["pressures"][-1] < target_P:
                        # Need even higher upper bound
                        P_hi = ss_outlet_P + 2000.0

                    converged = False
                    for _ in range(max_iter):
                        P_mid = (P_lo + P_hi) / 2.0
                        res = _march_forward(
                            P_mid, edited_fluid, ss_inlet_T, mass_flow_kgs,
                            id_m, roughness_m, ss_segments, seg_length, seg_elevation,
                            ss_htc, ss_ambient)
                        calc_out_P = res["pressures"][-1]
                        if abs(calc_out_P - target_P) < tol:
                            converged = True
                            break
                        if calc_out_P > target_P:
                            P_hi = P_mid     # inlet P is too high
                        else:
                            P_lo = P_mid     # inlet P is too low

                    if not converged:
                        st.warning(
                            f"Inlet pressure solver did not fully converge. "
                            f"Outlet P = {res['pressures'][-1]:.2f} bara "
                            f"(target {target_P:.2f} bara). Results shown for best estimate.")

                    ss_inlet_P = P_mid  # solved inlet pressure
                    positions = res["positions"]
                    pressures = res["pressures"]
                    temperatures = res["temperatures"]
                    velocities = res["velocities"]
                    densities = res["densities"]
                    viscosities = res["viscosities"]
                    flow_regimes = res["flow_regimes"]
                    reynolds_numbers = res["reynolds_numbers"]
                    phase_counts = res["phase_counts"]
                    liquid_holdups = res["liquid_holdups"]
                    gas_velocities = res["gas_velocities"]
                    liq_velocities = res["liq_velocities"]

                    # Check for multiphase at inlet
                    inlet_fluid = fluid_df(edited_fluid, lastIsPlusFraction=False, add_all_components=False)
                    inlet_fluid.setPressure(ss_inlet_P, "bara")
                    inlet_fluid.setTemperature(ss_inlet_T + 273.15, "K")
                    TPflash(inlet_fluid)
                    inlet_fluid.initProperties()
                    n_phases = inlet_fluid.getNumberOfPhases()
                    if n_phases > 1:
                        phase_names = []
                        for pi in range(n_phases):
                            try:
                                phase_names.append(str(inlet_fluid.getPhase(pi).getPhaseTypeName()))
                            except Exception:
                                phase_names.append(f"Phase {pi}")
                        st.info(f"Multiphase flow detected at inlet: {n_phases} phases "
                                f"({', '.join(phase_names)}). "
                                "Two-phase flow regime determined using the Beggs-Brill flow pattern map.")

                    # --- results ---
                    st.subheader("📊 Results Summary")
                    cr1, cr2, cr3, cr4 = st.columns(4)
                    total_dp = ss_inlet_P - pressures[-1]
                    avg_vel = np.mean(velocities) if velocities else 0
                    cr1.metric("Inlet Pressure (bara)", f"{ss_inlet_P:.2f}")
                    cr2.metric("Pressure Drop (bar)", f"{total_dp:.2f}")
                    cr3.metric("Temperature Drop (°C)", f"{ss_inlet_T - temperatures[-1]:.2f}")
                    cr4.metric("Avg Velocity (m/s)", f"{avg_vel:.2f}")
                    cr5, cr6, cr7, cr8 = st.columns(4)
                    cr5.metric("Outlet Temperature (°C)", f"{temperatures[-1]:.2f}")
                    cr6.metric("Flow Regime", flow_regimes[-1] if flow_regimes else "N/A")
                    cr7.metric("Avg Reynolds",
                               f"{np.mean(reynolds_numbers):,.0f}" if reynolds_numbers else "N/A")
                    cr8.metric("Mass Flow (kg/s)", f"{mass_flow_kgs:.2f}")
                    st.divider()

                    mid_pos = [(i + 0.5) * seg_length / 1000 for i in range(ss_segments)]

                    fig = make_subplots(rows=1, cols=2,
                                        subplot_titles=("Pressure Profile", "Temperature Profile"))
                    fig.add_trace(go.Scatter(x=[p / 1000 for p in positions], y=pressures,
                                             mode="lines", line=dict(color="#2196F3", width=2)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=[p / 1000 for p in positions], y=temperatures,
                                             mode="lines", line=dict(color="#F44336", width=2)), row=1, col=2)
                    fig.update_xaxes(title_text="Distance (km)")
                    fig.update_yaxes(title_text="Pressure (bara)", row=1, col=1)
                    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=2)
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                    fig2 = make_subplots(rows=1, cols=2,
                                         subplot_titles=("Velocity Profile", "Density Profile"))
                    fig2.add_trace(go.Scatter(x=mid_pos, y=velocities, mode="lines",
                                              line=dict(color="#4CAF50", width=2)), row=1, col=1)
                    fig2.add_trace(go.Scatter(x=mid_pos, y=densities, mode="lines",
                                              line=dict(color="#FF9800", width=2)), row=1, col=2)
                    fig2.update_xaxes(title_text="Distance (km)")
                    fig2.update_yaxes(title_text="Velocity (m/s)", row=1, col=1)
                    fig2.update_yaxes(title_text="Density (kg/m³)", row=1, col=2)
                    fig2.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig2, use_container_width=True)

                    st.subheader("📋 Segment Data")
                    seg_table = pd.DataFrame({
                        "Distance (km)": [round(p / 1000, 3) for p in mid_pos],
                        "Pressure (bara)": [round(p, 2) for p in pressures[1:]],
                        "Temperature (°C)": [round(t, 2) for t in temperatures[1:]],
                        "Velocity (m/s)": [round(v, 3) for v in velocities],
                        "Vsg (m/s)": gas_velocities,
                        "Vsl (m/s)": liq_velocities,
                        "Density (kg/m³)": [round(d, 2) for d in densities],
                        "Reynolds": [f"{r:,.0f}" for r in reynolds_numbers],
                        "Phases": phase_counts,
                        "Liq. Holdup": liquid_holdups,
                        "Flow Regime": flow_regimes,
                    })
                    st.dataframe(seg_table, use_container_width=True, hide_index=True)

                    # Erosional velocity check — every segment
                    erosion_exceeded = []
                    min_margin, min_margin_idx = float('inf'), 0
                    for i in range(len(velocities)):
                        v_eros = 122.0 / np.sqrt(densities[i]) if densities[i] > 0 else 50
                        margin = 1 - velocities[i] / v_eros if v_eros > 0 else 1
                        if margin < min_margin:
                            min_margin = margin; min_margin_idx = i
                        if velocities[i] > v_eros:
                            erosion_exceeded.append((mid_pos[i], velocities[i], v_eros))
                    if erosion_exceeded:
                        worst = max(erosion_exceeded, key=lambda x: x[1])
                        st.warning(
                            f"⚠️ Erosional velocity exceeded in {len(erosion_exceeded)} of "
                            f"{len(velocities)} segments! Worst at {worst[0]:.1f} km: "
                            f"{worst[1]:.1f} m/s vs limit {worst[2]:.1f} m/s.")
                    else:
                        v_at = velocities[min_margin_idx]
                        v_lim = 122.0 / np.sqrt(densities[min_margin_idx])
                        st.info(
                            f"Erosional velocity OK along entire pipeline. "
                            f"Tightest margin: {min_margin * 100:.0f}% at "
                            f"{mid_pos[min_margin_idx]:.1f} km "
                            f"(vel {v_at:.1f} m/s, limit {v_lim:.1f} m/s).")

                except Exception as e:
                    st.error(f"Calculation failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())


# #####################################################################
#  TAB 2 — DYNAMIC SIMULATION  (NeqSim TwoFluidPipe — OLGA-style)
# #####################################################################
with tab_dyn:
    st.markdown("""
    **NeqSim Two-Fluid Model** — OLGA-style transient multiphase pipeline
    simulation.  Solves separate conservation equations for gas and liquid
    phases with interfacial friction, Lagrangian slug tracking, and terrain
    effects.
    """)

    # =================================================================
    #  Pipeline elevation profile
    # =================================================================
    st.subheader("🗺️ Pipeline Elevation Profile")

    profile_presets = {
        "Flat": {"Distance (m)": [0.0, 5000.0, 10000.0],
                 "Elevation (m)": [0.0, 0.0, 0.0]},
        "Uphill": {"Distance (m)": [0.0, 5000.0, 10000.0],
                   "Elevation (m)": [0.0, 50.0, 100.0]},
        "Downhill": {"Distance (m)": [0.0, 5000.0, 10000.0],
                     "Elevation (m)": [0.0, -50.0, -100.0]},
        "Undulating": {"Distance (m)": [0.0, 2000.0, 4000.0, 6000.0, 8000.0, 10000.0],
                       "Elevation (m)": [0.0, -30.0, 10.0, -50.0, -20.0, 0.0]},
        "V-shape (Dip)": {"Distance (m)": [0.0, 3000.0, 5000.0, 7000.0, 10000.0],
                          "Elevation (m)": [0.0, -80.0, -100.0, -60.0, 0.0]},
        "Riser": {"Distance (m)": [0.0, 7000.0, 8000.0, 10000.0],
                  "Elevation (m)": [0.0, -200.0, -200.0, 0.0]},
    }

    preset = st.selectbox("Profile Preset", list(profile_presets.keys()), key="dyn_preset")

    if 'dyn_profile_df' not in st.session_state or st.session_state.get('_dyn_last_preset') != preset:
        st.session_state.dyn_profile_df = pd.DataFrame(profile_presets[preset])
        st.session_state._dyn_last_preset = preset

    profile_df = st.data_editor(
        st.session_state.dyn_profile_df,
        column_config={
            "Distance (m)": st.column_config.NumberColumn("Distance (m)", min_value=0.0, format="%.0f"),
            "Elevation (m)": st.column_config.NumberColumn("Elevation (m)", format="%.1f"),
        },
        num_rows="dynamic",
        use_container_width=True,
        key="dyn_profile_editor",
    )

    # Profile chart
    if len(profile_df) >= 2:
        fig_prof = go.Figure()
        fig_prof.add_trace(go.Scatter(
            x=[d / 1000 for d in profile_df["Distance (m)"]],
            y=profile_df["Elevation (m)"],
            mode="lines+markers",
            fill="tozeroy",
            line=dict(color="#1976D2", width=3),
            marker=dict(size=8),
            name="Elevation",
        ))
        fig_prof.update_layout(
            title="Pipeline Elevation Profile",
            xaxis_title="Distance (km)",
            yaxis_title="Elevation (m)",
            height=280,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig_prof, use_container_width=True)

    st.divider()

    # =================================================================
    #  Configuration
    # =================================================================
    st.subheader("⚙️ Configuration")
    dc1, dc2, dc3 = st.columns(3)

    with dc1:
        st.markdown("**Pipe Geometry**")
        dyn_diam_mm = st.number_input("Inner Diameter (mm)", value=200.0, min_value=10.0, step=10.0, key="dyn_diam")
        dyn_rough_um = st.number_input("Roughness (μm)", value=50.0, min_value=0.0, step=5.0, key="dyn_rough")
        dyn_nsec = st.number_input("Number of Sections", value=50, min_value=10, max_value=500, key="dyn_nsec")

    with dc2:
        st.markdown("**Inlet Conditions**")
        dyn_inP = st.number_input("Inlet Pressure (bara)", value=80.0, min_value=1.0, step=5.0, key="dyn_inP")
        dyn_inT = st.number_input("Inlet Temperature (°C)", value=40.0, step=5.0, key="dyn_inT")
        dyn_base_flow = st.number_input("Base Flow Rate (kg/hr)", value=5000.0, min_value=1.0, step=100.0,
                                        key="dyn_base_flow")

    with dc3:
        st.markdown("**Outlet & Environment**")
        dyn_outP = st.number_input("Outlet Pressure (bara)", value=50.0, min_value=1.0, step=5.0, key="dyn_outP")
        dyn_ambT = st.number_input("Ambient / Surface Temp (°C)", value=5.0, step=1.0, key="dyn_ambT")
        dyn_htc = st.number_input("Overall HTC (W/m²·K)", value=5.0, min_value=0.0, step=1.0,
                                  key="dyn_htc", help="0 = adiabatic")

    st.divider()

    # Flow rate slider + simulation controls
    st.subheader("🎛️ Flow Rate & Simulation")
    sc1, sc2 = st.columns([2, 1])
    with sc1:
        dyn_flow_pct = st.slider(
            "Feed Flow Rate Adjustment",
            min_value=0, max_value=300, value=100, step=5,
            format="%d %%",
            key="dyn_flow_pct",
            help="Adjust as percentage of base flow rate. "
                 "Change this and click Run to see the effect.",
        )
        effective_flow_kghr = dyn_base_flow * dyn_flow_pct / 100.0
        st.caption(f"Effective flow rate: **{effective_flow_kghr:,.0f} kg/hr** "
                   f"({effective_flow_kghr / 3600:.2f} kg/s)")

    with sc2:
        dyn_sim_time = st.number_input("Simulation Time (s)", value=600.0,
                                       min_value=10.0, step=60.0, key="dyn_simtime")
        dyn_steps = st.number_input("Live Display Steps", value=10,
                                    min_value=2, max_value=100, key="dyn_steps")
        dyn_slug = st.checkbox("Enable Slug Tracking", value=True, key="dyn_slug")

    st.divider()

    # =================================================================
    #  Run Dynamic Simulation
    # =================================================================
    if st.button("🌊 Run Dynamic Simulation", type="primary", key="dyn_run"):
        if edited_fluid["MolarComposition[-]"].sum() <= 0:
            st.error("Please enter a valid fluid composition.")
        elif len(profile_df) < 2:
            st.error("Pipeline profile needs at least 2 points.")
        else:
            try:
                from neqsim.thermo import fluid_df, TPflash
                from neqsim import jneqsim

                # ---- create inlet stream ----
                neqsim_fluid = fluid_df(edited_fluid, lastIsPlusFraction=False, add_all_components=False)
                Stream = jneqsim.process.equipment.stream.Stream
                inlet = Stream("Inlet", neqsim_fluid)
                inlet.setFlowRate(float(effective_flow_kghr), "kg/hr")
                inlet.setTemperature(float(dyn_inT), "C")
                inlet.setPressure(float(dyn_inP), "bara")
                inlet.run()

                # ---- create TwoFluidPipe ----
                TwoFluidPipe = jneqsim.process.equipment.pipeline.TwoFluidPipe
                pipe = TwoFluidPipe("Pipeline", inlet)

                distances = [float(d) for d in profile_df["Distance (m)"].tolist()]
                elevations = [float(e) for e in profile_df["Elevation (m)"].tolist()]
                pipe_length = max(distances) - min(distances)

                pipe.setLength(pipe_length)
                pipe.setDiameter(dyn_diam_mm / 1000.0)
                pipe.setNumberOfSections(int(dyn_nsec))
                pipe.setOutletPressure(float(dyn_outP), "bara")
                pipe.setRoughness(dyn_rough_um / 1e6)

                # Elevation profile
                try:
                    pipe.setElevationProfile(distances, elevations)
                except Exception:
                    total_elev = elevations[-1] - elevations[0]
                    pipe.setElevation(total_elev)

                # Heat transfer
                if dyn_htc > 0:
                    pipe.setHeatTransferCoefficient(float(dyn_htc))
                    pipe.setSurfaceTemperature(float(dyn_ambT), "C")

                # Slug tracking
                if dyn_slug:
                    pipe.setEnableSlugTracking(True)

                # ==========================================================
                #  STEADY-STATE  (initialisation)
                # ==========================================================
                with st.spinner("Running steady-state initialisation..."):
                    pipe.run()

                st.success(f"Steady-state converged. "
                           f"ΔP = {pipe.getInletPressure() - pipe.getOutletPressure():.2f} bar, "
                           f"Outlet T = {pipe.getOutletTemperature() - 273.15:.1f} °C")

                # Helper — read current profiles from pipe
                def _read_profiles(p):
                    pos = [x / 1000 for x in list(p.getPositionProfile())]
                    pres = [x / 1e5 for x in list(p.getPressureProfile())]
                    temp = [x - 273.15 for x in list(p.getTemperatureProfile())]
                    hold = list(p.getLiquidHoldupProfile())
                    gv = list(p.getGasVelocityProfile())
                    lv = list(p.getLiquidVelocityProfile())
                    reg = [str(r) for r in list(p.getFlowRegimeProfile())]
                    return pos, pres, temp, hold, gv, lv, reg

                pos, pres_ss, temp_ss, hold_ss, gv_ss, lv_ss, reg_ss = _read_profiles(pipe)

                # --- Steady-state summary ---
                st.subheader("📊 Steady-State Results")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Inlet P (bara)", f"{pres_ss[0]:.1f}")
                m2.metric("Outlet P (bara)", f"{pres_ss[-1]:.1f}")
                m3.metric("Outlet T (°C)", f"{temp_ss[-1]:.1f}")
                m4.metric("Flow Rate (kg/hr)", f"{effective_flow_kghr:,.0f}")

                m5, m6, m7, m8 = st.columns(4)
                m5.metric("ΔP (bar)", f"{pres_ss[0] - pres_ss[-1]:.2f}")
                m6.metric("Avg Holdup", f"{np.mean(hold_ss):.4f}")
                try:
                    m7.metric("Liq. Inventory (m³)", f"{pipe.getLiquidInventory():.3f}")
                except Exception:
                    m7.metric("Liq. Inventory (m³)", "N/A")
                mid_idx = len(reg_ss) // 2
                m8.metric("Mid-pipe Regime", reg_ss[mid_idx] if reg_ss else "N/A")

                # Steady-state profiles
                fig_ss = make_subplots(rows=2, cols=2,
                                       subplot_titles=("Pressure", "Temperature",
                                                        "Liquid Holdup", "Phase Velocities"))
                fig_ss.add_trace(go.Scatter(x=pos, y=pres_ss, mode="lines",
                                            line=dict(color="#2196F3", width=2), showlegend=False),
                                row=1, col=1)
                fig_ss.add_trace(go.Scatter(x=pos, y=temp_ss, mode="lines",
                                            line=dict(color="#F44336", width=2), showlegend=False),
                                row=1, col=2)
                fig_ss.add_trace(go.Scatter(x=pos, y=hold_ss, mode="lines",
                                            fill="tozeroy",
                                            line=dict(color="#4CAF50", width=2), showlegend=False),
                                row=2, col=1)
                fig_ss.add_trace(go.Scatter(x=pos, y=gv_ss, mode="lines", name="Gas",
                                            line=dict(color="#FF9800", width=2)),
                                row=2, col=2)
                fig_ss.add_trace(go.Scatter(x=pos, y=lv_ss, mode="lines", name="Liquid",
                                            line=dict(color="#2196F3", width=2)),
                                row=2, col=2)
                fig_ss.update_xaxes(title_text="Distance (km)")
                fig_ss.update_yaxes(title_text="bara", row=1, col=1)
                fig_ss.update_yaxes(title_text="°C", row=1, col=2)
                fig_ss.update_yaxes(title_text="Holdup (–)", row=2, col=1)
                fig_ss.update_yaxes(title_text="m/s", row=2, col=2)
                fig_ss.update_layout(height=650, legend=dict(x=0.75, y=0.35))
                st.plotly_chart(fig_ss, use_container_width=True)

                # Flow regime by section
                with st.expander("📋 Steady-State Segment Data"):
                    ss_df = pd.DataFrame({
                        "Distance (km)": [round(p, 2) for p in pos],
                        "P (bara)": [round(p, 2) for p in pres_ss],
                        "T (°C)": [round(t, 1) for t in temp_ss],
                        "Holdup": [round(h, 4) for h in hold_ss],
                        "Vg (m/s)": [round(v, 3) for v in gv_ss],
                        "Vl (m/s)": [round(v, 3) for v in lv_ss],
                        "Flow Regime": reg_ss,
                    })
                    st.dataframe(ss_df, use_container_width=True, hide_index=True)

                # ==========================================================
                #  TRANSIENT SIMULATION  — live updates
                # ==========================================================
                st.divider()
                st.subheader("🌊 Transient Simulation")
                st.markdown(f"Running **{dyn_sim_time:.0f} s** in {dyn_steps} display steps…")

                dt_step = dyn_sim_time / dyn_steps

                progress = st.progress(0)
                time_label = st.empty()
                live_chart = st.empty()
                holdup_bar = st.empty()
                accum_chart = st.empty()
                slug_box = st.empty()

                # History for time-series
                t_hist = [0.0]
                outP_hist = [pres_ss[-1]]
                outT_hist = [temp_ss[-1]]
                inv_hist = []
                try:
                    inv_hist.append(float(pipe.getLiquidInventory()))
                except Exception:
                    inv_hist.append(0.0)

                for step in range(dyn_steps):
                    pipe.runTransient(dt_step)
                    t_now = (step + 1) * dt_step
                    t_hist.append(t_now)

                    pos_t, pres_t, temp_t, hold_t, gv_t, lv_t, reg_t = _read_profiles(pipe)

                    outP_hist.append(pres_t[-1])
                    outT_hist.append(temp_t[-1])
                    try:
                        inv_hist.append(float(pipe.getLiquidInventory()))
                    except Exception:
                        inv_hist.append(inv_hist[-1])

                    progress.progress((step + 1) / dyn_steps)
                    time_label.markdown(f"**t = {t_now:.0f} s** / {dyn_sim_time:.0f} s")

                    # ---- live pressure + holdup profiles ----
                    with live_chart.container():
                        fig_l = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=(f"Pressure  (t = {t_now:.0f} s)",
                                            f"Liquid Holdup  (t = {t_now:.0f} s)"))
                        fig_l.add_trace(go.Scatter(x=pos_t, y=pres_t, mode="lines",
                                                    line=dict(color="#2196F3", width=2),
                                                    showlegend=False), row=1, col=1)
                        fig_l.add_trace(go.Scatter(x=pos_t, y=hold_t, mode="lines",
                                                    fill="tozeroy",
                                                    line=dict(color="#4CAF50", width=2),
                                                    showlegend=False), row=1, col=2)
                        fig_l.update_xaxes(title_text="Distance (km)")
                        fig_l.update_yaxes(title_text="bara", row=1, col=1)
                        fig_l.update_yaxes(title_text="Holdup (–)", row=1, col=2)
                        fig_l.update_layout(height=350, showlegend=False)
                        st.plotly_chart(fig_l, use_container_width=True)

                    # ---- holdup bar colouring ----
                    with holdup_bar.container():
                        colours = ["#F44336" if h > 0.5 else "#4CAF50" if h > 0.1 else "#90CAF9"
                                   for h in hold_t]
                        fig_hb = go.Figure(go.Bar(x=pos_t, y=hold_t,
                                                   marker_color=colours))
                        fig_hb.update_layout(
                            title=f"Liquid Holdup by Section (t = {t_now:.0f} s) — "
                                  f"🔴 > 0.5  🟢 > 0.1  🔵 < 0.1",
                            xaxis_title="Distance (km)",
                            yaxis_title="Holdup (–)",
                            height=280)
                        st.plotly_chart(fig_hb, use_container_width=True)

                    # ---- liquid accumulation & outlet conditions ----
                    with accum_chart.container():
                        fig_a = make_subplots(
                            rows=1, cols=3,
                            subplot_titles=("Liquid Inventory",
                                            "Outlet Pressure",
                                            "Outlet Temperature"))
                        fig_a.add_trace(go.Scatter(x=t_hist, y=inv_hist, mode="lines+markers",
                                                    line=dict(color="#FF9800", width=2),
                                                    showlegend=False), row=1, col=1)
                        fig_a.add_trace(go.Scatter(x=t_hist, y=outP_hist, mode="lines+markers",
                                                    line=dict(color="#2196F3", width=2),
                                                    showlegend=False), row=1, col=2)
                        fig_a.add_trace(go.Scatter(x=t_hist, y=outT_hist, mode="lines+markers",
                                                    line=dict(color="#F44336", width=2),
                                                    showlegend=False), row=1, col=3)
                        fig_a.update_xaxes(title_text="Time (s)")
                        fig_a.update_yaxes(title_text="m³", row=1, col=1)
                        fig_a.update_yaxes(title_text="bara", row=1, col=2)
                        fig_a.update_yaxes(title_text="°C", row=1, col=3)
                        fig_a.update_layout(height=300)
                        st.plotly_chart(fig_a, use_container_width=True)

                    # ---- slug statistics ----
                    if dyn_slug:
                        with slug_box.container():
                            try:
                                slug_txt = str(pipe.getSlugStatisticsSummary())
                                st.text(slug_txt)
                            except Exception:
                                st.caption("Slug tracking data not available.")

                # ==========================================================
                #  FINAL RESULTS
                # ==========================================================
                st.success(f"Transient simulation complete — {dyn_sim_time:.0f} s")

                st.subheader("📋 Final State")
                pos_f, pres_f, temp_f, hold_f, gv_f, lv_f, reg_f = _read_profiles(pipe)
                final_df = pd.DataFrame({
                    "Distance (km)": [round(p, 2) for p in pos_f],
                    "P (bara)": [round(p, 2) for p in pres_f],
                    "T (°C)": [round(t, 1) for t in temp_f],
                    "Holdup": [round(h, 4) for h in hold_f],
                    "Vg (m/s)": [round(v, 3) for v in gv_f],
                    "Vl (m/s)": [round(v, 3) for v in lv_f],
                    "Flow Regime": reg_f,
                })
                st.dataframe(final_df, use_container_width=True, hide_index=True)

                # Final phase velocity chart
                fig_fin = make_subplots(rows=1, cols=2,
                                        subplot_titles=("Phase Velocities (final)",
                                                         "Flow Regime Map (final)"))
                fig_fin.add_trace(go.Scatter(x=pos_f, y=gv_f, mode="lines",
                                             name="Gas", line=dict(color="#FF9800", width=2)),
                                 row=1, col=1)
                fig_fin.add_trace(go.Scatter(x=pos_f, y=lv_f, mode="lines",
                                             name="Liquid", line=dict(color="#2196F3", width=2)),
                                 row=1, col=1)

                # Flow regime as categorical colour
                regime_set = sorted(set(reg_f))
                cmap = {"STRATIFIED_SMOOTH": "#90CAF9", "STRATIFIED_WAVY": "#42A5F5",
                        "SLUG": "#F44336", "ANNULAR": "#FF9800",
                        "DISPERSED_BUBBLE": "#4CAF50", "MIST": "#9E9E9E"}
                for rg in regime_set:
                    mask_x = [pos_f[i] for i in range(len(reg_f)) if reg_f[i] == rg]
                    mask_y = [1] * len(mask_x)
                    fig_fin.add_trace(go.Bar(
                        x=mask_x, y=mask_y, name=rg,
                        marker_color=cmap.get(rg, "#9C27B0"),
                        showlegend=True,
                    ), row=1, col=2)
                fig_fin.update_xaxes(title_text="Distance (km)")
                fig_fin.update_yaxes(title_text="m/s", row=1, col=1)
                fig_fin.update_yaxes(visible=False, row=1, col=2)
                fig_fin.update_layout(height=400, barmode="stack",
                                      legend=dict(orientation="h", y=-0.15))
                st.plotly_chart(fig_fin, use_container_width=True)

                # Liquid accumulation summary
                st.subheader("💧 Liquid Accumulation Summary")
                la1, la2, la3 = st.columns(3)
                la1.metric("Final Liq. Inventory (m³)", f"{inv_hist[-1]:.3f}")
                la2.metric("Max Liq. Inventory (m³)", f"{max(inv_hist):.3f}")
                la3.metric("Inventory Change (m³)",
                           f"{inv_hist[-1] - inv_hist[0]:+.3f}")

                # Holdup heatmap over time would require storing all snapshots —
                # for now show the max holdup location
                max_hold_idx = int(np.argmax(hold_f))
                st.info(
                    f"Maximum holdup at final state: **{hold_f[max_hold_idx]:.4f}** "
                    f"at {pos_f[max_hold_idx]:.1f} km. "
                    f"Regime: {reg_f[max_hold_idx]}")

            except Exception as e:
                st.error(f"Dynamic simulation failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
