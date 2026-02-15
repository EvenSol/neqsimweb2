# -*- coding: utf-8 -*-
"""
Pipeline Hydraulics Page
========================
Single-phase and two-phase pipeline pressure/temperature drop calculations
using NeqSim fluidmechanics package.
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
Calculate **pressure drop**, **temperature profile**, and **flow regime** for single-phase
and two-phase flow in pipelines using NeqSim's fluid mechanics package.

Supports horizontal, inclined, and vertical pipes with roughness and heat transfer effects.
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
# Main — Pipeline Parameters
# =============================================================================
st.subheader("⚙️ Pipeline Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Geometry**")
    pipe_length_m = st.number_input("Pipe Length (m)", value=10000.0, min_value=1.0, step=100.0)
    inner_diameter_mm = st.number_input("Inner Diameter (mm)", value=200.0, min_value=10.0, step=10.0)
    roughness_um = st.number_input("Roughness (μm)", value=50.0, min_value=0.0, step=5.0)
    elevation_m = st.number_input("Total Elevation Change (m)", value=0.0, step=10.0,
                                  help="Positive = uphill, Negative = downhill")
    num_segments = st.number_input("Number of Segments", value=20, min_value=5, max_value=200)

with col2:
    st.markdown("**Inlet Conditions**")
    inlet_pressure_bara = st.number_input("Inlet Pressure (bara)", value=80.0, min_value=1.0, step=5.0)
    inlet_temp_C = st.number_input("Inlet Temperature (°C)", value=40.0, step=5.0)
    flow_rate = st.number_input("Flow Rate", value=10.0, min_value=0.001, step=1.0)
    flow_unit = st.selectbox("Flow Unit", ["MSm3/day", "kg/s", "m3/hr"])

with col3:
    st.markdown("**Environment / Heat Transfer**")
    ambient_temp_C = st.number_input("Ambient Temperature (°C)", value=5.0, step=1.0)
    overall_htc = st.number_input("Overall HTC (W/m²·K)", value=5.0, min_value=0.0, step=1.0,
                                  help="0 = adiabatic")
    pipe_material = st.selectbox("Pipe Material", ["Carbon Steel", "Stainless Steel", "Duplex SS", "CRA Lined"])

st.divider()

# =============================================================================
# Calculation
# =============================================================================
if st.button("🔧 Calculate Hydraulics", type="primary"):
    if edited_fluid["MolarComposition[-]"].sum() <= 0:
        st.error("Please enter a valid fluid composition.")
    else:
        with st.spinner("Running pipeline hydraulics calculation..."):
            try:
                from neqsim.thermo import fluid_df, TPflash, dataFrame
                from neqsim import jneqsim

                # Create fluid
                neqsim_fluid = fluid_df(edited_fluid, lastIsPlusFraction=False, add_all_components=False)
                neqsim_fluid.setPressure(inlet_pressure_bara, "bara")
                neqsim_fluid.setTemperature(inlet_temp_C + 273.15, "K")
                TPflash(neqsim_fluid)
                neqsim_fluid.initProperties()

                # Determine number of phases
                n_phases = neqsim_fluid.getNumberOfPhases()
                if n_phases > 1:
                    phase_names = []
                    for pi in range(n_phases):
                        try:
                            phase_names.append(str(neqsim_fluid.getPhase(pi).getPhaseTypeName()))
                        except Exception:
                            phase_names.append(f"Phase {pi}")
                    st.info(f"Multiphase flow detected at inlet: {n_phases} phases ({', '.join(phase_names)}). "
                            "Two-phase flow regime is determined using the Beggs-Brill flow pattern map.")

                # Inner diameter in meters
                id_m = inner_diameter_mm / 1000.0
                roughness_m = roughness_um / 1e6

                # Calculate flow properties at inlet
                rho = neqsim_fluid.getDensity("kg/m3")
                try:
                    mu = neqsim_fluid.getViscosity()  # returns Pa·s
                except Exception:
                    mu = 1e-5
                mw = neqsim_fluid.getMolarMass("kg/mol")

                # Convert flow rate to mass flow
                if flow_unit == "MSm3/day":
                    # Standard conditions: 15°C, 1 atm
                    std_fluid = fluid_df(edited_fluid, lastIsPlusFraction=False, add_all_components=False)
                    std_fluid.setPressure(1.01325, "bara")
                    std_fluid.setTemperature(15.0 + 273.15, "K")
                    TPflash(std_fluid)
                    std_fluid.initProperties()
                    rho_std = std_fluid.getDensity("kg/m3")
                    vol_flow_std = flow_rate * 1e6 / 86400  # Sm3/s
                    mass_flow_kgs = vol_flow_std * rho_std
                elif flow_unit == "kg/s":
                    mass_flow_kgs = flow_rate
                else:  # m3/hr
                    mass_flow_kgs = flow_rate / 3600.0 * rho

                # Pipe cross-section area
                area = np.pi / 4 * id_m ** 2

                # Segment calculation
                seg_length = pipe_length_m / num_segments
                seg_elevation = elevation_m / num_segments
                seg_angle_rad = np.arcsin(np.clip(seg_elevation / seg_length, -1, 1)) if seg_length > 0 else 0

                positions = [0]
                pressures = [inlet_pressure_bara]
                temperatures = [inlet_temp_C]
                velocities = []
                densities = []
                viscosities = []
                flow_regimes = []
                reynolds_numbers = []
                phase_counts = []
                liquid_holdups = []
                gas_velocities = []
                liq_velocities = []

                current_P = inlet_pressure_bara
                current_T = inlet_temp_C

                for seg in range(num_segments):
                    # Create fluid at current conditions
                    seg_fluid = fluid_df(edited_fluid, lastIsPlusFraction=False, add_all_components=False)
                    seg_fluid.setPressure(current_P, "bara")
                    seg_fluid.setTemperature(current_T + 273.15, "K")
                    TPflash(seg_fluid)
                    seg_fluid.initProperties()

                    seg_rho = seg_fluid.getDensity("kg/m3")
                    try:
                        seg_mu = seg_fluid.getViscosity()  # returns Pa·s
                    except Exception:
                        seg_mu = 1e-5

                    # Velocity
                    vel = mass_flow_kgs / (seg_rho * area) if seg_rho > 0 else 0

                    # Reynolds number
                    Re = seg_rho * vel * id_m / seg_mu if seg_mu > 0 else 1e6

                    # Friction factor (Colebrook-White approximation, Haaland equation)
                    rel_rough = roughness_m / id_m
                    if Re > 2300:
                        ff = (-1.8 * np.log10((rel_rough / 3.7) ** 1.11 + 6.9 / Re)) ** (-2)
                    elif Re > 0:
                        ff = 64 / Re
                    else:
                        ff = 0.02

                    # Flow regime detection and per-phase velocities
                    seg_n_phases = seg_fluid.getNumberOfPhases()
                    lambda_l = 0.0  # liquid volume fraction (no-slip holdup)
                    vsg = 0.0  # superficial gas velocity
                    vsl = 0.0  # superficial liquid velocity
                    if seg_n_phases > 1:
                        # Multi-phase: determine flow pattern (Beggs-Brill)
                        try:
                            gas_vol = 0.0
                            liq_vol = 0.0
                            gas_mass = 0.0
                            liq_mass = 0.0
                            for pi in range(seg_n_phases):
                                phase = seg_fluid.getPhase(pi)
                                pvol = phase.getVolume("m3")
                                prho = phase.getDensity("kg/m3")
                                ptype = str(phase.getPhaseTypeName()).lower()
                                if "gas" in ptype:
                                    gas_vol += pvol
                                    gas_mass += pvol * prho
                                else:
                                    liq_vol += pvol
                                    liq_mass += pvol * prho
                            total_vol = gas_vol + liq_vol
                            total_mass = gas_mass + liq_mass
                            lambda_l = liq_vol / total_vol if total_vol > 0 else 0.5

                            # Superficial velocities (phase volumetric flow / pipe area)
                            if total_mass > 0 and area > 0:
                                gas_mass_frac = gas_mass / total_mass
                                liq_mass_frac = liq_mass / total_mass
                                gas_rho_phase = gas_mass / gas_vol if gas_vol > 0 else seg_rho
                                liq_rho_phase = liq_mass / liq_vol if liq_vol > 0 else seg_rho
                                vsg = (mass_flow_kgs * gas_mass_frac / gas_rho_phase) / area if gas_rho_phase > 0 else 0
                                vsl = (mass_flow_kgs * liq_mass_frac / liq_rho_phase) / area if liq_rho_phase > 0 else 0

                            # Beggs-Brill flow pattern map
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
                                # Nearly single phase by volume
                                regime = "Turbulent" if Re > 2300 else "Laminar"
                        except Exception:
                            regime = "Multiphase"
                    else:
                        # Single-phase: entire flow is one phase
                        ptype = str(seg_fluid.getPhase(0).getPhaseTypeName()).lower()
                        if "gas" in ptype:
                            vsg = vel
                            vsl = 0.0
                        else:
                            vsg = 0.0
                            vsl = vel
                        if Re > 2300:
                            regime = "Turbulent"
                        elif Re > 0:
                            regime = "Laminar"
                        else:
                            regime = "Unknown"

                    # Frictional pressure drop (Pa)
                    dP_friction = ff * (seg_length / id_m) * 0.5 * seg_rho * vel ** 2

                    # Gravitational pressure drop (Pa)
                    dP_gravity = seg_rho * 9.81 * seg_elevation

                    # Total dP in bara
                    dP_total_bara = (dP_friction + dP_gravity) / 1e5

                    # Temperature change (simplified: heat transfer to ambient)
                    cp = 2000.0  # J/(kg·K) approximate
                    try:
                        cp = seg_fluid.getCp("J/kg/K")
                    except Exception:
                        pass
                    if overall_htc > 0 and mass_flow_kgs > 0:
                        perimeter = np.pi * id_m
                        Q_heat = overall_htc * perimeter * seg_length * (current_T - ambient_temp_C)
                        dT = Q_heat / (mass_flow_kgs * cp) if cp > 0 else 0
                    else:
                        dT = 0

                    # Update conditions
                    current_P = max(current_P - dP_total_bara, 1.0)
                    current_T = current_T - dT

                    # Store results
                    distance = (seg + 1) * seg_length
                    positions.append(distance)
                    pressures.append(current_P)
                    temperatures.append(current_T)
                    velocities.append(vel)
                    densities.append(seg_rho)
                    viscosities.append(seg_mu)
                    flow_regimes.append(regime)
                    reynolds_numbers.append(Re)
                    phase_counts.append(seg_n_phases)
                    liquid_holdups.append(round(lambda_l, 4))
                    gas_velocities.append(round(vsg, 3))
                    liq_velocities.append(round(vsl, 3))

                # =============================================================
                # Results Display
                # =============================================================
                st.subheader("📊 Results Summary")

                col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                total_dp = inlet_pressure_bara - pressures[-1]
                total_dt = inlet_temp_C - temperatures[-1]
                avg_vel = np.mean(velocities) if velocities else 0
                final_regime = flow_regimes[-1] if flow_regimes else "N/A"

                col_r1.metric("Pressure Drop (bar)", f"{total_dp:.2f}")
                col_r2.metric("Outlet Pressure (bara)", f"{pressures[-1]:.2f}")
                col_r3.metric("Temperature Drop (°C)", f"{total_dt:.2f}")
                col_r4.metric("Avg Velocity (m/s)", f"{avg_vel:.2f}")

                col_r5, col_r6, col_r7, col_r8 = st.columns(4)
                col_r5.metric("Outlet Temperature (°C)", f"{temperatures[-1]:.2f}")
                col_r6.metric("Flow Regime", final_regime)
                col_r7.metric("Avg Reynolds", f"{np.mean(reynolds_numbers):,.0f}" if reynolds_numbers else "N/A")
                col_r8.metric("Mass Flow (kg/s)", f"{mass_flow_kgs:.2f}")

                st.divider()

                # =============================================================
                # Pressure & Temperature Profile Charts
                # =============================================================
                st.subheader("📈 Pressure & Temperature Profiles")

                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Pressure Profile", "Temperature Profile"),
                )

                fig.add_trace(
                    go.Scatter(
                        x=[p / 1000 for p in positions],
                        y=pressures,
                        mode="lines",
                        name="Pressure",
                        line=dict(color="#2196F3", width=2),
                    ),
                    row=1, col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=[p / 1000 for p in positions],
                        y=temperatures,
                        mode="lines",
                        name="Temperature",
                        line=dict(color="#F44336", width=2),
                    ),
                    row=1, col=2,
                )

                fig.update_xaxes(title_text="Distance (km)", row=1, col=1)
                fig.update_xaxes(title_text="Distance (km)", row=1, col=2)
                fig.update_yaxes(title_text="Pressure (bara)", row=1, col=1)
                fig.update_yaxes(title_text="Temperature (°C)", row=1, col=2)
                fig.update_layout(height=450, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                # =============================================================
                # Velocity & Density Profiles
                # =============================================================
                st.subheader("📈 Velocity & Density Profiles")
                mid_positions = [(i + 0.5) * seg_length / 1000 for i in range(num_segments)]

                fig2 = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Velocity Profile", "Density Profile"),
                )
                fig2.add_trace(
                    go.Scatter(
                        x=mid_positions, y=velocities,
                        mode="lines", name="Velocity",
                        line=dict(color="#4CAF50", width=2),
                    ),
                    row=1, col=1,
                )
                fig2.add_trace(
                    go.Scatter(
                        x=mid_positions, y=densities,
                        mode="lines", name="Density",
                        line=dict(color="#FF9800", width=2),
                    ),
                    row=1, col=2,
                )
                fig2.update_xaxes(title_text="Distance (km)", row=1, col=1)
                fig2.update_xaxes(title_text="Distance (km)", row=1, col=2)
                fig2.update_yaxes(title_text="Velocity (m/s)", row=1, col=1)
                fig2.update_yaxes(title_text="Density (kg/m³)", row=1, col=2)
                fig2.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

                # =============================================================
                # Detailed Results Table
                # =============================================================
                st.subheader("📋 Segment Data")
                seg_table = pd.DataFrame({
                    "Distance (km)": [round(p / 1000, 3) for p in mid_positions],
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

                # Erosional velocity check — evaluate at every segment
                erosion_exceeded = []
                min_margin = float('inf')
                min_margin_idx = 0
                for i in range(len(velocities)):
                    seg_rho_i = densities[i]
                    v_eros_i = 122.0 / np.sqrt(seg_rho_i) if seg_rho_i > 0 else 50
                    margin_i = 1 - velocities[i] / v_eros_i if v_eros_i > 0 else 1
                    if margin_i < min_margin:
                        min_margin = margin_i
                        min_margin_idx = i
                    if velocities[i] > v_eros_i:
                        erosion_exceeded.append((mid_positions[i], velocities[i], v_eros_i))

                if erosion_exceeded:
                    worst = max(erosion_exceeded, key=lambda x: x[1])
                    st.warning(
                        f"⚠️ Erosional velocity exceeded in {len(erosion_exceeded)} of {len(velocities)} segments! "
                        f"Worst at {worst[0]:.1f} km: {worst[1]:.1f} m/s vs limit {worst[2]:.1f} m/s."
                    )
                else:
                    v_at = velocities[min_margin_idx]
                    v_lim = 122.0 / np.sqrt(densities[min_margin_idx])
                    st.info(
                        f"Erosional velocity OK along entire pipeline. "
                        f"Tightest margin: {min_margin * 100:.0f}% at {mid_positions[min_margin_idx]:.1f} km "
                        f"(vel {v_at:.1f} m/s, limit {v_lim:.1f} m/s)."
                    )

            except Exception as e:
                st.error(f"Calculation failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
