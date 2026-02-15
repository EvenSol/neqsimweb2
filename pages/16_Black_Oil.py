# -*- coding: utf-8 -*-
"""
Black Oil / Eclipse Export Page
===============================
Generate black oil PVT tables (Bo, Rs, Bg, μo, μg) from compositional
fluid models and export in Eclipse PVTO/PVDG format.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from theme import apply_theme
from fluids import default_fluid

st.set_page_config(
    page_title="Black Oil PVT Tables",
    page_icon='images/neqsimlogocircleflat.png',
    layout="wide",
)
apply_theme()

st.title('🛢️ Black Oil PVT Tables')
st.markdown("""
Generate **black oil PVT tables** from compositional fluid models using NeqSim.

The tool performs a **Differential Liberation (DL)** simulation to compute:
- **Bo** — Oil Formation Volume Factor (rm³/Sm³)
- **Rs** — Solution Gas-Oil Ratio (Sm³/Sm³)
- **Bg** — Gas Formation Volume Factor (rm³/Sm³)
- **μo** — Oil Viscosity (cP)
- **μg** — Gas Viscosity (cP)
- **ρo** — Oil Density (kg/m³)

Results can be exported in **Eclipse PVTO / PVDG** format.
""")
st.divider()

# =============================================================================
# Sidebar — Reservoir Conditions
# =============================================================================
with st.sidebar:
    st.header("🛢️ Reservoir Conditions")
    reservoir_temp_C = st.number_input("Reservoir Temperature (°C)", value=100.0, min_value=0.0, max_value=300.0)
    reservoir_pres_bara = st.number_input("Reservoir Pressure (bara)", value=300.0, min_value=10.0, max_value=1000.0)
    separator_pres_bara = st.number_input("Separator Pressure (bara)", value=10.0, min_value=1.0, max_value=100.0)
    separator_temp_C = st.number_input("Separator Temperature (°C)", value=25.0, min_value=0.0, max_value=100.0)

    st.divider()
    st.header("⚙️ Simulation Settings")
    num_pressure_steps = st.number_input("Pressure Steps", value=20, min_value=5, max_value=100)
    min_pressure_bara = st.number_input("Minimum Pressure (bara)", value=1.0, min_value=0.5, max_value=50.0)

    st.divider()
    st.header("🧪 Sample Fluids")
    sample_fluid = st.selectbox("Load Sample", ["Custom", "Black Oil", "Gas Condensate", "Volatile Oil"])

# =============================================================================
# Sample Fluid Definitions
# =============================================================================
BLACK_OIL_FLUID = {
    'ComponentName': ['nitrogen', 'CO2', 'methane', 'ethane', 'propane',
                      'i-butane', 'n-butane', 'i-pentane', 'n-pentane',
                      'n-hexane', 'C7', 'C8', 'C9', 'C10'],
    'MolarComposition[-]': [0.005, 0.01, 0.35, 0.07, 0.05,
                             0.02, 0.03, 0.015, 0.015,
                             0.02, 0.08, 0.08, 0.06, 0.195],
    'MolarMass[kg/mol]': [0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0913, 0.1041, 0.1190, 0.220],
    'RelativeDensity[-]': [0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0,
                            0.0, 0.746, 0.768, 0.790, 0.850],
}

GAS_CONDENSATE_FLUID = {
    'ComponentName': ['nitrogen', 'CO2', 'methane', 'ethane', 'propane',
                      'i-butane', 'n-butane', 'i-pentane', 'n-pentane',
                      'n-hexane', 'C7', 'C8', 'C9', 'C10'],
    'MolarComposition[-]': [0.01, 0.02, 0.70, 0.08, 0.04,
                             0.015, 0.02, 0.01, 0.01,
                             0.015, 0.03, 0.02, 0.015, 0.015],
    'MolarMass[kg/mol]': [0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0913, 0.1041, 0.1190, 0.180],
    'RelativeDensity[-]': [0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0,
                            0.0, 0.746, 0.768, 0.790, 0.830],
}

VOLATILE_OIL_FLUID = {
    'ComponentName': ['nitrogen', 'CO2', 'methane', 'ethane', 'propane',
                      'i-butane', 'n-butane', 'i-pentane', 'n-pentane',
                      'n-hexane', 'C7', 'C8', 'C9', 'C10'],
    'MolarComposition[-]': [0.005, 0.015, 0.55, 0.09, 0.06,
                             0.02, 0.025, 0.015, 0.015,
                             0.02, 0.05, 0.04, 0.035, 0.06],
    'MolarMass[kg/mol]': [0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0913, 0.1041, 0.1190, 0.200],
    'RelativeDensity[-]': [0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0,
                            0.0, 0.746, 0.768, 0.790, 0.840],
}

# =============================================================================
# Fluid Composition Input
# =============================================================================
if sample_fluid == "Black Oil":
    init_fluid = BLACK_OIL_FLUID
elif sample_fluid == "Gas Condensate":
    init_fluid = GAS_CONDENSATE_FLUID
elif sample_fluid == "Volatile Oil":
    init_fluid = VOLATILE_OIL_FLUID
else:
    init_fluid = default_fluid

if 'blackoil_fluid_df' not in st.session_state or sample_fluid != st.session_state.get('_blackoil_sample', 'Custom'):
    st.session_state.blackoil_fluid_df = pd.DataFrame(init_fluid)
    st.session_state._blackoil_sample = sample_fluid

with st.expander("📋 Fluid Composition", expanded=True):
    edited_fluid = st.data_editor(
        st.session_state.blackoil_fluid_df,
        column_config={
            "ComponentName": "Component Name",
            "MolarComposition[-]": st.column_config.NumberColumn(
                "Molar Composition [-]", min_value=0, max_value=1, format="%f"
            ),
            "MolarMass[kg/mol]": st.column_config.NumberColumn(
                "Molar Mass [kg/mol]", min_value=0, format="%f"
            ),
            "RelativeDensity[-]": st.column_config.NumberColumn(
                "Density [g/cm³]", min_value=0, format="%f"
            ),
        },
        num_rows='dynamic',
        use_container_width=True,
    )

    isplusfluid = st.checkbox("Last component is plus fraction", value=True)

st.divider()

# =============================================================================
# Run Simulation
# =============================================================================
if st.button("🛢️ Generate Black Oil Tables", type="primary"):
    if edited_fluid["MolarComposition[-]"].sum() <= 0:
        st.error("Please enter a valid fluid composition.")
    else:
        with st.spinner("Running differential liberation simulation..."):
            try:
                from neqsim.thermo import fluid_df, TPflash, dataFrame
                from neqsim import jneqsim

                # Create the fluid
                neqsim_fluid = fluid_df(edited_fluid, lastIsPlusFraction=isplusfluid, add_all_components=False)
                neqsim_fluid.autoSelectModel()

                # Generate pressure array (reservoir → min pressure)
                pressures = np.linspace(reservoir_pres_bara, min_pressure_bara, num_pressure_steps)

                # ============================================================
                # Differential Liberation Simulation
                # ============================================================
                # First: find bubble point
                neqsim_fluid.setPressure(reservoir_pres_bara, "bara")
                neqsim_fluid.setTemperature(reservoir_temp_C, "C")
                TPflash(neqsim_fluid)

                thermoOps = jneqsim.thermodynamicoperations.ThermodynamicOperations(neqsim_fluid)

                # Try to compute saturation pressure (bubble point)
                try:
                    thermoOps.calcSaturationPressure()
                    bubble_point = neqsim_fluid.getPressure("bara")
                except Exception:
                    bubble_point = None

                # Run DL at each pressure
                dl_results = {
                    "Pressure (bara)": [],
                    "Bo (rm3/Sm3)": [],
                    "Rs (Sm3/Sm3)": [],
                    "Oil Density (kg/m3)": [],
                    "Oil Viscosity (cP)": [],
                    "Gas Density (kg/m3)": [],
                    "Gas Viscosity (cP)": [],
                    "Bg (rm3/Sm3)": [],
                    "Z-factor": [],
                }

                # Reset fluid for DL
                dl_fluid = fluid_df(edited_fluid, lastIsPlusFraction=isplusfluid, add_all_components=False)
                dl_fluid.autoSelectModel()
                dl_fluid.setTemperature(reservoir_temp_C, "C")

                # Get standard condition density for Bo and Rs reference
                std_fluid = fluid_df(edited_fluid, lastIsPlusFraction=isplusfluid, add_all_components=False)
                std_fluid.autoSelectModel()
                std_fluid.setPressure(1.01325, "bara")
                std_fluid.setTemperature(15.0, "C")
                TPflash(std_fluid)
                std_fluid.initProperties()

                for P in pressures:
                    dl_fluid.setPressure(P, "bara")
                    dl_fluid.setTemperature(reservoir_temp_C, "C")
                    TPflash(dl_fluid)
                    dl_fluid.initProperties()

                    n_phases = dl_fluid.getNumberOfPhases()

                    # Oil properties
                    try:
                        oil_density = dl_fluid.getPhase("oil").getDensity("kg/m3")
                    except Exception:
                        try:
                            oil_density = dl_fluid.getPhase(0).getDensity("kg/m3")
                        except Exception:
                            oil_density = np.nan

                    try:
                        oil_viscosity = dl_fluid.getPhase("oil").getViscosity("cP")
                    except Exception:
                        try:
                            oil_viscosity = dl_fluid.getPhase(0).getViscosity("cP")
                        except Exception:
                            oil_viscosity = np.nan

                    # Gas properties
                    try:
                        gas_density = dl_fluid.getPhase("gas").getDensity("kg/m3")
                    except Exception:
                        gas_density = np.nan

                    try:
                        gas_viscosity = dl_fluid.getPhase("gas").getViscosity("cP")
                    except Exception:
                        gas_viscosity = np.nan

                    # Bo (simplified: ratio of oil volume at P,T to oil volume at standard conditions)
                    try:
                        oil_vol = dl_fluid.getPhase("oil").getVolume("m3")
                        # Use moles-based Bo
                        Bo = oil_density / dl_fluid.getPhase("oil").getMolarMass("kg/mol") if oil_density else np.nan
                        # Simplified Bo estimate using density ratio
                        Bo = std_fluid.getDensity("kg/m3") / oil_density if oil_density > 0 else np.nan
                    except Exception:
                        Bo = np.nan

                    # Rs (simplified: moles of gas in oil at P / moles of oil)
                    try:
                        if n_phases > 1:
                            gas_vol_std = dl_fluid.getPhase("gas").getVolume("m3") * P / 1.01325
                            oil_vol_std = dl_fluid.getPhase("oil").getVolume("m3")
                            Rs = gas_vol_std / oil_vol_std if oil_vol_std > 0 else 0
                        else:
                            Rs = 0  # No free gas above bubble point (all gas dissolved)
                    except Exception:
                        Rs = 0

                    # Bg
                    try:
                        z_factor = dl_fluid.getPhase("gas").getZ()
                        Bg = z_factor * reservoir_temp_C / (P * 273.15) * 1.01325 if P > 0 else np.nan
                    except Exception:
                        z_factor = np.nan
                        Bg = np.nan

                    dl_results["Pressure (bara)"].append(round(P, 2))
                    dl_results["Bo (rm3/Sm3)"].append(round(Bo, 6) if not np.isnan(Bo) else np.nan)
                    dl_results["Rs (Sm3/Sm3)"].append(round(Rs, 2) if not np.isnan(Rs) else np.nan)
                    dl_results["Oil Density (kg/m3)"].append(round(oil_density, 2) if not np.isnan(oil_density) else np.nan)
                    dl_results["Oil Viscosity (cP)"].append(round(oil_viscosity, 4) if not np.isnan(oil_viscosity) else np.nan)
                    dl_results["Gas Density (kg/m3)"].append(round(gas_density, 4) if not np.isnan(gas_density) else np.nan)
                    dl_results["Gas Viscosity (cP)"].append(round(gas_viscosity, 6) if not np.isnan(gas_viscosity) else np.nan)
                    dl_results["Bg (rm3/Sm3)"].append(round(Bg, 6) if not np.isnan(Bg) else np.nan)
                    dl_results["Z-factor"].append(round(z_factor, 4) if not np.isnan(z_factor) else np.nan)

                dl_df = pd.DataFrame(dl_results)

                # ============================================================
                # Display Results
                # ============================================================
                st.subheader("📊 Differential Liberation Results")

                if bubble_point is not None:
                    st.info(f"Estimated bubble point pressure: **{bubble_point:.2f} bara** at {reservoir_temp_C}°C")

                st.dataframe(dl_df, use_container_width=True, hide_index=True)

                # ============================================================
                # Charts
                # ============================================================
                st.subheader("📈 Black Oil Properties")

                fig = make_subplots(
                    rows=2, cols=3,
                    subplot_titles=(
                        "Bo vs Pressure", "Rs vs Pressure", "Oil Viscosity vs Pressure",
                        "Bg vs Pressure", "Oil Density vs Pressure", "Gas Viscosity vs Pressure",
                    ),
                    vertical_spacing=0.12,
                )

                # Bo
                fig.add_trace(go.Scatter(
                    x=dl_df["Pressure (bara)"], y=dl_df["Bo (rm3/Sm3)"],
                    mode="lines+markers", name="Bo", line=dict(color="#2196F3"),
                ), row=1, col=1)

                # Rs
                fig.add_trace(go.Scatter(
                    x=dl_df["Pressure (bara)"], y=dl_df["Rs (Sm3/Sm3)"],
                    mode="lines+markers", name="Rs", line=dict(color="#4CAF50"),
                ), row=1, col=2)

                # Oil Viscosity
                fig.add_trace(go.Scatter(
                    x=dl_df["Pressure (bara)"], y=dl_df["Oil Viscosity (cP)"],
                    mode="lines+markers", name="μo", line=dict(color="#FF9800"),
                ), row=1, col=3)

                # Bg
                fig.add_trace(go.Scatter(
                    x=dl_df["Pressure (bara)"], y=dl_df["Bg (rm3/Sm3)"],
                    mode="lines+markers", name="Bg", line=dict(color="#9C27B0"),
                ), row=2, col=1)

                # Oil Density
                fig.add_trace(go.Scatter(
                    x=dl_df["Pressure (bara)"], y=dl_df["Oil Density (kg/m3)"],
                    mode="lines+markers", name="ρo", line=dict(color="#F44336"),
                ), row=2, col=2)

                # Gas Viscosity
                fig.add_trace(go.Scatter(
                    x=dl_df["Pressure (bara)"], y=dl_df["Gas Viscosity (cP)"],
                    mode="lines+markers", name="μg", line=dict(color="#795548"),
                ), row=2, col=3)

                for col in range(1, 4):
                    fig.update_xaxes(title_text="Pressure (bara)", row=2, col=col)
                    fig.update_xaxes(title_text="Pressure (bara)", row=1, col=col)

                fig.update_yaxes(title_text="Bo (rm³/Sm³)", row=1, col=1)
                fig.update_yaxes(title_text="Rs (Sm³/Sm³)", row=1, col=2)
                fig.update_yaxes(title_text="μo (cP)", row=1, col=3)
                fig.update_yaxes(title_text="Bg (rm³/Sm³)", row=2, col=1)
                fig.update_yaxes(title_text="ρo (kg/m³)", row=2, col=2)
                fig.update_yaxes(title_text="μg (cP)", row=2, col=3)

                fig.update_layout(height=700, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                # ============================================================
                # Eclipse Export
                # ============================================================
                st.subheader("📤 Eclipse Format Export")

                col_exp1, col_exp2 = st.columns(2)

                with col_exp1:
                    st.markdown("**PVTO (Oil PVT Table)**")
                    pvto_lines = ["PVTO", "-- Rs      Pbub    Bo         Viscosity",
                                  "--  Sm3/Sm3  bara   rm3/Sm3     cP"]
                    valid_oil = dl_df.dropna(subset=["Bo (rm3/Sm3)", "Oil Viscosity (cP)"])
                    for _, row in valid_oil.iterrows():
                        rs_val = row.get("Rs (Sm3/Sm3)", 0)
                        p_val = row["Pressure (bara)"]
                        bo_val = row["Bo (rm3/Sm3)"]
                        muo_val = row["Oil Viscosity (cP)"]
                        pvto_lines.append(f"  {rs_val:10.4f}  {p_val:8.2f}  {bo_val:10.6f}  {muo_val:10.6f} /")
                    pvto_lines.append("/")
                    pvto_text = "\n".join(pvto_lines)
                    st.code(pvto_text, language="text")
                    st.download_button("Download PVTO", pvto_text, file_name="PVTO.inc", mime="text/plain")

                with col_exp2:
                    st.markdown("**PVDG (Gas PVT Table)**")
                    pvdg_lines = ["PVDG", "-- Pres      Bg          Viscosity",
                                  "--  bara     rm3/Sm3       cP"]
                    valid_gas = dl_df.dropna(subset=["Bg (rm3/Sm3)", "Gas Viscosity (cP)"])
                    for _, row in valid_gas.iterrows():
                        p_val = row["Pressure (bara)"]
                        bg_val = row["Bg (rm3/Sm3)"]
                        mug_val = row["Gas Viscosity (cP)"]
                        pvdg_lines.append(f"  {p_val:8.2f}  {bg_val:12.8f}  {mug_val:12.8f}")
                    pvdg_lines.append("/")
                    pvdg_text = "\n".join(pvdg_lines)
                    st.code(pvdg_text, language="text")
                    st.download_button("Download PVDG", pvdg_text, file_name="PVDG.inc", mime="text/plain")

                # Full CSV download
                st.divider()
                csv_data = dl_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Full PVT Table (CSV)",
                    csv_data,
                    file_name="black_oil_pvt.csv",
                    mime="text/csv",
                )

                st.success("Black oil PVT table generation complete!")

            except Exception as e:
                st.error(f"Calculation failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
