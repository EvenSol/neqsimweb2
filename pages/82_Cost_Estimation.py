# -*- coding: utf-8 -*-
"""
Cost Estimation Page
====================
Equipment CAPEX estimation using Turton et al. correlations,
OPEX estimation, and financial metrics (NPV, ROI, payback).
Uses NeqSim process simulation results for equipment sizing.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from theme import apply_theme

st.set_page_config(
    page_title="Cost Estimation",
    page_icon='images/neqsimlogocircleflat.png',
    layout="wide",
)
apply_theme()

st.title('💰 Cost Estimation')

st.markdown("""
Estimate **capital cost (CAPEX)** and **operating cost (OPEX)** for process equipment
using industry-standard correlations (Turton et al., Seider et al.).

Supports 14+ equipment types with material/pressure correction factors,
location factors for global regions, and financial metrics (NPV, ROI, payback period).
""")

st.divider()

# =============================================================================
# Constants & Correlations
# =============================================================================

# Turton et al. base cost correlations: log10(Cp0) = K1 + K2*log10(A) + K3*(log10(A))^2
# A = capacity parameter (varies by equipment), Cp0 in USD (CEPCI 397, year 2001)
EQUIPMENT_CORRELATIONS = {
    "Centrifugal Compressor": {
        "K1": 2.2891, "K2": 1.3604, "K3": -0.1027,
        "A_unit": "kW (shaft power)", "A_min": 75, "A_max": 30000,
        "description": "Centrifugal compressor (driver not included)",
    },
    "Reciprocating Compressor": {
        "K1": 2.2891, "K2": 1.3604, "K3": -0.1027,
        "A_unit": "kW (shaft power)", "A_min": 75, "A_max": 30000,
        "description": "Reciprocating compressor (driver not included)",
    },
    "Centrifugal Pump": {
        "K1": 3.3892, "K2": 0.0536, "K3": 0.1538,
        "A_unit": "kW (shaft power)", "A_min": 1, "A_max": 300,
        "description": "Centrifugal pump + motor",
    },
    "Electric Motor": {
        "K1": 2.4604, "K2": 1.4191, "K3": -0.1798,
        "A_unit": "kW", "A_min": 1, "A_max": 2500,
        "description": "Enclosed electric motor",
    },
    "Shell & Tube Heat Exchanger": {
        "K1": 4.3247, "K2": -0.3030, "K3": 0.1634,
        "A_unit": "m² (heat transfer area)", "A_min": 10, "A_max": 1000,
        "description": "Fixed-head shell & tube heat exchanger",
    },
    "Air Cooler": {
        "K1": 4.0336, "K2": 0.2341, "K3": 0.0497,
        "A_unit": "m² (face area)", "A_min": 10, "A_max": 10000,
        "description": "Air-cooled heat exchanger",
    },
    "Horizontal Pressure Vessel": {
        "K1": 3.5565, "K2": 0.3776, "K3": 0.0905,
        "A_unit": "m³ (volume)", "A_min": 0.1, "A_max": 628,
        "description": "Horizontal pressure vessel / separator",
    },
    "Vertical Pressure Vessel": {
        "K1": 3.4974, "K2": 0.4485, "K3": 0.1074,
        "A_unit": "m³ (volume)", "A_min": 0.1, "A_max": 628,
        "description": "Vertical pressure vessel / separator",
    },
    "Tray Column": {
        "K1": 3.4974, "K2": 0.4485, "K3": 0.1074,
        "A_unit": "m³ (volume)", "A_min": 0.3, "A_max": 520,
        "description": "Tray distillation column (shell only)",
    },
    "Sieve Tray": {
        "K1": 2.9949, "K2": 0.4465, "K3": 0.3961,
        "A_unit": "m² (tray area)", "A_min": 0.07, "A_max": 12.3,
        "description": "Sieve tray (per tray)",
    },
    "Valve (Control)": {
        "K1": 3.1052, "K2": 0.1738, "K3": 0.0024,
        "A_unit": "inch (valve diameter)", "A_min": 1, "A_max": 24,
        "description": "Control valve with actuator",
    },
    "Pipeline (Carbon Steel)": {
        "K1": 2.0, "K2": 1.0, "K3": 0.0,
        "A_unit": "m (length × 0.01 diameter)", "A_min": 1, "A_max": 100000,
        "description": "Carbon steel pipeline (installed)",
    },
    "Plate Heat Exchanger": {
        "K1": 4.6656, "K2": -0.1557, "K3": 0.1547,
        "A_unit": "m² (heat transfer area)", "A_min": 1, "A_max": 500,
        "description": "Gasketed plate heat exchanger",
    },
    "Fired Heater": {
        "K1": 1.9950, "K2": 1.1280, "K3": 0.0120,
        "A_unit": "kW (duty)", "A_min": 100, "A_max": 100000,
        "description": "Box-type fired heater",
    },
}

# Material correction factors (Fm)
MATERIAL_FACTORS = {
    "Carbon Steel": 1.0,
    "Stainless Steel 304": 1.7,
    "Stainless Steel 316": 2.1,
    "Monel 400": 3.2,
    "Inconel 600": 3.8,
    "Nickel 200": 4.5,
    "Titanium": 7.7,
    "Duplex SS": 2.9,
    "22Cr Duplex": 3.1,
    "25Cr Super Duplex": 3.6,
}

# Location factors
LOCATION_FACTORS = {
    "US Gulf Coast (base)": 1.0,
    "North Sea (Norway)": 1.65,
    "North Sea (UK)": 1.45,
    "Middle East": 0.85,
    "Southeast Asia": 0.90,
    "West Africa": 1.30,
    "Australia": 1.40,
    "Brazil": 1.15,
    "China": 0.75,
    "India": 0.70,
    "Canada": 1.15,
}

# CEPCI Index values for cost year escalation
CEPCI_INDICES = {
    2001: 397.0,
    2010: 550.8,
    2015: 556.8,
    2018: 603.1,
    2020: 596.2,
    2021: 708.0,
    2022: 816.0,
    2023: 797.9,
    2024: 810.0,
    2025: 825.0,  # estimated
}


def calc_base_cost(K1: float, K2: float, K3: float, A: float) -> float:
    """Calculate purchased equipment cost (Cp0) using Turton correlation."""
    log_A = np.log10(max(A, 1e-6))
    log_Cp0 = K1 + K2 * log_A + K3 * log_A ** 2
    return 10 ** log_Cp0


def calc_pressure_factor(P_barg: float, eq_type: str) -> float:
    """Simplified pressure correction factor (Fp)."""
    if P_barg <= 5:
        return 1.0
    elif P_barg <= 10:
        return 1.1
    elif P_barg <= 20:
        return 1.2
    elif P_barg <= 50:
        return 1.5
    elif P_barg <= 100:
        return 1.9
    elif P_barg <= 200:
        return 2.3
    else:
        return 3.0


def calc_bare_module_cost(Cp0: float, Fm: float, Fp: float, B1: float = 1.89, B2: float = 1.35) -> float:
    """Calculate bare module cost: CBM = Cp0 * (B1 + B2 * Fm * Fp)."""
    return Cp0 * (B1 + B2 * Fm * Fp)


def escalate_cost(cost_base: float, year_base: int, year_target: int) -> float:
    """Escalate cost using CEPCI indices."""
    cepci_base = CEPCI_INDICES.get(year_base, 397.0)
    cepci_target = CEPCI_INDICES.get(year_target, CEPCI_INDICES[max(CEPCI_INDICES.keys())])
    return cost_base * cepci_target / cepci_base


# =============================================================================
# Sidebar — Global Settings
# =============================================================================
with st.sidebar:
    st.header("⚙️ Global Settings")

    cost_year = st.selectbox("Cost Year", sorted(CEPCI_INDICES.keys(), reverse=True), index=0)
    location = st.selectbox("Location", list(LOCATION_FACTORS.keys()), index=1)
    loc_factor = LOCATION_FACTORS[location]

    currency = st.selectbox("Display Currency", ["USD", "EUR", "NOK", "GBP"], index=0)
    fx_rates = {"USD": 1.0, "EUR": 0.92, "NOK": 10.7, "GBP": 0.79}
    fx = fx_rates[currency]

    st.divider()
    st.subheader("💼 Financial Parameters")
    project_lifetime_yr = st.number_input("Project Lifetime (years)", value=20, min_value=1, max_value=50)
    discount_rate_pct = st.number_input("Discount Rate (%)", value=8.0, min_value=0.0, max_value=30.0)
    annual_revenue_musd = st.number_input(f"Annual Revenue (M {currency})", value=50.0, min_value=0.0)

    st.divider()
    st.subheader("🔧 OPEX Factors")
    maintenance_pct = st.number_input("Maintenance (% of CAPEX/yr)", value=3.0, min_value=0.0, max_value=15.0)
    insurance_pct = st.number_input("Insurance (% of CAPEX/yr)", value=1.0, min_value=0.0, max_value=5.0)
    electricity_usd_kwh = st.number_input(f"Electricity Cost ({currency}/kWh)", value=0.08, min_value=0.0, format="%.3f")
    operating_hours_yr = st.number_input("Operating Hours/Year", value=8400, min_value=1000, max_value=8760)

# =============================================================================
# Main — Equipment List Editor
# =============================================================================

st.subheader("📋 Equipment List")

st.markdown("Add equipment items and specify their capacity parameter. "
            "The tool will estimate purchased cost, bare module cost, and total installed cost.")

# Initialize equipment table
if 'cost_equipment' not in st.session_state:
    st.session_state.cost_equipment = pd.DataFrame({
        "Equipment Name": ["K-100 Compressor", "E-100 Cooler", "V-100 Separator"],
        "Equipment Type": ["Centrifugal Compressor", "Shell & Tube Heat Exchanger", "Vertical Pressure Vessel"],
        "Capacity (A)": [2000.0, 150.0, 5.0],
        "Operating Pressure (barg)": [80.0, 10.0, 60.0],
        "Material": ["Carbon Steel", "Stainless Steel 316", "Carbon Steel"],
        "Quantity": [1, 1, 1],
    })

eq_types = list(EQUIPMENT_CORRELATIONS.keys())
materials = list(MATERIAL_FACTORS.keys())

edited_eq = st.data_editor(
    st.session_state.cost_equipment,
    column_config={
        "Equipment Name": st.column_config.TextColumn("Name", width="medium"),
        "Equipment Type": st.column_config.SelectboxColumn("Type", options=eq_types, width="large"),
        "Capacity (A)": st.column_config.NumberColumn("Capacity (A)", min_value=0.001, format="%.1f",
                                                       help="Size parameter — see unit for each type"),
        "Operating Pressure (barg)": st.column_config.NumberColumn("P (barg)", min_value=0, format="%.1f"),
        "Material": st.column_config.SelectboxColumn("Material", options=materials),
        "Quantity": st.column_config.NumberColumn("Qty", min_value=1, max_value=50, default=1),
    },
    num_rows="dynamic",
    use_container_width=True,
)

# Show capacity parameter units
with st.expander("📏 Capacity Parameter Reference"):
    ref_data = []
    for name, corr in EQUIPMENT_CORRELATIONS.items():
        ref_data.append({
            "Equipment Type": name,
            "Capacity Unit (A)": corr["A_unit"],
            "Min A": corr["A_min"],
            "Max A": corr["A_max"],
            "Description": corr["description"],
        })
    st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)

st.divider()

# =============================================================================
# Calculations
# =============================================================================
if st.button("💰 Calculate Costs", type="primary"):
    if edited_eq.empty or len(edited_eq) == 0:
        st.error("Please add at least one equipment item.")
    else:
        with st.spinner("Estimating costs..."):
            results = []
            total_capex = 0.0
            total_power_kw = 0.0

            for _, row in edited_eq.iterrows():
                name = row.get("Equipment Name", "Unknown")
                eq_type = row.get("Equipment Type", "")
                A = float(row.get("Capacity (A)", 1.0))
                P_barg = float(row.get("Operating Pressure (barg)", 0.0))
                mat = row.get("Material", "Carbon Steel")
                qty = int(row.get("Quantity", 1))

                corr = EQUIPMENT_CORRELATIONS.get(eq_type)
                if corr is None:
                    st.warning(f"No correlation for '{eq_type}', skipping {name}")
                    continue

                # Base cost (USD 2001)
                Cp0_2001 = calc_base_cost(corr["K1"], corr["K2"], corr["K3"], A)

                # Escalate to target year
                Cp0 = escalate_cost(Cp0_2001, 2001, cost_year)

                # Correction factors
                Fm = MATERIAL_FACTORS.get(mat, 1.0)
                Fp = calc_pressure_factor(P_barg, eq_type)

                # Bare module cost
                CBM = calc_bare_module_cost(Cp0, Fm, Fp)

                # Total module cost (with contingency + fee)
                CTM = CBM * 1.18  # 18% for contingency + contractor fee

                # Apply location factor and quantity
                installed_cost = CTM * loc_factor * qty

                # Convert currency
                installed_local = installed_cost * fx

                total_capex += installed_cost * fx

                # Track power for OPEX
                if "Compressor" in eq_type or "Pump" in eq_type or "Motor" in eq_type:
                    total_power_kw += A * qty

                results.append({
                    "Equipment": name,
                    "Type": eq_type,
                    "Qty": qty,
                    f"Purchased Cost ({currency})": round(Cp0 * fx * qty, 0),
                    "Fm": round(Fm, 2),
                    "Fp": round(Fp, 2),
                    f"Bare Module ({currency})": round(CBM * fx * qty, 0),
                    f"Installed Cost ({currency})": round(installed_local, 0),
                })

            results_df = pd.DataFrame(results)

            # =============================================================================
            # Results Display
            # =============================================================================
            st.subheader("📊 Equipment Cost Breakdown")
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            # Summary metrics
            st.divider()
            col1, col2, col3, col4 = st.columns(4)

            total_capex_M = total_capex / 1e6
            annual_opex_maintenance = total_capex * maintenance_pct / 100
            annual_opex_insurance = total_capex * insurance_pct / 100
            annual_opex_power = total_power_kw * electricity_usd_kwh * operating_hours_yr * fx
            annual_opex = annual_opex_maintenance + annual_opex_insurance + annual_opex_power
            annual_opex_M = annual_opex / 1e6

            col1.metric(f"Total CAPEX (M {currency})", f"{total_capex_M:.2f}")
            col2.metric(f"Annual OPEX (M {currency})", f"{annual_opex_M:.2f}")
            col3.metric("Total Power (kW)", f"{total_power_kw:,.0f}")
            col4.metric("Location Factor", f"{loc_factor:.2f}")

            # =============================================================================
            # CAPEX Breakdown Chart
            # =============================================================================
            st.subheader("📈 CAPEX Breakdown")
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=results_df["Equipment"],
                        y=results_df[f"Installed Cost ({currency})"],
                        marker_color="#2196F3",
                    )
                ])
                fig_bar.update_layout(
                    title="Installed Cost by Equipment",
                    xaxis_title="Equipment",
                    yaxis_title=f"Cost ({currency})",
                    height=400,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_chart2:
                fig_pie = go.Figure(data=[
                    go.Pie(
                        labels=results_df["Equipment"],
                        values=results_df[f"Installed Cost ({currency})"],
                        hole=0.4,
                    )
                ])
                fig_pie.update_layout(title="Cost Distribution", height=400)
                st.plotly_chart(fig_pie, use_container_width=True)

            # =============================================================================
            # OPEX Breakdown
            # =============================================================================
            st.subheader("🔧 Annual OPEX Breakdown")
            opex_data = {
                "Category": ["Maintenance", "Insurance", "Electricity"],
                f"Annual Cost ({currency})": [
                    round(annual_opex_maintenance, 0),
                    round(annual_opex_insurance, 0),
                    round(annual_opex_power, 0),
                ],
            }
            st.dataframe(pd.DataFrame(opex_data), use_container_width=True, hide_index=True)

            # =============================================================================
            # Financial Metrics
            # =============================================================================
            st.subheader("📉 Financial Metrics")
            r = discount_rate_pct / 100.0
            annual_revenue = annual_revenue_musd * 1e6
            annual_cash_flow = annual_revenue - annual_opex

            # NPV
            npv = -total_capex
            for yr in range(1, project_lifetime_yr + 1):
                npv += annual_cash_flow / (1 + r) ** yr

            # Payback period
            cumulative = -total_capex
            payback = None
            for yr in range(1, project_lifetime_yr + 1):
                cumulative += annual_cash_flow
                if cumulative >= 0 and payback is None:
                    payback = yr

            # ROI
            total_profit = annual_cash_flow * project_lifetime_yr
            roi = (total_profit / total_capex * 100) if total_capex > 0 else 0

            col_f1, col_f2, col_f3, col_f4 = st.columns(4)
            col_f1.metric(f"NPV (M {currency})", f"{npv / 1e6:.2f}")
            col_f2.metric("ROI (%)", f"{roi:.1f}")
            col_f3.metric("Payback (years)", f"{payback if payback else '>'+str(project_lifetime_yr)}")
            col_f4.metric(f"Annual Cash Flow (M {currency})", f"{annual_cash_flow / 1e6:.2f}")

            # Cash flow chart
            years = list(range(0, project_lifetime_yr + 1))
            cum_cf = [-total_capex]
            for yr in range(1, project_lifetime_yr + 1):
                cum_cf.append(cum_cf[-1] + annual_cash_flow / (1 + r) ** yr)

            fig_cf = go.Figure()
            fig_cf.add_trace(go.Scatter(
                x=years, y=[c / 1e6 for c in cum_cf],
                mode="lines+markers",
                name="Cumulative Discounted Cash Flow",
                line=dict(color="#4CAF50", width=2),
            ))
            fig_cf.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
            fig_cf.update_layout(
                title="Cumulative Discounted Cash Flow",
                xaxis_title="Year",
                yaxis_title=f"Cumulative DCF (M {currency})",
                height=400,
            )
            st.plotly_chart(fig_cf, use_container_width=True)

        st.success("Cost estimation complete!")
