import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from neqsim.thermo import fluid, TPflash
from neqsim import jneqsim
from theme import apply_theme, theme_toggle

st.set_page_config(page_title="Compressor Performance", page_icon='images/neqsimlogocircleflat.png')
apply_theme()
theme_toggle()

st.title('Compressor Performance Calculator')

"""
Calculate compressor performance parameters using the **GERG-2008** equation of state.
This tool calculates polytropic head, polytropic efficiency, and power consumption
based on measured operating data (flow rates, pressures, and temperatures).

**Supported fluids:** Standard test fluids (CO2, Methane, Nitrogen) or custom GERG-2008 mixtures.

The GERG-2008 equation of state provides high accuracy for compressibility factor calculations,
which is essential for accurate polytropic head and efficiency calculations.
"""

st.divider()

# GERG-2008 compatible components (18 components)
gerg2008_components = [
    "methane", "nitrogen", "CO2", "ethane", "propane", 
    "i-butane", "n-butane", "i-pentane", "n-pentane", "n-hexane",
    "n-heptane", "n-octane", "hydrogen", "oxygen", "CO", 
    "water", "helium", "argon"
]

# Standard test fluids for compressor testing
test_fluids = {
    "Methane (CH4)": {"methane": 100.0},
    "Carbon Dioxide (CO2)": {"CO2": 100.0},
    "Nitrogen (N2)": {"nitrogen": 100.0},
    "Hydrogen (H2)": {"hydrogen": 100.0},
    "Methane/CO2 Mix (90/10)": {"methane": 90.0, "CO2": 10.0},
    "Methane/N2 Mix (90/10)": {"methane": 90.0, "nitrogen": 10.0},
    "Custom GERG-2008 Mixture": None,  # Special case for custom composition
}

# Default custom fluid composition
default_custom_fluid = {
    'ComponentName': gerg2008_components,
    'MolarComposition[-]': [
        90.0,  # methane
        1.5,   # nitrogen
        2.0,   # CO2
        4.0,   # ethane
        1.5,   # propane
        0.3,   # i-butane
        0.4,   # n-butane
        0.1,   # i-pentane
        0.1,   # n-pentane
        0.05,  # n-hexane
        0.02,  # n-heptane
        0.01,  # n-octane
        0.0,   # hydrogen
        0.0,   # oxygen
        0.0,   # CO
        0.0,   # water
        0.0,   # helium
        0.0    # argon
    ]
}

# Initialize session state for custom fluid
if 'compressor_custom_fluid_df' not in st.session_state:
    st.session_state.compressor_custom_fluid_df = pd.DataFrame(default_custom_fluid)

# Unit options
flow_units = {
    "m3/hr": {"label": "Volume Flow (m³/hr)", "to_m3_per_hr": 1.0},
    "Am3/hr": {"label": "Actual Volume Flow (Am³/hr)", "to_m3_per_hr": 1.0},
    "kg/s": {"label": "Mass Flow (kg/s)", "to_kg_per_s": 1.0},
    "kg/hr": {"label": "Mass Flow (kg/hr)", "to_kg_per_s": 1/3600},
    "MSm3/day": {"label": "Volume Flow (MSm³/day)", "to_Sm3_per_day": 1e6},
}

pressure_units = {
    "bara": {"label": "Pressure (bara)", "to_bara": 1.0},
    "barg": {"label": "Pressure (barg)", "to_bara": lambda x: x + 1.01325},
    "psia": {"label": "Pressure (psia)", "to_bara": lambda x: x * 0.0689476},
    "MPa": {"label": "Pressure (MPa)", "to_bara": lambda x: x * 10.0},
}

temperature_units = {
    "C": {"label": "Temperature (°C)", "to_C": lambda x: x},
    "K": {"label": "Temperature (K)", "to_C": lambda x: x - 273.15},
    "F": {"label": "Temperature (°F)", "to_C": lambda x: (x - 32) * 5/9},
}

# Initialize session state for units
if 'flow_unit' not in st.session_state:
    st.session_state['flow_unit'] = "m3/hr"
if 'pressure_unit' not in st.session_state:
    st.session_state['pressure_unit'] = "bara"
if 'temperature_unit' not in st.session_state:
    st.session_state['temperature_unit'] = "C"

# Initialize session state for operating data
if 'compressor_data' not in st.session_state:
    st.session_state['compressor_data'] = pd.DataFrame({
        'Flow Rate': [1000.0, 1200.0, 1500.0, 1800.0, 2000.0],
        'Inlet Pressure': [50.0, 50.0, 50.0, 50.0, 50.0],
        'Outlet Pressure': [120.0, 120.0, 120.0, 120.0, 120.0],
        'Inlet Temperature': [30.0, 30.0, 30.0, 30.0, 30.0],
        'Outlet Temperature': [95.0, 93.0, 90.0, 88.0, 87.0],
    })

# Sidebar for fluid selection
with st.sidebar:
    st.header("Fluid Selection")
    selected_fluid_name = st.selectbox(
        "Select Test Fluid",
        options=list(test_fluids.keys()),
        index=0
    )
    
    if selected_fluid_name != "Custom GERG-2008 Mixture":
        st.info(f"Selected fluid composition: {test_fluids[selected_fluid_name]}")
    else:
        st.info("Define custom composition in the main panel")

# Helper function to get fluid composition dict
def get_fluid_composition():
    if selected_fluid_name == "Custom GERG-2008 Mixture":
        # Build dict from custom dataframe
        comp_dict = {}
        for idx, row in st.session_state.compressor_custom_fluid_df.iterrows():
            if pd.notna(row['ComponentName']) and row['MolarComposition[-]'] > 0:
                comp_dict[row['ComponentName']] = row['MolarComposition[-]']
        return comp_dict
    else:
        return test_fluids[selected_fluid_name]

# Main content
with st.expander("📋 Fluid Composition", expanded=True):
    st.write(f"**Selected Fluid:** {selected_fluid_name}")
    
    # Show custom fluid editor if custom mixture selected
    if selected_fluid_name == "Custom GERG-2008 Mixture":
        st.write("Define your custom GERG-2008 fluid composition:")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button('Reset to Default', key='reset_custom_fluid'):
                st.session_state.compressor_custom_fluid_df = pd.DataFrame(default_custom_fluid)
                st.rerun()
            
            hidecomponents = st.checkbox('Show only active', key='hide_inactive_comp')
        
        display_df = st.session_state.compressor_custom_fluid_df
        if hidecomponents:
            display_df = display_df[display_df['MolarComposition[-]'] > 0]
        
        edited_fluid_df = st.data_editor(
            display_df,
            column_config={
                "ComponentName": st.column_config.SelectboxColumn(
                    "Component Name",
                    options=gerg2008_components,
                    help="Select from GERG-2008 compatible components"
                ),
                "MolarComposition[-]": st.column_config.NumberColumn(
                    "Molar Composition [-]", 
                    min_value=0, 
                    max_value=100, 
                    format="%.2f",
                    help="Enter molar composition (will be normalized)"
                ),
            },
            num_rows='dynamic',
            key='custom_fluid_editor'
        )
        
        # Update session state
        if not hidecomponents:
            st.session_state.compressor_custom_fluid_df = edited_fluid_df
        
        st.caption("💡 Composition will be normalized before simulation")
    
    # Create GERG-2008 fluid for display
    fluid_composition = get_fluid_composition()
    
    if fluid_composition and len(fluid_composition) > 0:
        try:
            jneqsim.util.database.NeqSimDataBase.setCreateTemporaryTables(True)
            display_fluid = fluid("gerg-2008")
            for comp_name, comp_moles in fluid_composition.items():
                display_fluid.addComponent(comp_name, float(comp_moles))
            display_fluid.setPressure(50.0, 'bara')
            display_fluid.setTemperature(30.0, 'C')
            display_fluid.initThermoProperties()
            
            st.divider()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Molar Mass", f"{display_fluid.getMolarMass()*1000:.2f} g/mol")
            with col2:
                st.metric("Z-factor @ 50 bara, 30°C", f"{display_fluid.getZ():.4f}")
            with col3:
                st.metric("Cp/Cv (κ)", f"{display_fluid.getGamma():.4f}")
        except Exception as e:
            st.warning(f"Could not calculate fluid properties: {e}")
    else:
        st.warning("Please add components to your custom mixture")

st.divider()

with st.expander("📊 Operating Data Input", expanded=True):
    st.write("Enter compressor operating data points. Each row represents a different operating condition.")
    
    # Unit selection
    st.subheader("Select Units")
    col_u1, col_u2, col_u3 = st.columns(3)
    with col_u1:
        selected_flow_unit = st.selectbox(
            "Flow Rate Unit",
            options=list(flow_units.keys()),
            index=list(flow_units.keys()).index(st.session_state['flow_unit']),
            key='flow_unit_select'
        )
        st.session_state['flow_unit'] = selected_flow_unit
    with col_u2:
        selected_pressure_unit = st.selectbox(
            "Pressure Unit",
            options=list(pressure_units.keys()),
            index=list(pressure_units.keys()).index(st.session_state['pressure_unit']),
            key='pressure_unit_select'
        )
        st.session_state['pressure_unit'] = selected_pressure_unit
    with col_u3:
        selected_temp_unit = st.selectbox(
            "Temperature Unit",
            options=list(temperature_units.keys()),
            index=list(temperature_units.keys()).index(st.session_state['temperature_unit']),
            key='temp_unit_select'
        )
        st.session_state['temperature_unit'] = selected_temp_unit
    
    st.divider()
    
    # Reset button
    if st.button('Reset to Example Data'):
        st.session_state['compressor_data'] = pd.DataFrame({
            'Flow Rate': [1000.0, 1200.0, 1500.0, 1800.0, 2000.0],
            'Inlet Pressure': [50.0, 50.0, 50.0, 50.0, 50.0],
            'Outlet Pressure': [120.0, 120.0, 120.0, 120.0, 120.0],
            'Inlet Temperature': [30.0, 30.0, 30.0, 30.0, 30.0],
            'Outlet Temperature': [95.0, 93.0, 90.0, 88.0, 87.0],
        })
        st.session_state['flow_unit'] = "m3/hr"
        st.session_state['pressure_unit'] = "bara"
        st.session_state['temperature_unit'] = "C"
        st.rerun()
    
    # Dynamic column labels based on selected units
    flow_label = f"Flow Rate ({selected_flow_unit})"
    p_in_label = f"Inlet Pressure ({selected_pressure_unit})"
    p_out_label = f"Outlet Pressure ({selected_pressure_unit})"
    t_in_label = f"Inlet Temperature ({selected_temp_unit})"
    t_out_label = f"Outlet Temperature ({selected_temp_unit})"
    
    edited_data = st.data_editor(
        st.session_state['compressor_data'].dropna().reset_index(drop=True),
        num_rows='dynamic',
        column_config={
            'Flow Rate': st.column_config.NumberColumn(
                flow_label,
                min_value=0.01,
                max_value=1000000,
                format='%.2f',
                help=f'Flow rate in {selected_flow_unit}'
            ),
            'Inlet Pressure': st.column_config.NumberColumn(
                p_in_label,
                min_value=0.0,
                max_value=1000,
                format='%.2f',
                help=f'Suction pressure in {selected_pressure_unit}'
            ),
            'Outlet Pressure': st.column_config.NumberColumn(
                p_out_label,
                min_value=0.0,
                max_value=2000,
                format='%.2f',
                help=f'Discharge pressure in {selected_pressure_unit}'
            ),
            'Inlet Temperature': st.column_config.NumberColumn(
                t_in_label,
                min_value=-273.15,
                max_value=500,
                format='%.1f',
                help=f'Suction temperature in {selected_temp_unit}'
            ),
            'Outlet Temperature': st.column_config.NumberColumn(
                t_out_label,
                min_value=-273.15,
                max_value=800,
                format='%.1f',
                help=f'Discharge temperature in {selected_temp_unit}'
            ),
        }
    )
    
    st.session_state['compressor_data'] = edited_data
    
    st.caption(f"📐 Units: Flow = {selected_flow_unit}, Pressure = {selected_pressure_unit}, Temperature = {selected_temp_unit}")

st.divider()

# Helper function for unit conversion
def convert_pressure_to_bara(value, unit):
    if unit == "bara":
        return value
    elif unit == "barg":
        return value + 1.01325
    elif unit == "psia":
        return value * 0.0689476
    elif unit == "MPa":
        return value * 10.0
    return value

def convert_temperature_to_C(value, unit):
    if unit == "C":
        return value
    elif unit == "K":
        return value - 273.15
    elif unit == "F":
        return (value - 32) * 5/9
    return value

# Calculate button
if st.button('Calculate Compressor Performance', type='primary'):
    fluid_composition = get_fluid_composition()
    
    if not fluid_composition or len(fluid_composition) == 0:
        st.error('Please define a valid fluid composition.')
    elif edited_data.empty or edited_data.dropna().empty:
        st.error('Please enter operating data before calculating.')
    else:
        with st.spinner('Calculating compressor performance using GERG-2008...'):
            try:
                jneqsim.util.database.NeqSimDataBase.setCreateTemporaryTables(True)
                
                results = []
                
                # Get selected units
                flow_unit = st.session_state['flow_unit']
                pressure_unit = st.session_state['pressure_unit']
                temp_unit = st.session_state['temperature_unit']
                
                for idx, row in edited_data.dropna().iterrows():
                    flow_value = row['Flow Rate']
                    p_in_raw = row['Inlet Pressure']
                    p_out_raw = row['Outlet Pressure']
                    t_in_raw = row['Inlet Temperature']
                    t_out_raw = row['Outlet Temperature']
                    
                    # Convert units to standard (bara, C)
                    p_in = convert_pressure_to_bara(p_in_raw, pressure_unit)
                    p_out = convert_pressure_to_bara(p_out_raw, pressure_unit)
                    t_in = convert_temperature_to_C(t_in_raw, temp_unit)
                    t_out = convert_temperature_to_C(t_out_raw, temp_unit)
                    
                    # Create inlet fluid
                    inlet_fluid = fluid("gerg-2008")
                    for comp_name, comp_moles in fluid_composition.items():
                        inlet_fluid.addComponent(comp_name, float(comp_moles))
                    
                    inlet_fluid.setPressure(float(p_in), 'bara')
                    inlet_fluid.setTemperature(float(t_in), 'C')
                    TPflash(inlet_fluid)
                    inlet_fluid.initThermoProperties()
                    inlet_fluid.initPhysicalProperties()
                    
                    # Get inlet properties
                    z_in = inlet_fluid.getZ()
                    h_in = inlet_fluid.getEnthalpy() / inlet_fluid.getNumberOfMoles() / inlet_fluid.getMolarMass() / 1000  # kJ/kg
                    s_in = inlet_fluid.getEntropy() / inlet_fluid.getNumberOfMoles() / inlet_fluid.getMolarMass() / 1000  # kJ/kg/K
                    cp_in = inlet_fluid.getCp() / inlet_fluid.getNumberOfMoles() / inlet_fluid.getMolarMass() / 1000  # kJ/kg/K
                    cv_in = inlet_fluid.getCv() / inlet_fluid.getNumberOfMoles() / inlet_fluid.getMolarMass() / 1000  # kJ/kg/K
                    kappa_in = cp_in / cv_in if cv_in > 0 else inlet_fluid.getGamma()
                    MW = inlet_fluid.getMolarMass() * 1000  # g/mol -> kg/kmol
                    rho_in = inlet_fluid.getDensity()  # kg/m3
                    T_in_K = t_in + 273.15
                    
                    # Convert flow to mass flow (kg/s) based on unit
                    if flow_unit == "kg/s":
                        mass_flow = flow_value
                    elif flow_unit == "kg/hr":
                        mass_flow = flow_value / 3600.0
                    elif flow_unit in ["m3/hr", "Am3/hr"]:
                        # Actual volume flow at inlet conditions
                        mass_flow = flow_value * rho_in / 3600.0  # m3/hr * kg/m3 / 3600 = kg/s
                    elif flow_unit == "MSm3/day":
                        # Standard conditions: 15C, 1.01325 bara
                        std_fluid = fluid("gerg-2008")
                        for comp_name, comp_moles in fluid_composition.items():
                            std_fluid.addComponent(comp_name, float(comp_moles))
                        std_fluid.setPressure(1.01325, 'bara')
                        std_fluid.setTemperature(15.0, 'C')
                        TPflash(std_fluid)
                        std_fluid.initThermoProperties()
                        rho_std = std_fluid.getDensity()  # kg/m3 at std conditions
                        # MSm3/day to kg/s: flow * 1e6 m3/day * rho_std / 86400 s/day
                        mass_flow = flow_value * 1e6 * rho_std / 86400.0
                    else:
                        mass_flow = flow_value  # Default, assume kg/s
                    
                    # Create outlet fluid at actual conditions
                    outlet_fluid = fluid("gerg-2008")
                    for comp_name, comp_moles in fluid_composition.items():
                        outlet_fluid.addComponent(comp_name, float(comp_moles))
                    
                    outlet_fluid.setPressure(float(p_out), 'bara')
                    outlet_fluid.setTemperature(float(t_out), 'C')
                    TPflash(outlet_fluid)
                    outlet_fluid.initThermoProperties()
                    outlet_fluid.initPhysicalProperties()
                    
                    # Get outlet properties
                    z_out = outlet_fluid.getZ()
                    h_out = outlet_fluid.getEnthalpy() / outlet_fluid.getNumberOfMoles() / outlet_fluid.getMolarMass() / 1000  # kJ/kg
                    s_out = outlet_fluid.getEntropy() / outlet_fluid.getNumberOfMoles() / outlet_fluid.getMolarMass() / 1000  # kJ/kg/K
                    cp_out = outlet_fluid.getCp() / outlet_fluid.getNumberOfMoles() / outlet_fluid.getMolarMass() / 1000  # kJ/kg/K
                    cv_out = outlet_fluid.getCv() / outlet_fluid.getNumberOfMoles() / outlet_fluid.getMolarMass() / 1000  # kJ/kg/K
                    kappa_out = cp_out / cv_out if cv_out > 0 else outlet_fluid.getGamma()
                    rho_out = outlet_fluid.getDensity()
                    
                    # Create isentropic outlet fluid (same entropy as inlet)
                    isentropic_fluid = fluid("gerg-2008")
                    for comp_name, comp_moles in fluid_composition.items():
                        isentropic_fluid.addComponent(comp_name, float(comp_moles))
                    isentropic_fluid.setPressure(float(p_out), 'bara')
                    isentropic_fluid.setTemperature(float(t_out), 'C')  # Initial guess
                    
                    # Find isentropic temperature using PS flash
                    thermoOps = jneqsim.thermodynamicoperations.ThermodynamicOperations(isentropic_fluid)
                    s_in_total = inlet_fluid.getEntropy()  # Total entropy at inlet
                    try:
                        thermoOps.PSflash(s_in_total)
                        isentropic_fluid.initThermoProperties()
                        h_out_isen = isentropic_fluid.getEnthalpy() / isentropic_fluid.getNumberOfMoles() / isentropic_fluid.getMolarMass() / 1000  # kJ/kg
                        t_out_isen = isentropic_fluid.getTemperature() - 273.15  # Convert to Celsius
                    except:
                        # Fallback: estimate isentropic temperature
                        kappa_avg = (kappa_in + kappa_out) / 2
                        t_out_isen = T_in_K * (p_out/p_in)**((kappa_avg-1)/kappa_avg) - 273.15
                        h_out_isen = h_in + cp_in * (t_out_isen - t_in)
                    
                    # Calculate actual work (enthalpy change)
                    actual_work = h_out - h_in  # kJ/kg
                    
                    # Calculate isentropic work
                    isentropic_work = h_out_isen - h_in  # kJ/kg
                    
                    # Isentropic efficiency
                    eta_isen = isentropic_work / actual_work if actual_work > 0 else 0
                    
                    # Calculate polytropic efficiency using the Schultz method
                    # Average compressibility factor
                    z_avg = (z_in + z_out) / 2
                    
                    # Average kappa
                    kappa_avg = (kappa_in + kappa_out) / 2
                    
                    # Pressure ratio
                    pr = p_out / p_in
                    
                    # Calculate polytropic exponent from measured data
                    # Using: T2/T1 = (P2/P1)^((n-1)/n)
                    T_out_K = t_out + 273.15
                    if pr > 1 and T_out_K > T_in_K:
                        log_T_ratio = np.log(T_out_K / T_in_K)
                        log_P_ratio = np.log(pr)
                        if log_P_ratio > 0 and log_T_ratio > 0:
                            n_minus_1_over_n = log_T_ratio / log_P_ratio
                            n = 1 / (1 - n_minus_1_over_n) if n_minus_1_over_n < 1 else 1.5
                        else:
                            n = kappa_avg / (kappa_avg - 1 + 0.001) * 0.8  # Estimate
                    else:
                        n = kappa_avg / (kappa_avg - 1 + 0.001) * 0.8  # Estimate
                    
                    # Polytropic efficiency
                    # eta_p = (n-1)/n * k/(k-1)
                    if n > 1 and kappa_avg > 1:
                        eta_poly = ((n - 1) / n) * (kappa_avg / (kappa_avg - 1))
                        eta_poly = min(max(eta_poly, 0.5), 1.0)  # Clamp to reasonable range
                    else:
                        eta_poly = eta_isen * 1.02  # Approximation
                    
                    # Polytropic head calculation using GERG-2008
                    # Hp = z_avg * R * T1 / MW * n/(n-1) * [(P2/P1)^((n-1)/n) - 1]
                    R = 8.314  # J/(mol·K)
                    if n > 1:
                        polytropic_head = z_avg * R * T_in_K / (MW / 1000) * (n / (n - 1)) * (pr**((n - 1) / n) - 1) / 1000  # kJ/kg
                    else:
                        polytropic_head = actual_work * eta_poly  # Fallback
                    
                    # Power calculation
                    power_kW = mass_flow * actual_work  # kW
                    power_MW = power_kW / 1000  # MW
                    
                    # Polytropic power (shaft power required)
                    polytropic_power_kW = mass_flow * polytropic_head / eta_poly if eta_poly > 0 else power_kW
                    
                    # Volume flow at inlet conditions
                    vol_flow_in = mass_flow / rho_in * 3600  # m³/hr
                    
                    results.append({
                        'Flow Rate (kg/s)': mass_flow,
                        'Inlet P (bara)': p_in,
                        'Outlet P (bara)': p_out,
                        'Inlet T (°C)': t_in,
                        'Outlet T (°C)': t_out,
                        'Pressure Ratio': pr,
                        'Z inlet': z_in,
                        'Z outlet': z_out,
                        'κ inlet': kappa_in,
                        'κ outlet': kappa_out,
                        'Polytropic Exp (n)': n,
                        'Isentropic Eff (%)': eta_isen * 100,
                        'Polytropic Eff (%)': eta_poly * 100,
                        'Polytropic Head (kJ/kg)': polytropic_head,
                        'Actual Work (kJ/kg)': actual_work,
                        'Power (kW)': power_kW,
                        'Power (MW)': power_MW,
                        'Vol Flow Inlet (m³/hr)': vol_flow_in,
                    })
                
                results_df = pd.DataFrame(results)
                
                st.success('Compressor performance calculations completed successfully!')
                
                # Display results
                st.subheader("📊 Calculation Results")
                
                # Key metrics display
                if len(results_df) > 0:
                    avg_poly_eff = results_df['Polytropic Eff (%)'].mean()
                    avg_poly_head = results_df['Polytropic Head (kJ/kg)'].mean()
                    total_power = results_df['Power (MW)'].mean()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Avg Polytropic Efficiency", f"{avg_poly_eff:.1f} %")
                    with col2:
                        st.metric("Avg Polytropic Head", f"{avg_poly_head:.1f} kJ/kg")
                    with col3:
                        st.metric("Avg Power", f"{total_power:.2f} MW")
                    with col4:
                        st.metric("Pressure Ratio", f"{results_df['Pressure Ratio'].mean():.2f}")
                
                st.divider()
                
                # Detailed results table
                st.subheader("Detailed Results")
                
                # Format for display
                display_df = results_df.copy()
                display_columns = [
                    'Flow Rate (kg/s)', 'Pressure Ratio', 'Polytropic Eff (%)', 
                    'Polytropic Head (kJ/kg)', 'Power (MW)', 'Z inlet', 'Z outlet',
                    'κ inlet', 'Polytropic Exp (n)', 'Vol Flow Inlet (m³/hr)'
                ]
                
                st.dataframe(
                    display_df[display_columns].style.format({
                        'Flow Rate (kg/s)': '{:.2f}',
                        'Pressure Ratio': '{:.3f}',
                        'Polytropic Eff (%)': '{:.2f}',
                        'Polytropic Head (kJ/kg)': '{:.2f}',
                        'Power (MW)': '{:.3f}',
                        'Z inlet': '{:.4f}',
                        'Z outlet': '{:.4f}',
                        'κ inlet': '{:.4f}',
                        'Polytropic Exp (n)': '{:.4f}',
                        'Vol Flow Inlet (m³/hr)': '{:.1f}',
                    }),
                    use_container_width=True
                )
                
                st.divider()
                
                # Plots
                st.subheader("📈 Performance Curves")
                
                tab1, tab2, tab3 = st.tabs(["Efficiency vs Flow", "Head vs Flow", "Power vs Flow"])
                
                with tab1:
                    fig_eff = go.Figure()
                    fig_eff.add_trace(go.Scatter(
                        x=results_df['Flow Rate (kg/s)'],
                        y=results_df['Polytropic Eff (%)'],
                        mode='lines+markers',
                        name='Polytropic Efficiency',
                        marker=dict(size=10),
                        line=dict(width=2)
                    ))
                    fig_eff.add_trace(go.Scatter(
                        x=results_df['Flow Rate (kg/s)'],
                        y=results_df['Isentropic Eff (%)'],
                        mode='lines+markers',
                        name='Isentropic Efficiency',
                        marker=dict(size=10),
                        line=dict(width=2, dash='dash')
                    ))
                    fig_eff.update_layout(
                        title='Efficiency vs Mass Flow Rate',
                        xaxis_title='Mass Flow Rate (kg/s)',
                        yaxis_title='Efficiency (%)',
                        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_eff, use_container_width=True)
                
                with tab2:
                    fig_head = go.Figure()
                    fig_head.add_trace(go.Scatter(
                        x=results_df['Flow Rate (kg/s)'],
                        y=results_df['Polytropic Head (kJ/kg)'],
                        mode='lines+markers',
                        name='Polytropic Head',
                        marker=dict(size=10, color='green'),
                        line=dict(width=2, color='green')
                    ))
                    fig_head.update_layout(
                        title='Polytropic Head vs Mass Flow Rate',
                        xaxis_title='Mass Flow Rate (kg/s)',
                        yaxis_title='Polytropic Head (kJ/kg)',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_head, use_container_width=True)
                
                with tab3:
                    fig_power = go.Figure()
                    fig_power.add_trace(go.Scatter(
                        x=results_df['Flow Rate (kg/s)'],
                        y=results_df['Power (MW)'],
                        mode='lines+markers',
                        name='Shaft Power',
                        marker=dict(size=10, color='red'),
                        line=dict(width=2, color='red'),
                        fill='tozeroy',
                        fillcolor='rgba(255, 0, 0, 0.1)'
                    ))
                    fig_power.update_layout(
                        title='Power vs Mass Flow Rate',
                        xaxis_title='Mass Flow Rate (kg/s)',
                        yaxis_title='Power (MW)',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_power, use_container_width=True)
                
                # Download results
                st.divider()
                st.subheader("📥 Download Results")
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name=f"compressor_performance_{selected_fluid_name.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Calculation failed: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

# Theory section
with st.expander("📚 Theory & Equations", expanded=False):
    st.markdown("""
    ### Polytropic Head Calculation
    
    The polytropic head is calculated using the GERG-2008 equation of state for accurate 
    compressibility factor determination:
    
    $$H_p = \\frac{Z_{avg} \\cdot R \\cdot T_1}{M_w} \\cdot \\frac{n}{n-1} \\cdot \\left[\\left(\\frac{P_2}{P_1}\\right)^{\\frac{n-1}{n}} - 1\\right]$$
    
    Where:
    - $H_p$ = Polytropic head (kJ/kg)
    - $Z_{avg}$ = Average compressibility factor (inlet + outlet) / 2
    - $R$ = Universal gas constant (8.314 J/mol·K)
    - $T_1$ = Suction temperature (K)
    - $M_w$ = Molecular weight (kg/kmol)
    - $n$ = Polytropic exponent
    - $P_1, P_2$ = Suction and discharge pressures
    
    ### Polytropic Exponent
    
    The polytropic exponent is calculated from measured temperature and pressure data:
    
    $$\\frac{T_2}{T_1} = \\left(\\frac{P_2}{P_1}\\right)^{\\frac{n-1}{n}}$$
    
    ### Polytropic Efficiency
    
    $$\\eta_p = \\frac{n-1}{n} \\cdot \\frac{\\kappa}{\\kappa-1}$$
    
    Where $\\kappa = C_p/C_v$ is the isentropic exponent.
    
    ### Power Calculation
    
    $$W = \\dot{m} \\cdot \\Delta h = \\dot{m} \\cdot (h_2 - h_1)$$
    
    Where:
    - $W$ = Shaft power (kW)
    - $\\dot{m}$ = Mass flow rate (kg/s)
    - $\\Delta h$ = Specific enthalpy change (kJ/kg)
    
    ### GERG-2008 Equation of State
    
    The GERG-2008 equation of state (ISO 20765-2) provides highly accurate thermodynamic 
    properties for natural gas mixtures with uncertainties of:
    - Density: ±0.1%
    - Speed of sound: ±0.1%
    - Heat capacity: ±1%
    """)
