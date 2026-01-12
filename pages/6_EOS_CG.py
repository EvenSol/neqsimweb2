import streamlit as st
import pandas as pd
from neqsim.thermo import TPflash, dataFrame
from neqsim import jneqsim
from theme import apply_theme, theme_toggle

st.set_page_config(page_title="EOS-CG", page_icon='images/neqsimlogocircleflat.png')
apply_theme()
theme_toggle()

st.title('EOS-CG')

"""
The **EOS-CG** (Equation of State for Combustion Gases) is an extension of the GERG-2008 framework 
designed for **Carbon Capture and Storage (CCS)** and **combustion gas** applications. It provides 
high-accuracy property calculations for CO₂-rich mixtures and flue gases.

EOS-CG supports **27 components**, including all GERG-2008 components plus additional species 
relevant to CCS and combustion:

**GERG-2008 components (21):**
- **Alkanes**: Methane, Ethane, Propane, n-Butane, i-Butane, n-Pentane, i-Pentane, n-Hexane, n-Heptane, n-Octane, n-Nonane, n-Decane
- **Non-hydrocarbons**: Nitrogen, CO₂, Hydrogen, Oxygen, CO, Water, Helium, Argon, H₂S

**Additional CCS/Combustion components (6):**
- SO₂ (Sulfur Dioxide)
- NO (Nitrogen Monoxide)  
- NO₂ (Nitrogen Dioxide)
- HCl (Hydrogen Chloride)
- Cl₂ (Chlorine)
- COS (Carbonyl Sulfide)

The flash calculation determines phase equilibrium at specified temperatures and pressures 
using the EOS-CG multiparameter equation of state.
"""

st.divider()

# Define EOS-CG compatible components (27 components)
# GERG-2008 base components + CCS/combustion specific components
eoscg_components = [
    # GERG-2008 components
    "methane", "nitrogen", "CO2", "ethane", "propane", 
    "i-butane", "n-butane", "i-pentane", "n-pentane", "n-hexane",
    "n-heptane", "n-octane", "n-nonane", "n-decane",
    "hydrogen", "oxygen", "CO", "water", "H2S", "helium", "argon",
    # CCS/Combustion specific components
    "SO2", "NO", "NO2", "HCl", "Cl2", "COS"
]

# Initialize default fluid composition for EOS-CG (CO2-rich CCS mixture)
default_eoscg_fluid = {
    'ComponentName': eoscg_components,
    'MolarComposition[-]': [
        0.5,   # methane
        2.0,   # nitrogen
        95.0,  # CO2 (dominant in CCS)
        0.0,   # ethane
        0.0,   # propane
        0.0,   # i-butane
        0.0,   # n-butane
        0.0,   # i-pentane
        0.0,   # n-pentane
        0.0,   # n-hexane
        0.0,   # n-heptane
        0.0,   # n-octane
        0.0,   # n-nonane
        0.0,   # n-decane
        0.0,   # hydrogen
        1.0,   # oxygen
        0.0,   # CO
        0.5,   # water
        0.0,   # H2S
        0.0,   # helium
        1.0,   # argon
        0.0,   # SO2
        0.0,   # NO
        0.0,   # NO2
        0.0,   # HCl
        0.0,   # Cl2
        0.0    # COS
    ]
}

st.text("Set fluid composition (EOS-CG components only):")

# Reset button to clear session state
if st.button('Reset to Default Composition'):
    st.session_state.eoscg_fluid_df = pd.DataFrame(default_eoscg_fluid)
    if 'eoscg_edited_df' in st.session_state:
        del st.session_state['eoscg_edited_df']
    st.rerun()

# Initialize session state - also check if old invalid components exist
if 'eoscg_fluid_df' not in st.session_state:
    st.session_state.eoscg_fluid_df = pd.DataFrame(default_eoscg_fluid)
else:
    # Check for invalid components and reset if found
    existing_components = st.session_state.eoscg_fluid_df['ComponentName'].tolist()
    invalid = [c for c in existing_components if c not in eoscg_components and pd.notna(c)]
    if invalid:
        st.session_state.eoscg_fluid_df = pd.DataFrame(default_eoscg_fluid)
        st.warning(f"Reset composition due to invalid components: {invalid}")

if 'eoscg_tp_data' not in st.session_state:
    st.session_state['eoscg_tp_data'] = pd.DataFrame({
        'Temperature (C)': [25.0, 40.0],
        'Pressure (bara)': [100.0, 150.0]
    })

# Show only active components option
hidecomponents = st.checkbox('Show only active components')

# Fluid composition editor
st.edited_df = st.data_editor(
    st.session_state.eoscg_fluid_df,
    column_config={
        "ComponentName": st.column_config.SelectboxColumn(
            "Component Name",
            options=eoscg_components,
            help="Select from EOS-CG compatible components only"
        ),
        "MolarComposition[-]": st.column_config.NumberColumn(
            "Molar Composition [-]", 
            min_value=0, 
            max_value=100, 
            format="%f",
            help="Enter molar composition (will be normalized)"
        ),
    },
    num_rows='dynamic'
)

# Update session state with edited data
st.session_state.eoscg_fluid_df = st.edited_df

# Filter display if showing only active components
if hidecomponents:
    active_df = st.edited_df[st.edited_df['MolarComposition[-]'] > 0]
    if not active_df.empty:
        st.write("**Active components:**")
        st.dataframe(active_df, hide_index=True)

st.info("💡 Note: EOS-CG supports 27 components including CCS/combustion species. Composition will be normalized before simulation.")

st.divider()

# Temperature and Pressure input
st.text("Input Pressures and Temperatures")
st.edited_dfTP = st.data_editor(
    st.session_state.eoscg_tp_data.dropna().reset_index(drop=True),
    num_rows='dynamic',
    column_config={
        'Temperature (C)': st.column_config.NumberColumn(
            label="Temperature (°C)",
            min_value=-200.0,
            max_value=500.0,
            format='%f',
            help='Enter the temperature in degrees Celsius.'
        ),
        'Pressure (bara)': st.column_config.NumberColumn(
            label="Pressure (bara)",
            min_value=0.0,
            max_value=700.0,
            format='%f',
            help='Enter the pressure in bar absolute. EOS-CG valid for high pressures relevant to CCS.'
        ),
    }
)

st.divider()

if st.button('Run EOS-CG TP Flash'):
    if st.edited_df['MolarComposition[-]'].sum() > 0:
        # Validate that all components are EOS-CG compatible
        invalid_components = [comp for comp in st.edited_df['ComponentName'].tolist() 
                             if comp not in eoscg_components and pd.notna(comp)]
        
        if invalid_components:
            st.error(f"Invalid components detected: {invalid_components}. Only EOS-CG components are allowed.")
        else:
            if st.edited_dfTP.dropna().empty:
                st.error('No data to perform calculations. Please input temperature and pressure values.')
            else:
                with st.spinner('Running EOS-CG flash calculations...'):
                    try:
                        # Set up temporary database tables
                        jneqsim.util.database.NeqSimDataBase.setCreateTemporaryTables(True)
                        
                        # Create EOS-CG fluid using Java API directly
                        # SystemEOSCGEos is the EOS-CG equation of state for CCS applications
                        eoscg_fluid = jneqsim.thermo.system.SystemEOSCGEos(298.15, 1.0)
                        
                        # Add components with their compositions
                        for idx, row in st.edited_df.iterrows():
                            comp_name = row['ComponentName']
                            comp_moles = row['MolarComposition[-]']
                            if pd.notna(comp_name) and comp_moles > 0:
                                eoscg_fluid.addComponent(comp_name, float(comp_moles))
                        
                        eoscg_fluid.createDatabase(True)
                        
                        # Initialize results list
                        results_list = []
                        
                        # Iterate over each T-P condition
                        for idx, row in st.edited_dfTP.dropna().iterrows():
                            temp = row['Temperature (C)']
                            pressure = row['Pressure (bara)']
                            
                            eoscg_fluid.setPressure(float(pressure), 'bara')
                            eoscg_fluid.setTemperature(float(temp), 'C')
                            
                            TPflash(eoscg_fluid)
                            eoscg_fluid.initThermoProperties()
                            eoscg_fluid.initPhysicalProperties()
                            
                            results_list.append(dataFrame(eoscg_fluid))
                        
                        st.success('EOS-CG flash calculations finished successfully!')
                        
                        st.subheader("Results:")
                        # Combine all results into a single dataframe
                        combined_results = pd.concat(results_list, ignore_index=True)
                        
                        # Display the results
                        st.dataframe(combined_results)
                        
                        # Additional properties display
                        st.divider()
                        st.subheader("Additional Properties (Last Calculation):")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Temperature", f"{eoscg_fluid.getTemperature('C'):.2f} °C")
                            st.metric("Pressure", f"{eoscg_fluid.getPressure('bara'):.2f} bara")
                            st.metric("Compressibility (Z)", f"{eoscg_fluid.getZ():.6f}")
                            st.metric("Density", f"{eoscg_fluid.getDensity('kg/m3'):.4f} kg/m³")
                        
                        with col2:
                            st.metric("Molar Mass", f"{eoscg_fluid.getMolarMass('kg/mol')*1000:.4f} g/mol")
                            st.metric("Enthalpy", f"{eoscg_fluid.getEnthalpy('J/mol'):.2f} J/mol")
                            st.metric("Entropy", f"{eoscg_fluid.getEntropy('J/molK'):.4f} J/(mol·K)")
                            if eoscg_fluid.hasPhaseType("gas"):
                                gas_phase = eoscg_fluid.getPhase("gas")
                                st.metric("Speed of Sound", f"{gas_phase.getSoundSpeed():.2f} m/s")
                        
                    except Exception as e:
                        st.error(f"Calculation failed: {str(e)}")
                        st.exception(e)
    else:
        st.error('The sum of Molar Composition must be greater than 0. Please adjust your inputs.')

# Sidebar info
st.sidebar.markdown("### About EOS-CG")
st.sidebar.markdown("""
**EOS-CG** (Equation of State for Combustion Gases) is a Helmholtz energy 
equation of state extending GERG-2008 for CCS and combustion applications.

**Key Features:**
- Extension of GERG-2008
- Optimized for CO₂-rich mixtures
- Includes combustion gas impurities
- 27 supported components

**Applications:**
- Carbon Capture & Storage (CCS)
- CO₂ transport pipelines
- Flue gas/exhaust mixtures
- Oxy-fuel combustion systems
- Blue/green hydrogen with CO₂

**Additional Components (vs GERG-2008):**
- SO₂, NO, NO₂, HCl, Cl₂, COS

**Reference:**
Gernert, J. and Span, R. (2016). 
*EOS-CG: A Helmholtz energy mixture model for humid gases and CCS mixtures*. 
J. Chem. Thermodynamics, 93, 274-293.
""")
