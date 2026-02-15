import streamlit as st
import pandas as pd
from neqsim.thermo import TPflash, dataFrame
from neqsim import jneqsim
from theme import apply_theme
from fluids import fluid_library_selector

st.set_page_config(page_title="GERG-2008", page_icon='images/neqsimlogocircleflat.png')
apply_theme()

st.title('GERG-2008')

"""
The GERG-2008 equation of state is a reference equation for natural gas calculations, 
developed according to ISO 20765-2. It provides high accuracy (±0.1% in density) for 
natural gas property calculations and is the standard for custody transfer and fiscal metering.

GERG-2008 supports the following **18 components** typical for natural gas mixtures:
- **Alkanes**: Methane, Ethane, Propane, n-Butane, i-Butane, n-Pentane, i-Pentane, n-Hexane, n-Heptane, n-Octane
- **Non-hydrocarbons**: Nitrogen, CO2, Hydrogen, Oxygen, CO (Carbon Monoxide), Water, Helium, Argon

The flash calculation will determine phase equilibrium at specified temperatures and pressures 
using the highly accurate GERG-2008 multiparameter equation of state.
"""

st.divider()

# Define GERG-2008 compatible components (21 components)
# Note: Component names must match NeqSim database naming conventions
gerg2008_components = [
    "methane", "nitrogen", "CO2", "ethane", "propane", 
    "i-butane", "n-butane", "i-pentane", "n-pentane", "n-hexane",
    "n-heptane", "n-octane", "hydrogen", "oxygen", "CO", 
    "water", "helium", "argon"
]

# Initialize default fluid composition for GERG-2008
default_gerg_fluid = {
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

st.text("Set fluid composition (GERG-2008 components only):")

# Reset button to clear session state
if st.button('Reset to Default Composition'):
    st.session_state.gerg_fluid_df = pd.DataFrame(default_gerg_fluid)
    if 'gerg_edited_df' in st.session_state:
        del st.session_state['gerg_edited_df']
    st.rerun()

# Initialize session state - also check if old invalid components exist
if 'gerg_fluid_df' not in st.session_state:
    st.session_state.gerg_fluid_df = pd.DataFrame(default_gerg_fluid)
else:
    # Check for invalid components and reset if found
    existing_components = st.session_state.gerg_fluid_df['ComponentName'].tolist()
    invalid = [c for c in existing_components if c not in gerg2008_components and pd.notna(c)]
    if invalid:
        st.session_state.gerg_fluid_df = pd.DataFrame(default_gerg_fluid)
        st.warning(f"Reset composition due to invalid components: {invalid}")

if 'gerg_tp_data' not in st.session_state:
    st.session_state['gerg_tp_data'] = pd.DataFrame({
        'Temperature (C)': [20.0, 25.0],
        'Pressure (bara)': [50.0, 100.0]
    })

# Show only active components option
hidecomponents = st.checkbox('Show active components')

# Filter display if showing only active components
if hidecomponents and 'gerg_edited_df' in st.session_state:
    st.session_state.gerg_fluid_df = st.session_state.gerg_edited_df[
        st.session_state.gerg_edited_df['MolarComposition[-]'] > 0
    ]

# Fluid composition editor
st.edited_df = st.data_editor(
    st.session_state.gerg_fluid_df,
    column_config={
        "ComponentName": st.column_config.SelectboxColumn(
            "Component Name",
            options=gerg2008_components,
            help="Select from GERG-2008 compatible components only"
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

# Store edited df for later use (don't update main session state to avoid reruns)
st.session_state.gerg_edited_df = st.edited_df

with st.expander("📂 Fluid Library", expanded=False):
    if fluid_library_selector('gerg', 'gerg_fluid_df'):
        st.rerun()

st.info("💡 Note: Only the 18 GERG-2008 components are available for selection. Composition will be normalized before simulation.")

st.divider()

# Temperature and Pressure input
st.text("Input Pressures and Temperatures")
st.edited_dfTP = st.data_editor(
    st.session_state.gerg_tp_data.dropna().reset_index(drop=True),
    num_rows='dynamic',
    column_config={
        'Temperature (C)': st.column_config.NumberColumn(
            label="Temperature (°C)",
            min_value=-200.0,
            max_value=450.0,
            format='%f',
            help='Enter the temperature in degrees Celsius. GERG-2008 valid range: 90-450 K (-183 to 177°C for highest accuracy)'
        ),
        'Pressure (bara)': st.column_config.NumberColumn(
            label="Pressure (bara)",
            min_value=0.0,
            max_value=350.0,
            format='%f',
            help='Enter the pressure in bar absolute. GERG-2008 valid range: up to 350 bar'
        ),
    }
)

# GERG-2008 variant selection
st.divider()
use_gerg_h2 = st.checkbox(
    'Use GERG-2008-H2 (improved hydrogen parameters)', 
    help='Use the GERG-2008-H2 variant with improved hydrogen binary interaction parameters (Beckmüller et al. 2022). Recommended for hydrogen-rich mixtures.'
)

if st.button('Run GERG-2008 TP Flash'):
    if st.edited_df['MolarComposition[-]'].sum() > 0:
        # Validate that all components are GERG-2008 compatible
        invalid_components = [comp for comp in st.edited_df['ComponentName'].tolist() 
                             if comp not in gerg2008_components and pd.notna(comp)]
        
        if invalid_components:
            st.error(f"Invalid components detected: {invalid_components}. Only GERG-2008 components are allowed.")
        else:
            if st.edited_dfTP.dropna().empty:
                st.error('No data to perform calculations. Please input temperature and pressure values.')
            else:
                with st.spinner('Running GERG-2008 flash calculations...'):
                    try:
                        # Set up temporary database tables
                        jneqsim.util.database.NeqSimDataBase.setCreateTemporaryTables(True)
                        
                        # Create GERG-2008 fluid using SystemGERG2008Eos for proper H2 support
                        gerg_fluid = jneqsim.thermo.system.SystemGERG2008Eos(273.15, 1.0)
                        
                        # Enable GERG-2008-H2 model if selected
                        if use_gerg_h2:
                            gerg_fluid.useHydrogenEnhancedModel()
                        
                        # Add components with their compositions
                        for idx, row in st.edited_df.iterrows():
                            comp_name = row['ComponentName']
                            comp_moles = row['MolarComposition[-]']
                            if pd.notna(comp_name) and comp_moles > 0:
                                gerg_fluid.addComponent(comp_name, float(comp_moles))
                        
                        # Initialize results list
                        results_list = []
                        
                        # Iterate over each T-P condition
                        for idx, row in st.edited_dfTP.dropna().iterrows():
                            temp = row['Temperature (C)']
                            pressure = row['Pressure (bara)']
                            
                            gerg_fluid.setPressure(float(pressure), 'bara')
                            gerg_fluid.setTemperature(float(temp), 'C')
                            
                            TPflash(gerg_fluid)
                            gerg_fluid.initThermoProperties()
                            gerg_fluid.initPhysicalProperties()
                            
                            results_list.append(dataFrame(gerg_fluid))
                        
                        model_name = gerg_fluid.getModelName()
                        st.success(f'Flash calculations finished successfully using {model_name}!')
                        
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
                            st.metric("Temperature", f"{gerg_fluid.getTemperature('C'):.2f} °C")
                            st.metric("Pressure", f"{gerg_fluid.getPressure('bara'):.2f} bara")
                            st.metric("Compressibility (Z)", f"{gerg_fluid.getZ():.6f}")
                            st.metric("Density", f"{gerg_fluid.getDensity('kg/m3'):.4f} kg/m³")
                        
                        with col2:
                            st.metric("Molar Mass", f"{gerg_fluid.getMolarMass('kg/mol')*1000:.4f} g/mol")
                            st.metric("Enthalpy", f"{gerg_fluid.getEnthalpy('J/mol'):.2f} J/mol")
                            st.metric("Entropy", f"{gerg_fluid.getEntropy('J/molK'):.4f} J/(mol·K)")
                            if gerg_fluid.hasPhaseType("gas"):
                                gas_phase = gerg_fluid.getPhase("gas")
                                st.metric("Speed of Sound", f"{gas_phase.getSoundSpeed():.2f} m/s")
                        
                    except Exception as e:
                        st.error(f"Calculation failed: {str(e)}")
                        st.exception(e)
    else:
        st.error('The sum of Molar Composition must be greater than 0. Please adjust your inputs.')

# Sidebar info
st.sidebar.markdown("### About GERG-2008")
st.sidebar.markdown("""
**GERG-2008** is a reference equation of state for natural gas developed by the 
European Gas Research Group (GERG).

**Key Features:**
- ISO 20765-2 standard
- ±0.1% accuracy in density
- Valid for natural gas mixtures
- 18 supported components

**Valid Ranges:**
- Temperature: 90-450 K
- Pressure: up to 350 bar

**Reference:**
Kunz, O. and Wagner, W. (2012). 
*The GERG-2008 Wide-Range Equation of State for Natural Gases and Other Mixtures*. 
J. Chem. Eng. Data, 57, 3032-3091.
""")
