import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from neqsim.thermo import fluid, TPflash
from neqsim import jneqsim
from theme import apply_theme
import json
import concurrent.futures
import time
import google.generativeai as genai

def get_gemini_api_key():
    """Get Gemini API key from secrets or session state."""
    try:
        if 'GEMINI_API_KEY' in st.secrets:
            return st.secrets['GEMINI_API_KEY']
    except Exception:
        pass
    return st.session_state.get('gemini_api_key', '')

def is_ai_enabled():
    """Check if AI features are enabled."""
    return st.session_state.get('ai_enabled', False) and get_gemini_api_key()

st.set_page_config(page_title="Compressor Performance", page_icon='images/neqsimlogocircleflat.png')
apply_theme()

st.title('Compressor Performance Calculator')

# Comprehensive documentation expander at the top
with st.expander("📖 **Documentation - User Manual & Method Reference**", expanded=False):
    st.markdown(r"""
    # Compressor Performance Calculator
    
    ## 1. Overview
    
    This tool calculates centrifugal compressor performance parameters from measured operating data 
    using rigorous thermodynamic calculations based on advanced equations of state. It is designed 
    for performance testing, commissioning verification, condition monitoring, and comparing actual 
    performance against manufacturer curves.
    
    **Primary Outputs:**
    - Polytropic head (kJ/kg)
    - Polytropic efficiency (%)
    - Isentropic efficiency (%)
    - Compression power (kW, MW)
    - Polytropic exponent (n)
    
    **Supported Equations of State:**
    - GERG-2008 (default) - Most accurate for natural gas applications
    - Peng-Robinson (PR) - General purpose cubic EoS
    - Soave-Redlich-Kwong (SRK) - General purpose cubic EoS
    
    ---
    
    ## 2. Quick Start Workflow
    
    | Step | Action | Description |
    |------|--------|-------------|
    | 1️⃣ | **Select Fluid** | Choose a preset gas or define custom mixture composition |
    | 2️⃣ | **Select EoS Model** | GERG-2008 recommended for natural gas |
    | 3️⃣ | **Choose Calculation Method** | Schultz (fast) or NeqSim Detailed (accurate) |
    | 4️⃣ | **Enter Operating Data** | Input measured P, T, and flow at inlet/outlet |
    | 5️⃣ | **Calculate** | Click "Calculate Performance" button |
    | 6️⃣ | **Analyze Results** | View plots, compare with manufacturer curves |
    
    ---
    
    ## 3. Calculation Methods
    
    Three calculation methods are available, each with different trade-offs between speed and accuracy:
    
    ### 3.1 Schultz Analytical Method
    
    The traditional analytical approach based on Schultz (1962). This method calculates the 
    polytropic exponent directly from measured temperature and pressure ratios.
    
    **Polytropic Exponent Calculation:**
    
    From the polytropic process relation:
    $$\frac{T_2}{T_1} = \left(\frac{P_2}{P_1}\right)^{\frac{n-1}{n}}$$
    
    Taking logarithms and solving for n:
    $$\frac{n-1}{n} = \frac{\ln(T_2/T_1)}{\ln(P_2/P_1)}$$
    
    $$n = \frac{1}{1 - \frac{\ln(T_2/T_1)}{\ln(P_2/P_1)}}$$
    
    **Polytropic Efficiency:**
    
    From the relation between polytropic exponent, isentropic exponent, and efficiency:
    $$n = \frac{1}{1 - \frac{\kappa-1}{\kappa \cdot \eta_p}}$$
    
    Solving for polytropic efficiency:
    $$\eta_p = \frac{n \cdot (\kappa - 1)}{\kappa \cdot (n - 1)}$$
    
    Where:
    - $n$ = polytropic exponent (calculated from measured data)
    - $\kappa$ = isentropic exponent (Cp/Cv), averaged between inlet and outlet
    - $\eta_p$ = polytropic efficiency
    
    **Polytropic Head (Enthalpy-Based):**
    
    $$H_p = \eta_p \times (h_2 - h_1)$$
    
    Where $h_1$ and $h_2$ are specific enthalpies at inlet and outlet conditions, 
    calculated using the selected equation of state.
    
    **Power Calculation:**
    
    $$P = \dot{m} \times (h_2 - h_1)$$
    
    Where:
    - $P$ = shaft power (kW)
    - $\dot{m}$ = mass flow rate (kg/s)
    - $(h_2 - h_1)$ = specific enthalpy rise (kJ/kg)
    
    **Advantages:** Fast calculation, no iteration required
    **Limitations:** Less accurate for high pressure ratios or highly non-ideal gases
    
    ---
    
    ### 3.2 NeqSim Process Model (Simple)
    
    Uses NeqSim's compressor model with the Schultz polytropic method internally. 
    The efficiency is solved iteratively to match the measured outlet temperature.
    
    **Algorithm:**
    1. Create inlet stream at measured P, T, and composition
    2. Set compressor outlet pressure to measured value
    3. Use Newton-Raphson iteration to find efficiency that produces measured outlet temperature
    4. Run compressor model with solved efficiency to get head and power
    
    **Advantages:** Good balance of accuracy and speed
    **Limitations:** Single-step integration may have small errors for high pressure ratios
    
    ---
    
    ### 3.3 NeqSim Process Model (Detailed) (Default)
    
    The most accurate method, using multi-step thermodynamic integration through the 
    compression path. Based on research from NTNU's thermal turbomachinery group.
    
    **Multi-Step Integration:**
    
    The compression is divided into N small pressure steps (configurable, default 40):
    
    $$\Delta P = \frac{P_2 - P_1}{N}$$
    
    At each step i, the following are calculated:
    1. Isentropic outlet state via entropy-flash (PSflash)
    2. Actual outlet state using polytropic efficiency
    3. Cumulative enthalpy change
    
    This approach properly accounts for:
    - Variation of thermodynamic properties along the compression path
    - Real-gas effects captured by the equation of state
    - Non-constant isentropic exponent
    
    **Efficiency Solving:**
    
    The method iteratively adjusts polytropic efficiency until the calculated outlet 
    temperature matches the measured outlet temperature:
    
    $$|T_{out,calc} - T_{out,measured}| < 10^{-5} \text{ K}$$
    
    **Advantages:** Most accurate, especially for:
    - High pressure ratios (>3:1)
    - Near-critical conditions
    - Gases with significant non-ideal behavior
    - Multi-component mixtures
    
    **Limitations:** Slower calculation (configurable trade-off with step count)
    
    ---
    
    ## 4. Isentropic Efficiency Calculation
    
    Isentropic efficiency is calculated using the PS-flash (pressure-entropy flash) to find 
    the isentropic outlet state:
    
    $$\eta_s = \frac{h_{2s} - h_1}{h_2 - h_1} = \frac{\text{Isentropic Work}}{\text{Actual Work}}$$
    
    Where:
    - $h_1$ = inlet specific enthalpy
    - $h_2$ = actual outlet specific enthalpy (from measured T, P)
    - $h_{2s}$ = isentropic outlet specific enthalpy (same entropy as inlet, at outlet pressure)
    
    ---
    
    ## 5. Input Data Requirements
    
    ### 5.1 Required Operating Data
    
    | Parameter | Description | Units |
    |-----------|-------------|-------|
    | Speed | Compressor rotational speed | RPM |
    | Flow | Volumetric or mass flow rate | Various (see below) |
    | Inlet P | Inlet pressure (absolute) | bara |
    | Outlet P | Outlet pressure (absolute) | bara |
    | Inlet T | Inlet temperature | °C or K |
    | Outlet T | Outlet temperature | °C or K |
    
    ### 5.2 Supported Flow Units
    
    | Unit | Description |
    |------|-------------|
    | kg/s | Mass flow rate |
    | kg/hr | Mass flow rate |
    | m³/hr (Am³/hr) | Actual volume flow at inlet conditions |
    | MSm³/day | Standard volume flow (at 15°C, 1.01325 bara) |
    
    ### 5.3 Data Input Methods
    
    - **Manual Entry:** Edit the data table directly
    - **CSV Import:** Upload a CSV file with operating data
    - **Excel Import:** Upload an Excel file (.xlsx)
    
    ---
    
    ## 6. Manufacturer Curve Comparison
    
    ### 6.1 Adding Manufacturer Curves
    
    Manufacturer performance curves can be added for comparison:
    - Head vs Flow curves at various speeds
    - Efficiency vs Flow curves at various speeds
    
    ### 6.2 Deviation Analysis
    
    The tool calculates deviations between measured and expected performance:
    
    $$\Delta H = H_{measured} - H_{curve}$$
    $$\Delta \eta = \eta_{measured} - \eta_{curve}$$
    
    Status indicators:
    - ✅ **OK:** Within ±2% of expected
    - ⚠️ **Warning:** 2-5% deviation
    - ❌ **Check:** >5% deviation
    
    ### 6.3 Gas Composition Correction (Khader Method)
    
    When operating with a different gas composition than the design gas, 
    manufacturer curves can be corrected using Mach number similarity:
    
    $$\frac{Q_{new}}{Q_{ref}} = \frac{a_{new}}{a_{ref}}$$
    
    Where $a$ is the speed of sound, calculated from the equation of state.
    
    ---
    
    ## 7. Output Results
    
    ### 7.1 Results Table
    
    | Column | Description | Unit |
    |--------|-------------|------|
    | Speed | Rotational speed | RPM |
    | Mass Flow | Mass flow rate | kg/hr |
    | Pressure Ratio | P_out / P_in | - |
    | Polytropic Exp (n) | Polytropic exponent | - |
    | Isentropic Eff | Isentropic efficiency | % |
    | Polytropic Eff | Polytropic efficiency | % |
    | Polytropic Head | Specific polytropic head | kJ/kg |
    | Actual Work | Specific enthalpy rise | kJ/kg |
    | Power | Shaft power | kW, MW |
    
    ### 7.2 Plots
    
    - **Polytropic Head vs Flow:** With manufacturer curve overlay
    - **Polytropic Efficiency vs Flow:** With manufacturer curve overlay
    - **Power vs Flow:** Compression power consumption
    
    ---
    
    ## 8. Thermodynamic Properties
    
    All thermodynamic properties are calculated using the selected equation of state:
    
    | Property | Symbol | Description |
    |----------|--------|-------------|
    | Compressibility | Z | Deviation from ideal gas |
    | Enthalpy | h | Specific enthalpy (kJ/kg) |
    | Entropy | s | Specific entropy (kJ/kg·K) |
    | Heat Capacity | Cp, Cv | Isobaric/isochoric heat capacity |
    | Isentropic Exponent | κ = Cp/Cv | Ratio of heat capacities |
    | Density | ρ | Mass density (kg/m³) |
    | Speed of Sound | a | For Mach number calculations |
    
    ---
    
    ## 9. File Formats
    
    ### 9.1 Manufacturer Curves JSON
    
    ```json
    {
      "flow_unit": "m3/hr",
      "curves": [
        {
          "speed": 10000,
          "flow": [100, 200, 300, 400],
          "head": [50, 48, 44, 38],
          "efficiency": [70, 78, 80, 75]
        }
      ]
    }
    ```
    
    ### 9.2 Operating Data CSV
    
    Required columns (names are flexible, mapped via dropdown):
    - Speed (RPM)
    - Flow (with unit specification)
    - Inlet Pressure
    - Outlet Pressure  
    - Inlet Temperature
    - Outlet Temperature
    
    ---
    
    ## 10. References & Standards
    
    | Reference | Description |
    |-----------|-------------|
    | ASME PTC 10 (1997) | Performance Test Code on Compressors and Exhausters |
    | Schultz, J.M. (1962) | "The Polytropic Analysis of Centrifugal Compressors" - ASME |
    | GERG-2008 | European Gas Research Group equation of state |
    | Peng-Robinson (1976) | Cubic equation of state for hydrocarbon systems |
    | Soave-Redlich-Kwong (1972) | Modified RK equation of state |
    | Hundseid & Bakken (2006) | "Wet Gas Performance Analysis" ASME GT2006-91035 |
    | Khader (2015) | Gas composition correction using Mach number similarity |
    
    ---
    
    ## 11. Tips & Best Practices
    
    1. **Choose the right EoS:** GERG-2008 for natural gas, PR/SRK for general hydrocarbons
    2. **Use Detailed method** for high pressure ratios (>3:1) or near-critical conditions
    3. **Verify input data:** Ensure pressures are absolute (not gauge)
    4. **Check temperature units:** Confirm whether input is °C or K
    5. **Validate with known points:** Test with manufacturer guarantee point first
    8. **Monitor efficiency trends:** Declining efficiency may indicate fouling or damage
    
    ---
    
    ## 12. Troubleshooting
    
    | Issue | Possible Cause | Solution |
    |-------|----------------|----------|
    | Efficiency > 100% | Outlet temperature too low for pressure ratio | Verify temperature measurement |
    | Efficiency < 50% | Outlet temperature too high or data error | Check for gas leaks or recycle |
    | Power mismatch | Flow measurement error | Verify flow meter calibration |
    | Head deviation | Gas composition difference | Use composition correction |
    | Calculation fails | Invalid thermodynamic state | Check if conditions are in valid range |
    
    
    """)

st.divider()

st.info("""
**💡 Quick Start:** Select fluid composition → Enter operating data (P, T, Flow) → Click "Calculate Performance"

Expand the **Documentation** section above for detailed method descriptions, equations, and troubleshooting guidance.
""")

st.divider()

# Supported components (compatible with GERG-2008, PR, and SRK)
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
    "Typical Natural Gas (North Sea)": {
        "methane": 85.0, "ethane": 7.0, "propane": 3.0, "i-butane": 0.5, 
        "n-butane": 0.8, "CO2": 2.0, "nitrogen": 1.5, "i-pentane": 0.1, "n-pentane": 0.1
    },
    "Lean Natural Gas": {
        "methane": 95.0, "ethane": 2.5, "propane": 0.5, "CO2": 1.0, "nitrogen": 1.0
    },
    "Rich Natural Gas": {
        "methane": 78.0, "ethane": 10.0, "propane": 5.0, "i-butane": 1.5, 
        "n-butane": 2.0, "CO2": 2.0, "nitrogen": 1.0, "i-pentane": 0.25, "n-pentane": 0.25
    },
    "Export Gas (Processed)": {
        "methane": 92.0, "ethane": 4.0, "propane": 1.5, "CO2": 1.5, "nitrogen": 1.0
    },
    "Custom Mixture": None,  # Special case for custom composition
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
        'Speed (RPM)': [10000.0, 10000.0, 10000.0, 10000.0, 10000.0],
        'Flow Rate': [1000.0, 1200.0, 1500.0, 1800.0, 2000.0],
        'Inlet Pressure': [50.0, 50.0, 50.0, 50.0, 50.0],
        'Outlet Pressure': [100.0, 95.0, 90.0, 85.0, 80.0],
        'Inlet Temperature': [30.0, 30.0, 30.0, 30.0, 30.0],
        'Outlet Temperature': [115.0, 105.0, 95.0, 87.0, 82.0],
    })

# Initialize session state for compressor curves
if 'compressor_curves' not in st.session_state:
    st.session_state['compressor_curves'] = []  # List of curve sets, each with speed, flow, head, efficiency

if 'show_compressor_curves' not in st.session_state:
    st.session_state['show_compressor_curves'] = False

# Initialize session state for calculation method
if 'calc_method' not in st.session_state:
    st.session_state['calc_method'] = "NeqSim Process Model (Detailed)"
if 'num_calc_steps' not in st.session_state:
    st.session_state['num_calc_steps'] = 10
if 'polytropic_efficiency_input' not in st.session_state:
    st.session_state['polytropic_efficiency_input'] = 75.0

# Initialize session state for equation of state model
if 'eos_model' not in st.session_state:
    st.session_state['eos_model'] = "GERG-2008"

# EoS model options mapping
eos_model_options = {
    "GERG-2008": "gerg-2008",
    "Peng-Robinson": "pr",
    "Soave-Redlich-Kwong": "srk"
}

# Sidebar for fluid selection
with st.sidebar:
    st.header("Fluid Selection")
    selected_fluid_name = st.selectbox(
        "Select Test Fluid",
        options=list(test_fluids.keys()),
        index=0
    )
    
    if selected_fluid_name != "Custom Mixture":
        st.info(f"Selected fluid composition: {test_fluids[selected_fluid_name]}")
    else:
        st.info("Define custom composition in the main panel")
    
    # Equation of State model selection
    selected_eos = st.selectbox(
        "Equation of State Model",
        options=list(eos_model_options.keys()),
        index=list(eos_model_options.keys()).index(st.session_state['eos_model']),
        help="GERG-2008: High accuracy for natural gas. PR: Peng-Robinson cubic EoS. SRK: Soave-Redlich-Kwong cubic EoS."
    )
    st.session_state['eos_model'] = selected_eos
    
    st.divider()
    st.header("Calculation Method")
    
    calc_method = st.selectbox(
        "Select Method",
        options=["NeqSim Process Model (Detailed)", "NeqSim Process Model (Simple)", "Schultz (Analytical)"],
        index=["NeqSim Process Model (Detailed)", "NeqSim Process Model (Simple)", "Schultz (Analytical)"].index(st.session_state['calc_method']) if st.session_state['calc_method'] in ["NeqSim Process Model (Detailed)", "NeqSim Process Model (Simple)", "Schultz (Analytical)"] else 0,
        help="Schultz: Analytical polytropic analysis. NeqSim Simple: Faster calculation. NeqSim Detailed: Multi-step polytropic (more accurate but slower)."
    )
    st.session_state['calc_method'] = calc_method
    
    if calc_method == "NeqSim Process Model (Detailed)":
        st.info("🔧 Uses NeqSim's detailed polytropic method with multi-step integration. More accurate but slower.")
        
        num_steps = st.slider(
            "Number of Calculation Steps",
            min_value=5,
            max_value=20,
            value=st.session_state['num_calc_steps'],
            step=5,
            help="More steps = higher accuracy but slower calculation"
        )
        st.session_state['num_calc_steps'] = num_steps
    elif calc_method == "NeqSim Process Model (Simple)":
        st.info("⚡ Uses NeqSim's simple polytropic method. Faster calculation with good accuracy.")
    
    # AI Analysis section - only show if AI is enabled
    if is_ai_enabled():
        st.divider()
        st.header("🤖 AI Analysis")
        st.success(f"✓ AI enabled ({st.session_state.get('ai_model', 'gemini-2.0-flash')})")

# Helper function to get the selected EoS model code
def get_selected_eos_model():
    return eos_model_options.get(st.session_state['eos_model'], "gerg-2008")


# Helper function to calculate polytropic exponent from measured data
def calculate_polytropic_exponent(T_in_K, T_out_K, pr, kappa_avg):
    """
    Calculate polytropic exponent n from measured temperature and pressure ratios.
    
    Using: T2/T1 = (P2/P1)^((n-1)/n)
    Solving: (n-1)/n = log(T2/T1) / log(P2/P1)
             n = 1 / (1 - (n-1)/n)
    
    Args:
        T_in_K: Inlet temperature in Kelvin
        T_out_K: Outlet temperature in Kelvin
        pr: Pressure ratio (P_out / P_in)
        kappa_avg: Average isentropic exponent for fallback
        
    Returns:
        Polytropic exponent n
    """
    if pr > 1 and T_out_K > T_in_K:
        log_T_ratio = np.log(T_out_K / T_in_K)
        log_P_ratio = np.log(pr)
        if log_P_ratio > 0 and log_T_ratio > 0:
            n_minus_1_over_n = log_T_ratio / log_P_ratio
            return 1 / (1 - n_minus_1_over_n) if n_minus_1_over_n < 1 else 1.5
    # Fallback estimate
    return kappa_avg / (kappa_avg - 1 + 0.001) * 0.8

# Helper function to get fluid composition dict
def get_fluid_composition():
    if selected_fluid_name == "Custom Mixture":
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
    st.write(f"**Selected Fluid:** {selected_fluid_name} | **EoS Model:** {st.session_state['eos_model']}")
    
    # Show custom fluid editor if custom mixture selected
    if selected_fluid_name == "Custom Mixture":
        st.write("Define your custom fluid composition:")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button('Reset to Default', key='reset_custom_fluid'):
                st.session_state.compressor_custom_fluid_df = pd.DataFrame(default_custom_fluid)
                st.rerun()
            
            hidecomponents = st.checkbox('Show only active', key='hide_inactive_comp')
        
        display_df = st.session_state.compressor_custom_fluid_df
        if hidecomponents:
            display_df = display_df[display_df['MolarComposition[-]'] > 0]
        
        # Use a form to prevent table reload while typing
        with st.form("fluid_composition_form"):
            st.caption("💡 Edit composition and click **Apply** when done.")
            
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
                        format="%.4f",
                        help="Enter molar composition (will be normalized)"
                    ),
                },
                num_rows='dynamic',
                key='custom_fluid_editor'
            )
            
            submitted_fluid = st.form_submit_button("Apply", type='primary')
            
            if submitted_fluid:
                # Update session state
                if not hidecomponents:
                    st.session_state.compressor_custom_fluid_df = edited_fluid_df
                st.success("✅ Composition updated!")
        
        st.caption("💡 Composition will be normalized before simulation")
    
    # Create fluid for display using selected EoS model
    fluid_composition = get_fluid_composition()
    
    if fluid_composition and len(fluid_composition) > 0:
        try:
            jneqsim.util.database.NeqSimDataBase.setCreateTemporaryTables(True)
            display_fluid = fluid(get_selected_eos_model())
            for comp_name, comp_moles in fluid_composition.items():
                display_fluid.addComponent(comp_name, float(comp_moles))
            display_fluid.setMixingRule('classic')
            display_fluid.setPressure(50.0, 'bara')
            display_fluid.setTemperature(30.0, 'C')
            TPflash(display_fluid)
            display_fluid.initThermoProperties()
            
            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Molar Mass", f"{display_fluid.getMolarMass()*1000:.2f} g/mol")
            with col2:
                st.metric("Z-factor @ 50 bara, 30°C", f"{display_fluid.getZ():.4f}")
            with col3:
                st.metric("Cp/Cv (κ)", f"{display_fluid.getGamma():.4f}")
            with col4:
                st.metric("EoS Model", st.session_state['eos_model'])
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
    
    # Data import/export section
    st.subheader("📁 Import/Export Data")
    col_import1, col_import2 = st.columns(2)
    
    with col_import1:
        uploaded_csv = st.file_uploader(
            "📤 Import Operating Data (CSV/Excel)", 
            type=['csv', 'xlsx', 'xls'],
            key='operating_data_upload',
            help="Upload a CSV or Excel file with columns: Speed (RPM), Flow Rate, Inlet Pressure, Outlet Pressure, Inlet Temperature, Outlet Temperature"
        )
        if uploaded_csv is not None:
            try:
                if uploaded_csv.name.endswith('.csv'):
                    imported_df = pd.read_csv(uploaded_csv)
                else:
                    imported_df = pd.read_excel(uploaded_csv)
                
                # Try to map columns to expected names
                column_mapping = {}
                expected_cols = ['Speed (RPM)', 'Flow Rate', 'Inlet Pressure', 'Outlet Pressure', 'Inlet Temperature', 'Outlet Temperature']
                
                for expected in expected_cols:
                    for col in imported_df.columns:
                        col_lower = col.lower().replace('_', ' ').replace('-', ' ')
                        expected_lower = expected.lower()
                        if expected_lower in col_lower or col_lower in expected_lower:
                            column_mapping[col] = expected
                            break
                        # Also check partial matches
                        if 'speed' in col_lower and 'speed' in expected_lower:
                            column_mapping[col] = expected
                        elif 'flow' in col_lower and 'flow' in expected_lower:
                            column_mapping[col] = expected
                        elif 'inlet' in col_lower and 'pressure' in col_lower and 'inlet p' in expected_lower:
                            column_mapping[col] = expected
                        elif 'outlet' in col_lower and 'pressure' in col_lower and 'outlet p' in expected_lower:
                            column_mapping[col] = expected
                        elif 'inlet' in col_lower and 'temp' in col_lower and 'inlet t' in expected_lower:
                            column_mapping[col] = expected
                        elif 'outlet' in col_lower and 'temp' in col_lower and 'outlet t' in expected_lower:
                            column_mapping[col] = expected
                
                if column_mapping:
                    imported_df = imported_df.rename(columns=column_mapping)
                
                # Ensure all expected columns exist
                for col in expected_cols:
                    if col not in imported_df.columns:
                        if col == 'Speed (RPM)':
                            imported_df[col] = 10000.0  # Default speed
                        else:
                            st.warning(f"Column '{col}' not found in file. Please add it manually.")
                
                st.session_state['compressor_data'] = imported_df[expected_cols].dropna()
                st.success(f"Imported {len(imported_df)} data points")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to import data: {e}")
    
    with col_import2:
        # Download template
        template_df = pd.DataFrame({
            'Speed (RPM)': [10000.0, 10000.0, 10000.0],
            'Flow Rate': [1000.0, 1500.0, 2000.0],
            'Inlet Pressure': [50.0, 50.0, 50.0],
            'Outlet Pressure': [120.0, 120.0, 120.0],
            'Inlet Temperature': [30.0, 30.0, 30.0],
            'Outlet Temperature': [110.0, 105.0, 102.0],
        })
        csv_template = template_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Template CSV",
            data=csv_template,
            file_name="compressor_data_template.csv",
            mime="text/csv",
            help="Download a template CSV file with the expected column format"
        )
    
    st.divider()
    
    # Reset button
    col_reset1, col_reset2 = st.columns([1, 3])
    with col_reset1:
        if st.button('Reset to Example Data'):
            st.session_state['compressor_data'] = pd.DataFrame({
                'Speed (RPM)': [10000.0, 10000.0, 10000.0, 10000.0, 10000.0],
                'Flow Rate': [1000.0, 1200.0, 1500.0, 1800.0, 2000.0],
                'Inlet Pressure': [50.0, 50.0, 50.0, 50.0, 50.0],
                'Outlet Pressure': [100.0, 95.0, 90.0, 85.0, 80.0],
                'Inlet Temperature': [30.0, 30.0, 30.0, 30.0, 30.0],
                'Outlet Temperature': [115.0, 105.0, 95.0, 87.0, 82.0],
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
    
    # Use a form to prevent table reload while typing
    with st.form("operating_data_form"):
        st.caption("💡 Edit the table below and click **Apply Changes** when done.")
        
        edited_data = st.data_editor(
            st.session_state['compressor_data'].dropna().reset_index(drop=True),
            num_rows='dynamic',
            column_config={
                'Speed (RPM)': st.column_config.NumberColumn(
                    "Speed (RPM)",
                    min_value=100.0,
                    max_value=50000.0,
                    format='%.0f',
                    help='Compressor rotational speed in RPM'
                ),
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
            },
            key='operating_data_editor'
        )
        
        col_submit1, col_submit2 = st.columns([1, 4])
        with col_submit1:
            submitted = st.form_submit_button("Apply Changes", type='primary')
        with col_submit2:
            st.caption(f"📐 Units: Flow = {selected_flow_unit}, Pressure = {selected_pressure_unit}, Temperature = {selected_temp_unit}")
        
        if submitted:
            st.session_state['compressor_data'] = edited_data
            st.success("✅ Data updated!")

st.divider()

# Compressor Curves Section
with st.expander("📈 Compressor Manufacturer Curves (Optional)", expanded=st.session_state['show_compressor_curves']):
    st.write("Add compressor performance curves from manufacturer data. Each curve represents a different speed.")
    
    col_curve1, col_curve2 = st.columns([1, 1])
    with col_curve1:
        show_curves = st.checkbox('Show curves on plots', value=st.session_state['show_compressor_curves'], key='show_curves_checkbox')
        st.session_state['show_compressor_curves'] = show_curves
    with col_curve2:
        curve_flow_unit = st.selectbox(
            "Curve Flow Unit",
            options=list(flow_units.keys()),
            index=list(flow_units.keys()).index(st.session_state.get('curve_flow_unit', st.session_state['flow_unit'])),
            key='curve_flow_unit_select',
            help="Flow unit used in the compressor curves"
        )
        st.session_state['curve_flow_unit'] = curve_flow_unit
    
    st.divider()
    
    # File operations for curves
    st.subheader("💾 Save/Load Curves")
    col_file1, col_file2 = st.columns(2)
    
    with col_file1:
        # Save curves to file
        if st.session_state['compressor_curves']:
            curves_json = json.dumps({
                'flow_unit': st.session_state.get('curve_flow_unit', 'm3/hr'),
                'curves': st.session_state['compressor_curves']
            }, indent=2)
            st.download_button(
                label="📥 Download Curves (JSON)",
                data=curves_json,
                file_name="compressor_curves.json",
                mime="application/json"
            )
        else:
            st.info("Add curves to enable download")
    
    with col_file2:
        # Load curves from file
        uploaded_file = st.file_uploader("📤 Load Curves (JSON)", type=['json'], key='curve_upload')
        if uploaded_file is not None:
            try:
                loaded_data = json.load(uploaded_file)
                if 'curves' in loaded_data:
                    st.session_state['compressor_curves'] = loaded_data['curves']
                    if 'flow_unit' in loaded_data:
                        st.session_state['curve_flow_unit'] = loaded_data['flow_unit']
                    st.success(f"Loaded {len(loaded_data['curves'])} curve(s)")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to load curves: {e}")
    
    st.divider()
    
    # Add new curve
    st.subheader("➕ Add New Curve")
    
    new_speed = st.number_input("Speed (RPM)", min_value=100.0, max_value=50000.0, value=10000.0, step=100.0, key='new_curve_speed')
    
    # Default curve data
    default_curve_data = {
        'Flow': [800.0, 1000.0, 1200.0, 1500.0, 1800.0, 2000.0],
        'Polytropic Head (kJ/kg)': [85.0, 82.0, 78.0, 72.0, 64.0, 58.0],
        'Efficiency (%)': [72.0, 76.0, 78.0, 77.0, 74.0, 70.0]
    }
    
    if 'new_curve_data' not in st.session_state:
        st.session_state['new_curve_data'] = pd.DataFrame(default_curve_data)
    
    # Use a form to prevent table reload while typing
    with st.form("curve_data_form"):
        st.write(f"Enter curve data points (Flow in {curve_flow_unit}):")
        st.caption("💡 Edit the table and click **Apply & Add Curve** when done.")
        
        edited_curve = st.data_editor(
            st.session_state['new_curve_data'],
            num_rows='dynamic',
            column_config={
                'Flow': st.column_config.NumberColumn(
                    f"Flow ({curve_flow_unit})",
                    min_value=0.0,
                    format='%.2f'
                ),
                'Polytropic Head (kJ/kg)': st.column_config.NumberColumn(
                    "Polytropic Head (kJ/kg)",
                    min_value=0.0,
                    format='%.2f'
                ),
                'Efficiency (%)': st.column_config.NumberColumn(
                    "Efficiency (%)",
                    min_value=0.0,
                    max_value=100.0,
                    format='%.1f'
                )
            },
            key='curve_data_editor'
        )
        
        col_form1, col_form2 = st.columns([1, 1])
        with col_form1:
            add_curve_submitted = st.form_submit_button("Apply & Add Curve", type='primary')
        with col_form2:
            update_only = st.form_submit_button("Apply Changes Only")
        
        if add_curve_submitted:
            st.session_state['new_curve_data'] = edited_curve
            if not edited_curve.empty and len(edited_curve.dropna()) > 1:
                new_curve = {
                    'speed': new_speed,
                    'flow': edited_curve['Flow'].dropna().tolist(),
                    'head': edited_curve['Polytropic Head (kJ/kg)'].dropna().tolist(),
                    'efficiency': edited_curve['Efficiency (%)'].dropna().tolist()
                }
                st.session_state['compressor_curves'].append(new_curve)
                # Reset the new curve data
                st.session_state['new_curve_data'] = pd.DataFrame(default_curve_data)
                st.success(f"Added curve for {new_speed} RPM")
                st.rerun()
            else:
                st.warning("Please enter at least 2 data points")
        elif update_only:
            st.session_state['new_curve_data'] = edited_curve
            st.success("✅ Table data saved!")
    
    col_reset1, col_reset2 = st.columns([1, 3])
    with col_reset1:
        if st.button("Reset Input"):
            st.session_state['new_curve_data'] = pd.DataFrame(default_curve_data)
            st.rerun()
    
    st.divider()
    
    # Display existing curves
    if st.session_state['compressor_curves']:
        st.subheader("📋 Existing Curves")
        
        for i, curve in enumerate(st.session_state['compressor_curves']):
            with st.container():
                col_c1, col_c2, col_c3 = st.columns([2, 2, 1])
                with col_c1:
                    st.write(f"**{curve['speed']:.0f} RPM**")
                with col_c2:
                    st.write(f"{len(curve['flow'])} points")
                with col_c3:
                    if st.button("🗑️", key=f'delete_curve_{i}', help=f"Delete curve at {curve['speed']} RPM"):
                        st.session_state['compressor_curves'].pop(i)
                        st.rerun()
                
                # Show curve data in a small table
                curve_df = pd.DataFrame({
                    f"Flow ({st.session_state.get('curve_flow_unit', 'm3/hr')})": curve['flow'],
                    'Head (kJ/kg)': curve['head'],
                    'Eff (%)': curve['efficiency']
                })
                st.dataframe(curve_df, use_container_width=True, height=100)
        
        if st.button("Clear All Curves", type='secondary'):
            st.session_state['compressor_curves'] = []
            st.rerun()
    else:
        st.info("No curves added yet. Add curves above or load from file.")
    
    st.divider()
    
    # Generate Updated Curves Section - nested expander
    with st.expander("🔄 Generate Updated Curves for New Gas", expanded=False):
        st.markdown("""
        Generate updated compressor curves when gas composition or molecular weight changes.
        This uses **Mach number similarity** methods (Khader, 2015; Lüdtke, 2004; Schultz, 1962).
    
    ---
    
    **Physical Basis:**
    
    Centrifugal compressor performance depends on the **Machine Mach Number** ($Ma_u$):
    
    $$Ma_u = \\frac{U_{tip}}{c_s} = \\frac{\\pi D N}{c_s}$$
    
    Where:
    - $U_{tip}$ = Impeller tip speed (m/s)
    - $D$ = First stage impeller exit diameter (m)
    - $N$ = Rotational speed (rev/s)
    - $c_s$ = Speed of sound in gas (m/s)
    
    The speed of sound is calculated from:
    $$c_s = \\sqrt{\\gamma Z R T / M_w}$$
    
    Where $\\gamma$ = isentropic exponent, $Z$ = compressibility factor, $R$ = gas constant, $T$ = temperature, $M_w$ = molar mass.
    
    ---
    
    **Correction Equations (Constant Mach Number Similarity):**
    
    | Parameter | Correction | Physical Basis |
    |-----------|------------|----------------|
    | Polytropic Head | $H_{new} = H_{ref} \\times \\left(\\frac{c_{s,new}}{c_{s,ref}}\\right)^2$ | $H_p \\propto U_{tip}^2 \\propto c_s^2$ at constant $Ma$ |
    | Volumetric Flow | $Q_{new} = Q_{ref} \\times \\frac{c_{s,new}}{c_{s,ref}}$ | $Q \\propto U_{tip} \\propto c_s$ at constant $Ma$ |
    | Polytropic Efficiency | $\\eta_{p,new} \\approx \\eta_{p,ref}$ | Approximately invariant |
    
    ---
    
    **References:**
    - Khader, M.A. (2015). *Effect of Gas Composition on Centrifugal Compressor Performance*, ASME Turbo Expo.
    - Lüdtke, K.H. (2004). *Process Centrifugal Compressors*, Springer.
    - Schultz, J.M. (1962). "The Polytropic Analysis of Centrifugal Compressors", *J. Eng. Power*, 84(1), 69-82.
        """)
    
        if st.session_state['compressor_curves']:
            # Reference conditions (original curves)
            st.write("**Reference Conditions (Original Curves):**")
            col_ref1, col_ref2 = st.columns(2)
            with col_ref1:
                ref_mw = st.number_input("Reference MW (g/mol)", min_value=2.0, max_value=100.0, value=18.0, step=0.1, key='ref_mw',
                                         help="Molecular weight of the gas used for the original manufacturer curves")
            with col_ref2:
                impeller_diameter = st.number_input("1st Stage Impeller Exit Dia (mm)", min_value=50.0, max_value=2000.0, value=500.0, step=10.0, key='impeller_dia',
                                                    help="First stage impeller exit diameter - used for Machine Mach number calculation")
            
            col_ref3, col_ref4 = st.columns(2)
            with col_ref3:
                ref_temp = st.number_input("Reference Temp (°C)", min_value=-50.0, max_value=150.0, value=30.0, step=1.0, key='ref_temp',
                                           help="Temperature at which original curves were measured")
            with col_ref4:
                ref_pressure = st.number_input("Reference Pressure (bara)", min_value=1.0, max_value=500.0, value=50.0, step=1.0, key='ref_pressure',
                                               help="Pressure at which original curves were measured")
            
            st.write("**New Conditions (Current Fluid):**")
            st.info(f"Using selected fluid: **{selected_fluid_name}**")
            
            # Calculate new conditions from selected fluid
            new_fluid_composition = get_fluid_composition()
            
            if new_fluid_composition and len(new_fluid_composition) > 0:
                try:
                    # Create reference fluid (approximate using ideal gas properties)
                    R = 8.314  # J/(mol·K)
                    gamma_ref = 1.3  # Typical for natural gas
                    T_ref_K = ref_temp + 273.15
                    # Speed of sound approximation: c = sqrt(gamma * R * T / MW)
                    c_s_ref = np.sqrt(gamma_ref * R * T_ref_K / (ref_mw / 1000))  # m/s
                    
                    # Create new fluid and calculate properties
                    jneqsim.util.database.NeqSimDataBase.setCreateTemporaryTables(True)
                    new_fluid = fluid(get_selected_eos_model())
                    for comp_name, comp_moles in new_fluid_composition.items():
                        new_fluid.addComponent(comp_name, float(comp_moles))
                    new_fluid.setMixingRule('classic')
                    new_fluid.setPressure(ref_pressure, 'bara')
                    new_fluid.setTemperature(ref_temp, 'C')
                    TPflash(new_fluid)
                    new_fluid.initThermoProperties()
                    
                    new_mw = new_fluid.getMolarMass() * 1000  # g/mol
                    gamma_new = new_fluid.getGamma()
                    z_new = new_fluid.getZ()
                    T_new_K = ref_temp + 273.15
                    # More accurate speed of sound using selected EoS
                    c_s_new = np.sqrt(gamma_new * z_new * R * T_new_K / (new_mw / 1000))  # m/s
                    
                    # Calculate Machine Mach Number for a reference speed (use first curve speed)
                    ref_speed_rpm = st.session_state['compressor_curves'][0]['speed']
                    D_m = impeller_diameter / 1000  # Convert mm to m
                    U_tip = np.pi * D_m * ref_speed_rpm / 60  # Tip speed in m/s
                    Ma_ref = U_tip / c_s_ref  # Machine Mach number at reference conditions
                    Ma_new = U_tip / c_s_new  # Machine Mach number at new conditions
                    
                    col_new1, col_new2, col_new3, col_new4 = st.columns(4)
                    with col_new1:
                        st.metric("New MW", f"{new_mw:.2f} g/mol", f"{new_mw - ref_mw:+.2f}")
                    with col_new2:
                        st.metric("Sound Speed Ratio", f"{c_s_new/c_s_ref:.3f}")
                    with col_new3:
                        st.metric("κ (Cp/Cv)", f"{gamma_new:.3f}")
                    with col_new4:
                        st.metric(f"Ma @ {ref_speed_rpm:.0f} RPM", f"{Ma_new:.3f}", f"{Ma_new - Ma_ref:+.3f} vs ref")
                    
                    # Calculate correction factors
                    sound_speed_ratio = c_s_new / c_s_ref
                    head_correction = sound_speed_ratio ** 2  # Head scales with c_s^2
                    flow_correction = sound_speed_ratio  # Flow scales with c_s
                    
                    st.write(f"**Correction Factors:** Head × {head_correction:.3f}, Flow × {flow_correction:.3f}")
                    st.caption(f"Tip speed: {U_tip:.1f} m/s | Sound speed (ref): {c_s_ref:.1f} m/s | Sound speed (new): {c_s_new:.1f} m/s")
                    
                    if st.button("🔄 Generate Corrected Curves", type='primary'):
                        corrected_curves = []
                        
                        for curve in st.session_state['compressor_curves']:
                            corrected_flow = [f * flow_correction for f in curve['flow']]
                            corrected_head = [h * head_correction for h in curve['head']]
                            # Efficiency is approximately independent of gas properties
                            corrected_eff = curve['efficiency'].copy() if isinstance(curve['efficiency'], list) else list(curve['efficiency'])
                            
                            corrected_curves.append({
                                'speed': curve['speed'],
                                'flow': corrected_flow,
                                'head': corrected_head,
                                'efficiency': corrected_eff
                            })
                        
                        # Store corrected curves with metadata
                        st.session_state['corrected_curves'] = {
                            'curves': corrected_curves,
                            'reference_mw': ref_mw,
                            'new_mw': new_mw,
                            'fluid_name': selected_fluid_name,
                            'flow_unit': st.session_state.get('curve_flow_unit', 'm3/hr')
                        }
                        st.success(f"Generated {len(corrected_curves)} corrected curve(s) for MW = {new_mw:.2f} g/mol")
                    
                    # Display corrected curves if available
                    if 'corrected_curves' in st.session_state and st.session_state['corrected_curves']:
                        corrected_data = st.session_state['corrected_curves']
                        st.divider()
                        st.write(f"**Corrected Curves** (MW: {corrected_data.get('reference_mw', 0):.1f} → {corrected_data.get('new_mw', 0):.1f} g/mol)")
                        
                        for curve in corrected_data['curves']:
                            corr_df = pd.DataFrame({
                                f"Flow ({corrected_data.get('flow_unit', 'm3/hr')})": curve['flow'],
                                'Head (kJ/kg)': curve['head'],
                                'Eff (%)': curve['efficiency']
                            })
                            st.write(f"**{curve['speed']:.0f} RPM:**")
                            st.dataframe(corr_df, use_container_width=True, height=100)
                        
                        col_action1, col_action2 = st.columns(2)
                        with col_action1:
                            if st.button("✅ Use Corrected Curves"):
                                st.session_state['compressor_curves'] = corrected_data['curves']
                                st.session_state['corrected_curves'] = None
                                st.success("Replaced original curves with corrected curves")
                                st.rerun()
                        with col_action2:
                            # Download corrected curves
                            corr_json = json.dumps({
                                'flow_unit': corrected_data.get('flow_unit', 'm3/hr'),
                                'reference_mw': corrected_data.get('reference_mw'),
                                'corrected_mw': corrected_data.get('new_mw'),
                                'curves': corrected_data['curves']
                            }, indent=2)
                            st.download_button(
                                label="📥 Download Corrected Curves",
                                data=corr_json,
                                file_name=f"corrected_curves_MW{corrected_data.get('new_mw', 0):.0f}.json",
                                mime="application/json"
                            )
                            
                except Exception as e:
                    st.error(f"Failed to calculate new fluid properties: {e}")
            else:
                st.warning("Please select a valid fluid composition first")
        else:
            st.info("Add original manufacturer curves above first, then generate corrected curves for the new gas.")

    st.divider()
    
    # Generate Curves from Measured Data Section - nested expander
    with st.expander("📈 Generate Curves from Measured Data", expanded=False):
        st.markdown("""
        Develop compressor performance curves from measured operating data using polynomial regression 
        and **affinity law normalization** (Saravanamuttoo et al., 2017; Brown, 2005).
        
        ---
        
        **Affinity Laws (Fan Laws):**
        
        For a centrifugal compressor operating on geometrically similar conditions:
        
        | Parameter | Relationship | Equation |
        |-----------|--------------|----------|
        | Volumetric Flow | $Q \\propto N$ | $\\frac{Q_1}{Q_2} = \\frac{N_1}{N_2}$ |
        | Polytropic Head | $H_p \\propto N^2$ | $\\frac{H_{p1}}{H_{p2}} = \\left(\\frac{N_1}{N_2}\\right)^2$ |
        | Power | $P \\propto N^3$ | $\\frac{P_1}{P_2} = \\left(\\frac{N_1}{N_2}\\right)^3$ |
        | Polytropic Efficiency | $\\eta_p \\approx const$ | Independent of speed |
        
        ---
        
        **Curve Fitting Method:**
        
        1. **Normalize to reference speed using affinity laws:** All measured points are transformed to equivalent conditions at $N_{ref}$:
           - $Q_{norm} = Q_{meas} \\times \\frac{N_{ref}}{N_{meas}}$
           - $H_{norm} = H_{meas} \\times \\left(\\frac{N_{ref}}{N_{meas}}\\right)^2$
        
        2. **Polynomial regression:** Fit characteristic curves using least-squares:
           - Head: $H_p(Q) = a_n Q^n + a_{n-1} Q^{n-1} + \cdots + a_1 Q + a_0$ (typically $n=2$)
           - Efficiency: $\\eta_p(Q) = b_n Q^n + b_{n-1} Q^{n-1} + \cdots + b_1 Q + b_0$ (bell-shaped)
        
        3. **Scale to target speeds:** Apply affinity laws in reverse to generate curves at any speed.
        
        ---
        
        **References:**
        - Saravanamuttoo, H.I.H., et al. (2017). *Gas Turbine Theory*, 7th Ed., Pearson.
        - Brown, R.N. (2005). *Compressors: Selection and Sizing*, 3rd Ed., Gulf Publishing.
        - ASME PTC 10 (1997). *Performance Test Code on Compressors and Exhausters*.
        """)

        # Check if we have calculated results with speed data
        if 'calculated_results' in st.session_state and st.session_state.calculated_results is not None:
            results_df = st.session_state.calculated_results
            
            # Check if we have the required columns (use actual column names from results)
            required_cols = ['Polytropic Head (kJ/kg)', 'Polytropic Eff (%)']
            flow_col = 'Vol Flow Inlet (m³/hr)'  # This is the actual column name in results
            
            has_required = all(col in results_df.columns for col in required_cols) and flow_col in results_df.columns
            has_speed = 'Speed (RPM)' in results_df.columns
            
            # Check if manufacturer curves are available for single-point adjustment
            mfr_curves = st.session_state.get('compressor_curves', [])
            num_points = len(results_df)
            
            # Mode 1: Single or few points with manufacturer curves - adjust existing curves
            if has_required and num_points >= 1 and num_points < 3 and mfr_curves:
                st.subheader("📐 Adjust Manufacturer Curves from Measured Data")
                st.info(f"""
                **Single-Point Curve Adjustment Mode** ({num_points} data point{'s' if num_points > 1 else ''})
                
                With fewer than 3 data points, you can adjust the manufacturer curves based on measured deviations.
                This applies a correction factor to shift the entire curve set to match your measured performance.
                
                **Method:** 
                - Calculate deviation between measured and expected values at the operating point
                - Apply proportional or offset correction to all curves
                """)
                
                # Calculate deviations from manufacturer curves
                deviations = []
                for idx, row in results_df.iterrows():
                    speed = row.get('Speed (RPM)', 0)
                    flow = row[flow_col]
                    measured_eff = row['Polytropic Eff (%)']
                    measured_head = row['Polytropic Head (kJ/kg)']
                    
                    # Find matching curve (within 5% speed tolerance for single point mode)
                    for curve in mfr_curves:
                        if speed > 0 and abs(speed - curve['speed']) / curve['speed'] < 0.05:
                            curve_flows = np.array(curve['flow'])
                            curve_effs = np.array(curve['efficiency'])
                            curve_heads = np.array(curve['head'])
                            
                            if flow >= min(curve_flows) * 0.9 and flow <= max(curve_flows) * 1.1:
                                expected_eff = np.interp(flow, curve_flows, curve_effs)
                                expected_head = np.interp(flow, curve_flows, curve_heads)
                                
                                eff_ratio = measured_eff / expected_eff if expected_eff > 0 else 1.0
                                head_ratio = measured_head / expected_head if expected_head > 0 else 1.0
                                eff_offset = measured_eff - expected_eff
                                head_offset = measured_head - expected_head
                                
                                deviations.append({
                                    'speed': speed,
                                    'flow': flow,
                                    'measured_eff': measured_eff,
                                    'expected_eff': expected_eff,
                                    'measured_head': measured_head,
                                    'expected_head': expected_head,
                                    'eff_ratio': eff_ratio,
                                    'head_ratio': head_ratio,
                                    'eff_offset': eff_offset,
                                    'head_offset': head_offset
                                })
                            break
                
                if deviations:
                    # Display measured deviations
                    st.write("**Measured Deviations from Manufacturer Curves:**")
                    dev_df = pd.DataFrame(deviations)
                    
                    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
                    avg_eff_ratio = np.mean([d['eff_ratio'] for d in deviations])
                    avg_head_ratio = np.mean([d['head_ratio'] for d in deviations])
                    avg_eff_offset = np.mean([d['eff_offset'] for d in deviations])
                    avg_head_offset = np.mean([d['head_offset'] for d in deviations])
                    
                    with col_d1:
                        st.metric("Eff Ratio", f"{avg_eff_ratio:.3f}", f"{(avg_eff_ratio-1)*100:+.1f}%")
                    with col_d2:
                        st.metric("Head Ratio", f"{avg_head_ratio:.3f}", f"{(avg_head_ratio-1)*100:+.1f}%")
                    with col_d3:
                        st.metric("Eff Offset", f"{avg_eff_offset:+.2f}%")
                    with col_d4:
                        st.metric("Head Offset", f"{avg_head_offset:+.2f} kJ/kg")
                    
                    # Correction method selection
                    correction_method = st.radio(
                        "Correction Method",
                        ["Proportional (Ratio)", "Additive (Offset)"],
                        horizontal=True,
                        help="**Proportional:** Multiply curves by measured/expected ratio. Best for systematic degradation.\n\n**Additive:** Add offset to curves. Best for calibration differences."
                    )
                    
                    if st.button("📈 Generate Adjusted Curves", type='primary', key='gen_adjusted_curves'):
                        adjusted_curves = []
                        
                        for curve in mfr_curves:
                            if correction_method == "Proportional (Ratio)":
                                adjusted_eff = [e * avg_eff_ratio for e in curve['efficiency']]
                                adjusted_head = [h * avg_head_ratio for h in curve['head']]
                            else:
                                adjusted_eff = [e + avg_eff_offset for e in curve['efficiency']]
                                adjusted_head = [h + avg_head_offset for h in curve['head']]
                            
                            # Clamp efficiency to reasonable range
                            adjusted_eff = [max(0, min(100, e)) for e in adjusted_eff]
                            
                            adjusted_curves.append({
                                'speed': curve['speed'],
                                'flow': curve['flow'].copy() if isinstance(curve['flow'], list) else curve['flow'],
                                'head': adjusted_head,
                                'efficiency': adjusted_eff
                            })
                        
                        # Store as generated curves
                        gen_flow_unit = st.session_state.get('curve_flow_unit', 'm³/hr')
                        st.session_state['generated_curves'] = {
                            'curves': adjusted_curves,
                            'flow_unit': gen_flow_unit,
                            'adjustment_method': correction_method,
                            'eff_correction': avg_eff_ratio if correction_method == "Proportional (Ratio)" else avg_eff_offset,
                            'head_correction': avg_head_ratio if correction_method == "Proportional (Ratio)" else avg_head_offset,
                            'source': 'adjusted_from_manufacturer',
                            'num_reference_points': len(deviations)
                        }
                        st.success(f"Generated {len(adjusted_curves)} adjusted curves based on {len(deviations)} measured point(s)")
                        st.rerun()
                else:
                    st.warning("Could not match measured points to manufacturer curves. Check that speeds match within 5% tolerance and flows are within curve range.")
                    
            # Mode 2: 3+ data points - polynomial curve fitting (existing logic)
            elif has_required and len(results_df) >= 3:
                st.subheader("⚙️ Curve Generation Settings")
                col_gen1, col_gen2 = st.columns(2)
                with col_gen1:
                    poly_order_head = st.selectbox("Polynomial Order (Head)", [2, 3, 4], index=0,
                                                    help="Order of polynomial for head vs flow curve. 2nd order (parabola) is typical.")
                with col_gen2:
                    poly_order_eff = st.selectbox("Polynomial Order (Efficiency)", [2, 3, 4], index=0,
                                                   help="Order of polynomial for efficiency vs flow curve. 2nd order gives bell curve.")
                
                # Get unique speeds from data or use default
                if has_speed and results_df['Speed (RPM)'].notna().any():
                    unique_speeds = sorted(results_df['Speed (RPM)'].dropna().unique())
                    if len(unique_speeds) > 0:
                        ref_speed_default = float(np.median(unique_speeds))
                    else:
                        ref_speed_default = 10000.0
                else:
                    ref_speed_default = 10000.0
                    unique_speeds = [ref_speed_default]
                
                col_gen3, col_gen4 = st.columns(2)
                with col_gen3:
                    ref_speed_fit = st.number_input("Reference Speed for Normalization (RPM)", 
                                                     min_value=1000.0, max_value=50000.0, 
                                                     value=ref_speed_default, step=100.0,
                                                     help="All data points will be normalized to this speed for curve fitting")
                with col_gen4:
                    num_curve_points = st.number_input("Points per Curve", min_value=5, max_value=50, value=20,
                                                        help="Number of points to generate on each curve")
                
                # Speeds to generate curves for - prefer manufacturer curve speeds if available
                st.write("**Speeds to Generate Curves (RPM):**")
                
                # Get speeds from manufacturer curves if available
                mfr_curves = st.session_state.get('compressor_curves', [])
                if mfr_curves:
                    mfr_speeds = sorted([c['speed'] for c in mfr_curves])
                    default_speeds = ", ".join([f"{s:.0f}" for s in mfr_speeds])
                    st.caption("ℹ️ Using speeds from manufacturer curves")
                elif has_speed and len(unique_speeds) > 1:
                    default_speeds = ", ".join([f"{s:.0f}" for s in unique_speeds])
                    st.caption("ℹ️ Using speeds from measured data")
                else:
                    default_speeds = "8000, 9000, 10000, 11000, 12000"
                    st.caption("ℹ️ Using default speeds - add manufacturer curves or measured speed data to auto-populate")
                
                target_speeds_str = st.text_input("Enter speeds separated by commas (add or modify as needed)", value=default_speeds,
                                                   help="Curves will be generated for these speeds. Add more speeds or modify as needed.")
                
                try:
                    target_speeds = [float(s.strip()) for s in target_speeds_str.split(",") if s.strip()]
                except (ValueError, AttributeError):
                    target_speeds = [10000.0]
                    st.warning("Invalid speed format. Using default 10000 RPM.")
            
                if st.button("🔧 Generate Curves from Data", type='primary'):
                    try:
                        # Extract data
                        flows = results_df[flow_col].values
                        heads = results_df['Polytropic Head (kJ/kg)'].values
                        effs = results_df['Polytropic Eff (%)'].values
                        
                        if has_speed and results_df['Speed (RPM)'].notna().any():
                            speeds = results_df['Speed (RPM)'].fillna(ref_speed_fit).values
                        else:
                            speeds = np.full(len(flows), ref_speed_fit)
                        
                        # Filter out invalid data
                        valid_mask = (flows > 0) & (heads > 0) & (effs > 0) & (effs <= 100) & (speeds > 0)
                        flows = flows[valid_mask]
                        heads = heads[valid_mask]
                        effs = effs[valid_mask]
                        speeds = speeds[valid_mask]
                        
                        if len(flows) < 3:
                            st.error("Need at least 3 valid data points to generate curves")
                        else:
                            # Normalize to reference speed using fan laws
                            speed_ratio = ref_speed_fit / speeds
                            flows_norm = flows * speed_ratio  # Q ∝ N
                            heads_norm = heads * (speed_ratio ** 2)  # H ∝ N²
                            effs_norm = effs  # Efficiency is approximately speed-independent
                            
                            # Determine flow range - use manufacturer curves scaled by affinity laws
                            if mfr_curves:
                                # Use exact flow range from manufacturer curves (normalized to reference speed)
                                all_mfr_flows = []
                                for curve in mfr_curves:
                                    # Normalize manufacturer curve flows to reference speed using affinity laws: Q ∝ N
                                    curve_speed_ratio = ref_speed_fit / curve['speed']
                                    normalized_flows = [f * curve_speed_ratio for f in curve['flow']]
                                    all_mfr_flows.extend(normalized_flows)
                                
                                flow_min = min(all_mfr_flows)
                                flow_max = max(all_mfr_flows)
                                st.caption(f"ℹ️ Flow range from manufacturer curves (scaled by affinity laws): {flow_min:.0f} to {flow_max:.0f} (at {ref_speed_fit:.0f} RPM)")
                            else:
                                # Fall back to measured data range
                                flow_min = flows_norm.min()
                                flow_max = flows_norm.max()
                                st.caption(f"ℹ️ Flow range from measured data: {flow_min:.0f} to {flow_max:.0f} (at {ref_speed_fit:.0f} RPM)")
                            
                            flow_range = np.linspace(flow_min, flow_max, num_curve_points)
                            
                            # Fit Head vs Flow (typically parabolic: H = a*Q² + b*Q + c)
                            head_coeffs = np.polyfit(flows_norm, heads_norm, poly_order_head)
                            head_poly = np.poly1d(head_coeffs)
                            
                            # Fit Efficiency vs Flow (typically bell-shaped)
                            eff_coeffs = np.polyfit(flows_norm, effs_norm, poly_order_eff)
                            eff_poly = np.poly1d(eff_coeffs)
                            
                            # Calculate R² values for fit quality
                            head_pred = head_poly(flows_norm)
                            eff_pred = eff_poly(flows_norm)
                            
                            ss_res_head = np.sum((heads_norm - head_pred) ** 2)
                            ss_tot_head = np.sum((heads_norm - np.mean(heads_norm)) ** 2)
                            r2_head = 1 - (ss_res_head / ss_tot_head) if ss_tot_head > 0 else 0
                            
                            ss_res_eff = np.sum((effs_norm - eff_pred) ** 2)
                            ss_tot_eff = np.sum((effs_norm - np.mean(effs_norm)) ** 2)
                            r2_eff = 1 - (ss_res_eff / ss_tot_eff) if ss_tot_eff > 0 else 0
                            
                            st.write(f"**Curve Fit Quality:** Head R² = {r2_head:.3f}, Efficiency R² = {r2_eff:.3f}")
                            
                            if r2_head < 0.7 or r2_eff < 0.7:
                                st.warning("⚠️ Low R² values indicate poor curve fit. Consider adding more data points or adjusting polynomial order.")
                            
                            # Generate curves for each target speed
                            generated_curves = []
                            
                            for target_speed in target_speeds:
                                speed_scale = target_speed / ref_speed_fit
                                
                                # Scale from reference speed to target speed
                                curve_flows = (flow_range * speed_scale).tolist()  # Q ∝ N
                                curve_heads = (head_poly(flow_range) * (speed_scale ** 2)).tolist()  # H ∝ N²
                                curve_effs = np.clip(eff_poly(flow_range), 0, 100).tolist()  # Efficiency stays same
                                
                                generated_curves.append({
                                    'speed': target_speed,
                                    'flow': curve_flows,
                                    'head': curve_heads,
                                    'efficiency': curve_effs
                                })
                            
                            # Store generated curves
                            # Use the curve flow unit from session or default to m3/hr
                            gen_flow_unit = st.session_state.get('curve_flow_unit', 'm³/hr')
                            st.session_state['generated_curves'] = {
                                'curves': generated_curves,
                                'flow_unit': gen_flow_unit,
                                'ref_speed': ref_speed_fit,
                                'r2_head': r2_head,
                                'r2_eff': r2_eff,
                                'head_coeffs': head_coeffs.tolist(),
                                'eff_coeffs': eff_coeffs.tolist()
                            }
                            
                            st.success(f"Generated {len(generated_curves)} performance curves from {len(flows)} data points")
                            
                    except Exception as e:
                        st.error(f"Failed to generate curves: {e}")
                
                # Display generated curves
                if 'generated_curves' in st.session_state and st.session_state.get('generated_curves'):
                    gen_data = st.session_state['generated_curves']
                    
                    st.divider()
                    st.write(f"**Generated Curves** (R²: Head={gen_data.get('r2_head', 0):.3f}, Eff={gen_data.get('r2_eff', 0):.3f})")
                    
                    # Plot the fitted curves with original data
                    fig_fit = go.Figure()
                    
                    # Add original data points
                    if 'calculated_results' in st.session_state and st.session_state.calculated_results is not None:
                        orig_df = st.session_state.calculated_results
                        if flow_col in orig_df.columns and 'Polytropic Head (kJ/kg)' in orig_df.columns:
                            fig_fit.add_trace(go.Scatter(
                                x=orig_df[flow_col],
                                y=orig_df['Polytropic Head (kJ/kg)'],
                                mode='markers',
                                name='Measured Data',
                                marker=dict(size=10, color='red', symbol='circle')
                            ))
                    
                    # Add fitted curves
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
                    for i, curve in enumerate(gen_data['curves']):
                        color = colors[i % len(colors)]
                        fig_fit.add_trace(go.Scatter(
                            x=curve['flow'],
                            y=curve['head'],
                            mode='lines',
                            name=f"{curve['speed']:.0f} RPM",
                            line=dict(color=color, width=2)
                        ))
                    
                    fig_fit.update_layout(
                        title="Generated Head Curves vs Measured Data",
                        xaxis_title=f"Flow ({gen_data.get('flow_unit', 'm³/hr')})",
                        yaxis_title="Polytropic Head (kJ/kg)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        height=450
                    )
                    st.plotly_chart(fig_fit, use_container_width=True)
                    
                    # Add efficiency plot
                    fig_eff = go.Figure()
                    
                    # Add original efficiency data points
                    if 'calculated_results' in st.session_state and st.session_state.calculated_results is not None:
                        orig_df = st.session_state.calculated_results
                        if flow_col in orig_df.columns and 'Polytropic Eff (%)' in orig_df.columns:
                            fig_eff.add_trace(go.Scatter(
                                x=orig_df[flow_col],
                                y=orig_df['Polytropic Eff (%)'],
                                mode='markers',
                                name='Measured Data',
                                marker=dict(size=10, color='red', symbol='circle')
                            ))
                    
                    # Add fitted efficiency curves
                    for i, curve in enumerate(gen_data['curves']):
                        color = colors[i % len(colors)]
                        fig_eff.add_trace(go.Scatter(
                            x=curve['flow'],
                            y=curve['efficiency'],
                            mode='lines',
                            name=f"{curve['speed']:.0f} RPM",
                            line=dict(color=color, width=2)
                        ))
                    
                    fig_eff.update_layout(
                        title="Generated Efficiency Curves vs Measured Data",
                        xaxis_title=f"Flow ({gen_data.get('flow_unit', 'm³/hr')})",
                        yaxis_title="Polytropic Efficiency (%)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        height=400
                    )
                    st.plotly_chart(fig_eff, use_container_width=True)
                    
                    # Show curve data tables
                    with st.expander("📋 View Curve Data Tables"):
                        for curve in gen_data['curves']:
                            gen_df = pd.DataFrame({
                                f"Flow ({gen_data.get('flow_unit', 'm3/hr')})": [f"{v:.2f}" for v in curve['flow']],
                                'Head (kJ/kg)': [f"{v:.2f}" for v in curve['head']],
                                'Eff (%)': [f"{v:.1f}" for v in curve['efficiency']]
                            })
                            st.write(f"**{curve['speed']:.0f} RPM:**")
                            st.dataframe(gen_df, use_container_width=True, height=100)
                    
                    st.divider()
                    st.subheader("💾 Save Generated Curves")
                    
                    # Prepare JSON data for download
                    gen_json = json.dumps({
                        'flow_unit': gen_data.get('flow_unit', 'm3/hr'),
                        'generated_from_data': True,
                        'generation_date': str(pd.Timestamp.now()),
                        'reference_speed_rpm': gen_data.get('ref_speed'),
                        'fit_quality': {
                            'r2_head': round(gen_data.get('r2_head', 0), 4),
                            'r2_eff': round(gen_data.get('r2_eff', 0), 4)
                        },
                        'polynomial_coefficients': {
                            'head': gen_data.get('head_coeffs'),
                            'efficiency': gen_data.get('eff_coeffs')
                        },
                        'curves': gen_data['curves']
                    }, indent=2)
                    
                    # Action buttons in clear layout
                    col_save1, col_save2, col_save3 = st.columns(3)
                    with col_save1:
                        st.download_button(
                            label="📥 Save Curves to JSON File",
                            data=gen_json,
                            file_name="generated_compressor_curves.json",
                            mime="application/json",
                            key='download_generated',
                            type='primary'
                        )
                    with col_save2:
                        if st.button("✅ Use as Reference Curves", key='use_generated', 
                                     help="Set these curves as the manufacturer/reference curves for deviation analysis"):
                            st.session_state['compressor_curves'] = gen_data['curves']
                            st.session_state['curve_flow_unit'] = gen_data.get('flow_unit', 'm3/hr')
                            st.success("Generated curves are now set as reference curves!")
                            st.rerun()
                    with col_save3:
                        if st.button("🗑️ Clear Generated Curves", key='clear_generated'):
                            st.session_state['generated_curves'] = None
                            st.rerun()
                    
                    st.caption("💡 **Tip:** Save to JSON to preserve curves for future sessions, or 'Use as Reference' to compare future measurements against these curves.")
            else:
                # Check if we have 1-2 points but no manufacturer curves
                if num_points >= 1 and num_points < 3 and not mfr_curves:
                    st.warning(f"""
                    **{num_points} data point{'s' if num_points > 1 else ''} available** - Not enough for polynomial curve fitting (need 3+).
                    
                    **Option:** Load manufacturer curves first, then we can adjust them based on your measured data.
                    """)
                else:
                    st.info("Need at least 3 data points with valid Polytropic Head and Efficiency values for polynomial curve fitting.")
                if st.button("🔄 Run Calculations Now", key='run_calc_from_measured', type='primary'):
                    st.session_state['trigger_calculation'] = True
                    st.rerun()
        else:
            st.warning("No calculated results available yet. Click the button below to run compressor calculations first.")
            if st.button("🔄 Run Compressor Calculations", key='run_calc_from_measured_main', type='primary'):
                st.session_state['trigger_calculation'] = True
                st.rerun()

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

# Check if calculation was triggered from "Generate Curves from Measured Data" section
trigger_calculation = st.session_state.pop('trigger_calculation', False)

# Calculate button
if st.button('Calculate Compressor Performance', type='primary') or trigger_calculation:
    fluid_composition = get_fluid_composition()
    
    if not fluid_composition or len(fluid_composition) == 0:
        st.error('Please define a valid fluid composition.')
    elif edited_data.empty or edited_data.dropna().empty:
        st.error('Please enter operating data before calculating.')
    else:
        calc_method = st.session_state.get('calc_method', 'NeqSim Process Model (Detailed)')
        eos_name = st.session_state.get('eos_model', 'GERG-2008')
        spinner_msg = f'Calculating compressor performance using {eos_name}...'
        if calc_method == "NeqSim Process Model (Detailed)":
            spinner_msg = f'Calculating using NeqSim detailed model ({eos_name}, {st.session_state["num_calc_steps"]} steps)...'
        elif calc_method == "NeqSim Process Model (Simple)":
            spinner_msg = f'Calculating using NeqSim simple model ({eos_name})...'
        
        with st.spinner(spinner_msg):
            try:
                jneqsim.util.database.NeqSimDataBase.setCreateTemporaryTables(True)
                
                results = []
                
                # Get selected units
                flow_unit = st.session_state['flow_unit']
                pressure_unit = st.session_state['pressure_unit']
                temp_unit = st.session_state['temperature_unit']
                
                for idx, row in edited_data.dropna().iterrows():
                    speed_rpm = row.get('Speed (RPM)', 0.0)
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
                    inlet_fluid = fluid(get_selected_eos_model())
                    for comp_name, comp_moles in fluid_composition.items():
                        inlet_fluid.addComponent(comp_name, float(comp_moles))
                    inlet_fluid.setMixingRule('classic')
                    
                    inlet_fluid.setPressure(float(p_in), 'bara')
                    inlet_fluid.setTemperature(float(t_in), 'C')
                    TPflash(inlet_fluid)
                    inlet_fluid.initThermoProperties()
                    inlet_fluid.initPhysicalProperties()
                    
                    # Get inlet properties
                    z_in = inlet_fluid.getZ()
                    h_in = inlet_fluid.getEnthalpy("kJ/kg")  # Specific enthalpy in kJ/kg
                    cp_in = inlet_fluid.getCp("kJ/kgK")  # Specific Cp in kJ/kg/K
                    cv_in = inlet_fluid.getCv("kJ/kgK")  # Specific Cv in kJ/kg/K
                    kappa_in = cp_in / cv_in if cv_in > 0 else inlet_fluid.getGamma()
                    MW = inlet_fluid.getMolarMass() * 1000  # g/mol -> kg/kmol
                    rho_in = inlet_fluid.getDensity()  # kg/m3
                    T_in_K = t_in + 273.15
                    T_out_K = t_out + 273.15
                    pr = p_out / p_in
                    
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
                        std_fluid = fluid(get_selected_eos_model())
                        for comp_name, comp_moles in fluid_composition.items():
                            std_fluid.addComponent(comp_name, float(comp_moles))
                        std_fluid.setMixingRule('classic')
                        std_fluid.setPressure(1.01325, 'bara')
                        std_fluid.setTemperature(15.0, 'C')
                        TPflash(std_fluid)
                        std_fluid.initThermoProperties()
                        std_fluid.initPhysicalProperties()
                        rho_std = std_fluid.getDensity()  # kg/m3 at std conditions
                        # MSm3/day to kg/s: flow * 1e6 m3/day * rho_std / 86400 s/day
                        mass_flow = flow_value * 1e6 * rho_std / 86400.0
                    else:
                        mass_flow = flow_value  # Default, assume kg/s
                    
                    # Create outlet fluid at actual conditions
                    outlet_fluid = fluid(get_selected_eos_model())
                    for comp_name, comp_moles in fluid_composition.items():
                        outlet_fluid.addComponent(comp_name, float(comp_moles))
                    outlet_fluid.setMixingRule('classic')
                    outlet_fluid.setPressure(float(p_out), 'bara')
                    outlet_fluid.setTemperature(float(t_out), 'C')
                    TPflash(outlet_fluid)
                    outlet_fluid.initThermoProperties()
                    outlet_fluid.initPhysicalProperties()
                    
                    # Get outlet properties
                    z_out = outlet_fluid.getZ()
                    h_out = outlet_fluid.getEnthalpy("kJ/kg")  # Specific enthalpy in kJ/kg
                    kappa_out = outlet_fluid.getGamma()  # Use direct gamma for outlet
                    rho_out = outlet_fluid.getDensity()
                    
                    # Common calculated values
                    actual_work = h_out - h_in  # kJ/kg
                    kappa_avg = (kappa_in + kappa_out) / 2
                    z_avg = (z_in + z_out) / 2
                    
                    # Check which calculation method to use
                    calc_method = st.session_state.get('calc_method', 'NeqSim Process Model (Detailed)')
                    
                    if calc_method in ["NeqSim Process Model (Detailed)", "NeqSim Process Model (Simple)"]:
                        # Use NeqSim process compressor with detailed polytropic method
                        # Create a stream for the compressor inlet
                        process_fluid = fluid(get_selected_eos_model())
                        for comp_name, comp_moles in fluid_composition.items():
                            process_fluid.addComponent(comp_name, float(comp_moles))
                        process_fluid.setMixingRule('classic')
                        process_fluid.setPressure(float(p_in), 'bara')
                        process_fluid.setTemperature(float(t_in), 'C')
                        process_fluid.setTotalFlowRate(float(mass_flow), 'kg/sec')
                        TPflash(process_fluid)
                        
                        # Create stream and compressor
                        inlet_stream = jneqsim.process.equipment.stream.Stream("inlet", process_fluid)
                        inlet_stream.run()
                        
                        compressor = jneqsim.process.equipment.compressor.Compressor("compressor", inlet_stream)
                        compressor.setOutletPressure(float(p_out), "bara")
                        compressor.setUsePolytropicCalc(True)
                        
                        # Set polytropic method based on user selection
                        if calc_method == "NeqSim Process Model (Detailed)":
                            compressor.setPolytropicMethod("detailed")
                            compressor.setNumberOfCompressorCalcSteps(st.session_state['num_calc_steps'])
                        else:
                            # Simple mode - faster calculation
                            compressor.setPolytropicMethod("schultz")
                        
                        # Debug: verify method settings (only on first row)
                        if idx == 0:
                            method_used = compressor.getPolytropicMethod() if hasattr(compressor, 'getPolytropicMethod') else "N/A"
                            steps_used = compressor.getNumberOfCompressorCalcSteps() if hasattr(compressor, 'getNumberOfCompressorCalcSteps') else "N/A"
                            st.caption(f"🔧 Method: {method_used}, Steps: {steps_used}")
                        
                        # Solve for polytropic efficiency based on measured outlet temperature
                        # Convert outlet temperature to Kelvin for solveEfficiency method
                        t_out_K = t_out + 273.15
                        eta_poly = compressor.solveEfficiency(t_out_K)
                        
                        # Validate the solved efficiency - must be between 0 and 1 (0-100%)
                        # If invalid, the measured outlet temperature may be thermodynamically inconsistent
                        eta_poly_float = float(eta_poly) if eta_poly is not None else None
                        is_valid = eta_poly_float is not None and not np.isnan(eta_poly_float) and eta_poly_float > 0 and eta_poly_float <= 1.0
                        
                        if not is_valid:
                            # Invalid efficiency - outlet temperature may be unrealistic
                            # Fall back to Schultz method for this point
                            eta_poly_str = f"{eta_poly_float:.3f}" if eta_poly_float is not None else "None"
                            st.warning(f"⚠️ Row {idx}: NeqSim returned η={eta_poly_str} for T_out={t_out}°C, P_ratio={p_out/p_in:.2f}. Using Schultz method as fallback.")
                            
                            # Use Schultz analytical method as fallback
                            # Calculate polytropic exponent from measured data
                            n = calculate_polytropic_exponent(T_in_K, T_out_K, pr, kappa_avg)
                            
                            # Polytropic efficiency using Schultz
                            # From: n = 1 / (1 - (κ-1)/κ / η_p)
                            # Solving for η_p: η_p = n*(κ-1) / (κ*(n-1))
                            if n > 1 and kappa_avg > 1:
                                eta_poly_calc = (n * (kappa_avg - 1)) / (kappa_avg * (n - 1))
                                # Check if calculated efficiency is unrealistic (>100%)
                                if eta_poly_calc > 1.0:
                                    st.error(f"❌ Row {idx}: Schultz method gives efficiency {eta_poly_calc*100:.1f}% (>100%), which is thermodynamically impossible. "
                                            f"The outlet temperature {t_out}°C is too low for compression from {p_in} to {p_out} bara. "
                                            f"Please increase the outlet temperature. Setting efficiency to 100% for display.")
                                    eta_poly = 1.0  # Cap at 100% for display but flag the issue
                                elif eta_poly_calc < 0.3:
                                    st.warning(f"⚠️ Row {idx}: Calculated efficiency is very low ({eta_poly_calc*100:.1f}%). Check measured data.")
                                    eta_poly = eta_poly_calc
                                else:
                                    eta_poly = eta_poly_calc
                            else:
                                eta_poly = 0.75  # Default fallback
                            
                            # Calculate polytropic head using enthalpy-based method
                            # Hp = η_p × actual_work (consistent with NeqSim)
                            polytropic_head = actual_work * eta_poly if actual_work > 0 and eta_poly > 0 else 0
                            eta_isen = eta_poly * 0.98
                            power_kW = mass_flow * actual_work
                            power_MW = power_kW / 1000
                        else:
                            # Valid efficiency from NeqSim
                            # Set the solved efficiency and run the compressor to get all results
                            compressor.setPolytropicEfficiency(eta_poly)
                            compressor.run()
                            
                            # Get results from compressor after running
                            eta_isen = compressor.getIsentropicEfficiency()
                            polytropic_head = compressor.getPolytropicFluidHead()  # kJ/kg
                            n = compressor.getPolytropicExponent()
                            
                            # Power from MEASURED enthalpy difference for consistency
                            power_kW = mass_flow * actual_work  # kW
                            power_MW = power_kW / 1000  # MW
                            
                            # If polytropic exponent is still 0, calculate from measured data
                            if n == 0 or n is None:
                                n = calculate_polytropic_exponent(T_in_K, T_out_K, pr, kappa_avg)
                        
                        vol_flow_in = mass_flow / rho_in * 3600  # m³/hr
                        mass_flow_kg_hr = mass_flow * 3600  # kg/hr
                        
                        results.append({
                            'Speed (RPM)': speed_rpm,
                            'Mass Flow (kg/hr)': mass_flow_kg_hr,
                            'Inlet P (bara)': p_in,
                            'Outlet P (bara)': p_out,
                            'Inlet T (°C)': t_in,
                            'Outlet T (°C)': t_out,  # Use measured outlet temperature
                            'Pressure Ratio': pr,
                            'Density Inlet (kg/m³)': rho_in,
                            'Density Outlet (kg/m³)': rho_out,
                            'Z inlet': z_in,
                            'Z outlet': z_out,
                            'κ inlet': kappa_in,
                            'κ outlet': kappa_out,
                            'Polytropic Exp (n)': n,
                            'Isentropic Eff (%)': eta_isen * 100 if eta_isen is not None else None,
                            'Polytropic Eff (%)': eta_poly * 100,
                            'Polytropic Head (kJ/kg)': polytropic_head,
                            'Actual Work (kJ/kg)': actual_work,
                            'Power (kW)': power_kW,
                            'Power (MW)': power_MW,
                            'Vol Flow Inlet (m³/hr)': vol_flow_in,
                        })
                    else:
                        # Use Schultz analytical method
                        # Create isentropic outlet fluid (same entropy as inlet)
                        isentropic_fluid = fluid(get_selected_eos_model())
                        for comp_name, comp_moles in fluid_composition.items():
                            isentropic_fluid.addComponent(comp_name, float(comp_moles))
                        isentropic_fluid.setPressure(float(p_out), 'bara')
                        isentropic_fluid.setTemperature(float(t_out), 'C')  # Initial guess
                        
                        # Find isentropic temperature using PS flash
                        thermoOps = jneqsim.thermodynamicoperations.ThermodynamicOperations(isentropic_fluid)
                        s_in_total = inlet_fluid.getEntropy()  # Total entropy at inlet
                        try:
                            thermoOps.PSflash(s_in_total)
                            isentropic_fluid.initProperties()
                            h_out_isen = isentropic_fluid.getEnthalpy("kJ/kg")
                        except Exception:
                            # Fallback: estimate isentropic enthalpy using ideal gas relation
                            t_out_isen = T_in_K * pr**((kappa_avg-1)/kappa_avg) - 273.15
                            h_out_isen = h_in + cp_in * (t_out_isen - t_in)
                        
                        # Calculate isentropic work
                        isentropic_work = h_out_isen - h_in  # kJ/kg
                        
                        # Isentropic efficiency
                        eta_isen = isentropic_work / actual_work if actual_work > 0 else 0
                        
                        # Calculate polytropic exponent from measured data
                        n = calculate_polytropic_exponent(T_in_K, T_out_K, pr, kappa_avg)
                        
                        # Polytropic efficiency from Schultz formula
                        # From: n = 1 / (1 - (κ-1)/κ / η_p)
                        # Solving for η_p: η_p = n*(κ-1) / (κ*(n-1))
                        if n > 1 and kappa_avg > 1:
                            eta_poly = (n * (kappa_avg - 1)) / (kappa_avg * (n - 1))
                            # Check for unrealistic efficiency values
                            if eta_poly > 1.0:
                                st.error(f"❌ Row {idx}: Schultz method gives efficiency {eta_poly*100:.1f}% (>100%). "
                                        f"The outlet temperature {t_out}°C is too low for compression from {p_in:.1f} to {p_out:.1f} bara. "
                                        f"Please check measured outlet temperature.")
                                eta_poly = 1.0  # Cap at 100% for calculation
                            elif eta_poly < 0.4:
                                st.warning(f"⚠️ Row {idx}: Calculated efficiency is very low ({eta_poly*100:.1f}%). Check measured data.")
                        else:
                            eta_poly = eta_isen * 0.98 if eta_isen > 0 else 0.75  # Approximation
                        
                        # Polytropic head (enthalpy-based, consistent with NeqSim)
                        polytropic_head = actual_work * eta_poly if actual_work > 0 and eta_poly > 0 else 0
                        
                        # Power calculation
                        power_kW = mass_flow * actual_work  # Gas power in kW
                        power_MW = power_kW / 1000  # MW
                        
                        # Volume flow at inlet conditions
                        vol_flow_in = mass_flow / rho_in * 3600  # m³/hr
                        mass_flow_kg_hr = mass_flow * 3600  # kg/hr
                        
                        results.append({
                            'Speed (RPM)': speed_rpm,
                            'Mass Flow (kg/hr)': mass_flow_kg_hr,
                            'Inlet P (bara)': p_in,
                            'Outlet P (bara)': p_out,
                            'Inlet T (°C)': t_in,
                            'Outlet T (°C)': t_out,
                            'Pressure Ratio': pr,
                            'Density Inlet (kg/m³)': rho_in,
                            'Density Outlet (kg/m³)': rho_out,
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
                
                # Store results in session state for curve generation
                st.session_state.calculated_results = results_df
                
                # Calculate deviation from manufacturer curves if available
                mfr_curves = st.session_state.get('compressor_curves', [])
                if mfr_curves:
                    # Add deviation columns
                    results_df['Expected Eff (%)'] = np.nan
                    results_df['Expected Head (kJ/kg)'] = np.nan
                    results_df['Eff Deviation (%)'] = np.nan
                    results_df['Head Deviation (%)'] = np.nan
                    results_df['Status'] = ''
                    
                    for idx, row in results_df.iterrows():
                        speed = row['Speed (RPM)']
                        flow = row['Vol Flow Inlet (m³/hr)']
                        
                        # Find matching curve (within 1% speed tolerance)
                        matching_curve = None
                        for curve in mfr_curves:
                            if curve['speed'] > 0 and abs(speed - curve['speed']) / curve['speed'] < 0.01:
                                matching_curve = curve
                                break
                        
                        if matching_curve:
                            # Interpolate expected values from curve
                            curve_flows = np.array(matching_curve['flow'])
                            curve_effs = np.array(matching_curve['efficiency'])
                            curve_heads = np.array(matching_curve['head'])
                            
                            if flow >= min(curve_flows) and flow <= max(curve_flows):
                                expected_eff = np.interp(flow, curve_flows, curve_effs)
                                expected_head = np.interp(flow, curve_flows, curve_heads)
                                
                                eff_deviation = ((row['Polytropic Eff (%)'] - expected_eff) / expected_eff) * 100
                                head_deviation = ((row['Polytropic Head (kJ/kg)'] - expected_head) / expected_head) * 100
                                
                                results_df.at[idx, 'Expected Eff (%)'] = expected_eff
                                results_df.at[idx, 'Expected Head (kJ/kg)'] = expected_head
                                results_df.at[idx, 'Eff Deviation (%)'] = eff_deviation
                                results_df.at[idx, 'Head Deviation (%)'] = head_deviation
                                
                                # Determine status
                                if abs(eff_deviation) <= 2 and abs(head_deviation) <= 3:
                                    results_df.at[idx, 'Status'] = '✅ OK'
                                elif abs(eff_deviation) <= 5 and abs(head_deviation) <= 7:
                                    results_df.at[idx, 'Status'] = '⚠️ Warning'
                                else:
                                    results_df.at[idx, 'Status'] = '❌ Check'
                            else:
                                results_df.at[idx, 'Status'] = '⚠️ Out of range'
                
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
                
                # Check if deviation columns exist (curves were provided)
                has_deviation = 'Status' in display_df.columns and display_df['Status'].notna().any()
                
                if has_deviation:
                    # Show performance comparison section
                    st.subheader("🔍 Performance vs Expected (from Curves)")
                    
                    # Summary status
                    status_counts = display_df['Status'].value_counts()
                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                    with col_s1:
                        ok_count = status_counts.get('✅ OK', 0)
                        st.metric("✅ Within Spec", ok_count)
                    with col_s2:
                        warn_count = status_counts.get('⚠️ Warning', 0) + status_counts.get('⚠️ Out of range', 0)
                        st.metric("⚠️ Warning", warn_count)
                    with col_s3:
                        check_count = status_counts.get('❌ Check', 0)
                        st.metric("❌ Needs Check", check_count)
                    with col_s4:
                        if display_df['Eff Deviation (%)'].notna().any():
                            avg_eff_dev = display_df['Eff Deviation (%)'].mean()
                            st.metric("Avg Eff Deviation", f"{avg_eff_dev:+.1f}%")
                    
                    st.divider()
                    
                    # Show deviation table
                    deviation_columns = [
                        'Speed (RPM)', 'Vol Flow Inlet (m³/hr)', 'Polytropic Eff (%)', 'Expected Eff (%)',
                        'Eff Deviation (%)', 'Polytropic Head (kJ/kg)', 'Expected Head (kJ/kg)', 
                        'Head Deviation (%)', 'Status'
                    ]
                    
                    # Only show columns that exist
                    available_dev_cols = [c for c in deviation_columns if c in display_df.columns]
                    
                    def color_status(val):
                        if pd.isna(val):
                            return ''
                        if '✅' in str(val):
                            return 'background-color: #d4edda'
                        elif '⚠️' in str(val):
                            return 'background-color: #fff3cd'
                        elif '❌' in str(val):
                            return 'background-color: #f8d7da'
                        return ''
                    
                    def color_deviation(val):
                        if pd.isna(val):
                            return ''
                        if abs(val) <= 2:
                            return 'background-color: #d4edda'
                        elif abs(val) <= 5:
                            return 'background-color: #fff3cd'
                        else:
                            return 'background-color: #f8d7da'
                    
                    styled_dev_df = display_df[available_dev_cols].style.format({
                        'Speed (RPM)': '{:.0f}',
                        'Vol Flow Inlet (m³/hr)': '{:.1f}',
                        'Polytropic Eff (%)': '{:.2f}',
                        'Expected Eff (%)': '{:.2f}',
                        'Eff Deviation (%)': '{:+.2f}',
                        'Polytropic Head (kJ/kg)': '{:.2f}',
                        'Expected Head (kJ/kg)': '{:.2f}',
                        'Head Deviation (%)': '{:+.2f}',
                    }, na_rep='-').map(color_status, subset=['Status']).map(
                        color_deviation, subset=['Eff Deviation (%)', 'Head Deviation (%)']
                    )
                    
                    st.dataframe(styled_dev_df, use_container_width=True)
                    
                    st.caption("**Status Legend:** ✅ OK = Within ±2% eff / ±3% head | ⚠️ Warning = Within ±5% eff / ±7% head | ❌ Check = Outside tolerance")
                    
                    st.divider()
                
                # Standard results table
                st.subheader("📋 Full Calculation Details")
                
                display_columns = [
                    'Speed (RPM)', 'Mass Flow (kg/hr)', 'Vol Flow Inlet (m³/hr)', 'Pressure Ratio', 
                    'Density Inlet (kg/m³)', 'Density Outlet (kg/m³)',
                    'Polytropic Eff (%)',
                    'Polytropic Head (kJ/kg)', 'Power (MW)', 'Z inlet', 'Z outlet',
                    'κ inlet', 'κ outlet', 'Polytropic Exp (n)'
                ]
                
                st.dataframe(
                    display_df[display_columns].style.format({
                        'Speed (RPM)': '{:.0f}',
                        'Mass Flow (kg/hr)': '{:.1f}',
                        'Vol Flow Inlet (m³/hr)': '{:.1f}',
                        'Pressure Ratio': '{:.3f}',
                        'Density Inlet (kg/m³)': '{:.3f}',
                        'Density Outlet (kg/m³)': '{:.3f}',
                        'Polytropic Eff (%)': '{:.2f}',
                        'Polytropic Head (kJ/kg)': '{:.2f}',
                        'Power (MW)': '{:.3f}',
                        'Z inlet': '{:.4f}',
                        'Z outlet': '{:.4f}',
                        'κ inlet': '{:.4f}',
                        'κ outlet': '{:.4f}',
                        'Polytropic Exp (n)': '{:.4f}',
                    }, na_rep='-'),
                    use_container_width=True
                )
                
                st.divider()
                
                # Plots
                st.subheader("📈 Performance Curves")
                
                # Get the flow column for x-axis based on selected unit
                flow_unit = st.session_state['flow_unit']
                
                # Calculate flow in the input unit for x-axis
                if flow_unit in ["kg/s"]:
                    x_flow = results_df['Mass Flow (kg/hr)'] / 3600
                    x_label = f'Flow Rate ({flow_unit})'
                elif flow_unit == "kg/hr":
                    x_flow = results_df['Mass Flow (kg/hr)']
                    x_label = f'Flow Rate ({flow_unit})'
                elif flow_unit in ["m3/hr", "Am3/hr"]:
                    x_flow = results_df['Vol Flow Inlet (m³/hr)']
                    x_label = f'Flow Rate ({flow_unit})'
                elif flow_unit == "MSm3/day":
                    # Convert back from kg/s to MSm3/day
                    # Use dropna() to match the same rows used in calculations
                    x_flow = edited_data.dropna()['Flow Rate'].values[:len(results_df)]
                    x_label = f'Flow Rate ({flow_unit})'
                else:
                    x_flow = results_df['Mass Flow (kg/hr)']
                    x_label = 'Flow Rate (kg/hr)'
                
                # Store x_flow in results for use in plots
                results_df['Plot Flow'] = x_flow.values if hasattr(x_flow, 'values') else x_flow
                
                # Get compressor curves if enabled
                show_mfr_curves = st.session_state.get('show_compressor_curves', False)
                mfr_curves = st.session_state.get('compressor_curves', [])
                curve_flow_unit = st.session_state.get('curve_flow_unit', flow_unit)
                
                # Get generated/adjusted curves if available (from measured data fitting)
                gen_curves_data = st.session_state.get('generated_curves', None)
                gen_curves = gen_curves_data.get('curves', []) if gen_curves_data else []
                
                # Get MW-corrected curves if available (from "Generate Updated Curves for New Gas")
                corrected_curves_data = st.session_state.get('corrected_curves', None)
                corrected_curves = corrected_curves_data.get('curves', []) if corrected_curves_data else []
                
                # Color palette for curves and measured points
                curve_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', 
                               '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
                
                # Get unique speeds from measured data
                unique_speeds = sorted(results_df['Speed (RPM)'].unique())
                
                # Build a mapping from curve speeds to colors for matching measured points
                speed_to_color = {}
                
                # First, assign colors to curve speeds if curves exist
                if mfr_curves:
                    for i, curve in enumerate(mfr_curves):
                        speed_to_color[curve['speed']] = curve_colors[i % len(curve_colors)]
                
                # Then, assign colors to measured speeds that don't match any curve
                color_index = len(mfr_curves) if mfr_curves else 0
                for speed in unique_speeds:
                    # Check if this speed already has a color (matches a curve)
                    has_color = speed in speed_to_color
                    if not has_color:
                        # Check for close match (within 1% tolerance)
                        for curve_speed in speed_to_color.keys():
                            if curve_speed > 0 and abs(speed - curve_speed) / curve_speed < 0.01:
                                has_color = True
                                break
                    
                    if not has_color:
                        # Assign a new color from the palette
                        speed_to_color[speed] = curve_colors[color_index % len(curve_colors)]
                        color_index += 1
                
                # Function to get color for a speed
                def get_speed_color(speed, default_color='#1f77b4'):
                    if speed in speed_to_color:
                        return speed_to_color[speed]
                    # Try to find closest matching curve speed (within 1% tolerance)
                    for curve_speed, color in speed_to_color.items():
                        if curve_speed > 0 and abs(speed - curve_speed) / curve_speed < 0.01:
                            return color
                    return default_color
                
                # Show legend for curve types if multiple curve sources are present
                curve_legend_parts = []
                if show_mfr_curves and mfr_curves:
                    curve_legend_parts.append("**···** Original (dotted)")
                if corrected_curves:
                    curve_legend_parts.append("**- -** MW-Corrected (dashed)")
                if gen_curves:
                    curve_legend_parts.append("**—** Fitted (solid)")
                if curve_legend_parts:
                    st.caption("**Curve Legend:** " + " | ".join(curve_legend_parts) + " | **●** Measured Data")
                
                tab1, tab2, tab3 = st.tabs(["Polytropic Efficiency vs Flow", "Head vs Flow", "Power vs Flow"])
                
                with tab1:
                    fig_eff = go.Figure()
                    
                    # Add manufacturer curves first (as background - dotted lines)
                    if show_mfr_curves and mfr_curves:
                        for i, curve in enumerate(mfr_curves):
                            color = curve_colors[i % len(curve_colors)]
                            fig_eff.add_trace(go.Scatter(
                                x=curve['flow'],
                                y=curve['efficiency'],
                                mode='lines',
                                name=f"Original {curve['speed']:.0f} RPM",
                                line=dict(width=2, color=color, dash='dot'),
                                opacity=0.6,
                                legendgroup=f"speed_{curve['speed']}"
                            ))
                    
                    # Add MW-corrected curves (dashed lines - from new gas composition)
                    if show_mfr_curves and corrected_curves:
                        for i, curve in enumerate(corrected_curves):
                            color = curve_colors[i % len(curve_colors)]
                            fig_eff.add_trace(go.Scatter(
                                x=curve['flow'],
                                y=curve['efficiency'],
                                mode='lines',
                                name=f"MW-Corrected {curve['speed']:.0f} RPM",
                                line=dict(width=2, color=color, dash='dash'),
                                opacity=0.8,
                                legendgroup=f"speed_{curve['speed']}_corr"
                            ))
                    
                    # Add generated/adjusted curves (solid lines - from measured data)
                    if gen_curves:
                        for i, curve in enumerate(gen_curves):
                            color = curve_colors[i % len(curve_colors)]
                            fig_eff.add_trace(go.Scatter(
                                x=curve['flow'],
                                y=curve['efficiency'],
                                mode='lines',
                                name=f"Fitted {curve['speed']:.0f} RPM",
                                line=dict(width=3, color=color),
                                legendgroup=f"speed_{curve['speed']}"
                            ))
                    
                    # Add calculated data points grouped by speed
                    for speed in sorted(unique_speeds):
                        speed_data = results_df[results_df['Speed (RPM)'] == speed]
                        color = get_speed_color(speed)
                        # Check if this speed matches a curve
                        matches_curve = speed in speed_to_color or any(abs(speed - cs) / cs < 0.01 for cs in speed_to_color.keys()) if speed_to_color else False
                        fig_eff.add_trace(go.Scatter(
                            x=speed_data['Plot Flow'],
                            y=speed_data['Polytropic Eff (%)'],
                            mode='markers',
                            name=f'Measured @ {speed:.0f} RPM',
                            marker=dict(size=12, color=color, symbol='circle', 
                                       line=dict(width=2, color='white')),
                            hovertemplate=f'Speed: {speed:.0f} RPM<br>Flow: %{{x:.1f}}<br>Efficiency: %{{y:.2f}}%<extra></extra>',
                            legendgroup=f"speed_{speed}"
                        ))
                    
                    fig_eff.update_layout(
                        title='Polytropic Efficiency vs Flow Rate',
                        xaxis_title=x_label,
                        yaxis_title='Polytropic Efficiency (%)',
                        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                        hovermode='closest'
                    )
                    st.plotly_chart(fig_eff, use_container_width=True)
                
                with tab2:
                    fig_head = go.Figure()
                    
                    # Add manufacturer curves first (as background - dotted lines)
                    if show_mfr_curves and mfr_curves:
                        for i, curve in enumerate(mfr_curves):
                            color = curve_colors[i % len(curve_colors)]
                            fig_head.add_trace(go.Scatter(
                                x=curve['flow'],
                                y=curve['head'],
                                mode='lines',
                                name=f"Original {curve['speed']:.0f} RPM",
                                line=dict(width=2, color=color, dash='dot'),
                                opacity=0.6,
                                legendgroup=f"speed_{curve['speed']}"
                            ))
                    
                    # Add MW-corrected curves (dashed lines - from new gas composition)
                    if show_mfr_curves and corrected_curves:
                        for i, curve in enumerate(corrected_curves):
                            color = curve_colors[i % len(curve_colors)]
                            fig_head.add_trace(go.Scatter(
                                x=curve['flow'],
                                y=curve['head'],
                                mode='lines',
                                name=f"MW-Corrected {curve['speed']:.0f} RPM",
                                line=dict(width=2, color=color, dash='dash'),
                                opacity=0.8,
                                legendgroup=f"speed_{curve['speed']}_corr"
                            ))
                    
                    # Add generated/adjusted curves (solid lines - from measured data)
                    if gen_curves:
                        for i, curve in enumerate(gen_curves):
                            color = curve_colors[i % len(curve_colors)]
                            fig_head.add_trace(go.Scatter(
                                x=curve['flow'],
                                y=curve['head'],
                                mode='lines',
                                name=f"Fitted {curve['speed']:.0f} RPM",
                                line=dict(width=3, color=color),
                                legendgroup=f"speed_{curve['speed']}"
                            ))
                    
                    # Add calculated data points grouped by speed
                    for speed in sorted(unique_speeds):
                        speed_data = results_df[results_df['Speed (RPM)'] == speed]
                        color = get_speed_color(speed, default_color='green')
                        fig_head.add_trace(go.Scatter(
                            x=speed_data['Plot Flow'],
                            y=speed_data['Polytropic Head (kJ/kg)'],
                            mode='markers',
                            name=f'Measured @ {speed:.0f} RPM',
                            marker=dict(size=12, color=color, symbol='circle',
                                       line=dict(width=2, color='white')),
                            hovertemplate=f'Speed: {speed:.0f} RPM<br>Flow: %{{x:.1f}}<br>Head: %{{y:.2f}} kJ/kg<extra></extra>',
                            legendgroup=f"speed_{speed}"
                        ))
                    
                    fig_head.update_layout(
                        title='Polytropic Head vs Flow Rate',
                        xaxis_title=x_label,
                        yaxis_title='Polytropic Head (kJ/kg)',
                        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                        hovermode='closest'
                    )
                    st.plotly_chart(fig_head, use_container_width=True)
                
                with tab3:
                    fig_power = go.Figure()
                    
                    # Add calculated data points grouped by speed
                    for speed in sorted(unique_speeds):
                        speed_data = results_df[results_df['Speed (RPM)'] == speed]
                        color = get_speed_color(speed, default_color='red')
                        fig_power.add_trace(go.Scatter(
                            x=speed_data['Plot Flow'],
                            y=speed_data['Power (MW)'],
                            mode='markers',
                            name=f'Measured @ {speed:.0f} RPM',
                            marker=dict(size=12, color=color, symbol='circle',
                                       line=dict(width=2, color='white')),
                            hovertemplate=f'Speed: {speed:.0f} RPM<br>Flow: %{{x:.1f}}<br>Power: %{{y:.3f}} MW<extra></extra>'
                        ))
                    
                    fig_power.update_layout(
                        title='Power vs Flow Rate',
                        xaxis_title=x_label,
                        yaxis_title='Power (MW)',
                        hovermode='closest'
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
    
    The polytropic head is calculated using the selected equation of state for accurate 
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
    properties for natural gas mixtures with typical uncertainties of:
    - Density: ±0.1%
    - Speed of sound: ±0.1%
    - Heat capacity: ±1%
    """)

# AI Analysis Section - Only shown if AI is enabled
if is_ai_enabled():
    gemini_api_key = get_gemini_api_key()
    if 'calculated_results' in st.session_state and st.session_state['calculated_results'] is not None:
        st.divider()
        st.header("🤖 Performance Analysis")
        
        with st.expander("**AI-Powered Compressor Analysis**", expanded=True):
            st.markdown("""
            Click the button below to get AI-powered insights on your compressor performance results.
            
            **The analysis includes:**
            - Performance evaluation and efficiency assessment
            - Comparison with typical industry benchmarks
            - Potential issues and troubleshooting recommendations
            - Operational optimization suggestions
            - Curve shape analysis and deviations from manufacturer data
            
            **Reference Documents:**
            The analysis is based on established industry standards and technical literature:
            
            | Document | Description |
            |----------|-------------|
            | ASME PTC 10 (1997) | Performance Test Code on Compressors and Exhausters |
            | ISO 5389 (2005) | Turbocompressors - Performance test code |
            | API 617 (2022) | Axial and Centrifugal Compressors and Expander-compressors |
            | Schultz, J.M. (1962) | "The Polytropic Analysis of Centrifugal Compressors" - ASME |
            | Huntington (1985) | "Thermodynamic Analysis of Centrifugal Compressors" |
            | GERG-2008 (ISO 20765-2) | European Gas Research Group equation of state |
            | Hundseid & Bakken (2006) | "Wet Gas Performance Analysis" ASME GT2006-91035 |
            | Khader (2015) | Gas composition correction using Mach number similarity |
            | Lüdtke (2004) | "Process Centrifugal Compressors" - Springer |
            | Bloch & Godse (2006) | "Compressors and Modern Process Applications" |
            """)
            
            if st.button("🔍 Analyze Performance", type="primary", key="ai_analyze_btn"):
                try:
                    results_df = st.session_state['calculated_results']
                    
                    # Prepare summary statistics for AI
                    avg_efficiency = results_df['Polytropic Eff (%)'].mean() if 'Polytropic Eff (%)' in results_df.columns else 0
                    avg_head = results_df['Polytropic Head (kJ/kg)'].mean() if 'Polytropic Head (kJ/kg)' in results_df.columns else 0
                    avg_power = results_df['Power (MW)'].mean() if 'Power (MW)' in results_df.columns else 0
                    avg_pr = results_df['Pressure Ratio'].mean() if 'Pressure Ratio' in results_df.columns else 0
                    avg_flow = results_df['Mass Flow (kg/hr)'].mean() if 'Mass Flow (kg/hr)' in results_df.columns else 0
                    
                    min_eff = results_df['Polytropic Eff (%)'].min() if 'Polytropic Eff (%)' in results_df.columns else 0
                    max_eff = results_df['Polytropic Eff (%)'].max() if 'Polytropic Eff (%)' in results_df.columns else 0
                    
                    # Get manufacturer curves if available
                    mfr_curves = st.session_state.get('compressor_curves', [])
                    has_mfr_curves = isinstance(mfr_curves, list) and len(mfr_curves) > 0
                    
                    # Get fitted curves if available
                    fitted_curves = st.session_state.get('generated_curves', [])
                    has_fitted_curves = isinstance(fitted_curves, list) and len(fitted_curves) > 0
                    
                    # Get MW-corrected curves if available
                    corrected_curves = st.session_state.get('corrected_curves', [])
                    has_corrected_curves = isinstance(corrected_curves, list) and len(corrected_curves) > 0
                    
                    # Build curve data strings for AI
                    mfr_curve_text = ""
                    if has_mfr_curves:
                        mfr_curve_text = "\n## Manufacturer Curves (Design Performance):\n"
                        for i, curve in enumerate(mfr_curves):
                            if isinstance(curve, dict):
                                mfr_curve_text += f"\n### Speed {curve.get('speed', 'N/A')} RPM:\n"
                                mfr_curve_text += f"- Flow points: {curve.get('flow', [])}\n"
                                mfr_curve_text += f"- Head (kJ/kg): {curve.get('head', [])}\n"
                                mfr_curve_text += f"- Efficiency (%): {curve.get('efficiency', [])}\n"
                            else:
                                mfr_curve_text += f"\n### Curve {i+1}: {str(curve)[:200]}\n"
                    
                    fitted_curve_text = ""
                    if has_fitted_curves:
                        fitted_curve_text = "\n## Fitted Curves (from Measured Data):\n"
                        for i, curve in enumerate(fitted_curves):
                            if isinstance(curve, dict):
                                fitted_curve_text += f"\n### Speed {curve.get('speed', 'N/A')} RPM:\n"
                                fitted_curve_text += f"- Flow points: {curve.get('flow', [])}\n"
                                fitted_curve_text += f"- Head (kJ/kg): {curve.get('head', [])}\n"
                                fitted_curve_text += f"- Efficiency (%): {curve.get('efficiency', [])}\n"
                            else:
                                fitted_curve_text += f"\n### Curve {i+1}: {str(curve)[:200]}\n"
                    
                    corrected_curve_text = ""
                    if has_corrected_curves:
                        corrected_curve_text = "\n## MW-Corrected Curves (Adjusted for Gas Composition):\n"
                        for i, curve in enumerate(corrected_curves):
                            if isinstance(curve, dict):
                                corrected_curve_text += f"\n### Speed {curve.get('speed', 'N/A')} RPM:\n"
                                corrected_curve_text += f"- Flow points: {curve.get('flow', [])}\n"
                                corrected_curve_text += f"- Head (kJ/kg): {curve.get('head', [])}\n"
                                corrected_curve_text += f"- Efficiency (%): {curve.get('efficiency', [])}\n"
                            else:
                                corrected_curve_text += f"\n### Curve {i+1}: {str(curve)[:200]}\n"
                    
                    # Build the AI prompt with curve analysis
                    prompt = f"""
                    You are an expert centrifugal compressor performance engineer. Analyze the following compressor test data, performance curves, and provide detailed insights including curve shape analysis and deviations.
                    
                    Base your analysis on established industry standards and technical references:
                    
                    **Test Codes & Standards:**
                    - ASME PTC 10 (1997): Performance Test Code on Compressors and Exhausters - test procedures, uncertainty analysis, acceptance criteria
                    - ISO 5389 (2005): Turbocompressors - Performance test code - international test standards
                    - API 617 (2022): Axial and Centrifugal Compressors - design, materials, shop testing requirements
                    
                    **Thermodynamic Analysis:**
                    - Schultz, J.M. (1962): "The Polytropic Analysis of Centrifugal Compressors" (ASME Journal of Engineering for Power) - polytropic efficiency method
                    - Huntington, R.A. (1985): "Thermodynamic Analysis of Centrifugal Compressors" - real gas effects
                    - Mallen & Saville (1977): "Polytropic Processes in the Performance Prediction of Centrifugal Compressors"
                    
                    **Equations of State:**
                    - GERG-2008 (ISO 20765-2): European Gas Research Group - high accuracy natural gas properties
                    - Peng-Robinson (1976): Cubic equation of state for hydrocarbon systems
                    - Soave-Redlich-Kwong (1972): Modified RK equation of state
                    - AGA8-DC92: Natural gas compressibility calculations
                    
                    **Advanced Methods:**
                    - Hundseid & Bakken (2006): "Wet Gas Performance Analysis" ASME GT2006-91035 - multi-step polytropic integration
                    - Khader (2015): Gas composition correction using Mach number similarity
                    - Evans & Huble (1981): Surge margin and operating envelope analysis
                    
                    **Reference Books:**
                    - Lüdtke, K.H. (2004): "Process Centrifugal Compressors" - Springer
                    - Bloch & Godse (2006): "Compressors and Modern Process Applications"
                    - Brown (2005): "Compressors: Selection and Sizing" - Gulf Publishing
                    
                    ## Test Results Summary:
                    - Number of operating points: {len(results_df)}
                    - Average polytropic efficiency: {avg_efficiency:.1f}%
                    - Efficiency range: {min_eff:.1f}% to {max_eff:.1f}%
                    - Average polytropic head: {avg_head:.1f} kJ/kg
                    - Average power consumption: {avg_power:.2f} MW
                    - Average pressure ratio: {avg_pr:.2f}
                    - Average mass flow: {avg_flow:.0f} kg/hr
                    
                    ## Curve Data Available:
                    - Manufacturer curves: {'Yes' if has_mfr_curves else 'No'}
                    - Fitted curves from measurements: {'Yes' if has_fitted_curves else 'No'}
                    - MW-corrected curves: {'Yes' if has_corrected_curves else 'No'}
                    
                    ## Full Results Data:
                    {results_df.to_string()}
                    {mfr_curve_text}
                    {fitted_curve_text}
                    {corrected_curve_text}
                    
                    Please provide a comprehensive analysis:
                    
                    1. **Performance Assessment**: Evaluate the overall compressor performance. Is the efficiency typical for a centrifugal compressor? (70-85% is typical range)
                    
                    2. **Operating Point Analysis**: Analyze the operating points. Are there any concerning patterns (e.g., operation near surge, low efficiency points, choke proximity)?
                    
                    3. **Curve Shape Analysis** (if curves available):
                       - Analyze the shape of efficiency and head curves
                       - Is the efficiency curve parabolic as expected? Is the peak at the right flow?
                       - Is the head curve showing normal falling characteristic with flow?
                       - Any abnormal inflections, flat spots, or irregular shapes?
                    
                    4. **Deviation Analysis** (if manufacturer and fitted/measured curves available):
                       - Compare measured performance to manufacturer curves
                       - Quantify deviations in head and efficiency at similar operating points
                       - Is the deviation parallel (uniform across flow range) or divergent (varies with flow)?
                       - Parallel shift suggests: overall degradation (fouling, wear)
                       - Divergent pattern suggests: specific issues (tip clearance at high speed, surge at low flow)
                    
                    5. **MW Correction Assessment** (if MW-corrected curves available):
                       - Are the MW corrections reasonable for the gas composition change?
                       - Does the corrected curve better match the measured data?
                    
                    6. **Potential Issues**: Based on the data and curve analysis:
                       - Low efficiency: fouling, wear, seal leakage, measurement errors
                       - Head deviation: gas composition changes, impeller damage, internal leakage
                       - Abnormal curve shape: partial fouling, erosion, mechanical damage
                    
                    7. **Recommendations**: Provide 3-4 actionable recommendations for:
                       - Performance optimization
                       - Maintenance considerations
                       - Operating improvements
                       - Further testing or investigation needed
                    
                    8. **Summary**: One-paragraph executive summary of the compressor condition and priority actions.
                    
                    Keep the response practical for an operations/maintenance engineer. Use specific numbers and percentages where possible.
                    """
                    
                    with st.spinner(f"🔄 Analyzing with {st.session_state.get('ai_model', 'gemini-2.0-flash')}..."):
                        genai.configure(api_key=gemini_api_key)
                        selected_model = st.session_state.get('ai_model', 'gemini-2.0-flash')
                        
                        system_instruction = """You are an expert centrifugal compressor performance engineer with 20+ years of experience in rotating equipment analysis, performance testing, and troubleshooting.

Your analysis is grounded in established industry standards and technical literature:

TEST CODES & STANDARDS:
- ASME PTC 10 (1997): Performance Test Code on Compressors and Exhausters
- ISO 5389 (2005): Turbocompressors - Performance test code
- API 617 (2022): Axial and Centrifugal Compressors and Expander-compressors

THERMODYNAMIC ANALYSIS:
- Schultz, J.M. (1962): "The Polytropic Analysis of Centrifugal Compressors" - ASME
- Huntington, R.A. (1985): "Thermodynamic Analysis of Centrifugal Compressors"
- Mallen & Saville (1977): "Polytropic Processes in the Performance Prediction"

EQUATIONS OF STATE:
- GERG-2008 (ISO 20765-2): European Gas Research Group
- Peng-Robinson (1976) and Soave-Redlich-Kwong (1972) equations of state
- AGA8-DC92: Natural gas compressibility

ADVANCED METHODS:
- Hundseid & Bakken (2006): "Wet Gas Performance Analysis" ASME GT2006-91035
- Khader (2015): Gas composition correction using Mach number similarity
- Evans & Huble (1981): Surge margin analysis

REFERENCE BOOKS:
- Lüdtke (2004): "Process Centrifugal Compressors" - Springer
- Bloch & Godse (2006): "Compressors and Modern Process Applications"
- Brown (2005): "Compressors: Selection and Sizing"

Reference these standards when making recommendations. Use ASME PTC 10 and ISO 5389 acceptance criteria when evaluating performance deviations. Cite specific standards where applicable."""
                        
                        try:
                            model = genai.GenerativeModel(
                                selected_model,
                                system_instruction=system_instruction
                            )
                            response = model.generate_content(
                                prompt,
                                generation_config=genai.types.GenerationConfig(
                                    max_output_tokens=1500,
                                    temperature=0.3
                                )
                            )
                            ai_analysis = response.text
                        except Exception as model_error:
                            # Try fallback to gemini-2.0-flash if primary model fails
                            st.warning(f"⚠️ {selected_model} unavailable, trying gemini-2.0-flash...")
                            model = genai.GenerativeModel(
                                'gemini-2.0-flash',
                                system_instruction=system_instruction
                            )
                            response = model.generate_content(
                                prompt,
                                generation_config=genai.types.GenerationConfig(
                                    max_output_tokens=1500,
                                    temperature=0.3
                                )
                            )
                            ai_analysis = response.text
                        
                        st.markdown("---")
                        st.markdown("### 📊 AI Analysis Results")
                        st.markdown(ai_analysis)
                        
                        # Store in session state for later reference
                        st.session_state['ai_analysis'] = ai_analysis
                        
                except Exception as e:
                    error_msg = str(e)
                    st.error(f"AI analysis failed: {error_msg}")
                    if "connection" in error_msg.lower():
                        st.warning("🔌 **Connection Issue**: Check your internet connection or firewall settings.")
                    elif "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                        st.warning("🔑 **API Key Issue**: Your API key may be invalid or expired.")
                    elif "rate_limit" in error_msg.lower() or "quota" in error_msg.lower():
                        st.warning("⏱️ **Rate Limit**: Too many requests. Wait a moment and try again.")
                    else:
                        st.info("💡 **Tip**: Try selecting a different model in the sidebar.")
            
            # Show previous analysis if available
            if 'ai_analysis' in st.session_state and st.session_state['ai_analysis']:
                with st.expander("📜 Previous AI Analysis", expanded=False):
                    st.markdown(st.session_state['ai_analysis'])
            
            # Follow-up Q&A section
            st.divider()
            st.subheader("💬 Ask Follow-up Questions")
            st.markdown("Ask questions about the analysis, request clarifications, or explore specific aspects of the compressor performance.")
            
            # Initialize chat history in session state
            if 'ai_chat_history' not in st.session_state:
                st.session_state['ai_chat_history'] = []
            
            # Display chat history
            for message in st.session_state['ai_chat_history']:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input for follow-up questions
            if follow_up_question := st.chat_input("Ask a question about the compressor analysis..."):
                # Add user message to chat history
                st.session_state['ai_chat_history'].append({"role": "user", "content": follow_up_question})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(follow_up_question)
                
                # Prepare context for follow-up
                try:
                    results_df = st.session_state['calculated_results']
                    previous_analysis = st.session_state.get('ai_analysis', '')
                    
                    # Build conversation context for Gemini
                    context_prompt = f"""You are an expert centrifugal compressor performance engineer with 20+ years of experience. 
                    You have access to the compressor test data and your previous analysis. 
                    Answer follow-up questions concisely and practically.
                    Reference specific data points and numbers when relevant.
                    If asked about something not in the data, explain what additional information would be needed.
                    
                    Context - Previous Analysis:
                    {previous_analysis}
                    
                    Context - Test Data Summary:
                    {results_df.to_string() if results_df is not None else 'No data available'}
                    
                    Conversation History:
                    """
                    
                    # Add chat history to context
                    for msg in st.session_state['ai_chat_history'][:-1]:  # Exclude the current question
                        role = "User" if msg["role"] == "user" else "Assistant"
                        context_prompt += f"\n{role}: {msg['content']}"
                    
                    context_prompt += f"\n\nUser's current question: {follow_up_question}\n\nProvide a helpful response:"
                    
                    # Get AI response
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            genai.configure(api_key=gemini_api_key)
                            selected_model = st.session_state.get('ai_model', 'gemini-2.0-flash')
                            try:
                                model = genai.GenerativeModel(selected_model)
                                response = model.generate_content(
                                    context_prompt,
                                    generation_config=genai.types.GenerationConfig(
                                        max_output_tokens=1000,
                                        temperature=0.3
                                    )
                                )
                                assistant_response = response.text
                            except Exception:
                                # Fallback to gemini-2.0-flash
                                model = genai.GenerativeModel('gemini-2.0-flash')
                                response = model.generate_content(
                                    context_prompt,
                                    generation_config=genai.types.GenerationConfig(
                                        max_output_tokens=1000,
                                        temperature=0.3
                                    )
                                )
                                assistant_response = response.text
                            
                            st.markdown(assistant_response)
                            
                            # Add assistant response to chat history
                            st.session_state['ai_chat_history'].append({"role": "assistant", "content": assistant_response})
                
                except Exception as e:
                    st.error(f"Failed to get response: {str(e)}")
            
            # Clear chat button
            if st.session_state['ai_chat_history']:
                if st.button("🗑️ Clear Chat History", key="clear_chat_btn"):
                    st.session_state['ai_chat_history'] = []
                    st.rerun()
    else:
        st.divider()
        st.info("🤖 **AI Analysis Available**: Run calculations first to enable AI-powered performance analysis.")