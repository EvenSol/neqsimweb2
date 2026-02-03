import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from neqsim.thermo import fluid, TPflash
from neqsim import jneqsim

def create_cpa_fluid(use_electrolyte=False):
    """
    Create a CPA-SRK-EOS fluid using the Statoil implementation.
    
    Args:
        use_electrolyte: If True, use Electrolyte-CPA-EOS for salinity effects.
                        If False, use standard CPA-SRK-EOS.
    
    This explicitly uses SystemSrkCPAstatoil (or SystemElectrolyteCPAstatoil) which is optimized for:
    - Water-hydrocarbon VLE
    - CO2 and H2S solubility in water
    - Produced water degassing calculations
    - Ion effects on gas solubility (electrolyte version)
    
    Returns:
        CPA equation of state fluid object (standard or electrolyte)
    """
    if use_electrolyte:
        return jneqsim.thermo.system.SystemElectrolyteCPAstatoil(288.15, 1.0)
    else:
        return jneqsim.thermo.system.SystemSrkCPAstatoil(288.15, 1.0)

def create_soreide_whitson_fluid():
    """
    Create a Søreide-Whitson Peng-Robinson fluid for water-gas systems with salinity.
    
    The Søreide-Whitson model uses modified alpha functions and mixing rules
    specifically designed for:
    - Gas solubility in brine (saline water)
    - CO2, H2S, and hydrocarbon solubility with salinity effects
    - Accurate phase equilibrium for produced water systems
    
    This model directly accounts for salinity through modified binary interaction
    parameters (kij) that are temperature and salinity dependent.
    
    Returns:
        Søreide-Whitson Peng-Robinson EoS fluid object
    """
    return jneqsim.thermo.system.SystemSoreideWhitson(288.15, 1.0)

st.set_page_config(
    page_title="Emission Calculator",
    page_icon='images/neqsimlogocircleflat.png',
    layout="wide"
)

st.title("🌱 Methane & CO₂ Emission Calculator")
st.markdown("""
Quantify greenhouse gas emissions (CO₂, CH₄, nmVOC) from offshore process streams using 
rigorous thermodynamic modeling with the CPA equation of state. Also supports the Søreide-Whitson 
method for accurate gas solubility calculations in saline water systems.

📚 [NeqSim Emissions Guide](https://equinor.github.io/neqsim/emissions/) | 
🔬 Validated ±3.6% accuracy against field data
""")

with st.expander("📖 Documentation & Help", expanded=False):
    doc_tab1, doc_tab2, doc_tab3 = st.tabs(["Emission Sources", "Methods", "References"])
    
    with doc_tab1:
        st.markdown("""
        ### Supported Emission Sources
        | Source | Description |
        |--------|-------------|
        | **Produced Water Degassing** | Multi-stage degassing (Degasser, CFU, Caisson) |
        | **TEG Regeneration** | Flash drum off-gas from glycol dehydration |
        | **Tank Breathing/Loading** | Storage tank vapor losses |
        | **Cold Vent Streams** | Pressure relief and maintenance vents |
        
        ### Key Terms
        | Term | Definition |
        |------|------------|
        | **nmVOC** | Non-methane Volatile Organic Compounds (C₂+ hydrocarbons) |
        | **GWP-100** | Global Warming Potential over 100-year horizon |
        | **CO₂eq** | Total emissions as CO₂ equivalent |
        | **CPA** | Cubic-Plus-Association equation of state |
        """)
    
    with doc_tab2:
        st.markdown("""
        ### Calculation Methods
        
        **Lab Sample Analysis**: Uses measured gas composition from flashing a water sample 
        at standard conditions (1 atm, 15°C).
        
        **Separator/Absorber Equilibrium**: Calculates dissolved gas from thermodynamic 
        equilibrium using the selected equation of state.
        
        ### Thermodynamic Models
        
        **CPA-SRK-EOS (Statoil)** with Setschenow correction:
        - Water-hydrocarbon VLE with analytical salting-out correction
        - Good for general hydrocarbon-water systems
        - CO₂ and H₂S solubility in water
        
        **Søreide-Whitson PR-EoS**:
        - Peng-Robinson EoS with salinity-dependent kij parameters
        - Specifically designed for gas-brine equilibrium
        - Recommended for high salinity (>50,000 ppm) or CO₂/H₂S-rich systems
        """)
    
    with doc_tab3:
        st.markdown("""
        ### Regulatory Compliance
        - Norwegian: Aktivitetsforskriften §70
        - EU: ETS Directive, Methane Regulation 2024/1787
        - International: ISO 14064-1, OGMP 2.0
        
        ### Resources
        - [Offshore Emission Reporting Guide](https://equinor.github.io/neqsim/emissions/OFFSHORE_EMISSION_REPORTING.html)
        - [Produced Water Tutorial](https://equinor.github.io/neqsim/examples/ProducedWaterEmissions_Tutorial.html)
        - [NeqSim GitHub](https://github.com/equinor/neqsim)
        """)

st.divider()

# ===================== SESSION STATE INITIALIZATION =====================
# Default gas composition: laboratory flash gas analysis (1 atm, 15°C)
# Values represent mole fractions of released gas, not the water-gas mixture
# Reference: Typical North Sea produced water (high CO₂ content characteristic of mature fields)
if 'emission_gas_df' not in st.session_state:
    st.session_state.emission_gas_df = pd.DataFrame({
        'ComponentName': ['CO2', 'methane', 'ethane', 'propane', 'i-butane', 'n-butane'],
        'MolarComposition[-]': [0.51, 0.44, 0.037, 0.010, 0.001, 0.002]
    })

if 'teg_fluid_df' not in st.session_state:
    st.session_state.teg_fluid_df = pd.DataFrame({
        'ComponentName': ['TEG', 'water', 'methane', 'ethane', 'propane', 'i-butane', 'n-butane', 
                          'i-pentane', 'n-pentane', 'n-hexane', 'CO2', 'benzene'],
        'MolarComposition[-]': [0.85, 0.05, 0.03, 0.015, 0.010, 0.003, 0.005, 
                                0.002, 0.002, 0.002, 0.01, 0.021]
    })

# TEG absorber inlet gas composition (for equilibrium method)
if 'teg_absorber_gas_df' not in st.session_state:
    st.session_state.teg_absorber_gas_df = pd.DataFrame({
        'ComponentName': ['methane', 'ethane', 'propane', 'i-butane', 'n-butane', 
                          'i-pentane', 'n-pentane', 'n-hexane', 'CO2', 'nitrogen', 'benzene', 'toluene'],
        'MolarComposition[-]': [0.82, 0.06, 0.03, 0.008, 0.012, 0.005, 0.005, 
                                0.004, 0.02, 0.01, 0.013, 0.013]
    })

# Typical first stage separator gas compositions
SEPARATOR_GAS_PRESETS = {
    'Custom': None,
    'North Sea - Mature Field': {
        'ComponentName': ['nitrogen', 'CO2', 'H2S', 'methane', 'ethane', 'propane', 'i-butane', 'n-butane', 'i-pentane', 'n-pentane', 'n-hexane'],
        'MolarComposition[-]': [0.008, 0.035, 0.0001, 0.82, 0.065, 0.035, 0.008, 0.012, 0.005, 0.006, 0.006]
    },
    'North Sea - High CO₂': {
        'ComponentName': ['nitrogen', 'CO2', 'H2S', 'methane', 'ethane', 'propane', 'i-butane', 'n-butane', 'i-pentane', 'n-pentane'],
        'MolarComposition[-]': [0.005, 0.12, 0.0002, 0.72, 0.07, 0.045, 0.01, 0.015, 0.007, 0.008]
    },
    'Norwegian Continental Shelf - Lean Gas': {
        'ComponentName': ['nitrogen', 'CO2', 'methane', 'ethane', 'propane', 'i-butane', 'n-butane'],
        'MolarComposition[-]': [0.01, 0.02, 0.91, 0.04, 0.015, 0.003, 0.002]
    },
    'Associated Gas - Rich': {
        'ComponentName': ['nitrogen', 'CO2', 'H2S', 'methane', 'ethane', 'propane', 'i-butane', 'n-butane', 'i-pentane', 'n-pentane', 'n-hexane'],
        'MolarComposition[-]': [0.005, 0.025, 0.0005, 0.70, 0.10, 0.08, 0.02, 0.03, 0.015, 0.015, 0.01]
    },
    'Condensate Field': {
        'ComponentName': ['nitrogen', 'CO2', 'methane', 'ethane', 'propane', 'i-butane', 'n-butane', 'i-pentane', 'n-pentane', 'n-hexane'],
        'MolarComposition[-]': [0.003, 0.015, 0.75, 0.09, 0.06, 0.02, 0.025, 0.015, 0.012, 0.01]
    },
    'Sour Gas Field': {
        'ComponentName': ['nitrogen', 'CO2', 'H2S', 'methane', 'ethane', 'propane', 'i-butane', 'n-butane'],
        'MolarComposition[-]': [0.005, 0.08, 0.03, 0.75, 0.06, 0.04, 0.015, 0.02]
    }
}

# Initialize separator gas composition
if 'separator_gas_df' not in st.session_state:
    preset = SEPARATOR_GAS_PRESETS['North Sea - Mature Field']
    st.session_state.separator_gas_df = pd.DataFrame(preset)

# ===================== SIDEBAR CONFIGURATION =====================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Emission Source Selection
    st.subheader("📍 Emission Source")
    emission_source = st.selectbox(
        "Select Source Type",
        ["Produced Water Degassing", "TEG Regeneration", "Tank Breathing/Loading", "Cold Vent"],
        help="Different emission sources have different process configurations"
    )
    
    st.divider()
    
    # Calculation Method Selection (for Produced Water Degassing and TEG Regeneration)
    if emission_source == "Produced Water Degassing":
        st.subheader("🔬 Calculation Method")
        calc_method = st.radio(
            "Select Method",
            ["Lab Sample Analysis", "Separator Equilibrium"],
            help="""
            **Lab Sample Analysis**: Uses measured gas composition from flashing a water sample at standard conditions (1 atm, 15°C). Best when lab data is available.
            
            **Separator Equilibrium**: Calculates water-gas equilibrium from the first stage separator gas composition. Uses thermodynamic model to determine dissolved gas content.
            """
        )
        
        st.divider()
    elif emission_source == "TEG Regeneration":
        st.subheader("🔬 Calculation Method")
        calc_method = st.radio(
            "Select Method",
            ["Rich TEG Composition", "Absorber Equilibrium"],
            help="""
            **Rich TEG Composition**: Direct input of rich TEG composition (measured or estimated). Best when lab data is available.
            
            **Absorber Equilibrium**: Calculates TEG-gas equilibrium from the absorber inlet gas composition. Uses thermodynamic model to determine absorbed components.
            """
        )
        
        st.divider()
    else:
        calc_method = "Lab Sample Analysis"  # Default for other sources
    
    # Process Conditions
    st.subheader("🔧 Process Conditions")
    
    if emission_source == "Produced Water Degassing":
        inlet_temp = st.number_input("Separator Temperature (°C)", value=80.0, min_value=-50.0, max_value=300.0,
                                     help="Temperature in the first stage separator")
        inlet_pressure = st.number_input("Separator Pressure (bara)", value=30.0, min_value=1.0, max_value=500.0,
                                         help="Pressure in the first stage separator")
        water_flow_m3hr = st.number_input("Water Flow Rate (m³/hr)", value=100.0, min_value=1.0, max_value=10000.0, help="Produced water volumetric flow")
        total_flow = water_flow_m3hr * 1000  # Convert to kg/hr (approx density 1000 kg/m³)
        
        # Salinity
        st.subheader("🧂 Salinity Effects")
        
        # Thermodynamic model selection for salinity handling
        salinity_model = st.selectbox(
            "Thermodynamic Model",
            ["CPA-SRK (Setschenow correction)", "Søreide-Whitson PR-EoS"],
            index=0,
            help="""Select the thermodynamic model for handling salinity effects:
            
**CPA-SRK (Setschenow correction)**: Uses CPA equation of state with an analytical Setschenow salting-out correction factor. Good for general use with hydrocarbon-water systems.
            
**Søreide-Whitson PR-EoS**: Uses Peng-Robinson EoS with modified alpha functions and salinity-dependent binary interaction parameters (kij). Specifically designed for gas-brine systems. Recommended for high salinity (>50,000 ppm) or when CO₂/H₂S accuracy is critical."""
        )
        
        salinity_ppm = st.number_input(
            "Salinity (ppm TDS)", 
            value=0, 
            min_value=0, 
            max_value=300000,
            help="Total dissolved solids. Set to 0 for freshwater. Seawater ~35,000 ppm, produced water can be 100,000+ ppm. Salinity reduces gas solubility (salting-out effect)."
        )
        
        # Calculate salting-out factor based on Setschenow equation (used for CPA model)
        # Higher salinity = lower gas solubility = potentially lower emissions (less dissolved gas)
        if salinity_ppm > 0:
            cs = 0.12  # Setschenow coefficient for CH4 in NaCl (typical value)
            molality = salinity_ppm / 58440 / (1 - salinity_ppm/1e6)  # Convert ppm to molality
            salting_out_factor = 10 ** (-cs * molality)
            if salinity_model == "CPA-SRK (Setschenow correction)":
                st.info(f"Setschenow salting-out factor: {salting_out_factor:.3f} — Gas solubility reduced to {salting_out_factor*100:.1f}% of freshwater value")
            else:
                st.info(f"Søreide-Whitson model will use salinity-dependent kij parameters (molality: {molality:.2f} mol/kg)")
        else:
            salting_out_factor = 1.0
            molality = 0.0
        
        # Degassing Stages
        st.subheader("🔀 Degassing Stages")
        num_stages = st.selectbox("Number of Stages", [1, 2, 3], index=2)
        
        stage_pressures = []
        stage_names = ['Degasser', 'CFU', 'Caisson']
        default_pressures = [4.0, 1.5, 1.01325]
        
        for i in range(num_stages):
            p = st.number_input(
                f"Stage {i+1} ({stage_names[i]}) (bara)",
                value=default_pressures[i],
                min_value=0.1,
                max_value=inlet_pressure,
                key=f"stage_{i}_pressure"
            )
            stage_pressures.append(p)
        
        # All stages at same temperature for produced water degassing
        stage_temps = [inlet_temp] * num_stages
    
    elif emission_source == "TEG Regeneration":
        inlet_temp = st.number_input("Rich TEG Temperature (°C)", value=35.0, min_value=0.0, max_value=100.0)
        inlet_pressure = st.number_input("Contactor Pressure (bara)", value=52.0, min_value=1.0, max_value=200.0)
        total_flow = st.number_input("TEG Circulation Rate (kg/hr)", value=6000.0, min_value=100.0, max_value=100000.0)
        
        st.subheader("🔄 TEG Regeneration Process")
        col_flash, col_regen = st.columns(2)
        
        with col_flash:
            flash_pressure = st.number_input("Flash Drum Pressure (bara)", value=3.0, min_value=0.5, max_value=20.0,
                                            help="Pressure of HP flash drum upstream of regenerator")
            flash_gas_destination = st.selectbox(
                "Flash Gas Destination",
                ["Emission (Flare/Vent)", "Recovered (Fuel Gas)"],
                help="If recovered, flash gas emissions are not counted"
            )
        
        with col_regen:
            regen_pressure = st.number_input("Regenerator Pressure (bara)", value=1.1, min_value=0.8, max_value=3.0,
                                            help="Pressure at regenerator still column")
            regen_temp = st.number_input("Regenerator Temperature (°C)", value=204.0, min_value=150.0, max_value=220.0,
                                        help="Reboiler temperature")
        
        water_flow_m3hr = total_flow / 1000  # For consistency in export, convert kg/hr to approx m³/hr
        salinity_ppm = 0
        salting_out_factor = 1.0
        molality = 0.0
        
        # Define stages based on flash gas destination
        if flash_gas_destination == "Emission (Flare/Vent)":
            num_stages = 2
            stage_pressures = [flash_pressure, regen_pressure]
            stage_temps = [inlet_temp, regen_temp]  # Flash at rich TEG temp, regenerator at regen temp
            stage_names = ['Flash Drum', 'Regenerator']
        else:
            # Flash gas recovered - only regenerator emissions count
            num_stages = 1
            stage_pressures = [regen_pressure]
            stage_temps = [regen_temp]
            stage_names = ['Regenerator']
    
    elif emission_source == "Tank Breathing/Loading":
        inlet_temp = st.number_input("Tank Temperature (°C)", value=25.0, min_value=-20.0, max_value=60.0)
        inlet_pressure = st.number_input("Tank Pressure (bara)", value=1.05, min_value=1.0, max_value=2.0)
        total_flow = st.number_input("Throughput (kg/hr)", value=50000.0, min_value=100.0, max_value=1000000.0)
        breathing_rate = st.number_input("Breathing Rate (%)", value=0.1, min_value=0.01, max_value=1.0,
                                        help="Percentage of tank volume vented per hour. TODO: Implement breathing loss calculation.")
        # TODO: Implement tank breathing/working loss calculations using breathing_rate
        # Formula: breathing_loss = throughput * breathing_rate/100 * vapor_fraction
        water_flow_m3hr = total_flow / 1000  # For consistency in export, convert kg/hr to approx m³/hr
        salinity_ppm = 0
        salting_out_factor = 1.0
        molality = 0.0
        num_stages = 1
        stage_pressures = [1.01325]
        stage_temps = [inlet_temp]
        stage_names = ['Atmosphere']
    
    else:  # Cold Vent
        inlet_temp = st.number_input("Vent Temperature (°C)", value=20.0, min_value=-50.0, max_value=100.0)
        inlet_pressure = st.number_input("Upstream Pressure (bara)", value=10.0, min_value=1.0, max_value=100.0)
        total_flow = st.number_input("Vent Rate (kg/hr)", value=500.0, min_value=1.0, max_value=10000.0)
        water_flow_m3hr = total_flow / 1000  # For consistency in export, convert kg/hr to approx m³/hr
        salinity_ppm = 0
        salting_out_factor = 1.0
        molality = 0.0
        num_stages = 1
        stage_pressures = [1.01325]
        stage_temps = [inlet_temp]
        stage_names = ['Atmosphere']
    
    st.divider()
    
    # GWP Factors
    st.subheader("🌍 GWP Factors")
    gwp_standard = st.selectbox(
        "GWP Standard", 
        ["IPCC AR5 (2014)", "IPCC AR6 (2021)"],
        help="Global Warming Potential standard"
    )
    
    gwp_ch4 = 28.0 if gwp_standard == "IPCC AR5 (2014)" else 29.8
    
    gwp_nmvoc = st.number_input(
        "nmVOC GWP-100", 
        value=2.2, 
        min_value=0.1, 
        max_value=10.0,
        help="GWP for non-methane VOCs (C₂+ hydrocarbons)"
    )
    
    st.caption(f"CH₄ GWP: {gwp_ch4}")
    
    st.divider()
    
    # Advanced Options
    st.subheader("⚡ Advanced Options")
    use_process_sim = st.checkbox(
        "Use Process Simulation",
        value=True,
        help="Use NeqSim process equipment (Separator, ThrottlingValve) for rigorous multi-stage degassing simulation with proper material balance tracking"
    )

# ===================== MAIN CONTENT - TABS =====================
main_tab1, main_tab2, main_tab3 = st.tabs(["📊 Emission Calculator", "🔬 What-If Analysis", "📈 Uncertainty Analysis"])

# ===================== TAB 1: EMISSION CALCULATOR =====================
with main_tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if emission_source == "Produced Water Degassing":
            if calc_method == "Lab Sample Analysis":
                st.subheader("📋 Laboratory Gas Analysis")
                st.caption("Enter gas composition from flashing water sample to standard conditions (1 atm, 15°C)")
                
                available_components = ['CO2', 'methane', 'ethane', 'propane', 'H2S',
                                       'i-butane', 'n-butane', 'i-pentane', 'n-pentane', 'n-hexane',
                                       'nitrogen']
                current_df = st.session_state.emission_gas_df
                
                edited_df = st.data_editor(
                    current_df,
                    column_config={
                        "ComponentName": st.column_config.SelectboxColumn(
                            "Component",
                            options=available_components,
                            required=True
                        ),
                        "MolarComposition[-]": st.column_config.NumberColumn(
                            "Mole Fraction",
                            min_value=0.0,
                            max_value=1.0,
                            format="%.6f"
                        )
                    },
                    num_rows='dynamic',
                    use_container_width=True,
                    key="lab_gas_editor"
                )
                st.session_state.emission_gas_df = edited_df
                
            else:  # Separator Equilibrium method
                st.subheader("🛢️ First Stage Separator Gas")
                st.caption("Enter gas composition from the first stage separator (or select a typical composition)")
                
                # Preset selection - use session state to track selection
                if 'selected_preset' not in st.session_state:
                    st.session_state.selected_preset = 'North Sea - Mature Field'
                
                preset_name = st.selectbox(
                    "Select Typical Composition",
                    list(SEPARATOR_GAS_PRESETS.keys()),
                    index=list(SEPARATOR_GAS_PRESETS.keys()).index(st.session_state.selected_preset),
                    help="Select a predefined separator gas composition or choose 'Custom' to enter your own",
                    key="preset_selector"
                )
                
                # Only update if preset changed (not on every rerun)
                if preset_name != st.session_state.selected_preset:
                    st.session_state.selected_preset = preset_name
                    if preset_name != 'Custom' and SEPARATOR_GAS_PRESETS[preset_name] is not None:
                        st.session_state.separator_gas_df = pd.DataFrame(SEPARATOR_GAS_PRESETS[preset_name])
                
                available_components = ['nitrogen', 'CO2', 'H2S', 'methane', 'ethane', 'propane',
                                       'i-butane', 'n-butane', 'i-pentane', 'n-pentane', 'n-hexane',
                                       'n-heptane', 'n-octane']
                
                edited_df = st.data_editor(
                    st.session_state.separator_gas_df,
                    column_config={
                        "ComponentName": st.column_config.SelectboxColumn(
                            "Component",
                            options=available_components,
                            required=True
                        ),
                        "MolarComposition[-]": st.column_config.NumberColumn(
                            "Mole Fraction",
                            min_value=0.0,
                            max_value=1.0,
                            format="%.6f"
                        )
                    },
                    num_rows='dynamic',
                    use_container_width=True,
                    key="sep_gas_editor"
                )
                st.session_state.separator_gas_df = edited_df
                
                st.info("""
                **Method**: The separator gas will be put in thermodynamic equilibrium with water 
                at separator conditions. The dissolved gas content in water is calculated using 
                the CPA equation of state. This water is then flashed through the degassing stages.
                """)
                
        elif emission_source == "TEG Regeneration":
            # Extended component list including aromatics for TEG processes
            teg_available_components = ['TEG', 'water', 'methane', 'ethane', 'propane', 'i-butane', 'n-butane',
                                        'i-pentane', 'n-pentane', 'n-hexane', 'n-heptane', 'n-octane',
                                        'CO2', 'H2S', 'nitrogen', 'benzene', 'toluene', 
                                        'ethylbenzene', 'm-xylene', 'o-xylene', 'p-xylene']
            
            if calc_method == "Rich TEG Composition":
                st.subheader(f"📋 Rich TEG Composition")
                st.caption("Enter the composition of rich TEG from the absorber bottom.")
                current_df = st.session_state.teg_fluid_df
                
                edited_df = st.data_editor(
                    current_df,
                    column_config={
                        "ComponentName": st.column_config.SelectboxColumn(
                            "Component",
                            options=teg_available_components,
                            required=True
                        ),
                        "MolarComposition[-]": st.column_config.NumberColumn(
                            "Mole Fraction",
                            min_value=0.0,
                            max_value=1.0,
                            format="%.6f"
                        )
                    },
                    num_rows='dynamic',
                    use_container_width=True,
                    key="teg_editor"
                )
                st.session_state.teg_fluid_df = edited_df
            else:  # Absorber Equilibrium method
                st.subheader(f"📋 Absorber Inlet Gas Composition")
                st.caption("Enter the wet gas composition entering the TEG absorber. Rich TEG composition will be calculated.")
                
                # Gas components for absorber inlet
                absorber_gas_components = ['methane', 'ethane', 'propane', 'i-butane', 'n-butane',
                                          'i-pentane', 'n-pentane', 'n-hexane', 'n-heptane', 'n-octane',
                                          'CO2', 'H2S', 'nitrogen', 'benzene', 'toluene',
                                          'ethylbenzene', 'm-xylene', 'o-xylene', 'p-xylene']
                current_df = st.session_state.teg_absorber_gas_df
                
                edited_df = st.data_editor(
                    current_df,
                    column_config={
                        "ComponentName": st.column_config.SelectboxColumn(
                            "Component",
                            options=absorber_gas_components,
                            required=True
                        ),
                        "MolarComposition[-]": st.column_config.NumberColumn(
                            "Mole Fraction",
                            min_value=0.0,
                            max_value=1.0,
                            format="%.6f"
                        )
                    },
                    num_rows='dynamic',
                    use_container_width=True,
                    key="teg_absorber_gas_editor"
                )
                st.session_state.teg_absorber_gas_df = edited_df
                
                with st.expander("💧 TEG Absorber Settings", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        teg_purity = st.number_input("Lean TEG Purity (wt%)", value=99.5, min_value=95.0, max_value=99.99,
                                                    help="TEG concentration in lean glycol")
                        teg_rate = st.number_input("TEG Rate (L TEG/kg H₂O)", value=25.0, min_value=10.0, max_value=50.0,
                                                  help="Typical: 20-35 L TEG per kg water removed")
                    with col2:
                        inlet_water_content = st.number_input("Inlet Water Content (mg/Sm³)", value=1500.0, min_value=100.0, max_value=5000.0,
                                                             help="Water content in wet gas")
                        gas_flow_rate = st.number_input("Gas Flow Rate (MSm³/d)", value=5.0, min_value=0.1, max_value=50.0,
                                                       help="Wet gas flow rate")
        else:
            st.subheader(f"📋 Fluid Composition - {emission_source}")
            available_components = ['CO2', 'H2S', 'methane', 'ethane', 'propane', 
                                   'i-butane', 'n-butane', 'i-pentane', 'n-pentane', 'n-hexane',
                                   'nitrogen', 'water']
            current_df = st.session_state.emission_gas_df
            
            edited_df = st.data_editor(
                current_df,
                column_config={
                    "ComponentName": st.column_config.SelectboxColumn(
                        "Component",
                        options=available_components,
                        required=True
                    ),
                    "MolarComposition[-]": st.column_config.NumberColumn(
                        "Mole Fraction",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.6f"
                    )
                },
                num_rows='dynamic',
                use_container_width=True,
                key="other_editor"
            )
            st.session_state.emission_gas_df = edited_df
        
        # Composition summary
        total_comp = edited_df['MolarComposition[-]'].sum()
        if abs(total_comp - 1.0) > 0.001:
            st.warning(f"Total composition: {total_comp:.4f} (normalization required)")
        else:
            st.success(f"Total composition: {total_comp:.4f} (valid)")
    
    with col2:
        if emission_source == "Produced Water Degassing" and calc_method == "Separator Equilibrium":
            st.subheader("📊 Separator Gas Composition")
            chart_title = "First Stage Separator Gas (mol%)"
        else:
            st.subheader("📊 Flash Gas Composition")
            chart_title = "Lab Flash Gas Composition (mol%)" if emission_source == "Produced Water Degassing" else "Fluid Composition (mol%)"
        
        vis_df = edited_df[edited_df['MolarComposition[-]'] > 0.001].copy()
        
        if len(vis_df) > 0:
            fig_comp = go.Figure(data=[go.Pie(
                labels=vis_df['ComponentName'],
                values=vis_df['MolarComposition[-]'],
                hole=0.4,
                textinfo='label+percent',
                marker=dict(colors=['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0', '#00BCD4', '#795548', '#607D8B'])
            )])
            fig_comp.update_layout(
                title=chart_title,
                showlegend=True,
                height=350
            )
            st.plotly_chart(fig_comp, use_container_width=True)
    
    st.divider()
    
    # Calculate button
    if st.button("🔬 Calculate Emissions", type="primary", use_container_width=True):
        
        if edited_df['MolarComposition[-]'].sum() <= 0:
            st.error("Gas composition required. Enter component mole fractions above.")
        else:
            with st.spinner("Running thermodynamic calculations..."):
                try:
                    # Import process equipment classes
                    Stream = jneqsim.process.equipment.stream.Stream
                    Separator = jneqsim.process.equipment.separator.Separator
                    
                    if emission_source == "Produced Water Degassing":
                        
                        if calc_method == "Separator Equilibrium":
                            # =============== SEPARATOR EQUILIBRIUM METHOD ===============
                            # Create a gas-water system at separator conditions
                            # Let it equilibrate, then extract the water phase with dissolved gas
                            
                            # Determine which thermodynamic model to use
                            use_soreide_whitson = salinity_model == "Søreide-Whitson PR-EoS"
                            
                            if use_soreide_whitson:
                                st.info(f"Using Separator Equilibrium method with **Søreide-Whitson PR-EoS** (salinity: {salinity_ppm:,} ppm, molality: {molality:.2f} mol/kg)...")
                            elif salinity_ppm > 0:
                                st.info(f"Using Separator Equilibrium method with **CPA-SRK + Setschenow correction** (salinity: {salinity_ppm:,} ppm, factor: {salting_out_factor:.3f})...")
                            else:
                                st.info("Using Separator Equilibrium method with **CPA-SRK**: calculating dissolved gas from VLE...")
                            
                            # Filter valid components (non-zero, non-water)
                            valid_gas_df = edited_df[
                                (edited_df['MolarComposition[-]'] > 0) & 
                                (edited_df['ComponentName'] != 'water')
                            ].copy()
                            
                            if len(valid_gas_df) == 0:
                                st.error("No valid gas components found. Please enter at least one gas component with non-zero mole fraction.")
                                st.stop()
                            
                            # Create gas-water equilibrium system
                            gas_scale = 100.0  # Large gas excess to preserve composition
                            water_moles = 1000.0  # Consistent water amount
                            max_iterations = 10
                            
                            for iteration in range(max_iterations):
                                if use_soreide_whitson:
                                    # Use Søreide-Whitson PR-EoS with salinity-dependent kij
                                    equilibrium_fluid = create_soreide_whitson_fluid()
                                else:
                                    # Use standard CPA - salting-out applied analytically later
                                    equilibrium_fluid = create_cpa_fluid(use_electrolyte=False)
                                
                                # Add gas components (scaled to preserve composition at equilibrium)
                                components_added = []
                                for _, row in valid_gas_df.iterrows():
                                    equilibrium_fluid.addComponent(row['ComponentName'], row['MolarComposition[-]'] * gas_scale)
                                    components_added.append(row['ComponentName'])
                                
                                # Add water
                                equilibrium_fluid.addComponent('water', water_moles)
                                
                                # Configure Søreide-Whitson model (requires setTotalFlowRate and addSalinity)
                                if use_soreide_whitson:
                                    # Set total flow rate (required for Søreide-Whitson model)
                                    total_moles = sum(valid_gas_df['MolarComposition[-]']) * gas_scale + water_moles
                                    equilibrium_fluid.setTotalFlowRate(total_moles, "mole/sec")
                                    
                                    # Add salinity (always call addSalinity for Søreide-Whitson, even if 0)
                                    # Convert ppm to moles NaCl: molality * kg water = moles NaCl
                                    kg_water = water_moles * 18.015 / 1000.0  # kg of water
                                    moles_nacl = molality * kg_water  # molality is 0 if salinity_ppm is 0
                                    equilibrium_fluid.addSalinity(moles_nacl, "mole/sec")
                                
                                # Configure model
                                equilibrium_fluid.createDatabase(True)
                                if use_soreide_whitson:
                                    equilibrium_fluid.setMixingRule(11)  # Søreide-Whitson mixing rule
                                else:
                                    equilibrium_fluid.setMixingRule(10)  # CPA mixing rule
                                equilibrium_fluid.setMultiPhaseCheck(True)
                                
                                # Set separator conditions
                                equilibrium_fluid.setTemperature(inlet_temp, 'C')
                                equilibrium_fluid.setPressure(inlet_pressure, 'bara')
                                
                                # Flash to equilibrium
                                TPflash(equilibrium_fluid)
                                equilibrium_fluid.initProperties()
                                
                                # Check for two phases
                                has_gas = equilibrium_fluid.hasPhaseType('gas')
                                has_water = equilibrium_fluid.hasPhaseType('aqueous')
                                
                                if has_gas and has_water:
                                    break  # Success - we have both phases
                                
                                # Double water and try again
                                water_moles *= 2.0
                            
                            if not (has_gas and has_water):
                                st.error(f"Could not achieve gas-water equilibrium after {max_iterations} iterations. Check conditions.")
                                st.stop()
                            
                            # Extract the aqueous phase composition (water with dissolved gas)
                            aqueous_phase = equilibrium_fluid.getPhase('aqueous')
                            
                            # Create a new fluid representing just the produced water
                            # Use the same model type for consistency in degassing stages
                            if use_soreide_whitson:
                                process_fluid = create_soreide_whitson_fluid()
                            else:
                                process_fluid = create_cpa_fluid(use_electrolyte=False)
                            
                            # Get composition of aqueous phase
                            # Apply salting-out correction to gas components only for CPA model
                            dissolved_gas_info = []
                            components_in_water = 0
                            total_gas_x = 0.0  # Track total gas mole fraction for renormalization
                            water_x = 0.0
                            
                            # First pass: collect compositions and calculate corrected values
                            comp_data = []
                            for idx in range(aqueous_phase.getNumberOfComponents()):
                                comp = aqueous_phase.getComponent(idx)
                                comp_name = str(comp.getComponentName())
                                x_i = float(comp.getx())
                                if x_i > 1e-10:
                                    if comp_name == 'water':
                                        water_x = x_i
                                    else:
                                        # For Søreide-Whitson, salinity is already accounted for in kij
                                        # For CPA, apply Setschenow salting-out factor
                                        if use_soreide_whitson:
                                            x_corrected = x_i  # No additional correction needed
                                        else:
                                            x_corrected = x_i * salting_out_factor
                                        total_gas_x += x_corrected
                                    comp_data.append((comp_name, x_i))
                            
                            # Renormalize: water + corrected gas = 1.0
                            water_new = 1.0 - total_gas_x
                            
                            for comp_name, x_i in comp_data:
                                if comp_name == 'water':
                                    process_fluid.addComponent(comp_name, water_new)
                                    components_in_water += 1
                                else:
                                    if use_soreide_whitson:
                                        x_corrected = x_i
                                    else:
                                        x_corrected = x_i * salting_out_factor
                                    process_fluid.addComponent(comp_name, x_corrected)
                                    components_in_water += 1
                                    dissolved_gas_info.append({
                                        'Component': comp_name,
                                        'Mole Fraction': x_corrected,
                                        'ppm (molar)': x_corrected * 1e6
                                    })
                            
                            # Verify process_fluid has components
                            if components_in_water == 0:
                                st.error("No components extracted from aqueous phase. This may indicate a thermodynamic calculation issue.")
                                st.stop()
                            
                            # Load binary interaction parameters and set mixing rule
                            process_fluid.createDatabase(True)
                            if use_soreide_whitson:
                                # Søreide-Whitson requires setTotalFlowRate and addSalinity before setMixingRule
                                process_fluid.setTotalFlowRate(1.0, "mole/sec")
                                # Always add salinity (even if 0) for Søreide-Whitson
                                kg_water_process = water_new * 18.015 / 1000.0
                                moles_nacl_process = molality * kg_water_process  # 0 if no salinity
                                process_fluid.addSalinity(moles_nacl_process, "mole/sec")
                                process_fluid.setMixingRule(11)  # Søreide-Whitson mixing rule
                            else:
                                process_fluid.setMixingRule(10)  # CPA mixing rule
                            process_fluid.setMultiPhaseCheck(True)  # Enable multi-phase detection
                            
                            # Display dissolved gas composition
                            if dissolved_gas_info:
                                st.markdown("**Calculated Dissolved Gas in Water at Separator Conditions:**")
                                dissolved_df = pd.DataFrame(dissolved_gas_info)
                                st.dataframe(dissolved_df.style.format({
                                    'Mole Fraction': '{:.2e}',
                                    'ppm (molar)': '{:.1f}'
                                }), use_container_width=True)
                            else:
                                st.warning("No dissolved gas detected in aqueous phase at these conditions.")
                            
                            # Set conditions and flow for process fluid
                            process_fluid.setTemperature(inlet_temp, 'C')
                            process_fluid.setPressure(inlet_pressure, 'bara')
                            process_fluid.setTotalFlowRate(total_flow, 'kg/hr')
                            
                        else:
                            # =============== LAB SAMPLE ANALYSIS METHOD ===============
                            # Original method: use lab gas composition ratios with assumed dissolved gas content
                            process_fluid = create_cpa_fluid()  # CPA-SRK-EOS-statoil
                            
                            # Add water as dominant component (99.9% of total moles)
                            process_fluid.addComponent('water', 0.999)
                            
                            # Add dissolved gas components - realistic solubility
                            # At 30 bara, 80°C: CH4 solubility ~0.0015 mol/mol, CO2 ~0.003 mol/mol
                            # Total dissolved gas ~0.001 mol/mol (0.1%)
                            gas_scale = 0.001  # 0.1 mol% dissolved gas - realistic for produced water
                            for _, row in edited_df.iterrows():
                                # Skip water and zero composition components
                                if row['MolarComposition[-]'] > 0 and row['ComponentName'] != 'water':
                                    process_fluid.addComponent(row['ComponentName'], row['MolarComposition[-]'] * gas_scale)
                            
                            # Load binary interaction parameters and set mixing rule
                            process_fluid.createDatabase(True)
                            process_fluid.setMixingRule(10)
                            process_fluid.setMultiPhaseCheck(True)  # Enable multi-phase detection
                            
                            # Set initial conditions
                            process_fluid.setTemperature(inlet_temp, 'C')
                            process_fluid.setPressure(inlet_pressure, 'bara')
                            process_fluid.setTotalFlowRate(total_flow, 'kg/hr')
                            
                    elif emission_source == "TEG Regeneration":
                        # TEG Regeneration: calculate rich TEG composition
                        if calc_method == "Absorber Equilibrium":
                            # =============== ABSORBER EQUILIBRIUM METHOD ===============
                            # Create a gas-TEG system at absorber conditions
                            # Let it equilibrate, then extract the liquid (rich TEG) phase with absorbed components
                            
                            st.info("Using Absorber Equilibrium method: calculating absorbed components from gas-TEG VLE...")
                            
                            # Get absorber gas composition
                            absorber_gas_df = st.session_state.teg_absorber_gas_df
                            valid_gas_df = absorber_gas_df[
                                (absorber_gas_df['MolarComposition[-]'] > 0) & 
                                (absorber_gas_df['ComponentName'] != 'TEG') &
                                (absorber_gas_df['ComponentName'] != 'water')
                            ].copy()
                            
                            if len(valid_gas_df) == 0:
                                st.error("No valid gas components found. Please enter at least one gas component with non-zero mole fraction.")
                                st.stop()
                            
                            # Create gas-TEG equilibrium system
                            # Use iterative approach: adjust TEG amount until proper liquid phase exists
                            # TEG is the dominant liquid component, gas components will dissolve into it
                            gas_scale = 100.0  # Large gas excess to preserve gas composition
                            teg_moles = 10.0  # Start with TEG (will be scaled)
                            water_in_teg = teg_moles * (1.0 - teg_purity/100.0) / (teg_purity/100.0)  # Water in lean TEG
                            max_iterations = 10
                            
                            for iteration in range(max_iterations):
                                equilibrium_fluid = create_cpa_fluid(use_electrolyte=False)
                                
                                # Add gas components (scaled to preserve composition at equilibrium)
                                components_added = []
                                for _, row in valid_gas_df.iterrows():
                                    equilibrium_fluid.addComponent(row['ComponentName'], row['MolarComposition[-]'] * gas_scale)
                                    components_added.append(row['ComponentName'])
                                
                                # Add TEG (lean glycol)
                                equilibrium_fluid.addComponent('TEG', teg_moles)
                                
                                # Add water in lean TEG (based on purity)
                                if water_in_teg > 0.001:
                                    equilibrium_fluid.addComponent('water', water_in_teg)
                                
                                # Configure CPA model
                                equilibrium_fluid.createDatabase(True)
                                equilibrium_fluid.setMixingRule(10)
                                equilibrium_fluid.setMultiPhaseCheck(True)
                                
                                # Set absorber conditions (contactor T and P)
                                equilibrium_fluid.setTemperature(inlet_temp, 'C')
                                equilibrium_fluid.setPressure(inlet_pressure, 'bara')
                                
                                # Flash to equilibrium
                                TPflash(equilibrium_fluid)
                                equilibrium_fluid.initProperties()
                                
                                # Check for gas and liquid phases
                                # Note: TEG-rich phase appears as 'aqueous' in CPA-Statoil, not 'oil'
                                has_gas = equilibrium_fluid.hasPhaseType('gas')
                                has_teg_liquid = equilibrium_fluid.hasPhaseType('aqueous')
                                
                                if has_gas and has_teg_liquid:
                                    break  # Success - we have both phases
                                
                                # Increase TEG and try again
                                teg_moles *= 2.0
                                water_in_teg = teg_moles * (1.0 - teg_purity/100.0) / (teg_purity/100.0)
                            
                            if not (has_gas and has_teg_liquid):
                                st.warning(f"Could not achieve gas-TEG equilibrium after {max_iterations} iterations. Using direct composition method as fallback.")
                                # Fallback to direct composition
                                teg_df = st.session_state.teg_fluid_df
                                process_fluid = create_cpa_fluid()
                                for _, row in teg_df.iterrows():
                                    if row['MolarComposition[-]'] > 0:
                                        process_fluid.addComponent(row['ComponentName'], row['MolarComposition[-]'])
                                process_fluid.createDatabase(True)
                                process_fluid.setMixingRule(10)
                                process_fluid.setMultiPhaseCheck(True)
                                process_fluid.setTemperature(inlet_temp, 'C')
                                process_fluid.setPressure(inlet_pressure, 'bara')
                                process_fluid.setTotalFlowRate(total_flow, 'kg/hr')
                            else:
                                # Extract the aqueous/liquid phase composition (rich TEG with absorbed components)
                                # TEG-rich phase is classified as 'aqueous' in CPA-Statoil due to TEG's polar nature
                                teg_liquid_phase = equilibrium_fluid.getPhase('aqueous')
                                
                                # Create a new CPA fluid representing the rich TEG
                                process_fluid = create_cpa_fluid(use_electrolyte=False)
                                
                                # Get composition of TEG-rich liquid phase (rich TEG)
                                absorbed_gas_info = []
                                components_in_teg = 0
                                for idx in range(teg_liquid_phase.getNumberOfComponents()):
                                    comp = teg_liquid_phase.getComponent(idx)
                                    comp_name = str(comp.getComponentName())
                                    x_i = float(comp.getx())  # mole fraction in TEG-rich (aqueous) phase
                                    if x_i > 1e-12:
                                        process_fluid.addComponent(comp_name, x_i)
                                        components_in_teg += 1
                                        if comp_name not in ['TEG', 'water']:
                                            absorbed_gas_info.append({
                                                'Component': comp_name,
                                                'Mole Fraction in Rich TEG': x_i,
                                                'ppm (molar)': x_i * 1e6
                                            })
                                
                                # Verify process_fluid has components
                                if components_in_teg == 0:
                                    st.error("No components extracted from TEG phase. This may indicate a thermodynamic calculation issue.")
                                    st.stop()
                                
                                # Load binary interaction parameters and set mixing rule
                                process_fluid.createDatabase(True)
                                process_fluid.setMixingRule(10)
                                process_fluid.setMultiPhaseCheck(True)
                                
                                # Display absorbed gas composition in rich TEG
                                if absorbed_gas_info:
                                    st.markdown("**Calculated Absorbed Components in Rich TEG at Absorber Conditions:**")
                                    absorbed_df = pd.DataFrame(absorbed_gas_info)
                                    st.dataframe(absorbed_df.style.format({
                                        'Mole Fraction in Rich TEG': '{:.2e}',
                                        'ppm (molar)': '{:.1f}'
                                    }), use_container_width=True)
                                else:
                                    st.warning("No absorbed gas detected in TEG phase at these conditions.")
                                
                                # Set conditions and flow for process fluid
                                process_fluid.setTemperature(inlet_temp, 'C')
                                process_fluid.setPressure(inlet_pressure, 'bara')
                                process_fluid.setTotalFlowRate(total_flow, 'kg/hr')
                        else:
                            # =============== RICH TEG COMPOSITION METHOD ===============
                            # Direct input of rich TEG composition
                            teg_df = st.session_state.teg_fluid_df
                            
                            process_fluid = create_cpa_fluid()  # CPA-SRK-EOS-statoil
                            for _, row in teg_df.iterrows():
                                if row['MolarComposition[-]'] > 0:
                                    process_fluid.addComponent(row['ComponentName'], row['MolarComposition[-]'])
                            
                            # Load binary interaction parameters and set mixing rule
                            process_fluid.createDatabase(True)
                            process_fluid.setMixingRule(10)
                            process_fluid.setMultiPhaseCheck(True)  # Enable multi-phase detection
                            
                            process_fluid.setTemperature(inlet_temp, 'C')
                            process_fluid.setPressure(inlet_pressure, 'bara')
                            process_fluid.setTotalFlowRate(total_flow, 'kg/hr')
                        
                    else:
                        # For other sources (Tank, Cold Vent), use composition directly
                        process_fluid = create_cpa_fluid()  # CPA-SRK-EOS-statoil
                        for _, row in edited_df.iterrows():
                            if row['MolarComposition[-]'] > 0:
                                process_fluid.addComponent(row['ComponentName'], row['MolarComposition[-]'])
                        
                        # Load binary interaction parameters and set mixing rule
                        process_fluid.createDatabase(True)
                        process_fluid.setMixingRule(10)
                        process_fluid.setMultiPhaseCheck(True)  # Enable multi-phase detection
                        
                        process_fluid.setTemperature(inlet_temp, 'C')
                        process_fluid.setPressure(inlet_pressure, 'bara')
                        process_fluid.setTotalFlowRate(total_flow, 'kg/hr')
                    
                    # Flash at inlet conditions
                    TPflash(process_fluid)
                    process_fluid.initProperties()
                    
                    # Multi-stage calculations
                    emissions_data = []
                    
                    if use_process_sim and emission_source == "Produced Water Degassing":
                        # Use process simulation with ThrottlingValve + Separator equipment
                        ThrottlingValve = jneqsim.process.equipment.valve.ThrottlingValve
                        
                        inlet_stream = Stream("PW-Feed", process_fluid)
                        inlet_stream.run()
                        
                        current_stream = inlet_stream
                        for i, pressure in enumerate(stage_pressures):
                            # Use throttling valve to reduce pressure before separator
                            valve = ThrottlingValve(f"Valve-{stage_names[i]}", current_stream)
                            valve.setOutletPressure(pressure)
                            valve.run()
                            
                            # Create separator at reduced pressure
                            sep = Separator(stage_names[i], valve.getOutletStream())
                            sep.run()
                            
                            stage_result = {
                                'Stage': stage_names[i],
                                'Pressure (bara)': pressure,
                                'Temperature (°C)': stage_temps[i],
                                'Total_kghr': 0.0,
                                'CO2_kghr': 0.0,
                                'CH4_kghr': 0.0,
                                'C2_kghr': 0.0,
                                'C3_kghr': 0.0,
                                'C4plus_kghr': 0.0,
                                'nmVOC_kghr': 0.0,
                                'H2S_kghr': 0.0
                            }
                            
                            # Get gas emissions from separator
                            gas_stream = sep.getGasOutStream()
                            if gas_stream is not None:
                                gas_stream.run()
                                gas_fluid = gas_stream.getFluid()
                                if gas_fluid.hasPhaseType('gas'):
                                    gas = gas_fluid.getPhase('gas')
                                    stage_result['Total_kghr'] = gas.getFlowRate('kg/hr')
                                    
                                    # Extract individual component flows
                                    c4plus_total = 0.0
                                    
                                    for comp_name in ['CO2', 'methane', 'ethane', 'propane', 'H2S', 'nitrogen',
                                                     'i-butane', 'n-butane', 'i-pentane', 'n-pentane', 'n-hexane',
                                                     'n-heptane', 'n-octane', 'benzene', 'toluene', 'ethylbenzene',
                                                     'm-xylene', 'o-xylene', 'p-xylene']:
                                        try:
                                            comp = gas.getComponent(comp_name)
                                            if comp is not None:
                                                flow = comp.getFlowRate('kg/hr')
                                                if comp_name == 'CO2':
                                                    stage_result['CO2_kghr'] = flow
                                                elif comp_name == 'methane':
                                                    stage_result['CH4_kghr'] = flow
                                                elif comp_name == 'ethane':
                                                    stage_result['C2_kghr'] = flow
                                                elif comp_name == 'propane':
                                                    stage_result['C3_kghr'] = flow
                                                elif comp_name == 'H2S':
                                                    stage_result['H2S_kghr'] = flow
                                                elif comp_name in ['i-butane', 'n-butane', 'i-pentane', 'n-pentane', 
                                                                  'n-hexane', 'n-heptane', 'n-octane',
                                                                  'benzene', 'toluene', 'ethylbenzene',
                                                                  'm-xylene', 'o-xylene', 'p-xylene']:
                                                    c4plus_total += flow
                                        except:
                                            pass
                                    
                                    stage_result['C4plus_kghr'] = c4plus_total
                                    # nmVOC = C2 + C3 + C4+ (all non-methane hydrocarbons except CO2 and H2S)
                                    stage_result['nmVOC_kghr'] = stage_result['C2_kghr'] + stage_result['C3_kghr'] + c4plus_total
                            
                            emissions_data.append(stage_result)
                            
                            # Use liquid outlet as feed to next stage
                            liquid_stream = sep.getLiquidOutStream()
                            if liquid_stream is not None:
                                current_stream = liquid_stream
                    else:
                        # Use simple flash calculation (fallback method)
                        current_fluid = process_fluid
                        
                        for i, pressure in enumerate(stage_pressures):
                            stage_fluid = current_fluid.clone()
                            stage_fluid.setPressure(pressure, 'bara')
                            stage_fluid.setTemperature(stage_temps[i], 'C')
                            TPflash(stage_fluid)
                            stage_fluid.initProperties()
                            
                            stage_result = {
                                'Stage': stage_names[i],
                                'Pressure (bara)': pressure,
                                'Temperature (°C)': stage_temps[i],
                                'Total_kghr': 0.0,
                                'CO2_kghr': 0.0,
                                'CH4_kghr': 0.0,
                                'C2_kghr': 0.0,
                                'C3_kghr': 0.0,
                                'C4plus_kghr': 0.0,
                                'nmVOC_kghr': 0.0,
                                'H2S_kghr': 0.0
                            }
                            
                            if stage_fluid.hasPhaseType('gas'):
                                gas = stage_fluid.getPhase('gas')
                                stage_result['Total_kghr'] = gas.getFlowRate('kg/hr')
                                
                                # Extract individual component flows including C4+ and aromatics
                                c4plus_total = 0.0
                                for comp_name in ['CO2', 'methane', 'ethane', 'propane', 'H2S', 'nitrogen',
                                                 'i-butane', 'n-butane', 'i-pentane', 'n-pentane', 'n-hexane',
                                                 'n-heptane', 'n-octane', 'benzene', 'toluene', 'ethylbenzene',
                                                 'm-xylene', 'o-xylene', 'p-xylene']:
                                    try:
                                        comp = gas.getComponent(comp_name)
                                        if comp is not None:
                                            flow = comp.getFlowRate('kg/hr')
                                            if comp_name == 'CO2':
                                                stage_result['CO2_kghr'] = flow
                                            elif comp_name == 'methane':
                                                stage_result['CH4_kghr'] = flow
                                            elif comp_name == 'ethane':
                                                stage_result['C2_kghr'] = flow
                                            elif comp_name == 'propane':
                                                stage_result['C3_kghr'] = flow
                                            elif comp_name == 'H2S':
                                                stage_result['H2S_kghr'] = flow
                                            elif comp_name in ['i-butane', 'n-butane', 'i-pentane', 'n-pentane', 
                                                              'n-hexane', 'n-heptane', 'n-octane',
                                                              'benzene', 'toluene', 'ethylbenzene',
                                                              'm-xylene', 'o-xylene', 'p-xylene']:
                                                c4plus_total += flow
                                    except:
                                        pass
                                
                                stage_result['C4plus_kghr'] = c4plus_total
                                # nmVOC = C2 + C3 + C4+ (all non-methane hydrocarbons)
                                stage_result['nmVOC_kghr'] = stage_result['C2_kghr'] + stage_result['C3_kghr'] + c4plus_total
                            
                            emissions_data.append(stage_result)
                            
                            if stage_fluid.hasPhaseType('aqueous') or stage_fluid.hasPhaseType('oil'):
                                current_fluid = stage_fluid
                    
                    # Create results DataFrame
                    results_df = pd.DataFrame(emissions_data)
                    
                    # Calculate totals
                    total_gas = results_df['Total_kghr'].sum()
                    total_co2 = results_df['CO2_kghr'].sum()
                    total_ch4 = results_df['CH4_kghr'].sum()
                    total_nmvoc = results_df['nmVOC_kghr'].sum()
                    total_h2s = results_df['H2S_kghr'].sum()
                    
                    # CO2 equivalents
                    co2eq_hr = total_co2 + total_ch4 * gwp_ch4 + total_nmvoc * gwp_nmvoc
                    co2eq_year = co2eq_hr * 8760 / 1000
                    
                    # Store results in session state for other tabs
                    st.session_state.emission_results = {
                        'total_gas': total_gas,
                        'total_co2': total_co2,
                        'total_ch4': total_ch4,
                        'total_nmvoc': total_nmvoc,
                        'total_h2s': total_h2s,
                        'co2eq_hr': co2eq_hr,
                        'co2eq_year': co2eq_year,
                        'results_df': results_df
                    }
                    
                    st.success("Calculation completed successfully")
                    
                    # Results display
                    result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs(
                        ["📊 Summary", "📈 By Stage", "📉 Method Comparison", "📥 Export"]
                    )
                    
                    with result_tab1:
                        st.subheader("Total Emissions Summary")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Gas", f"{total_gas:.1f} kg/hr", f"{total_gas*8760/1000:.0f} t/yr")
                        with col2:
                            st.metric("CO₂", f"{total_co2:.1f} kg/hr", f"{total_co2*8760/1000:.0f} t/yr")
                        with col3:
                            st.metric("Methane (CH₄)", f"{total_ch4:.1f} kg/hr", f"{total_ch4*8760/1000:.0f} t/yr")
                        with col4:
                            st.metric("nmVOC", f"{total_nmvoc:.1f} kg/hr", f"{total_nmvoc*8760/1000:.0f} t/yr")
                        
                        st.divider()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("🌍 CO₂ Equivalent", f"{co2eq_hr:.0f} kg/hr", f"{co2eq_year:.0f} tonnes CO₂eq/yr")
                        
                        with col2:
                            if total_gas > 0:
                                fig_pie = go.Figure(data=[go.Pie(
                                    labels=['CO₂', 'Methane', 'nmVOC', 'Other'],
                                    values=[total_co2, total_ch4, total_nmvoc, max(0, total_gas - total_co2 - total_ch4 - total_nmvoc)],
                                    hole=0.4,
                                    marker=dict(colors=['#2196F3', '#FF9800', '#4CAF50', '#9E9E9E']),
                                    textinfo='label+percent'
                                )])
                                fig_pie.update_layout(title="Emission Composition (by mass)", height=300)
                                st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with result_tab2:
                        st.subheader("Emissions by Stage")
                        
                        display_df = results_df.copy()
                        display_df.columns = ['Stage', 'Pressure (bara)', 'Temperature (°C)', 'Total (kg/hr)', 'CO₂ (kg/hr)', 
                                             'CH₄ (kg/hr)', 'C₂ (kg/hr)', 'C₃ (kg/hr)', 'C₄+ (kg/hr)', 'nmVOC (kg/hr)', 'H₂S (kg/hr)']
                        st.dataframe(display_df, use_container_width=True)
                        
                        if len(results_df) > 0:
                            fig_bar = go.Figure()
                            fig_bar.add_trace(go.Bar(name='CO₂', x=results_df['Stage'], y=results_df['CO2_kghr'], marker_color='#2196F3'))
                            fig_bar.add_trace(go.Bar(name='Methane', x=results_df['Stage'], y=results_df['CH4_kghr'], marker_color='#FF9800'))
                            fig_bar.add_trace(go.Bar(name='nmVOC', x=results_df['Stage'], y=results_df['nmVOC_kghr'], marker_color='#4CAF50'))
                            chart_title = "Emissions by Process Stage" if emission_source == "TEG Regeneration" else "Emissions by Degassing Stage"
                            fig_bar.update_layout(title=chart_title, xaxis_title="Stage", yaxis_title="Emission Rate (kg/hr)", barmode='group', height=400)
                            st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # nmVOC breakdown for TEG Regeneration
                        if emission_source == "TEG Regeneration" and total_nmvoc > 0:
                            st.markdown("### nmVOC Composition Breakdown")
                            st.caption("Non-methane volatile organic compounds (C₂+ hydrocarbons)")
                            
                            total_c2 = results_df['C2_kghr'].sum()
                            total_c3 = results_df['C3_kghr'].sum()
                            total_c4plus = results_df['C4plus_kghr'].sum()
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                nmvoc_breakdown = pd.DataFrame({
                                    'Component': ['Ethane (C₂)', 'Propane (C₃)', 'C₄+ (butanes, pentanes, hexane, aromatics)'],
                                    'kg/hr': [total_c2, total_c3, total_c4plus],
                                    't/yr': [total_c2 * 8760 / 1000, total_c3 * 8760 / 1000, total_c4plus * 8760 / 1000],
                                    '% of nmVOC': [total_c2 / total_nmvoc * 100 if total_nmvoc > 0 else 0,
                                                  total_c3 / total_nmvoc * 100 if total_nmvoc > 0 else 0,
                                                  total_c4plus / total_nmvoc * 100 if total_nmvoc > 0 else 0]
                                })
                                st.dataframe(nmvoc_breakdown.style.format({
                                    'kg/hr': '{:.2f}',
                                    't/yr': '{:.2f}',
                                    '% of nmVOC': '{:.1f}%'
                                }), use_container_width=True)
                            
                            with col2:
                                fig_nmvoc = go.Figure(data=[go.Pie(
                                    labels=['Ethane (C₂)', 'Propane (C₃)', 'C₄+ (incl. aromatics)'],
                                    values=[total_c2, total_c3, total_c4plus],
                                    hole=0.4,
                                    marker=dict(colors=['#8BC34A', '#4CAF50', '#2E7D32']),
                                    textinfo='label+percent'
                                )])
                                fig_nmvoc.update_layout(title="nmVOC Composition", height=280)
                                st.plotly_chart(fig_nmvoc, use_container_width=True)
                            
                            st.info("""
                            **nmVOC Definition**: Non-methane Volatile Organic Compounds include all hydrocarbons 
                            except methane (C₂+). For TEG regeneration, this typically includes:
                            - Light hydrocarbons absorbed from the gas (ethane, propane, butanes)
                            - Heavier hydrocarbons (pentanes, hexane)
                            - Aromatics (benzene, toluene, xylenes - BTEX)
                            
                            BTEX compounds are particularly important for environmental reporting due to their 
                            health impacts and are regulated separately in some jurisdictions.
                            """)
                    
                    with result_tab3:
                        st.subheader("NeqSim vs Norwegian Handbook Method")
                        
                        if emission_source == "TEG Regeneration":
                            st.markdown("""
                            ### TEG Regeneration Emissions Analysis
                            
                            **Note:** The Norwegian Handbook Method (Retningslinje 044) emission factors are designed 
                            for produced water degassing and are **not directly applicable** to TEG regeneration emissions.
                            
                            TEG regeneration emissions depend on:
                            - TEG circulation rate and rich TEG composition
                            - Flash drum and regenerator operating conditions
                            - Hydrocarbon absorption in the contactor (function of gas composition, T, P)
                            - BTEX content in the inlet gas
                            
                            **NeqSim Thermodynamic Method** provides component-specific emissions including:
                            """)
                            
                            neqsim_co2_t = total_co2 * 8760 / 1000
                            neqsim_ch4_t = total_ch4 * 8760 / 1000
                            neqsim_nmvoc_t = total_nmvoc * 8760 / 1000
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                teg_summary = pd.DataFrame({
                                    'Component': ['CO₂', 'Methane (CH₄)', 'nmVOC (C₂+)', 'Total Hydrocarbon'],
                                    'kg/hr': [total_co2, total_ch4, total_nmvoc, total_ch4 + total_nmvoc],
                                    't/yr': [neqsim_co2_t, neqsim_ch4_t, neqsim_nmvoc_t, neqsim_ch4_t + neqsim_nmvoc_t]
                                })
                                st.dataframe(teg_summary.style.format({
                                    'kg/hr': '{:.2f}',
                                    't/yr': '{:.2f}'
                                }), use_container_width=True)
                            
                            with col2:
                                # CO2 equivalent breakdown
                                co2_from_co2 = total_co2 * 8760 / 1000  # t CO2eq/yr
                                co2_from_ch4 = total_ch4 * gwp_ch4 * 8760 / 1000  # t CO2eq/yr
                                co2_from_nmvoc = total_nmvoc * gwp_nmvoc * 8760 / 1000  # t CO2eq/yr
                                
                                fig_co2eq = go.Figure(data=[go.Pie(
                                    labels=['From CO₂', f'From CH₄ (GWP={gwp_ch4})', f'From nmVOC (GWP={gwp_nmvoc})'],
                                    values=[co2_from_co2, co2_from_ch4, co2_from_nmvoc],
                                    hole=0.4,
                                    marker=dict(colors=['#2196F3', '#FF9800', '#4CAF50']),
                                    textinfo='label+percent'
                                )])
                                fig_co2eq.update_layout(title="CO₂eq Contribution by Source", height=280)
                                st.plotly_chart(fig_co2eq, use_container_width=True)
                            
                            st.info(f"""
                            **CO₂ Equivalent Summary (GWP-100):**
                            - CO₂ direct: {co2_from_co2:.1f} t CO₂eq/yr
                            - CH₄ contribution: {co2_from_ch4:.1f} t CO₂eq/yr (GWP = {gwp_ch4})
                            - nmVOC contribution: {co2_from_nmvoc:.1f} t CO₂eq/yr (GWP = {gwp_nmvoc})
                            - **Total: {co2eq_year:.1f} t CO₂eq/yr**
                            """)
                            
                            # ===================== EPA/INDUSTRY BENCHMARK COMPARISON =====================
                            st.divider()
                            st.markdown("### 📊 Comparison with EPA/Industry Benchmarks")
                            
                            st.markdown("""
                            The EPA Natural Gas STAR program and industry standards provide reference values 
                            for TEG dehydrator emissions. These benchmarks can help validate your results.
                            """)
                            
                            # Calculate benchmark metrics
                            # EPA benchmark: ~1 scf CH4 absorbed per gallon TEG circulated (without flash tank)
                            # Convert TEG flow from kg/hr to gal/hr (TEG density ~1.125 kg/L, 3.785 L/gal)
                            teg_density_kg_per_gal = 1.125 * 3.785  # ~4.26 kg/gal
                            teg_flow_gal_hr = total_flow / teg_density_kg_per_gal
                            
                            # Convert CH4 from kg/hr to scf/hr (CH4 density at SC: 0.0178 lb/scf = 0.00807 kg/scf)
                            ch4_density_kg_per_scf = 0.00807
                            ch4_scf_hr = total_ch4 / ch4_density_kg_per_scf if total_ch4 > 0 else 0
                            
                            # Calculate scf CH4 per gallon TEG
                            ch4_per_gal_teg = ch4_scf_hr / teg_flow_gal_hr if teg_flow_gal_hr > 0 else 0
                            
                            # BTEX fraction calculation (benzene, toluene, xylenes in C4+)
                            # Extract BTEX emissions from results if available
                            btex_kghr = 0.0
                            for stage_result in emissions_data:
                                # BTEX is included in C4plus, estimate based on typical TEG absorption
                                # Typical BTEX fraction: 10-30% of C4+ for rich gas
                                pass
                            
                            # Estimate BTEX from C4+ (conservative estimate: ~20% of C4+)
                            total_c4plus = results_df['C4plus_kghr'].sum()
                            estimated_btex = total_c4plus * 0.20  # Conservative estimate
                            btex_fraction = (estimated_btex / total_nmvoc * 100) if total_nmvoc > 0 else 0
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("#### Your Calculated Values vs EPA Benchmarks")
                                
                                benchmark_data = pd.DataFrame({
                                    'Parameter': [
                                        'CH₄ per gal TEG',
                                        'Total VOC (nmVOC)',
                                        'BTEX fraction of nmVOC',
                                        'CH₄ % of total emissions'
                                    ],
                                    'Your Value': [
                                        f"{ch4_per_gal_teg:.2f} scf/gal",
                                        f"{total_nmvoc:.2f} kg/hr",
                                        f"{btex_fraction:.1f}% (est.)",
                                        f"{(total_ch4/(total_gas)*100) if total_gas > 0 else 0:.1f}%"
                                    ],
                                    'EPA/Industry Benchmark': [
                                        "~1 scf/gal (no flash tank)",
                                        "Varies by gas composition",
                                        "10-30% typical",
                                        "Varies (typically 30-70%)"
                                    ],
                                    'Status': [
                                        "✅ Within range" if 0.1 < ch4_per_gal_teg < 3.0 else "⚠️ Check inputs",
                                        "ℹ️ Site-specific",
                                        "✅ Within range" if 5 < btex_fraction < 40 else "ℹ️ Site-specific",
                                        "ℹ️ Site-specific"
                                    ]
                                })
                                st.dataframe(benchmark_data, use_container_width=True, hide_index=True)
                            
                            with col2:
                                st.markdown("#### Key EPA Reference Values")
                                st.markdown("""
                                | Parameter | EPA Value | Source |
                                |-----------|-----------|--------|
                                | CH₄ absorption | 1 scf/gal TEG | Natural Gas STAR |
                                | Additional (gas-assist pump) | +2 scf/gal | Natural Gas STAR |
                                | TEG-to-water ratio | 2-5 gal/lb H₂O | Industry standard |
                                | Rule-of-thumb | 3 gal TEG/lb H₂O | Industry standard |
                                """)
                                
                                # Emission intensity metric
                                if teg_flow_gal_hr > 0:
                                    emission_intensity = (total_ch4 + total_nmvoc) / teg_flow_gal_hr
                                    st.metric(
                                        "Emission Intensity",
                                        f"{emission_intensity:.3f} kg HC/gal TEG",
                                        help="Total hydrocarbon emissions per gallon of TEG circulated"
                                    )
                            
                            # Regulatory references
                            with st.expander("📚 Regulatory References & Standards", expanded=False):
                                st.markdown("""
                                ### U.S. EPA Regulations
                                
                                | Regulation | Description | Applicability |
                                |------------|-------------|---------------|
                                | **40 CFR 98 Subpart W** | GHG Reporting for glycol dehydrators | Facilities ≥25,000 t CO₂eq/yr |
                                | **40 CFR 60 NSPS OOOOa/b** | New Source Performance Standards | New/modified sources |
                                | **40 CFR 63 Subpart HH** | HAP emissions (BTEX) | Major HAP sources |
                                
                                ### Calculation Methodology
                                
                                EPA recommends using **simulation software** for glycol dehydrator emissions 
                                (40 CFR 98.233(e)) because:
                                
                                > *"A default emission factor is not available to adequately estimate emissions 
                                > due to the wide variety of configurations and operating conditions."*
                                
                                The NeqSim thermodynamic approach used here aligns with EPA guidance for 
                                rigorous emission quantification.
                                
                                ### Other Standards
                                
                                | Standard | Region | Notes |
                                |----------|--------|-------|
                                | **OGMP 2.0** | International | Methane reporting framework |
                                | **EU Methane Regulation 2024/1787** | European Union | Mandatory monitoring |
                                | **Aktivitetsforskriften §70** | Norway | Offshore emission reporting |
                                
                                ### References
                                
                                - [EPA Natural Gas STAR - Glycol Dehydrators](https://www.epa.gov/natural-gas-star-program/glycol-dehydrators)
                                - [EPA Subpart W Reporting](https://www.epa.gov/ghgreporting/subpart-w-petroleum-and-natural-gas-systems)
                                - [Optimize Glycol Circulation](https://www.epa.gov/natural-gas-star-program/optimize-glycol-circulation)
                                """)
                            
                            # Assessment summary
                            if ch4_per_gal_teg > 0:
                                if ch4_per_gal_teg < 0.5:
                                    st.success(f"""
                                    **Assessment:** Your CH₄ emission rate ({ch4_per_gal_teg:.2f} scf/gal TEG) is **below** 
                                    the EPA benchmark of ~1 scf/gal. This may indicate:
                                    - Effective flash tank separation recovering methane
                                    - Lower hydrocarbon content in inlet gas
                                    - Well-optimized TEG circulation rate
                                    """)
                                elif ch4_per_gal_teg < 1.5:
                                    st.info(f"""
                                    **Assessment:** Your CH₄ emission rate ({ch4_per_gal_teg:.2f} scf/gal TEG) is 
                                    **within typical range** of EPA benchmarks. This is expected for standard 
                                    TEG dehydration without vapor recovery.
                                    """)
                                elif ch4_per_gal_teg < 3.0:
                                    st.warning(f"""
                                    **Assessment:** Your CH₄ emission rate ({ch4_per_gal_teg:.2f} scf/gal TEG) is 
                                    **above** the basic EPA benchmark. This may indicate:
                                    - Gas-assist glycol circulation pump in use (+2 scf/gal)
                                    - Higher than optimal TEG circulation rate
                                    - Consider [optimizing glycol circulation](https://www.epa.gov/natural-gas-star-program/optimize-glycol-circulation)
                                    """)
                                else:
                                    st.error(f"""
                                    **Assessment:** Your CH₄ emission rate ({ch4_per_gal_teg:.2f} scf/gal TEG) is 
                                    **significantly above** industry benchmarks. Consider:
                                    - Verifying input data and TEG circulation rate
                                    - Installing flash tank separator for vapor recovery
                                    - Optimizing glycol circulation rate
                                    - Replacing gas-assist pump with electric pump
                                    """)
                        else:
                            st.markdown("""
                            The **Norwegian Handbook Method** (Retningslinje 044) uses fixed emission factors:
                            
                            | Factor | Value | Scope |
                            |--------|-------|-------|
                            | f_CH₄ | 14 g/(m³·bar) | Methane only |
                            | f_nmVOC | 3.5 g/(m³·bar) | C₂+ hydrocarbons only |
                            
                            ⚠️ **Critical Limitation:** The conventional method ignores CO₂ emissions entirely.
                            CO₂ typically represents 50–80% of total dissolved gas in produced water, 
                            yet is neither quantified nor reported using handbook factors.
                            """)
                            
                            # Conventional calculation (tonnes/year)
                            # Formula: emission = factor × volume × pressure_drop × 1e-6 (g→tonnes)
                            water_vol_m3_year = water_flow_m3hr * 8760 if emission_source == "Produced Water Degassing" else total_flow / 1000 * 8760
                            pressure_drop_bar = inlet_pressure - 1.01325
                            
                            conv_ch4 = 14.0 * water_vol_m3_year * pressure_drop_bar * 1e-6  # tonnes/year
                            conv_nmvoc = 3.5 * water_vol_m3_year * pressure_drop_bar * 1e-6  # tonnes/year
                            conv_co2eq = conv_ch4 * gwp_ch4 + conv_nmvoc * gwp_nmvoc
                            
                            neqsim_co2_t = total_co2 * 8760 / 1000
                            neqsim_ch4_t = total_ch4 * 8760 / 1000
                            neqsim_nmvoc_t = total_nmvoc * 8760 / 1000
                            
                            # Calculate differences - thermodynamic method typically shows LOWER emissions
                            # because conventional factors are overly conservative
                            ch4_diff = ((neqsim_ch4_t - conv_ch4) / conv_ch4 * 100) if conv_ch4 > 0 else 0
                            nmvoc_diff = ((neqsim_nmvoc_t - conv_nmvoc) / conv_nmvoc * 100) if conv_nmvoc > 0 else 0
                            co2eq_diff = ((co2eq_year - conv_co2eq) / conv_co2eq * 100) if conv_co2eq > 0 else 0
                            
                            comparison_data = {
                                'Parameter': ['CO₂ (t/yr)', 'CH₄ (t/yr)', 'nmVOC (t/yr)', 'CO₂eq (t/yr)'],
                                'Conventional': [f"Not measured", f"{conv_ch4:.1f}", f"{conv_nmvoc:.1f}", f"{conv_co2eq:.1f}"],
                                'NeqSim Thermodynamic': [f"{neqsim_co2_t:.1f}", f"{neqsim_ch4_t:.1f}", f"{neqsim_nmvoc_t:.1f}", f"{co2eq_year:.1f}"],
                                'Difference': [f"+{neqsim_co2_t:.1f} t/yr (ignored by conv.)", 
                                              f"{ch4_diff:+.0f}%",
                                              f"{nmvoc_diff:+.0f}%",
                                              f"{co2eq_diff:+.0f}%"]
                            }
                            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
                            
                            # Calculate GWMF for comparison (only meaningful for Produced Water Degassing)
                            if emission_source == "Produced Water Degassing" and total_gas > 0 and pressure_drop_bar > 0:
                                gwmf_total = (total_gas * 1000) / water_flow_m3hr / pressure_drop_bar
                                gwmf_co2 = (total_co2 * 1000) / water_flow_m3hr / pressure_drop_bar
                                gwmf_ch4 = (total_ch4 * 1000) / water_flow_m3hr / pressure_drop_bar
                                gwmf_nmvoc = (total_nmvoc * 1000) / water_flow_m3hr / pressure_drop_bar
                                
                                # CO2 equivalent GWMF: CO2×1 + CH4×GWP + nmVOC×GWP
                                gwmf_co2eq = gwmf_co2 * 1.0 + gwmf_ch4 * gwp_ch4 + gwmf_nmvoc * gwp_nmvoc
                                # Conventional CO2eq: CH4(14)×GWP + nmVOC(3.5)×GWP (no CO2 measured)
                                conv_gwmf_co2eq = 14.0 * gwp_ch4 + 3.5 * gwp_nmvoc
                                
                                st.markdown(f"""
                                **Gas-to-Water Mass Factors (GWMF):**
                                | Component | NeqSim | Conventional |
                                |-----------|--------|---------------|
                                | CO₂ | {gwmf_co2:.1f} g/m³/bar | Not reported |
                                | CH₄ | {gwmf_ch4:.1f} g/m³/bar | 14 g/m³/bar |
                                | nmVOC | {gwmf_nmvoc:.1f} g/m³/bar | 3.5 g/m³/bar |
                                | **Total** | **{gwmf_total:.1f} g/m³/bar** | ~17.5 g/m³/bar |
                                | **CO₂eq** | **{gwmf_co2eq:.0f} g CO₂eq/m³/bar** | ~{conv_gwmf_co2eq:.0f} g CO₂eq/m³/bar |
                                
                                *Note: GWMF depends strongly on separator gas composition. CO₂ has higher intrinsic solubility 
                                than CH₄ (per unit partial pressure), but actual dissolved amounts are proportional to gas 
                                composition. With typical 3-12% CO₂ gas, expect GWMF(CO₂) ≈ 1-5 g/m³/bar.*
                                """)
                            
                            if ch4_diff < 0:
                                st.success(f"""
                                **Key Finding:** Thermodynamic method calculates **{-ch4_diff:.0f}% lower CH₄** emissions than conventional factors.
                                
                                The conventional method overestimates hydrocarbon emissions because:
                                1. Fixed factors do not account for actual fluid composition
                                2. Temperature and pressure effects on gas solubility are not considered
                                
                                **Important:** The thermodynamic method also quantifies {neqsim_co2_t:.1f} t/yr of CO₂ emissions 
                                that are not captured by the conventional handbook method.
                                """)
                            else:
                                st.info(f"""
                                **Note:** The conventional method does not account for CO₂ emissions.
                                The thermodynamic method quantifies {neqsim_co2_t:.1f} t/yr of CO₂ that would otherwise be unreported.
                                """)
                    
                    with result_tab4:
                        st.subheader("Export Results")
                        
                        # Prepare export data
                        export_summary = pd.DataFrame({
                            'Parameter': ['Total Gas (kg/hr)', 'CO₂ (kg/hr)', 'CH₄ (kg/hr)', 'nmVOC (kg/hr)', 
                                         'CO₂eq (kg/hr)', 'CO₂ (t/yr)', 'CH₄ (t/yr)', 'nmVOC (t/yr)', 'CO₂eq (t/yr)'],
                            'Value': [total_gas, total_co2, total_ch4, total_nmvoc, co2eq_hr,
                                     total_co2*8760/1000, total_ch4*8760/1000, total_nmvoc*8760/1000, co2eq_year]
                        })
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            csv_summary = export_summary.to_csv(index=False)
                            st.download_button(
                                "📄 Download Summary (CSV)",
                                csv_summary,
                                "emission_summary.csv",
                                "text/csv",
                                use_container_width=True
                            )
                        
                        with col2:
                            csv_stages = results_df.to_csv(index=False)
                            st.download_button(
                                "📄 Download Stage Details (CSV)",
                                csv_stages,
                                "emission_stages.csv",
                                "text/csv",
                                use_container_width=True
                            )
                        
                        st.markdown("### Calculation Parameters")
                        flow_display = f"{water_flow_m3hr} m³/hr" if emission_source == "Produced Water Degassing" else f"{total_flow} kg/hr"
                        method_display = calc_method if emission_source == "Produced Water Degassing" else "Direct Composition"
                        # Show which EOS was used
                        if emission_source == "Produced Water Degassing" and salinity_ppm > 0:
                            eos_display = "Electrolyte-CPA (salinity effects)"
                        else:
                            eos_display = "CPA (Cubic-Plus-Association)"
                        params_df = pd.DataFrame({
                            'Parameter': ['Emission Source', 'Calculation Method', 'Inlet Temperature', 'Inlet Pressure', 
                                         'Flow Rate', 'GWP Standard', 'CH₄ GWP', 'nmVOC GWP', 'Salinity', 'Thermodynamic Model'],
                            'Value': [emission_source, method_display, f"{inlet_temp} °C", f"{inlet_pressure} bara",
                                     flow_display, gwp_standard, str(gwp_ch4), str(gwp_nmvoc), f"{salinity_ppm} ppm", eos_display]
                        })
                        st.dataframe(params_df, use_container_width=True)
                        
                        st.markdown("### Audit Trail")
                        st.caption("""
                        This calculation uses the NeqSim thermodynamic library with the CPA equation of state.
                        Method validated against Gudrun platform field data with ±3.6% accuracy.
                        For regulatory reporting, retain input compositions and process conditions.
                        """)
                
                except Exception as e:
                    st.error(f"Calculation error: {str(e)}")
                    st.info("Verify fluid composition and process conditions are within valid ranges.")

# ===================== TAB 2: WHAT-IF ANALYSIS =====================
with main_tab2:
    st.subheader("🔬 What-If Scenario Analysis")
    st.markdown("Evaluate how operational parameter changes affect emission rates.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Parameter to Vary")
        vary_param = st.selectbox(
            "Select Parameter",
            ["Separator Temperature", "Separator Pressure", "Flow Rate"],
            key="whatif_param"
        )
        
        if vary_param == "Separator Temperature":
            base_value = inlet_temp
            min_val = st.number_input("Min Temperature (°C)", value=base_value - 20)
            max_val = st.number_input("Max Temperature (°C)", value=base_value + 20)
            unit = "°C"
        elif vary_param == "Separator Pressure":
            base_value = inlet_pressure
            min_val = st.number_input("Min Pressure (bara)", value=max(5.0, base_value - 15))
            max_val = st.number_input("Max Pressure (bara)", value=base_value + 15)
            unit = "bara"
        else:
            base_value = total_flow
            min_val = st.number_input("Min Flow (kg/hr)", value=base_value * 0.5)
            max_val = st.number_input("Max Flow (kg/hr)", value=base_value * 1.5)
            unit = "kg/hr"
        
        num_points = st.slider("Number of Points", 5, 20, 10)
        
        run_whatif = st.button("▶️ Run What-If Analysis", use_container_width=True)
    
    with col2:
        if run_whatif:
            with st.spinner("Running scenarios..."):
                param_values = np.linspace(min_val, max_val, num_points)
                scenario_results = []
                
                # Get current composition based on method
                if emission_source == "TEG Regeneration":
                    if calc_method == "Absorber Equilibrium":
                        comp_df = st.session_state.teg_absorber_gas_df
                    else:
                        comp_df = st.session_state.teg_fluid_df
                elif emission_source == "Produced Water Degassing" and calc_method == "Separator Equilibrium":
                    comp_df = st.session_state.separator_gas_df
                else:
                    comp_df = st.session_state.emission_gas_df
                
                # Determine if we need electrolyte CPA (salinity > 0)
                use_electrolyte = emission_source == "Produced Water Degassing" and salinity_ppm > 0
                
                for val in param_values:
                    try:
                        # Create CPA fluid (mixing rule set after adding components)
                        scenario_fluid = create_cpa_fluid(use_electrolyte=False)  # Process fluid doesn't need ions
                        
                        # Build fluid with water + dissolved gas
                        if emission_source == "Produced Water Degassing":
                            if calc_method == "Separator Equilibrium":
                                # Iterative equilibrium: double water until two phases exist
                                gas_scale = 100.0
                                # Start with more water for electrolyte CPA
                                water_moles = 1000.0 if use_electrolyte else 100.0
                                
                                for _ in range(10):  # Max 10 iterations
                                    equilibrium_fluid = create_cpa_fluid(use_electrolyte=use_electrolyte)
                                    for _, row in comp_df.iterrows():
                                        if row['MolarComposition[-]'] > 0 and row['ComponentName'] != 'water':
                                            equilibrium_fluid.addComponent(row['ComponentName'], row['MolarComposition[-]'] * gas_scale)
                                    equilibrium_fluid.addComponent('water', water_moles)
                                    
                                    # Add ions if salinity > 0
                                    if use_electrolyte:
                                        kg_water = water_moles * 18.015 / 1000.0
                                        ion_moles = molality * kg_water
                                        equilibrium_fluid.addComponent('Na+', ion_moles)
                                        equilibrium_fluid.addComponent('Cl-', ion_moles)
                                    
                                    equilibrium_fluid.createDatabase(True)
                                    equilibrium_fluid.setMixingRule(10)
                                    equilibrium_fluid.setMultiPhaseCheck(True)
                                    
                                    # Set conditions based on which parameter is being varied
                                    if vary_param == "Separator Temperature":
                                        equilibrium_fluid.setTemperature(val, 'C')
                                        equilibrium_fluid.setPressure(inlet_pressure, 'bara')
                                    elif vary_param == "Separator Pressure":
                                        equilibrium_fluid.setTemperature(inlet_temp, 'C')
                                        equilibrium_fluid.setPressure(val, 'bara')
                                    else:
                                        equilibrium_fluid.setTemperature(inlet_temp, 'C')
                                        equilibrium_fluid.setPressure(inlet_pressure, 'bara')
                                    
                                    TPflash(equilibrium_fluid)
                                    equilibrium_fluid.initProperties()
                                    
                                    if equilibrium_fluid.hasPhaseType('gas') and equilibrium_fluid.hasPhaseType('aqueous'):
                                        break
                                    water_moles *= 2.0
                                
                                # Extract aqueous phase composition (exclude ions)
                                if equilibrium_fluid.hasPhaseType('aqueous'):
                                    aqueous_phase = equilibrium_fluid.getPhase('aqueous')
                                    for idx in range(aqueous_phase.getNumberOfComponents()):
                                        comp = aqueous_phase.getComponent(idx)
                                        comp_name = str(comp.getComponentName())  # Convert Java String to Python
                                        x_i = float(comp.getx())
                                        if x_i > 1e-10 and comp_name not in ['Na+', 'Cl-', 'K+', 'Ca++', 'Mg++', 'Ba++', 'SO4--', 'CO3--']:
                                            scenario_fluid.addComponent(comp_name, x_i)
                                else:
                                    continue  # Skip if no aqueous phase
                            else:
                                # Use lab sample method
                                scenario_fluid.addComponent('water', 0.999)
                                gas_scale = 0.001  # Realistic dissolved gas content
                                for _, row in comp_df.iterrows():
                                    if row['MolarComposition[-]'] > 0 and row['ComponentName'] != 'water':
                                        scenario_fluid.addComponent(row['ComponentName'], row['MolarComposition[-]'] * gas_scale)
                        elif emission_source == "TEG Regeneration" and calc_method == "Absorber Equilibrium":
                            # TEG Absorber Equilibrium: create gas-TEG system and extract liquid phase
                            gas_scale = 100.0
                            teg_moles = 10.0
                            water_in_teg = teg_moles * (1.0 - teg_purity/100.0) / (teg_purity/100.0)
                            
                            for _ in range(10):
                                equilibrium_fluid = create_cpa_fluid(use_electrolyte=False)
                                for _, row in comp_df.iterrows():
                                    if row['MolarComposition[-]'] > 0 and row['ComponentName'] not in ['TEG', 'water']:
                                        equilibrium_fluid.addComponent(row['ComponentName'], row['MolarComposition[-]'] * gas_scale)
                                equilibrium_fluid.addComponent('TEG', teg_moles)
                                if water_in_teg > 0.001:
                                    equilibrium_fluid.addComponent('water', water_in_teg)
                                
                                equilibrium_fluid.createDatabase(True)
                                equilibrium_fluid.setMixingRule(10)
                                equilibrium_fluid.setMultiPhaseCheck(True)
                                
                                if vary_param == "Separator Temperature":
                                    equilibrium_fluid.setTemperature(val, 'C')
                                    equilibrium_fluid.setPressure(inlet_pressure, 'bara')
                                elif vary_param == "Separator Pressure":
                                    equilibrium_fluid.setTemperature(inlet_temp, 'C')
                                    equilibrium_fluid.setPressure(val, 'bara')
                                else:
                                    equilibrium_fluid.setTemperature(inlet_temp, 'C')
                                    equilibrium_fluid.setPressure(inlet_pressure, 'bara')
                                
                                TPflash(equilibrium_fluid)
                                equilibrium_fluid.initProperties()
                                
                                # TEG-rich phase appears as 'aqueous' in CPA-Statoil
                                if equilibrium_fluid.hasPhaseType('gas') and equilibrium_fluid.hasPhaseType('aqueous'):
                                    break
                                teg_moles *= 2.0
                                water_in_teg = teg_moles * (1.0 - teg_purity/100.0) / (teg_purity/100.0)
                            
                            # Extract aqueous/TEG phase composition
                            if equilibrium_fluid.hasPhaseType('aqueous'):
                                teg_liquid_phase = equilibrium_fluid.getPhase('aqueous')
                                for idx in range(teg_liquid_phase.getNumberOfComponents()):
                                    comp = teg_liquid_phase.getComponent(idx)
                                    comp_name = str(comp.getComponentName())
                                    x_i = float(comp.getx())
                                    if x_i > 1e-12:
                                        scenario_fluid.addComponent(comp_name, x_i)
                            else:
                                continue  # Skip if no liquid phase
                        else:
                            for _, row in comp_df.iterrows():
                                if row['MolarComposition[-]'] > 0:
                                    scenario_fluid.addComponent(row['ComponentName'], row['MolarComposition[-]'])
                        
                        # Set mixing rule AFTER adding all components
                        scenario_fluid.createDatabase(True)  # Load binary interaction parameters
                        scenario_fluid.setMixingRule(10)
                        
                        if vary_param == "Separator Temperature":
                            scenario_fluid.setTemperature(val, 'C')
                            scenario_fluid.setPressure(inlet_pressure, 'bara')
                        elif vary_param == "Separator Pressure":
                            scenario_fluid.setTemperature(inlet_temp, 'C')
                            scenario_fluid.setPressure(val, 'bara')
                        else:
                            scenario_fluid.setTemperature(inlet_temp, 'C')
                            scenario_fluid.setPressure(inlet_pressure, 'bara')
                        
                        flow = val if vary_param == "Flow Rate" else total_flow
                        scenario_fluid.setTotalFlowRate(flow, 'kg/hr')
                        
                        TPflash(scenario_fluid)
                        scenario_fluid.initProperties()
                        
                        # Flash to first degassing stage pressure (user-configured)
                        degasser_pressure = stage_pressures[0] if stage_pressures else 4.0
                        scenario_fluid.setPressure(degasser_pressure, 'bara')
                        TPflash(scenario_fluid)
                        scenario_fluid.initProperties()
                        
                        co2_flow = 0
                        ch4_flow = 0
                        
                        if scenario_fluid.hasPhaseType('gas'):
                            gas = scenario_fluid.getPhase('gas')
                            try:
                                co2_flow = gas.getComponent('CO2').getFlowRate('kg/hr')
                            except:
                                pass
                            try:
                                ch4_flow = gas.getComponent('methane').getFlowRate('kg/hr')
                            except:
                                pass
                        
                        co2eq = co2_flow + ch4_flow * gwp_ch4
                        
                        scenario_results.append({
                            'Parameter': val,
                            'CO2 (kg/hr)': co2_flow,
                            'CH4 (kg/hr)': ch4_flow,
                            'CO2eq (kg/hr)': co2eq
                        })
                    except:
                        pass
                
                if scenario_results:
                    scenario_df = pd.DataFrame(scenario_results)
                    
                    fig_whatif = make_subplots(rows=1, cols=2, subplot_titles=("Emission Rates", "CO₂ Equivalent"))
                    
                    fig_whatif.add_trace(
                        go.Scatter(x=scenario_df['Parameter'], y=scenario_df['CO2 (kg/hr)'], name='CO₂', line=dict(color='#2196F3')),
                        row=1, col=1
                    )
                    fig_whatif.add_trace(
                        go.Scatter(x=scenario_df['Parameter'], y=scenario_df['CH4 (kg/hr)'], name='CH₄', line=dict(color='#FF9800')),
                        row=1, col=1
                    )
                    fig_whatif.add_trace(
                        go.Scatter(x=scenario_df['Parameter'], y=scenario_df['CO2eq (kg/hr)'], name='CO₂eq', line=dict(color='#4CAF50', width=3)),
                        row=1, col=2
                    )
                    
                    fig_whatif.update_xaxes(title_text=f"{vary_param} ({unit})", row=1, col=1)
                    fig_whatif.update_xaxes(title_text=f"{vary_param} ({unit})", row=1, col=2)
                    fig_whatif.update_yaxes(title_text="Emission Rate (kg/hr)", row=1, col=1)
                    fig_whatif.update_yaxes(title_text="CO₂eq (kg/hr)", row=1, col=2)
                    
                    fig_whatif.update_layout(height=400, showlegend=True)
                    st.plotly_chart(fig_whatif, use_container_width=True)
                    
                    st.dataframe(scenario_df, use_container_width=True)

# ===================== TAB 3: UNCERTAINTY ANALYSIS =====================
with main_tab3:
    st.subheader("📈 Monte Carlo Uncertainty Analysis")
    st.markdown("""
    Quantify emission uncertainty by propagating measurement uncertainties through the thermodynamic model.
    
    | Parameter | Default Uncertainty | Basis |
    |-----------|---------------------|-------|
    | Temperature | ±2°C | Typical instrument accuracy |
    | Pressure | ±0.1 bar | Pressure transmitter specification |
    | Flow rate | ±3% | Flow meter uncertainty |
    | Composition | ±5% relative | Laboratory analysis repeatability |
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Uncertainty Settings")
        temp_uncertainty = st.number_input("Temperature uncertainty (°C)", value=2.0, min_value=0.1, max_value=10.0)
        press_uncertainty = st.number_input("Pressure uncertainty (bar)", value=0.1, min_value=0.01, max_value=1.0)
        flow_uncertainty = st.number_input("Flow uncertainty (%)", value=3.0, min_value=0.1, max_value=10.0)
        comp_uncertainty = st.number_input("Composition uncertainty (%)", value=5.0, min_value=0.1, max_value=20.0)
        
        n_samples = st.slider("Number of Monte Carlo samples", 100, 2000, 500)
        
        run_mc = st.button("▶️ Run Uncertainty Analysis", use_container_width=True)
    
    with col2:
        if run_mc:
            with st.spinner(f"Running {n_samples} Monte Carlo simulations..."):
                mc_results = []
                
                # Get base values
                base_temp = inlet_temp
                base_press = inlet_pressure
                base_flow = total_flow
                
                # Get composition based on method
                if emission_source == "TEG Regeneration":
                    if calc_method == "Absorber Equilibrium":
                        comp_df = st.session_state.teg_absorber_gas_df
                    else:
                        comp_df = st.session_state.teg_fluid_df
                elif emission_source == "Produced Water Degassing" and calc_method == "Separator Equilibrium":
                    comp_df = st.session_state.separator_gas_df
                else:
                    comp_df = st.session_state.emission_gas_df
                
                # Determine if we need electrolyte CPA (salinity > 0)
                use_electrolyte = emission_source == "Produced Water Degassing" and salinity_ppm > 0
                
                progress_bar = st.progress(0)
                
                for i in range(n_samples):
                    try:
                        # Perturb parameters
                        temp_pert = np.random.normal(base_temp, temp_uncertainty)
                        press_pert = np.random.normal(base_press, press_uncertainty)
                        flow_pert = np.random.normal(base_flow, base_flow * flow_uncertainty / 100)
                        
                        # Create CPA fluid (mixing rule set after adding components)
                        mc_fluid = create_cpa_fluid(use_electrolyte=False)  # Process fluid doesn't need ions
                        
                        # Build fluid with water + dissolved gas for produced water
                        if emission_source == "Produced Water Degassing":
                            if calc_method == "Separator Equilibrium":
                                # Iterative equilibrium: double water until two phases exist
                                gas_scale = 100.0
                                # Start with more water for electrolyte CPA
                                water_moles = 1000.0 if use_electrolyte else 100.0
                                
                                for _ in range(10):  # Max 10 iterations
                                    equilibrium_fluid = create_cpa_fluid(use_electrolyte=use_electrolyte)
                                    for _, row in comp_df.iterrows():
                                        if row['MolarComposition[-]'] > 0 and row['ComponentName'] != 'water':
                                            pert_comp = row['MolarComposition[-]'] * gas_scale * np.random.normal(1.0, comp_uncertainty/100)
                                            equilibrium_fluid.addComponent(row['ComponentName'], max(0.001, pert_comp))
                                    water_pert = water_moles * np.random.normal(1.0, comp_uncertainty/100)
                                    equilibrium_fluid.addComponent('water', water_pert)
                                    
                                    # Add ions if salinity > 0
                                    if use_electrolyte:
                                        kg_water = water_pert * 18.015 / 1000.0
                                        ion_moles = molality * kg_water
                                        equilibrium_fluid.addComponent('Na+', ion_moles)
                                        equilibrium_fluid.addComponent('Cl-', ion_moles)
                                    
                                    equilibrium_fluid.createDatabase(True)
                                    equilibrium_fluid.setMixingRule(10)
                                    equilibrium_fluid.setMultiPhaseCheck(True)
                                    
                                    equilibrium_fluid.setTemperature(temp_pert, 'C')
                                    equilibrium_fluid.setPressure(press_pert, 'bara')
                                    
                                    TPflash(equilibrium_fluid)
                                    equilibrium_fluid.initProperties()
                                    
                                    if equilibrium_fluid.hasPhaseType('gas') and equilibrium_fluid.hasPhaseType('aqueous'):
                                        break
                                    water_moles *= 2.0
                                
                                # Extract aqueous phase composition (exclude ions)
                                if equilibrium_fluid.hasPhaseType('aqueous'):
                                    aqueous_phase = equilibrium_fluid.getPhase('aqueous')
                                    for j in range(aqueous_phase.getNumberOfComponents()):
                                        comp = aqueous_phase.getComponent(j)
                                        comp_name = str(comp.getComponentName())  # Convert Java String to Python
                                        x_i = float(comp.getx())
                                        if x_i > 1e-10 and comp_name not in ['Na+', 'Cl-', 'K+', 'Ca++', 'Mg++', 'Ba++', 'SO4--', 'CO3--']:
                                            mc_fluid.addComponent(comp_name, x_i)
                                else:
                                    continue  # Skip if no aqueous phase
                            else:
                                # Lab sample method
                                mc_fluid.addComponent('water', 0.999 * np.random.normal(1.0, comp_uncertainty/100))
                                gas_scale = 0.001  # Realistic dissolved gas content
                                for _, row in comp_df.iterrows():
                                    if row['MolarComposition[-]'] > 0 and row['ComponentName'] != 'water':
                                        pert_comp = row['MolarComposition[-]'] * gas_scale * np.random.normal(1.0, comp_uncertainty/100)
                                        mc_fluid.addComponent(row['ComponentName'], max(0.00001, pert_comp))
                        elif emission_source == "TEG Regeneration" and calc_method == "Absorber Equilibrium":
                            # TEG Absorber Equilibrium: create gas-TEG system and extract liquid phase
                            gas_scale = 100.0
                            teg_moles = 10.0
                            water_in_teg = teg_moles * (1.0 - teg_purity/100.0) / (teg_purity/100.0)
                            
                            for _ in range(10):
                                equilibrium_fluid = create_cpa_fluid(use_electrolyte=False)
                                for _, row in comp_df.iterrows():
                                    if row['MolarComposition[-]'] > 0 and row['ComponentName'] not in ['TEG', 'water']:
                                        pert_comp = row['MolarComposition[-]'] * gas_scale * np.random.normal(1.0, comp_uncertainty/100)
                                        equilibrium_fluid.addComponent(row['ComponentName'], max(0.001, pert_comp))
                                equilibrium_fluid.addComponent('TEG', teg_moles)
                                if water_in_teg > 0.001:
                                    equilibrium_fluid.addComponent('water', water_in_teg)
                                
                                equilibrium_fluid.createDatabase(True)
                                equilibrium_fluid.setMixingRule(10)
                                equilibrium_fluid.setMultiPhaseCheck(True)
                                
                                equilibrium_fluid.setTemperature(temp_pert, 'C')
                                equilibrium_fluid.setPressure(press_pert, 'bara')
                                
                                TPflash(equilibrium_fluid)
                                equilibrium_fluid.initProperties()
                                
                                # TEG-rich phase appears as 'aqueous' in CPA-Statoil
                                if equilibrium_fluid.hasPhaseType('gas') and equilibrium_fluid.hasPhaseType('aqueous'):
                                    break
                                teg_moles *= 2.0
                                water_in_teg = teg_moles * (1.0 - teg_purity/100.0) / (teg_purity/100.0)
                            
                            # Extract aqueous/TEG phase composition
                            if equilibrium_fluid.hasPhaseType('aqueous'):
                                teg_liquid_phase = equilibrium_fluid.getPhase('aqueous')
                                for j in range(teg_liquid_phase.getNumberOfComponents()):
                                    comp = teg_liquid_phase.getComponent(j)
                                    comp_name = str(comp.getComponentName())
                                    x_i = float(comp.getx())
                                    if x_i > 1e-12:
                                        mc_fluid.addComponent(comp_name, x_i)
                            else:
                                continue  # Skip if no liquid phase
                        else:
                            for _, row in comp_df.iterrows():
                                if row['MolarComposition[-]'] > 0:
                                    pert_comp = row['MolarComposition[-]'] * np.random.normal(1.0, comp_uncertainty/100)
                                    mc_fluid.addComponent(row['ComponentName'], max(0.001, pert_comp))
                        
                        # Set mixing rule AFTER adding all components
                        mc_fluid.createDatabase(True)  # Load binary interaction parameters
                        mc_fluid.setMixingRule(10)
                        
                        mc_fluid.setTemperature(temp_pert, 'C')
                        mc_fluid.setPressure(press_pert, 'bara')
                        mc_fluid.setTotalFlowRate(flow_pert, 'kg/hr')
                        
                        TPflash(mc_fluid)
                        mc_fluid.initProperties()
                        
                        # Flash to first degassing stage pressure (user-configured)
                        degasser_pressure = stage_pressures[0] if stage_pressures else 4.0
                        mc_fluid.setPressure(degasser_pressure, 'bara')
                        TPflash(mc_fluid)
                        mc_fluid.initProperties()
                        
                        co2_flow = 0
                        ch4_flow = 0
                        
                        if mc_fluid.hasPhaseType('gas'):
                            gas = mc_fluid.getPhase('gas')
                            try:
                                co2_flow = gas.getComponent('CO2').getFlowRate('kg/hr')
                            except:
                                pass
                            try:
                                ch4_flow = gas.getComponent('methane').getFlowRate('kg/hr')
                            except:
                                pass
                        
                        co2eq = co2_flow + ch4_flow * gwp_ch4
                        mc_results.append({'CO2': co2_flow, 'CH4': ch4_flow, 'CO2eq': co2eq})
                    except:
                        pass
                    
                    if i % 50 == 0:
                        progress_bar.progress(i / n_samples)
                
                progress_bar.progress(1.0)
                
                if mc_results:
                    mc_df = pd.DataFrame(mc_results)
                    
                    # Statistics
                    stats = {
                        'Statistic': ['Mean', 'Std Dev', '5th Percentile', '95th Percentile', 'Relative Uncertainty (95% CI)'],
                        'CO₂ (kg/hr)': [
                            f"{mc_df['CO2'].mean():.1f}",
                            f"{mc_df['CO2'].std():.1f}",
                            f"{np.percentile(mc_df['CO2'], 5):.1f}",
                            f"{np.percentile(mc_df['CO2'], 95):.1f}",
                            f"±{(np.percentile(mc_df['CO2'], 95) - np.percentile(mc_df['CO2'], 5)) / 2 / mc_df['CO2'].mean() * 100:.1f}%" if mc_df['CO2'].mean() > 0 else "N/A"
                        ],
                        'CH₄ (kg/hr)': [
                            f"{mc_df['CH4'].mean():.1f}",
                            f"{mc_df['CH4'].std():.1f}",
                            f"{np.percentile(mc_df['CH4'], 5):.1f}",
                            f"{np.percentile(mc_df['CH4'], 95):.1f}",
                            f"±{(np.percentile(mc_df['CH4'], 95) - np.percentile(mc_df['CH4'], 5)) / 2 / mc_df['CH4'].mean() * 100:.1f}%" if mc_df['CH4'].mean() > 0 else "N/A"
                        ],
                        'CO₂eq (kg/hr)': [
                            f"{mc_df['CO2eq'].mean():.1f}",
                            f"{mc_df['CO2eq'].std():.1f}",
                            f"{np.percentile(mc_df['CO2eq'], 5):.1f}",
                            f"{np.percentile(mc_df['CO2eq'], 95):.1f}",
                            f"±{(np.percentile(mc_df['CO2eq'], 95) - np.percentile(mc_df['CO2eq'], 5)) / 2 / mc_df['CO2eq'].mean() * 100:.1f}%" if mc_df['CO2eq'].mean() > 0 else "N/A"
                        ]
                    }
                    
                    st.dataframe(pd.DataFrame(stats), use_container_width=True)
                    
                    # Histogram
                    fig_hist = make_subplots(rows=1, cols=3, subplot_titles=("CO₂ Distribution", "CH₄ Distribution", "CO₂eq Distribution"))
                    
                    fig_hist.add_trace(go.Histogram(x=mc_df['CO2'], nbinsx=30, marker_color='#2196F3', name='CO₂'), row=1, col=1)
                    fig_hist.add_trace(go.Histogram(x=mc_df['CH4'], nbinsx=30, marker_color='#FF9800', name='CH₄'), row=1, col=2)
                    fig_hist.add_trace(go.Histogram(x=mc_df['CO2eq'], nbinsx=30, marker_color='#4CAF50', name='CO₂eq'), row=1, col=3)
                    
                    fig_hist.update_layout(height=350, showlegend=False)
                    fig_hist.update_xaxes(title_text="kg/hr")
                    fig_hist.update_yaxes(title_text="Frequency")
                    
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                    st.success(f"Analysis completed: {len(mc_results)} simulations")
