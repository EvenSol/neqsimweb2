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

st.set_page_config(
    page_title="Emission Calculator",
    page_icon='images/neqsimlogocircleflat.png',
    layout="wide"
)

st.title("🌱 Methane & CO₂ Emission Calculator")
st.markdown("""
Quantify greenhouse gas emissions (CO₂, CH₄, nmVOC) from offshore process streams using 
rigorous thermodynamic modeling with the Cubic-Plus-Association (CPA) equation of state.

| Parameter | Value |
|-----------|-------|
| **Input** | Laboratory gas analysis OR first stage separator gas composition |
| **Accuracy** | ±3.6% validated against field data (Gudrun platform) |
| **Method** | Physics-based CPA equation of state with full component accounting |

📚 [NeqSim Emissions & Sustainability Guide](https://equinor.github.io/neqsim/emissions/)
""")

# Emission sources covered
with st.expander("📍 Emission Sources Covered", expanded=False):
    st.markdown("""
    This calculator supports venting emission sources:
    
    | Source | Description | Typical Reduction Potential |
    |--------|-------------|----------------------------|
    | **Produced Water Degassing** | Multi-stage degassing (Degasser, CFU, Caisson) | 15–40% |
    | **TEG Regeneration** | Flash drum off-gas from glycol dehydration | 10–25% |
    | **Tank Breathing/Loading** | Storage tank vapor losses during operations | 20–50% |
    | **Cold Vent Streams** | Pressure relief and maintenance vents | Variable |
    
    Venting emissions represent 5–20% of platform total emissions but offer 
    significant reduction potential through operational optimization.
    """)

with st.expander("🔬 Calculation Methods for Produced Water", expanded=False):
    st.markdown("""
    Two methods are available for produced water degassing calculations:
    
    ### 1. Lab Sample Analysis (Default)
    Uses measured gas composition from flashing a water sample at standard conditions (1 atm, 15°C).
    
    | Aspect | Description |
    |--------|-------------|
    | **Input** | Laboratory gas analysis of flash gas |
    | **Best for** | When actual lab measurements are available |
    | **Assumption** | Fixed dissolved gas content scaled by measured composition ratios |
    
    ### 2. Separator Equilibrium
    Calculates dissolved gas content from thermodynamic equilibrium between first stage separator gas and water.
    
    | Aspect | Description |
    |--------|-------------|
    | **Input** | First stage separator gas composition |
    | **Best for** | Design studies, what-if analysis, when lab data unavailable |
    | **Method** | VLE flash at separator conditions determines actual dissolved gas content |
    
    The **Separator Equilibrium** method is more rigorous as it accounts for:
    - Actual separator temperature and pressure
    - Component-specific solubility (CO₂ is much more soluble than CH₄)
    - Non-ideal behavior via CPA equation of state
    """)

with st.expander("📖 Regulatory Framework & Standards", expanded=False):
    st.markdown("""
    ### Applicable Regulations
    
    | Regulation | Jurisdiction | NeqSim Compliance |
    |------------|--------------|-------------------|
    | **Aktivitetsforskriften §70** | Norway | ✅ Virtual measurement methodology |
    | **EU ETS Directive** | European Union | ✅ CO₂ equivalent reporting |
    | **EU Methane Regulation 2024/1787** | European Union | ✅ Source-level CH₄ quantification |
    | **OGMP 2.0** | International | ✅ Level 4/5 site-specific |
    | **ISO 14064-1:2018** | International | ✅ Organization-level GHG |
    
    ### Norwegian Reporting Requirements
    The Norwegian Petroleum Directorate accepts thermodynamic emission calculations 
    as a **virtual measurement methodology** per Aktivitetsforskriften §70.
    
    **Key requirement:** Demonstrate that the model is validated and provides 
    uncertainty estimates better than conventional handbook methods.
    
    ### References
    - [IOGP Report 521 (2019)](https://www.iogp.org/bookstore/product/methods-for-estimating-atmospheric-emissions-from-e-p-operations/) — E&P emission estimation methods
    - [EU Methane Regulation 2024/1787](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1787) — EU methane requirements
    """)

with st.expander("🧪 Thermodynamic Method & Validation", expanded=False):
    st.markdown("""
    ### Why CPA-SRK-EOS (Statoil) Equation of State?
    
    This calculator uses the **CPA-SRK-EOS-statoil** (Cubic-Plus-Association with Soave-Redlich-Kwong) 
    equation of state, specifically designed for systems containing polar and associating compounds 
    like water, CO₂, and H₂S. This implementation is optimized for North Sea reservoir fluids.
    
    | Property | SRK/PR (Cubic) | CPA-SRK (Statoil) |
    |----------|----------------|-------------------|
    | Water-hydrocarbon VLE | Poor | ✅ Excellent |
    | CO₂ solubility in water | Moderate | ✅ Accurate |
    | H₂S partitioning | Poor | ✅ Good |
    | Computation speed | Fast | Fast |
    
    ### Validation Results
    
    The method was validated against field data from the **Gudrun platform** (Norwegian Continental Shelf):
    
    | Metric | NeqSim | Handbook Method |
    |--------|--------|-----------------|
    | **Accuracy** | ±3.6% | ±50% or worse |
    | **CO₂ capture** | ✅ Yes (50-80% of total) | ❌ Not measured |
    | **Bias** | Low | High (conservative) |
    
    ### Key Publications
    - Kontogeorgis & Folas (2010) — *Thermodynamic Models for Industrial Applications* 
      [DOI: 10.1002/9780470747537](https://doi.org/10.1002/9780470747537)
    - GFMW 2023 — NeqSim validation paper (internal Equinor)
    """)

with st.expander("📚 Glossary of Terms", expanded=False):
    st.markdown("""
    | Term | Definition |
    |------|------------|
    | **GWP** | Global Warming Potential — relative climate impact compared to CO₂ over 100 years |
    | **GWP-100** | GWP evaluated over a 100-year time horizon |
    | **CO₂eq** | CO₂ equivalent — total emissions expressed as equivalent CO₂ impact |
    | **nmVOC** | Non-methane Volatile Organic Compounds — C₂+ hydrocarbons |
    | **CPA** | Cubic-Plus-Association equation of state |
    | **VLE** | Vapor-Liquid Equilibrium |
    | **Setschenow coefficient** | Parameter describing how salt reduces gas solubility |
    | **Salting-out** | Reduced gas solubility in saline water vs freshwater |
    | **GWMF** | Gas-to-Water Mass Factor — emission factor in g/(m³·bar) |
    | **CFU** | Compact Flotation Unit — degassing/water treatment equipment |
    | **Caisson** | Final water discharge point (atmospheric pressure) |
    | **TEG** | Triethylene Glycol — used for gas dehydration |
    """)

with st.expander("🔗 Additional Resources", expanded=False):
    st.markdown("""
    ### Interactive Tutorials
    - [Produced Water Emissions Tutorial](https://equinor.github.io/neqsim/examples/ProducedWaterEmissions_Tutorial.html) — Jupyter notebook
    - [Norwegian Methods Comparison](https://equinor.github.io/neqsim/examples/NorwegianEmissionMethods_Comparison.html) — Validation
    - [Run in Google Colab](https://colab.research.google.com/github/equinor/neqsim/blob/master/docs/examples/ProducedWaterEmissions_Tutorial.ipynb) — No installation needed
    
    ### Technical Documentation
    - [Offshore Emission Reporting Guide](https://equinor.github.io/neqsim/emissions/OFFSHORE_EMISSION_REPORTING.html) — Comprehensive reference
    - [API Reference (Chapter 43)](https://equinor.github.io/neqsim/REFERENCE_MANUAL_INDEX.html#chapter-43-sustainability--emissions) — EmissionsCalculator class
    - [Java Example](https://github.com/equinor/neqsim/blob/master/docs/examples/OffshoreEmissionReportingExample.java) — Code sample
    
    ### Support
    - [GitHub Issues](https://github.com/equinor/neqsim/issues) — Bug reports and feature requests
    - [GitHub Discussions](https://github.com/equinor/neqsim/discussions) — Q&A forum
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
        'ComponentName': ['TEG', 'water', 'methane', 'CO2', 'benzene'],
        'MolarComposition[-]': [0.90, 0.05, 0.02, 0.01, 0.02]
    })

# TEG absorber inlet gas composition (for equilibrium method)
if 'teg_absorber_gas_df' not in st.session_state:
    st.session_state.teg_absorber_gas_df = pd.DataFrame({
        'ComponentName': ['methane', 'ethane', 'propane', 'CO2', 'nitrogen', 'benzene', 'toluene'],
        'MolarComposition[-]': [0.85, 0.06, 0.03, 0.02, 0.01, 0.015, 0.015]
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
        salinity_ppm = st.number_input(
            "Salinity (ppm TDS)", 
            value=0, 
            min_value=0, 
            max_value=300000,
            help="Total dissolved solids. Set to 0 for freshwater. Seawater ~35,000 ppm, produced water can be 100,000+ ppm. Salinity reduces gas solubility (salting-out effect)."
        )
        
        # Calculate salting-out factor based on Setschenow equation
        # Higher salinity = lower gas solubility = potentially lower emissions (less dissolved gas)
        if salinity_ppm > 0:
            cs = 0.12  # Setschenow coefficient for CH4 in NaCl (typical value)
            molality = salinity_ppm / 58440 / (1 - salinity_ppm/1e6)  # Convert ppm to molality
            salting_out_factor = 10 ** (-cs * molality)
            st.info(f"Salting-out factor: {salting_out_factor:.3f} — Gas solubility reduced to {salting_out_factor*100:.1f}% of freshwater value")
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
        help="Global Warming Potential standard. AR6 is the latest IPCC assessment (2021). Most regulations still reference AR5."
    )
    
    if gwp_standard == "IPCC AR5 (2014)":
        gwp_ch4 = 28.0
        gwp_n2o = 265.0
    else:
        gwp_ch4 = 29.8
        gwp_n2o = 273.0
    
    gwp_nmvoc = st.number_input(
        "nmVOC GWP-100", 
        value=2.2, 
        min_value=0.1, 
        max_value=10.0,
        help="GWP for non-methane VOCs (C₂+ hydrocarbons). Default 2.2 is an average for light hydrocarbons. Range: ethane ~1.4, propane ~2.3, butanes ~2.5"
    )
    
    st.caption(f"CH₄ GWP: {gwp_ch4} | N₂O GWP: {gwp_n2o}")
    
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
        
        # Show method description
        if emission_source == "Produced Water Degassing":
            with st.expander("📖 Method Description", expanded=False):
                if calc_method == "Lab Sample Analysis":
                    st.markdown("""
                    **Lab Sample Analysis Method:**
                    1. A water sample is collected from the process
                    2. The sample is flashed to standard conditions (1 atm, 15°C) in the laboratory
                    3. The released gas composition is analyzed (this is the input above)
                    4. The gas composition ratios are used to estimate dissolved gas in process water
                    5. Multi-stage degassing is simulated to calculate total emissions
                    
                    **Advantage:** Based on actual measured data from your facility
                    **Limitation:** Lab conditions differ from process conditions; assumes fixed dissolved gas content
                    """)
                else:
                    st.markdown("""
                    **Separator Equilibrium Method:**
                    1. Enter the gas composition from the first stage separator
                    2. NeqSim calculates thermodynamic equilibrium between gas and water at separator T & P
                    3. The CPA equation of state determines how much of each component dissolves in water
                    4. The saturated water is then flashed through the degassing stages
                    5. Released gas at each stage is calculated using rigorous VLE
                    
                    **Advantage:** Physics-based; accounts for actual separator conditions (T, P, composition)
                    **Limitation:** Assumes equilibrium is reached in separator (good for most cases)
                    """)
    
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
                            
                            # Determine if we need electrolyte CPA (salinity > 0)
                            use_electrolyte = salinity_ppm > 0
                            if use_electrolyte:
                                st.info("Using Separator Equilibrium method with **Electrolyte-CPA-EoS** (salinity effects enabled)...")
                            else:
                                st.info("Using Separator Equilibrium method: calculating dissolved gas from VLE...")
                            
                            # Filter valid components (non-zero, non-water)
                            valid_gas_df = edited_df[
                                (edited_df['MolarComposition[-]'] > 0) & 
                                (edited_df['ComponentName'] != 'water')
                            ].copy()
                            
                            if len(valid_gas_df) == 0:
                                st.error("No valid gas components found. Please enter at least one gas component with non-zero mole fraction.")
                                st.stop()
                            
                            # Create gas-water equilibrium system
                            # Use iterative approach: double water until two phases exist
                            # This ensures gas composition matches input while having proper aqueous phase
                            gas_scale = 100.0  # Large gas excess to preserve composition
                            # Start with more water for electrolyte CPA (needs proper water/ion ratio for stability)
                            water_moles = 1000.0 if use_electrolyte else 100.0
                            max_iterations = 10
                            
                            for iteration in range(max_iterations):
                                equilibrium_fluid = create_cpa_fluid(use_electrolyte=use_electrolyte)
                                
                                # Add gas components (scaled to preserve composition at equilibrium)
                                components_added = []
                                for _, row in valid_gas_df.iterrows():
                                    equilibrium_fluid.addComponent(row['ComponentName'], row['MolarComposition[-]'] * gas_scale)
                                    components_added.append(row['ComponentName'])
                                
                                # Add water
                                equilibrium_fluid.addComponent('water', water_moles)
                                
                                # Add ions if salinity > 0 (for electrolyte CPA)
                                if use_electrolyte:
                                    # Calculate Na+ and Cl- moles based on salinity and water amount
                                    # molality (mol/kg) was already calculated from ppm
                                    # water_moles * 18.015 g/mol / 1000 = kg of water
                                    kg_water = water_moles * 18.015 / 1000.0
                                    ion_moles = molality * kg_water
                                    equilibrium_fluid.addComponent('Na+', ion_moles)
                                    equilibrium_fluid.addComponent('Cl-', ion_moles)
                                
                                # Configure CPA model
                                equilibrium_fluid.createDatabase(True)
                                equilibrium_fluid.setMixingRule(10)
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
                            
                            # Create a new CPA fluid representing just the produced water
                            # For process calculations (degassing), we don't need electrolyte CPA
                            # since we're flashing the water phase (ions don't flash)
                            process_fluid = create_cpa_fluid(use_electrolyte=False)
                            
                            # Get composition of aqueous phase (exclude ions for process fluid)
                            dissolved_gas_info = []
                            components_in_water = 0
                            for idx in range(aqueous_phase.getNumberOfComponents()):
                                comp = aqueous_phase.getComponent(idx)
                                comp_name = str(comp.getComponentName())  # Convert Java String to Python
                                x_i = float(comp.getx())  # mole fraction in aqueous phase
                                # Skip ions for process fluid (they don't flash)
                                if x_i > 1e-10 and comp_name not in ['Na+', 'Cl-', 'K+', 'Ca++', 'Mg++', 'Ba++', 'SO4--', 'CO3--']:
                                    process_fluid.addComponent(comp_name, x_i)
                                    components_in_water += 1
                                    if comp_name != 'water':
                                        dissolved_gas_info.append({
                                            'Component': comp_name,
                                            'Mole Fraction': x_i,
                                            'ppm (molar)': x_i * 1e6
                                        })
                            
                            # Verify process_fluid has components
                            if components_in_water == 0:
                                st.error("No components extracted from aqueous phase. This may indicate a thermodynamic calculation issue.")
                                st.stop()
                            
                            # Load binary interaction parameters and set mixing rule
                            process_fluid.createDatabase(True)
                            process_fluid.setMixingRule(10)
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
                        # TEG Regeneration: use rich TEG composition directly
                        # Get the correct dataframe based on calculation method
                        if calc_method == "Absorber Equilibrium":
                            # TODO: Calculate rich TEG from absorber equilibrium
                            # For now, use the absorber gas as proxy
                            st.info("Using Absorber Equilibrium method (simplified): Using absorber inlet gas as proxy for rich TEG composition...")
                            teg_df = st.session_state.teg_absorber_gas_df
                        else:
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
                            
                            # Carbon cost estimates
                            co2_tax_nok = 1565
                            co2_tax_eu = 80
                            annual_cost_nok = co2eq_year * co2_tax_nok
                            annual_cost_eu = co2eq_year * co2_tax_eu
                            
                            st.markdown(f"""
                            **Estimated Annual Carbon Costs**
                            | Scheme | Rate | Annual Cost |
                            |--------|------|-------------|
                            | Norwegian CO₂ Tax | NOK 1,565/t | NOK {annual_cost_nok:,.0f} |
                            | EU ETS | ~€80/t | €{annual_cost_eu:,.0f} |
                            """)
                        
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
                            fig_bar.update_layout(title="Emissions by Degassing Stage", xaxis_title="Stage", yaxis_title="Emission Rate (kg/hr)", barmode='group', height=400)
                            st.plotly_chart(fig_bar, use_container_width=True)
                    
                    with result_tab3:
                        st.subheader("NeqSim vs Norwegian Handbook Method")
                        
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

# ===================== INFORMATION EXPANDERS =====================
st.divider()

with st.expander("📚 Regulatory Framework"):
    st.markdown("""
    ### Norwegian Continental Shelf (NCS)
    | Regulation | Scope |
    |------------|-------|
    | **Aktivitetsforskriften §70** | Requirements for measurement and calculation of emissions |
    | **Norwegian CO₂ Tax** | ~NOK 1,565/tonne CO₂ (2024 rate) |
    
    ### European Union
    | Regulation | Scope |
    |------------|-------|
    | **EU ETS Directive 2003/87/EC** | Emissions trading system for installations |
    | **EU Methane Regulation 2024/1787** | Source-level methane emission requirements |
    | **MRV Regulation 2015/757** | Monitoring, Reporting, and Verification requirements |
    
    ### International Standards
    | Standard | Scope |
    |----------|-------|
    | **ISO 14064-1:2018** | Organization-level GHG quantification |
    | **IOGP Report 521** | Methods for estimating fugitive emissions |
    | **OGMP 2.0** | Oil & Gas Methane Partnership reporting framework |
    """)

with st.expander("🔬 Thermodynamic Method"):
    st.markdown("""
    ### CPA Equation of State
    
    NeqSim employs the **Cubic-Plus-Association (CPA)** equation of state, which provides accurate 
    vapor-liquid equilibrium calculations for water-hydrocarbon systems.
    
    **Binary Interaction Parameters (kij):**
    
    | System | kij | Data Source |
    |--------|-----|-------------|
    | Water–CO₂ | -0.112 | High-pressure VLE data |
    | Water–CH₄ | 0.0115 | Solubility measurements |
    | Water–C₂H₆ | 0.48 | Solubility measurements |
    | Water–C₃H₈ | 0.49 | Solubility measurements |
    
    The negative kij for CO₂ reflects attractive water-CO₂ interactions, 
    resulting in higher CO₂ solubility compared to non-polar hydrocarbons.
    
    ### Validation Results
    | Study | Accuracy | Notes |
    |-------|----------|-------|
    | North Sea field (12 months) | ±3.6% | Continuous operation |
    | PVT laboratory | ±2.1% | Controlled conditions |
    | Conventional handbook | ±50% or worse | Fixed factors only |
    
    ### Key Advantage
    The thermodynamic method provides **full component accounting**, including CO₂ emissions 
    (typically 40–80% of dissolved gas) that are not captured by conventional handbook methods.
    """)

with st.expander("📖 References & Documentation"):
    st.markdown("""
    ### NeqSim Documentation
    - [Emissions & Sustainability Guide](https://equinor.github.io/neqsim/emissions/) — Comprehensive reference for emission calculations
    - [Offshore Emission Reporting Guide](https://equinor.github.io/neqsim/emissions/OFFSHORE_EMISSION_REPORTING.html) — Regulatory framework and methodology
    - [Produced Water Emissions Tutorial](https://equinor.github.io/neqsim/examples/ProducedWaterEmissions_Tutorial.html) — Interactive Jupyter notebook
    - [Norwegian Methods Comparison](https://equinor.github.io/neqsim/examples/NorwegianEmissionMethods_Comparison.html) — Validation against handbook method
    
    ### Scientific References
    1. Kontogeorgis, G.M. & Folas, G.K. (2010). *Thermodynamic Models for Industrial Applications*. Wiley. [DOI: 10.1002/9780470747537](https://doi.org/10.1002/9780470747537)
    2. Duan, Z. & Sun, R. (2003). An improved model calculating CO₂ solubility in pure water and aqueous NaCl solutions. *Chemical Geology*, 193(3-4), 257-271.
    3. IPCC (2014, 2021). *Assessment Reports AR5/AR6* — Global Warming Potential values.
    
    ### Industry Guidelines
    - IOGP Report 521 (2019) — Methods for estimating atmospheric emissions from E&P operations
    - Norsk olje og gass — Handbook for quantification of direct emissions (Retningslinje 044)
    - [EU Methane Regulation 2024/1787](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1787)
    
    ### Software Resources
    - [NeqSim GitHub Repository](https://github.com/equinor/neqsim)
    - [NeqSim Python Package](https://github.com/equinor/neqsim-python)
    - [Run in Google Colab](https://colab.research.google.com/github/equinor/neqsim/blob/master/docs/examples/ProducedWaterEmissions_Tutorial.ipynb) (no installation required)
    """)
