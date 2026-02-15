import streamlit as st
import pandas as pd
import neqsim
import time
from neqsim.thermo.thermoTools import fluidcreator, fluid_df, hydt, dataFrame, TPflash
from neqsim.thermo import fluid
from neqsim import jneqsim
from fluids import default_fluid
from fluids import fluid_library_selector
import plotly.graph_objects as go
from theme import apply_theme

st.set_page_config(page_title="Gas Hydrate", page_icon='images/neqsimlogocircleflat.png')
apply_theme()

st.title('Gas Hydrate Calculation')
"""
Gas hydrate calculations are done using the CPA-EoS combined with a model for the solid hydrate phase.
For electrolyte systems (with ions), the Electrolyte-CPA-EoS (Statoil) is used.
"""
st.divider()

# Default gas composition (without water and inhibitors)
gas_only_fluid = {
    'ComponentName': ["nitrogen", "CO2", "H2S", "methane", "ethane", "propane", "i-butane", "n-butane", "i-pentane", "n-pentane", "n-hexane", "C7", "C8", "C9", "C10"],
    'MolarComposition[-]': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'MolarMass[kg/mol]': [None, None, None, None, None, None, None, None, None, None, None, 0.100, 0.114, 0.128, 0.142],
    'RelativeDensity[-]': [None, None, None, None, None, None, None, None, None, None, None, 0.746, 0.768, 0.790, 0.810]
}

# Full composition fluid (with water, inhibitors, and ions)
full_composition_fluid = {
    'ComponentName': ["water", "methanol", "MEG", "TEG", "nitrogen", "CO2", "H2S", "methane", "ethane", "propane", "i-butane", "n-butane", "i-pentane", "n-pentane", "n-hexane", "Na+", "Cl-", "K+", "Ca++", "Mg++", "Ba++", "Sr++", "Fe++", "SO4--", "HCO3-", "CO3--"],
    'MolarComposition[-]': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'MolarMass[kg/mol]': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    'RelativeDensity[-]': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
}

# Hydrate inhibitors list
INHIBITORS = {
    "None": None,
    "MEG (Mono-ethylene glycol)": "MEG",
    "Methanol": "methanol",
    "TEG (Tri-ethylene glycol)": "TEG",
    "DEG (Di-ethylene glycol)": "DEG"
}

# Define calculation mode
calc_mode = st.radio(
    "Select Calculation Mode",
    options=["Gas Composition Mode", "Full Composition Mode"],
    help="""
    **Gas Composition Mode**: Enter only gas components. Water, inhibitor, and salinity are set separately.
    
    **Full Composition Mode**: Enter all components including water, inhibitors, and ions directly.
    """,
    horizontal=True
)

st.divider()

if calc_mode == "Gas Composition Mode":
    st.subheader("Gas Composition Mode")
    st.caption("Enter gas composition only. Water is added automatically. Optionally add hydrate inhibitor and salinity.")
    
    with st.expander("📋 Set Gas Composition", expanded=True):
        st.markdown("""
        **How to use:**
        - Edit molar compositions in the table below
        - **Add new components**: Click the "+" button or add a new row with any [valid component name](https://github.com/equinor/neqsim/blob/master/src/main/resources/data/COMP.csv)
        - **Pseudo-components**: Add custom fractions (e.g., C11, C15, C20) by specifying the name, molar mass, and density
        - Components with zero composition are ignored
        """)
        # Reset button to restore default composition
        if st.button('Reset to Default Gas Composition'):
            st.session_state.hydrate_gas_df = pd.DataFrame(gas_only_fluid)
            st.rerun()

        hidecomponents_gas = st.checkbox('Show active components only', key='hide_gas_comp')
        if hidecomponents_gas and 'hydrate_gas_edited_df' in st.session_state:
            st.session_state.hydrate_gas_df = st.session_state.hydrate_gas_edited_df[
                st.session_state.hydrate_gas_edited_df['MolarComposition[-]'] > 0
            ]
        
        if 'hydrate_gas_uploaded_file' in st.session_state and st.session_state.hydrate_gas_uploaded_file is not None and not hidecomponents_gas:
            try:
                st.session_state.hydrate_gas_df = pd.read_csv(st.session_state.hydrate_gas_uploaded_file)
                numeric_columns = ['MolarComposition[-]', 'MolarMass[kg/mol]', 'RelativeDensity[-]']
                st.session_state.hydrate_gas_df[numeric_columns] = st.session_state.hydrate_gas_df[numeric_columns].astype(float)
            except Exception as e:
                st.warning(f'Could not load file: {e}')
                st.session_state.hydrate_gas_df = pd.DataFrame(gas_only_fluid)

        if 'hydrate_gas_df' not in st.session_state:
            st.session_state.hydrate_gas_df = pd.DataFrame(gas_only_fluid)

        gas_edited_df = st.data_editor(
            st.session_state.hydrate_gas_df,
            column_config={
                "ComponentName": "Component Name",
                "MolarComposition[-]": st.column_config.NumberColumn("Molar Composition [-]", min_value=0, max_value=10000, format="%f"),
                "MolarMass[kg/mol]": st.column_config.NumberColumn("Molar Mass [kg/mol]", min_value=0, max_value=10000, format="%f kg/mol"),
                "RelativeDensity[-]": st.column_config.NumberColumn("Density [gr/cm3]", min_value=1e-10, max_value=10.0, format="%f gr/cm3"),
            },
            num_rows='dynamic',
            key='gas_editor'
        )
        st.session_state.hydrate_gas_edited_df = gas_edited_df
        
        isplusfluid_gas = st.checkbox('Plus Fluid (last component is C7+ or similar)', key='gas_plus_fluid',
            help="Check this if the last component in your composition is a plus fraction (e.g., C7+, C10+). "
                 "The molar mass and density must be specified for plus fractions.")
        st.caption("💡 Gas composition will be normalized before simulation")

    with st.expander("📂 Fluid Library", expanded=False):
        if fluid_library_selector('hydrate_gas', 'hydrate_gas_df'):
            st.rerun()
        
        st.info("""
        **Adding Pseudo-components / Plus Fractions:**
        - You can add any hydrocarbon fraction by typing a name like `C11`, `C15`, `C20`, or `C7+`
        - For pseudo-components, you **must specify** both **Molar Mass** and **Density**
        - If "Plus Fluid" is checked, the **last component** is treated as the plus fraction (C7+, C10+, etc.)
        
        | Example | Molar Mass (kg/mol) | Density (g/cm³) |
        |---------|---------------------|------------------|
        | C7+ | 0.100 | 0.746 |
        | C10+ | 0.142 | 0.810 |
        | C15 | 0.206 | 0.836 |
        | C20 | 0.275 | 0.860 |
        """)
    
    with st.expander("🧪 Hydrate Inhibitor Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            inhibitor_name = st.selectbox(
                "Select Inhibitor",
                options=list(INHIBITORS.keys()),
                index=0,
                help="Select a thermodynamic hydrate inhibitor"
            )
        with col2:
            inhibitor_wt_pct = st.number_input(
                "Inhibitor Concentration (wt% in aqueous phase)",
                min_value=0.0,
                max_value=80.0,
                value=0.0,
                step=5.0,
                disabled=(inhibitor_name == "None"),
                help="Weight percent of inhibitor in the aqueous phase"
            )
        
        st.info("""
        💡 **Thermodynamic Inhibitors** shift the hydrate equilibrium to lower temperatures:
        - **MEG**: ~1°C depression per wt%, can be regenerated, low vapor losses
        - **Methanol**: ~1.5°C depression per wt%, high vapor losses
        - **TEG/DEG**: Lower volatility, but high viscosity at low temperatures
        """)
    
    with st.expander("🧂 Salinity Settings (Formation Water)", expanded=False):
        st.caption("Salinity is modeled using Na⁺ and Cl⁻ ions. Add additional ions as needed.")
        
        st.warning("""
        ⚠️ **Note:** The salinity you specify is the **feed water salinity** (formation water composition before mixing with gas).
        At equilibrium, some water may partition into the gas phase while ions remain in the aqueous phase,
        so the actual aqueous phase salinity may be slightly higher than the specified value.
        """)
        
        use_salinity = st.checkbox("Include Salinity Effects", value=False)
        
        # Molar masses for conversion (g/mol)
        MOLAR_MASSES = {
            "NaCl": 58.44,
            "K+": 39.10,
            "Ca++": 40.08,
            "Mg++": 24.31,
            "Ba++": 137.33,
            "SO4--": 96.06,
            "CO3--": 60.01
        }
        
        # Unit options for salinity input
        SALINITY_UNITS = {
            "g/kg": "g/kg",
            "g/L": "g/L", 
            "mol/kg": "mol/kg"
        }
        
        # Salinity basis options
        SALINITY_BASIS = {
            "Pure water only": "water",
            "Aqueous phase (water + inhibitor)": "aqueous"
        }
        
        if use_salinity:
            # Salinity basis and unit selectors
            basis_col, unit_col = st.columns(2)
            with basis_col:
                salinity_basis = st.selectbox(
                    "Salinity Basis",
                    options=list(SALINITY_BASIS.keys()),
                    index=0,
                    help="""
                    **Pure water only**: Salinity is defined per kg of water only (thermodynamically correct, standard definition of molality).
                    
                    **Aqueous phase**: Salinity is defined per kg of aqueous phase (water + inhibitor). More practical when measuring total brine concentration.
                    """
                )
            salinity_basis_key = SALINITY_BASIS[salinity_basis]
            
            with unit_col:
                salinity_unit = st.selectbox(
                    "Salinity Unit",
                    options=list(SALINITY_UNITS.keys()),
                    index=0,
                    help="""
                    **g/kg**: Grams of salt per kg (default, most common)
                    **g/L**: Grams of salt per liter (≈ g/kg for dilute solutions)
                    **mol/kg**: Molality (moles per kg)
                    """
                )
            unit_key = SALINITY_UNITS[salinity_unit]
            
            # Display clarifying note based on selection
            if salinity_basis_key == "water":
                basis_note = "📝 Salinity is calculated per kg of **pure water** (excludes inhibitor mass)"
            else:
                basis_note = "📝 Salinity is calculated per kg of **aqueous phase** (water + inhibitor combined)"
            st.caption(basis_note)
            
            # Set input ranges and defaults based on unit
            if unit_key == "g/kg":
                nacl_max, nacl_default, nacl_step = 350.0, 35.0, 1.0
                ion_max, ion_step = 50.0, 0.1
                nacl_help = f"Grams of NaCl per kg of {salinity_basis.lower()}. Seawater ≈ 35 g/kg."
            elif unit_key == "g/L":
                nacl_max, nacl_default, nacl_step = 350.0, 35.0, 1.0
                ion_max, ion_step = 50.0, 0.1
                nacl_help = f"Grams of NaCl per liter of {salinity_basis.lower()}. Seawater ≈ 35 g/L."
            else:  # mol/kg
                nacl_max, nacl_default, nacl_step = 6.0, 0.6, 0.1
                ion_max, ion_step = 1.0, 0.01
                nacl_help = f"Moles of NaCl per kg of {salinity_basis.lower()}. Seawater ≈ 0.6 mol/kg."
            
            nacl_input = st.number_input(
                f"NaCl Concentration ({unit_key})",
                min_value=0.0,
                max_value=nacl_max,
                value=nacl_default,
                step=nacl_step,
                help=nacl_help
            )
            
            # Convert input to mol/kg for calculations
            if unit_key == "g/kg":
                nacl_molal = nacl_input / MOLAR_MASSES["NaCl"]
            elif unit_key == "g/L":
                # g/L ≈ g/kg for water (density ≈ 1 kg/L at standard conditions)
                nacl_molal = nacl_input / MOLAR_MASSES["NaCl"]
            else:  # mol/kg
                nacl_molal = nacl_input
            
            st.caption("Additional ions can be added for more accurate formation water modeling:")
            
            # Additional ions
            add_other_ions = st.checkbox("Add other formation water ions", value=False)
            if add_other_ions:
                ion_col1, ion_col2, ion_col3 = st.columns(3)
                with ion_col1:
                    k_input = st.number_input(f"K⁺ ({unit_key})", min_value=0.0, max_value=ion_max, value=0.0, step=ion_step)
                    ca_input = st.number_input(f"Ca²⁺ ({unit_key})", min_value=0.0, max_value=ion_max, value=0.0, step=ion_step)
                with ion_col2:
                    mg_input = st.number_input(f"Mg²⁺ ({unit_key})", min_value=0.0, max_value=ion_max, value=0.0, step=ion_step)
                    ba_input = st.number_input(f"Ba²⁺ ({unit_key})", min_value=0.0, max_value=ion_max/3, value=0.0, step=ion_step)
                with ion_col3:
                    so4_input = st.number_input(f"SO₄²⁻ ({unit_key})", min_value=0.0, max_value=ion_max*2, value=0.0, step=ion_step)
                    co3_input = st.number_input(f"CO₃²⁻ ({unit_key})", min_value=0.0, max_value=ion_max, value=0.0, step=ion_step)
                
                # Convert to mol/kg based on selected unit
                if unit_key == "g/kg" or unit_key == "g/L":
                    k_molal = k_input / MOLAR_MASSES["K+"]
                    ca_molal = ca_input / MOLAR_MASSES["Ca++"]
                    mg_molal = mg_input / MOLAR_MASSES["Mg++"]
                    ba_molal = ba_input / MOLAR_MASSES["Ba++"]
                    so4_molal = so4_input / MOLAR_MASSES["SO4--"]
                    co3_molal = co3_input / MOLAR_MASSES["CO3--"]
                else:  # mol/kg - direct input
                    k_molal = k_input
                    ca_molal = ca_input
                    mg_molal = mg_input
                    ba_molal = ba_input
                    so4_molal = so4_input
                    co3_molal = co3_input
            else:
                k_molal = ca_molal = mg_molal = ba_molal = so4_molal = co3_molal = 0.0
            
            st.info("💡 When ions are present, the **Electrolyte-CPA-EoS (Statoil)** is used for accurate ion-solvent interactions.")
        else:
            nacl_molal = k_molal = ca_molal = mg_molal = ba_molal = so4_molal = co3_molal = 0.0
            salinity_basis_key = "water"  # Default value when salinity is disabled

else:  # Full Composition Mode
    st.subheader("Full Composition Mode")
    st.caption("Enter all components including water, inhibitors, and ions directly in the fluid composition.")
    
    with st.expander("📋 Set Fluid Composition", expanded=True):
        st.markdown("""
        **How to use:**
        - Edit molar compositions in the table below
        - **Add new components**: Click the "+" button or add a new row with any [valid component name](https://github.com/equinor/neqsim/blob/master/src/main/resources/data/COMP.csv)
        - **Pseudo-components**: Add custom fractions (e.g., C11, C15, C20) by specifying the name, molar mass, and density
        - **Ions**: Use names like `Na+`, `Cl-`, `Ca++`, `SO4--` for formation water ions
        - Components with zero composition are ignored
        """)
        # Reset button to restore default composition
        if st.button('Reset to Default Composition'):
            st.session_state.hydrate_fluid_df = pd.DataFrame(full_composition_fluid)
            st.rerun()

        hidecomponents = st.checkbox('Show active components only')
        if hidecomponents and 'hydrate_edited_df' in st.session_state:
            st.session_state.hydrate_fluid_df = st.session_state.hydrate_edited_df[
                st.session_state.hydrate_edited_df['MolarComposition[-]'] > 0
            ]
        
        if 'hydrate_uploaded_file' in st.session_state and st.session_state.hydrate_uploaded_file is not None and not hidecomponents:
            try:
                st.session_state.hydrate_fluid_df = pd.read_csv(st.session_state.hydrate_uploaded_file)
                numeric_columns = ['MolarComposition[-]', 'MolarMass[kg/mol]', 'RelativeDensity[-]']
                st.session_state.hydrate_fluid_df[numeric_columns] = st.session_state.hydrate_fluid_df[numeric_columns].astype(float)
            except Exception as e:
                st.warning(f'Could not load file: {e}')
                st.session_state.hydrate_fluid_df = pd.DataFrame(full_composition_fluid)

        if 'hydrate_fluid_df' not in st.session_state:
            st.session_state.hydrate_fluid_df = pd.DataFrame(full_composition_fluid)

        st.edited_df = st.data_editor(
            st.session_state.hydrate_fluid_df,
            column_config={
                "ComponentName": "Component Name",
                "MolarComposition[-]": st.column_config.NumberColumn("Molar Composition [-]", min_value=0, max_value=10000, format="%f"),
                "MolarMass[kg/mol]": st.column_config.NumberColumn("Molar Mass [kg/mol]", min_value=0, max_value=10000, format="%f kg/mol"),
                "RelativeDensity[-]": st.column_config.NumberColumn("Density [gr/cm3]", min_value=1e-10, max_value=10.0, format="%f gr/cm3"),
            },
            num_rows='dynamic'
        )
        st.session_state.hydrate_edited_df = st.edited_df

        isplusfluid = st.checkbox('Plus Fluid (last component is C7+ or similar)',
            help="Check this if the last component in your composition is a plus fraction (e.g., C7+, C10+). "
                 "The molar mass and density must be specified for plus fractions.")
        st.caption("💡 Fluid composition will be normalized before simulation")
        
        st.info("""
        **Adding Pseudo-components / Plus Fractions:**
        - You can add any hydrocarbon fraction by typing a name like `C11`, `C15`, `C20`, or `C7+`
        - For pseudo-components, you **must specify** both **Molar Mass** and **Density**
        - If "Plus Fluid" is checked, the **last component** is treated as the plus fraction
        
        | Example | Molar Mass (kg/mol) | Density (g/cm³) |
        |---------|---------------------|------------------|
        | C7+ | 0.100 | 0.746 |
        | C10+ | 0.142 | 0.810 |
        | C15 | 0.206 | 0.836 |
        | C20 | 0.275 | 0.860 |
        """)
        
        st.info("""
        **Available Ions for Formation Water:**
        - Cations: Na⁺, K⁺, Ca²⁺, Mg²⁺, Ba²⁺, Sr²⁺, Fe²⁺
        - Anions: Cl⁻, SO₄²⁻, HCO₃⁻, CO₃²⁻, Br⁻, OH⁻
        
        When ions are present, the **Electrolyte-CPA-EoS (Statoil)** is automatically selected.
        Otherwise, **CPA-EoS (Statoil)** is used.
        """)

st.divider()

# Pressure input - common for both modes
st.text("Input Pressures for Hydrate Equilibrium Calculation")

if 'hydrate_tp_data' not in st.session_state:
    st.session_state['hydrate_tp_data'] = pd.DataFrame({
        'Pressure (bara)': [50.0, 100.0, 150.0, 200.0],
        'Temperature (C)': [None, None, None, None]
    })

st.edited_dfTP = st.data_editor(
    st.session_state.hydrate_tp_data['Pressure (bara)'].dropna().reset_index(drop=True),
    num_rows='dynamic',
    column_config={
        'Pressure (bara)': st.column_config.NumberColumn(
            label="Pressure (bara)",
            min_value=0.0,
            max_value=1000,
            format='%f',
            help='Enter the pressure in bar absolute.'
        )
    }
)

if st.button('Run Hydrate Calculation'):
    with st.spinner('Calculating hydrate equilibrium...'):
        try:
            if calc_mode == "Gas Composition Mode":
                # Build fluid from gas composition + water + inhibitor + ions
                gas_df = st.session_state.hydrate_gas_edited_df
                
                # Check if gas composition is provided
                if gas_df['MolarComposition[-]'].sum() <= 0:
                    st.error('Please enter a gas composition.')
                    st.stop()
                
                # Determine if we have ions (for model selection)
                has_ions = use_salinity and (nacl_molal > 0 or k_molal > 0 or ca_molal > 0 or mg_molal > 0 or ba_molal > 0 or so4_molal > 0 or co3_molal > 0)
                
                # Select appropriate EoS
                if has_ions:
                    neqsim_fluid = fluid("Electrolyte-CPA-EoS")
                    st.info("Using **Electrolyte-CPA-EoS (Statoil)** due to presence of ions.")
                else:
                    neqsim_fluid = fluid("cpa")
                    st.info("Using **CPA-EoS (Statoil)**")
                
                # Add gas components (with support for plus fractions)
                total_gas_moles = gas_df['MolarComposition[-]'].sum()
                active_rows = gas_df[gas_df['MolarComposition[-]'] > 0]
                num_active = len(active_rows)
                
                for i, (idx, row) in enumerate(active_rows.iterrows()):
                    comp_name = row['ComponentName']
                    molar_comp = row['MolarComposition[-]']
                    molar_mass = row.get('MolarMass[kg/mol]', None)
                    rel_density = row.get('RelativeDensity[-]', None)
                    
                    # Check if this is the last component and plus fluid is enabled
                    is_last = (i == num_active - 1)
                    if is_last and isplusfluid_gas and molar_mass is not None and rel_density is not None:
                        # Add as plus fraction with TB (true boiling point) characterization
                        neqsim_fluid.addTBPfraction(comp_name, molar_comp, molar_mass, rel_density)
                    elif molar_mass is not None and rel_density is not None and pd.notna(molar_mass) and pd.notna(rel_density):
                        # Component with specified molar mass and density (pseudo-component)
                        neqsim_fluid.addTBPfraction(comp_name, molar_comp, molar_mass, rel_density)
                    else:
                        # Regular component
                        neqsim_fluid.addComponent(comp_name, molar_comp)
                
                # Add water (assume some water is always present for hydrate formation)
                # Typically 1-5 mol% water relative to gas
                water_moles = total_gas_moles * 0.05  # 5% water relative to gas
                neqsim_fluid.addComponent("water", water_moles)
                
                # Add inhibitor if selected
                inhibitor_moles = 0.0
                mass_inhibitor = 0.0
                if inhibitor_name != "None" and inhibitor_wt_pct > 0:
                    inhibitor_component = INHIBITORS[inhibitor_name]
                    # Calculate inhibitor moles based on wt% in aqueous phase
                    # Molar masses: MEG=62.07, Methanol=32.04, TEG=150.17, DEG=106.12
                    molar_masses = {"MEG": 62.07, "methanol": 32.04, "TEG": 150.17, "DEG": 106.12}
                    mm_inhibitor = molar_masses.get(inhibitor_component, 62.07)
                    mm_water = 18.015
                    
                    # wt% = (m_inh / (m_inh + m_water)) * 100
                    # Solve for moles of inhibitor given water moles
                    # n_inh * MM_inh / (n_inh * MM_inh + n_water * MM_water) = wt%/100
                    wt_frac = inhibitor_wt_pct / 100.0
                    mass_water = water_moles * mm_water
                    mass_inhibitor = (wt_frac * mass_water) / (1 - wt_frac)
                    inhibitor_moles = mass_inhibitor / mm_inhibitor
                    
                    neqsim_fluid.addComponent(inhibitor_component, inhibitor_moles)
                
                # Add ions if salinity is enabled
                if has_ions:
                    # Calculate basis mass for salinity (kg)
                    mm_water = 18.015
                    kg_water = water_moles * mm_water / 1000.0
                    
                    if salinity_basis_key == "aqueous":
                        # Salinity per kg of aqueous phase (water + inhibitor)
                        kg_aqueous = kg_water + (mass_inhibitor / 1000.0)
                        salinity_basis_mass = kg_aqueous
                    else:
                        # Salinity per kg of pure water (default, thermodynamically correct)
                        salinity_basis_mass = kg_water
                    
                    # Calculate total Cl- needed for electroneutrality
                    # Start with Cl- from NaCl
                    total_cl_moles = nacl_molal * salinity_basis_mass
                    
                    # Add Na+ from NaCl
                    na_moles = nacl_molal * salinity_basis_mass
                    neqsim_fluid.addComponent("Na+", na_moles)
                    
                    # Add other cations and calculate additional Cl- for electroneutrality
                    if k_molal > 0:
                        neqsim_fluid.addComponent("K+", k_molal * salinity_basis_mass)
                        total_cl_moles += k_molal * salinity_basis_mass  # K+ has +1 charge
                    if ca_molal > 0:
                        neqsim_fluid.addComponent("Ca++", ca_molal * salinity_basis_mass)
                        total_cl_moles += 2 * ca_molal * salinity_basis_mass  # Ca++ has +2 charge
                    if mg_molal > 0:
                        neqsim_fluid.addComponent("Mg++", mg_molal * salinity_basis_mass)
                        total_cl_moles += 2 * mg_molal * salinity_basis_mass  # Mg++ has +2 charge
                    if ba_molal > 0:
                        neqsim_fluid.addComponent("Ba++", ba_molal * salinity_basis_mass)
                        total_cl_moles += 2 * ba_molal * salinity_basis_mass  # Ba++ has +2 charge
                    
                    # Add anions (reduce Cl- needed for electroneutrality)
                    if so4_molal > 0:
                        neqsim_fluid.addComponent("SO4--", so4_molal * salinity_basis_mass)
                        total_cl_moles -= 2 * so4_molal * salinity_basis_mass  # SO4-- has -2 charge
                    if co3_molal > 0:
                        neqsim_fluid.addComponent("CO3--", co3_molal * salinity_basis_mass)
                        total_cl_moles -= 2 * co3_molal * salinity_basis_mass  # CO3-- has -2 charge
                    
                    # Add Cl- to balance all cations (ensure electroneutrality)
                    if total_cl_moles > 0:
                        neqsim_fluid.addComponent("Cl-", total_cl_moles)
                
                # Set mixing rule
                neqsim_fluid.setMixingRule(10)  # CPA/Electrolyte mixing rule
                neqsim_fluid.setMultiPhaseCheck(True)  # Enable multiphase flash
                neqsim_fluid.setHydrateCheck(True)
                
            else:  # Full Composition Mode
                comp_df = st.session_state.hydrate_edited_df
                
                # Check if water composition is greater than 0
                water_row = comp_df[comp_df['ComponentName'] == 'water']
                if water_row.empty or water_row['MolarComposition[-]'].iloc[0] <= 0:
                    st.error('Water Molar Composition must be greater than 0 for hydrate calculations. Please adjust your inputs.')
                    st.stop()
                
                # Check if there are ions in the composition
                ion_names = ['Na+', 'Cl-', 'K+', 'Ca++', 'Mg++', 'Ba++', 'Sr++', 'Fe++', 'SO4--', 'HCO3-', 'CO3--', 'Br-', 'OH-']
                has_ions = any(
                    comp_df[(comp_df['ComponentName'] == ion) & (comp_df['MolarComposition[-]'] > 0)].shape[0] > 0
                    for ion in ion_names
                )
                
                # Select appropriate EoS
                if has_ions:
                    neqsim_fluid = fluid("Electrolyte-CPA-EoS")
                    st.info("Using **Electrolyte-CPA-EoS (Statoil)** due to presence of ions.")
                else:
                    neqsim_fluid = fluid("cpa")
                    st.info("Using **CPA-EoS (Statoil)**")
                
                # Add all components (with support for plus fractions)
                active_rows = comp_df[comp_df['MolarComposition[-]'] > 0]
                num_active = len(active_rows)
                
                for i, (idx, row) in enumerate(active_rows.iterrows()):
                    comp_name = row['ComponentName']
                    molar_comp = row['MolarComposition[-]']
                    molar_mass = row.get('MolarMass[kg/mol]', None)
                    rel_density = row.get('RelativeDensity[-]', None)
                    
                    # Check if this is the last component and plus fluid is enabled
                    is_last = (i == num_active - 1)
                    if is_last and isplusfluid and molar_mass is not None and rel_density is not None:
                        # Add as plus fraction with TB (true boiling point) characterization
                        neqsim_fluid.addTBPfraction(comp_name, molar_comp, molar_mass, rel_density)
                    elif molar_mass is not None and rel_density is not None and pd.notna(molar_mass) and pd.notna(rel_density):
                        # Component with specified molar mass and density (pseudo-component)
                        neqsim_fluid.addTBPfraction(comp_name, molar_comp, molar_mass, rel_density)
                    else:
                        # Regular component
                        neqsim_fluid.addComponent(comp_name, molar_comp)
                
                # Set mixing rule
                neqsim_fluid.setMixingRule(10)
                neqsim_fluid.setMultiPhaseCheck(True)  # Enable multiphase flash
                neqsim_fluid.setHydrateCheck(True)
            
            # Run hydrate calculations for each pressure
            results_list = []
            pres_list = []
            fluid_results_list = []
            
            # Define a reasonable initial temperature guess for hydrate calculations
            initial_temp_guess = 25.0  # °C - typical starting point
            
            # Sort pressures to improve convergence (start from lower pressures)
            pressures_sorted = sorted(st.edited_dfTP.dropna().tolist())
            
            # Store a pristine copy of the fluid for display purposes
            # This ensures display fluids always have correct settings
            pristine_fluid = neqsim_fluid.clone()
            
            for i, pressure in enumerate(pressures_sorted):
                pres_list.append(pressure)
                
                # Clone fluid for each calculation to prevent state pollution
                # This is critical for electrolyte systems with ions (Na+, Cl-)
                # where cumulative state changes can cause convergence issues
                fluid_clone = neqsim_fluid.clone()
                fluid_clone.setPressure(pressure, 'bara')
                fluid_clone.setTemperature(initial_temp_guess, 'C')
                
                # For electrolyte systems, disable multiPhaseCheck after first calculation
                # to speed up convergence (stability analysis is expensive for ions)
                if i > 0 and has_ions:
                    fluid_clone.setMultiPhaseCheck(False)
                
                hydrate_temp = hydt(fluid_clone) - 273.15
                results_list.append(hydrate_temp)
                
                # Create a fresh clone from pristine fluid for property display
                # For electrolyte systems, disable multiPhaseCheck to avoid spurious
                # oil phases from stability analysis (false positives with CPA-EoS)
                display_fluid = pristine_fluid.clone()
                display_fluid.setPressure(pressure, 'bara')
                display_fluid.setTemperature(hydrate_temp, 'C')
                TPflash(display_fluid)
                fluid_results_list.append(dataFrame(display_fluid))
            
            # Store results
            st.session_state['hydrate_tp_data'] = pd.DataFrame({
                'Pressure (bara)': pres_list,
                'Temperature (C)': results_list
            })
            st.session_state['hydrate_tp_data'] = st.session_state['hydrate_tp_data'].sort_values('Pressure (bara)')
            
            st.success('Hydrate calculation finished successfully!')
            
            # Display results table
            st.subheader("Results")
            st.data_editor(
                st.session_state.hydrate_tp_data.reset_index(drop=True),
                num_rows='dynamic',
                column_config={
                    'Pressure (bara)': st.column_config.NumberColumn(
                        label="Pressure (bara)",
                        min_value=0.0,
                        max_value=1000,
                        format='%f',
                        help='Pressure in bar absolute.'
                    ),
                    'Temperature (C)': st.column_config.NumberColumn(
                        label="Hydrate Temperature (°C)",
                        min_value=-273.15,
                        max_value=1000,
                        format='%.2f',
                        disabled=True
                    ),
                }
            )
            
            st.divider()
            
            # Create interactive Plotly chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state['hydrate_tp_data']['Temperature (C)'],
                y=st.session_state['hydrate_tp_data']['Pressure (bara)'],
                mode='lines+markers',
                name='Hydrate Equilibrium',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=10, color='#2E86AB'),
                hovertemplate='T: %{x:.1f} °C<br>P: %{y:.1f} bara<extra></extra>'
            ))
            
            fig.update_layout(
                title=dict(text='Hydrate Equilibrium Line', font=dict(size=20)),
                xaxis_title='Temperature (°C)',
                yaxis_title='Pressure (bara)',
                hovermode='closest',
                template='plotly_white',
                height=450
            )
            
            # Add hydrate zone annotation
            fig.add_annotation(
                x=st.session_state['hydrate_tp_data']['Temperature (C)'].min() - 2,
                y=st.session_state['hydrate_tp_data']['Pressure (bara)'].mean(),
                text="← Hydrate Zone",
                showarrow=False,
                font=dict(size=12, color='red')
            )
            
            st.plotly_chart(fig, width='stretch')
            
            st.divider()
            
            # Show fluid details
            with st.expander("🔬 Detailed Fluid Properties", expanded=False):
                combined_results = pd.concat(fluid_results_list, ignore_index=True)
                st.dataframe(combined_results)
            
            # Show model info
            with st.expander("ℹ️ Model Information", expanded=False):
                list1 = neqsim_fluid.getComponentNames()
                l1 = list(list1)
                string_list = [str(element) for element in l1]
                st.write("**Components in fluid:**")
                st.write(", ".join(string_list))
                st.write(f"**Model name:** {neqsim_fluid.getModelName()}")
            
            # Try to get AI recommendations
            try:
                input_text = "What scientific experimental hydrate equilibrium data are available for mixtures of " + ", ".join(string_list)
                openapitext = st.make_request(input_text)
                st.write(openapitext)
            except Exception:
                pass  # AI features optional
            
        except Exception as e:
            st.error(f'Calculation failed: {str(e)}')
            import traceback
            st.error(traceback.format_exc())

# File upload in sidebar
st.sidebar.file_uploader(
    "Import Gas Composition" if calc_mode == "Gas Composition Mode" else "Import Fluid Composition", 
    key='hydrate_gas_uploaded_file' if calc_mode == "Gas Composition Mode" else 'hydrate_uploaded_file', 
    help='Fluids can be saved by hovering over the fluid window and clicking the "Download as CSV" button in the upper-right corner.'
)

# Additional information in sidebar
with st.sidebar.expander("ℹ️ About Hydrate Inhibitors"):
    st.markdown("""
    **Thermodynamic Inhibitors:**
    - Shift hydrate equilibrium to lower temperatures
    - Work by lowering water activity
    
    **Common Inhibitors:**
    | Inhibitor | Depression per wt% | Advantages |
    |-----------|-------------------|------------|
    | MEG | ~1.0°C | Regenerable, low losses |
    | Methanol | ~1.5°C | Strong effect |
    | TEG | ~0.8°C | Very low losses |
    
    **Salt Effects:**
    - Salts also act as hydrate inhibitors
    - Effect depends on ionic strength
    - Common in produced formation water
    """)
