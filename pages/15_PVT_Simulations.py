# -*- coding: utf-8 -*-
"""
PVT Simulations Page
=====================
Industry-standard PVT experiments for reservoir fluid characterization:
- Constant Composition Expansion (CCE)
- Constant Volume Depletion (CVD)
- Differential Liberation (DL)
- Separator Test
- Swelling Test
- GOR/Bo Curves
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from neqsim.thermo import fluid_df, TPflash
from neqsim import jneqsim
from fluids import default_fluid
from theme import apply_theme

st.set_page_config(page_title="PVT Simulations", page_icon='images/neqsimlogocircleflat.png', layout="wide")
apply_theme()

st.title('🛢️ PVT Simulations')

st.markdown("""
PVT (Pressure-Volume-Temperature) simulations are industry-standard experiments used for reservoir fluid characterization. 
These experiments help understand fluid behavior during production and optimize recovery strategies.

**Available Experiments:**
- **CCE**: Constant Composition Expansion - Measure bubble point and fluid compressibility
- **CVD**: Constant Volume Depletion - Gas condensate liquid dropout curve
- **Differential Liberation**: Simulate reservoir depletion with gas removal
- **Separator Test**: Multi-stage separation optimization
- **Swelling Test**: Gas injection effects on oil (EOR studies)
- **GOR/Bo Curves**: Solution gas-oil ratio and formation volume factor
""")

st.divider()

# =============================================================================
# Sidebar Configuration
# =============================================================================
with st.sidebar:
    st.header("📋 Fluid Configuration")
    
    st.subheader("Sample Fluids")
    sample_fluid = st.selectbox(
        "Load sample fluid",
        ["Custom", "Black Oil", "Gas Condensate", "Volatile Oil"],
        help="Load a predefined fluid composition"
    )
    
    st.divider()
    st.subheader("Reservoir Conditions")
    reservoir_temp = st.number_input("Reservoir Temperature (°C)", value=100.0, min_value=0.0, max_value=300.0)
    reservoir_pres = st.number_input("Reservoir Pressure (bara)", value=300.0, min_value=1.0, max_value=1000.0)
    
    st.divider()
    st.file_uploader(
        "Import Fluid (CSV)",
        key='pvt_uploaded_file',
        help='Import a fluid composition from CSV file'
    )
    
    st.divider()
    st.markdown("""
    ### 📖 References
    - [NeqSim PVT Documentation](https://github.com/equinor/neqsim/tree/main/docs/pvtsimulation)
    - [Whitson PVT Theory](https://wiki.whitson.com/phase_behavior/pvt_exp/)
    """)

# =============================================================================
# Sample Fluid Definitions
# =============================================================================
black_oil_fluid = {
    'ComponentName': ["nitrogen", "CO2", "methane", "ethane", "propane", "i-butane", "n-butane", 
                      "i-pentane", "n-pentane", "n-hexane", "C7", "C8", "C9", "C10"],
    'MolarComposition[-]': [0.5, 1.5, 40.0, 8.0, 6.0, 1.5, 3.0, 1.5, 2.0, 3.0, 8.0, 7.0, 6.0, 12.0],
    'MolarMass[kg/mol]': [None, None, None, None, None, None, None, None, None, None, 
                          0.0913, 0.1041, 0.1188, 0.200],
    'RelativeDensity[-]': [None, None, None, None, None, None, None, None, None, None, 
                           0.746, 0.768, 0.79, 0.832]
}

gas_condensate_fluid = {
    'ComponentName': ["nitrogen", "CO2", "methane", "ethane", "propane", "i-butane", "n-butane", 
                      "i-pentane", "n-pentane", "n-hexane", "C7", "C8", "C9", "C10"],
    'MolarComposition[-]': [1.0, 2.0, 75.0, 7.0, 4.0, 1.0, 2.0, 0.8, 1.0, 1.5, 2.0, 1.5, 0.8, 0.4],
    'MolarMass[kg/mol]': [None, None, None, None, None, None, None, None, None, None, 
                          0.0913, 0.1041, 0.1188, 0.150],
    'RelativeDensity[-]': [None, None, None, None, None, None, None, None, None, None, 
                           0.746, 0.768, 0.79, 0.80]
}

volatile_oil_fluid = {
    'ComponentName': ["nitrogen", "CO2", "methane", "ethane", "propane", "i-butane", "n-butane", 
                      "i-pentane", "n-pentane", "n-hexane", "C7", "C8", "C9", "C10"],
    'MolarComposition[-]': [0.8, 2.5, 55.0, 10.0, 7.0, 2.0, 3.5, 2.0, 2.5, 3.0, 5.0, 4.0, 2.0, 0.7],
    'MolarMass[kg/mol]': [None, None, None, None, None, None, None, None, None, None, 
                          0.0913, 0.1041, 0.1188, 0.160],
    'RelativeDensity[-]': [None, None, None, None, None, None, None, None, None, None, 
                           0.746, 0.768, 0.79, 0.81]
}

# Select fluid based on dropdown
if sample_fluid == "Black Oil":
    initial_fluid = black_oil_fluid
elif sample_fluid == "Gas Condensate":
    initial_fluid = gas_condensate_fluid
elif sample_fluid == "Volatile Oil":
    initial_fluid = volatile_oil_fluid
else:
    initial_fluid = black_oil_fluid  # Default to black oil for custom

# Session state management
if 'pvt_fluid_df' not in st.session_state or st.session_state.get('pvt_sample_fluid') != sample_fluid:
    st.session_state.pvt_fluid_df = pd.DataFrame(initial_fluid)
    st.session_state.pvt_sample_fluid = sample_fluid

# Handle uploaded file
if 'pvt_uploaded_file' in st.session_state and st.session_state.pvt_uploaded_file is not None:
    try:
        st.session_state.pvt_fluid_df = pd.read_csv(st.session_state.pvt_uploaded_file)
        numeric_columns = ['MolarComposition[-]', 'MolarMass[kg/mol]', 'RelativeDensity[-]']
        for col in numeric_columns:
            if col in st.session_state.pvt_fluid_df.columns:
                st.session_state.pvt_fluid_df[col] = pd.to_numeric(st.session_state.pvt_fluid_df[col], errors='coerce')
    except Exception as e:
        st.error(f"Error loading file: {e}")

# =============================================================================
# Fluid Composition Editor
# =============================================================================
st.subheader("🧪 Fluid Composition")

with st.expander("Edit Fluid Composition", expanded=False):
    edited_fluid_df = st.data_editor(
        st.session_state.pvt_fluid_df,
        column_config={
            "ComponentName": st.column_config.TextColumn("Component"),
            "MolarComposition[-]": st.column_config.NumberColumn("Molar Composition", min_value=0.0, format="%.4f"),
            "MolarMass[kg/mol]": st.column_config.NumberColumn("Molar Mass [kg/mol]", min_value=0.0, format="%.4f"),
            "RelativeDensity[-]": st.column_config.NumberColumn("Density [g/cm³]", min_value=0.0, format="%.4f"),
        },
        num_rows='dynamic',
        use_container_width=True
    )
    st.session_state.pvt_fluid_df = edited_fluid_df

isplusfluid = st.checkbox('Last component is Plus Fraction (C10+)', value=True)

st.divider()

# =============================================================================
# Create tabs for different PVT experiments
# =============================================================================
tab_char, tab_cce, tab_cvd, tab_dl, tab_sep, tab_swell, tab_gorbo = st.tabs([
    "🔬 Fluid Characterization",
    "📊 CCE (Constant Composition Expansion)",
    "💧 CVD (Constant Volume Depletion)",
    "🔄 Differential Liberation", 
    "⚗️ Separator Test",
    "💉 Swelling Test",
    "📈 GOR/Bo Curves"
])

# =============================================================================
# Helper function to create fluid
# =============================================================================
def create_neqsim_fluid(df, is_plus_fraction=True):
    """Create a NeqSim fluid from DataFrame"""
    jneqsim.util.database.NeqSimDataBase.setCreateTemporaryTables(True)
    neqsim_fluid = fluid_df(df, lastIsPlusFraction=is_plus_fraction, add_all_components=False)
    neqsim_fluid.setModel("srk")
    neqsim_fluid.setMixingRule("classic")
    neqsim_fluid.setMultiPhaseCheck(True)
    jneqsim.util.database.NeqSimDataBase.setCreateTemporaryTables(False)
    return neqsim_fluid

def create_characterized_fluid(df, is_plus_fraction=True, plus_fraction_model="Pedersen", 
                                lumping_model="PVTlumpingModel", num_pseudo_components=12,
                                gamma_alpha=1.0, gamma_eta=90.0,
                                eos_model="SRK"):
    """Create a NeqSim fluid with plus fraction characterization
    
    Two workflows supported:
    1. Single plus fraction (e.g., C10+ only): Add light ends + one plus fraction, then characterize
    2. TBP table + plus fraction (e.g., C7-C9 + C10+): Add TBP fractions + plus fraction, then characterize
    
    The characterization expands the PLUS FRACTION only, not existing TBP fractions.
    """
    jneqsim.util.database.NeqSimDataBase.setCreateTemporaryTables(True)
    
    # Map EoS selection to NeqSim model names
    eos_mapping = {
        "SRK": "srk",
        "PR": "pr",
        "PR78": "pr78",
        "CPA": "cpa"
    }
    neqsim_eos = eos_mapping.get(eos_model, "srk")
    
    # Create empty fluid with selected EoS
    from neqsim.thermo import fluid
    neqsim_fluid = fluid(neqsim_eos)
    
    # Get valid rows (non-zero composition)
    valid_rows = df[df['MolarComposition[-]'] > 0].reset_index(drop=True)
    
    # Add components from DataFrame
    for idx, row in valid_rows.iterrows():
        comp_name = row['ComponentName']
        molar_comp = row['MolarComposition[-]']
        
        # Check if this is the LAST component
        is_last = (idx == len(valid_rows) - 1)
        has_molar_mass = pd.notna(row.get('MolarMass[kg/mol]', None)) and row.get('MolarMass[kg/mol]', 0) > 0
        has_density = pd.notna(row.get('RelativeDensity[-]', None)) and row.get('RelativeDensity[-]', 0) > 0
        
        if is_last and is_plus_fraction and has_molar_mass and has_density:
            # Last component with MW/density → Plus Fraction (will be characterized)
            molar_mass = float(row['MolarMass[kg/mol]'])
            density = float(row['RelativeDensity[-]'])
            neqsim_fluid.addPlusFraction(comp_name, molar_comp, molar_mass, density)
        elif has_molar_mass and has_density:
            # Intermediate component with MW/density → TBP Fraction (kept as-is)
            molar_mass = float(row['MolarMass[kg/mol]'])
            density = float(row['RelativeDensity[-]'])
            neqsim_fluid.addTBPfraction(comp_name, molar_comp, molar_mass, density)
        else:
            # Regular defined component (N2, CO2, C1-C6, etc.)
            neqsim_fluid.addComponent(comp_name, molar_comp)
    
    if is_plus_fraction:
        # Configure characterization - this expands the PLUS FRACTION only
        char = neqsim_fluid.getCharacterization()
        char.setPlusFractionModel(plus_fraction_model)
        
        # Handle lumping model explicitly
        if lumping_model.lower() == "no lumping":
            char.setLumpingModel("no lumping")
            # Do NOT set numberOfPseudoComponents for "no lumping"
        elif lumping_model == "Standard":
            char.setLumpingModel("standard")
            char.getLumpingModel().setNumberOfPseudoComponents(num_pseudo_components)
        else:
            # PVTlumpingModel
            char.setLumpingModel("PVTlumpingModel")
            char.getLumpingModel().setNumberOfPseudoComponents(num_pseudo_components)
        
        # Set gamma model parameters if using Whitson
        if plus_fraction_model == "Whitson Gamma Model":
            try:
                char.getPlusFractionModel().setGammaParameters(gamma_alpha, gamma_eta)
            except Exception:
                pass
        
        # Run characterization - this expands ONLY the plus fraction
        char.characterisePlusFraction()
    
    neqsim_fluid.setMixingRule('classic')
    neqsim_fluid.setMultiPhaseCheck(True)
    neqsim_fluid.useVolumeCorrection(True)
    
    jneqsim.util.database.NeqSimDataBase.setCreateTemporaryTables(False)
    return neqsim_fluid

# =============================================================================
# TAB 0: Fluid Characterization
# =============================================================================
with tab_char:
    st.header("🔬 Fluid Characterization")
    st.markdown("""
    **Plus Fraction Characterization** splits heavy hydrocarbon fractions (C7+, C10+, etc.) into 
    pseudo-components for accurate PVT modeling. This is essential for reservoir simulation.
    
    **Available Methods:**
    - **Pedersen**: Exponential distribution based on carbon number
    - **Pedersen Heavy Oil**: Extended Pedersen model for heavy oils (up to C200)
    - **Whitson Gamma**: Statistical gamma distribution with tunable shape parameter (α)
    
    **Lumping Models:**
    - **PVTlumpingModel**: Groups pseudo-components into a specified number of lumps
    - **No Lumping**: Keeps all individual pseudo-components
    - **Standard**: Default NeqSim lumping
    """)
    
    st.divider()
    
    # Configuration columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Characterization Settings")
        
        eos_model = st.selectbox(
            "Equation of State",
            ["SRK", "PR", "PR78", "CPA"],
            help="""
            **SRK**: Soave-Redlich-Kwong - general purpose, good for gas systems
            **PR**: Peng-Robinson - better liquid density predictions
            **PR78**: Peng-Robinson (1978) - improved version for heavier hydrocarbons
            **CPA**: Cubic-Plus-Association - for polar/associating compounds (water, MEG, methanol)
            
            All use volume correction by default for improved density predictions.
            """
        )
        
        plus_fraction_model = st.selectbox(
            "Plus Fraction Distribution Model",
            ["Pedersen", "Pedersen Heavy Oil", "Whitson Gamma Model"],
            help="""
            **Pedersen**: Standard exponential mole fraction distribution
            **Pedersen Heavy Oil**: For oils with MW > 600 g/mol
            **Whitson Gamma**: Continuous gamma distribution, more flexible
            """
        )
        
        lumping_model = st.selectbox(
            "Lumping Model",
            ["Standard", "PVTlumpingModel", "no lumping"],
            help="""
            **Standard**: Default lumping based on carbon number ranges
            **PVTlumpingModel**: Lump into equal weight fractions
            **No Lumping**: Keep all pseudo-components separate
            """
        )
        
        num_pseudo = st.slider(
            "Number of Pseudo-components (Final Fluid)", 
            min_value=3, max_value=20, value=6,
            help="Number of lumped pseudo-components in the final characterized fluid. The plus fraction is first expanded into many components, then grouped (lumped) into this many pseudo-components. Typical values: 4-12."
        )
    
    with col2:
        st.subheader("⚙️ Model Parameters")
        
        # Whitson Gamma specific parameters
        if plus_fraction_model == "Whitson Gamma Model":
            st.info("**Whitson Gamma Distribution Parameters**")
            
            gamma_alpha = st.number_input(
                "α (Shape Parameter)",
                min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                help="""
                Shape parameter of gamma distribution:
                - **0.5-1.0**: Gas condensates
                - **1.0-2.0**: Black oils  
                - **2.0-4.0**: Heavy oils
                Typical value: 1.0 (exponential distribution)
                """
            )
            
            gamma_eta = st.number_input(
                "η (Minimum Molecular Weight) [g/mol]",
                min_value=60.0, max_value=120.0, value=90.0, step=2.0,
                help="""
                Minimum molecular weight in distribution:
                - **84**: C7+ characterization
                - **90**: C7+ (default)
                - **98**: C8+ characterization
                """
            )
            
            st.markdown("""
            **Gamma Distribution Formula:**
            
            $p(M) = \\frac{(M - \\eta)^{\\alpha - 1}}{\\beta^\\alpha \\cdot \\Gamma(\\alpha)} \\exp\\left(-\\frac{M - \\eta}{\\beta}\\right)$
            
            Where β is calculated from: $\\bar{M} = \\eta + \\alpha \\cdot \\beta$
            """)
        else:
            gamma_alpha = 1.0
            gamma_eta = 90.0
            
            st.info("**Pedersen Model Info**")
            st.markdown("""
            The Pedersen model uses an exponential distribution:
            
            $z_i = \\exp(A + B \\cdot i)$
            
            Where:
            - $z_i$ = mole fraction of carbon number i
            - A, B = fitted coefficients
            - i = carbon number
            
            The model automatically calculates:
            - Molar masses from PVTsim correlations
            - Densities from established correlations
            - Critical properties using Lee-Kesler relations
            """)
    
    st.divider()
    
    # Input composition section
    st.subheader("📊 Input Composition (with Plus Fraction)")
    
    # Option to extend composition to higher carbon numbers
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info("💡 The last component in the fluid table should be the Plus Fraction (e.g., C10+, C20+). Make sure 'Last component is Plus Fraction' is checked above.")
    with col2:
        extend_to_c80 = st.checkbox("Extend input table to C80", value=False,
                                    help="Add components up to C80 to the input table")
    
    if extend_to_c80:
        # Generate extended composition template
        st.subheader("Extended Composition Editor (C7-C80)")
        
        # Start with existing components that are not heavy fractions
        base_components = ['nitrogen', 'CO2', 'methane', 'ethane', 'propane', 
                           'i-butane', 'n-butane', 'i-pentane', 'n-pentane', 'n-hexane']
        
        # Generate C7 through C80
        heavy_components = [f'C{i}' for i in range(7, 81)]
        all_components = base_components + heavy_components
        
        # Create extended DataFrame
        if 'extended_fluid_df' not in st.session_state:
            extended_data = {
                'ComponentName': all_components,
                'MolarComposition[-]': [0.5, 1.5, 40.0, 8.0, 6.0, 1.5, 3.0, 1.5, 2.0, 3.0] + 
                                        [0.5] * 10 + [0.1] * 20 + [0.01] * 44,  # Decreasing composition
                'MolarMass[kg/mol]': [None] * 10 + [0.0913 + i*0.014 for i in range(74)],
                'RelativeDensity[-]': [None] * 10 + [0.746 + i*0.003 for i in range(74)]
            }
            st.session_state.extended_fluid_df = pd.DataFrame(extended_data)
        
        extended_edited_df = st.data_editor(
            st.session_state.extended_fluid_df,
            column_config={
                "ComponentName": st.column_config.TextColumn("Component", disabled=True),
                "MolarComposition[-]": st.column_config.NumberColumn("Molar Composition", min_value=0.0, format="%.6f"),
                "MolarMass[kg/mol]": st.column_config.NumberColumn("Molar Mass [kg/mol]", min_value=0.0, format="%.4f"),
                "RelativeDensity[-]": st.column_config.NumberColumn("Density [g/cm³]", min_value=0.0, format="%.4f"),
            },
            height=400,
            use_container_width=True
        )
        st.session_state.extended_fluid_df = extended_edited_df
        char_input_df = extended_edited_df[extended_edited_df['MolarComposition[-]'] > 0].copy()
    else:
        char_input_df = edited_fluid_df.copy()
    
    st.divider()
    
    # Run characterization button
    if st.button("🔬 Run Characterization", key="run_char", type="primary"):
        if char_input_df['MolarComposition[-]'].sum() > 0:
            with st.spinner("Running plus fraction characterization..."):
                try:
                    # Create characterized fluid
                    char_fluid = create_characterized_fluid(
                        char_input_df, 
                        is_plus_fraction=isplusfluid,
                        plus_fraction_model=plus_fraction_model,
                        lumping_model=lumping_model,
                        num_pseudo_components=num_pseudo,
                        gamma_alpha=gamma_alpha,
                        gamma_eta=gamma_eta,
                        eos_model=eos_model
                    )
                    
                    # Store the characterized fluid in session state
                    st.session_state.characterized_fluid = char_fluid
                    
                    # Set conditions and flash
                    char_fluid.setTemperature(reservoir_temp + 273.15, "K")
                    char_fluid.setPressure(reservoir_pres, "bara")
                    TPflash(char_fluid)
                    char_fluid.initThermoProperties()
                    char_fluid.initPhysicalProperties()
                    
                    # Count components correctly using the lumping model
                    n_total = char_fluid.getNumberOfComponents()
                    
                    # Get lumped component count from lumping model (the correct way)
                    try:
                        lumping = char_fluid.getCharacterization().getLumpingModel()
                        n_lumped = int(lumping.getNumberOfLumpedComponents())
                    except Exception:
                        n_lumped = 0
                    
                    # Count defined (non-TBP) components
                    n_defined = 0
                    for i in range(n_total):
                        comp = char_fluid.getPhase(0).getComponent(i)
                        if not (comp.isIsTBPfraction() or comp.isIsPlusFraction()):
                            n_defined += 1
                    
                    n_pseudo_actual = n_total - n_defined
                    
                    st.success(f"✅ Characterization completed using **{eos_model}** EoS! Final fluid has **{n_total}** components ({n_defined} defined + {n_pseudo_actual} pseudo-components from {n_lumped} lumps).")
                    
                    # Display characterized composition
                    st.subheader("📊 Final Characterized Fluid Composition")
                    if n_lumped > 0:
                        st.info(f"Plus fraction expanded and lumped into **{n_lumped} pseudo-components** ready for PVT simulations.")
                    else:
                        st.info(f"Fluid has **{n_pseudo_actual} pseudo-components** (no lumping applied).")
                    
                    # Build composition table
                    comp_data = []
                    for i in range(char_fluid.getNumberOfComponents()):
                        comp = char_fluid.getPhase(0).getComponent(i)
                        comp_data.append({
                            'Component': str(comp.getComponentName()),  # Convert Java String to Python str
                            'Mole Fraction [-]': float(comp.getz()),
                            'Molar Mass [kg/mol]': float(comp.getMolarMass()),
                            'Tc [K]': float(comp.getTC()),
                            'Pc [bara]': float(comp.getPC()),
                            'Acentric Factor': float(comp.getAcentricFactor()),
                            'Is Pseudo': bool(comp.isIsPlusFraction() or comp.isIsTBPfraction())
                        })
                    
                    char_comp_df = pd.DataFrame(comp_data)
                    
                    # Display in columns
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.dataframe(
                            char_comp_df.style.format({
                                'Mole Fraction [-]': '{:.6f}',
                                'Molar Mass [kg/mol]': '{:.4f}',
                                'Tc [K]': '{:.2f}',
                                'Pc [bara]': '{:.2f}',
                                'Acentric Factor': '{:.4f}'
                            }),
                            use_container_width=True,
                            height=400
                        )
                    
                    with col2:
                        st.metric("Total Components", n_total)
                        st.metric("Defined Components", n_defined)
                        st.metric("Pseudo-components", n_pseudo_actual)
                        
                        # Calculate mole fraction sum for verification
                        mol_sum = sum(c['Mole Fraction [-]'] for c in comp_data)
                        st.metric("Σ Mole Fractions", f"{mol_sum:.6f}")
                        
                        # Show lumping model info (the correct way)
                        if n_lumped > 0:
                            st.metric("Lumped Groups", n_lumped)
                            st.markdown("**Lumped Component Names:**")
                            for i in range(min(n_lumped, 10)):  # Show max 10
                                try:
                                    name = str(lumping.getLumpedComponentName(i))
                                    st.text(f"  {i+1}. {name}")
                                except Exception:
                                    pass
                    
                    # Visualization of mole fraction distribution
                    st.subheader("📈 Mole Fraction Distribution")
                    
                    # Filter to show only characterized components
                    heavy_comps = char_comp_df[
                        (char_comp_df['Molar Mass [kg/mol]'] > 0.08) | 
                        char_comp_df['Component'].str.contains('C[0-9]', regex=True)
                    ]
                    
                    fig = make_subplots(rows=1, cols=2, 
                                        subplot_titles=('Mole Fraction vs Component', 
                                                        'Mole Fraction vs Molar Mass'))
                    
                    # Bar chart of mole fractions
                    fig.add_trace(
                        go.Bar(x=char_comp_df['Component'], 
                               y=char_comp_df['Mole Fraction [-]'],
                               name='Mole Fraction',
                               marker_color='steelblue'),
                        row=1, col=1
                    )
                    
                    # Scatter plot: Mole fraction vs Molar Mass
                    fig.add_trace(
                        go.Scatter(x=heavy_comps['Molar Mass [kg/mol]'] * 1000,  # Convert to g/mol
                                   y=heavy_comps['Mole Fraction [-]'],
                                   mode='markers+lines',
                                   name='Distribution',
                                   marker=dict(size=8, color='coral')),
                        row=1, col=2
                    )
                    
                    fig.update_layout(height=400, showlegend=False)
                    fig.update_xaxes(title_text="Component", row=1, col=1, tickangle=45)
                    fig.update_xaxes(title_text="Molar Mass [g/mol]", row=1, col=2)
                    fig.update_yaxes(title_text="Mole Fraction [-]", row=1, col=1)
                    fig.update_yaxes(title_text="Mole Fraction [-]", row=1, col=2, type="log")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Flash results at reservoir conditions
                    st.subheader("⚡ Flash Results at Reservoir Conditions")
                    st.markdown(f"**T = {reservoir_temp}°C, P = {reservoir_pres} bara**")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Number of Phases", char_fluid.getNumberOfPhases())
                    col2.metric("Vapor Fraction", f"{char_fluid.getBeta():.4f}")
                    
                    if char_fluid.hasPhaseType("gas"):
                        gas_rho = char_fluid.getPhase("gas").getDensity("kg/m3")
                        col3.metric("Gas Density", f"{gas_rho:.2f} kg/m³")
                    
                    if char_fluid.hasPhaseType("oil"):
                        oil_rho = char_fluid.getPhase("oil").getDensity("kg/m3")
                        col4.metric("Oil Density", f"{oil_rho:.2f} kg/m³")
                    
                    # Download characterized composition
                    st.subheader("📥 Export Characterized Fluid")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Full properties export
                        st.download_button(
                            label="📥 Download Full Properties (CSV)",
                            data=char_comp_df.to_csv(index=False),
                            file_name="characterized_fluid_full.csv",
                            mime="text/csv",
                            help="Download all component properties (Tc, Pc, acentric factor, etc.)"
                        )
                    
                    with col2:
                        # Simplified composition export (for import into other tools)
                        simple_df = char_comp_df[['Component', 'Mole Fraction [-]', 'Molar Mass [kg/mol]']].copy()
                        simple_df.columns = ['ComponentName', 'MolarComposition[-]', 'MolarMass[kg/mol]']
                        st.download_button(
                            label="📥 Download Composition Only (CSV)",
                            data=simple_df.to_csv(index=False),
                            file_name="characterized_fluid_composition.csv",
                            mime="text/csv",
                            help="Download simplified composition for import into NeqSim or other simulators"
                        )
                    
                except Exception as e:
                    st.error(f"Error during characterization: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.warning("Please enter a valid fluid composition (molar sum > 0)")

# =============================================================================
# TAB 1: CCE (Constant Composition Expansion)
# =============================================================================
with tab_cce:
    st.header("Constant Composition Expansion (CCE)")
    st.markdown("""
    CCE simulates isothermal depressurization without removing any material. 
    Used to determine **bubble/dew point pressure**, **relative volume**, and **fluid compressibility**.
    
    **Key Outputs:**
    - Saturation pressure (bubble/dew point)
    - Relative volume (V/V_sat)
    - Y-factor (above saturation)
    - Isothermal compressibility
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        cce_temp = st.number_input("Temperature (°C)", value=reservoir_temp, key="cce_temp")
    with col2:
        cce_pres_range = st.slider("Pressure Range (bara)", min_value=1.0, max_value=500.0, 
                                    value=(10.0, reservoir_pres), key="cce_pres_range")
    
    num_pressure_steps = st.slider("Number of Pressure Steps", min_value=5, max_value=50, value=20, key="cce_steps")
    
    if st.button("🚀 Run CCE Simulation", key="run_cce"):
        if edited_fluid_df['MolarComposition[-]'].sum() > 0:
            with st.spinner("Running CCE simulation..."):
                try:
                    # Create fluid
                    fluid = create_neqsim_fluid(edited_fluid_df, isplusfluid)
                    
                    # Create CCE simulation
                    cce = jneqsim.pvtsimulation.simulation.ConstantMassExpansion(fluid)
                    cce.setTemperature(cce_temp + 273.15, "K")
                    
                    # Set pressure steps
                    pressures = np.linspace(cce_pres_range[1], cce_pres_range[0], num_pressure_steps).tolist()
                    cce.setPressures(pressures)
                    
                    # Run simulation
                    cce.runCalc()
                    
                    # Get results
                    p_results = list(cce.getPressures())
                    rel_vol = list(cce.getRelativeVolume())
                    y_factor = list(cce.getYfactor())
                    density = list(cce.getDensity())
                    z_gas = list(cce.getZgas())
                    sat_pressure = cce.getSaturationPressure()
                    
                    # Create results DataFrame
                    cce_results = pd.DataFrame({
                        'Pressure [bara]': p_results,
                        'Relative Volume [-]': rel_vol,
                        'Y-Factor [-]': y_factor,
                        'Density [kg/m³]': density,
                        'Z-gas [-]': z_gas
                    })
                    
                    # Display saturation pressure
                    st.success(f"✅ CCE simulation completed! Saturation Pressure: **{sat_pressure:.2f} bara**")
                    
                    # Create plots
                    fig = make_subplots(rows=2, cols=2, 
                                        subplot_titles=('Relative Volume vs Pressure', 
                                                       'Y-Factor vs Pressure',
                                                       'Density vs Pressure',
                                                       'Z-gas vs Pressure'))
                    
                    # Plot 1: Relative Volume
                    fig.add_trace(go.Scatter(x=p_results, y=rel_vol, mode='lines+markers', 
                                            name='Relative Volume', line=dict(color='blue')), row=1, col=1)
                    fig.add_vline(x=sat_pressure, line_dash="dash", line_color="red", row=1, col=1)
                    
                    # Plot 2: Y-Factor
                    valid_y = [(p, y) for p, y in zip(p_results, y_factor) if y > 0]
                    if valid_y:
                        fig.add_trace(go.Scatter(x=[v[0] for v in valid_y], y=[v[1] for v in valid_y], 
                                                mode='lines+markers', name='Y-Factor', line=dict(color='green')), row=1, col=2)
                    
                    # Plot 3: Density
                    fig.add_trace(go.Scatter(x=p_results, y=density, mode='lines+markers', 
                                            name='Density', line=dict(color='orange')), row=2, col=1)
                    
                    # Plot 4: Z-gas
                    valid_z = [(p, z) for p, z in zip(p_results, z_gas) if z > 0]
                    if valid_z:
                        fig.add_trace(go.Scatter(x=[v[0] for v in valid_z], y=[v[1] for v in valid_z], 
                                                mode='lines+markers', name='Z-gas', line=dict(color='purple')), row=2, col=2)
                    
                    fig.update_layout(height=700, showlegend=False, 
                                     title_text=f"CCE Results at {cce_temp}°C (Psat = {sat_pressure:.2f} bara)")
                    fig.update_xaxes(title_text="Pressure [bara]")
                    fig.update_yaxes(title_text="Relative Volume [-]", row=1, col=1)
                    fig.update_yaxes(title_text="Y-Factor [-]", row=1, col=2)
                    fig.update_yaxes(title_text="Density [kg/m³]", row=2, col=1)
                    fig.update_yaxes(title_text="Z-gas [-]", row=2, col=2)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display results table
                    st.subheader("📋 Results Table")
                    st.dataframe(cce_results.round(4), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error running CCE simulation: {e}")
        else:
            st.warning("Please enter a valid fluid composition (sum > 0)")

# =============================================================================
# TAB 2: CVD (Constant Volume Depletion)
# =============================================================================
with tab_cvd:
    st.header("Constant Volume Depletion (CVD)")
    st.markdown("""
    CVD is the standard experiment for **gas condensate reservoirs**. Gas is removed at each pressure 
    step to maintain constant cell volume, simulating reservoir depletion behavior.
    
    **Key Outputs:**
    - Liquid dropout curve (retrograde condensation)
    - Gas Z-factor
    - Cumulative moles depleted
    - Relative volume
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        cvd_temp = st.number_input("Temperature (°C)", value=reservoir_temp, key="cvd_temp")
    with col2:
        cvd_pres_range = st.slider("Pressure Range (bara)", min_value=1.0, max_value=500.0, 
                                    value=(10.0, reservoir_pres), key="cvd_pres_range")
    
    cvd_steps = st.slider("Number of Pressure Steps", min_value=5, max_value=30, value=15, key="cvd_steps")
    
    if st.button("🚀 Run CVD Simulation", key="run_cvd"):
        if edited_fluid_df['MolarComposition[-]'].sum() > 0:
            with st.spinner("Running CVD simulation..."):
                try:
                    # Create fluid
                    fluid = create_neqsim_fluid(edited_fluid_df, isplusfluid)
                    
                    # Create CVD simulation
                    cvd = jneqsim.pvtsimulation.simulation.ConstantVolumeDepletion(fluid)
                    cvd.setTemperature(cvd_temp + 273.15, "K")
                    
                    # Set pressure steps (decreasing)
                    pressures = np.linspace(cvd_pres_range[1], cvd_pres_range[0], cvd_steps).tolist()
                    cvd.setPressures(pressures)
                    
                    # Run simulation
                    cvd.runCalc()
                    
                    # Get results
                    p_results = list(cvd.getPressures())
                    rel_vol = list(cvd.getRelativeVolume())
                    liq_dropout = list(cvd.getLiquidRelativeVolume())
                    z_gas = list(cvd.getZgas())
                    cum_depleted = list(cvd.getCummulativeMolePercDepleted())
                    sat_pressure = cvd.getSaturationPressure()
                    
                    st.success(f"✅ CVD simulation completed! Saturation Pressure (Dew Point): **{sat_pressure:.2f} bara**")
                    
                    # Create plots
                    fig = make_subplots(rows=2, cols=2, 
                                        subplot_titles=('Liquid Dropout vs Pressure', 
                                                       'Gas Z-Factor vs Pressure',
                                                       'Relative Volume vs Pressure',
                                                       'Cumulative Moles Depleted vs Pressure'))
                    
                    # Plot 1: Liquid Dropout (retrograde condensation curve)
                    valid_liq = [(p, l) for p, l in zip(p_results, liq_dropout) if l >= 0]
                    if valid_liq:
                        fig.add_trace(go.Scatter(x=[v[0] for v in valid_liq], y=[v[1] for v in valid_liq], 
                                                mode='lines+markers', name='Liquid Dropout', 
                                                line=dict(color='green'), fill='tozeroy'), row=1, col=1)
                    fig.add_vline(x=sat_pressure, line_dash="dash", line_color="red", row=1, col=1)
                    
                    # Plot 2: Z-gas
                    valid_z = [(p, z) for p, z in zip(p_results, z_gas) if z > 0]
                    if valid_z:
                        fig.add_trace(go.Scatter(x=[v[0] for v in valid_z], y=[v[1] for v in valid_z], 
                                                mode='lines+markers', name='Z-gas', line=dict(color='blue')), row=1, col=2)
                    
                    # Plot 3: Relative Volume
                    valid_rel = [(p, r) for p, r in zip(p_results, rel_vol) if r > 0]
                    if valid_rel:
                        fig.add_trace(go.Scatter(x=[v[0] for v in valid_rel], y=[v[1] for v in valid_rel], 
                                                mode='lines+markers', name='Relative Volume', line=dict(color='orange')), row=2, col=1)
                    
                    # Plot 4: Cumulative Moles Depleted
                    valid_cum = [(p, c) for p, c in zip(p_results, cum_depleted) if c >= 0]
                    if valid_cum:
                        fig.add_trace(go.Scatter(x=[v[0] for v in valid_cum], y=[v[1] for v in valid_cum], 
                                                mode='lines+markers', name='Cum. Depleted', line=dict(color='purple')), row=2, col=2)
                    
                    fig.update_layout(height=700, showlegend=False, 
                                     title_text=f"CVD Results at {cvd_temp}°C (Psat = {sat_pressure:.2f} bara)")
                    fig.update_xaxes(title_text="Pressure [bara]")
                    fig.update_yaxes(title_text="Liquid Dropout [%]", row=1, col=1)
                    fig.update_yaxes(title_text="Z-gas [-]", row=1, col=2)
                    fig.update_yaxes(title_text="Relative Volume [-]", row=2, col=1)
                    fig.update_yaxes(title_text="Cumulative Depleted [mol%]", row=2, col=2)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display results table
                    st.subheader("📋 Results Table")
                    cvd_results = pd.DataFrame({
                        'Pressure [bara]': p_results,
                        'Liquid Dropout [%]': liq_dropout,
                        'Z-gas [-]': z_gas,
                        'Relative Volume [-]': rel_vol,
                        'Cumulative Depleted [mol%]': cum_depleted
                    })
                    st.dataframe(cvd_results.round(4), use_container_width=True)
                    
                    # Key metrics
                    col1, col2, col3 = st.columns(3)
                    max_liq = max([v[1] for v in valid_liq]) if valid_liq else 0
                    max_liq_pres = [v[0] for v in valid_liq if v[1] == max_liq][0] if valid_liq and max_liq > 0 else 0
                    col1.metric("Max Liquid Dropout", f"{max_liq:.2f}%")
                    col2.metric("@ Pressure", f"{max_liq_pres:.1f} bara")
                    if valid_cum:
                        col3.metric("Final Depletion", f"{valid_cum[-1][1]:.1f} mol%")
                    
                    # Download button
                    st.download_button(
                        label="📥 Download CVD Results (CSV)",
                        data=cvd_results.to_csv(index=False),
                        file_name="cvd_results.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error running CVD simulation: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.warning("Please enter a valid fluid composition (sum > 0)")

# =============================================================================
# TAB 3: Differential Liberation
# =============================================================================
with tab_dl:
    st.header("Differential Liberation (DL)")
    st.markdown("""
    Differential Liberation simulates reservoir depletion by **removing gas at each pressure step**.
    This mimics the behavior of gas escaping from oil in the reservoir.
    
    **Key Outputs:**
    - Bo (Oil Formation Volume Factor)
    - Rs (Solution Gas-Oil Ratio)
    - Bg (Gas Formation Volume Factor)
    - Oil density
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        dl_temp = st.number_input("Temperature (°C)", value=reservoir_temp, key="dl_temp")
    with col2:
        dl_pres_range = st.slider("Pressure Range (bara)", min_value=1.0, max_value=500.0, 
                                   value=(1.0, reservoir_pres), key="dl_pres_range")
    
    dl_steps = st.slider("Number of Pressure Steps", min_value=5, max_value=30, value=15, key="dl_steps")
    
    if st.button("🚀 Run Differential Liberation", key="run_dl"):
        if edited_fluid_df['MolarComposition[-]'].sum() > 0:
            with st.spinner("Running Differential Liberation..."):
                try:
                    # Create fluid
                    fluid = create_neqsim_fluid(edited_fluid_df, isplusfluid)
                    
                    # Create DL simulation
                    dl = jneqsim.pvtsimulation.simulation.DifferentialLiberation(fluid)
                    dl.setTemperature(dl_temp + 273.15, "K")
                    
                    # Set pressure steps
                    pressures = np.linspace(dl_pres_range[1], dl_pres_range[0], dl_steps).tolist()
                    dl.setPressures(pressures)
                    
                    # Run simulation
                    dl.runCalc()
                    
                    # Get results
                    p_results = list(dl.getPressures())
                    bo = list(dl.getBo())
                    rs = list(dl.getRs())
                    bg = list(dl.getBg())
                    oil_density = list(dl.getOilDensity())
                    sat_pressure = dl.getSaturationPressure()
                    
                    st.success(f"✅ Differential Liberation completed! Saturation Pressure: **{sat_pressure:.2f} bara**")
                    
                    # Create plots
                    fig = make_subplots(rows=2, cols=2,
                                        subplot_titles=('Bo (Oil FVF) vs Pressure',
                                                       'Rs (Solution GOR) vs Pressure',
                                                       'Bg (Gas FVF) vs Pressure',
                                                       'Oil Density vs Pressure'))
                    
                    # Plot Bo
                    valid_bo = [(p, b) for p, b in zip(p_results, bo) if b > 0]
                    if valid_bo:
                        fig.add_trace(go.Scatter(x=[v[0] for v in valid_bo], y=[v[1] for v in valid_bo],
                                                mode='lines+markers', name='Bo', line=dict(color='blue')), row=1, col=1)
                    
                    # Plot Rs
                    valid_rs = [(p, r) for p, r in zip(p_results, rs) if r >= 0]
                    if valid_rs:
                        fig.add_trace(go.Scatter(x=[v[0] for v in valid_rs], y=[v[1] for v in valid_rs],
                                                mode='lines+markers', name='Rs', line=dict(color='green')), row=1, col=2)
                    
                    # Plot Bg
                    valid_bg = [(p, b) for p, b in zip(p_results, bg) if b > 0]
                    if valid_bg:
                        fig.add_trace(go.Scatter(x=[v[0] for v in valid_bg], y=[v[1] for v in valid_bg],
                                                mode='lines+markers', name='Bg', line=dict(color='orange')), row=2, col=1)
                    
                    # Plot Oil Density
                    valid_density = [(p, d) for p, d in zip(p_results, oil_density) if d > 0]
                    if valid_density:
                        fig.add_trace(go.Scatter(x=[v[0] for v in valid_density], y=[v[1] for v in valid_density],
                                                mode='lines+markers', name='Oil Density', line=dict(color='red')), row=2, col=2)
                    
                    fig.update_layout(height=700, showlegend=False,
                                     title_text=f"Differential Liberation at {dl_temp}°C")
                    fig.update_xaxes(title_text="Pressure [bara]")
                    fig.update_yaxes(title_text="Bo [m³/Sm³]", row=1, col=1)
                    fig.update_yaxes(title_text="Rs [Sm³/Sm³]", row=1, col=2)
                    fig.update_yaxes(title_text="Bg [m³/Sm³]", row=2, col=1)
                    fig.update_yaxes(title_text="Density [kg/m³]", row=2, col=2)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Results table
                    dl_results = pd.DataFrame({
                        'Pressure [bara]': p_results,
                        'Bo [m³/Sm³]': bo,
                        'Rs [Sm³/Sm³]': rs,
                        'Bg [m³/Sm³]': bg,
                        'Oil Density [kg/m³]': oil_density
                    })
                    
                    st.subheader("📋 Results Table")
                    st.dataframe(dl_results.round(4), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error running Differential Liberation: {e}")
        else:
            st.warning("Please enter a valid fluid composition")

# =============================================================================
# TAB 3: Separator Test
# =============================================================================
with tab_sep:
    st.header("Separator Test")
    st.markdown("""
    Separator test simulates **multi-stage separation** to optimize surface facilities.
    Determines stock tank oil properties and optimal separator conditions.
    
    **Key Outputs:**
    - GOR at each stage
    - Bo factor
    - Optimal separator conditions
    """)
    
    st.subheader("Define Separator Stages")
    
    # Default separator stages
    if 'sep_stages' not in st.session_state:
        st.session_state.sep_stages = pd.DataFrame({
            'Stage': ['HP Separator', 'LP Separator', 'Stock Tank'],
            'Pressure [bara]': [50.0, 10.0, 1.01325],
            'Temperature [°C]': [60.0, 40.0, 15.0]
        })
    
    sep_stages = st.data_editor(
        st.session_state.sep_stages,
        column_config={
            'Stage': st.column_config.TextColumn('Stage Name'),
            'Pressure [bara]': st.column_config.NumberColumn('Pressure [bara]', min_value=0.1, max_value=200.0),
            'Temperature [°C]': st.column_config.NumberColumn('Temperature [°C]', min_value=-50.0, max_value=200.0)
        },
        num_rows='dynamic',
        use_container_width=True
    )
    st.session_state.sep_stages = sep_stages
    
    if st.button("🚀 Run Separator Test", key="run_sep"):
        if edited_fluid_df['MolarComposition[-]'].sum() > 0 and len(sep_stages) > 0:
            with st.spinner("Running Separator Test..."):
                try:
                    # Create fluid
                    fluid = create_neqsim_fluid(edited_fluid_df, isplusfluid)
                    
                    # Create separator test
                    sep_test = jneqsim.pvtsimulation.simulation.SeparatorTest(fluid)
                    
                    # Set separator conditions
                    sep_pressures = sep_stages['Pressure [bara]'].tolist()
                    sep_temps = sep_stages['Temperature [°C]'].tolist()
                    
                    # Convert temps to Kelvin for separator test
                    sep_temps_k = [t + 273.15 for t in sep_temps]
                    
                    sep_test.setSeparatorConditions(sep_temps_k, sep_pressures)
                    sep_test.runCalc()
                    
                    # Get results
                    gor = list(sep_test.getGOR())
                    bo = list(sep_test.getBofactor())
                    
                    st.success("✅ Separator Test completed!")
                    
                    # Create visualization
                    fig = make_subplots(rows=1, cols=2,
                                        subplot_titles=('GOR at Each Stage', 'Bo Factor at Each Stage'))
                    
                    stage_names = sep_stages['Stage'].tolist()
                    
                    # GOR bar chart
                    fig.add_trace(go.Bar(x=stage_names[:len(gor)], y=gor, name='GOR',
                                        marker_color='steelblue'), row=1, col=1)
                    
                    # Bo bar chart
                    fig.add_trace(go.Bar(x=stage_names[:len(bo)], y=bo, name='Bo',
                                        marker_color='coral'), row=1, col=2)
                    
                    fig.update_layout(height=400, showlegend=False,
                                     title_text="Separator Test Results")
                    fig.update_yaxes(title_text="GOR [Sm³/Sm³]", row=1, col=1)
                    fig.update_yaxes(title_text="Bo [m³/Sm³]", row=1, col=2)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Results table
                    sep_results = pd.DataFrame({
                        'Stage': stage_names[:min(len(gor), len(stage_names))],
                        'Pressure [bara]': sep_pressures[:min(len(gor), len(sep_pressures))],
                        'Temperature [°C]': sep_temps[:min(len(gor), len(sep_temps))],
                        'GOR [Sm³/Sm³]': gor[:min(len(gor), len(stage_names))],
                        'Bo [m³/Sm³]': bo[:min(len(bo), len(stage_names))]
                    })
                    
                    st.subheader("📋 Results Table")
                    st.dataframe(sep_results.round(4), use_container_width=True)
                    
                    # Total GOR and final Bo
                    if len(gor) > 0:
                        st.metric("Total GOR", f"{sum(gor):.2f} Sm³/Sm³")
                    if len(bo) > 0:
                        st.metric("Final Bo", f"{bo[-1]:.4f} m³/Sm³")
                    
                except Exception as e:
                    st.error(f"Error running Separator Test: {e}")
        else:
            st.warning("Please enter a valid fluid composition and separator stages")

# =============================================================================
# TAB 4: Swelling Test
# =============================================================================
with tab_swell:
    st.header("Swelling Test")
    st.markdown("""
    Swelling test evaluates the effect of **gas injection** on oil properties.
    Critical for EOR (Enhanced Oil Recovery) studies with CO₂ or hydrocarbon gas injection.
    
    **Key Outputs:**
    - Swelling factor (relative oil volume)
    - Saturation pressure change
    - Miscibility assessment
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        swell_temp = st.number_input("Temperature (°C)", value=reservoir_temp, key="swell_temp")
        injection_gas = st.selectbox("Injection Gas", ["CO2", "methane", "nitrogen", "ethane"], key="swell_gas")
    with col2:
        max_injection = st.slider("Max Gas Injection (mol%)", min_value=5.0, max_value=50.0, value=30.0)
        injection_steps = st.slider("Injection Steps", min_value=3, max_value=15, value=7)
    
    if st.button("🚀 Run Swelling Test", key="run_swell"):
        if edited_fluid_df['MolarComposition[-]'].sum() > 0:
            with st.spinner("Running Swelling Test..."):
                try:
                    # Create oil fluid
                    oil = create_neqsim_fluid(edited_fluid_df, isplusfluid)
                    
                    # Create injection gas fluid
                    from neqsim.thermo import fluid
                    inj_gas = fluid("srk")
                    inj_gas.addComponent(injection_gas, 100.0)
                    inj_gas.createDatabase(True)
                    inj_gas.setMixingRule(2)
                    
                    # Create swelling test
                    swell = jneqsim.pvtsimulation.simulation.SwellingTest(oil)
                    swell.setInjectionGas(inj_gas)
                    swell.setTemperature(swell_temp + 273.15, "K")
                    
                    # Set injection amounts
                    mol_percent_injected = np.linspace(0, max_injection, injection_steps).tolist()
                    swell.setCummulativeMolePercentGasInjected(mol_percent_injected)
                    
                    # Run simulation
                    swell.runCalc()
                    
                    # Get results
                    pressures = list(swell.getPressures())
                    rel_oil_vol = list(swell.getRelativeOilVolume())
                    
                    st.success(f"✅ Swelling Test completed with {injection_gas} injection!")
                    
                    # Create plots
                    fig = make_subplots(rows=1, cols=2,
                                        subplot_titles=('Saturation Pressure vs Gas Injection',
                                                       'Relative Oil Volume vs Gas Injection'))
                    
                    # Saturation pressure
                    fig.add_trace(go.Scatter(x=mol_percent_injected, y=pressures,
                                            mode='lines+markers', name='Psat',
                                            line=dict(color='red', width=2)), row=1, col=1)
                    
                    # Relative oil volume (swelling factor)
                    fig.add_trace(go.Scatter(x=mol_percent_injected, y=rel_oil_vol,
                                            mode='lines+markers', name='Swelling Factor',
                                            line=dict(color='blue', width=2)), row=1, col=2)
                    
                    fig.update_layout(height=400, showlegend=False,
                                     title_text=f"Swelling Test with {injection_gas} at {swell_temp}°C")
                    fig.update_xaxes(title_text=f"{injection_gas} Injection [mol%]")
                    fig.update_yaxes(title_text="Saturation Pressure [bara]", row=1, col=1)
                    fig.update_yaxes(title_text="Relative Oil Volume [-]", row=1, col=2)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Results table
                    swell_results = pd.DataFrame({
                        f'{injection_gas} Injection [mol%]': mol_percent_injected,
                        'Saturation Pressure [bara]': pressures,
                        'Relative Oil Volume [-]': rel_oil_vol
                    })
                    
                    st.subheader("📋 Results Table")
                    st.dataframe(swell_results.round(4), use_container_width=True)
                    
                    # Key metrics
                    if len(rel_oil_vol) > 1:
                        max_swell = max(rel_oil_vol)
                        st.metric("Maximum Swelling Factor", f"{max_swell:.3f}")
                    if len(pressures) > 1:
                        pres_increase = pressures[-1] - pressures[0]
                        st.metric("Pressure Increase", f"{pres_increase:.2f} bara")
                    
                except Exception as e:
                    st.error(f"Error running Swelling Test: {e}")
        else:
            st.warning("Please enter a valid fluid composition")

# =============================================================================
# TAB 5: GOR/Bo Curves
# =============================================================================
with tab_gorbo:
    st.header("GOR/Bo Curves")
    st.markdown("""
    Generate **Solution Gas-Oil Ratio (GOR)** and **Formation Volume Factor (Bo)** curves 
    as functions of pressure. Essential for reservoir simulation and production forecasting.
    
    **Key Outputs:**
    - GOR vs Pressure curve
    - Bo vs Pressure curve
    - Reservoir fluid characterization
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        gorbo_temp = st.number_input("Temperature (°C)", value=reservoir_temp, key="gorbo_temp")
    with col2:
        gorbo_pres_range = st.slider("Pressure Range (bara)", min_value=1.0, max_value=500.0,
                                      value=(1.0, reservoir_pres), key="gorbo_pres_range")
    
    gorbo_steps = st.slider("Number of Pressure Points", min_value=10, max_value=50, value=25, key="gorbo_steps")
    
    if st.button("🚀 Calculate GOR/Bo Curves", key="run_gorbo"):
        if edited_fluid_df['MolarComposition[-]'].sum() > 0:
            with st.spinner("Calculating GOR/Bo curves..."):
                try:
                    # Create fluid
                    fluid = create_neqsim_fluid(edited_fluid_df, isplusfluid)
                    
                    # Create GOR simulation
                    gor_sim = jneqsim.pvtsimulation.simulation.GOR(fluid)
                    
                    # Set pressure and temperature arrays (both must be same length)
                    pressures = np.linspace(gorbo_pres_range[1], gorbo_pres_range[0], gorbo_steps).tolist()
                    temperatures = [gorbo_temp + 273.15] * len(pressures)  # Same temperature for all pressure points
                    gor_sim.setTemperaturesAndPressures(temperatures, pressures)
                    
                    # Run simulation
                    gor_sim.runCalc()
                    
                    # Get results - use the pressure array we set
                    p_results = pressures
                    gor = list(gor_sim.getGOR())
                    bo = list(gor_sim.getBofactor())
                    
                    # Ensure all arrays have the same length (use the minimum length)
                    min_len = min(len(p_results), len(gor), len(bo))
                    p_results = p_results[:min_len]
                    gor = gor[:min_len]
                    bo = bo[:min_len]
                    
                    st.success("✅ GOR/Bo calculation completed!")
                    
                    # Create combined plot
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Add GOR trace
                    valid_gor = [(p, g) for p, g in zip(p_results, gor) if g >= 0]
                    if valid_gor:
                        fig.add_trace(go.Scatter(x=[v[0] for v in valid_gor], y=[v[1] for v in valid_gor],
                                                mode='lines+markers', name='GOR',
                                                line=dict(color='green', width=2)), secondary_y=False)
                    
                    # Add Bo trace
                    valid_bo = [(p, b) for p, b in zip(p_results, bo) if b > 0]
                    if valid_bo:
                        fig.add_trace(go.Scatter(x=[v[0] for v in valid_bo], y=[v[1] for v in valid_bo],
                                                mode='lines+markers', name='Bo',
                                                line=dict(color='blue', width=2)), secondary_y=True)
                    
                    fig.update_layout(height=500,
                                     title_text=f"GOR and Bo vs Pressure at {gorbo_temp}°C",
                                     legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
                    fig.update_xaxes(title_text="Pressure [bara]")
                    fig.update_yaxes(title_text="GOR [Sm³/Sm³]", secondary_y=False, color="green")
                    fig.update_yaxes(title_text="Bo [m³/Sm³]", secondary_y=True, color="blue")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Individual plots
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_gor = go.Figure()
                        if valid_gor:
                            fig_gor.add_trace(go.Scatter(x=[v[0] for v in valid_gor], y=[v[1] for v in valid_gor],
                                                        mode='lines+markers', name='GOR',
                                                        fill='tozeroy', line=dict(color='green')))
                        fig_gor.update_layout(title="Solution GOR vs Pressure",
                                             xaxis_title="Pressure [bara]",
                                             yaxis_title="GOR [Sm³/Sm³]",
                                             height=350)
                        st.plotly_chart(fig_gor, use_container_width=True)
                    
                    with col2:
                        fig_bo = go.Figure()
                        if valid_bo:
                            fig_bo.add_trace(go.Scatter(x=[v[0] for v in valid_bo], y=[v[1] for v in valid_bo],
                                                       mode='lines+markers', name='Bo',
                                                       fill='tozeroy', line=dict(color='blue')))
                        fig_bo.update_layout(title="Bo (Oil FVF) vs Pressure",
                                            xaxis_title="Pressure [bara]",
                                            yaxis_title="Bo [m³/Sm³]",
                                            height=350)
                        st.plotly_chart(fig_bo, use_container_width=True)
                    
                    # Results table
                    gorbo_results = pd.DataFrame({
                        'Pressure [bara]': p_results,
                        'GOR [Sm³/Sm³]': gor,
                        'Bo [m³/Sm³]': bo
                    })
                    
                    st.subheader("📋 Results Table")
                    st.dataframe(gorbo_results.round(4), use_container_width=True)
                    
                    # Key metrics
                    col1, col2, col3 = st.columns(3)
                    if valid_gor:
                        col1.metric("Max GOR", f"{max([v[1] for v in valid_gor]):.1f} Sm³/Sm³")
                    if valid_bo:
                        col2.metric("Max Bo", f"{max([v[1] for v in valid_bo]):.4f} m³/Sm³")
                        col3.metric("Min Bo", f"{min([v[1] for v in valid_bo]):.4f} m³/Sm³")
                    
                except Exception as e:
                    st.error(f"Error calculating GOR/Bo: {e}")
        else:
            st.warning("Please enter a valid fluid composition")

# =============================================================================
# Footer
# =============================================================================
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9em;">
    <p>PVT Simulations powered by <a href="https://github.com/equinor/neqsim">NeqSim</a></p>
    <p>Reference: Whitson CH, Brulé MR. Phase Behavior. SPE Monograph Series, 2000.</p>
</div>
""", unsafe_allow_html=True)
