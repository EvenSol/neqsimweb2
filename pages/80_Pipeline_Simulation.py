"""
Pipeline Flow Simulation - NeqSim Web Application
Supports single-phase, two-phase, and three-phase flow simulations
with steady-state and dynamic (transient) capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from neqsim.thermo import fluid, fluid_df, TPflash, dataFrame
from neqsim import jneqsim
from fluids import default_fluid
from theme import apply_theme, theme_toggle
import time

st.set_page_config(
    page_title="Pipeline Simulation", 
    page_icon='images/neqsimlogocircleflat.png',
    layout="wide"
)
apply_theme()
theme_toggle()

# =============================================================================
# HEADER AND DOCUMENTATION
# =============================================================================
st.title('🔧 Pipeline Flow Simulation')

st.markdown("""
**Comprehensive pipeline simulation** supporting single-phase gas, two-phase (gas-liquid), 
and three-phase (gas-oil-water) flow with both steady-state and dynamic simulations.

**Key Features:**
- 🌡️ **Thermodynamic Models**: GERG-2008 for gas, CPA for polar/water systems, PR/SRK for oil-gas
- 📊 **Flow Models**: Beggs & Brill, Two-Fluid Model, Drift-Flux
- ⚡ **Simulation Modes**: Steady-state and transient (dynamic)
- 📈 **Flow Regime Detection**: Automatic identification of stratified, slug, annular, bubble flow
- 🎯 **Compositional Tracking**: Track gas quality changes along pipeline
""")

st.divider()

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if 'pipeline_fluid_df' not in st.session_state:
    st.session_state.pipeline_fluid_df = pd.DataFrame({
        'ComponentName': ["nitrogen", "CO2", "methane", "ethane", "propane", 
                         "i-butane", "n-butane", "i-pentane", "n-pentane", "n-hexane", "n-heptane", "n-octane", "water"],
        'MolarComposition[-]': [1.5, 2.5, 85.0, 5.0, 3.0, 0.5, 0.8, 0.3, 0.2, 0.1, 0.05, 0.05, 0.0],
        'MolarMass[kg/mol]': [None]*13,
        'RelativeDensity[-]': [None]*13
    })

if 'pipeline_results' not in st.session_state:
    st.session_state.pipeline_results = None

# =============================================================================
# SIDEBAR - QUICK SETTINGS & INFO
# =============================================================================
with st.sidebar:
    st.header("📚 Quick Reference")
    
    with st.expander("🔬 Thermodynamic Models", expanded=False):
        st.markdown("""
        | Model | Best For |
        |-------|----------|
        | **GERG-2008** | High-accuracy gas custody transfer |
        | **SRK-EoS** | General hydrocarbon systems |
        | **PR-EoS** | Oil & gas, phase envelopes |
        | **CPA-EoS** | Water, MEG, methanol systems |
        | **UMR-PRU** | Wide-range natural gas |
        """)
    
    with st.expander("🌊 Flow Regimes", expanded=False):
        st.markdown("""
        **Horizontal/Near-Horizontal:**
        - Stratified Smooth/Wavy
        - Intermittent (Slug/Plug)
        - Annular/Mist
        - Dispersed Bubble
        
        **Vertical Upward:**
        - Bubble, Slug, Churn, Annular
        """)
    
    with st.expander("📐 Typical Parameters", expanded=False):
        st.markdown("""
        | Parameter | Typical Range |
        |-----------|---------------|
        | Roughness (steel) | 15-50 µm |
        | U-value (buried) | 2-5 W/m²K |
        | U-value (subsea) | 5-25 W/m²K |
        | Max velocity (gas) | 20-25 m/s |
        | Max velocity (liquid) | 3-5 m/s |
        """)

# =============================================================================
# MAIN CONFIGURATION TABS
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧪 Fluid Composition", 
    "🔧 Pipeline Geometry", 
    "⚙️ Simulation Settings",
    "▶️ Run Simulation",
    "📊 Results & Analysis"
])

# =============================================================================
# TAB 1: FLUID COMPOSITION
# =============================================================================
with tab1:
    st.subheader("Fluid Composition & Thermodynamic Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Predefined fluid templates
        fluid_template = st.selectbox(
            "Load Fluid Template",
            ["Custom", "Lean Natural Gas", "Rich Gas/Condensate", "Wet Gas with Water", 
             "Oil with Associated Gas", "CO2-Rich (CCS)", "Hydrogen Blend"],
            help="Select a predefined composition or customize"
        )
        
        # Apply template
        if fluid_template == "Lean Natural Gas":
            template_data = {
                'ComponentName': ["nitrogen", "CO2", "methane", "ethane", "propane", "i-butane", "n-butane"],
                'MolarComposition[-]': [0.8, 1.5, 94.0, 2.5, 0.8, 0.2, 0.2],
                'MolarMass[kg/mol]': [None]*7,
                'RelativeDensity[-]': [None]*7
            }
        elif fluid_template == "Rich Gas/Condensate":
            template_data = {
                'ComponentName': ["nitrogen", "CO2", "methane", "ethane", "propane", "i-butane", "n-butane", 
                                 "i-pentane", "n-pentane", "n-hexane", "C7", "C8", "C9", "C10"],
                'MolarComposition[-]': [0.5, 3.0, 75.0, 7.0, 5.0, 1.0, 2.0, 1.0, 1.0, 1.5, 1.5, 1.0, 0.3, 0.2],
                'MolarMass[kg/mol]': [None]*10 + [0.0913, 0.1041, 0.1188, 0.136],
                'RelativeDensity[-]': [None]*10 + [0.746, 0.768, 0.79, 0.787]
            }
        elif fluid_template == "Wet Gas with Water":
            template_data = {
                'ComponentName': ["nitrogen", "CO2", "H2S", "methane", "ethane", "propane", "n-butane", "water"],
                'MolarComposition[-]': [1.0, 2.0, 0.5, 88.0, 4.0, 2.0, 1.0, 1.5],
                'MolarMass[kg/mol]': [None]*8,
                'RelativeDensity[-]': [None]*8
            }
        elif fluid_template == "Oil with Associated Gas":
            template_data = {
                'ComponentName': ["nitrogen", "CO2", "methane", "ethane", "propane", "i-butane", "n-butane",
                                 "i-pentane", "n-pentane", "n-hexane", "C7", "C8", "C9", "C10", "C11", "C12"],
                'MolarComposition[-]': [0.3, 1.5, 45.0, 8.0, 6.0, 1.5, 3.0, 2.0, 2.5, 4.0, 8.0, 7.0, 5.0, 4.0, 1.5, 0.7],
                'MolarMass[kg/mol]': [None]*10 + [0.0913, 0.1041, 0.1188, 0.136, 0.150, 0.164],
                'RelativeDensity[-]': [None]*10 + [0.746, 0.768, 0.79, 0.787, 0.793, 0.804]
            }
        elif fluid_template == "CO2-Rich (CCS)":
            template_data = {
                'ComponentName': ["CO2", "nitrogen", "oxygen", "methane", "water", "H2S", "argon"],
                'MolarComposition[-]': [95.0, 2.0, 0.5, 1.0, 0.5, 0.5, 0.5],
                'MolarMass[kg/mol]': [None]*7,
                'RelativeDensity[-]': [None]*7
            }
        elif fluid_template == "Hydrogen Blend":
            template_data = {
                'ComponentName': ["hydrogen", "methane", "ethane", "nitrogen", "CO2"],
                'MolarComposition[-]': [20.0, 75.0, 3.0, 1.5, 0.5],
                'MolarMass[kg/mol]': [None]*5,
                'RelativeDensity[-]': [None]*5
            }
        else:
            template_data = None
        
        if template_data and st.button("Apply Template", key="apply_template"):
            st.session_state.pipeline_fluid_df = pd.DataFrame(template_data)
            st.rerun()
    
    with col2:
        # Thermodynamic model selection
        thermo_model = st.selectbox(
            "Thermodynamic Model",
            ["Auto-Select", "GERG-2008", "SRK-EoS", "PR-EoS", "CPA-EoS", "UMR-PRU-EoS"],
            help="""
            **Auto-Select**: Automatically chooses based on composition
            - GERG-2008 for pure natural gas
            - CPA for water/MEG/methanol
            - SRK/PR for general hydrocarbons
            """
        )
        st.session_state.thermo_model = thermo_model
        
        # Plus fraction handling
        is_plus_fluid = st.checkbox(
            "Plus Fraction (C7+ characterization)",
            help="Enable if your heaviest component represents a C7+ fraction"
        )
        st.session_state.is_plus_fluid = is_plus_fluid
    
    st.divider()
    
    # Fluid composition editor
    col1, col2 = st.columns([3, 1])
    
    with col1:
        show_active_only = st.checkbox("Show only active components (> 0)")
        
        display_df = st.session_state.pipeline_fluid_df.copy()
        if show_active_only:
            display_df = display_df[display_df['MolarComposition[-]'] > 0]
        
        edited_df = st.data_editor(
            display_df,
            column_config={
                "ComponentName": st.column_config.TextColumn(
                    "Component",
                    help="Use NeqSim component names (e.g., methane, CO2, water)"
                ),
                "MolarComposition[-]": st.column_config.NumberColumn(
                    "Molar Composition",
                    min_value=0,
                    max_value=100,
                    format="%.4f"
                ),
                "MolarMass[kg/mol]": st.column_config.NumberColumn(
                    "Molar Mass [kg/mol]",
                    help="Required for plus fractions (C7+)",
                    format="%.4f"
                ),
                "RelativeDensity[-]": st.column_config.NumberColumn(
                    "Density [g/cm³]",
                    help="Required for plus fractions (C7+)",
                    format="%.3f"
                ),
            },
            num_rows='dynamic',
            width='stretch'
        )
        
        if not show_active_only:
            st.session_state.pipeline_fluid_df = edited_df
    
    with col2:
        st.markdown("**Composition Summary**")
        total_comp = edited_df['MolarComposition[-]'].sum()
        st.metric("Total", f"{total_comp:.2f}")
        
        if total_comp > 0:
            # Check for water/polar components
            polar_components = ['water', 'MEG', 'TEG', 'methanol', 'ethanol']
            has_polar = any(
                comp in edited_df['ComponentName'].values 
                for comp in polar_components 
                if edited_df[edited_df['ComponentName'] == comp]['MolarComposition[-]'].sum() > 0
            )
            
            if has_polar:
                st.info("🔬 Polar components detected - CPA-EoS recommended")
            
            # Check for hydrogen
            h2_content = edited_df[edited_df['ComponentName'] == 'hydrogen']['MolarComposition[-]'].sum()
            if h2_content > 5:
                st.info("⚡ High H₂ content - GERG-2008-H2 recommended")
        
        st.caption("💡 Composition will be normalized")

# =============================================================================
# TAB 2: PIPELINE GEOMETRY
# =============================================================================
with tab2:
    st.subheader("Pipeline Geometry Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Basic Dimensions**")
        pipe_length = st.number_input(
            "Pipeline Length [m]",
            min_value=10.0,
            max_value=1000000.0,
            value=100000.0,  # 100 km default for noticeable pressure drop
            step=10000.0,
            help="Total pipeline length in meters"
        )
        st.session_state.pipe_length = pipe_length
        
        pipe_diameter = st.number_input(
            "Inner Diameter [mm]",
            min_value=10.0,
            max_value=2000.0,
            value=500.0,  # 500 mm (0.5 m) typical for transmission pipelines
            step=50.0,
            help="Pipeline inner diameter"
        )
        st.session_state.pipe_diameter = pipe_diameter
        
        wall_roughness = st.number_input(
            "Wall Roughness [µm]",
            min_value=1.0,
            max_value=500.0,
            value=45.0,
            step=5.0,
            help="Internal surface roughness (steel: ~45µm)"
        )
        st.session_state.wall_roughness = wall_roughness
    
    with col2:
        st.markdown("**Discretization**")
        num_segments = st.slider(
            "Number of Segments",
            min_value=5,
            max_value=200,
            value=50,
            help="More segments = higher accuracy but slower"
        )
        st.session_state.num_segments = num_segments
        
        segment_length = pipe_length / num_segments
        st.info(f"📏 Segment length: {segment_length:.1f} m")
        
        st.markdown("**Pipe Schedule (Optional)**")
        pipe_schedule = st.selectbox(
            "Pipe Schedule",
            ["Custom", "Sch 40", "Sch 80", "Sch 160", "XXS"],
            help="Standard pipe schedules for wall thickness"
        )
    
    with col3:
        st.markdown("**Elevation Profile**")
        elevation_type = st.selectbox(
            "Profile Type",
            ["Horizontal", "Constant Incline", "Riser/Downhill", 
             "Terrain with Low Point", "Custom Profile"],
            help="Pipeline elevation configuration"
        )
        st.session_state.elevation_type = elevation_type
        
        if elevation_type == "Constant Incline":
            inlet_elevation = st.number_input("Inlet Elevation [m]", value=0.0)
            outlet_elevation = st.number_input("Outlet Elevation [m]", value=0.0)
            st.session_state.inlet_elevation = inlet_elevation
            st.session_state.outlet_elevation = outlet_elevation
            
        elif elevation_type == "Riser/Downhill":
            riser_height = st.number_input(
                "Height Change [m]",
                value=100.0,
                help="Positive = uphill, Negative = downhill"
            )
            st.session_state.riser_height = riser_height
            
        elif elevation_type == "Terrain with Low Point":
            low_point_position = st.slider(
                "Low Point Position [%]",
                0, 100, 50,
                help="Location of low point as % of length"
            )
            low_point_depth = st.number_input(
                "Low Point Depth [m]",
                min_value=0.0,
                value=20.0
            )
            st.session_state.low_point_position = low_point_position
            st.session_state.low_point_depth = low_point_depth
    
    st.divider()
    
    # Heat Transfer Configuration
    st.markdown("**🌡️ Heat Transfer**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        enable_heat_transfer = st.checkbox("Enable Heat Transfer", value=True)
        st.session_state.enable_heat_transfer = enable_heat_transfer
    
    if enable_heat_transfer:
        with col2:
            overall_htc = st.number_input(
                "Overall U-value [W/m²K]",
                min_value=0.1,
                max_value=100.0,
                value=5.0,
                help="Typical: Buried=2-5, Subsea insulated=1-3, Bare subsea=15-25"
            )
            st.session_state.overall_htc = overall_htc
        
        with col3:
            ambient_temp = st.number_input(
                "Ambient Temperature [°C]",
                min_value=-50.0,
                max_value=50.0,
                value=4.0,
                help="Surrounding temperature (soil/seawater)"
            )
            st.session_state.ambient_temp = ambient_temp
    
    # Elevation profile preview
    if elevation_type != "Horizontal":
        st.markdown("**Elevation Profile Preview**")
        positions = np.linspace(0, pipe_length, num_segments + 1)
        
        if elevation_type == "Constant Incline":
            elevations = np.linspace(inlet_elevation, outlet_elevation, num_segments + 1)
        elif elevation_type == "Riser/Downhill":
            elevations = np.linspace(0, riser_height, num_segments + 1)
        elif elevation_type == "Terrain with Low Point":
            # Create a valley profile
            x_norm = positions / pipe_length
            low_pos = low_point_position / 100
            elevations = -low_point_depth * np.exp(-((x_norm - low_pos) ** 2) / 0.05)
        else:
            elevations = np.zeros(num_segments + 1)
        
        fig_elev = go.Figure()
        fig_elev.add_trace(go.Scatter(
            x=positions/1000, 
            y=elevations,
            mode='lines',
            fill='tozeroy',
            name='Elevation'
        ))
        fig_elev.update_layout(
            height=200,
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis_title="Position [km]",
            yaxis_title="Elevation [m]"
        )
        st.plotly_chart(fig_elev, width='stretch')
        st.session_state.elevation_profile = elevations

# =============================================================================
# TAB 3: SIMULATION SETTINGS
# =============================================================================
with tab3:
    st.subheader("Simulation Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🌊 Flow Configuration**")
        
        flow_type = st.selectbox(
            "Number of Phases",
            ["Single Phase (Gas)", "Two Phase (Gas-Liquid)", "Three Phase (Gas-Oil-Water)"],
            help="Select the number of fluid phases"
        )
        st.session_state.flow_type = flow_type
        
        if flow_type != "Single Phase (Gas)":
            flow_model = st.selectbox(
                "Flow Model",
                ["Beggs & Brill (Industry Standard)", 
                 "TwoFluidPipe (Drift-Flux)", 
                 "Simple Pipe (Adiabatic)"],
                help="""
                **Beggs & Brill**: Most widely used, empirical correlations
                **TwoFluidPipe**: Modern drift-flux with flow regime detection
                **Simple Pipe**: Basic pressure drop calculation
                """
            )
            st.session_state.flow_model = flow_model
            
            if "TwoFluidPipe" in flow_model:
                auto_flow_regime = st.checkbox(
                    "Automatic Flow Regime Detection",
                    value=True,
                    help="Detect stratified, slug, annular, bubble flow patterns"
                )
                st.session_state.auto_flow_regime = auto_flow_regime
        else:
            st.session_state.flow_model = "OnePhasePipe"
    
    with col2:
        st.markdown("**⏱️ Simulation Mode**")
        
        sim_mode = st.radio(
            "Simulation Type",
            ["Steady State", "Transient (Dynamic)"],
            help="""
            **Steady State**: Single equilibrium calculation
            **Transient**: Time-dependent simulation for startups, shutdowns, slugging
            """
        )
        st.session_state.sim_mode = sim_mode
        
        if sim_mode == "Transient (Dynamic)":
            st.warning("⚠️ Transient simulations can be computationally intensive")
            
            total_time = st.number_input(
                "Total Simulation Time [s]",
                min_value=10.0,
                max_value=86400.0,
                value=3600.0,
                step=60.0,
                help="Maximum: 24 hours (86400 s)"
            )
            st.session_state.total_time = total_time
            
            time_step = st.number_input(
                "Output Time Step [s]",
                min_value=0.1,
                max_value=60.0,
                value=10.0,
                help="Results recorded at this interval"
            )
            st.session_state.time_step = time_step
            
            transient_scenario = st.selectbox(
                "Transient Scenario",
                ["Constant Inlet", "Flow Rate Ramp-Up", "Pressure Step Change", 
                 "Composition Change (Pigging)"],
                help="Type of transient event to simulate"
            )
            st.session_state.transient_scenario = transient_scenario
    
    st.divider()
    
    # Boundary Conditions
    st.markdown("**🎯 Boundary Conditions**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("*Inlet Conditions*")
        inlet_pressure = st.number_input(
            "Inlet Pressure [bara]",
            min_value=1.0,
            max_value=500.0,
            value=150.0,  # Higher pressure for transmission pipelines
            step=10.0
        )
        st.session_state.inlet_pressure = inlet_pressure
        
        inlet_temp = st.number_input(
            "Inlet Temperature [°C]",
            min_value=-50.0,
            max_value=200.0,
            value=50.0,  # Typical compressor discharge temperature
            step=5.0
        )
        st.session_state.inlet_temp = inlet_temp
    
    with col2:
        st.markdown("*Flow Rate*")
        flow_rate_unit = st.selectbox(
            "Flow Rate Unit",
            ["MSm³/day", "kg/s", "kg/hr", "MMscfd", "m³/hr"],
            help="MSm³/day is typical for gas transmission pipelines"
        )
        st.session_state.flow_rate_unit = flow_rate_unit
        
        # Default values based on unit - reasonable for transmission pipeline
        default_rates = {"kg/s": 100.0, "kg/hr": 360000.0, "MSm³/day": 10.0, "MMscfd": 350.0, "m³/hr": 5000.0}
        flow_rate = st.number_input(
            f"Flow Rate [{flow_rate_unit}]",
            min_value=0.001,
            max_value=1e9,
            value=default_rates.get(flow_rate_unit, 10.0),
            format="%.2f"
        )
        st.session_state.flow_rate = flow_rate
    
    with col3:
        st.markdown("*Outlet Condition*")
        outlet_bc_type = st.selectbox(
            "Outlet Boundary",
            ["Calculate (from inlet)", "Specified Pressure"],
            help="How outlet pressure is determined"
        )
        st.session_state.outlet_bc_type = outlet_bc_type
        
        if outlet_bc_type == "Specified Pressure":
            outlet_pressure = st.number_input(
                "Outlet Pressure [bara]",
                min_value=1.0,
                max_value=500.0,
                value=50.0,
                step=5.0
            )
            st.session_state.outlet_pressure = outlet_pressure

# =============================================================================
# TAB 4: RUN SIMULATION
# =============================================================================
with tab4:
    st.subheader("Run Pipeline Simulation")
    
    # Summary of configuration
    with st.expander("📋 Configuration Summary", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Fluid**")
            active_comps = st.session_state.pipeline_fluid_df[
                st.session_state.pipeline_fluid_df['MolarComposition[-]'] > 0
            ]
            st.write(f"• Components: {len(active_comps)}")
            st.write(f"• Model: {st.session_state.get('thermo_model', 'Auto-Select')}")
        
        with col2:
            st.markdown("**Pipeline**")
            st.write(f"• Length: {st.session_state.get('pipe_length', 10000):.0f} m")
            st.write(f"• Diameter: {st.session_state.get('pipe_diameter', 200):.0f} mm")
            st.write(f"• Segments: {st.session_state.get('num_segments', 50)}")
        
        with col3:
            st.markdown("**Simulation**")
            st.write(f"• Type: {st.session_state.get('sim_mode', 'Steady State')}")
            st.write(f"• Flow: {st.session_state.get('flow_type', 'Single Phase')}")
            st.write(f"• P_in: {st.session_state.get('inlet_pressure', 80):.1f} bara")
    
    st.divider()
    
    # Run button
    if st.button("▶️ Run Simulation", type="primary", width='stretch'):
        try:
            with st.spinner("Setting up simulation..."):
                # Setup database
                jneqsim.util.database.NeqSimDataBase.setCreateTemporaryTables(True)
                
                # Get configuration
                composition_df = st.session_state.pipeline_fluid_df
                thermo_model = st.session_state.get('thermo_model', 'Auto-Select')
                flow_type = st.session_state.get('flow_type', 'Single Phase (Gas)')
                flow_model_name = st.session_state.get('flow_model', 'OnePhasePipe')
                sim_mode = st.session_state.get('sim_mode', 'Steady State')
                
                # Create fluid based on model selection
                active_fluid = composition_df[composition_df['MolarComposition[-]'] > 0].copy()
                
                if thermo_model == "Auto-Select":
                    # Check for polar components
                    polar = ['water', 'MEG', 'TEG', 'methanol']
                    has_polar = any(c in active_fluid['ComponentName'].values for c in polar)
                    
                    if has_polar and active_fluid[active_fluid['ComponentName'].isin(polar)]['MolarComposition[-]'].sum() > 0.1:
                        neqsim_fluid = fluid_df(active_fluid, lastIsPlusFraction=st.session_state.get('is_plus_fluid', False))
                        neqsim_fluid.setModel("CPAs-SRK-EOS")
                        model_used = "CPA-EoS (auto-selected for polar components)"
                    elif 'hydrogen' in active_fluid['ComponentName'].values and active_fluid[active_fluid['ComponentName'] == 'hydrogen']['MolarComposition[-]'].sum() > 5:
                        neqsim_fluid = fluid("gerg-2008")
                        for _, row in active_fluid.iterrows():
                            if row['MolarComposition[-]'] > 0:
                                neqsim_fluid.addComponent(row['ComponentName'], float(row['MolarComposition[-]']))
                        model_used = "GERG-2008 (auto-selected for hydrogen)"
                    else:
                        neqsim_fluid = fluid_df(active_fluid, lastIsPlusFraction=st.session_state.get('is_plus_fluid', False))
                        neqsim_fluid.autoSelectModel()
                        model_used = f"Auto-selected: {neqsim_fluid.getModelName()}"
                elif thermo_model == "GERG-2008":
                    neqsim_fluid = fluid("gerg-2008")
                    for _, row in active_fluid.iterrows():
                        if row['MolarComposition[-]'] > 0:
                            neqsim_fluid.addComponent(row['ComponentName'], float(row['MolarComposition[-]']))
                    model_used = "GERG-2008"
                elif thermo_model == "CPA-EoS":
                    neqsim_fluid = fluid_df(active_fluid, lastIsPlusFraction=st.session_state.get('is_plus_fluid', False))
                    neqsim_fluid.setModel("CPAs-SRK-EOS")
                    model_used = "CPA-EoS"
                elif thermo_model == "PR-EoS":
                    neqsim_fluid = fluid_df(active_fluid, lastIsPlusFraction=st.session_state.get('is_plus_fluid', False))
                    neqsim_fluid.setModel("PrEos")
                    model_used = "PR-EoS"
                elif thermo_model == "UMR-PRU-EoS":
                    neqsim_fluid = fluid_df(active_fluid, lastIsPlusFraction=st.session_state.get('is_plus_fluid', False))
                    neqsim_fluid.setModel("UMR-PRU-EoS")
                    model_used = "UMR-PRU-EoS"
                else:
                    neqsim_fluid = fluid_df(active_fluid, lastIsPlusFraction=st.session_state.get('is_plus_fluid', False))
                    model_used = "SRK-EoS"
                
                st.info(f"🔬 Thermodynamic Model: {model_used}")
                
                # Set inlet conditions FIRST
                neqsim_fluid.setTemperature(float(st.session_state.get('inlet_temp', 40)), 'C')
                neqsim_fluid.setPressure(float(st.session_state.get('inlet_pressure', 80)), 'bara')
                
                # Set flow rate BEFORE TPflash (important for proper initialization)
                flow_rate = st.session_state.get('flow_rate', 10)  # Default 10 kg/s is more reasonable
                flow_unit = st.session_state.get('flow_rate_unit', 'kg/s')
                
                # Convert flow rate units
                if flow_unit == "kg/s":
                    neqsim_fluid.setTotalFlowRate(float(flow_rate), "kg/sec")
                elif flow_unit == "kg/hr":
                    neqsim_fluid.setTotalFlowRate(float(flow_rate), "kg/hr")
                elif flow_unit == "MSm³/day":
                    neqsim_fluid.setTotalFlowRate(float(flow_rate), "MSm3/day")
                elif flow_unit == "MMscfd":
                    neqsim_fluid.setTotalFlowRate(float(flow_rate * 28316.85), "Sm3/day")  # Convert MMscfd to Sm3/day
                else:
                    neqsim_fluid.setTotalFlowRate(float(flow_rate), "m3/hr")
                
                # NOW do TPflash and initialize physical properties
                TPflash(neqsim_fluid)
                neqsim_fluid.initPhysicalProperties()
                
                # Initialize multi-phase check for two/three phase
                if flow_type != "Single Phase (Gas)":
                    neqsim_fluid.setMultiPhaseCheck(True)
                
                # Create inlet stream
                Stream = jneqsim.process.equipment.stream.Stream
                inlet_stream = Stream("inlet", neqsim_fluid)
                inlet_stream.run()
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Running pipeline simulation..."):
                status_text.text("Creating pipeline model...")
                progress_bar.progress(10)
                
                # Get pipeline geometry
                pipe_length = st.session_state.get('pipe_length', 10000)
                pipe_diameter = st.session_state.get('pipe_diameter', 200) / 1000  # Convert to meters
                wall_roughness = st.session_state.get('wall_roughness', 45) / 1e6  # Convert to meters
                num_segments = st.session_state.get('num_segments', 50)
                
                # =========================================================
                # PHASE 1 & 2: SINGLE-PHASE AND TWO-PHASE STEADY STATE
                # =========================================================
                
                if flow_type == "Single Phase (Gas)":
                    status_text.text("Setting up single-phase gas pipeline...")
                    progress_bar.progress(20)
                    
                    # Check if heat transfer is enabled
                    enable_heat_transfer = st.session_state.get('enable_heat_transfer', False)
                    
                    # Use PipeBeggsAndBrills which works well for both single-phase and two-phase
                    # and has proper heat transfer support
                    PipeBeggsAndBrills = jneqsim.process.equipment.pipeline.PipeBeggsAndBrills
                    HeatTransferMode = jneqsim.process.equipment.pipeline.PipeBeggsAndBrills.HeatTransferMode
                    
                    # Map flow units to NeqSim format
                    neqsim_flow_unit = "kg/sec"  # Default
                    if flow_unit == "kg/s":
                        neqsim_flow_unit = "kg/sec"
                    elif flow_unit == "kg/hr":
                        neqsim_flow_unit = "kg/hr"
                    elif flow_unit == "MSm³/day":
                        neqsim_flow_unit = "MSm3/day"
                    elif flow_unit == "MMscfd":
                        # Convert MMscfd to MSm3/day (1 MMscfd ≈ 0.02832 MSm3/day)
                        flow_rate = flow_rate * 0.02832
                        neqsim_flow_unit = "MSm3/day"
                    else:
                        neqsim_flow_unit = "m3/hr"
                    
                    # Set flow rate on the inlet stream
                    inlet_stream.setFlowRate(float(flow_rate), neqsim_flow_unit)
                    inlet_stream.run()
                    
                    # Get elevation settings for angle calculation
                    elevation_type = st.session_state.get('elevation_type', 'Horizontal')
                    inlet_elev = 0.0
                    outlet_elev = 0.0
                    if elevation_type == "Constant Incline":
                        inlet_elev = st.session_state.get('inlet_elevation', 0.0)
                        outlet_elev = st.session_state.get('outlet_elevation', 0.0)
                    elif elevation_type == "Riser/Downhill":
                        outlet_elev = st.session_state.get('riser_height', 0.0)
                    
                    # Calculate elevation change and pipe angle
                    height_diff = outlet_elev - inlet_elev
                    if abs(pipe_length) > 0:
                        angle_rad = np.arctan(height_diff / pipe_length)
                        angle_deg = float(np.degrees(angle_rad))
                    else:
                        angle_deg = 0.0
                    
                    # Heat transfer settings
                    amb_temp_c = st.session_state.get('ambient_temp', 4.0)
                    htc = st.session_state.get('overall_htc', 5.0)
                    
                    # Create the pipeline using PipeBeggsAndBrills
                    pipeline = PipeBeggsAndBrills("GasPipeline", inlet_stream)
                    pipeline.setLength(float(pipe_length))
                    pipeline.setDiameter(float(pipe_diameter))
                    pipeline.setPipeWallRoughness(float(wall_roughness))
                    pipeline.setNumberOfIncrements(int(num_segments))
                    pipeline.setAngle(angle_deg)
                    pipeline.setElevation(height_diff)
                    
                    # Configure heat transfer mode
                    if enable_heat_transfer:
                        pipeline.setHeatTransferMode(HeatTransferMode.SPECIFIED_U)
                        pipeline.setHeatTransferCoefficient(float(htc))
                        pipeline.setConstantSurfaceTemperature(float(amb_temp_c), "C")
                        pipe_model_name = f"PipeBeggsAndBrills (U={htc} W/m²K)"
                    else:
                        pipeline.setHeatTransferMode(HeatTransferMode.ADIABATIC)
                        pipe_model_name = "PipeBeggsAndBrills (Adiabatic)"
                    
                    # Debug: show pipeline configuration
                    st.write(f"**Debug Info:** Length={pipe_length}m, Diameter={pipe_diameter*1000:.1f}mm, Segments={num_segments}")
                    st.write(f"**Debug Info:** Inlet P={inlet_pressure:.2f} bara, T={inlet_temp:.1f}°C, Flow={flow_rate} {neqsim_flow_unit}")
                    st.write(f"**Debug Info:** Elevation: {height_diff:.1f}m, Angle: {angle_deg:.2f}°")
                    if enable_heat_transfer:
                        st.write(f"**Debug Info:** Heat Transfer: U={htc} W/m²K, Ambient={amb_temp_c:.1f}°C")
                    
                    status_text.text("Running Beggs & Brill calculation...")
                    progress_bar.progress(40)
                    
                    # Create and run process
                    ProcessSystem = jneqsim.process.processmodel.ProcessSystem
                    process = ProcessSystem()
                    process.add(inlet_stream)
                    process.add(pipeline)
                    
                    status_text.text("Running steady-state simulation...")
                    progress_bar.progress(50)
                    
                    process.run()
                    
                    progress_bar.progress(80)
                    
                    # Get outlet stream
                    outlet_stream = pipeline.getOutletStream()
                    
                    # Debug: Check outlet stream directly
                    st.write(f"**Debug Outlet:** P={outlet_stream.getPressure('bara'):.2f} bara, T={outlet_stream.getTemperature('C'):.1f}°C")
                    
                    # Get profiles from the pipeline
                    try:
                        pressure_profile = list(pipeline.getPressureProfile("bara"))
                        temp_profile = list(pipeline.getTemperatureProfile("C"))
                        node_positions = list(np.linspace(0, pipe_length, len(pressure_profile)))
                    except Exception as e:
                        st.write(f"**Debug:** Could not get profiles: {e}")
                        pressure_profile = [inlet_stream.getPressure("bara"), outlet_stream.getPressure("bara")]
                        temp_profile = [inlet_stream.getTemperature("C"), outlet_stream.getTemperature("C")]
                        node_positions = [0, pipe_length]
                    
                    # Calculate density and velocity profiles (approximate)
                    density_profile = []
                    velocity_profile = []
                    area = np.pi * (pipe_diameter / 2) ** 2
                    mass_flow = inlet_stream.getFlowRate("kg/sec")
                    
                    for i, p in enumerate(pressure_profile):
                        # Approximate density using ideal gas behavior scaled
                        inlet_density_val = inlet_stream.getFluid().getDensity("kg/m3")
                        inlet_p_val = inlet_stream.getPressure("bara")
                        inlet_t_val = inlet_stream.getTemperature("K")
                        t_at_node = temp_profile[i] + 273.15  # Convert to K
                        rho = inlet_density_val * (p / inlet_p_val) * (inlet_t_val / t_at_node)
                        density_profile.append(rho)
                        v = mass_flow / (rho * area) if rho > 0 else 0
                        velocity_profile.append(v)
                    
                    # Debug: show first and last node values
                    if len(pressure_profile) > 1:
                        st.write(f"**Debug Profile:** First: P={pressure_profile[0]:.2f} bara, T={temp_profile[0]:.1f}°C")
                        st.write(f"**Debug Profile:** Last: P={pressure_profile[-1]:.2f} bara, T={temp_profile[-1]:.1f}°C")
                        st.write(f"**Debug Profile:** ΔP = {pressure_profile[0] - pressure_profile[-1]:.3f} bar")
                    
                    progress_bar.progress(90)
                    
                    # Get inlet and outlet values
                    inlet_p = pressure_profile[0]
                    outlet_p = pressure_profile[-1]
                    inlet_t = temp_profile[0]
                    outlet_t = temp_profile[-1]
                    inlet_density = density_profile[0]
                    outlet_density = density_profile[-1]
                    inlet_velocity = velocity_profile[0]
                    outlet_velocity = velocity_profile[-1]
                    
                    # Get mass flow from stream
                    mass_flow = inlet_stream.getFlowRate("kg/sec")
                    
                    # Try to get friction factor from the pipeline
                    try:
                        friction_factor = pipeline.getFrictionFactor()
                    except:
                        # Estimate friction factor using Colebrook approximation
                        Re = (inlet_density * inlet_velocity * pipe_diameter) / 1.5e-5  # Approximate for gas
                        if Re > 0:
                            friction_factor = 0.25 / (np.log10(wall_roughness / (3.7 * pipe_diameter) + 5.74 / Re**0.9))**2
                        else:
                            friction_factor = 0.02
                    
                    results = {
                        'model_type': pipe_model_name,
                        'model_used': model_used,
                        'pressure_profile': pressure_profile,
                        'temperature_profile': temp_profile,
                        'positions': node_positions,
                        'density_profile': density_profile,
                        'velocity_profile': velocity_profile,
                        'inlet_pressure': inlet_p,
                        'outlet_pressure': outlet_p,
                        'pressure_drop': inlet_p - outlet_p,
                        'inlet_temp': inlet_t,
                        'outlet_temp': outlet_t,
                        'mass_flow': mass_flow,
                        'inlet_density': inlet_density,
                        'outlet_density': outlet_density,
                        'inlet_velocity': inlet_velocity,
                        'outlet_velocity': outlet_velocity,
                        'pipe_length': pipe_length,
                        'pipe_diameter': pipe_diameter * 1000,
                        'num_nodes': len(pressure_profile),
                        'num_segments': num_segments,
                        'friction_factor': friction_factor,
                    }
                    
                else:
                    # TWO-PHASE / THREE-PHASE FLOW
                    status_text.text("Setting up multiphase pipeline...")
                    progress_bar.progress(20)
                    
                    if "Beggs" in flow_model_name:
                        # PHASE 2: Beggs & Brill correlation
                        PipeBeggsAndBrills = jneqsim.process.equipment.pipeline.PipeBeggsAndBrills
                        HeatTransferMode = jneqsim.process.equipment.pipeline.PipeBeggsAndBrills.HeatTransferMode
                        
                        pipeline = PipeBeggsAndBrills("MultiphasePipeline", inlet_stream)
                        pipeline.setLength(float(pipe_length))
                        pipeline.setDiameter(float(pipe_diameter))
                        pipeline.setPipeWallRoughness(float(wall_roughness))
                        pipeline.setNumberOfIncrements(int(num_segments))
                        pipeline.setAngle(0)  # Horizontal for now
                        
                        # Set elevation if specified
                        elevation_type = st.session_state.get('elevation_type', 'Horizontal')
                        height_diff = 0.0
                        if elevation_type == "Constant Incline":
                            inlet_elev = st.session_state.get('inlet_elevation', 0)
                            outlet_elev = st.session_state.get('outlet_elevation', 0)
                            height_diff = outlet_elev - inlet_elev
                            angle_rad = np.arctan(height_diff / pipe_length)
                            pipeline.setAngle(float(np.degrees(angle_rad)))
                            pipeline.setElevation(height_diff)
                        elif elevation_type == "Riser/Downhill":
                            height_diff = st.session_state.get('riser_height', 0)
                            angle_rad = np.arctan(height_diff / pipe_length)
                            pipeline.setAngle(float(np.degrees(angle_rad)))
                            pipeline.setElevation(height_diff)
                        
                        # Configure heat transfer
                        enable_heat_transfer = st.session_state.get('enable_heat_transfer', False)
                        if enable_heat_transfer:
                            htc = st.session_state.get('overall_htc', 5.0)
                            amb_temp_c = st.session_state.get('ambient_temp', 4.0)
                            pipeline.setHeatTransferMode(HeatTransferMode.SPECIFIED_U)
                            pipeline.setHeatTransferCoefficient(float(htc))
                            pipeline.setConstantSurfaceTemperature(float(amb_temp_c), "C")
                        else:
                            pipeline.setHeatTransferMode(HeatTransferMode.ADIABATIC)
                        
                        if st.session_state.get('outlet_bc_type') == "Specified Pressure":
                            pipeline.setOutletPressure(float(st.session_state.get('outlet_pressure', 50)))
                        
                        status_text.text("Running Beggs & Brill calculation...")
                        progress_bar.progress(50)
                        
                        ProcessSystem = jneqsim.process.processmodel.ProcessSystem
                        process = ProcessSystem()
                        process.add(inlet_stream)
                        process.add(pipeline)
                        process.run()
                        
                        progress_bar.progress(80)
                        
                        outlet_stream = pipeline.getOutletStream()
                        
                        # Get profiles if available
                        try:
                            pressure_profile = list(pipeline.getPressureProfile("bara"))
                            temp_profile = list(pipeline.getTemperatureProfile("C"))
                            positions = np.linspace(0, pipe_length, len(pressure_profile))
                        except:
                            pressure_profile = [inlet_stream.getPressure("bara"), outlet_stream.getPressure("bara")]
                            temp_profile = [inlet_stream.getTemperature("C"), outlet_stream.getTemperature("C")]
                            positions = [0, pipe_length]
                        
                        results = {
                            'model_type': 'Beggs & Brill',
                            'model_used': model_used,
                            'inlet_pressure': inlet_stream.getPressure("bara"),
                            'outlet_pressure': outlet_stream.getPressure("bara"),
                            'pressure_drop': inlet_stream.getPressure("bara") - outlet_stream.getPressure("bara"),
                            'inlet_temp': inlet_stream.getTemperature("C"),
                            'outlet_temp': outlet_stream.getTemperature("C"),
                            'mass_flow': inlet_stream.getFlowRate("kg/sec"),
                            'pipe_length': pipe_length,
                            'pipe_diameter': pipe_diameter * 1000,
                            'pressure_profile': pressure_profile,
                            'temperature_profile': temp_profile,
                            'positions': positions,
                        }
                        
                        # Try to get holdup
                        try:
                            holdup_profile = list(pipeline.getLiquidHoldupProfile())
                            results['holdup_profile'] = holdup_profile
                        except:
                            pass
                        
                        # Try to get flow regime
                        try:
                            flow_regime = pipeline.getFlowRegime()
                            results['flow_regime'] = flow_regime
                        except:
                            results['flow_regime'] = "Not available"
                    
                    elif "TwoFluidPipe" in flow_model_name:
                        # PHASE 3: Two-Fluid Model with drift-flux
                        status_text.text("Setting up Two-Fluid model...")
                        
                        TwoFluidPipe = jneqsim.process.equipment.pipeline.TwoFluidPipe
                        pipeline = TwoFluidPipe("TwoFluidPipeline", inlet_stream)
                        pipeline.setLength(float(pipe_length))
                        pipeline.setDiameter(float(pipe_diameter))
                        pipeline.setNumberOfSections(int(num_segments))
                        pipeline.setRoughness(float(wall_roughness))
                        
                        # Set elevation profile
                        elevation_type = st.session_state.get('elevation_type', 'Horizontal')
                        if elevation_type != "Horizontal" and 'elevation_profile' in st.session_state:
                            elevations = st.session_state.elevation_profile
                            pipeline.setElevationProfile(elevations.tolist())
                        
                        # Heat transfer
                        if st.session_state.get('enable_heat_transfer', False):
                            htc = st.session_state.get('overall_htc', 5.0)
                            amb_temp = st.session_state.get('ambient_temp', 4.0)
                            pipeline.setHeatTransferCoefficient(float(htc))
                            pipeline.setSurfaceTemperature(float(amb_temp), "C")
                        
                        if st.session_state.get('outlet_bc_type') == "Specified Pressure":
                            pipeline.setOutletPressure(float(st.session_state.get('outlet_pressure', 50)), "bara")
                        
                        status_text.text("Running Two-Fluid simulation...")
                        progress_bar.progress(50)
                        
                        pipeline.run()
                        
                        progress_bar.progress(80)
                        
                        outlet_stream = pipeline.getOutletStream()
                        
                        # Extract profiles
                        try:
                            pressure_profile = list(pipeline.getPressureProfile("bara"))
                            holdup_profile = list(pipeline.getLiquidHoldupProfile())
                            positions = np.linspace(0, pipe_length, len(pressure_profile))
                        except Exception as e:
                            pressure_profile = [inlet_stream.getPressure("bara"), outlet_stream.getPressure("bara")]
                            holdup_profile = []
                            positions = [0, pipe_length]
                        
                        results = {
                            'model_type': 'TwoFluidPipe (Drift-Flux)',
                            'model_used': model_used,
                            'inlet_pressure': inlet_stream.getPressure("bara"),
                            'outlet_pressure': outlet_stream.getPressure("bara"),
                            'pressure_drop': inlet_stream.getPressure("bara") - outlet_stream.getPressure("bara"),
                            'inlet_temp': inlet_stream.getTemperature("C"),
                            'outlet_temp': outlet_stream.getTemperature("C"),
                            'mass_flow': inlet_stream.getFlowRate("kg/sec"),
                            'pipe_length': pipe_length,
                            'pipe_diameter': pipe_diameter * 1000,
                            'pressure_profile': pressure_profile,
                            'positions': positions,
                            'holdup_profile': holdup_profile,
                        }
                        
                        # Get liquid inventory
                        try:
                            liq_inventory = pipeline.getLiquidInventory("m3")
                            results['liquid_inventory'] = liq_inventory
                        except:
                            pass
                    
                    else:
                        # Simple Pipe
                        AdiabaticPipe = jneqsim.process.equipment.pipeline.AdiabaticPipe
                        pipeline = AdiabaticPipe("SimplePipe", inlet_stream)
                        pipeline.setLength(float(pipe_length))
                        pipeline.setDiameter(float(pipe_diameter))
                        pipeline.setPipeWallRoughness(float(wall_roughness))
                        
                        ProcessSystem = jneqsim.process.processmodel.ProcessSystem
                        process = ProcessSystem()
                        process.add(inlet_stream)
                        process.add(pipeline)
                        process.run()
                        
                        outlet_stream = pipeline.getOutletStream()
                        
                        results = {
                            'model_type': 'Simple Adiabatic Pipe',
                            'model_used': model_used,
                            'inlet_pressure': inlet_stream.getPressure("bara"),
                            'outlet_pressure': outlet_stream.getPressure("bara"),
                            'pressure_drop': inlet_stream.getPressure("bara") - outlet_stream.getPressure("bara"),
                            'inlet_temp': inlet_stream.getTemperature("C"),
                            'outlet_temp': outlet_stream.getTemperature("C"),
                            'mass_flow': inlet_stream.getFlowRate("kg/sec"),
                            'pipe_length': pipe_length,
                            'pipe_diameter': pipe_diameter * 1000,
                        }
                
                progress_bar.progress(100)
                status_text.text("✅ Simulation completed!")
                
                # Store results
                st.session_state.pipeline_results = results
                st.success(f"Simulation completed successfully using {results['model_type']}")
                
                # Show key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Pressure Drop", f"{results['pressure_drop']:.2f} bar")
                with col2:
                    st.metric("Outlet Pressure", f"{results['outlet_pressure']:.2f} bara")
                with col3:
                    st.metric("Outlet Temperature", f"{results['outlet_temp']:.1f} °C")
                with col4:
                    if 'liquid_inventory' in results:
                        st.metric("Liquid Inventory", f"{results['liquid_inventory']:.2f} m³")
                    else:
                        st.metric("Mass Flow", f"{results['mass_flow']:.2f} kg/s")
                
        except Exception as e:
            st.error(f"Simulation failed: {str(e)}")
            st.exception(e)

# =============================================================================
# TAB 5: RESULTS & ANALYSIS
# =============================================================================
with tab5:
    st.subheader("Results & Analysis")
    
    if st.session_state.pipeline_results is None:
        st.info("👆 Run a simulation first to see results here")
    else:
        results = st.session_state.pipeline_results
        
        # Summary metrics
        st.markdown("### 📊 Key Results")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model", results['model_type'])
            st.metric("Pressure Drop", f"{results['pressure_drop']:.2f} bar")
        
        with col2:
            st.metric("Inlet P", f"{results['inlet_pressure']:.2f} bara")
            st.metric("Outlet P", f"{results['outlet_pressure']:.2f} bara")
        
        with col3:
            st.metric("Inlet T", f"{results['inlet_temp']:.1f} °C")
            st.metric("Outlet T", f"{results['outlet_temp']:.1f} °C")
        
        with col4:
            st.metric("Mass Flow", f"{results['mass_flow']:.2f} kg/s")
            if 'inlet_velocity' in results:
                st.metric("Inlet Velocity", f"{results['inlet_velocity']:.1f} m/s")
            if 'liquid_inventory' in results:
                st.metric("Liquid Inventory", f"{results['liquid_inventory']:.2f} m³")
        
        st.divider()
        
        # =====================================================================
        # ENHANCED PLOTS SECTION
        # =====================================================================
        if 'pressure_profile' in results and len(results['pressure_profile']) > 2:
            st.markdown("### 📈 Pipeline Profiles")
            
            positions_km = np.array(results['positions']) / 1000
            pressure_profile = np.array(results['pressure_profile'])
            
            # ---- PLOT 1: Pressure and Temperature Combined ----
            st.markdown("#### Pressure & Temperature Along Pipeline")
            
            fig_pt = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Pressure trace
            fig_pt.add_trace(
                go.Scatter(
                    x=positions_km,
                    y=pressure_profile,
                    mode='lines+markers',
                    name='Pressure',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=4),
                    hovertemplate='Position: %{x:.2f} km<br>Pressure: %{y:.2f} bara<extra></extra>'
                ),
                secondary_y=False
            )
            
            # Temperature trace
            if 'temperature_profile' in results:
                temp_profile = np.array(results['temperature_profile'])
                fig_pt.add_trace(
                    go.Scatter(
                        x=positions_km,
                        y=temp_profile,
                        mode='lines+markers',
                        name='Temperature',
                        line=dict(color='#d62728', width=3),
                        marker=dict(size=4),
                        hovertemplate='Position: %{x:.2f} km<br>Temperature: %{y:.2f} °C<extra></extra>'
                    ),
                    secondary_y=True
                )
            
            fig_pt.update_layout(
                height=450,
                title=dict(text='Pressure and Temperature Profile', x=0.5),
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                margin=dict(l=60, r=60, t=80, b=60)
            )
            fig_pt.update_xaxes(title_text="Pipeline Position [km]", showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig_pt.update_yaxes(title_text="Pressure [bara]", secondary_y=False, showgrid=True, gridwidth=1, gridcolor='lightgray', title_font=dict(color='#1f77b4'))
            fig_pt.update_yaxes(title_text="Temperature [°C]", secondary_y=True, title_font=dict(color='#d62728'))
            
            st.plotly_chart(fig_pt, use_container_width=True)
            
            # ---- PLOT 2: Pressure Gradient ----
            st.markdown("#### Pressure Gradient")
            
            # Calculate pressure gradient (bar/km)
            if len(positions_km) > 1:
                dp = np.diff(pressure_profile)  # Pressure difference between nodes
                dx = np.diff(positions_km)  # Distance between nodes (km)
                pressure_gradient = -dp / dx  # Positive = pressure drop per km
                gradient_positions = (positions_km[:-1] + positions_km[1:]) / 2  # Mid-points
                
                fig_grad = go.Figure()
                fig_grad.add_trace(
                    go.Scatter(
                        x=gradient_positions,
                        y=pressure_gradient,
                        mode='lines+markers',
                        name='Pressure Gradient',
                        line=dict(color='#9467bd', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(148, 103, 189, 0.3)',
                        hovertemplate='Position: %{x:.2f} km<br>Gradient: %{y:.4f} bar/km<extra></extra>'
                    )
                )
                
                # Add average line
                avg_gradient = (pressure_profile[0] - pressure_profile[-1]) / (positions_km[-1] - positions_km[0]) if positions_km[-1] > positions_km[0] else 0
                fig_grad.add_hline(
                    y=avg_gradient,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text=f"Average: {avg_gradient:.4f} bar/km",
                    annotation_position="top right"
                )
                
                fig_grad.update_layout(
                    height=350,
                    title=dict(text='Pressure Gradient Along Pipeline', x=0.5),
                    xaxis_title="Pipeline Position [km]",
                    yaxis_title="Pressure Gradient [bar/km]",
                    hovermode='x unified',
                    margin=dict(l=60, r=60, t=80, b=60)
                )
                
                st.plotly_chart(fig_grad, use_container_width=True)
            
            # ---- PLOT 3: Velocity and Density (for single-phase gas) ----
            if 'velocity_profile' in results and 'density_profile' in results:
                velocity_profile = np.array(results['velocity_profile'])
                density_profile = np.array(results['density_profile'])
                
                if len(velocity_profile) > 0 and np.max(velocity_profile) > 0:
                    st.markdown("#### Velocity & Density Along Pipeline")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_vel = go.Figure()
                        fig_vel.add_trace(
                            go.Scatter(
                                x=positions_km,
                                y=velocity_profile,
                                mode='lines+markers',
                                name='Velocity',
                                line=dict(color='#2ca02c', width=3),
                                marker=dict(size=4),
                                fill='tozeroy',
                                fillcolor='rgba(44, 160, 44, 0.2)',
                                hovertemplate='Position: %{x:.2f} km<br>Velocity: %{y:.2f} m/s<extra></extra>'
                            )
                        )
                        fig_vel.update_layout(
                            height=350,
                            title=dict(text='Gas Velocity Profile', x=0.5),
                            xaxis_title="Position [km]",
                            yaxis_title="Velocity [m/s]",
                            margin=dict(l=50, r=30, t=60, b=50)
                        )
                        st.plotly_chart(fig_vel, use_container_width=True)
                    
                    with col2:
                        fig_rho = go.Figure()
                        fig_rho.add_trace(
                            go.Scatter(
                                x=positions_km,
                                y=density_profile,
                                mode='lines+markers',
                                name='Density',
                                line=dict(color='#ff7f0e', width=3),
                                marker=dict(size=4),
                                fill='tozeroy',
                                fillcolor='rgba(255, 127, 14, 0.2)',
                                hovertemplate='Position: %{x:.2f} km<br>Density: %{y:.2f} kg/m³<extra></extra>'
                            )
                        )
                        fig_rho.update_layout(
                            height=350,
                            title=dict(text='Gas Density Profile', x=0.5),
                            xaxis_title="Position [km]",
                            yaxis_title="Density [kg/m³]",
                            margin=dict(l=50, r=30, t=60, b=50)
                        )
                        st.plotly_chart(fig_rho, use_container_width=True)
            
            # ---- PLOT 4: Liquid Holdup (for multiphase) ----
            if 'holdup_profile' in results and len(results.get('holdup_profile', [])) > 0:
                holdup_profile = np.array(results['holdup_profile'])
                
                st.markdown("#### Liquid Holdup Profile")
                
                fig_holdup = go.Figure()
                fig_holdup.add_trace(
                    go.Scatter(
                        x=positions_km,
                        y=holdup_profile,
                        mode='lines+markers',
                        name='Liquid Holdup',
                        line=dict(color='#17becf', width=3),
                        marker=dict(size=5),
                        fill='tozeroy',
                        fillcolor='rgba(23, 190, 207, 0.3)',
                        hovertemplate='Position: %{x:.2f} km<br>Holdup: %{y:.4f}<extra></extra>'
                    )
                )
                fig_holdup.update_layout(
                    height=350,
                    title=dict(text='Liquid Holdup Along Pipeline', x=0.5),
                    xaxis_title="Pipeline Position [km]",
                    yaxis_title="Liquid Holdup [-]",
                    yaxis=dict(range=[0, max(1.0, np.max(holdup_profile) * 1.1)]),
                    margin=dict(l=60, r=60, t=80, b=60)
                )
                st.plotly_chart(fig_holdup, use_container_width=True)
            
            # ---- PLOT 5: P-T Phase Diagram View ----
            if 'temperature_profile' in results:
                st.markdown("#### P-T Operating Line")
                
                temp_profile = np.array(results['temperature_profile'])
                
                fig_phase = go.Figure()
                
                # Operating line with color gradient based on position
                fig_phase.add_trace(
                    go.Scatter(
                        x=temp_profile,
                        y=pressure_profile,
                        mode='lines+markers',
                        name='Operating Line',
                        line=dict(color='#1f77b4', width=2),
                        marker=dict(
                            size=8,
                            color=positions_km,
                            colorscale='Viridis',
                            colorbar=dict(title="Position [km]", thickness=15),
                            showscale=True
                        ),
                        hovertemplate='T: %{x:.2f} °C<br>P: %{y:.2f} bara<extra></extra>'
                    )
                )
                
                # Mark inlet and outlet
                fig_phase.add_trace(
                    go.Scatter(
                        x=[temp_profile[0]],
                        y=[pressure_profile[0]],
                        mode='markers+text',
                        name='Inlet',
                        marker=dict(size=15, color='green', symbol='circle'),
                        text=['Inlet'],
                        textposition='top right',
                        hovertemplate='Inlet<br>T: %{x:.2f} °C<br>P: %{y:.2f} bara<extra></extra>'
                    )
                )
                fig_phase.add_trace(
                    go.Scatter(
                        x=[temp_profile[-1]],
                        y=[pressure_profile[-1]],
                        mode='markers+text',
                        name='Outlet',
                        marker=dict(size=15, color='red', symbol='square'),
                        text=['Outlet'],
                        textposition='bottom left',
                        hovertemplate='Outlet<br>T: %{x:.2f} °C<br>P: %{y:.2f} bara<extra></extra>'
                    )
                )
                
                fig_phase.update_layout(
                    height=450,
                    title=dict(text='Operating Conditions (P-T Diagram)', x=0.5),
                    xaxis_title="Temperature [°C]",
                    yaxis_title="Pressure [bara]",
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    margin=dict(l=60, r=80, t=80, b=60)
                )
                
                st.plotly_chart(fig_phase, use_container_width=True)
            
            # ---- Summary Statistics Table ----
            st.markdown("#### Pipeline Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Pressure**")
                stats_p = {
                    'Inlet': f"{pressure_profile[0]:.2f} bara",
                    'Outlet': f"{pressure_profile[-1]:.2f} bara",
                    'Total Drop': f"{pressure_profile[0] - pressure_profile[-1]:.2f} bar",
                    'Drop %': f"{100 * (pressure_profile[0] - pressure_profile[-1]) / pressure_profile[0]:.1f} %",
                    'Avg Gradient': f"{(pressure_profile[0] - pressure_profile[-1]) / (positions_km[-1]):.4f} bar/km" if positions_km[-1] > 0 else "N/A"
                }
                for key, val in stats_p.items():
                    st.write(f"• {key}: **{val}**")
            
            with col2:
                if 'temperature_profile' in results:
                    st.markdown("**Temperature**")
                    temp_profile = np.array(results['temperature_profile'])
                    stats_t = {
                        'Inlet': f"{temp_profile[0]:.1f} °C",
                        'Outlet': f"{temp_profile[-1]:.1f} °C",
                        'ΔT': f"{temp_profile[-1] - temp_profile[0]:.2f} °C",
                        'Min': f"{np.min(temp_profile):.1f} °C",
                        'Max': f"{np.max(temp_profile):.1f} °C"
                    }
                    for key, val in stats_t.items():
                        st.write(f"• {key}: **{val}**")
            
            with col3:
                if 'velocity_profile' in results and len(results['velocity_profile']) > 0:
                    st.markdown("**Velocity**")
                    velocity_profile = np.array(results['velocity_profile'])
                    stats_v = {
                        'Inlet': f"{velocity_profile[0]:.2f} m/s",
                        'Outlet': f"{velocity_profile[-1]:.2f} m/s",
                        'Min': f"{np.min(velocity_profile):.2f} m/s",
                        'Max': f"{np.max(velocity_profile):.2f} m/s",
                        'Avg': f"{np.mean(velocity_profile):.2f} m/s"
                    }
                    for key, val in stats_v.items():
                        st.write(f"• {key}: **{val}**")
                elif 'density_profile' in results and len(results['density_profile']) > 0:
                    st.markdown("**Density**")
                    density_profile = np.array(results['density_profile'])
                    stats_d = {
                        'Inlet': f"{density_profile[0]:.2f} kg/m³",
                        'Outlet': f"{density_profile[-1]:.2f} kg/m³",
                        'Min': f"{np.min(density_profile):.2f} kg/m³",
                        'Max': f"{np.max(density_profile):.2f} kg/m³"
                    }
                    for key, val in stats_d.items():
                        st.write(f"• {key}: **{val}**")
        
        # Flow regime info
        if 'flow_regime' in results:
            st.markdown(f"**Flow Regime:** {results['flow_regime']}")
        
        st.divider()
        
        # Export results
        st.markdown("### 💾 Export Results")
        
        # Create export dataframe
        export_data = {
            'Parameter': ['Model Type', 'Thermo Model', 'Pipe Length [m]', 'Pipe Diameter [mm]',
                         'Inlet Pressure [bara]', 'Outlet Pressure [bara]', 'Pressure Drop [bar]',
                         'Inlet Temperature [°C]', 'Outlet Temperature [°C]', 'Mass Flow [kg/s]'],
            'Value': [results['model_type'], results['model_used'], results['pipe_length'], 
                     results['pipe_diameter'], results['inlet_pressure'], results['outlet_pressure'],
                     results['pressure_drop'], results['inlet_temp'], results['outlet_temp'],
                     results['mass_flow']]
        }
        
        export_df = pd.DataFrame(export_data)
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(export_df, hide_index=True, width='stretch')
        
        with col2:
            csv = export_df.to_csv(index=False)
            st.download_button(
                "📥 Download Summary (CSV)",
                csv,
                "pipeline_results.csv",
                "text/csv",
                width='stretch'
            )
            
            if 'pressure_profile' in results:
                profile_df = pd.DataFrame({
                    'Position [m]': results['positions'],
                    'Pressure [bara]': results['pressure_profile'],
                })
                if 'temperature_profile' in results:
                    profile_df['Temperature [C]'] = results['temperature_profile']
                if 'holdup_profile' in results and len(results['holdup_profile']) > 0:
                    profile_df['Liquid Holdup [-]'] = results['holdup_profile']
                
                profile_csv = profile_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Profiles (CSV)",
                    profile_csv,
                    "pipeline_profiles.csv",
                    "text/csv",
                    width='stretch'
                )

# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.caption("""
**Pipeline Simulation** | NeqSim Web Application  
Models: Single-phase gas (GERG-2008), Two-phase (Beggs & Brill, Two-Fluid), Three-phase (with water cut)  
[NeqSim Documentation](https://github.com/equinor/neqsim) | [Report Issues](https://github.com/equinor/neqsimweb2/issues)
""")
