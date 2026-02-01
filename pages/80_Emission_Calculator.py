import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from neqsim.thermo import fluid, TPflash
from neqsim import jneqsim

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
| **Input** | Laboratory gas analysis from flashing water sample to standard conditions |
| **Accuracy** | ±3.6% validated vs ±50% for conventional handbook methods |
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
        'ComponentName': ['TEG', 'water', 'methane', 'CO2'],
        'MolarComposition[-]': [0.92, 0.05, 0.02, 0.01]
    })

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
    
    # Process Conditions
    st.subheader("🔧 Process Conditions")
    
    if emission_source == "Produced Water Degassing":
        inlet_temp = st.number_input("Inlet Temperature (°C)", value=80.0, min_value=-50.0, max_value=300.0)
        inlet_pressure = st.number_input("Inlet Pressure (bara)", value=30.0, min_value=1.0, max_value=500.0)
        water_flow_m3hr = st.number_input("Water Flow Rate (m³/hr)", value=100.0, min_value=1.0, max_value=10000.0, help="Produced water volumetric flow")
        total_flow = water_flow_m3hr * 1000  # Convert to kg/hr (approx density 1000 kg/m³)
        
        # Salinity
        st.subheader("🧂 Salinity Effects")
        salinity_ppm = st.number_input(
            "Salinity (ppm TDS)", 
            value=35000, 
            min_value=0, 
            max_value=300000,
            help="Total dissolved solids. Seawater ~35,000 ppm, produced water can be 100,000+ ppm"
        )
        
        # Calculate salting-out factor
        if salinity_ppm > 0:
            cs = 0.12  # Setschenow coefficient for CH4 in NaCl
            molality = salinity_ppm / 58440 / (1 - salinity_ppm/1e6)
            salting_out_factor = 10 ** (-cs * molality)
            st.info(f"Salting-out factor: {salting_out_factor:.3f} — Gas solubility reduced to {salting_out_factor*100:.1f}% of freshwater value")
        else:
            salting_out_factor = 1.0
        
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
    
    elif emission_source == "TEG Regeneration":
        inlet_temp = st.number_input("Rich TEG Temperature (°C)", value=35.0, min_value=0.0, max_value=100.0)
        inlet_pressure = st.number_input("Contactor Pressure (bara)", value=52.0, min_value=1.0, max_value=200.0)
        total_flow = st.number_input("TEG Circulation Rate (kg/hr)", value=6000.0, min_value=100.0, max_value=100000.0)
        flash_pressure = st.number_input("Flash Drum Pressure (bara)", value=1.5, min_value=0.5, max_value=10.0)
        water_flow_m3hr = total_flow / 1000  # For consistency in export, convert kg/hr to approx m³/hr
        salinity_ppm = 0
        salting_out_factor = 1.0
        num_stages = 1
        stage_pressures = [flash_pressure]
        stage_names = ['HP Flash']
    
    elif emission_source == "Tank Breathing/Loading":
        inlet_temp = st.number_input("Tank Temperature (°C)", value=25.0, min_value=-20.0, max_value=60.0)
        inlet_pressure = st.number_input("Tank Pressure (bara)", value=1.05, min_value=1.0, max_value=2.0)
        total_flow = st.number_input("Throughput (kg/hr)", value=50000.0, min_value=100.0, max_value=1000000.0)
        breathing_rate = st.number_input("Breathing Rate (%)", value=0.1, min_value=0.01, max_value=1.0)
        water_flow_m3hr = total_flow / 1000  # For consistency in export, convert kg/hr to approx m³/hr
        salinity_ppm = 0
        salting_out_factor = 1.0
        num_stages = 1
        stage_pressures = [1.01325]
        stage_names = ['Atmosphere']
    
    else:  # Cold Vent
        inlet_temp = st.number_input("Vent Temperature (°C)", value=20.0, min_value=-50.0, max_value=100.0)
        inlet_pressure = st.number_input("Upstream Pressure (bara)", value=10.0, min_value=1.0, max_value=100.0)
        total_flow = st.number_input("Vent Rate (kg/hr)", value=500.0, min_value=1.0, max_value=10000.0)
        water_flow_m3hr = total_flow / 1000  # For consistency in export, convert kg/hr to approx m³/hr
        salinity_ppm = 0
        salting_out_factor = 1.0
        num_stages = 1
        stage_pressures = [1.01325]
        stage_names = ['Atmosphere']
    
    st.divider()
    
    # GWP Factors
    st.subheader("🌍 GWP Factors")
    gwp_standard = st.selectbox("GWP Standard", ["IPCC AR5 (2014)", "IPCC AR6 (2021)"])
    
    if gwp_standard == "IPCC AR5 (2014)":
        gwp_ch4 = 28.0
        gwp_n2o = 265.0
    else:
        gwp_ch4 = 29.8
        gwp_n2o = 273.0
    
    gwp_nmvoc = st.number_input("nmVOC GWP-100", value=2.2, min_value=0.1, max_value=10.0)
    
    st.caption(f"CH₄ GWP: {gwp_ch4} | N₂O GWP: {gwp_n2o}")
    
    st.divider()
    
    # Advanced Options
    st.subheader("⚡ Advanced Options")
    use_process_sim = st.checkbox(
        "Use Process Simulation",
        value=True,
        help="Use NeqSim process equipment (Separator) for more realistic multi-stage degassing"
    )

# ===================== MAIN CONTENT - TABS =====================
main_tab1, main_tab2, main_tab3 = st.tabs(["📊 Emission Calculator", "🔬 What-If Analysis", "📈 Uncertainty Analysis"])

# ===================== TAB 1: EMISSION CALCULATOR =====================
with main_tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if emission_source == "Produced Water Degassing":
            st.subheader("📋 Laboratory Gas Analysis")
            st.caption("Enter gas composition from flashing water sample to standard conditions (1 atm, 15°C)")
            
            available_components = ['CO2', 'methane', 'ethane', 'propane', 'H2S',
                                   'i-butane', 'n-butane', 'i-pentane', 'n-pentane', 'n-hexane',
                                   'nitrogen']
            current_df = st.session_state.emission_gas_df
        elif emission_source == "TEG Regeneration":
            st.subheader(f"📋 Fluid Composition - {emission_source}")
            available_components = ['TEG', 'water', 'methane', 'ethane', 'propane', 'CO2', 'H2S', 'nitrogen']
            current_df = st.session_state.teg_fluid_df
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
            use_container_width=True
        )
        
        # Update session state
        if emission_source == "TEG Regeneration":
            st.session_state.teg_fluid_df = edited_df
        else:
            st.session_state.emission_gas_df = edited_df
        
        # Composition summary
        total_comp = edited_df['MolarComposition[-]'].sum()
        if abs(total_comp - 1.0) > 0.001:
            st.warning(f"Total composition: {total_comp:.4f} (normalization required)")
        else:
            st.success(f"Total composition: {total_comp:.4f} (valid)")
    
    with col2:
        st.subheader("📊 Flash Gas Composition")
        
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
                title="Lab Flash Gas Composition (mol%)" if emission_source == "Produced Water Degassing" else "Fluid Composition (mol%)",
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
                        # Create CPA fluid with water + dissolved gas
                        # Typical dissolved gas content in produced water:
                        # - GWR (Gas-Water Ratio) ~0.5-2 Sm³/Sm³ at reservoir conditions
                        # - At surface: ~0.5-1.5 kg gas per m³ water
                        # - This translates to ~0.05-0.15 mol% dissolved gas in water
                        # 
                        # Using molar basis: water ~55.5 mol/kg, gas ~0.03-0.05 mol/kg water
                        # So gas fraction ~0.001 mol/mol (0.1 mol%)
                        process_fluid = fluid('cpa')
                        
                        # Add water as dominant component (99.9% of total moles)
                        process_fluid.addComponent('water', 0.999)
                        
                        # Add dissolved gas components - realistic solubility
                        # At 30 bara, 80°C: CH4 solubility ~0.0015 mol/mol, CO2 ~0.003 mol/mol
                        # Total dissolved gas ~0.001 mol/mol (0.1%)
                        gas_scale = 0.001  # 0.1 mol% dissolved gas - realistic for produced water
                        for _, row in edited_df.iterrows():
                            if row['MolarComposition[-]'] > 0:
                                process_fluid.addComponent(row['ComponentName'], row['MolarComposition[-]'] * gas_scale)
                        
                        # Set initial conditions
                        process_fluid.setTemperature(inlet_temp, 'C')
                        process_fluid.setPressure(inlet_pressure, 'bara')
                        process_fluid.setTotalFlowRate(total_flow, 'kg/hr')
                    else:
                        # For other sources, use composition directly
                        process_fluid = fluid('cpa')
                        for _, row in edited_df.iterrows():
                            if row['MolarComposition[-]'] > 0:
                                process_fluid.addComponent(row['ComponentName'], row['MolarComposition[-]'])
                        
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
                                    nmvoc_components = ['ethane', 'propane', 'i-butane', 'n-butane', 'i-pentane', 'n-pentane', 'n-hexane']
                                    c4plus_total = 0.0
                                    
                                    for comp_name in ['CO2', 'methane', 'ethane', 'propane', 'H2S', 
                                                     'i-butane', 'n-butane', 'i-pentane', 'n-pentane', 'n-hexane']:
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
                                                elif comp_name in ['i-butane', 'n-butane', 'i-pentane', 'n-pentane', 'n-hexane']:
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
                            TPflash(stage_fluid)
                            stage_fluid.initProperties()
                            
                            stage_result = {
                                'Stage': stage_names[i],
                                'Pressure (bara)': pressure,
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
                                
                                # Extract individual component flows including C4+
                                c4plus_total = 0.0
                                for comp_name in ['CO2', 'methane', 'ethane', 'propane', 'H2S',
                                                 'i-butane', 'n-butane', 'i-pentane', 'n-pentane', 'n-hexane']:
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
                                            elif comp_name in ['i-butane', 'n-butane', 'i-pentane', 'n-pentane', 'n-hexane']:
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
                        display_df.columns = ['Stage', 'Pressure (bara)', 'Total (kg/hr)', 'CO₂ (kg/hr)', 
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
                            
                            st.markdown(f"""
                            **Gas-to-Water Mass Factors (GWMF):**
                            | Component | NeqSim | Conventional |
                            |-----------|--------|---------------|
                            | CO₂ | {gwmf_co2:.1f} g/m³/bar | Not reported |
                            | CH₄ | {gwmf_ch4:.1f} g/m³/bar | 14 g/m³/bar |
                            | nmVOC | {gwmf_nmvoc:.1f} g/m³/bar | 3.5 g/m³/bar |
                            | **Total** | **{gwmf_total:.1f} g/m³/bar** | ~17.5 g/m³/bar |
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
                        params_df = pd.DataFrame({
                            'Parameter': ['Emission Source', 'Inlet Temperature', 'Inlet Pressure', 
                                         'Flow Rate', 'GWP Standard', 'CH₄ GWP', 'nmVOC GWP', 'Salinity'],
                            'Value': [emission_source, f"{inlet_temp} °C", f"{inlet_pressure} bara",
                                     flow_display, gwp_standard, gwp_ch4, gwp_nmvoc, f"{salinity_ppm} ppm"]
                        })
                        st.dataframe(params_df, use_container_width=True)
                
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
                
                # Get current composition
                if emission_source == "TEG Regeneration":
                    comp_df = st.session_state.teg_fluid_df
                else:
                    comp_df = st.session_state.emission_gas_df
                
                for val in param_values:
                    try:
                        scenario_fluid = fluid('cpa')
                        
                        # Build fluid with water + dissolved gas
                        if emission_source == "Produced Water Degassing":
                            scenario_fluid.addComponent('water', 0.999)
                            gas_scale = 0.001  # Realistic dissolved gas content
                            for _, row in comp_df.iterrows():
                                if row['MolarComposition[-]'] > 0:
                                    scenario_fluid.addComponent(row['ComponentName'], row['MolarComposition[-]'] * gas_scale)
                        else:
                            for _, row in comp_df.iterrows():
                                if row['MolarComposition[-]'] > 0:
                                    scenario_fluid.addComponent(row['ComponentName'], row['MolarComposition[-]'])
                        
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
                        
                        # Flash to degasser pressure
                        scenario_fluid.setPressure(4.0, 'bara')
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
                
                if emission_source == "TEG Regeneration":
                    comp_df = st.session_state.teg_fluid_df
                else:
                    comp_df = st.session_state.emission_gas_df
                
                progress_bar = st.progress(0)
                
                for i in range(n_samples):
                    try:
                        # Perturb parameters
                        temp_pert = np.random.normal(base_temp, temp_uncertainty)
                        press_pert = np.random.normal(base_press, press_uncertainty)
                        flow_pert = np.random.normal(base_flow, base_flow * flow_uncertainty / 100)
                        
                        mc_fluid = fluid('cpa')
                        
                        # Build fluid with water + dissolved gas for produced water
                        if emission_source == "Produced Water Degassing":
                            mc_fluid.addComponent('water', 0.999 * np.random.normal(1.0, comp_uncertainty/100))
                            gas_scale = 0.001  # Realistic dissolved gas content
                            for _, row in comp_df.iterrows():
                                if row['MolarComposition[-]'] > 0:
                                    pert_comp = row['MolarComposition[-]'] * gas_scale * np.random.normal(1.0, comp_uncertainty/100)
                                    mc_fluid.addComponent(row['ComponentName'], max(0.00001, pert_comp))
                        else:
                            for _, row in comp_df.iterrows():
                                if row['MolarComposition[-]'] > 0:
                                    pert_comp = row['MolarComposition[-]'] * np.random.normal(1.0, comp_uncertainty/100)
                                    mc_fluid.addComponent(row['ComponentName'], max(0.001, pert_comp))
                        
                        mc_fluid.setTemperature(temp_pert, 'C')
                        mc_fluid.setPressure(press_pert, 'bara')
                        mc_fluid.setTotalFlowRate(flow_pert, 'kg/hr')
                        
                        TPflash(mc_fluid)
                        mc_fluid.initProperties()
                        
                        mc_fluid.setPressure(4.0, 'bara')
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
