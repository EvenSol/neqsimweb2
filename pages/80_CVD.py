import streamlit as st
import pandas as pd
from neqsim.thermo.thermoTools import fluid_df
from neqsim.thermo import CVD
from fluids import detailedHC_data

st.title('Constant Volume Depletion (CVD)')
"""
Perform a constant volume depletion simulation for a reservoir fluid. The NeqSim
library will automatically select an appropriate thermodynamic model based on the
input composition.
"""

st.divider()
st.text("Set reservoir fluid composition:")

hidecomponents = st.checkbox('Show active components')
if hidecomponents:
    st.edited_df['MolarComposition[-]'] = st.edited_df['MolarComposition[-]']
    st.session_state.activefluid_df = st.edited_df[st.edited_df['MolarComposition[-]'] > 0]

if 'uploaded_file' in st.session_state and hidecomponents == False:
    try:
        st.session_state.activefluid_df = pd.read_csv(st.session_state.uploaded_file)
        numeric_columns = ['MolarComposition[-]']
        st.session_state.activefluid_df[numeric_columns] = st.session_state.activefluid_df[numeric_columns].astype(float)
    except:
        st.session_state.activefluid_df = pd.DataFrame(detailedHC_data)

if 'activefluid_df' not in st.session_state or st.session_state.get('activefluid_name') != 'detailedHC_data':
    st.session_state.activefluid_df = pd.DataFrame(detailedHC_data)
    st.session_state.activefluid_name = 'detailedHC_data'

st.edited_df = st.data_editor(
    st.session_state.activefluid_df,
    column_config={
        "ComponentName": "Component Name",
        "MolarComposition[-]": st.column_config.NumberColumn("Molar Composition [-]", min_value=0, max_value=10000, format="%f"),
        "MolarMass[kg/mol]": st.column_config.NumberColumn("Molar Mass [kg/mol]", min_value=0, max_value=10000, format="%f kg/mol"),
        "RelativeDensity[-]": st.column_config.NumberColumn("Density [gr/cm3]", min_value=1e-10, max_value=10.0, format="%f gr/cm3"),
    },
    num_rows='dynamic'
)

isplusfluid = st.checkbox('Plus Fluid')

st.text("Fluid composition will be normalized before simulation")
st.divider()

if 'cvd_pressure_data' not in st.session_state:
    st.session_state['cvd_pressure_data'] = pd.DataFrame({'Pressure (bara)': [400.0, 350.0, 300.0]})

st.text("Input Pressures (bara)")
st.edited_pressure = st.data_editor(
    st.session_state.cvd_pressure_data,
    num_rows='dynamic',
    column_config={
        'Pressure (bara)': st.column_config.NumberColumn(
            label="Pressure (bara)",
            min_value=0.0,
            max_value=1000.0,
            format='%f',
            help='Enter pressure in bar absolute.'
        )
    }
)

temperature = st.number_input('Temperature (C)', value=50.0)

if st.button('Run CVD Simulation'):
    if st.edited_df['MolarComposition[-]'].sum() > 0:
        pressures = st.edited_pressure['Pressure (bara)'].dropna().tolist()
        neqsim_fluid = fluid_df(st.edited_df, lastIsPlusFraction=isplusfluid, add_all_components=False).autoSelectModel()
        rel_vol = []
        liq_rel_vol = []
        zgas = []
        zmix = []
        cum_depl = []
        CVD(neqsim_fluid, pressures, temperature, rel_vol, liq_rel_vol, zgas, zmix, cum_depl)
        results = pd.DataFrame({
            'Pressure (bara)': pressures,
            'Relative Volume [-]': rel_vol,
            'Liquid Relative Volume [-]': liq_rel_vol,
            'Zgas [-]': zgas,
            'Zmix [-]': zmix,
            'Cum Mole% Depleted': cum_depl,
        })
        st.subheader("CVD Results")
        st.data_editor(results)
    else:
        st.error('The sum of Molar Composition must be greater than 0. Please adjust your inputs.')

st.sidebar.file_uploader("Import Fluid", key='uploaded_file', help='Fluids can be saved by hovering over the fluid window and clicking the "Download as CSV" button in the upper-right corner.')
