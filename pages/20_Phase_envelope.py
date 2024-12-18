import streamlit as st
import pandas as pd
import neqsim
from neqsim.thermo import fluid_df, phaseenvelope, TPflash, dataFrame
from neqsim import jneqsim
import matplotlib.pyplot as plt
from fluids import detailedHC_data

st.title('Phase Envelope')

"""
NeqSim uses the UMR-PRU-EoS model for calculations of the phase envelope. The UMR-PRU-EoS is a predictive equation of state that combines the PR EoS with an original UNIFAC-type model for the excess Gibbs energy (GE), through the universal mixing rules (UMR). The model is called UMR-PRU (Universal Mixing Rule Peng Robinson UNIFAC) and it is an accurate model for calculation of cricondenbar and hydrocarbon dew points.
"""

st.text("Set fluid composition:")
# Sample data for the DataFrame

hidecomponents = st.checkbox('Show active components')
if hidecomponents:
   st.edited_df['MolarComposition[-]'] = st.edited_df['MolarComposition[-]']
   st.session_state.activefluid_df = st.edited_df[st.edited_df['MolarComposition[-]'] > 0]
   
if 'uploaded_file' in st.session_state and hidecomponents == False:
    try:
        st.session_state.activefluid_df = pd.read_csv(st.session_state.uploaded_file)
        numeric_columns = ['MolarComposition[-]', 'MolarMass[kg/mol]', 'RelativeDensity[-]']
        st.session_state.activefluid_df[numeric_columns] = st.session_state.activefluid_df[numeric_columns].astype(float)
    except:
        st.session_state.activefluid_df = pd.DataFrame(detailedHC_data)

if 'activefluid_df' not in st.session_state or st.session_state.get('activefluid_name') != 'detailedHC_data':
   st.session_state.activefluid_name = 'detailedHC_data'
   st.session_state.activefluid_df = pd.DataFrame(detailedHC_data)

st.edited_df = st.data_editor(
    st.session_state.activefluid_df,
    column_config={
        "ComponentName": "Component Name",
        "MolarComposition[-]": st.column_config.NumberColumn(
        ),
        "MolarMass[kg/mol]": st.column_config.NumberColumn(
            "Molar Mass [kg/mol]", min_value=0, max_value=10000, format="%f kg/mol"
        ),
        "RelativeDensity[-]": st.column_config.NumberColumn(
            "Density [gr/cm3]", min_value=1e-10, max_value=10.0, format="%f gr/cm3"
        ),
    },
num_rows='dynamic')
isplusfluid = st.checkbox('Plus Fluid')

usePR = st.checkbox('Peng Robinson EoS', help='use standard Peng Robinson EoS')

st.text("Fluid composition will be normalized before simulation")
st.divider()

if st.button('Run'):
    if st.edited_df['MolarComposition[-]'].sum() > 0:
        modelname = "UMR-PRU-EoS"
        if(usePR):
           modelname = "PrEos"
        neqsim_fluid = fluid_df(st.edited_df, lastIsPlusFraction=isplusfluid, add_all_components=False).setModel(modelname)
        st.success('Successfully created fluid')
        st.subheader("Results:")
        thermoOps = jneqsim.thermodynamicoperations.ThermodynamicOperations(neqsim_fluid)
        thermoOps.calcPTphaseEnvelope2()
        fig, ax = plt.subplots()
        dewts = [x-273.15 for x in list(thermoOps.getOperation().get("dewT"))]
        dewps = list(thermoOps.getOperation().get("dewP"))
        bubts = [x-273.15 for x in list(thermoOps.getOperation().get("bubT"))]
        bubps = list(thermoOps.getOperation().get("bubP"))
        plt.plot(dewts,dewps, label="dew point")
        plt.plot(bubts, bubps, label="bubble point")
        plt.title('PT envelope')
        plt.xlabel('Temperature [C]')
        plt.ylabel('Pressure [bara]')
        plt.legend()
        plt.grid(True)
        st.pyplot(fig)
        st.divider()
        cricobar = thermoOps.getOperation().get("cricondenbar")
        cricotherm = thermoOps.getOperation().get("cricondentherm")
        st.write('cricondentherm ', round(cricotherm[1],2), ' bara, ',  round(cricotherm[0]-273.15,2), ' C')
        st.write('cricondenbar ', round(cricobar[1],2), ' bara, ', round(cricobar[0]-273.15,2), ' C')
        dewdatapoints = pd.DataFrame(
        {'dew temperatures [C]': dewts,
         'dew pressures [bara]':dewps,
        }
        )
        bubdatapoints = pd.DataFrame(
        {'bub temperatures [C]': bubts,
         'bub pressures [bara]':bubps,
        }
        )
        st.divider()
        st.write('dew points')
        st.data_editor(dewdatapoints)
        st.write('bubble points')
        st.data_editor(bubdatapoints)
    else:
        st.error('The sum of Molar Composition must be greater than 0. Please adjust your inputs.')

st.sidebar.file_uploader("Import Fluid", key='uploaded_file')