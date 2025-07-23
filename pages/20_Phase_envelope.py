import streamlit as st
import pandas as pd
import neqsim
from neqsim.thermo import fluid_df, phaseenvelope, TPflash, dataFrame
from neqsim import jneqsim
import matplotlib.pyplot as plt
from fluids import detailedHC_data

st.title('Phase Envelope')

"""
NeqSim uses the UMR-PRU-EoS model for calculations of the phase envelope. 
The UMR-PRU-EoS is a predictive equation of state that combines the PR EoS 
with an original UNIFAC-type model for the excess Gibbs energy (GE), 
through the universal mixing rules (UMR). The model is called UMR-PRU 
(Universal Mixing Rule Peng Robinson UNIFAC) and it is an accurate model 
for calculation of cricondenbar and hydrocarbon dew points.
"""

st.text("Set fluid composition:")

# -----------------------------------------------------------------------------
# 1. Manage session state and reading fluid data
# -----------------------------------------------------------------------------
hidecomponents = st.checkbox('Show active components')

# Only try to filter if "st.edited_df" has already been defined somewhere
# The code below assumes st.edited_df is populated after st.data_editor
# so we wrap it in a check to avoid errors on first load:
if hidecomponents and 'edited_df' in st.session_state:
    st.edited_df['MolarComposition[-]'] = st.edited_df['MolarComposition[-]']
    st.session_state.activefluid_df = st.edited_df[
        st.edited_df['MolarComposition[-]'] > 0
    ]

# Check for uploaded file in sidebar
if 'uploaded_file' in st.session_state and not hidecomponents:
    try:
        st.session_state.activefluid_df = pd.read_csv(st.session_state.uploaded_file)
        numeric_columns = ['MolarComposition[-]', 'MolarMass[kg/mol]', 'RelativeDensity[-]']
        st.session_state.activefluid_df[numeric_columns] = (
            st.session_state.activefluid_df[numeric_columns].astype(float)
        )
    except:
        st.session_state.activefluid_df = pd.DataFrame(detailedHC_data)

# If we don't yet have a fluid DataFrame or it doesn't match the name, reset
if 'activefluid_df' not in st.session_state or st.session_state.get('activefluid_name') != 'detailedHC_data':
    st.session_state.activefluid_name = 'detailedHC_data'
    st.session_state.activefluid_df = pd.DataFrame(detailedHC_data)

# -----------------------------------------------------------------------------
# 2. Let user edit the fluid data
# -----------------------------------------------------------------------------
st.edited_df = st.data_editor(
    st.session_state.activefluid_df,
    column_config={
        "ComponentName": "Component Name",
        "MolarComposition[-]": st.column_config.NumberColumn(
            "Molar Composition [-]",
            min_value=0.0,
            max_value=1e6,
            format="%f"
        ),
        "MolarMass[kg/mol]": st.column_config.NumberColumn(
            "Molar Mass [kg/mol]",
            min_value=0,
            max_value=10000,
            format="%f kg/mol"
        ),
        "RelativeDensity[-]": st.column_config.NumberColumn(
            "Density [gr/cm3]",
            min_value=1e-10,
            max_value=10.0,
            format="%f gr/cm3"
        ),
    },
    num_rows='dynamic'
)

isplusfluid = st.checkbox('Plus Fluid')
usePR = st.checkbox('Peng Robinson EoS', help='use standard Peng Robinson EoS')

st.text("Fluid composition will be normalized before simulation")
st.divider()

# -----------------------------------------------------------------------------
# 3. When user clicks "Run," do the phase envelope + final TP flash
# -----------------------------------------------------------------------------
if st.button('Run'):
    if st.edited_df['MolarComposition[-]'].sum() > 0:
        # Select the model name
        modelname = "UMR-PRU-EoS"
        if usePR:
            modelname = "PrEos"
        
        # Create the fluid in Java with chosen model
        jneqsim.util.database.NeqSimDataBase.setCreateTemporaryTables(True)
        neqsim_fluid = fluid_df(
            st.edited_df,
            lastIsPlusFraction=isplusfluid,
            add_all_components=False
        ).setModel(modelname)
        jneqsim.util.database.NeqSimDataBase.setCreateTemporaryTables(False)
        
        st.success('Successfully created fluid')
        st.subheader("Results:")
        
        # Calculate phase envelope
        thermoOps = jneqsim.thermodynamicoperations.ThermodynamicOperations(neqsim_fluid)
        thermoOps.calcPTphaseEnvelope2()
        
        # Get dew/bubble point data
        dewts = [x - 273.15 for x in list(thermoOps.getOperation().get("dewT"))]
        dewps = list(thermoOps.getOperation().get("dewP"))
        bubts = [x - 273.15 for x in list(thermoOps.getOperation().get("bubT"))]
        bubps = list(thermoOps.getOperation().get("bubP"))
        
        # Plot the PT envelope
        fig, ax = plt.subplots()
        plt.plot(dewts, dewps, label="dew point")
        plt.plot(bubts, bubps, label="bubble point")
        plt.title('PT envelope')
        plt.xlabel('Temperature [°C]')
        plt.ylabel('Pressure [bara]')
        plt.legend()
        plt.grid(True)
        st.pyplot(fig)
        st.divider()
        
        # Display cricondenbar and cricondentherm
        cricobar = thermoOps.getOperation().get("cricondenbar")
        cricotherm = thermoOps.getOperation().get("cricondentherm")
        criticalT = thermoOps.getOperation().get("criticalPoint1")[0]
        criticalP = thermoOps.getOperation().get("criticalPoint1")[1] 
        mixingrulename = neqsim_fluid.getMixingRuleName
        modelname = neqsim_fluid.getModelName
        
        st.write('critical point ', 
                 round(criticalP, 2), ' bara, ', 
                 round(criticalT - 273.15, 2), ' °C')
        st.write('cricondentherm ', 
                 round(cricotherm[1], 2), ' bara, ', 
                 round(cricotherm[0] - 273.15, 2), ' °C')
        st.write('cricondenbar ', 
                 round(cricobar[1], 2), ' bara, ', 
                 round(cricobar[0] - 273.15, 2), ' °C')
        st.write('Using ', 
                 modelname, ' with ', 
                 mixingrulename)        
        
        
        # Show dew/bubble data points
        dewdatapoints = pd.DataFrame(
            {
                'dew temperatures [°C]': dewts,
                'dew pressures [bara]': dewps,
            }
        )
        bubdatapoints = pd.DataFrame(
            {
                'bub temperatures [°C]': bubts,
                'bub pressures [bara]': bubps,
            }
        )
        
        st.divider()
        st.write('dew points')
        st.data_editor(dewdatapoints)
        st.write('bubble points')
        st.data_editor(bubdatapoints)
        
        # ---------------------------------------------------------------------
        # 4. Final TP flash at ~1 atm (1.01325 bara) and 15 °C
        # ---------------------------------------------------------------------
        st.divider()
        st.subheader("Standard condition TP flash (1 atm and 15 °C)")
        pressure = 1.01325  # 1 atm in bara
        temp = 15.0         # °C
        neqsim_fluid.setPressure(pressure, 'bara')
        neqsim_fluid.setTemperature(temp, 'C')
        TPflash(neqsim_fluid)
        
        # Retrieve a DataFrame of results
        flash_results_df = dataFrame(neqsim_fluid)
        st.data_editor(flash_results_df, key="final_tpflash_result")
        
    else:
        st.error('The sum of Molar Composition must be greater than 0. Please adjust your inputs.')

# -----------------------------------------------------------------------------
# 5. Sidebar: file uploader
# -----------------------------------------------------------------------------
st.sidebar.file_uploader(
    "Import Fluid",
    key='uploaded_file',
    help='Fluids can be saved by hovering over the fluid window and clicking the "Download as CSV" button in the upper-right corner.'
)
