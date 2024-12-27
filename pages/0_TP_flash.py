import streamlit as st
import pandas as pd
import time
import neqsim
from neqsim.thermo.thermoTools import fluidcreator, fluid_df, TPflash, dataFrame
from fluids import default_fluid

st.title('TP flash')
"""
NeqSim TP flash will select the best thermodynamic model based on the fluid composition. For fluids containing polar components it will use the CPA-EoS.
"""
st.divider()
st.text("Set fluid composition:")

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
        st.session_state.activefluid_df = pd.DataFrame(default_fluid)

if 'activefluid_df' not in st.session_state or st.session_state.get('activefluid_name') != 'default_fluid':
    st.session_state.activefluid_df = pd.DataFrame(default_fluid)
    st.session_state.activefluid_name = 'default_fluid'

if 'tp_flash_data' not in st.session_state:
    st.session_state['tp_flash_data'] = pd.DataFrame({
        'Temperature (C)': [20.0, 25.0],  # Default example temperature
        'Pressure (bara)': [1.0, 10.0]  # Default example pressure
    })

st.edited_df = st.data_editor(
    st.session_state.activefluid_df,
    column_config={
        "ComponentName": "Component Name",
        "MolarComposition[-]": st.column_config.NumberColumn("Molar Composition [-]", min_value=0, max_value=10000, format="%f"),
        "MolarMass[kg/mol]": st.column_config.NumberColumn(
            "Molar Mass [kg/mol]", min_value=0, max_value=10000, format="%f kg/mol"
        ),
        "RelativeDensity[-]": st.column_config.NumberColumn(
            "Density [gr/cm3]", min_value=1e-10, max_value=10.0, format="%f gr/cm3"
        ),
    },
num_rows='dynamic')
isplusfluid = st.checkbox('Plus Fluid')

st.text("Fluid composition will be normalized before simulation")
st.divider()
# Use st.data_editor for inputting temperature and pressure
st.text("Input Pressures and Temperatures")
st.edited_dfTP = st.data_editor(
    st.session_state.tp_flash_data.dropna().reset_index(drop=True),
    num_rows='dynamic',  # Allows dynamic number of rows
    column_config={
        'Temperature (C)': st.column_config.NumberColumn(
            label="Temperature (C)",
            min_value=-273.15,  # Minimum temperature in Celsius
            max_value=1000,     # Maximum temperature in Celsius
            format='%f',        # Decimal format
            help='Enter the temperature in degrees Celsius.'  # Help text for guidance
        ),
        'Pressure (bara)': st.column_config.NumberColumn(
            label="Pressure (bara)",
            min_value=0.0,      # Minimum pressure
            max_value=1000,     # Maximum pressure
            format='%f',        # Decimal format
            help='Enter the pressure in bar absolute.'  # Help text for guidance
        ),
    }
)

if st.button('Run TP Flash Calculations'):
    if st.edited_df['MolarComposition[-]'].sum() > 0:
        # Check if the dataframe is empty
        if st.session_state.tp_flash_data.empty:
            st.error('No data to perform calculations. Please input temperature and pressure values.')
        else:
            # Initialize a list to store results
            results_list = []
            neqsim_fluid = fluid_df(st.edited_df, lastIsPlusFraction=isplusfluid, add_all_components=False).autoSelectModel()
            
            # Iterate over each row and perform calculations
            for idx, row in st.edited_dfTP.dropna().iterrows():
                temp = row['Temperature (C)']
                pressure = row['Pressure (bara)']
                neqsim_fluid.setPressure(pressure, 'bara')
                neqsim_fluid.setTemperature(temp, 'C')
                TPflash(neqsim_fluid)
                #results_df = st.data_editor(dataFrame(neqsim_fluid))
                results_list.append(dataFrame(neqsim_fluid))
            
            st.success('Flash calculations finished successfully!')
            st.subheader("Results:")
            # Combine all results into a single dataframe
            combined_results = pd.concat(results_list, ignore_index=True)
            
            # Display the combined results
            #st.subheader('Combined TP Flash Results')
            #st.dataframe(combined_results)
            results_df = st.data_editor(combined_results)
            st.divider()
            list1 = neqsim_fluid.getComponentNames()
            l1 = list(list1)
            string_list = [str(element) for element in l1]
            delimiter = ", "
            result_string = delimiter.join(string_list)
            try:
                input = "What scientific experimental equilibrium data are available for mixtures of " + result_string + " at temperature around " + str(temp) + " Celsius and pressure around " + str(pressure) + " bar."
                openapitext = st.make_request(input)
                st.write(openapitext)
            except:
                st.write('OpenAI key needed for data analysis')
    else:
        st.error('The sum of Molar Composition must be greater than 0. Please adjust your inputs.')
        
st.sidebar.file_uploader("Import Fluid", key='uploaded_file', help='Fluids can be saved by hovering over the fluid window and clicking the "Download as CSV" button in the upper-right corner.')