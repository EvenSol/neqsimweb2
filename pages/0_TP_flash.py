import streamlit as st
import pandas as pd
import time
import neqsim
from neqsim.thermo.thermoTools import fluidcreator, fluid_df, TPflash, dataFrame
from fluids import default_fluid
from theme import apply_theme

st.set_page_config(page_title="TP Flash", page_icon='images/neqsimlogocircleflat.png')
apply_theme()

st.title('TP flash')
"""
The NeqSim flash model will select the best thermodynamic model based on the fluid composition. For fluids containing polar components it will use the CPA-EoS.
For non-polar fluids it will use the SRK/PR-EoS. The flash will calculate the phase equilibrium for given composition at the specified temperatures and pressures.

You can select components from a predifined component list. Alterative component names ([see available components](https://github.com/equinor/neqsim/blob/master/src/main/resources/data/COMP.csv)) can be used by manually editing the table.
"""
st.divider()

with st.expander("📋 Set Fluid Composition", expanded=True):
    # Reset button to restore default composition
    if st.button('Reset to Default Composition'):
        st.session_state.tpflash_fluid_df = pd.DataFrame(default_fluid)
        st.rerun()

    hidecomponents = st.checkbox('Show active components')
    if hidecomponents and 'tpflash_edited_df' in st.session_state:
        st.session_state.tpflash_fluid_df = st.session_state.tpflash_edited_df[
            st.session_state.tpflash_edited_df['MolarComposition[-]'] > 0
        ]

    if 'tpflash_uploaded_file' in st.session_state and st.session_state.tpflash_uploaded_file is not None and not hidecomponents:
        try:
            st.session_state.tpflash_fluid_df = pd.read_csv(st.session_state.tpflash_uploaded_file)
            numeric_columns = ['MolarComposition[-]', 'MolarMass[kg/mol]', 'RelativeDensity[-]']
            st.session_state.tpflash_fluid_df[numeric_columns] = st.session_state.tpflash_fluid_df[numeric_columns].astype(float)
        except Exception as e:
            st.warning(f'Could not load file: {e}')
            st.session_state.tpflash_fluid_df = pd.DataFrame(default_fluid)

    if 'tpflash_fluid_df' not in st.session_state:
        st.session_state.tpflash_fluid_df = pd.DataFrame(default_fluid)

    if 'tpflash_tp_data' not in st.session_state:
        st.session_state['tpflash_tp_data'] = pd.DataFrame({
            'Temperature (C)': [20.0, 25.0],  # Default example temperature
            'Pressure (bara)': [1.0, 10.0]  # Default example pressure
        })

    st.edited_df = st.data_editor(
        st.session_state.tpflash_fluid_df,
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

    # Store edited df for later use
    st.session_state.tpflash_edited_df = st.edited_df

    isplusfluid = st.checkbox('Plus Fluid')
    
    st.caption("💡 Fluid composition will be normalized before simulation")
with st.expander("🌡️ Input Pressures and Temperatures", expanded=True):
    st.edited_dfTP = st.data_editor(
        st.session_state.tpflash_tp_data.dropna().reset_index(drop=True),
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
        if st.session_state.tpflash_tp_data.empty:
            st.error('No data to perform calculations. Please input temperature and pressure values.')
        else:
            with st.spinner('Running flash calculations...'):
                try:
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
                        results_list.append(dataFrame(neqsim_fluid))
                    
                    st.success('Flash calculations finished successfully!')
                    st.subheader("Results:")
                    # Combine all results into a single dataframe
                    combined_results = pd.concat(results_list, ignore_index=True)
                    
                    # Display the combined results
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
                    except Exception:
                        pass  # AI features optional
                except Exception as e:
                    st.error(f'Calculation failed: {str(e)}')
    else:
        st.error('The sum of Molar Composition must be greater than 0. Please adjust your inputs.')
        
st.sidebar.file_uploader("Import Fluid", key='tpflash_uploaded_file', help='Fluids can be saved by hovering over the fluid window and clicking the "Download as CSV" button in the upper-right corner.')
