import streamlit as st
import pandas as pd
import neqsim
import time
from neqsim.thermo.thermoTools import fluidcreator, fluid_df, hydt, dataFrame
from fluids import default_fluid
import plotly.graph_objects as go

st.set_page_config(page_title="Gas Hydrate", page_icon='images/neqsimlogocircleflat.png')

st.title('Gas Hydrate Calculation')
"""
Gas hydrate calculations are done using the CPA-EoS combined with a model for the solid hydrate phase.
"""
st.divider()
st.text("Set fluid composition:")

# Reset button to restore default composition
if st.button('Reset to Default Composition'):
    st.session_state.hydrate_fluid_df = pd.DataFrame(default_fluid)
    st.rerun()

hidecomponents = st.checkbox('Show active components')
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
        st.session_state.hydrate_fluid_df = pd.DataFrame(default_fluid)

if 'hydrate_fluid_df' not in st.session_state:
    st.session_state.hydrate_fluid_df = pd.DataFrame(default_fluid)

if 'hydrate_tp_data' not in st.session_state:
    st.session_state['hydrate_tp_data'] = pd.DataFrame({
        'Pressure (bara)': [50.0, 100.0, 150.0, 200.0],   # Default example pressure
        'Temperature (C)': [None, None, None, None]  # Default temperature
    })

st.edited_df = st.data_editor(
    st.session_state.hydrate_fluid_df,
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

# Store edited df for later use
st.session_state.hydrate_edited_df = st.edited_df

isplusfluid = st.checkbox('Plus Fluid')

st.text("Fluid composition will be normalized before simulation")

st.divider()

st.text("Input Pressures and Temperatures")

st.edited_dfTP = st.data_editor(
    st.session_state.hydrate_tp_data['Pressure (bara)'].dropna().reset_index(drop=True),
    num_rows='dynamic',  # Allows dynamic number of rows
    column_config={
        'Pressure (bara)': st.column_config.NumberColumn(
            label="Pressure (bara)",
            min_value=0.0,      # Minimum pressure
            max_value=1000,     # Maximum pressure
            format='%f',        # Decimal format
            help='Enter the pressure in bar absolute.'  # Help text for guidance
        )
    }
)

if st.button('Run'):
    # Check if water's MolarComposition[-] is greater than 0
    water_row = st.edited_df[st.edited_df['ComponentName'] == 'water']  # Adjust 'ComponentName' and 'water' as necessary
    if not water_row.empty and water_row['MolarComposition[-]'].iloc[0] > 0:
        with st.spinner('Calculating hydrate equilibrium...'):
            try:
                neqsim_fluid = fluid_df(st.edited_df, lastIsPlusFraction=False, add_all_components=False).autoSelectModel()
                results_list = []
                pres_list = []
                fluid_results_list = []
                for pres in st.edited_dfTP.dropna():
                    pressure = pres
                    pres_list.append(pressure)
                    neqsim_fluid.setPressure(pressure, 'bara')
                    results_list.append(hydt(neqsim_fluid)-273.15)
                    fluid_results_list.append(dataFrame(neqsim_fluid))
                st.session_state['hydrate_tp_data'] = pd.DataFrame({
                    'Pressure (bara)': pres_list,   # Default example pressure
                    'Temperature (C)': results_list  # Default temperature
                })
                st.session_state['hydrate_tp_data'] = st.session_state['hydrate_tp_data'].sort_values('Pressure (bara)')
                st.success('Hydrate calculation finished successfully!')
                combined_results = pd.concat(fluid_results_list, ignore_index=True)
           
                if st.session_state.get('refresh', True):
                    st.edited_dfTP2 = st.data_editor(
                        st.session_state.hydrate_tp_data.reset_index(drop=True),
                        num_rows='dynamic',  # Allows dynamic number of rows
                        column_config={
                            'Pressure (bara)': st.column_config.NumberColumn(
                                label="Pressure (bara)",
                                min_value=0.0,      # Minimum pressure
                                max_value=1000,     # Maximum pressure
                                format='%f',        # Decimal format
                                help='Enter the pressure in bar absolute.'  # Help text for guidance
                            ),
                            'Temperature (C)': st.column_config.NumberColumn(
                                label="Temperature (C)",
                                min_value=-273.15,  # Minimum temperature in Celsius
                                max_value=1000,     # Maximum temperature in Celsius
                                format='%f',        # Decimal format
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
                
                st.plotly_chart(fig, width='stretch')
                st.divider()
                results_df = st.data_editor(combined_results)
                st.divider()
                list1 = neqsim_fluid.getComponentNames()
                l1 = list(list1)
                string_list = [str(element) for element in l1]
                delimiter = ", "
                result_string = delimiter.join(string_list)
                try:
                    input = "What scientific experimental hydrate equilibrium data are available for mixtures of " + result_string
                    openapitext = st.make_request(input)
                    st.write(openapitext)
                except Exception:
                    st.info('💡 Enter OpenAI API key in the sidebar for AI-powered data analysis')
                st.session_state['rerender'] = not st.session_state.get('rerender', False)
            except Exception as e:
                st.error(f'Calculation failed: {str(e)}')
    else:
        st.error('Water Molar Composition must be greater than 0. Please adjust your inputs.')

st.sidebar.file_uploader("Import Fluid", key='hydrate_uploaded_file', help='Fluids can be saved by hovering over the fluid window and clicking the "Download as CSV" button in the upper-right corner.')
