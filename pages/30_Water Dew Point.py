import streamlit as st
import pandas as pd
import neqsim
from neqsim.thermo.thermoTools import fluidcreator, fluid_df, TPflash, dataFrame, hydt, waterdewt
from fluids import default_fluid
import plotly.graph_objects as go
from theme import apply_theme

st.set_page_config(page_title="Water Dew Point", page_icon='images/neqsimlogocircleflat.png')
apply_theme()

st.title('Water Dew Point')
st.divider()

with st.expander("📋 Set Fluid Composition", expanded=True):
    # Reset button to restore default composition
    if st.button('Reset to Default Composition'):
        st.session_state.waterdew_fluid_df = pd.DataFrame(default_fluid)
        st.rerun()

    hidecomponents = st.checkbox('Show active components')
    if hidecomponents and 'waterdew_edited_df' in st.session_state:
        st.session_state.waterdew_fluid_df = st.session_state.waterdew_edited_df[
            st.session_state.waterdew_edited_df['MolarComposition[-]'] > 0
        ]
       
    if 'waterdew_uploaded_file' in st.session_state and st.session_state.waterdew_uploaded_file is not None and not hidecomponents:
        try:
            st.session_state.waterdew_fluid_df = pd.read_csv(st.session_state.waterdew_uploaded_file)
            numeric_columns = ['MolarComposition[-]', 'MolarMass[kg/mol]', 'RelativeDensity[-]']
            st.session_state.waterdew_fluid_df[numeric_columns] = st.session_state.waterdew_fluid_df[numeric_columns].astype(float)
        except Exception as e:
            st.warning(f'Could not load file: {e}')
            st.session_state.waterdew_fluid_df = pd.DataFrame(default_fluid)

    if 'waterdew_fluid_df' not in st.session_state:
        st.session_state.waterdew_fluid_df = pd.DataFrame(default_fluid)

    if 'waterdew_tp_data' not in st.session_state:
        st.session_state['waterdew_tp_data'] = pd.DataFrame({
            'Pressure (bara)': [50.0, 100.0, 150.0, 200.0],   # Default example pressure
            'Temperature (C)': [None, None, None, None]  # Default temperature
        })

    st.edited_df = st.data_editor(
        st.session_state.waterdew_fluid_df,
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
    st.session_state.waterdew_edited_df = st.edited_df

    isplusfluid = st.checkbox('Plus Fluid')

    st.caption("💡 Fluid composition will be normalized before simulation")
with st.expander("📊 Input Pressures", expanded=True):
    st.edited_dfTP = st.data_editor(
        st.session_state.waterdew_tp_data['Pressure (bara)'].dropna().reset_index(drop=True),
        num_rows='dynamic',  # Allows dynamic number of rows
        column_config={
            'Pressure (bara)': st.column_config.NumberColumn(
                label="Pressure (bara)",
                min_value=1e-10,      # Minimum pressure (exclude zero)
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
        pressure_values = st.edited_dfTP.dropna()
        if (pressure_values <= 0).any():
            st.error('Pressure must be greater than 0 bara. Please update the pressure inputs before running calculations.')
        else:
            with st.spinner('Calculating water dew point...'):
                try:
                    neqsim_fluid = fluid_df(st.edited_df, lastIsPlusFraction=False, add_all_components=False).autoSelectModel()
                    results_list = []
                    results_list2 = []
                    pres_list = []
                    fluid_results_list = []
                    for pressure in pressure_values:
                        pres_list.append(pressure)
                        neqsim_fluid.setPressure(pressure, 'bara')
                        results_list.append(hydt(neqsim_fluid)-273.15)
                        results_list2.append(waterdewt(neqsim_fluid)-273.15)
                        fluid_results_list.append(dataFrame(neqsim_fluid))
                    st.session_state['waterdew_tp_data'] = pd.DataFrame({
                        'Pressure (bara)': pres_list,   # Default example pressure
                        'Hydrate Temperature (C)': results_list,  # Default temperature
                        'Aqueous Temperature (C)': results_list2  # Default temperature
                    })
                    st.session_state['waterdew_tp_data'] = st.session_state['waterdew_tp_data'].sort_values('Pressure (bara)')
                    st.success('Hydrate calculation finished successfully!')
                    combined_results = pd.concat(fluid_results_list, ignore_index=True)

                    if st.session_state.get('refresh', True):
                        st.edited_dfTP2 = st.data_editor(
                            st.session_state.waterdew_tp_data.dropna().reset_index(drop=True),
                            num_rows='dynamic',  # Allows dynamic number of rows
                            column_config={
                                'Pressure (bara)': st.column_config.NumberColumn(
                                    label="Pressure (bara)",
                                    min_value=1e-10,      # Minimum pressure (exclude zero)
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
                    
                    # Hydrate temperature line
                    fig.add_trace(go.Scatter(
                        x=st.session_state['waterdew_tp_data']['Hydrate Temperature (C)'],
                        y=st.session_state['waterdew_tp_data']['Pressure (bara)'],
                        mode='lines+markers',
                        name='Hydrate Temperature',
                        line=dict(color='#2E86AB', width=2),
                        marker=dict(size=8, symbol='circle'),
                        hovertemplate='Hydrate<br>T: %{x:.1f} °C<br>P: %{y:.1f} bara<extra></extra>'
                    ))
                    
                    # Aqueous dew point line
                    fig.add_trace(go.Scatter(
                        x=st.session_state['waterdew_tp_data']['Aqueous Temperature (C)'],
                        y=st.session_state['waterdew_tp_data']['Pressure (bara)'],
                        mode='lines+markers',
                        name='Aqueous Dew Point',
                        line=dict(color='#E63946', width=2, dash='dash'),
                        marker=dict(size=8, symbol='x'),
                        hovertemplate='Aqueous Dew Point<br>T: %{x:.1f} °C<br>P: %{y:.1f} bara<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title=dict(text='Dew Point Lines', font=dict(size=20)),
                        xaxis_title='Temperature (°C)',
                        yaxis_title='Pressure (bara)',
                        hovermode='closest',
                        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01),
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
                        input = "What scientific experimental water dew point equilibrium data are available for mixtures of " + result_string
                        openapitext = st.make_request(input)
                        st.write(openapitext)
                    except Exception:
                        pass  # AI features optional
                    st.session_state['rerender'] = not st.session_state.get('rerender', False)
                except Exception as e:
                    st.error(f'Calculation failed: {str(e)}')
    else:
        st.error('Water Molar Composition must be greater than 0. Please adjust your inputs.')

st.sidebar.file_uploader("Import Fluid", key='waterdew_uploaded_file', help='Fluids can be saved by hovering over the fluid window and clicking the "Download as CSV" button in the upper-right corner.')