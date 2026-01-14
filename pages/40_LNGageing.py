import streamlit as st
from neqsim.thermo import fluid
from neqsim import jneqsim
import pandas as pd
from fluids import lng_fluid
from neqsim.thermo.thermoTools import fluid_df
from io import BytesIO
from theme import apply_theme

st.set_page_config(page_title="LNG Ageing", page_icon='images/neqsimlogocircleflat.png')
apply_theme()

col1, col2 = st.columns([30,70])

with col2:
    # Streamlit page configuration
    st.title('LNG Ageing Simulation')

with col1:
    st.image('images/LNGship.jpg')

st.divider()
"""
Calculating the aging of LNG during transportation involves considering various factors such as temperature, pressure, composition, and transport time. However, it is important to note that LNG aging is a complex process influenced by many variables, and there is no standardized calculation method. In the following method we assume a gas composition in equilibrium with the LNG at the bubble point. The Peng Robinson EOS is used for thermodynamic calculations and the Klosek-McKinley method is used for calculating LNG density.

To estimate LNG aging, NeqSim considers the following factors:

1. Boil-off gas (BOG): BOG refers to the vaporization of LNG that occurs during storage and transportation. It mainly consists of lighter hydrocarbons that evaporate more easily than the heavier. The rate of BOG formation depends on factors such as temperature, pressure, insulation, and containment system efficiency. By monitoring and measuring the BOG, you can estimate the extent of aging.

2. Composition changes: As LNG ages, lighter components, such as nitrogen and methane, can evaporate more readily than heavier hydrocarbons. This ageing process leads to changes in the LNG composition. The rate of composition change depends on factors like transport time, temperature, and initial composition.

"""
st.divider()

with st.expander("📋 Initial LNG Composition", expanded=True):
    # Reset button to restore default composition
    if st.button('Reset to Default Composition'):
        st.session_state.lng_fluid_df = pd.DataFrame(lng_fluid)
        st.rerun()

    hidecomponents = st.checkbox('Show active components')
    if hidecomponents and 'lng_edited_df' in st.session_state:
        st.session_state.lng_fluid_df = st.session_state.lng_edited_df[
            st.session_state.lng_edited_df['MolarComposition[-]'] > 0
        ]

    if 'lng_uploaded_file' in st.session_state and st.session_state.lng_uploaded_file is not None and not hidecomponents:
        try:
            st.session_state.lng_fluid_df = pd.read_csv(st.session_state.lng_uploaded_file)
            numeric_columns = ['MolarComposition[-]']
            st.session_state.lng_fluid_df[numeric_columns] = st.session_state.lng_fluid_df[numeric_columns].astype(float)
        except Exception as e:
            st.warning(f'Could not load file: {e}')
            st.session_state.lng_fluid_df = pd.DataFrame(lng_fluid)

    # Initialize fluid DataFrame if not present
    if 'lng_fluid_df' not in st.session_state:
        st.session_state.lng_fluid_df = pd.DataFrame(lng_fluid)

    # Create an editable data table for the fluid composition
    st.edited_df = st.data_editor(
        st.session_state.lng_fluid_df,
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
        num_rows='dynamic'
    )

    # Store edited df for later use
    st.session_state.lng_edited_df = st.edited_df

    st.caption("💡 Fluid composition will be normalized before simulation")
# Add a visual divider
st.divider()

# LNG Ageing Simulation Parameters
st.subheader('LNG Ageing Simulation Parameters')

# Input for transport pressure in bara
pressure_transport = st.number_input('Transport Pressure (bara)', min_value=0.0, value=1.01325)
# Input for initial volume in cubic meters
volume_initial = st.number_input('Initial Volume (m3)', min_value=0.0, value=120000.0)
# Input for boil-off rate in percentage
BOR = st.number_input('Boil-off Rate (%)', min_value=0.0, value=0.15)
time_transport = st.number_input('Transport Time (hours)', min_value=0.0, value=24.0)
standard_version = st.selectbox(
    'ISO6976 standard version:',
    ('2004', '2016'),
    index=1  # Default to the second option, which is '2016'
)
energy_ref_temp = st.selectbox(
    'ISO6976 energy reference temperature:',
    (0, 15, 15.55, 20, 25),
    index=1  # Default to the second option, which is '2016'
)
volume_ref_temp = st.selectbox(
    'ISO6976 volume reference temperature:',
    (0, 15, 15.55, 20),
    index=1  # Default to the second option, which is '2016'
)
if st.button('Simulate Ageing'):
    if st.edited_df['MolarComposition[-]'].sum() > 0:
        with st.spinner('Running LNG ageing simulation...'):
            try:
                # Create fluid from user input
                fluid = fluid_df(st.edited_df).autoSelectModel()
                fluid.setPressure(pressure_transport, 'bara')
                fluid.setTemperature(-160.0, "C")  # setting a guessed initial temperature
                
                # Creating ship system for LNG ageing
                ship = jneqsim.fluidmechanics.flowsystem.twophaseflowsystem.shipsystem.LNGship(fluid, volume_initial, BOR / 100.0)
                ship.useStandardVersion("", standard_version)
                ship.getStandardISO6976().setEnergyRefT(energy_ref_temp)
                ship.getStandardISO6976().setVolRefT(volume_ref_temp)
                ship.setEndTime(time_transport)
                ship.createSystem()
                ship.solveSteadyState(0)
                ship.solveTransient(0)
                ageingresults = ship.getResults("temp")

                # Assuming ageingresults is already obtained from the simulation
                results = ageingresults[1:]  # Data rows
                columns = ageingresults[0]   # Column headers

                # Clean the column names to ensure uniqueness and handle empty or None values
                cleaned_columns = []
                seen = set()
                for i, col in enumerate(columns):
                    new_col = col if col not in (None, '') else f"Unnamed_{i}"
                    if new_col in seen:
                        new_col = f"{new_col}_{i}"
                    seen.add(new_col)
                    cleaned_columns.append(new_col)

                # Creating DataFrame from results with cleaned column names
                resultsDF = pd.DataFrame([[float(str(j).replace(',', '')) for j in i] for i in results], columns=cleaned_columns)
                resultsDF.columns = ['time', 'temperature','WI','GCV','density','volume','C1','C2','C3','iC4','nC4','iC5','nC5','C6','N2','energy', 'GCV_mass', 'gC1','gC2','gC3','giC4','gnC4','giC5','gnC5','gC6','gN2']

                # Displaying the results DataFrame in Streamlit
                st.subheader('Ageing Simulation Results')
                st.dataframe(resultsDF)

                # Function to convert DataFrame to Excel and offer download
                def convert_df_to_excel(df):
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False)
                    processed_data = output.getvalue()
                    return processed_data

                # Download button for the results in Excel format
                excel_data = convert_df_to_excel(resultsDF)
                st.download_button(label='📥 Download Excel',
                                data=excel_data,
                                file_name='lng_ageing_results.xlsx',
                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                st.divider()
                """
                Units:

                Temperature : Celsius

                GCV: MJ/(S)m3

                WI: MJ/(S)m3

                Density: kg/m3

                Volume: m3
                
                Composition: C1-C6 molar fraction of LNG, gC1-gC6 molar fraction of boil off gas 

                GCV_mass: MJ/kg

                Energy: MJ (total energy)
                
                """
            except Exception as e:
                st.error(f'Simulation failed: {str(e)}')
    else:
        st.error('The sum of Molar Composition must be greater than 0. Please adjust your inputs.')

    
st.sidebar.file_uploader("Import Fluid", key='lng_uploaded_file', help='Fluids can be saved by hovering over the fluid window and clicking the "Download as CSV" button in the upper-right corner.')