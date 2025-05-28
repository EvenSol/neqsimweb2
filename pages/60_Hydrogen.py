import streamlit as st
import pandas as pd
import neqsim
from neqsim.thermo import fluid, TPflash

st.title("Hydrogen Properties")
st.write(
    """
    This application calculates properties of hydrogen using the **Leachman model** for thermal properties and the **Modified Muzny model** for viscosity(Modified by master students at NTNU).
    
    Select the hydrogen type and specify the temperature (°C) and pressure (bara).

    **Hydrogen types:**
    - **Normal Hydrogen:** The natural equilibrium mixture (approximately **3:1 ortho to para**).
    - **Para Hydrogen:** **100% para hydrogen**.
    - **Ortho Hydrogen:** **100% ortho hydrogen**.

    **Models used:**
    - **Leachman model:** Thermodynamic properties (e.g., density, enthalpy, heat capacity, etc.).
    - **Modified Muzny model:** Viscosity (gas only).
    
    **Uncertainties:**
    - Density: Typically within **0.04%** for **250–450 K** and **p < 300 MPa**, increasing at lower temperatures and higher pressures.
    - Heat capacity: Within **1%** across the range.
    - Speed of sound: Within **0.5% for p < 100 MPa**.
    - Viscosity: **Unknown**.
    """
)


# Hydrogen type selection
hydrogen_type = st.selectbox("Select Hydrogen Type", ["Normal Hydrogen", "Para Hydrogen", "Ortho Hydrogen"])

# Mapping dictionary for hydrogen types
hydrogen_type_mapping = {
    "normal hydrogen": "normal",
    "para hydrogen": "para",
    "ortho hydrogen": "ortho"
}

# Convert the selected hydrogen type to the expected format
mapped_hydrogen_type = hydrogen_type_mapping[hydrogen_type.lower()]

# Initialize session state for temperature and pressure input
if 'tp_flash_data' not in st.session_state:
    st.session_state['tp_flash_data'] = pd.DataFrame({
        'Temperature (C)': [20.0, 25.0],  # Default example temperature
        'Pressure (bara)': [1.0, 10.0]  # Default example pressure
    })

st.divider()
st.text("Input Pressures and Temperatures")
st.session_state.tp_flash_data = st.data_editor(
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

if st.button('Run Hydrogen Property Calculations'):
    if st.session_state.tp_flash_data.empty:
        st.error('No data to perform calculations. Please input temperature and pressure values.')
    else:
        results_list = []
        neqsim_fluid = fluid("srk")  # Use SRK EoS
        neqsim_fluid.addComponent('hydrogen', 1.0, "mol/sec")  # Add hydrogen component
        
        for idx, row in st.session_state.tp_flash_data.iterrows():
            temp = row['Temperature (C)']
            pressure = row['Pressure (bara)']
            neqsim_fluid.setPressure(pressure, 'bara')
            neqsim_fluid.setTemperature(temp, 'C')
            TPflash(neqsim_fluid)
            neqsim_fluid.initThermoProperties()
            neqsim_fluid.getPhase(0).getPhysicalProperties().setViscosityModel("Muzny_mod")
            neqsim_fluid.initPhysicalProperties()
            
            
            # Check number of phases
            num_phases = neqsim_fluid.getNumberOfPhases()
            
            if num_phases > 0:
                phase = neqsim_fluid.getPhase(0)
                try:
                    density = phase.getDensity_Leachman(mapped_hydrogen_type)
                    properties = phase.getProperties_Leachman(mapped_hydrogen_type)
                    if phase.getPhaseTypeName() == "gas":
                        viscosity = neqsim_fluid.getPhase(0).getPhysicalProperties().getViscosity()*10**6
                    else:
                        viscosity = "N/A for this phase"
                    #viscosity = neqsim_fluid.getPhase(0).getPhysicalProperties().getViscosity()*10**6
                    results_list.append({
                        "Temperature (C)": temp,
                        "Pressure (bara)": pressure,
                        "Phase type": phase.getPhaseTypeName(),
                        "Density (kg/m³)": density,
                        #"Pressure [kPa]": properties[0],
                        "Compressibility factor": properties[1],
                        #"d(P)/d(rho) [kPa/(mol/l)]": properties[2],
                        #"d^2(P)/d(rho)^2 [kPa/(mol/l)^2]": properties[3],
                        #"d^2(P)/d(T^2) [kPa/K]": properties[4],
                        #"d(P)/d(T) [kPa/K]": properties[5],
                        "Energy [J/mol]": properties[6],
                        "Enthalpy [J/mol]": properties[7],
                        "Entropy [J/mol-K]": properties[8],
                        "Isochoric heat capacity [J/mol-K]": properties[9],
                        "Isobaric heat capacity [J/mol-K]": properties[10],
                        "Speed of sound [m/s]": properties[11],
                        "Gibbs energy [J/mol]": properties[12],
                        "Joule-Thomson coefficient [K/kPa]": properties[13],
                        "Isentropic exponent": properties[14],
                        "Viscosity [\u03BCPa·s]": viscosity
                    })
                except AttributeError:
                    st.error("Leachman model calculations are not available for the selected phase.")
                    break
            else:
                st.error("No valid phases found for Leachman model calculations.")
        
        st.success('Hydrogen property calculations completed!')
        st.subheader("Results:")
        combined_results = pd.DataFrame(results_list)
        st.data_editor(combined_results)

st.sidebar.file_uploader("Upload Data", key='uploaded_file', help='Upload a CSV file containing temperature and pressure values.')
