import streamlit as st
import pandas as pd
import neqsim
from neqsim.thermo import fluid, TPflash

st.title("Helium Properties")
st.write(
    """
    This application calculates properties of helium using the **VEGA** equation of state.

    Select the helium type and specify the temperature (°C) and pressure (bara).

    **Models used:**
    - **VEGA EoS:** Thermodynamic properties (e.g., density, enthalpy, heat capacity, etc.).
    - **Modified KTA model:** Viscosity (gas only).(Modified by master students at NTNU)

    
    **Uncertainties:**
    - Density: Typically within **0.02% up to 40 MPa**, increasing at higher pressures.
    - Sound speed: Within **0.02% up to 60 MPa**, but larger deviations at higher pressures.
    - Heat capacity: Typically within **3%**, except near the λ-transition and critical point.
    - Viscosity: **Unknown**.
    """
)

# Helium phase selection (Normal Fluid Only)
st.write("Helium calculations are valid **only for normal helium (helium I)** up to **1500 K and 2000 MPa**.")

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
    num_rows='dynamic',
    column_config={
        'Temperature (C)': st.column_config.NumberColumn(
            label="Temperature (C)",
            min_value=-270.9732,
            max_value=1500,
            format='%f',
            help='Enter the temperature in degrees Celsius.'
        ),
        'Pressure (bara)': st.column_config.NumberColumn(
            label="Pressure (bara)",
            min_value=0.0,
            max_value=2000,
            format='%f',
            help='Enter the pressure in bar absolute.'
        ),
    }
)

if st.button('Run Helium Property Calculations'):
    if st.session_state.tp_flash_data.empty:
        st.error('No data to perform calculations. Please input temperature and pressure values.')
    else:
        results_list = []
        neqsim_fluid = fluid("srk")  # Use SRK EoS (VEGA EoS needs implementation)
        neqsim_fluid.addComponent('helium', 1.0, "mol/sec")  # Add helium component
        
        for idx, row in st.session_state.tp_flash_data.iterrows():
            temp = row['Temperature (C)']
            pressure = row['Pressure (bara)']
            neqsim_fluid.setPressure(pressure, 'bara')
            neqsim_fluid.setTemperature(temp, 'C')
            TPflash(neqsim_fluid)
            neqsim_fluid.initThermoProperties()
            neqsim_fluid.getPhase(0).getPhysicalProperties().setViscosityModel("KTA_mod")
            neqsim_fluid.initPhysicalProperties()

            
            # Check number of phases
            num_phases = neqsim_fluid.getNumberOfPhases()
            #st.write(f"Number of detected phases: {num_phases}")

            if num_phases > 0:
                phase = neqsim_fluid.getPhase(0)
                #st.write(f"Phase type: {phase.getPhaseTypeName()}")
                try:
                    density = phase.getDensity_Vega()
                    properties = phase.getProperties_Vega()
                    if phase.getPhaseTypeName() == "gas":
                        viscosity = neqsim_fluid.getPhase(0).getPhysicalProperties().getViscosity()*10**6
                    else:
                        viscosity = "N/A for this phase"
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
                        "Joule-Thomson coefficient [K/bar]": properties[13]*100,
                        "Isentropic exponent": properties[14],
                        "Viscosity [\u03BCPa·s]": viscosity
                    })
                except AttributeError:
                    st.error("VEGA model calculations are not available for the selected phase.")
                    break
            else:
                st.error("No valid phases found for VEGA model calculations.")

        st.success('Helium property calculations completed!')
        st.subheader("Results:")
        combined_results = pd.DataFrame(results_list)
        st.data_editor(combined_results)

st.sidebar.file_uploader("Upload Data", key='uploaded_file', help='Upload a CSV file containing temperature and pressure values.')
