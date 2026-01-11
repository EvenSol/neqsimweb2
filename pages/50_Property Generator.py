import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from theme import apply_theme, theme_toggle

# NeqSim imports
import neqsim
from neqsim.thermo.thermoTools import fluid_df, TPflash
from fluids import default_fluid

# ----------------------------------------------------------------------------------
# HELPER FUNCTION: Compute Property
# ----------------------------------------------------------------------------------


def compute_property(neqsim_fluid, phase_name: str, property_name: str):
    """
    Compute the selected property from the specified phase or the overall system.
    Handles both general and component-specific properties.
    """
    def get_phase_number(fluid, p_name):
        try:
            return fluid.getPhaseNumberOfPhase(p_name)
        except:
            return None

    # Handle component-specific properties
    component_specific_prefixes = [
        "mole fraction ",
        "wt fraction ",
        "activity coefficient ",
        "fugacity coefficient ",
    ]

    if any(property_name.startswith(prefix) for prefix in component_specific_prefixes):
        if " " not in property_name:
            return f"Invalid property name: {property_name}"
        prop_type, component = property_name.split(" ", 1)
        prop_type = prop_type.lower()

        # Determine phase
        if phase_name.lower() == "overall":
            phase_num = 0  # Overall phase is typically phase 0
        else:
            phase_num = get_phase_number(neqsim_fluid, phase_name)
            if phase_num is None:
                return f"No {phase_name} phase"

        if phase_name.lower() != "overall":
            phase = neqsim_fluid.getPhase(phase_num)
        else:
            phase = neqsim_fluid.getPhase(0)

        # Check if component exists in the phase
        component_obj = phase.getComponent(component)
        if component_obj is None:
            return f"Component '{component}' not found in {phase_name} phase"

        if prop_type == "mole fraction":
            return component_obj.getx()
        elif prop_type == "wt fraction":
            # Calculate weight fraction
            return (component_obj.getx() * component_obj.getMolarMass()) / neqsim_fluid.getMolarMass()
        elif prop_type == "activity coefficient":
            return phase.getActivityCoefficient(component_obj.getComponentNumber())
        elif prop_type == "fugacity coefficient":
            return component_obj.getFugacityCoefficient()
        else:
            return f"Property type '{prop_type}' not recognized."

    # Handle general properties
    if phase_name.lower() == "overall":
        if property_name == "density":
            return neqsim_fluid.getDensity()
        elif property_name == "viscosity":
            return neqsim_fluid.getViscosity()
        elif property_name == "compressibility":
            return neqsim_fluid.getZ()
        elif property_name == "JouleThomson coef.":
            return neqsim_fluid.getJouleThomsonCoefficient()
        elif property_name == "heat capacity Cp":
            return neqsim_fluid.getCp() / (neqsim_fluid.getMolarMass() * neqsim_fluid.getTotalNumberOfMoles() * 1000.0)
        elif property_name == "heat capacity Cv":
            return neqsim_fluid.getCv() / (neqsim_fluid.getMolarMass() * neqsim_fluid.getTotalNumberOfMoles() * 1000.0)
        elif property_name == "enthalpy":
            return neqsim_fluid.getEnthalpy() / (neqsim_fluid.getMolarMass() * neqsim_fluid.getTotalNumberOfMoles() * 1000.0)
        elif property_name == "entropy":
            return neqsim_fluid.getEntropy() / (neqsim_fluid.getMolarMass() * neqsim_fluid.getTotalNumberOfMoles() * 1000.0)
        elif property_name == "phase fraction (mole)":
            return neqsim_fluid.getPhase(0).getBeta()
        elif property_name == "phase fraction (volume)":
            return neqsim_fluid.getPhase(0).getVolume() / neqsim_fluid.getVolume()
        elif property_name == "phase fraction (mass)":
            phase_mass = neqsim_fluid.getPhase(
                0).getNumberOfMolesInPhase() * neqsim_fluid.getPhase(0).getMolarMass()
            total_mass = neqsim_fluid.getTotalNumberOfMoles() * neqsim_fluid.getMolarMass()
            return phase_mass / total_mass
        elif property_name == "number of phases":
            nop = neqsim_fluid.getNumberOfPhases()
            if nop == 1:
                return nop, None, None
            elif nop == 2:
                return nop, neqsim_fluid.getPhase(1).getNumberOfMolesInPhase() * neqsim_fluid.getPhase(1).getMolarMass() / (neqsim_fluid.getPhase(0).getNumberOfMolesInPhase()*23.64/1e9), None
            elif nop == 3:
                return nop, neqsim_fluid.getPhase(1).getNumberOfMolesInPhase() * neqsim_fluid.getPhase(1).getMolarMass() / (neqsim_fluid.getPhase(0).getNumberOfMolesInPhase()*23.64/1e9), neqsim_fluid.getPhase(2).getNumberOfMolesInPhase() * neqsim_fluid.getPhase(2).getMolarMass() / (neqsim_fluid.getPhase(0).getNumberOfMolesInPhase()*23.64/1e9)
        elif property_name == "gas-oil interfacial tension":
            if neqsim_fluid.hasPhaseType("gas") and neqsim_fluid.hasPhaseType("oil"):
                neqsim_fluid.calcInterfaceProperties()
                return neqsim_fluid.getInterphaseProperties().getSurfaceTension(
                    neqsim_fluid.getPhaseNumberOfPhase("gas"),
                    neqsim_fluid.getPhaseNumberOfPhase("oil")
                )
            else:
                return "No gas-oil interface"
        elif property_name == "gas-aqueous interfacial tension":
            if neqsim_fluid.hasPhaseType("gas") and neqsim_fluid.hasPhaseType("aqueous"):
                neqsim_fluid.calcInterfaceProperties()
                return neqsim_fluid.getInterphaseProperties().getSurfaceTension(
                    neqsim_fluid.getPhaseNumberOfPhase("gas"),
                    neqsim_fluid.getPhaseNumberOfPhase("aqueous")
                )
            else:
                return "No gas-aqueous interface"
        elif property_name == "oil-aqueous interfacial tension":
            if neqsim_fluid.hasPhaseType("oil") and neqsim_fluid.hasPhaseType("aqueous"):
                neqsim_fluid.calcInterfaceProperties()
                return neqsim_fluid.getInterphaseProperties().getSurfaceTension(
                    neqsim_fluid.getPhaseNumberOfPhase("oil"),
                    neqsim_fluid.getPhaseNumberOfPhase("aqueous")
                )
            else:
                return "No oil-aqueous interface"
        elif property_name == "wc":
            if neqsim_fluid.hasPhaseType("oil") and neqsim_fluid.hasPhaseType("aqueous"):
                return neqsim_fluid.getPhase("aqueous").getVolume() / (neqsim_fluid.getPhase("oil").getVolume() + neqsim_fluid.getPhase("aqueous").getVolume()), \
                    neqsim_fluid.getPhase("aqueous").getNumberOfMolesInPhase() * neqsim_fluid.getPhase("aqueous").getMolarMass() / (neqsim_fluid.getPhase(0).getNumberOfMolesInPhase()*23.64/1e9),\
                    neqsim_fluid.getPhase("oil").getNumberOfMolesInPhase() * neqsim_fluid.getPhase("oil").getMolarMass() / (neqsim_fluid.getPhase(0).getNumberOfMolesInPhase()*23.64/1e9)
            elif neqsim_fluid.hasPhaseType("oil"):
                return 0, 0, neqsim_fluid.getPhase("oil").getNumberOfMolesInPhase() * neqsim_fluid.getPhase("oil").getMolarMass() / (neqsim_fluid.getPhase(0).getNumberOfMolesInPhase()*23.64/1e9)
            elif neqsim_fluid.hasPhaseType("aqueous"):
                return 1, neqsim_fluid.getPhase("aqueous").getNumberOfMolesInPhase() * neqsim_fluid.getPhase("aqueous").getMolarMass() / (neqsim_fluid.getPhase(0).getNumberOfMolesInPhase()*23.64/1e9), 0
            else:
                return np.nan
        else:
            return f"{property_name} not defined for overall system"
    else:
        # Phase-specific properties
        if not neqsim_fluid.hasPhaseType(phase_name):
            return f"No {phase_name} phase"
        else:
            phase_num = get_phase_number(neqsim_fluid, phase_name)
            phase = neqsim_fluid.getPhase(phase_num)

            if property_name == "density":
                return phase.getPhysicalProperties().getDensity()
            elif property_name == "viscosity":
                return phase.getPhysicalProperties().getViscosity()
            elif property_name == "compressibility":
                return phase.getZ()
            elif property_name == "JouleThomson coef.":
                return phase.getJouleThomsonCoefficient()
            elif property_name == "heat capacity Cp":
                return phase.getCp() / (phase.getMolarMass() * phase.getNumberOfMolesInPhase() * 1000.0)
            elif property_name == "heat capacity Cv":
                return phase.getCv() / (phase.getMolarMass() * phase.getNumberOfMolesInPhase() * 1000.0)
            elif property_name == "enthalpy":
                return phase.getEnthalpy() / (phase.getMolarMass() * phase.getNumberOfMolesInPhase() * 1000.0)
            elif property_name == "entropy":
                return phase.getEntropy() / (phase.getMolarMass() * phase.getNumberOfMolesInPhase() * 1000.0)
            elif property_name == "phase fraction (mole)":
                return phase.getBeta()
            elif property_name == "phase fraction (volume)":
                return phase.getVolume() / neqsim_fluid.getVolume()
            elif property_name == "phase fraction (mass)":
                phase_mass = phase.getNumberOfMolesInPhase() * phase.getMolarMass()
                total_mass = neqsim_fluid.getTotalNumberOfMoles() * neqsim_fluid.getMolarMass()
                return phase_mass / total_mass
            elif property_name == "number of phases":
                return neqsim_fluid.getNumberOfPhases()
            elif property_name == "gas-oil interfacial tension":
                if neqsim_fluid.hasPhaseType("gas") and neqsim_fluid.hasPhaseType("oil"):
                    neqsim_fluid.calcInterfaceProperties()
                    return neqsim_fluid.getInterphaseProperties().getSurfaceTension(
                        neqsim_fluid.getPhaseNumberOfPhase("gas"),
                        neqsim_fluid.getPhaseNumberOfPhase("oil")
                    )
                else:
                    return "No gas-oil interface"
            elif property_name == "gas-aqueous interfacial tension":
                if neqsim_fluid.hasPhaseType("gas") and neqsim_fluid.hasPhaseType("aqueous"):
                    neqsim_fluid.calcInterfaceProperties()
                    return neqsim_fluid.getInterphaseProperties().getSurfaceTension(
                        neqsim_fluid.getPhaseNumberOfPhase("gas"),
                        neqsim_fluid.getPhaseNumberOfPhase("aqueous")
                    )
                else:
                    return "No gas-aqueous interface"
            elif property_name == "oil-aqueous interfacial tension":
                if neqsim_fluid.hasPhaseType("oil") and neqsim_fluid.hasPhaseType("aqueous"):
                    neqsim_fluid.calcInterfaceProperties()
                    return neqsim_fluid.getInterphaseProperties().getSurfaceTension(
                        neqsim_fluid.getPhaseNumberOfPhase("oil"),
                        neqsim_fluid.getPhaseNumberOfPhase("aqueous")
                    )
                else:
                    return "No oil-aqueous interface"
            else:
                return f"{property_name} not defined for {phase_name} phase"

# ----------------------------------------------------------------------------------
# MAIN STREAMLIT APPLICATION
# ----------------------------------------------------------------------------------


def main():
    st.title("Property Generator")

    st.write("""
    This application allows you to define a fluid composition, set up a grid of temperatures and pressures,
    select a phase and a property, and calculate the selected property across the defined grid. Use the left sidebar to define the grid and select the property to calculate.
    """)

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
    
    # ----------------------------------------------------------------------------------
    # SIDEBAR: Grid Setup and Property Selection
    # ----------------------------------------------------------------------------------
    st.sidebar.header("Grid Setup & Property Selection")
    
    # Temperature Range Inputs
    min_temp = st.sidebar.number_input("Min Temperature [°C]", value=20.0, step=1.0)
    max_temp = st.sidebar.number_input("Max Temperature [°C]", value=80.0, step=1.0)
    n_temp = st.sidebar.number_input("Number of Temperature Points", value=5, min_value=1, step=1)
    
    # Pressure Range Inputs
    min_pres = st.sidebar.number_input("Min Pressure [bara]", value=1.0, step=1.0)
    max_pres = st.sidebar.number_input("Max Pressure [bara]", value=100.0, step=1.0)
    n_pres = st.sidebar.number_input("Number of Pressure Points", value=5, min_value=1, step=1)
    
    # Phase Selection
    phase_options = ["overall", "gas", "oil", "aqueous"]
    phase_name = st.sidebar.selectbox("Select Phase", phase_options, index=0)
    
    # Dynamically generate property options based on current fluid composition
    component_names = st.edited_df["ComponentName"].tolist()
    
    # Base property names
    base_property_names = [
        "wc",
        "density",
        "viscosity",
        "compressibility",
        "JouleThomson coef.",
        "heat capacity Cp",
        "heat capacity Cv",
        "enthalpy",
        "entropy",
        "phase fraction (mole)",
        "phase fraction (volume)",
        "phase fraction (mass)",
        "number of phases",
        "gas-oil interfacial tension",
        "gas-aqueous interfacial tension",
        "oil-aqueous interfacial tension",
    ]
    
    # Component-specific properties
    component_specific_properties = []
    for comp in component_names:
        component_specific_properties.extend([
            f"mole fraction {comp}",
            f"wt fraction {comp}",
            f"activity coefficient {comp}",
            f"fugacity coefficient {comp}",
        ])
    
    # Combined property options
    all_property_options = base_property_names + component_specific_properties
    
    # Property Selection
    property_name = st.sidebar.selectbox("Select Property", all_property_options, index=0)
    
    # ----------------------------------------------------------------------------------
    # UNIT MAPPING
    # ----------------------------------------------------------------------------------
    # Define units for base properties
    unit_dict = {
        "density": "kg/m³",
        "viscosity": "Pa·s",
        "compressibility": "unitless",
        "JouleThomson coef.": "K/J",
        "heat capacity Cp": "J/(kg·K)",
        "heat capacity Cv": "J/(kg·K)",
        "enthalpy": "J/kg",
        "entropy": "J/(kg·K)",
        "phase fraction (mole)": "unitless",
        "phase fraction (volume)": "unitless",
        "phase fraction (mass)": "unitless",
        "number of phases": "unitless",
        "gas-oil interfacial tension": "N/m",
        "gas-aqueous interfacial tension": "N/m",
        "oil-aqueous interfacial tension": "N/m",
    }
    
    # For component-specific properties, units are unitless
    component_specific_units = {
        "mole fraction": "unitless",
        "wt fraction": "unitless",
        "activity coefficient": "unitless",
        "fugacity coefficient": "unitless",
    }
    
    # Determine unit based on selected property
    unit = "unitless"  # Default
    for prefix in component_specific_units:
        if property_name.startswith(prefix):
            unit = component_specific_units[prefix]
            break
    else:
        unit = unit_dict.get(property_name, "")
        
    # ----------------------------------------------------------------------------------
    # SIDEBAR: Thermodynamic Model Selection
    # ----------------------------------------------------------------------------------
    st.sidebar.header("Thermodynamic Model Selection")
    
    # Dropdown for selecting the thermodynamic model, including an automatic option
    thermo_model_options = {
        'Automatic': None,  # This option will allow NeqSim to auto-select the model
        'SRK': "srk",
        'PR': "pr",
        'CPA': "cpa",
        'UMR': "umr"
    }
    thermo_model_choice = st.sidebar.selectbox("Select Thermodynamic Model", list(thermo_model_options.keys()))
    
    # ----------------------------------------------------------------------------------
    # RUN CALCULATION BUTTON
    # ----------------------------------------------------------------------------------
    run_button = st.button("Run Grid Calculations")
    
    # ----------------------------------------------------------------------------------
    # DISPLAY RESULTS IN 2D GRID
    # ----------------------------------------------------------------------------------
    if run_button:
        # 1) Check fluid composition
        total_molar_frac = st.edited_df["MolarComposition[-]"].sum()
        if total_molar_frac <= 0:
            st.error("Total Molar Composition must be greater than 0.")
        else:
            # 2) Normalize fluid composition
            normalized_df = st.edited_df.copy()
            normalized_df["MolarComposition[-]"] = normalized_df["MolarComposition[-]"] / total_molar_frac

            # 3) Build fluid using NeqSim
            try:
                # Get the selected model or None for automatic selection
                modelName = thermo_model_options[thermo_model_choice]
                if modelName:  # A specific model was selected
                    neqsim_fluid = fluid_df(
                        normalized_df,
                        lastIsPlusFraction=isplusfluid,
                        modelName=modelName,  # Model name must correspond to NeqSim's supported models
                        add_all_components=False
                    )
                    neqsim_fluid.autoSelectMixingRule()
                else:  # Automatic selection
                    neqsim_fluid = fluid_df(
                        normalized_df,
                        lastIsPlusFraction=isplusfluid,
                        add_all_components=False
                    ).autoSelectModel()
                neqsim_fluid.setMultiPhaseCheck(True)
            except Exception as e:
                st.error(f"Error creating fluid: {e}")
                st.stop()

            # 4) Generate Temperature and Pressure arrays
            if n_temp > 1:
                T_range = np.linspace(min_temp, max_temp, int(n_temp)).tolist()
            else:
                T_range = [min_temp]

            if n_pres > 1:
                P_range = np.linspace(min_pres, max_pres, int(n_pres)).tolist()
            else:
                P_range = [min_pres]

            # 5) Initialize results list
            results_list = []

            # Initialize matrices for phase_mass and phase_mass2
            phase_mass_list = []
            phase_mass2_list = []

            # 6) Loop over Pressure and Temperature
            for P in P_range:
                row = {"Pressure [bara]": P}
                rowm = {"Pressure [bara]": P}
                rowm2 = {"Pressure [bara]": P}
                
                for T in T_range:
                    try:
                        neqsim_fluid.setTemperature(T, "C")
                        neqsim_fluid.setPressure(P, "bara")
                        TPflash(neqsim_fluid)

                        # Initialize physical properties
                        neqsim_fluid.init(3)
                        neqsim_fluid.initPhysicalProperties()

                        # Compute the selected property
                        if property_name == "number of phases" or property_name == "wc":
                            value, phase_mass, phase_mass2 = compute_property(neqsim_fluid, phase_name, property_name)
                            col_name = f"T={T:.2f} °C"
                            if phase_mass is not None:
                                rowm[col_name] = f"{phase_mass:.2g}"  # Round to 2 significant figures if not None
                            else:
                                rowm[col_name] = None  # Or set a default value or keep as None
                            
                            if phase_mass2 is not None:
                                rowm2[col_name] = f"{phase_mass2:.2g}"  # Round to 2 significant figures if not None
                            else:
                                rowm2[col_name] = None  # Or set a default value or keep as None
                        else:
                            value = compute_property(neqsim_fluid, phase_name, property_name)

                        # Assign to row with temperature as column
                        col_name = f"T={T:.2f} °C"
                        row[col_name] = value
                    except Exception as e:
                        col_name = f"T={T:.2f} °C"
                        row[col_name] = f"Error: {e}"
                results_list.append(row)
                
                if property_name == "number of phases" or property_name == "wc":
                    phase_mass_list.append(rowm)
                    phase_mass2_list.append(rowm2)

            # 7) Convert results to DataFrame
            results_df = pd.DataFrame(results_list)
            results_long_df = results_df.melt(id_vars=["Pressure [bara]"], var_name="Temperature", value_name=property_name)
            
            # This should be done only if 'number of phases' is the selected property
            if property_name == "number of phases" or property_name == "wc":
                phase_mass_df = pd.DataFrame(phase_mass_list)
                phase_mass2_df = pd.DataFrame(phase_mass2_list)
                                
                # Melt the phase mass dataframes to long format
                phase_mass_long_df = phase_mass_df.melt(id_vars=["Pressure [bara]"], var_name="Temperature", value_name="kg/MSm3")
                phase_mass2_long_df = phase_mass2_df.melt(id_vars=["Pressure [bara]"], var_name="Temperature", value_name="kg/MSm3_b")
                

                # Merge the mass dataframes with the results_long_df on the common columns
                results_long_df = pd.merge(results_long_df, phase_mass_long_df, on=["Pressure [bara]", "Temperature"], how='left')
                results_long_df = pd.merge(results_long_df, phase_mass2_long_df, on=["Pressure [bara]", "Temperature"], how='left')  
            
            # 8) Display unit above the table
            if unit:
                st.markdown(f"**Unit of `{property_name}`**: {unit}")
            else:
                st.markdown(f"**Unit of `{property_name}`**: Not defined")

            # 9) Display results
            st.success("Grid calculation completed successfully!")
            st.subheader(f"Results for '{property_name}' in '{phase_name}' phase")
            st.dataframe(results_df.style.hide(axis="index"), use_container_width=True)

            # 10) Download option
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv_data,
                file_name="neqsim_property_grid_results.csv",
                mime="text/csv"
            )

            # 11) Pablo added this to plot figure: #### Pablo start

            # Convert DataFrame to 2D numpy array for plotting
            data_with_errors = results_df.drop("Pressure [bara]", axis=1).to_numpy()

            # Replace non-numeric error values with np.nan
            data_for_heatmap = np.array([[np.nan if isinstance(value, str) and "Error" in value else value for value in row] for row in data_with_errors])

        
            # Create a figure and axis for the heatmap
            fig, ax = plt.subplots()
            heatmap = ax.imshow(data_for_heatmap, cmap='viridis', interpolation='nearest', aspect='auto')
        
            # Setting the aspect ratio to 'auto' will make the heatmap fill the figure while preserving the data proportions.
        
            # Set the tick labels for the heatmap
            ax.set_xticks(np.arange(len(T_range)))
            ax.set_yticks(np.arange(len(P_range)))
        
            # Labeling the tick labels with the actual temperature/pressure values
            ax.set_xticklabels([f"{T:.2f} °C" for T in T_range])
            ax.set_yticklabels([f"{P} bara" for P in P_range])
        
            # Rotate the tick labels so they don't overlap
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Reverse the y-axis (pressure) so it increases upwards
            ax.invert_yaxis()        

            # Add gridlines to the heatmap
            ax.grid(which='both', color='gray', linestyle='-', linewidth=1)

            # Make sure the grid is behind the heatmap
            ax.set_axisbelow(True)            

            # Add a colorbar next to the heatmap to indicate the color scale
            plt.colorbar(heatmap)
        
            # Display the figure using streamlit
            st.pyplot(fig)            
           
            # Create interactive plot with Plotly
            hover_data = [property_name]
            if 'kg/MSm3' in results_long_df.columns:
                hover_data.append('kg/MSm3')
            if 'kg/MSm3_b' in results_long_df.columns:
                hover_data.append('kg/MSm3_b')
                
            # Filter out the rows where 'property_name' contains errors
            results_long_df_clean = results_long_df[~results_long_df[property_name].astype(str).str.contains("Error")]

            fig2 = px.scatter(
                results_long_df_clean,
                x="Temperature",
                y="Pressure [bara]",
                color=property_name,  # Color by property value
                hover_data=hover_data,
                title=f"{property_name} across Temperature and Pressure"
            )
            
            # Adjust layout for better readability
            fig2.update_layout(
                xaxis_title="Temperature",
                yaxis_title="Pressure [bara]",
                coloraxis_colorbar=dict(
                    title=unit
                )
            )
            st.plotly_chart(fig2, use_container_width=True)
            #### Pablo ends


    
    
    st.divider()
    
    # ----------------------------------------------------------------------------------
    # OPTIONAL: File Uploader for Fluid Composition
    # ----------------------------------------------------------------------------------
    uploaded_file = st.sidebar.file_uploader(
        "Import Fluid Composition CSV",
        type=["csv"],
        help="Upload a CSV file with columns: ComponentName, MolarComposition[-], MolarMass[kg/mol], RelativeDensity[-]."
    )
    
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            # Validate required columns
            required_columns = {"ComponentName", "MolarComposition[-]", "MolarMass[kg/mol]", "RelativeDensity[-]"}
            if not required_columns.issubset(uploaded_df.columns):
                st.sidebar.error("Uploaded CSV does not contain the required columns.")
            else:
                st.session_state.activefluid_df = uploaded_df
                st.sidebar.success("Fluid composition imported successfully!")
        except Exception as e:
            st.sidebar.error(f"Error reading uploaded file: {e}")

if __name__ == "__main__":
    main()
