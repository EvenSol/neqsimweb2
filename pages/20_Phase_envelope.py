import streamlit as st
import pandas as pd
import neqsim
from neqsim.thermo import fluid_df, phaseenvelope, TPflash, dataFrame
from neqsim import jneqsim
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from fluids import detailedHC_data, fluid_library_selector
import numpy as np
from scipy.interpolate import UnivariateSpline
from theme import apply_theme

st.set_page_config(page_title="Phase Envelope", page_icon='images/neqsimlogocircleflat.png')
apply_theme()

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

with st.expander("📋 Fluid Composition", expanded=True):
    # Reset button to restore default composition
    if st.button('Reset to Default Composition'):
        st.session_state.phaseenv_fluid_df = pd.DataFrame(detailedHC_data)
        st.rerun()

    hidecomponents = st.checkbox('Show active components')

    # Only try to filter if phaseenv_edited_df has already been defined
    if hidecomponents and 'phaseenv_edited_df' in st.session_state:
        st.session_state.phaseenv_fluid_df = st.session_state.phaseenv_edited_df[
            st.session_state.phaseenv_edited_df['MolarComposition[-]'] > 0
        ]

    # Check for uploaded file in sidebar
    if 'phaseenv_uploaded_file' in st.session_state and st.session_state.phaseenv_uploaded_file is not None and not hidecomponents:
        try:
            st.session_state.phaseenv_fluid_df = pd.read_csv(st.session_state.phaseenv_uploaded_file)
            numeric_columns = ['MolarComposition[-]', 'MolarMass[kg/mol]', 'RelativeDensity[-]']
            st.session_state.phaseenv_fluid_df[numeric_columns] = (
                st.session_state.phaseenv_fluid_df[numeric_columns].astype(float)
            )
        except Exception as e:
            st.warning(f'Could not load file: {e}')
            st.session_state.phaseenv_fluid_df = pd.DataFrame(detailedHC_data)

    # If we don't yet have a fluid DataFrame, initialize it
    if 'phaseenv_fluid_df' not in st.session_state:
        st.session_state.phaseenv_fluid_df = pd.DataFrame(detailedHC_data)

    # -------------------------------------------------------------------------
    # 2. Let user edit the fluid data
    # -------------------------------------------------------------------------
    st.edited_df = st.data_editor(
        st.session_state.phaseenv_fluid_df,
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

    # Store edited df for later use
    st.session_state.phaseenv_edited_df = st.edited_df

    isplusfluid = st.checkbox('Plus Fluid')
    usePR = st.checkbox('Peng Robinson EoS', help='use standard Peng Robinson EoS')

    st.caption("💡 Fluid composition will be normalized before simulation")

with st.expander("📂 Fluid Library", expanded=False):
    if fluid_library_selector('phaseenv', 'phaseenv_fluid_df'):
        st.rerun()
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

        mixingrulename = neqsim_fluid.getMixingRuleName()
        modelname = neqsim_fluid.getModelName()
        
        # Calculate phase envelope
        thermoOps = jneqsim.thermodynamicoperations.ThermodynamicOperations(neqsim_fluid)
        thermoOps.calcPTphaseEnvelope2()
        
        # Get dew/bubble point data
        dewts = [x - 273.15 for x in list(thermoOps.getOperation().get("dewT"))]
        dewps = list(thermoOps.getOperation().get("dewP"))
        bubts = [x - 273.15 for x in list(thermoOps.getOperation().get("bubT"))]
        bubps = list(thermoOps.getOperation().get("bubP"))

        if len(dewts) == 0 and len(bubts) == 0:
            P_min = 1.0
            P_max = 150.0
            T_min = -50.0
            T_max = 150.0
            T_step = 15.0
            P_step = 5.0

            # Calculate phase envelope
            thermoOps.calcPTphaseEnvelopeNew3(P_min, P_max, T_min, T_max, P_step, T_step)
            op = thermoOps.getOperation()

            # Extract phase envelope points from Java op object
            pressurePhaseEnvelope = op.getPressurePhaseEnvelope()
            temperaturePhaseEnvelope = op.getTemperaturePhaseEnvelope()

            # Convert to numpy arrays for plotting
            pressurePhaseEnvelope_np = np.array(pressurePhaseEnvelope)
            temperaturePhaseEnvelope_np = np.array(temperaturePhaseEnvelope)

           # Use nearest-neighbor path for a more physical envelope
            points = np.column_stack((temperaturePhaseEnvelope_np, pressurePhaseEnvelope_np))
            n_points = len(points)
            used = np.zeros(n_points, dtype=bool)
            path = []
            current = 0
            path.append(current)
            used[current] = True
            for _ in range(1, n_points):
                dists = np.linalg.norm(points - points[current], axis=1)
                dists[used] = np.inf
                next_idx = np.argmin(dists)
                path.append(next_idx)
                used[next_idx] = True
                current = next_idx
            points_nn = points[path]
            temp_nn = points_nn[:,0]
            press_nn = points_nn[:,1]

            # Use the path index as the independent variable for spline interpolation
            path_idx = np.arange(len(temp_nn))

            # Spline interpolation along the nearest-neighbor path (not sorted by temperature)
            spline_temp = UnivariateSpline(path_idx, temp_nn, s=0, k=1)
            spline_press = UnivariateSpline(path_idx, press_nn, s=0, k=5)
            path_fine = np.linspace(0, len(temp_nn)-1, 500)
            temp_fine = spline_temp(path_fine)
            press_fine = spline_press(path_fine)
            cricondenbar = [0, 0]
            cricondentherm = [0, 0]

            # Cricondenbar: max pressure and corresponding temperature (on the spline)
            cricondenbar_idx = np.argmax(press_fine)
            cricondenbar[1] = press_fine[cricondenbar_idx]
            cricondenbar[0] = temp_fine[cricondenbar_idx]

            # Cricondentherm: max temperature and corresponding pressure (on the spline)
            cricondentherm_idx = np.argmax(temp_fine)
            cricondentherm[0] = temp_fine[cricondentherm_idx]
            cricondentherm[1] = press_fine[cricondentherm_idx]

            fig, ax = plt.subplots()
            plt.plot(temp_fine, press_fine, color='green')
            plt.plot(temp_nn, press_nn, '-o', color='purple', alpha=0.5, label='Phase Envelope')

            # Select 4 points near the top of the phase envelope (highest pressures)
            num_top = 4
            # Get indices of the 4 highest pressure points
            idx_top = np.argsort(press_nn)[-num_top:]
            # Sort these indices by temperature for a smooth plot
            idx_top_sorted = idx_top[np.argsort(top_temps := temp_nn[idx_top])]
            top_temps = temp_nn[idx_top_sorted]
            top_press = press_nn[idx_top_sorted]

            # Interpolate with a cubic spline (k=3) through these 4 points
            spline_top = UnivariateSpline(top_temps, top_press, s=0, k=3)
            t_fine_top = np.linspace(top_temps[0], top_temps[-1], 100)
            p_fine_top = spline_top(t_fine_top)

            # Calculate cricondenbar from the interpolated top region
            cricondenbar_idx_top = np.argmax(p_fine_top)
            cricondenbar[1] = p_fine_top[cricondenbar_idx_top]
            cricondenbar[0] = t_fine_top[cricondenbar_idx_top]

            plt.plot(t_fine_top, p_fine_top, color='blue', label='Cubic Spline (Top 4 Points)')

            plt.scatter([cricondenbar[0]], [cricondenbar[1]], color='red', zorder=5, label='Cricondenbar')
            plt.scatter([cricondentherm[0]], [cricondentherm[1]], color='orange', zorder=5, label='Cricondentherm')
            plt.xlabel('Temperature (°C)')
            plt.ylabel('Pressure (bara)')
            plt.grid(True)


            # Create interactive Plotly figure
            fig_plotly = go.Figure()
            
            # Phase envelope line (smooth spline)
            fig_plotly.add_trace(go.Scatter(
                x=temp_fine, y=press_fine,
                mode='lines',
                name='Phase Envelope',
                line=dict(color='#2E86AB', width=3),
                hovertemplate='T: %{x:.1f} °C<br>P: %{y:.2f} bara<extra></extra>'
            ))
            
            # Data points
            fig_plotly.add_trace(go.Scatter(
                x=temp_nn, y=press_nn,
                mode='markers',
                name='Calculated Points',
                marker=dict(color='#7B2CBF', size=8, opacity=0.6),
                hovertemplate='T: %{x:.1f} °C<br>P: %{y:.2f} bara<extra></extra>'
            ))
            
            # Cricondenbar point
            fig_plotly.add_trace(go.Scatter(
                x=[cricondenbar[0]], y=[cricondenbar[1]],
                mode='markers+text',
                name=f'Cricondenbar ({cricondenbar[1]:.1f} bara)',
                marker=dict(color='#E63946', size=14, symbol='diamond'),
                text=['Cricondenbar'],
                textposition='top center',
                hovertemplate=f'Cricondenbar<br>T: {cricondenbar[0]:.1f} °C<br>P: {cricondenbar[1]:.2f} bara<extra></extra>'
            ))
            
            # Cricondentherm point
            fig_plotly.add_trace(go.Scatter(
                x=[cricondentherm[0]], y=[cricondentherm[1]],
                mode='markers+text',
                name=f'Cricondentherm ({cricondentherm[0]:.1f} °C)',
                marker=dict(color='#F4A261', size=14, symbol='diamond'),
                text=['Cricondentherm'],
                textposition='bottom center',
                hovertemplate=f'Cricondentherm<br>T: {cricondentherm[0]:.1f} °C<br>P: {cricondentherm[1]:.2f} bara<extra></extra>'
            ))
            
            fig_plotly.update_layout(
                title=dict(text='Phase Envelope (PT Diagram)', font=dict(size=20)),
                xaxis_title='Temperature (°C)',
                yaxis_title='Pressure (bara)',
                hovermode='closest',
                legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01),
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig_plotly, width='stretch')
            st.divider()

            # --- Add cricondenbar to the phase envelope table, exclude cricondentherm, no 'Type' column, round to 2 decimals ---
            all_points = np.vstack([
                np.column_stack((temp_nn, press_nn)),
                [cricondenbar[0], cricondenbar[1]]
            ])
            n_points = len(all_points)
            used = np.zeros(n_points, dtype=bool)
            path = []
            current = 0
            path.append(current)
            used[current] = True
            for _ in range(1, n_points):
                dists = np.linalg.norm(all_points - all_points[current], axis=1)
                dists[used] = np.inf
                next_idx = np.argmin(dists)
                path.append(next_idx)
                used[next_idx] = True
                current = next_idx

            all_points_nn = all_points[path]
            phase_env_df = pd.DataFrame({
                'Temperature [°C]': np.round(all_points_nn[:,0], 2),
                'Pressure [bara]': np.round(all_points_nn[:,1], 2)
            })

            st.write('Phase envelope points:')
            st.data_editor(phase_env_df, key='phase_env_table')

            st.write('cricondentherm ', 
                    round(cricondentherm[1], 2), ' bara, ', 
                    round(cricondentherm[0], 2), ' °C')
            st.write('cricondenbar ', 
                    round(cricondenbar[1], 2), ' bara, ', 
                    round(cricondenbar[0], 2), ' °C')
        else:
            # Create interactive Plotly figure for dew/bubble points
            fig_plotly = go.Figure()
            
            # Dew point line
            fig_plotly.add_trace(go.Scatter(
                x=dewts, y=dewps,
                mode='lines+markers',
                name='Dew Point',
                line=dict(color='#2E86AB', width=2),
                marker=dict(size=6),
                hovertemplate='Dew Point<br>T: %{x:.1f} °C<br>P: %{y:.2f} bara<extra></extra>'
            ))
            
            # Bubble point line
            fig_plotly.add_trace(go.Scatter(
                x=bubts, y=bubps,
                mode='lines+markers',
                name='Bubble Point',
                line=dict(color='#E63946', width=2),
                marker=dict(size=6),
                hovertemplate='Bubble Point<br>T: %{x:.1f} °C<br>P: %{y:.2f} bara<extra></extra>'
            ))
            
            # Get critical points for annotation
            cricobar = thermoOps.getOperation().get("cricondenbar")
            cricotherm = thermoOps.getOperation().get("cricondentherm")
            criticalT = thermoOps.getOperation().get("criticalPoint1")[0]
            criticalP = thermoOps.getOperation().get("criticalPoint1")[1]
            
            # Add critical point
            fig_plotly.add_trace(go.Scatter(
                x=[criticalT - 273.15], y=[criticalP],
                mode='markers+text',
                name=f'Critical Point',
                marker=dict(color='#2D6A4F', size=12, symbol='star'),
                text=['Critical'],
                textposition='top right',
                hovertemplate=f'Critical Point<br>T: {criticalT - 273.15:.1f} °C<br>P: {criticalP:.2f} bara<extra></extra>'
            ))
            
            # Add cricondenbar
            fig_plotly.add_trace(go.Scatter(
                x=[cricobar[0] - 273.15], y=[cricobar[1]],
                mode='markers+text',
                name=f'Cricondenbar ({cricobar[1]:.1f} bara)',
                marker=dict(color='#7B2CBF', size=12, symbol='diamond'),
                text=['Cricondenbar'],
                textposition='top center',
                hovertemplate=f'Cricondenbar<br>T: {cricobar[0] - 273.15:.1f} °C<br>P: {cricobar[1]:.2f} bara<extra></extra>'
            ))
            
            # Add cricondentherm
            fig_plotly.add_trace(go.Scatter(
                x=[cricotherm[0] - 273.15], y=[cricotherm[1]],
                mode='markers+text',
                name=f'Cricondentherm ({cricotherm[0] - 273.15:.1f} °C)',
                marker=dict(color='#F4A261', size=12, symbol='diamond'),
                text=['Cricondentherm'],
                textposition='bottom right',
                hovertemplate=f'Cricondentherm<br>T: {cricotherm[0] - 273.15:.1f} °C<br>P: {cricotherm[1]:.2f} bara<extra></extra>'
            ))
            
            fig_plotly.update_layout(
                title=dict(text='Phase Envelope (PT Diagram)', font=dict(size=20)),
                xaxis_title='Temperature (°C)',
                yaxis_title='Pressure (bara)',
                hovermode='closest',
                legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01),
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig_plotly, width='stretch')
            st.divider()
            
            # Display cricondenbar and cricondentherm
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
        thermoOps.TPflash()
        
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
    key='phaseenv_uploaded_file',
    help='Fluids can be saved by hovering over the fluid window and clicking the "Download as CSV" button in the upper-right corner.'
)
