import math
from typing import Dict, Tuple

import pandas as pd
import streamlit as st
import neqsim
from neqsim.thermo.thermoTools import fluidcreator, fluid_df, TPflash, dataFrame

from fluids import default_fluid
from services.ai import (
    create_issue_messages,
    explain_validation_issue,
    plan_scenario,
    summarize_flash_results,
)
from services.retrieval import query_equilibrium_knowledge
from components.chat import render_what_if_chat
from components.assistant import render_ai_helper


st.title('TP flash')
render_ai_helper(
    "tp_flash",
    {
        "default_components": len(default_fluid['ComponentName']),
        "default_schedule_rows": 2,
    },
)

"""
The NeqSim flash model will select the best thermodynamic model based on the fluid composition. For fluids containing polar components it will use the CPA-EoS.
For non-polar fluids it will use the SRK/PR-EoS. The flash will calculate the phase equilibrium for given composition at the specified temperatures and pressures.

You can select components from a predifined component list. Alterative component names ([see available components](https://github.com/equinor/neqsim/blob/master/src/main/resources/data/COMP.csv)) can be used by manually editing the table.
"""

st.divider()

if 'activefluid_df' not in st.session_state or st.session_state.get('activefluid_name') != 'default_fluid':
    st.session_state.activefluid_df = pd.DataFrame(default_fluid)
    st.session_state.activefluid_name = 'default_fluid'

if 'tp_flash_data' not in st.session_state:
    st.session_state['tp_flash_data'] = pd.DataFrame({
        'Temperature (C)': [20.0, 25.0],
        'Pressure (bara)': [1.0, 10.0]
    })

st.subheader('Natural-language scenario setup')
scenario_prompt = st.text_area(
    'Describe the fluid and conditions (e.g. "simulate 90% methane and 10% ethane at 40 °C and 30 bar")',
    key='scenario_prompt',
)
if st.button('Apply scenario plan') and scenario_prompt.strip():
    scenario = plan_scenario(scenario_prompt)
    st.session_state.activefluid_df = scenario['fluid']
    st.session_state.tp_flash_data = scenario['schedule']
    st.success('Scenario applied to the editors below.')

st.divider()
st.text('Set fluid composition:')

uploaded_file = st.sidebar.file_uploader(
    'Import Fluid',
    key='uploaded_file',
    help='Fluids can be saved by hovering over the fluid window and clicking the "Download as CSV" button in the upper-right corner.',
)
if uploaded_file is not None:
    try:
        imported_df = pd.read_csv(uploaded_file)
        numeric_columns = ['MolarComposition[-]', 'MolarMass[kg/mol]', 'RelativeDensity[-]']
        imported_df[numeric_columns] = imported_df[numeric_columns].astype(float)
        st.session_state.activefluid_df = imported_df
    except Exception as exc:  # pragma: no cover - user facing guard
        st.warning(f'Could not import the provided file ({exc}). Falling back to defaults.')
        st.session_state.activefluid_df = pd.DataFrame(default_fluid)


hidecomponents = st.checkbox('Show active components only')
if hidecomponents:
    editor_df = st.session_state.activefluid_df[st.session_state.activefluid_df['MolarComposition[-]'] > 0].copy()
    editor_df = editor_df.reset_index(drop=True)
else:
    editor_df = st.session_state.activefluid_df

edited_df = st.data_editor(
    editor_df,
    column_config={

        'ComponentName': 'Component Name',
        'MolarComposition[-]': st.column_config.NumberColumn(
            'Molar Composition [-]', min_value=0.0, max_value=10000.0, format='%f'
        ),
        'MolarMass[kg/mol]': st.column_config.NumberColumn(
            'Molar Mass [kg/mol]', min_value=0.0, max_value=10000.0, format='%f kg/mol'
        ),
        'RelativeDensity[-]': st.column_config.NumberColumn(
            'Density [gr/cm3]', min_value=1e-10, max_value=10.0, format='%f gr/cm3'
        ),

    },
    num_rows='dynamic',
)
if hidecomponents:
    base_df = st.session_state.activefluid_df.copy()
    active_mask = base_df['MolarComposition[-]'] > 0
    base_df.loc[active_mask, :] = edited_df.values
    st.session_state.activefluid_df = base_df
else:
    st.session_state.activefluid_df = edited_df

isplusfluid = st.checkbox('Plus Fluid')

st.text('Fluid composition will be normalized before simulation')
st.divider()
st.text('Input Pressures and Temperatures')

st.session_state.tp_flash_data = st.data_editor(
    st.session_state.tp_flash_data.dropna().reset_index(drop=True),
    num_rows='dynamic',
    column_config={
        'Temperature (C)': st.column_config.NumberColumn(
            label='Temperature (C)',
            min_value=-273.15,
            max_value=1000.0,
            format='%f',
            help='Enter the temperature in degrees Celsius.'
        ),
        'Pressure (bara)': st.column_config.NumberColumn(
            label='Pressure (bara)',
            min_value=0.0,
            max_value=1000.0,
            format='%f',
            help='Enter the pressure in bar absolute.'
        ),
    },
)


def _normalize_fluid(df: pd.DataFrame) -> pd.DataFrame:
    total = df['MolarComposition[-]'].sum()
    if total > 0:
        normalized = df.copy()
        normalized['MolarComposition[-]'] = normalized['MolarComposition[-]'] / total
        return normalized
    return df


def validate_inputs(fluid_frame: pd.DataFrame, schedule_frame: pd.DataFrame) -> Tuple[Dict[str, str], ...]:
    issues = create_issue_messages(fluid_frame, schedule_frame)
    return tuple(issues)


def run_flash_simulation(
    fluid_frame: pd.DataFrame,
    schedule_frame: pd.DataFrame,
    is_plus: bool,
) -> Tuple[pd.DataFrame, float, float]:
    if schedule_frame.empty:
        return pd.DataFrame(), math.nan, math.nan

    neqsim_fluid = fluid_df(fluid_frame, lastIsPlusFraction=is_plus, add_all_components=False).autoSelectModel()
    results_list = []
    last_temp = math.nan
    last_pres = math.nan

    for _, row in schedule_frame.dropna().iterrows():
        temp = float(row['Temperature (C)'])
        pressure = float(row['Pressure (bara)'])
        neqsim_fluid.setPressure(pressure, 'bara')
        neqsim_fluid.setTemperature(temp, 'C')
        TPflash(neqsim_fluid)
        results_list.append(dataFrame(neqsim_fluid))
        last_temp = temp
        last_pres = pressure

    combined_results = pd.concat(results_list, ignore_index=True) if results_list else pd.DataFrame()
    return combined_results, last_temp, last_pres


issues = validate_inputs(st.session_state.activefluid_df, st.session_state.tp_flash_data)
if issues:
    for issue in issues:
        if issue['level'] == 'error':
            st.error(issue['message'])
        else:
            st.warning(issue['message'])
    explanation = explain_validation_issue(issues, st.session_state.activefluid_df, st.session_state.tp_flash_data)
    st.info(explanation)
    if any(issue['code'] == 'composition_normalize' for issue in issues):
        if st.button('Normalize compositions automatically'):
            st.session_state.activefluid_df = _normalize_fluid(st.session_state.activefluid_df)
            st.experimental_rerun()

run_results = None
last_temp = math.nan
last_pres = math.nan

if st.button('Run TP Flash Calculations'):
    if any(issue['level'] == 'error' for issue in issues):
        st.error('Resolve input errors before running the simulation.')
    else:
        normalized_fluid = _normalize_fluid(st.session_state.activefluid_df)
        run_results, last_temp, last_pres = run_flash_simulation(
            normalized_fluid,
            st.session_state.tp_flash_data,
            isplusfluid,
        )
        if run_results is not None and not run_results.empty:
            st.success('Flash calculations finished successfully!')
            st.subheader('Results:')
            results_df = st.data_editor(run_results)
            st.session_state['tp_flash_last_results'] = run_results
            component_names = normalized_fluid['ComponentName'][normalized_fluid['MolarComposition[-]'] > 0]
            knowledge_results = query_equilibrium_knowledge(
                component_names.tolist(),
                last_temp if not math.isnan(last_temp) else 0.0,
                last_pres if not math.isnan(last_pres) else 0.0,
            )
            st.divider()
            st.subheader('Knowledge base insights')
            for entry in knowledge_results:
                st.markdown(f"**{entry['title']}** (confidence {entry['confidence']:.0%})")
                st.markdown(entry['summary'])
                st.caption(f"Source: {entry['source']}")

            st.divider()
            if st.checkbox('Summarize results with AI'):
                context = {
                    'temperature': f'{last_temp:.2f} °C' if not math.isnan(last_temp) else 'n/a',
                    'pressure': f'{last_pres:.2f} bara' if not math.isnan(last_pres) else 'n/a',
                }
                summary = summarize_flash_results(run_results, context)
                st.markdown(summary)
                st.download_button(
                    'Download summary as Markdown',
                    data=summary,
                    file_name='tp_flash_summary.md',
                    mime='text/markdown',
                )
        else:
            st.error('No results were produced. Check your inputs and try again.')

render_what_if_chat(
    _normalize_fluid(st.session_state.activefluid_df),
    st.session_state.tp_flash_data,
    isplusfluid,
    run_flash_simulation,
)
