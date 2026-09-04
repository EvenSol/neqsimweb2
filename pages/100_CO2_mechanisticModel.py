"""
CO2 Impurity Mechanistic Model — Streamlit Page
================================================
Interactive front-end for the CSTR-style CO2-impurity kinetics / wall-corrosion
engine implemented in :mod:`co2_mechanistic_model` (``AutoclaveExperiment``).

The user defines the reactor/coupon geometry once, then builds a feed schedule
table (one row per phase: start time, pressure, temperature, mass flow, feed
concentrations and a label). Rows can be added/removed with the built-in
data-editor "+"/"-" controls. Running the simulation calls into NeqSim
(via the SRK EoS, when available) for the thermodynamics and displays the
resulting species/reaction/corrosion/wash-water results as tables and charts,
mirroring the analysis performed in ``pages/CDC171.ipynb``.
"""

import re

import pandas as pd
import streamlit as st

from theme import apply_theme
from co2_mechanistic_model import AutoclaveExperiment

st.set_page_config(
    page_title="CO2 Mechanistic Model",
    page_icon='images/neqsimlogocircleflat.png',
    layout="wide",
)
apply_theme()

FEED_SPECIES = ('H2O', 'H2S', 'SO2', 'NO2', 'O2', 'NO')
MATERIALS = ('carbon_steel', 'magnetite', 'stainless_steel', 'inert')


def _numeric_column_config(df: pd.DataFrame, fmt: str = "%.4g") -> dict:
    """Compact display format for every float column (avoids raw floats like
    0.00000000000000000000006 by switching to scientific notation when needed)."""
    return {
        col: st.column_config.NumberColumn(col, format=fmt)
        for col, dtype in df.dtypes.items() if pd.api.types.is_float_dtype(dtype)
    }

st.title("🧪 CO2 Impurity Mechanistic Model")
st.markdown("""
Mechanistic CSTR simulator for trace-impurity reaction chemistry and wall corrosion
in CO2 streams, built on NeqSim thermodynamics (SRK EoS). Define the reactor/coupon
geometry, build a feed schedule (pressure, temperature, mass flow and impurity
concentrations vs. time), then run the simulation to view species trajectories,
reaction-rate rankings, corrosion rates and wash-water chemistry.
""")

with st.expander("ℹ️ About this model"):
    st.markdown("""
The reactor is modeled as a CSTR fed with CO2 plus trace impurities
(H2S, SO2, NO2, NO, O2, H2O). Each row of the **feed schedule** below defines a new
phase that starts at the given time and runs until the next row's start time (the
last phase runs until **Termination hour**). Whenever pressure/temperature change
between phases, the SRK flash is re-evaluated, updating molar density, fugacity
coefficients and residence time.

The kinetic parameters are illustrative/uncalibrated and are provided for
**screening purposes only**; they are not qualified design data.
""")

st.divider()

# =============================================================================
# Reactor / coupon geometry
# =============================================================================
st.subheader("1️⃣ Reactor & coupon geometry")

st.caption("Flow rate, pressure and temperature are set per phase in the feed schedule table below, "
           "not here.")
geom_col1, geom_col2, geom_col3 = st.columns(3)
with geom_col1:
    volume_ml = st.number_input("Volume [mL]", min_value=1.0, value=330.0, step=10.0)
with geom_col2:
    diameter_cm = st.number_input("Inner diameter [cm]", min_value=0.1, value=6.5, step=0.1)
    material = st.selectbox("Material", MATERIALS, index=MATERIALS.index('carbon_steel'))
with geom_col3:
    coupon_diameter_cm = st.number_input("Coupon diameter [cm]", min_value=0.0, value=3.0, step=0.5)
    coupon_thickness_mm = st.number_input("Coupon thickness [mm]", min_value=0.0, value=5.0, step=0.5)

# =============================================================================
# Feed schedule editor
# =============================================================================
st.subheader("2️⃣ Feed schedule")
st.caption("One row per phase. Feed concentrations are ppm-mol; leave at 0 for pure CO2.")

FEED_COLUMNS = (
    ['Start (h)', 'Pressure (bar)', 'Temperature (C)', 'Mass flow (g/h)']
    + [f'{sp} (ppm)' for sp in FEED_SPECIES] + ['Label']
)

if 'co2mm_feed_df' not in st.session_state:
    st.session_state.co2mm_feed_df = pd.DataFrame({
        col: pd.Series(dtype=('object' if col == 'Label' else 'float64'))
        for col in FEED_COLUMNS
    })

with st.expander("📋 Paste from Excel (replaces the whole table)"):
    st.caption("Copy a cell range from Excel with columns in this exact order: "
               + ", ".join(FEED_COLUMNS) + ". Paste below and click Load — this replaces "
               "whatever is currently in the table. Header rows are detected and skipped "
               "automatically, so you can copy your Excel headers along with the data.")
    pasted_text = st.text_area("Pasted range", key='co2mm_paste_area', height=120,
                                placeholder="0\t30\t2\t47\t0\t0\t0\t0\t0\t0\tInitial CO2")
    if st.button("Load pasted data", key='co2mm_load_paste'):
        parsed_rows = []
        skipped_header_lines = 0
        for line in pasted_text.strip('\n').splitlines():
            line = line.strip()
            if not line:
                continue
            if '\t' in line:
                cells = line.split('\t')
            elif ',' in line:
                cells = line.split(',')
            else:
                cells = re.split(r'\s{2,}|\s+', line)
            # A real data row must have numeric Start/Pressure/Temperature/Mass flow; anything
            # else (e.g. a copied Excel header row) is skipped instead of turning into a 0-row.
            numeric_ok = True
            numeric_values = {}
            for i, col in enumerate(FEED_COLUMNS[:4]):
                raw = cells[i].strip() if i < len(cells) else ''
                try:
                    numeric_values[col] = float(raw.replace(',', '.'))
                except ValueError:
                    numeric_ok = False
                    break
            if not numeric_ok:
                skipped_header_lines += 1
                continue
            row = dict(numeric_values)
            for i, col in enumerate(FEED_COLUMNS[4:], start=4):
                raw = cells[i].strip() if i < len(cells) else ''
                if col == 'Label':
                    row[col] = raw
                else:
                    try:
                        row[col] = float(raw.replace(',', '.')) if raw else 0.0
                    except ValueError:
                        row[col] = 0.0
            parsed_rows.append(row)
        if parsed_rows:
            st.session_state.co2mm_feed_df = pd.DataFrame(parsed_rows)
            msg = f"Loaded {len(parsed_rows)} row(s), replacing the table."
            if skipped_header_lines:
                msg += f" Skipped {skipped_header_lines} non-numeric header line(s)."
            st.success(msg)
            st.rerun()
        else:
            st.warning("No numeric data rows detected in the pasted text.")

# Static defaults (not tied to the sidebar reactor settings) so that changing the sidebar
# never reshapes the column config and never resets/clears the table's own data.
column_config = {
    'Start (h)': st.column_config.NumberColumn('Start (h)', min_value=0.0, format="%.2f", default=0.0),
    'Pressure (bar)': st.column_config.NumberColumn('Pressure (bar)', min_value=0.0, format="%.2f",
                                                     default=0.0),
    'Temperature (C)': st.column_config.NumberColumn('Temperature (°C)', format="%.2f", default=0.0),
    'Mass flow (g/h)': st.column_config.NumberColumn('Mass flow (g/h)', min_value=0.0, format="%.2f",
                                                      default=0.0),
    'Label': st.column_config.TextColumn('Label', default=''),
}
for sp in FEED_SPECIES:
    column_config[f'{sp} (ppm)'] = st.column_config.NumberColumn(f'{sp} (ppm)', min_value=0.0,
                                                                  format="%.3f", default=0.0)

feed_df = st.data_editor(
    st.session_state.co2mm_feed_df,
    num_rows='dynamic',
    hide_index=True,
    column_config=column_config,
    key='co2mm_feed_editor',
)

# Safety net for any cell still left blank (e.g. a pasted range shorter than the table).
_row_defaults = {col: ('' if col == 'Label' else 0.0) for col in FEED_COLUMNS}
feed_df = feed_df.fillna(_row_defaults)

if st.button("➕ Add row", key='co2mm_add_row'):
    new_row = {col: ('' if col == 'Label' else 0.0) for col in FEED_COLUMNS}
    st.session_state.co2mm_feed_df = pd.concat(
        [feed_df, pd.DataFrame([new_row])], ignore_index=True)
    st.rerun()

run_clicked = st.button("▶️ Run simulation", type="primary")

# =============================================================================
# Run
# =============================================================================
def _build_phases(df: pd.DataFrame):
    # A row only counts as a real phase once it has a positive pressure; this lets the
    # pre-filled blank template rows (all zeros) sit unused until the user fills them in.
    rows = df.dropna(subset=['Start (h)'])
    rows = rows[rows['Pressure (bar)'] > 0].sort_values('Start (h)')
    phases = []
    for _, row in rows.iterrows():
        feed = {sp: float(row.get(f'{sp} (ppm)', 0.0) or 0.0) for sp in FEED_SPECIES}
        phases.append((float(row['Start (h)']), {
            'pressure_bar': float(row['Pressure (bar)']),
            'temp_C': float(row['Temperature (C)']),
            'mass_flow_g_h': float(row['Mass flow (g/h)']),
            'feed': feed,
            'label': str(row['Label']) if row.get('Label') else None,
        }))
    return phases


if run_clicked:
    phases_feed = _build_phases(feed_df)
    if not phases_feed:
        st.warning("Add at least one feed-schedule row before running.")
    else:
        first = phases_feed[0][1]
        # No explicit termination-hour input: run 50 h past the last phase's start time.
        termination_hour = phases_feed[-1][0] + 50.0
        try:
            with st.spinner("Running mechanistic simulation..."):
                autoclave = AutoclaveExperiment(
                    volume_ml=volume_ml,
                    mass_flow_g_h=first['mass_flow_g_h'],
                    diameter_cm=diameter_cm,
                    temp_C=first['temp_C'],
                    pressure_bar=first['pressure_bar'],
                    material=material,
                    coupon_diameter_cm=coupon_diameter_cm,
                    coupon_thickness_mm=coupon_thickness_mm,
                )
                autoclave.set_phases(phases_feed, termination_hour=termination_hour).run()
            st.session_state.co2mm_autoclave = autoclave
            st.success(f"Simulation complete (ran to {termination_hour:g} h).")
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")

autoclave = st.session_state.get('co2mm_autoclave')

if autoclave is None:
    st.info("Configure the reactor/feed schedule above and click **Run simulation** to see results.")
else:
    st.divider()
    st.subheader("3️⃣ Results")

    tab_summary, tab_species, tab_reactions, tab_surface, tab_wash_water, tab_wash_ic = st.tabs([
        "Autoclave summary", "Species vs time", "Reaction rates", "Surface & corrosion",
        "Wash-water pH", "Autoclave wash (IC)",
    ])

    with tab_summary:
        geom = autoclave.exp.model.get_reactor_geometry()
        props = autoclave.exp.model.get_fluid_properties()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Volume [mL]", f"{geom['volume_ml']:.1f}")
        m1.metric("Diameter [cm]", f"{geom['diameter_cm']:.2f}")
        m2.metric("Length [cm]", f"{geom['length_cm']:.2f}")
        m2.metric("Cross-section [cm²]", f"{geom['cross_sectional_area_cm2']:.3f}")
        m3.metric("Density [kg/m³]", f"{props['mass_density_kg_m3']:.2f}")
        m3.metric("Phase", props['phase'].capitalize())
        m4.metric("Residence time [h]", f"{geom['residence_time_hours']:.4f}")
        m4.metric("Mass flow [g/h]", f"{geom['mass_flow_g_h']:.1f}")

        st.markdown("**Full reactor report**")
        st.text(autoclave.get_reactor_report())

    with tab_species:
        values_df = autoclave.get_values()
        st.caption(f"{len(values_df)} time points x {values_df.shape[1] - 1} species")
        fig, _ = autoclave.build_plot('reactant species')
        st.pyplot(fig)
        fig, _ = autoclave.build_plot('reaction products')
        st.pyplot(fig)
        st.dataframe(values_df.iloc[::max(len(values_df) // 200, 1)], hide_index=True,
                      column_config=_numeric_column_config(values_df))

    with tab_reactions:
        st.markdown("**Overall reaction-activity ranking**")
        reaction_table = autoclave.get_reaction_table()
        st.dataframe(reaction_table, hide_index=True, column_config=_numeric_column_config(reaction_table))
        st.markdown("**Top reaction pathways per phase**")
        step_reaction_table = autoclave.get_step_reaction_table()
        st.dataframe(step_reaction_table, hide_index=True,
                      column_config=_numeric_column_config(step_reaction_table))
        st.markdown("**Mass balance closure**")
        mass_balance_table = autoclave.get_mass_balance_table()
        st.dataframe(mass_balance_table, hide_index=True,
                      column_config=_numeric_column_config(mass_balance_table))

    with tab_surface:
        surface_df = autoclave.get_surface_data()
        st.metric("End-of-run corrosion rate [mm/yr]", f"{surface_df['corrosion_rate_mm_yr'].iloc[-1]:.3f}")
        fig, _ = autoclave.build_plot('surface reaction products')
        st.pyplot(fig)
        fig, _ = autoclave.build_plot('corrosion rate')
        st.pyplot(fig)
        st.dataframe(surface_df.iloc[::max(len(surface_df) // 200, 1)], hide_index=True,
                      column_config=_numeric_column_config(surface_df))

    with tab_wash_water:
        water_mass_g = st.number_input("Wash water mass [g]", min_value=1.0, value=500.0, step=10.0,
                                        key='co2mm_water_mass')
        wash_df = autoclave.get_wash_water_pH_table(water_mass_g=water_mass_g)
        fig, _ = autoclave.build_plot('wash water pH', water_mass_g=water_mass_g)
        st.pyplot(fig)
        st.dataframe(wash_df.iloc[::max(len(wash_df) // 200, 1)], hide_index=True,
                      column_config=_numeric_column_config(wash_df))

    with tab_wash_ic:
        wash_mass_g = st.number_input("Autoclave wash mass [g]", min_value=1.0, value=30.0, step=5.0,
                                       key='co2mm_wash_mass')
        wash_ic_df = autoclave.get_autoclave_wash_table(wash_mass_g=wash_mass_g)
        st.dataframe(wash_ic_df.iloc[::max(len(wash_ic_df) // 200, 1)], hide_index=True,
                      column_config=_numeric_column_config(wash_ic_df))
        st.markdown("**End-of-run IC-style summary**")
        wash_summary = autoclave.get_autoclave_wash_summary(wash_mass_g=wash_mass_g)
        st.dataframe(wash_summary, column_config=_numeric_column_config(wash_summary))
