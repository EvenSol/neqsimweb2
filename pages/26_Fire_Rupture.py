# -*- coding: utf-8 -*-
"""
Fire Rupture (Time-to-Rupture) Page
===================================
Streamlit front-end for the Equinor fire-rupture / strain-rate model.

The user supplies the inputs (blowdown pressure profile, fire definitions and the
pipes in the blowdown segment) in spreadsheet-style editors; the heavy lifting is
done by the pure-Python engine in :mod:`fire_rupture`, which reproduces the
original Excel workbook ("Fire rupture calc - Strain rate model").

Calculates, for up to several pipes simultaneously exposed to up to three fires:

* the heat-up of the pipe wall,
* the time to rupture (creep strain exceeding the strain limit),
* the rupture pressure, and
* the resulting gas / liquid release rate from the rupture location.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from theme import apply_theme
import fire_rupture as fr

st.set_page_config(
    page_title="Fire Rupture",
    page_icon='images/neqsimlogocircleflat.png',
    layout="wide",
)
apply_theme()


# =============================================================================
# Builders / helpers (defined early so the sidebar can use them)
# =============================================================================
def _build_fires(df: pd.DataFrame):
    fires = []
    for _, r in df.dropna(subset=["Name"]).iterrows():
        fires.append(fr.Fire(
            name=str(r["Name"]),
            fire_temp_C=float(r["FireTemp[°C]"]),
            gas_temp_C=float(r["GasTemp[°C]"]),
            h_conv=float(r["h_conv[W/m²K]"]),
            fire_emissivity=float(r["FireEmiss[-]"]),
            metal_emissivity=float(r["MetalEmiss[-]"]),
            metal_absorptivity=float(r["MetalAbsorp[-]"]),
        ))
    return fires


def _build_pipe(row) -> fr.Pipe:
    nom = row.get("NominalDiameter[inch]")
    return fr.Pipe(
        name=str(row["Name"]),
        od_mm=float(row["OD[mm]"]),
        wall_mm=float(row["Wall[mm]"]),
        material=str(row["Material"]),
        corrosion_allowance_mm=float(row.get("Corrosion[mm]", 0.0) or 0.0),
        wall_tolerance_frac=float(row.get("WallTolerance[frac]", 0.0) or 0.0),
        weight_stress_MPa=float(row.get("WeightStress[MPa]", 0.0) or 0.0),
        fluid_density=float(row.get("FluidDensity[kg/m³]", 23.75) or 23.75),
        fluid_heat_capacity=float(row.get("FluidCp[J/kgK]", 2283.0) or 2283.0),
        gas_mw=float(row.get("GasMW[g/mol]", 18.2) or 18.2),
        nominal_inch=(float(nom) if nom is not None and not pd.isna(nom) else None),
    )


def _fmt(value, fmt="{:.2f}"):
    return fmt.format(value) if value is not None else "N/A"


st.title('🔥 Fire Rupture — Time to Rupture')
st.markdown("""
Calculates the **heat-up of pipes (or vessel walls) exposed to fire** and predicts
the **time to rupture** using a creep / strain-rate model. Several pipes can be
evaluated simultaneously, each exposed to up to three different fires, all sharing
the **same blowdown pressure profile** (i.e. they belong to the same blowdown
segment).

For each pipe the model computes the transient wall temperature, the von&nbsp;Mises
stress (from the blowdown pressure and an optional external *weight* stress), the
accumulated creep strain and — once the strain exceeds the temperature-dependent
strain limit — the **rupture time, rupture pressure and release rate**.
""")

with st.expander("ℹ️ About this calculation / how to use"):
    st.markdown("""
**About this spreadsheet (from the original workbook):**

This calculates heat-up of pipes being exposed to fire (it can also be used for
vessel-wall heat-up). All specified pipes are exposed to the **same pressure
profile**, i.e. they are assumed to be within the same blowdown segment. When
studying a new blowdown segment, a new pressure profile must be supplied.

The supported materials are *22Cr Duplex, 25Cr Duplex, CS&nbsp;235LT, CS&nbsp;360LT,
6Mo and SS316*. The material conductivity, heat capacity, rupture stress (UTS) and
strain limit are modelled as a function of wall temperature and change during
heat-up. The actual stress is calculated from the pressure; the actual strain is
accumulated over time and compared with the strain limit. **Rupture** is defined
when the strain exceeds the strain limit.

The **release rate** from the rupture location is calculated for three locations:
1. A *long* distance between the reservoir and the rupture, with flow from **2 sides**
   *(recommended as the release rate)*;
2. The same but from **1 side** (e.g. rupture near a shutdown valve);
3. A *short* pipe, i.e. rupture near the reservoir.

A **liquid** release rate is calculated instead if the fluid density is larger than
500&nbsp;kg/m³ (the gas/liquid switch). Liquid releases are normally one-sided
(pushed out by the gas reservoir).

**How to use:** edit the pressure profile, fire definitions and pipe list below,
then press **Run calculation**.
""")

st.divider()

# =============================================================================
# Sidebar — calculation settings
# =============================================================================
with st.sidebar:
    st.header("⚙️ Calculation settings")
    time_step_s = st.number_input(
        "Time step [s]", min_value=0.5, max_value=60.0, value=5.0, step=0.5,
        help="Integration time step for the temperature and strain calculation.")
    initial_temp_C = st.number_input(
        "Initial segment temperature [°C]", min_value=-50.0, max_value=300.0,
        value=20.0, step=1.0)

    st.divider()
    st.header("🔥 Fires")
    st.caption("Radiative source + convective gas load applied to the pipe surface.")
    if 'fr_fires_df' not in st.session_state:
        st.session_state.fr_fires_df = pd.DataFrame([
            {"Name": f.name, "FireTemp[°C]": f.fire_temp_C, "GasTemp[°C]": f.gas_temp_C,
             "h_conv[W/m²K]": f.h_conv, "FireEmiss[-]": f.fire_emissivity,
             "MetalEmiss[-]": f.metal_emissivity, "MetalAbsorp[-]": f.metal_absorptivity}
            for f in fr.default_fires()
        ])
    fires_df = st.data_editor(
        st.session_state.fr_fires_df,
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "Name": st.column_config.TextColumn("Name"),
            "FireTemp[°C]": st.column_config.NumberColumn("Fire T [°C]", format="%.0f"),
            "GasTemp[°C]": st.column_config.NumberColumn("Gas T [°C]", format="%.0f"),
            "h_conv[W/m²K]": st.column_config.NumberColumn("h [W/m²K]", format="%.0f"),
            "FireEmiss[-]": st.column_config.NumberColumn("Fire ε", format="%.2f"),
            "MetalEmiss[-]": st.column_config.NumberColumn("Metal ε", format="%.2f"),
            "MetalAbsorp[-]": st.column_config.NumberColumn("Metal α", format="%.2f"),
        },
        key="fr_fires_editor",
    )

    # Initial heat-flux QA outputs (cf. workbook cells AB14/AB15), evaluated at
    # the cold surface temperature (= initial segment temperature).
    try:
        flux_rows = []
        for f in _build_fires(fires_df):
            flux_rows.append({
                "Fire": f.name,
                "Incident [kW/m²]": f.incident_flux(initial_temp_C),
                "Absorbed [kW/m²]": f.absorbed_flux(initial_temp_C),
            })
        if flux_rows:
            st.caption("Initial heat flux (at cold wall):")
            st.dataframe(
                pd.DataFrame(flux_rows),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Incident [kW/m²]": st.column_config.NumberColumn(format="%.1f"),
                    "Absorbed [kW/m²]": st.column_config.NumberColumn(format="%.1f"),
                },
            )
    except Exception:
        pass

# =============================================================================
# Pressure profile (blowdown)
# =============================================================================
st.subheader("1️⃣ Blowdown pressure profile")
st.caption("Absolute pressure (bara) versus time (minutes). All pipes in the segment "
           "share this profile. Paste your own profile to study a new segment.")

if 'fr_profile_df' not in st.session_state:
    st.session_state.fr_profile_df = pd.DataFrame(
        fr.DEFAULT_PRESSURE_PROFILE, columns=["Time[min]", "Pressure[bara]"])

col_prof, col_plot = st.columns([1, 2])
with col_prof:
    profile_df = st.data_editor(
        st.session_state.fr_profile_df,
        num_rows="dynamic",
        hide_index=True,
        height=360,
        column_config={
            "Time[min]": st.column_config.NumberColumn("Time [min]", format="%.4f"),
            "Pressure[bara]": st.column_config.NumberColumn("Pressure [bara]", format="%.4f"),
        },
        key="fr_profile_editor",
    )
with col_plot:
    prof_clean = profile_df.dropna().sort_values("Time[min]")
    if len(prof_clean) >= 2:
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(
            x=prof_clean["Time[min]"], y=prof_clean["Pressure[bara]"],
            mode="lines+markers", name="Pressure", line=dict(color="#d62728")))
        fig_p.update_layout(
            title="Blowdown pressure profile",
            xaxis_title="Time [min]", yaxis_title="Pressure [bara]",
            height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_p, use_container_width=True)

st.divider()

# =============================================================================
# Pipe segment definitions
# =============================================================================
st.subheader("2️⃣ Pipes in the blowdown segment")
st.caption("One row per pipe (or vessel wall). Fluid density > 500 kg/m³ is treated "
           "as a liquid release; otherwise a gas release is assumed.")

if 'fr_pipes_df' not in st.session_state:
    rows = []
    for p in fr.DEFAULT_PIPES:
        rows.append({
            "Name": p["Name"],
            "OD[mm]": p["OD[mm]"],
            "Wall[mm]": p["Wall[mm]"],
            "Material": "22Cr duplex",
            "NominalDiameter[inch]": p["NominalDiameter[inch]"],
            "Corrosion[mm]": 0.0,
            "WallTolerance[frac]": 0.125,
            "WeightStress[MPa]": 0.0,
            "FluidDensity[kg/m³]": 23.7487037155012,
            "FluidCp[J/kgK]": 2283.35469905934,
            "GasMW[g/mol]": 18.2,
        })
    st.session_state.fr_pipes_df = pd.DataFrame(rows)

pipes_df = st.data_editor(
    st.session_state.fr_pipes_df,
    num_rows="dynamic",
    hide_index=True,
    column_config={
        "Name": st.column_config.TextColumn("Name"),
        "OD[mm]": st.column_config.NumberColumn("OD [mm]", min_value=1.0, format="%.1f"),
        "Wall[mm]": st.column_config.NumberColumn("Wall [mm]", min_value=0.1, format="%.2f"),
        "Material": st.column_config.SelectboxColumn(
            "Material", options=list(fr.MATERIALS.keys()), required=True),
        "NominalDiameter[inch]": st.column_config.NumberColumn("Nom. [inch]", format="%.1f"),
        "Corrosion[mm]": st.column_config.NumberColumn("Corr. [mm]", min_value=0.0, format="%.2f"),
        "WallTolerance[frac]": st.column_config.NumberColumn(
            "Wall tol. [frac]", min_value=0.0, max_value=0.5, format="%.3f"),
        "WeightStress[MPa]": st.column_config.NumberColumn("Weight σ [MPa]", format="%.1f"),
        "FluidDensity[kg/m³]": st.column_config.NumberColumn("Fluid ρ [kg/m³]", format="%.2f"),
        "FluidCp[J/kgK]": st.column_config.NumberColumn("Fluid Cp [J/kgK]", format="%.1f"),
        "GasMW[g/mol]": st.column_config.NumberColumn("Gas MW [g/mol]", format="%.2f"),
    },
    key="fr_pipes_editor",
)

with st.expander("📚 Pipe-class database (reference dimensions)"):
    st.caption("Common pipe dimensions for convenience — copy values into the table above.")
    st.dataframe(pd.DataFrame(fr.DEFAULT_PIPE_DATABASE), hide_index=True,
                 use_container_width=True)

st.divider()

# =============================================================================
# Run
# =============================================================================
run = st.button("▶️ Run calculation", type="primary", use_container_width=True)


if run:
    prof = profile_df.dropna().sort_values("Time[min]")
    if len(prof) < 2:
        st.error("Please provide at least two points in the pressure profile.")
        st.stop()
    fires = _build_fires(fires_df)
    if not fires:
        st.error("Please define at least one fire in the sidebar.")
        st.stop()
    pipes_input = pipes_df.dropna(subset=["Name", "OD[mm]", "Wall[mm]"])
    if pipes_input.empty:
        st.error("Please define at least one pipe.")
        st.stop()

    prof_t = prof["Time[min]"].tolist()
    prof_p = prof["Pressure[bara]"].tolist()

    pipe_results = []
    with st.spinner("Running fire-rupture calculation…"):
        for _, row in pipes_input.iterrows():
            try:
                pipe = _build_pipe(row)
                pipe_results.append(fr.evaluate_pipe(
                    pipe, fires, prof_t, prof_p,
                    time_step_s=float(time_step_s),
                    initial_temp_C=float(initial_temp_C)))
            except Exception as exc:  # surface a clear message per pipe
                st.error(f"Pipe '{row.get('Name', '?')}' failed: {exc}")

    if not pipe_results:
        st.stop()

    st.session_state.fr_results = pipe_results
    st.session_state.fr_fire_names = [f.name for f in fires]

# =============================================================================
# Results
# =============================================================================
if st.session_state.get("fr_results"):
    pipe_results = st.session_state.fr_results
    fire_names = st.session_state.fr_fire_names

    st.subheader("📊 Summary — time to rupture")

    summary_rows = []
    for pr in pipe_results:
        for fres in pr.fires:
            summary_rows.append({
                "Pipe": pr.pipe.name,
                "Material": pr.pipe.material,
                "Fire": fres.fire_name,
                "Ruptures": "Yes" if fres.ruptured else "No",
                "Time to rupture [min]": fres.time_to_rupture_min,
                "Rupture pressure [barg]": fres.rupture_pressure_barg,
                "Release area [m²]": pr.pipe.release_cross_area_m2,
                "Gas 2-sided [kg/s]": fres.release_gas_2sides,
                "Gas 1-sided [kg/s]": fres.release_gas_1side,
                "Gas short pipe [kg/s]": fres.release_gas_short,
                "Liquid [kg/s]": fres.release_liquid,
            })
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(
        summary_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Time to rupture [min]": st.column_config.NumberColumn(format="%.2f"),
            "Rupture pressure [barg]": st.column_config.NumberColumn(format="%.2f"),
            "Release area [m²]": st.column_config.NumberColumn(format="%.5f"),
            "Gas 2-sided [kg/s]": st.column_config.NumberColumn(format="%.2f"),
            "Gas 1-sided [kg/s]": st.column_config.NumberColumn(format="%.2f"),
            "Gas short pipe [kg/s]": st.column_config.NumberColumn(format="%.2f"),
            "Liquid [kg/s]": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption("The **2-sided gas** rate is normally recommended as the release rate. "
               "A liquid rate is reported only when the fluid density > 500 kg/m³.")

    st.download_button(
        "⬇️ Download summary (CSV)",
        summary_df.to_csv(index=False).encode("utf-8"),
        file_name="fire_rupture_summary.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("📈 Transient details")

    pipe_names = [pr.pipe.name for pr in pipe_results]
    sel_pipe = st.selectbox("Select pipe", pipe_names)
    sel_fire = st.selectbox("Select fire", fire_names)

    sel_pr = next(pr for pr in pipe_results if pr.pipe.name == sel_pipe)
    sel_fres = next(f for f in sel_pr.fires if f.fire_name == sel_fire)

    cols = st.columns(3)
    cols[0].metric("Time to rupture",
                   f"{sel_fres.time_to_rupture_min:.2f} min" if sel_fres.ruptured else "No rupture")
    cols[1].metric("Rupture pressure",
                   f"{sel_fres.rupture_pressure_barg:.2f} barg" if sel_fres.ruptured else "—")
    if sel_pr.pipe.is_liquid:
        cols[2].metric("Liquid release", _fmt(sel_fres.release_liquid, "{:.2f} kg/s"))
    else:
        cols[2].metric("Gas release (2-sided)", _fmt(sel_fres.release_gas_2sides, "{:.2f} kg/s"))

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Wall temperature", "Accumulated strain vs limit"),
        vertical_spacing=0.12)

    fig.add_trace(go.Scatter(x=sel_fres.time_min, y=sel_fres.mean_temp_C,
                             name="Mean wall T", line=dict(color="#ff7f0e")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sel_fres.time_min, y=sel_fres.surface_temp_C,
                             name="Surface T", line=dict(color="#d62728", dash="dot")), row=1, col=1)

    fig.add_trace(go.Scatter(x=sel_fres.time_min, y=sel_fres.strain,
                             name="Strain", line=dict(color="#1f77b4")), row=2, col=1)
    fig.add_trace(go.Scatter(x=sel_fres.time_min, y=sel_fres.strain_limit,
                             name="Strain limit", line=dict(color="#2ca02c", dash="dash")), row=2, col=1)

    if sel_fres.ruptured:
        fig.add_vline(x=sel_fres.time_to_rupture_min, line=dict(color="grey", dash="dot"),
                      annotation_text="rupture", row="all")

    fig.update_yaxes(title_text="Temperature [°C]", row=1, col=1)
    fig.update_yaxes(title_text="Strain [-]", row=2, col=1)
    fig.update_xaxes(title_text="Time [min]", row=2, col=1)
    fig.update_layout(height=600, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)
