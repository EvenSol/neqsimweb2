import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from neqsim import jneqsim
from neqsim.thermo import fluid_df

from theme import apply_theme


st.set_page_config(
    page_title="Elemental Sulfur",
    page_icon="images/neqsimlogocircleflat.png",
    layout="wide",
)
apply_theme()

st.title("Elemental Sulfur Solubility and Precipitation")
st.markdown(
    """
Evaluate elemental sulfur (S8) solubility, solid sulfur precipitation, and operating margins for sour gas,
condensate, and oil-rich systems. The calculation uses the NeqSim solid sulfur flash workflow: SRK equation
of state, Huron-Vidal mixing rule, multiphase checks, and `TPSolidflash` with S8 as the solid phase.

The main result is **Dissolved S8**, reported on a feed basis and including S8 dissolved in gas plus any
hydrocarbon liquid or condensate phase. The table also reports **S8 in gas** and **S8 in condensate/oil**
separately so you can see when liquid dropout inside the phase envelope increases sulfur carrying capacity.
"""
)
with st.expander("Result interpretation", expanded=False):
    st.markdown(
        """
- **Solid sulfur = Yes** means the supplied S8 inventory is above the equilibrium dissolved capacity at the selected condition.
- **Dissolved S8 [mg/Sm3]** is the total non-solid S8 on a feed basis: gas + hydrocarbon liquid/condensate.
- **S8 in gas [mg/Sm3]** is the gas-phase contribution only and may decrease when condensate forms.
- **S8 in condensate/oil [mg/Sm3 feed]** is the hydrocarbon liquid contribution; this often explains higher total solubility inside the phase envelope.
- **S8 solubility contours** use the same total dissolved S8 basis, so contours inside the envelope can reflect condensate uptake.
"""
    )
st.divider()


S8_MOLAR_MASS_KG_PER_MOL = 256.48e-3
GAS_CONSTANT = 8.314462618
STANDARD_TEMPERATURE_K = 288.15
STANDARD_PRESSURE_PA = 101325.0
S8_MOLFRAC_TO_MG_SM3 = (
    S8_MOLAR_MASS_KG_PER_MOL
    * 1.0e6
    * STANDARD_PRESSURE_PA
    / (GAS_CONSTANT * STANDARD_TEMPERATURE_K)
)
PHASE_ENVELOPE_EXCLUDED_COMPONENTS = {"s8", "oxygen", "so2", "water"}
PHASE_ENVELOPE_MODEL_OPTIONS = {
    "SRK": "srk",
    "PR": "PrEos",
}
SULFUR_REACTIONS = {
    "Direct H2S oxidation": {
        "equation": "8 H2S + 4 O2 -> S8 + 8 H2O",
        "stoichiometry": {"H2S": -8.0, "oxygen": -4.0, "S8": 1.0, "water": 8.0},
        "limiting_reactants": ["H2S", "oxygen"],
    },
    "Claus reaction": {
        "equation": "16 H2S + 8 SO2 -> 3 S8 + 16 H2O",
        "stoichiometry": {"H2S": -16.0, "SO2": -8.0, "S8": 3.0, "water": 16.0},
        "limiting_reactants": ["H2S", "SO2"],
    },
}


default_sulfur_fluid = {
    "ComponentName": [
        "nitrogen",
        "CO2",
        "H2S",
        "oxygen",
        "methane",
        "ethane",
        "propane",
        "i-butane",
        "n-butane",
        "i-pentane",
        "n-pentane",
        "n-hexane",
        "C7",
        "water",
        "SO2",
        "S8",
    ],
    "MolarComposition[-]": [
        0.005,
        0.020,
        50.0e-6,
        1.0e-6,
        0.920,
        0.035,
        0.012,
        0.003,
        0.002,
        0.0008,
        0.0005,
        0.0002,
        0.001,
        0.0005,
        0.0,
        10.0e-6,
    ],
    "MolarMass[kg/mol]": [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        0.0913,
        None,
        None,
        None,
    ],
    "RelativeDensity[-]": [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        0.746,
        None,
        None,
        None,
    ],
}


default_oil_sulfur_fluid = {
    "ComponentName": [
        "nitrogen",
        "CO2",
        "H2S",
        "methane",
        "ethane",
        "propane",
        "i-butane",
        "n-butane",
        "i-pentane",
        "n-pentane",
        "n-hexane",
        "C7",
        "C8",
        "C9",
        "C10",
        "C12",
        "C15",
        "C20",
        "S8",
        "C30",
    ],
    "MolarComposition[-]": [
        0.006,
        0.018,
        200.0e-6,
        0.420,
        0.090,
        0.065,
        0.025,
        0.030,
        0.018,
        0.018,
        0.020,
        0.040,
        0.038,
        0.035,
        0.032,
        0.045,
        0.055,
        0.060,
        10.0e-6,
        0.025,
    ],
    "MolarMass[kg/mol]": [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        0.0913,
        0.1070,
        0.1210,
        0.1420,
        0.1700,
        0.2060,
        0.2750,
        None,
        0.4200,
    ],
    "RelativeDensity[-]": [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        0.746,
        0.765,
        0.781,
        0.800,
        0.820,
        0.836,
        0.860,
        None,
        0.910,
    ],
}


def _normalise_name(component_name):
    return str(component_name).strip().lower()


def _ensure_composition_columns(composition_df):
    prepared = composition_df.copy()
    for column in ["ComponentName", "MolarComposition[-]", "MolarMass[kg/mol]", "RelativeDensity[-]"]:
        if column not in prepared.columns:
            prepared[column] = None
    return prepared[["ComponentName", "MolarComposition[-]", "MolarMass[kg/mol]", "RelativeDensity[-]"]]


def _component_value(composition_df, component_name):
    matches = composition_df[
        composition_df["ComponentName"].map(_normalise_name) == component_name.lower()
    ]
    if matches.empty:
        return 0.0
    return float(matches.iloc[0]["MolarComposition[-]"])


def _set_component_value(composition_df, component_name, value):
    updated = composition_df.copy()
    mask = updated["ComponentName"].map(_normalise_name) == component_name.lower()
    if mask.any():
        updated.loc[mask, "MolarComposition[-]"] = value
        return updated

    new_row = {
        "ComponentName": component_name,
        "MolarComposition[-]": value,
        "MolarMass[kg/mol]": None,
        "RelativeDensity[-]": None,
    }
    return pd.concat([updated, pd.DataFrame([new_row])], ignore_index=True)


def _add_component_value(composition_df, component_name, delta_value):
    return _set_component_value(
        composition_df,
        component_name,
        _component_value(composition_df, component_name) + float(delta_value),
    )


def _has_pseudo_properties(row):
    molar_mass = pd.to_numeric(row.get("MolarMass[kg/mol]"), errors="coerce")
    relative_density = pd.to_numeric(row.get("RelativeDensity[-]"), errors="coerce")
    return bool(
        pd.notna(molar_mass)
        and pd.notna(relative_density)
        and float(molar_mass) > 0.0
        and float(relative_density) > 0.0
    )


def _composition_for_fluid_df(composition_df, is_plus_fraction):
    fluid_df_input = composition_df.copy().reset_index(drop=True)
    if not is_plus_fraction:
        return fluid_df_input, False

    eligible_indices = [
        index
        for index, row in fluid_df_input.iterrows()
        if _has_pseudo_properties(row) and _normalise_name(row["ComponentName"]) not in PHASE_ENVELOPE_EXCLUDED_COMPONENTS
    ]
    if not eligible_indices:
        return fluid_df_input, False

    plus_index = eligible_indices[-1]
    plus_row = fluid_df_input.loc[[plus_index]]
    fluid_df_input = pd.concat(
        [fluid_df_input.drop(index=plus_index), plus_row],
        ignore_index=True,
    )
    return fluid_df_input, True


def _composition_has_pseudo_components(composition_df):
    return any(
        _has_pseudo_properties(row)
        and _normalise_name(row["ComponentName"]) not in PHASE_ENVELOPE_EXCLUDED_COMPONENTS
        for _, row in composition_df.iterrows()
    )


def _prepare_composition(composition_df, s8_ppmv, use_s8_input, add_stoichiometric_s8):
    prepared = _ensure_composition_columns(composition_df)
    numeric_columns = ["MolarComposition[-]", "MolarMass[kg/mol]", "RelativeDensity[-]"]
    for column in numeric_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared["ComponentName"] = prepared["ComponentName"].astype(str).str.strip()
    prepared = prepared[prepared["ComponentName"] != ""]

    if use_s8_input:
        prepared = _set_component_value(prepared, "S8", s8_ppmv * 1.0e-6)

    if add_stoichiometric_s8:
        h2s = _component_value(prepared, "H2S")
        oxygen = _component_value(prepared, "oxygen")
        s8_from_oxidation = min(h2s / 8.0, oxygen / 4.0)
        prepared = _set_component_value(
            prepared,
            "S8",
            _component_value(prepared, "S8") + max(s8_from_oxidation, 0.0),
        )

    prepared = prepared[prepared["MolarComposition[-]"] > 0.0].reset_index(drop=True)
    return prepared


def _calculate_reaction_final_composition(composition_df, reaction_name, conversion_percent):
    reaction = SULFUR_REACTIONS[reaction_name]
    stoichiometry = reaction["stoichiometry"]
    feed = _ensure_composition_columns(composition_df)
    feed = feed.copy()
    numeric_columns = ["MolarComposition[-]", "MolarMass[kg/mol]", "RelativeDensity[-]"]
    for column in numeric_columns:
        feed[column] = pd.to_numeric(feed[column], errors="coerce")
    feed["ComponentName"] = feed["ComponentName"].astype(str).str.strip()
    feed = feed[(feed["ComponentName"] != "") & (feed["MolarComposition[-]"] > 0.0)].reset_index(drop=True)

    if feed.empty:
        return {
            "final_composition": feed,
            "reaction_summary": pd.DataFrame(),
            "extent": 0.0,
            "equation": reaction["equation"],
        }

    maximum_extent = math.inf
    limiting_component = None
    for component_name in reaction["limiting_reactants"]:
        coefficient = abs(stoichiometry[component_name])
        available = _component_value(feed, component_name)
        candidate_extent = available / coefficient if coefficient > 0.0 else math.inf
        if candidate_extent < maximum_extent:
            maximum_extent = candidate_extent
            limiting_component = component_name

    if not math.isfinite(maximum_extent):
        maximum_extent = 0.0

    extent = max(maximum_extent, 0.0) * max(min(float(conversion_percent), 100.0), 0.0) / 100.0
    final = feed.copy()
    rows = []
    for component_name, coefficient in stoichiometry.items():
        initial_amount = _component_value(feed, component_name)
        final_amount = max(initial_amount + coefficient * extent, 0.0)
        final = _set_component_value(final, component_name, final_amount)
        rows.append(
            {
                "Component": component_name,
                "Stoichiometric coefficient": coefficient,
                "Initial relative mol": initial_amount,
                "Change relative mol": coefficient * extent,
                "Final relative mol": final_amount,
            }
        )

    final = final[final["MolarComposition[-]"] > 0.0].reset_index(drop=True)
    total_final = float(final["MolarComposition[-]"].sum())
    if total_final > 0.0:
        final["Final mole fraction [-]"] = final["MolarComposition[-]"] / total_final

    s8_formed = max(stoichiometry.get("S8", 0.0) * extent, 0.0)
    return {
        "final_composition": final,
        "reaction_summary": pd.DataFrame(rows),
        "extent": extent,
        "maximum_extent": maximum_extent,
        "limiting_component": limiting_component,
        "s8_formed_mol": s8_formed,
        "s8_formed_mg_sm3": s8_formed / max(total_final, 1.0e-30) * S8_MOLFRAC_TO_MG_SM3,
        "equation": reaction["equation"],
    }


def _phase_envelope_composition(composition_df):
    envelope_df = composition_df[
        ~composition_df["ComponentName"].map(_normalise_name).isin(PHASE_ENVELOPE_EXCLUDED_COMPONENTS)
    ].copy()
    return envelope_df[envelope_df["MolarComposition[-]"] > 0.0].reset_index(drop=True)


def _create_sulfur_fluid(composition_df, temperature_c, pressure_bara, is_plus_fraction, model_name="srk"):
    fluid_input, effective_plus_fraction = _composition_for_fluid_df(composition_df, is_plus_fraction)
    jneqsim.util.database.NeqSimDataBase.setCreateTemporaryTables(True)
    try:
        system = fluid_df(
            fluid_input,
            lastIsPlusFraction=effective_plus_fraction,
            add_all_components=False,
        )
        system = system.setModel(model_name)
    finally:
        jneqsim.util.database.NeqSimDataBase.setCreateTemporaryTables(False)

    system.setMixingRule(2)
    system.setMultiPhaseCheck(True)
    system.setSolidPhaseCheck("S8")
    system.setTemperature(float(temperature_c), "C")
    system.setPressure(float(pressure_bara), "bara")
    return system


def _create_phase_envelope_fluid(composition_df, is_plus_fraction, model_name):
    fluid_input, effective_plus_fraction = _composition_for_fluid_df(composition_df, is_plus_fraction)
    jneqsim.util.database.NeqSimDataBase.setCreateTemporaryTables(True)
    try:
        system = fluid_df(
            fluid_input,
            lastIsPlusFraction=effective_plus_fraction,
            add_all_components=False,
        ).setModel(model_name)
    finally:
        jneqsim.util.database.NeqSimDataBase.setCreateTemporaryTables(False)
    return system


def _run_solid_flash(system):
    operations = jneqsim.thermodynamicoperations.ThermodynamicOperations(system)
    operations.TPSolidflash()
    try:
        system.initProperties()
    except Exception:
        pass


def _phase_envelope_from_operation(operation, model_name, is_plus_fraction, fallback_used):
    try:
        dew_temperatures = [float(value) - 273.15 for value in list(operation.get("dewT"))]
        dew_pressures = [float(value) for value in list(operation.get("dewP"))]
        bubble_temperatures = [float(value) - 273.15 for value in list(operation.get("bubT"))]
        bubble_pressures = [float(value) for value in list(operation.get("bubP"))]
    except Exception:
        dew_temperatures = []
        dew_pressures = []
        bubble_temperatures = []
        bubble_pressures = []

    if len(dew_temperatures) == 0 and len(bubble_temperatures) == 0:
        temperatures = _phase_envelope_temperatures_to_c(
            [float(value) for value in list(operation.getTemperaturePhaseEnvelope())]
        )
        pressures = [float(value) for value in list(operation.getPressurePhaseEnvelope())]
        if len(temperatures) == 0:
            raise RuntimeError("Phase envelope calculation returned no points.")
        return {
            "dew": pd.DataFrame({"Temperature [degC]": temperatures, "Pressure [bara]": pressures}),
            "bubble": pd.DataFrame(),
            "model": model_name,
            "plus_fraction_used": is_plus_fraction,
            "fallback_used": fallback_used,
        }

    return {
        "dew": pd.DataFrame({"Temperature [degC]": dew_temperatures, "Pressure [bara]": dew_pressures}),
        "bubble": pd.DataFrame({"Temperature [degC]": bubble_temperatures, "Pressure [bara]": bubble_pressures}),
        "model": model_name,
        "plus_fraction_used": is_plus_fraction,
        "fallback_used": fallback_used,
    }


def _calculate_phase_envelope_for_attempt(composition_df, is_plus_fraction, model_name, method_name):
    system = _create_phase_envelope_fluid(composition_df, is_plus_fraction, model_name)
    operations = jneqsim.thermodynamicoperations.ThermodynamicOperations(system)
    if method_name == "michelsen":
        operations.calcPTphaseEnvelope2()
    elif method_name == "grid":
        operations.calcPTphaseEnvelopeNew3(1.0, 200.0, -80.0, 200.0, 5.0, 10.0)
    else:
        raise ValueError(f"Unknown phase-envelope method: {method_name}")

    phase_envelope = _phase_envelope_from_operation(
        operations.getOperation(),
        model_name,
        is_plus_fraction,
        method_name != "michelsen",
    )
    phase_envelope["method_used"] = method_name
    return phase_envelope


def _calculate_phase_envelope(composition_df, is_plus_fraction, model_name):
    if composition_df.empty or composition_df["MolarComposition[-]"].sum() <= 0.0:
        return {"dew": pd.DataFrame(), "bubble": pd.DataFrame(), "model": model_name, "attempt_errors": []}

    has_pseudo_components = _composition_has_pseudo_components(composition_df)
    fallback_models = [model_name]
    for candidate_model in PHASE_ENVELOPE_MODEL_OPTIONS.values():
        if candidate_model not in fallback_models:
            fallback_models.append(candidate_model)

    attempts = []
    for attempt_model in fallback_models:
        for attempt_plus_fraction in [is_plus_fraction, True if has_pseudo_components else is_plus_fraction]:
            attempt = (attempt_model, attempt_plus_fraction)
            if attempt not in attempts:
                attempts.append(attempt)

    errors = []
    for attempt_model, attempt_plus_fraction in attempts:
        for method_name in ["michelsen", "grid"]:
            try:
                phase_envelope = _calculate_phase_envelope_for_attempt(
                    composition_df,
                    attempt_plus_fraction,
                    attempt_model,
                    method_name,
                )
                phase_envelope["requested_model"] = model_name
                phase_envelope["attempt_errors"] = errors
                return phase_envelope
            except Exception as error:
                errors.append(
                    {
                        "model": attempt_model,
                        "plus_fraction_used": attempt_plus_fraction,
                        "method": method_name,
                        "error": str(error),
                    }
                )

    raise RuntimeError("All phase-envelope attempts failed: " + "; ".join(item["error"] for item in errors))


def _phase_envelope_min_temperature(phase_envelope):
    temperatures = []
    for branch_name in ["dew", "bubble"]:
        branch = phase_envelope.get(branch_name, pd.DataFrame())
        if not branch.empty and "Temperature [degC]" in branch.columns:
            temperatures.extend(pd.to_numeric(branch["Temperature [degC]"], errors="coerce").dropna().tolist())
    return min(temperatures) if temperatures else math.nan


def _phase_envelope_temperatures_to_c(temperatures):
    values = [float(value) for value in temperatures]
    finite_values = [value for value in values if math.isfinite(value)]
    if finite_values and np.nanmedian(finite_values) > 170.0:
        return [value - 273.15 for value in values]
    return values

def _phase_envelope_range(phase_envelope, column_name):
    values = []
    for branch_name in ["dew", "bubble"]:
        branch = phase_envelope.get(branch_name, pd.DataFrame())
        if not branch.empty and column_name in branch:
            values.extend(branch[column_name].dropna().astype(float).tolist())
    if not values:
        return math.nan, math.nan
    return min(values), max(values)


def _phase_component_x(system, phase_name, component_name):
    try:
        phase_number = system.getPhaseNumberOfPhase(phase_name)
        if phase_number >= 0:
            return float(system.getPhase(phase_number).getComponent(component_name).getx())
    except Exception:
        pass
    try:
        return float(system.getPhase(0).getComponent(component_name).getx())
    except Exception:
        return math.nan


def _phase_type_name(system, phase_index):
    try:
        return str(system.getPhase(phase_index).getPhaseTypeName()).lower()
    except Exception:
        return ""


def _phase_component_mole_fraction(system, phase_index, component_name):
    try:
        return float(system.getPhase(phase_index).getComponent(component_name).getx())
    except Exception:
        return math.nan


def _phase_beta(system, phase_index):
    try:
        return float(system.getBeta(phase_index))
    except Exception:
        try:
            return float(system.getPhase(phase_index).getBeta())
        except Exception:
            return math.nan


def _dissolved_s8(system):
    total_s8_mol_fraction = 0.0
    gas_s8_mol_fraction = math.nan
    condensate_s8_mol_fraction = 0.0
    condensate_beta = 0.0

    for phase_index in range(int(system.getNumberOfPhases())):
        phase_type = _phase_type_name(system, phase_index)
        if "solid" in phase_type or "aqueous" in phase_type or "water" in phase_type:
            continue
        phase_s8 = _phase_component_mole_fraction(system, phase_index, "S8")
        if math.isnan(phase_s8):
            continue
        beta = _phase_beta(system, phase_index)
        if math.isnan(beta):
            beta = 0.0
        total_s8_mol_fraction += beta * phase_s8
        if "gas" in phase_type or "vapor" in phase_type or "vapour" in phase_type:
            gas_s8_mol_fraction = phase_s8
        else:
            condensate_s8_mol_fraction += beta * phase_s8
            condensate_beta += beta

    return {
        "total_mol_fraction": total_s8_mol_fraction,
        "total_mg_sm3": total_s8_mol_fraction * S8_MOLFRAC_TO_MG_SM3,
        "gas_mol_fraction": gas_s8_mol_fraction,
        "gas_mg_sm3": gas_s8_mol_fraction * S8_MOLFRAC_TO_MG_SM3 if not math.isnan(gas_s8_mol_fraction) else math.nan,
        "condensate_mg_sm3": condensate_s8_mol_fraction * S8_MOLFRAC_TO_MG_SM3,
        "condensate_beta": condensate_beta,
    }


def _solid_flash_result(composition_df, temperature_c, pressure_bara, is_plus_fraction, model_name="srk"):
    system = _create_sulfur_fluid(composition_df, temperature_c, pressure_bara, is_plus_fraction, model_name)
    _run_solid_flash(system)

    solid_present = bool(system.hasPhaseType("solid"))
    solid_phase_fraction = 0.0
    solid_weight_percent = 0.0

    if solid_present:
        try:
            solid_phase_number = system.getPhaseNumberOfPhase("solid")
            solid_phase_fraction = float(system.getBeta(solid_phase_number))
            solid_weight_percent = float(system.getWtFraction(solid_phase_number)) * 100.0
        except Exception:
            solid_phase_fraction = math.nan
            solid_weight_percent = math.nan

    dissolved_s8 = _dissolved_s8(system)
    s8_gas_molfrac = dissolved_s8["gas_mol_fraction"]
    s8_gas_mg_sm3 = dissolved_s8["gas_mg_sm3"]
    h2s_gas_molfrac = _phase_component_x(system, "gas", "H2S")
    supplied_s8_molfrac = _component_value(composition_df, "S8") / max(
        float(composition_df["MolarComposition[-]"].sum()),
        1.0e-30,
    )
    supplied_s8_mg_sm3 = supplied_s8_molfrac * S8_MOLFRAC_TO_MG_SM3

    return {
        "Temperature [degC]": float(temperature_c),
        "Pressure [bara]": float(pressure_bara),
        "Solid sulfur present": solid_present,
        "Solid phase fraction [-]": solid_phase_fraction,
        "Solid sulfur [wt%]": solid_weight_percent,
        "Dissolved S8 [mol frac feed basis]": dissolved_s8["total_mol_fraction"],
        "Dissolved S8 [mg/Sm3]": dissolved_s8["total_mg_sm3"],
        "S8 in gas [mol frac]": s8_gas_molfrac,
        "S8 in gas [mg/Sm3]": s8_gas_mg_sm3,
        "S8 in condensate/oil [mg/Sm3 feed]": dissolved_s8["condensate_mg_sm3"],
        "Condensate/oil phase fraction [-]": dissolved_s8["condensate_beta"],
        "Supplied S8 [mg/Sm3]": supplied_s8_mg_sm3,
        "H2S in gas [ppmv]": h2s_gas_molfrac * 1.0e6 if not math.isnan(h2s_gas_molfrac) else math.nan,
        "Number of phases": int(system.getNumberOfPhases()),
    }


def _sulfur_solubility_at(
    composition_df,
    temperature_c,
    pressure_bara,
    is_plus_fraction,
    saturation_s8_ppmv,
    model_name="srk",
):
    saturated_df = _set_component_value(composition_df, "S8", saturation_s8_ppmv * 1.0e-6)
    return _solid_flash_result(saturated_df, temperature_c, pressure_bara, is_plus_fraction, model_name)


def _run_solubility_grid(
    composition_df,
    temperature_start_c,
    temperature_end_c,
    temperature_points,
    pressure_start_bara,
    pressure_end_bara,
    pressure_points,
    is_plus_fraction,
    saturation_s8_ppmv,
    model_name="srk",
):
    temperatures = np.linspace(float(temperature_start_c), float(temperature_end_c), int(temperature_points))
    pressures = np.linspace(float(pressure_start_bara), float(pressure_end_bara), int(pressure_points))
    rows = []
    z_values = []
    errors = []

    for pressure_bara in pressures:
        z_row = []
        for temperature_c in temperatures:
            try:
                result = _sulfur_solubility_at(
                    composition_df,
                    temperature_c,
                    pressure_bara,
                    is_plus_fraction,
                    saturation_s8_ppmv,
                    model_name,
                )
                solubility = result["Dissolved S8 [mg/Sm3]"]
                solid_sulfur_present = result["Solid sulfur present"]
                point_error = ""
            except Exception as error:
                solubility = math.nan
                solid_sulfur_present = math.nan
                point_error = str(error)
                errors.append(
                    {
                        "Temperature [degC]": float(temperature_c),
                        "Pressure [bara]": float(pressure_bara),
                        "error": point_error,
                    }
                )
            z_row.append(solubility)
            rows.append(
                {
                    "Temperature [degC]": float(temperature_c),
                    "Pressure [bara]": float(pressure_bara),
                    "S8 solubility [mg/Sm3]": solubility,
                    "S8 in gas [mg/Sm3]": result.get("S8 in gas [mg/Sm3]", math.nan)
                    if not point_error
                    else math.nan,
                    "S8 in condensate/oil [mg/Sm3 feed]": result.get(
                        "S8 in condensate/oil [mg/Sm3 feed]",
                        math.nan,
                    )
                    if not point_error
                    else math.nan,
                    "Condensate/oil phase fraction [-]": result.get("Condensate/oil phase fraction [-]", math.nan)
                    if not point_error
                    else math.nan,
                    "Solid sulfur present": solid_sulfur_present,
                    "Calculation error": point_error,
                }
            )
        z_values.append(z_row)

    return {
        "temperatures": temperatures,
        "pressures": pressures,
        "solubility": np.array(z_values),
        "table": pd.DataFrame(rows),
        "errors": errors,
        "model": model_name,
    }


def _run_temperature_sweep(composition_df, pressure_bara, start_c, end_c, step_c, is_plus_fraction):
    temperatures = np.arange(float(start_c), float(end_c) + 0.5 * float(step_c), float(step_c))
    rows = []
    for temperature_c in temperatures:
        rows.append(_solid_flash_result(composition_df, temperature_c, pressure_bara, is_plus_fraction))
    return pd.DataFrame(rows)


def _run_pressure_sweep(composition_df, temperature_c, start_bara, end_bara, step_bara, is_plus_fraction):
    pressures = np.arange(float(start_bara), float(end_bara) + 0.5 * float(step_bara), float(step_bara))
    rows = []
    for pressure_bara in pressures:
        rows.append(_solid_flash_result(composition_df, temperature_c, pressure_bara, is_plus_fraction))
    return pd.DataFrame(rows)


def _cooling_onset_temperature(results_df):
    solid_rows = results_df[results_df["Solid sulfur present"]]
    if solid_rows.empty:
        return math.nan
    return float(solid_rows["Temperature [degC]"].max())


def _plot_sweep(results_df, x_column, title):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=results_df[x_column],
            y=results_df["Dissolved S8 [mg/Sm3]"],
            mode="lines+markers",
            name="Dissolved S8 (gas + oil)",
            yaxis="y1",
        )
    )
    if "S8 in gas [mg/Sm3]" in results_df.columns:
        fig.add_trace(
            go.Scatter(
                x=results_df[x_column],
                y=results_df["S8 in gas [mg/Sm3]"],
                mode="lines",
                name="S8 in gas only",
                yaxis="y1",
                line=dict(dash="dash"),
            )
        )
    if "S8 in condensate/oil [mg/Sm3 feed]" in results_df.columns:
        fig.add_trace(
            go.Scatter(
                x=results_df[x_column],
                y=results_df["S8 in condensate/oil [mg/Sm3 feed]"],
                mode="lines",
                name="S8 in condensate/oil",
                yaxis="y1",
                line=dict(dash="dot"),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=results_df[x_column],
            y=results_df["Solid sulfur [wt%]"],
            mode="lines+markers",
            name="Solid sulfur",
            yaxis="y2",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_column,
        yaxis=dict(title="Dissolved S8 [mg/Sm3]"),
        yaxis2=dict(title="Solid sulfur [wt%]", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=40, r=40, t=70, b=40),
    )
    return fig


def _parse_contour_levels(level_text):
    levels = []
    for raw_value in str(level_text).split(","):
        try:
            value = float(raw_value.strip())
            if value > 0.0:
                levels.append(value)
        except ValueError:
            pass
    return sorted(set(levels))


def _phase_envelope_polygon(phase_envelope):
    polygon_x = []
    polygon_y = []

    for branch_name, reverse_branch in (("dew", False), ("bubble", True)):
        branch = phase_envelope.get(branch_name, pd.DataFrame())
        if branch.empty:
            continue
        branch_points = branch[["Temperature [degC]", "Pressure [bara]"]].dropna().sort_values("Temperature [degC]")
        if reverse_branch:
            branch_points = branch_points.iloc[::-1]
        polygon_x.extend(branch_points["Temperature [degC]"].astype(float).tolist())
        polygon_y.extend(branch_points["Pressure [bara]"].astype(float).tolist())

    if len(polygon_x) < 3:
        return None

    filtered_x = []
    filtered_y = []
    for temperature_c, pressure_bara in zip(polygon_x, polygon_y):
        if filtered_x and math.isclose(temperature_c, filtered_x[-1]) and math.isclose(pressure_bara, filtered_y[-1]):
            continue
        filtered_x.append(temperature_c)
        filtered_y.append(pressure_bara)

    if len(filtered_x) < 3:
        return None
    return np.asarray(filtered_x, dtype=float), np.asarray(filtered_y, dtype=float)


def _phase_envelope_grid_mask(phase_envelope, temperatures, pressures):
    polygon = _phase_envelope_polygon(phase_envelope)
    if polygon is None:
        return None

    polygon_x, polygon_y = polygon
    grid_x, grid_y = np.meshgrid(np.asarray(temperatures, dtype=float), np.asarray(pressures, dtype=float))
    inside = np.zeros(grid_x.shape, dtype=bool)
    point_count = len(polygon_x)

    for index in range(point_count):
        previous_index = (index - 1) % point_count
        x_current = polygon_x[index]
        y_current = polygon_y[index]
        x_previous = polygon_x[previous_index]
        y_previous = polygon_y[previous_index]
        crosses_pressure = (y_current > grid_y) != (y_previous > grid_y)
        crossing_temperature = (x_previous - x_current) * (grid_y - y_current) / (
            y_previous - y_current + 1.0e-30
        ) + x_current
        inside ^= crosses_pressure & (grid_x < crossing_temperature)

    return inside


def _point_inside_polygon(point_x, point_y, polygon_x, polygon_y):
    inside = False
    point_count = len(polygon_x)
    for index in range(point_count):
        previous_index = (index - 1) % point_count
        x_current = polygon_x[index]
        y_current = polygon_y[index]
        x_previous = polygon_x[previous_index]
        y_previous = polygon_y[previous_index]
        if (y_current > point_y) != (y_previous > point_y):
            crossing_x = (x_previous - x_current) * (point_y - y_current) / (
                y_previous - y_current + 1.0e-30
            ) + x_current
            if point_x < crossing_x:
                inside = not inside
    return inside


def _contour_segments_inside_envelope(
    phase_envelope,
    temperatures,
    pressures,
    z_values,
    contour_level,
    inside_envelope_only=True,
):
    polygon = _phase_envelope_polygon(phase_envelope) if inside_envelope_only else None
    if inside_envelope_only and polygon is None:
        return None

    polygon_x, polygon_y = polygon if polygon is not None else ([], [])
    temperatures = np.asarray(temperatures, dtype=float)
    pressures = np.asarray(pressures, dtype=float)
    z_values = np.asarray(z_values, dtype=float)
    segment_x = []
    segment_y = []

    for pressure_index in range(len(pressures) - 1):
        for temperature_index in range(len(temperatures) - 1):
            corners = [
                (
                    temperatures[temperature_index],
                    pressures[pressure_index],
                    z_values[pressure_index, temperature_index],
                ),
                (
                    temperatures[temperature_index + 1],
                    pressures[pressure_index],
                    z_values[pressure_index, temperature_index + 1],
                ),
                (
                    temperatures[temperature_index + 1],
                    pressures[pressure_index + 1],
                    z_values[pressure_index + 1, temperature_index + 1],
                ),
                (
                    temperatures[temperature_index],
                    pressures[pressure_index + 1],
                    z_values[pressure_index + 1, temperature_index],
                ),
            ]
            crossings = []
            for start_index, end_index in ((0, 1), (1, 2), (2, 3), (3, 0)):
                x_start, y_start, z_start = corners[start_index]
                x_end, y_end, z_end = corners[end_index]
                if not np.isfinite(z_start) or not np.isfinite(z_end) or math.isclose(z_start, z_end):
                    continue
                if contour_level < min(z_start, z_end) or contour_level > max(z_start, z_end):
                    continue
                fraction = (contour_level - z_start) / (z_end - z_start)
                crossings.append((x_start + fraction * (x_end - x_start), y_start + fraction * (y_end - y_start)))

            unique_crossings = []
            for crossing in crossings:
                if crossing not in unique_crossings:
                    unique_crossings.append(crossing)

            for segment_index in range(0, len(unique_crossings) - 1, 2):
                x_first, y_first = unique_crossings[segment_index]
                x_second, y_second = unique_crossings[segment_index + 1]
                midpoint_x = 0.5 * (x_first + x_second)
                midpoint_y = 0.5 * (y_first + y_second)
                if not inside_envelope_only or _point_inside_polygon(midpoint_x, midpoint_y, polygon_x, polygon_y):
                    segment_x.extend([x_first, x_second, None])
                    segment_y.extend([y_first, y_second, None])

    return segment_x, segment_y


def _automatic_inside_envelope_levels(phase_envelope, solubility_grid, solubility_values, max_levels=3):
    envelope_mask = _phase_envelope_grid_mask(
        phase_envelope,
        solubility_grid["temperatures"],
        solubility_grid["pressures"],
    )
    if envelope_mask is None or not envelope_mask.any():
        return []

    inside_values = solubility_values[envelope_mask & np.isfinite(solubility_values) & (solubility_values > 0.0)]
    if inside_values.size == 0:
        return []

    min_inside = float(np.nanmin(inside_values))
    max_inside = float(np.nanmax(inside_values))
    if not math.isfinite(min_inside) or not math.isfinite(max_inside):
        return []
    if math.isclose(min_inside, max_inside):
        return [min_inside]
    return np.geomspace(min_inside, max_inside, int(max_levels)).tolist()


def _plot_phase_envelope_with_sulfur(phase_envelope, solubility_grid, contour_levels, display_range=None):
    fig = go.Figure()

    if not phase_envelope["dew"].empty:
        fig.add_trace(
            go.Scatter(
                x=phase_envelope["dew"]["Temperature [degC]"],
                y=phase_envelope["dew"]["Pressure [bara]"],
                mode="lines+markers",
                name="Hydrocarbon dew/envelope",
                line=dict(color="#005B82", width=3),
                marker=dict(size=5),
                hovertemplate="T: %{x:.1f} degC<br>P: %{y:.2f} bara<extra></extra>",
            )
        )

    if not phase_envelope["bubble"].empty:
        fig.add_trace(
            go.Scatter(
                x=phase_envelope["bubble"]["Temperature [degC]"],
                y=phase_envelope["bubble"]["Pressure [bara]"],
                mode="lines+markers",
                name="Hydrocarbon bubble line",
                line=dict(color="#58A4B0", width=3, dash="dash"),
                marker=dict(size=5),
                hovertemplate="T: %{x:.1f} degC<br>P: %{y:.2f} bara<extra></extra>",
            )
        )

    solubility_values = np.asarray(solubility_grid["solubility"], dtype=float)
    log_solubility = np.where(
        np.isfinite(solubility_values) & (solubility_values > 0.0),
        np.log10(solubility_values),
        math.nan,
    )

    if np.isfinite(log_solubility).any():
        contour_palette = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02", "#A6761D"]
        finite_solubility = solubility_values[np.isfinite(solubility_values) & (solubility_values > 0.0)]
        min_solubility = float(np.nanmin(finite_solubility))
        max_solubility = float(np.nanmax(finite_solubility))
        requested_levels = contour_levels
        if not requested_levels:
            requested_levels = np.geomspace(min_solubility, max_solubility, 7).tolist()
        visible_levels = [level for level in requested_levels if min_solubility <= level <= max_solubility]
        inside_contour_count = 0
        for index, level in enumerate(visible_levels):
            contour_color = contour_palette[index % len(contour_palette)]
            full_segments = _contour_segments_inside_envelope(
                phase_envelope,
                solubility_grid["temperatures"],
                solubility_grid["pressures"],
                log_solubility,
                math.log10(level),
                inside_envelope_only=False,
            )
            if full_segments is not None and full_segments[0]:
                fig.add_trace(
                    go.Scatter(
                        x=full_segments[0],
                        y=full_segments[1],
                        mode="lines",
                        name=f"S8 {level:g} mg/Sm3",
                        line=dict(width=2, color=contour_color),
                        hovertemplate=(
                            f"S8 contour: {level:g} mg/Sm3"
                            "<br>T: %{x:.1f} degC<br>P: %{y:.2f} bara<extra></extra>"
                        ),
                    )
                )
            clipped_segments = _contour_segments_inside_envelope(
                phase_envelope,
                solubility_grid["temperatures"],
                solubility_grid["pressures"],
                log_solubility,
                math.log10(level),
            )
            if clipped_segments is not None and clipped_segments[0]:
                inside_contour_count += 1
                fig.add_trace(
                    go.Scatter(
                        x=clipped_segments[0],
                        y=clipped_segments[1],
                        mode="lines",
                        name=f"S8 {level:g} mg/Sm3 inside envelope",
                        line=dict(width=4, color=contour_color, dash="dash"),
                        hovertemplate=(
                            f"S8 contour: {level:g} mg/Sm3 inside envelope"
                            "<br>T: %{x:.1f} degC<br>P: %{y:.2f} bara<extra></extra>"
                        ),
                    )
                )

        if inside_contour_count == 0:
            automatic_levels = _automatic_inside_envelope_levels(phase_envelope, solubility_grid, solubility_values)
            for auto_index, level in enumerate(automatic_levels):
                if any(math.isclose(level, existing_level, rel_tol=0.05) for existing_level in visible_levels):
                    continue
                clipped_segments = _contour_segments_inside_envelope(
                    phase_envelope,
                    solubility_grid["temperatures"],
                    solubility_grid["pressures"],
                    log_solubility,
                    math.log10(level),
                )
                if clipped_segments is None or not clipped_segments[0]:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=clipped_segments[0],
                        y=clipped_segments[1],
                        mode="lines",
                        name=f"S8 {level:.2g} mg/Sm3 inside envelope",
                        line=dict(
                            width=3,
                            dash="dot",
                            color=contour_palette[(len(visible_levels) + auto_index) % len(contour_palette)],
                        ),
                        hovertemplate=(
                            f"S8 contour: {level:.2g} mg/Sm3 inside envelope"
                            "<br>T: %{x:.1f} degC<br>P: %{y:.2f} bara<extra></extra>"
                        ),
                    )
                )
    fig.update_layout(
        title="Hydrocarbon phase envelope with S8 solubility contours",
        xaxis_title="Temperature [degC]",
        yaxis_title="Pressure [bara]",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=40, r=40, t=80, b=40),
    )
    if display_range is not None:
        fig.update_xaxes(range=[display_range["temperature_min"], display_range["temperature_max"]])
        fig.update_yaxes(range=[display_range["pressure_min"], display_range["pressure_max"]])
    return fig


with st.sidebar:
    st.header("Calculation basis")
    temperature_c = st.number_input("Point temperature [degC]", value=10.0, step=1.0)
    pressure_bara = st.number_input("Point pressure [bara]", value=70.0, min_value=0.1, step=5.0)
    s8_ppmv = st.number_input("S8 feed concentration [ppmv]", value=10.0, min_value=0.0, step=1.0)
    use_s8_input = st.checkbox("Use S8 ppm input", value=True)
    add_stoichiometric_s8 = st.checkbox(
        "Add S8 from H2S + O2 screening stoichiometry",
        value=False,
        help="Adds min(H2S/8, O2/4) as S8 from the direct oxidation screening reaction 8 H2S + 4 O2 -> S8 + 8 H2O.",
    )
    is_plus_fraction = st.checkbox(
        "Characterize final oil pseudo-component as plus fraction",
        value=False,
        help="Use this for real oil/condensate fluids with a final heavy row such as C7, C20, or C30. S8 is kept out of the plus-fraction slot automatically.",
    )

    st.divider()
    st.header("Reaction estimate")
    reaction_name = st.selectbox("Sulfur reaction", options=list(SULFUR_REACTIONS.keys()), index=0)
    reaction_conversion_percent = st.slider(
        "Conversion of limiting reactant [%]",
        min_value=0.0,
        max_value=100.0,
        value=100.0,
        step=1.0,
    )

    st.divider()
    st.header("Sweep ranges")
    temp_start = st.number_input("Temperature sweep start [degC]", value=-20.0, step=5.0)
    temp_end = st.number_input("Temperature sweep end [degC]", value=120.0, step=5.0)
    temp_step = st.number_input("Temperature sweep step [degC]", value=5.0, min_value=0.5, step=1.0)
    pressure_start = st.number_input("Pressure sweep start [bara]", value=10.0, min_value=0.1, step=5.0)
    pressure_end = st.number_input("Pressure sweep end [bara]", value=150.0, min_value=0.1, step=5.0)
    pressure_step = st.number_input("Pressure sweep step [bara]", value=5.0, min_value=0.5, step=1.0)

    st.divider()
    st.header("Envelope map")
    phase_model_label = st.selectbox(
        "Envelope and S8 grid model",
        options=list(PHASE_ENVELOPE_MODEL_OPTIONS.keys()),
        index=0,
        help="Use SRK or PR for both the hydrocarbon phase-envelope calculation and the S8 solid-flash solubility grid. S8, water, oxygen, and SO2 are excluded from the envelope feed only.",
    )
    phase_model_name = PHASE_ENVELOPE_MODEL_OPTIONS[phase_model_label]
    map_temp_start = st.number_input("Map temperature start [degC]", value=-40.0, step=5.0)
    map_temp_end = st.number_input("Map temperature end [degC]", value=140.0, step=5.0)
    map_pressure_start = st.number_input("Map pressure start [bara]", value=1.0, min_value=0.1, step=5.0)
    map_pressure_end = st.number_input("Map pressure end [bara]", value=160.0, min_value=0.1, step=5.0)
    map_temperature_points = st.number_input("Temperature grid points", value=9, min_value=5, max_value=60, step=2)
    map_pressure_points = st.number_input("Pressure grid points", value=9, min_value=5, max_value=60, step=2)
    solubility_temp_start = st.number_input("S8 contour temperature start [degC]", value=-40.0, step=5.0)
    solubility_temp_end = st.number_input("S8 contour temperature end [degC]", value=140.0, step=5.0)
    extend_solubility_to_envelope_min = st.checkbox(
        "Extend S8 grid to lowest envelope temperature",
        value=False,
        help="Overrides the S8 contour start temperature when you need sulfur solubility far below the displayed operating range. Very low envelope temperatures can make sparse contour lines hard to read.",
    )
    solubility_saturation_ppmv = st.number_input(
        "S8 inventory for solubility grid [ppmv]",
        value=1000.0,
        min_value=1.0,
        step=100.0,
        help="A high S8 loading is used to keep the solid phase present, so total dissolved S8 in gas plus hydrocarbon liquid is interpreted as the solubility/carrying capacity.",
    )
    contour_level_text = st.text_input(
        "S8 contour levels [mg/Sm3]",
        value="0.1, 0.5, 1, 5, 10, 50, 100",
    )

    st.file_uploader(
        "Import composition CSV",
        key="sulfur_uploaded_file",
        help="Use the same columns as the composition table. Include S8 to model supplied elemental sulfur.",
    )


with st.expander("Fluid composition", expanded=True):
    preset_cols = st.columns(2)
    with preset_cols[0]:
        if st.button("Load default sour gas", use_container_width=True):
            st.session_state.sulfur_fluid_df = pd.DataFrame(default_sulfur_fluid)
            st.rerun()
    with preset_cols[1]:
        if st.button("Load oil with pseudo-components", use_container_width=True):
            st.session_state.sulfur_fluid_df = pd.DataFrame(default_oil_sulfur_fluid)
            st.rerun()

    hide_components = st.checkbox("Show active components only")
    if hide_components and "sulfur_edited_df" in st.session_state:
        st.session_state.sulfur_fluid_df = st.session_state.sulfur_edited_df[
            st.session_state.sulfur_edited_df["MolarComposition[-]"] > 0.0
        ]

    if (
        "sulfur_uploaded_file" in st.session_state
        and st.session_state.sulfur_uploaded_file is not None
        and not hide_components
    ):
        try:
            st.session_state.sulfur_fluid_df = _ensure_composition_columns(
                pd.read_csv(st.session_state.sulfur_uploaded_file)
            )
            numeric_columns = ["MolarComposition[-]", "MolarMass[kg/mol]", "RelativeDensity[-]"]
            st.session_state.sulfur_fluid_df[numeric_columns] = st.session_state.sulfur_fluid_df[
                numeric_columns
            ].astype(float)
        except Exception as error:
            st.warning(f"Could not load file: {error}")
            st.session_state.sulfur_fluid_df = pd.DataFrame(default_sulfur_fluid)

    if "sulfur_fluid_df" not in st.session_state:
        st.session_state.sulfur_fluid_df = pd.DataFrame(default_sulfur_fluid)

    edited_df = st.data_editor(
        st.session_state.sulfur_fluid_df,
        column_config={
            "ComponentName": st.column_config.TextColumn("Component"),
            "MolarComposition[-]": st.column_config.NumberColumn(
                "Molar composition [-]", min_value=0.0, max_value=10000.0, format="%.10f"
            ),
            "MolarMass[kg/mol]": st.column_config.NumberColumn(
                "Molar mass [kg/mol]", min_value=0.0, max_value=10000.0, format="%.6f"
            ),
            "RelativeDensity[-]": st.column_config.NumberColumn(
                "Relative density [-]", min_value=0.0, max_value=10.0, format="%.6f"
            ),
        },
        num_rows="dynamic",
        use_container_width=True,
    )
    st.session_state.sulfur_edited_df = edited_df
    st.caption(
        "Use `oxygen` for O2 and `S8` for elemental sulfur. For oil assays, add TBP pseudo-components such as `C7`, `C10`, `C20` with molar mass and relative density. Check plus-fraction characterization when the final heavy row is a heavy fraction such as `C7`, `C20`, or `C30`."
    )
    with st.expander("Oil and pseudo-component input", expanded=False):
        st.markdown(
            """
- Defined components such as methane, CO2, H2S, C2-C6 can be entered by name.
- Rows with both `MolarMass[kg/mol]` and `RelativeDensity[-]` are treated as TBP/pseudo-components by NeqSim.
- If plus-fraction characterization is enabled, the last eligible pseudo-component row is moved to the end before fluid creation and treated as the plus fraction.
- S8, oxygen, SO2, and water are never treated as the plus fraction, even if S8 is added from the sidebar input.

Typical oil rows: `C7` 0.0913 kg/mol / 0.746, `C10` 0.142 kg/mol / 0.800, `C20` 0.275 kg/mol / 0.860, final plus-fraction row `C30` 0.420 kg/mol / 0.910.
"""
        )


prepared_df = _prepare_composition(edited_df, s8_ppmv, use_s8_input, add_stoichiometric_s8)

if prepared_df.empty or prepared_df["MolarComposition[-]"].sum() <= 0.0:
    st.error("The fluid composition must contain at least one component with a positive molar composition.")
    st.stop()

with st.expander("Prepared NeqSim feed", expanded=False):
    st.dataframe(prepared_df, use_container_width=True)


run_point, run_temperature, run_pressure, run_reaction = st.columns(4)
with run_point:
    point_clicked = st.button("Run point check", use_container_width=True)
with run_temperature:
    temp_clicked = st.button("Run temperature sweep", use_container_width=True)
with run_pressure:
    pressure_clicked = st.button("Run pressure sweep", use_container_width=True)
with run_reaction:
    reaction_clicked = st.button("Run reaction estimate", use_container_width=True)
run_map_col, _ = st.columns([1, 2])
with run_map_col:
    map_clicked = st.button("Run phase envelope map", use_container_width=True)

tab_point, tab_temp, tab_pressure, tab_reaction, tab_map, tab_notes = st.tabs(
    [
        "Point check",
        "Temperature sweep",
        "Pressure sweep",
        "Reaction estimate",
        "Phase envelope map",
        "Engineering notes",
    ]
)

with tab_point:
    if point_clicked:
        with st.spinner("Running S8 solid flash at the selected operating point..."):
            try:
                result = _solid_flash_result(prepared_df, temperature_c, pressure_bara, is_plus_fraction)
                st.session_state.sulfur_point_result = result
            except Exception as error:
                st.error(f"Point calculation failed: {error}")

    if "sulfur_point_result" in st.session_state:
        result = st.session_state.sulfur_point_result
        metric_cols = st.columns(4)
        metric_cols[0].metric("Solid sulfur", "Yes" if result["Solid sulfur present"] else "No")
        metric_cols[1].metric("Dissolved S8", f"{result['Dissolved S8 [mg/Sm3]']:.3g} mg/Sm3")
        metric_cols[2].metric("Solid sulfur", f"{result['Solid sulfur [wt%]']:.3g} wt%")
        metric_cols[3].metric(
            "S8 in condensate/oil",
            f"{result['S8 in condensate/oil [mg/Sm3 feed]']:.3g} mg/Sm3",
        )

        if result["Solid sulfur present"]:
            st.warning(
                "Solid S8 is predicted at the selected condition. The dissolved S8 value includes gas plus oil/condensate uptake; the gas-only value is reported separately in the table."
            )
        else:
            st.success(
                "No solid S8 is predicted for the supplied S8 loading at this condition. The dissolved S8 value includes gas plus oil/condensate uptake, so it can increase when condensate forms."
            )

        st.caption(
            "Interpretation: compare `Supplied S8 [mg/Sm3]` with `Dissolved S8 [mg/Sm3]`. If solid sulfur is present, the dissolved value is the equilibrium carrying capacity and the excess inventory is in the solid phase."
        )

        st.dataframe(pd.DataFrame([result]).T.rename(columns={0: "Value"}), use_container_width=True)
    else:
        st.info("Run the point check to evaluate whether the selected gas/oil mixture is saturated or precipitating S8.")

with tab_temp:
    if temp_clicked:
        if temp_end < temp_start:
            st.error("Temperature sweep end must be greater than or equal to the start temperature.")
        else:
            with st.spinner("Running temperature sweep with S8 solid flash..."):
                try:
                    temp_results = _run_temperature_sweep(
                        prepared_df,
                        pressure_bara,
                        temp_start,
                        temp_end,
                        temp_step,
                        is_plus_fraction,
                    )
                    st.session_state.sulfur_temp_results = temp_results
                except Exception as error:
                    st.error(f"Temperature sweep failed: {error}")

    if "sulfur_temp_results" in st.session_state:
        temp_results = st.session_state.sulfur_temp_results
        onset_temperature = _cooling_onset_temperature(temp_results)
        metric_cols = st.columns(3)
        metric_cols[0].metric(
            "Cooling onset temperature",
            "Not found" if math.isnan(onset_temperature) else f"{onset_temperature:.2f} degC",
        )
        metric_cols[1].metric("Solid points", int(temp_results["Solid sulfur present"].sum()))
        metric_cols[2].metric(
            "Max solid sulfur",
            f"{temp_results['Solid sulfur [wt%]'].max():.3g} wt%",
        )
        st.caption(
            "The plotted dissolved S8 curve is gas plus hydrocarbon liquid. A growing condensate/oil contribution can raise total sulfur capacity even if gas-only S8 falls."
        )
        st.plotly_chart(
            _plot_sweep(temp_results, "Temperature [degC]", "S8 solubility and precipitation vs temperature"),
            use_container_width=True,
        )
        st.dataframe(temp_results, use_container_width=True)
    else:
        st.info("Run the temperature sweep to find the cooling temperature where S8 precipitation begins.")

with tab_pressure:
    if pressure_clicked:
        if pressure_end < pressure_start:
            st.error("Pressure sweep end must be greater than or equal to the start pressure.")
        else:
            with st.spinner("Running pressure sweep with S8 solid flash..."):
                try:
                    pressure_results = _run_pressure_sweep(
                        prepared_df,
                        temperature_c,
                        pressure_start,
                        pressure_end,
                        pressure_step,
                        is_plus_fraction,
                    )
                    st.session_state.sulfur_pressure_results = pressure_results
                except Exception as error:
                    st.error(f"Pressure sweep failed: {error}")

    if "sulfur_pressure_results" in st.session_state:
        pressure_results = st.session_state.sulfur_pressure_results
        metric_cols = st.columns(3)
        metric_cols[0].metric("Solid points", int(pressure_results["Solid sulfur present"].sum()))
        metric_cols[1].metric(
            "Max dissolved S8",
            f"{pressure_results['Dissolved S8 [mg/Sm3]'].max():.3g} mg/Sm3",
        )
        metric_cols[2].metric(
            "Max solid sulfur",
            f"{pressure_results['Solid sulfur [wt%]'].max():.3g} wt%",
        )
        st.caption(
            "Use the gas-only and condensate/oil columns in the table to identify whether pressure changes are moving S8 into liquid hydrocarbon or into solid sulfur."
        )
        st.plotly_chart(
            _plot_sweep(pressure_results, "Pressure [bara]", "S8 solubility and precipitation vs pressure"),
            use_container_width=True,
        )
        st.dataframe(pressure_results, use_container_width=True)
    else:
        st.info("Run the pressure sweep to screen pressure-reduction or recompression scenarios.")

with tab_reaction:
    if reaction_clicked:
        with st.spinner("Running sulfur reaction material balance..."):
            try:
                reaction_feed_df = _prepare_composition(edited_df, s8_ppmv, use_s8_input, False)
                reaction_result = _calculate_reaction_final_composition(
                    reaction_feed_df,
                    reaction_name,
                    reaction_conversion_percent,
                )
                final_flash_feed = reaction_result["final_composition"].drop(
                    columns=["Final mole fraction [-]"], errors="ignore"
                )
                if not final_flash_feed.empty:
                    reaction_result["solid_flash"] = _solid_flash_result(
                        final_flash_feed,
                        temperature_c,
                        pressure_bara,
                        is_plus_fraction,
                    )
                st.session_state.sulfur_reaction_result = reaction_result
            except Exception as error:
                st.error(f"Reaction estimate failed: {error}")

    if "sulfur_reaction_result" in st.session_state:
        reaction_result = st.session_state.sulfur_reaction_result
        metric_cols = st.columns(4)
        metric_cols[0].metric("Reaction", reaction_result["equation"])
        metric_cols[1].metric("Limiting reactant", str(reaction_result.get("limiting_component", "None")))
        metric_cols[2].metric("Reaction extent", f"{reaction_result.get('extent', 0.0):.4g} rel mol")
        metric_cols[3].metric("S8 formed", f"{reaction_result.get('s8_formed_mg_sm3', 0.0):.3g} mg/Sm3")

        if reaction_result.get("extent", 0.0) <= 0.0:
            st.warning("No reaction extent was calculated. Check that the feed contains both required reactants.")

        if "solid_flash" in reaction_result:
            flash_result = reaction_result["solid_flash"]
            flash_cols = st.columns(3)
            flash_cols[0].metric("Solid sulfur at point", "Yes" if flash_result["Solid sulfur present"] else "No")
            flash_cols[1].metric("Dissolved S8 after reaction", f"{flash_result['Dissolved S8 [mg/Sm3]']:.3g} mg/Sm3")
            flash_cols[2].metric("Solid sulfur after reaction", f"{flash_result['Solid sulfur [wt%]']:.3g} wt%")
            st.caption(
                "The reaction estimate is stoichiometric only. The follow-up solid flash partitions the produced S8 between gas, oil/condensate, and solid sulfur at the selected point condition."
            )

        st.markdown("**Reaction material balance**")
        st.dataframe(reaction_result["reaction_summary"], use_container_width=True)
        st.markdown("**Calculated final composition**")
        st.dataframe(reaction_result["final_composition"], use_container_width=True)

        if st.button("Use calculated final composition as feed", use_container_width=True):
            st.session_state.sulfur_fluid_df = reaction_result["final_composition"].drop(
                columns=["Final mole fraction [-]"], errors="ignore"
            )
            st.rerun()
    else:
        st.info(
            "Run the reaction estimate to convert the input gas by stoichiometry and calculate the final composition before solubility checks."
        )

with tab_map:
    if map_clicked:
        if map_temp_end < map_temp_start:
            st.error("Map temperature end must be greater than or equal to the start temperature.")
        elif map_pressure_end < map_pressure_start:
            st.error("Map pressure end must be greater than or equal to the start pressure.")
        elif solubility_temp_end < solubility_temp_start:
            st.error("S8 contour temperature end must be greater than or equal to the start temperature.")
        else:
            with st.spinner("Calculating hydrocarbon phase envelope and S8 solubility contours..."):
                try:
                    envelope_df = _phase_envelope_composition(prepared_df)
                    phase_envelope = _calculate_phase_envelope(
                        envelope_df,
                        is_plus_fraction,
                        phase_model_name,
                    )
                    envelope_min_temperature = _phase_envelope_min_temperature(phase_envelope)
                    envelope_temperature_min, envelope_temperature_max = _phase_envelope_range(
                        phase_envelope, "Temperature [degC]"
                    )
                    envelope_pressure_min, envelope_pressure_max = _phase_envelope_range(
                        phase_envelope, "Pressure [bara]"
                    )
                    display_temperature_min = map_temp_start
                    display_temperature_max = map_temp_end
                    display_pressure_min = map_pressure_start
                    display_pressure_max = map_pressure_end
                    if math.isfinite(envelope_temperature_min):
                        display_temperature_min = min(display_temperature_min, envelope_temperature_min - 5.0)
                    if math.isfinite(envelope_temperature_max):
                        display_temperature_max = max(display_temperature_max, envelope_temperature_max + 5.0)
                    if math.isfinite(envelope_pressure_min):
                        display_pressure_min = min(display_pressure_min, max(0.0, envelope_pressure_min * 0.9))
                    if math.isfinite(envelope_pressure_max):
                        display_pressure_max = max(display_pressure_max, envelope_pressure_max * 1.1)
                    solubility_temperature_start = solubility_temp_start
                    if extend_solubility_to_envelope_min and not math.isnan(envelope_min_temperature):
                        solubility_temperature_start = min(solubility_temp_start, envelope_min_temperature)
                    solubility_grid = _run_solubility_grid(
                        prepared_df,
                        solubility_temperature_start,
                        solubility_temp_end,
                        map_temperature_points,
                        map_pressure_start,
                        map_pressure_end,
                        map_pressure_points,
                        is_plus_fraction,
                        solubility_saturation_ppmv,
                        phase_model_name,
                    )
                    st.session_state.sulfur_phase_map = {
                        "requested_model_label": phase_model_label,
                        "requested_model_name": phase_model_name,
                        "phase_envelope": phase_envelope,
                        "solubility_grid": solubility_grid,
                        "contour_levels": _parse_contour_levels(contour_level_text),
                        "envelope_composition": envelope_df,
                        "envelope_min_temperature": envelope_min_temperature,
                        "envelope_temperature_min": envelope_temperature_min,
                        "envelope_temperature_max": envelope_temperature_max,
                        "display_range": {
                            "temperature_min": display_temperature_min,
                            "temperature_max": display_temperature_max,
                            "pressure_min": display_pressure_min,
                            "pressure_max": display_pressure_max,
                        },
                        "solubility_temperature_start": solubility_temperature_start,
                        "solubility_temperature_end": solubility_temp_end,
                    }
                except Exception as error:
                    st.error(f"Phase envelope map failed: {error}")

    if "sulfur_phase_map" in st.session_state:
        phase_map = st.session_state.sulfur_phase_map
        map_matches_current_model = phase_map.get("requested_model_name") == phase_model_name
        if not map_matches_current_model:
            st.warning(
                "The displayed controls now select "
                f"{phase_model_label}, but the stored map was calculated with "
                f"{phase_map.get('requested_model_label', phase_map.get('requested_model_name', 'Unknown'))}. "
                "Press `Run phase envelope map` to recalculate with the selected model."
            )
            st.stop()
        solubility_table = phase_map["solubility_grid"]["table"]
        metric_cols = st.columns(4)
        metric_cols[0].metric(
            "Envelope points",
            len(phase_map["phase_envelope"]["dew"]) + len(phase_map["phase_envelope"]["bubble"]),
        )
        metric_cols[1].metric(
            "Min dissolved S8 capacity",
            f"{solubility_table['S8 solubility [mg/Sm3]'].min():.3g} mg/Sm3",
        )
        metric_cols[2].metric(
            "Max dissolved S8 capacity",
            f"{solubility_table['S8 solubility [mg/Sm3]'].max():.3g} mg/Sm3",
        )
        metric_cols[3].metric(
            "Solid grid points",
            int(solubility_table["Solid sulfur present"].sum()),
        )
        map_info_cols = st.columns(3)
        map_info_cols[0].metric(
            "Lowest envelope temperature",
            "Not available"
            if math.isnan(phase_map.get("envelope_min_temperature", math.nan))
            else f"{phase_map['envelope_min_temperature']:.2f} degC",
        )
        map_info_cols[1].metric(
            "S8 contour starts at",
            f"{phase_map.get('solubility_temperature_start', solubility_temp_start):.2f} degC",
        )
        map_info_cols[2].metric(
            "S8 contour ends at",
            f"{phase_map.get('solubility_temperature_end', solubility_temp_end):.2f} degC",
        )
        display_range = phase_map.get("display_range")
        if display_range is not None and (
            display_range["temperature_min"] < map_temp_start
            or display_range["temperature_max"] > map_temp_end
            or display_range["pressure_min"] < map_pressure_start
            or display_range["pressure_max"] > map_pressure_end
        ):
            st.info(
                "The calculated hydrocarbon phase envelope extends outside the requested map window, so the plot axes were expanded automatically to include the envelope."
            )
        envelope_meta_cols = st.columns(5)
        envelope_meta_cols[0].metric("Selected model", phase_map.get("requested_model_label", "Unknown"))
        envelope_meta_cols[1].metric("Envelope actual model", phase_map["phase_envelope"].get("model", "Unknown"))
        envelope_meta_cols[2].metric(
            "Envelope method",
            phase_map["phase_envelope"].get("method_used", "Unknown"),
        )
        envelope_meta_cols[3].metric(
            "Plus fraction in envelope",
            "Yes" if phase_map["phase_envelope"].get("plus_fraction_used", False) else "No",
        )
        envelope_meta_cols[4].metric(
            "Envelope fallback",
            "Yes" if phase_map["phase_envelope"].get("fallback_used", False) else "No",
        )
        st.caption(
            f"S8 solubility grid model: {phase_map['solubility_grid'].get('model', 'Unknown')}. The envelope calculation excludes S8/reactive species, while the S8 grid includes S8 and runs the solid-flash calculation."
        )
        if phase_map["phase_envelope"].get("attempt_errors"):
            with st.expander("Phase envelope fallback attempts", expanded=False):
                st.dataframe(pd.DataFrame(phase_map["phase_envelope"]["attempt_errors"]), use_container_width=True)
        if phase_map["solubility_grid"].get("errors"):
            st.warning(
                f"{len(phase_map['solubility_grid']['errors'])} S8 solubility grid point(s) failed and were plotted as gaps."
            )
            with st.expander("Failed S8 solubility grid points", expanded=False):
                st.dataframe(pd.DataFrame(phase_map["solubility_grid"]["errors"]), use_container_width=True)
        st.caption(
            "Contour basis: total dissolved S8 in gas plus hydrocarbon liquid/condensate. Full iso-lines are shown across the calculated map range; dashed overlays highlight the portions inside the phase envelope."
        )
        st.plotly_chart(
            _plot_phase_envelope_with_sulfur(
                phase_map["phase_envelope"],
                phase_map["solubility_grid"],
                phase_map["contour_levels"],
                phase_map.get("display_range"),
            ),
            use_container_width=True,
        )
        with st.expander("Envelope composition used", expanded=False):
            st.dataframe(phase_map["envelope_composition"], use_container_width=True)
        with st.expander("S8 solubility grid", expanded=False):
            st.dataframe(solubility_table, use_container_width=True)
    else:
        st.info(
            "Run the phase envelope map to overlay the hydrocarbon phase boundary with constant S8 solubility contours."
        )

with tab_notes:
    st.markdown(
        """
**How to use the results**

- A solid sulfur flag means the supplied S8 loading exceeds the equilibrium dissolved capacity at that temperature and pressure.
- Dissolved S8 is reported on a feed basis and includes gas plus hydrocarbon liquid/condensate. Use the gas-only and condensate/oil columns to see where the sulfur is held.
- In a cooling study, the highest temperature with solid sulfur present is the practical S8 precipitation onset during cooldown for the supplied S8 inventory.
- Pressure-reduction equipment, cold separators, heat exchangers, low-velocity pipe sections, and metering stations are typical locations where the margin can disappear.
- The optional H2S + O2 stoichiometric screening is a conservative source-term estimate, not a kinetic or catalyst model.
- The phase envelope map removes S8, oxygen, SO2, and water before calculating the hydrocarbon dew/bubble boundary, then overlays total dissolved S8 contours from separate solid-flash grid calculations.
- Iso-lines inside the phase envelope can be physically meaningful: condensate or oil may dissolve additional S8, so total dissolved S8 can increase even when gas-only S8 decreases.

**Model basis**

- Thermodynamic model: SRK fluid from `fluid_df`, Huron-Vidal mixing rule `2`.
- Solid model: NeqSim S8 solid phase check with `TPSolidflash`.
- Hydrocarbon phase envelope: selected phase-envelope EOS in the sidebar, excluding reactive/solid species from the envelope feed.
- S8 concentration conversion: S8 mole fraction on a feed or gas-phase basis to mg/Sm3 at 15 degC and 1 atm.
"""
    )