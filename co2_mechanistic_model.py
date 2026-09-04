"""
NEQSIM CO2 IMPURITY KINETIC MODEL & MULTI-PHASE EXPERIMENT ENGINE
====================================================================================================
Illustrative rate-law engine for trace-impurity chemistry in CO2 streams. Uses the NeqSim Java
SRK EOS when the Python package is available, falling back to a documented screening correlation
otherwise. Includes a lumped wall-corrosion model and a CSTR-style multi-phase experiment runner.

Reaction set (deliberately kept small -- see REACTION_NAMES for the exact stoichiometry):
R1 (SO2+O2+H2O->H2SO4, extremely slow), R2 (H2S+NO2->SO2+NO), R3a (SO2+NO2+H2O->NO+H2SO4, slow),
R4 (2NO+O2<->2NO2, fast), R5 (3NO2+H2O<->2HNO3+NO, very fast), R7 (H2S+NO->NH3+SO2, fast),
R10 (NH3+NO+O2->N2O), R11 (H2S+NO->N2O+S8), R12 (H2S+O2->H2SO4, NO2-catalysed),
R13 (NO2+H2S->H2SO4+NO), plus 3 wall-corrosion paths (HNO3->Fe(NO3)2, H2SO4->FeSO4, O2->Fe2O3).

The default kinetic parameters are uncalibrated and must not be used as qualified design data.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

try:
    from scipy.integrate import solve_ivp
except ImportError as scipy_import_error:
    solve_ivp = None
    SCIPY_IMPORT_ERROR = scipy_import_error
else:
    SCIPY_IMPORT_ERROR = None


def _trapz(y, x):
    """Trapezoidal integration, tolerant of the numpy>=2.0 np.trapz->np.trapezoid rename."""
    integrator = getattr(np, 'trapezoid', None) or np.trapz
    return integrator(y, x)


def _cumulative_trapz(y, x):
    """Cumulative trapezoidal integral of ``y`` over ``x`` (same length as ``y``/``x``, first
    element 0.0). Scipy-free by design, matching this module's optional-scipy fallback."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    increments = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    return np.concatenate([[0.0], np.cumsum(increments)])



# ==================================================================================================
# FUNDAMENTAL PHYSICAL AND THERMODYNAMIC CONSTANTS
# ==================================================================================================
R_GAS = 8.314462618             # Universal Gas Constant [J / (mol * K)]
MW_CO2 = 44.0095                # Molar Mass of CO2 [g / mol]
MW_N2 = 28.0134                 # Molar Mass of N2 [g / mol]
MW_H2O = 18.0153                # Molar Mass of H2O [g / mol]
MW_HNO3 = 63.0129               # Molar Mass of HNO3 [g / mol]
MW_H2SO4 = 98.079                # Molar Mass of H2SO4 [g / mol]
MW_NH3 = 17.0305                # Molar Mass of NH3 [g / mol]
P_CRIT_CO2_BAR = 73.8           # Critical Pressure of CO2 [bar]
T_CRIT_CO2_K = 304.13           # Critical Temperature of CO2 [K]


# ==================================================================================================
# ILLUSTRATIVE, UNCALIBRATED REACTION KINETIC PARAMETERS
# ==================================================================================================
DEFAULT_KINETIC_PARAMS = {
    'R1':  {'name': 'SO2 + 0.5 O2 + H2O <-> H2SO4',           'A': 5.0e5,     'Ea': 30000.0, 'units': 'm3 / (kmol * s)'},   # extremely slow
    'R2':  {'name': 'H2S + 3 NO2 <-> SO2 + H2O + 3 NO',       'A': 1.0e10,    'Ea': 30000.0, 'units': 'm3 / (kmol * s)'},
    'R3a': {'name': 'SO2 + NO2 + H2O <-> NO + H2SO4',         'A': 1.0e5,     'Ea': 35000.0, 'units': 'm3 / (kmol * s)'},   # slow
    'R4':  {'name': '2 NO + O2 <-> 2 NO2',                    'A': 500.0,     'Ea': -4400.0, 'units': 'm6 / (kmol2 * s)'},  # fast
    'R5':  {'name': '3 NO2 + H2O <-> 2 HNO3 + NO',            'A': 2.4e6,     'Ea': 28000.0, 'units': 'm3 / (kmol * s)'},   # very fast
    'R7':  {'name': '5 H2S + 6 NO + 4 H2O -> 6 NH3 + 5 SO2',  'A': 2.0e6,     'Ea': 12000.0, 'units': 'm3 / (kmol * s)'},   # fast
    'R10': {'name': '4 NH3 + 4 NO + 3 O2 -> 4 N2O + 6 H2O',    'A': 1.0e2,     'Ea': 20000.0, 'units': 'm6 / (kmol2 * s)'},
    'R11': {'name': 'H2S + 2 NO -> N2O + 1/8 S8 + H2O',        'A': 1.0e2,     'Ea': 20000.0, 'units': 'm3 / (kmol * s)'},
    'R12': {'name': 'H2S + 2 O2 <-> H2SO4 (NO2-catalysed)',    'A': 1.0e14,    'Ea': 35000.0, 'units': 'm6 / (kmol2 * s), phase-scaled'},
    'R13': {'name': '4 NO2 + H2S <-> H2SO4 + 4 NO',            'A': 1.0e13,    'Ea': 30000.0, 'units': 'm4.5 / (kmol4.5 * s), phase-scaled'},
    'R15': {'name': '4 NO2 <-> 2 N2O + 3 O2',                  'A': 1.0e2,     'Ea': 40000.0, 'units': 'm3 / (kmol * s)'},
}

# ==================================================================================================
# GENERAL REFERENCE CARBON-STEEL / WET-CO2 PARAMETER SET
# --------------------------------------------------------------------------------------------------
# A single generic set of reaction-kinetics and wall-corrosion constants for wet
# dense/gaseous CO2 streams containing H2S, SO2, NO2 and O2 in contact with carbon steel. These
# describe the chemistry and corrosion mechanism themselves; reactor geometry and feed composition
# are configured separately. Apply this set
# with CO2ImpurityKineticsModel.apply_calibrated_profile() / CO2ImpurityReactorExperiment(...,
# calibrated_profile='carbon_steel_wet_co2') / AutoclaveExperiment(...).
# The values are illustrative reference inputs, not published experimental results. Validate them
# for the intended operating envelope before using the model for engineering decisions.
# ==================================================================================================
CARBON_STEEL_WET_CO2_KINETICS = {
    'R2':  {'A': 1.8375e10 * 0.97, 'Ea_kJ_mol': 30.0},  # H2S + 3 NO2 -> SO2 + H2O + 3 NO
    'R3a': {'A': 1.1e11 * 0.15,   'Ea_kJ_mol': 30.0},   # SO2 + NO2 + H2O -> NO + H2SO4
    'R4':  {'A': 1.5e6,   'Ea_kJ_mol': -4.4},   # 2 NO + O2 -> 2 NO2
    'R5':  {'A': 2.4e6,   'Ea_kJ_mol': -5.0},   # 3 NO2 + H2O -> 2 HNO3 + NO
    'R7':  {'A': 3750.0,  'Ea_kJ_mol': -10.0},  # 5 H2S + 6 NO + 4 H2O -> 6 NH3 + 5 SO2
    'R10': {'A': 1.0e9,   'Ea_kJ_mol': 20.0},   # 4 NH3 + 4 NO + 3 O2 -> 4 N2O + 6 H2O
    'R11': {'A': 1.6e7,   'Ea_kJ_mol': 20.0},   # H2S + 2 NO -> N2O + 1/8 S8 + H2O
    'R12': {'A': 6.0e14,  'Ea_kJ_mol': 35.0},   # H2S + 2 O2 -> H2SO4 (NO2-catalysed, NO2 not consumed)
    'R13': {'A': 1.0e13,  'Ea_kJ_mol': 30.0},   # 4 NO2 + H2S -> H2SO4 + 4 NO
    'R15': {'A': 1.0e18,  'Ea_kJ_mol': 40.0},   # 4 NO2 -> 2 N2O + 3 O2
    'r15_f_phase_exponent': 1.0,
    'r15_o2_inhib_ref_ppm': 15.0,
    'r15_o2_inhib_hill_n': 2.0,
    # R15's O2 activation gate (see rhs()/__init__ docstring): 0.0 disables it (exact no-op)
    # until calibrated.
    'r15_o2_activation_ref_ppm': 0.0,
    'r15_o2_activation_hill_n': 2.0,
    'r15_no2_cap_ppm': 0.0,
    'r15_no2_cap_hill_n': 2.0,
    'r15_n2o_cap_ppm': 6.0,
    'r15_n2o_cap_hill_n': 20.0,
    # R15's O2-presence gate (see rhs()/__init__ docstring): 0.0 disables it (exact no-op)
    # until calibrated for the specific brief/extended zero-O2-feed windows it targets.
    'r15_o2_presence_ref_ppm': 0.0,
    'r15_o2_presence_hill_n': 1.0,
    'r11_o2_ref_ppm': 2.0,
    'r11_o2_hill_n': 2.0,
    'r11_o2_gain': 0.9,
    'r2_no2_boost_ref_ppm': 8.0,
    'r2_no2_boost_hill_n': 10.0,
    'r2_no2_boost_gain': 2.0,
    # LaggedO2 time constant (see rhs()/o2_lag_tau_hours docstring): 0.0 disables it (exact
    # no-op) until calibrated.
    'o2_lag_tau_hours': 0.0,
    'o2_feed_lag_tau_hours': 20.0,
    'o2_feed_lag_rise_tau_hours': 1.0,
    'r12_density_independent': False,
    'r12_no2_order': 1.0,
    # Order of R13's NO2 (forward) / NO (reverse) terms -- 4.0 is the shipped/validated default
    # (matches the literal stoichiometry). See r13_no2_order docstring on __init__.
    'r13_no2_order': 4.0,

    # R5's reverse-term NO activity (see r5_no_activity docstring on CO2ImpurityKineticsModel):
    # decoupled from the shared phi_dict['NO']=0.05 (a fix for R4's O2-recycling behaviour,
    # irrelevant to R5). Kept at the ideal-fugacity value of 1.0 so R5 reaches its own genuine
    # equilibrium instead of having its reverse term artificially starved.
    'r5_no_activity': 1.0,
    'r4_no_activity': 0.05,
    'r4_surface_gain': 14.0,
    'r3a_bore_gain': 2.0,
    'r15_surface_suppress_gain': 60.0,
    'r15_sulfur_ref_ppm': 0.05,
    'r15_sulfur_hill_n': 2.0,
    'wall_o2_feed_o2_ref_ppm': 15.0,
    'wall_o2_feed_o2_hill_n': 8.0,
    'wall_no2_feed_o2_ref_ppm': 15.0,
    'wall_no2_feed_o2_hill_n': 4.0,
    'r3a_feed_o2_ref_ppm': 15.0,
    'r3a_feed_o2_hill_n': 4.0,
    'r3a_feed_o2_floor': 0.0,
    'r3a_feed_o2_cap_ppm': 60.0,
    'wall_no2_o2_presence_ref_ppm': 1.0,
    'wall_no2_o2_presence_hill_n': 2.0,
    'r3a_o2_presence_ref_ppm': 0.0,
    'r3a_o2_presence_hill_n': 1.0,
    'r2_o2_presence_ref_ppm': 0.0,
    'r2_o2_presence_hill_n': 1.0,
    # wall_no2's NO product brake (see _wall_no2_rate docstring): 0.0 disables it (exact no-op)
    # until calibrated.
    'wall_no2_no_cap_ppm': 0.0,
    'wall_no2_no_cap_hill_n': 2.0,
    # wall_no2's NO2 Langmuir adsorption isotherm (see _wall_no2_rate docstring): 0.0 disables
    # it (exact no-op, falls back to the plain wall_no2_potency power law) until calibrated.
    'wall_no2_langmuir_half_ppm': 0.0,

    'r3a_no_escape_frac': 0.0,

    # R1 (SO2 + 0.5 O2 + H2O -> H2SO4) autocatalytic acceleration (see r1_autocat_gain docstring
    # on __init__): a saturating multiplier on BOTH directions of R1, driven by the cumulative
    # (never-decreasing) H2SO4 ever produced across R1/R3a/R12/R13, not the current standing H2SO4
    # ppm -- so wall-corrosion consumption of H2SO4 (Fe + H2SO4 -> FeSO4 + H2) cannot undo the
    # acceleration or feed back into Keq1's reverse term. gain=0.0 disables it (no autocatalysis).
    'r1_autocat_gain': 0.0,
    'r1_autocat_ref_ppm': 10.0,     # cumulative H2SO4 (ppm-equivalent) giving half the max boost

    'r3a_autocat_gain': 20.0,
    'r3a_autocat_ref_ppm': 40.0,
    'r3a_autocat_hill_n': 8.0,
    'r3a_autocat_surface_suppress_gain': 20.0,

    'condensation_exponent': 2.0,
    'rho_m_reference': 24.0,               # kmol/m^3

    'wall_k_intrinsic': 1.8e-7,             # O2 path intrinsic rate [mol O2 / (m^2 s ppm)]
    'wall_o2_potency': 1.0,                # linear in O2 ppm
    'wall_o2_sat_ref': 0.0,
    # wall_rho_pass/wall_hill_n (dense-CO2 passivation) are UNUSED by any active wall path.
    # wall_acid_gain/exponent/background drive _acid_enhancement (the O2 path's acid-history
    # gating, see above) -- background=0.0 (no floor: zero acid history genuinely means zero
    # enhancement, matching "ties O2 attack to acid formation").
    'wall_acid_gain': 0.06,
    'wall_acid_exponent': 1.0,
    'wall_acid_background': 0.0,
    'wall_acid_gain2': 0.0012,
    'wall_acid_exponent2': 4.0,
    'wall_rho_pass': 5.0,
    'wall_hill_n': 3.0,
    'wall_consume_h2o': False,
    # wall_h2o_* below govern ONLY the disabled FeCO3 path's _effective_g_h2o gate; the three
    # ACTIVE acid paths use the simpler _water_saturation_fraction instead (see above).
    'wall_h2o_mode': 'wet_film',
    'wall_h2o_enhancement_factor': 1.0,    # tunable non-ideality correction on the Antoine dew point
    'wall_h2o_deliq_ref_ppm': 30.0,        # acid ppm that halves the solubility limit (deliquescence)
    'wall_h2o_hill_n': 1.0,                # wetted-fraction Langmuir/Hill sharpness around the dew point
    'wall_h2o_excess_ref_ppm': 100.0,      # excess-water scale for the film-severity growth term
    'wall_h2o_excess_exponent': 1.0,       # excess-water growth exponent (severity keeps rising)
    'wall_feco3_k_intrinsic': 0.0,         # Disabled by default
    'wall_feco3_potency': 1.0,             # linear in dissolved-CO2 molarity (unused while disabled)
    'wall_hno3_corrosion_k_intrinsic': 1.0e-7,  # HNO3 wall-film path intrinsic rate [mol Fe / (m^2 s ppm)]
    'wall_hno3_corrosion_potency': 1.0,         # linear in HNO3 ppm
    'wall_h2so4_k_intrinsic': 6.0e-9,      # sulfuric-acid path intrinsic rate [mol Fe / (m^2 s ppm)]
    'wall_h2so4_potency': 1.0,             # linear in H2SO4 ppm
    'wall_no2_k_intrinsic': 5.0e-6,        # [mol NO2 / (m^2 s ppm)]
    'wall_no2_potency': 1.0,               # linear in NO2 ppm
    # O2-depletion Hill gate for wall_no2 (see _wall_no2_rate docstring): strong (gate~1) once O2
    # is nearly exhausted, weak (gate~0) while O2 stays abundant.
    'wall_no2_o2_ref_ppm': 2.0,
    'wall_no2_o2_hill_n': 4.0,
    'wall_o2_gas_phase_gain': 0.6,
    'wall_gas_phase_gain': 0.6,
    'wall_gas_phase_rho_ref': 5.0,
    'wall_gas_phase_hill_n': 2.0,
    'wall_s8_k_intrinsic': 1.0e-9,         # [mol H2S / (m^2 s ppm)]
    'wall_s8_h2s_potency': 1.0,            # linear in H2S ppm
    'wall_s8_o2_potency': 0.5,             # sub-linear in O2 ppm
    # Surface-catalysed SO2 oxidation (see _wall_so2_rate docstring): 0.0 (disabled) until
    # calibrated -- new mechanism, targets a genuinely time/exposure-dependent SO2 depletion
    # some experiments show late in a long, sustained-high-O2 run.
    'wall_so2_k_intrinsic': 0.0,
    'wall_so2_potency': 1.0,
    'wall_so2_exposure_threshold_ppm_h': 0.0,
    'wall_so2_exposure_hill_n': 2.0,
}

DG_SO2_STDGIBBS = -300.1e3
DG_O2_STDGIBBS = 0.0
DG_H2O_STDGIBBS = -237.1e3
DG_H2SO4_STDGIBBS = -690.1e3
DG_H2S_STDGIBBS = -33.4e3
DG_NO2_STDGIBBS = 51.3e3
DG_NO_STDGIBBS = 86.6e3
# Using aqueous molecular HNO3 (-79.9) rather than gaseous (-73.5) so that Keq5 is
# phase-consistent with H2O(l) already used above.
DG_HNO3_STDGIBBS = -79.9e3
DG_NH3_STDGIBBS = -16.4e3       # NIST standard Gibbs energy of formation, NH3(g), 298 K
DG_N2O_STDGIBBS = 104.2e3       # NIST standard Gibbs energy of formation, N2O(g), 298 K
DG_S8_STDGIBBS = 0.0            # S8(s, rhombic) is the reference state of the element
_R4_SURFACE_SV_REF_CM_INV = 4.0 / 6.5 + 2.0 / (330.0 / (np.pi * (6.5**2) / 4.0))

MAX_KEQ_EXPONENT = 300.0        # Exponential ceiling to prevent numerical overflow in Keq
MIN_CONCENTRATION_FLOOR = 1e-25  # Minimum concentration floor to prevent log underflow in ODEs
MOISTURE_REF_PPM = 50.0         # Reference moisture concentration scale for hydration factor [ppm]
_WALL_NO2_T_REF_K = 249.15


class CO2ImpurityKineticsModel:
    """
    Mechanistic simulator for impurity-reaction screening in CO2 streams.

    Uses NeqSim SRK thermodynamics when available. The fallback correlation and kinetic
    parameters are illustrative and require independent calibration and validation.
    """

    SPECIES = (
        'H2S', 'SO2', 'NO2', 'NO', 'O2', 'H2O',
        'H2SO4', 'HNO3', 'S8', 'NH3', 'N2O', 'H2'
    )

    # Extra ODE states appended after SPECIES (see rhs()/simulate()): two accumulated solid
    # wall-corrosion products (FeSO4, Fe(NO3)2), and three cumulative, NEVER-DECREASING "total
    # ever produced" trackers (CumH2SO4, CumHNO3, CumNH3) -- kept deliberately separate from the
    # actual gas-phase H2SO4/HNO3/NH3 concentration (which the wall reactions, reverse rates and
    # (for NH3) R10 DO consume/reduce), so that acid/base consumed by any of those does not erase
    # the reaction history that drives R1's autocatalytic acceleration (see set_r1_autocat
    # docstring), the autoclave-wash IC-analysis estimate (see get_autoclave_wash_table), or any
    # future use.
    EXTRA_STATE_KEYS = ('FeSO4', 'FeNO32', 'CumH2SO4', 'CumHNO3', 'CumNO2Exposure', 'LaggedO2',
                        'CumO2Exposure', 'LaggedO2Feed', 'CumNH3')

    SUPPORTED_MATERIALS = ('carbon_steel', 'magnetite', 'stainless_steel', 'inert')

    def __init__(self, T_kelvin=298.15, P_bar=100.0, water_ppm=50.0, material='carbon_steel',
                 # Phase-condensation multiplier for heterogeneous / wet-film reactions
                 # (R2, R12, R13).
                 # f_phase = (rho_m / rho_m_reference) ** condensation_exponent
                 # 0.0 disables it (backward-compatible default).
                 condensation_exponent=0.0,
                 rho_m_reference=24.0,
                 # Wall-corrosion O2 sink (adsorbed-moisture-film gating, see _wall_o2_rate).
                 # Set wall_area_m2 > 0 to enable.
                 wall_area_m2=0.0,
                 wall_k_intrinsic=1.0e-4,
                 wall_o2_potency=1.0,
                 # Reference water-saturation fraction for wall_o2's combined density+wetness
                 # gas-phase gate (see _wall_gas_phase_enhancement docstring): 0.0 (default)
                 # disables the combined effect (sat/density stay independent, backward-
                 # compatible).
                 wall_o2_sat_ref=0.0,
                 wall_rho_pass=5.0,
                 wall_hill_n=3.0,
                 wall_k_h2o_ppm=3.0,
                 wall_acid_gain=1.0,
                 wall_acid_exponent=1.5,
                 wall_acid_background=0.02,
                 # Optional SECOND power-law term in _acid_enhancement (see its docstring):
                 # gain2=0.0 (default) makes it a complete no-op, backward-compatible.
                 wall_acid_gain2=0.0,
                 wall_acid_exponent2=6.0,
                 wall_consume_h2o=False,
                 # Wet-film activation mode: 'density' (legacy Langmuir g(H2O), the constructor
                 # default) or 'wet_film' (solubility/dew-point theory: a separate aqueous acid
                 # film only exists once gas-phase H2O exceeds its solubility limit in the CO2
                 # phase -- estimated from the Antoine vapor-pressure of pure water, scaled by an
                 # illustrative enhancement factor for CO2/H2O non-ideality, and lowered by
                 # hygroscopic-acid deliquescence. Above that limit the *fraction of the coupon
                 # wetted* saturates quickly (Hill function), but the film's severity keeps
                 # growing with the excess (unabsorbed) water via a separate power-law term --
                 # more free water means a thicker/more conductive electrolyte film, not just a
                 # wetted/dry switch. Used by the general reference profile).
                 wall_h2o_mode='density',
                 wall_h2o_enhancement_factor=1.0,
                 wall_h2o_deliq_ref_ppm=30.0,
                 wall_h2o_hill_n=4.0,
                 wall_h2o_excess_ref_ppm=100.0,
                 wall_h2o_excess_exponent=1.0,
                 # Carbonic-acid wall path: Fe + CO2(aq) + H2O -> FeCO3 + H2. Driven by the
                 # Henry's-law dissolved-CO2 concentration (see _co2_aqueous_solubility_mol_l),
                 # not by a tracked ppm-level species -- CO2 is the bulk carrier gas here, so
                 # unlike the trace-acid paths below this one is not flow-throughput limited.
                 # Real NO2/H2SO4 sink; whenever little of those has converted (low HNO3/H2SO4)
                 # but the gas is wet, this is expected to be the dominant corrosion path.
                 wall_feco3_k_intrinsic=0.0,
                 wall_feco3_potency=1.0,
                 # Nitric-acid wall path: 8 HNO3 + 3 Fe -> 3 Fe(NO3)2 + 2 NO + 4 H2O -- HNO3
                 # (already formed in the bulk gas by R5) directly attacks bare steel, releasing
                 # NO and water back to the gas phase. A real HNO3 sink coupled back into the
                 # species ODEs whenever wall_hno3_corrosion_k_intrinsic > 0.
                 wall_hno3_corrosion_k_intrinsic=0.0,
                 wall_hno3_corrosion_potency=1.0,
                 # Sulfuric-acid wall path: Fe + H2SO4 -> FeSO4 + H2. H2SO4 is a
                 # coupled back into the species ODEs whenever wall_h2so4_k_intrinsic > 0.
                 wall_h2so4_k_intrinsic=0.0,
                 wall_h2so4_potency=1.0,
                 # Direct dry gas-solid NO2 wall path: 2 Fe + 3 NO2 -> Fe2O3 + 3 NO (see
                 # _wall_no2_rate docstring). A real NO2 sink that also produces NO directly,
                 # coupled back into the species ODEs whenever wall_no2_k_intrinsic > 0.
                 wall_no2_k_intrinsic=0.0,
                 wall_no2_potency=1.0,
                 # Cold-favoured Arrhenius term for the wall NO2 path (see _wall_no2_rate
                 # docstring): 0.0 (default) disables it, backward-compatible.
                 wall_no2_ea_kj_mol=0.0,
                 # Absolute-humidity Langmuir gating reference ppm for the wall NO2 path (see
                 # _wall_no2_rate docstring): 0.0 (default) keeps the relative-dew-point gating.
                 wall_no2_h2o_ppm_ref=0.0,
                 # Cumulative-NO2-exposure induction-period threshold/sharpness for the wall NO2
                 # path (see _wall_no2_rate docstring): threshold=0.0 (default) disables it.
                 wall_no2_exposure_threshold_ppm_h=0.0,
                 wall_no2_exposure_hill_n=4.0,
                 # O2-depletion gating reference ppm/sharpness for the wall NO2 path (see
                 # _wall_no2_rate docstring): ref=0.0 (default) disables it.
                 wall_no2_o2_ref_ppm=0.0,
                 wall_no2_o2_hill_n=2.0,
                 # Gas-phase-favoured enhancement for the O2/NO2 wall paths (see
                 # _wall_gas_phase_enhancement docstring): gain=0.0 (default) disables it.
                 # wall_o2_gas_phase_gain (O2 attack) and wall_gas_phase_gain (NO2 attack) are
                 # independent -- the two paths need not share the same gas-phase sensitivity.
                 wall_o2_gas_phase_gain=0.0,
                 wall_gas_phase_gain=0.0,
                 wall_gas_phase_rho_ref=5.0,
                 wall_gas_phase_hill_n=2.0,
                 # Carbon-steel-catalysed Claus-type path: 8 H2S + 4 O2 -> S8 + 8 H2O. Does not
                 # consume Fe (catalytic, not corrosive); only active for carbon_steel/magnetite
                 # (see _wall_s8_rate). Guessed as kinetically slow.
                 wall_s8_k_intrinsic=0.0,
                 wall_s8_h2s_potency=1.0,
                 wall_s8_o2_potency=0.5,
                 # Surface-catalysed SO2 oxidation (see _wall_so2_rate): SO2 + 0.5 O2 + H2O ->
                 # H2SO4, catalysed by an oxide layer that builds up with CUMULATIVE O2 exposure
                 # (CumO2Exposure ODE state -- see EXTRA_STATE_KEYS), not instantaneous O2.
                 # wall_so2_k_intrinsic=0.0 (default) disables it entirely.
                 wall_so2_k_intrinsic=0.0,
                 wall_so2_potency=1.0,
                 wall_so2_exposure_threshold_ppm_h=0.0,
                 wall_so2_exposure_hill_n=2.0,
                 # R1 (SO2 + 0.5 O2 + H2O -> H2SO4) autocatalytic acceleration: a phenomenological
                 # stand-in for the real radical-chain/trace-catalysed SO2-oxidation-to-sulfuric-
                 # acid mechanism (e.g. the historical "lead chamber" NOx-mediated pathway, or
                 # aqueous-film trace-metal-catalysed SO2 autoxidation) -- both directions of R1
                 # are scaled by the SAME factor (a catalyst speeds up the approach to
                 # equilibrium, it does not shift Keq1). Saturating (Langmuir-form) in the
                 # cumulative H2SO4 ever produced (``CumH2SO4``, see EXTRA_STATE_KEYS), not the
                 # current standing H2SO4 ppm, so wall-corrosion consumption of H2SO4 cannot undo
                 # the acceleration. 0.0 disables it (backward-compatible default).
                 r1_autocat_gain=0.0,
                 r1_autocat_ref_ppm=10.0,
                 # R3a (SO2 + NO2 + H2O -> NO + H2SO4) autocatalytic acceleration: same
                 # saturating (Langmuir-form) mechanism and SAME cumulative-H2SO4 driver as
                 # r1_autocat above (both directions scaled equally, Keq3 unchanged) -- models
                 # NO2+SO2 conversion itself speeding up once enough H2SO4 has accumulated (e.g.
                 # an autocatalytic acid-film effect), independent of r1_autocat's own gain so
                 # each reaction can be tuned separately. 0.0 disables it (backward-compatible).
                 r3a_autocat_gain=0.0,
                 r3a_autocat_ref_ppm=10.0,
                 r3a_autocat_hill_n=1.0,
                 r3a_autocat_surface_suppress_gain=0.0,
                 # R12 (H2S+2O2->H2SO4, NO2-catalysed) is a candidate wet-film reaction like R2,
                 # so defaults to the same f_phase suppression -- but unlike R2 it is a genuine
                 # catalytic oxidation that plausibly still proceeds in a dilute gas phase (no
                 # bulk liquid film needed, just adsorbed NO2 on whatever surface/droplets are
                 # present). Set True to bypass f_phase entirely for R12 (rate = 1x regardless of
                 # density). False (default) keeps the original wet-film-suppressed behaviour.
                 r12_density_independent=False,
                 r12_no2_order=1.0,
                 # Order of R13's NO2 (forward) / NO (reverse) rate-law terms. 4.0 (default)
                 # matches R13's literal 4NO2+H2S<->H2SO4+4NO stoichiometry, but is self-defeating
                 # at trace concentrations: as R13 consumes NO2, its own rate collapses even
                 # faster (4th power), so it can never finish depleting a residual NO2 excess no
                 # matter how large its A-factor. Unlike r12_no2_order (a true catalyst, cancels
                 # out of Keq12 exactly), NO2/NO are genuine reactant/product here, so lowering
                 # this DELIBERATELY shifts R13's apparent equilibrium point toward more forward
                 # conversion at low NO2 -- a kinetic-order choice, not a thermodynamic one (real
                 # multi-step mechanisms often have empirical orders that don't match the lumped
                 # overall stoichiometry) -- while keeping the SAME 4:1 mass-balance stoichiometry
                 # in rhs().
                 r13_no2_order=4.0,
                 # R15's own f_phase exponent (f_phase**this, default 1.0 = same suppression as
                 # R2/R12/R13, backward-compatible). <1.0 weakens R15's density suppression.
                 r15_f_phase_exponent=1.0,
                 # R4's surface-to-volume enhancement gain. 2 NO + O2 -> 2 NO2 is termolecular
                 # and runs far faster on a wetted wall film than in the bulk, so its apparent
                 # rate tracks the vessel's A/V ratio (4/d + 2/L). The factor is
                 # 1 + gain * (A_V / _R4_SURFACE_SV_REF_CM_INV - 1), clamped at >=1.0 and
                 # normalised to the reference autoclave, so it is EXACTLY 1.0
                 # for every standard-bore rig and only lifts R4 in a narrower vessel.
                 # 0.0 (default) disables it entirely.
                 r4_surface_gain=0.0,
                 r3a_bore_gain=0.0,
                 # Narrow-bore surface quenching of R15's N2O channel, 1/(1 + gain * excess),
                 # exactly 1.0 at the reference bore. See _r15_surface_suppression.
                 r15_surface_suppress_gain=0.0,
                 # Shared sulfur-catalyst gate for R15 and the wall NO2 path (see
                 # _sulfur_catalyst_gate). 0.0 disables it.
                 r15_sulfur_ref_ppm=0.0,
                 r15_sulfur_hill_n=2.0,
                 # Feed-O2 passivation gates (see _feed_o2_passivation). 0.0 disables each.
                 wall_o2_feed_o2_ref_ppm=0.0,
                 wall_o2_feed_o2_hill_n=4.0,
                 wall_no2_feed_o2_ref_ppm=0.0,
                 wall_no2_feed_o2_hill_n=4.0,
                 r3a_feed_o2_ref_ppm=0.0,
                 r3a_feed_o2_hill_n=4.0,
                 # R3a-ONLY floor on its own feed-O2 passivation gate (see _feed_o2_passivation's
                 # ``floor`` argument) -- caps how strongly O2 can suppress R3a, so SOME SO2+NO2
                 # reaction always keeps proceeding (just slower at higher O2), instead of R3a
                 # being able to shut off almost entirely at very high sustained feed O2. 0.0
                 # disables it (exact pre-existing behaviour, unbounded suppression).
                 r3a_feed_o2_floor=0.0,
                 r3a_feed_o2_cap_ppm=0.0,
                 # O2-PRESENCE gates (distinct from the feed-O2 PASSIVATION gates above): those
                 # suppress at sustained HIGH feed; these suppress when O2 is NOT being fed at
                 # all (genuinely anoxic window), a case where wall_no2's own O2-scarcity gate
                 # and R15's own O2-inhibition gate are BOTH wide open with nothing to throttle
                 # them, letting NO2 crash and NO/N2O spike within a single brief or extended
                 # zero-O2-feed phase. Langmuir form (feed_ppm/(feed_ppm+ref)): ->0 only as
                 # feed->0, ->1 for ANY nonzero feed given a small ref. 0.0 (default) disables
                 # each (factor always 1.0, backward-compatible).
                 wall_no2_o2_presence_ref_ppm=0.0,
                 wall_no2_o2_presence_hill_n=1.0,
                 r3a_o2_presence_ref_ppm=0.0,
                 r3a_o2_presence_hill_n=1.0,
                 r2_o2_presence_ref_ppm=0.0,
                 r2_o2_presence_hill_n=1.0,
                 # R2's NO2-abundance boost: 1 + gain*ratio/(1+ratio), ratio=(C_NO2_ppm/ref)**n --
                 # mirrors r11_o2_gain exactly (same Langmuir-saturating form), but keyed to NO2
                 # instead of O2 and applied to R2 (H2S+3NO2->SO2+H2O+3NO) instead of R11. Lets R2
                 # speed up specifically when NO2 is abundant without a flat, NO2-independent A2
                 # change (which would raise R2's rate identically at low- and high-NO2 checkpoints
                 # alike). ref_ppm<=0 or gain=0.0 (default) disables it (exact no-op).
                 r2_no2_boost_ref_ppm=0.0,
                 r2_no2_boost_hill_n=2.0,
                 r2_no2_boost_gain=0.0,
                 # wall_no2's NO product brake (see _wall_no2_rate docstring): mirrors R15's
                 # N2O brake -- throttles as standing NO approaches/exceeds this ceiling, so
                 # wall_no2 cannot keep converting NO2 into unbounded extra NO once NO is
                 # already abundant. 0.0 (default) disables it (exact no-op).
                 wall_no2_no_cap_ppm=0.0,
                 wall_no2_no_cap_hill_n=2.0,
                 # Langmuir adsorption isotherm for wall_no2's NO2 dependence (see
                 # _wall_no2_rate docstring): a finite number of coupon active sites means the
                 # attack rate should saturate at high NO2 instead of scaling unboundedly.
                 # 0.0 (default) disables it (falls back to the plain wall_no2_potency power law).
                 wall_no2_langmuir_half_ppm=0.0,
                 # R15's O2 PRODUCT-inhibition gate (see rhs()): O2 is a product of
                 # 4 NO2 -> 2 N2O + 3 O2, so an accumulated O2 pool blocks the surface/radical
                 # chain that carries it. Forward factor 1/(1+(O2_ppm/ref)**n), i.e. R15 runs at
                 # full strength in an O2-starved gas and is throttled once O2 is abundant.
                 # ref_ppm=0.0 (default) disables the gate entirely (factor always 1.0,
                 # backward-compatible).
                 r15_o2_inhib_ref_ppm=0.0,
                 r15_o2_inhib_hill_n=2.0,
                 # R15's O2 ACTIVATION gate (see rhs()): the inhibition gate above is
                 # permissive AT O2=0 (nothing left to inhibit with), so it cannot itself stop
                 # R15 during a genuinely anoxic window. This gate instead puts INSTANTANEOUS
                 # O2 directly into the rate as something R15 NEEDS to proceed at all --
                 # Hill-activation form ratio**n/(1+ratio**n), exactly 0 at O2=0 regardless of
                 # NO2, rising to 1 as O2 becomes abundant. ref_ppm=0.0 (default) disables it
                 # entirely (factor always 1.0, backward-compatible).
                 r15_o2_activation_ref_ppm=0.0,
                 r15_o2_activation_hill_n=2.0,
                 # R15's NO2 Langmuir cap (see rhs()): saturates the EFFECTIVE NO2 concentration
                 # feeding R15's forward term only, so a very high standing NO2 (far beyond any
                 # experiment r15_o2_inhib was calibrated against) cannot drive an unbounded
                 # NO2^4 forward rate purely because its own O2-inhibition gate happens to be
                 # wide open (e.g. a genuinely zero-O2-feed window). ppm=0.0 (default) disables
                 # it entirely (exact no-op, backward-compatible).
                 r15_no2_cap_ppm=0.0,
                 r15_no2_cap_hill_n=2.0,
                 # R15's N2O product brake (see rhs()): saturating gate on the FORWARD term,
                 # keyed on the STANDING N2O concentration itself -- unlike r15_o2_inhib and
                 # the reverse/equilibrium term (both need O2, which is ALSO absent during a
                 # genuinely zero-O2-feed window), this brake still works when O2=0. ppm=0.0
                 # (default) disables it entirely (exact no-op, backward-compatible).
                 r15_n2o_cap_ppm=0.0,
                 r15_n2o_cap_hill_n=2.0,
                 # R15's O2-PRESENCE gate (see rhs()): keys off whether O2 is being FED AT ALL
                 # (exogenous C_in[4]), unlike r15_o2_inhib_ref_ppm above which reacts to the
                 # instantaneous/consumed O2 concentration. During a genuinely zero-O2-feed
                 # window r15_o2_inhib is ALWAYS wide open (nothing left to inhibit with),
                 # letting R15 run at full, unthrottled strength off whatever NO2 happens to be
                 # standing -- this gate throttles that specific case while leaving R15 at full
                 # strength whenever O2 is genuinely being fed (its normal operating regime,
                 # including experiments that need R15's own O2 byproduct for their O2 targets).
                 # Langmuir form (feed_ppm/(feed_ppm+ref)): ->0 only as feed->0, ->1 for ANY
                 # nonzero feed given a small ref. ref_ppm=0.0 (default) disables it entirely
                 # (factor always 1.0, backward-compatible).
                 r15_o2_presence_ref_ppm=0.0,
                 r15_o2_presence_hill_n=1.0,
                 # R11's forward rate gets an O2-abundance boost (see rhs()): 0.0 (default)
                 # disables it (boost always 1.0, backward-compatible) -- ADDS strength above
                 # the current A11 value once O2 is abundant, never subtracts from it.
                 r11_o2_ref_ppm=0.0,
                 r11_o2_hill_n=2.0,
                 r11_o2_gain=0.0,
                 # Symmetric lag time constant (hours) for the LaggedO2 ODE state (see rhs()
                 # docstring). 0.0 (default) disables it entirely -- r15_o2_inhib/wall_no2's
                 # O2-scarcity gate then read raw instantaneous O2, as before.
                 o2_lag_tau_hours=0.0,
                 # ASYMMETRIC lag time constants (hours) for the LaggedO2Feed ODE state (see
                 # rhs() docstring) -- fast rise, slow fall, tracking the FED (not instantaneous)
                 # O2 concentration for use ONLY by wall_o2's own feed-passivation gate.
                 # o2_feed_lag_tau_hours=0.0 (default) disables it entirely (wall_o2 falls back
                 # to the raw, discontinuous feed step, as before).
                 o2_feed_lag_tau_hours=0.0,
                 o2_feed_lag_rise_tau_hours=1.0,
                 # CO2-species binary interaction parameters (kij) for the SRK fugacity flash,
                 # keyed by SPECIES name (e.g. {'NO2': 0.7}). Default 0.0 for every pair (i.e.
                 # NeqSim's own database value, which is 0 for these uncommon trace pairs).
                 # Tunable override for species whose bulk-CO2-phase fugacity coefficient is not
                 # representative of their true reactive availability -- e.g. once a separate
                 # aqueous/acid film is included, NO2 may leave the bulk CO2 phase and react
                 # there more readily than a kij=0 bulk-phase SRK flash represents.
                 srk_kij_co2=None):
        self.T = T_kelvin
        self.P = P_bar
        self.water_ppm = water_ppm
        self.material = material.lower().replace(' ', '_')
        if self.material not in self.SUPPORTED_MATERIALS:
            self.material = 'carbon_steel'

        self.kinetic_params = {k: v.copy() for k, v in DEFAULT_KINETIC_PARAMS.items()}

        self.diameter_cm = 6.50
        self.volume_ml = 300.0
        self.mass_flow_g_h = 50.0
        self.length_cm = self.volume_ml / (np.pi * (self.diameter_cm**2) / 4.0)

        self.srk_kij_co2 = dict(srk_kij_co2) if srk_kij_co2 else {}
        self.molar_density, self.phase, self.phi_dict = self._calculate_srk_fugacities(T_kelvin, P_bar)
        # Deliberate phi overrides (see set_phi_override), replayed after any re-flash so a
        # temperature/pressure change cannot silently revert them to raw SRK values.
        self._phi_overrides = {}
        self._water_solubility_ppm_base = self._calculate_water_solubility_ppm(T_kelvin, P_bar)

        # Phase-condensation switch (constant per (T, P))
        self.condensation_exponent = float(condensation_exponent)
        self.rho_m_reference = float(rho_m_reference)
        self._f_phase = self._compute_f_phase()

        # R5's reverse-term NO activity, decoupled from the shared/global phi_dict['NO'] used by
        # R4/R7/R11/R13 (that global value is a tuned fix for R4's 2NO+O2->2NO2 over-recycling
        # specifically, only relevant when O2 is co-fed -- applying that same suppression inside
        # R5's own reverse term is unrelated to R4 and, at high-NO2/no-O2 feeds (no R4 activity to
        # correct for), silently starves R5's reverse reaction and lets HNO3 run away well past
        # its true equilibrium. Default 1.0 (ideal fugacity, consistent with NO2/H2O/HNO3 already
        # being treated as ideal elsewhere) so R5 alone reaches its own genuine equilibrium point.
        self.r5_no_activity = 1.0

        # R4's forward-term NO activity, decoupled from the shared/global phi_dict['NO'] for the
        # same reason as r5_no_activity above (see set_r4_no_activity docstring).
        self.r4_no_activity = 1.0

        # R4's surface-to-volume enhancement gain (see r4_surface_gain in __init__'s signature).
        # 0.0 (default) is an exact no-op, and it stays 1.0 at the reference bore regardless.
        self.r4_surface_gain = float(r4_surface_gain)
        self.r3a_bore_gain = float(r3a_bore_gain)
        self.r15_surface_suppress_gain = float(r15_surface_suppress_gain)
        self.r15_sulfur_ref_ppm = float(r15_sulfur_ref_ppm)
        self.r15_sulfur_hill_n = float(r15_sulfur_hill_n)
        self.wall_o2_feed_o2_ref_ppm = float(wall_o2_feed_o2_ref_ppm)
        self.wall_o2_feed_o2_hill_n = float(wall_o2_feed_o2_hill_n)
        self.wall_no2_feed_o2_ref_ppm = float(wall_no2_feed_o2_ref_ppm)
        self.wall_no2_feed_o2_hill_n = float(wall_no2_feed_o2_hill_n)
        self.r3a_feed_o2_ref_ppm = float(r3a_feed_o2_ref_ppm)
        self.r3a_feed_o2_hill_n = float(r3a_feed_o2_hill_n)
        self.r3a_feed_o2_floor = float(r3a_feed_o2_floor)
        self.r3a_feed_o2_cap_ppm = float(r3a_feed_o2_cap_ppm)
        self.wall_no2_o2_presence_ref_ppm = float(wall_no2_o2_presence_ref_ppm)
        self.wall_no2_o2_presence_hill_n = float(wall_no2_o2_presence_hill_n)
        self.wall_no2_no_cap_ppm = float(wall_no2_no_cap_ppm)
        self.wall_no2_no_cap_hill_n = float(wall_no2_no_cap_hill_n)
        self.wall_no2_langmuir_half_ppm = float(wall_no2_langmuir_half_ppm)
        self.r3a_o2_presence_ref_ppm = float(r3a_o2_presence_ref_ppm)
        self.r3a_o2_presence_hill_n = float(r3a_o2_presence_hill_n)
        self.r2_o2_presence_ref_ppm = float(r2_o2_presence_ref_ppm)
        self.r2_o2_presence_hill_n = float(r2_o2_presence_hill_n)

        # R3a's reverse-term NO activity fraction that "escapes" the liquid/wet film, scaled by
        # f_phase (see set_r3a_no_escape_frac docstring): genuinely shifts R3a's apparent
        # equilibrium toward more forward conversion (SO2+NO2+H2O -> NO+H2SO4) specifically at
        # dense/liquid-like conditions, without touching Keq3a itself or affecting gas-phase
        # (low f_phase) conditions. 0.0 (default) is backward-compatible/no-op.
        self.r3a_no_escape_frac = 0.0

        # Wall corrosion (heterogeneous O2 sink on carbon-steel coupon)
        self.wall_area_m2 = float(wall_area_m2)
        self.wall_k_intrinsic = float(wall_k_intrinsic)
        self.wall_o2_potency = float(wall_o2_potency)
        self.wall_o2_sat_ref = float(wall_o2_sat_ref)
        self.wall_rho_pass = float(wall_rho_pass)
        self.wall_hill_n = float(wall_hill_n)
        self.wall_k_h2o_ppm = float(wall_k_h2o_ppm)
        self.wall_acid_gain = float(wall_acid_gain)
        self.wall_acid_exponent = float(wall_acid_exponent)
        self.wall_acid_background = float(wall_acid_background)
        self.wall_acid_gain2 = float(wall_acid_gain2)
        self.wall_acid_exponent2 = float(wall_acid_exponent2)
        self.wall_consume_h2o = bool(wall_consume_h2o)
        self.wall_h2o_mode = str(wall_h2o_mode)
        self.wall_h2o_enhancement_factor = float(wall_h2o_enhancement_factor)
        self.wall_h2o_deliq_ref_ppm = float(wall_h2o_deliq_ref_ppm)
        self.wall_h2o_hill_n = float(wall_h2o_hill_n)
        self.wall_h2o_excess_ref_ppm = float(wall_h2o_excess_ref_ppm)
        self.wall_h2o_excess_exponent = float(wall_h2o_excess_exponent)
        self.wall_feco3_k_intrinsic = float(wall_feco3_k_intrinsic)
        self.wall_feco3_potency = float(wall_feco3_potency)
        self.wall_hno3_corrosion_k_intrinsic = float(wall_hno3_corrosion_k_intrinsic)
        self.wall_hno3_corrosion_potency = float(wall_hno3_corrosion_potency)
        self.wall_h2so4_k_intrinsic = float(wall_h2so4_k_intrinsic)
        self.wall_h2so4_potency = float(wall_h2so4_potency)
        self.wall_no2_k_intrinsic = float(wall_no2_k_intrinsic)
        self.wall_no2_potency = float(wall_no2_potency)
        self.wall_no2_ea_kj_mol = float(wall_no2_ea_kj_mol)
        self.wall_no2_h2o_ppm_ref = float(wall_no2_h2o_ppm_ref)
        self.wall_no2_exposure_threshold_ppm_h = float(wall_no2_exposure_threshold_ppm_h)
        self.wall_no2_exposure_hill_n = float(wall_no2_exposure_hill_n)
        self.wall_no2_o2_ref_ppm = float(wall_no2_o2_ref_ppm)
        self.wall_no2_o2_hill_n = float(wall_no2_o2_hill_n)
        self.wall_o2_gas_phase_gain = float(wall_o2_gas_phase_gain)
        self.wall_gas_phase_gain = float(wall_gas_phase_gain)
        self.wall_gas_phase_rho_ref = float(wall_gas_phase_rho_ref)
        self.wall_gas_phase_hill_n = float(wall_gas_phase_hill_n)
        self.wall_s8_k_intrinsic = float(wall_s8_k_intrinsic)
        self.wall_s8_h2s_potency = float(wall_s8_h2s_potency)
        self.wall_s8_o2_potency = float(wall_s8_o2_potency)
        self.wall_so2_k_intrinsic = float(wall_so2_k_intrinsic)
        self.wall_so2_potency = float(wall_so2_potency)
        self.wall_so2_exposure_threshold_ppm_h = float(wall_so2_exposure_threshold_ppm_h)
        self.wall_so2_exposure_hill_n = float(wall_so2_exposure_hill_n)
        self.r1_autocat_gain = float(r1_autocat_gain)
        self.r1_autocat_ref_ppm = float(r1_autocat_ref_ppm)
        self.r3a_autocat_gain = float(r3a_autocat_gain)
        self.r3a_autocat_ref_ppm = float(r3a_autocat_ref_ppm)
        self.r3a_autocat_hill_n = float(r3a_autocat_hill_n)
        self.r3a_autocat_surface_suppress_gain = float(r3a_autocat_surface_suppress_gain)
        self.r12_density_independent = bool(r12_density_independent)
        self.r12_no2_order = float(r12_no2_order)
        self.r13_no2_order = float(r13_no2_order)
        self.r15_f_phase_exponent = float(r15_f_phase_exponent)
        self.r15_o2_inhib_ref_ppm = float(r15_o2_inhib_ref_ppm)
        self.r15_o2_inhib_hill_n = float(r15_o2_inhib_hill_n)
        self.r15_o2_activation_ref_ppm = float(r15_o2_activation_ref_ppm)
        self.r15_o2_activation_hill_n = float(r15_o2_activation_hill_n)
        self.r15_no2_cap_ppm = float(r15_no2_cap_ppm)
        self.r15_no2_cap_hill_n = float(r15_no2_cap_hill_n)
        self.r15_n2o_cap_ppm = float(r15_n2o_cap_ppm)
        self.r15_n2o_cap_hill_n = float(r15_n2o_cap_hill_n)
        self.r15_o2_presence_ref_ppm = float(r15_o2_presence_ref_ppm)
        self.r15_o2_presence_hill_n = float(r15_o2_presence_hill_n)
        self.r11_o2_ref_ppm = float(r11_o2_ref_ppm)
        self.r11_o2_hill_n = float(r11_o2_hill_n)
        self.r11_o2_gain = float(r11_o2_gain)
        self.r2_no2_boost_ref_ppm = float(r2_no2_boost_ref_ppm)
        self.r2_no2_boost_hill_n = float(r2_no2_boost_hill_n)
        self.r2_no2_boost_gain = float(r2_no2_boost_gain)
        self.o2_lag_tau_hours = float(o2_lag_tau_hours)
        self.o2_feed_lag_tau_hours = float(o2_feed_lag_tau_hours)
        self.o2_feed_lag_rise_tau_hours = float(o2_feed_lag_rise_tau_hours)
        self._wall_theta_pass = self._compute_wall_theta_pass()

    def _compute_f_phase(self):
        if self.condensation_exponent <= 0.0 or self.rho_m_reference <= 0.0:
            return 1.0
        return (self.molar_density / self.rho_m_reference) ** self.condensation_exponent

    def _compute_wall_theta_pass(self):
        if self.wall_rho_pass <= 0.0:
            return 0.0
        x = (self.molar_density / self.wall_rho_pass) ** self.wall_hill_n
        return x / (1.0 + x)

    def _wall_k_total(self):
        """Lumped wall rate constant K = k_intrinsic * A_S/V [1/s]."""
        V_m3 = max(self.volume_ml, 1e-9) * 1e-6
        return self.wall_k_intrinsic * (self.wall_area_m2 / V_m3)

    def _total_acid_ppm(self, C_NO2, C_H2SO4, C_HNO3):
        """Instantaneous NO2 + H2SO4 + HNO3 gas-phase loading, expressed as an equivalent
        mole-fraction ppm of the bulk gas (kmol / kmol gas * 1e6). Deliberately includes bare
        NO2 gas itself (not just the acids it forms) since the wall O2-attack signal is meant to
        track the NO2 dosing schedule directly.
        """
        return (max(C_NO2, 0.0) + max(C_H2SO4, 0.0) + max(C_HNO3, 0.0)) \
            / max(self.molar_density, 1e-9) * 1e6

    def _acid_enhancement(self, C_NO2, C_H2SO4, C_HNO3):
        """NO2-dosing-driven gating factor for the wall O2 sink (dimensionless).

        Scales with the instantaneous NO2 + H2SO4 + HNO3 loading (see ``_total_acid_ppm``), not
        a cumulative tracker, so it heals immediately once NO2 dosing stops (a known, accepted
        simplification).

        Two additive power-law terms: a gentle one (``wall_acid_gain``/``wall_acid_exponent``)
        plus an optional steep second one (``wall_acid_gain2``/``wall_acid_exponent2``, default
        gain2=0.0 i.e. inert) meant to stay negligible at low NO2/acid loading while engaging
        strongly only once the loading climbs high -- a threshold response layered on top of the
        gentle term, not a replacement for it.
        """
        total_acid_ppm = self._total_acid_ppm(C_NO2, C_H2SO4, C_HNO3)
        enhancement = self.wall_acid_background + self.wall_acid_gain * total_acid_ppm ** self.wall_acid_exponent
        if self.wall_acid_gain2 > 0.0:
            enhancement += self.wall_acid_gain2 * total_acid_ppm ** self.wall_acid_exponent2
        return enhancement

    def _water_solubility_ppm(self):
        """Water solubility (dew point) in the CO2-rich phase [ppm mol].

        Returns the SRK-based value cached at construction (see
        ``_calculate_water_solubility_ppm``), scaled by ``wall_h2o_enhancement_factor`` -- an
        illustrative, tunable residual correction on top of the thermodynamic estimate. This
        anchors the wet-film threshold to a genuine solubility-limit concept (water condenses
        into a separate phase once the feed exceeds this dew point) rather than an arbitrary
        empirical ppm constant.
        """
        return self._water_solubility_ppm_base * self.wall_h2o_enhancement_factor

    def _calculate_water_solubility_ppm(self, T_K, P_bar):
        """Water solubility (dew point) in the CO2-rich phase [ppm mol], via a modified
        Raoult's-law estimate corrected by the SRK fugacity coefficient of water.

        y_H2O,sat = P_sat_H2O(T) / (phi_H2O * P), with ``P_sat_H2O(T)`` from the Antoine
        equation and ``phi_H2O`` the water fugacity coefficient already computed by
        ``_calculate_srk_fugacities`` for the bulk gas (assumes the condensed phase is
        essentially pure water, i.e. x_H2O ~ 1, gamma ~ 1). ``phi_H2O`` is what captures the
        *real* (non-ideal) SRK behaviour of water in dense/supercritical CO2 -- the actual
        physical reason the solubility departs from the naive ideal-gas value -- while
        reusing the single VLE calculation already performed at construction rather than
        issuing extra NeqSim flashes (repeated flashes are unreliable in this environment's
        JVM/JPype setup, see AGENTS notes).
        """
        T_C = T_K - 273.15
        log10_p_mmhg = 8.07131 - 1730.63 / (233.426 + T_C)   # Antoine eq., water, 1-100 C
        p_sat_bar = (10.0 ** log10_p_mmhg) * 1.33322e-3       # mmHg -> bar
        phi_h2o = max(self.phi_dict.get('H2O', 1.0), 1e-6)
        return (p_sat_bar / (phi_h2o * max(P_bar, 1e-9))) * 1e6

    def _strong_acid_ppm(self, C_H2SO4, C_HNO3):
        """Hygroscopic strong-acid loading (H2SO4 + HNO3) [ppm], driving deliquescence.

        Deliberately excludes NO2: NO2 gas itself is not strongly hygroscopic, so with no
        HNO3/H2SO4 formed yet the wetted-film threshold stays at the pure-water solubility
        limit (i.e. it is fine to run with water up to saturation on NO2 alone). Once HNO3/
        H2SO4 actually form, their strong affinity for water stabilizes a liquid film at a
        lower bulk water content than pure-water saturation (deliquescence).
        """
        return (C_H2SO4 + C_HNO3) / max(self.molar_density, 1e-9) * 1e6

    def _co2_aqueous_solubility_mol_l(self):
        """Henry's-law dissolved-CO2 concentration in the condensed water film [mol/L].

        Uses the van't Hoff temperature dependence of CO2's Henry's-law solubility constant
        (K_H,298 = 0.034 mol/(L*atm), dH/R ~ 2400 K; standard literature values, e.g. Sander
        2015) and takes the CO2 partial pressure as the total system pressure -- the bulk gas
        here is essentially pure CO2. This grounds the carbonic-acid corrosion driving force
        in real gas-solubility physics (and its temperature dependence) instead of an
        arbitrary empirical constant.
        """
        K_H_298 = 0.034   # mol / (L * atm)
        DH_OVER_R = 2400.0   # K
        K_H_T = K_H_298 * np.exp(DH_OVER_R * (1.0 / self.T - 1.0 / 298.15))
        P_co2_atm = self.P * 0.986923   # bar -> atm
        return K_H_T * P_co2_atm

    def _effective_g_h2o(self, h2o_ppm, C_H2SO4, C_HNO3):
        """Wet-film activation factor: wetted-fraction x excess-film severity.

        ``'density'`` mode is the legacy Langmuir form h2o_ppm / (h2o_ppm + K). ``'wet_film'``
        mode is a two-part solubility/dew-point model: (1) a Hill-saturating *wetted fraction*
        of the coupon, active once H2O exceeds the SRK-flash water-solubility limit from
        ``_water_solubility_ppm()`` (lowered by hygroscopic strong-acid deliquescence, see
        ``_strong_acid_ppm``); and (2) an unbounded *excess-film severity* term, growing with
        the water beyond that limit, since more free water builds a thicker/more conductive
        electrolyte film rather than simply toggling corrosion on/off. The return value is
        therefore a multiplier that need not stay below 1.
        """
        if h2o_ppm <= 0.0:
            return 0.0
        if self.wall_h2o_mode == 'wet_film':
            w_sat = self._water_solubility_ppm()
            strong_acid_ppm = self._strong_acid_ppm(C_H2SO4, C_HNO3)
            w_sat_eff = w_sat / (1.0 + max(strong_acid_ppm, 0.0) / self.wall_h2o_deliq_ref_ppm)
            ratio = (h2o_ppm / max(w_sat_eff, 1e-9)) ** self.wall_h2o_hill_n
            wetted_fraction = ratio / (1.0 + ratio)
            excess_ppm = max(h2o_ppm - w_sat_eff, 0.0)
            severity = 1.0 + (excess_ppm / max(self.wall_h2o_excess_ref_ppm, 1e-9)) \
                ** self.wall_h2o_excess_exponent
            return wetted_fraction * severity
        return h2o_ppm / (h2o_ppm + self.wall_k_h2o_ppm)

    def _wall_gas_phase_enhancement(self, gain, sat=None, sat_ref=None):
        """Gas-phase-favoured Hill enhancement for surface oxidative attack (O2, NO2 wall
        paths): direct gas-solid electrochemical corrosion is understood to be MORE prominent in
        a dilute gas phase (thin adsorbed-moisture-film corrosion, more direct molecular contact
        with the bare metal) than in a dense/liquid CO2 phase (where the bulk fluid effectively
        shields the surface) -- the OPPOSITE density dependence to the wet-film R2/R12/R13/R15
        reactions (which need a genuine condensed film, favoured by density). ``ratio =
        wall_gas_phase_rho_ref/rho_m`` grows as the bulk density drops below the reference.

        If ``sat``/``sat_ref`` are given (O2 path only, see ``wall_o2_sat_ref``), the water-
        saturation fraction is folded into the SAME ratio (``ratio *= sat/sat_ref``) instead of
        being a separate multiplier, since relative water saturation can differ materially
        between rigs of similar absolute H2O ppm once their dew points differ (e.g. warmer, low-
        density gas streams). ``enhancement = 1 + gain*ratio**n/(1+ratio**n)``. ``gain`` is
        passed in per-caller (``wall_o2_gas_phase_gain`` for the O2 path, ``wall_gas_phase_gain``
        for the NO2 path) so the two attack paths can be tuned independently. ``gain<=0.0``
        disables it (enhancement always 1.0).
        """
        if gain <= 0.0 or self.wall_gas_phase_rho_ref <= 0.0:
            return 1.0
        ratio = self.wall_gas_phase_rho_ref / max(self.molar_density, 1e-9)
        if sat is not None and sat_ref is not None and sat_ref > 0.0:
            ratio *= max(sat, 0.0) / sat_ref
        ratio_n = ratio ** self.wall_gas_phase_hill_n
        return 1.0 + gain * ratio_n / (1.0 + ratio_n)

    def _feed_o2_passivation(self, C_O2_feed, ref_ppm, hill_n, floor=0.0, cap_ppm=0.0):
        """Passivation gate keyed to the FED O2 level: ``1/(1+(feed_ppm/ref)**n)``.

        Sustained high O2 exposure accelerates passive oxide-film growth, shielding the surface
        from further active attack. Driven by the FED (exogenous) O2 rather than the
        instantaneous bulk value on purpose: the instantaneous value is itself depressed by the
        very reactions this gates, which would otherwise close a self-reinforcing loop.
        ``ref_ppm <= 0`` disables it (returns 1.0).

        ``floor`` clamps the minimum returned value (default 0.0, i.e. no floor). This is a
        per-caller argument, not a shared default -- wall_o2/wall_no2 both genuinely need
        near-total suppression at high sustained feed O2, so only R3a's own call site passes a
        nonzero floor (see ``r3a_feed_o2_floor``).

        ``cap_ppm`` clamps the fed ppm value itself (before the gate formula), so any feed at or
        above ``cap_ppm`` is treated identically to ``cap_ppm`` (default 0.0 = no cap). Also a
        per-caller argument, R3a-only (see ``r3a_feed_o2_cap_ppm``).
        """
        if ref_ppm <= 0.0 or C_O2_feed is None:
            return 1.0
        feed_ppm = max(C_O2_feed, 0.0) / max(self.molar_density, 1e-9) * 1e6
        if cap_ppm > 0.0:
            feed_ppm = min(feed_ppm, cap_ppm)
        gate = 1.0 / (1.0 + (feed_ppm / ref_ppm) ** hill_n)
        return max(floor, gate)

    def _o2_presence_gate(self, C_O2_feed, ref_ppm, hill_n):
        """O2-PRESENCE gate keyed to the FED O2 level: ``ratio/(1+ratio)``, ``ratio =
        (feed_ppm/ref)**n`` -- the mirror image of ``_feed_o2_passivation`` above.

        Several reactions/wall paths have their OWN gate that reacts to LOW instantaneous O2 by
        firing MORE strongly (R15's product-inhibition release, wall_no2's O2-scarcity gate) --
        physically correct for the ordinary case of O2 being merely depleted BY CONSUMPTION while
        still genuinely being fed. But during a window where O2 is not being fed AT ALL (exactly
        zero, whether a brief pulse or an extended anoxic hold), those gates are ALWAYS wide open
        with nothing left to throttle them, letting the reaction run unchecked off whatever NO2
        happens to be standing. This gate throttles that specific, genuinely-anoxic case while
        leaving the reaction at full strength the moment ANY real O2 feed resumes -- driven by
        the FED (exogenous) value for the same self-reinforcing-loop reason as
        ``_feed_o2_passivation``. ``ref_ppm <= 0`` disables it (returns 1.0).
        """
        if ref_ppm <= 0.0 or C_O2_feed is None:
            return 1.0
        feed_ppm = max(C_O2_feed, 0.0) / max(self.molar_density, 1e-9) * 1e6
        if feed_ppm <= 0.0:
            return 0.0
        ratio = (feed_ppm / ref_ppm) ** hill_n
        return ratio / (1.0 + ratio)

    def _wall_o2_rate(self, C_O2, h2o_ppm, C_NO2, C_H2SO4, C_HNO3, C_O2_feed=None):
        """O2-driven Fe2O3 wall-corrosion sink rate [kmol O2/(m^3 s)]: 4 Fe + 3 O2 -> 2 Fe2O3.

        Gated by the SAME ``_water_saturation_fraction`` wetting index as the other two active
        wall paths, AND by ``_acid_enhancement`` driven by the INSTANTANEOUS NO2+H2SO4+HNO3
        loading (see ``_total_acid_ppm``) -- direct dry O2 attack on bare steel is slow; real
        rust formation in this system is understood to be driven by the acid film/NO2 exposure
        the coupon currently sees, which is genuinely zero before any NO2 dosing starts (no
        artificial always-on "background" floor needed to bootstrap it). Also scaled by
        ``_wall_gas_phase_enhancement`` (see its docstring, via ``wall_o2_gas_phase_gain`` and
        ``wall_o2_sat_ref``): O2 attack is favoured in a dilute, relatively wetter gas phase over
        a dense/liquid, relatively drier one.
        """
        if self.wall_area_m2 <= 0.0 or self.wall_k_intrinsic <= 0.0:
            return 0.0
        sat = self._water_saturation_fraction(h2o_ppm, C_H2SO4, C_HNO3)
        enhancement = self._acid_enhancement(C_NO2, C_H2SO4, C_HNO3)
        gas_phase = self._wall_gas_phase_enhancement(self.wall_o2_gas_phase_gain, sat=sat,
                                                      sat_ref=self.wall_o2_sat_ref)
        passivation = self._feed_o2_passivation(C_O2_feed, self.wall_o2_feed_o2_ref_ppm,
                                                self.wall_o2_feed_o2_hill_n)
        o2_ppm = max(C_O2, 0.0) / max(self.molar_density, 1e-9) * 1e6
        V_m3 = max(self.volume_ml, 1e-9) * 1e-6
        A_sv = self.wall_area_m2 / V_m3
        return self.wall_k_intrinsic * sat * enhancement * gas_phase * passivation * \
            (o2_ppm ** self.wall_o2_potency) * A_sv * 1e-3

    def _water_saturation_fraction(self, h2o_ppm, C_H2SO4=0.0, C_HNO3=0.0):
        """Fraction of the (acid-lowered) water dew point reached by the bulk gas, in [0, 1].

        A simple relative-humidity-like wetting index driving how much of the coupon surface
        carries an adsorbed/condensed aqueous film for the three ACTIVE acid wall-corrosion paths
        (O2, HNO3+Fe, H2SO4+Fe): ramps linearly with the gas H2O content and saturates at 1 once
        the gas reaches/exceeds its own effective solubility limit -- a genuinely dry gas cannot
        support acid attack on bare steel, and a fully saturated/wet one cannot be gated any
        harder than "fully wetted" by this simple index. The dew point itself is lowered by
        hygroscopic H2SO4/HNO3 deliquescence (see ``_strong_acid_ppm``), the SAME correction
        already applied to the ``'wet_film'`` FeCO3 path's ``_effective_g_h2o`` -- once real acid
        has formed, a concentrated H2SO4/HNO3 film is far more hygroscopic than pure water and
        stays wetted at a lower bulk H2O content than the naive pure-water dew point implies, an
        effect that matters more at cold conditions where the pure-water dew point is already
        only tens of ppm.
        """
        w_sat = self._water_solubility_ppm()
        strong_acid_ppm = self._strong_acid_ppm(C_H2SO4, C_HNO3)
        w_sat_eff = w_sat / (1.0 + max(strong_acid_ppm, 0.0) / self.wall_h2o_deliq_ref_ppm)
        return float(np.clip(max(h2o_ppm, 0.0) / max(w_sat_eff, 1e-9), 0.0, 1.0))

    def _wall_feco3_rate(self, h2o_ppm, C_H2SO4, C_HNO3):
        """Carbonic-acid (iron-carbonate) wall path [kmol Fe/(m^3 s)]:
        Fe + CO2(aq) + H2O -> FeCO3 + H2.

        Driven by the wetted-film factor (a liquid film must exist at all) times the
        Henry's-law dissolved-CO2 concentration (``_co2_aqueous_solubility_mol_l``). CO2 is
        the bulk carrier gas here, not a tracked ppm-level species, so unlike the trace-acid
        paths this one is not limited by a fixed feed-gas throughput -- consistent with
        classical wet-CO2 corrosion, which dominates whenever little strong acid has formed
        (low HNO3/H2SO4) but the gas remains wet.
        """
        if self.wall_area_m2 <= 0.0 or self.wall_feco3_k_intrinsic <= 0.0:
            return 0.0
        g_h2o = self._effective_g_h2o(h2o_ppm, C_H2SO4, C_HNO3)
        co2_aq = self._co2_aqueous_solubility_mol_l()
        V_m3 = max(self.volume_ml, 1e-9) * 1e-6
        A_sv = self.wall_area_m2 / V_m3
        return self.wall_feco3_k_intrinsic * g_h2o * (co2_aq ** self.wall_feco3_potency) * A_sv * 1e-3

    def _wall_hno3_corrosion_rate(self, h2o_ppm, C_HNO3, C_H2SO4=0.0):
        """Nitric-acid wall-film corrosion rate [kmol Fe(NO3)2/(m^3 s)]:
        8 HNO3 + 3 Fe -> 3 Fe(NO3)2 + 2 NO + 4 H2O.

        HNO3 (formed in the bulk gas by R5, 3 NO2 + H2O <-> 2 HNO3 + NO) is the species that
        actually attacks bare iron -- unlike the earlier NO2-driven formulation, this reaction
        is now the textbook dilute-nitric-acid/iron reaction, releasing NO and water rather than
        consuming NO2 directly. Gated by the ``_water_saturation_fraction`` wetting index (a
        liquid/adsorbed film must exist at all for an aqueous acid attack to proceed, now also
        lowered by HNO3/H2SO4 deliquescence -- see that method's docstring) AND by the
        SAME ``_f_phase`` heterogeneous/wet-film density-ratio factor used for R2/R12/R13: a
        wall acid
        film is a wet, condensed-phase-like environment that forms far more readily against
        dense/liquid CO2 than against a dilute low-density gas, where there is genuinely less
        residence time/molecular contact available for the film to build up and react. Returned
        rate is defined as the rate of Fe(NO3)2 formation (1:1 with Fe consumed); the caller
        (``rhs``) applies the remaining 8:2:4 HNO3:NO:H2O stoichiometric ratios relative to it.
        """
        if self.wall_area_m2 <= 0.0 or self.wall_hno3_corrosion_k_intrinsic <= 0.0:
            return 0.0
        sat = self._water_saturation_fraction(h2o_ppm, C_H2SO4, C_HNO3)
        hno3_ppm = max(C_HNO3, 0.0) / max(self.molar_density, 1e-9) * 1e6
        V_m3 = max(self.volume_ml, 1e-9) * 1e-6
        A_sv = self.wall_area_m2 / V_m3
        return self.wall_hno3_corrosion_k_intrinsic * sat * self._f_phase * \
            (hno3_ppm ** self.wall_hno3_corrosion_potency) * A_sv * 1e-3

    def _wall_h2so4_rate(self, h2o_ppm, C_H2SO4, C_HNO3=0.0):
        """Sulfuric-acid wall path Fe rate [kmol Fe/(m^3 s)]: Fe + H2SO4 -> FeSO4 + H2.

        Gated by the ``_water_saturation_fraction`` wetting index (now also lowered by HNO3/
        H2SO4 deliquescence) AND ``_f_phase`` (same heterogeneous/wet-film density-ratio
        reasoning as ``_wall_hno3_corrosion_rate`` above). Applied in ``rhs`` whenever
        ``wall_h2so4_k_intrinsic`` > 0.
        """
        if self.wall_area_m2 <= 0.0 or self.wall_h2so4_k_intrinsic <= 0.0:
            return 0.0
        sat = self._water_saturation_fraction(h2o_ppm, C_H2SO4, C_HNO3)
        h2so4_ppm = max(C_H2SO4, 0.0) / max(self.molar_density, 1e-9) * 1e6
        V_m3 = max(self.volume_ml, 1e-9) * 1e-6
        A_sv = self.wall_area_m2 / V_m3
        return self.wall_h2so4_k_intrinsic * sat * self._f_phase * \
            (h2so4_ppm ** self.wall_h2so4_potency) * A_sv * 1e-3

    def _sulfur_catalyst_gate(self, C_H2S_raw, C_H2SO4_raw):
        """Hill gate on the presence of ANY sulfur species (H2S + H2SO4), in RAW ppm.

        Both the NO2 -> N2O disproportionation (R15) and the dry NO2 + Fe wall attack are
        understood to need a sulfur co-contaminant to proceed at a meaningful rate. Uses raw
        (not phi-scaled) concentration per the wall-decoupling rule: H2SO4's SRK fugacity
        coefficient is ~2e-8 here, so a phi-scaled read would keep this gate permanently shut.
        ``ref_ppm <= 0`` disables it (returns 1.0).
        """
        if self.r15_sulfur_ref_ppm <= 0.0:
            return 1.0
        sulfur_ppm = (max(C_H2S_raw, 0.0) + max(C_H2SO4_raw, 0.0)) \
            / max(self.molar_density, 1e-9) * 1e6
        ratio_n = (sulfur_ppm / self.r15_sulfur_ref_ppm) ** self.r15_sulfur_hill_n
        return ratio_n / (1.0 + ratio_n)

    def _wall_no2_rate(self, C_NO2, h2o_ppm, cum_no2_exposure=0.0, C_O2=None,
                       C_H2S_raw=0.0, C_H2SO4_raw=0.0, C_O2_feed=None, C_O2_lagged=None,
                       C_NO=None):
        """NO2 wall-corrosion sink rate [kmol NO2/(m^3 s)]: 2 Fe + 3 NO2 -> Fe2O3 + 3 NO.

        Like the acid-film paths above, this ACCELERATES WITH water saturation (adsorbed-
        moisture-film corrosion, not a bone-dry chemisorption process): a thin electrolyte film
        is what enables the ionic/electrochemical corrosion mechanism, and this effect is
        understood to be MORE prominent in a dilute gas phase (see ``_wall_gas_phase_enhancement``)
        than in a dense/liquid CO2 phase. Provides NO2 with a real, Fe-mediated sink that also
        produces NO directly without routing through HNO3/R5, so it is NOT subject to R5's own
        Keq5-governed equilibrium.

        Optional Arrhenius term (``wall_no2_ea_kj_mol``, default 0.0 = no-op): the dew-point-
        relative dry_factor ALONE makes this reaction look MORE favourable at warmer conditions
        (water solubility/dew point rises faster with T than typical feed H2O ppm does, so a
        warmer stream can appear relatively "drier"), the opposite of the cold-favoured behaviour
        intended. A negative ``wall_no2_ea_kj_mol`` counteracts this by making the reaction
        genuinely SLOWER at higher T (real physical precedent: adsorption-limited heterogeneous
        kinetics, where surface coverage falls as T rises, can show a net negative apparent
        activation energy). ``wall_no2_k_intrinsic`` keeps its calibrated meaning AT
        ``_WALL_NO2_T_REF_K``; the Arrhenius term only rescales the rate at OTHER temperatures
        relative to that.

        Optional ABSOLUTE-humidity gating (``wall_no2_h2o_ppm_ref`` > 0): replaces the relative-
        dew-point ``wet_factor`` above with a plain Langmuir adsorption isotherm in the ACTUAL
        H2O ppm present (``h2o_ppm/(h2o_ppm+ref)``), independent of the (T-dependent) dew point --
        physically, a real Langmuir water-adsorption model is driven by the absolute concentration
        of water molecules available to adsorb, not a dimensionless relative-humidity ratio.
        0.0 (default) keeps the relative-dew-point behaviour.

        Optional cumulative-exposure INDUCTION PERIOD (``wall_no2_exposure_threshold_ppm_h`` >
        0): a real corrosion phenomenon -- a passive oxide film only breaks down (exposing fresh,
        reactive bare Fe) after SUSTAINED aggressive-species exposure, not instantaneously (e.g.
        pitting-corrosion incubation times in the literature). Gated by a Hill/saturating
        function of ``cum_no2_exposure`` (ppm-hours of NO2 the wall has ever been exposed to,
        tracked as the ``CumNO2Exposure`` ODE state, see ``EXTRA_STATE_KEYS``): ``activation =
        ratio**n / (1+ratio**n)`` where ``ratio = cum_ppm_h / wall_no2_exposure_threshold_ppm_h``
        and ``n = wall_no2_exposure_hill_n``. 0.0 (default) disables it (activation always 1.0,
        backward-compatible).

        Optional O2-DEPLETION gating (``wall_no2_o2_ref_ppm`` > 0): real carbon-steel corrosion
        follows genuinely different mechanisms/product speciation under O2-depleted (anaerobic-
        like) vs O2-rich (aerobic) conditions -- gates this reaction to favour the O2-scarce
        regime via a Hill function in O2 ppm: ``o2_gate = 1/(1+(O2_ppm/ref)**n)``, so it is near
        1 when O2 is nearly exhausted and falls toward 0 once O2 is abundant. 0.0 (default)
        disables it (o2_gate always 1.0, backward-compatible).

        Optional O2-PRESENCE gating (``wall_no2_o2_presence_ref_ppm`` > 0, see
        ``_o2_presence_gate``): the O2-DEPLETION gate just above is ALSO wide open during a
        genuinely zero-O2-FEED window (not just ordinary consumption-driven depletion), letting
        this reaction crash the standing NO2 pool and spike NO within a single brief or extended
        anoxic phase. This gate throttles that specific case via the FED (not instantaneous) O2
        level, independent of the depletion gate above. 0.0 (default) disables it (factor always
        1.0, backward-compatible).
        """
        if self.wall_area_m2 <= 0.0 or self.wall_no2_k_intrinsic <= 0.0:
            return 0.0
        if self.wall_no2_h2o_ppm_ref > 0.0:
            wet_factor = max(h2o_ppm, 0.0) / (max(h2o_ppm, 0.0) + self.wall_no2_h2o_ppm_ref)
        else:
            wet_factor = self._water_saturation_fraction(h2o_ppm)
        activation = 1.0
        if self.wall_no2_exposure_threshold_ppm_h > 0.0:
            cum_ppm_h = max(cum_no2_exposure, 0.0) / max(self.molar_density, 1e-9) * 1e6 / 3600.0
            ratio_n = (cum_ppm_h / self.wall_no2_exposure_threshold_ppm_h) ** self.wall_no2_exposure_hill_n
            activation = ratio_n / (1.0 + ratio_n)
        o2_gate = 1.0
        if self.wall_no2_o2_ref_ppm > 0.0 and C_O2 is not None:
            # Reads the LAGGED O2 signal when enabled (see o2_lag_tau_hours), same reasoning
            # as R15's own product-inhibition gate above.
            o2_source = C_O2_lagged if (self.o2_lag_tau_hours > 0.0 and C_O2_lagged is not None) else C_O2
            o2_ppm = max(o2_source, 0.0) / max(self.molar_density, 1e-9) * 1e6
            ratio_n = (o2_ppm / self.wall_no2_o2_ref_ppm) ** self.wall_no2_o2_hill_n
            o2_gate = 1.0 / (1.0 + ratio_n)
        arrhenius_factor = 1.0
        if self.wall_no2_ea_kj_mol != 0.0:
            ea_j = self.wall_no2_ea_kj_mol * 1000.0
            arrhenius_factor = np.exp(-ea_j / (R_GAS * self.T) + ea_j / (R_GAS * _WALL_NO2_T_REF_K))
        gas_phase = self._wall_gas_phase_enhancement(self.wall_gas_phase_gain)
        sulfur_gate = self._sulfur_catalyst_gate(C_H2S_raw, C_H2SO4_raw)
        passivation = self._feed_o2_passivation(C_O2_feed, self.wall_no2_feed_o2_ref_ppm,
                                                self.wall_no2_feed_o2_hill_n)
        presence = self._o2_presence_gate(C_O2_feed, self.wall_no2_o2_presence_ref_ppm,
                                          self.wall_no2_o2_presence_hill_n)
        # NO product brake (see wall_no2_no_cap_ppm docstring): mirrors R15's N2O brake --
        # throttles this reaction as standing NO approaches/exceeds a chosen ceiling, so it
        # cannot indefinitely keep converting NO2 into MORE NO once NO is already abundant.
        no_brake = 1.0
        if self.wall_no2_no_cap_ppm > 0.0 and C_NO is not None:
            no_ppm_wall = max(C_NO, 0.0) / max(self.molar_density, 1e-9) * 1e6
            no_brake = 1.0 / (1.0 + (no_ppm_wall / self.wall_no2_no_cap_ppm) ** self.wall_no2_no_cap_hill_n)
        no2_ppm = max(C_NO2, 0.0) / max(self.molar_density, 1e-9) * 1e6
        # Langmuir adsorption isotherm for NO2 on the coupon surface (see
        # wall_no2_langmuir_half_ppm docstring): a finite number of active sites means the
        # attack rate must saturate at high NO2, not keep scaling with the power law forever.
        # `half_ppm * theta` recovers the plain (potency=1) linear term at low NO2 (theta<<1,
        # so this is a strict generalisation, not a competing knob) and flattens to a hard
        # ceiling of `half_ppm` once NO2 >> half_ppm, instead of growing unboundedly.
        if self.wall_no2_langmuir_half_ppm > 0.0:
            no2_term = self.wall_no2_langmuir_half_ppm * no2_ppm / (no2_ppm + self.wall_no2_langmuir_half_ppm)
        else:
            no2_term = no2_ppm ** self.wall_no2_potency
        V_m3 = max(self.volume_ml, 1e-9) * 1e-6
        A_sv = self.wall_area_m2 / V_m3
        return no_brake * self.wall_no2_k_intrinsic * wet_factor * arrhenius_factor * activation * o2_gate * \
            gas_phase * sulfur_gate * passivation * presence * no2_term * A_sv * 1e-3

    def _wall_so2_rate(self, C_SO2, h2o_ppm, cum_o2_exposure=0.0):
        """Surface-catalysed SO2 oxidation sink [kmol SO2/(m^3 s)]: SO2 + 0.5 O2 + H2O -> H2SO4
        (same net stoichiometry as the homogeneous R1, but catalysed by an accumulating surface
        oxide layer -- iron oxide is a real, if weak, industrial SO2-oxidation catalyst, the
        same chemistry underlying the "contact process" for sulfuric acid manufacture).

        Gated by an induction-period Hill function of CUMULATIVE O2 EXPOSURE (ppm-hours the
        wall has ever seen, tracked as the ``CumO2Exposure`` ODE state -- mirrors
        ``wall_no2_exposure_threshold_ppm_h``'s exact pattern): a genuinely built-up oxide layer,
        not instantaneous O2 presence, is what's understood to catalyse this. Water-saturation
        gated like the other acid-forming wall paths (``_water_saturation_fraction``).
        ``wall_so2_k_intrinsic <= 0`` (default) disables it entirely.
        """
        if self.wall_area_m2 <= 0.0 or self.wall_so2_k_intrinsic <= 0.0:
            return 0.0
        wet_factor = self._water_saturation_fraction(h2o_ppm)
        activation = 1.0
        if self.wall_so2_exposure_threshold_ppm_h > 0.0:
            cum_ppm_h = max(cum_o2_exposure, 0.0) / max(self.molar_density, 1e-9) * 1e6 / 3600.0
            ratio_n = (cum_ppm_h / self.wall_so2_exposure_threshold_ppm_h) ** self.wall_so2_exposure_hill_n
            activation = ratio_n / (1.0 + ratio_n)
        so2_ppm = max(C_SO2, 0.0) / max(self.molar_density, 1e-9) * 1e6
        V_m3 = max(self.volume_ml, 1e-9) * 1e-6
        A_sv = self.wall_area_m2 / V_m3
        return self.wall_so2_k_intrinsic * wet_factor * activation * \
            (so2_ppm ** self.wall_so2_potency) * A_sv * 1e-3

    def _wall_s8_rate(self, C_H2S, C_O2):
        """Carbon-steel-catalysed Claus-type surface reaction [kmol H2S/(m^3 s)]:
        8 H2S + 4 O2 -> S8 + 8 H2O.

        Requires the steel coupon as a catalyst (negligible without a carbon-steel/magnetite
        wall present) and is not gated by water -- a genuinely dry, catalytic gas-surface
        reaction, unlike the acid-film corrosion paths above. Guessed as kinetically slow (see
        ``wall_s8_k_intrinsic``).
        """
        if self.wall_area_m2 <= 0.0 or self.wall_s8_k_intrinsic <= 0.0:
            return 0.0
        if self.material not in ('carbon_steel', 'magnetite'):
            return 0.0
        h2s_ppm = max(C_H2S, 0.0) / max(self.molar_density, 1e-9) * 1e6
        o2_ppm = max(C_O2, 0.0) / max(self.molar_density, 1e-9) * 1e6
        V_m3 = max(self.volume_ml, 1e-9) * 1e-6
        A_sv = self.wall_area_m2 / V_m3
        return self.wall_s8_k_intrinsic * (h2s_ppm ** self.wall_s8_h2s_potency) * \
            (o2_ppm ** self.wall_s8_o2_potency) * A_sv * 1e-3

    def get_wall_deposit_rates(self, C_O2, h2o_ppm, C_NO2, C_H2SO4, C_HNO3, C_H2S=0.0,
                                cum_no2_exposure=0.0, C_O2_feed=None, C_O2_lagged=None,
                                C_SO2=0.0, cum_o2_exposure=0.0, C_NO=None, C_O2_feed_lagged=None):
        """Instantaneous wall-corrosion product formation rates [kmol/(m^3 s)], all coupled
        back into the species ODEs in ``rhs``: ``r_wall_o2`` (O2 -> Fe2O3, gated by the
        instantaneous NO2+H2SO4+HNO3 loading via ``C_NO2``/``C_H2SO4``/``C_HNO3``),
        ``r_wall_no2`` (2 Fe + 3 NO2 -> Fe2O3 + 3 NO, see ``_wall_no2_rate``),
        ``r_hno3_corrosion`` (HNO3 -> Fe(NO3)2) and ``r_h2so4`` (H2SO4 -> FeSO4,
        hydrogen-producing) are the four ACTIVE corrosion paths. ``r_feco3`` (CO2(aq) ->
        FeCO3) is disabled by default (see the module-level parameter-set docstring) and
        returns 0.0 unless explicitly re-enabled. ``r_wall_s8``
        (H2S + O2 -> S8 + H2O, carbon-steel-catalysed, no Fe consumed) is a separate catalytic
        (non-corrosion) path.

        ``C_O2_feed`` (fed, not instantaneous, O2 concentration) forwards to ``_wall_o2_rate``/
        ``_wall_no2_rate``'s feed-O2 passivation gate (see ``_feed_o2_passivation``) -- omitting
        it (``None``, the default) leaves that gate fully OPEN, matching its own no-op default,
        so callers that don't track a feed schedule (e.g. a single fixed-feed calculation) are
        unaffected; callers reconstructing a real multi-phase run (see
        ``_compute_reaction_rate_series``) MUST pass it to stay consistent with ``rhs()``.
        """
        return {
            'r_wall_o2': self._wall_o2_rate(C_O2, h2o_ppm, C_NO2, C_H2SO4, C_HNO3,
                                            C_O2_feed=(C_O2_feed_lagged if self.o2_feed_lag_tau_hours > 0.0
                                                       else C_O2_feed)),
            'r_feco3': self._wall_feco3_rate(h2o_ppm, C_H2SO4, C_HNO3),
            'r_hno3_corrosion': self._wall_hno3_corrosion_rate(h2o_ppm, C_HNO3, C_H2SO4),
            'r_h2so4': self._wall_h2so4_rate(h2o_ppm, C_H2SO4, C_HNO3),
            'r_wall_no2': self._wall_no2_rate(C_NO2, h2o_ppm, cum_no2_exposure, C_O2=C_O2,
                                              C_H2S_raw=C_H2S, C_H2SO4_raw=C_H2SO4,
                                              C_O2_feed=C_O2_feed, C_O2_lagged=C_O2_lagged,
                                              C_NO=C_NO),
            'r_wall_s8': self._wall_s8_rate(C_H2S, C_O2),
            'r_wall_so2': self._wall_so2_rate(C_SO2, h2o_ppm, cum_o2_exposure),
        }

    def set_reaction_constants(self, reaction_identifier, A_forward=None, Ea_forward_kJ_mol=None):
        rxn_id = None
        clean_id = str(reaction_identifier).strip().lower()

        if clean_id.upper() in self.kinetic_params:
            rxn_id = clean_id.upper()
        elif 'r3a' in clean_id or ('so2 + no2 + h2o' in clean_id and 'h2s' not in clean_id):
            rxn_id = 'R3a'
        elif 'r2' in clean_id or 'h2s + 3 no2' in clean_id:
            rxn_id = 'R2'
        elif 'r1' in clean_id or 'so2 + 0.5 o2' in clean_id:
            rxn_id = 'R1'
        elif 'r4' in clean_id or '2 no + o2' in clean_id:
            rxn_id = 'R4'
        elif 'r5' in clean_id or '3 no2 + h2o' in clean_id:
            rxn_id = 'R5'
        elif 'r7' in clean_id or '5 h2s + 6 no' in clean_id:
            rxn_id = 'R7'
        elif 'r13' in clean_id or 'h2so4 + 4 no' in clean_id:
            rxn_id = 'R13'
        elif 'r12' in clean_id or 'h2s + 2 o2' in clean_id:
            rxn_id = 'R12'

        if rxn_id and rxn_id in self.kinetic_params:
            if A_forward is not None:
                self.kinetic_params[rxn_id]['A'] = float(A_forward)
            if Ea_forward_kJ_mol is not None:
                self.kinetic_params[rxn_id]['Ea'] = float(Ea_forward_kJ_mol) * 1000.0

    def configure_wall_corrosion(self,
                                 area_m2=None,
                                 coupon_diameter_cm=None,
                                 coupon_thickness_mm=None,
                                 k_intrinsic=None,
                                 o2_potency=None,
                                 o2_sat_ref=None,
                                 rho_pass=None,
                                 hill_n=None,
                                 k_h2o_ppm=None,
                                 acid_exponent=None,
                                 acid_background=None,
                                 acid_gain=None,
                                 consume_h2o=None,
                                 h2o_mode=None,
                                 h2o_enhancement_factor=None,
                                 h2o_deliq_ref_ppm=None,
                                 h2o_hill_n=None,
                                 h2o_excess_ref_ppm=None,
                                 h2o_excess_exponent=None,
                                 feco3_k_intrinsic=None,
                                 feco3_potency=None,
                                 hno3_corrosion_k_intrinsic=None,
                                 hno3_corrosion_potency=None,
                                 h2so4_k_intrinsic=None,
                                 h2so4_potency=None,
                                 no2_k_intrinsic=None,
                                 no2_potency=None,
                                 no2_ea_kj_mol=None,
                                 no2_h2o_ppm_ref=None,
                                 no2_exposure_threshold_ppm_h=None,
                                 no2_exposure_hill_n=None,
                                 no2_o2_ref_ppm=None,
                                 no2_o2_hill_n=None,
                                 o2_gas_phase_gain=None,
                                 gas_phase_gain=None,
                                 gas_phase_rho_ref=None,
                                 gas_phase_hill_n=None,
                                 s8_k_intrinsic=None,
                                 s8_h2s_potency=None,
                                 s8_o2_potency=None,
                                 so2_k_intrinsic=None,
                                 so2_potency=None,
                                 so2_exposure_threshold_ppm_h=None,
                                 so2_exposure_hill_n=None,
                                 acid_gain2=None,
                                 acid_exponent2=None):
        """Configure the wall-corrosion O2 sink after construction.

        Provide either ``area_m2`` directly or a ``(coupon_diameter_cm, coupon_thickness_mm)``
        pair to compute the disc-coupon wetted area (2 faces + edge). All other arguments are
        optional overrides for the wall-model parameters.
        """
        if coupon_diameter_cm is not None and coupon_thickness_mm is not None:
            r = float(coupon_diameter_cm) * 0.5e-2
            h = float(coupon_thickness_mm) * 1e-3
            self.wall_area_m2 = 2.0 * np.pi * r * r + np.pi * (2.0 * r) * h
        elif area_m2 is not None:
            self.wall_area_m2 = float(area_m2)
        if k_intrinsic is not None:
            self.wall_k_intrinsic = float(k_intrinsic)
        if o2_potency is not None:
            self.wall_o2_potency = float(o2_potency)
        if o2_sat_ref is not None:
            self.wall_o2_sat_ref = float(o2_sat_ref)
        if rho_pass is not None:
            self.wall_rho_pass = float(rho_pass)
        if hill_n is not None:
            self.wall_hill_n = float(hill_n)
        if k_h2o_ppm is not None:
            self.wall_k_h2o_ppm = float(k_h2o_ppm)
        if acid_exponent is not None:
            self.wall_acid_exponent = float(acid_exponent)
        if acid_background is not None:
            self.wall_acid_background = float(acid_background)
        if acid_gain is not None:
            self.wall_acid_gain = float(acid_gain)
        if acid_gain2 is not None:
            self.wall_acid_gain2 = float(acid_gain2)
        if acid_exponent2 is not None:
            self.wall_acid_exponent2 = float(acid_exponent2)
        if consume_h2o is not None:
            self.wall_consume_h2o = bool(consume_h2o)
        if h2o_mode is not None:
            self.wall_h2o_mode = str(h2o_mode)
        if h2o_enhancement_factor is not None:
            self.wall_h2o_enhancement_factor = float(h2o_enhancement_factor)
        if h2o_deliq_ref_ppm is not None:
            self.wall_h2o_deliq_ref_ppm = float(h2o_deliq_ref_ppm)
        if h2o_hill_n is not None:
            self.wall_h2o_hill_n = float(h2o_hill_n)
        if h2o_excess_ref_ppm is not None:
            self.wall_h2o_excess_ref_ppm = float(h2o_excess_ref_ppm)
        if h2o_excess_exponent is not None:
            self.wall_h2o_excess_exponent = float(h2o_excess_exponent)
        if feco3_k_intrinsic is not None:
            self.wall_feco3_k_intrinsic = float(feco3_k_intrinsic)
        if feco3_potency is not None:
            self.wall_feco3_potency = float(feco3_potency)
        if hno3_corrosion_k_intrinsic is not None:
            self.wall_hno3_corrosion_k_intrinsic = float(hno3_corrosion_k_intrinsic)
        if hno3_corrosion_potency is not None:
            self.wall_hno3_corrosion_potency = float(hno3_corrosion_potency)
        if h2so4_k_intrinsic is not None:
            self.wall_h2so4_k_intrinsic = float(h2so4_k_intrinsic)
        if h2so4_potency is not None:
            self.wall_h2so4_potency = float(h2so4_potency)
        if no2_k_intrinsic is not None:
            self.wall_no2_k_intrinsic = float(no2_k_intrinsic)
        if no2_potency is not None:
            self.wall_no2_potency = float(no2_potency)
        if no2_ea_kj_mol is not None:
            self.wall_no2_ea_kj_mol = float(no2_ea_kj_mol)
        if no2_h2o_ppm_ref is not None:
            self.wall_no2_h2o_ppm_ref = float(no2_h2o_ppm_ref)
        if no2_exposure_threshold_ppm_h is not None:
            self.wall_no2_exposure_threshold_ppm_h = float(no2_exposure_threshold_ppm_h)
        if no2_exposure_hill_n is not None:
            self.wall_no2_exposure_hill_n = float(no2_exposure_hill_n)
        if no2_o2_ref_ppm is not None:
            self.wall_no2_o2_ref_ppm = float(no2_o2_ref_ppm)
        if no2_o2_hill_n is not None:
            self.wall_no2_o2_hill_n = float(no2_o2_hill_n)
        if o2_gas_phase_gain is not None:
            self.wall_o2_gas_phase_gain = float(o2_gas_phase_gain)
        if gas_phase_gain is not None:
            self.wall_gas_phase_gain = float(gas_phase_gain)
        if gas_phase_rho_ref is not None:
            self.wall_gas_phase_rho_ref = float(gas_phase_rho_ref)
        if gas_phase_hill_n is not None:
            self.wall_gas_phase_hill_n = float(gas_phase_hill_n)
        if s8_k_intrinsic is not None:
            self.wall_s8_k_intrinsic = float(s8_k_intrinsic)
        if s8_h2s_potency is not None:
            self.wall_s8_h2s_potency = float(s8_h2s_potency)
        if s8_o2_potency is not None:
            self.wall_s8_o2_potency = float(s8_o2_potency)
        if so2_k_intrinsic is not None:
            self.wall_so2_k_intrinsic = float(so2_k_intrinsic)
        if so2_potency is not None:
            self.wall_so2_potency = float(so2_potency)
        if so2_exposure_threshold_ppm_h is not None:
            self.wall_so2_exposure_threshold_ppm_h = float(so2_exposure_threshold_ppm_h)
        if so2_exposure_hill_n is not None:
            self.wall_so2_exposure_hill_n = float(so2_exposure_hill_n)
        self._wall_theta_pass = self._compute_wall_theta_pass()

    def set_phase_condensation(self, exponent, rho_m_reference=None):
        """Configure the phase-condensation multiplier for R2, R12 and R13.

        Set ``exponent = 0`` to disable (uniform f_phase = 1).
        """
        self.condensation_exponent = float(exponent)
        if rho_m_reference is not None:
            self.rho_m_reference = float(rho_m_reference)
        self._f_phase = self._compute_f_phase()

    def override_f_phase(self, value):
        """Directly set ``f_phase``, the wet-film/heterogeneous-reaction multiplier used ONLY
        by R2, R12 and R13 (the H2S-dependent NO2 reactions), bypassing the density-ratio
        formula in ``_compute_f_phase``.

        Use this (rather than ``set_srk_kij`` on NO2) to make NO2 react faster specifically via
        R2/R12/R13 when process-specific validation indicates that pathway is under-represented
        -- without also
        perturbing R3a/R4/R5, which all use NO2's bulk SRK fugacity too but have no H2S
        dependence and were not shown to need correcting. (Overriding NO2's SRK kij instead was
        tried and rejected: R5 is cubic in [NO2] and R4's reverse term is quadratic in [NO2], so
        a single shared NO2 fugacity blows both of those up far more than R2/R12/R13, wrongly
        consuming NO2 well before H2S is even fed. Boosting H2S or O2 instead was also tried and
        rejected: NO2 is the genuinely scarce reagent in R2/R12/R13, and the old (now-removed)
        R6/R8 paths provided an NO2-independent sink for H2S/O2, so they won the competition for
        the boosted H2S long before R2/R9 (R9's own successor, R12/R13) did -- NO2 itself has to
        become more available for R2/R12/R13 specifically.)
        This is a pure Python attribute assignment -- no NeqSim flash involved, so it is always
        safe to call, including repeatedly.
        """
        self._f_phase = float(value)

    def set_srk_kij(self, species_name, kij):
        """Override the CO2-``species_name`` binary interaction parameter in the SRK flash.

        Recomputes ``phi_dict``/``molar_density``/``phase``, the water-solubility limit and
        ``f_phase``, since all of them derive from the fugacity flash. Default kij is 0.0 for
        every pair (NeqSim's own database has no regressed value for these trace pairs), which
        for a strongly polar/reactive species like NO2 or SO2 in dense/liquid CO2 can produce
        an unrealistically small bulk-phase fugacity coefficient. This is a deliberate, tuned
        override rather than a measured value, and should only be used when process-specific
        validation indicates the bulk single-phase SRK flash is not representative. Prefer
        ``override_f_phase`` instead when only R2/R12/R13 (not every reaction using that
        species) should be affected -- see its docstring for why that matters for NO2
        specifically.
        """
        self.srk_kij_co2[species_name] = float(kij)
        self.molar_density, self.phase, self.phi_dict = self._calculate_srk_fugacities(self.T, self.P)
        self._water_solubility_ppm_base = self._calculate_water_solubility_ppm(self.T, self.P)
        self._f_phase = self._compute_f_phase()

    def set_conditions(self, temp_C=None, pressure_bar=None):
        """Re-evaluate the reactor state at a new temperature and/or pressure.

        Redoes the SRK flash, so ``molar_density`` (and therefore the CSTR residence time and
        every concentration-based rate law), ``phase``, ``phi_dict``, the water-solubility/dew
        point and ``f_phase`` all move with the new conditions. Arrhenius terms pick up the new
        ``self.T`` automatically. Deliberate ``set_phi_override``/``set_ideal_fugacity`` values
        are replayed afterwards so a condition change cannot silently revert them.

        No-op (and no NeqSim flash) when neither argument changes the current state.
        """
        new_T = self.T if temp_C is None else float(temp_C) + 273.15
        new_P = self.P if pressure_bar is None else float(pressure_bar)
        if new_T == self.T and new_P == self.P:
            return
        self.T = new_T
        self.P = new_P
        self.molar_density, self.phase, self.phi_dict = self._calculate_srk_fugacities(new_T, new_P)
        for species, value in self._phi_overrides.items():
            self.phi_dict[species] = value
        self._water_solubility_ppm_base = self._calculate_water_solubility_ppm(new_T, new_P)
        self._f_phase = self._compute_f_phase()

    def set_phi_override(self, species_name, value):
        """Force ``phi_dict[species_name]`` to ``value``, overriding whatever the SRK flash
        computed for it. Pure Python attribute set (no NeqSim flash), always safe to call.

        See ``set_ideal_fugacity`` for the ``value=1.0`` case. Values other than 1.0 are a tuned
        adjustment on top of that -- e.g. NO's SRK phi (~10 at -26C/31bar) makes R4 (2 NO + O2
        -> 2 NO2) pull essentially all the NO produced by R2/R12/R13 straight back to NO2 (see
        AGENTS notes); reducing phi_NO below 1 weakens that pull and lets a genuine steady-state
        NO residual persist, at the cost of being a tuned rather than a generally derived value.
        """
        self.phi_dict[species_name] = float(value)
        self._phi_overrides[species_name] = float(value)
        if species_name == 'H2O':
            self._water_solubility_ppm_base = self._calculate_water_solubility_ppm(self.T, self.P)

    def set_ideal_fugacity(self, species_name):
        """Force ``phi_dict[species_name]`` to 1.0 (ideal-mixture assumption), overriding
        whatever the SRK flash computed for it.

        The raw SRK flash can give NO2/HNO3 fugacity
        coefficients so extreme (~0.008 and ~0.0002 at -26C/31bar) that R5 (NO2 + H2O -> HNO3 +
        NO) makes literally zero progress in any realistic time, while a `kij` correction large
        enough to unstick the forward reaction leaves the reverse reaction still crippled by
        HNO3's own tiny phi and overshoots the real equilibrium several-fold (see
        `set_srk_kij` notes). Setting NO2/HNO3/H2O/NO to ideal (phi=1) instead reproduces the
        This is a simplified treatment for trace species and requires case-specific review.
        This is a pure Python attribute set (no NeqSim flash), always safe to call.
        """
        self.set_phi_override(species_name, 1.0)

    def set_r5_no_activity(self, value):
        """Set R5's reverse-term NO activity, decoupled from ``phi_dict['NO']`` (see
        ``r5_no_activity`` in ``__init__`` for the rationale). Pure Python attribute set,
        always safe to call.
        """
        self.r5_no_activity = float(value)

    def set_r4_no_activity(self, value):
        """Set R4's forward-term NO activity, decoupled from ``phi_dict['NO']`` (see
        ``r4_no_activity`` in ``__init__`` for the rationale). Pure Python attribute set,
        always safe to call.
        """
        self.r4_no_activity = float(value)

    def set_r4_surface_gain(self, value):
        """Set R4's surface-to-volume enhancement gain (see ``r4_surface_gain`` on ``__init__``).
        0.0 disables the term; it is 1.0 at the reference bore for any gain.
        """
        self.r4_surface_gain = float(value)

    def set_r3a_bore_gain(self, value):
        """Set R3a's surface-to-volume enhancement gain (see ``r3a_bore_gain`` on ``__init__``).
        Pure Python attribute set, always safe to call.
        """
        self.r3a_bore_gain = float(value)

    def set_r15_surface_suppress_gain(self, value):
        """Set R15's narrow-bore surface-quenching gain (see ``_r15_surface_suppression``).
        0.0 disables the term; it is 1.0 at the reference bore for any gain.
        """
        self.r15_surface_suppress_gain = float(value)

    def set_r15_sulfur_gate(self, ref_ppm=None, hill_n=None):
        """Set the shared sulfur-catalyst gate for R15 and the wall NO2 path (see
        ``_sulfur_catalyst_gate``). ``ref_ppm=0.0`` disables it.
        """
        if ref_ppm is not None:
            self.r15_sulfur_ref_ppm = float(ref_ppm)
        if hill_n is not None:
            self.r15_sulfur_hill_n = float(hill_n)

    def set_feed_o2_passivation(self, wall_o2_ref_ppm=None, wall_o2_hill_n=None,
                                wall_no2_ref_ppm=None, wall_no2_hill_n=None,
                                r3a_ref_ppm=None, r3a_hill_n=None, r3a_floor=None,
                                r3a_cap_ppm=None):
        """Set the feed-O2 passivation gates (see ``_feed_o2_passivation``). 0.0 disables each."""
        if wall_o2_ref_ppm is not None:
            self.wall_o2_feed_o2_ref_ppm = float(wall_o2_ref_ppm)
        if wall_o2_hill_n is not None:
            self.wall_o2_feed_o2_hill_n = float(wall_o2_hill_n)
        if wall_no2_ref_ppm is not None:
            self.wall_no2_feed_o2_ref_ppm = float(wall_no2_ref_ppm)
        if wall_no2_hill_n is not None:
            self.wall_no2_feed_o2_hill_n = float(wall_no2_hill_n)
        if r3a_ref_ppm is not None:
            self.r3a_feed_o2_ref_ppm = float(r3a_ref_ppm)
        if r3a_hill_n is not None:
            self.r3a_feed_o2_hill_n = float(r3a_hill_n)
        if r3a_floor is not None:
            self.r3a_feed_o2_floor = float(r3a_floor)
        if r3a_cap_ppm is not None:
            self.r3a_feed_o2_cap_ppm = float(r3a_cap_ppm)

    def set_o2_presence_gates(self, wall_no2_ref_ppm=None, wall_no2_hill_n=None,
                              r3a_ref_ppm=None, r3a_hill_n=None,
                              r2_ref_ppm=None, r2_hill_n=None):
        """Set the O2-presence gates (see ``_o2_presence_gate``). 0.0 disables each."""
        if wall_no2_ref_ppm is not None:
            self.wall_no2_o2_presence_ref_ppm = float(wall_no2_ref_ppm)
        if wall_no2_hill_n is not None:
            self.wall_no2_o2_presence_hill_n = float(wall_no2_hill_n)
        if r3a_ref_ppm is not None:
            self.r3a_o2_presence_ref_ppm = float(r3a_ref_ppm)
        if r3a_hill_n is not None:
            self.r3a_o2_presence_hill_n = float(r3a_hill_n)
        if r2_ref_ppm is not None:
            self.r2_o2_presence_ref_ppm = float(r2_ref_ppm)
        if r2_hill_n is not None:
            self.r2_o2_presence_hill_n = float(r2_hill_n)

    def set_wall_no2_no_cap(self, ppm=None, hill_n=None):
        """Set wall_no2's NO product brake (see ``wall_no2_no_cap_ppm`` docstring on
        ``__init__``). Pure Python attribute set, always safe to call.
        """
        if ppm is not None:
            self.wall_no2_no_cap_ppm = float(ppm)
        if hill_n is not None:
            self.wall_no2_no_cap_hill_n = float(hill_n)

    def set_wall_no2_langmuir(self, half_ppm):
        """Set wall_no2's NO2 Langmuir half-saturation ppm (see ``wall_no2_langmuir_half_ppm``
        docstring on ``__init__``). Pure Python attribute set, always safe to call.
        """
        self.wall_no2_langmuir_half_ppm = float(half_ppm)

    def _surface_sv_excess(self):
        """Fractional excess of the vessel surface-to-volume ratio over the reference bore.

        A/V for a cylinder is ``4/d + 2/L``, normalised to ``_R4_SURFACE_SV_REF_CM_INV`` (a
       reference autoclave). Exactly 0.0 at (or above) the reference bore, so
        every standard-bore rig is untouched by any term built on this; only a narrower bore
        gives a positive excess.
        """
        a_sv = 4.0 / max(self.diameter_cm, 1e-9) + 2.0 / max(self.length_cm, 1e-9)
        return max(0.0, a_sv / _R4_SURFACE_SV_REF_CM_INV - 1.0)

    def _r4_surface_factor(self):
        """Wall-film enhancement of R4 from the vessel surface-to-volume ratio (>=1.0)."""
        if self.r4_surface_gain <= 0.0:
            return 1.0
        return 1.0 + self.r4_surface_gain * self._surface_sv_excess()

    def _r3a_bore_factor(self):
        """Wall-film enhancement of R3a's base rate from the vessel surface-to-volume ratio
        (>=1.0). Same signal as ``_r4_surface_factor`` -- exactly 1.0 for every cm rig.
        """
        if self.r3a_bore_gain <= 0.0:
            return 1.0
        return 1.0 + self.r3a_bore_gain * self._surface_sv_excess()

    def _r15_surface_suppression(self):
        """Surface quenching of R15's N2O channel in a narrow bore (<=1.0).

        The 4 NO2 -> 2 N2O + 3 O2 route runs through an N2O4-like intermediate; a high wall area
        per unit volume gives that intermediate a competing heterogeneous fate instead, so the
        homogeneous N2O channel is throttled. Exactly 1.0 at the reference bore.
        """
        if self.r15_surface_suppress_gain <= 0.0:
            return 1.0
        return 1.0 / (1.0 + self.r15_surface_suppress_gain * self._surface_sv_excess())

    def _r3a_autocat_surface_suppression(self):
        """Bore-specific suppression of R3a's autocat (see ``r3a_autocat_surface_suppress_gain``
        docstring on ``__init__``): 1.0 (no-op) at the reference bore, falls below 1.0 only for
        a narrower bore.
        """
        if self.r3a_autocat_surface_suppress_gain <= 0.0:
            return 1.0
        return 1.0 / (1.0 + self.r3a_autocat_surface_suppress_gain * self._surface_sv_excess())

    def set_r3a_no_escape_frac(self, value):
        """Set the fraction of NO that "escapes" R3a's reverse term in the liquid/wet film,
        scaled by f_phase (see ``r3a_no_escape_frac`` in ``__init__`` for the rationale). Pure
        Python attribute set, always safe to call.
        """
        self.r3a_no_escape_frac = float(value)

    def set_r1_autocat(self, gain=None, ref_ppm=None):
        """Set R1's autocatalytic acceleration parameters (see ``r1_autocat_gain`` docstring on
        ``__init__``). Pure Python attribute set, always safe to call.
        """
        if gain is not None:
            self.r1_autocat_gain = float(gain)
        if ref_ppm is not None:
            self.r1_autocat_ref_ppm = float(ref_ppm)

    def set_r3a_autocat(self, gain=None, ref_ppm=None, surface_suppress_gain=None, hill_n=None):
        """Set R3a's autocatalytic acceleration parameters (see ``r3a_autocat_gain`` docstring
        on ``__init__``). Pure Python attribute set, always safe to call.
        """
        if gain is not None:
            self.r3a_autocat_gain = float(gain)
        if ref_ppm is not None:
            self.r3a_autocat_ref_ppm = float(ref_ppm)
        if hill_n is not None:
            self.r3a_autocat_hill_n = float(hill_n)
        if surface_suppress_gain is not None:
            self.r3a_autocat_surface_suppress_gain = float(surface_suppress_gain)

    def set_r12_density_independent(self, value):
        """Set whether R12 bypasses f_phase (see ``r12_density_independent`` docstring on
        ``__init__``). Pure Python attribute set, always safe to call.
        """
        self.r12_density_independent = bool(value)

    def set_r12_no2_order(self, value):
        """Set R12's NO2 catalytic-term order (see ``r12_no2_order`` docstring on ``__init__``).
        Pure Python attribute set, always safe to call.
        """
        self.r12_no2_order = float(value)

    def set_r13_no2_order(self, value):
        """Set R13's NO2 term order (see ``r13_no2_order`` docstring on ``__init__``). Pure
        Python attribute set, always safe to call.
        """
        self.r13_no2_order = float(value)

    def set_r15_f_phase_exponent(self, value):
        """Set R15's own f_phase exponent (see ``r15_f_phase_exponent`` docstring on
        ``__init__``). Pure Python attribute set, always safe to call.
        """
        self.r15_f_phase_exponent = float(value)

    def set_r15_o2_inhibition(self, ref_ppm=None, hill_n=None):
        """Set R15's O2 product-inhibition gate (see ``r15_o2_inhib_ref_ppm`` docstring on
        ``__init__``). Pure Python attribute set, always safe to call.
        """
        if ref_ppm is not None:
            self.r15_o2_inhib_ref_ppm = float(ref_ppm)
        if hill_n is not None:
            self.r15_o2_inhib_hill_n = float(hill_n)

    def set_r15_o2_activation(self, ref_ppm=None, hill_n=None):
        """Set R15's O2 activation gate (see ``r15_o2_activation_ref_ppm`` docstring on
        ``__init__``). Pure Python attribute set, always safe to call.
        """
        if ref_ppm is not None:
            self.r15_o2_activation_ref_ppm = float(ref_ppm)
        if hill_n is not None:
            self.r15_o2_activation_hill_n = float(hill_n)

    def set_r15_no2_cap(self, ppm=None, hill_n=None):
        """Set R15's NO2 Langmuir cap (see ``r15_no2_cap_ppm`` docstring on ``__init__``). Pure
        Python attribute set, always safe to call.
        """
        if ppm is not None:
            self.r15_no2_cap_ppm = float(ppm)
        if hill_n is not None:
            self.r15_no2_cap_hill_n = float(hill_n)

    def set_r15_n2o_cap(self, ppm=None, hill_n=None):
        """Set R15's N2O product brake (see ``r15_n2o_cap_ppm`` docstring on ``__init__``). Pure
        Python attribute set, always safe to call.
        """
        if ppm is not None:
            self.r15_n2o_cap_ppm = float(ppm)
        if hill_n is not None:
            self.r15_n2o_cap_hill_n = float(hill_n)

    def set_r15_o2_presence(self, ref_ppm=None, hill_n=None):
        """Set R15's O2-presence gate (see ``r15_o2_presence_ref_ppm`` docstring on
        ``__init__``). Pure Python attribute set, always safe to call.
        """
        if ref_ppm is not None:
            self.r15_o2_presence_ref_ppm = float(ref_ppm)
        if hill_n is not None:
            self.r15_o2_presence_hill_n = float(hill_n)

    def set_r11_o2_gate(self, ref_ppm=None, hill_n=None, gain=None):
        """Set R11's O2-abundance boost parameters (see ``r11_o2_ref_ppm`` docstring on
        ``__init__``). Pure Python attribute set, always safe to call.
        """
        if ref_ppm is not None:
            self.r11_o2_ref_ppm = float(ref_ppm)
        if hill_n is not None:
            self.r11_o2_hill_n = float(hill_n)
        if gain is not None:
            self.r11_o2_gain = float(gain)

    def set_r2_no2_boost(self, ref_ppm=None, hill_n=None, gain=None):
        """Set R2's NO2-abundance boost parameters (see ``r2_no2_boost_gain`` docstring on
        ``__init__``). Pure Python attribute set, always safe to call.
        """
        if ref_ppm is not None:
            self.r2_no2_boost_ref_ppm = float(ref_ppm)
        if hill_n is not None:
            self.r2_no2_boost_hill_n = float(hill_n)
        if gain is not None:
            self.r2_no2_boost_gain = float(gain)

    def set_o2_lag_tau_hours(self, value):
        """Set the LaggedO2 relaxation time constant in hours (see rhs() docstring). 0.0
        disables it (falls back to raw instantaneous O2 everywhere it's used).
        """
        self.o2_lag_tau_hours = float(value)

    def set_o2_feed_lag_tau_hours(self, value, rise_tau_hours=None):
        """Set the LaggedO2Feed asymmetric (fast-rise/slow-fall) relaxation time constants in
        hours (see rhs() docstring). ``value`` is the FALLING (feed-decreasing) tau; 0.0
        disables the whole mechanism (wall_o2 falls back to the raw, discontinuous feed step).
        """
        self.o2_feed_lag_tau_hours = float(value)
        if rise_tau_hours is not None:
            self.o2_feed_lag_rise_tau_hours = float(rise_tau_hours)

    def apply_calibrated_profile(self, profile='carbon_steel_wet_co2'):
        """Apply the general reference carbon-steel / wet-CO2 kinetics + wall-corrosion set.

        This is a single generic parameter set (see the module-level
        ``CARBON_STEEL_WET_CO2_KINETICS`` dict for the full list of constants and their meaning
        and units). It sets:
          - Homogeneous gas-phase Arrhenius parameters for R2, R3a, R4, R5, R7, R10, R11, R12 and
            R13 (R1 keeps its ``DEFAULT_KINETIC_PARAMS`` value -- deliberately uncalibrated/
            extremely slow, per the reaction-speed guidance this set is built from).
          - The phase-condensation multiplier for the heterogeneous R2/R12/R13 reactions.
          - The wall-corrosion mechanism (HNO3 -> Fe(NO3)2, H2SO4 -> FeSO4, and O2 -> Fe2O3),
            with the acid paths gated by the ``_water_saturation_fraction`` wetting index and
            conservative 1:1 Fe:solid-product stoichiometry. The O2 path is additionally gated by
            cumulative H2SO4/HNO3 ever produced (``CumH2SO4``/``CumHNO3``), tying rust formation
            to real acid-exposure history rather than an ungated background rate. A carbonic-acid
            path (CO2(aq) -> FeCO3, see ``_co2_aqueous_solubility_mol_l``) also exists in code
            but is disabled by default (``wall_feco3_k_intrinsic=0.0``).
          - NO2/HNO3/H2O treated as ideal fugacity (phi=1). The raw SRK fugacity coefficients
            for these species (generic corresponding-states critical-property correlations, no
            regressed CO2 kij) are extreme (phi~0.008/0.0002) at the cold/cryogenic conditions
            typical of cold trace-component applications -- an extrapolation artefact for strongly polar/
            associating species being modelled below their normal boiling points (NO2: 21C,
            HNO3: 83C), not a measured value. Ideal fugacity is itself an approximation, but is
            the one that reproduces an independent Gibbs-equilibrium estimate for a NO2/O2/H2O-
            in-CO2 system; no better-grounded number is available (checked: NeqSim's own
            GibbsReactor chemical-equilibrium solver does not reliably converge for this trace/
            cryogenic system even with adaptive-step/Armijo/regularised algorithms, and the
            regressed literature CO2/NO2 kij data, e.g. Camy et al. 2011, only covers
            concentrated (>=15 mol%) mixtures at 25-55C, not trace ppm at -26C).
          - NO's own fugacity coefficient fixed at 0.05 (tuned, not measured/derived): the raw
            SRK value (~10) lets R4 (2NO+O2->2NO2) recycle essentially all produced NO straight
            back to NO2, leaving no genuine NO to persist; 0.05 lets a real ~0.5ppm NO steady
            state emerge instead, matching observations.
          - R5's forward and reverse rates use the real Gibbs-free-energy-based Keq5 directly --
            no extra multiplier on top of it.
          - R5's reverse-term NO activity (``r5_no_activity``, see its docstring on ``__init__``)
            is decoupled from the shared/global NO fugacity above, so high-NO2/no-O2 feeds (no
            R4 activity) don't inherit R4's unrelated tuning and let HNO3 run away past its own
            Gibbs-equilibrium bound.

        It does not set reactor geometry, coupon area, or feed composition -- those are configured
        separately (e.g. via ``configure_wall_corrosion(coupon_diameter_cm=..., ...)``) and are the
        same regardless of which profile is applied. ``'carbon_steel_wet_co2'`` is the only profile.
        """
        if profile != 'carbon_steel_wet_co2':
            raise ValueError(f'Unknown profile: {profile}')

        for species in ('NO2', 'HNO3', 'H2O'):
            self.set_ideal_fugacity(species)
        self.set_phi_override('NO', 0.05)

        p = CARBON_STEEL_WET_CO2_KINETICS
        self.set_r5_no_activity(p['r5_no_activity'])
        self.set_r4_no_activity(p['r4_no_activity'])
        self.set_r4_surface_gain(p['r4_surface_gain'])
        self.set_r3a_bore_gain(p['r3a_bore_gain'])
        self.set_r15_surface_suppress_gain(p['r15_surface_suppress_gain'])
        self.set_r15_sulfur_gate(ref_ppm=p['r15_sulfur_ref_ppm'],
                                 hill_n=p['r15_sulfur_hill_n'])
        self.set_feed_o2_passivation(
            wall_o2_ref_ppm=p['wall_o2_feed_o2_ref_ppm'],
            wall_o2_hill_n=p['wall_o2_feed_o2_hill_n'],
            wall_no2_ref_ppm=p['wall_no2_feed_o2_ref_ppm'],
            wall_no2_hill_n=p['wall_no2_feed_o2_hill_n'],
            r3a_ref_ppm=p['r3a_feed_o2_ref_ppm'],
            r3a_hill_n=p['r3a_feed_o2_hill_n'],
            r3a_floor=p['r3a_feed_o2_floor'],
            r3a_cap_ppm=p['r3a_feed_o2_cap_ppm'])
        self.set_o2_presence_gates(
            wall_no2_ref_ppm=p['wall_no2_o2_presence_ref_ppm'],
            wall_no2_hill_n=p['wall_no2_o2_presence_hill_n'],
            r3a_ref_ppm=p['r3a_o2_presence_ref_ppm'],
            r3a_hill_n=p['r3a_o2_presence_hill_n'],
            r2_ref_ppm=p['r2_o2_presence_ref_ppm'],
            r2_hill_n=p['r2_o2_presence_hill_n'])
        self.set_wall_no2_no_cap(ppm=p['wall_no2_no_cap_ppm'], hill_n=p['wall_no2_no_cap_hill_n'])
        self.set_wall_no2_langmuir(p['wall_no2_langmuir_half_ppm'])
        self.set_r3a_no_escape_frac(p['r3a_no_escape_frac'])
        self.set_r1_autocat(gain=p['r1_autocat_gain'], ref_ppm=p['r1_autocat_ref_ppm'])
        self.set_r3a_autocat(gain=p['r3a_autocat_gain'], ref_ppm=p['r3a_autocat_ref_ppm'],
                             surface_suppress_gain=p['r3a_autocat_surface_suppress_gain'],
                             hill_n=p['r3a_autocat_hill_n'])
        self.set_r12_density_independent(p['r12_density_independent'])
        self.set_r12_no2_order(p['r12_no2_order'])
        self.set_r13_no2_order(p['r13_no2_order'])
        self.set_r15_f_phase_exponent(p['r15_f_phase_exponent'])
        self.set_r15_o2_inhibition(ref_ppm=p['r15_o2_inhib_ref_ppm'],
                                   hill_n=p['r15_o2_inhib_hill_n'])
        self.set_r15_o2_activation(ref_ppm=p['r15_o2_activation_ref_ppm'],
                                   hill_n=p['r15_o2_activation_hill_n'])
        self.set_r15_no2_cap(ppm=p['r15_no2_cap_ppm'], hill_n=p['r15_no2_cap_hill_n'])
        self.set_r15_n2o_cap(ppm=p['r15_n2o_cap_ppm'], hill_n=p['r15_n2o_cap_hill_n'])
        self.set_r15_o2_presence(ref_ppm=p['r15_o2_presence_ref_ppm'], hill_n=p['r15_o2_presence_hill_n'])
        self.set_r11_o2_gate(ref_ppm=p['r11_o2_ref_ppm'], hill_n=p['r11_o2_hill_n'], gain=p['r11_o2_gain'])
        self.set_r2_no2_boost(ref_ppm=p['r2_no2_boost_ref_ppm'], hill_n=p['r2_no2_boost_hill_n'],
                              gain=p['r2_no2_boost_gain'])
        self.set_o2_lag_tau_hours(p['o2_lag_tau_hours'])
        self.set_o2_feed_lag_tau_hours(p['o2_feed_lag_tau_hours'], rise_tau_hours=p['o2_feed_lag_rise_tau_hours'])
        for rxn_id in ('R2', 'R3a', 'R4', 'R5', 'R7', 'R10', 'R11', 'R12', 'R13', 'R15'):
            self.set_reaction_constants(rxn_id, A_forward=p[rxn_id]['A'],
                                         Ea_forward_kJ_mol=p[rxn_id]['Ea_kJ_mol'])
        self.set_phase_condensation(exponent=p['condensation_exponent'],
                                     rho_m_reference=p['rho_m_reference'])
        self.configure_wall_corrosion(
            k_intrinsic=p['wall_k_intrinsic'],
            o2_potency=p['wall_o2_potency'],
            o2_sat_ref=p['wall_o2_sat_ref'],
            rho_pass=p['wall_rho_pass'],
            hill_n=p['wall_hill_n'],
            acid_exponent=p['wall_acid_exponent'],
            acid_background=p['wall_acid_background'],
            acid_gain=p['wall_acid_gain'],
            acid_gain2=p['wall_acid_gain2'],
            acid_exponent2=p['wall_acid_exponent2'],
            consume_h2o=p['wall_consume_h2o'],
            h2o_mode=p['wall_h2o_mode'],
            h2o_enhancement_factor=p['wall_h2o_enhancement_factor'],
            h2o_deliq_ref_ppm=p['wall_h2o_deliq_ref_ppm'],
            h2o_hill_n=p['wall_h2o_hill_n'],
            h2o_excess_ref_ppm=p['wall_h2o_excess_ref_ppm'],
            h2o_excess_exponent=p['wall_h2o_excess_exponent'],
            feco3_k_intrinsic=p['wall_feco3_k_intrinsic'],
            feco3_potency=p['wall_feco3_potency'],
            hno3_corrosion_k_intrinsic=p['wall_hno3_corrosion_k_intrinsic'],
            hno3_corrosion_potency=p['wall_hno3_corrosion_potency'],
            h2so4_k_intrinsic=p['wall_h2so4_k_intrinsic'],
            h2so4_potency=p['wall_h2so4_potency'],
            no2_k_intrinsic=p['wall_no2_k_intrinsic'],
            no2_potency=p['wall_no2_potency'],
            no2_o2_ref_ppm=p['wall_no2_o2_ref_ppm'],
            no2_o2_hill_n=p['wall_no2_o2_hill_n'],
            o2_gas_phase_gain=p['wall_o2_gas_phase_gain'],
            gas_phase_gain=p['wall_gas_phase_gain'],
            gas_phase_rho_ref=p['wall_gas_phase_rho_ref'],
            gas_phase_hill_n=p['wall_gas_phase_hill_n'],
            s8_k_intrinsic=p['wall_s8_k_intrinsic'],
            s8_h2s_potency=p['wall_s8_h2s_potency'],
            s8_o2_potency=p['wall_s8_o2_potency'],
            so2_k_intrinsic=p['wall_so2_k_intrinsic'],
            so2_potency=p['wall_so2_potency'],
            so2_exposure_threshold_ppm_h=p['wall_so2_exposure_threshold_ppm_h'],
            so2_exposure_hill_n=p['wall_so2_exposure_hill_n'],
        )

    def set_reactor_geometry(self, diameter_cm=None, length_cm=None, volume_ml=None, mass_flow_g_h=None):
        if diameter_cm is not None:
            self.diameter_cm = float(diameter_cm)
        if mass_flow_g_h is not None:
            self.mass_flow_g_h = float(mass_flow_g_h)

        A_cross_cm2 = np.pi * (self.diameter_cm**2) / 4.0

        if volume_ml is not None:
            self.volume_ml = float(volume_ml)
            self.length_cm = self.volume_ml / A_cross_cm2
        elif length_cm is not None:
            self.length_cm = float(length_cm)
            self.volume_ml = A_cross_cm2 * self.length_cm

    def get_fluid_properties(self):
        return {
            'temperature_C': self.T - 273.15,
            'temperature_K': self.T,
            'pressure_bar': self.P,
            'phase': self.phase,
            'molar_density_kmol_m3': self.molar_density,
            'mass_density_kg_m3': self.molar_density * MW_CO2,
            'mass_density_g_ml': (self.molar_density * MW_CO2) * 1e-3,
            'phi_fugacities': self.phi_dict,
            'thermodynamic_backend': self.thermodynamic_backend,
        }

    def get_reaction_rates(self, moisture_ppm=10.0):
        return self._calculate_pure_physical_rate_constants(moisture_ppm)

    def get_reactor_geometry(self):
        A_cross_cm2 = np.pi * (self.diameter_cm**2) / 4.0
        rho_g_ml = (self.molar_density * MW_CO2) * 1e-3
        m_reactor_g = self.volume_ml * rho_g_ml
        tau_hours = m_reactor_g / self.mass_flow_g_h if self.mass_flow_g_h > 0 else 0.0

        return {
            'volume_ml': self.volume_ml,
            'volume_m3': self.volume_ml * 1e-6,
            'diameter_cm': self.diameter_cm,
            'diameter_m': self.diameter_cm * 1e-2,
            'cross_sectional_area_cm2': A_cross_cm2,
            'cross_sectional_area_m2': A_cross_cm2 * 1e-4,
            'length_cm': self.length_cm,
            'length_m': self.length_cm * 1e-2,
            'mass_flow_g_h': self.mass_flow_g_h,
            'inventory_mass_g': m_reactor_g,
            'residence_time_hours': tau_hours,
            'residence_time_seconds': tau_hours * 3600.0
        }

    def generate_reactor_report(self):
        geom = self.get_reactor_geometry()
        props = self.get_fluid_properties()

        report_lines = [
            "1. Reactor Geometry & Length (L) Derivation",
            f"Target Volume (V): {geom['volume_ml']:.1f} mL = {geom['volume_ml']:.1f} cm3 = {geom['volume_m3']:.1e} m3",
            f"Inner Diameter (D): {geom['diameter_cm']:.2f} cm = {geom['diameter_m']:.4f} m",
            f"Cross-Sectional Area (A_cross): A_cross = pi * D^2 / 4 = pi * ({geom['diameter_cm']:.2f} cm)^2 / 4 = {geom['cross_sectional_area_cm2']:.4f} cm2 ({geom['cross_sectional_area_m2']:.5e} m2)",
            f"Calculated Reactor Length (L): L = V / A_cross = {geom['volume_ml']:.1f} cm3 / {geom['cross_sectional_area_cm2']:.4f} cm2 = {geom['length_cm']:.4f} cm ({geom['length_m']:.6f} m)",
            "",
            f"2. Hydrodynamic Residence Time (tau) at {props['pressure_bar']:.1f} bar, {props['temperature_C']:.1f}°C",
            f"Fluid density from {props['thermodynamic_backend']}: {props['phase'].capitalize()} CO2 density rho = {props['mass_density_kg_m3']:.2f} kg/m3 (rho_m = {props['molar_density_kmol_m3']:.4f} kmol/m3).",
            f"Liquid Mass Inventory: m_reactor = {geom['volume_ml']:.1f} mL * {props['mass_density_g_ml']:.5f} g/mL = {geom['inventory_mass_g']:.2f} grams of {props['phase']} CO2.",
            f"Mass Flow Rate (m_dot): {geom['mass_flow_g_h']:.1f} g/h.",
            f"CSTR Residence Time (tau): tau = m_reactor / m_dot = {geom['inventory_mass_g']:.2f} g / {geom['mass_flow_g_h']:.1f} g/h = {geom['residence_time_hours']:.4f} HOURS ({geom['residence_time_seconds']:.1f} seconds)"
        ]

        return "\n".join(report_lines)

    def get_table_results(self, sim_results, resolution_hours=1.0):
        t_h = sim_results['time_hours']
        max_h = t_h[-1]
        target_hours = np.arange(0.0, max_h + resolution_hours/2.0, resolution_hours)

        rows = []
        for target in target_hours:
            row = {'Time (h)': round(float(target), 1)}
            for species, decimals in {
                'H2S': 2, 'SO2': 2, 'NO2': 2, 'NO': 4, 'O2': 2,
                'H2O': 2, 'H2SO4': 4, 'HNO3': 4, 'NH3': 4, 'S8': 4, 'N2O': 4
            }.items():
                value = np.interp(target, t_h, sim_results['ppm'][species])
                row[f'{species} (ppm)'] = round(max(0.0, float(value)), decimals)
            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def _calculate_srk_fugacities(self, T_K, P_bar):
        try:
            from neqsim.thermo.thermoTools import TPflash, fluid

            f = fluid("srk")
            f.setTemperature(T_K)
            f.setPressure(P_bar)

            f.addComponent("CO2", 0.99995)
            f.addComponent("H2S", 10.0e-6)
            f.addComponent("oxygen", 10.0e-6)
            f.addComponent("water", 10.0e-6)
            f.addComponent("ammonia", 10.0e-6)
            f.addComponent("S8", 10.0e-6)

            f.addComponent("SO2", 10.0e-6, 430.8, 78.84, 0.2454)
            f.addComponent("NO2", 10.0e-6, 431.4, 101.0, 0.834)
            f.addComponent("NO", 10.0e-6, 180.0, 64.8, 0.588)
            f.addComponent("H2SO4", 10.0e-6, 924.0, 64.0, 0.536)
            f.addComponent("HNO3", 10.0e-6, 520.0, 68.9, 0.714)

            f.setMixingRule("classic")

            if self.srk_kij_co2:
                # Component add order above fixes these indices (CO2 is always 0).
                component_index = {
                    'H2S': 1, 'O2': 2, 'H2O': 3, 'NH3': 4, 'S8': 5,
                    'SO2': 6, 'NO2': 7, 'NO': 8, 'H2SO4': 9, 'HNO3': 10,
                }
                mixing_rule = f.getPhase(0).getMixingRule()
                for species_name, kij in self.srk_kij_co2.items():
                    idx = component_index.get(species_name)
                    if idx is not None:
                        mixing_rule.setBinaryInteractionParameter(0, idx, float(kij))

            TPflash(f)

            phase = f.getPhase(0)
            phase_type = str(phase.getPhaseTypeName()).lower()
            if "gas" in phase_type or "vap" in phase_type:
                phase_name = "gas"
            else:
                phase_name = "liquid"

            density_kg_m3 = float(phase.getDensity())
            molar_mass_g_mol = float(phase.getMolarMass()) * 1000.0
            rho_m = density_kg_m3 / molar_mass_g_mol if molar_mass_g_mol > 0 else density_kg_m3 / 44.0095

            # Refine the bulk molar density with NeqSim's Span-Wagner reference equation of
            # state for CO2 -- materially more accurate than SRK for CO2 PVT behaviour (the
            # SRK mixture flash above is kept only for the trace-species fugacity
            # coefficients/phase determination, since Span-Wagner is pure-CO2-only and cannot
            # see the trace impurities). CO2 is 99.995 mol% of this mixture, so substituting
            # its own pure-component density for the bulk mixture density is a well-justified
            # approximation. Falls back silently to the SRK-derived rho_m above if this
            # secondary flash fails for any reason (e.g. outside Span-Wagner's valid range).
            backend_label = "NeqSim SRK EOS"
            try:
                f_sw = fluid("span-wagner", temperature=T_K, pressure=P_bar)
                TPflash(f_sw)
                phase_sw = f_sw.getPhase(0)
                sw_density_kg_m3 = float(phase_sw.getDensity())
                sw_molar_mass_g_mol = float(phase_sw.getMolarMass()) * 1000.0
                if sw_molar_mass_g_mol > 0:
                    rho_m = sw_density_kg_m3 / sw_molar_mass_g_mol
                    backend_label = "NeqSim SRK EOS (fugacities) + Span-Wagner (CO2 density)"
            except Exception as sw_error:
                warnings.warn(
                    f"NeqSim Span-Wagner density refinement failed ({sw_error}); "
                    "using the SRK mixture density instead.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            phi_dict = {}
            for i in range(phase.getNumberOfComponents()):
                comp = phase.getComponent(i)
                name = str(comp.getComponentName())
                phi = float(comp.getFugacityCoefficient())

                if name == "oxygen":
                    phi_dict["O2"] = phi
                elif name == "water":
                    phi_dict["H2O"] = phi
                elif name == "ammonia":
                    phi_dict["NH3"] = phi
                elif name in self.SPECIES:
                    phi_dict[name] = phi

            for s in self.SPECIES:
                if s not in phi_dict:
                    phi_dict[s] = 0.95 if phase_name == "liquid" else 0.65

            self.thermodynamic_backend = backend_label
            return max(rho_m, 0.05), phase_name, phi_dict

        except (ImportError, ModuleNotFoundError, RuntimeError, Exception) as error:
            warnings.warn(
                "NeqSim Python thermodynamics is unavailable; using the tutorial's "
                "screening density/fugacity correlation instead. "
                f"Original error: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            phi_dict = {}
            if T_K < T_CRIT_CO2_K:
                Tr = T_K / T_CRIT_CO2_K
                tau = 1.0 - Tr
                ln_Pr = (-7.06 * tau + 1.94 * (tau**1.5) - 1.64 * (tau**3) - 2.5 * (tau**4)) / Tr
                P_sat = P_CRIT_CO2_BAR * np.exp(ln_Pr)
            else:
                P_sat = P_CRIT_CO2_BAR

            if P_bar < P_sat:
                phase_name = "gas"
                Z = 0.75 + 0.15 * (T_K / 300.0) - 0.05 * (P_bar / 40.0)
                Z = max(min(Z, 0.95), 0.60)
                rho_kg_m3 = (P_bar * 1e5 * (MW_CO2 * 1e-3)) / (Z * R_GAS * T_K)
                phi_CO2 = np.exp(min(0.0, -0.15 * (P_bar / 30.0) * (298.15 / T_K)))
                for s in self.SPECIES:
                    phi_dict[s] = phi_CO2 * 0.65
            else:
                phase_name = "liquid"
                if T_K <= 250.0:
                    rho_kg_m3 = 1060.0 - 1.2 * (T_K - 240.0) + 1.5 * (P_bar - 20.0)
                else:
                    rho_kg_m3 = 820.0 + 2.5 * (P_bar - P_CRIT_CO2_BAR) - 4.0 * (T_K - T_CRIT_CO2_K)
                for s in self.SPECIES:
                    phi_dict[s] = 0.95

            rho_m = max(rho_kg_m3 / MW_CO2, 0.05)
            self.thermodynamic_backend = "illustrative screening correlation"
            return rho_m, phase_name, phi_dict

    def _calculate_pure_physical_rate_constants(self, moisture_ppm):
        T = self.T

        dG1 = DG_H2SO4_STDGIBBS - (DG_SO2_STDGIBBS + 0.5 * DG_O2_STDGIBBS + DG_H2O_STDGIBBS)
        Keq1 = max(np.exp(min(-dG1 / (R_GAS * T), MAX_KEQ_EXPONENT)), 1e-15)

        dG2 = (DG_SO2_STDGIBBS + DG_H2O_STDGIBBS + 3.0 * DG_NO_STDGIBBS) - (DG_H2S_STDGIBBS + 3.0 * DG_NO2_STDGIBBS)
        Keq2 = max(np.exp(min(-dG2 / (R_GAS * T), MAX_KEQ_EXPONENT)), 1e-15)

        dG3 = (DG_NO_STDGIBBS + DG_H2SO4_STDGIBBS) - (DG_SO2_STDGIBBS + DG_NO2_STDGIBBS + DG_H2O_STDGIBBS)
        Keq3 = max(np.exp(min(-dG3 / (R_GAS * T), MAX_KEQ_EXPONENT)), 1e-15)

        dG4 = (2.0 * DG_NO2_STDGIBBS) - (2.0 * DG_NO_STDGIBBS + DG_O2_STDGIBBS)
        Keq4 = max(np.exp(min(-dG4 / (R_GAS * T), MAX_KEQ_EXPONENT)), 1e-15)

        dG5 = (2.0 * DG_HNO3_STDGIBBS + DG_NO_STDGIBBS) - (3.0 * DG_NO2_STDGIBBS + DG_H2O_STDGIBBS)
        Keq5 = np.exp(-dG5 / (R_GAS * T))

        # R7: 5 H2S + 6 NO + 4 H2O -> 6 NH3 + 5 SO2. dG7 ~ -1000 kJ/mol at 298K (NO is a
        # high-energy species; reducing it to NH3 while oxidising H2S to SO2 is strongly
        # exergonic) -- Keq7 is astronomical, confirming the reverse reaction is genuinely
        # negligible rather than an assumed simplification. Clipped at MAX_KEQ_EXPONENT like
        # the other equilibria; a 6th/5th-order reverse term is not worth the numerical risk
        # for a rate that underflows to ~0 regardless, so it is computed here for transparency
        # only and is not wired into the forward-only r7 rate law in rhs().
        dG7 = (6.0 * DG_NH3_STDGIBBS + 5.0 * DG_SO2_STDGIBBS) - \
            (5.0 * DG_H2S_STDGIBBS + 6.0 * DG_NO_STDGIBBS + 4.0 * DG_H2O_STDGIBBS)
        Keq7 = np.exp(min(-dG7 / (R_GAS * T), MAX_KEQ_EXPONENT))

        # R12: H2S + 2 O2 -> H2SO4 (NO2-catalysed; NO2 appears in the rate law on both sides,
        # so it accelerates the approach to equilibrium without shifting Keq12 itself).
        dG12 = DG_H2SO4_STDGIBBS - (DG_H2S_STDGIBBS + 2.0 * DG_O2_STDGIBBS)
        Keq12 = max(np.exp(min(-dG12 / (R_GAS * T), MAX_KEQ_EXPONENT)), 1e-15)

        # R13: 4 NO2 + H2S -> H2SO4 + 4 NO
        dG13 = (DG_H2SO4_STDGIBBS + 4.0 * DG_NO_STDGIBBS) - (4.0 * DG_NO2_STDGIBBS + DG_H2S_STDGIBBS)
        Keq13 = max(np.exp(min(-dG13 / (R_GAS * T), MAX_KEQ_EXPONENT)), 1e-15)

        # R10: 4 NH3 + 4 NO + 3 O2 -> 4 N2O + 6 H2O. dG10 ~ -1300 kJ/mol at 298K (consuming 4
        # high-energy NO plus forming stable H2O dominates over N2O's own modest instability) --
        # Keq10 is astronomical (clipped at MAX_KEQ_EXPONENT), confirming the reverse is
        # genuinely negligible rather than an assumed simplification, same treatment as Keq7.
        dG10 = (4.0 * DG_N2O_STDGIBBS + 6.0 * DG_H2O_STDGIBBS) - \
            (4.0 * DG_NH3_STDGIBBS + 4.0 * DG_NO_STDGIBBS + 3.0 * DG_O2_STDGIBBS)
        Keq10 = max(np.exp(min(-dG10 / (R_GAS * T), MAX_KEQ_EXPONENT)), 1e-15)

        # R11: H2S + 2 NO -> N2O + 1/8 S8 + H2O. dG11 ~ -273 kJ/mol at 298K -> Keq11 ~ 4e47 --
        # large but NOT clipped at MAX_KEQ_EXPONENT (unlike Keq7/Keq10), i.e. a genuine finite
        # equilibrium constant, not an artefact of the exponential ceiling.
        dG11 = (DG_N2O_STDGIBBS + 0.125 * DG_S8_STDGIBBS + DG_H2O_STDGIBBS) - \
            (DG_H2S_STDGIBBS + 2.0 * DG_NO_STDGIBBS)
        Keq11 = max(np.exp(min(-dG11 / (R_GAS * T), MAX_KEQ_EXPONENT)), 1e-15)

        dG15 = (2.0 * DG_N2O_STDGIBBS + 3.0 * DG_O2_STDGIBBS) - (4.0 * DG_NO2_STDGIBBS)
        Keq15 = max(np.exp(min(-dG15 / (R_GAS * T), MAX_KEQ_EXPONENT)), 1e-15)

        p = self.kinetic_params
        k1_f = p['R1']['A'] * np.exp(-p['R1']['Ea'] / (R_GAS * T))
        k2_f = p['R2']['A'] * np.exp(-p['R2']['Ea'] / (R_GAS * T))
        k3a_f = p['R3a']['A'] * np.exp(-p['R3a']['Ea'] / (R_GAS * T))
        k4_f = p['R4']['A'] * np.exp(-p['R4']['Ea'] / (R_GAS * T)) if p['R4']['Ea'] > 0 else p['R4']['A'] * np.exp(530.0 / T)
        k5_f = p['R5']['A'] * np.exp(-p['R5']['Ea'] / (R_GAS * T))
        k7_f = p['R7']['A'] * np.exp(-p['R7']['Ea'] / (R_GAS * T))

        # R10: 4 NH3 + 4 NO + 3 O2 -> 4 N2O + 6 H2O (real Keq10 above; reverse wired in rhs())
        k10_f = p['R10']['A'] * np.exp(-p['R10']['Ea'] / (R_GAS * T))
        # R11: H2S + 2 NO -> N2O + 1/8 S8 + H2O (real Keq11 above; reverse wired in rhs())
        k11_f = p['R11']['A'] * np.exp(-p['R11']['Ea'] / (R_GAS * T))
        k12_f = p['R12']['A'] * np.exp(-p['R12']['Ea'] / (R_GAS * T))
        k13_f = p['R13']['A'] * np.exp(-p['R13']['Ea'] / (R_GAS * T))
        # R15: 4 NO2 -> 2 N2O + 3 O2 (real Keq15 above; both directions matter)
        k15_f = p['R15']['A'] * np.exp(-p['R15']['Ea'] / (R_GAS * T))

        k1_r = k1_f / Keq1 if Keq1 > 1e-15 else 0.0
        k2_r = k2_f / Keq2 if Keq2 > 1e-15 else 0.0
        k3a_r = k3a_f / Keq3 if Keq3 > 1e-15 else 0.0
        k4_r = k4_f / Keq4 if Keq4 > 1e-15 else 0.0
        k5_r = k5_f / Keq5
        k7_r = k7_f / Keq7  # ~0 in practice (Keq7 astronomical); see dG7 note above
        k10_r = k10_f / Keq10 if Keq10 > 1e-15 else 0.0  # ~0 in practice (Keq10 astronomical)
        k11_r = k11_f / Keq11 if Keq11 > 1e-15 else 0.0
        k12_r = k12_f / Keq12 if Keq12 > 1e-15 else 0.0
        k13_r = k13_f / Keq13 if Keq13 > 1e-15 else 0.0
        k15_r = k15_f / Keq15 if Keq15 > 1e-15 else 0.0

        safe_moisture_ppm = max(float(moisture_ppm), 0.0)
        moisture_factor = 0.25 + 0.75 * (1.0 - np.exp(-min(safe_moisture_ppm / MOISTURE_REF_PPM, 50.0)))
        k1_f *= moisture_factor
        k3a_f *= moisture_factor

        return {
            'k1_f': k1_f, 'k1_r': k1_r, 'Keq1': Keq1,
            'k2_f': k2_f, 'k2_r': k2_r, 'Keq2': Keq2,
            'k3a_f': k3a_f, 'k3a_r': k3a_r, 'Keq3': Keq3,
            'k4_f': k4_f, 'k4_r': k4_r, 'Keq4': Keq4,
            'k5_f': k5_f, 'k5_r': k5_r, 'Keq5': Keq5,
            'k7_f': k7_f, 'k7_r': k7_r, 'Keq7': Keq7,
            'k10_f': k10_f, 'k10_r': k10_r, 'Keq10': Keq10,
            'k11_f': k11_f, 'k11_r': k11_r, 'Keq11': Keq11,
            'k12_f': k12_f, 'k12_r': k12_r, 'Keq12': Keq12,
            'k13_f': k13_f, 'k13_r': k13_r, 'Keq13': Keq13,
            'k15_f': k15_f, 'k15_r': k15_r, 'Keq15': Keq15,
            'material': self.material,
            'moisture_factor': moisture_factor,
            'f_phase': self._f_phase,
        }

    def rhs(self, t, C, rates_dict, C_in=None, space_time_sec=None, inflow_only=False):
        n_species = len(self.SPECIES)
        C_raw = np.clip(C[:n_species], MIN_CONCENTRATION_FLOOR, 1e5 * self.molar_density)
        # Extra ODE states (appended after the gas species, see simulate()): cumulative solid
        # FeSO4/Fe(NO3)2 corrosion product, and cumulative (never-decreasing) total H2SO4/HNO3
        # ever produced [kmol/m^3] -- see EXTRA_STATE_KEYS. None have an inflow/outflow term, see
        # the dC_dt assembly at the bottom.
        n_extra = len(self.EXTRA_STATE_KEYS)
        C_wall_solid = np.maximum(0.0, C[n_species:]) if len(C) > n_species else np.zeros(n_extra)
        C_cum_h2so4 = C_wall_solid[2] if len(C_wall_solid) > 2 else 0.0
        C_cum_hno3 = C_wall_solid[3] if len(C_wall_solid) > 3 else 0.0
        C_cum_no2_exposure = C_wall_solid[4] if len(C_wall_solid) > 4 else 0.0
        C_lagged_o2 = C_wall_solid[5] if len(C_wall_solid) > 5 else 0.0
        C_cum_o2_exposure = C_wall_solid[6] if len(C_wall_solid) > 6 else 0.0
        C_lagged_o2_feed = C_wall_solid[7] if len(C_wall_solid) > 7 else 0.0

        phi = self.phi_dict
        C_H2S   = max(0.0, C_raw[0] * phi['H2S'])
        C_SO2   = max(0.0, C_raw[1] * phi['SO2'])
        C_NO2   = max(0.0, C_raw[2] * phi['NO2'])
        C_NO    = max(0.0, C_raw[3] * phi['NO'])
        C_O2    = max(0.0, C_raw[4] * phi['O2'])
        C_H2O   = max(0.0, C_raw[5] * phi['H2O'])
        C_H2SO4 = max(0.0, C_raw[6] * phi['H2SO4'])
        C_HNO3  = max(0.0, C_raw[7] * phi['HNO3'])
        C_S8    = max(0.0, C_raw[8] * phi['S8'])
        C_NH3   = max(0.0, C_raw[9] * phi['NH3'])
        C_N2O   = max(0.0, C_raw[10] * phi['N2O'])

        k1_f, k1_r   = rates_dict['k1_f'], rates_dict['k1_r']
        k2_f, k2_r   = rates_dict['k2_f'], rates_dict['k2_r']
        k3a_f, k3a_r = rates_dict['k3a_f'], rates_dict['k3a_r']
        k4_f, k4_r   = rates_dict['k4_f'], rates_dict['k4_r']
        k5_f, k5_r   = rates_dict['k5_f'], rates_dict['k5_r']
        k7_f         = rates_dict['k7_f']
        k10_f, k10_r = rates_dict.get('k10_f', 0.0), rates_dict.get('k10_r', 0.0)
        k11_f, k11_r = rates_dict.get('k11_f', 0.0), rates_dict.get('k11_r', 0.0)
        k12_f, k12_r = rates_dict.get('k12_f', 0.0), rates_dict.get('k12_r', 0.0)
        k13_f, k13_r = rates_dict.get('k13_f', 0.0), rates_dict.get('k13_r', 0.0)
        f_phase      = rates_dict.get('f_phase', self._f_phase)

        # Shared driver for both R1's and R3a's autocatalytic acceleration (see their
        # docstrings on __init__): TOTAL H2SO4 ever produced, not the current standing ppm.
        cum_h2so4_ppm = C_cum_h2so4 / max(self.molar_density, 1e-9) * 1e6

        r1 = k1_f * C_SO2 * (C_O2**0.5) * C_H2O - k1_r * C_H2SO4
        if self.r1_autocat_gain > 0.0:
            # Saturating (Langmuir-form) acceleration driven by TOTAL H2SO4 ever produced (not
            # the current, wall-consumable standing ppm) -- scales BOTH directions of R1
            # equally, so it speeds up the approach to equilibrium without shifting Keq1.
            r1_autocat = 1.0 + self.r1_autocat_gain * cum_h2so4_ppm / (self.r1_autocat_ref_ppm + cum_h2so4_ppm)
            r1 *= r1_autocat
        # R2, R12 and R13 are heterogeneous/wet-film reactions - scaled by f_phase
        # (constant per T,P; 1.0 when phase-condensation switch is disabled).
        C_O2_feed = C_in[4] if C_in is not None else None
        r2_no2_boost = 1.0
        if self.r2_no2_boost_ref_ppm > 0.0:
            no2_ppm_r2 = max(C_NO2, 0.0) / max(self.molar_density, 1e-9) * 1e6
            ratio_r2 = (no2_ppm_r2 / self.r2_no2_boost_ref_ppm) ** self.r2_no2_boost_hill_n
            r2_no2_boost = 1.0 + self.r2_no2_boost_gain * ratio_r2 / (1.0 + ratio_r2)
        r2 = r2_no2_boost * (k2_f * C_H2S * C_NO2 - k2_r * C_SO2 * C_H2O * (C_NO**3)) * f_phase \
            * self._o2_presence_gate(C_O2_feed, self.r2_o2_presence_ref_ppm,
                                     self.r2_o2_presence_hill_n)
        # R3a's reverse term uses a liquid/wet-film-scaled NO activity (see r3a_no_escape_frac
        # docstring): some of the NO produced escapes the film into bulk gas before it can drive
        # the reverse reaction, an effect that grows with f_phase (dense/liquid) and vanishes in
        # low-density gas phase -- genuinely shifts R3a's apparent equilibrium, unlike f_phase.
        # Multiplies the existing (already phi-scaled) C_NO, so frac=0.0 is an exact no-op.
        C_NO_r3a = C_NO * (1.0 - self.r3a_no_escape_frac * f_phase)
        r3a = self._r3a_bore_factor() * (k3a_f * C_SO2 * C_NO2 * C_H2O - k3a_r * C_NO_r3a * C_H2SO4) \
            * self._feed_o2_passivation(C_O2_feed, self.r3a_feed_o2_ref_ppm,
                                        self.r3a_feed_o2_hill_n,
                                        floor=self.r3a_feed_o2_floor,
                                        cap_ppm=self.r3a_feed_o2_cap_ppm) \
            * self._o2_presence_gate(C_O2_feed, self.r3a_o2_presence_ref_ppm,
                                     self.r3a_o2_presence_hill_n)
        if self.r3a_autocat_gain > 0.0:
            # Same mechanism as r1_autocat (see above), independently tunable gain/ref_ppm, plus
            # a bore-specific suppression that only ever engages at a narrower-than-reference
            # bore (see _r3a_autocat_surface_suppression docstring).
            cum_h2so4_n = cum_h2so4_ppm ** self.r3a_autocat_hill_n
            ref_ppm_n = self.r3a_autocat_ref_ppm ** self.r3a_autocat_hill_n
            r3a_autocat = 1.0 + self.r3a_autocat_gain * cum_h2so4_n / (ref_ppm_n + cum_h2so4_n)
            r3a_autocat = 1.0 + (r3a_autocat - 1.0) * self._r3a_autocat_surface_suppression()
            r3a *= r3a_autocat
        C_NO_r4 = max(0.0, C_raw[3] * self.r4_no_activity * self._r4_surface_factor())
        r4 = k4_f * (C_NO_r4**2) * C_O2 - k4_r * (C_NO2**2)
        # R5's reverse term uses its own decoupled NO activity (see r5_no_activity docstring)
        # instead of the shared phi['NO'], so it genuinely shifts R5's equilibrium point rather
        # than just changing how fast it is approached (unlike f_phase).
        C_NO_r5 = max(0.0, C_raw[3] * self.r5_no_activity)
        r5 = k5_f * (C_NO2**3) * C_H2O - k5_r * (C_HNO3**2) * C_NO_r5
        r7 = k7_f * C_H2S * C_NO * C_H2O
        # R10: 4 NH3 + 4 NO + 3 O2 -> 4 N2O + 6 H2O (N2O-selectivity side-reaction of NH3-SCR/
        # deNOx chemistry; a genuine dead-end for the NH3 that R7 produces from H2S + NO).
        # Reverse term uses the products' literal stoichiometric powers (N2O^4, H2O^6), matching
        # R2's convention (forward kept as the simplified/empirical 1st-order-each rate law, the
        # true kinetic mechanism; reverse enforces the real Keq10 -- see its docstring, clipped
        # astronomical, so this reverse term is negligible in practice but not simply omitted).
        r10 = k10_f * C_NH3 * C_NO * C_O2 - k10_r * (C_N2O**4) * (C_H2O**6)
        # R11: H2S + 2 NO -> N2O + 1/8 S8 + H2O (direct NO reduction by H2S, the "chemo-
        # denitrification" analogue of the historical R8's H2S + O2 -> S8 + H2O; main N2O
        # source since it draws on the abundant H2S/NO pool instead of the trace NH3 byproduct
        # of R7). A genuine homogeneous gas-phase bimolecular reaction (not a wet-film/
        # heterogeneous one like R2/R12/R13), so it is NOT scaled by f_phase -- it must stay a
        # real, density-independent NO sink or NO fails to recombine fast enough at
        # low-density gas-phase conditions. Reverse term uses the products' literal stoichiometric
        # powers (N2O^1, S8^(1/8), H2O^1); Keq11 is large but finite (~4e47, not clipped, see its
        # docstring), so this reverse term is genuinely negligible at any reachable concentration
        # rather than assumed away.
        r11_o2_boost = 1.0
        if self.r11_o2_ref_ppm > 0.0:
            o2_ppm_r11 = max(C_O2, 0.0) / max(self.molar_density, 1e-9) * 1e6
            ratio_n = (o2_ppm_r11 / self.r11_o2_ref_ppm) ** self.r11_o2_hill_n
            r11_o2_boost = 1.0 + self.r11_o2_gain * ratio_n / (1.0 + ratio_n)
        r11 = r11_o2_boost * k11_f * C_H2S * C_NO - k11_r * C_N2O * (C_S8**0.125) * C_H2O
        # R12: H2S + 2 O2 -> H2SO4, NO2-catalysed (NO2 is not consumed -- appears on both the
        # forward and reverse term, raised to r12_no2_order, so it speeds up the approach to
        # Keq12 without shifting it).
        C_NO2_r12 = C_NO2 ** self.r12_no2_order
        r12 = (k12_f * C_H2S * C_O2 * C_NO2_r12 - k12_r * C_H2SO4 * C_NO2_r12) \
            * (1.0 if self.r12_density_independent else f_phase)
        # R13: 4 NO2 + H2S -> H2SO4 + 4 NO
        C_NO2_r13 = C_NO2 ** self.r13_no2_order
        C_NO_r13 = C_NO ** self.r13_no2_order
        r13 = (k13_f * C_H2S * C_NO2_r13 - k13_r * C_H2SO4 * C_NO_r13) * f_phase
        # R15: 4 NO2 -> 2 N2O + 3 O2 -- a pure gas-phase NO2 decomposition, no H2S/NO needed, so
        # it doesn't compete with R2/R11/R12/R13 for those shared reagents. f_phase-scaled (like
        # R2/R12/R13): real NOx disproportionation is understood to be surface/radical-chain
        # mediated, favoured in a denser, more condensed-film-like environment, not simple gas-
        # phase pyrolysis. Reverse term uses literal powers.
        k15_f, k15_r = rates_dict.get('k15_f', 0.0), rates_dict.get('k15_r', 0.0)
        f_phase_r15 = f_phase ** self.r15_f_phase_exponent
        # O2 product-inhibition gate (see r15_o2_inhib_ref_ppm docstring): applied to the
        # FORWARD term only, so it throttles how fast R15 runs without touching Keq15 (the
        # reverse term still enforces the real equilibrium). Reads the LAGGED O2 signal when
        # enabled (see o2_lag_tau_hours) instead of raw instantaneous O2: a brief interruption
        # (much shorter than tau) barely moves the lagged value, so the gate does not swing
        # wide open for a momentary pulse; a genuinely extended depletion (much longer than tau)
        # still reaches it normally.
        r15_o2_inhib = 1.0
        if self.r15_o2_inhib_ref_ppm > 0.0:
            o2_source = C_lagged_o2 if self.o2_lag_tau_hours > 0.0 else C_raw[4]
            o2_ppm_r15 = max(o2_source, 0.0) / max(self.molar_density, 1e-9) * 1e6
            r15_o2_inhib = 1.0 / (1.0 + (o2_ppm_r15 / self.r15_o2_inhib_ref_ppm)
                                  ** self.r15_o2_inhib_hill_n)
        # O2 ACTIVATION gate (see r15_o2_activation_ref_ppm docstring): the inhibition gate
        # above is permissive AT O2=0 by construction (nothing left to inhibit with), so it
        # cannot stop R15 during a genuinely anoxic window -- this gate puts O2 directly into
        # the rate as something the reaction NEEDS to proceed at all, the opposite shape,
        # so it is exactly 0 the instant instantaneous O2 hits 0, regardless of NO2. 0.0
        # (default) disables it (exact no-op, backward-compatible).
        r15_o2_activation = 1.0
        if self.r15_o2_activation_ref_ppm > 0.0:
            o2_ppm_act = max(C_raw[4], 0.0) / max(self.molar_density, 1e-9) * 1e6
            ratio_act = (o2_ppm_act / self.r15_o2_activation_ref_ppm) ** self.r15_o2_activation_hill_n \
                if o2_ppm_act > 0.0 else 0.0
            r15_o2_activation = ratio_act / (1.0 + ratio_act)
        # NO2 Langmuir cap (see r15_no2_cap_ppm docstring): saturates the EFFECTIVE NO2 feeding
        # the forward term only, so a very high standing NO2 (with r15_o2_inhib's gate wide open
        # during a genuinely zero-O2-feed window) cannot drive an unbounded NO2^4 forward rate.
        C_NO2_r15 = C_NO2
        if self.r15_no2_cap_ppm > 0.0:
            no2_ppm_r15 = max(C_raw[2], 0.0) / max(self.molar_density, 1e-9) * 1e6
            if no2_ppm_r15 > 0.0:
                ratio_n = (no2_ppm_r15 / self.r15_no2_cap_ppm) ** self.r15_no2_cap_hill_n
                no2_ppm_r15_capped = self.r15_no2_cap_ppm * ratio_n / (1.0 + ratio_n)
            else:
                no2_ppm_r15_capped = 0.0
            C_NO2_r15 = no2_ppm_r15_capped * 1e-6 * self.molar_density * phi['NO2']
        # N2O product brake (see r15_n2o_cap_ppm docstring): applied to the FORWARD term only,
        # gated on the STANDING N2O concentration itself -- unlike r15_o2_inhib (needs O2) and
        # the reverse/equilibrium term (ALSO needs O2, since O2 is a reactant of the reverse
        # reaction), this brake still works during a genuinely zero-O2 window, when neither of
        # those other two brakes can act at all.
        r15_n2o_brake = 1.0
        if self.r15_n2o_cap_ppm > 0.0:
            n2o_ppm_r15 = max(C_raw[10], 0.0) / max(self.molar_density, 1e-9) * 1e6
            r15_n2o_brake = 1.0 / (1.0 + (n2o_ppm_r15 / self.r15_n2o_cap_ppm)
                                   ** self.r15_n2o_cap_hill_n)
        # O2-PRESENCE gate (see _o2_presence_gate docstring): reads FED (not instantaneous) O2
        # via C_O2_feed, already computed above for R3a's own feed gate.
        r15_o2_presence = self._o2_presence_gate(C_O2_feed, self.r15_o2_presence_ref_ppm,
                                                 self.r15_o2_presence_hill_n)
        r15 = (r15_o2_inhib * r15_o2_activation * r15_n2o_brake * self._r15_surface_suppression()
               * self._sulfur_catalyst_gate(C_raw[0], C_raw[6]) * k15_f * (C_NO2_r15**4)
               - k15_r * (C_N2O**2) * (C_O2**3)) * f_phase_r15 * r15_o2_presence

        R_H2S   = - r2 - 5.0 * r7 - r11 - r12 - r13
        R_SO2   = - r1 + r2 - r3a + 5.0 * r7
        R_NO2   = - 3.0 * r2 - r3a + 2.0 * r4 - 3.0 * r5 - 4.0 * r13 - 4.0 * r15
        R_NO    = + 3.0 * r2 + r3a - 2.0 * r4 + r5 - 6.0 * r7 - 4.0 * r10 - 2.0 * r11 + 4.0 * r13
        R_O2    = - 0.5 * r1 - r4 - 3.0 * r10 - 2.0 * r12 + 3.0 * r15
        R_H2O   = - r1 + r2 - r3a - r5 - 4.0 * r7 + 6.0 * r10 + r11
        R_H2SO4 = + r1 + r3a + r12 + r13
        R_HNO3  = + 2.0 * r5
        R_S8    = + 0.125 * r11
        R_NH3   = + 6.0 * r7 - 4.0 * r10
        R_N2O   = + 4.0 * r10 + r11 + 2.0 * r15
        R_H2    = 0.0

        # Cumulative "total ever produced" trackers (see EXTRA_STATE_KEYS): only the forward,
        # acid-forming contribution counts (never negative), so a reversible reaction's own
        # back-reaction does not erase history already counted, and neither does the wall
        # reactions' subsequent consumption of the actual gas-phase H2SO4/HNO3 pool.
        cum_h2so4_rate = max(0.0, r1) + max(0.0, r3a) + max(0.0, r12) + max(0.0, r13)
        cum_hno3_rate = max(0.0, r5)
        # R7 (5 H2S+6 NO+4 H2O->6 NH3+5 SO2) is the only NH3-producing reaction and is already
        # forward-only (no reverse term wired into rhs(), see Keq7 docstring above) -- max(0, ..)
        # kept anyway for the same defensive-consistency reason as the other Cum* trackers.
        # R10 (4 NH3+4 NO+3 O2->4 N2O+6 H2O) genuinely consumes already-formed NH3 afterwards,
        # but -- exactly like wall corrosion consuming the actual H2SO4/HNO3 pool above -- that
        # consumption must NOT erase this "ever produced" history.
        cum_nh3_rate = 6.0 * max(0.0, r7)

        # Symmetric first-order relaxation of instantaneous O2 (see o2_lag_tau_hours docstring
        # on __init__): tracks C_raw[4] with a lag, used ONLY as an alternate input to R15's/
        # wall_no2's own low-O2-favoured gates so a BRIEF O2 interruption/restoration (much
        # shorter than tau) does not fully swing those gates open, while a genuinely extended
        # depletion (much longer than tau) still reaches them normally. 0.0 (default) disables
        # it (state stays frozen at 0, unused -- the gates fall back to raw instantaneous O2).
        lagged_o2_rate = 0.0
        if self.o2_lag_tau_hours > 0.0:
            lagged_o2_rate = (C_raw[4] - C_lagged_o2) / (self.o2_lag_tau_hours * 3600.0)

        # ASYMMETRIC first-order relaxation of the FED (not instantaneous) O2 concentration
        # (see o2_feed_lag_tau_hours docstring): tracks C_in[4] itself, used ONLY as an
        # alternate input to wall_o2's OWN feed-passivation gate. A discrete feed step (e.g. a
        # brief "O2 stopped" phase) currently flips that gate fully open INSTANTLY, letting
        # wall_o2 devour the O2 still physically present in the vessel far faster than dilution
        # alone -- a real passive oxide film built up over a long high-O2 exposure would not
        # vanish the instant the feed valve closes. Fast rise (o2_feed_lag_rise_tau_hours) when
        # feed is increasing, slow fall (o2_feed_lag_tau_hours) when decreasing -- deliberately
        # asymmetric, unlike the symmetric LaggedO2 state above. 0.0 (default) disables it
        # (state frozen at 0, unused -- wall_o2 falls back to the raw instantaneous feed step).
        lagged_o2_feed_rate = 0.0
        if self.o2_feed_lag_tau_hours > 0.0 and C_O2_feed is not None:
            feed_now = max(C_O2_feed, 0.0)
            rising = feed_now > C_lagged_o2_feed
            tau_hours = self.o2_feed_lag_rise_tau_hours if rising else self.o2_feed_lag_tau_hours
            lagged_o2_feed_rate = (feed_now - C_lagged_o2_feed) / (max(tau_hours, 1e-9) * 3600.0)

        # Wall corrosion sinks (HNO3->Fe(NO3)2, H2SO4->FeSO4, O2->Fe2O3, NO2->Fe2O3).
        # Only fires when wall_area_m2 > 0.
        r_hno3corr = 0.0
        r_h2so4 = 0.0
        cum_no2_rate = 0.0
        cum_o2_rate = 0.0
        r_wall_so2 = 0.0
        if self.wall_area_m2 > 0.0:
            # Wall/corrosion severity is driven by how much of each species is physically
            # present (mole-fraction ppm), NOT by the phi-scaled "reactive availability"
            # concentration used for the homogeneous gas reactions above -- those are two
            # different physical questions (see AGENTS notes: overriding a species' SRK phi
            # for gas kinetics purposes must not silently rescale the wall-corrosion
            # calibration, which was fit against the raw amount present).
            h2o_ppm_here = C_raw[5] / max(self.molar_density, 1e-9) * 1e6
            C_NO2_wall = C_raw[2]
            C_H2SO4_wall = C_raw[6]
            C_HNO3_wall = C_raw[7]
            C_SO2_wall = C_raw[1]
            # O2 exposure accumulates regardless of whether wall_so2 itself is enabled, so its
            # induction period can be pre-existing/ready the moment it is turned on (same
            # convention as cum_no2_rate below).
            cum_o2_rate = C_raw[4]
            if self.wall_so2_k_intrinsic > 0.0:
                # SO2 + 0.5 O2 + H2O -> H2SO4 (surface-catalysed path, see _wall_so2_rate)
                r_wall_so2 = self._wall_so2_rate(C_SO2_wall, h2o_ppm_here, C_cum_o2_exposure)
                R_SO2 -= r_wall_so2
                R_H2SO4 += r_wall_so2
            r_wall_o2 = self._wall_o2_rate(C_O2, h2o_ppm_here, C_NO2_wall, C_H2SO4_wall, C_HNO3_wall,
                                           C_O2_feed=(C_lagged_o2_feed if self.o2_feed_lag_tau_hours > 0.0
                                                      else C_O2_feed))
            R_O2 -= r_wall_o2
            if self.wall_consume_h2o:
                R_H2O -= r_wall_o2
            if self.wall_hno3_corrosion_k_intrinsic > 0.0:
                # 8 HNO3 + 3 Fe -> 3 Fe(NO3)2 + 2 NO + 4 H2O. r_hno3corr is defined as the rate
                # of Fe(NO3)2 formation (1:1 with Fe consumed); the other species' ratios are
                # relative to that (8/3 HNO3, 2/3 NO, 4/3 H2O per unit Fe(NO3)2 formed).
                r_hno3corr = self._wall_hno3_corrosion_rate(h2o_ppm_here, C_HNO3_wall, C_H2SO4_wall)
                R_HNO3 -= (8.0 / 3.0) * r_hno3corr
                R_NO += (2.0 / 3.0) * r_hno3corr
                R_H2O += (4.0 / 3.0) * r_hno3corr
            if self.wall_h2so4_k_intrinsic > 0.0:
                # Fe + H2SO4 -> FeSO4 + H2 (conservative 1:1)
                r_h2so4 = self._wall_h2so4_rate(h2o_ppm_here, C_H2SO4_wall, C_HNO3_wall)
                R_H2SO4 -= r_h2so4
                R_H2 += r_h2so4
            if self.wall_no2_k_intrinsic > 0.0:
                # 2 Fe + 3 NO2 -> Fe2O3 + 3 NO (dry gas-solid path, see _wall_no2_rate)
                r_wall_no2 = self._wall_no2_rate(C_NO2_wall, h2o_ppm_here, C_cum_no2_exposure,
                                                 C_O2=C_O2, C_H2S_raw=C_raw[0],
                                                 C_H2SO4_raw=C_raw[6], C_O2_feed=C_O2_feed,
                                                 C_O2_lagged=C_lagged_o2, C_NO=C_raw[3])
                R_NO2 -= r_wall_no2
                R_NO += r_wall_no2
            # NO2 exposure accumulates regardless of whether wall_no2 itself is enabled, so the
            # induction period can be pre-existing/ready the moment it is turned on.
            cum_no2_rate = C_NO2_wall
            R_H2 += self._wall_feco3_rate(h2o_ppm_here, C_H2SO4_wall, C_HNO3_wall)
            # 8 H2S + 4 O2 -> S8 + 8 H2O, carbon-steel-catalysed (Claus-type surface reaction;
            # requires the steel wall as catalyst, not a homogeneous gas-phase pathway).
            r_wall_s8 = self._wall_s8_rate(C_H2S, C_O2)
            R_H2S -= r_wall_s8
            R_O2 -= 0.5 * r_wall_s8
            R_H2O += r_wall_s8
            R_S8 += 0.125 * r_wall_s8


        R_vector = np.array([
            R_H2S, R_SO2, R_NO2, R_NO, R_O2, R_H2O,
            R_H2SO4, R_HNO3, R_S8, R_NH3, R_N2O, R_H2
        ])

        if C_in is not None and space_time_sec is not None and space_time_sec > 0.0:
            if inflow_only:
                # Vessel pressurization/fill stage: feed gas (already at its dosed impurity
                # composition) enters a vessel with no outflow yet (back-pressure regulation
                # only starts once the vessel reaches its target inventory) -- unlike the
                # steady CSTR term below, there is no "-C/space_time_sec" loss term.
                dC_dt_species = C_in[:n_species] / space_time_sec + R_vector
            else:
                dC_dt_species = (C_in[:n_species] - C[:n_species]) / space_time_sec + R_vector
        else:
            dC_dt_species = R_vector

        if len(C) > n_species:
            # Solid corrosion product accumulates on the coupon, not in the flowing gas: pure
            # accumulation, no inflow/outflow term regardless of the branch above.
            dC_dt_wall_solid = np.array([r_h2so4, r_hno3corr, cum_h2so4_rate, cum_hno3_rate,
                                         cum_no2_rate, lagged_o2_rate, cum_o2_rate,
                                         lagged_o2_feed_rate, cum_nh3_rate])
            return np.concatenate([dC_dt_species, dC_dt_wall_solid])
        return dC_dt_species

    def simulate(self, initial_ppm, duration_sec=100000.0, num_points=100, feed_ppm=None, space_time_sec=None,
                 inflow_only=False, initial_wall_solid=None):
        if solve_ivp is None:
            raise ImportError(
                "SciPy is required to run the kinetics integration. Install it with "
                "`python -m pip install scipy`."
            ) from SCIPY_IMPORT_ERROR

        t_span = (0.0, duration_sec)
        t_eval = np.linspace(0.0, duration_sec, num_points)

        n_species = len(self.SPECIES)
        n_extra = len(self.EXTRA_STATE_KEYS)
        C0 = np.zeros(n_species + n_extra)   # + accumulated solid/cumulative states, see EXTRA_STATE_KEYS
        for idx, spec in enumerate(self.SPECIES):
            if spec in initial_ppm:
                C0[idx] = (initial_ppm[spec] * 1.0e-6) * self.molar_density
        if initial_wall_solid is not None:
            for offset, key in enumerate(self.EXTRA_STATE_KEYS):
                C0[n_species + offset] = max(0.0, initial_wall_solid.get(key, 0.0))

        C_in = None
        if feed_ppm is not None:
            C_in = np.zeros(n_species + n_extra)   # trailing entries unused, see rhs()
            for idx, spec in enumerate(self.SPECIES):
                if spec in feed_ppm:
                    C_in[idx] = (feed_ppm[spec] * 1.0e-6) * self.molar_density

        moisture_basis = feed_ppm if feed_ppm is not None else initial_ppm
        moisture_ppm = moisture_basis.get('H2O', self.water_ppm)
        rates_dict = self._calculate_pure_physical_rate_constants(moisture_ppm)

        sol = solve_ivp(
            fun=lambda t, y: self.rhs(t, y, rates_dict, C_in=C_in, space_time_sec=space_time_sec,
                                       inflow_only=inflow_only),
            t_span=t_span,
            y0=C0,
            t_eval=t_eval,
            method='Radau',
            rtol=1e-6,
            atol=1e-12
        )

        if not sol.success:
            raise RuntimeError(f"Kinetics integration failed: {sol.message}")

        ppm_results = {}
        for idx, spec in enumerate(self.SPECIES):
            raw_ppm = (sol.y[idx, :] / self.molar_density) * 1.0e6
            ppm_results[spec] = np.maximum(0.0, raw_ppm)

        wall_solid = {
            key: np.maximum(0.0, sol.y[n_species + offset, :])
            for offset, key in enumerate(self.EXTRA_STATE_KEYS)
        }

        return {
            'time_seconds': sol.t,
            'time_hours': sol.t / 3600.0,
            'ppm': ppm_results,
            'wall_solid': wall_solid,
            'molar_density': self.molar_density,
            'phase': self.phase,
            'phi': self.phi_dict,
            'rates': rates_dict
        }


# ==================================================================================================
# HIGH-LEVEL MULTI-PHASE CSTR EXPERIMENT MANAGER CLASS
# ==================================================================================================
class CO2ImpurityReactorExperiment:
    """
    High-level Manager for Setting Up, Configuring, and Executing Multi-Phase CSTR Experiments.
    Uses NeqSim Java SRK EOS when available, with an explicit screening fallback.
    """

    def __init__(self, target_pressure_bar=25.0, target_temp_C=-25.0, diameter_cm=6.5, volume_ml=300.0, mass_flow_g_h=50.0, material='carbon_steel',
                 condensation_exponent=0.0, rho_m_reference=24.0,
                 wall_area_m2=0.0, wall_k_intrinsic=1.0e-4,
                 wall_rho_pass=5.0, wall_hill_n=3.0, wall_k_h2o_ppm=3.0,
                 wall_acid_exponent=1.5, wall_acid_background=0.02, wall_acid_gain=1.0,
                 wall_consume_h2o=False,
                 calibrated_profile=None):
        self.target_P = float(target_pressure_bar)
        self.target_T_C = float(target_temp_C)
        self.target_T_K = self.target_T_C + 273.15
        self.diameter_cm = float(diameter_cm)
        self.volume_ml = float(volume_ml)
        self.mass_flow_g_h = float(mass_flow_g_h)
        self.material = material

        self.initial_gas = 'N2'
        self.initial_P_bar = 1.0
        self.initial_T_C = 25.0

        self.model = CO2ImpurityKineticsModel(
            T_kelvin=self.target_T_K,
            P_bar=self.target_P,
            material=self.material,
            condensation_exponent=condensation_exponent,
            rho_m_reference=rho_m_reference,
            wall_area_m2=wall_area_m2,
            wall_k_intrinsic=wall_k_intrinsic,
            wall_rho_pass=wall_rho_pass,
            wall_hill_n=wall_hill_n,
            wall_k_h2o_ppm=wall_k_h2o_ppm,
            wall_acid_exponent=wall_acid_exponent,
            wall_acid_background=wall_acid_background,
            wall_acid_gain=wall_acid_gain,
            wall_consume_h2o=wall_consume_h2o,
        )
        self.model.set_reactor_geometry(
            diameter_cm=self.diameter_cm,
            volume_ml=self.volume_ml,
            mass_flow_g_h=self.mass_flow_g_h
        )
        if calibrated_profile is not None:
            self.model.apply_calibrated_profile(profile=calibrated_profile)

        self.phases = []
        self.simulation_results = None

    def configure_wall_corrosion(self, **kwargs):
        """Forward wall-corrosion configuration to the underlying model."""
        self.model.configure_wall_corrosion(**kwargs)

    def set_phase_condensation(self, exponent, rho_m_reference=None):
        """Forward phase-condensation configuration to the underlying model."""
        self.model.set_phase_condensation(exponent, rho_m_reference=rho_m_reference)

    def override_f_phase(self, value):
        """Forward a direct f_phase override to the underlying model."""
        self.model.override_f_phase(value)

    def set_srk_kij(self, species_name, kij):
        """Forward a CO2-species SRK binary interaction parameter override to the model."""
        self.model.set_srk_kij(species_name, kij)

    def set_ideal_fugacity(self, species_name):
        """Forward an ideal-fugacity (phi=1) override to the underlying model."""
        self.model.set_ideal_fugacity(species_name)

    def set_phi_override(self, species_name, value):
        """Forward an arbitrary fugacity-coefficient override to the underlying model."""
        self.model.set_phi_override(species_name, value)

    def set_r5_no_activity(self, value):
        """Forward R5's decoupled reverse-term NO activity to the underlying model."""
        self.model.set_r5_no_activity(value)

    def apply_calibrated_profile(self, profile='carbon_steel_wet_co2'):
        """Apply a named calibrated parameter profile to the underlying model."""
        self.model.apply_calibrated_profile(profile=profile)

    def set_initial_vessel_charge(self, gas_name='N2', pressure_bar=1.0, temp_C=25.0):
        """Record initial-charge metadata.

        The tutorial kinetics state tracks only the impurity species in ``SPECIES``. The initial
        inert-gas inventory is therefore reported as metadata and is not included in the species
        ODE balance.
        """
        self.initial_gas = str(gas_name).upper()
        self.initial_P_bar = float(pressure_bar)
        self.initial_T_C = float(temp_C)

    def set_reactor_geometry(self, diameter_cm=None, length_cm=None, volume_ml=None, mass_flow_g_h=None):
        self.model.set_reactor_geometry(
            diameter_cm=diameter_cm,
            length_cm=length_cm,
            volume_ml=volume_ml,
            mass_flow_g_h=mass_flow_g_h
        )
        geom = self.model.get_reactor_geometry()
        self.diameter_cm = geom['diameter_cm']
        self.volume_ml = geom['volume_ml']
        self.mass_flow_g_h = geom['mass_flow_g_h']

    def set_reaction_constants(self, reaction_identifier, A_forward=None, Ea_forward_kJ_mol=None):
        self.model.set_reaction_constants(reaction_identifier, A_forward, Ea_forward_kJ_mol)

    def add_phase(self, duration_hours, feed_ppm, phase_name=None, mass_flow_g_h=None,
                  temp_C=None, pressure_bar=None):
        p_idx = len(self.phases)
        name = phase_name if phase_name else f"Phase {p_idx}"
        phase_mass_flow_g_h = self.mass_flow_g_h if mass_flow_g_h is None else float(mass_flow_g_h)
        if phase_mass_flow_g_h < 0.0:
            raise ValueError(f'Phase mass flow must be non-negative, got {phase_mass_flow_g_h} g/h')

        feed = {s: 0.0 for s in self.model.SPECIES}
        if isinstance(feed_ppm, dict):
            for k, v in feed_ppm.items():
                if k in feed:
                    feed[k] = float(v)

        self.phases.append({
            'name': name,
            'duration_hours': float(duration_hours),
            'feed_ppm': feed,
            'mass_flow_g_h': phase_mass_flow_g_h,
            'temp_C': None if temp_C is None else float(temp_C),
            'pressure_bar': None if pressure_bar is None else float(pressure_bar),
        })

    def clear_phases(self):
        self.phases = []
        self.simulation_results = None

    def generate_reactor_report(self):
        report = self.model.generate_reactor_report()
        charge = (
            f"\nInitial vessel charge metadata: {self.initial_gas}, "
            f"{self.initial_P_bar:.3f} bar, {self.initial_T_C:.3f} °C "
            "(not included in the impurity-species ODE balance)"
        )
        return report + charge

    def run_experiment(self):
        if not self.phases:
            self.add_phase(10.0, {s: 0.0 for s in self.model.SPECIES}, "Phase 0: Pressurization & Pure CO2 Flow")
            self.add_phase(20.0, {'SO2': 10.0, 'NO2': 10.0, 'O2': 10.0, 'H2O': 10.0}, "Phase 1: 10 ppm Without H2S")
            self.add_phase(20.0, {'H2S': 10.0, 'SO2': 10.0, 'NO2': 10.0, 'O2': 10.0, 'H2O': 10.0}, "Phase 2: 10 ppm All Impurities")

        all_t_h = []
        all_ppm = {s: [] for s in self.model.SPECIES}
        extra_keys = self.model.EXTRA_STATE_KEYS
        all_wall_solid = {k: [] for k in extra_keys}
        current_cumulative_t = 0.0

        current_state_ppm = {s: 0.0 for s in self.model.SPECIES}
        current_wall_solid = {k: 0.0 for k in extra_keys}

        for idx, phase in enumerate(self.phases):
            dur_h = phase['duration_hours']
            feed = phase['feed_ppm']
            phase_mass_flow_g_h = phase['mass_flow_g_h']
            # Re-flash first: molar_density feeds the residence time and the fill-mass target.
            self.model.set_conditions(temp_C=phase.get('temp_C'),
                                      pressure_bar=phase.get('pressure_bar'))
            rho_g_ml = (self.model.molar_density * MW_CO2) * 1e-3
            m_target_g = self.volume_ml * rho_g_ml
            self.set_reactor_geometry(mass_flow_g_h=phase_mass_flow_g_h)
            tau_sec = self.model.get_reactor_geometry()['residence_time_seconds']
            t_fill_hours = m_target_g / phase_mass_flow_g_h if phase_mass_flow_g_h > 0 else 0.0

            if idx == 0 and dur_h >= t_fill_hours:
                res_fill = self.model.simulate(
                    initial_ppm=current_state_ppm,
                    duration_sec=t_fill_hours * 3600.0,
                    num_points=max(int(t_fill_hours * 10), 50),
                    feed_ppm=feed,
                    space_time_sec=t_fill_hours * 3600.0,
                    inflow_only=True,
                    initial_wall_solid=current_wall_solid,
                )

                fill_state = {s: res_fill['ppm'][s][-1] for s in self.model.SPECIES}
                fill_wall_solid = {k: v[-1] for k, v in res_fill['wall_solid'].items()}
                rem_dur_h = dur_h - t_fill_hours

                if rem_dur_h > 0.001:
                    res_flow = self.model.simulate(
                        initial_ppm=fill_state,
                        duration_sec=rem_dur_h * 3600.0,
                        num_points=max(int(rem_dur_h * 10), 30),
                        feed_ppm=feed,
                        space_time_sec=tau_sec,
                        initial_wall_solid=fill_wall_solid,
                    )
                    t_res = np.concatenate([res_fill['time_hours'], t_fill_hours + res_flow['time_hours']])
                    ppm_res = {s: np.concatenate([res_fill['ppm'][s], res_flow['ppm'][s]]) for s in self.model.SPECIES}
                    wall_solid_res = {k: np.concatenate([res_fill['wall_solid'][k], res_flow['wall_solid'][k]])
                                      for k in extra_keys}
                else:
                    t_res = res_fill['time_hours']
                    ppm_res = res_fill['ppm']
                    wall_solid_res = res_fill['wall_solid']
            else:
                res_flow = self.model.simulate(
                    initial_ppm=current_state_ppm,
                    duration_sec=dur_h * 3600.0,
                    num_points=max(int(dur_h * 10), 100),
                    feed_ppm=feed,
                    space_time_sec=tau_sec,
                    initial_wall_solid=current_wall_solid,
                )
                t_res = res_flow['time_hours']
                ppm_res = res_flow['ppm']
                wall_solid_res = res_flow['wall_solid']

            all_t_h.append(current_cumulative_t + t_res)
            for s in self.model.SPECIES:
                all_ppm[s].append(ppm_res[s])
            for k in extra_keys:
                all_wall_solid[k].append(wall_solid_res[k])

            current_cumulative_t += dur_h
            current_state_ppm = {s: ppm_res[s][-1] for s in self.model.SPECIES}
            current_wall_solid = {k: wall_solid_res[k][-1] for k in extra_keys}

        master_t = np.concatenate(all_t_h)
        master_ppm = {s: np.concatenate(all_ppm[s]) for s in self.model.SPECIES}
        master_wall_solid = {k: np.concatenate(v) for k, v in all_wall_solid.items()}

        self.simulation_results = {
            'time_hours': master_t,
            'ppm': master_ppm,
            'wall_solid': master_wall_solid,
            'phases': self.phases
        }

        return self.simulation_results

    def get_table_results(self, resolution_hours=1.0):
        if self.simulation_results is None:
            self.run_experiment()

        return self.model.get_table_results(self.simulation_results, resolution_hours=resolution_hours)

    def plot_results(self, save_path=None, title="Multi-Phase CSTR CO2 Impurity Kinetics"):
        if self.simulation_results is None:
            self.run_experiment()

        t_h = self.simulation_results['time_hours']
        ppm = self.simulation_results['ppm']

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

        ax1.plot(t_h, ppm['H2S'], label='H2S', linewidth=2.0, color='#e74c3c')
        ax1.plot(t_h, ppm['SO2'], label='SO2', linewidth=2.0, color='#f39c12')
        ax1.plot(t_h, ppm['NO2'], label='NO2', linewidth=2.0, color='#9b59b6')
        ax1.plot(t_h, ppm['O2'],  label='O2',  linewidth=2.0, color='#2ecc71')
        ax1.plot(t_h, ppm['H2O'], label='H2O', linewidth=2.0, color='#3498db')
        ax1.set_ylabel('Reactants (ppm)', fontsize=11, fontweight='bold')
        ax1.set_title(title, fontsize=13, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(loc='upper right', frameon=True)

        ax2.plot(t_h, ppm['H2SO4'], label='H2SO4 Sulfuric Acid (Formed)', linewidth=4.0, color='#FF0033', marker='o', markevery=40)
        ax2.fill_between(t_h, ppm['H2SO4'], color='#FF0033', alpha=0.3, label='H2SO4 Shaded Acid Accumulation')
        ax2.set_ylabel('H2SO4 Acid (ppm)', fontsize=11, fontweight='bold')
        max_h2so4 = np.max(ppm['H2SO4'])
        ax2.set_ylim(0.0, max(max_h2so4 * 1.35, 2.0))
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(loc='upper left', frameon=True)

        if max_h2so4 > 0.05:
            max_idx = np.argmax(ppm['H2SO4'])
            max_t = t_h[max_idx]
            time_span = max(float(t_h[-1] - t_h[0]), 1.0)
            annotation_t = max(float(t_h[0]), max_t - 0.2 * time_span)
            ax2.annotate(
                f'H2SO4 Acid Peak: {max_h2so4:.2f} ppm',
                xy=(max_t, max_h2so4),
                xytext=(annotation_t, max_h2so4 + 0.8),
                arrowprops={'facecolor': '#FF0033', 'shrink': 0.08, 'width': 3.0, 'headwidth': 10.0},
                fontsize=12,
                fontweight='bold',
                color='#B20000',
                bbox={'boxstyle': 'round,pad=0.3', 'fc': '#FFE6E6', 'ec': '#FF0033', 'lw': 1.5}
            )

        ax3.plot(t_h, ppm['NO'],    label='NO Gas',      linewidth=2.5, color='#8e44ad')
        ax3.plot(t_h, ppm['NH3'],   label='NH3 Ammonia', linewidth=2.5, color='#16a085')
        ax3.plot(t_h, ppm['S8'],    label='S8 Elemental Sulfur', linewidth=2.0, color='#f1c40f')
        ax3.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Gaseous Products (ppm)', fontsize=11, fontweight='bold')
        ax3.grid(True, linestyle='--', alpha=0.6)
        ax3.legend(loc='upper right', frameon=True)

        cum_t = 0.0
        for phase in self.phases[:-1]:
            cum_t += phase['duration_hours']
            ax1.axvline(cum_t, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
            ax2.axvline(cum_t, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
            ax3.axvline(cum_t, color='black', linestyle=':', linewidth=1.5, alpha=0.7)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved successfully to: {save_path}")

        return fig, (ax1, ax2, ax3)


# ==================================================================================================
# COMPACT NOTEBOOK FACADE FOR SEQUENTIAL-PHASE AUTOCLAVE CORROSION SIMULATIONS
# ==================================================================================================
# Atom counts per gas-phase species, used only by AutoclaveExperiment.get_mass_balance_table() to
# verify N/S/H/O are conserved (feed in = outflow + gas-phase accumulation + wall-solid deposit).
# ==================================================================================================
SPECIES_ATOM_COUNTS = {
    'H2S':   {'H': 2, 'S': 1},
    'SO2':   {'S': 1, 'O': 2},
    'NO2':   {'N': 1, 'O': 2},
    'NO':    {'N': 1, 'O': 1},
    'O2':    {'O': 2},
    'H2O':   {'H': 2, 'O': 1},
    'H2SO4': {'H': 2, 'S': 1, 'O': 4},
    'HNO3':  {'H': 1, 'N': 1, 'O': 3},
    'S8':    {'S': 8},
    'NH3':   {'N': 1, 'H': 3},
    'N2O':   {'N': 2, 'O': 1},
    'H2':    {'H': 2},
}


# ==================================================================================================
# WASH-WATER pH MODEL (illustrative, screening-level -- see AutoclaveExperiment.
# get_wash_water_pH_table() docstring for the full physical model description)
# ==================================================================================================
# Standard literature aqueous-equilibrium constants at 298.15 K, each with a van't Hoff
# temperature-correction enthalpy (J/mol). Not independently regressed for this specific
# mixture -- reasonable, widely tabulated textbook values used for a screening-level estimate.
KW_298 = 1.0e-14                   # H2O <-> H+ + OH-
DH_KW = 55800.0

KH_CO2_298_MOL_L_ATM = 3.3e-2      # CO2(g) <-> CO2(aq), mol/(L*atm)
DH_KH_CO2 = -19950.0               # exothermic dissolution (d ln(KH)/d(1/T) ~= +2400 K)

KA1_CO2_298 = 4.45e-7              # CO2(aq) + H2O <-> H+ + HCO3-   (pKa1 = 6.35)
DH_KA1_CO2 = 7700.0

KA2_CO2_298 = 4.69e-11             # HCO3- <-> H+ + CO3^2-          (pKa2 = 10.33)
DH_KA2_CO2 = 14900.0

KA_NH4_298 = 5.6e-10               # NH4+ <-> NH3(aq) + H+          (pKa = 9.25)
DH_KA_NH4 = 52200.0

KA2_H2SO4_298 = 1.2e-2             # HSO4- <-> H+ + SO4^2- (pKa2 = 1.92; T-independence assumed)

# Ion <-> neutral-species molar masses for ion-chromatography-style reporting (see
# AutoclaveExperiment.get_autoclave_wash_table()) -- IC measures the dissociated ionic species,
# not the neutral parent acid/base tracked by the kinetics model.
PROTON_MASS_G_MOL = 1.008
M_SO4_2MINUS = MW_H2SO4 - 2.0 * PROTON_MASS_G_MOL   # H2SO4 -> SO4^2-
M_NO3_MINUS = MW_HNO3 - PROTON_MASS_G_MOL           # HNO3  -> NO3-
M_NH4_PLUS = MW_NH3 + PROTON_MASS_G_MOL             # NH3   -> NH4+ (protonated in acidic water)


def _van_t_hoff(k_298, dh_j_per_mol, temp_kelvin):
    """van't Hoff correction: ln(K/K298) = -dH/R * (1/T - 1/298.15)."""
    return k_298 * np.exp(-dh_j_per_mol / R_GAS * (1.0 / temp_kelvin - 1.0 / 298.15))


def _wash_water_charge_balance(pH, co2_aq, nh3_t, h2so4_t, hno3_t, kw, ka1, ka2, ka_nh4, ka2_so4):
    """Net cation - anion charge (mol/L) for the open CO2 / NH3 / H2SO4 / HNO3 system; the root
    of this (strictly monotonically decreasing in pH) is the solution pH."""
    h = 10.0 ** (-pH)
    oh = kw / h
    hco3 = ka1 * co2_aq / h
    co3 = ka1 * ka2 * co2_aq / h ** 2
    nh4 = nh3_t * h / (h + ka_nh4)
    so4 = h2so4_t * ka2_so4 / (h + ka2_so4)     # H2SO4's 1st proton is taken as fully dissociated
    hso4 = h2so4_t - so4
    no3 = hno3_t                                 # strong acid, fully dissociated
    cations = h + nh4
    anions = oh + hco3 + 2.0 * co3 + hso4 + 2.0 * so4 + no3
    return cations - anions


def _solve_wash_water_pH(co2_aq, nh3_t, h2so4_t, hno3_t, temp_kelvin=298.15):
    """Bisection solve for wash-water pH (scipy-free, dependency-light, robust to the
    guaranteed-monotonic charge-balance residual)."""
    kw = _van_t_hoff(KW_298, DH_KW, temp_kelvin)
    ka1 = _van_t_hoff(KA1_CO2_298, DH_KA1_CO2, temp_kelvin)
    ka2 = _van_t_hoff(KA2_CO2_298, DH_KA2_CO2, temp_kelvin)
    ka_nh4 = _van_t_hoff(KA_NH4_298, DH_KA_NH4, temp_kelvin)

    args = (co2_aq, nh3_t, h2so4_t, hno3_t, kw, ka1, ka2, ka_nh4, KA2_H2SO4_298)
    lo, hi = 0.0, 14.0
    f_lo = _wash_water_charge_balance(lo, *args)
    f_hi = _wash_water_charge_balance(hi, *args)
    if f_lo * f_hi > 0.0:
        return 7.0  # degenerate/edge case (e.g. all concentrations ~0) -- report neutral
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f_mid = _wash_water_charge_balance(mid, *args)
        if f_lo * f_mid <= 0.0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


class AutoclaveExperiment:
    """Notebook-friendly facade over :class:`CO2ImpurityReactorExperiment`.

    Bundles the sequential A-I feed-schedule bookkeeping, the general calibrated
    carbon-steel / wet-CO2 kinetics + wall-corrosion parameter set (same constants for every
    experiment; see ``CARBON_STEEL_WET_CO2_KINETICS``), reaction-activity ranking and the
    standard plot set behind a handful of methods, so a notebook only needs to supply reactor
    geometry and a feed schedule:

        autoclave = AutoclaveExperiment(volume_ml=530, mass_flow_g_h=150, diameter_cm=2,
                                         temp_C=25, pressure_bar=20, material='carbon_steel',
                                         coupon_diameter_cm=4.0, coupon_thickness_mm=1)
        autoclave.set_phases(PHASES_FEED, termination_hour=305.0).run()
        autoclave.get_values()                              # ppm vs time, all species
        autoclave.build_plot('reactant species')
        autoclave.build_plot('reaction products')
        autoclave.get_reaction_table()                       # overall pathway ranking
        autoclave.get_step_reaction_table()                  # ranking per A-I step
        autoclave.build_plot('surface reaction products')
        autoclave.build_plot('corrosion rate')
        autoclave.get_surface_data()                         # corrosion numbers vs time
        autoclave.get_mass_balance_table()                   # N/S/H/O closure check per phase
        autoclave.get_wash_water_pH_table(water_mass_g=30.0)  # wash-water pH vs time
        autoclave.build_plot('wash water pH', water_mass_g=30.0)
        autoclave.get_autoclave_wash_table(wash_mass_g=30.0)  # IC-style SO4/NO3/NH4 vs time
        autoclave.get_autoclave_wash_summary(wash_mass_g=30.0)  # end-of-run IC report layout
    """


    REACTION_NAMES = {
        'R1': 'SO2 + 0.5 O2 + H2O -> H2SO4',
        'R2': 'H2S + 3 NO2 -> SO2 + H2O + 3 NO',
        'R3A': 'SO2 + NO2 + H2O -> NO + H2SO4',
        'R4': '2 NO + O2 -> 2 NO2',
        'R5': '3 NO2 + H2O -> 2 HNO3 + NO',
        'R7': '5 H2S + 6 NO + 4 H2O -> 6 NH3 + 5 SO2',
        'R10': '4 NH3 + 4 NO + 3 O2 -> 4 N2O + 6 H2O',
        'R11': 'H2S + 2 NO -> N2O + 1/8 S8 + H2O',
        'R12': 'H2S + 2 O2 -> H2SO4 (NO2-catalysed)',
        'R13': '4 NO2 + H2S -> H2SO4 + 4 NO',
        'R15': '4 NO2 -> 2 N2O + 3 O2',
        'Wall O2 / Fe2O3': '4 Fe + 3 O2 -> 2 Fe2O3 (surface path)',
        'Wall NO2 / Fe2O3': '2 Fe + 3 NO2 -> Fe2O3 + 3 NO (dry surface path)',
        'FeCO3 deposit': 'Fe + CO2(aq) + H2O -> FeCO3 + H2 (surface path, disabled by default)',
        'Wall HNO3 / Fe(NO3)2': '8 HNO3 + 3 Fe -> 3 Fe(NO3)2 + 2 NO + 4 H2O (surface path)',
        'Wall H2SO4 / FeSO4': 'Fe + H2SO4 -> FeSO4 + H2 (surface path)',
        'Wall S8 / Claus': '8 H2S + 4 O2 -> S8 + 8 H2O (carbon-steel-catalysed, surface path)',
        'Wall SO2 / H2SO4': 'SO2 + 0.5 O2 + H2O -> H2SO4 (surface-catalysed, disabled by default)',
    }
    REACTANT_STYLE = {'H2S': '#e74c3c', 'SO2': '#f39c12', 'NO2': '#9b59b6', 'O2': '#2ecc71', 'H2O': '#3498db'}
    PRODUCT_STYLE = {
        'H2SO4': ('#c0392b', '-'), 'HNO3': ('#8e44ad', '-'), 'S8': ('#f1c40f', '-'),
        'NH3': ('#16a085', '-'), 'NO': ('#7f8c8d', '--'), 'N2O': ('#d35400', '--'),
        'H2': ('#2980b9', '-'),
    }
    PHASE_COLORS = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c']

    M_FE, M_FE2O3 = 55.845, 159.688
    M_FECO3 = 115.856
    M_FE_NO3_2 = 179.86
    M_FE_SO4 = 151.91
    RHO_FE_KG_M3 = 7850.0

    def __init__(self, volume_ml, mass_flow_g_h, diameter_cm, temp_C, pressure_bar,
                 material='carbon_steel', coupon_diameter_cm=3.0, coupon_thickness_mm=5.0,
                 calibrated_profile='carbon_steel_wet_co2'):
        self.volume_ml = float(volume_ml)
        self.mass_flow_g_h = float(mass_flow_g_h)
        self.diameter_cm = float(diameter_cm)
        self.temp_C = float(temp_C)
        self.pressure_bar = float(pressure_bar)
        self.material = material

        self.exp = CO2ImpurityReactorExperiment(
            target_pressure_bar=self.pressure_bar,
            target_temp_C=self.temp_C,
            diameter_cm=self.diameter_cm,
            volume_ml=self.volume_ml,
            mass_flow_g_h=self.mass_flow_g_h,
            material=self.material,
            calibrated_profile=calibrated_profile,
        )
        if coupon_diameter_cm is not None and coupon_thickness_mm is not None:
            self.exp.configure_wall_corrosion(
                coupon_diameter_cm=coupon_diameter_cm, coupon_thickness_mm=coupon_thickness_mm)
        self.exp.set_initial_vessel_charge(
            gas_name='CO2', pressure_bar=self.pressure_bar, temp_C=self.temp_C)

        self.phases = []
        self.phase_durations = []
        self.phase_bounds = []
        self.results = None
        self._reaction_rate_series = None
        self._reactant_plot_limits = (None, None)

    # ---------------------------------------------------------------------- setup
    @staticmethod
    def _phase_label(feed, prev, index):
        if not feed:
            return 'Pure CO2'
        changes = []
        for sp in sorted(set(feed) | set(prev)):
            curr, old = feed.get(sp, 0.0), prev.get(sp, 0.0)
            if curr == old:
                continue
            if old == 0.0:
                changes.append(f'+{sp} {curr:g}')
            elif curr == 0.0:
                changes.append(f'-{sp}')
            else:
                changes.append(f'{sp} {old:g}->{curr:g}')
        return ', '.join(changes) if changes else f'Phase {index}'

    def set_phases(self, phases_feed, termination_hour):
        """Build the phase schedule from ``(start_hour, mass_flow_g_h, feed_ppm[, label])`` entries.

        Each phase's duration is derived automatically from consecutive start times; the
        final phase ends at ``termination_hour``. When supplied, ``mass_flow_g_h`` sets that
        phase's CSTR residence time and feed/outflow throughput; otherwise the constructor's
        ``mass_flow_g_h`` is used. Legacy ``(start_hour, feed_ppm[, label[, mass_flow_g_h]])``
        entries remain supported.

        A phase may also be given as ``(start_hour, dict)`` where the dict carries any of
        ``pressure_bar`` / ``temp_C`` / ``mass_flow_g_h`` / ``feed`` / ``label`` -- the form to
        use when a run changes conditions mid-experiment (e.g. stepping the temperature down
        partway through a run). Whenever ``temp_C``/``pressure_bar`` change, the SRK flash is re-evaluated
        at the start of that phase, so molar density, fugacity coefficients, the water dew
        point and the residence time are all recomputed rather than carried over. Omitted keys
        inherit the constructor's values.
        """
        def _unpack(entry):
            """-> (start_h, feed, label, mass_flow_g_h, temp_C, pressure_bar)."""
            if len(entry) == 2 and isinstance(entry[1], dict) and (
                    set(entry[1]) - set(self.exp.model.SPECIES)):
                spec = entry[1]
                return (
                    float(entry[0]),
                    spec.get('feed', spec.get('feed_ppm', {})),
                    spec.get('label', spec.get('name')),
                    float(spec.get('mass_flow_g_h', self.mass_flow_g_h)),
                    spec.get('temp_C'),
                    spec.get('pressure_bar'),
                )
            if len(entry) >= 3 and isinstance(entry[1], (int, float)):
                return (
                    float(entry[0]),
                    entry[2],
                    (entry[3] if len(entry) >= 4 else None),
                    float(entry[1]),
                    None,
                    None,
                )
            return (
                float(entry[0]),
                entry[1],
                (entry[2] if len(entry) >= 3 else None),
                (float(entry[3]) if len(entry) >= 4 else self.mass_flow_g_h),
                None,
                None,
            )

        starts = [_unpack(e)[0] for e in phases_feed]
        if starts != sorted(starts):
            raise ValueError(f'phases_feed start times must be non-decreasing: {starts}')
        ends = starts[1:] + [float(termination_hour)]

        self.phases, self.phase_durations = [], []
        prev = {}
        for start, end, entry in zip(starts, ends, phases_feed):
            _, feed, override, phase_mass_flow_g_h, _, _ = _unpack(entry)
            dur = end - start
            if dur <= 0:
                raise ValueError(f'Phase at {start} h has non-positive duration {dur} h '
                                  f'(next start / termination = {end} h)')
            if phase_mass_flow_g_h < 0.0:
                raise ValueError(
                    f'Phase at {start} h has negative mass flow {phase_mass_flow_g_h} g/h')
            label = override or self._phase_label(feed, prev, len(self.phases))
            self.phases.append((label, feed))
            self.phase_durations.append(dur)
            prev = feed

        self.phase_bounds, cum = [], 0.0
        for dur in self.phase_durations:
            self.phase_bounds.append((cum, cum + dur))
            cum += dur

        self.exp.clear_phases()
        for (name, feed), dur, entry in zip(self.phases, self.phase_durations, phases_feed):
            _, _, _, phase_mass_flow_g_h, phase_temp_C, phase_pressure_bar = _unpack(entry)
            self.exp.add_phase(
                duration_hours=dur,
                feed_ppm=feed,
                phase_name=name,
                mass_flow_g_h=phase_mass_flow_g_h,
                temp_C=phase_temp_C,
                pressure_bar=phase_pressure_bar,
            )
        return self

    def get_reactor_report(self):
        return self.exp.generate_reactor_report()

    # ------------------------------------------------------------------------ run
    def run(self):
        if not self.phases:
            raise RuntimeError('Call set_phases(...) before run().')
        self.results = self.exp.run_experiment()
        self._reaction_rate_series = None
        return self

    @property
    def t_h(self):
        return self.results['time_hours']

    @property
    def ppm(self):
        return self.results['ppm']

    @property
    def wall_solid(self):
        """Accumulated solid corrosion product [kmol/m^3], as actually integrated by the ODE
        solver (see ``rhs``) -- the true, self-consistent trajectory driving the autocatalytic
        O2 enhancement, not a post-hoc reconstruction."""
        return self.results['wall_solid']

    # ----------------------------------------------------------------------- data
    def get_values(self):
        """All simulated outlet concentrations (ppm) vs time, as a DataFrame."""
        if self.results is None:
            raise RuntimeError('Call run() before get_values().')
        df = pd.DataFrame({'time_hours': self.t_h})
        for species in self.exp.model.SPECIES:
            df[species] = self.ppm[species]
        return df

    def _phase_masks(self):
        return [(self.t_h >= start) & (self.t_h <= end) for start, end in self.phase_bounds]

    def _compute_reaction_rate_series(self):
        if self._reaction_rate_series is not None:
            return self._reaction_rate_series

        model = self.exp.model
        rho_m, phi = model.molar_density, model.phi_dict
        t_h, ppm = self.t_h, self.ppm
        C = {s: np.clip(ppm[s] * 1e-6 * rho_m * phi[s], 0.0, None) for s in model.SPECIES}
        no_r5 = np.clip(ppm['NO'] * 1e-6 * rho_m * model.r5_no_activity, 0.0, None)
        no_r4 = np.clip(ppm['NO'] * 1e-6 * rho_m * model.r4_no_activity, 0.0, None)

        o2_feed_ppm_series = np.zeros_like(t_h)
        for (_, feed), mask in zip(self.phases, self._phase_masks()):
            o2_feed_ppm_series[mask] = feed.get('O2', 0.0)
        C_O2_feed_series = o2_feed_ppm_series * 1e-6 * rho_m

        names = ('R1', 'R2', 'R3A', 'R4', 'R5', 'R7', 'R10', 'R11', 'R12', 'R13', 'R15',
                 'Wall O2 / Fe2O3', 'FeCO3 deposit', 'Wall HNO3 / Fe(NO3)2', 'Wall H2SO4 / FeSO4',
                 'Wall NO2 / Fe2O3', 'Wall S8 / Claus', 'Wall SO2 / H2SO4')
        rate = {name: np.zeros_like(t_h) for name in names}

        for i in range(len(t_h)):
            k = model.get_reaction_rates(moisture_ppm=ppm['H2O'][i])
            h2s, so2, no2 = C['H2S'][i], C['SO2'][i], C['NO2'][i]
            no, o2, h2o = C['NO'][i], C['O2'][i], C['H2O'][i]
            h2so4, hno3 = C['H2SO4'][i], C['HNO3'][i]
            nh3, s8, n2o = C['NH3'][i], C['S8'][i], C['N2O'][i]
            c_h2s_raw = ppm['H2S'][i] * 1e-6 * rho_m
            c_h2so4_raw = ppm['H2SO4'][i] * 1e-6 * rho_m
            c_o2_feed = C_O2_feed_series[i]
            c_o2_lagged = self.wall_solid['LaggedO2'][i] if 'LaggedO2' in self.wall_solid else 0.0

            rate['R1'][i] = k['k1_f'] * so2 * o2**0.5 * h2o - k['k1_r'] * h2so4
            r2_no2_boost = 1.0
            if model.r2_no2_boost_ref_ppm > 0.0:
                no2_ppm_r2 = no2 / max(rho_m, 1e-9) * 1e6
                ratio_r2 = (no2_ppm_r2 / model.r2_no2_boost_ref_ppm) ** model.r2_no2_boost_hill_n
                r2_no2_boost = 1.0 + model.r2_no2_boost_gain * ratio_r2 / (1.0 + ratio_r2)
            rate['R2'][i] = r2_no2_boost * (k['f_phase'] * k['k2_f'] * h2s * no2 - k['k2_r'] * so2 * h2o * no**3) \
                * model._o2_presence_gate(c_o2_feed, model.r2_o2_presence_ref_ppm,
                                          model.r2_o2_presence_hill_n)
            no_r3a = no * (1.0 - model.r3a_no_escape_frac * k['f_phase'])
            rate['R3A'][i] = (k['k3a_f'] * so2 * no2 * h2o - k['k3a_r'] * no_r3a * h2so4) \
                * model._feed_o2_passivation(c_o2_feed, model.r3a_feed_o2_ref_ppm,
                                             model.r3a_feed_o2_hill_n,
                                             floor=model.r3a_feed_o2_floor,
                                             cap_ppm=model.r3a_feed_o2_cap_ppm) \
                * model._o2_presence_gate(c_o2_feed, model.r3a_o2_presence_ref_ppm,
                                          model.r3a_o2_presence_hill_n)
            rate['R4'][i] = k['k4_f'] * no_r4[i]**2 * o2 - k['k4_r'] * no2**2
            rate['R5'][i] = k['k5_f'] * no2**3 * h2o - k['k5_r'] * hno3**2 * no_r5[i]
            rate['R7'][i] = k['k7_f'] * h2s * no * h2o
            rate['R10'][i] = k.get('k10_f', 0.0) * nh3 * no * o2 - k.get('k10_r', 0.0) * n2o**4 * h2o**6
            r11_o2_boost = 1.0
            if model.r11_o2_ref_ppm > 0.0:
                o2_ppm_r11 = o2 / max(rho_m, 1e-9) * 1e6
                ratio_n = (o2_ppm_r11 / model.r11_o2_ref_ppm) ** model.r11_o2_hill_n
                r11_o2_boost = 1.0 + model.r11_o2_gain * ratio_n / (1.0 + ratio_n)
            rate['R11'][i] = r11_o2_boost * k.get('k11_f', 0.0) * h2s * no - k.get('k11_r', 0.0) * n2o * s8**0.125 * h2o
            r12_scale = 1.0 if model.r12_density_independent else k['f_phase']
            no2_r12 = no2 ** model.r12_no2_order
            rate['R12'][i] = r12_scale * (k.get('k12_f', 0.0) * h2s * o2 * no2_r12
                                           - k.get('k12_r', 0.0) * h2so4 * no2_r12)
            rate['R13'][i] = k['f_phase'] * (k.get('k13_f', 0.0) * h2s * no2 ** model.r13_no2_order
                                              - k.get('k13_r', 0.0) * h2so4 * no ** model.r13_no2_order)
            r15_o2_inhib = 1.0
            if model.r15_o2_inhib_ref_ppm > 0.0:
                if model.o2_lag_tau_hours > 0.0:
                    o2_ppm_r15 = max(c_o2_lagged, 0.0) / max(rho_m, 1e-9) * 1e6
                else:
                    o2_ppm_r15 = ppm['O2'][i]
                r15_o2_inhib = 1.0 / (1.0 + (o2_ppm_r15 / model.r15_o2_inhib_ref_ppm)
                                      ** model.r15_o2_inhib_hill_n)
            r15_o2_activation = 1.0
            if model.r15_o2_activation_ref_ppm > 0.0:
                o2_ppm_act = ppm['O2'][i]
                ratio_act = (o2_ppm_act / model.r15_o2_activation_ref_ppm) ** model.r15_o2_activation_hill_n \
                    if o2_ppm_act > 0.0 else 0.0
                r15_o2_activation = ratio_act / (1.0 + ratio_act)
            no2_r15 = no2
            if model.r15_no2_cap_ppm > 0.0:
                no2_ppm_r15 = ppm['NO2'][i]
                if no2_ppm_r15 > 0.0:
                    ratio_n = (no2_ppm_r15 / model.r15_no2_cap_ppm) ** model.r15_no2_cap_hill_n
                    no2_ppm_r15_capped = model.r15_no2_cap_ppm * ratio_n / (1.0 + ratio_n)
                else:
                    no2_ppm_r15_capped = 0.0
                no2_r15 = no2_ppm_r15_capped * 1e-6 * rho_m * phi['NO2']
            r15_o2_presence = 1.0
            if model.r15_o2_presence_ref_ppm > 0.0:
                o2_feed_ppm_r15 = c_o2_feed / max(rho_m, 1e-9) * 1e6 if c_o2_feed else 0.0
                ratio = (o2_feed_ppm_r15 / model.r15_o2_presence_ref_ppm) ** model.r15_o2_presence_hill_n \
                    if o2_feed_ppm_r15 > 0.0 else 0.0
                r15_o2_presence = ratio / (1.0 + ratio)
            r15_n2o_brake = 1.0
            if model.r15_n2o_cap_ppm > 0.0:
                n2o_ppm_r15 = ppm['N2O'][i]
                r15_n2o_brake = 1.0 / (1.0 + (n2o_ppm_r15 / model.r15_n2o_cap_ppm) ** model.r15_n2o_cap_hill_n)
            rate['R15'][i] = r15_o2_presence * k['f_phase'] ** model.r15_f_phase_exponent * (r15_o2_inhib * r15_o2_activation * r15_n2o_brake * model._r15_surface_suppression() * model._sulfur_catalyst_gate(c_h2s_raw, c_h2so4_raw) * k.get('k15_f', 0.0) * no2_r15**4 - k.get('k15_r', 0.0) * n2o**2 * o2**3)

            # Wall/corrosion severity uses the raw (mole-fraction-based) concentrations,
            # matching rhs(), not the phi-scaled reactive concentrations used just above for
            # the homogeneous R1-R13.
            h2s_raw = ppm['H2S'][i] * 1e-6 * rho_m
            no2_raw = ppm['NO2'][i] * 1e-6 * rho_m
            no_raw = ppm['NO'][i] * 1e-6 * rho_m
            h2so4_raw = ppm['H2SO4'][i] * 1e-6 * rho_m
            hno3_raw = ppm['HNO3'][i] * 1e-6 * rho_m
            cum_no2_i = self.wall_solid['CumNO2Exposure'][i] if 'CumNO2Exposure' in self.wall_solid else 0.0
            lagged_o2_i = self.wall_solid['LaggedO2'][i] if 'LaggedO2' in self.wall_solid else None
            lagged_o2_feed_i = self.wall_solid['LaggedO2Feed'][i] if 'LaggedO2Feed' in self.wall_solid else None
            cum_o2_i = self.wall_solid['CumO2Exposure'][i] if 'CumO2Exposure' in self.wall_solid else 0.0
            so2_raw = ppm['SO2'][i] * 1e-6 * rho_m
            wall = model.get_wall_deposit_rates(o2, ppm['H2O'][i], no2_raw, h2so4_raw, hno3_raw,
                                                 C_H2S=h2s_raw, cum_no2_exposure=cum_no2_i,
                                                 C_O2_feed=c_o2_feed, C_O2_lagged=lagged_o2_i,
                                                 C_SO2=so2_raw, cum_o2_exposure=cum_o2_i,
                                                 C_NO=no_raw, C_O2_feed_lagged=lagged_o2_feed_i)
            rate['Wall O2 / Fe2O3'][i] = wall['r_wall_o2']
            rate['FeCO3 deposit'][i] = wall['r_feco3']
            rate['Wall HNO3 / Fe(NO3)2'][i] = wall['r_hno3_corrosion']
            rate['Wall H2SO4 / FeSO4'][i] = wall['r_h2so4']
            rate['Wall NO2 / Fe2O3'][i] = wall['r_wall_no2']
            rate['Wall S8 / Claus'][i] = wall['r_wall_s8']
            rate['Wall SO2 / H2SO4'][i] = wall['r_wall_so2']

        self._reaction_rate_series = rate
        return rate

    def _integrated_extent_mmol(self, series, mask):
        if not np.any(mask):
            return 0.0
        t_s = self.t_h * 3600.0
        V_m3 = self.volume_ml * 1e-6
        return np.trapezoid(np.abs(series[mask]), t_s[mask]) * V_m3 * 1e9

    def get_reaction_table(self, min_share_pct=1.0):
        """Overall reaction-activity ranking integrated over the whole run."""
        rate = self._compute_reaction_rate_series()
        full_mask = np.ones_like(self.t_h, dtype=bool)
        phase_masks = self._phase_masks()

        total_extent = {name: self._integrated_extent_mmol(series, full_mask)
                        for name, series in rate.items()}
        total_turnover = sum(total_extent.values()) or 1.0

        rows = []
        for name, extent in total_extent.items():
            share = 100.0 * extent / total_turnover
            if extent <= 0.0 or share < min_share_pct:
                continue
            phase_extents = [self._integrated_extent_mmol(rate[name], m) for m in phase_masks]
            peak_phase = int(np.argmax(phase_extents))
            rows.append({
                'Reaction': name,
                'Net reaction': self.REACTION_NAMES[name],
                'Integrated extent (mmol)': extent,
                'Share of modeled turnover (%)': share,
                'Most active step': self.phases[peak_phase][0],
            })
        table = pd.DataFrame(rows)
        if not table.empty:
            table = table.sort_values('Integrated extent (mmol)', ascending=False).reset_index(drop=True)
        return table

    def get_step_reaction_table(self, min_share_pct=2.0, top_n=3):
        """Top reaction pathways within each individual phase (A, B, C, ...)."""
        rate = self._compute_reaction_rate_series()
        phase_masks = self._phase_masks()

        rows = []
        for (name, _), (start_h, end_h), mask in zip(self.phases, self.phase_bounds, phase_masks):
            extents = {r: self._integrated_extent_mmol(series, mask) for r, series in rate.items()}
            step_turnover = sum(extents.values())
            if step_turnover <= 0.0:
                continue
            ranked = sorted(extents.items(), key=lambda kv: kv[1], reverse=True)
            material = [(r, e) for r, e in ranked if 100.0 * e / step_turnover >= min_share_pct][:top_n]
            if not material:
                material = ranked[:1]
            for r, extent in material:
                rows.append({
                    'Step': name,
                    'Time (h)': f'{start_h:g}-{end_h:g}',
                    'Reaction': r,
                    'Net reaction': self.REACTION_NAMES[r],
                    'Integrated extent (mmol)': extent,
                    'Share within step (%)': 100.0 * extent / step_turnover,
                })
        return pd.DataFrame(rows)

    def get_surface_data(self):
        """Corrosion-product mass accumulation and corrosion rate vs time.

        ``Fe2O3_mg``/``FeCO3_mg``/``FeNO32_mg``/``FeSO4_mg``/``Fe_lost_mg`` and
        ``corrosion_rate_mm_yr`` (its time-gradient) are all mass-balanced against the wall
        reactions' extents (Fe consumed = solid product formed). Under the general reference
        profile ``Fe2O3_mg`` collects BOTH active Fe2O3 paths (O2 via ``wall_k_intrinsic``, and
        NO2 via ``wall_no2_k_intrinsic``), and ``FeNO32_mg``/``FeSO4_mg`` come from the HNO3 and
        H2SO4 paths. Only ``FeCO3_mg`` is 0 -- the carbonic-acid path is disabled
        (``wall_feco3_k_intrinsic=0.0``) and remains available via
        ``configure_wall_corrosion(feco3_k_intrinsic=...)`` when required.
        """
        model = self.exp.model
        rate = self._compute_reaction_rate_series()
        t_h = self.t_h
        t_s = t_h * 3600.0
        dt = np.diff(t_s, prepend=0.0)
        V_m3 = self.volume_ml * 1e-6

        n_o2_lost = np.cumsum(rate['Wall O2 / Fe2O3'] * V_m3 * dt)     # kmol
        n_fe2o3 = n_o2_lost * (2.0 / 3.0)                               # 4 Fe + 3 O2 -> 2 Fe2O3
        n_no2_lost = np.cumsum(rate['Wall NO2 / Fe2O3'] * V_m3 * dt)   # kmol NO2 consumed
        n_fe2o3 += n_no2_lost * (1.0 / 3.0)                             # 2 Fe + 3 NO2 -> Fe2O3 + 3 NO
        n_feco3 = np.cumsum(rate['FeCO3 deposit'] * V_m3 * dt)         # Fe + CO2(aq) + H2O -> FeCO3
        # FeSO4/Fe(NO3)2 are tracked as real ODE states (see rhs()/simulate()) -- use the true
        # solved trajectory directly rather than re-integrating the reconstructed rate series.
        n_fe_no3_2 = self.wall_solid['FeNO32'] * V_m3      # 8 HNO3+3Fe -> 3 Fe(NO3)2+2NO+4H2O
        n_fe_so4 = self.wall_solid['FeSO4'] * V_m3          # Fe + H2SO4 -> FeSO4
        n_fe_total = n_o2_lost * (4.0 / 3.0) + n_no2_lost * (2.0 / 3.0) + n_feco3 + n_fe_no3_2 + n_fe_so4

        fe2o3_mg = n_fe2o3 * self.M_FE2O3 * 1e6
        feco3_mg = n_feco3 * self.M_FECO3 * 1e6
        fe_no3_2_mg = n_fe_no3_2 * self.M_FE_NO3_2 * 1e6
        fe_so4_mg = n_fe_so4 * self.M_FE_SO4 * 1e6
        fe_lost_mg = n_fe_total * self.M_FE * 1e6

        fe_lost_kg = fe_lost_mg / 1e6
        d_fe_dt_kg_s = np.gradient(fe_lost_kg, t_s, edge_order=1)
        area = model.wall_area_m2
        depth_rate_m_s = d_fe_dt_kg_s / (area * self.RHO_FE_KG_M3) if area > 0 else np.zeros_like(t_h)
        corrosion_rate_mm_yr = np.clip(depth_rate_m_s * 3600.0 * 24.0 * 365.25 * 1000.0, 0.0, None)

        # Acid-history enhancement actually driving _wall_o2_rate: instantaneous NO2+H2SO4+HNO3
        # gas-phase loading (see _acid_enhancement), not a cumulative or solid-product tracker.
        total_acid_ppm = self.ppm['NO2'] + self.ppm['H2SO4'] + self.ppm['HNO3']
        enhancement = model.wall_acid_background + model.wall_acid_gain * total_acid_ppm ** model.wall_acid_exponent

        return pd.DataFrame({
            'time_hours': t_h,
            'Fe2O3_mg': fe2o3_mg,
            'FeCO3_mg': feco3_mg,
            'FeNO32_mg': fe_no3_2_mg,
            'FeSO4_mg': fe_so4_mg,
            'Fe_lost_mg': fe_lost_mg,
            'corrosion_rate_mm_yr': corrosion_rate_mm_yr,
            'acid_enhancement': enhancement,
        })

    def get_mass_balance_table(self):
        """Per-phase N/S/H/O atom-balance closure check: fed in = outflow + gas-phase
        accumulation + wall-solid deposit, for every element.

        For each A-I phase and each element E in {N, S, H, O}, computes (all in mmol):
          - ``fed``: atoms entering with the feed stream over the phase duration.
          - ``outflow``: atoms leaving with the CSTR outflow (trapezoidal integral of
            outlet ppm(t) x molar throughput over the phase).
          - ``accumulation``: change in the vessel's own gas-phase inventory of that element
            (end of phase minus start of phase).
          - ``wall_deposit``: change in that element's content of the four solid corrosion
            products (Fe2O3, FeCO3, Fe(NO3)2, FeSO4) over the phase -- this is what lets the
            balance close even though the reaction network moves atoms from the tracked gas
            species into an untracked solid phase.
          - ``residual`` = fed - outflow - accumulation - wall_deposit, and ``residual_pct``
            relative to ``fed`` (or ``NaN`` when nothing of that element was fed) -- should be
            close to zero; a large residual points at a stoichiometry bug in ``rhs()``.

        Uses each phase's molar throughput (`mass_flow_g_h` / ``MW_CO2``) for every species
        since CO2 is the overwhelming bulk carrier gas and the impurities are trace-level. Known
        approximation: the very first phase of any experiment includes an initial vessel-fill
        sub-period with no outflow yet (see ``run_experiment``'s ``inflow_only`` stage); this
        method still assumes full steady outflow throughout, so the first phase's residual is
        typically larger (the fill sub-period's assumed-but-nonexistent outflow) than later
        phases, which close to within ~1%.
        """
        if self.results is None:
            raise RuntimeError('Call run() before get_mass_balance_table().')

        model = self.exp.model
        t_h, t_s = self.t_h, self.t_h * 3600.0
        V_m3 = self.volume_ml * 1e-6
        molar_flow_kmol_s = np.zeros_like(t_h)
        for phase, (t0, t1) in zip(self.exp.phases, self.phase_bounds):
            phase_flow_kmol_s = (phase['mass_flow_g_h'] / 1000.0) / MW_CO2 / 3600.0
            molar_flow_kmol_s[(t_h >= t0) & (t_h <= t1)] = phase_flow_kmol_s
        elements = ('N', 'S', 'H', 'O')

        # Gas-phase inventory and outflow flux of each element vs time.
        inventory_kmol = {e: np.zeros_like(t_h) for e in elements}
        outflow_kmol_s = {e: np.zeros_like(t_h) for e in elements}
        for species in model.SPECIES:
            counts = SPECIES_ATOM_COUNTS.get(species, {})
            if not counts:
                continue
            mole_fraction = self.ppm[species] * 1e-6
            C_kmol_m3 = mole_fraction * model.molar_density
            mole_flow_kmol_s = mole_fraction * molar_flow_kmol_s
            for e, n_atoms in counts.items():
                inventory_kmol[e] += n_atoms * C_kmol_m3 * V_m3
                outflow_kmol_s[e] += n_atoms * mole_flow_kmol_s

        # Element content of the four solid wall-corrosion products vs time.
        surf = self.get_surface_data()
        n_fe_no3_2 = surf['FeNO32_mg'] / (self.M_FE_NO3_2 * 1e6)   # kmol Fe(NO3)2
        n_fe_so4 = surf['FeSO4_mg'] / (self.M_FE_SO4 * 1e6)        # kmol FeSO4
        n_feco3 = surf['FeCO3_mg'] / (self.M_FECO3 * 1e6)          # kmol FeCO3
        n_fe2o3 = surf['Fe2O3_mg'] / (self.M_FE2O3 * 1e6)          # kmol Fe2O3
        wall_kmol = {
            'N': 2.0 * n_fe_no3_2.to_numpy(),
            'S': 1.0 * n_fe_so4.to_numpy(),
            'H': np.zeros_like(t_h),
            'O': (6.0 * n_fe_no3_2 + 4.0 * n_fe_so4 + 3.0 * n_feco3 + 3.0 * n_fe2o3).to_numpy(),
        }

        rows = []
        for (name, feed_ppm), phase, (t0, t1) in zip(self.phases, self.exp.phases, self.phase_bounds):
            i0 = int(np.searchsorted(t_h, t0))
            i1 = int(np.searchsorted(t_h, t1, side='right')) - 1
            i1 = max(i1, i0)
            duration_s = (t1 - t0) * 3600.0
            phase_molar_flow_kmol_s = (phase['mass_flow_g_h'] / 1000.0) / MW_CO2 / 3600.0

            for e in elements:
                fed_mmol = sum(
                    feed_ppm.get(sp, 0.0) * 1e-6 * phase_molar_flow_kmol_s * duration_s * counts[e]
                    for sp, counts in SPECIES_ATOM_COUNTS.items() if e in counts
                ) * 1e6  # kmol -> mmol

                outflow_mmol = _trapz(outflow_kmol_s[e][i0:i1 + 1], t_s[i0:i1 + 1]) * 1e6
                accumulation_mmol = (inventory_kmol[e][i1] - inventory_kmol[e][i0]) * 1e6
                wall_mmol = (wall_kmol[e][i1] - wall_kmol[e][i0]) * 1e6

                residual_mmol = fed_mmol - outflow_mmol - accumulation_mmol - wall_mmol
                residual_pct = (residual_mmol / fed_mmol * 100.0) if abs(fed_mmol) > 1e-30 else np.nan

                rows.append({
                    'Phase': name, 'Time (h)': f'{t0:g}-{t1:g}', 'Element': e,
                    'Fed (mmol)': fed_mmol, 'Outflow (mmol)': outflow_mmol,
                    'Accumulation (mmol)': accumulation_mmol, 'Wall deposit (mmol)': wall_mmol,
                    'Residual (mmol)': residual_mmol, 'Residual (%)': residual_pct,
                })

        return pd.DataFrame(rows)

    def get_wash_water_pH_table(self, water_mass_g, wash_temp_C=25.0, co2_partial_pressure_atm=1.0):
        """Estimate the pH of a fixed mass of wash water continuously scrubbing the reactor's
        CO2 off-gas, vs time (illustrative, screening-level).

        Physical model:
          - The off-gas is almost pure CO2 (ppm-level impurities), so it is treated as an OPEN
            system: bubbling it through the wash water holds dissolved CO2 at its Henry's-law
            equilibrium (``[CO2(aq)] = KH(T) * co2_partial_pressure_atm``) for the entire run,
            then buffered through the standard two-step carbonic-acid equilibrium
            (CO2(aq)+H2O <-> H+ + HCO3- <-> 2H+ + CO3^2-). This does NOT accumulate with time,
            since the gas supply is effectively infinite relative to a small water sample.
          - NH3, H2SO4 and HNO3 are assumed to be retained quantitatively -- negligible vapor
            pressure back to the gas phase -- so their absorbed amount ACCUMULATES in the fixed
            water mass over time. Each species' cumulative moles absorbed is the time-integral
            of (outlet ppm x CO2 molar throughput), divided by the (assumed constant, non-
            evaporating) water volume.
          - pH is then the root of the H+ / OH- / HCO3- / CO3^2- / NH4+ / HSO4- / SO4^2- / NO3-
            charge balance at every time point (see ``_solve_wash_water_pH``).
          - Deliberate simplification (flagged, not silent): H2S, SO2, NO2, O2, NO, N2O, S8 and
            H2 are excluded from the charge balance -- only CO2 plus the 3 named species are
            modeled.

        Parameters
        ----------
        water_mass_g : float
            Mass of (initially pure) wash water the whole run's off-gas is bubbled through.
        wash_temp_C : float, default 25.0
            Wash-water temperature, independent of the reactor's own operating temperature
            (washing is treated as a separate bench-scale step at ambient conditions).
        co2_partial_pressure_atm : float, default 1.0
            CO2 partial pressure seen by the wash water (vented to ~ambient pressure).
        """
        if self.results is None:
            raise RuntimeError('Call run() before get_wash_water_pH_table().')
        if water_mass_g <= 0.0:
            raise ValueError(f'water_mass_g must be positive, got {water_mass_g}')

        t_h, t_s = self.t_h, self.t_h * 3600.0
        ppm = self.ppm

        molar_flow_kmol_s = np.zeros_like(t_h)
        for phase, (t0, t1) in zip(self.exp.phases, self.phase_bounds):
            phase_flow_kmol_s = (phase['mass_flow_g_h'] / 1000.0) / MW_CO2 / 3600.0
            molar_flow_kmol_s[(t_h >= t0) & (t_h <= t1)] = phase_flow_kmol_s

        water_l = water_mass_g / 1000.0  # rho_water ~= 1.0 g/mL
        cum_mol_l = {}
        for species in ('NH3', 'H2SO4', 'HNO3'):
            mole_flow_kmol_s = ppm[species] * 1e-6 * molar_flow_kmol_s
            cum_kmol = _cumulative_trapz(mole_flow_kmol_s, t_s)
            cum_mol_l[species] = np.clip(cum_kmol * 1000.0 / water_l, 0.0, None)  # kmol -> mol, / L

        temp_kelvin = wash_temp_C + 273.15
        co2_aq_mol_l = _van_t_hoff(KH_CO2_298_MOL_L_ATM, DH_KH_CO2, temp_kelvin) * co2_partial_pressure_atm

        pH = np.array([
            _solve_wash_water_pH(co2_aq_mol_l, cum_mol_l['NH3'][i], cum_mol_l['H2SO4'][i],
                                  cum_mol_l['HNO3'][i], temp_kelvin=temp_kelvin)
            for i in range(len(t_h))
        ])

        return pd.DataFrame({
            'time_hours': t_h,
            'CO2_aq_mol_L': np.full_like(t_h, co2_aq_mol_l),
            'NH3_total_mol_L': cum_mol_l['NH3'],
            'H2SO4_total_mol_L': cum_mol_l['H2SO4'],
            'HNO3_total_mol_L': cum_mol_l['HNO3'],
            'pH': pH,
        })

    def get_autoclave_wash_table(self, wash_mass_g):
        """Estimate ion-chromatography-style SO4^2-/NO3-/NH4+ results for a fixed mass of water
        used to rinse the autoclave's OWN internals after the run, vs time (illustrative,
        screening-level; companion to ``get_wash_water_pH_table``, which instead models washing
        the CO2 OFF-GAS through an external bottle).

        Physical model: treats ALL H2SO4, HNO3 and NH3 ever chemically formed by the reaction
        network as having stayed inside the vessel (zero net transport out with the CO2 outflow, and no further consumption by any other pathway --
        e.g. wall corrosion consuming H2SO4/HNO3, or R10 consuming NH3) -- i.e. reads directly
        from the model's own cumulative "total ever produced" ODE states (``CumH2SO4``/
        ``CumHNO3``/``CumNH3``, see ``EXTRA_STATE_KEYS``), each a never-decreasing kmol/m^3-
        equivalent concentration. Multiplying by the reactor's own volume gives total moles ever
        formed at each time point; dividing by the wash water's volume gives the IC-style
        concentration a lab would measure after rinsing the vessel with ``wash_mass_g`` grams of
        water. Reported as the measured ionic species (SO4^2-, NO3-, NH4+), matching real
        ion-chromatography output, not the neutral parent acid/base (H2SO4/HNO3/NH3).

        Parameters
        ----------
        wash_mass_g : float
            Mass of (initially pure) water used to rinse the autoclave internals.
        """
        if self.results is None:
            raise RuntimeError('Call run() before get_autoclave_wash_table().')
        if wash_mass_g <= 0.0:
            raise ValueError(f'wash_mass_g must be positive, got {wash_mass_g}')

        t_h = self.t_h
        V_m3 = self.volume_ml * 1e-6
        wash_l = wash_mass_g / 1000.0  # rho_water ~= 1.0 g/mL

        cum_h2so4_kmol = self.wall_solid['CumH2SO4'] * V_m3
        cum_hno3_kmol = self.wall_solid['CumHNO3'] * V_m3
        cum_nh3_kmol = self.wall_solid.get('CumNH3', np.zeros_like(t_h)) * V_m3

        # kmol -> mol (x1e3) -> umol (x1e6): combined factor x1e9.
        so4_umol = cum_h2so4_kmol * 1e9
        no3_umol = cum_hno3_kmol * 1e9
        nh4_umol = cum_nh3_kmol * 1e9

        # umol * g/mol = ug; x1e-3 -> mg; / L -> mg/L.
        so4_mg_l = so4_umol * M_SO4_2MINUS * 1e-3 / wash_l
        no3_mg_l = no3_umol * M_NO3_MINUS * 1e-3 / wash_l
        nh4_mg_l = nh4_umol * M_NH4_PLUS * 1e-3 / wash_l

        return pd.DataFrame({
            'time_hours': t_h,
            'SO4_mg_L': so4_mg_l, 'NO3_mg_L': no3_mg_l, 'NH4_mg_L': nh4_mg_l,
            'SO4_umol': so4_umol, 'NO3_umol': no3_umol, 'NH4_umol': nh4_umol,
        })

    def get_autoclave_wash_summary(self, wash_mass_g):
        """End-of-run snapshot of ``get_autoclave_wash_table`` laid out like a lab's ion-
        chromatography report: one row of mg/L results, one row of total umol present in
        ``wash_mass_g`` grams of wash water, columns SO4^2-/NO3-/NH4+."""
        final = self.get_autoclave_wash_table(wash_mass_g=wash_mass_g).iloc[-1]
        return pd.DataFrame(
            {
                'SO4^2-': [final['SO4_mg_L'], final['SO4_umol']],
                'NO3-': [final['NO3_mg_L'], final['NO3_umol']],
                'NH4+': [final['NH4_mg_L'], final['NH4_umol']],
            },
            index=[
                'IC analysis result (mg/l)',
                f'IC analysis, total in {wash_mass_g:g} g water (umol)',
            ],
        )

    # ---------------------------------------------------------------------- plots
    def _condition_string(self):
        return (f'{self.volume_ml:g} mL, {self.mass_flow_g_h:g} g/hr CO2, '
                f'{self.temp_C:+g} \u00b0C / {self.pressure_bar:g} bar, '
                f'{self.material.replace("_", " ")}')

    def _shade_phases(self, ax, y_top=None):
        for i, (t0, t1) in enumerate(self.phase_bounds):
            ax.axvspan(t0, t1, color=self.PHASE_COLORS[i % len(self.PHASE_COLORS)], alpha=0.55, zorder=0)
            ax.axvline(t1, color='0.7', lw=0.7, zorder=1)
        if y_top is not None:
            max_len = max(len(name) for name, _ in self.phases)
            fontsize = 8 if max_len <= 20 else (7 if max_len <= 30 else 6)
            rotation = 0 if max_len <= 25 else 20
            for (t0, t1), (name, _) in zip(self.phase_bounds, self.phases):
                ax.text(0.5 * (t0 + t1), y_top, name, ha='center', va='top',
                        fontsize=fontsize, color='#2c3e50', rotation=rotation)

    def _feed_profile(self, species):
        t_pts, y_pts, cum = [], [], 0.0
        for (_, feed), dur in zip(self.phases, self.phase_durations):
            val = float(feed.get(species, 0.0))
            t_pts.extend([cum, cum + dur])
            y_pts.extend([val, val])
            cum += dur
        return np.array(t_pts), np.array(y_pts)

    def set_reactant_plot_limits(self, left=None, right=None):
        """Set reactant-axis upper limits; use ``None`` to restore automatic scaling.

        For example: ``autoclave.set_reactant_plot_limits(left=20, right=800)``.
        """
        for name, limit in (('left', left), ('right', right)):
            if limit is not None and float(limit) <= 0.0:
                raise ValueError(f'{name} plot limit must be positive or None.')
        self._reactant_plot_limits = (left, right)
        return self

    def _reactant_ylim(self, species, limit):
        if limit is not None:
            return (0.0, float(limit))
        peak = max(
            max((float(feed.get(sp, 0.0)) for _, feed in self.phases), default=0.0)
            for sp in species
        )
        peak = max(peak, max(float(np.max(self.ppm[sp])) for sp in species))
        return (0.0, max(peak * 1.15, 1.0))

    def _plot_reactant_species(self, left_species=('H2S', 'SO2', 'NO2'), right_species=('H2O', 'O2'),
                                left_ylim=None, right_ylim=None):
        t_h, ppm = self.t_h, self.ppm
        fig, ax = plt.subplots(figsize=(12, 5.2))
        ax_r = ax.twinx()
        ax_r.grid(False)
        ax_r.spines['top'].set_visible(False)

        lines = []
        for sp in left_species:
            color = self.REACTANT_STYLE[sp]
            tf, yf = self._feed_profile(sp)
            lines += ax.plot(tf, yf, color=color, ls='--', lw=1.3, alpha=0.9, label=f'{sp} feed')
            lines += ax.plot(t_h, ppm[sp], color=color, ls='-', lw=2.0, label=f'{sp} outlet')
        for sp in right_species:
            color = self.REACTANT_STYLE[sp]
            tf, yf = self._feed_profile(sp)
            lines += ax_r.plot(tf, yf, color=color, ls='--', lw=1.3, alpha=0.9, label=f'{sp} feed (R)')
            lines += ax_r.plot(t_h, ppm[sp], color=color, ls='-', lw=2.0, label=f'{sp} outlet (R)')

        configured_left, configured_right = self._reactant_plot_limits
        ax.set_ylim(*self._reactant_ylim(left_species,
                                         configured_left if left_ylim is None else left_ylim))
        ax_r.set_ylim(*self._reactant_ylim(right_species,
                                           configured_right if right_ylim is None else right_ylim))
        self._shade_phases(ax, y_top=ax.get_ylim()[1])

        ax.set_title(f'Reactant species \u2014 dashed = injected, solid = outlet\n{self._condition_string()}')
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('H2S / SO2 / NO2  (ppm)')
        ax_r.set_ylabel('H2O / O2  (ppm)  \u2014 right axis', color='#555')
        ax_r.tick_params(axis='y', colors='#555')
        ax.set_xlim(0, t_h[-1])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(lines, [l.get_label() for l in lines], loc='lower center',
                  bbox_to_anchor=(0.5, 1.14), ncol=5, frameon=True, fontsize=9)
        plt.tight_layout()
        plt.show()
        return fig, (ax, ax_r)

    def _plot_reaction_products(self):
        t_h, ppm = self.t_h, self.ppm
        fig, ax = plt.subplots(figsize=(12, 4.5))
        for sp, (color, ls) in self.PRODUCT_STYLE.items():
            ax.plot(t_h, ppm[sp], color=color, ls=ls, lw=2.2, label=sp)
        ax.fill_between(t_h, ppm['H2SO4'], color=self.PRODUCT_STYLE['H2SO4'][0], alpha=0.18)

        y_top = max(np.max(ppm[s]) for s in self.PRODUCT_STYLE) * 1.25 + 1e-6
        ax.set_ylim(0, max(y_top, 1.0))
        self._shade_phases(ax, y_top=ax.get_ylim()[1])

        ax.set_title(f'Reaction products in the autoclave\n{self._condition_string()}')
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Concentration (ppm)')
        ax.set_xlim(0, t_h[-1])
        ax.legend(loc='upper left', ncol=5, frameon=True)
        plt.tight_layout()
        plt.show()
        return fig, ax

    def _plot_surface_products(self):
        surf = self.get_surface_data()
        area_cm2 = self.exp.model.wall_area_m2 * 1e4
        k_wall = self.exp.model.wall_hno3_corrosion_k_intrinsic

        fig, ax1 = plt.subplots(figsize=(12, 4.5))
        if self.exp.model.wall_k_intrinsic > 0.0:
            ax1.plot(surf['time_hours'], surf['Fe2O3_mg'], color='#c0392b', lw=2.2, label='Fe$_2$O$_3$ (O$_2$ attack)')
        if self.exp.model.wall_feco3_k_intrinsic > 0.0:
            ax1.plot(surf['time_hours'], surf['FeCO3_mg'], color='#2c3e50', lw=2.2,
                     label='FeCO$_3$ (carbonic-acid attack)')
        if self.exp.model.wall_hno3_corrosion_k_intrinsic > 0.0:
            ax1.plot(surf['time_hours'], surf['FeNO32_mg'], color='#8e44ad', lw=2.2,
                     label='Fe(NO$_3$)$_2$ (HNO$_3$ attack)')
        if self.exp.model.wall_h2so4_k_intrinsic > 0.0:
            ax1.plot(surf['time_hours'], surf['FeSO4_mg'], color='#d4a017', lw=2.2,
                     label='FeSO$_4$ (H$_2$SO$_4$ attack)')
        ax1.plot(surf['time_hours'], surf['Fe_lost_mg'], color='#7f8c8d', lw=1.6, ls='--', label='Fe lost (total)')
        ax1.set_ylabel('Cumulative mass (mg)')
        ax1.set_xlabel('Time (h)')
        self._shade_phases(ax1, y_top=None)
        ax1.set_title(f'Surface reaction products on {self.material.replace("_", " ")} coupon '
                      f'(A = {area_cm2:.2f} cm$^2$, k$_{{wall,HNO_3}}$ = {k_wall:.1e} mol/(m$^2$ s ppm))')
        ax1.legend(loc='upper left', frameon=True)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return fig, ax1

    def _plot_corrosion_rate(self):
        surf = self.get_surface_data()
        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.plot(surf['time_hours'], surf['corrosion_rate_mm_yr'], color='#e67e22', lw=2)
        ax.set_ylabel('Corrosion rate (mm/yr)')
        ax.set_xlabel('Time (h)')
        self._shade_phases(ax, y_top=None)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Equivalent corrosion penetration rate\n{self._condition_string()}')
        plt.tight_layout()
        plt.show()
        return fig, ax

    def _plot_wash_water_pH(self, water_mass_g=30.0, wash_temp_C=25.0, co2_partial_pressure_atm=1.0):
        wash_df = self.get_wash_water_pH_table(
            water_mass_g=water_mass_g, wash_temp_C=wash_temp_C,
            co2_partial_pressure_atm=co2_partial_pressure_atm)

        fig, ax = plt.subplots(figsize=(12, 4.0))
        ax.plot(wash_df['time_hours'], wash_df['pH'], color='#16a085', lw=2.2)
        ax.axhline(7.0, color='0.6', lw=1.0, ls=':', label='neutral (pH 7)')
        ax.set_ylabel('Wash-water pH')
        ax.set_xlabel('Time (h)')
        self._shade_phases(ax, y_top=None)
        ax.set_xlim(0, self.t_h[-1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', frameon=True)
        ax.set_title(f'Wash-water pH ({water_mass_g:g} g water, {wash_temp_C:g} \u00b0C, CO2 '
                     f'Henry\u2019s-law equilibrium + NH3/H2SO4/HNO3 fully retained)\n'
                     f'{self._condition_string()}')
        plt.tight_layout()
        plt.show()
        return fig, ax

    _PLOT_KINDS = {
        'reactant_species': _plot_reactant_species,
        'reactants': _plot_reactant_species,
        'reaction_products': _plot_reaction_products,
        'products': _plot_reaction_products,
        'surface_reaction_products': _plot_surface_products,
        'surface_products': _plot_surface_products,
        'corrosion_rate': _plot_corrosion_rate,
        'corrosion': _plot_corrosion_rate,
        'wash_water_ph': _plot_wash_water_pH,
        'ph': _plot_wash_water_pH,
    }

    def build_plot(self, kind, **kwargs):
        """Render one of: 'reactant species', 'reaction products',
        'surface reaction products', 'corrosion rate'."""
        if self.results is None:
            raise RuntimeError('Call run() before build_plot().')
        key = kind.strip().lower().replace(' ', '_')
        method = self._PLOT_KINDS.get(key)
        if method is None:
            raise ValueError(f"Unknown plot kind {kind!r}. Choose from: "
                              f"{sorted(set(self._PLOT_KINDS))}")
        return method(self, **kwargs)
