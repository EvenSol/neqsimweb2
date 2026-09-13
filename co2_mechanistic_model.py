"""Screening kinetics for trace impurities and wall corrosion in CO2.

Uses NeqSim thermodynamic properties when available, with an approximate fallback,
and sequential fixed-pressure CSTR simulation. Species concentrations are kmol/m3,
kinetic time is seconds, and feed concentrations are ppm-mol.

Reference kinetics and surface activities are empirical and require independent
validation. This is not qualified design data or a rigorous multiphase reaction
equilibrium model. Pressurization and retained liquid-film inventories are not modeled.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

try:
    from scipy.integrate import solve_ivp
    from scipy.optimize import approx_fprime
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


R_GAS = 8.314462618             # Universal Gas Constant [J / (mol * K)]
BIMOLECULAR_ENCOUNTER_RATE_REF = 1.0e11
MW_CO2 = 44.0095                # Molar Mass of CO2 [g / mol]
MW_N2 = 28.0134                 # Molar Mass of N2 [g / mol]
MW_H2O = 18.0153                # Molar Mass of H2O [g / mol]
MW_HNO3 = 63.0129               # Molar Mass of HNO3 [g / mol]
MW_H2SO4 = 98.079                # Molar Mass of H2SO4 [g / mol]
MW_NH3 = 17.0305                # Molar Mass of NH3 [g / mol]
P_CRIT_CO2_BAR = 73.8           # Critical Pressure of CO2 [bar]
T_CRIT_CO2_K = 304.13           # Critical Temperature of CO2 [K]


DEFAULT_KINETIC_PARAMS = {
    'R1':  {'name': 'SO2 + 0.5 O2 + H2O <-> H2SO4',           'A': 5.0e5,     'Ea': 30000.0, 'units': 'm3 / (kmol * s)'},
    'R2':  {'name': 'H2S + 3 NO2 <-> SO2 + H2O + 3 NO',       'A': 1.0e10,    'Ea': 30000.0, 'units': 'm3 / (kmol * s)'},
    'R3a': {'name': 'SO2 + NO2 + H2O <-> NO + H2SO4',         'A': 1.0e5,     'Ea': 35000.0, 'units': 'm3 / (kmol * s)'},
    'R4':  {'name': '2 NO + O2 <-> 2 NO2',                    'A': 500.0,     'Ea': -4400.0, 'units': 'm6 / (kmol2 * s)'},
    'R5':  {'name': '3 NO2 + H2O <-> 2 HNO3 + NO',            'A': 2.4e6,     'Ea': 28000.0, 'units': 'm3 / (kmol * s)'},
    'R7':  {'name': '5 H2S + 6 NO + 4 H2O -> 6 NH3 + 5 SO2',  'A': 2.0e6,     'Ea': 12000.0, 'units': 'm3 / (kmol * s)'},
    'R10': {'name': '4 NH3 + 4 NO + 3 O2 -> 4 N2O + 6 H2O',    'A': 1.0e2,     'Ea': 20000.0, 'units': 'm6 / (kmol2 * s)'},
    'R11': {'name': 'H2S + 2 NO -> N2O + 1/8 S8 + H2O',        'A': 1.0e2,     'Ea': 20000.0, 'units': 'm3 / (kmol * s)'},
    'R12': {'name': 'H2S + 2 O2 <-> H2SO4 (NO2-catalysed)',    'A': 1.0e14,    'Ea': 35000.0, 'units': 'm6 / (kmol2 * s), phase-scaled'},
    'R13': {'name': '4 NO2 + H2S <-> H2SO4 + 4 NO',            'A': 1.0e13,    'Ea': 30000.0, 'units': 'm4.5 / (kmol4.5 * s), phase-scaled'},
    'R15': {'name': '4 NO2 <-> 2 N2O + 3 O2',                  'A': 1.0e2,     'Ea': 40000.0, 'units': 'm3 / (kmol * s)'},
    'R16': {'name': '2 NO2 <-> N2O4',                          'A': 1.0e3,     'Ea': 0.0,     'units': 'm3 / (kmol * s)'},
    'R17': {'name': '2 NO2 + H2O <-> HNO3 + HNO2',             'A': 1.0e5,     'Ea': 20000.0, 'units': 'm3 / (kmol * s)'},
    'R18': {'name': 'HNO2 + 0.5 O2 -> HNO3',                   'A': 1.0e5,     'Ea': 20000.0, 'units': 'm3 / (kmol * s)'},
}

CARBON_STEEL_WET_CO2_KINETICS = {
    'R1':  {'A': 5.0e2, 'Ea_kJ_mol': 30.0},
    'R2':  {'A': 33402393.195301704, 'Ea_kJ_mol': 29.0},
    'R3a': {'A': 170000.0, 'Ea_kJ_mol': -5.0},
    'R4':  {'A': 1.068774571145334e16, 'Ea_kJ_mol': 40.0},
    'R5':  {'A': 4.839357043036789e-5, 'Ea_kJ_mol': -35.0},
    'R7':  {'A': 2.3542893999927066e25, 'Ea_kJ_mol': 98.0},
    'R10': {'A': 0.0,     'Ea_kJ_mol': 20.0},
    'R11': {'A': 0.0,     'Ea_kJ_mol': 20.0},
    'R12': {'A': 100.0, 'Ea_kJ_mol': -12.0},
        'R13': {'A': 19274337.70287051, 'Ea_kJ_mol': -48.0},
    'R15': {'A': 0.0,     'Ea_kJ_mol': 40.0},
    'R16': {'A': 1.0e3,   'Ea_kJ_mol': 0.0},
    'R17': {'A': 2.6973892231563043e31, 'Ea_kJ_mol': 130.0},
    'R18': {'A': 2.86211893023342e38, 'Ea_kJ_mol': 180.0},
    'r17_no2_order': 0.1,
    'r17_h2o_order': 0.1,
    'r17_wet_order': 2.0,
    'r17_reference_kmol_m3': 1.0e-5,
    'r17_floor_kmol_m3': 1.0e-7,
    'r17_no2_activation_reference_kmol_m3': 9.3e-6,
    'r17_no2_activation_hill_n': 14.0,
    'r17_dense_co2_inhibition_f_phase_ref': 0.4,
    'r17_dense_co2_inhibition_f_phase_hill_n': 3.0,
    'r17_dense_co2_inhibition_wet_ref': 0.4,
    'r17_dense_co2_inhibition_wet_hill_n': 2.0,
    'r15_f_phase_exponent': 1.0,
    'r15_o2_inhib_ref_ppm': 15.0,
    'r15_o2_inhib_hill_n': 2.0,
    'r15_o2_activation_ref_ppm': 0.0,
    'r15_o2_activation_hill_n': 2.0,
    'r15_no2_cap_ppm': 18.0,
    'r15_no2_cap_hill_n': 6.0,
    'r15_dimer_dh_kj_mol': 114.0,
    'r15_dimer_t_ref_k': 275.15,
    'r15_n2o_cap_ppm': 6.0,
    'r15_n2o_cap_hill_n': 20.0,
    'r15_o2_presence_ref_ppm': 0.0,
    'r15_o2_presence_hill_n': 1.0,
    'r11_o2_ref_ppm': 2.0,
    'r11_o2_hill_n': 2.0,
    'r11_o2_gain': 0.9,
    'r2_no2_boost_ref_ppm': 8.0,
    'r2_no2_boost_hill_n': 10.0,
    'r2_no2_boost_gain': 2.0,
    'r2_f_phase_exponent': 2.0,
    'r2_no2_excess_ratio_ref': 0.0,
    'r2_no2_excess_ratio_hill_n': 12.0,
    'o2_lag_tau_hours': 0.0,
    'o2_feed_lag_tau_hours': 20.0,
    'o2_feed_lag_rise_tau_hours': 1.0,
    'r12_density_independent': False,
    'r12_f_phase_exponent': 1.2,
    'r12_no2_order': 1.0,
    'r12_no2_saturation_kmol_m3': 1e-5,
    'r12_h2s_order': 1.35,
    'r12_reference_kmol_m3': 1e-5,
    'r12_floor_kmol_m3': 1e-12,
    'r13_no2_order': 4.0,
    'r13_no2_rate_order': 1.0,
    'r13_reference_kmol_m3': 1e-5,
    'r13_floor_kmol_m3': 1e-12,
    'h2s_no2_temperature': {'reference_K': 276.15, 'suppression_kj_mol': 90.0, 'onset_width_K': 5.0},
    'dilute_redox': {
        'rho_ref_kmol_m3': 5.0,
        'density_hill_n': 6.0,
        'r2_k_ref': 3300.0,
        'r2_ea_kj_mol': -10.0,
        'r2_no2_ref_kmol_m3': 9e-6,
        'r2_no2_gain': 5.0,
        'r2_no2_hill_n': 8.0,
        'r2_h2s_half_kmol_m3': 1e-8,
        'r4_k_ref': 2.65e10,
        'r4_ea_kj_mol': -4.4,
        'r4_no_half_kmol_m3': 5e-9,
        'r4_no_reference_kmol_m3': 2.44e-6,
        'r4_no_saturation_order': 2.0,
        'r13_branch_fraction': 0.05,
        'sulfur_acid_weight': 0.0,
    },

    'r5_no_activity': 1.0,
    'r5_no2_order': 2.2,
    'r5_reference_kmol_m3': 1e-5,
    'r5_floor_kmol_m3': 1e-12,
    'r7_no_order': 6.0,
    'r7_reference_kmol_m3': 1e-5,
    'r7_floor_kmol_m3': 1e-12,
    'r7_no_activation_reference_kmol_m3': 3e-7,
    'r7_no_activation_hill_n': 3.0,
    'r7_cold_availability': {'midpoint_K': 273.09848358815213, 'width_K': 2.2397382295291317},
    'r7_water_saturation': {'half_kmol_m3': 0.0005, 'reference_kmol_m3': 1e-3},
    'r4_no_activity': 0.015,
    'r4_o2_half_ppm': 30.0,
    'r4_surface_gain': 14.0,
    'r3a_bore_gain': 14.0,
    'r15_surface_suppress_gain': 60.0,
    'r15_sulfur_ref_ppm': 0.075,
    'r15_sulfur_hill_n': 4.0,
    'wall_o2_feed_o2_ref_ppm': 10.0,
    'wall_o2_feed_o2_hill_n': 8.0,
    'wall_o2_h2s_relief': 10.0,
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
    'r3a_h2s_feed_inhibition_ref_ppm': 0.2,
    'r3a_h2s_feed_inhibition_hill_n': 22.0,
    'r3a_h2s_feed_inhibition_no2_ref_ppm': 5.0,
    'r3a_h2s_feed_inhibition_no2_hill_n': 16.0,
    'r3a_h2s_feed_inhibition_gain': 0.0,
    'r2_o2_presence_ref_ppm': 0.0,
    'r2_o2_presence_hill_n': 1.0,
    'wall_no2_no_cap_ppm': 0.99,
    'wall_no2_no_cap_hill_n': 20.0,
    'wall_no2_no_cap_gas_weighted': True,
    'wall_no2_langmuir_half_ppm': 2.0,

    'r3a_no_escape_frac': 0.0,

    'r1_autocat_gain': 0.0,
    'r1_autocat_ref_ppm': 10.0,
    'r1_autocat_hill_n': 1.0,
    'r1_feed_o2_ref_ppm': 0.0,
    'r1_feed_o2_hill_n': 4.0,
    'r1_feed_o2_cap_ppm': 0.0,

    'r3a_autocat_gain': 0.0,
    'r3a_autocat_ref_ppm': 13.3,
    'r3a_autocat_hill_n': 8.0,
    'r3a_autocat_surface_suppress_gain': 20.0,
    'r3a_acid_film_k_ref': 1.0e4,
    'r4_acid_film_k_ref': 5.0e9,
    'r3a_acid_film_ea_kj_mol': 60.0,
    'r3a_acid_film_ref_ppm': 41.5,
    'r3a_acid_film_hill_n': 4.0,
    'r3a_environment': {
        'wetting_reference': 0.27,
        'wetting_order': 2.0,
        'history_fraction': 1.0,
        'current_acid_reference_ppm': 10.0,
        'oxygen_supply_fraction': 1.0,
        'base_so2_inhibition_ref_ppm': 1.0,
        'base_so2_inhibition_order': 3.6,
        'base_so2_inhibition_floor': 0.000058823529411764706,
        'base_so2_activation_ref_ppm': 1.665,
        'base_so2_activation_order': 6.0,
        'oxygen_presence_ref_ppm': 0.1,
        'oxygen_present_multiplier': 0.02,
    },
    'r3a_dilute_acid': {
        'k_ref': 4.845987896e9,
        'history_ref_ppm': 65.0,
        'history_order': 1.0,
        'no2_half_kmol_m3': 5e-7,
        'density_order': 2.0,
        'acid_ref_kmol_m3': 3.4495762e-5,
        'acid_order': 1.525456,
        'background_k_ref': 1.257677783e6,
        'h2s_half_kmol_m3': 2e-7,
        'acid_response_hours': 10.0,
        'history_fraction': 0.0,
        'nitric_history_gain': 0.0,
        'nitric_history_ref_ppm': 100.0,
        'nitric_history_order': 4.0,
    },
    'r3a_conditioned_acid': {
        'k_ref': 6.0e7,
        'ea_kj_mol': 90.0,
        'acid_ref_ppm': 0.2,
        'acid_order': 12.0,
        'history_ref_ppm': 20.0,
        'so2_ref_ppm': 0.7,
        'so2_order': 8.0,
        'no_ref_ppm': 0.3,
        'limit_initial_ppm_h': 0.03,
        'limit_max_ppm_h': 0.15,
        'limit_acid_ref_ppm': 20.0,
        'limit_acid_order': 1.0,
    },

    'acid_so2_sat_ref_ppm': 0.0,
    'acid_so2_sat_hill_n': 20.0,

    'condensation_exponent': 2.0,
    'rho_m_reference': 24.0,

    'wall_k_intrinsic': 3.0e-8,             # O2 path intrinsic rate [mol O2 / (m^2 s ppm)]
    'wall_o2_potency': 1.0,
    'wall_o2_f_phase_exponent': 2.0,
    'wall_o2_sat_ref': 0.0,
    'wall_acid_gain': 0.06,
    'wall_acid_exponent': 1.0,
    'wall_acid_background': 0.0,
    'wall_acid_gain2': 0.0012,
    'wall_acid_exponent2': 4.0,
    'wall_rho_pass': 5.0,
    'wall_hill_n': 3.0,
    'wall_consume_h2o': False,
    'wall_h2o_mode': 'wet_film',
    'wall_h2o_enhancement_factor': 1.0,
    'wall_h2o_deliq_ref_ppm': 30.0,
    'wall_h2o_hill_n': 1.0,
    'wall_h2o_excess_ref_ppm': 100.0,
    'wall_h2o_excess_exponent': 1.0,
    'wall_feco3_k_intrinsic': 0.0,
    'wall_feco3_potency': 1.0,
    'wall_hno3_corrosion_k_intrinsic': 3.0e-9,
    'wall_hno3_corrosion_potency': 3.5,
    'wall_h2so4_k_intrinsic': 6.0e-9,      # sulfuric-acid path intrinsic rate [mol Fe / (m^2 s ppm)]
    'wall_h2so4_potency': 1.0,
    'wall_no2_k_intrinsic': 1.0e-5,        # [mol NO2 / (m^2 s ppm)]
    'wall_no2_potency': 1.0,
    'wall_no2_o2_ref_ppm': 2.0,
    'wall_no2_o2_hill_n': 4.0,
    'wall_o2_gas_phase_gain': 10.0,
    'wall_gas_phase_gain': 600.0,
    'wall_gas_phase_rho_ref': 5.0,
    'wall_gas_phase_hill_n': 4.0,
    'wall_s8_k_intrinsic': 1.0e-9,         # [mol H2S / (m^2 s ppm)]
    'wall_s8_h2s_potency': 1.0,
    'wall_s8_o2_potency': 0.5,
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
DG_HNO3_STDGIBBS = -79.9e3
DG_NH3_STDGIBBS = -16.4e3       # NIST standard Gibbs energy of formation, NH3(g), 298 K
DG_N2O_STDGIBBS = 104.2e3       # NIST standard Gibbs energy of formation, N2O(g), 298 K
DG_N2O4_STDGIBBS = 97.9e3       # NIST standard Gibbs energy of formation, N2O4(g), 298 K
DG_HNO2_STDGIBBS = -42.97e3     # NIST standard Gibbs energy of formation, HNO2(g), 298 K
DG_S8_STDGIBBS = 0.0
_R4_SURFACE_SV_REF_CM_INV = 4.0 / 6.5 + 2.0 / (330.0 / (np.pi * (6.5**2) / 4.0))

MAX_KEQ_EXPONENT = 300.0
MIN_CONCENTRATION_FLOOR = 1e-25
MOISTURE_REF_PPM = 50.0         # Reference moisture concentration scale for hydration factor [ppm]
_WALL_NO2_T_REF_K = 249.15


def _fractional_activity(concentration, exponent, floor=MIN_CONCENTRATION_FLOOR):
    """Keep fractional-order depletion differentiable at numerical zero."""
    concentration = max(float(concentration), 0.0)
    return concentration * (concentration + floor) ** (exponent - 1.0)


class CO2ImpurityKineticsModel:
    """Screen impurity reactions in CO2 with empirical kinetic and surface models.

    Cumulative acid states are kinetic-history proxies, not retained acid mass.
    """

    SPECIES = (
        'H2S', 'SO2', 'NO2', 'NO', 'O2', 'H2O',
        'H2SO4', 'HNO3', 'S8', 'NH3', 'N2O', 'H2', 'N2O4', 'HNO2'
    )

    EXTRA_STATE_KEYS = ('FeSO4', 'FeNO32', 'CumH2SO4', 'CumHNO3', 'CumNO2Exposure', 'LaggedO2',
                        'CumO2Exposure', 'LaggedO2Feed', 'CumNH3', 'AcidSiteActivity')

    SUPPORTED_MATERIALS = ('carbon_steel', 'magnetite', 'stainless_steel', 'inert')

    def __init__(self, T_kelvin=298.15, P_bar=100.0, water_ppm=50.0, material='carbon_steel',
                 condensation_exponent=0.0,
                 rho_m_reference=24.0,
                 wall_area_m2=0.0,
                 wall_k_intrinsic=1.0e-4,
                 wall_o2_potency=1.0,
                 wall_o2_f_phase_exponent=0.0,
                 wall_o2_sat_ref=0.0,
                 wall_rho_pass=5.0,
                 wall_hill_n=3.0,
                 wall_k_h2o_ppm=3.0,
                 wall_acid_gain=1.0,
                 wall_acid_exponent=1.5,
                 wall_acid_background=0.02,
                 wall_acid_gain2=0.0,
                 wall_acid_exponent2=6.0,
                 wall_consume_h2o=False,
                 wall_h2o_mode='density',
                 wall_h2o_enhancement_factor=1.0,
                 wall_h2o_deliq_ref_ppm=30.0,
                 wall_h2o_hill_n=4.0,
                 wall_h2o_excess_ref_ppm=100.0,
                 wall_h2o_excess_exponent=1.0,
                 wall_feco3_k_intrinsic=0.0,
                 wall_feco3_potency=1.0,
                 wall_hno3_corrosion_k_intrinsic=0.0,
                 wall_hno3_corrosion_potency=1.0,
                 wall_h2so4_k_intrinsic=0.0,
                 wall_h2so4_potency=1.0,
                 wall_no2_k_intrinsic=0.0,
                 wall_no2_potency=1.0,
                 wall_no2_ea_kj_mol=0.0,
                 wall_no2_h2o_ppm_ref=0.0,
                 wall_no2_exposure_threshold_ppm_h=0.0,
                 wall_no2_exposure_hill_n=4.0,
                 wall_no2_o2_ref_ppm=0.0,
                 wall_no2_o2_hill_n=2.0,
                 wall_o2_gas_phase_gain=0.0,
                 wall_gas_phase_gain=0.0,
                 wall_gas_phase_rho_ref=5.0,
                 wall_gas_phase_hill_n=2.0,
                 wall_s8_k_intrinsic=0.0,
                 wall_s8_h2s_potency=1.0,
                 wall_s8_o2_potency=0.5,
                 wall_so2_k_intrinsic=0.0,
                 wall_so2_potency=1.0,
                 wall_so2_exposure_threshold_ppm_h=0.0,
                 wall_so2_exposure_hill_n=2.0,
                 r1_autocat_gain=0.0,
                 r1_autocat_ref_ppm=10.0,
                 r1_feed_o2_ref_ppm=0.0,
                 r1_feed_o2_hill_n=4.0,
                 r1_feed_o2_cap_ppm=0.0,
                 r3a_autocat_gain=0.0,
                 r3a_autocat_ref_ppm=10.0,
                 r3a_autocat_hill_n=1.0,
                 r3a_autocat_surface_suppress_gain=0.0,
                 acid_so2_sat_ref_ppm=0.0,
                 acid_so2_sat_hill_n=2.0,
                 r12_density_independent=False,
                 r12_f_phase_exponent=1.0,
                 r12_no2_order=1.0,
                 r17_dense_co2_inhibition_f_phase_ref=0.0,
                 r17_dense_co2_inhibition_f_phase_hill_n=3.0,
                 r17_dense_co2_inhibition_wet_ref=0.4,
                 r17_dense_co2_inhibition_wet_hill_n=2.0,
                 r13_no2_order=4.0,
                 r15_f_phase_exponent=1.0,
                 r4_o2_half_ppm=0.0,
                 r4_surface_gain=0.0,
                 r3a_bore_gain=0.0,
                 r15_surface_suppress_gain=0.0,
                 r15_sulfur_ref_ppm=0.0,
                 r15_sulfur_hill_n=2.0,
                 wall_o2_feed_o2_ref_ppm=0.0,
                 wall_o2_feed_o2_hill_n=4.0,
                 wall_no2_feed_o2_ref_ppm=0.0,
                 wall_no2_feed_o2_hill_n=4.0,
                 r3a_feed_o2_ref_ppm=0.0,
                 r3a_feed_o2_hill_n=4.0,
                 r3a_feed_o2_floor=0.0,
                 r3a_feed_o2_cap_ppm=0.0,
                 wall_no2_o2_presence_ref_ppm=0.0,
                 wall_no2_o2_presence_hill_n=1.0,
                 r3a_o2_presence_ref_ppm=0.0,
                 r3a_o2_presence_hill_n=1.0,
                 r3a_h2s_feed_inhibition_ref_ppm=0.0,
                 r3a_h2s_feed_inhibition_hill_n=4.0,
                 r3a_h2s_feed_inhibition_no2_ref_ppm=0.0,
                 r3a_h2s_feed_inhibition_no2_hill_n=4.0,
                 r3a_h2s_feed_inhibition_gain=0.0,
                 r2_o2_presence_ref_ppm=0.0,
                 r2_o2_presence_hill_n=1.0,
                 r2_no2_boost_ref_ppm=0.0,
                 r2_no2_boost_hill_n=2.0,
                 r2_no2_boost_gain=0.0,
                 r2_f_phase_exponent=1.0,
                 r2_no2_excess_ratio_ref=0.0,
                 r2_no2_excess_ratio_hill_n=4.0,
                 wall_no2_no_cap_ppm=0.0,
                 wall_no2_no_cap_hill_n=2.0,
                 wall_no2_langmuir_half_ppm=0.0,
                 r15_o2_inhib_ref_ppm=0.0,
                 r15_o2_inhib_hill_n=2.0,
                 r15_o2_activation_ref_ppm=0.0,
                 r15_o2_activation_hill_n=2.0,
                 r15_no2_cap_ppm=0.0,
                 r15_no2_cap_hill_n=2.0,
                 r15_dimer_dh_kj_mol=0.0,
                 r15_dimer_t_ref_k=278.15,
                 r15_n2o_cap_ppm=0.0,
                 r15_n2o_cap_hill_n=2.0,
                 r15_o2_presence_ref_ppm=0.0,
                 r15_o2_presence_hill_n=1.0,
                 r11_o2_ref_ppm=0.0,
                 r11_o2_hill_n=2.0,
                 r11_o2_gain=0.0,
                 o2_lag_tau_hours=0.0,
                 o2_feed_lag_tau_hours=0.0,
                 o2_feed_lag_rise_tau_hours=1.0,
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
        self._phi_overrides = {}
        self._water_solubility_ppm_base = self._calculate_water_solubility_ppm(T_kelvin, P_bar)

        self.condensation_exponent = float(condensation_exponent)
        self.rho_m_reference = float(rho_m_reference)
        self._f_phase = self._compute_f_phase()

        self.r5_no_activity = 1.0
        self.set_r5_no2_order()
        self.set_r7_no_order()

        self.r4_no_activity = 1.0
        self.r4_o2_half_ppm = float(r4_o2_half_ppm)

        self.r4_surface_gain = float(r4_surface_gain)
        self.r3a_bore_gain = float(r3a_bore_gain)
        self.r15_surface_suppress_gain = float(r15_surface_suppress_gain)
        self.r15_sulfur_ref_ppm = float(r15_sulfur_ref_ppm)
        self.r15_sulfur_hill_n = float(r15_sulfur_hill_n)
        self.wall_o2_feed_o2_ref_ppm = float(wall_o2_feed_o2_ref_ppm)
        self.wall_o2_feed_o2_hill_n = float(wall_o2_feed_o2_hill_n)
        self.wall_o2_h2s_relief = 0.0
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
        self.wall_no2_no_cap_gas_weighted = False
        self.wall_no2_langmuir_half_ppm = float(wall_no2_langmuir_half_ppm)
        self.r3a_o2_presence_ref_ppm = float(r3a_o2_presence_ref_ppm)
        self.r3a_o2_presence_hill_n = float(r3a_o2_presence_hill_n)
        self.r3a_h2s_feed_inhibition_ref_ppm = float(r3a_h2s_feed_inhibition_ref_ppm)
        self.r3a_h2s_feed_inhibition_hill_n = float(r3a_h2s_feed_inhibition_hill_n)
        self.r3a_h2s_feed_inhibition_no2_ref_ppm = float(r3a_h2s_feed_inhibition_no2_ref_ppm)
        self.r3a_h2s_feed_inhibition_no2_hill_n = float(r3a_h2s_feed_inhibition_no2_hill_n)
        self.r3a_h2s_feed_inhibition_gain = float(r3a_h2s_feed_inhibition_gain)
        self.r2_o2_presence_ref_ppm = float(r2_o2_presence_ref_ppm)
        self.r2_o2_presence_hill_n = float(r2_o2_presence_hill_n)

        self.r3a_no_escape_frac = 0.0

        self.wall_area_m2 = float(wall_area_m2)
        self.wall_k_intrinsic = float(wall_k_intrinsic)
        self.wall_o2_potency = float(wall_o2_potency)
        self.wall_o2_f_phase_exponent = float(wall_o2_f_phase_exponent)
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
        self.r1_autocat_hill_n = 1.0
        self.set_r17_orders()
        self.set_r17_dense_co2_inhibition(
            f_phase_ref=r17_dense_co2_inhibition_f_phase_ref,
            f_phase_hill_n=r17_dense_co2_inhibition_f_phase_hill_n,
            wet_ref=r17_dense_co2_inhibition_wet_ref,
            wet_hill_n=r17_dense_co2_inhibition_wet_hill_n)
        self.r1_feed_o2_ref_ppm = float(r1_feed_o2_ref_ppm)
        self.r1_feed_o2_hill_n = float(r1_feed_o2_hill_n)
        self.r1_feed_o2_cap_ppm = float(r1_feed_o2_cap_ppm)
        self.r3a_autocat_gain = float(r3a_autocat_gain)
        self.r3a_autocat_ref_ppm = float(r3a_autocat_ref_ppm)
        self.r3a_autocat_hill_n = float(r3a_autocat_hill_n)
        self.r3a_autocat_surface_suppress_gain = float(r3a_autocat_surface_suppress_gain)
        self.set_r3a_acid_film()
        self.acid_so2_sat_ref_ppm = float(acid_so2_sat_ref_ppm)
        self.acid_so2_sat_hill_n = float(acid_so2_sat_hill_n)
        self.r12_density_independent = bool(r12_density_independent)
        self.r12_f_phase_exponent = float(r12_f_phase_exponent)
        self.r12_no2_order = float(r12_no2_order)
        self.set_r12_rate_shape()
        self.r13_no2_order = float(r13_no2_order)
        self.set_r13_no2_rate_order()
        self.r15_f_phase_exponent = float(r15_f_phase_exponent)
        self.r15_o2_inhib_ref_ppm = float(r15_o2_inhib_ref_ppm)
        self.r15_o2_inhib_hill_n = float(r15_o2_inhib_hill_n)
        self.r15_o2_activation_ref_ppm = float(r15_o2_activation_ref_ppm)
        self.r15_o2_activation_hill_n = float(r15_o2_activation_hill_n)
        self.r15_no2_cap_ppm = float(r15_no2_cap_ppm)
        self.r15_no2_cap_hill_n = float(r15_no2_cap_hill_n)
        self.r15_dimer_dh_kj_mol = float(r15_dimer_dh_kj_mol)
        self.r15_dimer_t_ref_k = float(r15_dimer_t_ref_k)
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
        self.r2_f_phase_exponent = float(r2_f_phase_exponent)
        self.r2_no2_excess_ratio_ref = float(r2_no2_excess_ratio_ref)
        self.r2_no2_excess_ratio_hill_n = float(r2_no2_excess_ratio_hill_n)
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
        mole-fraction ppm of the bulk gas (kmol / kmol gas * 1e6). Deliberately
        includes bare NO2 gas itself (not just the acids it forms) since the wall
        O2-attack signal is meant to track the NO2 dosing schedule directly.
        """
        return (max(C_NO2, 0.0) + max(C_H2SO4, 0.0) + max(C_HNO3, 0.0)) \
            / max(self.molar_density, 1e-9) * 1e6

    def _acid_enhancement(self, C_NO2, C_H2SO4, C_HNO3):
        """NO2-dosing-driven gating factor for the wall O2 sink (dimensionless)."""
        total_acid_ppm = self._total_acid_ppm(C_NO2, C_H2SO4, C_HNO3)
        enhancement = self.wall_acid_background + self.wall_acid_gain * total_acid_ppm ** self.wall_acid_exponent
        if self.wall_acid_gain2 > 0.0:
            enhancement += self.wall_acid_gain2 * total_acid_ppm ** self.wall_acid_exponent2
        return enhancement

    def _water_solubility_ppm(self):
        """Water solubility (dew point) in the CO2-rich phase [ppm mol]."""
        return self._water_solubility_ppm_base * self.wall_h2o_enhancement_factor

    def _calculate_water_solubility_ppm(self, T_K, P_bar):
        """Estimate CO2-rich-phase water saturation [ppm-mol] using Antoine and SRK.

        The Antoine coefficients cover 1-100 C; subzero use is an extrapolation.
        """
        T_C = T_K - 273.15
        log10_p_mmhg = 8.07131 - 1730.63 / (233.426 + T_C)
        p_sat_bar = (10.0 ** log10_p_mmhg) * 1.33322e-3
        phi_h2o = max(self.phi_dict.get('H2O', 1.0), 1e-6)
        return (p_sat_bar / (phi_h2o * max(P_bar, 1e-9))) * 1e6

    def _strong_acid_ppm(self, C_H2SO4, C_HNO3):
        """Hygroscopic strong-acid loading (H2SO4 + HNO3) [ppm], driving deliquescence."""
        return (C_H2SO4 + C_HNO3) / max(self.molar_density, 1e-9) * 1e6

    def _co2_aqueous_solubility_mol_l(self):
        """Estimate dissolved CO2 [mol/L] using Henry's law and a van't Hoff correction.

        Uses K_H(298 K) = 0.034 mol/(L atm) and dH/R = 2400 K (Sander, 2015).
        """
        K_H_298 = 0.034
        DH_OVER_R = 2400.0
        K_H_T = K_H_298 * np.exp(DH_OVER_R * (1.0 / self.T - 1.0 / 298.15))
        P_co2_atm = self.P * 0.986923
        return K_H_T * P_co2_atm

    def _effective_g_h2o(self, h2o_ppm, C_H2SO4, C_HNO3):
        """Wet-film activation factor: wetted-fraction x excess-film severity."""
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
        """Return a fitted, bounded wall-rate enhancement favouring lower CO2 density.

        Optional water saturation modifies the density ratio used by the Hill factor.
        """
        if gain <= 0.0 or self.wall_gas_phase_rho_ref <= 0.0:
            return 1.0
        ratio = self.wall_gas_phase_rho_ref / max(self.molar_density, 1e-9)
        if sat is not None and sat_ref is not None and sat_ref > 0.0:
            ratio *= max(sat, 0.0) / sat_ref
        ratio_n = ratio ** self.wall_gas_phase_hill_n
        return 1.0 + gain * ratio_n / (1.0 + ratio_n)

    def _feed_o2_passivation(self, C_O2_feed, ref_ppm, hill_n, floor=0.0, cap_ppm=0.0):
        """Passivation gate keyed to the FED O2 level: ``1/(1+(feed_ppm/ref)**n)``."""
        if ref_ppm <= 0.0 or C_O2_feed is None:
            return 1.0
        feed_ppm = max(C_O2_feed, 0.0) / max(self.molar_density, 1e-9) * 1e6
        if cap_ppm > 0.0:
            feed_ppm = min(feed_ppm, cap_ppm)
        gate = 1.0 / (1.0 + (feed_ppm / ref_ppm) ** hill_n)
        return max(floor, gate)

    def set_dilute_redox(self, parameters=None):
        """Set fitted redox rates at 298.15 K with a smooth density blend.

        Optional NO saturation uses raw kmol/m3; the acid branch is a mobility
        ratio, not a separate reaction or a thermodynamic phase fraction.
        """
        if parameters is None:
            self.dilute_redox = None
            return
        positive = ('rho_ref_kmol_m3', 'density_hill_n', 'r2_no2_ref_kmol_m3', 'r2_no2_hill_n')
        nonnegative = ('r2_k_ref', 'r2_no2_gain', 'r4_k_ref')
        energies = ('r2_ea_kj_mol', 'r4_ea_kj_mol')
        required = set(positive + nonnegative + energies)
        optional = {'r2_h2s_half_kmol_m3': 0.0, 'r4_no_half_kmol_m3': 0.0,
                    'r4_no_reference_kmol_m3': 1e-6, 'r4_no_saturation_order': 1.0,
                    'r13_branch_fraction': 0.0, 'sulfur_acid_weight': 1.0}
        if required - set(parameters) or set(parameters) - (required | set(optional)):
            raise ValueError('Incomplete or unknown dilute-redox parameters')
        settings = {**optional, **{key: float(value) for key, value in parameters.items()}}
        if (not all(np.isfinite(value) for value in settings.values())
                or any(settings[key] <= 0.0 for key in positive)
            or any(settings[key] < 0.0 for key in nonnegative)
            or settings['r2_h2s_half_kmol_m3'] < 0.0
            or settings['r4_no_half_kmol_m3'] < 0.0 or settings['r4_no_reference_kmol_m3'] <= 0.0
            or not 0.0 < settings['r4_no_saturation_order'] <= 2.0
            or not 0.0 <= settings['r13_branch_fraction'] < 1.0
            or not 0.0 <= settings['sulfur_acid_weight'] <= 1.0):
            raise ValueError('Invalid dilute-redox coefficient or reference')
        self.dilute_redox = settings

    def _dilute_redox_weight(self):
        settings = getattr(self, 'dilute_redox', None)
        if settings is None:
            return 0.0
        reference = settings['rho_ref_kmol_m3']
        density = max(self.molar_density, 0.0)
        if density <= reference:
            return 1.0 / (1.0 + (density / reference) ** settings['density_hill_n'])
        inverse = (reference / density) ** settings['density_hill_n']
        return inverse / (1.0 + inverse)

    def _dilute_reference_rate(self, reaction):
        settings = self.dilute_redox
        return settings[reaction + '_k_ref'] * np.exp(settings[reaction + '_ea_kj_mol'] * 1000.0
                                                     / R_GAS * (1.0 / 298.15 - 1.0 / self.T))

    def _r2_effective_coefficient(self, dense_coefficient, no2, h2s=0.0):
        weight = self._dilute_redox_weight()
        if weight <= 0.0 or dense_coefficient <= 0.0:
            return dense_coefficient
        activation = self._reactant_activation_factor(no2, self.dilute_redox['r2_no2_ref_kmol_m3'],
                                                      self.dilute_redox['r2_no2_hill_n'])
        dilute_coefficient = self._dilute_reference_rate('r2') \
            * (1.0 + self.dilute_redox['r2_no2_gain'] * activation)
        half = self.dilute_redox['r2_h2s_half_kmol_m3']
        if half > 0.0:
            dilute_coefficient *= half / (half + max(h2s, 0.0))
        return (1.0 - weight) * dense_coefficient + weight * dilute_coefficient

    def _r4_effective_coefficient(self, dense_coefficient):
        weight = self._dilute_redox_weight()
        if weight <= 0.0 or dense_coefficient <= 0.0:
            return dense_coefficient
        return (1.0 - weight) * dense_coefficient + weight * self._dilute_reference_rate('r4')

    def _r4_dilute_no_factor(self, no):
        """Bounded trace-NO recycling mobility, common to both reaction directions."""
        settings = getattr(self, 'dilute_redox', None)
        if settings is None or settings['r4_no_half_kmol_m3'] <= 0.0:
            return 1.0
        half = settings['r4_no_half_kmol_m3']
        saturation = max(1.0, (half + settings['r4_no_reference_kmol_m3']) / (half + max(no, 0.0)))
        saturation **= settings['r4_no_saturation_order']
        return 1.0 + self._dilute_redox_weight() ** 2 * (saturation - 1.0)

    def _r13_dilute_factor(self, coefficient, r2_coefficient):
        """Add dilute acid-branch mobility without altering stoichiometry or equilibrium."""
        settings = getattr(self, 'dilute_redox', None)
        if settings is None or coefficient <= 0.0 or r2_coefficient <= 0.0:
            return 1.0
        fraction = settings['r13_branch_fraction']
        return 1.0 + self._dilute_redox_weight() ** 2 * fraction / (1.0 - fraction) \
            * r2_coefficient / coefficient

    def _bimolecular_encounter_factor(self, effective_rate_constant):
        """Finite encounter mobility for apparent bimolecular kinetics."""
        limit = BIMOLECULAR_ENCOUNTER_RATE_REF * self.T / 298.15
        return 1.0 / (1.0 + max(0.0, effective_rate_constant) / limit)

    def set_h2s_no2_temperature(self, parameters=None):
        """Set common cold-favoured availability for NO2-dependent H2S oxidation."""
        if parameters is None:
            self.h2s_no2_temperature = None
            return
        required = {'reference_K', 'suppression_kj_mol'}
        if required - set(parameters) or set(parameters) - (required | {'onset_width_K'}):
            raise ValueError('Require H2S/NO2 reference temperature and suppression energy')
        settings = {'onset_width_K': 0.0, **{key: float(value) for key, value in parameters.items()}}
        if (not all(np.isfinite(value) for value in settings.values())
                or settings['reference_K'] <= 0.0 or settings['suppression_kj_mol'] < 0.0
                or settings['onset_width_K'] < 0.0):
            raise ValueError('Require finite H2S/NO2 reference > 0, suppression >= 0, and onset width >= 0')
        self.h2s_no2_temperature = settings

    def _h2s_no2_temperature_factor(self):
        settings = getattr(self, 'h2s_no2_temperature', None)
        if settings is None or self.T <= settings['reference_K']:
            return 1.0
        inverse_difference = 1.0 / settings['reference_K'] - 1.0 / self.T
        if settings['onset_width_K'] > 0.0:
            inverse_width = settings['onset_width_K'] / settings['reference_K'] ** 2
            inverse_difference *= inverse_difference / (inverse_difference + inverse_width)
        exponent = -settings['suppression_kj_mol'] * 1000.0 / R_GAS * inverse_difference
        return float(np.exp(max(exponent, -700.0)))

    def _r3a_h2s_feed_inhibition(self, C_H2S_feed, C_NO2_feed=None):
        """Return R3a inhibition from the reducing inlet chemistry."""
        if self.r3a_h2s_feed_inhibition_ref_ppm <= 0.0 or C_H2S_feed is None:
            return 1.0
        h2s_ppm = max(C_H2S_feed, 0.0) / max(self.molar_density, 1e-9) * 1e6
        h2s_ratio = (h2s_ppm / self.r3a_h2s_feed_inhibition_ref_ppm) \
            ** self.r3a_h2s_feed_inhibition_hill_n
        if self.r3a_h2s_feed_inhibition_no2_ref_ppm <= 0.0:
            return 1.0 / (1.0 + h2s_ratio)
        if self.r3a_h2s_feed_inhibition_gain <= 0.0 or C_NO2_feed is None:
            return 1.0
        no2_ppm = max(C_NO2_feed, 0.0) / max(self.molar_density, 1e-9) * 1e6
        no2_ratio = (no2_ppm / self.r3a_h2s_feed_inhibition_no2_ref_ppm) \
            ** self.r3a_h2s_feed_inhibition_no2_hill_n
        h2s_available = h2s_ratio / (1.0 + h2s_ratio)
        no2_available = no2_ratio / (1.0 + no2_ratio)
        return 1.0 / (1.0 + self.r3a_h2s_feed_inhibition_gain * h2s_available * no2_available)

    def _r2_no2_excess_gate(self, C_H2S_feed, C_NO2_feed):
        """Return R2's inlet oxidant-excess availability gate."""
        if (self.r2_no2_excess_ratio_ref <= 0.0 or C_H2S_feed is None
                or C_NO2_feed is None):
            return 1.0
        h2s_feed = max(C_H2S_feed, 0.0)
        if h2s_feed <= 0.0:
            return 1.0
        ratio = max(C_NO2_feed, 0.0) / h2s_feed / self.r2_no2_excess_ratio_ref
        ratio_n = ratio ** self.r2_no2_excess_ratio_hill_n
        return ratio_n / (1.0 + ratio_n)

    def _o2_presence_gate(self, C_O2_feed, ref_ppm, hill_n):
        """O2-PRESENCE gate keyed to the FED O2 level: ``ratio/(1+ratio)``, ``ratio =
        (feed_ppm/ref)**n`` -- the mirror image of ``_feed_o2_passivation`` above.
        """
        if ref_ppm <= 0.0 or C_O2_feed is None:
            return 1.0
        feed_ppm = max(C_O2_feed, 0.0) / max(self.molar_density, 1e-9) * 1e6
        if feed_ppm <= 0.0:
            return 0.0
        ratio = (feed_ppm / ref_ppm) ** hill_n
        return ratio / (1.0 + ratio)

    def _wall_o2_rate(self, C_O2, h2o_ppm, C_NO2, C_H2SO4, C_HNO3, C_O2_feed=None,
                     C_H2S_feed=None):
        """O2-driven Fe2O3 wall-corrosion sink rate [kmol O2/(m^3 s)]: 4 Fe + 3 O2 -> 2
        Fe2O3.
        """
        if self.wall_area_m2 <= 0.0 or self.wall_k_intrinsic <= 0.0:
            return 0.0
        sat = self._water_saturation_fraction(h2o_ppm, C_H2SO4, C_HNO3)
        enhancement = self._acid_enhancement(C_NO2, C_H2SO4, C_HNO3)
        gas_phase = self._wall_gas_phase_enhancement(self.wall_o2_gas_phase_gain, sat=sat,
                                                      sat_ref=self.wall_o2_sat_ref)
        reference = self.wall_o2_feed_o2_ref_ppm
        if reference > 0.0 and C_H2S_feed is not None:
            reference += self.wall_o2_h2s_relief * max(C_H2S_feed, 0.0) / self.molar_density * 1e6
        passivation = self._feed_o2_passivation(C_O2_feed, reference,
                                                self.wall_o2_feed_o2_hill_n)
        o2_ppm = max(C_O2, 0.0) / max(self.molar_density, 1e-9) * 1e6
        V_m3 = max(self.volume_ml, 1e-9) * 1e-6
        A_sv = self.wall_area_m2 / V_m3
        return self.wall_k_intrinsic * sat * enhancement * gas_phase * passivation * \
            self._f_phase ** self.wall_o2_f_phase_exponent * \
            (o2_ppm ** self.wall_o2_potency) * A_sv * 1e-3

    def _water_saturation_fraction(self, h2o_ppm, C_H2SO4=0.0, C_HNO3=0.0):
        """Fraction of the (acid-lowered) water dew point reached by the bulk gas, in [0,
        1].
        """
        w_sat = self._water_solubility_ppm()
        strong_acid_ppm = self._strong_acid_ppm(C_H2SO4, C_HNO3)
        w_sat_eff = w_sat / (1.0 + max(strong_acid_ppm, 0.0) / self.wall_h2o_deliq_ref_ppm)
        return float(np.clip(max(h2o_ppm, 0.0) / max(w_sat_eff, 1e-9), 0.0, 1.0))

    def _wall_feco3_rate(self, h2o_ppm, C_H2SO4, C_HNO3):
        """Carbonic-acid (iron-carbonate) wall path [kmol Fe/(m^3 s)]: Fe + CO2(aq) + H2O
        -> FeCO3 + H2.
        """
        if self.wall_area_m2 <= 0.0 or self.wall_feco3_k_intrinsic <= 0.0:
            return 0.0
        g_h2o = self._effective_g_h2o(h2o_ppm, C_H2SO4, C_HNO3)
        co2_aq = self._co2_aqueous_solubility_mol_l()
        V_m3 = max(self.volume_ml, 1e-9) * 1e-6
        A_sv = self.wall_area_m2 / V_m3
        return self.wall_feco3_k_intrinsic * g_h2o * (co2_aq ** self.wall_feco3_potency) * A_sv * 1e-3

    def _wall_hno3_corrosion_rate(self, h2o_ppm, C_HNO3, C_H2SO4=0.0):
        """Nitric-acid wall-film corrosion rate [kmol Fe(NO3)2/(m^3 s)]: 8 HNO3 + 3 Fe ->
        3 Fe(NO3)2 + 2 NO + 4 H2O.
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
        """Sulfuric-acid wall path Fe rate [kmol Fe/(m^3 s)]: Fe + H2SO4 -> FeSO4 + H2."""
        if self.wall_area_m2 <= 0.0 or self.wall_h2so4_k_intrinsic <= 0.0:
            return 0.0
        sat = self._water_saturation_fraction(h2o_ppm, C_H2SO4, C_HNO3)
        h2so4_ppm = max(C_H2SO4, 0.0) / max(self.molar_density, 1e-9) * 1e6
        V_m3 = max(self.volume_ml, 1e-9) * 1e-6
        A_sv = self.wall_area_m2 / V_m3
        return self.wall_h2so4_k_intrinsic * sat * self._f_phase * \
            (h2so4_ppm ** self.wall_h2so4_potency) * A_sv * 1e-3

    def _sulfur_catalyst_gate(self, C_H2S_raw, C_H2SO4_raw):
        """Return bounded sulfur activation with optional dilute acid weighting."""
        if self.r15_sulfur_ref_ppm <= 0.0:
            return 1.0
        acid_weight = 1.0
        settings = getattr(self, 'dilute_redox', None)
        if settings is not None:
            half = settings['r2_h2s_half_kmol_m3']
            reductant = max(C_H2S_raw, 0.0)
            availability = reductant / (half + reductant) if half > 0.0 else 1.0
            acid_weight -= self._dilute_redox_weight() ** 2 \
                * (1.0 - settings['sulfur_acid_weight']) * (1.0 - availability)
        sulfur_ppm = (max(C_H2S_raw, 0.0) + acid_weight * max(C_H2SO4_raw, 0.0)) \
            / max(self.molar_density, 1e-9) * 1e6
        ratio_n = (sulfur_ppm / self.r15_sulfur_ref_ppm) ** self.r15_sulfur_hill_n
        return ratio_n / (1.0 + ratio_n)

    def _wall_no2_rate(self, C_NO2, h2o_ppm, cum_no2_exposure=0.0, C_O2=None,
                       C_H2S_raw=0.0, C_H2SO4_raw=0.0, C_O2_feed=None, C_O2_lagged=None,
                       C_NO=None):
        """NO2 wall-corrosion sink rate [kmol NO2/(m^3 s)]: 2 Fe + 3 NO2 -> Fe2O3 + 3 NO."""
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
        no_brake = 1.0
        if self.wall_no2_no_cap_ppm > 0.0 and C_NO is not None:
            no_ppm_wall = max(C_NO, 0.0) / max(self.molar_density, 1e-9) * 1e6
            if self.wall_no2_no_cap_gas_weighted:
                no_ppm_wall *= self._dilute_redox_weight()
            no_brake = 1.0 / (1.0 + (no_ppm_wall / self.wall_no2_no_cap_ppm) ** self.wall_no2_no_cap_hill_n)
        no2_ppm = max(C_NO2, 0.0) / max(self.molar_density, 1e-9) * 1e6
        if self.wall_no2_langmuir_half_ppm > 0.0:
            no2_term = self.wall_no2_langmuir_half_ppm * no2_ppm / (no2_ppm + self.wall_no2_langmuir_half_ppm)
        else:
            no2_term = no2_ppm ** self.wall_no2_potency
        V_m3 = max(self.volume_ml, 1e-9) * 1e-6
        A_sv = self.wall_area_m2 / V_m3
        return no_brake * self.wall_no2_k_intrinsic * wet_factor * arrhenius_factor * activation * o2_gate * \
            gas_phase * sulfur_gate * passivation * presence * no2_term * A_sv * 1e-3

    def _wall_so2_rate(self, C_SO2, h2o_ppm, cum_o2_exposure=0.0):
        """Surface-catalysed SO2 oxidation sink [kmol SO2/(m^3 s)]: SO2 + 0.5 O2 + H2O ->
        H2SO4 (same net stoichiometry as the homogeneous R1, but catalysed by an
        accumulating surface oxide layer -- iron oxide is a real, if weak, industrial
        SO2-oxidation catalyst, the same chemistry underlying the "contact process"
        for sulfuric acid manufacture).
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
        """Carbon-steel-catalysed Claus-type surface reaction [kmol H2S/(m^3 s)]: 8 H2S +
        4 O2 -> S8 + 8 H2O.
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
            _fractional_activity(o2_ppm, self.wall_s8_o2_potency,
                                 MIN_CONCENTRATION_FLOOR / self.molar_density * 1e6) * A_sv * 1e-3

    def get_wall_deposit_rates(self, C_O2, h2o_ppm, C_NO2, C_H2SO4, C_HNO3, C_H2S=0.0,
                                cum_no2_exposure=0.0, C_O2_feed=None, C_O2_lagged=None,
                                C_SO2=0.0, cum_o2_exposure=0.0, C_NO=None, C_O2_feed_lagged=None,
                                C_H2S_feed=None):
        """Return wall-pathway rates [kmol/(m3 s)] consistent with the species balances.

        Includes corrosion and catalytic pathways; a disabled pathway returns zero.
        """
        return {
            'r_wall_o2': self._wall_o2_rate(C_O2, h2o_ppm, C_NO2, C_H2SO4, C_HNO3,
                                            C_O2_feed=(C_O2_feed_lagged if self.o2_feed_lag_tau_hours > 0.0
                                                       else C_O2_feed), C_H2S_feed=C_H2S_feed),
            'r_feco3': self._wall_feco3_rate(h2o_ppm, C_H2SO4, C_HNO3),
            'r_hno3_corrosion': self._wall_hno3_corrosion_rate(h2o_ppm, C_HNO3, C_H2SO4),
            'r_h2so4': self._wall_h2so4_rate(h2o_ppm, C_H2SO4, C_HNO3),
            'r_wall_no2': self._wall_no2_rate(C_NO2, h2o_ppm, cum_no2_exposure, C_O2=C_O2,
                                              C_H2S_raw=C_H2S, C_H2SO4_raw=C_H2SO4,
                                              C_O2_feed=C_O2_feed, C_O2_lagged=C_O2_lagged,
                                              C_NO=C_NO),
            'r_wall_s8': self._wall_s8_rate(C_H2S * self.phi_dict['H2S'], C_O2),
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
                                 o2_f_phase_exponent=None,
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
        """Configure the wall-corrosion O2 sink after construction."""
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
        if o2_f_phase_exponent is not None:
            self.wall_o2_f_phase_exponent = float(o2_f_phase_exponent)
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
        """Directly set ``f_phase``, the wet-film/heterogeneous-reaction multiplier used
        ONLY by R2, R12 and R13 (the H2S-dependent NO2 reactions), bypassing the
        density-ratio formula in ``_compute_f_phase``.
        """
        self._f_phase = float(value)

    def set_srk_kij(self, species_name, kij):
        """Override the CO2-``species_name`` binary interaction parameter in the SRK
        flash.
        """
        self.srk_kij_co2[species_name] = float(kij)
        self.molar_density, self.phase, self.phi_dict = self._calculate_srk_fugacities(self.T, self.P)
        self._water_solubility_ppm_base = self._calculate_water_solubility_ppm(self.T, self.P)
        self._f_phase = self._compute_f_phase()

    def set_conditions(self, temp_C=None, pressure_bar=None):
        """Re-evaluate the reactor state at a new temperature and/or pressure."""
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
        """Force ``phi_dict[species_name]`` to ``value``, overriding whatever the SRK
        flash computed for it. Pure Python attribute set (no NeqSim flash), always
        safe to call.
        """
        self.phi_dict[species_name] = float(value)
        self._phi_overrides[species_name] = float(value)
        if species_name == 'H2O':
            self._water_solubility_ppm_base = self._calculate_water_solubility_ppm(self.T, self.P)

    def set_ideal_fugacity(self, species_name):
        """Force ``phi_dict[species_name]`` to 1.0 (ideal-mixture assumption), overriding
        whatever the SRK flash computed for it.
        """
        self.set_phi_override(species_name, 1.0)

    def set_r5_no_activity(self, value):
        """Set R5 reverse-term NO activity independently of its fugacity coefficient."""
        self.r5_no_activity = float(value)

    def set_r4_no_activity(self, value):
        """Set R4 forward-term NO activity independently of its fugacity coefficient."""
        self.r4_no_activity = float(value)

    def set_r4_o2_half_ppm(self, value):
        """Set R4's O2 half-saturation concentration in ppm; 0.0 restores first-order O2."""
        self.r4_o2_half_ppm = float(value)

    def set_r4_surface_gain(self, value):
        """Set R4's surface-to-volume enhancement gain; zero disables it."""
        self.r4_surface_gain = float(value)

    def set_r3a_bore_gain(self, value):
        """Set R3a's surface-to-volume enhancement gain; zero disables it."""
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
                                r3a_cap_ppm=None, wall_o2_h2s_relief=None):
        """Set the feed-O2 passivation gates (see ``_feed_o2_passivation``). 0.0 disables each."""
        if wall_o2_h2s_relief is not None:
            if not np.isfinite(wall_o2_h2s_relief) or wall_o2_h2s_relief < 0.0:
                raise ValueError('H2S passivation relief must be finite and nonnegative')
            self.wall_o2_h2s_relief = float(wall_o2_h2s_relief)
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

    def set_r3a_h2s_feed_inhibition(self, ref_ppm=None, hill_n=None, no2_ref_ppm=None,
                                     no2_hill_n=None, gain=None):
        """Configure R3a's optional inlet H2S/NO2 competition mechanism."""
        if ref_ppm is not None:
            self.r3a_h2s_feed_inhibition_ref_ppm = float(ref_ppm)
        if hill_n is not None:
            self.r3a_h2s_feed_inhibition_hill_n = float(hill_n)
        if no2_ref_ppm is not None:
            self.r3a_h2s_feed_inhibition_no2_ref_ppm = float(no2_ref_ppm)
        if no2_hill_n is not None:
            self.r3a_h2s_feed_inhibition_no2_hill_n = float(no2_hill_n)
        if gain is not None:
            self.r3a_h2s_feed_inhibition_gain = float(gain)

    def set_r2_no2_excess_gate(self, ratio_ref=None, hill_n=None):
        """Set R2's inlet NO2/H2S excess gate; a nonpositive ratio reference disables it."""
        if ratio_ref is not None:
            self.r2_no2_excess_ratio_ref = float(ratio_ref)
        if hill_n is not None:
            self.r2_no2_excess_ratio_hill_n = float(hill_n)

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

    def set_wall_no2_no_cap(self, ppm=None, hill_n=None, gas_weighted=None):
        """Set NO product inhibition of wall-mediated NO2 conversion.

        Optional density weighting is an empirical activity, not a phase fraction.
        """
        if gas_weighted is not None:
            if not isinstance(gas_weighted, (bool, np.bool_)):
                raise ValueError('Gas-weighted NO inhibition must be a boolean')
            self.wall_no2_no_cap_gas_weighted = bool(gas_weighted)
        if ppm is not None:
            self.wall_no2_no_cap_ppm = float(ppm)
        if hill_n is not None:
            self.wall_no2_no_cap_hill_n = float(hill_n)

    def set_wall_no2_langmuir(self, half_ppm):
        """Set NO2 half-saturation [ppm] for the wall pathway."""
        self.wall_no2_langmuir_half_ppm = float(half_ppm)

    def _surface_sv_excess(self):
        """Fractional excess of the vessel surface-to-volume ratio over the reference
        bore.
        """
        a_sv = 4.0 / max(self.diameter_cm, 1e-9) + 2.0 / max(self.length_cm, 1e-9)
        return max(0.0, a_sv / _R4_SURFACE_SV_REF_CM_INV - 1.0)

    def _r4_surface_factor(self):
        """Wall-film enhancement of R4 from the vessel surface-to-volume ratio (>=1.0)."""
        if self.r4_surface_gain <= 0.0:
            return 1.0
        return 1.0 + self.r4_surface_gain * self._surface_sv_excess()

    def _r3a_bore_factor(self):
        """Return R3a's surface-to-volume enhancement relative to the reference geometry."""
        if self.r3a_bore_gain <= 0.0:
            return 1.0
        return 1.0 + self.r3a_bore_gain * self._surface_sv_excess()

    def _r15_surface_suppression(self):
        """Surface quenching of R15's N2O channel in a narrow bore (<=1.0)."""
        if self.r15_surface_suppress_gain <= 0.0:
            return 1.0
        return 1.0 / (1.0 + self.r15_surface_suppress_gain * self._surface_sv_excess())

    def _r3a_autocat_surface_suppression(self):
        """Reduce R3a autocatalysis when surface-to-volume ratio exceeds the reference."""
        if self.r3a_autocat_surface_suppress_gain <= 0.0:
            return 1.0
        return 1.0 / (1.0 + self.r3a_autocat_surface_suppress_gain * self._surface_sv_excess())

    def set_r3a_no_escape_frac(self, value):
        """Set the phase-scaled reduction of NO activity in R3a's reverse term."""
        self.r3a_no_escape_frac = float(value)

    def set_r1_autocat(self, gain=None, ref_ppm=None, hill_n=None):
        """Set R1 acid activation: gain, reference [ppm-equivalent], and Hill exponent."""
        if hill_n is not None and (not np.isfinite(hill_n) or hill_n <= 0.0):
            raise ValueError('R1 Hill exponent must be finite and positive')
        if ref_ppm is not None and (not np.isfinite(ref_ppm) or ref_ppm <= 0.0):
            raise ValueError('R1 activation reference must be finite and positive')
        if gain is not None and not np.isfinite(gain):
            raise ValueError('R1 activation gain must be finite')
        if gain is not None:
            self.r1_autocat_gain = float(gain)
        if ref_ppm is not None:
            self.r1_autocat_ref_ppm = float(ref_ppm)
        if hill_n is not None:
            self.r1_autocat_hill_n = float(hill_n)

    def _r1_autocat_factor(self, cumulative_acid_ppm):
        """Bounded acid activation; exponent one retains the original Langmuir form."""
        if self.r1_autocat_gain <= 0.0 or cumulative_acid_ppm <= 0.0:
            return 1.0
        saturation = self._reactant_activation_factor(
            cumulative_acid_ppm, self.r1_autocat_ref_ppm, self.r1_autocat_hill_n)
        return 1.0 + self.r1_autocat_gain * saturation

    @staticmethod
    def _reactant_activation_factor(concentration, reference, hill_n):
        """Bounded loading activation; a zero reference disables it."""
        if reference <= 0.0:
            return 1.0
        if concentration <= 0.0:
            return 0.0
        log_ratio = hill_n * (np.log(reference) - np.log(concentration))
        return 1.0 / (1.0 + np.exp(np.clip(log_ratio, -700.0, 700.0)))

    def set_r5_no2_order(self, order=3.0, reference_kmol_m3=1e-5, floor_kmol_m3=1e-12):
        """Set a normalized apparent NO2 order without changing R5's equilibrium condition."""
        if not np.isfinite(order) or order <= 0.0:
            raise ValueError('R5 NO2 order must be finite and positive')
        if not np.isfinite(reference_kmol_m3) or not 0.0 < floor_kmol_m3 < reference_kmol_m3:
            raise ValueError('Require finite R5 reference > floor > 0')
        self.r5_no2_order = float(order)
        self.r5_reference_kmol_m3 = float(reference_kmol_m3)
        self.r5_floor_kmol_m3 = float(floor_kmol_m3)

    def _r5_rate_factor(self, no2):
        """Dimensionless mobility common to the forward and reverse rates."""
        return (max(no2, self.r5_floor_kmol_m3) / self.r5_reference_kmol_m3) ** (self.r5_no2_order - 3.0)

    def set_r7_no_order(self, order=1.0, reference_kmol_m3=1e-5, floor_kmol_m3=1e-12,
                        activation_reference_kmol_m3=0.0, activation_hill_n=1.0):
        """Set apparent NO order and optional bounded activation using NO activity."""
        if not np.isfinite(order) or order <= 0.0:
            raise ValueError('R7 NO order must be finite and positive')
        if not np.isfinite(reference_kmol_m3) or not 0.0 < floor_kmol_m3 < reference_kmol_m3:
            raise ValueError('Require finite R7 reference > floor > 0')
        if (not np.isfinite(activation_reference_kmol_m3) or activation_reference_kmol_m3 < 0.0
                or not np.isfinite(activation_hill_n) or activation_hill_n <= 0.0):
            raise ValueError('Require finite nonnegative R7 activation reference and positive Hill exponent')
        self.r7_no_order = float(order)
        self.r7_reference_kmol_m3 = float(reference_kmol_m3)
        self.r7_floor_kmol_m3 = float(floor_kmol_m3)
        self.r7_no_activation_reference_kmol_m3 = float(activation_reference_kmol_m3)
        self.r7_no_activation_hill_n = float(activation_hill_n)

    def _r7_rate_factor(self, no):
        """Dimensionless correction; the original NO factor still vanishes at zero."""
        order_factor = (max(no, self.r7_floor_kmol_m3) / self.r7_reference_kmol_m3) ** (self.r7_no_order - 1.0)
        return order_factor * self._reactant_activation_factor(
            no, self.r7_no_activation_reference_kmol_m3, self.r7_no_activation_hill_n)

    def set_r7_water_saturation(self, parameters=None):
        """Set a normalized Langmuir water dependence for the apparent R7 rate."""
        if parameters is None:
            self.r7_water_saturation = None
            return
        if set(parameters) != {'half_kmol_m3', 'reference_kmol_m3'}:
            raise ValueError('Require water half-saturation and reference concentrations')
        settings = {key: float(value) for key, value in parameters.items()}
        if (not all(np.isfinite(value) for value in settings.values())
                or settings['half_kmol_m3'] < 0.0 or settings['reference_kmol_m3'] <= 0.0):
            raise ValueError('Require finite water half-saturation >= 0 and reference > 0')
        self.r7_water_saturation = settings

    def _r7_water_activity(self, water):
        """Return finite, monotone water availability, with an exact dry limit."""
        concentration = max(float(water), 0.0)
        settings = getattr(self, 'r7_water_saturation', None)
        if settings is None or settings['half_kmol_m3'] <= 0.0:
            return concentration
        half = settings['half_kmol_m3']
        return concentration / (half + concentration) * (half + settings['reference_kmol_m3'])

    def set_r7_cold_availability(self, parameters=None):
        """Set a bounded apparent cold-site availability multiplying the R7 coefficient."""
        if parameters is None:
            self.r7_cold_availability = None
            return
        if set(parameters) != {'midpoint_K', 'width_K'}:
            raise ValueError('Require cold-availability midpoint_K and width_K')
        settings = {key: float(value) for key, value in parameters.items()}
        if any(not np.isfinite(value) or value <= 0.0 for value in settings.values()):
            raise ValueError('Cold-availability temperatures must be finite and positive')
        self.r7_cold_availability = settings

    def _r7_cold_factor(self):
        settings = getattr(self, 'r7_cold_availability', None)
        if settings is None:
            return 1.0
        exponent = (self.T - settings['midpoint_K']) / settings['width_K']
        return 1.0 / (1.0 + np.exp(np.clip(exponent, -700.0, 700.0)))

    def set_r17_orders(self, no2=2.0, h2o=1.0, wet=2.0, reference_kmol_m3=1e-5, floor_kmol_m3=1e-7,
                       activation_reference_kmol_m3=0.0, activation_hill_n=1.0):
        """Set real apparent orders with symmetric, dimensionless rate corrections."""
        if any(not np.isfinite(order) or order <= 0.0 for order in (no2, wet)):
            raise ValueError('R17 NO2 and wet orders must be finite and positive')
        if not np.isfinite(h2o):
            raise ValueError('R17 water order must be finite')
        if not np.isfinite(reference_kmol_m3) or not 0.0 < floor_kmol_m3 < reference_kmol_m3:
            raise ValueError('Require finite R17 reference > floor > 0')
        if (not np.isfinite(activation_reference_kmol_m3) or activation_reference_kmol_m3 < 0.0
                or not np.isfinite(activation_hill_n) or activation_hill_n <= 0.0):
            raise ValueError('Require finite nonnegative R17 activation reference and positive Hill exponent')
        self.r17_no2_order = float(no2)
        self.r17_h2o_order = float(h2o)
        self.r17_wet_order = float(wet)
        self.r17_reference_kmol_m3 = float(reference_kmol_m3)
        self.r17_floor_kmol_m3 = float(floor_kmol_m3)
        self.r17_no2_activation_reference_kmol_m3 = float(activation_reference_kmol_m3)
        self.r17_no2_activation_hill_n = float(activation_hill_n)

    def set_r17_dense_co2_inhibition(self, f_phase_ref=0.0, f_phase_hill_n=3.0,
                                      wet_ref=0.4, wet_hill_n=2.0):
        """Set the dense-CO2 inhibition of R17; a nonpositive phase reference disables it."""
        if not np.isfinite(f_phase_ref) or f_phase_ref < 0.0:
            raise ValueError('R17 dense-CO2 phase reference must be finite and nonnegative')
        if f_phase_ref > 0.0 and (
                not np.isfinite(f_phase_hill_n) or f_phase_hill_n <= 0.0
                or not np.isfinite(wet_ref) or wet_ref <= 0.0
                or not np.isfinite(wet_hill_n) or wet_hill_n < 0.0):
            raise ValueError('Require positive R17 dense-CO2 references and phase exponent')
        self.r17_dense_co2_inhibition_f_phase_ref = float(f_phase_ref)
        self.r17_dense_co2_inhibition_f_phase_hill_n = float(f_phase_hill_n)
        self.r17_dense_co2_inhibition_wet_ref = float(wet_ref)
        self.r17_dense_co2_inhibition_wet_hill_n = float(wet_hill_n)

    def _r17_dense_co2_inhibition(self, water_ppm):
        """Return R17 mobility in dense CO2; a developed water film relieves the quench."""
        if self.r17_dense_co2_inhibition_f_phase_ref <= 0.0:
            return 1.0
        wet = float(np.clip(max(water_ppm, 0.0) / max(self._water_solubility_ppm(), 1e-9), 0.0, 1.0))
        phase_ratio = self._f_phase / self.r17_dense_co2_inhibition_f_phase_ref
        wet_ratio = self.r17_dense_co2_inhibition_wet_ref / max(wet, 1e-7)
        return 1.0 / (1.0 + phase_ratio ** self.r17_dense_co2_inhibition_f_phase_hill_n
                      * wet_ratio ** self.r17_dense_co2_inhibition_wet_hill_n)

    def _r17_rate_factor(self, no2, h2o, water_ppm):
        """Common forward/reverse mobility, including the existing wet-film fraction."""
        wet = float(np.clip(max(water_ppm, 0.0) / max(self._water_solubility_ppm(), 1e-9), 0.0, 1.0))
        no2_factor = (max(no2, self.r17_floor_kmol_m3) / self.r17_reference_kmol_m3) ** (self.r17_no2_order - 2.0)
        h2o_factor = (max(h2o, self.r17_floor_kmol_m3) / self.r17_reference_kmol_m3) ** (self.r17_h2o_order - 1.0)
        wet_factor = max(wet, 1e-7) ** (self.r17_wet_order - 2.0)
        activation = self._reactant_activation_factor(
            no2, self.r17_no2_activation_reference_kmol_m3, self.r17_no2_activation_hill_n)
        return no2_factor * h2o_factor * wet_factor * wet ** 2 * activation \
            * self._r17_dense_co2_inhibition(water_ppm)

    def set_r3a_autocat(self, gain=None, ref_ppm=None, surface_suppress_gain=None, hill_n=None):
        """Set R3a acid activation and its optional surface-to-volume suppression."""
        if gain is not None:
            self.r3a_autocat_gain = float(gain)
        if ref_ppm is not None:
            self.r3a_autocat_ref_ppm = float(ref_ppm)
        if hill_n is not None:
            self.r3a_autocat_hill_n = float(hill_n)
        if surface_suppress_gain is not None:
            self.r3a_autocat_surface_suppress_gain = float(surface_suppress_gain)

    def _r3a_autocat_factor(self, cumulative_acid_ppm):
        """Acid activation shared by the solver and reaction reporting."""
        if self.r3a_autocat_gain <= 0.0:
            return 1.0
        if cumulative_acid_ppm <= 0.0:
            saturation = 0.0
        else:
            ratio = self.r3a_autocat_ref_ppm / cumulative_acid_ppm
            if ratio >= 1.0:
                scaled = (1.0 / ratio) ** self.r3a_autocat_hill_n
                saturation = scaled / (1.0 + scaled)
            else:
                saturation = 1.0 / (1.0 + ratio ** self.r3a_autocat_hill_n)
        activation = 1.0 + self.r3a_autocat_gain * saturation
        return 1.0 + (activation - 1.0) * self._r3a_autocat_surface_suppression()

    def set_r3a_acid_film(self, k_ref=0.0, ea_kj_mol=60.0, ref_ppm=100.0, hill_n=4.0,
                         r4_k_ref=0.0):
        """Configure apparent acid-film SO2 oxidation and optional NO recycling."""
        values = tuple(float(value) for value in (k_ref, ea_kj_mol, ref_ppm, hill_n, r4_k_ref))
        if not all(np.isfinite(value) for value in values):
            raise ValueError('Acid-film parameters must be finite')
        if min(values[0], values[4]) < 0.0 or values[2] <= 0.0 or values[3] <= 0.0:
            raise ValueError('Acid-film rate must be nonnegative; reference and Hill order must be positive')
        (self.r3a_acid_film_k_ref, self.r3a_acid_film_ea_kj_mol,
         self.r3a_acid_film_ref_ppm, self.r3a_acid_film_hill_n, self.r4_acid_film_k_ref) = values

    def _acid_film_kinetic_factor(self, cumulative_acid_ppm):
        """Shared acid activation, separate from inlet-based surface inhibition."""
        if cumulative_acid_ppm <= 0.0:
            return 0.0
        if cumulative_acid_ppm <= self.r3a_acid_film_ref_ppm:
            scaled = (cumulative_acid_ppm / self.r3a_acid_film_ref_ppm) ** self.r3a_acid_film_hill_n
            availability = scaled / (1.0 + scaled)
        else:
            availability = 1.0 / (1.0 + (self.r3a_acid_film_ref_ppm / cumulative_acid_ppm)
                                  ** self.r3a_acid_film_hill_n)
        temperature_factor = np.exp(self.r3a_acid_film_ea_kj_mol * 1000.0 / R_GAS
                                    * (1.0 / 298.15 - 1.0 / self.T))
        return temperature_factor * availability

    def _r3a_acid_film_rate_constant(self, cumulative_acid_ppm):
        """Apparent SO2 oxidation coefficient in m6/(kmol2 s)."""
        if self.r3a_acid_film_k_ref <= 0.0:
            return 0.0
        return self.r3a_acid_film_k_ref * self._acid_film_kinetic_factor(cumulative_acid_ppm)

    def set_r3a_dilute_acid(self, parameters=None):
        """Set local-acid activation and acid-history conditioning of dilute R3a.

        Local concentrations are kmol/m3; history is ppm-equivalent, not retained mass.
        """
        if parameters is None:
            self.r3a_dilute_acid = None
            return
        required = {'k_ref', 'history_ref_ppm', 'history_order', 'no2_half_kmol_m3', 'density_order'}
        optional = {'acid_ref_kmol_m3': 0.0, 'acid_order': 1.0,
                'nitric_history_gain': 0.0, 'nitric_history_ref_ppm': 100.0,
            'nitric_history_order': 2.0, 'background_k_ref': 0.0, 'history_fraction': 1.0,
                'h2s_half_kmol_m3': 0.0, 'acid_response_hours': 0.0}
        if required - set(parameters) or set(parameters) - (required | set(optional)):
            raise ValueError('Incomplete or unknown dilute-acid parameters')
        settings = {**optional, **{key: float(value) for key, value in parameters.items()}}
        if (not all(np.isfinite(value) for value in settings.values())
                or settings['k_ref'] < 0.0
                or any(settings[key] <= 0.0 for key in required - {'k_ref'})
                or settings['acid_ref_kmol_m3'] < 0.0 or settings['acid_order'] <= 0.0
                or settings['background_k_ref'] < 0.0
                or settings['h2s_half_kmol_m3'] < 0.0
                or settings['acid_response_hours'] < 0.0
                or not 0.0 <= settings['history_fraction'] <= 1.0
                or settings['nitric_history_gain'] < 0.0 or settings['nitric_history_ref_ppm'] <= 0.0
                or settings['nitric_history_order'] <= 0.0):
            raise ValueError('Invalid dilute-acid coefficient or reference')
        self.r3a_dilute_acid = settings

    def _r3a_dilute_acid_rate_constant(self, acid_history_ppm, no2, sulfuric_acid=0.0,
                                     nitric_history_ppm=0.0, h2s=0.0, acid_sites=None):
        """Reversible acid-assisted mobility with optional competing H2S uptake."""
        settings = getattr(self, 'r3a_dilute_acid', None)
        if settings is None or settings['k_ref'] <= 0.0:
            return 0.0
        activation = self._reactant_activation_factor(
            max(acid_history_ppm, 0.0), settings['history_ref_ppm'], settings['history_order'])
        activation = 1.0 - settings['history_fraction'] * (1.0 - activation)
        if settings['acid_response_hours'] > 0.0 and acid_sites is not None:
            activation *= float(np.clip(acid_sites, 0.0, 1.0))
        else:
            activation *= self._reactant_activation_factor(
                max(sulfuric_acid, 0.0), settings['acid_ref_kmol_m3'], settings['acid_order'])
        activation *= 1.0 + settings['nitric_history_gain'] * self._reactant_activation_factor(
            max(nitric_history_ppm, 0.0), settings['nitric_history_ref_ppm'], settings['nitric_history_order'])
        coefficient = (settings['background_k_ref'] + settings['k_ref'] * activation) \
            * self._dilute_redox_weight() ** settings['density_order']
        if settings['h2s_half_kmol_m3'] > 0.0:
            coefficient /= 1.0 + max(h2s, 0.0) / settings['h2s_half_kmol_m3']
        return coefficient / (1.0 + max(no2, 0.0) / settings['no2_half_kmol_m3'])

    def _acid_site_response(self, sulfuric_acid, activity):
        """Relax bounded acid-site activity; this state carries no material inventory."""
        settings = getattr(self, 'r3a_dilute_acid', None)
        if settings is None or settings['acid_response_hours'] <= 0.0:
            return 0.0
        target = self._reactant_activation_factor(
            max(sulfuric_acid, 0.0), settings['acid_ref_kmol_m3'], settings['acid_order'])
        return (target - float(np.clip(activity, 0.0, 1.0))) / (settings['acid_response_hours'] * 3600.0)

    def set_r3a_conditioned_acid(self, parameters=None):
        """Set empirical acid-conditioned R3a activity, not a new material source."""
        if parameters is None:
            self.r3a_conditioned_acid = None
            return
        positive = {'acid_ref_ppm', 'acid_order', 'history_ref_ppm',
                    'so2_ref_ppm', 'so2_order', 'no_ref_ppm'}
        required = positive | {'k_ref', 'ea_kj_mol'}
        optional = {'limit_initial_ppm_h': 0.0, 'limit_max_ppm_h': 0.0,
            'limit_acid_ref_ppm': 20.0, 'limit_acid_order': 1.0}
        if required - set(parameters) or set(parameters) - (required | set(optional)):
            raise ValueError('Incomplete or unknown conditioned-acid parameters')
        settings = {**optional, **{key: float(value) for key, value in parameters.items()}}
        if (not all(np.isfinite(value) for value in settings.values())
            or settings['k_ref'] < 0.0 or any(settings[key] <= 0.0 for key in positive)
            or not 0.0 <= settings['limit_initial_ppm_h'] <= settings['limit_max_ppm_h']
            or settings['limit_acid_ref_ppm'] <= 0.0 or settings['limit_acid_order'] <= 0.0):
            raise ValueError('Invalid conditioned-acid reference or coefficient')
        self.r3a_conditioned_acid = settings

    def _r3a_conditioned_acid_rate_constant(self, history_ppm, sulfuric_acid, so2, no):
        """Return reversible mobility from raw concentrations and the acid-history proxy."""
        settings = getattr(self, 'r3a_conditioned_acid', None)
        if settings is None or settings['k_ref'] <= 0.0:
            return 0.0
        scale = 1e6 / max(self.molar_density, 1e-9)
        acid_activity = self._reactant_activation_factor(
            max(sulfuric_acid, 0.0) * scale, settings['acid_ref_ppm'], settings['acid_order'])
        history_activity = self._reactant_activation_factor(max(history_ppm, 0.0), settings['history_ref_ppm'], 1.0)
        substrate = self._reactant_activation_factor(settings['so2_ref_ppm'], max(so2, 0.0) * scale,
                                                     settings['so2_order'])
        product = self._reactant_activation_factor(settings['no_ref_ppm'], max(no, 0.0) * scale, 2.0)
        temperature = np.exp(settings['ea_kj_mol'] * 1000.0 / R_GAS * (1.0 / 298.15 - 1.0 / self.T))
        return settings['k_ref'] * temperature * acid_activity * history_activity * substrate * product

    def _limit_r3a_conditioned_rate(self, rate, acid_history_ppm):
        """Limit net turnover with one symmetric mobility, preserving equilibrium."""
        settings = getattr(self, 'r3a_conditioned_acid', None)
        if settings is None or settings['limit_max_ppm_h'] <= 0.0:
            return rate
        activation = self._reactant_activation_factor(
            max(acid_history_ppm, 0.0), settings['limit_acid_ref_ppm'], settings['limit_acid_order'])
        ceiling_ppm_h = settings['limit_initial_ppm_h'] \
            + (settings['limit_max_ppm_h'] - settings['limit_initial_ppm_h']) * activation
        if ceiling_ppm_h <= 0.0:
            return 0.0
        ceiling = ceiling_ppm_h * self.molar_density * 1e-6 / 3600.0
        return rate / (1.0 + abs(rate) / ceiling)

    def set_r3a_environment(self, parameters=None):
        """Configure R3a wetting and acid availability without changing its equilibrium."""
        defaults = {'wetting_reference': 0.0, 'wetting_order': 2.0,
                'history_fraction': 1.0, 'current_acid_reference_ppm': 10.0,
            'oxygen_supply_fraction': 1.0, 'base_so2_inhibition_ref_ppm': 0.0,
                'base_so2_inhibition_order': 3.0, 'base_so2_inhibition_floor': 0.0,
                'base_so2_activation_ref_ppm': 0.0, 'base_so2_activation_order': 1.0,
                'oxygen_presence_ref_ppm': 0.0, 'oxygen_present_multiplier': 1.0}
        if parameters is None:
            self.r3a_environment = defaults
            return
        if set(parameters) - set(defaults):
            raise ValueError('Unknown R3a environment parameter')
        settings = {**defaults, **{key: float(value) for key, value in parameters.items()}}
        if (not all(np.isfinite(value) for value in settings.values())
                or not 0.0 <= settings['wetting_reference'] <= 1.0
                or not 0.0 <= settings['history_fraction'] <= 1.0
                or not 0.0 <= settings['oxygen_supply_fraction'] <= 1.0
                or settings['wetting_order'] <= 0.0
                or settings['base_so2_inhibition_ref_ppm'] < 0.0
                or settings['base_so2_inhibition_order'] <= 0.0
                or not 0.0 <= settings['base_so2_inhibition_floor'] <= 1.0
                or settings['base_so2_activation_ref_ppm'] < 0.0
                or settings['base_so2_activation_order'] <= 0.0
                or settings['oxygen_presence_ref_ppm'] < 0.0
                or not 0.0 <= settings['oxygen_present_multiplier'] <= 1.0
                or settings['current_acid_reference_ppm'] <= 0.0):
            raise ValueError('Invalid R3a wetting or acid-availability parameters')
        self.r3a_environment = settings

    def _r3a_base_so2_inhibition(self, so2_ppm):
        """Bounded apparent substrate inhibition of the base SO2/NO2 channel."""
        settings = getattr(self, 'r3a_environment', None)
        if settings is None or settings['base_so2_inhibition_ref_ppm'] <= 0.0:
            return 1.0
        reference = settings['base_so2_inhibition_ref_ppm']
        concentration = max(so2_ppm, 0.0)
        exponent = settings['base_so2_inhibition_order']
        if concentration <= reference:
            availability = 1.0 / (1.0 + (concentration / reference) ** exponent)
        else:
            scaled = (reference / concentration) ** exponent
            availability = scaled / (1.0 + scaled)
        if settings['base_so2_activation_ref_ppm'] > 0.0:
            availability *= self._reactant_activation_factor(
                concentration, settings['base_so2_activation_ref_ppm'], settings['base_so2_activation_order'])
        floor = settings['base_so2_inhibition_floor']
        return floor + (1.0 - floor) * availability

    def _r3a_oxygen_inhibition(self, oxygen, oxygen_supply):
        settings = getattr(self, 'r3a_environment', None)
        if settings is not None and settings['oxygen_presence_ref_ppm'] > 0.0:
            oxygen_ppm = max(float(oxygen), 0.0) / self.molar_density * 1e6
            fraction = min(oxygen_ppm / settings['oxygen_presence_ref_ppm'], 1.0)
            presence = fraction * fraction * (3.0 - 2.0 * fraction)
            return 1.0 - (1.0 - settings['oxygen_present_multiplier']) * presence
        supply_fraction = 1.0 if settings is None else settings['oxygen_supply_fraction']
        if oxygen_supply is None:
            signal = oxygen
        else:
            signal = supply_fraction * oxygen_supply + (1.0 - supply_fraction) * oxygen
        return self._feed_o2_passivation(signal, self.r3a_feed_o2_ref_ppm,
                                        self.r3a_feed_o2_hill_n, floor=self.r3a_feed_o2_floor,
                                        cap_ppm=self.r3a_feed_o2_cap_ppm)

    def _r3a_acid_signal(self, cumulative_acid_ppm, sulfuric_acid):
        settings = getattr(self, 'r3a_environment', None)
        if settings is None or settings['history_fraction'] == 1.0:
            return cumulative_acid_ppm
        current_ppm = max(sulfuric_acid, 0.0) / self.molar_density * 1e6
        current_equivalent = current_ppm * self.r3a_acid_film_ref_ppm \
            / settings['current_acid_reference_ppm']
        return settings['history_fraction'] * max(cumulative_acid_ppm, 0.0) \
            + (1.0 - settings['history_fraction']) * current_equivalent

    def _r3a_wetting_factor(self, water_ppm, sulfuric_acid, nitric_acid):
        settings = getattr(self, 'r3a_environment', None)
        if settings is None or settings['wetting_reference'] <= 0.0:
            return 1.0
        wet = self._water_saturation_fraction(water_ppm, sulfuric_acid, nitric_acid)
        return min(1.0, wet / settings['wetting_reference']) ** settings['wetting_order']

    def _r4_acid_film_rate_constant(self, cumulative_acid_ppm):
        """Apparent NO oxidation coefficient sharing R3a's acid availability."""
        if self.r4_acid_film_k_ref <= 0.0:
            return 0.0
        return self.r4_acid_film_k_ref * self._acid_film_kinetic_factor(cumulative_acid_ppm)

    def set_acid_so2_saturation(self, ref_ppm=None, hill_n=None):
        """Set shared R1/R3a SO2 inhibition; a nonpositive reference disables it."""
        if ref_ppm is not None:
            self.acid_so2_sat_ref_ppm = float(ref_ppm)
        if hill_n is not None:
            self.acid_so2_sat_hill_n = float(hill_n)

    def _acid_so2_saturation(self, C_SO2_raw):
        """Finite-capacity SO2 uptake factor shared by R1 and R3a, in (0, 1]."""
        if self.acid_so2_sat_ref_ppm <= 0.0:
            return 1.0
        so2_ppm = max(C_SO2_raw, 0.0) / max(self.molar_density, 1e-9) * 1e6
        if so2_ppm <= 0.0:
            return 1.0
        return 1.0 / (1.0 + (so2_ppm / self.acid_so2_sat_ref_ppm) ** self.acid_so2_sat_hill_n)

    def set_r12_density_independent(self, value):
        """Set whether R12 bypasses the density-based phase multiplier."""
        self.r12_density_independent = bool(value)

    def set_r12_f_phase_exponent(self, value):
        """Set R12's wet-film phase exponent; 1.0 preserves the shared phase factor."""
        self.r12_f_phase_exponent = float(value)

    def set_r12_no2_order(self, value):
        """Set the apparent NO2 order in R12's catalyst activity."""
        self.r12_no2_order = float(value)

    def set_r13_no2_order(self, value):
        """Set the NO2 and NO exponents in R13's forward and reverse terms."""
        self.r13_no2_order = float(value)

    def set_r12_rate_shape(self, h2s_order=1.0, no2_saturation_kmol_m3=0.0,
                           reference_kmol_m3=1e-5, floor_kmol_m3=1e-12):
        """Configure symmetric R12 loading corrections; zero saturation disables the cap."""
        values = (h2s_order, no2_saturation_kmol_m3, reference_kmol_m3, floor_kmol_m3)
        if (not all(np.isfinite(value) for value in values) or h2s_order <= 0.0
                or no2_saturation_kmol_m3 < 0.0 or not reference_kmol_m3 > floor_kmol_m3 > 0.0):
            raise ValueError('Require positive H2S order, nonnegative saturation, and reference > floor > 0')
        self.r12_h2s_order = float(h2s_order)
        self.r12_no2_saturation_kmol_m3 = float(no2_saturation_kmol_m3)
        self.r12_reference_kmol_m3 = float(reference_kmol_m3)
        self.r12_floor_kmol_m3 = float(floor_kmol_m3)

    def _r12_catalyst_factor(self, no2):
        no2 = max(float(no2), 0.0)
        saturation = self.r12_no2_saturation_kmol_m3
        if saturation <= 0.0:
            return no2 ** self.r12_no2_order
        if no2 > saturation:
            return saturation ** self.r12_no2_order / (1.0 + (saturation / no2) ** self.r12_no2_order)
        return no2 ** self.r12_no2_order / (1.0 + (no2 / saturation) ** self.r12_no2_order)

    def _r12_h2s_rate_factor(self, h2s):
        return (max(float(h2s), self.r12_floor_kmol_m3) / self.r12_reference_kmol_m3) ** (self.r12_h2s_order - 1.0)

    def set_r13_no2_rate_order(self, order=4.0, reference_kmol_m3=1e-5, floor_kmol_m3=1e-12):
        """Apply an NO2 loading correction relative to fourth order to both R13 directions."""
        if (not all(np.isfinite(value) for value in (order, reference_kmol_m3, floor_kmol_m3))
                or order <= 0.0 or not reference_kmol_m3 > floor_kmol_m3 > 0.0):
            raise ValueError('Require positive finite order and reference > floor > 0')
        self.r13_no2_rate_order = float(order)
        self.r13_reference_kmol_m3 = float(reference_kmol_m3)
        self.r13_floor_kmol_m3 = float(floor_kmol_m3)

    def _r13_rate_factor(self, no2):
        return (max(float(no2), self.r13_floor_kmol_m3) / self.r13_reference_kmol_m3) ** (self.r13_no2_rate_order - 4.0)

    def set_r15_f_phase_exponent(self, value):
        """Set R15's density-based phase exponent."""
        self.r15_f_phase_exponent = float(value)

    def set_r15_o2_inhibition(self, ref_ppm=None, hill_n=None):
        """Set R15 oxygen inhibition using a ppm reference and Hill exponent."""
        if ref_ppm is not None:
            self.r15_o2_inhib_ref_ppm = float(ref_ppm)
        if hill_n is not None:
            self.r15_o2_inhib_hill_n = float(hill_n)

    def set_r15_o2_activation(self, ref_ppm=None, hill_n=None):
        """Set R15 activation by local oxygen; a nonpositive reference disables it."""
        if ref_ppm is not None:
            self.r15_o2_activation_ref_ppm = float(ref_ppm)
        if hill_n is not None:
            self.r15_o2_activation_hill_n = float(hill_n)

    def set_r15_no2_cap(self, ppm=None, hill_n=None):
        """Set the soft cap on NO2 activity in R15's forward term."""
        if ppm is not None:
            self.r15_no2_cap_ppm = float(ppm)
        if hill_n is not None:
            self.r15_no2_cap_hill_n = float(hill_n)

    def set_r15_dimer_availability(self, dh_kj_mol=None, t_ref_k=None):
        """Set R15 dimer-availability energy [kJ/mol] and reference temperature [K]."""
        if dh_kj_mol is not None:
            self.r15_dimer_dh_kj_mol = float(dh_kj_mol)
        if t_ref_k is not None:
            self.r15_dimer_t_ref_k = float(t_ref_k)

    def _r15_dimer_availability(self):
        """Return bounded temperature-dependent dimer availability for R15."""
        if self.r15_dimer_dh_kj_mol <= 0.0 or self.T <= self.r15_dimer_t_ref_k:
            return 1.0
        dh_j = self.r15_dimer_dh_kj_mol * 1000.0
        return float(np.exp(-dh_j / R_GAS * (1.0 / self.r15_dimer_t_ref_k - 1.0 / self.T)))

    def set_r15_n2o_cap(self, ppm=None, hill_n=None):
        """Set R15 product inhibition using an N2O reference [ppm] and Hill exponent."""
        if ppm is not None:
            self.r15_n2o_cap_ppm = float(ppm)
        if hill_n is not None:
            self.r15_n2o_cap_hill_n = float(hill_n)

    def set_r15_o2_presence(self, ref_ppm=None, hill_n=None):
        """Set R15 activation by oxygen supply; a nonpositive reference disables it."""
        if ref_ppm is not None:
            self.r15_o2_presence_ref_ppm = float(ref_ppm)
        if hill_n is not None:
            self.r15_o2_presence_hill_n = float(hill_n)

    def set_r11_o2_gate(self, ref_ppm=None, hill_n=None, gain=None):
        """Set the bounded oxygen-dependent enhancement of R11."""
        if ref_ppm is not None:
            self.r11_o2_ref_ppm = float(ref_ppm)
        if hill_n is not None:
            self.r11_o2_hill_n = float(hill_n)
        if gain is not None:
            self.r11_o2_gain = float(gain)

    def set_r2_no2_boost(self, ref_ppm=None, hill_n=None, gain=None):
        """Set the bounded NO2-loading enhancement of R2."""
        if ref_ppm is not None:
            self.r2_no2_boost_ref_ppm = float(ref_ppm)
        if hill_n is not None:
            self.r2_no2_boost_hill_n = float(hill_n)
        if gain is not None:
            self.r2_no2_boost_gain = float(gain)

    def set_r2_f_phase_exponent(self, value):
        """Set R2's wet-film phase exponent; 1.0 preserves the shared phase factor."""
        self.r2_f_phase_exponent = float(value)

    def set_o2_lag_tau_hours(self, value):
        """Set the local-oxygen relaxation time [h]; zero disables the lag."""
        self.o2_lag_tau_hours = float(value)

    def set_o2_feed_lag_tau_hours(self, value, rise_tau_hours=None):
        """Set falling and optional rising oxygen-supply relaxation times [h].

        A zero falling time disables the supply lag.
        """
        self.o2_feed_lag_tau_hours = float(value)
        if rise_tau_hours is not None:
            self.o2_feed_lag_rise_tau_hours = float(rise_tau_hours)

    def apply_calibrated_profile(self, profile='carbon_steel_wet_co2'):
        """Apply the empirical carbon-steel/wet-CO2 reference parameter set."""
        if profile != 'carbon_steel_wet_co2':
            raise ValueError(f'Unknown profile: {profile}')

        for species in ('NO2', 'HNO3', 'H2O'):
            self.set_ideal_fugacity(species)
        self.set_phi_override('NO', 0.05)

        p = CARBON_STEEL_WET_CO2_KINETICS
        self.set_r5_no_activity(p['r5_no_activity'])
        self.set_r5_no2_order(p['r5_no2_order'], reference_kmol_m3=p['r5_reference_kmol_m3'],
                     floor_kmol_m3=p['r5_floor_kmol_m3'])
        self.set_r7_no_order(p['r7_no_order'], reference_kmol_m3=p['r7_reference_kmol_m3'],
                    floor_kmol_m3=p['r7_floor_kmol_m3'],
                    activation_reference_kmol_m3=p['r7_no_activation_reference_kmol_m3'],
                    activation_hill_n=p['r7_no_activation_hill_n'])
        self.set_r4_no_activity(p['r4_no_activity'])
        self.set_r3a_environment(p.get('r3a_environment'))
        self.set_dilute_redox(p.get('dilute_redox'))
        self.set_h2s_no2_temperature(p.get('h2s_no2_temperature'))
        self.set_r3a_dilute_acid(p.get('r3a_dilute_acid'))
        self.set_r3a_conditioned_acid(p.get('r3a_conditioned_acid'))
        self.set_r7_cold_availability(p.get('r7_cold_availability'))
        self.set_r7_water_saturation(p.get('r7_water_saturation'))
        self.set_r4_o2_half_ppm(p['r4_o2_half_ppm'])
        self.set_r4_surface_gain(p['r4_surface_gain'])
        self.set_r3a_bore_gain(p['r3a_bore_gain'])
        self.set_r15_surface_suppress_gain(p['r15_surface_suppress_gain'])
        self.set_r15_sulfur_gate(ref_ppm=p['r15_sulfur_ref_ppm'],
                                 hill_n=p['r15_sulfur_hill_n'])
        self.set_feed_o2_passivation(
            wall_o2_ref_ppm=p['wall_o2_feed_o2_ref_ppm'],
            wall_o2_hill_n=p['wall_o2_feed_o2_hill_n'],
            wall_o2_h2s_relief=p.get('wall_o2_h2s_relief', 0.0),
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
        self.set_r3a_h2s_feed_inhibition(
            ref_ppm=p['r3a_h2s_feed_inhibition_ref_ppm'],
            hill_n=p['r3a_h2s_feed_inhibition_hill_n'],
            no2_ref_ppm=p['r3a_h2s_feed_inhibition_no2_ref_ppm'],
            no2_hill_n=p['r3a_h2s_feed_inhibition_no2_hill_n'],
            gain=p['r3a_h2s_feed_inhibition_gain'])
        self.set_wall_no2_no_cap(ppm=p['wall_no2_no_cap_ppm'], hill_n=p['wall_no2_no_cap_hill_n'],
                     gas_weighted=p.get('wall_no2_no_cap_gas_weighted', False))
        self.set_wall_no2_langmuir(p['wall_no2_langmuir_half_ppm'])
        self.set_r3a_no_escape_frac(p['r3a_no_escape_frac'])
        self.set_r1_autocat(gain=p['r1_autocat_gain'], ref_ppm=p['r1_autocat_ref_ppm'],
                    hill_n=p['r1_autocat_hill_n'])
        self.set_r17_orders(no2=p['r17_no2_order'], h2o=p['r17_h2o_order'], wet=p['r17_wet_order'],
                    reference_kmol_m3=p['r17_reference_kmol_m3'], floor_kmol_m3=p['r17_floor_kmol_m3'],
                    activation_reference_kmol_m3=p['r17_no2_activation_reference_kmol_m3'],
                    activation_hill_n=p['r17_no2_activation_hill_n'])
        self.set_r17_dense_co2_inhibition(
            f_phase_ref=p['r17_dense_co2_inhibition_f_phase_ref'],
            f_phase_hill_n=p['r17_dense_co2_inhibition_f_phase_hill_n'],
            wet_ref=p['r17_dense_co2_inhibition_wet_ref'],
            wet_hill_n=p['r17_dense_co2_inhibition_wet_hill_n'])
        self.r1_feed_o2_ref_ppm = float(p['r1_feed_o2_ref_ppm'])
        self.r1_feed_o2_hill_n = float(p['r1_feed_o2_hill_n'])
        self.r1_feed_o2_cap_ppm = float(p['r1_feed_o2_cap_ppm'])
        self.set_r3a_autocat(gain=p['r3a_autocat_gain'], ref_ppm=p['r3a_autocat_ref_ppm'],
                             surface_suppress_gain=p['r3a_autocat_surface_suppress_gain'],
                             hill_n=p['r3a_autocat_hill_n'])
        self.set_r3a_acid_film(k_ref=p.get('r3a_acid_film_k_ref', 0.0),
                     ea_kj_mol=p.get('r3a_acid_film_ea_kj_mol', 60.0),
                     ref_ppm=p.get('r3a_acid_film_ref_ppm', 100.0),
                             hill_n=p.get('r3a_acid_film_hill_n', 4.0),
                             r4_k_ref=p.get('r4_acid_film_k_ref', 0.0))
        self.set_acid_so2_saturation(ref_ppm=p['acid_so2_sat_ref_ppm'],
                                     hill_n=p['acid_so2_sat_hill_n'])
        self.set_r12_density_independent(p['r12_density_independent'])
        self.set_r12_f_phase_exponent(p['r12_f_phase_exponent'])
        self.set_r12_no2_order(p['r12_no2_order'])
        self.set_r12_rate_shape(h2s_order=p['r12_h2s_order'],
                no2_saturation_kmol_m3=p['r12_no2_saturation_kmol_m3'],
                reference_kmol_m3=p['r12_reference_kmol_m3'], floor_kmol_m3=p['r12_floor_kmol_m3'])
        self.set_r13_no2_order(p['r13_no2_order'])
        self.set_r13_no2_rate_order(p['r13_no2_rate_order'],
                reference_kmol_m3=p['r13_reference_kmol_m3'], floor_kmol_m3=p['r13_floor_kmol_m3'])
        self.set_r15_f_phase_exponent(p['r15_f_phase_exponent'])
        self.set_r15_o2_inhibition(ref_ppm=p['r15_o2_inhib_ref_ppm'],
                                   hill_n=p['r15_o2_inhib_hill_n'])
        self.set_r15_o2_activation(ref_ppm=p['r15_o2_activation_ref_ppm'],
                                   hill_n=p['r15_o2_activation_hill_n'])
        self.set_r15_no2_cap(ppm=p['r15_no2_cap_ppm'], hill_n=p['r15_no2_cap_hill_n'])
        self.set_r15_dimer_availability(dh_kj_mol=p['r15_dimer_dh_kj_mol'],
                                        t_ref_k=p['r15_dimer_t_ref_k'])
        self.set_r15_n2o_cap(ppm=p['r15_n2o_cap_ppm'], hill_n=p['r15_n2o_cap_hill_n'])
        self.set_r15_o2_presence(ref_ppm=p['r15_o2_presence_ref_ppm'], hill_n=p['r15_o2_presence_hill_n'])
        self.set_r11_o2_gate(ref_ppm=p['r11_o2_ref_ppm'], hill_n=p['r11_o2_hill_n'], gain=p['r11_o2_gain'])
        self.set_r2_no2_boost(ref_ppm=p['r2_no2_boost_ref_ppm'], hill_n=p['r2_no2_boost_hill_n'],
                              gain=p['r2_no2_boost_gain'])
        self.set_r2_f_phase_exponent(p['r2_f_phase_exponent'])
        self.set_r2_no2_excess_gate(ratio_ref=p['r2_no2_excess_ratio_ref'],
                        hill_n=p['r2_no2_excess_ratio_hill_n'])
        self.set_o2_lag_tau_hours(p['o2_lag_tau_hours'])
        self.set_o2_feed_lag_tau_hours(p['o2_feed_lag_tau_hours'], rise_tau_hours=p['o2_feed_lag_rise_tau_hours'])
        for rxn_id in ('R1', 'R2', 'R3a', 'R4', 'R5', 'R7', 'R10', 'R11', 'R12', 'R13', 'R15', 'R16',
                       'R17', 'R18'):
            self.set_reaction_constants(rxn_id, A_forward=p[rxn_id]['A'],
                                         Ea_forward_kJ_mol=p[rxn_id]['Ea_kJ_mol'])
        self.set_phase_condensation(exponent=p['condensation_exponent'],
                                     rho_m_reference=p['rho_m_reference'])
        self.configure_wall_corrosion(
            k_intrinsic=p['wall_k_intrinsic'],
            o2_potency=p['wall_o2_potency'],
            o2_f_phase_exponent=p['wall_o2_f_phase_exponent'],
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

        dG7 = (6.0 * DG_NH3_STDGIBBS + 5.0 * DG_SO2_STDGIBBS) - \
            (5.0 * DG_H2S_STDGIBBS + 6.0 * DG_NO_STDGIBBS + 4.0 * DG_H2O_STDGIBBS)
        Keq7 = np.exp(min(-dG7 / (R_GAS * T), MAX_KEQ_EXPONENT))

        dG12 = DG_H2SO4_STDGIBBS - (DG_H2S_STDGIBBS + 2.0 * DG_O2_STDGIBBS)
        Keq12 = max(np.exp(min(-dG12 / (R_GAS * T), MAX_KEQ_EXPONENT)), 1e-15)

        dG13 = (DG_H2SO4_STDGIBBS + 4.0 * DG_NO_STDGIBBS) - (4.0 * DG_NO2_STDGIBBS + DG_H2S_STDGIBBS)
        Keq13 = max(np.exp(min(-dG13 / (R_GAS * T), MAX_KEQ_EXPONENT)), 1e-15)

        dG10 = (4.0 * DG_N2O_STDGIBBS + 6.0 * DG_H2O_STDGIBBS) - \
            (4.0 * DG_NH3_STDGIBBS + 4.0 * DG_NO_STDGIBBS + 3.0 * DG_O2_STDGIBBS)
        Keq10 = max(np.exp(min(-dG10 / (R_GAS * T), MAX_KEQ_EXPONENT)), 1e-15)

        dG11 = (DG_N2O_STDGIBBS + 0.125 * DG_S8_STDGIBBS + DG_H2O_STDGIBBS) - \
            (DG_H2S_STDGIBBS + 2.0 * DG_NO_STDGIBBS)
        Keq11 = max(np.exp(min(-dG11 / (R_GAS * T), MAX_KEQ_EXPONENT)), 1e-15)

        dG15 = (2.0 * DG_N2O_STDGIBBS + 3.0 * DG_O2_STDGIBBS) - (4.0 * DG_NO2_STDGIBBS)
        Keq15 = max(np.exp(min(-dG15 / (R_GAS * T), MAX_KEQ_EXPONENT)), 1e-15)

        dG16 = DG_N2O4_STDGIBBS - 2.0 * DG_NO2_STDGIBBS
        Keq16 = max(np.exp(min(-dG16 / (R_GAS * T), MAX_KEQ_EXPONENT)), 1e-15)

        dG17 = (DG_HNO3_STDGIBBS + DG_HNO2_STDGIBBS) - (2.0 * DG_NO2_STDGIBBS + DG_H2O_STDGIBBS)
        Keq17 = max(np.exp(min(-dG17 / (R_GAS * T), MAX_KEQ_EXPONENT)), 1e-15)
        dG18 = DG_HNO3_STDGIBBS - (DG_HNO2_STDGIBBS + 0.5 * DG_O2_STDGIBBS)
        Keq18 = max(np.exp(min(-dG18 / (R_GAS * T), MAX_KEQ_EXPONENT)), 1e-15)

        p = self.kinetic_params
        k1_f = p['R1']['A'] * np.exp(-p['R1']['Ea'] / (R_GAS * T))
        k2_f = p['R2']['A'] * np.exp(-p['R2']['Ea'] / (R_GAS * T))
        k3a_f = p['R3a']['A'] * np.exp(-p['R3a']['Ea'] / (R_GAS * T))
        k4_f = p['R4']['A'] * np.exp(-p['R4']['Ea'] / (R_GAS * T)) if p['R4']['Ea'] > 0 else p['R4']['A'] * np.exp(530.0 / T)
        k4_f = self._r4_effective_coefficient(k4_f)
        k5_f = p['R5']['A'] * np.exp(-p['R5']['Ea'] / (R_GAS * T))
        k7_f = p['R7']['A'] * np.exp(-p['R7']['Ea'] / (R_GAS * T)) * self._r7_cold_factor()

        k10_f = p['R10']['A'] * np.exp(-p['R10']['Ea'] / (R_GAS * T))
        k11_f = p['R11']['A'] * np.exp(-p['R11']['Ea'] / (R_GAS * T))
        k12_f = p['R12']['A'] * np.exp(-p['R12']['Ea'] / (R_GAS * T))
        k13_f = p['R13']['A'] * np.exp(-p['R13']['Ea'] / (R_GAS * T))
        k15_f = p['R15']['A'] * np.exp(-p['R15']['Ea'] / (R_GAS * T))
        k16_f = p['R16']['A'] * np.exp(-p['R16']['Ea'] / (R_GAS * T))
        k17_f = p['R17']['A'] * np.exp(-p['R17']['Ea'] / (R_GAS * T))
        k18_f = p['R18']['A'] * np.exp(-p['R18']['Ea'] / (R_GAS * T))

        k1_r = k1_f / Keq1 if Keq1 > 1e-15 else 0.0
        k2_r = k2_f / Keq2 if Keq2 > 1e-15 else 0.0
        k3a_r = k3a_f / Keq3 if Keq3 > 1e-15 else 0.0
        k4_r = k4_f / Keq4 if Keq4 > 1e-15 else 0.0
        k5_r = k5_f / Keq5
        k7_r = k7_f / Keq7
        k10_r = k10_f / Keq10 if Keq10 > 1e-15 else 0.0
        k11_r = k11_f / Keq11 if Keq11 > 1e-15 else 0.0
        k12_r = k12_f / Keq12 if Keq12 > 1e-15 else 0.0
        k13_r = k13_f / Keq13 if Keq13 > 1e-15 else 0.0
        k15_r = k15_f / Keq15 if Keq15 > 1e-15 else 0.0
        k16_r = k16_f / Keq16 if Keq16 > 1e-15 else 0.0
        k17_r = k17_f / Keq17 if Keq17 > 1e-15 else 0.0
        k18_r = k18_f / Keq18 if Keq18 > 1e-15 else 0.0

        safe_moisture_ppm = max(float(moisture_ppm), 0.0)
        moisture_factor = 0.25 + 0.75 * (1.0 - np.exp(-min(safe_moisture_ppm / MOISTURE_REF_PPM, 50.0)))
        k1_f *= moisture_factor
        k1_r *= moisture_factor
        k3a_f *= moisture_factor
        k3a_r *= moisture_factor

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
            'k16_f': k16_f, 'k16_r': k16_r, 'Keq16': Keq16,
            'k17_f': k17_f, 'k17_r': k17_r, 'Keq17': Keq17,
            'k18_f': k18_f, 'k18_r': k18_r, 'Keq18': Keq18,
            'material': self.material,
            'moisture_factor': moisture_factor,
            'f_phase': self._f_phase,
        }

    def rhs(self, t, C, rates_dict, C_in=None, space_time_sec=None, inflow_only=False):
        n_species = len(self.SPECIES)
        C_raw = np.clip(C[:n_species], 0.0, 1e5 * self.molar_density)
        n_extra = len(self.EXTRA_STATE_KEYS)
        C_wall_solid = np.maximum(0.0, C[n_species:]) if len(C) > n_species else np.zeros(n_extra)
        C_cum_h2so4 = C_wall_solid[2] if len(C_wall_solid) > 2 else 0.0
        C_cum_hno3 = C_wall_solid[3] if len(C_wall_solid) > 3 else 0.0
        C_cum_no2_exposure = C_wall_solid[4] if len(C_wall_solid) > 4 else 0.0
        C_lagged_o2 = C_wall_solid[5] if len(C_wall_solid) > 5 else 0.0
        C_cum_o2_exposure = C_wall_solid[6] if len(C_wall_solid) > 6 else 0.0
        C_lagged_o2_feed = C_wall_solid[7] if len(C_wall_solid) > 7 else 0.0
        acid_sites = C_wall_solid[9] if len(C_wall_solid) > 9 else 0.0

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
        C_N2O4  = max(0.0, C_raw[12] * phi['N2O4']) if len(C_raw) > 12 else 0.0
        C_HNO2  = max(0.0, C_raw[13] * phi['HNO2']) if len(C_raw) > 13 else 0.0

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

        cum_h2so4_ppm = C_cum_h2so4 / max(self.molar_density, 1e-9) * 1e6

        C_O2_feed = C_in[4] if C_in is not None else None
        C_H2S_feed = C_in[0] if C_in is not None else None
        C_NO2_feed = C_in[2] if C_in is not None else None
        r1 = k1_f * C_SO2 * _fractional_activity(C_O2, 0.5) * C_H2O - k1_r * C_H2SO4
        acid_so2_sat = self._acid_so2_saturation(C_raw[1])
        r1 *= acid_so2_sat * self._feed_o2_passivation(C_O2_feed, self.r1_feed_o2_ref_ppm,
                                                       self.r1_feed_o2_hill_n,
                                                       cap_ppm=self.r1_feed_o2_cap_ppm)
        if self.r1_autocat_gain > 0.0:
            r1_autocat = self._r1_autocat_factor(cum_h2so4_ppm)
            r1 *= r1_autocat
        r2_no2_boost = 1.0
        if self.r2_no2_boost_ref_ppm > 0.0:
            no2_ppm_r2 = max(C_NO2, 0.0) / max(self.molar_density, 1e-9) * 1e6
            ratio_r2 = (no2_ppm_r2 / self.r2_no2_boost_ref_ppm) ** self.r2_no2_boost_hill_n
            r2_no2_boost = 1.0 + self.r2_no2_boost_gain * ratio_r2 / (1.0 + ratio_r2)
        r2_scale = r2_no2_boost * f_phase ** self.r2_f_phase_exponent \
            * self._o2_presence_gate(C_O2_feed, self.r2_o2_presence_ref_ppm,
                                     self.r2_o2_presence_hill_n) \
            * self._r2_no2_excess_gate(C_H2S_feed, C_NO2_feed)
        r13_scale = f_phase * self._r13_rate_factor(C_NO2)
        r13_bimolecular_coefficient = k13_f * r13_scale \
            * max(C_NO2, self.r13_floor_kmol_m3) ** (self.r13_no2_order - 1.0)
        r2_coefficient = self._r2_effective_coefficient(k2_f * r2_scale, C_NO2, C_H2S)
        r13_dilute = self._r13_dilute_factor(r13_bimolecular_coefficient, r2_coefficient)
        r13_scale *= r13_dilute
        r13_bimolecular_coefficient *= r13_dilute
        h2s_no2_encounter = self._bimolecular_encounter_factor(r2_coefficient + r13_bimolecular_coefficient)
        h2s_no2_encounter *= self._h2s_no2_temperature_factor()
        r2 = h2s_no2_encounter * r2_coefficient \
            * (C_H2S * C_NO2 - C_SO2 * C_H2O * (C_NO**3) / rates_dict['Keq2'])
        C_NO_r3a = C_NO * (1.0 - self.r3a_no_escape_frac * f_phase)
        r3a = self._r3a_bore_factor() * (k3a_f * C_SO2 * C_NO2 * C_H2O - k3a_r * C_NO_r3a * C_H2SO4) \
            * acid_so2_sat \
            * self._r3a_base_so2_inhibition(C_raw[1] / self.molar_density * 1e6) \
            * self._r3a_oxygen_inhibition(C_raw[4], C_O2_feed) \
            * self._o2_presence_gate(C_O2_feed, self.r3a_o2_presence_ref_ppm,
                                     self.r3a_o2_presence_hill_n) \
            * self._r3a_h2s_feed_inhibition(C_H2S_feed, C_NO2_feed)
        r3a_acid_signal = self._r3a_acid_signal(cum_h2so4_ppm, C_raw[6])
        if self.r3a_autocat_gain > 0.0:
            r3a_autocat = self._r3a_autocat_factor(r3a_acid_signal)
            r3a *= r3a_autocat
        r3a_acid_coefficient = self._r3a_acid_film_rate_constant(r3a_acid_signal) \
            + self._r3a_dilute_acid_rate_constant(
                r3a_acid_signal, C_NO2, C_raw[6], C_cum_hno3 / self.molar_density * 1e6,
                C_H2S, acid_sites)
        r3a_driving_force = C_SO2 * C_NO2 * C_H2O - C_NO * C_H2SO4 / rates_dict['Keq3']
        r3a_conditioned = self._r3a_conditioned_acid_rate_constant(
            r3a_acid_signal, C_raw[6], C_raw[1], C_raw[3]) * r3a_driving_force
        r3a += r3a_acid_coefficient * r3a_driving_force \
            + self._limit_r3a_conditioned_rate(r3a_conditioned, cum_h2so4_ppm)
        r3a *= self._r3a_wetting_factor(C_raw[5] / self.molar_density * 1e6, C_raw[6], C_raw[7])
        C_NO_r4 = max(0.0, C_raw[3] * self.r4_no_activity * self._r4_surface_factor())
        o2_ppm_r4 = max(C_raw[4], 0.0) / max(self.molar_density, 1e-9) * 1e6
        if self.r4_o2_half_ppm > 0.0:
            C_O2_r4 = C_O2 * (9.6 + self.r4_o2_half_ppm) / (o2_ppm_r4 + self.r4_o2_half_ppm)
        else:
            C_O2_r4 = C_O2
        r4 = k4_f * (C_NO_r4**2) * C_O2_r4 - k4_r * (C_NO2**2)
        r4 += self._r4_acid_film_rate_constant(cum_h2so4_ppm) \
            * (C_NO ** 2 * C_O2 - C_NO2 ** 2 / rates_dict['Keq4'])
        r4 *= self._r4_dilute_no_factor(C_raw[3])
        C_NO_r5 = max(0.0, C_raw[3] * self.r5_no_activity)
        r5 = self._r5_rate_factor(C_NO2) * (k5_f * (C_NO2**3) * C_H2O - k5_r * (C_HNO3**2) * C_NO_r5)
        r7 = self._r7_rate_factor(C_NO) * k7_f * C_H2S * C_NO * self._r7_water_activity(C_H2O)
        r10 = k10_f * C_NH3 * C_NO * C_O2 - k10_r * (C_N2O**4) * (C_H2O**6)
        r11_o2_boost = 1.0
        if self.r11_o2_ref_ppm > 0.0:
            o2_ppm_r11 = max(C_O2, 0.0) / max(self.molar_density, 1e-9) * 1e6
            ratio_n = (o2_ppm_r11 / self.r11_o2_ref_ppm) ** self.r11_o2_hill_n
            r11_o2_boost = 1.0 + self.r11_o2_gain * ratio_n / (1.0 + ratio_n)
        r11 = r11_o2_boost * k11_f * C_H2S * C_NO - k11_r * C_N2O * (C_S8**0.125) * C_H2O
        C_NO2_r12 = self._r12_catalyst_factor(C_NO2)
        r12 = (k12_f * C_H2S * C_O2 * C_NO2_r12 - k12_r * C_H2SO4 * C_NO2_r12) \
            * (1.0 if self.r12_density_independent else f_phase ** self.r12_f_phase_exponent) \
            * self._r12_h2s_rate_factor(C_H2S)
        r12 *= self._h2s_no2_temperature_factor()
        C_NO2_r13 = C_NO2 ** self.r13_no2_order
        C_NO_r13 = C_NO ** self.r13_no2_order
        r13 = h2s_no2_encounter * r13_scale \
            * (k13_f * C_H2S * C_NO2_r13 - k13_r * C_H2SO4 * C_NO_r13)
        k15_f, k15_r = rates_dict.get('k15_f', 0.0), rates_dict.get('k15_r', 0.0)
        k16_f, k16_r = rates_dict.get('k16_f', 0.0), rates_dict.get('k16_r', 0.0)
        k17_f, k17_r = rates_dict.get('k17_f', 0.0), rates_dict.get('k17_r', 0.0)
        k18_f, k18_r = rates_dict.get('k18_f', 0.0), rates_dict.get('k18_r', 0.0)
        f_phase_r15 = f_phase ** self.r15_f_phase_exponent
        r15_o2_inhib = 1.0
        if self.r15_o2_inhib_ref_ppm > 0.0:
            o2_source = C_lagged_o2 if self.o2_lag_tau_hours > 0.0 else C_raw[4]
            o2_ppm_r15 = max(o2_source, 0.0) / max(self.molar_density, 1e-9) * 1e6
            r15_o2_inhib = 1.0 / (1.0 + (o2_ppm_r15 / self.r15_o2_inhib_ref_ppm)
                                  ** self.r15_o2_inhib_hill_n)
        r15_o2_activation = 1.0
        if self.r15_o2_activation_ref_ppm > 0.0:
            o2_ppm_act = max(C_raw[4], 0.0) / max(self.molar_density, 1e-9) * 1e6
            ratio_act = (o2_ppm_act / self.r15_o2_activation_ref_ppm) ** self.r15_o2_activation_hill_n \
                if o2_ppm_act > 0.0 else 0.0
            r15_o2_activation = ratio_act / (1.0 + ratio_act)
        C_NO2_r15 = C_NO2
        if self.r15_no2_cap_ppm > 0.0:
            no2_ppm_r15 = max(C_raw[2], 0.0) / max(self.molar_density, 1e-9) * 1e6
            if no2_ppm_r15 > 0.0:
                ratio_n = (no2_ppm_r15 / self.r15_no2_cap_ppm) ** self.r15_no2_cap_hill_n
                no2_ppm_r15_capped = no2_ppm_r15 / (1.0 + ratio_n) ** (1.0 / self.r15_no2_cap_hill_n)
            else:
                no2_ppm_r15_capped = 0.0
            C_NO2_r15 = no2_ppm_r15_capped * 1e-6 * self.molar_density * phi['NO2']
        r15_n2o_brake = 1.0
        if self.r15_n2o_cap_ppm > 0.0:
            n2o_ppm_r15 = max(C_raw[10], 0.0) / max(self.molar_density, 1e-9) * 1e6
            r15_n2o_brake = 1.0 / (1.0 + (n2o_ppm_r15 / self.r15_n2o_cap_ppm)
                                   ** self.r15_n2o_cap_hill_n)
        r15_o2_presence = self._o2_presence_gate(C_O2_feed, self.r15_o2_presence_ref_ppm,
                                                 self.r15_o2_presence_hill_n)
        r15 = (r15_o2_inhib * r15_o2_activation * r15_n2o_brake * self._r15_surface_suppression()
               * self._r15_dimer_availability()
               * self._sulfur_catalyst_gate(C_raw[0], C_raw[6]) * k15_f * (C_NO2_r15**4)
               - k15_r * (C_N2O**2) * (C_O2**3)) * f_phase_r15 * r15_o2_presence
        r16 = k16_f * (C_NO2**2) - k16_r * C_N2O4
        h2o_ppm_r17 = C_raw[5] / max(self.molar_density, 1e-9) * 1e6
        r17_wet = self._r17_rate_factor(C_NO2, C_H2O, h2o_ppm_r17)
        r17 = (k17_f * (C_NO2**2) * C_H2O - k17_r * C_HNO3 * C_HNO2) * r17_wet
        r18 = k18_f * C_HNO2 * _fractional_activity(C_O2, 0.5) - k18_r * C_HNO3

        R_H2S   = - r2 - 5.0 * r7 - r11 - r12 - r13
        R_SO2   = - r1 + r2 - r3a + 5.0 * r7
        R_NO2   = - 3.0 * r2 - r3a + 2.0 * r4 - 3.0 * r5 - 4.0 * r13 - 4.0 * r15 - 2.0 * r16 - 2.0 * r17
        R_N2O4  = + r16
        R_HNO2  = + r17 - r18
        R_NO    = + 3.0 * r2 + r3a - 2.0 * r4 + r5 - 6.0 * r7 - 4.0 * r10 - 2.0 * r11 + 4.0 * r13
        R_O2    = - 0.5 * r1 - r4 - 3.0 * r10 - 2.0 * r12 + 3.0 * r15 - 0.5 * r18
        R_H2O   = - r1 + r2 - r3a - r5 - 4.0 * r7 + 6.0 * r10 + r11 - r17
        R_H2SO4 = + r1 + r3a + r12 + r13
        R_HNO3  = + 2.0 * r5 + r17 + r18
        R_S8    = + 0.125 * r11
        R_NH3   = + 6.0 * r7 - 4.0 * r10
        R_N2O   = + 4.0 * r10 + r11 + 2.0 * r15
        R_H2    = 0.0

        cum_h2so4_rate = max(0.0, r1) + max(0.0, r3a) + max(0.0, r12) + max(0.0, r13)
        cum_hno3_rate = 2.0 * max(0.0, r5) + max(0.0, r17) + max(0.0, r18)
        cum_nh3_rate = 6.0 * max(0.0, r7)

        lagged_o2_rate = 0.0
        if self.o2_lag_tau_hours > 0.0:
            lagged_o2_rate = (C_raw[4] - C_lagged_o2) / (self.o2_lag_tau_hours * 3600.0)

        lagged_o2_feed_rate = 0.0
        if self.o2_feed_lag_tau_hours > 0.0 and C_O2_feed is not None:
            feed_now = max(C_O2_feed, 0.0)
            rising = feed_now > C_lagged_o2_feed
            tau_hours = self.o2_feed_lag_rise_tau_hours if rising else self.o2_feed_lag_tau_hours
            lagged_o2_feed_rate = (feed_now - C_lagged_o2_feed) / (max(tau_hours, 1e-9) * 3600.0)

        r_hno3corr = 0.0
        r_h2so4 = 0.0
        cum_no2_rate = 0.0
        cum_o2_rate = 0.0
        r_wall_so2 = 0.0
        if self.wall_area_m2 > 0.0:
            h2o_ppm_here = C_raw[5] / max(self.molar_density, 1e-9) * 1e6
            C_NO2_wall = C_raw[2]
            C_H2SO4_wall = C_raw[6]
            C_HNO3_wall = C_raw[7]
            C_SO2_wall = C_raw[1]
            cum_o2_rate = C_raw[4]
            if self.wall_so2_k_intrinsic > 0.0:
                r_wall_so2 = self._wall_so2_rate(C_SO2_wall, h2o_ppm_here, C_cum_o2_exposure)
                R_SO2 -= r_wall_so2
                R_H2SO4 += r_wall_so2
            r_wall_o2 = self._wall_o2_rate(C_O2, h2o_ppm_here, C_NO2_wall, C_H2SO4_wall, C_HNO3_wall,
                                           C_O2_feed=(C_lagged_o2_feed if self.o2_feed_lag_tau_hours > 0.0
                                                      else C_O2_feed), C_H2S_feed=C_H2S_feed)
            R_O2 -= r_wall_o2
            if self.wall_consume_h2o:
                R_H2O -= r_wall_o2
            if self.wall_hno3_corrosion_k_intrinsic > 0.0:
                r_hno3corr = self._wall_hno3_corrosion_rate(h2o_ppm_here, C_HNO3_wall, C_H2SO4_wall)
                R_HNO3 -= (8.0 / 3.0) * r_hno3corr
                R_NO += (2.0 / 3.0) * r_hno3corr
                R_H2O += (4.0 / 3.0) * r_hno3corr
            if self.wall_h2so4_k_intrinsic > 0.0:
                r_h2so4 = self._wall_h2so4_rate(h2o_ppm_here, C_H2SO4_wall, C_HNO3_wall)
                R_H2SO4 -= r_h2so4
                R_H2 += r_h2so4
            if self.wall_no2_k_intrinsic > 0.0:
                r_wall_no2 = self._wall_no2_rate(C_NO2_wall, h2o_ppm_here, C_cum_no2_exposure,
                                                 C_O2=C_O2, C_H2S_raw=C_raw[0],
                                                 C_H2SO4_raw=C_raw[6], C_O2_feed=C_O2_feed,
                                                 C_O2_lagged=C_lagged_o2, C_NO=C_raw[3])
                R_NO2 -= r_wall_no2
                R_NO += r_wall_no2
            cum_no2_rate = C_NO2_wall
            R_H2 += self._wall_feco3_rate(h2o_ppm_here, C_H2SO4_wall, C_HNO3_wall)
            r_wall_s8 = self._wall_s8_rate(C_H2S, C_O2)
            R_H2S -= r_wall_s8
            R_O2 -= 0.5 * r_wall_s8
            R_H2O += r_wall_s8
            R_S8 += 0.125 * r_wall_s8


        R_vector = np.array([
            R_H2S, R_SO2, R_NO2, R_NO, R_O2, R_H2O,
            R_H2SO4, R_HNO3, R_S8, R_NH3, R_N2O, R_H2, R_N2O4, R_HNO2
        ])

        if C_in is not None and space_time_sec is not None and space_time_sec > 0.0:
            if inflow_only:
                dC_dt_species = C_in[:n_species] / space_time_sec + R_vector
            else:
                dC_dt_species = (C_in[:n_species] - C[:n_species]) / space_time_sec + R_vector
        else:
            dC_dt_species = R_vector

        if len(C) > n_species:
            dC_dt_wall_solid = np.array([r_h2so4, r_hno3corr, cum_h2so4_rate, cum_hno3_rate,
                                         cum_no2_rate, lagged_o2_rate, cum_o2_rate,
                                         lagged_o2_feed_rate, cum_nh3_rate,
                                         self._acid_site_response(C_raw[6], acid_sites)])
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
        C0 = np.zeros(n_species + n_extra)
        for idx, spec in enumerate(self.SPECIES):
            if spec in initial_ppm:
                C0[idx] = (initial_ppm[spec] * 1.0e-6) * self.molar_density
        if initial_wall_solid is not None:
            for offset, key in enumerate(self.EXTRA_STATE_KEYS):
                C0[n_species + offset] = max(0.0, initial_wall_solid.get(key, 0.0))

        C_in = None
        if feed_ppm is not None:
            C_in = np.zeros(n_species + n_extra)
            for idx, spec in enumerate(self.SPECIES):
                if spec in feed_ppm:
                    C_in[idx] = (feed_ppm[spec] * 1.0e-6) * self.molar_density

        moisture_basis = feed_ppm if feed_ppm is not None else initial_ppm
        moisture_ppm = moisture_basis.get('H2O', self.water_ppm)
        rates_dict = self._calculate_pure_physical_rate_constants(moisture_ppm)

        def derivative(time, state):
            return self.rhs(time, state, rates_dict, C_in=C_in, space_time_sec=space_time_sec,
                            inflow_only=inflow_only)

        def jacobian(time, state):
            steps = np.sqrt(np.finfo(float).eps) * np.maximum(np.abs(state), 1e-32)
            return approx_fprime(state, lambda trial: derivative(time, trial), steps)

        absolute_tolerance = np.full(len(C0), 1e-13)
        absolute_tolerance[[self.SPECIES.index(species) for species in ('O2', 'H2S', 'NO2')]] = 1e-32
        sol = solve_ivp(
            fun=derivative,
            t_span=t_span,
            y0=C0,
            t_eval=t_eval,
            method='BDF',
            jac=jacobian,
            rtol=2e-8,
            atol=absolute_tolerance
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


class CO2ImpurityReactorExperiment:
    """Configure and run sequential fixed-pressure CSTR simulations."""

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

        self.initial_gas = 'CO2'
        self.initial_P_bar = self.target_P
        self.initial_T_C = self.target_T_C

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
        """Apply a named empirical parameter profile to the underlying model."""
        self.model.apply_calibrated_profile(profile=profile)

    def set_initial_vessel_charge(self, gas_name='N2', pressure_bar=1.0, temp_C=25.0):
        """Specify the carrier charge at the start of the impurity-dosing clock."""
        if (not np.isfinite(pressure_bar) or pressure_bar <= 0.0
                or not np.isfinite(temp_C) or temp_C <= -273.15):
            raise ValueError('Initial charge requires positive finite pressure and absolute temperature')
        self.initial_gas = str(gas_name).strip().upper()
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
        if not np.isfinite(duration_hours) or duration_hours <= 0.0:
            raise ValueError('Phase duration must be finite and positive')
        p_idx = len(self.phases)
        name = phase_name if phase_name else f"Phase {p_idx}"
        phase_mass_flow_g_h = self.mass_flow_g_h if mass_flow_g_h is None else float(mass_flow_g_h)
        if not np.isfinite(phase_mass_flow_g_h) or phase_mass_flow_g_h < 0.0:
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
            f"\nInitial vessel charge: {self.initial_gas}, "
            f"{self.initial_P_bar:.3f} bar, {self.initial_T_C:.3f} °C "
            "(fixed-pressure through-flow from t=0; no artificial vessel-fill stage)"
        )
        return report + charge

    def run_experiment(self):
        if not self.phases:
            self.add_phase(10.0, {s: 0.0 for s in self.model.SPECIES}, "Phase 0: Pure CO2 Flow")
            self.add_phase(20.0, {'SO2': 10.0, 'NO2': 10.0, 'O2': 10.0, 'H2O': 10.0}, "Phase 1: 10 ppm Without H2S")
            self.add_phase(20.0, {'H2S': 10.0, 'SO2': 10.0, 'NO2': 10.0, 'O2': 10.0, 'H2O': 10.0}, "Phase 2: 10 ppm All Impurities")

        first_phase = self.phases[0]
        first_pressure = first_phase['pressure_bar'] if first_phase['pressure_bar'] is not None else self.target_P
        first_temperature = first_phase['temp_C'] if first_phase['temp_C'] is not None else self.target_T_C
        if (self.initial_gas != 'CO2'
                or not np.isclose(self.initial_P_bar, first_pressure, rtol=1e-6, atol=1e-9)
                or not np.isclose(self.initial_T_C, first_temperature, rtol=0.0, atol=1e-6)):
            raise ValueError(
                'The initial charge must be pure CO2 at the first phase pressure and temperature. '
                'This fixed-pressure model does not simulate pressurization or carrier-gas displacement; '
                'start impurity dosing after conditioning, or use a variable-inventory model.'
            )

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
            self.model.set_conditions(temp_C=phase.get('temp_C'),
                                      pressure_bar=phase.get('pressure_bar'))
            self.set_reactor_geometry(mass_flow_g_h=phase_mass_flow_g_h)
            tau_sec = self.model.get_reactor_geometry()['residence_time_seconds']
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

            first_sample = 0 if idx == 0 else 1
            all_t_h.append(current_cumulative_t + t_res[first_sample:])
            for s in self.model.SPECIES:
                all_ppm[s].append(ppm_res[s][first_sample:])
            for k in extra_keys:
                all_wall_solid[k].append(wall_solid_res[k][first_sample:])

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
    'N2O4':  {'N': 2, 'O': 4},
    'HNO2':  {'H': 1, 'N': 1, 'O': 2},
    'H2':    {'H': 2},
}


KW_298 = 1.0e-14
DH_KW = 55800.0

KH_CO2_298_MOL_L_ATM = 3.3e-2
DH_KH_CO2 = -19950.0

KA1_CO2_298 = 4.45e-7
DH_KA1_CO2 = 7700.0

KA2_CO2_298 = 4.69e-11
DH_KA2_CO2 = 14900.0

KA_NH4_298 = 5.6e-10
DH_KA_NH4 = 52200.0

KA2_H2SO4_298 = 1.2e-2

PROTON_MASS_G_MOL = 1.008
M_SO4_2MINUS = MW_H2SO4 - 2.0 * PROTON_MASS_G_MOL
M_NO3_MINUS = MW_HNO3 - PROTON_MASS_G_MOL
M_NH4_PLUS = MW_NH3 + PROTON_MASS_G_MOL


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
    so4 = h2so4_t * ka2_so4 / (h + ka2_so4)
    hso4 = h2so4_t - so4
    no3 = hno3_t
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
        return 7.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f_mid = _wash_water_charge_balance(mid, *args)
        if f_lo * f_mid <= 0.0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


class AutoclaveExperiment:
    """Notebook-friendly facade over :class:`CO2ImpurityReactorExperiment`."""


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
        'R16': '2 NO2 <-> N2O4',
        'R17': '2 NO2 + H2O -> HNO3 + HNO2',
        'R18': 'HNO2 + 0.5 O2 -> HNO3',
        'Wall O2 / Fe2O3': '4 Fe + 3 O2 -> 2 Fe2O3 (surface path)',
        'Wall NO2 / Fe2O3': '2 Fe + 3 NO2 -> Fe2O3 + 3 NO (dry surface path)',
        'FeCO3 deposit': 'Fe + CO2(aq) + H2O -> FeCO3 + H2 (surface path, disabled by default)',
        'Wall HNO3 / Fe(NO3)2': '8 HNO3 + 3 Fe -> 3 Fe(NO3)2 + 2 NO + 4 H2O (surface path)',
        'Wall H2SO4 / FeSO4': 'Fe + H2SO4 -> FeSO4 + H2 (surface path)',
        'Wall S8 / Claus': '8 H2S + 4 O2 -> S8 + 8 H2O (carbon-steel-catalysed, surface path)',
        'Wall SO2 / H2SO4': 'SO2 + 0.5 O2 + H2O -> H2SO4 (surface-catalysed, disabled by default)',
    }
    REACTION_SPECIES = {
        'R1': {'SO2': -1, 'O2': -0.5, 'H2O': -1, 'H2SO4': 1},
        'R2': {'H2S': -1, 'NO2': -3, 'SO2': 1, 'H2O': 1, 'NO': 3},
        'R3A': {'SO2': -1, 'NO2': -1, 'H2O': -1, 'NO': 1, 'H2SO4': 1},
        'R4': {'NO': -2, 'O2': -1, 'NO2': 2},
        'R5': {'NO2': -3, 'H2O': -1, 'HNO3': 2, 'NO': 1},
        'R7': {'H2S': -5, 'NO': -6, 'H2O': -4, 'NH3': 6, 'SO2': 5},
        'R10': {'NH3': -4, 'NO': -4, 'O2': -3, 'N2O': 4, 'H2O': 6},
        'R11': {'H2S': -1, 'NO': -2, 'N2O': 1, 'S8': 0.125, 'H2O': 1},
        'R12': {'H2S': -1, 'O2': -2, 'H2SO4': 1},
        'R13': {'NO2': -4, 'H2S': -1, 'H2SO4': 1, 'NO': 4},
        'R15': {'NO2': -4, 'N2O': 2, 'O2': 3},
        'R16': {'NO2': -2, 'N2O4': 1},
        'R17': {'NO2': -2, 'H2O': -1, 'HNO3': 1, 'HNO2': 1},
        'R18': {'HNO2': -1, 'O2': -0.5, 'HNO3': 1},
        'Wall O2 / Fe2O3': {'O2': -1},
        'Wall NO2 / Fe2O3': {'NO2': -1, 'NO': 1},
        'FeCO3 deposit': {'H2': 1},
        'Wall HNO3 / Fe(NO3)2': {'HNO3': -8.0 / 3.0, 'NO': 2.0 / 3.0, 'H2O': 4.0 / 3.0},
        'Wall H2SO4 / FeSO4': {'H2SO4': -1, 'H2': 1},
        'Wall S8 / Claus': {'H2S': -1, 'O2': -0.5, 'H2O': 1, 'S8': 0.125},
        'Wall SO2 / H2SO4': {'SO2': -1, 'H2SO4': 1},
    }
    REACTANT_STYLE = {'H2S': '#e74c3c', 'SO2': '#f39c12', 'NO2': '#9b59b6', 'O2': '#2ecc71', 'H2O': '#3498db'}
    PRODUCT_STYLE = {
        'H2SO4': ('#c0392b', '-'), 'HNO3': ('#8e44ad', '-'), 'S8': ('#f1c40f', '-'),
        'NH3': ('#16a085', '-'), 'NO': ('#7f8c8d', '--'), 'N2O': ('#d35400', '--'),
        'N2O4': ('#27ae60', '--'),
        'HNO2': ('#e67e22', '-'),
        'H2': ('#2980b9', '-'),
    }
    PHASE_COLORS = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c']
    INTERACTIVE_PLOT_CONFIG = {
        'responsive': True, 'displaylogo': False, 'scrollZoom': True, 'showTips': False,
        'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'sendDataToCloud', 'share'],
    }

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
        """Build a sequential feed schedule ending at termination_hour [h].

        Accept (start, feed[, label, flow]), (start, flow, feed[, label]), or
        (start, spec) entries. A spec can override feed, flow [g/h], temperature
        [C], pressure [bar], and label for that phase.
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
        no_r4 = np.clip(ppm['NO'] * 1e-6 * rho_m * model.r4_no_activity * model._r4_surface_factor(), 0.0, None)

        o2_feed_ppm_series = np.zeros_like(t_h)
        h2s_feed_ppm_series = np.zeros_like(t_h)
        no2_feed_ppm_series = np.zeros_like(t_h)
        h2o_feed_ppm_series = np.zeros_like(t_h)
        for (_, feed), mask in zip(self.phases, self._phase_masks()):
            o2_feed_ppm_series[mask] = feed.get('O2', 0.0)
            h2s_feed_ppm_series[mask] = feed.get('H2S', 0.0)
            no2_feed_ppm_series[mask] = feed.get('NO2', 0.0)
            h2o_feed_ppm_series[mask] = feed.get('H2O', model.water_ppm)
        C_O2_feed_series = o2_feed_ppm_series * 1e-6 * rho_m
        C_H2S_feed_series = h2s_feed_ppm_series * 1e-6 * rho_m
        C_NO2_feed_series = no2_feed_ppm_series * 1e-6 * rho_m

        names = ('R1', 'R2', 'R3A', 'R4', 'R5', 'R7', 'R10', 'R11', 'R12', 'R13', 'R15', 'R16',
                 'R17', 'R18',
                 'Wall O2 / Fe2O3', 'FeCO3 deposit', 'Wall HNO3 / Fe(NO3)2', 'Wall H2SO4 / FeSO4',
                 'Wall NO2 / Fe2O3', 'Wall S8 / Claus', 'Wall SO2 / H2SO4')
        rate = {name: np.zeros_like(t_h) for name in names}

        for i in range(len(t_h)):
            k = model.get_reaction_rates(moisture_ppm=h2o_feed_ppm_series[i])
            h2s, so2, no2 = C['H2S'][i], C['SO2'][i], C['NO2'][i]
            no, o2, h2o = C['NO'][i], C['O2'][i], C['H2O'][i]
            h2so4, hno3 = C['H2SO4'][i], C['HNO3'][i]
            nh3, s8, n2o = C['NH3'][i], C['S8'][i], C['N2O'][i]
            c_h2s_raw = ppm['H2S'][i] * 1e-6 * rho_m
            c_h2so4_raw = ppm['H2SO4'][i] * 1e-6 * rho_m
            c_o2_feed = C_O2_feed_series[i]
            c_h2s_feed = C_H2S_feed_series[i]
            c_no2_feed = C_NO2_feed_series[i]
            c_o2_lagged = self.wall_solid['LaggedO2'][i] if 'LaggedO2' in self.wall_solid else 0.0
            c_so2_raw = ppm['SO2'][i] * 1e-6 * rho_m
            acid_so2_sat = model._acid_so2_saturation(c_so2_raw)

            rate['R1'][i] = (k['k1_f'] * so2 * _fractional_activity(o2, 0.5) * h2o - k['k1_r'] * h2so4) * acid_so2_sat \
                * model._feed_o2_passivation(c_o2_feed, model.r1_feed_o2_ref_ppm,
                                             model.r1_feed_o2_hill_n,
                                             cap_ppm=model.r1_feed_o2_cap_ppm)
            cumulative_acid_ppm = self.wall_solid['CumH2SO4'][i] / max(rho_m, 1e-9) * 1e6
            rate['R1'][i] *= model._r1_autocat_factor(cumulative_acid_ppm)
            r2_no2_boost = 1.0
            if model.r2_no2_boost_ref_ppm > 0.0:
                no2_ppm_r2 = no2 / max(rho_m, 1e-9) * 1e6
                ratio_r2 = (no2_ppm_r2 / model.r2_no2_boost_ref_ppm) ** model.r2_no2_boost_hill_n
                r2_no2_boost = 1.0 + model.r2_no2_boost_gain * ratio_r2 / (1.0 + ratio_r2)
            r2_scale = r2_no2_boost * k['f_phase'] ** model.r2_f_phase_exponent \
                * model._o2_presence_gate(c_o2_feed, model.r2_o2_presence_ref_ppm,
                                          model.r2_o2_presence_hill_n) \
                * model._r2_no2_excess_gate(c_h2s_feed, c_no2_feed)
            r13_scale = k['f_phase'] * model._r13_rate_factor(no2)
            r13_bimolecular_coefficient = k['k13_f'] * r13_scale \
                * max(no2, model.r13_floor_kmol_m3) ** (model.r13_no2_order - 1.0)
            r2_coefficient = model._r2_effective_coefficient(k['k2_f'] * r2_scale, no2, h2s)
            r13_dilute = model._r13_dilute_factor(r13_bimolecular_coefficient, r2_coefficient)
            r13_scale *= r13_dilute
            r13_bimolecular_coefficient *= r13_dilute
            h2s_no2_encounter = model._bimolecular_encounter_factor(r2_coefficient + r13_bimolecular_coefficient)
            h2s_no2_encounter *= model._h2s_no2_temperature_factor()
            rate['R2'][i] = h2s_no2_encounter * r2_coefficient \
                * (h2s * no2 - so2 * h2o * no**3 / k['Keq2'])
            no_r3a = no * (1.0 - model.r3a_no_escape_frac * k['f_phase'])
            r3a_acid_signal = model._r3a_acid_signal(cumulative_acid_ppm, c_h2so4_raw)
            rate['R3A'][i] = model._r3a_bore_factor() * model._r3a_autocat_factor(r3a_acid_signal) \
                * (k['k3a_f'] * so2 * no2 * h2o - k['k3a_r'] * no_r3a * h2so4) \
                * acid_so2_sat \
                * model._r3a_base_so2_inhibition(ppm['SO2'][i]) \
                * model._r3a_oxygen_inhibition(ppm['O2'][i] * rho_m * 1e-6, c_o2_feed) \
                * model._o2_presence_gate(c_o2_feed, model.r3a_o2_presence_ref_ppm,
                                          model.r3a_o2_presence_hill_n) \
                * model._r3a_h2s_feed_inhibition(c_h2s_feed, c_no2_feed)
            r3a_acid_coefficient = model._r3a_acid_film_rate_constant(r3a_acid_signal) \
                + model._r3a_dilute_acid_rate_constant(
                    r3a_acid_signal, no2, c_h2so4_raw, self.wall_solid['CumHNO3'][i] / rho_m * 1e6,
                    h2s, self.wall_solid['AcidSiteActivity'][i])
            r3a_driving_force = so2 * no2 * h2o - no * h2so4 / k['Keq3']
            r3a_conditioned = model._r3a_conditioned_acid_rate_constant(
                r3a_acid_signal, c_h2so4_raw, c_so2_raw, ppm['NO'][i] * rho_m * 1e-6) * r3a_driving_force
            rate['R3A'][i] += r3a_acid_coefficient * r3a_driving_force \
                + model._limit_r3a_conditioned_rate(r3a_conditioned, cumulative_acid_ppm)
            rate['R3A'][i] *= model._r3a_wetting_factor(
                ppm['H2O'][i], c_h2so4_raw, ppm['HNO3'][i] * rho_m * 1e-6)
            c_o2_r4 = o2
            if model.r4_o2_half_ppm > 0.0:
                c_o2_r4 *= (9.6 + model.r4_o2_half_ppm) / (ppm['O2'][i] + model.r4_o2_half_ppm)
            rate['R4'][i] = k['k4_f'] * no_r4[i]**2 * c_o2_r4 - k['k4_r'] * no2**2
            rate['R4'][i] += model._r4_acid_film_rate_constant(cumulative_acid_ppm) \
                * (no ** 2 * o2 - no2 ** 2 / k['Keq4'])
            rate['R4'][i] *= model._r4_dilute_no_factor(ppm['NO'][i] * rho_m * 1e-6)
            rate['R5'][i] = model._r5_rate_factor(no2) * (k['k5_f'] * no2**3 * h2o
                                                        - k['k5_r'] * hno3**2 * no_r5[i])
            rate['R7'][i] = model._r7_rate_factor(no) * k['k7_f'] * h2s * no * model._r7_water_activity(h2o)
            rate['R10'][i] = k.get('k10_f', 0.0) * nh3 * no * o2 - k.get('k10_r', 0.0) * n2o**4 * h2o**6
            r11_o2_boost = 1.0
            if model.r11_o2_ref_ppm > 0.0:
                o2_ppm_r11 = o2 / max(rho_m, 1e-9) * 1e6
                ratio_n = (o2_ppm_r11 / model.r11_o2_ref_ppm) ** model.r11_o2_hill_n
                r11_o2_boost = 1.0 + model.r11_o2_gain * ratio_n / (1.0 + ratio_n)
            rate['R11'][i] = r11_o2_boost * k.get('k11_f', 0.0) * h2s * no - k.get('k11_r', 0.0) * n2o * s8**0.125 * h2o
            r12_scale = 1.0 if model.r12_density_independent else k['f_phase'] ** model.r12_f_phase_exponent
            no2_r12 = model._r12_catalyst_factor(no2)
            rate['R12'][i] = r12_scale * model._r12_h2s_rate_factor(h2s) * (k.get('k12_f', 0.0) * h2s * o2 * no2_r12
                                           - k.get('k12_r', 0.0) * h2so4 * no2_r12)
            rate['R12'][i] *= model._h2s_no2_temperature_factor()
            rate['R13'][i] = h2s_no2_encounter * r13_scale * (k.get('k13_f', 0.0) * h2s * no2 ** model.r13_no2_order
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
                    no2_ppm_r15_capped = no2_ppm_r15 / (1.0 + ratio_n) ** (1.0 / model.r15_no2_cap_hill_n)
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
            rate['R15'][i] = r15_o2_presence * k['f_phase'] ** model.r15_f_phase_exponent * (r15_o2_inhib * r15_o2_activation * r15_n2o_brake * model._r15_surface_suppression() * model._r15_dimer_availability() * model._sulfur_catalyst_gate(c_h2s_raw, c_h2so4_raw) * k.get('k15_f', 0.0) * no2_r15**4 - k.get('k15_r', 0.0) * n2o**2 * o2**3)
            n2o4 = C['N2O4'][i] if 'N2O4' in C else 0.0
            rate['R16'][i] = k.get('k16_f', 0.0) * no2**2 - k.get('k16_r', 0.0) * n2o4
            hno2 = C['HNO2'][i] if 'HNO2' in C else 0.0
            rate['R17'][i] = (k.get('k17_f', 0.0) * no2**2 * h2o - k.get('k17_r', 0.0) * hno3 * hno2) \
                * model._r17_rate_factor(no2, h2o, ppm['H2O'][i])
            rate['R18'][i] = k.get('k18_f', 0.0) * hno2 * _fractional_activity(o2, 0.5) - k.get('k18_r', 0.0) * hno3

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
                                                 C_NO=no_raw, C_O2_feed_lagged=lagged_o2_feed_i,
                                                 C_H2S_feed=c_h2s_feed)
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
        return np.trapezoid(np.abs(series[mask]), t_s[mask]) * V_m3 * 1e6

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
        """Corrosion-product mass accumulation and corrosion rate vs time."""
        model = self.exp.model
        rate = self._compute_reaction_rate_series()
        t_h = self.t_h
        t_s = t_h * 3600.0
        dt = np.diff(t_s, prepend=0.0)
        V_m3 = self.volume_ml * 1e-6

        n_o2_lost = np.cumsum(rate['Wall O2 / Fe2O3'] * V_m3 * dt)
        n_fe2o3 = n_o2_lost * (2.0 / 3.0)
        n_no2_lost = np.cumsum(rate['Wall NO2 / Fe2O3'] * V_m3 * dt)
        n_fe2o3 += n_no2_lost * (1.0 / 3.0)
        n_feco3 = np.cumsum(rate['FeCO3 deposit'] * V_m3 * dt)
        n_fe_no3_2 = self.wall_solid['FeNO32'] * V_m3
        n_fe_so4 = self.wall_solid['FeSO4'] * V_m3
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
        """
        if self.results is None:
            raise RuntimeError('Call run() before get_mass_balance_table().')

        model = self.exp.model
        t_h, t_s = self.t_h, self.t_h * 3600.0
        V_m3 = self.volume_ml * 1e-6
        elements = ('N', 'S', 'H', 'O')

        inventory_kmol = {e: np.zeros_like(t_h) for e in elements}
        for species in model.SPECIES:
            counts = SPECIES_ATOM_COUNTS.get(species, {})
            if not counts:
                continue
            mole_fraction = self.ppm[species] * 1e-6
            C_kmol_m3 = mole_fraction * model.molar_density
            for e, n_atoms in counts.items():
                inventory_kmol[e] += n_atoms * C_kmol_m3 * V_m3

        surf = self.get_surface_data()
        n_fe_no3_2 = surf['FeNO32_mg'] / (self.M_FE_NO3_2 * 1e6)
        n_fe_so4 = surf['FeSO4_mg'] / (self.M_FE_SO4 * 1e6)
        n_feco3 = surf['FeCO3_mg'] / (self.M_FECO3 * 1e6)
        n_fe2o3 = surf['Fe2O3_mg'] / (self.M_FE2O3 * 1e6)
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
                ) * 1e6

                outflow_mmol = _trapz(inventory_kmol[e][i0:i1 + 1], t_s[i0:i1 + 1]) \
                    * phase_molar_flow_kmol_s / (model.molar_density * V_m3) * 1e6
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
        """Estimate the pH of a fixed mass of wash water continuously scrubbing the
        reactor's CO2 off-gas, vs time (illustrative, screening-level).
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

        water_l = water_mass_g / 1000.0
        cum_mol_l = {}
        for species in ('NH3', 'H2SO4', 'HNO3'):
            mole_flow_kmol_s = ppm[species] * 1e-6 * molar_flow_kmol_s
            cum_kmol = _cumulative_trapz(mole_flow_kmol_s, t_s)
            cum_mol_l[species] = np.clip(cum_kmol * 1000.0 / water_l, 0.0, None)

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
        """Estimate ion-chromatography-style SO4^2-/NO3-/NH4+ results for a fixed mass of
        water used to rinse the autoclave's OWN internals after the run, vs time
        (illustrative, screening-level; companion to ``get_wash_water_pH_table``,
        which instead models washing the CO2 OFF-GAS through an external bottle).
        """
        if self.results is None:
            raise RuntimeError('Call run() before get_autoclave_wash_table().')
        if wash_mass_g <= 0.0:
            raise ValueError(f'wash_mass_g must be positive, got {wash_mass_g}')

        t_h = self.t_h
        V_m3 = self.volume_ml * 1e-6
        wash_l = wash_mass_g / 1000.0

        cum_h2so4_kmol = self.wall_solid['CumH2SO4'] * V_m3
        cum_hno3_kmol = self.wall_solid['CumHNO3'] * V_m3
        cum_nh3_kmol = self.wall_solid.get('CumNH3', np.zeros_like(t_h)) * V_m3

        so4_umol = cum_h2so4_kmol * 1e9
        no3_umol = cum_hno3_kmol * 1e9
        nh4_umol = cum_nh3_kmol * 1e9

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

    def _plot_interactive_reactions(self, reaction=None, show=True, show_feed=False, height=720):
        """Plot component concentrations and one key reaction per feed step."""
        from html import escape
        from textwrap import wrap
        try:
            import plotly.graph_objects as go
        except ImportError as error:
            raise ImportError("Interactive reactions require Plotly: python -m pip install plotly") from error

        if any(phase.get(key) is not None for phase in self.exp.phases for key in ('temp_C', 'pressure_bar')):
            raise ValueError('Interactive reaction rates currently require constant temperature and pressure.')
        if not np.isfinite(height) or height < 500:
            raise ValueError('Interactive plot height must be finite and at least 500 pixels.')
        names = {name.lower(): name for name in self.REACTION_NAMES}
        selected = None if reaction is None else names.get(str(reaction).strip().lower())
        if reaction is not None and selected is None:
            raise ValueError(f'Unknown reaction {reaction!r}. Choose from {list(self.REACTION_NAMES)}')

        model = self.exp.model
        rate_series = self._compute_reaction_rate_series()
        coefficients = {name: dict(values) for name, values in self.REACTION_SPECIES.items()}
        if model.wall_consume_h2o:
            coefficients['Wall O2 / Fe2O3']['H2O'] = -1
        species = list(model.SPECIES)
        colors = {name: style[0] for name, style in self.PRODUCT_STYLE.items()}
        colors.update(self.REACTANT_STYLE)
        overview_species = {'H2O', 'H2S', 'SO2', 'NO2', 'O2', 'NO', 'H2SO4', 'HNO3', 'S8'}
        figure = go.Figure()
        for name in species:
            figure.add_trace(go.Scatter(
                x=self.t_h, y=self.ppm[name], name=name, legendgroup=name,
                line=dict(color=colors.get(name, '#59636b'), width=2.3),
                visible=True if name in overview_species else 'legendonly',
                hovertemplate=f'{name}: %{{y:.4g}} ppm-mol<br>%{{x:.2f}} h<extra></extra>',
            ))
            if show_feed and any(feed.get(name, 0.0) for _, feed in self.phases):
                feed_time, feed_ppm = self._feed_profile(name)
                figure.add_trace(go.Scatter(
                    x=feed_time, y=feed_ppm, name=f'{name} feed', legendgroup=name, showlegend=False,
                    line=dict(color=colors.get(name, '#59636b'), width=1.1, dash='dot'),
                    opacity=0.5, visible=True if name in overview_species else 'legendonly',
                    hovertemplate=f'{name} feed: %{{y:.4g}} ppm-mol<br>%{{x:.2f}} h<extra></extra>',
                ))

        full_range = [float(self.t_h[0]), float(self.t_h[-1])]
        buttons = [dict(label='Full run', method='relayout',
                        args=[{'xaxis.range': full_range, 'annotations': []}])]
        step_reactions = []
        for step, ((start, end), (phase_name, _), mask) in enumerate(
                zip(self.phase_bounds, self.phases, self._phase_masks()), start=1):
            indices = np.flatnonzero(mask)
            indices = indices[(indices >= np.searchsorted(self.t_h, start, side='right') - 1)
                              & (indices <= np.searchsorted(self.t_h, end, side='left'))]
            mask = np.zeros_like(self.t_h, dtype=bool)
            mask[indices] = True
            scores = {
                name: self._integrated_extent_mmol(np.asarray(values), mask)
                * sum(abs(value) for value in coefficients[name].values())
                for name, values in rate_series.items()
            }
            pathway = selected or max(scores, key=scores.get)
            if scores[pathway] <= 0.0:
                pathway = None
            text = f'<b>Step {step}: no active reaction</b>'
            direction = None
            if pathway is not None:
                rates = np.asarray(rate_series[pathway])[indices]
                equation = self.REACTION_NAMES[pathway]
                direction = 'mixed' if np.any(rates > 0.0) and np.any(rates < 0.0) else (
                    'reverse' if np.any(rates < 0.0) else 'forward')
                if direction != 'forward':
                    equation = equation.replace(' -> ', ' <-> ' if direction == 'mixed' else ' <- ')
                formatted = '<br>'.join(escape(line) for line in wrap(equation, width=30))
                formatted = formatted.replace('&lt;-&gt;', '&#8596;').replace('-&gt;', '&#8594;')
                formatted = formatted.replace('&lt;-', '&#8592;')
                text = f'<b>Step {step}: {escape(pathway)}</b><br>{formatted}'
                peak_index = indices[int(np.argmax(np.abs(rates)))]
                formed = {name: coefficient * rate_series[pathway][peak_index]
                          for name, coefficient in coefficients[pathway].items()}
                component = max(formed, key=formed.get)
                hour = 0.5 * (start + end)
                figure.add_trace(go.Scatter(
                    x=[hour], y=[float(np.interp(hour, self.t_h, self.ppm[component]))],
                    mode='markers', name=f'Step {step}', showlegend=False,
                    marker=dict(symbol='diamond', size=8, color=colors.get(component, '#59636b'),
                                line=dict(color='white', width=1)),
                    meta=dict(step=step, pathway=pathway, component=component),
                    hovertemplate=f'{text}<br>{escape(phase_name)}<br>{start:g}-{end:g} h<extra></extra>',
                ))
            annotation = dict(x=0.5, y=-0.14, xref='paper', yref='paper', xanchor='center',
                              yanchor='top', align='center', text=text, showarrow=False, font=dict(size=12))
            buttons.append(dict(label=f'Step {step}', method='relayout',
                                args=[{'xaxis.range': [float(start), float(end)],
                                       'annotations': [annotation]}]))
            step_reactions.append(dict(step=step, phase=phase_name, start=float(start), end=float(end),
                                       reaction=pathway, direction=direction))
            figure.add_vline(x=start, line_width=0.6, line_dash='dot', line_color='#bdc4c9')
        figure.update_layout(
            title=dict(text='Component concentrations', x=0.02, y=0.99, font=dict(size=18)),
            height=int(height), autosize=True, template='plotly_white',
            font=dict(family='Aptos, Calibri, sans-serif', size=12, color='#26323a'),
            margin=dict(l=65, r=40, t=95, b=225), hovermode='closest',
            modebar=dict(orientation='v'),
            legend=dict(orientation='h', y=-0.35, yanchor='top', x=0.5, xanchor='center',
                        entrywidth=36, tracegroupgap=0, groupclick='togglegroup', font=dict(size=11)),
            updatemenus=[dict(buttons=buttons, active=0, direction='down', x=0, y=1.13,
                             xanchor='left', yanchor='top')],
            meta=dict(step_reactions=step_reactions, concentration_units='ppm-mol',
                      reaction_ranking='integrated absolute component conversion',
                      conditions=self._condition_string()),
        )
        figure.update_xaxes(range=full_range, title_text='Time (h)', showgrid=True, gridcolor='#e8ecee')
        figure.update_yaxes(title_text='Concentration (ppm-mol)', rangemode='tozero')
        if show:
            figure.show(config=self.INTERACTIVE_PLOT_CONFIG)
        return figure

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
        'interactive_reactions': _plot_interactive_reactions,
    }

    def build_plot(self, kind, **kwargs):
        """Render one of: 'reactant species', 'reaction products',
        'surface reaction products', 'corrosion rate', 'wash water pH',
        or 'interactive reactions' (optional Plotly dependency)."""
        if self.results is None:
            raise RuntimeError('Call run() before build_plot().')
        key = kind.strip().lower().replace(' ', '_')
        method = self._PLOT_KINDS.get(key)
        if method is None:
            raise ValueError(f"Unknown plot kind {kind!r}. Choose from: "
                              f"{sorted(set(self._PLOT_KINDS))}")
        return method(self, **kwargs)
