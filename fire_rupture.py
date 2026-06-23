# -*- coding: utf-8 -*-
"""
Fire rupture (time-to-rupture) calculation engine
=================================================

Python re-implementation of the Equinor "Fire rupture calc – Strain rate model"
spreadsheet (Rev 4.x). Calculates the heat-up of a pressurised pipe (or vessel
wall) exposed to fire and predicts the time to rupture using a creep / strain-rate
(Garofalo / Sellars-Tegart) model. Also estimates the gas / liquid release rate
from the rupture location.

The physics mirrors the original workbook:

* Transient lumped heat balance for the metal wall + contained fluid.
* Temperature dependent material properties (Cp, k, UTS, strain limit).
* Thick-wall (Lamé) stress state combined into a von Mises stress, including an
  optional external "weight" axial stress.
* A creep strain-rate law calibrated so that the model reproduces the tabulated
  UTS curve at a reference strain rate (the ``T_corr`` correction).
* Rupture when the accumulated strain exceeds the temperature dependent strain
  limit.
* Choked-gas / liquid orifice release-rate correlations at the rupture pressure.

This module has no external dependencies beyond ``numpy`` / ``math`` and can be
unit-tested independently of Streamlit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

STEFAN_BOLTZMANN = 5.67e-08  # W/m2K4
STEEL_DENSITY = 7850.0       # kg/m3 (all listed materials)
GAS_CONSTANT = 8314.0        # J/kmol-K

# ---------------------------------------------------------------------------
# Material database (temperature dependent properties + creep constants)
# Extracted directly from the "Material" sheet of the reference workbook.
# Temperature grid is shared by all materials (deg C).
# ---------------------------------------------------------------------------
_TEMP_GRID = [0, 100, 200, 300, 400, 500, 600, 700, 750,
              800, 900, 1000, 1100, 1150, 1350]

MATERIALS: Dict[str, dict] = {
    "22Cr duplex": {
        "code": 1,
        "temp": _TEMP_GRID,
        "cp": [480, 500, 530, 550, 590, 635, 670, 710, 730, 750, 790, 840, 870, 879.9, 900],
        "k": [15, 16, 17, 18, 20, 24, 28, 28, 28, 28, 28, 28, 28, 28, 28],
        "uts": [650, 605, 553, 540, 533, 462, 371, 247, 202, 157, 78, 28, 12, 4, 0],
        "strain_effect": [1, 1, 1, 1, 1, 1, 1, 1, 0.9, 0.7, 1, 1.4, 2.3, 2.3, 2.3],
        "strain_limit": [0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.23, 0.26, 0.33, 0.5, 0.5, 0.5],
        "A": 12311757039.742393, "alpha": 0.03, "n": 2.156323825544742,
        "QR": 34115.772963646705, "ref_rate": 0.03,
    },
    "SS316": {
        "code": 2,
        "temp": _TEMP_GRID,
        "cp": [479.81656050955417, 495.06496815286624, 511.3299363057325, 520.4789808917197,
               528.6114649681529, 538.7770700636943, 549.9592356687898, 560.1248407643312,
               564, 568.2573248407643, 574.3566878980891, 574.3566878980891, 574.3566878980891,
               574.3566878980891, 574],
        "k": [13.5, 14.9, 16.7, 18.3, 19.8, 21.3, 22.7, 24.2, 24.8, 25.6, 27.1, 28.6, 30.5, 34.2, 34.2],
        "uts": [485, 467, 429, 426, 421, 398, 363, 277, 208.5, 140, 80, 45, 20, 10, 0],
        "strain_effect": [1, 1, 1, 1, 1, 1, 1, 1, 0.7, 0.5, 0.3, 0.25, 0.3, 0.3, 0.3],
        "strain_limit": [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.16, 0.17, 0.18, 0.2, 0.23, 0.25, 0.3],
        "A": 23449524917238.52, "alpha": 0.015, "n": 3.9045874502366082,
        "QR": 42315.65824131732, "ref_rate": 0.05,
    },
    "CS 235LT": {
        "code": 3,
        "temp": _TEMP_GRID,
        "cp": [450, 480, 510, 550, 600, 660, 750, 900, 1450, 820, 540, 540, 540, 540, 540],
        "k": [54.2, 50.95, 47.45, 43.7, 40.45, 37.2, 33.95, 30.7, 28, 27.4, 27.4, 27.4, 27.4, 27.4, 27.4],
        "uts": [420, 407, 397, 382, 370, 308, 189, 92, 81.5, 71, 53, 29, 17, 4, 0],
        "strain_effect": [1, 1, 1, 1, 1, 1, 1, 1, 0.7, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3],
        "strain_limit": [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.2, 0.2, 0.2, 0.2, 0.3],
        "A": 4425186183349.352, "alpha": 0.03, "n": 3.7222401933501312,
        "QR": 41462.45192432549, "ref_rate": 0.05,
    },
    "CS 360LT": {
        "code": 4,
        "temp": _TEMP_GRID,
        "cp": [450, 480, 510, 550, 600, 660, 750, 900, 1450, 820, 540, 540, 540, 540, 540],
        "k": [54.2, 50.95, 47.45, 43.7, 40.45, 37.2, 33.95, 30.7, 28, 27.4, 27.4, 27.4, 27.4, 27.4, 27.4],
        "uts": [545, 529, 515, 496, 480, 400, 245, 120, 99, 78, 60, 38, 22, 5, 0],
        "strain_effect": [1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.4, 0.6, 0.8, 0.9, 1, 1],
        "strain_limit": [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.2, 0.2, 0.2, 0.2, 0.3],
        "A": 7508371948335.949, "alpha": 0.015, "n": 4.346909864208215,
        "QR": 38301.22162583593, "ref_rate": 0.05,
    },
    "25Cr duplex": {
        "code": 5,
        "temp": _TEMP_GRID,
        "cp": [480, 500, 530, 550, 590, 635, 670, 710, 730, 750, 790, 840, 870, 879.9, 900],
        "k": [15, 16, 17, 18, 20, 24, 28, 28, 28, 28, 28, 28, 28, 28, 28],
        "uts": [750, 698, 638, 638, 613, 531, 427, 284, 234.5, 185, 100, 31, 13, 5, 0],
        "strain_effect": [1, 1, 1, 1, 1, 1, 1, 1, 0.9, 0.7, 1, 1.2, 2.2, 2.3, 2.2],
        "strain_limit": [0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.23, 0.26, 0.33, 0.5, 0.5, 0.5],
        "A": 352941684892.5195, "alpha": 0.03, "n": 2.1486991997520266,
        "QR": 39301.04947184141, "ref_rate": 0.03,
    },
    "6MO": {
        "code": 6,
        "temp": _TEMP_GRID,
        "cp": [479.81656050955417, 495.06496815286624, 511.3299363057325, 520.4789808917197,
               528.6114649681529, 538.7770700636943, 549.9592356687898, 560.1248407643312,
               564, 568.2573248407643, 574.3566878980891, 574.3566878980891, 574.3566878980891,
               574.3566878980891, 574],
        "k": [13.5, 14.9, 16.7, 18.3, 19.8, 21.3, 22.7, 24.2, 24.8, 25.6, 27.1, 28.6, 30.5, 34.2, 34.2],
        "uts": [650, 646, 589, 557, 546, 528, 480, 380, 315, 250, 135, 90, 40, 20, 0],
        "strain_effect": [1, 1, 1, 1, 1, 1, 1, 1, 0.7, 0.5, 0.3, 0.3, 0.3, 0.25, 0.3],
        "strain_limit": [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.2, 0.15, 0.15, 0.2, 0.2, 0.2, 0.2, 0.3],
        "A": 2443499691962.7837, "alpha": 0.015, "n": 3.0808262954383796,
        "QR": 41637.0426470313, "ref_rate": 0.05,
    },
}

# Common aliases used in pipe-class databases.
MATERIAL_ALIASES = {
    "DX": "22Cr duplex",
    "SDX": "25Cr duplex",
    "DUPLEX": "22Cr duplex",
    "SUPERDUPLEX": "25Cr duplex",
    "22CR DUPLEX": "22Cr duplex",
    "25CR DUPLEX": "25Cr duplex",
    "CS 235": "CS 235LT",
    "CS 360": "CS 360LT",
    "6 MO": "6MO",
}


def resolve_material(name: str) -> str:
    """Return the canonical material key for a (possibly aliased) name."""
    if name in MATERIALS:
        return name
    key = str(name).strip().upper()
    if key in MATERIAL_ALIASES:
        return MATERIAL_ALIASES[key]
    for mat in MATERIALS:
        if mat.upper() == key:
            return mat
    raise KeyError(f"Unknown material '{name}'. Valid: {list(MATERIALS)}")


# ---------------------------------------------------------------------------
# Default fire definitions (the 3 fires from the reference workbook).
# ---------------------------------------------------------------------------
@dataclass
class Fire:
    """A fire load definition used to compute the absorbed heat flux."""
    name: str
    fire_temp_C: float          # radiative source temperature
    gas_temp_C: float           # convective gas temperature
    h_conv: float               # convective heat transfer coefficient [W/m2K]
    fire_emissivity: float = 1.0
    metal_emissivity: float = 0.85
    metal_absorptivity: float = 0.85

    def incident_flux(self, surface_temp_C: float) -> float:
        """Incident heat flux [kW/m2] (uses metal emissivity for radiation in)."""
        return (
            STEFAN_BOLTZMANN
            * (self.fire_emissivity * 1.0 * (self.fire_temp_C + 273.0) ** 4
               - self.metal_emissivity * (surface_temp_C + 273.0) ** 4)
            + self.h_conv * (self.gas_temp_C - surface_temp_C)
        ) / 1000.0

    def absorbed_flux(self, surface_temp_C: float) -> float:
        """Absorbed heat flux [kW/m2] into the metal surface."""
        return (
            STEFAN_BOLTZMANN
            * (self.fire_emissivity * self.metal_absorptivity * (self.fire_temp_C + 273.0) ** 4
               - self.metal_emissivity * (surface_temp_C + 273.0) ** 4)
            + self.h_conv * (self.gas_temp_C - surface_temp_C)
        ) / 1000.0


def default_fires() -> List[Fire]:
    """The 3 standard fires from the reference workbook."""
    return [
        Fire("Large jet fire 350 kW/m2", 1155, 1155, 100),
        Fire("Pool fire 250 kW/m2", 1125, 1125, 30),
        Fire("Small jet fire 250 kW/m2", 1000, 1000, 100),
    ]


# ---------------------------------------------------------------------------
# Pipe / segment definitions.
# ---------------------------------------------------------------------------
@dataclass
class Pipe:
    """A pipe (or vessel wall) to be evaluated."""
    name: str
    od_mm: float                       # outer diameter [mm]
    wall_mm: float                     # nominal wall thickness [mm]
    material: str = "22Cr duplex"
    corrosion_allowance_mm: float = 0.0
    wall_tolerance_frac: float = 0.125  # fraction (e.g. 0.125 = 12.5 %)
    weight_stress_MPa: float = 0.0      # external/axial "weight" stress
    fluid_density: float = 23.75        # kg/m3 (>500 => treated as liquid)
    fluid_heat_capacity: float = 2283.0  # J/kgK
    gas_mw: float = 18.2                # g/mol (only used for gas release)
    nominal_inch: Optional[float] = None  # nominal diameter [inch] for release scaling

    # ----- derived geometry (per the workbook InOut sheet) -----
    @property
    def od_m(self) -> float:
        return self.od_mm / 1000.0

    @property
    def wall_used_m(self) -> float:
        """Effective wall after tolerance and corrosion (m)."""
        return (self.wall_mm / 1000.0) * (1.0 - self.wall_tolerance_frac) \
            - self.corrosion_allowance_mm / 1000.0

    @property
    def id_used_m(self) -> float:
        return self.od_m - 2.0 * self.wall_used_m

    @property
    def outer_area_per_m(self) -> float:
        return math.pi * self.od_m  # m2 per m length

    @property
    def volume_per_m(self) -> float:
        return math.pi / 4.0 * self.id_used_m ** 2  # m3 per m length

    @property
    def wall_mass_per_m(self) -> float:
        return math.pi * ((self.od_m + self.id_used_m) / 2.0) \
            * self.wall_used_m * STEEL_DENSITY  # kg per m length

    @property
    def is_liquid(self) -> bool:
        return self.fluid_density > 500.0

    @property
    def effective_nominal_inch(self) -> float:
        if self.nominal_inch is not None:
            return self.nominal_inch
        return self.od_mm / 25.4  # approximate nominal size


# ---------------------------------------------------------------------------
# Results.
# ---------------------------------------------------------------------------
@dataclass
class FireResult:
    """Result of one pipe exposed to one fire."""
    fire_name: str
    ruptured: bool
    time_to_rupture_min: Optional[float]
    rupture_pressure_barg: Optional[float]
    # transient histories (numpy arrays, one entry per time step)
    time_min: np.ndarray = field(default_factory=lambda: np.array([]))
    pressure_barg: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_temp_C: np.ndarray = field(default_factory=lambda: np.array([]))
    surface_temp_C: np.ndarray = field(default_factory=lambda: np.array([]))
    strain: np.ndarray = field(default_factory=lambda: np.array([]))
    strain_limit: np.ndarray = field(default_factory=lambda: np.array([]))
    von_mises_MPa: np.ndarray = field(default_factory=lambda: np.array([]))
    # release rates (kg/s) evaluated at rupture pressure
    release_gas_2sides: Optional[float] = None
    release_gas_1side: Optional[float] = None
    release_gas_short: Optional[float] = None
    release_liquid: Optional[float] = None


@dataclass
class PipeResult:
    """Result of one pipe across all fires."""
    pipe: Pipe
    fires: List[FireResult]


# ---------------------------------------------------------------------------
# Core helpers.
# ---------------------------------------------------------------------------
def _interp(temp: float, grid: Sequence[float], values: Sequence[float]) -> float:
    """Linear interpolation of a tabulated property at ``temp`` (deg C)."""
    return float(np.interp(temp, grid, values))


def _t_corr(temp_C: float, uts_MPa: float, mat: dict) -> float:
    """Temperature correction so the creep law reproduces the UTS curve.

    Above 900 deg C the correction is unity (raw activation energy is used).
    """
    if temp_C >= 900.0:
        return 1.0
    sinh_arg = mat["alpha"] * uts_MPa
    # Guard against non-positive sinh argument near the top of the UTS curve.
    sh = math.sinh(sinh_arg)
    if sh <= 0:
        return 1.0
    return ((temp_C + 273.0) / mat["QR"]) * (
        math.log(sh) * mat["n"] - math.log(mat["ref_rate"]) + math.log(mat["A"])
    )


def _strain_rate(temp_C: float, von_mises_MPa: float, strain_effect: float,
                 t_corr: float, mat: dict) -> float:
    """Creep strain rate (per minute) from the Garofalo / Sellars-Tegart law."""
    return (
        strain_effect
        * mat["A"]
        * math.sinh(mat["alpha"] * von_mises_MPa) ** mat["n"]
        * math.exp(-t_corr * mat["QR"] / (temp_C + 273.0))
    )


def _lame_stresses(pressure_barg: float, od_strained: float, id_strained: float
                   ) -> Tuple[float, float, float]:
    """Thick-wall radial, hoop and longitudinal stress [MPa].

    Uses outer/inner *diameters* (the ratio is dimensionless). The 0.1 factor
    converts bar to MPa.
    """
    R = od_strained
    Ti = id_strained
    if R ** 2 - Ti ** 2 <= 0:
        return 0.0, 0.0, 0.0
    mean_d = (R + Ti) / 2.0
    term = Ti ** 2 * R ** 2 / mean_d ** 2
    sr = 0.1 * pressure_barg * (Ti ** 2 - term) / (R ** 2 - Ti ** 2)
    sh = 0.1 * pressure_barg * (Ti ** 2 + term) / (R ** 2 - Ti ** 2)
    sl = 0.1 * pressure_barg * Ti ** 2 / (R ** 2 - Ti ** 2)
    return sr, sh, sl


def _von_mises(sr: float, sh: float, sl: float, weight_stress: float) -> float:
    sla = sl + weight_stress
    return math.sqrt(
        sr ** 2 + sh ** 2 + sla ** 2 - sr * sh - sr * sla - sh * sla
    )


# ---------------------------------------------------------------------------
# Transient solver.
# ---------------------------------------------------------------------------
def simulate_pipe_fire(pipe: Pipe, fire: Fire,
                       profile_time_min: Sequence[float],
                       profile_pressure_bara: Sequence[float],
                       time_step_s: float = 5.0,
                       initial_temp_C: float = 20.0,
                       max_time_min: Optional[float] = None,
                       strain_cap: float = 2.0) -> FireResult:
    """Integrate the wall heat-up / strain accumulation for one pipe & fire.

    Parameters mirror the reference workbook: the pressure profile is supplied as
    absolute pressure (bara) versus time (min); internally the gauge pressure
    (barg = bara - 1) drives the stress, matching the spreadsheet.
    """
    mat = MATERIALS[resolve_material(pipe.material)]
    prof_t = np.asarray(profile_time_min, dtype=float)
    prof_p = np.asarray(profile_pressure_bara, dtype=float)

    if max_time_min is None:
        max_time_min = float(prof_t.max())
    n_steps = int(round(max_time_min * 60.0 / time_step_s))

    od_m = pipe.od_m
    wall_used = pipe.wall_used_m
    outer_area = pipe.outer_area_per_m
    volume = pipe.volume_per_m
    wall_mass = pipe.wall_mass_per_m
    weight_stress = pipe.weight_stress_MPa

    def pressure_barg(t_min: float) -> float:
        # Step look-up matching the workbook VLOOKUP(..., TRUE): take the value
        # of the largest profile time <= t_min; subtract 1 to convert bara->barg.
        idx = int(np.searchsorted(prof_t, t_min, side='right')) - 1
        idx = max(0, min(idx, len(prof_p) - 1))
        return float(prof_p[idx]) - 1.0

    # --- initial state (t = 0) ---
    t_mean = initial_temp_C
    eps = 0.0
    od_strained = od_m * (1.0 + eps)
    wall_strained = wall_used / (1.0 + eps)
    id_strained = od_strained - 2.0 * wall_strained

    q_abs = fire.absorbed_flux(t_mean)          # surface == mean initially
    k_val = _interp(t_mean, mat["temp"], mat["k"])
    cp_val = _interp(t_mean, mat["temp"], mat["cp"])

    times, pres, tmean_h, tsurf_h = [0.0], [pressure_barg(0.0)], [t_mean], [t_mean]
    strain_h, limit_h, vm_h = [eps], [_interp(t_mean, mat["temp"], mat["strain_limit"])], [0.0]

    ruptured = False
    t_rupture = None
    p_rupture = None

    for i in range(1, n_steps + 1):
        t_sec = i * time_step_s
        t_min = t_sec / 60.0
        p_barg = pressure_barg(t_min)

        # --- lumped heat balance (uses previous-step flux / Cp) ---
        if pipe.is_liquid:
            fluid_heat = pipe.fluid_density * volume * pipe.fluid_heat_capacity
        else:
            gas_density = p_barg * 1e5 * pipe.gas_mw / (GAS_CONSTANT * (t_mean + 273.0))
            fluid_heat = gas_density * volume * pipe.fluid_heat_capacity
        denom = wall_mass * cp_val + fluid_heat
        t_mean_new = t_mean + (q_abs * 1000.0 * outer_area * time_step_s) / denom

        # surface temperature with through-wall gradient (previous-step q/k/wall)
        t_surf = t_mean_new + 0.5 * q_abs * 1000.0 * wall_strained / k_val

        # --- update fluxes & properties at the new mean temperature ---
        q_abs_new = fire.absorbed_flux(t_surf)
        k_new = _interp(t_mean_new, mat["temp"], mat["k"])
        cp_new = _interp(t_mean_new, mat["temp"], mat["cp"])
        uts_new = _interp(t_mean_new, mat["temp"], mat["uts"])
        se_new = _interp(t_mean_new, mat["temp"], mat["strain_effect"])
        limit_new = _interp(t_mean_new, mat["temp"], mat["strain_limit"])

        # --- stress (uses previous strained geometry, current pressure) ---
        sr, sh, sl = _lame_stresses(p_barg, od_strained, id_strained)
        vm = _von_mises(sr, sh, sl, weight_stress)

        # --- strain accumulation ---
        tc = _t_corr(t_mean_new, uts_new, mat)
        deps = _strain_rate(t_mean_new, vm, se_new, tc, mat)
        eps_new = min(eps + time_step_s * deps / 60.0, strain_cap)

        # --- update geometry for next step ---
        od_strained = od_m * (1.0 + eps_new)
        wall_strained = wall_used / (1.0 + eps_new)
        id_strained = od_strained - 2.0 * wall_strained

        # --- record ---
        times.append(t_min)
        pres.append(p_barg)
        tmean_h.append(t_mean_new)
        tsurf_h.append(t_surf)
        strain_h.append(eps_new)
        limit_h.append(limit_new)
        vm_h.append(vm)

        # --- rupture check ---
        # The workbook flags rupture on the first step where the accumulated
        # strain exceeds the temperature dependent strain limit.
        if not ruptured and eps_new > limit_new:
            ruptured = True
            t_rupture = t_min
            p_rupture = p_barg

        # advance
        t_mean = t_mean_new
        eps = eps_new
        q_abs = q_abs_new
        k_val = k_new
        cp_val = cp_new

        if ruptured:
            break

    result = FireResult(
        fire_name=fire.name,
        ruptured=ruptured,
        time_to_rupture_min=t_rupture,
        rupture_pressure_barg=p_rupture,
        time_min=np.asarray(times),
        pressure_barg=np.asarray(pres),
        mean_temp_C=np.asarray(tmean_h),
        surface_temp_C=np.asarray(tsurf_h),
        strain=np.asarray(strain_h),
        strain_limit=np.asarray(limit_h),
        von_mises_MPa=np.asarray(vm_h),
    )

    if ruptured:
        _add_release_rates(result, pipe, p_rupture, t_rupture)

    return result


# ---------------------------------------------------------------------------
# Release-rate correlations (evaluated at the rupture pressure).
# ---------------------------------------------------------------------------
def _gas_temp_at_rupture(time_to_rupture_min: float) -> float:
    """Estimated gas temperature at rupture (deg C) as a function of rupture time."""
    if time_to_rupture_min < 3.0:
        return 100.0
    if time_to_rupture_min < 6.0:
        return 150.0
    return 200.0


def _release_scale1(nominal_inch: float, id_m: float) -> float:
    if nominal_inch < 11.0:
        return 1.0 / (21.923 * id_m ** 2 - 9.1442 * id_m + 1.8326)
    if nominal_inch < 21.0:
        return 1.11
    return 1.05


def _release_scale2(nominal_inch: float, id_m: float) -> float:
    if nominal_inch < 24.0:
        return -10.519 * id_m ** 3 + 13.598 * id_m ** 2 - 5.9408 * id_m + 1.9408
    return 1.0


def _add_release_rates(result: FireResult, pipe: Pipe,
                       p_rupture_barg: float, time_to_rupture_min: float,
                       k1: float = 1.3, z: float = 1.0, cd: float = 1.0) -> None:
    """Populate gas / liquid release rates on ``result`` (kg/s)."""
    # cross-sectional area uses the *nominal* inner diameter
    id_nom_m = (pipe.od_mm - 2.0 * pipe.wall_mm) / 1000.0
    cross_area = math.pi / 4.0 * id_nom_m ** 2  # m2
    nominal_inch = pipe.effective_nominal_inch

    if pipe.is_liquid:
        sg = pipe.fluid_density / 1000.0
        # kg/s from a liquid orifice (one-sided release pushed by gas reservoir)
        result.release_liquid = (
            (cross_area * 1e6) * 0.62 / 11.78
            / math.sqrt(sg / (p_rupture_barg * 100.0))
        ) / 60.0 * sg
        return

    # --- choked gas release ---
    t_gas = _gas_temp_at_rupture(time_to_rupture_min)
    a_coef = pipe.gas_mw / (z * GAS_CONSTANT * (273.0 + t_gas))
    b = 2.0 / (k1 + 1.0)
    c = (k1 + 1.0) / (k1 - 1.0)
    d = b ** c * a_coef
    e = d * k1
    f = math.sqrt(e)

    scale1 = _release_scale1(nominal_inch, id_nom_m)
    scale2 = _release_scale2(nominal_inch, id_nom_m)

    flow_2sides = cd * cross_area * p_rupture_barg * 1e5 * f * scale1
    result.release_gas_2sides = flow_2sides
    result.release_gas_1side = flow_2sides / 2.0
    result.release_gas_short = (flow_2sides / 2.0) * scale2


# ---------------------------------------------------------------------------
# Convenience driver.
# ---------------------------------------------------------------------------
def evaluate_pipe(pipe: Pipe, fires: Sequence[Fire],
                  profile_time_min: Sequence[float],
                  profile_pressure_bara: Sequence[float],
                  time_step_s: float = 5.0,
                  initial_temp_C: float = 20.0,
                  max_time_min: Optional[float] = None) -> PipeResult:
    """Evaluate a single pipe against all supplied fires."""
    results = [
        simulate_pipe_fire(
            pipe, fire, profile_time_min, profile_pressure_bara,
            time_step_s=time_step_s, initial_temp_C=initial_temp_C,
            max_time_min=max_time_min,
        )
        for fire in fires
    ]
    return PipeResult(pipe=pipe, fires=results)


# ---------------------------------------------------------------------------
# Default data taken from the reference workbook (for pre-populating the UI).
# ---------------------------------------------------------------------------
DEFAULT_PRESSURE_PROFILE: List[Tuple[float, float]] = [
    (0.0, 61.3), (0.08333, 59.7053), (0.16667, 58.6349), (0.25, 57.3483), (0.33333, 55.9731),
    (0.41667, 54.5583), (0.5, 53.1249), (0.58333, 51.6842), (0.66667, 50.2472), (0.75, 48.8145),
    (0.83333, 47.3977), (0.91667, 46.004), (1.0, 44.6383), (1.08333, 43.3046), (1.16667, 42.0058),
    (1.25, 40.7442), (1.33333, 39.5206), (1.41667, 38.3367), (1.5, 37.1941), (1.58333, 36.0922),
    (1.66667, 35.0302), (1.75, 34.0082), (1.83333, 33.0255), (1.91667, 32.0817), (2.0, 31.1758),
    (2.08333, 30.307), (2.16667, 29.4743), (2.25, 28.6765), (2.33333, 27.9126), (2.41667, 27.1814),
    (2.5, 26.4814), (2.58333, 25.8119), (2.66667, 25.1716), (2.75, 24.5595), (2.83333, 23.9742),
    (2.91667, 23.4147), (3.0, 22.8799), (3.08333, 22.3683), (3.16667, 21.8793), (3.25, 21.4119),
    (3.33333, 20.9651), (3.41667, 20.5382), (3.5, 20.13), (3.58333, 19.7396), (3.66667, 19.3656),
    (3.75, 19.0076), (3.83333, 18.6653), (3.91667, 18.3374), (4.0, 18.0237), (4.08333, 17.7234),
    (4.16667, 17.4359), (4.25, 17.1605), (4.33333, 16.8967), (4.41667, 16.644), (4.5, 16.4017),
    (4.58333, 16.1696), (4.66667, 15.9467), (4.75, 15.7329), (4.83333, 15.5277), (4.91667, 15.3307),
    (5.0, 15.1415), (5.16667, 14.7853), (5.33333, 14.4567), (5.5, 14.1527), (5.66667, 13.8718),
    (5.83333, 13.6117), (6.0, 13.3695), (6.16667, 13.1436), (6.33333, 12.9327), (6.5, 12.7352),
    (6.66667, 12.5502), (6.83333, 12.3776), (7.0, 12.2163), (7.16667, 12.0655), (7.33333, 11.9243),
    (7.5, 11.7922), (7.66667, 11.669), (7.83333, 11.5528), (8.0, 11.4431), (8.33333, 11.2416),
    (8.66667, 11.0612), (9.0, 10.8993), (9.33333, 10.753), (9.66667, 10.6201), (10.0, 10.4985),
    (10.33333, 10.3878), (10.66667, 10.2862), (11.0, 10.1923), (11.33333, 10.1051), (11.66667, 10.022),
    (12.0, 9.9422), (12.33333, 9.8651), (12.66667, 9.7896), (13.0, 9.7151), (13.33333, 9.6406),
    (13.66667, 9.5665), (14.0, 9.4922), (14.33333, 9.4168), (14.66667, 9.3394), (15.0, 9.2597),
]

# A subset of the embedded pipe-class database (class, nominal inch, OD mm, wall mm).
DEFAULT_PIPE_DATABASE: List[dict] = [
    {"PipeClass": "DD100", "NominalDiameter[inch]": 1, "OD[mm]": 33.4, "Wall[mm]": 3.38},
    {"PipeClass": "DD100", "NominalDiameter[inch]": 1.5, "OD[mm]": 48.3, "Wall[mm]": 3.68},
    {"PipeClass": "DD100", "NominalDiameter[inch]": 2, "OD[mm]": 60.3, "Wall[mm]": 2.77},
    {"PipeClass": "GD200X", "NominalDiameter[inch]": 2, "OD[mm]": 60.3, "Wall[mm]": 5.54},
    {"PipeClass": "DD100", "NominalDiameter[inch]": 3, "OD[mm]": 88.9, "Wall[mm]": 3.05},
    {"PipeClass": "DD100", "NominalDiameter[inch]": 4, "OD[mm]": 114.3, "Wall[mm]": 6.02},
    {"PipeClass": "DD100", "NominalDiameter[inch]": 6, "OD[mm]": 168.3, "Wall[mm]": 7.11},
    {"PipeClass": "DD100", "NominalDiameter[inch]": 8, "OD[mm]": 219.1, "Wall[mm]": 6.35},
    {"PipeClass": "DD100", "NominalDiameter[inch]": 10, "OD[mm]": 273.1, "Wall[mm]": 7.8},
    {"PipeClass": "DD100", "NominalDiameter[inch]": 12, "OD[mm]": 323.9, "Wall[mm]": 9.53},
    {"PipeClass": "DD100", "NominalDiameter[inch]": 14, "OD[mm]": 355.6, "Wall[mm]": 11.13},
    {"PipeClass": "DD100", "NominalDiameter[inch]": 18, "OD[mm]": 457.2, "Wall[mm]": 12.7},
    {"PipeClass": "DD100", "NominalDiameter[inch]": 24, "OD[mm]": 610.0, "Wall[mm]": 17.48},
]

# The 6 example pipes pre-loaded in the reference workbook (InOut F8:T13).
DEFAULT_PIPES: List[dict] = [
    {"Name": "Pipe 1", "NominalDiameter[inch]": 3, "OD[mm]": 88.9, "Wall[mm]": 3.7},
    {"Name": "Pipe 2", "NominalDiameter[inch]": 8, "OD[mm]": 219.1, "Wall[mm]": 6.35},
    {"Name": "Pipe 3", "NominalDiameter[inch]": 10, "OD[mm]": 273.1, "Wall[mm]": 7.8},
    {"Name": "Pipe 4", "NominalDiameter[inch]": 12, "OD[mm]": 323.9, "Wall[mm]": 9.53},
    {"Name": "Pipe 5", "NominalDiameter[inch]": 14, "OD[mm]": 355.6, "Wall[mm]": 11.13},
    {"Name": "Pipe 6", "NominalDiameter[inch]": 18, "OD[mm]": 457.2, "Wall[mm]": 12.7},
]
