"""
Safety Systems — PSV sizing, relief valve analysis, blowdown valve sizing.

Calculates:
  - Required relief rate for fire / blocked outlet / thermal expansion scenarios
  - PSV orifice area (API 520/521)
  - Blowdown valve Cv requirements
  - Safety system summary report

Uses NeqSim fluid properties for accurate gas/liquid relief calculations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .process_model import NeqSimProcessModel


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ReliefScenario:
    """One relief scenario for a piece of equipment."""
    equipment_name: str
    scenario: str             # FIRE, BLOCKED_OUTLET, THERMAL_EXPANSION, CONTROL_FAILURE
    set_pressure_bara: float = 0.0
    relieving_pressure_bara: float = 0.0
    required_relief_rate_kg_hr: float = 0.0
    required_orifice_area_mm2: float = 0.0
    api_orifice_letter: str = ""
    api_orifice_area_mm2: float = 0.0
    fluid_phase: str = "gas"   # gas, liquid, two_phase
    detail: str = ""


@dataclass
class SafetyReport:
    """Complete safety system analysis."""
    scenarios: List[ReliefScenario] = field(default_factory=list)
    equipment_with_psv: List[str] = field(default_factory=list)
    total_psv_count: int = 0
    max_relief_rate_kg_hr: float = 0.0
    flare_load_kg_hr: float = 0.0
    blowdown_time_s: float = 0.0
    method: str = "API_520_521"
    message: str = ""


# ---------------------------------------------------------------------------
# API 526 standard PSV orifice sizes
# ---------------------------------------------------------------------------

_API_ORIFICES = [
    ("D", 71.0),     # mm²
    ("E", 126.0),
    ("F", 198.0),
    ("G", 325.0),
    ("H", 506.0),
    ("J", 830.0),
    ("K", 1186.0),
    ("L", 1841.0),
    ("M", 2323.0),
    ("N", 2800.0),
    ("P", 4116.0),
    ("Q", 7126.0),
    ("R", 10323.0),
    ("T", 16774.0),
]


def _select_api_orifice(required_mm2: float) -> Tuple[str, float]:
    """Select the standard API orifice that meets the required area."""
    for letter, area in _API_ORIFICES:
        if area >= required_mm2:
            return letter, area
    # Larger than T — return T
    return "T+", required_mm2


# ---------------------------------------------------------------------------
# Relief rate calculations
# ---------------------------------------------------------------------------

def _fire_relief_rate(
    wetted_area_m2: float,
    latent_heat_kJ_kg: float = 200.0,
    insulation_factor: float = 1.0,
    environment_factor: float = 1.0,
) -> float:
    """API 521 fire case heat input and relief rate.

    Q = C₁ × F × A^0.82 (for wetted area in m²)
    Returns relief rate in kg/hr.
    """
    # API 521 Table 4: Q = 43200 × F × A^0.82 (imperial) converted to metric
    # Using SI: Q (W) = 70900 × F × A_w^0.82 (A_w in m²)
    Q_watts = 70900.0 * environment_factor * insulation_factor * (wetted_area_m2 ** 0.82)
    Q_kJ_hr = Q_watts * 3.6  # W to kJ/hr
    if latent_heat_kJ_kg > 0:
        return Q_kJ_hr / latent_heat_kJ_kg
    return 0.0


def _gas_psv_area(
    relief_rate_kg_hr: float,
    set_pressure_bara: float,
    temperature_K: float,
    molar_mass_kg_mol: float = 0.020,
    k_ratio: float = 1.3,
    z_factor: float = 0.9,
    kd: float = 0.975,
    kb: float = 1.0,
    kc: float = 1.0,
) -> float:
    """API 520 gas/vapor PSV sizing. Returns required area in mm²."""
    # W = C × Kd × P1 × Kb × Kc × A × sqrt(M / (T × Z))
    # C = 0.03948 * sqrt(k * (2/(k+1))^((k+1)/(k-1)))
    C = 0.03948 * math.sqrt(k_ratio * (2.0 / (k_ratio + 1)) ** ((k_ratio + 1) / (k_ratio - 1)))

    P1 = set_pressure_bara * 1.10 * 100.0  # 10% accumulation, kPa(a)

    W = relief_rate_kg_hr  # kg/hr

    if C * kd * P1 * kb * kc <= 0:
        return 0.0

    # A in mm² (API 520 SI coefficient system: W in kg/hr, P in kPa(a),
    # T in K, M in kg/kmol → A directly in mm²)
    A = W * math.sqrt(temperature_K * z_factor) / (C * kd * P1 * kb * kc * math.sqrt(molar_mass_kg_mol * 1000))
    return A  # already in mm² per API 520 SI units


def _liquid_psv_area(
    relief_rate_kg_hr: float,
    set_pressure_bara: float,
    back_pressure_bara: float = 1.013,
    density_kg_m3: float = 800.0,
    viscosity_cP: float = 1.0,
    kd: float = 0.65,
) -> float:
    """API 520 liquid PSV sizing. Returns area in mm²."""
    delta_p = (set_pressure_bara * 1.10 - back_pressure_bara) * 100.0  # kPa
    if delta_p <= 0:
        return 0.0

    Q_m3_hr = relief_rate_kg_hr / density_kg_m3
    # A = Q / (Kd × Kw × Kv × √(2×ΔP/ρ))
    # Simplified:
    velocity = math.sqrt(2.0 * delta_p * 1000.0 / density_kg_m3)  # m/s
    if velocity <= 0:
        return 0.0
    A_m2 = (Q_m3_hr / 3600.0) / (kd * velocity)
    return A_m2 * 1e6  # mm²


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_safety_analysis(
    model: NeqSimProcessModel,
    design_pressure_factor: float = 1.1,
    include_fire: bool = True,
    include_blocked_outlet: bool = True,
) -> SafetyReport:
    """
    Run safety system sizing analysis.

    Evaluates each pressure vessel and determines PSV requirements
    per API 520/521.
    """
    report = SafetyReport()
    proc = model.get_process()
    if proc is None:
        report.message = "No process available."
        return report

    try:
        model.run()
    except Exception:
        pass

    units = model.list_units()
    scenarios: List[ReliefScenario] = []
    max_relief = 0.0

    _VESSEL_TYPES = {"Separator", "TwoPhaseSeparator", "ThreePhaseSeparator",
                     "GasScrubber", "GasScrubberSimple", "Tank"}
    _COMPRESSOR_TYPES = {"Compressor"}
    _VALVE_TYPES = {"ThrottlingValve", "ControlValve"}
    _HX_TYPES = {"HeatExchanger", "Cooler", "Heater"}

    for u_info in units:
        u_name = u_info.name
        java_type = u_info.unit_type
        try:
            java_obj = model.get_unit(u_name)
        except KeyError:
            continue

        props = u_info.properties

        # --- Determine operating pressure ---
        operating_p = 0.0
        # Try outlet pressure from properties
        outlet_p = props.get("outletPressure_bara", 0)
        if outlet_p and outlet_p > 0:
            operating_p = outlet_p
        else:
            # Read from outlet stream
            for m in ("getOutletStream", "getOutStream", "getGasOutStream"):
                if hasattr(java_obj, m):
                    try:
                        s = getattr(java_obj, m)()
                        operating_p = float(s.getPressure("bara"))
                        break
                    except Exception:
                        pass
            # Fallback: inlet stream
            if operating_p <= 0:
                for m in ("getInletStream", "getInStream", "getFeedStream"):
                    if hasattr(java_obj, m):
                        try:
                            s = getattr(java_obj, m)()
                            operating_p = float(s.getPressure("bara"))
                            break
                        except Exception:
                            pass
        if operating_p <= 0:
            operating_p = 50.0  # last-resort default

        set_pressure = operating_p * design_pressure_factor

        # --- Get fluid properties from the unit's inlet stream ---
        temp_K = 300.0
        molar_mass = 0.020
        z_factor = 0.9
        density = 50.0
        latent_heat = 200.0
        k_ratio = 1.3
        fluid = None

        try:
            for m_name in ("getInletStream", "getInStream", "getFeedStream"):
                if hasattr(java_obj, m_name):
                    stream = getattr(java_obj, m_name)()
                    if stream:
                        fluid = stream.getFluid()
                        if fluid:
                            try:
                                temp_K = float(fluid.getTemperature("K"))
                            except Exception:
                                pass
                            try:
                                molar_mass = float(fluid.getMolarMass())
                            except Exception:
                                pass
                            try:
                                z_factor = float(fluid.getPhase("gas").getZ())
                            except Exception:
                                try:
                                    z_factor = float(fluid.getPhase(0).getZ())
                                except Exception:
                                    pass
                            try:
                                density = float(fluid.getDensity("kg/m3"))
                            except Exception:
                                pass
                            try:
                                k_ratio = float(fluid.getGamma())
                            except Exception:
                                pass
                            # Estimate latent heat from enthalpy difference
                            try:
                                n_phases = int(fluid.getNumberOfPhases())
                                if n_phases >= 2:
                                    h_gas = float(fluid.getPhase("gas").getEnthalpy("kJ/kg"))
                                    h_liq = float(fluid.getPhase("oil").getEnthalpy("kJ/kg"))
                                    calc_lh = abs(h_gas - h_liq)
                                    if calc_lh > 10:
                                        latent_heat = calc_lh
                            except Exception:
                                pass
                        break
        except Exception:
            pass

        # --- Fire case (vessels, heat exchangers) ---
        if include_fire and java_type in (_VESSEL_TYPES | _HX_TYPES):
            # Estimate wetted area from dimensions
            wetted_area = 20.0  # default m²
            try:
                md = java_obj.getMechanicalDesign() if hasattr(java_obj, "getMechanicalDesign") else None
                if md is not None:
                    try:
                        d = float(md.getInnerDiameter())
                        L = float(md.getTantanLength()) if hasattr(md, "getTantanLength") else d * 3
                        if d > 0 and L > 0:
                            wetted_area = math.pi * d * L * 0.5  # 50% wetted
                    except Exception:
                        pass
                elif hasattr(java_obj, "getInternalDiameter"):
                    d = float(java_obj.getInternalDiameter())
                    if d > 0:
                        L = d * 3
                        wetted_area = math.pi * d * L * 0.5
            except Exception:
                pass

            relief_rate = _fire_relief_rate(wetted_area, latent_heat)
            orifice = _gas_psv_area(relief_rate, set_pressure, temp_K,
                                     molar_mass, k_ratio=k_ratio,
                                     z_factor=z_factor)
            letter, std_area = _select_api_orifice(orifice)

            scenarios.append(ReliefScenario(
                equipment_name=u_name,
                scenario="FIRE",
                set_pressure_bara=set_pressure,
                relieving_pressure_bara=set_pressure * 1.10,
                required_relief_rate_kg_hr=relief_rate,
                required_orifice_area_mm2=orifice,
                api_orifice_letter=letter,
                api_orifice_area_mm2=std_area,
                fluid_phase="gas",
                detail=f"Wetted area={wetted_area:.1f} m², latent heat={latent_heat:.0f} kJ/kg",
            ))
            max_relief = max(max_relief, relief_rate)

        # --- Blocked outlet (compressors, pumps) ---
        if include_blocked_outlet and java_type in _COMPRESSOR_TYPES:
            flow_kg_hr = props.get("flow_kg_hr", 0)
            if not flow_kg_hr:
                try:
                    for m_name in ("getInletStream", "getInStream"):
                        if hasattr(java_obj, m_name):
                            flow_kg_hr = float(getattr(java_obj, m_name)().getFlowRate("kg/hr"))
                            break
                except Exception:
                    flow_kg_hr = 10000.0

            orifice = _gas_psv_area(flow_kg_hr, set_pressure, temp_K,
                                     molar_mass, k_ratio=k_ratio,
                                     z_factor=z_factor)
            letter, std_area = _select_api_orifice(orifice)

            scenarios.append(ReliefScenario(
                equipment_name=u_name,
                scenario="BLOCKED_OUTLET",
                set_pressure_bara=set_pressure,
                relieving_pressure_bara=set_pressure * 1.10,
                required_relief_rate_kg_hr=flow_kg_hr,
                required_orifice_area_mm2=orifice,
                api_orifice_letter=letter,
                api_orifice_area_mm2=std_area,
                fluid_phase="gas",
                detail=f"Full compressor flow at shut-in",
            ))
            max_relief = max(max_relief, flow_kg_hr)

        # --- Thermal expansion (liquid-full heat exchangers / coolers) ---
        if include_fire and java_type in _HX_TYPES:
            # If liquid-side could be blocked while heating continues
            duty_kW = abs(props.get("duty_kW", 0))
            if duty_kW > 0 and latent_heat > 0:
                # Thermal expansion relief: approximate from duty
                thermal_relief = duty_kW * 3600.0 / latent_heat  # kg/hr (conservative)
                if thermal_relief > 0:
                    orifice = _liquid_psv_area(thermal_relief, set_pressure,
                                                density_kg_m3=density)
                    letter, std_area = _select_api_orifice(orifice)
                    scenarios.append(ReliefScenario(
                        equipment_name=u_name,
                        scenario="THERMAL_EXPANSION",
                        set_pressure_bara=set_pressure,
                        relieving_pressure_bara=set_pressure * 1.10,
                        required_relief_rate_kg_hr=thermal_relief,
                        required_orifice_area_mm2=orifice,
                        api_orifice_letter=letter,
                        api_orifice_area_mm2=std_area,
                        fluid_phase="liquid",
                        detail=f"Duty={duty_kW:.0f} kW, liquid blocked in",
                    ))

    # Summary
    equipment_with_psv = sorted(set(s.equipment_name for s in scenarios))
    report.scenarios = scenarios
    report.equipment_with_psv = equipment_with_psv
    report.total_psv_count = len(equipment_with_psv)
    report.max_relief_rate_kg_hr = max_relief
    report.flare_load_kg_hr = max_relief
    report.method = "API_520_521"
    report.message = (
        f"Safety analysis: {len(scenarios)} relief scenarios "
        f"for {len(equipment_with_psv)} equipment items"
    )

    return report


# ---------------------------------------------------------------------------
# Format for LLM
# ---------------------------------------------------------------------------

def format_safety_result(report: SafetyReport) -> str:
    """Format safety analysis for LLM follow-up."""
    lines = ["=== SAFETY SYSTEM ANALYSIS (API 520/521) ==="]
    lines.append(report.message)
    lines.append(f"Total PSV scenarios: {report.total_psv_count}")
    lines.append(f"Equipment requiring PSVs: {', '.join(report.equipment_with_psv)}")
    lines.append(f"Maximum relief rate: {report.max_relief_rate_kg_hr:.0f} kg/hr")
    lines.append(f"Flare system design load: {report.flare_load_kg_hr:.0f} kg/hr")
    lines.append("")

    lines.append("=== RELIEF SCENARIOS ===")
    for s in report.scenarios:
        lines.append(f"  {s.equipment_name} — {s.scenario}:")
        lines.append(f"    Set pressure: {s.set_pressure_bara:.1f} bara")
        lines.append(f"    Relief rate: {s.required_relief_rate_kg_hr:.0f} kg/hr")
        lines.append(f"    Required orifice: {s.required_orifice_area_mm2:.0f} mm²")
        lines.append(f"    API orifice: {s.api_orifice_letter} ({s.api_orifice_area_mm2:.0f} mm²)")
        lines.append(f"    Phase: {s.fluid_phase}")
        lines.append(f"    Detail: {s.detail}")
        lines.append("")

    return "\n".join(lines)
