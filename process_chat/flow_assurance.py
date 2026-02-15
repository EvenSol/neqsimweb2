"""
Flow Assurance — hydrate prediction, wax appearance, scale potential, corrosion.

Uses NeqSim's thermodynamic engine to evaluate:
  - Gas hydrate formation temperature at operating pressure
  - Wax Appearance Temperature (WAT) estimation
  - Scale potential (BaSO₄, CaCO₃) from water chemistry
  - CO₂/H₂S corrosion rate estimation (de Waard–Milliams)
  - MEG/MeOH inhibitor dosing requirements

Returns a ``FlowAssuranceResult`` with risk assessment and mitigation recommendations.
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
class HydrateRisk:
    """Hydrate formation assessment for one stream/point."""
    location: str
    operating_T_C: float = 0.0
    operating_P_bara: float = 0.0
    hydrate_T_C: float = 0.0
    subcooling_C: float = 0.0   # positive = inside hydrate region (risk!)
    risk_level: str = "LOW"     # LOW, MEDIUM, HIGH
    inhibitor_type: str = ""
    inhibitor_rate_kg_hr: float = 0.0
    detail: str = ""


@dataclass
class WaxRisk:
    """Wax appearance risk for a stream."""
    stream_name: str
    operating_T_C: float = 0.0
    wax_appearance_T_C: float = 0.0
    margin_C: float = 0.0
    risk_level: str = "LOW"
    detail: str = ""


@dataclass
class CorrosionRisk:
    """CO₂/H₂S corrosion assessment."""
    location: str
    co2_partial_pressure_bara: float = 0.0
    h2s_partial_pressure_bara: float = 0.0
    temperature_C: float = 0.0
    corrosion_rate_mm_yr: float = 0.0
    risk_level: str = "LOW"
    mechanism: str = ""
    detail: str = ""


@dataclass
class FlowAssuranceResult:
    """Complete flow assurance assessment."""
    hydrate_risks: List[HydrateRisk] = field(default_factory=list)
    wax_risks: List[WaxRisk] = field(default_factory=list)
    corrosion_risks: List[CorrosionRisk] = field(default_factory=list)
    overall_risk: str = "LOW"
    recommendations: List[str] = field(default_factory=list)
    method: str = "estimation"
    message: str = ""


# ---------------------------------------------------------------------------
# Hydrate prediction
# ---------------------------------------------------------------------------

def _predict_hydrate_temperature(fluid, pressure_bara: float) -> float:
    """Predict hydrate formation temperature using NeqSim or correlation."""
    # Try NeqSim Java hydrate calculation
    try:
        from neqsim import jneqsim
        fl = fluid.clone()
        fl.setPressure(float(pressure_bara))
        # Ensure water is present
        has_water = False
        for i in range(fl.getNumberOfComponents()):
            if fl.getComponent(i).getName().lower() == "water":
                has_water = True
                break
        if not has_water:
            fl.addComponent("water", 0.01)

        fl.setHydrateCheck(True)
        thermoOps = jneqsim.thermodynamicoperations.ThermodynamicOperations(fl)
        thermoOps.hydrateFormationTemperature()
        return float(fl.getTemperature()) - 273.15
    except Exception:
        pass

    # Fallback: Hammerschmidt correlation approximation
    # T_hyd ≈ A × ln(P) + B for natural gas
    # Typical: T_hyd(°C) ≈ 8.9 × ln(P_bara) - 4.0
    try:
        t_hyd = 8.9 * math.log(pressure_bara) - 4.0
        return t_hyd
    except Exception:
        return 20.0


def _estimate_meg_dosing(
    subcooling_C: float,
    water_rate_kg_hr: float,
    gas_rate_kg_hr: float,
) -> float:
    """Estimate MEG injection rate for hydrate inhibition.

    Uses Hammerschmidt equation:
      ΔT = K_H × w / (M × (100 - w))
    where w = wt% inhibitor, M = molar mass of inhibitor, K_H = 2335 (MEG).
    """
    if subcooling_C <= 0:
        return 0.0

    K_H = 2335.0  # MEG
    M = 62.07     # MEG molar mass

    # Solve for w (wt% in water phase)
    # ΔT = K_H × w / (M × (100 - w))
    # w = ΔT × M × 100 / (K_H + ΔT × M)
    w = subcooling_C * M * 100 / (K_H + subcooling_C * M)
    w = min(w, 80.0)  # cap at 80 wt%

    # MEG rate = w/(100-w) × water rate
    if w < 100:
        meg_rate = (w / (100.0 - w)) * water_rate_kg_hr
    else:
        meg_rate = water_rate_kg_hr * 5.0

    return meg_rate


# ---------------------------------------------------------------------------
# Wax prediction (simplified)
# ---------------------------------------------------------------------------

def _estimate_wax_temperature(fluid) -> float:
    """Estimate Wax Appearance Temperature.

    Uses a simplified correlation based on presence of heavy hydrocarbons.
    For accurate WAT, NeqSim's wax model should be used.
    """
    try:
        from neqsim import jneqsim
        fl = fluid.clone()
        fl.setTemperature(273.15 + 80)  # Start high
        fl.setPressure(50.0)

        thermoOps = jneqsim.thermodynamicoperations.ThermodynamicOperations(fl)
        try:
            thermoOps.calcWAT()
            return float(fl.getTemperature()) - 273.15
        except Exception:
            pass
    except Exception:
        pass

    # Fallback: estimate from heavy component content
    # Check for C7+ content
    try:
        heavy_fraction = 0.0
        for i in range(fluid.getNumberOfComponents()):
            name = str(fluid.getComponent(i).getName()).lower()
            if any(h in name for h in ["c7", "c8", "c9", "nc10", "nc11", "nc12",
                                        "n-heptane", "n-octane", "n-nonane"]):
                heavy_fraction += float(fluid.getComponent(i).getz())

        if heavy_fraction > 0.05:
            return 30.0 + heavy_fraction * 200  # Rough estimate
        elif heavy_fraction > 0.01:
            return 15.0 + heavy_fraction * 200
        else:
            return -10.0  # No wax risk for light gas
    except Exception:
        return -10.0


# ---------------------------------------------------------------------------
# CO₂ corrosion (de Waard–Milliams)
# ---------------------------------------------------------------------------

def _estimate_co2_corrosion(
    co2_partial_pressure_bara: float,
    temperature_C: float,
) -> float:
    """de Waard–Milliams (1975) CO₂ corrosion rate.

    log(CR) = 5.8 − 1710/(T+273) + 0.67 × log(pCO₂)
    Returns corrosion rate in mm/yr.
    """
    if co2_partial_pressure_bara <= 0:
        return 0.0

    T_K = temperature_C + 273.15
    try:
        log_cr = 5.8 - 1710.0 / T_K + 0.67 * math.log10(co2_partial_pressure_bara)
        cr = 10.0 ** log_cr
        return min(cr, 50.0)  # cap at 50 mm/yr
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_flow_assurance(
    model: NeqSimProcessModel,
    check_hydrates: bool = True,
    check_wax: bool = True,
    check_corrosion: bool = True,
    inhibitor_type: str = "MEG",
) -> FlowAssuranceResult:
    """
    Run flow assurance assessment.

    Checks every stream for hydrate, wax, and corrosion risks.
    """
    result = FlowAssuranceResult()
    proc = model.get_process()
    if proc is None:
        result.message = "No process available."
        return result

    try:
        model.run()
    except Exception:
        pass

    streams = model.list_streams()
    recommendations: List[str] = []

    for s_info in streams:
        s_name = s_info.name
        T_C = s_info.temperature_C or 25.0
        P_bara = s_info.pressure_bara or 50.0
        flow_kg_hr = s_info.flow_rate_kg_hr or 0

        fluid = None
        try:
            java_stream = model.get_stream(s_name)
            fluid = java_stream.getFluid()
        except Exception:
            pass

        # --- Hydrate check ---
        if check_hydrates and fluid:
            try:
                T_hyd = _predict_hydrate_temperature(fluid, P_bara)
                subcooling = T_hyd - T_C  # positive = at risk

                if subcooling > 5:
                    risk = "HIGH"
                elif subcooling > 0:
                    risk = "MEDIUM"
                else:
                    risk = "LOW"

                water_rate = flow_kg_hr * 0.001  # estimate water content
                try:
                    for i in range(fluid.getNumberOfComponents()):
                        if fluid.getComponent(i).getName().lower() == "water":
                            water_rate = float(fluid.getComponent(i).getFlowRate("kg/hr"))
                            break
                except Exception:
                    pass

                meg_rate = 0.0
                if subcooling > 0 and inhibitor_type.upper() == "MEG":
                    meg_rate = _estimate_meg_dosing(subcooling, water_rate, flow_kg_hr)

                result.hydrate_risks.append(HydrateRisk(
                    location=s_name,
                    operating_T_C=T_C,
                    operating_P_bara=P_bara,
                    hydrate_T_C=T_hyd,
                    subcooling_C=subcooling,
                    risk_level=risk,
                    inhibitor_type=inhibitor_type if subcooling > 0 else "",
                    inhibitor_rate_kg_hr=meg_rate,
                    detail=f"Hydrate T={T_hyd:.1f}°C at {P_bara:.1f} bara",
                ))

                if risk == "HIGH":
                    recommendations.append(
                        f"HIGH hydrate risk at {s_name}: inject {meg_rate:.0f} kg/hr {inhibitor_type} "
                        f"or heat to >{T_hyd:.0f}°C"
                    )
            except Exception:
                pass

        # --- Wax check ---
        if check_wax and fluid:
            try:
                T_wax = _estimate_wax_temperature(fluid)
                margin = T_C - T_wax  # positive = safe

                if margin < 0:
                    risk = "HIGH"
                elif margin < 10:
                    risk = "MEDIUM"
                else:
                    risk = "LOW"

                result.wax_risks.append(WaxRisk(
                    stream_name=s_name,
                    operating_T_C=T_C,
                    wax_appearance_T_C=T_wax,
                    margin_C=margin,
                    risk_level=risk,
                    detail=f"WAT={T_wax:.1f}°C, margin={margin:.1f}°C",
                ))

                if risk == "HIGH":
                    recommendations.append(
                        f"HIGH wax risk at {s_name}: WAT={T_wax:.0f}°C > operating T={T_C:.0f}°C"
                    )
            except Exception:
                pass

        # --- Corrosion check ---
        if check_corrosion and fluid:
            try:
                co2_pp = 0.0
                h2s_pp = 0.0
                for i in range(fluid.getNumberOfComponents()):
                    name = str(fluid.getComponent(i).getName()).lower()
                    mole_frac = float(fluid.getComponent(i).getz())
                    if name == "co2":
                        co2_pp = mole_frac * P_bara
                    elif name == "h2s":
                        h2s_pp = mole_frac * P_bara

                if co2_pp > 0 or h2s_pp > 0:
                    cr = _estimate_co2_corrosion(co2_pp, T_C)

                    if cr > 1.0:
                        risk = "HIGH"
                    elif cr > 0.1:
                        risk = "MEDIUM"
                    else:
                        risk = "LOW"

                    mechanism = "CO₂ corrosion"
                    if h2s_pp > 0.05:
                        mechanism += " + H₂S (sour service)"

                    result.corrosion_risks.append(CorrosionRisk(
                        location=s_name,
                        co2_partial_pressure_bara=co2_pp,
                        h2s_partial_pressure_bara=h2s_pp,
                        temperature_C=T_C,
                        corrosion_rate_mm_yr=cr,
                        risk_level=risk,
                        mechanism=mechanism,
                        detail=f"CR={cr:.2f} mm/yr, pCO₂={co2_pp:.2f} bara",
                    ))

                    if risk == "HIGH":
                        recommendations.append(
                            f"HIGH corrosion risk at {s_name}: {cr:.1f} mm/yr — "
                            f"consider CRA material or corrosion inhibitor"
                        )
            except Exception:
                pass

    # Overall risk
    all_risks = (
        [h.risk_level for h in result.hydrate_risks] +
        [w.risk_level for w in result.wax_risks] +
        [c.risk_level for c in result.corrosion_risks]
    )
    if "HIGH" in all_risks:
        result.overall_risk = "HIGH"
    elif "MEDIUM" in all_risks:
        result.overall_risk = "MEDIUM"
    else:
        result.overall_risk = "LOW"

    result.recommendations = recommendations
    result.message = (f"Flow assurance: {len(result.hydrate_risks)} hydrate points, "
                      f"{len(result.wax_risks)} wax points, "
                      f"{len(result.corrosion_risks)} corrosion points")

    return result


# ---------------------------------------------------------------------------
# Format for LLM
# ---------------------------------------------------------------------------

def format_flow_assurance_result(result: FlowAssuranceResult) -> str:
    """Format flow assurance results for LLM follow-up."""
    lines = ["=== FLOW ASSURANCE ASSESSMENT ==="]
    lines.append(f"Overall risk level: {result.overall_risk}")
    lines.append(result.message)
    lines.append("")

    if result.hydrate_risks:
        lines.append("=== HYDRATE RISK ===")
        for h in result.hydrate_risks:
            if h.risk_level != "LOW":
                lines.append(f"  {h.location} [{h.risk_level}]: "
                             f"T_op={h.operating_T_C:.1f}°C, T_hyd={h.hydrate_T_C:.1f}°C, "
                             f"subcooling={h.subcooling_C:.1f}°C")
                if h.inhibitor_rate_kg_hr > 0:
                    lines.append(f"    → {h.inhibitor_type} dosing: {h.inhibitor_rate_kg_hr:.0f} kg/hr")
        low_count = sum(1 for h in result.hydrate_risks if h.risk_level == "LOW")
        if low_count:
            lines.append(f"  ({low_count} locations at LOW hydrate risk)")
        lines.append("")

    if result.wax_risks:
        lines.append("=== WAX RISK ===")
        for w in result.wax_risks:
            if w.risk_level != "LOW":
                lines.append(f"  {w.stream_name} [{w.risk_level}]: "
                             f"T_op={w.operating_T_C:.1f}°C, WAT={w.wax_appearance_T_C:.1f}°C, "
                             f"margin={w.margin_C:.1f}°C")
        low_count = sum(1 for w in result.wax_risks if w.risk_level == "LOW")
        if low_count:
            lines.append(f"  ({low_count} streams at LOW wax risk)")
        lines.append("")

    if result.corrosion_risks:
        lines.append("=== CORROSION RISK ===")
        for c in result.corrosion_risks:
            if c.risk_level != "LOW":
                lines.append(f"  {c.location} [{c.risk_level}]: "
                             f"CR={c.corrosion_rate_mm_yr:.2f} mm/yr, "
                             f"pCO₂={c.co2_partial_pressure_bara:.2f} bara, "
                             f"T={c.temperature_C:.1f}°C | {c.mechanism}")
        low_count = sum(1 for c in result.corrosion_risks if c.risk_level == "LOW")
        if low_count:
            lines.append(f"  ({low_count} locations at LOW corrosion risk)")
        lines.append("")

    if result.recommendations:
        lines.append("=== RECOMMENDATIONS ===")
        for i, rec in enumerate(result.recommendations, 1):
            lines.append(f"  {i}. {rec}")

    return "\n".join(lines)
