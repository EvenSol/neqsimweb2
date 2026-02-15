"""
Emissions Calculator — CO₂ / flare / fuel-gas / fugitive emissions for a NeqSim process.

Wraps NeqSim's combustion and emissions utilities where available, with a
Python fallback that estimates emissions from:
  - Fuel gas consumption (gas turbines, fired heaters)
  - Flare releases
  - Fugitive emissions (equipment leak factors)
  - Venting / blowdown

Returns an ``EmissionsResult`` dataclass suitable for UI display (tables, charts).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .process_model import NeqSimProcessModel, KPI


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class EmissionSource:
    """A single emissions source (equipment / flare / fugitive)."""
    name: str
    source_type: str          # FUEL_GAS, FLARE, FUGITIVE, VENTING, POWER
    co2_kg_hr: float = 0.0
    co2e_kg_hr: float = 0.0   # CO₂ equivalent (includes CH₄ GWP)
    ch4_kg_hr: float = 0.0
    nox_kg_hr: float = 0.0
    fuel_rate_kg_hr: float = 0.0
    detail: str = ""


@dataclass
class EmissionsResult:
    """Complete emissions analysis for a process."""
    sources: List[EmissionSource] = field(default_factory=list)
    total_co2_kg_hr: float = 0.0
    total_co2e_kg_hr: float = 0.0
    total_co2_tonnes_yr: float = 0.0
    total_co2e_tonnes_yr: float = 0.0
    total_ch4_kg_hr: float = 0.0
    total_nox_kg_hr: float = 0.0
    emission_intensity_kg_per_tonne: float = 0.0
    product_rate_kg_hr: float = 0.0
    method: str = "estimation"
    message: str = ""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Emission factors
_CO2_PER_KG_METHANE = 2.75       # kg CO₂ per kg methane burned  (stoichiometric)
_CO2_PER_KG_ETHANE  = 2.93
_CO2_PER_KG_PROPANE = 3.00
_CO2_PER_KG_NATGAS  = 2.75       # average natural gas
_CH4_GWP_100        = 28.0       # IPCC AR5 GWP for methane (100-yr)
_NOX_EF_GAS_TURBINE = 0.0015     # kg NOx per kg fuel (lean premix DLE)
_NOX_EF_FLARE       = 0.0005     # kg NOx per kg flared

# Fugitive emission factors (kg CH₄/hr per equipment piece, API/EPA)
_FUGITIVE_FACTORS: Dict[str, float] = {
    "Compressor":           0.5,     # seal leaks
    "Pump":                 0.05,
    "Separator":            0.02,
    "TwoPhaseSeparator":    0.02,
    "ThreePhaseSeparator":  0.02,
    "GasScrubber":          0.01,
    "ThrottlingValve":      0.01,
    "ControlValve":         0.02,
    "Mixer":                0.005,
    "HeatExchanger":        0.01,
    "Cooler":               0.005,
    "Heater":               0.005,
    "Pipeline":             0.01,
}

# Hours in a year
_HRS_PER_YEAR = 8760.0


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _estimate_fuel_gas_co2(power_kw: float) -> Tuple[float, float]:
    """Estimate CO₂ and fuel from gas-turbine / electric-drive power.

    Assumes gas-turbine drive with ~35% thermal efficiency for large
    compressors and electric-motor for pumps (grid emission factor zero
    by default — the user can extend this).

    Returns (co2_kg_hr, fuel_kg_hr).
    """
    if power_kw <= 0:
        return 0.0, 0.0
    # Gas turbine: LHV nat-gas ≈ 50 MJ/kg → 13.9 kWh/kg
    # 35% efficiency → fuel consumption = power / (0.35 * 13.9) kg/hr
    fuel_kg_hr = power_kw / (0.35 * 50.0 / 3.6)   # kW → kg/hr
    return fuel_kg_hr * _CO2_PER_KG_NATGAS, fuel_kg_hr


def _simple_class_name(java_obj) -> str:
    """Get the simple Java class name."""
    try:
        return str(java_obj.getClass().getSimpleName())
    except Exception:
        return "Unknown"


def calculate_emissions(
    model: NeqSimProcessModel,
    product_stream: Optional[str] = None,
    include_fugitives: bool = True,
    flare_streams: Optional[List[str]] = None,
    power_source: str = "gas_turbine",
) -> EmissionsResult:
    """
    Calculate process emissions.

    Parameters
    ----------
    model : NeqSimProcessModel
    product_stream : str, optional
        Name of the product/export stream (for intensity calc).
    include_fugitives : bool
        Include fugitive emission estimates.
    flare_streams : list[str], optional
        Stream names routed to flare.
    power_source : str
        'gas_turbine' or 'electric' — determines fuel-gas emissions.

    Returns
    -------
    EmissionsResult
    """
    result = EmissionsResult()
    proc = model.get_process()
    if proc is None:
        result.message = "No process system available."
        return result

    # Ensure latest state
    try:
        model.run()
    except Exception:
        pass

    sources: List[EmissionSource] = []

    # --- 1. Power-related fuel-gas emissions ---
    units = model.list_units()
    for u_info in units:
        u_name = u_info.name
        java_type = u_info.unit_type
        try:
            java_obj = model.get_unit(u_name)
        except KeyError:
            continue

        # Compressor / pump power
        power_kw = 0.0
        try:
            if hasattr(java_obj, "getPower"):
                # NeqSim getPower() always returns watts
                p = float(java_obj.getPower())
                power_kw = p / 1000.0
        except Exception:
            pass

        # Heater / cooler duty
        duty_kw = 0.0
        try:
            if hasattr(java_obj, "getEnergyInput"):
                # NeqSim getEnergyInput() returns watts
                d = float(java_obj.getEnergyInput())
                if d > 0:
                    duty_kw = d / 1000.0
        except Exception:
            pass

        if power_kw > 0 and power_source == "gas_turbine":
            co2, fuel = _estimate_fuel_gas_co2(power_kw)
            nox = fuel * _NOX_EF_GAS_TURBINE
            sources.append(EmissionSource(
                name=u_name,
                source_type="FUEL_GAS",
                co2_kg_hr=co2,
                co2e_kg_hr=co2,
                nox_kg_hr=nox,
                fuel_rate_kg_hr=fuel,
                detail=f"Power={power_kw:.0f} kW, gas-turbine drive",
            ))

        if duty_kw > 0 and java_type in ("Heater",):
            # Fired heater — assume 90% efficiency
            fuel_kg_hr = duty_kw / (0.90 * 50.0 / 3.6)
            co2 = fuel_kg_hr * _CO2_PER_KG_NATGAS
            sources.append(EmissionSource(
                name=u_name,
                source_type="FUEL_GAS",
                co2_kg_hr=co2,
                co2e_kg_hr=co2,
                fuel_rate_kg_hr=fuel_kg_hr,
                detail=f"Duty={duty_kw:.0f} kW, fired heater",
            ))

    # --- 2. Flare emissions ---
    if flare_streams:
        streams = model.list_streams()
        for fl_name in flare_streams:
            for s_info in streams:
                s_name = s_info.name
                if fl_name.lower() in s_name.lower():
                    flow_kg_hr = s_info.flow_rate_kg_hr or 0
                    if flow_kg_hr and flow_kg_hr > 0:
                        co2 = flow_kg_hr * _CO2_PER_KG_NATGAS * 0.98  # 98% combustion eff.
                        ch4 = flow_kg_hr * 0.02  # 2% uncombusted
                        nox = flow_kg_hr * _NOX_EF_FLARE
                        sources.append(EmissionSource(
                            name=s_name,
                            source_type="FLARE",
                            co2_kg_hr=co2,
                            co2e_kg_hr=co2 + ch4 * _CH4_GWP_100,
                            ch4_kg_hr=ch4,
                            nox_kg_hr=nox,
                            fuel_rate_kg_hr=flow_kg_hr,
                            detail=f"Flare flow={flow_kg_hr:.0f} kg/hr",
                        ))

    # --- 3. Fugitive emissions ---
    if include_fugitives:
        for u_info in units:
            u_name = u_info.name
            java_type = u_info.unit_type
            factor = _FUGITIVE_FACTORS.get(java_type, 0.0)
            if factor > 0:
                ch4 = factor
                co2e = ch4 * _CH4_GWP_100
                sources.append(EmissionSource(
                    name=u_name,
                    source_type="FUGITIVE",
                    ch4_kg_hr=ch4,
                    co2e_kg_hr=co2e,
                    detail=f"Equipment leak factor ({java_type})",
                ))

    # --- 4. Totals ---
    total_co2 = sum(s.co2_kg_hr for s in sources)
    total_co2e = sum(s.co2e_kg_hr for s in sources)
    total_ch4 = sum(s.ch4_kg_hr for s in sources)
    total_nox = sum(s.nox_kg_hr for s in sources)

    # Product rate for intensity
    product_rate = 0.0
    if product_stream:
        streams = model.list_streams()
        for s_info in streams:
            if product_stream.lower() in s_info.name.lower():
                product_rate = s_info.flow_rate_kg_hr or 0
                break
    if product_rate <= 0:
        # Try last stream
        streams = model.list_streams()
        if streams:
            last = streams[-1]
            product_rate = last.flow_rate_kg_hr or 0

    intensity = (total_co2e / product_rate * 1000) if product_rate > 0 else 0.0

    result.sources = sources
    result.total_co2_kg_hr = total_co2
    result.total_co2e_kg_hr = total_co2e
    result.total_co2_tonnes_yr = total_co2 * _HRS_PER_YEAR / 1000.0
    result.total_co2e_tonnes_yr = total_co2e * _HRS_PER_YEAR / 1000.0
    result.total_ch4_kg_hr = total_ch4
    result.total_nox_kg_hr = total_nox
    result.emission_intensity_kg_per_tonne = intensity
    result.product_rate_kg_hr = product_rate
    result.method = "estimation"
    result.message = f"Emissions estimated for {len(sources)} sources."

    return result


# ---------------------------------------------------------------------------
# Format for LLM
# ---------------------------------------------------------------------------

def format_emissions_result(result: EmissionsResult) -> str:
    """Format emissions for the LLM follow-up."""
    lines = ["=== EMISSIONS ANALYSIS ==="]
    lines.append(f"Method: {result.method}")
    lines.append(f"Total CO₂: {result.total_co2_kg_hr:.1f} kg/hr "
                 f"({result.total_co2_tonnes_yr:.0f} tonnes/yr)")
    lines.append(f"Total CO₂e: {result.total_co2e_kg_hr:.1f} kg/hr "
                 f"({result.total_co2e_tonnes_yr:.0f} tonnes/yr)")
    lines.append(f"Total CH₄: {result.total_ch4_kg_hr:.2f} kg/hr")
    lines.append(f"Total NOx: {result.total_nox_kg_hr:.3f} kg/hr")
    if result.product_rate_kg_hr > 0:
        lines.append(f"Emission intensity: {result.emission_intensity_kg_per_tonne:.1f} "
                     f"kg CO₂e / tonne product")
    lines.append("")

    # Sources breakdown
    lines.append("=== EMISSION SOURCES ===")
    for s in sorted(result.sources, key=lambda x: x.co2e_kg_hr, reverse=True):
        lines.append(f"  {s.name} [{s.source_type}]: "
                     f"CO₂={s.co2_kg_hr:.1f} kg/hr, CO₂e={s.co2e_kg_hr:.1f} kg/hr, "
                     f"CH₄={s.ch4_kg_hr:.2f} kg/hr | {s.detail}")

    return "\n".join(lines)
