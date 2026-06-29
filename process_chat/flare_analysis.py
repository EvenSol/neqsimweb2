"""
Flare Minimization Analysis — identify flare sources and recovery options.

Evaluates:
  - Sources of flare gas (relief valves, off-spec product, blowdown)
  - Estimated flare volumes and compositions
  - CO2 equivalent emissions from flaring
  - Recovery options (flare gas recovery, vapour recovery units)
  - Economic value of recovered gas
  - Regulatory compliance position

Returns a ``FlareAnalysisResult`` for UI display.
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
class FlareSource:
    """One source of flare gas."""
    name: str
    source_type: str     # RELIEF, OFF_SPEC, BLOWDOWN, ROUTINE
    flow_rate_kg_hr: float = 0.0
    temperature_C: float = 0.0
    pressure_bara: float = 0.0
    energy_content_kW: float = 0.0
    co2_equiv_tonnes_yr: float = 0.0
    detail: str = ""


@dataclass
class RecoveryOption:
    """Recovery option for flare gas."""
    name: str
    applicable_sources: List[str] = field(default_factory=list)
    recovery_pct: float = 0.0
    recovered_flow_kg_hr: float = 0.0
    capex_usd: float = 0.0
    opex_usd_yr: float = 0.0
    revenue_usd_yr: float = 0.0
    payback_years: float = 0.0
    co2_reduction_tonnes_yr: float = 0.0
    detail: str = ""


@dataclass
class FlareAnalysisResult:
    """Complete flare minimization analysis."""
    total_flare_rate_kg_hr: float = 0.0
    total_flare_energy_kW: float = 0.0
    total_co2_equiv_tonnes_yr: float = 0.0
    flare_value_usd_yr: float = 0.0
    sources: List[FlareSource] = field(default_factory=list)
    recovery_options: List[RecoveryOption] = field(default_factory=list)
    best_option: str = ""
    carbon_tax_exposure_usd_yr: float = 0.0
    carbon_price_usd_per_tonne: float = 50.0
    method: str = "process_analysis"
    message: str = ""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CO2 emission factor for flared nat gas (~2.7 kg CO2 / kg nat gas)
_CO2_FACTOR = 2.7
# Methane GWP (unburned CH4 slip assumed at 2% of flared volume)
_CH4_GWP = 28.0
_CH4_SLIP = 0.02
# Heating value of natural gas
_HHV_KJ_PER_KG = 48000.0
# Gas price for revenue estimate
_GAS_PRICE_USD_PER_KG = 0.12


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def run_flare_analysis(
    model: NeqSimProcessModel,
    carbon_price_usd_per_tonne: float = 50.0,
    gas_price_usd_per_kg: float = 0.12,
) -> FlareAnalysisResult:
    """
    Analyse flare sources in the process and evaluate recovery options.

    Args:
        model: NeqSim process model.
        carbon_price_usd_per_tonne: Carbon tax / ETS price.
        gas_price_usd_per_kg: Commodity price for recovered gas.

    Returns:
        FlareAnalysisResult with sources, recovery options, economics.
    """
    result = FlareAnalysisResult(
        method="process_analysis",
        carbon_price_usd_per_tonne=carbon_price_usd_per_tonne,
    )

    proc = model.get_process()
    if proc is None:
        result.message = "No process available."
        return result

    units = model.list_units()
    streams = model.list_streams()
    sources: List[FlareSource] = []

    # --------------- Identify potential flare sources ---------------

    for u_info in units:
        try:
            unit = model.get_unit(u_info.name)
            if unit is None:
                continue
            java_class = str(unit.getClass().getSimpleName())
        except Exception:
            continue

        # Separators: gas phase outlets can be flare-bound
        if "Separator" in java_class or "Scrubber" in java_class:
            try:
                gas_out = unit.getGasOutStream()
                if gas_out is not None:
                    flow = abs(float(gas_out.getFlowRate("kg/hr")))
                    temp = float(gas_out.getTemperature("C"))
                    pres = float(gas_out.getPressure("bara"))
                    # Consider low-pressure gas as potential flare source
                    if pres < 5.0 and flow > 0.1:
                        energy = flow * _HHV_KJ_PER_KG / 3600.0  # kW
                        co2_yr = _calc_co2_equiv(flow)
                        sources.append(FlareSource(
                            name=f"{u_info.name} LP gas",
                            source_type="ROUTINE",
                            flow_rate_kg_hr=flow,
                            temperature_C=temp,
                            pressure_bara=pres,
                            energy_content_kW=energy,
                            co2_equiv_tonnes_yr=co2_yr,
                            detail=f"Low-pressure gas from {u_info.name} "
                                   f"({pres:.1f} bara, {temp:.0f}°C)",
                        ))
            except Exception:
                pass

        # Valves: large pressure drops may indicate relief / blowdown path
        if java_class in ("ThrottlingValve", "Valve"):
            try:
                out_stream = unit.getOutletStream()
                if out_stream is None:
                    try:
                        out_stream = unit.getOutStream()
                    except Exception:
                        pass
                if out_stream is not None:
                    p_in = float(unit.getInletStream().getPressure("bara")) if hasattr(unit, 'getInletStream') else 0
                    p_out = float(out_stream.getPressure("bara"))
                    flow = abs(float(out_stream.getFlowRate("kg/hr")))
                    if p_in > 0 and p_out < 3.0 and (p_in / max(p_out, 0.1)) > 5.0:
                        energy = flow * _HHV_KJ_PER_KG / 3600.0
                        co2_yr = _calc_co2_equiv(flow)
                        sources.append(FlareSource(
                            name=f"{u_info.name} letdown",
                            source_type="BLOWDOWN",
                            flow_rate_kg_hr=flow,
                            temperature_C=float(out_stream.getTemperature("C")),
                            pressure_bara=p_out,
                            energy_content_kW=energy,
                            co2_equiv_tonnes_yr=co2_yr,
                            detail=f"Pressure letdown from {p_in:.0f} to {p_out:.1f} bara",
                        ))
            except Exception:
                pass

    # If no real flare sources found, create synthetic estimate
    if not sources:
        total_flow = 0.0
        for s in streams:
            try:
                st_obj = model.get_stream(s.name)
                total_flow += abs(float(st_obj.getFlowRate("kg/hr")))
            except Exception:
                pass
        # Typical flare = 0.5-1% of total flow
        if total_flow > 0:
            est_flare = total_flow * 0.005
            energy = est_flare * _HHV_KJ_PER_KG / 3600.0
            co2_yr = _calc_co2_equiv(est_flare)
            sources.append(FlareSource(
                name="Estimated routine flare",
                source_type="ROUTINE",
                flow_rate_kg_hr=est_flare,
                energy_content_kW=energy,
                co2_equiv_tonnes_yr=co2_yr,
                detail=f"Estimated at 0.5% of total throughput ({total_flow:,.0f} kg/hr)",
            ))

    # Totals
    total_rate = sum(s.flow_rate_kg_hr for s in sources)
    total_energy = sum(s.energy_content_kW for s in sources)
    total_co2 = sum(s.co2_equiv_tonnes_yr for s in sources)
    total_value = total_rate * gas_price_usd_per_kg * 8000  # 8000 hrs/yr

    result.sources = sources
    result.total_flare_rate_kg_hr = total_rate
    result.total_flare_energy_kW = total_energy
    result.total_co2_equiv_tonnes_yr = total_co2
    result.flare_value_usd_yr = total_value
    result.carbon_tax_exposure_usd_yr = total_co2 * carbon_price_usd_per_tonne

    # --------------- Recovery options ---------------
    source_names = [s.name for s in sources]

    # Option 1: Flare Gas Recovery Unit (FGRU)
    if total_rate > 0:
        recovery_pct = 0.90
        recovered = total_rate * recovery_pct
        capex = 1_500_000.0 + recovered * 100.0  # base + scale
        opex = capex * 0.05  # 5% of CAPEX / yr
        revenue = recovered * gas_price_usd_per_kg * 8000
        payback = capex / max(revenue - opex, 1) if (revenue - opex) > 0 else 99

        result.recovery_options.append(RecoveryOption(
            name="Flare Gas Recovery Unit (Compressor + KO Drum)",
            applicable_sources=source_names,
            recovery_pct=recovery_pct * 100,
            recovered_flow_kg_hr=recovered,
            capex_usd=capex,
            opex_usd_yr=opex,
            revenue_usd_yr=revenue,
            payback_years=payback,
            co2_reduction_tonnes_yr=total_co2 * recovery_pct,
            detail="Wet-gas compressor with KO drum re-injects into fuel gas or sales",
        ))

    # Option 2: Vapour Recovery Unit (VRU) - lower cost for small flows
    if total_rate > 0 and total_rate < 2000:
        recovery_pct = 0.85
        recovered = total_rate * recovery_pct
        capex = 500_000.0
        opex = capex * 0.04
        revenue = recovered * gas_price_usd_per_kg * 8000
        payback = capex / max(revenue - opex, 1) if (revenue - opex) > 0 else 99

        result.recovery_options.append(RecoveryOption(
            name="Vapour Recovery Unit (VRU)",
            applicable_sources=source_names,
            recovery_pct=recovery_pct * 100,
            recovered_flow_kg_hr=recovered,
            capex_usd=capex,
            opex_usd_yr=opex,
            revenue_usd_yr=revenue,
            payback_years=payback,
            co2_reduction_tonnes_yr=total_co2 * recovery_pct,
            detail="Ejector + condenser system for smaller flow recovery",
        ))

    # Option 3: Power generation from flare gas
    if total_energy > 100:
        gen_eff = 0.30
        power_kW = total_energy * gen_eff
        capex = power_kW * 1200  # $/kW installed for small genset
        opex = capex * 0.04
        electricity_price = 0.08  # $/kWh
        revenue = power_kW * 8000 * electricity_price
        payback = capex / max(revenue - opex, 1) if (revenue - opex) > 0 else 99

        result.recovery_options.append(RecoveryOption(
            name="On-site Power Generation (Gas Engine)",
            applicable_sources=source_names,
            recovery_pct=gen_eff * 100,
            recovered_flow_kg_hr=total_rate,
            capex_usd=capex,
            opex_usd_yr=opex,
            revenue_usd_yr=revenue,
            payback_years=payback,
            co2_reduction_tonnes_yr=total_co2 * 0.5,  # ~50% net reduction
            detail=f"Gas engine generating {power_kW:.0f} kW electricity",
        ))

    # Best option by payback
    if result.recovery_options:
        best = min(result.recovery_options, key=lambda o: o.payback_years)
        result.best_option = best.name

    result.message = (
        f"Total flare: {total_rate:,.0f} kg/hr ({total_energy:,.0f} kW thermal). "
        f"CO2 equiv: {total_co2:,.0f} tonnes/yr. "
        f"Value: ${total_value:,.0f}/yr. "
        f"Carbon tax exposure: ${result.carbon_tax_exposure_usd_yr:,.0f}/yr."
    )

    return result


def _calc_co2_equiv(flow_kg_hr: float) -> float:
    """Calculate CO2 equivalent emissions (tonnes/yr) for a flare stream."""
    hours_per_year = 8000
    # Direct CO2 from combustion
    co2_direct = flow_kg_hr * _CO2_FACTOR * hours_per_year / 1000.0  # tonnes/yr
    # Unburned methane (GWP contribution)
    ch4_slip = flow_kg_hr * _CH4_SLIP * _CH4_GWP * hours_per_year / 1000.0
    return co2_direct + ch4_slip


# ---------------------------------------------------------------------------
# Formatter for LLM
# ---------------------------------------------------------------------------

def format_flare_analysis_result(result: FlareAnalysisResult) -> str:
    """Format flare analysis result for LLM consumption."""
    lines = ["=== FLARE MINIMIZATION ANALYSIS ==="]

    lines.append(f"\nFLARE SUMMARY:")
    lines.append(f"  Total Flare Rate: {result.total_flare_rate_kg_hr:,.0f} kg/hr")
    lines.append(f"  Thermal Energy: {result.total_flare_energy_kW:,.0f} kW")
    lines.append(f"  CO2 Equivalent: {result.total_co2_equiv_tonnes_yr:,.0f} tonnes/yr")
    lines.append(f"  Value of Flared Gas: ${result.flare_value_usd_yr:,.0f}/yr")
    lines.append(f"  Carbon Tax Exposure: ${result.carbon_tax_exposure_usd_yr:,.0f}/yr "
                 f"(at ${result.carbon_price_usd_per_tonne}/tonne)")

    if result.sources:
        lines.append(f"\nFLARE SOURCES ({len(result.sources)}):")
        for s in result.sources:
            lines.append(
                f"  • {s.name} [{s.source_type}]: "
                f"{s.flow_rate_kg_hr:,.0f} kg/hr, "
                f"{s.co2_equiv_tonnes_yr:,.0f} tCO2e/yr"
            )
            if s.detail:
                lines.append(f"    {s.detail}")

    if result.recovery_options:
        lines.append(f"\nRECOVERY OPTIONS:")
        for opt in result.recovery_options:
            lines.append(f"\n  [{opt.name}]")
            lines.append(f"    Recovery: {opt.recovery_pct:.0f}% → {opt.recovered_flow_kg_hr:,.0f} kg/hr")
            lines.append(f"    CAPEX: ${opt.capex_usd:,.0f}")
            lines.append(f"    OPEX: ${opt.opex_usd_yr:,.0f}/yr")
            lines.append(f"    Revenue/Saving: ${opt.revenue_usd_yr:,.0f}/yr")
            lines.append(f"    Payback: {opt.payback_years:.1f} years")
            lines.append(f"    CO2 Reduction: {opt.co2_reduction_tonnes_yr:,.0f} tonnes/yr")

        if result.best_option:
            lines.append(f"\n  ★ RECOMMENDED: {result.best_option}")

    if result.message:
        lines.append(f"\n{result.message}")

    return "\n".join(lines)
