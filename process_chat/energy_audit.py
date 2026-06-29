"""
Energy Audit / Utility Balance — comprehensive energy accounting for the process.

Calculates:
  - Total power consumption (compressors, pumps)
  - Total cooling and heating duties
  - Specific energy consumption (kWh/tonne product)
  - Energy breakdown by equipment type and service
  - Comparison against industry benchmarks
  - Top energy consumers with improvement suggestions

Returns an ``EnergyAuditResult`` for UI display (tables, pie charts, benchmarks).
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
class EnergyConsumer:
    """One energy-consuming equipment item."""
    name: str
    equipment_type: str
    energy_type: str           # POWER, COOLING, HEATING
    consumption_kW: float = 0.0
    share_pct: float = 0.0     # % of total for this energy type
    detail: str = ""


@dataclass
class EnergyBreakdown:
    """Energy breakdown by category."""
    category: str
    power_kW: float = 0.0
    cooling_kW: float = 0.0
    heating_kW: float = 0.0


@dataclass
class BenchmarkComparison:
    """Comparison against industry benchmarks."""
    metric: str
    actual_value: float = 0.0
    benchmark_value: float = 0.0
    unit: str = ""
    status: str = "NORMAL"     # GOOD, NORMAL, POOR
    detail: str = ""


@dataclass
class ImprovementSuggestion:
    """A specific energy improvement suggestion."""
    equipment: str
    suggestion: str
    potential_saving_kW: float = 0.0
    potential_saving_pct: float = 0.0
    detail: str = ""


@dataclass
class EnergyAuditResult:
    """Complete energy audit result."""
    total_power_kW: float = 0.0
    total_cooling_kW: float = 0.0
    total_heating_kW: float = 0.0
    net_energy_kW: float = 0.0
    specific_energy_kWh_per_tonne: float = 0.0
    product_rate_kg_hr: float = 0.0
    product_stream: str = ""
    consumers: List[EnergyConsumer] = field(default_factory=list)
    breakdowns: List[EnergyBreakdown] = field(default_factory=list)
    benchmarks: List[BenchmarkComparison] = field(default_factory=list)
    suggestions: List[ImprovementSuggestion] = field(default_factory=list)
    top_power_consumers: List[str] = field(default_factory=list)
    top_heat_consumers: List[str] = field(default_factory=list)
    fuel_gas_rate_kg_hr: float = 0.0
    fuel_gas_cost_usd_hr: float = 0.0
    method: str = "process_analysis"
    message: str = ""


# ---------------------------------------------------------------------------
# Equipment type classification
# ---------------------------------------------------------------------------

_POWER_CONSUMERS = {"Compressor", "Pump", "ESPPump"}
_COOLING_UNITS = {"Cooler", "AirCooler", "WaterCooler"}
_HEATING_UNITS = {"Heater"}
_HX_UNITS = {"HeatExchanger", "MultiStreamHeatExchanger"}

# Industry benchmarks (GPSA / typical gas processing)
_BENCHMARKS = {
    "specific_energy_kWh_tonne": {
        "good": 30.0,
        "normal": 60.0,
        "poor": 100.0,
        "unit": "kWh/tonne",
        "source": "GPSA typical gas processing",
    },
    "compressor_efficiency": {
        "good": 0.80,
        "normal": 0.72,
        "poor": 0.65,
        "unit": "polytropic",
        "source": "Centrifugal compressor typical",
    },
}


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def run_energy_audit(
    model: NeqSimProcessModel,
    product_stream: Optional[str] = None,
    fuel_gas_price_usd_per_kg: float = 0.15,
) -> EnergyAuditResult:
    """
    Run comprehensive energy audit on the process.

    Args:
        model: The NeqSim process model.
        product_stream: Name of product/export stream for intensity calc.
        fuel_gas_price_usd_per_kg: Fuel gas price for cost estimation.

    Returns:
        EnergyAuditResult with complete energy accounting.
    """
    result = EnergyAuditResult(method="process_analysis")

    proc = model.get_process()
    if proc is None:
        result.message = "No process available."
        return result

    units = model.list_units()
    streams = model.list_streams()

    total_power = 0.0
    total_cooling = 0.0
    total_heating = 0.0
    consumers: List[EnergyConsumer] = []

    for u_info in units:
        try:
            unit = model.get_unit(u_info.name)
            if unit is None:
                continue
            java_class = str(unit.getClass().getSimpleName())
        except Exception:
            continue

        try:
            if java_class in _POWER_CONSUMERS:
                power = 0.0
                try:
                    power = abs(float(unit.getPower("kW")))
                except Exception:
                    pass
                if power > 0.01:
                    total_power += power
                    consumers.append(EnergyConsumer(
                        name=u_info.name,
                        equipment_type=java_class,
                        energy_type="POWER",
                        consumption_kW=power,
                        detail=f"{java_class}: {power:.0f} kW shaft power",
                    ))

            elif java_class in _COOLING_UNITS:
                duty = 0.0
                try:
                    duty = abs(float(unit.getDuty())) / 1000.0  # W → kW
                except Exception:
                    try:
                        duty = abs(float(unit.getEnergyInput())) / 1000.0
                    except Exception:
                        pass
                if duty > 0.01:
                    total_cooling += duty
                    consumers.append(EnergyConsumer(
                        name=u_info.name,
                        equipment_type=java_class,
                        energy_type="COOLING",
                        consumption_kW=duty,
                        detail=f"{java_class}: {duty:.0f} kW cooling duty",
                    ))

            elif java_class in _HEATING_UNITS:
                duty = 0.0
                try:
                    duty = abs(float(unit.getDuty())) / 1000.0
                except Exception:
                    try:
                        duty = abs(float(unit.getEnergyInput())) / 1000.0
                    except Exception:
                        pass
                if duty > 0.01:
                    total_heating += duty
                    consumers.append(EnergyConsumer(
                        name=u_info.name,
                        equipment_type=java_class,
                        energy_type="HEATING",
                        consumption_kW=duty,
                        detail=f"{java_class}: {duty:.0f} kW heating duty",
                    ))

        except Exception:
            continue

    # Calculate shares
    for c in consumers:
        if c.energy_type == "POWER" and total_power > 0:
            c.share_pct = c.consumption_kW / total_power * 100
        elif c.energy_type == "COOLING" and total_cooling > 0:
            c.share_pct = c.consumption_kW / total_cooling * 100
        elif c.energy_type == "HEATING" and total_heating > 0:
            c.share_pct = c.consumption_kW / total_heating * 100

    result.total_power_kW = total_power
    result.total_cooling_kW = total_cooling
    result.total_heating_kW = total_heating
    result.net_energy_kW = total_power + total_heating
    result.consumers = sorted(consumers, key=lambda x: x.consumption_kW, reverse=True)

    # Product stream for intensity calculation
    target_stream = ""
    if product_stream:
        for s in streams:
            if product_stream.lower() in s.name.lower():
                target_stream = s.name
                break
    if not target_stream:
        # Use last stream as product
        if streams:
            target_stream = streams[-1].name

    product_rate = 0.0
    if target_stream:
        try:
            s_obj = model.get_stream(target_stream)
            product_rate = float(s_obj.getFlowRate("kg/hr"))
        except Exception:
            pass

    result.product_stream = target_stream
    result.product_rate_kg_hr = product_rate

    if product_rate > 0:
        result.specific_energy_kWh_per_tonne = (
            result.net_energy_kW / (product_rate / 1000.0)
        )

    # Fuel gas estimation (assume gas turbine at 30% efficiency)
    if total_power > 0:
        fuel_power = total_power / 0.30  # thermal input
        result.fuel_gas_rate_kg_hr = fuel_power / 13.0  # ~13 kWh/kg nat gas
        result.fuel_gas_cost_usd_hr = result.fuel_gas_rate_kg_hr * fuel_gas_price_usd_per_kg

    # Breakdowns by equipment type
    type_power: Dict[str, float] = {}
    type_cooling: Dict[str, float] = {}
    type_heating: Dict[str, float] = {}
    for c in consumers:
        if c.energy_type == "POWER":
            type_power[c.equipment_type] = type_power.get(c.equipment_type, 0) + c.consumption_kW
        elif c.energy_type == "COOLING":
            type_cooling[c.equipment_type] = type_cooling.get(c.equipment_type, 0) + c.consumption_kW
        elif c.energy_type == "HEATING":
            type_heating[c.equipment_type] = type_heating.get(c.equipment_type, 0) + c.consumption_kW

    all_types = set(list(type_power.keys()) + list(type_cooling.keys()) + list(type_heating.keys()))
    for t in all_types:
        result.breakdowns.append(EnergyBreakdown(
            category=t,
            power_kW=type_power.get(t, 0),
            cooling_kW=type_cooling.get(t, 0),
            heating_kW=type_heating.get(t, 0),
        ))

    # Top consumers
    power_consumers = [c for c in consumers if c.energy_type == "POWER"]
    power_consumers.sort(key=lambda x: x.consumption_kW, reverse=True)
    result.top_power_consumers = [c.name for c in power_consumers[:3]]

    heat_consumers = [c for c in consumers if c.energy_type in ("COOLING", "HEATING")]
    heat_consumers.sort(key=lambda x: x.consumption_kW, reverse=True)
    result.top_heat_consumers = [c.name for c in heat_consumers[:3]]

    # Benchmarks
    if result.specific_energy_kWh_per_tonne > 0:
        bm = _BENCHMARKS["specific_energy_kWh_tonne"]
        status = "GOOD"
        if result.specific_energy_kWh_per_tonne > bm["poor"]:
            status = "POOR"
        elif result.specific_energy_kWh_per_tonne > bm["normal"]:
            status = "NORMAL"
        result.benchmarks.append(BenchmarkComparison(
            metric="Specific Energy Consumption",
            actual_value=result.specific_energy_kWh_per_tonne,
            benchmark_value=bm["normal"],
            unit=bm["unit"],
            status=status,
            detail=f"Good: <{bm['good']}, Normal: <{bm['normal']}, Poor: >{bm['poor']} {bm['unit']}",
        ))

    # Suggestions
    for c in power_consumers[:3]:
        if c.share_pct > 30:
            result.suggestions.append(ImprovementSuggestion(
                equipment=c.name,
                suggestion=f"Consider VSD (variable speed drive) for {c.name} — "
                           f"consumes {c.share_pct:.0f}% of total power",
                potential_saving_kW=c.consumption_kW * 0.10,
                potential_saving_pct=10.0,
                detail="VSD can reduce power by 10-20% at partial load",
            ))

    if total_cooling > 0 and total_heating > 0:
        recoverable = min(total_cooling, total_heating) * 0.4
        result.suggestions.append(ImprovementSuggestion(
            equipment="Process-wide",
            suggestion="Evaluate heat integration between hot and cold streams",
            potential_saving_kW=recoverable,
            potential_saving_pct=recoverable / result.net_energy_kW * 100 if result.net_energy_kW > 0 else 0,
            detail=f"Up to {recoverable:.0f} kW recoverable via heat exchange",
        ))

    result.message = (
        f"Total power: {total_power:.0f} kW, "
        f"Total cooling: {total_cooling:.0f} kW, "
        f"Total heating: {total_heating:.0f} kW. "
        f"Specific energy: {result.specific_energy_kWh_per_tonne:.1f} kWh/tonne."
    )

    return result


# ---------------------------------------------------------------------------
# Formatter for LLM
# ---------------------------------------------------------------------------

def format_energy_audit_result(result: EnergyAuditResult) -> str:
    """Format energy audit result for LLM consumption."""
    lines = ["=== ENERGY AUDIT / UTILITY BALANCE ==="]
    lines.append(f"Product Stream: {result.product_stream}")
    lines.append(f"Product Rate: {result.product_rate_kg_hr:,.0f} kg/hr")

    lines.append(f"\nENERGY SUMMARY:")
    lines.append(f"  Total Power (shaft): {result.total_power_kW:,.0f} kW")
    lines.append(f"  Total Cooling Duty: {result.total_cooling_kW:,.0f} kW")
    lines.append(f"  Total Heating Duty: {result.total_heating_kW:,.0f} kW")
    lines.append(f"  Net Energy Input: {result.net_energy_kW:,.0f} kW")
    lines.append(f"  Specific Energy: {result.specific_energy_kWh_per_tonne:.1f} kWh/tonne product")

    if result.fuel_gas_rate_kg_hr > 0:
        lines.append(f"\nFUEL GAS:")
        lines.append(f"  Estimated Fuel Gas Rate: {result.fuel_gas_rate_kg_hr:,.0f} kg/hr")
        lines.append(f"  Fuel Gas Cost: ${result.fuel_gas_cost_usd_hr:,.1f}/hr "
                     f"(${result.fuel_gas_cost_usd_hr * 8000:,.0f}/yr)")

    if result.consumers:
        lines.append(f"\nENERGY CONSUMERS (top):")
        for c in result.consumers[:10]:
            lines.append(
                f"  {c.name} ({c.equipment_type}): {c.consumption_kW:,.0f} kW "
                f"{c.energy_type} ({c.share_pct:.0f}%)"
            )

    if result.benchmarks:
        lines.append(f"\nBENCHMARK COMPARISON:")
        for bm in result.benchmarks:
            icon = {"GOOD": "✅", "NORMAL": "⚠️", "POOR": "❌"}.get(bm.status, "❓")
            lines.append(
                f"  {icon} {bm.metric}: {bm.actual_value:.1f} vs benchmark {bm.benchmark_value:.1f} "
                f"{bm.unit} [{bm.status}]"
            )
            lines.append(f"     {bm.detail}")

    if result.suggestions:
        lines.append(f"\nIMPROVEMENT SUGGESTIONS:")
        for s in result.suggestions:
            lines.append(f"  • {s.suggestion}")
            lines.append(f"    Potential saving: {s.potential_saving_kW:.0f} kW ({s.potential_saving_pct:.0f}%)")

    if result.message:
        lines.append(f"\n{result.message}")

    return "\n".join(lines)
