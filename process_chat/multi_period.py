"""
Multi-Period / Seasonal Planning — batch scenario comparison across operating periods.

Evaluates plant performance across multiple operating scenarios, e.g.:
  - Summer vs winter ambient temperatures
  - Low, medium and high feed rates
  - Different feed compositions (lean gas, rich gas)
  - Power import constraints

Each scenario is simulated independently, and results are compared side-by-side
for KPIs like production, energy, emissions, equipment utilisation.

Returns a ``MultiPeriodResult`` for UI display (comparison tables, charts).
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .process_model import NeqSimProcessModel


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ScenarioSpec:
    """Definition of one operating scenario / period."""
    name: str
    description: str = ""
    feed_flow_mult: float = 1.0      # multiplier on base feed
    ambient_temp_C: Optional[float] = None
    feed_temp_C: Optional[float] = None
    feed_pressure_bara: Optional[float] = None
    duration_hours: float = 2000.0     # hours/yr for this period
    extra_params: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScenarioKPI:
    """KPI values for one scenario."""
    name: str
    product_rate_kg_hr: float = 0.0
    total_power_kW: float = 0.0
    total_cooling_kW: float = 0.0
    total_heating_kW: float = 0.0
    specific_energy_kWh_tonne: float = 0.0
    co2_equiv_tonnes_yr: float = 0.0
    production_tonnes_period: float = 0.0
    warnings: List[str] = field(default_factory=list)


@dataclass
class MultiPeriodResult:
    """Complete multi-period analysis."""
    scenarios: List[ScenarioKPI] = field(default_factory=list)
    total_production_tonnes_yr: float = 0.0
    total_energy_MWh_yr: float = 0.0
    total_co2_tonnes_yr: float = 0.0
    avg_specific_energy: float = 0.0
    best_scenario: str = ""
    worst_scenario: str = ""
    kpi_names: List[str] = field(default_factory=list)
    method: str = "multi_period"
    message: str = ""


# ---------------------------------------------------------------------------
# Default scenarios
# ---------------------------------------------------------------------------

DEFAULT_SCENARIOS: List[ScenarioSpec] = [
    ScenarioSpec(
        name="Summer Peak",
        description="High ambient temperature, high demand",
        feed_flow_mult=1.1,
        ambient_temp_C=40.0,
        duration_hours=2000.0,
    ),
    ScenarioSpec(
        name="Summer Normal",
        description="Normal summer conditions",
        feed_flow_mult=1.0,
        ambient_temp_C=35.0,
        duration_hours=2000.0,
    ),
    ScenarioSpec(
        name="Winter Normal",
        description="Normal winter conditions",
        feed_flow_mult=1.0,
        ambient_temp_C=10.0,
        duration_hours=2000.0,
    ),
    ScenarioSpec(
        name="Winter Turndown",
        description="Low demand, cold ambient",
        feed_flow_mult=0.6,
        ambient_temp_C=5.0,
        duration_hours=2000.0,
    ),
]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def run_multi_period(
    model: NeqSimProcessModel,
    scenarios: Optional[List[ScenarioSpec]] = None,
    feed_stream: Optional[str] = None,
    product_stream: Optional[str] = None,
) -> MultiPeriodResult:
    """
    Run multi-period / seasonal planning analysis.

    Args:
        model: The NeqSim process model.
        scenarios: List of scenarios. Uses DEFAULT_SCENARIOS if None.
        feed_stream: Name of feed stream to modify.
        product_stream: Name of product stream for KPIs.

    Returns:
        MultiPeriodResult with per-scenario KPIs and totals.
    """
    result = MultiPeriodResult(method="multi_period")
    specs = scenarios if scenarios else DEFAULT_SCENARIOS

    proc = model.get_process()
    if proc is None:
        result.message = "No process available."
        return result

    streams = model.list_streams()
    units = model.list_units()

    # Find feed stream
    target_feed = ""
    if feed_stream:
        for s in streams:
            if feed_stream.lower() in s.name.lower():
                target_feed = s.name
                break
    if not target_feed and streams:
        target_feed = streams[0].name

    # Find product stream
    target_product = ""
    if product_stream:
        for s in streams:
            if product_stream.lower() in s.name.lower():
                target_product = s.name
                break
    if not target_product and streams:
        target_product = streams[-1].name

    # Get base feed conditions
    base_flow = 0.0
    base_temp = 0.0
    base_pres = 0.0
    try:
        feed_obj = model.get_stream(target_feed)
        base_flow = float(feed_obj.getFlowRate("kg/hr"))
        base_temp = float(feed_obj.getTemperature("C"))
        base_pres = float(feed_obj.getPressure("bara"))
    except Exception:
        pass

    total_prod = 0.0
    total_energy = 0.0
    total_co2 = 0.0

    for spec in specs:
        kpi = _evaluate_scenario(
            model, spec, target_feed, target_product,
            base_flow, base_temp, base_pres, units,
        )
        total_prod += kpi.production_tonnes_period
        total_energy += kpi.total_power_kW * spec.duration_hours / 1000.0  # MWh
        total_co2 += kpi.co2_equiv_tonnes_yr
        result.scenarios.append(kpi)

    result.total_production_tonnes_yr = total_prod
    result.total_energy_MWh_yr = total_energy
    result.total_co2_tonnes_yr = total_co2

    if result.scenarios:
        by_eff = sorted(result.scenarios, key=lambda k: k.specific_energy_kWh_tonne)
        result.best_scenario = by_eff[0].name if by_eff else ""
        result.worst_scenario = by_eff[-1].name if by_eff else ""
        if total_prod > 0:
            result.avg_specific_energy = total_energy * 1000.0 / total_prod  # kWh/tonne

    result.kpi_names = [
        "Product Rate (kg/hr)", "Power (kW)", "Cooling (kW)",
        "Heating (kW)", "Specific Energy (kWh/t)", "CO2 (t/yr)",
    ]

    result.message = (
        f"Analysed {len(specs)} scenarios. "
        f"Total annual production: {total_prod:,.0f} tonnes. "
        f"Total energy: {total_energy:,.0f} MWh. "
        f"Best efficiency: {result.best_scenario}. "
        f"Worst efficiency: {result.worst_scenario}."
    )

    return result


def _evaluate_scenario(
    model: NeqSimProcessModel,
    spec: ScenarioSpec,
    feed_name: str,
    product_name: str,
    base_flow: float,
    base_temp: float,
    base_pres: float,
    units: list,
) -> ScenarioKPI:
    """Evaluate a single scenario by cloning the process."""
    kpi = ScenarioKPI(name=spec.name)
    warnings: List[str] = []

    try:
        proc = model.get_process()
        if proc is None:
            kpi.warnings.append("No process")
            return kpi

        # Clone model for scenario
        proc_copy = proc.copy()

        # Modify feed conditions
        if feed_name:
            try:
                feed = _find_stream_in_process(proc_copy, feed_name)
                if feed is not None:
                    new_flow = base_flow * spec.feed_flow_mult
                    feed.setFlowRate(new_flow, "kg/hr")
                    if spec.feed_temp_C is not None:
                        feed.setTemperature(spec.feed_temp_C, "C")
                    elif spec.ambient_temp_C is not None:
                        # Use ambient as feed temp if nothing specific
                        pass
                    if spec.feed_pressure_bara is not None:
                        feed.setPressure(spec.feed_pressure_bara, "bara")
            except Exception as e:
                warnings.append(f"Failed to set feed: {e}")

        # Modify cooler outlet temps based on ambient
        if spec.ambient_temp_C is not None:
            for u_info in units:
                try:
                    unit = _find_unit_in_process(proc_copy, u_info.name)
                    if unit is None:
                        continue
                    jclass = str(unit.getClass().getSimpleName())
                    if jclass in ("Cooler", "AirCooler"):
                        # Air cooler limited by ambient + approach
                        approach = 10.0  # typical 10°C approach
                        min_out_temp = spec.ambient_temp_C + approach
                        try:
                            current_out = float(unit.getOutletStream().getTemperature("C"))
                            if current_out < min_out_temp:
                                unit.setOutTemperature(min_out_temp + 273.15)
                                warnings.append(
                                    f"{u_info.name}: outlet raised to {min_out_temp:.0f}°C "
                                    f"due to {spec.ambient_temp_C:.0f}°C ambient"
                                )
                        except Exception:
                            pass
                except Exception:
                    pass

        # Run process
        try:
            proc_copy.run()
        except Exception as e:
            warnings.append(f"Simulation failed: {e}")
            kpi.warnings = warnings
            return kpi

        # Extract KPIs
        if product_name:
            try:
                prod = _find_stream_in_process(proc_copy, product_name)
                if prod is not None:
                    kpi.product_rate_kg_hr = abs(float(prod.getFlowRate("kg/hr")))
                    kpi.production_tonnes_period = (
                        kpi.product_rate_kg_hr * spec.duration_hours / 1000.0
                    )
            except Exception:
                pass

        # Power and duty totals
        for u_info in units:
            try:
                unit = _find_unit_in_process(proc_copy, u_info.name)
                if unit is None:
                    continue
                jclass = str(unit.getClass().getSimpleName())

                if jclass in ("Compressor", "Pump"):
                    try:
                        kpi.total_power_kW += abs(float(unit.getPower("kW")))
                    except Exception:
                        pass

                elif jclass in ("Cooler", "AirCooler", "WaterCooler"):
                    try:
                        kpi.total_cooling_kW += abs(float(unit.getDuty())) / 1000.0
                    except Exception:
                        pass

                elif jclass == "Heater":
                    try:
                        kpi.total_heating_kW += abs(float(unit.getDuty())) / 1000.0
                    except Exception:
                        pass

            except Exception:
                pass

        if kpi.product_rate_kg_hr > 0:
            kpi.specific_energy_kWh_tonne = (
                kpi.total_power_kW / (kpi.product_rate_kg_hr / 1000.0)
            )

        # Simple CO2 estimate (fuel gas for power at 30% efficiency)
        if kpi.total_power_kW > 0:
            fuel_kg_hr = (kpi.total_power_kW / 0.30) / 13.0
            kpi.co2_equiv_tonnes_yr = fuel_kg_hr * 2.7 * spec.duration_hours / 1000.0

    except Exception as e:
        warnings.append(f"Scenario evaluation failed: {e}")

    kpi.warnings = warnings
    return kpi


def _find_stream_in_process(proc, name: str):
    """Find stream by name in a process copy."""
    try:
        for i in range(proc.getNumberOfNodes()):
            node = proc.getNode(i)
            if node is None:
                continue
            node_name = str(node.getName()) if hasattr(node, 'getName') else ""
            jclass = str(node.getClass().getSimpleName())
            if "Stream" in jclass and node_name.lower() == name.lower():
                return node
    except Exception:
        pass
    return None


def _find_unit_in_process(proc, name: str):
    """Find unit operation by name in a process copy."""
    try:
        for i in range(proc.getNumberOfNodes()):
            node = proc.getNode(i)
            if node is None:
                continue
            node_name = str(node.getName()) if hasattr(node, 'getName') else ""
            if node_name.lower() == name.lower():
                return node
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Formatter for LLM
# ---------------------------------------------------------------------------

def format_multi_period_result(result: MultiPeriodResult) -> str:
    """Format multi-period result for LLM consumption."""
    lines = ["=== MULTI-PERIOD / SEASONAL PLANNING ==="]

    lines.append(f"\nANNUAL SUMMARY:")
    lines.append(f"  Total Production: {result.total_production_tonnes_yr:,.0f} tonnes/yr")
    lines.append(f"  Total Energy: {result.total_energy_MWh_yr:,.0f} MWh/yr")
    lines.append(f"  Total CO2: {result.total_co2_tonnes_yr:,.0f} tonnes/yr")
    lines.append(f"  Avg Specific Energy: {result.avg_specific_energy:.1f} kWh/tonne")
    lines.append(f"  Best Efficiency: {result.best_scenario}")
    lines.append(f"  Worst Efficiency: {result.worst_scenario}")

    lines.append(f"\nSCENARIO COMPARISON:")
    lines.append(f"{'Scenario':<20} {'Product':>10} {'Power':>10} {'Cooling':>10} "
                 f"{'Heating':>10} {'Spec.E':>10} {'CO2':>10}")
    lines.append(f"{'':·<20} {'kg/hr':>10} {'kW':>10} {'kW':>10} "
                 f"{'kW':>10} {'kWh/t':>10} {'t/yr':>10}")
    lines.append("-" * 90)

    for s in result.scenarios:
        lines.append(
            f"{s.name:<20} {s.product_rate_kg_hr:>10,.0f} {s.total_power_kW:>10,.0f} "
            f"{s.total_cooling_kW:>10,.0f} {s.total_heating_kW:>10,.0f} "
            f"{s.specific_energy_kWh_tonne:>10,.1f} {s.co2_equiv_tonnes_yr:>10,.0f}"
        )
        for w in s.warnings:
            lines.append(f"  ⚠ {w}")

    if result.message:
        lines.append(f"\n{result.message}")

    return "\n".join(lines)
