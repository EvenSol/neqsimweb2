"""
Production & Reservoir Fluid Scenarios — study the effect of varying well
feed composition, GOR, water cut, and production profiles on process performance.

Supports:
  - Well feed composition sweep (lean gas → rich gas transition)
  - GOR (Gas-Oil Ratio) variation study
  - Water cut ramp-up (field life simulation)
  - Component injection / removal effect
  - Multi-well blending optimisation
  - Production decline profile evaluation

Returns a ``ProductionScenarioResult`` with per-case KPIs for comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .process_model import NeqSimProcessModel, KPI


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class WellCase:
    """Result for one well / feed condition."""
    name: str
    description: str = ""
    feed_composition: Dict[str, float] = field(default_factory=dict)
    kpis: Dict[str, float] = field(default_factory=dict)
    stream_conditions: Dict[str, float] = field(default_factory=dict)
    feasible: bool = True
    warnings: List[str] = field(default_factory=list)
    error: str = ""


@dataclass
class ProductionScenarioResult:
    """Complete production scenario analysis."""
    scenario_type: str = ""          # composition_sweep, gor_sweep, watercut_sweep, blend
    cases: List[WellCase] = field(default_factory=list)
    base_case: Optional[WellCase] = None
    sweep_variable: str = ""
    sweep_values: List[float] = field(default_factory=list)
    kpi_names: List[str] = field(default_factory=list)
    method: str = "clone_and_run"
    message: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_stream_conditions(model: NeqSimProcessModel, stream_name: str) -> Dict[str, float]:
    """Read T, P, flow from a named stream."""
    conds: Dict[str, float] = {}
    try:
        s = model.get_stream(stream_name)
        conds["temperature_C"] = float(s.getTemperature("C"))
        conds["pressure_bara"] = float(s.getPressure("bara"))
        conds["flow_kg_hr"] = float(s.getFlowRate("kg/hr"))
    except Exception:
        pass
    return conds


def _get_kpis(model: NeqSimProcessModel, kpi_names: List[str]) -> Dict[str, float]:
    """Run model and extract named KPIs (fuzzy match)."""
    try:
        rr = model.run()
        kpis = rr.kpis
        out: Dict[str, float] = {}
        for want in kpi_names:
            wl = want.lower()
            for k, v in kpis.items():
                if wl in k.lower() and v.value is not None:
                    out[k] = v.value
        # Always include totals
        for k in ("total_power_kW", "total_duty_kW"):
            if k in kpis and kpis[k].value is not None:
                out[k] = kpis[k].value
        return out
    except Exception:
        return {}


def _find_feed_stream(model: NeqSimProcessModel, stream_name: Optional[str] = None) -> Optional[str]:
    """Find the feed stream name (first stream or user-specified)."""
    streams = model.list_streams()
    if not streams:
        return None
    if stream_name:
        for s in streams:
            if stream_name.lower() in s.name.lower():
                return s.name
    return streams[0].name


def _set_stream_composition(model: NeqSimProcessModel, stream_name: str,
                            composition: Dict[str, float]):
    """Replace the composition of a stream with new mole-fraction-based values.
    
    Sets z-fractions for each component and re-normalises.
    """
    java_stream = model.get_stream(stream_name)
    fl = java_stream.getFluid()

    # Normalise composition
    total_frac = sum(composition.values())
    if total_frac <= 0:
        return

    for comp_name, frac in composition.items():
        norm_frac = frac / total_frac
        try:
            idx = fl.getComponentIndex(comp_name)
            if idx >= 0:
                fl.getComponent(idx).setz(norm_frac)
        except Exception:
            pass
    # Re-init after composition change
    try:
        fl.init(0)
        fl.init(1)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Composition sweep
# ---------------------------------------------------------------------------

def _run_composition_sweep(
    model: NeqSimProcessModel,
    feed_stream: str,
    component: str,
    min_frac: float,
    max_frac: float,
    n_points: int,
    response_kpis: List[str],
) -> ProductionScenarioResult:
    """Sweep a single component's mole fraction in the feed."""
    result = ProductionScenarioResult(
        scenario_type="composition_sweep",
        sweep_variable=f"{component} mole fraction",
        kpi_names=response_kpis,
    )

    values = [min_frac + (max_frac - min_frac) * i / max(n_points - 1, 1)
              for i in range(n_points)]
    result.sweep_values = values

    # Capture base case
    base_kpis = _get_kpis(model, response_kpis)
    base_conds = _extract_stream_conditions(model, feed_stream)
    result.base_case = WellCase(
        name="Base Case",
        kpis=base_kpis,
        stream_conditions=base_conds,
    )

    for val in values:
        clone = model.clone()
        try:
            js = clone.get_stream(feed_stream)
            fl = js.getFluid()

            # Get current composition and modify the target component
            comp_found = False
            for i in range(fl.getNumberOfComponents()):
                cname = str(fl.getComponent(i).getName())
                if cname.lower() == component.lower():
                    fl.getComponent(i).setz(float(val))
                    comp_found = True
                    break

            if not comp_found:
                # Try adding the component
                fl.addComponent(component, float(val))
                fl.setMixingRule(fl.getMixingRuleNumber())

            # Re-normalise remaining components
            fl.init(0)

            kpis = _get_kpis(clone, response_kpis)
            conds = _extract_stream_conditions(clone, feed_stream)

            result.cases.append(WellCase(
                name=f"{component}={val:.4f}",
                feed_composition={component: val},
                kpis=kpis,
                stream_conditions=conds,
                feasible=True,
            ))
        except Exception as e:
            result.cases.append(WellCase(
                name=f"{component}={val:.4f}",
                feed_composition={component: val},
                feasible=False,
                error=str(e),
            ))

    result.message = (
        f"Composition sweep: {component} from {min_frac:.4f} to {max_frac:.4f} "
        f"({n_points} points) in stream '{feed_stream}'"
    )
    return result


# ---------------------------------------------------------------------------
# GOR sweep
# ---------------------------------------------------------------------------

def _run_gor_sweep(
    model: NeqSimProcessModel,
    feed_stream: str,
    min_gor: float,
    max_gor: float,
    n_points: int,
    gas_components: Optional[List[str]] = None,
    oil_components: Optional[List[str]] = None,
    response_kpis: Optional[List[str]] = None,
) -> ProductionScenarioResult:
    """Sweep GOR by scaling gas vs liquid components in the feed.
    
    Gas components default to: methane, ethane, propane, CO2, nitrogen, H2S.
    Oil components default to: everything else (heavier HCs, C7+, etc.).
    """
    if response_kpis is None:
        response_kpis = ["power", "duty", "flow"]
    if gas_components is None:
        gas_components = ["methane", "ethane", "CO2", "nitrogen", "H2S"]
    if oil_components is None:
        oil_components = []  # will be auto-detected

    result = ProductionScenarioResult(
        scenario_type="gor_sweep",
        sweep_variable="GOR (Sm3/Sm3)",
        kpi_names=response_kpis,
    )

    gors = [min_gor + (max_gor - min_gor) * i / max(n_points - 1, 1)
            for i in range(n_points)]
    result.sweep_values = gors

    # Get base case
    base_kpis = _get_kpis(model, response_kpis)
    base_conds = _extract_stream_conditions(model, feed_stream)
    result.base_case = WellCase(name="Base Case", kpis=base_kpis, stream_conditions=base_conds)

    for gor_val in gors:
        clone = model.clone()
        try:
            js = clone.get_stream(feed_stream)
            fl = js.getFluid()

            # Classify components as gas or oil
            gas_indices = []
            oil_indices = []
            gas_lower = [g.lower() for g in gas_components]

            for i in range(fl.getNumberOfComponents()):
                cname = str(fl.getComponent(i).getName()).lower()
                if cname in gas_lower or cname in ("methane", "ethane", "propane",
                                                    "co2", "nitrogen", "h2s"):
                    gas_indices.append(i)
                elif cname != "water" and cname not in ("meg", "teg", "methanol"):
                    oil_indices.append(i)

            if not gas_indices or not oil_indices:
                result.cases.append(WellCase(
                    name=f"GOR={gor_val:.0f}",
                    feasible=False,
                    error="Cannot identify gas and oil components in feed",
                ))
                continue

            # Scale gas components relative to oil to achieve target GOR
            # GOR ~ sum(gas_moles) / sum(oil_moles) (simplified)
            total_oil = sum(float(fl.getComponent(i).getz()) for i in oil_indices)
            total_gas = sum(float(fl.getComponent(i).getz()) for i in gas_indices)

            if total_gas <= 0 or total_oil <= 0:
                result.cases.append(WellCase(
                    name=f"GOR={gor_val:.0f}",
                    feasible=False,
                    error="Zero gas or oil fraction in feed",
                ))
                continue

            current_gor = total_gas / total_oil
            if current_gor <= 0:
                continue

            scale_factor = gor_val / current_gor
            for i in gas_indices:
                old_z = float(fl.getComponent(i).getz())
                fl.getComponent(i).setz(old_z * scale_factor)

            fl.init(0)

            kpis = _get_kpis(clone, response_kpis)
            conds = _extract_stream_conditions(clone, feed_stream)

            result.cases.append(WellCase(
                name=f"GOR={gor_val:.0f}",
                feed_composition={"GOR": gor_val},
                kpis=kpis,
                stream_conditions=conds,
                feasible=True,
            ))
        except Exception as e:
            result.cases.append(WellCase(
                name=f"GOR={gor_val:.0f}",
                feasible=False,
                error=str(e),
            ))

    result.message = f"GOR sweep from {min_gor:.0f} to {max_gor:.0f} Sm3/Sm3 ({n_points} points)"
    return result


# ---------------------------------------------------------------------------
# Water cut sweep
# ---------------------------------------------------------------------------

def _run_watercut_sweep(
    model: NeqSimProcessModel,
    feed_stream: str,
    min_watercut_pct: float,
    max_watercut_pct: float,
    n_points: int,
    response_kpis: Optional[List[str]] = None,
) -> ProductionScenarioResult:
    """Sweep water cut (vol%) by adding water to the feed."""
    if response_kpis is None:
        response_kpis = ["power", "duty", "flow"]

    result = ProductionScenarioResult(
        scenario_type="watercut_sweep",
        sweep_variable="Water cut (%)",
        kpi_names=response_kpis,
    )

    wcuts = [min_watercut_pct + (max_watercut_pct - min_watercut_pct) * i / max(n_points - 1, 1)
             for i in range(n_points)]
    result.sweep_values = wcuts

    base_kpis = _get_kpis(model, response_kpis)
    base_conds = _extract_stream_conditions(model, feed_stream)
    result.base_case = WellCase(name="Base Case", kpis=base_kpis, stream_conditions=base_conds)

    for wc in wcuts:
        clone = model.clone()
        try:
            js = clone.get_stream(feed_stream)
            fl = js.getFluid()

            # Get total HC moles (everything except water)
            total_hc = 0.0
            water_idx = -1
            for i in range(fl.getNumberOfComponents()):
                cname = str(fl.getComponent(i).getName()).lower()
                if cname == "water":
                    water_idx = i
                else:
                    total_hc += float(fl.getComponent(i).getz())

            # Water cut as mole fraction approximation
            # wc% ≈ water_moles / (water_moles + hc_moles) * 100
            wc_clamped = min(wc, 99.9)
            water_frac = (wc_clamped / 100.0) * total_hc / (1.0 - wc_clamped / 100.0)

            if water_idx >= 0:
                fl.getComponent(water_idx).setz(water_frac)
            else:
                fl.addComponent("water", water_frac)
                fl.setMixingRule(fl.getMixingRuleNumber())

            fl.init(0)

            kpis = _get_kpis(clone, response_kpis)
            conds = _extract_stream_conditions(clone, feed_stream)

            result.cases.append(WellCase(
                name=f"WC={wc:.1f}%",
                feed_composition={"water_cut_pct": wc},
                kpis=kpis,
                stream_conditions=conds,
                feasible=True,
            ))
        except Exception as e:
            result.cases.append(WellCase(
                name=f"WC={wc:.1f}%",
                feasible=False,
                error=str(e),
            ))

    result.message = f"Water cut sweep from {min_watercut_pct:.1f}% to {max_watercut_pct:.1f}% ({n_points} points)"
    return result


# ---------------------------------------------------------------------------
# Multi-well blending
# ---------------------------------------------------------------------------

def _run_well_blend(
    model: NeqSimProcessModel,
    feed_stream: str,
    wells: List[Dict[str, Any]],
    response_kpis: Optional[List[str]] = None,
) -> ProductionScenarioResult:
    """Evaluate the process with different well-feed compositions.
    
    Each well dict: {"name": "Well A", "composition": {"methane": 0.85, ...}, "fraction": 0.5}
    If fraction is given, blends multiple wells. Otherwise runs each independently.
    """
    if response_kpis is None:
        response_kpis = ["power", "duty", "flow"]

    result = ProductionScenarioResult(
        scenario_type="well_blend",
        kpi_names=response_kpis,
    )

    base_kpis = _get_kpis(model, response_kpis)
    base_conds = _extract_stream_conditions(model, feed_stream)
    result.base_case = WellCase(name="Base Case", kpis=base_kpis, stream_conditions=base_conds)

    # Check if we should blend or run independently
    has_fractions = any("fraction" in w for w in wells)

    if has_fractions:
        # Blend mode: combine compositions by fraction
        blended: Dict[str, float] = {}
        for w in wells:
            frac = w.get("fraction", 1.0 / len(wells))
            for comp, mol_frac in w.get("composition", {}).items():
                blended[comp] = blended.get(comp, 0.0) + mol_frac * frac

        clone = model.clone()
        try:
            js = clone.get_stream(feed_stream)
            fl = js.getFluid()

            for comp, z in blended.items():
                for i in range(fl.getNumberOfComponents()):
                    if str(fl.getComponent(i).getName()).lower() == comp.lower():
                        fl.getComponent(i).setz(float(z))
                        break
            fl.init(0)

            kpis = _get_kpis(clone, response_kpis)
            conds = _extract_stream_conditions(clone, feed_stream)
            well_names = [w.get("name", f"Well {i+1}") for i, w in enumerate(wells)]

            result.cases.append(WellCase(
                name=f"Blend: {' + '.join(well_names)}",
                feed_composition=blended,
                kpis=kpis,
                stream_conditions=conds,
                feasible=True,
            ))
        except Exception as e:
            result.cases.append(WellCase(
                name="Blend",
                feasible=False,
                error=str(e),
            ))
    else:
        # Independent mode: evaluate each well composition separately
        for idx, w in enumerate(wells):
            well_name = w.get("name", f"Well {idx+1}")
            comp = w.get("composition", {})

            clone = model.clone()
            try:
                js = clone.get_stream(feed_stream)
                fl = js.getFluid()

                for cname, z in comp.items():
                    for i in range(fl.getNumberOfComponents()):
                        if str(fl.getComponent(i).getName()).lower() == cname.lower():
                            fl.getComponent(i).setz(float(z))
                            break
                fl.init(0)

                kpis = _get_kpis(clone, response_kpis)
                conds = _extract_stream_conditions(clone, feed_stream)

                result.cases.append(WellCase(
                    name=well_name,
                    description=w.get("description", ""),
                    feed_composition=comp,
                    kpis=kpis,
                    stream_conditions=conds,
                    feasible=True,
                ))
            except Exception as e:
                result.cases.append(WellCase(
                    name=well_name,
                    feasible=False,
                    error=str(e),
                ))

    result.message = f"Well blend analysis with {len(wells)} well(s)"
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_production_scenario(
    model: NeqSimProcessModel,
    scenario_type: str = "composition_sweep",
    feed_stream: Optional[str] = None,
    component: Optional[str] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    n_points: int = 8,
    response_kpis: Optional[List[str]] = None,
    gas_components: Optional[List[str]] = None,
    oil_components: Optional[List[str]] = None,
    wells: Optional[List[Dict[str, Any]]] = None,
) -> ProductionScenarioResult:
    """Run a production / reservoir fluid scenario study.

    Args:
        scenario_type: "composition_sweep", "gor_sweep", "watercut_sweep", "well_blend"
        feed_stream: Name of the feed stream (auto-detected if None)
        component: Component to sweep (for composition_sweep)
        min_value / max_value: Sweep range
        n_points: Number of sweep points
        response_kpis: KPIs to track (fuzzy-matched)
        gas_components / oil_components: For GOR sweep classification
        wells: Well definitions for blending study
    """
    fs = _find_feed_stream(model, feed_stream)
    if fs is None:
        r = ProductionScenarioResult()
        r.message = "No feed stream found in the process."
        return r

    if response_kpis is None:
        response_kpis = ["power", "duty", "flow", "temperature"]

    if scenario_type == "composition_sweep":
        if not component:
            return ProductionScenarioResult(message="Component name required for composition sweep.")
        return _run_composition_sweep(
            model, fs, component,
            min_value or 0.0, max_value or 1.0, n_points, response_kpis,
        )

    elif scenario_type == "gor_sweep":
        return _run_gor_sweep(
            model, fs,
            min_value or 50.0, max_value or 500.0, n_points,
            gas_components, oil_components, response_kpis,
        )

    elif scenario_type == "watercut_sweep":
        return _run_watercut_sweep(
            model, fs,
            min_value or 0.0, max_value or 50.0, n_points, response_kpis,
        )

    elif scenario_type == "well_blend":
        if not wells:
            return ProductionScenarioResult(message="Well definitions required for blending study.")
        return _run_well_blend(model, fs, wells, response_kpis)

    else:
        return ProductionScenarioResult(message=f"Unknown scenario type: {scenario_type}")


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

def format_production_scenario_result(result: ProductionScenarioResult) -> str:
    """Format result for LLM consumption."""
    lines = [f"=== PRODUCTION SCENARIO: {result.scenario_type.upper()} ==="]
    lines.append(result.message)
    lines.append("")

    if result.base_case:
        lines.append("BASE CASE:")
        for k, v in result.base_case.kpis.items():
            lines.append(f"  {k}: {v:.2f}")
        lines.append("")

    # Collect all KPI keys across cases
    all_kpi_keys = set()
    for c in result.cases:
        all_kpi_keys.update(c.kpis.keys())
    kpi_keys = sorted(all_kpi_keys)

    # Table header
    header = f"{'Case':<25}"
    for k in kpi_keys:
        short = k.split(".")[-1] if "." in k else k
        header += f"  {short:>15}"
    lines.append(header)
    lines.append("-" * len(header))

    for c in result.cases:
        if c.feasible:
            row = f"{c.name:<25}"
            for k in kpi_keys:
                val = c.kpis.get(k)
                row += f"  {val:>15.2f}" if val is not None else f"  {'N/A':>15}"
            lines.append(row)
        else:
            lines.append(f"{c.name:<25}  INFEASIBLE: {c.error}")

    # Delta summary vs base
    if result.base_case and result.cases:
        lines.append("")
        lines.append("DELTA vs BASE:")
        for c in result.cases:
            if not c.feasible:
                continue
            deltas = []
            for k in kpi_keys:
                base_v = result.base_case.kpis.get(k)
                case_v = c.kpis.get(k)
                if base_v is not None and case_v is not None and base_v != 0:
                    pct = (case_v - base_v) / abs(base_v) * 100
                    deltas.append(f"{k.split('.')[-1]}={pct:+.1f}%")
            if deltas:
                lines.append(f"  {c.name}: {', '.join(deltas)}")

    return "\n".join(lines)
