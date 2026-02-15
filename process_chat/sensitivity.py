"""
Batch Sensitivity Analysis — sweep one or two variables and record KPI responses.

Supports:
  - Single-variable sweep (tornado charts)
  - Two-variable sweep (surface / heatmap)
  - Multi-variable Latin Hypercube sampling

Returns a ``SensitivityResult`` with sweep data for plotting.
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
class SweepPoint:
    """One point in a parameter sweep."""
    input_values: Dict[str, float] = field(default_factory=dict)
    output_values: Dict[str, float] = field(default_factory=dict)
    feasible: bool = True
    error: str = ""


@dataclass
class TornadoBar:
    """One bar in a tornado chart (high/low impact of one variable)."""
    variable: str
    low_value: float
    high_value: float
    kpi_at_low: float
    kpi_at_high: float
    kpi_base: float
    delta_low: float = 0.0
    delta_high: float = 0.0


@dataclass
class SensitivityResult:
    """Complete sensitivity analysis result."""
    analysis_type: str = "single_sweep"  # single_sweep, tornado, two_variable, lhs
    sweep_variable: str = ""
    sweep_variable_2: str = ""
    response_kpis: List[str] = field(default_factory=list)
    sweep_points: List[SweepPoint] = field(default_factory=list)
    tornado_bars: List[TornadoBar] = field(default_factory=list)
    base_values: Dict[str, float] = field(default_factory=dict)
    n_points: int = 0
    method: str = "clone_and_run"
    message: str = ""


# ---------------------------------------------------------------------------
# Patch key application
# ---------------------------------------------------------------------------

def _apply_patch_key(model: NeqSimProcessModel, key: str, value: float) -> bool:
    """Apply a single patch key to a model. Returns True on success."""
    try:
        parts = key.split(".", 2)
        if len(parts) < 2:
            return False

        if parts[0] == "streams":
            stream_name = parts[1] if len(parts) >= 2 else ""
            param = parts[2] if len(parts) > 2 else ""
            streams = model.list_streams()
            for s_info in streams:
                if stream_name.lower() in s_info.name.lower():
                    try:
                        java_stream = model.get_stream(s_info.name)
                    except KeyError:
                        return False
                    if "pressure_bara" in param:
                        java_stream.setPressure(float(value), "bara")
                    elif "temperature_C" in param:
                        java_stream.setTemperature(float(value), "C")
                    elif "flow_kg_hr" in param:
                        java_stream.setFlowRate(float(value), "kg/hr")
                    else:
                        return False
                    return True
            return False

        elif parts[0] == "units":
            unit_name = parts[1] if len(parts) >= 2 else ""
            param = parts[2] if len(parts) > 2 else ""
            units = model.list_units()
            for u_info in units:
                if unit_name.lower() in u_info.name.lower():
                    try:
                        java_obj = model.get_unit(u_info.name)
                    except KeyError:
                        return False
                    if "outletpressure_bara" in param or "outpressure_bara" in param:
                        java_obj.setOutletPressure(float(value))
                    elif "outtemperature_c" in param.lower():
                        java_obj.setOutTemperature(float(value), "C")
                    elif "isentropicefficiency" in param.lower():
                        java_obj.setIsentropicEfficiency(float(value))
                    elif "speed" in param.lower():
                        java_obj.setSpeed(float(value))
                    else:
                        return False
                    return True
            return False

        return False
    except Exception:
        return False


def _extract_kpi_value(model: NeqSimProcessModel, kpi_name: str) -> Optional[float]:
    """Extract a single KPI value by name."""
    run_result = model.run()
    kpis = run_result.kpis
    # Exact match
    if kpi_name in kpis:
        return kpis[kpi_name].value
    # Fuzzy match
    kpi_lower = kpi_name.lower()
    for k, v in kpis.items():
        if kpi_lower in k.lower():
            return v.value
    return None


# ---------------------------------------------------------------------------
# Single-variable sweep
# ---------------------------------------------------------------------------

def _run_single_sweep(
    model: NeqSimProcessModel,
    variable: str,
    min_value: float,
    max_value: float,
    n_points: int,
    response_kpis: List[str],
) -> SensitivityResult:
    """Sweep one variable linearly and record KPI responses."""
    result = SensitivityResult(
        analysis_type="single_sweep",
        sweep_variable=variable,
        response_kpis=response_kpis,
    )

    # Get base KPIs
    try:
        run_result = model.run()
        base_kpis = run_result.kpis
        for kpi in response_kpis:
            for k, v in base_kpis.items():
                if kpi.lower() in k.lower() and v.value is not None:
                    result.base_values[k] = v.value
    except Exception:
        pass

    values = [min_value + (max_value - min_value) * i / max(n_points - 1, 1)
              for i in range(n_points)]

    for val in values:
        try:
            clone = model.clone()
            ok = _apply_patch_key(clone, variable, val)
            if not ok:
                result.sweep_points.append(SweepPoint(
                    input_values={variable: val},
                    feasible=False,
                    error=f"Could not apply {variable}={val}",
                ))
                continue

            run_result = clone.run()
            kpis = run_result.kpis

            output: Dict[str, float] = {}
            for kpi_name in response_kpis:
                for k, v in kpis.items():
                    if kpi_name.lower() in k.lower() and v.value is not None:
                        output[k] = v.value

            result.sweep_points.append(SweepPoint(
                input_values={variable: val},
                output_values=output,
                feasible=True,
            ))
        except Exception as e:
            result.sweep_points.append(SweepPoint(
                input_values={variable: val},
                feasible=False,
                error=str(e),
            ))

    result.n_points = len(result.sweep_points)
    result.message = f"Single sweep of {variable}: {min_value} → {max_value}, {n_points} points"
    return result


# ---------------------------------------------------------------------------
# Tornado analysis
# ---------------------------------------------------------------------------

def _run_tornado(
    model: NeqSimProcessModel,
    variables: List[Dict[str, Any]],
    response_kpi: str,
) -> SensitivityResult:
    """
    Tornado sensitivity: for each variable, evaluate at low and high
    bound while other variables stay at base.

    variables: [{"name": "streams.feed.pressure_bara", "low": 40, "high": 60}, ...]
    """
    result = SensitivityResult(
        analysis_type="tornado",
        response_kpis=[response_kpi],
    )

    # Get base
    try:
        model.run()
        kpi_base = _extract_kpi_value(model, response_kpi)
    except Exception:
        kpi_base = None

    if kpi_base is None:
        result.message = f"Could not extract base KPI: {response_kpi}"
        return result

    result.base_values[response_kpi] = kpi_base

    bars: List[TornadoBar] = []
    for var in variables:
        name = var["name"]
        low = var["low"]
        high = var["high"]

        kpi_low = kpi_base
        kpi_high = kpi_base

        # Low case
        try:
            clone = model.clone()
            _apply_patch_key(clone, name, low)
            clone.run()
            v = _extract_kpi_value(clone, response_kpi)
            if v is not None:
                kpi_low = v
        except Exception:
            pass

        # High case
        try:
            clone = model.clone()
            _apply_patch_key(clone, name, high)
            clone.run()
            v = _extract_kpi_value(clone, response_kpi)
            if v is not None:
                kpi_high = v
        except Exception:
            pass

        bars.append(TornadoBar(
            variable=name,
            low_value=low,
            high_value=high,
            kpi_at_low=kpi_low,
            kpi_at_high=kpi_high,
            kpi_base=kpi_base,
            delta_low=kpi_low - kpi_base,
            delta_high=kpi_high - kpi_base,
        ))

    # Sort by total impact
    bars.sort(key=lambda b: abs(b.delta_high - b.delta_low), reverse=True)
    result.tornado_bars = bars
    result.n_points = len(bars)
    result.message = f"Tornado analysis: {len(bars)} variables vs {response_kpi}"
    return result


# ---------------------------------------------------------------------------
# Two-variable sweep
# ---------------------------------------------------------------------------

def _run_two_variable_sweep(
    model: NeqSimProcessModel,
    variable_1: str,
    min_1: float,
    max_1: float,
    n_1: int,
    variable_2: str,
    min_2: float,
    max_2: float,
    n_2: int,
    response_kpis: List[str],
) -> SensitivityResult:
    """Sweep two variables on a grid and record KPI responses."""
    result = SensitivityResult(
        analysis_type="two_variable",
        sweep_variable=variable_1,
        sweep_variable_2=variable_2,
        response_kpis=response_kpis,
    )

    vals_1 = [min_1 + (max_1 - min_1) * i / max(n_1 - 1, 1) for i in range(n_1)]
    vals_2 = [min_2 + (max_2 - min_2) * i / max(n_2 - 1, 1) for i in range(n_2)]

    for v1 in vals_1:
        for v2 in vals_2:
            try:
                clone = model.clone()
                _apply_patch_key(clone, variable_1, v1)
                _apply_patch_key(clone, variable_2, v2)
                run_result = clone.run()
                kpis = run_result.kpis

                output: Dict[str, float] = {}
                for kpi_name in response_kpis:
                    for k, v in kpis.items():
                        if kpi_name.lower() in k.lower() and v.value is not None:
                            output[k] = v.value

                result.sweep_points.append(SweepPoint(
                    input_values={variable_1: v1, variable_2: v2},
                    output_values=output,
                    feasible=True,
                ))
            except Exception as e:
                result.sweep_points.append(SweepPoint(
                    input_values={variable_1: v1, variable_2: v2},
                    feasible=False,
                    error=str(e),
                ))

    result.n_points = len(result.sweep_points)
    result.message = f"Two-variable sweep: {variable_1} × {variable_2}, {n_1}×{n_2} grid"
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_sensitivity_analysis(
    model: NeqSimProcessModel,
    analysis_type: str = "single_sweep",
    variable: Optional[str] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    n_points: int = 10,
    response_kpis: Optional[List[str]] = None,
    variables: Optional[List[Dict[str, Any]]] = None,
    response_kpi: Optional[str] = None,
    variable_2: Optional[str] = None,
    min_2: Optional[float] = None,
    max_2: Optional[float] = None,
    n_2: Optional[int] = None,
) -> SensitivityResult:
    """
    Run a sensitivity analysis.

    Parameters
    ----------
    model : NeqSimProcessModel
    analysis_type : str
        'single_sweep', 'tornado', or 'two_variable'
    """
    if response_kpis is None:
        response_kpis = ["power", "duty", "temperature", "pressure"]

    if analysis_type == "tornado" and variables:
        return _run_tornado(model, variables, response_kpi or response_kpis[0])

    elif analysis_type == "two_variable" and variable and variable_2:
        return _run_two_variable_sweep(
            model, variable,
            min_value or 0, max_value or 100, n_points,
            variable_2,
            min_2 or 0, max_2 or 100, n_2 or n_points,
            response_kpis,
        )

    elif variable:
        return _run_single_sweep(
            model, variable,
            min_value or 0, max_value or 100,
            n_points, response_kpis,
        )

    else:
        return SensitivityResult(message="No variable specified for sensitivity analysis.")


# ---------------------------------------------------------------------------
# Format for LLM
# ---------------------------------------------------------------------------

def format_sensitivity_result(result: SensitivityResult) -> str:
    """Format sensitivity results for the LLM follow-up."""
    lines = [f"=== SENSITIVITY ANALYSIS ({result.analysis_type.upper()}) ==="]
    lines.append(f"Points: {result.n_points}")
    lines.append(result.message)
    lines.append("")

    if result.analysis_type == "tornado" and result.tornado_bars:
        lines.append("=== TORNADO RESULTS ===")
        lines.append(f"Base KPI value: {list(result.base_values.values())[0]:.2f}" if result.base_values else "")
        for bar in result.tornado_bars:
            span = abs(bar.delta_high - bar.delta_low)
            lines.append(f"  {bar.variable}: "
                         f"low({bar.low_value:.1f})→{bar.kpi_at_low:.2f}, "
                         f"high({bar.high_value:.1f})→{bar.kpi_at_high:.2f} "
                         f"[span={span:.2f}]")

    elif result.sweep_points:
        lines.append("=== SWEEP DATA ===")
        # Show header
        if result.sweep_points:
            first = result.sweep_points[0]
            in_keys = list(first.input_values.keys())
            out_keys = list(first.output_values.keys())[:5]
            header = "  ".join(f"{k[:20]:>20}" for k in in_keys + out_keys)
            lines.append(f"  {header}")
            for pt in result.sweep_points:
                if pt.feasible:
                    vals = [pt.input_values.get(k, 0) for k in in_keys]
                    vals += [pt.output_values.get(k, 0) for k in out_keys]
                    row = "  ".join(f"{v:>20.2f}" for v in vals)
                    lines.append(f"  {row}")
                else:
                    vals = [pt.input_values.get(k, 0) for k in in_keys]
                    row = "  ".join(f"{v:>20.2f}" for v in vals)
                    lines.append(f"  {row}  [FAILED: {pt.error}]")

    return "\n".join(lines)
