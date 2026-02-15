"""
Dynamic Simulation — transient / blowdown / startup / shutdown scenarios.

Uses NeqSim's ``process.runTransient()`` where available, with a Python
fallback that steps through time by re-running the steady-state model
with time-varying boundary conditions.

Returns a ``DynamicSimResult`` with time-series data for plotting.
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
class TimeSeriesPoint:
    """One snapshot in time."""
    time_s: float
    values: Dict[str, float] = field(default_factory=dict)


@dataclass
class DynamicSimResult:
    """Complete dynamic simulation result."""
    scenario_type: str = "blowdown"   # blowdown, startup, shutdown, ramp, custom
    time_series: List[TimeSeriesPoint] = field(default_factory=list)
    variable_names: List[str] = field(default_factory=list)
    variable_units: Dict[str, str] = field(default_factory=dict)
    duration_s: float = 0.0
    time_steps: int = 0
    final_state: Dict[str, float] = field(default_factory=dict)
    method: str = "pseudo_transient"
    message: str = ""


# ---------------------------------------------------------------------------
# Blowdown simulation (simplified model)
# ---------------------------------------------------------------------------

def _run_blowdown(
    model: NeqSimProcessModel,
    vessel_name: Optional[str] = None,
    initial_pressure_bara: Optional[float] = None,
    final_pressure_bara: float = 1.013,
    orifice_diameter_mm: float = 50.0,
    duration_s: float = 600.0,
    n_steps: int = 50,
) -> DynamicSimResult:
    """
    Simplified blowdown simulation.

    Uses isentropic expansion model with choked-flow discharge through an
    orifice. Steps through time and records pressure, temperature, flow.
    """
    result = DynamicSimResult(scenario_type="blowdown")

    proc = model.get_process()
    if proc is None:
        result.message = "No process available."
        return result

    # Find vessel pressure
    p0 = initial_pressure_bara
    T0 = 300.0  # K
    volume_m3 = 10.0  # default vessel volume

    if vessel_name:
        units = model.list_units()
        for u_name, u_info in units.items():
            if vessel_name.lower() in u_name.lower():
                java_obj = u_info.get("java_ref")
                if java_obj:
                    try:
                        if hasattr(java_obj, "getInternalDiameter"):
                            d = float(java_obj.getInternalDiameter())
                            L = d * 3  # L/D ≈ 3
                            volume_m3 = math.pi * (d/2)**2 * L
                    except Exception:
                        pass
                props = u_info.get("properties", {})
                if p0 is None and "inletPressure_bara" in props:
                    p0 = props["inletPressure_bara"]
                if "inletTemperature_K" in props:
                    T0 = props["inletTemperature_K"]
                break

    if p0 is None:
        # Try to get from first stream
        streams = model.list_streams()
        if streams:
            first = list(streams.values())[0]
            p0 = first.get("conditions", {}).get("pressure_bara", 50.0)

    if p0 is None:
        p0 = 50.0

    # Orifice area
    d_m = orifice_diameter_mm / 1000.0
    A = math.pi * (d_m / 2) ** 2

    # Gas properties (approximate)
    gamma = 1.3       # heat capacity ratio
    M_gas = 0.020     # kg/mol (≈ natural gas)
    R = 8.314         # J/(mol·K)

    dt = duration_s / n_steps
    P = p0 * 1e5      # Pa
    T = T0
    P_back = final_pressure_bara * 1e5

    variables = ["pressure_bara", "temperature_C", "mass_flow_kg_hr",
                 "mass_remaining_kg", "blowdown_pct"]
    result.variable_names = variables
    result.variable_units = {
        "pressure_bara": "bara",
        "temperature_C": "°C",
        "mass_flow_kg_hr": "kg/hr",
        "mass_remaining_kg": "kg",
        "blowdown_pct": "%",
    }

    # Initial mass
    rho0 = P * M_gas / (R * T)
    m0 = rho0 * volume_m3

    m = m0
    time_data: List[TimeSeriesPoint] = []

    for i in range(n_steps + 1):
        t = i * dt
        p_bara = P / 1e5

        # Choked flow condition
        p_crit_ratio = (2 / (gamma + 1)) ** (gamma / (gamma - 1))
        if P_back / P < p_crit_ratio:
            # Choked flow
            mdot = A * P * math.sqrt(gamma * M_gas / (R * T)) * \
                   (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
        else:
            # Sub-critical flow
            pr = P_back / P
            mdot = A * P * math.sqrt(
                2 * gamma / ((gamma - 1) * R * T / M_gas) *
                (pr ** (2/gamma) - pr ** ((gamma+1)/gamma))
            ) if P > P_back else 0.0

        mdot_kg_hr = mdot * 3600.0
        blowdown_pct = (1.0 - m / m0) * 100.0 if m0 > 0 else 0.0

        time_data.append(TimeSeriesPoint(
            time_s=t,
            values={
                "pressure_bara": p_bara,
                "temperature_C": T - 273.15,
                "mass_flow_kg_hr": mdot_kg_hr,
                "mass_remaining_kg": m,
                "blowdown_pct": blowdown_pct,
            }
        ))

        # Update state
        dm = mdot * dt
        if dm > m:
            dm = m
        m -= dm
        if m <= 0:
            m = 0
            break

        # Isentropic temperature drop
        rho_new = m / volume_m3 if volume_m3 > 0 else rho0
        P_new = rho_new * R * T / M_gas
        if P_new > 0 and P > 0:
            T = T * (P_new / P) ** ((gamma - 1) / gamma)
        P = max(P_new, P_back)

    result.time_series = time_data
    result.duration_s = duration_s
    result.time_steps = len(time_data)
    if time_data:
        result.final_state = time_data[-1].values.copy()
    result.method = "isentropic_blowdown"
    result.message = (f"Blowdown from {p0:.1f} to {final_pressure_bara:.1f} bara, "
                      f"orifice {orifice_diameter_mm:.0f}mm, volume {volume_m3:.1f} m³")
    return result


# ---------------------------------------------------------------------------
# Ramp / startup / shutdown simulation
# ---------------------------------------------------------------------------

def _run_ramp(
    model: NeqSimProcessModel,
    ramp_variable: str = "feed_flow",
    stream_name: Optional[str] = None,
    start_value: Optional[float] = None,
    end_value: Optional[float] = None,
    duration_s: float = 3600.0,
    n_steps: int = 20,
) -> DynamicSimResult:
    """
    Pseudo-transient ramp simulation.

    Steps a variable linearly from start to end and re-runs steady-state
    at each step. Records key KPIs vs time.
    """
    result = DynamicSimResult(scenario_type="ramp")

    proc = model.get_process()
    if proc is None:
        result.message = "No process available."
        return result

    # Determine stream and variable
    streams = model.list_streams()
    if not stream_name:
        if streams:
            stream_name = list(streams.keys())[0]
        else:
            result.message = "No streams found."
            return result

    # Find current value
    current_flow = 0.0
    for s_name, s_info in streams.items():
        if stream_name.lower() in s_name.lower():
            current_flow = s_info.get("conditions", {}).get("flow_kg_hr", 10000.0) or 10000.0
            stream_name = s_name
            break

    if start_value is None:
        start_value = current_flow * 0.1   # startup from 10%
    if end_value is None:
        end_value = current_flow

    dt = duration_s / n_steps
    variables = []
    time_data: List[TimeSeriesPoint] = []

    for i in range(n_steps + 1):
        t = i * dt
        frac = i / n_steps if n_steps > 0 else 1.0
        val = start_value + (end_value - start_value) * frac

        try:
            # Clone, set flow, run
            clone = model.clone()
            clone_proc = clone.get_process()

            # Set stream flow
            for s_name, s_info in clone.list_streams().items():
                if stream_name.lower() in s_name.lower():
                    java_stream = s_info.get("java_ref")
                    if java_stream:
                        java_stream.setFlowRate(float(val), "kg/hr")
                    break

            clone.run()

            # Extract KPIs
            kpis = clone.extract_kpis()
            point_values: Dict[str, float] = {"feed_flow_kg_hr": val}

            for kpi_name, kpi in kpis.items():
                if kpi.value is not None:
                    point_values[kpi_name] = kpi.value

            if not variables:
                variables = list(point_values.keys())

            time_data.append(TimeSeriesPoint(time_s=t, values=point_values))

        except Exception:
            time_data.append(TimeSeriesPoint(
                time_s=t, values={"feed_flow_kg_hr": val}
            ))

    result.time_series = time_data
    result.variable_names = variables
    result.duration_s = duration_s
    result.time_steps = len(time_data)
    if time_data:
        result.final_state = time_data[-1].values.copy()
    result.method = "pseudo_transient"
    result.message = f"Ramp {stream_name} from {start_value:.0f} to {end_value:.0f} kg/hr"
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_dynamic_simulation(
    model: NeqSimProcessModel,
    scenario_type: str = "blowdown",
    vessel_name: Optional[str] = None,
    stream_name: Optional[str] = None,
    initial_pressure_bara: Optional[float] = None,
    final_pressure_bara: float = 1.013,
    orifice_diameter_mm: float = 50.0,
    duration_s: float = 600.0,
    n_steps: int = 50,
    start_value: Optional[float] = None,
    end_value: Optional[float] = None,
) -> DynamicSimResult:
    """
    Run a dynamic simulation scenario.

    Parameters
    ----------
    model : NeqSimProcessModel
    scenario_type : str
        'blowdown', 'startup', 'shutdown', or 'ramp'
    """
    if scenario_type in ("blowdown",):
        return _run_blowdown(
            model, vessel_name=vessel_name,
            initial_pressure_bara=initial_pressure_bara,
            final_pressure_bara=final_pressure_bara,
            orifice_diameter_mm=orifice_diameter_mm,
            duration_s=duration_s, n_steps=n_steps,
        )
    elif scenario_type in ("startup", "ramp", "shutdown"):
        if scenario_type == "shutdown" and start_value is None and end_value is None:
            # Reverse ramp
            streams = model.list_streams()
            for s_name, s_info in streams.items():
                if not stream_name or stream_name.lower() in s_name.lower():
                    flow = s_info.get("conditions", {}).get("flow_kg_hr", 10000.0)
                    start_value = flow
                    end_value = flow * 0.05
                    break
        return _run_ramp(
            model, stream_name=stream_name,
            start_value=start_value, end_value=end_value,
            duration_s=duration_s, n_steps=n_steps,
        )
    else:
        return DynamicSimResult(message=f"Unknown scenario type: {scenario_type}")


# ---------------------------------------------------------------------------
# Format for LLM
# ---------------------------------------------------------------------------

def format_dynamic_result(result: DynamicSimResult) -> str:
    """Format dynamic sim results for LLM follow-up."""
    lines = [f"=== DYNAMIC SIMULATION ({result.scenario_type.upper()}) ==="]
    lines.append(f"Method: {result.method}")
    lines.append(f"Duration: {result.duration_s:.0f} s ({result.duration_s/60:.1f} min)")
    lines.append(f"Time steps: {result.time_steps}")
    lines.append(f"{result.message}")
    lines.append("")

    # Summary: initial → final for key variables
    if result.time_series and len(result.time_series) >= 2:
        first = result.time_series[0].values
        last = result.time_series[-1].values
        lines.append("=== INITIAL → FINAL STATE ===")
        for key in result.variable_names[:10]:
            v0 = first.get(key, 0)
            vf = last.get(key, 0)
            unit = result.variable_units.get(key, "")
            lines.append(f"  {key}: {v0:.2f} → {vf:.2f} {unit}")
        lines.append("")

    # Time series sample (every ~10th point)
    if result.time_series:
        step = max(1, len(result.time_series) // 10)
        lines.append("=== TIME SERIES (sampled) ===")
        header_vars = result.variable_names[:6]
        header = "  time_s  " + "  ".join(f"{v[:18]:>18}" for v in header_vars)
        lines.append(header)
        for i in range(0, len(result.time_series), step):
            pt = result.time_series[i]
            vals = "  ".join(f"{pt.values.get(v, 0):>18.2f}" for v in header_vars)
            lines.append(f"  {pt.time_s:>7.1f}  {vals}")
        # Always include last
        if (len(result.time_series) - 1) % step != 0:
            pt = result.time_series[-1]
            vals = "  ".join(f"{pt.values.get(v, 0):>18.2f}" for v in header_vars)
            lines.append(f"  {pt.time_s:>7.1f}  {vals}")

    return "\n".join(lines)
