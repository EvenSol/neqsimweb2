"""
Dynamic Simulation — transient / blowdown / startup / shutdown scenarios.

Uses NeqSim's ``process.runTransient()`` where available, with a Python
fallback that steps through time by re-running the steady-state model
with time-varying boundary conditions.

Returns a ``DynamicSimResult`` with time-series data for plotting.
"""
from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .process_model import NeqSimProcessModel

logger = logging.getLogger(__name__)


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
        for u_info in units:
            if vessel_name.lower() in u_info.name.lower():
                try:
                    java_obj = model.get_unit(u_info.name)
                    if hasattr(java_obj, "getInternalDiameter"):
                        try:
                            d = float(java_obj.getInternalDiameter())
                            L = d * 3  # L/D ≈ 3
                            volume_m3 = math.pi * (d/2)**2 * L
                        except Exception:
                            pass
                    # Read T and P from the vessel's inlet stream
                    for m_name in ("getInletStream", "getInStream", "getFeed"):
                        if hasattr(java_obj, m_name):
                            try:
                                inlet = getattr(java_obj, m_name)()
                                if inlet is not None:
                                    if p0 is None:
                                        p0 = float(inlet.getPressure("bara"))
                                    T0 = float(inlet.getTemperature("K"))
                                    break
                            except Exception:
                                pass
                except KeyError:
                    pass
                break

    if p0 is None:
        # Try to get from first stream
        streams = model.list_streams()
        if streams:
            first = streams[0]
            p0 = first.pressure_bara if first.pressure_bara is not None else 50.0

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

        # Isentropic expansion: update T and P from density ratio
        rho_new = m / volume_m3 if volume_m3 > 0 else rho0
        rho_old = (m + dm) / volume_m3 if volume_m3 > 0 else rho0
        if rho_old > 0 and rho_new > 0:
            ratio = rho_new / rho_old
            P_new = P * ratio ** gamma
            T = T * ratio ** (gamma - 1)
        else:
            P_new = rho_new * R * T / M_gas
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
# True transient simulation using NeqSim's runTransient()
# ---------------------------------------------------------------------------

# Equipment classes that support transient holdup tracking
_DYNAMIC_CLASSES = frozenset({
    "Separator", "ThreePhaseSeparator", "TwoPhaseSeparator",
    "GasScrubber", "GasScrubberSimple", "Tank",
})

# Equipment classes that are valves (Cv-based flow in transient)
_VALVE_CLASSES = frozenset({
    "ThrottlingValve", "ControlValve", "ESDValve", "PSDValve",
    "PressureControlValve", "LevelControlValve", "CheckValve",
    "BlowdownValve", "HIPPSValve",
})

# Equipment classes that support dynamic compressor features
_COMPRESSOR_CLASSES = frozenset({"Compressor"})


def _set_property_on_unit(unit, prop_lower: str, value: float) -> bool:
    """Apply a single property change to a Java unit.  Returns True on success."""
    # Valve opening
    if "opening" in prop_lower and hasattr(unit, "setPercentValveOpening"):
        unit.setPercentValveOpening(float(value))
        return True
    # Cv
    if prop_lower in ("cv", "valve_cv") and hasattr(unit, "setCv"):
        unit.setCv(float(value))
        return True
    # Outlet pressure
    if "outletpressure" in prop_lower or "outlet_pressure" in prop_lower:
        if hasattr(unit, "setOutletPressure"):
            unit.setOutletPressure(float(value), "bara")
            return True
    # Flow rate (on streams)
    if "flow" in prop_lower and hasattr(unit, "setFlowRate"):
        unit.setFlowRate(float(value), "kg/hr")
        return True
    # Speed (compressor)
    if "speed" in prop_lower and hasattr(unit, "setSpeed"):
        unit.setSpeed(float(value))
        return True
    # Temperature
    if "temperature" in prop_lower and hasattr(unit, "setTemperature"):
        unit.setTemperature(float(value), "C")
        return True
    # Pressure (generic)
    if "pressure" in prop_lower and hasattr(unit, "setPressure"):
        unit.setPressure(float(value), "bara")
        return True
    return False


def _extract_snapshot(proc, units_map: Dict[str, Any]) -> Dict[str, float]:
    """Extract current KPI values from a solved process."""
    values: Dict[str, float] = {}
    for name, u in units_map.items():
        java_class = ""
        try:
            java_class = str(u.getClass().getSimpleName())
        except Exception:
            pass
        prefix = name.replace(" ", "_")

        # Separator level, pressure, temperature, flows
        if java_class in _DYNAMIC_CLASSES:
            if hasattr(u, "getLiquidLevel"):
                try:
                    values[f"{prefix}.liquid_level_m"] = float(u.getLiquidLevel())
                except Exception:
                    pass
            if hasattr(u, "getPressure"):
                try:
                    values[f"{prefix}.pressure_bara"] = float(u.getPressure())
                except Exception:
                    pass
            if hasattr(u, "getTemperature"):
                try:
                    values[f"{prefix}.temperature_C"] = float(u.getTemperature("C"))
                except Exception:
                    pass
            if hasattr(u, "getLiquidVolumeFraction"):
                try:
                    values[f"{prefix}.liquid_vol_frac"] = float(u.getLiquidVolumeFraction())
                except Exception:
                    pass
            # Gas outlet flow
            for m in ("getGasOutStream",):
                if hasattr(u, m):
                    try:
                        s = getattr(u, m)()
                        if s is not None:
                            values[f"{prefix}.gas_out_flow_kg_hr"] = float(s.getFlowRate("kg/hr"))
                    except Exception:
                        pass
            # Liquid outlet flow
            for m in ("getLiquidOutStream",):
                if hasattr(u, m):
                    try:
                        s = getattr(u, m)()
                        if s is not None:
                            values[f"{prefix}.liq_out_flow_kg_hr"] = float(s.getFlowRate("kg/hr"))
                    except Exception:
                        pass

        # Valve opening & flow
        if java_class in _VALVE_CLASSES:
            if hasattr(u, "getPercentValveOpening"):
                try:
                    values[f"{prefix}.valve_opening_pct"] = float(u.getPercentValveOpening())
                except Exception:
                    pass
            for m in ("getOutletStream", "getOutStream"):
                if hasattr(u, m):
                    try:
                        s = getattr(u, m)()
                        if s is not None:
                            values[f"{prefix}.flow_kg_hr"] = float(s.getFlowRate("kg/hr"))
                            break
                    except Exception:
                        pass

        # Compressor
        if java_class in _COMPRESSOR_CLASSES:
            if hasattr(u, "getPower"):
                try:
                    values[f"{prefix}.power_kW"] = float(u.getPower()) / 1000.0
                except Exception:
                    pass
            if hasattr(u, "getSpeed"):
                try:
                    values[f"{prefix}.speed_rpm"] = float(u.getSpeed())
                except Exception:
                    pass

        # Generic outlet flow for any unit with an outlet stream
        if java_class not in _DYNAMIC_CLASSES and java_class not in _VALVE_CLASSES:
            for m in ("getOutletStream", "getOutStream", "getGasOutStream"):
                if hasattr(u, m):
                    try:
                        s = getattr(u, m)()
                        if s is not None:
                            flow = float(s.getFlowRate("kg/hr"))
                            if flow > 0:
                                values[f"{prefix}.flow_kg_hr"] = flow
                            p = float(s.getPressure("bara"))
                            values[f"{prefix}.pressure_bara"] = p
                            values[f"{prefix}.temperature_C"] = float(s.getTemperature("C"))
                            break
                    except Exception:
                        pass

    return values


def _run_transient(
    model: NeqSimProcessModel,
    changes: Optional[List[Dict[str, Any]]] = None,
    duration_s: float = 60.0,
    n_steps: int = 10,
    dt: Optional[float] = None,
) -> DynamicSimResult:
    """
    Run a true transient simulation using NeqSim's ``process.runTransient()``.

    Workflow (per the Dynamic Simulation Guide):
    1. Clone the model so the original is not mutated.
    2. Run steady-state to initialise.
    3. Set ``setCalculateSteadyState(false)`` on separators, tanks, and valves.
    4. Apply the requested changes (e.g. valve openings).
    5. Step ``process.runTransient(dt, id)`` for *n_steps*.
    6. Record KPIs at every step.
    7. Reset equipment back to steady-state mode (try-finally).

    Parameters
    ----------
    model : NeqSimProcessModel
        The process model (will be cloned internally).
    changes : list of dict, optional
        Each dict has ``{"unit": "<name>", "property": "<prop>", "value": <num>}``.
        Example: ``[{"unit": "VLV-100", "property": "percentValveOpening", "value": 10}]``
    duration_s : float
        Total simulation time in seconds.
    n_steps : int
        Number of transient time steps.
    dt : float, optional
        Time step size. If *None*, computed as ``duration_s / n_steps``.
    """
    result = DynamicSimResult(scenario_type="transient")
    changes = changes or []

    # --- Clone so we don't mutate the live model ---
    try:
        clone = model.clone()
    except Exception as exc:
        result.message = f"Cannot clone model for transient run: {exc}"
        return result

    proc = clone.get_process()
    if proc is None:
        result.message = "No process available."
        return result

    # --- 1. Initialise with steady-state ---
    try:
        clone.run()
    except Exception as exc:
        logger.warning("Steady-state init before transient failed: %s", exc)

    # --- Discover units ---
    try:
        java_units = list(proc.getUnitOperations())
    except Exception:
        java_units = []

    units_map: Dict[str, Any] = {}
    dynamic_equipment: list = []  # units that we switch to transient mode
    for u in java_units:
        try:
            name = str(u.getName())
        except Exception:
            continue
        units_map[name] = u
        java_class = ""
        try:
            java_class = str(u.getClass().getSimpleName())
        except Exception:
            pass

        # Switch separators / tanks / valves to dynamic mode
        if java_class in _DYNAMIC_CLASSES or java_class in _VALVE_CLASSES:
            if hasattr(u, "setCalculateSteadyState"):
                try:
                    u.setCalculateSteadyState(False)
                    dynamic_equipment.append(u)
                except Exception:
                    pass

    # --- 2. Apply changes (e.g. close valves to 10%) ---
    change_descriptions: List[str] = []
    for ch in changes:
        unit_name = ch.get("unit", "")
        prop = ch.get("property", "percentValveOpening")
        value = ch.get("value", 0)
        # Fuzzy-match unit name
        matched = None
        for uname, uobj in units_map.items():
            if unit_name.lower() in uname.lower() or uname.lower() in unit_name.lower():
                matched = (uname, uobj)
                break
        if matched:
            ok = _set_property_on_unit(matched[1], prop.lower(), value)
            if ok:
                change_descriptions.append(f"{matched[0]}.{prop} = {value}")
            else:
                change_descriptions.append(f"{matched[0]}.{prop} = {value} (unsupported)")
        else:
            change_descriptions.append(f"{unit_name} not found")

    # --- 3. Step through transient ---
    if dt is None:
        dt = duration_s / max(n_steps, 1)

    calc_id = uuid.uuid4()
    # Convert Python UUID to Java UUID
    try:
        from java.util import UUID as JavaUUID  # type: ignore[import]
        java_id = JavaUUID.fromString(str(calc_id))
    except ImportError:
        java_id = None  # fallback: let NeqSim generate its own

    time_data: List[TimeSeriesPoint] = []
    variable_names: List[str] = []

    try:
        for step in range(n_steps + 1):
            t = step * dt

            if step == 0:
                # Record initial state (before first transient step)
                snapshot = _extract_snapshot(proc, units_map)
            else:
                # Run one transient step
                try:
                    if java_id is not None:
                        proc.runTransient(dt, java_id)
                    else:
                        proc.runTransient(dt)
                except Exception as exc:
                    logger.warning("runTransient step %d failed: %s", step, exc)
                    break
                snapshot = _extract_snapshot(proc, units_map)

            if not variable_names and snapshot:
                variable_names = list(snapshot.keys())

            time_data.append(TimeSeriesPoint(time_s=t, values=snapshot))

    finally:
        # --- Reset to steady-state ---
        for u in dynamic_equipment:
            try:
                u.setCalculateSteadyState(True)
            except Exception:
                pass

    # --- Build result ---
    result.time_series = time_data
    result.variable_names = variable_names
    result.duration_s = duration_s
    result.time_steps = len(time_data)
    result.method = "neqsim_runTransient"
    if time_data:
        result.final_state = time_data[-1].values.copy()

    # Build variable_units from names
    for vn in variable_names:
        if "bara" in vn:
            result.variable_units[vn] = "bara"
        elif "_C" in vn:
            result.variable_units[vn] = "°C"
        elif "kg_hr" in vn:
            result.variable_units[vn] = "kg/hr"
        elif "level" in vn:
            result.variable_units[vn] = "m"
        elif "pct" in vn or "opening" in vn:
            result.variable_units[vn] = "%"
        elif "kW" in vn:
            result.variable_units[vn] = "kW"
        elif "rpm" in vn:
            result.variable_units[vn] = "RPM"

    changes_str = "; ".join(change_descriptions) if change_descriptions else "none"
    result.message = (
        f"Transient simulation: {n_steps} steps × {dt:.1f}s = {duration_s:.0f}s total. "
        f"Changes applied: {changes_str}"
    )
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
            stream_name = streams[0].name
        else:
            result.message = "No streams found."
            return result

    # Find current value
    current_flow = 0.0
    for s_info in streams:
        if stream_name.lower() in s_info.name.lower():
            current_flow = s_info.flow_rate_kg_hr if s_info.flow_rate_kg_hr is not None else 10000.0
            stream_name = s_info.name
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
            for cs_info in clone.list_streams():
                if stream_name.lower() in cs_info.name.lower():
                    try:
                        java_stream = clone.get_stream(cs_info.name)
                        if ramp_variable in ("pressure", "pressure_bara"):
                            java_stream.setPressure(float(val), "bara")
                        elif ramp_variable in ("temperature", "temperature_C"):
                            java_stream.setTemperature(float(val), "C")
                        else:
                            java_stream.setFlowRate(float(val), "kg/hr")
                    except KeyError:
                        pass
                    break

            run_result = clone.run()

            # Extract KPIs
            kpis = run_result.kpis
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
    # Determine unit label for message
    if ramp_variable in ("pressure", "pressure_bara"):
        ramp_unit = "bara"
    elif ramp_variable in ("temperature", "temperature_C"):
        ramp_unit = "°C"
    else:
        ramp_unit = "kg/hr"
    result.message = f"Ramp {stream_name} from {start_value:.0f} to {end_value:.0f} {ramp_unit}"
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
    ramp_variable: str = "feed_flow",
    changes: Optional[List[Dict[str, Any]]] = None,
    dt: Optional[float] = None,
) -> DynamicSimResult:
    """
    Run a dynamic simulation scenario.

    Parameters
    ----------
    model : NeqSimProcessModel
    scenario_type : str
        'blowdown', 'startup', 'shutdown', 'ramp', or 'transient'
    changes : list of dict, optional
        For 'transient' — list of ``{"unit": ..., "property": ..., "value": ...}``
    dt : float, optional
        For 'transient' — explicit time step size (seconds).
    """
    # ---- True transient via NeqSim runTransient() ----
    if scenario_type == "transient":
        return _run_transient(
            model,
            changes=changes,
            duration_s=duration_s,
            n_steps=n_steps,
            dt=dt,
        )

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
            for s_info in streams:
                if not stream_name or stream_name.lower() in s_info.name.lower():
                    flow = s_info.flow_rate_kg_hr or 10000.0
                    start_value = flow
                    end_value = flow * 0.05
                    break
        return _run_ramp(
            model, stream_name=stream_name,
            start_value=start_value, end_value=end_value,
            duration_s=duration_s, n_steps=n_steps,
            ramp_variable=ramp_variable,
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
