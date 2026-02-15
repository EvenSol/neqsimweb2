"""
PVT Simulation — standard PVT experiments using NeqSim's thermodynamic engine.

Supported experiments:
  - Constant Mass Expansion (CME)
  - Constant Volume Depletion (CVD)
  - Differential Liberation (DL)
  - Saturation Point (bubble/dew point)
  - Separator Test
  - Viscosity vs P/T

Uses the NeqSim Java PVT simulation classes where available, with a Python
fallback that performs flash calculations at each step.

Returns a ``PVTResult`` with tabular data for plotting.
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
class PVTDataPoint:
    """One point in a PVT experiment."""
    pressure_bara: float = 0.0
    temperature_C: float = 0.0
    values: Dict[str, float] = field(default_factory=dict)


@dataclass
class PVTResult:
    """Result of a PVT simulation experiment."""
    experiment_type: str = ""
    data_points: List[PVTDataPoint] = field(default_factory=list)
    column_names: List[str] = field(default_factory=list)
    column_units: Dict[str, str] = field(default_factory=dict)
    saturation_pressure_bara: float = 0.0
    saturation_temperature_C: float = 0.0
    saturation_type: str = ""  # "bubble" or "dew"
    fluid_description: str = ""
    method: str = "flash_stepping"
    message: str = ""


# ---------------------------------------------------------------------------
# Fluid creation from stream
# ---------------------------------------------------------------------------

def _get_fluid_from_model(
    model: NeqSimProcessModel,
    stream_name: Optional[str] = None,
) -> Any:
    """Extract a NeqSim fluid object from a model stream."""
    proc = model.get_process()
    if proc is None:
        return None

    streams = model.list_streams()
    target = None

    if stream_name:
        for s_info in streams:
            if stream_name.lower() in s_info.name.lower():
                try:
                    target = model.get_stream(s_info.name)
                except KeyError:
                    pass
                break

    if target is None and streams:
        try:
            target = model.get_stream(streams[0].name)
        except KeyError:
            pass

    if target is None:
        return None

    try:
        fluid = target.getFluid().clone()
        return fluid
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Saturation point calculation
# ---------------------------------------------------------------------------

def _calc_saturation_point(fluid, temperature_C: float) -> Tuple[float, str]:
    """Calculate bubble/dew point at given temperature."""
    try:
        from neqsim import jneqsim
        fluid_clone = fluid.clone()
        fluid_clone.setTemperature(float(temperature_C) + 273.15)

        thermoOps = jneqsim.thermodynamicOperations.ThermodynamicOperations(fluid_clone)

        # Try bubble point first
        try:
            thermoOps.bubblePointPressureFlash(False)
            p = fluid_clone.getPressure()
            if p > 0 and p < 1e6:
                return p, "bubble"
        except Exception:
            pass

        # Try dew point
        try:
            thermoOps.dewPointPressureFlash()
            p = fluid_clone.getPressure()
            if p > 0 and p < 1e6:
                return p, "dew"
        except Exception:
            pass

    except Exception:
        pass

    return 0.0, "unknown"


# ---------------------------------------------------------------------------
# Constant Mass Expansion (CME)
# ---------------------------------------------------------------------------

def _run_cme(
    fluid,
    temperature_C: float,
    p_start_bara: float,
    p_end_bara: float,
    n_steps: int,
) -> PVTResult:
    """Constant Mass Expansion: deplete pressure at constant T, measure volumes."""
    result = PVTResult(experiment_type="CME")

    try:
        from neqsim.thermo import TPflash

        pressures = [p_start_bara - (p_start_bara - p_end_bara) * i / max(n_steps - 1, 1)
                     for i in range(n_steps)]

        columns = ["relative_volume", "gas_Z_factor", "oil_density_kg_m3",
                    "gas_density_kg_m3", "gas_fraction", "oil_viscosity_cP"]
        result.column_names = columns
        result.column_units = {
            "relative_volume": "-", "gas_Z_factor": "-",
            "oil_density_kg_m3": "kg/m³", "gas_density_kg_m3": "kg/m³",
            "gas_fraction": "-", "oil_viscosity_cP": "cP",
        }

        for p in pressures:
            try:
                fl = fluid.clone()
                fl.setTemperature(float(temperature_C) + 273.15)
                fl.setPressure(float(p))
                TPflash(fl)
                fl.initProperties()

                gas = _get_gas_phase(fl)
                liq = _get_liquid_phase(fl)

                vals: Dict[str, float] = {}
                vals["gas_Z_factor"] = _safe_float(lambda: gas.getZ()) if gas else 0.0
                vals["oil_density_kg_m3"] = _safe_float(lambda: liq.getDensity("kg/m3")) if liq else 0.0
                vals["gas_density_kg_m3"] = _safe_float(lambda: gas.getDensity("kg/m3")) if gas else 0.0
                vals["gas_fraction"] = _safe_float(lambda: gas.getBeta()) if gas else 0.0
                vals["oil_viscosity_cP"] = _safe_float(lambda: liq.getViscosity("cP")) if liq else 0.0
                vals["relative_volume"] = _safe_float(lambda: fl.getVolume("m3"))

                result.data_points.append(PVTDataPoint(
                    pressure_bara=p, temperature_C=temperature_C, values=vals,
                ))
            except Exception:
                result.data_points.append(PVTDataPoint(
                    pressure_bara=p, temperature_C=temperature_C,
                ))

        # Normalize relative volume to saturation point
        sat_idx = None
        for i, dp in enumerate(result.data_points):
            if dp.values.get("gas_fraction", 0) > 0.001 and i > 0:
                sat_idx = i - 1
                break
        if sat_idx is not None and result.data_points[sat_idx].values.get("relative_volume", 0) > 0:
            v_sat = result.data_points[sat_idx].values["relative_volume"]
            for dp in result.data_points:
                if dp.values.get("relative_volume", 0) > 0:
                    dp.values["relative_volume"] /= v_sat
            result.saturation_pressure_bara = result.data_points[sat_idx].pressure_bara

        result.message = f"CME at {temperature_C}°C, {p_start_bara}→{p_end_bara} bara, {n_steps} steps"

    except Exception as e:
        result.message = f"CME failed: {str(e)}"

    return result


# ---------------------------------------------------------------------------
# Differential Liberation
# ---------------------------------------------------------------------------

def _run_diff_lib(
    fluid,
    temperature_C: float,
    p_start_bara: float,
    p_end_bara: float,
    n_steps: int,
) -> PVTResult:
    """Differential Liberation: remove gas at each pressure step."""
    result = PVTResult(experiment_type="Differential Liberation")

    try:
        from neqsim.thermo import TPflash

        pressures = [p_start_bara - (p_start_bara - p_end_bara) * i / max(n_steps - 1, 1)
                     for i in range(n_steps)]

        columns = ["solution_GOR_Sm3_Sm3", "oil_FVF_rm3_Sm3", "oil_density_kg_m3",
                    "gas_Z_factor", "gas_gravity"]
        result.column_names = columns

        fl = fluid.clone()
        fl.setTemperature(float(temperature_C) + 273.15)

        for p in pressures:
            try:
                fl.setPressure(float(p))
                TPflash(fl)
                fl.initProperties()

                gas = _get_gas_phase(fl)
                liq = _get_liquid_phase(fl)

                vals: Dict[str, float] = {}
                vals["gas_Z_factor"] = _safe_float(lambda: gas.getZ()) if gas else 0.0
                vals["oil_density_kg_m3"] = _safe_float(lambda: liq.getDensity("kg/m3")) if liq else 0.0
                vals["gas_gravity"] = (
                    _safe_float(lambda: gas.getMolarMass()) / 0.02896 if gas else 0.0
                )

                gas_vol = _safe_float(lambda: gas.getVolume("m3")) if gas else 0.0
                oil_vol = _safe_float(lambda: liq.getVolume("m3")) if liq else 0.0
                vals["solution_GOR_Sm3_Sm3"] = gas_vol / max(oil_vol, 1e-10) if oil_vol > 0 else 0.0
                vals["oil_FVF_rm3_Sm3"] = oil_vol

                result.data_points.append(PVTDataPoint(
                    pressure_bara=p, temperature_C=temperature_C, values=vals,
                ))

                # Remove liberated gas (simulate differential liberation)
                if gas is not None:
                    try:
                        n_comps = fl.getNumberOfComponents()
                        for j in range(n_comps):
                            gas_moles = _safe_float(
                                lambda _j=j: gas.getComponent(_j).getNumberOfMolesInPhase()
                            )
                            if gas_moles > 0:
                                current = float(fl.getComponent(j).getNumberOfmoles())
                                fl.getComponent(j).setNumberOfmoles(max(current - gas_moles, 1e-20))
                    except Exception:
                        pass

            except Exception:
                result.data_points.append(PVTDataPoint(
                    pressure_bara=p, temperature_C=temperature_C,
                ))

        result.message = f"Differential Liberation at {temperature_C}°C, {n_steps} steps"

    except Exception as e:
        result.message = f"Differential Liberation failed: {str(e)}"

    return result


# ---------------------------------------------------------------------------
# Constant Volume Depletion (CVD)
# ---------------------------------------------------------------------------

def _run_cvd(
    fluid,
    temperature_C: float,
    p_start_bara: float,
    p_end_bara: float,
    n_steps: int,
) -> PVTResult:
    """
    Constant Volume Depletion (CVD).

    Simulates a depletion experiment where gas is removed at each step to
    keep the total volume constant at the saturation-point cell volume.
    Reports cumulative liquid dropout (retrograde condensate volume %),
    gas Z-factor, gas gravity, gas produced (cumulative mol%), and
    liquid density.
    """
    result = PVTResult(experiment_type="CVD")

    try:
        from neqsim.thermo import TPflash

        columns = [
            "liquid_dropout_pct", "gas_Z_factor", "gas_gravity",
            "cumulative_gas_produced_mol_pct", "liquid_density_kg_m3",
        ]
        result.column_names = columns
        result.column_units = {
            "liquid_dropout_pct": "%",
            "gas_Z_factor": "-",
            "gas_gravity": "-",
            "cumulative_gas_produced_mol_pct": "mol%",
            "liquid_density_kg_m3": "kg/m³",
        }

        # Step 1: Flash at start pressure to find saturation cell volume
        fl = fluid.clone()
        fl.setTemperature(float(temperature_C) + 273.15)
        fl.setPressure(float(p_start_bara))
        TPflash(fl)
        fl.initProperties()

        cell_volume = _safe_float(lambda: fl.getVolume("m3"))
        if cell_volume <= 0:
            result.message = "CVD failed: could not compute cell volume at saturation."
            return result

        initial_moles = _safe_float(lambda: fl.getTotalNumberOfMoles())
        if initial_moles <= 0:
            initial_moles = 1.0

        pressures = [p_start_bara - (p_start_bara - p_end_bara) * i / max(n_steps - 1, 1)
                     for i in range(n_steps)]

        cumulative_gas_removed_moles = 0.0

        for p in pressures:
            try:
                fl.setPressure(float(p))
                TPflash(fl)
                fl.initProperties()

                gas = _get_gas_phase(fl)
                liq = _get_liquid_phase(fl)

                vals: Dict[str, float] = {}
                vals["gas_Z_factor"] = _safe_float(lambda: gas.getZ()) if gas else 0.0
                vals["gas_gravity"] = (
                    _safe_float(lambda: gas.getMolarMass()) / 0.02896 if gas else 0.0
                )
                vals["liquid_density_kg_m3"] = (
                    _safe_float(lambda: liq.getDensity("kg/m3")) if liq else 0.0
                )

                # Liquid dropout: volume of liquid / cell volume × 100
                liq_vol = _safe_float(lambda: liq.getVolume("m3")) if liq else 0.0
                vals["liquid_dropout_pct"] = (liq_vol / cell_volume * 100.0) if cell_volume > 0 else 0.0

                # Remove gas to restore cell volume
                # Gas to remove = (current_total_volume - cell_volume) worth of gas
                current_vol = _safe_float(lambda: fl.getVolume("m3"))
                excess_vol = current_vol - cell_volume

                if gas is not None and excess_vol > 0:
                    gas_vol = _safe_float(lambda: gas.getVolume("m3"))
                    if gas_vol > 0:
                        frac_to_remove = min(excess_vol / gas_vol, 0.999)
                        try:
                            n_comps = fl.getNumberOfComponents()
                            removed_moles = 0.0
                            for j in range(n_comps):
                                gas_moles = _safe_float(
                                    lambda _j=j: gas.getComponent(_j).getNumberOfMolesInPhase()
                                )
                                to_remove = gas_moles * frac_to_remove
                                if to_remove > 0:
                                    current = float(fl.getComponent(j).getNumberOfmoles())
                                    fl.getComponent(j).setNumberOfmoles(
                                        max(current - to_remove, 1e-20)
                                    )
                                    removed_moles += to_remove
                            cumulative_gas_removed_moles += removed_moles
                        except Exception:
                            pass

                vals["cumulative_gas_produced_mol_pct"] = (
                    cumulative_gas_removed_moles / initial_moles * 100.0
                )

                result.data_points.append(PVTDataPoint(
                    pressure_bara=p, temperature_C=temperature_C, values=vals,
                ))
            except Exception:
                result.data_points.append(PVTDataPoint(
                    pressure_bara=p, temperature_C=temperature_C,
                ))

        # Set saturation point
        result.saturation_pressure_bara = p_start_bara
        result.message = (
            f"CVD at {temperature_C}°C, {p_start_bara}→{p_end_bara} bara, {n_steps} steps"
        )

    except Exception as e:
        result.message = f"CVD failed: {str(e)}"

    return result


# ---------------------------------------------------------------------------
# Phase-access helpers
# ---------------------------------------------------------------------------

def _get_gas_phase(fl):
    """Return the gas phase object, trying multiple NeqSim naming conventions."""
    for name in ("gas", 0):
        try:
            ph = fl.getPhase(name)
            ptype = str(ph.getPhaseTypeName()).lower()
            if "gas" in ptype or "vapour" in ptype or "vapor" in ptype:
                return ph
        except Exception:
            pass
    # Last resort: phase index 0 often is gas after TP flash
    try:
        return fl.getPhase(0)
    except Exception:
        return None


def _get_liquid_phase(fl):
    """Return the oil/liquid phase, trying multiple NeqSim naming conventions."""
    for name in ("oil", "liquid", 1):
        try:
            ph = fl.getPhase(name)
            ptype = str(ph.getPhaseTypeName()).lower()
            if "oil" in ptype or "liquid" in ptype:
                return ph
        except Exception:
            pass
    # Fallback: try phase index 1
    try:
        n_phases = int(fl.getNumberOfPhases())
        if n_phases >= 2:
            return fl.getPhase(1)
    except Exception:
        pass
    return None


def _safe_float(fn, default: float = 0.0) -> float:
    """Call *fn* and return float, or *default* on any error / NaN."""
    try:
        v = float(fn())
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Separator Test
# ---------------------------------------------------------------------------

def _run_separator_test(
    fluid,
    stages: List[Dict[str, float]],
) -> PVTResult:
    """Multi-stage separator test."""
    result = PVTResult(experiment_type="Separator Test")

    try:
        from neqsim.thermo import TPflash

        columns = ["stage", "GOR_Sm3_Sm3", "oil_density_kg_m3", "gas_gravity",
                    "oil_FVF", "gas_Z_factor"]
        result.column_names = columns

        fl = fluid.clone()

        for i, stage in enumerate(stages):
            T = stage.get("temperature_C", 15.0)
            P = stage.get("pressure_bara", 1.013)

            fl.setTemperature(float(T) + 273.15)
            fl.setPressure(float(P))
            TPflash(fl)
            fl.initProperties()

            gas = _get_gas_phase(fl)
            liq = _get_liquid_phase(fl)

            vals: Dict[str, float] = {"stage": float(i + 1)}

            vals["gas_Z_factor"] = _safe_float(lambda: gas.getZ()) if gas else 0.0

            vals["oil_density_kg_m3"] = (
                _safe_float(lambda: liq.getDensity("kg/m3")) if liq else 0.0
            )

            vals["gas_gravity"] = (
                _safe_float(lambda: gas.getMolarMass()) / 0.02896 if gas else 0.0
            )

            gas_vol = _safe_float(lambda: gas.getVolume("m3")) if gas else 0.0
            oil_vol = _safe_float(lambda: liq.getVolume("m3")) if liq else 0.0
            vals["GOR_Sm3_Sm3"] = gas_vol / max(oil_vol, 1e-10) if oil_vol > 0 else 0.0
            vals["oil_FVF"] = oil_vol

            result.data_points.append(PVTDataPoint(
                pressure_bara=P, temperature_C=T, values=vals,
            ))

            # Feed liquid to next stage
            if liq is not None:
                try:
                    n_comps = fl.getNumberOfComponents()
                    for j in range(n_comps):
                        liq_moles = _safe_float(
                            lambda _j=j: liq.getComponent(_j).getNumberOfMolesInPhase()
                        )
                        fl.getComponent(j).setNumberOfmoles(max(liq_moles, 1e-20))
                except Exception:
                    pass

        result.message = f"Separator test: {len(stages)} stages"

    except Exception as e:
        result.message = f"Separator test failed: {str(e)}"

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_pvt_simulation(
    model: NeqSimProcessModel,
    experiment: str = "CME",
    stream_name: Optional[str] = None,
    temperature_C: float = 100.0,
    p_start_bara: float = 400.0,
    p_end_bara: float = 10.0,
    n_steps: int = 20,
    stages: Optional[List[Dict[str, float]]] = None,
) -> PVTResult:
    """
    Run a PVT simulation experiment.

    Parameters
    ----------
    model : NeqSimProcessModel
    experiment : str
        'CME', 'DL' (differential liberation), 'separator_test', 'saturation_point'
    stream_name : str, optional
        Stream to use as fluid source.
    """
    fluid = _get_fluid_from_model(model, stream_name)
    if fluid is None:
        return PVTResult(message="No fluid available from model.")

    # Get stream conditions as defaults
    streams = model.list_streams()
    for s_info in streams:
        if not stream_name or stream_name.lower() in s_info.name.lower():
            if temperature_C == 100.0 and s_info.temperature_C is not None:
                temperature_C = s_info.temperature_C
            if p_start_bara == 400.0 and s_info.pressure_bara is not None:
                p_start_bara = s_info.pressure_bara * 1.5
            break

    exp_lower = experiment.lower().replace(" ", "_").replace("-", "_")

    if exp_lower in ("cme", "constant_mass_expansion"):
        return _run_cme(fluid, temperature_C, p_start_bara, p_end_bara, n_steps)

    elif exp_lower in ("cvd", "constant_volume_depletion"):
        return _run_cvd(fluid, temperature_C, p_start_bara, p_end_bara, n_steps)

    elif exp_lower in ("dl", "differential_liberation"):
        return _run_diff_lib(fluid, temperature_C, p_start_bara, p_end_bara, n_steps)

    elif exp_lower in ("separator_test", "sep_test"):
        if stages is None:
            stages = [
                {"temperature_C": 40.0, "pressure_bara": 50.0},
                {"temperature_C": 30.0, "pressure_bara": 10.0},
                {"temperature_C": 15.0, "pressure_bara": 1.013},
            ]
        return _run_separator_test(fluid, stages)

    elif exp_lower in ("saturation", "saturation_point", "bubble_point", "dew_point"):
        p_sat, sat_type = _calc_saturation_point(fluid, temperature_C)
        result = PVTResult(
            experiment_type="Saturation Point",
            saturation_pressure_bara=p_sat,
            saturation_type=sat_type,
            message=f"{sat_type.capitalize()} point at {temperature_C}°C: {p_sat:.2f} bara",
        )
        return result

    else:
        return PVTResult(message=f"Unknown PVT experiment: {experiment}")


# ---------------------------------------------------------------------------
# Format for LLM
# ---------------------------------------------------------------------------

def format_pvt_result(result: PVTResult) -> str:
    """Format PVT results for the LLM follow-up."""
    lines = [f"=== PVT SIMULATION: {result.experiment_type.upper()} ==="]
    lines.append(result.message)

    if result.saturation_pressure_bara > 0:
        lines.append(f"Saturation point: {result.saturation_pressure_bara:.2f} bara "
                     f"({result.saturation_type})")
    lines.append("")

    if result.data_points:
        # Table header
        cols = ["P (bara)", "T (°C)"] + result.column_names
        header = "  ".join(f"{c[:18]:>18}" for c in cols)
        lines.append(header)

        for dp in result.data_points:
            vals = [dp.pressure_bara, dp.temperature_C]
            vals += [dp.values.get(c, 0) for c in result.column_names]
            row = "  ".join(f"{v:>18.4f}" for v in vals)
            lines.append(row)

    return "\n".join(lines)
