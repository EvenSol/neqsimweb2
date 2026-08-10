"""Validated NeqSim pipeline calculations used by the Streamlit pipeline page.

The module keeps numerical and Java-interoperability logic outside the page so
that both the Beggs-Brill and two-fluid calculation paths can be regression
tested without driving Streamlit widgets.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


REQUIRED_FLUID_COLUMNS = (
    "ComponentName",
    "MolarComposition[-]",
    "MolarMass[kg/mol]",
    "RelativeDensity[-]",
)

PIPELINE_FLUID_PRESETS: Mapping[str, Mapping[str, float]] = {
    "Lean natural gas": {
        "CO2": 0.02,
        "methane": 0.90,
        "ethane": 0.05,
        "propane": 0.03,
    },
    "Two-phase gas condensate": {
        "methane": 0.50,
        "ethane": 0.10,
        "propane": 0.10,
        "i-butane": 0.05,
        "n-butane": 0.05,
        "i-pentane": 0.05,
        "n-pentane": 0.05,
        "n-hexane": 0.10,
    },
    "Blank / custom": {},
}


class PipelineInputError(ValueError):
    """Raised when a pipeline input cannot define a physical calculation."""


class PipelineConvergenceError(RuntimeError):
    """Raised when the requested outlet pressure cannot be bracketed or solved."""


@dataclass(frozen=True)
class PressureSolveResult:
    """A native pipeline run whose outlet pressure matches the requested value."""

    pipe: Any
    inlet_stream: Any
    inlet_pressure_bara: float
    outlet_pressure_bara: float
    iterations: int


@dataclass(frozen=True)
class PipelineProfiles:
    """Unit-normalized profiles suitable for tables and plots."""

    position_km: tuple[float, ...]
    pressure_bara: tuple[float, ...]
    temperature_c: tuple[float, ...]
    liquid_holdup: tuple[float, ...]
    gas_velocity_m_s: tuple[float, ...]
    liquid_velocity_m_s: tuple[float, ...]
    flow_regime: tuple[str, ...]
    mixture_velocity_m_s: tuple[float, ...] = ()
    mixture_density_kg_m3: tuple[float, ...] = ()
    reynolds_number: tuple[float, ...] = ()


def fluid_from_preset(base_fluid: Mapping[str, Sequence[Any]], preset: str) -> pd.DataFrame:
    """Return a fresh standard NeqSim fluid table populated from *preset*."""

    if preset not in PIPELINE_FLUID_PRESETS:
        raise PipelineInputError(f"Unknown fluid preset: {preset}")
    fluid = pd.DataFrame(base_fluid).copy(deep=True)
    missing_columns = [column for column in REQUIRED_FLUID_COLUMNS if column not in fluid]
    if missing_columns:
        raise PipelineInputError(
            "Fluid table is missing required column(s): " + ", ".join(missing_columns)
        )
    fluid["MolarComposition[-]"] = 0.0
    for component, fraction in PIPELINE_FLUID_PRESETS[preset].items():
        rows = fluid["ComponentName"] == component
        if not rows.any():
            raise PipelineInputError(
                f"Fluid preset component '{component}' is absent from the base table."
            )
        fluid.loc[rows, "MolarComposition[-]"] = fraction
    return fluid


def normalize_fluid_composition(fluid_table: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize molar fractions while preserving plus-fraction data."""

    missing_columns = [column for column in REQUIRED_FLUID_COLUMNS if column not in fluid_table]
    if missing_columns:
        raise PipelineInputError(
            "Fluid table is missing required column(s): " + ", ".join(missing_columns)
        )

    normalized = fluid_table.loc[:, REQUIRED_FLUID_COLUMNS].copy(deep=True)
    if normalized["ComponentName"].isna().any():
        raise PipelineInputError("Every fluid row must have a component name.")
    normalized["ComponentName"] = normalized["ComponentName"].astype(str).str.strip()
    if (normalized["ComponentName"] == "").any():
        raise PipelineInputError("Every fluid row must have a component name.")
    if normalized["ComponentName"].duplicated().any():
        duplicate = normalized.loc[
            normalized["ComponentName"].duplicated(), "ComponentName"
        ].iloc[0]
        raise PipelineInputError(f"Duplicate fluid component: {duplicate}")

    composition = pd.to_numeric(
        normalized["MolarComposition[-]"], errors="coerce"
    ).fillna(0.0)
    if not np.isfinite(composition.to_numpy(dtype=float)).all():
        raise PipelineInputError("Molar fractions must be finite numbers.")
    if (composition < 0.0).any():
        raise PipelineInputError("Molar fractions cannot be negative.")
    total = float(composition.sum())
    if total <= 0.0:
        raise PipelineInputError(
            "Enter a fluid composition or load one of the pipeline presets."
        )
    normalized["MolarComposition[-]"] = composition / total
    return normalized


def standard_mass_flow_kg_s(
    fluid_table: pd.DataFrame,
    flow_value: float,
    flow_unit: str,
    actual_pressure_bara: float,
    actual_temperature_c: float,
    thermodynamic_model: str = "srk",
) -> float:
    """Convert a user flow input to kg/s using the selected NeqSim fluid."""

    flow_value = _positive_finite("Flow rate", flow_value)
    fluid_table = normalize_fluid_composition(fluid_table)
    if flow_unit == "kg/s":
        return flow_value

    from neqsim.thermo import TPflash

    reference_fluid = _create_neqsim_fluid(
        fluid_table,
        thermodynamic_model,
    )
    if flow_unit == "MSm3/day":
        reference_fluid.setPressure(1.01325, "bara")
        reference_fluid.setTemperature(15.0, "C")
        seconds_per_day = 86_400.0
        volume_flow_m3_s = flow_value * 1.0e6 / seconds_per_day
    elif flow_unit == "m3/hr":
        reference_fluid.setPressure(
            _positive_finite("Actual pressure", actual_pressure_bara),
            "bara",
        )
        reference_fluid.setTemperature(
            _finite("Actual temperature", actual_temperature_c),
            "C",
        )
        volume_flow_m3_s = flow_value / 3_600.0
    else:
        raise PipelineInputError(f"Unsupported flow unit: {flow_unit}")

    TPflash(reference_fluid)
    reference_fluid.initProperties()
    density_kg_m3 = float(reference_fluid.getDensity("kg/m3"))
    if not math.isfinite(density_kg_m3) or density_kg_m3 <= 0.0:
        raise PipelineInputError(
            "NeqSim returned a non-positive density for the flow conversion."
        )
    return volume_flow_m3_s * density_kg_m3


def build_beggs_brill_pipe(
    fluid_table: pd.DataFrame,
    inlet_pressure_bara: float,
    inlet_temperature_c: float,
    mass_flow_kg_s: float,
    length_m: float,
    diameter_m: float,
    roughness_m: float,
    elevation_m: float,
    number_of_increments: int,
    heat_transfer_coefficient_w_m2_k: float,
    ambient_temperature_c: float,
    thermodynamic_model: str = "srk",
) -> tuple[Any, Any]:
    """Build and run NeqSim's native ``PipeBeggsAndBrills`` model."""

    from neqsim import jneqsim

    fluid_table = normalize_fluid_composition(fluid_table)
    inlet_pressure_bara = _positive_finite("Inlet pressure", inlet_pressure_bara)
    mass_flow_kg_s = _positive_finite("Mass flow", mass_flow_kg_s)
    length_m = _positive_finite("Pipe length", length_m)
    diameter_m = _positive_finite("Pipe diameter", diameter_m)
    roughness_m = _nonnegative_finite("Pipe roughness", roughness_m)
    elevation_m = _finite("Elevation change", elevation_m)
    number_of_increments = _positive_integer(
        "Number of increments", number_of_increments
    )
    heat_transfer_coefficient_w_m2_k = _nonnegative_finite(
        "Heat-transfer coefficient", heat_transfer_coefficient_w_m2_k
    )
    ambient_temperature_c = _finite(
        "Ambient temperature", ambient_temperature_c
    )

    fluid = _create_neqsim_fluid(
        fluid_table,
        thermodynamic_model,
    )
    stream = jneqsim.process.equipment.stream.Stream("Pipeline inlet", fluid)
    stream.setFlowRate(mass_flow_kg_s * 3_600.0, "kg/hr")
    stream.setTemperature(_finite("Inlet temperature", inlet_temperature_c), "C")
    stream.setPressure(inlet_pressure_bara, "bara")
    stream.run()

    pipe_class = jneqsim.process.equipment.pipeline.PipeBeggsAndBrills
    pipe = pipe_class("Beggs-Brill pipeline", stream)
    pipe.setLength(length_m)
    pipe.setDiameter(diameter_m)
    pipe.setElevation(elevation_m)
    pipe.setNumberOfIncrements(number_of_increments)
    pipe.setPipeWallRoughness(roughness_m)
    if heat_transfer_coefficient_w_m2_k > 0.0:
        pipe.setConstantSurfaceTemperature(ambient_temperature_c, "C")
        pipe.setHeatTransferCoefficient(heat_transfer_coefficient_w_m2_k)
        pipe.setHeatTransferMode(pipe_class.HeatTransferMode.SPECIFIED_U)
    else:
        pipe.setHeatTransferMode(pipe_class.HeatTransferMode.ADIABATIC)
    pipe.run()
    return pipe, stream


def normalize_elevation_profile(
    distances_m: Sequence[float], elevations_m: Sequence[float]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Validate a profile and shift distance so the first point is the inlet."""

    if len(distances_m) != len(elevations_m):
        raise PipelineInputError("Distance and elevation profiles must have equal length.")
    if len(distances_m) < 2:
        raise PipelineInputError("Pipeline profile needs at least two points.")
    distances = tuple(_finite("Profile distance", value) for value in distances_m)
    elevations = tuple(_finite("Profile elevation", value) for value in elevations_m)
    if any(distance < 0.0 for distance in distances):
        raise PipelineInputError("Profile distances cannot be negative.")
    if any(right <= left for left, right in zip(distances, distances[1:])):
        raise PipelineInputError(
            "Profile distances must be strictly increasing without duplicates."
        )
    origin = distances[0]
    normalized_distances = tuple(distance - origin for distance in distances)
    if normalized_distances[-1] <= 0.0:
        raise PipelineInputError("Pipeline profile must have positive length.")
    return normalized_distances, elevations


def interpolate_section_elevations(
    distances_m: Sequence[float],
    elevations_m: Sequence[float],
    number_of_sections: int,
) -> tuple[float, ...]:
    """Interpolate absolute elevation at NeqSim two-fluid section centres."""

    distances, elevations = normalize_elevation_profile(distances_m, elevations_m)
    number_of_sections = _positive_integer(
        "Number of sections", number_of_sections
    )
    section_length = distances[-1] / number_of_sections
    section_centres = np.asarray(
        [(index + 0.5) * section_length for index in range(number_of_sections)],
        dtype=float,
    )
    return tuple(
        float(value)
        for value in np.interp(section_centres, distances, elevations)
    )


def build_two_fluid_pipe(
    fluid_table: pd.DataFrame,
    inlet_pressure_bara: float,
    inlet_temperature_c: float,
    mass_flow_kg_hr: float,
    diameter_m: float,
    roughness_m: float,
    distances_m: Sequence[float],
    elevations_m: Sequence[float],
    number_of_sections: int,
    heat_transfer_coefficient_w_m2_k: float,
    ambient_temperature_c: float,
    enable_slug_tracking: bool,
    thermodynamic_model: str = "srk",
) -> tuple[Any, Any]:
    """Build and run NeqSim's native ``TwoFluidPipe`` model."""

    from neqsim import jneqsim

    fluid_table = normalize_fluid_composition(fluid_table)
    inlet_pressure_bara = _positive_finite("Inlet pressure", inlet_pressure_bara)
    mass_flow_kg_hr = _positive_finite("Mass flow", mass_flow_kg_hr)
    diameter_m = _positive_finite("Pipe diameter", diameter_m)
    roughness_m = _nonnegative_finite("Pipe roughness", roughness_m)
    number_of_sections = _positive_integer(
        "Number of sections", number_of_sections
    )
    distances, elevations = normalize_elevation_profile(distances_m, elevations_m)
    section_elevations = interpolate_section_elevations(
        distances,
        elevations,
        number_of_sections,
    )
    heat_transfer_coefficient_w_m2_k = _nonnegative_finite(
        "Heat-transfer coefficient", heat_transfer_coefficient_w_m2_k
    )
    ambient_temperature_c = _finite(
        "Ambient temperature", ambient_temperature_c
    )

    fluid = _create_neqsim_fluid(
        fluid_table,
        thermodynamic_model,
    )
    stream = jneqsim.process.equipment.stream.Stream("Pipeline inlet", fluid)
    stream.setFlowRate(mass_flow_kg_hr, "kg/hr")
    stream.setTemperature(_finite("Inlet temperature", inlet_temperature_c), "C")
    stream.setPressure(inlet_pressure_bara, "bara")
    stream.run()

    pipe = jneqsim.process.equipment.pipeline.TwoFluidPipe(
        "Two-fluid pipeline",
        stream,
    )
    pipe.setLength(distances[-1])
    pipe.setDiameter(diameter_m)
    pipe.setNumberOfSections(number_of_sections)
    # TwoFluidPipe owns a dedicated roughness field; setRoughness is the API
    # that the section momentum equations consume.
    pipe.setRoughness(roughness_m)
    pipe.setElevationProfile(section_elevations)
    if heat_transfer_coefficient_w_m2_k > 0.0:
        pipe.setHeatTransferCoefficient(heat_transfer_coefficient_w_m2_k)
        pipe.setSurfaceTemperature(ambient_temperature_c, "C")
    pipe.setEnableSlugTracking(bool(enable_slug_tracking))
    pipe.run()
    return pipe, stream


def solve_inlet_pressure(
    build_and_run: Callable[[float], tuple[Any, Any]],
    target_outlet_pressure_bara: float,
    tolerance_bar: float = 0.02,
    max_iterations: int = 40,
    max_inlet_pressure_bara: float = 5_000.0,
) -> PressureSolveResult:
    """Solve inlet pressure using a guarded bracket and native pipe runs.

    NeqSim's Beggs-Brill model intentionally raises when a trial pressure
    produces a negative outlet pressure. Such a trial is a valid lower bound,
    while other native errors remain visible to the caller.
    """

    target = _positive_finite("Target outlet pressure", target_outlet_pressure_bara)
    tolerance = _positive_finite("Pressure tolerance", tolerance_bar)
    max_iterations = _positive_integer("Maximum iterations", max_iterations)
    max_inlet = _positive_finite("Maximum inlet pressure", max_inlet_pressure_bara)
    if max_inlet <= target:
        raise PipelineInputError(
            "Maximum inlet pressure must exceed the target outlet pressure."
        )

    def evaluate(inlet_pressure: float) -> tuple[Any | None, Any | None, float]:
        try:
            pipe, stream = build_and_run(inlet_pressure)
        except Exception as error:
            if _is_insufficient_pressure_failure(error):
                return None, None, -math.inf
            raise
        outlet_pressure = float(pipe.getOutletPressure())
        if not math.isfinite(outlet_pressure) or outlet_pressure <= 0.0:
            raise PipelineConvergenceError(
                "NeqSim returned an invalid outlet pressure during bracketing."
            )
        return pipe, stream, outlet_pressure - target

    centre_pipe, centre_stream, centre_residual = evaluate(target)
    if abs(centre_residual) <= tolerance:
        return PressureSolveResult(
            centre_pipe,
            centre_stream,
            target,
            target + centre_residual,
            1,
        )

    if centre_residual > 0.0:
        high_pressure = target
        high_pipe = centre_pipe
        high_stream = centre_stream
        high_residual = centre_residual
        span = max(1.0, 0.1 * target)
        while True:
            low_pressure = max(1.0, target - span)
            low_pipe, low_stream, low_residual = evaluate(low_pressure)
            if low_residual <= 0.0:
                break
            if low_pressure <= 1.0:
                raise PipelineConvergenceError(
                    "The requested outlet pressure requires an inlet pressure below 1 bara."
                )
            span *= 2.0
    else:
        low_pressure = target
        low_pipe = centre_pipe
        low_stream = centre_stream
        low_residual = centre_residual
        span = max(10.0, 0.2 * target)
        while True:
            high_pressure = min(max_inlet, target + span)
            high_pipe, high_stream, high_residual = evaluate(high_pressure)
            if high_residual >= 0.0:
                break
            if high_pressure >= max_inlet:
                raise PipelineConvergenceError(
                    "Could not bracket the target outlet pressure below "
                    f"{max_inlet:.0f} bara inlet pressure. Reduce flow, length, or elevation."
                )
            span *= 2.0

    best_pipe = high_pipe
    best_stream = high_stream
    best_pressure = high_pressure
    best_residual = high_residual
    for iteration in range(1, max_iterations + 1):
        midpoint = 0.5 * (low_pressure + high_pressure)
        pipe, stream, residual = evaluate(midpoint)
        if math.isfinite(residual) and abs(residual) < abs(best_residual):
            best_pipe = pipe
            best_stream = stream
            best_pressure = midpoint
            best_residual = residual
        if math.isfinite(residual) and abs(residual) <= tolerance:
            return PressureSolveResult(
                pipe,
                stream,
                midpoint,
                target + residual,
                iteration,
            )
        if residual > 0.0:
            high_pressure = midpoint
            high_pipe = pipe
            high_stream = stream
            high_residual = residual
        else:
            low_pressure = midpoint
            low_pipe = pipe
            low_stream = stream
            low_residual = residual

    if best_pipe is None or best_stream is None:
        raise PipelineConvergenceError(
            "The pressure solver did not produce a valid native pipeline state."
        )
    raise PipelineConvergenceError(
        "Inlet pressure solve did not reach the requested tolerance: "
        f"best outlet pressure was {target + best_residual:.3f} bara at "
        f"{best_pressure:.3f} bara inlet pressure."
    )


def read_beggs_brill_profiles(pipe: Any) -> PipelineProfiles:
    """Read native Beggs-Brill profiles and convert them to display units."""

    profiles = PipelineProfiles(
        position_km=_float_tuple(pipe.getLengthProfile(), scale=1.0 / 1_000.0),
        pressure_bara=_float_tuple(pipe.getPressureProfile()),
        temperature_c=_float_tuple(pipe.getTemperatureProfile(), offset=-273.15),
        liquid_holdup=_float_tuple(pipe.getLiquidHoldupProfile()),
        gas_velocity_m_s=_float_tuple(pipe.getGasSuperficialVelocityProfile()),
        liquid_velocity_m_s=_float_tuple(
            pipe.getLiquidSuperficialVelocityProfile()
        ),
        flow_regime=tuple(str(value) for value in pipe.getFlowRegimeProfileList()),
        mixture_velocity_m_s=_float_tuple(
            pipe.getMixtureSuperficialVelocityProfile()
        ),
        mixture_density_kg_m3=_float_tuple(pipe.getMixtureDensityProfile()),
        reynolds_number=_float_tuple(pipe.getMixtureReynoldsNumber()),
    )
    _validate_profile_lengths(profiles)
    return profiles


def read_two_fluid_profiles(pipe: Any) -> PipelineProfiles:
    """Read native two-fluid profiles and convert Pa/K/m to display units."""

    profiles = PipelineProfiles(
        position_km=_float_tuple(pipe.getPositionProfile(), scale=1.0 / 1_000.0),
        pressure_bara=_float_tuple(pipe.getPressureProfile(), scale=1.0 / 100_000.0),
        temperature_c=_float_tuple(pipe.getTemperatureProfile(), offset=-273.15),
        liquid_holdup=_float_tuple(pipe.getLiquidHoldupProfile()),
        gas_velocity_m_s=_float_tuple(pipe.getGasVelocityProfile()),
        liquid_velocity_m_s=_float_tuple(pipe.getLiquidVelocityProfile()),
        flow_regime=tuple(str(value) for value in pipe.getFlowRegimeProfile()),
    )
    _validate_profile_lengths(profiles)
    return profiles


def _validate_profile_lengths(profiles: PipelineProfiles) -> None:
    required_profiles = {
        "position": profiles.position_km,
        "pressure": profiles.pressure_bara,
        "temperature": profiles.temperature_c,
        "liquid holdup": profiles.liquid_holdup,
        "gas velocity": profiles.gas_velocity_m_s,
        "liquid velocity": profiles.liquid_velocity_m_s,
        "flow regime": profiles.flow_regime,
    }
    expected_length = len(profiles.position_km)
    if expected_length == 0:
        raise PipelineConvergenceError("NeqSim returned empty pipeline profiles.")
    inconsistent = [
        name for name, values in required_profiles.items() if len(values) != expected_length
    ]
    optional_profiles = {
        "mixture velocity": profiles.mixture_velocity_m_s,
        "mixture density": profiles.mixture_density_kg_m3,
        "Reynolds number": profiles.reynolds_number,
    }
    inconsistent.extend(
        name
        for name, values in optional_profiles.items()
        if values and len(values) != expected_length
    )
    if inconsistent:
        raise PipelineConvergenceError(
            "NeqSim returned inconsistent profile lengths for: "
            + ", ".join(inconsistent)
        )


def _create_neqsim_fluid(
    fluid_table: pd.DataFrame,
    thermodynamic_model: str,
) -> Any:
    """Create a NeqSim fluid with an explicit and reproducible model choice."""

    from neqsim.thermo import fluid_df

    model = str(thermodynamic_model).strip().lower()
    if model == "auto":
        return fluid_df(
            fluid_table,
            lastIsPlusFraction=False,
            autoSetModel=True,
            add_all_components=False,
        )
    if model not in {"srk", "pr", "cpa"}:
        raise PipelineInputError(
            "Thermodynamic model must be one of: auto, srk, pr, cpa."
        )
    return fluid_df(
        fluid_table,
        lastIsPlusFraction=False,
        modelName=model,
        add_all_components=False,
    )


def _float_tuple(
    values: Sequence[Any],
    scale: float = 1.0,
    offset: float = 0.0,
) -> tuple[float, ...]:
    converted = tuple(float(value) * scale + offset for value in values)
    if not all(math.isfinite(value) for value in converted):
        raise PipelineConvergenceError("NeqSim returned a non-finite profile value.")
    return converted


def _is_insufficient_pressure_failure(error: Exception) -> bool:
    message = str(error).lower()
    return (
        "outlet pressure is negative" in message
        or "output pressure out" in message
        or "pressure became non-positive" in message
    )


def _finite(name: str, value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise PipelineInputError(f"{name} must be finite.")
    return result


def _positive_finite(name: str, value: Any) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise PipelineInputError(f"{name} must be greater than zero.")
    return result


def _nonnegative_finite(name: str, value: Any) -> float:
    result = _finite(name, value)
    if result < 0.0:
        raise PipelineInputError(f"{name} cannot be negative.")
    return result


def _positive_integer(name: str, value: Any) -> int:
    result = int(value)
    if result <= 0 or float(value) != result:
        raise PipelineInputError(f"{name} must be a positive integer.")
    return result
