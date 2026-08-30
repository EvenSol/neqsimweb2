"""Native NeqSim simulation and scoring for Plant Operator challenges.

The game layer deliberately stays thin: NeqSim owns the thermodynamics,
equipment calculations, process convergence, and conservation evidence.  This
module only defines a reproducible training case and translates its solved
engineering evidence into a transparent challenge score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from .process_builder import ProcessBuilder
from .process_model import ModelRunResult, NeqSimProcessModel, ProcessRunTimeoutError


CHALLENGE_NAME = "The 10% Throughput Challenge"
BASELINE_FLOW_KG_HR = 100_000.0
TARGET_FLOW_KG_HR = 110_000.0
MIN_EXPORT_PRESSURE_BARA = 128.0
MAX_EXPORT_TEMPERATURE_C = 45.0
MAX_COMPRESSOR_DISCHARGE_TEMPERATURE_C = 120.0
MAX_TOTAL_POWER_KW = 4_500.0
MAX_SPECIFIC_POWER_KWH_PER_TONNE = 41.0
MAX_COOLING_DUTY_KW = 7_000.0
MAX_BALANCE_ERROR_PCT = 0.10
CHALLENGE_TIMEOUT_MS = 180_000


@dataclass(frozen=True)
class ChallengeControls:
    """Player-adjustable operating decisions for the first challenge."""

    feed_flow_kg_hr: float = BASELINE_FLOW_KG_HR
    stage_1_pressure_bara: float = 80.0
    stage_2_pressure_bara: float = 130.0
    intercooler_temperature_c: float = 35.0
    export_temperature_c: float = 40.0


@dataclass(frozen=True)
class ChallengeEvidence:
    """Solved evidence used by the score; all values come from NeqSim."""

    feed_flow_kg_hr: float
    export_pressure_bara: float
    export_temperature_c: float
    stage_1_discharge_temperature_c: float
    stage_2_discharge_temperature_c: float
    total_power_kw: float
    total_cooling_duty_kw: float
    specific_power_kwh_per_tonne: float
    mass_balance_error_pct: float
    energy_balance_error_pct: float
    native_violations: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChallengeCheck:
    """One objective engineering acceptance check."""

    name: str
    passed: bool
    actual: str
    requirement: str


@dataclass(frozen=True)
class ChallengeAssessment:
    """Scored assessment returned to the game interface."""

    score: int
    won: bool
    grade: str
    checks: tuple[ChallengeCheck, ...]
    guidance: tuple[str, ...]


@dataclass
class ChallengeRun:
    """A reproducible NeqSim run plus the game-facing assessment."""

    controls: ChallengeControls
    spec: dict[str, Any]
    builder: ProcessBuilder
    model: NeqSimProcessModel
    result: ModelRunResult
    evidence: ChallengeEvidence
    assessment: ChallengeAssessment
    elapsed_seconds: float


def _finite_between(value: float, name: str, minimum: float, maximum: float) -> float:
    """Return a validated finite float inside an inclusive range."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric.")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite.")
    if not minimum <= numeric_value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum:g} and {maximum:g}."
        )
    return numeric_value


def validate_controls(controls: ChallengeControls) -> ChallengeControls:
    """Validate the public game input contract before starting the JVM."""
    if not isinstance(controls, ChallengeControls):
        raise ValueError("Challenge controls must use ChallengeControls.")
    values = {
        "feed_flow_kg_hr": _finite_between(
            controls.feed_flow_kg_hr,
            "Feed flow",
            90_000.0,
            125_000.0,
        ),
        "stage_1_pressure_bara": _finite_between(
            controls.stage_1_pressure_bara,
            "Stage 1 pressure",
            60.0,
            110.0,
        ),
        "stage_2_pressure_bara": _finite_between(
            controls.stage_2_pressure_bara,
            "Stage 2 pressure",
            115.0,
            145.0,
        ),
        "intercooler_temperature_c": _finite_between(
            controls.intercooler_temperature_c,
            "Intercooler temperature",
            20.0,
            60.0,
        ),
        "export_temperature_c": _finite_between(
            controls.export_temperature_c,
            "Export temperature",
            20.0,
            55.0,
        ),
    }
    if values["stage_1_pressure_bara"] >= values["stage_2_pressure_bara"]:
        raise ValueError("Stage 1 pressure must be below Stage 2 pressure.")
    return ChallengeControls(**values)


def build_challenge_spec(controls: ChallengeControls) -> dict[str, Any]:
    """Build the minimum reproducible ProcessBuilder specification."""
    controls = validate_controls(controls)
    fluid_spec = {
        "eos_model": "srk",
        "mixing_rule": 2,
        "components": {
            "nitrogen": 0.010,
            "CO2": 0.020,
            "methane": 0.850,
            "ethane": 0.060,
            "propane": 0.030,
            "i-butane": 0.008,
            "n-butane": 0.012,
            "i-pentane": 0.004,
            "n-pentane": 0.003,
            "n-hexane": 0.003,
        },
        "composition_basis": "mole_fraction",
        "temperature_C": 30.0,
        "pressure_bara": 50.0,
        "total_flow": controls.feed_flow_kg_hr,
        "flow_unit": "kg/hr",
    }
    process = [
        {
            "name": "feed gas",
            "type": "stream",
            "params": {
                "temperature_C": 30.0,
                "pressure_bara": 50.0,
                "flow_rate": controls.feed_flow_kg_hr,
                "flow_unit": "kg/hr",
            },
        },
        {
            "name": "inlet scrubber",
            "type": "separator",
            "outlet": "gas",
        },
        {
            "name": "compressor stage 1",
            "type": "compressor",
            "params": {
                "outlet_pressure_bara": controls.stage_1_pressure_bara,
                "isentropic_efficiency": 0.78,
            },
        },
        {
            "name": "intercooler",
            "type": "cooler",
            "params": {
                "outlet_temperature_C": controls.intercooler_temperature_c,
                "pressure_drop_bar": 0.0,
            },
        },
        {
            "name": "interstage scrubber",
            "type": "separator",
            "outlet": "gas",
        },
        {
            "name": "compressor stage 2",
            "type": "compressor",
            "params": {
                "outlet_pressure_bara": controls.stage_2_pressure_bara,
                "isentropic_efficiency": 0.78,
            },
        },
        {
            "name": "export cooler",
            "type": "cooler",
            "params": {
                "outlet_temperature_C": controls.export_temperature_c,
                "pressure_drop_bar": 0.0,
            },
        },
    ]
    return {
        "name": CHALLENGE_NAME,
        "description": (
            "Synthetic gas-compression training case. Increase throughput by "
            "10% while respecting export, power, temperature, and conservation "
            "constraints."
        ),
        "assumptions": [
            "Steady-state training scenario using the SRK equation of state.",
            "Pressures are absolute (bara).",
            "Flow is mass flow in kg/hr.",
            "Both compressor isentropic efficiencies are fixed at 0.78.",
            "The score is educational and is not a design certification.",
        ],
        "fluid": fluid_spec,
        "process": process,
    }


def _outlet_state(model: NeqSimProcessModel, unit_name: str) -> tuple[float, float]:
    """Return temperature in degC and pressure in bara for one unit outlet."""
    unit = model.get_unit(unit_name)
    outlet = None
    for getter_name in ("getOutletStream", "getOutStream", "getGasOutStream"):
        if not hasattr(unit, getter_name):
            continue
        try:
            outlet = getattr(unit, getter_name)()
        except Exception:
            continue
        if outlet is not None:
            break
    if outlet is None:
        raise RuntimeError(f"{unit_name} did not expose a solved outlet stream.")
    return (
        float(outlet.getTemperature("C")),
        float(outlet.getPressure("bara")),
    )


def collect_evidence(
    controls: ChallengeControls,
    model: NeqSimProcessModel,
    result: ModelRunResult,
) -> ChallengeEvidence:
    """Collect only solved NeqSim values used by the challenge score."""
    stage_1_temperature_c, _ = _outlet_state(model, "compressor stage 1")
    stage_2_temperature_c, _ = _outlet_state(model, "compressor stage 2")
    export_temperature_c, export_pressure_bara = _outlet_state(
        model,
        "export cooler",
    )

    total_power_kw = abs(float(result.kpis["total_power_kW"].value))
    total_cooling_duty_kw = abs(float(result.kpis["total_duty_kW"].value))
    specific_power = total_power_kw * 1000.0 / controls.feed_flow_kg_hr
    mass_balance = result.kpis.get("mass_balance_pct")
    energy_balance = result.kpis.get("energy_balance_pct")
    native_violations = tuple(
        constraint.name
        for constraint in result.constraints
        if str(constraint.status).strip().upper() not in {"OK", "WARN"}
        and constraint.name not in {"mass_balance", "energy_balance"}
    )
    return ChallengeEvidence(
        feed_flow_kg_hr=controls.feed_flow_kg_hr,
        export_pressure_bara=export_pressure_bara,
        export_temperature_c=export_temperature_c,
        stage_1_discharge_temperature_c=stage_1_temperature_c,
        stage_2_discharge_temperature_c=stage_2_temperature_c,
        total_power_kw=total_power_kw,
        total_cooling_duty_kw=total_cooling_duty_kw,
        specific_power_kwh_per_tonne=specific_power,
        mass_balance_error_pct=(
            float(mass_balance.value) if mass_balance is not None else math.inf
        ),
        energy_balance_error_pct=(
            float(energy_balance.value) if energy_balance is not None else math.inf
        ),
        native_violations=native_violations,
    )


def _bounded_score(value: float, best: float, worst: float, points: float) -> float:
    """Scale lower-is-better evidence into a bounded point contribution."""
    if not math.isfinite(value) or worst <= best:
        return 0.0
    fraction = (worst - value) / (worst - best)
    return points * min(1.0, max(0.0, fraction))


def assess_challenge(evidence: ChallengeEvidence) -> ChallengeAssessment:
    """Evaluate one solved operating strategy with a transparent score."""
    maximum_discharge_temperature_c = max(
        evidence.stage_1_discharge_temperature_c,
        evidence.stage_2_discharge_temperature_c,
    )
    checks = (
        ChallengeCheck(
            "Throughput target",
            evidence.feed_flow_kg_hr >= TARGET_FLOW_KG_HR,
            f"{evidence.feed_flow_kg_hr:,.0f} kg/hr",
            f"at least {TARGET_FLOW_KG_HR:,.0f} kg/hr",
        ),
        ChallengeCheck(
            "Export pressure",
            evidence.export_pressure_bara >= MIN_EXPORT_PRESSURE_BARA,
            f"{evidence.export_pressure_bara:.2f} bara",
            f"at least {MIN_EXPORT_PRESSURE_BARA:.0f} bara",
        ),
        ChallengeCheck(
            "Export temperature",
            evidence.export_temperature_c <= MAX_EXPORT_TEMPERATURE_C,
            f"{evidence.export_temperature_c:.2f} °C",
            f"at most {MAX_EXPORT_TEMPERATURE_C:.0f} °C",
        ),
        ChallengeCheck(
            "Compressor discharge temperature",
            maximum_discharge_temperature_c
            <= MAX_COMPRESSOR_DISCHARGE_TEMPERATURE_C,
            f"{maximum_discharge_temperature_c:.2f} °C maximum",
            (
                "at most "
                f"{MAX_COMPRESSOR_DISCHARGE_TEMPERATURE_C:.0f} °C"
            ),
        ),
        ChallengeCheck(
            "Compression power",
            evidence.total_power_kw <= MAX_TOTAL_POWER_KW,
            f"{evidence.total_power_kw:,.0f} kW",
            f"at most {MAX_TOTAL_POWER_KW:,.0f} kW",
        ),
        ChallengeCheck(
            "Specific compression energy",
            evidence.specific_power_kwh_per_tonne
            <= MAX_SPECIFIC_POWER_KWH_PER_TONNE,
            f"{evidence.specific_power_kwh_per_tonne:.2f} kWh/tonne",
            f"at most {MAX_SPECIFIC_POWER_KWH_PER_TONNE:.0f} kWh/tonne",
        ),
        ChallengeCheck(
            "Cooling-system load",
            evidence.total_cooling_duty_kw <= MAX_COOLING_DUTY_KW,
            f"{evidence.total_cooling_duty_kw:,.0f} kW",
            f"at most {MAX_COOLING_DUTY_KW:,.0f} kW",
        ),
        ChallengeCheck(
            "Mass balance",
            evidence.mass_balance_error_pct <= MAX_BALANCE_ERROR_PCT,
            f"{evidence.mass_balance_error_pct:.6g}% error",
            f"at most {MAX_BALANCE_ERROR_PCT:.2f}% error",
        ),
        ChallengeCheck(
            "Energy balance",
            evidence.energy_balance_error_pct <= MAX_BALANCE_ERROR_PCT,
            f"{evidence.energy_balance_error_pct:.6g}% error",
            f"at most {MAX_BALANCE_ERROR_PCT:.2f}% error",
        ),
        ChallengeCheck(
            "Native NeqSim checks",
            not evidence.native_violations,
            (
                "All native checks passed"
                if not evidence.native_violations
                else ", ".join(evidence.native_violations)
            ),
            "no failed or unavailable native constraints",
        ),
    )

    production_fraction = min(
        1.0,
        max(0.0, evidence.feed_flow_kg_hr / TARGET_FLOW_KG_HR),
    )
    production_points = 400.0 * production_fraction
    stretch_bonus = 50.0 * min(
        1.0,
        max(
            0.0,
            (evidence.feed_flow_kg_hr - TARGET_FLOW_KG_HR)
            / (TARGET_FLOW_KG_HR * 0.10),
        ),
    )
    energy_points = _bounded_score(
        evidence.specific_power_kwh_per_tonne,
        best=38.5,
        worst=MAX_SPECIFIC_POWER_KWH_PER_TONNE,
        points=250.0,
    )
    thermal_points = _bounded_score(
        maximum_discharge_temperature_c,
        best=70.0,
        worst=MAX_COMPRESSOR_DISCHARGE_TEMPERATURE_C,
        points=100.0,
    )
    cooling_points = _bounded_score(
        evidence.total_cooling_duty_kw,
        best=6_000.0,
        worst=MAX_COOLING_DUTY_KW,
        points=50.0,
    )
    integrity_checks = checks[-3:]
    integrity_points = 150.0 * (
        sum(check.passed for check in integrity_checks)
        / len(integrity_checks)
    )
    score = int(
        round(
            min(
                1000.0,
                production_points
                + stretch_bonus
                + energy_points
                + thermal_points
                + cooling_points
                + integrity_points,
            )
        )
    )
    won = all(check.passed for check in checks)
    if won and score >= 900:
        grade = "Outstanding operation"
    elif won:
        grade = "Challenge completed"
    elif score >= 700:
        grade = "Close — one more adjustment"
    elif score >= 500:
        grade = "Developing strategy"
    else:
        grade = "Plant constraints not controlled"

    failed_names = {check.name for check in checks if not check.passed}
    guidance: list[str] = []
    if "Throughput target" in failed_names:
        guidance.append("Increase feed flow to at least 110,000 kg/hr.")
    if "Export pressure" in failed_names:
        guidance.append("Raise the second-stage discharge-pressure setpoint.")
    if "Export temperature" in failed_names:
        guidance.append("Lower the export-cooler temperature setpoint.")
    if {
        "Compression power",
        "Specific compression energy",
    }.intersection(failed_names):
        ideal_interstage_pressure = math.sqrt(
            50.0 * evidence.export_pressure_bara
        )
        guidance.append(
            "Move the interstage pressure toward the equal-ratio value "
            f"of about {ideal_interstage_pressure:.0f} bara and keep the "
            "intercooler effective."
        )
    if "Compressor discharge temperature" in failed_names:
        guidance.append(
            "Reduce the intercooler setpoint or rebalance the pressure ratio."
        )
    if "Cooling-system load" in failed_names:
        guidance.append(
            "Reduce avoidable cooling load while keeping compression energy and "
            "discharge temperatures inside their limits."
        )
    if {"Mass balance", "Energy balance", "Native NeqSim checks"}.intersection(
        failed_names
    ):
        guidance.append(
            "Treat the run as invalid until the native validation evidence passes."
        )
    if not guidance:
        guidance.append(
            "All constraints pass. Try a higher throughput or lower specific "
            "energy to improve the score."
        )
    return ChallengeAssessment(
        score=score,
        won=won,
        grade=grade,
        checks=checks,
        guidance=tuple(guidance),
    )


def run_challenge(
    controls: ChallengeControls,
    *,
    timeout_ms: int = CHALLENGE_TIMEOUT_MS,
) -> ChallengeRun:
    """Build, solve, validate, and score one strategy within one time budget."""
    controls = validate_controls(controls)
    if isinstance(timeout_ms, bool):
        raise ValueError("Challenge timeout must be a positive integer.")
    try:
        timeout_ms = int(timeout_ms)
    except (TypeError, ValueError) as exc:
        raise ValueError("Challenge timeout must be a positive integer.") from exc
    if timeout_ms <= 0:
        raise ValueError("Challenge timeout must be a positive integer.")
    started = perf_counter()
    deadline = started + timeout_ms / 1000.0

    def remaining_budget_ms() -> int:
        remaining_ms = int((deadline - perf_counter()) * 1000.0)
        if remaining_ms <= 0:
            raise ProcessRunTimeoutError(
                f"Plant Operator run exceeded its {timeout_ms} ms budget; "
                "discard the partial model."
            )
        return remaining_ms

    spec = build_challenge_spec(controls)
    builder = ProcessBuilder()
    model = builder.build_from_spec_bounded(
        spec,
        timeout_ms=remaining_budget_ms(),
    )
    result = model.run_bounded(timeout_ms=remaining_budget_ms())
    evidence = model.run_bounded_operation(
        lambda: collect_evidence(controls, model, result),
        timeout_ms=remaining_budget_ms(),
        operation="Plant Operator evidence collection",
    )
    assessment = assess_challenge(evidence)
    return ChallengeRun(
        controls=controls,
        spec=spec,
        builder=builder,
        model=model,
        result=result,
        evidence=evidence,
        assessment=assessment,
        elapsed_seconds=perf_counter() - started,
    )
