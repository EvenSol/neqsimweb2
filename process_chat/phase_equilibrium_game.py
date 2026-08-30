"""Native phase-equilibrium and fluid-property training challenge.

NeqSim owns every flash and property calculation. This module defines one
reproducible rich-gas sample, validates the player controls, captures native
evidence, and translates the solved state into an explicit game assessment.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from time import perf_counter
from typing import Optional

_REQUIRED_JVM_OPENS = (
    "--add-opens=java.base/java.util=ALL-UNNAMED",
    "--add-opens=java.base/java.lang=ALL-UNNAMED",
    "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
    "--add-opens=java.base/java.io=ALL-UNNAMED",
)
_existing_java_options = os.environ.get("JAVA_TOOL_OPTIONS", "").split()
_missing_java_options = [
    option for option in _REQUIRED_JVM_OPENS if option not in _existing_java_options
]
if _missing_java_options:
    os.environ["JAVA_TOOL_OPTIONS"] = " ".join(
        [*_existing_java_options, *_missing_java_options]
    )

from .process_model import NeqSimProcessModel


CHALLENGE_NAME = "Catch the Retrograde Window"
CHALLENGE_TIMEOUT_MS = 60_000
MIN_TEMPERATURE_C = -20.0
MAX_TEMPERATURE_C = 80.0
MIN_PRESSURE_BARA = 10.0
MAX_PRESSURE_BARA = 130.0

MIN_LIQUID_FRACTION_MOL_PCT = 16.0
MAX_LIQUID_FRACTION_MOL_PCT = 20.0
MIN_GAS_DENSITY_KG_M3 = 78.0
MAX_GAS_DENSITY_KG_M3 = 92.0
MIN_GAS_Z_FACTOR = 0.80
MAX_GAS_Z_FACTOR = 0.83
MIN_LIQUID_DENSITY_KG_M3 = 480.0
MAX_LIQUID_DENSITY_KG_M3 = 510.0
MAX_LIQUID_VISCOSITY_CP = 0.12
MAX_PHASE_FRACTION_CLOSURE_ERROR = 1.0e-10

RICH_GAS_COMPOSITION = {
    "CO2": 0.005,
    "methane": 0.720,
    "ethane": 0.080,
    "propane": 0.060,
    "i-butane": 0.025,
    "n-butane": 0.035,
    "i-pentane": 0.015,
    "n-pentane": 0.015,
    "n-hexane": 0.020,
    "n-heptane": 0.015,
    "n-octane": 0.010,
}


@dataclass(frozen=True)
class PhaseControls:
    """Player-adjustable temperature and pressure for one TP flash."""

    temperature_c: float = 20.0
    pressure_bara: float = 50.0


@dataclass(frozen=True)
class ComponentEquilibrium:
    """Feed and equilibrium phase composition for one component."""

    component: str
    feed_mole_fraction: float
    gas_mole_fraction: Optional[float]
    liquid_mole_fraction: Optional[float]
    k_value: Optional[float]


@dataclass(frozen=True)
class PhaseEvidence:
    """Native NeqSim phase state and fluid properties used by the game."""

    temperature_c: float
    pressure_bara: float
    phase_types: tuple[str, ...]
    gas_fraction_mol_pct: Optional[float]
    liquid_fraction_mol_pct: Optional[float]
    gas_density_kg_m3: Optional[float]
    liquid_density_kg_m3: Optional[float]
    gas_z_factor: Optional[float]
    gas_viscosity_cp: Optional[float]
    liquid_viscosity_cp: Optional[float]
    mixture_enthalpy_kj_kg: Optional[float]
    mixture_cp_kj_kgk: Optional[float]
    phase_fraction_closure_error: Optional[float]
    components: tuple[ComponentEquilibrium, ...] = ()


@dataclass(frozen=True)
class PhaseCheck:
    """One objective acceptance check with its solved evidence."""

    name: str
    passed: bool
    actual: str
    requirement: str


@dataclass(frozen=True)
class PhaseAssessment:
    """Transparent score and feedback for one flash result."""

    score: int
    won: bool
    grade: str
    checks: tuple[PhaseCheck, ...]
    guidance: tuple[str, ...]


@dataclass(frozen=True)
class PhaseChallengeRun:
    """One completed native TP-flash attempt."""

    controls: PhaseControls
    evidence: PhaseEvidence
    assessment: PhaseAssessment
    elapsed_seconds: float


def _finite_between(
    value: float,
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite.")
    if not minimum <= numeric <= maximum:
        raise ValueError(
            f"{name} must be between {minimum:g} and {maximum:g}."
        )
    return numeric


def validate_controls(controls: PhaseControls) -> PhaseControls:
    """Validate public game controls before starting native calculation."""
    if not isinstance(controls, PhaseControls):
        raise ValueError("Phase controls must use PhaseControls.")
    return PhaseControls(
        temperature_c=_finite_between(
            controls.temperature_c,
            "Temperature",
            MIN_TEMPERATURE_C,
            MAX_TEMPERATURE_C,
        ),
        pressure_bara=_finite_between(
            controls.pressure_bara,
            "Pressure",
            MIN_PRESSURE_BARA,
            MAX_PRESSURE_BARA,
        ),
    )


def _validate_timeout_ms(timeout_ms: int) -> int:
    if isinstance(timeout_ms, bool):
        raise ValueError("Timeout must be a positive integer number of milliseconds.")
    try:
        numeric = float(timeout_ms)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Timeout must be a positive integer number of milliseconds."
        ) from exc
    if not math.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
        raise ValueError("Timeout must be a positive integer number of milliseconds.")
    return int(numeric)


def _safe_float(callback) -> Optional[float]:
    try:
        value = float(callback())
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _formatted(value: Optional[float], unit: str, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "unavailable"
    return f"{value:.{digits}f} {unit}".strip()


def _between(value: Optional[float], minimum: float, maximum: float) -> bool:
    return (
        value is not None
        and math.isfinite(value)
        and minimum <= value <= maximum
    )


def _below(value: Optional[float], maximum: float) -> bool:
    return value is not None and math.isfinite(value) and value < maximum


def assess_phase_evidence(evidence: PhaseEvidence) -> PhaseAssessment:
    """Score native evidence without inferring or repairing missing values."""
    two_phase = set(evidence.phase_types) == {"gas", "oil"}
    closure_passed = _between(
        evidence.phase_fraction_closure_error,
        0.0,
        MAX_PHASE_FRACTION_CLOSURE_ERROR,
    )
    checks = (
        PhaseCheck(
            "Equilibrium phase state",
            two_phase,
            " + ".join(evidence.phase_types) if evidence.phase_types else "unavailable",
            "exactly gas + hydrocarbon liquid",
        ),
        PhaseCheck(
            "Condensate split",
            _between(
                evidence.liquid_fraction_mol_pct,
                MIN_LIQUID_FRACTION_MOL_PCT,
                MAX_LIQUID_FRACTION_MOL_PCT,
            ),
            _formatted(evidence.liquid_fraction_mol_pct, "mol%"),
            f"{MIN_LIQUID_FRACTION_MOL_PCT:.0f}–{MAX_LIQUID_FRACTION_MOL_PCT:.0f} mol%",
        ),
        PhaseCheck(
            "Gas density",
            _between(
                evidence.gas_density_kg_m3,
                MIN_GAS_DENSITY_KG_M3,
                MAX_GAS_DENSITY_KG_M3,
            ),
            _formatted(evidence.gas_density_kg_m3, "kg/m³", 2),
            f"{MIN_GAS_DENSITY_KG_M3:.0f}–{MAX_GAS_DENSITY_KG_M3:.0f} kg/m³",
        ),
        PhaseCheck(
            "Gas compressibility",
            _between(
                evidence.gas_z_factor,
                MIN_GAS_Z_FACTOR,
                MAX_GAS_Z_FACTOR,
            ),
            _formatted(evidence.gas_z_factor, "", 4),
            f"Z = {MIN_GAS_Z_FACTOR:.2f}–{MAX_GAS_Z_FACTOR:.2f}",
        ),
        PhaseCheck(
            "Liquid density",
            _between(
                evidence.liquid_density_kg_m3,
                MIN_LIQUID_DENSITY_KG_M3,
                MAX_LIQUID_DENSITY_KG_M3,
            ),
            _formatted(evidence.liquid_density_kg_m3, "kg/m³", 2),
            f"{MIN_LIQUID_DENSITY_KG_M3:.0f}–{MAX_LIQUID_DENSITY_KG_M3:.0f} kg/m³",
        ),
        PhaseCheck(
            "Liquid viscosity",
            _below(evidence.liquid_viscosity_cp, MAX_LIQUID_VISCOSITY_CP),
            _formatted(evidence.liquid_viscosity_cp, "cP", 4),
            f"< {MAX_LIQUID_VISCOSITY_CP:.2f} cP",
        ),
        PhaseCheck(
            "Phase-fraction closure",
            closure_passed,
            _formatted(evidence.phase_fraction_closure_error, "", 3),
            f"≤ {MAX_PHASE_FRACTION_CLOSURE_ERROR:.0e}",
        ),
    )
    weights = (180, 220, 140, 140, 120, 100, 100)
    score = sum(weight for weight, check in zip(weights, checks) if check.passed)
    won = all(check.passed for check in checks)

    guidance: list[str] = []
    if not two_phase:
        if evidence.phase_types == ("gas",):
            guidance.append(
                "The sample is single-phase gas. Cool it or explore a different "
                "pressure to enter the phase envelope."
            )
        elif evidence.phase_types == ("oil",):
            guidance.append(
                "The sample is single-phase liquid. Warm it or reduce pressure "
                "to recover a gas phase."
            )
        else:
            guidance.append(
                "NeqSim did not return the required gas–oil equilibrium state; "
                "this attempt cannot win."
            )
    liquid_fraction = evidence.liquid_fraction_mol_pct
    if liquid_fraction is None:
        guidance.append(
            "Condensate fraction is unavailable; establish both gas and oil "
            "phases first."
        )
    elif liquid_fraction < MIN_LIQUID_FRACTION_MOL_PCT:
        guidance.append(
            "Condensate recovery is low. Try cooling the sample or increasing "
            "pressure."
        )
    elif liquid_fraction > MAX_LIQUID_FRACTION_MOL_PCT:
        guidance.append("Too much liquid dropped out. Try warming the sample or reducing pressure.")
    gas_density = evidence.gas_density_kg_m3
    if gas_density is not None and gas_density < MIN_GAS_DENSITY_KG_M3:
        guidance.append("Gas density is low; increasing pressure is the strongest local lever.")
    elif gas_density is not None and gas_density > MAX_GAS_DENSITY_KG_M3:
        guidance.append(
            "Gas density is high; reduce pressure while protecting the "
            "condensate target."
        )
    if not checks[3].passed:
        guidance.append(
            "Z-factor is outside its band. Adjust pressure and temperature "
            "together, then re-flash."
        )
    if not checks[4].passed or not checks[5].passed:
        guidance.append(
            "The liquid property targets favor a warmer, denser operating "
            "window than the starting point."
        )
    if not closure_passed:
        guidance.append("Phase-fraction evidence is missing or does not close; discard the result.")
    if won:
        guidance = [
            "Retrograde window captured. Compare the gas and liquid "
            "compositions: K > 1 favors vapor, while K < 1 favors liquid."
        ]

    grade = (
        "Phase window captured"
        if won
        else "Close — refine the flash conditions"
        if score >= 650
        else "Keep exploring the phase envelope"
    )
    return PhaseAssessment(
        score=int(score),
        won=won,
        grade=grade,
        checks=checks,
        guidance=tuple(guidance),
    )


def _native_phase_evidence(controls: PhaseControls) -> PhaseEvidence:
    """Execute one NeqSim SRK TP flash and extract its solved state."""
    from neqsim import jneqsim
    from neqsim.thermo import TPflash

    fluid = jneqsim.thermo.system.SystemSrkEos(
        controls.temperature_c + 273.15,
        controls.pressure_bara,
    )
    for component, mole_fraction in RICH_GAS_COMPOSITION.items():
        fluid.addComponent(component, mole_fraction)
    fluid.setMixingRule(2)
    fluid.setMultiPhaseCheck(True)
    TPflash(fluid)
    fluid.init(3)
    fluid.initProperties()

    phase_types = tuple(
        phase_type
        for phase_type in ("gas", "oil", "aqueous", "solid")
        if bool(fluid.hasPhaseType(phase_type))
    )
    gas = fluid.getPhase("gas") if "gas" in phase_types else None
    liquid = fluid.getPhase("oil") if "oil" in phase_types else None

    gas_beta = _safe_float(gas.getBeta) if gas is not None else None
    liquid_beta = _safe_float(liquid.getBeta) if liquid is not None else None
    phase_betas = [
        _safe_float(fluid.getPhase(index).getBeta)
        for index in range(int(fluid.getNumberOfPhases()))
    ]
    phase_fraction_closure_error = (
        abs(sum(phase_betas) - 1.0)
        if phase_betas and all(value is not None for value in phase_betas)
        else None
    )

    components: list[ComponentEquilibrium] = []
    for component, feed_fraction in RICH_GAS_COMPOSITION.items():
        gas_fraction = (
            _safe_float(lambda name=component: gas.getComponent(name).getx())
            if gas is not None
            else None
        )
        liquid_fraction = (
            _safe_float(lambda name=component: liquid.getComponent(name).getx())
            if liquid is not None
            else None
        )
        k_value = (
            gas_fraction / liquid_fraction
            if gas_fraction is not None
            and liquid_fraction is not None
            and liquid_fraction > 0.0
            else None
        )
        components.append(
            ComponentEquilibrium(
                component=component,
                feed_mole_fraction=feed_fraction,
                gas_mole_fraction=gas_fraction,
                liquid_mole_fraction=liquid_fraction,
                k_value=k_value,
            )
        )

    return PhaseEvidence(
        temperature_c=controls.temperature_c,
        pressure_bara=controls.pressure_bara,
        phase_types=phase_types,
        gas_fraction_mol_pct=gas_beta * 100.0 if gas_beta is not None else None,
        liquid_fraction_mol_pct=(
            liquid_beta * 100.0 if liquid_beta is not None else None
        ),
        gas_density_kg_m3=(
            _safe_float(lambda: gas.getDensity("kg/m3")) if gas is not None else None
        ),
        liquid_density_kg_m3=(
            _safe_float(lambda: liquid.getDensity("kg/m3"))
            if liquid is not None
            else None
        ),
        gas_z_factor=_safe_float(gas.getZ) if gas is not None else None,
        gas_viscosity_cp=(
            _safe_float(lambda: gas.getViscosity("cP")) if gas is not None else None
        ),
        liquid_viscosity_cp=(
            _safe_float(lambda: liquid.getViscosity("cP"))
            if liquid is not None
            else None
        ),
        mixture_enthalpy_kj_kg=_safe_float(lambda: fluid.getEnthalpy("kJ/kg")),
        mixture_cp_kj_kgk=_safe_float(lambda: fluid.getCp("kJ/kgK")),
        phase_fraction_closure_error=phase_fraction_closure_error,
        components=tuple(components),
    )


def run_phase_challenge(
    controls: PhaseControls,
    *,
    timeout_ms: int = CHALLENGE_TIMEOUT_MS,
) -> PhaseChallengeRun:
    """Run one bounded NeqSim phase-equilibrium game attempt."""
    validated_controls = validate_controls(controls)
    validated_timeout = _validate_timeout_ms(timeout_ms)
    started = perf_counter()

    def _bounded_flash() -> PhaseEvidence:
        try:
            return _native_phase_evidence(validated_controls)
        finally:
            # JPype auto-attaches Python worker threads to the JVM. Explicitly
            # detach this short-lived watchdog worker so command-line tests and
            # batch callers can shut down cleanly after the flash completes.
            try:
                import jpype

                if jpype.java.lang.Thread.isAttached():
                    jpype.java.lang.Thread.detach()
            except Exception:
                pass

    evidence = NeqSimProcessModel._run_bounded_call(
        _bounded_flash,
        validated_timeout,
        operation="phase-equilibrium flash",
    )
    assessment = assess_phase_evidence(evidence)
    return PhaseChallengeRun(
        controls=validated_controls,
        evidence=evidence,
        assessment=assessment,
        elapsed_seconds=perf_counter() - started,
    )
