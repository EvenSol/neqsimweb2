"""
Risk Analysis — equipment criticality, risk matrix, and Monte-Carlo availability.

Three layers (mirrors the NeqSim risk framework architecture):

  1. **Equipment failure modelling** – OREDA-based reliability data, failure
     modes (trip / degraded / maintenance).
  2. **Production impact analysis** – simulate each failure, measure production
     loss % → equipment criticality ranking.
  3. **Risk assessment** – 5×5 risk matrix + Monte-Carlo production availability.

Approach:
  * Try the Java-native NeqSim classes first (``RiskMatrix``,
    ``OperationalRiskSimulator``, ``ProductionImpactAnalyzer``).
  * If not available, fall back to a pure-Python implementation that
    re-uses ``NeqSimProcessModel.clone()`` / ``run()``.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .process_model import NeqSimProcessModel, KPI


# ──────────────────────────────────────────────
# Enums & constants
# ──────────────────────────────────────────────

class FailureType(str, Enum):
    TRIP = "TRIP"
    DEGRADED = "DEGRADED"
    MAINTENANCE = "MAINTENANCE"


class ProbabilityCategory(str, Enum):
    VERY_LOW = "VERY_LOW"       # < 0.1 /yr
    LOW = "LOW"                 # 0.1 – 0.5 /yr
    MEDIUM = "MEDIUM"           # 0.5 – 1.0 /yr
    HIGH = "HIGH"               # 1.0 – 2.0 /yr
    VERY_HIGH = "VERY_HIGH"     # > 2.0 /yr

    @classmethod
    def from_frequency(cls, freq: float) -> "ProbabilityCategory":
        if freq < 0.1:
            return cls.VERY_LOW
        if freq < 0.5:
            return cls.LOW
        if freq < 1.0:
            return cls.MEDIUM
        if freq < 2.0:
            return cls.HIGH
        return cls.VERY_HIGH


class ConsequenceCategory(str, Enum):
    NEGLIGIBLE = "NEGLIGIBLE"     # < 5 % loss
    MINOR = "MINOR"               # 5 – 20 %
    MODERATE = "MODERATE"         # 20 – 50 %
    MAJOR = "MAJOR"               # 50 – 80 %
    CATASTROPHIC = "CATASTROPHIC" # > 80 %

    @classmethod
    def from_production_loss(cls, pct: float) -> "ConsequenceCategory":
        if pct < 5:
            return cls.NEGLIGIBLE
        if pct < 20:
            return cls.MINOR
        if pct < 50:
            return cls.MODERATE
        if pct < 80:
            return cls.MAJOR
        return cls.CATASTROPHIC


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"
    EXTREME = "EXTREME"


# 5×5 risk score lookup (probability row × consequence column)
_PROB_INDEX = {
    ProbabilityCategory.VERY_LOW: 0,
    ProbabilityCategory.LOW: 1,
    ProbabilityCategory.MEDIUM: 2,
    ProbabilityCategory.HIGH: 3,
    ProbabilityCategory.VERY_HIGH: 4,
}
_CONS_INDEX = {
    ConsequenceCategory.NEGLIGIBLE: 0,
    ConsequenceCategory.MINOR: 1,
    ConsequenceCategory.MODERATE: 2,
    ConsequenceCategory.MAJOR: 3,
    ConsequenceCategory.CATASTROPHIC: 4,
}

# Score = (prob_idx + 1) * (cons_idx + 1)  → 1-25
def _risk_score(prob: ProbabilityCategory, cons: ConsequenceCategory) -> int:
    return (_PROB_INDEX[prob] + 1) * (_CONS_INDEX[cons] + 1)


def _risk_level(score: int) -> RiskLevel:
    if score <= 4:
        return RiskLevel.LOW
    if score <= 9:
        return RiskLevel.MEDIUM
    if score <= 14:
        return RiskLevel.HIGH
    if score <= 19:
        return RiskLevel.VERY_HIGH
    return RiskLevel.EXTREME


# ──────────────────────────────────────────────
# OREDA-style default reliability data
# ──────────────────────────────────────────────

# {equipment_java_class: (MTTF_hours, MTTR_hours)}
_OREDA_DEFAULTS: Dict[str, Tuple[float, float]] = {
    "Compressor":             (8_760,  24),
    "Pump":                   (17_520,  8),
    "ESPPump":                (17_520,  8),
    "Separator":              (43_800,  4),
    "TwoPhaseSeparator":      (43_800,  4),
    "ThreePhaseSeparator":    (43_800,  4),
    "GasScrubber":            (43_800,  4),
    "GasScrubberSimple":      (43_800,  4),
    "HeatExchanger":          (43_800, 12),
    "Cooler":                 (43_800, 12),
    "Heater":                 (43_800, 12),
    "ThrottlingValve":        (26_280,  4),
    "ControlValve":           (26_280,  4),
    "Expander":               (8_760,  48),
    "Mixer":                  (87_600,  2),
    "Splitter":               (87_600,  2),
    "Pipeline":               (87_600,  8),
    "AdiabaticPipe":          (87_600,  8),
}

# Default capacity factor when degraded (per equipment family)
_DEGRADED_CAPACITY: Dict[str, float] = {
    "Compressor": 0.7,
    "Pump": 0.8,
    "ESPPump": 0.8,
    "Separator": 0.7,
    "TwoPhaseSeparator": 0.7,
    "ThreePhaseSeparator": 0.7,
    "GasScrubber": 0.7,
    "GasScrubberSimple": 0.7,
    "HeatExchanger": 0.9,
    "Cooler": 0.9,
    "Heater": 0.9,
    "ThrottlingValve": 0.5,
    "ControlValve": 0.5,
    "Expander": 0.7,
}


# ──────────────────────────────────────────────
# Result data-classes
# ──────────────────────────────────────────────

@dataclass
class EquipmentReliability:
    """Reliability profile for one piece of equipment."""
    name: str
    equipment_type: str
    mttf_hours: float
    mttr_hours: float
    failure_rate_per_year: float   # λ = 8760 / MTTF
    availability: float            # A = MTTF / (MTTF + MTTR)


@dataclass
class FailureImpact:
    """Result of simulating a single equipment failure."""
    equipment_name: str
    equipment_type: str
    failure_type: str              # TRIP or DEGRADED
    production_loss_pct: float     # 0–100
    original_production_kg_hr: float
    failed_production_kg_hr: float
    criticality_index: float       # 0–1  (loss / max_loss across all equipment)
    affected_downstream: List[str] = field(default_factory=list)


@dataclass
class RiskMatrixItem:
    """One cell in the risk matrix."""
    equipment_name: str
    probability: ProbabilityCategory
    consequence: ConsequenceCategory
    risk_score: int
    risk_level: RiskLevel
    failure_rate_per_year: float
    production_loss_pct: float
    annual_cost_usd: float = 0.0   # λ × (prod_loss × price + repair_cost)


@dataclass
class MonteCarloResult:
    """Summary of a Monte-Carlo availability simulation."""
    iterations: int
    horizon_days: int
    expected_availability_pct: float
    expected_production_pct: float   # % of design
    p10_production_pct: float
    p50_production_pct: float
    p90_production_pct: float
    expected_downtime_hours_year: float
    expected_failure_events_year: float
    equipment_downtime_contribution: Dict[str, float] = field(default_factory=dict)


@dataclass
class RiskAnalysisResult:
    """Top-level result of the full risk analysis."""
    equipment_reliability: List[EquipmentReliability]
    failure_impacts: List[FailureImpact]
    risk_matrix: List[RiskMatrixItem]
    monte_carlo: Optional[MonteCarloResult]
    system_availability_pct: float
    most_critical_equipment: str
    summary_message: str = ""


# ──────────────────────────────────────────────
# Reliability helpers
# ──────────────────────────────────────────────

def _lookup_reliability(name: str, java_class: str,
                        java_unit: Any = None) -> EquipmentReliability:
    """
    Build an ``EquipmentReliability`` for one unit.

    Priority:
      1. Java ``ReliabilityDataSource`` (if available)
      2. OREDA lookup table (built-in)
    """
    mttf, mttr = _OREDA_DEFAULTS.get(java_class, (43_800, 8))

    # Try Java source
    try:
        from neqsim import jneqsim
        src = jneqsim.process.equipment.failure.ReliabilityDataSource.getInstance()
        mttf = float(src.getMTTF(java_class))
        mttr = float(src.getMTTR(java_class))
    except Exception:
        pass

    failure_rate = 8_760.0 / mttf   # failures / year
    availability = mttf / (mttf + mttr)

    return EquipmentReliability(
        name=name,
        equipment_type=java_class,
        mttf_hours=mttf,
        mttr_hours=mttr,
        failure_rate_per_year=failure_rate,
        availability=availability,
    )


# ──────────────────────────────────────────────
# Production-impact analysis (Python fallback)
# ──────────────────────────────────────────────

def _find_product_stream(model: NeqSimProcessModel) -> Optional[str]:
    """Heuristic: the last stream in the model is the product."""
    streams = model.list_streams()
    if not streams:
        return None
    return streams[-1].name


def _get_production(model: NeqSimProcessModel,
                    product_stream: str) -> float:
    """Return product stream flow in kg/hr.

    Reads directly from the Java stream object — does NOT call
    ``model.run()``.  The caller is responsible for ensuring the model
    is in a solved state (either the live model after a previous run,
    or a clone after ``clone.run()``).
    """
    # Primary: read directly from Java stream
    try:
        s = model.get_stream(product_stream)
        if s:
            val = float(s.getFlowRate("kg/hr"))
            if val > 0:
                return val
    except Exception:
        pass

    # Fallback: check KPI cache
    try:
        result = getattr(model, '_last_run_result', None)
        if result and result.kpis:
            key = f"{product_stream}.flow_kg_hr"
            if key in result.kpis and result.kpis[key].value is not None:
                return result.kpis[key].value
    except Exception:
        pass

    return 0.0


# Heuristic production-loss estimate when dynamic simulation cannot
# effectively model an equipment trip (e.g. upstream equipment overwrites
# the disabled stream during a sequential run).
_TRIP_LOSS_HEURISTIC: Dict[str, float] = {
    "Compressor":          100.0,
    "Pump":                100.0,
    "ESPPump":             100.0,
    "Expander":            100.0,
    "Separator":            80.0,
    "TwoPhaseSeparator":    80.0,
    "ThreePhaseSeparator":  80.0,
    "GasScrubber":          60.0,
    "GasScrubberSimple":    60.0,
    "HeatExchanger":        40.0,
    "Cooler":               30.0,
    "Heater":               30.0,
    "ThrottlingValve":     100.0,
    "ControlValve":        100.0,
    "Mixer":                50.0,
    "Splitter":             50.0,
    "Pipeline":            100.0,
    "AdiabaticPipe":       100.0,
}


def _simulate_trip(model: NeqSimProcessModel,
                   unit_name: str,
                   product_stream: str,
                   baseline_flow: float) -> FailureImpact:
    """
    Simulate a TRIP (complete shutdown) of *unit_name*.

    Strategy:
      1. Clone the model.
      2. Apply equipment-specific "trip" action (close valve, bypass
         compressor, zero heater duty, etc.).
      3. Re-run and compare product flow to baseline.
      4. If the simulation still shows 0 % loss (common when upstream
         equipment overwrites the modified stream during sequential
         execution), fall back to a heuristic loss estimate based on
         equipment type.
    """
    clone = model.clone()
    if clone is None:
        return FailureImpact(
            equipment_name=unit_name,
            equipment_type="",
            failure_type="TRIP",
            production_loss_pct=100.0,
            original_production_kg_hr=baseline_flow,
            failed_production_kg_hr=0.0,
            criticality_index=1.0,
        )

    java_class = ""
    applied = False
    try:
        unit = clone.get_unit(unit_name)
        if unit is not None:
            try:
                java_class = str(unit.getClass().getSimpleName())
            except Exception:
                java_class = type(unit).__name__

            # --- Equipment-specific trip strategies ---
            if java_class in ("ThrottlingValve", "ControlValve"):
                if hasattr(unit, "setPercentValveOpening"):
                    try:
                        unit.setPercentValveOpening(0)  # fully closed
                        applied = True
                    except Exception:
                        pass
            elif java_class == "Compressor":
                # Bypass: outlet pressure = inlet pressure (no compression)
                try:
                    inlet = unit.getInletStream()
                    p_in = float(inlet.getPressure("bara"))
                    unit.setOutletPressure(p_in)
                    applied = True
                except Exception:
                    pass
            elif java_class in ("Pump", "ESPPump"):
                try:
                    inlet = unit.getInletStream()
                    p_in = float(inlet.getPressure("bara"))
                    if hasattr(unit, "setOutletPressure"):
                        unit.setOutletPressure(p_in)
                        applied = True
                except Exception:
                    pass
            elif java_class == "Expander":
                try:
                    inlet = unit.getInletStream()
                    p_in = float(inlet.getPressure("bara"))
                    unit.setOutletPressure(p_in)
                    applied = True
                except Exception:
                    pass
            elif java_class in ("Cooler", "Heater", "HeatExchanger"):
                # No heat transfer – outlet temperature ≈ inlet temperature
                for m in ("getInletStream", "getInStream", "getFeedStream"):
                    if hasattr(unit, m):
                        try:
                            s = getattr(unit, m)()
                            t_in = float(s.getTemperature("C"))
                            if hasattr(unit, "setOutTemperature"):
                                unit.setOutTemperature(t_in, "C")
                                applied = True
                            break
                        except Exception:
                            pass

            # Generic fallback: try setOff / inlet-flow reduction
            if not applied:
                for method_name in ("setOff", "setIsActive"):
                    if hasattr(unit, method_name):
                        try:
                            if "Active" in method_name:
                                getattr(unit, method_name)(False)
                            else:
                                getattr(unit, method_name)()
                            applied = True
                            break
                        except Exception:
                            pass

            if not applied:
                for m in ("getInletStream", "getInStream", "getFeedStream"):
                    if hasattr(unit, m):
                        try:
                            s = getattr(unit, m)()
                            s.setFlowRate(0.001, "kg/hr")
                            applied = True
                            break
                        except Exception:
                            pass
    except Exception:
        pass

    # Re-run the cloned model
    try:
        clone.run()
    except Exception:
        pass

    failed_flow = _get_production(clone, product_stream)
    loss_pct = (
        max(0.0, (baseline_flow - failed_flow) / baseline_flow * 100.0)
        if baseline_flow > 0 else 0.0
    )

    # If the simulation shows negligible loss (< 1 %), the trip action
    # likely did not propagate through the sequential process run.
    # Apply a heuristic loss estimate instead.
    if loss_pct < 1.0 and java_class:
        loss_pct = _TRIP_LOSS_HEURISTIC.get(java_class, 50.0)
        failed_flow = baseline_flow * (1.0 - loss_pct / 100.0)

    return FailureImpact(
        equipment_name=unit_name,
        equipment_type=java_class,
        failure_type="TRIP",
        production_loss_pct=loss_pct,
        original_production_kg_hr=baseline_flow,
        failed_production_kg_hr=failed_flow,
        criticality_index=0.0,  # will be normalized later
    )


def _simulate_degraded(model: NeqSimProcessModel,
                       unit_name: str,
                       product_stream: str,
                       baseline_flow: float,
                       capacity_factor: float = 0.7) -> FailureImpact:
    """
    Simulate a DEGRADED failure (reduced capacity).

    Uses equipment-specific strategies where possible (reduced valve
    opening, lower compressor efficiency, etc.).  Falls back to a
    heuristic estimate when the dynamic simulation cannot propagate
    the degradation.
    """
    clone = model.clone()
    if clone is None:
        return FailureImpact(
            equipment_name=unit_name,
            equipment_type="",
            failure_type="DEGRADED",
            production_loss_pct=(1 - capacity_factor) * 100,
            original_production_kg_hr=baseline_flow,
            failed_production_kg_hr=baseline_flow * capacity_factor,
            criticality_index=0.0,
        )

    java_class = ""
    applied = False
    try:
        unit = clone.get_unit(unit_name)
        if unit is not None:
            try:
                java_class = str(unit.getClass().getSimpleName())
            except Exception:
                java_class = type(unit).__name__

            # --- Equipment-specific degraded strategies ---
            if java_class in ("ThrottlingValve", "ControlValve"):
                if hasattr(unit, "setPercentValveOpening"):
                    try:
                        unit.setPercentValveOpening(capacity_factor * 100)
                        applied = True
                    except Exception:
                        pass
            elif java_class == "Compressor":
                if hasattr(unit, "setIsentropicEfficiency"):
                    try:
                        eff = float(unit.getIsentropicEfficiency())
                        unit.setIsentropicEfficiency(eff * capacity_factor)
                        applied = True
                    except Exception:
                        pass
            elif java_class in ("Cooler", "Heater", "HeatExchanger"):
                # Reduce duty proportionally
                if hasattr(unit, "getDuty") and hasattr(unit, "setDuty"):
                    try:
                        duty = float(unit.getDuty())
                        unit.setDuty(duty * capacity_factor)
                        applied = True
                    except Exception:
                        pass

            # Generic fallback: reduce inlet flow
            if not applied:
                for m in ("getInletStream", "getInStream", "getFeedStream"):
                    if hasattr(unit, m):
                        try:
                            s = getattr(unit, m)()
                            current_flow = float(s.getFlowRate("kg/hr"))
                            s.setFlowRate(current_flow * capacity_factor, "kg/hr")
                            applied = True
                            break
                        except Exception:
                            pass
    except Exception:
        pass

    try:
        clone.run()
    except Exception:
        pass

    failed_flow = _get_production(clone, product_stream)
    loss_pct = (
        max(0.0, (baseline_flow - failed_flow) / baseline_flow * 100.0)
        if baseline_flow > 0 else 0.0
    )

    # If simulation shows negligible loss, use the capacity factor directly
    if loss_pct < 1.0:
        loss_pct = (1.0 - capacity_factor) * 100.0
        failed_flow = baseline_flow * capacity_factor

    return FailureImpact(
        equipment_name=unit_name,
        equipment_type=java_class,
        failure_type="DEGRADED",
        production_loss_pct=loss_pct,
        original_production_kg_hr=baseline_flow,
        failed_production_kg_hr=failed_flow,
        criticality_index=0.0,
    )


# ──────────────────────────────────────────────
# Monte-Carlo availability simulation (Python)
# ──────────────────────────────────────────────

def _monte_carlo_availability(
    equipment: List[EquipmentReliability],
    impact_map: Dict[str, float],    # name → production_loss_pct (for trip)
    iterations: int = 1_000,
    horizon_days: int = 365,
    seed: Optional[int] = 42,
) -> MonteCarloResult:
    """
    Lightweight Monte-Carlo simulation of system availability.

    For each iteration, simulate hour-by-hour equipment state (operating /
    failed) using exponential failure and repair distributions, then sum
    production over the year.  Returns percentile statistics.
    """
    rng = random.Random(seed)
    total_hours = horizon_days * 24
    design_production = 1.0  # normalised to 1.0

    # Pre-compute hourly failure probability
    equip_data = []
    for eq in equipment:
        lam_per_hour = eq.failure_rate_per_year / 8_760.0
        mu_per_hour = 1.0 / eq.mttr_hours if eq.mttr_hours > 0 else 1.0
        loss_frac = impact_map.get(eq.name, 0.0) / 100.0
        equip_data.append((eq.name, lam_per_hour, mu_per_hour, loss_frac))

    production_results: List[float] = []
    downtime_totals: Dict[str, float] = {eq.name: 0.0 for eq in equipment}
    total_events = 0.0

    for _ in range(iterations):
        # State: True = operating
        state = {name: True for name, *_ in equip_data}
        repair_remaining = {name: 0.0 for name, *_ in equip_data}
        cumulative = 0.0
        iter_events = 0

        for _h in range(total_hours):
            capacity = 1.0
            for name, lam, mu, loss in equip_data:
                if state[name]:
                    # Check for failure
                    if rng.random() < lam:
                        state[name] = False
                        # Sample repair time (exponential)
                        repair_remaining[name] = -math.log(max(rng.random(), 1e-12)) / mu
                        iter_events += 1
                else:
                    repair_remaining[name] -= 1.0
                    if repair_remaining[name] <= 0:
                        state[name] = True

                if not state[name]:
                    capacity *= (1.0 - loss)
                    downtime_totals[name] += 1.0 / iterations

            cumulative += design_production * capacity

        production_results.append(cumulative / total_hours * 100.0)
        total_events += iter_events

    production_results.sort()
    n = len(production_results)
    p10 = production_results[int(n * 0.10)]
    p50 = production_results[int(n * 0.50)]
    p90 = production_results[min(int(n * 0.90), n - 1)]
    mean_prod = sum(production_results) / n

    total_downtime_hrs = sum(downtime_totals.values())
    avg_events = total_events / iterations

    # Downtime contribution (fraction of total)
    contribution: Dict[str, float] = {}
    for name, hrs in downtime_totals.items():
        contribution[name] = (hrs / total_downtime_hrs * 100.0) if total_downtime_hrs > 0 else 0.0

    return MonteCarloResult(
        iterations=iterations,
        horizon_days=horizon_days,
        expected_availability_pct=mean_prod,
        expected_production_pct=mean_prod,
        p10_production_pct=p10,
        p50_production_pct=p50,
        p90_production_pct=p90,
        expected_downtime_hours_year=total_downtime_hrs,
        expected_failure_events_year=avg_events,
        equipment_downtime_contribution=contribution,
    )


# ──────────────────────────────────────────────
# Java-native risk analysis (primary path)
# ──────────────────────────────────────────────

def _try_java_risk(model: NeqSimProcessModel,
                   product_stream: Optional[str] = None,
                   feed_stream: Optional[str] = None,
                   mc_iterations: int = 1_000,
                   mc_days: int = 365,
                   ) -> Optional[RiskAnalysisResult]:
    """
    Attempt to use the Java-native ``RiskMatrix``,
    ``OperationalRiskSimulator``, and ``ProductionImpactAnalyzer``.
    """
    try:
        from neqsim import jneqsim
        RiskMatrix = jneqsim.process.safety.risk.RiskMatrix
        Simulator = jneqsim.process.safety.risk.OperationalRiskSimulator
        Analyzer = jneqsim.process.util.optimizer.ProductionImpactAnalyzer
        EqFailure = jneqsim.process.equipment.failure.EquipmentFailureMode
        RelSource = jneqsim.process.equipment.failure.ReliabilityDataSource
    except Exception:
        return None  # Java classes not available

    proc = model.get_process()
    if proc is None:
        return None

    try:
        # --- Risk matrix ---
        matrix = RiskMatrix(proc)
        matrix.buildRiskMatrix()
        matrix_json_str = str(matrix.toJson())
        import json as _json
        matrix_data = _json.loads(matrix_json_str)

        # --- Production impact ---
        analyzer = Analyzer(proc)
        if feed_stream:
            analyzer.setFeedStreamName(feed_stream)
        if product_stream:
            analyzer.setProductStreamName(product_stream)

        # --- Monte-Carlo ---
        sim = Simulator(proc)
        if feed_stream:
            sim.setFeedStreamName(feed_stream)
        if product_stream:
            sim.setProductStreamName(product_stream)
        sim.setRandomSeed(42)

        # Add reliability from OREDA source
        source = RelSource.getInstance()
        units_info = model.list_units()
        for u in units_info:
            java_class = u.unit_type
            try:
                rate = float(source.getFailureRate(java_class))
                mttr = float(source.getMTTR(java_class))
                sim.addEquipmentReliability(u.name, rate, mttr)
            except Exception:
                pass

        mc_result_java = sim.runSimulation(mc_iterations, mc_days)

        # Parse Java results into Python dataclasses
        # (Exact parsing depends on Java API — best-effort)
        reliability_list: List[EquipmentReliability] = []
        for u in units_info:
            reliability_list.append(_lookup_reliability(u.name, u.unit_type))

        impacts: List[FailureImpact] = []
        risk_items: List[RiskMatrixItem] = []

        # Try criticality ranking
        try:
            crit_map = analyzer.rankEquipmentByCriticality()
            for name_j in crit_map.keySet():
                name = str(name_j)
                ci = float(crit_map.get(name_j))
                # Simulate trip impact
                fm = EqFailure.trip(name)
                impact_result = analyzer.analyzeFailureImpact(fm)
                loss_pct = float(impact_result.getPercentLoss())
                rel = next((r for r in reliability_list if r.name == name), None)
                freq = rel.failure_rate_per_year if rel else 1.0
                prob = ProbabilityCategory.from_frequency(freq)
                cons = ConsequenceCategory.from_production_loss(loss_pct)
                score = _risk_score(prob, cons)
                impacts.append(FailureImpact(
                    equipment_name=name,
                    equipment_type="",
                    failure_type="TRIP",
                    production_loss_pct=loss_pct,
                    original_production_kg_hr=0,
                    failed_production_kg_hr=0,
                    criticality_index=ci,
                ))
                risk_items.append(RiskMatrixItem(
                    equipment_name=name,
                    probability=prob,
                    consequence=cons,
                    risk_score=score,
                    risk_level=_risk_level(score),
                    failure_rate_per_year=freq,
                    production_loss_pct=loss_pct,
                ))
        except Exception:
            pass

        avail_pct = float(mc_result_java.getAvailability())
        # If availability looks like a fraction (0–1), convert to %
        if 0 < avail_pct <= 1.0:
            avail_pct *= 100.0

        # Production percentiles — Java API may return absolute
        # production (e.g. kg/year) instead of percentages.
        p10_raw = float(mc_result_java.getP10Production()) if hasattr(mc_result_java, "getP10Production") else avail_pct
        p50_raw = float(mc_result_java.getP50Production()) if hasattr(mc_result_java, "getP50Production") else avail_pct
        p90_raw = float(mc_result_java.getP90Production()) if hasattr(mc_result_java, "getP90Production") else avail_pct

        # Normalise: if values exceed 100 they are raw production,
        # not percentages.  Use P50 as the design-capacity reference.
        if max(p10_raw, p50_raw, p90_raw) > 100:
            design_ref = p50_raw if p50_raw > 0 else max(p10_raw, p90_raw, 1.0)
            p10_pct = p10_raw / design_ref * avail_pct
            p50_pct = p50_raw / design_ref * avail_pct
            p90_pct = p90_raw / design_ref * avail_pct
        else:
            p10_pct = p10_raw
            p50_pct = p50_raw
            p90_pct = p90_raw

        dt_hrs = float(mc_result_java.getExpectedDowntimeHours()) if hasattr(mc_result_java, "getExpectedDowntimeHours") else 0
        dt_events = float(mc_result_java.getExpectedDowntimeEvents()) if hasattr(mc_result_java, "getExpectedDowntimeEvents") else 0

        mc_res = MonteCarloResult(
            iterations=mc_iterations,
            horizon_days=mc_days,
            expected_availability_pct=avail_pct,
            expected_production_pct=avail_pct,
            p10_production_pct=p10_pct,
            p50_production_pct=p50_pct,
            p90_production_pct=p90_pct,
            expected_downtime_hours_year=dt_hrs,
            expected_failure_events_year=dt_events,
        )

        sys_avail = avail_pct
        most_crit = impacts[0].equipment_name if impacts else "unknown"

        # If the Java path produced empty impacts / risk items, the
        # criticality ranking failed silently.  Fall back to the Python
        # engine which uses heuristic trip-loss estimates.
        if not impacts and not risk_items:
            return None

        return RiskAnalysisResult(
            equipment_reliability=reliability_list,
            failure_impacts=impacts,
            risk_matrix=risk_items,
            monte_carlo=mc_res,
            system_availability_pct=sys_avail,
            most_critical_equipment=most_crit,
            summary_message="Risk analysis completed using NeqSim Java risk engine.",
        )

    except Exception:
        return None


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def run_risk_analysis(
    model: NeqSimProcessModel,
    product_stream: Optional[str] = None,
    feed_stream: Optional[str] = None,
    mc_iterations: int = 1_000,
    mc_days: int = 365,
    include_degraded: bool = True,
) -> RiskAnalysisResult:
    """
    Run a full risk analysis for the process.

    Parameters
    ----------
    model : NeqSimProcessModel
        The loaded process model.
    product_stream : str, optional
        Name of the product stream (auto-detected if omitted).
    feed_stream : str, optional
        Name of the feed stream (auto-detected if omitted).
    mc_iterations : int
        Monte-Carlo iterations (default 1 000).
    mc_days : int
        Simulation horizon in days (default 365).
    include_degraded : bool
        If True, also simulate degraded failure modes.

    Returns
    -------
    RiskAnalysisResult
    """
    # --- Try Java engine first ---
    java_result = _try_java_risk(model, product_stream, feed_stream,
                                 mc_iterations, mc_days)
    if java_result is not None:
        return java_result

    # --- Python fallback ---

    # Auto-detect streams
    if product_stream is None:
        product_stream = _find_product_stream(model)
    if product_stream is None:
        raise ValueError("Could not detect a product stream. Please specify product_stream.")

    if feed_stream is None:
        streams = model.list_streams()
        if streams:
            feed_stream = streams[0].name

    # Baseline production
    baseline_flow = _get_production(model, product_stream)
    if baseline_flow <= 0:
        raise ValueError(f"Baseline production from '{product_stream}' is zero or negative.")

    # Enumerate units
    units_info = model.list_units()
    if not units_info:
        raise ValueError("No unit operations found in the model.")

    # 1. Equipment reliability
    reliability_list: List[EquipmentReliability] = []
    for u in units_info:
        java_class = u.unit_type
        rel = _lookup_reliability(u.name, java_class)
        reliability_list.append(rel)

    # 2. Failure impact (trip each unit)
    trip_impacts: List[FailureImpact] = []
    for u in units_info:
        impact = _simulate_trip(model, u.name, product_stream, baseline_flow)
        trip_impacts.append(impact)

    # Normalise criticality index
    max_loss = max((fi.production_loss_pct for fi in trip_impacts), default=1.0)
    if max_loss <= 0:
        max_loss = 1.0
    for fi in trip_impacts:
        fi.criticality_index = fi.production_loss_pct / max_loss

    # 2b. Degraded impacts (optional)
    degraded_impacts: List[FailureImpact] = []
    if include_degraded:
        for u in units_info:
            cap = _DEGRADED_CAPACITY.get(u.unit_type, 0.8)
            impact = _simulate_degraded(model, u.name, product_stream,
                                        baseline_flow, capacity_factor=cap)
            degraded_impacts.append(impact)

    all_impacts = trip_impacts + degraded_impacts

    # 3. Build risk matrix (based on trip impacts)
    risk_items: List[RiskMatrixItem] = []
    for fi in trip_impacts:
        rel = next((r for r in reliability_list if r.name == fi.equipment_name), None)
        freq = rel.failure_rate_per_year if rel else 1.0
        prob = ProbabilityCategory.from_frequency(freq)
        cons = ConsequenceCategory.from_production_loss(fi.production_loss_pct)
        score = _risk_score(prob, cons)
        risk_items.append(RiskMatrixItem(
            equipment_name=fi.equipment_name,
            probability=prob,
            consequence=cons,
            risk_score=score,
            risk_level=_risk_level(score),
            failure_rate_per_year=freq,
            production_loss_pct=fi.production_loss_pct,
        ))

    # Sort risk items by score descending
    risk_items.sort(key=lambda r: r.risk_score, reverse=True)

    # 4. Monte-Carlo availability
    impact_map = {fi.equipment_name: fi.production_loss_pct for fi in trip_impacts}
    mc = _monte_carlo_availability(
        reliability_list, impact_map,
        iterations=mc_iterations,
        horizon_days=mc_days,
    )

    # 5. System availability (analytical — series)
    sys_avail = 1.0
    for r in reliability_list:
        sys_avail *= r.availability
    sys_avail_pct = sys_avail * 100.0

    # Most critical equipment
    most_crit = max(trip_impacts, key=lambda fi: fi.criticality_index).equipment_name if trip_impacts else "unknown"

    return RiskAnalysisResult(
        equipment_reliability=reliability_list,
        failure_impacts=all_impacts,
        risk_matrix=risk_items,
        monte_carlo=mc,
        system_availability_pct=sys_avail_pct,
        most_critical_equipment=most_crit,
        summary_message="Risk analysis completed using Python fallback engine.",
    )


# ──────────────────────────────────────────────
# Formatting for LLM consumption
# ──────────────────────────────────────────────

def format_risk_result(result: RiskAnalysisResult) -> str:
    """Format a RiskAnalysisResult as structured text for the LLM."""
    lines: List[str] = []

    lines.append("═══════════════════════════════════════════════════════════")
    lines.append("              RISK ANALYSIS RESULTS")
    lines.append("═══════════════════════════════════════════════════════════")
    lines.append("")

    # System availability
    lines.append(f"System Availability (analytical): {result.system_availability_pct:.2f}%")
    lines.append(f"Most Critical Equipment: {result.most_critical_equipment}")
    lines.append("")

    # Equipment reliability table
    lines.append("─── EQUIPMENT RELIABILITY (OREDA) ────────────────────────")
    lines.append(f"{'Equipment':<25} {'Type':<18} {'MTTF (hrs)':<12} {'MTTR (hrs)':<12} {'λ (/yr)':<10} {'A (%)':<8}")
    for r in result.equipment_reliability:
        lines.append(
            f"{r.name:<25} {r.equipment_type:<18} {r.mttf_hours:<12.0f} "
            f"{r.mttr_hours:<12.0f} {r.failure_rate_per_year:<10.2f} "
            f"{r.availability*100:<8.2f}"
        )
    lines.append("")

    # Failure impact / criticality
    lines.append("─── EQUIPMENT CRITICALITY (Trip Impact) ──────────────────")
    trip_impacts = [fi for fi in result.failure_impacts if fi.failure_type == "TRIP"]
    trip_impacts.sort(key=lambda fi: fi.criticality_index, reverse=True)
    lines.append(f"{'Equipment':<25} {'Loss %':<10} {'CI':<8} {'Failed Flow (kg/hr)':<20}")
    for fi in trip_impacts:
        lines.append(
            f"{fi.equipment_name:<25} {fi.production_loss_pct:<10.1f} "
            f"{fi.criticality_index:<8.2f} {fi.failed_production_kg_hr:<20.0f}"
        )
    lines.append("")

    # Degraded impacts
    deg_impacts = [fi for fi in result.failure_impacts if fi.failure_type == "DEGRADED"]
    if deg_impacts:
        lines.append("─── DEGRADED OPERATION IMPACT ─────────────────────────────")
        lines.append(f"{'Equipment':<25} {'Loss %':<10} {'Failed Flow (kg/hr)':<20}")
        for fi in deg_impacts:
            lines.append(
                f"{fi.equipment_name:<25} {fi.production_loss_pct:<10.1f} "
                f"{fi.failed_production_kg_hr:<20.0f}"
            )
        lines.append("")

    # Risk matrix
    lines.append("─── RISK MATRIX ──────────────────────────────────────────")
    lines.append(f"{'Equipment':<25} {'Probability':<14} {'Consequence':<14} {'Score':<8} {'Level':<12}")
    for ri in result.risk_matrix:
        lines.append(
            f"{ri.equipment_name:<25} {ri.probability.value:<14} "
            f"{ri.consequence.value:<14} {ri.risk_score:<8d} {ri.risk_level.value:<12}"
        )
    lines.append("")

    # Monte Carlo
    mc = result.monte_carlo
    if mc:
        lines.append("─── MONTE-CARLO AVAILABILITY SIMULATION ──────────────────")
        lines.append(f"  Iterations:            {mc.iterations:,d}")
        lines.append(f"  Horizon:               {mc.horizon_days} days")
        lines.append(f"  Expected Availability: {mc.expected_availability_pct:.1f}%")
        lines.append(f"  Expected Production:   {mc.expected_production_pct:.1f}% of design")
        lines.append(f"  P10 Production:        {mc.p10_production_pct:.1f}%")
        lines.append(f"  P50 Production:        {mc.p50_production_pct:.1f}%")
        lines.append(f"  P90 Production:        {mc.p90_production_pct:.1f}%")
        lines.append(f"  Expected Downtime:     {mc.expected_downtime_hours_year:.0f} hrs/year")
        lines.append(f"  Expected Events:       {mc.expected_failure_events_year:.1f} failures/year")

        if mc.equipment_downtime_contribution:
            lines.append("")
            lines.append("  Equipment Downtime Contribution:")
            for name, pct in sorted(mc.equipment_downtime_contribution.items(),
                                    key=lambda x: x[1], reverse=True):
                if pct > 0.1:
                    lines.append(f"    {name:<25} {pct:.1f}%")

    lines.append("")
    lines.append(f"({result.summary_message})")
    lines.append("═══════════════════════════════════════════════════════════")

    return "\n".join(lines)
