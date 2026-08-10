"""Regression tests for exact-model Studio engineering-study evidence."""

from dataclasses import dataclass, field
import unittest

from studio.results_context import (
    ACTIVE_CASE_STATE_KEY,
    FLOWSHEET_CASE_STATE_KEY,
    FLOWSHEET_RESULT_STATE_KEY,
)
from studio.study_context import current_study_evidence, engineering_unit


@dataclass
class SweepPoint:
    input_values: dict
    output_values: dict
    feasible: bool = True
    error: str = ""


@dataclass
class Sensitivity:
    analysis_type: str = "single_sweep"
    sweep_variable: str = "streams.feed.flow_kg_hr"
    response_kpis: list = field(default_factory=lambda: ["total_power_kW"])
    sweep_points: list = field(default_factory=list)
    tornado_bars: list = field(default_factory=list)
    n_points: int = 2
    method: str = "clone_and_run"
    message: str = "Completed"


@dataclass
class Utilization:
    name: str = "K-1"
    equipment_type: str = "compressor"
    utilization: float = 0.92
    constraint_name: str = "maximum power"
    detail: str = "Within limit"


@dataclass
class Optimization:
    optimal_flow_kg_hr: float = 120_000.0
    original_flow_kg_hr: float = 100_000.0
    max_increase_pct: float = 20.0
    bottleneck_equipment: str = "K-1"
    bottleneck_type: str = "compressor"
    bottleneck_utilization: float = 0.92
    utilization_breakdown: list = field(default_factory=lambda: [Utilization()])
    iterations: list = field(default_factory=list)
    kpis_at_optimum: dict = field(default_factory=dict)
    search_algorithm: str = "golden_section"
    converged: bool = True
    message: str = "Converged"


@dataclass
class KPI:
    name: str
    value: float
    unit: str


@dataclass
class RunResult:
    kpis: dict = field(default_factory=dict)


@dataclass
class Scenario:
    name: str


@dataclass
class ScenarioResult:
    scenario: Scenario
    result: RunResult
    success: bool = True
    error: str = ""


@dataclass
class Comparison:
    base: ScenarioResult
    cases: list = field(default_factory=list)
    delta_kpis: list = field(default_factory=list)
    constraint_summary: list = field(default_factory=list)
    patch_log: list = field(default_factory=list)


@dataclass
class EmissionSource:
    name: str = "K-1 driver"
    source_type: str = "FUEL_GAS"
    co2_kg_hr: float = 250.0
    co2e_kg_hr: float = 250.0
    ch4_kg_hr: float = 0.0
    nox_kg_hr: float = 0.1
    fuel_rate_kg_hr: float = 91.0
    detail: str = "Gas-turbine screening"


@dataclass
class Emissions:
    sources: list = field(default_factory=lambda: [EmissionSource()])
    total_co2_kg_hr: float = 250.0
    total_co2e_kg_hr: float = 250.0
    total_co2_tonnes_yr: float = 2190.0
    total_co2e_tonnes_yr: float = 2190.0
    total_ch4_kg_hr: float = 0.0
    total_nox_kg_hr: float = 0.1
    emission_intensity_kg_per_tonne: float = 2.5
    product_rate_kg_hr: float = 100_000.0
    method: str = "estimation"
    message: str = "Screening estimate"


@dataclass
class EnergyConsumer:
    name: str = "K-1"
    equipment_type: str = "Compressor"
    energy_type: str = "POWER"
    consumption_kW: float = 4500.0
    share_pct: float = 100.0
    detail: str = "Shaft power"


@dataclass
class Benchmark:
    metric: str = "specific energy"
    actual_value: float = 45.0
    benchmark_value: float = 60.0
    unit: str = "kWh/tonne"
    status: str = "GOOD"
    detail: str = "Screening benchmark"


@dataclass
class Suggestion:
    equipment: str = "K-1"
    suggestion: str = "Review driver efficiency"
    potential_saving_kW: float = 200.0
    potential_saving_pct: float = 4.4
    detail: str = "Screening opportunity"


@dataclass
class EnergyAudit:
    total_power_kW: float = 4500.0
    total_cooling_kW: float = 1000.0
    total_heating_kW: float = 0.0
    net_energy_kW: float = 5500.0
    specific_energy_kWh_per_tonne: float = 45.0
    product_rate_kg_hr: float = 100_000.0
    product_stream: str = "export"
    consumers: list = field(default_factory=lambda: [EnergyConsumer()])
    benchmarks: list = field(default_factory=lambda: [Benchmark()])
    suggestions: list = field(default_factory=lambda: [Suggestion()])
    fuel_gas_rate_kg_hr: float = 91.0
    fuel_gas_cost_usd_hr: float = 13.65
    method: str = "process_analysis"
    message: str = "Completed"


class ChatSession:
    def __init__(
        self,
        model,
        sensitivity=None,
        optimization=None,
        comparison=None,
        emissions=None,
        energy_audit=None,
    ):
        self.model = model
        self._sensitivity = sensitivity
        self._optimization = optimization
        self._comparison = comparison
        self._emissions = emissions
        self._energy_audit = energy_audit

    def get_last_sensitivity(self):
        return self._sensitivity

    def get_last_optimization(self):
        return self._optimization

    def get_last_comparison(self):
        return self._comparison

    def get_last_emissions(self):
        return self._emissions

    def get_last_energy_audit(self):
        return self._energy_audit


class StudioStudyContextTest(unittest.TestCase):
    def setUp(self):
        self.model = object()
        self.spec = {"schema_version": 4, "name": "Case A", "fluid": {}}
        self.session = {
            ACTIVE_CASE_STATE_KEY: {
                "case_id": "case-a",
                "status": "solved",
                "case_spec": self.spec,
                "runtime": {
                    "model_available": True,
                    "model_name": "case-a.neqsim",
                    "solved_signature": "sha-a",
                },
            },
            FLOWSHEET_CASE_STATE_KEY: {
                "spec": self.spec,
                "model": self.model,
                "result": object(),
                "signature": "sha-a",
            },
            FLOWSHEET_RESULT_STATE_KEY: True,
        }

    def test_same_model_studies_include_signature_units_and_failures(self):
        sensitivity = Sensitivity(
            sweep_points=[
                SweepPoint(
                    {"streams.feed.flow_kg_hr": 90_000.0},
                    {"total_power_kW": 4_500.0},
                ),
                SweepPoint(
                    {"streams.feed.flow_kg_hr": 130_000.0},
                    {},
                    feasible=False,
                    error="Compressor limit",
                ),
            ]
        )
        self.session["chat_session"] = ChatSession(
            self.model,
            sensitivity=sensitivity,
            optimization=Optimization(),
        )

        evidence = current_study_evidence(self.session)

        self.assertTrue(evidence["available"])
        self.assertEqual(evidence["provenance"]["solved_signature"], "sha-a")
        self.assertEqual(evidence["sensitivity"]["point_rows"][0]["Unit"], "kW")
        self.assertFalse(evidence["sensitivity"]["point_rows"][1]["Feasible"])
        self.assertEqual(
            evidence["optimization"]["bottleneck_utilization_pct"],
            92.0,
        )

    def test_different_runtime_model_fails_closed(self):
        self.session["chat_session"] = ChatSession(object(), Sensitivity())

        evidence = current_study_evidence(self.session)

        self.assertFalse(evidence["available"])
        self.assertIn("different runtime model", evidence["reason"])
        self.assertNotIn("sensitivity", evidence)

    def test_retained_message_studies_survive_later_chat_turns(self):
        sensitivity = Sensitivity(
            sweep_points=[
                SweepPoint(
                    {"streams.feed.flow_kg_hr": 90_000.0},
                    {"total_power_kW": 4_500.0},
                )
            ]
        )
        self.session["chat_session"] = ChatSession(self.model)
        self.session["chat_messages"] = [
            {
                "role": "assistant",
                "sensitivity": sensitivity,
                "_study_model": self.model,
                "_study_signature": "sha-a",
            },
            {
                "role": "assistant",
                "optimization": Optimization(),
                "_study_model": self.model,
                "_study_signature": "sha-a",
            },
            {
                "role": "assistant",
                "emissions": Emissions(),
                "energy_audit": EnergyAudit(),
                "_study_model": self.model,
                "_study_signature": "sha-a",
            },
            {"role": "assistant", "content": "Ordinary follow-up answer"},
        ]

        evidence = current_study_evidence(self.session)

        self.assertTrue(evidence["available"])
        self.assertEqual(evidence["sensitivity"]["n_points"], 2)
        self.assertTrue(evidence["optimization"]["converged"])
        self.assertEqual(evidence["emissions"]["method"], "estimation")
        self.assertEqual(
            evidence["energy_audit"]["metric_rows"][0]["Unit"],
            "kW",
        )

    def test_scenario_emissions_and_energy_evidence_preserve_failures_and_units(self):
        comparison = Comparison(
            base=ScenarioResult(
                Scenario("BASE"),
                RunResult({"total_power_kW": KPI("total_power_kW", 4000.0, "kW")}),
            ),
            cases=[
                ScenarioResult(
                    Scenario("High rate"),
                    RunResult(),
                    success=False,
                    error="Compressor limit",
                )
            ],
            delta_kpis=[
                {
                    "scenario": "High rate",
                    "kpi": "total_power_kW",
                    "base": 4000.0,
                    "case": 5000.0,
                    "delta": 1000.0,
                    "delta_pct": 25.0,
                    "unit": "kW",
                }
            ],
            constraint_summary=[
                {
                    "scenario": "High rate",
                    "constraint": "compressor power",
                    "status": "VIOLATION",
                    "detail": "Above design basis",
                }
            ],
        )
        self.session["chat_session"] = ChatSession(
            self.model,
            comparison=comparison,
            emissions=Emissions(),
            energy_audit=EnergyAudit(),
        )

        evidence = current_study_evidence(self.session)

        self.assertEqual(
            evidence["scenario_comparison"]["case_rows"][1]["Status"],
            "Failed",
        )
        self.assertEqual(
            evidence["scenario_comparison"]["case_rows"][1]["Error"],
            "Compressor limit",
        )
        self.assertEqual(
            evidence["scenario_comparison"]["kpi_rows"][0]["Unit"],
            "kW",
        )
        self.assertEqual(
            evidence["emissions"]["metric_rows"][0],
            {"Metric": "CO2", "Value": 250.0, "Unit": "kg/hr"},
        )
        self.assertEqual(
            evidence["energy_audit"]["benchmark_rows"][0]["Status"],
            "GOOD",
        )

    def test_retained_message_from_another_model_is_ignored(self):
        self.session["chat_session"] = ChatSession(self.model)
        self.session["chat_messages"] = [
            {
                "role": "assistant",
                "sensitivity": Sensitivity(),
                "_study_model": object(),
            }
        ]

        evidence = current_study_evidence(self.session)

        self.assertTrue(evidence["available"])
        self.assertIsNone(evidence["sensitivity"])
        self.assertIn("No Process Chat engineering evidence", evidence["reason"])

    def test_retained_message_from_another_solved_signature_is_ignored(self):
        self.session["chat_session"] = ChatSession(self.model)
        self.session["chat_messages"] = [
            {
                "role": "assistant",
                "emissions": Emissions(),
                "_study_model": self.model,
                "_study_signature": "sha-before-resolve",
            }
        ]

        evidence = current_study_evidence(self.session)

        self.assertTrue(evidence["available"])
        self.assertIsNone(evidence["emissions"])
        self.assertIn("No Process Chat engineering evidence", evidence["reason"])

    def test_dirty_case_does_not_publish_study_evidence(self):
        self.session[ACTIVE_CASE_STATE_KEY]["status"] = "dirty"
        self.session["chat_session"] = ChatSession(self.model, Sensitivity())

        evidence = current_study_evidence(self.session)

        self.assertFalse(evidence["available"])
        self.assertIn("dirty", evidence["reason"])

    def test_units_are_explicit_or_declared_unavailable(self):
        self.assertEqual(engineering_unit("total_power_kW"), "kW")
        self.assertEqual(engineering_unit("feed_pressure_bara"), "bara")
        self.assertEqual(engineering_unit("unknown_metric"), "not reported")


if __name__ == "__main__":
    unittest.main()
