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


class ChatSession:
    def __init__(self, model, sensitivity=None, optimization=None):
        self.model = model
        self._sensitivity = sensitivity
        self._optimization = optimization

    def get_last_sensitivity(self):
        return self._sensitivity

    def get_last_optimization(self):
        return self._optimization


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
            },
            {
                "role": "assistant",
                "optimization": Optimization(),
                "_study_model": self.model,
            },
            {"role": "assistant", "content": "Ordinary follow-up answer"},
        ]

        evidence = current_study_evidence(self.session)

        self.assertTrue(evidence["available"])
        self.assertEqual(evidence["sensitivity"]["n_points"], 2)
        self.assertTrue(evidence["optimization"]["converged"])

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
        self.assertIn("No sensitivity", evidence["reason"])

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
