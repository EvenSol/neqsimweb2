"""Focused regressions for the shared Studio result presentation boundary."""

from dataclasses import dataclass
import sys
import types
import unittest

from studio.results_context import (
    ACTIVE_CASE_STATE_KEY,
    FLOWSHEET_CASE_HISTORY_STATE_KEY,
    FLOWSHEET_CASE_STATE_KEY,
    FLOWSHEET_RESULT_STATE_KEY,
    RESULT_DESTINATION_STATE_KEY,
    StudioResultsUnavailable,
    build_result_summary,
    case_history_rows,
    equipment_design_rows,
    load_current_result_context,
    remember_result_destination,
    selected_result_section,
    stream_rows,
)


@dataclass
class FakeKPI:
    name: str
    value: float
    unit: str


@dataclass
class FakeConstraint:
    name: str
    status: str
    detail: str


class FakeResult:
    kpis = {
        "total_power_kW": FakeKPI("total_power_kW", 1000.0, "kW"),
        "total_duty_kW": FakeKPI("total_duty_kW", 500.0, "kW"),
        "mass_balance_pct": FakeKPI("mass_balance_pct", 0.0001, "%"),
    }
    constraints = [FakeConstraint("pump_design.P-1", "WARN", "Near limit")]
    raw = {}


class FakeStream:
    name = "Feed"
    temperature_C = 20.0
    pressure_bara = 40.0
    flow_rate_kg_hr = 10_000.0
    flow_rate_mol_sec = 123.0
    process_system = "Area A"
    owner_name = "Feed"


class FakeUnit:
    name = "P-1"
    unit_type = "pump"
    process_system = "Area A"
    properties = {
        "designFlowCapacity_m3_per_hr": 120.0,
        "flowMargin_m3_per_hr": 20.0,
        "flowUtilization_pct": 83.333,
    }


class FakeModel:
    def list_streams(self):
        return [FakeStream()]

    def list_units(self):
        return [FakeUnit()]


def _fake_diagnostics_module():
    module = types.ModuleType("process_chat.solver_diagnostics")
    module.aggregate_validation_status = lambda statuses: (
        "WARN" if "WARN" in list(statuses) else "OK"
    )
    module.aggregate_convergence = lambda result: {
        "applicable": False,
        "converged": True,
    }
    module.aggregate_energy_balance = lambda result: {
        "applicable": True,
        "imbalance_pct": 0.001,
    }
    module.aggregate_unit_balances = lambda result: {
        "applicable": True,
        "max_mass_imbalance_pct": 0.0001,
    }
    module.solved_feed_flow_kg_hr = lambda result, fallback: fallback
    return module


class StudioResultsContextTest(unittest.TestCase):
    def setUp(self):
        self.spec = {
            "schema_version": 4,
            "name": "Case A",
            "fluid": {"total_flow": 10_000.0},
        }
        self.session = {
            ACTIVE_CASE_STATE_KEY: {
                "case_id": "case-a",
                "name": "Case A",
                "status": "warning",
                "case_spec": self.spec,
                "runtime": {
                    "model_available": True,
                    "solved_signature": "sha-a",
                },
            },
            FLOWSHEET_CASE_STATE_KEY: {
                "spec": self.spec,
                "model": FakeModel(),
                "result": FakeResult(),
                "signature": "sha-a",
                "warnings": ["Input basis warning"],
                "run_record": {"neqsim_version": "test"},
            },
            FLOWSHEET_RESULT_STATE_KEY: True,
        }

    def test_exact_current_result_reuses_native_model(self):
        context = load_current_result_context(self.session)

        self.assertIs(context.model, self.session[FLOWSHEET_CASE_STATE_KEY]["model"])
        self.assertEqual(context.signature, "sha-a")
        self.assertEqual(stream_rows(context)[0]["Pressure [bara]"], 40.0)
        design = equipment_design_rows(context)
        self.assertEqual(design[0]["Operating value"], 100.0)
        self.assertEqual(design[0]["Status"], "WARN")

    def test_dirty_or_stale_results_fail_closed(self):
        self.session[ACTIVE_CASE_STATE_KEY]["status"] = "dirty"
        with self.assertRaisesRegex(StudioResultsUnavailable, "dirty"):
            load_current_result_context(self.session)

        self.session[ACTIVE_CASE_STATE_KEY]["status"] = "solved"
        self.session[ACTIVE_CASE_STATE_KEY]["runtime"]["solved_signature"] = "other"
        with self.assertRaisesRegex(StudioResultsUnavailable, "does not match"):
            load_current_result_context(self.session)

    def test_summary_surfaces_warning_and_explicit_units(self):
        old_process_chat = sys.modules.get("process_chat")
        old_diagnostics = sys.modules.get("process_chat.solver_diagnostics")
        process_chat = types.ModuleType("process_chat")
        process_chat.__path__ = []
        sys.modules["process_chat"] = process_chat
        sys.modules["process_chat.solver_diagnostics"] = _fake_diagnostics_module()
        try:
            summary = build_result_summary(
                load_current_result_context(self.session)
            )
        finally:
            if old_process_chat is None:
                sys.modules.pop("process_chat", None)
            else:
                sys.modules["process_chat"] = old_process_chat
            if old_diagnostics is None:
                sys.modules.pop("process_chat.solver_diagnostics", None)
            else:
                sys.modules["process_chat.solver_diagnostics"] = old_diagnostics

        self.assertEqual(summary["engineering_state"], "Solved with warnings")
        self.assertEqual(summary["validation_status"], "WARN")
        self.assertEqual(summary["metrics"]["specific_energy"]["unit"], "kWh/t")

    def test_case_history_excludes_private_specs_and_native_objects(self):
        self.session[FLOWSHEET_CASE_HISTORY_STATE_KEY] = [
            {
                "Case": "Case A",
                "Power [kW]": 1000.0,
                "_signature": "sha-a",
                "_spec": self.spec,
                "native": FakeModel(),
            }
        ]

        self.assertEqual(
            case_history_rows(self.session),
            [{"Case": "Case A", "Power [kW]": 1000.0}],
        )

    def test_result_card_routes_to_its_advertised_workspace(self):
        remember_result_destination(self.session, "equipment")
        self.assertEqual(
            selected_result_section(self.session),
            "Equipment & design",
        )
        remember_result_destination(self.session, "studies")
        self.assertEqual(selected_result_section(self.session), "Case studies")
        self.assertEqual(
            self.session[RESULT_DESTINATION_STATE_KEY],
            "studies",
        )

        with self.assertRaisesRegex(ValueError, "Unsupported"):
            remember_result_destination(self.session, "drawings")


if __name__ == "__main__":
    unittest.main()
