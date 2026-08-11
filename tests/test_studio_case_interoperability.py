"""End-to-end migration gates for Classic-compatible Studio case files."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

from studio.case_context import (
    PENDING_OPEN,
    STUDIO_CASE_CONTEXT_STATE_KEY,
    STUDIO_PENDING_CASE_STATE_KEY,
    decode_portable_case,
    encode_portable_case,
    recent_cases,
)


class StudioCaseInteroperabilityTest(unittest.TestCase):
    """Exercise the real flowsheet importer through the shared Studio handoff."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.flowsheet_path = (
            cls.project_root / "pages" / "35_Process_Flowsheet_Studio.py"
        )

    def _run(self, app: AppTest) -> AppTest:
        app.run(timeout=120)
        if app.exception:
            details = "\n".join(str(item.value) for item in app.exception)
            self.fail(f"Studio flowsheet raised exceptions:\n{details}")
        return app

    def _baseline_app(self) -> tuple[AppTest, dict]:
        app = AppTest.from_file(str(self.flowsheet_path))
        app.session_state["classic_case"] = {"keep": True}
        self._run(app)
        context = deepcopy(app.session_state[STUDIO_CASE_CONTEXT_STATE_KEY])
        self.assertEqual(context["case_schema_version"], 4)
        return app, context

    def _queue_open(self, app: AppTest, case_spec: dict) -> None:
        app.session_state[STUDIO_PENDING_CASE_STATE_KEY] = {
            "action": PENDING_OPEN,
            "case_spec": deepcopy(case_spec),
            "preserve_identity": False,
        }

    def test_schema_v1_through_v4_open_and_round_trip_as_v4(self):
        app, baseline = self._baseline_app()
        previous_case_id = baseline["case_id"]

        for schema_version in (1, 2, 3, 4):
            candidate = deepcopy(baseline["case_spec"])
            candidate["schema_version"] = schema_version
            candidate["name"] = f"Interoperability schema v{schema_version}"
            original_candidate = deepcopy(candidate)

            self._queue_open(app, candidate)
            self._run(app)

            context = deepcopy(app.session_state[STUDIO_CASE_CONTEXT_STATE_KEY])
            self.assertNotEqual(context["case_id"], previous_case_id)
            self.assertEqual(context["name"], candidate["name"])
            self.assertEqual(context["case_schema_version"], 4)
            self.assertEqual(context["case_spec"]["schema_version"], 4)
            self.assertEqual(
                decode_portable_case(encode_portable_case(context["case_spec"])),
                context["case_spec"],
            )
            self.assertEqual(candidate, original_candidate)
            self.assertEqual(app.session_state["classic_case"], {"keep": True})

            migration_notice = (
                f"Schema-v{schema_version} case migrated to Studio schema v4."
            )
            success_messages = [item.value for item in app.success]
            if schema_version < 4:
                self.assertTrue(
                    any(migration_notice in message for message in success_messages),
                    success_messages,
                )
            else:
                self.assertFalse(
                    any(
                        "case migrated to Studio schema" in message
                        for message in success_messages
                    ),
                    success_messages,
                )
            previous_case_id = context["case_id"]

    def test_future_schema_fails_closed_without_replacing_active_or_classic_case(self):
        app, baseline = self._baseline_app()
        future_case = deepcopy(baseline["case_spec"])
        future_case["schema_version"] = 5
        future_case["name"] = "Unsupported future case"

        self._queue_open(app, future_case)
        self._run(app)

        context = deepcopy(app.session_state[STUDIO_CASE_CONTEXT_STATE_KEY])
        self.assertEqual(context["case_id"], baseline["case_id"])
        self.assertEqual(context["case_fingerprint"], baseline["case_fingerprint"])
        self.assertEqual(context["case_spec"], baseline["case_spec"])
        self.assertEqual(app.session_state["classic_case"], {"keep": True})
        self.assertNotIn(
            future_case["name"],
            [case["name"] for case in recent_cases(app.session_state)],
        )
        error_messages = [item.value for item in app.error]
        self.assertTrue(
            any(
                "Unsupported schema_version. Expected version 1, 2, 3, or 4."
                in message
                for message in error_messages
            ),
            error_messages,
        )


if __name__ == "__main__":
    unittest.main()
