"""Cross-page Streamlit interaction regressions for Classic and Studio."""

from copy import deepcopy
from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from studio.case_context import (
    STUDIO_CASE_CONTEXT_STATE_KEY,
    STUDIO_PENDING_CASE_STATE_KEY,
    STUDIO_RECENT_CASES_STATE_KEY,
    encode_portable_case,
)
from studio.results_context import RESULT_DESTINATION_STATE_KEY


CLASSIC_MARKER_KEY = "classic_workspace_regression_marker"
CLASSIC_MARKER_VALUE = {"case": "must survive Studio navigation"}


class StudioWorkspaceInteractionTest(unittest.TestCase):
    """Exercise the user-facing workspace path through the real app entrypoint."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.entrypoint = cls.project_root / "welcome.py"

    def _new_app(self) -> AppTest:
        app = AppTest.from_file(str(self.entrypoint), default_timeout=120)
        app.session_state[CLASSIC_MARKER_KEY] = CLASSIC_MARKER_VALUE
        return self._run(app)

    def _run(
        self,
        app: AppTest,
        *,
        uploaded_case: bytes | None = None,
    ) -> AppTest:
        if uploaded_case is None:
            app.run(timeout=120)
        else:
            with patch(
                "streamlit.file_uploader",
                return_value=BytesIO(uploaded_case),
            ):
                app.run(timeout=120)
        if app.exception:
            details = "\n".join(str(item.value) for item in app.exception)
            self.fail(f"Workspace interaction raised exceptions:\n{details}")
        return app

    def _click(
        self,
        app: AppTest,
        label: str,
        *,
        target_page: str | None = None,
        uploaded_case: bytes | None = None,
    ) -> AppTest:
        matches = [button for button in app.button if button.label == label]
        self.assertEqual(
            len(matches),
            1,
            f"Expected one {label!r} action, found {len(matches)}.",
        )
        matches[0].click()
        app = self._run(app, uploaded_case=uploaded_case)
        if target_page is not None:
            # AppTest follows the first switch-page request automatically, but an
            # explicit page selection is required before interacting with a second
            # page in the same multipage test session.
            app.switch_page(target_page)
            app = self._run(app)
        return app

    def _assert_classic_marker(self, app: AppTest) -> None:
        self.assertEqual(app.session_state[CLASSIC_MARKER_KEY], CLASSIC_MARKER_VALUE)

    def test_classic_entry_new_continue_and_recent_case_preserve_identity(self):
        app = self._new_app()
        self.assertIn("Open NeqSim Studio", [button.label for button in app.button])

        app = self._click(
            app,
            "Open NeqSim Studio",
            target_page="pages/00_NeqSim_Studio.py",
        )
        self.assertIn("Case workspace", [item.value for item in app.subheader])
        self._assert_classic_marker(app)

        app = self._click(
            app,
            "＋ New process case",
            target_page="pages/35_Process_Flowsheet_Studio.py",
        )
        self.assertIn("🏭 Process Flowsheet Studio", [item.value for item in app.title])
        self.assertNotIn(STUDIO_PENDING_CASE_STATE_KEY, app.session_state)
        first_context = app.session_state[STUDIO_CASE_CONTEXT_STATE_KEY]
        case_id = first_context["case_id"]
        self.assertEqual(first_context["case_schema_version"], 4)
        self._assert_classic_marker(app)

        app = self._click(
            app,
            "← Studio home",
            target_page="pages/00_NeqSim_Studio.py",
        )
        self.assertIn("Continue active case", [button.label for button in app.button])
        app = self._click(
            app,
            "Continue active case",
            target_page="pages/35_Process_Flowsheet_Studio.py",
        )
        self.assertEqual(
            app.session_state[STUDIO_CASE_CONTEXT_STATE_KEY]["case_id"],
            case_id,
        )
        self._assert_classic_marker(app)

        app = self._click(
            app,
            "← Studio home",
            target_page="pages/00_NeqSim_Studio.py",
        )
        recent_label = f"Open recent case · {first_context['name']}"
        self.assertIn(recent_label, [button.label for button in app.button])
        app = self._click(
            app,
            recent_label,
            target_page="pages/35_Process_Flowsheet_Studio.py",
        )
        self.assertEqual(
            app.session_state[STUDIO_CASE_CONTEXT_STATE_KEY]["case_id"],
            case_id,
        )
        self._assert_classic_marker(app)

    def test_result_destinations_fail_closed_and_return_to_active_flowsheet(self):
        for label, destination in (
            ("Open Equipment Design", "equipment"),
            ("Open Engineering Studies", "studies"),
        ):
            with self.subTest(destination=destination):
                app = self._new_app()
                app = self._click(
                    app,
                    "Open NeqSim Studio",
                    target_page="pages/00_NeqSim_Studio.py",
                )
                app = self._click(
                    app,
                    "＋ New process case",
                    target_page="pages/35_Process_Flowsheet_Studio.py",
                )
                case_id = app.session_state[STUDIO_CASE_CONTEXT_STATE_KEY]["case_id"]
                app = self._click(
                    app,
                    "← Studio home",
                    target_page="pages/00_NeqSim_Studio.py",
                )

                app = self._click(
                    app,
                    label,
                    target_page="pages/10_Studio_Results.py",
                )
                self.assertIn("Engineering Results", [item.value for item in app.title])
                self.assertEqual(
                    app.session_state[RESULT_DESTINATION_STATE_KEY],
                    destination,
                )
                self.assertIn(
                    "Open Process Flowsheet",
                    [button.label for button in app.button],
                )
                self._assert_classic_marker(app)

                app = self._click(
                    app,
                    "Open Process Flowsheet",
                    target_page="pages/35_Process_Flowsheet_Studio.py",
                )
                self.assertIn(
                    "🏭 Process Flowsheet Studio",
                    [item.value for item in app.title],
                )
                self.assertEqual(
                    app.session_state[STUDIO_CASE_CONTEXT_STATE_KEY]["case_id"],
                    case_id,
                )
                self._assert_classic_marker(app)

    def _active_case_on_studio_home(self) -> tuple[AppTest, dict]:
        app = self._new_app()
        app = self._click(
            app,
            "Open NeqSim Studio",
            target_page="pages/00_NeqSim_Studio.py",
        )
        app = self._click(
            app,
            "＋ New process case",
            target_page="pages/35_Process_Flowsheet_Studio.py",
        )
        baseline = deepcopy(
            app.session_state[STUDIO_CASE_CONTEXT_STATE_KEY]
        )
        app = self._click(
            app,
            "← Studio home",
            target_page="pages/00_NeqSim_Studio.py",
        )
        return app, baseline

    def test_uploaded_supported_case_migrates_with_new_identity(self):
        app, baseline = self._active_case_on_studio_home()
        candidate = deepcopy(baseline["case_spec"])
        candidate["schema_version"] = 1
        candidate["name"] = "Uploaded schema v1 case"
        payload = encode_portable_case(candidate)

        app = self._run(app, uploaded_case=payload)
        app = self._click(
            app,
            "Open uploaded case",
            target_page="pages/35_Process_Flowsheet_Studio.py",
            uploaded_case=payload,
        )

        opened = app.session_state[STUDIO_CASE_CONTEXT_STATE_KEY]
        self.assertNotEqual(opened["case_id"], baseline["case_id"])
        self.assertEqual(opened["name"], candidate["name"])
        self.assertEqual(opened["case_schema_version"], 4)
        self.assertEqual(opened["case_spec"]["schema_version"], 4)
        self.assertNotIn(STUDIO_PENDING_CASE_STATE_KEY, app.session_state)
        self.assertTrue(
            any(
                "Schema-v1 case migrated to Studio schema v4."
                in item.value
                for item in app.success
            )
        )
        self._assert_classic_marker(app)

    def test_uploaded_rejected_cases_preserve_active_recent_and_classic_state(self):
        for label, payload, target_page, expected_error in (
            (
                "invalid JSON",
                b"{not valid JSON",
                None,
                "The Studio case is not valid JSON",
            ),
            (
                "future schema",
                None,
                "pages/35_Process_Flowsheet_Studio.py",
                "Unsupported schema_version. Expected version 1, 2, 3, or 4.",
            ),
        ):
            with self.subTest(payload=label):
                app, baseline = self._active_case_on_studio_home()
                recent_before = deepcopy(
                    app.session_state[STUDIO_RECENT_CASES_STATE_KEY]
                )
                if payload is None:
                    future_case = deepcopy(baseline["case_spec"])
                    future_case["schema_version"] = 5
                    future_case["name"] = "Rejected future upload"
                    payload = encode_portable_case(future_case)

                app = self._run(app, uploaded_case=payload)
                app = self._click(
                    app,
                    "Open uploaded case",
                    target_page=target_page,
                    uploaded_case=payload,
                )

                active = app.session_state[STUDIO_CASE_CONTEXT_STATE_KEY]
                self.assertEqual(active["case_id"], baseline["case_id"])
                self.assertEqual(
                    active["case_fingerprint"],
                    baseline["case_fingerprint"],
                )
                self.assertEqual(active["case_spec"], baseline["case_spec"])
                self.assertEqual(
                    app.session_state[STUDIO_RECENT_CASES_STATE_KEY],
                    recent_before,
                )
                self.assertNotIn(
                    STUDIO_PENDING_CASE_STATE_KEY,
                    app.session_state,
                )
                self.assertTrue(
                    any(expected_error in item.value for item in app.error),
                    [item.value for item in app.error],
                )
                self._assert_classic_marker(app)

if __name__ == "__main__":
    unittest.main()
