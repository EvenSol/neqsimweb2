"""Hosted Streamlit smoke test for the NeqSim Studio shell."""

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

from studio.case_context import (
    STUDIO_RECENT_CASES_STATE_KEY,
    build_case_context,
)


def _portable_case(name: str) -> dict:
    return {
        "schema_version": 4,
        "name": name,
        "fluid": {
            "eos_model": "srk",
            "flow_unit": "kg/hr",
        },
        "process": [],
    }


class StudioShellTest(unittest.TestCase):
    """Require the new Studio entry page to render without exceptions."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.studio_path = cls.project_root / "pages" / "00_NeqSim_Studio.py"

    def test_studio_dashboard_renders(self):
        app = AppTest.from_file(str(self.studio_path)).run(timeout=30)

        if app.exception:
            details = "\n".join(str(item.value) for item in app.exception)
            self.fail(f"Studio shell raised exceptions:\n{details}")

        button_labels = [button.label for button in app.button]
        self.assertIn("＋ New process case", button_labels)
        self.assertIn("Open uploaded case", button_labels)
        self.assertIn("⚙️ Open Process Flowsheet", button_labels)
        self.assertIn("Open Classic", button_labels)
        self.assertIn("Open Process Flowsheet", button_labels)
        coming_soon_labels = [
            label for label in button_labels if label.startswith("Coming soon · ")
        ]
        self.assertEqual(
            set(coming_soon_labels),
            {
                "Coming soon · Thermodynamics & PVT",
                "Coming soon · Dynamics & Controls",
                "Coming soon · Engineering Drawings",
                "Coming soon · Examples & Tutorials",
            },
        )
        self.assertEqual(len(coming_soon_labels), len(set(coming_soon_labels)))
        uploader_labels = [uploader.label for uploader in app.file_uploader]
        self.assertIn("Open portable case JSON", uploader_labels)

    def test_recent_case_actions_include_case_names(self):
        app = AppTest.from_file(str(self.studio_path))
        app.session_state[STUDIO_RECENT_CASES_STATE_KEY] = [
            build_case_context(_portable_case("Inlet compression")),
            build_case_context(_portable_case("Export compression")),
        ]
        app.run(timeout=30)

        if app.exception:
            details = "\n".join(str(item.value) for item in app.exception)
            self.fail(f"Studio shell raised exceptions:\n{details}")

        button_labels = [button.label for button in app.button]
        recent_case_labels = [
            label for label in button_labels if label.startswith("Open recent case · ")
        ]
        self.assertEqual(
            set(recent_case_labels),
            {
                "Open recent case · Inlet compression",
                "Open recent case · Export compression",
            },
        )
        self.assertEqual(len(recent_case_labels), len(set(recent_case_labels)))
        self.assertNotIn("Open", button_labels)


if __name__ == "__main__":
    unittest.main()
