"""Hosted Streamlit smoke test for the NeqSim Studio shell."""

from html import escape
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

from studio.case_context import (
    STUDIO_CASE_CONTEXT_STATE_KEY,
    STUDIO_RECENT_CASES_STATE_KEY,
    build_case_context,
)
from studio.navigation import STUDIO_DESTINATIONS


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

    def test_browser_facing_structure_is_semantic_and_responsive(self):
        app = AppTest.from_file(str(self.studio_path)).run(timeout=30)

        if app.exception:
            details = "\n".join(str(item.value) for item in app.exception)
            self.fail(f"Studio shell raised exceptions:\n{details}")

        markup = "\n".join(str(item.value) for item in app.markdown)
        self.assertIn(
            '<section class="studio-hero" '
            'aria-labelledby="studio-page-title">',
            markup,
        )
        self.assertIn(
            '<h1 class="studio-title" id="studio-page-title">',
            markup,
        )
        self.assertIn('aria-label="Workspace status"', markup)
        self.assertIn("@media (max-width: 720px)", markup)
        self.assertIn("padding-left: 1rem", markup)
        self.assertIn("padding-right: 1rem", markup)

        for destination in STUDIO_DESTINATIONS:
            title_id = f"workflow-title-{destination.key}"
            self.assertIn(
                f'<article class="workflow-card" '
                f'aria-labelledby="{title_id}">',
                markup,
            )
            self.assertIn(
                f'<h3 class="workflow-title" id="{title_id}">',
                markup,
            )
            self.assertIn(escape(destination.title), markup)

    def test_case_lifecycle_controls_keep_unique_visible_names(self):
        app = AppTest.from_file(str(self.studio_path))
        app.session_state[STUDIO_CASE_CONTEXT_STATE_KEY] = (
            build_case_context(_portable_case("Accessible compression case"))
        )
        app.run(timeout=30)

        if app.exception:
            details = "\n".join(str(item.value) for item in app.exception)
            self.fail(f"Studio shell raised exceptions:\n{details}")

        button_labels = [str(button.label).strip() for button in app.button]
        self.assertTrue(all(button_labels))
        self.assertEqual(len(button_labels), len(set(button_labels)))

        for widget_name in ("checkbox", "file_uploader", "text_input"):
            labels = [
                str(widget.label).strip()
                for widget in app.get(widget_name)
            ]
            self.assertTrue(labels)
            self.assertTrue(all(labels))

        download_labels = [
            str(button.label).strip()
            for button in app.get("download_button")
        ]
        self.assertEqual(download_labels, ["Download active case"])

        planned_actions = [
            button
            for button in app.button
            if button.label.startswith("Coming soon · ")
        ]
        self.assertTrue(planned_actions)
        self.assertTrue(all(button.disabled for button in planned_actions))

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
