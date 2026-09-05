"""Regression tests for stable and experimental Streamlit navigation."""

from __future__ import annotations

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

from app_navigation import (
    STABLE_PAGE_PATHS,
    discover_page_paths,
    experimental_page_specs,
    stable_page_specs,
)


STUDIO_PATH = "pages/00_NeqSim_Studio.py"


class AppNavigationPolicyTest(unittest.TestCase):
    """Verify the stable boundary and complete experimental inventory."""

    def test_normal_mode_contains_only_requested_stable_tools(self):
        self.assertEqual(
            tuple(spec.path for spec in stable_page_specs()),
            STABLE_PAGE_PATHS,
        )
        self.assertEqual(
            tuple(spec.title for spec in stable_page_specs()),
            (
                "TP Flash",
                "Phase Envelope",
                "Gas Hydrate",
                "Hydrogen",
                "EOS-CG",
            ),
        )

    def test_neqsim_studio_is_experimental_only(self):
        stable_paths = {spec.path for spec in stable_page_specs()}
        experimental_specs = experimental_page_specs()

        self.assertNotIn(STUDIO_PATH, stable_paths)
        self.assertIn(STUDIO_PATH, {spec.path for spec in experimental_specs})
        self.assertEqual(
            next(spec.title for spec in experimental_specs if spec.path == STUDIO_PATH),
            "NeqSim Studio",
        )

    def test_experimental_mode_preserves_every_discovered_page(self):
        registered_paths = {
            *(spec.path for spec in stable_page_specs()),
            *(spec.path for spec in experimental_page_specs()),
        }
        self.assertEqual(registered_paths, set(discover_page_paths()))

class AppNavigationInteractionTest(unittest.TestCase):
    """Exercise the mode control through the deployed app entry point."""

    def test_studio_front_page_action_tracks_experimental_mode(self):
        app_path = Path(__file__).resolve().parents[1] / "welcome.py"
        app = AppTest.from_file(str(app_path)).run(timeout=30)

        self.assertFalse(app.exception)
        self.assertNotIn("Open NeqSim Studio", [button.label for button in app.button])
        experimental_toggle = next(
            toggle
            for toggle in app.sidebar.toggle
            if toggle.label == "Experimental mode"
        )
        self.assertFalse(experimental_toggle.value)

        experimental_toggle.set_value(True)
        app.run(timeout=30)

        self.assertFalse(app.exception)
        self.assertIn("Open NeqSim Studio", [button.label for button in app.button])
        experimental_toggle = next(
            toggle
            for toggle in app.sidebar.toggle
            if toggle.label == "Experimental mode"
        )
        self.assertTrue(experimental_toggle.value)

        experimental_toggle.set_value(False)
        app.run(timeout=30)

        self.assertFalse(app.exception)
        self.assertNotIn("Open NeqSim Studio", [button.label for button in app.button])

    def test_studio_url_is_unavailable_in_normal_mode(self):
        app_path = Path(__file__).resolve().parents[1] / "welcome.py"
        app = AppTest.from_file(str(app_path)).run(timeout=30)

        try:
            app.switch_page(STUDIO_PATH).run(timeout=30)
        except ValueError as unavailable_page:
            self.assertIn("Could not find a navigation page", str(unavailable_page))
            return

        self.assertFalse(app.exception)
        self.assertNotIn("Case workspace", [item.value for item in app.subheader])
        self.assertNotIn("Open NeqSim Studio", [button.label for button in app.button])


if __name__ == "__main__":
    unittest.main()
