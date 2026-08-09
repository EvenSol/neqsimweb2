"""Regression tests for the new Studio shell and Classic-preservation contract."""

from pathlib import Path
import unittest

from studio.navigation import (
    STATUS_AVAILABLE,
    VALID_STATUSES,
    STUDIO_DESTINATIONS,
    destination_by_key,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StudioNavigationTest(unittest.TestCase):
    """Protect the boundary between the existing Classic app and new Studio shell."""

    def test_destination_keys_and_routes_are_stable_and_unique(self):
        keys = [destination.key for destination in STUDIO_DESTINATIONS]

        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(
            all(
                destination.status in VALID_STATUSES
                for destination in STUDIO_DESTINATIONS
            )
        )

        for destination in STUDIO_DESTINATIONS:
            if destination.status == STATUS_AVAILABLE:
                self.assertTrue(destination.page)
                self.assertTrue(destination.page.startswith("pages/"))

    def test_existing_flowsheet_studio_is_first_available_workflow(self):
        destination = destination_by_key("flowsheet")

        self.assertTrue(destination.available)
        self.assertEqual(
            destination.page,
            "pages/35_Process_Flowsheet_Studio.py",
        )

    def test_unknown_destination_fails_loudly(self):
        with self.assertRaisesRegex(KeyError, "Unknown Studio destination"):
            destination_by_key("not-a-real-workflow")

    def test_classic_home_and_studio_entry_remain_separate(self):
        welcome_source = (PROJECT_ROOT / "welcome.py").read_text(encoding="utf-8")
        studio_source = (
            PROJECT_ROOT / "pages" / "00_NeqSim_Studio.py"
        ).read_text(encoding="utf-8")

        # Classic information and its existing sidebar-driven workflow remain.
        self.assertIn("### About NeqSim", welcome_source)
        self.assertIn("### Getting Started", welcome_source)
        self.assertIn("Enable AI Features", welcome_source)

        # Studio is added without moving the mature flowsheet page or Classic home.
        self.assertIn(
            'st.switch_page("pages/00_NeqSim_Studio.py")',
            welcome_source,
        )
        self.assertIn('st.switch_page("welcome.py")', studio_source)
        self.assertIn(
            'st.switch_page("pages/35_Process_Flowsheet_Studio.py")',
            studio_source,
        )


if __name__ == "__main__":
    unittest.main()
