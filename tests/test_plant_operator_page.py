"""Source-level regression tests for the Plant Operator Streamlit page."""

import ast
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from studio.navigation import destination_by_key


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAGE_PATH = PROJECT_ROOT / "pages" / "36_NeqSim_Plant_Operator.py"


class PlantOperatorPageTest(unittest.TestCase):
    """Keep the game discoverable, bounded, accessible, and interoperable."""

    def test_page_is_valid_python_and_navigation_target(self):
        source = PAGE_PATH.read_text(encoding="utf-8")
        ast.parse(source)

        destination = destination_by_key("plant_operator")
        self.assertTrue(destination.available)
        self.assertEqual(destination.page, "pages/36_NeqSim_Plant_Operator.py")

    def test_page_exposes_primary_game_flow_and_native_evidence(self):
        source = PAGE_PATH.read_text(encoding="utf-8")

        self.assertIn("▶ Run operating strategy", source)
        self.assertIn("run_challenge(", source)
        self.assertIn("Native validation", source)
        self.assertIn("Download reproducible challenge JSON", source)
        self.assertIn("Analyze solved attempt in Process Chat", source)
        self.assertIn('role="img"', source)
        self.assertIn('aria-label="Mission constraints"', source)
        self.assertIn("@media (max-width: 600px)", source)

    def test_initial_control_room_renders_without_starting_native_solve(self):
        app = AppTest.from_file(str(PAGE_PATH)).run(timeout=30)

        if app.exception:
            details = "\n".join(str(item.value) for item in app.exception)
            self.fail(f"Plant Operator page raised exceptions:\n{details}")

        self.assertIn(
            "▶ Run operating strategy",
            [button.label for button in app.button],
        )
        self.assertEqual(
            [slider.label for slider in app.slider],
            [
                "Feed throughput [kg/hr]",
                "Stage 1 discharge pressure [bara]",
                "Stage 2 discharge pressure [bara]",
                "Intercooler outlet temperature [°C]",
                "Export cooler outlet temperature [°C]",
            ],
        )

    def test_page_preserves_shared_timeout_and_stale_result_guards(self):
        source = PAGE_PATH.read_text(encoding="utf-8")

        self.assertIn("timeout_ms=CHALLENGE_TIMEOUT_MS", source)
        self.assertIn("last_run.controls == current_controls", source)
        self.assertIn("native model was discarded", source)
        self.assertIn("_existing_java_tool_options", source)
        self.assertIn("_existing_java_tokens", source)
        self.assertIn("_missing_jvm_opens", source)
        self.assertIn("option not in _existing_java_tokens", source)

    def test_page_reconciles_partial_jvm_opens_without_dropping_options(self):
        existing = (
            "--enable-native-access=ALL-UNNAMED "
            "--add-opens=java.base/java.net=ALL-UNNAMED "
            "--add-opens=java.base/java.util=ALL-UNNAMED"
        )
        required = (
            "--add-opens=java.base/java.util=ALL-UNNAMED",
            "--add-opens=java.base/java.lang=ALL-UNNAMED",
            "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
            "--add-opens=java.base/java.io=ALL-UNNAMED",
        )

        with patch.dict(os.environ, {"JAVA_TOOL_OPTIONS": existing}):
            app = AppTest.from_file(str(PAGE_PATH)).run(timeout=30)
            options = os.environ["JAVA_TOOL_OPTIONS"].split()

            if app.exception:
                details = "\n".join(str(item.value) for item in app.exception)
                self.fail(f"Plant Operator page raised exceptions:\n{details}")
            self.assertIn("--enable-native-access=ALL-UNNAMED", options)
            self.assertIn("--add-opens=java.base/java.net=ALL-UNNAMED", options)
            for required_option in required:
                self.assertEqual(options.count(required_option), 1)


if __name__ == "__main__":
    unittest.main()
