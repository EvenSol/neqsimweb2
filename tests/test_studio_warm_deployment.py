"""Headless regression for Streamlit warm-deployment module refreshes."""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path

import process_chat.flowsheet_editor as flowsheet_editor
from streamlit.testing.v1 import AppTest


class StudioWarmDeploymentTest(unittest.TestCase):
    """Exercise the deployed page with a deliberately stale module cache."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        if "theme" not in sys.modules:
            try:
                importlib.import_module("theme")
            except ModuleNotFoundError:
                theme = types.ModuleType("theme")
                theme.apply_theme = lambda: None
                theme.theme_toggle = lambda: None
                sys.modules["theme"] = theme

    def tearDown(self):
        importlib.reload(flowsheet_editor)

    def test_page_recovers_stale_editor_module(self):
        del flowsheet_editor.connect_graph_ports
        studio_path = (
            self.project_root / "pages" / "35_Process_Flowsheet_Studio.py"
        )

        app = AppTest.from_file(str(studio_path)).run(timeout=120)

        if app.exception:
            details = "\n".join(str(item.value) for item in app.exception)
            self.fail(f"Studio raised exceptions after warm reload:\n{details}")
        self.assertTrue(callable(flowsheet_editor.connect_graph_ports))


if __name__ == "__main__":
    unittest.main()
