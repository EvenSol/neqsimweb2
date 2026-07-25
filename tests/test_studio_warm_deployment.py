"""Headless regression for Streamlit warm-deployment module refreshes."""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path

import process_chat.flowsheet_editor as flowsheet_editor
import process_chat.solver_diagnostics as solver_diagnostics
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
        importlib.reload(solver_diagnostics)

    def _run_studio(self):
        studio_path = (
            self.project_root / "pages" / "35_Process_Flowsheet_Studio.py"
        )

        app = AppTest.from_file(str(studio_path)).run(timeout=120)

        if app.exception:
            details = "\n".join(str(item.value) for item in app.exception)
            self.fail(f"Studio raised exceptions after warm reload:\n{details}")

    def test_page_recovers_stale_editor_module(self):
        del flowsheet_editor.connect_graph_ports

        self._run_studio()

        self.assertTrue(callable(flowsheet_editor.connect_graph_ports))

    def test_page_recovers_stale_solver_diagnostics_module(self):
        del solver_diagnostics.aggregate_energy_balance
        del solver_diagnostics.energy_transfer_rows

        self._run_studio()

        self.assertTrue(
            callable(solver_diagnostics.aggregate_energy_balance)
        )
        self.assertTrue(callable(solver_diagnostics.energy_transfer_rows))

    def test_page_recovers_multiple_stale_local_modules(self):
        del flowsheet_editor.connect_graph_ports
        del solver_diagnostics.aggregate_energy_balance

        self._run_studio()

        self.assertTrue(callable(flowsheet_editor.connect_graph_ports))
        self.assertTrue(
            callable(solver_diagnostics.aggregate_energy_balance)
        )


if __name__ == "__main__":
    unittest.main()
