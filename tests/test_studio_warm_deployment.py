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
        return app

    def test_page_recovers_stale_editor_module(self):
        del flowsheet_editor.connect_graph_ports

        self._run_studio()

        self.assertTrue(callable(flowsheet_editor.connect_graph_ports))

    def test_page_recovers_stale_solver_diagnostics_module(self):
        del solver_diagnostics.aggregate_energy_balance
        del solver_diagnostics.aggregate_unit_balances
        del solver_diagnostics.energy_transfer_rows
        del solver_diagnostics.unit_balance_rows

        self._run_studio()

        self.assertTrue(
            callable(solver_diagnostics.aggregate_energy_balance)
        )
        self.assertTrue(
            callable(solver_diagnostics.aggregate_unit_balances)
        )
        self.assertTrue(callable(solver_diagnostics.energy_transfer_rows))
        self.assertTrue(callable(solver_diagnostics.unit_balance_rows))

    def test_page_recovers_multiple_stale_local_modules(self):
        del flowsheet_editor.connect_graph_ports
        del solver_diagnostics.aggregate_energy_balance

        self._run_studio()

        self.assertTrue(callable(flowsheet_editor.connect_graph_ports))
        self.assertTrue(
            callable(solver_diagnostics.aggregate_energy_balance)
        )

    def test_solved_page_reports_and_exports_unit_closure(self):
        app = self._run_studio()
        run_button = next(
            button
            for button in app.button
            if button.label == "▶ Run NeqSim flowsheet"
        )

        run_button.click()
        app.run(timeout=240)

        if app.exception:
            details = "\n".join(str(item.value) for item in app.exception)
            self.fail(f"Studio solve raised exceptions:\n{details}")
        closure_metrics = {
            metric.label: metric
            for metric in app.metric
            if metric.label.startswith("Maximum unit ")
        }
        self.assertEqual(
            set(closure_metrics),
            {
                "Maximum unit mass imbalance",
                "Maximum unit energy imbalance",
            },
        )
        mass_metric = closure_metrics["Maximum unit mass imbalance"]
        self.assertTrue(mass_metric.value.endswith(" %"))
        self.assertIn(" / ", mass_metric.help)
        energy_metric = closure_metrics["Maximum unit energy imbalance"]
        if energy_metric.value == "n/a":
            self.assertEqual(energy_metric.help, "n/a")
        else:
            self.assertTrue(energy_metric.value.endswith(" %"))
            self.assertIn(" / ", energy_metric.help)
        self.assertTrue(
            any(
                "Mass imbalance [%]" in dataframe.value.columns
                and "Energy imbalance [%]" in dataframe.value.columns
                and "Inlet enthalpy flow [kW]" in dataframe.value.columns
                and "Outlet enthalpy flow [kW]" in dataframe.value.columns
                for dataframe in app.dataframe
            )
        )
        self.assertIn(
            "Download engineering workbook",
            [button.label for button in app.get("download_button")],
        )
        captions = [caption.value for caption in app.caption]
        self.assertTrue(
            any(
                "Mass residual is outlet mass flow minus inlet mass flow."
                in caption
                for caption in captions
            )
        )
        self.assertTrue(
            any(
                "Energy residual is outlet enthalpy flow minus inlet "
                "enthalpy flow minus signed external energy transfer."
                in caption
                for caption in captions
            )
        )


if __name__ == "__main__":
    unittest.main()
