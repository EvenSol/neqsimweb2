"""Headless regression for Streamlit warm-deployment module refreshes."""

from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from typing import Any

import process_chat.flowsheet_editor as flowsheet_editor
import process_chat.solver_diagnostics as solver_diagnostics
from streamlit.testing.v1 import AppTest


def _standalone_exit_code(result: unittest.TestResult) -> int:
    """Match unittest's standalone status without waiting for JVM teardown."""
    if result.testsRun == 0:
        return 5
    return 0 if result.wasSuccessful() else 1


def _flush_standalone_output(*streams) -> None:
    """Best-effort flush logs before the standalone process hard-exits."""
    for stream in streams:
        try:
            stream.flush()
        except (BrokenPipeError, OSError, ValueError):
            continue


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

    def _load_studio_function(self, function_name):
        studio_path = (
            self.project_root / "pages" / "35_Process_Flowsheet_Studio.py"
        )
        tree = ast.parse(studio_path.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == function_name
        )
        namespace = {"Any": Any}
        exec(
            compile(
                ast.Module(body=[function], type_ignores=[]),
                str(studio_path),
                "exec",
            ),
            namespace,
        )
        return namespace[function_name]

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

    def test_page_exposes_feed_and_standalone_equipment_creation(self):
        app = self._run_studio()
        button_labels = {button.label for button in app.button}

        self.assertIn("Add feed stream", button_labels)
        self.assertIn("Add equipment node", button_labels)
        self.assertIn("Connect selected ports", button_labels)

    def test_graph_object_name_falls_back_for_legacy_records(self):
        graph_object_name = self._load_studio_function(
            "_graph_object_name"
        )

        self.assertEqual(
            graph_object_name({"id": "feed-a", "name": "Feed A"}, "feed-a"),
            "Feed A",
        )
        self.assertEqual(
            graph_object_name({"id": "feed-a"}, "feed-a"),
            "feed-a",
        )
        self.assertEqual(
            graph_object_name({"id": "feed-a", "name": " "}, "feed-a"),
            "feed-a",
        )

    def test_standalone_no_test_selection_returns_five(self):
        studio_test_path = Path(__file__).resolve()
        completed = subprocess.run(
            [
                sys.executable,
                str(studio_test_path),
                "-k",
                "definitely_no_matching_test",
            ],
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        self.assertEqual(
            completed.returncode,
            5,
            completed.stdout + completed.stderr,
        )

    def test_standalone_flush_continues_after_broken_pipe(self):
        flushed = []

        class BrokenStream:
            def flush(self):
                raise BrokenPipeError

        class HealthyStream:
            def flush(self):
                flushed.append(True)

        _flush_standalone_output(BrokenStream(), HealthyStream())

        self.assertEqual(flushed, [True])

    def test_zero_coverage_is_not_reported_as_not_applicable(self):
        coverage_label = self._load_studio_function(
            "_unit_balance_coverage_label"
        )
        zero_coverage = {
            "applicable": False,
            "coverage_complete": False,
            "energy_unit_count": 0.0,
            "energy_coverage_complete": False,
        }
        no_candidates = {
            **zero_coverage,
            "coverage_complete": True,
        }

        self.assertEqual(
            coverage_label(zero_coverage),
            "Material unavailable; energy not audited",
        )
        self.assertEqual(coverage_label(no_candidates), "Not applicable")

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
    program = unittest.main(exit=False)
    exit_code = _standalone_exit_code(program.result)
    try:
        _flush_standalone_output(sys.stdout, sys.stderr)
    finally:
        os._exit(exit_code)
