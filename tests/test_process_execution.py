"""Regression tests for bounded, fail-loud native process execution."""

from __future__ import annotations

import ast
import unittest
import threading
from pathlib import Path
from unittest.mock import patch

from process_chat.process_builder import ProcessBuilder
from process_chat.process_model import (
    NeqSimProcessModel,
    ProcessExecutionError,
    ProcessRunTimeoutError,
)


class ProcessRunTimeoutTest(unittest.TestCase):
    """Keep worker cancellation and the total convergence budget bounded."""

    def test_interrupt_wait_is_bounded_and_model_is_discarded(self):
        class _Thread:
            def __init__(self, stops_after_interrupt):
                self.stops_after_interrupt = stops_after_interrupt
                self.interrupted = False
                self.join_timeouts = []

            def join(self, timeout_ms):
                self.join_timeouts.append(timeout_ms)

            def isAlive(self):
                return not (
                    self.interrupted and self.stops_after_interrupt
                )

            def interrupt(self):
                self.interrupted = True

        class _Process:
            def __init__(self, thread):
                self.thread = thread

            def runAsThread(self):
                return self.thread

        for stops_after_interrupt in (True, False):
            with self.subTest(stops_after_interrupt=stops_after_interrupt):
                thread = _Thread(stops_after_interrupt)
                with self.assertRaisesRegex(
                    ProcessRunTimeoutError,
                    "discard",
                ):
                    NeqSimProcessModel._run_native_thread(
                        _Process(thread),
                        25,
                        cancellation_grace_ms=10,
                    )

                self.assertTrue(thread.interrupted)
                self.assertEqual(thread.join_timeouts, [25, 10])

    def test_cancellation_grace_must_not_use_unbounded_zero_join(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            NeqSimProcessModel._run_native_thread(
                object(),
                25,
                cancellation_grace_ms=0,
            )

    def test_convergence_timeout_is_one_total_budget(self):
        class _JavaClass:
            @staticmethod
            def getSimpleName():
                return "Heater"

        class _Unit:
            @staticmethod
            def getClass():
                return _JavaClass()

            @staticmethod
            def getDuty():
                return 0.0

        class _CompletedThread:
            @staticmethod
            def join(timeout_ms):
                return None

            @staticmethod
            def isAlive():
                return False

        class _Process:
            def __init__(self):
                self.units = [_Unit(), _Unit(), _Unit()]
                self.run_count = 0

            def getUnitOperations(self):
                return self.units

            def runAsThread(self):
                self.run_count += 1
                return _CompletedThread()

        process = _Process()
        with patch(
            "process_chat.process_model.monotonic",
            side_effect=(100.0, 100.001, 100.010),
        ):
            with self.assertRaisesRegex(
                ProcessRunTimeoutError,
                "exceeded 5 ms",
            ):
                NeqSimProcessModel._run_until_converged(
                    process,
                    max_runs=3,
                    timeout_ms=5,
                )

        self.assertEqual(process.run_count, 1)

    def test_sequential_convergence_fallback_uses_remaining_budget(self):
        class _JavaClass:
            @staticmethod
            def getSimpleName():
                return "Heater"

        class _Unit:
            @staticmethod
            def getClass():
                return _JavaClass()

            @staticmethod
            def getDuty():
                return 0.0

        class _CompletedThread:
            @staticmethod
            def join(timeout_ms):
                return None

            @staticmethod
            def isAlive():
                return False

        class _Process:
            def __init__(self):
                self.units = [_Unit(), _Unit(), _Unit()]
                self.threaded_runs = 0
                self.sequential_runs = 0

            def getUnitOperations(self):
                return self.units

            def runAsThread(self):
                self.threaded_runs += 1
                return _CompletedThread()

            def runSequential(self):
                self.sequential_runs += 1

        process = _Process()
        with patch(
            "process_chat.process_model.monotonic",
            side_effect=(100.0, 100.1, 100.2, 100.3, 100.4),
        ):
            self.assertTrue(
                NeqSimProcessModel._run_until_converged(
                    process,
                    max_runs=4,
                    timeout_ms=1_000,
                )
            )

        self.assertEqual(process.threaded_runs, 3)
        self.assertEqual(process.sequential_runs, 1)

    def test_process_model_fallback_shares_one_total_budget(self):
        class _ProcessModel:
            @staticmethod
            def runAsThread():
                raise RuntimeError("parallel dispatch unavailable")

            @staticmethod
            def getAllProcesses():
                return ("first", "second")

        observed_timeouts = []

        def run_child(process_system, *, timeout_ms, **kwargs):
            observed_timeouts.append((process_system, timeout_ms))
            return True

        with (
            patch(
                "process_chat.process_model.monotonic",
                side_effect=(10.0, 10.1, 10.2, 10.4),
            ),
            patch.object(
                NeqSimProcessModel,
                "_run_until_converged",
                side_effect=run_child,
            ),
        ):
            self.assertTrue(
                NeqSimProcessModel._run_process_model(
                    _ProcessModel(),
                    timeout_ms=1_000,
                )
            )

        self.assertEqual(
            observed_timeouts,
            [("first", 800), ("second", 599)],
        )

    def test_python_orchestrated_native_closure_wait_is_bounded(self):
        release = threading.Event()

        def blocking_call():
            release.wait(timeout=1)

        try:
            with self.assertRaisesRegex(
                ProcessRunTimeoutError,
                "mixer closure exceeded 1 ms",
            ):
                NeqSimProcessModel._run_bounded_call(
                    blocking_call,
                    1,
                    operation="mixer closure",
                )
        finally:
            release.set()

    def test_complete_process_construction_wait_is_bounded(self):
        release = threading.Event()
        builder = ProcessBuilder()

        def blocking_build(*args, **kwargs):
            release.wait(timeout=1)

        try:
            with (
                patch.object(
                    builder,
                    "build_from_spec",
                    side_effect=blocking_build,
                ),
                self.assertRaisesRegex(
                    ProcessRunTimeoutError,
                    "process construction exceeded 1 ms",
                ),
            ):
                builder.build_from_spec_bounded({}, timeout_ms=1)
        finally:
            release.set()

    def test_complete_model_run_and_result_extraction_wait_is_bounded(self):
        release = threading.Event()
        model = object.__new__(NeqSimProcessModel)

        def blocking_run(*args, **kwargs):
            release.wait(timeout=1)

        try:
            with (
                patch.object(model, "run", side_effect=blocking_run),
                self.assertRaisesRegex(
                    ProcessRunTimeoutError,
                    "process solve and result extraction exceeded 1 ms",
                ),
            ):
                model.run_bounded(timeout_ms=1)
        finally:
            release.set()

    def test_complete_model_serialization_wait_is_bounded(self):
        release = threading.Event()
        model = object.__new__(NeqSimProcessModel)

        def blocking_save():
            release.wait(timeout=1)

        try:
            with (
                patch.object(model, "save_bytes", side_effect=blocking_save),
                self.assertRaisesRegex(
                    ProcessRunTimeoutError,
                    "process serialization exceeded 1 ms",
                ),
            ):
                model.save_bytes_bounded(timeout_ms=1)
        finally:
            release.set()


class ProcessRunFailureTest(unittest.TestCase):
    """Prevent failed native workers from publishing partial solved state."""

    @staticmethod
    def _model():
        model = object.__new__(NeqSimProcessModel)
        model._proc = object()
        model._is_process_model = False
        model._enforce_acyclic_mixer_energy = False
        model._equipment_design_bases = {}
        model._direct_unit_run_provenance = {}
        model._heat_exchanger_state_snapshots = {}
        return model

    def test_run_rejects_failed_worker_before_result_extraction(self):
        model = self._model()
        with (
            patch.object(
                NeqSimProcessModel,
                "_run_until_converged",
                return_value=False,
            ),
            patch.object(model, "_index_model_objects") as reindex,
            patch.object(model, "_extract_results") as extract,
        ):
            with self.assertRaisesRegex(
                ProcessExecutionError,
                "no solved results were published",
            ):
                model.run(timeout_ms=30)

        reindex.assert_not_called()
        extract.assert_not_called()

    def test_rerun_rejects_failed_worker_before_reindexing(self):
        model = self._model()
        with (
            patch.object(
                NeqSimProcessModel,
                "_run_until_converged",
                return_value=False,
            ),
            patch.object(model, "_index_model_objects") as reindex,
        ):
            with self.assertRaisesRegex(
                ProcessExecutionError,
                "discard this process model",
            ):
                model.rerun(timeout_ms=30)

        reindex.assert_not_called()

    def test_model_run_shares_budget_with_mixer_closure(self):
        model = self._model()
        model._enforce_acyclic_mixer_energy = True
        with (
            patch(
                "process_chat.process_model.monotonic",
                side_effect=(10.0, 10.1, 10.4),
            ),
            patch.object(
                NeqSimProcessModel,
                "_run_until_converged",
                return_value=True,
            ) as run_process,
            patch.object(
                NeqSimProcessModel,
                "_run_acyclic_mixer_energy_closure",
                return_value=True,
            ) as close_mixer,
            patch.object(model, "_index_model_objects"),
            patch.object(model, "_capture_heat_exchanger_state_snapshots"),
            patch.object(model, "_extract_results", return_value=object()),
        ):
            model.run(timeout_ms=1_000)

        self.assertEqual(run_process.call_args.kwargs["timeout_ms"], 900)
        self.assertEqual(close_mixer.call_args.kwargs["timeout_ms"], 599)


class StudioExecutionContractTest(unittest.TestCase):
    """Keep the page wired to the bounded adapter execution contract."""

    def test_studio_uses_explicit_budget_and_separate_timeout_state(self):
        studio_path = (
            Path(__file__).resolve().parents[1]
            / "pages"
            / "35_Process_Flowsheet_Studio.py"
        )
        source = studio_path.read_text(encoding="utf-8")

        self.assertIn("STUDIO_SOLVE_TIMEOUT_MS = 180_000", source)
        self.assertIn(
            'f"{STUDIO_SOLVE_TIMEOUT_MS} ms total "',
            source,
        )
        self.assertNotIn("exceeded the 180000 ms", source)
        self.assertIn(
            "timeout_ms=remaining_execution_budget_ms()",
            source,
        )
        self.assertIn("builder.build_from_spec_bounded(", source)
        self.assertIn("model.run_bounded(", source)
        self.assertIn("builder.save_neqsim_bytes_bounded(", source)
        self.assertIn("execution_deadline", source)
        self.assertIn("except TimeoutError as exc:", source)
        self.assertTrue(issubclass(ProcessRunTimeoutError, TimeoutError))
        self.assertIn(
            'f"{STUDIO_SOLVE_TIMEOUT_MS / 1000:.0f} s execution budget. "',
            source,
        )
        self.assertNotIn("exceeded the 180 s execution budget", source)
        self.assertIn('solver_status = "Timed out"', source)
        self.assertIn(
            'st.session_state[FAILURE_KIND_STATE_KEY] = "timeout"',
            source,
        )
        self.assertIn("no results were published", source)

    def test_timeout_classification_survives_streamlit_rerun(self):
        studio_path = (
            Path(__file__).resolve().parents[1]
            / "pages"
            / "35_Process_Flowsheet_Studio.py"
        )
        parsed = ast.parse(studio_path.read_text(encoding="utf-8"))
        solver_status_node = next(
            node
            for node in parsed.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_solver_status"
        )
        namespace = {}
        function_module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                solver_status_node,
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(function_module)
        exec(compile(function_module, str(studio_path), "exec"), namespace)

        status, is_current = namespace["_solver_status"](
            current_signature="case-a",
            stored_state=None,
            has_result=False,
            failure_signature="case-a",
            failure_kind="timeout",
        )

        self.assertEqual(status, "Timed out")
        self.assertFalse(is_current)

        status, is_current = namespace["_solver_status"](
            current_signature="case-a",
            stored_state={"signature": "case-a"},
            has_result=True,
            failure_signature="case-a",
            failure_kind="timeout",
        )

        self.assertEqual(status, "Timed out")
        self.assertFalse(is_current)


if __name__ == "__main__":
    unittest.main()
