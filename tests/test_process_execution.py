"""Regression tests for bounded, fail-loud native process execution."""

from __future__ import annotations

import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
