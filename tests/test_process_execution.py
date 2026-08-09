"""Regression tests for bounded, fail-loud native process execution."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from process_chat.process_model import (
    NeqSimProcessModel,
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


if __name__ == "__main__":
    unittest.main()
