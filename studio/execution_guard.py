"""Process-local arbitration for complete Studio native transactions."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator


_NATIVE_EXECUTION_TRANSACTION_LOCK = threading.Lock()


class StudioExecutionBusyError(TimeoutError):
    """Raised when a Studio native transaction cannot start within its budget."""


@contextmanager
def native_execution_transaction(timeout_ms: int) -> Iterator[None]:
    """Serialize in-process NeqSim transactions within one bounded wait."""
    if timeout_ms <= 0:
        raise ValueError("timeout_ms must be positive for transaction waits")
    acquired = _NATIVE_EXECUTION_TRANSACTION_LOCK.acquire(
        timeout=timeout_ms / 1000.0
    )
    if not acquired:
        raise StudioExecutionBusyError(
            "Native NeqSim execution capacity remained busy for "
            f"{int(timeout_ms)} ms; no process model was built."
        )
    try:
        yield
    finally:
        _NATIVE_EXECUTION_TRANSACTION_LOCK.release()
