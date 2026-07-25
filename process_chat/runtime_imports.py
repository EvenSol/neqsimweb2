"""Deployment-safe imports for local modules used by Streamlit pages."""

from __future__ import annotations

import importlib
import threading
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable


_MODULE_LOCKS: dict[str, threading.Lock] = {}
_MODULE_LOCKS_GUARD = threading.Lock()


def _module_lock(module_name: str) -> threading.Lock:
    """Return the stable reload lock for one module name."""
    with _MODULE_LOCKS_GUARD:
        return _MODULE_LOCKS.setdefault(module_name, threading.Lock())


def _validated_symbol_names(symbol_names: Iterable[str]) -> tuple[str, ...]:
    """Return unique, non-empty symbol names in caller-specified order."""
    if isinstance(symbol_names, (str, bytes)):
        raise TypeError("symbol_names must be an iterable of names")

    names = tuple(symbol_names)
    if not names:
        raise ValueError("symbol_names cannot be empty")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("symbol_names must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("symbol_names cannot contain duplicates")
    return names


def _assert_local_module(module: ModuleType, project_root: Path) -> None:
    """Reject reload attempts for modules outside the deployed project."""
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise ImportError(
            f"Cannot refresh {module.__name__!r}: module has no source file"
        )

    resolved_file = Path(module_file).resolve()
    resolved_root = project_root.resolve()
    try:
        resolved_file.relative_to(resolved_root)
    except ValueError as exc:
        raise ImportError(
            f"Refusing to refresh non-project module {module.__name__!r} "
            f"from {resolved_file}"
        ) from exc


def import_local_symbols(
    module_name: str,
    symbol_names: Iterable[str],
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Import symbols, refreshing a stale local module cache once if needed.

    Streamlit can rerun a changed page in a long-lived interpreter while an
    imported project module remains cached from the preceding deployment.
    When newly deployed symbols are missing, reload that local module once
    from the current checkout and then fail explicitly if it is still
    incompatible.
    """
    if not isinstance(module_name, str) or not module_name.strip():
        raise ValueError("module_name must be a non-empty string")
    names = _validated_symbol_names(symbol_names)

    module = importlib.import_module(module_name)
    with _module_lock(module_name):
        missing = [name for name in names if not hasattr(module, name)]
        if missing:
            _assert_local_module(module, Path(project_root))
            importlib.invalidate_caches()
            for name in names:
                module.__dict__.pop(name, None)
            module = importlib.reload(module)
            missing = [name for name in names if not hasattr(module, name)]

        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ImportError(
                f"Module {module_name!r} does not provide required symbols "
                f"after refresh: {missing_text}"
            )

        return {name: getattr(module, name) for name in names}
