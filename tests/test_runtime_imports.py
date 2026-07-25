"""Regression tests for deployment-safe local module imports."""

from __future__ import annotations

import importlib
import sys
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import process_chat.flowsheet_editor as flowsheet_editor
from process_chat.runtime_imports import import_local_symbols


class LocalSymbolImportTest(unittest.TestCase):
    """Exercise fresh, stale, incompatible, and unsafe module imports."""

    def setUp(self):
        self.project_root = Path(__file__).resolve().parents[1]

    def tearDown(self):
        importlib.reload(flowsheet_editor)
        sys.modules.pop("outside_project_module", None)

    def test_returns_requested_symbols_without_refreshing_current_module(self):
        with mock.patch(
            "process_chat.runtime_imports.importlib.reload",
            wraps=importlib.reload,
        ) as reload_module:
            symbols = import_local_symbols(
                "process_chat.flowsheet_editor",
                ("connect_graph_ports", "disconnect_graph_connection"),
                project_root=self.project_root,
            )

        self.assertIs(
            symbols["connect_graph_ports"],
            flowsheet_editor.connect_graph_ports,
        )
        self.assertIs(
            symbols["disconnect_graph_connection"],
            flowsheet_editor.disconnect_graph_connection,
        )
        reload_module.assert_not_called()

    def test_refreshes_stale_project_module_once(self):
        del flowsheet_editor.connect_graph_ports

        with mock.patch(
            "process_chat.runtime_imports.importlib.reload",
            wraps=importlib.reload,
        ) as reload_module:
            symbols = import_local_symbols(
                "process_chat.flowsheet_editor",
                ("connect_graph_ports",),
                project_root=self.project_root,
            )

        self.assertTrue(callable(symbols["connect_graph_ports"]))
        self.assertTrue(callable(flowsheet_editor.connect_graph_ports))
        reload_module.assert_called_once_with(flowsheet_editor)

    def test_serializes_concurrent_refreshes_for_one_module(self):
        del flowsheet_editor.connect_graph_ports
        reload_started = threading.Event()
        second_started = threading.Event()
        release_reload = threading.Event()
        actual_reload = importlib.reload

        def delayed_reload(module):
            reload_started.set()
            self.assertTrue(release_reload.wait(timeout=5))
            return actual_reload(module)

        def import_symbols(started=None):
            if started is not None:
                started.set()
            return import_local_symbols(
                "process_chat.flowsheet_editor",
                ("connect_graph_ports",),
                project_root=self.project_root,
            )

        with mock.patch(
            "process_chat.runtime_imports.importlib.reload",
            side_effect=delayed_reload,
        ) as reload_module:
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(import_symbols)
                self.assertTrue(reload_started.wait(timeout=5))
                second = executor.submit(import_symbols, second_started)
                self.assertTrue(second_started.wait(timeout=5))
                release_reload.set()
                results = (first.result(timeout=5), second.result(timeout=5))

        self.assertTrue(
            all(callable(result["connect_graph_ports"]) for result in results)
        )
        reload_module.assert_called_once_with(flowsheet_editor)

    def test_missing_symbol_fails_after_one_refresh(self):
        with mock.patch(
            "process_chat.runtime_imports.importlib.reload",
            wraps=importlib.reload,
        ) as reload_module:
            with self.assertRaisesRegex(
                ImportError,
                "does not provide required symbols after refresh: not_available",
            ):
                import_local_symbols(
                    "process_chat.flowsheet_editor",
                    ("not_available",),
                    project_root=self.project_root,
                )

        reload_module.assert_called_once_with(flowsheet_editor)

    def test_removed_export_does_not_survive_refresh(self):
        removed_name = "_removed_studio_export"
        setattr(flowsheet_editor, removed_name, object())
        self.addCleanup(
            lambda: flowsheet_editor.__dict__.pop(removed_name, None)
        )
        del flowsheet_editor.connect_graph_ports

        with self.assertRaisesRegex(
            ImportError,
            f"after refresh: {removed_name}",
        ):
            import_local_symbols(
                "process_chat.flowsheet_editor",
                ("connect_graph_ports", removed_name),
                project_root=self.project_root,
            )

        self.assertTrue(callable(flowsheet_editor.connect_graph_ports))
        self.assertFalse(hasattr(flowsheet_editor, removed_name))

    def test_failed_refresh_restores_requested_namespace(self):
        original_disconnect = flowsheet_editor.disconnect_graph_connection
        del flowsheet_editor.connect_graph_ports

        with mock.patch(
            "process_chat.runtime_imports.importlib.reload",
            side_effect=RuntimeError("reload failed"),
        ) as reload_module:
            with self.assertRaisesRegex(RuntimeError, "reload failed"):
                import_local_symbols(
                    "process_chat.flowsheet_editor",
                    ("connect_graph_ports", "disconnect_graph_connection"),
                    project_root=self.project_root,
                )

        self.assertFalse(hasattr(flowsheet_editor, "connect_graph_ports"))
        self.assertIs(
            flowsheet_editor.disconnect_graph_connection,
            original_disconnect,
        )
        reload_module.assert_called_once_with(flowsheet_editor)

    def test_refuses_to_refresh_module_outside_project(self):
        outside_module = types.ModuleType("outside_project_module")
        outside_module.__file__ = "/tmp/outside_project_module.py"
        sys.modules[outside_module.__name__] = outside_module

        with self.assertRaisesRegex(
            ImportError,
            "Refusing to refresh non-project module",
        ):
            import_local_symbols(
                outside_module.__name__,
                ("new_symbol",),
                project_root=self.project_root,
            )

    def test_rejects_invalid_module_and_symbol_requests(self):
        invalid_requests = (
            ("", ("symbol",), "module_name must be a non-empty string"),
            ("module", (), "symbol_names cannot be empty"),
            ("module", "symbol", "symbol_names must be an iterable"),
            ("module", (" ",), "symbol_names must contain non-empty strings"),
            ("module", ("a", "a"), "symbol_names cannot contain duplicates"),
        )
        for module_name, names, message in invalid_requests:
            with self.subTest(module_name=module_name, names=names):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    import_local_symbols(
                        module_name,
                        names,
                        project_root=self.project_root,
                    )

    def test_studio_page_uses_deployment_safe_editor_imports(self):
        studio_path = (
            self.project_root / "pages" / "35_Process_Flowsheet_Studio.py"
        )
        studio_source = studio_path.read_text(encoding="utf-8")

        self.assertIn(
            'import_local_symbols(\n'
            '        "process_chat.flowsheet_editor",',
            studio_source,
        )
        self.assertNotIn(
            "from process_chat.flowsheet_editor import",
            studio_source,
        )


if __name__ == "__main__":
    unittest.main()
