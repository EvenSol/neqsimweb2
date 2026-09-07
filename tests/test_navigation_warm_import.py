"""Router regressions for a deployment with the previous navigation module cached."""

import os
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class Sidebar:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class WarmNavigationImportTest(unittest.TestCase):
    def run_router(self, url, initial_state=None):
        # Reproduce the pre-PR-136 public API still held in sys.modules.
        old_navigation = ModuleType("app_navigation")
        old_navigation.experimental_page_specs = lambda: [
            SimpleNamespace(path="pages/100_CO2_mechanisticModel.py", title="CO2")
        ]
        old_navigation.stable_page_specs = lambda: [
            SimpleNamespace(path="pages/0_TP_flash.py", title="TP Flash")
        ]
        streamlit = ModuleType("streamlit")
        streamlit.session_state = dict(initial_state or {})
        streamlit.context = SimpleNamespace(url=url)
        streamlit.sidebar = Sidebar()
        streamlit.divider = lambda: None
        streamlit.caption = lambda *args: None
        streamlit.toggle = lambda *args, **kwargs: None
        executed = []

        def page(path, **kwargs):
            route = "" if kwargs.get("default") else (
                Path(path).stem.split("_", 1)[1]
            )
            return SimpleNamespace(
                url_path=route, run=lambda: executed.append(path)
            )

        def navigation(groups):
            requested = url.rsplit("/", 1)[-1]
            pages = [item for group in groups.values() for item in group]
            return next(
                (item for item in pages if item.url_path == requested), pages[0]
            )

        streamlit.Page = page
        streamlit.navigation = navigation
        with patch.dict(sys.modules, {
            "app_navigation": old_navigation, "streamlit": streamlit,
        }), patch.dict(os.environ):
            runpy.run_path(str(ROOT / "welcome.py"), run_name="__main__")
        return streamlit.session_state, executed

    def test_direct_co2_link_with_previous_module_cached(self):
        state, executed = self.run_router(
            "https://neqsim.streamlit.app/CO2_mechanisticModel"
        )
        self.assertTrue(state["experimental_mode"])
        self.assertTrue(state["_experimental_mode_toggle"])
        self.assertEqual(executed, ["pages/100_CO2_mechanisticModel.py"])

    def test_home_with_previous_module_cached(self):
        state, executed = self.run_router("https://neqsim.streamlit.app/")
        self.assertFalse(state["experimental_mode"])
        self.assertEqual(executed, ["home.py"])

    def test_explicit_off_survives_rerun_with_previous_module_cached(self):
        state, executed = self.run_router(
            "https://neqsim.streamlit.app/CO2_mechanisticModel",
            {"experimental_mode": False, "_experimental_mode_toggle": False},
        )
        self.assertFalse(state["experimental_mode"])
        self.assertEqual(executed, ["home.py"])


if __name__ == "__main__":
    unittest.main()
