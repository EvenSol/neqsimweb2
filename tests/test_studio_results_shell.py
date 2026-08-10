"""Streamlit smoke test for the Studio engineering-results page."""

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


class StudioResultsShellTest(unittest.TestCase):
    def test_unsolved_workspace_renders_actionable_state(self):
        project_root = Path(__file__).resolve().parents[1]
        app = AppTest.from_file(
            str(project_root / "pages" / "10_Studio_Results.py")
        ).run(timeout=30)

        if app.exception:
            details = "\n".join(str(item.value) for item in app.exception)
            self.fail(f"Studio results page raised exceptions:\n{details}")

        self.assertIn("Open Process Flowsheet", [button.label for button in app.button])


if __name__ == "__main__":
    unittest.main()
