"""Hosted Streamlit smoke test for the NeqSim Studio shell."""

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


class StudioShellTest(unittest.TestCase):
    """Require the new Studio entry page to render without exceptions."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.studio_path = cls.project_root / "pages" / "00_NeqSim_Studio.py"

    def test_studio_dashboard_renders(self):
        app = AppTest.from_file(str(self.studio_path)).run(timeout=30)

        if app.exception:
            details = "\n".join(str(item.value) for item in app.exception)
            self.fail(f"Studio shell raised exceptions:\n{details}")

        button_labels = [button.label for button in app.button]
        self.assertIn("＋ New process case", button_labels)
        self.assertIn("Open uploaded case", button_labels)
        self.assertIn("⚙️ Open Process Flowsheet", button_labels)
        self.assertIn("Open Classic", button_labels)
        self.assertIn("Open Process Flowsheet", button_labels)
        self.assertIn("Coming soon", button_labels)
        uploader_labels = [uploader.label for uploader in app.file_uploader]
        self.assertIn("Open portable case JSON", uploader_labels)


if __name__ == "__main__":
    unittest.main()
