"""Source-level guardrails for Studio-aware Process Chat handoff."""

from pathlib import Path
import unittest

from studio.navigation import STATUS_AVAILABLE, destination_by_key


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProcessChatStudioHandoffTest(unittest.TestCase):
    def test_process_chat_is_an_available_studio_destination(self):
        destination = destination_by_key("chat")

        self.assertEqual(destination.status, STATUS_AVAILABLE)
        self.assertEqual(destination.page, "pages/90_Process_Chat.py")

    def test_process_chat_consumes_and_updates_shared_case_context(self):
        source = (PROJECT_ROOT / "pages" / "90_Process_Chat.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("from studio.case_context import", source)
        self.assertIn("get_active_case(st.session_state)", source)
        self.assertIn("mark_active_runtime_changed(", source)
        self.assertIn("detach_active_runtime_model(", source)
        self.assertIn('st.switch_page("pages/00_NeqSim_Studio.py")', source)
        self.assertIn(
            'st.switch_page("pages/35_Process_Flowsheet_Studio.py")',
            source,
        )


if __name__ == "__main__":
    unittest.main()
