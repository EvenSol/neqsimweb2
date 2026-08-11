"""HTTP health regressions for the Classic and Studio Streamlit entry points."""

from __future__ import annotations

from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.error import URLError
from urllib.request import urlopen

from studio.navigation import STUDIO_DESTINATIONS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROBED_WORKSPACE_ROUTES = frozenset(
    {
        "welcome.py",
        "pages/00_NeqSim_Studio.py",
        "pages/10_Studio_Results.py",
        "pages/25_Pipeline.py",
        "pages/35_Process_Flowsheet_Studio.py",
        "pages/90_Process_Chat.py",
    }
)


def _free_port() -> int:
    """Return an available local TCP port for one isolated Streamlit probe."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class WorkspaceHttpHealthTest(unittest.TestCase):
    """Require both user-selectable workspaces to start and stay healthy."""

    def _probe_streamlit_page(self, relative_page: str) -> None:
        page = PROJECT_ROOT / relative_page
        port = _free_port()
        root_url = f"http://127.0.0.1:{port}/"
        health_url = f"http://127.0.0.1:{port}/_stcore/health"

        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(page),
                    "--global.developmentMode=false",
                    "--server.headless=true",
                    "--server.address=127.0.0.1",
                    f"--server.port={port}",
                ],
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                deadline = time.monotonic() + 45.0
                last_error = "Streamlit did not become reachable."
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        last_error = f"Streamlit exited with code {process.returncode}."
                        break
                    try:
                        with urlopen(health_url, timeout=2.0) as response:
                            body = response.read().decode("utf-8").strip()
                            if response.status == 200 and body == "ok":
                                break
                    except (OSError, URLError) as exc:
                        last_error = str(exc)
                    time.sleep(0.5)
                else:
                    last_error = "Timed out waiting for Streamlit health endpoint."

                if process.poll() is not None or time.monotonic() >= deadline:
                    log_file.seek(0)
                    self.fail(
                        f"{relative_page} failed startup: {last_error}\n"
                        f"{log_file.read()}"
                    )

                for _ in range(2):
                    self.assertIsNone(
                        process.poll(),
                        f"{relative_page} exited during the health probe.",
                    )
                    with urlopen(root_url, timeout=5.0) as response:
                        self.assertEqual(response.status, 200)
                        self.assertGreater(len(response.read()), 0)
                    with urlopen(health_url, timeout=5.0) as response:
                        self.assertEqual(response.status, 200)
                        self.assertEqual(
                            response.read().decode("utf-8").strip(),
                            "ok",
                        )
                    time.sleep(0.5)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=8.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5.0)

    def test_all_available_studio_routes_have_a_fresh_process_gate(self):
        available_routes = {
            destination.page
            for destination in STUDIO_DESTINATIONS
            if destination.available
        }

        self.assertTrue(available_routes)
        self.assertLessEqual(available_routes, PROBED_WORKSPACE_ROUTES)

    def test_classic_http_health(self):
        self._probe_streamlit_page("welcome.py")

    def test_studio_http_health(self):
        self._probe_streamlit_page("pages/00_NeqSim_Studio.py")

    def test_studio_results_http_health(self):
        self._probe_streamlit_page("pages/10_Studio_Results.py")

    def test_process_flowsheet_studio_http_health(self):
        self._probe_streamlit_page("pages/35_Process_Flowsheet_Studio.py")

    def test_process_chat_http_health(self):
        self._probe_streamlit_page("pages/90_Process_Chat.py")

    def test_pipeline_http_health(self):
        self._probe_streamlit_page("pages/25_Pipeline.py")


if __name__ == "__main__":
    unittest.main()
