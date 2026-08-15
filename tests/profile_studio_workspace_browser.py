"""Profile a representative large solved Studio workspace in Chromium."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

from playwright.sync_api import sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "studio_large_results_browser_app.py"
PROFILE_OUTPUT = PROJECT_ROOT / "studio-browser-profile.json"
HOST = "127.0.0.1"
PORT = 8765
BASE_URL = f"http://{HOST}:{PORT}"
HEALTH_URL = f"{BASE_URL}/_stcore/health"

FIRST_CONTENTFUL_PAINT_BUDGET_MS = 10_000.0
VIEW_READY_BUDGET_SECONDS = 20.0
VIEW_NETWORK_BUDGET_BYTES = 50 * 1024 * 1024
HEAP_BUDGET_BYTES = 512 * 1024 * 1024


def _wait_for_health(process: subprocess.Popen[str], timeout_seconds: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "no response"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, _ = process.communicate(timeout=5)
            raise RuntimeError(
                f"Streamlit exited before becoming healthy (code {process.returncode}):\n"
                f"{stdout}"
            )
        try:
            with urlopen(HEALTH_URL, timeout=2) as response:
                body = response.read().decode("utf-8").strip()
                if response.status == 200 and body == "ok":
                    return
                last_error = f"HTTP {response.status}: {body!r}"
        except (OSError, URLError) as error:
            last_error = str(error)
        time.sleep(0.25)
    raise TimeoutError(f"Streamlit health did not become ready: {last_error}")


def _chromium_metrics(session, page) -> dict[str, float]:
    raw_metrics = session.send("Performance.getMetrics")["metrics"]
    metrics = {item["name"]: item["value"] for item in raw_metrics}
    paints = {
        item["name"]: item["startTime"]
        for item in page.evaluate("performance.getEntriesByType('paint')")
    }
    return {
        "first_paint_ms": float(paints.get("first-paint", -1.0)),
        "first_contentful_paint_ms": float(
            paints.get("first-contentful-paint", -1.0)
        ),
        "javascript_heap_bytes": float(metrics.get("JSHeapUsedSize", -1.0)),
        "dom_nodes": float(metrics.get("Nodes", -1.0)),
        "documents": float(metrics.get("Documents", -1.0)),
    }


def _assert_profile(profile: dict[str, object]) -> None:
    initial = profile["streams_view"]
    equipment = profile["equipment_view"]

    if initial["ready_seconds"] >= VIEW_READY_BUDGET_SECONDS:
        raise AssertionError(
            f"Streams browser view took {initial['ready_seconds']:.3f}s; "
            f"budget is {VIEW_READY_BUDGET_SECONDS:.1f}s"
        )
    if equipment["ready_seconds"] >= VIEW_READY_BUDGET_SECONDS:
        raise AssertionError(
            f"Equipment browser view took {equipment['ready_seconds']:.3f}s; "
            f"budget is {VIEW_READY_BUDGET_SECONDS:.1f}s"
        )

    first_contentful_paint = initial["first_contentful_paint_ms"]
    if not 0.0 <= first_contentful_paint < FIRST_CONTENTFUL_PAINT_BUDGET_MS:
        raise AssertionError(
            f"First contentful paint was {first_contentful_paint:.3f}ms; "
            f"budget is {FIRST_CONTENTFUL_PAINT_BUDGET_MS:.1f}ms"
        )

    for label, view in (("streams", initial), ("equipment", equipment)):
        network_bytes = view["network_bytes"]
        if not 0 < network_bytes < VIEW_NETWORK_BUDGET_BYTES:
            raise AssertionError(
                f"{label} network transfer was {network_bytes} bytes; expected a "
                f"positive value below {VIEW_NETWORK_BUDGET_BYTES} bytes"
            )
        heap_bytes = view["javascript_heap_bytes"]
        if not 0 < heap_bytes < HEAP_BUDGET_BYTES:
            raise AssertionError(
                f"{label} JavaScript heap was {heap_bytes:.0f} bytes; expected a "
                f"positive value below {HEAP_BUDGET_BYTES} bytes"
            )


def run_profile() -> dict[str, object]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(FIXTURE),
        "--server.headless=true",
        f"--server.address={HOST}",
        f"--server.port={PORT}",
        "--browser.gatherUsageStats=false",
    ]
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        _wait_for_health(process)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                args=["--enable-precise-memory-info"],
            )
            context = browser.new_context(viewport={"width": 1440, "height": 1000})
            page = context.new_page()
            session = context.new_cdp_session(page)
            session.send("Network.enable")
            session.send("Performance.enable")
            received_bytes = [0]

            def record_data(params):
                received_bytes[0] += int(params.get("encodedDataLength", 0))

            def record_websocket_frame(params):
                payload = params.get("response", {}).get("payloadData", "")
                received_bytes[0] += len(payload.encode("utf-8"))

            session.on("Network.dataReceived", record_data)
            session.on("Network.webSocketFrameReceived", record_websocket_frame)

            started = time.perf_counter()
            page.goto(BASE_URL, wait_until="domcontentloaded", timeout=30_000)
            page.get_by_text("Solved streams · 2000", exact=True).wait_for(
                timeout=30_000
            )
            page.locator('[data-testid="stDataFrame"]').first.wait_for(
                state="visible",
                timeout=30_000,
            )
            streams_ready = time.perf_counter() - started
            streams_network = received_bytes[0]
            streams_metrics = _chromium_metrics(session, page)

            equipment_started = time.perf_counter()
            page.get_by_text("Equipment & design", exact=True).click()
            page.get_by_text("Solved equipment · 1000", exact=True).wait_for(
                timeout=30_000
            )
            page.get_by_text(
                "Operating versus design basis",
                exact=True,
            ).wait_for(timeout=30_000)
            page.locator('[data-testid="stDataFrame"]').nth(2).wait_for(
                state="visible",
                timeout=30_000,
            )
            equipment_ready = time.perf_counter() - equipment_started
            equipment_network = received_bytes[0] - streams_network
            equipment_metrics = _chromium_metrics(session, page)

            profile = {
                "fixture": "2,000 streams / 1,000 equipment / 8,000 design rows",
                "browser": browser.version,
                "budgets": {
                    "first_contentful_paint_ms": FIRST_CONTENTFUL_PAINT_BUDGET_MS,
                    "view_ready_seconds": VIEW_READY_BUDGET_SECONDS,
                    "view_network_bytes": VIEW_NETWORK_BUDGET_BYTES,
                    "javascript_heap_bytes": HEAP_BUDGET_BYTES,
                },
                "streams_view": {
                    "ready_seconds": streams_ready,
                    "network_bytes": streams_network,
                    **streams_metrics,
                },
                "equipment_view": {
                    "ready_seconds": equipment_ready,
                    "network_bytes": equipment_network,
                    **equipment_metrics,
                },
            }
            browser.close()
            return profile
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def main() -> None:
    profile = run_profile()
    PROFILE_OUTPUT.write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(profile, indent=2, sort_keys=True))
    _assert_profile(profile)


if __name__ == "__main__":
    main()
