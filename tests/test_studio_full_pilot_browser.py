"""Run the full Classic-to-Studio solve, export, and Process Chat pilot."""

from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
import zipfile

from playwright.sync_api import Page, sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = PROJECT_ROOT / "welcome.py"
EVIDENCE_OUTPUT = PROJECT_ROOT / "studio-full-pilot-browser.json"
HOST = "127.0.0.1"
PORT = 8767
BASE_URL = f"http://{HOST}:{PORT}"
HEALTH_URL = f"{BASE_URL}/_stcore/health"
VIEWPORT = {"width": 1440, "height": 1000}


def _wait_for_health(
    process: subprocess.Popen[str],
    timeout_seconds: float = 90.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "no response"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, _ = process.communicate(timeout=5)
            raise RuntimeError(
                "Streamlit exited before becoming healthy "
                f"(code {process.returncode}):\n{stdout}"
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


def _probe_application(process: subprocess.Popen[str]) -> dict[str, object]:
    if process.poll() is not None:
        stdout, _ = process.communicate(timeout=5)
        raise RuntimeError(
            "Streamlit exited during the pilot "
            f"(code {process.returncode}):\n{stdout}"
        )

    with urlopen(BASE_URL, timeout=5) as response:
        root_status = response.status
        response.read()
    with urlopen(HEALTH_URL, timeout=5) as response:
        health_status = response.status
        health_body = response.read().decode("utf-8").strip()

    if root_status != 200:
        raise AssertionError(f"Classic root returned HTTP {root_status}")
    if health_status != 200 or health_body != "ok":
        raise AssertionError(
            f"Streamlit health returned HTTP {health_status}: {health_body!r}"
        )
    return {
        "root_status": root_status,
        "health_status": health_status,
        "health_body": health_body,
        "process_live": process.poll() is None,
    }


def _page_diagnostic(page: Page) -> str:
    try:
        url = page.url
        body_text = page.locator("body").inner_text(timeout=5_000)
    except Exception as diagnostic_error:
        return (
            "browser diagnostics unavailable after page failure: "
            f"{type(diagnostic_error).__name__}: {diagnostic_error}"
        )
    compact_text = " ".join(body_text.split())
    return f"url={url!r}; visible_text={compact_text[:1600]!r}"


def _click_button(page: Page, name: str, timeout: int = 30_000) -> None:
    last_error: Exception | None = None
    for _ in range(3):
        action = page.get_by_role("button", name=name, exact=True)
        try:
            action.wait_for(state="visible", timeout=timeout)
            action.scroll_into_view_if_needed()
            page.wait_for_timeout(750)
            action.click()
            return
        except Exception as error:
            last_error = error
            page.wait_for_timeout(750)
    raise AssertionError(
        f"{name} remained unavailable across Streamlit reruns: {last_error}"
    )


def _download_bytes(page: Page, button_name: str) -> tuple[str, bytes]:
    last_error: Exception | None = None
    for _ in range(3):
        action = page.get_by_role("button", name=button_name, exact=True)
        try:
            action.wait_for(state="visible", timeout=30_000)
            action.scroll_into_view_if_needed()
            with page.expect_download(timeout=30_000) as download_info:
                action.click()
            download = download_info.value
            downloaded_path = download.path()
            if downloaded_path is None:
                raise AssertionError(
                    f"{button_name} did not produce a local download"
                )
            return (
                download.suggested_filename,
                Path(downloaded_path).read_bytes(),
            )
        except Exception as error:
            last_error = error
            page.wait_for_timeout(750)
    raise AssertionError(
        f"{button_name} remained unavailable across Streamlit reruns: {last_error}"
    )


def _validate_case_export(filename: str, payload: bytes) -> dict[str, object]:
    if filename != "process_flowsheet_case.json":
        raise AssertionError(f"Unexpected portable case filename: {filename!r}")
    case_spec = json.loads(payload.decode("utf-8"))
    if case_spec.get("schema_version") != 4:
        raise AssertionError(
            "Solved browser export did not use canonical schema version 4"
        )
    required = {"name", "units", "connections", "process", "inlets"}
    missing = sorted(required.difference(case_spec))
    if missing:
        raise AssertionError(f"Solved browser export is missing keys: {missing}")
    return {
        "filename": filename,
        "bytes": len(payload),
        "schema_version": case_spec["schema_version"],
        "case_name": case_spec["name"],
        "unit_count": len(case_spec["units"]),
        "connection_count": len(case_spec["connections"]),
    }


def _validate_workbook_export(filename: str, payload: bytes) -> dict[str, object]:
    if filename != "process_flowsheet_engineering_workbook.xlsx":
        raise AssertionError(f"Unexpected workbook filename: {filename!r}")
    with zipfile.ZipFile(BytesIO(payload)) as workbook:
        names = set(workbook.namelist())
        required = {"[Content_Types].xml", "xl/workbook.xml"}
        missing = sorted(required.difference(names))
        if missing:
            raise AssertionError(f"Engineering workbook is malformed: {missing}")
        workbook_xml = workbook.read("xl/workbook.xml").decode(
            "utf-8",
            errors="replace",
        )
        if "<sheet " not in workbook_xml:
            raise AssertionError("Engineering workbook contains no worksheets")
        sheet_count = workbook_xml.count("<sheet ")
    return {
        "filename": filename,
        "bytes": len(payload),
        "worksheet_count": sheet_count,
    }


def run_browser_pilot() -> dict[str, object]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    environment["MALLOC_ARENA_MAX"] = "2"
    environment["JAVA_TOOL_OPTIONS"] = (
        "-Xms256m -Xmx2048m "
        "--add-opens=java.base/java.util=ALL-UNNAMED "
        "--add-opens=java.base/java.lang=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
        "--add-opens=java.base/java.io=ALL-UNNAMED"
    )
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ENTRYPOINT),
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
        health_probes = [_probe_application(process)]
        time.sleep(0.25)
        health_probes.append(_probe_application(process))

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                channel="chromium",
                args=[
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                ],
            )
            browser_version = browser.version
            context = browser.new_context(viewport=VIEWPORT)
            page = context.new_page()
            page_errors: list[str] = []
            page.on("pageerror", lambda error: page_errors.append(str(error)))

            page.goto(BASE_URL, wait_until="domcontentloaded", timeout=30_000)
            page.get_by_role(
                "heading",
                name="NeqSim",
                exact=True,
                level=1,
            ).wait_for(state="visible", timeout=30_000)
            _click_button(page, "Open NeqSim Studio")
            page.get_by_role(
                "heading",
                name="Engineering simulation, in one workspace.",
                exact=True,
                level=1,
            ).wait_for(state="visible", timeout=30_000)

            _click_button(page, "＋ New process case")
            page.get_by_text(
                "Build and solve a reproducible NeqSim process case",
                exact=False,
            ).wait_for(state="visible", timeout=60_000)

            _click_button(page, "▶ Run NeqSim flowsheet", timeout=60_000)
            try:
                page.get_by_text(
                    "The NeqSim flowsheet solved and is ready for review.",
                    exact=True,
                ).wait_for(state="visible", timeout=180_000)
            except Exception as error:
                raise AssertionError(
                    "The real starter case did not solve through the browser; "
                    + _page_diagnostic(page)
                ) from error

            case_filename, case_payload = _download_bytes(
                page,
                "Download case JSON",
            )
            workbook_filename, workbook_payload = _download_bytes(
                page,
                "Download engineering workbook",
            )
            case_export = _validate_case_export(case_filename, case_payload)
            workbook_export = _validate_workbook_export(
                workbook_filename,
                workbook_payload,
            )

            _click_button(page, "← Studio home")
            page.get_by_role(
                "heading",
                name="Engineering simulation, in one workspace.",
                exact=True,
                level=1,
            ).wait_for(state="visible", timeout=60_000)
            _click_button(page, "Open Process Chat")
            try:
                page.get_by_role(
                    "heading",
                    name="Process Chat",
                    exact=False,
                    level=1,
                ).wait_for(state="visible", timeout=60_000)
            except Exception as error:
                raise AssertionError(
                    "Process Chat did not load through its Studio action; "
                    + _page_diagnostic(page)
                ) from error

            case_banner = page.get_by_text("Studio case:", exact=False).first
            case_banner.wait_for(state="visible", timeout=30_000)
            banner_text = " ".join(case_banner.inner_text().split())
            if "Solved" not in banner_text:
                raise AssertionError(
                    "Process Chat did not receive solved Studio case state: "
                    f"{banner_text!r}"
                )

            overview = page.locator("details").filter(
                has_text="Process Model Overview"
            )
            overview.wait_for(state="visible", timeout=30_000)
            overview.locator("summary").click()
            page.get_by_role(
                "heading",
                name="Unit Operations",
                exact=True,
            ).wait_for(state="visible", timeout=30_000)
            page.get_by_role(
                "heading",
                name="Streams",
                exact=True,
            ).wait_for(state="visible", timeout=30_000)
            page.get_by_placeholder(
                "Ask about your process model..."
            ).wait_for(state="visible", timeout=30_000)

            _click_button(page, "Process flowsheet")
            page.get_by_text(
                "Build and solve a reproducible NeqSim process case",
                exact=False,
            ).wait_for(state="visible", timeout=60_000)
            page.get_by_text("Solver: Solved", exact=False).wait_for(
                state="visible",
                timeout=30_000,
            )

            if page_errors:
                raise AssertionError(f"Browser page errors: {page_errors}")
            browser.close()

        health_probes.append(_probe_application(process))
        time.sleep(0.25)
        health_probes.append(_probe_application(process))

        return {
            "browser": browser_version,
            "journey": (
                "Classic -> Studio -> starter solve -> portable JSON/workbook "
                "exports -> Process Chat solved-model handoff -> flowsheet"
            ),
            "viewport": VIEWPORT,
            "health_probes": health_probes,
            "case_export": case_export,
            "workbook_export": workbook_export,
            "process_chat": {
                "case_banner": banner_text,
                "live_model_overview": True,
                "bounded_chat_input_available": True,
                "provider_call_executed": False,
                "returned_to_solved_flowsheet": True,
            },
            "page_errors": page_errors,
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _write_evidence(evidence: dict[str, object]) -> None:
    EVIDENCE_OUTPUT.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))


def main() -> None:
    try:
        evidence = run_browser_pilot()
    except Exception as error:
        _write_evidence(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        raise

    evidence["status"] = "passed"
    _write_evidence(evidence)


if __name__ == "__main__":
    main()
