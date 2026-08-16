"""Exercise independent native solves in two live Studio browser sessions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

from test_studio_full_pilot_browser import (
    BASE_URL,
    ENTRYPOINT,
    HEALTH_URL,
    HOST,
    PORT,
    PROJECT_ROOT,
    VIEWPORT,
    _click_button,
    _download_bytes,
    _probe_application,
    _validate_case_export,
    _validate_workbook_export,
    _wait_for_health,
)


EVIDENCE_OUTPUT = PROJECT_ROOT / "studio-concurrent-solve-browser.json"
SOLVED_MESSAGE = "The NeqSim flowsheet solved and is ready for review."


def _open_new_case(
    browser: Browser,
    case_name: str,
) -> tuple[BrowserContext, Page, list[str]]:
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

    case_input = page.get_by_label("Case name", exact=True)
    case_input.fill(case_name)
    case_input.press("Tab")
    page.wait_for_timeout(1_000)
    if page.get_by_label("Case name", exact=True).input_value() != case_name:
        raise AssertionError(f"Studio did not retain case name {case_name!r}")
    return context, page, page_errors


def _start_solve(page: Page) -> float:
    started = time.monotonic()
    _click_button(page, "▶ Run NeqSim flowsheet", timeout=60_000)
    return started


def _wait_for_solve(page: Page, case_name: str) -> float:
    started = time.monotonic()
    page.get_by_text(SOLVED_MESSAGE, exact=True).wait_for(
        state="visible",
        timeout=180_000,
    )
    if page.get_by_label("Case name", exact=True).input_value() != case_name:
        raise AssertionError(
            f"Solved session changed case identity for {case_name!r}"
        )
    return time.monotonic() - started


def _export_session(page: Page) -> dict[str, object]:
    case_filename, case_payload = _download_bytes(page, "Download case JSON")
    workbook_filename, workbook_payload = _download_bytes(
        page,
        "Download engineering workbook",
    )
    return {
        "case": _validate_case_export(case_filename, case_payload),
        "workbook": _validate_workbook_export(
            workbook_filename,
            workbook_payload,
        ),
    }


def _open_process_chat(page: Page, expected_case_name: str) -> str:
    _click_button(page, "← Studio home")
    page.get_by_role(
        "heading",
        name="Engineering simulation, in one workspace.",
        exact=True,
        level=1,
    ).wait_for(state="visible", timeout=60_000)
    _click_button(page, "Open Process Chat")
    page.get_by_role(
        "heading",
        name="Process Chat",
        exact=False,
        level=1,
    ).wait_for(state="visible", timeout=60_000)
    banner = page.get_by_text("Studio case:", exact=False).first
    banner.wait_for(state="visible", timeout=30_000)
    banner_text = " ".join(banner.inner_text().split())
    if expected_case_name not in banner_text or "Solved" not in banner_text:
        raise AssertionError(
            "Process Chat received the wrong solved Studio case: "
            f"{banner_text!r}"
        )
    return banner_text


def run_concurrent_solve_gate() -> dict[str, object]:
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

    contexts: list[BrowserContext] = []
    try:
        _wait_for_health(process)
        health_probes = [_probe_application(process)]
        time.sleep(0.25)
        health_probes.append(_probe_application(process))

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                channel="chromium",
                args=["--disable-dev-shm-usage", "--disable-gpu"],
            )
            browser_version = browser.version
            first_name = "Concurrent native solve A"
            second_name = "Concurrent native solve B"
            first_context, first_page, first_errors = _open_new_case(
                browser,
                first_name,
            )
            second_context, second_page, second_errors = _open_new_case(
                browser,
                second_name,
            )
            contexts.extend((first_context, second_context))

            first_started = _start_solve(first_page)
            first_solved_before_second_dispatch = first_page.get_by_text(
                SOLVED_MESSAGE,
                exact=True,
            ).is_visible()
            second_started = _start_solve(second_page)
            dispatch_gap_seconds = second_started - first_started
            if first_solved_before_second_dispatch:
                raise AssertionError(
                    "The first solve completed before the peer request was "
                    "dispatched; concurrent-session coverage was not achieved."
                )
            if dispatch_gap_seconds > 10.0:
                raise AssertionError(
                    "Concurrent solve requests were dispatched too far apart: "
                    f"{dispatch_gap_seconds:.3f} s"
                )

            first_wait_seconds = _wait_for_solve(first_page, first_name)
            second_wait_seconds = _wait_for_solve(second_page, second_name)
            first_export = _export_session(first_page)
            second_export = _export_session(second_page)
            if first_export["case"]["case_name"] != first_name:
                raise AssertionError("First session exported the wrong case")
            if second_export["case"]["case_name"] != second_name:
                raise AssertionError("Second session exported the wrong case")
            if first_export["case"]["case_name"] == second_export["case"]["case_name"]:
                raise AssertionError("Concurrent sessions shared one case identity")

            first_banner = _open_process_chat(first_page, first_name)
            second_banner = _open_process_chat(second_page, second_name)

            _click_button(first_page, "← Studio home")
            first_page.get_by_role(
                "heading",
                name="Engineering simulation, in one workspace.",
                exact=True,
                level=1,
            ).wait_for(state="visible", timeout=60_000)
            _click_button(first_page, "← NeqSim Classic")
            first_page.get_by_role(
                "heading",
                name="NeqSim",
                exact=True,
                level=1,
            ).wait_for(state="visible", timeout=30_000)

            second_banner_after_peer_classic = second_page.get_by_text(
                "Studio case:",
                exact=False,
            ).first
            second_banner_after_peer_classic.wait_for(
                state="visible",
                timeout=30_000,
            )
            peer_banner_text = " ".join(
                second_banner_after_peer_classic.inner_text().split()
            )
            if second_name not in peer_banner_text or "Solved" not in peer_banner_text:
                raise AssertionError(
                    "Peer Classic navigation changed the second solved session: "
                    f"{peer_banner_text!r}"
                )

            page_errors = first_errors + second_errors
            if page_errors:
                raise AssertionError(f"Browser page errors: {page_errors}")
            browser.close()
            contexts.clear()

        health_probes.append(_probe_application(process))
        time.sleep(0.25)
        health_probes.append(_probe_application(process))
        return {
            "browser": browser_version,
            "sessions": {
                "first": {
                    "case_name": first_name,
                    "wait_seconds": first_wait_seconds,
                    "exports": first_export,
                    "process_chat_banner": first_banner,
                },
                "second": {
                    "case_name": second_name,
                    "wait_seconds": second_wait_seconds,
                    "exports": second_export,
                    "process_chat_banner": second_banner,
                    "banner_after_peer_classic": peer_banner_text,
                },
            },
            "dispatch_gap_seconds": dispatch_gap_seconds,
            "first_solved_before_second_dispatch": False,
            "distinct_case_names": True,
            "independent_process_chat_handoffs": True,
            "peer_classic_navigation_isolated": True,
            "health_probes": health_probes,
            "page_errors": page_errors,
            "execution_boundary": (
                "simultaneously live Streamlit sessions; this does not claim "
                "process-isolated JVM execution"
            ),
        }
    finally:
        for context in contexts:
            try:
                context.close()
            except Exception:
                pass
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
        evidence = run_concurrent_solve_gate()
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
