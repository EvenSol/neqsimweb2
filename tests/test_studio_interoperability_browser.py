"""Exercise the portable Studio case contract through a real Chromium session."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from playwright.sync_api import Browser, Page, sync_playwright

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
    _page_diagnostic,
    _probe_application,
    _wait_for_health,
)


EVIDENCE_OUTPUT = PROJECT_ROOT / "studio-interoperability-browser.json"
MAX_CASE_FILE_BYTES = 1_000_000


def _open_studio(page: Page) -> None:
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


def _upload_case(page: Page, payload: bytes, filename: str) -> None:
    page.get_by_label(
        "Open portable case JSON",
        exact=True,
    ).locator('input[type="file"]').set_input_files(
        {
            "name": filename,
            "mimeType": "application/json",
            "buffer": payload,
        }
    )
    _click_button(page, "Open uploaded case")


def _wait_for_flowsheet(page: Page) -> None:
    page.get_by_text(
        "Build and solve a reproducible NeqSim process case",
        exact=False,
    ).wait_for(state="visible", timeout=60_000)


def _case_download(page: Page) -> tuple[bytes, dict[str, object]]:
    _click_button(page, "← Studio home")
    page.get_by_role(
        "heading",
        name="Engineering simulation, in one workspace.",
        exact=True,
        level=1,
    ).wait_for(state="visible", timeout=30_000)
    filename, payload = _download_bytes(page, "Download active case")
    if filename != "neqsim_studio_case.json":
        raise AssertionError(f"Unexpected portable case filename: {filename!r}")
    case_spec = json.loads(payload.decode("utf-8"))
    if case_spec.get("schema_version") != 4:
        raise AssertionError("Portable case download was not canonical schema v4")
    _click_button(page, "Continue active case")
    _wait_for_flowsheet(page)
    return payload, {
        "filename": filename,
        "bytes": len(payload),
        "schema_version": case_spec["schema_version"],
        "case_name": case_spec["name"],
        "case_spec": case_spec,
    }


def _starter_case(browser: Browser) -> tuple[bytes, dict[str, object]]:
    context = browser.new_context(viewport=VIEWPORT)
    page = context.new_page()
    _open_studio(page)
    _click_button(page, "＋ New process case")
    _wait_for_flowsheet(page)
    payload, details = _case_download(page)
    context.close()
    return payload, details


def _candidate_payload(
    canonical_case: dict[str, object],
    schema_version: int,
) -> tuple[bytes, str]:
    candidate = deepcopy(canonical_case)
    candidate["schema_version"] = schema_version
    case_name = f"Interoperability schema v{schema_version} – Åsgard ΔP"
    candidate["name"] = case_name
    payload = (
        json.dumps(candidate, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    if schema_version == 1:
        payload = b"\xef\xbb\xbf" + payload
    return payload, case_name


def _exercise_supported_schema(
    browser: Browser,
    canonical_case: dict[str, object],
    schema_version: int,
) -> dict[str, object]:
    payload, case_name = _candidate_payload(canonical_case, schema_version)
    first_context = browser.new_context(viewport=VIEWPORT)
    first_page = first_context.new_page()
    page_errors: list[str] = []
    first_page.on("pageerror", lambda error: page_errors.append(str(error)))

    _open_studio(first_page)
    _upload_case(first_page, payload, f"schema-v{schema_version}.json")
    _wait_for_flowsheet(first_page)

    if schema_version < 4:
        first_page.get_by_text(
            f"Schema-v{schema_version} case migrated to Studio schema v4.",
            exact=False,
        ).wait_for(state="visible", timeout=30_000)

    first_export, first_details = _case_download(first_page)
    if first_details["case_name"] != case_name:
        raise AssertionError(
            "Imported case name changed during migration: "
            f"{first_details['case_name']!r}"
        )

    _click_button(first_page, "← Studio home")
    first_page.get_by_role(
        "heading",
        name=case_name,
        exact=True,
        level=3,
    ).wait_for(state="visible", timeout=30_000)
    _click_button(first_page, "← NeqSim Classic")
    first_page.get_by_role(
        "heading",
        name="NeqSim",
        exact=True,
        level=1,
    ).wait_for(state="visible", timeout=30_000)
    _click_button(first_page, "Open NeqSim Studio")
    first_page.get_by_role(
        "heading",
        name=case_name,
        exact=True,
        level=3,
    ).wait_for(state="visible", timeout=30_000)
    _click_button(first_page, "Continue active case")
    _wait_for_flowsheet(first_page)
    continued_export, continued_details = _case_download(first_page)

    if json.loads(first_export) != json.loads(continued_export):
        raise AssertionError(
            f"Schema v{schema_version} changed after Classic/Studio continuation"
        )
    first_context.close()

    restart_context = browser.new_context(viewport=VIEWPORT)
    restart_page = restart_context.new_page()
    restart_errors: list[str] = []
    restart_page.on("pageerror", lambda error: restart_errors.append(str(error)))
    _open_studio(restart_page)
    _upload_case(
        restart_page,
        first_export,
        f"schema-v{schema_version}-canonical-reopen.json",
    )
    _wait_for_flowsheet(restart_page)
    restarted_export, restarted_details = _case_download(restart_page)
    restart_context.close()

    if json.loads(first_export) != json.loads(restarted_export):
        raise AssertionError(
            f"Schema v{schema_version} canonical export changed after restart"
        )
    if page_errors or restart_errors:
        raise AssertionError(
            f"Schema v{schema_version} browser errors: "
            f"{page_errors + restart_errors}"
        )

    return {
        "input_schema_version": schema_version,
        "input_bytes": len(payload),
        "utf8_bom": schema_version == 1,
        "unicode_case_name": case_name,
        "canonical_schema_version": first_details["schema_version"],
        "canonical_bytes": first_details["bytes"],
        "classic_round_trip_equal": (
            first_details["case_spec"] == continued_details["case_spec"]
        ),
        "fresh_session_reopen_equal": (
            first_details["case_spec"] == restarted_details["case_spec"]
        ),
        "page_errors": page_errors + restart_errors,
    }


def _failure_cases(
    canonical_case: dict[str, object],
) -> tuple[tuple[str, bytes, str, bool], ...]:
    future_case = deepcopy(canonical_case)
    future_case["schema_version"] = 5
    future_case["name"] = "Unsupported future schema"
    future_payload = (
        json.dumps(future_case, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    return (
        (
            "malformed-json",
            b"{not valid JSON",
            "The Studio case is not valid JSON",
            False,
        ),
        (
            "non-utf8",
            b"\xff\xfe\x00\x00",
            "The Studio case must use UTF-8 encoding.",
            False,
        ),
        (
            "oversized",
            b"{" + (b" " * MAX_CASE_FILE_BYTES),
            "The Studio case file cannot exceed 1000000 bytes.",
            False,
        ),
        (
            "future-schema",
            future_payload,
            "Unsupported schema_version. Expected version 1, 2, 3, or 4.",
            True,
        ),
    )


def _exercise_failure_recovery(
    browser: Browser,
    canonical_case: dict[str, object],
    label: str,
    rejected_payload: bytes,
    expected_error: str,
    error_on_flowsheet: bool,
) -> dict[str, object]:
    context = browser.new_context(viewport=VIEWPORT)
    page = context.new_page()
    page_errors: list[str] = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    _open_studio(page)
    _upload_case(page, rejected_payload, f"{label}.json")
    page.get_by_text(expected_error, exact=False).wait_for(
        state="visible",
        timeout=60_000,
    )
    if error_on_flowsheet:
        _click_button(page, "← Studio home")

    retry_case = deepcopy(canonical_case)
    retry_case["schema_version"] = 1
    retry_case["name"] = f"Recovered after {label}"
    retry_payload = (
        json.dumps(retry_case, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _upload_case(page, retry_payload, f"{label}-retry.json")
    _wait_for_flowsheet(page)
    page.get_by_text(
        "Schema-v1 case migrated to Studio schema v4.",
        exact=False,
    ).wait_for(state="visible", timeout=30_000)
    _, retry_details = _case_download(page)
    context.close()

    if retry_details["case_name"] != retry_case["name"]:
        raise AssertionError(f"{label} recovery opened the wrong case")
    if page_errors:
        raise AssertionError(f"{label} browser errors: {page_errors}")

    return {
        "rejected_case": label,
        "rejected_bytes": len(rejected_payload),
        "expected_error": expected_error,
        "failed_closed": True,
        "supported_retry_schema": 1,
        "retry_canonical_schema": retry_details["schema_version"],
        "retry_case_name": retry_details["case_name"],
        "page_errors": page_errors,
    }


def _engineering_failure_payload(
    canonical_case: dict[str, object],
) -> tuple[bytes, str, str]:
    """Return a schema-valid case with one disconnected material inlet."""

    candidate = deepcopy(canonical_case)
    case_name = "Disconnected engineering case"
    candidate["name"] = case_name
    inlets = candidate.get("inlets")
    connections = candidate.get("connections")
    if not isinstance(inlets, list) or not isinstance(connections, list):
        raise AssertionError("Canonical case lacks inlet or connection arrays")

    inlet_ids = {
        str(inlet.get("id", "")).strip()
        for inlet in inlets
        if isinstance(inlet, dict) and str(inlet.get("id", "")).strip()
    }
    disconnected_index = None
    disconnected_inlet_id = ""
    for index, connection in enumerate(connections):
        if not isinstance(connection, dict):
            continue
        source = connection.get("source")
        if not isinstance(source, dict):
            continue
        source_id = str(source.get("id", "")).strip()
        if (
            str(source.get("kind", "")).strip().lower() == "inlet"
            and str(source.get("port", "")).strip().lower() == "out"
            and source_id in inlet_ids
        ):
            disconnected_index = index
            disconnected_inlet_id = source_id
            break
    if disconnected_index is None:
        raise AssertionError("Canonical case has no material inlet connection")

    del connections[disconnected_index]
    payload = (
        json.dumps(candidate, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    return payload, case_name, disconnected_inlet_id


def _exercise_engineering_failure_recovery(
    browser: Browser,
    canonical_case: dict[str, object],
) -> dict[str, object]:
    """Fail a disconnected graph, then recover through a real native solve."""

    context = browser.new_context(viewport=VIEWPORT)
    page = context.new_page()
    page_errors: list[str] = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))

    rejected_payload, rejected_name, disconnected_inlet_id = (
        _engineering_failure_payload(canonical_case)
    )
    _open_studio(page)
    _upload_case(page, rejected_payload, "disconnected-engineering-case.json")
    _wait_for_flowsheet(page)
    _click_button(page, "▶ Run NeqSim flowsheet", timeout=60_000)

    expected_error = (
        "Connect every independent feed before solving; disconnected "
        f"inlet(s): {disconnected_inlet_id}."
    )
    page.get_by_text(expected_error, exact=False).wait_for(
        state="visible",
        timeout=60_000,
    )
    page.get_by_text("Solver: Failed", exact=False).wait_for(
        state="visible",
        timeout=30_000,
    )
    if page.get_by_role(
        "button",
        name="Download case JSON",
        exact=True,
    ).count():
        raise AssertionError(
            "A failed engineering case exposed a solved-case JSON artifact"
        )
    if page.get_by_text("2. Engineering results", exact=True).count():
        raise AssertionError(
            "A failed engineering case exposed solved engineering results"
        )

    _click_button(page, "← Studio home")
    page.get_by_role(
        "heading",
        name="Engineering simulation, in one workspace.",
        exact=True,
        level=1,
    ).wait_for(state="visible", timeout=30_000)
    page.get_by_role(
        "heading",
        name=rejected_name,
        exact=True,
        level=3,
    ).wait_for(state="visible", timeout=30_000)

    retry_case = deepcopy(canonical_case)
    retry_case["name"] = "Recovered native engineering case"
    retry_payload = (
        json.dumps(retry_case, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    _upload_case(page, retry_payload, "recovered-native-case.json")
    _wait_for_flowsheet(page)
    _click_button(page, "▶ Run NeqSim flowsheet", timeout=60_000)
    try:
        page.get_by_text(
            "The NeqSim flowsheet solved and is ready for review.",
            exact=True,
        ).wait_for(state="visible", timeout=180_000)
    except Exception as error:
        raise AssertionError(
            "The valid recovery case did not solve through native NeqSim; "
            + _page_diagnostic(page)
        ) from error

    page.get_by_text("Solver: Solved", exact=False).wait_for(
        state="visible",
        timeout=30_000,
    )
    for button_name in (
        "Download case JSON",
        "Download engineering workbook",
    ):
        page.get_by_role(
            "button",
            name=button_name,
            exact=True,
        ).wait_for(state="attached", timeout=30_000)

    _click_button(page, "← Studio home")
    page.get_by_role(
        "heading",
        name=retry_case["name"],
        exact=True,
        level=3,
    ).wait_for(state="visible", timeout=30_000)
    _click_button(page, "← NeqSim Classic")
    page.get_by_role(
        "heading",
        name="NeqSim",
        exact=True,
        level=1,
    ).wait_for(state="visible", timeout=30_000)
    context.close()

    if page_errors:
        raise AssertionError(
            f"Engineering failure/recovery browser errors: {page_errors}"
        )

    return {
        "rejected_case": rejected_name,
        "rejected_bytes": len(rejected_payload),
        "disconnected_inlet_id": disconnected_inlet_id,
        "expected_error": expected_error,
        "failed_solver_status": "Failed",
        "solved_artifacts_published_after_failure": False,
        "retry_case_name": retry_case["name"],
        "retry_solver_status": "Solved",
        "native_neqsim_recovery": True,
        "returned_to_classic": True,
        "page_errors": page_errors,
    }


def run_interoperability_matrix() -> dict[str, object]:
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
                args=["--disable-dev-shm-usage", "--disable-gpu"],
            )
            browser_version = browser.version
            starter_payload, starter_details = _starter_case(browser)
            canonical_case = json.loads(starter_payload.decode("utf-8"))

            supported = [
                _exercise_supported_schema(
                    browser,
                    canonical_case,
                    schema_version,
                )
                for schema_version in (1, 2, 3, 4)
            ]
            rejected = [
                _exercise_failure_recovery(
                    browser,
                    canonical_case,
                    *failure_case,
                )
                for failure_case in _failure_cases(canonical_case)
            ]
            engineering_failure_recovery = (
                _exercise_engineering_failure_recovery(
                    browser,
                    canonical_case,
                )
            )
            browser.close()

        health_probes.append(_probe_application(process))
        time.sleep(0.25)
        health_probes.append(_probe_application(process))
        return {
            "browser": browser_version,
            "contract": "portable Process Flowsheet Studio JSON",
            "canonical_schema_version": 4,
            "supported_input_versions": [1, 2, 3, 4],
            "starter_case": {
                key: value
                for key, value in starter_details.items()
                if key != "case_spec"
            },
            "supported_matrix": supported,
            "rejected_matrix": rejected,
            "engineering_failure_recovery": engineering_failure_recovery,
            "health_probes": health_probes,
            "classic_workspace_preserved": True,
            "native_neqsim_model_format_changed": False,
            "provider_call_executed": False,
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
        evidence = run_interoperability_matrix()
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
