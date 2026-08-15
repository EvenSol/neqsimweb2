"""Exercise Classic/Studio accessibility and responsive behavior in Chromium."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

from playwright.sync_api import Locator, Page, sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = PROJECT_ROOT / "welcome.py"
EVIDENCE_OUTPUT = PROJECT_ROOT / "studio-accessibility-browser.json"
HOST = "127.0.0.1"
PORT = 8766
BASE_URL = f"http://{HOST}:{PORT}"
HEALTH_URL = f"{BASE_URL}/_stcore/health"
DESKTOP_VIEWPORT = {"width": 1440, "height": 1000}
MOBILE_VIEWPORT = {"width": 390, "height": 844}
MAX_HORIZONTAL_OVERFLOW_PX = 2.0
MIN_TOUCH_TARGET_HEIGHT_PX = 44.0


def _wait_for_health(
    process: subprocess.Popen[str],
    timeout_seconds: float = 60.0,
) -> None:
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


def _probe_application(process: subprocess.Popen[str]) -> dict[str, object]:
    if process.poll() is not None:
        stdout, _ = process.communicate(timeout=5)
        raise RuntimeError(
            f"Streamlit exited during browser journey (code {process.returncode}):\n"
            f"{stdout}"
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


def _ax_summary(session) -> dict[str, object]:
    nodes = session.send("Accessibility.getFullAXTree").get("nodes", [])
    names_by_role: dict[str, list[str]] = {}
    for node in nodes:
        role = str(node.get("role", {}).get("value", ""))
        name = str(node.get("name", {}).get("value", "")).strip()
        if role and name:
            names_by_role.setdefault(role, []).append(name)

    return {
        "node_count": len(nodes),
        "heading_names": sorted(set(names_by_role.get("heading", []))),
        "button_names": sorted(set(names_by_role.get("button", []))),
    }


def _require_ax_name(
    summary: dict[str, object],
    role_key: str,
    expected_name: str,
) -> None:
    names = summary[role_key]
    if expected_name not in names:
        raise AssertionError(
            f"Accessibility tree is missing {expected_name!r} in {role_key}: {names}"
        )


def _main_layout(page: Page) -> dict[str, object]:
    return page.evaluate(
        """() => {
            const main = document.querySelector('[data-testid="stMain"]')
                || document.querySelector('main')
                || document.documentElement;
            const cards = Array.from(document.querySelectorAll('.workflow-card'))
                .map((element) => {
                    const rect = element.getBoundingClientRect();
                    return {
                        left: rect.left,
                        right: rect.right,
                        width: rect.width,
                    };
                });
            return {
                viewport_width: window.innerWidth,
                client_width: main.clientWidth,
                scroll_width: main.scrollWidth,
                horizontal_overflow: Math.max(0, main.scrollWidth - main.clientWidth),
                workflow_cards: cards,
            };
        }"""
    )


def _assert_layout(layout: dict[str, object], label: str) -> None:
    overflow = float(layout["horizontal_overflow"])
    if overflow > MAX_HORIZONTAL_OVERFLOW_PX:
        raise AssertionError(
            f"{label} main content overflows horizontally by {overflow:.1f}px"
        )

    viewport_width = float(layout["viewport_width"])
    for index, card in enumerate(layout["workflow_cards"]):
        if float(card["width"]) <= 0:
            raise AssertionError(f"{label} workflow card {index} has no width")
        if float(card["left"]) < -MAX_HORIZONTAL_OVERFLOW_PX:
            raise AssertionError(
                f"{label} workflow card {index} starts outside the viewport"
            )
        if float(card["right"]) > viewport_width + MAX_HORIZONTAL_OVERFLOW_PX:
            raise AssertionError(
                f"{label} workflow card {index} ends outside the viewport"
            )


def _touch_target(locator: Locator, name: str) -> dict[str, float]:
    locator.wait_for(state="visible", timeout=30_000)
    box = locator.bounding_box()
    if box is None:
        raise AssertionError(f"{name} has no visible bounding box")
    height = float(box["height"])
    if height < MIN_TOUCH_TARGET_HEIGHT_PX:
        raise AssertionError(
            f"{name} touch target is {height:.1f}px high; "
            f"minimum is {MIN_TOUCH_TARGET_HEIGHT_PX:.1f}px"
        )
    return {
        "x": float(box["x"]),
        "y": float(box["y"]),
        "width": float(box["width"]),
        "height": height,
    }


def _wait_for_classic(page: Page) -> None:
    page.get_by_role(
        "heading",
        name="NeqSim",
        exact=True,
        level=1,
    ).wait_for(state="visible", timeout=30_000)
    page.get_by_role(
        "button",
        name="Open NeqSim Studio",
        exact=True,
    ).wait_for(state="visible", timeout=30_000)


def _open_studio(page: Page) -> None:
    page.get_by_role(
        "button",
        name="Open NeqSim Studio",
        exact=True,
    ).click()
    page.locator("h1#studio-page-title").wait_for(
        state="visible",
        timeout=30_000,
    )
    page.get_by_role(
        "button",
        name="Open Classic",
        exact=True,
    ).wait_for(state="visible", timeout=30_000)


def _assert_planned_actions(page: Page) -> list[str]:
    labels = [
        "Coming soon · Thermodynamics & PVT",
        "Coming soon · Dynamics & Controls",
        "Coming soon · Engineering Drawings",
        "Coming soon · Examples & Tutorials",
    ]
    for label in labels:
        action = page.get_by_role("button", name=label, exact=True)
        action.wait_for(state="visible", timeout=30_000)
        if not action.is_disabled():
            raise AssertionError(f"Planned Studio action is unexpectedly enabled: {label}")
    return labels


def run_browser_journey() -> dict[str, object]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
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
            browser = playwright.chromium.launch(headless=True)
            browser_version = browser.version
            context = browser.new_context(viewport=DESKTOP_VIEWPORT)
            page = context.new_page()
            page_errors: list[str] = []
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            session = context.new_cdp_session(page)

            page.goto(BASE_URL, wait_until="domcontentloaded", timeout=30_000)
            _wait_for_classic(page)
            classic_ax = _ax_summary(session)
            _require_ax_name(classic_ax, "heading_names", "NeqSim")
            _require_ax_name(
                classic_ax,
                "button_names",
                "Open NeqSim Studio",
            )
            classic_desktop_layout = _main_layout(page)
            _assert_layout(classic_desktop_layout, "Classic desktop")

            _open_studio(page)
            studio_ax = _ax_summary(session)
            _require_ax_name(
                studio_ax,
                "heading_names",
                "Engineering simulation, in one workspace.",
            )
            _require_ax_name(studio_ax, "button_names", "Open Classic")
            _require_ax_name(
                studio_ax,
                "button_names",
                "＋ New process case",
            )
            planned_actions = _assert_planned_actions(page)
            if page.locator('[aria-label="Workspace status"]').count() != 1:
                raise AssertionError("Studio workspace status landmark is missing")
            studio_desktop_layout = _main_layout(page)
            _assert_layout(studio_desktop_layout, "Studio desktop")

            page.get_by_role(
                "button",
                name="Open Classic",
                exact=True,
            ).click()
            _wait_for_classic(page)

            page.set_viewport_size(MOBILE_VIEWPORT)
            page.goto(BASE_URL, wait_until="domcontentloaded", timeout=30_000)
            _wait_for_classic(page)
            classic_mobile_layout = _main_layout(page)
            _assert_layout(classic_mobile_layout, "Classic mobile")
            classic_mobile_action = _touch_target(
                page.get_by_role(
                    "button",
                    name="Open NeqSim Studio",
                    exact=True,
                ),
                "Open NeqSim Studio",
            )

            _open_studio(page)
            studio_mobile_layout = _main_layout(page)
            _assert_layout(studio_mobile_layout, "Studio mobile")
            studio_mobile_ax = _ax_summary(session)
            _require_ax_name(
                studio_mobile_ax,
                "heading_names",
                "Engineering simulation, in one workspace.",
            )
            studio_mobile_actions = {
                "new_process_case": _touch_target(
                    page.locator("button").filter(
                        has_text="New process case"
                    ).first,
                    "New process case",
                ),
                "open_classic": _touch_target(
                    page.get_by_role(
                        "button",
                        name="Open Classic",
                        exact=True,
                    ),
                    "Open Classic",
                ),
            }

            if page_errors:
                raise AssertionError(f"Browser page errors: {page_errors}")

            browser.close()

        health_probes.append(_probe_application(process))
        time.sleep(0.25)
        health_probes.append(_probe_application(process))

        return {
            "browser": browser_version,
            "journey": "Classic -> Studio -> Classic; mobile Classic -> Studio",
            "viewports": {
                "desktop": DESKTOP_VIEWPORT,
                "mobile": MOBILE_VIEWPORT,
            },
            "health_probes": health_probes,
            "desktop": {
                "classic_accessibility": classic_ax,
                "studio_accessibility": studio_ax,
                "classic_layout": classic_desktop_layout,
                "studio_layout": studio_desktop_layout,
                "planned_disabled_actions": planned_actions,
            },
            "mobile": {
                "classic_layout": classic_mobile_layout,
                "studio_layout": studio_mobile_layout,
                "classic_action": classic_mobile_action,
                "studio_actions": studio_mobile_actions,
                "studio_accessibility": studio_mobile_ax,
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


def main() -> None:
    evidence = run_browser_journey()
    EVIDENCE_OUTPUT.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
