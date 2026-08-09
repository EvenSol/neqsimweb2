"""Regression tests for the new Studio shell and Classic-preservation contract."""

from pathlib import Path

import pytest

from studio.navigation import (
    STATUS_AVAILABLE,
    VALID_STATUSES,
    STUDIO_DESTINATIONS,
    destination_by_key,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_destination_keys_and_routes_are_stable_and_unique():
    keys = [destination.key for destination in STUDIO_DESTINATIONS]

    assert len(keys) == len(set(keys))
    assert all(destination.status in VALID_STATUSES for destination in STUDIO_DESTINATIONS)

    for destination in STUDIO_DESTINATIONS:
        if destination.status == STATUS_AVAILABLE:
            assert destination.page
            assert destination.page.startswith("pages/")


def test_existing_flowsheet_studio_is_first_available_studio_workflow():
    destination = destination_by_key("flowsheet")

    assert destination.available
    assert destination.page == "pages/35_Process_Flowsheet_Studio.py"


def test_unknown_destination_fails_loudly():
    with pytest.raises(KeyError, match="Unknown Studio destination"):
        destination_by_key("not-a-real-workflow")


def test_classic_home_and_studio_entry_remain_separate():
    welcome_source = (PROJECT_ROOT / "welcome.py").read_text(encoding="utf-8")
    studio_source = (
        PROJECT_ROOT / "pages" / "00_NeqSim_Studio.py"
    ).read_text(encoding="utf-8")

    # Classic information and its existing sidebar-driven workflow remain in welcome.py.
    assert "### About NeqSim" in welcome_source
    assert "### Getting Started" in welcome_source
    assert "Enable AI Features" in welcome_source

    # The gateway adds Studio without moving the mature flowsheet page or replacing Classic.
    assert 'st.switch_page("pages/00_NeqSim_Studio.py")' in welcome_source
    assert 'st.switch_page("welcome.py")' in studio_source
    assert 'st.switch_page("pages/35_Process_Flowsheet_Studio.py")' in studio_source
