"""Navigation policy for the NeqSim Streamlit application."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent
PAGES_DIRECTORY = APP_ROOT / "pages"

# This order is intentional: normal mode should present only the requested
# stable tools, independent of the numeric ordering of files in pages/.
STABLE_PAGE_PATHS = (
    "pages/0_TP_flash.py",
    "pages/20_Phase_envelope.py",
    "pages/10_Gas_Hydrate.py",
    "pages/60_Hydrogen.py",
    "pages/6_EOS_CG.py",
)

TITLE_OVERRIDES = {
    "pages/00_NeqSim_Studio.py": "NeqSim Studio",
    "pages/0_TP_flash.py": "TP Flash",
    "pages/5_GERG2008.py": "GERG-2008",
    "pages/6_EOS_CG.py": "EOS-CG",
    "pages/10_Gas_Hydrate.py": "Gas Hydrate",
    "pages/10_Studio_Results.py": "Studio Results",
    "pages/15_PVT_Simulations.py": "PVT Simulations",
    "pages/20_Phase_envelope.py": "Phase Envelope",
    "pages/35_Process_Flowsheet_Studio.py": "Process Flowsheet Studio",
    "pages/80_Emission_Calculator.py": "Emission Calculator",
    "pages/81_TEG_Dehydration_Emissions.py": "TEG Dehydration Emissions",
    "pages/100_CO2_mechanisticModel.py": "CO₂ Mechanistic Model",
}

@dataclass(frozen=True)
class PageSpec:
    """A page path and its user-facing navigation title."""

    path: str
    title: str


def _page_sort_key(page_path: str) -> tuple[int, str]:
    """Sort pages by an optional numeric filename prefix, then by name."""
    filename = Path(page_path).name
    match = re.match(r"^(\d+)_", filename)
    order = int(match.group(1)) if match else 10_000
    return order, filename.casefold()


def page_title(page_path: str) -> str:
    """Return a readable title while preserving domain abbreviations."""
    if page_path in TITLE_OVERRIDES:
        return TITLE_OVERRIDES[page_path]

    stem = Path(page_path).stem
    stem = re.sub(r"^\d+_", "", stem)
    return stem.replace("_", " ")


def discover_page_paths() -> tuple[str, ...]:
    """Discover every Python page in deterministic menu order."""
    return tuple(
        sorted(
            (
                path.relative_to(APP_ROOT).as_posix()
                for path in PAGES_DIRECTORY.glob("*.py")
                if path.is_file()
            ),
            key=_page_sort_key,
        )
    )


def stable_page_specs() -> tuple[PageSpec, ...]:
    """Return the pages exposed when experimental mode is disabled."""
    missing = [path for path in STABLE_PAGE_PATHS if not (APP_ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"Stable Streamlit pages are missing: {', '.join(missing)}")
    return tuple(PageSpec(path, page_title(path)) for path in STABLE_PAGE_PATHS)


def experimental_page_specs() -> tuple[PageSpec, ...]:
    """Return pages added to the menu when experimental mode is enabled."""
    return tuple(
        PageSpec(path, page_title(path))
        for path in discover_page_paths()
        if path not in STABLE_PAGE_PATHS
    )
