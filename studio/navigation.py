"""Navigation metadata for the NeqSim Studio workspace.

The Studio shell intentionally keeps destinations declarative. Pages can move from
``planned`` to ``available`` without duplicating calculation logic in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


STATUS_AVAILABLE = "available"
STATUS_PLANNED = "planned"
STATUS_CORE_IN_PROGRESS = "core-in-progress"
VALID_STATUSES = {
    STATUS_AVAILABLE,
    STATUS_PLANNED,
    STATUS_CORE_IN_PROGRESS,
}


@dataclass(frozen=True)
class StudioDestination:
    """Describe one top-level Studio engineering workflow."""

    key: str
    title: str
    description: str
    icon: str
    status: str
    page: Optional[str] = None

    @property
    def available(self) -> bool:
        """Return whether the destination is safe to open from Studio."""

        return self.status == STATUS_AVAILABLE and bool(self.page)


STUDIO_DESTINATIONS = (
    StudioDestination(
        key="flowsheet",
        title="Process Flowsheet",
        description=(
            "Build and solve reusable NeqSim process cases with explicit streams, "
            "equipment, subflowsheets, convergence diagnostics and workbooks."
        ),
        icon="⚙️",
        status=STATUS_AVAILABLE,
        page="pages/35_Process_Flowsheet_Studio.py",
    ),
    StudioDestination(
        key="thermodynamics",
        title="Thermodynamics & PVT",
        description=(
            "A unified Studio workflow for fluid characterization, phase behavior "
            "and property studies backed by NeqSim."
        ),
        icon="🧪",
        status=STATUS_PLANNED,
    ),
    StudioDestination(
        key="equipment",
        title="Equipment Design",
        description=(
            "Review solved operating values, design capacities, utilization, margins "
            "and explicit engineering-limit evidence for the active case."
        ),
        icon="📐",
        status=STATUS_AVAILABLE,
        page="pages/10_Studio_Results.py",
    ),
    StudioDestination(
        key="studies",
        title="Engineering Studies",
        description=(
            "Review solved case comparisons and continue sensitivity, adjust/"
            "specification and bounded optimization in the inherited flowsheet tools."
        ),
        icon="📊",
        status=STATUS_AVAILABLE,
        page="pages/10_Studio_Results.py",
    ),
    StudioDestination(
        key="dynamics",
        title="Dynamics & Controls",
        description=(
            "Validated steady-state-to-dynamic handoff, controllers and transient "
            "workflows as NeqSim core capabilities mature."
        ),
        icon="🔄",
        status=STATUS_PLANNED,
    ),
    StudioDestination(
        key="drawings",
        title="Engineering Drawings",
        description=(
            "Professional PFD, P&ID, DEXPI and drawing-register workflows consuming "
            "the canonical NeqSim engineering-diagram model."
        ),
        icon="🗺️",
        status=STATUS_CORE_IN_PROGRESS,
    ),
    StudioDestination(
        key="chat",
        title="Process Chat",
        description=(
            "A case-aware engineering copilot that uses deterministic NeqSim "
            "calculations and evidence as its source of truth."
        ),
        icon="💬",
        status=STATUS_AVAILABLE,
        page="pages/90_Process_Chat.py",
    ),
    StudioDestination(
        key="examples",
        title="Examples & Tutorials",
        description=(
            "Open validated engineering examples and connect Studio workflows to "
            "executable NeqSim-Colab reference cases."
        ),
        icon="📚",
        status=STATUS_PLANNED,
    ),
)


def destination_by_key(key: str) -> StudioDestination:
    """Return one Studio destination by stable key."""

    for destination in STUDIO_DESTINATIONS:
        if destination.key == key:
            return destination
    raise KeyError(f"Unknown Studio destination: {key}")
