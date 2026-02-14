"""
Patch schema — structured model-edit definitions.

The LLM outputs ONLY these objects (or their JSON equivalent).
Your code applies them deterministically to the NeqSim process model.

Two levels:
  1. InputPatch  — safe MVP: changes only ProcessInput fields (pressures, temps, flows, dP)
  2. PatchOp     — advanced: arbitrary set/scale/add_unit/connect operations (MVP-2+)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


# ---------------------------------------------------------------------------
# Level 1 — Input-only patches (safe, MVP)
# ---------------------------------------------------------------------------

@dataclass
class AddUnitOp:
    """
    Insert a new equipment unit into the process topology.
    
    Example:
        AddUnitOp(
            name="new intercooler",
            equipment_type="cooler",
            insert_after="1st stage compressor",
            params={"outlet_temperature_C": 35.0, "pressure_drop_bar": 0.2}
        )
    """
    name: str                              # name for the new unit
    equipment_type: str                    # "cooler", "heater", "compressor", "separator", "valve", "expander"
    insert_after: str                      # name of existing unit to insert after
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InputPatch:
    """
    Change one or more ProcessInput fields, and optionally add new units.
    
    Supports absolute values and relative operations:
        Absolute:  {"Psep1": 25.0}
        Relative:  {"Psep1": {"op": "add", "value": 5.0}}
        Scale:     {"feed_rate": {"op": "scale", "value": 1.1}}
    
    Supports adding new equipment via add_units list.
    """
    changes: Dict[str, Any]
    add_units: List[AddUnitOp] = field(default_factory=list)
    note: Optional[str] = None

    def apply_to(self, base_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge this patch into a base input dict, returning a new dict.
        Supports absolute values, add, and scale operations.
        """
        merged = copy.deepcopy(base_input)
        for key, val in self.changes.items():
            if key not in merged:
                raise KeyError(f"Unknown ProcessInput field: {key}")
            if isinstance(val, dict) and "op" in val:
                op = val["op"]
                v = val["value"]
                current = merged[key]
                if op == "add":
                    merged[key] = current + v
                elif op == "scale":
                    merged[key] = current * v
                else:
                    raise ValueError(f"Unknown relative op: {op}")
            else:
                merged[key] = val
        return merged


@dataclass
class Scenario:
    """
    A named what-if or planning scenario.
    
    The LLM produces these. The scenario engine applies them deterministically.
    """
    name: str
    description: str
    patch: InputPatch
    assumptions: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Level 2 — Advanced patch operations (MVP-2: equipment insertion, rewiring)
# ---------------------------------------------------------------------------

OpType = Literal[
    "set",          # set a value on a unit/stream
    "scale",        # multiply a numeric value
    "add_unit",     # insert new equipment from template
    "connect",      # connect two ports
    "disconnect",   # disconnect a port
    "set_control",  # change controller mode/setpoint
]


@dataclass
class PatchOp:
    """
    A single structured operation on the process model.
    Used in MVP-2+ for equipment insertion and rewiring.
    """
    op: OpType
    target: str                             # canonical path e.g. "units.23-KA-01.speed_rpm"
    value: Any = None                       # for set / set_control
    factor: Optional[float] = None          # for scale
    template: Optional[str] = None          # for add_unit (e.g. "cooler", "compressor_stage")
    params: Optional[Dict[str, Any]] = None # template parameters
    src: Optional[str] = None               # connect source port path
    dst: Optional[str] = None               # connect destination port path
    note: Optional[str] = None              # human note shown in report


@dataclass
class AdvancedScenario:
    """
    An advanced scenario using PatchOp operations (MVP-2+).
    """
    name: str
    description: str
    ops: List[PatchOp]
    assumptions: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------

def scenario_from_dict(d: dict) -> Scenario:
    """Parse a Scenario from LLM JSON output."""
    patch_data = d.get("patch", {})

    # Parse add_units if present
    add_units = []
    for au in patch_data.get("add_units", []):
        add_units.append(AddUnitOp(
            name=au["name"],
            equipment_type=au["equipment_type"],
            insert_after=au["insert_after"],
            params=au.get("params", {}),
        ))

    patch = InputPatch(
        changes=patch_data.get("changes", {}),
        add_units=add_units,
        note=patch_data.get("note")
    )
    return Scenario(
        name=d["name"],
        description=d["description"],
        patch=patch,
        assumptions=d.get("assumptions", {})
    )


def scenarios_from_json(data: dict) -> List[Scenario]:
    """Parse a list of Scenarios from LLM JSON output."""
    return [scenario_from_dict(s) for s in data.get("scenarios", [])]
