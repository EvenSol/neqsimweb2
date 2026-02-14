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
class AddComponentOp:
    """
    Add chemical components to an existing stream's fluid.
    
    Used when the engineer wants to add new substances to the feed
    (e.g. add inhibitor, add heavy hydrocarbons, inject water).
    
    Example:
        AddComponentOp(
            stream_name="feed gas",
            components={"n-decane": 500.0, "benzene": 300.0, "water": 200.0},
            flow_unit="kg/hr",
        )
    """
    stream_name: str                        # name of stream to modify
    components: Dict[str, float]            # component_name -> flow_rate
    flow_unit: str = "kg/hr"                # "kg/hr", "mol/sec", etc.


@dataclass
class AddStreamOp:
    """
    Add a new inlet stream and (optionally) mix it into the process.

    Example:
        AddStreamOp(
            name="liquid feed",
            insert_after="inlet separator",
            components={"nC10": 500.0, "benzene": 300.0, "water": 200.0},
            flow_unit="kg/hr",
            temperature_C=30.0,
            pressure_bara=30.0,
        )
    """
    name: str
    insert_after: str                        # unit to insert mixer after
    components: Dict[str, float]             # component_name -> flow_rate
    flow_unit: str = "kg/hr"                 # "kg/hr", "mol/sec", etc.
    temperature_C: Optional[float] = None
    pressure_bara: Optional[float] = None
    base_stream: Optional[str] = None        # stream to clone thermo system from
    mixer_name: Optional[str] = None         # optional mixer unit name


@dataclass
class TargetSpec:
    """
    Iterative target-seeking specification.
    
    The solver adjusts a scale factor applied to the inputs until
    the target KPI reaches the desired value.
    
    Example: "Add heavy components until separator liquid = 1000 kg/hr"
        TargetSpec(
            target_kpi="inlet separator.liquidOutStream.flow_kg_hr",
            target_value=1000.0,
            tolerance_pct=2.0,
            variable="component_scale",
            initial_guess=1.0,
            min_value=0.01,
            max_value=50.0,
        )
    """
    target_kpi: str                         # KPI key to match (from _extract_results)
    target_value: float                     # desired value
    tolerance_pct: float = 2.0              # acceptable relative error %
    variable: str = "component_scale"       # what to scale (currently: component flows)
    initial_guess: float = 1.0              # starting scale factor
    min_value: float = 0.01                 # lower bound
    max_value: float = 50.0                 # upper bound
    max_iterations: int = 20                # iteration limit


@dataclass
class AddProcessOp:
    """
    Add a sub-process system (ProcessModule) to the main process.
    
    Allows composing multiple process systems, e.g. adding a gas treatment
    module or a compression train module to an existing process.
    
    Example:
        AddProcessOp(
            name="dew point control module",
            process_file="dewpoint_control.neqsim",
            connect_inlet_to="inlet separator.gasOutStream",
            connect_outlet_to="1st stage compressor",
        )
    """
    name: str                               # name for the sub-process
    units: List[Dict[str, Any]]             # list of units to create [{equipment_type, name, params}, ...]
    insert_after: str                       # insert the whole sub-process after this unit
    connect_outlet_to: Optional[str] = None # connect the last unit's outlet to this existing unit's inlet


@dataclass
class InputPatch:
    """
    Change one or more ProcessInput fields, and optionally add new units/components.
    
    Supports absolute values and relative operations:
        Absolute:  {"Psep1": 25.0}
        Relative:  {"Psep1": {"op": "add", "value": 5.0}}
        Scale:     {"feed_rate": {"op": "scale", "value": 1.1}}
    
    Supports adding new equipment via add_units list.
    Supports adding new inlet streams via add_streams list.
    Supports adding chemical components via add_components list.
    Supports iterative target-seeking via targets list.
    Supports adding sub-process systems via add_process list.
    """
    changes: Dict[str, Any]
    add_units: List[AddUnitOp] = field(default_factory=list)
    add_streams: List[AddStreamOp] = field(default_factory=list)
    add_components: List[AddComponentOp] = field(default_factory=list)
    targets: List[TargetSpec] = field(default_factory=list)
    add_process: List[AddProcessOp] = field(default_factory=list)
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

    # Parse add_streams if present
    add_streams = []
    for ast in patch_data.get("add_streams", []):
        add_streams.append(AddStreamOp(
            name=ast["name"],
            insert_after=ast["insert_after"],
            components=ast.get("components", {}),
            flow_unit=ast.get("flow_unit", "kg/hr"),
            temperature_C=ast.get("temperature_C"),
            pressure_bara=ast.get("pressure_bara"),
            base_stream=ast.get("base_stream"),
            mixer_name=ast.get("mixer_name"),
        ))

    # Parse add_components if present
    add_components = []
    for ac in patch_data.get("add_components", []):
        add_components.append(AddComponentOp(
            stream_name=ac["stream_name"],
            components=ac["components"],
            flow_unit=ac.get("flow_unit", "kg/hr"),
        ))

    # Parse targets if present
    targets = []
    for t in patch_data.get("targets", []):
        targets.append(TargetSpec(
            target_kpi=t["target_kpi"],
            target_value=float(t["target_value"]),
            tolerance_pct=float(t.get("tolerance_pct", 2.0)),
            variable=t.get("variable", "component_scale"),
            initial_guess=float(t.get("initial_guess", 1.0)),
            min_value=float(t.get("min_value", 0.01)),
            max_value=float(t.get("max_value", 50.0)),
            max_iterations=int(t.get("max_iterations", 20)),
        ))

    # Parse add_process if present
    add_process = []
    for ap in patch_data.get("add_process", []):
        add_process.append(AddProcessOp(
            name=ap["name"],
            units=ap.get("units", []),
            insert_after=ap["insert_after"],
            connect_outlet_to=ap.get("connect_outlet_to"),
        ))

    patch = InputPatch(
        changes=patch_data.get("changes", {}),
        add_units=add_units,
        add_streams=add_streams,
        add_components=add_components,
        targets=targets,
        add_process=add_process,
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
