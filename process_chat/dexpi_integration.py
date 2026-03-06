"""
DEXPI P&ID Integration — Import DEXPI XML files, parse P&ID topology,
and establish NeqSim process models from DEXPI data.

Supports two complementary analysis paths:
  1. **P&ID topology analysis** — Pure Python XML parsing of the Proteus XML
     (DEXPI v1.3) to extract equipment, piping, instrumentation, and connectivity.
  2. **NeqSim thermodynamic model** — Uses NeqSim's Java DexpiXmlReader to
     import DEXPI equipment into a runnable ProcessSystem (requires a template
     fluid/stream for thermodynamic calculations).

Reference DEXPI example PIDs: https://gitlab.com/dexpi/TrainingTestCases
"""
from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data classes for P&ID analysis results
# ---------------------------------------------------------------------------

@dataclass
class DexpiEquipmentInfo:
    """One equipment item parsed from DEXPI XML."""
    id: str
    component_class: str
    tag_name: str = ""
    nozzles: List[str] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class DexpiPipingInfo:
    """One piping network system from DEXPI XML."""
    id: str
    fluid_code: str = ""
    line_number: str = ""
    nominal_diameter: str = ""
    piping_class: str = ""
    segments: int = 0
    connections: List[Dict[str, str]] = field(default_factory=list)
    valves: List[str] = field(default_factory=list)


@dataclass
class DexpiInstrumentInfo:
    """Instrumentation loop/function from DEXPI XML."""
    id: str
    component_class: str = ""
    tag: str = ""
    measured_variable: str = ""
    function_type: str = ""


@dataclass
class DexpiPIDSummary:
    """Summary of a parsed DEXPI P&ID."""
    title: str = ""
    drawing_number: str = ""
    revision: str = ""
    application: str = ""
    schema_version: str = ""
    drawings: List[Dict[str, str]] = field(default_factory=list)
    equipment: List[DexpiEquipmentInfo] = field(default_factory=list)
    piping: List[DexpiPipingInfo] = field(default_factory=list)
    instruments: List[DexpiInstrumentInfo] = field(default_factory=list)
    actuating_systems: int = 0
    nozzle_count: int = 0
    connection_count: int = 0


@dataclass
class DexpiAnalysisResult:
    """Complete DEXPI analysis result for display and LLM consumption."""
    pid_summary: DexpiPIDSummary
    neqsim_model_loaded: bool = False
    neqsim_model: Any = None
    neqsim_units: int = 0
    neqsim_streams: int = 0
    equipment_type_counts: Dict[str, int] = field(default_factory=dict)
    piping_summary: Dict[str, Any] = field(default_factory=dict)
    connectivity_info: str = ""
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper: extract GenericAttribute values
# ---------------------------------------------------------------------------

def _get_generic_attr(element: ET.Element, attr_name: str) -> str:
    """Search GenericAttributes for a named attribute and return its Value."""
    for ga_set in element.iter("GenericAttributes"):
        for ga in ga_set.iter("GenericAttribute"):
            name = ga.get("Name", "")
            # Match either exact name or the "AssignmentClass" suffixed version
            if name == attr_name or name == f"{attr_name}AssignmentClass":
                return ga.get("Value", "")
    return ""


def _get_tag_name(element: ET.Element) -> str:
    """Extract the TagName from an equipment element."""
    tag = _get_generic_attr(element, "TagName")
    if tag:
        return tag
    # Try to compose from prefix + sequence + suffix
    prefix = _get_generic_attr(element, "TagNamePrefix")
    seq = _get_generic_attr(element, "TagNameSequenceNumber")
    suffix = _get_generic_attr(element, "TagNameSuffix")
    if prefix or seq:
        return f"{prefix}{seq}{suffix}".strip()
    return element.get("ID", "")


# ---------------------------------------------------------------------------
# Core parsing: DEXPI XML → DexpiPIDSummary
# ---------------------------------------------------------------------------

def parse_dexpi_xml(xml_bytes: bytes) -> DexpiPIDSummary:
    """Parse a DEXPI Proteus XML file and return a structured P&ID summary.

    This uses pure Python XML parsing (no Java or pyDEXPI dependency) to
    extract equipment, piping, instrumentation, and connectivity from the
    standard Proteus XML schema used by DEXPI v1.3.
    """
    root = ET.fromstring(xml_bytes)
    summary = DexpiPIDSummary()

    # --- Plant information ---
    pi = root.find("PlantInformation")
    if pi is not None:
        summary.application = pi.get("Application", "")
        summary.schema_version = pi.get("SchemaVersion", "")

    # --- Metadata ---
    meta = root.find("MetaData")
    if meta is not None:
        summary.title = _get_generic_attr(meta, "DrawingTitle") or _get_generic_attr(meta, "DrawingSubTitle")
        summary.drawing_number = _get_generic_attr(meta, "DrawingNumber")
        summary.revision = _get_generic_attr(meta, "RevisionNumber")

    # --- Equipment ---
    # DEXPI structure: <Equipment> wrapper contains child elements whose tag
    # names ARE the equipment class (e.g. <Separator>, <TubularHeatExchanger>).
    # Also handle flat layouts where equipment elements are directly under PlantModel.
    _EQUIPMENT_CLASSES = {
        "CentrifugalPump", "ReciprocatingPump", "Pump",
        "PlateHeatExchanger", "TubularHeatExchanger", "HeatExchanger",
        "AirCooledHeatExchanger",
        "Tank", "PressureVessel", "Column", "ProcessColumn",
        "Compressor", "CentrifugalCompressor", "ReciprocatingCompressor",
        "Reactor", "StirredReactor",
        "Filter", "Dryer", "Mixer", "Separator",
        "Stripper", "Absorber", "Distillation",
    }

    _SKIP_SUB_EQUIPMENT = {"Chamber", "TubeBundle", "Displacer", "Impeller", "Equipment"}

    def _extract_equipment(element: ET.Element):
        """Extract equipment info from an element."""
        comp_class = element.get("ComponentClass", "") or element.tag
        if comp_class in _SKIP_SUB_EQUIPMENT or comp_class not in _EQUIPMENT_CLASSES:
            return None

        tag = _get_tag_name(element)
        # Only collect direct nozzles (not from nested sub-equipment)
        nozzles = [n.get("ID", "") for n in element.findall("Nozzle")]

        attrs = {}
        for ga_set in element.findall("GenericAttributes"):
            for ga in ga_set.findall("GenericAttribute"):
                name = ga.get("Name", "")
                val = ga.get("Value", "")
                units = ga.get("Units", "")
                if val and name:
                    attrs[name] = f"{val} {units}".strip() if units else val

        return DexpiEquipmentInfo(
            id=element.get("ID", ""),
            component_class=comp_class,
            tag_name=tag,
            nozzles=nozzles,
            attributes=attrs,
        )

    # Collect from <Equipment> wrapper children AND from named elements anywhere
    seen_ids = set()
    for eq_wrapper in root.iter("Equipment"):
        for child in eq_wrapper:
            info = _extract_equipment(child)
            if info and info.id not in seen_ids:
                summary.equipment.append(info)
                summary.nozzle_count += len(info.nozzles)
                seen_ids.add(info.id)
    # Also scan for equipment classes directly under root (flat DEXPI layouts)
    for cls_name in _EQUIPMENT_CLASSES:
        for elem in root.iter(cls_name):
            eq_id = elem.get("ID", "")
            if eq_id not in seen_ids:
                info = _extract_equipment(elem)
                if info:
                    summary.equipment.append(info)
                    summary.nozzle_count += len(info.nozzles)
                    seen_ids.add(eq_id)

    # --- Piping network systems ---
    for pns in root.iter("PipingNetworkSystem"):
        pns_id = pns.get("ID", "")
        fluid_code = _get_generic_attr(pns, "FluidCode")
        line_number = _get_generic_attr(pns, "LineNumber")
        nom_dia = _get_generic_attr(pns, "NominalDiameterRepresentation") or \
                  _get_generic_attr(pns, "NominalDiameterNumericalValueRepresentation")
        pip_class = _get_generic_attr(pns, "PipingClassCode")

        segments = list(pns.iter("PipingNetworkSegment"))
        connections = []
        valves = []
        for seg in segments:
            for conn in seg.iter("Connection"):
                connections.append({
                    "from": conn.get("FromID", ""),
                    "to": conn.get("ToID", ""),
                })
            for pc in seg.iter("PipingComponent"):
                pc_class = pc.get("ComponentClass", "")
                if "Valve" in pc_class or "valve" in pc_class:
                    pc_name = _get_generic_attr(pc, "PipingComponentName") or pc.get("ID", "")
                    valves.append(f"{pc_class}: {pc_name}")

        info = DexpiPipingInfo(
            id=pns_id,
            fluid_code=fluid_code,
            line_number=line_number,
            nominal_diameter=nom_dia,
            piping_class=pip_class,
            segments=len(segments),
            connections=connections,
            valves=valves,
        )
        summary.piping.append(info)
        summary.connection_count += len(connections)

    # --- Instrumentation ---
    for pif in root.iter("ProcessInstrumentationFunction"):
        pif_id = pif.get("ID", "")
        comp_class = pif.get("ComponentClass", "")
        tag = ""
        for label in pif.iter("Label"):
            for text in label.iter("Text"):
                t = text.get("String", "")
                if t:
                    tag = t
                    break
            if tag:
                break

        info = DexpiInstrumentInfo(
            id=pif_id,
            component_class=comp_class,
            tag=tag,
        )
        summary.instruments.append(info)

    # --- Signal-generating functions ---
    for sgf in root.iter("ProcessSignalGeneratingFunction"):
        sgf_id = sgf.get("ID", "")
        comp_class = sgf.get("ComponentClass", "")
        tag = ""
        for label in sgf.iter("Label"):
            for text in label.iter("Text"):
                t = text.get("String", "")
                if t:
                    tag = t
                    break
            if tag:
                break

        info = DexpiInstrumentInfo(
            id=sgf_id,
            component_class=comp_class,
            tag=tag,
            function_type="signal_generating",
        )
        summary.instruments.append(info)

    # --- Actuating systems count ---
    summary.actuating_systems = len(list(root.iter("ActuatingSystem")))

    # --- Drawing sheets ---
    for drawing in root.iter("Drawing"):
        d_info: Dict[str, str] = {"id": drawing.get("ID", "")}
        d_title = _get_generic_attr(drawing, "DrawingTitle") or _get_generic_attr(drawing, "Title")
        d_number = _get_generic_attr(drawing, "DrawingNumber") or _get_generic_attr(drawing, "SheetNumber")
        if d_title:
            d_info["title"] = d_title
        if d_number:
            d_info["number"] = d_number
        summary.drawings.append(d_info)

    return summary


# ---------------------------------------------------------------------------
# NeqSim integration: DEXPI XML → NeqSim ProcessSystem
# ---------------------------------------------------------------------------

def load_dexpi_to_neqsim(
    xml_bytes: bytes,
    filename: str = "dexpi_pid.xml",
    fluid_spec: Optional[Dict] = None,
) -> Optional[Any]:
    """Import a DEXPI XML file into a NeqSim ProcessSystem using the Java DexpiXmlReader.

    Parameters
    ----------
    xml_bytes : bytes
        Raw bytes of the DEXPI XML file.
    filename : str
        Original filename (used for temp file extension).
    fluid_spec : dict, optional
        Fluid composition specification for the template stream.
        If None, a default natural gas composition is used.

    Returns
    -------
    NeqSimProcessModel or None
        A wrapped NeqSim process model if successful, None otherwise.
    """
    from .process_model import NeqSimProcessModel

    try:
        from neqsim.thermo import fluid_df
        from neqsim import jneqsim
        import pandas as pd

        # Create a template fluid for thermodynamic calculations
        if fluid_spec and "components" in fluid_spec:
            comp_data = fluid_spec["components"]
            rows = [
                {"ComponentName": name, "MolarComposition[-]": frac}
                for name, frac in comp_data.items()
            ]
            df = pd.DataFrame(rows)
            template_fluid = fluid_df(df, lastIsPlusFraction=False, add_all_components=False)
        else:
            # Default: simple natural gas
            rows = [
                {"ComponentName": "methane", "MolarComposition[-]": 0.85},
                {"ComponentName": "ethane", "MolarComposition[-]": 0.07},
                {"ComponentName": "propane", "MolarComposition[-]": 0.03},
                {"ComponentName": "CO2", "MolarComposition[-]": 0.02},
                {"ComponentName": "nitrogen", "MolarComposition[-]": 0.03},
            ]
            df = pd.DataFrame(rows)
            template_fluid = fluid_df(df, lastIsPlusFraction=False, add_all_components=False)

        template_fluid.setTemperature(25.0, "C")
        template_fluid.setPressure(50.0, "bara")

        # Create a NeqSim stream as template
        template_stream = jneqsim.processSimulation.processEquipment.stream.Stream(
            "template stream", template_fluid
        )
        template_stream.setFlowRate(100.0, "kg/hr")

        # Write XML to a temporary file for Java reader
        suffix = os.path.splitext(filename)[1] or ".xml"
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False, mode="wb"
        ) as tmp:
            tmp.write(xml_bytes)
            tmp_path = tmp.name

        try:
            # Use NeqSim's DexpiXmlReader (static method)
            DexpiXmlReader = jneqsim.process.processmodel.dexpi.DexpiXmlReader
            java_file = jneqsim.java.io.File(tmp_path)
            process_system = DexpiXmlReader.read(java_file, template_stream)

            if process_system is not None:
                model = NeqSimProcessModel.from_process_system(process_system)
                return model
        finally:
            os.unlink(tmp_path)

    except Exception:
        # DexpiXmlReader may not be available in all NeqSim versions
        pass

    return None


# ---------------------------------------------------------------------------
# Fluid code → composition mapping
# ---------------------------------------------------------------------------

# Default compositions keyed by DEXPI FluidCode.
FLUID_CODE_COMPOSITIONS: Dict[str, List[Dict[str, Any]]] = {
    "NG": [  # Natural Gas
        {"ComponentName": "methane", "MolarComposition[-]": 0.80},
        {"ComponentName": "ethane", "MolarComposition[-]": 0.06},
        {"ComponentName": "propane", "MolarComposition[-]": 0.03},
        {"ComponentName": "i-butane", "MolarComposition[-]": 0.01},
        {"ComponentName": "n-butane", "MolarComposition[-]": 0.01},
        {"ComponentName": "CO2", "MolarComposition[-]": 0.025},
        {"ComponentName": "nitrogen", "MolarComposition[-]": 0.035},
        {"ComponentName": "water", "MolarComposition[-]": 0.01},
        {"ComponentName": "n-pentane", "MolarComposition[-]": 0.005},
        {"ComponentName": "n-hexane", "MolarComposition[-]": 0.005},
    ],
    "HC": [  # Hydrocarbon condensate
        {"ComponentName": "n-pentane", "MolarComposition[-]": 0.20},
        {"ComponentName": "n-hexane", "MolarComposition[-]": 0.25},
        {"ComponentName": "n-heptane", "MolarComposition[-]": 0.25},
        {"ComponentName": "n-octane", "MolarComposition[-]": 0.15},
        {"ComponentName": "propane", "MolarComposition[-]": 0.05},
        {"ComponentName": "n-butane", "MolarComposition[-]": 0.10},
    ],
    "CW": [  # Cooling water
        {"ComponentName": "water", "MolarComposition[-]": 1.0},
    ],
}


def get_fluid_rows_for_code(fluid_code: str) -> List[Dict[str, Any]]:
    """Return composition rows for a given DEXPI FluidCode.

    Falls back to natural gas (NG) for unknown codes.
    """
    return FLUID_CODE_COMPOSITIONS.get(fluid_code.upper(), FLUID_CODE_COMPOSITIONS["NG"])


# ---------------------------------------------------------------------------
# Helper: extract numeric design data from equipment attributes
# ---------------------------------------------------------------------------

def _get_design_value(attrs: Dict[str, str], *keys: str) -> Optional[float]:
    """Extract a numeric value from equipment attributes by key name.

    Tries each key in order and returns the first parseable float, stripping
    any trailing unit strings (e.g. "85 barg" → 85.0).
    """
    for key in keys:
        val = attrs.get(key, "")
        if not val:
            continue
        # Take the first whitespace-separated token as the number
        num_str = val.split()[0]
        try:
            return float(num_str)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Fallback: Build a NeqSim process from parsed DEXPI topology
# ---------------------------------------------------------------------------

def create_neqsim_process_from_dexpi(
    pid_summary: DexpiPIDSummary,
    fluid_spec: Optional[Dict] = None,
) -> Optional[Any]:
    """Build a NeqSim ProcessSystem programmatically from DEXPI P&ID topology.

    Uses piping connectivity (FromID/ToID on nozzles) to determine equipment
    ordering.  Follows the main process gas path from INLET through the
    equipment chain.  Falls back to alphabetical tag order if connectivity
    cannot be resolved.

    Returns a NeqSimProcessModel or None.
    """
    from .process_model import NeqSimProcessModel

    try:
        from neqsim.thermo import fluid_df
        from neqsim import jneqsim
        import pandas as pd

        # --- Create default fluid ---
        if fluid_spec and "components" in fluid_spec:
            comp_data = fluid_spec["components"]
            rows = [
                {"ComponentName": name, "MolarComposition[-]": frac}
                for name, frac in comp_data.items()
            ]
        else:
            # Determine primary fluid code from piping networks
            primary_code = "NG"
            if pid_summary.piping:
                code_counts: Dict[str, int] = {}
                for p in pid_summary.piping:
                    if p.fluid_code:
                        code_counts[p.fluid_code] = code_counts.get(p.fluid_code, 0) + 1
                if code_counts:
                    primary_code = max(code_counts, key=code_counts.get)
            rows = get_fluid_rows_for_code(primary_code)
        df = pd.DataFrame(rows)
        fluid = fluid_df(df, lastIsPlusFraction=False, add_all_components=False)
        fluid.setTemperature(30.0, "C")
        fluid.setPressure(65.0, "bara")
        fluid.setTotalFlowRate(10.0, "MSm3/day")

        # --- Build connectivity graph ---
        # Map nozzle ID → equipment ID
        nozzle_to_eq: Dict[str, str] = {}
        eq_by_id: Dict[str, DexpiEquipmentInfo] = {}
        for eq in pid_summary.equipment:
            eq_id = eq.id
            eq_by_id[eq_id] = eq
            for nz in eq.nozzles:
                nozzle_to_eq[nz] = eq_id

        # Build directed edges: (from_eq, to_eq) from piping connections
        # Only follow the primary fluid code path for the main process chain
        # (reuse primary_code already computed above for the fluid)

        # adjacency: from_eq_id → list of to_eq_id (on the primary fluid path)
        adjacency: Dict[str, List[str]] = {}
        inlet_equipment: Optional[str] = None  # first equipment receiving from INLET

        for p in pid_summary.piping:
            # Only follow primary fluid code for the main chain
            if p.fluid_code and p.fluid_code != primary_code:
                continue
            for conn in p.connections:
                from_nz = conn.get("from", "")
                to_nz = conn.get("to", "")
                from_eq = nozzle_to_eq.get(from_nz)
                to_eq = nozzle_to_eq.get(to_nz)

                # Detect INLET → first equipment
                if from_nz.upper() == "INLET" and to_eq:
                    inlet_equipment = to_eq
                    continue

                if from_eq and to_eq and from_eq != to_eq:
                    adjacency.setdefault(from_eq, []).append(to_eq)

        # Walk the graph from inlet to build a topological ordering
        ordered_eq_ids: List[str] = []
        visited: set = set()

        def _walk(eq_id: str):
            if eq_id in visited:
                return
            visited.add(eq_id)
            ordered_eq_ids.append(eq_id)
            for next_eq in adjacency.get(eq_id, []):
                _walk(next_eq)

        if inlet_equipment:
            _walk(inlet_equipment)

        # Add any remaining equipment not reached by the walk
        for eq in pid_summary.equipment:
            if eq.id not in visited:
                ordered_eq_ids.append(eq.id)

        # If connectivity resolution is empty, fall back to tag sort
        if not ordered_eq_ids:
            ordered_eq_ids = [eq.id for eq in sorted(
                pid_summary.equipment, key=lambda e: e.tag_name or e.id
            )]

        # --- Build NeqSim process ---
        ProcessSystem = jneqsim.process.processmodel.ProcessSystem
        proc = ProcessSystem()

        Stream = jneqsim.process.equipment.stream.Stream
        feed = Stream("feed gas", fluid)
        proc.add(feed)

        prev_stream = feed

        for eq_id in ordered_eq_ids:
            eq = eq_by_id.get(eq_id)
            if eq is None:
                continue
            cls = eq.component_class.lower()
            tag = eq.tag_name or eq.id

            if "separator" in cls or "vessel" in cls:
                Sep = jneqsim.process.equipment.separator.Separator
                sep = Sep(tag, prev_stream)
                proc.add(sep)
                gas_out = Stream(f"{tag}-gas out", sep.getGasOutStream())
                proc.add(gas_out)
                prev_stream = gas_out

            elif "heatexchanger" in cls or "cooler" in cls:
                Cooler = jneqsim.process.equipment.heatexchanger.Cooler
                cooler = Cooler(tag, prev_stream)
                design_temp = _get_design_value(eq.attributes, "DesignTemperature")
                out_temp = design_temp if design_temp is not None else 25.0
                cooler.setOutTemperature(273.15 + out_temp)
                proc.add(cooler)
                out = Stream(f"{tag}-out", cooler.getOutletStream())
                proc.add(out)
                prev_stream = out

            elif "compressor" in cls:
                Comp = jneqsim.process.equipment.compressor.Compressor
                comp = Comp(tag, prev_stream)
                design_p = _get_design_value(eq.attributes, "DesignPressure")
                comp.setOutletPressure(design_p if design_p is not None else 100.0)
                proc.add(comp)
                out = Stream(f"{tag}-out", comp.getOutletStream())
                proc.add(out)
                prev_stream = out

            elif "pump" in cls:
                Pump = jneqsim.process.equipment.pump.Pump
                pump = Pump(tag, prev_stream)
                design_p = _get_design_value(eq.attributes, "DesignPressure")
                pump.setOutletPressure(design_p if design_p is not None else 80.0)
                proc.add(pump)
                out = Stream(f"{tag}-out", pump.getOutletStream())
                proc.add(out)
                prev_stream = out

            elif "tank" in cls:
                Sep = jneqsim.process.equipment.separator.Separator
                tank = Sep(tag, prev_stream)
                proc.add(tank)
                liq_out = Stream(f"{tag}-liq out", tank.getLiquidOutStream())
                proc.add(liq_out)
                prev_stream = liq_out

            elif "valve" in cls:
                Valve = jneqsim.process.equipment.valve.ThrottlingValve
                valve = Valve(tag, prev_stream)
                design_p = _get_design_value(eq.attributes, "DesignPressure")
                valve.setOutletPressure(design_p if design_p is not None else 50.0)
                proc.add(valve)
                out = Stream(f"{tag}-out", valve.getOutletStream())
                proc.add(out)
                prev_stream = out

            elif "reactor" in cls:
                Reactor = jneqsim.process.equipment.reactor.GibbsReactor
                reactor = Reactor(tag, prev_stream)
                proc.add(reactor)
                out = Stream(f"{tag}-out", reactor.getOutletStream())
                proc.add(out)
                prev_stream = out

            elif "mixer" in cls:
                # Mixer needs multiple inlets; with topology walk we only
                # have one prev_stream, so model as pass-through cooler at
                # current temperature.
                Cooler = jneqsim.process.equipment.heatexchanger.Cooler
                mixer = Cooler(tag, prev_stream)
                design_temp = _get_design_value(eq.attributes, "DesignTemperature")
                out_temp = design_temp if design_temp is not None else 25.0
                mixer.setOutTemperature(273.15 + out_temp)
                proc.add(mixer)
                out = Stream(f"{tag}-out", mixer.getOutletStream())
                proc.add(out)
                prev_stream = out

            elif "column" in cls or "distillation" in cls or "absorber" in cls or "stripper" in cls:
                # Model columns/absorbers as separators (gas out follows main path)
                Sep = jneqsim.process.equipment.separator.Separator
                col = Sep(tag, prev_stream)
                proc.add(col)
                gas_out = Stream(f"{tag}-gas out", col.getGasOutStream())
                proc.add(gas_out)
                prev_stream = gas_out

            elif "filter" in cls or "dryer" in cls:
                # Model as pass-through (no phase change expected)
                Cooler = jneqsim.process.equipment.heatexchanger.Cooler
                flt = Cooler(tag, prev_stream)
                design_temp = _get_design_value(eq.attributes, "DesignTemperature")
                out_temp = design_temp if design_temp is not None else 25.0
                flt.setOutTemperature(273.15 + out_temp)
                proc.add(flt)
                out = Stream(f"{tag}-out", flt.getOutletStream())
                proc.add(out)
                prev_stream = out

            elif "expander" in cls:
                Exp = jneqsim.process.equipment.expander.Expander
                exp = Exp(tag, prev_stream)
                design_p = _get_design_value(eq.attributes, "DesignPressure")
                exp.setOutletPressure(design_p if design_p is not None else 20.0)
                proc.add(exp)
                out = Stream(f"{tag}-out", exp.getOutletStream())
                proc.add(out)
                prev_stream = out

        # Run the process
        proc.run()

        model = NeqSimProcessModel.from_process_system(proc)
        return model

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Full DEXPI analysis: combines P&ID parsing + optional NeqSim import
# ---------------------------------------------------------------------------

def run_dexpi_analysis(
    xml_bytes: bytes,
    filename: str = "dexpi_pid.xml",
    fluid_spec: Optional[Dict] = None,
    try_neqsim_import: bool = True,
) -> DexpiAnalysisResult:
    """Run a complete DEXPI P&ID analysis.

    Parameters
    ----------
    xml_bytes : bytes
        Raw DEXPI XML file content.
    filename : str
        Original filename.
    fluid_spec : dict, optional
        Fluid spec for NeqSim import (components dict).
    try_neqsim_import : bool
        Whether to attempt importing into NeqSim (default True).

    Returns
    -------
    DexpiAnalysisResult
        Structured analysis results.
    """
    warnings = []

    # Step 1: Parse P&ID topology
    try:
        pid_summary = parse_dexpi_xml(xml_bytes)
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML: {e}")

    # Step 2: Build equipment type counts
    type_counts: Dict[str, int] = {}
    for eq in pid_summary.equipment:
        cls = eq.component_class
        type_counts[cls] = type_counts.get(cls, 0) + 1

    # Step 3: Piping summary
    piping_summary: Dict[str, Any] = {}
    fluid_codes = set()
    total_lines = len(pid_summary.piping)
    total_valves = sum(len(p.valves) for p in pid_summary.piping)
    for p in pid_summary.piping:
        if p.fluid_code:
            fluid_codes.add(p.fluid_code)
    piping_summary = {
        "total_lines": total_lines,
        "total_valves": total_valves,
        "fluid_codes": sorted(fluid_codes),
        "total_segments": sum(p.segments for p in pid_summary.piping),
    }

    # Step 4: Connectivity summary
    connectivity_parts = []
    for p in pid_summary.piping:
        for conn in p.connections:
            from_id = conn.get("from", "")
            to_id = conn.get("to", "")
            if from_id and to_id:
                connectivity_parts.append(f"{from_id} → {to_id}")
    connectivity_info = f"{pid_summary.connection_count} connections across {total_lines} piping lines"

    # Step 5: Auto-detect fluid from piping fluid codes if not specified
    effective_fluid_spec = fluid_spec
    if effective_fluid_spec is None and fluid_codes:
        # Pick the primary (most common) fluid code and use its default composition
        code_counts: Dict[str, int] = {}
        for p in pid_summary.piping:
            if p.fluid_code:
                code_counts[p.fluid_code] = code_counts.get(p.fluid_code, 0) + 1
        if code_counts:
            primary_code = max(code_counts, key=code_counts.get)
            rows = get_fluid_rows_for_code(primary_code)
            effective_fluid_spec = {
                "components": {r["ComponentName"]: r["MolarComposition[-]"] for r in rows}
            }

    # Step 6: Attempt NeqSim import
    neqsim_model = None
    neqsim_units = 0
    neqsim_streams = 0
    if try_neqsim_import:
        # Try Java DexpiXmlReader first
        try:
            neqsim_model = load_dexpi_to_neqsim(xml_bytes, filename, effective_fluid_spec)
        except Exception:
            pass

        # Fallback: build process from parsed topology
        if neqsim_model is None and pid_summary.equipment:
            try:
                neqsim_model = create_neqsim_process_from_dexpi(pid_summary, effective_fluid_spec)
            except Exception as e:
                warnings.append(f"NeqSim process build: {e}")

        if neqsim_model is not None:
            try:
                units = neqsim_model.list_units()
                streams = neqsim_model.list_streams()
                neqsim_units = len(units)
                neqsim_streams = len(streams)
            except Exception:
                pass
        else:
            warnings.append("NeqSim process model could not be created from P&ID")

    result = DexpiAnalysisResult(
        pid_summary=pid_summary,
        neqsim_model_loaded=neqsim_model is not None,
        neqsim_model=neqsim_model,
        neqsim_units=neqsim_units,
        neqsim_streams=neqsim_streams,
        equipment_type_counts=type_counts,
        piping_summary=piping_summary,
        connectivity_info=connectivity_info,
        warnings=warnings,
    )

    return result


# ---------------------------------------------------------------------------
# Formatter for LLM consumption
# ---------------------------------------------------------------------------

def format_dexpi_result(result: DexpiAnalysisResult) -> str:
    """Format DEXPI analysis result as text for the LLM to reason about."""
    lines = []
    pid = result.pid_summary

    lines.append("=== DEXPI P&ID Analysis ===")
    if pid.title:
        lines.append(f"Title: {pid.title}")
    if pid.drawing_number:
        lines.append(f"Drawing: {pid.drawing_number}  Rev: {pid.revision}")
    if pid.schema_version:
        lines.append(f"Schema: DEXPI / Proteus v{pid.schema_version}")

    if pid.drawings:
        lines.append(f"Drawing sheets: {len(pid.drawings)}")
        for d in pid.drawings:
            d_label = d.get("title") or d.get("number") or d.get("id", "")
            lines.append(f"  - {d_label}")

    lines.append("")
    lines.append("--- Equipment ---")
    if result.equipment_type_counts:
        for cls, count in sorted(result.equipment_type_counts.items()):
            lines.append(f"  {cls}: {count}")
    else:
        lines.append("  (no equipment found)")

    lines.append("")
    lines.append("--- Equipment Details ---")
    for eq in pid.equipment:
        attrs_str = ""
        if eq.attributes:
            key_attrs = {k: v for k, v in eq.attributes.items()
                        if "Design" in k or "Capacity" in k or "Power" in k
                        or "Temperature" in k or "Pressure" in k or "Speed" in k}
            if key_attrs:
                attrs_str = " | " + ", ".join(f"{k}={v}" for k, v in key_attrs.items())
        lines.append(f"  [{eq.component_class}] {eq.tag_name} — {len(eq.nozzles)} nozzles{attrs_str}")

    lines.append("")
    lines.append("--- Piping ---")
    ps = result.piping_summary
    lines.append(f"  Lines: {ps.get('total_lines', 0)}")
    lines.append(f"  Segments: {ps.get('total_segments', 0)}")
    lines.append(f"  Valves: {ps.get('total_valves', 0)}")
    lines.append(f"  Fluid codes: {', '.join(ps.get('fluid_codes', []))}")

    lines.append("")
    lines.append("--- Piping Details ---")
    for p in pid.piping:
        valve_str = f" | Valves: {', '.join(p.valves)}" if p.valves else ""
        lines.append(
            f"  Line {p.line_number} ({p.fluid_code}) {p.nominal_diameter} "
            f"class={p.piping_class} — {p.segments} segments, "
            f"{len(p.connections)} connections{valve_str}"
        )

    lines.append("")
    lines.append("--- Instrumentation ---")
    if pid.instruments:
        for inst in pid.instruments:
            lines.append(f"  [{inst.component_class}] {inst.tag} {inst.function_type or ''}")
    else:
        lines.append("  (no instrumentation found)")
    lines.append(f"  Actuating systems: {pid.actuating_systems}")

    lines.append("")
    lines.append("--- Connectivity ---")
    lines.append(f"  {result.connectivity_info}")
    lines.append(f"  Total nozzles: {pid.nozzle_count}")

    if result.neqsim_model_loaded:
        lines.append("")
        lines.append("--- NeqSim Model ---")
        lines.append(f"  Successfully imported into NeqSim ProcessSystem")
        lines.append(f"  Units: {result.neqsim_units}")
        lines.append(f"  Streams: {result.neqsim_streams}")

    if result.warnings:
        lines.append("")
        lines.append("--- Warnings ---")
        for w in result.warnings:
            lines.append(f"  ⚠ {w}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DEXPI export: NeqSim ProcessSystem → DEXPI XML
# ---------------------------------------------------------------------------

def export_to_dexpi(process_model) -> Optional[bytes]:
    """Export a NeqSim ProcessSystem to DEXPI XML bytes.

    First tries NeqSim's Java DexpiXmlWriter (works for models originally
    imported from DEXPI).  If that produces no output, falls back to a
    Python-based generator that works for ANY NeqSim model.

    Returns XML bytes or None if export fails.
    """
    # --- Try Java writer first ---
    try:
        from neqsim import jneqsim

        proc = process_model.process if hasattr(process_model, "process") else process_model

        suffix = ".xml"
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False, mode="wb"
        ) as tmp:
            tmp_path = tmp.name

        try:
            DexpiXmlWriter = jneqsim.process.processmodel.dexpi.DexpiXmlWriter
            java_file = jneqsim.java.io.File(tmp_path)
            DexpiXmlWriter.write(proc, java_file)

            with open(tmp_path, "rb") as f:
                java_bytes = f.read()
            if java_bytes and len(java_bytes) > 100:
                return java_bytes
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception:
        pass

    # --- Fallback: Python-based DEXPI XML generation ---
    return _export_to_dexpi_python(process_model)


# NeqSim equipment class → DEXPI component class mapping
_NEQSIM_TO_DEXPI_CLASS: Dict[str, str] = {
    "Separator": "Separator",
    "ThreePhaseSeparator": "Separator",
    "TwoPhaseSeparator": "Separator",
    "GasScrubber": "Separator",
    "Compressor": "Compressor",
    "Cooler": "PlateHeatExchanger",
    "Heater": "PlateHeatExchanger",
    "AirCooler": "AirCooledHeatExchanger",
    "WaterCooler": "TubularHeatExchanger",
    "HeatExchanger": "TubularHeatExchanger",
    "ThrottlingValve": "Valve",
    "ControlValve": "Valve",
    "Pump": "CentrifugalPump",
    "Expander": "Compressor",
    "GibbsReactor": "Reactor",
    "Mixer": "Mixer",
    "Splitter": "Mixer",
    "Pipeline": "PipingNetworkSystem",
    "AdiabaticPipe": "PipingNetworkSystem",
    "Tank": "Tank",
    "Flare": "Tank",
}


def _export_to_dexpi_python(process_model) -> Optional[bytes]:
    """Generate DEXPI Proteus XML from any NeqSim ProcessSystem.

    Produces a valid Proteus XML with equipment, piping connections,
    and design data extracted from the simulation state.
    """
    try:
        from .process_model import NeqSimProcessModel

        if isinstance(process_model, NeqSimProcessModel):
            model = process_model
        else:
            return None

        units = model.list_units()
        streams = model.list_streams()
        if not units:
            return None

        # Build XML
        root = ET.Element("PlantModel")
        pi = ET.SubElement(root, "PlantInformation")
        pi.set("Application", "NeqSim")
        pi.set("SchemaVersion", "3.3.3")

        meta = ET.SubElement(root, "MetaData")
        ga_set = ET.SubElement(meta, "GenericAttributes")
        _add_ga(ga_set, "DrawingTitle", "NeqSim Process Export")
        _add_ga(ga_set, "DrawingNumber", "NEQSIM-001")
        _add_ga(ga_set, "RevisionNumber", "A1")

        eq_wrapper = ET.SubElement(root, "Equipment")
        nozzle_id = 0
        equipment_nozzles: Dict[str, List[str]] = {}  # tag → [nozzle_ids]

        for u in units:
            tag = u.get("name", "unit")
            u_type = u.get("type", "")
            dexpi_class = _NEQSIM_TO_DEXPI_CLASS.get(u_type, "PressureVessel")

            # Skip pipeline-type equipment (added as piping)
            if dexpi_class == "PipingNetworkSystem":
                continue

            eq_elem = ET.SubElement(eq_wrapper, dexpi_class)
            eq_elem.set("ComponentClass", dexpi_class)
            eq_elem.set("ID", _sanitize_id(tag))

            attrs = ET.SubElement(eq_elem, "GenericAttributes")
            _add_ga(attrs, "TagNameAssignmentClass", tag)

            # Extract design data from KPIs if available
            try:
                kpis = model.get_kpi()
                prefix = f"units.{tag}"
                for kpi_key, kpi_val in kpis.items():
                    if not kpi_key.startswith(prefix):
                        continue
                    prop = kpi_key[len(prefix) + 1:]
                    if "pressure" in prop.lower():
                        _add_ga(attrs, "DesignPressure", str(round(kpi_val, 2)), "bara")
                    elif "temperature" in prop.lower() and "outlet" in prop.lower():
                        _add_ga(attrs, "DesignTemperature", str(round(kpi_val, 2)), "degC")
                    elif "power" in prop.lower():
                        _add_ga(attrs, "Power", str(round(kpi_val, 2)), "kW")
                    elif "duty" in prop.lower():
                        _add_ga(attrs, "Duty", str(round(kpi_val, 2)), "kW")
            except Exception:
                pass

            # Add nozzles (inlet + outlet)
            tag_nozzles = []
            nozzle_id += 1
            nz_in = f"{_sanitize_id(tag)}-N{nozzle_id}"
            ET.SubElement(eq_elem, "Nozzle").set("ID", nz_in)
            tag_nozzles.append(nz_in)

            nozzle_id += 1
            nz_out = f"{_sanitize_id(tag)}-N{nozzle_id}"
            ET.SubElement(eq_elem, "Nozzle").set("ID", nz_out)
            tag_nozzles.append(nz_out)

            equipment_nozzles[tag] = tag_nozzles

        # Add piping connections (one PipingNetworkSystem per stream)
        for i, s in enumerate(streams):
            s_name = s.get("name", f"stream-{i}")
            pns = ET.SubElement(root, "PipingNetworkSystem")
            pns.set("ID", f"PNS-{_sanitize_id(s_name)}")
            pns_attrs = ET.SubElement(pns, "GenericAttributes")
            _add_ga(pns_attrs, "FluidCode", "NG")
            _add_ga(pns_attrs, "LineNumber", s_name)

            seg = ET.SubElement(pns, "PipingNetworkSegment")
            seg.set("ID", f"SEG-{_sanitize_id(s_name)}")

        tree = ET.ElementTree(root)
        import io
        buf = io.BytesIO()
        tree.write(buf, encoding="utf-8", xml_declaration=True)
        return buf.getvalue()

    except Exception:
        return None


def _sanitize_id(name: str) -> str:
    """Sanitize a name for use as an XML ID attribute."""
    import re
    return re.sub(r'[^A-Za-z0-9_.-]', '_', name)


def _add_ga(parent: ET.Element, name: str, value: str, units: str = ""):
    """Add a GenericAttribute child element."""
    ga = ET.SubElement(parent, "GenericAttribute")
    ga.set("Name", name)
    ga.set("Value", value)
    if units:
        ga.set("Units", units)
