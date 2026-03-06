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
import re
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
            # Use NeqSim's DexpiXmlReader
            reader = jneqsim.processSimulation.processSystem.dexpi.DexpiXmlReader()
            process_system = reader.read(tmp_path, template_stream)

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
# Fallback: Build a NeqSim process from parsed DEXPI topology
# ---------------------------------------------------------------------------

def create_neqsim_process_from_dexpi(
    pid_summary: DexpiPIDSummary,
    fluid_spec: Optional[Dict] = None,
) -> Optional[Any]:
    """Build a NeqSim ProcessSystem programmatically from DEXPI P&ID topology.

    Creates a feed stream with a default natural gas composition and adds
    process equipment (separators, heat exchangers, compressors, pumps)
    inferred from the P&ID equipment list.  This is the fallback when the
    Java DexpiXmlReader is not available.

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
            rows = [
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
            ]
        df = pd.DataFrame(rows)
        fluid = fluid_df(df, lastIsPlusFraction=False, add_all_components=False)
        fluid.setTemperature(30.0, "C")
        fluid.setPressure(65.0, "bara")
        fluid.setTotalFlowRate(10.0, "MSm3/day")

        # --- Build process ---
        ProcessSystem = jneqsim.process.processmodel.ProcessSystem
        proc = ProcessSystem()

        # Feed stream
        Stream = jneqsim.process.equipment.stream.Stream
        feed = Stream("feed gas", fluid)
        proc.add(feed)

        prev_stream = feed

        # Sort equipment by tag for repeatable ordering
        sorted_eq = sorted(pid_summary.equipment, key=lambda e: e.tag_name or e.id)

        for eq in sorted_eq:
            cls = eq.component_class.lower()
            tag = eq.tag_name or eq.id

            if "separator" in cls or "vessel" in cls:
                Sep = jneqsim.process.equipment.separator.Separator
                sep = Sep(tag, prev_stream)
                proc.add(sep)
                # Gas outlet feeds the next unit
                gas_out = Stream(f"{tag}-gas out", sep.getGasOutStream())
                proc.add(gas_out)
                prev_stream = gas_out

            elif "heatexchanger" in cls or "cooler" in cls:
                Cooler = jneqsim.process.equipment.heatexchanger.Cooler
                cooler = Cooler(tag, prev_stream)
                cooler.setOutTemperature(273.15 + 25.0)
                proc.add(cooler)
                out = Stream(f"{tag}-out", cooler.getOutletStream())
                proc.add(out)
                prev_stream = out

            elif "compressor" in cls:
                Comp = jneqsim.process.equipment.compressor.Compressor
                comp = Comp(tag, prev_stream)
                comp.setOutletPressure(100.0)
                proc.add(comp)
                out = Stream(f"{tag}-out", comp.getOutletStream())
                proc.add(out)
                prev_stream = out

            elif "pump" in cls:
                Pump = jneqsim.process.equipment.pump.Pump
                pump = Pump(tag, prev_stream)
                pump.setOutletPressure(80.0)
                proc.add(pump)
                out = Stream(f"{tag}-out", pump.getOutletStream())
                proc.add(out)
                prev_stream = out

            elif "valve" in cls:
                Valve = jneqsim.process.equipment.valve.ThrottlingValve
                valve = Valve(tag, prev_stream)
                valve.setOutletPressure(50.0)
                proc.add(valve)
                out = Stream(f"{tag}-out", valve.getOutletStream())
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

    # Step 5: Attempt NeqSim import
    neqsim_model = None
    neqsim_units = 0
    neqsim_streams = 0
    if try_neqsim_import:
        # Try Java DexpiXmlReader first
        try:
            neqsim_model = load_dexpi_to_neqsim(xml_bytes, filename, fluid_spec)
        except Exception:
            pass

        # Fallback: build process from parsed topology
        if neqsim_model is None and pid_summary.equipment:
            try:
                neqsim_model = create_neqsim_process_from_dexpi(pid_summary, fluid_spec)
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
