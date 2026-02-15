"""
Process Model Adapter — wraps a loaded NeqSim ProcessSystem or ProcessModel.

Supports loading from:
  - .neqsim files (compressed XML, recommended)
  - .xml files (uncompressed)
  - In-memory ProcessSystem objects

Provides:
  - Introspection: list units, streams, tags, properties
  - Clone-by-reload: safe scenario isolation via file re-deserialization
  - KPI extraction: powers, duties, stream conditions, mass balance
  - JSON report access
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Ensure JVM starts with --add-opens flags for XStream / Java 17+ compat
# ---------------------------------------------------------------------------

# JAVA_TOOL_OPTIONS is picked up by JNI_CreateJavaVM regardless of who starts
# the JVM (our monkey-patch, neqsim, or another library).  Setting it early
# guarantees the flags are present even when the JVM is already running by the
# time _patch_jvm_startup() executes.
_ADD_OPENS = (
    "--add-opens=java.base/java.util=ALL-UNNAMED "
    "--add-opens=java.base/java.lang=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
    "--add-opens=java.base/java.io=ALL-UNNAMED"
)
_existing = os.environ.get("JAVA_TOOL_OPTIONS", "")
if "add-opens" not in _existing:
    os.environ["JAVA_TOOL_OPTIONS"] = (
        f"{_existing} {_ADD_OPENS}".strip() if _existing else _ADD_OPENS
    )


def _patch_jvm_startup():
    """
    Monkey-patch ``jpype.startJVM`` so that ``--add-opens`` flags are injected
    *before* the JVM is created (neqsim triggers JVM start on import).

    Belt-and-suspenders alongside the JAVA_TOOL_OPTIONS env var above.
    """
    try:
        import jpype
        if jpype.isJVMStarted():
            return                                     # too late – JVM already up

        _real = jpype.startJVM

        def _start_with_opens(*args, **kwargs):
            opens = [
                "--add-opens=java.base/java.util=ALL-UNNAMED",
                "--add-opens=java.base/java.lang=ALL-UNNAMED",
                "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
                "--add-opens=java.base/java.io=ALL-UNNAMED",
            ]
            _real(*args, *opens, **kwargs)

        jpype.startJVM = _start_with_opens
    except Exception:
        pass  # best-effort; the converter workaround below handles the rest


_patch_jvm_startup()          # runs once at module-import time


# ---------------------------------------------------------------------------
# Data classes for structured results
# ---------------------------------------------------------------------------

@dataclass
class KPI:
    name: str
    value: float
    unit: str


@dataclass
class ConstraintStatus:
    name: str
    status: str      # "OK" | "WARN" | "VIOLATION" | "UNKNOWN"
    detail: str


@dataclass
class ModelRunResult:
    kpis: Dict[str, KPI]
    constraints: List[ConstraintStatus]
    json_report: Optional[dict] = None
    raw: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Unit / stream info for display and LLM tag resolution
# ---------------------------------------------------------------------------

@dataclass
class UnitInfo:
    name: str
    unit_type: str
    java_class: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamInfo:
    name: str
    temperature_C: Optional[float] = None
    pressure_bara: Optional[float] = None
    flow_rate_kg_hr: Optional[float] = None
    flow_rate_mol_sec: Optional[float] = None


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------

class NeqSimProcessModel:
    """
    Wraps a NeqSim ProcessSystem loaded from a .neqsim file.
    
    Provides introspection, cloning, and scenario execution capabilities
    for the chat + what-if engine.
    """

    def __init__(self, process_system, source_bytes: Optional[bytes] = None):
        """
        Args:
            process_system: A NeqSim ProcessSystem Java object.
            source_bytes: Original file bytes for clone-by-reload.
        """
        self._proc = process_system
        self._source_bytes = source_bytes
        self._units: Dict[str, Any] = {}
        self._streams: Dict[str, Any] = {}
        self._index_model_objects()

    # ----- Factory methods -----

    @staticmethod
    def _deserialize_xml_string(xml_string: str):
        """Deserialize a NeqSim object from an XML string using XStream.

        Tries multiple strategies combining two axes:

        - **Converter**: custom ``ReflectionConverter`` at priority -5
          (bypasses broken ``readObject``/``writeObject``) vs. plain default
          converter stack.
        - **Reference mode**: XStream's default XPath-relative references
          vs. ``ID_REFERENCES`` (numeric ``reference="9"`` style used by
          many NeqSim-saved files).

        Returns the first successfully deserialized object.
        """
        import jpype

        XStream = jpype.JClass("com.thoughtworks.xstream.XStream")
        AnyTypePermission = jpype.JClass(
            "com.thoughtworks.xstream.security.AnyTypePermission"
        )
        ReflectionConverter = jpype.JClass(
            "com.thoughtworks.xstream.converters.reflection.ReflectionConverter"
        )

        # XStream mode constants
        ID_REFERENCES = int(XStream.ID_REFERENCES)
        # Default mode is XPath-relative (no explicit setMode needed)

        strategies = [
            # (use_custom_converter, use_id_references)
            (True,  False),   # custom converter + default XPath refs
            (True,  True),    # custom converter + numeric ID refs
            (False, True),    # plain XStream  + numeric ID refs
            (False, False),   # plain XStream  + default XPath refs
        ]

        last_err = None
        for use_custom, use_id_refs in strategies:
            try:
                xstream = XStream()
                xstream.addPermission(AnyTypePermission.ANY)
                xstream.ignoreUnknownElements()
                if use_id_refs:
                    xstream.setMode(ID_REFERENCES)
                if use_custom:
                    rc = ReflectionConverter(
                        xstream.getMapper(), xstream.getReflectionProvider()
                    )
                    xstream.registerConverter(rc, -5)
                return xstream.fromXML(xml_string)
            except Exception as e:
                last_err = e

        # All strategies exhausted — raise the last error
        raise last_err

    @classmethod
    def from_file(cls, filepath: str) -> "NeqSimProcessModel":
        """Load a ProcessSystem from a .neqsim or .xml file."""
        import zipfile
        import neqsim
        from neqsim import jneqsim

        with open(filepath, "rb") as f:
            file_bytes = f.read()

        loaded = None
        is_zip = zipfile.is_zipfile(filepath)
        ext = os.path.splitext(filepath)[1].lower()
        errors_seen: list = []  # collect errors for diagnostics

        if ext in (".neqsim", ".zip") or ext not in (".xml",):
            # Try the library's Java-based ZIP reader first
            try:
                loaded = neqsim.open_neqsim(filepath)
            except Exception as e:
                errors_seen.append(f"open_neqsim: {e}")
                loaded = None

            # Fallback: extract XML from ZIP in Python (avoids Java stream issues)
            if loaded is None and is_zip:
                try:
                    with zipfile.ZipFile(filepath, "r") as zf:
                        # Look for process.xml or any .xml inside
                        xml_name = None
                        for name in zf.namelist():
                            if name.lower().endswith(".xml"):
                                xml_name = name
                                break
                        if xml_name:
                            xml_content = zf.read(xml_name).decode("utf-8")
                            loaded = cls._deserialize_xml_string(xml_content)
                        else:
                            errors_seen.append("ZIP contains no .xml file")
                except Exception as e:
                    errors_seen.append(f"ZIP XML deserialization: {e}")
                    loaded = None

        # Plain XML fallback — only makes sense for non-ZIP files
        if loaded is None and not is_zip:
            try:
                loaded = neqsim.open_xml(filepath)
            except Exception as e:
                errors_seen.append(f"open_xml: {e}")

        if loaded is None:
            detail = "\n".join(errors_seen) if errors_seen else "All loaders returned None"
            raise RuntimeError(
                f"Failed to load process model.\n\n"
                f"Tried {len(errors_seen)} loading method(s):\n{detail}"
            )

        # Run to initialize internal state.
        # Complex processes with recycles/mixers that reference downstream
        # streams may need multiple runs to converge after deserialization.
        cls._run_until_converged(loaded)
        return cls(loaded, source_bytes=file_bytes)

    @staticmethod
    def _run_until_converged(proc, max_runs: int = 5, timeout_ms: int = 180000):
        """
        Run the process repeatedly until convergence or *max_runs*.

        After XStream deserialization, recycle loops and implicit back-
        connections (mixers referencing downstream streams) may not converge
        in a single pass.  Strategy:

        1. Before the first run, reset all Recycle units so stale convergence
           flags from serialisation do not short-circuit the iteration logic.
        2. Run the process (threaded or synchronous).
        3. If total |power| + |duty| across energy-consuming units is still
           effectively zero, reset Recycles again and retry.
        4. On the 3rd attempt, try ``runSequential()`` as a fallback —
           it runs each unit block in strict order which sometimes helps
           complex topologies converge.
        """
        _POWER_UNITS = {"Compressor", "Pump", "ESPPump", "Expander", "GasTurbine"}
        _DUTY_UNITS  = {"Cooler", "Heater", "HeatExchanger", "AirCooler", "WaterCooler",
                        "MultiStreamHeatExchanger"}

        def _reset_recycles(units):
            """Reset convergence state on every Recycle unit."""
            for u in units:
                try:
                    if str(u.getClass().getSimpleName()) == "Recycle":
                        if hasattr(u, "resetIterations"):
                            u.resetIterations()
                        if hasattr(u, "resetAccelerationState"):
                            u.resetAccelerationState()
                        if hasattr(u, "setTolerance"):
                            u.setTolerance(1.0e-4)
                except Exception:
                    pass

        def _check_energy(units):
            """Return (has_energy_unit, total_energy_W)."""
            total = 0.0
            has = False
            for u in units:
                uclass = str(u.getClass().getSimpleName())
                if uclass in _POWER_UNITS:
                    has = True
                    try:
                        total += abs(float(u.getPower()))
                    except Exception:
                        pass
                elif uclass in _DUTY_UNITS:
                    has = True
                    try:
                        total += abs(float(u.getDuty()))
                    except Exception:
                        pass
            return has, total

        try:
            units = list(proc.getUnitOperations())
        except Exception:
            units = []

        # Simple process — one run is enough
        if len(units) <= 2:
            try:
                if timeout_ms > 0:
                    thread = proc.runAsThread()
                    thread.join(timeout_ms)
                    if thread.isAlive():
                        thread.interrupt()
                        thread.join()
                else:
                    proc.run()
            except Exception:
                pass
            return

        # Reset recycles before the very first run
        _reset_recycles(units)

        for attempt in range(max_runs):
            try:
                if attempt >= 3 and hasattr(proc, "runSequential"):
                    # Fallback: strict sequential execution
                    proc.runSequential()
                elif timeout_ms > 0:
                    thread = proc.runAsThread()
                    thread.join(timeout_ms)
                    if thread.isAlive():
                        thread.interrupt()
                        thread.join()
                        break  # timed out — stop retrying
                else:
                    proc.run()
            except Exception:
                pass

            has_energy, total_energy = _check_energy(units)

            if not has_energy or total_energy > 1.0:
                break  # converged (non-zero energy or no energy units)

            # Still zero — reset recycles and try again
            _reset_recycles(units)

    @classmethod
    def from_bytes(cls, file_bytes: bytes, filename: str = "process.neqsim") -> "NeqSimProcessModel":
        """Load a ProcessSystem from in-memory bytes (e.g. Streamlit file_uploader)."""
        ext = os.path.splitext(filename)[1].lower()
        suffix = ext if ext else ".neqsim"
        
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            model = cls.from_file(tmp_path)
            model._source_bytes = file_bytes
            return model
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    @classmethod
    def from_process_system(cls, process_system) -> "NeqSimProcessModel":
        """Wrap an existing ProcessSystem object (e.g. built in code)."""
        import neqsim

        # Serialize to bytes for cloning
        with tempfile.NamedTemporaryFile(suffix=".neqsim", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            neqsim.save_neqsim(process_system, tmp_path)
            with open(tmp_path, "rb") as f:
                file_bytes = f.read()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        return cls(process_system, source_bytes=file_bytes)

    # ----- Cloning -----

    def refresh_source_bytes(self):
        """Re-serialize the current process state so future clones see any
        structural modifications (added units, streams, etc.)."""
        import neqsim

        with tempfile.NamedTemporaryFile(suffix=".neqsim", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            neqsim.save_neqsim(self._proc, tmp_path)
            with open(tmp_path, "rb") as f:
                self._source_bytes = f.read()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def clone(self) -> "NeqSimProcessModel":
        """
        Create an independent copy by re-deserializing from the original bytes.
        This gives a fully isolated process for scenario runs.
        """
        if self._source_bytes is None:
            raise RuntimeError(
                "Cannot clone: no source bytes available. "
                "Load from file or use from_process_system() to enable cloning."
            )
        return NeqSimProcessModel.from_bytes(self._source_bytes)

    # ----- Introspection -----

    def _index_model_objects(self):
        """Discover all unit operations and streams in the process."""
        self._units.clear()
        self._streams.clear()

        proc = self._proc

        # Get all unit operations
        try:
            units = list(proc.getUnitOperations())
        except Exception:
            try:
                units = list(proc.getUnitOperationList())
            except Exception:
                units = []

        for u in units:
            try:
                name = str(u.getName()) if u.getName() else None
                if name:
                    self._units[name] = u
            except Exception:
                pass

        # Discover streams from unit in/out connections.
        # Always use qualified keys ("unitName.streamName") as primary to
        # guarantee stable KPI comparisons across base vs scenario runs.
        # Also add short aliases for stream names that are globally unique.
        seen_java_ids = set()  # track Java object identity to skip duplicates
        raw_name_count: Dict[str, int] = {}  # count how many units produce same stream name

        for u in units:
            try:
                uname = str(u.getName()) if u.getName() else "unknown"
            except Exception:
                uname = "unknown"

            # Only index OUTLET streams — inlet streams are always the
            # same Java object as a prior unit's outlet, so indexing them
            # would double-count or create key collisions within a unit
            # (e.g. compressor inlet and outlet both named "gasOutStream").
            for method_name in (
                "getOutletStream", "getOutStream",
                "getGasOutStream", "getOilOutStream",
                "getLiquidOutStream", "getWaterOutStream",
                "getSplitStream",
            ):
                if hasattr(u, method_name):
                    try:
                        if method_name == "getSplitStream":
                            for i in range(10):
                                try:
                                    s = u.getSplitStream(i)
                                    if s is not None:
                                        sname = str(s.getName()) if s.getName() else None
                                        if sname:
                                            try:
                                                java_id = int(s.hashCode())
                                            except Exception:
                                                java_id = id(s)
                                            if java_id in seen_java_ids:
                                                continue  # same Java object already indexed
                                            seen_java_ids.add(java_id)
                                            key = f"{uname}.{sname}"
                                            if key not in self._streams:
                                                self._streams[key] = s
                                            raw_name_count[sname] = raw_name_count.get(sname, 0) + 1
                                except Exception:
                                    break
                        else:
                            s = getattr(u, method_name)()
                            if s is not None:
                                sname = str(s.getName()) if s.getName() else None
                                if sname:
                                    try:
                                        java_id = int(s.hashCode())
                                    except Exception:
                                        java_id = id(s)
                                    if java_id in seen_java_ids:
                                        continue  # same Java object already indexed
                                    seen_java_ids.add(java_id)
                                    key = f"{uname}.{sname}"
                                    self._streams[key] = s
                                    raw_name_count[sname] = raw_name_count.get(sname, 0) + 1
                    except Exception:
                        pass

        # Also index units that are streams themselves (Stream objects added to process)
        for name, u in list(self._units.items()):
            try:
                java_class = str(u.getClass().getSimpleName())
                if "Stream" in java_class and name not in self._streams:
                    self._streams[name] = u
                    raw_name_count[name] = raw_name_count.get(name, 0) + 1
            except Exception:
                pass

        # Add short (unqualified) aliases for globally unique stream names
        # so users / LLM can reference them with short names.
        unique_streams = {sname for sname, cnt in raw_name_count.items() if cnt == 1}
        for key, s in list(self._streams.items()):
            try:
                sname = str(s.getName()) if s.getName() else None
            except Exception:
                sname = None
            if sname and sname in unique_streams and sname not in self._streams:
                self._streams[sname] = s

    def get_process(self):
        """Return the underlying Java ProcessSystem object."""
        return self._proc

    def get_diagram_dot(
        self,
        style: str = "HYSYS",
        detail_level: str = "ENGINEERING",
        show_stream_values: bool = True,
        use_stream_tables: bool = False,
        show_control_equipment: bool = True,
        title: str = "",
    ) -> str:
        """Export the process flow diagram as a Graphviz DOT string.

        Parameters
        ----------
        style : str
            Diagram style: ``HYSYS`` (default), ``NEQSIM``, ``PROII``, or ``ASPEN_PLUS``.
        detail_level : str
            Detail level: ``CONCEPTUAL``, ``ENGINEERING`` (default), or ``DEBUG``.
        show_stream_values : bool
            Show temperature, pressure, and flow on streams.
        use_stream_tables : bool
            Use HTML table labels (True) or simple text (False).
        show_control_equipment : bool
            Show recycle/adjuster/calculator equipment.
        title : str
            Diagram title. Uses process name if empty.

        Returns
        -------
        str
            Graphviz DOT source string.
        """
        from neqsim import jneqsim

        DiagramStyle = jneqsim.process.processmodel.diagram.DiagramStyle
        DiagramDetailLevel = jneqsim.process.processmodel.diagram.DiagramDetailLevel

        style_map = {
            "HYSYS": DiagramStyle.HYSYS,
            "NEQSIM": DiagramStyle.NEQSIM,
            "PROII": DiagramStyle.PROII,
            "ASPEN_PLUS": DiagramStyle.ASPEN_PLUS,
        }
        level_map = {
            "CONCEPTUAL": DiagramDetailLevel.CONCEPTUAL,
            "ENGINEERING": DiagramDetailLevel.ENGINEERING,
            "DEBUG": DiagramDetailLevel.DEBUG,
        }

        exporter = self._proc.createDiagramExporter()
        exporter.setDiagramStyle(style_map.get(style.upper(), DiagramStyle.HYSYS))
        exporter.setDetailLevel(level_map.get(detail_level.upper(), DiagramDetailLevel.ENGINEERING))
        exporter.setShowStreamValues(show_stream_values)
        exporter.setUseStreamTables(use_stream_tables)
        exporter.setShowControlEquipment(show_control_equipment)
        if title:
            exporter.setTitle(title)

        return str(exporter.toDOT())

    # Unit types that legitimately produce power or duty
    _POWER_UNITS = {"Compressor", "Pump", "ESPPump", "Expander", "GasTurbine"}
    _DUTY_UNITS = {"Cooler", "Heater", "HeatExchanger", "AirCooler", "WaterCooler",
                   "MultiStreamHeatExchanger"}
    _HEAT_EXCHANGE_UNITS = _DUTY_UNITS  # units where outlet temperature matters

    def list_units(self) -> List[UnitInfo]:
        """List all unit operations with type info and key properties."""
        result = []
        for name, u in self._units.items():
            try:
                java_class = str(u.getClass().getSimpleName())
            except Exception:
                java_class = "Unknown"

            props = {}
            # Try to extract common properties
            for prop, getter in [
                ("power_kW", "getPower"),
                ("duty_kW", "getDuty"),
                ("isentropicEfficiency", "getIsentropicEfficiency"),
                ("polytropicEfficiency", "getPolytropicEfficiency"),
                ("outletPressure_bara", "getOutletPressure"),
            ]:
                if hasattr(u, getter):
                    try:
                        val = getattr(u, getter)()
                        if val is None:
                            continue
                        fval = float(val)
                        if prop in ("power_kW", "duty_kW"):
                            fval = fval / 1000.0  # W -> kW
                        # Fallback: if duty is 0 for a heat-exchange unit, try getEnergyInput
                        if fval == 0.0 and prop == "duty_kW" and java_class in self._DUTY_UNITS:
                            if hasattr(u, "getEnergyInput"):
                                try:
                                    fval = float(u.getEnergyInput()) / 1000.0
                                except Exception:
                                    pass
                        # Skip zero power/duty for units that don't produce them
                        if fval == 0.0 and prop == "power_kW" and java_class not in self._POWER_UNITS:
                            continue
                        if fval == 0.0 and prop == "duty_kW" and java_class not in self._DUTY_UNITS:
                            continue
                        props[prop] = fval
                    except Exception:
                        pass

            # Outlet temperature for heaters/coolers/heat exchangers
            if java_class in self._HEAT_EXCHANGE_UNITS:
                for m in ("getOutletStream", "getOutStream"):
                    if hasattr(u, m):
                        try:
                            s = getattr(u, m)()
                            if s is not None:
                                props["outTemperature_C"] = float(s.getTemperature("C"))
                                break
                        except Exception:
                            pass

            # Flow rate, T, P for Stream-type units
            if java_class == "Stream":
                try:
                    props["flow_kg_hr"] = float(u.getFlowRate("kg/hr"))
                except Exception:
                    pass
                try:
                    props["temperature_C"] = float(u.getTemperature("C"))
                except Exception:
                    pass
                try:
                    props["pressure_bara"] = float(u.getPressure("bara"))
                except Exception:
                    pass

            result.append(UnitInfo(name=name, unit_type=java_class, java_class=java_class, properties=props))
        return result

    def list_streams(self) -> List[StreamInfo]:
        """List all streams with current conditions."""
        result = []
        for name, s in self._streams.items():
            info = StreamInfo(name=name)
            try:
                info.temperature_C = float(s.getTemperature("C"))
            except Exception:
                pass
            try:
                info.pressure_bara = float(s.getPressure("bara"))
            except Exception:
                try:
                    info.pressure_bara = float(s.getPressure())
                except Exception:
                    pass
            try:
                info.flow_rate_kg_hr = float(s.getFlowRate("kg/hr"))
            except Exception:
                pass
            try:
                info.flow_rate_mol_sec = float(s.getFlowRate("mol/sec"))
            except Exception:
                pass
            result.append(info)
        return result

    def list_tags(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a tag dictionary for LLM intent resolution.
        Maps canonical paths to type + aliases.
        """
        tags = {}
        for info in self.list_units():
            tags[f"units.{info.name}"] = {
                "type": info.unit_type,
                "aliases": [info.name],
                "properties": info.properties,
            }
        for info in self.list_streams():
            tags[f"streams.{info.name}"] = {
                "type": "Stream",
                "aliases": [info.name],
                "conditions": {
                    "temperature_C": info.temperature_C,
                    "pressure_bara": info.pressure_bara,
                    "flow_rate_kg_hr": info.flow_rate_kg_hr,
                },
            }
        return tags

    # ----- Value access for scenarios -----

    def get_unit(self, name: str):
        """Get a unit operation by name. Raises KeyError if not found."""
        if name in self._units:
            return self._units[name]
        # Also try via process.getUnit()
        try:
            u = self._proc.getUnit(name)
            if u is not None:
                return u
        except Exception:
            pass
        raise KeyError(f"Unit not found: {name}")

    def get_stream(self, name: str):
        """Get a stream by name (supports qualified, unqualified, and case-insensitive names)."""
        # Exact match
        if name in self._streams:
            return self._streams[name]
        # Suffix match (e.g. "outStream" -> "intercooler.outStream")
        for key, s in self._streams.items():
            if key.endswith(f".{name}"):
                return s
        # Case-insensitive match
        name_lower = name.lower()
        for key, s in self._streams.items():
            if key.lower() == name_lower or key.lower().endswith(f".{name_lower}"):
                return s
        # Try Java getUnit — Stream units are both units and streams
        try:
            u = self._proc.getUnit(name)
            if u is not None:
                return u
        except Exception:
            pass
        raise KeyError(f"Stream not found: '{name}'. Available: {list(self._streams.keys())[:20]}")

    # ----- Run and report -----

    def run(self, timeout_ms: int = 120000) -> ModelRunResult:
        """
        Run the process and extract KPIs and constraints.
        
        Uses multiple-pass convergence for processes with recycles.
        
        Args:
            timeout_ms: Timeout in milliseconds. If >0, runs in a thread.
        """
        self._run_until_converged(self._proc, max_runs=5, timeout_ms=timeout_ms)

        # Re-index model objects after running so references are fresh
        self._index_model_objects()

        return self._extract_results()

    def _extract_results(self) -> ModelRunResult:
        """Extract KPIs, constraints, and JSON report from solved process."""
        kpis: Dict[str, KPI] = {}
        constraints: List[ConstraintStatus] = []

        # Collect power and duty from all units
        total_power_kW = 0.0
        total_duty_kW = 0.0

        for name, u in self._units.items():
            try:
                uclass = str(u.getClass().getSimpleName())
            except Exception:
                uclass = ""

            if hasattr(u, "getPower"):
                try:
                    power_kW = float(u.getPower()) / 1000.0
                    # Skip zero power for units that don't produce it
                    if power_kW == 0.0 and uclass not in self._POWER_UNITS:
                        pass
                    else:
                        kpis[f"{name}.power_kW"] = KPI(f"{name}.power_kW", power_kW, "kW")
                        total_power_kW += power_kW
                except Exception:
                    pass
            if hasattr(u, "getDuty"):
                try:
                    duty_kW = float(u.getDuty()) / 1000.0
                    # Fallback: if duty is 0 for a heat-exchange unit, try getEnergyInput
                    if duty_kW == 0.0 and uclass in self._DUTY_UNITS:
                        if hasattr(u, "getEnergyInput"):
                            try:
                                duty_kW = float(u.getEnergyInput()) / 1000.0
                            except Exception:
                                pass
                    # Skip zero duty for units that don't produce it
                    if duty_kW == 0.0 and uclass not in self._DUTY_UNITS:
                        pass
                    else:
                        kpis[f"{name}.duty_kW"] = KPI(f"{name}.duty_kW", duty_kW, "kW")
                        total_duty_kW += abs(duty_kW)
                except Exception:
                    pass

        kpis["total_power_kW"] = KPI("total_power_kW", total_power_kW, "kW")
        kpis["total_duty_kW"] = KPI("total_duty_kW", total_duty_kW, "kW")

        # Try to get JSON report
        json_report = None
        try:
            from neqsim import jneqsim
            report_obj = jneqsim.process.util.report.Report(self._proc)
            json_str = str(report_obj.generateJsonReport())
            json_report = json.loads(json_str)
        except Exception:
            try:
                json_str = str(self._proc.getReport_json())
                json_report = json.loads(json_str)
            except Exception:
                pass

        # Extract all properties from JSON report into flat KPIs
        if json_report:
            self._flatten_json_report(json_report, kpis)

        # Extract detailed unit operation properties (utilization, sizing, performance)
        self._extract_unit_properties(kpis)

        # Extract mechanical design data (wall thickness, weights, dimensions, cost)
        self._extract_mechanical_design(kpis)

        # Add convergence warning if all power/duty are zero
        if total_power_kW == 0.0 and total_duty_kW == 0.0:
            has_energy_unit = False
            energy_unit_names = []
            for name, u in self._units.items():
                try:
                    uclass = str(u.getClass().getSimpleName())
                    if uclass in self._POWER_UNITS | self._DUTY_UNITS:
                        has_energy_unit = True
                        energy_unit_names.append(f"{name} ({uclass})")
                except Exception:
                    pass
            if has_energy_unit:
                # Gather recycle error details if available
                recycle_info = []
                for name, u in self._units.items():
                    try:
                        if str(u.getClass().getSimpleName()) == "Recycle":
                            parts = [f"{name}"]
                            for prop, getter in [
                                ("errT", "getErrorTemperature"),
                                ("errF", "getErrorFlow"),
                                ("iter", "getIterations"),
                            ]:
                                if hasattr(u, getter):
                                    try:
                                        val = float(getattr(u, getter)())
                                        parts.append(f"{prop}={val:.4g}")
                                    except Exception:
                                        pass
                            recycle_info.append(" ".join(parts))
                    except Exception:
                        pass
                msg = (
                    "All power/duty values are zero — the process may not have converged. "
                    "This can happen with complex recycle loops after deserialization."
                )
                if energy_unit_names:
                    msg += f" Energy units: {', '.join(energy_unit_names[:5])}."
                if recycle_info:
                    msg += f" Recycle state: {'; '.join(recycle_info)}."
                constraints.append(ConstraintStatus("convergence", "WARN", msg))

        # Extract calculated fluid properties from streams (viscosity, Z, JT, TVP, RVP, etc.)
        self._extract_stream_fluid_properties(kpis)

        # Mass balance check — identify true terminal product streams.
        #
        # Strategy: In processes with recycles, mixers, and multiple product
        # streams, we cannot blindly count separator liquid drains as products
        # because many are recirculated back into the process.
        #
        # Instead, we detect terminal product streams by looking for explicit
        # Stream-type units added AFTER all process equipment (a common
        # NeqSim convention for marking product streams like "export gas",
        # "export oil", "fuel gas").  If none are found, we fall back to
        # the last non-utility unit's ALL outlets.
        try:
            proc = self._proc
            all_units = list(proc.getUnitOperations())
            feed_flow = 0.0
            product_flow = 0.0
            product_details = []  # for diagnostic output

            _utility_types = {"Recycle", "Adjuster", "Calculator", "SetPoint"}

            if all_units:
                # Feed flow: first unit in the process
                first = all_units[0]
                try:
                    feed_flow = float(first.getFlowRate("kg/hr"))
                except Exception:
                    for m in ("getOutletStream", "getOutStream", "getGasOutStream"):
                        if hasattr(first, m):
                            try:
                                feed_flow = float(getattr(first, m)().getFlowRate("kg/hr"))
                                break
                            except Exception:
                                pass

                # --- Detect terminal product streams ---
                # Find the last non-Stream, non-utility unit in process order.
                # Any Stream-type unit appearing AFTER it is a terminal product.
                last_equip_idx = -1
                for i, u in enumerate(all_units):
                    uclass = str(u.getClass().getSimpleName())
                    if uclass != "Stream" and uclass not in _utility_types:
                        last_equip_idx = i

                terminal_stream_units = []
                for i, u in enumerate(all_units):
                    if i > last_equip_idx and i > 0:  # skip first unit (feed)
                        uclass = str(u.getClass().getSimpleName())
                        if uclass == "Stream":
                            terminal_stream_units.append(u)

                if terminal_stream_units:
                    # Explicit terminal streams — use them as products
                    for s in terminal_stream_units:
                        try:
                            flow = float(s.getFlowRate("kg/hr"))
                            sname = str(s.getName()) if s.getName() else "product"
                            if abs(flow) > 0.01:
                                product_flow += flow
                                product_details.append(f"{sname}={flow:.0f}")
                            else:
                                # Report 0-flow terminal streams for diagnostics
                                product_details.append(f"{sname}=0 (no flow)")
                        except Exception:
                            pass
                else:
                    # Fallback: use the last non-utility unit's ALL outlets
                    last = None
                    for i in range(len(all_units) - 1, -1, -1):
                        uclass = str(all_units[i].getClass().getSimpleName())
                        if uclass not in _utility_types:
                            last = all_units[i]
                            break

                    if last is not None:
                        last_class = str(last.getClass().getSimpleName())
                        seen_outlet_ids: set = set()

                        def _add_outlet_flow(stream_obj, label: str) -> float:
                            """Add stream flow if not already counted (dedup by hashCode)."""
                            nonlocal product_flow
                            try:
                                sid = int(stream_obj.hashCode())
                            except Exception:
                                sid = id(stream_obj)
                            if sid in seen_outlet_ids:
                                return 0.0
                            seen_outlet_ids.add(sid)
                            flow = float(stream_obj.getFlowRate("kg/hr"))
                            if abs(flow) > 0.01:
                                sname = str(stream_obj.getName()) if stream_obj.getName() else label
                                product_flow += flow
                                product_details.append(f"{sname}={flow:.0f}")
                            return flow

                        # Gas outlet
                        for m in ("getOutletStream", "getOutStream", "getGasOutStream"):
                            if hasattr(last, m):
                                try:
                                    s = getattr(last, m)()
                                    _add_outlet_flow(s, "gas_out")
                                    break
                                except Exception:
                                    pass
                        # Oil outlet (three-phase separators)
                        if hasattr(last, "getOilOutStream"):
                            try:
                                _add_outlet_flow(last.getOilOutStream(), "oil")
                            except Exception:
                                pass
                        # Liquid outlet
                        if hasattr(last, "getLiquidOutStream"):
                            try:
                                _add_outlet_flow(last.getLiquidOutStream(), "liquid")
                            except Exception:
                                pass
                        # Water outlet (three-phase separators)
                        if hasattr(last, "getWaterOutStream"):
                            try:
                                _add_outlet_flow(last.getWaterOutStream(), "water")
                            except Exception:
                                pass

            # Fallback: match by stream name keywords
            if feed_flow == 0.0:
                for name, s in self._streams.items():
                    try:
                        flow = float(s.getFlowRate("kg/hr"))
                    except Exception:
                        continue
                    lower = name.lower()
                    if any(kw in lower for kw in ("feed", "inlet", "well", "input")):
                        feed_flow += flow
                    elif any(kw in lower for kw in ("export", "product", "outlet", "output", "fuel")):
                        product_flow += flow

            if feed_flow > 0:
                balance_pct = abs(feed_flow - product_flow) / feed_flow * 100
                kpis["mass_balance_pct"] = KPI("mass_balance_pct", balance_pct, "%")
                detail_str = ", ".join(product_details) if product_details else f"{product_flow:.0f}"
                status = "OK" if balance_pct < 1.0 else "WARN" if balance_pct < 5.0 else "VIOLATION"
                constraints.append(ConstraintStatus(
                    "mass_balance", status,
                    f"Feed={feed_flow:.0f} kg/hr, Products={product_flow:.0f} kg/hr ({detail_str}), imbalance={balance_pct:.2f}%"
                ))
        except Exception:
            pass

        return ModelRunResult(
            kpis=kpis,
            constraints=constraints,
            json_report=json_report,
            raw={
                "unit_names": list(self._units.keys()),
                "stream_names": list(self._streams.keys()),
            }
        )

    def _extract_unit_properties(self, kpis: Dict[str, KPI]):
        """
        Extract detailed equipment-level properties from each unit operation.

        Covers compressor performance, separator capacity, cooler/heater sizing,
        pump/valve characteristics, and general utilization metrics.
        """
        for name, u in self._units.items():
            try:
                java_class = str(u.getClass().getSimpleName())
            except Exception:
                continue

            prefix = f"{name}"

            # ---------- Compressor ----------
            if java_class in ("Compressor",):
                for prop, getter, unit in [
                    ("polytropicHead_kJkg", "getPolytropicHead", "kJ/kg"),
                    ("polytropicHeadMeter", "getPolytropicHeadMeter", "m"),
                    ("polytropicExponent", "getPolytropicExponent", "[-]"),
                    ("compressionRatio", "getCompressionRatio", "[-]"),
                    ("actualCompressionRatio", "getActualCompressionRatio", "[-]"),
                    ("inletTemperature_K", "getInletTemperature", "K"),
                    ("outletTemperature_K", "getOutletTemperature", "K"),
                    ("inletPressure_bara", "getInletPressure", "bara"),
                    ("speed_rpm", "getSpeed", "rpm"),
                    ("maxSpeed_rpm", "getMaximumSpeed", "rpm"),
                    ("minSpeed_rpm", "getMinimumSpeed", "rpm"),
                    ("distanceToSurge", "getDistanceToSurge", "[-]"),
                    ("surgeFlowRate", "getSurgeFlowRate", "m3/hr"),
                    ("maxUtilization", "getMaxUtilization", "[-]"),
                    ("maxUtilizationPercent", "getMaxUtilizationPercent", "%"),
                ]:
                    if hasattr(u, getter):
                        try:
                            val = float(getattr(u, getter)())
                            kpis[f"{prefix}.{prop}"] = KPI(f"{prefix}.{prop}", val, unit)
                        except Exception:
                            pass
                # Entropy production & exergy
                try:
                    kpis[f"{prefix}.entropyProduction_JK"] = KPI(
                        f"{prefix}.entropyProduction_JK",
                        float(u.getEntropyProduction("J/K")), "J/K"
                    )
                except Exception:
                    pass
                try:
                    kpis[f"{prefix}.exergyChange_J"] = KPI(
                        f"{prefix}.exergyChange_J",
                        float(u.getExergyChange("J", 288.15)), "J"
                    )
                except Exception:
                    pass

            # ---------- Separator / Scrubber ----------
            elif "Separator" in java_class or "Scrubber" in java_class:
                for prop, getter, unit in [
                    ("gasLoadFactor", "getGasLoadFactor", "m/s"),
                    ("designGasLoadFactor", "getDesignGasLoadFactor", "m/s"),
                    ("gasSuperficialVelocity", "getGasSuperficialVelocity", "m/s"),
                    ("maxAllowableGasVelocity", "getMaxAllowableGasVelocity", "m/s"),
                    ("liquidLevel", "getLiquidLevel", "m"),
                    ("designLiquidLevel", "getDesignLiquidLevelFraction", "[-]"),
                    ("gasCarryunderFraction", "getGasCarryunderFraction", "[-]"),
                    ("liquidCarryoverFraction", "getLiquidCarryoverFraction", "[-]"),
                    ("internalDiameter_m", "getInternalDiameter", "m"),
                    ("separatorLength_m", "getSeparatorLength", "m"),
                    ("efficiency", "getEfficiency", "[-]"),
                    ("maxUtilization", "getMaxUtilization", "[-]"),
                    ("maxUtilizationPercent", "getMaxUtilizationPercent", "%"),
                ]:
                    if hasattr(u, getter):
                        try:
                            val = float(getattr(u, getter)())
                            kpis[f"{prefix}.{prop}"] = KPI(f"{prefix}.{prop}", val, unit)
                        except Exception:
                            pass

            # ---------- Cooler / Heater / HeatExchanger ----------
            elif java_class in ("Cooler", "Heater", "HeatExchanger", "AirCooler", "WaterCooler"):
                for prop, getter, unit in [
                    ("pressureDrop_bar", "getPressureDrop", "bar"),
                    ("inletTemperature_K", "getInletTemperature", "K"),
                    ("outletTemperature_K", "getOutletTemperature", "K"),
                    ("inletPressure_bara", "getInletPressure", "bara"),
                    ("outletPressure_bara", "getOutletPressure", "bara"),
                    ("maxDesignDuty_W", "getMaxDesignDuty", "W"),
                    ("energyInput_W", "getEnergyInput", "W"),
                ]:
                    if hasattr(u, getter):
                        try:
                            val = float(getattr(u, getter)())
                            kpis[f"{prefix}.{prop}"] = KPI(f"{prefix}.{prop}", val, unit)
                        except Exception:
                            pass
                # UA value for HeatExchanger
                if java_class == "HeatExchanger" and hasattr(u, "getUAvalue"):
                    try:
                        kpis[f"{prefix}.UAvalue"] = KPI(
                            f"{prefix}.UAvalue", float(u.getUAvalue()), "W/K"
                        )
                    except Exception:
                        pass

            # ---------- Pump / ESPPump ----------
            elif java_class in ("Pump", "ESPPump"):
                for prop, getter, unit in [
                    ("inletPressure_bara", "getInletPressure", "bara"),
                    ("outletPressure_bara", "getOutletPressure", "bara"),
                    ("efficiency", "getIsentropicEfficiency", "[-]"),
                    ("head_m", "getHead", "m"),
                ]:
                    if hasattr(u, getter):
                        try:
                            val = float(getattr(u, getter)())
                            kpis[f"{prefix}.{prop}"] = KPI(f"{prefix}.{prop}", val, unit)
                        except Exception:
                            pass

            # ---------- Valve ----------
            elif "Valve" in java_class:
                for prop, getter, unit in [
                    ("outletPressure_bara", "getOutletPressure", "bara"),
                    ("pressureDrop_bar", "getPressureDrop", "bar"),
                ]:
                    if hasattr(u, getter):
                        try:
                            val = float(getattr(u, getter)())
                            kpis[f"{prefix}.{prop}"] = KPI(f"{prefix}.{prop}", val, unit)
                        except Exception:
                            pass
                # Cv
                if hasattr(u, "getCv"):
                    try:
                        kpis[f"{prefix}.Cv"] = KPI(
                            f"{prefix}.Cv", float(u.getCv()), "[-]"
                        )
                    except Exception:
                        pass

            # ---------- Splitter ----------
            elif "Splitter" in java_class:
                if hasattr(u, "getSplitStream"):
                    for j in range(10):
                        try:
                            s = u.getSplitStream(j)
                            if s is not None:
                                flow = float(s.getFlowRate("kg/hr"))
                                kpis[f"{prefix}.splitStream{j}_flow_kg_hr"] = KPI(
                                    f"{prefix}.splitStream{j}_flow_kg_hr", flow, "kg/hr"
                                )
                        except Exception:
                            break

            # ---------- Recycle ----------
            elif java_class == "Recycle":
                for prop, getter, unit in [
                    ("errorTemperature", "getErrorTemperature", "K"),
                    ("errorPressure", "getErrorPressure", "bara"),
                    ("errorFlow", "getErrorFlow", "[-]"),
                    ("errorComposition", "getErrorComposition", "[-]"),
                    ("iterations", "getIterations", "[-]"),
                ]:
                    if hasattr(u, getter):
                        try:
                            val = float(getattr(u, getter)())
                            kpis[f"{prefix}.{prop}"] = KPI(f"{prefix}.{prop}", val, unit)
                        except Exception:
                            pass

            # ---------- Sizing report (all equipment) ----------
            if hasattr(u, "getSizingReportJson"):
                try:
                    sizing_json = str(u.getSizingReportJson())
                    sizing = json.loads(sizing_json)
                    if isinstance(sizing, dict):
                        for sk, sv in sizing.items():
                            if isinstance(sv, (int, float)) and sk != "equipmentName":
                                kpis[f"{prefix}.sizing.{sk}"] = KPI(
                                    f"{prefix}.sizing.{sk}", float(sv), ""
                                )
                except Exception:
                    pass

    def _extract_mechanical_design(self, kpis: Dict[str, KPI]):
        """
        Extract mechanical design data from each unit operation.

        Reads existing mechanical design data that was set during explicit
        ``autoSize()`` calls.  Does **not** call ``initMechanicalDesign()``
        or ``calcDesign()`` — those would recalculate dimensions from
        current operating conditions and silently overwrite the frozen
        auto-sized design (e.g. changing a valve opening would change
        its inner diameter and weight).

        Also runs SystemMechanicalDesign on the entire process to get:
        - Total weight, plot space (footprint), total volume
        - Weight breakdown by equipment type and discipline
        - Equipment count by type
        - Total power, cooling/heating duty summaries
        """
        from neqsim import jneqsim
        import math

        # --- Per-unit mechanical design ---
        for name, u in self._units.items():
            if not hasattr(u, 'getMechanicalDesign'):
                continue

            prefix = name
            try:
                # Read existing mechanical design — do NOT re-initialise or
                # recalculate so that auto-sized values are preserved.
                md = u.getMechanicalDesign()
                if md is None:
                    continue

                # Wall thickness
                for prop, getter, unit in [
                    ("mechDesign.wallThickness_mm", "getWallThickness", "mm"),
                    ("mechDesign.innerDiameter_m", "getInnerDiameter", "m"),
                    ("mechDesign.outerDiameter_m", "getOuterDiameter", "m"),
                    ("mechDesign.tantanLength_m", "getTantanLength", "m"),
                ]:
                    if hasattr(md, getter):
                        try:
                            val = float(getattr(md, getter)())
                            if math.isnan(val) or val == 0.0:
                                continue
                            # Convert wall thickness from m to mm
                            if "wallThickness" in prop:
                                val = val * 1000.0
                            kpis[f"{prefix}.{prop}"] = KPI(f"{prefix}.{prop}", val, unit)
                        except Exception:
                            pass

                # Weights
                for prop, getter, unit in [
                    ("mechDesign.weightTotal_kg", "getWeightTotal", "kg"),
                    ("mechDesign.weightVesselShell_kg", "getWeigthVesselShell", "kg"),
                    ("mechDesign.weightInternals_kg", "getWeigthInternals", "kg"),
                    ("mechDesign.weightPiping_kg", "getWeightPiping", "kg"),
                    ("mechDesign.weightNozzles_kg", "getWeightNozzle", "kg"),
                    ("mechDesign.weightStructuralSteel_kg", "getWeightStructualSteel", "kg"),
                    ("mechDesign.weightElectroInstrument_kg", "getWeightElectroInstrument", "kg"),
                    ("mechDesign.weightVessel_kg", "getWeightVessel", "kg"),
                ]:
                    if hasattr(md, getter):
                        try:
                            val = float(getattr(md, getter)())
                            if math.isnan(val) or val == 0.0:
                                continue
                            kpis[f"{prefix}.{prop}"] = KPI(f"{prefix}.{prop}", val, unit)
                        except Exception:
                            pass

                # Module dimensions (space/footprint)
                for prop, getter, unit in [
                    ("mechDesign.moduleLength_m", "getModuleLength", "m"),
                    ("mechDesign.moduleWidth_m", "getModuleWidth", "m"),
                    ("mechDesign.moduleHeight_m", "getModuleHeight", "m"),
                    ("mechDesign.totalVolume_m3", "getVolumeTotal", "m3"),
                ]:
                    if hasattr(md, getter):
                        try:
                            val = float(getattr(md, getter)())
                            if math.isnan(val) or val == 0.0:
                                continue
                            kpis[f"{prefix}.{prop}"] = KPI(f"{prefix}.{prop}", val, unit)
                        except Exception:
                            pass

                # Design pressures and temperatures
                for prop, getter, unit in [
                    ("mechDesign.maxDesignPressure_bara", "getMaxDesignPressure", "bara"),
                    ("mechDesign.minDesignPressure_bara", "getMinDesignPressure", "bara"),
                    ("mechDesign.maxDesignTemperature_C", "getMaxDesignTemperatureLimit", "C"),
                    ("mechDesign.minDesignTemperature_C", "getMinDesignTemperatureLimit", "C"),
                    ("mechDesign.maxOperatingPressure_bara", "getMaxOperationPressure", "bara"),
                    ("mechDesign.maxOperatingTemperature_C", "getMaxOperationTemperature", "C"),
                ]:
                    if hasattr(md, getter):
                        try:
                            val = float(getattr(md, getter)())
                            if math.isnan(val) or val == 0.0:
                                continue
                            kpis[f"{prefix}.{prop}"] = KPI(f"{prefix}.{prop}", val, unit)
                        except Exception:
                            pass

                # Material properties
                for prop, getter, unit in [
                    ("mechDesign.maxAllowableStress_Pa", "getMaxAllowableStress", "Pa"),
                    ("mechDesign.tensileStrength_Pa", "getTensileStrength", "Pa"),
                    ("mechDesign.jointEfficiency", "getJointEfficiency", "[-]"),
                    ("mechDesign.corrosionAllowance_m", "getCorrosionAllowance", "m"),
                ]:
                    if hasattr(md, getter):
                        try:
                            val = float(getattr(md, getter)())
                            if math.isnan(val) or val == 0.0:
                                continue
                            kpis[f"{prefix}.{prop}"] = KPI(f"{prefix}.{prop}", val, unit)
                        except Exception:
                            pass

                # Construction material (string value stored as special KPI)
                if hasattr(md, 'getConstrutionMaterial'):
                    try:
                        mat = str(md.getConstrutionMaterial())
                        if mat and mat != 'null' and mat != 'None':
                            # Store as a "string KPI" with value 0 and unit = material name
                            kpis[f"{prefix}.mechDesign.material"] = KPI(
                                f"{prefix}.mechDesign.material", 0.0, mat
                            )
                    except Exception:
                        pass

                # Cost estimation per unit
                if hasattr(md, 'getCostEstimate'):
                    try:
                        ce = md.getCostEstimate()
                        if ce is not None:
                            cost = float(ce.getTotalCost())
                            if cost > 0:
                                kpis[f"{prefix}.cost.totalCost_USD"] = KPI(
                                    f"{prefix}.cost.totalCost_USD", cost, "USD"
                                )
                    except Exception:
                        pass

                # JSON-based mechanical design report (comprehensive)
                if hasattr(md, 'toJson'):
                    try:
                        md_json = str(md.toJson())
                        md_data = json.loads(md_json)
                        if isinstance(md_data, dict):
                            for mk, mv in md_data.items():
                                if isinstance(mv, (int, float)) and not math.isnan(mv) and mv != 0.0:
                                    # Skip fields already extracted above
                                    if mk not in ('totalWeight', 'wallThickness',
                                                   'innerDiameter', 'outerDiameter',
                                                   'tantanLength', 'totalVolume'):
                                        kpis[f"{prefix}.mechDesign.json.{mk}"] = KPI(
                                            f"{prefix}.mechDesign.json.{mk}", float(mv), ""
                                        )
                                elif isinstance(mv, str) and mv and mv != 'null':
                                    # Store design standard, equipment type etc.
                                    if mk in ('designStandard', 'equipmentType',
                                              'equipmentClass', 'casingType'):
                                        kpis[f"{prefix}.mechDesign.json.{mk}"] = KPI(
                                            f"{prefix}.mechDesign.json.{mk}", 0.0, mv
                                        )
                    except Exception:
                        pass

            except Exception:
                pass

        # --- System-level mechanical design (totals, footprint, weight breakdown) ---
        # NOTE: we intentionally skip ``runDesignCalculation()`` because it
        # calls ``initMechanicalDesign()`` + ``calcDesign()`` on every unit —
        # exactly the recalculation we want to avoid after auto-sizing.
        # The totals below will use whatever per-unit design data is already
        # present (i.e. from an earlier autoSize call).
        try:
            SMD = jneqsim.process.mechanicaldesign.SystemMechanicalDesign
            smd = SMD(self._proc)

            # System totals
            for prop, getter, unit in [
                ("system.totalWeight_kg", "getTotalWeight", "kg"),
                ("system.totalVolume_m3", "getTotalVolume", "m3"),
                ("system.plotSpace_m2", "getTotalPlotSpace", "m2"),
                ("system.footprintLength_m", "getTotalFootprintLength", "m"),
                ("system.footprintWidth_m", "getTotalFootprintWidth", "m"),
                ("system.maxEquipmentHeight_m", "getMaxEquipmentHeight", "m"),
                ("system.totalPowerRequired_kW", "getTotalPowerRequired", "kW"),
                ("system.totalCoolingDuty_kW", "getTotalCoolingDuty", "kW"),
                ("system.totalHeatingDuty_kW", "getTotalHeatingDuty", "kW"),
                ("system.netPowerRequirement_kW", "getNetPowerRequirement", "kW"),
            ]:
                if hasattr(smd, getter):
                    try:
                        val = float(getattr(smd, getter)())
                        if math.isnan(val):
                            continue
                        # Convert W to kW for power/duty values
                        if "Power" in getter or "Duty" in getter or "Power" in prop:
                            val = val / 1000.0
                        kpis[prop] = KPI(prop, val, unit)
                    except Exception:
                        pass

            # Number of modules
            try:
                n_modules = int(smd.getTotalNumberOfModules())
                kpis["system.numberOfModules"] = KPI("system.numberOfModules", float(n_modules), "[-]")
            except Exception:
                pass

            # Weight breakdown by equipment type
            try:
                wbt = smd.getWeightByEquipmentType()
                if wbt is not None:
                    for k in wbt.keySet():
                        w = float(wbt.get(k))
                        if w > 0:
                            kpis[f"system.weightByType.{k}_kg"] = KPI(
                                f"system.weightByType.{k}_kg", w, "kg"
                            )
            except Exception:
                pass

            # Weight breakdown by discipline
            try:
                wbd = smd.getWeightByDiscipline()
                if wbd is not None:
                    for k in wbd.keySet():
                        w = float(wbd.get(k))
                        if w > 0:
                            kpis[f"system.weightByDiscipline.{k}_kg"] = KPI(
                                f"system.weightByDiscipline.{k}_kg", w, "kg"
                            )
            except Exception:
                pass

            # Equipment count by type
            try:
                ec = smd.getEquipmentCountByType()
                if ec is not None:
                    for k in ec.keySet():
                        cnt = int(ec.get(k))
                        kpis[f"system.equipmentCount.{k}"] = KPI(
                            f"system.equipmentCount.{k}", float(cnt), "[-]"
                        )
            except Exception:
                pass

            # Total cost across all equipment
            total_cost = 0.0
            for kpi_key, kpi_val in kpis.items():
                if kpi_key.endswith(".cost.totalCost_USD"):
                    total_cost += kpi_val.value
            if total_cost > 0:
                kpis["system.totalCost_USD"] = KPI("system.totalCost_USD", total_cost, "USD")

        except Exception:
            pass

    def _flatten_json_report(self, json_report: dict, kpis: Dict[str, KPI]):
        """
        Flatten the nested JSON report into queryable KPI entries.
        
        Produces keys like:
          "report.feed gas.properties.gas.density"
          "report.1st stage compressor.power"
          "report.inlet separator.gas.conditions.gas.temperature"
          "report.inlet separator.gas.composition.gas.methane"
        """
        for unit_name, unit_data in json_report.items():
            if not isinstance(unit_data, dict):
                continue
            prefix = f"report.{unit_name}"
            self._flatten_dict(unit_data, prefix, kpis)

    def _flatten_dict(self, data: dict, prefix: str, kpis: Dict[str, KPI]):
        """Recursively flatten a nested dict into KPIs."""
        for key, val in data.items():
            full_key = f"{prefix}.{key}"
            if isinstance(val, dict):
                # Check if it's a {value, unit} leaf
                if "value" in val and "unit" in val:
                    try:
                        fval = float(val["value"])
                        unit = str(val.get("unit", ""))
                        kpis[full_key] = KPI(full_key, fval, unit)
                    except (ValueError, TypeError):
                        pass  # skip non-numeric values
                else:
                    self._flatten_dict(val, full_key, kpis)
            elif isinstance(val, (int, float)):
                kpis[full_key] = KPI(full_key, float(val), "")
            # Skip strings/lists that aren't value/unit pairs

    def _extract_stream_fluid_properties(self, kpis: Dict[str, KPI]):
        """
        Extract calculated fluid properties from stream objects.
        
        Adds properties not in the standard JSON report such as:
        viscosity, thermal conductivity, Z-factor, JT coefficient,
        sound speed, TVP (true vapor pressure), RVP (Reid vapor pressure).
        
        Skips streams with near-zero flow (< 0.01 kg/hr) to avoid
        numerically spurious values from empty separator outlets.
        Deduplicates by Java object id to avoid repeated entries for
        the same stream registered under multiple aliases (e.g.
        'feed gas' and 'feed gas.feed gas').
        """
        from neqsim import jneqsim

        seen_ids: set = set()

        # Sort by key length so shorter (unqualified) names are preferred
        sorted_streams = sorted(self._streams.items(), key=lambda x: len(x[0]))

        for stream_name, s in sorted_streams:
            # Deduplicate: only process each Java stream object once.
            # Use Java hashCode() since JPype proxies have different Python ids.
            try:
                java_hash = int(s.hashCode())
            except Exception:
                java_hash = id(s)
            if java_hash in seen_ids:
                continue
            seen_ids.add(java_hash)

            try:
                fluid = s.getFluid()
                if fluid is None:
                    continue
            except Exception:
                continue

            # Skip streams with near-zero flow to avoid spurious values
            # (e.g. empty liquid outlets from gas-only separators)
            try:
                flow = float(s.getFlowRate("kg/hr"))
                if abs(flow) < 0.01:
                    continue
            except Exception:
                pass

            prefix = f"{stream_name}"

            # --- Stream conditions (T, P, flow) ---
            try:
                temp = float(fluid.getTemperature("C"))
                kpis[f"{prefix}.temperature_C"] = KPI(f"{prefix}.temperature_C", temp, "C")
            except Exception:
                pass
            try:
                pres = float(fluid.getPressure("bara"))
                kpis[f"{prefix}.pressure_bara"] = KPI(f"{prefix}.pressure_bara", pres, "bara")
            except Exception:
                pass
            try:
                flow = float(s.getFlowRate("kg/hr"))
                kpis[f"{prefix}.flow_kg_hr"] = KPI(f"{prefix}.flow_kg_hr", flow, "kg/hr")
            except Exception:
                pass

            # --- Phase-level properties ---
            prop_methods = [
                ("viscosity_Pa_s", "getViscosity", "Pa·s"),
                ("kinematic_viscosity_m2_s", "getKinematicViscosity", "m2/s"),
                ("thermal_conductivity_W_mK", "getThermalConductivity", "W/(m·K)"),
                ("Z_factor", "getZ", "[-]"),
                ("compressibility_Z", "getZ", "[-]"),
                ("density_kg_m3", "getDensity", "kg/m3"),
                ("molar_mass_kg_mol", "getMolarMass", "kg/mol"),
                ("molar_volume_m3_mol", "getMolarVolume", "m3/mol"),
                ("enthalpy_J_kg", "getEnthalpy", "J/kg"),
                ("entropy_J_kgK", "getEntropy", "J/(kg·K)"),
                ("Cp_kJ_kgK", "getCp", "kJ/(kg·K)"),
                ("Cv_kJ_kgK", "getCv", "kJ/(kg·K)"),
                ("JT_coefficient_K_bar", "getJouleThomsonCoefficient", "K/bar"),
                ("sound_speed_m_s", "getSoundSpeed", "m/s"),
                ("beta", "getBeta", "[-]"),
            ]

            for prop_name, method_name, unit in prop_methods:
                if hasattr(fluid, method_name):
                    try:
                        val = float(getattr(fluid, method_name)())
                        if prop_name.startswith("Cp_") or prop_name.startswith("Cv_"):
                            val = val / 1000.0  # J -> kJ
                        kpis[f"{prefix}.{prop_name}"] = KPI(
                            f"{prefix}.{prop_name}", val, unit
                        )
                    except Exception:
                        pass

            # --- TVP (True Vapor Pressure) at stream temperature ---
            try:
                fluid_tvp = fluid.clone()
                fluid_tvp.init(0)
                ops_tvp = jneqsim.thermodynamicoperations.ThermodynamicOperations(fluid_tvp)
                ops_tvp.bubblePointPressureFlash(False)
                tvp = float(fluid_tvp.getPressure("bara"))
                kpis[f"{prefix}.TVP_bara"] = KPI(f"{prefix}.TVP_bara", tvp, "bara")
            except Exception:
                pass

            # --- RVP (Reid Vapor Pressure) at 37.8°C (100°F) ---
            try:
                fluid_rvp = fluid.clone()
                fluid_rvp.setTemperature(37.8, "C")
                fluid_rvp.init(0)
                ops_rvp = jneqsim.thermodynamicoperations.ThermodynamicOperations(fluid_rvp)
                ops_rvp.bubblePointPressureFlash(False)
                rvp = float(fluid_rvp.getPressure("bara"))
                kpis[f"{prefix}.RVP_bara"] = KPI(f"{prefix}.RVP_bara", rvp, "bara")
            except Exception:
                pass

            # --- Number of phases ---
            try:
                n_phases = int(fluid.getNumberOfPhases())
                kpis[f"{prefix}.number_of_phases"] = KPI(
                    f"{prefix}.number_of_phases", float(n_phases), "[-]"
                )
            except Exception:
                pass

            # --- Phase fractions (gas/oil/water) ---
            try:
                n_phases = int(fluid.getNumberOfPhases())
                for ph_idx in range(n_phases):
                    try:
                        phase = fluid.getPhase(ph_idx)
                        phase_type = str(phase.getPhaseTypeName()).lower()
                        mole_frac = float(phase.getBeta())
                        kpis[f"{prefix}.{phase_type}_phase_fraction"] = KPI(
                            f"{prefix}.{phase_type}_phase_fraction", mole_frac, "[-]"
                        )
                        # Phase-specific density
                        ph_density = float(phase.getDensity("kg/m3"))
                        kpis[f"{prefix}.{phase_type}_density_kg_m3"] = KPI(
                            f"{prefix}.{phase_type}_density_kg_m3", ph_density, "kg/m3"
                        )
                        # Phase-specific viscosity
                        try:
                            ph_visc = float(phase.getViscosity("kg/msec"))
                            kpis[f"{prefix}.{phase_type}_viscosity_Pa_s"] = KPI(
                                f"{prefix}.{phase_type}_viscosity_Pa_s", ph_visc, "Pa·s"
                            )
                        except Exception:
                            pass
                    except Exception:
                        pass
            except Exception:
                pass

    def get_json_report(self) -> Optional[dict]:
        """Get the full JSON report from the last run."""
        try:
            from neqsim import jneqsim
            report_obj = jneqsim.process.util.report.Report(self._proc)
            json_str = str(report_obj.generateJsonReport())
            return json.loads(json_str)
        except Exception:
            try:
                json_str = str(self._proc.getReport_json())
                return json.loads(json_str)
            except Exception:
                return None

    def query_properties(self, query: str, _cached_result: Optional[ModelRunResult] = None) -> str:
        """
        Run the model and return properties matching a natural-language query.
        
        Used by the LLM to answer READ-ONLY property questions like
        "What is the TVP of the feed gas?" or "What is the density of the export stream?"
        
        The query is matched against all KPI keys (case-insensitive substring match).
        Returns a formatted text with matching properties.
        
        Args:
            query: Natural-language search terms (e.g. "feed gas TVP").
            _cached_result: If provided, reuse this run result instead of re-running.
        """
        # Run the model to get current state (or reuse cached result)
        result = _cached_result if _cached_result is not None else self.run()
        
        # Normalize query for matching
        query_lower = query.lower().strip()
        
        # Split into search terms (support multi-word like "feed gas tvp")
        terms = query_lower.split()
        
        # Find matching KPIs
        matches = []
        for key, kpi in result.kpis.items():
            key_lower = key.lower()
            if all(term in key_lower for term in terms):
                matches.append(kpi)
        
        if not matches:
            # Try less strict matching (any term matches)
            for key, kpi in result.kpis.items():
                key_lower = key.lower()
                if any(term in key_lower for term in terms):
                    matches.append(kpi)
        
        if not matches:
            # List all available property categories  
            categories = set()
            for key in result.kpis.keys():
                parts = key.split(".")
                if len(parts) >= 2:
                    categories.add(f"{parts[0]}.{parts[1]}" if not key.startswith("report.") else f"{parts[1]}.{parts[2]}")
            avail = "\n".join(f"  - {c}" for c in sorted(categories)[:30])
            return f"No properties matching '{query}' found.\n\nAvailable property categories:\n{avail}"
        
        lines = [f"Properties matching '{query}':"]
        for kpi in sorted(matches, key=lambda k: k.name):
            lines.append(f"  {kpi.name} = {kpi.value:.6g} {kpi.unit}")
        
        return "\n".join(lines)

    def get_model_summary(self) -> str:
        """
        Generate a human-readable summary of the process model.
        Used as context for the LLM. Includes topology (connectivity).
        """
        units = self.list_units()
        streams = self.list_streams()

        lines = []
        lines.append(f"Process Model Summary")
        try:
            lines.append(f"Name: {self._proc.getName()}")
        except Exception:
            pass
        lines.append(f"Units: {len(units)}")
        lines.append(f"Streams: {len(streams)}")
        lines.append("")

        # Process topology — show units in order with inlet/outlet stream conditions
        lines.append("== Process Topology (units in process order) ==")
        try:
            ordered_units = list(self._proc.getUnitOperations())
        except Exception:
            ordered_units = []

        for idx, u in enumerate(ordered_units):
            try:
                name = str(u.getName())
                utype = str(u.getClass().getSimpleName())
            except Exception:
                continue

            # Unit properties
            props = {}
            for prop, getter in [
                ("power_kW", "getPower"),
                ("duty_kW", "getDuty"),
                ("isentropicEfficiency", "getIsentropicEfficiency"),
                ("polytropicEfficiency", "getPolytropicEfficiency"),
                ("outletPressure_bara", "getOutletPressure"),
            ]:
                if hasattr(u, getter):
                    try:
                        val = getattr(u, getter)()
                        if val is None:
                            continue
                        fval = float(val)
                        if prop in ("power_kW", "duty_kW"):
                            fval = fval / 1000.0
                        # Fallback: if duty is 0 for a heat-exchange unit, try getEnergyInput
                        if fval == 0.0 and prop == "duty_kW" and utype in self._DUTY_UNITS:
                            if hasattr(u, "getEnergyInput"):
                                try:
                                    fval = float(u.getEnergyInput()) / 1000.0
                                except Exception:
                                    pass
                        # Skip zero power/duty for non-relevant equipment
                        if fval == 0.0 and prop == "power_kW" and utype not in self._POWER_UNITS:
                            continue
                        if fval == 0.0 and prop == "duty_kW" and utype not in self._DUTY_UNITS:
                            continue
                        props[prop] = fval
                    except Exception:
                        pass

            # Outlet temperature for heaters/coolers
            if utype in self._HEAT_EXCHANGE_UNITS:
                for m in ("getOutletStream", "getOutStream"):
                    if hasattr(u, m):
                        try:
                            s = getattr(u, m)()
                            if s is not None:
                                props["outTemperature_C"] = float(s.getTemperature("C"))
                                break
                        except Exception:
                            pass

            prop_str = ", ".join(f"{k}={v:.2f}" for k, v in props.items()) if props else ""

            # Inlet stream conditions
            inlet_str = ""
            for m in ("getInletStream", "getInStream", "getFeed"):
                if hasattr(u, m):
                    try:
                        s = getattr(u, m)()
                        if s is not None:
                            sname = str(s.getName()) if s.getName() else "?"
                            T = float(s.getTemperature("C"))
                            P = float(s.getPressure("bara"))
                            inlet_str = f"IN: {sname} (T={T:.1f}°C, P={P:.2f} bara)"
                            break
                    except Exception:
                        pass

            # Outlet stream conditions — show ALL outlets for separators
            outlet_strs = []
            is_separator = "Separator" in utype or "Scrubber" in utype

            if is_separator:
                # Show gas, oil, liquid, water outlets separately
                for m, label in [
                    ("getGasOutStream", "GAS"),
                    ("getOilOutStream", "OIL"),
                    ("getLiquidOutStream", "LIQ"),
                    ("getWaterOutStream", "WATER"),
                ]:
                    if hasattr(u, m):
                        try:
                            s = getattr(u, m)()
                            if s is not None:
                                sname = str(s.getName()) if s.getName() else label
                                T = float(s.getTemperature("C"))
                                P = float(s.getPressure("bara"))
                                F = float(s.getFlowRate("kg/hr"))
                                outlet_strs.append(
                                    f"OUT ({label}): {sname} (T={T:.1f}°C, P={P:.2f} bara, F={F:.1f} kg/hr)"
                                )
                        except Exception:
                            pass
            else:
                for m in ("getOutletStream", "getOutStream", "getGasOutStream"):
                    if hasattr(u, m):
                        try:
                            s = getattr(u, m)()
                            if s is not None:
                                sname = str(s.getName()) if s.getName() else "?"
                                T = float(s.getTemperature("C"))
                                P = float(s.getPressure("bara"))
                                outlet_strs.append(f"OUT: {sname} (T={T:.1f}°C, P={P:.2f} bara)")
                                break
                        except Exception:
                            pass

            # Splitter: show split streams
            if "Splitter" in utype and hasattr(u, "getSplitStream"):
                for j in range(10):
                    try:
                        s = u.getSplitStream(j)
                        if s is not None:
                            sname = str(s.getName()) if s.getName() else f"split_{j}"
                            F = float(s.getFlowRate("kg/hr"))
                            outlet_strs.append(f"OUT (SPLIT {j}): {sname} (F={F:.1f} kg/hr)")
                    except Exception:
                        break

            line = f"  [{idx}] {name} ({utype})"
            if prop_str:
                line += f" — {prop_str}"
            lines.append(line)
            if inlet_str:
                lines.append(f"        {inlet_str}")
            for outlet_str in outlet_strs:
                lines.append(f"        {outlet_str}")

        lines.append("")
        lines.append("== All Streams ==")
        for s in streams:
            parts = []
            if s.temperature_C is not None:
                parts.append(f"T={s.temperature_C:.1f}°C")
            if s.pressure_bara is not None:
                parts.append(f"P={s.pressure_bara:.2f} bara")
            if s.flow_rate_kg_hr is not None:
                parts.append(f"F={s.flow_rate_kg_hr:.1f} kg/hr")
            lines.append(f"  {s.name}: {', '.join(parts)}")

        return "\n".join(lines)
