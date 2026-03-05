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
    process_system: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamInfo:
    name: str
    temperature_C: Optional[float] = None
    pressure_bara: Optional[float] = None
    flow_rate_kg_hr: Optional[float] = None
    flow_rate_mol_sec: Optional[float] = None
    process_system: str = ""


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------

class NeqSimProcessModel:
    """
    Wraps a NeqSim ProcessSystem **or ProcessModel** loaded from a .neqsim file.

    A ``ProcessModel`` contains multiple named ``ProcessSystem`` objects.
    This adapter transparently handles both:

    - Single ``ProcessSystem``: behaves as before.
    - ``ProcessModel``: iterates over all child ``ProcessSystem`` instances
      when indexing units/streams, extracting results, and generating summaries.

    Provides introspection, cloning, and scenario execution capabilities
    for the chat + what-if engine.
    """

    def __init__(self, process_system, source_bytes: Optional[bytes] = None):
        """
        Args:
            process_system: A NeqSim ProcessSystem **or ProcessModel** Java object.
            source_bytes: Original file bytes for clone-by-reload.
        """
        self._proc = process_system
        self._source_bytes = source_bytes
        self._units: Dict[str, Any] = {}
        self._streams: Dict[str, Any] = {}
        self._is_process_model = self._detect_process_model(process_system)
        self._index_model_objects()

    # ----- ProcessModel detection -----

    @staticmethod
    def _detect_process_model(obj) -> bool:
        """Return True if *obj* is a NeqSim ``ProcessModel`` (multi-system)."""
        try:
            cls_name = str(obj.getClass().getSimpleName())
            if cls_name == "ProcessModel":
                return True
        except Exception:
            pass
        try:
            full_name = str(obj.getClass().getName())
            if "ProcessModel" in full_name and "ProcessSystem" not in full_name:
                return True
        except Exception:
            pass
        # Duck-type: ProcessModel has getAllProcesses() but not getUnitOperations()
        return hasattr(obj, "getAllProcesses") and not hasattr(obj, "getUnitOperations")

    @property
    def is_process_model(self) -> bool:
        """True when the underlying Java object is a ProcessModel (multi-system)."""
        return self._is_process_model

    def get_process_systems(self) -> List[Any]:
        """Return the list of child ProcessSystem objects.

        For a single ProcessSystem this returns ``[self._proc]``.
        For a ProcessModel it returns all children from ``getAllProcesses()``.
        If ``getAllProcesses()`` fails or returns nothing, falls back to
        returning ``[self._proc]`` so callers always have something to iterate.
        """
        if self._is_process_model:
            try:
                children = list(self._proc.getAllProcesses())
                if children:
                    return children
            except Exception:
                pass
            # Fallback: if ProcessModel itself has getUnitOperations, treat it
            # as a single process system so units/streams are still discovered.
            if hasattr(self._proc, "getUnitOperations"):
                return [self._proc]
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "ProcessModel.getAllProcesses() returned no children "
                "and object lacks getUnitOperations — model will appear empty."
            )
            return []
        return [self._proc]

    def get_process_system_names(self) -> List[str]:
        """Return names of all child ProcessSystems (ProcessModel only)."""
        if not self._is_process_model:
            try:
                return [str(self._proc.getName())]
            except Exception:
                return ["process"]
        names = []
        try:
            for ps in self._proc.getAllProcesses():
                try:
                    names.append(str(ps.getName()))
                except Exception:
                    names.append("unnamed")
        except Exception:
            pass
        return names

    def get_all_unit_operations(self) -> list:
        """Return a flat list of all Java unit-operation objects.

        For a single ProcessSystem this delegates to getUnitOperations().
        For a ProcessModel it iterates every child ProcessSystem.
        """
        all_units: list = []
        if self._is_process_model:
            for ps in self.get_process_systems():
                try:
                    all_units.extend(list(ps.getUnitOperations()))
                except Exception:
                    pass
        else:
            try:
                all_units = list(self._proc.getUnitOperations())
            except Exception:
                pass
        return all_units

    def find_process_system_for_unit(self, unit_name: str):
        """Return the ProcessSystem that contains the named unit.

        For a single ProcessSystem, returns that system directly.
        For a ProcessModel, searches each child ProcessSystem.
        Returns None if not found.
        """
        if not self._is_process_model:
            return self._proc
        for ps in self.get_process_systems():
            try:
                for u in ps.getUnitOperations():
                    try:
                        if str(u.getName()) == unit_name:
                            return ps
                    except Exception:
                        pass
            except Exception:
                pass
        # Fallback: case-insensitive search
        unit_lower = unit_name.lower()
        for ps in self.get_process_systems():
            try:
                for u in ps.getUnitOperations():
                    try:
                        if str(u.getName()).lower() == unit_lower:
                            return ps
                    except Exception:
                        pass
            except Exception:
                pass
        return None

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

    def save_bytes(self) -> Optional[bytes]:
        """Return the current process state as serialized .neqsim bytes.

        Works for both ProcessSystem and ProcessModel.
        """
        self.refresh_source_bytes()
        return self._source_bytes

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
        """Discover all unit operations and streams in the process.

        For a ``ProcessModel`` (multi-system), iterates every child
        ``ProcessSystem`` and collects units/streams across all of them.
        Unit names are kept as-is when unique; when a name appears in
        multiple process systems it is qualified with the system name.
        """
        self._units.clear()
        self._streams.clear()
        self._unit_ps_name: Dict[str, str] = {}
        self._stream_ps_name: Dict[str, str] = {}

        # Collect all (process_system_name, unit_operations_list) pairs
        ps_units: List[Tuple[str, list]] = []
        if self._is_process_model:
            for ps in self.get_process_systems():
                try:
                    ps_name = str(ps.getName()) if ps.getName() else "unnamed"
                except Exception:
                    ps_name = "unnamed"
                try:
                    units = list(ps.getUnitOperations())
                except Exception:
                    try:
                        units = list(ps.getUnitOperationList())
                    except Exception:
                        units = []
                ps_units.append((ps_name, units))
        else:
            proc = self._proc
            ps_name = ""
            try:
                ps_name = str(proc.getName()) if proc.getName() else ""
            except Exception:
                pass
            try:
                units = list(proc.getUnitOperations())
            except Exception:
                try:
                    units = list(proc.getUnitOperationList())
                except Exception:
                    units = []
            ps_units.append((ps_name, units))

        # Flatten all units, detect name collisions across systems
        all_units_flat: list = []  # (ps_name, unit, raw_name)
        name_count: Dict[str, int] = {}
        for ps_name, units in ps_units:
            for u in units:
                try:
                    raw_name = str(u.getName()) if u.getName() else None
                except Exception:
                    raw_name = None
                if raw_name:
                    all_units_flat.append((ps_name, u, raw_name))
                    name_count[raw_name] = name_count.get(raw_name, 0) + 1

        # Register units — qualify with process-system name when ambiguous
        for ps_name, u, raw_name in all_units_flat:
            if name_count[raw_name] > 1 and ps_name:
                key = f"{ps_name}/{raw_name}"
            else:
                key = raw_name
            # Deduplicate: if the key already exists, append a numeric suffix
            if key in self._units:
                suffix = 2
                while f"{key}_{suffix}" in self._units:
                    suffix += 1
                key = f"{key}_{suffix}"
            self._units[key] = u
            self._unit_ps_name[key] = ps_name

        # Discover streams from unit in/out connections.
        # Always use qualified keys ("unitName.streamName") as primary to
        # guarantee stable KPI comparisons across base vs scenario runs.
        # Also add short aliases for stream names that are globally unique.
        seen_java_ids = set()  # track Java object identity to skip duplicates
        raw_name_count: Dict[str, int] = {}  # count how many units produce same stream name

        # Iterate all_units_flat to preserve ps_name for stream tracking
        for ps_name, u, _raw_name in all_units_flat:
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
                                                self._stream_ps_name[key] = ps_name
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
                                    self._stream_ps_name[key] = ps_name
                                    raw_name_count[sname] = raw_name_count.get(sname, 0) + 1
                    except Exception:
                        pass

        # Also index units that are streams themselves (Stream objects added to process)
        for name, u in list(self._units.items()):
            try:
                java_class = str(u.getClass().getSimpleName())
                if "Stream" in java_class and name not in self._streams:
                    self._streams[name] = u
                    self._stream_ps_name[name] = self._unit_ps_name.get(name, "")
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
                self._stream_ps_name[sname] = self._stream_ps_name.get(key, "")

    def get_process(self):
        """Return the underlying Java object (ProcessSystem or ProcessModel).

        For a ``ProcessModel``, this returns the ``ProcessModel`` itself —
        callers that need individual ``ProcessSystem`` objects should use
        :meth:`get_process_systems` instead.
        """
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

        Tries the Java ``createDiagramExporter()`` first (available on
        ``ProcessSystem``).  If the method does not exist (e.g. on
        ``ProcessModel``) or fails at runtime, falls back to a pure-Python
        DOT generator built from the indexed units and streams.

        Parameters
        ----------
        style : str
            Diagram style hint (used by Java exporter; ignored by fallback).
        detail_level : str
            ``CONCEPTUAL``, ``ENGINEERING`` (default), or ``DEBUG``.
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

        # --- Helper: try the Java exporter on a single ProcessSystem ---
        def _try_java_exporter(ps, ps_title: str) -> Optional[str]:
            """Return DOT from Java exporter or None on failure."""
            if not hasattr(ps, "createDiagramExporter"):
                return None
            try:
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

                exporter = ps.createDiagramExporter()
                exporter.setDiagramStyle(style_map.get(style.upper(), DiagramStyle.HYSYS))
                exporter.setDetailLevel(level_map.get(detail_level.upper(), DiagramDetailLevel.ENGINEERING))
                exporter.setShowStreamValues(show_stream_values)
                exporter.setUseStreamTables(use_stream_tables)
                exporter.setShowControlEquipment(show_control_equipment)
                if ps_title:
                    exporter.setTitle(ps_title)
                return str(exporter.toDOT())
            except Exception:
                return None

        # --- Attempt Java exporter on each ProcessSystem ---
        dots: list = []  # (ps_name, dot_string)
        for ps in self.get_process_systems():
            try:
                ps_name = str(ps.getName()) if ps.getName() else ""
            except Exception:
                ps_name = ""
            result = _try_java_exporter(ps, title or ps_name)
            if result:
                dots.append((ps_name, result))

        if dots:
            if len(dots) == 1:
                return dots[0][1]
            # Merge multiple DOTs into a single digraph with subgraph clusters
            return self._merge_dots(dots, title=title)

        # --- Fallback: pure-Python DOT generator ---
        return self._generate_dot_fallback(
            detail_level=detail_level,
            show_stream_values=show_stream_values,
            show_control_equipment=show_control_equipment,
            title=title,
        )

    def get_diagram_dots(
        self,
        style: str = "HYSYS",
        detail_level: str = "ENGINEERING",
        show_stream_values: bool = True,
        use_stream_tables: bool = False,
        show_control_equipment: bool = True,
    ) -> List[Tuple[str, str]]:
        """Return a list of ``(system_name, dot_string)`` tuples.

        For a single ProcessSystem the list has one entry.
        For a ProcessModel each child ProcessSystem gets its own DOT.

        This is useful for rendering each system in its own tab / expander.
        """
        results: List[Tuple[str, str]] = []
        systems = self.get_process_systems()
        for ps in systems:
            try:
                ps_name = str(ps.getName()) if ps.getName() else "Process"
            except Exception:
                ps_name = "Process"
            # Try Java exporter first
            if hasattr(ps, "createDiagramExporter"):
                try:
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
                    exporter = ps.createDiagramExporter()
                    exporter.setDiagramStyle(style_map.get(style.upper(), DiagramStyle.HYSYS))
                    exporter.setDetailLevel(level_map.get(detail_level.upper(), DiagramDetailLevel.ENGINEERING))
                    exporter.setShowStreamValues(show_stream_values)
                    exporter.setUseStreamTables(use_stream_tables)
                    exporter.setShowControlEquipment(show_control_equipment)
                    exporter.setTitle(ps_name)
                    dot = str(exporter.toDOT())
                    if dot:
                        results.append((ps_name, dot))
                        continue
                except Exception:
                    pass

        # If Java exporter produced nothing for any system, use fallback for all
        if len(results) < len(systems):
            fallback = self._generate_dot_fallback(
                detail_level=detail_level,
                show_stream_values=show_stream_values,
                show_control_equipment=show_control_equipment,
            )
            if fallback:
                combined_name = "Process"
                try:
                    combined_name = str(self._proc.getName()) or "Process"
                except Exception:
                    pass
                results = [(combined_name, fallback)]

        return results

    @staticmethod
    def _merge_dots(dots: List[Tuple[str, str]], title: str = "") -> str:
        """Merge multiple ``digraph { ... }`` DOT strings into a single
        DOT graph using ``subgraph cluster_*`` blocks.

        Each child DOT's contents are extracted and placed inside a named
        cluster so they render as grouped regions of a single diagram.
        """
        import re

        overall_title = title or "Process Model"
        parts = [
            "digraph ProcessModel {",
            '  graph [rankdir=LR splines=ortho nodesep=0.8 ranksep=1.2',
            f'         fontname="Arial" fontsize=14 label="{overall_title}"',
            '         labelloc=t labeljust=c bgcolor="white" pad=0.5 compound=true];',
            '  node [fontname="Arial" fontsize=10 style=filled];',
            '  edge [fontname="Arial" fontsize=8 color="#666666"];',
            "",
        ]

        for idx, (ps_name, dot_str) in enumerate(dots):
            # Extract body between first '{' and last '}'
            body_match = re.search(r"\{(.*)\}", dot_str, re.DOTALL)
            if not body_match:
                continue
            body = body_match.group(1)

            # Remove any graph-level attributes that would conflict
            # (label, bgcolor, rankdir, etc.) — they're on lines starting
            # with 'graph [' or standalone attribute statements
            body_lines = []
            for line in body.split("\n"):
                stripped = line.strip()
                if stripped.startswith("graph [") or stripped.startswith("graph["):
                    continue
                # Keep node/edge defaults and actual nodes/edges
                body_lines.append(line)
            body_clean = "\n".join(body_lines)

            safe_name = re.sub(r"[^a-zA-Z0-9]", "_", ps_name)
            parts.append(f"  subgraph cluster_{idx}_{safe_name} {{")
            parts.append(f'    label="{ps_name}";')
            parts.append('    style=dashed;')
            parts.append(f'    color="#999999";')
            parts.append('    fontname="Arial";')
            parts.append('    fontsize=12;')
            parts.append(body_clean)
            parts.append("  }")
            parts.append("")

        parts.append("}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Pure-Python DOT fallback
    # ------------------------------------------------------------------

    # Shape mapping for unit operation types
    _UNIT_SHAPES = {
        "Stream": ("ellipse", "#E8F5E9"),
        "Compressor": ("box", "#BBDEFB"),
        "Pump": ("box", "#BBDEFB"),
        "ESPPump": ("box", "#BBDEFB"),
        "Expander": ("box", "#C8E6C9"),
        "GasTurbine": ("box", "#C8E6C9"),
        "Cooler": ("box", "#B3E5FC"),
        "Heater": ("box", "#FFCCBC"),
        "HeatExchanger": ("box", "#FFE0B2"),
        "AirCooler": ("box", "#B3E5FC"),
        "WaterCooler": ("box", "#B3E5FC"),
        "Separator": ("hexagon", "#FFF9C4"),
        "ThreePhaseSeparator": ("hexagon", "#FFF9C4"),
        "TwoPhaseSeparator": ("hexagon", "#FFF9C4"),
        "Mixer": ("invtriangle", "#E1BEE7"),
        "Splitter": ("triangle", "#E1BEE7"),
        "ThrottlingValve": ("diamond", "#F0F4C3"),
        "Valve": ("diamond", "#F0F4C3"),
        "Recycle": ("doubleoctagon", "#D7CCC8"),
        "Absorber": ("box3d", "#DCEDC8"),
        "DistillationColumn": ("box3d", "#DCEDC8"),
        "Filter": ("trapezium", "#CFD8DC"),
        "WellStream": ("ellipse", "#E8F5E9"),
    }

    _CONTROL_TYPES = {"Recycle", "Calculator", "Adjuster", "SetPoint"}

    def _generate_dot_fallback(
        self,
        detail_level: str = "ENGINEERING",
        show_stream_values: bool = True,
        show_control_equipment: bool = True,
        title: str = "",
    ) -> str:
        """Build a Graphviz DOT string from indexed units and streams.

        Uses process execution order and inlet/outlet stream matching to
        determine connectivity between unit operations.
        """

        # Collect ordered unit info from each ProcessSystem
        all_units_ordered: list = []  # (unit_java_obj, name, java_class)
        for ps in self.get_process_systems():
            try:
                ops = list(ps.getUnitOperations())
            except Exception:
                try:
                    ops = list(ps.getUnitOperationList())
                except Exception:
                    ops = []
            for u in ops:
                try:
                    name = str(u.getName()) if u.getName() else "unit"
                except Exception:
                    name = "unit"
                try:
                    cls = str(u.getClass().getSimpleName())
                except Exception:
                    cls = "Unknown"
                all_units_ordered.append((u, name, cls))

        # If no units from process systems, fall back to indexed units
        if not all_units_ordered:
            for name, u in self._units.items():
                try:
                    cls = str(u.getClass().getSimpleName())
                except Exception:
                    cls = "Unknown"
                all_units_ordered.append((u, name, cls))

        if not all_units_ordered:
            return 'digraph G { label="No units found"; }'

        # Assign stable node IDs
        node_ids: Dict[str, str] = {}
        for idx, (_u, name, _cls) in enumerate(all_units_ordered):
            node_ids[name] = f"n{idx}"

        # Build connectivity: outlet_hash → source unit name
        # and inlet_hash → destination unit name
        _OUTLET_METHODS = (
            "getOutletStream", "getOutStream",
            "getGasOutStream", "getOilOutStream",
            "getLiquidOutStream", "getWaterOutStream",
        )
        _INLET_METHODS = ("getInletStream", "getInStream", "getFeed", "getFeedStream")

        # Map: Java stream id → (source_unit_name, stream_label)
        outlet_map: Dict[int, Tuple[str, str]] = {}
        # Map: Java stream id → dest_unit_name
        inlet_map: Dict[int, str] = {}

        # Gather stream conditions for edge labels
        stream_conditions: Dict[int, str] = {}

        def _stream_id(s) -> int:
            try:
                return int(s.hashCode())
            except Exception:
                return id(s)

        def _stream_label(s, method_name: str) -> str:
            try:
                sname = str(s.getName()) if s.getName() else ""
            except Exception:
                sname = ""
            # Tag multi-phase outlets
            if "Gas" in method_name:
                return sname or "gas"
            elif "Oil" in method_name or "Liquid" in method_name:
                return sname or "liquid"
            elif "Water" in method_name:
                return sname or "water"
            return sname

        def _stream_condition_label(s) -> str:
            parts = []
            try:
                t = float(s.getTemperature("C"))
                parts.append(f"{t:.1f} °C")
            except Exception:
                pass
            try:
                p = float(s.getPressure("bara"))
                parts.append(f"{p:.1f} bara")
            except Exception:
                pass
            try:
                f = float(s.getFlowRate("kg/hr"))
                if f > 0:
                    parts.append(f"{f:.0f} kg/h")
            except Exception:
                pass
            return "\\n".join(parts)

        for u, name, cls in all_units_ordered:
            # Outlets
            for mname in _OUTLET_METHODS:
                if hasattr(u, mname):
                    try:
                        s = getattr(u, mname)()
                        if s is not None:
                            sid = _stream_id(s)
                            if sid not in outlet_map:
                                outlet_map[sid] = (name, _stream_label(s, mname))
                            if show_stream_values and sid not in stream_conditions:
                                cond = _stream_condition_label(s)
                                if cond:
                                    stream_conditions[sid] = cond
                    except Exception:
                        pass

            # Splitter outputs via getSplitStream(i)
            if hasattr(u, "getSplitStream"):
                for i in range(10):
                    try:
                        s = u.getSplitStream(i)
                        if s is not None:
                            sid = _stream_id(s)
                            if sid not in outlet_map:
                                slabel = _stream_label(s, "getSplitStream")
                                outlet_map[sid] = (name, slabel or f"split_{i}")
                            if show_stream_values and sid not in stream_conditions:
                                cond = _stream_condition_label(s)
                                if cond:
                                    stream_conditions[sid] = cond
                    except Exception:
                        break

            # Inlets
            for mname in _INLET_METHODS:
                if hasattr(u, mname):
                    try:
                        s = getattr(u, mname)()
                        if s is not None:
                            sid = _stream_id(s)
                            inlet_map[sid] = name
                    except Exception:
                        pass

        # Build edges from matching outlet → inlet stream IDs
        edges: list = []  # (src_name, dst_name, label)
        matched_sources = set()
        matched_dests = set()
        for sid, (src, slabel) in outlet_map.items():
            if sid in inlet_map:
                dst = inlet_map[sid]
                if src != dst:  # skip self-loops
                    edge_label = slabel
                    if show_stream_values and sid in stream_conditions:
                        if edge_label:
                            edge_label += "\\n" + stream_conditions[sid]
                        else:
                            edge_label = stream_conditions[sid]
                    edges.append((src, dst, edge_label))
                    matched_sources.add(src)
                    matched_dests.add(dst)

        # For units with no connectivity found, connect sequentially
        # (fallback for units where inlet/outlet methods are not standard)
        unconnected = [
            name for _u, name, cls in all_units_ordered
            if name not in matched_sources and name not in matched_dests
            and cls not in self._CONTROL_TYPES
        ]
        # Don't sequentially connect if we already have good edges
        if not edges and len(all_units_ordered) > 1:
            # No edges found at all — connect in process order
            prev = None
            for _u, name, cls in all_units_ordered:
                if not show_control_equipment and cls in self._CONTROL_TYPES:
                    continue
                if prev is not None:
                    edges.append((prev, name, ""))
                prev = name

        # Determine diagram title
        if not title:
            try:
                title = str(self._proc.getName()) if self._proc.getName() else "Process Flow Diagram"
            except Exception:
                title = "Process Flow Diagram"

        # --- Generate DOT ---
        lines = [
            "digraph ProcessFlowDiagram {",
            '  graph [rankdir=LR splines=ortho nodesep=0.8 ranksep=1.2',
            f'         fontname="Arial" fontsize=12 label="{title}"',
            '         labelloc=t labeljust=c bgcolor="white" pad=0.5];',
            '  node [fontname="Arial" fontsize=10 style=filled];',
            '  edge [fontname="Arial" fontsize=8 color="#666666"];',
            "",
        ]

        # Nodes
        for _u, name, cls in all_units_ordered:
            if not show_control_equipment and cls in self._CONTROL_TYPES:
                continue
            nid = node_ids[name]
            shape, fill = self._UNIT_SHAPES.get(cls, ("box", "#E0E0E0"))
            # Build label
            if detail_level == "CONCEPTUAL":
                label = name
            else:
                label = f"{name}\\n[{cls}]"
                # Add key properties
                if cls in self._POWER_UNITS:
                    try:
                        pwr = float(_u.getPower()) / 1000.0
                        if abs(pwr) > 0.01:
                            label += f"\\n{pwr:.1f} kW"
                    except Exception:
                        pass
                elif cls in self._DUTY_UNITS:
                    try:
                        duty = float(_u.getDuty()) / 1000.0
                        if abs(duty) > 0.01:
                            label += f"\\n{duty:.1f} kW"
                    except Exception:
                        pass

            lines.append(
                f'  {nid} [label="{label}" shape={shape} fillcolor="{fill}"];'
            )

        lines.append("")

        # Edges
        seen_edges = set()
        for src, dst, label in edges:
            src_id = node_ids.get(src)
            dst_id = node_ids.get(dst)
            if src_id and dst_id:
                edge_key = (src_id, dst_id)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                if label:
                    lines.append(f'  {src_id} -> {dst_id} [label="{label}"];')
                else:
                    lines.append(f'  {src_id} -> {dst_id};')

        lines.append("}")
        return "\n".join(lines)

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

            ps_name = self._unit_ps_name.get(name, "")

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

            result.append(UnitInfo(name=name, unit_type=java_class, java_class=java_class, process_system=ps_name, properties=props))
        return result

    def list_streams(self) -> List[StreamInfo]:
        """List all streams with current conditions."""
        result = []
        for name, s in self._streams.items():
            ps_name = self._stream_ps_name.get(name, "")
            info = StreamInfo(name=name, process_system=ps_name)
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
        # Case-insensitive fallback
        name_lower = name.lower()
        for key, u in self._units.items():
            if key.lower() == name_lower:
                return u
        # For ProcessModel, units might be qualified with process-system name
        for key, u in self._units.items():
            if key.endswith(f"/{name}") or key.endswith(f"/{name_lower}"):
                return u
        # Also try via process.getUnit() (ProcessSystem only)
        if not self._is_process_model:
            try:
                u = self._proc.getUnit(name)
                if u is not None:
                    return u
            except Exception:
                pass
        else:
            # ProcessModel: search each child ProcessSystem
            for ps in self.get_process_systems():
                try:
                    u = ps.getUnit(name)
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
        if not self._is_process_model:
            try:
                u = self._proc.getUnit(name)
                if u is not None:
                    return u
            except Exception:
                pass
        else:
            for ps in self.get_process_systems():
                try:
                    u = ps.getUnit(name)
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
        if self._is_process_model:
            # ProcessModel has its own run() that iterates all children
            self._run_process_model(self._proc, timeout_ms=timeout_ms)
        else:
            self._run_until_converged(self._proc, max_runs=5, timeout_ms=timeout_ms)

        # Re-index model objects after running so references are fresh
        self._index_model_objects()

        return self._extract_results()

    def rerun(self, timeout_ms: int = 120000):
        """Re-run the process without extracting results.

        Convenience method for callers that just need to re-execute the
        simulation (e.g. after modifying parameters) and then re-index.
        Handles both ProcessSystem and ProcessModel transparently.
        """
        if self._is_process_model:
            self._run_process_model(self._proc, timeout_ms=timeout_ms)
        else:
            self._run_until_converged(self._proc, max_runs=5, timeout_ms=timeout_ms)
        self._index_model_objects()

    @staticmethod
    def _run_process_model(proc_model, timeout_ms: int = 180000):
        """Run a ProcessModel (which iterates all child ProcessSystems)."""
        try:
            if timeout_ms > 0:
                thread = proc_model.runAsThread()
                thread.join(timeout_ms)
                if thread.isAlive():
                    thread.interrupt()
                    thread.join()
            else:
                proc_model.run()
        except Exception:
            # Fallback: run each ProcessSystem individually
            try:
                for ps in proc_model.getAllProcesses():
                    try:
                        NeqSimProcessModel._run_until_converged(ps)
                    except Exception:
                        pass
            except Exception:
                pass

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
        if self._is_process_model:
            # ProcessModel has its own getReport_json() that aggregates all systems
            try:
                json_str = str(self._proc.getReport_json())
                json_report = json.loads(json_str)
            except Exception:
                # Fallback: collect reports from each ProcessSystem
                try:
                    from neqsim import jneqsim
                    combined = {}
                    for ps in self.get_process_systems():
                        try:
                            ps_name = str(ps.getName()) if ps.getName() else "process"
                            report_obj = jneqsim.process.util.report.Report(ps)
                            r_str = str(report_obj.generateJsonReport())
                            r_data = json.loads(r_str)
                            if isinstance(r_data, dict):
                                # Prefix keys with process system name if multiple
                                for k, v in r_data.items():
                                    combined[f"{ps_name}/{k}"] = v
                        except Exception:
                            pass
                    if combined:
                        json_report = combined
                except Exception:
                    pass
        else:
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
            # Collect all unit operations across all process systems
            all_units = []
            if self._is_process_model:
                for ps in self.get_process_systems():
                    try:
                        all_units.extend(list(ps.getUnitOperations()))
                    except Exception:
                        pass
            else:
                try:
                    all_units = list(self._proc.getUnitOperations())
                except Exception:
                    pass
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

            # For ProcessModel, aggregate SystemMechanicalDesign across all children
            proc_systems = self.get_process_systems() if self._is_process_model else [self._proc]
            smd_list = []
            for ps in proc_systems:
                try:
                    smd_list.append(SMD(ps))
                except Exception:
                    pass

            if not smd_list:
                raise RuntimeError("No SystemMechanicalDesign created")

            # Aggregate additive system totals across all process systems
            additive_props = [
                ("system.totalWeight_kg", "getTotalWeight", "kg"),
                ("system.totalVolume_m3", "getTotalVolume", "m3"),
                ("system.plotSpace_m2", "getTotalPlotSpace", "m2"),
                ("system.totalPowerRequired_kW", "getTotalPowerRequired", "kW"),
                ("system.totalCoolingDuty_kW", "getTotalCoolingDuty", "kW"),
                ("system.totalHeatingDuty_kW", "getTotalHeatingDuty", "kW"),
                ("system.netPowerRequirement_kW", "getNetPowerRequirement", "kW"),
            ]
            max_props = [
                ("system.footprintLength_m", "getTotalFootprintLength", "m"),
                ("system.footprintWidth_m", "getTotalFootprintWidth", "m"),
                ("system.maxEquipmentHeight_m", "getMaxEquipmentHeight", "m"),
            ]
            for prop, getter, unit in additive_props:
                total_val = 0.0
                for smd in smd_list:
                    if hasattr(smd, getter):
                        try:
                            val = float(getattr(smd, getter)())
                            if not math.isnan(val):
                                if "Power" in getter or "Duty" in getter or "Power" in prop:
                                    val = val / 1000.0
                                total_val += val
                        except Exception:
                            pass
                if total_val != 0.0:
                    kpis[prop] = KPI(prop, total_val, unit)
            for prop, getter, unit in max_props:
                max_val = 0.0
                for smd in smd_list:
                    if hasattr(smd, getter):
                        try:
                            val = float(getattr(smd, getter)())
                            if not math.isnan(val) and val > max_val:
                                max_val = val
                        except Exception:
                            pass
                if max_val > 0.0:
                    kpis[prop] = KPI(prop, max_val, unit)

            # Number of modules (sum across systems)
            total_modules = 0
            for smd in smd_list:
                try:
                    total_modules += int(smd.getTotalNumberOfModules())
                except Exception:
                    pass
            if total_modules > 0:
                kpis["system.numberOfModules"] = KPI("system.numberOfModules", float(total_modules), "[-]")

            # Weight breakdown by equipment type (aggregate)
            weight_by_type: Dict[str, float] = {}
            for smd in smd_list:
                try:
                    wbt = smd.getWeightByEquipmentType()
                    if wbt is not None:
                        for k in wbt.keySet():
                            w = float(wbt.get(k))
                            if w > 0:
                                weight_by_type[str(k)] = weight_by_type.get(str(k), 0.0) + w
                except Exception:
                    pass
            for k, w in weight_by_type.items():
                kpis[f"system.weightByType.{k}_kg"] = KPI(f"system.weightByType.{k}_kg", w, "kg")

            # Weight breakdown by discipline (aggregate)
            weight_by_disc: Dict[str, float] = {}
            for smd in smd_list:
                try:
                    wbd = smd.getWeightByDiscipline()
                    if wbd is not None:
                        for k in wbd.keySet():
                            w = float(wbd.get(k))
                            if w > 0:
                                weight_by_disc[str(k)] = weight_by_disc.get(str(k), 0.0) + w
                except Exception:
                    pass
            for k, w in weight_by_disc.items():
                kpis[f"system.weightByDiscipline.{k}_kg"] = KPI(f"system.weightByDiscipline.{k}_kg", w, "kg")

            # Equipment count by type (aggregate)
            equip_count: Dict[str, int] = {}
            for smd in smd_list:
                try:
                    ec = smd.getEquipmentCountByType()
                    if ec is not None:
                        for k in ec.keySet():
                            equip_count[str(k)] = equip_count.get(str(k), 0) + int(ec.get(k))
                except Exception:
                    pass
            for k, cnt in equip_count.items():
                kpis[f"system.equipmentCount.{k}"] = KPI(f"system.equipmentCount.{k}", float(cnt), "[-]")

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
        if self._is_process_model:
            try:
                json_str = str(self._proc.getReport_json())
                return json.loads(json_str)
            except Exception:
                # Fallback: collect from children
                from neqsim import jneqsim
                combined = {}
                for ps in self.get_process_systems():
                    try:
                        ps_name = str(ps.getName()) if ps.getName() else "process"
                        report_obj = jneqsim.process.util.report.Report(ps)
                        r_str = str(report_obj.generateJsonReport())
                        r_data = json.loads(r_str)
                        if isinstance(r_data, dict):
                            for k, v in r_data.items():
                                combined[f"{ps_name}/{k}"] = v
                    except Exception:
                        pass
                return combined if combined else None
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

    def get_unit_json_report(self, unit_name: str) -> Optional[dict]:
        """Get the JSON report for a single unit operation.

        Extracts the unit's section from the full process JSON report.
        Falls back to reading directly from the Java unit if available.
        """
        full = self.get_json_report()
        if full:
            # Try exact match
            if unit_name in full:
                return {unit_name: full[unit_name]}
            # Case-insensitive / substring match
            ul = unit_name.lower()
            for k, v in full.items():
                if ul in k.lower():
                    return {k: v}
        return None

    def get_stream_json_report(self, stream_name: str) -> Optional[dict]:
        """Get the JSON report section for a specific stream."""
        full = self.get_json_report()
        if full:
            if stream_name in full:
                return {stream_name: full[stream_name]}
            sl = stream_name.lower()
            for k, v in full.items():
                if sl in k.lower():
                    return {k: v}
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
        if self._is_process_model:
            ps_names = self.get_process_system_names()
            lines.append(f"Type: ProcessModel ({len(ps_names)} process systems)")
            lines.append(f"Process Systems: {', '.join(ps_names)}")
        else:
            try:
                lines.append(f"Name: {self._proc.getName()}")
            except Exception:
                pass
        lines.append(f"Units: {len(units)}")
        lines.append(f"Streams: {len(streams)}")
        lines.append("")

        # Process topology — show units in order with inlet/outlet stream conditions
        # For ProcessModel, show each process system separately
        if self._is_process_model:
            for ps in self.get_process_systems():
                try:
                    ps_name = str(ps.getName()) if ps.getName() else "unnamed"
                except Exception:
                    ps_name = "unnamed"
                lines.append(f"== Process System: {ps_name} ==")
                try:
                    ordered_units = list(ps.getUnitOperations())
                except Exception:
                    ordered_units = []
                self._append_topology(lines, ordered_units)
                lines.append("")
        else:
            lines.append("== Process Topology (units in process order) ==")
            try:
                ordered_units = list(self._proc.getUnitOperations())
            except Exception:
                ordered_units = []
            self._append_topology(lines, ordered_units)

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

    def _append_topology(self, lines: list, ordered_units: list):
        """Render a list of ordered unit operations into *lines*."""
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
                        if fval == 0.0 and prop == "duty_kW" and utype in self._DUTY_UNITS:
                            if hasattr(u, "getEnergyInput"):
                                try:
                                    fval = float(u.getEnergyInput()) / 1000.0
                                except Exception:
                                    pass
                        if fval == 0.0 and prop == "power_kW" and utype not in self._POWER_UNITS:
                            continue
                        if fval == 0.0 and prop == "duty_kW" and utype not in self._DUTY_UNITS:
                            continue
                        props[prop] = fval
                    except Exception:
                        pass

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

            inlet_str = ""
            for m in ("getInletStream", "getInStream", "getFeed", "getFeedStream"):
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

            outlet_strs = []
            is_separator = "Separator" in utype or "Scrubber" in utype

            if is_separator:
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
