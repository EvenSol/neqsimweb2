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
import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


_MATERIAL_BOUNDARY_ZERO_FLOW_KG_HR = 0.01
_COMPONENT_BALANCE_OK_PCT = 0.01
_COMPONENT_BALANCE_WARN_PCT = 1.0
_ENERGY_BALANCE_OK_PCT = 0.01
_ENERGY_BALANCE_WARN_PCT = 1.0
_UNIT_BALANCE_SCALE_FLOOR_KG_HR = 1.0e-9
_UNIT_BALANCE_SCALE_FLOOR_KW = 1.0e-9
_MAX_NATIVE_SPLIT_STREAM_COUNT = 256
_STANDARD_GRAVITY_M_S2 = 9.80665
_STUDIO_METADATA_MEMBER = "neqsimweb2/studio_metadata.json"
_STUDIO_METADATA_SCHEMA_VERSION = 1
_PUMP_DESIGN_CAPACITY_LIMITS = {
    "design_flow_capacity_m3_per_hr": (0.001, 1_000_000.0),
    "design_head_capacity_m": (0.1, 20_000.0),
    "motor_rating_kw": (0.001, 1_000_000.0),
}
_HEAT_EXCHANGER_DESIGN_CAPACITY_LIMITS = {
    "design_duty_capacity_kw": (0.001, 100_000_000.0),
    "design_ua_capacity_w_per_k": (1.0, 1_000_000_000.0),
}
_EQUIPMENT_DESIGN_CAPACITY_LIMITS = (
    _PUMP_DESIGN_CAPACITY_LIMITS,
    _HEAT_EXCHANGER_DESIGN_CAPACITY_LIMITS,
)
_MATERIAL_STREAM_UNIT_CLASSES = {
    "equilibriumstream",
    "stream",
    "wellstream",
}
_MATERIAL_CONNECTIVITY_UNSAFE_UNIT_CLASSES = {
    "co2electrolyzer",
    "electrolyzer",
    "tank",
}
_MATERIAL_PREFER_EXPLICIT_INLET_COLLECTION_CLASSES = {
    "gasscrubber",
    "gasscrubbersimple",
    "separator",
    "threephaseseparator",
    "twophaseseparator",
}
_MATERIAL_PRIVATE_INLET_FIELDS = {
    "co2electrolyzer": ("inletStream",),
    "electrolyzer": ("waterInlet",),
    "tank": ("inletStreamMixer",),
}
_SPECIES_CHANGING_UNIT_CLASSES = {
    "fuelcell",
    "gasturbine",
    "h2sscavenger",
    "simpleabsorber",
}
_SPECIES_CHANGING_UNIT_TOKENS = (
    "burner",
    "gasifier",
    "reformer",
    "reactive",
    "reactor",
    "electrolyzer",
    "flare",
    "combust",
    "fuelcell",
    "scavenger",
)
_SPECIES_CONSERVING_UNIT_CLASSES = {
    "absorber",
    "adiabaticpipe",
    "adiabatictwophasepipe",
    "adjuster",
    "aircooler",
    "calculator",
    "checkvalve",
    "componentsplitter",
    "compressor",
    "controlvalve",
    "cooler",
    "distillationcolumn",
    "ejector",
    "equilibriumstream",
    "esppump",
    "expander",
    "filter",
    "gasscrubber",
    "gasscrubbersimple",
    "heater",
    "heatexchanger",
    "hydrocyclone",
    "membraneseparator",
    "mixer",
    "multistreamheatexchanger",
    "pipebeggsandbrills",
    "pipeline",
    "pump",
    "recycle",
    "separator",
    "setpoint",
    "simpleflowline",
    "simpletegabsorber",
    "simpletpoutpipeline",
    "splitter",
    "stream",
    "threephaseseparator",
    "throttlingvalve",
    "turboexpandercompressor",
    "twophaseseparator",
    "valve",
    "watercooler",
    "waterstrippercolumn",
    "wellflow",
    "wellstream",
}
_ENERGY_BALANCE_ADIABATIC_UNIT_CLASSES = {
    "adjuster",
    "calculator",
    "checkvalve",
    "componentsplitter",
    "controlvalve",
    "equilibriumstream",
    "filter",
    "gasscrubber",
    "gasscrubbersimple",
    "heatexchanger",
    "mixer",
    "recycle",
    "separator",
    "setpoint",
    "splitter",
    "stream",
    "threephaseseparator",
    "throttlingvalve",
    "twophaseseparator",
    "valve",
    "wellstream",
}
_ENERGY_BALANCE_POWER_UNIT_CLASSES = {
    "compressor",
    "esppump",
    "expander",
    "pump",
}
_ENERGY_BALANCE_DUTY_UNIT_CLASSES = {
    "aircooler",
    "cooler",
    "heater",
    "watercooler",
}


def _is_native_mixer_class(java_class: str) -> bool:
    """Return whether a native unit is Mixer or one of its subclasses."""
    return str(java_class).endswith("Mixer")


def _native_split_stream_count(unit: Any) -> Optional[int]:
    """Return a validated native splitter outlet count when available."""
    try:
        split_count = int(unit.getSplitNumber())
    except Exception:
        return None
    if split_count < 0 or split_count > _MAX_NATIVE_SPLIT_STREAM_COUNT:
        return None
    return split_count


def _split_stream_probe_count(unit: Any, fallback_limit: int) -> int:
    """Prefer native splitter topology while retaining bounded legacy probing."""
    split_count = _native_split_stream_count(unit)
    if split_count is not None:
        return split_count
    return max(0, min(int(fallback_limit), _MAX_NATIVE_SPLIT_STREAM_COUNT))


class _NativeObjectIdentitySet:
    """Retain exact native or Python object references without value equality."""

    def __init__(self) -> None:
        self._python_objects: List[Any] = []
        self._java_map: Any = None
        try:
            import jpype

            if jpype.isJVMStarted():
                identity_map = jpype.JClass("java.util.IdentityHashMap")
                self._java_map = identity_map()
        except Exception:
            pass

    def contains(self, value: Any) -> bool:
        """Return whether this exact native or Python reference was recorded."""
        if self._java_map is not None:
            try:
                return bool(self._java_map.containsKey(value))
            except Exception:
                pass
        return any(recorded is value for recorded in self._python_objects)

    def add(self, value: Any) -> None:
        """Retain one exact reference, ignoring an already recorded alias."""
        if self._java_map is not None:
            try:
                self._java_map.put(value, True)
                return
            except Exception:
                pass
        if not self.contains(value):
            self._python_objects.append(value)


class _MaterialBoundaryIdentityTracker:
    """Track native stream references without relying on collision-prone hashes."""

    _ROLES = ("feed", "product")

    def __init__(self) -> None:
        self._role_streams = {
            role: _NativeObjectIdentitySet()
            for role in self._ROLES
        }

    def _validate_role(self, role: str) -> None:
        if role not in self._ROLES:
            raise ValueError(
                "Material boundary identity role must be feed or product."
            )

    def contains(self, role: str, stream: Any) -> bool:
        """Return whether this exact stream reference was recorded for a role."""
        self._validate_role(role)
        return self._role_streams[role].contains(stream)

    def add(self, role: str, stream: Any) -> None:
        """Remember one exact native or Python stream reference for a role."""
        self._validate_role(role)
        self._role_streams[role].add(stream)


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
    owner_name: str = ""


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

    def __init__(
        self,
        process_system,
        source_bytes: Optional[bytes] = None,
        enforce_acyclic_mixer_energy: bool = False,
        trusted_solved: bool = False,
        allow_direct_runs: bool = False,
        equipment_design_bases: Optional[
            Dict[str, Dict[str, float]]
        ] = None,
    ):
        """
        Args:
            process_system: A NeqSim ProcessSystem **or ProcessModel** Java object.
            source_bytes: Original file bytes for clone-by-reload.
            enforce_acyclic_mixer_energy: Recheck adiabatic mixer energy
                closure after each acyclic graph execution.
            trusted_solved: Capture solved exchanger state only when the
                adapter observed the successful run that produced it.
            allow_direct_runs: Accept direct-run exchanger UUID patterns for
                that observed successful run.
            equipment_design_bases: Validated engineering capacities keyed by
                native unit name. These do not modify the NeqSim solve.
        """
        self._proc = process_system
        self._source_bytes = source_bytes
        self._units: Dict[str, Any] = {}
        self._streams: Dict[str, Any] = {}
        self._is_process_model = self._detect_process_model(process_system)
        self._enforce_acyclic_mixer_energy = bool(
            enforce_acyclic_mixer_energy
        )
        self._direct_unit_run_provenance: Dict[
            str,
            Tuple[str, Tuple[str, str], Tuple[str, str]],
        ] = {}
        self._heat_exchanger_state_snapshots: Dict[str, Tuple[Any, ...]] = {}
        self._equipment_design_bases = {
            str(unit_name): {
                str(property_name): float(value)
                for property_name, value in basis.items()
            }
            for unit_name, basis in (equipment_design_bases or {}).items()
        }
        self._index_model_objects()
        if trusted_solved:
            self._capture_heat_exchanger_state_snapshots(
                allow_direct_runs=allow_direct_runs
            )

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

    @staticmethod
    def _read_studio_metadata(archive) -> Dict[str, Dict[str, float]]:
        """Read validated Studio-only metadata from a native NeqSim archive."""
        if _STUDIO_METADATA_MEMBER not in archive.namelist():
            return {}
        try:
            metadata = json.loads(
                archive.read(_STUDIO_METADATA_MEMBER).decode("utf-8")
            )
        except Exception as exc:
            raise RuntimeError(
                "Saved NeqSim model contains invalid Studio metadata."
            ) from exc
        if not isinstance(metadata, dict) or metadata.get(
            "schema_version"
        ) != _STUDIO_METADATA_SCHEMA_VERSION:
            raise RuntimeError(
                "Saved NeqSim model uses unsupported Studio metadata."
            )
        raw_bases = metadata.get("equipment_design_bases", {})
        if not isinstance(raw_bases, dict):
            raise RuntimeError(
                "Saved NeqSim model has invalid equipment design metadata."
            )
        bases: Dict[str, Dict[str, float]] = {}
        for unit_name, raw_basis in raw_bases.items():
            if (
                not isinstance(unit_name, str)
                or not unit_name.strip()
                or not isinstance(raw_basis, dict)
            ):
                raise RuntimeError(
                    "Saved NeqSim model has invalid equipment design metadata."
                )
            matching_limits = [
                limits
                for limits in _EQUIPMENT_DESIGN_CAPACITY_LIMITS
                if set(raw_basis) == set(limits)
            ]
            if len(matching_limits) != 1:
                raise RuntimeError(
                    "Saved NeqSim model has invalid equipment design metadata."
                )
            capacity_limits = matching_limits[0]
            basis: Dict[str, float] = {}
            for property_name, (minimum, maximum) in (
                capacity_limits.items()
            ):
                raw_value = raw_basis[property_name]
                if (
                    isinstance(raw_value, bool)
                    or not isinstance(raw_value, (int, float))
                ):
                    raise RuntimeError(
                        "Saved NeqSim model has invalid equipment design "
                        "metadata."
                    )
                value = float(raw_value)
                if not math.isfinite(value):
                    raise RuntimeError(
                        "Saved NeqSim model has non-finite equipment design "
                        "metadata."
                    )
                if not minimum <= value <= maximum:
                    raise RuntimeError(
                        "Saved NeqSim model has out-of-range equipment design "
                        "metadata."
                    )
                basis[property_name] = value
            bases[unit_name] = basis
        return bases

    @classmethod
    def from_file(cls, filepath: str) -> "NeqSimProcessModel":
        """Load a ProcessSystem and optional Studio metadata from a model file."""
        import zipfile
        import neqsim
        from neqsim import jneqsim

        with open(filepath, "rb") as f:
            file_bytes = f.read()

        loaded = None
        is_zip = zipfile.is_zipfile(filepath)
        equipment_design_bases: Dict[str, Dict[str, float]] = {}
        if is_zip:
            with zipfile.ZipFile(filepath, "r") as archive:
                equipment_design_bases = cls._read_studio_metadata(archive)
        ext = os.path.splitext(filepath)[1].lower()
        errors_seen: list = []  # collect errors for diagnostics

        if ext in (".neqsim", ".zip") or ext not in (".xml",):
            # Try our own ZIP XML extraction first — it uses
            # ignoreUnknownElements() so version-mismatched fields
            # (like tagName) are silently skipped without noisy Java logs.
            if is_zip:
                try:
                    with zipfile.ZipFile(filepath, "r") as zf:
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

            # Fallback: the library's Java-based ZIP reader
            if loaded is None:
                try:
                    loaded = neqsim.open_neqsim(filepath)
                except Exception as e:
                    errors_seen.append(f"open_neqsim: {e}")
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
        process_run_succeeded = cls._run_until_converged(loaded)
        return cls(
            loaded,
            source_bytes=file_bytes,
            trusted_solved=process_run_succeeded,
            equipment_design_bases=equipment_design_bases,
        )

    @staticmethod
    def _async_run_status_succeeded(proc: Any) -> bool:
        """Return native worker status when the process exposes it."""
        try:
            run_status = proc.getRunStatus()
        except Exception:
            return True
        if run_status is None:
            return True
        try:
            return bool(run_status.isSuccess())
        except Exception:
            return True

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
                        return False
                    if not NeqSimProcessModel._async_run_status_succeeded(
                        proc
                    ):
                        return False
                else:
                    proc.run()
            except Exception:
                return False
            return True

        # Reset recycles before the very first run
        _reset_recycles(units)

        completed_run = False
        for attempt in range(max_runs):
            # The zero-energy warm-up heuristic may keep trying after a
            # completed pass.  Success must describe the most recent attempt,
            # not any earlier one whose state a later failed run may have
            # partially overwritten.
            completed_run = False
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
                        return False
                    if not NeqSimProcessModel._async_run_status_succeeded(
                        proc
                    ):
                        continue
                else:
                    proc.run()
            except Exception:
                continue
            completed_run = True

            has_energy, total_energy = _check_energy(units)

            if not has_energy or total_energy > 1.0:
                return True

            # Still zero — reset recycles and try again
            _reset_recycles(units)
        # Zero energy is a warm-up heuristic, not proof that execution
        # failed: idle equipment and equal-temperature exchangers are valid.
        return completed_run

    @staticmethod
    def _run_acyclic_mixer_energy_closure(
        proc,
        relative_tolerance: float = 1.0e-7,
    ) -> bool:
        """Run an ordered graph pass and report whether it completed."""
        try:
            units = list(proc.getUnitOperations())
        except Exception as exc:
            raise RuntimeError(
                "Could not inspect acyclic graph units for energy closure."
            ) from exc

        has_mixer = any(
            str(unit.getClass().getSimpleName()) == "Mixer"
            for unit in units
        )
        if not has_mixer:
            return False

        from jpype import JClass
        from neqsim import jneqsim

        run_id = JClass("java.util.UUID").randomUUID()
        operations_class = (
            jneqsim.thermodynamicoperations.ThermodynamicOperations
        )
        for unit in units:
            unit.run(run_id)
            if str(unit.getClass().getSimpleName()) != "Mixer":
                continue

            target_enthalpy = float(unit.calcMixStreamEnthalpy())
            outlet_system = unit.getOutletStream().getThermoSystem()
            outlet_system.init(3)
            actual_enthalpy = float(outlet_system.getEnthalpy())
            energy_scale = max(abs(target_enthalpy), 1.0)
            relative_error = abs(
                actual_enthalpy - target_enthalpy
            ) / energy_scale
            if relative_error <= relative_tolerance:
                continue

            try:
                operations_class(outlet_system).PHflash(target_enthalpy)
                outlet_system.init(3)
            except Exception as exc:
                raise RuntimeError(
                    f"Mixer '{unit.getName()}' could not close its "
                    "adiabatic energy balance."
                ) from exc

            actual_enthalpy = float(outlet_system.getEnthalpy())
            relative_error = abs(
                actual_enthalpy - target_enthalpy
            ) / energy_scale
            if not math.isfinite(relative_error) or (
                relative_error > relative_tolerance
            ):
                raise RuntimeError(
                    f"Mixer '{unit.getName()}' energy balance did not "
                    f"converge (relative residual {relative_error:.3e})."
                )
        return True

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
    def from_process_system(
        cls,
        process_system,
        enforce_acyclic_mixer_energy: bool = False,
        trusted_solved: bool = False,
        allow_direct_runs: bool = False,
        equipment_design_bases: Optional[
            Dict[str, Dict[str, float]]
        ] = None,
    ) -> "NeqSimProcessModel":
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

        return cls(
            process_system,
            source_bytes=file_bytes,
            enforce_acyclic_mixer_energy=enforce_acyclic_mixer_energy,
            trusted_solved=trusted_solved,
            allow_direct_runs=allow_direct_runs,
            equipment_design_bases=equipment_design_bases,
        )

    # ----- Cloning -----

    def refresh_source_bytes(self):
        """Re-serialize the current process state so future clones see any
        structural modifications (added units, streams, etc.)."""
        import neqsim

        with tempfile.NamedTemporaryFile(suffix=".neqsim", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            neqsim.save_neqsim(self._proc, tmp_path)
            if self._equipment_design_bases:
                import zipfile

                if not zipfile.is_zipfile(tmp_path):
                    raise RuntimeError(
                        "Cannot preserve Studio equipment design metadata: "
                        "native NeqSim serialization is not a ZIP archive."
                    )
                metadata = {
                    "schema_version": _STUDIO_METADATA_SCHEMA_VERSION,
                    "equipment_design_bases": self._equipment_design_bases,
                }
                metadata_json = json.dumps(
                    metadata,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                with zipfile.ZipFile(
                    tmp_path,
                    "a",
                    compression=zipfile.ZIP_DEFLATED,
                ) as archive:
                    archive.writestr(
                        _STUDIO_METADATA_MEMBER,
                        metadata_json.encode("utf-8"),
                    )
            with open(tmp_path, "rb") as f:
                self._source_bytes = f.read()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def save_bytes(self) -> Optional[bytes]:
        """Return the current process state as serialized .neqsim bytes.

        Studio-only equipment design metadata is stored as a versioned JSON
        member inside the native ZIP archive. Native NeqSim readers continue
        to use the XML member and ignore the additional metadata member.
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
        clone = NeqSimProcessModel.from_bytes(self._source_bytes)
        clone._enforce_acyclic_mixer_energy = (
            self._enforce_acyclic_mixer_energy
        )
        clone._equipment_design_bases = {
            unit_name: dict(basis)
            for unit_name, basis in self._equipment_design_bases.items()
        }
        clone.rerun()
        return clone

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
        seen_streams = _NativeObjectIdentitySet()
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
                            for i in range(_split_stream_probe_count(u, 10)):
                                try:
                                    s = u.getSplitStream(i)
                                    if s is not None:
                                        sname = str(s.getName()) if s.getName() else None
                                        if sname:
                                            if seen_streams.contains(s):
                                                continue  # same Java object already indexed
                                            seen_streams.add(s)
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
                                    if seen_streams.contains(s):
                                        continue  # same Java object already indexed
                                    seen_streams.add(s)
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

    def _process_unit_groups(self) -> List[List[Any]]:
        """Return ordered unit-operation groups for material-boundary analysis."""
        process_systems = (
            self.get_process_systems() if self._is_process_model else [self._proc]
        )
        groups: List[List[Any]] = []
        for process_system in process_systems:
            try:
                units = list(process_system.getUnitOperations())
            except Exception:
                try:
                    units = list(process_system.getUnitOperationList())
                except Exception:
                    units = []
            groups.append(units)
        return groups

    @staticmethod
    def _leading_material_feed_streams(units: List[Any]) -> List[Any]:
        """Return material-stream units preceding the first process equipment."""
        utility_types = {"Recycle", "Adjuster", "Calculator", "SetPoint"}
        feeds: List[Any] = []
        for unit in units:
            try:
                unit_class = str(unit.getClass().getSimpleName())
            except Exception:
                break
            if unit_class.lower() in _MATERIAL_STREAM_UNIT_CLASSES:
                feeds.append(unit)
                continue
            if unit_class in utility_types:
                continue
            break
        return feeds

    @staticmethod
    def _trailing_material_product_streams(units: List[Any]) -> List[Any]:
        """Return material-stream units following the final process equipment."""
        utility_types = {"Recycle", "Adjuster", "Calculator", "SetPoint"}
        last_equipment_index = -1
        for index, unit in enumerate(units):
            try:
                unit_class = str(unit.getClass().getSimpleName())
            except Exception:
                continue
            if (
                unit_class.lower() not in _MATERIAL_STREAM_UNIT_CLASSES
                and unit_class not in utility_types
            ):
                last_equipment_index = index
        if last_equipment_index < 0:
            return []

        products: List[Any] = []
        for unit in units[last_equipment_index + 1:]:
            try:
                unit_class = str(unit.getClass().getSimpleName())
            except Exception:
                continue
            if unit_class.lower() in _MATERIAL_STREAM_UNIT_CLASSES:
                products.append(unit)
        return products

    @staticmethod
    def _fallback_material_outlet_streams(
        unit: Any,
    ) -> List[Tuple[Any, str]]:
        """Return every discoverable material outlet on a terminal unit."""
        outlets: List[Tuple[Any, str]] = []

        if hasattr(unit, "getOutletStreams"):
            try:
                for index, stream in enumerate(unit.getOutletStreams()):
                    if stream is not None:
                        outlets.append((stream, f"out_{index}"))
            except Exception:
                pass

        for method_name in ("getOutStream", "getSplitStream"):
            if not hasattr(unit, method_name):
                continue
            probe_count = (
                _split_stream_probe_count(unit, 100)
                if method_name == "getSplitStream"
                else 100
            )
            for index in range(probe_count):
                try:
                    stream = getattr(unit, method_name)(index)
                except Exception:
                    break
                if stream is None:
                    break
                outlets.append((stream, f"out_{index}"))

        for method_name, label in (
            ("getOutletStream", "gas_out"),
            ("getOutStream", "gas_out"),
            ("getGasOutStream", "gas_out"),
            ("getCompressorOutletStream", "compressor_out"),
            ("getExpanderOutletStream", "expander_out"),
            ("getHydrogenOutStream", "hydrogen"),
            ("getOxygenOutStream", "oxygen"),
            ("getGasProductStream", "gas_product"),
            ("getLiquidProductStream", "liquid_product"),
            ("getOilOutStream", "oil"),
            ("getLiquidOutStream", "liquid"),
            ("getWaterOutStream", "water"),
        ):
            if not hasattr(unit, method_name):
                continue
            try:
                stream = getattr(unit, method_name)()
            except Exception:
                continue
            if stream is not None:
                outlets.append((stream, label))

        return outlets

    @staticmethod
    def _material_inlet_streams(unit: Any) -> List[Any]:
        """Return every discoverable material inlet on a native unit."""
        inlets: List[Any] = []
        try:
            unit_class = str(unit.getClass().getSimpleName()).lower()
        except Exception:
            unit_class = ""

        if hasattr(unit, "getInletStreams"):
            try:
                inlets.extend(
                    stream
                    for stream in unit.getInletStreams()
                    if stream is not None
                )
            except Exception:
                pass

        if (
            inlets
            and unit_class
            in _MATERIAL_PREFER_EXPLICIT_INLET_COLLECTION_CLASSES
        ):
            unique_inlets = []
            inlet_identities = _MaterialBoundaryIdentityTracker()
            for stream in inlets:
                if inlet_identities.contains("feed", stream):
                    continue
                inlet_identities.add("feed", stream)
                unique_inlets.append(stream)
            return unique_inlets

        for method_name in (
            "getInStream",
            "getFeedStream",
            "getStream",
            "getInputStream",
        ):
            if not hasattr(unit, method_name):
                continue
            for index in range(100):
                try:
                    stream = getattr(unit, method_name)(index)
                except Exception:
                    break
                if stream is None:
                    break
                inlets.append(stream)

        for method_name in (
            "getInletStream",
            "getInStream",
            "getFeed",
            "getFeedStream",
            "getCompressorInletStream",
            "getExpanderInletStream",
            "getCompressorFeedStream",
            "getExpanderFeedStream",
            "getSolventInStream",
            "getMotiveStream",
            "getSuctionStream",
        ):
            if not hasattr(unit, method_name):
                continue
            try:
                stream = getattr(unit, method_name)()
            except Exception:
                continue
            if stream is not None:
                inlets.append(stream)

        for field_name in _MATERIAL_PRIVATE_INLET_FIELDS.get(
            unit_class,
            (),
        ):
            field_value = None
            try:
                field_value = getattr(unit, field_name)
            except Exception:
                pass
            if field_value is None:
                try:
                    declaring_class = unit.getClass()
                    while declaring_class is not None:
                        try:
                            field = declaring_class.getDeclaredField(
                                field_name
                            )
                            field.setAccessible(True)
                            field_value = field.get(unit)
                            break
                        except Exception:
                            declaring_class = (
                                declaring_class.getSuperclass()
                            )
                except Exception:
                    pass
            if field_value is None:
                continue
            try:
                field_class = str(
                    field_value.getClass().getSimpleName()
                ).lower()
            except Exception:
                field_class = ""
            if field_name == "inletStreamMixer":
                inlets.extend(
                    NeqSimProcessModel._material_inlet_streams(
                        field_value
                    )
                )
                continue
            if (
                field_class in _MATERIAL_STREAM_UNIT_CLASSES
                or hasattr(field_value, "getFluid")
            ):
                inlets.append(field_value)
                continue
            inlets.extend(
                NeqSimProcessModel._material_inlet_streams(field_value)
            )

        unique_inlets = []
        inlet_identities = _MaterialBoundaryIdentityTracker()
        for stream in inlets:
            if inlet_identities.contains("feed", stream):
                continue
            inlet_identities.add("feed", stream)
            unique_inlets.append(stream)
        return unique_inlets

    @staticmethod
    def _material_fluid_reference(stream: Any) -> Optional[Any]:
        """Return the native fluid identity used to recognize stream aliases."""
        for method_name in ("getFluid", "getThermoSystem"):
            if not hasattr(stream, method_name):
                continue
            try:
                fluid = getattr(stream, method_name)()
            except Exception:
                continue
            if fluid is not None:
                return fluid
        return None

    @staticmethod
    def _material_stream_source_reference(stream: Any) -> Optional[Any]:
        """Return the specific upstream stream wrapped by a stream alias."""
        for method_name in (
            "getSourceStream",
            "getUpstreamStream",
            "getParentStream",
        ):
            if not hasattr(stream, method_name):
                continue
            try:
                source = getattr(stream, method_name)()
            except Exception:
                continue
            if source is not None and source is not stream:
                return source

        try:
            stream_class = stream.getClass()
            if (
                str(stream_class.getSimpleName()).lower()
                not in _MATERIAL_STREAM_UNIT_CLASSES
            ):
                return None
            while stream_class is not None:
                try:
                    field = stream_class.getDeclaredField("stream")
                    field.setAccessible(True)
                    source = field.get(stream)
                    if source is not None and source is not stream:
                        return source
                    return None
                except Exception:
                    stream_class = stream_class.getSuperclass()
        except Exception:
            pass
        return None

    @staticmethod
    def _material_consumption_trackers(
        units: List[Any],
    ) -> Tuple[
        _MaterialBoundaryIdentityTracker,
        _MaterialBoundaryIdentityTracker,
    ]:
        """Return native stream and fluid identities consumed by equipment."""
        consumed_streams = _MaterialBoundaryIdentityTracker()
        consumed_fluids = _MaterialBoundaryIdentityTracker()
        for unit in units:
            try:
                unit_class = str(
                    unit.getClass().getSimpleName()
                ).lower()
            except Exception:
                continue
            if unit_class in _MATERIAL_STREAM_UNIT_CLASSES:
                continue
            for stream in NeqSimProcessModel._material_inlet_streams(unit):
                consumed_streams.add("feed", stream)
                fluid = NeqSimProcessModel._material_fluid_reference(
                    stream
                )
                if fluid is not None:
                    consumed_fluids.add("feed", fluid)
        return consumed_streams, consumed_fluids

    @staticmethod
    def _connectivity_material_boundaries(
        units: List[Any],
    ) -> Tuple[List[Any], List[Tuple[Any, str]]]:
        """Discover external sources and terminal sinks from native ports."""
        consumed, consumed_fluids = (
            NeqSimProcessModel._material_consumption_trackers(units)
        )
        produced = _MaterialBoundaryIdentityTracker()
        produced_fluids = _MaterialBoundaryIdentityTracker()
        stream_units: List[Any] = []
        equipment_outlets: List[Tuple[Any, str]] = []

        for unit in units:
            try:
                unit_class = str(
                    unit.getClass().getSimpleName()
                ).lower()
            except Exception:
                continue
            if unit_class in _MATERIAL_STREAM_UNIT_CLASSES:
                stream_units.append(unit)
                continue
            for stream, label in (
                NeqSimProcessModel._fallback_material_outlet_streams(unit)
            ):
                produced.add("product", stream)
                fluid = NeqSimProcessModel._material_fluid_reference(
                    stream
                )
                if fluid is not None:
                    produced_fluids.add("product", fluid)
                equipment_outlets.append((stream, label))

        feed_candidates = []
        candidate_streams = _MaterialBoundaryIdentityTracker()
        for stream in stream_units:
            fluid = NeqSimProcessModel._material_fluid_reference(stream)
            is_consumed = consumed.contains("feed", stream) or (
                fluid is not None
                and consumed_fluids.contains("feed", fluid)
            )
            is_produced = produced.contains("product", stream) or (
                fluid is not None
                and produced_fluids.contains("product", fluid)
            )
            if is_consumed and not is_produced:
                feed_candidates.append(
                    (
                        stream,
                        NeqSimProcessModel._material_stream_source_reference(
                            stream
                        ),
                    )
                )
                candidate_streams.add("feed", stream)

        feeds = []
        for stream, source_stream in feed_candidates:
            if (
                source_stream is not None
                and candidate_streams.contains("feed", source_stream)
            ):
                continue
            feeds.append(stream)

        products = []
        for stream, label in equipment_outlets:
            fluid = NeqSimProcessModel._material_fluid_reference(stream)
            is_consumed = consumed.contains("feed", stream) or (
                fluid is not None
                and consumed_fluids.contains("feed", fluid)
            )
            if not is_consumed:
                products.append((stream, label))
        return feeds, products

    @staticmethod
    def _component_balance_exclusion_names(
        units: List[Any],
    ) -> List[str]:
        """Return species-changing or unclassified native equipment."""
        excluded_units: List[str] = []
        for unit in units:
            try:
                unit_class = str(unit.getClass().getSimpleName())
            except Exception:
                continue
            normalized_class = unit_class.lower()
            reactive_mode = False
            if (
                normalized_class == "distillationcolumn"
                and hasattr(unit, "isReactive")
            ):
                try:
                    reactive_mode = bool(unit.isReactive())
                except Exception:
                    pass
            species_changing = (
                reactive_mode
                or normalized_class in _SPECIES_CHANGING_UNIT_CLASSES
                or any(
                    token in normalized_class
                    for token in _SPECIES_CHANGING_UNIT_TOKENS
                )
            )
            if (
                not species_changing
                and normalized_class in _SPECIES_CONSERVING_UNIT_CLASSES
            ):
                continue
            try:
                unit_name = str(unit.getName()).strip()
            except Exception:
                unit_name = ""
            label = unit_name or unit_class
            if not species_changing:
                label = f"{label} (unclassified {unit_class})"
            excluded_units.append(label)
        return excluded_units

    @staticmethod
    def _system_energy_transfers(
        units: List[Any],
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Return audited signed energy transfers and excluded equipment."""
        transfers: List[Dict[str, Any]] = []
        excluded_units: List[str] = []
        for unit in units:
            try:
                unit_class = str(unit.getClass().getSimpleName())
            except Exception:
                continue
            normalized_class = unit_class.lower()
            try:
                unit_name = str(unit.getName()).strip()
            except Exception:
                unit_name = ""
            label = unit_name or unit_class

            if normalized_class == "pipebeggsandbrills":
                try:
                    heat_transfer_mode = str(
                        unit.getHeatTransferMode()
                    ).strip().upper()
                except Exception:
                    heat_transfer_mode = ""
                if heat_transfer_mode == "ADIABATIC":
                    continue
                excluded_units.append(
                    f"{label} (unaudited {unit_class} "
                    f"heat-transfer mode {heat_transfer_mode or 'unknown'})"
                )
                continue

            if normalized_class in _ENERGY_BALANCE_ADIABATIC_UNIT_CLASSES:
                continue
            if normalized_class in _ENERGY_BALANCE_POWER_UNIT_CLASSES:
                getter_names = ("getPower",)
                transfer_kind = "shaft_work"
            elif normalized_class in _ENERGY_BALANCE_DUTY_UNIT_CLASSES:
                getter_names = ("getDuty", "getEnergyInput")
                transfer_kind = "heat"
            else:
                excluded_units.append(
                    f"{label} (unaudited {unit_class})"
                )
                continue

            energy_transfer_w = None
            for getter_name in getter_names:
                if not hasattr(unit, getter_name):
                    continue
                try:
                    candidate = float(getattr(unit, getter_name)())
                except Exception:
                    continue
                if math.isfinite(candidate):
                    energy_transfer_w = candidate
                    if candidate != 0.0 or getter_name == getter_names[-1]:
                        break
            if energy_transfer_w is None:
                excluded_units.append(
                    f"{label} ({transfer_kind} unavailable)"
                )
                continue
            transfers.append(
                {
                    "unit_name": label,
                    "unit_type": unit_class,
                    "transfer_kind": transfer_kind,
                    "energy_transfer_kW": energy_transfer_w / 1000.0,
                }
            )
        return transfers, excluded_units

    @staticmethod
    def _distinct_material_outlet_streams(unit: Any) -> List[Any]:
        """Return distinct native material outlets for one unit."""
        tracker = _MaterialBoundaryIdentityTracker()
        streams: List[Any] = []
        for stream, _ in (
            NeqSimProcessModel._fallback_material_outlet_streams(unit)
        ):
            if tracker.contains("product", stream):
                continue
            tracker.add("product", stream)
            streams.append(stream)
        return streams

    @staticmethod
    def _unit_external_energy_transfer_kW(
        unit: Any,
    ) -> Optional[float]:
        """Return audited signed external energy for one unit."""
        transfers, excluded_units = (
            NeqSimProcessModel._system_energy_transfers([unit])
        )
        if excluded_units:
            return None
        return sum(
            float(transfer["energy_transfer_kW"])
            for transfer in transfers
        )

    def _extract_unit_balance_diagnostics(self) -> Dict[str, Any]:
        """Capture explicit-port mass and energy closure for each unit."""
        rows: List[Dict[str, Any]] = []
        excluded_units: List[str] = []
        control_unit_classes = {
            "adjuster",
            "calculator",
            "recycle",
            "setpoint",
        }
        process_name_counts: Dict[str, int] = {}

        for process_index, process_system in enumerate(
            self.get_process_systems()
        ):
            try:
                process_name = str(process_system.getName()).strip()
            except Exception:
                process_name = ""
            if not process_name or process_name.lower() == "null":
                process_name = f"process {process_index + 1}"
            process_name_counts[process_name] = (
                process_name_counts.get(process_name, 0) + 1
            )
            if process_name_counts[process_name] > 1:
                process_name = (
                    f"{process_name} [{process_name_counts[process_name]}]"
                )
            try:
                units = list(process_system.getUnitOperations())
            except Exception:
                try:
                    units = list(process_system.getUnitOperationList())
                except Exception:
                    units = []

            unit_name_counts: Dict[Tuple[str, str], int] = {}
            for unit in units:
                try:
                    unit_type = str(
                        unit.getClass().getSimpleName()
                    ).strip()
                except Exception:
                    continue
                normalized_type = unit_type.lower()
                if (
                    normalized_type in _MATERIAL_STREAM_UNIT_CLASSES
                    or normalized_type in control_unit_classes
                ):
                    continue
                try:
                    unit_name = str(unit.getName()).strip()
                except Exception:
                    unit_name = ""
                unit_name = unit_name or unit_type
                unit_identity = (unit_name, unit_type)
                unit_name_counts[unit_identity] = (
                    unit_name_counts.get(unit_identity, 0) + 1
                )
                if unit_name_counts[unit_identity] > 1:
                    unit_name = (
                        f"{unit_name} [{unit_name_counts[unit_identity]}]"
                    )
                unit_label = (
                    f"{process_name}/{unit_name} ({unit_type})"
                )

                inlet_streams = self._material_inlet_streams(unit)
                outlet_streams = (
                    self._distinct_material_outlet_streams(unit)
                )
                if not inlet_streams or not outlet_streams:
                    excluded_units.append(
                        f"{unit_label}: explicit material ports unavailable"
                    )
                    continue

                try:
                    inlet_records = [
                        self._material_boundary_record(
                            stream,
                            "feed",
                            f"{unit_name} inlet {index + 1}",
                        )
                        for index, stream in enumerate(inlet_streams)
                    ]
                    outlet_records = [
                        self._material_boundary_record(
                            stream,
                            "product",
                            f"{unit_name} outlet {index + 1}",
                        )
                        for index, stream in enumerate(outlet_streams)
                    ]
                except ValueError as exc:
                    excluded_units.append(f"{unit_label}: {exc}")
                    continue

                inlet_mass = sum(
                    float(record["mass_flow_kg_hr"])
                    for record in inlet_records
                )
                outlet_mass = sum(
                    float(record["mass_flow_kg_hr"])
                    for record in outlet_records
                )
                mass_residual = outlet_mass - inlet_mass
                mass_scale = max(
                    abs(inlet_mass),
                    abs(outlet_mass),
                    _UNIT_BALANCE_SCALE_FLOOR_KG_HR,
                )
                row: Dict[str, Any] = {
                    "process_system": process_name,
                    "unit_name": unit_name,
                    "unit_type": unit_type,
                    "inlet_count": len(inlet_records),
                    "outlet_count": len(outlet_records),
                    "inlet_mass_flow_kg_hr": inlet_mass,
                    "outlet_mass_flow_kg_hr": outlet_mass,
                    "mass_residual_kg_hr": mass_residual,
                    "mass_imbalance_pct": (
                        abs(mass_residual) / mass_scale * 100.0
                    ),
                    "inlet_enthalpy_kW": None,
                    "outlet_enthalpy_kW": None,
                    "external_energy_transfer_kW": None,
                    "energy_residual_kW": None,
                    "energy_imbalance_pct": None,
                }

                inlet_enthalpies = [
                    record["enthalpy_flow_kW"]
                    for record in inlet_records
                ]
                outlet_enthalpies = [
                    record["enthalpy_flow_kW"]
                    for record in outlet_records
                ]
                external_transfer = (
                    self._unit_external_energy_transfer_kW(unit)
                )
                if (
                    external_transfer is not None
                    and all(
                        value is not None
                        for value in (
                            inlet_enthalpies + outlet_enthalpies
                        )
                    )
                ):
                    inlet_enthalpy = sum(
                        float(value) for value in inlet_enthalpies
                    )
                    outlet_enthalpy = sum(
                        float(value) for value in outlet_enthalpies
                    )
                    energy_residual = (
                        outlet_enthalpy
                        - inlet_enthalpy
                        - external_transfer
                    )
                    energy_scale = max(
                        abs(inlet_enthalpy),
                        abs(outlet_enthalpy),
                        abs(external_transfer),
                        _UNIT_BALANCE_SCALE_FLOOR_KW,
                    )
                    row.update(
                        {
                            "inlet_enthalpy_kW": inlet_enthalpy,
                            "outlet_enthalpy_kW": outlet_enthalpy,
                            "external_energy_transfer_kW": (
                                external_transfer
                            ),
                            "energy_residual_kW": energy_residual,
                            "energy_imbalance_pct": (
                                abs(energy_residual)
                                / energy_scale
                                * 100.0
                            ),
                        }
                    )
                rows.append(row)

        excluded_units = list(dict.fromkeys(excluded_units))
        return {
            "applicable": bool(rows),
            "coverage_complete": not excluded_units,
            "rows": rows,
            "excluded_units": excluded_units,
        }

    @staticmethod
    def _material_boundary_component_flows(
        stream: Any,
        total_molar_flow: Optional[float],
    ) -> Optional[Dict[str, float]]:
        """Return solved overall component molar flows in mol/s when available."""
        if total_molar_flow is None or not hasattr(stream, "getFluid"):
            return None
        try:
            fluid = stream.getFluid()
            phase = fluid.getPhase(0)
            component_count = int(phase.getNumberOfComponents())
        except Exception:
            return None

        component_flows: Dict[str, float] = {}
        for index in range(component_count):
            try:
                component = phase.getComponent(index)
                name = str(component.getName()).strip()
                overall_fraction = float(component.getz())
            except Exception:
                return None
            if (
                not name
                or name in component_flows
                or not math.isfinite(overall_fraction)
                or overall_fraction < -1.0e-12
            ):
                return None
            component_flow = total_molar_flow * max(overall_fraction, 0.0)
            if not math.isfinite(component_flow):
                return None
            component_flows[name] = component_flow
        return component_flows

    @staticmethod
    def _material_boundary_record(
        stream: Any,
        role: str,
        fallback_name: str,
    ) -> Dict[str, Any]:
        """Return one explicit-unit record for a solved material boundary."""
        if role not in {"feed", "product"}:
            raise ValueError("Material boundary role must be feed or product.")
        try:
            name = str(stream.getName()) if stream.getName() else fallback_name
            mass_flow = float(stream.getFlowRate("kg/hr"))
        except Exception as exc:
            raise ValueError(
                f"Could not read solved {role} boundary '{fallback_name}'."
            ) from exc
        if not math.isfinite(mass_flow):
            raise ValueError(
                f"Solved {role} boundary '{name}' has a non-finite mass flow."
            )
        is_no_flow = (
            abs(mass_flow) <= _MATERIAL_BOUNDARY_ZERO_FLOW_KG_HR
        )
        if is_no_flow:
            mass_flow = 0.0

        record: Dict[str, Any] = {
            "role": role,
            "stream_name": name,
            "mass_flow_kg_hr": mass_flow,
            "temperature_C": None,
            "pressure_bara": None,
            "molar_flow_mol_sec": None,
            "enthalpy_flow_kW": None,
            "component_molar_flows_mol_sec": None,
        }
        for key, getter_name, unit in (
            ("temperature_C", "getTemperature", "C"),
            ("pressure_bara", "getPressure", "bara"),
            ("molar_flow_mol_sec", "getFlowRate", "mol/sec"),
        ):
            try:
                value = float(getattr(stream, getter_name)(unit))
            except Exception:
                continue
            if math.isfinite(value):
                record[key] = value
        if is_no_flow:
            record["molar_flow_mol_sec"] = 0.0
            record["enthalpy_flow_kW"] = 0.0
        else:
            try:
                fluid = stream.getFluid()
                fluid.init(3)
                enthalpy_flow_kW = float(fluid.getEnthalpy()) / 1000.0
            except Exception:
                enthalpy_flow_kW = None
            if (
                enthalpy_flow_kW is not None
                and math.isfinite(enthalpy_flow_kW)
            ):
                record["enthalpy_flow_kW"] = enthalpy_flow_kW
        record["component_molar_flows_mol_sec"] = (
            NeqSimProcessModel._material_boundary_component_flows(
                stream,
                record["molar_flow_mol_sec"],
            )
        )
        return record

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
                for i in range(_split_stream_probe_count(u, 10)):
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
                    duty_is_trusted = (
                        cls != "HeatExchanger"
                        or self._heat_exchanger_solution_is_trusted(
                            self._indexed_unit_name_for_native(
                                _u,
                                name,
                            ),
                            _u,
                            cls,
                        )
                    )
                    if duty_is_trusted:
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

    @staticmethod
    def _pump_operating_properties(unit: Any) -> Dict[str, float]:
        """Return finite solved pump properties using supported native APIs.

        NeqSim ESP pumps expose ``getActualHead``, which reflects stage
        head and gas-void-fraction degradation. Other pressure-specified pumps
        do not expose native head, so their hydraulic head is derived from the
        solved pressure rise and inlet density. Hydraulic power is derived from
        pressure rise and inlet actual volumetric flow.
        """
        properties: Dict[str, float] = {}
        try:
            native_unit_class = (
                str(unit.getClass().getSimpleName()).strip().lower()
            )
        except Exception:
            native_unit_class = ""
        for property_name, getter_name in (
            ("inletPressure_bara", "getInletPressure"),
            ("outletPressure_bara", "getOutletPressure"),
            ("inletTemperature_K", "getInletTemperature"),
            ("outletTemperature_K", "getOutletTemperature"),
            ("efficiency", "getIsentropicEfficiency"),
            ("speed_rpm", "getSpeed"),
        ):
            if not hasattr(unit, getter_name):
                continue
            try:
                value = float(getattr(unit, getter_name)())
            except Exception:
                continue
            if (
                property_name == "efficiency"
                and native_unit_class == "esppump"
            ):
                value /= 100.0
            if math.isfinite(value):
                properties[property_name] = value

        if hasattr(unit, "getPower"):
            try:
                shaft_power_kW = float(unit.getPower()) / 1000.0
                if math.isfinite(shaft_power_kW):
                    properties["shaftPower_kW"] = shaft_power_kW
            except Exception:
                pass

        try:
            inlet_stream = unit.getInletStream()
            density_kg_m3 = float(
                inlet_stream.getFluid().getDensity("kg/m3")
            )
            volumetric_flow_m3_s = float(
                inlet_stream.getFlowRate("m3/sec")
            )
        except Exception:
            return properties

        if (
            not math.isfinite(density_kg_m3)
            or density_kg_m3 <= 0.0
            or not math.isfinite(volumetric_flow_m3_s)
            or volumetric_flow_m3_s < 0.0
        ):
            return properties
        properties["inletDensity_kg_m3"] = density_kg_m3
        properties["inletVolumetricFlow_m3_s"] = volumetric_flow_m3_s

        inlet_pressure = properties.get("inletPressure_bara")
        outlet_pressure = properties.get("outletPressure_bara")
        if inlet_pressure is None or outlet_pressure is None:
            return properties
        pressure_rise_bar = outlet_pressure - inlet_pressure
        if not math.isfinite(pressure_rise_bar):
            return properties

        properties["pressureRise_bar"] = pressure_rise_bar
        pressure_rise_pa = pressure_rise_bar * 1.0e5
        if hasattr(unit, "getActualHead"):
            try:
                actual_head_m = float(unit.getActualHead())
            except Exception:
                actual_head_m = math.nan
            if math.isfinite(actual_head_m) and actual_head_m >= 0.0:
                properties["head_m"] = actual_head_m
        else:
            properties["head_m"] = (
                pressure_rise_pa
                / (density_kg_m3 * _STANDARD_GRAVITY_M_S2)
            )
        properties["hydraulicPower_kW"] = (
            pressure_rise_pa * volumetric_flow_m3_s / 1000.0
        )
        return properties

    def _pump_design_properties(
        self,
        unit_name: str,
        unit: Any,
    ) -> Dict[str, float]:
        """Compare a solved pump with its opt-in engineering capacities."""
        basis = getattr(self, "_equipment_design_bases", {}).get(unit_name)
        if (
            not basis
            or set(basis) != set(_PUMP_DESIGN_CAPACITY_LIMITS)
        ):
            return {}
        properties = {
            "designFlowCapacity_m3_per_hr": basis[
                "design_flow_capacity_m3_per_hr"
            ],
            "designHeadCapacity_m": basis["design_head_capacity_m"],
            "motorRating_kW": basis["motor_rating_kw"],
        }
        operating = self._pump_operating_properties(unit)
        comparisons = (
            (
                "inletVolumetricFlow_m3_s",
                3600.0,
                "designFlowCapacity_m3_per_hr",
                "flowUtilization_pct",
                "flowMargin_m3_per_hr",
            ),
            (
                "head_m",
                1.0,
                "designHeadCapacity_m",
                "headUtilization_pct",
                "headMargin_m",
            ),
            (
                "shaftPower_kW",
                1.0,
                "motorRating_kW",
                "motorUtilization_pct",
                "motorMargin_kW",
            ),
        )
        for (
            operating_key,
            conversion,
            capacity_key,
            utilization_key,
            margin_key,
        ) in comparisons:
            raw_value = operating.get(operating_key)
            if raw_value is None:
                continue
            actual_value = float(raw_value) * conversion
            capacity = properties[capacity_key]
            if not math.isfinite(actual_value):
                continue
            properties[utilization_key] = 100.0 * actual_value / capacity
            properties[margin_key] = capacity - actual_value
        return properties

    @staticmethod
    def _pump_design_property_unit(property_name: str) -> str:
        return {
            "designFlowCapacity_m3_per_hr": "m3/hr",
            "designHeadCapacity_m": "m",
            "motorRating_kW": "kW",
            "flowUtilization_pct": "%",
            "headUtilization_pct": "%",
            "motorUtilization_pct": "%",
            "flowMargin_m3_per_hr": "m3/hr",
            "headMargin_m": "m",
            "motorMargin_kW": "kW",
        }[property_name]

    def _pump_design_constraint(
        self,
        unit_name: str,
        unit: Any,
    ) -> ConstraintStatus:
        """Return a fail-loud capacity status for one designed pump."""
        properties = self._pump_design_properties(unit_name, unit)
        utilization_names = (
            ("flow", "flowUtilization_pct", "flowMargin_m3_per_hr"),
            ("head", "headUtilization_pct", "headMargin_m"),
            ("motor", "motorUtilization_pct", "motorMargin_kW"),
        )
        if any(
            utilization not in properties or margin not in properties
            for _, utilization, margin in utilization_names
        ):
            return ConstraintStatus(
                f"pump_design.{unit_name}",
                "UNKNOWN",
                "Pump design basis is active, but complete native flow, "
                "head, and shaft-power results are unavailable.",
            )
        violations = [
            label
            for label, _, margin in utilization_names
            if properties[margin] < -1.0e-9
        ]
        utilization = ", ".join(
            f"{label}={properties[key]:.6g}%"
            for label, key, _ in utilization_names
        )
        return ConstraintStatus(
            f"pump_design.{unit_name}",
            "VIOLATION" if violations else "OK",
            (
                "Pump exceeds " + ", ".join(violations)
                + f" capacity; utilization: {utilization}."
                if violations
                else f"Pump is inside design capacities; utilization: {utilization}."
            ),
        )

    def _heat_exchanger_design_properties(
        self,
        unit_name: str,
        unit: Any,
    ) -> Dict[str, float]:
        """Compare a solved two-sided exchanger with explicit capacities."""
        basis = getattr(self, "_equipment_design_bases", {}).get(unit_name)
        if (
            not basis
            or set(basis) != set(_HEAT_EXCHANGER_DESIGN_CAPACITY_LIMITS)
        ):
            return {}
        properties = {
            "designDutyCapacity_kW": basis["design_duty_capacity_kw"],
            "designUACapacity_W_K": basis[
                "design_ua_capacity_w_per_k"
            ],
        }
        operating = self._heat_exchanger_operating_properties(
            unit,
            getattr(self, "_direct_unit_run_provenance", {}).get(unit_name),
            getattr(self, "_heat_exchanger_state_snapshots", {}).get(
                unit_name
            ),
        )
        comparisons = (
            (
                "heatTransferDuty_kW",
                "designDutyCapacity_kW",
                "dutyUtilization_pct",
                "dutyMargin_kW",
            ),
            (
                "UA_W_K",
                "designUACapacity_W_K",
                "uaUtilization_pct",
                "uaMargin_W_K",
            ),
        )
        for operating_key, capacity_key, utilization_key, margin_key in (
            comparisons
        ):
            actual_value = operating.get(operating_key)
            if actual_value is None or not math.isfinite(actual_value):
                continue
            capacity = properties[capacity_key]
            properties[utilization_key] = (
                100.0 * abs(actual_value) / capacity
            )
            properties[margin_key] = capacity - abs(actual_value)
        return properties

    @staticmethod
    def _heat_exchanger_design_property_unit(property_name: str) -> str:
        """Return explicit units for exchanger design-basis results."""
        return {
            "designDutyCapacity_kW": "kW",
            "designUACapacity_W_K": "W/K",
            "dutyUtilization_pct": "%",
            "uaUtilization_pct": "%",
            "dutyMargin_kW": "kW",
            "uaMargin_W_K": "W/K",
        }[property_name]

    def _heat_exchanger_design_constraint(
        self,
        unit_name: str,
        unit: Any,
    ) -> ConstraintStatus:
        """Return a fail-loud duty and UA capacity status."""
        properties = self._heat_exchanger_design_properties(unit_name, unit)
        utilization_names = (
            ("duty", "dutyUtilization_pct", "dutyMargin_kW"),
            ("UA", "uaUtilization_pct", "uaMargin_W_K"),
        )
        if any(
            utilization not in properties or margin not in properties
            for _, utilization, margin in utilization_names
        ):
            return ConstraintStatus(
                f"heat_exchanger_design.{unit_name}",
                "UNKNOWN",
                "Heat-exchanger design basis is active, but complete "
                "trusted duty and UA results are unavailable.",
            )
        violations = [
            label
            for label, _, margin in utilization_names
            if properties[margin] < -1.0e-9
        ]
        utilization = ", ".join(
            f"{label}={properties[key]:.6g}%"
            for label, key, _ in utilization_names
        )
        return ConstraintStatus(
            f"heat_exchanger_design.{unit_name}",
            "VIOLATION" if violations else "OK",
            (
                "Heat exchanger exceeds " + ", ".join(violations)
                + f" capacity; utilization: {utilization}."
                if violations
                else (
                    "Heat exchanger is inside design capacities; "
                    f"utilization: {utilization}."
                )
            ),
        )

    @staticmethod
    def _compressor_map_properties(unit: Any) -> Dict[str, Any]:
        """Return finite native map state and operating-limit margins.

        Properties are exposed only when the compressor is actively solving
        speed against a chart with at least three corrected-speed curves.
        Positive surge and stonewall distances indicate an operating point
        inside the respective map boundary.
        """
        try:
            if not bool(unit.isSolveSpeed()):
                return {}
            chart = unit.getCompressorChart()
            corrected_speeds = [float(value) for value in chart.getSpeeds()]
        except Exception:
            return {}
        if (
            len(corrected_speeds) < 3
            or any(
                not math.isfinite(value) or value <= 0.0
                for value in corrected_speeds
            )
        ):
            return {}

        properties: Dict[str, Any] = {
            "mapEnabled": True,
            "mapSpeedCurveCount": len(corrected_speeds),
            "mapMinimumSpeed_rpm": min(corrected_speeds),
            "mapMaximumSpeed_rpm": max(corrected_speeds),
        }
        numeric_getters = (
            ("mapOperatingSpeed_rpm", "getSpeed"),
            ("mapSpeedRatioToMinimum", "getRatioToMinSpeed"),
            ("mapSpeedRatioToMaximum", "getRatioToMaxSpeed"),
            ("mapDistanceToSurgeFraction", "getDistanceToSurge"),
            (
                "mapDistanceToStoneWallFraction",
                "getDistanceToStoneWall",
            ),
            ("mapSurgeFlowRate_m3_per_hr", "getSurgeFlowRate"),
            (
                "mapSurgeFlowMargin_m3_per_hr",
                "getSurgeFlowRateMargin",
            ),
        )
        for property_name, getter_name in numeric_getters:
            if not hasattr(unit, getter_name):
                continue
            try:
                value = float(getattr(unit, getter_name)())
            except Exception:
                continue
            if math.isfinite(value):
                properties[property_name] = value

        try:
            below_minimum = bool(unit.isLowerThanMinSpeed())
            above_maximum = bool(unit.isHigherThanMaxSpeed())
            properties["mapWithinSpeedRange"] = not (
                below_minimum or above_maximum
            )
        except Exception:
            pass
        for property_name, getter_name in (
            ("mapInSurge", "isSurge"),
            ("mapInStoneWall", "isStoneWall"),
        ):
            try:
                properties[property_name] = bool(
                    getattr(unit, getter_name)()
                )
            except Exception:
                pass
        return properties

    @staticmethod
    def _compressor_map_property_unit(property_name: str) -> str:
        """Return the explicit engineering unit for compressor-map data."""
        if property_name.endswith("_rpm"):
            return "rpm"
        if property_name.endswith("_m3_per_hr"):
            return "m3/hr"
        if property_name.endswith("Fraction"):
            return "-"
        if property_name == "mapSpeedCurveCount":
            return "curves"
        return "-"

    @staticmethod
    def _heat_exchanger_operating_properties(
        unit: Any,
        direct_run_provenance: Optional[
            Tuple[str, Tuple[str, str], Tuple[str, str]]
        ] = None,
        solved_state_snapshot: Optional[Tuple[Any, ...]] = None,
    ) -> Dict[str, float]:
        """Return explicit solved hot/cold-side heat-exchanger properties.

        Native stream indices preserve insertion order, so the hot and cold
        sides are classified from solved inlet temperatures. Side duties are
        positive heat-transfer magnitudes. Properties are withheld unless both
        sides have complete, nonzero, finite solved states and the native unit
        records a completed calculation.
        """
        try:
            calculation_identifier = unit.getCalculationIdentifier()
        except Exception:
            return {}
        if calculation_identifier is None:
            return {}
        if solved_state_snapshot is not None:
            if (
                not solved_state_snapshot
                or str(calculation_identifier)
                != solved_state_snapshot[0]
            ):
                return {}
            current_snapshot = (
                NeqSimProcessModel._heat_exchanger_boundary_state_signature(
                    unit
                )
            )
            if current_snapshot != solved_state_snapshot:
                return {}

        indexed_sides: List[Dict[str, Any]] = []
        inlet_identifiers_match_exchanger: List[bool] = []
        inlet_calculation_identifiers: List[str] = []
        outlet_calculation_identifiers: List[str] = []
        for index in (0, 1):
            streams: Dict[str, Any] = {}
            for boundary, getter_name in (
                ("inlet", "getInStream"),
                ("outlet", "getOutStream"),
            ):
                try:
                    streams[boundary] = getattr(unit, getter_name)(index)
                except Exception:
                    return {}

            side_state: Dict[str, Any] = {"index": index}
            for boundary, stream in streams.items():
                try:
                    stream_calculation_identifier = (
                        stream.getCalculationIdentifier()
                    )
                    if stream_calculation_identifier is None:
                        return {}
                    if (
                        boundary == "outlet"
                        and str(stream_calculation_identifier)
                        != str(calculation_identifier)
                    ):
                        return {}
                    if boundary == "inlet":
                        inlet_calculation_identifiers.append(
                            str(stream_calculation_identifier)
                        )
                        inlet_identifiers_match_exchanger.append(
                            str(stream_calculation_identifier)
                            == str(calculation_identifier)
                        )
                    else:
                        outlet_calculation_identifiers.append(
                            str(stream_calculation_identifier)
                        )
                    temperature_C = float(stream.getTemperature("C"))
                    pressure_bara = float(stream.getPressure("bara"))
                    flow_kg_hr = float(stream.getFlowRate("kg/hr"))
                    fluid = stream.getFluid()
                    fluid.init(3)
                    enthalpy_flow_kW = (
                        float(fluid.getEnthalpy()) / 1000.0
                    )
                except Exception:
                    return {}
                values = (
                    temperature_C,
                    pressure_bara,
                    flow_kg_hr,
                    enthalpy_flow_kW,
                )
                if not all(math.isfinite(value) for value in values):
                    return {}
                if abs(flow_kg_hr) <= _MATERIAL_BOUNDARY_ZERO_FLOW_KG_HR:
                    return {}
                side_state[boundary] = {
                    "temperature_C": temperature_C,
                    "pressure_bara": pressure_bara,
                    "flow_kg_hr": flow_kg_hr,
                    "enthalpy_flow_kW": enthalpy_flow_kW,
                }
            indexed_sides.append(side_state)

        if not all(inlet_identifiers_match_exchanger):
            current_provenance = (
                str(calculation_identifier),
                tuple(inlet_calculation_identifiers),
                tuple(outlet_calculation_identifiers),
            )
            if direct_run_provenance != current_provenance:
                return {}

        indexed_sides.sort(
            key=lambda side: side["inlet"]["temperature_C"],
            reverse=True,
        )
        properties: Dict[str, float] = {}
        named_sides = {
            "hot": indexed_sides[0],
            "cold": indexed_sides[1],
        }
        for side_name, side_state in named_sides.items():
            for boundary_name in ("inlet", "outlet"):
                state = side_state[boundary_name]
                property_boundary = boundary_name.capitalize()
                properties[
                    f"{side_name}{property_boundary}Temperature_C"
                ] = state["temperature_C"]
                properties[
                    f"{side_name}{property_boundary}Pressure_bara"
                ] = state["pressure_bara"]
                properties[
                    f"{side_name}{property_boundary}Flow_kg_hr"
                ] = state["flow_kg_hr"]

        try:
            flow_arrangement = str(
                unit.getFlowArrangement()
            ).strip().casefold()
        except Exception:
            flow_arrangement = ""
        is_co_current = any(
            marker in flow_arrangement
            for marker in (
                "co-current",
                "co current",
                "cocurrent",
                "parallel",
            )
        )
        if is_co_current:
            terminal_differences_K = (
                named_sides["hot"]["inlet"]["temperature_C"]
                - named_sides["cold"]["inlet"]["temperature_C"],
                named_sides["hot"]["outlet"]["temperature_C"]
                - named_sides["cold"]["outlet"]["temperature_C"],
            )
        else:
            terminal_differences_K = (
                named_sides["hot"]["inlet"]["temperature_C"]
                - named_sides["cold"]["outlet"]["temperature_C"],
                named_sides["hot"]["outlet"]["temperature_C"]
                - named_sides["cold"]["inlet"]["temperature_C"],
            )
        properties["approachTemperature_K"] = min(
            terminal_differences_K
        )

        for property_name, getter_name, scale in (
            ("UA_W_K", "getUAvalue", 1.0),
            ("heatTransferDuty_kW", "getDuty", 1.0 / 1000.0),
            ("thermalEffectiveness", "getThermalEffectiveness", 1.0),
        ):
            try:
                value = float(getattr(unit, getter_name)()) * scale
            except Exception:
                continue
            if math.isfinite(value) and abs(value) < 1.0e300:
                if property_name == "heatTransferDuty_kW":
                    value = abs(value)
                properties[property_name] = value

        hot_side_transfer_kW = (
            named_sides["hot"]["inlet"]["enthalpy_flow_kW"]
            - named_sides["hot"]["outlet"]["enthalpy_flow_kW"]
        )
        cold_side_transfer_kW = (
            named_sides["cold"]["outlet"]["enthalpy_flow_kW"]
            - named_sides["cold"]["inlet"]["enthalpy_flow_kW"]
        )
        hot_side_duty_kW = abs(hot_side_transfer_kW)
        cold_side_duty_kW = abs(cold_side_transfer_kW)
        duty_closure_kW = (
            hot_side_transfer_kW - cold_side_transfer_kW
        )
        duty_closure_pct = (
            abs(duty_closure_kW)
            / max(
                hot_side_duty_kW,
                cold_side_duty_kW,
                _UNIT_BALANCE_SCALE_FLOOR_KW,
            )
            * 100.0
        )
        properties["hotSideDuty_kW"] = hot_side_duty_kW
        properties["coldSideDuty_kW"] = cold_side_duty_kW
        properties["dutyClosure_kW"] = duty_closure_kW
        properties["dutyClosure_pct"] = duty_closure_pct
        return properties

    @staticmethod
    def _native_scalar_field_signature(
        native_object: Any,
    ) -> Optional[Tuple[Any, ...]]:
        """Capture deterministic primitive/string/enum native fields."""
        if native_object is None:
            return None
        try:
            object_class = native_object.getClass()
            class_name = str(object_class.getName())
        except Exception:
            return None
        field_values: List[Tuple[str, Any]] = []
        declaring_class = object_class
        while declaring_class is not None:
            try:
                declared_fields = list(
                    declaring_class.getDeclaredFields()
                )
            except Exception:
                break
            for field in declared_fields:
                try:
                    if int(field.getModifiers()) & 8:
                        continue
                    field.setAccessible(True)
                    field_name = str(field.getName())
                    field_type = field.getType()
                    type_name = str(field_type.getName())
                    if type_name == "boolean":
                        value = bool(
                            field.getBoolean(native_object)
                        )
                    elif type_name in (
                        "byte",
                        "short",
                        "int",
                        "long",
                    ):
                        value = int(field.get(native_object))
                    elif type_name in ("float", "double"):
                        value = float(field.get(native_object))
                        if not math.isfinite(value):
                            return None
                    elif type_name == "java.lang.String" or bool(
                        field_type.isEnum()
                    ):
                        raw_value = field.get(native_object)
                        value = (
                            None
                            if raw_value is None
                            else str(raw_value).strip().casefold()
                        )
                    else:
                        continue
                except Exception:
                    continue
                field_values.append((field_name, value))
            try:
                declaring_class = declaring_class.getSuperclass()
            except Exception:
                break
        return (
            class_name,
            tuple(sorted(field_values)),
        )

    @staticmethod
    def _heat_exchanger_boundary_state_signature(
        unit: Any,
    ) -> Optional[Tuple[Any, ...]]:
        """Return a deterministic native boundary-state signature."""
        try:
            calculation_identifier = unit.getCalculationIdentifier()
        except Exception:
            return None
        if calculation_identifier is None:
            return None
        stream_states: List[Tuple[Any, ...]] = []
        for index in (0, 1):
            for getter_name in ("getInStream", "getOutStream"):
                try:
                    stream = getattr(unit, getter_name)(index)
                    stream_identifier = stream.getCalculationIdentifier()
                    if stream_identifier is None:
                        return None
                    fluid = stream.getFluid()
                    fluid.init(3)
                    state = (
                        str(stream_identifier),
                        float(stream.getTemperature("C")),
                        float(stream.getPressure("bara")),
                        float(stream.getFlowRate("kg/hr")),
                        float(fluid.getEnthalpy()) / 1000.0,
                    )
                except Exception:
                    return None
                if not all(
                    math.isfinite(value)
                    for value in state[1:]
                ):
                    return None
                stream_states.append(state)
        try:
            flow_arrangement = str(
                unit.getFlowArrangement()
            ).strip().casefold()
        except Exception:
            flow_arrangement = ""
        solution_settings: List[Tuple[str, Any]] = []
        for getter_name in (
            "isActive",
            "isLockedInactive",
            "getUAvalue",
            "getThermalEffectiveness",
            "getDeltaT",
            "getDuty",
            "getEnergyInput",
            "getOutletTemperature",
            "getApproachTemperature",
            "getMinApproachTemperature",
            "getHotColdDutyBalance",
            "getDesignDuty",
            "getDesignUAValue",
            "getMaxDesignDuty",
            "getDesignMode",
            "getRatingArea",
            "getRatingU",
            "getShellPasses",
            "getMinOutletTemperature",
            "getMaxOutletTemperature",
            "hasMinOutletTemperatureLimit",
            "hasMaxOutletTemperatureLimit",
        ):
            try:
                raw_value = getattr(unit, getter_name)()
            except Exception:
                value = None
            else:
                if isinstance(raw_value, bool):
                    value = raw_value
                else:
                    try:
                        numeric_value = float(raw_value)
                    except (TypeError, ValueError):
                        value = str(raw_value).strip().casefold()
                    else:
                        if not math.isfinite(numeric_value):
                            return None
                        value = numeric_value
            solution_settings.append((getter_name, value))
        use_delta_T: Optional[bool] = None
        for getter_name in ("isUseDeltaT", "getUseDeltaT"):
            try:
                use_delta_T = bool(getattr(unit, getter_name)())
                break
            except Exception:
                continue
        if use_delta_T is None:
            try:
                declaring_class = unit.getClass()
                while declaring_class is not None:
                    try:
                        field = declaring_class.getDeclaredField(
                            "useDeltaT"
                        )
                        field.setAccessible(True)
                        use_delta_T = bool(field.getBoolean(unit))
                        break
                    except Exception:
                        declaring_class = (
                            declaring_class.getSuperclass()
                        )
            except Exception:
                pass
        solution_settings.append(("useDeltaT", use_delta_T))
        try:
            rating_calculator = unit.getRatingCalculator()
        except Exception:
            rating_calculator = None
        solution_settings.append(
            (
                "ratingCalculator",
                NeqSimProcessModel._native_scalar_field_signature(
                    rating_calculator
                ),
            )
        )
        configuration = (
            flow_arrangement,
            tuple(solution_settings),
        )
        return (
            str(calculation_identifier),
            tuple(stream_states),
            configuration,
        )

    @staticmethod
    def _splitter_operating_properties(unit: Any) -> Dict[str, float]:
        """Return solved splitter allocation and flow-closure properties.

        Native ``Splitter`` exposes its exact outlet count and configured
        factors. Solved outlet flow is used for the reported allocation so
        fixed-flow and remainder specifications are represented correctly.
        Legacy objects without a readable count retain bounded branch-flow
        probing, but completeness-dependent totals and closure are withheld.
        """
        properties: Dict[str, float] = {}
        split_count = _native_split_stream_count(unit)
        topology_count_known = split_count is not None
        probe_count = (
            split_count
            if topology_count_known
            else _split_stream_probe_count(unit, 10)
        )

        try:
            inlet_flow_kg_hr = float(
                unit.getInletStream().getFlowRate("kg/hr")
            )
        except Exception:
            inlet_flow_kg_hr = math.nan
        if math.isfinite(inlet_flow_kg_hr):
            properties["inletFlow_kg_hr"] = inlet_flow_kg_hr

        outlet_flow_total_kg_hr = 0.0
        solved_outlet_flows_kg_hr: Dict[int, float] = {}
        solved_outlet_count = 0
        for index in range(probe_count):
            try:
                configured_fraction = float(unit.getSplitFactor(index))
            except Exception:
                configured_fraction = math.nan
            if math.isfinite(configured_fraction):
                properties[
                    f"configuredBranch{index}Fraction"
                ] = configured_fraction
            try:
                split_stream = unit.getSplitStream(index)
                outlet_flow_kg_hr = float(
                    split_stream.getFlowRate("kg/hr")
                )
            except Exception:
                continue
            if not math.isfinite(outlet_flow_kg_hr):
                continue
            properties[f"branch{index}Flow_kg_hr"] = outlet_flow_kg_hr
            solved_outlet_flows_kg_hr[index] = outlet_flow_kg_hr
            outlet_flow_total_kg_hr += outlet_flow_kg_hr
            solved_outlet_count += 1

        if topology_count_known:
            properties["branchCount"] = float(split_count)
        properties["solvedBranchCount"] = float(solved_outlet_count)
        if (
            topology_count_known
            and split_count is not None
            and solved_outlet_count == split_count
        ):
            properties["outletFlowTotal_kg_hr"] = outlet_flow_total_kg_hr
        if (
            topology_count_known
            and split_count is not None
            and solved_outlet_count == split_count
            and math.isfinite(inlet_flow_kg_hr)
        ):
            flow_closure_kg_hr = (
                outlet_flow_total_kg_hr - inlet_flow_kg_hr
            )
            properties["flowClosure_kg_hr"] = flow_closure_kg_hr
            properties["flowClosure_pct"] = (
                abs(flow_closure_kg_hr)
                / max(
                    abs(inlet_flow_kg_hr),
                    abs(outlet_flow_total_kg_hr),
                    _UNIT_BALANCE_SCALE_FLOOR_KG_HR,
                )
                * 100.0
            )
            if abs(inlet_flow_kg_hr) > _UNIT_BALANCE_SCALE_FLOOR_KG_HR:
                solved_fraction_sum = 0.0
                for index, outlet_flow_kg_hr in solved_outlet_flows_kg_hr.items():
                    solved_fraction = outlet_flow_kg_hr / inlet_flow_kg_hr
                    properties[f"branch{index}Fraction"] = solved_fraction
                    solved_fraction_sum += solved_fraction
                properties["splitFractionSum"] = solved_fraction_sum
        return properties

    @staticmethod
    def _mixer_operating_properties(unit: Any) -> Dict[str, float]:
        """Return solved mixer inlet allocations and mass-flow closure."""
        properties: Dict[str, float] = {}
        try:
            inlet_count = int(unit.getNumberOfInputStreams())
        except Exception:
            return properties
        if (
            inlet_count < 0
            or inlet_count > _MAX_NATIVE_SPLIT_STREAM_COUNT
        ):
            return properties

        properties["inletCount"] = float(inlet_count)
        inlet_flow_total_kg_hr = 0.0
        solved_inlet_count = 0
        inlet_flows: Dict[int, float] = {}
        for index in range(inlet_count):
            try:
                inlet_flow_kg_hr = float(
                    unit.getStream(index).getFlowRate("kg/hr")
                )
            except Exception:
                continue
            if not math.isfinite(inlet_flow_kg_hr):
                continue
            properties[f"inlet{index}Flow_kg_hr"] = inlet_flow_kg_hr
            inlet_flows[index] = inlet_flow_kg_hr
            inlet_flow_total_kg_hr += inlet_flow_kg_hr
            solved_inlet_count += 1

        properties["solvedInletCount"] = float(solved_inlet_count)
        if solved_inlet_count == inlet_count:
            properties["inletFlowTotal_kg_hr"] = inlet_flow_total_kg_hr
            if (
                abs(inlet_flow_total_kg_hr)
                > _UNIT_BALANCE_SCALE_FLOOR_KG_HR
            ):
                for index, inlet_flow_kg_hr in inlet_flows.items():
                    properties[f"inlet{index}Fraction"] = (
                        inlet_flow_kg_hr / inlet_flow_total_kg_hr
                    )

        try:
            outlet_flow_kg_hr = float(
                unit.getOutletStream().getFlowRate("kg/hr")
            )
        except Exception:
            outlet_flow_kg_hr = math.nan
        if math.isfinite(outlet_flow_kg_hr):
            properties["outletFlow_kg_hr"] = outlet_flow_kg_hr
        if (
            math.isfinite(outlet_flow_kg_hr)
            and solved_inlet_count == inlet_count
        ):
            flow_closure_kg_hr = (
                outlet_flow_kg_hr - inlet_flow_total_kg_hr
            )
            properties["flowClosure_kg_hr"] = flow_closure_kg_hr
            properties["flowClosure_pct"] = (
                abs(flow_closure_kg_hr)
                / max(
                    abs(inlet_flow_total_kg_hr),
                    abs(outlet_flow_kg_hr),
                    _UNIT_BALANCE_SCALE_FLOOR_KG_HR,
                )
                * 100.0
            )
        return properties

    @staticmethod
    def _separator_design_properties(unit: Any) -> Dict[str, Any]:
        """Return explicit native sizing results only after opt-in auto-size."""
        try:
            if not bool(unit.isAutoSized()):
                return {}
        except Exception:
            return {}

        properties: Dict[str, Any] = {"designAutoSized": True}
        for key, getter in (
            ("designGasLoadFactor_m_per_s", "getDesignGasLoadFactor"),
            ("designLiquidLevelFraction", "getDesignLiquidLevelFraction"),
            ("designInternalDiameter_m", "getInternalDiameter"),
            ("designSeparatorLength_m", "getSeparatorLength"),
        ):
            if not hasattr(unit, getter):
                continue
            try:
                value = float(getattr(unit, getter)())
            except Exception:
                continue
            if math.isfinite(value) and value > 0.0:
                properties[key] = value

        try:
            mechanical_design = unit.getMechanicalDesign()
        except Exception:
            mechanical_design = None
        if mechanical_design is not None:
            for key, getter in (
                ("designRetentionTime_s", "getRetentionTime"),
                ("designVolume_m3", "getVolumeTotal"),
            ):
                if not hasattr(mechanical_design, getter):
                    continue
                try:
                    value = float(getattr(mechanical_design, getter)())
                except Exception:
                    continue
                if math.isfinite(value) and value > 0.0:
                    properties[key] = value
        return properties

    @staticmethod
    def _separator_design_property_unit(property_name: str) -> str:
        """Return the explicit engineering unit for separator design data."""
        if property_name.endswith("_m_per_s"):
            return "m/s"
        if property_name.endswith("_m3"):
            return "m3"
        if property_name.endswith("_m"):
            return "m"
        if property_name.endswith("_s"):
            return "s"
        if property_name == "designAutoSized":
            return "boolean"
        return "[-]"

    @staticmethod
    def _routing_property_unit(property_name: str) -> str:
        """Return the explicit engineering unit for routing properties."""
        if property_name.endswith("_kg_hr"):
            return "kg/hr"
        if property_name.endswith("_pct"):
            return "%"
        return "[-]"

    @staticmethod
    def _heat_exchanger_property_unit(property_name: str) -> str:
        """Return the explicit engineering unit for exchanger properties."""
        if property_name.endswith("Temperature_C"):
            return "°C"
        if property_name.endswith("Temperature_K"):
            return "K"
        if property_name.endswith("Pressure_bara"):
            return "bara"
        if property_name.endswith("Flow_kg_hr"):
            return "kg/hr"
        if property_name.endswith("Duty_kW") or property_name.endswith(
            "Closure_kW"
        ):
            return "kW"
        if property_name.endswith("Closure_pct"):
            return "%"
        if property_name == "UA_W_K":
            return "W/K"
        return "[-]"

    @staticmethod
    def _is_inactive_heat_exchanger(unit: Any, java_class: str) -> bool:
        """Return whether a native exchanger must suppress solved outputs."""
        if java_class != "HeatExchanger":
            return False
        try:
            if bool(unit.isLockedInactive()):
                return True
        except Exception:
            pass
        try:
            if not bool(unit.isActive()):
                return True
        except Exception:
            pass
        return False

    def _heat_exchanger_solution_is_trusted(
        self,
        unit_name: str,
        unit: Any,
        java_class: str,
    ) -> bool:
        """Return whether current exchanger outputs match an observed solve."""
        if java_class != "HeatExchanger" or self._is_inactive_heat_exchanger(
            unit,
            java_class,
        ):
            return False
        current_snapshot = self._heat_exchanger_boundary_state_signature(unit)
        if current_snapshot is None:
            return False
        snapshots = getattr(
            self,
            "_heat_exchanger_state_snapshots",
            {},
        )
        if unit_name in snapshots:
            return snapshots[unit_name] == current_snapshot
        unit_name_casefold = unit_name.casefold()
        matching_names = {
            snapshot_name
            for snapshot_name in snapshots
            if snapshot_name.casefold() == unit_name_casefold
            or snapshot_name.casefold().endswith(
                f"/{unit_name_casefold}"
            )
        }
        if len(matching_names) != 1:
            return False
        matching_name = next(iter(matching_names))
        return snapshots[matching_name] == current_snapshot

    def _indexed_unit_name_for_native(
        self,
        unit: Any,
        preferred_name: str,
    ) -> str:
        """Resolve one indexed unit name by exact native object identity."""
        identity = _NativeObjectIdentitySet()
        identity.add(unit)
        matching_names = [
            indexed_name
            for indexed_name, indexed_unit in self._units.items()
            if identity.contains(indexed_unit)
        ]
        if preferred_name in matching_names:
            return preferred_name
        if len(matching_names) == 1:
            return matching_names[0]
        return preferred_name

    def _indexed_unit_name_for_process_system(
        self,
        process_system: Any,
        report_name: str,
        process_system_name: str,
    ) -> Optional[str]:
        """Resolve a module report entry through its native unit identity.

        ``None`` identifies a current native unit that is not indexed by this
        wrapper.  Callers must treat that identity as untrusted rather than
        transferring solved provenance from a removed, same-named unit.
        """
        try:
            units = list(process_system.getUnitOperations())
        except Exception:
            units = []
        for unit in units:
            try:
                raw_name = str(unit.getName())
            except Exception:
                continue
            if report_name not in (
                raw_name,
                f"{process_system_name}/{raw_name}",
            ):
                continue
            indexed_name = self._indexed_unit_name_for_native(
                unit,
                f"{process_system_name}/{raw_name}",
            )
            identity = _NativeObjectIdentitySet()
            identity.add(unit)
            if not any(
                identity.contains(indexed_unit)
                for indexed_unit in self._units.values()
            ):
                return None
            return indexed_name
        return (
            f"{process_system_name}/{report_name}"
            if process_system_name
            else report_name
        )

    def list_units(self) -> List[UnitInfo]:
        """List all unit operations with type info and key properties."""
        result = []
        report_duty_lookup = self._report_unit_duty_lookup()
        for name, u in self._units.items():
            try:
                java_class = str(u.getClass().getSimpleName())
            except Exception:
                java_class = "Unknown"

            ps_name = self._unit_ps_name.get(name, "")

            props = {}
            exchanger_solution_is_trusted = (
                java_class != "HeatExchanger"
                or self._report_unit_duty_suppression(
                    name,
                    report_duty_lookup,
                )
                is False
            )
            # Try to extract common properties
            for prop, getter in [
                ("power_kW", "getPower"),
                ("duty_kW", "getDuty"),
                ("isentropicEfficiency", "getIsentropicEfficiency"),
                ("polytropicEfficiency", "getPolytropicEfficiency"),
                ("outletPressure_bara", "getOutletPressure"),
            ]:
                if (
                    prop == "duty_kW"
                    and java_class == "HeatExchanger"
                    and not exchanger_solution_is_trusted
                ):
                    continue
                if hasattr(u, getter):
                    try:
                        val = getattr(u, getter)()
                        if val is None:
                            continue
                        fval = float(val)
                        if prop in ("power_kW", "duty_kW"):
                            fval = fval / 1000.0  # W -> kW
                        if (
                            prop == "isentropicEfficiency"
                            and java_class == "ESPPump"
                        ):
                            fval /= 100.0
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

            if java_class in ("Pump", "ESPPump"):
                props.update(self._pump_operating_properties(u))
                props.update(self._pump_design_properties(name, u))

            if java_class == "Compressor":
                props.update(self._compressor_map_properties(u))

            if (
                java_class == "HeatExchanger"
                and exchanger_solution_is_trusted
            ):
                props.update(
                    self._heat_exchanger_operating_properties(
                        u,
                        getattr(
                            self,
                            "_direct_unit_run_provenance",
                            {},
                        ).get(name),
                    )
                )

            if "Splitter" in java_class:
                props.update(self._splitter_operating_properties(u))

            if _is_native_mixer_class(java_class):
                props.update(self._mixer_operating_properties(u))

            if "Separator" in java_class or "Scrubber" in java_class:
                props.update(self._separator_design_properties(u))

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
        """List exact streams in topology order with stable display aliases."""
        result = []
        stream_groups = []
        for name, s in self._streams.items():
            for identity, _grouped_stream, aliases in stream_groups:
                if identity.contains(s):
                    aliases.append(name)
                    break
            else:
                identity = _NativeObjectIdentitySet()
                identity.add(s)
                stream_groups.append((identity, s, [name]))

        for _identity, s, aliases in stream_groups:
            name = min(
                aliases,
                key=lambda alias: (
                    str(alias).count("."),
                    len(str(alias)),
                    str(alias).casefold(),
                    str(alias),
                ),
            )
            ps_name = self._stream_ps_name.get(name, "")
            owner_name = next(
                (
                    str(alias).split(".", 1)[0]
                    for alias in aliases
                    if "." in str(alias)
                ),
                "",
            )
            info = StreamInfo(
                name=name,
                process_system=ps_name,
                owner_name=owner_name,
            )
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

    def record_direct_unit_run(self, unit_name: str) -> None:
        """Record explicit provenance for a completed direct exchanger run.

        NeqSim 3.16 assigns separate UUIDs to exchanger inlets during
        ``HeatExchanger.run(UUID)``. Call this immediately after that direct
        run so solved workbook and Process Chat properties can distinguish it
        from inlet streams recalculated after an older ProcessSystem solve.
        ProcessSystem runs do not need this marker because every boundary uses
        the process calculation UUID.
        """
        canonical_name = None
        name_lower = unit_name.lower()
        for candidate_name in self._units:
            if candidate_name == unit_name or (
                candidate_name.lower() == name_lower
            ) or candidate_name.lower().endswith(
                f"/{name_lower}"
            ):
                canonical_name = candidate_name
                break
        if canonical_name is None:
            raise KeyError(f"Unit not found: {unit_name}")
        unit = self._units[canonical_name]
        try:
            java_class = str(unit.getClass().getSimpleName())
        except Exception as exc:
            raise ValueError(
                f"Cannot inspect direct-run unit '{unit_name}'."
            ) from exc
        if java_class != "HeatExchanger":
            raise ValueError(
                "Direct-run provenance is currently supported only for "
                "native HeatExchanger units."
            )
        try:
            calculation_identifier = str(
                unit.getCalculationIdentifier()
            )
            inlet_identifiers = tuple(
                str(unit.getInStream(index).getCalculationIdentifier())
                for index in (0, 1)
            )
            outlet_identifiers = tuple(
                str(unit.getOutStream(index).getCalculationIdentifier())
                for index in (0, 1)
            )
        except Exception as exc:
            raise ValueError(
                f"Direct-run provenance for '{unit_name}' is incomplete."
            ) from exc
        if (
            calculation_identifier == "None"
            or "None" in inlet_identifiers
            or "None" in outlet_identifiers
        ):
            raise ValueError(
                f"Direct-run provenance for '{unit_name}' is incomplete."
            )
        if outlet_identifiers != (
            calculation_identifier,
            calculation_identifier,
        ):
            raise ValueError(
                f"Direct-run outlets for '{unit_name}' do not share the "
                "exchanger calculation identifier."
            )
        inlet_matches = tuple(
            identifier == calculation_identifier
            for identifier in inlet_identifiers
        )
        if any(inlet_matches):
            raise ValueError(
                f"'{unit_name}' does not have the direct-run identifier "
                "pattern; use the normal ProcessSystem result path."
            )
        self._direct_unit_run_provenance[canonical_name] = (
            calculation_identifier,
            inlet_identifiers,
            outlet_identifiers,
        )
        state_snapshot = self._heat_exchanger_boundary_state_signature(
            unit
        )
        if state_snapshot is None:
            raise ValueError(
                f"Direct-run state for '{unit_name}' is incomplete."
            )
        self._heat_exchanger_state_snapshots[
            canonical_name
        ] = state_snapshot

    def _capture_heat_exchanger_state_snapshots(
        self,
        allow_direct_runs: bool = False,
    ) -> None:
        """Capture solved exchanger boundaries owned by the wrapper run."""
        self._direct_unit_run_provenance.clear()
        self._heat_exchanger_state_snapshots.clear()
        for name, unit in self._units.items():
            try:
                if str(unit.getClass().getSimpleName()) != "HeatExchanger":
                    continue
                if self._is_inactive_heat_exchanger(unit, "HeatExchanger"):
                    continue
                calculation_identifier = unit.getCalculationIdentifier()
                if calculation_identifier is None:
                    continue
                inlet_identifiers = tuple(
                    unit.getInStream(index).getCalculationIdentifier()
                    for index in (0, 1)
                )
                outlet_identifiers = tuple(
                    unit.getOutStream(index).getCalculationIdentifier()
                    for index in (0, 1)
                )
            except Exception:
                continue
            if any(identifier is None for identifier in inlet_identifiers):
                continue
            if any(identifier is None for identifier in outlet_identifiers):
                continue
            calculation_identifier_str = str(calculation_identifier)
            inlet_identifier_strings = tuple(
                str(identifier) for identifier in inlet_identifiers
            )
            outlet_identifier_strings = tuple(
                str(identifier) for identifier in outlet_identifiers
            )
            if outlet_identifier_strings != (
                calculation_identifier_str,
                calculation_identifier_str,
            ):
                continue
            inlet_matches = tuple(
                identifier == calculation_identifier_str
                for identifier in inlet_identifier_strings
            )
            if not all(inlet_matches):
                if not allow_direct_runs or any(inlet_matches):
                    continue
                self._direct_unit_run_provenance[name] = (
                    calculation_identifier_str,
                    inlet_identifier_strings,
                    outlet_identifier_strings,
                )
            snapshot = self._heat_exchanger_boundary_state_signature(unit)
            if snapshot is not None:
                self._heat_exchanger_state_snapshots[name] = snapshot

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
        self._direct_unit_run_provenance.clear()
        self._heat_exchanger_state_snapshots.clear()
        direct_closure_ran = False
        if self._is_process_model:
            # ProcessModel has its own run() that iterates all children
            process_run_succeeded = self._run_process_model(
                self._proc,
                timeout_ms=timeout_ms,
            )
        else:
            process_run_succeeded = self._run_until_converged(
                self._proc,
                max_runs=5,
                timeout_ms=timeout_ms,
            )
            if (
                process_run_succeeded
                and self._enforce_acyclic_mixer_energy
            ):
                direct_closure_ran = (
                    self._run_acyclic_mixer_energy_closure(self._proc)
                )

        # Re-index model objects after running so references are fresh
        self._index_model_objects()
        if process_run_succeeded:
            self._capture_heat_exchanger_state_snapshots(
                allow_direct_runs=direct_closure_ran
            )

        return self._extract_results()

    def rerun(self, timeout_ms: int = 120000):
        """Re-run the process without extracting results.

        Convenience method for callers that just need to re-execute the
        simulation (e.g. after modifying parameters) and then re-index.
        Handles both ProcessSystem and ProcessModel transparently.
        """
        self._direct_unit_run_provenance.clear()
        self._heat_exchanger_state_snapshots.clear()
        direct_closure_ran = False
        if self._is_process_model:
            process_run_succeeded = self._run_process_model(
                self._proc,
                timeout_ms=timeout_ms,
            )
        else:
            process_run_succeeded = self._run_until_converged(
                self._proc,
                max_runs=5,
                timeout_ms=timeout_ms,
            )
            if (
                process_run_succeeded
                and self._enforce_acyclic_mixer_energy
            ):
                direct_closure_ran = (
                    self._run_acyclic_mixer_energy_closure(self._proc)
                )
        self._index_model_objects()
        if process_run_succeeded:
            self._capture_heat_exchanger_state_snapshots(
                allow_direct_runs=direct_closure_ran
            )

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
                    return False
                if not NeqSimProcessModel._async_run_status_succeeded(
                    proc_model
                ):
                    return False
            else:
                proc_model.run()
            return True
        except Exception:
            # Fallback: run each ProcessSystem individually
            try:
                process_systems = list(proc_model.getAllProcesses())
            except Exception:
                return False
            if not process_systems:
                return False
            return all(
                NeqSimProcessModel._run_until_converged(ps)
                for ps in process_systems
            )

    @staticmethod
    def _optional_nonnegative_number(unit: Any, getter: str) -> Optional[float]:
        """Read one finite non-negative native diagnostic value."""
        if not hasattr(unit, getter):
            return None
        try:
            value = abs(float(getattr(unit, getter)()))
        except Exception:
            return None
        return value if math.isfinite(value) else None

    def _extract_convergence_diagnostics(self) -> Dict[str, Any]:
        """Capture native recycle and adjuster convergence diagnostics."""
        rows: List[Dict[str, Any]] = []
        suggestions: List[str] = []

        for process_index, process_system in enumerate(
            self.get_process_systems()
        ):
            try:
                process_name = str(process_system.getName()).strip()
            except Exception:
                process_name = ""
            if not process_name or process_name.lower() == "null":
                process_name = f"process {process_index + 1}"

            try:
                units = list(process_system.getUnitOperations())
            except Exception:
                units = []

            for unit in units:
                try:
                    java_class = str(
                        unit.getClass().getSimpleName()
                    ).strip()
                except Exception:
                    continue
                if java_class not in {"Recycle", "Adjuster"}:
                    continue
                try:
                    unit_name = str(unit.getName()).strip()
                except Exception:
                    unit_name = ""
                if not unit_name:
                    unit_name = java_class

                try:
                    converged = bool(unit.solved())
                except Exception:
                    converged = False

                iterations = self._optional_nonnegative_number(
                    unit,
                    "getIterations",
                )
                max_iterations = self._optional_nonnegative_number(
                    unit,
                    "getMaxIterations",
                )
                row: Dict[str, Any] = {
                    "process_system": process_name,
                    "unit_name": unit_name,
                    "unit_type": java_class.lower(),
                    "converged": converged,
                    "iterations": (
                        int(iterations)
                        if iterations is not None
                        else None
                    ),
                    "max_iterations": (
                        int(max_iterations)
                        if max_iterations is not None
                        and max_iterations >= 1.0
                        else None
                    ),
                    "dominant_error": None,
                    "acceleration_method": None,
                    "flow_error": None,
                    "temperature_error": None,
                    "pressure_error": None,
                    "composition_error": None,
                    "error": None,
                    "flow_tolerance": None,
                    "temperature_tolerance": None,
                    "pressure_tolerance": None,
                    "composition_tolerance": None,
                    "tolerance": None,
                }
                if java_class == "Recycle":
                    getter_fields = {
                        "flow_error": "getErrorFlow",
                        "temperature_error": "getErrorTemperature",
                        "pressure_error": "getErrorPressure",
                        "composition_error": "getErrorComposition",
                        "flow_tolerance": "getFlowTolerance",
                        "temperature_tolerance": (
                            "getTemperatureTolerance"
                        ),
                        "pressure_tolerance": "getPressureTolerance",
                        "composition_tolerance": (
                            "getCompositionTolerance"
                        ),
                    }
                    for field_name, getter in getter_fields.items():
                        row[field_name] = (
                            self._optional_nonnegative_number(
                                unit,
                                getter,
                            )
                        )
                    error_ratios = []
                    for error_name in (
                        "flow",
                        "temperature",
                        "pressure",
                        "composition",
                    ):
                        error_value = row[f"{error_name}_error"]
                        tolerance = row[f"{error_name}_tolerance"]
                        if error_value is None:
                            continue
                        scale = (
                            tolerance
                            if tolerance is not None and tolerance > 0.0
                            else 1.0
                        )
                        error_ratios.append(
                            (error_value / scale, error_name)
                        )
                    if error_ratios:
                        row["dominant_error"] = max(error_ratios)[1]
                    if hasattr(unit, "getAccelerationMethod"):
                        try:
                            row["acceleration_method"] = str(
                                unit.getAccelerationMethod()
                            ).strip() or None
                        except Exception:
                            pass
                else:
                    row["error"] = self._optional_nonnegative_number(
                        unit,
                        "getError",
                    )
                    row["tolerance"] = (
                        self._optional_nonnegative_number(
                            unit,
                            "getTolerance",
                        )
                    )
                    row["dominant_error"] = "target"
                rows.append(row)

            try:
                from neqsim import jneqsim

                analyzer = (
                    jneqsim.process.equipment.util.ConvergenceDiagnostics(
                        process_system
                    )
                )
                native_report = analyzer.analyze()
                for suggestion in native_report.getSuggestions():
                    text = str(suggestion).strip()
                    if text and not text.startswith(
                        "No recycle or adjuster units found"
                    ):
                        suggestions.append(text)
            except Exception:
                pass

        unique_suggestions = list(dict.fromkeys(suggestions))
        if rows and not all(row["converged"] for row in rows):
            if not unique_suggestions:
                unique_suggestions.append(
                    "Review tear-stream estimates, tolerances, and "
                    "acceleration settings for unconverged units."
                )
        return {
            "applicable": bool(rows),
            "converged": (
                all(row["converged"] for row in rows) if rows else None
            ),
            "rows": rows,
            "suggestions": unique_suggestions,
        }

    def _extract_results(self) -> ModelRunResult:
        """Extract KPIs, constraints, and JSON report from solved process."""
        kpis: Dict[str, KPI] = {}
        constraints: List[ConstraintStatus] = []

        # Collect power and duty from all units
        total_power_kW = 0.0
        total_duty_kW = 0.0
        report_duty_lookup = self._report_unit_duty_lookup()

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
            exchanger_duty_is_trusted = (
                uclass != "HeatExchanger"
                or self._report_unit_duty_suppression(
                    name,
                    report_duty_lookup,
                )
                is False
            )
            if hasattr(u, "getDuty") and exchanger_duty_is_trusted:
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
                            native_ps_name = str(ps.getName())
                            ps_name = native_ps_name or "process"
                            report_obj = jneqsim.process.util.report.Report(ps)
                            r_str = str(report_obj.generateJsonReport())
                            r_data = json.loads(r_str)
                            if isinstance(r_data, dict):
                                r_data = self._filter_json_report_duties(
                                    r_data,
                                    report_duty_lookup,
                                    process_system_name=native_ps_name,
                                    process_system=ps,
                                )
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
            json_report = self._filter_json_report_duties(
                json_report,
                report_duty_lookup,
            )
            self._flatten_json_report(
                json_report,
                kpis,
                report_duty_lookup,
            )

        # Extract detailed unit operation properties (utilization, sizing, performance)
        self._extract_unit_properties(kpis, report_duty_lookup)

        for unit_name, design_basis in getattr(
            self,
            "_equipment_design_bases",
            {},
        ).items():
            is_pump_basis = set(design_basis) == set(
                _PUMP_DESIGN_CAPACITY_LIMITS
            )
            is_exchanger_basis = set(design_basis) == set(
                _HEAT_EXCHANGER_DESIGN_CAPACITY_LIMITS
            )
            constraint_prefix = (
                "pump_design"
                if is_pump_basis
                else "heat_exchanger_design"
                if is_exchanger_basis
                else "equipment_design"
            )
            unit = self._units.get(unit_name)
            if unit is None:
                constraints.append(
                    ConstraintStatus(
                        f"{constraint_prefix}.{unit_name}",
                        "UNKNOWN",
                        "Equipment design basis references a unit that is not "
                        "present in the solved process.",
                    )
                )
                continue
            try:
                java_class = str(unit.getClass().getSimpleName())
            except Exception:
                java_class = ""
            expected_classes = (
                ("Pump", "ESPPump")
                if is_pump_basis
                else ("HeatExchanger",)
                if is_exchanger_basis
                else ()
            )
            if java_class not in expected_classes:
                constraints.append(
                    ConstraintStatus(
                        f"{constraint_prefix}.{unit_name}",
                        "UNKNOWN",
                        "Equipment design basis references an incompatible "
                        "unit type.",
                    )
                )
                continue
            if is_pump_basis:
                constraints.append(
                    self._pump_design_constraint(unit_name, unit)
                )
            else:
                constraints.append(
                    self._heat_exchanger_design_constraint(unit_name, unit)
                )

        for unit_name, unit in self._units.items():
            try:
                java_class = str(unit.getClass().getSimpleName())
            except Exception:
                continue
            if java_class != "Compressor":
                continue
            map_properties = self._compressor_map_properties(unit)
            if not map_properties:
                continue
            required_states = (
                "mapWithinSpeedRange",
                "mapInSurge",
                "mapInStoneWall",
            )
            if any(
                state not in map_properties for state in required_states
            ):
                constraints.append(
                    ConstraintStatus(
                        f"compressor_map.{unit_name}",
                        "UNKNOWN",
                        "Native compressor map is active, but complete "
                        "speed/surge/stonewall state is unavailable.",
                    )
                )
                continue
            violations = []
            if not map_properties["mapWithinSpeedRange"]:
                violations.append("speed outside corrected-speed curves")
            if map_properties["mapInSurge"]:
                violations.append("operating point in surge region")
            if map_properties["mapInStoneWall"]:
                violations.append("operating point in stonewall region")
            status = "VIOLATION" if violations else "OK"
            speed = map_properties.get("mapOperatingSpeed_rpm")
            surge_distance = map_properties.get(
                "mapDistanceToSurgeFraction"
            )
            stonewall_distance = map_properties.get(
                "mapDistanceToStoneWallFraction"
            )
            operating_summary = (
                f"speed={speed:.6g} rpm, "
                f"surge distance={surge_distance:.6g}, "
                f"stonewall distance={stonewall_distance:.6g}"
                if all(
                    isinstance(value, (int, float))
                    and math.isfinite(float(value))
                    for value in (speed, surge_distance, stonewall_distance)
                )
                else "native map state available"
            )
            detail = (
                "; ".join(violations) + f"; {operating_summary}."
                if violations
                else f"Operating point is inside map limits; {operating_summary}."
            )
            constraints.append(
                ConstraintStatus(
                    f"compressor_map.{unit_name}",
                    status,
                    detail,
                )
            )

        # Extract mechanical design data (wall thickness, weights, dimensions, cost)
        self._extract_mechanical_design(kpis)

        convergence_diagnostics = self._extract_convergence_diagnostics()
        convergence_rows = convergence_diagnostics["rows"]
        if convergence_diagnostics["applicable"]:
            unconverged_rows = [
                row for row in convergence_rows if not row["converged"]
            ]
            iteration_values = [
                int(row["iterations"])
                for row in convergence_rows
                if row["iterations"] is not None
            ]
            kpis["convergence_unit_count"] = KPI(
                "convergence_unit_count",
                float(len(convergence_rows)),
                "count",
            )
            kpis["convergence_unconverged_count"] = KPI(
                "convergence_unconverged_count",
                float(len(unconverged_rows)),
                "count",
            )
            if iteration_values:
                kpis["convergence_max_iterations"] = KPI(
                    "convergence_max_iterations",
                    float(max(iteration_values)),
                    "iterations",
                )
            if unconverged_rows:
                detail = ", ".join(
                    f"{row['unit_name']} ({row['dominant_error'] or 'state'})"
                    for row in unconverged_rows
                )
                constraints.append(
                    ConstraintStatus(
                        "convergence",
                        "VIOLATION",
                        "Native iterative-unit convergence failed: "
                        f"{detail}.",
                    )
                )
            else:
                max_iterations = (
                    max(iteration_values) if iteration_values else 0
                )
                constraints.append(
                    ConstraintStatus(
                        "convergence",
                        "OK",
                        f"{len(convergence_rows)} native iterative unit(s) "
                        f"converged; maximum iterations={max_iterations}.",
                    )
                )
        else:
            constraints.append(
                ConstraintStatus(
                    "convergence",
                    "OK",
                    "Feed-forward process has no recycle or adjuster "
                    "convergence loops.",
                )
            )

        unit_balance_diagnostics = (
            self._extract_unit_balance_diagnostics()
        )
        from .solver_diagnostics import aggregate_unit_balances

        unit_balance_summary = aggregate_unit_balances(
            ModelRunResult(
                kpis={},
                constraints=[],
                raw={
                    "unit_balance_diagnostics": (
                        unit_balance_diagnostics
                    )
                },
            )
        )
        if unit_balance_summary["applicable"]:
            unit_count = float(unit_balance_summary["unit_count"])
            maximum_mass_imbalance = float(
                unit_balance_summary["max_mass_imbalance_pct"]
            )
            kpis["unit_balance_count"] = KPI(
                "unit_balance_count",
                unit_count,
                "count",
            )
            kpis["unit_mass_balance_max_pct"] = KPI(
                "unit_mass_balance_max_pct",
                maximum_mass_imbalance,
                "%",
            )
            mass_status = (
                "OK"
                if maximum_mass_imbalance < _COMPONENT_BALANCE_OK_PCT
                else "WARN"
                if maximum_mass_imbalance < _COMPONENT_BALANCE_WARN_PCT
                else "VIOLATION"
            )
            constraints.append(
                ConstraintStatus(
                    "unit_mass_balance",
                    mass_status,
                    f"{int(unit_count)} explicit-port unit(s) checked; "
                    "maximum relative mass imbalance="
                    f"{maximum_mass_imbalance:.6g}%.",
                )
            )

            energy_unit_count = float(
                unit_balance_summary["energy_unit_count"]
            )
            maximum_energy_imbalance = unit_balance_summary[
                "max_energy_imbalance_pct"
            ]
            if maximum_energy_imbalance is not None:
                maximum_energy_imbalance = float(
                    maximum_energy_imbalance
                )
                kpis["unit_energy_balance_count"] = KPI(
                    "unit_energy_balance_count",
                    energy_unit_count,
                    "count",
                )
                kpis["unit_energy_balance_max_pct"] = KPI(
                    "unit_energy_balance_max_pct",
                    maximum_energy_imbalance,
                    "%",
                )
                energy_status = (
                    "OK"
                    if maximum_energy_imbalance < _ENERGY_BALANCE_OK_PCT
                    else "WARN"
                    if maximum_energy_imbalance
                    < _ENERGY_BALANCE_WARN_PCT
                    else "VIOLATION"
                )
                constraints.append(
                    ConstraintStatus(
                        "unit_energy_balance",
                        energy_status,
                        f"{int(energy_unit_count)} audited unit(s) checked; "
                        "maximum relative energy imbalance="
                        f"{maximum_energy_imbalance:.6g}%.",
                    )
                )
        if not unit_balance_summary["coverage_complete"]:
            constraints.append(
                ConstraintStatus(
                    "unit_balance_coverage",
                    "UNKNOWN",
                    "Per-unit closure is unavailable for: "
                    + ", ".join(
                        unit_balance_summary["excluded_units"]
                    )
                    + ".",
                )
            )

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
                constraints.append(
                    ConstraintStatus("execution_quality", "WARN", msg)
                )

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
        material_boundaries: List[Dict[str, Any]] = []
        component_balances: List[Dict[str, Any]] = []
        energy_transfers: List[Dict[str, Any]] = []
        material_balance_applicable: Optional[bool] = None
        component_balance_applicable: Optional[bool] = None
        energy_balance_applicable: Optional[bool] = None
        try:
            material_boundary_identities = _MaterialBoundaryIdentityTracker()

            def _record_material_boundary(
                stream: Any,
                role: str,
                fallback_name: str,
            ) -> Optional[Dict[str, Any]]:
                """Record one native boundary identity once per material role."""
                if material_boundary_identities.contains(role, stream):
                    return None
                record = self._material_boundary_record(
                    stream,
                    role,
                    fallback_name,
                )
                material_boundary_identities.add(role, stream)
                material_boundaries.append(record)
                return record

            unit_groups = self._process_unit_groups()
            all_units = [
                unit
                for process_units in unit_groups
                for unit in process_units
            ]
            connectivity_unsafe_units: List[str] = []
            for unit in all_units:
                try:
                    unit_class = str(
                        unit.getClass().getSimpleName()
                    )
                except Exception:
                    continue
                if (
                    unit_class.lower()
                    not in _MATERIAL_CONNECTIVITY_UNSAFE_UNIT_CLASSES
                ):
                    continue
                if self._material_inlet_streams(unit):
                    continue
                try:
                    unit_name = str(unit.getName()).strip()
                except Exception:
                    unit_name = ""
                connectivity_unsafe_units.append(
                    unit_name or unit_class
                )
            material_balance_applicable = not connectivity_unsafe_units
            feed_flow = 0.0
            feed_details = []
            product_flow = 0.0
            product_details = []  # for diagnostic output

            _utility_types = {"Recycle", "Adjuster", "Calculator", "SetPoint"}

            if all_units:
                connected_feeds, connected_products = (
                    self._connectivity_material_boundaries(all_units)
                )
                consumed_streams, consumed_fluids = (
                    self._material_consumption_trackers(all_units)
                )
                feed_streams = connected_feeds or [
                    stream
                    for process_units in unit_groups
                    for stream in self._leading_material_feed_streams(
                        process_units
                    )
                ]
                for stream in feed_streams:
                    try:
                        record = _record_material_boundary(
                            stream,
                            "feed",
                            "feed",
                        )
                        if record is None:
                            continue
                        flow = record["mass_flow_kg_hr"]
                        name = record["stream_name"]
                        feed_flow += flow
                        feed_details.append(f"{name}={flow:.0f}")
                    except Exception:
                        pass
                def _add_outlet_flow(
                    stream_obj: Any,
                    label: str,
                ) -> float:
                    """Add a distinct product stream flow."""
                    nonlocal product_flow
                    record = _record_material_boundary(
                        stream_obj,
                        "product",
                        label,
                    )
                    if record is None:
                        return 0.0
                    flow = record["mass_flow_kg_hr"]
                    sname = record["stream_name"]
                    if abs(flow) > _MATERIAL_BOUNDARY_ZERO_FLOW_KG_HR:
                        product_flow += flow
                        product_details.append(f"{sname}={flow:.0f}")
                    else:
                        product_details.append(f"{sname}=0 (no flow)")
                    return flow

                terminal_stream_units = []
                if not connectivity_unsafe_units:
                    terminal_stream_units = [
                        stream
                        for process_units in unit_groups
                        for stream in (
                            self._trailing_material_product_streams(
                                process_units
                            )
                        )
                    ]
                explicit_product_fluids = (
                    _MaterialBoundaryIdentityTracker()
                )
                for stream in terminal_stream_units:
                    fluid = self._material_fluid_reference(stream)
                    if (
                        consumed_streams.contains("feed", stream)
                        or (
                            fluid is not None
                            and consumed_fluids.contains("feed", fluid)
                        )
                    ):
                        continue
                    try:
                        _add_outlet_flow(stream, "product")
                    except Exception:
                        continue
                    if fluid is not None:
                        explicit_product_fluids.add("product", fluid)

                for stream, label in (
                    [] if connectivity_unsafe_units else connected_products
                ):
                    fluid = self._material_fluid_reference(stream)
                    if (
                        fluid is not None
                        and explicit_product_fluids.contains(
                            "product",
                            fluid,
                        )
                    ):
                        continue
                    try:
                        _add_outlet_flow(stream, label)
                    except Exception:
                        pass

                if (
                    not connectivity_unsafe_units
                    and not terminal_stream_units
                    and not connected_products
                ):
                    # Compatibility fallback for native units whose ports
                    # cannot be inspected through the supported interfaces.
                    for process_units in unit_groups:
                        last = None
                        for unit in reversed(process_units):
                            try:
                                unit_class = str(
                                    unit.getClass().getSimpleName()
                                )
                            except Exception:
                                continue
                            if unit_class not in _utility_types:
                                last = unit
                                break
                        if last is None:
                            continue
                        for stream, label in (
                            self._fallback_material_outlet_streams(last)
                        ):
                            try:
                                _add_outlet_flow(stream, label)
                            except Exception:
                                pass

            # Fallback: match by stream name keywords
            if feed_flow == 0.0:
                for name, s in self._streams.items():
                    lower = name.lower()
                    if any(
                        keyword in lower
                        for keyword in ("feed", "inlet", "well", "input")
                    ):
                        try:
                            record = _record_material_boundary(
                                s,
                                "feed",
                                name,
                            )
                        except ValueError:
                            continue
                        if record is None:
                            continue
                        flow = record["mass_flow_kg_hr"]
                        feed_flow += flow
                    elif (
                        not connectivity_unsafe_units
                        and any(
                            keyword in lower
                            for keyword in (
                                "export",
                                "product",
                                "outlet",
                                "output",
                                "fuel",
                            )
                        )
                    ):
                        try:
                            record = _record_material_boundary(
                                s,
                                "product",
                                name,
                            )
                        except ValueError:
                            continue
                        if record is None:
                            continue
                        flow = record["mass_flow_kg_hr"]
                        product_flow += flow

            feed_boundary_count = sum(
                1
                for boundary in material_boundaries
                if boundary["role"] == "feed"
            )
            product_boundary_count = sum(
                1
                for boundary in material_boundaries
                if boundary["role"] == "product"
            )
            if feed_boundary_count:
                kpis["material_feed_count"] = KPI(
                    "material_feed_count",
                    float(feed_boundary_count),
                    "count",
                )
                kpis["material_feed_flow_kg_hr"] = KPI(
                    "material_feed_flow_kg_hr",
                    feed_flow,
                    "kg/hr",
                )
            if product_boundary_count:
                kpis["material_product_count"] = KPI(
                    "material_product_count",
                    float(product_boundary_count),
                    "count",
                )
                kpis["material_product_flow_kg_hr"] = KPI(
                    "material_product_flow_kg_hr",
                    product_flow,
                    "kg/hr",
                )

            if feed_flow > 0 and material_balance_applicable:
                balance_pct = abs(feed_flow - product_flow) / feed_flow * 100
                kpis["mass_balance_pct"] = KPI("mass_balance_pct", balance_pct, "%")
                feed_detail_str = (
                    ", ".join(feed_details)
                    if feed_details
                    else f"{feed_flow:.0f}"
                )
                detail_str = ", ".join(product_details) if product_details else f"{product_flow:.0f}"
                status = "OK" if balance_pct < 1.0 else "WARN" if balance_pct < 5.0 else "VIOLATION"
                constraints.append(ConstraintStatus(
                    "mass_balance", status,
                    f"Feeds={feed_flow:.0f} kg/hr ({feed_detail_str}), "
                    f"Products={product_flow:.0f} kg/hr ({detail_str}), "
                    f"imbalance={balance_pct:.2f}%"
                ))
            elif feed_flow > 0 and connectivity_unsafe_units:
                constraints.append(
                    ConstraintStatus(
                        "mass_balance",
                        "UNKNOWN",
                        "System material closure is unavailable because "
                        "native inlet connectivity cannot be inspected for: "
                        f"{', '.join(connectivity_unsafe_units)}.",
                    )
                )

            excluded_units = self._component_balance_exclusion_names(
                all_units
            )
            component_balance_applicable = not excluded_units
            if excluded_units:
                constraints.append(
                    ConstraintStatus(
                        "component_balance",
                        "UNKNOWN",
                        "Species-level boundary closure is not applicable "
                        "to species-changing or unclassified equipment. "
                        f"Units: {', '.join(excluded_units)}.",
                    )
                )
            else:
                from .solver_diagnostics import component_balance_rows

                try:
                    component_balances = component_balance_rows(
                        ModelRunResult(
                            kpis={},
                            constraints=[],
                            raw={
                                "material_boundaries": material_boundaries,
                                "component_balance_applicable": True,
                            },
                        )
                    )
                except ValueError as exc:
                    component_balance_applicable = False
                    component_balances = []
                    constraints.append(
                        ConstraintStatus(
                            "component_balance",
                            "UNKNOWN",
                            "Component balance unavailable: "
                            f"{exc}",
                        )
                    )
            if component_balance_applicable and component_balances:
                worst_component = max(
                    component_balances,
                    key=lambda row: float(row["imbalance_pct"]),
                )
                maximum_imbalance = float(
                    worst_component["imbalance_pct"]
                )
                kpis["component_balance_count"] = KPI(
                    "component_balance_count",
                    float(len(component_balances)),
                    "count",
                )
                kpis["component_balance_max_pct"] = KPI(
                    "component_balance_max_pct",
                    maximum_imbalance,
                    "%",
                )
                component_status = (
                    "OK"
                    if maximum_imbalance < _COMPONENT_BALANCE_OK_PCT
                    else "WARN"
                    if maximum_imbalance < _COMPONENT_BALANCE_WARN_PCT
                    else "VIOLATION"
                )
                constraints.append(
                    ConstraintStatus(
                        "component_balance",
                        component_status,
                        "Maximum component imbalance="
                        f"{maximum_imbalance:.6g}% "
                        f"({worst_component['component']}); "
                        f"{len(component_balances)} components checked.",
                    )
                )

            energy_transfers, energy_excluded_units = (
                self._system_energy_transfers(all_units)
            )
            energy_balance_applicable = not energy_excluded_units
            if energy_excluded_units:
                constraints.append(
                    ConstraintStatus(
                        "energy_balance",
                        "UNKNOWN",
                        "System energy closure is unavailable for unaudited "
                        "or unreadable equipment: "
                        f"{', '.join(energy_excluded_units)}.",
                    )
                )
            else:
                from .solver_diagnostics import aggregate_energy_balance

                try:
                    energy_summary = aggregate_energy_balance(
                        ModelRunResult(
                            kpis={},
                            constraints=[],
                            raw={
                                "material_boundaries": material_boundaries,
                                "energy_transfers": energy_transfers,
                                "energy_balance_applicable": True,
                            },
                        )
                    )
                except ValueError as exc:
                    energy_balance_applicable = False
                    constraints.append(
                        ConstraintStatus(
                            "energy_balance",
                            "UNKNOWN",
                            f"Energy balance unavailable: {exc}",
                        )
                    )
                else:
                    for name, unit, summary_key in (
                        (
                            "material_feed_enthalpy_kW",
                            "kW",
                            "feed_enthalpy_kW",
                        ),
                        (
                            "material_product_enthalpy_kW",
                            "kW",
                            "product_enthalpy_kW",
                        ),
                        (
                            "external_energy_transfer_kW",
                            "kW",
                            "external_energy_transfer_kW",
                        ),
                        (
                            "energy_balance_residual_kW",
                            "kW",
                            "residual_kW",
                        ),
                        (
                            "energy_balance_pct",
                            "%",
                            "imbalance_pct",
                        ),
                    ):
                        kpis[name] = KPI(
                            name,
                            float(energy_summary[summary_key]),
                            unit,
                        )
                    energy_imbalance = float(
                        energy_summary["imbalance_pct"]
                    )
                    energy_status = (
                        "OK"
                        if energy_imbalance < _ENERGY_BALANCE_OK_PCT
                        else "WARN"
                        if energy_imbalance < _ENERGY_BALANCE_WARN_PCT
                        else "VIOLATION"
                    )
                    constraints.append(
                        ConstraintStatus(
                            "energy_balance",
                            energy_status,
                            "Products - feeds - external transfer="
                            f"{energy_summary['residual_kW']:+.6g} kW; "
                            f"relative imbalance={energy_imbalance:.6g}%. "
                            "Positive external transfer adds energy to "
                            "the material system.",
                        )
                    )
        except Exception:
            pass

        return ModelRunResult(
            kpis=kpis,
            constraints=constraints,
            json_report=json_report,
            raw={
                "unit_names": list(self._units.keys()),
                "stream_names": list(self._streams.keys()),
                "material_boundaries": material_boundaries,
                "material_balance_applicable": material_balance_applicable,
                "component_balances": component_balances,
                "component_balance_applicable": component_balance_applicable,
                "energy_transfers": energy_transfers,
                "energy_balance_applicable": energy_balance_applicable,
                "convergence_diagnostics": convergence_diagnostics,
                "unit_balance_diagnostics": unit_balance_diagnostics,
            }
        )

    def _extract_unit_properties(
        self,
        kpis: Dict[str, KPI],
        report_duty_lookup: Optional[
            Dict[str, List[Tuple[str, bool]]]
        ] = None,
    ):
        """
        Extract detailed equipment-level properties from each unit operation.

        Covers compressor performance, separator capacity, cooler/heater sizing,
        pump/valve characteristics, and general utilization metrics.
        """
        if report_duty_lookup is None:
            report_duty_lookup = self._report_unit_duty_lookup()
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
                for prop, val in self._compressor_map_properties(u).items():
                    kpis[f"{prefix}.{prop}"] = KPI(
                        f"{prefix}.{prop}",
                        val,
                        self._compressor_map_property_unit(prop),
                    )
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
                for prop, val in self._separator_design_properties(u).items():
                    numeric_value = 1.0 if val is True else float(val)
                    kpis[f"{prefix}.{prop}"] = KPI(
                        f"{prefix}.{prop}",
                        numeric_value,
                        self._separator_design_property_unit(prop),
                    )

            # ---------- Cooler / Heater / HeatExchanger ----------
            elif java_class in ("Cooler", "Heater", "HeatExchanger", "AirCooler", "WaterCooler"):
                exchanger_duty_is_trusted = (
                    java_class != "HeatExchanger"
                    or self._report_unit_duty_suppression(
                        name,
                        report_duty_lookup,
                    )
                    is False
                )
                for prop, getter, unit in [
                    ("pressureDrop_bar", "getPressureDrop", "bar"),
                    ("inletTemperature_K", "getInletTemperature", "K"),
                    ("outletTemperature_K", "getOutletTemperature", "K"),
                    ("inletPressure_bara", "getInletPressure", "bara"),
                    ("outletPressure_bara", "getOutletPressure", "bara"),
                    ("maxDesignDuty_W", "getMaxDesignDuty", "W"),
                    ("energyInput_W", "getEnergyInput", "W"),
                ]:
                    if (
                        getter == "getEnergyInput"
                        and not exchanger_duty_is_trusted
                    ):
                        continue
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
                if (
                    java_class == "HeatExchanger"
                    and exchanger_duty_is_trusted
                ):
                    for prop, val in (
                        self._heat_exchanger_operating_properties(
                            u,
                            getattr(
                                self,
                                "_direct_unit_run_provenance",
                                {},
                            ).get(name),
                        ).items()
                    ):
                        kpis[f"{prefix}.{prop}"] = KPI(
                            f"{prefix}.{prop}",
                            val,
                            self._heat_exchanger_property_unit(prop),
                        )
                if java_class == "HeatExchanger":
                    for prop, val in self._heat_exchanger_design_properties(
                        name,
                        u,
                    ).items():
                        kpis[f"{prefix}.{prop}"] = KPI(
                            f"{prefix}.{prop}",
                            val,
                            self._heat_exchanger_design_property_unit(prop),
                        )

            # ---------- Pipeline hydraulics ----------
            elif java_class in (
                "Pipeline",
                "AdiabaticPipe",
                "PipeBeggsAndBrills",
                "OnePhasePipeLine",
            ):
                for prop, getter, unit in [
                    ("length_m", "getLength", "m"),
                    ("diameter_m", "getDiameter", "m"),
                    ("roughness_m", "getPipeWallRoughness", "m"),
                    ("inletPressure_bara", "getInletPressure", "bara"),
                    ("outletPressure_bara", "getOutletPressure", "bara"),
                    ("pressureDrop_bar", "getPressureDrop", "bar"),
                    ("inletTemperature_K", "getInletTemperature", "K"),
                    ("outletTemperature_K", "getOutletTemperature", "K"),
                ]:
                    if hasattr(u, getter):
                        try:
                            val = float(getattr(u, getter)())
                            if not math.isfinite(val):
                                continue
                            kpis[f"{prefix}.{prop}"] = KPI(
                                f"{prefix}.{prop}", val, unit
                            )
                        except Exception:
                            pass

                if java_class == "PipeBeggsAndBrills":
                    hydraulic_values = []
                    try:
                        hydraulic_values.append(
                            (
                                "velocity_m_s",
                                float(u.getMixtureVelocity()),
                                "m/s",
                            )
                        )
                    except Exception:
                        pass
                    try:
                        reynolds_profile = list(
                            u.getMixtureReynoldsNumber()
                        )
                        if reynolds_profile:
                            hydraulic_values.append(
                                (
                                    "reynoldsNumber",
                                    float(reynolds_profile[-1]),
                                    "[-]",
                                )
                            )
                    except Exception:
                        pass
                else:
                    hydraulic_values = []
                    for prop, getter, unit in (
                        ("velocity_m_s", "getVelocity", "m/s"),
                        ("reynoldsNumber", "getReynoldsNumber", "[-]"),
                        ("frictionFactor", "getFrictionFactor", "[-]"),
                    ):
                        if not hasattr(u, getter):
                            continue
                        try:
                            hydraulic_values.append(
                                (prop, float(getattr(u, getter)()), unit)
                            )
                        except Exception:
                            pass
                for prop, val, unit in hydraulic_values:
                    if math.isfinite(val):
                        kpis[f"{prefix}.{prop}"] = KPI(
                            f"{prefix}.{prop}", val, unit
                        )

            # ---------- Expander ----------
            elif java_class == "Expander":
                for prop, getter, unit in [
                    ("inletPressure_bara", "getInletPressure", "bara"),
                    ("outletPressure_bara", "getOutletPressure", "bara"),
                    ("inletTemperature_K", "getInletTemperature", "K"),
                    ("outletTemperature_K", "getOutletTemperature", "K"),
                    (
                        "isentropicEfficiency",
                        "getIsentropicEfficiency",
                        "[-]",
                    ),
                ]:
                    if hasattr(u, getter):
                        try:
                            val = float(getattr(u, getter)())
                            kpis[f"{prefix}.{prop}"] = KPI(
                                f"{prefix}.{prop}", val, unit
                            )
                        except Exception:
                            pass
                if hasattr(u, "getPower"):
                    try:
                        # NeqSim reports recovered shaft work as negative
                        # process power. Present recovery as a positive KPI
                        # while retaining the signed .power_kW system KPI.
                        recovered_power_kW = -float(u.getPower()) / 1000.0
                        kpis[f"{prefix}.recoveredPower_kW"] = KPI(
                            f"{prefix}.recoveredPower_kW",
                            recovered_power_kW,
                            "kW",
                        )
                    except Exception:
                        pass

            # ---------- Pump / ESPPump ----------
            elif java_class in ("Pump", "ESPPump"):
                pump_property_units = {
                    "inletPressure_bara": "bara",
                    "outletPressure_bara": "bara",
                    "pressureRise_bar": "bar",
                    "inletTemperature_K": "K",
                    "outletTemperature_K": "K",
                    "efficiency": "[-]",
                    "speed_rpm": "rpm",
                    "shaftPower_kW": "kW",
                    "inletDensity_kg_m3": "kg/m3",
                    "inletVolumetricFlow_m3_s": "m3/s",
                    "head_m": "m",
                    "hydraulicPower_kW": "kW",
                }
                for prop, val in self._pump_operating_properties(u).items():
                    kpis[f"{prefix}.{prop}"] = KPI(
                        f"{prefix}.{prop}",
                        val,
                        pump_property_units[prop],
                    )
                for prop, val in self._pump_design_properties(name, u).items():
                    kpis[f"{prefix}.{prop}"] = KPI(
                        f"{prefix}.{prop}",
                        val,
                        self._pump_design_property_unit(prop),
                    )

            # ---------- Valve ----------
            elif "Valve" in java_class:
                for prop, getter, unit in [
                    ("outletPressure_bara", "getOutletPressure", "bara"),
                    ("pressureDrop_bar", "getPressureDrop", "bar"),
                    (
                        "percentValveOpening",
                        "getPercentValveOpening",
                        "%",
                    ),
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
                            f"{prefix}.Cv", float(u.getCv()), "US Cv"
                        )
                    except Exception:
                        pass

            # ---------- Splitter ----------
            elif "Splitter" in java_class:
                splitter_properties = self._splitter_operating_properties(u)
                for prop, val in splitter_properties.items():
                    kpis[f"{prefix}.{prop}"] = KPI(
                        f"{prefix}.{prop}",
                        val,
                        self._routing_property_unit(prop),
                    )
                    if prop.startswith("branch") and prop.endswith(
                        "Flow_kg_hr"
                    ):
                        branch_index = prop[
                            len("branch") : -len("Flow_kg_hr")
                        ]
                        if branch_index.isdigit():
                            legacy_prop = (
                                f"splitStream{branch_index}_flow_kg_hr"
                            )
                            kpis[f"{prefix}.{legacy_prop}"] = KPI(
                                f"{prefix}.{legacy_prop}",
                                val,
                                "kg/hr",
                            )

            # ---------- Mixer ----------
            elif _is_native_mixer_class(java_class):
                for prop, val in self._mixer_operating_properties(u).items():
                    kpis[f"{prefix}.{prop}"] = KPI(
                        f"{prefix}.{prop}",
                        val,
                        self._routing_property_unit(prop),
                    )

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

    def _flatten_json_report(
        self,
        json_report: dict,
        kpis: Dict[str, KPI],
        duty_lookup: Optional[
            Dict[str, List[Tuple[str, bool]]]
        ] = None,
    ):
        """
        Flatten the nested JSON report into queryable KPI entries.
        
        Produces keys like:
          "report.feed gas.properties.gas.density"
          "report.1st stage compressor.power"
          "report.inlet separator.gas.conditions.gas.temperature"
          "report.inlet separator.gas.composition.gas.methane"
        """
        process_system_names = {
            name
            for name in self._unit_ps_name.values()
            if name
        }
        current_process_systems = {}
        if self._is_process_model:
            for current_process_system in self.get_process_systems():
                try:
                    current_name = str(
                        current_process_system.getName()
                    )
                except Exception:
                    continue
                current_process_systems[current_name] = (
                    current_process_system
                )
        process_system_names.update(current_process_systems)
        if duty_lookup is None:
            duty_lookup = self._report_unit_duty_lookup()
        for report_name, report_data in json_report.items():
            if not isinstance(report_data, dict):
                continue
            prefix = f"report.{report_name}"
            is_process_system_container = (
                self._is_process_model
                and report_name in process_system_names
            )
            suppress_duty = (
                None
                if is_process_system_container
                else self._report_unit_duty_suppression(
                    report_name,
                    duty_lookup,
                )
            )
            if suppress_duty is not None:
                self._flatten_dict(
                    report_data,
                    prefix,
                    kpis,
                    suppress_duty=suppress_duty,
                )
                continue

            unmatched_data = {}
            matched_nested_unit = False
            for nested_name, nested_data in report_data.items():
                if isinstance(nested_data, dict):
                    current_process_system = (
                        current_process_systems.get(report_name)
                    )
                    lookup_name = (
                        self._indexed_unit_name_for_process_system(
                            current_process_system,
                            nested_name,
                            report_name,
                        )
                        if current_process_system is not None
                        else f"{report_name}/{nested_name}"
                    )
                    nested_suppression = (
                        True
                        if lookup_name is None
                        else self._report_unit_duty_suppression(
                            lookup_name,
                            duty_lookup,
                        )
                    )
                    if nested_suppression is not None:
                        self._flatten_dict(
                            nested_data,
                            f"{prefix}.{nested_name}",
                            kpis,
                            suppress_duty=nested_suppression,
                        )
                        matched_nested_unit = True
                        continue
                unmatched_data[nested_name] = nested_data
            self._flatten_dict(
                unmatched_data if matched_nested_unit else report_data,
                prefix,
                kpis,
            )

    def _report_unit_duty_lookup(
        self,
    ) -> Dict[str, List[Tuple[str, bool]]]:
        """Index current report identities and solved-duty trust once per pass."""
        lookup: Dict[str, List[Tuple[str, bool]]] = {}
        current_units: List[Tuple[str, Any]] = []
        current_identities = _NativeObjectIdentitySet()
        unit_process_system_names = getattr(
            self,
            "_unit_ps_name",
            {},
        )
        try:
            process_systems = self.get_process_systems()
        except Exception:
            process_systems = []
        native_inventory_available = bool(process_systems)
        for process_system in process_systems:
            try:
                process_system_name = str(process_system.getName())
            except Exception:
                process_system_name = ""
            try:
                native_units = list(process_system.getUnitOperations())
            except Exception:
                native_units = []
            for native_unit in native_units:
                current_units.append((process_system_name, native_unit))
                current_identities.add(native_unit)

        indexed_suppression: Dict[str, bool] = {}
        for indexed_name, unit in self._units.items():
            try:
                java_class = str(unit.getClass().getSimpleName())
            except Exception:
                continue
            try:
                raw_name = str(unit.getName())
            except Exception:
                raw_name = indexed_name.rsplit("/", 1)[-1]
            process_system_name = unit_process_system_names.get(
                indexed_name,
                "",
            )
            qualified_report_name = (
                f"{process_system_name}/{raw_name}"
                if process_system_name
                else raw_name
            )
            suppress_duty = (
                java_class == "HeatExchanger"
                and (
                    (
                        native_inventory_available
                        and not current_identities.contains(unit)
                    )
                    or not self._heat_exchanger_solution_is_trusted(
                        indexed_name,
                        unit,
                        java_class,
                    )
                )
            )
            indexed_suppression[indexed_name] = suppress_duty
            for report_name in {
                indexed_name,
                raw_name,
                qualified_report_name,
            }:
                lookup.setdefault(report_name, []).append(
                    (indexed_name, suppress_duty)
                )

        for process_system_name, unit in current_units:
            try:
                raw_name = str(unit.getName())
                java_class = str(unit.getClass().getSimpleName())
            except Exception:
                continue
            identity = _NativeObjectIdentitySet()
            identity.add(unit)
            matching_indexed_names = [
                indexed_name
                for indexed_name, indexed_unit in self._units.items()
                if identity.contains(indexed_unit)
            ]
            suppress_duty = (
                java_class == "HeatExchanger"
                and (
                    not matching_indexed_names
                    or any(
                        indexed_suppression.get(indexed_name, True)
                        for indexed_name in matching_indexed_names
                    )
                )
            )
            lookup_identity = (
                matching_indexed_names[0]
                if matching_indexed_names
                else f"{process_system_name}/{raw_name}#unindexed"
            )
            qualified_report_name = (
                f"{process_system_name}/{raw_name}"
                if process_system_name
                else raw_name
            )
            for report_name in {raw_name, qualified_report_name}:
                lookup.setdefault(report_name, []).append(
                    (lookup_identity, suppress_duty)
                )
        return lookup

    def _report_unit_duty_suppression(
        self,
        report_name: str,
        duty_lookup: Optional[
            Dict[str, List[Tuple[str, bool]]]
        ] = None,
    ) -> Optional[bool]:
        """Return an indexed report unit's duty suppression, if matched."""
        matches = (duty_lookup or self._report_unit_duty_lookup()).get(
            report_name
        )
        if not matches:
            return None
        return any(suppress_duty for _, suppress_duty in matches)

    def _copy_report_data(
        self,
        data: Any,
        suppress_duty: bool = False,
    ) -> Any:
        """Copy JSON report data while removing untrusted duty-derived fields."""
        if isinstance(data, dict):
            return {
                key: self._copy_report_data(
                    value,
                    suppress_duty=suppress_duty,
                )
                for key, value in data.items()
                if not (
                    suppress_duty
                    and str(key).strip().casefold().startswith("duty")
                )
            }
        if isinstance(data, list):
            return [
                self._copy_report_data(
                    value,
                    suppress_duty=suppress_duty,
                )
                for value in data
            ]
        return data

    def _filter_json_report_duties(
        self,
        json_report: dict,
        duty_lookup: Optional[
            Dict[str, List[Tuple[str, bool]]]
        ] = None,
        process_system_name: str = "",
        process_system: Any = None,
    ) -> dict:
        """Return a public JSON report without untrusted exchanger duties."""
        process_system_names = {
            name
            for name in self._unit_ps_name.values()
            if name
        }
        current_process_systems = {}
        if self._is_process_model:
            for current_process_system in self.get_process_systems():
                try:
                    current_name = str(
                        current_process_system.getName()
                    )
                except Exception:
                    continue
                current_process_systems[current_name] = (
                    current_process_system
                )
        process_system_names.update(current_process_systems)
        if duty_lookup is None:
            duty_lookup = self._report_unit_duty_lookup()
        filtered = {}
        for report_name, report_data in json_report.items():
            is_process_system_container = (
                self._is_process_model
                and report_name in process_system_names
                and isinstance(report_data, dict)
            )
            if is_process_system_container:
                nested_report = {}
                current_process_system = current_process_systems.get(
                    report_name
                )
                for nested_name, nested_data in report_data.items():
                    lookup_name = (
                        self._indexed_unit_name_for_process_system(
                            current_process_system,
                            nested_name,
                            report_name,
                        )
                        if current_process_system is not None
                        else f"{report_name}/{nested_name}"
                    )
                    suppression = (
                        True
                        if lookup_name is None
                        else self._report_unit_duty_suppression(
                            lookup_name,
                            duty_lookup,
                        )
                    )
                    nested_report[nested_name] = self._copy_report_data(
                        nested_data,
                        suppress_duty=bool(suppression),
                    )
                filtered[report_name] = nested_report
                continue
            if process_system is not None:
                lookup_report_name = (
                    self._indexed_unit_name_for_process_system(
                        process_system,
                        report_name,
                        process_system_name,
                    )
                )
            else:
                lookup_report_name = (
                    f"{process_system_name}/{report_name}"
                    if process_system_name
                    and not report_name.startswith(
                        f"{process_system_name}/"
                    )
                    else report_name
                )
            suppression = (
                True
                if lookup_report_name is None
                else self._report_unit_duty_suppression(
                    lookup_report_name,
                    duty_lookup,
                )
            )
            filtered[report_name] = self._copy_report_data(
                report_data,
                suppress_duty=bool(suppression),
            )
        return filtered

    def _flatten_dict(
        self,
        data: dict,
        prefix: str,
        kpis: Dict[str, KPI],
        suppress_duty: bool = False,
    ):
        """Recursively flatten a nested dict into KPIs."""
        for key, val in data.items():
            if suppress_duty and str(key).strip().casefold().startswith(
                "duty"
            ):
                continue
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
                    self._flatten_dict(
                        val,
                        full_key,
                        kpis,
                        suppress_duty=suppress_duty,
                    )
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
        Deduplicates by exact native object identity to avoid repeated entries for
        the same stream registered under multiple aliases (e.g.
        'feed gas' and 'feed gas.feed gas').
        """
        from neqsim import jneqsim

        seen_streams = _NativeObjectIdentitySet()

        # Sort by key length so shorter (unqualified) names are preferred
        sorted_streams = sorted(self._streams.items(), key=lambda x: len(x[0]))

        for stream_name, s in sorted_streams:
            # Deduplicate by exact native object identity. Java hashCode() is
            # value-based for several NeqSim streams and may collide.
            if seen_streams.contains(s):
                continue
            seen_streams.add(s)

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
                return self._filter_json_report_duties(
                    json.loads(json_str)
                )
            except Exception:
                # Fallback: collect from children
                from neqsim import jneqsim
                combined = {}
                for ps in self.get_process_systems():
                    try:
                        native_ps_name = str(ps.getName())
                        ps_name = native_ps_name or "process"
                        report_obj = jneqsim.process.util.report.Report(ps)
                        r_str = str(report_obj.generateJsonReport())
                        r_data = json.loads(r_str)
                        if isinstance(r_data, dict):
                            r_data = self._filter_json_report_duties(
                                r_data,
                                process_system_name=native_ps_name,
                                process_system=ps,
                            )
                            for k, v in r_data.items():
                                combined[f"{ps_name}/{k}"] = v
                    except Exception:
                        pass
                return (
                    self._filter_json_report_duties(combined)
                    if combined
                    else None
                )
        try:
            from neqsim import jneqsim
            report_obj = jneqsim.process.util.report.Report(self._proc)
            json_str = str(report_obj.generateJsonReport())
            return self._filter_json_report_duties(
                json.loads(json_str)
            )
        except Exception:
            try:
                json_str = str(self._proc.getReport_json())
                return self._filter_json_report_duties(
                    json.loads(json_str)
                )
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

    def get_module_json_report(self, module_name: str) -> Optional[dict]:
        """Get the JSON report for a specific process system (module).

        For a ProcessModel with multiple child ProcessSystems, generates
        the report for the matching child system only.
        For a single ProcessSystem, returns the full report if the name matches.
        """
        ml = module_name.lower()
        for ps in self.get_process_systems():
            try:
                ps_name = str(ps.getName()) if ps.getName() else "process"
                if ml in ps_name.lower():
                    from neqsim import jneqsim
                    report_obj = jneqsim.process.util.report.Report(ps)
                    r_str = str(report_obj.generateJsonReport())
                    return self._filter_json_report_duties(
                        json.loads(r_str),
                        process_system_name=ps_name,
                        process_system=ps,
                    )
            except Exception:
                continue
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
                self._append_topology(
                    lines,
                    ordered_units,
                    process_system_name=ps_name,
                )
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

    def _append_topology(
        self,
        lines: list,
        ordered_units: list,
        process_system_name: str = "",
    ):
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
                if (
                    prop == "duty_kW"
                    and utype == "HeatExchanger"
                    and not self._heat_exchanger_solution_is_trusted(
                        self._indexed_unit_name_for_native(
                            u,
                            (
                                f"{process_system_name}/{name}"
                                if process_system_name
                                else name
                            ),
                        ),
                        u,
                        utype,
                    )
                ):
                    continue
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
                for j in range(_split_stream_probe_count(u, 10)):
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
