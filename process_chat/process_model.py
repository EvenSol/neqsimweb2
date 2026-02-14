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

    @classmethod
    def from_file(cls, filepath: str) -> "NeqSimProcessModel":
        """Load a ProcessSystem from a .neqsim or .xml file."""
        import neqsim
        from neqsim import jneqsim

        with open(filepath, "rb") as f:
            file_bytes = f.read()

        ext = os.path.splitext(filepath)[1].lower()
        if ext in (".neqsim", ".zip"):
            loaded = neqsim.open_neqsim(filepath)
        elif ext == ".xml":
            loaded = neqsim.open_xml(filepath)
        else:
            # Try .neqsim format first, fall back to xml
            loaded = neqsim.open_neqsim(filepath)
            if loaded is None:
                loaded = neqsim.open_xml(filepath)

        if loaded is None:
            raise RuntimeError(f"Failed to load process from: {filepath}")

        # Run to initialize internal state
        loaded.run()
        return cls(loaded, source_bytes=file_bytes)

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
            java_class = u.getClass().getSimpleName()
            if "Stream" in java_class and name not in self._streams:
                self._streams[name] = u
                raw_name_count[name] = raw_name_count.get(name, 0) + 1

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

    def list_units(self) -> List[UnitInfo]:
        """List all unit operations with type info."""
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
                ("outletPressure_bara", "getOutletPressure"),
            ]:
                if hasattr(u, getter):
                    try:
                        val = getattr(u, getter)()
                        if prop in ("power_kW", "duty_kW"):
                            val = float(val) / 1000.0  # W -> kW
                        else:
                            val = float(val)
                        props[prop] = val
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
        """Get a unit operation by name."""
        if name in self._units:
            return self._units[name]
        # Also try via process.getUnit()
        try:
            return self._proc.getUnit(name)
        except Exception:
            raise KeyError(f"Unit not found: {name}")

    def get_stream(self, name: str):
        """Get a stream by name (supports both qualified and unqualified names)."""
        if name in self._streams:
            return self._streams[name]
        # Try matching as unqualified name (e.g. "outStream" -> "intercooler.outStream")
        for key, s in self._streams.items():
            if key.endswith(f".{name}"):
                return s
        raise KeyError(f"Stream not found: {name}")

    # ----- Run and report -----

    def run(self, timeout_ms: int = 120000) -> ModelRunResult:
        """
        Run the process and extract KPIs and constraints.
        
        Args:
            timeout_ms: Timeout in milliseconds. If >0, runs in a thread.
        """
        proc = self._proc

        if timeout_ms > 0:
            thread = proc.runAsThread()
            thread.join(timeout_ms)
            if thread.isAlive():
                thread.interrupt()
                thread.join()
                raise TimeoutError(f"Process simulation timed out after {timeout_ms}ms")
        else:
            proc.run()

        return self._extract_results()

    def _extract_results(self) -> ModelRunResult:
        """Extract KPIs, constraints, and JSON report from solved process."""
        kpis: Dict[str, KPI] = {}
        constraints: List[ConstraintStatus] = []

        # Collect power and duty from all units
        total_power_kW = 0.0
        total_duty_kW = 0.0

        for name, u in self._units.items():
            if hasattr(u, "getPower"):
                try:
                    power_kW = float(u.getPower()) / 1000.0
                    kpis[f"{name}.power_kW"] = KPI(f"{name}.power_kW", power_kW, "kW")
                    total_power_kW += power_kW
                except Exception:
                    pass
            if hasattr(u, "getDuty"):
                try:
                    duty_kW = float(u.getDuty()) / 1000.0
                    kpis[f"{name}.duty_kW"] = KPI(f"{name}.duty_kW", duty_kW, "kW")
                    total_duty_kW += abs(duty_kW)
                except Exception:
                    pass

        kpis["total_power_kW"] = KPI("total_power_kW", total_power_kW, "kW")
        kpis["total_duty_kW"] = KPI("total_duty_kW", total_duty_kW, "kW")

        # Collect stream conditions for key streams
        for name, s in self._streams.items():
            try:
                P = float(s.getPressure("bara"))
                kpis[f"{name}.pressure_bara"] = KPI(f"{name}.pressure_bara", P, "bara")
            except Exception:
                pass
            try:
                T = float(s.getTemperature("C"))
                kpis[f"{name}.temperature_C"] = KPI(f"{name}.temperature_C", T, "C")
            except Exception:
                pass
            try:
                F = float(s.getFlowRate("kg/hr"))
                kpis[f"{name}.flow_kg_hr"] = KPI(f"{name}.flow_kg_hr", F, "kg/hr")
            except Exception:
                pass

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

        # Mass balance check — compare feed to the sum of ALL terminal
        # streams: last unit's outlets PLUS liquid drains from intermediate
        # separators (which leave the process boundary).
        try:
            proc = self._proc
            all_units = list(proc.getUnitOperations())
            feed_flow = 0.0
            product_flow = 0.0

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

                # Product flow: last unit's outlets
                last = all_units[-1]
                for m in ("getOutletStream", "getOutStream", "getGasOutStream"):
                    if hasattr(last, m):
                        try:
                            product_flow += float(getattr(last, m)().getFlowRate("kg/hr"))
                            break
                        except Exception:
                            pass
                if hasattr(last, "getLiquidOutStream"):
                    try:
                        product_flow += float(last.getLiquidOutStream().getFlowRate("kg/hr"))
                    except Exception:
                        pass

                # Add liquid drains from ALL intermediate separators
                # (these leave the process boundary and are not consumed downstream)
                for u in all_units[:-1]:  # exclude last unit, already counted
                    java_class = str(u.getClass().getSimpleName())
                    if "Separator" in java_class or "Scrubber" in java_class:
                        if hasattr(u, "getLiquidOutStream"):
                            try:
                                liq_flow = float(u.getLiquidOutStream().getFlowRate("kg/hr"))
                                if liq_flow > 0.01:  # non-trivial liquid drain
                                    product_flow += liq_flow
                            except Exception:
                                pass

            # Fallback: match by stream name keywords
            if feed_flow == 0.0:
                for name, s in self._streams.items():
                    flow = float(s.getFlowRate("kg/hr"))
                    lower = name.lower()
                    if any(kw in lower for kw in ("feed", "inlet", "well", "input")):
                        feed_flow += flow
                    elif any(kw in lower for kw in ("export", "product", "outlet", "output", "fuel")):
                        product_flow += flow

            if feed_flow > 0:
                balance_pct = abs(feed_flow - product_flow) / feed_flow * 100
                kpis["mass_balance_pct"] = KPI("mass_balance_pct", balance_pct, "%")
                status = "OK" if balance_pct < 1.0 else "WARN" if balance_pct < 5.0 else "VIOLATION"
                constraints.append(ConstraintStatus(
                    "mass_balance", status,
                    f"Feed={feed_flow:.0f} kg/hr, Products={product_flow:.0f} kg/hr, imbalance={balance_pct:.2f}%"
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
                ("outletPressure_bara", "getOutletPressure"),
            ]:
                if hasattr(u, getter):
                    try:
                        val = getattr(u, getter)()
                        if prop in ("power_kW", "duty_kW"):
                            val = float(val) / 1000.0
                        else:
                            val = float(val)
                        props[prop] = val
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

            # Outlet stream conditions
            outlet_str = ""
            for m in ("getOutletStream", "getOutStream", "getGasOutStream"):
                if hasattr(u, m):
                    try:
                        s = getattr(u, m)()
                        if s is not None:
                            sname = str(s.getName()) if s.getName() else "?"
                            T = float(s.getTemperature("C"))
                            P = float(s.getPressure("bara"))
                            outlet_str = f"OUT: {sname} (T={T:.1f}°C, P={P:.2f} bara)"
                            break
                    except Exception:
                        pass

            line = f"  [{idx}] {name} ({utype})"
            if prop_str:
                line += f" — {prop_str}"
            lines.append(line)
            if inlet_str:
                lines.append(f"        {inlet_str}")
            if outlet_str:
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
