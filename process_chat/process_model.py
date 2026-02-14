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

        # Extract calculated fluid properties from streams (viscosity, Z, JT, TVP, RVP, etc.)
        self._extract_stream_fluid_properties(kpis)

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

    def query_properties(self, query: str) -> str:
        """
        Run the model and return properties matching a natural-language query.
        
        Used by the LLM to answer READ-ONLY property questions like
        "What is the TVP of the feed gas?" or "What is the density of the export stream?"
        
        The query is matched against all KPI keys (case-insensitive substring match).
        Returns a formatted text with matching properties.
        """
        # Run the model to get current state
        result = self.run()
        
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
