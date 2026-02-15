"""
Process Builder — create NeqSim processes from scratch via chat.

Supports:
  - Building a full process from a structured specification (fluid + units)
  - Incremental additions to an existing built process
  - Python script generation (reproduces the process programmatically)
  - .neqsim file export (save / download)
"""
from __future__ import annotations

import os
import re
import tempfile
from typing import Any, Dict, List, Optional

from .process_model import NeqSimProcessModel


# ---------------------------------------------------------------------------
# EOS model mapping
# ---------------------------------------------------------------------------

_EOS_CLASSES = {
    "srk": "SystemSrkEos",
    "pr": "SystemPrEos",
    "pr78": "SystemPrEos1978",
    "cpa": "SystemSrkCPA",
    "cpa-srk": "SystemSrkCPA",
    "cpa-pr": "SystemPrCPA",
    "umr-pru": "SystemUMRPRUEos",
    "gerg2008": "SystemGERG2008Eos",
    "pcsaft": "SystemPCSAFT",
    "ideal": "SystemIdealGas",
}

# Equipment type → (Java sub-package.ClassName, default outlet getter)
_EQUIP_INFO: Dict[str, tuple] = {
    "stream":                 ("stream.Stream",                       None),
    "separator":              ("separator.Separator",                 "getGasOutStream"),
    "two_phase_separator":    ("separator.TwoPhaseSeparator",         "getGasOutStream"),
    "three_phase_separator":  ("separator.ThreePhaseSeparator",       "getGasOutStream"),
    "gas_scrubber":           ("separator.GasScrubber",               "getGasOutStream"),
    "compressor":             ("compressor.Compressor",               "getOutletStream"),
    "cooler":                 ("heatexchanger.Cooler",                "getOutletStream"),
    "heater":                 ("heatexchanger.Heater",                "getOutletStream"),
    "air_cooler":             ("heatexchanger.AirCooler",             "getOutletStream"),
    "water_cooler":           ("heatexchanger.WaterCooler",           "getOutletStream"),
    "heat_exchanger":         ("heatexchanger.HeatExchanger",         "getOutletStream"),
    "valve":                  ("valve.ThrottlingValve",               "getOutletStream"),
    "control_valve":          ("valve.ControlValve",                  "getOutletStream"),
    "expander":               ("expander.Expander",                   "getOutletStream"),
    "pump":                   ("pump.Pump",                           "getOutletStream"),
    "mixer":                  ("mixer.Mixer",                         "getOutletStream"),
    "splitter":               ("splitter.Splitter",                   "getSplitStream"),
    "pipeline":               ("pipeline.Pipeline",                   "getOutletStream"),
    "adiabatic_pipe":         ("pipeline.AdiabaticPipe",              "getOutletStream"),
    "simple_absorber":        ("absorber.SimpleAbsorber",             "getGasOutStream"),
    "simple_teg_absorber":    ("absorber.SimpleTEGAbsorber",          "getGasOutStream"),
    "gibbs_reactor":          ("reactor.GibbsReactor",                "getOutletStream"),
    "ejector":                ("ejector.Ejector",                     "getOutletStream"),
    "flare":                  ("flare.Flare",                         "getOutletStream"),
    "filter":                 ("filter.Filter",                       "getOutletStream"),
    "tank":                   ("tank.Tank",                           "getOutletStream"),
    "recycle":                ("util.Recycle",                        "getOutletStream"),
    "adjuster":               ("util.Adjuster",                      "getOutletStream"),
    "electrolyzer":           ("electrolyzer.Electrolyzer",          "getOutletStream"),
    "well_flow":              ("pipeline.PipeBeggsAndBrills",        "getOutletStream"),
    "adsorber":               ("absorber.SimpleAbsorber",            "getGasOutStream"),
    "distillation_column":    ("distillation.DistillationColumn",    "getGasOutStream"),
    "component_splitter":     ("splitter.ComponentSplitter",         "getOutletStream"),
    "gas_turbine":            ("compressor.Compressor",              "getOutletStream"),
    "membrane_separator":     ("separator.Separator",                "getGasOutStream"),
    "esp_pump":               ("pump.Pump",                          "getOutletStream"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_truthy(val) -> bool:
    """Return True for truthy values (handles string 'true'/'yes'/'1' too)."""
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    return str(val).lower().strip() in ("true", "yes", "1")


def _build_mixer(base, name, stream):
    """Build a Mixer with fallback if 2-arg constructor is unavailable."""
    try:
        return base.mixer.Mixer(name, stream)
    except Exception:
        m = base.mixer.Mixer(name)
        m.addStream(stream)
        return m


# ---------------------------------------------------------------------------
# Parameter setter mapping (key → Java setter call string)
# ---------------------------------------------------------------------------

_PARAM_SETTERS = {
    "outlet_pressure_bara":    lambda v: f"setOutletPressure({float(v)}, 'bara')",
    "outletpressure_bara":     lambda v: f"setOutletPressure({float(v)}, 'bara')",
    "outlet_pressure_barg":    lambda v: f"setOutletPressure({float(v)}, 'barg')",
    "outletpressure_barg":     lambda v: f"setOutletPressure({float(v)}, 'barg')",
    "outlet_temperature_c":    lambda v: f"setOutTemperature({float(v)}, 'C')",
    "outtemperature_c":        lambda v: f"setOutTemperature({float(v)}, 'C')",
    "isentropic_efficiency":   lambda v: f"setIsentropicEfficiency({float(v)})",
    "isentropicefficiency":    lambda v: f"setIsentropicEfficiency({float(v)})",
    "polytropic_efficiency":   lambda v: f"setPolytropicEfficiency({float(v)})",
    "polytropicefficiency":    lambda v: f"setPolytropicEfficiency({float(v)})",
    "pressure_drop_bar":       lambda v: f"setPressureDrop({float(v)})",
    "pressure_drop":           lambda v: f"setPressureDrop({float(v)})",
    "duty_kw":                 lambda v: f"setDuty({float(v) * 1000})",
    "duty":                    lambda v: f"setDuty({float(v) * 1000})",
    "speed":                   lambda v: f"setSpeed({float(v)})",
    "compression_ratio":       lambda v: f"setCompressionRatio({float(v)})",
    "compressionratio":        lambda v: f"setCompressionRatio({float(v)})",
    "use_polytropic_calc":     lambda v: f"setUsePolytropicCalc({str(_is_truthy(v))})",
    "usepolytropiccalc":       lambda v: f"setUsePolytropicCalc({str(_is_truthy(v))})",
    "cv":                      lambda v: f"setCv({float(v)})",
    "flow_coefficient":        lambda v: f"setCv({float(v)})",
    "percent_valve_opening":   lambda v: f"setPercentValveOpening({float(v)})",
    "efficiency":              lambda v: f"setEfficiency({float(v)})",
    "head":                    lambda v: f"setHead({float(v)})",
    "length":                  lambda v: f"setLength({float(v)})",
    "pipe_length":             lambda v: f"setLength({float(v)})",
    "diameter":                lambda v: f"setDiameter({float(v)})",
    "pipe_diameter":           lambda v: f"setDiameter({float(v)})",
    "roughness":               lambda v: f"setRoughness({float(v)})",
    "split_factor":            lambda v: f"setSplitFactor({float(v)})",
    "number_of_stages":        lambda v: f"setNumberOfStages({int(v)})",
    "numberofstages":          lambda v: f"setNumberOfStages({int(v)})",
    "ua_value":                lambda v: f"setUAvalue({float(v)})",
    "tolerance":               lambda v: f"setTolerance({float(v)})",
    "target_variable":         lambda v: f"setTargetVariable('{v}')",
    "target_value":            lambda v: f"setTargetValue({float(v)})",
    "power_kw":                lambda v: f"setPower({float(v) * 1000})",
    "energy_input_kw":         lambda v: f"setEnergyInput({float(v) * 1000})",
    "use_compressor_chart":    lambda v: None,  # handled separately, not a Java setter
    "chart_template":          lambda v: None,  # handled separately, not a Java setter
    "chart_num_speeds":        lambda v: None,  # handled separately, not a Java setter
}


# ---------------------------------------------------------------------------
# Helper: apply a parameter to a Java unit object
# ---------------------------------------------------------------------------

def _apply_param(unit, key: str, value):
    """Apply a single parameter to a NeqSim unit operation Java object."""
    k = key.lower().strip()

    if k in ("outlet_pressure_bara", "outletpressure_bara"):
        if hasattr(unit, "setOutletPressure"):
            unit.setOutletPressure(float(value), "bara")
    elif k in ("outlet_pressure_barg", "outletpressure_barg"):
        if hasattr(unit, "setOutletPressure"):
            unit.setOutletPressure(float(value), "barg")
    elif k in ("outlet_temperature_c", "outtemperature_c"):
        if hasattr(unit, "setOutTemperature"):
            unit.setOutTemperature(float(value), "C")
    elif k in ("isentropic_efficiency", "isentropicefficiency"):
        if hasattr(unit, "setIsentropicEfficiency"):
            unit.setIsentropicEfficiency(float(value))
    elif k in ("polytropic_efficiency", "polytropicefficiency"):
        if hasattr(unit, "setPolytropicEfficiency"):
            unit.setPolytropicEfficiency(float(value))
    elif k in ("pressure_drop_bar", "pressure_drop"):
        if hasattr(unit, "setPressureDrop"):
            unit.setPressureDrop(float(value))
    elif k in ("duty_kw", "duty"):
        if hasattr(unit, "setDuty"):
            unit.setDuty(float(value) * 1000)  # kW → W
    elif k == "speed":
        if hasattr(unit, "setSpeed"):
            unit.setSpeed(float(value))
    elif k in ("compression_ratio", "compressionratio"):
        if hasattr(unit, "setCompressionRatio"):
            unit.setCompressionRatio(float(value))
    elif k in ("use_polytropic_calc", "usepolytropiccalc"):
        if hasattr(unit, "setUsePolytropicCalc"):
            unit.setUsePolytropicCalc(_is_truthy(value))
    elif k in ("cv", "flow_coefficient"):
        if hasattr(unit, "setCv"):
            unit.setCv(float(value))
    elif k in ("percent_valve_opening",):
        if hasattr(unit, "setPercentValveOpening"):
            unit.setPercentValveOpening(float(value))
    elif k == "efficiency":
        if hasattr(unit, "setEfficiency"):
            unit.setEfficiency(float(value))
    elif k == "head":
        if hasattr(unit, "setHead"):
            unit.setHead(float(value))
    elif k in ("length", "pipe_length"):
        if hasattr(unit, "setLength"):
            unit.setLength(float(value))
    elif k in ("diameter", "pipe_diameter"):
        if hasattr(unit, "setDiameter"):
            unit.setDiameter(float(value))
    elif k == "roughness":
        if hasattr(unit, "setRoughness"):
            unit.setRoughness(float(value))
    elif k == "split_factor":
        if hasattr(unit, "setSplitFactor"):
            unit.setSplitFactor(float(value))
    elif k in ("number_of_stages", "numberofstages"):
        if hasattr(unit, "setNumberOfStages"):
            unit.setNumberOfStages(int(value))
    elif k == "ua_value":
        if hasattr(unit, "setUAvalue"):
            unit.setUAvalue(float(value))
    elif k == "tolerance":
        if hasattr(unit, "setTolerance"):
            unit.setTolerance(float(value))
    elif k in ("target_variable",):
        if hasattr(unit, "setTargetVariable"):
            unit.setTargetVariable(str(value))
    elif k in ("target_value",):
        if hasattr(unit, "setTargetValue"):
            unit.setTargetValue(float(value))
    elif k in ("power_kw",):
        if hasattr(unit, "setPower"):
            unit.setPower(float(value) * 1000)
    elif k in ("energy_input_kw",):
        if hasattr(unit, "setEnergyInput"):
            unit.setEnergyInput(float(value) * 1000)
    elif k == "use_compressor_chart":
        # Handled after unit creation in _create_unit
        pass
    elif k in ("chart_template", "chart_num_speeds"):
        # Handled after unit creation in _create_unit
        pass


# ---------------------------------------------------------------------------
# Helper: get outlet stream from a unit
# ---------------------------------------------------------------------------

def _get_outlet(unit, outlet_type: str = "gas"):
    """Get the appropriate outlet stream from a unit operation."""
    ot = outlet_type.lower()

    if ot in ("liquid", "oil", "liquidoutstream", "oiloutstream"):
        for m in ("getLiquidOutStream", "getOilOutStream"):
            if hasattr(unit, m):
                try:
                    return getattr(unit, m)()
                except Exception:
                    pass

    if ot in ("water", "aqueous", "wateroutstream"):
        if hasattr(unit, "getWaterOutStream"):
            try:
                return unit.getWaterOutStream()
            except Exception:
                pass

    # Default: gas / main outlet
    for m in ("getGasOutStream", "getOutletStream", "getOutStream"):
        if hasattr(unit, m):
            try:
                s = getattr(unit, m)()
                if s is not None:
                    return s
            except Exception:
                pass

    # Splitter: getSplitStream requires an index
    if hasattr(unit, "getSplitStream"):
        try:
            s = unit.getSplitStream(0)
            if s is not None:
                return s
        except Exception:
            pass

    # For Stream objects, the stream itself is the outlet
    return unit


# ---------------------------------------------------------------------------
# ProcessBuilder class
# ---------------------------------------------------------------------------

class ProcessBuilder:
    """Build a NeqSim process from a structured specification dict.

    Usage::

        builder = ProcessBuilder()
        model = builder.build_from_spec({
            "name": "Gas Compression",
            "fluid": {
                "eos_model": "srk",
                "components": {"methane": 0.85, "ethane": 0.07, ...},
                "composition_basis": "mole_fraction",
                "temperature_C": 25.0,
                "pressure_bara": 50.0,
                "total_flow": 10000, "flow_unit": "kg/hr",
            },
            "process": [
                {"name": "feed gas",          "type": "stream"},
                {"name": "inlet separator",   "type": "separator"},
                {"name": "compressor 1",      "type": "compressor",
                 "params": {"outlet_pressure_bara": 100}},
                {"name": "aftercooler",       "type": "cooler",
                 "params": {"outlet_temperature_C": 35}},
            ],
        })
        script = builder.to_python_script()
        raw    = builder.save_neqsim_bytes()
    """

    def __init__(self):
        self._spec: Optional[dict] = None
        self._model: Optional[NeqSimProcessModel] = None
        self._process_name = "New Process"
        self._build_log: List[str] = []

    # -- Public properties --------------------------------------------------

    @property
    def model(self) -> Optional[NeqSimProcessModel]:
        return self._model

    @property
    def spec(self) -> Optional[dict]:
        return self._spec

    @property
    def process_name(self) -> str:
        return self._process_name

    @property
    def build_log(self) -> List[str]:
        return list(self._build_log)

    # -- Build from spec ----------------------------------------------------

    def build_from_spec(self, spec: dict) -> NeqSimProcessModel:
        """Build a complete process from a specification dict.

        The spec must contain:
          - ``fluid`` — fluid definition (components, EOS, conditions)
          - ``process`` — ordered list of unit steps

        Returns the wrapped :class:`NeqSimProcessModel`.
        """
        from neqsim import jneqsim

        self._spec = spec
        self._process_name = spec.get("name", "New Process")
        self._build_log.clear()

        fluid_spec = spec.get("fluid", {})
        process_steps = spec.get("process", [])

        if not process_steps:
            raise ValueError("Process spec must contain at least one step in 'process'.")

        # 1. Create the thermodynamic fluid
        fluid = self._create_fluid(fluid_spec)
        self._build_log.append(
            f"Created fluid: EOS={fluid_spec.get('eos_model', 'srk')}, "
            f"{len(fluid_spec.get('components', {}))} components"
        )

        # 2. Build process system
        ProcessSystem = jneqsim.process.processmodel.ProcessSystem
        proc = ProcessSystem()

        built_units: Dict[str, Any] = {}   # name → Java unit object
        prev_unit = None                   # previous unit object
        prev_outlet_type = "gas"           # which outlet the previous step requested

        for step in process_steps:
            name = step["name"]
            eq_type = step["type"].lower()
            params = step.get("params", {})
            outlet_type = step.get("outlet", "gas")
            inlet_ref = step.get("inlet", None)

            if eq_type == "stream":
                # ---- Feed stream ----
                StreamClass = jneqsim.process.equipment.stream.Stream
                stream_fluid = fluid.clone()

                # Override T/P from params
                if "temperature_C" in params:
                    stream_fluid.setTemperature(float(params["temperature_C"]), "C")
                if "pressure_bara" in params:
                    stream_fluid.setPressure(float(params["pressure_bara"]), "bara")
                if "flow_rate" in params and "flow_unit" in params:
                    stream_fluid.setTotalFlowRate(float(params["flow_rate"]),
                                                  params["flow_unit"])

                unit = StreamClass(name, stream_fluid)
                proc.add(unit)
                built_units[name] = unit
                prev_unit = unit
                prev_outlet_type = outlet_type
                self._build_log.append(f"Added stream: {name}")

            else:
                # ---- Equipment unit ----
                # Resolve inlet stream
                inlet_stream = self._resolve_inlet(
                    inlet_ref, built_units, prev_unit, prev_outlet_type
                )
                if inlet_stream is None:
                    raise ValueError(
                        f"No inlet stream for unit '{name}'. "
                        "Define a stream first or specify 'inlet'."
                    )

                unit = self._create_unit(name, eq_type, inlet_stream, params)
                proc.add(unit)
                built_units[name] = unit
                prev_unit = unit
                prev_outlet_type = outlet_type
                param_desc = ", ".join(f"{k}={v}" for k, v in params.items())
                self._build_log.append(
                    f"Added {eq_type}: {name}"
                    + (f" ({param_desc})" if param_desc else "")
                )

        # 3. Run the process
        self._build_log.append("Running simulation...")
        NeqSimProcessModel._run_until_converged(proc)

        # 4. Wrap in NeqSimProcessModel
        self._model = NeqSimProcessModel.from_process_system(proc)
        self._build_log.append("Process built and converged successfully.")
        return self._model

    # -- Python script export -----------------------------------------------

    def to_python_script(self) -> str:
        """Generate a Python script that reproduces this process."""
        if self._spec is None:
            return "# No process specification available.\n"

        lines: List[str] = []
        fluid_spec = self._spec.get("fluid", {})
        process_steps = self._spec.get("process", [])

        # --- Header ---
        lines.append('"""')
        lines.append(f"NeqSim Process: {self._process_name}")
        lines.append("Auto-generated by NeqSim Process Chat")
        lines.append('"""')
        lines.append("from neqsim import jneqsim")
        lines.append("import neqsim")
        lines.append("")

        # --- Fluid ---
        eos = fluid_spec.get("eos_model", "srk").lower()
        eos_class = _EOS_CLASSES.get(eos, "SystemSrkEos")
        temp_C = fluid_spec.get("temperature_C", 25.0)
        pres_bara = fluid_spec.get("pressure_bara", 50.0)
        temp_K = temp_C + 273.15
        components = fluid_spec.get("components", {})
        basis = fluid_spec.get("composition_basis", "mole_fraction")
        total_flow = fluid_spec.get("total_flow", 100.0)
        flow_unit = fluid_spec.get("flow_unit", "kg/hr")

        lines.append("# ── Create fluid ──")
        lines.append(f"fluid = jneqsim.thermo.system.{eos_class}({temp_K}, {pres_bara})")

        if basis in ("mole_fraction", "mole_percent"):
            for comp, frac in components.items():
                actual = frac / 100.0 if basis == "mole_percent" else frac
                lines.append(f"fluid.addComponent('{comp}', {actual})")
            lines.append(f"fluid.setTotalFlowRate({total_flow}, '{flow_unit}')")
        elif basis == "molar_flow_mol_sec":
            for comp, flow in components.items():
                lines.append(f"fluid.addComponent('{comp}', {flow}, 'mol/sec')")
        elif basis == "mass_flow_kg_hr":
            for comp, flow in components.items():
                lines.append(f"fluid.addComponent('{comp}', {flow}, 'kg/hr')")
        else:
            for comp, frac in components.items():
                lines.append(f"fluid.addComponent('{comp}', {frac})")
            lines.append(f"fluid.setTotalFlowRate({total_flow}, '{flow_unit}')")

        mixing_rule = fluid_spec.get("mixing_rule", 2)
        lines.append(f"fluid.setMixingRule({mixing_rule})")
        lines.append("")

        # --- Process ---
        lines.append("# ── Build process ──")
        lines.append("process = jneqsim.process.processmodel.ProcessSystem()")
        lines.append("")

        var_names: Dict[str, str] = {}      # unit name → Python variable
        prev_var: Optional[str] = None
        prev_type: Optional[str] = None
        prev_outlet: str = "gas"

        for step in process_steps:
            name = step["name"]
            eq_type = step["type"].lower()
            params = step.get("params", {})
            outlet_type = step.get("outlet", "gas")
            inlet_ref = step.get("inlet", None)

            var = _to_var_name(name)
            var_names[name] = var
            lines.append(f"# {name}")

            if eq_type == "stream":
                lines.append(
                    f"{var} = jneqsim.process.equipment.stream.Stream('{name}', fluid)"
                )
                lines.append(f"process.add({var})")
                prev_var = var
                prev_type = "stream"
                prev_outlet = outlet_type
                lines.append("")
                continue

            # Determine inlet expression
            inlet_expr = self._inlet_expression(
                inlet_ref, var_names, prev_var, prev_type, prev_outlet,
                self._spec.get("process", [])
            )

            # Java class path
            info = _EQUIP_INFO.get(eq_type)
            if info:
                java_path = f"jneqsim.process.equipment.{info[0]}"
            else:
                java_path = f"jneqsim.process.equipment.{eq_type}"

            lines.append(f"{var} = {java_path}('{name}', {inlet_expr})")

            # Parameters
            for pkey, pval in params.items():
                setter_fn = _PARAM_SETTERS.get(pkey.lower().strip())
                if setter_fn:
                    setter_call = setter_fn(pval)
                    if setter_call is not None:
                        lines.append(f"{var}.{setter_call}")

            lines.append(f"process.add({var})")
            prev_var = var
            prev_type = eq_type
            prev_outlet = outlet_type
            lines.append("")

        # --- Run & save ---
        lines.append("# ── Run process ──")
        lines.append("process.run()")
        lines.append("")
        safe = _safe_filename(self._process_name)
        lines.append("# ── Save to file ──")
        lines.append(f"neqsim.save_neqsim(process, '{safe}.neqsim')")
        lines.append("")
        lines.append('print("Process simulation complete!")')
        lines.append("")

        return "\n".join(lines)

    # -- .neqsim file export ------------------------------------------------

    def save_neqsim_bytes(self) -> Optional[bytes]:
        """Serialize the built process to .neqsim ZIP bytes for download."""
        if self._model is None:
            return None

        import neqsim
        proc = self._model.get_process()

        with tempfile.NamedTemporaryFile(suffix=".neqsim", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            neqsim.save_neqsim(proc, tmp_path)
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # -- Build summary (for LLM context) ------------------------------------

    def get_build_summary(self) -> str:
        """Return a concise text summary of the current build state."""
        if self._spec is None:
            return "No process has been built yet."

        parts = [f"Process: {self._process_name}"]

        fluid_spec = self._spec.get("fluid", {})
        comps = fluid_spec.get("components", {})
        if comps:
            comp_list = ", ".join(f"{c}: {v}" for c, v in comps.items())
            parts.append(f"Fluid: {fluid_spec.get('eos_model', 'srk').upper()} EOS — {comp_list}")
            parts.append(
                f"  Conditions: {fluid_spec.get('temperature_C', '?')}°C, "
                f"{fluid_spec.get('pressure_bara', '?')} bara, "
                f"{fluid_spec.get('total_flow', '?')} {fluid_spec.get('flow_unit', 'kg/hr')}"
            )

        steps = self._spec.get("process", [])
        if steps:
            parts.append(f"Units ({len(steps)}):")
            for i, s in enumerate(steps):
                p_str = ""
                if s.get("params"):
                    p_str = " — " + ", ".join(
                        f"{k}={v}" for k, v in s["params"].items()
                    )
                parts.append(f"  [{i}] {s['name']} ({s['type']}){p_str}")

        if self._build_log:
            parts.append("Build log:")
            for entry in self._build_log[-5:]:
                parts.append(f"  • {entry}")

        return "\n".join(parts)

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  Private helpers                                                     ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    def _create_fluid(self, fluid_spec: dict):
        """Create a NeqSim thermoSystem from a fluid specification."""
        from neqsim import jneqsim

        # --- normal component-based creation ----------------------------------
        eos = fluid_spec.get("eos_model", "srk").lower()
        components = fluid_spec.get("components", {})
        basis = fluid_spec.get("composition_basis", "mole_fraction")
        temp_C = fluid_spec.get("temperature_C", 25.0)
        pres_bara = fluid_spec.get("pressure_bara", 50.0)
        total_flow = fluid_spec.get("total_flow", 100.0)
        flow_unit = fluid_spec.get("flow_unit", "kg/hr")

        temp_K = temp_C + 273.15
        eos_class_name = _EOS_CLASSES.get(eos, "SystemSrkEos")
        EosClass = getattr(jneqsim.thermo.system, eos_class_name)

        fluid = EosClass(temp_K, pres_bara)

        if basis in ("mole_fraction", "mole_percent"):
            for comp, frac in components.items():
                actual = frac / 100.0 if basis == "mole_percent" else frac
                fluid.addComponent(comp, float(actual))
            fluid.setTotalFlowRate(float(total_flow), flow_unit)
        elif basis == "molar_flow_mol_sec":
            for comp, flow in components.items():
                fluid.addComponent(comp, float(flow), "mol/sec")
        elif basis == "mass_flow_kg_hr":
            for comp, flow in components.items():
                fluid.addComponent(comp, float(flow), "kg/hr")
        else:
            # Default: treat values as mole fractions
            for comp, frac in components.items():
                fluid.addComponent(comp, float(frac))
            fluid.setTotalFlowRate(float(total_flow), flow_unit)

        mixing_rule = fluid_spec.get("mixing_rule", 2)
        fluid.setMixingRule(int(mixing_rule))

        return fluid

    def _resolve_inlet(
        self,
        inlet_ref: Optional[str],
        built_units: Dict[str, Any],
        prev_unit,
        prev_outlet_type: str,
    ):
        """Resolve the inlet stream for a new unit."""
        if inlet_ref:
            # Explicit reference: "unit_name" or "unit_name.liquidOutStream"
            if "." in inlet_ref:
                parts = inlet_ref.split(".", 1)
                ref_unit = built_units.get(parts[0])
                if ref_unit:
                    return _get_outlet(ref_unit, parts[1])
            elif inlet_ref in built_units:
                return _get_outlet(built_units[inlet_ref], "gas")

        # Auto-chain from previous unit
        if prev_unit is not None:
            return _get_outlet(prev_unit, prev_outlet_type)

        return None

    def _create_unit(self, name: str, eq_type: str, inlet_stream, params: dict):
        """Instantiate a NeqSim equipment unit and apply its parameters."""
        from neqsim import jneqsim
        base = jneqsim.process.equipment

        constructors = {
            "separator":              lambda n, s: base.separator.Separator(n, s),
            "two_phase_separator":    lambda n, s: base.separator.TwoPhaseSeparator(n, s),
            "three_phase_separator":  lambda n, s: base.separator.ThreePhaseSeparator(n, s),
            "gas_scrubber":           lambda n, s: base.separator.GasScrubber(n, s),
            "compressor":             lambda n, s: base.compressor.Compressor(n, s),
            "cooler":                 lambda n, s: base.heatexchanger.Cooler(n, s),
            "heater":                 lambda n, s: base.heatexchanger.Heater(n, s),
            "air_cooler":             lambda n, s: base.heatexchanger.AirCooler(n, s),
            "water_cooler":           lambda n, s: base.heatexchanger.WaterCooler(n, s),
            "heat_exchanger":         lambda n, s: base.heatexchanger.HeatExchanger(n, s),
            "valve":                  lambda n, s: base.valve.ThrottlingValve(n, s),
            "control_valve":          lambda n, s: base.valve.ControlValve(n, s),
            "expander":               lambda n, s: base.expander.Expander(n, s),
            "pump":                   lambda n, s: base.pump.Pump(n, s),
            "mixer":                  lambda n, s: _build_mixer(base, n, s),
            "splitter":               lambda n, s: base.splitter.Splitter(n, s),
            "pipeline":               lambda n, s: base.pipeline.Pipeline(n, s),
            "adiabatic_pipe":         lambda n, s: base.pipeline.AdiabaticPipe(n, s),
            "simple_absorber":        lambda n, s: base.absorber.SimpleAbsorber(n, s),
            "simple_teg_absorber":    lambda n, s: base.absorber.SimpleTEGAbsorber(n, s),
            "gibbs_reactor":          lambda n, s: base.reactor.GibbsReactor(n, s),
            "ejector":                lambda n, s: base.ejector.Ejector(n, s),
            "flare":                  lambda n, s: base.flare.Flare(n, s),
            "filter":                 lambda n, s: base.filter.Filter(n, s),
            "tank":                   lambda n, s: base.tank.Tank(n, s),
            "recycle":                lambda n, s: base.util.Recycle(n, s),
            "adjuster":               lambda n, s: base.util.Adjuster(n, s),
        }

        # Try dynamic class resolution for newer equipment types
        _DYNAMIC_TYPES = {
            "electrolyzer":       "electrolyzer.Electrolyzer",
            "well_flow":          "pipeline.PipeBeggsAndBrills",
            "adsorber":           "absorber.SimpleAbsorber",
            "distillation_column":"distillation.DistillationColumn",
            "component_splitter": "splitter.ComponentSplitter",
            "gas_turbine":        "compressor.Compressor",
            "membrane_separator": "separator.Separator",
            "esp_pump":           "pump.Pump",
        }

        ctor = constructors.get(eq_type)
        if ctor is None:
            # Try dynamic resolution
            dyn_path = _DYNAMIC_TYPES.get(eq_type)
            if dyn_path:
                try:
                    parts = dyn_path.split(".")
                    pkg = getattr(base, parts[0])
                    cls = getattr(pkg, parts[1])
                    unit = cls(name, inlet_stream)
                    for k, v in params.items():
                        _apply_param(unit, k, v)
                    return unit
                except Exception as e:
                    raise ValueError(f"Failed to create '{eq_type}': {e}") from e
            raise ValueError(f"Unknown equipment type: '{eq_type}'")

        unit = ctor(name, inlet_stream)

        # Apply parameters
        for k, v in params.items():
            _apply_param(unit, k, v)

        # If use_compressor_chart is requested, generate and apply a chart
        if eq_type == "compressor" and _is_truthy(params.get("use_compressor_chart")):
            try:
                CompressorChartGenerator = (
                    jneqsim.process.equipment.compressor.CompressorChartGenerator
                )
                chart_template = str(params.get("chart_template", "CENTRIFUGAL_STANDARD"))
                chart_num_speeds = int(params.get("chart_num_speeds", 5))

                unit.run()  # need a run before chart generation
                generator = CompressorChartGenerator(unit)
                chart = generator.generateFromTemplate(chart_template, chart_num_speeds)
                unit.setCompressorChartType('interpolate and extrapolate')
                unit.setCompressorChart(chart)
                unit.getCompressorChart().setHeadUnit('kJ/kg')
                unit.setSolveSpeed(True)
                unit.setUsePolytropicCalc(True)
                unit.run()
            except Exception:
                pass  # fall back to outlet-pressure mode

        return unit

    # -- Python script helpers ----------------------------------------------

    def _inlet_expression(
        self,
        inlet_ref: Optional[str],
        var_names: Dict[str, str],
        prev_var: Optional[str],
        prev_type: Optional[str],
        prev_outlet: str,
        all_steps: List[dict],
    ) -> str:
        """Produce the Python expression for a unit's inlet stream."""
        # Explicit reference
        if inlet_ref:
            if "." in inlet_ref:
                parts = inlet_ref.split(".", 1)
                ref_var = var_names.get(parts[0], parts[0])
                return f"{ref_var}.{parts[1]}()"
            elif inlet_ref in var_names:
                # Find type to pick outlet getter
                for st in all_steps:
                    if st["name"] == inlet_ref:
                        rtype = st["type"].lower()
                        rout = st.get("outlet", "gas")
                        return self._outlet_call_expr(
                            var_names[inlet_ref], rtype, rout
                        )
                return f"{var_names[inlet_ref]}"

        # Auto-chain from previous unit
        if prev_var:
            return self._outlet_call_expr(prev_var, prev_type or "stream", prev_outlet)

        return "fluid"

    @staticmethod
    def _outlet_call_expr(var: str, eq_type: str, outlet_type: str = "gas") -> str:
        """Return e.g. ``sep.getGasOutStream()`` for script generation."""
        if eq_type == "stream":
            return var  # Stream IS the outlet

        ot = outlet_type.lower()
        if ot in ("liquid", "oil"):
            return f"{var}.getLiquidOutStream()"
        if ot == "water":
            return f"{var}.getWaterOutStream()"

        # Splitter needs index argument
        if eq_type == "splitter":
            return f"{var}.getSplitStream(0)"

        info = _EQUIP_INFO.get(eq_type)
        if info and info[1]:
            return f"{var}.{info[1]}()"
        return f"{var}.getOutletStream()"


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def _to_var_name(name: str) -> str:
    """Convert a unit/stream name to a valid Python variable name."""
    var = re.sub(r"[^a-zA-Z0-9]", "_", name.lower())
    var = re.sub(r"_+", "_", var).strip("_")
    if not var or var[0].isdigit():
        var = "_" + var
    return var


def _safe_filename(name: str) -> str:
    """Convert a process name to a safe filename (no extension)."""
    s = re.sub(r"[^a-zA-Z0-9_\- ]", "", name)
    return s.replace(" ", "_").lower() or "process"
