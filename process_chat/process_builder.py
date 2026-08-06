"""
Process Builder — create NeqSim processes from scratch via chat.

Supports:
  - Building a full process from a structured specification (fluid + units)
  - Incremental additions to an existing built process
  - Python script generation (reproduces the process programmatically)
  - .neqsim file export (save / download)
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
from typing import Any, Dict, List, Optional

from .graph_schema import (
    canonical_material_output_port,
    material_connection_name,
)
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

_COMPRESSOR_CHART_TEMPLATES = (
    "CENTRIFUGAL_STANDARD",
    "CENTRIFUGAL_HIGH_FLOW",
    "CENTRIFUGAL_HIGH_HEAD",
    "PIPELINE",
    "EXPORT",
    "INJECTION",
    "GAS_LIFT",
    "REFRIGERATION",
    "BOOSTER",
    "SINGLE_STAGE",
    "MULTISTAGE_INLINE",
    "INTEGRALLY_GEARED",
    "OVERHUNG",
)

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
    "pipeline":               ("pipeline.PipeBeggsAndBrills",        "getOutletStream"),
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
    "ua_w_per_k":              lambda v: f"setUAvalue({float(v)})",
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
    "roughness":               lambda v: f"setPipeWallRoughness({float(v)})",
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
    elif k in ("ua_w_per_k", "ua_value"):
        if hasattr(unit, "setUAvalue"):
            unit.setUAvalue(float(value))
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
        elif hasattr(unit, "setIsentropicEfficiency"):
            # Pump exposes isentropic efficiency rather than the generic
            # separator-style setEfficiency API. Native ESPPump currently
            # stores this inherited property as percent and divides it by 100
            # during its multiphase calculation, while Studio uses fractions.
            efficiency = float(value)
            try:
                is_esp_pump = (
                    str(unit.getClass().getSimpleName()).strip().lower()
                    == "esppump"
                )
            except Exception:
                is_esp_pump = False
            if is_esp_pump:
                efficiency *= 100.0
            unit.setIsentropicEfficiency(efficiency)
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
        if hasattr(unit, "setPipeWallRoughness"):
            unit.setPipeWallRoughness(float(value))
        elif hasattr(unit, "setRoughness"):
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
    elif k in ("auto_size", "design_gas_load_factor_m_per_s"):
        # Applied as one post-solve mechanical-design request so sizing uses
        # the converged native feed state.
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

    # -- Native fluid construction ------------------------------------------

    def create_fluid_from_spec(self, fluid_spec: dict):
        """Create a fresh NeqSim thermodynamic system from one fluid definition.

        Temperature is expressed in degrees Celsius, pressure in absolute bara,
        and flow uses the explicit flow_unit in the specification. Repeated
        calls create independent native systems for separate process inlets.
        """
        if not isinstance(fluid_spec, dict):
            raise ValueError("Fluid specification must be an object.")
        return self._create_fluid(dict(fluid_spec))

    def create_inlet_streams(
        self,
        inlet_specs: List[dict],
    ) -> Dict[str, Any]:
        """Create independent native stream objects for validated process inlets.

        Each entry requires inlet_id, name, and a ProcessBuilder-compatible
        fluid_spec. Returned streams are keyed by inlet id and are not attached
        to a ProcessSystem, leaving graph execution responsible for ordering.
        """
        from neqsim import jneqsim

        if not isinstance(inlet_specs, list) or not inlet_specs:
            raise ValueError("Inlet specifications must be a non-empty array.")

        StreamClass = jneqsim.process.equipment.stream.Stream
        streams: Dict[str, Any] = {}
        stream_names: set[str] = set()
        for inlet_index, inlet_spec in enumerate(inlet_specs):
            if not isinstance(inlet_spec, dict):
                raise ValueError(
                    f"Inlet specification {inlet_index} must be an object."
                )
            inlet_id = str(inlet_spec.get("inlet_id", "")).strip()
            stream_name = str(inlet_spec.get("name", "")).strip()
            fluid_spec = inlet_spec.get("fluid_spec")
            if not inlet_id:
                raise ValueError(
                    f"Inlet specification {inlet_index} requires inlet_id."
                )
            if not stream_name:
                raise ValueError(f"Inlet '{inlet_id}' requires a stream name.")
            if inlet_id in streams:
                raise ValueError(f"Inlet id '{inlet_id}' is duplicated.")
            if stream_name in stream_names:
                raise ValueError(f"Inlet stream name '{stream_name}' is duplicated.")
            if not isinstance(fluid_spec, dict):
                raise ValueError(f"Inlet '{inlet_id}' requires a fluid_spec object.")

            fluid = self.create_fluid_from_spec(fluid_spec)
            streams[inlet_id] = StreamClass(stream_name, fluid)
            stream_names.add(stream_name)
        return streams

    def resolve_material_output(
        self,
        endpoint: dict,
        inlet_streams: Dict[str, Any],
        unit_objects: Dict[str, Any],
    ):
        """Resolve one validated graph source endpoint to a native stream.

        Inlets expose material port 'out'. Unit ports use explicit names:
        'out'/'main', 'gas'/'vapor', 'liquid'/'oil', 'water'/'aqueous', or
        indexed splitter ports such as 'out_0' and 'split_1'. Missing objects,
        unsupported ports, failed getters, and null streams are reported
        explicitly instead of silently falling back to another outlet.
        """
        if not isinstance(endpoint, dict):
            raise ValueError("Material source endpoint must be an object.")
        if not isinstance(inlet_streams, dict) or not isinstance(unit_objects, dict):
            raise ValueError("Material source registries must be objects.")

        source_kind = str(endpoint.get("kind", "")).strip().lower()
        source_id = str(endpoint.get("id", "")).strip()
        source_port = str(endpoint.get("port", "")).strip().lower()
        if not source_id or not source_port:
            raise ValueError("Material source endpoint requires id and port.")

        if source_kind == "inlet":
            if source_port != "out":
                raise ValueError(
                    f"Inlet '{source_id}' exposes only material output port 'out'."
                )
            if source_id not in inlet_streams:
                raise ValueError(f"Unknown material inlet '{source_id}'.")
            return inlet_streams[source_id]

        if source_kind != "unit":
            raise ValueError(
                f"Unsupported material source kind '{source_kind or '<empty>'}'."
            )
        if source_id not in unit_objects:
            raise ValueError(f"Unknown material unit '{source_id}'.")
        unit = unit_objects[source_id]

        try:
            native_unit_class = str(
                unit.getClass().getSimpleName()
            ).strip().lower()
        except Exception:
            native_unit_class = ""
        if native_unit_class == "heatexchanger":
            source_port = canonical_material_output_port(
                source_port,
                "heat_exchanger",
            )

        heat_exchanger_port_index = {
            "hot_out": 0,
            "cold_out": 1,
        }.get(source_port)
        if heat_exchanger_port_index is not None:
            getter_names = ("getOutStream",)
            getter_args = (heat_exchanger_port_index,)
        else:
            getter_names = None
            getter_args = ()

        indexed_port = re.fullmatch(r"(?:out|split)[_-]?(\d+)", source_port)
        if getter_names is not None:
            pass
        elif indexed_port:
            getter_names = ("getSplitStream",)
            getter_args = (int(indexed_port.group(1)),)
        else:
            getter_args = ()
            getter_names_by_port = {
                "out": ("getOutletStream", "getOutStream", "getGasOutStream"),
                "main": ("getOutletStream", "getOutStream", "getGasOutStream"),
                "gas": ("getGasOutStream",),
                "vapor": ("getGasOutStream",),
                "liquid": ("getLiquidOutStream", "getOilOutStream"),
                "oil": ("getOilOutStream", "getLiquidOutStream"),
                "water": ("getWaterOutStream",),
                "aqueous": ("getWaterOutStream",),
            }
            getter_names = getter_names_by_port.get(source_port)
            if getter_names is None:
                raise ValueError(
                    f"Unsupported material output port '{source_port}' on "
                    f"unit '{source_id}'."
                )

        last_error: Optional[Exception] = None
        for getter_name in getter_names:
            if not hasattr(unit, getter_name):
                continue
            try:
                stream = getattr(unit, getter_name)(*getter_args)
            except Exception as exc:
                last_error = exc
                continue
            if stream is not None:
                return stream

        message = (
            f"Unit '{source_id}' could not provide material output "
            f"port '{source_port}'."
        )
        if last_error is not None:
            raise ValueError(message) from last_error
        raise ValueError(message)

    @staticmethod
    def _configure_graph_splitter(
        unit: Any,
        unit_id: str,
        unit_spec: dict,
    ) -> List[float]:
        """Map declared indexed output ports to normalized native split factors."""
        ports = unit_spec.get("ports")
        if not isinstance(ports, dict):
            raise ValueError(f"Splitter '{unit_id}' requires a ports object.")
        material_outputs = ports.get("material_out")
        if not isinstance(material_outputs, list) or len(material_outputs) < 2:
            raise ValueError(
                f"Splitter '{unit_id}' requires at least two material output ports."
            )

        params = unit_spec.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"Splitter '{unit_id}' params must be an object.")
        if "split_factors" in params:
            raw_factors = params["split_factors"]
        elif "split_factor" in params:
            if len(material_outputs) != 2:
                raise ValueError(
                    f"Splitter '{unit_id}' legacy split_factor requires "
                    "exactly two material output ports."
                )
            if type(params["split_factor"]) is bool:
                raise ValueError(
                    f"Splitter '{unit_id}' split_factor must be numeric."
                )
            try:
                legacy_factor = float(params["split_factor"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Splitter '{unit_id}' split_factor must be numeric."
                ) from exc
            raw_factors = [legacy_factor, 1.0 - legacy_factor]
        else:
            raw_factors = [1.0] * len(material_outputs)
        if not isinstance(raw_factors, list):
            raise ValueError(
                f"Splitter '{unit_id}' requires a split_factors array."
            )
        if len(raw_factors) != len(material_outputs):
            raise ValueError(
                f"Splitter '{unit_id}' split_factors must match its "
                "material output ports."
            )

        factors_by_index: Dict[int, float] = {}
        for port_name, raw_factor in zip(material_outputs, raw_factors):
            cleaned_port = str(port_name).strip().lower()
            indexed_port = re.fullmatch(
                r"(?:out|split)[_-]?(\d+)",
                cleaned_port,
            )
            if indexed_port is None:
                raise ValueError(
                    f"Splitter '{unit_id}' output port '{cleaned_port}' "
                    "must identify a native split index."
                )
            split_index = int(indexed_port.group(1))
            if split_index in factors_by_index:
                raise ValueError(
                    f"Splitter '{unit_id}' maps multiple ports to split index "
                    f"{split_index}."
                )
            if type(raw_factor) is bool:
                raise ValueError(
                    f"Splitter '{unit_id}' split factors must be numeric."
                )
            try:
                factor = float(raw_factor)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Splitter '{unit_id}' split factors must be numeric."
                ) from exc
            if not math.isfinite(factor) or factor < 0.0:
                raise ValueError(
                    f"Splitter '{unit_id}' split factors must be finite and "
                    "non-negative."
                )
            factors_by_index[split_index] = factor

        expected_indices = set(range(len(material_outputs)))
        if set(factors_by_index) != expected_indices:
            raise ValueError(
                f"Splitter '{unit_id}' output indices must be contiguous from "
                f"0 to {len(material_outputs) - 1}."
            )

        factor_scale = max(factors_by_index.values())
        if factor_scale <= 0.0:
            raise ValueError(
                f"Splitter '{unit_id}' split factors must have a positive sum."
            )
        scaled_factors = [
            factors_by_index[index] / factor_scale
            for index in range(len(material_outputs))
        ]
        scaled_total = sum(scaled_factors)
        normalized_factors = [
            factor / scaled_total for factor in scaled_factors
        ]

        if not hasattr(unit, "setSplitFactors"):
            raise ValueError(
                f"Splitter '{unit_id}' does not expose native setSplitFactors."
            )
        native_factors: Any = normalized_factors
        try:
            from jpype import JArray, JDouble
        except ImportError:
            pass
        else:
            native_factors = JArray(JDouble)(normalized_factors)
        try:
            unit.setSplitFactors(native_factors)
        except Exception as exc:
            raise ValueError(
                f"Splitter '{unit_id}' could not apply native split factors."
            ) from exc

        return normalized_factors

    def _add_terminal_material_streams(
        self,
        unit_specs: List[dict],
        connections: List[dict],
        inlet_streams: Dict[str, Any],
        unit_objects: Dict[str, Any],
        process_system: Any,
        reserved_names: set[str],
    ) -> Dict[str, Any]:
        """Add named native streams for every unconnected material output port."""
        from neqsim import jneqsim

        unit_types = {
            str(unit_spec.get("id", "")).strip(): str(
                unit_spec.get("type", "")
            ).strip()
            for unit_spec in unit_specs
            if isinstance(unit_spec, dict)
        }
        connected_outputs: set[tuple[str, str]] = set()
        for connection in connections:
            source = connection["source"]
            if str(source.get("kind", "")).strip().lower() != "unit":
                continue
            source_id = str(source.get("id", "")).strip()
            connected_outputs.add(
                (
                    source_id,
                    canonical_material_output_port(
                        source.get("port", ""),
                        unit_types.get(source_id),
                    ),
                )
            )

        StreamClass = jneqsim.process.equipment.stream.Stream
        terminal_streams: Dict[str, Any] = {}
        used_name_keys = {
            str(name).strip().casefold()
            for name in reserved_names
            if str(name).strip()
        }
        for unit_spec in unit_specs:
            unit_id = str(unit_spec["id"]).strip()
            unit_name = str(unit_spec["name"]).strip()
            ports = unit_spec.get("ports")
            if not isinstance(ports, dict):
                raise ValueError(f"Unit '{unit_id}' requires a ports object.")
            material_outputs = ports.get("material_out")
            if not isinstance(material_outputs, list):
                raise ValueError(
                    f"Unit '{unit_id}' requires a material_out ports array."
                )
            for raw_port in material_outputs:
                output_port = str(raw_port).strip().lower()
                canonical_output_port = (
                    canonical_material_output_port(
                        output_port,
                        unit_spec.get("type"),
                    )
                )
                if not output_port:
                    raise ValueError(
                        f"Unit '{unit_id}' has an empty material output port."
                    )
                endpoint_key = (unit_id, canonical_output_port)
                if endpoint_key in connected_outputs:
                    continue

                boundary_name = f"{unit_name} [{output_port}] product"
                boundary_name_key = boundary_name.casefold()
                if boundary_name_key in used_name_keys:
                    raise ValueError(
                        f"Terminal stream name '{boundary_name}' is duplicated."
                    )
                source_stream = self.resolve_material_output(
                    {
                        "kind": "unit",
                        "id": unit_id,
                        "port": output_port,
                    },
                    inlet_streams,
                    unit_objects,
                )
                try:
                    terminal_stream = StreamClass(boundary_name, source_stream)
                    process_system.add(terminal_stream)
                except Exception as exc:
                    raise ValueError(
                        f"Could not create terminal stream for unit '{unit_id}' "
                        f"port '{output_port}'."
                    ) from exc

                boundary_id = f"{unit_id}:{output_port}"
                terminal_streams[boundary_id] = terminal_stream
                used_name_keys.add(boundary_name_key)
                self._build_log.append(
                    f"Added terminal product stream: {boundary_id}"
                )

        if not terminal_streams:
            raise ValueError(
                "Acyclic graph requires at least one unconnected material "
                "output port."
            )
        return terminal_streams

    def _add_material_connection_stream(
        self,
        connection: dict,
        inlet_streams: Dict[str, Any],
        unit_objects: Dict[str, Any],
        process_system: Any,
        connection_streams: Dict[str, Any],
    ) -> Any:
        """Materialize one named stream between explicit graph ports."""
        from neqsim import jneqsim

        connection_id = str(connection.get("id", "")).strip()
        if connection_id in connection_streams:
            raise ValueError(
                f"Material connection '{connection_id}' was materialized twice."
            )
        source_stream = self.resolve_material_output(
            connection["source"],
            inlet_streams,
            unit_objects,
        )
        stream_name = material_connection_name(connection)
        StreamClass = jneqsim.process.equipment.stream.Stream
        try:
            material_stream = StreamClass(stream_name, source_stream)
            process_system.add(material_stream)
        except Exception as exc:
            raise ValueError(
                f"Could not create native material stream '{stream_name}' "
                f"for connection '{connection_id}'."
            ) from exc
        connection_streams[connection_id] = material_stream
        self._build_log.append(
            f"Added material stream: {connection_id} ({stream_name})"
        )
        return material_stream

    def build_acyclic_graph(
        self,
        graph_spec: dict,
        inlet_specs: List[dict],
        execution_order: List[str],
    ) -> NeqSimProcessModel:
        """Build and solve a validated acyclic material-flow graph.

        The graph specification contains unit nodes and explicit material
        connections; inlet_specs contains ProcessBuilder-compatible independent
        fluids. execution_order must list every inlet and unit once in dependency
        order. Mixers may combine multiple upstream material streams. Energy
        links and recycles remain explicit later solver stages and are rejected.
        """
        from neqsim import jneqsim

        if not isinstance(graph_spec, dict):
            raise ValueError("Graph specification must be an object.")
        if not isinstance(inlet_specs, list) or not inlet_specs:
            raise ValueError("Acyclic graph execution requires inlet specifications.")
        if not isinstance(execution_order, list) or not execution_order:
            raise ValueError("Acyclic graph execution requires an execution order.")

        unit_specs = graph_spec.get("units")
        connections = graph_spec.get("connections")
        if not isinstance(unit_specs, list):
            raise ValueError("Graph specification requires a units array.")
        if not isinstance(connections, list):
            raise ValueError("Graph specification requires a connections array.")

        inlet_ids: list[str] = []
        inlet_names: set[str] = set()
        for inlet_index, inlet_spec in enumerate(inlet_specs):
            if not isinstance(inlet_spec, dict):
                raise ValueError(
                    f"Inlet specification {inlet_index} must be an object."
                )
            inlet_id = str(inlet_spec.get("inlet_id", "")).strip()
            inlet_name = str(inlet_spec.get("name", "")).strip()
            if not inlet_id or not inlet_name:
                raise ValueError(
                    f"Inlet specification {inlet_index} requires inlet_id and name."
                )
            if inlet_id in inlet_ids:
                raise ValueError(f"Inlet id '{inlet_id}' is duplicated.")
            if inlet_name in inlet_names:
                raise ValueError(f"Inlet stream name '{inlet_name}' is duplicated.")
            inlet_ids.append(inlet_id)
            inlet_names.add(inlet_name)

        indexed_units: Dict[str, dict] = {}
        unit_names: set[str] = set()
        for unit_index, unit_spec in enumerate(unit_specs):
            if not isinstance(unit_spec, dict):
                raise ValueError(f"Unit specification {unit_index} must be an object.")
            unit_id = str(unit_spec.get("id", "")).strip()
            unit_name = str(unit_spec.get("name", "")).strip()
            unit_type = str(unit_spec.get("type", "")).strip().lower()
            if not unit_id or not unit_name or not unit_type:
                raise ValueError(
                    f"Unit specification {unit_index} requires id, name, and type."
                )
            if unit_id in indexed_units or unit_id in inlet_ids:
                raise ValueError(f"Graph object id '{unit_id}' is duplicated.")
            if unit_name in unit_names or unit_name in inlet_names:
                raise ValueError(f"Process object name '{unit_name}' is duplicated.")
            params = unit_spec.get("params", {})
            if not isinstance(params, dict):
                raise ValueError(f"Unit '{unit_id}' params must be an object.")
            indexed_units[unit_id] = unit_spec
            unit_names.add(unit_name)
            ports = unit_spec.get("ports")
            material_outputs = (
                ports.get("material_out")
                if isinstance(ports, dict)
                else None
            )
            if isinstance(material_outputs, list):
                canonical_outputs = [
                    canonical_material_output_port(port, unit_type)
                    for port in material_outputs
                ]
                if (
                    unit_type == "heat_exchanger"
                    and canonical_outputs
                    != ["hot_out", "cold_out"]
                ):
                    raise ValueError(
                        f"Heat exchanger '{unit_id}' requires declared "
                        "material output ports in fixed order: hot_out, "
                        "cold_out."
                    )
                if len(canonical_outputs) != len(set(canonical_outputs)):
                    raise ValueError(
                        f"Unit '{unit_id}' material output ports alias the "
                        "same native outlet."
                    )

        requested_equipment_design_bases = (
            self._requested_equipment_design_bases(
                unit_specs
            )
        )
        # Graph construction normalizes native unit names at its schema
        # boundary. Keep adapter metadata keyed by that exact native name.
        equipment_design_bases = {
            unit_name.strip(): dict(design_basis)
            for unit_name, design_basis in (
                requested_equipment_design_bases.items()
            )
        }

        expected_ids = [*inlet_ids, *indexed_units]
        ordered_ids = [str(node_id).strip() for node_id in execution_order]
        if any(not node_id for node_id in ordered_ids):
            raise ValueError("Execution order cannot contain an empty object id.")
        if len(ordered_ids) != len(set(ordered_ids)):
            raise ValueError("Execution order object ids must be unique.")
        if set(ordered_ids) != set(expected_ids):
            missing = sorted(set(expected_ids).difference(ordered_ids))
            unexpected = sorted(set(ordered_ids).difference(expected_ids))
            details = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected: {', '.join(unexpected)}")
            raise ValueError(
                "Execution order must contain every graph object once ("
                + "; ".join(details)
                + ")."
            )

        incoming_material: Dict[str, list[dict]] = {
            unit_id: [] for unit_id in indexed_units
        }
        connection_ids: set[str] = set()
        connection_name_keys: set[str] = set()
        connection_names: set[str] = set()
        connected_source_ports: set[tuple[str, str, str]] = set()
        reserved_name_keys = {
            name.casefold() for name in inlet_names.union(unit_names)
        }
        for connection_index, connection in enumerate(connections):
            if not isinstance(connection, dict):
                raise ValueError(
                    f"Connection specification {connection_index} must be an object."
                )
            connection_type = str(connection.get("type", "")).strip().lower()
            connection_id = str(connection.get("id", "")).strip()
            if not connection_id:
                raise ValueError(f"Connection {connection_index} requires an id.")
            if connection_id in connection_ids:
                raise ValueError(
                    f"Connection id '{connection_id}' is duplicated."
                )
            connection_ids.add(connection_id)
            if connection_type != "material":
                raise ValueError(
                    f"Connection '{connection_id}' is not a material connection. "
                    "Energy links require a later executor stage."
                )
            connection_name = material_connection_name(connection)
            connection_name_key = connection_name.casefold()
            if connection_name_key in connection_name_keys:
                raise ValueError(
                    f"Material stream name '{connection_name}' is duplicated."
                )
            if connection_name_key in reserved_name_keys:
                raise ValueError(
                    f"Material stream name '{connection_name}' conflicts with "
                    "an inlet or equipment name."
                )
            connection_name_keys.add(connection_name_key)
            connection_names.add(connection_name)
            source = connection.get("source")
            target = connection.get("target")
            if not isinstance(source, dict) or not isinstance(target, dict):
                raise ValueError(
                    f"Connection '{connection_id}' requires source and target objects."
                )
            source_kind = str(source.get("kind", "")).strip().lower()
            source_id = str(source.get("id", "")).strip()
            source_unit_type = (
                indexed_units[source_id].get("type")
                if source_kind == "unit" and source_id in indexed_units
                else None
            )
            canonical_source_port = canonical_material_output_port(
                source.get("port", ""),
                source_unit_type,
            )
            if source_kind == "unit" and source_id in indexed_units:
                source_ports = indexed_units[source_id].get("ports")
                declared_outputs = (
                    source_ports.get("material_out")
                    if isinstance(source_ports, dict)
                    else None
                )
                canonical_declared_outputs = (
                    {
                        canonical_material_output_port(
                            port,
                            source_unit_type,
                        )
                        for port in declared_outputs
                    }
                    if isinstance(declared_outputs, list)
                    else set()
                )
                if canonical_source_port not in canonical_declared_outputs:
                    raise ValueError(
                        f"Connection '{connection_id}' uses undeclared "
                        f"material output port '{source.get('port', '')}' "
                        f"on unit '{source_id}'."
                    )
            source_key = (
                source_kind,
                source_id,
                canonical_source_port,
            )
            if source_key in connected_source_ports:
                raise ValueError(
                    f"Material output port {source_key[1]}:{source_key[2]} "
                    "already has a connection; use a splitter for branching."
                )
            connected_source_ports.add(source_key)
            target_kind = str(target.get("kind", "")).strip().lower()
            target_id = str(target.get("id", "")).strip()
            if target_kind != "unit" or target_id not in indexed_units:
                raise ValueError(
                    f"Connection '{connection_id}' requires a known unit target."
                )
            incoming_material[target_id].append(connection)

        connected_output_ports = {
            (
                source_id,
                canonical_material_output_port(
                    connection["source"].get("port", ""),
                    indexed_units.get(source_id, {}).get("type"),
                ),
            )
            for connection in connections
            if str(connection["source"].get("kind", "")).strip().lower()
            == "unit"
            for source_id in [
                str(connection["source"].get("id", "")).strip()
            ]
        }
        terminal_boundary_name_keys: set[str] = set()
        for unit_id, unit_spec in indexed_units.items():
            ports = unit_spec.get("ports")
            material_outputs = (
                ports.get("material_out")
                if isinstance(ports, dict)
                else []
            )
            for raw_port in material_outputs:
                output_port = str(raw_port).strip().lower()
                canonical_output_port = (
                    canonical_material_output_port(
                        output_port,
                        unit_spec.get("type"),
                    )
                )
                if (
                    not output_port
                    or (unit_id, canonical_output_port)
                    in connected_output_ports
                ):
                    continue
                boundary_name = (
                    f"{str(unit_spec['name']).strip()} "
                    f"[{output_port}] product"
                )
                boundary_name_key = boundary_name.casefold()
                if boundary_name_key in connection_name_keys:
                    raise ValueError(
                        f"Material stream name '{boundary_name}' conflicts "
                        "with a terminal product boundary."
                    )
                if boundary_name_key in reserved_name_keys:
                    raise ValueError(
                        f"Terminal product stream name '{boundary_name}' "
                        "conflicts with an inlet or equipment name."
                    )
                if boundary_name_key in terminal_boundary_name_keys:
                    raise ValueError(
                        f"Terminal product stream name '{boundary_name}' "
                        "is duplicated."
                    )
                terminal_boundary_name_keys.add(boundary_name_key)

        process_name = str(graph_spec.get("name", "Graph Process")).strip()
        self._process_name = process_name or "Graph Process"
        self._spec = {
            "name": self._process_name,
            "graph": graph_spec,
            "inlet_specs": inlet_specs,
            "execution_order": list(ordered_ids),
        }
        self._build_log.clear()

        inlet_streams = self.create_inlet_streams(inlet_specs)
        ProcessSystem = jneqsim.process.processmodel.ProcessSystem
        process_system = ProcessSystem()
        unit_objects: Dict[str, Any] = {}
        connection_streams: Dict[str, Any] = {}

        for node_id in ordered_ids:
            if node_id in inlet_streams:
                process_system.add(inlet_streams[node_id])
                self._build_log.append(f"Added inlet stream: {node_id}")
                continue

            unit_spec = indexed_units[node_id]
            unit_type = str(unit_spec["type"]).strip().lower()
            incoming = sorted(
                incoming_material[node_id],
                key=lambda connection: (
                    str(connection["target"].get("port", "")).strip(),
                    str(connection["id"]).strip(),
                ),
            )
            if unit_type in {"mixer", "separator", "heat_exchanger"}:
                ports = unit_spec.get("ports")
                declared_inputs = (
                    ports.get("material_in")
                    if isinstance(ports, dict)
                    else None
                )
                if not isinstance(declared_inputs, list) or not declared_inputs:
                    raise ValueError(
                        f"{unit_type.capitalize()} '{node_id}' requires "
                        "declared material inlet ports."
                    )
                normalized_declared_inputs = [
                    str(port).strip()
                    for port in declared_inputs
                ]
                if (
                    any(not port for port in normalized_declared_inputs)
                    or len(set(normalized_declared_inputs))
                    != len(normalized_declared_inputs)
                ):
                    raise ValueError(
                        f"{unit_type.capitalize()} '{node_id}' requires "
                        "unique non-empty declared material inlet ports."
                    )
                if (
                    unit_type == "heat_exchanger"
                    and normalized_declared_inputs
                    != ["hot_in", "cold_in"]
                ):
                    raise ValueError(
                        f"Heat exchanger '{node_id}' requires declared "
                        "material inlet ports in fixed order: hot_in, "
                        "cold_in."
                    )
                connected_inputs = [
                    str(connection["target"].get("port", "")).strip()
                    for connection in incoming
                ]
                missing_inputs = sorted(
                    set(normalized_declared_inputs).difference(
                        connected_inputs
                    )
                )
                unexpected_inputs = sorted(
                    set(connected_inputs).difference(
                        normalized_declared_inputs
                    )
                )
                if (
                    len(connected_inputs) != len(normalized_declared_inputs)
                    or missing_inputs
                    or unexpected_inputs
                ):
                    details = []
                    if missing_inputs:
                        details.append(
                            "missing: " + ", ".join(missing_inputs)
                        )
                    if unexpected_inputs:
                        details.append(
                            "unexpected: " + ", ".join(unexpected_inputs)
                        )
                    if not details:
                        details.append(
                            "declared "
                            f"{len(normalized_declared_inputs)}, connected "
                            f"{len(connected_inputs)}"
                        )
                    raise ValueError(
                        f"{unit_type.capitalize()} '{node_id}' material inlet "
                        "connections must match declared ports ("
                        + "; ".join(details)
                        + ")."
                    )
                if unit_type == "heat_exchanger":
                    incoming_by_port = {
                        str(connection["target"].get("port", "")).strip(): (
                            connection
                        )
                        for connection in incoming
                    }
                    incoming = [
                        incoming_by_port[port]
                        for port in normalized_declared_inputs
                    ]
            if not incoming:
                raise ValueError(
                    f"Unit '{node_id}' requires at least one material inlet."
                )

            if unit_type in {"mixer", "heat_exchanger"} or (
                unit_type == "separator" and len(incoming) > 1
            ):
                minimum_inlets = (
                    2
                    if unit_type in {"mixer", "heat_exchanger"}
                    else 1
                )
                if len(incoming) < minimum_inlets:
                    raise ValueError(
                        f"{unit_type.capitalize()} '{node_id}' requires at "
                        f"least {minimum_inlets} material inlet"
                        f"{'s' if minimum_inlets != 1 else ''}."
                    )
                source_streams = [
                    self._add_material_connection_stream(
                        connection,
                        inlet_streams,
                        unit_objects,
                        process_system,
                        connection_streams,
                    )
                    for connection in incoming
                ]
                unit = self._create_unit(
                    str(unit_spec["name"]).strip(),
                    unit_type,
                    source_streams[0],
                    dict(unit_spec.get("params", {})),
                )
                for connection, source_stream in zip(
                    incoming[1:],
                    source_streams[1:],
                ):
                    try:
                        if unit_type == "heat_exchanger":
                            unit.addInStream(source_stream)
                        else:
                            unit.addStream(source_stream)
                    except Exception as exc:
                        connection_id = str(connection["id"]).strip()
                        raise ValueError(
                            f"{unit_type.capitalize()} '{node_id}' could not "
                            f"add material connection '{connection_id}'."
                        ) from exc
                self._build_log.append(
                    f"Added graph {unit_type}: {node_id} "
                    f"({len(source_streams)} material inlets)"
                )
            else:
                if len(incoming) != 1:
                    raise ValueError(
                        f"Unit '{node_id}' requires exactly one material inlet; "
                        f"found {len(incoming)}."
                    )
                source_stream = self._add_material_connection_stream(
                    incoming[0],
                    inlet_streams,
                    unit_objects,
                    process_system,
                    connection_streams,
                )
                unit = self._create_unit(
                    str(unit_spec["name"]).strip(),
                    unit_type,
                    source_stream,
                    dict(unit_spec.get("params", {})),
                )
                self._build_log.append(
                    f"Added graph unit: {node_id} ({unit_type})"
                )

            if unit_type == "splitter":
                split_factors = self._configure_graph_splitter(
                    unit,
                    node_id,
                    unit_spec,
                )
                factor_summary = ", ".join(
                    f"out_{index}={factor:.6f}"
                    for index, factor in enumerate(split_factors)
                )
                self._build_log.append(
                    f"Configured graph splitter: {node_id} ({factor_summary})"
                )

            process_system.add(unit)
            unit_objects[node_id] = unit

        self._add_terminal_material_streams(
            unit_specs,
            connections,
            inlet_streams,
            unit_objects,
            process_system,
            inlet_names.union(unit_names).union(connection_names),
        )
        self._build_log.append("Running acyclic graph simulation...")
        process_run_succeeded = NeqSimProcessModel._run_until_converged(
            process_system
        )
        if not process_run_succeeded:
            raise RuntimeError(
                "Acyclic graph simulation did not complete successfully."
            )
        direct_closure_ran = (
            NeqSimProcessModel._run_acyclic_mixer_energy_closure(
                process_system
            )
        )
        mapped_units = self._apply_requested_compressor_charts(
            unit_specs,
            unit_objects,
        )
        if mapped_units:
            self._build_log.append(
                "Running closed compressor-map rerun for: "
                + ", ".join(mapped_units)
            )
            process_run_succeeded = NeqSimProcessModel._run_until_converged(
                process_system
            )
            if not process_run_succeeded:
                raise RuntimeError(
                    "Acyclic graph compressor-map rerun did not complete "
                    "successfully."
                )
            post_map_closure_ran = (
                NeqSimProcessModel._run_acyclic_mixer_energy_closure(
                    process_system
                )
            )
            direct_closure_ran = direct_closure_ran or post_map_closure_ran
        designed_units = self._apply_requested_mechanical_designs(
            unit_specs,
            unit_objects,
        )
        if designed_units:
            if direct_closure_ran:
                self._build_log.append(
                    "Closed acyclic mixer energy balance before mechanical "
                    "design."
                )
            self._build_log.append(
                "Running closed design rerun for: "
                + ", ".join(designed_units)
            )
            process_run_succeeded = NeqSimProcessModel._run_until_converged(
                process_system
            )
            if not process_run_succeeded:
                raise RuntimeError(
                    "Acyclic graph design rerun did not complete "
                    "successfully."
                )
            post_design_closure_ran = (
                NeqSimProcessModel._run_acyclic_mixer_energy_closure(
                    process_system
                )
            )
            direct_closure_ran = (
                direct_closure_ran or post_design_closure_ran
            )
            if post_design_closure_ran:
                self._build_log.append(
                    "Closed acyclic mixer energy balance after mechanical "
                    "design rerun."
                )
        self._model = NeqSimProcessModel.from_process_system(
            process_system,
            enforce_acyclic_mixer_energy=True,
            trusted_solved=True,
            allow_direct_runs=direct_closure_ran,
            equipment_design_bases=equipment_design_bases,
        )
        if equipment_design_bases:
            self._build_log.append(
                "Registered equipment design basis for: "
                + ", ".join(equipment_design_bases)
            )
        self._build_log.append("Acyclic graph built and converged successfully.")
        return self._model

    @staticmethod
    def _compressor_chart_settings(
        params: dict,
    ) -> tuple[bool, str, int]:
        """Validate one backward-compatible native compressor-map request."""
        raw_enabled = params.get("use_compressor_chart", False)
        if type(raw_enabled) is bool:
            enabled = raw_enabled
        elif isinstance(raw_enabled, str) and raw_enabled.strip().lower() in {
            "true",
            "yes",
            "1",
            "false",
            "no",
            "0",
        }:
            enabled = _is_truthy(raw_enabled)
        else:
            raise ValueError(
                "Compressor use_compressor_chart must be boolean."
            )

        raw_template = params.get(
            "chart_template",
            "CENTRIFUGAL_STANDARD",
        )
        if not isinstance(raw_template, str):
            raise ValueError("Compressor chart_template must be a string.")
        template = raw_template.strip().upper()
        if template not in _COMPRESSOR_CHART_TEMPLATES:
            raise ValueError(
                "Compressor chart_template must be one of: "
                + ", ".join(_COMPRESSOR_CHART_TEMPLATES)
                + "."
            )

        raw_num_speeds = params.get("chart_num_speeds", 5)
        if isinstance(raw_num_speeds, bool):
            raise ValueError("Compressor chart_num_speeds must be an integer.")
        try:
            numeric_num_speeds = float(raw_num_speeds)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Compressor chart_num_speeds must be an integer."
            ) from exc
        if (
            not math.isfinite(numeric_num_speeds)
            or not numeric_num_speeds.is_integer()
        ):
            raise ValueError("Compressor chart_num_speeds must be an integer.")
        num_speeds = int(numeric_num_speeds)
        if not 3 <= num_speeds <= 12:
            raise ValueError(
                "Compressor chart_num_speeds must be between 3 and 12."
            )
        return enabled, template, num_speeds

    @classmethod
    def _apply_requested_compressor_charts(
        cls,
        unit_specs: List[dict],
        unit_objects: Dict[str, Any],
    ) -> List[str]:
        """Generate requested native maps from converged design points."""
        from neqsim import jneqsim

        mapped_units: List[str] = []
        for unit_spec in unit_specs:
            if not isinstance(unit_spec, dict):
                continue
            params = unit_spec.get("params", {})
            if not isinstance(params, dict):
                continue
            has_map_request = any(
                key in params
                for key in (
                    "use_compressor_chart",
                    "chart_template",
                    "chart_num_speeds",
                )
            )
            if not has_map_request:
                continue
            unit_type = str(unit_spec.get("type", "")).strip().lower()
            if unit_type != "compressor":
                raise ValueError(
                    "Native compressor maps are supported only for "
                    "compressor units."
                )
            enabled, template, num_speeds = cls._compressor_chart_settings(
                params
            )
            if not enabled:
                continue
            unit_id = str(
                unit_spec.get("id", unit_spec.get("name", ""))
            ).strip()
            unit = unit_objects.get(unit_id)
            if unit is None:
                raise ValueError(
                    f"Compressor map target '{unit_id}' was not built."
                )
            try:
                generator = (
                    jneqsim.process.equipment.compressor
                    .CompressorChartGenerator(unit)
                )
                chart = generator.generateFromTemplate(
                    template,
                    num_speeds,
                )
                unit.setCompressorChartType("interpolate and extrapolate")
                unit.setCompressorChart(chart)
                unit.getCompressorChart().setHeadUnit("kJ/kg")
                unit.setSolveSpeed(True)
                unit.setUsePolytropicCalc(True)
            except Exception as exc:
                raise RuntimeError(
                    f"Compressor '{unit_id}' native map construction failed."
                ) from exc
            mapped_units.append(unit_id)
        return mapped_units

    @staticmethod
    def _pump_design_settings(
        params: dict,
    ) -> tuple[bool, float, float, float]:
        """Validate one backward-compatible pump design-limit request."""
        raw_enabled = params.get("use_design_basis", False)
        if type(raw_enabled) is not bool:
            raise ValueError("Pump use_design_basis must be boolean.")

        definitions = (
            (
                "design_flow_capacity_m3_per_hr",
                100.0,
                0.001,
                1_000_000.0,
                "m3/hr",
            ),
            (
                "design_head_capacity_m",
                600.0,
                0.1,
                20_000.0,
                "m",
            ),
            (
                "motor_rating_kw",
                100.0,
                0.001,
                1_000_000.0,
                "kW",
            ),
        )
        values: list[float] = []
        for key, default, minimum, maximum, unit in definitions:
            raw_value = params.get(key, default)
            if isinstance(raw_value, bool):
                raise ValueError(f"Pump {key} must be numeric.")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Pump {key} must be numeric.") from exc
            if not math.isfinite(value):
                raise ValueError(f"Pump {key} must be finite.")
            if not minimum <= value <= maximum:
                raise ValueError(
                    f"Pump {key} must be between {minimum} and {maximum} "
                    f"{unit}."
                )
            values.append(value)
        return raw_enabled, values[0], values[1], values[2]

    @classmethod
    def _requested_pump_design_bases(
        cls,
        unit_specs: List[dict],
    ) -> Dict[str, Dict[str, float]]:
        """Return validated opt-in pump design capacities by unit name."""
        design_bases: Dict[str, Dict[str, float]] = {}
        design_keys = (
            "use_design_basis",
            "design_flow_capacity_m3_per_hr",
            "design_head_capacity_m",
            "motor_rating_kw",
        )
        for unit_spec in unit_specs:
            if not isinstance(unit_spec, dict):
                continue
            params = unit_spec.get("params", {})
            if not isinstance(params, dict):
                continue
            unit_type = str(unit_spec.get("type", "")).strip().lower()
            if unit_type != "pump":
                has_pump_specific_key = any(
                    key in params for key in design_keys[1:]
                )
                if has_pump_specific_key or (
                    unit_type != "heat_exchanger"
                    and "use_design_basis" in params
                ):
                    raise ValueError(
                        "Pump design-basis properties are supported only for "
                        "pump units."
                    )
                continue
            if not any(key in params for key in design_keys):
                continue
            enabled, flow, head, motor = cls._pump_design_settings(params)
            if not enabled:
                continue
            unit_name = str(unit_spec.get("name", ""))
            if not unit_name.strip():
                raise ValueError("Pump design basis requires a unit name.")
            design_bases[unit_name] = {
                "design_flow_capacity_m3_per_hr": flow,
                "design_head_capacity_m": head,
                "motor_rating_kw": motor,
            }
        return design_bases

    @staticmethod
    def _heat_exchanger_design_settings(
        params: dict,
    ) -> tuple[bool, float, float]:
        """Validate one backward-compatible exchanger capacity request."""
        raw_enabled = params.get("use_design_basis", False)
        if type(raw_enabled) is not bool:
            raise ValueError(
                "Heat exchanger use_design_basis must be boolean."
            )
        definitions = (
            (
                "design_duty_capacity_kw",
                2_500.0,
                0.001,
                100_000_000.0,
                "kW",
            ),
            (
                "design_ua_capacity_w_per_k",
                125_000.0,
                1.0,
                1_000_000_000.0,
                "W/K",
            ),
        )
        values: list[float] = []
        for key, default, minimum, maximum, unit in definitions:
            raw_value = params.get(key, default)
            if isinstance(raw_value, bool):
                raise ValueError(
                    f"Heat exchanger {key} must be numeric."
                )
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Heat exchanger {key} must be numeric."
                ) from exc
            if not math.isfinite(value):
                raise ValueError(
                    f"Heat exchanger {key} must be finite."
                )
            if not minimum <= value <= maximum:
                raise ValueError(
                    f"Heat exchanger {key} must be between {minimum} and "
                    f"{maximum} {unit}."
                )
            values.append(value)
        return raw_enabled, values[0], values[1]

    @classmethod
    def _requested_heat_exchanger_design_bases(
        cls,
        unit_specs: List[dict],
    ) -> Dict[str, Dict[str, float]]:
        """Return validated opt-in exchanger capacities by unit name."""
        design_bases: Dict[str, Dict[str, float]] = {}
        design_keys = (
            "use_design_basis",
            "design_duty_capacity_kw",
            "design_ua_capacity_w_per_k",
        )
        for unit_spec in unit_specs:
            if not isinstance(unit_spec, dict):
                continue
            params = unit_spec.get("params", {})
            if not isinstance(params, dict):
                continue
            unit_type = str(unit_spec.get("type", "")).strip().lower()
            if unit_type != "heat_exchanger":
                has_exchanger_specific_key = any(
                    key in params for key in design_keys[1:]
                )
                if has_exchanger_specific_key or (
                    unit_type != "pump"
                    and "use_design_basis" in params
                ):
                    raise ValueError(
                        "Heat-exchanger design-basis properties are supported "
                        "only for heat_exchanger units."
                    )
                continue
            if not any(key in params for key in design_keys):
                continue
            enabled, duty, ua = cls._heat_exchanger_design_settings(params)
            if not enabled:
                continue
            unit_name = str(unit_spec.get("name", ""))
            if not unit_name.strip():
                raise ValueError(
                    "Heat exchanger design basis requires a unit name."
                )
            design_bases[unit_name] = {
                "design_duty_capacity_kw": duty,
                "design_ua_capacity_w_per_k": ua,
            }
        return design_bases

    @classmethod
    def _requested_equipment_design_bases(
        cls,
        unit_specs: List[dict],
    ) -> Dict[str, Dict[str, float]]:
        """Return every validated opt-in reporting design basis."""
        design_bases = cls._requested_pump_design_bases(unit_specs)
        exchanger_bases = cls._requested_heat_exchanger_design_bases(
            unit_specs
        )
        duplicate_names = set(design_bases).intersection(exchanger_bases)
        if duplicate_names:
            raise ValueError(
                "Equipment design-basis unit names must be unique: "
                + ", ".join(sorted(duplicate_names))
                + "."
            )
        design_bases.update(exchanger_bases)
        return design_bases

    @staticmethod
    def _separator_design_settings(
        params: dict,
    ) -> tuple[bool, Optional[float]]:
        """Validate one explicit native separator-design request."""
        raw_enabled = params.get("auto_size", False)
        if type(raw_enabled) is not bool:
            raise ValueError("Separator auto_size must be boolean.")

        raw_gas_load = params.get("design_gas_load_factor_m_per_s")
        if raw_gas_load is None:
            return raw_enabled, None
        if isinstance(raw_gas_load, bool):
            raise ValueError(
                "Separator design gas-load factor must be numeric."
            )
        try:
            gas_load = float(raw_gas_load)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Separator design gas-load factor must be numeric."
            ) from exc
        if not math.isfinite(gas_load) or not 0.01 <= gas_load <= 1.0:
            raise ValueError(
                "Separator design gas-load factor must be between 0.01 "
                "and 1.0 m/s."
            )
        return raw_enabled, gas_load

    @classmethod
    def _apply_requested_mechanical_designs(
        cls,
        unit_specs: List[dict],
        unit_objects: Dict[str, Any],
    ) -> List[str]:
        """Auto-size requested separators from their converged feed states."""
        designed_units: List[str] = []
        for unit_spec in unit_specs:
            if not isinstance(unit_spec, dict):
                continue
            unit_type = str(unit_spec.get("type", "")).strip().lower()
            params = unit_spec.get("params", {})
            if not isinstance(params, dict):
                continue
            has_design_request = any(
                key in params
                for key in (
                    "auto_size",
                    "design_gas_load_factor_m_per_s",
                )
            )
            if not has_design_request:
                continue
            if unit_type != "separator":
                raise ValueError(
                    "Native mechanical sizing is currently supported only "
                    "for separator units."
                )
            enabled, gas_load = cls._separator_design_settings(params)
            if not enabled:
                continue
            unit_id = str(
                unit_spec.get("id", unit_spec.get("name", ""))
            ).strip()
            unit = unit_objects.get(unit_id)
            if unit is None:
                raise ValueError(
                    f"Separator design target '{unit_id}' was not built."
                )
            if gas_load is not None:
                if not hasattr(unit, "setDesignGasLoadFactor"):
                    raise ValueError(
                        f"Separator '{unit_id}' does not expose a native "
                        "design gas-load factor."
                    )
                unit.setDesignGasLoadFactor(gas_load)
            if not hasattr(unit, "autoSize"):
                raise ValueError(
                    f"Separator '{unit_id}' does not expose native autoSize."
                )
            unit.autoSize()
            designed_units.append(unit_id)
        return designed_units

    # -- Build from spec ----------------------------------------------------

    def build_from_spec(self, spec: dict) -> NeqSimProcessModel:
        """Build a complete process from a specification dict.

        Generic graph specs contain ``graph``, ``inlet_specs``, and
        ``execution_order`` and are delegated to :meth:`build_acyclic_graph`.
        Legacy linear specs remain supported and must contain:

          - ``fluid`` — fluid definition (components, EOS, conditions)
          - ``process`` — ordered list of unit steps

        Returns the wrapped :class:`NeqSimProcessModel`.
        """
        if not isinstance(spec, dict):
            raise ValueError("Process specification must be an object.")
        if "graph" in spec:
            graph_spec = spec.get("graph")
            raw_wrapper_name = spec.get("name")
            wrapper_name = (
                str(raw_wrapper_name).strip()
                if raw_wrapper_name is not None
                else ""
            )
            if isinstance(graph_spec, dict) and wrapper_name:
                graph_spec = {**graph_spec, "name": wrapper_name}
            return self.build_acyclic_graph(
                graph_spec,
                spec.get("inlet_specs"),
                spec.get("execution_order"),
            )

        self._spec = spec
        self._process_name = spec.get("name", "New Process")
        self._build_log.clear()

        from neqsim import jneqsim

        fluid_spec = spec.get("fluid", {})
        process_steps = spec.get("process", [])

        if not process_steps:
            raise ValueError("Process spec must contain at least one step in 'process'.")
        equipment_design_bases = self._requested_equipment_design_bases(
            process_steps
        )

        # 1. Create the thermodynamic fluid
        fluid = self.create_fluid_from_spec(fluid_spec)
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
        process_run_succeeded = NeqSimProcessModel._run_until_converged(
            proc
        )
        if not process_run_succeeded:
            raise RuntimeError(
                "Process simulation did not complete successfully."
            )
        mapped_units = self._apply_requested_compressor_charts(
            process_steps,
            built_units,
        )
        if mapped_units:
            self._build_log.append(
                "Running closed compressor-map rerun for: "
                + ", ".join(mapped_units)
            )
            process_run_succeeded = NeqSimProcessModel._run_until_converged(
                proc
            )
            if not process_run_succeeded:
                raise RuntimeError(
                    "Process compressor-map rerun did not complete "
                    "successfully."
                )
        designed_units = self._apply_requested_mechanical_designs(
            process_steps,
            built_units,
        )
        if designed_units:
            self._build_log.append(
                "Running closed design rerun for: "
                + ", ".join(designed_units)
            )
            process_run_succeeded = NeqSimProcessModel._run_until_converged(
                proc
            )
            if not process_run_succeeded:
                raise RuntimeError(
                    "Process design rerun did not complete successfully."
                )

        # 4. Wrap in NeqSimProcessModel
        self._model = NeqSimProcessModel.from_process_system(
            proc,
            trusted_solved=True,
            equipment_design_bases=equipment_design_bases,
        )
        if equipment_design_bases:
            self._build_log.append(
                "Registered equipment design basis for: "
                + ", ".join(equipment_design_bases)
            )
        self._build_log.append("Process built and converged successfully.")
        return self._model

    # -- Python script export -----------------------------------------------

    def to_python_script(self) -> str:
        """Generate a Python script that reproduces this process."""
        if self._spec is None:
            return "# No process specification available.\n"
        if "graph" in self._spec:
            return self._graph_python_script()

        lines: List[str] = []
        fluid_spec = self._spec.get("fluid", {})
        process_steps = self._spec.get("process", [])
        equipment_design_bases = self._requested_equipment_design_bases(
            process_steps
        )

        # --- Header ---
        lines.append('"""')
        lines.append(f"NeqSim Process: {self._process_name}")
        lines.append("Auto-generated by NeqSim Process Chat")
        lines.append('"""')
        lines.append("from neqsim import jneqsim")
        lines.append("import neqsim")
        if equipment_design_bases:
            lines.append("import json")
            lines.append(
                "from process_chat.process_model import NeqSimProcessModel"
            )
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
        if eos == "gerg2008":
            lines.append("fluid.setMultiPhaseCheck(False)  # GERG-2008 does not support multi-phase check")
        lines.append("")

        # --- Process ---
        lines.append("# ── Build process ──")
        lines.append("process = jneqsim.process.processmodel.ProcessSystem()")
        lines.append("")

        var_names: Dict[str, str] = {}      # unit name → Python variable
        separator_designs: List[tuple[str, Optional[float]]] = []
        compressor_charts: List[tuple[str, str, int]] = []
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

            if eq_type == "pipeline":
                lines.append(
                    f"{var}.setHeatTransferMode("
                    "jneqsim.process.equipment.pipeline.PipeBeggsAndBrills."
                    "HeatTransferMode.ADIABATIC)"
                )

            # Parameters
            for pkey, pval in params.items():
                normalized_key = pkey.lower().strip()
                if (
                    normalized_key == "efficiency"
                    and eq_type in ("pump", "esp_pump")
                ):
                    setter_call = (
                        f"setIsentropicEfficiency({float(pval)})"
                    )
                    lines.append(f"{var}.{setter_call}")
                    continue
                setter_fn = _PARAM_SETTERS.get(normalized_key)
                if setter_fn:
                    setter_call = setter_fn(pval)
                    if setter_call is not None:
                        lines.append(f"{var}.{setter_call}")

            if eq_type == "separator":
                auto_size, gas_load = self._separator_design_settings(params)
                if auto_size:
                    separator_designs.append((var, gas_load))
            if eq_type == "compressor":
                use_chart, chart_template, chart_num_speeds = (
                    self._compressor_chart_settings(params)
                )
                if use_chart:
                    compressor_charts.append(
                        (var, chart_template, chart_num_speeds)
                    )

            lines.append(f"process.add({var})")
            prev_var = var
            prev_type = eq_type
            prev_outlet = outlet_type
            lines.append("")

        # --- Run & save ---
        lines.append("# ── Run process ──")
        lines.append("process.run()")
        if compressor_charts:
            lines.append("")
            lines.append(
                "# ── Generate native compressor maps and close process ──"
            )
            for var, chart_template, chart_num_speeds in compressor_charts:
                generator_var = f"{var}_chart_generator"
                chart_var = f"{var}_chart"
                lines.append(
                    f"{generator_var} = jneqsim.process.equipment.compressor."
                    f"CompressorChartGenerator({var})"
                )
                lines.append(
                    f"{chart_var} = {generator_var}.generateFromTemplate("
                    f"{chart_template!r}, {chart_num_speeds})"
                )
                lines.append(
                    f"{var}.setCompressorChartType("
                    "'interpolate and extrapolate')"
                )
                lines.append(f"{var}.setCompressorChart({chart_var})")
                lines.append(
                    f"{var}.getCompressorChart().setHeadUnit('kJ/kg')"
                )
                lines.append(f"{var}.setSolveSpeed(True)")
                lines.append(f"{var}.setUsePolytropicCalc(True)")
            lines.append("process.run()")
        if separator_designs:
            lines.append("")
            lines.append(
                "# ── Apply native mechanical design and close process ──"
            )
            for var, gas_load in separator_designs:
                if gas_load is not None:
                    lines.append(
                        f"{var}.setDesignGasLoadFactor({gas_load!r})"
                    )
                lines.append(f"{var}.autoSize()")
            lines.append("process.run()")
        if equipment_design_bases:
            lines.append("")
            lines.append(
                "# ── Studio equipment design basis (reporting metadata) ──"
            )
            lines.append(
                "equipment_design_bases = "
                + json.dumps(
                    equipment_design_bases,
                    allow_nan=False,
                    indent=4,
                    sort_keys=True,
                )
            )
            lines.extend(
                [
                    "model = NeqSimProcessModel.from_process_system(",
                    "    process,",
                    "    trusted_solved=True,",
                    "    equipment_design_bases=equipment_design_bases,",
                    ")",
                    "result = model.run(timeout_ms=180_000)",
                ]
            )
        lines.append("")
        safe = _safe_filename(self._process_name)
        lines.append("# ── Save to file ──")
        if equipment_design_bases:
            lines.extend(
                [
                    f"with open('{safe}.neqsim', 'wb') as model_file:",
                    "    model_file.write(model.save_bytes())",
                ]
            )
            try:
                serialized_case = json.dumps(
                    self._spec,
                    allow_nan=False,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Equipment design replay requires finite JSON-compatible "
                    "case data."
                ) from exc
            lines.extend(
                [
                    "# Keep the complete editable case with the native model.",
                    f"case_data = json.loads({serialized_case!r})",
                    (
                        f"with open('{safe}.case.json', 'w', "
                        "encoding='utf-8') as case_file:"
                    ),
                    "    json.dump(",
                    "        case_data,",
                    "        case_file,",
                    "        allow_nan=False,",
                    "        ensure_ascii=False,",
                    "        indent=2,",
                    "        sort_keys=True,",
                    "    )",
                ]
            )
        else:
            lines.append(f"neqsim.save_neqsim(process, '{safe}.neqsim')")
        lines.append("")
        lines.append('print("Process simulation complete!")')
        lines.append("")

        return "\n".join(lines)

    def _graph_python_script(self) -> str:
        """Generate a replayable script for the generic acyclic graph schema."""
        graph_spec = self._spec.get("graph")
        inlet_specs = self._spec.get("inlet_specs")
        execution_order = self._spec.get("execution_order")
        if not isinstance(graph_spec, dict):
            raise ValueError("Graph script export requires a graph object.")
        if not isinstance(inlet_specs, list) or not inlet_specs:
            raise ValueError(
                "Graph script export requires a non-empty inlet_specs array."
            )
        if not isinstance(execution_order, list) or not execution_order:
            raise ValueError(
                "Graph script export requires a non-empty execution_order array."
            )

        case_payload = {
            "name": str(self._spec.get("name", self._process_name)),
            "graph": graph_spec,
            "inlet_specs": inlet_specs,
            "execution_order": execution_order,
        }
        try:
            serialized_case = json.dumps(
                case_payload,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Graph script export requires finite JSON-compatible case data."
            ) from exc

        safe = _safe_filename(self._process_name)
        display_name = json.dumps(str(self._process_name), ensure_ascii=True)
        lines = [
            f"# NeqSim Process: {display_name}",
            "# Auto-generated from the generic Process Flowsheet Studio graph.",
            "# Run from this repository checkout with neqsim installed.",
            "import json",
            "",
            "from process_chat.process_builder import ProcessBuilder",
            "",
            f"case_data = json.loads({serialized_case!r})",
            "builder = ProcessBuilder()",
            "model = builder.build_acyclic_graph(",
            '    case_data["graph"],',
            '    case_data["inlet_specs"],',
            '    case_data["execution_order"],',
            ")",
            "result = model.run(timeout_ms=180_000)",
            "process = model.get_process()",
            "",
            "material_balance_applicable = result.raw.get(",
            '    "material_balance_applicable",',
            "    True,",
            ") is not False",
            "component_balance_applicable = result.raw.get(",
            '    "component_balance_applicable",',
            "    True,",
            ") is not False",
            "energy_balance_applicable = result.raw.get(",
            '    "energy_balance_applicable",',
            "    True,",
            ") is not False",
            "required_kpis = [",
            '    "material_feed_count",',
            '    "material_feed_flow_kg_hr",',
            '    "material_product_count",',
            '    "material_product_flow_kg_hr",',
            "]",
            "if material_balance_applicable:",
            '    required_kpis.append("mass_balance_pct")',
            "if component_balance_applicable:",
            '    required_kpis.append("component_balance_max_pct")',
            "if energy_balance_applicable:",
            '    required_kpis.append("energy_balance_pct")',
            "missing_kpis = [",
            "    name for name in required_kpis if name not in result.kpis",
            "]",
            "if missing_kpis:",
            "    raise RuntimeError(",
            '        "Graph replay did not produce required KPI(s): "',
            '        + ", ".join(missing_kpis)',
            "    )",
            "validation_names = []",
            "if material_balance_applicable:",
            '    validation_names.append("mass_balance")',
            "if component_balance_applicable:",
            '    validation_names.append("component_balance")',
            "if energy_balance_applicable:",
            '    validation_names.append("energy_balance")',
            "validation_status = {",
            "    constraint.name: constraint.status",
            "    for constraint in result.constraints",
            "    if constraint.name in validation_names",
            "}",
            "failed_validation = [",
            "    name",
            "    for name in validation_names",
            '    if validation_status.get(name) != "OK"',
            "]",
            "if failed_validation:",
            "    raise RuntimeError(",
            '        "Graph replay validation did not pass: "',
            '        + ", ".join(',
            '            f"{name}={validation_status.get(name, \'missing\')}"',
            "            for name in failed_validation",
            "        )",
            "    )",
            "",
            'feed_count = result.kpis["material_feed_count"].value',
            'feed_flow = result.kpis["material_feed_flow_kg_hr"].value',
            'product_count = result.kpis["material_product_count"].value',
            'product_flow = result.kpis["material_product_flow_kg_hr"].value',
            'print(f"Feed boundaries: {feed_count:.0f}")',
            'print(f"Feed flow: {feed_flow:.6f} kg/hr")',
            'print(f"Product boundaries: {product_count:.0f}")',
            'print(f"Product flow: {product_flow:.6f} kg/hr")',
            "if material_balance_applicable:",
            '    mass_residual = result.kpis["mass_balance_pct"].value',
            '    print(f"Mass imbalance: {mass_residual:.6e} %")',
            "else:",
            '    print("Mass imbalance: not applicable")',
            "if component_balance_applicable:",
            (
                '    component_residual = '
                'result.kpis["component_balance_max_pct"].value'
            ),
            (
                '    print('
                'f"Maximum component imbalance: {component_residual:.6e} %"'
                ")"
            ),
            "else:",
            '    print("Maximum component imbalance: not applicable")',
            "if energy_balance_applicable:",
            '    energy_residual = result.kpis["energy_balance_pct"].value',
            '    print(f"Energy imbalance: {energy_residual:.6e} %")',
            "else:",
            '    print("Energy imbalance: not applicable")',
            "",
            f"with open({safe + '.neqsim'!r}, 'wb') as model_file:",
            "    model_file.write(model.save_bytes())",
            'print("Process simulation complete!")',
            "",
        ]
        return "\n".join(lines)

    # -- .neqsim file export ------------------------------------------------

    def save_neqsim_bytes(self) -> Optional[bytes]:
        """Serialize the process and Studio metadata for model download."""
        if self._model is None:
            return None
        return self._model.save_bytes()

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

        if eos == "gerg2008":
            fluid.setMultiPhaseCheck(False)  # GERG-2008 does not support multi-phase check

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
            "pipeline":               lambda n, s: base.pipeline.PipeBeggsAndBrills(n, s),
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

        if eq_type == "pipeline":
            unit.setHeatTransferMode(
                base.pipeline.PipeBeggsAndBrills.HeatTransferMode.ADIABATIC
            )

        # Apply parameters
        for k, v in params.items():
            _apply_param(unit, k, v)

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
