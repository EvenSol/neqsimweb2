"""Regression tests for solved material-boundary diagnostic adapters."""

from __future__ import annotations

import math
import unittest
import weakref
from types import SimpleNamespace

from tests import test_process_graph_conservation as graph_conservation

from process_chat.process_builder import ProcessBuilder
from process_chat.process_model import (
    NeqSimProcessModel,
    _MaterialBoundaryIdentityTracker,
)
from process_chat.solver_diagnostics import (
    aggregate_convergence,
    aggregate_energy_balance,
    aggregate_material_balance,
    aggregate_unit_balances,
    aggregate_validation_status,
    component_balance_rows,
    convergence_rows,
    energy_transfer_rows,
    material_boundary_rows,
    solved_feed_flow_kg_hr,
    unit_balance_rows,
)


class _FallbackStream:
    def __init__(
        self,
        name,
        mass_flow=None,
        hash_code=None,
        class_name="Stream",
        enthalpy_flow_w=None,
    ):
        self._name = name
        self._mass_flow = mass_flow
        self._hash_code = hash_code
        self._class_name = class_name
        self._enthalpy_flow_w = enthalpy_flow_w

    def hashCode(self):
        return self._hash_code if self._hash_code is not None else id(self)

    def getClass(self):
        return _JavaClass(self._class_name)

    def getName(self):
        return self._name

    def getFlowRate(self, unit):
        if self._mass_flow is None:
            raise RuntimeError("unreadable stream")
        if unit == "kg/hr":
            return self._mass_flow
        if unit == "mol/sec":
            return self._mass_flow / 100.0
        raise ValueError(unit)

    def getTemperature(self, unit):
        if unit != "C":
            raise ValueError(unit)
        return 20.0

    def getPressure(self, unit):
        if unit != "bara":
            raise ValueError(unit)
        return 45.0

    def getFluid(self):
        if self._enthalpy_flow_w is None:
            raise RuntimeError("unreadable fluid")
        return _FallbackFluid(self._enthalpy_flow_w)


class _FallbackFluid:
    def __init__(self, enthalpy_flow_w):
        self._enthalpy_flow_w = enthalpy_flow_w

    def init(self, level):
        if level != 3:
            raise ValueError(level)

    def getEnthalpy(self):
        return self._enthalpy_flow_w


class _FallbackAliasedStream(_FallbackStream):
    def __init__(self, name, mass_flow, fluid_reference):
        super().__init__(name, mass_flow)
        self._fluid_reference = fluid_reference

    def getFluid(self):
        return self._fluid_reference


class _FallbackProcess:
    def __init__(self, units=None):
        self._units = units or []

    def getUnitOperations(self):
        return self._units


class _JavaClass:
    def __init__(self, name):
        self._name = name

    def getSimpleName(self):
        return self._name


class _FallbackEquipment:
    def getClass(self):
        return _JavaClass("Mixer")


class _FallbackHeatExchanger:
    def __init__(self, outlets):
        self._outlets = outlets

    def getClass(self):
        return _JavaClass("HeatExchanger")

    def getName(self):
        return "terminal exchanger"

    def getOutletStreams(self):
        return self._outlets

    def getOutletStream(self):
        return self._outlets[0]


class _FallbackTerminalMixer(_FallbackEquipment):
    def __init__(self, outlet, inlets=None):
        self._outlet = outlet
        self._inlets = inlets or []

    def getOutletStream(self):
        return self._outlet

    def getInletStreams(self):
        return self._inlets


class _FallbackHeater:
    def __init__(self, inlet, outlet):
        self._inlet = inlet
        self._outlet = outlet

    def getClass(self):
        return _JavaClass("Heater")

    def getInletStream(self):
        return self._inlet

    def getOutletStream(self):
        return self._outlet


class _FallbackSplitter:
    def __init__(self, inlet, outlets):
        self._inlet = inlet
        self._outlets = outlets

    def getClass(self):
        return _JavaClass("Splitter")

    def getInletStreams(self):
        return [self._inlet]

    def getOutletStreams(self):
        return self._outlets


class _FallbackAbsorber:
    def __init__(self, gas_inlet, solvent_inlet, outlet):
        self._gas_inlet = gas_inlet
        self._solvent_inlet = solvent_inlet
        self._outlet = outlet

    def getClass(self):
        return _JavaClass("SimpleTEGAbsorber")

    def getInStream(self):
        return self._gas_inlet

    def getSolventInStream(self):
        return self._solvent_inlet

    def getOutletStream(self):
        return self._outlet


class _FallbackTurboExpander:
    def __init__(
        self,
        compressor_inlet,
        expander_inlet,
        compressor_outlet,
        expander_outlet,
    ):
        self._compressor_inlet = compressor_inlet
        self._expander_inlet = expander_inlet
        self._compressor_outlet = compressor_outlet
        self._expander_outlet = expander_outlet

    def getClass(self):
        return _JavaClass("TurboExpanderCompressor")

    def getName(self):
        return "terminal turbo-expander"

    def getInletStreams(self):
        return [self._expander_inlet]

    def getCompressorFeedStream(self):
        return self._compressor_inlet

    def getExpanderFeedStream(self):
        return self._expander_inlet

    def getOutletStream(self):
        return self._compressor_outlet

    def getCompressorOutletStream(self):
        return self._compressor_outlet

    def getExpanderOutletStream(self):
        return self._expander_outlet


class _FallbackEjector:
    def __init__(self, motive_inlet, suction_inlet, outlet):
        self._motive_inlet = motive_inlet
        self._suction_inlet = suction_inlet
        self._outlet = outlet

    def getClass(self):
        return _JavaClass("Ejector")

    def getMotiveStream(self):
        return self._motive_inlet

    def getSuctionStream(self):
        return self._suction_inlet

    def getOutletStream(self):
        return self._outlet


class _FallbackInletMixer:
    def __init__(self, *inlets):
        self._inlets = inlets

    def getClass(self):
        return _JavaClass("Mixer")

    def getInletStreams(self):
        return self._inlets


class _FallbackTank:
    def __init__(self, gas_outlet, liquid_outlet, inlet_stream=None):
        self._gas_outlet = gas_outlet
        self._liquid_outlet = liquid_outlet
        self.inletStreamMixer = _FallbackInletMixer(inlet_stream)

    def getClass(self):
        return _JavaClass("Tank")

    def getName(self):
        return "storage tank"

    def getGasOutStream(self):
        return self._gas_outlet

    def getLiquidOutStream(self):
        return self._liquid_outlet


class _FallbackElectrolyzer:
    def __init__(self, hydrogen_outlet, oxygen_outlet, water_inlet=None):
        self._hydrogen_outlet = hydrogen_outlet
        self._oxygen_outlet = oxygen_outlet
        self.waterInlet = water_inlet

    def getClass(self):
        return _JavaClass("Electrolyzer")

    def getHydrogenOutStream(self):
        return self._hydrogen_outlet

    def getOxygenOutStream(self):
        return self._oxygen_outlet


class _FallbackCO2Electrolyzer:
    def __init__(self, gas_outlet, liquid_outlet, inlet_stream=None):
        self._gas_outlet = gas_outlet
        self._liquid_outlet = liquid_outlet
        self.inletStream = inlet_stream

    def getClass(self):
        return _JavaClass("CO2Electrolyzer")

    def getGasProductStream(self):
        return self._gas_outlet

    def getLiquidProductStream(self):
        return self._liquid_outlet


class _FallbackProcessModel:
    def __init__(self, processes):
        self._processes = processes

    def getAllProcesses(self):
        return self._processes


class _FallbackReactiveEquipment:
    def __init__(self, class_name="GibbsReactor"):
        self._class_name = class_name

    def getClass(self):
        return _JavaClass(self._class_name)

    def getName(self):
        return self._class_name


class _FallbackUnknownEquipment(_FallbackReactiveEquipment):
    def __init__(self):
        super().__init__("CustomNativeEquipment")


class _FallbackDistillationColumn(_FallbackReactiveEquipment):
    def __init__(self, reactive):
        super().__init__("DistillationColumn")
        self._reactive = reactive

    def isReactive(self):
        return self._reactive


class ValidationSummaryTest(unittest.TestCase):
    """Preserve incomplete validation as a non-passing aggregate state."""

    def test_status_precedence_preserves_unknown_checks(self):
        self.assertEqual(aggregate_validation_status([]), "OK")
        self.assertEqual(aggregate_validation_status(["OK"]), "OK")
        self.assertEqual(
            aggregate_validation_status(["OK", "UNKNOWN"]),
            "UNKNOWN",
        )
        self.assertEqual(
            aggregate_validation_status(["UNKNOWN", "WARN"]),
            "WARN",
        )
        self.assertEqual(
            aggregate_validation_status(["WARN", "VIOLATION"]),
            "VIOLATION",
        )

    def test_unrecognized_status_is_incomplete(self):
        self.assertEqual(
            aggregate_validation_status(["OK", "not-reported"]),
            "UNKNOWN",
        )


class ConvergenceDiagnosticsTest(unittest.TestCase):
    """Validate strict adapters and native iterative-unit evidence."""

    @staticmethod
    def _convergence_result(diagnostics):
        return SimpleNamespace(
            raw={"convergence_diagnostics": diagnostics},
            kpis={},
        )

    @staticmethod
    def _build_native_recycle_case(flow_scale):
        from neqsim import jneqsim

        equipment = jneqsim.process.equipment
        fluid = jneqsim.thermo.system.SystemSrkEos(298.15, 30.0)
        fluid.addComponent("methane", 0.9)
        fluid.addComponent("ethane", 0.1)
        fluid.setMixingRule(2)

        feed = equipment.stream.Stream("fresh feed", fluid)
        feed.setFlowRate(1000.0 * flow_scale, "kg/hr")
        recycle_guess = feed.clone("recycle guess")
        recycle_guess.setFlowRate(100.0 * flow_scale, "kg/hr")
        mixer = equipment.mixer.Mixer("feed mixer")
        mixer.addStream(feed)
        mixer.addStream(recycle_guess)
        compressor = equipment.compressor.Compressor(
            "compressor",
            mixer.getOutletStream(),
        )
        compressor.setOutletPressure(50.0, "bara")
        cooler = equipment.heatexchanger.Cooler(
            "cooler",
            compressor.getOutletStream(),
        )
        cooler.setOutTemperature(303.15)
        splitter = equipment.splitter.Splitter(
            "splitter",
            cooler.getOutletStream(),
        )
        splitter.setSplitFactors([0.1, 0.9])
        recycle = equipment.util.Recycle("gas recycle")
        recycle.addStream(splitter.getSplitStream(0))
        recycle.setOutletStream(recycle_guess)
        recycle.setTolerance(1.0e-4)
        recycle.setMaxIterations(50)

        process = jneqsim.process.processmodel.ProcessSystem()
        for unit in (
            feed,
            recycle_guess,
            mixer,
            compressor,
            cooler,
            splitter,
            recycle,
        ):
            process.add(unit)
        return NeqSimProcessModel(process), splitter

    def test_legacy_and_feed_forward_state_remain_explicit(self):
        legacy = SimpleNamespace(raw={})
        self.assertEqual(
            aggregate_convergence(legacy),
            {
                "applicable": None,
                "converged": None,
                "unit_count": None,
                "unconverged_count": None,
                "max_iterations": None,
                "suggestions": [],
            },
        )

        feed_forward = self._convergence_result(
            {
                "applicable": False,
                "converged": None,
                "rows": [],
                "suggestions": [],
            }
        )
        self.assertEqual(convergence_rows(feed_forward), [])
        self.assertIs(
            aggregate_convergence(feed_forward)["applicable"],
            False,
        )

    def test_rows_are_isolated_and_aggregate_unconverged_units(self):
        diagnostics = {
            "applicable": True,
            "converged": False,
            "rows": [
                {
                    "process_system": "gas plant",
                    "unit_name": "gas recycle",
                    "unit_type": "recycle",
                    "converged": False,
                    "iterations": 10,
                    "max_iterations": 10,
                    "dominant_error": "flow",
                    "acceleration_method": "DIRECT_SUBSTITUTION",
                    "flow_error": 0.1,
                    "flow_tolerance": 0.01,
                },
                {
                    "process_system": "gas plant",
                    "unit_name": "pressure adjuster",
                    "unit_type": "adjuster",
                    "converged": True,
                    "iterations": None,
                    "error": 0.001,
                    "tolerance": 0.01,
                },
            ],
            "suggestions": ["Improve the recycle initial estimate."],
        }
        result = self._convergence_result(diagnostics)

        rows = convergence_rows(result)
        summary = aggregate_convergence(result)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["dominant_error"], "flow")
        self.assertEqual(summary["converged"], False)
        self.assertEqual(summary["unit_count"], 2.0)
        self.assertEqual(summary["unconverged_count"], 1.0)
        self.assertEqual(summary["max_iterations"], 10.0)
        rows[0]["unit_name"] = "changed"
        self.assertEqual(
            diagnostics["rows"][0]["unit_name"],
            "gas recycle",
        )

    def test_rejects_conflicting_or_duplicate_unit_state(self):
        conflicting = self._convergence_result(
            {
                "applicable": True,
                "converged": True,
                "rows": [
                    {
                        "process_system": "main",
                        "unit_name": "recycle",
                        "unit_type": "recycle",
                        "converged": False,
                    }
                ],
                "suggestions": [],
            }
        )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            aggregate_convergence(conflicting)

        duplicate_row = {
            "process_system": "main",
            "unit_name": "recycle",
            "unit_type": "recycle",
            "converged": True,
        }
        duplicate = self._convergence_result(
            {
                "applicable": True,
                "converged": True,
                "rows": [duplicate_row, dict(duplicate_row)],
                "suggestions": [],
            }
        )
        with self.assertRaisesRegex(ValueError, "duplicates"):
            convergence_rows(duplicate)

    def test_native_recycle_convergence_and_nearby_flow(self):
        for flow_scale in (1.0, 1.05):
            with self.subTest(flow_scale=flow_scale):
                model, splitter = self._build_native_recycle_case(
                    flow_scale
                )
                result = model.run(timeout_ms=180_000)

                rows = convergence_rows(result)
                summary = aggregate_convergence(result)
                constraint = next(
                    item
                    for item in result.constraints
                    if item.name == "convergence"
                )

                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["unit_name"], "gas recycle")
                self.assertTrue(rows[0]["converged"])
                self.assertEqual(rows[0]["iterations"], 3)
                self.assertLess(
                    rows[0]["flow_error"],
                    rows[0]["flow_tolerance"],
                )
                self.assertEqual(rows[0]["dominant_error"], "flow")
                self.assertEqual(
                    rows[0]["acceleration_method"],
                    "DIRECT_SUBSTITUTION",
                )
                self.assertTrue(summary["converged"])
                self.assertEqual(summary["max_iterations"], 3.0)
                self.assertEqual(constraint.status, "OK")
                self.assertEqual(
                    result.kpis[
                        "convergence_unconverged_count"
                    ].value,
                    0.0,
                )
                product_flow = float(
                    splitter.getSplitStream(1).getFlowRate("kg/hr")
                )
                self.assertAlmostEqual(
                    product_flow,
                    1000.0 * flow_scale,
                    delta=0.2,
                )


class UnitBalanceDiagnosticsTest(unittest.TestCase):
    """Validate the strict per-unit material and energy closure contract."""

    @staticmethod
    def _build_separator_case(
        separator_type,
        inlet_count,
        flow_scale,
    ):
        from neqsim import jneqsim

        inlet_specs = [
            {
                "inlet_id": "feed-a",
                "name": "feed a",
                "fluid_spec": {
                    "eos_model": "srk",
                    "mixing_rule": 2,
                    "components": {
                        "methane": 0.90,
                        "ethane": 0.10,
                    },
                    "composition_basis": "mole_fraction",
                    "temperature_C": 25.0,
                    "pressure_bara": 45.0,
                    "total_flow": 60_000.0 * flow_scale,
                    "flow_unit": "kg/hr",
                },
            },
            {
                "inlet_id": "feed-b",
                "name": "feed b",
                "fluid_spec": {
                    "eos_model": "srk",
                    "mixing_rule": 2,
                    "components": {
                        "methane": 0.90,
                        "ethane": 0.10,
                    },
                    "composition_basis": "mole_fraction",
                    "temperature_C": 25.0,
                    "pressure_bara": 45.0,
                    "total_flow": 40_000.0 * flow_scale,
                    "flow_unit": "kg/hr",
                },
            },
        ][:inlet_count]
        streams = ProcessBuilder().create_inlet_streams(inlet_specs)
        process = jneqsim.process.processmodel.ProcessSystem()
        process.setName(f"{separator_type} closure")
        for stream in streams.values():
            process.add(stream)

        separator_class = getattr(
            jneqsim.process.equipment.separator,
            separator_type,
        )
        separator = separator_class(
            "inlet separator",
            streams["feed-a"],
        )
        if inlet_count == 2:
            separator.addStream(streams["feed-b"])
        process.add(separator)
        return NeqSimProcessModel(process)

    @staticmethod
    def _result(diagnostics):
        return SimpleNamespace(
            raw={"unit_balance_diagnostics": diagnostics},
            kpis={},
        )

    @staticmethod
    def _row():
        return {
            "process_system": "gas plant",
            "unit_name": "feed mixer",
            "unit_type": "Mixer",
            "inlet_count": 2,
            "outlet_count": 1,
            "inlet_mass_flow_kg_hr": 100000.0,
            "outlet_mass_flow_kg_hr": 100000.0,
            "mass_residual_kg_hr": 0.0,
            "mass_imbalance_pct": 0.0,
            "inlet_enthalpy_kW": -15000.0,
            "outlet_enthalpy_kW": -15000.0,
            "external_energy_transfer_kW": 0.0,
            "energy_residual_kW": 0.0,
            "energy_imbalance_pct": 0.0,
        }

    def test_legacy_and_empty_diagnostics_remain_explicit(self):
        legacy = aggregate_unit_balances(SimpleNamespace(raw={}))
        self.assertIsNone(legacy["applicable"])
        self.assertIsNone(legacy["coverage_complete"])
        self.assertIsNone(legacy["unit_count"])
        self.assertIsNone(legacy["max_mass_imbalance_unit"])
        self.assertIsNone(legacy["max_energy_imbalance_unit"])

        empty = self._result(
            {
                "applicable": False,
                "coverage_complete": True,
                "rows": [],
                "excluded_units": [],
            }
        )
        summary = aggregate_unit_balances(empty)
        self.assertFalse(summary["applicable"])
        self.assertTrue(summary["coverage_complete"])
        self.assertEqual(summary["unit_count"], 0.0)

    def test_rows_are_isolated_and_aggregate_partial_coverage(self):
        diagnostics = {
            "applicable": True,
            "coverage_complete": False,
            "rows": [self._row()],
            "excluded_units": ["column (DistillationColumn)"],
        }
        result = self._result(diagnostics)

        rows = unit_balance_rows(result)
        summary = aggregate_unit_balances(result)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["inlet_count"], 2)
        self.assertEqual(summary["unit_count"], 1.0)
        self.assertEqual(summary["energy_unit_count"], 1.0)
        self.assertEqual(summary["max_mass_imbalance_pct"], 0.0)
        self.assertEqual(summary["max_energy_imbalance_pct"], 0.0)
        expected_limiting_unit = {
            "process_system": "gas plant",
            "unit_name": "feed mixer",
            "unit_type": "Mixer",
        }
        self.assertEqual(
            summary["max_mass_imbalance_unit"],
            expected_limiting_unit,
        )
        self.assertEqual(
            summary["max_energy_imbalance_unit"],
            expected_limiting_unit,
        )
        self.assertEqual(
            summary["excluded_units"],
            ["column (DistillationColumn)"],
        )
        rows[0]["unit_name"] = "changed"
        self.assertEqual(
            diagnostics["rows"][0]["unit_name"],
            "feed mixer",
        )

    def test_allows_mass_only_rows(self):
        row = self._row()
        for field_name in (
            "inlet_enthalpy_kW",
            "outlet_enthalpy_kW",
            "external_energy_transfer_kW",
            "energy_residual_kW",
            "energy_imbalance_pct",
        ):
            row[field_name] = None
        result = self._result(
            {
                "applicable": True,
                "coverage_complete": True,
                "rows": [row],
                "excluded_units": [],
            }
        )

        summary = aggregate_unit_balances(result)

        self.assertEqual(summary["energy_unit_count"], 0.0)
        self.assertIsNone(summary["max_energy_imbalance_pct"])
        self.assertIsNone(summary["max_energy_imbalance_unit"])

    def test_limiting_unit_attribution_is_deterministic_for_ties(self):
        later_row = self._row()
        later_row["process_system"] = "second train"
        later_row["unit_name"] = "z cooler"
        later_row["mass_imbalance_pct"] = 0.25
        later_row["energy_imbalance_pct"] = 0.5
        earlier_row = self._row()
        earlier_row["process_system"] = "first train"
        earlier_row["unit_name"] = "a compressor"
        earlier_row["mass_imbalance_pct"] = 0.25
        earlier_row["energy_imbalance_pct"] = 0.5
        result = self._result(
            {
                "applicable": True,
                "coverage_complete": True,
                "rows": [later_row, earlier_row],
                "excluded_units": [],
            }
        )

        summary = aggregate_unit_balances(result)

        expected = {
            "process_system": "first train",
            "unit_name": "a compressor",
            "unit_type": "Mixer",
        }
        self.assertEqual(summary["max_mass_imbalance_unit"], expected)
        self.assertEqual(summary["max_energy_imbalance_unit"], expected)
        summary["max_mass_imbalance_unit"]["unit_name"] = "changed"
        self.assertEqual(
            result.raw["unit_balance_diagnostics"]["rows"][1]["unit_name"],
            "a compressor",
        )

    def test_rejects_incomplete_or_conflicting_diagnostics(self):
        incomplete = self._row()
        incomplete["energy_residual_kW"] = None
        with self.assertRaisesRegex(ValueError, "incomplete energy"):
            unit_balance_rows(
                self._result(
                    {
                        "applicable": True,
                        "coverage_complete": True,
                        "rows": [incomplete],
                        "excluded_units": [],
                    }
                )
            )

        with self.assertRaisesRegex(ValueError, "coverage state conflicts"):
            aggregate_unit_balances(
                self._result(
                    {
                        "applicable": True,
                        "coverage_complete": True,
                        "rows": [self._row()],
                        "excluded_units": ["unknown unit"],
                    }
                )
            )

        with self.assertRaisesRegex(ValueError, "applicability conflicts"):
            aggregate_unit_balances(
                self._result(
                    {
                        "applicable": False,
                        "coverage_complete": True,
                        "rows": [self._row()],
                        "excluded_units": [],
                    }
                )
            )

    def test_rejects_booleans_in_numeric_closure_fields(self):
        numeric_fields = (
            "inlet_mass_flow_kg_hr",
            "outlet_mass_flow_kg_hr",
            "mass_residual_kg_hr",
            "mass_imbalance_pct",
            "inlet_enthalpy_kW",
            "outlet_enthalpy_kW",
            "external_energy_transfer_kW",
            "energy_residual_kW",
            "energy_imbalance_pct",
        )
        for field_name in numeric_fields:
            with self.subTest(field_name=field_name):
                row = self._row()
                row[field_name] = True
                with self.assertRaisesRegex(
                    ValueError,
                    f"field '{field_name}' must be numeric",
                ):
                    unit_balance_rows(
                        self._result(
                            {
                                "applicable": True,
                                "coverage_complete": True,
                                "rows": [row],
                                "excluded_units": [],
                            }
                        )
                    )

    def test_native_closure_and_nearby_operating_points(self):
        benchmark = (
            graph_conservation.MultiInletMixerConservationTest
        )
        for flow_scale in (1.0, 1.05):
            with self.subTest(
                case="two-inlet mixer",
                flow_scale=flow_scale,
            ):
                _, model = benchmark._build_case(flow_scale)
                result = model.run(timeout_ms=180_000)
                rows = unit_balance_rows(result)
                summary = aggregate_unit_balances(result)

                self.assertEqual(
                    [
                        (
                            row["unit_name"],
                            row["inlet_count"],
                            row["outlet_count"],
                        )
                        for row in rows
                    ],
                    [("feed mixer", 2, 1)],
                )
                self.assertTrue(summary["coverage_complete"])
                self.assertLess(
                    summary["max_mass_imbalance_pct"],
                    1.0e-6,
                )
                self.assertLess(
                    summary["max_energy_imbalance_pct"],
                    1.0e-6,
                )

            with self.subTest(
                case="compression and cooling",
                flow_scale=flow_scale,
            ):
                _, model = benchmark._build_compression_cooling_case(
                    flow_scale
                )
                result = model.run(timeout_ms=180_000)
                rows = unit_balance_rows(result)
                summary = aggregate_unit_balances(result)

                self.assertEqual(
                    [row["unit_name"] for row in rows],
                    ["compressor", "cooler"],
                )
                self.assertGreater(
                    rows[0]["external_energy_transfer_kW"],
                    0.0,
                )
                self.assertLess(
                    rows[1]["external_energy_transfer_kW"],
                    0.0,
                )
                self.assertTrue(summary["coverage_complete"])
                self.assertLess(
                    summary["max_mass_imbalance_pct"],
                    1.0e-6,
                )
                self.assertLess(
                    summary["max_energy_imbalance_pct"],
                    1.0e-6,
                )
                unit_constraints = {
                    item.name: item.status
                    for item in result.constraints
                    if item.name.startswith("unit_")
                }
                self.assertEqual(
                    unit_constraints,
                    {
                        "unit_mass_balance": "OK",
                        "unit_energy_balance": "OK",
                    },
                )

    def test_native_separator_closure_uses_only_external_feeds(self):
        for separator_type, outlet_count in (
            ("Separator", 2),
            ("ThreePhaseSeparator", 3),
        ):
            for inlet_count in (1, 2):
                for flow_scale in (1.0, 1.05):
                    with self.subTest(
                        separator_type=separator_type,
                        inlet_count=inlet_count,
                        flow_scale=flow_scale,
                    ):
                        model = self._build_separator_case(
                            separator_type,
                            inlet_count,
                            flow_scale,
                        )
                        result = model.run(timeout_ms=180_000)
                        rows = unit_balance_rows(result)
                        summary = aggregate_unit_balances(result)

                        self.assertEqual(len(rows), 1)
                        self.assertEqual(
                            rows[0]["unit_name"],
                            "inlet separator",
                        )
                        self.assertEqual(
                            rows[0]["inlet_count"],
                            inlet_count,
                        )
                        self.assertEqual(
                            rows[0]["outlet_count"],
                            outlet_count,
                        )
                        self.assertTrue(summary["coverage_complete"])
                        self.assertLess(
                            summary["max_mass_imbalance_pct"],
                            1.0e-6,
                        )
                        self.assertLess(
                            summary["max_energy_imbalance_pct"],
                            1.0e-6,
                        )


def _result(rows=None, **kpi_values):
    return SimpleNamespace(
        raw={"material_boundaries": rows or []},
        kpis={
            name: SimpleNamespace(value=value)
            for name, value in kpi_values.items()
        },
    )


class MaterialBoundaryDiagnosticsTest(unittest.TestCase):
    """Validate strict rows, aggregation, and legacy compatibility."""

    def test_returns_isolated_rows_and_aggregates_multiple_feeds(self):
        source_rows = [
            {
                "role": "feed",
                "stream_name": "dry gas",
                "mass_flow_kg_hr": 60_000,
                "temperature_C": 20,
                "pressure_bara": 45,
                "molar_flow_mol_sec": 900,
                "enthalpy_flow_kW": 1250,
            },
            {
                "role": "feed",
                "stream_name": "rich gas",
                "mass_flow_kg_hr": 40_000,
                "temperature_C": 35,
                "pressure_bara": 45,
                "molar_flow_mol_sec": 500,
                "enthalpy_flow_kW": 750,
            },
            {
                "role": "product",
                "stream_name": "mixed product",
                "mass_flow_kg_hr": 100_000,
                "temperature_C": 25,
                "pressure_bara": 45,
                "molar_flow_mol_sec": 1400,
                "enthalpy_flow_kW": 2000,
            },
        ]
        result = _result(
            source_rows,
            mass_balance_pct=1.0e-12,
        )

        rows = material_boundary_rows(result)
        rows[0]["mass_flow_kg_hr"] = 1.0
        self.assertEqual(
            result.raw["material_boundaries"][0]["mass_flow_kg_hr"],
            60_000,
        )
        summary = aggregate_material_balance(result)
        self.assertEqual(summary["feed_count"], 2.0)
        self.assertEqual(summary["product_count"], 1.0)
        self.assertEqual(summary["feed_flow_kg_hr"], 100_000.0)
        self.assertEqual(summary["product_flow_kg_hr"], 100_000.0)
        self.assertEqual(summary["imbalance_pct"], 1.0e-12)
        self.assertEqual(
            solved_feed_flow_kg_hr(result, 60_000.0),
            100_000.0,
        )
        self.assertEqual(rows[0]["enthalpy_flow_kW"], 1250.0)

    def test_extracts_native_boundary_enthalpy_flow_in_kw(self):
        record = NeqSimProcessModel._material_boundary_record(
            _FallbackStream(
                "warm feed",
                100.0,
                enthalpy_flow_w=1_250_000.0,
            ),
            "feed",
            "feed",
        )

        self.assertEqual(record["enthalpy_flow_kW"], 1250.0)

    def test_zero_flow_boundary_has_zero_enthalpy_flow(self):
        record = NeqSimProcessModel._material_boundary_record(
            _FallbackStream(
                "empty product",
                0.0,
                enthalpy_flow_w=99_000.0,
            ),
            "product",
            "product",
        )

        self.assertEqual(record["mass_flow_kg_hr"], 0.0)
        self.assertEqual(record["enthalpy_flow_kW"], 0.0)

    def test_aggregates_signed_system_energy_closure(self):
        rows = [
            {
                "role": "feed",
                "stream_name": "feed",
                "mass_flow_kg_hr": 100_000.0,
                "enthalpy_flow_kW": 514.109802,
            },
            {
                "role": "product",
                "stream_name": "product",
                "mass_flow_kg_hr": 100_000.0,
                "enthalpy_flow_kW": -90.016530,
            },
        ]
        result = _result(rows)
        result.raw.update(
            {
                "energy_balance_applicable": True,
                "energy_transfers": [
                    {
                        "unit_name": "compressor",
                        "unit_type": "Compressor",
                        "transfer_kind": "shaft_work",
                        "energy_transfer_kW": 3490.419834,
                    },
                    {
                        "unit_name": "cooler",
                        "unit_type": "Cooler",
                        "transfer_kind": "heat",
                        "energy_transfer_kW": -4094.546166,
                    },
                ],
            }
        )

        transfers = energy_transfer_rows(result)
        summary = aggregate_energy_balance(result)

        self.assertEqual(len(transfers), 2)
        self.assertAlmostEqual(
            summary["external_energy_transfer_kW"],
            -604.126332,
            places=6,
        )
        self.assertAlmostEqual(summary["residual_kW"], 0.0, places=6)
        self.assertLess(summary["imbalance_pct"], 1.0e-6)

    def test_energy_balance_rejects_missing_positive_flow_enthalpy(self):
        result = _result(
            [
                {
                    "role": "feed",
                    "stream_name": "feed",
                    "mass_flow_kg_hr": 100.0,
                    "enthalpy_flow_kW": None,
                },
                {
                    "role": "product",
                    "stream_name": "product",
                    "mass_flow_kg_hr": 100.0,
                    "enthalpy_flow_kW": 10.0,
                },
            ]
        )
        result.raw["energy_balance_applicable"] = True

        with self.assertRaisesRegex(
            ValueError,
            "incomplete for positive-flow row",
        ):
            aggregate_energy_balance(result)

    def test_energy_balance_preserves_explicit_inapplicability(self):
        result = _result([])
        result.raw.update(
            {
                "energy_balance_applicable": False,
                "energy_transfers": [],
            }
        )

        summary = aggregate_energy_balance(result)

        self.assertIs(summary["applicable"], False)
        self.assertIsNone(summary["imbalance_pct"])

    def test_legacy_boundaries_without_enthalpy_remain_unknown(self):
        result = _result(
            [
                {
                    "role": "feed",
                    "stream_name": "legacy feed",
                    "mass_flow_kg_hr": 100.0,
                },
                {
                    "role": "product",
                    "stream_name": "legacy product",
                    "mass_flow_kg_hr": 100.0,
                },
            ]
        )

        summary = aggregate_energy_balance(result)

        self.assertIsNone(summary["applicable"])
        self.assertIsNone(summary["imbalance_pct"])

    def test_energy_transfer_extraction_avoids_pump_duty_double_count(self):
        class _FallbackPump:
            def getClass(self):
                return _JavaClass("Pump")

            def getName(self):
                return "feed pump"

            def getPower(self):
                return 1_500_000.0

            def getDuty(self):
                return 1_500_000.0

        transfers, excluded = NeqSimProcessModel._system_energy_transfers(
            [_FallbackPump()]
        )

        self.assertEqual(excluded, [])
        self.assertEqual(
            transfers,
            [
                {
                    "unit_name": "feed pump",
                    "unit_type": "Pump",
                    "transfer_kind": "shaft_work",
                    "energy_transfer_kW": 1500.0,
                }
            ],
        )

    def test_aggregates_component_feed_and_product_closure(self):
        rows = [
            {
                "role": "feed",
                "stream_name": "dry gas",
                "mass_flow_kg_hr": 60_000,
                "molar_flow_mol_sec": 100.0,
                "component_molar_flows_mol_sec": {
                    "methane": 95.0,
                    "ethane": 5.0,
                },
            },
            {
                "role": "feed",
                "stream_name": "rich gas",
                "mass_flow_kg_hr": 40_000,
                "molar_flow_mol_sec": 50.0,
                "component_molar_flows_mol_sec": {
                    "methane": 40.0,
                    "ethane": 10.0,
                },
            },
            {
                "role": "product",
                "stream_name": "mixed product",
                "mass_flow_kg_hr": 100_000,
                "molar_flow_mol_sec": 150.0,
                "component_molar_flows_mol_sec": {
                    "methane": 135.0,
                    "ethane": 15.0,
                },
            },
        ]

        balances = component_balance_rows(_result(rows))

        self.assertEqual(
            [row["component"] for row in balances],
            ["ethane", "methane"],
        )
        for row in balances:
            self.assertEqual(
                row["feed_molar_flow_mol_sec"],
                row["product_molar_flow_mol_sec"],
            )
            self.assertEqual(row["residual_molar_flow_mol_sec"], 0.0)
            self.assertEqual(row["imbalance_pct"], 0.0)

    def test_component_balance_preserves_legacy_and_rejects_gaps(self):
        legacy = _result(
            [
                {
                    "role": "feed",
                    "stream_name": "legacy feed",
                    "mass_flow_kg_hr": 1.0,
                    "molar_flow_mol_sec": 1.0,
                }
            ]
        )
        self.assertEqual(component_balance_rows(legacy), [])

        current_without_components = _result(
            [
                {
                    "role": "feed",
                    "stream_name": "feed",
                    "mass_flow_kg_hr": 1.0,
                    "molar_flow_mol_sec": None,
                    "component_molar_flows_mol_sec": None,
                }
            ]
        )
        current_without_components.raw[
            "component_balance_applicable"
        ] = True
        with self.assertRaisesRegex(ValueError, "incomplete"):
            component_balance_rows(current_without_components)

        incomplete = _result(
            [
                {
                    "role": "feed",
                    "stream_name": "feed",
                    "mass_flow_kg_hr": 1.0,
                    "molar_flow_mol_sec": 1.0,
                    "component_molar_flows_mol_sec": {},
                },
                {
                    "role": "product",
                    "stream_name": "product",
                    "mass_flow_kg_hr": 1.0,
                    "molar_flow_mol_sec": 1.0,
                    "component_molar_flows_mol_sec": {"methane": 1.0},
                },
            ]
        )
        with self.assertRaisesRegex(ValueError, "incomplete"):
            component_balance_rows(incomplete)

        missing_molar_flow = _result(
            [
                {
                    "role": "feed",
                    "stream_name": "feed",
                    "mass_flow_kg_hr": 1.0,
                    "molar_flow_mol_sec": None,
                    "component_molar_flows_mol_sec": None,
                },
                {
                    "role": "product",
                    "stream_name": "product",
                    "mass_flow_kg_hr": 1.0,
                    "molar_flow_mol_sec": 1.0,
                    "component_molar_flows_mol_sec": {"methane": 1.0},
                },
            ]
        )
        with self.assertRaisesRegex(ValueError, "incomplete"):
            component_balance_rows(missing_molar_flow)

        only_feed = _result(
            [
                {
                    "role": "feed",
                    "stream_name": "feed",
                    "mass_flow_kg_hr": 1.0,
                    "component_molar_flows_mol_sec": {
                        "methane": 1.0,
                    },
                }
            ]
        )
        with self.assertRaisesRegex(
            ValueError,
            "feed and product boundaries",
        ):
            component_balance_rows(only_feed)

        malformed = _result(
            [
                {
                    "role": "feed",
                    "stream_name": "feed",
                    "mass_flow_kg_hr": 1.0,
                    "component_molar_flows_mol_sec": {
                        "methane": -1.0,
                    },
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            component_balance_rows(malformed)

    def test_reactive_process_marks_species_closure_not_applicable(self):
        feed = _FallbackStream("feed", 100.0)
        product = _FallbackStream("product", 100.0)
        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._proc = _FallbackProcess(
            [feed, _FallbackReactiveEquipment(), product]
        )
        model._source_bytes = None
        model._units = {}
        model._streams = {"feed": feed, "product": product}
        model._is_process_model = False
        model._enforce_acyclic_mixer_energy = False

        result = model._extract_results()

        self.assertFalse(result.raw["component_balance_applicable"])
        self.assertEqual(result.raw["component_balances"], [])
        self.assertEqual(component_balance_rows(result), [])
        component_constraint = next(
            constraint
            for constraint in result.constraints
            if constraint.name == "component_balance"
        )
        self.assertEqual(component_constraint.status, "UNKNOWN")
        self.assertIn("not applicable", component_constraint.detail)
        self.assertNotIn("component_balance_max_pct", result.kpis)

    def test_species_changing_equipment_is_reactive(self):
        for class_name in (
            "GasTurbine",
            "CombustionEngine",
            "FuelCell",
            "H2SScavenger",
            "SimpleAbsorber",
            "FurnaceBurner",
            "SyngasBurnerZone",
            "BiomassGasifier",
            "AutothermalReformer",
            "CatalyticTubeReformer",
            "ReformerFurnace",
            "ReactiveTray",
        ):
            with self.subTest(class_name=class_name):
                self.assertEqual(
                    NeqSimProcessModel._component_balance_exclusion_names(
                        [_FallbackReactiveEquipment(class_name)]
                    ),
                    [class_name],
                )

    def test_only_reactive_distillation_columns_change_species(self):
        self.assertEqual(
            NeqSimProcessModel._component_balance_exclusion_names(
                [_FallbackDistillationColumn(True)]
            ),
            ["DistillationColumn"],
        )
        self.assertEqual(
            NeqSimProcessModel._component_balance_exclusion_names(
                [_FallbackDistillationColumn(False)]
            ),
            [],
        )

    def test_unclassified_native_equipment_disables_species_closure(self):
        self.assertEqual(
            NeqSimProcessModel._component_balance_exclusion_names(
                [_FallbackUnknownEquipment()]
            ),
            [
                "CustomNativeEquipment "
                "(unclassified CustomNativeEquipment)"
            ],
        )
        self.assertEqual(
            NeqSimProcessModel._component_balance_exclusion_names(
                [_FallbackEquipment()]
            ),
            [],
        )

    def test_supported_conserving_equipment_enables_species_closure(self):
        class_names = (
            "TurboExpanderCompressor",
            "GasScrubberSimple",
            "Hydrocyclone",
            "CheckValve",
            "WaterStripperColumn",
            "AdiabaticTwoPhasePipe",
            "SimpleTPoutPipeline",
            "MembraneSeparator",
            "WellFlow",
            "SimpleFlowLine",
            "EquilibriumStream",
        )
        for class_name in class_names:
            with self.subTest(class_name=class_name):
                self.assertEqual(
                    NeqSimProcessModel._component_balance_exclusion_names(
                        [_FallbackReactiveEquipment(class_name)]
                    ),
                    [],
                )

    def test_material_boundary_discovery_accepts_stream_subclasses(self):
        for class_name in ("Stream", "EquilibriumStream", "WellStream"):
            with self.subTest(class_name=class_name):
                feed = _FallbackStream(
                    "feed",
                    100.0,
                    class_name=class_name,
                )
                product = _FallbackStream(
                    "product",
                    100.0,
                    class_name=class_name,
                )
                units = [feed, _FallbackEquipment(), product]

                self.assertEqual(
                    NeqSimProcessModel._leading_material_feed_streams(
                        units
                    ),
                    [feed],
                )
                self.assertEqual(
                    NeqSimProcessModel._trailing_material_product_streams(
                        units
                    ),
                    [product],
                )

    def test_fallback_enumerates_all_heat_exchanger_products(self):
        feed_a = _FallbackStream("feed a", 100.0)
        feed_b = _FallbackStream("feed b", 100.0)
        product_a = _FallbackStream("product a", 100.0)
        product_b = _FallbackStream("product b", 100.0)
        exchanger = _FallbackHeatExchanger([product_a, product_b])
        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._proc = _FallbackProcess([feed_a, feed_b, exchanger])
        model._source_bytes = None
        model._units = {}
        model._streams = {
            "feed a": feed_a,
            "feed b": feed_b,
            "product a": product_a,
            "product b": product_b,
        }
        model._is_process_model = False
        model._enforce_acyclic_mixer_energy = False

        result = model._extract_results()

        self.assertEqual(
            [
                (row["role"], row["stream_name"])
                for row in result.raw["material_boundaries"]
            ],
            [
                ("feed", "feed a"),
                ("feed", "feed b"),
                ("product", "product a"),
                ("product", "product b"),
            ],
        )
        self.assertEqual(
            result.kpis["material_product_count"].value,
            2.0,
        )
        self.assertEqual(result.kpis["mass_balance_pct"].value, 0.0)

    def test_fallback_enumerates_both_turbo_expander_products(self):
        feed_a = _FallbackStream("compressor feed", 100.0)
        feed_b = _FallbackStream("expander feed", 100.0)
        product_a = _FallbackStream("compressor product", 100.0)
        product_b = _FallbackStream("expander product", 100.0)
        turbo_expander = _FallbackTurboExpander(
            feed_a,
            feed_b,
            product_a,
            product_b,
        )
        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._proc = _FallbackProcess(
            [feed_a, feed_b, turbo_expander]
        )
        model._source_bytes = None
        model._units = {}
        model._streams = {
            "compressor feed": feed_a,
            "expander feed": feed_b,
            "compressor product": product_a,
            "expander product": product_b,
        }
        model._is_process_model = False
        model._enforce_acyclic_mixer_energy = False

        result = model._extract_results()

        self.assertEqual(
            [
                row["stream_name"]
                for row in result.raw["material_boundaries"]
                if row["role"] == "product"
            ],
            ["compressor product", "expander product"],
        )
        self.assertEqual(result.kpis["mass_balance_pct"].value, 0.0)

    def test_fallback_enumerates_electrolyzer_products(self):
        cases = (
            (
                _FallbackElectrolyzer,
                ("hydrogen product", "oxygen product"),
            ),
            (
                _FallbackCO2Electrolyzer,
                ("gas product", "liquid product"),
            ),
        )
        for equipment_type, product_names in cases:
            with self.subTest(equipment=equipment_type.__name__):
                feed = _FallbackStream("feed", 100.0)
                processed_feed = _FallbackStream(
                    "processed feed",
                    100.0,
                )
                first_product = _FallbackStream(
                    product_names[0],
                    40.0,
                )
                second_product = _FallbackStream(
                    product_names[1],
                    60.0,
                )
                equipment = equipment_type(
                    first_product,
                    second_product,
                    processed_feed,
                )
                heater = _FallbackHeater(feed, processed_feed)
                model = NeqSimProcessModel.__new__(NeqSimProcessModel)
                model._proc = _FallbackProcess(
                    [feed, heater, equipment]
                )
                model._source_bytes = None
                model._units = {}
                model._streams = {
                    stream.getName(): stream
                    for stream in (
                        feed,
                        processed_feed,
                        first_product,
                        second_product,
                    )
                }
                model._is_process_model = False
                model._enforce_acyclic_mixer_energy = False

                result = model._extract_results()

                self.assertEqual(
                    [
                        row["stream_name"]
                        for row in result.raw["material_boundaries"]
                        if row["role"] == "product"
                    ],
                    list(product_names),
                )
                self.assertEqual(
                    result.raw["material_balance_applicable"],
                    True,
                )
                self.assertEqual(
                    result.kpis["mass_balance_pct"].value,
                    0.0,
                )
                mass_constraint = next(
                    constraint
                    for constraint in result.constraints
                    if constraint.name == "mass_balance"
                )
                self.assertEqual(mass_constraint.status, "OK")
                component_constraint = next(
                    constraint
                    for constraint in result.constraints
                    if constraint.name == "component_balance"
                )
                self.assertEqual(
                    component_constraint.status,
                    "UNKNOWN",
                )

    def test_unsafe_child_preserves_independent_process_products(self):
        unsafe_feed = _FallbackStream("unsafe feed", 100.0)
        processed_feed = _FallbackStream("processed feed", 100.0)
        hydrogen = _FallbackStream("hydrogen product", 40.0)
        oxygen = _FallbackStream("oxygen product", 60.0)
        safe_feed = _FallbackStream("safe feed", 50.0)
        safe_product_fluid = object()
        raw_safe_product = _FallbackAliasedStream(
            "raw safe product",
            50.0,
            safe_product_fluid,
        )
        named_safe_product = _FallbackAliasedStream(
            "safe product",
            50.0,
            safe_product_fluid,
        )
        process_model = _FallbackProcessModel(
            [
                _FallbackProcess(
                    [
                        unsafe_feed,
                        safe_feed,
                        _FallbackHeater(unsafe_feed, processed_feed),
                        _FallbackHeater(
                            safe_feed,
                            raw_safe_product,
                        ),
                        _FallbackElectrolyzer(
                            hydrogen,
                            oxygen,
                            processed_feed,
                        ),
                        named_safe_product,
                    ]
                ),
            ]
        )
        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._proc = process_model
        model._source_bytes = None
        model._units = {}
        model._streams = {
            stream.getName(): stream
            for stream in (
                unsafe_feed,
                processed_feed,
                hydrogen,
                oxygen,
                safe_feed,
                raw_safe_product,
                named_safe_product,
            )
        }
        model._is_process_model = True
        model._enforce_acyclic_mixer_energy = False

        result = model._extract_results()

        self.assertEqual(
            [
                row["stream_name"]
                for row in result.raw["material_boundaries"]
                if row["role"] == "product"
            ],
            ["safe product", "hydrogen product", "oxygen product"],
        )
        self.assertTrue(result.raw["material_balance_applicable"])
        self.assertEqual(
            result.kpis["mass_balance_pct"].value,
            0.0,
        )

    def test_fallback_discovers_products_for_each_child_process(self):
        feed_a = _FallbackStream("feed a", 100.0)
        feed_b = _FallbackStream("feed b", 100.0)
        product_a = _FallbackStream("product a", 100.0)
        product_b = _FallbackStream("product b", 100.0)
        process_model = _FallbackProcessModel(
            [
                _FallbackProcess(
                    [feed_a, _FallbackTerminalMixer(product_a)]
                ),
                _FallbackProcess(
                    [feed_b, _FallbackTerminalMixer(product_b)]
                ),
            ]
        )
        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._proc = process_model
        model._source_bytes = None
        model._units = {}
        model._streams = {
            "feed a": feed_a,
            "feed b": feed_b,
            "product a": product_a,
            "product b": product_b,
        }
        model._is_process_model = True
        model._enforce_acyclic_mixer_energy = False

        result = model._extract_results()

        self.assertEqual(
            [
                (row["role"], row["stream_name"])
                for row in result.raw["material_boundaries"]
            ],
            [
                ("feed", "feed a"),
                ("feed", "feed b"),
                ("product", "product a"),
                ("product", "product b"),
            ],
        )
        self.assertEqual(
            result.kpis["material_product_count"].value,
            2.0,
        )
        self.assertEqual(result.kpis["mass_balance_pct"].value, 0.0)

    def test_connectivity_crosses_process_model_children(self):
        feed = _FallbackStream("feed", 100.0)
        child_boundary = _FallbackStream("child boundary", 100.0)
        product = _FallbackStream("product", 100.0)
        process_model = _FallbackProcessModel(
            [
                _FallbackProcess(
                    [
                        feed,
                        _FallbackHeater(feed, child_boundary),
                        child_boundary,
                    ]
                ),
                _FallbackProcess(
                    [_FallbackHeater(child_boundary, product)]
                ),
            ]
        )
        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._proc = process_model
        model._source_bytes = None
        model._units = {}
        model._streams = {
            stream.getName(): stream
            for stream in (feed, child_boundary, product)
        }
        model._is_process_model = True
        model._enforce_acyclic_mixer_energy = False

        result = model._extract_results()

        self.assertEqual(
            [
                (row["role"], row["stream_name"])
                for row in result.raw["material_boundaries"]
            ],
            [("feed", "feed"), ("product", "product")],
        )
        self.assertEqual(result.kpis["mass_balance_pct"].value, 0.0)

    def test_connectivity_discovers_parallel_terminal_branches(self):
        feed = _FallbackStream("feed", 200.0)
        branch_a = _FallbackStream("branch a", 100.0)
        branch_b = _FallbackStream("branch b", 100.0)
        product_a = _FallbackStream("product a", 100.0)
        product_b = _FallbackStream("product b", 100.0)
        splitter = _FallbackSplitter(feed, [branch_a, branch_b])
        heater_a = _FallbackHeater(branch_a, product_a)
        heater_b = _FallbackHeater(branch_b, product_b)
        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._proc = _FallbackProcess(
            [feed, splitter, heater_a, heater_b]
        )
        model._source_bytes = None
        model._units = {}
        model._streams = {
            stream.getName(): stream
            for stream in (
                feed,
                branch_a,
                branch_b,
                product_a,
                product_b,
            )
        }
        model._is_process_model = False
        model._enforce_acyclic_mixer_energy = False

        result = model._extract_results()

        self.assertEqual(
            [
                row["stream_name"]
                for row in result.raw["material_boundaries"]
                if row["role"] == "product"
            ],
            ["product a", "product b"],
        )
        self.assertEqual(result.kpis["mass_balance_pct"].value, 0.0)

    def test_explicit_product_alias_preserves_implicit_peer_branch(self):
        feed = _FallbackStream("feed", 200.0)
        branch_a = _FallbackStream("branch a", 100.0)
        branch_b = _FallbackStream("branch b", 100.0)
        product_a_fluid = object()
        raw_product_a = _FallbackAliasedStream(
            "raw product a",
            100.0,
            product_a_fluid,
        )
        explicit_product_a = _FallbackAliasedStream(
            "named product a",
            100.0,
            product_a_fluid,
        )
        product_b = _FallbackStream("product b", 100.0)
        splitter = _FallbackSplitter(feed, [branch_a, branch_b])
        heater_a = _FallbackHeater(branch_a, raw_product_a)
        heater_b = _FallbackHeater(branch_b, product_b)
        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._proc = _FallbackProcess(
            [
                feed,
                splitter,
                heater_a,
                heater_b,
                explicit_product_a,
            ]
        )
        model._source_bytes = None
        model._units = {}
        model._streams = {
            stream.getName(): stream
            for stream in (
                feed,
                branch_a,
                branch_b,
                raw_product_a,
                explicit_product_a,
                product_b,
            )
        }
        model._is_process_model = False
        model._enforce_acyclic_mixer_energy = False

        result = model._extract_results()

        self.assertEqual(
            [
                row["stream_name"]
                for row in result.raw["material_boundaries"]
                if row["role"] == "product"
            ],
            ["named product a", "product b"],
        )
        self.assertEqual(result.kpis["mass_balance_pct"].value, 0.0)

    def test_connectivity_matches_intermediate_stream_fluid_alias(self):
        feed = _FallbackStream("feed", 100.0)
        intermediate_fluid = object()
        raw_intermediate = _FallbackAliasedStream(
            "raw intermediate",
            100.0,
            intermediate_fluid,
        )
        named_intermediate = _FallbackAliasedStream(
            "named intermediate",
            100.0,
            intermediate_fluid,
        )
        product = _FallbackStream("product", 100.0)
        heater = _FallbackHeater(feed, raw_intermediate)
        compressor = _FallbackTerminalMixer(
            product,
            [named_intermediate],
        )
        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._proc = _FallbackProcess(
            [feed, heater, named_intermediate, compressor]
        )
        model._source_bytes = None
        model._units = {}
        model._streams = {
            stream.getName(): stream
            for stream in (
                feed,
                raw_intermediate,
                named_intermediate,
                product,
            )
        }
        model._is_process_model = False
        model._enforce_acyclic_mixer_energy = False

        result = model._extract_results()

        self.assertEqual(
            [
                (row["role"], row["stream_name"])
                for row in result.raw["material_boundaries"]
            ],
            [("feed", "feed"), ("product", "product")],
        )
        self.assertEqual(
            result.kpis["material_feed_flow_kg_hr"].value,
            100.0,
        )
        self.assertEqual(result.kpis["mass_balance_pct"].value, 0.0)

    def test_connectivity_discovers_absorber_solvent_feed(self):
        gas_feed = _FallbackStream("gas feed", 100.0)
        heated_gas = _FallbackStream("heated gas", 100.0)
        solvent_feed = _FallbackStream("solvent feed", 25.0)
        product = _FallbackStream("absorber product", 125.0)
        heater = _FallbackHeater(gas_feed, heated_gas)
        absorber = _FallbackAbsorber(
            heated_gas,
            solvent_feed,
            product,
        )
        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._proc = _FallbackProcess(
            [gas_feed, heater, solvent_feed, absorber]
        )
        model._source_bytes = None
        model._units = {}
        model._streams = {
            stream.getName(): stream
            for stream in (
                gas_feed,
                heated_gas,
                solvent_feed,
                product,
            )
        }
        model._is_process_model = False
        model._enforce_acyclic_mixer_energy = False

        result = model._extract_results()

        self.assertEqual(
            [
                row["stream_name"]
                for row in result.raw["material_boundaries"]
                if row["role"] == "feed"
            ],
            ["gas feed", "solvent feed"],
        )
        self.assertEqual(result.kpis["mass_balance_pct"].value, 0.0)

    def test_connectivity_discovers_ejector_motive_and_suction_feeds(self):
        motive_feed = _FallbackStream("motive feed", 100.0)
        heated_motive = _FallbackStream("heated motive", 100.0)
        suction_feed = _FallbackStream("suction feed", 25.0)
        product = _FallbackStream("ejector product", 125.0)
        heater = _FallbackHeater(motive_feed, heated_motive)
        ejector = _FallbackEjector(
            heated_motive,
            suction_feed,
            product,
        )
        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._proc = _FallbackProcess(
            [motive_feed, heater, suction_feed, ejector]
        )
        model._source_bytes = None
        model._units = {}
        model._streams = {
            stream.getName(): stream
            for stream in (
                motive_feed,
                heated_motive,
                suction_feed,
                product,
            )
        }
        model._is_process_model = False
        model._enforce_acyclic_mixer_energy = False

        result = model._extract_results()

        self.assertEqual(
            [
                row["stream_name"]
                for row in result.raw["material_boundaries"]
                if row["role"] == "feed"
            ],
            ["motive feed", "suction feed"],
        )
        self.assertEqual(result.kpis["mass_balance_pct"].value, 0.0)

    def test_private_tank_connectivity_supports_mass_closure(self):
        feed = _FallbackStream("feed", 100.0)
        tank_inlet = _FallbackStream("tank inlet", 100.0)
        gas_product = _FallbackStream("gas product", 70.0)
        liquid_product = _FallbackStream("liquid product", 30.0)
        heater = _FallbackHeater(feed, tank_inlet)
        tank = _FallbackTank(gas_product, liquid_product, tank_inlet)
        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._proc = _FallbackProcess([feed, heater, tank])
        model._source_bytes = None
        model._units = {}
        model._streams = {
            stream.getName(): stream
            for stream in (
                feed,
                tank_inlet,
                gas_product,
                liquid_product,
            )
        }
        model._is_process_model = False
        model._enforce_acyclic_mixer_energy = False

        result = model._extract_results()

        self.assertTrue(result.raw["material_balance_applicable"])
        self.assertEqual(
            result.kpis["mass_balance_pct"].value,
            0.0,
        )
        mass_constraint = next(
            constraint
            for constraint in result.constraints
            if constraint.name == "mass_balance"
        )
        self.assertEqual(mass_constraint.status, "OK")
        component_constraint = next(
            constraint
            for constraint in result.constraints
            if constraint.name == "component_balance"
        )
        self.assertEqual(component_constraint.status, "UNKNOWN")

    def test_unresolved_tank_connectivity_marks_closure_unknown(self):
        feed = _FallbackStream("feed", 100.0)
        tank_inlet = _FallbackStream("tank inlet", 100.0)
        gas_product = _FallbackStream("gas product", 70.0)
        liquid_product = _FallbackStream("liquid product", 30.0)
        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._proc = _FallbackProcess(
            [
                feed,
                _FallbackHeater(feed, tank_inlet),
                _FallbackTank(gas_product, liquid_product),
            ]
        )
        model._source_bytes = None
        model._units = {}
        model._streams = {
            stream.getName(): stream
            for stream in (
                feed,
                tank_inlet,
                gas_product,
                liquid_product,
            )
        }
        model._is_process_model = False
        model._enforce_acyclic_mixer_energy = False

        result = model._extract_results()

        self.assertFalse(result.raw["material_balance_applicable"])
        self.assertNotIn("mass_balance_pct", result.kpis)
        self.assertNotIn("material_product_count", result.kpis)
        self.assertNotIn("material_product_flow_kg_hr", result.kpis)
        self.assertEqual(
            [
                row
                for row in result.raw["material_boundaries"]
                if row["role"] == "product"
            ],
            [],
        )
        self.assertIsNone(
            aggregate_material_balance(result)["imbalance_pct"]
        )
        mass_constraint = next(
            constraint
            for constraint in result.constraints
            if constraint.name == "mass_balance"
        )
        self.assertEqual(mass_constraint.status, "UNKNOWN")
        self.assertIn("storage tank", mass_constraint.detail)

    def test_unresolved_connectivity_skips_named_product_fallback(self):
        gas_product = _FallbackStream("gas product", 70.0)
        liquid_product = _FallbackStream("liquid product", 30.0)
        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._proc = _FallbackProcess(
            [_FallbackTank(gas_product, liquid_product)]
        )
        model._source_bytes = None
        model._units = {}
        model._streams = {
            stream.getName(): stream
            for stream in (gas_product, liquid_product)
        }
        model._is_process_model = False
        model._enforce_acyclic_mixer_energy = False

        result = model._extract_results()

        self.assertFalse(result.raw["material_balance_applicable"])
        self.assertNotIn("material_product_count", result.kpis)
        self.assertNotIn("material_product_flow_kg_hr", result.kpis)
        self.assertEqual(result.raw["material_boundaries"], [])

    def test_connectivity_discovers_feed_after_upstream_equipment(self):
        feed_a = _FallbackStream("feed a", 100.0)
        heated_feed = _FallbackStream("heated feed", 100.0)
        feed_b = _FallbackStream("feed b", 100.0)
        product = _FallbackStream("product", 200.0)
        heater = _FallbackHeater(feed_a, heated_feed)
        mixer = _FallbackTerminalMixer(
            product,
            [heated_feed, feed_b],
        )
        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._proc = _FallbackProcess(
            [feed_a, heater, feed_b, mixer]
        )
        model._source_bytes = None
        model._units = {}
        model._streams = {
            stream.getName(): stream
            for stream in (
                feed_a,
                heated_feed,
                feed_b,
                product,
            )
        }
        model._is_process_model = False
        model._enforce_acyclic_mixer_energy = False

        result = model._extract_results()

        self.assertEqual(
            [
                row["stream_name"]
                for row in result.raw["material_boundaries"]
                if row["role"] == "feed"
            ],
            ["feed a", "feed b"],
        )
        self.assertEqual(result.kpis["mass_balance_pct"].value, 0.0)

    def test_missing_current_component_data_returns_unknown(self):
        feed = _FallbackStream("feed", 100.0)
        product = _FallbackStream("product", 100.0)
        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._proc = _FallbackProcess(
            [feed, _FallbackEquipment(), product]
        )
        model._source_bytes = None
        model._units = {}
        model._streams = {"feed": feed, "product": product}
        model._is_process_model = False
        model._enforce_acyclic_mixer_energy = False

        result = model._extract_results()

        self.assertFalse(result.raw["component_balance_applicable"])
        self.assertEqual(result.raw["component_balances"], [])
        self.assertEqual(component_balance_rows(result), [])
        component_constraint = next(
            constraint
            for constraint in result.constraints
            if constraint.name == "component_balance"
        )
        self.assertEqual(component_constraint.status, "UNKNOWN")
        self.assertIn("unavailable", component_constraint.detail)
        self.assertIn("incomplete", component_constraint.detail)
        self.assertNotIn("component_balance_max_pct", result.kpis)

    def test_uses_solver_kpis_before_legacy_feed_fallback(self):
        result = _result(
            material_feed_count=2,
            material_product_count=1,
            material_feed_flow_kg_hr=100_000,
            material_product_flow_kg_hr=99_500,
        )

        summary = aggregate_material_balance(result)
        self.assertEqual(summary["feed_count"], 2.0)
        self.assertEqual(summary["product_count"], 1.0)
        self.assertEqual(summary["imbalance_pct"], 0.5)
        self.assertEqual(
            solved_feed_flow_kg_hr(result, 60_000.0),
            100_000.0,
        )

        legacy = _result()
        self.assertEqual(
            solved_feed_flow_kg_hr(legacy, 60_000.0),
            60_000.0,
        )

    def test_rejects_malformed_or_non_finite_diagnostics(self):
        invalid_rows = (
            ([{"role": "utility", "stream_name": "x",
               "mass_flow_kg_hr": 1}], "invalid role"),
            ([{"role": "feed", "stream_name": "",
               "mass_flow_kg_hr": 1}], "requires a stream name"),
            ([{"role": "feed", "stream_name": "x",
               "mass_flow_kg_hr": math.nan}], "must be finite"),
            ("not-an-array", "must be an array"),
        )
        for rows, message in invalid_rows:
            with self.subTest(message=message):
                result = SimpleNamespace(
                    raw={"material_boundaries": rows},
                    kpis={},
                )
                with self.assertRaisesRegex(ValueError, message):
                    material_boundary_rows(result)

        with self.assertRaisesRegex(ValueError, "finite and positive"):
            solved_feed_flow_kg_hr(_result(), 0.0)

    def test_unreadable_fallback_stream_does_not_hide_later_boundaries(self):
        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._proc = _FallbackProcess()
        model._source_bytes = None
        model._units = {}
        model._streams = {
            "broken feed": _FallbackStream("broken feed"),
            "backup feed": _FallbackStream("backup feed", 100.0),
            "export product": _FallbackStream("export product", 100.0),
        }
        model._is_process_model = False
        model._enforce_acyclic_mixer_energy = False

        result = model._extract_results()

        self.assertEqual(
            [
                (row["role"], row["stream_name"])
                for row in result.raw["material_boundaries"]
            ],
            [
                ("feed", "backup feed"),
                ("product", "export product"),
            ],
        )
        self.assertEqual(result.kpis["mass_balance_pct"].value, 0.0)

    def test_name_fallback_deduplicates_existing_terminal_boundaries(self):
        zero_feed = _FallbackStream("zero feed", 0.0)
        backup_feed = _FallbackStream("backup feed", 100.0)
        product = _FallbackStream("terminal product", 100.0)
        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._proc = _FallbackProcess(
            [zero_feed, _FallbackEquipment(), product]
        )
        model._source_bytes = None
        model._units = {}
        model._streams = {
            "zero feed": zero_feed,
            "backup feed": backup_feed,
            "terminal product": product,
            "product alias": product,
        }
        model._is_process_model = False
        model._enforce_acyclic_mixer_energy = False

        result = model._extract_results()

        rows = result.raw["material_boundaries"]
        self.assertEqual(
            [
                (row["role"], row["stream_name"])
                for row in rows
            ],
            [
                ("feed", "zero feed"),
                ("product", "terminal product"),
                ("feed", "backup feed"),
            ],
        )
        self.assertEqual(
            aggregate_material_balance(result)["product_flow_kg_hr"],
            100.0,
        )
        self.assertEqual(result.kpis["mass_balance_pct"].value, 0.0)

    def test_no_flow_boundary_rows_match_solver_kpis(self):
        feed = _FallbackStream("feed", 100.0)
        trace_product = _FallbackStream("trace product", 0.005)
        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._proc = _FallbackProcess(
            [feed, _FallbackEquipment(), trace_product]
        )
        model._source_bytes = None
        model._units = {}
        model._streams = {
            "feed": feed,
            "trace product": trace_product,
        }
        model._is_process_model = False
        model._enforce_acyclic_mixer_energy = False

        result = model._extract_results()

        summary = aggregate_material_balance(result)
        self.assertEqual(summary["product_flow_kg_hr"], 0.0)
        self.assertEqual(
            result.kpis["material_product_flow_kg_hr"].value,
            0.0,
        )
        self.assertEqual(
            [
                (
                    row["mass_flow_kg_hr"],
                    row["molar_flow_mol_sec"],
                )
                for row in result.raw["material_boundaries"]
                if row["role"] == "product"
            ],
            [(0.0, 0.0)],
        )
        self.assertEqual(result.kpis["mass_balance_pct"].value, 100.0)

    def test_boundary_counts_include_only_successful_records(self):
        feed = _FallbackStream("feed", 100.0)
        broken_feed = _FallbackStream("broken feed")
        product = _FallbackStream("product", 100.0)
        broken_product = _FallbackStream("broken product")
        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._proc = _FallbackProcess(
            [
                feed,
                broken_feed,
                _FallbackEquipment(),
                product,
                broken_product,
            ]
        )
        model._source_bytes = None
        model._units = {}
        model._streams = {
            "feed": feed,
            "broken feed": broken_feed,
            "product": product,
            "broken product": broken_product,
        }
        model._is_process_model = False
        model._enforce_acyclic_mixer_energy = False

        result = model._extract_results()

        summary = aggregate_material_balance(result)
        self.assertEqual(summary["feed_count"], 1.0)
        self.assertEqual(summary["product_count"], 1.0)
        self.assertEqual(
            result.kpis["material_feed_count"].value,
            1.0,
        )
        self.assertEqual(
            result.kpis["material_product_count"].value,
            1.0,
        )
        self.assertEqual(result.kpis["mass_balance_pct"].value, 0.0)

    def test_distinct_streams_survive_native_hash_collisions(self):
        first_feed = _FallbackStream("feed", 40.0, hash_code=17)
        second_feed = _FallbackStream("feed", 60.0, hash_code=17)
        product = _FallbackStream("product", 100.0, hash_code=23)
        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._proc = _FallbackProcess(
            [
                first_feed,
                second_feed,
                _FallbackEquipment(),
                product,
            ]
        )
        model._source_bytes = None
        model._units = {}
        model._streams = {
            "feed one": first_feed,
            "feed two": second_feed,
            "product": product,
        }
        model._is_process_model = False
        model._enforce_acyclic_mixer_energy = False

        result = model._extract_results()

        summary = aggregate_material_balance(result)
        self.assertEqual(summary["feed_count"], 2.0)
        self.assertEqual(summary["feed_flow_kg_hr"], 100.0)
        self.assertEqual(summary["product_count"], 1.0)
        self.assertEqual(summary["product_flow_kg_hr"], 100.0)
        self.assertEqual(result.kpis["mass_balance_pct"].value, 0.0)

    def test_native_reference_tracking_ignores_value_hash_equality(self):
        from neqsim import jneqsim

        first_fluid = jneqsim.thermo.system.SystemSrkEos(293.15, 45.0)
        second_fluid = jneqsim.thermo.system.SystemSrkEos(308.15, 45.0)
        first_fluid.addComponent("methane", 1.0)
        second_fluid.addComponent("methane", 1.0)
        first_stream = jneqsim.process.equipment.stream.Stream(
            "feed",
            first_fluid,
        )
        second_stream = jneqsim.process.equipment.stream.Stream(
            "feed",
            second_fluid,
        )
        tracker = _MaterialBoundaryIdentityTracker()

        self.assertEqual(
            int(first_stream.hashCode()),
            int(second_stream.hashCode()),
        )
        self.assertTrue(first_stream.equals(second_stream))
        self.assertFalse(tracker.contains("feed", first_stream))
        tracker.add("feed", first_stream)
        self.assertTrue(tracker.contains("feed", first_stream))
        self.assertFalse(tracker.contains("feed", second_stream))
        self.assertFalse(tracker.contains("product", first_stream))
        tracker.add("feed", second_stream)
        self.assertTrue(tracker.contains("feed", second_stream))

    def test_native_private_opaque_unit_inlets_are_discovered(self):
        from neqsim import jneqsim

        fluid = jneqsim.thermo.system.SystemSrkEos(293.15, 45.0)
        fluid.addComponent("water", 1.0)
        stream = jneqsim.process.equipment.stream.Stream("feed", fluid)
        equipment = (
            jneqsim.process.equipment.electrolyzer.Electrolyzer(
                "water electrolyzer",
                stream,
            ),
            jneqsim.process.equipment.electrolyzer.CO2Electrolyzer(
                "co2 electrolyzer",
                stream,
            ),
            jneqsim.process.equipment.tank.Tank(
                "tank",
                stream,
            ),
        )

        for unit in equipment:
            with self.subTest(unit=str(unit.getName())):
                inlets = NeqSimProcessModel._material_inlet_streams(unit)
                self.assertEqual(len(inlets), 1)
                self.assertTrue(inlets[0] == stream)

    def test_reference_tracker_rejects_invalid_roles(self):
        tracker = _MaterialBoundaryIdentityTracker()
        stream = _FallbackStream("feed", 1.0)

        with self.assertRaisesRegex(ValueError, "feed or product"):
            tracker.contains("utility", stream)
        with self.assertRaisesRegex(ValueError, "feed or product"):
            tracker.add("utility", stream)

    def test_reference_tracker_retains_fallback_streams(self):
        tracker = _MaterialBoundaryIdentityTracker()
        stream = _FallbackStream("feed", 1.0)
        stream_reference = weakref.ref(stream)

        tracker.add("feed", stream)
        del stream

        self.assertIsNotNone(stream_reference())
        self.assertTrue(tracker.contains("feed", stream_reference()))


if __name__ == "__main__":
    unittest.main()
