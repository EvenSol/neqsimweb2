"""Native conservation benchmark for the generic ProcessBuilder graph executor."""

from __future__ import annotations

import ast
from contextlib import redirect_stdout
import io
import json
import math
import os
import tempfile
import unittest
import zipfile
from unittest.mock import mock_open, patch

from process_chat.flowsheet_editor import (
    add_catalog_unit,
    connect_graph_ports,
    create_graph_history,
    extend_material_path,
    insert_mixer_on_connection,
    process_unit_property_rows,
    record_graph_history,
    redo_graph_history,
    replace_inline_unit,
    replace_inline_unit_type,
    resize_mixer_inlet_ports,
    resize_separator_inlet_ports,
    resize_splitter_outlet_ports,
    undo_graph_history,
    update_inline_unit_properties,
    update_splitter_allocations,
)
from process_chat.process_builder import ProcessBuilder, _apply_param
from process_chat.chat_tools import (
    ProcessChatSession,
    _classify_build_change,
)
from process_chat.process_model import (
    NeqSimProcessModel,
    ProcessExecutionError,
)
from process_chat.patch_schema import AddUnitOp, InputPatch, Scenario, TargetSpec
from process_chat.scenario_engine import (
    apply_add_units,
    apply_patch_to_model,
    run_scenarios,
)
from process_chat.solver_diagnostics import (
    aggregate_energy_balance,
    aggregate_unit_balances,
    unit_balance_rows,
)


class ExpanderPropertyExtractionTest(unittest.TestCase):
    """Validate explicit solved-property mapping for native expanders."""

    def test_reports_expander_state_and_positive_recovered_power(self):
        class _JavaClass:
            @staticmethod
            def getSimpleName():
                return "Expander"

        class _Expander:
            @staticmethod
            def getClass():
                return _JavaClass()

            @staticmethod
            def getInletPressure():
                return 80.0

            @staticmethod
            def getOutletPressure():
                return 30.0

            @staticmethod
            def getInletTemperature():
                return 303.15

            @staticmethod
            def getOutletTemperature():
                return 243.66

            @staticmethod
            def getIsentropicEfficiency():
                return 0.80

            @staticmethod
            def getPower():
                return -236_788.86

        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._units = {"turbo expander": _Expander()}
        kpis = {}

        model._extract_unit_properties(kpis)

        expected = {
            "inletPressure_bara": (80.0, "bara"),
            "outletPressure_bara": (30.0, "bara"),
            "inletTemperature_K": (303.15, "K"),
            "outletTemperature_K": (243.66, "K"),
            "isentropicEfficiency": (0.80, "[-]"),
            "recoveredPower_kW": (236.78886, "kW"),
        }
        for property_name, (value, unit) in expected.items():
            with self.subTest(property_name=property_name):
                kpi = kpis[f"turbo expander.{property_name}"]
                self.assertAlmostEqual(kpi.value, value, delta=1.0e-10)
                self.assertEqual(kpi.unit, unit)


class PumpParameterApplicationTest(unittest.TestCase):
    """Protect pump efficiency application and replay-script parity."""

    def test_pump_efficiency_uses_native_isentropic_setter(self):
        class _Pump:
            efficiency = None

            def setIsentropicEfficiency(self, value):
                self.efficiency = value

        pump = _Pump()

        _apply_param(pump, "efficiency", 0.75)

        self.assertEqual(pump.efficiency, 0.75)


    def test_native_esp_efficiency_converts_fraction_to_percent(self):
        class _JavaClass:
            @staticmethod
            def getSimpleName():
                return "ESPPump"

        class _ESPPump:
            efficiency = None

            @staticmethod
            def getClass():
                return _JavaClass()

            def setIsentropicEfficiency(self, value):
                self.efficiency = value

        esp_pump = _ESPPump()

        _apply_param(esp_pump, "efficiency", 0.75)

        self.assertEqual(esp_pump.efficiency, 75.0)

    def test_generic_efficiency_still_prefers_equipment_setter(self):
        class _Separator:
            efficiency = None

            def setEfficiency(self, value):
                self.efficiency = value

            def setIsentropicEfficiency(self, value):
                raise AssertionError("generic setter should take precedence")

        separator = _Separator()

        _apply_param(separator, "efficiency", 0.92)

        self.assertEqual(separator.efficiency, 0.92)

    def test_legacy_pump_script_replays_isentropic_efficiency(self):
        builder = ProcessBuilder()
        builder._process_name = "Pump replay regression"
        builder._spec = {
            "name": "Pump replay regression",
            "fluid": {
                "eos_model": "srk",
                "components": {"water": 1.0},
            },
            "process": [
                {"name": "feed", "type": "stream"},
                {
                    "name": "export pump",
                    "type": "pump",
                    "params": {
                        "outlet_pressure_bara": 80.0,
                        "efficiency": 0.75,
                    },
                },
            ],
        }

        script = builder.to_python_script()

        self.assertIn(
            "export_pump.setIsentropicEfficiency(0.75)",
            script,
        )
        self.assertNotIn("export_pump.setEfficiency", script)


class SeparatorDesignApplicationTest(unittest.TestCase):
    """Protect opt-in native separator sizing and its strict design basis."""

    def test_applies_design_basis_only_to_opted_in_separator(self):
        class _Separator:
            gas_load = None
            size_calls = 0

            def setDesignGasLoadFactor(self, value):
                self.gas_load = value

            def autoSize(self):
                self.size_calls += 1

        separator = _Separator()
        designed = ProcessBuilder._apply_requested_mechanical_designs(
            [
                {
                    "id": "inlet-separator",
                    "type": "separator",
                    "params": {
                        "auto_size": True,
                        "design_gas_load_factor_m_per_s": 0.08,
                    },
                }
            ],
            {"inlet-separator": separator},
        )

        self.assertEqual(designed, ["inlet-separator"])
        self.assertEqual(separator.gas_load, 0.08)
        self.assertEqual(separator.size_calls, 1)

        separator.size_calls = 0
        self.assertEqual(
            ProcessBuilder._apply_requested_mechanical_designs(
                [
                    {
                        "id": "inlet-separator",
                        "type": "separator",
                        "params": {
                            "auto_size": False,
                            "design_gas_load_factor_m_per_s": 0.11,
                        },
                    }
                ],
                {"inlet-separator": separator},
            ),
            [],
        )
        self.assertEqual(separator.size_calls, 0)

    def test_rejects_invalid_or_nonseparator_design_requests(self):
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            ProcessBuilder._separator_design_settings(
                {"auto_size": 1}
            )
        with self.assertRaisesRegex(ValueError, "between 0.01 and 1.0"):
            ProcessBuilder._separator_design_settings(
                {
                    "auto_size": True,
                    "design_gas_load_factor_m_per_s": 0.0,
                }
            )
        with self.assertRaisesRegex(ValueError, "only for separator"):
            ProcessBuilder._apply_requested_mechanical_designs(
                [
                    {
                        "id": "compressor",
                        "type": "compressor",
                        "params": {"auto_size": True},
                    }
                ],
                {"compressor": object()},
            )

    def test_legacy_separator_script_replays_design_and_closed_rerun(self):
        builder = ProcessBuilder()
        builder._process_name = "Legacy separator design replay"
        builder._spec = {
            "name": "Legacy separator design replay",
            "fluid": {
                "eos_model": "srk",
                "components": {"methane": 1.0},
            },
            "process": [
                {"name": "feed", "type": "stream"},
                {
                    "name": "inlet separator",
                    "type": "separator",
                    "params": {
                        "auto_size": True,
                        "design_gas_load_factor_m_per_s": 0.08,
                    },
                },
            ],
        }

        script = builder.to_python_script()

        first_run = script.index("process.run()")
        gas_load = script.index(
            "inlet_separator.setDesignGasLoadFactor(0.08)"
        )
        auto_size = script.index("inlet_separator.autoSize()")
        second_run = script.index("process.run()", first_run + 1)
        self.assertLess(first_run, gas_load)
        self.assertLess(gas_load, auto_size)
        self.assertLess(auto_size, second_run)
        self.assertEqual(script.count("process.run()"), 2)


class CompressorChartApplicationTest(unittest.TestCase):
    """Protect strict native compressor-map setup and replay parity."""

    def test_validates_supported_map_settings_and_legacy_boolean_strings(self):
        self.assertEqual(
            ProcessBuilder._compressor_chart_settings(
                {
                    "use_compressor_chart": True,
                    "chart_template": "EXPORT",
                    "chart_num_speeds": 7,
                }
            ),
            (True, "EXPORT", 7),
        )
        self.assertEqual(
            ProcessBuilder._compressor_chart_settings(
                {
                    "use_compressor_chart": "false",
                    "chart_template": "pipeline",
                    "chart_num_speeds": "5",
                }
            ),
            (False, "PIPELINE", 5),
        )

        invalid_cases = (
            (
                {"use_compressor_chart": 1},
                "use_compressor_chart must be boolean",
            ),
            (
                {"chart_template": "NOT_A_NATIVE_TEMPLATE"},
                "chart_template must be one of",
            ),
            (
                {"chart_num_speeds": 5.5},
                "chart_num_speeds must be an integer",
            ),
            (
                {"chart_num_speeds": 2},
                "chart_num_speeds must be between 3 and 12",
            ),
        )
        for params, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    ProcessBuilder._compressor_chart_settings(params)

    def test_legacy_script_replays_map_after_solved_design_point(self):
        builder = ProcessBuilder()
        builder._process_name = "Legacy compressor map replay"
        builder._spec = {
            "name": "Legacy compressor map replay",
            "fluid": {
                "eos_model": "srk",
                "components": {"methane": 1.0},
                "temperature_C": 25.0,
                "pressure_bara": 30.0,
                "total_flow": 10_000.0,
                "flow_unit": "kg/hr",
            },
            "process": [
                {"name": "feed", "type": "stream"},
                {
                    "name": "export compressor",
                    "type": "compressor",
                    "params": {
                        "outlet_pressure_bara": 60.0,
                        "isentropic_efficiency": 0.78,
                        "use_compressor_chart": True,
                        "chart_template": "EXPORT",
                        "chart_num_speeds": 7,
                    },
                },
            ],
        }

        script = builder.to_python_script()

        first_run = script.index("process.run()")
        generator = script.index("CompressorChartGenerator")
        second_run = script.index("process.run()", first_run + 1)
        self.assertLess(first_run, generator)
        self.assertLess(generator, second_run)
        self.assertIn("generateFromTemplate('EXPORT', 7)", script)
        self.assertIn("setSolveSpeed(True)", script)
        self.assertIn("setUsePolytropicCalc(True)", script)
        self.assertEqual(script.count("process.run()"), 2)


class CompressorMapReportingTest(unittest.TestCase):
    """Validate workbook and KPI mapping for native compressor limits."""

    class _JavaClass:
        @staticmethod
        def getSimpleName():
            return "Compressor"

    class _Chart:
        @staticmethod
        def getSpeeds():
            return [2000.0, 2500.0, 3000.0]

    class _Compressor:
        @staticmethod
        def getClass():
            return CompressorMapReportingTest._JavaClass()

        @staticmethod
        def isSolveSpeed():
            return True

        @staticmethod
        def getCompressorChart():
            return CompressorMapReportingTest._Chart()

        @staticmethod
        def getSpeed():
            return 2700.0

        @staticmethod
        def getRatioToMinSpeed():
            return 1.35

        @staticmethod
        def getRatioToMaxSpeed():
            return 0.90

        @staticmethod
        def getDistanceToSurge():
            return 0.25

        @staticmethod
        def getDistanceToStoneWall():
            return 0.40

        @staticmethod
        def getSurgeFlowRate():
            return 300.0

        @staticmethod
        def getSurgeFlowRateMargin():
            return 75.0

        @staticmethod
        def isLowerThanMinSpeed():
            return False

        @staticmethod
        def isHigherThanMaxSpeed():
            return False

        @staticmethod
        def isSurge():
            return False

        @staticmethod
        def isStoneWall():
            return False

    def test_reports_explicit_map_units_to_workbook_and_kpis(self):
        compressor = self._Compressor()
        properties = NeqSimProcessModel._compressor_map_properties(
            compressor
        )
        self.assertEqual(properties["mapSpeedCurveCount"], 3)
        self.assertEqual(properties["mapMaximumSpeed_rpm"], 3000.0)
        self.assertTrue(properties["mapWithinSpeedRange"])
        self.assertFalse(properties["mapInSurge"])

        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._units = {"export compressor": compressor}
        model._unit_ps_name = {"export compressor": "main"}
        model._report_unit_duty_lookup = lambda: {}
        workbook_unit = model.list_units()[0]
        self.assertEqual(workbook_unit.properties, properties)

        kpis = {}
        model._extract_unit_properties(kpis, {})
        expected_units = {
            "mapEnabled": "-",
            "mapSpeedCurveCount": "curves",
            "mapMinimumSpeed_rpm": "rpm",
            "mapMaximumSpeed_rpm": "rpm",
            "mapOperatingSpeed_rpm": "rpm",
            "mapSpeedRatioToMinimum": "-",
            "mapSpeedRatioToMaximum": "-",
            "mapDistanceToSurgeFraction": "-",
            "mapDistanceToStoneWallFraction": "-",
            "mapSurgeFlowRate_m3_per_hr": "m3/hr",
            "mapSurgeFlowMargin_m3_per_hr": "m3/hr",
            "mapWithinSpeedRange": "-",
            "mapInSurge": "-",
            "mapInStoneWall": "-",
        }
        for property_name, unit in expected_units.items():
            with self.subTest(property_name=property_name):
                self.assertEqual(
                    kpis[f"export compressor.{property_name}"].unit,
                    unit,
                )


class NativeCompressorMapBenchmarkTest(unittest.TestCase):
    """Benchmark a fixed native compressor map at nearby flow points."""

    @staticmethod
    def _build_case():
        units, compressor_id = add_catalog_unit(
            [],
            "compressor",
            "map compressor",
        )
        units = update_inline_unit_properties(
            units,
            compressor_id,
            {
                "outlet_pressure_bara": 60.0,
                "isentropic_efficiency": 0.78,
                "use_compressor_chart": True,
                "chart_template": "PIPELINE",
                "chart_num_speeds": 7,
            },
        )
        graph_spec = {
            "name": "Native compressor map benchmark",
            "units": units,
            "connections": [
                {
                    "id": "feed-to-map-compressor",
                    "type": "material",
                    "source": {
                        "kind": "inlet",
                        "id": "feed",
                        "port": "out",
                    },
                    "target": {
                        "kind": "unit",
                        "id": compressor_id,
                        "port": "in",
                    },
                }
            ],
        }
        inlet_specs = [
            {
                "inlet_id": "feed",
                "name": "feed",
                "fluid_spec": {
                    "eos_model": "srk",
                    "mixing_rule": 2,
                    "components": {
                        "methane": 0.90,
                        "ethane": 0.06,
                        "propane": 0.03,
                        "n-butane": 0.01,
                    },
                    "composition_basis": "mole_fraction",
                    "temperature_C": 25.0,
                    "pressure_bara": 30.0,
                    "total_flow": 10_000.0,
                    "flow_unit": "kg/hr",
                },
            }
        ]
        builder = ProcessBuilder()
        model = builder.build_acyclic_graph(
            graph_spec,
            inlet_specs,
            ["feed", compressor_id],
        )
        return builder, model, graph_spec

    def test_native_map_closes_and_stays_inside_limits_nearby(self):
        builder, model, graph_spec = self._build_case()
        compressor = model.get_unit("map compressor")
        feed = model.get_unit("feed")

        baseline = model.run()
        baseline_properties = next(
            unit.properties
            for unit in model.list_units()
            if unit.name == "map compressor"
        )
        baseline_speed = baseline_properties["mapOperatingSpeed_rpm"]

        feed.setFlowRate(9_500.0, "kg/hr")
        nearby = model.run()
        nearby_properties = next(
            unit.properties
            for unit in model.list_units()
            if unit.name == "map compressor"
        )

        for label, result, properties in (
            ("baseline", baseline, baseline_properties),
            ("nearby", nearby, nearby_properties),
        ):
            with self.subTest(label=label):
                self.assertTrue(properties["mapEnabled"])
                self.assertEqual(properties["mapSpeedCurveCount"], 7)
                self.assertTrue(properties["mapWithinSpeedRange"])
                self.assertFalse(properties["mapInSurge"])
                self.assertFalse(properties["mapInStoneWall"])
                self.assertGreater(
                    properties["mapDistanceToSurgeFraction"],
                    0.0,
                )
                self.assertGreater(
                    properties["mapDistanceToStoneWallFraction"],
                    0.0,
                )
                for balance_name in (
                    "mass_balance_pct",
                    "component_balance_max_pct",
                    "energy_balance_pct",
                    "unit_mass_balance_max_pct",
                    "unit_energy_balance_max_pct",
                ):
                    self.assertLess(result.kpis[balance_name].value, 1.0e-6)
                map_constraint = next(
                    constraint
                    for constraint in result.constraints
                    if constraint.name == "compressor_map.map compressor"
                )
                self.assertEqual(map_constraint.status, "OK")

        self.assertLess(
            nearby_properties["mapOperatingSpeed_rpm"],
            baseline_speed,
        )
        self.assertAlmostEqual(
            float(compressor.getOutletPressure()),
            60.0,
            delta=1.0e-9,
        )
        self.assertEqual(
            json.loads(json.dumps(graph_spec, allow_nan=False)),
            graph_spec,
        )
        self.assertIn(
            "Running closed compressor-map rerun for: map-compressor",
            builder.build_log,
        )
        print(
            "native compressor map benchmark:",
            f"baseline_speed={baseline_speed:.6f} rpm",
            (
                "baseline_surge_distance="
                f"{baseline_properties['mapDistanceToSurgeFraction']:.6f}"
            ),
            (
                "nearby_speed="
                f"{nearby_properties['mapOperatingSpeed_rpm']:.6f} rpm"
            ),
            (
                "nearby_stonewall_distance="
                f"{nearby_properties['mapDistanceToStoneWallFraction']:.6f}"
            ),
        )


class HeatExchangerPropertyExtractionTest(unittest.TestCase):
    """Validate explicit solved-property mapping for two-sided exchangers."""

    def test_workbook_reports_hot_cold_conditions_and_duty_closure(self):
        class _JavaClass:
            @staticmethod
            def getSimpleName():
                return "HeatExchanger"

        class _Fluid:
            def __init__(self, enthalpy_w):
                self.enthalpy_w = enthalpy_w
                self.initialized = False

            def init(self, level):
                if level != 3:
                    raise AssertionError(level)
                self.initialized = True

            def getEnthalpy(self):
                if not self.initialized:
                    raise AssertionError("enthalpy requires level-3 init")
                return self.enthalpy_w

        class _Stream:
            def __init__(
                self,
                temperature_c,
                pressure_bara,
                flow_kg_hr,
                enthalpy_w,
                calculation_identifier="solved-calculation",
            ):
                self.temperature_c = temperature_c
                self.pressure_bara = pressure_bara
                self.flow_kg_hr = flow_kg_hr
                self.fluid = _Fluid(enthalpy_w)
                self.calculation_identifier = calculation_identifier

            def getCalculationIdentifier(self):
                return self.calculation_identifier

            def getTemperature(self, unit):
                if unit != "C":
                    raise AssertionError(unit)
                return self.temperature_c

            def getPressure(self, unit):
                if unit != "bara":
                    raise AssertionError(unit)
                return self.pressure_bara

            def getFlowRate(self, unit):
                if unit != "kg/hr":
                    raise AssertionError(unit)
                return self.flow_kg_hr

            def getFluid(self):
                return self.fluid

        class _HeatExchanger:
            inlet_streams = [
                _Stream(120.0, 50.0, 50_000.0, 3_300_000.0),
                _Stream(20.0, 49.5, 40_000.0, -150_000.0),
            ]
            outlet_streams = [
                _Stream(53.0, 50.0, 50_000.0, 900_000.0),
                _Stream(103.5, 49.5, 40_000.0, 2_250_000.0),
            ]

            @staticmethod
            def getClass():
                return _JavaClass()

            @staticmethod
            def getCalculationIdentifier():
                return "solved-calculation"

            @classmethod
            def getInStream(cls, index):
                return cls.inlet_streams[index]

            @classmethod
            def getOutStream(cls, index):
                return cls.outlet_streams[index]

            @staticmethod
            def getUAvalue():
                return 100_000.0

            @staticmethod
            def getDuty():
                return 2_400_000.0

            @staticmethod
            def getApproachTemperature():
                return 16.5

            @staticmethod
            def getThermalEffectiveness():
                return 0.83

        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._units = {"cross exchanger": _HeatExchanger()}
        model._unit_ps_name = {"cross exchanger": "main"}
        model._direct_unit_run_provenance = {}
        model._heat_exchanger_state_snapshots = {}
        model._capture_heat_exchanger_state_snapshots()

        properties = model.list_units()[0].properties

        expected = {
            "hotInletTemperature_C": 120.0,
            "hotOutletTemperature_C": 53.0,
            "coldInletTemperature_C": 20.0,
            "coldOutletTemperature_C": 103.5,
            "hotInletPressure_bara": 50.0,
            "hotOutletPressure_bara": 50.0,
            "coldInletPressure_bara": 49.5,
            "coldOutletPressure_bara": 49.5,
            "hotInletFlow_kg_hr": 50_000.0,
            "hotOutletFlow_kg_hr": 50_000.0,
            "coldInletFlow_kg_hr": 40_000.0,
            "coldOutletFlow_kg_hr": 40_000.0,
            "UA_W_K": 100_000.0,
            "heatTransferDuty_kW": 2_400.0,
            "approachTemperature_K": 16.5,
            "thermalEffectiveness": 0.83,
            "hotSideDuty_kW": 2_400.0,
            "coldSideDuty_kW": 2_400.0,
            "dutyClosure_kW": 0.0,
            "dutyClosure_pct": 0.0,
        }
        for property_name, value in expected.items():
            with self.subTest(property_name=property_name):
                self.assertAlmostEqual(
                    properties[property_name],
                    value,
                    delta=1.0e-12,
                )

        kpis = {}
        model._extract_unit_properties(kpis)
        expected_units = {
            "hotInletTemperature_C": "°C",
            "hotOutletTemperature_C": "°C",
            "coldInletTemperature_C": "°C",
            "coldOutletTemperature_C": "°C",
            "hotInletPressure_bara": "bara",
            "hotOutletPressure_bara": "bara",
            "coldInletPressure_bara": "bara",
            "coldOutletPressure_bara": "bara",
            "hotInletFlow_kg_hr": "kg/hr",
            "hotOutletFlow_kg_hr": "kg/hr",
            "coldInletFlow_kg_hr": "kg/hr",
            "coldOutletFlow_kg_hr": "kg/hr",
            "UA_W_K": "W/K",
            "heatTransferDuty_kW": "kW",
            "approachTemperature_K": "K",
            "thermalEffectiveness": "[-]",
            "hotSideDuty_kW": "kW",
            "coldSideDuty_kW": "kW",
            "dutyClosure_kW": "kW",
            "dutyClosure_pct": "%",
        }
        for property_name, unit in expected_units.items():
            with self.subTest(kpi_property=property_name):
                kpi = kpis[f"cross exchanger.{property_name}"]
                self.assertAlmostEqual(
                    kpi.value,
                    expected[property_name],
                    delta=1.0e-12,
                )
                self.assertEqual(kpi.unit, unit)

        class _ReversedHeatExchanger(_HeatExchanger):
            inlet_streams = list(reversed(_HeatExchanger.inlet_streams))
            outlet_streams = list(reversed(_HeatExchanger.outlet_streams))

            @staticmethod
            def getDuty():
                return -2_400_000.0

            @staticmethod
            def getApproachTemperature():
                return -33.0

        reversed_properties = (
            NeqSimProcessModel._heat_exchanger_operating_properties(
                _ReversedHeatExchanger()
            )
        )
        self.assertEqual(reversed_properties["hotInletTemperature_C"], 120.0)
        self.assertEqual(reversed_properties["coldInletTemperature_C"], 20.0)
        self.assertEqual(reversed_properties["hotSideDuty_kW"], 2_400.0)
        self.assertEqual(reversed_properties["coldSideDuty_kW"], 2_400.0)
        self.assertEqual(reversed_properties["heatTransferDuty_kW"], 2_400.0)
        self.assertEqual(reversed_properties["approachTemperature_K"], 16.5)

        class _CoCurrentHeatExchanger(_HeatExchanger):
            @staticmethod
            def getFlowArrangement():
                return "co-current"

        co_current_properties = (
            NeqSimProcessModel._heat_exchanger_operating_properties(
                _CoCurrentHeatExchanger()
            )
        )
        self.assertEqual(
            co_current_properties["approachTemperature_K"],
            -50.5,
        )

        class _IncompleteHeatExchanger(_HeatExchanger):
            @classmethod
            def getOutStream(cls, index):
                if index == 1:
                    raise RuntimeError("cold outlet is not solved")
                return super().getOutStream(index)

        self.assertEqual(
            NeqSimProcessModel._heat_exchanger_operating_properties(
                _IncompleteHeatExchanger()
            ),
            {},
        )

        class _NeverRunHeatExchanger(_HeatExchanger):
            @staticmethod
            def getCalculationIdentifier():
                return None

        self.assertEqual(
            NeqSimProcessModel._heat_exchanger_operating_properties(
                _NeverRunHeatExchanger()
            ),
            {},
        )

        class _StaleBoundaryHeatExchanger(_HeatExchanger):
            inlet_streams = list(_HeatExchanger.inlet_streams)
            inlet_streams[0] = _Stream(
                120.0,
                50.0,
                50_000.0,
                3_300_000.0,
                calculation_identifier="newer-inlet-calculation",
            )

        self.assertEqual(
            NeqSimProcessModel._heat_exchanger_operating_properties(
                _StaleBoundaryHeatExchanger()
            ),
            {},
        )

        class _DirectRunHeatExchanger(_HeatExchanger):
            inlet_streams = [
                _Stream(
                    stream.temperature_c,
                    stream.pressure_bara,
                    stream.flow_kg_hr,
                    stream.fluid.enthalpy_w,
                    calculation_identifier=f"direct-inlet-{index}",
                )
                for index, stream in enumerate(_HeatExchanger.inlet_streams)
            ]

        direct_provenance = (
            "solved-calculation",
            ("direct-inlet-0", "direct-inlet-1"),
            ("solved-calculation", "solved-calculation"),
        )
        direct_properties = (
            NeqSimProcessModel._heat_exchanger_operating_properties(
                _DirectRunHeatExchanger(),
                direct_provenance,
            )
        )
        self.assertEqual(direct_properties["heatTransferDuty_kW"], 2_400.0)

        direct_model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        direct_model._units = {
            "direct exchanger": _DirectRunHeatExchanger()
        }
        direct_model._unit_ps_name = {"direct exchanger": "main"}
        direct_model._direct_unit_run_provenance = {}
        direct_model._heat_exchanger_state_snapshots = {}
        direct_model.record_direct_unit_run("direct exchanger")
        self.assertEqual(
            direct_model._direct_unit_run_provenance[
                "direct exchanger"
            ],
            direct_provenance,
        )
        self.assertEqual(
            direct_model.list_units()[0].properties[
                "heatTransferDuty_kW"
            ],
            2_400.0,
        )

        qualified_model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        qualified_model._units = {
            "main/direct exchanger": _DirectRunHeatExchanger()
        }
        qualified_model._unit_ps_name = {
            "main/direct exchanger": "main"
        }
        qualified_model._direct_unit_run_provenance = {}
        qualified_model._heat_exchanger_state_snapshots = {}
        qualified_model.record_direct_unit_run("direct exchanger")
        self.assertEqual(
            qualified_model._direct_unit_run_provenance[
                "main/direct exchanger"
            ],
            direct_provenance,
        )

        auto_capture_model = NeqSimProcessModel.__new__(
            NeqSimProcessModel
        )
        auto_capture_model._units = {
            "direct exchanger": _DirectRunHeatExchanger()
        }
        auto_capture_model._unit_ps_name = {
            "direct exchanger": "main"
        }
        auto_capture_model._direct_unit_run_provenance = {}
        auto_capture_model._heat_exchanger_state_snapshots = {}
        auto_capture_model._capture_heat_exchanger_state_snapshots(
            allow_direct_runs=True
        )
        self.assertEqual(
            auto_capture_model._direct_unit_run_provenance[
                "direct exchanger"
            ],
            direct_provenance,
        )
        self.assertEqual(
            auto_capture_model.list_units()[0].properties[
                "heatTransferDuty_kW"
            ],
            2_400.0,
        )
        auto_capture_model._capture_heat_exchanger_state_snapshots()
        self.assertNotIn(
            "direct exchanger",
            auto_capture_model._direct_unit_run_provenance,
        )
        self.assertNotIn(
            "heatTransferDuty_kW",
            auto_capture_model.list_units()[0].properties,
        )

        class _ProcessRunHeatExchanger(_HeatExchanger):
            inlet_streams = [
                _Stream(
                    stream.temperature_c,
                    stream.pressure_bara,
                    stream.flow_kg_hr,
                    stream.fluid.enthalpy_w,
                    calculation_identifier=f"process-inlet-{index}",
                )
                for index, stream in enumerate(_HeatExchanger.inlet_streams)
            ]
            outlet_streams = [
                _Stream(
                    stream.temperature_c,
                    stream.pressure_bara,
                    stream.flow_kg_hr,
                    stream.fluid.enthalpy_w,
                    calculation_identifier=f"process-outlet-{index}",
                )
                for index, stream in enumerate(_HeatExchanger.outlet_streams)
            ]

        process_run_model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        process_run_model._units = {
            "process exchanger": _ProcessRunHeatExchanger()
        }
        process_run_model._unit_ps_name = {
            "process exchanger": "main"
        }
        process_run_model._direct_unit_run_provenance = {}
        process_run_model._heat_exchanger_state_snapshots = {}
        process_run_model._capture_heat_exchanger_state_snapshots()
        self.assertNotIn(
            "heatTransferDuty_kW",
            process_run_model.list_units()[0].properties,
        )
        process_run_model._capture_heat_exchanger_state_snapshots(
            trust_completed_process_run=True
        )
        self.assertEqual(
            process_run_model.list_units()[0].properties[
                "heatTransferDuty_kW"
            ],
            2_400.0,
        )

        class _MutableSolvedHeatExchanger(_HeatExchanger):
            inlet_streams = [
                _Stream(120.0, 50.0, 50_000.0, 3_300_000.0),
                _Stream(20.0, 49.5, 40_000.0, -150_000.0),
            ]
            outlet_streams = [
                _Stream(53.0, 50.0, 50_000.0, 900_000.0),
                _Stream(103.5, 49.5, 40_000.0, 2_250_000.0),
            ]

        mutable_model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        mutable_model._units = {
            "mutable exchanger": _MutableSolvedHeatExchanger()
        }
        mutable_model._unit_ps_name = {"mutable exchanger": "main"}
        mutable_model._direct_unit_run_provenance = {}
        mutable_model._heat_exchanger_state_snapshots = {}
        mutable_model._capture_heat_exchanger_state_snapshots()
        self.assertEqual(
            mutable_model.list_units()[0].properties[
                "heatTransferDuty_kW"
            ],
            2_400.0,
        )
        _MutableSolvedHeatExchanger.inlet_streams[
            0
        ].temperature_c = 110.0
        self.assertNotIn(
            "heatTransferDuty_kW",
            mutable_model.list_units()[0].properties,
        )

        class _MutableSettingsHeatExchanger(_HeatExchanger):
            ua_W_K = 100_000.0
            flow_arrangement = "concentric tube counterflow"
            thermal_effectiveness = 0.83
            delta_T_K = 5.0
            use_delta_T = False

            @classmethod
            def getUAvalue(cls):
                return cls.ua_W_K

            @classmethod
            def getFlowArrangement(cls):
                return cls.flow_arrangement

            @classmethod
            def getThermalEffectiveness(cls):
                return cls.thermal_effectiveness

            @classmethod
            def getDeltaT(cls):
                return cls.delta_T_K

            @classmethod
            def isUseDeltaT(cls):
                return cls.use_delta_T

        settings_model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        settings_model._units = {
            "settings exchanger": _MutableSettingsHeatExchanger()
        }
        settings_model._unit_ps_name = {"settings exchanger": "main"}
        settings_model._direct_unit_run_provenance = {}
        settings_model._heat_exchanger_state_snapshots = {}
        settings_model._capture_heat_exchanger_state_snapshots()
        _MutableSettingsHeatExchanger.ua_W_K = 200_000.0
        self.assertNotIn(
            "heatTransferDuty_kW",
            settings_model.list_units()[0].properties,
        )
        _MutableSettingsHeatExchanger.thermal_effectiveness = 0.83
        settings_model._capture_heat_exchanger_state_snapshots()
        _MutableSettingsHeatExchanger.use_delta_T = True
        self.assertNotIn(
            "heatTransferDuty_kW",
            settings_model.list_units()[0].properties,
        )
        _MutableSettingsHeatExchanger.ua_W_K = 100_000.0
        settings_model._capture_heat_exchanger_state_snapshots()
        _MutableSettingsHeatExchanger.flow_arrangement = "co-current"
        self.assertNotIn(
            "heatTransferDuty_kW",
            settings_model.list_units()[0].properties,
        )
        _MutableSettingsHeatExchanger.flow_arrangement = (
            "concentric tube counterflow"
        )
        settings_model._capture_heat_exchanger_state_snapshots()
        _MutableSettingsHeatExchanger.thermal_effectiveness = 0.25
        self.assertNotIn(
            "heatTransferDuty_kW",
            settings_model.list_units()[0].properties,
        )

        class _StaleDualInletHeatExchanger(_DirectRunHeatExchanger):
            inlet_streams = [
                _Stream(
                    120.0,
                    50.0,
                    50_000.0,
                    4_000_000.0,
                    calculation_identifier="new-hot-calculation",
                ),
                _Stream(
                    20.0,
                    49.5,
                    40_000.0,
                    -900_000.0,
                    calculation_identifier="new-cold-calculation",
                ),
            ]

        self.assertEqual(
            NeqSimProcessModel._heat_exchanger_operating_properties(
                _StaleDualInletHeatExchanger(),
                direct_provenance,
            ),
            {},
        )

        class _StaleOutletHeatExchanger(_HeatExchanger):
            outlet_streams = list(_HeatExchanger.outlet_streams)
            outlet_streams[0] = _Stream(
                53.0,
                50.0,
                50_000.0,
                900_000.0,
                calculation_identifier="older-outlet-calculation",
            )

        self.assertEqual(
            NeqSimProcessModel._heat_exchanger_operating_properties(
                _StaleOutletHeatExchanger()
            ),
            {},
        )

        class _CrossedTemperatureHeatExchanger(_HeatExchanger):
            outlet_streams = [
                _HeatExchanger.outlet_streams[0],
                _Stream(125.0, 49.5, 40_000.0, 2_250_000.0),
            ]

        crossed_properties = (
            NeqSimProcessModel._heat_exchanger_operating_properties(
                _CrossedTemperatureHeatExchanger()
            )
        )
        self.assertEqual(crossed_properties["approachTemperature_K"], -5.0)

        class _BothSidesLoseHeatExchanger(_HeatExchanger):
            inlet_streams = [
                _HeatExchanger.inlet_streams[0],
                _Stream(20.0, 49.5, 40_000.0, 2_250_000.0),
            ]
            outlet_streams = [
                _HeatExchanger.outlet_streams[0],
                _Stream(103.5, 49.5, 40_000.0, -150_000.0),
            ]

        both_lose_properties = (
            NeqSimProcessModel._heat_exchanger_operating_properties(
                _BothSidesLoseHeatExchanger()
            )
        )
        self.assertEqual(both_lose_properties["hotSideDuty_kW"], 2_400.0)
        self.assertEqual(both_lose_properties["coldSideDuty_kW"], 2_400.0)
        self.assertEqual(both_lose_properties["dutyClosure_kW"], 4_800.0)
        self.assertEqual(both_lose_properties["dutyClosure_pct"], 200.0)


class ProcessRunCompletionTest(unittest.TestCase):
    """Distinguish completed idle calculations from execution failures."""

    def test_completed_zero_energy_process_is_successful(self):
        class _JavaClass:
            def __init__(self, simple_name):
                self.simple_name = simple_name

            def getSimpleName(self):
                return self.simple_name

        class _Unit:
            def __init__(self, simple_name):
                self.java_class = _JavaClass(simple_name)

            def getClass(self):
                return self.java_class

            @staticmethod
            def getDuty():
                return 0.0

        class _Process:
            def __init__(self):
                self.run_count = 0
                self.units = [
                    _Unit("Stream"),
                    _Unit("Heater"),
                    _Unit("Stream"),
                ]

            def getUnitOperations(self):
                return self.units

            def run(self):
                self.run_count += 1

        process = _Process()

        self.assertTrue(
            NeqSimProcessModel._run_until_converged(
                process,
                max_runs=3,
                timeout_ms=0,
            )
        )
        self.assertEqual(process.run_count, 3)

    def test_failed_warmups_override_earlier_zero_energy_success(self):
        class _JavaClass:
            @staticmethod
            def getSimpleName():
                return "Heater"

        class _Unit:
            @staticmethod
            def getClass():
                return _JavaClass()

            @staticmethod
            def getDuty():
                return 0.0

        class _Process:
            def __init__(self):
                self.run_count = 0
                self.units = [_Unit(), _Unit(), _Unit()]

            def getUnitOperations(self):
                return self.units

            def run(self):
                self.run_count += 1
                if self.run_count > 1:
                    raise RuntimeError("warm-up failed")

        process = _Process()

        self.assertFalse(
            NeqSimProcessModel._run_until_converged(
                process,
                max_runs=3,
                timeout_ms=0,
            )
        )
        self.assertEqual(process.run_count, 3)

    def test_native_worker_failure_is_not_successful(self):
        from neqsim import jneqsim

        fluid = jneqsim.thermo.system.SystemSrkEos(300.15, 50.0)
        fluid.addComponent("methane", 1.0)
        feed = jneqsim.process.equipment.stream.Stream("feed", fluid)
        exchanger = (
            jneqsim.process.equipment.heatexchanger.HeatExchanger(
                "incomplete exchanger"
            )
        )
        process = jneqsim.process.processmodel.ProcessSystem(
            "worker failure regression"
        )
        process.add(feed)
        process.add(exchanger)

        self.assertFalse(
            NeqSimProcessModel._run_until_converged(
                process,
                timeout_ms=30_000,
            )
        )
        self.assertFalse(bool(process.getRunStatus().isSuccess()))
        self.assertIn(
            "inStream[0]",
            str(process.getRunStatus().getFailedUnitError()),
        )


class SplitterPropertyExtractionTest(unittest.TestCase):
    """Validate solved splitter allocations and explicit flow closure."""

    def test_native_count_drives_index_diagram_and_summary_past_ten_branches(self):
        class _JavaClass:
            def __init__(self, simple_name):
                self.simple_name = simple_name

            def getSimpleName(self):
                return self.simple_name

            def getName(self):
                return f"neqsim.process.equipment.{self.simple_name}"

        class _Stream:
            def __init__(self, name, native_id, flow_kg_hr):
                self.name = name
                self.native_id = native_id
                self.flow_kg_hr = flow_kg_hr

            def getName(self):
                return self.name

            def hashCode(self):
                return self.native_id

            def getTemperature(self, unit):
                if unit != "C":
                    raise AssertionError(unit)
                return 25.0

            def getPressure(self, unit):
                if unit != "bara":
                    raise AssertionError(unit)
                return 10.0

            def getFlowRate(self, unit):
                if unit != "kg/hr":
                    raise AssertionError(unit)
                return self.flow_kg_hr

        inlet = _Stream("feed", 1, 120.0)
        branches = [
            _Stream(f"branch {index}", 100 + index, 10.0)
            for index in range(12)
        ]

        class _Splitter:
            @staticmethod
            def getName():
                return "wide split"

            @staticmethod
            def getClass():
                return _JavaClass("Splitter")

            @staticmethod
            def getSplitNumber():
                return len(branches)

            @staticmethod
            def getInletStream():
                return inlet

            @staticmethod
            def getSplitStream(index):
                return branches[index]

            @staticmethod
            def getSplitFactor(_index):
                return 1.0 / len(branches)

        class _Sink:
            @staticmethod
            def getName():
                return "branch 11 sink"

            @staticmethod
            def getClass():
                return _JavaClass("Sink")

            @staticmethod
            def getInletStream():
                return branches[11]

        class _Process:
            @staticmethod
            def getName():
                return "wide splitter process"

            @staticmethod
            def getClass():
                return _JavaClass("ProcessSystem")

            @staticmethod
            def getUnitOperations():
                return [_Splitter(), _Sink()]

        model = NeqSimProcessModel(_Process())

        self.assertIn("wide split.branch 11", model._streams)
        diagram = model.get_diagram_dot(show_stream_values=False)
        self.assertIn('n0 -> n1 [label="branch 11"]', diagram)
        summary = model.get_model_summary()
        self.assertIn("OUT (SPLIT 11): branch 11", summary)

    def test_reports_native_branch_fractions_and_workbook_properties(self):
        class _JavaClass:
            @staticmethod
            def getSimpleName():
                return "Splitter"

        class _Stream:
            def __init__(self, flow_kg_hr):
                self.flow_kg_hr = flow_kg_hr

            def getFlowRate(self, unit):
                if unit != "kg/hr":
                    raise AssertionError(unit)
                return self.flow_kg_hr

        class _Splitter:
            streams = [_Stream(25.0), _Stream(75.0)]

            @staticmethod
            def getClass():
                return _JavaClass()

            @staticmethod
            def getSplitNumber():
                return 2

            @staticmethod
            def getInletStream():
                return _Stream(100.0)

            @classmethod
            def getSplitStream(cls, index):
                return cls.streams[index]

            @staticmethod
            def getSplitFactor(index):
                return (0.25, 0.75)[index]

        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._units = {"product split": _Splitter()}
        model._unit_ps_name = {"product split": "main"}
        kpis = {}

        model._extract_unit_properties(kpis)

        expected = {
            "inletFlow_kg_hr": (100.0, "kg/hr"),
            "branchCount": (2.0, "[-]"),
            "solvedBranchCount": (2.0, "[-]"),
            "branch0Flow_kg_hr": (25.0, "kg/hr"),
            "branch1Flow_kg_hr": (75.0, "kg/hr"),
            "branch0Fraction": (0.25, "[-]"),
            "branch1Fraction": (0.75, "[-]"),
            "configuredBranch0Fraction": (0.25, "[-]"),
            "configuredBranch1Fraction": (0.75, "[-]"),
            "outletFlowTotal_kg_hr": (100.0, "kg/hr"),
            "flowClosure_kg_hr": (0.0, "kg/hr"),
            "flowClosure_pct": (0.0, "%"),
            "splitFractionSum": (1.0, "[-]"),
        }
        for property_name, (value, unit) in expected.items():
            with self.subTest(property_name=property_name):
                kpi = kpis[f"product split.{property_name}"]
                self.assertAlmostEqual(kpi.value, value, delta=1.0e-12)
                self.assertEqual(kpi.unit, unit)

        workbook_properties = model.list_units()[0].properties
        for property_name, (value, _unit) in expected.items():
            with self.subTest(workbook_property=property_name):
                self.assertAlmostEqual(
                    workbook_properties[property_name],
                    value,
                    delta=1.0e-12,
                )
        self.assertEqual(
            kpis["product split.splitStream0_flow_kg_hr"].value,
            25.0,
        )
        self.assertEqual(
            kpis["product split.splitStream0_flow_kg_hr"].unit,
            "kg/hr",
        )

    def test_preserves_native_topology_count_when_one_branch_is_unreadable(self):
        class _Stream:
            def __init__(self, flow_kg_hr):
                self.flow_kg_hr = flow_kg_hr

            def getFlowRate(self, unit):
                if unit != "kg/hr":
                    raise AssertionError(unit)
                return self.flow_kg_hr

        class _PartlyReadableSplitter:
            @staticmethod
            def getSplitNumber():
                return 3

            @staticmethod
            def getInletStream():
                return _Stream(100.0)

            @staticmethod
            def getSplitStream(index):
                if index == 2:
                    raise RuntimeError("native branch unavailable")
                return _Stream(50.0)

            @staticmethod
            def getSplitFactor(index):
                return (0.5, 0.5, 0.0)[index]

        properties = NeqSimProcessModel._splitter_operating_properties(
            _PartlyReadableSplitter()
        )

        self.assertEqual(properties["branchCount"], 3.0)
        self.assertEqual(properties["solvedBranchCount"], 2.0)
        self.assertEqual(properties["configuredBranch2Fraction"], 0.0)
        self.assertNotIn("outletFlowTotal_kg_hr", properties)
        self.assertNotIn("branch0Fraction", properties)
        self.assertNotIn("branch1Fraction", properties)
        self.assertNotIn("flowClosure_pct", properties)

    def test_zero_flow_splitter_withholds_undefined_fraction_sum(self):
        class _Stream:
            @staticmethod
            def getFlowRate(unit):
                if unit != "kg/hr":
                    raise AssertionError(unit)
                return 0.0

        class _ZeroFlowSplitter:
            @staticmethod
            def getSplitNumber():
                return 2

            @staticmethod
            def getInletStream():
                return _Stream()

            @staticmethod
            def getSplitStream(_index):
                return _Stream()

            @staticmethod
            def getSplitFactor(index):
                return (0.5, 0.5)[index]

        properties = NeqSimProcessModel._splitter_operating_properties(
            _ZeroFlowSplitter()
        )

        self.assertEqual(properties["flowClosure_pct"], 0.0)
        self.assertNotIn("splitFractionSum", properties)

        class _NearZeroFlowSplitter(_ZeroFlowSplitter):
            @staticmethod
            def getInletStream():
                class _NearZeroStream:
                    @staticmethod
                    def getFlowRate(unit):
                        if unit != "kg/hr":
                            raise AssertionError(unit)
                        return 1.0e-12

                return _NearZeroStream()

        near_zero_properties = (
            NeqSimProcessModel._splitter_operating_properties(
                _NearZeroFlowSplitter()
            )
        )
        self.assertNotIn("branch0Fraction", near_zero_properties)
        self.assertNotIn("branch1Fraction", near_zero_properties)
        self.assertNotIn("splitFractionSum", near_zero_properties)

    def test_legacy_splitter_keeps_flow_kpis_without_claiming_topology(self):
        class _JavaClass:
            @staticmethod
            def getSimpleName():
                return "Splitter"

        class _Stream:
            def __init__(self, flow_kg_hr):
                self.flow_kg_hr = flow_kg_hr

            def getFlowRate(self, unit):
                if unit != "kg/hr":
                    raise AssertionError(unit)
                return self.flow_kg_hr

        class _LegacySplitter:
            streams = [_Stream(25.0), _Stream(75.0)]

            @staticmethod
            def getClass():
                return _JavaClass()

            @staticmethod
            def getInletStream():
                return _Stream(100.0)

            @classmethod
            def getSplitStream(cls, index):
                return cls.streams[index]

        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._units = {"legacy split": _LegacySplitter()}
        properties = NeqSimProcessModel._splitter_operating_properties(
            _LegacySplitter()
        )

        self.assertNotIn("branchCount", properties)
        self.assertEqual(properties["solvedBranchCount"], 2.0)
        self.assertEqual(properties["branch0Flow_kg_hr"], 25.0)
        self.assertEqual(properties["branch1Flow_kg_hr"], 75.0)
        self.assertNotIn("outletFlowTotal_kg_hr", properties)
        self.assertNotIn("branch0Fraction", properties)
        self.assertNotIn("branch1Fraction", properties)
        self.assertNotIn("flowClosure_kg_hr", properties)

        kpis = {}
        model._extract_unit_properties(kpis)
        self.assertEqual(
            kpis["legacy split.splitStream0_flow_kg_hr"].value,
            25.0,
        )
        self.assertEqual(
            kpis["legacy split.splitStream1_flow_kg_hr"].value,
            75.0,
        )


class MixerPropertyExtractionTest(unittest.TestCase):
    """Validate solved mixer inlet allocations and explicit flow closure."""

    def test_reports_native_inlet_fractions_and_workbook_properties(self):
        class _JavaClass:
            @staticmethod
            def getSimpleName():
                return "Mixer"

        class _Stream:
            def __init__(self, flow_kg_hr):
                self.flow_kg_hr = flow_kg_hr

            def getFlowRate(self, unit):
                if unit != "kg/hr":
                    raise AssertionError(unit)
                return self.flow_kg_hr

        class _Mixer:
            streams = [_Stream(40.0), _Stream(60.0)]

            @staticmethod
            def getClass():
                return _JavaClass()

            @classmethod
            def getNumberOfInputStreams(cls):
                return len(cls.streams)

            @classmethod
            def getStream(cls, index):
                return cls.streams[index]

            @staticmethod
            def getOutletStream():
                return _Stream(100.0)

        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._units = {"feed mixer": _Mixer()}
        model._unit_ps_name = {"feed mixer": "main"}
        kpis = {}

        model._extract_unit_properties(kpis)

        expected = {
            "inletCount": (2.0, "[-]"),
            "solvedInletCount": (2.0, "[-]"),
            "inlet0Flow_kg_hr": (40.0, "kg/hr"),
            "inlet1Flow_kg_hr": (60.0, "kg/hr"),
            "inlet0Fraction": (0.4, "[-]"),
            "inlet1Fraction": (0.6, "[-]"),
            "inletFlowTotal_kg_hr": (100.0, "kg/hr"),
            "outletFlow_kg_hr": (100.0, "kg/hr"),
            "flowClosure_kg_hr": (0.0, "kg/hr"),
            "flowClosure_pct": (0.0, "%"),
        }
        for property_name, (value, unit) in expected.items():
            with self.subTest(property_name=property_name):
                kpi = kpis[f"feed mixer.{property_name}"]
                self.assertAlmostEqual(kpi.value, value, delta=1.0e-12)
                self.assertEqual(kpi.unit, unit)

        workbook_properties = model.list_units()[0].properties
        for property_name, (value, _unit) in expected.items():
            with self.subTest(workbook_property=property_name):
                self.assertAlmostEqual(
                    workbook_properties[property_name],
                    value,
                    delta=1.0e-12,
                )

    def test_mixer_subclasses_emit_workbook_and_kpi_routing_properties(self):
        class _JavaClass:
            def __init__(self, simple_name):
                self.simple_name = simple_name

            def getSimpleName(self):
                return self.simple_name

        class _Stream:
            def __init__(self, flow_kg_hr):
                self.flow_kg_hr = flow_kg_hr

            def getFlowRate(self, unit):
                if unit != "kg/hr":
                    raise AssertionError(unit)
                return self.flow_kg_hr

        for java_class in (
            "StaticMixer",
            "StaticNeqMixer",
            "StaticPhaseMixer",
        ):
            with self.subTest(java_class=java_class):
                class _MixerSubclass:
                    @staticmethod
                    def getClass():
                        return _JavaClass(java_class)

                    @staticmethod
                    def getNumberOfInputStreams():
                        return 2

                    @staticmethod
                    def getStream(index):
                        return (_Stream(40.0), _Stream(60.0))[index]

                    @staticmethod
                    def getOutletStream():
                        return _Stream(100.0)

                model = NeqSimProcessModel.__new__(NeqSimProcessModel)
                model._units = {"native mixer": _MixerSubclass()}
                model._unit_ps_name = {"native mixer": "main"}
                kpis = {}

                model._extract_unit_properties(kpis)

                self.assertEqual(
                    kpis["native mixer.inletFlowTotal_kg_hr"].value,
                    100.0,
                )
                self.assertEqual(
                    model.list_units()[0].properties[
                        "inletFlowTotal_kg_hr"
                    ],
                    100.0,
                )

    def test_specialized_mixers_skip_generic_equilibrium_ph_closure(self):
        class _JavaClass:
            def __init__(self, simple_name):
                self.simple_name = simple_name

            def getSimpleName(self):
                return self.simple_name

        class _SpecializedMixer:
            def __init__(self, simple_name):
                self.simple_name = simple_name

            def getClass(self):
                return _JavaClass(self.simple_name)

            @staticmethod
            def run(_run_id):
                raise AssertionError(
                    "specialized mixer must not use generic PH closure"
                )

        for java_class in (
            "StaticMixer",
            "StaticNeqMixer",
            "StaticPhaseMixer",
        ):
            with self.subTest(java_class=java_class):
                class _Process:
                    @staticmethod
                    def getUnitOperations():
                        return [_SpecializedMixer(java_class)]

                NeqSimProcessModel._run_acyclic_mixer_energy_closure(
                    _Process()
                )

    def test_withholds_fractions_and_closure_for_partial_inlet_coverage(self):
        class _Stream:
            def __init__(self, flow_kg_hr):
                self.flow_kg_hr = flow_kg_hr

            def getFlowRate(self, unit):
                if unit != "kg/hr":
                    raise AssertionError(unit)
                return self.flow_kg_hr

        class _PartlyReadableMixer:
            @staticmethod
            def getNumberOfInputStreams():
                return 3

            @staticmethod
            def getStream(index):
                if index == 2:
                    raise RuntimeError("native inlet unavailable")
                return (_Stream(40.0), _Stream(60.0))[index]

            @staticmethod
            def getOutletStream():
                return _Stream(200.0)

        properties = NeqSimProcessModel._mixer_operating_properties(
            _PartlyReadableMixer()
        )

        self.assertEqual(properties["inletCount"], 3.0)
        self.assertEqual(properties["solvedInletCount"], 2.0)
        self.assertNotIn("inletFlowTotal_kg_hr", properties)
        self.assertNotIn("inlet0Fraction", properties)
        self.assertNotIn("inlet1Fraction", properties)
        self.assertNotIn("flowClosure_pct", properties)


class PumpPropertyExtractionTest(unittest.TestCase):
    """Validate solved pump properties and derived hydraulic quantities."""

    def test_reports_pump_state_power_head_and_workbook_properties(self):
        class _JavaClass:
            @staticmethod
            def getSimpleName():
                return "Pump"

        class _Fluid:
            @staticmethod
            def getDensity(unit):
                if unit != "kg/m3":
                    raise AssertionError(unit)
                return 800.0

        class _InletStream:
            @staticmethod
            def getFluid():
                return _Fluid()

            @staticmethod
            def getFlowRate(unit):
                if unit != "m3/sec":
                    raise AssertionError(unit)
                return 0.01

        class _Pump:
            @staticmethod
            def getClass():
                return _JavaClass()

            @staticmethod
            def getInletPressure():
                return 10.0

            @staticmethod
            def getOutletPressure():
                return 30.0

            @staticmethod
            def getInletTemperature():
                return 298.15

            @staticmethod
            def getOutletTemperature():
                return 299.25

            @staticmethod
            def getIsentropicEfficiency():
                return 0.75

            @staticmethod
            def getSpeed():
                return 1_000.0

            @staticmethod
            def getPower():
                return 30_000.0

            @staticmethod
            def getDuty():
                return 30_000.0

            @staticmethod
            def getInletStream():
                return _InletStream()

        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._units = {"export pump": _Pump()}
        model._unit_ps_name = {"export pump": "main"}
        kpis = {}

        model._extract_unit_properties(kpis)

        expected = {
            "inletPressure_bara": (10.0, "bara"),
            "outletPressure_bara": (30.0, "bara"),
            "pressureRise_bar": (20.0, "bar"),
            "inletTemperature_K": (298.15, "K"),
            "outletTemperature_K": (299.25, "K"),
            "efficiency": (0.75, "[-]"),
            "speed_rpm": (1_000.0, "rpm"),
            "shaftPower_kW": (30.0, "kW"),
            "inletDensity_kg_m3": (800.0, "kg/m3"),
            "inletVolumetricFlow_m3_s": (0.01, "m3/s"),
            "head_m": (254.92905324448208, "m"),
            "hydraulicPower_kW": (20.0, "kW"),
        }
        for property_name, (value, unit) in expected.items():
            with self.subTest(property_name=property_name):
                kpi = kpis[f"export pump.{property_name}"]
                self.assertAlmostEqual(kpi.value, value, delta=1.0e-10)
                self.assertEqual(kpi.unit, unit)

        workbook_properties = model.list_units()[0].properties
        self.assertAlmostEqual(workbook_properties["head_m"], expected["head_m"][0])
        self.assertAlmostEqual(
            workbook_properties["hydraulicPower_kW"],
            expected["hydraulicPower_kW"][0],
        )

    def test_esp_uses_native_actual_head_instead_of_bulk_density(self):
        class _JavaClass:
            @staticmethod
            def getSimpleName():
                return "ESPPump"

        class _Fluid:
            @staticmethod
            def getDensity(unit):
                if unit != "kg/m3":
                    raise AssertionError(unit)
                return 50.0

        class _InletStream:
            @staticmethod
            def getFluid():
                return _Fluid()

            @staticmethod
            def getFlowRate(unit):
                if unit != "m3/sec":
                    raise AssertionError(unit)
                return 0.01

        class _ESPPump:
            @staticmethod
            def getClass():
                return _JavaClass()

            @staticmethod
            def getInletPressure():
                return 10.0

            @staticmethod
            def getOutletPressure():
                return 30.0

            @staticmethod
            def getActualHead():
                return 450.0

            @staticmethod
            def getIsentropicEfficiency():
                return 75.0

            @staticmethod
            def getInletStream():
                return _InletStream()

        esp_pump = _ESPPump()
        properties = NeqSimProcessModel._pump_operating_properties(esp_pump)

        self.assertEqual(properties["efficiency"], 0.75)
        self.assertEqual(properties["head_m"], 450.0)
        self.assertEqual(properties["pressureRise_bar"], 20.0)
        self.assertEqual(properties["hydraulicPower_kW"], 20.0)
        bulk_density_head = (
            20.0e5 / (50.0 * 9.80665)
        )
        self.assertNotAlmostEqual(
            properties["head_m"],
            bulk_density_head,
            delta=1.0e-6,
        )

        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._units = {"well esp": esp_pump}
        model._unit_ps_name = {"well esp": "main"}
        kpis = {}
        model._extract_unit_properties(kpis)
        self.assertEqual(kpis["well esp.efficiency"].value, 0.75)
        workbook_properties = model.list_units()[0].properties
        self.assertEqual(workbook_properties["efficiency"], 0.75)
        self.assertEqual(
            workbook_properties["isentropicEfficiency"],
            0.75,
        )


class HeatExchangerDesignBasisModelTest(unittest.TestCase):
    """Protect typed exchanger design metadata and solved capacity margins."""

    def test_validates_and_collects_enabled_exchanger_design_basis(self):
        self.assertEqual(
            ProcessBuilder._heat_exchanger_design_settings(
                {
                    "use_design_basis": True,
                    "design_duty_capacity_kw": 2_500.0,
                    "design_ua_capacity_w_per_k": 125_000.0,
                }
            ),
            (True, 2_500.0, 125_000.0),
        )
        units = [
            {
                "name": "cross exchanger",
                "type": "heat_exchanger",
                "params": {
                    "ua_w_per_k": 100_000.0,
                    "use_design_basis": True,
                    "design_duty_capacity_kw": 2_500.0,
                    "design_ua_capacity_w_per_k": 125_000.0,
                },
            },
            {
                "name": "spare exchanger",
                "type": "heat_exchanger",
                "params": {"use_design_basis": False},
            },
        ]
        expected = {
            "cross exchanger": {
                "design_duty_capacity_kw": 2_500.0,
                "design_ua_capacity_w_per_k": 125_000.0,
            }
        }
        self.assertEqual(
            ProcessBuilder._requested_heat_exchanger_design_bases(units),
            expected,
        )
        self.assertEqual(
            ProcessBuilder._requested_equipment_design_bases(units),
            expected,
        )

        for params, message in (
            ({"use_design_basis": 1}, "must be boolean"),
            ({"design_duty_capacity_kw": math.nan}, "must be finite"),
            ({"design_ua_capacity_w_per_k": 0.0}, "must be between"),
        ):
            with self.subTest(params=params):
                with self.assertRaisesRegex(ValueError, message):
                    ProcessBuilder._heat_exchanger_design_settings(params)

        with self.assertRaisesRegex(
            ValueError,
            "only for heat_exchanger units",
        ):
            ProcessBuilder._requested_heat_exchanger_design_bases(
                [
                    {
                        "name": "not an exchanger",
                        "type": "compressor",
                        "params": {"design_duty_capacity_kw": 2_500.0},
                    }
                ]
            )

    def test_reports_duty_and_ua_margins_with_explicit_units(self):
        class _JavaClass:
            @staticmethod
            def getSimpleName():
                return "HeatExchanger"

        class _HeatExchanger:
            @staticmethod
            def getClass():
                return _JavaClass()

        exchanger = _HeatExchanger()
        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._equipment_design_bases = {
            "cross exchanger": {
                "design_duty_capacity_kw": 2_500.0,
                "design_ua_capacity_w_per_k": 125_000.0,
            }
        }
        model._direct_unit_run_provenance = {}
        model._heat_exchanger_state_snapshots = {
            "cross exchanger": ("trusted",)
        }
        model._units = {"cross exchanger": exchanger}
        model._unit_ps_name = {"cross exchanger": "main"}
        operating = {
            "heatTransferDuty_kW": 2_400.0,
            "UA_W_K": 100_000.0,
        }
        with patch.object(
            NeqSimProcessModel,
            "_heat_exchanger_operating_properties",
            return_value=operating,
        ), patch.object(
            model,
            "_report_unit_duty_suppression",
            return_value=False,
        ):
            properties = model._heat_exchanger_design_properties(
                "cross exchanger",
                exchanger,
            )
            constraint = model._heat_exchanger_design_constraint(
                "cross exchanger",
                exchanger,
            )
            workbook = model.list_units()[0].properties

        self.assertEqual(properties["designDutyCapacity_kW"], 2_500.0)
        self.assertEqual(properties["designUACapacity_W_K"], 125_000.0)
        self.assertAlmostEqual(properties["dutyUtilization_pct"], 96.0)
        self.assertAlmostEqual(properties["uaUtilization_pct"], 80.0)
        self.assertAlmostEqual(properties["dutyMargin_kW"], 100.0)
        self.assertAlmostEqual(properties["uaMargin_W_K"], 25_000.0)
        self.assertEqual(constraint.status, "OK")
        self.assertEqual(workbook["designDutyCapacity_kW"], 2_500.0)
        self.assertEqual(workbook["dutyMargin_kW"], 100.0)
        expected_units = {
            "designDutyCapacity_kW": "kW",
            "designUACapacity_W_K": "W/K",
            "dutyUtilization_pct": "%",
            "uaUtilization_pct": "%",
            "dutyMargin_kW": "kW",
            "uaMargin_W_K": "W/K",
        }
        for property_name, unit in expected_units.items():
            with self.subTest(property_name=property_name):
                self.assertEqual(
                    model._heat_exchanger_design_property_unit(
                        property_name
                    ),
                    unit,
                )

        model._equipment_design_bases["cross exchanger"][
            "design_duty_capacity_kw"
        ] = 2_000.0
        with patch.object(
            NeqSimProcessModel,
            "_heat_exchanger_operating_properties",
            return_value=operating,
        ):
            violation = model._heat_exchanger_design_constraint(
                "cross exchanger",
                object(),
            )
        self.assertEqual(violation.status, "VIOLATION")
        self.assertIn("duty", violation.detail)

        with patch.object(
            NeqSimProcessModel,
            "_heat_exchanger_operating_properties",
            return_value={"UA_W_K": 100_000.0},
        ):
            unknown = model._heat_exchanger_design_constraint(
                "cross exchanger",
                object(),
            )
        self.assertEqual(unknown.status, "UNKNOWN")

    def test_saved_metadata_accepts_exact_exchanger_capacity_schema(self):
        valid_basis = {
            "design_duty_capacity_kw": 2_500.0,
            "design_ua_capacity_w_per_k": 125_000.0,
        }
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "neqsimweb2/studio_metadata.json",
                json.dumps(
                    {
                        "schema_version": 1,
                        "equipment_design_bases": {
                            "cross exchanger": valid_basis,
                        },
                    }
                ),
            )
        buffer.seek(0)
        with zipfile.ZipFile(buffer, "r") as archive:
            self.assertEqual(
                NeqSimProcessModel._read_studio_metadata(archive),
                {"cross exchanger": valid_basis},
            )

        for invalid_basis in (
            {"design_duty_capacity_kw": 2_500.0},
            {**valid_basis, "design_ua_capacity_w_per_k": 0.0},
            {**valid_basis, "motor_rating_kw": 100.0},
        ):
            with self.subTest(invalid_basis=invalid_basis):
                invalid_buffer = io.BytesIO()
                with zipfile.ZipFile(invalid_buffer, "w") as archive:
                    archive.writestr(
                        "neqsimweb2/studio_metadata.json",
                        json.dumps(
                            {
                                "schema_version": 1,
                                "equipment_design_bases": {
                                    "cross exchanger": invalid_basis,
                                },
                            }
                        ),
                    )
                invalid_buffer.seek(0)
                with zipfile.ZipFile(invalid_buffer, "r") as archive:
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "equipment design metadata",
                    ):
                        NeqSimProcessModel._read_studio_metadata(archive)

    def test_mismatched_equipment_schema_is_withheld_before_kpi_reads(self):
        class _JavaClass:
            @staticmethod
            def getSimpleName():
                return "Pump"

        class _Pump:
            @staticmethod
            def getClass():
                return _JavaClass()

        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._equipment_design_bases = {
            "replacement pump": {
                "design_duty_capacity_kw": 2_500.0,
                "design_ua_capacity_w_per_k": 125_000.0,
            }
        }
        model._units = {"replacement pump": _Pump()}
        model._unit_ps_name = {"replacement pump": "main"}

        self.assertEqual(
            model._pump_design_properties(
                "replacement pump",
                model._units["replacement pump"],
            ),
            {},
        )
        kpis = {}
        model._extract_unit_properties(kpis)
        self.assertNotIn(
            "replacement pump.designFlowCapacity_m3_per_hr",
            kpis,
        )


class ValveDesignBasisModelTest(unittest.TestCase):
    """Protect typed valve Cv metadata, margins, and fail-loud status."""

    class _JavaClass:
        @staticmethod
        def getSimpleName():
            return "ThrottlingValve"

    class _Valve:
        def __init__(self, cv=18.0):
            self.cv = cv

        @staticmethod
        def getClass():
            return ValveDesignBasisModelTest._JavaClass()

        def getCv(self):
            return self.cv

    def test_validates_and_collects_enabled_valve_design_basis(self):
        self.assertEqual(
            ProcessBuilder._valve_design_settings(
                {
                    "use_design_basis": True,
                    "design_cv_capacity_us": 20.0,
                }
            ),
            (True, 20.0),
        )
        units = [
            {
                "name": "metering valve",
                "type": "valve",
                "params": {
                    "outlet_pressure_bara": 30.0,
                    "percent_valve_opening": 80.0,
                    "use_design_basis": True,
                    "design_cv_capacity_us": 20.0,
                },
            },
            {
                "name": "spare valve",
                "type": "valve",
                "params": {"use_design_basis": False},
            },
        ]
        expected = {
            "metering valve": {"design_cv_capacity_us": 20.0}
        }
        self.assertEqual(
            ProcessBuilder._requested_valve_design_bases(units),
            expected,
        )
        self.assertEqual(
            ProcessBuilder._requested_equipment_design_bases(units),
            expected,
        )

        for params, message in (
            ({"use_design_basis": 1}, "must be boolean"),
            ({"design_cv_capacity_us": math.nan}, "must be finite"),
            ({"design_cv_capacity_us": 0.0}, "must be between"),
        ):
            with self.subTest(params=params):
                with self.assertRaisesRegex(ValueError, message):
                    ProcessBuilder._valve_design_settings(params)

        with self.assertRaisesRegex(ValueError, "only for valve units"):
            ProcessBuilder._requested_valve_design_bases(
                [
                    {
                        "name": "not a valve",
                        "type": "compressor",
                        "params": {"design_cv_capacity_us": 20.0},
                    }
                ]
            )

    def test_rejects_fixed_cv_only_when_required_cv_screen_is_active(self):
        for fixed_key in (
            "cv",
            "valve_cv",
            "flow_coefficient",
            "Cv",
            "VALVE_CV",
            "FLOW_COEFFICIENT",
        ):
            with self.subTest(fixed_key=fixed_key):
                with self.assertRaisesRegex(
                    ValueError,
                    "cannot be combined",
                ):
                    ProcessBuilder._valve_design_settings(
                        {
                            "use_design_basis": True,
                            "design_cv_capacity_us": 40.0,
                            fixed_key: 25.0,
                        }
                    )

                self.assertEqual(
                    ProcessBuilder._valve_design_settings(
                        {
                            "use_design_basis": False,
                            fixed_key: 25.0,
                        }
                    ),
                    (False, 100.0),
                )

    def test_reports_rated_cv_capacity_margin_and_constraint(self):
        valve = self._Valve()
        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._equipment_design_bases = {
            "metering valve": {"design_cv_capacity_us": 20.0}
        }
        model._units = {"metering valve": valve}
        model._unit_ps_name = {"metering valve": "main"}

        properties = model._valve_design_properties(
            "metering valve",
            valve,
        )
        self.assertEqual(properties["designCvCapacity_US"], 20.0)
        self.assertAlmostEqual(properties["cvUtilization_pct"], 90.0)
        self.assertAlmostEqual(properties["cvMargin_US"], 2.0)
        self.assertEqual(
            model._valve_design_property_unit("designCvCapacity_US"),
            "US Cv",
        )
        self.assertEqual(
            model._valve_design_property_unit("cvUtilization_pct"),
            "%",
        )
        self.assertEqual(
            model._valve_design_property_unit("cvMargin_US"),
            "US Cv",
        )
        self.assertEqual(
            model._valve_design_constraint("metering valve", valve).status,
            "OK",
        )

        kpis = {}
        model._extract_unit_properties(kpis)
        self.assertEqual(
            kpis["metering valve.designCvCapacity_US"].value,
            20.0,
        )
        self.assertEqual(
            kpis["metering valve.cvUtilization_pct"].unit,
            "%",
        )
        self.assertEqual(kpis["metering valve.cvMargin_US"].unit, "US Cv")
        workbook = model.list_units()[0].properties
        self.assertEqual(workbook["designCvCapacity_US"], 20.0)
        self.assertAlmostEqual(workbook["cvUtilization_pct"], 90.0)
        self.assertAlmostEqual(workbook["cvMargin_US"], 2.0)

        model._equipment_design_bases["metering valve"][
            "design_cv_capacity_us"
        ] = 15.0
        violation = model._valve_design_constraint("metering valve", valve)
        self.assertEqual(violation.status, "VIOLATION")
        self.assertIn("exceeds rated Cv", violation.detail)

        valve.cv = math.nan
        unknown = model._valve_design_constraint("metering valve", valve)
        self.assertEqual(unknown.status, "UNKNOWN")

    def test_saved_metadata_accepts_only_exact_valve_capacity_schema(self):
        valid_basis = {"design_cv_capacity_us": 20.0}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "neqsimweb2/studio_metadata.json",
                json.dumps(
                    {
                        "schema_version": 1,
                        "equipment_design_bases": {
                            "metering valve": valid_basis,
                        },
                    }
                ),
            )
        buffer.seek(0)
        with zipfile.ZipFile(buffer, "r") as archive:
            self.assertEqual(
                NeqSimProcessModel._read_studio_metadata(archive),
                {"metering valve": valid_basis},
            )

        for invalid_basis in (
            {},
            {"design_cv_capacity_us": 0.0},
            {"design_cv_capacity_us": math.inf},
            {"design_cv_capacity_us": 20.0, "motor_rating_kw": 100.0},
        ):
            with self.subTest(invalid_basis=invalid_basis):
                invalid_buffer = io.BytesIO()
                with zipfile.ZipFile(invalid_buffer, "w") as archive:
                    archive.writestr(
                        "neqsimweb2/studio_metadata.json",
                        json.dumps(
                            {
                                "schema_version": 1,
                                "equipment_design_bases": {
                                    "metering valve": invalid_basis,
                                },
                            }
                        ),
                    )
                invalid_buffer.seek(0)
                with zipfile.ZipFile(invalid_buffer, "r") as archive:
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "equipment design metadata",
                    ):
                        NeqSimProcessModel._read_studio_metadata(archive)

    def test_withholds_valve_properties_for_another_equipment_schema(self):
        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._equipment_design_bases = {
            "metering valve": {
                "design_duty_capacity_kw": 2_500.0,
                "design_ua_capacity_w_per_k": 125_000.0,
            }
        }

        self.assertEqual(
            model._valve_design_properties(
                "metering valve",
                self._Valve(),
            ),
            {},
        )


class NativeValveDesignPerformanceTest(unittest.TestCase):
    """Validate rated-Cv margins, persistence, and Process Chat handoff."""

    @staticmethod
    def _specification(
        flow_scale: float,
        design_cv_capacity_us: float = 40.0,
    ) -> dict:
        return {
            "name": "Native valve Cv design benchmark",
            "fluid": {
                "eos_model": "srk",
                "mixing_rule": 2,
                "components": {"methane": 0.90, "ethane": 0.10},
                "composition_basis": "mole_fraction",
                "temperature_C": 30.0,
                "pressure_bara": 80.0,
                "total_flow": 10_000.0 * flow_scale,
                "flow_unit": "kg/hr",
            },
            "process": [
                {"name": "feed", "type": "stream"},
                {
                    "name": "metering valve",
                    "type": "valve",
                    "params": {
                        "outlet_pressure_bara": 30.0,
                        "percent_valve_opening": 60.0,
                        "use_design_basis": True,
                        "design_cv_capacity_us": design_cv_capacity_us,
                    },
                },
            ],
        }

    @staticmethod
    def _constraint(result: ModelRunResult) -> ConstraintStatus:
        return next(
            constraint
            for constraint in result.constraints
            if constraint.name == "valve_design.metering valve"
        )

    def test_native_rated_cv_round_trips_and_updates_incrementally(self):
        cv_by_scale = {}
        utilization_by_scale = {}
        baseline_builder = None
        baseline_model = None
        for flow_scale in (1.0, 1.05):
            with self.subTest(flow_scale=flow_scale):
                builder = ProcessBuilder()
                model = builder.build_from_spec(
                    self._specification(flow_scale)
                )
                result = model.run(timeout_ms=180_000)
                valve = model.get_unit("metering valve")
                cv = float(valve.getCv())
                cv_kpi = result.kpis["metering valve.Cv"]
                capacity_kpi = result.kpis[
                    "metering valve.designCvCapacity_US"
                ]
                utilization_kpi = result.kpis[
                    "metering valve.cvUtilization_pct"
                ]
                margin_kpi = result.kpis["metering valve.cvMargin_US"]

                self.assertAlmostEqual(cv_kpi.value, cv, delta=1.0e-12)
                self.assertEqual(cv_kpi.unit, "US Cv")
                self.assertEqual(capacity_kpi.value, 40.0)
                self.assertEqual(capacity_kpi.unit, "US Cv")
                self.assertAlmostEqual(
                    utilization_kpi.value,
                    100.0 * cv / 40.0,
                    delta=1.0e-10,
                )
                self.assertEqual(utilization_kpi.unit, "%")
                self.assertAlmostEqual(
                    margin_kpi.value,
                    40.0 - cv,
                    delta=1.0e-10,
                )
                self.assertEqual(margin_kpi.unit, "US Cv")
                self.assertEqual(self._constraint(result).status, "OK")
                self.assertAlmostEqual(
                    float(valve.getOutletStream().getPressure("bara")),
                    30.0,
                    delta=0.05,
                )
                self.assertAlmostEqual(
                    float(valve.getPercentValveOpening()),
                    60.0,
                    delta=1.0e-12,
                )
                self.assertLess(
                    result.kpis["mass_balance_pct"].value,
                    1.0e-6,
                )
                self.assertLess(
                    result.kpis["component_balance_max_pct"].value,
                    1.0e-6,
                )
                self.assertLess(
                    result.kpis["energy_balance_pct"].value,
                    1.0e-6,
                )
                self.assertIn(
                    "Registered equipment design basis for: metering valve",
                    builder.build_log,
                )
                generated_script = builder.to_python_script()
                self.assertIn('"design_cv_capacity_us": 40.0', generated_script)
                self.assertIn(
                    "NeqSimProcessModel.from_process_system(",
                    generated_script,
                )
                compile(
                    generated_script,
                    "<valve-cv-design-replay>",
                    "exec",
                )

                cv_by_scale[flow_scale] = cv
                utilization_by_scale[flow_scale] = utilization_kpi.value
                if flow_scale == 1.0:
                    baseline_builder = builder
                    baseline_model = model
                print(
                    "native valve design benchmark:",
                    f"scale={flow_scale:.2f}",
                    f"Cv={cv:.6f} US Cv",
                    f"utilization={utilization_kpi.value:.6f}%",
                    f"margin={margin_kpi.value:.6f} US Cv",
                    f"mass={result.kpis['mass_balance_pct'].value:.3e}%",
                    "components="
                    f"{result.kpis['component_balance_max_pct'].value:.3e}%",
                    f"energy={result.kpis['energy_balance_pct'].value:.3e}%",
                )

        self.assertGreater(cv_by_scale[1.05], cv_by_scale[1.0])
        self.assertGreater(
            utilization_by_scale[1.05],
            utilization_by_scale[1.0],
        )
        self.assertIsNotNone(baseline_builder)
        self.assertIsNotNone(baseline_model)

        saved = NeqSimProcessModel.from_bytes(baseline_model.save_bytes())
        saved_result = saved.run(timeout_ms=180_000)
        self.assertEqual(
            saved_result.kpis[
                "metering valve.designCvCapacity_US"
            ].value,
            40.0,
        )
        self.assertEqual(self._constraint(saved_result).status, "OK")

        updated_specification = self._specification(
            1.0,
            design_cv_capacity_us=35.0,
        )
        change_type, param_changes, fluid_changes, extra_steps = (
            _classify_build_change(
                baseline_builder.spec,
                updated_specification,
            )
        )
        self.assertEqual(change_type, "property_update")
        session = object.__new__(ProcessChatSession)
        session.model = baseline_model
        session._builder = baseline_builder
        session.history = []
        session._system_prompt = ""
        session._llm_followup = lambda client, types: "updated"
        with patch(
            "process_chat.chat_tools.build_system_prompt",
            return_value="updated prompt",
        ), patch(
            "process_chat.chat_tools._build_model_built_result",
            return_value={},
        ):
            response = session._handle_incremental_update(
                "update valve rated Cv",
                updated_specification,
                change_type,
                param_changes,
                fluid_changes,
                extra_steps,
                None,
                None,
            )
        self.assertEqual(response, "updated")
        self.assertEqual(
            baseline_model._equipment_design_bases["metering valve"][
                "design_cv_capacity_us"
            ],
            35.0,
        )
        updated_result = baseline_model.run(timeout_ms=180_000)
        self.assertEqual(self._constraint(updated_result).status, "VIOLATION")

        reloaded = NeqSimProcessModel.from_bytes(
            baseline_builder.save_neqsim_bytes()
        )
        reloaded_result = reloaded.run(timeout_ms=180_000)
        self.assertEqual(
            reloaded_result.kpis[
                "metering valve.designCvCapacity_US"
            ].value,
            35.0,
        )
        self.assertEqual(
            self._constraint(reloaded_result).status,
            "VIOLATION",
        )

    def test_native_rerun_recalculates_required_cv_after_flow_change(self):
        builder = ProcessBuilder()
        model = builder.build_from_spec(
            self._specification(1.0, design_cv_capacity_us=50.0)
        )
        baseline = model.run(timeout_ms=180_000)
        baseline_cv = baseline.kpis["metering valve.Cv"].value
        self.assertEqual(self._constraint(baseline).status, "OK")

        feed = model.get_unit("feed")
        feed.getFluid().setTotalFlowRate(20_000.0, "kg/hr")
        doubled = model.run(timeout_ms=180_000)
        doubled_cv = doubled.kpis["metering valve.Cv"].value

        self.assertAlmostEqual(
            doubled_cv,
            2.0 * baseline_cv,
            delta=1.0e-8,
        )
        self.assertEqual(self._constraint(doubled).status, "VIOLATION")
        self.assertLess(doubled.kpis["mass_balance_pct"].value, 1.0e-6)
        self.assertLess(
            doubled.kpis["component_balance_max_pct"].value,
            1.0e-6,
        )

        repeated = model.run(timeout_ms=180_000)
        self.assertAlmostEqual(
            repeated.kpis["metering valve.Cv"].value,
            doubled_cv,
            delta=1.0e-12,
        )

    def test_incremental_fixed_cv_edit_is_rejected_before_native_mutation(self):
        builder = ProcessBuilder()
        model = builder.build_from_spec(self._specification(1.0))
        model.run(timeout_ms=180_000)
        original_cv = float(model.get_unit("metering valve").getCv())
        original_basis = {
            name: dict(basis)
            for name, basis in model._equipment_design_bases.items()
        }

        invalid_specification = json.loads(json.dumps(builder.spec))
        invalid_specification["process"][1]["params"]["cv"] = 25.0
        change_type, param_changes, fluid_changes, extra_steps = (
            _classify_build_change(builder.spec, invalid_specification)
        )
        self.assertEqual(change_type, "property_update")

        session = object.__new__(ProcessChatSession)
        session.model = model
        session._builder = builder
        session.history = []
        session._system_prompt = ""
        session._handle_build = lambda *args, **kwargs: "rejected"
        response = session._handle_incremental_update(
            "set a fixed Cv while screening required Cv",
            invalid_specification,
            change_type,
            param_changes,
            fluid_changes,
            extra_steps,
            None,
            None,
        )

        self.assertEqual(response, "rejected")
        self.assertEqual(
            float(model.get_unit("metering valve").getCv()),
            original_cv,
        )
        self.assertEqual(model._equipment_design_bases, original_basis)

    def test_scenario_fixed_cv_patch_is_rejected_without_mutation(self):
        builder = ProcessBuilder()
        model = builder.build_from_spec(self._specification(1.0))
        model.run(timeout_ms=180_000)
        original_cv = float(model.get_unit("metering valve").getCv())

        comparison = run_scenarios(
            model,
            [
                Scenario(
                    name="invalid fixed Cv",
                    description="Reject fixed Cv while required-Cv screen is active",
                    patch=InputPatch(
                        changes={"units.METERING VALVE.cv": 25.0}
                    ),
                )
            ],
            timeout_ms=180_000,
        )

        self.assertFalse(comparison.cases[0].success)
        self.assertIn("fixed valve Cv", comparison.cases[0].error)
        self.assertEqual(
            float(model.get_unit("metering valve").getCv()),
            original_cv,
        )
        self.assertEqual(comparison.patch_log[-1]["status"], "FAILED")

    def test_target_solver_rejects_fixed_cv_for_active_screen(self):
        builder = ProcessBuilder()
        model = builder.build_from_spec(self._specification(1.0))
        model.run(timeout_ms=180_000)

        comparison = run_scenarios(
            model,
            [
                Scenario(
                    name="invalid target Cv",
                    description="Reject Cv as the manipulated variable",
                    patch=InputPatch(
                        changes={},
                        targets=[
                            TargetSpec(
                                target_kpi="metering valve.Cv",
                                target_value=30.0,
                                variable="unit_param",
                                unit_name="METERING VALVE",
                                unit_param="Cv",
                                initial_guess=25.0,
                                min_value=10.0,
                                max_value=50.0,
                            )
                        ],
                    ),
                )
            ],
            timeout_ms=180_000,
        )

        self.assertFalse(comparison.cases[0].success)
        self.assertIn("target-solved", comparison.cases[0].error)
        self.assertEqual(comparison.patch_log[-1]["status"], "FAILED")

    def test_target_solver_propagates_added_valve_failures(self):
        builder = ProcessBuilder()
        model = builder.build_from_spec(self._specification(1.0))
        model.run(timeout_ms=180_000)

        invalid_patches = (
            InputPatch(
                changes={},
                add_units=[
                    AddUnitOp(
                        name="trim valve",
                        equipment_type="valve",
                        insert_after="metering valve",
                        params={
                            "outlet_pressure_bara": 20.0,
                            "use_design_basis": True,
                            "design_cv_capacity_us": 50.0,
                            "valve_cv": 25.0,
                        },
                    )
                ],
            ),
            InputPatch(
                changes={"units.trim valve.cv": 25.0},
                add_units=[
                    AddUnitOp(
                        name="trim valve",
                        equipment_type="valve",
                        insert_after="metering valve",
                        params={
                            "outlet_pressure_bara": 20.0,
                            "use_design_basis": True,
                            "design_cv_capacity_us": 50.0,
                        },
                    )
                ],
            ),
        )

        for index, invalid_patch in enumerate(invalid_patches):
            with self.subTest(index=index):
                invalid_patch.targets = [
                    TargetSpec(
                        target_kpi="metering valve.Cv",
                        target_value=40.0,
                        variable="stream_scale",
                        stream_name="feed",
                        initial_guess=1.0,
                        min_value=0.5,
                        max_value=1.5,
                    )
                ]
                comparison = run_scenarios(
                    model,
                    [
                        Scenario(
                            name=f"invalid added valve target {index}",
                            description=(
                                "Propagate added-valve validation failures"
                            ),
                            patch=invalid_patch,
                        )
                    ],
                    timeout_ms=180_000,
                )

                self.assertFalse(comparison.cases[0].success)
                self.assertIn(
                    "Iterative solver failed",
                    comparison.cases[0].error,
                )
                self.assertEqual(
                    comparison.patch_log[-1]["status"],
                    "FAILED",
                )

    def test_scenario_added_valve_registers_rated_cv_metadata(self):
        builder = ProcessBuilder()
        model = builder.build_from_spec(self._specification(1.0))
        model.run(timeout_ms=180_000)

        comparison = run_scenarios(
            model,
            [
                Scenario(
                    name="add trim valve",
                    description="Add a screened trim valve",
                    patch=InputPatch(
                        changes={},
                        add_units=[
                            AddUnitOp(
                                name="trim valve",
                                equipment_type="valve",
                                insert_after="metering valve",
                                params={
                                    "outlet_pressure_bara": 20.0,
                                    "percent_valve_opening": 20.0,
                                    "use_design_basis": True,
                                    "design_cv_capacity_us": 100.0,
                                },
                            )
                        ],
                    ),
                )
            ],
            timeout_ms=180_000,
        )

        case = comparison.cases[0]
        self.assertTrue(case.success, case.error)
        self.assertEqual(
            case.result.kpis["trim valve.designCvCapacity_US"].value,
            100.0,
        )
        self.assertAlmostEqual(
            case.result.kpis["trim valve.percentValveOpening"].value,
            20.0,
            delta=1.0e-12,
        )
        self.assertEqual(
            next(
                constraint.status
                for constraint in case.result.constraints
                if constraint.name == "valve_design.trim valve"
            ),
            "VIOLATION",
        )

    def test_added_valve_basis_uses_qualified_multi_system_name(self):
        from neqsim import jneqsim

        def _train(
            train_name: str,
            feed_name: str,
            valve_name: str,
            outlet_pressure_bara: float,
        ):
            specification = self._specification(1.0)
            specification["name"] = train_name
            specification["process"][0]["name"] = feed_name
            valve_spec = specification["process"][1]
            valve_spec["name"] = valve_name
            valve_spec["params"] = {
                "outlet_pressure_bara": outlet_pressure_bara,
                "percent_valve_opening": 60.0,
            }
            process = ProcessBuilder().build_from_spec(
                specification
            ).get_process()
            process.setName(train_name)
            return process

        train_a = _train(
            "train-a",
            "feed a",
            "metering valve a",
            30.0,
        )
        train_b = _train(
            "train-b",
            "feed b",
            "trim valve",
            25.0,
        )
        process_model = jneqsim.process.processmodel.ProcessModel()
        self.assertTrue(process_model.add("train-a", train_a))
        self.assertTrue(process_model.add("train-b", train_b))
        model = NeqSimProcessModel(process_model)

        operation_log = apply_add_units(
            model,
            [
                AddUnitOp(
                    name="trim valve",
                    equipment_type="valve",
                    insert_after="metering valve a",
                    params={
                        "outlet_pressure_bara": 20.0,
                        "percent_valve_opening": 20.0,
                        "use_design_basis": True,
                        "design_cv_capacity_us": 100.0,
                    },
                )
            ],
        )

        self.assertFalse(
            [
                entry
                for entry in operation_log
                if entry.get("status") == "FAILED"
            ]
        )
        self.assertEqual(
            model._equipment_design_bases,
            {
                "train-a/trim valve": {
                    "design_cv_capacity_us": 100.0,
                }
            },
        )
        result = model.run(timeout_ms=180_000)
        self.assertEqual(
            result.kpis[
                "train-a/trim valve.designCvCapacity_US"
            ].value,
            100.0,
        )
        self.assertEqual(
            next(
                constraint.status
                for constraint in result.constraints
                if constraint.name == "valve_design.train-a/trim valve"
            ),
            "VIOLATION",
        )

    def test_existing_screen_is_remapped_without_screening_duplicate(self):
        from neqsim import jneqsim

        def _train(
            train_name: str,
            feed_name: str,
            valve_name: str,
            outlet_pressure_bara: float,
        ):
            specification = self._specification(1.0)
            specification["name"] = train_name
            specification["process"][0]["name"] = feed_name
            valve_spec = specification["process"][1]
            valve_spec["name"] = valve_name
            valve_spec["params"] = {
                "outlet_pressure_bara": outlet_pressure_bara,
                "percent_valve_opening": 60.0,
            }
            process = ProcessBuilder().build_from_spec(
                specification
            ).get_process()
            process.setName(train_name)
            return process

        train_a = _train("train-a", "feed a", "trim valve", 30.0)
        train_b = _train(
            "train-b",
            "feed b",
            "metering valve b",
            25.0,
        )
        process_model = jneqsim.process.processmodel.ProcessModel()
        self.assertTrue(process_model.add("train-a", train_a))
        self.assertTrue(process_model.add("train-b", train_b))
        model = NeqSimProcessModel(process_model)
        model._equipment_design_bases = {
            "trim valve": {"design_cv_capacity_us": 100.0}
        }

        operation_log = apply_add_units(
            model,
            [
                AddUnitOp(
                    name="trim valve",
                    equipment_type="valve",
                    insert_after="metering valve b",
                    params={
                        "outlet_pressure_bara": 20.0,
                        "percent_valve_opening": 60.0,
                    },
                )
            ],
        )

        self.assertFalse(
            [
                entry
                for entry in operation_log
                if entry.get("status") == "FAILED"
            ]
        )
        self.assertEqual(
            model._equipment_design_bases,
            {
                "train-a/trim valve": {
                    "design_cv_capacity_us": 100.0,
                }
            },
        )
        patch_log = apply_patch_to_model(
            model,
            InputPatch(
                changes={"units.train-b/trim valve.cv": 25.0}
            ),
        )
        self.assertEqual(patch_log[-1]["status"], "OK", patch_log)
        self.assertEqual(
            float(model.get_unit("train-b/trim valve").getCv()),
            25.0,
        )

        result = model.run(timeout_ms=180_000)
        self.assertEqual(
            result.kpis[
                "train-a/trim valve.designCvCapacity_US"
            ].value,
            100.0,
        )
        self.assertNotIn(
            "train-b/trim valve.designCvCapacity_US",
            result.kpis,
        )
        self.assertNotEqual(
            next(
                constraint.status
                for constraint in result.constraints
                if constraint.name == "valve_design.train-a/trim valve"
            ),
            "UNKNOWN",
        )

    def test_same_system_duplicate_is_rejected_before_reindexing(self):
        specification = self._specification(1.0)
        specification["name"] = "train"
        specification["process"][1]["name"] = "trim valve"
        model = ProcessBuilder().build_from_spec(specification)

        operation_log = apply_add_units(
            model,
            [
                AddUnitOp(
                    name="trim valve",
                    equipment_type="valve",
                    insert_after="feed",
                    params={
                        "outlet_pressure_bara": 50.0,
                        "percent_valve_opening": 100.0,
                    },
                )
            ],
        )

        self.assertTrue(
            [
                entry
                for entry in operation_log
                if entry.get("status") == "FAILED"
            ]
        )
        self.assertEqual(
            model._equipment_design_bases,
            {
                "trim valve": {
                    "design_cv_capacity_us": 40.0,
                }
            },
        )
        self.assertEqual(
            [unit.name for unit in model.list_units()],
            ["feed", "trim valve"],
        )

    def test_duplicate_added_valves_fail_before_topology_mutation(self):
        specification = self._specification(1.0)
        specification["name"] = "train"
        model = ProcessBuilder().build_from_spec(specification)

        operation_log = apply_add_units(
            model,
            [
                AddUnitOp(
                    name="trim valve",
                    equipment_type="valve",
                    insert_after="metering valve",
                    params={
                        "outlet_pressure_bara": 20.0,
                        "percent_valve_opening": 20.0,
                        "use_design_basis": True,
                        "design_cv_capacity_us": 50.0,
                    },
                ),
                AddUnitOp(
                    name="trim valve",
                    equipment_type="valve",
                    insert_after="metering valve",
                    params={
                        "outlet_pressure_bara": 20.0,
                        "percent_valve_opening": 100.0,
                        "use_design_basis": True,
                        "design_cv_capacity_us": 100.0,
                    },
                ),
            ],
        )

        self.assertTrue(
            [
                entry
                for entry in operation_log
                if entry.get("status") == "FAILED"
            ]
        )
        opening_to_capacity = {}
        for unit_name, basis in model._equipment_design_bases.items():
            if unit_name.startswith("train/trim valve"):
                opening = float(
                    model.get_unit(unit_name).getPercentValveOpening()
                )
                opening_to_capacity[opening] = basis[
                    "design_cv_capacity_us"
                ]
        self.assertEqual(opening_to_capacity, {})
        self.assertEqual(
            [unit.name for unit in model.list_units()],
            ["feed", "metering valve"],
        )

    def test_process_chat_addition_preserves_remapped_valve_basis(self):
        from neqsim import jneqsim

        train_a = ProcessBuilder().build_from_spec(
            self._specification(1.0)
        ).get_process()
        train_a.setName("train-a")
        train_a.getUnit("metering valve").setName("trim valve")

        train_b_specification = self._specification(1.0)
        train_b_specification["process"][0]["name"] = "feed b"
        train_b_specification["process"][1]["name"] = "metering valve b"
        train_b = ProcessBuilder().build_from_spec(
            train_b_specification
        ).get_process()
        train_b.setName("train-b")

        process_model = jneqsim.process.processmodel.ProcessModel()
        self.assertTrue(process_model.add("train-a", train_a))
        self.assertTrue(process_model.add("train-b", train_b))
        model = NeqSimProcessModel(process_model)
        model._equipment_design_bases = {
            "trim valve": {"design_cv_capacity_us": 40.0}
        }

        session = object.__new__(ProcessChatSession)
        session.model = model
        session._builder = None
        session.history = []
        session._system_prompt = ""
        session._llm_followup = lambda client, types: "updated"
        with patch(
            "process_chat.chat_tools.build_system_prompt",
            return_value="updated prompt",
        ), patch(
            "process_chat.chat_tools._build_model_built_result",
            return_value={},
        ):
            response = session._handle_build(
                "add duplicate valve",
                {
                    "add": [
                        {
                            "name": "trim valve",
                            "type": "valve",
                            "insert_after": "metering valve b",
                            "params": {
                                "outlet_pressure_bara": 20.0,
                                "percent_valve_opening": 60.0,
                            },
                        }
                    ]
                },
                None,
                None,
            )

        self.assertEqual(response, "updated")
        self.assertEqual(
            model._equipment_design_bases,
            {
                "train-a/trim valve": {
                    "design_cv_capacity_us": 40.0,
                }
            },
        )


class PumpDesignBasisApplicationTest(unittest.TestCase):
    """Protect strict opt-in pump capacities and solved-model propagation."""

    def test_validates_and_collects_only_enabled_pump_design_bases(self):
        self.assertEqual(
            ProcessBuilder._pump_design_settings(
                {
                    "use_design_basis": True,
                    "design_flow_capacity_m3_per_hr": 40.0,
                    "design_head_capacity_m": 400.0,
                    "motor_rating_kw": 35.0,
                }
            ),
            (True, 40.0, 400.0, 35.0),
        )
        units = [
            {
                "name": " export pump ",
                "type": "pump",
                "params": {
                    "use_design_basis": True,
                    "design_flow_capacity_m3_per_hr": 40.0,
                    "design_head_capacity_m": 400.0,
                    "motor_rating_kw": 35.0,
                },
            },
            {
                "name": "spare pump",
                "type": "pump",
                "params": {"use_design_basis": False},
            },
        ]
        self.assertEqual(
            ProcessBuilder._requested_pump_design_bases(units),
            {
                " export pump ": {
                    "design_flow_capacity_m3_per_hr": 40.0,
                    "design_head_capacity_m": 400.0,
                    "motor_rating_kw": 35.0,
                }
            },
        )

        invalid_cases = (
            ({"use_design_basis": 1}, "use_design_basis must be boolean"),
            (
                {"design_flow_capacity_m3_per_hr": math.nan},
                "must be finite",
            ),
            ({"design_head_capacity_m": 0.0}, "must be between"),
            ({"motor_rating_kw": True}, "must be numeric"),
        )
        for params, message in invalid_cases:
            with self.subTest(params=params):
                with self.assertRaisesRegex(ValueError, message):
                    ProcessBuilder._pump_design_settings(params)

        with self.assertRaisesRegex(ValueError, "only for pump units"):
            ProcessBuilder._requested_pump_design_bases(
                [
                    {
                        "name": "not a pump",
                        "type": "compressor",
                        "params": {"use_design_basis": False},
                    }
                ]
            )

    def test_reports_design_margins_with_explicit_units_and_status(self):
        class _JavaClass:
            @staticmethod
            def getSimpleName():
                return "Pump"

        class _Fluid:
            @staticmethod
            def getDensity(unit):
                if unit != "kg/m3":
                    raise AssertionError(unit)
                return 800.0

        class _InletStream:
            @staticmethod
            def getFluid():
                return _Fluid()

            @staticmethod
            def getFlowRate(unit):
                if unit != "m3/sec":
                    raise AssertionError(unit)
                return 0.01

        class _Pump:
            @staticmethod
            def getClass():
                return _JavaClass()

            @staticmethod
            def getInletPressure():
                return 10.0

            @staticmethod
            def getOutletPressure():
                return 30.0

            @staticmethod
            def getPower():
                return 30_000.0

            @staticmethod
            def getInletStream():
                return _InletStream()

        pump = _Pump()
        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._equipment_design_bases = {
            "export pump": {
                "design_flow_capacity_m3_per_hr": 40.0,
                "design_head_capacity_m": 300.0,
                "motor_rating_kw": 35.0,
            }
        }
        model._units = {"export pump": pump}
        model._unit_ps_name = {"export pump": "main"}
        properties = model._pump_design_properties("export pump", pump)

        self.assertAlmostEqual(properties["flowUtilization_pct"], 90.0)
        self.assertAlmostEqual(properties["flowMargin_m3_per_hr"], 4.0)
        self.assertAlmostEqual(
            properties["headUtilization_pct"],
            84.97635108149402,
        )
        self.assertAlmostEqual(properties["motorMargin_kW"], 5.0)
        self.assertEqual(
            model._pump_design_constraint("export pump", pump).status,
            "OK",
        )

        kpis = {}
        model._extract_unit_properties(kpis)
        expected_units = {
            "designFlowCapacity_m3_per_hr": "m3/hr",
            "designHeadCapacity_m": "m",
            "motorRating_kW": "kW",
            "flowUtilization_pct": "%",
            "headUtilization_pct": "%",
            "motorUtilization_pct": "%",
            "flowMargin_m3_per_hr": "m3/hr",
            "headMargin_m": "m",
            "motorMargin_kW": "kW",
        }
        for property_name, unit in expected_units.items():
            with self.subTest(property_name=property_name):
                self.assertEqual(
                    kpis[f"export pump.{property_name}"].unit,
                    unit,
                )
        workbook = model.list_units()[0].properties
        self.assertEqual(workbook["designHeadCapacity_m"], 300.0)
        self.assertAlmostEqual(workbook["motorMargin_kW"], 5.0)

        model._equipment_design_bases["export pump"]["motor_rating_kw"] = 25.0
        violation = model._pump_design_constraint("export pump", pump)
        self.assertEqual(violation.status, "VIOLATION")
        self.assertIn("motor", violation.detail)

        class _IncompletePump(_Pump):
            getPower = None

        incomplete = _IncompletePump()
        unknown = model._pump_design_constraint("export pump", incomplete)
        self.assertEqual(unknown.status, "UNKNOWN")

    def test_legacy_script_preserves_design_basis_as_reporting_metadata(self):
        builder = ProcessBuilder()
        builder._process_name = "Legacy pump design replay"
        builder._spec = {
            "name": "Legacy pump design replay",
            "fluid": {
                "eos_model": "srk",
                "components": {"n-hexane": 1.0},
                "composition_basis": "mole_fraction",
                "temperature_C": 25.0,
                "pressure_bara": 10.0,
                "total_flow": 20_000.0,
                "flow_unit": "kg/hr",
                "mixing_rule": 2,
            },
            "process": [
                {"name": "feed", "type": "stream"},
                {
                    "name": "export pump",
                    "type": "pump",
                    "params": {
                        "outlet_pressure_bara": 40.0,
                        "efficiency": 0.75,
                        "use_design_basis": True,
                        "design_flow_capacity_m3_per_hr": 40.0,
                        "design_head_capacity_m": 500.0,
                        "motor_rating_kw": 60.0,
                    },
                },
            ],
        }

        script = builder.to_python_script()

        self.assertEqual(script.count("process.run()"), 1)
        self.assertIn("equipment_design_bases = {", script)
        self.assertIn('"export pump": {', script)
        self.assertIn('"design_flow_capacity_m3_per_hr": 40.0', script)
        self.assertIn('"design_head_capacity_m": 500.0', script)
        self.assertIn('"motor_rating_kw": 60.0', script)
        self.assertIn(
            "from process_chat.process_model import NeqSimProcessModel",
            script,
        )
        self.assertIn(
            "equipment_design_bases=equipment_design_bases",
            script,
        )
        self.assertIn("result = model.run(timeout_ms=180_000)", script)
        self.assertIn("model_file.write(model.save_bytes())", script)
        self.assertIn("legacy_pump_design_replay.case.json", script)
        self.assertIn("json.dump(", script)
        compile(script, "<pump-design-replay>", "exec")

        namespace = {}
        original_directory = os.getcwd()
        with tempfile.TemporaryDirectory() as temporary_directory:
            try:
                os.chdir(temporary_directory)
                exec(
                    compile(script, "<pump-design-replay>", "exec"),
                    namespace,
                )
                with open(
                    "legacy_pump_design_replay.case.json",
                    encoding="utf-8",
                ) as case_file:
                    saved_case = json.load(case_file)
                with open(
                    "legacy_pump_design_replay.neqsim",
                    "rb",
                ) as model_file:
                    saved_model = model_file.read()
            finally:
                os.chdir(original_directory)

        replay_result = namespace["result"]
        self.assertEqual(
            replay_result.kpis["export pump.motorRating_kW"].value,
            60.0,
        )
        self.assertEqual(
            next(
                constraint.status
                for constraint in replay_result.constraints
                if constraint.name == "pump_design.export pump"
            ),
            "OK",
        )
        self.assertEqual(saved_case, builder._spec)

        reloaded = NeqSimProcessModel.from_bytes(saved_model)
        reloaded_result = reloaded.run(timeout_ms=180_000)
        self.assertEqual(
            reloaded_result.kpis["export pump.motorRating_kW"].value,
            60.0,
        )
        self.assertEqual(
            next(
                constraint.status
                for constraint in reloaded_result.constraints
                if constraint.name == "pump_design.export pump"
            ),
            "OK",
        )

    def test_rejects_incomplete_or_out_of_range_saved_design_metadata(self):
        valid_basis = {
            "design_flow_capacity_m3_per_hr": 40.0,
            "design_head_capacity_m": 500.0,
            "motor_rating_kw": 60.0,
        }
        invalid_bases = []
        for missing_key in valid_basis:
            incomplete = dict(valid_basis)
            incomplete.pop(missing_key)
            invalid_bases.append(incomplete)
        invalid_bases.extend(
            (
                {**valid_basis, "unknown_capacity": 1.0},
                {**valid_basis, "motor_rating_kw": 0.0},
                {
                    **valid_basis,
                    "design_head_capacity_m": 20_000.1,
                },
            )
        )

        for basis in invalid_bases:
            with self.subTest(basis=basis):
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w") as archive:
                    archive.writestr(
                        "neqsimweb2/studio_metadata.json",
                        json.dumps(
                            {
                                "schema_version": 1,
                                "equipment_design_bases": {
                                    "export pump": basis,
                                },
                            }
                        ),
                    )
                buffer.seek(0)
                with zipfile.ZipFile(buffer, "r") as archive:
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "equipment design metadata",
                    ):
                        NeqSimProcessModel._read_studio_metadata(archive)


class NativePumpPerformanceTest(unittest.TestCase):
    """Benchmark editable native pump performance at nearby points."""

    @staticmethod
    def _run_case(
        flow_scale: float,
        efficiency: float,
        motor_rating_kw: float = 60.0,
        pump_name: str = "export pump",
    ):
        units, pump_id = add_catalog_unit([], "pump", pump_name)
        units[0]["params"].update(
            {
                "outlet_pressure_bara": 40.0,
                "efficiency": efficiency,
                "use_design_basis": True,
                "design_flow_capacity_m3_per_hr": 40.0,
                "design_head_capacity_m": 500.0,
                "motor_rating_kw": motor_rating_kw,
            }
        )
        graph_spec = {
            "name": "Native pump performance benchmark",
            "units": units,
            "connections": [
                {
                    "id": "feed-to-export-pump",
                    "type": "material",
                    "source": {
                        "kind": "inlet",
                        "id": "feed",
                        "port": "out",
                    },
                    "target": {
                        "kind": "unit",
                        "id": pump_id,
                        "port": "in",
                    },
                }
            ],
        }
        expected_flow = 20_000.0 * flow_scale
        inlet_specs = [
            {
                "inlet_id": "feed",
                "name": "feed",
                "fluid_spec": {
                    "eos_model": "srk",
                    "mixing_rule": 2,
                    "components": {
                        "n-hexane": 0.85,
                        "n-heptane": 0.15,
                    },
                    "composition_basis": "mole_fraction",
                    "temperature_C": 25.0,
                    "pressure_bara": 10.0,
                    "total_flow": expected_flow,
                    "flow_unit": "kg/hr",
                },
            }
        ]
        builder = ProcessBuilder()
        model = builder.build_acyclic_graph(
            graph_spec,
            inlet_specs,
            ["feed", pump_id],
        )
        result = model.run(timeout_ms=180_000)
        return builder, graph_spec, model, result, expected_flow

    def test_saved_neqsim_round_trip_preserves_pump_design_metadata(self):
        builder, _, _, _, _ = self._run_case(
            1.0,
            0.75,
            pump_name=" export pump ",
        )
        saved_bytes = builder.save_neqsim_bytes()

        self.assertIsNotNone(saved_bytes)
        with zipfile.ZipFile(io.BytesIO(saved_bytes), "r") as archive:
            metadata = json.loads(
                archive.read(
                    "neqsimweb2/studio_metadata.json"
                ).decode("utf-8")
            )
        self.assertEqual(metadata["schema_version"], 1)
        self.assertEqual(
            metadata["equipment_design_bases"]["export pump"][
                "motor_rating_kw"
            ],
            60.0,
        )

        reloaded = NeqSimProcessModel.from_bytes(saved_bytes)
        reloaded_result = reloaded.run(timeout_ms=180_000)
        self.assertEqual(
            reloaded_result.kpis["export pump.motorRating_kW"].value,
            60.0,
        )
        self.assertEqual(
            next(
                constraint.status
                for constraint in reloaded_result.constraints
                if constraint.name == "pump_design.export pump"
            ),
            "OK",
        )

    def test_graph_replay_saved_model_preserves_pump_design_metadata(self):
        builder, _, _, _, _ = self._run_case(1.0, 0.75)
        script = builder.to_python_script()
        namespace: dict[str, object] = {}

        with tempfile.TemporaryDirectory() as temporary_directory:
            previous_directory = os.getcwd()
            try:
                os.chdir(temporary_directory)
                exec(
                    compile(script, "<pump-graph-replay>", "exec"),
                    namespace,
                )
                saved_path = os.path.join(
                    temporary_directory,
                    "native_pump_performance_benchmark.neqsim",
                )
                reloaded = NeqSimProcessModel.from_file(saved_path)
            finally:
                os.chdir(previous_directory)

        reloaded_result = reloaded.run(timeout_ms=180_000)
        self.assertEqual(
            reloaded_result.kpis["export pump.motorRating_kW"].value,
            60.0,
        )
        self.assertEqual(
            next(
                constraint.status
                for constraint in reloaded_result.constraints
                if constraint.name == "pump_design.export pump"
            ),
            "OK",
        )

    def test_incremental_pump_design_edit_updates_saved_model_metadata(self):
        specification = {
            "name": "Incremental pump design benchmark",
            "fluid": {
                "eos_model": "srk",
                "mixing_rule": 2,
                "components": {
                    "n-hexane": 0.85,
                    "n-heptane": 0.15,
                },
                "composition_basis": "mole_fraction",
                "temperature_C": 25.0,
                "pressure_bara": 10.0,
                "total_flow": 20_000.0,
                "flow_unit": "kg/hr",
            },
            "process": [
                {
                    "name": "feed",
                    "type": "stream",
                },
                {
                    "name": "export pump",
                    "type": "pump",
                    "params": {
                        "outlet_pressure_bara": 40.0,
                        "efficiency": 0.75,
                        "use_design_basis": True,
                        "design_flow_capacity_m3_per_hr": 40.0,
                        "design_head_capacity_m": 500.0,
                        "motor_rating_kw": 60.0,
                    },
                },
            ],
        }
        builder = ProcessBuilder()
        model = builder.build_from_spec(specification)
        updated_specification = json.loads(json.dumps(specification))
        updated_specification["process"][1]["params"][
            "motor_rating_kw"
        ] = 30.0
        change_type, param_changes, fluid_changes, extra_steps = (
            _classify_build_change(
                builder.spec,
                updated_specification,
            )
        )
        self.assertEqual(change_type, "property_update")

        session = object.__new__(ProcessChatSession)
        session.model = model
        session._builder = builder
        session.history = []
        session._system_prompt = ""
        session._llm_followup = lambda client, types: "updated"
        with patch(
            "process_chat.chat_tools.build_system_prompt",
            return_value="updated prompt",
        ), patch(
            "process_chat.chat_tools._build_model_built_result",
            return_value={},
        ):
            response = session._handle_incremental_update(
                "update pump motor rating",
                updated_specification,
                change_type,
                param_changes,
                fluid_changes,
                extra_steps,
                None,
                None,
            )
        self.assertEqual(response, "updated")
        self.assertEqual(
            model._equipment_design_bases["export pump"][
                "motor_rating_kw"
            ],
            30.0,
        )

        saved_bytes = builder.save_neqsim_bytes()
        reloaded = NeqSimProcessModel.from_bytes(saved_bytes)
        reloaded_result = reloaded.run(timeout_ms=180_000)
        self.assertEqual(
            reloaded_result.kpis["export pump.motorRating_kW"].value,
            30.0,
        )
        self.assertEqual(
            next(
                constraint.status
                for constraint in reloaded_result.constraints
                if constraint.name == "pump_design.export pump"
            ),
            "VIOLATION",
        )

    def test_incremental_add_pump_updates_saved_model_metadata(self):
        specification = {
            "name": "Incremental add pump design benchmark",
            "fluid": {
                "eos_model": "srk",
                "mixing_rule": 2,
                "components": {
                    "n-hexane": 0.85,
                    "n-heptane": 0.15,
                },
                "composition_basis": "mole_fraction",
                "temperature_C": 25.0,
                "pressure_bara": 10.0,
                "total_flow": 20_000.0,
                "flow_unit": "kg/hr",
            },
            "process": [
                {
                    "name": "feed",
                    "type": "stream",
                },
            ],
        }
        builder = ProcessBuilder()
        model = builder.build_from_spec(specification)
        add_specification = {
            "add": [
                {
                    "name": " export pump ",
                    "type": "pump",
                    "insert_after": "feed",
                    "params": {
                        "outlet_pressure_bara": 40.0,
                        "efficiency": 0.75,
                        "use_design_basis": True,
                        "design_flow_capacity_m3_per_hr": 40.0,
                        "design_head_capacity_m": 500.0,
                        "motor_rating_kw": 60.0,
                    },
                },
            ],
        }

        session = object.__new__(ProcessChatSession)
        session.model = model
        session._builder = builder
        session.history = []
        session._system_prompt = ""
        session._llm_followup = lambda client, types: "updated"
        with patch(
            "process_chat.chat_tools.build_system_prompt",
            return_value="updated prompt",
        ), patch(
            "process_chat.chat_tools._build_model_built_result",
            return_value={},
        ):
            response = session._handle_build(
                "add export pump",
                add_specification,
                None,
                None,
            )

        self.assertEqual(response, "updated")
        self.assertEqual(
            model._equipment_design_bases[" export pump "][
                "motor_rating_kw"
            ],
            60.0,
        )
        self.assertEqual(
            builder.spec["process"][-1]["name"],
            " export pump ",
        )

        saved_bytes = builder.save_neqsim_bytes()
        reloaded = NeqSimProcessModel.from_bytes(saved_bytes)
        reloaded_result = reloaded.run(timeout_ms=180_000)
        self.assertEqual(
            reloaded_result.kpis[" export pump .motorRating_kW"].value,
            60.0,
        )
        self.assertEqual(
            next(
                constraint.status
                for constraint in reloaded_result.constraints
                if constraint.name == "pump_design. export pump "
            ),
            "OK",
        )

    def test_native_pump_conserves_and_trends_with_flow_and_efficiency(self):
        shaft_power = {}
        hydraulic_power = {}
        head = {}
        flow_utilization = {}
        motor_utilization = {}

        for flow_scale in (1.0, 1.05):
            for efficiency in (0.75, 0.85):
                with self.subTest(
                    flow_scale=flow_scale,
                    efficiency=efficiency,
                ):
                    builder, graph_spec, model, result, expected_flow = (
                        self._run_case(flow_scale, efficiency)
                    )
                    pump = model.get_unit("export pump")
                    key = (flow_scale, efficiency)
                    shaft_power[key] = result.kpis[
                        "export pump.shaftPower_kW"
                    ].value
                    hydraulic_power[key] = result.kpis[
                        "export pump.hydraulicPower_kW"
                    ].value
                    head[key] = result.kpis["export pump.head_m"].value
                    flow_utilization[key] = result.kpis[
                        "export pump.flowUtilization_pct"
                    ].value
                    motor_utilization[key] = result.kpis[
                        "export pump.motorUtilization_pct"
                    ].value

                    self.assertAlmostEqual(
                        float(pump.getIsentropicEfficiency()),
                        efficiency,
                        delta=1.0e-12,
                    )
                    self.assertAlmostEqual(
                        result.kpis["export pump.pressureRise_bar"].value,
                        30.0,
                        delta=1.0e-10,
                    )
                    self.assertAlmostEqual(
                        result.kpis["material_product_flow_kg_hr"].value,
                        expected_flow,
                        delta=max(1.0e-6 * expected_flow, 1.0e-3),
                    )
                    for balance_name in (
                        "mass_balance_pct",
                        "component_balance_max_pct",
                        "energy_balance_pct",
                        "unit_mass_balance_max_pct",
                        "unit_energy_balance_max_pct",
                    ):
                        self.assertLess(
                            result.kpis[balance_name].value,
                            1.0e-6,
                        )
                    workbook_unit = next(
                        unit
                        for unit in model.list_units()
                        if unit.name == "export pump"
                    )
                    self.assertAlmostEqual(
                        workbook_unit.properties["efficiency"],
                        efficiency,
                    )
                    self.assertIn("head_m", workbook_unit.properties)
                    self.assertEqual(
                        workbook_unit.properties[
                            "designFlowCapacity_m3_per_hr"
                        ],
                        40.0,
                    )
                    self.assertGreater(
                        workbook_unit.properties["flowMargin_m3_per_hr"],
                        0.0,
                    )
                    pump_constraint = next(
                        constraint
                        for constraint in result.constraints
                        if constraint.name == "pump_design.export pump"
                    )
                    self.assertEqual(pump_constraint.status, "OK")
                    self.assertEqual(
                        json.loads(json.dumps(graph_spec, allow_nan=False)),
                        graph_spec,
                    )
                    self.assertIn(
                        "Registered equipment design basis for: export pump",
                        builder.build_log,
                    )
                    self.assertIn(
                        "Acyclic graph built and converged successfully.",
                        builder.build_log,
                    )
                    print(
                        "native pump benchmark:",
                        f"scale={flow_scale:.2f}",
                        f"efficiency={efficiency:.2f}",
                        f"shaft={shaft_power[key]:.6f} kW",
                        f"hydraulic={hydraulic_power[key]:.6f} kW",
                        f"head={head[key]:.6f} m",
                        f"energy={result.kpis['energy_balance_pct'].value:.3e}%",
                    )

        for efficiency in (0.75, 0.85):
            self.assertGreater(
                shaft_power[(1.05, efficiency)],
                shaft_power[(1.0, efficiency)],
            )
            self.assertGreater(
                hydraulic_power[(1.05, efficiency)],
                hydraulic_power[(1.0, efficiency)],
            )
            self.assertAlmostEqual(
                head[(1.05, efficiency)],
                head[(1.0, efficiency)],
                delta=1.0e-8,
            )
        for flow_scale in (1.0, 1.05):
            self.assertLess(
                shaft_power[(flow_scale, 0.85)],
                shaft_power[(flow_scale, 0.75)],
            )
            self.assertAlmostEqual(
                hydraulic_power[(flow_scale, 0.85)],
                hydraulic_power[(flow_scale, 0.75)],
                delta=1.0e-10,
            )
        for efficiency in (0.75, 0.85):
            self.assertGreater(
                flow_utilization[(1.05, efficiency)],
                flow_utilization[(1.0, efficiency)],
            )
            self.assertGreater(
                motor_utilization[(1.05, efficiency)],
                motor_utilization[(1.0, efficiency)],
            )

        _, _, model, baseline_result, _ = self._run_case(1.0, 0.75)
        clone = model.clone()
        clone_result = clone.run(timeout_ms=180_000)
        self.assertAlmostEqual(
            clone_result.kpis["export pump.motorRating_kW"].value,
            60.0,
        )
        self.assertEqual(
            next(
                constraint.status
                for constraint in clone_result.constraints
                if constraint.name == "pump_design.export pump"
            ),
            "OK",
        )
        self.assertAlmostEqual(
            clone_result.kpis["export pump.shaftPower_kW"].value,
            baseline_result.kpis["export pump.shaftPower_kW"].value,
            delta=1.0e-8,
        )

        _, _, _, underrated_result, _ = self._run_case(
            1.0,
            0.75,
            motor_rating_kw=1.0,
        )
        underrated_constraint = next(
            constraint
            for constraint in underrated_result.constraints
            if constraint.name == "pump_design.export pump"
        )
        self.assertEqual(underrated_constraint.status, "VIOLATION")
        self.assertIn("motor", underrated_constraint.detail)
        for balance_name in (
            "mass_balance_pct",
            "component_balance_max_pct",
            "energy_balance_pct",
            "unit_mass_balance_max_pct",
            "unit_energy_balance_max_pct",
        ):
            self.assertLess(
                underrated_result.kpis[balance_name].value,
                1.0e-6,
            )


class PipelineConstructionTest(unittest.TestCase):
    """Validate executable native construction for palette pipelines."""

    def test_pipeline_catalog_builds_adiabatic_pipe_with_geometry(self):
        units, pipeline_id = add_catalog_unit(
            [],
            "pipeline",
            "transport pipeline",
        )
        graph_spec = {
            "name": "Native pipeline construction regression",
            "units": units,
            "connections": [
                {
                    "id": "feed-to-transport-pipeline",
                    "type": "material",
                    "source": {
                        "kind": "inlet",
                        "id": "feed",
                        "port": "out",
                    },
                    "target": {
                        "kind": "unit",
                        "id": pipeline_id,
                        "port": "in",
                    },
                }
            ],
        }
        inlet_specs = [
            {
                "inlet_id": "feed",
                "name": "feed",
                "fluid_spec": {
                    "eos_model": "srk",
                    "mixing_rule": 2,
                    "components": {
                        "methane": 0.90,
                        "ethane": 0.10,
                    },
                    "composition_basis": "mole_fraction",
                    "temperature_C": 20.0,
                    "pressure_bara": 80.0,
                    "total_flow": 10_000.0,
                    "flow_unit": "kg/hr",
                },
            }
        ]
        builder = ProcessBuilder()
        model = builder.build_acyclic_graph(
            graph_spec,
            inlet_specs,
            ["feed", pipeline_id],
        )
        pipeline = model.get_unit("transport pipeline")

        self.assertEqual(
            str(pipeline.getClass().getSimpleName()),
            "PipeBeggsAndBrills",
        )
        self.assertEqual(str(pipeline.getHeatTransferMode()), "ADIABATIC")
        self.assertAlmostEqual(float(pipeline.getLength()), 1000.0)
        self.assertAlmostEqual(float(pipeline.getDiameter()), 0.30)
        self.assertAlmostEqual(
            float(pipeline.getPipeWallRoughness()),
            1.0e-5,
        )
        self.assertIn(
            "Acyclic graph built and converged successfully.",
            builder.build_log,
        )


class PipelinePropertyExtractionTest(unittest.TestCase):
    """Validate solved pipeline hydraulic properties and explicit units."""

    def test_reports_pipeline_geometry_state_and_hydraulics(self):
        class _JavaClass:
            @staticmethod
            def getSimpleName():
                return "AdiabaticPipe"

        class _Pipeline:
            @staticmethod
            def getClass():
                return _JavaClass()

            @staticmethod
            def getLength():
                return 1000.0

            @staticmethod
            def getDiameter():
                return 0.30

            @staticmethod
            def getPipeWallRoughness():
                return 1.0e-5

            @staticmethod
            def getInletPressure():
                return 80.0

            @staticmethod
            def getOutletPressure():
                return 79.99532

            @staticmethod
            def getPressureDrop():
                return 0.00468

            @staticmethod
            def getInletTemperature():
                return 293.15

            @staticmethod
            def getOutletTemperature():
                return 293.15

            @staticmethod
            def getVelocity():
                return 0.57594

            @staticmethod
            def getReynoldsNumber():
                return 890_206.0

            @staticmethod
            def getFrictionFactor():
                return 0.01240

        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._units = {"transport pipeline": _Pipeline()}
        kpis = {}

        model._extract_unit_properties(kpis)

        expected = {
            "length_m": (1000.0, "m"),
            "diameter_m": (0.30, "m"),
            "roughness_m": (1.0e-5, "m"),
            "inletPressure_bara": (80.0, "bara"),
            "outletPressure_bara": (79.99532, "bara"),
            "pressureDrop_bar": (0.00468, "bar"),
            "inletTemperature_K": (293.15, "K"),
            "outletTemperature_K": (293.15, "K"),
            "velocity_m_s": (0.57594, "m/s"),
            "reynoldsNumber": (890_206.0, "[-]"),
            "frictionFactor": (0.01240, "[-]"),
        }
        for property_name, (value, unit) in expected.items():
            with self.subTest(property_name=property_name):
                kpi = kpis[f"transport pipeline.{property_name}"]
                self.assertAlmostEqual(kpi.value, value, delta=1.0e-10)
                self.assertEqual(kpi.unit, unit)

    def test_reports_beggs_brill_profile_hydraulics(self):
        class _JavaClass:
            @staticmethod
            def getSimpleName():
                return "PipeBeggsAndBrills"

        class _Pipeline:
            @staticmethod
            def getClass():
                return _JavaClass()

            @staticmethod
            def getMixtureVelocity():
                return 0.57398

            @staticmethod
            def getMixtureReynoldsNumber():
                return [889_635.0, 889_652.0]

        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._units = {"transport pipeline": _Pipeline()}
        kpis = {}

        model._extract_unit_properties(kpis)

        self.assertEqual(
            kpis["transport pipeline.velocity_m_s"].unit,
            "m/s",
        )
        self.assertAlmostEqual(
            kpis["transport pipeline.velocity_m_s"].value,
            0.57398,
        )
        self.assertEqual(
            kpis["transport pipeline.reynoldsNumber"].unit,
            "[-]",
        )
        self.assertAlmostEqual(
            kpis["transport pipeline.reynoldsNumber"].value,
            889_652.0,
        )


class PipelineDesignBasisModelTest(unittest.TestCase):
    """Protect pipeline hydraulic capacities, margins, and status."""

    class _JavaClass:
        def __init__(self, name="PipeBeggsAndBrills"):
            self.name = name

        def getSimpleName(self):
            return self.name

    class _Pipeline:
        def __init__(
            self,
            pressure_drop_bar=0.0048,
            velocity_m_s=0.58,
            java_class="PipeBeggsAndBrills",
        ):
            self.pressure_drop_bar = pressure_drop_bar
            self.velocity_m_s = velocity_m_s
            self.java_class = java_class

        def getClass(self):
            return PipelineDesignBasisModelTest._JavaClass(self.java_class)

        def getPressureDrop(self):
            return self.pressure_drop_bar

        def getMixtureVelocity(self):
            return self.velocity_m_s

        def getVelocity(self):
            return self.velocity_m_s

    @staticmethod
    def _model(pipeline):
        model = NeqSimProcessModel.__new__(NeqSimProcessModel)
        model._equipment_design_bases = {
            "transport pipeline": {
                "design_pressure_drop_capacity_bar": 0.005,
                "design_velocity_capacity_m_per_s": 0.60,
            }
        }
        model._units = {"transport pipeline": pipeline}
        model._unit_ps_name = {"transport pipeline": "main"}
        return model

    def test_reports_hydraulic_capacity_margins_with_explicit_units(self):
        pipeline = self._Pipeline()
        model = self._model(pipeline)

        properties = model._pipeline_design_properties(
            "transport pipeline",
            pipeline,
        )

        self.assertEqual(properties["designPressureDropCapacity_bar"], 0.005)
        self.assertEqual(properties["designVelocityCapacity_m_s"], 0.60)
        self.assertAlmostEqual(properties["pressureDropUtilization_pct"], 96.0)
        self.assertAlmostEqual(
            properties["velocityUtilization_pct"],
            96.66666666666667,
        )
        self.assertAlmostEqual(properties["pressureDropMargin_bar"], 0.0002)
        self.assertAlmostEqual(properties["velocityMargin_m_s"], 0.02)
        self.assertEqual(
            model._pipeline_design_constraint(
                "transport pipeline",
                pipeline,
            ).status,
            "OK",
        )
        expected_units = {
            "designPressureDropCapacity_bar": "bar",
            "designVelocityCapacity_m_s": "m/s",
            "pressureDropUtilization_pct": "%",
            "velocityUtilization_pct": "%",
            "pressureDropMargin_bar": "bar",
            "velocityMargin_m_s": "m/s",
        }
        for property_name, unit in expected_units.items():
            with self.subTest(property_name=property_name):
                self.assertEqual(
                    model._pipeline_design_property_unit(property_name),
                    unit,
                )

        kpis = {}
        model._extract_unit_properties(kpis)
        self.assertEqual(
            kpis[
                "transport pipeline.designPressureDropCapacity_bar"
            ].unit,
            "bar",
        )
        self.assertEqual(
            kpis["transport pipeline.velocityMargin_m_s"].unit,
            "m/s",
        )
        workbook_properties = model.list_units()[0].properties
        self.assertAlmostEqual(
            workbook_properties["pressureDropMargin_bar"],
            0.0002,
        )

    def test_validates_and_collects_enabled_pipeline_design_basis(self):
        self.assertEqual(
            ProcessBuilder._pipeline_design_settings(
                {
                    "use_design_basis": True,
                    "design_pressure_drop_capacity_bar": 0.005,
                    "design_velocity_capacity_m_per_s": 0.60,
                }
            ),
            (True, 0.005, 0.60),
        )
        units = [
            {
                "name": "transport pipeline",
                "type": "pipeline",
                "params": {
                    "length": 1_000.0,
                    "diameter": 0.30,
                    "roughness": 1.0e-5,
                    "use_design_basis": True,
                    "design_pressure_drop_capacity_bar": 0.005,
                    "design_velocity_capacity_m_per_s": 0.60,
                },
            },
            {
                "name": "spare pipeline",
                "type": "pipeline",
                "params": {"use_design_basis": False},
            },
        ]
        expected = {
            "transport pipeline": {
                "design_pressure_drop_capacity_bar": 0.005,
                "design_velocity_capacity_m_per_s": 0.60,
            }
        }
        self.assertEqual(
            ProcessBuilder._requested_pipeline_design_bases(units),
            expected,
        )
        self.assertEqual(
            ProcessBuilder._requested_equipment_design_bases(units),
            expected,
        )

        for params, message in (
            ({"use_design_basis": 1}, "must be boolean"),
            (
                {"design_pressure_drop_capacity_bar": math.nan},
                "must be finite",
            ),
            (
                {"design_velocity_capacity_m_per_s": 0.0},
                "must be between",
            ),
        ):
            with self.subTest(params=params):
                with self.assertRaisesRegex(ValueError, message):
                    ProcessBuilder._pipeline_design_settings(params)

        with self.assertRaisesRegex(ValueError, "only for pipeline units"):
            ProcessBuilder._requested_pipeline_design_bases(
                [
                    {
                        "name": "not a pipeline",
                        "type": "compressor",
                        "params": {
                            "design_pressure_drop_capacity_bar": 1.0,
                        },
                    }
                ]
            )

    def test_editor_defaults_migrate_legacy_geometry_without_enabling_screen(self):
        rows = process_unit_property_rows(
            "pipeline",
            {
                "length": 2_000.0,
                "diameter": 0.40,
                "roughness": 2.0e-5,
            },
        )
        row_by_key = {row["key"]: row for row in rows}
        self.assertFalse(row_by_key["use_design_basis"]["value"])
        self.assertEqual(
            row_by_key["design_pressure_drop_capacity_bar"]["value"],
            1.0,
        )
        self.assertEqual(
            row_by_key["design_pressure_drop_capacity_bar"]["unit"],
            "bar",
        )
        self.assertEqual(
            row_by_key["design_velocity_capacity_m_per_s"]["unit"],
            "m/s",
        )

        units, pipeline_id = add_catalog_unit(
            [],
            "pipeline",
            "transport pipeline",
        )
        self.assertFalse(units[0]["params"]["use_design_basis"])
        self.assertEqual(
            units[0]["params"]["design_pressure_drop_capacity_bar"],
            1.0,
        )
        self.assertEqual(pipeline_id, units[0]["id"])

    def test_violates_or_fails_loud_when_native_results_require_it(self):
        pipeline = self._Pipeline(
            pressure_drop_bar=0.0051,
            velocity_m_s=0.61,
        )
        model = self._model(pipeline)
        violation = model._pipeline_design_constraint(
            "transport pipeline",
            pipeline,
        )
        self.assertEqual(violation.status, "VIOLATION")
        self.assertIn("pressure drop", violation.detail)
        self.assertIn("velocity", violation.detail)

        pipeline.velocity_m_s = math.nan
        unknown = model._pipeline_design_constraint(
            "transport pipeline",
            pipeline,
        )
        self.assertEqual(unknown.status, "UNKNOWN")

    def test_supports_one_phase_velocity_and_absolute_pressure_drop(self):
        pipeline = self._Pipeline(
            pressure_drop_bar=-0.0048,
            velocity_m_s=-0.58,
            java_class="OnePhasePipeLine",
        )
        model = self._model(pipeline)
        properties = model._pipeline_design_properties(
            "transport pipeline",
            pipeline,
        )
        self.assertAlmostEqual(properties["pressureDropUtilization_pct"], 96.0)
        self.assertAlmostEqual(
            properties["velocityUtilization_pct"],
            96.66666666666667,
        )

    def test_screens_maximum_finite_native_velocity_profile(self):
        class _ProfilePipeline(self._Pipeline):
            def getMixtureSuperficialVelocityProfile(self):
                return [0.58, math.nan, -0.62, 0.59]

            def getLengthProfile(self):
                return [0.0, 500.0, 1_000.0, 1_500.0]

        pipeline = _ProfilePipeline(velocity_m_s=0.58)
        model = self._model(pipeline)

        properties = model._pipeline_design_properties(
            "transport pipeline",
            pipeline,
        )

        self.assertAlmostEqual(
            properties["velocityUtilization_pct"],
            103.33333333333333,
        )
        self.assertAlmostEqual(properties["velocityMargin_m_s"], -0.02)
        self.assertAlmostEqual(
            properties["governingHydraulicUtilization_pct"],
            103.33333333333333,
        )
        self.assertAlmostEqual(
            properties["governingHydraulicMargin_pct"],
            -3.3333333333333286,
        )
        self.assertEqual(properties["velocityCriticalSegment_index"], 2.0)
        self.assertEqual(properties["velocityCriticalLength_m"], 1_000.0)
        self.assertEqual(
            model._pipeline_design_property_unit(
                "velocityCriticalSegment_index"
            ),
            "[-]",
        )
        self.assertEqual(
            model._pipeline_design_property_unit("velocityCriticalLength_m"),
            "m",
        )
        constraint = model._pipeline_design_constraint(
            "transport pipeline",
            pipeline,
        )
        self.assertEqual(constraint.status, "VIOLATION")
        self.assertIn("velocity", constraint.detail)
        self.assertIn("governing=velocity", constraint.detail)

        kpis = {}
        model._extract_unit_properties(kpis)
        self.assertEqual(
            kpis[
                "transport pipeline.velocityCriticalSegment_index"
            ].value,
            2.0,
        )
        self.assertEqual(
            kpis["transport pipeline.velocityCriticalLength_m"].unit,
            "m",
        )

    def test_saved_metadata_accepts_only_exact_pipeline_capacity_schema(self):
        valid_basis = {
            "design_pressure_drop_capacity_bar": 0.005,
            "design_velocity_capacity_m_per_s": 0.60,
        }
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "neqsimweb2/studio_metadata.json",
                json.dumps(
                    {
                        "schema_version": 1,
                        "equipment_design_bases": {
                            "transport pipeline": valid_basis,
                        },
                    }
                ),
            )
        buffer.seek(0)
        with zipfile.ZipFile(buffer, "r") as archive:
            self.assertEqual(
                NeqSimProcessModel._read_studio_metadata(archive),
                {"transport pipeline": valid_basis},
            )

        for invalid_basis in (
            {"design_pressure_drop_capacity_bar": 0.005},
            {**valid_basis, "design_velocity_capacity_m_per_s": 0.0},
            {**valid_basis, "motor_rating_kw": 100.0},
        ):
            with self.subTest(invalid_basis=invalid_basis):
                invalid_buffer = io.BytesIO()
                with zipfile.ZipFile(invalid_buffer, "w") as archive:
                    archive.writestr(
                        "neqsimweb2/studio_metadata.json",
                        json.dumps(
                            {
                                "schema_version": 1,
                                "equipment_design_bases": {
                                    "transport pipeline": invalid_basis,
                                },
                            }
                        ),
                    )
                invalid_buffer.seek(0)
                with zipfile.ZipFile(invalid_buffer, "r") as archive:
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "equipment design metadata",
                    ):
                        NeqSimProcessModel._read_studio_metadata(archive)


class NativePipelineHydraulicsTest(unittest.TestCase):
    """Benchmark adiabatic native pipeline hydraulics and closure."""

    @staticmethod
    def _run_case(
        flow_scale: float,
        roughness_m: float,
        *,
        design_pressure_drop_capacity_bar: float | None = None,
        design_velocity_capacity_m_per_s: float | None = None,
    ):
        units, pipeline_id = add_catalog_unit(
            [],
            "pipeline",
            "transport pipeline",
        )
        units[0]["params"]["roughness"] = roughness_m
        if design_pressure_drop_capacity_bar is not None:
            units[0]["params"].update(
                {
                    "use_design_basis": True,
                    "design_pressure_drop_capacity_bar": (
                        design_pressure_drop_capacity_bar
                    ),
                    "design_velocity_capacity_m_per_s": (
                        design_velocity_capacity_m_per_s
                    ),
                }
            )
        graph_spec = {
            "name": "Native adiabatic pipeline benchmark",
            "units": units,
            "connections": [
                {
                    "id": "feed-to-transport-pipeline",
                    "type": "material",
                    "source": {
                        "kind": "inlet",
                        "id": "feed",
                        "port": "out",
                    },
                    "target": {
                        "kind": "unit",
                        "id": pipeline_id,
                        "port": "in",
                    },
                }
            ],
        }
        expected_flow = 10_000.0 * flow_scale
        inlet_specs = [
            {
                "inlet_id": "feed",
                "name": "feed",
                "fluid_spec": {
                    "eos_model": "srk",
                    "mixing_rule": 2,
                    "components": {
                        "methane": 0.90,
                        "ethane": 0.10,
                    },
                    "composition_basis": "mole_fraction",
                    "temperature_C": 20.0,
                    "pressure_bara": 80.0,
                    "total_flow": expected_flow,
                    "flow_unit": "kg/hr",
                },
            }
        ]
        builder = ProcessBuilder()
        model = builder.build_acyclic_graph(
            graph_spec,
            inlet_specs,
            ["feed", pipeline_id],
        )
        result = model.run(timeout_ms=180_000)

        return builder, graph_spec, model, result, expected_flow

    def test_native_design_screen_crosses_limits_at_nearby_point(self):
        results = {}
        for flow_scale in (1.0, 1.05):
            with self.subTest(flow_scale=flow_scale):
                _, _, model, result, expected_flow = self._run_case(
                    flow_scale,
                    1.0e-5,
                    design_pressure_drop_capacity_bar=0.005,
                    design_velocity_capacity_m_per_s=0.60,
                )
                constraint = next(
                    item
                    for item in result.constraints
                    if item.name == "pipeline_design.transport pipeline"
                )
                results[flow_scale] = (result, constraint)
                self.assertEqual(
                    model._equipment_design_bases,
                    {
                        "transport pipeline": {
                            "design_pressure_drop_capacity_bar": 0.005,
                            "design_velocity_capacity_m_per_s": 0.60,
                        }
                    },
                )
                self.assertAlmostEqual(
                    result.kpis["material_product_flow_kg_hr"].value,
                    expected_flow,
                    delta=max(1.0e-6 * expected_flow, 1.0e-3),
                )
                for balance_name in (
                    "mass_balance_pct",
                    "component_balance_max_pct",
                    "energy_balance_pct",
                    "unit_mass_balance_max_pct",
                    "unit_energy_balance_max_pct",
                ):
                    self.assertLess(
                        result.kpis[balance_name].value,
                        1.0e-6,
                    )
                self.assertAlmostEqual(
                    result.kpis[
                        "transport pipeline."
                        "governingHydraulicMargin_pct"
                    ].value,
                    min(
                        result.kpis[
                            "transport pipeline.pressureDropMargin_bar"
                        ].value
                        / 0.005
                        * 100.0,
                        result.kpis[
                            "transport pipeline.velocityMargin_m_s"
                        ].value
                        / 0.60
                        * 100.0,
                    ),
                )

        baseline, baseline_constraint = results[1.0]
        nearby, nearby_constraint = results[1.05]
        self.assertEqual(baseline_constraint.status, "OK")
        self.assertEqual(nearby_constraint.status, "VIOLATION")
        self.assertGreater(
            baseline.kpis[
                "transport pipeline.pressureDropMargin_bar"
            ].value,
            0.0,
        )
        self.assertGreater(
            baseline.kpis["transport pipeline.velocityMargin_m_s"].value,
            0.0,
        )
        self.assertLess(
            nearby.kpis[
                "transport pipeline.pressureDropMargin_bar"
            ].value,
            0.0,
        )
        self.assertLess(
            nearby.kpis["transport pipeline.velocityMargin_m_s"].value,
            0.0,
        )
        self.assertEqual(
            nearby.kpis[
                "transport pipeline.designPressureDropCapacity_bar"
            ].unit,
            "bar",
        )
        self.assertEqual(
            nearby.kpis[
                "transport pipeline.designVelocityCapacity_m_s"
            ].unit,
            "m/s",
        )
        baseline_governing = baseline.kpis[
            "transport pipeline.governingHydraulicUtilization_pct"
        ].value
        nearby_governing = nearby.kpis[
            "transport pipeline.governingHydraulicUtilization_pct"
        ].value
        print(
            "native pipeline design screen:",
            "baseline=OK",
            "nearby=VIOLATION",
            "baseline_drop_margin="
            f"{baseline.kpis['transport pipeline.pressureDropMargin_bar'].value:.9f} bar",
            "nearby_drop_margin="
            f"{nearby.kpis['transport pipeline.pressureDropMargin_bar'].value:.9f} bar",
            "baseline_velocity_margin="
            f"{baseline.kpis['transport pipeline.velocityMargin_m_s'].value:.9f} m/s",
            "nearby_velocity_margin="
            f"{nearby.kpis['transport pipeline.velocityMargin_m_s'].value:.9f} m/s",
            "baseline_governing="
            f"{baseline_governing:.9f}%",
            "nearby_governing="
            f"{nearby_governing:.9f}%",
        )

    def test_native_pipeline_conserves_and_trends_at_nearby_points(self):
        pressure_drop = {}
        velocity = {}
        reynolds = {}

        for flow_scale in (1.0, 1.05):
            for roughness_m in (1.0e-5, 1.0e-4):
                with self.subTest(
                    flow_scale=flow_scale,
                    roughness_m=roughness_m,
                ):
                    builder, graph_spec, model, result, expected_flow = (
                        self._run_case(flow_scale, roughness_m)
                    )
                    pipeline = model.get_unit("transport pipeline")
                    drop = result.kpis[
                        "transport pipeline.pressureDrop_bar"
                    ]
                    speed = result.kpis[
                        "transport pipeline.velocity_m_s"
                    ]
                    reynolds_number = result.kpis[
                        "transport pipeline.reynoldsNumber"
                    ]
                    pressure_drop[(flow_scale, roughness_m)] = drop.value
                    velocity[(flow_scale, roughness_m)] = speed.value
                    reynolds[(flow_scale, roughness_m)] = (
                        reynolds_number.value
                    )

                    self.assertEqual(
                        str(pipeline.getClass().getSimpleName()),
                        "PipeBeggsAndBrills",
                    )
                    self.assertEqual(
                        str(pipeline.getHeatTransferMode()),
                        "ADIABATIC",
                    )
                    self.assertEqual(drop.unit, "bar")
                    self.assertEqual(speed.unit, "m/s")
                    self.assertEqual(reynolds_number.unit, "[-]")
                    self.assertGreater(drop.value, 0.0)
                    self.assertLess(drop.value, 0.02)
                    self.assertGreater(speed.value, 0.0)
                    self.assertGreater(reynolds_number.value, 100_000.0)
                    self.assertAlmostEqual(
                        result.kpis[
                            "material_product_flow_kg_hr"
                        ].value,
                        expected_flow,
                        delta=max(1.0e-6 * expected_flow, 1.0e-3),
                    )
                    for balance_name in (
                        "mass_balance_pct",
                        "component_balance_max_pct",
                        "energy_balance_pct",
                        "unit_mass_balance_max_pct",
                        "unit_energy_balance_max_pct",
                    ):
                        self.assertLess(
                            result.kpis[balance_name].value,
                            1.0e-6,
                        )
                    self.assertFalse(
                        [
                            constraint
                            for constraint in result.constraints
                            if constraint.status == "VIOLATION"
                        ]
                    )
                    self.assertIn(
                        "Acyclic graph built and converged successfully.",
                        builder.build_log,
                    )
                    self.assertEqual(
                        json.loads(json.dumps(graph_spec, allow_nan=False)),
                        graph_spec,
                    )
                    print(
                        "native pipeline benchmark:",
                        f"scale={flow_scale:.2f}",
                        f"roughness={roughness_m:.1e} m",
                        f"drop={drop.value:.9f} bar",
                        f"velocity={speed.value:.9f} m/s",
                        f"Re={reynolds_number.value:.3f}",
                        f"mass={result.kpis['mass_balance_pct'].value:.3e}%",
                        "components="
                        f"{result.kpis['component_balance_max_pct'].value:.3e}%",
                        f"energy={result.kpis['energy_balance_pct'].value:.3e}%",
                    )

        for roughness_m in (1.0e-5, 1.0e-4):
            self.assertGreater(
                pressure_drop[(1.05, roughness_m)],
                pressure_drop[(1.0, roughness_m)],
            )
            self.assertGreater(
                velocity[(1.05, roughness_m)],
                velocity[(1.0, roughness_m)],
            )
            self.assertGreater(
                reynolds[(1.05, roughness_m)],
                reynolds[(1.0, roughness_m)],
            )
        for flow_scale in (1.0, 1.05):
            self.assertGreater(
                pressure_drop[(flow_scale, 1.0e-4)],
                pressure_drop[(flow_scale, 1.0e-5)],
            )


class NativeExpanderConservationTest(unittest.TestCase):
    """Benchmark native expander recovery and nearby-point closure."""

    @staticmethod
    def _run_case(flow_scale: float, efficiency: float):
        units, expander_id = add_catalog_unit(
            [],
            "expander",
            "turbo expander",
        )
        units[0]["params"].update(
            {
                "outlet_pressure_bara": 30.0,
                "isentropic_efficiency": efficiency,
            }
        )
        graph_spec = {
            "name": "Native expander recovery benchmark",
            "units": units,
            "connections": [
                {
                    "id": "feed-to-turbo-expander",
                    "type": "material",
                    "source": {
                        "kind": "inlet",
                        "id": "feed",
                        "port": "out",
                    },
                    "target": {
                        "kind": "unit",
                        "id": expander_id,
                        "port": "in",
                    },
                }
            ],
        }
        expected_flow = 10_000.0 * flow_scale
        inlet_specs = [
            {
                "inlet_id": "feed",
                "name": "feed",
                "fluid_spec": {
                    "eos_model": "srk",
                    "mixing_rule": 2,
                    "components": {
                        "methane": 0.90,
                        "ethane": 0.10,
                    },
                    "composition_basis": "mole_fraction",
                    "temperature_C": 30.0,
                    "pressure_bara": 80.0,
                    "total_flow": expected_flow,
                    "flow_unit": "kg/hr",
                },
            }
        ]
        builder = ProcessBuilder()
        model = builder.build_acyclic_graph(
            graph_spec,
            inlet_specs,
            ["feed", expander_id],
        )
        result = model.run(timeout_ms=180_000)

        return builder, graph_spec, result, expected_flow

    def test_native_recovery_conserves_and_trends_at_nearby_points(self):
        recovered_power = {}
        outlet_temperature = {}

        for flow_scale in (1.0, 1.05):
            for efficiency in (0.80, 0.70):
                with self.subTest(
                    flow_scale=flow_scale,
                    efficiency=efficiency,
                ):
                    builder, graph_spec, result, expected_flow = (
                        self._run_case(flow_scale, efficiency)
                    )
                    signed_power = result.kpis[
                        "turbo expander.power_kW"
                    ]
                    recovery = result.kpis[
                        "turbo expander.recoveredPower_kW"
                    ]
                    temperature = result.kpis[
                        "turbo expander.outletTemperature_K"
                    ]
                    recovered_power[(flow_scale, efficiency)] = (
                        recovery.value
                    )
                    outlet_temperature[(flow_scale, efficiency)] = (
                        temperature.value
                    )

                    self.assertEqual(signed_power.unit, "kW")
                    self.assertEqual(recovery.unit, "kW")
                    self.assertLess(signed_power.value, 0.0)
                    self.assertGreater(recovery.value, 0.0)
                    self.assertAlmostEqual(
                        recovery.value,
                        -signed_power.value,
                        delta=1.0e-9,
                    )
                    self.assertAlmostEqual(
                        result.kpis[
                            "turbo expander.outletPressure_bara"
                        ].value,
                        30.0,
                        delta=0.05,
                    )
                    self.assertAlmostEqual(
                        result.kpis[
                            "turbo expander.isentropicEfficiency"
                        ].value,
                        efficiency,
                        delta=1.0e-12,
                    )
                    self.assertAlmostEqual(
                        result.kpis[
                            "material_product_flow_kg_hr"
                        ].value,
                        expected_flow,
                        delta=max(1.0e-6 * expected_flow, 1.0e-3),
                    )
                    for balance_name in (
                        "mass_balance_pct",
                        "component_balance_max_pct",
                        "energy_balance_pct",
                        "unit_mass_balance_max_pct",
                        "unit_energy_balance_max_pct",
                    ):
                        self.assertLess(
                            result.kpis[balance_name].value,
                            1.0e-6,
                        )
                    self.assertFalse(
                        [
                            constraint
                            for constraint in result.constraints
                            if constraint.status == "VIOLATION"
                        ]
                    )
                    self.assertIn(
                        "Acyclic graph built and converged successfully.",
                        builder.build_log,
                    )
                    self.assertEqual(
                        json.loads(json.dumps(graph_spec, allow_nan=False)),
                        graph_spec,
                    )
                    print(
                        "native expander benchmark:",
                        f"scale={flow_scale:.2f}",
                        f"efficiency={efficiency:.2f}",
                        f"recovery={recovery.value:.6f} kW",
                        f"outlet={temperature.value:.6f} K",
                        f"mass={result.kpis['mass_balance_pct'].value:.3e}%",
                        "components="
                        f"{result.kpis['component_balance_max_pct'].value:.3e}%",
                        f"energy={result.kpis['energy_balance_pct'].value:.3e}%",
                    )

        for flow_scale in (1.0, 1.05):
            self.assertGreater(
                recovered_power[(flow_scale, 0.80)],
                recovered_power[(flow_scale, 0.70)],
            )
            self.assertLess(
                outlet_temperature[(flow_scale, 0.80)],
                outlet_temperature[(flow_scale, 0.70)],
            )
        for efficiency in (0.80, 0.70):
            self.assertAlmostEqual(
                recovered_power[(1.05, efficiency)]
                / recovered_power[(1.0, efficiency)],
                1.05,
                delta=1.0e-8,
            )
            self.assertAlmostEqual(
                outlet_temperature[(1.05, efficiency)],
                outlet_temperature[(1.0, efficiency)],
                delta=0.05,
            )


class MultiInletMixerConservationTest(unittest.TestCase):
    """Validate material and energy closure for independent graph inlets."""

    def test_resolves_explicit_heat_exchanger_output_sides(self):
        hot_out = object()
        cold_out = object()

        class _JavaClass:
            @staticmethod
            def getSimpleName():
                return "HeatExchanger"

        class _HeatExchanger:
            @staticmethod
            def getClass():
                return _JavaClass()

            @staticmethod
            def getOutStream(index):
                return [hot_out, cold_out][index]

        builder = ProcessBuilder()
        units = {"cross-exchanger": _HeatExchanger()}
        for port, expected in (
            ("hot_out", hot_out),
            ("cold_out", cold_out),
            ("out_0", hot_out),
            ("out_1", cold_out),
        ):
            with self.subTest(port=port):
                self.assertIs(
                    builder.resolve_material_output(
                        {
                            "kind": "unit",
                            "id": "cross-exchanger",
                            "port": port,
                        },
                        {},
                        units,
                    ),
                    expected,
                )

    @staticmethod
    def _build_two_sided_heat_exchanger_case(
        flow_scale: float,
        declared_input_ports=None,
        declared_output_ports=None,
        downstream_source_port=None,
        design_basis=None,
    ):
        inlet_specs = []
        for inlet_id, name, temperature_C, total_flow, components in (
            (
                "hot-feed",
                "hot feed",
                120.0,
                50_000.0,
                {"methane": 0.90, "ethane": 0.10},
            ),
            (
                "cold-feed",
                "cold feed",
                20.0,
                40_000.0,
                {"methane": 0.95, "ethane": 0.05},
            ),
        ):
            inlet_specs.append(
                {
                    "inlet_id": inlet_id,
                    "name": name,
                    "fluid_spec": {
                        "eos_model": "srk",
                        "mixing_rule": 2,
                        "components": components,
                        "composition_basis": "mole_fraction",
                        "temperature_C": temperature_C,
                        "pressure_bara": 50.0,
                        "total_flow": total_flow * flow_scale,
                        "flow_unit": "kg/hr",
                    },
                }
            )
        graph_spec = {
            "name": "Native two-sided heat exchanger benchmark",
            "units": [
                {
                    "id": "cross-exchanger",
                    "name": "cross exchanger",
                    "type": "heat_exchanger",
                    "ports": {
                        "material_in": list(
                            declared_input_ports
                            or ("hot_in", "cold_in")
                        ),
                        "material_out": list(
                            declared_output_ports
                            or ("hot_out", "cold_out")
                        ),
                    },
                    "params": {
                        "ua_w_per_k": 100_000.0,
                        **(
                            {
                                "use_design_basis": True,
                                **design_basis,
                            }
                            if design_basis is not None
                            else {}
                        ),
                    },
                }
            ],
            "connections": [
                {
                    "id": "hot-side-feed",
                    "name": "hot side feed",
                    "type": "material",
                    "source": {
                        "kind": "inlet",
                        "id": "hot-feed",
                        "port": "out",
                    },
                    "target": {
                        "kind": "unit",
                        "id": "cross-exchanger",
                        "port": "hot_in",
                    },
                },
                {
                    "id": "cold-side-feed",
                    "name": "cold side feed",
                    "type": "material",
                    "source": {
                        "kind": "inlet",
                        "id": "cold-feed",
                        "port": "out",
                    },
                    "target": {
                        "kind": "unit",
                        "id": "cross-exchanger",
                        "port": "cold_in",
                    },
                },
            ],
        }
        execution_order = ["hot-feed", "cold-feed", "cross-exchanger"]
        if downstream_source_port is not None:
            graph_spec["units"].append(
                {
                    "id": "hot-side-heater",
                    "name": "hot side heater",
                    "type": "heater",
                    "ports": {
                        "material_in": ["in"],
                        "material_out": ["out"],
                    },
                    "params": {"out_temperature_C": 60.0},
                }
            )
            graph_spec["connections"].append(
                {
                    "id": "hot-side-product",
                    "name": "hot side product",
                    "type": "material",
                    "source": {
                        "kind": "unit",
                        "id": "cross-exchanger",
                        "port": downstream_source_port,
                    },
                    "target": {
                        "kind": "unit",
                        "id": "hot-side-heater",
                        "port": "in",
                    },
                }
            )
            execution_order.append("hot-side-heater")
        builder = ProcessBuilder()
        model = builder.build_acyclic_graph(
            graph_spec,
            inlet_specs,
            execution_order,
        )
        return builder, model

    def test_native_exchanger_design_basis_round_trips_nearby_points(self):
        design_basis = {
            "design_duty_capacity_kw": 100_000.0,
            "design_ua_capacity_w_per_k": 125_000.0,
        }
        duties = {}
        for flow_scale in (1.0, 1.05):
            with self.subTest(flow_scale=flow_scale):
                builder, model = self._build_two_sided_heat_exchanger_case(
                    flow_scale,
                    design_basis=design_basis,
                )
                result = model.run(timeout_ms=180_000)
                duties[flow_scale] = result.kpis[
                    "cross exchanger.heatTransferDuty_kW"
                ].value
                self.assertEqual(
                    result.kpis[
                        "cross exchanger.designDutyCapacity_kW"
                    ].unit,
                    "kW",
                )
                self.assertEqual(
                    result.kpis[
                        "cross exchanger.designUACapacity_W_K"
                    ].unit,
                    "W/K",
                )
                self.assertAlmostEqual(
                    result.kpis[
                        "cross exchanger.uaUtilization_pct"
                    ].value,
                    80.0,
                    delta=1.0e-12,
                )
                constraint = next(
                    constraint
                    for constraint in result.constraints
                    if constraint.name
                    == "heat_exchanger_design.cross exchanger"
                )
                self.assertEqual(constraint.status, "OK")
                for balance_name in (
                    "mass_balance_pct",
                    "component_balance_max_pct",
                    "energy_balance_pct",
                    "unit_mass_balance_max_pct",
                    "unit_energy_balance_max_pct",
                ):
                    self.assertLess(
                        result.kpis[balance_name].value,
                        1.0e-6,
                    )

                saved_model = builder.save_neqsim_bytes()
                reloaded = NeqSimProcessModel.from_bytes(saved_model)
                reloaded_result = reloaded.run(timeout_ms=180_000)
                self.assertEqual(
                    reloaded_result.kpis[
                        "cross exchanger.designDutyCapacity_kW"
                    ].value,
                    100_000.0,
                )
                self.assertEqual(
                    next(
                        item.status
                        for item in reloaded_result.constraints
                        if item.name
                        == "heat_exchanger_design.cross exchanger"
                    ),
                    "OK",
                )
                self.assertIn(
                    "Registered equipment design basis for: "
                    "cross exchanger",
                    builder.build_log,
                )
                print(
                    "native exchanger design benchmark:",
                    f"scale={flow_scale:.2f}",
                    f"duty={duties[flow_scale]:.6f} kW",
                    "ua=100000.000000 W/K",
                    "ua_utilization=80.000000%",
                    f"mass={result.kpis['mass_balance_pct'].value:.3e}%",
                    "components="
                    f"{result.kpis['component_balance_max_pct'].value:.3e}%",
                    f"energy={result.kpis['energy_balance_pct'].value:.3e}%",
                )
        self.assertGreater(duties[1.05], duties[1.0])

    @staticmethod
    def _build_mixer_heat_exchanger_case():
        inlet_specs = [
            {
                "inlet_id": inlet_id,
                "name": name,
                "fluid_spec": {
                    "eos_model": "srk",
                    "mixing_rule": 2,
                    "components": components,
                    "composition_basis": "mole_fraction",
                    "temperature_C": temperature_C,
                    "pressure_bara": 50.0,
                    "total_flow": flow_kg_hr,
                    "flow_unit": "kg/hr",
                },
            }
            for (
                inlet_id,
                name,
                temperature_C,
                flow_kg_hr,
                components,
            ) in (
                (
                    "hot-feed-a",
                    "hot feed a",
                    100.0,
                    25_000.0,
                    {"methane": 0.90, "ethane": 0.10},
                ),
                (
                    "hot-feed-b",
                    "hot feed b",
                    140.0,
                    25_000.0,
                    {"methane": 0.90, "ethane": 0.10},
                ),
                (
                    "cold-feed",
                    "cold feed",
                    20.0,
                    40_000.0,
                    {"methane": 0.95, "ethane": 0.05},
                ),
            )
        ]
        graph_spec = {
            "name": "Mixer and heat-exchanger clone benchmark",
            "units": [
                {
                    "id": "hot-mixer",
                    "name": "hot mixer",
                    "type": "mixer",
                    "ports": {
                        "material_in": ["in_0", "in_1"],
                        "material_out": ["out"],
                    },
                    "params": {},
                },
                {
                    "id": "cross-exchanger",
                    "name": "cross exchanger",
                    "type": "heat_exchanger",
                    "ports": {
                        "material_in": ["hot_in", "cold_in"],
                        "material_out": ["hot_out", "cold_out"],
                    },
                    "params": {"ua_w_per_k": 100_000.0},
                },
            ],
            "connections": [
                {
                    "id": "hot-a-to-mixer",
                    "type": "material",
                    "source": {
                        "kind": "inlet",
                        "id": "hot-feed-a",
                        "port": "out",
                    },
                    "target": {
                        "kind": "unit",
                        "id": "hot-mixer",
                        "port": "in_0",
                    },
                },
                {
                    "id": "hot-b-to-mixer",
                    "type": "material",
                    "source": {
                        "kind": "inlet",
                        "id": "hot-feed-b",
                        "port": "out",
                    },
                    "target": {
                        "kind": "unit",
                        "id": "hot-mixer",
                        "port": "in_1",
                    },
                },
                {
                    "id": "mixer-to-exchanger",
                    "type": "material",
                    "source": {
                        "kind": "unit",
                        "id": "hot-mixer",
                        "port": "out",
                    },
                    "target": {
                        "kind": "unit",
                        "id": "cross-exchanger",
                        "port": "hot_in",
                    },
                },
                {
                    "id": "cold-to-exchanger",
                    "type": "material",
                    "source": {
                        "kind": "inlet",
                        "id": "cold-feed",
                        "port": "out",
                    },
                    "target": {
                        "kind": "unit",
                        "id": "cross-exchanger",
                        "port": "cold_in",
                    },
                },
            ],
        }
        builder = ProcessBuilder()
        model = builder.build_acyclic_graph(
            graph_spec,
            inlet_specs,
            [
                "hot-feed-a",
                "hot-feed-b",
                "cold-feed",
                "hot-mixer",
                "cross-exchanger",
            ],
        )
        return builder, model

    def test_rejects_noncanonical_heat_exchanger_inlet_contract(self):
        for declared_ports in (
            ("cold_in", "hot_in"),
            ("hot_in", "cold_in", "spare_in"),
        ):
            with self.subTest(declared_ports=declared_ports):
                with self.assertRaisesRegex(
                    ValueError,
                    "fixed order: hot_in, cold_in",
                ):
                    self._build_two_sided_heat_exchanger_case(
                        1.0,
                        declared_ports,
                    )

    def test_rejects_undeclared_heat_exchanger_output_alias(self):
        with self.assertRaisesRegex(
            ValueError,
            "uses undeclared material output port 'out'",
        ):
            self._build_two_sided_heat_exchanger_case(
                1.0,
                downstream_source_port="out",
            )

    def test_rejects_noncanonical_heat_exchanger_output_contract(self):
        for declared_ports in (
            ("out", "hot_out", "cold_out"),
            ("cold_out", "hot_out"),
        ):
            with self.subTest(declared_ports=declared_ports):
                with self.assertRaisesRegex(
                    ValueError,
                    "fixed order: hot_out, cold_out",
                ):
                    self._build_two_sided_heat_exchanger_case(
                        1.0,
                        declared_output_ports=declared_ports,
                    )

    def test_accepts_indexed_heat_exchanger_output_aliases(self):
        _, model = self._build_two_sided_heat_exchanger_case(
            1.0,
            declared_output_ports=("out_0", "out_1"),
        )
        result = model.run(timeout_ms=180_000)
        self.assertEqual(result.kpis["material_product_count"].value, 2.0)
        self.assertLess(result.kpis["mass_balance_pct"].value, 1.0e-6)

    def test_native_two_sided_heat_exchanger_conserves_nearby_points(self):
        outlet_temperatures = []
        solved_duties = []
        solved_effectiveness = []
        for flow_scale in (1.0, 1.05):
            with self.subTest(flow_scale=flow_scale):
                builder, model = (
                    self._build_two_sided_heat_exchanger_case(flow_scale)
                )
                result = model.run(timeout_ms=180_000)
                exchanger = model.get_unit("cross exchanger")
                hot_out_C = float(
                    exchanger.getOutStream(0).getTemperature("C")
                )
                cold_out_C = float(
                    exchanger.getOutStream(1).getTemperature("C")
                )
                outlet_temperatures.append((hot_out_C, cold_out_C))

                self.assertEqual(
                    builder.spec["graph"]["units"][0]["params"],
                    {"ua_w_per_k": 100_000.0},
                )
                self.assertAlmostEqual(
                    float(exchanger.getUAvalue()),
                    100_000.0,
                )
                self.assertGreater(hot_out_C, 20.0)
                self.assertLess(hot_out_C, 120.0)
                self.assertGreater(cold_out_C, 20.0)
                self.assertLess(cold_out_C, 120.0)
                self.assertGreater(
                    float(exchanger.getApproachTemperature()),
                    0.0,
                )

                exchanger_properties = next(
                    unit.properties
                    for unit in model.list_units()
                    if unit.name == "cross exchanger"
                )
                solved_duties.append(
                    exchanger_properties["heatTransferDuty_kW"]
                )
                solved_effectiveness.append(
                    exchanger_properties["thermalEffectiveness"]
                )
                for side, expected_flow in (
                    ("hot", 50_000.0 * flow_scale),
                    ("cold", 40_000.0 * flow_scale),
                ):
                    self.assertAlmostEqual(
                        exchanger_properties[f"{side}InletFlow_kg_hr"],
                        expected_flow,
                        delta=1.0e-6 * expected_flow,
                    )
                    self.assertAlmostEqual(
                        exchanger_properties[f"{side}OutletFlow_kg_hr"],
                        expected_flow,
                        delta=1.0e-6 * expected_flow,
                    )
                self.assertAlmostEqual(
                    exchanger_properties["hotOutletTemperature_C"],
                    hot_out_C,
                    delta=1.0e-10,
                )
                self.assertAlmostEqual(
                    exchanger_properties["coldOutletTemperature_C"],
                    cold_out_C,
                    delta=1.0e-10,
                )
                self.assertAlmostEqual(
                    exchanger_properties["UA_W_K"],
                    100_000.0,
                    delta=1.0e-10,
                )
                self.assertGreater(
                    exchanger_properties["heatTransferDuty_kW"],
                    0.0,
                )
                self.assertAlmostEqual(
                    exchanger_properties["hotSideDuty_kW"],
                    exchanger_properties["heatTransferDuty_kW"],
                    delta=1.0e-5,
                )
                self.assertAlmostEqual(
                    exchanger_properties["coldSideDuty_kW"],
                    exchanger_properties["heatTransferDuty_kW"],
                    delta=1.0e-5,
                )
                self.assertLess(
                    exchanger_properties["dutyClosure_pct"],
                    1.0e-6,
                )
                for property_name, unit in (
                    ("UA_W_K", "W/K"),
                    ("heatTransferDuty_kW", "kW"),
                    ("hotSideDuty_kW", "kW"),
                    ("coldSideDuty_kW", "kW"),
                    ("dutyClosure_pct", "%"),
                ):
                    kpi = result.kpis[
                        f"cross exchanger.{property_name}"
                    ]
                    self.assertAlmostEqual(
                        kpi.value,
                        exchanger_properties[property_name],
                        delta=1.0e-10,
                    )
                    self.assertEqual(kpi.unit, unit)

                self.assertEqual(
                    result.kpis["material_feed_count"].value,
                    2.0,
                )
                self.assertEqual(
                    result.kpis["material_product_count"].value,
                    2.0,
                )
                self.assertLess(
                    result.kpis["mass_balance_pct"].value,
                    1.0e-6,
                )
                self.assertLess(
                    result.kpis["component_balance_max_pct"].value,
                    1.0e-6,
                )
                self.assertLess(
                    result.kpis["energy_balance_pct"].value,
                    1.0e-6,
                )
                validation = {
                    constraint.name: constraint.status
                    for constraint in result.constraints
                }
                self.assertEqual(validation["mass_balance"], "OK")
                self.assertEqual(validation["component_balance"], "OK")
                self.assertEqual(validation["energy_balance"], "OK")

                exchanger_rows = [
                    row
                    for row in unit_balance_rows(result)
                    if row["unit_name"] == "cross exchanger"
                ]
                self.assertEqual(len(exchanger_rows), 1)
                self.assertEqual(exchanger_rows[0]["inlet_count"], 2)
                self.assertEqual(exchanger_rows[0]["outlet_count"], 2)
                unit_summary = aggregate_unit_balances(result)
                self.assertTrue(unit_summary["coverage_complete"])
                self.assertLess(
                    unit_summary["max_mass_imbalance_pct"],
                    1.0e-6,
                )
                self.assertLess(
                    unit_summary["max_energy_imbalance_pct"],
                    1.0e-6,
                )
                print(
                    "native heat-exchanger benchmark:",
                    f"scale={flow_scale:.2f}",
                    "duty="
                    f"{exchanger_properties['heatTransferDuty_kW']:.6f} kW",
                    f"effectiveness={exchanger_properties['thermalEffectiveness']:.6f}",
                    "side-closure="
                    f"{exchanger_properties['dutyClosure_pct']:.3e}%",
                    "system-energy="
                    f"{result.kpis['energy_balance_pct'].value:.3e}%",
                )

        self.assertGreater(
            outlet_temperatures[1][0],
            outlet_temperatures[0][0],
        )
        self.assertLess(
            outlet_temperatures[1][1],
            outlet_temperatures[0][1],
        )
        self.assertGreater(solved_duties[1], solved_duties[0])
        self.assertLess(solved_effectiveness[1], solved_effectiveness[0])

    def test_native_co_current_approach_uses_parallel_terminals(self):
        _, model = self._build_two_sided_heat_exchanger_case(1.0)
        exchanger = model.get_unit("cross exchanger")
        exchanger.setFlowArrangement("co-current")

        result = model.run(timeout_ms=180_000)
        properties = next(
            unit.properties
            for unit in model.list_units()
            if unit.name == "cross exchanger"
        )
        hot_in_C = float(exchanger.getInStream(0).getTemperature("C"))
        cold_in_C = float(exchanger.getInStream(1).getTemperature("C"))
        hot_out_C = float(exchanger.getOutStream(0).getTemperature("C"))
        cold_out_C = float(exchanger.getOutStream(1).getTemperature("C"))
        expected_approach_K = min(
            hot_in_C - cold_in_C,
            hot_out_C - cold_out_C,
        )

        self.assertEqual(str(exchanger.getFlowArrangement()), "co-current")
        self.assertLess(expected_approach_K, 0.0)
        self.assertAlmostEqual(
            properties["approachTemperature_K"],
            expected_approach_K,
            delta=1.0e-10,
        )
        self.assertLess(result.kpis["mass_balance_pct"].value, 1.0e-6)
        self.assertLess(
            result.kpis["component_balance_max_pct"].value,
            1.0e-6,
        )
        self.assertLess(result.kpis["energy_balance_pct"].value, 1.0e-6)

    def test_native_setting_edits_invalidate_exchanger_snapshot(self):
        _, model = self._build_two_sided_heat_exchanger_case(1.0)
        model.run(timeout_ms=180_000)
        exchanger = model.get_unit("cross exchanger")

        exchanger.setUAvalue(110_000.0)
        self.assertNotIn(
            "heatTransferDuty_kW",
            next(
                unit.properties
                for unit in model.list_units()
                if unit.name == "cross exchanger"
            ),
        )

        result = model.run(timeout_ms=180_000)
        exchanger.setFlowArrangement("co-current")
        self.assertNotIn(
            "heatTransferDuty_kW",
            next(
                unit.properties
                for unit in model.list_units()
                if unit.name == "cross exchanger"
            ),
        )
        model.run(timeout_ms=180_000)
        exchanger.setThermalEffectiveness(0.25)
        self.assertNotIn(
            "heatTransferDuty_kW",
            next(
                unit.properties
                for unit in model.list_units()
                if unit.name == "cross exchanger"
            ),
        )
        exchanger.setDeltaT(5.0)
        exchanger.setUseDeltaT(False)
        model.run(timeout_ms=180_000)
        exchanger.setUseDeltaT(True)
        self.assertNotIn(
            "heatTransferDuty_kW",
            next(
                unit.properties
                for unit in model.list_units()
                if unit.name == "cross exchanger"
            ),
        )
        self.assertLess(result.kpis["mass_balance_pct"].value, 1.0e-6)
        self.assertLess(
            result.kpis["component_balance_max_pct"].value,
            1.0e-6,
        )
        self.assertLess(result.kpis["energy_balance_pct"].value, 1.0e-6)

    def test_native_rating_edits_invalidate_exchanger_snapshot(self):
        from neqsim import jneqsim

        _, model = self._build_two_sided_heat_exchanger_case(1.0)
        model.run(timeout_ms=180_000)
        exchanger = model.get_unit("cross exchanger")
        rating_calculator = (
            jneqsim.process.mechanicaldesign.heatexchanger
            .ThermalDesignCalculator()
        )

        exchanger.setRatingCalculator(rating_calculator)
        exchanger.setRatingArea(1_000.0)
        self.assertNotIn(
            "heatTransferDuty_kW",
            next(
                unit.properties
                for unit in model.list_units()
                if unit.name == "cross exchanger"
            ),
        )

        model.run(timeout_ms=180_000)
        rating_calculator.setTubeCount(500)
        self.assertNotIn(
            "heatTransferDuty_kW",
            next(
                unit.properties
                for unit in model.list_units()
                if unit.name == "cross exchanger"
            ),
        )

    def test_native_locked_inactive_invalidates_exchanger_snapshot(self):
        _, model = self._build_two_sided_heat_exchanger_case(1.0)
        model.run(timeout_ms=180_000)
        exchanger = model.get_unit("cross exchanger")

        self.assertFalse(bool(exchanger.isLockedInactive()))
        with patch.object(
            NeqSimProcessModel,
            "_heat_exchanger_boundary_state_signature",
            wraps=(
                NeqSimProcessModel._heat_exchanger_boundary_state_signature
            ),
        ) as signature:
            listed_properties = next(
                unit.properties
                for unit in model.list_units()
                if unit.name == "cross exchanger"
            )
        self.assertIn("heatTransferDuty_kW", listed_properties)
        self.assertEqual(signature.call_count, 1)
        with patch.object(
            NeqSimProcessModel,
            "_heat_exchanger_boundary_state_signature",
            wraps=(
                NeqSimProcessModel._heat_exchanger_boundary_state_signature
            ),
        ) as signature:
            trusted_result = model._extract_results()
        self.assertIn(
            "cross exchanger.duty_kW",
            trusted_result.kpis,
        )
        self.assertEqual(signature.call_count, 1)
        solved_snapshot = model._heat_exchanger_state_snapshots[
            "cross exchanger"
        ]
        model._heat_exchanger_state_snapshots = {
            "train-a/cross exchanger": solved_snapshot,
        }
        self.assertTrue(
            model._heat_exchanger_solution_is_trusted(
                "cross exchanger",
                exchanger,
                "HeatExchanger",
            )
        )
        model._heat_exchanger_state_snapshots[
            "train-b/cross exchanger"
        ] = solved_snapshot
        self.assertFalse(
            model._heat_exchanger_solution_is_trusted(
                "cross exchanger",
                exchanger,
                "HeatExchanger",
            )
        )
        model._heat_exchanger_state_snapshots = {
            "cross exchanger": solved_snapshot,
        }
        exchanger.setEnergyInput(123_456.0)
        self.assertFalse(
            model._heat_exchanger_solution_is_trusted(
                "cross exchanger",
                exchanger,
                "HeatExchanger",
            )
        )
        active_unsolved_result = model._extract_results()
        self.assertNotIn(
            "cross exchanger.energyInput_W",
            active_unsolved_result.kpis,
        )
        self.assertNotIn(
            "cross exchanger.duty_kW",
            active_unsolved_result.kpis,
        )
        exchanger.setLockedInactive(True)
        self.assertTrue(bool(exchanger.isLockedInactive()))
        inactive_properties = next(
            unit.properties
            for unit in model.list_units()
            if unit.name == "cross exchanger"
        )
        self.assertNotIn("heatTransferDuty_kW", inactive_properties)
        self.assertNotIn("duty_kW", inactive_properties)

        result = model.run(timeout_ms=180_000)
        rerun_properties = next(
            unit.properties
            for unit in model.list_units()
            if unit.name == "cross exchanger"
        )
        self.assertNotIn("heatTransferDuty_kW", rerun_properties)
        self.assertNotIn("duty_kW", rerun_properties)
        self.assertNotIn("cross exchanger.duty_kW", result.kpis)
        self.assertNotIn("cross exchanger.energyInput_W", result.kpis)
        self.assertNotIn("report.cross exchanger.duty", result.kpis)
        self.assertNotIn(
            "report.cross exchanger.dutyBalance",
            result.kpis,
        )
        module_name = str(model.get_process().getName())
        for public_report in (
            result.json_report,
            model.get_json_report(),
            model.get_unit_json_report("cross exchanger"),
            model.get_module_json_report(module_name),
        ):
            self.assertIsNotNone(public_report)
            exchanger_report = public_report["cross exchanger"]
            self.assertNotIn("duty", exchanger_report)
            self.assertNotIn("dutyBalance", exchanger_report)
        self.assertEqual(result.kpis["total_duty_kW"].value, 0.0)
        exchanger_summary = next(
            line
            for line in model.get_model_summary().splitlines()
            if "cross exchanger (HeatExchanger)" in line
        )
        self.assertNotIn("duty_kW=", exchanger_summary)
        prefixed_report_kpis = {}
        model._unit_ps_name["cross exchanger"] = "train-a"
        model._flatten_json_report(
            {
                "train-a/cross exchanger": {
                    "duty": float(exchanger.getDuty()),
                    "dutyBalance": 1.0,
                    "feedTemperature1": 120.0,
                },
            },
            prefixed_report_kpis,
        )
        self.assertNotIn(
            "report.train-a/cross exchanger.duty",
            prefixed_report_kpis,
        )
        self.assertNotIn(
            "report.train-a/cross exchanger.dutyBalance",
            prefixed_report_kpis,
        )
        self.assertEqual(
            prefixed_report_kpis[
                "report.train-a/cross exchanger.feedTemperature1"
            ].value,
            120.0,
        )
        filtered_prefixed_report = model._filter_json_report_duties(
            {
                "train-a/cross exchanger": {
                    "duty": float(exchanger.getDuty()),
                    "dutyBalance": 1.0,
                    "feedTemperature1": 120.0,
                },
            }
        )
        self.assertEqual(
            filtered_prefixed_report["train-a/cross exchanger"],
            {"feedTemperature1": 120.0},
        )
        nested_report_kpis = {}
        model._flatten_json_report(
            {
                "train-a": {
                    "cross exchanger": {
                        "duty": float(exchanger.getDuty()),
                        "dutyBalance": 1.0,
                        "feedTemperature1": 120.0,
                    },
                },
            },
            nested_report_kpis,
        )
        self.assertNotIn(
            "report.train-a.cross exchanger.duty",
            nested_report_kpis,
        )
        self.assertNotIn(
            "report.train-a.cross exchanger.dutyBalance",
            nested_report_kpis,
        )
        self.assertEqual(
            nested_report_kpis[
                "report.train-a.cross exchanger.feedTemperature1"
            ].value,
            120.0,
        )
        model._is_process_model = True
        model._units["train-a"] = model.get_unit("hot side feed")
        model._unit_ps_name["train-a"] = "train-a"
        collision_report_kpis = {}
        model._flatten_json_report(
            {
                "train-a": {
                    "cross exchanger": {
                        "duty": float(exchanger.getDuty()),
                        "dutyBalance": 1.0,
                        "feedTemperature1": 120.0,
                    },
                },
            },
            collision_report_kpis,
        )
        self.assertNotIn(
            "report.train-a.cross exchanger.duty",
            collision_report_kpis,
        )
        self.assertNotIn(
            "report.train-a.cross exchanger.dutyBalance",
            collision_report_kpis,
        )
        self.assertEqual(
            collision_report_kpis[
                "report.train-a.cross exchanger.feedTemperature1"
            ].value,
            120.0,
        )
        filtered_nested_report = model._filter_json_report_duties(
            {
                "train-a": {
                    "cross exchanger": {
                        "duty": float(exchanger.getDuty()),
                        "dutyBalance": 1.0,
                        "feedTemperature1": 120.0,
                    },
                },
            }
        )
        self.assertEqual(
            filtered_nested_report["train-a"]["cross exchanger"],
            {"feedTemperature1": 120.0},
        )
        del model._units["train-a"]
        del model._unit_ps_name["train-a"]
        model._is_process_model = False

        exchanger.setLockedInactive(False)
        reenabled_properties = next(
            unit.properties
            for unit in model.list_units()
            if unit.name == "cross exchanger"
        )
        self.assertNotIn("duty_kW", reenabled_properties)
        unsolved_result = model._extract_results()
        self.assertNotIn("cross exchanger.duty_kW", unsolved_result.kpis)
        self.assertNotIn(
            "cross exchanger.energyInput_W",
            unsolved_result.kpis,
        )
        self.assertNotIn(
            "report.cross exchanger.duty",
            unsolved_result.kpis,
        )
        self.assertNotIn(
            "report.cross exchanger.dutyBalance",
            unsolved_result.kpis,
        )
        self.assertEqual(unsolved_result.kpis["total_duty_kW"].value, 0.0)

        solved_result = model.run(timeout_ms=180_000)
        solved_properties = next(
            unit.properties
            for unit in model.list_units()
            if unit.name == "cross exchanger"
        )
        self.assertGreater(solved_properties["duty_kW"], 0.0)
        self.assertGreater(
            solved_result.kpis["cross exchanger.duty_kW"].value,
            0.0,
        )
        self.assertIn(
            "cross exchanger.energyInput_W",
            solved_result.kpis,
        )
        self.assertGreater(
            solved_result.kpis["report.cross exchanger.duty"].value,
            0.0,
        )
        self.assertIn(
            "report.cross exchanger.dutyBalance",
            solved_result.kpis,
        )
        self.assertLess(
            solved_result.kpis["mass_balance_pct"].value,
            1.0e-6,
        )

    def test_rewrapping_edited_process_does_not_trust_stale_snapshot(self):
        _, model = self._build_two_sided_heat_exchanger_case(1.0)
        model.run(timeout_ms=180_000)
        process_system = model.get_process()
        model.get_unit("cross exchanger").getInStream(0).setTemperature(
            110.0,
            "C",
        )

        wrapped_model = NeqSimProcessModel.from_process_system(
            process_system
        )
        properties = next(
            unit.properties
            for unit in wrapped_model.list_units()
            if unit.name == "cross exchanger"
        )

        self.assertNotIn("heatTransferDuty_kW", properties)

    def test_same_named_module_exchangers_keep_qualified_trust(self):
        from neqsim import jneqsim

        _, train_a_model = self._build_two_sided_heat_exchanger_case(1.0)
        _, train_b_model = self._build_two_sided_heat_exchanger_case(1.05)
        train_a = train_a_model.get_process()
        train_b = train_b_model.get_process()
        train_a.setName("train-a")
        train_b.setName("train-b")
        process_model = jneqsim.process.processmodel.ProcessModel()
        self.assertTrue(process_model.add("train-a", train_a))
        self.assertTrue(process_model.add("train-b", train_b))
        model = NeqSimProcessModel(process_model)

        model.run(timeout_ms=180_000)
        solved_summary_lines = [
            line
            for line in model.get_model_summary().splitlines()
            if "cross exchanger (HeatExchanger)" in line
        ]
        self.assertEqual(len(solved_summary_lines), 2)
        self.assertTrue(
            all("duty_kW=" in line for line in solved_summary_lines)
        )
        solved_dot = model._generate_dot_fallback()
        self.assertIn("2389.2 kW", solved_dot)
        self.assertIn("2480.0 kW", solved_dot)

        train_b.setName("train-b-renamed")
        model.get_unit(
            "train-b/cross exchanger"
        ).setLockedInactive(True)
        model.run(timeout_ms=180_000)

        train_a_report = model.get_module_json_report("train-a")
        train_b_report = model.get_module_json_report(
            "train-b-renamed"
        )
        self.assertIn("duty", train_a_report["cross exchanger"])
        self.assertIn("dutyBalance", train_a_report["cross exchanger"])
        self.assertNotIn("duty", train_b_report["cross exchanger"])
        self.assertNotIn(
            "dutyBalance",
            train_b_report["cross exchanger"],
        )
        full_report = model.get_json_report()
        renamed_exchanger_report = full_report[
            "train-b-renamed"
        ]["cross exchanger"]
        self.assertNotIn("duty", renamed_exchanger_report)
        self.assertNotIn("dutyBalance", renamed_exchanger_report)

        train_b.setName("")
        empty_named_exchanger_report = model.get_json_report()[
            ""
        ]["cross exchanger"]
        self.assertNotIn("duty", empty_named_exchanger_report)
        self.assertNotIn(
            "dutyBalance",
            empty_named_exchanger_report,
        )

        class _FailingAggregateReport:
            @staticmethod
            def getReport_json():
                raise RuntimeError("force child-report fallback")

            @staticmethod
            def getAllProcesses():
                return process_model.getAllProcesses()

        model._proc = _FailingAggregateReport()
        fallback_report = model.get_json_report()[
            "process/cross exchanger"
        ]
        self.assertNotIn("duty", fallback_report)
        self.assertNotIn("dutyBalance", fallback_report)
        fallback_result = model._extract_results()
        self.assertNotIn(
            "report.process/cross exchanger.duty",
            fallback_result.kpis,
        )
        self.assertNotIn(
            "report.process/cross exchanger.dutyBalance",
            fallback_result.kpis,
        )
        model._proc = process_model
        train_b.setName("train-b-renamed")

        summary = model.get_model_summary()
        train_a_section = summary.split(
            "== Process System: train-a ==",
            1,
        )[1].split(
            "== Process System: train-b-renamed ==",
            1,
        )[0]
        train_b_section = summary.split(
            "== Process System: train-b-renamed ==",
            1,
        )[1]
        train_a_exchanger = next(
            line
            for line in train_a_section.splitlines()
            if "cross exchanger (HeatExchanger)" in line
        )
        train_b_exchanger = next(
            line
            for line in train_b_section.splitlines()
            if "cross exchanger (HeatExchanger)" in line
        )
        self.assertIn("duty_kW=", train_a_exchanger)
        self.assertNotIn("duty_kW=", train_b_exchanger)

        _, replacement_model = (
            self._build_two_sided_heat_exchanger_case(1.0)
        )
        replacement_model.run(timeout_ms=180_000)
        replacement_exchanger = replacement_model.get_unit(
            "cross exchanger"
        )
        replacement_exchanger.setLockedInactive(True)
        train_a.removeUnit("cross exchanger")
        train_a.add(replacement_exchanger)

        replacement_report = model.get_json_report()[
            "train-a"
        ]["cross exchanger"]
        self.assertNotIn("duty", replacement_report)
        self.assertNotIn("dutyBalance", replacement_report)
        replacement_result = model._extract_results()
        self.assertNotIn(
            "report.train-a.cross exchanger.duty",
            replacement_result.kpis,
        )
        self.assertNotIn(
            "report.train-a.cross exchanger.dutyBalance",
            replacement_result.kpis,
        )
        self.assertNotIn(
            "train-a/cross exchanger.duty_kW",
            replacement_result.kpis,
        )
        self.assertNotIn(
            "train-a/cross exchanger.energyInput_W",
            replacement_result.kpis,
        )
        self.assertEqual(
            replacement_result.kpis["total_duty_kW"].value,
            0.0,
        )

    def test_replaced_native_exchanger_invalidates_all_solved_outputs(self):
        _, model = self._build_two_sided_heat_exchanger_case(1.0)
        model.run(timeout_ms=180_000)
        process_system = model.get_process()

        _, replacement_model = (
            self._build_two_sided_heat_exchanger_case(1.05)
        )
        replacement_model.run(timeout_ms=180_000)
        replacement_exchanger = replacement_model.get_unit(
            "cross exchanger"
        )
        replacement_exchanger.setLockedInactive(True)
        process_system.removeUnit("cross exchanger")
        process_system.add(replacement_exchanger)

        workbook_properties = next(
            unit.properties
            for unit in model.list_units()
            if unit.name == "cross exchanger"
        )
        self.assertNotIn("duty_kW", workbook_properties)
        self.assertNotIn(
            "heatTransferDuty_kW",
            workbook_properties,
        )
        public_report = model.get_json_report()["cross exchanger"]
        self.assertNotIn("duty", public_report)
        self.assertNotIn("dutyBalance", public_report)

        result = model._extract_results()
        self.assertNotIn("cross exchanger.duty_kW", result.kpis)
        self.assertNotIn(
            "cross exchanger.energyInput_W",
            result.kpis,
        )
        self.assertNotIn(
            "cross exchanger.heatTransferDuty_kW",
            result.kpis,
        )
        self.assertNotIn(
            "report.cross exchanger.duty",
            result.kpis,
        )
        self.assertNotIn(
            "report.cross exchanger.dutyBalance",
            result.kpis,
        )
        self.assertEqual(result.kpis["total_duty_kW"].value, 0.0)

    def test_case_distinct_module_exchangers_prefer_exact_identity(self):
        from neqsim import jneqsim

        _, train_a_model = self._build_two_sided_heat_exchanger_case(1.0)
        _, train_b_model = self._build_two_sided_heat_exchanger_case(1.05)
        train_b_model.get_unit("cross exchanger").setName(
            "Cross Exchanger"
        )
        train_a = train_a_model.get_process()
        train_b = train_b_model.get_process()
        train_a.setName("train-a")
        train_b.setName("train-b")
        process_model = jneqsim.process.processmodel.ProcessModel()
        self.assertTrue(process_model.add("train-a", train_a))
        self.assertTrue(process_model.add("train-b", train_b))
        model = NeqSimProcessModel(process_model)

        result = model.run(timeout_ms=180_000)
        self.assertIn("cross exchanger.duty_kW", result.kpis)
        self.assertIn("Cross Exchanger.duty_kW", result.kpis)
        self.assertIn(
            "cross exchanger",
            model._heat_exchanger_state_snapshots,
        )
        self.assertIn(
            "Cross Exchanger",
            model._heat_exchanger_state_snapshots,
        )
        summary_lines = [
            line
            for line in model.get_model_summary().splitlines()
            if "exchanger (heatexchanger)" in line.casefold()
        ]
        self.assertEqual(len(summary_lines), 2)
        self.assertTrue(all("duty_kW=" in line for line in summary_lines))
        dot = model._generate_dot_fallback()
        self.assertIn("2389.2 kW", dot)
        self.assertIn("2480.0 kW", dot)

    def test_failed_rerun_clears_unchanged_direct_run_provenance(self):
        _, model = self._build_mixer_heat_exchanger_case()
        model.run(timeout_ms=180_000)
        self.assertIn(
            "cross exchanger",
            model._direct_unit_run_provenance,
        )

        with self.assertRaisesRegex(
            ProcessExecutionError,
            "discard this process model",
        ):
            with patch.object(
                model,
                "_run_until_converged",
                return_value=False,
            ), patch.object(
                model,
                "_run_acyclic_mixer_energy_closure",
                side_effect=AssertionError(
                    "failed process run must not start direct closure"
                ),
            ):
                model.rerun(timeout_ms=180_000)

        self.assertNotIn(
            "cross exchanger",
            model._direct_unit_run_provenance,
        )
        properties = next(
            unit.properties
            for unit in model.list_units()
            if unit.name == "cross exchanger"
        )
        self.assertNotIn("heatTransferDuty_kW", properties)
        self.assertNotIn("duty_kW", properties)

    def test_non_mixer_closure_does_not_authorize_direct_runs(self):
        _, model = self._build_two_sided_heat_exchanger_case(1.0)

        self.assertFalse(
            model._run_acyclic_mixer_energy_closure(
                model.get_process()
            )
        )

    def test_failed_run_does_not_replace_trusted_exchanger_snapshot(self):
        _, model = self._build_two_sided_heat_exchanger_case(1.0)
        model.run(timeout_ms=180_000)
        exchanger = model.get_unit("cross exchanger")
        exchanger.getInStream(0).setTemperature(110.0, "C")

        with self.assertRaisesRegex(
            ProcessExecutionError,
            "no solved results were published",
        ):
            with patch.object(
                model,
                "_run_until_converged",
                return_value=False,
            ), patch.object(
                model,
                "_run_acyclic_mixer_energy_closure",
                side_effect=AssertionError(
                    "failed process run must not start direct closure"
                ),
            ):
                model.run(timeout_ms=180_000)

        properties = next(
            unit.properties
            for unit in model.list_units()
            if unit.name == "cross exchanger"
        )
        self.assertNotIn("heatTransferDuty_kW", properties)
        self.assertNotIn(
            "cross exchanger.heatTransferDuty_kW",
            model._extract_results().kpis,
        )

    def test_clone_recaptures_mixer_heat_exchanger_provenance(self):
        _, model = self._build_mixer_heat_exchanger_case()
        model.run(timeout_ms=180_000)

        cloned_model = model.clone()
        properties = next(
            unit.properties
            for unit in cloned_model.list_units()
            if unit.name == "cross exchanger"
        )
        result = cloned_model._extract_results()

        self.assertGreater(properties["heatTransferDuty_kW"], 0.0)
        self.assertIn(
            "cross exchanger",
            cloned_model._direct_unit_run_provenance,
        )
        self.assertLess(properties["dutyClosure_pct"], 1.0e-6)
        self.assertLess(result.kpis["mass_balance_pct"].value, 1.0e-6)
        self.assertLess(
            result.kpis["component_balance_max_pct"].value,
            1.0e-6,
        )
        self.assertLess(result.kpis["energy_balance_pct"].value, 1.0e-6)

    def test_build_from_spec_dispatches_generic_graph_schema(self):
        builder = ProcessBuilder()
        graph_spec = {
            "name": "Generic graph",
            "units": [],
            "connections": [],
        }
        inlet_specs = [
            {
                "inlet_id": "feed",
                "name": "Feed",
                "fluid_spec": {},
            }
        ]
        execution_order = ["feed"]
        expected_model = object()

        with (
            patch(
                "process_chat.process_builder.monotonic",
                return_value=10.0,
            ),
            patch.object(
                builder,
                "build_acyclic_graph",
                return_value=expected_model,
            ) as graph_builder,
        ):
            model = builder.build_from_spec(
                {
                    "name": "Generic graph",
                    "graph": graph_spec,
                    "inlet_specs": inlet_specs,
                    "execution_order": execution_order,
                }
            )

        self.assertIs(model, expected_model)
        graph_builder.assert_called_once_with(
            graph_spec,
            inlet_specs,
            execution_order,
            timeout_ms=180000,
            _deadline=190.0,
        )

    def test_build_from_spec_preserves_generic_wrapper_name(self):
        inlet_specs = [
            {
                "inlet_id": "feed",
                "name": "Feed",
                "fluid_spec": {},
            }
        ]
        for nested_name in (None, "Stale graph name"):
            with self.subTest(nested_name=nested_name):
                builder = ProcessBuilder()
                graph_spec = {
                    "units": [],
                    "connections": [],
                }
                if nested_name is not None:
                    graph_spec["name"] = nested_name

                with patch.object(
                    builder,
                    "build_acyclic_graph",
                    return_value=object(),
                ) as graph_builder:
                    builder.build_from_spec(
                        {
                            "name": "Canonical case name",
                            "graph": graph_spec,
                            "inlet_specs": inlet_specs,
                            "execution_order": ["feed"],
                        }
                    )

                dispatched_graph = graph_builder.call_args.args[0]
                self.assertEqual(
                    dispatched_graph["name"],
                    "Canonical case name",
                )
                self.assertEqual(
                    graph_spec.get("name"),
                    nested_name,
                )

    def test_build_from_spec_ignores_null_generic_wrapper_name(self):
        builder = ProcessBuilder()
        graph_spec = {
            "name": "Nested graph name",
            "units": [],
            "connections": [],
        }

        with patch.object(
            builder,
            "build_acyclic_graph",
            return_value=object(),
        ) as graph_builder:
            builder.build_from_spec(
                {
                    "name": None,
                    "graph": graph_spec,
                    "inlet_specs": [
                        {
                            "inlet_id": "feed",
                            "name": "Feed",
                            "fluid_spec": {},
                        }
                    ],
                    "execution_order": ["feed"],
                }
            )

        self.assertIs(graph_builder.call_args.args[0], graph_spec)
        self.assertEqual(graph_spec["name"], "Nested graph name")

    def test_build_from_spec_replays_native_two_inlet_graph(self):
        source_builder, source_model = self._build_case(1.0)
        source_result = source_model.run(timeout_ms=180_000)
        replay_builder = ProcessBuilder()

        replay_model = replay_builder.build_from_spec(source_builder.spec)
        replay_result = replay_model.run(timeout_ms=180_000)

        self.assertEqual(replay_builder.spec, source_builder.spec)
        for kpi_name in (
            "material_feed_count",
            "material_feed_flow_kg_hr",
            "material_product_count",
            "material_product_flow_kg_hr",
            "mass_balance_pct",
            "component_balance_max_pct",
            "energy_balance_pct",
        ):
            self.assertAlmostEqual(
                replay_result.kpis[kpi_name].value,
                source_result.kpis[kpi_name].value,
                delta=max(
                    abs(source_result.kpis[kpi_name].value) * 1.0e-6,
                    1.0e-6,
                ),
            )
        validation = {
            constraint.name: constraint.status
            for constraint in replay_result.constraints
        }
        self.assertEqual(validation["mass_balance"], "OK")
        self.assertEqual(validation["component_balance"], "OK")
        self.assertEqual(validation["energy_balance"], "OK")
        self.assertLess(
            replay_result.kpis["mass_balance_pct"].value,
            1.0e-6,
        )
        self.assertLess(
            replay_result.kpis["component_balance_max_pct"].value,
            1.0e-6,
        )
        self.assertLess(
            replay_result.kpis["energy_balance_pct"].value,
            1.0e-6,
        )
        print(
            "native Process Chat graph handoff benchmark:",
            "feed=100000.0 kg/hr",
            "mass="
            f"{replay_result.kpis['mass_balance_pct'].value:.3e}%",
            "components="
            f"{replay_result.kpis['component_balance_max_pct'].value:.3e}%",
            "energy="
            f"{replay_result.kpis['energy_balance_pct'].value:.3e}%",
        )

    def test_graph_python_export_embeds_exact_schema_and_compiles(self):
        builder = ProcessBuilder()
        builder._process_name = 'Two-feed "satellite" mixer'
        builder._spec = {
            "name": 'Two-feed "satellite" mixer',
            "graph": {
                "name": 'Two-feed "satellite" mixer',
                "units": [
                    {
                        "id": "feed-mixer",
                        "name": "Feed mixer",
                        "type": "mixer",
                        "params": {},
                        "ports": {
                            "material_in": ["in_0", "in_1"],
                            "material_out": ["out"],
                        },
                    }
                ],
                "connections": [],
            },
            "inlet_specs": [
                {
                    "inlet_id": "feed-a",
                    "name": "Northern feed",
                    "fluid_spec": {
                        "eos_model": "srk",
                        "components": {"methane": 1.0},
                    },
                },
                {
                    "inlet_id": "feed-b",
                    "name": "Sør feed",
                    "fluid_spec": {
                        "eos_model": "srk",
                        "components": {"methane": 1.0},
                    },
                },
            ],
            "execution_order": ["feed-a", "feed-b", "feed-mixer"],
        }

        script = builder.to_python_script()

        compile(script, "process_flowsheet_model.py", "exec")
        payload_line = next(
            line for line in script.splitlines()
            if line.startswith("case_data = json.loads(")
        )
        serialized_literal = payload_line.removeprefix(
            "case_data = json.loads("
        ).removesuffix(")")
        exported_case = json.loads(ast.literal_eval(serialized_literal))
        self.assertEqual(exported_case, builder._spec)
        self.assertIn(
            "from process_chat.process_builder import ProcessBuilder",
            script,
        )
        self.assertIn(
            "# Run from this repository checkout with neqsim installed.",
            script,
        )
        self.assertNotIn("EvenSol/neqsimweb2", script)
        self.assertIn("builder.build_acyclic_graph(", script)
        self.assertIn("model.run(timeout_ms=180_000)", script)
        self.assertIn(
            '"material_product_flow_kg_hr",',
            script,
        )
        self.assertIn(
            '"Graph replay validation did not pass: "',
            script,
        )
        self.assertIn(
            'print(f"Energy imbalance: {energy_residual:.6e} %")',
            script,
        )
        self.assertIn(
            "two-feed_satellite_mixer.neqsim",
            script,
        )

        builder._process_name = 'Mixer """\nraise RuntimeError("header")'
        adversarial_script = builder.to_python_script()
        compile(adversarial_script, "process_flowsheet_model.py", "exec")
        self.assertTrue(
            adversarial_script.startswith(
                '# NeqSim Process: "Mixer \\"\\"\\"'
                '\\nraise RuntimeError(\\"header\\")"\n'
            )
        )
        self.assertNotIn("\nraise RuntimeError", adversarial_script)

    def test_graph_python_export_replays_native_two_inlet_case(self):
        builder, source_model = self._build_case(1.05)
        source_result = source_model.run(timeout_ms=180_000)
        script = builder.to_python_script()
        payload_line = next(
            line for line in script.splitlines()
            if line.startswith("case_data = json.loads(")
        )
        serialized_literal = payload_line.removeprefix(
            "case_data = json.loads("
        ).removesuffix(")")
        self.assertEqual(
            json.loads(ast.literal_eval(serialized_literal)),
            builder.spec,
        )
        namespace: dict[str, object] = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            previous_directory = os.getcwd()
            try:
                os.chdir(temp_dir)
                output = io.StringIO()
                with redirect_stdout(output):
                    exec(
                        compile(
                            script,
                            "process_flowsheet_model.py",
                            "exec",
                        ),
                        namespace,
                    )
                saved_path = os.path.join(
                    temp_dir,
                    "native_two-inlet_mixer_benchmark.neqsim",
                )
                self.assertTrue(os.path.isfile(saved_path))
                self.assertGreater(os.path.getsize(saved_path), 0)
            finally:
                os.chdir(previous_directory)

        replay_result = namespace["result"]
        self.assertEqual(
            replay_result.kpis["material_feed_count"].value,
            2.0,
        )
        self.assertEqual(
            replay_result.kpis["material_product_count"].value,
            1.0,
        )
        for kpi_name in (
            "material_feed_flow_kg_hr",
            "material_product_flow_kg_hr",
            "mass_balance_pct",
            "component_balance_max_pct",
            "energy_balance_pct",
        ):
            self.assertAlmostEqual(
                replay_result.kpis[kpi_name].value,
                source_result.kpis[kpi_name].value,
                delta=max(
                    abs(source_result.kpis[kpi_name].value) * 1.0e-6,
                    1.0e-6,
                ),
            )
        validation = {
            constraint.name: constraint.status
            for constraint in replay_result.constraints
        }
        self.assertEqual(validation["mass_balance"], "OK")
        self.assertEqual(validation["component_balance"], "OK")
        self.assertEqual(validation["energy_balance"], "OK")
        self.assertLess(
            replay_result.kpis["mass_balance_pct"].value,
            1.0e-6,
        )
        self.assertLess(
            replay_result.kpis["component_balance_max_pct"].value,
            1.0e-6,
        )
        self.assertLess(
            replay_result.kpis["energy_balance_pct"].value,
            1.0e-6,
        )
        replay_output = output.getvalue()
        self.assertIn("Feed boundaries: 2", replay_output)
        self.assertIn("Feed flow: 105000.000000 kg/hr", replay_output)
        self.assertIn("Product boundaries: 1", replay_output)
        self.assertIn("Process simulation complete!", replay_output)
        print(
            "native exported graph replay benchmark:",
            "feed=105000.0 kg/hr",
            "mass="
            f"{replay_result.kpis['mass_balance_pct'].value:.3e}%",
            "components="
            f"{replay_result.kpis['component_balance_max_pct'].value:.3e}%",
            "energy="
            f"{replay_result.kpis['energy_balance_pct'].value:.3e}%",
        )

    def test_graph_python_export_allows_inapplicable_balance_audits(self):
        builder = ProcessBuilder()
        builder._process_name = "Unaudited transport"
        builder._spec = {
            "graph": {
                "name": "Unaudited transport",
                "units": [],
                "connections": [],
            },
            "inlet_specs": [
                {
                    "inlet_id": "feed",
                    "name": "feed",
                    "fluid_spec": {},
                }
            ],
            "execution_order": ["feed"],
        }

        class _Value:
            def __init__(self, value):
                self.value = value

        class _Constraint:
            def __init__(self, name, status):
                self.name = name
                self.status = status

        class _Result:
            raw = {
                "material_balance_applicable": False,
                "component_balance_applicable": False,
                "energy_balance_applicable": False,
            }
            kpis = {
                "material_feed_count": _Value(1.0),
                "material_feed_flow_kg_hr": _Value(12_000.0),
                "material_product_count": _Value(1.0),
                "material_product_flow_kg_hr": _Value(12_000.0),
            }
            constraints = [
                _Constraint("mass_balance", "UNKNOWN"),
                _Constraint("component_balance", "UNKNOWN"),
                _Constraint("energy_balance", "UNKNOWN"),
            ]

        result = _Result()

        class _Model:
            @staticmethod
            def run(timeout_ms):
                if timeout_ms != 180_000:
                    raise AssertionError("Unexpected replay timeout.")
                return result

            @staticmethod
            def get_process():
                return object()

            @staticmethod
            def save_bytes():
                return b"serialized-model"

        output = io.StringIO()
        saved_model = mock_open()
        with (
            patch.object(
                ProcessBuilder,
                "build_acyclic_graph",
                return_value=_Model(),
            ),
            patch("builtins.open", saved_model),
            redirect_stdout(output),
        ):
            namespace: dict[str, object] = {}
            exec(
                compile(
                    builder.to_python_script(),
                    "process_flowsheet_model.py",
                    "exec",
                ),
                namespace,
            )

        saved_model.assert_called_once_with(
            "unaudited_transport.neqsim",
            "wb",
        )
        saved_model().write.assert_called_once_with(b"serialized-model")
        self.assertIs(namespace["result"], result)
        self.assertIn(
            "Mass imbalance: not applicable",
            output.getvalue(),
        )
        self.assertIn(
            "Maximum component imbalance: not applicable",
            output.getvalue(),
        )
        self.assertIn(
            "Energy imbalance: not applicable",
            output.getvalue(),
        )
        self.assertIn(
            "Process simulation complete!",
            output.getvalue(),
        )

    @staticmethod
    def _component_molar_flows(stream):
        fluid = stream.getFluid()
        total_flow = float(stream.getFlowRate("mol/sec"))
        phase = fluid.getPhase(0)
        return {
            str(phase.getComponent(index).getName()): (
                total_flow
                * float(phase.getComponent(index).getz())
            )
            for index in range(int(phase.getNumberOfComponents()))
        }

    @staticmethod
    def _build_case(flow_scale: float):
        inlet_specs = [
            {
                "inlet_id": "dry-gas",
                "name": "dry gas",
                "fluid_spec": {
                    "eos_model": "srk",
                    "mixing_rule": 2,
                    "components": {
                        "methane": 0.95,
                        "ethane": 0.05,
                    },
                    "composition_basis": "mole_fraction",
                    "temperature_C": 20.0,
                    "pressure_bara": 45.0,
                    "total_flow": 60_000.0 * flow_scale,
                    "flow_unit": "kg/hr",
                },
            },
            {
                "inlet_id": "rich-gas",
                "name": "rich gas",
                "fluid_spec": {
                    "eos_model": "srk",
                    "mixing_rule": 2,
                    "components": {
                        "methane": 0.80,
                        "ethane": 0.20,
                    },
                    "composition_basis": "mole_fraction",
                    "temperature_C": 35.0,
                    "pressure_bara": 45.0,
                    "total_flow": 40_000.0 * flow_scale,
                    "flow_unit": "kg/hr",
                },
            },
        ]
        graph_spec = {
            "name": "Native two-inlet mixer benchmark",
            "units": [
                {
                    "id": "feed-mixer",
                    "name": "feed mixer",
                    "type": "mixer",
                    "ports": {
                        "material_in": ["in_0", "in_1"],
                        "material_out": ["out"],
                    },
                    "params": {},
                }
            ],
            "connections": [
                {
                    "id": "dry-gas-to-mixer",
                    "type": "material",
                    "source": {
                        "kind": "inlet",
                        "id": "dry-gas",
                        "port": "out",
                    },
                    "target": {
                        "kind": "unit",
                        "id": "feed-mixer",
                        "port": "in_0",
                    },
                },
                {
                    "id": "rich-gas-to-mixer",
                    "type": "material",
                    "source": {
                        "kind": "inlet",
                        "id": "rich-gas",
                        "port": "out",
                    },
                    "target": {
                        "kind": "unit",
                        "id": "feed-mixer",
                        "port": "in_1",
                    },
                },
            ],
        }
        builder = ProcessBuilder()
        model = builder.build_acyclic_graph(
            graph_spec,
            inlet_specs,
            ["dry-gas", "rich-gas", "feed-mixer"],
        )
        return builder, model

    @staticmethod
    def _build_compression_cooling_case(flow_scale: float):
        inlet_specs = [
            {
                "inlet_id": "feed",
                "name": "feed",
                "fluid_spec": {
                    "eos_model": "srk",
                    "mixing_rule": 2,
                    "components": {
                        "methane": 0.90,
                        "ethane": 0.10,
                    },
                    "composition_basis": "mole_fraction",
                    "temperature_C": 25.0,
                    "pressure_bara": 30.0,
                    "total_flow": 100_000.0 * flow_scale,
                    "flow_unit": "kg/hr",
                },
            }
        ]
        graph_spec = {
            "name": "Native compression energy benchmark",
            "units": [
                {
                    "id": "compressor",
                    "name": "compressor",
                    "type": "compressor",
                    "ports": {
                        "material_in": ["in"],
                        "material_out": ["out"],
                    },
                    "params": {
                        "outlet_pressure_bara": 60.0,
                        "isentropic_efficiency": 0.80,
                    },
                },
                {
                    "id": "cooler",
                    "name": "cooler",
                    "type": "cooler",
                    "ports": {
                        "material_in": ["in"],
                        "material_out": ["out"],
                    },
                    "params": {
                        "outlet_temperature_C": 30.0,
                        "pressure_drop_bar": 0.5,
                    },
                },
            ],
            "connections": [
                {
                    "id": "feed-to-compressor",
                    "type": "material",
                    "source": {
                        "kind": "inlet",
                        "id": "feed",
                        "port": "out",
                    },
                    "target": {
                        "kind": "unit",
                        "id": "compressor",
                        "port": "in",
                    },
                },
                {
                    "id": "compressor-to-cooler",
                    "type": "material",
                    "source": {
                        "kind": "unit",
                        "id": "compressor",
                        "port": "out",
                    },
                    "target": {
                        "kind": "unit",
                        "id": "cooler",
                        "port": "in",
                    },
                },
            ],
        }
        builder = ProcessBuilder()
        model = builder.build_acyclic_graph(
            graph_spec,
            inlet_specs,
            ["feed", "compressor", "cooler"],
        )
        return builder, model

    @staticmethod
    def _build_separator_liquid_pump_case(flow_scale: float):
        inlet_specs = [
            {
                "inlet_id": "well-fluid",
                "name": "well fluid",
                "fluid_spec": {
                    "eos_model": "srk",
                    "mixing_rule": 2,
                    "components": {
                        "methane": 0.50,
                        "n-hexane": 0.50,
                    },
                    "composition_basis": "mole_fraction",
                    "temperature_C": 20.0,
                    "pressure_bara": 20.0,
                    "total_flow": 20_000.0 * flow_scale,
                    "flow_unit": "kg/hr",
                },
            }
        ]
        editor_inlets = [
            {
                "id": inlet["inlet_id"],
                "name": inlet["name"],
                **inlet["fluid_spec"],
            }
            for inlet in inlet_specs
        ]
        units = [
            {
                "id": "inlet-separator",
                "name": "inlet separator",
                "type": "separator",
                "ports": {
                    "material_in": ["in"],
                    "material_out": ["gas", "liquid"],
                },
                "params": {},
            }
        ]
        connections = [
            {
                "id": "well-fluid-to-separator",
                "type": "material",
                "source": {
                    "kind": "inlet",
                    "id": "well-fluid",
                    "port": "out",
                },
                "target": {
                    "kind": "unit",
                    "id": "inlet-separator",
                    "port": "in",
                },
            }
        ]
        history = create_graph_history(units, connections, editor_inlets)
        units, connections, pump_id, _ = extend_material_path(
            editor_inlets,
            units,
            connections,
            {
                "kind": "unit",
                "id": "inlet-separator",
                "port": "liquid",
            },
            "pump",
            "condensate pump",
        )
        units = update_inline_unit_properties(
            units,
            pump_id,
            {
                "outlet_pressure_bara": 40.0,
                "efficiency": 0.75,
            },
        )
        history = record_graph_history(
            history,
            units,
            connections,
            editor_inlets,
        )
        units, connections, heater_id, _ = extend_material_path(
            editor_inlets,
            units,
            connections,
            {
                "kind": "unit",
                "id": pump_id,
                "port": "out",
            },
            "heater",
            "condensate heater",
        )
        units = update_inline_unit_properties(
            units,
            heater_id,
            {
                "outlet_temperature_C": 60.0,
                "pressure_drop_bar": 0.5,
            },
        )
        history = record_graph_history(
            history,
            units,
            connections,
            editor_inlets,
        )
        graph_spec = json.loads(
            json.dumps(
                {
                    "name": "Separator liquid routing benchmark",
                    "units": units,
                    "connections": connections,
                },
                allow_nan=False,
            )
        )
        builder = ProcessBuilder()
        model = builder.build_acyclic_graph(
            graph_spec,
            inlet_specs,
            [
                "well-fluid",
                "inlet-separator",
                "condensate-pump",
                "condensate-heater",
            ],
        )
        return builder, model, history

    @staticmethod
    def _build_palette_mixer_separator_case(
        flow_scale: float,
        auto_size: bool = False,
    ):
        inlet_specs = [
            {
                "inlet_id": "gas-rich-feed",
                "name": "gas rich feed",
                "fluid_spec": {
                    "eos_model": "srk",
                    "mixing_rule": 2,
                    "components": {
                        "methane": 0.80,
                        "n-hexane": 0.20,
                    },
                    "composition_basis": "mole_fraction",
                    "temperature_C": 20.0,
                    "pressure_bara": 20.0,
                    "total_flow": 10_000.0 * flow_scale,
                    "flow_unit": "kg/hr",
                },
            },
            {
                "inlet_id": "liquid-rich-feed",
                "name": "liquid rich feed",
                "fluid_spec": {
                    "eos_model": "srk",
                    "mixing_rule": 2,
                    "components": {
                        "methane": 0.20,
                        "n-hexane": 0.80,
                    },
                    "composition_basis": "mole_fraction",
                    "temperature_C": 20.0,
                    "pressure_bara": 20.0,
                    "total_flow": 10_000.0 * flow_scale,
                    "flow_unit": "kg/hr",
                },
            },
        ]
        editor_inlets = [
            {
                "id": inlet["inlet_id"],
                "name": inlet["name"],
                **inlet["fluid_spec"],
            }
            for inlet in inlet_specs
        ]
        reserved_ids = {
            inlet["inlet_id"]
            for inlet in inlet_specs
        }
        reserved_names = {
            inlet["name"]
            for inlet in inlet_specs
        }
        units, mixer_id = add_catalog_unit(
            [],
            "mixer",
            "feed mixer",
            reserved_ids,
            reserved_names,
        )
        connections = []
        history = create_graph_history(
            units,
            connections,
            editor_inlets,
        )
        for inlet_id, inlet_port in (
            ("gas-rich-feed", "in_0"),
            ("liquid-rich-feed", "in_1"),
        ):
            connections, _ = connect_graph_ports(
                editor_inlets,
                units,
                connections,
                "material",
                {
                    "kind": "inlet",
                    "id": inlet_id,
                    "port": "out",
                },
                {
                    "kind": "unit",
                    "id": mixer_id,
                    "port": inlet_port,
                },
            )
        history = record_graph_history(
            history,
            units,
            connections,
            editor_inlets,
        )
        units, connections, separator_id, _ = extend_material_path(
            editor_inlets,
            units,
            connections,
            {
                "kind": "unit",
                "id": mixer_id,
                "port": "out",
            },
            "separator",
            "product separator",
        )
        units = update_inline_unit_properties(
            units,
            separator_id,
            {
                "auto_size": auto_size,
                "design_gas_load_factor_m_per_s": 0.11,
            },
        )
        history = record_graph_history(
            history,
            units,
            connections,
            editor_inlets,
        )
        graph_spec = json.loads(
            json.dumps(
                {
                    "name": "Palette-built mixer separator benchmark",
                    "units": units,
                    "connections": connections,
                },
                allow_nan=False,
            )
        )
        builder = ProcessBuilder()
        model = builder.build_acyclic_graph(
            graph_spec,
            inlet_specs,
            [
                "gas-rich-feed",
                "liquid-rich-feed",
                mixer_id,
                separator_id,
            ],
        )
        return builder, model, history, graph_spec

    @staticmethod
    def _build_palette_three_feed_mixer_case(flow_scale: float):
        inlet_specs = []
        compositions = (
            {"methane": 0.90, "ethane": 0.10},
            {"methane": 0.70, "ethane": 0.30},
            {"methane": 0.50, "ethane": 0.50},
        )
        base_flows = (10_000.0, 15_000.0, 5_000.0)
        for index, (composition, base_flow) in enumerate(
            zip(compositions, base_flows)
        ):
            inlet_specs.append(
                {
                    "inlet_id": f"feed-{index}",
                    "name": f"feed {index}",
                    "fluid_spec": {
                        "eos_model": "srk",
                        "mixing_rule": 2,
                        "components": composition,
                        "composition_basis": "mole_fraction",
                        "temperature_C": 20.0 + 5.0 * index,
                        "pressure_bara": 30.0,
                        "total_flow": base_flow * flow_scale,
                        "flow_unit": "kg/hr",
                    },
                }
            )
        editor_inlets = [
            {
                "id": inlet["inlet_id"],
                "name": inlet["name"],
                **inlet["fluid_spec"],
            }
            for inlet in inlet_specs
        ]
        units, mixer_id = add_catalog_unit(
            [],
            "mixer",
            "three feed mixer",
            {inlet["inlet_id"] for inlet in inlet_specs},
            {inlet["name"] for inlet in inlet_specs},
        )
        units = resize_mixer_inlet_ports(
            units,
            [],
            mixer_id,
            3,
        )
        connections = []
        for index, inlet in enumerate(editor_inlets):
            connections, _ = connect_graph_ports(
                editor_inlets,
                units,
                connections,
                "material",
                {
                    "kind": "inlet",
                    "id": inlet["id"],
                    "port": "out",
                },
                {
                    "kind": "unit",
                    "id": mixer_id,
                    "port": f"in_{index}",
                },
            )
        graph_spec = {
            "name": "Palette-built three-feed mixer benchmark",
            "units": units,
            "connections": connections,
        }
        builder = ProcessBuilder()
        model = builder.build_acyclic_graph(
            graph_spec,
            inlet_specs,
            ["feed-0", "feed-1", "feed-2", mixer_id],
        )
        return builder, model

    @staticmethod
    def _build_palette_three_feed_separator_case(
        flow_scale: float,
        auto_size: bool = False,
        design_gas_load_factor_m_per_s: float = 0.11,
    ):
        compositions = (
            {"methane": 0.90, "n-hexane": 0.10},
            {"methane": 0.20, "n-hexane": 0.80},
            {"methane": 0.60, "n-hexane": 0.40},
        )
        inlet_specs = [
            {
                "inlet_id": f"separator-feed-{index}",
                "name": f"separator feed {index}",
                "fluid_spec": {
                    "eos_model": "srk",
                    "mixing_rule": 2,
                    "components": composition,
                    "composition_basis": "mole_fraction",
                    "temperature_C": 20.0,
                    "pressure_bara": 20.0,
                    "total_flow": 10_000.0 * flow_scale,
                    "flow_unit": "kg/hr",
                },
            }
            for index, composition in enumerate(compositions)
        ]
        editor_inlets = [
            {
                "id": inlet["inlet_id"],
                "name": inlet["name"],
                **inlet["fluid_spec"],
            }
            for inlet in inlet_specs
        ]
        units, separator_id = add_catalog_unit(
            [],
            "separator",
            "three feed separator",
            {inlet["inlet_id"] for inlet in inlet_specs},
            {inlet["name"] for inlet in inlet_specs},
        )
        units = update_inline_unit_properties(
            units,
            separator_id,
            {
                "auto_size": auto_size,
                "design_gas_load_factor_m_per_s": (
                    design_gas_load_factor_m_per_s
                ),
            },
        )
        units = resize_separator_inlet_ports(
            units,
            [],
            separator_id,
            3,
        )
        connections = []
        for index, inlet in enumerate(editor_inlets):
            target_port = "in" if index == 0 else f"in_{index}"
            connections, _ = connect_graph_ports(
                editor_inlets,
                units,
                connections,
                "material",
                {
                    "kind": "inlet",
                    "id": inlet["id"],
                    "port": "out",
                },
                {
                    "kind": "unit",
                    "id": separator_id,
                    "port": target_port,
                },
            )
        graph_spec = {
            "name": "Palette-built three-feed separator benchmark",
            "units": units,
            "connections": connections,
        }
        builder = ProcessBuilder()
        model = builder.build_acyclic_graph(
            graph_spec,
            inlet_specs,
            [
                "separator-feed-0",
                "separator-feed-1",
                "separator-feed-2",
                separator_id,
            ],
        )
        return builder, model

    @staticmethod
    def _build_reorganized_original_process(flow_scale: float):
        inlet_specs = [
            {
                "inlet_id": "main-feed",
                "name": "main feed",
                "fluid_spec": {
                    "eos_model": "srk",
                    "mixing_rule": 2,
                    "components": {
                        "methane": 0.90,
                        "ethane": 0.10,
                    },
                    "composition_basis": "mole_fraction",
                    "temperature_C": 25.0,
                    "pressure_bara": 30.0,
                    "total_flow": 12_000.0 * flow_scale,
                    "flow_unit": "kg/hr",
                },
            },
            {
                "inlet_id": "satellite-feed",
                "name": "satellite feed",
                "fluid_spec": {
                    "eos_model": "srk",
                    "mixing_rule": 2,
                    "components": {
                        "methane": 0.75,
                        "ethane": 0.25,
                    },
                    "composition_basis": "mole_fraction",
                    "temperature_C": 35.0,
                    "pressure_bara": 30.0,
                    "total_flow": 8_000.0 * flow_scale,
                    "flow_unit": "kg/hr",
                },
            },
        ]
        editor_inlets = [
            {
                "id": inlet["inlet_id"],
                "name": inlet["name"],
                **inlet["fluid_spec"],
            }
            for inlet in inlet_specs
        ]
        units, compressor_id = add_catalog_unit(
            [],
            "compressor",
            "export compressor",
            {"main-feed", "satellite-feed"},
            {"main feed", "satellite feed"},
        )
        units = update_inline_unit_properties(
            units,
            compressor_id,
            {
                "outlet_pressure_bara": 60.0,
                "isentropic_efficiency": 0.78,
            },
        )
        connections, feed_connection_id = connect_graph_ports(
            editor_inlets,
            units,
            [],
            "material",
            {"kind": "inlet", "id": "main-feed", "port": "out"},
            {"kind": "unit", "id": compressor_id, "port": "in"},
        )
        units, connections, cooler_id, _ = extend_material_path(
            editor_inlets,
            units,
            connections,
            {"kind": "unit", "id": compressor_id, "port": "out"},
            "cooler",
            "export cooler",
        )
        units = update_inline_unit_properties(
            units,
            cooler_id,
            {
                "outlet_temperature_C": 35.0,
                "pressure_drop_bar": 0.5,
            },
        )
        original_units = json.loads(json.dumps(units, allow_nan=False))
        original_connections = json.loads(
            json.dumps(connections, allow_nan=False)
        )
        history = create_graph_history(
            original_units,
            original_connections,
            editor_inlets,
        )
        units, connections, mixer_id, _ = insert_mixer_on_connection(
            editor_inlets,
            units,
            connections,
            feed_connection_id,
            "feed mixer",
        )
        connections, _ = connect_graph_ports(
            editor_inlets,
            units,
            connections,
            "material",
            {
                "kind": "inlet",
                "id": "satellite-feed",
                "port": "out",
            },
            {"kind": "unit", "id": mixer_id, "port": "in_1"},
        )
        (
            units,
            connections,
            replacement_compressor_id,
        ) = replace_inline_unit(
            units,
            connections,
            compressor_id,
            "compressor",
            "replacement export compressor",
            {compressor_id},
        )
        units = update_inline_unit_properties(
            units,
            replacement_compressor_id,
            {
                "outlet_pressure_bara": 60.0,
                "isentropic_efficiency": 0.78,
            },
        )
        history = record_graph_history(
            history,
            units,
            connections,
            editor_inlets,
        )
        graph_spec = json.loads(
            json.dumps(
                {
                    "name": "Reorganized original compression process",
                    "units": units,
                    "connections": connections,
                },
                allow_nan=False,
            )
        )
        builder = ProcessBuilder()
        model = builder.build_acyclic_graph(
            graph_spec,
            inlet_specs,
            [
                "main-feed",
                "satellite-feed",
                mixer_id,
                replacement_compressor_id,
                cooler_id,
            ],
        )
        return (
            builder,
            model,
            history,
            graph_spec,
            original_units,
            original_connections,
        )

    @staticmethod
    def _build_palette_splitter_branches_case(
        flow_scale: float,
        split_factor: float = 0.5,
        persisted_split_params: dict | None = None,
    ):
        inlet_specs = [
            {
                "inlet_id": "mixed-feed",
                "name": "mixed feed",
                "fluid_spec": {
                    "eos_model": "srk",
                    "mixing_rule": 2,
                    "components": {
                        "methane": 0.50,
                        "n-hexane": 0.50,
                    },
                    "composition_basis": "mole_fraction",
                    "temperature_C": 20.0,
                    "pressure_bara": 20.0,
                    "total_flow": 20_000.0 * flow_scale,
                    "flow_unit": "kg/hr",
                },
            }
        ]
        editor_inlets = [
            {
                "id": inlet["inlet_id"],
                "name": inlet["name"],
                **inlet["fluid_spec"],
            }
            for inlet in inlet_specs
        ]
        units, splitter_id = add_catalog_unit(
            [],
            "splitter",
            "product split",
            {"mixed-feed"},
            {"mixed feed"},
        )
        connections, _ = connect_graph_ports(
            editor_inlets,
            units,
            [],
            "material",
            {
                "kind": "inlet",
                "id": "mixed-feed",
                "port": "out",
            },
            {
                "kind": "unit",
                "id": splitter_id,
                "port": "in",
            },
        )
        history = create_graph_history(
            units,
            connections,
            editor_inlets,
        )
        if split_factor != 0.5:
            units = update_inline_unit_properties(
                units,
                splitter_id,
                {"split_factor": split_factor},
            )
            history = record_graph_history(
                history,
                units,
                connections,
                editor_inlets,
            )
        if persisted_split_params is not None:
            splitter = next(
                unit for unit in units if unit["id"] == splitter_id
            )
            splitter["params"] = dict(persisted_split_params)
        units, connections, pump_id, _ = extend_material_path(
            editor_inlets,
            units,
            connections,
            {
                "kind": "unit",
                "id": splitter_id,
                "port": "out_0",
            },
            "pump",
            "branch pump",
        )
        units = update_inline_unit_properties(
            units,
            pump_id,
            {
                "outlet_pressure_bara": 40.0,
                "efficiency": 0.75,
            },
        )
        history = record_graph_history(
            history,
            units,
            connections,
            editor_inlets,
        )
        units, connections, heater_id, _ = extend_material_path(
            editor_inlets,
            units,
            connections,
            {
                "kind": "unit",
                "id": splitter_id,
                "port": "out_1",
            },
            "heater",
            "branch heater",
        )
        units = update_inline_unit_properties(
            units,
            heater_id,
            {
                "outlet_temperature_C": 60.0,
                "pressure_drop_bar": 0.5,
            },
        )
        history = record_graph_history(
            history,
            units,
            connections,
            editor_inlets,
        )
        graph_spec = json.loads(
            json.dumps(
                {
                    "name": "Palette-built editable splitter benchmark",
                    "units": units,
                    "connections": connections,
                },
                allow_nan=False,
            )
        )
        builder = ProcessBuilder()
        model = builder.build_acyclic_graph(
            graph_spec,
            inlet_specs,
            [
                "mixed-feed",
                splitter_id,
                pump_id,
                heater_id,
            ],
        )
        return builder, model, history, graph_spec

    @staticmethod
    def _build_palette_three_way_splitter_case(flow_scale: float):
        inlet_specs = [
            {
                "inlet_id": "branch-feed",
                "name": "branch feed",
                "fluid_spec": {
                    "eos_model": "srk",
                    "mixing_rule": 2,
                    "components": {
                        "methane": 0.80,
                        "n-hexane": 0.20,
                    },
                    "composition_basis": "mole_fraction",
                    "temperature_C": 25.0,
                    "pressure_bara": 30.0,
                    "total_flow": 30_000.0 * flow_scale,
                    "flow_unit": "kg/hr",
                },
            }
        ]
        editor_inlets = [
            {
                "id": inlet["inlet_id"],
                "name": inlet["name"],
                **inlet["fluid_spec"],
            }
            for inlet in inlet_specs
        ]
        units, splitter_id = add_catalog_unit(
            [],
            "splitter",
            "three-way product splitter",
            {"branch-feed"},
            {"branch feed"},
        )
        units = resize_splitter_outlet_ports(
            units,
            [],
            splitter_id,
            3,
        )
        units = update_splitter_allocations(
            units,
            splitter_id,
            [2.0, 3.0, 5.0],
        )
        connections, _ = connect_graph_ports(
            editor_inlets,
            units,
            [],
            "material",
            {
                "kind": "inlet",
                "id": "branch-feed",
                "port": "out",
            },
            {
                "kind": "unit",
                "id": splitter_id,
                "port": "in",
            },
        )
        graph_spec = json.loads(
            json.dumps(
                {
                    "name": "Palette-built three-way splitter benchmark",
                    "units": units,
                    "connections": connections,
                },
                allow_nan=False,
            )
        )
        builder = ProcessBuilder()
        model = builder.build_acyclic_graph(
            graph_spec,
            inlet_specs,
            ["branch-feed", splitter_id],
        )
        return builder, model, graph_spec

    def test_splitter_defaults_preserve_explicit_legacy_settings(self):
        from neqsim import jneqsim as _jneqsim  # noqa: F401

        class RecordingSplitter:
            def setSplitFactors(self, values):
                self.values = [float(value) for value in values]

        def unit_spec(params, outputs=None):
            return {
                "ports": {
                    "material_out": outputs or ["out_0", "out_1"],
                },
                "params": params,
            }

        for params, expected in (
            ({}, [0.5, 0.5]),
            ({"split_factor": 0.2}, [0.2, 0.8]),
            ({"split_factors": [3.0, 1.0]}, [0.75, 0.25]),
            ({"split_factors": [1.0e308, 1.0e308]}, [0.5, 0.5]),
        ):
            with self.subTest(params=params):
                splitter = RecordingSplitter()
                actual = ProcessBuilder._configure_graph_splitter(
                    splitter,
                    "product-split",
                    unit_spec(params),
                )
                self.assertEqual(actual, expected)
                self.assertEqual(splitter.values, expected)

        for params, message in (
            ({"split_factors": None}, "requires a split_factors array"),
            ({"split_factor": True}, "split_factor must be numeric"),
            (
                {"split_factor": 0.2},
                "legacy split_factor requires exactly two",
            ),
        ):
            with self.subTest(params=params, message=message):
                outputs = (
                    ["out_0", "out_1", "out_2"]
                    if "exactly two" in message
                    else None
                )
                with self.assertRaisesRegex(ValueError, message):
                    ProcessBuilder._configure_graph_splitter(
                        RecordingSplitter(),
                        "product-split",
                        unit_spec(params, outputs),
                    )

    def test_native_imported_extreme_split_weights_close_equally(self):
        builder, model, _, _ = (
            self._build_palette_splitter_branches_case(
                1.0,
                persisted_split_params={
                    "split_factors": [1.0e308, 1.0e308],
                },
            )
        )
        result = model.run(timeout_ms=180_000)
        product_flows = sorted(
            row["mass_flow_kg_hr"]
            for row in result.raw["material_boundaries"]
            if row["role"] == "product"
        )

        self.assertEqual(len(product_flows), 2)
        self.assertAlmostEqual(product_flows[0], 10_000.0, delta=0.02)
        self.assertAlmostEqual(product_flows[1], 10_000.0, delta=0.02)
        self.assertLess(result.kpis["mass_balance_pct"].value, 1.0e-6)
        self.assertLess(
            result.kpis["component_balance_max_pct"].value,
            1.0e-6,
        )
        self.assertLess(result.kpis["energy_balance_pct"].value, 1.0e-6)
        self.assertIn(
            "Configured graph splitter: product-split "
            "(out_0=0.500000, out_1=0.500000)",
            builder.build_log,
        )

    def test_native_reorganized_original_process_closes_at_nearby_point(self):
        for flow_scale in (1.0, 1.05):
            with self.subTest(flow_scale=flow_scale):
                (
                    builder,
                    model,
                    history,
                    graph_spec,
                    original_units,
                    original_connections,
                ) = self._build_reorganized_original_process(flow_scale)
                result = model.run(timeout_ms=180_000)
                expected_flow = 20_000.0 * flow_scale
                product_flows = [
                    row["mass_flow_kg_hr"]
                    for row in result.raw["material_boundaries"]
                    if row["role"] == "product"
                ]

                self.assertEqual(len(product_flows), 1)
                self.assertAlmostEqual(
                    product_flows[0],
                    expected_flow,
                    delta=max(1.0e-6 * expected_flow, 1.0e-3),
                )
                self.assertLess(
                    result.kpis["mass_balance_pct"].value,
                    1.0e-6,
                )
                self.assertLess(
                    result.kpis["component_balance_max_pct"].value,
                    1.0e-6,
                )
                self.assertLess(
                    result.kpis["energy_balance_pct"].value,
                    1.0e-6,
                )
                self.assertIn(
                    "Acyclic graph built and converged successfully.",
                    builder.build_log,
                )
                self.assertNotIn(
                    "export-compressor",
                    {
                        unit["id"]
                        for unit in graph_spec["units"]
                    },
                )
                self.assertIn(
                    "replacement-export-compressor",
                    {
                        unit["id"]
                        for unit in graph_spec["units"]
                    },
                )
                self.assertIn(
                    "Added graph unit: replacement-export-compressor "
                    "(compressor)",
                    builder.build_log,
                )
                persisted = json.loads(
                    json.dumps(graph_spec, allow_nan=False)
                )
                self.assertEqual(persisted, graph_spec)

                history, restored = undo_graph_history(history)
                self.assertEqual(restored["units"], original_units)
                self.assertEqual(
                    restored["connections"],
                    original_connections,
                )
                history, redone = redo_graph_history(history)
                self.assertEqual(redone["units"], graph_spec["units"])
                self.assertEqual(
                    redone["connections"],
                    graph_spec["connections"],
                )
                print(
                    "native intuitive reorganization benchmark:",
                    f"scale={flow_scale:.2f}",
                    f"feed={expected_flow:.1f} kg/hr",
                    f"product={product_flows[0]:.1f} kg/hr",
                    f"mass={result.kpis['mass_balance_pct'].value:.3e}%",
                    "components="
                    f"{result.kpis['component_balance_max_pct'].value:.3e}%",
                    f"energy={result.kpis['energy_balance_pct'].value:.3e}%",
                )

    def test_native_two_inlet_mass_energy_and_nearby_point(self):
        for flow_scale in (1.0, 1.05):
            with self.subTest(flow_scale=flow_scale):
                builder, model = self._build_case(flow_scale)
                result = model.run(timeout_ms=180_000)
                units = list(model.get_process().getUnitOperations())

                self.assertEqual(
                    [str(unit.getName()) for unit in units],
                    [
                        "dry gas",
                        "rich gas",
                        "dry-gas-to-mixer",
                        "rich-gas-to-mixer",
                        "feed mixer",
                        "feed mixer [out] product",
                    ],
                )
                expected_flow = 100_000.0 * flow_scale
                product_flow = float(units[-1].getFlowRate("kg/hr"))
                self.assertTrue(math.isfinite(product_flow))
                self.assertAlmostEqual(
                    product_flow,
                    expected_flow,
                    delta=max(1.0e-6 * expected_flow, 1.0e-3),
                )

                feed_enthalpy = 0.0
                for stream in units[:2]:
                    stream.getFluid().init(3)
                    feed_enthalpy += float(
                        stream.getFluid().getEnthalpy()
                    )
                units[-1].getFluid().init(3)
                product_enthalpy = float(
                    units[-1].getFluid().getEnthalpy()
                )
                energy_residual = abs(product_enthalpy - feed_enthalpy)
                energy_scale = max(abs(feed_enthalpy), 1.0)
                relative_energy_residual = energy_residual / energy_scale
                self.assertLess(relative_energy_residual, 1.0e-6)

                feed_component_flows = {}
                for stream in units[:2]:
                    for component, molar_flow in (
                        self._component_molar_flows(stream).items()
                    ):
                        feed_component_flows[component] = (
                            feed_component_flows.get(component, 0.0)
                            + molar_flow
                        )
                product_component_flows = (
                    self._component_molar_flows(units[-1])
                )
                self.assertEqual(
                    set(product_component_flows),
                    set(feed_component_flows),
                )
                for component, feed_component_flow in (
                    feed_component_flows.items()
                ):
                    component_scale = max(
                        abs(feed_component_flow),
                        1.0e-12,
                    )
                    self.assertLess(
                        abs(
                            product_component_flows[component]
                            - feed_component_flow
                        )
                        / component_scale,
                        1.0e-6,
                    )

                self.assertEqual(
                    result.kpis["material_feed_count"].value,
                    2.0,
                )
                self.assertEqual(
                    result.kpis["material_product_count"].value,
                    1.0,
                )
                self.assertAlmostEqual(
                    result.kpis["material_feed_flow_kg_hr"].value,
                    expected_flow,
                    delta=max(1.0e-6 * expected_flow, 1.0e-3),
                )
                self.assertAlmostEqual(
                    result.kpis["material_product_flow_kg_hr"].value,
                    expected_flow,
                    delta=max(1.0e-6 * expected_flow, 1.0e-3),
                )
                self.assertLess(
                    result.kpis["mass_balance_pct"].value,
                    1.0e-6,
                )
                boundary_rows = result.raw["material_boundaries"]
                self.assertEqual(
                    [
                        (row["role"], row["stream_name"])
                        for row in boundary_rows
                    ],
                    [
                        ("feed", "dry gas"),
                        ("feed", "rich gas"),
                        ("product", "feed mixer [out] product"),
                    ],
                )
                self.assertAlmostEqual(
                    sum(
                        row["mass_flow_kg_hr"]
                        for row in boundary_rows
                        if row["role"] == "feed"
                    ),
                    expected_flow,
                    delta=max(1.0e-6 * expected_flow, 1.0e-3),
                )
                self.assertAlmostEqual(
                    sum(
                        row["mass_flow_kg_hr"]
                        for row in boundary_rows
                        if row["role"] == "product"
                    ),
                    expected_flow,
                    delta=max(1.0e-6 * expected_flow, 1.0e-3),
                )
                for row in boundary_rows:
                    self.assertTrue(
                        math.isfinite(row["mass_flow_kg_hr"])
                    )
                    self.assertTrue(math.isfinite(row["temperature_C"]))
                    self.assertTrue(math.isfinite(row["pressure_bara"]))
                    self.assertTrue(
                        math.isfinite(row["molar_flow_mol_sec"])
                    )
                    self.assertTrue(
                        math.isfinite(row["enthalpy_flow_kW"])
                    )
                    self.assertEqual(
                        set(row["component_molar_flows_mol_sec"]),
                        {"methane", "ethane"},
                    )
                    self.assertAlmostEqual(
                        sum(
                            row[
                                "component_molar_flows_mol_sec"
                            ].values()
                        ),
                        row["molar_flow_mol_sec"],
                        delta=max(
                            1.0e-9 * row["molar_flow_mol_sec"],
                            1.0e-9,
                        ),
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
                self.assertEqual(component_constraint.status, "OK")
                energy_constraint = next(
                    constraint
                    for constraint in result.constraints
                    if constraint.name == "energy_balance"
                )
                self.assertEqual(energy_constraint.status, "OK")
                self.assertIs(
                    result.raw["energy_balance_applicable"],
                    True,
                )
                self.assertEqual(result.raw["energy_transfers"], [])
                energy_summary = aggregate_energy_balance(result)
                self.assertIs(energy_summary["applicable"], True)
                self.assertLess(
                    energy_summary["imbalance_pct"],
                    1.0e-6,
                )
                self.assertLess(
                    result.kpis["energy_balance_pct"].value,
                    1.0e-6,
                )
                self.assertEqual(
                    result.kpis["component_balance_count"].value,
                    2.0,
                )
                self.assertLess(
                    result.kpis["component_balance_max_pct"].value,
                    1.0e-6,
                )
                self.assertEqual(
                    [
                        row["component"]
                        for row in result.raw["component_balances"]
                    ],
                    ["ethane", "methane"],
                )
                self.assertFalse(
                    [
                        constraint
                        for constraint in result.constraints
                        if constraint.status == "VIOLATION"
                    ]
                )
                self.assertIn(
                    "Acyclic graph built and converged successfully.",
                    builder.build_log,
                )
                print(
                    "native mixer benchmark:",
                    f"scale={flow_scale:.2f}",
                    f"feed={expected_flow:.1f} kg/hr",
                    f"mass={result.kpis['mass_balance_pct'].value:.3e}%",
                    f"energy={relative_energy_residual:.3e}",
                    "components=closed",
                )

                clone = model.clone()
                clone_result = clone.run(timeout_ms=180_000)
                clone_units = list(
                    clone.get_process().getUnitOperations()
                )
                clone_feed_enthalpy = 0.0
                for stream in clone_units[:2]:
                    stream.getFluid().init(3)
                    clone_feed_enthalpy += float(
                        stream.getFluid().getEnthalpy()
                    )
                clone_units[-1].getFluid().init(3)
                clone_product_enthalpy = float(
                    clone_units[-1].getFluid().getEnthalpy()
                )
                clone_energy_scale = max(
                    abs(clone_feed_enthalpy),
                    1.0,
                )
                self.assertLess(
                    abs(
                        clone_product_enthalpy
                        - clone_feed_enthalpy
                    )
                    / clone_energy_scale,
                    1.0e-6,
                )
                self.assertLess(
                    clone_result.kpis["mass_balance_pct"].value,
                    1.0e-6,
                )
                self.assertLess(
                    clone_result.kpis["energy_balance_pct"].value,
                    1.0e-6,
                )

    def test_native_replaced_valve_conserves_at_nearby_points(self):
        for flow_scale in (1.0, 1.05):
            with self.subTest(flow_scale=flow_scale):
                units, unit_id = add_catalog_unit(
                    [],
                    "cooler",
                    "export conditioning",
                )
                units = replace_inline_unit_type(
                    units,
                    unit_id,
                    "valve",
                )
                graph_spec = {
                    "name": "Native replaced-equipment benchmark",
                    "units": units,
                    "connections": [
                        {
                            "id": "feed-to-export-conditioning",
                            "type": "material",
                            "source": {
                                "kind": "inlet",
                                "id": "feed",
                                "port": "out",
                            },
                            "target": {
                                "kind": "unit",
                                "id": unit_id,
                                "port": "in",
                            },
                        }
                    ],
                }
                inlet_specs = [
                    {
                        "inlet_id": "feed",
                        "name": "feed",
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
                            "total_flow": 20_000.0 * flow_scale,
                            "flow_unit": "kg/hr",
                        },
                    }
                ]
                builder = ProcessBuilder()
                model = builder.build_acyclic_graph(
                    graph_spec,
                    inlet_specs,
                    ["feed", unit_id],
                )
                result = model.run(timeout_ms=180_000)
                process_units = list(
                    model.get_process().getUnitOperations()
                )
                valve = next(
                    unit
                    for unit in process_units
                    if str(unit.getName()) == "export conditioning"
                )

                self.assertEqual(graph_spec["units"][0]["id"], unit_id)
                self.assertEqual(graph_spec["units"][0]["type"], "valve")
                self.assertEqual(
                    graph_spec["units"][0]["params"],
                    {
                        "outlet_pressure_bara": 40.0,
                        "percent_valve_opening": 100.0,
                        "use_design_basis": False,
                        "design_cv_capacity_us": 100.0,
                    },
                )
                self.assertAlmostEqual(
                    float(valve.getOutletStream().getPressure("bara")),
                    40.0,
                    delta=0.05,
                )
                self.assertAlmostEqual(
                    float(valve.getPercentValveOpening()),
                    100.0,
                    delta=1.0e-12,
                )
                opening_kpi = result.kpis[
                    "export conditioning.percentValveOpening"
                ]
                self.assertAlmostEqual(
                    opening_kpi.value,
                    100.0,
                    delta=1.0e-12,
                )
                self.assertEqual(opening_kpi.unit, "%")
                cv_kpi = result.kpis["export conditioning.Cv"]
                self.assertGreater(cv_kpi.value, 0.0)
                self.assertEqual(cv_kpi.unit, "US Cv")
                expected_flow = 20_000.0 * flow_scale
                self.assertAlmostEqual(
                    result.kpis["material_product_flow_kg_hr"].value,
                    expected_flow,
                    delta=max(1.0e-6 * expected_flow, 1.0e-3),
                )
                self.assertLess(
                    result.kpis["mass_balance_pct"].value,
                    1.0e-6,
                )
                self.assertLess(
                    result.kpis["component_balance_max_pct"].value,
                    1.0e-6,
                )
                self.assertLess(
                    result.kpis["energy_balance_pct"].value,
                    1.0e-6,
                )
                self.assertFalse(
                    [
                        constraint
                        for constraint in result.constraints
                        if constraint.status == "VIOLATION"
                    ]
                )
                self.assertIn(
                    "Acyclic graph built and converged successfully.",
                    builder.build_log,
                )
                self.assertEqual(
                    json.loads(json.dumps(graph_spec, allow_nan=False)),
                    graph_spec,
                )
                print(
                    "native replaced-valve benchmark:",
                    f"scale={flow_scale:.2f}",
                    f"feed={expected_flow:.1f} kg/hr",
                    f"mass={result.kpis['mass_balance_pct'].value:.3e}%",
                    "components="
                    f"{result.kpis['component_balance_max_pct'].value:.3e}%",
                    f"energy={result.kpis['energy_balance_pct'].value:.3e}%",
                )

    def test_native_valve_opening_sizes_cv_at_nearby_points(self):
        cv_by_scale = {}
        for flow_scale in (1.0, 1.05):
            cv_by_opening = {}
            temperature_by_opening = {}
            for opening_pct in (100.0, 60.0):
                with self.subTest(
                    flow_scale=flow_scale,
                    opening_pct=opening_pct,
                ):
                    units, valve_id = add_catalog_unit(
                        [],
                        "valve",
                        "metering valve",
                    )
                    units[0]["params"].update(
                        {
                            "outlet_pressure_bara": 30.0,
                            "percent_valve_opening": opening_pct,
                        }
                    )
                    graph_spec = {
                        "name": "Native valve opening benchmark",
                        "units": units,
                        "connections": [
                            {
                                "id": "feed-to-metering-valve",
                                "type": "material",
                                "source": {
                                    "kind": "inlet",
                                    "id": "feed",
                                    "port": "out",
                                },
                                "target": {
                                    "kind": "unit",
                                    "id": valve_id,
                                    "port": "in",
                                },
                            }
                        ],
                    }
                    expected_flow = 10_000.0 * flow_scale
                    inlet_specs = [
                        {
                            "inlet_id": "feed",
                            "name": "feed",
                            "fluid_spec": {
                                "eos_model": "srk",
                                "mixing_rule": 2,
                                "components": {
                                    "methane": 0.90,
                                    "ethane": 0.10,
                                },
                                "composition_basis": "mole_fraction",
                                "temperature_C": 30.0,
                                "pressure_bara": 80.0,
                                "total_flow": expected_flow,
                                "flow_unit": "kg/hr",
                            },
                        }
                    ]
                    builder = ProcessBuilder()
                    model = builder.build_acyclic_graph(
                        graph_spec,
                        inlet_specs,
                        ["feed", valve_id],
                    )
                    result = model.run(timeout_ms=180_000)
                    valve = next(
                        unit
                        for unit in model.get_process().getUnitOperations()
                        if str(unit.getName()) == "metering valve"
                    )

                    opening_kpi = result.kpis[
                        "metering valve.percentValveOpening"
                    ]
                    cv_kpi = result.kpis["metering valve.Cv"]
                    self.assertAlmostEqual(
                        float(valve.getPercentValveOpening()),
                        opening_pct,
                        delta=1.0e-12,
                    )
                    self.assertAlmostEqual(
                        opening_kpi.value,
                        opening_pct,
                        delta=1.0e-12,
                    )
                    self.assertEqual(opening_kpi.unit, "%")
                    self.assertAlmostEqual(
                        cv_kpi.value,
                        float(valve.getCv()),
                        delta=1.0e-12,
                    )
                    self.assertEqual(cv_kpi.unit, "US Cv")
                    self.assertAlmostEqual(
                        float(valve.getOutletStream().getPressure("bara")),
                        30.0,
                        delta=0.05,
                    )
                    self.assertAlmostEqual(
                        result.kpis["material_product_flow_kg_hr"].value,
                        expected_flow,
                        delta=max(1.0e-6 * expected_flow, 1.0e-3),
                    )
                    self.assertLess(
                        result.kpis["mass_balance_pct"].value,
                        1.0e-6,
                    )
                    self.assertLess(
                        result.kpis["component_balance_max_pct"].value,
                        1.0e-6,
                    )
                    self.assertLess(
                        result.kpis["energy_balance_pct"].value,
                        1.0e-6,
                    )
                    self.assertFalse(
                        [
                            constraint
                            for constraint in result.constraints
                            if constraint.status == "VIOLATION"
                        ]
                    )
                    self.assertIn(
                        "Acyclic graph built and converged successfully.",
                        builder.build_log,
                    )
                    self.assertEqual(
                        json.loads(json.dumps(graph_spec, allow_nan=False)),
                        graph_spec,
                    )
                    cv_by_opening[opening_pct] = cv_kpi.value
                    temperature_by_opening[opening_pct] = float(
                        valve.getOutletStream().getTemperature("C")
                    )
                    print(
                        "native valve-opening benchmark:",
                        f"scale={flow_scale:.2f}",
                        f"opening={opening_pct:.1f}%",
                        f"Cv={cv_kpi.value:.6f} US Cv",
                        f"mass={result.kpis['mass_balance_pct'].value:.3e}%",
                        "components="
                        f"{result.kpis['component_balance_max_pct'].value:.3e}%",
                        f"energy={result.kpis['energy_balance_pct'].value:.3e}%",
                    )

            self.assertGreater(cv_by_opening[60.0], cv_by_opening[100.0])
            self.assertGreater(
                cv_by_opening[60.0] / cv_by_opening[100.0],
                1.5,
            )
            self.assertAlmostEqual(
                temperature_by_opening[60.0],
                temperature_by_opening[100.0],
                delta=0.05,
            )
            cv_by_scale[flow_scale] = cv_by_opening

        for opening_pct in (100.0, 60.0):
            self.assertGreater(
                cv_by_scale[1.05][opening_pct],
                cv_by_scale[1.0][opening_pct],
            )

    def test_native_separator_liquid_routes_to_pump_and_closes(self):
        for flow_scale in (1.0, 1.05):
            with self.subTest(flow_scale=flow_scale):
                builder, model, history = (
                    self._build_separator_liquid_pump_case(
                        flow_scale
                    )
                )
                result = model.run(timeout_ms=180_000)
                units = list(model.get_process().getUnitOperations())
                names = [str(unit.getName()) for unit in units]

                self.assertIn("condensate pump", names)
                self.assertIn("condensate heater", names)
                self.assertIn("inlet separator [gas] product", names)
                self.assertIn("condensate heater [out] product", names)
                pump = next(
                    unit
                    for unit in units
                    if str(unit.getName()) == "condensate pump"
                )
                self.assertGreater(
                    float(pump.getInletStream().getFlowRate("kg/hr")),
                    1.0,
                )
                self.assertAlmostEqual(
                    float(pump.getOutletStream().getPressure("bara")),
                    40.0,
                    delta=0.05,
                )
                heater = next(
                    unit
                    for unit in units
                    if str(unit.getName()) == "condensate heater"
                )
                self.assertAlmostEqual(
                    float(heater.getOutletStream().getTemperature("C")),
                    60.0,
                    delta=0.05,
                )
                self.assertAlmostEqual(
                    float(heater.getOutletStream().getPressure("bara")),
                    39.5,
                    delta=0.05,
                )

                expected_flow = 20_000.0 * flow_scale
                self.assertEqual(
                    result.kpis["material_feed_count"].value,
                    1.0,
                )
                self.assertEqual(
                    result.kpis["material_product_count"].value,
                    2.0,
                )
                self.assertAlmostEqual(
                    result.kpis["material_feed_flow_kg_hr"].value,
                    expected_flow,
                    delta=max(1.0e-6 * expected_flow, 1.0e-3),
                )
                self.assertAlmostEqual(
                    result.kpis["material_product_flow_kg_hr"].value,
                    expected_flow,
                    delta=max(1.0e-6 * expected_flow, 1.0e-3),
                )
                self.assertLess(
                    result.kpis["mass_balance_pct"].value,
                    1.0e-6,
                )
                component_constraint = next(
                    constraint
                    for constraint in result.constraints
                    if constraint.name == "component_balance"
                )
                self.assertEqual(component_constraint.status, "OK")
                energy_constraint = next(
                    constraint
                    for constraint in result.constraints
                    if constraint.name == "energy_balance"
                )
                self.assertEqual(energy_constraint.status, "OK")
                self.assertLess(
                    result.kpis["energy_balance_pct"].value,
                    1.0e-6,
                )
                self.assertIn(
                    "Added graph unit: condensate-pump (pump)",
                    builder.build_log,
                )
                self.assertIn(
                    "Added graph unit: condensate-heater (heater)",
                    builder.build_log,
                )
                history, pump_draft = undo_graph_history(history)
                self.assertEqual(
                    [unit["id"] for unit in pump_draft["units"]],
                    ["inlet-separator", "condensate-pump"],
                )
                history, final_draft = redo_graph_history(history)
                self.assertEqual(
                    [unit["id"] for unit in final_draft["units"]],
                    [
                        "inlet-separator",
                        "condensate-pump",
                        "condensate-heater",
                    ],
                )
                print(
                    "native separator-liquid chain benchmark:",
                    f"scale={flow_scale:.2f}",
                    f"feed={expected_flow:.1f} kg/hr",
                    f"mass={result.kpis['mass_balance_pct'].value:.3e}%",
                    f"energy={result.kpis['energy_balance_pct'].value:.3e}%",
                    f"components={component_constraint.status.lower()}",
                )

    def test_palette_built_two_feed_separator_round_trip_and_closure(self):
        for flow_scale in (1.0, 1.05):
            with self.subTest(flow_scale=flow_scale):
                builder, model, history, graph_spec = (
                    self._build_palette_mixer_separator_case(flow_scale)
                )
                result = model.run(timeout_ms=180_000)
                units = list(model.get_process().getUnitOperations())
                names = [str(unit.getName()) for unit in units]

                self.assertEqual(
                    names,
                    [
                        "gas rich feed",
                        "liquid rich feed",
                        (
                            "material-gas-rich-feed-out-to-"
                            "feed-mixer-in-0"
                        ),
                        (
                            "material-liquid-rich-feed-out-to-"
                            "feed-mixer-in-1"
                        ),
                        "feed mixer",
                        (
                            "material-feed-mixer-out-to-"
                            "product-separator-in"
                        ),
                        "product separator",
                        "product separator [gas] product",
                        "product separator [liquid] product",
                    ],
                )
                expected_flow = 20_000.0 * flow_scale
                boundary_rows = result.raw["material_boundaries"]
                product_rows = [
                    row
                    for row in boundary_rows
                    if row["role"] == "product"
                ]
                self.assertEqual(len(product_rows), 2)
                self.assertTrue(
                    all(row["mass_flow_kg_hr"] > 1.0 for row in product_rows)
                )
                self.assertEqual(
                    result.kpis["material_feed_count"].value,
                    2.0,
                )
                self.assertEqual(
                    result.kpis["material_product_count"].value,
                    2.0,
                )
                self.assertAlmostEqual(
                    result.kpis["material_feed_flow_kg_hr"].value,
                    expected_flow,
                    delta=max(1.0e-6 * expected_flow, 1.0e-3),
                )
                self.assertAlmostEqual(
                    result.kpis["material_product_flow_kg_hr"].value,
                    expected_flow,
                    delta=max(1.0e-6 * expected_flow, 1.0e-3),
                )
                self.assertLess(
                    result.kpis["mass_balance_pct"].value,
                    1.0e-6,
                )
                self.assertLess(
                    result.kpis["component_balance_max_pct"].value,
                    1.0e-6,
                )
                self.assertLess(
                    result.kpis["energy_balance_pct"].value,
                    1.0e-6,
                )
                for constraint_name in (
                    "mass_balance",
                    "component_balance",
                    "energy_balance",
                ):
                    constraint = next(
                        constraint
                        for constraint in result.constraints
                        if constraint.name == constraint_name
                    )
                    self.assertEqual(constraint.status, "OK")
                self.assertIn(
                    "Added graph mixer: feed-mixer (2 material inlets)",
                    builder.build_log,
                )
                self.assertIn(
                    "Added graph unit: product-separator (separator)",
                    builder.build_log,
                )
                self.assertNotIn(
                    "Closed acyclic mixer energy balance before mechanical "
                    "design.",
                    builder.build_log,
                )

                persisted = json.loads(
                    json.dumps(graph_spec, allow_nan=False)
                )
                self.assertEqual(persisted, graph_spec)
                history, connected_mixer_draft = undo_graph_history(history)
                self.assertEqual(
                    [unit["id"] for unit in connected_mixer_draft["units"]],
                    ["feed-mixer"],
                )
                self.assertEqual(
                    len(connected_mixer_draft["connections"]),
                    2,
                )
                history, final_draft = redo_graph_history(history)
                self.assertEqual(
                    final_draft["units"],
                    graph_spec["units"],
                )
                self.assertEqual(
                    final_draft["connections"],
                    graph_spec["connections"],
                )
                print(
                    "native palette mixer-separator benchmark:",
                    f"scale={flow_scale:.2f}",
                    f"feed={expected_flow:.1f} kg/hr",
                    f"mass={result.kpis['mass_balance_pct'].value:.3e}%",
                    "components="
                    f"{result.kpis['component_balance_max_pct'].value:.3e}%",
                    f"energy={result.kpis['energy_balance_pct'].value:.3e}%",
                )

    def test_palette_built_three_feed_mixer_conserves_at_nearby_points(self):
        for flow_scale in (1.0, 1.05):
            with self.subTest(flow_scale=flow_scale):
                builder, model = self._build_palette_three_feed_mixer_case(
                    flow_scale
                )
                result = model.run(timeout_ms=180_000)
                expected_flow = 30_000.0 * flow_scale

                self.assertEqual(
                    result.kpis["material_feed_count"].value,
                    3.0,
                )
                self.assertEqual(
                    result.kpis["material_product_count"].value,
                    1.0,
                )
                self.assertAlmostEqual(
                    result.kpis["material_feed_flow_kg_hr"].value,
                    expected_flow,
                    delta=max(1.0e-6 * expected_flow, 1.0e-3),
                )
                self.assertAlmostEqual(
                    result.kpis["material_product_flow_kg_hr"].value,
                    expected_flow,
                    delta=max(1.0e-6 * expected_flow, 1.0e-3),
                )
                self.assertLess(
                    result.kpis["mass_balance_pct"].value,
                    1.0e-6,
                )
                self.assertLess(
                    result.kpis["component_balance_max_pct"].value,
                    1.0e-6,
                )
                self.assertLess(
                    result.kpis["energy_balance_pct"].value,
                    1.0e-6,
                )
                mixer_prefix = "three feed mixer"
                self.assertEqual(
                    result.kpis[f"{mixer_prefix}.inletCount"].value,
                    3.0,
                )
                self.assertAlmostEqual(
                    result.kpis[
                        f"{mixer_prefix}.inletFlowTotal_kg_hr"
                    ].value,
                    expected_flow,
                    delta=max(1.0e-6 * expected_flow, 1.0e-3),
                )
                self.assertAlmostEqual(
                    result.kpis[
                        f"{mixer_prefix}.outletFlow_kg_hr"
                    ].value,
                    expected_flow,
                    delta=max(1.0e-6 * expected_flow, 1.0e-3),
                )
                self.assertLess(
                    result.kpis[
                        f"{mixer_prefix}.flowClosure_pct"
                    ].value,
                    1.0e-8,
                )
                for index, expected_fraction in enumerate(
                    (1.0 / 3.0, 0.5, 1.0 / 6.0)
                ):
                    self.assertAlmostEqual(
                        result.kpis[
                            f"{mixer_prefix}.inlet{index}Fraction"
                        ].value,
                        expected_fraction,
                        delta=1.0e-8,
                    )
                mixer_properties = next(
                    unit.properties
                    for unit in model.list_units()
                    if unit.name == mixer_prefix
                )
                self.assertEqual(mixer_properties["inletCount"], 3.0)
                self.assertLess(
                    mixer_properties["flowClosure_pct"],
                    1.0e-8,
                )
                self.assertIn(
                    "Added graph mixer: three-feed-mixer "
                    "(3 material inlets)",
                    builder.build_log,
                )
                print(
                    "native three-feed mixer benchmark:",
                    f"scale={flow_scale:.2f}",
                    f"feed={expected_flow:.1f} kg/hr",
                    f"mass={result.kpis['mass_balance_pct'].value:.3e}%",
                    "components="
                    f"{result.kpis['component_balance_max_pct'].value:.3e}%",
                    f"energy={result.kpis['energy_balance_pct'].value:.3e}%",
                )

    def test_multi_inlet_units_require_every_declared_port(self):
        inlet_specs = [
            {
                "inlet_id": f"feed-{index}",
                "name": f"feed {index}",
                "fluid_spec": {
                    "eos_model": "srk",
                    "mixing_rule": 2,
                    "components": {"methane": 1.0},
                    "composition_basis": "mole_fraction",
                    "temperature_C": 20.0,
                    "pressure_bara": 20.0,
                    "total_flow": 10_000.0,
                    "flow_unit": "kg/hr",
                },
            }
            for index in range(2)
        ]
        editor_inlets = [
            {
                "id": inlet["inlet_id"],
                "name": inlet["name"],
                **inlet["fluid_spec"],
            }
            for inlet in inlet_specs
        ]
        for unit_type, declared_count, connected_count in (
            ("mixer", 3, 2),
            ("separator", 2, 1),
        ):
            with self.subTest(unit_type=unit_type):
                units, unit_id = add_catalog_unit(
                    [],
                    unit_type,
                    f"incomplete {unit_type}",
                )
                resize = (
                    resize_mixer_inlet_ports
                    if unit_type == "mixer"
                    else resize_separator_inlet_ports
                )
                units = resize(
                    units,
                    [],
                    unit_id,
                    declared_count,
                )
                connections = []
                for index in range(connected_count):
                    target_port = (
                        f"in_{index}"
                        if unit_type == "mixer"
                        else ("in" if index == 0 else f"in_{index}")
                    )
                    connections, _ = connect_graph_ports(
                        editor_inlets,
                        units,
                        connections,
                        "material",
                        {
                            "kind": "inlet",
                            "id": f"feed-{index}",
                            "port": "out",
                        },
                        {
                            "kind": "unit",
                            "id": unit_id,
                            "port": target_port,
                        },
                    )
                with self.assertRaisesRegex(
                    ValueError,
                    "connections must match declared ports",
                ):
                    ProcessBuilder().build_acyclic_graph(
                        {
                            "name": f"Incomplete {unit_type}",
                            "units": units,
                            "connections": connections,
                        },
                        inlet_specs,
                        ["feed-0", "feed-1", unit_id],
                    )

    def test_legacy_mixer_port_matching_normalizes_declared_names(self):
        inlet_specs = [
            {
                "inlet_id": f"feed-{index}",
                "name": f"feed {index}",
                "fluid_spec": {
                    "eos_model": "srk",
                    "mixing_rule": 2,
                    "components": {"methane": 1.0},
                    "composition_basis": "mole_fraction",
                    "temperature_C": 20.0,
                    "pressure_bara": 20.0,
                    "total_flow": 10_000.0,
                    "flow_unit": "kg/hr",
                },
            }
            for index in range(2)
        ]
        graph_spec = {
            "name": "Legacy padded mixer ports",
            "units": [
                {
                    "id": "legacy-mixer",
                    "name": "legacy mixer",
                    "type": "mixer",
                    "ports": {
                        "material_in": ["feed_a ", "feed_b"],
                        "material_out": ["out"],
                    },
                    "params": {},
                }
            ],
            "connections": [
                {
                    "id": f"feed-{index}-to-legacy-mixer",
                    "type": "material",
                    "source": {
                        "kind": "inlet",
                        "id": f"feed-{index}",
                        "port": "out",
                    },
                    "target": {
                        "kind": "unit",
                        "id": "legacy-mixer",
                        "port": target_port,
                    },
                }
                for index, target_port in enumerate(
                    ("feed_a ", "feed_b")
                )
            ],
        }

        builder = ProcessBuilder()
        model = builder.build_acyclic_graph(
            graph_spec,
            inlet_specs,
            ["feed-0", "feed-1", "legacy-mixer"],
        )

        self.assertIsNotNone(model)
        self.assertIn(
            "Added graph mixer: legacy-mixer (2 material inlets)",
            builder.build_log,
        )

    def test_palette_built_three_feed_separator_conserves_nearby_points(self):
        for flow_scale in (1.0, 1.05):
            with self.subTest(flow_scale=flow_scale):
                builder, model = self._build_palette_three_feed_separator_case(
                    flow_scale
                )
                result = model.run(timeout_ms=180_000)
                expected_flow = 30_000.0 * flow_scale

                self.assertEqual(
                    result.kpis["material_feed_count"].value,
                    3.0,
                )
                self.assertEqual(
                    result.kpis["material_product_count"].value,
                    2.0,
                )
                self.assertAlmostEqual(
                    result.kpis["material_feed_flow_kg_hr"].value,
                    expected_flow,
                    delta=max(1.0e-6 * expected_flow, 1.0e-3),
                )
                self.assertAlmostEqual(
                    result.kpis["material_product_flow_kg_hr"].value,
                    expected_flow,
                    delta=max(1.0e-6 * expected_flow, 1.0e-3),
                )
                self.assertLess(
                    result.kpis["mass_balance_pct"].value,
                    1.0e-6,
                )
                self.assertLess(
                    result.kpis["component_balance_max_pct"].value,
                    1.0e-6,
                )
                self.assertLess(
                    result.kpis["energy_balance_pct"].value,
                    1.0e-6,
                )
                self.assertIn(
                    "Added graph separator: three-feed-separator "
                    "(3 material inlets)",
                    builder.build_log,
                )
                print(
                    "native three-feed separator benchmark:",
                    f"scale={flow_scale:.2f}",
                    f"feed={expected_flow:.1f} kg/hr",
                    f"mass={result.kpis['mass_balance_pct'].value:.3e}%",
                    "components="
                    f"{result.kpis['component_balance_max_pct'].value:.3e}%",
                    f"energy={result.kpis['energy_balance_pct'].value:.3e}%",
                )

    def test_native_separator_design_closes_at_nearby_points(self):
        design_points = []
        for flow_scale in (1.0, 1.05):
            with self.subTest(flow_scale=flow_scale):
                builder, model = self._build_palette_three_feed_separator_case(
                    flow_scale,
                    auto_size=True,
                    design_gas_load_factor_m_per_s=0.11,
                )
                result = model.run(timeout_ms=180_000)
                separator_info = next(
                    unit
                    for unit in model.list_units()
                    if unit.name == "three feed separator"
                )
                properties = separator_info.properties

                self.assertIs(properties["designAutoSized"], True)
                self.assertAlmostEqual(
                    properties["designGasLoadFactor_m_per_s"],
                    0.107,
                    delta=1.0e-12,
                )
                self.assertGreater(properties["designInternalDiameter_m"], 0.0)
                self.assertGreater(properties["designSeparatorLength_m"], 0.0)
                self.assertEqual(properties["designRetentionTime_s"], 120.0)
                self.assertGreater(properties["designVolume_m3"], 0.0)
                self.assertLess(result.kpis["mass_balance_pct"].value, 1.0e-6)
                self.assertLess(
                    result.kpis["component_balance_max_pct"].value,
                    1.0e-6,
                )
                self.assertLess(result.kpis["energy_balance_pct"].value, 1.0e-6)
                self.assertEqual(
                    result.kpis[
                        "three feed separator.designAutoSized"
                    ].unit,
                    "boolean",
                )
                self.assertIn(
                    "Running closed design rerun for: three-feed-separator",
                    builder.build_log,
                )
                self.assertIn('"auto_size": true', builder.to_python_script())
                design_points.append(properties)
                print(
                    "native separator design benchmark:",
                    f"scale={flow_scale:.2f}",
                    "diameter="
                    f"{properties['designInternalDiameter_m']:.6f} m",
                    "length="
                    f"{properties['designSeparatorLength_m']:.6f} m",
                    "retention="
                    f"{properties['designRetentionTime_s']:.1f} s",
                    f"mass={result.kpis['mass_balance_pct'].value:.3e}%",
                    "components="
                    f"{result.kpis['component_balance_max_pct'].value:.3e}%",
                    f"energy={result.kpis['energy_balance_pct'].value:.3e}%",
                )

        self.assertGreaterEqual(
            design_points[1]["designInternalDiameter_m"],
            design_points[0]["designInternalDiameter_m"],
        )
        self.assertGreaterEqual(
            design_points[1]["designVolume_m3"],
            design_points[0]["designVolume_m3"],
        )

    def test_native_mixer_separator_design_uses_closed_feed_state(self):
        builder, model, _, _ = self._build_palette_mixer_separator_case(
            1.0,
            auto_size=True,
        )
        result = model.run(timeout_ms=180_000)
        separator = next(
            unit
            for unit in model.get_process().getUnitOperations()
            if str(unit.getName()) == "product separator"
        )
        designed_diameter = float(separator.getInternalDiameter())

        separator.autoSize()
        closed_basis_diameter = float(separator.getInternalDiameter())

        self.assertAlmostEqual(
            designed_diameter,
            closed_basis_diameter,
            delta=1.0e-9,
        )
        self.assertLess(result.kpis["mass_balance_pct"].value, 1.0e-6)
        self.assertLess(result.kpis["energy_balance_pct"].value, 1.0e-6)
        self.assertIn(
            "Closed acyclic mixer energy balance before mechanical design.",
            builder.build_log,
        )
        self.assertIn(
            "Closed acyclic mixer energy balance after mechanical design "
            "rerun.",
            builder.build_log,
        )

    def test_palette_built_equal_splitter_branches_round_trip_and_close(self):
        for flow_scale in (1.0, 1.05):
            with self.subTest(flow_scale=flow_scale):
                builder, model, history, graph_spec = (
                    self._build_palette_splitter_branches_case(flow_scale)
                )
                result = model.run(timeout_ms=180_000)
                units = list(model.get_process().getUnitOperations())
                names = [str(unit.getName()) for unit in units]

                self.assertIn("product split", names)
                self.assertIn("branch pump", names)
                self.assertIn("branch heater", names)
                expected_flow = 20_000.0 * flow_scale
                boundary_rows = result.raw["material_boundaries"]
                product_rows = [
                    row
                    for row in boundary_rows
                    if row["role"] == "product"
                ]
                self.assertEqual(len(product_rows), 2)
                for row in product_rows:
                    self.assertAlmostEqual(
                        row["mass_flow_kg_hr"],
                        expected_flow / 2.0,
                        delta=max(1.0e-6 * expected_flow, 1.0e-3),
                    )
                self.assertEqual(
                    result.kpis["material_feed_count"].value,
                    1.0,
                )
                self.assertEqual(
                    result.kpis["material_product_count"].value,
                    2.0,
                )
                self.assertAlmostEqual(
                    result.kpis["material_product_flow_kg_hr"].value,
                    expected_flow,
                    delta=max(1.0e-6 * expected_flow, 1.0e-3),
                )
                self.assertLess(
                    result.kpis["mass_balance_pct"].value,
                    1.0e-6,
                )
                self.assertLess(
                    result.kpis["component_balance_max_pct"].value,
                    1.0e-6,
                )
                self.assertLess(
                    result.kpis["energy_balance_pct"].value,
                    1.0e-6,
                )
                for constraint_name in (
                    "mass_balance",
                    "component_balance",
                    "energy_balance",
                ):
                    constraint = next(
                        constraint
                        for constraint in result.constraints
                        if constraint.name == constraint_name
                    )
                    self.assertEqual(constraint.status, "OK")
                self.assertIn(
                    "Configured graph splitter: product-split "
                    "(out_0=0.500000, out_1=0.500000)",
                    builder.build_log,
                )

                persisted = json.loads(
                    json.dumps(graph_spec, allow_nan=False)
                )
                self.assertEqual(persisted, graph_spec)
                splitter = next(
                    unit
                    for unit in persisted["units"]
                    if unit["id"] == "product-split"
                )
                self.assertEqual(
                    splitter["params"],
                    {"split_factor": 0.5},
                )
                history, pump_only_draft = undo_graph_history(history)
                self.assertEqual(
                    [unit["id"] for unit in pump_only_draft["units"]],
                    ["product-split", "branch-pump"],
                )
                history, final_draft = redo_graph_history(history)
                self.assertEqual(final_draft["units"], graph_spec["units"])
                self.assertEqual(
                    final_draft["connections"],
                    graph_spec["connections"],
                )
                print(
                    "native palette equal-split benchmark:",
                    f"scale={flow_scale:.2f}",
                    f"feed={expected_flow:.1f} kg/hr",
                    f"mass={result.kpis['mass_balance_pct'].value:.3e}%",
                    "components="
                    f"{result.kpis['component_balance_max_pct'].value:.3e}%",
                    f"energy={result.kpis['energy_balance_pct'].value:.3e}%",
                )

    def test_palette_three_way_splitter_round_trip_and_close(self):
        for flow_scale in (1.0, 1.05):
            with self.subTest(flow_scale=flow_scale):
                builder, model, graph_spec = (
                    self._build_palette_three_way_splitter_case(flow_scale)
                )
                result = model.run(timeout_ms=180_000)
                expected_flow = 30_000.0 * flow_scale
                product_flows = sorted(
                    row["mass_flow_kg_hr"]
                    for row in result.raw["material_boundaries"]
                    if row["role"] == "product"
                )

                self.assertEqual(len(product_flows), 3)
                for actual, factor in zip(product_flows, (0.2, 0.3, 0.5)):
                    self.assertAlmostEqual(
                        actual,
                        expected_flow * factor,
                        delta=max(1.0e-6 * expected_flow, 1.0e-3),
                    )
                self.assertAlmostEqual(
                    result.kpis["material_product_flow_kg_hr"].value,
                    expected_flow,
                    delta=max(1.0e-6 * expected_flow, 1.0e-3),
                )
                self.assertEqual(
                    result.kpis["material_product_count"].value,
                    3.0,
                )
                self.assertLess(
                    result.kpis["mass_balance_pct"].value,
                    1.0e-6,
                )
                self.assertLess(
                    result.kpis["component_balance_max_pct"].value,
                    1.0e-6,
                )
                self.assertLess(
                    result.kpis["energy_balance_pct"].value,
                    1.0e-6,
                )
                splitter_prefix = "three-way product splitter"
                self.assertEqual(
                    result.kpis[f"{splitter_prefix}.branchCount"].value,
                    3.0,
                )
                self.assertEqual(
                    result.kpis[
                        f"{splitter_prefix}.solvedBranchCount"
                    ].value,
                    3.0,
                )
                self.assertAlmostEqual(
                    result.kpis[
                        f"{splitter_prefix}.inletFlow_kg_hr"
                    ].value,
                    expected_flow,
                    delta=max(1.0e-6 * expected_flow, 1.0e-3),
                )
                self.assertLess(
                    result.kpis[
                        f"{splitter_prefix}.flowClosure_pct"
                    ].value,
                    1.0e-8,
                )
                self.assertAlmostEqual(
                    result.kpis[
                        f"{splitter_prefix}.splitFractionSum"
                    ].value,
                    1.0,
                    delta=1.0e-8,
                )
                for index, expected_fraction in enumerate(
                    (0.2, 0.3, 0.5)
                ):
                    self.assertAlmostEqual(
                        result.kpis[
                            f"{splitter_prefix}.branch{index}Fraction"
                        ].value,
                        expected_fraction,
                        delta=1.0e-8,
                    )
                    self.assertAlmostEqual(
                        result.kpis[
                            f"{splitter_prefix}.configuredBranch"
                            f"{index}Fraction"
                        ].value,
                        expected_fraction,
                        delta=1.0e-8,
                    )
                splitter_properties = next(
                    unit.properties
                    for unit in model.list_units()
                    if unit.name == splitter_prefix
                )
                self.assertEqual(splitter_properties["branchCount"], 3.0)
                self.assertLess(
                    splitter_properties["flowClosure_pct"],
                    1.0e-8,
                )
                self.assertIn(
                    "Configured graph splitter: three-way-product-splitter "
                    "(out_0=0.200000, out_1=0.300000, out_2=0.500000)",
                    builder.build_log,
                )
                persisted = json.loads(
                    json.dumps(graph_spec, allow_nan=False)
                )
                splitter = persisted["units"][0]
                self.assertEqual(
                    splitter["ports"]["material_out"],
                    ["out_0", "out_1", "out_2"],
                )
                self.assertEqual(
                    splitter["params"],
                    {"split_factors": [0.2, 0.3, 0.5]},
                )
                print(
                    "native palette three-way splitter benchmark:",
                    f"scale={flow_scale:.2f}",
                    f"feed={expected_flow:.1f} kg/hr",
                    f"mass={result.kpis['mass_balance_pct'].value:.3e}%",
                    "components="
                    f"{result.kpis['component_balance_max_pct'].value:.3e}%",
                    f"energy={result.kpis['energy_balance_pct'].value:.3e}%",
                )

    def test_palette_edited_unequal_split_round_trip_and_close(self):
        for flow_scale in (1.0, 1.05):
            with self.subTest(flow_scale=flow_scale):
                builder, model, history, graph_spec = (
                    self._build_palette_splitter_branches_case(
                        flow_scale,
                        split_factor=0.3,
                    )
                )
                result = model.run(timeout_ms=180_000)
                expected_flow = 20_000.0 * flow_scale
                product_flows = sorted(
                    row["mass_flow_kg_hr"]
                    for row in result.raw["material_boundaries"]
                    if row["role"] == "product"
                )
                self.assertEqual(len(product_flows), 2)
                self.assertAlmostEqual(
                    product_flows[0],
                    expected_flow * 0.3,
                    delta=max(1.0e-6 * expected_flow, 1.0e-3),
                )
                self.assertAlmostEqual(
                    product_flows[1],
                    expected_flow * 0.7,
                    delta=max(1.0e-6 * expected_flow, 1.0e-3),
                )
                self.assertAlmostEqual(
                    result.kpis["material_product_flow_kg_hr"].value,
                    expected_flow,
                    delta=max(1.0e-6 * expected_flow, 1.0e-3),
                )
                self.assertLess(
                    result.kpis["mass_balance_pct"].value,
                    1.0e-6,
                )
                self.assertLess(
                    result.kpis["component_balance_max_pct"].value,
                    1.0e-6,
                )
                self.assertLess(
                    result.kpis["energy_balance_pct"].value,
                    1.0e-6,
                )
                for constraint_name in (
                    "mass_balance",
                    "component_balance",
                    "energy_balance",
                ):
                    constraint = next(
                        constraint
                        for constraint in result.constraints
                        if constraint.name == constraint_name
                    )
                    self.assertEqual(constraint.status, "OK")
                self.assertIn(
                    "Configured graph splitter: product-split "
                    "(out_0=0.300000, out_1=0.700000)",
                    builder.build_log,
                )

                persisted = json.loads(
                    json.dumps(graph_spec, allow_nan=False)
                )
                splitter = next(
                    unit
                    for unit in persisted["units"]
                    if unit["id"] == "product-split"
                )
                self.assertEqual(
                    splitter["params"],
                    {"split_factor": 0.3},
                )

                history, pump_only_draft = undo_graph_history(history)
                self.assertEqual(
                    [unit["id"] for unit in pump_only_draft["units"]],
                    ["product-split", "branch-pump"],
                )
                history, split_only_draft = undo_graph_history(history)
                self.assertEqual(
                    [unit["id"] for unit in split_only_draft["units"]],
                    ["product-split"],
                )
                self.assertEqual(
                    split_only_draft["units"][0]["params"],
                    {"split_factor": 0.3},
                )
                history, default_draft = undo_graph_history(history)
                self.assertEqual(
                    default_draft["units"][0]["params"],
                    {"split_factor": 0.5},
                )
                history, edited_draft = redo_graph_history(history)
                self.assertEqual(
                    edited_draft["units"][0]["params"],
                    {"split_factor": 0.3},
                )
                history, _ = redo_graph_history(history)
                history, final_draft = redo_graph_history(history)
                self.assertEqual(final_draft["units"], graph_spec["units"])
                self.assertEqual(
                    final_draft["connections"],
                    graph_spec["connections"],
                )
                print(
                    "native palette unequal-split benchmark:",
                    f"scale={flow_scale:.2f}",
                    f"feed={expected_flow:.1f} kg/hr",
                    f"branches={product_flows[0]:.1f}/"
                    f"{product_flows[1]:.1f} kg/hr",
                    f"mass={result.kpis['mass_balance_pct'].value:.3e}%",
                    "components="
                    f"{result.kpis['component_balance_max_pct'].value:.3e}%",
                    f"energy={result.kpis['energy_balance_pct'].value:.3e}%",
                )

    def test_native_signed_work_heat_and_nearby_point(self):
        for flow_scale in (1.0, 1.05):
            with self.subTest(flow_scale=flow_scale):
                _, model = self._build_compression_cooling_case(
                    flow_scale
                )
                result = model.run(timeout_ms=180_000)
                summary = aggregate_energy_balance(result)
                transfers = result.raw["energy_transfers"]

                self.assertEqual(
                    [
                        (
                            row["unit_type"],
                            row["transfer_kind"],
                        )
                        for row in transfers
                    ],
                    [
                        ("Compressor", "shaft_work"),
                        ("Cooler", "heat"),
                    ],
                )
                self.assertGreater(
                    transfers[0]["energy_transfer_kW"],
                    0.0,
                )
                self.assertLess(
                    transfers[1]["energy_transfer_kW"],
                    0.0,
                )
                self.assertAlmostEqual(
                    summary["external_energy_transfer_kW"],
                    (
                        summary["product_enthalpy_kW"]
                        - summary["feed_enthalpy_kW"]
                    ),
                    delta=max(
                        abs(summary["product_enthalpy_kW"]) * 1.0e-8,
                        1.0e-6,
                    ),
                )
                self.assertLess(summary["imbalance_pct"], 1.0e-6)
                energy_constraint = next(
                    constraint
                    for constraint in result.constraints
                    if constraint.name == "energy_balance"
                )
                self.assertEqual(energy_constraint.status, "OK")
                print(
                    "native compression energy benchmark:",
                    f"scale={flow_scale:.2f}",
                    "work="
                    f"{transfers[0]['energy_transfer_kW']:.3f} kW",
                    "heat="
                    f"{transfers[1]['energy_transfer_kW']:.3f} kW",
                    "energy="
                    f"{summary['imbalance_pct']:.3e}%",
                )


if __name__ == "__main__":
    unittest.main()
