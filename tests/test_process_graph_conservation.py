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
from unittest.mock import patch

from process_chat.flowsheet_editor import (
    add_catalog_unit,
    connect_graph_ports,
    create_graph_history,
    extend_material_path,
    insert_mixer_on_connection,
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
from process_chat.process_model import NeqSimProcessModel
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


class NativePumpPerformanceTest(unittest.TestCase):
    """Benchmark editable native pump performance at nearby points."""

    @staticmethod
    def _run_case(flow_scale: float, efficiency: float):
        units, pump_id = add_catalog_unit([], "pump", "export pump")
        units[0]["params"].update(
            {
                "outlet_pressure_bara": 40.0,
                "efficiency": efficiency,
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

    def test_native_pump_conserves_and_trends_with_flow_and_efficiency(self):
        shaft_power = {}
        hydraulic_power = {}
        head = {}

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
                        json.loads(json.dumps(graph_spec, allow_nan=False)),
                        graph_spec,
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


class NativePipelineHydraulicsTest(unittest.TestCase):
    """Benchmark adiabatic native pipeline hydraulics and closure."""

    @staticmethod
    def _run_case(flow_scale: float, roughness_m: float):
        units, pipeline_id = add_catalog_unit(
            [],
            "pipeline",
            "transport pipeline",
        )
        units[0]["params"]["roughness"] = roughness_m
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
                    "params": {"ua_w_per_k": 100_000.0},
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

        with patch.object(
            builder,
            "build_acyclic_graph",
            return_value=expected_model,
        ) as graph_builder:
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

        output = io.StringIO()
        with (
            patch.object(
                ProcessBuilder,
                "build_acyclic_graph",
                return_value=_Model(),
            ),
            patch("neqsim.save_neqsim") as save_model,
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

        save_model.assert_called_once()
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
    def _build_palette_mixer_separator_case(flow_scale: float):
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
    def _build_palette_three_feed_separator_case(flow_scale: float):
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
