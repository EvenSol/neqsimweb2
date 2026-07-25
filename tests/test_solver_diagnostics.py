"""Regression tests for solved material-boundary diagnostic adapters."""

from __future__ import annotations

import math
import unittest
import weakref
from types import SimpleNamespace

from process_chat.process_model import (
    NeqSimProcessModel,
    _MaterialBoundaryIdentityTracker,
)
from process_chat.solver_diagnostics import (
    aggregate_material_balance,
    aggregate_validation_status,
    component_balance_rows,
    material_boundary_rows,
    solved_feed_flow_kg_hr,
)


class _FallbackStream:
    def __init__(
        self,
        name,
        mass_flow=None,
        hash_code=None,
        class_name="Stream",
    ):
        self._name = name
        self._mass_flow = mass_flow
        self._hash_code = hash_code
        self._class_name = class_name

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
            },
            {
                "role": "feed",
                "stream_name": "rich gas",
                "mass_flow_kg_hr": 40_000,
                "temperature_C": 35,
                "pressure_bara": 45,
                "molar_flow_mol_sec": 500,
            },
            {
                "role": "product",
                "stream_name": "mixed product",
                "mass_flow_kg_hr": 100_000,
                "temperature_C": 25,
                "pressure_bara": 45,
                "molar_flow_mol_sec": 1400,
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
