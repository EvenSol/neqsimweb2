"""Focused regression tests for pure flowsheet-editor schema helpers."""

from __future__ import annotations

import copy
import json
import unittest

from process_chat.flowsheet_editor import (
    add_catalog_unit,
    apply_graph_draft,
    build_graph_draft_dot,
    clone_material_inlet,
    connect_graph_ports,
    create_graph_draft,
    create_graph_history,
    create_inline_unit_spec,
    disconnect_graph_connection,
    extend_material_path,
    graph_connection_rows,
    graph_history_status,
    graph_port_rows,
    inlet_composition_property_rows,
    inlet_condition_property_rows,
    inline_unit_catalog,
    inline_unit_catalog_rows,
    inline_unit_property_rows,
    insert_inline_unit_on_connection,
    material_connection_rows,
    process_unit_property_rows,
    record_graph_history,
    redo_graph_history,
    remove_material_inlet,
    remove_inline_unit,
    rename_material_inlet,
    rename_inline_unit,
    undo_graph_history,
    update_inlet_composition,
    update_inlet_conditions,
    update_inline_unit_properties,
    update_process_unit_properties,
    validate_catalog_unit,
)


class UnitCatalogTest(unittest.TestCase):
    """Validate deterministic, isolated unit-operation metadata."""

    def test_catalog_is_searchable_and_returns_isolated_copies(self):
        rows = inline_unit_catalog_rows()
        self.assertEqual(
            [row["Type"] for row in rows],
            [
                "compressor",
                "cooler",
                "heater",
                "valve",
                "pump",
                "expander",
                "pipeline",
                "separator",
                "mixer",
                "splitter",
            ],
        )
        self.assertTrue(all(row["Category"] for row in rows))
        self.assertTrue(all(row["Description"] for row in rows))
        splitter_row = rows[-1]
        self.assertEqual(
            splitter_row["Equipment"],
            "Splitter",
        )
        self.assertIn(
            "editable out_0 flow fraction",
            splitter_row["Description"],
        )

        first_copy = inline_unit_catalog()
        first_copy["cooler"]["ports"]["material_out"].append("changed")
        second_copy = inline_unit_catalog()
        self.assertEqual(
            second_copy["cooler"]["ports"]["material_out"],
            ["out"],
        )

    def test_create_unit_uses_defaults_and_collision_free_id(self):
        unit = create_inline_unit_spec(
            "cooler",
            "Export Cooler",
            {"export-cooler", "export-cooler-2"},
        )
        self.assertEqual(unit["id"], "export-cooler-3")
        self.assertEqual(unit["name"], "Export Cooler")
        self.assertEqual(unit["ports"]["material_in"], ["in"])
        self.assertEqual(unit["params"]["outlet_temperature_C"], 35.0)
        validate_catalog_unit(unit)

    def test_adds_standalone_unit_for_later_port_connection(self):
        units = [
            {
                "id": "inlet-scrubber",
                "name": "Inlet scrubber",
                "type": "separator",
            }
        ]
        updated, unit_id = add_catalog_unit(
            units,
            "pump",
            "Condensate Pump",
            {"condensate-pump"},
            {"Feed"},
        )

        self.assertEqual(unit_id, "condensate-pump-2")
        self.assertEqual(updated[-1]["type"], "pump")
        self.assertEqual(updated[-1]["ports"]["material_in"], ["in"])
        self.assertEqual(updated[-1]["ports"]["material_out"], ["out"])
        self.assertEqual(updated[-1]["params"]["efficiency"], 0.75)
        self.assertEqual(len(units), 1)

    def test_adds_multi_inlet_mixer_separator_and_splitter_nodes(self):
        units, mixer_id = add_catalog_unit(
            [],
            "mixer",
            "Feed mixer",
            {"feed-a", "feed-b"},
            {"Feed A", "Feed B"},
        )
        units, separator_id = add_catalog_unit(
            units,
            "separator",
            "Product separator",
        )
        units, splitter_id = add_catalog_unit(
            units,
            "splitter",
            "Export split",
        )

        self.assertEqual(mixer_id, "feed-mixer")
        self.assertEqual(
            units[0]["ports"],
            {
                "material_in": ["in_0", "in_1"],
                "material_out": ["out"],
            },
        )
        self.assertEqual(separator_id, "product-separator")
        self.assertEqual(
            units[1]["ports"],
            {
                "material_in": ["in"],
                "material_out": ["gas", "liquid"],
            },
        )
        self.assertEqual(splitter_id, "export-split")
        self.assertEqual(
            units[2]["ports"],
            {
                "material_in": ["in"],
                "material_out": ["out_0", "out_1"],
            },
        )
        self.assertEqual(units[0]["params"], {})
        self.assertEqual(units[1]["params"], {})
        self.assertEqual(units[2]["params"], {"split_factor": 0.5})
        validate_catalog_unit(units[0])
        validate_catalog_unit(units[1])
        validate_catalog_unit(units[2])

    def test_standalone_unit_rejects_duplicate_names_without_mutation(self):
        units = [
            {
                "id": "pump",
                "name": "Condensate Pump",
                "type": "pump",
            }
        ]

        with self.assertRaisesRegex(ValueError, "duplicated"):
            add_catalog_unit(units, "pump", " condensate pump ")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            add_catalog_unit(units, "column", "Stabilizer")
        with self.assertRaisesRegex(ValueError, "duplicated"):
            add_catalog_unit(
                units,
                "pump",
                "Feed",
                reserved_names={"feed"},
            )
        self.assertEqual(len(units), 1)

    def test_standalone_unit_ignores_blank_stored_names_before_validation(self):
        units = [
            {"id": "legacy-unit", "type": "pump"},
            {"id": "blank-unit", "name": "   ", "type": "pump"},
        ]

        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            add_catalog_unit(
                units,
                "pump",
                "   ",
                reserved_names={"", "   "},
            )
        self.assertEqual(len(units), 2)

    def test_null_legacy_names_do_not_create_false_collisions(self):
        units = [{"id": "legacy-unit", "name": None, "type": "pump"}]
        updated_units, unit_id = add_catalog_unit(
            units,
            "pump",
            "None",
            reserved_names={None, ""},
        )
        self.assertEqual(updated_units[-1]["name"], "None")
        self.assertEqual(unit_id, "none")

        renamed_units = rename_inline_unit(
            [
                {
                    "id": "pump-a",
                    "name": "Pump A",
                    "type": "pump",
                    "ports": {
                        "material_in": ["in"],
                        "material_out": ["out"],
                    },
                    "params": {
                        "outlet_pressure_bara": 100.0,
                        "efficiency": 0.75,
                    },
                }
            ],
            "pump-a",
            "None",
            reserved_names={None, " "},
        )
        self.assertEqual(renamed_units[0]["name"], "None")

    def test_property_metadata_has_explicit_units_and_valid_defaults(self):
        catalog = inline_unit_catalog()

        for unit_type, definition in catalog.items():
            with self.subTest(unit_type=unit_type):
                rows = inline_unit_property_rows(
                    unit_type,
                    definition["default_params"],
                )
                self.assertEqual(
                    [row["key"] for row in rows],
                    list(definition["default_params"]),
                )
                self.assertTrue(all(row["label"] for row in rows))
                self.assertTrue(all(row["unit"] for row in rows))
                self.assertTrue(
                    all(
                        row["minimum"] <= row["value"] <= row["maximum"]
                        for row in rows
                    )
                )

        pressure_row = inline_unit_property_rows("compressor")[0]
        self.assertEqual(pressure_row["unit"], "bara (absolute)")
        self.assertEqual(pressure_row["format"], "%.2f")
        splitter_row = inline_unit_property_rows("splitter")[0]
        self.assertEqual(splitter_row["key"], "split_factor")
        self.assertEqual(splitter_row["label"], "Outlet out_0 flow fraction")
        self.assertEqual(splitter_row["unit"], "-")
        self.assertEqual(splitter_row["value"], 0.5)
        self.assertEqual(splitter_row["minimum"], 0.0)
        self.assertEqual(splitter_row["maximum"], 1.0)

    def test_splitter_property_rows_migrate_normalized_array_alias(self):
        rows = inline_unit_property_rows(
            "splitter",
            {"split_factors": [3.0, 7.0]},
        )
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["value"], 0.3)

        splitter = create_inline_unit_spec("splitter", "Product split", set())
        splitter["params"] = {"split_factors": [3.0, 7.0]}
        validate_catalog_unit(splitter)

    def test_property_metadata_rejects_invalid_parameter_shapes(self):
        invalid_cases = (
            (
                "compressor",
                {"outlet_pressure_bara": 80.0},
                "missing property 'isentropic_efficiency'",
            ),
            (
                "valve",
                {"outlet_pressure_bara": 40.0, "unknown": 1.0},
                "unsupported property 'unknown'",
            ),
            (
                "pump",
                {"outlet_pressure_bara": 80.0, "efficiency": True},
                "property 'efficiency' must be numeric",
            ),
            (
                "pipeline",
                {
                    "length": 1000.0,
                    "diameter": 0.0,
                    "roughness": 1.0e-5,
                },
                "property 'diameter' must be between",
            ),
            (
                "splitter",
                {},
                "missing property 'split_factor'",
            ),
            (
                "splitter",
                {
                    "split_factor": 0.5,
                    "split_factors": [0.5, 0.5],
                },
                "conflicting split_factor and split_factors",
            ),
            (
                "splitter",
                {"split_factors": [0.2, 0.3, 0.5]},
                "must contain exactly two values",
            ),
        )
        for unit_type, params, message in invalid_cases:
            with self.subTest(unit_type=unit_type, message=message):
                with self.assertRaisesRegex(ValueError, message):
                    inline_unit_property_rows(unit_type, params)

    def test_process_property_rows_cover_template_separators(self):
        compressor_params = {
            "outlet_pressure_bara": 125,
            "isentropic_efficiency": 0.82,
        }

        self.assertEqual(
            process_unit_property_rows("compressor", compressor_params),
            inline_unit_property_rows("compressor", compressor_params),
        )
        self.assertEqual(process_unit_property_rows("separator"), [])
        self.assertEqual(process_unit_property_rows("separator", {}), [])

    def test_process_property_rows_reject_invalid_requests(self):
        invalid_cases = (
            ("separator", {"pressure_drop_bar": 1.0}, "unsupported property"),
            ("separator", [], "params must be an object"),
            ("mixer", {"pressure_drop_bar": 1.0}, "unsupported property"),
            ("splitter", {"split_factor": 1.1}, "must be between"),
        )
        for unit_type, params, message in invalid_cases:
            with self.subTest(unit_type=unit_type):
                with self.assertRaisesRegex(ValueError, message):
                    process_unit_property_rows(unit_type, params)

    def test_invalid_catalog_requests_fail_explicitly(self):
        for unit_type, name, message in (
            ("unknown", "Unit", "Unsupported inline unit type"),
            ("cooler", " ", "name cannot be empty"),
            ("cooler", "x" * 81, "cannot exceed 80"),
        ):
            with self.subTest(unit_type=unit_type, name=name):
                with self.assertRaisesRegex(ValueError, message):
                    create_inline_unit_spec(unit_type, name, set())

        malformed = create_inline_unit_spec("valve", "Valve", set())
        malformed["ports"]["material_out"] = ["wrong"]
        with self.assertRaisesRegex(ValueError, "ports do not match"):
            validate_catalog_unit(malformed)


class InletConditionMetadataTest(unittest.TestCase):
    """Validate explicit-unit metadata for reusable material inlets."""

    def test_inlet_conditions_have_deterministic_units_and_bounds(self):
        rows = inlet_condition_property_rows(
            {
                "id": "feed-a",
                "temperature_C": 25.0,
                "pressure_bara": 50.0,
                "total_flow": 100_000.0,
                "flow_unit": "kg/hr",
            }
        )
        self.assertEqual(
            [row["key"] for row in rows],
            ["temperature_C", "pressure_bara", "total_flow"],
        )
        self.assertEqual(
            [row["unit"] for row in rows],
            ["°C", "bara (absolute)", "kg/hr"],
        )
        self.assertEqual(rows[0]["minimum"], -100.0)
        self.assertEqual(rows[1]["minimum"], 1.0)
        self.assertEqual(rows[2]["maximum"], 10_000_000.0)

    def test_invalid_inlet_condition_metadata_fails_explicitly(self):
        valid = {
            "id": "feed-a",
            "temperature_C": 25.0,
            "pressure_bara": 50.0,
            "total_flow": 100_000.0,
            "flow_unit": "kg/hr",
        }
        invalid_cases = (
            (None, "must be an object"),
            ({**valid, "id": " "}, "non-empty id"),
            ({**valid, "flow_unit": "kg/s"}, "mass flow in kg/hr"),
            (
                {key: value for key, value in valid.items() if key != "pressure_bara"},
                "missing condition 'pressure_bara'",
            ),
            ({**valid, "temperature_C": True}, "must be numeric"),
            ({**valid, "pressure_bara": float("inf")}, "must be finite"),
            ({**valid, "total_flow": 0.0}, "must be between"),
        )
        for inlet, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    inlet_condition_property_rows(inlet)


class InletCompositionMetadataTest(unittest.TestCase):
    """Validate explicit-unit composition metadata for material inlets."""

    def test_composition_rows_preserve_registry_order_and_units(self):
        rows = inlet_composition_property_rows(
            {
                "id": "feed-a",
                "composition": {
                    "methane": 0.80,
                    "ethane": 0.15,
                    "propane": 0.05,
                },
                "composition_basis": "mole_fraction",
            }
        )

        self.assertEqual(
            [row["component"] for row in rows],
            ["methane", "ethane", "propane"],
        )
        self.assertEqual(
            [row["mole_fraction"] for row in rows],
            [0.80, 0.15, 0.05],
        )
        self.assertTrue(all(row["unit"] == "mol/mol" for row in rows))
        self.assertTrue(all(row["minimum"] == 0.0 for row in rows))
        self.assertTrue(all(row["maximum"] == 1.0 for row in rows))

    def test_invalid_composition_metadata_fails_explicitly(self):
        valid = {
            "id": "feed-a",
            "composition": {"methane": 0.90, "ethane": 0.10},
            "composition_basis": "mole_fraction",
        }
        invalid_cases = (
            (None, "must be an object"),
            ({**valid, "id": " "}, "non-empty id"),
            ({**valid, "composition_basis": "mass_fraction"}, "mole-fraction"),
            ({**valid, "composition": {}}, "non-empty composition"),
            (
                {**valid, "composition": {" ": 0.90, "ethane": 0.10}},
                "empty component name",
            ),
            (
                {
                    **valid,
                    "composition": {"methane": 0.90, "Methane": 0.10},
                },
                "duplicate component",
            ),
            (
                {**valid, "composition": {"methane": True, "ethane": 0.0}},
                "must be numeric",
            ),
            (
                {
                    **valid,
                    "composition": {
                        "methane": float("inf"),
                        "ethane": 0.0,
                    },
                },
                "must be finite",
            ),
            (
                {**valid, "composition": {"methane": 1.1, "ethane": -0.1}},
                "between 0 and 1",
            ),
            (
                {**valid, "composition": {"methane": 0.80, "ethane": 0.10}},
                "must sum to 1.0",
            ),
        )
        for inlet, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    inlet_composition_property_rows(inlet)


class InletConditionUpdateTest(unittest.TestCase):
    """Validate independent, immutable updates for multiple graph inlets."""

    def setUp(self):
        self.inlets = [
            {
                "id": "feed-a",
                "name": "feed A",
                "fluid_package_id": "base-fluid",
                "composition": {"methane": 0.90, "ethane": 0.10},
                "composition_basis": "mole_fraction",
                "temperature_C": 25.0,
                "pressure_bara": 50.0,
                "total_flow": 60_000.0,
                "flow_unit": "kg/hr",
            },
            {
                "id": "feed-b",
                "name": "feed B",
                "fluid_package_id": "base-fluid",
                "composition": {"methane": 0.80, "ethane": 0.20},
                "composition_basis": "mole_fraction",
                "temperature_C": 35.0,
                "pressure_bara": 45.0,
                "total_flow": 40_000.0,
                "flow_unit": "kg/hr",
            },
        ]

    def test_updates_one_inlet_without_changing_characterization_or_peer(self):
        updated = update_inlet_conditions(
            self.inlets,
            "feed-b",
            {
                "temperature_C": 30.0,
                "pressure_bara": 47.5,
                "total_flow": 42_000.0,
            },
        )

        self.assertEqual(updated[0], self.inlets[0])
        self.assertEqual(updated[1]["temperature_C"], 30.0)
        self.assertEqual(updated[1]["pressure_bara"], 47.5)
        self.assertEqual(updated[1]["total_flow"], 42_000.0)
        self.assertEqual(
            updated[1]["fluid_package_id"],
            self.inlets[1]["fluid_package_id"],
        )
        self.assertEqual(
            updated[1]["composition"],
            self.inlets[1]["composition"],
        )
        self.assertEqual(self.inlets[1]["temperature_C"], 35.0)

    def test_normalizes_numeric_values_and_supports_partial_updates(self):
        updated = update_inlet_conditions(
            self.inlets,
            "feed-a",
            {"pressure_bara": "52.5"},
        )
        self.assertEqual(updated[0]["pressure_bara"], 52.5)
        self.assertIsInstance(updated[0]["pressure_bara"], float)
        self.assertEqual(updated[0]["temperature_C"], 25.0)
        self.assertEqual(updated[0]["total_flow"], 60_000.0)

    def test_invalid_updates_fail_without_mutating_inputs(self):
        invalid_cases = (
            ("missing", {}, "Unknown material inlet"),
            ("feed-a", [], "updates must be an object"),
            ("feed-a", {"composition": {}}, "unsupported condition"),
            ("feed-a", {"pressure_bara": 0.0}, "must be between"),
        )
        for inlet_id, updates, message in invalid_cases:
            with self.subTest(inlet_id=inlet_id, message=message):
                with self.assertRaisesRegex(ValueError, message):
                    update_inlet_conditions(
                        self.inlets,
                        inlet_id,
                        updates,
                    )

        duplicated = [self.inlets[0], copy.deepcopy(self.inlets[0])]
        with self.assertRaisesRegex(ValueError, "duplicated"):
            update_inlet_conditions(duplicated, "feed-a", {})
        self.assertEqual(self.inlets[0]["pressure_bara"], 50.0)


class InletCompositionUpdateTest(unittest.TestCase):
    """Validate isolated normalized composition updates for graph inlets."""

    def setUp(self):
        self.inlets = [
            {
                "id": "feed-a",
                "name": "feed A",
                "fluid_package_id": "base-fluid",
                "composition": {"methane": 0.90, "ethane": 0.10},
                "composition_basis": "mole_fraction",
                "temperature_C": 25.0,
                "pressure_bara": 50.0,
                "total_flow": 60_000.0,
                "flow_unit": "kg/hr",
            },
            {
                "id": "feed-b",
                "name": "feed B",
                "fluid_package_id": "base-fluid",
                "composition": {"methane": 0.80, "ethane": 0.20},
                "composition_basis": "mole_fraction",
                "temperature_C": 35.0,
                "pressure_bara": 45.0,
                "total_flow": 40_000.0,
                "flow_unit": "kg/hr",
            },
        ]

    def test_updates_one_inlet_without_changing_peer_or_conditions(self):
        updated = update_inlet_composition(
            self.inlets,
            "feed-b",
            {"methane": 0.70, "ethane": 0.30},
        )

        self.assertEqual(updated[0], self.inlets[0])
        self.assertEqual(
            updated[1]["composition"],
            {"methane": 0.70, "ethane": 0.30},
        )
        self.assertEqual(updated[1]["temperature_C"], 35.0)
        self.assertEqual(updated[1]["pressure_bara"], 45.0)
        self.assertEqual(updated[1]["total_flow"], 40_000.0)
        self.assertEqual(
            updated[1]["fluid_package_id"],
            self.inlets[1]["fluid_package_id"],
        )
        self.assertEqual(
            self.inlets[1]["composition"],
            {"methane": 0.80, "ethane": 0.20},
        )

    def test_normalizes_entered_fractions_in_registry_order(self):
        updated = update_inlet_composition(
            self.inlets,
            "feed-a",
            {"ethane": "0.4", "methane": "0.6"},
        )

        self.assertEqual(
            list(updated[0]["composition"]),
            ["methane", "ethane"],
        )
        self.assertAlmostEqual(updated[0]["composition"]["methane"], 0.6)
        self.assertAlmostEqual(updated[0]["composition"]["ethane"], 0.4)
        self.assertIsInstance(
            updated[0]["composition"]["methane"],
            float,
        )

        normalized = update_inlet_composition(
            self.inlets,
            "feed-a",
            {"methane": 0.45, "ethane": 0.05},
        )
        self.assertAlmostEqual(
            sum(normalized[0]["composition"].values()),
            1.0,
        )
        self.assertAlmostEqual(
            normalized[0]["composition"]["methane"],
            0.9,
        )

    def test_invalid_updates_fail_without_mutating_inputs(self):
        invalid_cases = (
            ("missing", {}, "Unknown material inlet"),
            ("feed-a", [], "composition must be an object"),
            (
                "feed-a",
                {"methane": 1.0},
                "shared component registry exactly",
            ),
            (
                "feed-a",
                {"methane": True, "ethane": 0.0},
                "must be numeric",
            ),
            (
                "feed-a",
                {"methane": float("nan"), "ethane": 0.0},
                "must be finite",
            ),
            (
                "feed-a",
                {"methane": 1.1, "ethane": -0.1},
                "between 0 and 1",
            ),
            (
                "feed-a",
                {"methane": 0.0, "ethane": 0.0},
                "total must be positive",
            ),
        )
        for inlet_id, composition, message in invalid_cases:
            with self.subTest(inlet_id=inlet_id, message=message):
                with self.assertRaisesRegex(ValueError, message):
                    update_inlet_composition(
                        self.inlets,
                        inlet_id,
                        composition,
                    )

        duplicated = [self.inlets[0], copy.deepcopy(self.inlets[0])]
        with self.assertRaisesRegex(ValueError, "duplicated"):
            update_inlet_composition(
                duplicated,
                "feed-a",
                {"methane": 0.90, "ethane": 0.10},
            )
        self.assertEqual(
            self.inlets[0]["composition"],
            {"methane": 0.90, "ethane": 0.10},
        )


class MaterialInletLifecycleTest(unittest.TestCase):
    """Validate safe creation, naming, and removal of independent feeds."""

    def setUp(self):
        self.inlets = [
            {
                "id": "feed",
                "name": "Feed",
                "fluid_package_id": "base-fluid",
                "temperature_C": 25.0,
                "pressure_bara": 50.0,
                "total_flow": 100_000.0,
                "flow_unit": "kg/hr",
                "composition_basis": "mole_fraction",
                "composition": {"methane": 0.9, "ethane": 0.1},
            }
        ]

    def test_clone_creates_collision_free_compatible_independent_feed(self):
        with_first, first_id = clone_material_inlet(
            self.inlets,
            "feed",
            "Tie-in Feed",
        )
        updated, second_id = clone_material_inlet(
            with_first,
            "feed",
            "Tie in Feed",
            {"tie-in-feed"},
        )

        self.assertEqual(first_id, "tie-in-feed")
        self.assertEqual(second_id, "tie-in-feed-2")
        self.assertEqual(updated[1]["fluid_package_id"], "base-fluid")
        self.assertEqual(updated[1]["composition"], self.inlets[0]["composition"])
        updated[1]["composition"]["methane"] = 0.8
        self.assertEqual(updated[0]["composition"]["methane"], 0.9)
        self.assertEqual(self.inlets[0]["composition"]["methane"], 0.9)
        self.assertEqual(len(self.inlets), 1)

    def test_rename_preserves_stable_identity_and_rejects_duplicate_name(self):
        inlets, inlet_id = clone_material_inlet(
            self.inlets,
            "feed",
            "Tie-in Feed",
        )
        renamed = rename_material_inlet(inlets, inlet_id, "Satellite Feed")

        self.assertEqual(renamed[1]["id"], inlet_id)
        self.assertEqual(renamed[1]["name"], "Satellite Feed")
        self.assertEqual(inlets[1]["name"], "Tie-in Feed")
        with self.assertRaisesRegex(ValueError, "duplicated"):
            rename_material_inlet(inlets, inlet_id, "feed")
        with self.assertRaisesRegex(ValueError, "duplicated"):
            rename_material_inlet(
                inlets,
                inlet_id,
                "Condensate Pump",
                {"condensate pump"},
            )

    def test_null_legacy_feed_names_do_not_create_false_collisions(self):
        legacy_inlet = {
            **copy.deepcopy(self.inlets[0]),
            "id": "legacy-feed",
            "name": None,
        }
        inlets, inlet_id = clone_material_inlet(
            [*self.inlets, legacy_inlet],
            "feed",
            "Satellite",
            reserved_names={None, ""},
        )
        renamed = rename_material_inlet(
            inlets,
            inlet_id,
            "None",
            reserved_names={None, " "},
        )

        self.assertEqual(renamed[-1]["name"], "None")

    def test_remove_requires_an_unconnected_nonprotected_secondary_feed(self):
        inlets, inlet_id = clone_material_inlet(
            self.inlets,
            "feed",
            "Tie-in Feed",
        )
        connection = {
            "id": "tie-in-to-mixer",
            "type": "material",
            "source": {"kind": "inlet", "id": inlet_id, "port": "out"},
            "target": {"kind": "unit", "id": "mixer", "port": "in_1"},
        }

        with self.assertRaisesRegex(ValueError, "still connected"):
            remove_material_inlet(inlets, [connection], inlet_id)
        with self.assertRaisesRegex(ValueError, "protected"):
            remove_material_inlet(inlets, [], "feed", {"feed"})

        remaining = remove_material_inlet(inlets, [], inlet_id)
        self.assertEqual(remaining, self.inlets)
        self.assertEqual(len(inlets), 2)

    def test_invalid_clone_and_last_inlet_removal_fail_without_mutation(self):
        invalid_clones = (
            ("missing", "Feed B", "Unknown material inlet"),
            ("feed", " ", "name cannot be empty"),
            ("feed", "feed", "duplicated"),
        )
        for source_id, name, message in invalid_clones:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    clone_material_inlet(self.inlets, source_id, name)
        with self.assertRaisesRegex(ValueError, "duplicated"):
            clone_material_inlet(
                self.inlets,
                "feed",
                "Pump",
                reserved_names={"pump"},
            )

        with self.assertRaisesRegex(ValueError, "at least one"):
            remove_material_inlet(self.inlets, [], "feed")
        self.assertEqual(len(self.inlets), 1)


class InlineInsertionTest(unittest.TestCase):
    """Validate transactional connection splitting for draft graphs."""

    def setUp(self):
        self.units = [
            {
                "id": "compressor-1",
                "name": "compressor 1",
                "type": "compressor",
                "ports": {
                    "material_in": ["in"],
                    "material_out": ["out"],
                },
                "params": {
                    "outlet_pressure_bara": 80.0,
                    "isentropic_efficiency": 0.78,
                },
            }
        ]
        self.connections = [
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
                    "id": "compressor-1",
                    "port": "in",
                },
            }
        ]

    def test_insert_splits_selected_path_without_mutating_inputs(self):
        new_units, new_connections, new_unit_id = (
            insert_inline_unit_on_connection(
                self.units,
                self.connections,
                "feed-to-compressor",
                "cooler",
                "Feed Cooler",
            )
        )

        self.assertEqual(new_unit_id, "feed-cooler")
        self.assertEqual(
            [unit["id"] for unit in new_units],
            ["feed-cooler", "compressor-1"],
        )
        self.assertEqual(
            new_connections[0]["source"],
            self.connections[0]["source"],
        )
        self.assertEqual(
            new_connections[0]["target"],
            {"kind": "unit", "id": "feed-cooler", "port": "in"},
        )
        self.assertEqual(
            new_connections[1]["source"],
            {"kind": "unit", "id": "feed-cooler", "port": "out"},
        )
        self.assertEqual(
            new_connections[1]["target"],
            self.connections[0]["target"],
        )
        self.assertEqual([unit["id"] for unit in self.units], ["compressor-1"])
        self.assertEqual(len(self.connections), 1)

    def test_insert_uses_unique_object_and_connection_ids(self):
        units = [
            *self.units,
            create_inline_unit_spec("cooler", "Feed Cooler", set()),
        ]
        connections = [
            *self.connections,
            {
                "id": "feed-cooler-2-to-compressor-1",
                "type": "material",
                "source": {
                    "kind": "unit",
                    "id": "unused",
                    "port": "out",
                },
                "target": {
                    "kind": "unit",
                    "id": "unused-target",
                    "port": "in",
                },
            },
        ]
        new_units, new_connections, new_unit_id = (
            insert_inline_unit_on_connection(
                units,
                connections,
                "feed-to-compressor",
                "cooler",
                "Feed Cooler",
            )
        )
        self.assertEqual(new_unit_id, "feed-cooler-2")
        self.assertEqual(
            new_connections[1]["id"],
            "feed-cooler-2-to-compressor-1-2",
        )
        self.assertEqual(len(new_units), 3)

    def test_insert_reserves_unconnected_inlet_ids(self):
        new_units, _, new_unit_id = insert_inline_unit_on_connection(
            self.units,
            self.connections,
            "feed-to-compressor",
            "cooler",
            "Satellite Feed",
            {"satellite-feed"},
        )

        self.assertEqual(new_unit_id, "satellite-feed-2")
        self.assertEqual(new_units[0]["id"], "satellite-feed-2")

    def test_invalid_path_requests_fail_without_partial_edits(self):
        invalid_cases = (
            ("missing", self.connections, "Unknown graph connection"),
            (
                "feed-to-compressor",
                [{**self.connections[0], "type": "energy"}],
                "only be inserted in material paths",
            ),
            (
                "feed-to-compressor",
                [{**self.connections[0], "target": {}}],
                "target needs kind",
            ),
        )
        for connection_id, connections, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    insert_inline_unit_on_connection(
                        self.units,
                        connections,
                        connection_id,
                        "cooler",
                        "New Cooler",
                    )
        self.assertEqual([unit["id"] for unit in self.units], ["compressor-1"])
        self.assertEqual(len(self.connections), 1)


class InlineUnitLifecycleTest(unittest.TestCase):
    """Validate safe rename and removal of added inline equipment."""

    def setUp(self):
        self.units = [
            create_inline_unit_spec(
                "cooler",
                "Product Cooler",
                set(),
            )
        ]
        self.connections = [
            {
                "id": "feed-to-cooler",
                "type": "material",
                "source": {
                    "kind": "inlet",
                    "id": "feed",
                    "port": "out",
                },
                "target": {
                    "kind": "unit",
                    "id": "product-cooler",
                    "port": "in",
                },
            }
        ]

    def _insert_valve(self):
        return insert_inline_unit_on_connection(
            self.units,
            self.connections,
            "feed-to-cooler",
            "valve",
            "Product Valve",
        )

    def test_rename_preserves_stable_id_routes_and_inputs(self):
        units, connections, inserted_id = self._insert_valve()
        renamed = rename_inline_unit(
            units,
            inserted_id,
            " Export Pressure Valve ",
        )

        renamed_unit = next(
            unit for unit in renamed if unit["id"] == inserted_id
        )
        self.assertEqual(inserted_id, "product-valve")
        self.assertEqual(renamed_unit["name"], "Export Pressure Valve")
        self.assertEqual(
            [connection["id"] for connection in connections],
            ["feed-to-cooler", "product-valve-to-product-cooler"],
        )
        self.assertEqual(
            next(unit for unit in units if unit["id"] == inserted_id)["name"],
            "Product Valve",
        )
        self.assertEqual(self.units[0]["name"], "Product Cooler")

    def test_rename_rejects_unknown_invalid_or_duplicate_names(self):
        units, _, inserted_id = self._insert_valve()
        invalid_cases = (
            ("missing", "Renamed", "Unknown graph unit"),
            (inserted_id, " ", "name cannot be empty"),
            (inserted_id, "x" * 81, "cannot exceed 80"),
            (inserted_id, "product cooler", "already in use"),
        )
        for unit_id, new_name, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    rename_inline_unit(units, unit_id, new_name)
        with self.assertRaisesRegex(ValueError, "already in use"):
            rename_inline_unit(
                units,
                inserted_id,
                "Satellite Feed",
                {"satellite feed"},
            )

    def test_property_update_is_normalized_and_input_safe(self):
        updated = update_inline_unit_properties(
            self.units,
            "product-cooler",
            {
                "outlet_temperature_C": 42,
                "pressure_drop_bar": 1,
            },
        )

        self.assertEqual(
            updated[0]["params"],
            {
                "outlet_temperature_C": 42.0,
                "pressure_drop_bar": 1.0,
            },
        )
        self.assertEqual(
            self.units[0]["params"],
            {
                "outlet_temperature_C": 35.0,
                "pressure_drop_bar": 0.0,
            },
        )
        validate_catalog_unit(updated[0])

    def test_property_update_rejects_invalid_requests_without_mutation(self):
        invalid_cases = (
            ("missing", {}, "Unknown graph unit"),
            (
                "product-cooler",
                {"outlet_temperature_C": float("nan")},
                "must be finite",
            ),
            (
                "product-cooler",
                {"pressure_drop_bar": -0.1},
                "must be between",
            ),
            (
                "product-cooler",
                {"unknown": 1.0},
                "unsupported property",
            ),
            (
                "product-cooler",
                [],
                "updates must be an object",
            ),
        )
        original = json.loads(json.dumps(self.units))
        for unit_id, updates, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    update_inline_unit_properties(
                        self.units,
                        unit_id,
                        updates,
                    )
        self.assertEqual(self.units, original)


    def test_remove_restores_original_route_without_mutating_inputs(self):
        units, connections, inserted_id = self._insert_valve()
        restored_units, restored_connections = remove_inline_unit(
            units,
            connections,
            inserted_id,
        )

        self.assertEqual(restored_units, self.units)
        self.assertEqual(restored_connections, self.connections)
        self.assertEqual(len(units), 2)
        self.assertEqual(len(connections), 2)
        self.assertTrue(
            any(unit["id"] == inserted_id for unit in units)
        )

    def test_remove_unconnected_and_terminal_standalone_units(self):
        units, pump_id = add_catalog_unit(
            self.units,
            "pump",
            "Condensate pump",
        )
        unconnected_units, unconnected_connections = remove_inline_unit(
            units,
            self.connections,
            pump_id,
        )
        self.assertEqual(unconnected_units, self.units)
        self.assertEqual(unconnected_connections, self.connections)

        terminal_connection = {
            "id": "cooler-to-pump",
            "type": "material",
            "source": {
                "kind": "unit",
                "id": "product-cooler",
                "port": "out",
            },
            "target": {
                "kind": "unit",
                "id": pump_id,
                "port": "in",
            },
        }
        terminal_units, terminal_connections = remove_inline_unit(
            units,
            [*self.connections, terminal_connection],
            pump_id,
        )
        self.assertEqual(terminal_units, self.units)
        self.assertEqual(terminal_connections, self.connections)
        self.assertEqual(len(units), len(self.units) + 1)

    def test_remove_rejects_branches_and_nonmaterial_references(self):
        units, connections, inserted_id = self._insert_valve()
        branch = {
            "id": "valve-branch",
            "type": "material",
            "source": {
                "kind": "unit",
                "id": inserted_id,
                "port": "out",
            },
            "target": {
                "kind": "unit",
                "id": "branch-target",
                "port": "in",
            },
        }
        energy_link = {
            "id": "valve-energy",
            "type": "energy",
            "source": {
                "kind": "unit",
                "id": inserted_id,
                "port": "energy",
            },
            "target": {
                "kind": "unit",
                "id": "utility",
                "port": "in",
            },
        }
        for extra_connection, message in (
            (branch, "requires no connections"),
            (energy_link, "has unsupported connections"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    remove_inline_unit(
                        units,
                        [*connections, extra_connection],
                        inserted_id,
                    )
        self.assertEqual(len(units), 2)
        self.assertEqual(len(connections), 2)


class ProcessUnitPropertyUpdateTest(unittest.TestCase):
    """Validate property updates beyond palette-inserted equipment."""

    def setUp(self):
        self.units = [
            {
                "id": "inlet-scrubber",
                "name": "inlet scrubber",
                "type": "separator",
                "ports": {
                    "material_in": ["in"],
                    "material_out": ["gas", "liquid"],
                },
            },
            {
                "id": "compressor-stage-1",
                "name": "compressor stage 1",
                "type": "compressor",
                "ports": {
                    "material_in": ["in"],
                    "material_out": ["out"],
                },
                "params": {
                    "outlet_pressure_bara": 80.0,
                    "isentropic_efficiency": 0.78,
                },
            },
        ]

    def test_updates_template_unit_without_mutating_inputs(self):
        updated = update_process_unit_properties(
            self.units,
            "compressor-stage-1",
            {
                "outlet_pressure_bara": 90,
                "isentropic_efficiency": 0.80,
            },
        )

        self.assertEqual(
            updated[1]["params"],
            {
                "outlet_pressure_bara": 90.0,
                "isentropic_efficiency": 0.8,
            },
        )
        self.assertEqual(
            self.units[1]["params"]["outlet_pressure_bara"],
            80.0,
        )

    def test_separator_accepts_noop_and_preserves_paramless_shape(self):
        updated = update_process_unit_properties(
            self.units,
            "inlet-scrubber",
            {},
        )

        self.assertNotIn("params", updated[0])
        self.assertEqual(updated, self.units)

    def test_updates_splitter_and_canonicalizes_legacy_array(self):
        splitter = create_inline_unit_spec(
            "splitter",
            "Product split",
            set(),
        )
        splitter["params"] = {"split_factors": [3.0, 7.0]}

        updated = update_process_unit_properties(
            [splitter],
            splitter["id"],
            {"split_factor": 0.4},
        )

        self.assertEqual(updated[0]["params"], {"split_factor": 0.4})
        self.assertEqual(
            splitter["params"],
            {"split_factors": [3.0, 7.0]},
        )

    def test_rejects_invalid_generic_updates_without_mutation(self):
        duplicated = [*self.units, dict(self.units[1])]
        invalid_cases = (
            (
                self.units,
                "inlet-scrubber",
                {"pressure_drop_bar": 1.0},
                "unsupported property",
            ),
            (
                self.units,
                "compressor-stage-1",
                {"outlet_pressure_bara": float("inf")},
                "must be finite",
            ),
            (
                self.units,
                "missing",
                {},
                "Unknown graph unit",
            ),
            (
                duplicated,
                "compressor-stage-1",
                {},
                "duplicated",
            ),
        )
        original = json.loads(json.dumps(self.units))
        for units, unit_id, updates, message in invalid_cases:
            with self.subTest(unit_id=unit_id, message=message):
                with self.assertRaisesRegex(ValueError, message):
                    update_process_unit_properties(
                        units,
                        unit_id,
                        updates,
                    )
        self.assertEqual(self.units, original)


class GraphDraftLifecycleTest(unittest.TestCase):
    """Validate isolated draft persistence and presentation helpers."""

    def setUp(self):
        self.units = [
            create_inline_unit_spec(
                "cooler",
                "Product Cooler",
                set(),
            )
        ]
        self.connections = [
            {
                "id": "feed-to-cooler",
                "type": "material",
                "source": {
                    "kind": "inlet",
                    "id": "feed",
                    "port": "out",
                },
                "target": {
                    "kind": "unit",
                    "id": "product-cooler",
                    "port": "in",
                },
            }
        ]

    def test_create_and_apply_draft_preserve_original_objects(self):
        draft = create_graph_draft(self.units, self.connections)
        case_spec = {
            "schema_version": 3,
            "name": "case",
            "units": [],
            "connections": [],
        }
        updated = apply_graph_draft(case_spec, draft)

        self.assertEqual(updated["units"], self.units)
        self.assertEqual(updated["connections"], self.connections)
        draft["units"][0]["name"] = "changed draft"
        updated["units"][0]["name"] = "changed case"
        self.assertEqual(self.units[0]["name"], "Product Cooler")
        self.assertEqual(case_spec["units"], [])

    def test_inlet_aware_draft_applies_without_mutating_case_or_draft(self):
        original_inlets = [
            {
                "id": "feed-a",
                "name": "Feed A",
                "temperature_C": 20.0,
            }
        ]
        edited_inlets = [
            {
                "id": "feed-a",
                "name": "Feed A",
                "temperature_C": 35.0,
            },
            {
                "id": "feed-b",
                "name": "Feed B",
                "temperature_C": 10.0,
            },
        ]
        draft = create_graph_draft(
            self.units,
            self.connections,
            edited_inlets,
        )
        case_spec = {
            "schema_version": 3,
            "inlets": original_inlets,
            "units": [],
            "connections": [],
        }

        updated = apply_graph_draft(case_spec, draft)

        self.assertEqual(updated["inlets"], edited_inlets)
        self.assertEqual(case_spec["inlets"], original_inlets)
        updated["inlets"][0]["temperature_C"] = 50.0
        draft["inlets"][1]["name"] = "changed"
        self.assertEqual(edited_inlets[0]["temperature_C"], 35.0)
        self.assertEqual(edited_inlets[1]["name"], "Feed B")

    def test_legacy_draft_preserves_existing_case_inlets(self):
        legacy_draft = create_graph_draft(self.units, self.connections)
        case_spec = {
            "schema_version": 3,
            "inlets": [{"id": "feed-a", "name": "Feed A"}],
            "units": [],
            "connections": [],
        }

        updated = apply_graph_draft(case_spec, legacy_draft)

        self.assertEqual(updated["inlets"], case_spec["inlets"])
        self.assertNotIn("inlets", legacy_draft)

    def test_legacy_draft_rejects_unit_id_retained_by_case_inlet(self):
        legacy_draft = create_graph_draft(self.units, self.connections)
        case_spec = {
            "schema_version": 3,
            "inlets": [
                {"id": "product-cooler", "name": "Conflicting feed"}
            ],
            "units": [],
            "connections": [],
        }

        with self.assertRaisesRegex(ValueError, "both an inlet and a unit"):
            apply_graph_draft(case_spec, legacy_draft)

    def test_null_draft_inlets_cannot_bypass_retained_inlet_validation(self):
        malformed_draft = create_graph_draft(
            self.units,
            self.connections,
        )
        malformed_draft["inlets"] = None
        case_spec = {
            "schema_version": 3,
            "inlets": [
                {"id": "product-cooler", "name": "Conflicting feed"}
            ],
            "units": [],
            "connections": [],
        }

        with self.assertRaisesRegex(
            ValueError,
            "Graph draft inlets must be an array",
        ):
            apply_graph_draft(case_spec, malformed_draft)

    def test_material_connection_rows_ignore_energy_links(self):
        rows = material_connection_rows(
            [
                *self.connections,
                {
                    "id": "heater-duty",
                    "type": "energy",
                    "source": {
                        "kind": "unit",
                        "id": "utility",
                        "port": "out",
                    },
                    "target": {
                        "kind": "unit",
                        "id": "heater",
                        "port": "energy",
                    },
                },
            ]
        )
        self.assertEqual(
            rows,
            [
                {
                    "id": "feed-to-cooler",
                    "label": "feed:out → product-cooler:in",
                }
            ],
        )

    def test_invalid_drafts_fail_before_application(self):
        invalid_drafts = (
            (
                {
                    "schema_version": 2,
                    "units": self.units,
                    "connections": self.connections,
                },
                "Unsupported graph draft schema",
            ),
            (
                {
                    "schema_version": 1,
                    "units": [*self.units, self.units[0]],
                    "connections": self.connections,
                },
                "unit id 'product-cooler' is duplicated",
            ),
            (
                {
                    "schema_version": 1,
                    "inlets": [
                        {"id": "feed"},
                        {"id": "feed"},
                    ],
                    "units": self.units,
                    "connections": self.connections,
                },
                "inlet id 'feed' is duplicated",
            ),
            (
                {
                    "schema_version": 1,
                    "inlets": [{"id": "product-cooler"}],
                    "units": self.units,
                    "connections": self.connections,
                },
                "duplicated between an inlet and a unit",
            ),
            (
                {
                    "schema_version": 1,
                    "units": self.units,
                    "connections": [
                        {**self.connections[0], "type": "signal"}
                    ],
                },
                "has invalid type",
            ),
        )
        for draft, message in invalid_drafts:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    apply_graph_draft(
                        {"units": [], "connections": []},
                        draft,
                    )

    def test_inserted_draft_round_trips_through_case_json(self):
        units, connections, inserted_id = (
            insert_inline_unit_on_connection(
                self.units,
                self.connections,
                "feed-to-cooler",
                "valve",
                "Product pressure valve",
            )
        )
        draft = create_graph_draft(units, connections)
        encoded_case = json.dumps(
            {
                "schema_version": 3,
                "name": "persisted draft",
                "units": draft["units"],
                "connections": draft["connections"],
            },
            allow_nan=False,
        )
        loaded_case = json.loads(encoded_case)
        loaded_draft = create_graph_draft(
            loaded_case["units"],
            loaded_case["connections"],
        )
        restored_case = apply_graph_draft(
            {
                "schema_version": 3,
                "name": "starter",
                "units": [],
                "connections": [],
            },
            loaded_draft,
        )

        self.assertEqual(inserted_id, "product-pressure-valve")
        self.assertEqual(restored_case["units"], units)
        self.assertEqual(restored_case["connections"], connections)
        self.assertEqual(restored_case["name"], "starter")


class GraphHistoryTest(unittest.TestCase):
    """Validate bounded, branch-aware graph edit history."""

    def setUp(self):
        self.units = [
            create_inline_unit_spec(
                "cooler",
                "Product Cooler",
                set(),
            )
        ]
        self.connections = [
            {
                "id": "feed-to-cooler",
                "type": "material",
                "source": {
                    "kind": "inlet",
                    "id": "feed",
                    "port": "out",
                },
                "target": {
                    "kind": "unit",
                    "id": "product-cooler",
                    "port": "in",
                },
            }
        ]

    def _history_with_insert(self):
        history = create_graph_history(self.units, self.connections)
        units, connections, _ = insert_inline_unit_on_connection(
            self.units,
            self.connections,
            "feed-to-cooler",
            "valve",
            "Product Valve",
        )
        return record_graph_history(history, units, connections)

    def test_undo_and_redo_return_isolated_graph_revisions(self):
        history = self._history_with_insert()
        self.assertEqual(
            graph_history_status(history),
            {
                "position": 2,
                "total": 2,
                "can_undo": True,
                "can_redo": False,
            },
        )

        undone, starter = undo_graph_history(history)
        self.assertEqual(starter["units"], self.units)
        self.assertTrue(graph_history_status(undone)["can_redo"])
        redone, inserted = redo_graph_history(undone)
        self.assertEqual(len(inserted["units"]), 2)
        self.assertEqual(redone, history)

        starter["units"][0]["name"] = "changed"
        redone["entries"][0]["units"][0]["name"] = "also changed"
        self.assertEqual(self.units[0]["name"], "Product Cooler")
        self.assertEqual(
            history["entries"][0]["units"][0]["name"],
            "Product Cooler",
        )

    def test_record_after_undo_discards_abandoned_redo_branch(self):
        inserted_history = self._history_with_insert()
        undone, _ = undo_graph_history(inserted_history)
        renamed_units = rename_inline_unit(
            self.units,
            "product-cooler",
            "Export Cooler",
        )
        branched = record_graph_history(
            undone,
            renamed_units,
            self.connections,
        )

        self.assertEqual(graph_history_status(branched)["total"], 2)
        with self.assertRaisesRegex(ValueError, "no later revision"):
            redo_graph_history(branched)
        self.assertEqual(
            branched["entries"][1]["units"][0]["name"],
            "Export Cooler",
        )

    def test_duplicate_revisions_are_ignored_and_history_is_bounded(self):
        history = create_graph_history(self.units, self.connections)
        unchanged = record_graph_history(
            history,
            self.units,
            self.connections,
        )
        self.assertEqual(unchanged, history)

        for suffix in range(1, 5):
            renamed_units = rename_inline_unit(
                self.units,
                "product-cooler",
                f"Product Cooler {suffix}",
            )
            history = record_graph_history(
                history,
                renamed_units,
                self.connections,
                max_entries=3,
            )
        self.assertEqual(graph_history_status(history)["total"], 3)
        self.assertEqual(
            history["entries"][0]["units"][0]["name"],
            "Product Cooler 2",
        )
        self.assertEqual(
            history["entries"][2]["units"][0]["name"],
            "Product Cooler 4",
        )

    def test_invalid_histories_and_limits_fail_explicitly(self):
        history = create_graph_history(self.units, self.connections)
        invalid_cases = (
            (None, "must be an object"),
            ({**history, "schema_version": 2}, "Unsupported graph history"),
            ({**history, "entries": []}, "non-empty array"),
            ({**history, "cursor": True}, "must be an integer"),
            ({**history, "cursor": 2}, "outside the entry range"),
        )
        for invalid_history, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    graph_history_status(invalid_history)

        for limit in (True, 1, 2.5):
            with self.subTest(limit=limit):
                with self.assertRaisesRegex(ValueError, "at least 2"):
                    record_graph_history(
                        history,
                        self.units,
                        self.connections,
                        max_entries=limit,
                    )
        with self.assertRaisesRegex(ValueError, "no earlier revision"):
            undo_graph_history(history)

    def test_history_undo_and_redo_include_inlet_conditions(self):
        inlets = [
            {
                "id": "feed-a",
                "name": "Feed A",
                "fluid_package_id": "shared",
                "composition": {"methane": 1.0},
                "composition_basis": "mole_fraction",
                "temperature_C": 20.0,
                "pressure_bara": 50.0,
                "total_flow": 60_000.0,
                "flow_unit": "kg/hr",
            }
        ]
        history = create_graph_history(
            self.units,
            self.connections,
            inlets,
        )
        updated_inlets = update_inlet_conditions(
            inlets,
            "feed-a",
            {"temperature_C": 30.0},
        )
        history = record_graph_history(
            history,
            self.units,
            self.connections,
            updated_inlets,
        )

        undone, original = undo_graph_history(history)
        self.assertEqual(original["inlets"][0]["temperature_C"], 20.0)
        redone, edited = redo_graph_history(undone)
        self.assertEqual(edited["inlets"][0]["temperature_C"], 30.0)
        self.assertEqual(redone, history)


class GraphPortConnectionTest(unittest.TestCase):
    """Validate explicit material and energy port connection editing."""

    def setUp(self):
        self.inlets = [
            {"id": "feed-a", "name": "Feed A"},
            {"id": "feed-b", "name": "Feed B"},
        ]
        self.units = [
            {
                "id": "mixer",
                "name": "Feed mixer",
                "type": "mixer",
                "ports": {
                    "material_in": ["in_0", "in_1"],
                    "material_out": ["out"],
                },
            },
            {
                "id": "heater",
                "name": "Feed heater",
                "type": "heater",
                "ports": {
                    "material_in": ["in"],
                    "material_out": ["out"],
                    "energy_in": ["duty"],
                },
            },
            {
                "id": "splitter",
                "name": "Product splitter",
                "type": "splitter",
                "ports": {
                    "material_in": ["in"],
                    "material_out": ["out_0", "out_1"],
                },
            },
            {
                "id": "utility",
                "name": "Heating utility",
                "type": "utility",
                "ports": {
                    "energy_out": ["duty"],
                },
            },
        ]
        self.connections = [
            {
                "id": "feed-a-mixer",
                "type": "material",
                "source": {
                    "kind": "inlet",
                    "id": "feed-a",
                    "port": "out",
                },
                "target": {
                    "kind": "unit",
                    "id": "mixer",
                    "port": "in_0",
                },
            },
            {
                "id": "mixer-heater",
                "type": "material",
                "source": {
                    "kind": "unit",
                    "id": "mixer",
                    "port": "out",
                },
                "target": {
                    "kind": "unit",
                    "id": "heater",
                    "port": "in",
                },
            },
            {
                "id": "utility-heater",
                "type": "energy",
                "source": {
                    "kind": "unit",
                    "id": "utility",
                    "port": "duty",
                },
                "target": {
                    "kind": "unit",
                    "id": "heater",
                    "port": "duty",
                },
            },
        ]

    def test_port_rows_report_declared_ports_and_occupancy(self):
        material_sources = graph_port_rows(
            self.inlets,
            self.units,
            self.connections,
            "material",
            "source",
        )
        material_targets = graph_port_rows(
            self.inlets,
            self.units,
            self.connections,
            "material",
            "target",
        )
        source_occupancy = {
            (row["id"], row["port"]): row["connected"]
            for row in material_sources
        }
        target_occupancy = {
            (row["id"], row["port"]): row["connected"]
            for row in material_targets
        }

        self.assertTrue(source_occupancy[("feed-a", "out")])
        self.assertFalse(source_occupancy[("feed-b", "out")])
        self.assertTrue(source_occupancy[("mixer", "out")])
        self.assertFalse(source_occupancy[("splitter", "out_1")])
        self.assertTrue(target_occupancy[("mixer", "in_0")])
        self.assertFalse(target_occupancy[("mixer", "in_1")])
        self.assertTrue(target_occupancy[("heater", "in")])
        self.assertFalse(target_occupancy[("splitter", "in")])

        available_sources = graph_port_rows(
            self.inlets,
            self.units,
            self.connections,
            "energy",
            "source",
            available_only=True,
        )
        self.assertEqual(available_sources, [])

    def test_connects_second_inlet_without_mutating_inputs(self):
        source = {
            "kind": "inlet",
            "id": "feed-b",
            "port": "out",
        }
        target = {
            "kind": "unit",
            "id": "mixer",
            "port": "in_1",
        }
        updated, connection_id = connect_graph_ports(
            self.inlets,
            self.units,
            self.connections,
            "material",
            source,
            target,
        )

        self.assertEqual(
            connection_id,
            "material-feed-b-out-to-mixer-in-1",
        )
        self.assertEqual(updated[-1]["source"], source)
        self.assertEqual(updated[-1]["target"], target)
        self.assertEqual(len(self.connections), 3)
        source["id"] = "changed"
        self.assertEqual(updated[-1]["source"]["id"], "feed-b")

    def test_connects_splitter_branch_and_energy_link(self):
        without_energy = disconnect_graph_connection(
            self.inlets,
            self.units,
            self.connections,
            "utility-heater",
        )
        with_splitter, material_id = connect_graph_ports(
            self.inlets,
            self.units,
            without_energy,
            "material",
            {"kind": "unit", "id": "heater", "port": "out"},
            {"kind": "unit", "id": "splitter", "port": "in"},
        )
        updated, energy_id = connect_graph_ports(
            self.inlets,
            self.units,
            with_splitter,
            "energy",
            {"kind": "unit", "id": "utility", "port": "duty"},
            {"kind": "unit", "id": "heater", "port": "duty"},
        )

        self.assertEqual(
            material_id,
            "material-heater-out-to-splitter-in",
        )
        self.assertEqual(
            energy_id,
            "energy-utility-duty-to-heater-duty",
        )
        self.assertEqual(updated[-1]["type"], "energy")

    def test_connects_separator_liquid_to_new_standalone_pump(self):
        units = [
            {
                "id": "separator",
                "name": "Inlet separator",
                "type": "separator",
                "ports": {
                    "material_in": ["in"],
                    "material_out": ["gas", "liquid"],
                },
            }
        ]
        units, pump_id = add_catalog_unit(
            units,
            "pump",
            "Condensate pump",
            {"feed"},
        )
        connections = [
            {
                "id": "feed-to-separator",
                "type": "material",
                "source": {"kind": "inlet", "id": "feed", "port": "out"},
                "target": {"kind": "unit", "id": "separator", "port": "in"},
            }
        ]

        updated, connection_id = connect_graph_ports(
            [{"id": "feed", "name": "Feed"}],
            units,
            connections,
            "material",
            {"kind": "unit", "id": "separator", "port": "liquid"},
            {"kind": "unit", "id": pump_id, "port": "in"},
        )

        self.assertEqual(
            connection_id,
            "material-separator-liquid-to-condensate-pump-in",
        )
        self.assertEqual(updated[-1]["source"]["port"], "liquid")
        self.assertEqual(updated[-1]["target"]["id"], pump_id)
        available_sources = graph_port_rows(
            [{"id": "feed", "name": "Feed"}],
            units,
            updated,
            "material",
            "source",
            available_only=True,
        )
        self.assertNotIn(
            ("separator", "liquid"),
            {(row["id"], row["port"]) for row in available_sources},
        )

    def test_extends_separator_liquid_path_through_pump_and_heater(self):
        inlets = [{"id": "feed", "name": "Well fluid"}]
        units = [
            {
                "id": "separator",
                "name": "Inlet separator",
                "type": "separator",
                "ports": {
                    "material_in": ["in"],
                    "material_out": ["gas", "liquid"],
                },
            }
        ]
        connections = [
            {
                "id": "feed-to-separator",
                "type": "material",
                "source": {"kind": "inlet", "id": "feed", "port": "out"},
                "target": {"kind": "unit", "id": "separator", "port": "in"},
            }
        ]

        units, connections, pump_id, pump_connection_id = (
            extend_material_path(
                inlets,
                units,
                connections,
                {"kind": "unit", "id": "separator", "port": "liquid"},
                "pump",
                "Condensate pump",
            )
        )
        units, connections, heater_id, heater_connection_id = (
            extend_material_path(
                inlets,
                units,
                connections,
                {"kind": "unit", "id": pump_id, "port": "out"},
                "heater",
                "Condensate heater",
            )
        )

        self.assertEqual(pump_id, "condensate-pump")
        self.assertEqual(heater_id, "condensate-heater")
        self.assertEqual(
            pump_connection_id,
            "material-separator-liquid-to-condensate-pump-in",
        )
        self.assertEqual(
            heater_connection_id,
            "material-condensate-pump-out-to-condensate-heater-in",
        )
        self.assertEqual(
            [unit["id"] for unit in units],
            ["separator", "condensate-pump", "condensate-heater"],
        )
        self.assertEqual(len(connections), 3)
        self.assertEqual(
            graph_port_rows(
                inlets,
                units,
                connections,
                "material",
                "source",
                available_only=True,
            )[-1]["endpoint"],
            {
                "kind": "unit",
                "id": "condensate-heater",
                "port": "out",
            },
        )

    def test_path_extension_failure_does_not_mutate_graph(self):
        original_units = copy.deepcopy(self.units)
        original_connections = copy.deepcopy(self.connections)

        with self.assertRaisesRegex(ValueError, "already has a connection"):
            extend_material_path(
                self.inlets,
                self.units,
                self.connections,
                {"kind": "inlet", "id": "feed-a", "port": "out"},
                "pump",
                "Extra pump",
            )

        self.assertEqual(self.units, original_units)
        self.assertEqual(self.connections, original_connections)

    def test_rejects_occupied_undeclared_and_self_connections(self):
        invalid_cases = (
            (
                "material",
                {"kind": "inlet", "id": "feed-a", "port": "out"},
                {"kind": "unit", "id": "mixer", "port": "in_1"},
                "output port feed-a:out already has",
            ),
            (
                "material",
                {"kind": "inlet", "id": "feed-b", "port": "out"},
                {"kind": "unit", "id": "heater", "port": "in"},
                "input port heater:in already has",
            ),
            (
                "material",
                {"kind": "unit", "id": "heater", "port": "missing"},
                {"kind": "unit", "id": "splitter", "port": "in"},
                "not a declared output",
            ),
            (
                "material",
                {"kind": "unit", "id": "splitter", "port": "out_0"},
                {"kind": "unit", "id": "splitter", "port": "in"},
                "cannot connect a node to itself",
            ),
        )
        for connection_type, source, target, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    connect_graph_ports(
                        self.inlets,
                        self.units,
                        self.connections,
                        connection_type,
                        source,
                        target,
                    )
        self.assertEqual(len(self.connections), 3)

    def test_disconnect_preserves_order_and_original_graph(self):
        updated = disconnect_graph_connection(
            self.inlets,
            self.units,
            self.connections,
            "mixer-heater",
        )

        self.assertEqual(
            [connection["id"] for connection in updated],
            ["feed-a-mixer", "utility-heater"],
        )
        self.assertEqual(
            [connection["id"] for connection in self.connections],
            ["feed-a-mixer", "mixer-heater", "utility-heater"],
        )
        with self.assertRaisesRegex(ValueError, "Unknown graph connection"):
            disconnect_graph_connection(
                self.inlets,
                self.units,
                self.connections,
                "missing",
            )

    def test_connection_rows_cover_material_and_energy_paths(self):
        rows = graph_connection_rows(
            self.inlets,
            self.units,
            self.connections,
        )

        self.assertEqual(
            [row["type"] for row in rows],
            ["material", "material", "energy"],
        )
        self.assertEqual(
            rows[-1]["label"],
            "ENERGY · utility:duty → heater:duty",
        )

    def test_invalid_port_inventory_fails_explicitly(self):
        invalid_cases = (
            (
                [{**self.inlets[0]}, {**self.inlets[0]}],
                self.units,
                self.connections,
                "Graph object id 'feed-a' is duplicated",
            ),
            (
                self.inlets,
                [
                    *self.units,
                    {
                        "id": "broken",
                        "ports": {"material_in": ["in", "in"]},
                    },
                ],
                self.connections,
                "target port 'broken:in' is duplicated",
            ),
            (
                self.inlets,
                self.units,
                [
                    *self.connections,
                    {
                        **self.connections[0],
                        "id": "feed-a-again",
                        "target": {
                            "kind": "unit",
                            "id": "mixer",
                            "port": "in_1",
                        },
                    },
                ],
                "output port feed-a:out already has",
            ),
        )
        for inlets, units, connections, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    graph_connection_rows(inlets, units, connections)

        for connection_type, direction, available_only, message in (
            ("signal", "source", False, "material or energy"),
            ("material", "sideways", False, "source or target"),
            ("material", "source", 1, "must be a boolean"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    graph_port_rows(
                        self.inlets,
                        self.units,
                        self.connections,
                        connection_type,
                        direction,
                        available_only,
                    )


class GraphDraftDiagramTest(unittest.TestCase):
    """Validate deterministic draft diagrams for arbitrary process graphs."""

    def setUp(self):
        self.inlets = [
            {"id": "feed-a", "name": "Feed A"},
            {"id": "feed-b", "name": "Feed B"},
        ]
        self.units = [
            {
                "id": "mixer",
                "name": "Feed mixer",
                "type": "mixer",
                "ports": {
                    "material_in": ["in_0", "in_1"],
                    "material_out": ["out"],
                },
            },
            {
                "id": "heater",
                "name": "Feed heater",
                "type": "heater",
                "ports": {
                    "material_in": ["in"],
                    "material_out": ["out"],
                    "energy_in": ["duty"],
                },
            },
            {
                "id": "utility",
                "name": "Heating utility",
                "type": "utility",
                "ports": {
                    "energy_out": ["duty"],
                },
            },
            {
                "id": "separator",
                "name": "Product separator",
                "type": "separator",
                "ports": {
                    "material_in": ["in"],
                    "material_out": ["gas", "liquid"],
                },
            },
        ]
        self.connections = [
            {
                "id": "feed-a-mixer",
                "type": "material",
                "source": {
                    "kind": "inlet",
                    "id": "feed-a",
                    "port": "out",
                },
                "target": {
                    "kind": "unit",
                    "id": "mixer",
                    "port": "in_0",
                },
            },
            {
                "id": "feed-b-mixer",
                "type": "material",
                "source": {
                    "kind": "inlet",
                    "id": "feed-b",
                    "port": "out",
                },
                "target": {
                    "kind": "unit",
                    "id": "mixer",
                    "port": "in_1",
                },
            },
            {
                "id": "mixer-heater",
                "type": "material",
                "source": {
                    "kind": "unit",
                    "id": "mixer",
                    "port": "out",
                },
                "target": {
                    "kind": "unit",
                    "id": "heater",
                    "port": "in",
                },
            },
            {
                "id": "utility-heater",
                "type": "energy",
                "source": {
                    "kind": "unit",
                    "id": "utility",
                    "port": "duty",
                },
                "target": {
                    "kind": "unit",
                    "id": "heater",
                    "port": "duty",
                },
            },
            {
                "id": "heater-separator",
                "type": "material",
                "source": {
                    "kind": "unit",
                    "id": "heater",
                    "port": "out",
                },
                "target": {
                    "kind": "unit",
                    "id": "separator",
                    "port": "in",
                },
            },
        ]

    def test_layout_shows_multi_inlet_energy_and_product_boundaries(self):
        dot = build_graph_draft_dot(
            self.inlets,
            self.units,
            self.connections,
        )

        self.assertIn('rankdir="LR"', dot)
        self.assertIn("Feed A\\nINLET", dot)
        self.assertIn("Feed mixer\\nMIXER", dot)
        self.assertIn("separator:gas\\nPRODUCT", dot)
        self.assertIn("separator:liquid\\nPRODUCT", dot)
        self.assertEqual(dot.count(" -> "), 7)
        self.assertEqual(dot.count('style="dashed"'), 1)
        self.assertIn("duty → duty", dot)

    def test_layout_is_connection_order_independent_and_input_safe(self):
        forward = build_graph_draft_dot(
            self.inlets,
            self.units,
            self.connections,
        )
        reverse = build_graph_draft_dot(
            self.inlets,
            self.units,
            list(reversed(self.connections)),
        )

        self.assertEqual(forward, reverse)
        self.assertEqual(self.connections[0]["source"]["id"], "feed-a")

    def test_layout_quotes_user_labels_and_uses_internal_node_ids(self):
        inlets = [
            {
                "id": 'feed"; unsafe',
                "name": 'Feed "quoted"',
            }
        ]
        connection = {
            **self.connections[0],
            "source": {
                "kind": "inlet",
                "id": 'feed"; unsafe',
                "port": "out",
            },
        }
        dot = build_graph_draft_dot(inlets, self.units, [connection])

        self.assertIn(r'Feed \"quoted\"\nINLET', dot)
        self.assertNotIn('feed"; unsafe ->', dot)
        self.assertIn("inlet_0 -> unit_0", dot)

    def test_invalid_preview_graphs_fail_explicitly(self):
        invalid_cases = (
            (
                None,
                self.units,
                self.connections,
                "Graph preview inlets must be an array",
            ),
            (
                "not-an-array",
                self.units,
                self.connections,
                "inlets must be an array",
            ),
            (
                [self.inlets[0], self.inlets[0]],
                self.units,
                self.connections,
                "inlet id 'feed-a' is duplicated",
            ),
            (
                [{"id": "mixer"}],
                self.units,
                self.connections,
                "both an inlet and a unit",
            ),
            (
                self.inlets,
                self.units,
                [
                    {
                        **self.connections[0],
                        "source": {
                            "kind": "inlet",
                            "id": "missing",
                            "port": "out",
                        },
                    }
                ],
                "unknown source inlet 'missing'",
            ),
            (
                self.inlets,
                self.units,
                [
                    {
                        **self.connections[0],
                        "target": {
                            "kind": "unit",
                            "id": "mixer",
                            "port": "missing",
                        },
                    }
                ],
                "uses undeclared material_in port 'missing'",
            ),
        )
        for inlets, units, connections, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build_graph_draft_dot(inlets, units, connections)


if __name__ == "__main__":
    unittest.main()
