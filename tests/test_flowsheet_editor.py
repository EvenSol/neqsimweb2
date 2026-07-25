"""Focused regression tests for pure flowsheet-editor schema helpers."""

from __future__ import annotations

import json
import unittest

from process_chat.flowsheet_editor import (
    apply_graph_draft,
    build_graph_draft_dot,
    connect_graph_ports,
    create_graph_draft,
    create_graph_history,
    create_inline_unit_spec,
    disconnect_graph_connection,
    graph_connection_rows,
    graph_history_status,
    graph_port_rows,
    inline_unit_catalog,
    inline_unit_catalog_rows,
    insert_inline_unit_on_connection,
    material_connection_rows,
    record_graph_history,
    redo_graph_history,
    remove_inline_unit,
    rename_inline_unit,
    undo_graph_history,
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
            ],
        )
        self.assertTrue(all(row["Category"] for row in rows))
        self.assertTrue(all(row["Description"] for row in rows))

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
            (branch, "requires exactly one incoming"),
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
