"""Focused regression tests for pure flowsheet-editor schema helpers."""

from __future__ import annotations

import unittest

from process_chat.flowsheet_editor import (
    create_inline_unit_spec,
    inline_unit_catalog,
    inline_unit_catalog_rows,
    insert_inline_unit_on_connection,
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


if __name__ == "__main__":
    unittest.main()
