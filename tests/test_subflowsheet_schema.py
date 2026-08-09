"""Focused schema tests for explicit Studio subflowsheet boundaries."""

from __future__ import annotations

import copy
import unittest

from process_chat.subflowsheet_schema import (
    subflowsheet_membership,
    validate_subflowsheets,
)


class SubflowsheetSchemaTest(unittest.TestCase):
    def setUp(self):
        self.units = [
            {
                "id": "compressor",
                "name": "Compression",
                "type": "compressor",
                "ports": {
                    "material_in": ["in"],
                    "material_out": ["out"],
                    "energy_in": [],
                    "energy_out": [],
                },
            },
            {
                "id": "cooler",
                "name": "Aftercooler",
                "type": "cooler",
                "ports": {
                    "material_in": ["in"],
                    "material_out": ["out"],
                    "energy_in": [],
                    "energy_out": [],
                },
            },
        ]
        self.connections = [
            {
                "id": "feed-to-compressor",
                "type": "material",
                "source": {"kind": "inlet", "id": "feed", "port": "out"},
                "target": {"kind": "unit", "id": "compressor", "port": "in"},
            },
            {
                "id": "compressor-to-cooler",
                "type": "material",
                "source": {"kind": "unit", "id": "compressor", "port": "out"},
                "target": {"kind": "unit", "id": "cooler", "port": "in"},
            },
        ]
        self.subflowsheets = [
            {
                "id": "compression-train",
                "name": "Compression train",
                "unit_ids": ["compressor", "cooler"],
                "boundary_ports": [
                    {
                        "id": "feed",
                        "name": "Train feed",
                        "type": "material",
                        "direction": "inlet",
                        "endpoint": {
                            "kind": "unit",
                            "id": "compressor",
                            "port": "in",
                        },
                    },
                    {
                        "id": "product",
                        "name": "Train product",
                        "type": "material",
                        "direction": "outlet",
                        "endpoint": {
                            "kind": "unit",
                            "id": "cooler",
                            "port": "out",
                        },
                    },
                ],
            }
        ]

    def test_accepts_crossing_feed_and_terminal_product_boundaries(self):
        validate_subflowsheets(
            self.subflowsheets,
            self.units,
            self.connections,
        )
        self.assertEqual(
            subflowsheet_membership(self.subflowsheets),
            {
                "compressor": "compression-train",
                "cooler": "compression-train",
            },
        )

    def test_requires_every_crossing_port_to_be_declared(self):
        candidate = copy.deepcopy(self.subflowsheets)
        candidate[0]["boundary_ports"] = candidate[0]["boundary_ports"][1:]

        with self.assertRaisesRegex(ValueError, "must declare.*inlet"):
            validate_subflowsheets(candidate, self.units, self.connections)

    def test_rejects_boundary_on_non_member_unit(self):
        candidate = copy.deepcopy(self.subflowsheets)
        candidate[0]["unit_ids"] = ["compressor"]

        with self.assertRaisesRegex(ValueError, "non-member unit 'cooler'"):
            validate_subflowsheets(candidate, self.units, self.connections)

    def test_rejects_unit_membership_in_multiple_groups(self):
        candidate = copy.deepcopy(self.subflowsheets)
        duplicate = copy.deepcopy(candidate[0])
        duplicate["id"] = "second-train"
        duplicate["name"] = "Second train"
        candidate.append(duplicate)

        with self.assertRaisesRegex(ValueError, "belongs to both"):
            validate_subflowsheets(candidate, self.units, self.connections)

    def test_rejects_unused_inlet_boundary(self):
        candidate = copy.deepcopy(self.subflowsheets)
        candidate[0]["boundary_ports"][0]["endpoint"]["id"] = "cooler"

        with self.assertRaisesRegex(ValueError, "not an active graph boundary"):
            validate_subflowsheets(candidate, self.units, self.connections)


if __name__ == "__main__":
    unittest.main()
