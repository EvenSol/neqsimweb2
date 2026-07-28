"""Native regressions for named material connections in generic graphs."""

from __future__ import annotations

import unittest

from process_chat.process_builder import ProcessBuilder
from process_chat.process_model import NeqSimProcessModel


class MaterialStreamIdentityTest(unittest.TestCase):
    """Require graph stream names to survive native execution."""

    @staticmethod
    def _inlet(inlet_id: str, name: str, flow_kg_hr: float) -> dict:
        return {
            "inlet_id": inlet_id,
            "name": name,
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
                "total_flow": flow_kg_hr,
                "flow_unit": "kg/hr",
            },
        }

    def test_named_two_feed_connections_are_solved_internal_streams(self):
        inlet_specs = [
            self._inlet("well-a", "Well A feed", 12_000.0),
            self._inlet("well-b", "Well B feed", 8_000.0),
        ]
        graph_spec = {
            "name": "Named two-feed mixer",
            "units": [
                {
                    "id": "inlet-mixer",
                    "name": "Inlet mixer",
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
                    "id": "well-a-to-mixer",
                    "name": "Well A to inlet mixer",
                    "type": "material",
                    "source": {
                        "kind": "inlet",
                        "id": "well-a",
                        "port": "out",
                    },
                    "target": {
                        "kind": "unit",
                        "id": "inlet-mixer",
                        "port": "in_0",
                    },
                },
                {
                    "id": "well-b-to-mixer",
                    "name": "Well B to inlet mixer",
                    "type": "material",
                    "source": {
                        "kind": "inlet",
                        "id": "well-b",
                        "port": "out",
                    },
                    "target": {
                        "kind": "unit",
                        "id": "inlet-mixer",
                        "port": "in_1",
                    },
                },
            ],
        }

        builder = ProcessBuilder()
        model = builder.build_acyclic_graph(
            graph_spec,
            inlet_specs,
            ["well-a", "well-b", "inlet-mixer"],
        )
        stream_names = {stream.name for stream in model.list_streams()}
        self.assertIn("Well A to inlet mixer", stream_names)
        self.assertIn("Well B to inlet mixer", stream_names)
        self.assertIsNotNone(model.get_stream("Well A to inlet mixer"))
        self.assertIsNotNone(model.get_stream("Well B to inlet mixer"))
        feed_records = [
            NeqSimProcessModel._material_boundary_record(
                model.get_stream(feed_name),
                "feed",
                feed_name,
            )
            for feed_name in ("Well A feed", "Well B feed")
        ]
        product_record = NeqSimProcessModel._material_boundary_record(
            model.get_stream("Inlet mixer [out] product"),
            "product",
            "Inlet mixer [out] product",
        )
        feed_mass_flow = sum(
            record["mass_flow_kg_hr"] for record in feed_records
        )
        self.assertAlmostEqual(
            feed_mass_flow,
            20_000.0,
            delta=0.02,
        )
        self.assertAlmostEqual(
            product_record["mass_flow_kg_hr"],
            20_000.0,
            delta=0.02,
        )
        mass_residual_pct = (
            abs(product_record["mass_flow_kg_hr"] - feed_mass_flow)
            / feed_mass_flow
            * 100.0
        )
        self.assertLess(mass_residual_pct, 1.0e-6)

        product_components = product_record[
            "component_molar_flows_mol_sec"
        ]
        self.assertIsNotNone(product_components)
        for component_name, product_flow in product_components.items():
            feed_flow = sum(
                record["component_molar_flows_mol_sec"][component_name]
                for record in feed_records
            )
            component_residual_pct = (
                abs(product_flow - feed_flow)
                / max(feed_flow, product_flow, 1.0e-12)
                * 100.0
            )
            self.assertLess(component_residual_pct, 1.0e-6)

        feed_enthalpy_kW = sum(
            record["enthalpy_flow_kW"] for record in feed_records
        )
        product_enthalpy_kW = product_record["enthalpy_flow_kW"]
        energy_residual_pct = (
            abs(product_enthalpy_kW - feed_enthalpy_kW)
            / max(
                abs(feed_enthalpy_kW),
                abs(product_enthalpy_kW),
                1.0,
            )
            * 100.0
        )
        self.assertLess(energy_residual_pct, 1.0e-6)

    def test_rejects_duplicate_or_reserved_stream_names_before_native_build(self):
        inlet_specs = [
            self._inlet("well-a", "Well A feed", 12_000.0),
            self._inlet("well-b", "Well B feed", 8_000.0),
        ]
        base_connection = {
            "type": "material",
            "source": {
                "kind": "inlet",
                "id": "well-a",
                "port": "out",
            },
            "target": {
                "kind": "unit",
                "id": "inlet-mixer",
                "port": "in_0",
            },
        }
        unit = {
            "id": "inlet-mixer",
            "name": "Inlet mixer",
            "type": "mixer",
            "ports": {
                "material_in": ["in_0", "in_1"],
                "material_out": ["out"],
            },
            "params": {},
        }
        invalid_names = (
            ("duplicate", "Duplicate stream", "duplicate STREAM"),
            ("reserved", "Well A feed", "Other stream"),
            (
                "product",
                "Inlet mixer [out] product",
                "Other stream",
            ),
        )
        for label, first_name, second_name in invalid_names:
            with self.subTest(label=label):
                connections = [
                    {
                        **base_connection,
                        "id": "well-a-to-mixer",
                        "name": first_name,
                    },
                    {
                        **base_connection,
                        "id": "well-b-to-mixer",
                        "name": second_name,
                        "source": {
                            "kind": "inlet",
                            "id": "well-b",
                            "port": "out",
                        },
                        "target": {
                            "kind": "unit",
                            "id": "inlet-mixer",
                            "port": "in_1",
                        },
                    },
                ]
                with self.assertRaisesRegex(
                    ValueError,
                    "duplicated|conflicts",
                ):
                    ProcessBuilder().build_acyclic_graph(
                        {
                            "name": "Invalid stream names",
                            "units": [unit],
                            "connections": connections,
                        },
                        inlet_specs,
                        ["well-a", "well-b", "inlet-mixer"],
                    )

    def test_rejects_duplicate_source_port_before_native_build(self):
        inlet_specs = [
            self._inlet("well-a", "Well A feed", 12_000.0),
        ]
        graph_spec = {
            "name": "Invalid implicit branch",
            "units": [
                {
                    "id": "inlet-mixer",
                    "name": "Inlet mixer",
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
                    "id": "well-a-to-mixer-0",
                    "name": "Well A branch 0",
                    "type": "material",
                    "source": {
                        "kind": "inlet",
                        "id": "well-a",
                        "port": "out",
                    },
                    "target": {
                        "kind": "unit",
                        "id": "inlet-mixer",
                        "port": "in_0",
                    },
                },
                {
                    "id": "well-a-to-mixer-1",
                    "name": "Well A branch 1",
                    "type": "material",
                    "source": {
                        "kind": "inlet",
                        "id": "well-a",
                        "port": "out",
                    },
                    "target": {
                        "kind": "unit",
                        "id": "inlet-mixer",
                        "port": "in_1",
                    },
                },
            ],
        }

        with self.assertRaisesRegex(
            ValueError,
            "output port well-a:out already has a connection",
        ):
            ProcessBuilder().build_acyclic_graph(
                graph_spec,
                inlet_specs,
                ["well-a", "inlet-mixer"],
            )


if __name__ == "__main__":
    unittest.main()
