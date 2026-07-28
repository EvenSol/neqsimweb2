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

    def test_native_alias_dedup_is_scoped_to_its_source_stream(self):
        from neqsim import jneqsim

        shared_fluid = ProcessBuilder().create_fluid_from_spec(
            self._inlet("shared", "Shared", 100.0)["fluid_spec"]
        )
        StreamClass = jneqsim.process.equipment.stream.Stream
        HeaterClass = jneqsim.process.equipment.heatexchanger.Heater
        direct_feed = StreamClass("direct feed", shared_fluid)
        aliased_feed = StreamClass("aliased feed", shared_fluid)
        named_alias = StreamClass("named feed connection", aliased_feed)
        heater_a = HeaterClass("heater a", direct_feed)
        heater_b = HeaterClass("heater b", named_alias)

        feeds, _ = NeqSimProcessModel._connectivity_material_boundaries(
            [
                direct_feed,
                aliased_feed,
                named_alias,
                heater_a,
                heater_b,
            ]
        )

        self.assertEqual(
            [str(stream.getName()) for stream in feeds],
            ["direct feed", "aliased feed"],
        )

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

    def test_rejects_aliased_source_ports_before_native_build(self):
        inlet_specs = [
            self._inlet("well-a", "Well A feed", 12_000.0),
        ]
        units = [
            {
                "id": "heater",
                "name": "Feed heater",
                "type": "heater",
                "ports": {
                    "material_in": ["in"],
                    "material_out": ["out"],
                },
                "params": {},
            },
            {
                "id": "cooler-a",
                "name": "Branch cooler A",
                "type": "cooler",
                "ports": {
                    "material_in": ["in"],
                    "material_out": ["out"],
                },
                "params": {},
            },
            {
                "id": "cooler-b",
                "name": "Branch cooler B",
                "type": "cooler",
                "ports": {
                    "material_in": ["in"],
                    "material_out": ["out"],
                },
                "params": {},
            },
        ]
        connections = [
            {
                "id": "well-a-to-heater",
                "name": "Well A to heater",
                "type": "material",
                "source": {
                    "kind": "inlet",
                    "id": "well-a",
                    "port": "out",
                },
                "target": {
                    "kind": "unit",
                    "id": "heater",
                    "port": "in",
                },
            },
            {
                "id": "heater-to-cooler-a",
                "name": "Heater branch A",
                "type": "material",
                "source": {
                    "kind": "unit",
                    "id": "heater",
                    "port": "out",
                },
                "target": {
                    "kind": "unit",
                    "id": "cooler-a",
                    "port": "in",
                },
            },
            {
                "id": "heater-to-cooler-b",
                "name": "Heater branch B",
                "type": "material",
                "source": {
                    "kind": "unit",
                    "id": "heater",
                    "port": "main",
                },
                "target": {
                    "kind": "unit",
                    "id": "cooler-b",
                    "port": "in",
                },
            },
        ]

        with self.assertRaisesRegex(
            ValueError,
            "output port heater:out already has a connection",
        ):
            ProcessBuilder().build_acyclic_graph(
                {
                    "name": "Invalid aliased branch",
                    "units": units,
                    "connections": connections,
                },
                inlet_specs,
                ["well-a", "heater", "cooler-a", "cooler-b"],
            )

    def test_rejects_generic_and_gas_separator_output_aliases(self):
        inlet_specs = [
            self._inlet("well-a", "Well A feed", 12_000.0),
        ]
        graph_spec = {
            "name": "Invalid separator gas branch",
            "units": [
                {
                    "id": "separator",
                    "name": "Inlet separator",
                    "type": "separator",
                    "ports": {
                        "material_in": ["in"],
                        "material_out": ["out", "gas"],
                    },
                    "params": {},
                }
            ],
            "connections": [
                {
                    "id": "well-a-to-separator",
                    "name": "Well A to separator",
                    "type": "material",
                    "source": {
                        "kind": "inlet",
                        "id": "well-a",
                        "port": "out",
                    },
                    "target": {
                        "kind": "unit",
                        "id": "separator",
                        "port": "in",
                    },
                }
            ],
        }

        with self.assertRaisesRegex(
            ValueError,
            "material output ports alias the same native outlet",
        ):
            ProcessBuilder().build_acyclic_graph(
                graph_spec,
                inlet_specs,
                ["well-a", "separator"],
            )

    def test_reserves_raw_terminal_alias_name_before_native_build(self):
        inlet_specs = [
            self._inlet("well-a", "Well A feed", 12_000.0),
        ]
        graph_spec = {
            "name": "Conflicting aliased terminal name",
            "units": [
                {
                    "id": "separator",
                    "name": "Inlet separator",
                    "type": "separator",
                    "ports": {
                        "material_in": ["in"],
                        "material_out": ["vapor"],
                    },
                    "params": {},
                }
            ],
            "connections": [
                {
                    "id": "well-a-to-separator",
                    "name": "INLET SEPARATOR [VAPOR] PRODUCT",
                    "type": "material",
                    "source": {
                        "kind": "inlet",
                        "id": "well-a",
                        "port": "out",
                    },
                    "target": {
                        "kind": "unit",
                        "id": "separator",
                        "port": "in",
                    },
                }
            ],
        }

        with self.assertRaisesRegex(
            ValueError,
            "conflicts with a terminal product boundary",
        ):
            ProcessBuilder().build_acyclic_graph(
                graph_spec,
                inlet_specs,
                ["well-a", "separator"],
            )

    def test_rejects_case_insensitive_duplicate_terminal_boundaries(self):
        inlet_specs = [
            self._inlet("well-a", "Well A feed", 12_000.0),
            self._inlet("well-b", "Well B feed", 8_000.0),
        ]
        graph_spec = {
            "name": "Ambiguous product boundaries",
            "units": [
                {
                    "id": "heater-a",
                    "name": "Product heater",
                    "type": "heater",
                    "ports": {
                        "material_in": ["in"],
                        "material_out": ["out"],
                    },
                    "params": {"outlet_temperature_c": 25.0},
                },
                {
                    "id": "heater-b",
                    "name": "product HEATER",
                    "type": "heater",
                    "ports": {
                        "material_in": ["in"],
                        "material_out": ["out"],
                    },
                    "params": {"outlet_temperature_c": 25.0},
                },
            ],
            "connections": [
                {
                    "id": "well-a-to-heater",
                    "name": "Well A transfer",
                    "type": "material",
                    "source": {
                        "kind": "inlet",
                        "id": "well-a",
                        "port": "out",
                    },
                    "target": {
                        "kind": "unit",
                        "id": "heater-a",
                        "port": "in",
                    },
                },
                {
                    "id": "well-b-to-heater",
                    "name": "Well B transfer",
                    "type": "material",
                    "source": {
                        "kind": "inlet",
                        "id": "well-b",
                        "port": "out",
                    },
                    "target": {
                        "kind": "unit",
                        "id": "heater-b",
                        "port": "in",
                    },
                },
            ],
        }

        with self.assertRaisesRegex(
            ValueError,
            "Terminal product stream name .* is duplicated",
        ):
            ProcessBuilder().build_acyclic_graph(
                graph_spec,
                inlet_specs,
                ["well-a", "well-b", "heater-a", "heater-b"],
            )


if __name__ == "__main__":
    unittest.main()
