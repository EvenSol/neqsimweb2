"""Native conservation benchmark for the generic ProcessBuilder graph executor."""

from __future__ import annotations

import math
import unittest

from process_chat.process_builder import ProcessBuilder


class MultiInletMixerConservationTest(unittest.TestCase):
    """Validate material and energy closure for independent graph inlets."""

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
                mass_constraint = next(
                    constraint
                    for constraint in result.constraints
                    if constraint.name == "mass_balance"
                )
                self.assertEqual(mass_constraint.status, "OK")
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


if __name__ == "__main__":
    unittest.main()
