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
    undo_graph_history,
    update_inline_unit_properties,
)
from process_chat.process_builder import ProcessBuilder
from process_chat.solver_diagnostics import aggregate_energy_balance


class MultiInletMixerConservationTest(unittest.TestCase):
    """Validate material and energy closure for independent graph inlets."""

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
                    {"outlet_pressure_bara": 40.0},
                )
                self.assertAlmostEqual(
                    float(valve.getOutletStream().getPressure("bara")),
                    40.0,
                    delta=0.05,
                )
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
                        "feed mixer",
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
