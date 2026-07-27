"""Headless regression for Streamlit warm-deployment module refreshes."""

from __future__ import annotations

import ast
import importlib
import json
import math
import os
import subprocess
import sys
import types
import unittest
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import process_chat.flowsheet_editor as flowsheet_editor
import process_chat.solver_diagnostics as solver_diagnostics
from streamlit.testing.v1 import AppTest


def _standalone_exit_code(result: unittest.TestResult) -> int:
    """Match unittest's standalone status without waiting for JVM teardown."""
    if result.testsRun == 0:
        return 5
    return 0 if result.wasSuccessful() else 1


def _flush_standalone_output(*streams) -> None:
    """Best-effort flush logs before the standalone process hard-exits."""
    for stream in streams:
        try:
            stream.flush()
        except (BrokenPipeError, OSError, ValueError):
            continue


class StudioWarmDeploymentTest(unittest.TestCase):
    """Exercise the deployed page with a deliberately stale module cache."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        if "theme" not in sys.modules:
            try:
                importlib.import_module("theme")
            except ModuleNotFoundError:
                theme = types.ModuleType("theme")
                theme.apply_theme = lambda: None
                theme.theme_toggle = lambda: None
                sys.modules["theme"] = theme

    def tearDown(self):
        importlib.reload(flowsheet_editor)
        importlib.reload(solver_diagnostics)

    def _run_studio(self):
        studio_path = (
            self.project_root / "pages" / "35_Process_Flowsheet_Studio.py"
        )

        app = AppTest.from_file(str(studio_path)).run(timeout=120)

        if app.exception:
            details = "\n".join(str(item.value) for item in app.exception)
            self.fail(f"Studio raised exceptions after warm reload:\n{details}")
        return app

    def _load_studio_function(self, function_name):
        studio_path = (
            self.project_root / "pages" / "35_Process_Flowsheet_Studio.py"
        )
        tree = ast.parse(studio_path.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == function_name
        )
        namespace = {"Any": Any, "json": json, "math": math}
        exec(
            compile(
                ast.Module(body=[function], type_ignores=[]),
                str(studio_path),
                "exec",
            ),
            namespace,
        )
        return namespace[function_name]

    def test_page_recovers_stale_editor_module(self):
        del flowsheet_editor.connect_graph_ports

        self._run_studio()

        self.assertTrue(callable(flowsheet_editor.connect_graph_ports))

    def test_page_recovers_stale_solver_diagnostics_module(self):
        del solver_diagnostics.aggregate_energy_balance
        del solver_diagnostics.aggregate_unit_balances
        del solver_diagnostics.energy_transfer_rows
        del solver_diagnostics.unit_balance_rows

        self._run_studio()

        self.assertTrue(
            callable(solver_diagnostics.aggregate_energy_balance)
        )
        self.assertTrue(
            callable(solver_diagnostics.aggregate_unit_balances)
        )
        self.assertTrue(callable(solver_diagnostics.energy_transfer_rows))
        self.assertTrue(callable(solver_diagnostics.unit_balance_rows))

    def test_page_recovers_multiple_stale_local_modules(self):
        del flowsheet_editor.connect_graph_ports
        del solver_diagnostics.aggregate_energy_balance

        self._run_studio()

        self.assertTrue(callable(flowsheet_editor.connect_graph_ports))
        self.assertTrue(
            callable(solver_diagnostics.aggregate_energy_balance)
        )

    def test_page_exposes_feed_and_standalone_equipment_creation(self):
        app = self._run_studio()
        button_labels = {button.label for button in app.button}

        self.assertIn("Add feed stream", button_labels)
        self.assertIn("Add equipment node", button_labels)
        self.assertIn("Connect selected ports", button_labels)

    def test_mixer_insertion_exposes_second_source_and_solve_readiness(self):
        app = self._run_studio()

        self.assertIn(
            "Second mixer inlet",
            {selectbox.label for selectbox in app.selectbox},
        )
        self.assertTrue(
            any(
                "must be connected before solving" in str(markdown.value)
                for markdown in app.markdown
            )
        )

    def test_original_equipment_can_be_replaced_without_rebuilding_paths(self):
        app = self._run_studio()
        path_selector = next(
            selectbox
            for selectbox in app.selectbox
            if selectbox.label == "Material path to reorganize"
        )
        path_selector.set_value(
            "inlet-scrubber-gas-to-compressor-stage-1"
        )
        app.run(timeout=120)

        equipment_action = next(
            radio
            for radio in app.radio
            if radio.label == "Equipment action"
        )
        equipment_action.set_value("Replace downstream equipment")
        app.run(timeout=120)

        name_input = next(
            text_input
            for text_input in app.text_input
            if (
                text_input.label == "Equipment name"
                and "reorganize_equipment_name" in text_input.key
            )
        )
        name_input.set_value("Replacement compressor")
        app.run(timeout=120)
        replace_button = next(
            button
            for button in app.button
            if button.label
            == "Replace equipment and preserve surrounding path"
        )
        replace_button.click()
        app.run(timeout=120)

        if app.exception:
            details = "\n".join(str(item.value) for item in app.exception)
            self.fail(
                "Original equipment replacement raised exceptions:\n"
                + details
            )
        draft = app.session_state["flowsheet_studio_graph_draft"]
        unit_ids = {
            str(unit["id"])
            for unit in draft["units"]
            if isinstance(unit, dict)
        }
        self.assertNotIn("compressor-stage-1", unit_ids)
        self.assertIn("replacement-compressor", unit_ids)
        connection_endpoints = [
            (
                connection["source"]["id"],
                connection["target"]["id"],
            )
            for connection in draft["connections"]
            if connection.get("type") == "material"
        ]
        self.assertIn(
            ("inlet-scrubber", "replacement-compressor"),
            connection_endpoints,
        )
        self.assertIn(
            ("replacement-compressor", "intercooler"),
            connection_endpoints,
        )

    def test_replacement_disables_branching_downstream_equipment_up_front(self):
        app = self._run_studio()
        equipment_action = next(
            radio
            for radio in app.radio
            if radio.label == "Equipment action"
        )
        equipment_action.set_value("Replace downstream equipment")
        app.run(timeout=120)

        replace_button = next(
            button
            for button in app.button
            if button.label
            == "Replace equipment and preserve surrounding path"
        )
        self.assertTrue(replace_button.disabled)
        self.assertTrue(
            any(
                "cannot be replaced as one continuous path"
                in str(warning.value)
                for warning in app.warning
            )
        )

    def test_palette_routes_multi_port_units_through_standalone_workflow(self):
        app = self._run_studio()
        selectboxes = {
            selectbox.label: selectbox
            for selectbox in app.selectbox
        }

        standalone_options = set(
            selectboxes["Standalone equipment type"].options
        )
        inline_options = set(selectboxes["Equipment type"].options)
        extend_options = set(selectboxes["Next equipment"].options)
        self.assertIn("Mixer · Flow routing", standalone_options)
        self.assertIn("Separator · Separation", standalone_options)
        self.assertNotIn("Mixer · Flow routing", inline_options)
        self.assertNotIn("Separator · Separation", inline_options)
        self.assertNotIn("Mixer", extend_options)
        self.assertIn("Separator", extend_options)

    def test_page_clears_stale_multi_inlet_extend_selection(self):
        app = self._run_studio()
        extend_selectbox = next(
            selectbox
            for selectbox in app.selectbox
            if selectbox.label == "Next equipment"
        )
        studio_path = (
            self.project_root / "pages" / "35_Process_Flowsheet_Studio.py"
        )
        warm_app = AppTest.from_file(str(studio_path))
        warm_app.session_state[extend_selectbox.key] = "mixer"

        warm_app.run(timeout=120)

        if warm_app.exception:
            details = "\n".join(
                str(item.value) for item in warm_app.exception
            )
            self.fail(
                "Studio retained an invalid warm-session selection:\n"
                + details
            )
        extend_selectbox = next(
            selectbox
            for selectbox in warm_app.selectbox
            if selectbox.label == "Next equipment"
        )
        self.assertNotEqual(extend_selectbox.value, "mixer")

    def test_graph_object_name_falls_back_for_legacy_records(self):
        graph_object_name = self._load_studio_function(
            "_graph_object_name"
        )

        self.assertEqual(
            graph_object_name({"id": "feed-a", "name": "Feed A"}, "feed-a"),
            "Feed A",
        )
        self.assertEqual(
            graph_object_name({"id": "feed-a"}, "feed-a"),
            "feed-a",
        )
        self.assertEqual(
            graph_object_name({"id": "feed-a", "name": " "}, "feed-a"),
            "feed-a",
        )
        self.assertEqual(
            graph_object_name({"id": "feed-a", "name": None}, "feed-a"),
            "feed-a",
        )

    def test_material_output_notice_uses_declared_phase_ports(self):
        output_label = self._load_studio_function(
            "_material_output_selection_label"
        )

        self.assertEqual(
            output_label(
                {
                    "id": "condensate-pump",
                    "ports": {"material_out": ["out"]},
                }
            ),
            "'condensate-pump:out'",
        )
        self.assertEqual(
            output_label(
                {
                    "id": "product-separator",
                    "ports": {"material_out": ["gas", "liquid"]},
                }
            ),
            "one of 'product-separator:gas', "
            "'product-separator:liquid'",
        )

    def test_secondary_inlet_map_ignores_missing_and_blank_ids(self):
        secondary_inlet_map = self._load_studio_function(
            "_secondary_inlet_map"
        )
        primary = {"id": "feed-gas", "name": "Primary"}
        secondary = {"id": "liquid-feed", "name": "Liquid feed"}

        result = secondary_inlet_map(
            [
                primary,
                secondary,
                {"name": "Missing id"},
                {"id": "   ", "name": "Blank id"},
                {"id": None, "name": "Null id"},
                None,
            ],
            "feed-gas",
        )

        self.assertEqual(result, {"liquid-feed": secondary})

    def test_required_identifier_rejects_null_and_blank_ids(self):
        required_identifier = self._load_studio_function(
            "_required_identifier"
        )

        self.assertEqual(required_identifier(" feed-a ", "inlet id"), "feed-a")
        for invalid_id in (None, "", "   "):
            with self.subTest(invalid_id=invalid_id):
                with self.assertRaisesRegex(ValueError, "cannot be empty"):
                    required_identifier(invalid_id, "inlet id")

    def test_graph_name_set_ignores_null_and_blank_names(self):
        graph_name_set = self._load_studio_function("_graph_name_set")
        records = [
            {"id": "named", "name": " Feed A "},
            {"id": "null", "name": None},
            {"id": "blank", "name": "  "},
            {"id": "missing"},
            None,
        ]

        self.assertEqual(graph_name_set(records), {"Feed A"})
        self.assertEqual(
            graph_name_set(records, casefold=True),
            {"feed a"},
        )

    def test_terminal_names_are_reserved_for_unconnected_outputs(self):
        terminal_names = self._load_studio_function(
            "_terminal_material_stream_names"
        )
        units = [
            {
                "id": "separator",
                "name": "Inlet scrubber",
                "ports": {"material_out": ["gas", "liquid"]},
            }
        ]
        connections = [
            {
                "type": "material",
                "source": {
                    "kind": "unit",
                    "id": "separator",
                    "port": "gas",
                }
            },
            {
                "type": "energy",
                "source": {
                    "kind": "unit",
                    "id": "separator",
                    "port": "liquid",
                },
            },
        ]

        self.assertEqual(
            terminal_names(units, connections),
            {"Inlet scrubber [liquid] product"},
        )

    def test_registry_change_reconciles_cloned_feed_composition(self):
        reconcile = self._load_studio_function(
            "_reconcile_inlet_composition"
        )

        result = reconcile(
            {"methane": 0.7, "ethane": 0.3},
            {"methane": 0.5, "propane": 0.5},
        )

        self.assertEqual(list(result), ["methane", "propane"])
        self.assertAlmostEqual(result["methane"], 0.7 / 1.2)
        self.assertAlmostEqual(result["propane"], 0.5 / 1.2)
        self.assertAlmostEqual(sum(result.values()), 1.0)

    def test_terminal_names_conflict_with_feeds_and_equipment(self):
        terminal_name_conflicts = self._load_studio_function(
            "_terminal_name_conflicts"
        )

        self.assertEqual(
            terminal_name_conflicts(
                [
                    {
                        "id": "feed",
                        "name": "Inlet scrubber [liquid] product",
                    },
                    {
                        "id": "pump",
                        "name": "Cooler [out] product",
                    },
                    {"id": "blank", "name": " "},
                    {"id": "null", "name": None},
                ],
                {
                    "INLET SCRUBBER [LIQUID] PRODUCT",
                    "Cooler [out] product",
                },
            ),
            [
                "Cooler [out] product",
                "Inlet scrubber [liquid] product",
            ],
        )

    def test_feed_names_reserve_restorable_starter_products(self):
        reserved_feed_names = self._load_studio_function(
            "_reserved_feed_names"
        )
        graph_name_set = self._load_studio_function("_graph_name_set")
        terminal_names = self._load_studio_function(
            "_terminal_material_stream_names"
        )
        reserved_feed_names.__globals__.update(
            {
                "_graph_name_set": graph_name_set,
                "_terminal_material_stream_names": terminal_names,
            }
        )
        starter_units = [
            {
                "id": "export-cooler",
                "name": "Export cooler",
                "ports": {"material_out": ["out"]},
            }
        ]
        reserved_names = reserved_feed_names(
            [],
            [],
            starter_units,
            [],
        )

        self.assertIn("Export cooler [out] product", reserved_names)
        with self.assertRaisesRegex(ValueError, "duplicated"):
            flowsheet_editor.clone_material_inlet(
                [{"id": "feed", "name": "Feed", "ports": {}}],
                "feed",
                "Export cooler [out] product",
                reserved_names=reserved_names,
            )

    def test_import_rejects_restorable_starter_product_feed_name(self):
        validate_case_graph = self._load_studio_function(
            "_validate_case_graph"
        )
        terminal_name_conflicts = self._load_studio_function(
            "_terminal_name_conflicts"
        )
        terminal_material_stream_names = self._load_studio_function(
            "_terminal_material_stream_names"
        )
        starter_units = [
            {
                "id": "export-cooler",
                "name": "Export cooler",
                "ports": {"material_out": ["out"]},
            }
        ]
        validate_case_graph.__globals__.update(
            {
                "CASE_SCHEMA_VERSION": 3,
                "_validate_graph_integrity": lambda *args: None,
                "_terminal_name_conflicts": terminal_name_conflicts,
                "_terminal_material_stream_names": (
                    terminal_material_stream_names
                ),
                "_index_graph_objects": lambda *args: {},
                "_build_template_graph": lambda process: (
                    starter_units,
                    [],
                ),
                "validate_starter_unit_projection": lambda *args: None,
                "_build_execution_plan": lambda case_data: [],
            }
        )
        case_data = {
            "schema_version": 3,
            "inlets": [
                {
                    "id": "secondary-feed",
                    "name": "Export cooler [out] product",
                }
            ],
            "units": [],
            "connections": [],
        }

        with self.assertRaisesRegex(
            ValueError,
            "restorable starter product streams",
        ):
            validate_case_graph(case_data, [])

    def test_solve_readiness_rejects_disconnected_feeds(self):
        validate_solve_readiness = self._load_studio_function(
            "_validate_graph_solve_readiness"
        )
        case_spec = {
            "inlets": [
                {"id": "feed-gas"},
                {"id": "tie-in-feed"},
            ],
            "connections": [
                {
                    "source": {
                        "kind": "inlet",
                        "id": "feed-gas",
                        "port": "out",
                    }
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "tie-in-feed"):
            validate_solve_readiness(case_spec)
        case_spec["connections"].append(
            {
                "source": {
                    "kind": "inlet",
                    "id": "tie-in-feed",
                    "port": "out",
                }
            }
        )
        validate_solve_readiness(case_spec)

    def test_omitted_starter_controls_do_not_block_graph_validation(self):
        validate_case = self._load_studio_function("_validate_case")
        has_material_connection = self._load_studio_function(
            "_has_material_connection"
        )
        fluid = {
            "pressure_bara": 100.0,
            "total_flow": 1000.0,
            "eos_model": "srk",
        }
        process = [
            {},
            {},
            {
                "params": {
                    "outlet_pressure_bara": 80.0,
                    "isentropic_efficiency": 0.10,
                }
            },
            {"params": {"pressure_drop_bar": 5.0}},
            {},
            {
                "params": {
                    "outlet_pressure_bara": 70.0,
                    "isentropic_efficiency": 0.10,
                }
            },
            {"params": {"pressure_drop_bar": 5.0}},
        ]
        template_ids = {
            "inlet scrubber": "inlet-scrubber",
            "compressor stage 1": "compressor-stage-1",
            "intercooler": "intercooler",
            "interstage scrubber": "interstage-scrubber",
            "compressor stage 2": "compressor-stage-2",
            "export cooler": "export-cooler",
        }
        spec = {
            "fluid": fluid,
            "process": process,
            "units": [
                {"id": "intercooler"},
                {"id": "export-cooler"},
            ],
        }
        validate_case.__globals__.update(
            {
                "_build_execution_plan": lambda candidate: [],
                "_build_inlet_fluid_specs": lambda candidate: [
                    {
                        "inlet_id": "feed-gas",
                        "fluid_spec": candidate["fluid"],
                    }
                ],
                "PRIMARY_INLET_ID": "feed-gas",
                "TEMPLATE_UNIT_IDS": template_ids,
                "_has_material_connection": has_material_connection,
            }
        )

        self.assertEqual(validate_case(spec, 1.0), [])
        spec["process"] = [{}]
        spec["units"] = []
        self.assertEqual(validate_case(spec, 1.0), [])

        spec["process"] = process
        spec["units"].append({"id": "compressor-stage-1"})
        with self.assertRaisesRegex(ValueError, "efficiency"):
            validate_case(spec, 1.0)

    def test_reordered_starter_compressors_skip_template_pressure_order(self):
        validate_case = self._load_studio_function("_validate_case")
        has_material_connection = self._load_studio_function(
            "_has_material_connection"
        )
        fluid = {
            "pressure_bara": 50.0,
            "total_flow": 1000.0,
            "eos_model": "srk",
        }
        process = [
            {},
            {},
            {
                "params": {
                    "outlet_pressure_bara": 160.0,
                    "isentropic_efficiency": 0.80,
                }
            },
            {"params": {"pressure_drop_bar": 1.0}},
            {},
            {
                "params": {
                    "outlet_pressure_bara": 80.0,
                    "isentropic_efficiency": 0.80,
                }
            },
            {"params": {"pressure_drop_bar": 1.0}},
        ]
        template_ids = {
            "inlet scrubber": "inlet-scrubber",
            "compressor stage 1": "compressor-stage-1",
            "intercooler": "intercooler",
            "interstage scrubber": "interstage-scrubber",
            "compressor stage 2": "compressor-stage-2",
            "export cooler": "export-cooler",
        }
        spec = {
            "fluid": fluid,
            "process": process,
            "units": [
                {"id": "compressor-stage-1"},
                {"id": "compressor-stage-2"},
            ],
            "connections": [
                {
                    "type": "material",
                    "source": {
                        "kind": "unit",
                        "id": "compressor-stage-2",
                        "port": "out",
                    },
                    "target": {
                        "kind": "unit",
                        "id": "compressor-stage-1",
                        "port": "in",
                    },
                }
            ],
        }
        validate_case.__globals__.update(
            {
                "_build_execution_plan": lambda candidate: [],
                "_build_inlet_fluid_specs": lambda candidate: [
                    {
                        "inlet_id": "feed-gas",
                        "fluid_spec": candidate["fluid"],
                    }
                ],
                "PRIMARY_INLET_ID": "feed-gas",
                "TEMPLATE_UNIT_IDS": template_ids,
                "_has_material_connection": has_material_connection,
            }
        )

        self.assertEqual(validate_case(spec, 1.0), [])

    def test_retained_stage_one_validates_feed_pressure_without_stage_two(self):
        validate_case = self._load_studio_function("_validate_case")
        has_material_connection = self._load_studio_function(
            "_has_material_connection"
        )
        fluid = {
            "pressure_bara": 100.0,
            "total_flow": 1000.0,
            "eos_model": "srk",
        }
        process = [
            {},
            {},
            {
                "params": {
                    "outlet_pressure_bara": 80.0,
                    "isentropic_efficiency": 0.80,
                }
            },
        ]
        template_ids = {
            "inlet scrubber": "inlet-scrubber",
            "compressor stage 1": "compressor-stage-1",
            "intercooler": "intercooler",
            "interstage scrubber": "interstage-scrubber",
            "compressor stage 2": "compressor-stage-2",
            "export cooler": "export-cooler",
        }
        spec = {
            "fluid": fluid,
            "process": process,
            "units": [
                {"id": "inlet-scrubber"},
                {"id": "compressor-stage-1"},
            ],
            "connections": [
                {
                    "type": "material",
                    "source": {
                        "kind": "inlet",
                        "id": "feed-gas",
                        "port": "out",
                    },
                    "target": {
                        "kind": "unit",
                        "id": "inlet-scrubber",
                        "port": "in",
                    },
                },
                {
                    "type": "material",
                    "source": {
                        "kind": "unit",
                        "id": "inlet-scrubber",
                        "port": "gas",
                    },
                    "target": {
                        "kind": "unit",
                        "id": "compressor-stage-1",
                        "port": "in",
                    },
                },
            ],
        }
        validate_case.__globals__.update(
            {
                "_build_execution_plan": lambda candidate: [],
                "_build_inlet_fluid_specs": lambda candidate: [
                    {
                        "inlet_id": "feed-gas",
                        "fluid_spec": candidate["fluid"],
                    }
                ],
                "PRIMARY_INLET_ID": "feed-gas",
                "TEMPLATE_UNIT_IDS": template_ids,
                "_has_material_connection": has_material_connection,
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "feed pressure < stage 1 pressure",
        ):
            validate_case(spec, 1.0)

        spec["fluid"]["pressure_bara"] = 50.0
        self.assertEqual(validate_case(spec, 1.0), [])

    def test_pressure_profile_only_reports_retained_starter_operations(self):
        pressure_profile_dataframe = self._load_studio_function(
            "_pressure_profile_dataframe"
        )
        has_material_connection = self._load_studio_function(
            "_has_material_connection"
        )
        active_steps = self._load_studio_function(
            "_active_template_process_steps"
        )
        pandas = __import__("pandas")
        template_ids = {
            "compressor stage 1": "compressor-stage-1",
            "intercooler": "intercooler",
            "interstage scrubber": "interstage-scrubber",
            "compressor stage 2": "compressor-stage-2",
            "export cooler": "export-cooler",
        }
        pressure_profile_dataframe.__globals__.update(
            {
                "pd": pandas,
                "TEMPLATE_UNIT_IDS": template_ids,
                "_active_template_process_steps": active_steps,
                "_has_material_connection": has_material_connection,
            }
        )
        active_steps.__globals__["TEMPLATE_UNIT_IDS"] = template_ids
        spec = {
            "process": [
                {
                    "name": "compressor stage 1",
                    "params": {"outlet_pressure_bara": 80.0},
                },
                {
                    "name": "intercooler",
                    "params": {"pressure_drop_bar": 1.0},
                },
                {
                    "name": "compressor stage 2",
                    "params": {"outlet_pressure_bara": 160.0},
                },
                {
                    "name": "export cooler",
                    "params": {"pressure_drop_bar": 1.0},
                },
            ],
            "units": [],
            "connections": [],
        }
        equipment_table = pandas.DataFrame(
            [
                {
                    "Equipment": "compressor stage 1",
                    "outletPressure_bara": 80.0,
                }
            ]
        )

        omitted_profile = pressure_profile_dataframe(spec, equipment_table)
        self.assertTrue(omitted_profile.empty)
        self.assertIn("Status", omitted_profile.columns)

        spec["process"] = [{}, None, {"name": "  "}]
        placeholder_profile = pressure_profile_dataframe(spec, equipment_table)
        self.assertTrue(placeholder_profile.empty)

        spec["process"] = [
            {
                "name": "compressor stage 1",
                "params": {"outlet_pressure_bara": 80.0},
            },
            {
                "name": "intercooler",
                "params": {"pressure_drop_bar": 1.0},
            },
            {
                "name": "compressor stage 2",
                "params": {"outlet_pressure_bara": 160.0},
            },
            {
                "name": "export cooler",
                "params": {"pressure_drop_bar": 1.0},
            },
        ]
        spec["units"] = [{"id": "compressor-stage-1"}]
        retained_profile = pressure_profile_dataframe(spec, equipment_table)
        self.assertEqual(
            retained_profile["Operation"].tolist(),
            ["Compressor stage 1"],
        )
        self.assertEqual(retained_profile["Status"].tolist(), ["OK"])

        spec["units"] = [
            {"id": "compressor-stage-1"},
            {"id": "compressor-stage-2"},
            {"id": "intercooler"},
        ]
        spec["connections"] = [
            {
                "type": "material",
                "source": {
                    "kind": "unit",
                    "id": "compressor-stage-2",
                    "port": "out",
                },
                "target": {
                    "kind": "unit",
                    "id": "intercooler",
                    "port": "in",
                },
            }
        ]
        reordered_profile = pressure_profile_dataframe(spec, equipment_table)
        self.assertEqual(
            reordered_profile["Operation"].tolist(),
            ["Compressor stage 1", "Compressor stage 2"],
        )

    def test_workbook_and_history_only_report_active_starter_equipment(self):
        active_steps = self._load_studio_function(
            "_active_template_process_steps"
        )
        workbook_bytes = self._load_studio_function(
            "_engineering_workbook_bytes"
        )
        history_record = self._load_studio_function("_case_history_record")
        workbook_cell = self._load_studio_function("_workbook_cell")
        pandas = __import__("pandas")
        template_ids = {
            "compressor stage 1": "compressor-stage-1",
            "intercooler": "intercooler",
            "interstage scrubber": "interstage-scrubber",
            "compressor stage 2": "compressor-stage-2",
            "export cooler": "export-cooler",
        }
        active_steps.__globals__["TEMPLATE_UNIT_IDS"] = template_ids

        process = [
            {},
            {},
            {
                "params": {
                    "outlet_pressure_bara": 80.0,
                    "isentropic_efficiency": 0.76,
                },
            },
            {
                "name": "intercooler",
                "params": {
                    "outlet_temperature_C": 35.0,
                },
            },
            {},
            {
                "name": "compressor stage 2",
                "params": {
                    "outlet_pressure_bara": 160.0,
                    "isentropic_efficiency": 0.77,
                },
            },
            {
                "name": "export cooler",
                "params": {
                    "outlet_temperature_C": 30.0,
                    "pressure_drop_bar": 1.5,
                },
            },
        ]
        fluid = {
            "eos_model": "srk",
            "mixing_rule": "classic",
            "composition_basis": "mole fraction",
            "temperature_C": 20.0,
            "pressure_bara": 40.0,
            "total_flow": 20_000.0,
            "flow_unit": "kg/hr",
            "components": {"methane": 1.0},
        }
        spec = {
            "name": "Active summary",
            "fluid": fluid,
            "process": process,
            "fluid_packages": [
                {
                    "id": "base-fluid",
                    "name": "Base fluid",
                    "eos_model": "srk",
                    "mixing_rule": "classic",
                    "component_registry": {"methane": {}},
                    "binary_interaction_parameters": {"source": "NeqSim"},
                }
            ],
            "inlets": [
                {
                    "id": "feed-gas",
                    "name": "Feed gas",
                    "fluid_package_id": "base-fluid",
                    "temperature_C": 20.0,
                    "pressure_bara": 40.0,
                    "total_flow": 20_000.0,
                    "flow_unit": "kg/hr",
                    "composition_basis": "mole fraction",
                    "composition": {"methane": 1.0},
                }
            ],
            "units": [
                {
                    "id": "compressor-stage-1",
                    "name": "Compressor stage 1",
                    "type": "compressor",
                    "ports": {
                        "material_in": ["in"],
                        "material_out": ["out"],
                        "energy_in": [],
                        "energy_out": ["power"],
                    },
                    "properties": {},
                },
                {
                    "id": "intercooler",
                    "name": "Intercooler",
                    "type": "cooler",
                    "ports": {
                        "material_in": ["in"],
                        "material_out": ["out"],
                        "energy_in": [],
                        "energy_out": ["heat"],
                    },
                    "properties": {},
                },
                {
                    "id": "interstage-scrubber",
                    "name": "Interstage scrubber",
                    "type": "separator",
                    "ports": {
                        "material_in": ["in"],
                        "material_out": ["gas", "liquid"],
                        "energy_in": [],
                        "energy_out": [],
                    },
                    "properties": {},
                },
            ],
            "connections": [],
        }
        self.assertEqual(
            list(active_steps(spec)),
            ["compressor stage 1", "intercooler"],
        )

        empty_table = pandas.DataFrame()
        convergence_summary = {
            "unit_count": 0,
            "unconverged_count": 0,
            "max_iterations": 0,
        }
        unit_balance_summary = {
            "unit_count": 0,
            "max_mass_imbalance_pct": None,
            "max_mass_imbalance_unit": None,
            "energy_unit_count": 0,
            "max_energy_imbalance_pct": None,
            "max_energy_imbalance_unit": None,
            "excluded_units": [],
        }
        workbook_bytes.__globals__.update(
            {
                "BytesIO": BytesIO,
                "TEMPLATE_NAME": "Compression starter",
                "_active_template_process_steps": active_steps,
                "_build_execution_plan": lambda candidate: [],
                "_build_inlet_fluid_specs": lambda candidate: [],
                "_component_balance_dataframe": lambda result: empty_table,
                "_convergence_dataframe": lambda result: empty_table,
                "_convergence_state_label": lambda summary: "Converged",
                "_energy_balance_dataframe": lambda result: empty_table,
                "_energy_transfer_dataframe": lambda result: empty_table,
                "_kpi_value": lambda result, name: None,
                "_material_boundary_dataframe": lambda result: empty_table,
                "_unit_balance_coverage_label": lambda summary: "n/a",
                "_unit_balance_dataframe": lambda result: empty_table,
                "_unit_identity_label": lambda identity: "n/a",
                "_workbook_cell": workbook_cell,
                "aggregate_convergence": lambda result: convergence_summary,
                "aggregate_unit_balances": lambda result: unit_balance_summary,
                "json": json,
                "pd": pandas,
                "solved_feed_flow_kg_hr": (
                    lambda result, fallback: float(fallback)
                ),
            }
        )
        result = types.SimpleNamespace(constraints=[])
        workbook = workbook_bytes(
            spec,
            result,
            empty_table,
            empty_table,
            empty_table,
            empty_table,
            {},
        )
        with zipfile.ZipFile(BytesIO(workbook)) as archive:
            workbook_xml = "\n".join(
                archive.read(name).decode("utf-8")
                for name in archive.namelist()
                if name.endswith(".xml")
            )
        self.assertIn("Compressor stage 1", workbook_xml)
        self.assertNotIn("Compressor stage 2", workbook_xml)
        self.assertIn("Intercooler", workbook_xml)
        self.assertNotIn("Export cooler", workbook_xml)

        history_record.__globals__.update(
            {
                "_active_template_process_steps": active_steps,
                "_convergence_state_label": lambda summary: "Converged",
                "_kpi_value": lambda result, name: None,
                "aggregate_convergence": lambda result: convergence_summary,
                "aggregate_validation_status": lambda statuses: "PASS",
                "json": json,
                "solved_feed_flow_kg_hr": (
                    lambda result, fallback: float(fallback)
                ),
            }
        )
        history = history_record(spec, result, "abc12345")
        self.assertEqual(history["Stage 1 pressure [bara]"], 80.0)
        self.assertEqual(history["Stage 1 efficiency [-]"], 0.76)
        self.assertIsNone(history["Stage 2 pressure [bara]"])
        self.assertIsNone(history["Stage 2 efficiency [-]"])
        self.assertEqual(history["Intercooler pressure drop [bar]"], 0.0)
        self.assertIsNone(history["Export cooler pressure drop [bar]"])

    def test_disconnected_starter_inventory_requires_no_graph_references(self):
        unconnected_unit_map = self._load_studio_function(
            "_unconnected_unit_map"
        )
        units = [
            {"id": "inlet-scrubber", "name": "Inlet scrubber"},
            {"id": "compressor-stage-1", "name": "Compressor stage 1"},
            {"id": "intercooler", "name": "Intercooler"},
        ]
        connections = [
            {
                "type": "material",
                "source": {
                    "kind": "unit",
                    "id": "compressor-stage-1",
                    "port": "out",
                },
                "target": {
                    "kind": "unit",
                    "id": "product",
                    "port": "in",
                },
            },
            {
                "type": "energy",
                "source": {
                    "kind": "unit",
                    "id": "utility",
                    "port": "out",
                },
                "target": {
                    "kind": "unit",
                    "id": "intercooler",
                    "port": "energy",
                },
            },
            None,
        ]

        self.assertEqual(
            unconnected_unit_map(
                units,
                connections,
                {
                    "inlet-scrubber",
                    "compressor-stage-1",
                    "intercooler",
                },
            ),
            {"inlet-scrubber": units[0]},
        )

    def test_feed_draft_refreshes_current_template_unit_properties(self):
        apply_studio_graph_draft = self._load_studio_function(
            "_apply_studio_graph_draft"
        )
        reconcile = self._load_studio_function(
            "_reconcile_inlet_composition"
        )
        current_template_unit = {
            "id": "stage-1-compressor",
            "name": "Stage 1 compressor",
            "type": "compressor",
            "params": {"outlet_pressure_bara": 91.0},
        }
        stale_template_unit = {
            **current_template_unit,
            "params": {"outlet_pressure_bara": 80.0},
        }
        standalone_unit = {
            "id": "liquid-pump",
            "name": "Liquid pump",
            "type": "pump",
            "params": {"outlet_pressure_bara": 25.0},
        }
        primary_inlet = {
            "id": "feed-gas",
            "name": "Feed gas",
            "fluid_package_id": "base-fluid",
            "composition": {"methane": 0.8, "propane": 0.2},
        }
        secondary_inlet = {
            "id": "liquid-feed",
            "name": "Liquid feed",
            "fluid_package_id": "base-fluid",
            "composition": {"methane": 0.6, "ethane": 0.4},
        }
        case_spec = {
            "process": [{"name": "current controls"}],
            "inlets": [primary_inlet],
            "units": [current_template_unit],
            "connections": [],
        }
        draft = {
            "inlets": [primary_inlet, secondary_inlet],
            "units": [stale_template_unit, standalone_unit],
            "connections": [],
        }

        apply_studio_graph_draft.__globals__.update(
            {
                "PRIMARY_INLET_ID": "feed-gas",
                "_build_template_graph": lambda process: (
                    [current_template_unit],
                    [],
                ),
                "_reconcile_inlet_composition": reconcile,
                "apply_graph_draft": lambda case, graph: {
                    **case,
                    **graph,
                },
            }
        )
        result = apply_studio_graph_draft(case_spec, draft)

        self.assertEqual(result["units"][0], current_template_unit)
        self.assertEqual(result["units"][1], standalone_unit)
        self.assertEqual(
            list(result["inlets"][1]["composition"]),
            ["methane", "propane"],
        )
        self.assertAlmostEqual(
            sum(result["inlets"][1]["composition"].values()),
            1.0,
        )
        self.assertEqual(draft["units"][0], stale_template_unit)
        self.assertEqual(
            draft["inlets"][1]["composition"],
            {"methane": 0.6, "ethane": 0.4},
        )

    def test_standalone_no_test_selection_returns_five(self):
        studio_test_path = Path(__file__).resolve()
        completed = subprocess.run(
            [
                sys.executable,
                str(studio_test_path),
                "-k",
                "definitely_no_matching_test",
            ],
            cwd=self.project_root,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    filter(
                        None,
                        (
                            str(self.project_root),
                            os.environ.get("PYTHONPATH"),
                        ),
                    )
                ),
            },
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        self.assertEqual(
            completed.returncode,
            5,
            completed.stdout + completed.stderr,
        )

    def test_standalone_flush_continues_after_broken_pipe(self):
        flushed = []

        class BrokenStream:
            def flush(self):
                raise BrokenPipeError

        class HealthyStream:
            def flush(self):
                flushed.append(True)

        _flush_standalone_output(BrokenStream(), HealthyStream())

        self.assertEqual(flushed, [True])

    def test_zero_coverage_is_not_reported_as_not_applicable(self):
        coverage_label = self._load_studio_function(
            "_unit_balance_coverage_label"
        )
        zero_coverage = {
            "applicable": False,
            "coverage_complete": False,
            "energy_unit_count": 0.0,
            "energy_coverage_complete": False,
        }
        no_candidates = {
            **zero_coverage,
            "coverage_complete": True,
        }

        self.assertEqual(
            coverage_label(zero_coverage),
            "Material unavailable; energy not audited",
        )
        self.assertEqual(coverage_label(no_candidates), "Not applicable")

    def test_solved_page_reports_and_exports_unit_closure(self):
        app = self._run_studio()
        run_button = next(
            button
            for button in app.button
            if button.label == "▶ Run NeqSim flowsheet"
        )

        run_button.click()
        app.run(timeout=240)

        if app.exception:
            details = "\n".join(str(item.value) for item in app.exception)
            self.fail(f"Studio solve raised exceptions:\n{details}")
        closure_metrics = {
            metric.label: metric
            for metric in app.metric
            if metric.label.startswith("Maximum unit ")
        }
        self.assertEqual(
            set(closure_metrics),
            {
                "Maximum unit mass imbalance",
                "Maximum unit energy imbalance",
            },
        )
        mass_metric = closure_metrics["Maximum unit mass imbalance"]
        self.assertTrue(mass_metric.value.endswith(" %"))
        self.assertIn(" / ", mass_metric.help)
        energy_metric = closure_metrics["Maximum unit energy imbalance"]
        if energy_metric.value == "n/a":
            self.assertEqual(energy_metric.help, "n/a")
        else:
            self.assertTrue(energy_metric.value.endswith(" %"))
            self.assertIn(" / ", energy_metric.help)
        self.assertTrue(
            any(
                "Mass imbalance [%]" in dataframe.value.columns
                and "Energy imbalance [%]" in dataframe.value.columns
                and "Inlet enthalpy flow [kW]" in dataframe.value.columns
                and "Outlet enthalpy flow [kW]" in dataframe.value.columns
                for dataframe in app.dataframe
            )
        )
        self.assertIn(
            "Download engineering workbook",
            [button.label for button in app.get("download_button")],
        )
        captions = [caption.value for caption in app.caption]
        self.assertTrue(
            any(
                "Mass residual is outlet mass flow minus inlet mass flow."
                in caption
                for caption in captions
            )
        )
        self.assertTrue(
            any(
                "Energy residual is outlet enthalpy flow minus inlet "
                "enthalpy flow minus signed external energy transfer."
                in caption
                for caption in captions
            )
        )


if __name__ == "__main__":
    program = unittest.main(exit=False)
    exit_code = _standalone_exit_code(program.result)
    try:
        _flush_standalone_output(sys.stdout, sys.stderr)
    finally:
        os._exit(exit_code)
