"""Deterministic performance guards for large Studio workspace projections."""

from time import perf_counter
from types import SimpleNamespace
import unittest

from process_chat.flowsheet_editor import build_graph_draft_dot
from studio.results_context import (
    StudioResultContext,
    equipment_design_rows,
    equipment_rows,
    stream_rows,
)


GRAPH_UNIT_COUNT = 500
RESULT_STREAM_COUNT = 2_000
RESULT_UNIT_COUNT = 1_000
GRAPH_BUDGET_SECONDS = 3.0
RESULT_BUDGET_SECONDS = 3.0


def _large_linear_graph():
    inlets = [{"id": "feed", "name": "Area 1 feed"}]
    units = [
        {
            "id": f"unit-{index:04d}",
            "name": f"Area 1 cooler {index:04d}",
            "type": "cooler",
            "ports": {
                "material_in": ["in"],
                "material_out": ["out"],
            },
        }
        for index in range(GRAPH_UNIT_COUNT)
    ]
    connections = [
        {
            "id": "stream-0000",
            "name": "Feed to cooler 0000",
            "type": "material",
            "source": {"kind": "inlet", "id": "feed", "port": "out"},
            "target": {"kind": "unit", "id": "unit-0000", "port": "in"},
        }
    ]
    for index in range(1, GRAPH_UNIT_COUNT):
        connections.append(
            {
                "id": f"stream-{index:04d}",
                "name": f"Cooler {index - 1:04d} to cooler {index:04d}",
                "type": "material",
                "source": {
                    "kind": "unit",
                    "id": f"unit-{index - 1:04d}",
                    "port": "out",
                },
                "target": {
                    "kind": "unit",
                    "id": f"unit-{index:04d}",
                    "port": "in",
                },
            }
        )
    return inlets, units, connections


def _design_properties():
    return {
        "designFlowCapacity_m3_per_hr": 120.0,
        "flowMargin_m3_per_hr": 20.0,
        "flowUtilization_pct": 83.333,
        "designHeadCapacity_m": 600.0,
        "headMargin_m": 100.0,
        "headUtilization_pct": 83.333,
        "motorRating_kW": 1_000.0,
        "motorMargin_kW": 150.0,
        "motorUtilization_pct": 85.0,
        "designDutyCapacity_kW": 2_500.0,
        "dutyMargin_kW": 300.0,
        "dutyUtilization_pct": 88.0,
        "designUACapacity_W_K": 125_000.0,
        "uaMargin_W_K": 15_000.0,
        "uaUtilization_pct": 88.0,
        "designCvCapacity_US": 100.0,
        "cvMargin_US": 10.0,
        "cvUtilization_pct": 90.0,
        "designPressureDropCapacity_bar": 5.0,
        "pressureDropMargin_bar": 0.5,
        "pressureDropUtilization_pct": 90.0,
        "designVelocityCapacity_m_s": 20.0,
        "velocityMargin_m_s": 2.0,
        "velocityUtilization_pct": 90.0,
        "velocityCriticalSegment_index": 24.0,
        "velocityCriticalLength_m": 1_250.0,
    }


class _LargeResultModel:
    def __init__(self):
        self._streams = tuple(
            SimpleNamespace(
                name=f"Stream {index:04d}",
                temperature_C=20.0 + index % 15,
                pressure_bara=80.0 - 0.01 * index,
                flow_rate_kg_hr=100_000.0,
                flow_rate_mol_sec=2_000.0,
                process_system=f"Area {index % 10 + 1}",
                owner_name=f"Equipment {index % RESULT_UNIT_COUNT:04d}",
            )
            for index in range(RESULT_STREAM_COUNT)
        )
        properties = _design_properties()
        self._units = tuple(
            SimpleNamespace(
                name=f"Equipment {index:04d}",
                unit_type="multi-service",
                process_system=f"Area {index % 10 + 1}",
                properties=dict(properties),
            )
            for index in range(RESULT_UNIT_COUNT)
        )

    def list_streams(self):
        return self._streams

    def list_units(self):
        return self._units


class LargeStudioWorkspacePerformanceTest(unittest.TestCase):
    """Guard against order-of-magnitude regressions in shared projections."""

    def test_large_graph_preview_is_deterministic_and_bounded(self):
        inlets, units, connections = _large_linear_graph()

        started = perf_counter()
        forward = build_graph_draft_dot(inlets, units, connections)
        elapsed = perf_counter() - started
        reverse = build_graph_draft_dot(
            inlets,
            units,
            list(reversed(connections)),
        )

        self.assertEqual(forward, reverse)
        self.assertEqual(forward.count(" -> "), GRAPH_UNIT_COUNT + 1)
        self.assertIn("Area 1 feed\\nINLET", forward)
        self.assertIn("unit-0499:out\\nPRODUCT", forward)
        self.assertLess(
            elapsed,
            GRAPH_BUDGET_SECONDS,
            (
                f"{GRAPH_UNIT_COUNT}-unit graph projection took "
                f"{elapsed:.3f}s; budget is {GRAPH_BUDGET_SECONDS:.1f}s"
            ),
        )
        print(
            "large Studio graph baseline: "
            f"units={GRAPH_UNIT_COUNT} connections={len(connections)} "
            f"seconds={elapsed:.6f}"
        )

    def test_large_result_tables_are_complete_and_bounded(self):
        model = _LargeResultModel()
        context = StudioResultContext(
            active_case={},
            spec={},
            model=model,
            result=SimpleNamespace(constraints=()),
            run_record={},
            warnings=(),
            signature="large-workspace-baseline",
        )

        started = perf_counter()
        streams = stream_rows(context)
        equipment = equipment_rows(context)
        design = equipment_design_rows(context)
        elapsed = perf_counter() - started

        self.assertEqual(len(streams), RESULT_STREAM_COUNT)
        self.assertEqual(len(equipment), RESULT_UNIT_COUNT)
        self.assertEqual(len(design), RESULT_UNIT_COUNT * 8)
        self.assertEqual(streams[0]["Stream"], "Stream 0000")
        self.assertEqual(streams[-1]["Stream"], "Stream 1999")
        self.assertEqual(equipment[-1]["Equipment"], "Equipment 0999")
        self.assertEqual(design[0]["Status"], "UNKNOWN")
        self.assertLess(
            elapsed,
            RESULT_BUDGET_SECONDS,
            (
                f"large result projection took {elapsed:.3f}s; "
                f"budget is {RESULT_BUDGET_SECONDS:.1f}s"
            ),
        )
        print(
            "large Studio result baseline: "
            f"streams={len(streams)} equipment={len(equipment)} "
            f"design_rows={len(design)} seconds={elapsed:.6f}"
        )


if __name__ == "__main__":
    unittest.main()
