"""Regression tests for solved material-boundary diagnostic adapters."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

from process_chat.solver_diagnostics import (
    aggregate_material_balance,
    material_boundary_rows,
    solved_feed_flow_kg_hr,
)


def _result(rows=None, **kpi_values):
    return SimpleNamespace(
        raw={"material_boundaries": rows or []},
        kpis={
            name: SimpleNamespace(value=value)
            for name, value in kpi_values.items()
        },
    )


class MaterialBoundaryDiagnosticsTest(unittest.TestCase):
    """Validate strict rows, aggregation, and legacy compatibility."""

    def test_returns_isolated_rows_and_aggregates_multiple_feeds(self):
        source_rows = [
            {
                "role": "feed",
                "stream_name": "dry gas",
                "mass_flow_kg_hr": 60_000,
                "temperature_C": 20,
                "pressure_bara": 45,
                "molar_flow_mol_sec": 900,
            },
            {
                "role": "feed",
                "stream_name": "rich gas",
                "mass_flow_kg_hr": 40_000,
                "temperature_C": 35,
                "pressure_bara": 45,
                "molar_flow_mol_sec": 500,
            },
            {
                "role": "product",
                "stream_name": "mixed product",
                "mass_flow_kg_hr": 100_000,
                "temperature_C": 25,
                "pressure_bara": 45,
                "molar_flow_mol_sec": 1400,
            },
        ]
        result = _result(
            source_rows,
            mass_balance_pct=1.0e-12,
        )

        rows = material_boundary_rows(result)
        rows[0]["mass_flow_kg_hr"] = 1.0
        self.assertEqual(
            result.raw["material_boundaries"][0]["mass_flow_kg_hr"],
            60_000,
        )
        summary = aggregate_material_balance(result)
        self.assertEqual(summary["feed_count"], 2.0)
        self.assertEqual(summary["product_count"], 1.0)
        self.assertEqual(summary["feed_flow_kg_hr"], 100_000.0)
        self.assertEqual(summary["product_flow_kg_hr"], 100_000.0)
        self.assertEqual(summary["imbalance_pct"], 1.0e-12)
        self.assertEqual(
            solved_feed_flow_kg_hr(result, 60_000.0),
            100_000.0,
        )

    def test_uses_solver_kpis_before_legacy_feed_fallback(self):
        result = _result(
            material_feed_count=2,
            material_product_count=1,
            material_feed_flow_kg_hr=100_000,
            material_product_flow_kg_hr=99_500,
        )

        summary = aggregate_material_balance(result)
        self.assertEqual(summary["feed_count"], 2.0)
        self.assertEqual(summary["product_count"], 1.0)
        self.assertEqual(summary["imbalance_pct"], 0.5)
        self.assertEqual(
            solved_feed_flow_kg_hr(result, 60_000.0),
            100_000.0,
        )

        legacy = _result()
        self.assertEqual(
            solved_feed_flow_kg_hr(legacy, 60_000.0),
            60_000.0,
        )

    def test_rejects_malformed_or_non_finite_diagnostics(self):
        invalid_rows = (
            ([{"role": "utility", "stream_name": "x",
               "mass_flow_kg_hr": 1}], "invalid role"),
            ([{"role": "feed", "stream_name": "",
               "mass_flow_kg_hr": 1}], "requires a stream name"),
            ([{"role": "feed", "stream_name": "x",
               "mass_flow_kg_hr": math.nan}], "must be finite"),
            ("not-an-array", "must be an array"),
        )
        for rows, message in invalid_rows:
            with self.subTest(message=message):
                result = SimpleNamespace(
                    raw={"material_boundaries": rows},
                    kpis={},
                )
                with self.assertRaisesRegex(ValueError, message):
                    material_boundary_rows(result)

        with self.assertRaisesRegex(ValueError, "finite and positive"):
            solved_feed_flow_kg_hr(_result(), 0.0)


if __name__ == "__main__":
    unittest.main()
