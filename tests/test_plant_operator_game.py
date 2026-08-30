"""Unit tests for the NeqSim Plant Operator challenge contract."""

import math
from types import SimpleNamespace
import unittest

from process_chat.plant_operator_game import (
    ChallengeControls,
    ChallengeEvidence,
    TARGET_FLOW_KG_HR,
    assess_challenge,
    build_challenge_spec,
    collect_evidence,
    run_challenge,
    validate_controls,
)


class PlantOperatorGameTest(unittest.TestCase):
    """Protect reproducibility, validation, and transparent scoring."""

    def test_builds_minimum_reproducible_native_process_spec(self):
        controls = ChallengeControls(
            feed_flow_kg_hr=112_000.0,
            stage_1_pressure_bara=82.0,
            stage_2_pressure_bara=131.0,
            intercooler_temperature_c=32.0,
            export_temperature_c=39.0,
        )

        spec = build_challenge_spec(controls)

        self.assertEqual(spec["fluid"]["eos_model"], "srk")
        self.assertEqual(spec["fluid"]["flow_unit"], "kg/hr")
        self.assertAlmostEqual(sum(spec["fluid"]["components"].values()), 1.0)
        self.assertEqual(spec["fluid"]["total_flow"], 112_000.0)
        self.assertEqual(
            [step["type"] for step in spec["process"]],
            [
                "stream",
                "separator",
                "compressor",
                "cooler",
                "separator",
                "compressor",
                "cooler",
            ],
        )
        self.assertEqual(
            spec["process"][2]["params"]["outlet_pressure_bara"],
            82.0,
        )
        self.assertEqual(
            spec["process"][5]["params"]["outlet_pressure_bara"],
            131.0,
        )

    def test_rejects_invalid_or_nonphysical_controls_before_native_execution(self):
        with self.assertRaisesRegex(ValueError, "Feed flow must be finite"):
            validate_controls(ChallengeControls(feed_flow_kg_hr=math.nan))
        with self.assertRaisesRegex(ValueError, "Stage 1 pressure must be between"):
            validate_controls(ChallengeControls(stage_1_pressure_bara=140.0))
        with self.assertRaisesRegex(ValueError, "Export temperature must be between"):
            validate_controls(ChallengeControls(export_temperature_c=80.0))

    def test_rejects_non_convertible_timeout_with_public_value_error(self):
        for timeout_ms in (None, "abc", True, 0, -1):
            with self.subTest(timeout_ms=timeout_ms):
                with self.assertRaisesRegex(
                    ValueError,
                    "Challenge timeout must be a positive integer",
                ):
                    run_challenge(ChallengeControls(), timeout_ms=timeout_ms)

    def test_unknown_native_validation_status_blocks_a_win(self):
        class FakeOutlet:
            def getTemperature(self, _unit):
                return 40.0

            def getPressure(self, _unit):
                return 130.0

        class FakeUnit:
            def getOutletStream(self):
                return FakeOutlet()

        class FakeModel:
            def get_unit(self, _name):
                return FakeUnit()

        result = SimpleNamespace(
            kpis={
                "total_power_kW": SimpleNamespace(value=4_300.0),
                "total_duty_kW": SimpleNamespace(value=6_500.0),
                "mass_balance_pct": SimpleNamespace(value=0.001),
                "energy_balance_pct": SimpleNamespace(value=0.002),
            },
            constraints=(
                SimpleNamespace(name="unit_balance_coverage", status="UNKNOWN"),
                SimpleNamespace(name="execution_quality", status="WARN"),
            ),
        )

        evidence = collect_evidence(
            ChallengeControls(feed_flow_kg_hr=TARGET_FLOW_KG_HR),
            FakeModel(),
            result,
        )
        assessment = assess_challenge(evidence)

        self.assertEqual(evidence.native_violations, ("unit_balance_coverage",))
        self.assertFalse(assessment.won)
        native_check = next(
            check
            for check in assessment.checks
            if check.name == "Native NeqSim checks"
        )
        self.assertFalse(native_check.passed)

    def test_winning_evidence_requires_every_engineering_check(self):
        evidence = ChallengeEvidence(
            feed_flow_kg_hr=TARGET_FLOW_KG_HR,
            export_pressure_bara=130.0,
            export_temperature_c=40.0,
            stage_1_discharge_temperature_c=90.0,
            stage_2_discharge_temperature_c=105.0,
            total_power_kw=4_300.0,
            total_cooling_duty_kw=6_500.0,
            specific_power_kwh_per_tonne=39.1,
            mass_balance_error_pct=0.001,
            energy_balance_error_pct=0.002,
            native_violations=(),
        )

        assessment = assess_challenge(evidence)

        self.assertTrue(assessment.won)
        self.assertTrue(all(check.passed for check in assessment.checks))
        self.assertGreaterEqual(assessment.score, 700)
        self.assertLessEqual(assessment.score, 1000)

    def test_failed_evidence_returns_specific_operating_guidance(self):
        evidence = ChallengeEvidence(
            feed_flow_kg_hr=100_000.0,
            export_pressure_bara=120.0,
            export_temperature_c=50.0,
            stage_1_discharge_temperature_c=155.0,
            stage_2_discharge_temperature_c=160.0,
            total_power_kw=12_000.0,
            total_cooling_duty_kw=11_000.0,
            specific_power_kwh_per_tonne=120.0,
            mass_balance_error_pct=2.0,
            energy_balance_error_pct=2.0,
            native_violations=("compressor_map.stage 2",),
        )

        assessment = assess_challenge(evidence)

        self.assertFalse(assessment.won)
        failed = {check.name for check in assessment.checks if not check.passed}
        self.assertIn("Throughput target", failed)
        self.assertIn("Export pressure", failed)
        self.assertIn("Compression power", failed)
        self.assertTrue(
            any("equal-ratio" in guidance for guidance in assessment.guidance)
        )
        self.assertTrue(
            any("invalid" in guidance for guidance in assessment.guidance)
        )


if __name__ == "__main__":
    unittest.main()
