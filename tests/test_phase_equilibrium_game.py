"""Tests for the native phase-equilibrium and fluid-property game."""

import math
import unittest

from process_chat.phase_equilibrium_game import (
    RICH_GAS_COMPOSITION,
    ComponentEquilibrium,
    PhaseControls,
    PhaseEvidence,
    assess_phase_evidence,
    run_phase_challenge,
    validate_controls,
)


def _winning_evidence(**overrides) -> PhaseEvidence:
    values = {
        "temperature_c": 50.0,
        "pressure_bara": 80.0,
        "phase_types": ("gas", "oil"),
        "gas_fraction_mol_pct": 83.1,
        "liquid_fraction_mol_pct": 16.9,
        "gas_density_kg_m3": 79.5,
        "liquid_density_kg_m3": 499.5,
        "gas_z_factor": 0.819,
        "gas_viscosity_cp": 0.015,
        "liquid_viscosity_cp": 0.105,
        "mixture_enthalpy_kj_kg": -77.2,
        "mixture_cp_kj_kgk": 2.8,
        "phase_fraction_closure_error": 0.0,
        "components": (),
    }
    values.update(overrides)
    return PhaseEvidence(**values)


class PhaseEquilibriumGameTest(unittest.TestCase):
    """Protect controls, evidence integrity, and the calibrated native window."""

    def test_training_fluid_is_normalized_and_reproducible(self):
        self.assertAlmostEqual(sum(RICH_GAS_COMPOSITION.values()), 1.0)
        self.assertEqual(RICH_GAS_COMPOSITION["methane"], 0.720)
        self.assertIn("n-octane", RICH_GAS_COMPOSITION)

    def test_rejects_invalid_controls_before_native_execution(self):
        with self.assertRaisesRegex(ValueError, "Temperature must be finite"):
            validate_controls(PhaseControls(temperature_c=math.nan))
        with self.assertRaisesRegex(ValueError, "Pressure must be between"):
            validate_controls(PhaseControls(pressure_bara=500.0))
        with self.assertRaisesRegex(ValueError, "must use PhaseControls"):
            validate_controls(object())

    def test_rejects_invalid_timeout_before_native_execution(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            run_phase_challenge(PhaseControls(), timeout_ms=1.5)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            run_phase_challenge(PhaseControls(), timeout_ms="not-a-timeout")

    def test_every_native_property_target_is_required_to_win(self):
        assessment = assess_phase_evidence(_winning_evidence())

        self.assertTrue(assessment.won)
        self.assertEqual(assessment.score, 1000)
        self.assertTrue(all(check.passed for check in assessment.checks))

        missing_property = assess_phase_evidence(
            _winning_evidence(gas_z_factor=None)
        )
        self.assertFalse(missing_property.won)
        self.assertLess(missing_property.score, 1000)
        self.assertIn(
            "Gas compressibility",
            {check.name for check in missing_property.checks if not check.passed},
        )

    def test_wrong_phase_state_blocks_otherwise_matching_properties(self):
        assessment = assess_phase_evidence(
            _winning_evidence(
                phase_types=("gas",),
                liquid_fraction_mol_pct=None,
            )
        )

        self.assertFalse(assessment.won)
        self.assertTrue(any("single-phase gas" in item for item in assessment.guidance))

    def test_native_winning_window_and_starting_point_failure(self):
        winning = run_phase_challenge(PhaseControls(50.0, 80.0))
        starting = run_phase_challenge(PhaseControls(20.0, 50.0))

        self.assertTrue(winning.assessment.won)
        self.assertEqual(winning.assessment.score, 1000)
        self.assertFalse(starting.assessment.won)
        self.assertEqual(winning.evidence.phase_types, ("gas", "oil"))
        self.assertLessEqual(
            winning.evidence.phase_fraction_closure_error,
            1.0e-10,
        )
        self.assertEqual(len(winning.evidence.components), len(RICH_GAS_COMPOSITION))
        methane = next(
            row for row in winning.evidence.components if row.component == "methane"
        )
        octane = next(
            row for row in winning.evidence.components if row.component == "n-octane"
        )
        self.assertIsInstance(methane, ComponentEquilibrium)
        self.assertGreater(methane.k_value, 1.0)
        self.assertLess(octane.k_value, 1.0)


if __name__ == "__main__":
    unittest.main()
