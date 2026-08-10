"""Regression tests for the Pipeline page's native NeqSim calculation paths."""

from __future__ import annotations

import math
import unittest

from fluids import default_fluid
from pipeline_hydraulics import (
    PipelineInputError,
    build_beggs_brill_pipe,
    build_two_fluid_pipe,
    fluid_from_preset,
    interpolate_section_elevations,
    normalize_elevation_profile,
    normalize_fluid_composition,
    read_beggs_brill_profiles,
    read_two_fluid_profiles,
    solve_inlet_pressure,
)


class PipelineInputValidationTest(unittest.TestCase):
    """Keep invalid fluid and terrain data out of the native solvers."""

    def test_pipeline_presets_are_normalized_and_independent(self):
        lean = fluid_from_preset(default_fluid, "Lean natural gas")
        rich = fluid_from_preset(default_fluid, "Two-phase gas condensate")

        self.assertAlmostEqual(lean["MolarComposition[-]"].sum(), 1.0)
        self.assertAlmostEqual(rich["MolarComposition[-]"].sum(), 1.0)
        lean.loc[lean["ComponentName"] == "methane", "MolarComposition[-]"] = 0.0
        self.assertAlmostEqual(rich["MolarComposition[-]"].sum(), 1.0)

    def test_composition_is_normalized_without_changing_source(self):
        fluid = fluid_from_preset(default_fluid, "Lean natural gas")
        fluid["MolarComposition[-]"] *= 2.0

        normalized = normalize_fluid_composition(fluid)

        self.assertAlmostEqual(normalized["MolarComposition[-]"].sum(), 1.0)
        self.assertAlmostEqual(fluid["MolarComposition[-]"].sum(), 2.0)

    def test_blank_and_negative_compositions_fail_before_neqsim(self):
        blank = fluid_from_preset(default_fluid, "Blank / custom")
        with self.assertRaisesRegex(PipelineInputError, "Enter a fluid composition"):
            normalize_fluid_composition(blank)

        blank.loc[blank["ComponentName"] == "methane", "MolarComposition[-]"] = -1.0
        with self.assertRaisesRegex(PipelineInputError, "cannot be negative"):
            normalize_fluid_composition(blank)

    def test_profile_is_shifted_to_zero_and_interpolated_at_section_centres(self):
        distances, elevations = normalize_elevation_profile(
            [100.0, 200.0, 300.0],
            [10.0, 20.0, 10.0],
        )
        section_elevations = interpolate_section_elevations(
            distances,
            elevations,
            4,
        )

        self.assertEqual(distances, (0.0, 100.0, 200.0))
        self.assertEqual(elevations, (10.0, 20.0, 10.0))
        self.assertEqual(section_elevations, (12.5, 17.5, 17.5, 12.5))

    def test_unsorted_or_duplicate_profile_distances_are_rejected(self):
        for distances in ([0.0, 100.0, 50.0], [0.0, 100.0, 100.0]):
            with self.subTest(distances=distances):
                with self.assertRaisesRegex(PipelineInputError, "strictly increasing"):
                    normalize_elevation_profile(distances, [0.0, 10.0, 20.0])


class PressureSolverTest(unittest.TestCase):
    """Verify bracketing for ordinary pressure loss and downhill pressure gain."""

    class _Pipe:
        def __init__(self, outlet_pressure):
            self._outlet_pressure = outlet_pressure

        def getOutletPressure(self):
            return self._outlet_pressure

    @staticmethod
    def _negative_outlet_error():
        return RuntimeError("Outlet pressure is negative - output pressure out")

    def test_solver_treats_native_negative_outlet_trial_as_lower_bound(self):
        def build(inlet_pressure):
            if inlet_pressure < 90.0:
                raise self._negative_outlet_error()
            return self._Pipe(inlet_pressure - 40.0), object()

        result = solve_inlet_pressure(build, 60.0, tolerance_bar=1.0e-6)

        self.assertAlmostEqual(result.inlet_pressure_bara, 100.0, places=5)
        self.assertAlmostEqual(result.outlet_pressure_bara, 60.0, places=5)

    def test_solver_searches_below_target_for_downhill_pressure_gain(self):
        def build(inlet_pressure):
            return self._Pipe(inlet_pressure + 10.0), object()

        result = solve_inlet_pressure(build, 50.0, tolerance_bar=1.0e-6)

        self.assertAlmostEqual(result.inlet_pressure_bara, 40.0, places=5)
        self.assertAlmostEqual(result.outlet_pressure_bara, 50.0, places=5)

    def test_unrelated_native_error_is_not_hidden_by_bracketing(self):
        def build(_inlet_pressure):
            raise RuntimeError("component database unavailable")

        with self.assertRaisesRegex(RuntimeError, "component database unavailable"):
            solve_inlet_pressure(build, 50.0)


try:
    import neqsim  # noqa: F401

    NEQSIM_AVAILABLE = True
except ImportError:
    NEQSIM_AVAILABLE = False


@unittest.skipUnless(NEQSIM_AVAILABLE, "native NeqSim package is not installed")
class NativePipelineRegressionTest(unittest.TestCase):
    """Exercise both corrected Java models with gas and gas-condensate fluids."""

    def test_beggs_brill_uses_native_profiles_and_nearby_flow_trend(self):
        fluid = fluid_from_preset(default_fluid, "Lean natural gas")

        base_pipe, _ = build_beggs_brill_pipe(
            fluid,
            inlet_pressure_bara=60.0,
            inlet_temperature_c=40.0,
            mass_flow_kg_s=5_000.0 / 3_600.0,
            length_m=2_000.0,
            diameter_m=0.2,
            roughness_m=50.0e-6,
            elevation_m=0.0,
            number_of_increments=10,
            heat_transfer_coefficient_w_m2_k=0.0,
            ambient_temperature_c=5.0,
        )
        nearby_pipe, _ = build_beggs_brill_pipe(
            fluid,
            inlet_pressure_bara=60.0,
            inlet_temperature_c=40.0,
            mass_flow_kg_s=5_250.0 / 3_600.0,
            length_m=2_000.0,
            diameter_m=0.2,
            roughness_m=50.0e-6,
            elevation_m=0.0,
            number_of_increments=10,
            heat_transfer_coefficient_w_m2_k=0.0,
            ambient_temperature_c=5.0,
        )
        profiles = read_beggs_brill_profiles(base_pipe)

        self.assertEqual(len(profiles.position_km), 11)
        self.assertAlmostEqual(profiles.position_km[0], 0.0)
        self.assertAlmostEqual(profiles.position_km[-1], 2.0)
        self.assertTrue(all(math.isfinite(value) for value in profiles.pressure_bara))
        self.assertTrue(all(value >= 0.0 for value in profiles.liquid_holdup))
        self.assertGreater(base_pipe.getPressureDrop(), 0.0)
        self.assertGreater(nearby_pipe.getPressureDrop(), base_pipe.getPressureDrop())
        self.assertAlmostEqual(
            base_pipe.getOutletStream().getFlowRate("kg/hr"),
            5_000.0,
            delta=1.0e-6,
        )

    def test_two_fluid_multiphase_profiles_and_transient_remain_finite(self):
        fluid = fluid_from_preset(default_fluid, "Two-phase gas condensate")
        pipe, _ = build_two_fluid_pipe(
            fluid,
            inlet_pressure_bara=60.0,
            inlet_temperature_c=40.0,
            mass_flow_kg_hr=5_000.0,
            diameter_m=0.2,
            roughness_m=50.0e-6,
            distances_m=[0.0, 500.0, 1_000.0],
            elevations_m=[0.0, -10.0, 0.0],
            number_of_sections=20,
            heat_transfer_coefficient_w_m2_k=5.0,
            ambient_temperature_c=5.0,
            enable_slug_tracking=True,
        )
        steady_profiles = read_two_fluid_profiles(pipe)

        self.assertEqual(len(steady_profiles.position_km), 20)
        self.assertTrue(
            all(0.0 <= value <= 1.0 for value in steady_profiles.liquid_holdup)
        )
        self.assertGreater(max(steady_profiles.liquid_holdup), 0.0)
        self.assertGreater(pipe.getInletPressure(), pipe.getOutletPressure())
        self.assertAlmostEqual(
            pipe.getOutletFlowRate("kg/hr"),
            5_000.0,
            delta=1.0e-6,
        )

        pipe.runTransient(1.0)
        transient_profiles = read_two_fluid_profiles(pipe)
        self.assertEqual(len(transient_profiles.position_km), 20)
        self.assertTrue(
            all(math.isfinite(value) for value in transient_profiles.pressure_bara)
        )
        self.assertTrue(
            all(math.isfinite(value) for value in transient_profiles.temperature_c)
        )


if __name__ == "__main__":
    unittest.main()
