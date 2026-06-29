"""Regression checks for the fire rupture spreadsheet benchmark."""

import pytest

import fire_rupture as fr


PIPE_INPUTS = [
    ("Pipe 1", 88.9, 3.7, 3),
    ("Pipe 2", 219.1, 6.35, 8),
    ("Pipe 3", 273.1, 7.8, 10),
    ("Pipe 4", 323.9, 9.53, 12),
    ("Pipe 5", 355.6, 11.13, 14),
    ("Pipe 6", 457.2, 12.7, 18),
]

EXPECTED_RESULTS = {
    ("Pipe 1", "Large jet fire 350 kW/m2"): (True, 1.75, 33.0082, 0.005216810950826701, 22.576094735697623, 11.288047367848812, 17.397722472367846),
    ("Pipe 1", "Pool fire 250 kW/m2"): (True, 2.5833333333333335, 24.8119, 0.005216810950826701, 16.970201494557593, 8.485100747278796, 13.0776761596253),
    ("Pipe 1", "Small jet fire 250 kW/m2"): (False, None, None, 0.005216810950826701, None, None, None),
    ("Pipe 2", "Large jet fire 350 kW/m2"): (True, 2.9166666666666665, 22.9742, 0.03345871574296816, 141.33370987537032, 70.66685493768516, 84.90021031305847),
    ("Pipe 2", "Pool fire 250 kW/m2"): (True, 4.333333333333333, 15.8967, 0.03345871574296816, 91.83253479856087, 45.916267399280436, 55.16448641201176),
    ("Pipe 2", "Small jet fire 250 kW/m2"): (False, None, None, 0.03345871574296816, None, None, None),
    ("Pipe 3", "Large jet fire 350 kW/m2"): (True, 3.75, 18.0076, 0.052076806971772055, 152.80150433905723, 76.40075216952862, 86.5678938656936),
    ("Pipe 3", "Pool fire 250 kW/m2"): (True, 5.666666666666667, 13.1527, 0.052076806971772055, 111.60578567495492, 55.80289283747746, 63.22894431502856),
    ("Pipe 3", "Small jet fire 250 kW/m2"): (False, None, None, 0.052076806971772055, None, None, None),
    ("Pipe 4", "Large jet fire 350 kW/m2"): (True, 5.083333333333333, 14.1415, 0.07298502939549302, 173.9039175905992, 86.9519587952996, 95.25165081803725),
    ("Pipe 4", "Pool fire 250 kW/m2"): (True, 7.5, 10.7922, 0.07298502939549302, 125.50571034832026, 62.75285517416013, 68.7427072569535),
    ("Pipe 4", "Small jet fire 250 kW/m2"): (False, None, None, 0.07298502939549302, None, None, None),
    ("Pipe 5", "Large jet fire 350 kW/m2"): (True, 6.333333333333333, 11.9327, 0.08726995329312706, 165.9293376091388, 82.9646688045694, 89.75342576380818),
    ("Pipe 5", "Pool fire 250 kW/m2"): (True, 9.25, 9.8993, 0.08726995329312706, 137.6540340236617, 68.82701701183085, 74.45893114413892),
    ("Pipe 5", "Small jet fire 250 kW/m2"): (False, None, None, 0.08726995329312706, None, None, None),
    ("Pipe 6", "Large jet fire 350 kW/m2"): (True, 7.083333333333333, 11.2163, 0.14643846145917686, 261.7125131726678, 130.8562565863339, 139.2370189786576),
    ("Pipe 6", "Pool fire 250 kW/m2"): (True, 10.083333333333334, 9.4985, 0.14643846145917686, 221.63068983270642, 110.81534491635321, 117.91257587339669),
    ("Pipe 6", "Small jet fire 250 kW/m2"): (False, None, None, 0.14643846145917686, None, None, None),
}


def _build_benchmark_pipe(name: str, od_mm: float, wall_mm: float, nominal_inch: float) -> fr.Pipe:
    return fr.Pipe(
        name=name,
        od_mm=od_mm,
        wall_mm=wall_mm,
        material="22Cr duplex",
        corrosion_allowance_mm=0.0,
        wall_tolerance_frac=0.125,
        weight_stress_MPa=0.0,
        fluid_density=23.7487037155012,
        fluid_heat_capacity=2283.35469905934,
        gas_mw=18.2,
        nominal_inch=nominal_inch,
    )


@pytest.mark.parametrize("pipe_args", PIPE_INPUTS)
def test_default_fire_rupture_benchmark(pipe_args):
    profile_time_min, profile_pressure_bara = zip(*fr.DEFAULT_PRESSURE_PROFILE)
    pipe = _build_benchmark_pipe(*pipe_args)

    result = fr.evaluate_pipe(
        pipe,
        fr.default_fires(),
        profile_time_min,
        profile_pressure_bara,
        time_step_s=5.0,
        initial_temp_C=20.0,
    )

    for fire_result in result.fires:
        expected = EXPECTED_RESULTS[(pipe.name, fire_result.fire_name)]
        ruptured, time_min, pressure_barg, area_m2, gas_2s, gas_1s, gas_short = expected

        assert fire_result.ruptured is ruptured
        assert pipe.release_cross_area_m2 == pytest.approx(area_m2, rel=1e-12, abs=1e-12)

        if not ruptured:
            assert fire_result.time_to_rupture_min is None
            assert fire_result.rupture_pressure_barg is None
            assert fire_result.release_gas_2sides is None
            assert fire_result.release_gas_1side is None
            assert fire_result.release_gas_short is None
            assert fire_result.release_liquid is None
            continue

        assert fire_result.time_to_rupture_min == pytest.approx(time_min, rel=1e-12, abs=1e-12)
        assert fire_result.rupture_pressure_barg == pytest.approx(pressure_barg, rel=1e-12, abs=1e-12)
        assert fire_result.release_gas_2sides == pytest.approx(gas_2s, rel=1e-12, abs=1e-12)
        assert fire_result.release_gas_1side == pytest.approx(gas_1s, rel=1e-12, abs=1e-12)
        assert fire_result.release_gas_short == pytest.approx(gas_short, rel=1e-12, abs=1e-12)
        assert fire_result.release_liquid is None