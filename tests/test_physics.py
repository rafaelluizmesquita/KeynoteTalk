"""Numerical checks for the reusable physics models; run with pytest -q."""
import math
import numpy as np
from sbgea_keynote.config import PHYSICS
from sbgea_keynote.physics.projectile import (ideal_projectile, simulate_projectile, orbital_rhs,
                    specific_energy, integrate_orbit, orbital_examples,
                    simulate_satellite)


def test_ideal_projectile_matches_analytic_solution():
    exact = ideal_projectile(samples=801)
    numerical = simulate_projectile(drag=False, samples=801)
    assert numerical["impact_detected"]
    assert abs(exact["range"] - numerical["range"]) < 1e-6
    assert abs(exact["flight_time"] - numerical["flight_time"]) < 1e-7
    assert np.max(np.abs(exact["y"] - numerical["y"])) < 1e-6


def test_drag_integrator_converges_as_maximum_step_is_reduced():
    coarse=simulate_projectile(drag=True,samples=300,rtol=1e-9,atol=1e-10,max_step=.5)
    medium=simulate_projectile(drag=True,samples=300,rtol=1e-9,atol=1e-10,max_step=.1)
    fine=simulate_projectile(drag=True,samples=300,rtol=1e-10,atol=1e-11,max_step=.02)
    assert abs(fine["range"]-medium["range"]) < abs(coarse["range"]-medium["range"])
    assert abs(fine["flight_time"]-medium["flight_time"]) < \
           abs(coarse["flight_time"]-medium["flight_time"])


def test_drag_dissipates_energy_in_still_air():
    result = simulate_projectile(drag=True, wind="calm", samples=1200)
    e = 0.5 * (result["vx"]**2 + result["vy"]**2) + PHYSICS.g * result["y"]
    assert np.max(np.diff(e)) < 2e-4  # solver/interpolation round-off allowance
    assert e[-1] < e[0] - 1.0


def test_impact_event_stops_at_ground():
    result = simulate_projectile(drag=True, samples=100)
    assert result["impact_detected"]
    assert result["flight_time"] > 0
    assert result["y"][-1] == 0.0
    assert result["solver"].t[-1] < 10.0


def test_orbital_specific_energy_is_conserved():
    r0 = PHYSICS.earth_radius + 400_000.0
    v0 = math.sqrt(PHYSICS.earth_mu / r0)
    sol = integrate_orbit([0.0, r0, -v0, 0.0], 2.0 * math.pi * math.sqrt(r0**3 / PHYSICS.earth_mu), samples=600)
    energies = np.array([specific_energy(s) for s in np.column_stack(
        [sol["x"], sol["y"], sol["vx"], sol["vy"]])])
    assert np.ptp(energies) / abs(energies[0]) < 2e-9


def test_orbit_and_satellite_examples_are_finite_and_drag_lowers_altitude():
    examples = orbital_examples()
    assert examples["suborbital"]["impact_detected"]
    assert not examples["circular"]["impact_detected"]
    with_drag = simulate_satellite(days=2.0, samples=500, drag=True)
    vacuum = simulate_satellite(days=2.0, samples=500, drag=False)
    assert with_drag["final_altitude"] < with_drag["initial_altitude"]
    assert abs(vacuum["final_altitude"] - vacuum["initial_altitude"]) < 2.0
