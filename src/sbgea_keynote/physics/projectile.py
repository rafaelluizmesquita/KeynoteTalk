"""Reusable projectile, atmosphere, drag, and orbital dynamics (SI units)."""
from __future__ import annotations
import math
import numpy as np
from scipy.integrate import solve_ivp
from ..config import PHYSICS, PhysicsConfig, rock_mass


def launch_components(speed: float, angle_deg: float) -> tuple[float, float]:
    """Return horizontal and vertical launch velocity components in m/s."""
    a = math.radians(angle_deg)
    return speed * math.cos(a), speed * math.sin(a)


def ideal_projectile(speed=PHYSICS.launch_speed, angle_deg=PHYSICS.launch_angle_deg,
                     height=PHYSICS.launch_height, g=PHYSICS.g, samples=600):
    """Analytic flat-Earth projectile solution; returns t, x, y, vx, vy in SI."""
    vx0, vy0 = launch_components(speed, angle_deg)
    tf = (vy0 + math.sqrt(vy0 * vy0 + 2 * g * height)) / g
    t = np.linspace(0.0, tf, samples)
    x = vx0 * t
    y = height + vy0 * t - 0.5 * g * t * t
    vx = np.full_like(t, vx0)
    vy = vy0 - g * t
    return {"t": t, "x": x, "y": y, "vx": vx, "vy": vy,
            "range": float(x[-1]), "flight_time": float(tf)}


def air_density(altitude_m, config: PhysicsConfig = PHYSICS, model="exponential"):
    """Air density in kg/m^3; exponential option is a simplified isothermal model."""
    h = np.maximum(np.asarray(altitude_m, dtype=float), 0.0)
    if model == "constant":
        return np.zeros_like(h) + config.rho0
    if model == "exponential":
        return config.rho0 * np.exp(-h / config.scale_height)
    raise ValueError(f"Unknown atmosphere model: {model}")


def wind_speed(altitude_m, config: PhysicsConfig = PHYSICS, kind="calm"):
    """Horizontal wind in m/s, positive downrange."""
    h = np.maximum(np.asarray(altitude_m, dtype=float), 0.0)
    if kind == "calm":
        return np.zeros_like(h)
    if kind == "constant":
        return np.zeros_like(h) + config.wind_u0
    if kind == "shear":
        return config.wind_u0 + config.wind_shear * h
    raise ValueError(f"Unknown wind profile: {kind}")


def projectile_rhs(t, state, *, config=PHYSICS, drag=False, density="constant",
                   wind="calm", mass=None, diameter=None, cd=None, gravity=None):
    """2-D equations for [x, y, vx, vy]. Drag acts on air-relative velocity."""
    x, y, vx, vy = state
    g = config.g if gravity is None else gravity
    ax, ay = 0.0, -g
    if drag:
        m = rock_mass(config) if mass is None else mass
        d = config.rock_diameter if diameter is None else diameter
        coeff = config.rock_cd if cd is None else cd
        area = math.pi * (d / 2.0) ** 2
        rho = float(air_density(y, config, density))
        rel = np.array([vx - float(wind_speed(y, config, wind)), vy])
        accel = -0.5 * rho * coeff * area / m * np.linalg.norm(rel) * rel
        ax += accel[0]
        ay += accel[1]
    return [vx, vy, ax, ay]


def simulate_projectile(*, speed=PHYSICS.launch_speed, angle_deg=PHYSICS.launch_angle_deg,
                        height=PHYSICS.launch_height, config=PHYSICS, drag=False,
                        density="constant", wind="calm", mass=None, diameter=None,
                        cd=None, gravity=None, max_time=None, samples=600,
                        rtol=1e-10, atol=1e-11, max_step=0.05):
    """Integrate a projectile until impact; impact event is y=0, downward only."""
    vx0, vy0 = launch_components(speed, angle_deg)
    max_time = config.projectile_max_time if max_time is None else max_time
    def impact(t, state):
        return state[1]
    impact.terminal = True
    impact.direction = -1
    sol = solve_ivp(
        lambda t, s: projectile_rhs(t, s, config=config, drag=drag,
                                    density=density, wind=wind, mass=mass,
                                    diameter=diameter, cd=cd, gravity=gravity),
        (0.0, max_time), [0.0, height, vx0, vy0], events=impact,
        dense_output=True, rtol=rtol, atol=atol, max_step=max_step,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    tf = float(sol.t_events[0][0]) if len(sol.t_events[0]) else float(sol.t[-1])
    t = np.linspace(0.0, tf, samples)
    z = sol.sol(t)
    return {"t": t, "x": z[0], "y": np.maximum(z[1], 0), "vx": z[2],
            "vy": z[3], "range": float(z[0, -1]), "flight_time": tf,
            "solver": sol, "impact_detected": bool(len(sol.t_events[0]))}


def study_projectile(config=PHYSICS, *, density="constant", samples=600):
    """Higher-energy, 100 kg study projectile used only for the stage-4 inset."""
    return simulate_projectile(speed=config.study_speed,
        angle_deg=config.study_angle_deg, height=0.0, config=config, drag=True,
        density=density, wind="calm", mass=config.study_mass,
        diameter=config.study_diameter, cd=config.study_cd,
        max_time=config.study_max_time, samples=samples)


def orbital_rhs(t, state, *, mu=PHYSICS.earth_mu):
    """Newtonian two-body ECI acceleration for [x, y, vx, vy], SI units."""
    x, y, vx, vy = state
    r = math.hypot(x, y)
    ax, ay = -mu * x / r**3, -mu * y / r**3
    return [vx, vy, ax, ay]


def specific_energy(state, mu=PHYSICS.earth_mu):
    """Specific mechanical energy, J/kg."""
    x, y, vx, vy = state
    return 0.5 * (vx * vx + vy * vy) - mu / math.hypot(x, y)


def integrate_orbit(initial_state, duration, *, config=PHYSICS, samples=2000,
                    impact_radius=None, rtol=2e-10, atol=1e-8, max_step=None):
    """Integrate 2-D central gravity; optionally stop at a planetary surface."""
    event = None
    if impact_radius is not None:
        def event(t, state):
            return math.hypot(state[0], state[1]) - impact_radius
        event.terminal = True
        event.direction = -1
    kw = {"rtol": rtol, "atol": atol, "dense_output": True}
    if max_step is not None:
        kw["max_step"] = max_step
    sol = solve_ivp(lambda t, s: orbital_rhs(t, s, mu=config.earth_mu),
                    (0.0, duration), initial_state, events=event, **kw)
    if not sol.success:
        raise RuntimeError(sol.message)
    tf = float(sol.t_events[0][0]) if event is not None and len(sol.t_events[0]) else float(sol.t[-1])
    t = np.linspace(0.0, tf, samples)
    z = sol.sol(t)
    return {"t": t, "x": z[0], "y": z[1], "vx": z[2], "vy": z[3],
            "solver": sol, "impact_detected": bool(event is not None and len(sol.t_events[0]))}


def orbital_examples(config=PHYSICS):
    """Suborbital, circular, and escape trajectories from 300 km altitude."""
    r0 = config.earth_radius + config.orbit_start_altitude
    vc = math.sqrt(config.earth_mu / r0)
    vesc = math.sqrt(2.0) * vc
    # At the north point, -x is the local tangent in the co-rotating direction.
    initial = lambda speed: [0.0, r0, -speed, 0.0]
    sub = integrate_orbit(initial(config.suborbital_speed), config.suborbital_duration,
                          config=config, samples=700,
                          impact_radius=config.earth_radius, max_step=10.0)
    circular = integrate_orbit(initial(vc), config.circular_duration, config=config,
                               samples=1000, max_step=15.0)
    escape = integrate_orbit(initial(config.escape_speed_factor * vesc),
                             config.escape_duration, config=config, samples=800, max_step=12.0)
    return {"suborbital": sub, "circular": circular, "escape": escape,
            "circular_speed": vc, "escape_speed": vesc}


def satellite_rhs(t, state, *, config=PHYSICS, drag=True):
    """Satellite acceleration: central gravity plus illustrative thermo drag.

    The atmosphere rotates rigidly with Earth. Density is an exponential fit at
    high altitude for demonstration only, not an operational thermosphere model.
    """
    x, y, vx, vy = state
    r = math.hypot(x, y)
    ax, ay = -config.earth_mu * x / r**3, -config.earth_mu * y / r**3
    if drag:
        altitude = r - config.earth_radius
        rho = config.thermo_rho_ref * math.exp(
            -(altitude - config.thermo_reference_altitude) / config.thermo_scale_height)
        # omega x r for positive z-axis Earth rotation.
        air_vx, air_vy = -config.earth_omega * y, config.earth_omega * x
        rvx, rvy = vx - air_vx, vy - air_vy
        vrel = math.hypot(rvx, rvy)
        factor = -0.5 * rho * config.satellite_cd * config.satellite_area / config.satellite_mass * vrel
        ax += factor * rvx
        ay += factor * rvy
    return [vx, vy, ax, ay]


def simulate_satellite(config=PHYSICS, *, days=None, samples=5000, drag=True):
    """Integrate an illustrative LEO satellite for a disclosed long interval."""
    days = config.satellite_integration_days if days is None else days
    r0 = config.earth_radius + config.satellite_altitude
    vc = math.sqrt(config.earth_mu / r0)
    # At the top of Earth, -x is tangential and matches co-rotation.
    initial = [0.0, r0, -vc, 0.0]
    duration = days * 86400.0
    sol = solve_ivp(lambda t, s: satellite_rhs(t, s, config=config, drag=drag),
        (0.0, duration), initial, method="DOP853", rtol=2e-9, atol=1e-5,
        dense_output=True, max_step=90.0)
    if not sol.success:
        raise RuntimeError(sol.message)
    t = np.linspace(0.0, duration, samples)
    z = sol.sol(t)
    alt = np.hypot(z[0], z[1]) - config.earth_radius
    return {"t": t, "x": z[0], "y": z[1], "vx": z[2], "vy": z[3],
            "altitude": alt, "solver": sol, "initial_altitude": config.satellite_altitude,
            "final_altitude": float(alt[-1]), "days": days}
