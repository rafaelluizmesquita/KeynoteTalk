"""All physical, timing, and render parameters for From a Rock to Space."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class PhysicsConfig:
    # SI units
    g: float = 9.81                         # m s^-2
    launch_speed: float = 25.0              # m s^-1
    launch_angle_deg: float = 45.0          # degrees above horizontal
    launch_height: float = 1.7              # m
    rock_diameter: float = 0.12             # m, equivalent spherical diameter
    rock_density: float = 2600.0            # kg m^-3
    rock_cd: float = 0.90                   # dimensionless
    rho0: float = 1.225                     # kg m^-3, sea-level standard atmosphere
    scale_height: float = 8500.0            # m, simplified isothermal model
    wind_u0: float = 4.0                    # m s^-1, positive downrange
    wind_shear: float = 0.08                # s^-1; du/dy
    projectile_max_time: float = 300.0      # s, integration guard
    # A separate, explicitly illustrative heavier study projectile for the
    # high-altitude inset; it is not the hand-thrown rock.
    study_speed: float = 250.0              # m s^-1
    study_angle_deg: float = 45.0           # degrees
    study_mass: float = 100.0               # kg
    study_diameter: float = 0.20             # m
    study_cd: float = 0.80
    study_max_time: float = 600.0           # s, integration guard
    # Earth and orbital parameters
    earth_radius: float = 6_371_000.0       # m
    earth_mu: float = 3.986004418e14        # m^3 s^-2
    earth_omega: float = 7.2921159e-5       # rad s^-1
    orbit_start_altitude: float = 300_000.0 # m
    suborbital_speed: float = 7_500.0       # m s^-1, tangential, impacts Earth
    escape_speed_factor: float = 1.035      # factor times local escape speed
    suborbital_duration: float = 5_000.0    # s, integration horizon
    circular_duration: float = 6_000.0      # s, integration horizon
    escape_duration: float = 5_000.0        # s, integration horizon
    satellite_altitude: float = 450_000.0  # m
    satellite_mass: float = 150.0           # kg
    satellite_area: float = 8.0             # m^2, deployable high-area spacecraft
    satellite_cd: float = 2.20
    # Illustrative thermosphere, deliberately labeled non-operational.
    thermo_rho_ref: float = 2.0e-12         # kg m^-3 at reference altitude
    thermo_reference_altitude: float = 450_000.0 # m
    thermo_scale_height: float = 60_000.0   # m
    satellite_integration_days: float = 15.0
    visual_radial_exaggeration: float = 10.0

@dataclass(frozen=True)
class RenderConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    dpi: int = 96
    # Cumulative boundaries in seconds; 90 s total.
    stage_boundaries: tuple = (0.0, 14.0, 27.0, 40.0, 54.0, 75.0, 90.0)
    scale_transition_start: float = 54.0
    scale_transition_end: float = 58.0
    transparent_frames: int = 90

PHYSICS = PhysicsConfig()
RENDER = RenderConfig()


def rock_mass(config: PhysicsConfig = PHYSICS) -> float:
    return (4.0 / 3.0) * math.pi * (config.rock_diameter / 2.0) ** 3 * config.rock_density


@dataclass(frozen=True)
class MissionConfig:
    """Configurable mission physics. All quantities use SI units."""
    earth_radius_m: float = 6_371_000.0
    surface_gravity_m_s2: float = 9.80665
    earth_mu_m3_s2: float = 3.986004418e14
    gas_constant_air_j_kg_k: float = 287.05
    boltzmann_j_k: float = 1.380649e-23
    molecular_diameter_m: float = 3.70e-10
    viscosity_ref_pa_s: float = 1.716e-5
    viscosity_ref_temp_k: float = 273.15
    sutherland_temp_k: float = 110.4
    # Approximate US Standard Atmosphere 1976 reference nodes. Density is
    # interpolated log-linearly; temperature linearly. Above 110 km the profile
    # is extrapolated only for numerical guards, not as an operational model.
    atmosphere_altitude_m: tuple = (0, 11_000, 20_000, 32_000, 47_000, 51_000,
        60_000, 70_000, 80_000, 90_000, 100_000, 110_000)
    atmosphere_density_kg_m3: tuple = (1.225, 0.3639, 0.08803, 0.01322,
        0.001427, 0.000862, 3.096e-4, 8.283e-5, 1.846e-5, 3.416e-6,
        5.604e-7, 9.708e-8)
    atmosphere_temperature_k: tuple = (288.15, 216.65, 216.65, 228.65, 270.65,
        270.65, 247.02, 219.58, 198.64, 186.87, 195.08, 240.0)
    # Falling sphere: configurable shell mass/radius, drag coefficient is
    # computed from Re and Kn using an explicitly approximate continuum/FM blend.
    sphere_mass_kg: float = 0.10
    sphere_radius_m: float = 0.25
    sphere_free_molecular_cd: float = 2.0
    sphere_release_separation_vx_m_s: float = 0.0
    sphere_release_separation_vz_m_s: float = -8.0
    target_apogee_m: float = 105_000.0
    release_altitude_m: float = 105_000.0
    observation_stop_altitude_m: float = 60_000.0
    sphere_solver_stop_altitude_m: float = 50_000.0
    atmosphere_true_density_scale: float = 1.0
    # Simple vertical rocket. Thrust is calibrated to the requested apogee,
    # with variable mass, gravity, and axial atmospheric drag.
    rocket_initial_mass_kg: float = 800.0
    rocket_dry_mass_kg: float = 500.0
    rocket_burn_time_s: float = 30.0
    rocket_thrust_seed_n: float = 50_000.0
    rocket_thrust_bounds_n: tuple = (10_000.0, 150_000.0)
    rocket_reference_area_m2: float = 0.08
    rocket_cd: float = 0.50
    rocket_solver_max_step_s: float = 0.25
    sphere_solver_max_step_s: float = 0.50


@dataclass(frozen=True)
class WindProfile:
    altitude_m: tuple = (60_000., 70_000., 80_000., 90_000., 100_000., 110_000.)
    speed_m_s: tuple = (15., 30., 45., 20., -5., -20.)


@dataclass(frozen=True)
class TrackingConfig:
    # 2-D vertical plane radar sites; can add sites for triangulation.
    station_x_m: tuple = (-5_000.,)
    cadence_s: float = 1.0
    range_sigma_m: float = 2.0
    elevation_sigma_deg: float = 0.001
    range_elevation_correlation: float = 0.0
    missing_fraction: float = 0.06
    random_seed: int = 2026


@dataclass(frozen=True)
class RetrievalConfig:
    wind_knot_altitudes_m: tuple = (65_000., 75_000., 85_000., 95_000., 105_000.)
    wind_prior_sigma_m_s: float = 100.0
    curvature_prior_sigma_m_s: float = 45.0
    log_density_prior_sigma: float = 0.25
    model_error_floor_m_s: float = 8.0
    release_position_prior_sigma_m: float = 10.0
    release_velocity_prior_sigma_m_s: float = 2.0
    max_function_evaluations: int = 45
    finite_difference_step: float = 0.01


@dataclass(frozen=True)
class MissionRenderConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    dpi: int = 96
    # Eight acts, 110 s total; the first 30 s cover mechanics, drag, and wind.
    stage_durations_s: tuple = (15., 8., 7., 13., 17., 18., 19., 13.)
    keyframe_dpi: int = 180
    mesosphere_reference_altitude_m: float = 80_000.0
    mesospheric_view_min_altitude_m: float = 55_000.0
    mesospheric_view_max_altitude_m: float = 112_000.0
    # Preview the first 14% of the observed fall during the rocket/release act;
    # the tracking act continues from this exact state to the observation stop.
    sphere_descent_preview_fraction: float = 0.14
    release_zoom_start_fraction: float = 0.66
    release_zoom_end_fraction: float = 0.84
    release_camera_min_altitude_m: float = 90_000.0
    release_camera_max_altitude_m: float = 108_000.0


@dataclass(frozen=True)
class FabryPerotConfig:
    """Independent idealized Fabry--Pérot emission-line instrument model."""
    line_center_m: float = 630.0304e-9       # atomic oxygen red-line example
    emitting_species_mass_kg: float = 16.0 * 1.66053906660e-27
    gas_temperature_k: float = 200.0
    cavity_spacing_m: float = 0.01575
    finesse: float = 25.0
    focal_length_m: float = 0.05
    detector_pixels: int = 48
    pixel_pitch_m: float = 13.0e-6
    photon_scale_counts: float = 1600.0
    background_counts: float = 80.0
    read_noise_e: float = 3.0
    integration_wavelength_bins: int = 301
    random_seed: int = 6302026


MISSION = MissionConfig()
WIND_TRUTH = WindProfile()
TRACKING = TrackingConfig()
RETRIEVAL = RetrievalConfig()
MISSION_RENDER = MissionRenderConfig()
FPI = FabryPerotConfig()
