"""Scientific checks for atmospheric, rocket, and falling-sphere dynamics."""
from dataclasses import replace
import numpy as np
from sbgea_keynote.config import MISSION, WIND_TRUTH
from sbgea_keynote.mission.dynamics import (atmosphere, flow_properties, simulate_rocket, simulate_sphere,
                     simulate_mission, sphere_rhs, neutral_wind)


def test_mesosphere_reference_density_temperature_and_regime_flags():
    rho60,t60,_=atmosphere(60_000)
    rho80,t80,_=atmosphere(80_000)
    rho100,t100,_=atmosphere(100_000)
    assert rho60>rho80>rho100>0
    assert 180<t80<220 and 180<t100<250
    p80=flow_properties(80_000,300.,MISSION.sphere_radius_m)
    p100=flow_properties(100_000,300.,MISSION.sphere_radius_m)
    assert p80["reynolds"]>0 and p80["knudsen"]<p100["knudsen"]
    assert p100["uncertain_or_unsuitable"]
    assert "transicional" in p100["regime"]


def test_drag_uses_velocity_relative_to_neutral_wind():
    state=[0.,80_000.,0.,-100.]
    calm=replace(WIND_TRUTH,speed_m_s=(0.,)*len(WIND_TRUTH.speed_m_s))
    windy=replace(WIND_TRUTH,speed_m_s=(30.,)*len(WIND_TRUTH.speed_m_s))
    a0=sphere_rhs(0,state,wind_altitude_m=calm.altitude_m,
                  wind_speed_m_s=calm.speed_m_s)
    a1=sphere_rhs(0,state,wind_altitude_m=windy.altitude_m,
                  wind_speed_m_s=windy.speed_m_s)
    assert abs(a0[2])<1e-12
    assert a1[2]>0  # the sphere is accelerated downwind when air moves past it


def test_atmospheric_density_changes_descent_dynamics():
    initial=[0.,105_000.,0.,-8.]
    low=simulate_sphere(initial,WIND_TRUTH,density_scale=.7,samples=200)
    high=simulate_sphere(initial,WIND_TRUTH,density_scale=1.3,samples=200)
    assert abs(low["duration_s"]-high["duration_s"])>0.5
    assert abs(low["x"][-1]-high["x"][-1])>1.0


def test_rocket_target_and_payload_release_initial_conditions():
    m=simulate_mission()
    r=m["rocket"]; s0=m["sphere_initial_state"]
    assert abs(r["apogee_m"]-MISSION.target_apogee_m)<10.0
    assert np.isclose(s0[0],r["release_state"][0])
    assert np.isclose(s0[1],r["release_state"][1])
    assert np.isclose(s0[2],r["release_state"][2]+MISSION.sphere_release_separation_vx_m_s)
    assert np.isclose(s0[3],r["release_state"][3]+MISSION.sphere_release_separation_vz_m_s)
    assert m["sphere"]["impact_detected"]
    assert m["sphere"]["z"][-1] <= MISSION.sphere_solver_stop_altitude_m+1e-5
