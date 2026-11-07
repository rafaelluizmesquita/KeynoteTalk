"""Tests for the radar observation operator and deterministic noise."""
from dataclasses import replace
import numpy as np
from sbgea_keynote.config import TRACKING, MISSION, WIND_TRUTH
from sbgea_keynote.mission.dynamics import simulate_mission
from sbgea_keynote.mission.observations import (observation_operator,measurement_covariance,
                          generate_synthetic_observations)


def test_radar_observation_operator_round_trip_geometry():
    obs=observation_operator(np.array([0.,1000.]),np.array([100_000.,80_000.]),-5000.)
    assert obs.shape==(2,2)
    assert np.all(obs[:,0]>0)
    assert np.all((obs[:,1]>0)&(obs[:,1]<np.pi/2))


def test_measurement_covariance_is_positive_definite():
    cov=measurement_covariance(TRACKING)
    assert np.all(np.linalg.eigvalsh(cov)>0)


def test_synthetic_noise_is_reproducible_and_missing_data_are_masked():
    m=simulate_mission()
    obs1=generate_synthetic_observations(m["sphere"],TRACKING,MISSION,seed=7)
    obs2=generate_synthetic_observations(m["sphere"],TRACKING,MISSION,seed=7)
    obs3=generate_synthetic_observations(m["sphere"],TRACKING,MISSION,seed=8)
    assert np.array_equal(obs1["valid"],obs2["valid"])
    assert np.allclose(obs1["measurement"],obs2["measurement"],equal_nan=True)
    assert not np.allclose(obs1["measurement"],obs3["measurement"],equal_nan=True)
    assert np.isnan(obs1["measurement"][~obs1["valid"]]).all()
