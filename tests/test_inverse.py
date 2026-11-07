"""Inverse-problem stability, synthetic recovery, and poor-observation behavior."""
from dataclasses import replace
import numpy as np
import pytest
from sbgea_keynote.config import MISSION,TRACKING,RETRIEVAL,WIND_TRUTH
from sbgea_keynote.mission.workflow import run_fall_sphere_experiment
from sbgea_keynote.mission.observations import generate_synthetic_observations
from sbgea_keynote.mission.inverse import retrieve_neutral_wind

@pytest.fixture(scope="module")
def baseline():
    return run_fall_sphere_experiment(seed=2026)


def test_wind_profile_recovery_and_covariance_are_finite(baseline):
    fit=baseline["retrieval"]
    assert fit["success"]
    assert baseline["evaluation"]["rmse_m_s"]<12.0
    assert np.all(np.isfinite(fit["wind_m_s"]))
    assert np.all(np.isfinite(fit["measurement_wind_std_m_s"]))
    assert np.all(fit["total_wind_std_m_s"]>=fit["measurement_wind_std_m_s"])
    assert "no true wind supplied" in fit["model_source"]


def test_retrieval_covariance_and_resolution_diagnostics(baseline):
    fit=baseline["retrieval"]
    assert fit["wind_covariance"].shape==(5,5)
    assert fit["averaging_kernel"].shape==(5,5)
    assert np.all(np.diag(fit["wind_covariance"])>=0)


def test_poor_cadence_noise_and_dropout_increase_uncertainty(baseline):
    poor=replace(TRACKING,cadence_s=4.0,range_sigma_m=30.0,
                 elevation_sigma_deg=.03,missing_fraction=.20,random_seed=99)
    obs=generate_synthetic_observations(baseline["mission"]["sphere"],poor,MISSION,seed=99)
    fit=retrieve_neutral_wind(obs,baseline["mission"]["sphere_initial_state"],MISSION,RETRIEVAL)
    assert fit["success"]
    assert np.median(fit["measurement_wind_std_m_s"]) > \
           np.median(baseline["retrieval"]["measurement_wind_std_m_s"])
    assert np.all(np.isfinite(fit["wind_m_s"]))
