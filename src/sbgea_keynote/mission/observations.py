"""Synthetic radar tracking and its observation operator (range, elevation)."""
from __future__ import annotations
import math
import numpy as np
from ..config import MISSION, MissionConfig, TRACKING, TrackingConfig


def observation_operator(x_m, z_m, station_x_m):
    """Radar observables [slant range m, elevation rad] for 2-D point positions."""
    dx=np.asarray(x_m,dtype=float)-float(station_x_m)
    z=np.asarray(z_m,dtype=float)
    return np.stack([np.hypot(dx,z),np.arctan2(z,dx)],axis=-1)


def measurement_covariance(tracking: TrackingConfig = TRACKING):
    """2x2 covariance for [range, elevation], SI units (m², rad²)."""
    sr=tracking.range_sigma_m
    se=math.radians(tracking.elevation_sigma_deg)
    cov=tracking.range_elevation_correlation*sr*se
    return np.array([[sr*sr,cov],[cov,se*se]],dtype=float)


def generate_synthetic_observations(track, tracking: TrackingConfig = TRACKING,
                                    config: MissionConfig = MISSION, *, seed=None):
    """Generate noisy, optionally missing radar observations from a truth track.

    Noise is applied in polar instrument coordinates using the specified covariance.
    The retrieval receives only measured values, times, validity, and covariance.
    """
    cadence=float(tracking.cadence_s)
    valid_alt=config.observation_stop_altitude_m
    z=track["z"]
    candidates=np.flatnonzero(z>=valid_alt)
    if candidates.size < 2:
        raise ValueError("truth track does not span the configured observation altitude")
    t_end=float(track["t"][candidates[-1]])
    times=np.arange(0.0,t_end+0.25*cadence,cadence)
    truth=track["solver"].sol(times)
    n=len(times); nsite=len(tracking.station_x_m)
    cov=measurement_covariance(tracking)
    rng=np.random.default_rng(tracking.random_seed if seed is None else seed)
    noisy=np.full((n,nsite,2),np.nan,dtype=float)
    valid=np.zeros((n,nsite),dtype=bool)
    for j,site in enumerate(tracking.station_x_m):
        ideal=observation_operator(truth[0],truth[1],site)
        draws=rng.multivariate_normal(np.zeros(2),cov,size=n)
        noisy[:,j,:]=ideal+draws
        # Keep angles physically interpretable in the upper half-plane.
        noisy[:,j,1]=np.clip(noisy[:,j,1],0.0,math.pi/2)
        valid[:,j]=rng.random(n)>=tracking.missing_fraction
        noisy[~valid[:,j],j,:]=np.nan
    # Preserve the scheduled cadence, including fully missing epochs, so the
    # observation record documents data gaps and remains reproducible in shape.
    return {"time_s":times,"measurement":noisy,"valid":valid,
            "covariance":cov,"station_x_m":tuple(tracking.station_x_m),
            "cadence_s":cadence,"range_sigma_m":tracking.range_sigma_m,
            "elevation_sigma_deg":tracking.elevation_sigma_deg,
            "seed":tracking.random_seed if seed is None else seed,
            "n_generated":int(n),"n_used":int(np.sum(valid))}


def predict_radar_observations(track, times_s, station_x_m):
    """Apply radar geometry to dense-output model states at requested times."""
    state=track["solver"].sol(np.asarray(times_s,dtype=float))
    out=np.empty((len(times_s),len(station_x_m),2),dtype=float)
    for j,site in enumerate(station_x_m):
        out[:,j,:]=observation_operator(state[0],state[1],site)
    return out
