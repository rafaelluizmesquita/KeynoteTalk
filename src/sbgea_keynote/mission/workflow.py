"""End-to-end virtual falling-sphere experiment and design diagnostics."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from ..config import MISSION, TRACKING, RETRIEVAL, WIND_TRUTH
from .dynamics import simulate_mission, sphere_diagnostics
from .observations import generate_synthetic_observations
from .inverse import retrieve_neutral_wind, evaluate_wind_retrieval


def run_fall_sphere_experiment(mission=MISSION, tracking=TRACKING,
                               retrieval_config=RETRIEVAL, *, seed=None,
                               wind_truth=WIND_TRUTH):
    """Forward mission → synthetic observations → retrieval → post-fit score.

    The retrieval itself receives no truth profile; truth is only used afterward
    to score this synthetic experiment and plot the known-vs-estimated result.
    """
    simulated=simulate_mission(mission,wind_truth)
    obs=generate_synthetic_observations(simulated["sphere"],tracking,mission,seed=seed)
    fit=retrieve_neutral_wind(obs,simulated["sphere_initial_state"],mission,retrieval_config)
    score=evaluate_wind_retrieval(fit,wind_truth)
    diagnostics=sphere_diagnostics(simulated["sphere"],mission)
    return {"mission":simulated,"observations":obs,"retrieval":fit,
            "evaluation":score,"flow_diagnostics":diagnostics,
            "tracking_config":tracking,"retrieval_config":retrieval_config}


def drag_assumption_sensitivity(result, mission=MISSION, tracking=TRACKING,
                                retrieval_config=RETRIEVAL, *, fractional_change=0.2):
    """Refit with perturbed free-molecular Cd to expose model-assumption sensitivity."""
    center=mission.sphere_free_molecular_cd
    fits=[]
    for factor in (1.0-fractional_change,1.0+fractional_change):
        fit=retrieve_neutral_wind(result["observations"],
            result["mission"]["sphere_initial_state"],mission,retrieval_config,
            assumed_free_molecular_cd=center*factor)
        fits.append(fit)
    spread=np.abs(fits[1]["wind_m_s"]-fits[0]["wind_m_s"])/2
    return {"lower_cd":fits[0],"upper_cd":fits[1],"wind_half_spread_m_s":spread,
            "fractional_cd_change":fractional_change}


def experiment_summary(result):
    """Compact numerical metrics intended for a notebook or keynote panel."""
    r=result["retrieval"]; e=result["evaluation"]; obs=result["observations"]
    z=r["wind_altitude_m"]
    diag=np.diag(r["averaging_kernel"])
    return {"apogee_km":result["mission"]["rocket"]["apogee_m"]/1000,
        "calibrated_thrust_kn":result["mission"]["rocket"]["calibrated_thrust_n"]/1000,
        "release_altitude_km":result["mission"]["sphere_initial_state"][1]/1000,
        "sphere_duration_s":result["mission"]["sphere"]["duration_s"],
        "observations_used":obs["n_used"],"retrieval_rmse_m_s":e["rmse_m_s"],
        "median_measurement_std_m_s":float(np.median(r["measurement_wind_std_m_s"])),
        "median_total_std_m_s":float(np.median(r["total_wind_std_m_s"])),
        "density_scale_estimate":r["density_scale"],
        "useful_altitude_knots_km":z[r["total_wind_std_m_s"]<30]/1000,
        "averaging_kernel_diagonal":diag,
        "reduced_chi_square":r["reduced_chi_square"]}
