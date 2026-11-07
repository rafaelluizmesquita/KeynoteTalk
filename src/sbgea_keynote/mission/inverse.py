"""Forward-model-based neutral-wind retrieval from synthetic radar observations.

No truth wind is imported here. The solver estimates wind knots and a log-density
scale nuisance parameter from measurements, with explicit weak wind and curvature
priors. The returned covariance is a local Gauss--Newton approximation including
those priors. It does not include all aerodynamic/model discrepancy.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from ..config import MISSION, MissionConfig, RetrievalConfig, RETRIEVAL, WindProfile
from .dynamics import simulate_sphere
from .observations import predict_radar_observations


def _profile_from_parameters(values, altitudes):
    return WindProfile(tuple(float(v) for v in altitudes),tuple(float(v) for v in values))


def retrieve_neutral_wind(observations, initial_state, config: MissionConfig = MISSION,
                          retrieval: RetrievalConfig = RETRIEVAL, *,
                          assumed_free_molecular_cd=None, initial_wind_m_s=None):
    """Retrieve wind and density scale by covariance-weighted forward fitting.

    Unknowns: horizontal neutral wind at configured altitude knots plus log density
    scale. Cost includes per-observation polar covariance, a weak zero-wind prior,
    a curvature regularizer, and a log-density prior centered on the *nominal*
    atmosphere. The measured track supplies no hidden truth profile.
    """
    alt=np.asarray(retrieval.wind_knot_altitudes_m,dtype=float)
    nwind=len(alt)
    if nwind<2: raise ValueError("at least two wind knots are required")
    if initial_wind_m_s is None: initial_wind_m_s=np.zeros(nwind)
    # Fit release-state offsets as nuisance parameters with declared telemetry/
    # separation priors; do not treat the rocket model's state as exact truth.
    x0=np.r_[np.asarray(initial_wind_m_s,dtype=float),0.0,np.zeros(4)]
    times=np.asarray(observations["time_s"],dtype=float)
    measured=np.asarray(observations["measurement"],dtype=float)
    valid=np.asarray(observations["valid"],dtype=bool)
    sites=tuple(observations["station_x_m"])
    cov=np.asarray(observations["covariance"],dtype=float)
    chol=np.linalg.cholesky(cov)
    inv_chol=np.linalg.inv(chol)
    idx=np.argwhere(valid)
    if len(idx)<nwind*2: raise ValueError("too few valid observations for requested wind resolution")

    def model(theta):
        winds=theta[:nwind]
        rho_scale=float(np.exp(theta[nwind]))
        adjusted_initial=np.asarray(initial_state,dtype=float).copy()
        adjusted_initial += theta[nwind+1:nwind+5]
        profile=_profile_from_parameters(winds,alt)
        track=simulate_sphere(adjusted_initial,profile,config,density_scale=rho_scale,
            free_molecular_cd=assumed_free_molecular_cd,
            stop_altitude_m=config.sphere_solver_stop_altitude_m,samples=400)
        # Observation times stop at the configured altitude, above the ODE event.
        pred=predict_radar_observations(track,times,sites)
        return pred

    def residual(theta, include_prior=True):
        pred=model(theta)
        pieces=[]
        for i,j in idx:
            delta=pred[i,j]-measured[i,j]
            delta[1]=(delta[1]+np.pi)%(2*np.pi)-np.pi
            pieces.extend(inv_chol@delta)
        if include_prior:
            wind=theta[:nwind]
            pieces.extend(wind/retrieval.wind_prior_sigma_m_s)
            if nwind>=3:
                curvature=np.diff(wind,n=2)
                pieces.extend(curvature/retrieval.curvature_prior_sigma_m_s)
            pieces.append(theta[nwind]/retrieval.log_density_prior_sigma)
            offsets=theta[nwind+1:nwind+5]
            pieces.extend(offsets[:2]/retrieval.release_position_prior_sigma_m)
            pieces.extend(offsets[2:]/retrieval.release_velocity_prior_sigma_m_s)
        return np.asarray(pieces,dtype=float)

    fit=least_squares(residual,x0,method="trf",max_nfev=retrieval.max_function_evaluations,
        diff_step=retrieval.finite_difference_step,xtol=2e-4,ftol=2e-4,gtol=2e-4,
        x_scale="jac")
    j=fit.jac
    information=j.T@j
    parameter_cov=np.linalg.pinv(information,rcond=1e-10)
    std=np.sqrt(np.maximum(0,np.diag(parameter_cov)))
    pred=model(fit.x)
    raw=[]
    for i,jj in idx:
        d=pred[i,jj]-measured[i,jj]
        d[1]=(d[1]+np.pi)%(2*np.pi)-np.pi
        raw.extend(inv_chol@d)
    raw=np.asarray(raw)
    wind_cov=parameter_cov[:nwind,:nwind]
    # Local averaging-kernel diagnostic: response to the fitted wind knots.
    # Includes density as a nuisance parameter through the full covariance.
    data_jac=fit.jac[:len(idx)*2,:]
    averaging_kernel_full=parameter_cov @ data_jac.T @ data_jac
    averaging_kernel=averaging_kernel_full[:nwind,:nwind]
    rho_scale=float(np.exp(fit.x[nwind]))
    rho_scale_std=rho_scale*std[nwind]
    model_error=np.full(nwind,float(retrieval.model_error_floor_m_s))
    total_std=np.sqrt(std[:nwind]**2+model_error**2)
    return {"wind_altitude_m":alt,"wind_m_s":fit.x[:nwind],"wind_std_m_s":std[:nwind],
        "measurement_wind_std_m_s":std[:nwind],"model_error_std_m_s":model_error,
        "total_wind_std_m_s":total_std,
        "wind_covariance":wind_cov,"parameter_covariance":parameter_cov,
        "density_scale":rho_scale,"density_scale_std":rho_scale_std,
        "release_state_offset_m_m_s":fit.x[nwind+1:nwind+5],
        "release_state_offset_std":std[nwind+1:nwind+5],
        "predicted_measurements":pred,"observation_mask":valid,
        "whitened_measurement_residual":raw,"chi_square":float(raw@raw),
        "degrees_of_freedom":max(1,len(raw)-nwind-1),
        "reduced_chi_square":float((raw@raw)/max(1,len(raw)-nwind-1)),
        "averaging_kernel":averaging_kernel,"success":bool(fit.success),
        "message":fit.message,"nfev":fit.nfev,"cost":float(fit.cost),
        "estimated_profile":_profile_from_parameters(fit.x[:nwind],alt),
        "model_source":"nominal atmosphere, assumed drag law and observations; no true wind supplied",
        "model_error_note":"configured additive wind-error floor, separate from measurement covariance"}


def evaluate_wind_retrieval(result, truth_profile, *, evaluation_altitudes_m=None):
    """Post-fit diagnostic against a known synthetic truth; never used by retrieval."""
    if evaluation_altitudes_m is None:
        evaluation_altitudes_m=np.asarray(result["wind_altitude_m"])
    z=np.asarray(evaluation_altitudes_m,dtype=float)
    truth=np.interp(z,truth_profile.altitude_m,truth_profile.speed_m_s)
    estimate=np.interp(z,result["wind_altitude_m"],result["wind_m_s"])
    return {"altitude_m":z,"truth_m_s":truth,"estimate_m_s":estimate,
            "rmse_m_s":float(np.sqrt(np.mean((estimate-truth)**2))),
            "information_useful_altitudes_m":z[result["total_wind_std_m_s"]<30] if len(z)==len(result["total_wind_std_m_s"]) else np.array([])}
