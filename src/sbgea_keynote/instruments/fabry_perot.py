"""Independent proof-of-concept Fabry--Pérot wind instrument simulator.

Idealized chain: thermal emission line → Doppler shift → Airy transmission →
photon/read noise → weighted retrieval of line-of-sight velocity. No falling-sphere
code is imported. The model omits real etalon defects, order sorting, field-angle
calibration, chemistry, line blends, detector defects and full instrument optics.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import least_squares
from ..config import FPI, FabryPerotConfig

C_LIGHT_M_S=299_792_458.0
K_B_J_K=1.380649e-23


def thermal_sigma_wavelength_m(config: FabryPerotConfig=FPI):
    """One-dimensional thermal Doppler sigma for a Maxwellian emitter."""
    sigma_v=math.sqrt(K_B_J_K*config.gas_temperature_k/config.emitting_species_mass_kg)
    return config.line_center_m/C_LIGHT_M_S*sigma_v


def wavelength_grid(config: FabryPerotConfig=FPI, *, wind_los_m_s=0.0):
    """Wavelength samples spanning the thermal line and useful etalon orders."""
    sigma=thermal_sigma_wavelength_m(config)
    center=config.line_center_m*(1.0+wind_los_m_s/C_LIGHT_M_S)
    half=max(7.0*sigma,0.8*config.line_center_m**2/(2*config.cavity_spacing_m))
    return np.linspace(center-half,center+half,config.integration_wavelength_bins)


def emission_profile(wavelength_m, wind_los_m_s=0.0, config: FabryPerotConfig=FPI):
    """Normalized Gaussian emission line with thermal width and Doppler shift."""
    center=config.line_center_m*(1.0+wind_los_m_s/C_LIGHT_M_S)
    sigma=thermal_sigma_wavelength_m(config)
    profile=np.exp(-0.5*((np.asarray(wavelength_m)-center)/sigma)**2)/(sigma*math.sqrt(2*math.pi))
    return profile


def detector_geometry(config: FabryPerotConfig=FPI):
    """Return detector x/y coordinates, radius, and incidence angle in radians."""
    n=config.detector_pixels
    axis=(np.arange(n)-(n-1)/2)*config.pixel_pitch_m
    xx,yy=np.meshgrid(axis,axis)
    radius=np.hypot(xx,yy)
    theta=np.arctan2(radius,config.focal_length_m)
    return xx,yy,radius,theta


def airy_transmission(wavelength_m, theta_rad, config: FabryPerotConfig=FPI):
    """Ideal Airy transmission for a lossless etalon at incidence angle theta."""
    lam=np.asarray(wavelength_m,dtype=float)[:,None,None]
    theta=np.asarray(theta_rad,dtype=float)[None,:,:]
    order=round(2*config.cavity_spacing_m/config.line_center_m)
    phase=4*math.pi*config.cavity_spacing_m*np.cos(theta)/lam-2*math.pi*order
    coefficient=(2*config.finesse/math.pi)**2
    return 1.0/(1.0+coefficient*np.sin(phase/2.0)**2)


def expected_interferogram(wind_los_m_s, config: FabryPerotConfig=FPI, *,
                           amplitude_scale=1.0, background_counts=None):
    """Expected detector counts after spectral integration (no stochastic noise)."""
    lam=wavelength_grid(config,wind_los_m_s=wind_los_m_s)
    spectrum=emission_profile(lam,wind_los_m_s,config)
    _,_,_,theta=detector_geometry(config)
    transmission=airy_transmission(lam,theta,config)
    integrated=trapezoid(spectrum[:,None,None]*transmission,lam,axis=0)
    bg=config.background_counts if background_counts is None else background_counts
    return bg+config.photon_scale_counts*amplitude_scale*integrated


def simulate_interferogram(wind_los_m_s=120.0, config: FabryPerotConfig=FPI, *, seed=None):
    """Generate a synthetic interferogram with Poisson photon and Gaussian read noise."""
    expected=expected_interferogram(wind_los_m_s,config)
    rng=np.random.default_rng(config.random_seed if seed is None else seed)
    photons=rng.poisson(np.maximum(expected,0.0))
    measured=photons+rng.normal(0.0,config.read_noise_e,size=expected.shape)
    # The retrieval can estimate shot-noise variance from the measured counts;
    # do not pass the hidden expectation from the synthetic truth as a weight.
    variance=np.maximum(measured,0.0)+config.read_noise_e**2
    return {"expected_counts":expected,"counts":measured,"variance_counts2":variance,
            "true_wind_los_m_s":float(wind_los_m_s),"seed":config.random_seed if seed is None else seed,
            "config":config}


def retrieve_line_of_sight_wind(observation, config: FabryPerotConfig=FPI, *,
                                wind_bounds_m_s=(-500.0,500.0)):
    """Weighted least squares for LOS wind, line amplitude, and background.

    The noise variance is estimated from observed counts plus known read noise. The local
    covariance from the fit Jacobian is a conditional 1-sigma approximation, given
    the ideal instrument calibration and optical parameters supplied to the model.
    """
    data=np.asarray(observation["counts"],dtype=float)
    sigma=np.sqrt(np.asarray(observation["variance_counts2"],dtype=float))
    def residual(theta):
        v,logamp,bg=theta
        model=expected_interferogram(v,config,amplitude_scale=math.exp(logamp),
                                     background_counts=bg)
        return ((model-data)/sigma).ravel()
    # A small deterministic multi-start avoids local least-squares minima from
    # fringe phase ambiguity. Starts do not use the synthetic truth value.
    low,high=wind_bounds_m_s
    seeds=sorted(set(float(np.clip(s,low+1e-4,high-1e-4))
                     for s in (-300.,-150.,-25.,75.,225.,400.)))
    fits=[]
    for seed_v in seeds:
        x0=np.array([seed_v,math.log(config.photon_scale_counts),config.background_counts])
        fits.append(least_squares(residual,x0,bounds=([low,math.log(1.0),0.0],
            [high,math.log(1.0e6),1.0e5]),method="trf",x_scale="jac",
            max_nfev=80,xtol=1e-10,ftol=1e-10,gtol=1e-8))
    fit=min(fits,key=lambda f:f.cost)
    # Explicit central differences with absolute scales are more stable than a
    # relative finite-difference step when the fitted LOS velocity is near zero.
    steps=np.array([0.25,1.0e-4,0.1])
    jac=np.empty((data.size,3),dtype=float)
    for j,h in enumerate(steps):
        plus=fit.x.copy(); minus=fit.x.copy()
        plus[j]+=h; minus[j]-=h
        jac[:,j]=(residual(plus)-residual(minus))/(2*h)
    cov=np.linalg.pinv(jac.T@jac,rcond=1e-12)
    return {"wind_los_m_s":float(fit.x[0]),"wind_std_m_s":float(math.sqrt(max(0,cov[0,0]))),
        "amplitude_scale":float(math.exp(fit.x[1])),"background_counts":float(fit.x[2]),
        "covariance_parameters":cov,"fitted_counts":expected_interferogram(fit.x[0],config,
            amplitude_scale=math.exp(fit.x[1]),background_counts=fit.x[2]),
        "chi_square":float(np.sum(fit.fun**2)),"degrees_of_freedom":max(1,data.size-3),
        "reduced_chi_square":float(np.sum(fit.fun**2)/max(1,data.size-3)),
        "success":bool(fit.success),"message":fit.message}


def compare_instrument_precision(wind_los_m_s=120.0, config: FabryPerotConfig=FPI,
                                 *, photon_scales=(400.0,1600.0,6400.0),
                                 finesses=(12.0,25.0,40.0)):
    """Estimate local wind precision versus detected photons and finesse.

    The calculation is a noiseless local Fisher approximation. It exposes trends,
    not a full end-to-end calibration uncertainty budget.
    """
    rows=[]
    for photons in photon_scales:
        for finesse in finesses:
            c=FabryPerotConfig(**{**config.__dict__,"photon_scale_counts":float(photons),
                                  "finesse":float(finesse)})
            expected=expected_interferogram(wind_los_m_s,c)
            sig=np.sqrt(expected+c.read_noise_e**2)
            dv=0.5
            deriv=(expected_interferogram(wind_los_m_s+dv,c)-
                   expected_interferogram(wind_los_m_s-dv,c))/(2*dv)
            fisher=float(np.sum((deriv/sig)**2))
            rows.append({"photons":photons,"finesse":finesse,
                         "wind_sigma_m_s":math.sqrt(1/max(fisher,1e-30))})
    return rows
