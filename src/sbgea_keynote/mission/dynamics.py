"""Mesospheric atmosphere, rocket ascent, falling sphere, and flow-regime physics.

Coordinates: x is horizontal downrange, z is altitude above mean sea level. SI units.
The atmosphere follows a tabulated US Standard Atmosphere 1976-like reference from
0--110 km (log-linear density and linear temperature interpolation). Sphere drag
uses a Schiller--Naumann continuum estimate blended to a constant free-molecular
coefficient. The transitional/free-molecular approximation is explicitly flagged;
it is a research-planning demonstrator, not a flight-qualified aerodynamic model.
"""
from __future__ import annotations
from dataclasses import replace
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from ..config import MISSION, MissionConfig, WIND_TRUTH, WindProfile


def gravity_at_altitude(z_m, config: MissionConfig = MISSION):
    """Inverse-square gravity magnitude in m/s²."""
    z = np.maximum(np.asarray(z_m, dtype=float), -0.99 * config.earth_radius_m)
    return config.surface_gravity_m_s2 * (config.earth_radius_m / (config.earth_radius_m + z))**2


def atmosphere(z_m, config: MissionConfig = MISSION, *, density_scale=1.0):
    """Return density (kg/m³), temperature (K), and pressure (Pa).

    The USSA-like profile is log-linearly interpolated in density. Above the final
    node, the last scale height is extrapolated only to keep ODE evaluations finite.
    """
    z = np.asarray(z_m, dtype=float)
    h = np.maximum(z, 0.0)
    nodes = np.asarray(config.atmosphere_altitude_m, dtype=float)
    rho_nodes = np.asarray(config.atmosphere_density_kg_m3, dtype=float)
    temp_nodes = np.asarray(config.atmosphere_temperature_k, dtype=float)
    rho = np.exp(np.interp(h, nodes, np.log(rho_nodes)))
    temp = np.interp(h, nodes, temp_nodes)
    above = h > nodes[-1]
    if np.any(above):
        last_h = nodes[-1] - nodes[-2]
        scale_height = last_h / np.log(rho_nodes[-2] / rho_nodes[-1])
        rho = np.where(above, rho_nodes[-1] * np.exp(-(h-nodes[-1])/scale_height), rho)
        temp = np.where(above, temp_nodes[-1], temp)
    rho = rho * float(density_scale)
    pressure = rho * config.gas_constant_air_j_kg_k * temp
    if np.ndim(z_m) == 0:
        return float(rho), float(temp), float(pressure)
    return rho, temp, pressure


def neutral_wind(z_m, profile: WindProfile = WIND_TRUTH):
    """Piecewise-linear horizontal neutral wind u(z) in m/s."""
    return np.interp(np.asarray(z_m, dtype=float), profile.altitude_m, profile.speed_m_s,
                     left=profile.speed_m_s[0], right=profile.speed_m_s[-1])


def dynamic_viscosity(temp_k, config: MissionConfig = MISSION):
    """Sutherland-law dynamic viscosity of air in Pa s."""
    t = np.asarray(temp_k, dtype=float)
    tr, s, mur = config.viscosity_ref_temp_k, config.sutherland_temp_k, config.viscosity_ref_pa_s
    out = mur * (t / tr)**1.5 * (tr + s) / (t + s)
    return float(out) if np.ndim(temp_k) == 0 else out


def flow_properties(altitude_m: float, speed_relative_m_s: float, radius_m: float,
                    config: MissionConfig = MISSION, *, density_scale=1.0):
    """Reynolds number, Knudsen number, Mach number, Cd and regime label."""
    rho, temp, pressure = atmosphere(altitude_m, config, density_scale=density_scale)
    d = 2.0 * radius_m
    speed = max(abs(float(speed_relative_m_s)), 1e-8)
    mu = dynamic_viscosity(temp, config)
    re = rho * speed * d / max(mu, 1e-30)
    mean_free_path = (config.boltzmann_j_k * temp /
        (math.sqrt(2.0) * math.pi * config.molecular_diameter_m**2 * max(pressure, 1e-30)))
    kn = mean_free_path / d
    sound_speed = math.sqrt(1.4 * config.gas_constant_air_j_kg_k * temp)
    mach = speed / sound_speed
    # Schiller--Naumann sphere correlation: continuum range approximately
    # 0.1 < Re < 1000, with compressibility effects omitted.
    re_safe = max(re, 1e-8)
    cd_cont = 24.0/re_safe * (1.0 + 0.15*re_safe**0.687)
    cd_cont = min(cd_cont, 1.0e5)
    # Smooth bridge over 0.01 < Kn < 10. This is a documented approximation,
    # not a substitute for a Sentman/Henderson model or DSMC in rarefied flow.
    q = np.clip((math.log10(max(kn,1e-12)) - math.log10(0.01))/3.0, 0.0, 1.0)
    blend = float(q*q*(3.0-2.0*q))
    cd = (1.0-blend)*cd_cont + blend*config.sphere_free_molecular_cd
    if kn < 0.01:
        regime = "contínuo"
        uncertain = not (0.1 <= re <= 1000.0 and mach < 0.8)
    elif kn <= 10.0:
        regime = "transicional (estimativa híbrida)"
        uncertain = True
    else:
        regime = "molecular livre (aprox.)"
        uncertain = True
    if mach >= 0.8:
        uncertain = True
    return {"rho_kg_m3":rho,"temperature_k":temp,"pressure_pa":pressure,
            "dynamic_viscosity_pa_s":mu,"mean_free_path_m":mean_free_path,
            "reynolds":float(re),"knudsen":float(kn),"mach":float(mach),
            "cd_continuum":float(cd_cont),"cd":float(cd),"blend_free_molecular":blend,
            "regime":regime,"uncertain_or_unsuitable":bool(uncertain)}


def sphere_rhs(t, state, *, wind_altitude_m, wind_speed_m_s, config: MissionConfig = MISSION,
               density_scale=1.0, free_molecular_cd=None):
    """Falling sphere equations [x,z,vx,vz] with velocity relative to neutral air."""
    x,z,vx,vz = state
    rho,temp,_ = atmosphere(z,config,density_scale=density_scale)
    u = float(np.interp(z,wind_altitude_m,wind_speed_m_s,
                        left=wind_speed_m_s[0],right=wind_speed_m_s[-1]))
    rel = np.array([vx-u,vz])
    speed_rel = float(np.linalg.norm(rel))
    props = flow_properties(z,speed_rel,config.sphere_radius_m,config,
                           density_scale=density_scale)
    cd = props["cd"] if free_molecular_cd is None else _cd_with_fm_override(
        props,free_molecular_cd)
    area = math.pi*config.sphere_radius_m**2
    drag_acc = -0.5*rho*cd*area/config.sphere_mass_kg*speed_rel*rel
    return [vx,vz,drag_acc[0],drag_acc[1]-float(gravity_at_altitude(z,config))]


def _cd_with_fm_override(props, fm_cd):
    b=props["blend_free_molecular"]
    return (1-b)*props["cd_continuum"] + b*float(fm_cd)


def simulate_sphere(initial_state, wind_profile: WindProfile = WIND_TRUTH,
                    config: MissionConfig = MISSION, *, density_scale=None,
                    free_molecular_cd=None, stop_altitude_m=None, samples=1200,
                    rtol=2e-8, atol=2e-8):
    """Integrate sphere descent to a downward altitude event; dense output in SI."""
    density_scale = config.atmosphere_true_density_scale if density_scale is None else density_scale
    stop_altitude_m = config.sphere_solver_stop_altitude_m if stop_altitude_m is None else stop_altitude_m
    wind_z=np.asarray(wind_profile.altitude_m,dtype=float)
    wind_u=np.asarray(wind_profile.speed_m_s,dtype=float)
    def stop(t,s): return s[1]-stop_altitude_m
    stop.terminal=True; stop.direction=-1
    sol=solve_ivp(lambda t,s:sphere_rhs(t,s,wind_altitude_m=wind_z,wind_speed_m_s=wind_u,
                         config=config,density_scale=density_scale,
                         free_molecular_cd=free_molecular_cd),
        (0.0,1600.0),np.asarray(initial_state,dtype=float),events=stop,
        dense_output=True,rtol=rtol,atol=atol,max_step=config.sphere_solver_max_step_s)
    if not sol.success: raise RuntimeError(sol.message)
    tf=float(sol.t_events[0][0]) if len(sol.t_events[0]) else float(sol.t[-1])
    t=np.linspace(0.0,tf,samples); y=sol.sol(t)
    return {"t":t,"x":y[0],"z":y[1],"vx":y[2],"vz":y[3],"solver":sol,
            "duration_s":tf,"impact_detected":bool(len(sol.t_events[0])),
            "stop_altitude_m":stop_altitude_m,"initial_state":np.asarray(initial_state,dtype=float)}


def _rocket_integrate_for_thrust(thrust_n, config: MissionConfig, *, store=False):
    m0=config.rocket_initial_mass_kg; mdry=config.rocket_dry_mass_kg
    burn=config.rocket_burn_time_s; mdot=(m0-mdry)/burn
    def powered(t,s):
        z,v=s; m=max(m0-mdot*t,mdry)
        rho,_,_=atmosphere(z,config)
        drag=0.5*rho*config.rocket_cd*config.rocket_reference_area_m2*v*abs(v)/m
        return [v,thrust_n/m-float(gravity_at_altitude(z,config))-drag]
    power=solve_ivp(powered,(0,burn),[0.,0.],dense_output=True,rtol=2e-8,atol=1e-7,
                    max_step=config.rocket_solver_max_step_s)
    zb,vb=power.y[:,-1]
    def coast(t,s):
        z,v=s; rho,_,_=atmosphere(z,config)
        drag=0.5*rho*config.rocket_cd*config.rocket_reference_area_m2*v*abs(v)/mdry
        return [v,-float(gravity_at_altitude(z,config))-drag]
    def apogee_event(t,s): return s[1]
    apogee_event.terminal=True; apogee_event.direction=-1
    if vb <= 0:
        return (float(zb),power,None,mdry,mdot) if store else float(zb)
    coast_sol=solve_ivp(coast,(0,900),[zb,vb],events=apogee_event,dense_output=True,
        rtol=2e-8,atol=1e-7,max_step=config.rocket_solver_max_step_s)
    if not coast_sol.success: raise RuntimeError(coast_sol.message)
    apogee=float(coast_sol.y_events[0][0][0]) if len(coast_sol.t_events[0]) else float(coast_sol.y[0,-1])
    if store: return apogee,power,coast_sol,mdry,mdot
    return apogee


def simulate_rocket(config: MissionConfig = MISSION):
    """Calibrate constant engine thrust to target apogee, then simulate ascent.

    The 1-D vertical rocket uses variable mass, axial drag and variable gravity.
    Guidance, attitude, staging, winds and lateral forces are omitted. The rocket
    is a generic design study, not a historical vehicle.
    """
    target=config.target_apogee_m
    lo,hi=config.rocket_thrust_bounds_n
    if config.release_altitude_m > target:
        raise ValueError("release_altitude_m must not exceed target_apogee_m")
    f=lambda thrust:_rocket_integrate_for_thrust(thrust,config)-target
    flo,fhi=f(lo),f(hi)
    while fhi < 0 and hi < 1e6:
        hi*=1.7; fhi=f(hi)
    if flo*fhi > 0:
        raise ValueError(f"Could not bracket apogee target {target:g} m with thrust bounds")
    thrust=brentq(f,lo,hi,xtol=0.02,rtol=2e-8,maxiter=45)
    apogee,power,coast,mdry,mdot=_rocket_integrate_for_thrust(thrust,config,store=True)
    burn=config.rocket_burn_time_s
    tp=np.linspace(0,burn,250); zp=power.sol(tp)
    tc=np.linspace(0,float(coast.t[-1]),500); zc=coast.sol(tc)
    t=np.concatenate([tp,burn+tc[1:]])
    z=np.concatenate([zp[0],zc[0,1:]])
    vz=np.concatenate([zp[1],zc[1,1:]])
    mass=np.maximum(config.rocket_initial_mass_kg-mdot*np.minimum(t,burn),mdry)
    # Separation may occur at apogee or on the post-apogee coast at a chosen altitude.
    if abs(config.release_altitude_m-apogee) < 2.0:
        release_time=float(coast.t_events[0][0]) if len(coast.t_events[0]) else float(tc[-1])
        release_state=np.array([0.0,apogee,0.0,0.0])
    else:
        def release_event(tt,ss): return ss[0]-config.release_altitude_m
        release_event.terminal=True; release_event.direction=-1
        after=solve_ivp(lambda tt,ss:[ss[1],-float(gravity_at_altitude(ss[0],config))-
            0.5*atmosphere(ss[0],config)[0]*config.rocket_cd*config.rocket_reference_area_m2*
            ss[1]*abs(ss[1])/mdry],(0,float(coast.t_events[0][0])),[float(zp[0,-1]),float(zp[1,-1])],
            events=release_event,dense_output=True,rtol=2e-8,atol=1e-7,max_step=config.rocket_solver_max_step_s)
        if not len(after.t_events[0]): raise ValueError("rocket did not cross the requested release altitude")
        release_time=burn+float(after.t_events[0][0])
        rs=after.y_events[0][0]
        release_state=np.array([0.0,rs[0],0.0,rs[1]])
        # Truncate visualization history at release, preserving the physical ascent/coast.
        keep=t <= release_time
        if not keep[-1]:
            t=np.append(t[keep],release_time)
            z=np.append(z[keep],rs[0]); vz=np.append(vz[keep],rs[1])
            mass=np.append(mass[keep],mdry)
        else:
            t,z,vz,mass=t[keep],z[keep],vz[keep],mass[keep]
    return {"t":t,"z":z,"vz":vz,"mass":mass,"apogee_m":apogee,
            "calibrated_thrust_n":thrust,"release_time_s":release_time,
            "release_state":release_state,"burnout_altitude_m":float(zp[0,-1]),
            "burnout_speed_m_s":float(zp[1,-1]),"dry_mass_kg":mdry}


def sphere_release_state(rocket_result, config: MissionConfig = MISSION):
    """Inherit rocket ECI-like local state and add configurable separation velocity."""
    r=np.asarray(rocket_result["release_state"],dtype=float)
    return np.array([r[0],r[1],r[2]+config.sphere_release_separation_vx_m_s,
                     r[3]+config.sphere_release_separation_vz_m_s])


def simulate_mission(config: MissionConfig = MISSION, wind_truth: WindProfile = WIND_TRUTH):
    """Run rocket to separation, then independently integrate the released sphere."""
    rocket=simulate_rocket(config)
    initial=sphere_release_state(rocket,config)
    sphere=simulate_sphere(initial,wind_truth,config)
    return {"rocket":rocket,"sphere":sphere,"sphere_initial_state":initial,
            "wind_truth":wind_truth,"config":config}


def sphere_diagnostics(track, config: MissionConfig = MISSION, *, density_scale=1.0):
    """Compute aerodynamic regime diagnostics along a sphere trajectory."""
    vals=[flow_properties(z,math.hypot(vx-float(neutral_wind(z)),vz),
            config.sphere_radius_m,config,density_scale=density_scale)
          for z,vx,vz in zip(track["z"],track["vx"],track["vz"])]
    keys=("reynolds","knudsen","mach","cd","uncertain_or_unsuitable")
    out={k:np.array([v[k] for v in vals]) for k in keys}
    out["regime"]=np.array([v["regime"] for v in vals],dtype=object)
    return out
