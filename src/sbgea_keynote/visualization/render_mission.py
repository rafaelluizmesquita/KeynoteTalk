"""Render the 110-second falling-sphere mission keynote and its deliverables."""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.path import Path as MarkerPath

from ..config import (PHYSICS, MISSION, WIND_TRUTH, TRACKING, RETRIEVAL,
                      MISSION_RENDER as RC, WindProfile)
from ..physics.projectile import ideal_projectile, simulate_projectile
from ..mission.dynamics import (atmosphere, flow_properties, neutral_wind,
                                sphere_diagnostics, simulate_sphere)
from ..mission.observations import generate_synthetic_observations
from ..mission.inverse import retrieve_neutral_wind, evaluate_wind_retrieval
from ..mission.workflow import run_fall_sphere_experiment, drag_assumption_sensitivity
from ..paths import output_dir

OUT = output_dir("animation")
NAVY="#082F73"; BLUE="#2686D0"; PALE_BLUE="#BBDDF0"
GREEN="#70AE54"; ORANGE="#EC8135"; INK="#18344F"
MUTED="#64798B"; LIGHT="#E7EFF4"; WHITE="#FFFFFF"


def _smooth(q):
    q=float(np.clip(q,0,1)); return q*q*(3-2*q)


def _line_to_fraction(line,model,q,xkey="x",zkey="y",z_offset=0):
    n=max(2,int(np.clip(q,0,1)*(len(model[xkey])-1))+1)
    line.set_data(model[xkey][:n],model[zkey][:n]+z_offset)


def _set_scatter(scatter,x,z):
    if len(x): scatter.set_offsets(np.c_[x,z])
    else: scatter.set_offsets(np.empty((0,2)))


def _make_person(ax):
    x=-1.2; h=PHYSICS.launch_height
    head=patches.Circle((x,h-.15),.15,fc=NAVY,ec="none",zorder=8)
    ax.add_patch(head)
    joints=[((x,h-.3),(x,.7)),((x,h-.5),(0.0,h)),
            ((x,h-.5),(x-.65,.9)),((x,.7),(x-.5,.05)),
            ((x,.7),(x+.55,.05))]
    for a,b in joints:
        ax.add_line(Line2D([a[0],b[0]],[a[1],b[1]],color=NAVY,lw=2.8,
                           solid_capstyle="round",zorder=8))
    return head


def _draw_rocket_icon(ax):
    """Screen-sized schematic rocket that remains legible on a 100 km axis."""
    tr=ax.transAxes; w=.035; h=.16; cx=.5; y=.0
    body=patches.FancyBboxPatch((cx-w/2,y),w,h*.68,
        boxstyle="round,pad=0.003,rounding_size=0.008",transform=tr,
        fc="#DDE9F0",ec=NAVY,lw=1.3,zorder=14,clip_on=False)
    nose=patches.Polygon([[cx-w/2,y+h*.68],[cx,y+h],[cx+w/2,y+h*.68]],
        closed=True,transform=tr,fc=ORANGE,ec=NAVY,lw=1.1,zorder=15,clip_on=False)
    fin1=patches.Polygon([[cx-w/2,y+h*.28],[cx-w*1.25,y],[cx-w/2,y+h*.08]],
        closed=True,transform=tr,fc=BLUE,ec=NAVY,lw=.8,zorder=13,clip_on=False)
    fin2=patches.Polygon([[cx+w/2,y+h*.28],[cx+w*1.25,y],[cx+w/2,y+h*.08]],
        closed=True,transform=tr,fc=BLUE,ec=NAVY,lw=.8,zorder=13,clip_on=False)
    flame=patches.Polygon([[cx-w*.28,y],[cx,y-h*.38],[cx+w*.28,y]],
        closed=True,transform=tr,fc=ORANGE,ec="none",alpha=.0,zorder=12,clip_on=False)
    for p in (body,nose,fin1,fin2,flame): ax.add_patch(p)
    return [body,nose,fin1,fin2,flame]


def _place_rocket_icon(parts,altitude_fraction,flame=0.0):
    body,nose,fin1,fin2,plume=parts; w=.035; h=.16; cx=.5
    y=float(np.clip(altitude_fraction,0,1))-h*.5
    body.set_bounds(cx-w/2,y,w,h*.68)
    nose.set_xy([[cx-w/2,y+h*.68],[cx,y+h],[cx+w/2,y+h*.68]])
    fin1.set_xy([[cx-w/2,y+h*.28],[cx-w*1.25,y],[cx-w/2,y+h*.08]])
    fin2.set_xy([[cx+w/2,y+h*.28],[cx+w*1.25,y],[cx+w/2,y+h*.08]])
    plume.set_xy([[cx-w*.28,y],[cx,y-h*(.18+.24*flame)],[cx+w*.28,y]])
    plume.set_alpha(float(np.clip(flame,0,1)))


def _rock_marker_path():
    vertices=np.array([[-1.0,-.28],[-.82,.36],[-.27,.86],[.30,.96],
        [.72,.54],[.94,-.05],[.54,-.66],[-.30,-.83],[-.86,-.58],[-1.0,-.28]])
    codes=[MarkerPath.MOVETO]+[MarkerPath.LINETO]*(len(vertices)-2)+[MarkerPath.CLOSEPOLY]
    return MarkerPath(vertices,codes)


def _prepare_render_data():
    """Build nominal, parameter-sweep, no-wind, and deliberately degraded cases."""
    result=run_fall_sphere_experiment(seed=TRACKING.random_seed)
    result["local_ideal"]=ideal_projectile(samples=700)
    result["local_drag"]=simulate_projectile(drag=True,wind="calm",density="constant",samples=700)
    result["local_calm"]=result["local_drag"]
    result["local_uniform"]=simulate_projectile(drag=True,wind="constant",density="constant",samples=700)
    result["local_shear"]=simulate_projectile(drag=True,wind="shear",density="constant",samples=700)
    cds=np.linspace(0.0,PHYSICS.rock_cd,13)
    result["drag_sweep"]=[simulate_projectile(drag=True,wind="calm",density="constant",
        cd=float(cd),samples=500) for cd in cds]
    result["drag_cd_sweep"]=cds
    scales=np.linspace(0.0,1.0,13)
    result["wind_sweep"]=[simulate_projectile(drag=True,wind="shear",density="constant",
        config=replace(PHYSICS,wind_u0=PHYSICS.wind_u0*q,
                       wind_shear=PHYSICS.wind_shear*q),samples=500) for q in scales]
    result["wind_scale_sweep"]=scales
    calm_profile=replace(WIND_TRUTH,speed_m_s=tuple(0.0 for _ in WIND_TRUTH.speed_m_s))
    result["no_wind_sphere"]=simulate_sphere(result["mission"]["sphere_initial_state"],
        calm_profile,MISSION,samples=1200)
    poor=replace(TRACKING,cadence_s=4.0,range_sigma_m=30.0,
        elevation_sigma_deg=.03,missing_fraction=.20,random_seed=99)
    poor_obs=generate_synthetic_observations(result["mission"]["sphere"],poor,MISSION,seed=99)
    poor_fit=retrieve_neutral_wind(poor_obs,result["mission"]["sphere_initial_state"],
                                    MISSION,RETRIEVAL)
    result["poor_tracking"]=poor
    result["poor_observations"]=poor_obs
    result["poor_retrieval"]=poor_fit
    result["poor_evaluation"]=evaluate_wind_retrieval(poor_fit,WIND_TRUTH)
    return result


def _profile_axes(fig):
    axs=[]
    specs=[([.425,.49,.30,.24],"densidade (kg/m³)",BLUE),
           ([.425,.20,.30,.24],"temperatura (K)","#B15C42")]
    z=np.linspace(RC.mesospheric_view_min_altitude_m,
                  RC.mesospheric_view_max_altitude_m,300)
    rho,temp,_=atmosphere(z,MISSION)
    wind=neutral_wind(z,WIND_TRUTH)
    for i,(pos,xlabel,color) in enumerate(specs):
        a=fig.add_axes(pos,facecolor=WHITE); a.tick_params(labelsize=7.5,colors=MUTED,length=2,pad=1)
        a.set_ylim(z[0]/1000,z[-1]/1000); a.grid(color=LIGHT,lw=.55)
        if i==0: a.set_ylabel("altitude (km)",fontsize=8.5,color=MUTED,labelpad=1)
        else: a.set_ylabel("")
        if i==0:
            a.set_title(xlabel,loc="right",fontsize=9.0,color=INK,pad=1)
            a.tick_params(axis="x",labelbottom=False,bottom=False)
        else: a.set_xlabel(xlabel,fontsize=9.0,color=INK,labelpad=1)
        for sp in a.spines.values(): sp.set_color("#CBD8E1")
        axs.append(a)
    axs[0].semilogx(rho,z/1000,color=BLUE,lw=2.1)
    axs[1].plot(temp,z/1000,color="#B15C42",lw=2.1)
    axs[0].set_xlim(float(np.min(rho))*0.5,float(np.max(rho))*1.5)
    axs[1].set_xlim(float(np.min(temp))-15,float(np.max(temp))+15)
    return axs


def _make_scene(result):
    fig=plt.figure(figsize=(RC.width/RC.dpi,RC.height/RC.dpi),dpi=RC.dpi,facecolor=WHITE)
    ax=fig.add_axes([.05,.18,.66,.58],facecolor=WHITE)
    ax.set_aspect("auto")
    # Local projectile models from the opening mechanics problem.
    ideal=result["local_ideal"]; drag=result["local_drag"]
    calm=result["local_calm"]; uniform=result["local_uniform"]; shear=result["local_shear"]
    ax.set_xlim(-10,75); ax.set_ylim(-3,25); ax.set_axisbelow(True)
    ax.axhline(0,color=NAVY,lw=1.3,alpha=.7,zorder=2)
    ax.grid(color=LIGHT,lw=.55,alpha=.7)
    local_ideal,=ax.plot(ideal["x"],ideal["y"],ls=(0,(5,4)),color=BLUE,lw=2.1,label="ideal")
    local_drag,=ax.plot(drag["x"],drag["y"],color=ORANGE,lw=2.5,label="com arrasto")
    local_uniform,=ax.plot(uniform["x"],uniform["y"],color=NAVY,lw=1.9,ls=(0,(2,3)),label="vento uniforme")
    local_shear,=ax.plot(shear["x"],shear["y"],color=GREEN,lw=2.4,label="vento u(z)")
    trace,=ax.plot([],[],color=BLUE,lw=3.2,alpha=.82,zorder=7)
    person=_make_person(ax)
    launch_y=PHYSICS.launch_height
    rock_glow=ax.scatter([0],[launch_y],
        s=420,color=ORANGE,alpha=.13,linewidths=0,zorder=9)
    rock_path=_rock_marker_path()
    rock=ax.scatter([0],[launch_y],s=175,marker=rock_path,facecolor="#8C735E",
                     edgecolor="#4D453F",lw=1.0,zorder=11)
    rock_highlight_path=MarkerPath(np.array([[-.45,.10],[-.34,.58],[-.08,.66],
        [.12,.42],[-.03,.15],[-.45,.10]]),[MarkerPath.MOVETO]+[MarkerPath.LINETO]*4+[MarkerPath.CLOSEPOLY])
    rock_highlight=ax.scatter([0],[launch_y],s=175,marker=rock_highlight_path,
                              facecolor="#C2AC91",edgecolor="none",alpha=.75,zorder=12)
    impact_marker=ax.scatter([ideal["range"]],[0],s=125,marker="x",color=ORANGE,
                             linewidths=2.6,zorder=10)
    # Separate legends reveal only the assumptions introduced so far.
    common_legend=dict(loc="upper right",frameon=True,fontsize=9.8,
                       facecolor=WHITE,edgecolor="#D9E3EA")
    local_legend_1=ax.legend(handles=[local_ideal],labels=["ideal"],**common_legend)
    ax.add_artist(local_legend_1)
    ax.legend_=None
    local_legend_2=ax.legend(handles=[local_ideal,local_drag],
        labels=["ideal","com arrasto"],**common_legend)
    ax.add_artist(local_legend_2)
    ax.legend_=None
    local_legend_3=ax.legend(handles=[local_ideal,local_drag,local_uniform,local_shear],
        labels=["ideal","calmo","vento uniforme","vento u(z)"],**common_legend)
    ax.add_artist(local_legend_3)
    ax.legend_=None
    local_legends=[local_legend_1,local_legend_2,local_legend_3]
    control_rail,=ax.plot([.055,.30],[.105,.105],transform=ax.transAxes,
        color="#C8D7E1",lw=4,solid_capstyle="round",zorder=20)
    control_knob=ax.scatter([.055],[.105],s=100,transform=ax.transAxes,
        color=ORANGE,edgecolor=WHITE,lw=1.2,zorder=22)
    control_name=ax.text(.055,.16,"",transform=ax.transAxes,color=NAVY,
        fontsize=12,fontweight="bold",ha="left",va="bottom",zorder=21)
    control_value=ax.text(.31,.105,"",transform=ax.transAxes,color=INK,
        fontsize=11.5,ha="left",va="center",zorder=21)
    wind_arrows=[]
    arrow_heights=(5.0,12.0,20.0)
    for zz in arrow_heights:
        uu=PHYSICS.wind_u0+PHYSICS.wind_shear*zz
        wind_arrows.append(ax.annotate("",xy=(15+uu,zz),xytext=(15,zz),
             arrowprops=dict(arrowstyle="->",color=GREEN,lw=1.3,alpha=.6),zorder=7))

    # Mesospheric scene artists, the 60--110 km band is not drawn as a uniform gas.
    meso_bands=[]
    for low,high,label,color in [(60,85,"mesosfera", "#DCECF5"),
                                 (85,100,"mesopausa / baixa termosfera","#EEF5F8"),
                                 (100,110,"termosfera inferior","#F7F1E9")]:
        band=ax.axhspan(low*1000,high*1000,color=color,alpha=.9,zorder=0)
        text=ax.text(-5_650,0.5*(low+high)*1000,label,color=MUTED,fontsize=7.2,va="center")
        meso_bands.extend([band,text])
    meso_wind_arrows=[]
    for altitude in WIND_TRUTH.altitude_m:
        u=float(neutral_wind(altitude,WIND_TRUTH))
        x0=-5_000.; scale=28.0
        ar=ax.annotate("",xy=(x0+u*scale,altitude),xytext=(x0,altitude),
            arrowprops=dict(arrowstyle="-|>",color=GREEN,lw=1.7,alpha=.78),zorder=8)
        meso_wind_arrows.append(ar)
    altitude_line,=ax.plot([0,0],[RC.mesospheric_view_min_altitude_m,
        RC.mesospheric_view_max_altitude_m],color="#B5C6D2",lw=1,ls="--",zorder=1)
    profiles=_profile_axes(fig)
    for p in profiles: p.set_visible(False)

    # Rocket and released sphere tracks use separate state histories.
    rocket_line,=ax.plot([],[],color=NAVY,lw=2.5,zorder=6)
    rocket_icon=_draw_rocket_icon(ax)
    # A separate, instrumented test sphere rides at the rocket nose until
    # separation. This is a new experimental object, not the hand-thrown rock.
    payload_marker=patches.Circle((.5,.0),.014,transform=ax.transAxes,
        fc=ORANGE,ec=NAVY,lw=1.25,zorder=17,clip_on=False)
    payload_sensor=patches.Circle((.5,.0),.0045,transform=ax.transAxes,
        fc=WHITE,ec="none",zorder=18,clip_on=False)
    ax.add_patch(payload_marker);ax.add_patch(payload_sensor)
    rocket_event=ax.text(.035,.94,"",transform=ax.transAxes,color=NAVY,
        fontsize=11.2,fontweight="bold",va="top",
        bbox=dict(boxstyle="round,pad=.28",fc=WHITE,ec="#D9E3EA",alpha=.96),zorder=19)
    release_mark=ax.scatter([0],[MISSION.target_apogee_m],s=105,marker="o",facecolor=ORANGE,
                            edgecolor=WHITE,lw=1.4,zorder=15)
    mission=result["mission"]
    sphere=mission["sphere"]
    sphere_line,=ax.plot([],[],color=BLUE,lw=3.0,zorder=7)
    sphere_reference,=ax.plot([],[],color="#8E9DA8",lw=2.0,ls=(0,(4,4)),alpha=.9,zorder=5)
    sphere_marker=ax.scatter([0],[0],s=180,marker="o",facecolor=ORANGE,
                             edgecolor=NAVY,lw=1.7,zorder=14)
    sphere_sensor=ax.scatter([0],[0],s=24,marker="o",facecolor=WHITE,
                             edgecolor=NAVY,lw=.5,zorder=15)
    radar_site=ax.scatter([TRACKING.station_x_m[0]],[0],s=65,marker="^",color=NAVY,
                          edgecolor=WHITE,lw=.7,zorder=12)
    ray_lines=[]
    for _ in range(5):
        ln,=ax.plot([],[],color=ORANGE,lw=.9,alpha=.32,ls="--",zorder=3); ray_lines.append(ln)
    measurements=ax.scatter([],[],s=26,marker="o",facecolor=ORANGE,edgecolor=WHITE,
                            lw=.55,alpha=.9,zorder=11)

    # Stage-7 profile lines, truth is plotted only for post-fit synthetic scoring.
    wind_true_line,=ax.plot([],[],color=GREEN,lw=3.0,label="vento prescrito")
    wind_fit_line,=ax.plot([],[],color=ORANGE,lw=3.0,label="recuperado")
    wind_poor_line,=ax.plot([],[],color="#667784",lw=2.7,ls=(0,(4,3)),label="medidas degradadas")
    wind_caption=ax.text(.025,.96,"",transform=ax.transAxes,color=NAVY,fontsize=11.2,
        fontweight="bold",va="top",bbox=dict(boxstyle="round,pad=.3",fc=WHITE,ec="#D9E3EA",alpha=.95),zorder=20)
    uncertainty=None
    fit=result["retrieval"]; zfit=fit["wind_altitude_m"]/1000
    fit_sigma=fit["total_wind_std_m_s"]
    # Create an initially hidden confidence band, updated from retrieval only.
    wind_band=ax.fill_betweenx(zfit,fit["wind_m_s"]-fit_sigma,
        fit["wind_m_s"]+fit_sigma,color=ORANGE,alpha=.17,label="σ total")
    wind_legend=ax.legend(handles=[Line2D([],[],color=GREEN,lw=2.7,label="perfil verdadeiro"),
        Line2D([],[],color=ORANGE,lw=2.7,label="recuperado"),
        patches.Patch(facecolor=ORANGE,alpha=.17,label="incerteza total")],
        loc="lower right",frameon=True,fontsize=8.6,facecolor=WHITE,edgecolor="#D9E3EA")
    ax.legend_=None
    for legend in local_legends: legend.set_visible(False)
    wind_true_line.set_visible(False); wind_fit_line.set_visible(False)
    wind_poor_line.set_visible(False); wind_caption.set_visible(False)
    wind_band.set_visible(False); wind_legend.set_visible(False)

    # Final split scene: physical experiment at left, virtual experiment at right.
    pipeline_art=[]
    physical=patches.FancyBboxPatch((.02,.31),.43,.61,boxstyle="round,pad=.012,rounding_size=.025",
        fc="#F7FAFC",ec="#D4E2EA",lw=1.5,zorder=2)
    virtual=patches.FancyBboxPatch((.55,.31),.43,.61,boxstyle="round,pad=.012,rounding_size=.025",
        fc="#F7FAFC",ec="#D4E2EA",lw=1.5,zorder=2)
    ax.add_patch(physical);ax.add_patch(virtual);pipeline_art.extend([physical,virtual])
    for y0,y1,color in ((.44,.55,"#DCECF5"),(.55,.68,"#EEF5F8"),(.68,.82,"#F7F1E9")):
        band=patches.Rectangle((.065,y0),.34,y1-y0,transform=ax.transAxes,
            fc=color,ec="none",zorder=3);ax.add_patch(band);pipeline_art.append(band)
    headings=[ax.text(.235,.865,"Experimento físico",ha="center",va="center",
        fontsize=15,fontweight="bold",color=NAVY,zorder=5),
        ax.text(.765,.865,"Experimento virtual",ha="center",va="center",
        fontsize=15,fontweight="bold",color=NAVY,zorder=5)]
    pipeline_art.extend(headings)
    # A compact but recognizable rocket, sphere, and measured descent.
    rocket_body=patches.FancyBboxPatch((.16,.405),.055,.21,boxstyle="round,pad=.004",
        transform=ax.transAxes,fc="#DDE9F0",ec=NAVY,lw=1.5,zorder=7)
    rocket_nose=patches.Polygon([[.16,.615],[.1875,.69],[.215,.615]],closed=True,
        transform=ax.transAxes,fc=ORANGE,ec=NAVY,lw=1.2,zorder=8)
    rocket_fin=patches.Polygon([[.16,.455],[.13,.405],[.16,.425],
        [.215,.455],[.245,.405],[.215,.425]],closed=True,transform=ax.transAxes,
        fc=BLUE,ec=NAVY,lw=1.0,zorder=6)
    flight=Line2D([.1875,.29],[.70,.47],transform=ax.transAxes,color=BLUE,
        lw=2.2,ls=(0,(4,3)),zorder=6)
    sphere=patches.Circle((.29,.47),.018,transform=ax.transAxes,fc=ORANGE,
        ec=NAVY,lw=1.2,zorder=9)
    physical_text=ax.text(.235,.355,"Foguete  →  separação  →  esfera rastreada",
        ha="center",va="center",fontsize=9.8,color=INK,zorder=5)
    for art in (rocket_body,rocket_nose,rocket_fin,flight,sphere,physical_text):
        if isinstance(art,Line2D): ax.add_line(art)
        else: ax.add_patch(art) if isinstance(art,patches.Patch) else None
        pipeline_art.append(art)
    steps=[("Modelo físico",BLUE),("Experimento virtual",NAVY),
           ("Observações simuladas",GREEN),("Recuperação de parâmetros",ORANGE)]
    for i,(label,color) in enumerate(steps):
        y=.735-i*.105
        rect=patches.FancyBboxPatch((.61,y),.31,.075,
            boxstyle="round,pad=.008,rounding_size=.015",transform=ax.transAxes,
            fc=WHITE,ec=color,lw=1.8,zorder=5)
        txt=ax.text(.765,y+.0375,label,transform=ax.transAxes,ha="center",va="center",
            color=NAVY,fontsize=11.5,fontweight="bold",zorder=6)
        pipeline_art.extend([rect,txt])
        if i<3:
            arrow=ax.annotate("",xy=(.765,y-.024),xytext=(.765,y-.002),
                xycoords=ax.transAxes,textcoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-|>",color=MUTED,lw=1.4),zorder=5)
            pipeline_art.append(arrow)
    loop=patches.FancyArrowPatch((.955,.405),(.955,.80),connectionstyle="arc3,rad=-.18",
        transform=ax.transAxes,arrowstyle="-|>",mutation_scale=14,color=GREEN,lw=1.8,zorder=5)
    ax.add_patch(loop);pipeline_art.append(loop)
    feedback=ax.text(.765,.355,"Incertezas  →  otimizar  →  novo modelo",transform=ax.transAxes,
        ha="center",va="center",fontsize=9.8,fontweight="bold",color=GREEN,zorder=6)
    pipeline_art.append(feedback)
    for artist in pipeline_art: artist.set_visible(False)

    fig.text(.045,.958,"XI SBGEA 2026  •  GEofísica e Aeronomia Espacial".upper(),
             color=MUTED,fontsize=10.2,fontweight="bold",va="top")
    headline=fig.text(.045,.895,"Uma esfera. Uma pergunta.",color=NAVY,
                      fontsize=22.5,fontweight="bold",va="top")
    divider=Line2D([.735,.735],[.16,.80],transform=fig.transFigure,color="#D9E3EA",lw=1)
    fig.add_artist(divider)
    side_tag=fig.text(.755,.785,"ETAPA 01 / 08",color=BLUE,fontsize=10.2,fontweight="bold",va="top")
    side_title=fig.text(.755,.738,"",color=NAVY,fontsize=17.0,fontweight="bold",va="top",linespacing=1.1)
    side_body=fig.text(.755,.625,"",color=INK,fontsize=11.8,va="top",linespacing=1.35)
    side_stats=fig.text(.755,.37,"",color=MUTED,fontsize=11.0,va="top",linespacing=1.35)
    side_question=fig.text(.755,.205,"",color=ORANGE,fontsize=12.2,fontweight="bold",va="top",linespacing=1.22)
    footer=Line2D([.045,.95],[.115,.115],transform=fig.transFigure,color="#D9E3EA",lw=1)
    fig.add_artist(footer)
    stages=["01  Projétil","02  Arrasto","03  Vento","04  Mesosfera",
            "05  Foguete","06  Medidas","07  Recuperação","08  Simulador"]
    footer_items=[]
    for x,label in zip(np.linspace(.05,.86,8),stages):
        footer_items.append(fig.text(x,.071,label,color="#A6B2BC",fontsize=8.8,
                                    fontweight="bold",va="center"))
    note=fig.text(.045,.027,"Modelo numérico • hipóteses e incertezas explícitas",
                  color="#8494A1",fontsize=8.3,va="bottom")
    final_banner=patches.FancyBboxPatch((.035,.02),.93,.255,transform=ax.transAxes,
        boxstyle="round,pad=.012,rounding_size=.018",fc=WHITE,ec="#D7E4EA",
        lw=1.4,alpha=.97,zorder=30,visible=False)
    ax.add_patch(final_banner)
    final_question=ax.text(.5,.215,"Quanto podemos descobrir antes de lançar um único instrumento?",
        transform=ax.transAxes,color=NAVY,fontsize=16,fontweight="bold",ha="center",
        va="center",wrap=True,alpha=0,zorder=31)
    final_message=ax.text(.5,.105,"Os dois exigem ciência. Os dois podem produzir descobertas.\n"
        "Talvez você não possa lançar esse foguete. Mas pode ajudar a definir\n"
        "o experimento que um dia será lançado.",transform=ax.transAxes,color=INK,
        fontsize=12.8,ha="center",va="center",linespacing=1.2,alpha=0,zorder=31)
    result_line=ax.text(.5,.045,"",transform=ax.transAxes,color=GREEN,fontsize=11,
        fontweight="bold",ha="center",va="center",alpha=0,zorder=31)

    artists={"fig":fig,"ax":ax,"divider":divider,"footer":footer,"note":note,
        "ideal":ideal,"drag":drag,"calm":calm,
        "uniform":uniform,"shear":shear,"local_ideal":local_ideal,"local_drag":local_drag,
        "local_uniform":local_uniform,"local_shear":local_shear,"trace":trace,
        "person":person,"rock":rock,"rock_highlight":rock_highlight,"rock_glow":rock_glow,
        "impact_marker":impact_marker,"local_legends":local_legends,
        "control_rail":control_rail,"control_knob":control_knob,"control_name":control_name,
        "control_value":control_value,"wind_arrows":wind_arrows,"meso_wind_arrows":meso_wind_arrows,
        "meso_bands":meso_bands,"altitude_line":altitude_line,
        "profiles":profiles,"rocket_line":rocket_line,"rocket_icon":rocket_icon,
        "release_mark":release_mark,"payload_marker":payload_marker,
        "payload_sensor":payload_sensor,"rocket_event":rocket_event,
        "sphere_line":sphere_line,"sphere_reference":sphere_reference,
        "sphere_marker":sphere_marker,"sphere_sensor":sphere_sensor,
        "radar_site":radar_site,"ray_lines":ray_lines,"measurements":measurements,
        "wind_true_line":wind_true_line,"wind_fit_line":wind_fit_line,
        "wind_poor_line":wind_poor_line,"wind_caption":wind_caption,"wind_band":wind_band,
        "wind_legend":wind_legend,"pipeline_art":pipeline_art,"headline":headline,
        "side_tag":side_tag,"side_title":side_title,"side_body":side_body,
        "side_stats":side_stats,"side_question":side_question,"footer_items":footer_items,
        "final_banner":final_banner,"final_question":final_question,
        "final_message":final_message,"result_line":result_line,
        "result":result}
    return artists


def _side_copy(a,stage):
    r=a["result"]; m=r["mission"];fit=r["retrieval"];obs=r["observations"]
    sim=m["rocket"]; sphere=m["sphere"]
    titles=["Uma pedra.\nUma pergunta.","O arrasto muda\no alcance.",
        "O vento desloca\na previsão.",f"A {RC.mesosphere_reference_altitude_m/1000:g} km,\na atmosfera conta.",
        "A pergunta continua.\nO objeto muda.","O radar observa\na trajetória.",
        "O vento emerge\ndas medidas.","Experimento físico\ne virtual."]
    a["side_tag"].set_text(f"ETAPA {stage:02d} / 08")
    a["side_title"].set_text(titles[stage-1])
    if stage==1:
        a["side_body"].set_text("F = ma\nGravidade constante.")
        a["side_stats"].set_text(f"v₀={PHYSICS.launch_speed:.0f} m/s • θ={PHYSICS.launch_angle_deg:.0f}°\nAlcance previsto: {r['local_ideal']['range']:.1f} m")
        a["side_question"].set_text("Conseguimos prever\nonde ela vai cair?")
    elif stage==2:
        a["side_body"].set_text("A resistência do ar\nentra no modelo.")
        a["side_stats"].set_text("")
        a["side_question"].set_text("E se considerarmos a\nresistência do ar?")
    elif stage==3:
        a["side_body"].set_text("O arrasto usa a\nvelocidade relativa ao ar.")
        a["side_stats"].set_text("")
        a["side_question"].set_text("E se o ar também\nestiver em movimento?")
    elif stage==4:
        fd=r["flow_diagnostics"]
        z=sphere["z"]/1000
        ref_alt_km=RC.mesosphere_reference_altitude_m/1000
        j=int(np.argmin(abs(z-ref_alt_km)))
        rho,temp,_=atmosphere(RC.mesosphere_reference_altitude_m,MISSION)
        a["side_body"].set_text("Densidade e temperatura\nvariam com a altitude.")
        a["side_stats"].set_text(f"Perfil ref.: USSA76\n{ref_alt_km:g} km: {temp:.0f} K\nRe={fd['reynolds'][j]:.2g}; Kn={fd['knudsen'][j]:.3g}\nMach={fd['mach'][j]:.2g} • Cd incerto")
        a["side_question"].set_text(f"E se quisermos conhecer\nos ventos a {ref_alt_km:g} km?")
    elif stage==5:
        a["side_body"].set_text("A ideia segue; muda a escala.\nAgora, uma esfera\ninstrumentada.")
        a["side_stats"].set_text(f"Apogeu do modelo:\n{sim['apogee_m']/1000:.0f} km\nSeparação: {MISSION.sphere_release_separation_vz_m_s:.0f} m/s")
        a["side_question"].set_text("O foguete é o meio.\nA pergunta é o vento.")
    elif stage==6:
        track=m["sphere"];reference=r["no_wind_sphere"]
        stop_z=MISSION.observation_stop_altitude_m
        j=int(np.argmin(abs(track["z"]-stop_z)))
        jr=int(np.argmin(abs(reference["z"]-stop_z)))
        dx=track["x"][j]-reference["x"][jr]
        a["side_body"].set_text("Observável: alcance e elevação,\ncom ruído. Tracejado: sem vento.")
        a["side_stats"].set_text(f"Cadência: {obs['cadence_s']:.0f} s\nσ alcance: {obs['range_sigma_m']:.0f} m\nDesvio em {stop_z/1000:.0f} km:\n{dx:.0f} m")
        a["side_question"].set_text("O radar observa posição.\nNão mede vento direto.")
    elif stage==7:
        ev=r["evaluation"]
        useful=fit["wind_altitude_m"][fit["total_wind_std_m_s"]<10]/1000
        useful_text=(f"{useful.min():.0f}–{useful.max():.0f} km" if useful.size else "nenhuma faixa <10 m/s")
        a["side_body"].set_text("O ajuste usa posição,\ncadência e covariância.\nA verdade fica oculta\naté a avaliação.")
        a["side_stats"].set_text(f"RMSE sintético: {ev['rmse_m_s']:.1f} m/s\nσ total mediana: {np.median(fit['total_wind_std_m_s']):.1f} m/s\nFaixa σ<10: {useful_text}")
        a["side_question"].set_text("O que podemos recuperar\n— e com que confiança?")
    elif stage==8:
        a["side_body"].set_text("");a["side_stats"].set_text("");a["side_question"].set_text("")


def update_frame(frame,a,speed=1.0):
    result=a["result"];ax=a["ax"]
    t=frame/RC.fps*speed
    bounds=np.r_[0,np.cumsum(RC.stage_durations_s)]
    stage=min(int(np.searchsorted(bounds[1:],t,side="right"))+1,8)
    start=bounds[stage-1];duration=RC.stage_durations_s[stage-1]
    within=float(np.clip((t-start)/duration,0,1))
    headlines=["Conseguimos prever onde ela vai cair?",
        "O arrasto muda a trajetória diante dos nossos olhos.",
        "O vento desloca a previsão de pouso.",
        f"A {RC.mesosphere_reference_altitude_m/1000:g} km, a atmosfera também é parte do modelo.",
        "A pergunta continua. O objeto agora é uma esfera instrumentada.",
        "O radar não vê o vento: observa a trajetória.",
        "O vento emerge das medidas — com incerteza.",
        "Do experimento físico ao experimento virtual."]
    a["headline"].set_text(headlines[stage-1])
    a["headline"].set_fontsize(22.5)
    if stage==1 and within<.32:
        a["headline"].set_text("Tudo começa com curiosidade…\nAqui, começamos com uma pedra.")
        a["headline"].set_fontsize(19.5)
    a["divider"].set_visible(stage!=8)
    for key in ("side_tag","side_title","side_body","side_stats","side_question"):
        a[key].set_alpha(0.0 if stage==8 else 1.0)
    a["footer"].set_visible(stage!=8);a["note"].set_visible(stage!=8)
    for i,item in enumerate(a["footer_items"]):
        item.set_visible(stage!=8)
        item.set_color([BLUE,ORANGE,GREEN,BLUE,NAVY,ORANGE,ORANGE,GREEN][i]
                       if i==stage-1 else "#A6B2BC")
    _side_copy(a,stage)

    local_pos=np.array([.05,.18,.66,.58]);meso_pos=np.array([.05,.18,.32,.58])
    broad_pos=np.array([.05,.18,.66,.58])
    local_lines=("local_ideal","local_drag","local_uniform","local_shear","trace")
    for key in local_lines:a[key].set_visible(stage<=3)
    for key in ("person","rock","rock_highlight","rock_glow"):a[key].set_visible(stage<=3)
    a["impact_marker"].set_visible(stage<=3)
    a["control_rail"].set_visible(stage in (2,3))
    a["control_knob"].set_visible(stage in (2,3))
    a["control_name"].set_visible(stage in (2,3))
    a["control_value"].set_visible(stage in (2,3))
    for i,legend in enumerate(a["local_legends"],start=1):legend.set_visible(stage==i)
    for ar in a["wind_arrows"]:ar.set_visible(stage==3)
    for ar in a["meso_wind_arrows"]:ar.set_visible(stage in (4,5,6))
    for p in a["meso_bands"]:p.set_visible(stage in (4,5,6))
    a["altitude_line"].set_visible(stage==4)
    for p in a["profiles"]:p.set_visible(stage==4)
    a["rocket_line"].set_visible(stage==5)
    for p in a["rocket_icon"]:p.set_visible(stage==5 and within<.78)
    a["payload_marker"].set_visible(stage==5 and within<.78)
    a["payload_sensor"].set_visible(stage==5 and within<.78)
    a["rocket_event"].set_visible(stage==5)
    a["release_mark"].set_visible(stage==5 and within>=.77)
    a["sphere_line"].set_visible(stage==6 or (stage==5 and within>=.79))
    a["sphere_reference"].set_visible(stage==6 or (stage==5 and within>=.78))
    a["sphere_marker"].set_visible((stage==5 and within>=.78) or stage==6)
    a["sphere_sensor"].set_visible((stage==5 and within>=.78) or stage==6)
    a["radar_site"].set_visible(stage==6)
    for line in a["ray_lines"]:line.set_visible(stage==6)
    a["measurements"].set_visible(stage==6)
    a["wind_true_line"].set_visible(False);a["wind_fit_line"].set_visible(False)
    a["wind_poor_line"].set_visible(False);a["wind_caption"].set_visible(stage==7)
    a["wind_legend"].set_visible(False)
    for p in a["pipeline_art"]:p.set_visible(stage==8)
    a["final_banner"].set_visible(stage==8 and within>=.18)
    a["final_question"].set_alpha(_smooth((within-.05)/.15) if stage==8 else 0)
    a["final_message"].set_alpha(_smooth((within-.24)/.16) if stage==8 else 0)
    a["result_line"].set_alpha(0)

    # Smooth continuous camera from a hand-sized throw to the mesospheric column.
    if stage<=3:
        ax.set_position(local_pos);ax.set_xlim(-3,75);ax.set_ylim(-3,25)
        ax.set_xlabel("distância horizontal (m)",fontsize=10.5,color=MUTED)
        ax.set_ylabel("altura (m)",fontsize=10.5,color=MUTED)
        ax.grid(color=LIGHT,lw=.6,alpha=.65);ax.set_axisbelow(True);ax.set_axis_on()
        model=None;param_q=within
        if stage==1:
            model=a["ideal"];a["local_ideal"].set_alpha(.85)
            a["local_drag"].set_alpha(0);a["local_uniform"].set_alpha(0);a["local_shear"].set_alpha(0)
            trace_q=_smooth((within-.08)/.60)
        elif stage==2:
            k=min(int(param_q*(len(result["drag_sweep"])-1)),len(result["drag_sweep"])-1)
            model=result["drag_sweep"][k]
            a["local_ideal"].set_alpha(.28);a["local_drag"].set_alpha(1)
            a["local_drag"].set_data(model["x"],model["y"])
            a["local_uniform"].set_alpha(0);a["local_shear"].set_alpha(0)
            cd=float(result["drag_cd_sweep"][k])
            a["control_name"].set_text("Experimento virtual  ·  arrasto")
            a["control_value"].set_text(f"Cᵈ = {cd:.2f}")
            a["side_body"].set_text("O mesmo lançamento.\nAgora, o ar exerce força.")
            a["side_stats"].set_text(f"Cᵈ = {cd:.2f}\nAlcance previsto: {model['range']:.1f} m\nMudança: {model['range']-a['ideal']['range']:+.1f} m")
            trace_q=_smooth((within-.06)/.78)
        else:
            k=min(int(param_q*(len(result["wind_sweep"])-1)),len(result["wind_sweep"])-1)
            model=result["wind_sweep"][k]
            a["local_ideal"].set_alpha(.2);a["local_drag"].set_alpha(.34)
            a["local_uniform"].set_alpha(.72);a["local_shear"].set_alpha(1)
            a["local_shear"].set_data(model["x"],model["y"])
            scale=float(result["wind_scale_sweep"][k])
            a["control_name"].set_text("Experimento virtual  ·  vento")
            a["control_value"].set_text(f"u₀={PHYSICS.wind_u0*scale:.1f} m/s")
            a["side_body"].set_text("O arrasto depende da\nvelocidade relativa ao ar.")
            a["side_stats"].set_text(f"u₀ = {PHYSICS.wind_u0*scale:.1f} m/s\nS = {PHYSICS.wind_shear*scale:.2f} s⁻¹\nAlcance: {model['range']:.1f} m")
            trace_q=_smooth((within-.04)/.80)
            for ar,zz in zip(a["wind_arrows"],(5.0,12.0,20.0)):
                u=(PHYSICS.wind_u0+PHYSICS.wind_shear*zz)*scale
                ar.xy=(15+u,zz);ar.set_position((15,zz))
        kpos=.055+.245*param_q
        a["control_knob"].set_offsets([[kpos,.105]])
        if stage==1:
            trace_q=_smooth((within-.08)/.60)
        _line_to_fraction(a["trace"],model,trace_q)
        idx=min(int(trace_q*(len(model["t"])-1)),len(model["t"])-1)
        x=float(model["x"][idx]);z=float(model["y"][idx])
        for key in ("rock","rock_highlight","rock_glow"):
            _set_scatter(a[key],np.array([x]),np.array([z]))
        _set_scatter(a["impact_marker"],np.array([model["range"]]),np.array([0.0]))
        ax.set_axisbelow(True)
    elif stage==4:
        ax.set_axis_on();ax.grid(False)
        qcam=_smooth(within/.28)
        ax.set_position(local_pos*(1-qcam)+meso_pos*qcam)
        zmin=RC.mesospheric_view_min_altitude_m;zmax=RC.mesospheric_view_max_altitude_m
        ax.set_xlim(-3+(-6000+3)*qcam,75+(6000-75)*qcam)
        ax.set_ylim(-3+(zmin+3)*qcam,25+(zmax-25)*qcam)
        ax.set_ylabel("altitude (km)",fontsize=10.5,color=MUTED)
        ax.set_xlabel("coluna atmosférica  ·  distância esquemática",fontsize=9.5,color=MUTED)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f"{v/1000:g}"))
        if within>.28:ax.set_xlim(-6000,6000);ax.set_ylim(zmin,zmax)
    elif stage==5:
        ax.set_axis_on();ax.grid(False);ax.set_position(broad_pos)
        zoom=_smooth(within/.18)
        ax.set_xlim(-6000,6000)
        camera_bottom=RC.mesospheric_view_min_altitude_m*(1-zoom)
        camera_top=RC.mesospheric_view_max_altitude_m+8_000*zoom
        release_zoom=_smooth((within-RC.release_zoom_start_fraction)/
                             (RC.release_zoom_end_fraction-RC.release_zoom_start_fraction))
        camera_bottom=(camera_bottom*(1-release_zoom)+
                       RC.release_camera_min_altitude_m*release_zoom)
        camera_top=(camera_top*(1-release_zoom)+
                    RC.release_camera_max_altitude_m*release_zoom)
        ax.set_ylim(zmin:=camera_bottom,camera_top)
        ax.set_ylabel("altitude (km)",fontsize=10.5,color=MUTED)
        ax.set_xlabel("coluna atmosférica  ·  trajetória esquemática",fontsize=9.5,color=MUTED)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f"{v/1000:g}"))
        if within<.22:
            a["headline"].set_text("A pergunta continua; agora, uma esfera instrumentada.")
            a["side_body"].set_text("A pedra inspirou a pergunta.\nUma nova esfera, instrumentada,\nserá liberada pelo foguete.")
        rocket=result["mission"]["rocket"]
        qrocket=_smooth(np.clip(within/.78,0,1))
        target_time=qrocket*rocket["release_time_s"]
        j=max(2,min(int(np.searchsorted(rocket["t"],target_time,side="right")),len(rocket["t"])))
        a["rocket_line"].set_data(np.zeros(j),rocket["z"][:j])
        znow=float(rocket["z"][j-1]);lo,hi=ax.get_ylim()
        icon_altitude=float(np.clip((znow-lo)/(hi-lo),0,1))
        _place_rocket_icon(a["rocket_icon"],icon_altitude,flame=(1-qrocket)*.9)
        icon_base=icon_altitude-.16*.5
        payload_center=(.5,icon_base+.16*.5)
        a["payload_marker"].center=payload_center
        a["payload_sensor"].center=payload_center
        _set_scatter(a["release_mark"],np.array([0]),np.array([rocket["apogee_m"]]))
        if within<.67:
            a["rocket_event"].set_text("SUBIDA  •  ESFERA A BORDO")
        elif within<.78:
            a["rocket_event"].set_text("SEPARAÇÃO  •  INÍCIO DA MEDIÇÃO")
        else:
            a["rocket_event"].set_text("ESFERA EM QUEDA  •  O AR DESVIA A TRAJETÓRIA")
        if within>=.78:
            track=result["mission"]["sphere"]
            qdrop=_smooth((within-.78)/.22)
            handoff_time=float(result["observations"]["time_s"][-1])*RC.sphere_descent_preview_fraction
            preview_time=qdrop*handoff_time
            n=max(2,int(np.searchsorted(track["t"],preview_time,side="right")))
            a["sphere_line"].set_data(track["x"][:n],track["z"][:n])
            _set_scatter(a["sphere_marker"],np.array([track["x"][n-1]]),np.array([track["z"][n-1]]))
            _set_scatter(a["sphere_sensor"],np.array([track["x"][n-1]]),np.array([track["z"][n-1]]))
            reference=result["no_wind_sphere"]
            nr=max(2,int(np.searchsorted(reference["t"],preview_time,side="right")))
            a["sphere_reference"].set_data(reference["x"][:nr],reference["z"][:nr])
    elif stage==6:
        ax.set_axis_on();ax.grid(False)
        zoom=_smooth(within/.12)
        ax.set_position(meso_pos*(1-zoom)+broad_pos*zoom)
        bottom=RC.release_camera_min_altitude_m*(1-zoom)
        top=RC.release_camera_max_altitude_m*(1-zoom)+RC.mesospheric_view_max_altitude_m*zoom
        ax.set_xlim(-6500,1600);ax.set_ylim(bottom,top)
        ax.set_xlabel("distância horizontal ao ponto de lançamento (m)",fontsize=10,color=MUTED)
        ax.set_ylabel("altitude (km)",fontsize=10.5,color=MUTED)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f"{v/1000:g}"))
        track=result["mission"]["sphere"];reference=result["no_wind_sphere"];obs=result["observations"]
        tmax=float(obs["time_s"][-1])
        handoff_time=tmax*RC.sphere_descent_preview_fraction
        tnow=handoff_time+within*(tmax-handoff_time)
        nr=max(2,int(np.searchsorted(reference["t"],tnow,side="right")))
        a["sphere_reference"].set_data(reference["x"][:nr],reference["z"][:nr])
        n=max(2,int(np.searchsorted(track["t"],tnow,side="right")))
        a["sphere_line"].set_data(track["x"][:n],track["z"][:n])
        idx=min(n-1,len(track["t"])-1)
        _set_scatter(a["sphere_marker"],np.array([track["x"][idx]]),np.array([track["z"][idx]]))
        _set_scatter(a["sphere_sensor"],np.array([track["x"][idx]]),np.array([track["z"][idx]]))
        upto=np.flatnonzero(obs["time_s"]<=tnow)
        meas=obs["measurement"][upto];valid=obs["valid"][upto]
        sx=obs["station_x_m"][0];good=valid[:,0]
        mx=sx+meas[good,0,0]*np.cos(meas[good,0,1]);mz=meas[good,0,0]*np.sin(meas[good,0,1])
        _set_scatter(a["measurements"],mx,mz)
        for k,line in enumerate(a["ray_lines"]):
            if len(mx)>k*max(1,len(mx)//5):
                ii=min(len(mx)-1,k*max(1,len(mx)//5));line.set_data([sx,mx[ii]],[0,mz[ii]])
                line.set_visible(True)
            else:line.set_visible(False)
    elif stage==7:
        ax.set_axis_on();ax.set_position(broad_pos);ax.set_xlim(-65,65);ax.set_ylim(58,108)
        ax.set_xlabel("vento zonal (m/s)",fontsize=10.5,color=MUTED)
        ax.set_ylabel("altitude (km)",fontsize=10.5,color=MUTED)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f"{v:g}"));ax.grid(color=LIGHT,lw=.7)
        fit=result["retrieval"];poor=result["poor_retrieval"]
        truth=WIND_TRUTH;ztrue=np.asarray(truth.altitude_m)/1000
        a["wind_true_line"].set_data(truth.speed_m_s,ztrue)
        zfit=fit["wind_altitude_m"]/1000;zp=poor["wind_altitude_m"]/1000
        # Rebuild only the visible uncertainty ribbons; they are driven by their own covariance.
        for key in ("wind_band","poor_band"):
            artist=a.get(key)
            if artist is not None:
                try:artist.remove()
                except ValueError:pass
                a[key]=None
        if within<.14:
            a["wind_caption"].set_text("Perfil prescrito: conhecido apenas pelo simulador.")
            a["wind_true_line"].set_visible(True);a["wind_true_line"].set_alpha(1)
            a["wind_fit_line"].set_visible(False);a["wind_poor_line"].set_visible(False)
        elif within<.20:
            a["wind_caption"].set_text("A verdade fica oculta durante a recuperação.")
            a["wind_true_line"].set_visible(False);a["wind_fit_line"].set_visible(False)
            a["wind_poor_line"].set_visible(False)
        else:
            reveal=np.clip((within-.20)/.33,0,1)
            n=max(2,int(reveal*len(zfit)))
            a["wind_fit_line"].set_data(fit["wind_m_s"][:n],zfit[:n])
            a["wind_fit_line"].set_visible(True);a["wind_fit_line"].set_alpha(1)
            a["wind_band"]=ax.fill_betweenx(zfit[:n],
                fit["wind_m_s"][:n]-fit["total_wind_std_m_s"][:n],
                fit["wind_m_s"][:n]+fit["total_wind_std_m_s"][:n],
                color=ORANGE,alpha=.18,zorder=2)
            a["wind_true_line"].set_visible(within>=.69)
            a["wind_true_line"].set_alpha(float(np.clip((within-.69)/.08,0,1)))
            if within<.69:
                a["wind_caption"].set_text("O perfil recuperado emerge das observações.")
            elif within<.84:
                a["wind_caption"].set_text("Agora comparamos a recuperação à verdade sintética.")
            else:
                a["wind_caption"].set_text("Rastreamento degradado: menor resolução e maior incerteza.")
                a["wind_poor_line"].set_data(poor["wind_m_s"],zp)
                a["wind_poor_line"].set_visible(True)
                a["poor_band"]=ax.fill_betweenx(zp,
                    poor["wind_m_s"]-poor["total_wind_std_m_s"],
                    poor["wind_m_s"]+poor["total_wind_std_m_s"],
                    color="#667784",alpha=.12,zorder=1)
        a["wind_caption"].set_visible(True)
        if within>=.84:
            a["side_body"].set_text("Mesmo modelo direto.\nMenos dados e maior ruído\nreduzem o que se recupera.")
            a["side_stats"].set_text(
                f"Nominal: RMSE {result['evaluation']['rmse_m_s']:.1f} m/s\n"
                f"σ total: {np.median(fit['total_wind_std_m_s']):.1f} m/s\n"
                f"Degradado: RMSE {result['poor_evaluation']['rmse_m_s']:.1f} m/s\n"
                f"σ total: {np.median(poor['total_wind_std_m_s']):.1f} m/s")
            a["side_question"].set_text("A utilidade depende\ndas observações.")
    else:
        ax.set_position([.05,.13,.90,.70]);ax.set_xlim(0,1);ax.set_ylim(0,1)
        ax.set_axis_off();ax.set_xlabel("");ax.set_ylabel("")

    if stage==8:
        a["final_question"].set_alpha(_smooth((within-.05)/.12))
        a["final_message"].set_alpha(_smooth((within-.25)/.15))
    else:
        a["final_question"].set_alpha(0);a["final_message"].set_alpha(0)
        a["final_banner"].set_visible(False)
    return []


def render_master(path=None,playback_seconds=None):
    is_default_master=path is None and playback_seconds is None
    path=Path(path) if path is not None else OUT/"master_animation.mp4"
    data=_prepare_render_data()
    a=_make_scene(data)
    duration=float(playback_seconds or sum(RC.stage_durations_s))
    speed=sum(RC.stage_durations_s)/duration
    frames=int(round(duration*RC.fps))
    if not shutil.which("ffmpeg"):
        suffix="_30s" if playback_seconds else ""
        folder=OUT/f"mission_keynote_frames{suffix}"; folder.mkdir(exist_ok=True)
        print(f"ffmpeg ausente; salvando {frames} PNGs em {folder}",flush=True)
        for i in range(frames):
            update_frame(i,a,speed=speed); a["fig"].savefig(folder/f"frame_{i:05d}.png",dpi=RC.dpi,facecolor=WHITE)
        plt.close(a["fig"]); return folder
    writer=animation.FFMpegWriter(fps=RC.fps,codec="libx264",bitrate=5000,
        extra_args=["-pix_fmt","yuv420p","-preset","medium","-movflags","+faststart"])
    print(f"Rendering mission animation: {frames} frames / {duration:.0f} s",flush=True)
    with writer.saving(a["fig"],str(path),dpi=RC.dpi):
        for i in range(frames):
            update_frame(i,a,speed=speed); writer.grab_frame()
            if i and i%(RC.fps*10)==0: print(f"  {i/RC.fps:4.0f} s",flush=True)
    plt.close(a["fig"])
    if is_default_master:
        shutil.copy2(path,OUT/"keynote_master.mp4")
    return path


def render_short_version():
    """Render the redesigned 30 s physical experiment and slide-ready clips."""
    from render_visual_stages import render_master_and_stages
    render_master_and_stages()
    return OUT/"keynote_30s.mp4"


def render_segments(master=None):
    ff=shutil.which("ffmpeg")
    if not ff:
        print("ffmpeg ausente; não é possível codificar MP4s individuais."); return
    master=Path(master) if master else OUT/"master_animation.mp4"
    names=["projectile_motion","aerodynamic_drag","wind_effects","mesosphere_transition",
           "rocket_and_release","synthetic_measurements","wind_retrieval","experiment_design"]
    starts=np.r_[0,np.cumsum(RC.stage_durations_s)]
    for i,name in enumerate(names):
        dest=OUT/f"{name}.mp4"; start=starts[i]; duration=RC.stage_durations_s[i]
        cmd=[ff,"-y","-ss",f"{start:.3f}","-i",str(master),"-t",f"{duration:.3f}",
            "-an","-c:v","libx264","-preset","medium","-crf","19","-pix_fmt","yuv420p",
            "-movflags","+faststart",str(dest)]
        subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        print("Wrote",dest.name)


def render_figures_and_keyframes():
    result=_prepare_render_data()
    # Prior opening scene compared with the full mesospheric mission trajectory.
    ideal=result["local_ideal"]
    drag=result["local_drag"]
    wind=result["local_shear"]
    mission=result["mission"]; sphere=mission["sphere"]
    fig,(a,b)=plt.subplots(1,2,figsize=(16,8),dpi=150,facecolor=WHITE)
    fig.subplots_adjust(left=.08,right=.96,top=.76,bottom=.13,wspace=.25)
    fig.text(.08,.94,"XI SBGEA 2026  •  DA PEDRA À MISSÃO CIENTÍFICA",color=MUTED,fontsize=12,fontweight="bold")
    fig.text(.08,.875,"A física escala; as perguntas também",color=NAVY,fontsize=27,fontweight="bold")
    a.plot(ideal["x"],ideal["y"],ls="--",color=BLUE,lw=2.5,label="ideal")
    a.plot(drag["x"],drag["y"],color=ORANGE,lw=2.6,label="arrasto")
    a.plot(wind["x"],wind["y"],color=GREEN,lw=2.5,label="vento u(z)")
    a.set(xlim=(-2,70),ylim=(0,21),xlabel="distância (m)",ylabel="altura (m)")
    a.grid(color=LIGHT);a.legend(frameon=False)
    rock_vertices=np.array([[-1.0,-.28],[-.82,.36],[-.27,.86],[.30,.96],
        [.72,.54],[.94,-.05],[.54,-.66],[-.30,-.83],[-.86,-.58],[-1.0,-.28]])
    rock_codes=[MarkerPath.MOVETO]+[MarkerPath.LINETO]*(len(rock_vertices)-2)+[MarkerPath.CLOSEPOLY]
    a.scatter([0],[1.7],marker=MarkerPath(rock_vertices,rock_codes),s=180,
        facecolor="#8C735E",edgecolor="#4D453F",lw=.9,zorder=5)
    a.set_title("01–03  •  A pedra, o arrasto e o vento",loc="left",color=NAVY,fontsize=16)
    a2=b
    a2.plot(sphere["x"]/1000,sphere["z"]/1000,color=BLUE,lw=2.5,label="esfera instrumentada")
    obs=result["observations"];meas=obs["measurement"][:,0,:];good=obs["valid"][:,0]
    mx=obs["station_x_m"][0]+meas[good,0]*np.cos(meas[good,1])
    mz=meas[good,0]*np.sin(meas[good,1])
    a2.scatter(mx/1000,mz/1000,s=9,color=ORANGE,alpha=.65,label="radar sintético")
    a2.set(xlim=(-.2,1.0),ylim=(58,108),xlabel="distância horizontal (km)",ylabel="altitude (km)")
    a2.grid(color=LIGHT);a2.legend(frameon=False,loc="lower right")
    a2.set_title("04–07  •  Uma nova esfera na coluna mesosférica",loc="left",color=NAVY,fontsize=16)
    fig.savefig(OUT/"trajectory_comparison.png",dpi=180,facecolor=WHITE);plt.close(fig)

    fit=result["retrieval"];truth=WIND_TRUTH;poor=result["poor_retrieval"]
    fig,axs=plt.subplots(1,2,figsize=(15,8),dpi=170,facecolor=WHITE)
    fig.subplots_adjust(left=.08,right=.97,top=.80,bottom=.20,wspace=.23)
    fig.text(.08,.95,"XI SBGEA 2026  •  OBSERVAÇÕES SINTÉTICAS",color=MUTED,fontsize=12,fontweight="bold")
    fig.text(.08,.88,"O vento recuperado tem incerteza",color=NAVY,fontsize=25,fontweight="bold")
    for ax,case,title,rmse,color in (
        (axs[0],fit,"Rastreamento nominal",result["evaluation"]["rmse_m_s"],ORANGE),
        (axs[1],poor,"Rastreamento degradado",result["poor_evaluation"]["rmse_m_s"],"#667784")):
        zcase=case["wind_altitude_m"]/1000
        ax.fill_betweenx(zcase,case["wind_m_s"]-case["total_wind_std_m_s"],
            case["wind_m_s"]+case["total_wind_std_m_s"],color=color,alpha=.16,
            label="σ total (medição + piso de modelo)")
        ax.plot(truth.speed_m_s,np.array(truth.altitude_m)/1000,color=GREEN,lw=2.7,
            label="verdade sintética · avaliação")
        ax.errorbar(case["wind_m_s"],zcase,xerr=case["measurement_wind_std_m_s"],
            fmt="o-",color=color,lw=2.2,capsize=3,label="recuperado; barras: σ formal")
        ax.set(xlabel="vento zonal (m/s)",ylabel="altitude (km)",ylim=(58,112),xlim=(-70,90))
        ax.set_title(f"{title}  ·  RMSE {rmse:.1f} m/s",loc="left",color=NAVY,
                     fontsize=14,fontweight="bold",pad=12)
        ax.grid(color=LIGHT)
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.095),ncol=3,
        frameon=False,fontsize=9.5)
    fig.text(.08,.035,"RMSE contra o perfil prescrito, usado somente na avaliação após o ajuste.",
        color=MUTED,fontsize=9.5)
    fig.savefig(OUT/"wind_retrieval_comparison.png",dpi=180,facecolor=WHITE);plt.close(fig)

    folder=OUT/"mission_keyframes";folder.mkdir(exist_ok=True)
    slides=OUT/"slides";slides.mkdir(exist_ok=True)
    slide_names=["pedra","arrasto","vento","mesosfera","lancamento",
                 "rastreamento","recuperacao","simulador"]
    data=result
    starts=np.cumsum((0,)+RC.stage_durations_s[:-1])
    frame_fraction=np.array([.78,.78,.78,.78,.98,.90,.90,.90])
    frame_times=starts+np.array(RC.stage_durations_s)*frame_fraction
    for i,t in enumerate(frame_times,1):
        scene=_make_scene(data);update_frame(int(t*RC.fps),scene)
        scene["fig"].savefig(folder/f"stage_{i:02d}.png",dpi=RC.keyframe_dpi,facecolor=WHITE)
        scene["fig"].savefig(slides/f"slide_{i:02d}_{slide_names[i-1]}.png",
                             dpi=RC.keyframe_dpi,facecolor=WHITE)
        plt.close(scene["fig"])
    print("Wrote static figures, eight mission keyframes, and eight slide PNGs")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--all",action="store_true")
    p.add_argument("--master",action="store_true")
    p.add_argument("--short",action="store_true",help="render a 30 s accelerated keynote cut")
    p.add_argument("--segments",action="store_true")
    p.add_argument("--figures",action="store_true")
    args=p.parse_args()
    if not any(vars(args).values()): args.all=True
    if args.all or args.master: render_master()
    if args.all or args.segments: render_segments()
    if args.all or args.figures: render_figures_and_keyframes()
    if args.all or args.short: render_short_version()

if __name__=="__main__": main()
