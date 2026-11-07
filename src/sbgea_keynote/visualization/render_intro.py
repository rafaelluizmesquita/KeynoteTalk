"""Render the keynote animation, stage cuts, figures, and transparent rock frames."""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from PIL import Image, ImageDraw, ImageFilter

from ..config import PHYSICS, RENDER
from ..physics.projectile import (ideal_projectile, simulate_projectile, study_projectile,
                                  orbital_examples, simulate_satellite, air_density)
from ..paths import output_dir

OUT = output_dir("animation")
NAVY = "#082F73"
BLUE = "#2686D0"
GREEN = "#70AE54"
ORANGE = "#EC8135"
INK = "#16324F"
MUTED = "#5C6E80"
PALE = "#E8F1F7"
EARTH = "#E8F2F6"
WHITE = "#FFFFFF"


def _draw_rock(ax, x, y, *, size=190, alpha=1.0, zorder=20):
    """Recognizable irregular stone marker; deliberately illustrated out of scale."""
    verts = np.array([[-.56,-.30],[-.48,.25],[-.19,.51],[.26,.48],[.56,.17],
                      [.47,-.31],[.16,-.52],[-.27,-.49],[-.56,-.30]])
    path = MplPath(verts, [MplPath.MOVETO] + [MplPath.LINETO]*7 + [MplPath.CLOSEPOLY])
    glow = ax.scatter([x], [y], s=size*2.2, marker=path, color=ORANGE, alpha=.17*alpha,
                      linewidths=0, zorder=zorder-2)
    body = ax.scatter([x], [y], s=size, marker=path, facecolor="#9A785F", edgecolor=NAVY,
                      linewidths=1.25, alpha=alpha, zorder=zorder)
    # small stone facet
    facet = ax.scatter([x], [y], s=size*.22, marker=path, facecolor="#D6B99D",
                       edgecolor="none", alpha=.8*alpha, zorder=zorder+1)
    return [glow, body, facet]


def _state_at(model, fraction):
    i = int(np.clip(fraction, 0, 1) * (len(model["t"]) - 1))
    return i, (model["x"][i], PHYSICS.earth_radius + model["y"][i])


def _set_reveal(line, model, fraction, y_offset=0.0):
    n = max(2, int(np.clip(fraction, 0, 1) * (len(model["t"]) - 1)))
    line.set_data(model["x"][:n], model["y"][:n] + y_offset)


def _add_person(ax):
    # Minimal silhouette at the launch point; size follows the physical local scale.
    head = patches.Circle((-5.5, PHYSICS.earth_radius + 1.58), .16,
                          facecolor=NAVY, edgecolor="none", zorder=10)
    ax.add_patch(head)
    parts = [((-5.5, PHYSICS.earth_radius+1.43),(-5.5, PHYSICS.earth_radius+.72)),
             ((-5.5, PHYSICS.earth_radius+1.27),(-4.65, PHYSICS.earth_radius+1.12)),
             ((-5.5, PHYSICS.earth_radius+1.25),(-6.15, PHYSICS.earth_radius+.92)),
             ((-5.5, PHYSICS.earth_radius+.72),(-6.05, PHYSICS.earth_radius+.08)),
             ((-5.5, PHYSICS.earth_radius+.72),(-4.95, PHYSICS.earth_radius+.08))]
    for a, b in parts:
        ax.add_line(Line2D([a[0],b[0]],[a[1],b[1]], color=NAVY, linewidth=3.0,
                           solid_capstyle="round", zorder=10))
    return [head]


def _study_inset(ax, const_model, exp_model):
    inset = ax.inset_axes([.535, .12, .43, .38], zorder=30)
    inset.set_facecolor(WHITE)
    for sp in inset.spines.values():
        sp.set_color("#B9C9D6")
    inset.plot(const_model["x"]/1000, const_model["y"]/1000, color=BLUE,
               lw=1.7, ls="--", label="ρ constante")
    inset.plot(exp_model["x"]/1000, exp_model["y"]/1000, color=ORANGE,
               lw=2.2, label="ρ varia com y")
    inset.set_xlim(0, 4.3); inset.set_ylim(0, 1.45)
    inset.set_title("Escala ampliada • projétil de estudo", fontsize=9.2,
                    color=NAVY, loc="left", pad=4, fontweight="bold")
    inset.set_xlabel("distância (km)", fontsize=8, labelpad=1)
    inset.set_ylabel("altitude (km)", fontsize=8, labelpad=1)
    inset.tick_params(labelsize=7, length=2, colors=MUTED)
    inset.grid(color="#E6EDF2", lw=.55)
    inset.legend(loc="upper right", fontsize=7, frameon=False, handlelength=1.5)
    return inset


def _make_scene():
    fig = plt.figure(figsize=(RENDER.width/RENDER.dpi, RENDER.height/RENDER.dpi),
                     dpi=RENDER.dpi, facecolor=WHITE)
    ax = fig.add_axes([.045,.185,.705,.57])
    ax.set_facecolor(WHITE)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    R = PHYSICS.earth_radius
    local_xlim = (-11.0, 82.0)
    local_ylim = (R-3.5, R+24.5)
    world_xlim = (-2.10*R, 2.10*R)
    world_ylim = (-1.05*R, 1.05*R)
    escape_xlim = (-36_000_000.0, 36_000_000.0)
    escape_ylim = (-18_500_000.0, 18_500_000.0)
    satellite_xlim = world_xlim
    satellite_ylim = (-1.25*R, 1.25*R)
    ax.set_xlim(*local_xlim); ax.set_ylim(*local_ylim)

    # Layered Earth remains hidden until the camera zooms out to the planetary scale.
    earth = patches.Circle((0,0), R, facecolor=EARTH, edgecolor=BLUE, linewidth=1.3,
                           alpha=0, zorder=0)
    ax.add_patch(earth)
    ground, = ax.plot([-20, 95], [R,R], color=NAVY, linewidth=1.5, alpha=.8, zorder=6)
    local_label = ax.text(66, R+.72, "superfície local", color=MUTED, fontsize=9, zorder=6)
    person = _add_person(ax)

    ideal = ideal_projectile(samples=800)
    drag = simulate_projectile(drag=True, density="constant", wind="calm", samples=800)
    wind0 = simulate_projectile(drag=True, density="constant", wind="constant", samples=800)
    wind_shear = simulate_projectile(drag=True, density="constant", wind="shear", samples=800)
    rho_exp = simulate_projectile(drag=True, density="exponential", wind="calm", samples=800)
    local_models = [ideal, drag, wind0, wind_shear, rho_exp]
    line_ideal, = ax.plot(ideal["x"], ideal["y"]+R, color=BLUE, lw=2.2,
                          ls=(0,(5,4)), alpha=.95, zorder=3)
    line_drag_bg, = ax.plot(drag["x"], drag["y"]+R, color=ORANGE, lw=2.3,
                            alpha=.20, zorder=3)
    line_drag, = ax.plot([], [], color=ORANGE, lw=2.9, zorder=5)
    line_wind0, = ax.plot([], [], color=NAVY, lw=2.0, ls=(0,(3,3)), alpha=.85,zorder=4)
    line_wind, = ax.plot([], [], color=GREEN, lw=2.5, zorder=5)
    line_rho, = ax.plot([], [], color=ORANGE, lw=2.6, ls=(0,(2,2)), zorder=5)
    line_ideal_trace, = ax.plot([], [], color=BLUE, lw=3.0, alpha=.75, zorder=7)

    # Subtle wind profile: arrows represent direction and relative speed, not vector scale.
    arrows = []
    for height in (4, 12, 20):
        speed = PHYSICS.wind_u0 + PHYSICS.wind_shear * height
        arr = ax.annotate("", xy=(8+speed*1.1, R+height), xytext=(8, R+height),
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4, alpha=.54), zorder=8)
        arrows.append(arr)
    wind_label = ax.text(7, R+21.7, "vento: u(y)", fontsize=9, color=GREEN, alpha=.8, zorder=8)

    study_c = study_projectile(density="constant", samples=700)
    study_e = study_projectile(density="exponential", samples=700)
    inset = _study_inset(ax, study_c, study_e)
    inset.set_visible(False)

    # Orbital examples are true numerical ECI solutions, all from 300 km altitude.
    orb = orbital_examples()
    orbit_specs = [
        (orb["suborbital"], ORANGE, "Suborbital"),
        (orb["circular"], BLUE, "Circular"),
        (orb["escape"], GREEN, "Escape")]
    orbit_lines = []
    for model, col, label in orbit_specs:
        line, = ax.plot([], [], color=col, lw=2.6, alpha=.95, zorder=5, label=label)
        orbit_lines.append(line)
    orbital_legend = ax.legend(handles=[
        Line2D([],[],color=ORANGE,lw=2.6,label="Suborbital"),
        Line2D([],[],color=BLUE,lw=2.6,label="Circular"),
        Line2D([],[],color=GREEN,lw=2.6,label="Escape")],
        loc="upper left",bbox_to_anchor=(.02,.94),frameon=True,
        facecolor=WHITE,edgecolor="#D9E3EA",framealpha=.95,
        fontsize=9.5,handlelength=1.8,labelspacing=.45)
    orbital_legend.set_visible(False)
    # Atmosphere shell, physically scaled; 450 km thickness is visible at Earth scale.
    atmosphere = patches.Circle((0,0), R+500_000, facecolor="none", edgecolor=BLUE,
                                linewidth=1.0, linestyle=(0,(3,4)), alpha=0, zorder=1)
    ax.add_patch(atmosphere)
    atmosphere_fill_outer = patches.Circle((0,0), R+500_000, facecolor="#B8D9EA",
                                           edgecolor="none", alpha=0, zorder=.5)
    ax.add_patch(atmosphere_fill_outer)
    # The Earth circle is drawn over the translucent shell interior.
    earth.set_zorder(1)
    atmosphere.set_zorder(2)

    sat_drag = simulate_satellite(samples=3200, drag=True)
    sat_vac = simulate_satellite(samples=800, drag=False)
    r0 = R + PHYSICS.satellite_altitude
    alt_delta = sat_drag["final_altitude"] - sat_drag["initial_altitude"]
    radial_shift = abs(alt_delta) * PHYSICS.visual_radial_exaggeration
    r_vis_end = r0 - radial_shift
    # A two-turn schematic spiral uses the integrated 15-day radial loss; radial
    # displacement is enlarged only in this illustration and is explicitly labeled.
    theta = np.linspace(0, 4*np.pi, 900)
    r_spiral = np.linspace(r0, r_vis_end, len(theta))
    spx, spy = -r_spiral*np.sin(theta), r_spiral*np.cos(theta)
    sat_spiral, = ax.plot(spx, spy, color=ORANGE, lw=2.8, alpha=0, zorder=7)
    sat_ref, = ax.plot(r0*np.cos(np.linspace(0,2*np.pi,500)),
                       r0*np.sin(np.linspace(0,2*np.pi,500)), color=BLUE,
                       lw=1.8, ls=(0,(5,4)), alpha=0, zorder=4)
    sat_marker = ax.scatter([0],[r0],s=140,marker="o",facecolor=ORANGE,
                            edgecolor=NAVY,linewidth=1.2,alpha=0,zorder=12)

    # Header and right-side scientific notes.
    fig.text(.045,.958,"XI SBGEA 2026   •   GEofísica e Aeronomia Espacial".upper(),
             color=MUTED,fontsize=10.5,fontweight="bold",va="top")
    headline = fig.text(.045,.895,"Uma pedra. Uma pergunta.",color=NAVY,
                        fontsize=22.5,fontweight="bold",va="top")
    fig.add_artist(Line2D([.765,.765],[.16,.80],transform=fig.transFigure,
                          color="#D9E3EA",lw=1.0))
    stage_tag = fig.text(.79,.785,"MODELO 01 / 06",color=BLUE,fontsize=10.5,
                         fontweight="bold",va="top")
    side_title = fig.text(.79,.738,"O caso ideal",color=NAVY,fontsize=19,
                          fontweight="bold",va="top",linespacing=1.1)
    side_body = fig.text(.79,.635,"",color=INK,fontsize=11.8,va="top",
                         linespacing=1.45)
    side_stat = fig.text(.79,.36,"",color=MUTED,fontsize=10.3,va="top",
                         linespacing=1.5)
    side_question = fig.text(.79,.205,"",color=ORANGE,fontsize=11.8,
                             fontweight="bold",va="top",linespacing=1.25)
    footer_rule = Line2D([.045,.95],[.115,.115],transform=fig.transFigure,
                         color="#D9E3EA",lw=1.0)
    fig.add_artist(footer_rule)
    labels = ["01  Ideal", "02  Arrasto", "03  Vento", "04  Atmosfera",
              "05  Órbita", "06  Satélite"]
    footer_items = []
    xs = np.linspace(.055,.91,6)
    for i,(x,lbl) in enumerate(zip(xs,labels)):
        footer_items.append(fig.text(x,.072,lbl,color="#A6B2BC",fontsize=10.5,
                                     fontweight="bold",va="center"))
    footnote = fig.text(.045,.027,"Ilustração científica • tamanhos visuais fora de escala",
                        color="#8494A1",fontsize=8.8,va="bottom")

    # A compact actual altitude-history chart for stage 6.
    sat_chart = fig.add_axes([.795,.39,.17,.13])
    sat_chart.set_facecolor("white")
    days = sat_drag["t"]/86400.0
    sat_chart.plot(days, sat_drag["altitude"]/1000, color=ORANGE, lw=1.8,
                   label="com arrasto")
    sat_chart.plot(days, np.full_like(days, PHYSICS.satellite_altitude/1000),
                   color=BLUE, lw=1.3, ls="--", label="sem arrasto")
    sat_chart.tick_params(labelsize=7, colors=MUTED, length=2, pad=1)
    sat_chart.set_xlabel("dias", fontsize=8, color=MUTED, labelpad=0)
    sat_chart.set_ylabel("altitude (km)", fontsize=8, color=MUTED, labelpad=1)
    sat_chart.grid(color="#E6EDF2", lw=.5)
    sat_chart.legend(fontsize=6.7, frameon=False, loc="lower left", ncol=1)
    for sp in sat_chart.spines.values(): sp.set_color("#D9E3EA")
    sat_chart.set_visible(False)

    # A tiny orbital marker for the current trajectory stage.
    glow = ax.scatter([0],[R+PHYSICS.launch_height],s=580,marker="o",
                      color=BLUE,alpha=.10,linewidths=0,zorder=18)
    rock = ax.scatter([0],[R+PHYSICS.launch_height],s=230,marker="o",
                      facecolor="#9A785F",edgecolor=NAVY,linewidth=1.0,
                      zorder=20)
    # Irregular polygon overlays on circular scatter, in display points.
    stone = _draw_rock(ax,0,R+PHYSICS.launch_height,size=215,zorder=21)

    artists = dict(fig=fig, ax=ax, earth=earth, atmosphere=atmosphere,
        atmosphere_fill_outer=atmosphere_fill_outer, ground=ground, person=person,
        ideal=ideal, drag=drag, wind0=wind0, wind_shear=wind_shear,
        rho_exp=rho_exp, line_ideal=line_ideal, line_drag_bg=line_drag_bg,
        line_drag=line_drag,line_wind0=line_wind0,line_wind=line_wind,
        line_rho=line_rho,line_ideal_trace=line_ideal_trace,arrows=arrows,
        wind_label=wind_label, inset=inset,study_c=study_c,study_e=study_e,
        orb=orb,orbit_lines=orbit_lines,orbit_specs=orbit_specs,
        orbital_legend=orbital_legend,
        sat_drag=sat_drag,sat_vac=sat_vac,sat_spiral=sat_spiral,sat_ref=sat_ref,
        sat_marker=sat_marker,radial_shift=radial_shift,r0=r0,
        headline=headline,stage_tag=stage_tag,side_title=side_title,
        side_body=side_body,side_stat=side_stat,side_question=side_question,
        footer_items=footer_items,sat_chart=sat_chart,glow=glow,rock=rock,
        stone=stone,local_xlim=local_xlim,local_ylim=local_ylim,
        world_xlim=world_xlim,world_ylim=world_ylim,escape_xlim=escape_xlim,
        escape_ylim=escape_ylim,satellite_xlim=satellite_xlim,
        satellite_ylim=satellite_ylim,local_label=local_label)
    return artists


def _set_side(a, stage):
    if stage == 1:
        a["stage_tag"].set_text("MODELO 01 / 06")
        a["side_title"].set_text("Começamos\npelo ideal.")
        a["side_body"].set_text("F = ma\n\ny(t) = y₀ + v₀ sen(θ)t\n        − ½gt²")
        a["side_stat"].set_text(f"v₀={PHYSICS.launch_speed:.0f} m/s • θ={PHYSICS.launch_angle_deg:.0f}°\nAlcance ideal: {a['ideal']['range']:.1f} m\nAnalítica confere com\na integração numérica.")
        a["side_question"].set_text("Primeira aproximação:\ngravidade constante.")
    elif stage == 2:
        d=a["drag"]; i=a["ideal"]
        a["stage_tag"].set_text("MODELO 02 / 06")
        a["side_title"].set_text("O ar também\nexerce força.")
        a["side_body"].set_text("Fₐ = −½ ρ Cᴅ A |v|v")
        a["side_stat"].set_text(f"Ideal: {i['range']:.1f} m • {i['flight_time']:.2f} s\nArrasto: {d['range']:.1f} m • {d['flight_time']:.2f} s\nΔ alcance: {d['range']-i['range']:+.1f} m\nRocha: {__import__('config').rock_mass():.2f} kg • d={PHYSICS.rock_diameter*100:.0f} cm")
        a["side_question"].set_text("E se levarmos em conta\na resistência do ar?")
    elif stage == 3:
        a["stage_tag"].set_text("MODELO 03 / 06")
        a["side_title"].set_text("O ar está\nem movimento.")
        a["side_body"].set_text("v_rel = v_rocha − u_ar(y)\nu(y) = u₀ + Sy")
        a["side_stat"].set_text(f"Sem vento: {a['drag']['range']:.1f} m\nVento constante: {a['wind0']['range']:.1f} m\nCom cisalhamento: {a['wind_shear']['range']:.1f} m\nu₀={PHYSICS.wind_u0:.0f} m/s • S={PHYSICS.wind_shear:.2f} s⁻¹\ng permanece constante.")
        a["side_question"].set_text("E se o próprio ar\nestiver em movimento?")
    elif stage == 4:
        lo=a["rho_exp"]; base=a["drag"]
        a["stage_tag"].set_text("MODELO 04 / 06")
        a["side_title"].set_text("A atmosfera\ntambém é física.")
        a["side_body"].set_text("ρ(y) = ρ₀ exp(−y/H)")
        a["side_stat"].set_text(f"H={PHYSICS.scale_height/1000:.1f} km • isoterma\nsimplificada.\nPedra (dezenas de m):\nΔ alcance = {lo['range']-base['range']:+.3f} m\nEscala ampliada no quadro:\nprojétil {PHYSICS.study_mass:.0f} kg, {PHYSICS.study_speed:.0f} m/s.")
        a["side_question"].set_text("O efeito cresce quando\na escala do problema muda.")
    elif stage == 5:
        o=a["orb"]
        a["stage_tag"].set_text("MODELO 05 / 06")
        a["side_title"].set_text("Newton, agora\nna escala da Terra.")
        a["side_body"].set_text("a⃗ = −μ r⃗ / |r⃗|³")
        a["side_stat"].set_text(f"Corpo de teste, a 300 km\nv_sub={PHYSICS.suborbital_speed/1000:.2f} km/s\nv_c={o['circular_speed']/1000:.2f} km/s\nv_esc={o['escape_speed']/1000:.2f} km/s\nÓrbita = queda livre contínua.")
        a["side_question"].set_text("E se o objeto tivesse\nvelocidade suficiente\npara entrar em órbita?")
    elif stage == 6:
        s=a["sat_drag"]
        a["stage_tag"].set_text("MODELO 06 / 06")
        a["side_title"].set_text("A mesma física.\nNovas escalas.")
        a["side_body"].set_text("a_g=−μr/|r|³\na_d=−½ρ(C_dA/m)\n|v_rel|v_rel")
        a["side_body"].set_fontsize(10.8)
        a["side_stat"].set_text(f"Co-rotação • termosfera\nexponencial ilustrativa\n(não operacional) • {s['days']:.0f} dias\nΔ altitude: {(s['final_altitude']-s['initial_altitude'])/1000:.1f} km\nTraço radial visual: {PHYSICS.visual_radial_exaggeration:.0f}×")
        a["side_question"].set_text("Quanto vale prever\nmelhor a trajetória de\num satélite?")


def _smoothstep(x):
    x = np.clip(x,0,1)
    return x*x*(3-2*x)


def _set_artist_pos(scatter, pos):
    scatter.set_offsets(np.array([[pos[0],pos[1]]]))


def update_frame(frame, a):
    t = frame / RENDER.fps
    ax=a["ax"]; R=PHYSICS.earth_radius
    if t < 14: stage=1
    elif t < 27: stage=2
    elif t < 40: stage=3
    elif t < 54: stage=4
    elif t < 75: stage=5
    else: stage=6

    # Three smooth views: local throw → Earth and LEO → escape scale → LEO.
    lx0,lx1=a["local_xlim"]; ly0,ly1=a["local_ylim"]
    wx0,wx1=a["world_xlim"]; wy0,wy1=a["world_ylim"]
    ex0,ex1=a["escape_xlim"]; ey0,ey1=a["escape_ylim"]
    sx0,sx1=a["satellite_xlim"]; sy0,sy1=a["satellite_ylim"]
    if t < 54:
        camera=0.0
        xlim,ylim=a["local_xlim"],a["local_ylim"]
    elif t < 58:
        camera=_smoothstep((t-54)/4.0)
        xlim=(lx0+(wx0-lx0)*camera,lx1+(wx1-lx1)*camera)
        ylim=(ly0+(wy0-ly0)*camera,ly1+(wy1-ly1)*camera)
    elif t < 75:
        camera=1.0
        q=_smoothstep((t-58)/11.5)
        xlim=(wx0+(ex0-wx0)*q,wx1+(ex1-wx1)*q)
        ylim=(wy0+(ey0-wy0)*q,wy1+(ey1-wy1)*q)
    else:
        camera=1.0
        q=_smoothstep((t-75)/3.0)
        xlim=(ex0+(sx0-ex0)*q,ex1+(sx1-ex1)*q)
        ylim=(ey0+(sy0-ey0)*q,ey1+(sy1-ey1)*q)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    a["earth"].set_alpha(.97*camera)
    a["atmosphere"].set_alpha(.35 if stage==6 else 0.0)
    a["atmosphere_fill_outer"].set_alpha(.10 if stage==6 else 0.0)
    a["ground"].set_alpha(max(0,1-camera))
    a["local_label"].set_alpha(max(0,1-camera))
    for p in a["person"]: p.set_alpha(max(0,1-camera))
    # Local model traces persist as assumptions are added, then dissolve during the
    # explicit camera-scale transition to ECI coordinates.
    fade=max(0,1-camera)
    a["line_ideal"].set_alpha(.90*fade)
    a["line_drag_bg"].set_alpha((.18 if stage>=2 else 0)*fade)
    a["line_ideal_trace"].set_alpha(.8*fade)
    a["line_drag"].set_alpha(fade if stage>=2 else 0)
    a["line_wind0"].set_alpha(.8*fade if stage>=3 else 0)
    a["line_wind"].set_alpha(fade if stage>=3 else 0)
    a["line_rho"].set_alpha(fade if stage>=4 else 0)
    a["inset"].set_visible(stage==4)
    for ar in a["arrows"]: ar.set_visible(stage==3 or stage==4)
    a["wind_label"].set_visible(stage==3 or stage==4)
    # Reveal each new numerical curve rather than drawing it all at once.
    if stage == 1:
        q=(t-1.5)/8.5
        _set_reveal(a["line_ideal_trace"],a["ideal"],q,R)
        _,p=_state_at(a["ideal"],q)
        local_pos=p
    elif stage == 2:
        _set_reveal(a["line_ideal_trace"],a["ideal"],1,R)
        q=(t-15.5)/9.5
        _set_reveal(a["line_drag"],a["drag"],q,R)
        _,local_pos=_state_at(a["drag"],q)
    elif stage == 3:
        _set_reveal(a["line_drag"],a["drag"],1,R)
        q0=(t-28.5)/4.8; qs=(t-33.2)/5.5
        _set_reveal(a["line_wind0"],a["wind0"],q0,R)
        _set_reveal(a["line_wind"],a["wind_shear"],qs,R)
        _,local_pos=_state_at(a["wind_shear"],qs)
    elif stage == 4:
        _set_reveal(a["line_drag"],a["drag"],1,R)
        _set_reveal(a["line_wind0"],a["wind0"],1,R)
        _set_reveal(a["line_wind"],a["wind_shear"],1,R)
        q=(t-40.5)/7.0
        _set_reveal(a["line_rho"],a["rho_exp"],q,R)
        _,local_pos=_state_at(a["rho_exp"],q)
    else:
        local_pos=(0,R+PHYSICS.launch_height)
    # Rock marker turns into a generic test particle after the scale transition.
    if stage <= 4:
        _set_artist_pos(a["rock"], local_pos)
        _set_artist_pos(a["glow"], local_pos)
        for sc in a["stone"]: _set_artist_pos(sc,local_pos)
        a["rock"].set_alpha(fade); a["glow"].set_alpha(.10*fade)
        for sc in a["stone"]: sc.set_alpha(sc.get_alpha() if sc.get_alpha() is not None else fade)
        # marker position is illustrative; the polygon artist's alpha is adjusted below
        for i,sc in enumerate(a["stone"]): sc.set_alpha((.17 if i==0 else 1.0)*fade)
    else:
        for sc in a["stone"]: sc.set_alpha(0)
        a["rock"].set_alpha(0); a["glow"].set_alpha(0)

    # Orbital-scale trajectories emerge only after the smooth camera pull-back.
    orb_alpha = _smoothstep((t-58)/1.3) if stage>=5 else 0
    a["orbital_legend"].set_visible(stage==5 and t>=58)
    for j,(model,col,label) in enumerate(a["orbit_specs"]):
        line=a["orbit_lines"][j]
        if stage==5 and t < 75:
            starts=[59.5,64.5,69.5]
            q=(t-starts[j])/4.5
            n=max(2,int(np.clip(q,0,1)*(len(model["t"])-1)))
            line.set_data(model["x"][:n],model["y"][:n])
            line.set_alpha(.95*orb_alpha if q>0 else 0)
        elif stage==6:
            line.set_alpha(.19)
        else:
            line.set_alpha(0)
    if stage==5:
        # Marker progresses through whichever of the three ECI solutions is being introduced.
        idx=0 if t<64.5 else (1 if t<69.5 else 2)
        st=[59.5,64.5,69.5][idx]
        q=np.clip((t-st)/4.5,0,1)
        model=a["orbit_specs"][idx][0]
        ii=int(q*(len(model["t"])-1))
        pos=(model["x"][ii],model["y"][ii])
        _set_artist_pos(a["rock"],pos); a["rock"].set_alpha(.95)
        a["glow"].set_alpha(.12); _set_artist_pos(a["glow"],pos)
    else:
        a["rock"].set_alpha(0); a["glow"].set_alpha(0)
    # Satellite stage: reference orbit and a true integrated 15-day radial loss,
    # displayed as a clearly disclosed 10x two-turn spiral for legibility.
    sat_on = stage==6
    a["sat_ref"].set_alpha(.7 if sat_on else 0)
    a["sat_spiral"].set_alpha(.95 if sat_on else 0)
    a["sat_chart"].set_visible(sat_on)
    a["sat_marker"].set_alpha(.98 if sat_on else 0)
    if sat_on:
        q=np.clip((t-75)/15,0,1)
        angle=2*np.pi*q
        r=a["r0"] - a["radial_shift"]*q
        _set_artist_pos(a["sat_marker"],(-r*np.sin(angle),r*np.cos(angle)))

    # Stage-specific copy and progression cue.
    _set_side(a,stage)
    if stage == 1 and t < 3.3:
        a["side_body"].set_text("F = ma")
    elif stage == 1:
        a["side_body"].set_text("y(t) = y₀ + v₀ sen(θ)t\n− ½gt²")
    titles={1:"A pedra obedece às leis de Newton.",
            2:"E se levarmos em conta a resistência do ar?",
            3:"E se o próprio ar estiver em movimento?",
            4:"E se a atmosfera mudar com a altitude?",
            5:"E se a Terra não fosse plana na escala do problema?",
            6:"A mesma física. Novas escalas. Novas perguntas."}
    if t >= 87:
        titles[6]="A mesma física. Novas escalas. Novas perguntas."
    a["headline"].set_text(titles[stage])
    for i,item in enumerate(a["footer_items"]):
        item.set_color([BLUE,ORANGE,GREEN,ORANGE,NAVY,BLUE][i] if i==stage-1 else "#A6B2BC")
    if stage==5 and 54<=t<58:
        a["side_question"].set_text("A escala muda.\nA pergunta também.")
    if stage==6 and t>=84:
        a["side_title"].set_text("Novas escalas.\nNovas perguntas.")
        a["side_question"].set_text("Quanto vale prever\nmelhor a trajetória de\num satélite?")
    return []


def render_master(path=None):
    path = Path(path) if path else OUT/"keynote_master.mp4"
    a=_make_scene()
    frames=int(RENDER.stage_boundaries[-1]*RENDER.fps)
    if not _ffmpeg():
        frame_dir=OUT/"keynote_frames"
        frame_dir.mkdir(exist_ok=True)
        print(f"ffmpeg ausente; exportando {frames} PNGs em {frame_dir}",flush=True)
        try:
            for frame in range(frames):
                update_frame(frame,a)
                a["fig"].savefig(frame_dir/f"frame_{frame:05d}.png",dpi=RENDER.dpi,
                                 facecolor=WHITE)
                if frame and frame%(RENDER.fps*10)==0:
                    print(f"  {frame/RENDER.fps:5.0f}/{RENDER.stage_boundaries[-1]:.0f} s",flush=True)
        finally:
            plt.close(a["fig"])
        return frame_dir
    writer=animation.FFMpegWriter(fps=RENDER.fps,codec="libx264",bitrate=4600,
        extra_args=["-pix_fmt","yuv420p","-preset","medium","-movflags","+faststart"])
    print(f"Rendering {frames} frames ({RENDER.stage_boundaries[-1]:.0f} s) → {path}",flush=True)
    try:
        with writer.saving(a["fig"],str(path),dpi=RENDER.dpi):
            for frame in range(frames):
                update_frame(frame,a)
                writer.grab_frame()
                if frame and frame%(RENDER.fps*10)==0:
                    print(f"  {frame/RENDER.fps:5.0f}/{RENDER.stage_boundaries[-1]:.0f} s",flush=True)
    finally:
        plt.close(a["fig"])
    return path


def _ffmpeg():
    return shutil.which("ffmpeg")


def render_segments(master=None):
    master=Path(master) if master else OUT/"keynote_master.mp4"
    ffmpeg=_ffmpeg()
    if not ffmpeg:
        print("ffmpeg ausente; não é possível codificar os segmentos MP4. Use a sequência PNG.")
        return
    names=["projectile","drag","wind","atmosphere","orbit","satellite"]
    bounds=RENDER.stage_boundaries
    for i,name in enumerate(names):
        start,duration=bounds[i],bounds[i+1]-bounds[i]
        dest=OUT/f"stage_{i+1:02d}_{name}.mp4"
        cmd=[ffmpeg,"-y","-ss",f"{start:.3f}","-i",str(master),"-t",f"{duration:.3f}",
             "-an","-c:v","libx264","-preset","medium","-crf","19",
             "-pix_fmt","yuv420p","-movflags","+faststart",str(dest)]
        subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        print(f"Wrote {dest.name} ({duration:.0f} s)")
    short=OUT/"keynote_30s.mp4"
    cmd=[ffmpeg,"-y","-i",str(master),"-vf","setpts=PTS/3.0","-an","-r",str(RENDER.fps),
         "-frames:v",str(30*RENDER.fps),
         "-c:v","libx264","-preset","medium","-crf","19","-pix_fmt","yuv420p",
         "-movflags","+faststart",str(short)]
    subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    print(f"Wrote {short.name} (30 s accelerated cut)")


def render_figures():
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.titleweight":"bold"})
    # Static comparison across models 1-3.
    ideal=ideal_projectile(samples=800)
    drag=simulate_projectile(drag=True,density="constant",wind="calm",samples=800)
    const=simulate_projectile(drag=True,density="constant",wind="constant",samples=800)
    shear=simulate_projectile(drag=True,density="constant",wind="shear",samples=800)
    fig,ax=plt.subplots(figsize=(16,9),dpi=120,facecolor=WHITE)
    fig.subplots_adjust(left=.09,right=.95,top=.80,bottom=.14)
    fig.text(.09,.94,"XI SBGEA 2026  •  DA PEDRA AO ESPAÇO",color=MUTED,fontsize=12,fontweight="bold")
    fig.text(.09,.875,"Cada pergunta acrescenta uma física relevante",color=NAVY,fontsize=27,fontweight="bold")
    for model,col,label,ls,lw in [
        (ideal,BLUE,f"Ideal • {ideal['range']:.1f} m",(0,(5,4)),3.0),
        (drag,ORANGE,f"Arrasto • {drag['range']:.1f} m", "-",3.0),
        (const,NAVY,f"Vento constante • {const['range']:.1f} m","-",2.7),
        (shear,GREEN,f"Vento com cisalhamento • {shear['range']:.1f} m","-",3.0)]:
        ax.plot(model["x"],model["y"],color=col,lw=lw,ls=ls,label=label)
    ax.axhline(0,color="#9EB0BD",lw=1.3)
    ax.set_xlim(-1,72); ax.set_ylim(0,21)
    ax.set_xlabel("distância horizontal (m)",fontsize=16,color=INK,labelpad=10)
    ax.set_ylabel("altura (m)",fontsize=16,color=INK,labelpad=10)
    ax.tick_params(labelsize=13,colors=MUTED)
    ax.grid(color="#E7EDF2",lw=.8)
    ax.legend(loc="upper right",frameon=True,facecolor="white",edgecolor="#D9E3EA",fontsize=13)
    ax.text(.02,.96,"v₀ = 25 m/s  •  θ = 45°  •  y₀ = 1,7 m",transform=ax.transAxes,
            ha="left",va="top",color=MUTED,fontsize=13)
    out=OUT/"projectile_comparison.png"
    fig.savefig(out,dpi=180,facecolor=WHITE)
    plt.close(fig)

    # Cross-scale infographic: local throw and ECI paths are separate honest scales.
    examples=orbital_examples()
    fig=plt.figure(figsize=(16,9),dpi=120,facecolor=WHITE)
    fig.text(.07,.94,"XI SBGEA 2026  •  A CONSTRUÇÃO DE UM MODELO",color=MUTED,fontsize=12,fontweight="bold")
    fig.text(.07,.875,"Do movimento de uma pedra à dinâmica orbital",color=NAVY,fontsize=27,fontweight="bold")
    ax1=fig.add_axes([.07,.19,.35,.59]); ax2=fig.add_axes([.50,.17,.43,.63]);
    ax1.set_facecolor(WHITE)
    ax1.plot(ideal["x"],ideal["y"],color=BLUE,lw=2.8,ls="--",label="vácuo / gravidade constante")
    ax1.plot(drag["x"],drag["y"],color=ORANGE,lw=2.8,label="arrasto atmosférico")
    ax1.axhline(0,color="#9EB0BD",lw=1.2)
    ax1.set_xlim(-2,70); ax1.set_ylim(0,21); ax1.grid(color="#E7EDF2",lw=.7)
    ax1.set_xlabel("distância (m)",fontsize=12); ax1.set_ylabel("altura (m)",fontsize=12)
    ax1.set_title("01  •  Escala local",loc="left",color=NAVY,fontsize=17,pad=14)
    ax1.legend(frameon=False,fontsize=10.5,loc="upper right")
    ax2.set_aspect("equal",adjustable="box"); ax2.set_facecolor(WHITE)
    R=PHYSICS.earth_radius
    ax2.add_patch(patches.Circle((0,0),R/1e6,facecolor=EARTH,edgecolor=BLUE,lw=1.4,zorder=0))
    for model,col,label in [(examples["suborbital"],ORANGE,"suborbital"),
                            (examples["circular"],BLUE,"circular"),
                            (examples["escape"],GREEN,"escape")]:
        ax2.plot(model["x"]/1e6,model["y"]/1e6,color=col,lw=2.7,label=label,zorder=3)
    ax2.set_xlim(-36,36); ax2.set_ylim(-18.5,18.5)
    ax2.set_title("02  •  Escala planetária — ECI, μ terrestre",loc="left",color=NAVY,fontsize=17,pad=14)
    ax2.axis("off")
    fig.text(.425,.48,"→",ha="center",va="center",fontsize=38,color=ORANGE,fontweight="bold")
    fig.text(.50,.09,"O ponto lançado é um corpo de teste idealizado a 300 km de altitude;\numa pedra lançada por uma pessoa não alcança a órbita.",
             color=MUTED,fontsize=11.5,ha="left",va="top")
    ax2.legend(loc="lower right",frameon=True,facecolor="white",edgecolor="#D9E3EA",fontsize=11)
    out2=OUT/"rock_to_orbit.png"
    fig.savefig(out2,dpi=180,facecolor=WHITE)
    plt.close(fig)
    print(f"Wrote {out.name}, {out2.name}")


def render_keyframes():
    """Save one high-resolution 16:9 still from each model stage."""
    folder=OUT/"keyframes"
    folder.mkdir(exist_ok=True)
    times=(10.0,24.0,37.0,49.0,73.0,88.0)
    for stage,t in enumerate(times,1):
        a=_make_scene()
        update_frame(int(t*RENDER.fps),a)
        path=folder/f"stage_{stage:02d}.png"
        a["fig"].savefig(path,dpi=180,facecolor=WHITE)
        plt.close(a["fig"])
    print(f"Wrote six high-resolution stage stills: {folder}")


def render_transparent_frames():
    """Export a transparent PNG sequence of the illustrated rock and glow trail."""
    folder=OUT/"transparent_rock_frames"
    folder.mkdir(exist_ok=True)
    # 3 seconds, 30 fps, isolated from the person/ground/background.
    W,H=1280,720
    for i in range(RENDER.transparent_frames):
        q=i/max(RENDER.transparent_frames-1,1)
        x=135+q*1000
        y=H-115-(4*q*(1-q))*260
        base=Image.new("RGBA",(W,H),(255,255,255,0))
        glow=Image.new("RGBA",(W,H),(0,0,0,0))
        gd=ImageDraw.Draw(glow)
        # Soft colored motion trace trails back along the path.
        for j in range(22):
            f=max(0,q-j*.006)
            tx=135+f*1000
            ty=H-115-(4*f*(1-f))*260
            radius=max(2,18-j*.65)
            alpha=max(0,int(70*(1-j/22)))
            gd.ellipse((tx-radius,ty-radius,tx+radius,ty+radius),fill=(38,134,208,alpha))
        glow=glow.filter(ImageFilter.GaussianBlur(15))
        base.alpha_composite(glow)
        d=ImageDraw.Draw(base)
        # Irregular polygon, with small facet, scaled for visibility in a slide.
        pts=[(x-24,y+9),(x-21,y-10),(x-9,y-22),(x+11,y-21),
             (x+24,y-7),(x+20,y+13),(x+7,y+23),(x-12,y+20)]
        d.polygon(pts,fill=(154,120,95,255),outline=(8,47,115,255),width=3)
        d.polygon([(x-12,y-12),(x-4,y-18),(x+7,y-17),(x+1,y-7)],fill=(214,185,157,220))
        base.save(folder/f"rock_{i:04d}.png",optimize=True)
    print(f"Wrote transparent sequence: {folder} ({RENDER.transparent_frames} PNGs)")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--all",action="store_true",help="render master, cuts, figures, and transparent sequence")
    p.add_argument("--master",action="store_true",help="render the 90 s master animation")
    p.add_argument("--segments",action="store_true",help="cut six stage clips and 30 s version from master")
    p.add_argument("--figures",action="store_true",help="render static comparison figures")
    p.add_argument("--transparent",action="store_true",help="render transparent rock PNG sequence")
    args=p.parse_args()
    if not any(vars(args).values()): args.all=True
    if args.all or args.master: render_master()
    if args.all or args.segments: render_segments()
    if args.all or args.figures:
        render_figures()
        render_keyframes()
    if args.all or args.transparent: render_transparent_frames()
    if (args.all or args.master) and not _ffmpeg():
        print("ffmpeg não encontrado: MP4 indisponível; salve como GIF/PNG com PillowWriter. Veja README.",file=sys.stderr)

if __name__=="__main__": main()
