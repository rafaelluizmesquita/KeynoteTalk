"""Render a continuous 30 s mission story and slide-sized scene clips.

All motion comes from the existing projectile, rocket, sphere, no-wind,
observation, and retrieval simulations in :mod:`render_mission`. Vector icons
are screen-space illustrations anchored to world-coordinate states.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path as MarkerPath
from matplotlib.lines import Line2D
from matplotlib.legend import Legend

from ..config import (MISSION, PHYSICS, TRACKING, WIND_TRUTH, MISSION_RENDER as RC)
from ..mission.dynamics import neutral_wind
from ..paths import output_dir
from .render_mission import _prepare_render_data, _rock_marker_path


OUT = output_dir("animation")
VIDEO_DIR = OUT / "slide_videos"
KEYFRAME_DIR = OUT / "slide_keyframes"
NAVY = "#082F73"
BLUE = "#2686D0"
GREEN = "#70AE54"
ORANGE = "#EC8135"
INK = "#18344F"
MUTED = "#64798B"
LIGHT = "#E7EFF4"
PALE_BLUE = "#EAF4FA"
WHITE = "#FFFFFF"

STAGES = (
    ("pedra_e_transicao", 4.0, "Tudo começa com curiosidade... Aqui, começamos com uma pedra."),
    ("resistencia_do_ar", 3.0, "A mesma pedra. Agora, o ar também exerce força."),
    ("vento_neutro", 3.0, "E se o próprio ar estiver em movimento?"),
    ("coluna_atmosferica", 4.0, "A 80 km, a atmosfera também faz parte do modelo."),
    ("lancamento_do_foguete", 5.0, "Um foguete leva a esfera à região de interesse."),
    ("separacao_da_esfera", 3.0, "Na liberação, a esfera herda o estado do foguete."),
    ("rastreamento_por_radar", 4.5, "O radar mede posições; o vento é uma inferência."),
    ("recuperacao_do_vento", 3.5, "Das observações ao perfil do vento, com incerteza."),
)
STAGE_SECONDS = np.asarray([s[1] for s in STAGES], dtype=float)
STAGE_BOUNDS = np.r_[0.0, np.cumsum(STAGE_SECONDS)]
TOTAL_SECONDS = float(STAGE_BOUNDS[-1])


def smooth(q):
    q = float(np.clip(q, 0.0, 1.0))
    return q * q * (3.0 - 2.0 * q)


def camera_frame(x0, x1, z0, z1, margin_x=0.06, margin_z=0.05):
    """Return safe world limits with independent horizontal/vertical scales."""
    dx = max(float(x1 - x0), 1e-6)
    dz = max(float(z1 - z0), 1e-6)
    return (float(x0 - dx * margin_x), float(x1 + dx * margin_x),
            float(z0 - dz * margin_z), float(z1 + dz * margin_z))


def lerp_frame(a, b, q):
    q = smooth(q)
    return tuple((1.0 - q) * x + q * y for x, y in zip(a, b))


def axes_point(x, z, limits):
    """Map world metres to the independent screen-normalized axes space."""
    x0, x1, z0, z1 = limits
    return (float((x - x0) / (x1 - x0)), float((z - z0) / (z1 - z0)))


def _gradient_rgba():
    """A restrained continuous vertical atmosphere tint, transparent to white."""
    nodes = np.array([0.00, 0.52, 0.60, 0.76, 0.84, 0.90, 1.00])
    colors = np.array([
        [1.00, 1.00, 1.00],
        [1.00, 1.00, 1.00],
        [0.91, 0.96, 0.985],
        [0.88, 0.94, 0.975],
        [0.94, 0.97, 0.975],
        [0.97, 0.95, 0.90],
        [0.98, 0.96, 0.92],
    ])
    z = np.linspace(0.0, 1.0, 512)
    rgb = np.column_stack([np.interp(z, nodes, colors[:, i]) for i in range(3)])
    rgba = np.ones((512, 2, 4), dtype=float)
    rgba[:, :, :3] = rgb[:, None, :]
    rgba[:, :, 3] = 1.0
    return rgba


class VectorRocket:
    """Readable vector sounding rocket, fixed-size on screen, world-anchored."""
    def __init__(self, ax):
        self.ax = ax
        tr = ax.transAxes
        self.w = 0.040
        self.h = 0.135
        self.body = patches.Polygon(np.zeros((6, 2)), closed=True, transform=tr,
            facecolor="#F8FBFD", edgecolor=NAVY, linewidth=1.8, zorder=20,
            joinstyle="round", clip_on=True)
        self.nose = patches.Polygon(np.zeros((3, 2)), closed=True, transform=tr,
            facecolor=ORANGE, edgecolor=NAVY, linewidth=1.5, zorder=21,
            joinstyle="round", clip_on=True)
        self.fin_left = patches.Polygon(np.zeros((3, 2)), closed=True, transform=tr,
            facecolor=BLUE, edgecolor=NAVY, linewidth=1.25, zorder=19, clip_on=True)
        self.fin_right = patches.Polygon(np.zeros((3, 2)), closed=True, transform=tr,
            facecolor=BLUE, edgecolor=NAVY, linewidth=1.25, zorder=19, clip_on=True)
        self.nozzle = patches.Rectangle((0, 0), 0, 0, transform=tr,
            facecolor=NAVY, edgecolor="none", zorder=18, clip_on=True)
        self.payload_band = patches.FancyBboxPatch((0, 0), 0, 0,
            boxstyle="round,pad=0.001,rounding_size=0.004", transform=tr,
            facecolor=ORANGE, edgecolor=NAVY, linewidth=0.9, zorder=22,
            clip_on=True)
        self.window = patches.Circle((0, 0), 0, transform=tr,
            facecolor=WHITE, edgecolor=NAVY, linewidth=0.8, zorder=23,
            clip_on=True)
        self.flame_outer = patches.Polygon(np.zeros((3, 2)), closed=True, transform=tr,
            facecolor=ORANGE, edgecolor="none", alpha=0.0, zorder=17, clip_on=True)
        self.flame_inner = patches.Polygon(np.zeros((3, 2)), closed=True, transform=tr,
            facecolor="#F5B56E", edgecolor="none", alpha=0.0, zorder=18, clip_on=True)
        self.artists = [self.flame_outer, self.flame_inner, self.fin_left,
            self.fin_right, self.nozzle, self.body, self.nose, self.payload_band,
            self.window]
        for p in self.artists:
            ax.add_patch(p)
        self.visible(False)

    def visible(self, value):
        for p in self.artists:
            p.set_visible(value)

    def place(self, x_m, z_m, limits, flame=0.0, alpha=1.0):
        cx, cy = axes_point(x_m, z_m, limits)
        w, h = self.w, self.h
        # The rocket's tail is anchored at its simulated position; its vector
        # length is illustrative and disclosed in the scene caption.
        y0 = cy
        self.body.set_xy(np.array([
            [cx-w*.25, y0+h*.12], [cx-w*.25, y0+h*.78],
            [cx-w*.15, y0+h*.86], [cx+w*.15, y0+h*.86],
            [cx+w*.25, y0+h*.78], [cx+w*.25, y0+h*.12]]))
        self.nose.set_xy(np.array([[cx-w*.25, y0+h*.78],
                                   [cx, y0+h], [cx+w*.25, y0+h*.78]]))
        self.fin_left.set_xy(np.array([[cx-w*.25, y0+h*.28],
            [cx-w*.58, y0+h*.04], [cx-w*.20, y0+h*.12]]))
        self.fin_right.set_xy(np.array([[cx+w*.25, y0+h*.28],
            [cx+w*.58, y0+h*.04], [cx+w*.20, y0+h*.12]]))
        self.nozzle.set_bounds(cx-w*.17, y0+h*.02, w*.34, h*.13)
        self.payload_band.set_bounds(cx-w*.22, y0+h*.57, w*.44, h*.09)
        self.window.center = (cx, y0+h*.43)
        self.window.radius = w*.075
        f = float(np.clip(flame, 0.0, 1.0))
        self.flame_outer.set_xy(np.array([[cx-w*.16, y0+h*.03],
            [cx, y0-h*(.14+.35*f)], [cx+w*.16, y0+h*.03]]))
        self.flame_inner.set_xy(np.array([[cx-w*.08, y0+h*.02],
            [cx, y0-h*(.08+.23*f)], [cx+w*.08, y0+h*.02]]))
        self.flame_outer.set_alpha(f * alpha)
        self.flame_inner.set_alpha(f * alpha * .8)
        for p in self.artists:
            if p not in (self.flame_outer, self.flame_inner):
                p.set_alpha(alpha)
        # Keep every rocket vertex inside the safe axes rectangle.
        verts = np.vstack([self.body.get_xy(), self.nose.get_xy(),
                           self.fin_left.get_xy(), self.fin_right.get_xy()])
        assert np.all(verts[:, 0] >= -0.01) and np.all(verts[:, 0] <= 1.01)
        assert np.all(verts[:, 1] >= -0.04) and np.all(verts[:, 1] <= 1.01)


class RadarIcon:
    """Small screen-space vector radar with pulsing arcs, no long beam fan."""
    def __init__(self, ax):
        self.ax = ax
        tr = ax.transAxes
        self.base = patches.Polygon(np.zeros((3, 2)), transform=tr,
            closed=True, facecolor=NAVY, edgecolor=NAVY, zorder=20, clip_on=True)
        self.dish = patches.Arc((0, 0), 0, 0, theta1=25, theta2=155,
            transform=tr, color=NAVY, linewidth=2.0, zorder=21)
        self.stem = Line2D([], [], transform=tr, color=NAVY, linewidth=1.8, zorder=21)
        self.arcs = [patches.Arc((0, 0), 0, 0, theta1=10, theta2=95,
            transform=tr, color=BLUE, linewidth=1.2, alpha=.0, zorder=22)
            for _ in range(3)]
        for p in [self.base, self.dish, self.stem, *self.arcs]:
            ax.add_patch(p) if isinstance(p, patches.Patch) else ax.add_line(p)
        self.visible(False)

    def visible(self, value):
        for p in [self.base, self.dish, self.stem, *self.arcs]:
            p.set_visible(value)

    def place(self, station_x, station_z, limits, pulse):
        cx, cy = axes_point(station_x, station_z, limits)
        w, h = .040, .042
        self.base.set_xy(np.array([[cx-w*.40, cy], [cx+w*.40, cy], [cx, cy+h*.20]]))
        self.dish.center = (cx, cy+h*.34)
        self.dish.width, self.dish.height = w*.52, h*.48
        self.stem.set_data([cx, cx], [cy+h*.20, cy+h*.37])
        for i, arc in enumerate(self.arcs, start=1):
            r = .025 + i*.018
            arc.center = (cx+w*.16, cy+h*.40)
            arc.width, arc.height = r, r*1.5
            arc.set_alpha(.2 + .45*float(pulse))


class VisualExperiment:
    def __init__(self, result):
        self.result = result
        self.ideal = result["local_ideal"]
        self.local_drag = result["local_drag"]
        self.local_uniform = result["local_uniform"]
        self.local_shear = result["local_shear"]
        self.mission = result["mission"]
        self.rocket_data = self.mission["rocket"]
        self.sphere_data = self.mission["sphere"]
        self.calm_data = result["no_wind_sphere"]
        self.obs = result["observations"]
        self.fit = result["retrieval"]

        self.fig = plt.figure(figsize=(RC.width/RC.dpi, RC.height/RC.dpi),
                              dpi=RC.dpi, facecolor=WHITE)
        self.ax = self.fig.add_axes([.035, .10, .93, .77], facecolor=WHITE, zorder=1)
        self.ax.set_axis_off()
        self.ax.set_xlim(-8, 72); self.ax.set_ylim(-3, 22)
        self.ax.set_autoscale_on(False)
        self.ax.set_clip_on(True)

        # The atmospheric image is in world coordinates; annotations and vector
        # vehicles use axes coordinates after mapping the simulated world state.
        self.atmos = self.ax.imshow(_gradient_rgba(), extent=(-8000, 3000, 0, 112000),
            origin="lower", aspect="auto", interpolation="bilinear", zorder=-20,
            alpha=0.0, clip_on=True)
        self.layer_lines = []
        for z in (60_000, 85_000, 100_000, 110_000):
            ln, = self.ax.plot([-8000, 3000], [z, z], color=WHITE, lw=1.0,
                alpha=0.0, zorder=-10, clip_on=True)
            self.layer_lines.append(ln)
        self.alt_labels = []
        for z in (60_000, 80_000, 100_000, 110_000):
            tx = self.ax.text(-7100, z, f"{z/1000:.0f} km", color=MUTED,
                fontsize=10.5, va="center", ha="left", alpha=0.0, zorder=3,
                clip_on=True)
            self.alt_labels.append(tx)
        self.layer_labels = [
            self.ax.text(-3300, 72_500, "MESOSFERA", color=NAVY, fontsize=13,
                fontweight="bold", alpha=0.0, ha="center", zorder=3, clip_on=True),
            self.ax.text(-3300, 85_000, "MESOPAUSA", color=MUTED, fontsize=11,
                fontweight="bold", alpha=0.0, ha="center", zorder=3, clip_on=True),
            self.ax.text(-3300, 105_000, "TERMOSFERA INFERIOR", color=MUTED, fontsize=10,
                fontweight="bold", alpha=0.0, ha="center", zorder=3, clip_on=True),
        ]
        self.wind_arrows = []
        self.wind_labels = []
        for z in WIND_TRUTH.altitude_m:
            u = float(neutral_wind(z, WIND_TRUTH))
            x0 = -4200.0
            arrow = self.ax.annotate("", xy=(x0 + u*22.0, z), xytext=(x0, z),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.7, alpha=.0,
                                mutation_scale=10), zorder=4, clip_on=True)
            self.wind_arrows.append(arrow)
            self.wind_labels.append((x0, z, u))

        # A fixed physical height marker makes the camera's continuous change
        # of scale legible: it is a real 80 km interval in world coordinates,
        # not an animated screen-space ruler.
        scale_x = 1850.0
        self.scale_riser, = self.ax.plot([scale_x, scale_x], [0.0, 80_000.0],
            color="#91A4B2", lw=1.0, ls=(0, (4, 4)), alpha=0.0,
            zorder=4, clip_on=True)
        self.scale_ticks = [
            self.ax.plot([scale_x-180, scale_x+180], [z, z], color="#91A4B2",
                lw=1.0, alpha=0.0, zorder=4, clip_on=True)[0]
            for z in (0.0, 80_000.0)]
        self.scale_text = self.ax.text(scale_x-120, 80_000.0,
            "80 km", color=NAVY, fontsize=10.5, fontweight="bold",
            ha="right", va="bottom", alpha=0.0, zorder=5, clip_on=True)

        self.ground, = self.ax.plot([], [], color=NAVY, lw=2.0, alpha=.0,
                                     zorder=2, clip_on=True)
        self.ideal_line, = self.ax.plot([], [], color=BLUE, lw=2.5,
            ls=(0, (5, 4)), alpha=.90, zorder=5, clip_on=True)
        self.drag_line, = self.ax.plot([], [], color=ORANGE, lw=2.6,
            alpha=0.0, zorder=6, clip_on=True)
        self.calm_local_line, = self.ax.plot([], [], color="#9AA8B2", lw=2.0,
            ls=(0, (4, 4)), alpha=0.0, zorder=5, clip_on=True)
        self.uniform_local_line, = self.ax.plot([], [], color=ORANGE, lw=2.2,
            ls=(0, (2, 2)), alpha=0.0, zorder=6, clip_on=True)
        self.shear_local_line, = self.ax.plot([], [], color=GREEN, lw=2.6,
            alpha=0.0, zorder=7, clip_on=True)
        self.rock_trace, = self.ax.plot([], [], color=BLUE, lw=2.7, alpha=.0,
                                        zorder=6, clip_on=True)
        self.drag_legend = Legend(self.ax,
            [Line2D([], [], color=BLUE, lw=2, ls=(0, (5, 4)), label="ideal"),
             Line2D([], [], color=ORANGE, lw=2.5, label="com arrasto")],
            ["ideal", "com arrasto"], loc="upper right", bbox_to_anchor=(.98, .80),
            frameon=True, framealpha=.94, facecolor=WHITE, edgecolor="#D9E3EA",
            fontsize=10.5)
        self.ax.add_artist(self.drag_legend)
        self.wind_legend = Legend(self.ax,
            [Line2D([], [], color=BLUE, lw=1.8, ls=(0, (5, 4)), label="ideal"),
             Line2D([], [], color="#9AA8B2", lw=2, ls=(0, (4, 4)), label="sem vento"),
             Line2D([], [], color=ORANGE, lw=2, ls=(0, (2, 2)), label="vento uniforme"),
             Line2D([], [], color=GREEN, lw=2.5, label="vento u(z)")],
            ["ideal", "sem vento", "vento uniforme", "vento u(z)"],
            loc="upper right", bbox_to_anchor=(.98, .80), frameon=True,
            framealpha=.94, facecolor=WHITE, edgecolor="#D9E3EA", fontsize=9.5)
        self.ax.add_artist(self.wind_legend)
        self.drag_legend.set_visible(False)
        self.wind_legend.set_visible(False)

        # Peak and impact markers are computed from the same trajectories that
        # are animated. Their compact readout sits in the clear band below the
        # equations, above the local trajectory viewport.
        self.metric_models = {
            "ideal": (self.ideal, BLUE, "Ideal"),
            "drag": (self.local_drag, ORANGE, "Com arrasto"),
            "calm": (self.local_drag, "#9AA8B2", "Sem vento"),
            "uniform": (self.local_uniform, ORANGE, "Vento uniforme"),
            "shear": (self.local_shear, GREEN, "Vento u(z)"),
        }
        self.metric_stats = {}
        for name, (data, color, label) in self.metric_models.items():
            peak_i = int(np.argmax(data["y"]))
            self.metric_stats[name] = {
                "peak_x": float(data["x"][peak_i]),
                "peak_y": float(data["y"][peak_i]),
                "range": float(data["x"][-1]-data["x"][0]),
            }
        self.metric_card = patches.FancyBboxPatch((.62, .91), .35, .08,
            boxstyle="round,pad=.012", transform=self.ax.transAxes,
            facecolor=WHITE, edgecolor="#D9E3EA", linewidth=1.0,
            alpha=.96, zorder=24, clip_on=True)
        self.ax.add_patch(self.metric_card)
        self.metric_launch = self.ax.text(.64, .965, "", transform=self.ax.transAxes,
            color=NAVY, fontsize=8.9, fontweight="bold", ha="left", va="center",
            zorder=26, clip_on=True)
        self.metric_columns = self.ax.text(.81, .965, "h máx. / alcance R (m)",
            transform=self.ax.transAxes, color=MUTED, fontsize=8.8,
            ha="left", va="center", zorder=26, clip_on=True)
        metric_pos = {
            "ideal": (.64, .925), "drag": (.81, .925),
            "calm": (.81, .925), "uniform": (.64, .885),
            "shear": (.81, .885),
        }
        self.metric_rows = {}
        self.apex_markers = {}
        self.range_ticks = {}
        for name, (_data, color, label) in self.metric_models.items():
            self.metric_rows[name] = self.ax.text(*metric_pos[name], "",
                transform=self.ax.transAxes, color=color, fontsize=8.9,
                ha="left", va="center", zorder=26, clip_on=True)
            self.apex_markers[name] = self.ax.scatter([], [], s=58,
                facecolor=color, edgecolor=WHITE, linewidths=1.1,
                alpha=0.0, zorder=18, clip_on=True)
            tick, = self.ax.plot([], [], color=color, lw=2.0,
                alpha=0.0, zorder=18, clip_on=True)
            self.range_ticks[name] = tick
        self._set_local_metrics(())
        person_x = -1.25
        self.person_head = patches.Circle((person_x, 1.55), .15,
            facecolor=NAVY, edgecolor="none", zorder=12, clip_on=True)
        self.ax.add_patch(self.person_head)
        joints = [((person_x, 1.4), (person_x, .70)),
                  ((person_x, 1.23), (0.0, 1.70)),
                  ((person_x, 1.20), (person_x-.48, .90)),
                  ((person_x, .70), (person_x-.38, .05)),
                  ((person_x, .70), (person_x+.43, .05))]
        self.person_limbs = []
        for p0, p1 in joints:
            line, = self.ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=NAVY,
                lw=3.0, solid_capstyle="round", zorder=11, clip_on=True)
            self.person_limbs.append(line)
        self.rock = self.ax.scatter([0], [1.7], marker=_rock_marker_path(), s=190,
            facecolor="#8C735E", edgecolor="#4D453F", linewidth=1.0,
            alpha=1.0, zorder=14, clip_on=True)
        self.rock_glow = self.ax.scatter([0], [1.7], s=420, c=ORANGE,
            alpha=0.12, linewidths=0, zorder=13, clip_on=True)

        self.rocket_path, = self.ax.plot([], [], color=NAVY, lw=2.1,
            alpha=0.58, zorder=6, clip_on=True)
        self.rocket = VectorRocket(self.ax)
        self.release_halo = self.ax.scatter([], [], s=400, facecolors="none",
            edgecolors=ORANGE, linewidths=2.0, alpha=0.0, zorder=15, clip_on=True)
        self.sphere_path, = self.ax.plot([], [], color=BLUE, lw=2.8,
            alpha=0.95, zorder=7, clip_on=True)
        self.calm_path, = self.ax.plot([], [], color="#9AA8B2", lw=1.9,
            ls=(0, (5, 5)), alpha=0.0, zorder=5, clip_on=True)
        self.sphere = self.ax.scatter([], [], s=175, facecolor=ORANGE,
            edgecolor=NAVY, linewidths=1.8, zorder=16, clip_on=True)
        self.sensor_dot = self.ax.scatter([], [], s=24, facecolor=WHITE,
            edgecolor=NAVY, linewidths=.6, zorder=17, clip_on=True)
        self.release_point = self.ax.scatter([], [], s=80, facecolor=WHITE,
            edgecolor=ORANGE, linewidths=2.0, zorder=12, clip_on=True)

        self.radar = RadarIcon(self.ax)
        self.observations = self.ax.scatter([], [], s=30, facecolor=ORANGE,
            edgecolor=WHITE, linewidths=.65, alpha=.95, zorder=13, clip_on=True)

        self.horizontal_note = self.ax.text(.975, .045,
            "Corte esquemático · escala horizontal ampliada para visualização",
            transform=self.ax.transAxes, ha="right", va="bottom", color=MUTED,
            fontsize=9.5, alpha=0.0, zorder=30)
        self.ascent_note = self.ax.text(.975, .075,
            "Foguete genérico · trajetória calculada · ícone fora de escala",
            transform=self.ax.transAxes, ha="right", va="bottom", color=MUTED,
            fontsize=9.5, alpha=0.0, zorder=30)
        self.release_note = self.ax.text(.965, .90,
            "Separação: estado herdado do foguete + Δv configurável",
            transform=self.ax.transAxes, ha="right", va="center", color=INK,
            fontsize=11.5, alpha=0.0, zorder=30,
            bbox=dict(boxstyle="round,pad=.35", fc=WHITE, ec="#D9E3EA", alpha=.96))
        self.radar_note = self.ax.text(.035, .16, "Radar sintético\nmedidas discretas de posição",
            transform=self.ax.transAxes, ha="left", va="bottom", color=INK,
            fontsize=11.5, alpha=0.0, zorder=30,
            bbox=dict(boxstyle="round,pad=.32", fc=WHITE, ec="#D9E3EA", alpha=.93))

        # A white overlay gives a deliberate cross-fade into scientific plots.
        self.veil_ax = self.fig.add_axes([0, 0, 1, 1], zorder=8)
        self.veil_ax.set_axis_off()
        self.veil_ax.patch.set_alpha(0.0)
        self.veil = patches.Rectangle((0, 0), 1, 1, transform=self.veil_ax.transAxes,
            facecolor=WHITE, edgecolor="none", alpha=0.0)
        self.veil_ax.add_patch(self.veil)

        self.traj_ax = self.fig.add_axes([.075, .22, .39, .54], facecolor=WHITE, zorder=10)
        self.wind_ax = self.fig.add_axes([.55, .22, .37, .54], facecolor=WHITE, zorder=10)
        self._make_science_plots()
        self._set_plot_alpha(0.0)
        # Keep the analytical panels completely out of the cinematic scenes.
        # Fading plot artists alone leaves axes grids and legend frames visible.
        self.traj_ax.set_visible(False)
        self.wind_ax.set_visible(False)

        self.title = self.fig.text(.055, .905, "", fontsize=23.5, color=NAVY,
            fontweight="bold", va="top", zorder=40)
        self.subtitle = self.fig.text(.055, .855, "", fontsize=15.5, color=INK,
            va="top", zorder=40)
        self.footer_rule = Line2D([.055, .945], [.082, .082], transform=self.fig.transFigure,
            color="#D9E3EA", linewidth=1.0, zorder=40)
        self.fig.add_artist(self.footer_rule)
        self.stage_label = self.fig.text(.055, .048, "", fontsize=9.5,
            color=BLUE, fontweight="bold", va="center", zorder=40)
        self.footer_date = self.fig.text(.5, .048,
            "XI SBGEA 2026 - 8 a 13 de novembro 2026", fontsize=9.5,
            color="#7D8790", ha="center", va="center", zorder=40)
        self.slide_number = self.fig.text(.945, .048, "", fontsize=11,
            color="#3E8EDE", fontweight="bold", ha="right", va="center", zorder=40)
        self.stage_progress = self.fig.add_axes([.56, .012, .38, .018], zorder=40)
        self.stage_progress.set_xlim(0, TOTAL_SECONDS)
        self.stage_progress.set_ylim(0, 1)
        self.stage_progress.axis("off")
        self.stage_progress.plot([0, TOTAL_SECONDS], [.5, .5], color=LIGHT,
            lw=4, solid_capstyle="round", clip_on=False)
        self.progress_line, = self.stage_progress.plot([0, 0], [.5, .5], color=BLUE,
            lw=4, solid_capstyle="round", clip_on=False)
        self.stage_markers = []
        for b in STAGE_BOUNDS[1:-1]:
            self.stage_markers.append(self.stage_progress.plot([b, b], [.15, .85],
                color=WHITE, lw=1.3, zorder=3)[0])

        self.local_frame = camera_frame(-7, 70, -1, 21, .06, .12)
        self.world_frame = camera_frame(-7600, 2400, -12_000, 125_000, .0, .0)
        # Include generous headroom around the release altitude and 110 km
        # label; the scene zooms here without putting typography on the edge.
        self.release_frame = camera_frame(-7600, 2400, 72_000, 111_000, .02, .08)
        self.local_to_world = self.world_frame
        self._set_camera(self.local_frame)

    def _make_science_plots(self):
        r = self.result
        trajectory = r["mission"]["sphere"]
        calm = r["no_wind_sphere"]
        obs = r["observations"]
        valid = obs["valid"][:, 0]
        polar = obs["measurement"][valid, 0, :]
        site_x = obs["station_x_m"][0]
        measured_x = site_x + polar[:, 0] * np.cos(polar[:, 1])
        measured_z = polar[:, 0] * np.sin(polar[:, 1])
        self.traj_ax.plot(calm["x"]/1000, calm["z"]/1000,
            color="#9AA8B2", lw=1.6, ls=(0, (4, 4)), label="sem vento")
        self.traj_ax.plot(trajectory["x"]/1000, trajectory["z"]/1000,
            color=BLUE, lw=2.4, label="com vento")
        self.traj_ax.scatter(measured_x/1000, measured_z/1000, s=12,
            color=ORANGE, alpha=.65, label="observações")
        self.traj_ax.set(xlim=(-5.8, .8), ylim=(58, 110),
            xlabel="distância horizontal (km)", ylabel="altitude (km)")
        self.traj_ax.set_title("Trajetória e observações", loc="left",
            color=NAVY, fontsize=14, fontweight="bold", pad=10)
        self._style_plot(self.traj_ax)
        self.traj_ax.legend(frameon=False, fontsize=9.2, loc="lower left", ncol=1)

        fit = r["retrieval"]
        zfit = fit["wind_altitude_m"] / 1000
        truth = np.asarray(WIND_TRUTH.altitude_m) / 1000
        self.wind_ax.fill_betweenx(zfit,
            fit["wind_m_s"] - fit["total_wind_std_m_s"],
            fit["wind_m_s"] + fit["total_wind_std_m_s"],
            color=ORANGE, alpha=.16, label="incerteza total")
        self.wind_ax.plot(WIND_TRUTH.speed_m_s, truth, color=GREEN, lw=2.4,
            label="perfil sintético (avaliação)")
        self.wind_ax.errorbar(fit["wind_m_s"], zfit,
            xerr=fit["measurement_wind_std_m_s"], fmt="o-", color=ORANGE,
            lw=2.0, ms=4.5, capsize=2.2, label="recuperado")
        self.wind_ax.set(xlim=(-55, 75), ylim=(58, 112),
            xlabel="vento zonal (m/s)", ylabel="altitude (km)")
        self.wind_ax.set_title("Perfil de vento recuperado", loc="left",
            color=NAVY, fontsize=14, fontweight="bold", pad=10)
        self._style_plot(self.wind_ax)
        self.wind_ax.legend(frameon=False, fontsize=8.9, loc="lower right")
        self.metric = self.fig.text(.5, .123,
            f"RMSE sintético {r['evaluation']['rmse_m_s']:.1f} m/s  ·  σ total mediana "
            f"{np.median(fit['total_wind_std_m_s']):.1f} m/s",
            ha="center", va="center", fontsize=11.2, color=MUTED, zorder=40)

    @staticmethod
    def _style_plot(ax):
        ax.grid(color=LIGHT, linewidth=.7)
        ax.tick_params(labelsize=10.0, colors=MUTED, length=3)
        ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
        for spine in ax.spines.values():
            spine.set_color("#CBD8E1")

    def _set_plot_alpha(self, alpha):
        for ax in (self.traj_ax, self.wind_ax):
            ax.patch.set_alpha(alpha)
            for child in ax.get_children():
                if hasattr(child, "set_alpha"):
                    try:
                        base = getattr(child, "_visual_base_alpha", None)
                        if base is None:
                            base = child.get_alpha()
                            child._visual_base_alpha = 1.0 if base is None else base
                        child.set_alpha(float(child._visual_base_alpha) * alpha)
                    except (AttributeError, TypeError, ValueError):
                        pass
            ax.title.set_alpha(alpha)
            ax.xaxis.label.set_alpha(alpha); ax.yaxis.label.set_alpha(alpha)
            for spine in ax.spines.values():
                spine.set_alpha(alpha)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_alpha(alpha)

    def _set_camera(self, limits):
        x0, x1, z0, z1 = limits
        if not (x1 > x0 and z1 > z0):
            raise ValueError("Camera limits must be increasing")
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(z0, z1)
        self.current_limits = limits

    def _fade_local_artists(self, alpha):
        self.person_head.set_alpha(alpha)
        for line in self.person_limbs:
            line.set_alpha(alpha)
        self.rock.set_alpha(alpha)
        self.rock_glow.set_alpha(.12*alpha)
        self.rock_trace.set_alpha(.88*alpha)
        self.ideal_line.set_alpha(.78*alpha)

    def _set_atmosphere(self, alpha):
        alpha = float(np.clip(alpha, 0.0, 1.0))
        self.atmos.set_alpha(alpha)
        for line in self.layer_lines:
            line.set_alpha(.8*alpha)
        for tx in self.alt_labels:
            tx.set_alpha(alpha)
        for tx in self.layer_labels:
            tx.set_alpha(alpha)
        for arrow in self.wind_arrows:
            arrow.arrow_patch.set_alpha(.72*alpha)
        self.ground.set_alpha(.75*alpha)

    def _rocket_interp(self, target_time):
        r = self.rocket_data
        z = float(np.interp(target_time, r["t"], r["z"]))
        return z

    def _update_scene(self, frame):
        t = float(frame / RC.fps)
        stage = min(int(np.searchsorted(STAGE_BOUNDS[1:], t, side="right")), len(STAGES)-1)
        start = STAGE_BOUNDS[stage]
        duration = STAGE_SECONDS[stage]
        q = float(np.clip((t-start)/duration, 0.0, 1.0))
        eased = smooth(q)
        self.title.set_text(STAGES[stage][2])
        equations = (
            r"$F=ma\qquad x(t)=x_0+v_0\cos\theta\,t\qquad y(t)=y_0+v_0\sin\theta\,t-\frac{1}{2}gt^2$",
            r"$\mathbf{F}_D=-\frac{1}{2}\rho C_DA|\mathbf{v}|\mathbf{v}$",
            r"$\mathbf{v}_{rel}=\mathbf{v}-\mathbf{u}(z)\qquad u(z)=u_0+Sz$",
            r"$\rho=\rho(z,T)\qquad \mathbf{u}=\mathbf{u}(z)\qquad g(z)=\frac{\mu}{(R_E+z)^2}$",
            r"$\dot z=v\qquad \dot v=\frac{T-D}{m(t)}-g(z)$",
            r"$\mathbf{x}_{s,0}=\mathbf{x}_{p}(t_0)\qquad \mathbf{v}_{s,0}=\mathbf{v}_{p}(t_0)+\Delta\mathbf{v}_{sep}$",
            r"$\mathbf{y}_i=h[\mathbf{x}(t_i)]+\boldsymbol{\epsilon}_i$",
            r"$\hat{\mathbf{u}}=\arg\min_{\mathbf{u}}\;\|\mathbf{y}-h(\mathbf{u})\|^2_{S^{-1}}+\lambda\|L\mathbf{u}\|^2$",
        )
        self.subtitle.set_text(equations[stage])
        self.stage_label.set_text(f"ETAPA {stage+1:02d} / {len(STAGES):02d}  ·  {STAGES[stage][0].replace('_',' ').upper()}")
        self.slide_number.set_text(str(stage + 7))
        self.progress_line.set_data([0.0, t], [.5, .5])

        # Reset scene visibility and camera-dependent elements for this frame.
        self.radar.visible(False)
        self.rocket.visible(False)
        self.drag_legend.set_visible(False)
        self.wind_legend.set_visible(False)
        self.observations.set_offsets(np.empty((0, 2)))
        self._set_local_metrics(())
        self._set_annotation_alpha(self.release_note, 0.0)
        self.release_note.set_visible(False)
        self._set_annotation_alpha(self.radar_note, 0.0)
        self.radar_note.set_visible(False)
        self.horizontal_note.set_alpha(0.0)
        self.ascent_note.set_alpha(0.0)
        self.sphere.set_offsets(np.empty((0, 2)))
        self.sensor_dot.set_offsets(np.empty((0, 2)))
        self.release_point.set_offsets(np.empty((0, 2)))
        self.release_halo.set_offsets(np.empty((0, 2)))
        self.rocket_path.set_data([], [])
        self.sphere_path.set_data([], [])
        self.calm_path.set_data([], [])
        for line in (self.drag_line, self.calm_local_line,
                     self.uniform_local_line, self.shear_local_line):
            line.set_alpha(0.0)
        self._set_plot_alpha(0.0)
        show_plots = stage == len(STAGES) - 1
        self.traj_ax.set_visible(show_plots)
        self.wind_ax.set_visible(show_plots)
        self.veil.set_alpha(0.0)
        self.metric.set_alpha(0.0)
        self.apogee_label.set_visible(False)
        self.scale_riser.set_alpha(0.0)
        for tick in self.scale_ticks:
            tick.set_alpha(0.0)
        self.scale_text.set_alpha(0.0)

        if stage == 0:
            self._set_camera(self.local_frame)
            self._set_atmosphere(0.0)
            self.ground.set_data([-8, 72], [0, 0])
            self.ground.set_alpha(.78)
            flight = float(self.ideal["flight_time"])
            f = min(1.0, q/.82)
            now = f*flight
            upto = max(2, int(np.searchsorted(self.ideal["t"], now, side="right")))
            x = self.ideal["x"][:upto]
            z = self.ideal["y"][:upto]
            self.ideal_line.set_data(x, z)
            visibility = 1.0-smooth((q-.86)/.14)
            self.ideal_line.set_alpha(.86*visibility)
            self.rock_trace.set_data(x, z)
            i = min(upto-1, len(x)-1)
            rock_xy = np.array([[x[i], z[i]]])
            self.rock.set_offsets(rock_xy)
            self.rock_glow.set_offsets(rock_xy)
            self.person_head.set_alpha(1.0)
            for limb in self.person_limbs:
                limb.set_alpha(1.0)
            self.rock.set_alpha(visibility)
            self.rock_glow.set_alpha(.12*visibility)
            self.rock_trace.set_color(BLUE)
            self.rock_trace.set_alpha(.88*visibility)
            self._set_local_metrics(("ideal",))
            self._set_atmosphere(0.0)
            # _set_atmosphere also fades the ground line; restore it after
            # that reset so the range tick has a visible reference surface.
            self.ground.set_alpha(.78)

        elif stage == 1:
            # The ideal path stays on screen while the drag trajectory is
            # recomputed from the same launch state and drawn in front of us.
            self._set_camera(self.local_frame)
            self._set_atmosphere(0.0)
            self.ground.set_data([-8, 72], [0, 0])
            self.ground.set_alpha(.78)
            self.ideal_line.set_data(self.ideal["x"], self.ideal["y"])
            self.ideal_line.set_alpha(.38)
            end = max(2, int(np.searchsorted(self.local_drag["t"],
                eased*self.local_drag["t"][-1], side="right")))
            end = min(end, len(self.local_drag["t"]))
            self.drag_line.set_data(self.local_drag["x"][:end], self.local_drag["y"][:end])
            self.drag_line.set_alpha(.95)
            tail = max(2, end-28)
            self.rock_trace.set_data(self.local_drag["x"][tail:end], self.local_drag["y"][tail:end])
            self.rock_trace.set_color(ORANGE)
            self.rock_trace.set_alpha(.92)
            now = eased*self.local_drag["t"][-1]
            rx = float(np.interp(now, self.local_drag["t"], self.local_drag["x"]))
            rz = float(np.interp(now, self.local_drag["t"], self.local_drag["y"]))
            self.rock.set_offsets([[rx, rz]])
            self.rock_glow.set_offsets([[rx, rz]])
            visibility = smooth(q/.10)*(1.0-smooth((q-.84)/.16))
            self.rock.set_alpha(visibility)
            self.rock_glow.set_alpha(.12*visibility)
            self.person_head.set_alpha(1.0)
            for limb in self.person_limbs:
                limb.set_alpha(1.0)
            self.drag_legend.set_visible(True)
            self._set_local_metrics(("ideal", "drag"))

        elif stage == 2:
            # Add the air velocity while retaining the earlier predictions.
            self._set_camera(self.local_frame)
            self._set_atmosphere(0.0)
            self.ground.set_data([-8, 72], [0, 0])
            self.ground.set_alpha(.78)
            self.ideal_line.set_data(self.ideal["x"], self.ideal["y"])
            self.ideal_line.set_alpha(.23)
            paths = (
                (self.calm_local_line, self.local_drag, min(1.0, q/.44), .86),
                (self.uniform_local_line, self.local_uniform, np.clip((q-.12)/.58, 0, 1), .92),
                (self.shear_local_line, self.local_shear, np.clip((q-.30)/.70, 0, 1), .98),
            )
            for line, data, fraction, alpha in paths:
                end = max(2, int(fraction*(len(data["t"])-1))+1)
                line.set_data(data["x"][:end], data["y"][:end])
                line.set_alpha(alpha)
            end = max(2, int(eased*(len(self.local_shear["t"])-1))+1)
            tail = max(2, end-28)
            self.rock_trace.set_data(self.local_shear["x"][tail:end], self.local_shear["y"][tail:end])
            self.rock_trace.set_color(GREEN)
            self.rock_trace.set_alpha(.92)
            now = eased*self.local_shear["t"][-1]
            rx = float(np.interp(now, self.local_shear["t"], self.local_shear["x"]))
            rz = float(np.interp(now, self.local_shear["t"], self.local_shear["y"]))
            self.rock.set_offsets([[rx, rz]])
            self.rock_glow.set_offsets([[rx, rz]])
            visibility = smooth(q/.10)*(1.0-smooth((q-.84)/.16))
            self.rock.set_alpha(visibility)
            self.rock_glow.set_alpha(.12*visibility)
            self.person_head.set_alpha(1.0)
            for limb in self.person_limbs:
                limb.set_alpha(1.0)
            self.wind_legend.set_visible(True)
            self._set_local_metrics(("ideal", "calm", "uniform", "shear"))

        elif stage == 3:
            # Open the horizontal frame first, then expand altitude to the
            # atmosphere; all calculated local paths shrink smoothly away.
            qx = smooth(q/.42)
            qz = smooth(q**.50)
            limits = tuple((1.0-f)*a + f*b for a, b, f in zip(
                self.local_frame, self.world_frame, (qx, qx, qz, qz)))
            self._set_camera(limits)
            self._set_atmosphere(eased)
            local_fade = 1.0-smooth(q/.60)
            self._fade_local_artists(local_fade)
            self._set_local_metrics(("ideal", "calm", "uniform", "shear"), alpha=local_fade)
            for line, alpha in ((self.drag_line,.95), (self.calm_local_line,.86),
                (self.uniform_local_line,.92), (self.shear_local_line,.98)):
                line.set_alpha(alpha*local_fade)
            self.ground.set_data([limits[0], limits[1]], [0, 0])
            self.ground.set_alpha(.75*eased)
            self.horizontal_note.set_alpha(.88*eased)
            for tx in self.alt_labels + self.layer_labels:
                tx.set_alpha(eased)
            for arrow in self.wind_arrows:
                arrow.arrow_patch.set_alpha(.68*eased)
            self._set_scale_marker(eased)

        elif stage == 4:
            limits = self.world_frame
            self._set_camera(limits)
            self._set_atmosphere(1.0)
            self._fade_local_artists(0.0)
            for line in (self.drag_line, self.calm_local_line,
                         self.uniform_local_line, self.shear_local_line):
                line.set_alpha(0.0)
            self.ground.set_data([limits[0], limits[1]], [0, 0])
            self.ground.set_alpha(.78)
            self.horizontal_note.set_alpha(.96)
            self._set_scale_marker(.9)
            rocket_time = eased*self.rocket_data["release_time_s"]
            z = self._rocket_interp(rocket_time)
            end = int(np.searchsorted(self.rocket_data["t"], rocket_time, side="right"))
            end = max(2, min(end, len(self.rocket_data["t"])))
            self.rocket_path.set_data(np.zeros(end), self.rocket_data["z"][:end])
            self.rocket_path.set_alpha(.55)
            self.rocket.visible(True)
            burn = MISSION.rocket_burn_time_s
            flame = np.clip((burn-rocket_time)/8.0, 0.0, 1.0)
            self.rocket.place(0.0, z, limits, flame=flame)
            self.apogee_label.set_text(f"apogeu do modelo  ·  {self.rocket_data['apogee_m']/1000:.0f} km")
            self.apogee_label.set_alpha(smooth((q-.72)/.18))

        elif stage == 5:
            zoom = smooth(q/.22)
            limits = lerp_frame(self.world_frame, self.release_frame, zoom)
            self._set_camera(limits)
            self._set_atmosphere(1.0)
            self._fade_local_artists(0.0)
            self.ground.set_data([limits[0], limits[1]], [0, 0])
            self.ground.set_alpha(.55*(1-zoom))
            self.horizontal_note.set_alpha(.92)
            self._set_scale_marker(.9*(1.0-smooth(q/.18)))
            self.ascent_note.set_alpha(.65*(1-q))
            release_time = float(self.obs["time_s"][-1] * RC.sphere_descent_preview_fraction)
            sphere_time = smooth(q)*release_time
            n = max(2, int(np.searchsorted(self.sphere_data["t"], sphere_time, side="right")))
            nr = max(2, int(np.searchsorted(self.calm_data["t"], sphere_time, side="right")))
            self.sphere_path.set_data(self.sphere_data["x"][:n], self.sphere_data["z"][:n])
            self.sphere_path.set_alpha(.95)
            self.calm_path.set_data(self.calm_data["x"][:nr], self.calm_data["z"][:nr])
            self.calm_path.set_alpha(.78)
            sx = float(np.interp(sphere_time, self.sphere_data["t"], self.sphere_data["x"]))
            sz = float(np.interp(sphere_time, self.sphere_data["t"], self.sphere_data["z"]))
            self.sphere.set_offsets([[sx, sz]])
            self.sphere.set_alpha(1.0)
            self.sensor_dot.set_offsets([[sx, sz]])
            self.sensor_dot.set_alpha(1.0)
            self.release_point.set_offsets([[0.0, self.rocket_data["apogee_m"]]])
            self.release_point.set_alpha(.9)
            halo_q = smooth(q/.25)
            self.release_halo.set_offsets([[sx, sz]])
            self.release_halo.set_sizes([150 + 360*(1-halo_q)])
            self.release_halo.set_alpha(.35*(1-halo_q))
            # The rocket vector fades only after separation; its simulated path persists.
            if q < .36:
                self.rocket.visible(True)
                self.rocket.place(0.0, self.rocket_data["apogee_m"], limits,
                                  flame=0.0, alpha=1.0-smooth(q/.36))
            release_alpha = .96*smooth(q/.12)*(1-smooth((q-.78)/.22))
            self._set_annotation_alpha(self.release_note, release_alpha)
            self.release_note.set_visible(True)
            self.apogee_label.set_text(f"esfera liberada  ·  {self.rocket_data['apogee_m']/1000:.0f} km")
            self._set_annotation_alpha(self.apogee_label, .92*(1.0-smooth(q/.32)))
            self.apogee_label.set_visible(True)
            self.horizontal_note.set_alpha(.92)

        elif stage == 6:
            camera_q = smooth(q/.18)
            limits = lerp_frame(self.release_frame, self.world_frame, camera_q)
            self._set_camera(limits)
            self._set_atmosphere(1.0)
            self._fade_local_artists(0.0)
            self.ground.set_data([limits[0], limits[1]], [0, 0])
            self.ground.set_alpha(.75*camera_q)
            self.horizontal_note.set_alpha(.92)
            self._set_annotation_alpha(self.radar_note, .98)
            self.radar_note.set_visible(True)
            self.radar.visible(True)
            pulse = .5 + .5*np.sin(2*np.pi*(t-STAGE_BOUNDS[6])*.8)
            station = TRACKING.station_x_m[0]
            self.radar.place(station, 0.0, limits, pulse)
            tmax = float(self.obs["time_s"][-1])
            handoff = tmax*RC.sphere_descent_preview_fraction
            sphere_time = handoff + eased*(tmax-handoff)
            n = max(2, int(np.searchsorted(self.sphere_data["t"], sphere_time, side="right")))
            nr = max(2, int(np.searchsorted(self.calm_data["t"], sphere_time, side="right")))
            n = min(n, len(self.sphere_data["t"]))
            nr = min(nr, len(self.calm_data["t"]))
            self.sphere_path.set_data(self.sphere_data["x"][:n], self.sphere_data["z"][:n])
            self.sphere_path.set_alpha(.98)
            self.calm_path.set_data(self.calm_data["x"][:nr], self.calm_data["z"][:nr])
            self.calm_path.set_alpha(.78)
            sx = float(np.interp(sphere_time, self.sphere_data["t"], self.sphere_data["x"]))
            sz = float(np.interp(sphere_time, self.sphere_data["t"], self.sphere_data["z"]))
            self.sphere.set_offsets([[sx, sz]])
            self.sensor_dot.set_offsets([[sx, sz]])
            self.observations.set_offsets(np.empty((0, 2)))
            ids = np.flatnonzero((self.obs["time_s"] <= sphere_time) & self.obs["valid"][:, 0])
            if ids.size:
                polar = self.obs["measurement"][ids, 0, :]
                mx = station + polar[:, 0]*np.cos(polar[:, 1])
                mz = polar[:, 0]*np.sin(polar[:, 1])
                self.observations.set_offsets(np.c_[mx, mz])
            self.horizontal_note.set_alpha(.92)

        else:
            self._set_atmosphere(0.0)
            self.veil.set_alpha(smooth(q/.40))
            reveal = smooth((q-.18)/.48)
            self._set_plot_alpha(reveal)
            self.metric.set_alpha(reveal)
            self.horizontal_note.set_alpha(0.0)
            self.radar_note.set_alpha(0.0)

        if stage == 4:
            self.apogee_label.set_visible(True)
            self._set_annotation_alpha(self.apogee_label, self.apogee_label.get_alpha())
        # A final aspect/safe-margin assertion catches accidental camera jumps
        # before they become clipped deliverables.
        fw, fh = self.fig.get_size_inches()
        assert abs((fw/fh) - (16/9)) < 1e-8
        assert self.ax.get_xlim()[1] > self.ax.get_xlim()[0]
        assert self.ax.get_ylim()[1] > self.ax.get_ylim()[0]
        return stage

    def _set_scale_marker(self, alpha):
        alpha = float(np.clip(alpha, 0.0, 1.0))
        self.scale_riser.set_alpha(alpha)
        for tick in self.scale_ticks:
            tick.set_alpha(alpha)
        self.scale_text.set_alpha(alpha)

    def _set_local_metrics(self, names, alpha=1.0):
        names = tuple(names)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        active = bool(names) and alpha > 0.0
        self.metric_card.set_visible(active)
        self.metric_launch.set_visible(active)
        self.metric_columns.set_visible(active)
        self.metric_launch.set_text(
            f"v₀={PHYSICS.launch_speed:.0f} m/s · θ={PHYSICS.launch_angle_deg:.0f}°")
        card_height = .08 if len(names) <= 2 else .15
        self.metric_card.set_bounds(.62, .99-card_height, .35, card_height)
        self.metric_card.set_alpha(.96*alpha)
        self.metric_launch.set_alpha(alpha)
        self.metric_columns.set_alpha(alpha)
        for name, (_data, _color, label) in self.metric_models.items():
            visible = name in names and active
            row = self.metric_rows[name]
            row.set_visible(visible)
            self.apex_markers[name].set_visible(visible)
            self.range_ticks[name].set_visible(visible)
            if not visible:
                self.apex_markers[name].set_alpha(0.0)
                self.range_ticks[name].set_alpha(0.0)
                continue
            stats = self.metric_stats[name]
            peak = f"{stats['peak_y']:.1f}".replace(".", ",")
            reach = f"{stats['range']:.1f}".replace(".", ",")
            row.set_text(f"{label}: {peak} / {reach}")
            row.set_alpha(alpha)
            self.apex_markers[name].set_offsets([[stats["peak_x"], stats["peak_y"]]])
            self.apex_markers[name].set_alpha(alpha)
            self.range_ticks[name].set_data([stats["range"], stats["range"]], [-.7, .7])
            self.range_ticks[name].set_alpha(alpha)

    @staticmethod
    def _set_annotation_alpha(text, alpha):
        alpha = float(np.clip(alpha, 0.0, 1.0))
        text.set_alpha(alpha)
        bbox = text.get_bbox_patch()
        if bbox is not None:
            bbox.set_alpha(alpha)

    def render(self, path):
        path = Path(path)
        writer = animation.FFMpegWriter(fps=RC.fps, codec="libx264", bitrate=5000,
            extra_args=["-pix_fmt", "yuv420p", "-preset", "medium", "-movflags", "+faststart"])
        keyframes = {
            round(t*RC.fps): name for t, name in [
                (1.0, "slide_01_pedra"), (5.5, "slide_02_arrasto"),
                (8.5, "slide_03_vento"), (12.0, "slide_04_atmosfera"),
                (16.5, "slide_05_foguete"), (20.5, "slide_06_liberacao_esfera"),
                (24.25, "slide_07_radar"), (28.25, "slide_08_recuperacao")]
        }
        KEYFRAME_DIR.mkdir(parents=True, exist_ok=True)
        for stale in KEYFRAME_DIR.glob("slide_*.png"):
            stale.unlink()
        total_frames = int(round(TOTAL_SECONDS*RC.fps))
        with writer.saving(self.fig, str(path), dpi=RC.dpi):
            for frame in range(total_frames):
                self._update_scene(frame)
                writer.grab_frame()
                if frame in keyframes:
                    self.fig.savefig(KEYFRAME_DIR/f"{keyframes[frame]}.png",
                        dpi=RC.keyframe_dpi, facecolor=WHITE)
                if frame and frame % (RC.fps*5) == 0:
                    print(f"  {frame/RC.fps:4.0f} / {TOTAL_SECONDS:.0f} s", flush=True)
        plt.close(self.fig)
        return path


def _add_apogee_label(scene):
    scene.apogee_label = scene.ax.text(.035, .20, "", transform=scene.ax.transAxes,
        color=NAVY, fontsize=12, fontweight="bold", ha="left", va="center",
        alpha=0.0, zorder=30,
        bbox=dict(boxstyle="round,pad=.32", fc=WHITE, ec="#D9E3EA", alpha=.95))


def render_master_and_stages():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    for stale in VIDEO_DIR.glob("slide_*.mp4"):
        stale.unlink()
    data = _prepare_render_data()
    scene = VisualExperiment(data)
    _add_apogee_label(scene)
    master = OUT / "keynote_30s.mp4"
    print(f"Rendering continuous {TOTAL_SECONDS:.1f} s keynote to {master}", flush=True)
    scene.render(master)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to split the rendered master into slide clips")
    for i, (name, duration, _title) in enumerate(STAGES):
        destination = VIDEO_DIR/f"slide_{i+1:02d}_{name}.mp4"
        start = float(STAGE_BOUNDS[i])
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start:.6f}", "-i", str(master), "-t", f"{duration:.6f}",
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)]
        subprocess.run(command, check=True)
        print(f"Wrote {destination.name} ({duration:g} s)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-only", action="store_true",
        help="render the 30-second master, without splitting slide clips")
    args = parser.parse_args()
    if args.master_only:
        OUT.mkdir(exist_ok=True)
        scene = VisualExperiment(_prepare_render_data())
        _add_apogee_label(scene)
        scene.render(OUT/"keynote_30s.mp4")
    else:
        render_master_and_stages()


if __name__ == "__main__":
    main()
