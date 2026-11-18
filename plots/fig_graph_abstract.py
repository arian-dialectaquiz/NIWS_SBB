"""
graphical_abstract.py

Graphical abstract of the Catarina near-inertial manuscript, a schematic drawn
entirely in code (no model output is read). Two block diagrams of the shelf and
slope of the South Brazil Bight seen from the south-east, the coast on the left

    blocks[0, 0]   the storm, the anticyclones drain the waves into the subinertial flow
    blocks[0, 1]   the wake, the flow returns most of it wherever the waves are

and below them three panels on the mechanisms, numbered as the markers in the
blocks

    mech[0, 0]   1 refraction gathers the waves in the anticyclones, where they
                   descend to the shear of the Brazil Current
    mech[0, 1]   2 the vertical shear production, the wave stress along the shear
                   in the storm (the waves lose) and against it in the wake (they gain)
    mech[0, 2]   3 the strain does little work on nearly circular inertial
                   currents that turn faster than it deforms them

The top face of each block is the sea surface (the eddies, the Brazil Current,
the storm track and the near-inertial currents) and the front face is a
cross-shelf section (the mixed layer, the isopycnals, the shear of the Brazil
Current, the near-inertial waves under the anticyclone, the exchange and the
dissipation). Depth and distance are schematic.

The colours follow the figures of the manuscript, steelblue for the wind work,
black for the exchange and darkorchid for the dissipation. Every colour is a
named colour, the legend sits below the blocks and no figure text has a comma.

    python graphical_abstract.py
"""

import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:                                    # the style of the manuscript figures when it is at hand
    import style_v3 as S
    S.apply()
    PAGE_W_IN, FS = S.PAGE_W_IN, dict(S.FS)
except Exception:
    PAGE_W_IN, FS = 7.48, dict(small=7.0, tick=7.5, label=8.5)
    mpl.rcParams.update({'font.family': 'sans-serif', 'mathtext.fontset': 'dejavusans'})

# =====================================================================
# CONFIGURATION
# =====================================================================

OUT_STEM = os.path.join(os.getcwd(), 'graphical_abstract')
FIGSIZE = (PAGE_W_IN, 5.0)
DPI = 400
VIEW = dict(elev=23.0, azim=-61.0)      # seen from the south-east
BOX_ASPECT = (325.0, 260.0, 215.0)      # x, y and z lengths of the drawn box
ZOOM = 1.04
BLOCK_TOP, BLOCK_BOTTOM = 1.03, 0.40    # figure fractions of the two blocks
MECH_TOP, MECH_BOTTOM = 0.345, 0.085    # figure fractions of the three mechanism panels
MECH_WIDTHS = (1.0, 1.25, 1.0)         # relative widths, the vertical panel holds two diagrams
REVERSAL_Y = 0.867                      # height of the reversal arrow, fraction of the block row

# geometry, x across the shelf (km, coast at 0), y along it (km), z schematic (surface 0, bottom -1)
X_LAND, X_OFF = -25.0, 300.0
Y_SOUTH, Y_NORTH = 0.0, 260.0
Z_FLOOR = -1.08                         # base of the block
Z_LAND = 0.04                           # height of the land
X_BREAK, X_FOOT = 110.0, 190.0          # shelf break and foot of the slope
H_COAST, H_BREAK = 0.07, 0.27           # depth at the coast and at the shelf break
Z_ML, Z_TC = -0.09, -0.46               # mixed-layer base and base of the thermocline
X_BC = 162.0                            # core of the Brazil Current
XA, RA = 236.0, 54.0                    # anticyclone cut by the section, centre and radius
XC, YC, RC = 246.0, 196.0, 36.0         # cyclone, centre and radius
Y_TRACK = 150.0                         # the storm crossed westward along this line
STORM = dict(x=150.0, y=150.0, z=0.56, r=58.0)

# named colours, one per quantity as in the manuscript figures
C_WIND = 'steelblue'                    # wind work
C_PI = 'black'                          # exchange
C_EPS = 'darkorchid'                    # dissipation and bottom drag
C_WAVE = 'darkgoldenrod'                # near-inertial waves
C_FLOW = 'midnightblue'                 # subinertial flow
C_ISO = 'slategrey'                     # isopycnals
C_ML = 'black'                          # mixed-layer base
C_TRACK = 'dimgrey'
C_STORM, C_STORM_EDGE = 'whitesmoke', 'grey'
C_SURFACE = 'lightcyan'
C_FRONT = dict(ml='paleturquoise', tc='lightskyblue', deep='lightsteelblue')
C_SIDE = dict(ml='powderblue', tc='skyblue', deep='cornflowerblue')
C_EARTH, C_EARTH_SIDE, C_LAND = 'tan', 'burlywood', 'darkseagreen'
C_ANTI, C_CYC = 'lightsteelblue', 'mistyrose'
C_TEXT, C_NOTE = 'black', 'dimgrey'
STAGE_COLOUR = {'storm': 'firebrick', 'wake': 'royalblue'}

# what changes between the two blocks
STAGES = {
    'storm': dict(
        title='During the storm',
        subtitle=None,
        storm=True,
        circles=[(52, 70, 9), (92, 112, 10), (122, 62, 10), (58, 126, 8), (200, 104, 9)],
        circles_anti=[(214, 20, 12), (250, 26, 13), (233, 46, 9)],
        rays=[(XA - 50, XA - 19, 1.6, 1.0), (XA + 50, XA + 19, 1.6, 1.0)],
        exchange=[(XA, -0.17, -0.385, 1.0)],          # x, z start, z end, width
        exchange_label='waves to flow',
        eps=[(40, 2.2), (78, 2.2), (116, 2.2)],       # x, line width
    ),
    'wake': dict(
        title='In the wake',
        subtitle=None,
        storm=False,
        circles=[(92, 112, 6), (122, 62, 6)],
        circles_anti=[(214, 20, 9), (250, 26, 10), (233, 46, 7)],
        rays=[(XA - 50, XA - 19, 1.1, 0.75), (XA + 50, XA + 19, 1.1, 0.75)],
        exchange=[(XA, -0.36, -0.17, 0.62), (190, -0.33, -0.16, 0.62), (285, -0.33, -0.16, 0.62)],
        exchange_label='flow to waves',
        eps=[(40, 1.2), (78, 1.2), (116, 1.2)],
    ),
}

# the numbered markers that tie the blocks to the mechanism panels
MARKERS = {1: (XA + 58.0, 40.0, 0.0), 2: (XA - 36.0, 0.0, -0.545), 3: (18.0, 98.0, 0.0)}

# mechanism panels
C_SHEAR = C_FLOW                        # the subinertial shear vector
C_STRESS = C_WAVE                       # the vertical wave Reynolds stress vector
C_STRAIN = 'silver'                     # streamlines of the strain

# drawing order, the faces first and what lies on them after
Z_FACE, Z_ON_TOP, Z_SIDE, Z_FRONT, Z_ON_FRONT, Z_AIR, Z_TEXT = 1, 3, 5, 6, 8, 12, 20


# =====================================================================
# GEOMETRY
# =====================================================================

def depth(x):
    """Schematic water depth across the shelf, positive down."""
    x = np.asarray(x, dtype=float)
    shelf = H_COAST + (H_BREAK - H_COAST) * np.clip(x / X_BREAK, 0.0, 1.0)
    s = np.clip((x - X_BREAK) / (X_FOOT - X_BREAK), 0.0, 1.0)
    return np.where(x <= X_BREAK, shelf, H_BREAK + (1.0 - H_BREAK) * 0.5 * (1.0 - np.cos(np.pi * s)))


def isopycnal(x, z0):
    """An isopycnal of the section, deepened in the anticyclone and rising
    towards the coast across the Brazil Current (the thermal-wind shear)."""
    bowl = -0.075 * np.exp(-((x - XA) / 36.0) ** 2)
    rise = 0.075 * 0.5 * (1.0 - np.tanh((x - X_BC) / 24.0))
    z = z0 + bowl + rise
    z = np.where(z < -depth(x) + 0.012, np.nan, z)
    return np.where(z > Z_ML - 0.012, np.nan, z)


# =====================================================================
# DRAWING TOOLS
# =====================================================================

class Arrow3D(FancyArrowPatch):
    """A flat arrow between two points of the block."""

    def __init__(self, p0, p1, **kw):
        super().__init__((0, 0), (0, 0), **kw)
        self._p0, self._p1 = p0, p1

    def do_3d_projection(self, renderer=None):
        xs, ys, zs = proj3d.proj_transform(*zip(self._p0, self._p1), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return float(np.min(zs))


def face(ax, pts, colour, zorder, edge='none', lw=0.0, alpha=1.0):
    ax.add_collection3d(Poly3DCollection([pts], facecolors=colour, edgecolors=edge, linewidths=lw,
                                         alpha=alpha, zorder=zorder))


def head(ax, p0, p1, colour, zorder, size=7.0, lw=1.0):
    """An arrow head at p1 that points along p0 to p1."""
    ax.add_artist(Arrow3D(p0, p1, arrowstyle='-|>', mutation_scale=size, color=colour, lw=lw,
                          shrinkA=0, shrinkB=0, zorder=zorder))


def block_arrow(ax, p0, p1, colour, zorder, width=1.0, scale=11.0):
    """A filled arrow, for the kinetic energy that the wind and the exchange move."""
    style = f'simple,head_length={0.95 * width:.2f},head_width={1.55 * width:.2f},tail_width={0.62 * width:.2f}'
    ax.add_artist(Arrow3D(p0, p1, arrowstyle=style, mutation_scale=scale, facecolor=colour, edgecolor='white',
                          lw=0.5, shrinkA=0, shrinkB=0, zorder=zorder))


def arc(ax, cx, cy, r, th0, th1, colour, zorder, lw=1.2, z=0.0, size=6.5):
    """A circular arrow on a horizontal plane, anticlockwise where th1 > th0."""
    th = np.radians(np.linspace(th0, th1, 80))
    x, y = cx + r * np.cos(th), cy + r * np.sin(th)
    ax.plot(x[:-3], y[:-3], np.full(th.size - 3, z), color=colour, lw=lw, solid_capstyle='butt', zorder=zorder)
    head(ax, (x[-8], y[-8], z), (x[-1], y[-1], z), colour, zorder, size=size, lw=lw)


def ray(ax, x0, x1, z0, z1, colour, zorder, lw=1.4, alpha=1.0, cycles=4.5, amp=4.5):
    """A wavy line on the front face from (x0, z0) to (x1, z1), a near-inertial
    wave that leaves the mixed layer."""
    t = np.linspace(0.0, 1.0, 160)
    taper = np.sin(np.pi * np.clip(t / 0.9, 0, 1)) ** 0.5
    x = x0 + (x1 - x0) * t + amp * taper * np.sin(2 * np.pi * cycles * t)
    z = z0 + (z1 - z0) * t
    ax.plot(x[:-6], np.zeros(t.size - 6), z[:-6], color=colour, lw=lw, alpha=alpha, zorder=zorder)
    head(ax, (x1 - 0.12 * (x1 - x0), 0.0, z1 - 0.12 * (z1 - z0)), (x1, 0.0, z1), colour, zorder, size=6.5, lw=lw)


# =====================================================================
# THE BLOCK
# =====================================================================

def draw_faces(ax):
    xs = np.linspace(0.0, X_OFF, 240)
    zb = -depth(xs)
    # sea surface and land
    face(ax, [(0, Y_SOUTH, 0), (X_OFF, Y_SOUTH, 0), (X_OFF, Y_NORTH, 0), (0, Y_NORTH, 0)], C_SURFACE, Z_FACE,
         edge='lightslategrey', lw=0.4)
    face(ax, [(X_LAND, Y_SOUTH, Z_LAND), (0, Y_SOUTH, Z_LAND), (0, Y_NORTH, Z_LAND), (X_LAND, Y_NORTH, Z_LAND)],
         C_LAND, Z_FACE, edge='dimgrey', lw=0.4)
    face(ax, [(X_LAND, Y_SOUTH, Z_FLOOR), (0, Y_SOUTH, Z_FLOOR), (0, Y_SOUTH, Z_LAND), (X_LAND, Y_SOUTH, Z_LAND)],
         C_EARTH, Z_FRONT, edge='dimgrey', lw=0.4)
    # the offshore face, three layers of water over the sea bed
    for z0, z1, c in ((0.0, Z_ML, C_SIDE['ml']), (Z_ML, Z_TC, C_SIDE['tc']), (Z_TC, -1.0, C_SIDE['deep']),
                      (-1.0, Z_FLOOR, C_EARTH_SIDE)):
        face(ax, [(X_OFF, Y_SOUTH, z0), (X_OFF, Y_NORTH, z0), (X_OFF, Y_NORTH, z1), (X_OFF, Y_SOUTH, z1)], c, Z_SIDE)
    ax.plot([X_OFF, X_OFF], [Y_SOUTH, Y_NORTH], [Z_ML, Z_ML], color=C_ML, lw=0.9, zorder=Z_SIDE + 1)
    # the section, the same layers cut by the sea bed
    for top, bot, c in ((0.0, Z_ML, C_FRONT['ml']), (Z_ML, Z_TC, C_FRONT['tc']), (Z_TC, -1.0, C_FRONT['deep'])):
        up, lo = np.maximum(top, zb), np.maximum(bot, zb)
        pts = [(x, Y_SOUTH, z) for x, z in zip(xs, up)] + [(x, Y_SOUTH, z) for x, z in zip(xs[::-1], lo[::-1])]
        face(ax, pts, c, Z_FRONT)
    pts = [(x, Y_SOUTH, z) for x, z in zip(xs, zb)] + [(X_OFF, Y_SOUTH, Z_FLOOR), (0, Y_SOUTH, Z_FLOOR)]
    face(ax, pts, C_EARTH, Z_FRONT, edge='dimgrey', lw=0.4)
    # edges of the block
    for xe, ye in ((X_OFF, Y_SOUTH), (X_OFF, Y_NORTH)):
        ax.plot([xe, xe], [ye, ye], [0.0, Z_FLOOR], color='dimgrey', lw=0.5, zorder=Z_ON_FRONT)
    ax.plot([0, X_OFF], [Y_SOUTH, Y_SOUTH], [0, 0], color='dimgrey', lw=0.6, zorder=Z_ON_FRONT)
    ax.plot([X_OFF, X_OFF], [Y_SOUTH, Y_NORTH], [Z_FLOOR, Z_FLOOR], color='dimgrey', lw=0.5, zorder=Z_ON_FRONT)


def draw_surface(ax, st, labels):
    # the storm track, westward to the coast
    ax.plot([X_OFF, 14], [Y_TRACK, Y_TRACK], [0, 0], color=C_TRACK, ls=(0, (4, 2)), lw=1.0, zorder=Z_ON_TOP)
    head(ax, (20, Y_TRACK, 0), (4, Y_TRACK, 0), C_TRACK, Z_ON_TOP, size=7.0)
    # the eddies, the anticyclone cut by the section (anticlockwise in the Southern Hemisphere)
    th = np.radians(np.linspace(0.0, 180.0, 90))
    face(ax, [(XA + RA * np.cos(t), RA * np.sin(t), 0.0) for t in th], C_ANTI, Z_ON_TOP - 1, edge=C_FLOW, lw=0.5)
    arc(ax, XA, 0.0, RA - 7.0, 12.0, 150.0, C_FLOW, Z_ON_TOP, lw=1.3)
    th = np.radians(np.linspace(0.0, 360.0, 120))
    face(ax, [(XC + RC * np.cos(t), YC + RC * np.sin(t), 0.0) for t in th], C_CYC, Z_ON_TOP - 1, edge=C_FLOW, lw=0.5)
    arc(ax, XC, YC, RC - 7.0, 250.0, -40.0, C_FLOW, Z_ON_TOP, lw=1.3)
    # the Brazil Current, poleward along the slope
    y = np.linspace(Y_NORTH - 6.0, 10.0, 120)
    x = X_BC + 9.0 * np.sin(2 * np.pi * (y - 30.0) / 300.0)
    ax.plot(x[:-4], y[:-4], np.zeros(y.size - 4), color=C_FLOW, lw=2.4, solid_capstyle='butt', zorder=Z_ON_TOP)
    for i in (40, 80, 119):
        head(ax, (x[i - 9], y[i - 9], 0.0), (x[i], y[i], 0.0), C_FLOW, Z_ON_TOP, size=10.0, lw=1.6)
    # the near-inertial currents, anticlockwise circles
    for cx, cy, r in st['circles'] + st['circles_anti']:
        arc(ax, cx, cy, r, 100.0, 420.0, C_WAVE, Z_ON_TOP + 1, lw=1.2, size=5.5)
    if labels:
        ax.text(XA, RA + 24.0, 0.0, 'anticyclone', ha='center', va='center', fontsize=FS['small'], color=C_TEXT,
                zorder=Z_TEXT)
        ax.text(XC, YC, 0.0, 'cyclone', ha='center', va='center', fontsize=FS['small'], color=C_TEXT, zorder=Z_TEXT)
        ax.text(X_LAND - 8.0, 150.0, Z_LAND, 'coast', ha='right', va='center', fontsize=FS['small'],
                color=C_TEXT, zorder=Z_TEXT)


def draw_section(ax, st, labels):
    xs = np.linspace(0.0, X_OFF, 400)
    # mixed-layer base and isopycnals
    zml = np.where(-depth(xs) < Z_ML, Z_ML, np.nan)
    ax.plot(xs, np.zeros_like(xs), zml, color=C_ML, lw=0.9, zorder=Z_ON_FRONT)
    for z0 in (-0.16, -0.235, -0.31, -0.385):
        ax.plot(xs, np.zeros_like(xs), isopycnal(xs, z0), color=C_ISO, lw=0.7, zorder=Z_ON_FRONT)
    # the Brazil Current towards the reader, weaker with depth
    for z, s in ((-0.045, 62.0), (-0.20, 36.0), (-0.36, 16.0)):
        ax.scatter([X_BC], [0.0], [z], s=s, facecolor='white', edgecolor=C_FLOW, linewidth=1.0, depthshade=False,
                   zorder=Z_ON_FRONT + 2)
        ax.scatter([X_BC], [0.0], [z], s=s / 9.0, color=C_FLOW, depthshade=False, zorder=Z_ON_FRONT + 3)
    # near-inertial waves that leave the mixed layer
    for x0, x1, lw, alpha in st['rays']:
        ray(ax, x0, x1, Z_ML, -0.37, C_WAVE, Z_ON_FRONT + 1, lw=lw, alpha=alpha)
    # the exchange, downward where the waves lose kinetic energy and upward where they gain it
    for x, z0, z1, w in st['exchange']:
        block_arrow(ax, (x, 0.0, z0), (x, 0.0, z1), C_PI, Z_ON_FRONT + 4, width=w)
    ax.text(XA - 22.0, 0.0, -0.545, st['exchange_label'], ha='left', va='center', fontsize=FS['small'],
            color=C_TEXT, fontweight='bold', zorder=Z_TEXT)
    # dissipation and bottom drag on the shelf and the upper slope
    for x, lw in st['eps']:
        zb = float(-depth(x))
        z0 = -0.015
        ax.add_artist(Arrow3D((x, 0.0, z0), (x, 0.0, zb + 0.004), arrowstyle='-|>', mutation_scale=7.0 + 1.5 * lw,
                              color=C_EPS, lw=lw, ls=(0, (2.2, 1.2)), shrinkA=0, shrinkB=0, zorder=Z_ON_FRONT + 2))
    ax.text(62.0, 0.0, -0.66, 'dissipated\non the shelf', ha='center', va='center', fontsize=FS['small'],
            color=C_TEXT, linespacing=0.95, zorder=Z_TEXT)
    if labels:
        ax.text(X_BC + 34.0, 0.0, -0.84, 'shear of the Brazil Current', ha='left', va='center',
                fontsize=FS['small'], color=C_TEXT, linespacing=0.95, zorder=Z_TEXT)
        ax.add_artist(Arrow3D((X_BC + 31.0, 0.0, -0.82), (X_BC + 2.0, 0.0, -0.42), arrowstyle='-', color=C_NOTE,
                              lw=0.5, shrinkA=0, shrinkB=0, zorder=Z_TEXT))


def draw_storm(ax):
    x0, y0, z0, r0 = STORM['x'], STORM['y'], STORM['z'], STORM['r']
    th = np.radians(np.linspace(0.0, 360.0, 120))
    face(ax, [(x0 + r0 * np.cos(t), y0 + r0 * np.sin(t), z0) for t in th], C_STORM, Z_AIR, edge=C_STORM_EDGE,
         lw=0.6, alpha=0.93)
    # spiral bands, clockwise in the Southern Hemisphere
    r = np.linspace(7.0, r0 - 4.0, 80)
    for a0 in (0.0, 120.0, 240.0):
        t = np.radians(a0) + 2.3 * np.log(r / 7.0)
        ax.plot(x0 + r * np.cos(t), y0 + r * np.sin(t), np.full(r.size, z0), color=C_STORM_EDGE, lw=1.6,
                zorder=Z_AIR + 1)
    arc(ax, x0, y0, 15.0, 200.0, -100.0, C_TRACK, Z_AIR + 2, lw=1.2, z=z0, size=6.0)
    ax.plot([x0, x0], [y0, y0], [0.0, z0], color=C_TRACK, ls='dotted', lw=0.9, zorder=Z_AIR - 2)
    ax.scatter([x0], [y0], [0.0], s=14, marker='x', color=C_TRACK, linewidth=1.0, depthshade=False,
               zorder=Z_ON_TOP + 2)
    # the wind work on the poleward flank, where the stress turns with the inertial current
    block_arrow(ax, (x0 - 4.0, 92.0, z0 - 0.08), (x0 - 4.0, 92.0, 0.03), C_WIND, Z_AIR - 1, width=1.25)
    ax.text(x0 - 24.0, 92.0, 0.47, 'wind work', ha='right', va='center', fontsize=FS['small'], color=C_TEXT,
            fontweight='bold', linespacing=0.95, zorder=Z_TEXT)
    ax.text(x0, y0, z0 + 0.27, 'Hurricane Catarina', ha='center', va='bottom', fontsize=FS['small'], color=C_TEXT,
            zorder=Z_TEXT)


def draw_block(fig, cell, key, labels):
    st = STAGES[key]
    ax = fig.add_subplot(cell, projection='3d', computed_zorder=False)
    ax.set_proj_type('ortho')
    ax.view_init(**VIEW)
    ax.set_xlim(X_LAND, X_OFF)
    ax.set_ylim(Y_SOUTH, Y_NORTH)
    ax.set_zlim(Z_FLOOR, 0.75)
    ax.set_box_aspect(BOX_ASPECT, zoom=ZOOM)
    ax.set_axis_off()
    draw_faces(ax)
    draw_surface(ax, st, labels)
    draw_section(ax, st, labels)
    if st['storm']:
        draw_storm(ax)
    for n, (x, y, z) in MARKERS.items():
        marker(ax, n, x, y, z)
    return ax


def marker(ax, n, x, y, z=None, size=None):
    """A numbered circle, in the block (3d) or in a mechanism panel (2d)."""
    kw = dict(ha='center', va='center', fontsize=size or FS['small'], fontweight='bold', color=C_TEXT,
              bbox=dict(boxstyle='circle,pad=0.22', fc='white', ec=C_TEXT, lw=0.8), zorder=Z_TEXT + 2)
    if z is None:
        ax.text(x, y, str(n), transform=ax.transAxes, **kw)
    else:
        ax.text(x, y, z, str(n), **kw)


# =====================================================================
# MECHANISM PANELS (2d, depth negative downward, equal aspect)
# =====================================================================

def wavy(ax, p0, p1, colour, lw=1.4, cycles=4.0, amp=0.03, zorder=4, alpha=1.0):
    """A wavy arrow from p0 to p1, a near-inertial wave."""
    t = np.linspace(0.0, 1.0, 160)
    dx, dz = p1[0] - p0[0], p1[1] - p0[1]
    n = np.array([-dz, dx]) / np.hypot(dx, dz)
    taper = np.sin(np.pi * np.clip(t / 0.88, 0, 1)) ** 0.5
    w = amp * taper * np.sin(2 * np.pi * cycles * t)
    x, z = p0[0] + dx * t + n[0] * w, p0[1] + dz * t + n[1] * w
    ax.plot(x[:-8], z[:-8], color=colour, lw=lw, alpha=alpha, zorder=zorder, solid_capstyle='butt')
    ax.add_patch(FancyArrowPatch((x[-14], z[-14]), (p1[0], p1[1]), arrowstyle='-|>', mutation_scale=7.5,
                                 color=colour, lw=lw, alpha=alpha, shrinkA=0, shrinkB=0, zorder=zorder))


def vec(ax, p0, p1, colour, lw=1.8, zorder=5, ms=10.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=ms, color=colour, lw=lw, shrinkA=0,
                                 shrinkB=0, zorder=zorder))


def mech_title(ax, n, title, subtitle):
    """The numbered title and a short statement above a mechanism panel."""
    nl = subtitle.count('\n') + 1
    y_sub = 3.0
    y_tit = y_sub + nl * FS['small'] * 1.22 + 4.0
    kw = dict(xy=(0.0, 1.0), xycoords='axes fraction', textcoords='offset points', annotation_clip=False)
    ax.annotate(subtitle, xytext=(0.0, y_sub), ha='left', va='bottom', fontsize=FS['small'], color=C_TEXT,
                linespacing=1.05, **kw)
    ax.annotate(str(n), xytext=(5.0, y_tit + 0.45 * FS['label']), ha='center', va='center', fontsize=FS['small'],
                fontweight='bold', color=C_TEXT, bbox=dict(boxstyle='circle,pad=0.22', fc='white', ec=C_TEXT, lw=0.8),
                **kw)
    ax.annotate(title, xytext=(14.0, y_tit), ha='left', va='bottom', fontsize=FS['label'], fontweight='bold',
                color=C_TEXT, **kw)


def mech_frame(ax, xlim, zlim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_aspect('equal')
    ax.set_anchor('N')                  # the three panels share their top edge
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color('lightgrey')
        sp.set_linewidth(0.6)


def mech_refraction(ax):
    """A section across an anticyclone. Its negative vorticity lowers the
    effective inertial frequency, so the waves are trapped in the core and
    descend there to the shear (Kunze 1985)."""
    x = np.linspace(-0.9, 0.9, 300)
    bowl = np.exp(-(x / 0.42) ** 2)
    zml = -0.13 - 0.03 * bowl
    ztc = -0.62 - 0.20 * bowl
    ax.fill_between(x, 0.0, zml, color=C_FRONT['ml'], lw=0, zorder=0)
    ax.fill_between(x, zml, ztc, color=C_FRONT['tc'], lw=0, zorder=0)
    ax.fill_between(x, ztc, -1.05, color=C_FRONT['deep'], lw=0, zorder=0)
    ax.plot(x, zml, color=C_ML, lw=0.9, zorder=2)
    for z0 in (-0.25, -0.36, -0.47):
        ax.plot(x, z0 - 0.16 * bowl, color=C_ISO, lw=0.7, zorder=1)
    # the eddy turns anticlockwise seen from above, out of the page on the west flank in a section looking north
    for xs, sym in ((-0.37, 'out'), (0.37, 'in')):
        ax.scatter([xs], [-0.065], s=60, facecolor='white', edgecolor=C_FLOW, linewidth=1.0, zorder=5)
        if sym == 'out':
            ax.scatter([xs], [-0.065], s=7, color=C_FLOW, zorder=6)
        else:
            ax.scatter([xs], [-0.065], s=26, marker='x', color=C_FLOW, linewidth=1.0, zorder=6)
    ax.text(0.0, -0.065, 'anticyclone', ha='center', va='center', fontsize=FS['small'], color=C_TEXT, zorder=6)
    # the waves leave the mixed layer, bend into the core and descend
    for s in (-1, 1):
        wavy(ax, (s * 0.74, -0.15), (s * 0.13, -0.80), C_WAVE, lw=1.4, cycles=4.5, amp=0.028)
        wavy(ax, (s * 0.86, -0.15), (s * 0.66, -0.36), C_WAVE, lw=1.0, cycles=2.0, amp=0.02, alpha=0.6)
    ax.text(0.0, -0.93, r'$f_{\mathrm{eff}} < f$', ha='center', va='center', fontsize=FS['label'], color=C_TEXT,
            zorder=6)
    for s in (-1, 1):
        ax.text(s * 0.70, -0.93, r'$f_{\mathrm{eff}} \approx f$', ha='center', va='center', fontsize=FS['small'],
                color=C_NOTE, zorder=6)
    mech_frame(ax, (-0.9, 0.9), (-1.05, 0.0))
    mech_title(ax, 1, 'Refraction', 'the waves gather in the anticyclones\nand descend to the shear')


def mech_vertical(ax):
    """The vertical shear production. On the left the Brazil Current weakens
    beneath the mixed layer, on the right the wave stress against the shear,
    aligned in the storm and opposed in the wake."""
    z = np.linspace(-1.0, 0.0, 200)
    u = 0.08 + 0.50 * 0.5 * (1.0 + np.tanh((z + 0.36) / 0.10))
    x0 = 0.06
    zml = -0.13
    ax.fill_between([0.0, 0.70], 0.0, zml, color=C_FRONT['ml'], lw=0, zorder=0)
    ax.fill_between([0.0, 0.70], zml, -1.05, color=C_FRONT['tc'], lw=0, zorder=0)
    ax.fill_between([0.0, 0.70], -0.20, -0.52, color=C_PI, alpha=0.10, lw=0, zorder=0)
    ax.plot([0.0, 0.70], [zml, zml], color=C_ML, lw=0.9, zorder=2)
    ax.plot(x0 + u, z, color=C_FLOW, lw=1.3, zorder=3)
    ax.plot([x0, x0], [0.0, -1.0], color=C_FLOW, lw=0.7, zorder=3)
    for zz in (-0.06, -0.24, -0.36, -0.48, -0.75):
        uu = 0.08 + 0.50 * 0.5 * (1.0 + np.tanh((zz + 0.36) / 0.10))
        vec(ax, (x0, zz), (x0 + uu, zz), C_FLOW, lw=1.0, ms=6.0)
    ax.text(0.37, -0.68, 'Brazil\nCurrent', ha='center', va='center', fontsize=FS['small'], color=C_TEXT,
            linespacing=0.95, zorder=6)
    ax.text(0.68, -0.46, 'shear', ha='right', va='center', fontsize=FS['small'], color=C_TEXT,
            fontweight='bold', zorder=6)
    ax.plot([0.72, 0.72], [0.0, -1.05], color='lightgrey', lw=0.6, zorder=2)
    # the two stages, the shear vector and the wave stress at the depth of the exchange
    cases = (('storm', -0.25, 28.0, 'along the shear', 'the waves lose'),
             ('wake', -0.64, 152.0, 'against the shear', 'the waves gain'))
    cx, L, xt = 0.95, 0.20, 1.30
    for name, cz, ang, how, what in cases:
        ax.scatter([cx], [cz], s=10, color=C_TEXT, zorder=7)
        vec(ax, (cx, cz), (cx + L, cz), C_SHEAR, lw=1.8)
        a = np.radians(ang)
        vec(ax, (cx, cz), (cx + L * np.cos(a), cz + L * np.sin(a)), C_STRESS, lw=1.8)
        th = np.radians(np.linspace(0.0, ang, 40))
        ax.plot(cx + 0.075 * np.cos(th), cz + 0.075 * np.sin(th), color=C_NOTE, lw=0.7, zorder=4)
        ax.text(xt, cz + 0.11, name, ha='left', va='center', fontsize=FS['label'], fontweight='bold',
                color=STAGE_COLOUR[name], zorder=6)
        ax.text(xt, cz, how, ha='left', va='center', fontsize=FS['small'], color=C_TEXT, zorder=6)
        ax.text(xt, cz - 0.10, what, ha='left', va='center', fontsize=FS['small'], color=C_TEXT,
                fontweight='bold', zorder=6)
    ax.text(cx + 0.08, -0.25 - 0.045, 'shear', ha='center', va='top', fontsize=FS['small'], color=C_SHEAR,
            zorder=6)
    ax.text(cx - 0.02, -0.135, 'wave stress', ha='center', va='bottom', fontsize=FS['small'], color=C_STRESS,
            zorder=6)
    mech_frame(ax, (0.0, 2.1), (-1.05, 0.0))
    mech_title(ax, 2, 'Vertical shear production', 'the wave stress meets the shear of the\nBrazil Current beneath the mixed layer')


def mech_strain(ax):
    """The strain of the eddies and a near-inertial current. The current is
    nearly circular and its ellipse turns at about twice the inertial frequency,
    much faster than the strain deforms it, so the deformation work averages
    toward zero over an inertial period."""
    g = np.linspace(-0.9, 0.9, 60)
    X, Y = np.meshgrid(g, np.linspace(-0.55, 0.55, 40))
    ax.streamplot(X, Y, X, -Y, color=C_STRAIN, linewidth=0.7, density=0.75, arrowsize=0.6, zorder=1)
    # the strain axes
    vec(ax, (0.50, 0.0), (0.84, 0.0), C_NOTE, lw=1.2, ms=8.0)
    vec(ax, (-0.50, 0.0), (-0.84, 0.0), C_NOTE, lw=1.2, ms=8.0)
    vec(ax, (0.0, 0.53), (0.0, 0.38), C_NOTE, lw=1.2, ms=8.0)
    vec(ax, (0.0, -0.53), (0.0, -0.38), C_NOTE, lw=1.2, ms=8.0)
    ax.text(0.86, 0.06, 'strain', ha='right', va='bottom', fontsize=FS['small'], color=C_NOTE, zorder=6)
    # the near-inertial current, a nearly circular ellipse turning anticlockwise
    t = np.linspace(0.0, 2 * np.pi, 200)
    ra, rb, phi = 0.27, 0.23, np.radians(35.0)
    ex = ra * np.cos(t) * np.cos(phi) - rb * np.sin(t) * np.sin(phi)
    ey = ra * np.cos(t) * np.sin(phi) + rb * np.sin(t) * np.cos(phi)
    ax.fill(ex, ey, color='white', alpha=0.85, lw=0, zorder=2)
    ax.plot(ex, ey, color=C_WAVE, lw=1.8, zorder=4)
    k = 60
    ax.add_patch(FancyArrowPatch((ex[k - 6], ey[k - 6]), (ex[k], ey[k]), arrowstyle='-|>', mutation_scale=9,
                                 color=C_WAVE, lw=1.6, shrinkA=0, shrinkB=0, zorder=5))
    # the fast turn of the ellipse
    th = np.radians(np.linspace(-30.0, 60.0, 50))
    rr = 0.36
    ax.plot(rr * np.cos(th[:-4]), rr * np.sin(th[:-4]), color=C_TEXT, lw=0.9, zorder=4)
    ax.add_patch(FancyArrowPatch((rr * np.cos(th[-6]), rr * np.sin(th[-6])), (rr * np.cos(th[-1]), rr * np.sin(th[-1])),
                                 arrowstyle='-|>', mutation_scale=8, color=C_TEXT, lw=0.9, shrinkA=0, shrinkB=0,
                                 zorder=5))
    ax.text(0.0, 0.0, 'nearly\ncircular', ha='center', va='center', fontsize=FS['small'], color=C_TEXT,
            linespacing=0.95, zorder=6)
    ax.text(-0.86, -0.50, 'the ellipse turns faster\nthan the strain deforms it', ha='left', va='bottom',
            fontsize=FS['small'], color=C_TEXT, linespacing=0.95, zorder=6,
            bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.0))
    mech_frame(ax, (-0.9, 0.9), (-0.55, 0.55))
    mech_title(ax, 3, 'Little work by the strain', 'nearly circular inertial currents keep the\nhorizontal shear production small')


# =====================================================================
# FIGURE
# =====================================================================

def main():
    fig = plt.figure(figsize=FIGSIZE)
    blocks = GridSpec(1, 2, figure=fig, left=0.0, right=1.0, top=BLOCK_TOP, bottom=BLOCK_BOTTOM, wspace=0.0)
    for i, key in enumerate(('storm', 'wake')):
        draw_block(fig, blocks[0, i], key, labels=(i == 0))
        xc = 0.25 + 0.5 * i
        fig.text(xc, 0.992, STAGES[key]['title'], ha='center', va='top', fontsize=FS['label'] + 2,
                 fontweight='bold', color=STAGE_COLOUR[key])
        fig.text(xc, 0.962, STAGES[key]['subtitle'], ha='center', va='top', fontsize=FS['label'], color=C_TEXT)

    # the reversal, between the two blocks
    yr = BLOCK_BOTTOM + REVERSAL_Y * (BLOCK_TOP - BLOCK_BOTTOM)
    fig.add_artist(FancyArrowPatch((0.455, yr), (0.545, yr), transform=fig.transFigure, arrowstyle='-|>',
                                   mutation_scale=12, color=C_PI, lw=1.4))
    fig.text(0.5, yr + 0.009, 'the exchange reverses', ha='center', va='bottom', fontsize=FS['small'], color=C_TEXT)
    fig.text(0.5, yr - 0.009, 'three inertial periods\nafter landfall', ha='center', va='top',
             fontsize=FS['small'], color=C_NOTE, linespacing=1.0)

    # the mechanisms, numbered as the markers in the blocks
    mech = GridSpec(1, 3, figure=fig, left=0.035, right=0.975, top=MECH_TOP, bottom=MECH_BOTTOM, wspace=0.14,
                    width_ratios=MECH_WIDTHS)
    mech_refraction(fig.add_subplot(mech[0, 0]))
    mech_vertical(fig.add_subplot(mech[0, 1]))
    mech_strain(fig.add_subplot(mech[0, 2]))

    # one legend below the blocks, outside the drawings
    hs = [Line2D([], [], color=C_WIND, lw=3.4, label='wind work'),
          Line2D([], [], color=C_PI, lw=3.4, label='exchange between waves and flow'),
          Line2D([], [], color=C_EPS, lw=1.8, ls=(0, (2.2, 1.2)), label='dissipation and bottom drag'),
          Line2D([], [], color=C_WAVE, lw=1.3, marker='o', ms=6, markerfacecolor='none',
                 label='near-inertial waves'),
          Line2D([], [], color=C_FLOW, lw=2.2, marker='o', ms=6, markerfacecolor='white',
                 label='subinertial flow'),
          Line2D([], [], color=C_ISO, lw=0.8, label='isopycnals'),
          Line2D([], [], color=C_ML, lw=0.9, label='mixed-layer base'),
          Line2D([], [], color=C_TRACK, lw=1.0, ls=(0, (4, 2)), label='storm track')]
    fig.legend(handles=hs, loc='lower center', bbox_to_anchor=(0.5, 0.005), ncol=4, frameon=False,
               fontsize=FS['small'], handlelength=2.4, columnspacing=1.4, handletextpad=0.6)

    fig.savefig(OUT_STEM + '.png', dpi=DPI, facecolor='white')
    fig.savefig(OUT_STEM + '.pdf', facecolor='white')
    print(f'saved {OUT_STEM}.png and .pdf')


if __name__ == '__main__':
    main()
