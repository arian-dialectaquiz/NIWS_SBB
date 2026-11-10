"""
fig07_plot.py

Figure 7, the wake decay and the kinetic energy pathway of the South sector,
drawn from fig07/fig07_data.nc (and the Figure 2 and 3 daily series it was
built from).

(a)  daily rho0 int K dz to the end of the record (log axis) with the
     exponential fits from the peak through T3 and through T4 (dashed, the
     e-folding stated), and the daily exchange rho0 int Pi dz on the right axis
(b)  cumulative near-inertial wind work, exchange, shear dissipation, bottom drag,
     lateral transport, change of K and residual from the start of T2 to the
     end of the record (kJ m-2), one line per term with its colour and style
(c)  the two-stage pathway as a schematic drawn in code, with the window
     magnitudes of Figure 6 in kJ m-2 (storm window and wake window side by
     side), arrows in the colours of (b)

    python fig07_plot.py
"""

import json
import os

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import style_v3 as S

S.apply()

# =====================================================================
# CONFIGURATION
# =====================================================================

FIG_DIR = os.path.join(os.getcwd(), 'fig07')
IN_NC = os.path.join(FIG_DIR, 'fig07_data.nc')
FIG02_NC = os.path.join(os.getcwd(), 'fig02', 'fig02_data.nc')
FIG03_NC = os.path.join(os.getcwd(), 'fig03', 'fig03_data.nc')
OUT_STEM = os.path.join(FIG_DIR, 'fig07_pathway')

FIGSIZE = (S.PAGE_W_IN, 9.0)
SECTOR = 'South'
STAGES = ('T2', 'T3')                 # the two columns of the schematic
LINES = [('W_NI', +1, r'$W_{NI}$', 'steelblue', 'solid', 1.8),
         ('Pi', +1, r'$\Pi$', 'black', 'solid', 1.8),
         ('eps_NI', -1, r'$-\varepsilon_{NI}$', 'darkorchid', 'dashdot', 1.6),
         ('drag_NI', -1, 'bottom drag', 'saddlebrown', 'dashdot', 1.4),
         ('transport', -1, 'lateral transport', 'teal', 'dashed', 1.6),
         ('wb', +1, r"$\langle w'b'\rangle$", 'forestgreen', 'dotted', 1.4),
         ('dK', +1, r'change of $K$', 'dimgrey', 'solid', 1.2),
         ('R', +1, 'residual', 'lightgrey', 'dotted', 1.6)]
PI_COLOUR = 'black'
NOTE_COLOUR = 'black'                 # the change of K, the residual and the unit in (c)
NOTE_SIZE = S.FS['label']
WINDOW_LABEL_Y = 1.012                # window names just above the top spine, outside the data
LEGEND_Y = 1.10                       # legends sit above the window names
FIT_COLOUR = {'T3': 'royalblue', 'T4': 'darkorchid'}

# =====================================================================
# LOAD
# =====================================================================

ds = xr.open_dataset(IN_NC)
FITS = json.loads(ds.attrs['fits'])
PATH = json.loads(ds.attrs['pathway'])
FATE = json.loads(ds.attrs['fate'])
WINDOWS = {k: tuple(ds.attrs[k]) for k in ('T1', 'T2', 'T3', 'T4') if k in ds.attrs}
LANDFALL = np.datetime64(str(ds.attrs['landfall']))
edges = ds['edge'].values
with xr.open_dataset(FIG02_NC) as d2:
    secs = [str(x) for x in d2['sector_name'].values]
    K_full = d2['K_series'].values[secs.index(SECTOR)]
    kdays = d2['day'].values
with xr.open_dataset(FIG03_NC) as d3:
    secs = [str(x) for x in d3['sector_name'].values]
    Pi_full = d3['Pi_series'].values[secs.index(SECTOR)]
    pdays = d3['day'].values


def shade_windows(ax, label_y=WINDOW_LABEL_Y, va='bottom'):
    for k in ('T1', 'T2', 'T3', 'T4'):
        if k in WINDOWS:
            a, b = WINDOWS[k]
            ax.axvspan(np.datetime64(a), np.datetime64(b) + np.timedelta64(1, 'D'), color=S.WINDOW[k]['shade'],
                       alpha=0.9, lw=0, zorder=0)
            ax.text(np.datetime64(a) + np.timedelta64(3, 'D') + np.timedelta64(12, 'h'), label_y, f'${k}$',
                    transform=ax.get_xaxis_transform(), fontsize=S.FS['small'], color=S.WINDOW[k]['color'],
                    fontweight='bold', ha='center', va=va, zorder=1, clip_on=False)
    ax.axvline(LANDFALL, color='black', ls=(0, (4, 2)), lw=0.9, zorder=2)


def legend_above(ax, handles, ncol):
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, LEGEND_Y), ncol=ncol, borderaxespad=0.0,
              handlelength=2.4, fontsize=S.FS['small'], frameon=False, columnspacing=1.2)


# =====================================================================
# PANELS
# =====================================================================

def panel_a(ax):
    shade_windows(ax)
    x = kdays + np.timedelta64(12, 'h')
    sp = S.SECTOR[SECTOR]
    ax.plot(x, K_full, color=sp['color'], ls='solid', marker='o', ms=3.2, lw=1.6, zorder=4)
    hs = [Line2D([], [], color=sp['color'], ls='solid', marker='o', ms=3.2, lw=1.6, label=r'$\rho_0 \int K\,\mathrm{d}z$')]
    kd = kdays.astype('datetime64[D]')
    ip = int(np.nanargmax(K_full))
    for name in ('T3', 'T4'):
        ie = int(np.where(kd <= np.datetime64(WINDOWS[name][1]))[0][-1])
        tau = FITS[f'efold_{name}_d']
        tt = np.arange(ie - ip + 1, dtype=float)
        y = K_full[ip] * np.exp(-tt / tau)
        ax.plot(x[ip:ie + 1], y, color=FIT_COLOUR[name], ls='dashed', lw=1.3, zorder=5)
        hs.append(Line2D([], [], color=FIT_COLOUR[name], ls='dashed', lw=1.3,
                         label=f'fit to the end of ${name}$ ({tau:.1f} d)'))
    ax.set_yscale('log')
    ax.set_ylabel(r'$\rho_0 \int K\,\mathrm{d}z$ (J m$^{-2}$)')
    ax.set_xlim(kdays[0], kdays[-1] + np.timedelta64(1, 'D'))
    ax.grid(True, ls=':', lw=0.4, alpha=0.5, zorder=0)
    ax2 = ax.twinx()
    ax2.plot(pdays + np.timedelta64(12, 'h'), Pi_full * 1e3, color=PI_COLOUR, ls='dotted', lw=1.6, zorder=3)
    ax2.axhline(0.0, color=PI_COLOUR, lw=0.5, alpha=0.6, zorder=1)
    ax2.set_ylabel(r'$\rho_0 \int \Pi\,\mathrm{d}z$ (mW m$^{-2}$)')
    lim = float(np.nanmax(np.abs(Pi_full))) * 1e3 * 1.1
    ax2.set_ylim(-lim, lim)
    hs.append(Line2D([], [], color=PI_COLOUR, ls='dotted', lw=1.6, label=r'$\rho_0 \int \Pi\,\mathrm{d}z$ (right)'))
    plt.setp(ax.get_xticklabels(), visible=False)
    legend_above(ax, hs, 4)
    S.panel_tag(ax, 'a', loc='upper right')


def panel_b(ax):
    shade_windows(ax)
    hs = []
    for key, sign, label, colour, ls, lw in LINES:
        if key == 'Pi':
            y = ds['cum_PiH'].values + ds['cum_PiV'].values
        elif key == 'transport':
            y = ds['cum_div_adv'].values + ds['cum_div_rad'].values
        else:
            y = ds[f'cum_{key}'].values
        ax.plot(edges, sign * y * 1e-3, color=colour, ls=ls, lw=lw, zorder=3)
        hs.append(Line2D([], [], color=colour, ls=ls, lw=lw, label=label))
    ax.axhline(0.0, color='0.3', lw=0.6, zorder=2)
    ax.set_ylabel(r'cumulative (kJ m$^{-2}$)')
    ax.set_xlim(kdays[0], kdays[-1] + np.timedelta64(1, 'D'))
    ax.grid(True, ls=':', lw=0.4, alpha=0.5, zorder=0)
    S.date_axis(ax, interval=4, label='2004')
    legend_above(ax, hs, 4)
    S.panel_tag(ax, 'b', loc='upper left')


def box(ax, x, y, w, h, text, fc, ec='black', fs=None):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.01,rounding_size=0.02',
                                fc=fc, ec=ec, lw=0.8, zorder=3))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs or S.FS['small'], zorder=4)


def arrow(ax, p0, p1, colour, value, label_dx=0.0, label_dy=0.0, rad=0.0, lw_scale=1.0):
    lw = 0.8 + 2.2 * min(abs(value), 12.0) ** 0.5 / 12.0 ** 0.5 * lw_scale
    a = FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=12, color=colour, lw=lw,
                        connectionstyle=f'arc3,rad={rad}', zorder=5)
    ax.add_patch(a)
    xm, ym = 0.5 * (p0[0] + p1[0]) + label_dx, 0.5 * (p0[1] + p1[1]) + label_dy
    ax.text(xm, ym, f'{value:+.2f}' if abs(value) < 1 else f'{value:+.1f}', ha='center', va='center',
            fontsize=S.FS['small'], color=colour, zorder=6,
            bbox=dict(fc='white', ec='none', alpha=0.8, pad=0.8))


def panel_c(ax):
    """Two columns, the storm window and the wake window, same box layout."""
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.axis('off')
    for i, k in enumerate(STAGES):
        if k not in PATH:
            continue
        r = PATH[k]
        x0 = i + 0.5
        ax.text(x0, 0.97, f'${k}$ ' + ('(the storm)' if k == 'T2' else '(the wake)'), ha='center', va='top',
                fontsize=S.FS['label'], fontweight='bold', color=S.WINDOW[k]['color'])
        # boxes
        box(ax, x0, 0.80, 0.42, 0.10, 'wind work', 'aliceblue')
        box(ax, x0, 0.50, 0.50, 0.13, 'near-inertial\nkinetic energy $K$', 'mistyrose')
        box(ax, x0 + 0.32, 0.22, 0.24, 0.11, 'subinertial\nkinetic energy', 'lavender')
        box(ax, x0 - 0.32, 0.22, 0.29, 0.11, 'dissipation\nand bottom drag', 'thistle')
        box(ax, x0, 0.08, 0.34, 0.09, 'lateral export', 'azure')
        # arrows, values in kJ m-2, positive along the arrow
        arrow(ax, (x0, 0.75), (x0, 0.565), 'steelblue', r['W_NI'], label_dx=0.14)
        pi = r['Pi']
        if pi < 0:      # K feeds the subinertial flow
            arrow(ax, (x0 + 0.12, 0.44), (x0 + 0.27, 0.28), 'black', -pi, label_dx=0.13, label_dy=0.04)
        else:           # the subinertial flow feeds K
            arrow(ax, (x0 + 0.27, 0.28), (x0 + 0.12, 0.44), 'black', pi, label_dx=0.13, label_dy=0.04)
        arrow(ax, (x0 - 0.12, 0.44), (x0 - 0.27, 0.28), 'darkorchid', r['eps'] + r['drag'], label_dx=-0.13, label_dy=0.04)
        tr = r['transport']
        if tr >= 0:
            arrow(ax, (x0, 0.435), (x0, 0.125), 'teal', tr, label_dx=0.12)
        else:
            arrow(ax, (x0, 0.125), (x0, 0.435), 'teal', -tr, label_dx=0.12)
        ax.text(x0 - 0.47, 0.60, f'change of $K$ {r["tend"]:+.2f}\nresidual {r["R"]:+.2f}', ha='left', va='bottom',
                fontsize=NOTE_SIZE, color=NOTE_COLOUR)
    ax.text(1.0, 0.0, 'kJ m$^{-2}$ per window', ha='center', va='bottom', fontsize=NOTE_SIZE, color=NOTE_COLOUR)
    ax.plot([1.0, 1.0], [0.02, 0.95], color='lightgrey', lw=0.8)
    S.panel_tag(ax, 'c', loc='upper left')


def write_numbers():
    L = ['FIGURE 7 PLOT NUMBERS', json.dumps(FITS), json.dumps(FATE)]
    with open(os.path.join(FIG_DIR, 'fig07_plot_numbers.txt'), 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print('\n'.join(L))


# =====================================================================
# FIGURE
# =====================================================================

def main():
    fig = plt.figure(figsize=FIGSIZE)
    g1 = GridSpec(2, 1, figure=fig, left=0.10, right=0.90, top=0.935, bottom=0.42, hspace=0.42)
    g2 = GridSpec(1, 1, figure=fig, left=0.04, right=0.98, top=0.33, bottom=0.02)
    ax_a = fig.add_subplot(g1[0, 0])
    ax_b = fig.add_subplot(g1[1, 0], sharex=ax_a)
    ax_c = fig.add_subplot(g2[0, 0])
    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)
    write_numbers()
    S.save(fig, OUT_STEM)


if __name__ == '__main__':
    main()