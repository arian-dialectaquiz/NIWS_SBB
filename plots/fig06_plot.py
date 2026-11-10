"""
fig06_plot.py

Figure 6, the two kinetic energy budgets, drawn from fig06/fig06_data.nc alone.

(a)  near-inertial kinetic energy budget of the South sector per window, every
     term integrated over the window (J m-2) on a scale linear within
     LINTHRESH and logarithmic beyond, sources upward and sinks downward, the
     residual hatched and the tendency as a marker
(b)  the subinertial kinetic energy budget of the South sector on the same scale
(c)  alpha_Pi per sector and stage, the estimate as a tick inside a band that
     spans the 95 % block-bootstrap interval, unresolved sectors hatched
(d)  subinertial kinetic energy per sector through the record, T2 shaded
(e, f)  depth-integrated near-inertial energy transport over the southern slope
     in T2 and T3, its divergence shaded (sink of the local budget) and the
     transport as arrows, one colourbar

Colours and hatches per term from style_v3.TERM, no comma in any figure text,
legends above the panels.

    python fig06_plot.py
"""

import json
import os

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import uniform_filter

import style_v3 as S

S.apply()

# =====================================================================
# CONFIGURATION
# =====================================================================

FIG_DIR = os.path.join(os.getcwd(), 'fig06')
IN_NC = os.path.join(FIG_DIR, 'fig06_data.nc')
OUT_STEM = os.path.join(FIG_DIR, 'fig06_budgets')

FIGSIZE = (S.PAGE_W_IN, 9.8)
SECTOR = 'South'
BUDGET_WINDOWS = ('T1', 'T2', 'T3', 'T4')
STAGES = ('T2', 'T3', 'T4')
ALPHA_SECTORS = ('South', 'Central', 'North', 'Domain')
LINTHRESH = 100.0                     # J m-2, linear part of the symlog axis
# near-inertial budget, each term with its sign as a source of K, the label, colour and hatch
NI_BARS = [('W_NI', +1, r'$W_{NI}$', 'steelblue', ''), ('PiH', +1, r'$\Pi_H$', 'darkorange', ''),
           ('PiV', +1, r'$\Pi_V$', 'firebrick', ''), ('wb', +1, r"$\langle w'b'\rangle$", 'forestgreen', ''),
           ('div_adv', -1, 'advective transport', 'peru', '//'), ('div_rad', -1, 'radiative transport', 'teal', '//'),
           ('eps_NI', -1, r'$\varepsilon_{NI}$', 'darkorchid', '..'), ('drag_NI', -1, 'bottom drag', 'saddlebrown', '..'),
           ('R', +1, 'residual', 'lightgrey', 'xx')]
SUB_BARS = [('W_S', +1, r'$W_S$', 'steelblue', ''), ('Pi_b', +1, r'$-\Pi$', 'black', ''),
            ('div_ws', -1, 'wave-stress work transport', 'teal', '//'), ('drag_b', -1, 'bottom drag', 'saddlebrown', '..'),
            ('R_b', +1, 'residual', 'lightgrey', 'xx')]
ZOOM = dict(lon=(-49.8, -46.0), lat=(-31.2, -28.0))
DIV_SCALE = None                      # mW m-2, symmetric limit of (e) and (f); None takes the 98th percentile
DIV_SMOOTH_CELLS = 5                  # display boxcar on the divergence, cells of the compute grid
FLUX_DECIM_KM = 24                    # arrows every this many km
FLUX_KEY = 1000.0                     # W m-1
FLUX_SCALE = 15000.0                  # W m-1 per axes width (larger gives shorter arrows)
CBAR_LABEL = 'transport divergence (mW m$^{-2}$)'
CBAR_FONT = S.FS['label'] + 1
CMAP_DIV = plt.get_cmap('RdBu_r')
ISOBATHS_MAPS = [iso for iso in S.ISOBATHS if iso[0] in (200, 1000)]

# =====================================================================
# LOAD
# =====================================================================

ds = xr.open_dataset(IN_NC)
ST = json.loads(ds.attrs['stats'])
B = ST['budgets']
ALPHA = ST['alpha']
WINDOWS = {k: tuple(ds.attrs[k]) for k in ('T1', 'T2', 'T3', 'T4') if k in ds.attrs}
LANDFALL = np.datetime64(str(ds.attrs['landfall']))
days = ds['day'].values
sectors = [str(s) for s in ds['sector_name'].values]
lon, lat = ds['lon'].values, ds['lat'].values
h, mask = ds['h'].values, ds['mask'].values
land = mask < 0.5
stride = int(ds.attrs['stride'])


def legend_above(ax, handles, ncol):
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=ncol, borderaxespad=0.0,
              handlelength=1.6, fontsize=S.FS['small'], frameon=False, columnspacing=1.0)


# =====================================================================
# PANELS
# =====================================================================

def panel_budget(ax, bars, tag, ylabel, tend_key):
    n = len(bars)
    width = 0.8 / n
    x = np.arange(len(BUDGET_WINDOWS))
    hs = []
    for i, (key, sign, label, colour, hatch) in enumerate(bars):
        vals = np.array([sign * B[SECTOR][k][key] for k in BUDGET_WINDOWS])
        ax.bar(x - 0.4 + (i + 0.5) * width, vals, width, color=colour, hatch=hatch, edgecolor='black',
               linewidth=0.4, zorder=3)
        hs.append(Patch(facecolor=colour, hatch=hatch, edgecolor='black', linewidth=0.4, label=label))
    tend = np.array([B[SECTOR][k][tend_key] for k in BUDGET_WINDOWS])
    ax.plot(x, tend, ls='none', marker='D', ms=5, color='black', markerfacecolor='white', zorder=5)
    hs.append(Line2D([], [], ls='none', marker='D', ms=5, color='black', markerfacecolor='white', label='tendency'))
    ax.axhline(0.0, color='0.3', lw=0.6, zorder=2)
    ax.set_yscale('symlog', linthresh=LINTHRESH, linscale=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'${k}$' for k in BUDGET_WINDOWS])
    ax.set_ylabel(ylabel)
    ax.grid(True, axis='y', ls=':', lw=0.4, alpha=0.5, zorder=0)
    ax.text(0.01, 0.03, 'sinks', transform=ax.transAxes, fontsize=S.FS['small'], color='dimgrey', ha='left', va='bottom')
    ax.text(0.01, 0.97, 'sources', transform=ax.transAxes, fontsize=S.FS['small'], color='dimgrey', ha='left', va='top')
    legend_above(ax, hs, 5 if n > 5 else 3)
    S.panel_tag(ax, tag, loc='upper right')


def panel_alpha(ax):
    shown = [sec for sec in ALPHA_SECTORS if ALPHA[sec]['resolved']]
    unres = [sec for sec in ALPHA_SECTORS if not ALPHA[sec]['resolved']]
    x0 = np.arange(len(shown))
    width = 0.8 / len(STAGES)
    hs = []
    for i, k in enumerate(STAGES):
        sp = S.WINDOW[k]
        for j, sec in enumerate(shown):
            a = ALPHA[sec]
            xc = x0[j] - 0.4 + (i + 0.5) * width
            lo, hi, est = a[k]['lo'], a[k]['hi'], a[k]['alpha']
            ax.bar(xc, hi - lo, width * 0.9, bottom=lo, color=sp['color'], alpha=0.3,
                   edgecolor=sp['color'], linewidth=0.5, zorder=3)
            ax.plot([xc - 0.45 * width, xc + 0.45 * width], [est, est], color=sp['color'], lw=1.8, zorder=5)
        hs.append(Patch(facecolor=sp['color'], alpha=0.3, edgecolor=sp['color'], label=f'${k}$'))

    ax.axhline(0.0, color='0.3', lw=0.6, zorder=2)
    ax.set_xticks(x0)
    ax.set_xticklabels(shown)
    ax.set_xlim(-0.6, len(shown) - 0.4)
    ax.set_ylabel(r'$\alpha_\Pi$')
    ax.grid(True, axis='y', ls=':', lw=0.4, alpha=0.5, zorder=0)
    legend_above(ax, hs, 3)
    S.panel_tag(ax, 'c', loc='upper right')


def panel_kb(ax):
    for k in ('T1', 'T2', 'T3', 'T4'):
        if k in WINDOWS:
            a, b = WINDOWS[k]
            ax.axvspan(np.datetime64(a), np.datetime64(b) + np.timedelta64(1, 'D'), color=S.WINDOW[k]['shade'],
                       alpha=0.9, lw=0, zorder=0)
            ax.text(np.datetime64(a) + np.timedelta64(3, 'D') + np.timedelta64(12, 'h'), 0.03, f'${k}$',
                    transform=ax.get_xaxis_transform(), fontsize=S.FS['small'], color=S.WINDOW[k]['color'],
                    ha='center', va='bottom', zorder=1)
    ax.axvline(LANDFALL, color='black', ls=(0, (4, 2)), lw=0.9, zorder=2)
    x = days + np.timedelta64(12, 'h')
    ser = ds['Kb_series'].values * 1e-3
    hs = []
    for i, name in enumerate(sectors):
        if name == 'Domain':
            continue
        sp = S.SECTOR[name]
        ax.plot(x, ser[i], color=sp['color'], ls=sp['ls'], lw=1.6, zorder=3)
        hs.append(S.line_handle(sp, name, marker=None))
    ax.set_ylabel(r'$\rho_0 \int \bar{K}\,\mathrm{d}z$ (kJ m$^{-2}$)')
    ax.set_xlim(days[0], days[-1] + np.timedelta64(1, 'D'))
    ax.grid(True, ls=':', lw=0.4, alpha=0.5, zorder=0)
    S.date_axis(ax, interval=8, label='2004')
    legend_above(ax, hs, 3)
    S.panel_tag(ax, 'd', loc='upper right')


def panel_transport(fig, ax, k, tag, left_labels, scale):
    inz = ((lon >= ZOOM['lon'][0]) & (lon <= ZOOM['lon'][1]) & (lat >= ZOOM['lat'][0]) & (lat <= ZOOM['lat'][1]))
    rows, cols = np.where(inz.any(axis=1))[0], np.where(inz.any(axis=0))[0]
    zs = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    lo_, la_, hz, mz = lon[zs], lat[zs], h[zs], mask[zs]
    lz = mz < 0.5
    div = (ds[f'div_adv_{k}'].values + ds[f'div_rad_{k}'].values)[zs] * 1e3
    valid = np.isfinite(div).astype(float)
    num = uniform_filter(np.nan_to_num(div, nan=0.0), DIV_SMOOTH_CELLS, mode='nearest')
    den = uniform_filter(valid, DIV_SMOOTH_CELLS, mode='nearest')
    div = np.where(den > 0.5, num / np.maximum(den, 1e-12), np.nan)
    div[lz] = np.nan
    Fx, Fy = ds[f'Fx_{k}'].values[zs], ds[f'Fy_{k}'].values[zs]
    tf = S._tf(ax)
    norm = mcolors.TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale)
    pm = ax.pcolormesh(lo_, la_, div, cmap=CMAP_DIV, norm=norm, shading='auto', zorder=0, **tf)
    q = max(int(round(FLUX_DECIM_KM / stride)), 1)
    qv = ax.quiver(lo_[::q, ::q], la_[::q, ::q], Fx[::q, ::q], Fy[::q, ::q], color='black', scale=FLUX_SCALE,
                   scale_units='width', width=0.0022, headwidth=4.0, headlength=4.5, zorder=6, **tf)
    S.basemap(ax, lo_, la_, mz, ZOOM, left_labels=left_labels)
    S.isobaths(ax, lo_, la_, np.where(lz, np.nan, hz), which=ISOBATHS_MAPS, lw=0.7)
    ax.text(0.03, 0.9, f'${k}$', transform=ax.transAxes, fontsize=S.FS['label'] + 1, fontweight='bold', ha='left',
            va='top', zorder=12)
    if left_labels:
        # the key above the panel, outside the data
        ax.quiverkey(qv, 0.08, 1.04, FLUX_KEY, f'{FLUX_KEY:g} W m$^{{-1}}$', labelpos='E', coordinates='axes',
                     fontproperties={'size': S.FS['small']}, zorder=20)
    S.panel_tag(ax, tag)
    return pm


def write_numbers():
    L = ['FIGURE 6 PLOT NUMBERS, South, J m-2 per window']
    for k in BUDGET_WINDOWS:
        r = B[SECTOR][k]
        L.append(f'{k}: tend {r["tend"]:+.0f} W_NI {r["W_NI"]:+.0f} PiH {r["PiH"]:+.0f} PiV {r["PiV"]:+.0f} wb {r["wb"]:+.0f} '
                 f'eps {r["eps_NI"]:+.0f} drag {r["drag_NI"]:+.0f} div_adv {r["div_adv"]:+.0f} div_rad {r["div_rad"]:+.0f} R {r["R"]:+.0f}')
    with open(os.path.join(FIG_DIR, 'fig06_plot_numbers.txt'), 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print('\n'.join(L))


# =====================================================================
# FIGURE
# =====================================================================

def main():
    global DIV_SCALE
    if DIV_SCALE is None:
        allv = np.concatenate([np.abs(v[np.isfinite(v)]) for v in
                               ((ds['div_adv_T2'].values + ds['div_rad_T2'].values) * 1e3,
                                (ds['div_adv_T3'].values + ds['div_rad_T3'].values) * 1e3)])
        q = float(np.quantile(allv, 0.98))
        DIV_SCALE = float(np.round(q, 1 - int(np.floor(np.log10(max(q, 1e-9))))))
    fig = plt.figure(figsize=FIGSIZE)
    g1 = GridSpec(1, 2, figure=fig, left=0.085, right=0.985, top=0.945, bottom=0.73, wspace=0.28, width_ratios=[1.7, 1.0])
    g2 = GridSpec(1, 2, figure=fig, left=0.085, right=0.985, top=0.625, bottom=0.41, wspace=0.28, width_ratios=[1.7, 1.0])
    g3 = GridSpec(1, 2, figure=fig, left=0.085, right=0.90, top=0.335, bottom=0.04, wspace=0.08)
    ax_a = fig.add_subplot(g1[0, 0])
    ax_c = fig.add_subplot(g1[0, 1])
    ax_b = fig.add_subplot(g2[0, 0], sharey=ax_a)
    ax_d = fig.add_subplot(g2[0, 1])
    panel_budget(ax_a, NI_BARS, 'a', r'near-inertial budget (J m$^{-2}$)', 'tend')
    panel_budget(ax_b, SUB_BARS, 'b', r'subinertial budget (J m$^{-2}$)', 'tend_b')
    panel_alpha(ax_c)
    panel_kb(ax_d)
    ax_e = S.map_axes(fig, g3[0, 0])
    ax_f = S.map_axes(fig, g3[0, 1])
    pm = panel_transport(fig, ax_e, 'T2', 'e', True, DIV_SCALE)
    panel_transport(fig, ax_f, 'T3', 'f', False, DIV_SCALE)
    ax_f.apply_aspect()
    t = DIV_SCALE
    cb = S.colorbar_outside(fig, ax_f, pm, CBAR_LABEL, extend='both', where='right', size=0.05, pad=0.04,
                            ticks=[-t, -t / 2, 0, t / 2, t], fmt=lambda v, _: f'{v:g}')
    cb.set_label(CBAR_LABEL, size=CBAR_FONT)
    cb.ax.tick_params(labelsize=S.FS['tick'])
    write_numbers()
    S.save(fig, OUT_STEM)


if __name__ == '__main__':
    main()