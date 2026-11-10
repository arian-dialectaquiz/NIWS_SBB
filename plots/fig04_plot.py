"""
fig04_plot.py

Figure 4, where the exchange occurs in the subinertial flow, drawn from
fig04/fig04_data.nc alone.

Top row, three small panels on one line
(a)  depth-integrated rho0 Pi of the South sector binned on OW / sigma_OW for
     T1 to T3, 95 % block-bootstrap bands
(b)  the same binned on f_eff / f = 1 + zeta / 2f, anticyclonic left of 1
(c)  the transfer per unit of near-inertial kinetic energy lambda binned on
     f_eff / f for T2 and T3 (y axis on the right)
Bottom, the large panel
(d)  the T3 exchange over the southern slope (3-cell display boxcar and the
     Figure 3 scale) with the T3 subinertial velocity of the upper 100 m as
     arrows, the contours zeta / f = +-0.15 and the 1000 m isobath. The panel
     letter, the window stamp and the arrow key sit on the continent, the
     contour key above the panel and the colourbar to its right.

Window colours and line styles follow style_v3.WINDOW, the bands take the line
colour at low alpha. One legend above the top row, no comma in any figure text.

    python fig04_plot.py
"""

import json
import os

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import uniform_filter

import style_v3 as S

S.apply()

# =====================================================================
# CONFIGURATION
# =====================================================================

FIG_DIR = os.path.join(os.getcwd(), 'fig04')
IN_NC = os.path.join(FIG_DIR, 'fig04_data.nc')
OUT_STEM = os.path.join(FIG_DIR, 'fig04_topology')

FIGSIZE = (S.PAGE_W_IN, 8.6)
CMASK = 'South'
CURVE_WINDOWS = ('T1', 'T2', 'T3')
LAMBDA_WINDOWS = ('T2', 'T3')
SERIES_UNIT = 1e3                     # W m-2 to mW m-2
LAMBDA_UNIT = 86400.0                 # s-1 to d-1
BAND_ALPHA = 0.18
AXIS_COLOUR = 'dimgrey'

# panel (d)
PI_SCALE = 5.0                        # mW m-2, the Figure 3 scale
PI_SMOOTH_CELLS = 3                   # the Figure 3 display boxcar
CMAP_PI = plt.get_cmap('RdBu_r')
ZOF_CONTOUR = 0.15
ZOF_SMOOTH_CELLS = 15                 # boxcar on zeta / f before contouring, cells
ZOF_COLOUR = 'black'
ISOBATH_D = (1000, 'grey', 'solid')   # the one isobath of (d)
QUIVER_DECIM = 22                     # cells between arrows
QUIVER_SCALE = 6.0                    # m s-1 per axes width
QUIVER_COLOUR = 'black'
QUIVER_KEY = 0.5                      # m s-1
# positions on the continent, in axes fractions of (d); the coast runs from about
# (0.31, 1.0) through (0.24, 0.85) to (0.11, 0.69) in the zoom box, so everything
# left of that line is land; move these two if a label touches the sea
STAMP_XY = (0.02, 0.915)              # the T3 stamp under the panel letter
QKEY_XY = (0.10, 0.80)                # centre of the key arrow, label above it

# =====================================================================
# LOAD
# =====================================================================

ds = xr.open_dataset(IN_NC)
ST = json.loads(ds.attrs['stats'])
WKEYS = [str(w) for w in ds['window'].values]
ROUTES = [str(r) for r in ds['route'].values]
MNAMES = [str(m) for m in ds['cmask'].values]
im = MNAMES.index(CMASK)
cen_ow, cen_f = ds['bin_ow'].values, ds['bin_f'].values
ZOOM_LON, ZOOM_LAT = tuple(ds.attrs['zoom_lon']), tuple(ds.attrs['zoom_lat'])
ZK = str(ds.attrs['zoom_window'])


def curve(kind, route, win):
    ir, iw = ROUTES.index(route), WKEYS.index(win)
    return (ds[f'C_{kind}'].values[ir, iw, im], ds[f'L_{kind}'].values[ir, iw, im], ds[f'H_{kind}'].values[ir, iw, im])


def window_lines(ax, x, kind, windows, scale, lw=1.6):
    for k in windows:
        sp = S.WINDOW[k]
        c, lo, hi = curve(kind, 'Pi', k)
        ax.fill_between(x, lo * scale, hi * scale, color=sp['color'], alpha=BAND_ALPHA, lw=0, zorder=1)
        ax.plot(x, c * scale, color=sp['color'], ls=sp['ls'], lw=lw, zorder=3)


def end_words(fig, ax, left, right):
    """Two words at the ends of the x-label line, outside the data area."""
    tr = mtransforms.offset_copy(ax.transAxes, fig=fig, y=-17, units='points')
    ax.text(0.0, 0.0, left, transform=tr, ha='left', va='top', fontsize=S.FS['small'], color=AXIS_COLOUR)
    ax.text(1.0, 0.0, right, transform=tr, ha='right', va='top', fontsize=S.FS['small'], color=AXIS_COLOUR)


def line_furniture(ax, xzero):
    ax.axhline(0.0, color=AXIS_COLOUR, lw=0.6, zorder=2)
    ax.axvline(xzero, color=AXIS_COLOUR, lw=0.6, ls='dotted', zorder=2)
    ax.grid(True, ls=':', lw=0.4, alpha=0.5, zorder=0)


def feff_axis(ax):
    ax.set_xlim(cen_f[0] - 0.01, cen_f[-1] + 0.01)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set_xlabel(r'$f_{\mathrm{eff}} / f$')


# =====================================================================
# PANELS
# =====================================================================

def panel_a(fig, ax):
    window_lines(ax, cen_ow, 'ow', CURVE_WINDOWS, SERIES_UNIT)
    line_furniture(ax, 0.0)
    ax.set_xlim(cen_ow[0] - 0.25, cen_ow[-1] + 0.25)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2.0))
    ax.set_xlabel(r'$\mathrm{OW} / \sigma_{\mathrm{OW}}$')
    ax.set_ylabel(r'$\rho_0 \int \Pi\,\mathrm{d}z$ (mW m$^{-2}$)')
    end_words(fig, ax, 'vorticity', 'strain')
    hs = [Line2D([], [], color=S.WINDOW[k]['color'], ls=S.WINDOW[k]['ls'], lw=1.6, label=f'${k}$')
          for k in CURVE_WINDOWS]
    hs.append(Patch(facecolor='grey', alpha=0.3, edgecolor='none', label='95 % interval'))
    ax.legend(handles=hs, loc='lower left', bbox_to_anchor=(0.0, 1.03), ncol=4, borderaxespad=0.0,
              handlelength=2.4, fontsize=S.FS['small'], frameon=False)
    S.panel_tag(ax, 'a')


def panel_b(fig, ax):
    window_lines(ax, cen_f, 'f', CURVE_WINDOWS, SERIES_UNIT)
    line_furniture(ax, 1.0)
    feff_axis(ax)
    plt.setp(ax.get_yticklabels(), visible=False)
    end_words(fig, ax, 'anticyclonic', 'cyclonic')
    S.panel_tag(ax, 'b')


def panel_c(fig, ax):
    for k in LAMBDA_WINDOWS:
        iw = WKEYS.index(k)
        sp = S.WINDOW[k]
        c = ds['lam'].values[iw, im] * LAMBDA_UNIT
        lo = ds['lam_lo'].values[iw, im] * LAMBDA_UNIT
        hi = ds['lam_hi'].values[iw, im] * LAMBDA_UNIT
        ax.fill_between(cen_f, lo, hi, color=sp['color'], alpha=BAND_ALPHA, lw=0, zorder=1)
        ax.plot(cen_f, c, color=sp['color'], ls=sp['ls'], lw=1.6, zorder=3)
    line_furniture(ax, 1.0)
    feff_axis(ax)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax.set_ylabel(r'$\lambda = \Pi / K$ (d$^{-1}$)')
    S.panel_tag(ax, 'c')


def panel_d(fig, ax):
    lon, lat = ds['zoom_lon'].values, ds['zoom_lat'].values
    h, mask = ds['zoom_h'].values, ds['zoom_mask'].values
    land = mask < 0.5
    raw = np.where(land, np.nan, ds['zoom_Pi'].values * SERIES_UNIT)
    valid = np.isfinite(raw).astype(float)
    num = uniform_filter(np.nan_to_num(raw, nan=0.0), PI_SMOOTH_CELLS, mode='nearest')
    den = uniform_filter(valid, PI_SMOOTH_CELLS, mode='nearest')
    pi = np.where(den > 0.5, num / np.maximum(den, 1e-12), np.nan)
    pi[land] = np.nan
    zof = ds['zoom_zof'].values
    ue, vn = ds['zoom_ue'].values, ds['zoom_vn'].values
    tf = S._tf(ax)

    norm = mcolors.TwoSlopeNorm(vmin=-PI_SCALE, vcenter=0.0, vmax=PI_SCALE)
    pm = ax.pcolormesh(lon, lat, pi, cmap=CMAP_PI, norm=norm, shading='auto', zorder=0, **tf)
    zs = uniform_filter(np.nan_to_num(zof, nan=0.0), ZOF_SMOOTH_CELLS, mode='nearest')
    zs[land] = np.nan
    ax.contour(lon, lat, zs, levels=[-ZOF_CONTOUR], colors=ZOF_COLOUR, linestyles='dashed', linewidths=0.9,
               zorder=3, **tf)
    ax.contour(lon, lat, zs, levels=[ZOF_CONTOUR], colors=ZOF_COLOUR, linestyles='solid', linewidths=0.9,
               zorder=3, **tf)
    q = QUIVER_DECIM
    qv = ax.quiver(lon[::q, ::q], lat[::q, ::q], ue[::q, ::q], vn[::q, ::q], color=QUIVER_COLOUR,
                   scale=QUIVER_SCALE, scale_units='width', width=0.0028, headwidth=3.8, headlength=4.2,
                   alpha=0.85, zorder=6, **tf)
    S.basemap(ax, lon, lat, mask, dict(lon=ZOOM_LON, lat=ZOOM_LAT), left_labels=True)
    S.isobaths(ax, lon, lat, np.where(land, np.nan, h), which=[ISOBATH_D], lw=0.8)

    ax.apply_aspect()                       # the map box takes its final size before the bar is placed
    ext = S.extend_for(pi, -PI_SCALE, PI_SCALE)
    S.colorbar_outside(fig, ax, pm, r'$\rho_0 \int \Pi\,\mathrm{d}z$ (mW m$^{-2}$)', extend=ext,
                       where='right', size=0.035, pad=0.03, length=0.85,
                       ticks=[-5, -2.5, 0, 2.5, 5], fmt=lambda v, _: f'{v:g}')

    # on the continent, the letter, the window and the arrow key
    S.panel_tag(ax, 'd')
    ax.text(STAMP_XY[0], STAMP_XY[1], f'${ZK}$', transform=ax.transAxes, fontsize=S.FS['label'] + 1,
            fontweight='bold', ha='left', va='top', zorder=20)
    ax.quiverkey(qv, QKEY_XY[0], QKEY_XY[1], QUIVER_KEY, f'{QUIVER_KEY:g} m s$^{{-1}}$', labelpos='N',
                 coordinates='axes', color=QUIVER_COLOUR, fontproperties={'size': S.FS['small']}, zorder=20)

    # the line key above the panel
    hs = [Line2D([], [], color=ZOF_COLOUR, ls='dashed', lw=0.9, label=rf'$\zeta / f = -{ZOF_CONTOUR:g}$'),
          Line2D([], [], color=ZOF_COLOUR, ls='solid', lw=0.9, label=rf'$\zeta / f = +{ZOF_CONTOUR:g}$'),
          Line2D([], [], color=ISOBATH_D[1], ls=ISOBATH_D[2], lw=0.9, label=f'{ISOBATH_D[0]} m')]
    ax.legend(handles=hs, loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=3, borderaxespad=0.0,
              handlelength=2.2, fontsize=S.FS['small'], frameon=False)


def write_numbers():
    L = ['FIGURE 4 PLOT NUMBERS', f'curves of the {CMASK} mask, bands 95 % block bootstrap']
    for k in CURVE_WINDOWS:
        c, lo, hi = curve('f', 'Pi', k)
        j = int(np.nanargmax(np.abs(c)))
        L.append(f'{k} on f_eff/f: extreme {c[j] * 1e3:+.3f} [{lo[j] * 1e3:+.3f} {hi[j] * 1e3:+.3f}] mW m-2 at {cen_f[j]:.2f}')
    for k in LAMBDA_WINDOWS:
        iw = WKEYS.index(k)
        lam = ds['lam'].values[iw, im] * LAMBDA_UNIT
        j = int(np.nanargmin(lam)) if k == 'T2' else int(np.nanargmax(lam))
        L.append(f'{k} lambda from {lam[0]:+.3f} d-1 (most anticyclonic bin) to {lam[-1]:+.3f} (most cyclonic), '
                 f'extreme {lam[j]:+.3f} at {cen_f[j]:.2f}')
    with open(os.path.join(FIG_DIR, 'fig04_plot_numbers.txt'), 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print('\n'.join(L))


# =====================================================================
# FIGURE
# =====================================================================

def main():
    fig = plt.figure(figsize=FIGSIZE)
    top = GridSpec(1, 3, figure=fig, left=0.085, right=0.915, top=0.955, bottom=0.715, wspace=0.07)
    bot = GridSpec(1, 1, figure=fig, left=0.11, right=0.84, top=0.605, bottom=0.045)
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(top[0, 2], sharex=ax_b)
    ax_d = S.map_axes(fig, bot[0, 0])
    panel_a(fig, ax_a)
    panel_b(fig, ax_b)
    panel_c(fig, ax_c)
    panel_d(fig, ax_d)
    write_numbers()
    S.save(fig, OUT_STEM)


if __name__ == '__main__':
    main()