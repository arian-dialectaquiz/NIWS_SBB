"""
fig03_plot.py

Figure 3, the exchange between the subinertial flow and the near-inertial waves,
drawn from fig03/fig03_data.nc and fig03/fig03_stats.json alone.

(a) (b) (c)  depth-integrated rho0 Pi over T1, T2 and T3 on one symmetric scale,
             positive from the subinertial flow to the waves, track in (b)
(d)          daily rho0 int Pi dz by sector through T4, landfall and the South
             reversal marked, the lag written in local inertial periods
(e)          South, the horizontal and the vertical shear production and their sum
(f)          South, the horizontal production split into the deformation work and
             the divergence work

Layout is set entirely by the GridSpec blocks in main(). Tags sit inside the axes,
no commas appear in any figure text, legends sit above the series panels.

    python fig03_plot.py
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

import style_v3 as S

# blue (to the subinertial flow) to white (zero) to red (to the waves); a map distinct from
# the balance map that Figure 2 keeps for zeta / f
CMAP_PI = plt.get_cmap('RdBu_r')

S.apply()

# =====================================================================
# CONFIGURATION
# =====================================================================

FIG_DIR = os.path.join(os.getcwd(), 'fig03')
IN_NC = os.path.join(FIG_DIR, 'fig03_data.nc')
IN_JSON = os.path.join(FIG_DIR, 'fig03_stats.json')
FIG01_NC = os.path.join(os.getcwd(), 'fig01', 'fig01_data.nc')
OUT_STEM = os.path.join(FIG_DIR, 'fig03_exchange')
OUT_TXT = os.path.join(FIG_DIR, 'fig03_plot_numbers.txt')

FIGSIZE = (S.PAGE_W_IN, 10.0)
MAP_EXTENT = S.MAP_EXTENT
PI_SCALE = 5.0                        # mW m-2, symmetric limit of (a) to (c) (98th percentile of |Pi| is 4.9); None takes it from the data
MAP_SMOOTH_CELLS = 3                  # boxcar on the maps for display only
ISOBATHS_MAPS = [iso for iso in S.ISOBATHS if iso[0] in (200, 1000)]
SERIES_UNIT = 1e3                     # W m-2 to mW m-2
SOUTH = 'South'

# =====================================================================
# LOAD
# =====================================================================

ds = xr.open_dataset(IN_NC)
with open(IN_JSON) as fh:
    ST = json.load(fh)
lon, lat = ds['lon'].values, ds['lat'].values
h, mask = ds['h'].values, ds['mask'].values
land = mask < 0.5
h_wet = np.where(land, np.nan, h)
WINDOWS = {k: tuple(ds.attrs[k]) for k in ('T1', 'T2', 'T3', 'T4') if k in ds.attrs}
LANDFALL = np.datetime64(str(ds.attrs['landfall']))
days = ds['day'].values
sectors = [str(s) for s in ds['sector_name'].values]
s_top, c_top = int(ds.attrs['south_top']), int(ds.attrs['central_top'])
REV = ST.get('reversal', {})
TF_SOUTH = float(ds.attrs['Tf_south_h'])
HAVE_W = bool(int(ds.attrs.get('have_w', 1)))

track = None
if os.path.exists(FIG01_NC):
    with xr.open_dataset(FIG01_NC) as d1:
        track = dict(lon=d1['track_lon'].values, lat=d1['track_lat'].values,
                     type=d1['track_type'].values.astype(str), time=d1['track_time'].values)


def smooth_map(a):
    from scipy.ndimage import uniform_filter
    if MAP_SMOOTH_CELLS <= 1:
        return a
    valid = np.isfinite(a).astype(float)
    num = uniform_filter(np.nan_to_num(a, nan=0.0), MAP_SMOOTH_CELLS, mode='nearest')
    den = uniform_filter(valid, MAP_SMOOTH_CELLS, mode='nearest')
    out = np.where(den > 0.5, num / np.maximum(den, 1e-12), np.nan)
    out[land] = np.nan
    return out


PI_MAPS = {k: smooth_map(np.where(land, np.nan, (ds[f'PiH_{k}'].values + ds[f'PiV_{k}'].values) * SERIES_UNIT))
           for k in ('T1', 'T2', 'T3')}
if PI_SCALE is None:
    allv = np.concatenate([np.abs(v[np.isfinite(v)]) for v in PI_MAPS.values()])
    q = float(np.quantile(allv, 0.98))
    PI_SCALE = float(np.round(q, 1 - int(np.floor(np.log10(max(q, 1e-6))))))
norm_pi = mcolors.TwoSlopeNorm(vmin=-PI_SCALE, vcenter=0.0, vmax=PI_SCALE)
ext_pi = S.extend_for(np.concatenate([v[np.isfinite(v)] for v in PI_MAPS.values()]), -PI_SCALE, PI_SCALE)
print(f'Pi colour scale +-{PI_SCALE:g} mW m-2 (extend {ext_pi})')


def map_furniture(ax, left_labels):
    S.basemap(ax, lon, lat, mask, MAP_EXTENT, left_labels=left_labels)
    S.isobaths(ax, lon, lat, h_wet, which=ISOBATHS_MAPS, lw=0.7)
    tf = S._tf(ax)
    for row in (s_top, c_top):
        ax.plot(lon[row, :], lat[row, :], color='black', lw=0.9, zorder=5, **tf)


# =====================================================================
# PANELS
# =====================================================================

def panel_map(fig, ax, key, tag, left_labels, with_track):
    tf = S._tf(ax)
    pm = ax.pcolormesh(lon, lat, PI_MAPS[key], cmap=CMAP_PI, norm=norm_pi, shading='auto', zorder=0, **tf)
    map_furniture(ax, left_labels=left_labels)
    if with_track and track is not None:
        S.track(ax, track['lon'], track['lat'], track['time'], track['type'], stage=True,
                dates=False, colour='black', ms=16)
    ax.text(0.03, 0.03, f'${key}$', transform=ax.transAxes, fontsize=S.FS['label'],
            fontweight='bold', ha='left', va='bottom', zorder=12)
    S.panel_tag(ax, tag)
    return pm


def wide_colorbar(fig, ax_left, ax_right, mappable, height=0.012, gap=0.035):
    """Horizontal colourbar below the map row, from the centre of ax_left to the
    centre of ax_right. The map axes take their final aspect before the positions
    are read."""
    for ax in (ax_left, ax_right):
        ax.apply_aspect()
    pl, pr = ax_left.get_position(), ax_right.get_position()
    x0 = 0.5 * (pl.x0 + pl.x1)
    x1 = 0.5 * (pr.x0 + pr.x1)
    y0 = min(pl.y0, pr.y0) - gap - height
    cax = fig.add_axes([x0, y0, x1 - x0, height])
    cb = fig.colorbar(mappable, cax=cax, orientation='horizontal', extend=ext_pi)
    t = PI_SCALE
    cb.set_ticks([-t, -t / 2, 0, t / 2, t])
    cb.formatter = mpl.ticker.FuncFormatter(lambda v, _: f'{v:g}')
    cb.update_ticks()
    cb.set_label(r'$\rho_0 \int \Pi\,\mathrm{d}z$ (mW m$^{-2}$)', size=S.FS['label'] - 1)
    cax.tick_params(labelsize=S.FS['tick'] - 1)
    cb.outline.set_linewidth(0.6)
    return cb


def sector_title(ax, name):
    """The sector named above the panel at the right, after the legend."""
    ax.text(1.0, 1.02, name, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=S.FS['label'], fontweight='bold', color=S.SECTOR[name]['color'])


def series_furniture(ax, last=False):
    for k in ('T1', 'T2', 'T3', 'T4'):
        if k in WINDOWS:
            a, b = WINDOWS[k]
            ax.axvspan(np.datetime64(a), np.datetime64(b) + np.timedelta64(1, 'D'),
                       color=S.WINDOW[k]['shade'], alpha=0.9, lw=0, zorder=0)
            if last:
                ax.text(np.datetime64(a) + np.timedelta64(3, 'D') + np.timedelta64(12, 'h'), 0.04, f'${k}$',
                        transform=ax.get_xaxis_transform(), fontsize=S.FS['small'],
                        color=S.WINDOW[k]['color'], ha='center', va='bottom', zorder=1)
    ax.axvline(LANDFALL, color='black', ls=(0, (4, 2)), lw=0.9, zorder=2)
    ax.axhline(0.0, color='0.3', lw=0.6, zorder=1)
    ax.grid(True, ls=':', lw=0.4, alpha=0.5, zorder=0)
    ax.set_xlim(days[0], days[-1] + np.timedelta64(1, 'D'))


def legend_above(ax, handles, ncol):
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=ncol,
              borderaxespad=0.0, handlelength=2.4, fontsize=S.FS['small'], frameon=False)


def panel_d(ax):
    series_furniture(ax)
    x = days + np.timedelta64(12, 'h')
    ser = ds['Pi_series'].values * SERIES_UNIT
    hs = []
    for i, name in enumerate(sectors):
        sp = S.SECTOR[name]
        ax.plot(x, ser[i], color=sp['color'], ls=sp['ls'], marker=sp['marker'], ms=3.5,
                lw=2.0 if name == 'Domain' else 1.6, zorder=3)
        hs.append(S.line_handle(sp, name))
    legend_above(ax, hs, 4)
    ax.set_ylabel(r'$\rho_0 \int \Pi\,\mathrm{d}z$ (mW m$^{-2}$)')
    plt.setp(ax.get_xticklabels(), visible=False)
    # the South reversal (first positive day after the minimum) named next to the curve
    if REV.get('reversal_day'):
        t_rev = np.datetime64(REV['reversal_day']) + np.timedelta64(12, 'h')
        y_txt = 0.5 * float(REV['minimum_mW_m2'])
        ax.text(t_rev + np.timedelta64(16, 'h'), y_txt,
                f'South reversal\n{REV["lag_inertial_periods"]:.1f} $T_f$ after landfall',
                fontsize=S.FS['small'], color=S.SECTOR[SOUTH]['color'], ha='left', va='center', zorder=6)
    S.panel_tag(ax, 'd', loc='upper right')


def panel_e(ax):
    series_furniture(ax)
    x = days + np.timedelta64(12, 'h')
    i_s = sectors.index(SOUTH)
    tot = ds['Pi_series'].values[i_s] * SERIES_UNIT
    ph = ds['PiH_series'].values[i_s] * SERIES_UNIT
    pv = ds['PiV_series'].values[i_s] * SERIES_UNIT
    specs = [('PiH', ph, r'$\Pi_H$ horizontal shear production'),
             ('PiV', pv, r'$\Pi_V$ vertical shear production'),
             ('Pi', tot, r'$\Pi$ total')]
    hs = []
    for key, y, lab in specs:
        sp = dict(S.TERM[key])
        ls = {'PiH': 'solid', 'PiV': 'dashed', 'Pi': 'dotted'}[key]
        ax.plot(x, y, color=sp['color'], ls=ls, lw=2.0 if key == 'Pi' else 1.6, zorder=3)
        hs.append(Line2D([], [], color=sp['color'], ls=ls, lw=1.6, label=lab))
    legend_above(ax, hs, 3)
    sector_title(ax, SOUTH)
    ax.set_ylabel(r'$\rho_0 \int \Pi\,\mathrm{d}z$ (mW m$^{-2}$)')
    plt.setp(ax.get_xticklabels(), visible=False)
    S.panel_tag(ax, 'e', loc='upper right')


def panel_f(ax):
    series_furniture(ax, last=True)
    x = days + np.timedelta64(12, 'h')
    i_s = sectors.index(SOUTH)
    dfm = ds['PiHdef_series'].values[i_s] * SERIES_UNIT
    dvg = ds['PiHdiv_series'].values[i_s] * SERIES_UNIT
    specs = [('def', dfm, r'deformation work $-M\Sigma\cos 2(\phi_w - \phi_s)$'),
             ('div', dvg, r'divergence work $-K\delta$')]
    hs = []
    for key, y, lab in specs:
        sp = S.TERM[key]
        ax.plot(x, y, color=sp['color'], ls=sp['ls'], lw=1.6, zorder=3)
        hs.append(S.line_handle(sp, lab))
    legend_above(ax, hs, 2)
    sector_title(ax, SOUTH)
    ax.set_ylabel(r'$\rho_0 \int \Pi_H\,\mathrm{d}z$ (mW m$^{-2}$)')
    S.date_axis(ax, interval=4, label='2004')
    S.panel_tag(ax, 'f', loc='upper right')


def write_numbers():
    L = ['FIGURE 3 PLOT NUMBERS', f'Pi colour scale +-{PI_SCALE:g} mW m-2 extend {ext_pi}',
         f'landfall {str(LANDFALL)[:16]} South inertial period {TF_SOUTH:.1f} h',
         f'reversal {json.dumps(REV)}', '']
    i_s = sectors.index(SOUTH)
    for key in ('Pi', 'PiH', 'PiV', 'PiHdef', 'PiHdiv'):
        y = ds[f'{key}_series'].values[i_s] * SERIES_UNIT
        j = int(np.nanargmin(y))
        L.append(f'South {key:7s} minimum {y[j]:+.3f} mW m-2 on {str(days[j])[:10]}, '
                 f'maximum {np.nanmax(y):+.3f} on {str(days[int(np.nanargmax(y))])[:10]}')
    for k in ('T1', 'T2', 'T3', 'T4'):
        if k in ST['table']['South']:
            r = ST['table']['South'][k]
            L.append(f'South {k}: Pi {r["Pi"] * 1e3:+.3f}, PiH {r["PiH"] * 1e3:+.3f} '
                     f'(def {r["PiHdef"] * 1e3:+.3f} div {r["PiHdiv"] * 1e3:+.3f}), PiV {r["PiV"] * 1e3:+.3f} mW m-2')
    with open(OUT_TXT, 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print('\n'.join(L))


# =====================================================================
# FIGURE
# =====================================================================

def main():
    fig = plt.figure(figsize=FIGSIZE)
    top = GridSpec(1, 3, figure=fig, left=0.065, right=0.985, top=0.98, bottom=0.70, wspace=0.08)
    bot = GridSpec(3, 1, figure=fig, left=0.095, right=0.985, top=0.565, bottom=0.05, hspace=0.30)
    ax_a = S.map_axes(fig, top[0, 0])
    ax_b = S.map_axes(fig, top[0, 1])
    ax_c = S.map_axes(fig, top[0, 2])
    ax_d = fig.add_subplot(bot[0, 0])
    ax_e = fig.add_subplot(bot[1, 0], sharex=ax_d)
    ax_f = fig.add_subplot(bot[2, 0], sharex=ax_d)

    pm = panel_map(fig, ax_a, 'T1', 'a', left_labels=True, with_track=False)
    panel_map(fig, ax_b, 'T2', 'b', left_labels=False, with_track=True)
    panel_map(fig, ax_c, 'T3', 'c', left_labels=False, with_track=False)
    wide_colorbar(fig, ax_a, ax_c, pm)
    panel_d(ax_d)
    panel_e(ax_e)
    panel_f(ax_f)
    fig.align_ylabels([ax_d, ax_e, ax_f])
    if not HAVE_W:
        ax_e.text(0.5, 0.5, 'w absent from the archive so $\\Pi_V$ is zero', transform=ax_e.transAxes,
                  ha='center', va='center', fontsize=S.FS['small'], color='firebrick')
    write_numbers()
    S.save(fig, OUT_STEM)


if __name__ == '__main__':
    main()