"""
fig01_plot.py

Figure 1, the storm, the wind forcing and the shelf. Draws from fig01_data.nc
and fig01_stats.json written by fig01_compute.py. No model file is opened and
nothing is computed here.

Layout, controlled entirely by the two GridSpec blocks in main()

    top[0, 0]   (a) bathymetry, isobaths, landmarks, stations, sectors, with an
                    inset of the two grids over the continent
    top[0, 1]   (b) near-inertial wind work integrated over T2, kJ m-2
    bot[0, :]   (c) wind stress magnitude at A to D (log) over a direction ribbon
    bot[1, 0]   (d) wind-work cospectrum against omega / f, 30 km boxes
    bot[1, 1]   (e) daily near-inertial wind work by sector through T4

Every legend sits outside the data area, every colourbar outside its panel, and
the colourbars extend only towards the side the data pass the limits.

    python fig01_plot.py
"""

import json

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import style_v3 as S

try:
    import cmocean.cm as cmo
    CMAP_WNI = cmo.thermal
except Exception:
    CMAP_WNI = plt.get_cmap('inferno')

S.apply()

# =====================================================================
# CONFIGURATION
# =====================================================================

import os
FIG_DIR = os.path.join(os.getcwd(), 'fig01')
IN_NC = os.path.join(FIG_DIR, 'fig01_data.nc')
IN_JSON = os.path.join(FIG_DIR, 'fig01_stats.json')
OUT_STEM = os.path.join(FIG_DIR, 'fig01_setting')

FIGSIZE = (S.PAGE_W_IN, 9.2)
MAP_EXTENT = S.MAP_EXTENT
BATHY_RANGE = (0.0, 2500.0)                   # m, colour scale of panel (a)
WNI_RANGE = (0.03, 30.0)                      # kJ m-2, log colour scale of panel (b)
WNI_WINDOW = 'T2'
INSET_EXTENT = [-58.0, -30.0, -40.0, -14.0]   # lon0, lon1, lat0, lat1
INSET_BOX = [0.015, 0.615, 0.34, 0.37]        # axes fraction, panel (a), over the continent

TAU_SMOOTH_H = 12
TAU_YLIM = (1e-2, 1e1)                        # N m-2
DIR_MIN = 0.02                                # N m-2, direction left blank below
DIR_CMAP = 'twilight'                         # cyclic, named
DIR_RATIO = 2.3                               # magnitude panel height over the ribbon

COSPEC_XLIM = (0.2, 12.0)                     # omega / f
COSPEC_YLIM = (1e-5, None)                    # W m-2 cph-1, None takes 3 x the maximum
NI_BAND_COLOUR = 'lemonchiffon'

SHADE_KEYS = ('T2',)
E_LINTHRESH = 1.0                             # mW m-2, linear band of panel (e)

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
time = pd.to_datetime(ds['time'].values)
TIME64 = ds['time'].values
WINDOWS = {k: tuple(ds.attrs[k]) for k in ('T1', 'T2', 'T3', 'T4') if k in ds.attrs}
LANDFALL = np.datetime64(str(ds.attrs['landfall']))
STATIONS = [n for n in ('A', 'B', 'C', 'D') if f'{n}_u_e' in ds]
STN_POS = {n: (float(ds.attrs[f'{n}_lon']), float(ds.attrs[f'{n}_lat'])) for n in STATIONS}
DEMOD = list(ds.attrs.get('demod_stations', ['D', 'A']))
NI_BAND = tuple(ds.attrs['ni_band'])
dt_h = float(ds.attrs['dt_hours'])


def smooth(x, hours):
    n = max(1, int(round(hours / dt_h)))
    return pd.Series(x).rolling(n, center=True, min_periods=1).mean().to_numpy()


def stress_direction(te, tn):
    """Compass bearing the stress points toward, 0 north, 90 east."""
    return np.degrees(np.arctan2(te, tn)) % 360.0


# =====================================================================
# PANELS
# =====================================================================

def panel_a(fig, ax):
    cmap = mcolors.LinearSegmentedColormap.from_list('bathy', S.BATHY_COLOURS, N=256)
    norm = mcolors.Normalize(*BATHY_RANGE)
    levels = np.linspace(BATHY_RANGE[0], BATHY_RANGE[1], 101)
    ext = S.extend_for(h_wet, *BATHY_RANGE)
    pm = ax.contourf(lon, lat, h_wet, levels=levels, cmap=cmap, norm=norm, extend=ext,
                     zorder=0, **S._tf(ax))
    S.basemap(ax, lon, lat, mask, MAP_EXTENT, left_labels=True)
    S.isobaths(ax, lon, lat, h_wet)
    rows = (ds['divider_south_central_lon'].values, ds['divider_south_central_lat'].values,
            ds['divider_central_north_lon'].values, ds['divider_central_north_lat'].values)
    tf = S._tf(ax)
    ax.plot(rows[0], rows[1], color='black', lw=1.1, zorder=5, **tf)
    ax.plot(rows[2], rows[3], color='black', lw=1.1, zorder=5, **tf)
    # sector names beyond the offshore edge of the subdomain, in the blank sea
    st_ = int(ds.attrs['stride'])
    s_top, c_top = int(ds.attrs['south_top']) // st_, int(ds.attrs['central_top']) // st_
    neta_ = lon.shape[0]
    kw = dict(fontsize=S.FS['label'], fontweight='bold', zorder=6, va='top', color='black', **tf)
    x_max = MAP_EXTENT['lon'][1] - 0.25
    for name, row in (('South', s_top // 2), ('Central', (s_top + c_top) // 2),
                      ('North', (c_top + neta_) // 2)):
        x = lon[row, -1] + 0.35
        if x + 1.6 > x_max:            # the label would leave the map, anchor it on the right edge
            ax.text(x_max, lat[row, -1] - 0.25, name, ha='right', **kw)
        else:
            ax.text(x, lat[row, -1] - 0.25, name, ha='left', **kw)
    S.landmarks(ax)
    S.stations(ax, STN_POS, label=False)
    S.colorbar_outside(fig, ax, pm, 'depth (m)', extend=ext, where='bottom',
                       size=0.035, pad=0.115, length=0.9,
                       ticks=[0, 500, 1000, 1500, 2000, 2500])

    # legend of the stations and the isobaths, over the open ocean
    hs = [Line2D([], [], marker=S.STATION[n]['marker'], color=S.STATION[n]['color'],
                 ls='None', markeredgecolor='black', markeredgewidth=0.5, markersize=7,
                 label=f'station {n}') for n in STATIONS]
    ih, il = S.isobath_handles()
    hs += [Line2D([], [], color=hh.get_color(), ls=hh.get_linestyle(), lw=1.2, label=ll)
           for hh, ll in zip(ih, il)]
    ax.legend(handles=hs, loc='lower left', bbox_to_anchor=(0.07, 1.0), ncol=4,
              fontsize=S.FS['small'], borderaxespad=0.0, handlelength=1.6,
              columnspacing=0.9, handletextpad=0.4)

    # inset, the two grids and the whole track
    if S.HAVE_CARTOPY:
        axi = ax.inset_axes(INSET_BOX, projection=S.PC)
        axi.set_extent(INSET_EXTENT, crs=S.PC)
        if S.natural_earth_available():
            axi.add_feature(S.cfeature.LAND.with_scale('110m'), facecolor=S.LAND_COLOUR, zorder=1)
            axi.coastlines(resolution='110m', linewidth=0.4, zorder=3)
        tfi = dict(transform=S.PC)
    else:
        axi = ax.inset_axes(INSET_BOX)
        axi.set_xlim(INSET_EXTENT[:2])
        axi.set_ylim(INSET_EXTENT[2:])
        axi.set_xticks([])
        axi.set_yticks([])
        tfi = {}
    for key, colour in (('parent', 'grey'), ('child', 'firebrick')):
        if f'perim_{key}_lon' in ds:
            axi.fill(ds[f'perim_{key}_lon'].values, ds[f'perim_{key}_lat'].values,
                     facecolor=colour, alpha=0.22, edgecolor=colour, lw=1.0, zorder=4, **tfi)
    axi.set_facecolor('aliceblue')
    for sp in axi.spines.values():
        sp.set_linewidth(0.7)
    axi.text(0.03, 0.96, '3 km', color='dimgrey', fontsize=S.FS['small'] - 2,
             transform=axi.transAxes, va='top')
    axi.text(0.03, 0.80, '1 km', color='firebrick', fontsize=S.FS['small'] - 2,
             transform=axi.transAxes, va='top')
    S.panel_tag(ax, 'a', pad=22)


def panel_b(fig, ax):
    W = ds[f'W_NI_{WNI_WINDOW}'].values / 1e3                       # kJ m-2
    wet = ~land & np.isfinite(W)
    neg = float(np.nansum((W < 0) & wet) / wet.sum())
    print(f'panel b: {WNI_WINDOW} injection, {100 * neg:.1f} % of the wet cells negative, '
          f'max {np.nanmax(W):.2f} kJ m-2')
    Wp = np.where(W > 0, W, np.nan)
    norm = mcolors.LogNorm(*WNI_RANGE)
    ext = S.extend_for(Wp, *WNI_RANGE)
    cmap = CMAP_WNI.copy() if hasattr(CMAP_WNI, 'copy') else CMAP_WNI
    pm = ax.pcolormesh(lon, lat, Wp, cmap=cmap, norm=norm, shading='auto', zorder=0, **S._tf(ax))
    ax.contourf(lon, lat, np.where(wet & (W <= 0), 1.0, np.nan), levels=[0.5, 1.5],
                colors='silver', zorder=0, **S._tf(ax))
    negm = wet & (W < 0)
    if negm.any():
        print(f'panel b: negative cells have median {np.nanmedian(W[negm]):.3f} and min '
              f'{np.nanmin(W[negm]):.3f} kJ m-2 (shown in silver)')
    S.basemap(ax, lon, lat, mask, MAP_EXTENT, left_labels=False)
    S.isobaths(ax, lon, lat, h_wet, which=[iso for iso in S.ISOBATHS if iso[0] in (200, 1000)])
    tf = S._tf(ax)
    for key in ('south_central', 'central_north'):
        ax.plot(ds[f'divider_{key}_lon'].values, ds[f'divider_{key}_lat'].values,
                color='black', lw=1.0, zorder=5, **tf)
    S.stations(ax, STN_POS, label=True)
    S.track(ax, ds['track_lon'].values, ds['track_lat'].values, ds['track_time'].values,
            ds['track_type'].values, stage=True, dates=False, colour='black')
    present = set(ds['track_type'].values.astype(str))
    hs, ls = S.track_stage_handles(present)
    hs.append(Patch(facecolor='silver'))
    ls.append('negative')
    ax.legend(hs, ls, loc='lower left', bbox_to_anchor=(0.07, 1.0), ncol=3,
              fontsize=S.FS['small'], borderaxespad=0.0, handlelength=1.0,
              columnspacing=0.9, handletextpad=0.4)
    a, b = WINDOWS[WNI_WINDOW]
    S.colorbar_outside(fig, ax, pm, rf'$\rho_0\int W_{{NI}}\,\mathrm{{d}}t$ over ${WNI_WINDOW}$ '
                                    rf'(kJ m$^{{-2}}$)', extend=ext, where='bottom',
                       size=0.035, pad=0.115, length=0.9, ticks=[0.03, 0.1, 0.3, 1, 3, 10, 30],
                       fmt=lambda v, _: f'{v:g}')
    S.panel_tag(ax, 'b', pad=22)


def panel_c(fig, ax_mag, ax_dir):
    for n in STATIONS:
        s = S.STATION[n]
        tau = ds[f'{n}_tau'].values
        ax_mag.plot(time, tau, color=s['color'], lw=0.45, alpha=0.28, zorder=2)
        ax_mag.plot(time, smooth(tau, TAU_SMOOTH_H), color=s['color'], lw=s['lw'],
                    ls=s['ls'], zorder=3, label=f'{n}')
    ax_mag.set_yscale('log')
    ax_mag.set_ylim(*TAU_YLIM)
    ax_mag.set_ylabel(r'$|\boldsymbol{\tau}|$ (N m$^{-2}$)')
    ax_mag.grid(True, which='major', ls=':', lw=0.4, alpha=0.5, zorder=0)
    S.shade_windows(ax_mag, WINDOWS, LANDFALL, keys=SHADE_KEYS, label=True)
    ax_mag.set_xlim(time[0], time[-1])
    plt.setp(ax_mag.get_xticklabels(), visible=False)
    hs, ls = ax_mag.get_legend_handles_labels()
    hs.append(Line2D([], [], color='black', ls=(0, (4, 2)), lw=0.9))
    ls.append('landfall')
    ax_mag.legend(hs, ls, loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=5,
                  borderaxespad=0.0, handlelength=2.4, title=None)
    S.panel_tag(ax_mag, 'c', pad=18)

    ang = np.full((len(STATIONS), time.size), np.nan)
    for i, n in enumerate(STATIONS):
        a = stress_direction(ds[f'{n}_tau_e'].values, ds[f'{n}_tau_n'].values)
        a[ds[f'{n}_tau'].values < DIR_MIN] = np.nan
        ang[i] = a
    cmap = plt.get_cmap(DIR_CMAP).copy()
    cmap.set_bad(alpha=0.0)
    S.shade_windows(ax_dir, WINDOWS, LANDFALL, keys=SHADE_KEYS, label=False)
    x = mdates.date2num(time.to_pydatetime())
    y = np.arange(len(STATIONS), dtype=float)
    pcm = ax_dir.pcolormesh(x, y, np.ma.masked_invalid(ang), cmap=cmap, vmin=0, vmax=360,
                            shading='nearest', zorder=2, rasterized=True)
    ax_dir.xaxis_date()
    ax_dir.set_xlim(time[0], time[-1])
    ax_dir.set_yticks(y)
    ax_dir.set_yticklabels(STATIONS)
    ax_dir.invert_yaxis()
    ax_dir.tick_params(axis='y', length=0)
    for yy in y[:-1] + 0.5:
        ax_dir.axhline(yy, color='white', lw=0.8, zorder=3)
    S.date_axis(ax_dir, interval=4, label=None)
    cb = S.colorbar_outside(fig, ax_dir, pcm, 'stress toward', where='right', size=0.012,
                            pad=0.006, ticks=[0, 90, 180, 270, 360])
    cb.ax.set_yticklabels(['N', 'E', 'S', 'W', 'N'])


def panel_d(ax):
    ax.axvspan(NI_BAND[0], NI_BAND[1], color=NI_BAND_COLOUR, zorder=0, lw=0)
    ax.axvline(1.0, color='0.4', lw=0.7, ls=':', zorder=1)
    top = 0.0
    for n in STATIONS:
        s = S.STATION[n]
        x = ds[f'{n}_cospec_omega_over_f'].values
        co = (ds[f'{n}_cospec_smooth'].values if f'{n}_cospec_smooth' in ds
              else ds[f'{n}_cospec'].values)
        top = max(top, float(np.nanmax(np.abs(co))))
        pos = np.where(co > 0, co, np.nan)
        neg = np.where(co < 0, -co, np.nan)
        ax.plot(x, pos, color=s['color'], lw=s['lw'], ls=s['ls'], zorder=4, label=n)
        ax.plot(x, neg, color=s['color'], lw=0.8, ls=(0, (1, 1.5)), alpha=0.7, zorder=3)
        od = float(ds.attrs[f'{n}_diurnal_over_f'])
        ax.plot([od], [0.99], marker='v', color=s['color'], markeredgecolor='black',
                markeredgewidth=0.4, ms=6, transform=ax.get_xaxis_transform(), clip_on=False,
                zorder=6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*COSPEC_XLIM)
    ax.set_ylim(COSPEC_YLIM[0], COSPEC_YLIM[1] if COSPEC_YLIM[1] else 3.0 * top)
    ax.set_xticks([0.2, 0.5, 1, 2, 5, 10])
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel(r'$\omega / f$')
    ax.set_ylabel(r'$|\Gamma_{\tau u}|$ (W m$^{-2}$ cph$^{-1}$)')
    ax.grid(True, which='major', ls=':', lw=0.4, alpha=0.5, zorder=0)
    hs = [Line2D([], [], color='black', lw=1.3, ls='solid'),
          Line2D([], [], color='black', lw=0.8, ls=(0, (1, 1.5))),
          Patch(facecolor=NI_BAND_COLOUR),
          Line2D([], [], marker='v', color='black', ls='None', ms=5)]
    ls = ['positive', 'negative', f'[{NI_BAND[0]:g}, {NI_BAND[1]:g}] $f$', '1 cpd']
    ax.legend(hs, ls, loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=4,
              borderaxespad=0.0, handlelength=1.6, columnspacing=0.9, fontsize=S.FS['small'])
    S.panel_tag(ax, 'd', pad=18)


def panel_e(ax):
    """Daily near-inertial wind work by sector, shelf and slope shallower than
    1000 m, area weighted, through T4."""
    day = ds['day'].values
    ser = ds['W_NI_series'].values * 1e3            # mW m-2
    names = [str(n) for n in ds['sector'].values]
    for k in ('T1', 'T2', 'T3', 'T4'):
        if k in WINDOWS:
            a, b = WINDOWS[k]
            ax.axvspan(np.datetime64(a), np.datetime64(b) + np.timedelta64(1, 'D'),
                       color=S.WINDOW[k]['shade'], alpha=0.9, lw=0, zorder=0)
            ax.text(np.datetime64(a) + np.timedelta64(3, 'D') + np.timedelta64(12, 'h'), 0.97,
                    f'${k}$', transform=ax.get_xaxis_transform(), fontsize=S.FS['small'],
                    color=S.WINDOW[k]['color'], ha='center', va='top', zorder=1)
    ax.axvline(LANDFALL, color='black', ls=(0, (4, 2)), lw=0.9, zorder=2)
    ax.axhline(0.0, color='0.5', lw=0.5, zorder=1)
    for i, name in enumerate(names):
        sp = S.SECTOR[name]
        ax.plot(day + np.timedelta64(12, 'h'), ser[i], color=sp['color'], ls=sp['ls'],
                lw=2.0 if name == 'Domain' else 1.6, marker=sp['marker'], ms=3.5,
                zorder=4 if name != 'Domain' else 3, label=name)
    ax.set_xlim(time[0], time[-1])
    ax.set_yscale('symlog', linthresh=E_LINTHRESH, linscale=0.6)
    top = float(np.nanmax(ser))
    ax.set_ylim(-E_LINTHRESH, 2.0 * top)
    ax.set_yticks([t for t in (-1, 0, 1, 3, 10, 30, 100, 300) if -E_LINTHRESH <= t <= 2.0 * top])
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_ylabel(r'$\rho_0 W_{NI}$ (mW m$^{-2}$)')
    ax.grid(True, ls=':', lw=0.4, alpha=0.5, zorder=0)
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=4, borderaxespad=0.0,
              handlelength=2.2, fontsize=S.FS['small'])
    S.date_axis(ax, interval=6, label='2004')
    S.panel_tag(ax, 'e', pad=18)


# =====================================================================
# ASSEMBLY
# =====================================================================

def main():
    fig = plt.figure(figsize=FIGSIZE)
    top = GridSpec(1, 2, figure=fig, left=0.07, right=0.985, top=0.955, bottom=0.675,
                   wspace=0.08)
    bot = GridSpec(2, 2, figure=fig, left=0.07, right=0.985, top=0.545, bottom=0.05,
                   height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0], hspace=0.5,
                   wspace=0.28)

    ax_a = S.map_axes(fig, top[0, 0])
    ax_b = S.map_axes(fig, top[0, 1])

    gs_c = GridSpecFromSubplotSpec(2, 1, subplot_spec=bot[0, :],
                                   height_ratios=[DIR_RATIO, 1.0], hspace=0.08)
    ax_c1 = fig.add_subplot(gs_c[0, 0])
    ax_c2 = fig.add_subplot(gs_c[1, 0], sharex=ax_c1)

    ax_d = fig.add_subplot(bot[1, 0])
    ax_e = fig.add_subplot(bot[1, 1])

    panel_a(fig, ax_a)
    panel_b(fig, ax_b)
    panel_c(fig, ax_c1, ax_c2)
    panel_d(ax_d)
    panel_e(ax_e)

    S.save(fig, OUT_STEM)


if __name__ == '__main__':
    main()