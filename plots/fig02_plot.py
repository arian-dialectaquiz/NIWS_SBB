"""
fig02_plot.py

Figure 2, the subinertial flow and the near-inertial kinetic energy. Draws from
fig02/fig02_data.nc and fig02/fig02_stats.json written by fig02_compute.py, and
takes the track from fig01/fig01_data.nc. Nothing is computed here.

Layout, controlled entirely by the GridSpec blocks in main()

    top[0, 0]   (a) T1 subinertial zeta / f of the upper 100 m, OW = 0 contour,
                    subinertial velocity arrows
    top[0, 1]   (b) rho0 int K dz over T1, log scale
    top[0, 2]   (c) the same over T3, with the track
    bot[0, :]   (d) daily area-weighted rho0 int K dz by sector, through T4
    bot[1, :]   (e) K-weighted mean of zeta / 2f by sector against the area mean,
                    the refraction of the wake into anticyclonic water, day by day

One colourbar serves (b) and (c). Every legend sits outside the data, the
colourbars below the maps, extended only where the data pass the limits.

    python fig02_plot.py
"""

import json
import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import style_v3 as S

try:
    import cmocean.cm as cmo
    CMAP_ZETA, CMAP_K = cmo.balance, cmo.thermal
except Exception:
    CMAP_ZETA, CMAP_K = plt.get_cmap('RdBu_r'), plt.get_cmap('inferno')

S.apply()

# =====================================================================
# CONFIGURATION
# =====================================================================

FIG_DIR = os.path.join(os.getcwd(), 'fig02')
IN_NC = os.path.join(FIG_DIR, 'fig02_data.nc')
IN_JSON = os.path.join(FIG_DIR, 'fig02_stats.json')
FIG01_NC = os.path.join(os.getcwd(), 'fig01', 'fig01_data.nc')
DAILY_NC = os.path.join(FIG_DIR, 'fig02_daily_sub.nc')     # written by fig02_daily_sub.py
OUT_STEM = os.path.join(FIG_DIR, 'fig02_energy')

FIGSIZE = (S.PAGE_W_IN, 9.2)
MAP_EXTENT = S.MAP_EXTENT
ZETA_SCALE = 0.3                      # zeta / f, symmetric
K_RANGE = (30.0, 3000.0)              # J m-2, log colour scale of (b) and (c), None takes quantiles
QUIVER_DECIM = 30                     # cells between arrows
OW_SMOOTH_CELLS = 9                   # boxcar on OW before the zero contour, cells
QUIVER_SCALE = 6.0                    # m s-1 per axes width
QUIVER_KEY = 0.5                      # m s-1
OW_CONTOUR_COLOUR = 'black'
ISOBATHS_MAPS = [iso for iso in S.ISOBATHS if iso[0] in (200, 1000)]

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
f2d = ds['f'].values
WINDOWS = {k: tuple(ds.attrs[k]) for k in ('T1', 'T2', 'T3', 'T4') if k in ds.attrs}
LANDFALL = np.datetime64(str(ds.attrs['landfall']))
days = ds['day'].values
sectors = [str(s) for s in ds['sector_name'].values]
series = ds['K_series'].values
stride = int(ds.attrs['stride'])
s_top, c_top = int(ds.attrs['south_top']), int(ds.attrs['central_top'])

track = None
if os.path.exists(FIG01_NC):
    with xr.open_dataset(FIG01_NC) as d1:
        track = dict(lon=d1['track_lon'].values, lat=d1['track_lat'].values,
                     type=d1['track_type'].values.astype(str), time=d1['track_time'].values)
        STN_POS = {n: (float(d1.attrs[f'{n}_lon']), float(d1.attrs[f'{n}_lat']))
                   for n in ('A', 'B', 'C', 'D') if f'{n}_lon' in d1.attrs}
else:
    STN_POS = {}

K1, K3 = ds['K_T1'].values, ds['K_T3'].values
if K_RANGE is None:
    K_RANGE = (max(float(np.nanquantile(K1, 0.02)), 1.0), float(np.nanquantile(K3, 0.995)))
norm_k = mcolors.LogNorm(*K_RANGE)
ext_k = S.extend_for(np.concatenate([K1[np.isfinite(K1)], K3[np.isfinite(K3)]]), *K_RANGE)
print(f'K colour range {K_RANGE[0]:.0f} to {K_RANGE[1]:.0f} J m-2, extend {ext_k}')


def map_furniture(ax, left_labels):
    S.basemap(ax, lon, lat, mask, MAP_EXTENT, left_labels=left_labels)
    S.isobaths(ax, lon, lat, h_wet, which=ISOBATHS_MAPS, lw=0.7)
    tf = S._tf(ax)
    for row in (s_top, c_top):
        ax.plot(lon[row, :], lat[row, :], color='black', lw=0.9, zorder=5, **tf)


# =====================================================================
# PANELS
# =====================================================================

def panel_a(fig, ax):
    zof = np.where(land, np.nan, ds['zeta_T1'].values / f2d)      # signed f, so anticyclonic is negative as in Figure 4
    ow = np.where(land, np.nan, ds['OW_T1'].values)
    ue, vn = ds['ue_T1'].values, ds['vn_T1'].values
    tf = S._tf(ax)
    norm = mcolors.TwoSlopeNorm(vmin=-ZETA_SCALE, vcenter=0.0, vmax=ZETA_SCALE)
    pm = ax.pcolormesh(lon, lat, zof, cmap=CMAP_ZETA, norm=norm, shading='auto', zorder=0, **tf)
    from scipy.ndimage import uniform_filter
    ow_s = uniform_filter(np.nan_to_num(ow, nan=0.0), OW_SMOOTH_CELLS, mode='nearest')
    ow_s[land] = np.nan
    ax.contour(lon, lat, ow_s, levels=[0.0], colors=OW_CONTOUR_COLOUR, linewidths=0.3,
               zorder=2, **tf)
    q = QUIVER_DECIM
    qv = ax.quiver(lon[::q, ::q], lat[::q, ::q], ue[::q, ::q], vn[::q, ::q], color='0.1',
                   scale=QUIVER_SCALE, scale_units='width', width=0.0035, headwidth=3.5,
                   zorder=6, **tf)
    map_furniture(ax, left_labels=True)
    ext = S.extend_for(zof, -ZETA_SCALE, ZETA_SCALE)
    S.colorbar_outside(fig, ax, pm, r'$\zeta / f$ of the upper 100 m during $T1$', extend=ext,
                       where='bottom', size=0.035, pad=0.115, length=0.9,
                       ticks=[-0.3, -0.15, 0, 0.15, 0.3])
    # the arrow key and the contour key inside the blank sea at the lower right
    ax.quiverkey(qv, 0.70, 0.115, QUIVER_KEY, f'{QUIVER_KEY:g} m s$^{{-1}}$', labelpos='E',
                 coordinates='axes', fontproperties={'size': S.FS['small']}, zorder=20)
    hs = [Line2D([], [], color=OW_CONTOUR_COLOUR, lw=0.8, label='OW = 0')]
    ax.legend(handles=hs, loc='lower right', bbox_to_anchor=(0.995, 0.005), ncol=1,
              fontsize=S.FS['small'], borderaxespad=0.0, handlelength=1.6, frameon=True,
              framealpha=0.85, edgecolor='none').set_zorder(20)
    S.panel_tag(ax, 'a')


def panel_k(fig, ax, key, tag, left_labels, with_track, cbar):
    K = np.where(land, np.nan, ds[f'K_{key}'].values)
    tf = S._tf(ax)
    pm = ax.pcolormesh(lon, lat, K, cmap=CMAP_K, norm=norm_k, shading='auto', zorder=0, **tf)
    map_furniture(ax, left_labels=left_labels)
    if with_track and track is not None:
        S.track(ax, track['lon'], track['lat'], track['time'], track['type'], stage=True,
                dates=False, colour='white', ms=16)
    if cbar:
        S.colorbar_outside(fig, ax, pm, r'$\rho_0 \int K\,\mathrm{d}z$ (J m$^{-2}$)', extend=ext_k,
                           where='bottom', size=0.035, pad=0.115, length=0.9,
                           ticks=[30, 100, 300, 1000, 3000], fmt=lambda v, _: f'{v:g}')
    ax.text(0.03, 0.03, f'${key}$', transform=ax.transAxes, fontsize=S.FS['label'],
            fontweight='bold', ha='left', va='bottom', zorder=12)
    S.panel_tag(ax, tag)
    return pm


def series_furniture(ax):
    for k in ('T1', 'T2', 'T3', 'T4'):
        if k in WINDOWS:
            a, b = WINDOWS[k]
            ax.axvspan(np.datetime64(a), np.datetime64(b) + np.timedelta64(1, 'D'),
                       color=S.WINDOW[k]['shade'], alpha=0.9, lw=0, zorder=0)
            ax.text(np.datetime64(a) + np.timedelta64(3, 'D') + np.timedelta64(12, 'h'), 0.97, f'${k}$',
                    transform=ax.get_xaxis_transform(), fontsize=S.FS['small'],
                    color=S.WINDOW[k]['color'], ha='center', va='top', zorder=1)
    ax.axvline(LANDFALL, color='black', ls=(0, (4, 2)), lw=0.9, zorder=2)
    ax.grid(True, ls=':', lw=0.4, alpha=0.5, zorder=0)
    ax.set_xlim(days[0], days[-1] + np.timedelta64(1, 'D'))


def panel_d(ax):
    series_furniture(ax)
    x = days + np.timedelta64(12, 'h')
    for i, name in enumerate(sectors):
        sp = S.SECTOR[name]
        ax.plot(x, series[i], color=sp['color'], ls=sp['ls'], marker=sp['marker'], ms=3.5,
                lw=2.0 if name == 'Domain' else 1.6, zorder=3, label=name)
    ax.set_yscale('log')
    ax.set_ylabel(r'$\rho_0 \int K\,\mathrm{d}z$ (J m$^{-2}$)')
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=4, borderaxespad=0.0,
              handlelength=2.2, fontsize=S.FS['small'])
    S.panel_tag(ax, 'd')


def refraction_series():
    """Where the near-inertial kinetic energy sits in the subinertial vorticity
    field, day by day. For each sector the K-weighted mean of zeta / 2f, which is
    f_eff / f - 1 of the water that holds the waves, against the plain area mean
    of zeta / 2f, which is what a wave field spread evenly over the sector would
    give. A K-weighted value below the area mean means the waves have collected
    in anticyclonic water, the refraction of Kunze (1985). The vorticity is the
    daily upper-100 m subinertial field when the compute wrote it, else that of
    the window that contains the day."""
    sector_id = ds['sector'].values
    shelf = (h <= float(ds.attrs['stat_max_depth_m'])) & ~land
    masks = {'South': shelf & (sector_id == 1), 'Central': shelf & (sector_id == 2),
             'North': shelf & (sector_id == 3), 'Domain': shelf}
    Kd = ds['K_daily'].values
    wkeys = [k for k in ('T1', 'T2', 'T3', 'T4') if k in WINDOWS]
    starts = np.array([np.datetime64(WINDOWS[k][0]) for k in wkeys])
    zof = {k: ds[f'zeta_{k}'].values / (2.0 * f2d) for k in wkeys}
    f_use, area, sub = f2d, ds['area'].values, 1
    zeta_daily = None
    if 'zeta_daily' in ds:
        zeta_daily = ds['zeta_daily'].values
        print('panel e: daily upper-layer subinertial vorticity from fig02_data.nc')
    elif os.path.exists(DAILY_NC):
        with xr.open_dataset(DAILY_NC) as dd:
            sub = int(dd.attrs['stride']) // stride
            zd = dd['zeta_daily'].values
            dd_days = dd['day'].values.astype('datetime64[D]')
        # the companion file sits on a coarser grid, so K and the masks are subsampled to it
        if sub < 1 or zd.shape[1:] != Kd[:, ::sub, ::sub].shape[1:]:
            raise SystemExit(f'{DAILY_NC} grid {zd.shape[1:]} does not match fig02_data.nc at stride {sub}')
        zeta_daily = np.full((days.size,) + zd.shape[1:], np.nan, np.float32)
        for di, d in enumerate(days):
            k = np.where(dd_days == d.astype('datetime64[D]'))[0]
            if k.size:
                zeta_daily[di] = zd[k[0]]
        print(f'panel e: daily upper-layer subinertial vorticity from fig02_daily_sub.nc (stride {sub} of the K grid)')
    else:
        print('panel e: window-mean vorticity (run fig02_daily_sub.py for the daily field)')
    if sub > 1:
        Kd = Kd[:, ::sub, ::sub]
        f_use = f2d[::sub, ::sub]
        area = area[::sub, ::sub] * sub * sub
        shelf_sub = ((h <= float(ds.attrs['stat_max_depth_m'])) & ~land)[::sub, ::sub]
        sector_sub = ds['sector'].values[::sub, ::sub]
        masks = {'South': shelf_sub & (sector_sub == 1), 'Central': shelf_sub & (sector_sub == 2),
                 'North': shelf_sub & (sector_sub == 3), 'Domain': shelf_sub}
        zof = {k: v[::sub, ::sub] for k, v in zof.items()}
    out = {name: dict(kw=np.full(days.size, np.nan), am=np.full(days.size, np.nan),
                      frac_K_anti=np.full(days.size, np.nan), frac_area_anti=np.nan)
           for name in masks}
    for name, m in masks.items():
        for di, d in enumerate(days):
            if zeta_daily is not None and np.isfinite(zeta_daily[di]).any():
                z = zeta_daily[di] / (2.0 * f_use)
            else:
                k = wkeys[max(0, int(np.searchsorted(starts, d, side='right')) - 1)]
                z = zof[k]
            good = m & np.isfinite(z) & np.isfinite(Kd[di])
            if not good.any():
                continue
            w = Kd[di][good] * area[good]
            out[name]['kw'][di] = float(np.sum(w * z[good]) / np.sum(w))
            out[name]['am'][di] = float(np.sum(area[good] * z[good]) / np.sum(area[good]))
            anti = z[good] < 0                        # zeta / 2f < 0, f_eff / f < 1, anticyclonic
            out[name]['frac_K_anti'][di] = float(np.sum(w[anti]) / np.sum(w))
        zz = zof['T1']
        good = m & np.isfinite(zz)
        out[name]['frac_area_anti'] = float(np.sum(area[good & (zz < 0)]) / np.sum(area[good]))
    return out


def panel_e(ax):
    R = refraction_series()
    series_furniture(ax)
    ax.axhline(0.0, color='0.4', lw=0.8, zorder=1)
    x = days + np.timedelta64(12, 'h')
    lines = []
    for name in sectors:
        sp = S.SECTOR[name]
        ax.plot(x, R[name]['kw'], color=sp['color'], ls=sp['ls'], marker=sp['marker'], ms=3.5,
                lw=2.0 if name == 'Domain' else 1.6, zorder=3)
        ax.plot(x, R[name]['am'], color=sp['color'], ls='solid', lw=0.8, alpha=0.6, zorder=2)
        lines.append(f'  {name:8s} area mean zeta/2f {np.nanmean(R[name]["am"]):+.3f}, K-weighted '
                     f'T1 {np.nanmean(R[name]["kw"][(days >= np.datetime64(WINDOWS["T1"][0])) & (days <= np.datetime64(WINDOWS["T1"][1]))]):+.3f}, '
                     + ', '.join(f'{k} {np.nanmean(R[name]["kw"][(days >= np.datetime64(WINDOWS[k][0])) & (days <= np.datetime64(WINDOWS[k][1]))]):+.3f}'
                                 for k in ('T2', 'T3', 'T4') if k in WINDOWS)
                     + f', minimum {np.nanmin(R[name]["kw"]):+.3f} on {str(days[int(np.nanargmin(R[name]["kw"]))])[:10]}'
                     + f'; share of K in anticyclonic water T1 {100 * np.nanmean(R[name]["frac_K_anti"][(days >= np.datetime64(WINDOWS["T1"][0])) & (days <= np.datetime64(WINDOWS["T1"][1]))]):.0f}%'
                     + ''.join(f', {k} {100 * np.nanmean(R[name]["frac_K_anti"][(days >= np.datetime64(WINDOWS[k][0])) & (days <= np.datetime64(WINDOWS[k][1]))]):.0f}%'
                               for k in ('T2', 'T3', 'T4') if k in WINDOWS)
                     + f' against an anticyclonic area of {100 * R[name]["frac_area_anti"]:.0f}%')
    ax.set_ylabel(r'$\langle \zeta / 2f \rangle$')
    hs = [Line2D([], [], color='0.3', lw=1.8, marker='o', ms=3.5, label='weighted by $K$'),
          Line2D([], [], color='0.3', lw=0.8, alpha=0.6, label='area mean')]
    ax.legend(handles=hs, loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=2,
              borderaxespad=0.0, handlelength=2.4, fontsize=S.FS['small'])
    S.date_axis(ax, interval=4, label='2004')
    S.panel_tag(ax, 'e')
    txt = ['FIGURE 2e, refraction of the wake into anticyclonic water',
           'K-weighted mean of zeta/2f (= f_eff/f - 1 of the water that holds the waves) against the area mean,',
           'shelf and slope h <= 1000 m, ' + ('daily' if ('zeta_daily' in ds or os.path.exists(DAILY_NC)) else 'window-mean') + ' vorticity of the upper 100 m'] + lines
    with open(os.path.join(FIG_DIR, 'fig02_refraction.txt'), 'w') as fh:
        fh.write('\n'.join(txt) + '\n')
    print('\n'.join(txt))


# =====================================================================
# ASSEMBLY
# =====================================================================

def main():
    fig = plt.figure(figsize=FIGSIZE)
    top = GridSpec(1, 3, figure=fig, left=0.065, right=0.985, top=0.975, bottom=0.665, wspace=0.08)
    bot = GridSpec(2, 1, figure=fig, left=0.065, right=0.985, top=0.53, bottom=0.055, hspace=0.26)
    ax_a = S.map_axes(fig, top[0, 0])
    ax_b = S.map_axes(fig, top[0, 1])
    ax_c = S.map_axes(fig, top[0, 2])
    ax_d = fig.add_subplot(bot[0, 0])
    ax_e = fig.add_subplot(bot[1, 0], sharex=ax_d)
    panel_a(fig, ax_a)
    panel_k(fig, ax_b, 'T1', 'b', left_labels=False, with_track=False, cbar=False)
    pm = panel_k(fig, ax_c, 'T3', 'c', left_labels=False, with_track=True, cbar=False)
    # one colourbar under (b) and (c) together
    pb, pc = ax_b.get_position(), ax_c.get_position()
    w = (pc.x1 - pb.x0) * 0.9
    x0 = pb.x0 + 0.5 * ((pc.x1 - pb.x0) - w)
    cax = fig.add_axes([x0, pb.y0 - pb.height * 0.115 - pb.height * 0.035, w, pb.height * 0.035])
    cb = fig.colorbar(pm, cax=cax, orientation='horizontal', extend=ext_k)
    cb.set_ticks([30, 100, 300, 1000, 3000])
    cb.formatter = mpl.ticker.FuncFormatter(lambda v, _: f'{v:g}')
    cb.update_ticks()
    cb.set_label(r'$\rho_0 \int K\,\mathrm{d}z$ (J m$^{-2}$)', size=S.FS['label'] - 1)
    cax.tick_params(labelsize=S.FS['tick'] - 1)
    panel_d(ax_d)
    panel_e(ax_e)
    S.save(fig, OUT_STEM)


if __name__ == '__main__':
    main()