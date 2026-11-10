"""
fig05_plot.py

Figure 5, the vertical structure of the exchange, drawn from fig05/fig05_data.nc
alone (South sector).

Row 1   (a) (b)  cross-slope sections of the vertical shear production Pi_V along
                 the station row in the two storm windows (default T2 and T3), with
                 the subinertial speed contoured, the mixed-layer depth and the
                 station, one colourbar from the centre of (a) to the centre of (b)
Row 2   (c) to (g)  sector-mean profiles per window of Pi_H, Pi_V (with the T2
                 anticyclonic and cyclonic water as thin lines), F, Sigma_z and the
                 orientation factor cos(theta_eff), sharing the depth axis
Row 3   (h)      depth and time of Pi at the station through T4, the mixed-layer
                 depth of each window drawn across it

Window colours and line styles follow style_v3.WINDOW. Tags inside, one legend
above the profile row, no comma in any figure text.

    python fig05_plot.py
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

S.apply()

# =====================================================================
# CONFIGURATION
# =====================================================================

FIG_DIR = os.path.join(os.getcwd(), 'fig05')
IN_NC = os.path.join(FIG_DIR, 'fig05_data.nc')
OUT_STEM = os.path.join(FIG_DIR, 'fig05_vertical')

FIGSIZE = (S.PAGE_W_IN, 10.0)
SECTION_WINDOWS = ('T2', 'T3')
PROFILE_WINDOWS = ('T1', 'T2', 'T3', 'T4')
SPLIT_WINDOW = 'T2'                   # the window whose anticyclonic and cyclonic Pi_V profiles are drawn
SEC_Z_MAX = 1000.0                    # m, depth axis of the sections (a) and (b)
PROF_Z_MAX = 300.0                    # m, depth axis of the profiles (c) to (g)
PROF_MIN_FRAC = 0.05                  # profiles drawn where at least this fraction of the class area is present
STN_Z_MAX = None                      # m, depth axis of (h); None takes the local depth of the station
SEC_SCALE = None                      # uW m-3, symmetric limit of the sections; None takes the 98th percentile
STN_SCALE = None                      # uW m-3, symmetric limit of (h); None takes the 98th percentile
CMAP_PI = plt.get_cmap('RdBu_r')
SPEED_LEVELS = (0.2, 0.4, 0.6, 0.8)   # m s-1, subinertial speed contours on the sections
SPEED_COLOUR = 'dimgrey'
MLD_COLOUR = 'black'
BOTTOM_COLOUR = 'lightgrey'
SEC_DEPTH_LIMIT = 1500.0              # the section ends where the bottom first passes this depth
PROF_UNIT = dict(PiH=1e6, PiV=1e6, F=1e6, Sz=1e3, cos=1.0)
PROF_LABEL = dict(PiH=r'$\Pi_H$ ($\mu$W m$^{-3}$)', PiV=r'$\Pi_V$ ($\mu$W m$^{-3}$)',
                  F=r'$F$ ($10^{-6}$ m$^2$ s$^{-2}$)', Sz=r'$\Sigma_z$ ($10^{-3}$ s$^{-1}$)',
                  cos=r'$\cos\theta_{\mathrm{eff}}$')
CLASS_STYLE = {'anti': dict(ls='dashdot', lw=1.0, label='anticyclonic regions'),
               'cyc': dict(ls='dotted', lw=1.0, label='cyclonic regions')}

# =====================================================================
# LOAD
# =====================================================================

ds = xr.open_dataset(IN_NC)
ST = json.loads(ds.attrs['stats'])
WINDOWS = {k: tuple(ds.attrs[k]) for k in ('T1', 'T2', 'T3', 'T4') if k in ds.attrs}
LANDFALL = np.datetime64(str(ds.attrs['landfall']))
days = ds['day'].values
zlev = ds['zlev'].values
dist = ds['sec_dist'].values
sec_h = ds['sec_h'].values
stn_dist = float(ds.attrs['station_dist_km'])
STN = str(ds.attrs['station'])


def prof(v, cls, k):
    """The class-mean profile where enough of the class area is present."""
    p = ds[f'prof_{v}_{cls}_{k}'].values.copy()
    p[ds[f'frac_{cls}_{k}'].values < PROF_MIN_FRAC] = np.nan
    return p


def section_xlim():
    wet = np.where(np.isfinite(sec_h) & (sec_h > 0) & np.isfinite(ds[f'sec_PiV_{SECTION_WINDOWS[0]}'].values[-1]))[0]
    x0 = dist[wet[0]]
    deep = np.where((np.arange(dist.size) > wet[0]) & (sec_h > SEC_DEPTH_LIMIT))[0]
    x1 = dist[deep[0]] if deep.size else dist[wet[-1]]
    return x0, x1


# =====================================================================
# PANELS
# =====================================================================

def panel_section(fig, ax, k, tag, left_labels):
    z = ds[f'sec_z_{k}'].values                      # (lev, xi)
    piv = ds[f'sec_PiV_{k}'].values * 1e6
    spd = ds[f'sec_speed_{k}'].values
    mld = ds[f'sec_mld_{k}'].values
    X = np.broadcast_to(dist[None, :], z.shape)
    norm = mcolors.TwoSlopeNorm(vmin=-SEC_SCALE, vcenter=0.0, vmax=SEC_SCALE)
    pm = ax.pcolormesh(X, -z, piv, cmap=CMAP_PI, norm=norm, shading='auto', zorder=0)
    ax.contour(X, -z, spd, levels=SPEED_LEVELS, colors=SPEED_COLOUR, linewidths=0.6, zorder=2)
    ax.plot(dist, mld, color=MLD_COLOUR, lw=1.2, zorder=3)
    ax.fill_between(dist, np.where(np.isfinite(sec_h), sec_h, SEC_Z_MAX), SEC_Z_MAX * 1.05, color=BOTTOM_COLOUR, lw=0, zorder=4)
    ax.plot(dist, np.where(np.isfinite(sec_h), sec_h, np.nan), color='black', lw=0.6, zorder=5)
    ax.axvline(stn_dist, color='black', ls=(0, (4, 2)), lw=0.9, zorder=6)
    x0, x1 = section_xlim()
    ax.set_xlim(x0, x1)
    ax.set_ylim(SEC_Z_MAX, 0)
    ax.set_xlabel('distance along the section (km)')
    if left_labels:
        ax.set_ylabel('depth (m)')
    else:
        plt.setp(ax.get_yticklabels(), visible=False)
    ax.text(0.98, 0.04, f'${k}$', transform=ax.transAxes, fontsize=S.FS['label'], fontweight='bold',
            ha='right', va='bottom', zorder=12)
    S.panel_tag(ax, tag, loc='lower left')
    return pm


def nice(q):
    return float(np.round(q, 1 - int(np.floor(np.log10(max(q, 1e-9))))))


def wide_colorbar(fig, ax_left, ax_right, mappable, label, ticks, height=0.012, gap=0.062):
    pl, pr = ax_left.get_position(), ax_right.get_position()
    x0, x1 = 0.5 * (pl.x0 + pl.x1), 0.5 * (pr.x0 + pr.x1)
    y0 = min(pl.y0, pr.y0) - gap - height
    cax = fig.add_axes([x0, y0, x1 - x0, height])
    cb = fig.colorbar(mappable, cax=cax, orientation='horizontal', extend='both')
    cb.set_ticks(ticks)
    cb.formatter = mpl.ticker.FuncFormatter(lambda v, _: f'{v:g}')
    cb.update_ticks()
    cb.set_label(label, size=S.FS['label'] - 1)
    cax.tick_params(labelsize=S.FS['tick'] - 1)
    cb.outline.set_linewidth(0.6)


def panel_profile(ax, v, tag, left_labels, split=False):
    for k in PROFILE_WINDOWS:
        sp = S.WINDOW[k]
        ax.plot(prof(v, 'South', k) * PROF_UNIT[v], zlev, color=sp['color'], ls=sp['ls'], lw=1.6, zorder=3)
    if split:
        sp = S.WINDOW[SPLIT_WINDOW]
        for cls, cs in CLASS_STYLE.items():
            if f'prof_{v}_{cls}_{SPLIT_WINDOW}' in ds and np.isfinite(prof(v, cls, SPLIT_WINDOW)).any():
                ax.plot(prof(v, cls, SPLIT_WINDOW) * PROF_UNIT[v], zlev,
                        color=sp['color'], ls=cs['ls'], lw=cs['lw'], zorder=2)
    ax.axvline(0.0, color='0.3', lw=0.6, zorder=1)
    ax.axhline(ST['windows'][SPLIT_WINDOW]['mld_south_m'], color=MLD_COLOUR, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.set_ylim(PROF_Z_MAX, 0)
    ax.set_xlabel(PROF_LABEL[v])
    ax.grid(True, ls=':', lw=0.4, alpha=0.5, zorder=0)
    if v == 'cos':
        ax.set_xlim(-1.05, 1.05)
    if left_labels:
        ax.set_ylabel('depth (m)')
    else:
        plt.setp(ax.get_yticklabels(), visible=False)
    S.panel_tag(ax, tag, loc='lower left' if v in ('F', 'Sz') else 'upper right')


def panel_station(fig, ax):
    z = -ds['stn_z'].values
    pi = (ds['stn_PiH'].values + ds['stn_PiV'].values) * 1e6
    scale = STN_SCALE if STN_SCALE is not None else nice(float(np.nanquantile(np.abs(pi[np.isfinite(pi)]), 0.98)))
    norm = mcolors.TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale)
    x = days + np.timedelta64(12, 'h')
    pm = ax.pcolormesh(x, z, pi, cmap=CMAP_PI, norm=norm, shading='nearest', zorder=0)
    for k in WINDOWS:
        a, b = np.datetime64(WINDOWS[k][0]), np.datetime64(WINDOWS[k][1]) + np.timedelta64(1, 'D')
        m = float(ds.attrs[f'stn_mld_{k}'])
        ax.plot([a, b], [m, m], color=MLD_COLOUR, lw=1.2, zorder=3)
        ax.text(a + np.timedelta64(3, 'D') + np.timedelta64(12, 'h'), 0.97, f'${k}$', transform=ax.get_xaxis_transform(),
                fontsize=S.FS['small'], color=S.WINDOW[k]['color'], ha='center', va='top', zorder=4)
        ax.axvline(b, color='0.5', lw=0.4, zorder=2)
    ax.axvline(LANDFALL, color='black', ls=(0, (4, 2)), lw=0.9, zorder=3)
    ax.set_ylim(STN_Z_MAX if STN_Z_MAX is not None else float(ds.attrs['station_h']), 0)
    ax.set_xlim(days[0], days[-1] + np.timedelta64(1, 'D'))
    ax.set_ylabel('depth (m)')
    S.date_axis(ax, interval=4, label='2004')
    ext = S.extend_for(pi, -scale, scale)
    S.colorbar_outside(fig, ax, pm, rf'$\Pi$ at station {STN} ($\mu$W m$^{{-3}}$)', extend=ext, where='right',
                       size=0.012, pad=0.012, ticks=[-scale, -scale / 2, 0, scale / 2, scale], fmt=lambda v, _: f'{v:g}')
    S.panel_tag(ax, 'h', loc='lower left')


def write_numbers():
    L = ['FIGURE 5 PLOT NUMBERS', f'section scale +-{SEC_SCALE:g} uW m-3']
    for k in PROFILE_WINDOWS:
        d = ST['windows'][k]
        L.append(f'{k}: PiH extremum {d["PiH_ext_uW_m3"]:+.2f} uW m-3 at {d["PiH_ext_depth_m"]:.0f} m, PiV {d["PiV_ext_uW_m3"]:+.2f} '
                 f'at {d["PiV_ext_depth_m"]:.0f} m, cos {d["cos_at_ext"]:+.2f}, mixed layer {d["mld_south_m"]:.0f} m')
    with open(os.path.join(FIG_DIR, 'fig05_plot_numbers.txt'), 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print('\n'.join(L))


# =====================================================================
# FIGURE
# =====================================================================

def main():
    global SEC_SCALE
    if SEC_SCALE is None:
        allv = np.concatenate([np.abs(ds[f'sec_PiV_{k}'].values[np.isfinite(ds[f'sec_PiV_{k}'].values)]) * 1e6
                               for k in SECTION_WINDOWS])
        SEC_SCALE = nice(float(np.quantile(allv, 0.98)))
    fig = plt.figure(figsize=FIGSIZE)
    g1 = GridSpec(1, 2, figure=fig, left=0.075, right=0.985, top=0.975, bottom=0.755, wspace=0.06)
    g2 = GridSpec(1, 5, figure=fig, left=0.075, right=0.985, top=0.575, bottom=0.355, wspace=0.10)
    g3 = GridSpec(1, 1, figure=fig, left=0.075, right=0.935, top=0.265, bottom=0.06)
    ax_a = fig.add_subplot(g1[0, 0])
    ax_b = fig.add_subplot(g1[0, 1], sharey=ax_a)
    pm = panel_section(fig, ax_a, SECTION_WINDOWS[0], 'a', True)
    panel_section(fig, ax_b, SECTION_WINDOWS[1], 'b', False)
    t = SEC_SCALE
    wide_colorbar(fig, ax_a, ax_b, pm, r'$\Pi_V$ ($\mu$W m$^{-3}$)', [-t, -t / 2, 0, t / 2, t])
    hs = [Line2D([], [], color=SPEED_COLOUR, lw=0.8, label='subinertial speed'),
          Line2D([], [], color=MLD_COLOUR, lw=1.2, label='mixed layer'),
          Line2D([], [], color='black', ls=(0, (4, 2)), lw=0.9, label=f'station {STN}')]
    ax_a.legend(handles=hs, loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=3, borderaxespad=0.0,
                handlelength=2.2, fontsize=S.FS['small'], frameon=False)

    axes = [fig.add_subplot(g2[0, i]) for i in range(5)]
    for ax in axes[1:]:
        ax.sharey(axes[0])
    for i, (v, tag) in enumerate(zip(('PiH', 'PiV', 'F', 'Sz', 'cos'), 'cdefg')):
        panel_profile(axes[i], v, tag, left_labels=(i == 0), split=(v == 'PiV'))
    hs = [Line2D([], [], color=S.WINDOW[k]['color'], ls=S.WINDOW[k]['ls'], lw=1.6, label=f'${k}$') for k in PROFILE_WINDOWS]
    sp = S.WINDOW[SPLIT_WINDOW]
    hs += [Line2D([], [], color=sp['color'], ls=cs['ls'], lw=cs['lw'], label=f'${SPLIT_WINDOW}$ {cs["label"]}')
           for cs in CLASS_STYLE.values()]
    hs.append(Line2D([], [], color=MLD_COLOUR, lw=0.6, ls=(0, (2, 2)), label=f'${SPLIT_WINDOW}$ mixed layer'))
    axes[0].legend(handles=hs, loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=7, borderaxespad=0.0,
                   handlelength=2.4, fontsize=S.FS['small'], frameon=False, columnspacing=1.2)

    ax_h = fig.add_subplot(g3[0, 0])
    panel_station(fig, ax_h)
    write_numbers()
    S.save(fig, OUT_STEM)


if __name__ == '__main__':
    main()