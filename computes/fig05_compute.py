"""
fig05_compute.py

The vertical structure of the exchange (Figure 5, second half of Question 2,
and Table 3). One row-block pass over the South sector rows of the archive,
level by level, reading u, v, w and rho.

The sector profiles are formed on fixed depth levels (ZLEV, every 5 m to
1000 m). Each column's window means are interpolated from the sigma levels to
those depths before the area average, so a profile value at 150 m is the mean
over the columns that reach 150 m. An area-weighted average on sigma levels
would mix the 150 m level of a slope column with the 10 m level of a shelf
column. The fraction of the class area present at each depth is written beside
the profile, and the profile times that fraction integrates to the sector
exchange of Figure 3, which the report checks.

Per level and record the pass forms, as in fig03_compute.py, the horizontal
shear production Pi_H and the near-inertial kinetic energy K on the rho levels,
and on the w levels the vertical shear production Pi_V = -rho0 F . d_z ubar, the
vertical wave Reynolds stress magnitude F = |(<u'w'>, <v'w'>)|, the subinertial
vertical shear Sigma_z = |d_z ubar|, their product F Sigma_z, the near-inertial
shear and the buoyancy flux rho0 <w'b'> with b' = -g rho' / rho0 from the
band-passed density. Window means at every level are accumulated into

    profiles   area-weighted over the South shelf and slope (h <= 1000 m) for
               the whole sector and for its anticyclonic and cyclonic water
               (f_eff / f of the upper 100 m from fig02_data.nc, per window),
               with the orientation factor cos(theta_eff) = -<Pi_V> / <F Sigma_z>
               formed from the means after the pass
    sections   across the slope along the row of the station, per window, with
               the subinertial speed and the mixed-layer depth
    station    daily Pi_H and Pi_V against depth through T4, with the
               mixed-layer depth per window

The mixed-layer depth is the de Boyer Montegut depth where the window-mean
density exceeds the surface value by 0.03 kg m-3. Table 3 (depth and value of
the Pi_H and Pi_V extrema, F, Sigma_z and cos(theta_eff) at the Pi_V extremum)
is written as fig05/table3_vertical.tex, and the share of each window's Pi_V
above and below the sector-mean mixed layer goes to the report.

    nohup python -u fig05_compute.py > fig05/log_fig05.txt 2>&1 &     # about 4 h
    nohup python -u fig05_compute.py --stride 2 > fig05/log_fig05.txt 2>&1 &   # about 1 h
"""

import argparse
import json
import os
import time as _time

import numpy as np
import xarray as xr

import common_v3 as C
from fig02_compute import mean_zeta
from fig03_compute import HALO, cgrid_tensor_t, smooth_h

TAG = 'fig05'
FIG_DIR = C.fig_dir('fig05')
FIG02_NC = os.path.join(os.getcwd(), 'fig02', 'fig02_data.nc')
OUT_NC = os.path.join(FIG_DIR, 'fig05_data.nc')
OUT_JSON = os.path.join(FIG_DIR, 'fig05_stats.json')
OUT_REPORT = os.path.join(FIG_DIR, 'fig05_report.txt')
OUT_TABLE = os.path.join(FIG_DIR, 'table3_vertical.tex')

STATION = 'D'                 # the slope station, from common_v3.STATIONS (subset grid indices)
SECTION_ETA = None            # row of the cross-slope section, None takes the station row
MLD_DRHO = 0.03               # kg m-3, de Boyer Montegut et al. (2004)
Z_MAX = 400.0                 # deepest level kept in the archive of profiles and sections, m
CLASSES = ('South', 'anti', 'cyc')
PVARS = ('PiH', 'PiV', 'K', 'F', 'Sz', 'FSz', 'nish', 'wb')
ZLEV = np.arange(2.5, 1000.0, 5.0)        # depths of the profiles, m, positive down
MIN_AREA_FRAC = 0.05                       # Table 3 extrema are searched where at least this fraction of the class area is present
FIG03_JSON = os.path.join(os.getcwd(), 'fig03', 'fig03_stats.json')
SVARS = ('PiH', 'PiV', 'F', 'Sz', 'FSz', 'nish', 'speed', 'z', 'zi')


def mld_from_profile(rho, z_r, thr):
    """Mixed-layer depth per column from a (nlev, ...) density with the surface
    at the last index, positive metres, the de Boyer Montegut threshold."""
    hit_all = rho > (rho[-1] + thr)[None]
    mld = np.full(rho.shape[1:], np.nan)
    for k in range(rho.shape[0] - 2, -1, -1):
        hit = hit_all[k] & ~np.isfinite(mld)
        mld[hit] = -z_r[k][hit]
    bad = ~np.isfinite(mld)
    mld[bad] = -z_r[0][bad]
    return mld


def to_depth_levels(vals, zr, zlev):
    """Linear interpolation of (nlev, rows, cols) column values at the rho depths
    zr (negative upward) onto the depths zlev (positive down). Above the top
    level the top value is kept, below the bottom level the result is NaN."""
    nlev = vals.shape[0]
    out = np.full((zlev.size,) + vals.shape[1:], np.nan)
    for i, zt in enumerate(-zlev):
        n = (zr <= zt).sum(axis=0)                      # levels at or below the target
        k = np.clip(n - 1, 0, nlev - 2)
        z0 = np.take_along_axis(zr, k[None], 0)[0]
        z1 = np.take_along_axis(zr, (k + 1)[None], 0)[0]
        v0 = np.take_along_axis(vals, k[None], 0)[0]
        v1 = np.take_along_axis(vals, (k + 1)[None], 0)[0]
        w = np.clip((zt - z0) / np.maximum(z1 - z0, 1e-6), 0.0, 1.0)
        val = v0 + w * (v1 - v0)
        val = np.where(n >= nlev, vals[-1], val)
        out[i] = np.where(n == 0, np.nan, val)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--eta-chunk', type=int, default=24)
    ap.add_argument('--workers', type=int, default=-1)
    args = ap.parse_args()
    st = args.stride
    t_start = _time.time()
    smooth = max(1, int(round(C.SMOOTH_CELLS / st)))

    arc = C.Archive()
    arc.describe(TAG)
    arc.assert_even(TAG)
    for v in ('w', 'rho'):
        if not arc.has(v):
            raise SystemExit(f'{v} is not in the archive')
    g = C.load_grid(arc, stride=st)
    lon, lat, h, mask = g['lon'], g['lat'], g['h'], g['mask']
    pm, pn, land, area, f2d = g['pm'], g['pn'], g['land'], g['area'], g['f']
    neta_all, nxi = lon.shape
    nt, nlev = arc.nt, arc.nlev
    masks = C.sector_masks(g, stride=st)
    shelf = masks['shelf']['South']
    eta_stop = int(np.where(shelf.any(axis=1))[0].max()) + 1
    stn = C.STATIONS[STATION]
    stn_row = int(round((stn['eta'] - C.ETA0) / st))
    stn_col = int(round((stn['xi'] - C.XI0) / st))
    sec_row = stn_row if SECTION_ETA is None else int(round((SECTION_ETA - C.ETA0) / st))
    if not (0 <= stn_row < eta_stop and 0 <= stn_col < nxi):
        raise SystemExit(f'station {STATION} row {stn_row} col {stn_col} outside the South rows')
    C.report(TAG, f'South rows 0 to {eta_stop} of {neta_all} x {nxi} x {nlev} (stride {st}), {nt} records, '
                  f'station {STATION} at row {stn_row} col {stn_col} (h {h[stn_row, stn_col]:.0f} m, '
                  f'{lat[stn_row, stn_col]:.2f} N {lon[stn_row, stn_col]:.2f} E), section row {sec_row}')

    fb = C.FilterBank(lat, land, arc.dt_h, nt, workers=args.workers)
    fb.attach_time(arc.time)
    fb.describe(TAG)
    valid = fb.valid
    idx = C.window_indices(arc.time, valid)
    wsl = {k: C.as_slice(v) for k, v in idx.items()}
    days = np.unique(arc.time[valid].astype('datetime64[D]'))
    day_of = arc.time.astype('datetime64[D]')
    dsl = [C.as_slice(np.where((day_of == d) & valid)[0]) for d in days]

    C.report(TAG, 'time-mean free surface and the vertical grid')
    sel_z = dict(eta_rho=C.ETA_SLICE, xi_rho=C.XI_SLICE)
    zeta_m = mean_zeta(arc, st, sel_z, nt)
    z_w = C.compute_z_w(arc.ds0, zeta_m, st, h=h)
    z_r = 0.5 * (z_w[:-1] + z_w[1:])
    Hz = np.nan_to_num(np.diff(z_w, axis=0), nan=0.0)

    # anticyclonic and cyclonic water per window, from the Figure 2 tensor
    anti = {}
    if os.path.exists(FIG02_NC):
        with xr.open_dataset(FIG02_NC) as d2:
            sub = st // int(d2.attrs['stride'])
            for k in C.WKEYS:
                if f'feff_{k}' in d2:
                    fe = (d2[f'feff_{k}'].values / d2['f'].values)[::sub, ::sub]
                    if fe.shape == (neta_all, nxi):
                        anti[k] = fe < 1.0
        C.report(TAG, f'anticyclonic masks from fig02_data.nc for {sorted(anti)}')
    else:
        C.report(TAG, 'fig02_data.nc absent, no anticyclonic and cyclonic split')

    def class_mask(cls, k, j0, j1):
        m = shelf[j0:j1, :]
        if cls == 'South' or k not in anti:
            return m if cls == 'South' else np.zeros_like(m)
        a = anti[k][j0:j1, :]
        return m & (a if cls == 'anti' else ~a)

    nz = ZLEV.size
    prof = {k: {c: {v: np.zeros(nz) for v in PVARS} for c in CLASSES} for k in C.WKEYS}
    parea = {k: {c: np.zeros(nz) for c in CLASSES} for k in C.WKEYS}
    carea = {k: {c: 0.0 for c in CLASSES} for k in C.WKEYS}
    mld_acc = {k: [0.0, 0.0] for k in C.WKEYS}
    sec = {k: {v: np.full((nlev, nxi), np.nan) for v in SVARS} for k in C.WKEYS}
    sec_mld = {k: np.full(nxi, np.nan) for k in C.WKEYS}
    stn_PiH = np.full((nlev, days.size), np.nan)
    stn_PiV = np.full((nlev, days.size), np.nan)
    stn_z = np.full(nlev, np.nan)
    stn_mld = {k: np.nan for k in C.WKEYS}
    sec_dens = {k: np.zeros((nlev, nxi)) for k in C.WKEYS}
    stn_dens = {k: np.zeros(nlev) for k in C.WKEYS}

    def read_block(var, de, dx, r0, r1, lev):
        sel = {de: slice(C.ETA0 + r0 * st, C.ETA0 + r1 * st, st), dx: slice(C.XI0, None)}
        sel['s_rho' if 's_rho' in arc.ds0[var].dims else 's_w'] = lev
        a = arc.read(var, sel)[..., ::st]
        return a[..., :nxi - 1] if dx == 'xi_u' else a[..., :nxi]

    nchunk = int(np.ceil(eta_stop / args.eta_chunk))
    t0 = _time.time()
    for ic, j0 in enumerate(range(0, eta_stop, args.eta_chunk)):
        tb = _time.time()
        j1 = min(j0 + args.eta_chunk, eta_stop)
        nrow = j1 - j0
        a0, a1 = max(j0 - HALO, 0), min(j1 + HALO, neta_all)
        b0, b1 = a0, min(j1 + 1, neta_all - 1)
        s0 = j0 - a0
        pmL, pnL = pm[a0:a1, :], pn[a0:a1, :]
        has_sec, has_stn = j0 <= sec_row < j1, j0 <= stn_row < j1
        dens_w = {k: np.zeros((nlev, nrow, nxi)) for k in C.WKEYS}
        wm = {k: {v: np.zeros((nlev, nrow, nxi)) for v in PVARS} for k in C.WKEYS}   # window means per level
        prev = None
        for kk in range(nlev):
            tl = _time.time()
            fu = fb.apply_both(read_block('u', 'eta_u', 'xi_u', a0, a1, kk), a0)
            fv = fb.apply_both(read_block('v', 'eta_v', 'xi_v', b0, b1, kk), b0)
            dlt, Sng, _, Ssg = cgrid_tensor_t(
                np.nan_to_num(smooth_h(fu['lp'], smooth), nan=0.0),
                np.nan_to_num(smooth_h(fv['lp'], smooth), nan=0.0), pmL, pnL)
            dlt, Sng, Ssg = (a[:, s0:s0 + nrow, :] for a in (dlt, Sng, Ssg))
            upr = C.u2rho(fu['bp'])[:, s0:s0 + nrow, :]
            vpr = C.v2rho(fv['bp'])[:, s0:s0 + nrow, :]
            ulr = C.u2rho(fu['lp'])[:, s0:s0 + nrow, :]
            vlr = C.v2rho(fv['lp'])[:, s0:s0 + nrow, :]
            del fu, fv
            pa = lambda a: fb.phase_average(a, j0)
            uu, vv, uv = pa(upr * upr), pa(vpr * vpr), pa(upr * vpr)
            PiH = -C.RHO0 * (0.5 * (uu - vv) * Sng + uv * Ssg + 0.5 * (uu + vv) * dlt)
            K = 0.5 * C.RHO0 * (uu + vv)
            del uu, vv, uv, dlt, Sng, Ssg
            rho = read_block('rho', 'eta_rho', 'xi_rho', j0, j1, kk)
            for k in C.WKEYS:
                dens_w[k][kk] = rho[wsl[k]].mean(axis=0)
            bp = -(C.G / C.RHO0) * fb.apply(rho, j0, 'bp')
            del rho
            zero = np.zeros_like(PiH)
            PiV, F, Sz, FSz, nish, wb = zero, zero, zero, zero, zero, zero
            if kk > 0:
                up_p, vp_p, ul_p, vl_p, bp_p = prev
                wp = fb.apply(read_block('w', 'eta_rho', 'xi_rho', j0, j1, kk), j0, 'bp')
                dz = np.maximum(0.5 * (Hz[kk, j0:j1, :] + Hz[kk - 1, j0:j1, :]), 1e-3)[None]
                uw, vw = pa(0.5 * (upr + up_p) * wp), pa(0.5 * (vpr + vp_p) * wp)
                shu, shv = (ulr - ul_p) / dz, (vlr - vl_p) / dz
                PiV = -C.RHO0 * (uw * shu + vw * shv)
                F = np.hypot(uw, vw)
                Sz = np.hypot(shu, shv)
                FSz = F * Sz
                nish = np.sqrt(np.maximum(pa(((upr - up_p) / dz) ** 2 + ((vpr - vp_p) / dz) ** 2), 0.0))
                wb = C.RHO0 * pa(wp * 0.5 * (bp + bp_p))
                del wp, uw, vw, shu, shv
            fields = dict(PiH=PiH, PiV=PiV, K=K, F=F, Sz=Sz, FSz=FSz, nish=nish, wb=wb)
            for k in C.WKEYS:
                s = wsl[k]
                m_ = {v: a[s].mean(axis=0) for v, a in fields.items()}
                for v, fld in m_.items():
                    wm[k][v][kk] = fld
                if has_sec:
                    r = sec_row - j0
                    for v in ('PiH', 'PiV', 'F', 'Sz', 'FSz', 'nish'):
                        sec[k][v][kk] = m_[v][r]
                    sec[k]['speed'][kk] = np.hypot(ulr[s].mean(axis=0), vlr[s].mean(axis=0))[r]
                    sec[k]['z'][kk], sec[k]['zi'][kk] = z_r[kk, sec_row, :], z_w[kk, sec_row, :]
            if has_stn:
                r = stn_row - j0
                for d, s in enumerate(dsl):
                    stn_PiH[kk, d] = PiH[s, r, stn_col].mean()
                    stn_PiV[kk, d] = PiV[s, r, stn_col].mean()
                stn_z[kk] = z_r[kk, stn_row, stn_col]
            prev = (upr, vpr, ulr, vlr, bp)
            del fields, PiH, PiV, K, F, Sz, FSz, nish, wb
            if ic == 0 and (kk % 5 == 0 or kk == nlev - 1):
                per = (_time.time() - tb) / (kk + 1)
                C.report(TAG, f'    block 1, level {kk + 1:2d}/{nlev}, {_time.time() - tl:5.1f} s, whole pass about '
                              f'{per * nlev * nchunk / 3600:.1f} h, finish near '
                              f'{_time.strftime("%d %b %H:%M", _time.localtime(tb + per * nlev * nchunk))}')
        del prev
        # the window means of the block onto the depth levels, then the class sums
        zrb = z_r[:, j0:j1, :]
        for k in C.WKEYS:
            for v in PVARS:
                vz = to_depth_levels(wm[k][v], zrb, ZLEV)
                for c in CLASSES:
                    cm = class_mask(c, k, j0, j1)
                    if not cm.any():
                        continue
                    aa = np.where(cm, area[j0:j1, :], 0.0)[None]
                    good = np.isfinite(vz)
                    prof[k][c][v] += np.nansum(np.where(good, vz, 0.0) * aa, axis=(1, 2))
                    if v == PVARS[0]:
                        parea[k][c] += np.nansum(np.where(good, aa, 0.0), axis=(1, 2))
                        carea[k][c] += float(np.nansum(aa))
                del vz
        del wm
        for k in C.WKEYS:
            ml = mld_from_profile(dens_w[k], z_r[:, j0:j1, :], MLD_DRHO)
            ml[land[j0:j1, :]] = np.nan
            if has_sec:
                sec_mld[k] = ml[sec_row - j0]
                sec_dens[k] = dens_w[k][:, sec_row - j0, :]
            if has_stn:
                stn_mld[k] = float(ml[stn_row - j0, stn_col])
                stn_dens[k] = dens_w[k][:, stn_row - j0, stn_col]
            cm = shelf[j0:j1, :]
            mld_acc[k][0] += float(np.nansum(np.where(cm, ml * area[j0:j1, :], 0.0)))
            mld_acc[k][1] += float(np.nansum(np.where(cm & np.isfinite(ml), area[j0:j1, :], 0.0)))
        del dens_w
        el = _time.time() - t0
        C.report(TAG, f'  block {ic + 1:2d}/{nchunk}, rows {j0:4d}-{j1:4d}, {(_time.time() - tb) / 60:.1f} min, '
                      f'elapsed {el / 60:.0f} min, about {el / (ic + 1) * (nchunk - ic - 1) / 60:.0f} min left, '
                      f'finish near {_time.strftime("%d %b %H:%M", _time.localtime(_time.time() + el / (ic + 1) * (nchunk - ic - 1)))}')

    # ---- reduce -------------------------------------------------------------------------
    P, FRAC = {}, {}
    for k in C.WKEYS:
        for c in CLASSES:
            A = parea[k][c]
            FRAC[(k, c)] = A / carea[k][c] if carea[k][c] > 0 else np.full(nz, np.nan)
            for v in PVARS:
                arr = np.full(nz, np.nan)
                gd = A > 0
                arr[gd] = prof[k][c][v][gd] / A[gd]
                P[(k, c, v)] = arr
            with np.errstate(invalid='ignore', divide='ignore'):
                P[(k, c, 'cos')] = np.where(P[(k, c, 'FSz')] > 0, -P[(k, c, 'PiV')] / (C.RHO0 * P[(k, c, 'FSz')]), np.nan)
    mld_south = {k: mld_acc[k][0] / mld_acc[k][1] for k in C.WKEYS}
    for k in C.WKEYS:
        with np.errstate(invalid='ignore', divide='ignore'):
            sec[k]['cos'] = np.where(sec[k]['FSz'] > 0, -sec[k]['PiV'] / (C.RHO0 * sec[k]['FSz']), np.nan)
    la, lo = np.deg2rad(lat[sec_row]), np.deg2rad(lon[sec_row])
    dd = 2 * 6371.0 * np.arcsin(np.sqrt(np.sin(np.diff(la) / 2) ** 2 + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(np.diff(lo) / 2) ** 2))
    sec_dist = np.concatenate([[0.0], np.cumsum(dd)])
    stn_dist = float(sec_dist[stn_col])
    dzl = float(ZLEV[1] - ZLEV[0])
    fig03 = None
    if os.path.exists(FIG03_JSON):
        with open(FIG03_JSON) as fh:
            fig03 = json.load(fh)['table']['South']

    # ---- Table 3 and the layer shares -------------------------------------------------------
    stats = {'config': dict(stride=st, station=STATION, station_row=stn_row, station_col=stn_col,
                            station_h=float(h[stn_row, stn_col]), station_lat=float(lat[stn_row, stn_col]),
                            station_lon=float(lon[stn_row, stn_col]), section_row=sec_row, mld_drho=MLD_DRHO,
                            z_max=Z_MAX, zlev_step_m=dzl, min_area_frac=MIN_AREA_FRAC, smooth_cells=smooth,
                            filters=fb.config(), sign='positive from the subinertial flow to the waves',
                            cos='cos(theta_eff) = -<Pi_V> / (rho0 <F Sigma_z>), negative gives Pi_V > 0',
                            profiles='means over the columns present at each depth, times the area fraction '
                                     'they integrate to the sector exchange'),
             'windows': {}}
    for k in C.WKEYS:
        fr = FRAC[(k, 'South')]
        inz = (ZLEV <= Z_MAX) & (fr >= MIN_AREA_FRAC)
        d = dict(mld_south_m=mld_south[k], mld_station_m=stn_mld[k],
                 depth_of_half_area_m=float(ZLEV[np.where(fr < 0.5)[0][0]]) if (fr < 0.5).any() else np.nan,
                 depth_of_min_area_frac_m=float(ZLEV[np.where(fr < MIN_AREA_FRAC)[0][0]]) if (fr < MIN_AREA_FRAC).any() else np.nan)
        for v in ('PiH', 'PiV'):
            p = P[(k, 'South', v)]
            j = int(np.nanargmax(np.where(inz, np.abs(p), np.nan)))
            d[f'{v}_ext_depth_m'] = float(ZLEV[j])
            d[f'{v}_ext_uW_m3'] = float(p[j] * 1e6)
            if v == 'PiV':
                d['F_at_ext_1e-6_m2_s-2'] = float(P[(k, 'South', 'F')][j] * 1e6)
                d['Sz_at_ext_1e-3_s-1'] = float(P[(k, 'South', 'Sz')][j] * 1e3)
                d['cos_at_ext'] = float(P[(k, 'South', 'cos')][j])
                d['nish_at_ext_1e-3_s-1'] = float(P[(k, 'South', 'nish')][j] * 1e3)
        above = ZLEV <= mld_south[k]
        for v in ('PiH', 'PiV', 'wb'):
            contrib = np.nan_to_num(P[(k, 'South', v)] * fr, nan=0.0) * dzl
            d[f'{v}_int_mW_m2'] = float(contrib.sum() * 1e3)
            d[f'{v}_int_above_mld_mW_m2'] = float(contrib[above].sum() * 1e3)
            d[f'{v}_int_below_mld_mW_m2'] = float(contrib[~above].sum() * 1e3)
            if fig03 and v in fig03[k]:
                d[f'{v}_int_over_fig03'] = d[f'{v}_int_mW_m2'] / (fig03[k][v] * 1e3) if fig03[k][v] else np.nan
        for c in ('anti', 'cyc'):
            frc = FRAC[(k, c)]
            pa_ = P[(k, c, 'PiV')]
            # the class integral relative to the whole South area, so the two classes add to the sector value
            contrib = np.nan_to_num(pa_ * frc, nan=0.0) * dzl * (carea[k][c] / carea[k]['South'] if carea[k]['South'] else np.nan)
            d[f'PiV_int_{c}_mW_m2'] = float(contrib.sum() * 1e3)
            okc = (ZLEV <= Z_MAX) & (frc >= MIN_AREA_FRAC) & np.isfinite(pa_)
            if okc.any():
                j = int(np.nanargmax(np.where(okc, np.abs(pa_), np.nan)))
                d[f'PiV_ext_depth_{c}_m'] = float(ZLEV[j])
                d[f'PiV_ext_{c}_uW_m3'] = float(pa_[j] * 1e6)
        stats['windows'][k] = d

    L = ['FIGURE 5 REPORT, numbers for the text and Table 3', f'run {_time.strftime("%Y-%m-%d %H:%M")}',
         f'South shelf and slope h <= {C.STAT_MAX_DEPTH_M:.0f} m, area weighted profiles, stride {st}, '
         f'station {STATION} (h {h[stn_row, stn_col]:.0f} m) and its row as the section, positive from the subinertial flow to the waves', '']
    L.append('Table 3, extrema of the sector-mean profiles above {:.0f} m where at least {:.0f} % of the sector area is present'.format(Z_MAX, 100 * MIN_AREA_FRAC))
    L.append(f'  {"":34s}' + ''.join(f'{k:>10s}' for k in C.WKEYS))
    rows = [('PiH extremum depth (m)', 'PiH_ext_depth_m', '{:10.0f}'), ('PiH extremum (uW m-3)', 'PiH_ext_uW_m3', '{:+10.2f}'),
            ('PiV extremum depth (m)', 'PiV_ext_depth_m', '{:10.0f}'), ('PiV extremum (uW m-3)', 'PiV_ext_uW_m3', '{:+10.2f}'),
            ('F at the PiV extremum (1e-6 m2 s-2)', 'F_at_ext_1e-6_m2_s-2', '{:10.2f}'),
            ('Sigma_z there (1e-3 s-1)', 'Sz_at_ext_1e-3_s-1', '{:10.2f}'), ('cos theta_eff there', 'cos_at_ext', '{:+10.2f}'),
            ('near-inertial shear there (1e-3 s-1)', 'nish_at_ext_1e-3_s-1', '{:10.2f}'),
            ('mixed layer, sector mean (m)', 'mld_south_m', '{:10.0f}'), ('mixed layer at the station (m)', 'mld_station_m', '{:10.0f}'),
            ('depth where half the area remains (m)', 'depth_of_half_area_m', '{:10.0f}')]
    for name, key, fmt in rows:
        L.append(f'  {name:34s}' + ''.join(fmt.format(stats['windows'][k][key]) for k in C.WKEYS))
    L.append('')
    L.append('depth integrals of profile x area fraction, mW m-2 (whole column, above / below the sector-mean mixed layer)')
    for v in ('PiH', 'PiV', 'wb'):
        L.append(f'  {v:4s}' + ''.join(f'  {k} {stats["windows"][k][f"{v}_int_mW_m2"]:+.3f} ({stats["windows"][k][f"{v}_int_above_mld_mW_m2"]:+.3f} / {stats["windows"][k][f"{v}_int_below_mld_mW_m2"]:+.3f})' for k in C.WKEYS))
    if fig03:
        L.append('  check against the Figure 3 sector means (ratio, 1 means the two passes agree): ' + ', '.join(
            f'{k} PiH {stats["windows"][k].get("PiH_int_over_fig03", np.nan):.2f} PiV {stats["windows"][k].get("PiV_int_over_fig03", np.nan):.2f}'
            for k in C.WKEYS))
    if anti:
        L.append('')
        L.append('Pi_V in anticyclonic and cyclonic regions (depth integral over the South area mW m-2, the two add to the sector value, extremum depth m and value uW m-3)')
        for k in C.WKEYS:
            d = stats['windows'][k]
            L.append(f'  {k}: anticyclonic {d.get("PiV_int_anti_mW_m2", np.nan):+.3f} at {d.get("PiV_ext_depth_anti_m", np.nan):.0f} m '
                     f'({d.get("PiV_ext_anti_uW_m3", np.nan):+.2f}), cyclonic {d.get("PiV_int_cyc_mW_m2", np.nan):+.3f} at '
                     f'{d.get("PiV_ext_depth_cyc_m", np.nan):.0f} m ({d.get("PiV_ext_cyc_uW_m3", np.nan):+.2f})')
    L.append('')
    L.append('amplitude factors against T1 at the depth of each window\'s PiV extremum: F ratio, Sigma_z ratio')
    f1 = stats['windows']['T1']['F_at_ext_1e-6_m2_s-2']
    s1 = stats['windows']['T1']['Sz_at_ext_1e-3_s-1']
    L.append('  ' + ', '.join(f'{k} {stats["windows"][k]["F_at_ext_1e-6_m2_s-2"] / f1:.1f} and {stats["windows"][k]["Sz_at_ext_1e-3_s-1"] / s1:.1f}'
                              for k in C.WKEYS if k != 'T1'))
    L += ['', f'wall time {(_time.time() - t_start) / 3600:.2f} h']
    C.write_report(OUT_REPORT, L)
    print('\n'.join(L))

    tex = [r'\begin{table}[H]', r'\centering',
           r'\caption{Depth and extremum value of the two shear productions in the South sector, area weighted over the '
           r'shelf and slope shallower than 1000~m, with the amplitude factors and the orientation factor of the vertical '
           r'shear production (Equation~\ref{eq:alignv}) at the depth of its extremum, and the mixed-layer depth of the sector.}',
           r'\label{tab:vertical}', r'\begin{tabular}{lrrrr}', r'\toprule', r' & $T1$ & $T2$ & $T3$ & $T4$ \\', r'\midrule']
    trows = [(r'$\PiH$ extremum depth (m)', 'PiH_ext_depth_m', '{:.0f}'), (r'$\PiH$ extremum value ($\mu$W~m$^{-3}$)', 'PiH_ext_uW_m3', '{:+.2f}'),
             (r'$\PiV$ extremum depth (m)', 'PiV_ext_depth_m', '{:.0f}'), (r'$\PiV$ extremum value ($\mu$W~m$^{-3}$)', 'PiV_ext_uW_m3', '{:+.2f}'),
             (r'$F$ ($10^{-6}$~m$^{2}$~s$^{-2}$)', 'F_at_ext_1e-6_m2_s-2', '{:.2f}'), (r'$\Sigma_z$ ($10^{-3}$~s$^{-1}$)', 'Sz_at_ext_1e-3_s-1', '{:.2f}'),
             (r'$\cos\theta_{\mathrm{eff}}$', 'cos_at_ext', '{:+.2f}'), (r'mixed-layer depth (m)', 'mld_south_m', '{:.0f}')]
    for name, key, fmt in trows:
        tex.append(f'{name} & ' + ' & '.join(fmt.format(stats['windows'][k][key]) for k in C.WKEYS) + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    with open(OUT_TABLE, 'w') as fh:
        fh.write('\n'.join(tex) + '\n')

    # ---- archive ---------------------------------------------------------------------------------
    data = {}
    for (k, c, v), arr in P.items():
        data[f'prof_{v}_{c}_{k}'] = (('zlev',), arr)
    for (k, c), arr in FRAC.items():
        data[f'frac_{c}_{k}'] = (('zlev',), arr)
    for k in C.WKEYS:
        for v in list(SVARS) + ['cos']:
            data[f'sec_{v}_{k}'] = (('lev', 'xi'), sec[k][v])
        data[f'sec_mld_{k}'] = (('xi',), sec_mld[k])
        data[f'sec_rho_{k}'] = (('lev', 'xi'), sec_dens[k])
        data[f'stn_rho_{k}'] = (('lev',), stn_dens[k])
    data['sec_dist'] = (('xi',), sec_dist)
    data['sec_h'] = (('xi',), h[sec_row])
    data['sec_lon'] = (('xi',), lon[sec_row])
    data['sec_lat'] = (('xi',), lat[sec_row])
    data['stn_PiH'] = (('lev', 'day'), stn_PiH)
    data['stn_PiV'] = (('lev', 'day'), stn_PiV)
    data['stn_z'] = (('lev',), stn_z)
    out = xr.Dataset(data, coords=dict(day=days.astype('datetime64[ns]'), zlev=ZLEV),
                     attrs=dict(source=' + '.join(arc.paths), stride=st, rho0=C.RHO0, station=STATION,
                                station_row=stn_row, station_col=stn_col, station_dist_km=stn_dist,
                                station_h=float(h[stn_row, stn_col]), station_lon=float(lon[stn_row, stn_col]),
                                station_lat=float(lat[stn_row, stn_col]), section_row=sec_row,
                                **{f'stn_mld_{k}': float(stn_mld[k]) for k in C.WKEYS},
                                **{f'mld_south_{k}': float(mld_south[k]) for k in C.WKEYS},
                                **{k: list(v) for k, v in C.WINDOWS.items()}, landfall=str(C.LANDFALL_DEFAULT)[:16],
                                z_max=Z_MAX, mld_drho=MLD_DRHO, classes=list(CLASSES), filters=json.dumps(fb.config()),
                                stats=json.dumps(stats, default=float),
                                note=('Window-mean profiles on the depth levels zlev (W m-3 for PiH, PiV and wb, J m-3 '
                                      'for K, m2 s-2 for F, s-1 for Sz and nish, m3 s-3 for FSz), the mean over the '
                                      'columns of the class present at each depth, with frac the fraction of the class '
                                      'area present there, for the South shelf and slope and its anticyclonic and '
                                      'cyclonic regions. Sections along the station row and daily station profiles stay '
                                      'on the sigma levels with their own depths.'),
                                created=_time.strftime('%Y-%m-%d %H:%M')))
    out.to_netcdf(OUT_NC)
    with open(OUT_JSON, 'w') as fh:
        json.dump(stats, fh, indent=2, default=float)
    print(f'wrote {OUT_NC}, {OUT_JSON} and {OUT_TABLE}')
    arc.close()


if __name__ == '__main__':
    main()