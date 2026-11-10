"""
fig03_piv_check.py

Robustness of the South sector exchange to the filter. The storm injects the
waves impulsively, so the band pass leaks part of the burst into the cross terms
(8 % of the South T2 kinetic energy in Table 1), and the vertical shear production
that carries the T2 drain must not depend on where the band is drawn. The South
rows are recomputed at a reduced stride with several variants of the band and of
the phase-average length, and the window means of Pi_H, Pi_V and Pi and the daily
South series are written for each variant.

Variants (name, band in f, phase average in inertial periods):
    base      [0.8, 1.2]   1      the Figure 3 definition
    narrow    [0.85, 1.15] 1
    wide      [0.8, 1.4]   1      admits the blue-shifted forced response
    long_pa   [0.8, 1.2]   2

    nohup python -u fig03_piv_check.py --stride 3 > fig03/log_piv_check.txt 2>&1 &
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

TAG = 'pivchk'
FIG_DIR = C.fig_dir('fig03')
OUT_NC = os.path.join(FIG_DIR, 'fig03_piv_check.nc')
OUT_TXT = os.path.join(FIG_DIR, 'fig03_piv_check.txt')
FIG03_JSON = os.path.join(FIG_DIR, 'fig03_stats.json')
VARIANTS = {'base': ((0.8, 1.2), 1.0), 'narrow': ((0.85, 1.15), 1.0),
            'wide': ((0.8, 1.4), 1.0), 'long_pa': ((0.8, 1.2), 2.0)}
SECTOR = 'South'


def bank_for(band, pa, lat, land, dt_h, nt, time, workers):
    """A FilterBank built with the module constants temporarily replaced."""
    band0, pa0 = C.NI_BAND, C.PHASE_AVG_PERIODS
    C.NI_BAND, C.PHASE_AVG_PERIODS = tuple(band), float(pa)
    try:
        fb = C.FilterBank(lat, land, dt_h, nt, workers=workers)
    finally:
        C.NI_BAND, C.PHASE_AVG_PERIODS = band0, pa0
    fb.attach_time(time)
    return fb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--eta-chunk', type=int, default=40)
    ap.add_argument('--workers', type=int, default=-1)
    args = ap.parse_args()
    st = args.stride
    t_start = _time.time()
    smooth = max(1, int(round(C.SMOOTH_CELLS / st)))

    arc = C.Archive()
    arc.describe(TAG)
    arc.assert_even(TAG)
    if not arc.has('w'):
        raise SystemExit('w is not in the archive')
    g = C.load_grid(arc, stride=st)
    lon, lat, h, mask = g['lon'], g['lat'], g['h'], g['mask']
    pm, pn, land, area = g['pm'], g['pn'], g['land'], g['area']
    masks = C.sector_masks(g, stride=st)
    sec_mask = masks['shelf'][SECTOR]
    neta_all, nxi = lon.shape
    neta = int(np.where(sec_mask.any(axis=1))[0].max()) + 1       # rows that hold the sector
    nt, nlev = arc.nt, arc.nlev
    C.report(TAG, f'{SECTOR} rows 0 to {neta} of {neta_all} x {nxi} x {nlev} (stride {st}), {nt} records')

    banks = {v: bank_for(b, p, lat, land, arc.dt_h, nt, arc.time, args.workers) for v, (b, p) in VARIANTS.items()}
    for v, fb in banks.items():
        C.report(TAG, f'variant {v}: band {VARIANTS[v][0]} f, phase average {VARIANTS[v][1]:g} Tf, '
                      f'edge {fb.edge} samples, valid {str(fb.valid_span()[0])[:13]} to {str(fb.valid_span()[1])[:13]}')
    idx = {v: C.window_indices(arc.time, fb.valid) for v, fb in banks.items()}
    wsl = {v: {k: C.as_slice(i) for k, i in d.items()} for v, d in idx.items()}
    days = np.unique(arc.time[banks['base'].valid].astype('datetime64[D]'))
    day_of = arc.time.astype('datetime64[D]')
    dsl = {v: [C.as_slice(np.where((day_of == d) & fb.valid)[0]) for d in days] for v, fb in banks.items()}

    sel_z = dict(eta_rho=C.ETA_SLICE, xi_rho=C.XI_SLICE)
    Hz = np.nan_to_num(C.compute_Hz(arc.ds0, mean_zeta(arc, st, sel_z, nt), st, h=h), nan=0.0)

    terms = ('PiH', 'PiV')
    W = {v: {t: {k: np.zeros((neta, nxi)) for k in C.WKEYS} for t in terms} for v in VARIANTS}
    D = {v: {t: np.zeros((days.size, neta, nxi)) for t in terms} for v in VARIANTS}

    def read_block(var, de, dx, r0, r1, lev):
        # rows are strided in the read (each row is one contiguous run in the file) and the
        # columns are subsampled in memory, which a strided column read would make slow
        sel = {de: slice(C.ETA0 + r0 * st, C.ETA0 + r1 * st, st), dx: slice(C.XI0, None)}
        sel['s_rho' if 's_rho' in arc.ds0[var].dims else 's_w'] = lev
        a = arc.read(var, sel)[..., ::st]
        return a[..., :nxi - 1] if dx == 'xi_u' else a[..., :nxi]

    nchunk = int(np.ceil(neta / args.eta_chunk))
    t0 = _time.time()
    for ic, j0 in enumerate(range(0, neta, args.eta_chunk)):
        tb = _time.time()
        j1 = min(j0 + args.eta_chunk, neta)
        nrow = j1 - j0
        a0, a1 = max(j0 - HALO, 0), min(j1 + HALO, neta_all)
        b0, b1 = a0, min(j1 + 1, neta_all - 1)
        s0 = j0 - a0
        pmL, pnL = pm[a0:a1, :], pn[a0:a1, :]
        acc_w = {v: {t: {k: np.zeros((nrow, nxi)) for k in C.WKEYS} for t in terms} for v in VARIANTS}
        acc_d = {v: {t: np.zeros((days.size, nrow, nxi)) for t in terms} for v in VARIANTS}
        prev = {v: None for v in VARIANTS}
        for kk in range(nlev):
            tl = _time.time()
            u = read_block('u', 'eta_u', 'xi_u', a0, a1, kk)
            v_ = read_block('v', 'eta_v', 'xi_v', b0, b1, kk)
            w = read_block('w', 'eta_rho', 'xi_rho', j0, j1, kk) if kk > 0 else None
            hz = Hz[kk, j0:j1, :]
            dz = np.maximum(0.5 * (Hz[kk, j0:j1, :] + Hz[kk - 1, j0:j1, :]), 1e-3) if kk > 0 else None
            for vn, fb in banks.items():
                fu = fb.apply_both(u, a0)
                fv = fb.apply_both(v_, b0)
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
                PiH = -C.RHO0 * (0.5 * pa(upr * upr - vpr * vpr) * Sng + pa(upr * vpr) * Ssg
                                 + 0.5 * pa(upr * upr + vpr * vpr) * dlt)
                for k in C.WKEYS:
                    acc_w[vn]['PiH'][k] += PiH[wsl[vn][k]].mean(axis=0) * hz
                for d, s in enumerate(dsl[vn]):
                    acc_d[vn]['PiH'][d] += PiH[s].mean(axis=0) * hz
                del PiH, dlt, Sng, Ssg
                if kk > 0:
                    up_prev, vp_prev, ub_prev, vb_prev = prev[vn]
                    wp = fb.apply(w, j0, 'bp')
                    PiV = -C.RHO0 * (pa(0.5 * (upr + up_prev) * wp) * (ulr - ub_prev) / dz[None]
                                     + pa(0.5 * (vpr + vp_prev) * wp) * (vlr - vb_prev) / dz[None])
                    del wp
                    for k in C.WKEYS:
                        acc_w[vn]['PiV'][k] += PiV[wsl[vn][k]].mean(axis=0) * dz
                    for d, s in enumerate(dsl[vn]):
                        acc_d[vn]['PiV'][d] += PiV[s].mean(axis=0) * dz
                    del PiV
                prev[vn] = (upr, vpr, ulr, vlr)
            del u, v_, w
            if ic == 0 and (kk % 5 == 0 or kk == nlev - 1):
                per = (_time.time() - tb) / (kk + 1)
                C.report(TAG, f'    block 1, level {kk + 1:2d}/{nlev}, {_time.time() - tl:5.1f} s, whole pass about '
                              f'{per * nlev * nchunk / 60:.0f} min, finish near '
                              f'{_time.strftime("%d %b %H:%M", _time.localtime(tb + per * nlev * nchunk))}')
        for vn in VARIANTS:
            for t in terms:
                for k in C.WKEYS:
                    W[vn][t][k][j0:j1] = acc_w[vn][t][k]
                D[vn][t][:, j0:j1] = acc_d[vn][t]
        del acc_w, acc_d, prev
        el = _time.time() - t0
        C.report(TAG, f'  block {ic + 1:2d}/{nchunk}, elapsed {el / 60:.0f} min, about '
                      f'{el / (ic + 1) * (nchunk - ic - 1) / 60:.0f} min left')

    m = sec_mask[:neta]
    ar = area[:neta]
    table = {}
    ser = {}
    for vn in VARIANTS:
        table[vn] = {}
        for k in C.WKEYS:
            ph = C.area_mean(W[vn]['PiH'][k], m, ar)
            pv = C.area_mean(W[vn]['PiV'][k], m, ar)
            table[vn][k] = dict(PiH=ph, PiV=pv, Pi=ph + pv)
        ser[vn] = {t: np.array([C.area_mean(D[vn][t][d], m, ar) for d in range(days.size)]) for t in terms}
        ser[vn]['Pi'] = ser[vn]['PiH'] + ser[vn]['PiV']

    full = None
    if os.path.exists(FIG03_JSON):
        with open(FIG03_JSON) as fh:
            full = json.load(fh)['table'][SECTOR]

    L = [f'FIGURE 3 FILTER CHECK, {SECTOR} sector exchange per variant',
         f'run {_time.strftime("%Y-%m-%d %H:%M")}, stride {st}, boxcar {smooth} cells, shelf and slope h <= '
         f'{C.STAT_MAX_DEPTH_M:.0f} m, area weighted, mW m-2, positive to the waves', '']
    for vn, (b, p) in VARIANTS.items():
        L.append(f'  variant {vn:8s} band [{b[0]}, {b[1]}] f, phase average {p:g} Tf')
    L.append('')
    L.append(f'  {"variant":8s} {"term":5s} ' + ' '.join(f'{k:>9s}' for k in C.WKEYS) + '    ratio to base per window')
    for vn in VARIANTS:
        for t in ('Pi', 'PiH', 'PiV'):
            vals = [table[vn][k][t] * 1e3 for k in C.WKEYS]
            base = [table['base'][k][t] * 1e3 for k in C.WKEYS]
            rat = ' '.join(f'{v / b:6.2f}' if b else '   nan' for v, b in zip(vals, base))
            L.append(f'  {vn:8s} {t:5s} ' + ' '.join(f'{v:+9.4f}' for v in vals) + '    ' + rat)
    if full:
        L.append('')
        L.append(f'  full resolution (fig03_stats.json) ' + ' '.join(f'{k} Pi {full[k]["Pi"] * 1e3:+.4f} PiV {full[k]["PiV"] * 1e3:+.4f}' for k in C.WKEYS))
        L.append('  base at this stride over full resolution, Pi: ' + ' '.join(
            f'{k} {table["base"][k]["Pi"] / full[k]["Pi"]:.2f}' if full[k]['Pi'] else f'{k} nan' for k in C.WKEYS))
    L.append('')
    L.append('  daily South Pi, minimum day and first positive day after it, per variant')
    landfall = C.LANDFALL_DEFAULT
    Tf = C.inertial_period_hours(float(np.nanmean(lat[:neta][m])))
    for vn in VARIANTS:
        s = ser[vn]['Pi']
        after = days + np.timedelta64(12, 'h') > np.datetime64(landfall)
        imin = int(np.nanargmin(np.where(after, s, np.nan)))
        pos = np.where(after & (np.arange(days.size) > imin) & (s > 0))[0]
        rev = str(days[pos[0]])[:10] if pos.size else 'none'
        lag = ((days[pos[0]] + np.timedelta64(12, 'h') - np.datetime64(landfall)).astype('timedelta64[s]').astype(float)
               / 3600.0 / Tf) if pos.size else np.nan
        L.append(f'  {vn:8s} minimum {s[imin] * 1e3:+.3f} on {str(days[imin])[:10]}, reversal {rev} ({lag:.1f} Tf after landfall)')
    L += ['', f'wall time {(_time.time() - t_start) / 60:.1f} min']
    C.write_report(OUT_TXT, L)
    print('\n'.join(L))

    data = {}
    for vn in VARIANTS:
        for t in ('Pi', 'PiH', 'PiV'):
            data[f'{t}_series_{vn}'] = (('day',), ser[vn][t])
    xr.Dataset(data, coords=dict(day=days.astype('datetime64[ns]')),
               attrs=dict(sector=SECTOR, stride=st, smooth_cells=smooth, variants=json.dumps(VARIANTS),
                          table=json.dumps(table, default=float),
                          note='area-weighted daily depth-integrated exchange of the sector, W m-2, per variant')
               ).to_netcdf(OUT_NC)
    print(f'wrote {OUT_NC} and {OUT_TXT}')
    arc.close()


if __name__ == '__main__':
    main()