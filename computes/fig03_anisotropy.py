"""
fig03_anisotropy.py

The terms of the polar form of the horizontal shear production, per sector and
window, at a reduced resolution (stride 3, about a thirtieth of the Figure 3 pass).

    Pi_H = -M Sigma cos[2(phi_w - phi_s)] - K delta,

with M = 0.5 sqrt((<u'u'> - <v'v'>)^2 + 4 <u'v'>^2) the anisotropy of the wave
stress (M = K for rectilinear motion, 0 for circular motion), K the near-inertial
kinetic energy, Sigma the strain and delta the divergence of the subinertial
flow. The question is whether the deformation work is small because the strain
is small or because the near-inertial motion is close to circular (M << K). The
script writes, per sector and window, the depth-integrated and area-averaged M
and K and their ratio (the effective anisotropy), the bound rho0 M Sigma on the
deformation work, the deformation and divergence works themselves at this
resolution and their ratio to the full-resolution values of fig03_stats.json,
and the effective alignment cos[2(phi_w - phi_s)] = -PiHdef / (rho0 M Sigma).

    nohup python -u fig03_anisotropy.py --stride 3 > fig03/log_anisotropy.txt 2>&1 &
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

TAG = 'aniso'
FIG_DIR = C.fig_dir('fig03')
OUT_NC = os.path.join(FIG_DIR, 'fig03_anisotropy.nc')
OUT_TXT = os.path.join(FIG_DIR, 'fig03_anisotropy.txt')
FIG03_JSON = os.path.join(FIG_DIR, 'fig03_stats.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--eta-chunk', type=int, default=40)
    ap.add_argument('--workers', type=int, default=-1)
    args = ap.parse_args()
    st = args.stride
    t_start = _time.time()
    smooth = max(1, int(round(C.SMOOTH_CELLS / st)))      # the same 3 km boxcar at any stride

    arc = C.Archive()
    arc.describe(TAG)
    arc.assert_even(TAG)
    g = C.load_grid(arc, stride=st)
    lon, lat, h, mask = g['lon'], g['lat'], g['h'], g['mask']
    pm, pn, land, area = g['pm'], g['pn'], g['land'], g['area']
    neta, nxi = lon.shape
    nt, nlev = arc.nt, arc.nlev
    C.report(TAG, f'grid {neta} x {nxi} x {nlev} (stride {st}, boxcar {smooth} cells), {nt} records')

    fb = C.FilterBank(lat, land, arc.dt_h, nt, workers=args.workers)
    fb.attach_time(arc.time)
    fb.describe(TAG)
    idx = C.window_indices(arc.time, fb.valid)
    wsl = {k: C.as_slice(v) for k, v in idx.items()}

    sel_z = dict(eta_rho=C.ETA_SLICE, xi_rho=C.XI_SLICE)
    Hz = np.nan_to_num(C.compute_Hz(arc.ds0, mean_zeta(arc, st, sel_z, nt), st, h=h), nan=0.0)

    names = ('M', 'K', 'MSigma', 'Kdelta', 'PiHdef', 'PiHdiv')
    W = {n: {k: np.zeros((neta, nxi)) for k in C.WKEYS} for n in names}

    def read_block(var, de, dx, r0, r1, lev):
        # rows are strided in the read (each row is one contiguous run in the file) and the
        # columns are subsampled in memory, which a strided column read would make slow
        sel = {de: slice(C.ETA0 + r0 * st, C.ETA0 + r1 * st, st), dx: slice(C.XI0, None), 's_rho': lev}
        a = arc.read(var, sel)[..., ::st]
        return a[..., :nxi - 1] if dx == 'xi_u' else a[..., :nxi]

    nchunk = int(np.ceil(neta / args.eta_chunk))
    t0 = _time.time()
    for ic, j0 in enumerate(range(0, neta, args.eta_chunk)):
        tb = _time.time()
        j1 = min(j0 + args.eta_chunk, neta)
        nrow = j1 - j0
        a0, a1 = max(j0 - HALO, 0), min(j1 + HALO, neta)
        b0, b1 = a0, min(j1 + 1, neta - 1)
        s0 = j0 - a0
        pmL, pnL = pm[a0:a1, :], pn[a0:a1, :]
        acc = {n: {k: np.zeros((nrow, nxi)) for k in C.WKEYS} for n in names}
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
            del fu, fv
            pa = lambda a: fb.phase_average(a, j0)
            dev = pa(upr * upr - vpr * vpr)
            tra = pa(upr * upr + vpr * vpr)
            uv = pa(upr * vpr)
            del upr, vpr
            M = 0.5 * np.sqrt(dev * dev + 4.0 * uv * uv)
            K = 0.5 * tra
            Sigma = np.sqrt(Sng * Sng + Ssg * Ssg)
            fields = {'M': M, 'K': K, 'MSigma': M * Sigma, 'Kdelta': K * dlt,
                      'PiHdef': -(0.5 * dev * Sng + uv * Ssg), 'PiHdiv': -(0.5 * tra * dlt)}
            hz = Hz[kk, j0:j1, :]
            for n, f in fields.items():
                for k in C.WKEYS:
                    acc[n][k] += f[wsl[k]].mean(axis=0) * hz
            del fields, M, K, Sigma, dev, tra, uv, dlt, Sng, Ssg
            if ic == 0 and (kk % 5 == 0 or kk == nlev - 1):
                per = (_time.time() - tb) / (kk + 1)
                C.report(TAG, f'    block 1, level {kk + 1:2d}/{nlev}, {_time.time() - tl:5.1f} s, whole pass about '
                              f'{per * nlev * nchunk / 60:.0f} min, finish near '
                              f'{_time.strftime("%d %b %H:%M", _time.localtime(tb + per * nlev * nchunk))}')
        for n in names:
            for k in C.WKEYS:
                W[n][k][j0:j1] = acc[n][k]
        el = _time.time() - t0
        C.report(TAG, f'  block {ic + 1:2d}/{nchunk}, elapsed {el / 60:.0f} min, about '
                      f'{el / (ic + 1) * (nchunk - ic - 1) / 60:.0f} min left')

    for n in names:
        for k in C.WKEYS:
            W[n][k][land] = np.nan

    masks = C.sector_masks(g, stride=st)
    shelf = masks['shelf']
    full = {}
    if os.path.exists(FIG03_JSON):
        with open(FIG03_JSON) as fh:
            full = json.load(fh)['table']
    res = {}
    L = ['FIGURE 3 ANISOTROPY CHECK, the polar form per sector and window',
         f'run {_time.strftime("%Y-%m-%d %H:%M")}, stride {st}, boxcar {smooth} cells, shelf and slope h <= '
         f'{C.STAT_MAX_DEPTH_M:.0f} m, area weighted, depth integrated', '',
         'M / K is the effective anisotropy of the near-inertial stress (1 rectilinear, 0 circular).',
         'rho0 M Sigma bounds the deformation work, cos2 = -PiHdef / (rho0 M Sigma) is the effective alignment.',
         'Kdelta check = PiHdiv / (-rho0 K delta), which is 1 by construction.',
         'ratio = the works at this stride over the full-resolution values of fig03_stats.json.', '']
    hdr = f'  {"sector":8s} {"win":4s} {"M (J m-2)":>10s} {"K (J m-2)":>10s} {"M/K":>6s} {"rho0 M Sig":>11s} {"PiHdef":>8s} {"cos2":>6s} {"PiHdiv":>8s} {"Kd chk":>7s} {"def rat":>8s} {"div rat":>8s}'
    L.append(hdr + '   (mW m-2 for the works and the bound)')
    for sec in C.SECTOR_NAMES:
        res[sec] = {}
        m = shelf[sec]
        for k in C.WKEYS:
            am = {n: C.area_mean(W[n][k], m, area) for n in names}
            r = dict(M=C.RHO0 * am['M'], K=C.RHO0 * am['K'], M_over_K=am['M'] / am['K'] if am['K'] else np.nan,
                     MSigma_mW=C.RHO0 * am['MSigma'] * 1e3, PiHdef_mW=C.RHO0 * am['PiHdef'] * 1e3,
                     cos2=-am['PiHdef'] / am['MSigma'] if am['MSigma'] else np.nan,
                     PiHdiv_mW=C.RHO0 * am['PiHdiv'] * 1e3,
                     Kdelta_check=am['PiHdiv'] / (-am['Kdelta']) if am['Kdelta'] else np.nan)
            if full:
                r['def_ratio'] = r['PiHdef_mW'] / (full[sec][k]['PiHdef'] * 1e3) if full[sec][k]['PiHdef'] else np.nan
                r['div_ratio'] = r['PiHdiv_mW'] / (full[sec][k]['PiHdiv'] * 1e3) if full[sec][k]['PiHdiv'] else np.nan
            res[sec][k] = r
            L.append(f'  {sec:8s} {k:4s} {r["M"]:10.1f} {r["K"]:10.1f} {r["M_over_K"]:6.2f} {r["MSigma_mW"]:11.4f} '
                     f'{r["PiHdef_mW"]:+8.4f} {r["cos2"]:+6.2f} {r["PiHdiv_mW"]:+8.4f} {r["Kdelta_check"]:7.3f} '
                     + (f'{r["def_ratio"]:8.2f} {r["div_ratio"]:8.2f}' if full else ''))
    L += ['', f'wall time {(_time.time() - t_start) / 60:.1f} min']
    C.write_report(OUT_TXT, L)
    print('\n'.join(L))

    data = {f'{n}_{k}': (('eta', 'xi'), W[n][k]) for n in names for k in C.WKEYS}
    data.update(h=(('eta', 'xi'), h), mask=(('eta', 'xi'), mask), area=(('eta', 'xi'), area))
    xr.Dataset(data, coords=dict(lon=(('eta', 'xi'), lon), lat=(('eta', 'xi'), lat)),
               attrs=dict(stride=st, smooth_cells=smooth, rho0=C.RHO0, filters=json.dumps(fb.config()),
                          table=json.dumps(res, default=float),
                          note='depth-integrated window means per unit mass (multiply by rho0), '
                               'M anisotropy, K near-inertial kinetic energy, MSigma and Kdelta the '
                               'polar-form bounds, PiHdef and PiHdiv the works')).to_netcdf(OUT_NC)
    print(f'wrote {OUT_NC} and {OUT_TXT}')
    arc.close()


if __name__ == '__main__':
    main()