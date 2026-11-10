"""
fig02_daily_sub.py

The daily upper-100 m subinertial velocity and vorticity for Figure 2e, without
repeating the full Figure 2 pass. Only the low pass is needed, so the archive
is read every --tstep hours (3 h resolves the 40 to 52 h cutoff with margin) and
at --stride cells (2 km is enough to weight K by the vorticity), which cuts the
pass to about a tenth of fig02_compute.py. The product is identical in meaning
to the zeta_daily that fig02_compute.py writes on a full pass, and fig02_plot.py
reads either.

Writes fig02/fig02_daily_sub.nc.

    nohup python -u fig02_daily_sub.py --stride 2 --tstep 3 > fig02/log_daily_sub.txt 2>&1 &
"""

import argparse
import json
import os
import time as _time

import numpy as np
import xarray as xr

import common_v3 as C
from fig02_compute import mean_zeta

TAG = 'dailysub'
FIG_DIR = C.fig_dir('fig02')
OUT_NC = os.path.join(FIG_DIR, 'fig02_daily_sub.nc')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--tstep', type=int, default=3, help='hours between the records read')
    ap.add_argument('--eta-chunk', type=int, default=40)
    ap.add_argument('--workers', type=int, default=-1)
    ap.add_argument('--contiguous-read', action='store_true',
                    help='read every record and subsample in memory (try it if the strided '
                         'read is slow on the archive)')
    args = ap.parse_args()
    st, ts = args.stride, args.tstep
    t_start = _time.time()

    arc = C.Archive()
    arc.describe(TAG)
    arc.assert_even(TAG)
    g = C.load_grid(arc, stride=st)
    lon, lat, h, mask = g['lon'], g['lat'], g['h'], g['mask']
    pm, pn, angle, land = g['pm'], g['pn'], g['angle'], g['land']
    neta, nxi = lon.shape
    nlev = arc.nlev
    it_all = np.arange(0, arc.nt, ts)
    time = arc.time[it_all]
    nt = time.size
    dt_h = arc.dt_h * ts
    C.report(TAG, f'grid {neta} x {nxi} (stride {st}), {nt} records every {ts} h')

    fb = C.FilterBank(lat, land, dt_h, nt, need_bp=False, need_pa=False, workers=args.workers)
    fb.attach_time(time)
    fb.describe(TAG)
    valid = fb.valid
    days = np.unique(time[valid].astype('datetime64[D]'))
    day_of = time.astype('datetime64[D]')
    dsl = [C.as_slice(np.where((day_of == d) & valid)[0]) for d in days]

    sel_z = dict(eta_rho=C.ETA_SLICE, xi_rho=C.XI_SLICE)
    zeta_m = mean_zeta(arc, st, sel_z, arc.nt)
    Hz = np.nan_to_num(C.compute_Hz(arc.ds0, zeta_m, st, h=h), nan=0.0)
    depth_from_top = np.cumsum(Hz[::-1], axis=0)[::-1]
    inlayer = depth_from_top <= C.UPPER_LAYER_M
    inlayer[-1] = True
    Hz_lay = Hz * inlayer
    Hz_u = 0.5 * (Hz_lay[:, :, :-1] + Hz_lay[:, :, 1:])
    Hz_v = 0.5 * (Hz_lay[:, :-1, :] + Hz_lay[:, 1:, :])
    Hsum_u, Hsum_v = Hz_u.sum(axis=0), Hz_v.sum(axis=0)

    ub_u_d = np.zeros((days.size, neta, nxi - 1), np.float32)
    vb_v_d = np.zeros((days.size, neta - 1, nxi), np.float32)

    def read_rows(var, de, dx, r0, r1, lev):
        sel = {de: slice(C.ETA0 + r0 * st, C.ETA0 + r1 * st, st), dx: slice(C.XI0, None, st), 's_rho': lev}
        a = arc.read(var, sel)[::ts] if args.contiguous_read else arc.read(var, sel, it=it_all)
        # at a stride above 1 the strided faces can outnumber the strided cells by one
        return a[:, :, :nxi - 1] if dx == 'xi_u' else a[:, :, :nxi]

    nchunk = int(np.ceil(neta / args.eta_chunk))
    t0 = _time.time()
    for ic, j0 in enumerate(range(0, neta, args.eta_chunk)):
        j1 = min(j0 + args.eta_chunk, neta)
        vr0, vr1 = max(j0 - 1, 0), min(j1, neta - 1)
        w0, w1 = j0, min(j1, neta - 1)
        lv = w0 - vr0
        tb = _time.time()
        for kk in range(nlev):
            if not inlayer[kk, j0:j1, :].any():
                continue
            tl = _time.time()
            ul = fb.apply(read_rows('u', 'eta_u', 'xi_u', j0, j1, kk), j0, 'lp')
            vl = fb.apply(read_rows('v', 'eta_v', 'xi_v', vr0, vr1, kk), vr0, 'lp')
            if ic == 0 and (kk % 5 == 0 or kk == nlev - 1):
                C.report(TAG, f'    block 1, level {kk + 1:2d}/{nlev}, {_time.time() - tl:5.1f} s, '
                              f'whole pass about {(_time.time() - tb) / (kk + 1) * nlev * nchunk / 60:.0f} min')
            for d, s_ in enumerate(dsl):
                ub_u_d[d, j0:j1] += ul[s_].mean(axis=0) * Hz_u[kk, j0:j1, :]
                if w1 > w0:
                    vb_v_d[d, w0:w1] += vl[s_].mean(axis=0)[lv:lv + (w1 - w0)] * Hz_v[kk, w0:w1, :]
            del ul, vl
        el = _time.time() - t0
        C.report(TAG, f'  block {ic + 1:2d}/{nchunk}, elapsed {el / 60:.1f} min, about '
                      f'{el / (ic + 1) * (nchunk - ic - 1) / 60:.1f} min left')

    mask_u = (mask[:, :-1] * mask[:, 1:]) > 0.5
    mask_v = (mask[:-1, :] * mask[1:, :]) > 0.5
    ue_d = np.full((days.size, neta, nxi), np.nan, np.float32)
    vn_d = np.full((days.size, neta, nxi), np.nan, np.float32)
    zeta_d = np.full((days.size, neta, nxi), np.nan, np.float32)
    for d in range(days.size):
        uu = np.where((Hsum_u > 0) & mask_u, ub_u_d[d] / np.maximum(Hsum_u, 1e-9), np.nan)
        vv = np.where((Hsum_v > 0) & mask_v, vb_v_d[d] / np.maximum(Hsum_v, 1e-9), np.nan)
        us = np.nan_to_num(C.nan_boxcar(uu, C.SMOOTH_CELLS), nan=0.0)
        vs = np.nan_to_num(C.nan_boxcar(vv, C.SMOOTH_CELLS), nan=0.0)
        _, _, zd, _ = C.cgrid_gradient_tensor(us, vs, pm, pn)
        e_, n_ = C.grid_to_east_north(C.u2rho(np.nan_to_num(uu, nan=0.0)),
                                      C.v2rho(np.nan_to_num(vv, nan=0.0)), angle)
        zd[land], e_[land], n_[land] = np.nan, np.nan, np.nan
        zeta_d[d], ue_d[d], vn_d[d] = zd, e_, n_

    out = xr.Dataset(
        data_vars={'zeta_daily': (('day', 'eta', 'xi'), zeta_d),
                   'ue_daily': (('day', 'eta', 'xi'), ue_d), 'vn_daily': (('day', 'eta', 'xi'), vn_d),
                   'h': (('eta', 'xi'), h), 'mask': (('eta', 'xi'), mask), 'f': (('eta', 'xi'), g['f']),
                   'area': (('eta', 'xi'), g['area'])},
        coords=dict(lon=(('eta', 'xi'), lon), lat=(('eta', 'xi'), lat), day=days.astype('datetime64[ns]')),
        attrs=dict(source=' + '.join(arc.paths), stride=st, tstep_h=ts, upper_layer_m=C.UPPER_LAYER_M,
                   smooth_cells=C.SMOOTH_CELLS, filters=json.dumps(fb.config()),
                   note=('Daily upper-layer subinertial velocity (east, north) and vorticity, from '
                         'the low-passed velocity read every tstep hours at the given stride.'),
                   created=_time.strftime('%Y-%m-%d %H:%M')))
    for v in out.data_vars:
        out[v].encoding.update(zlib=True, complevel=4)
    out.to_netcdf(OUT_NC)
    C.report(TAG, f'wrote {OUT_NC}, wall time {(_time.time() - t_start) / 60:.1f} min')
    arc.close()


if __name__ == '__main__':
    main()