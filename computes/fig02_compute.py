"""
fig02_compute.py

Every quantity of Figure 2 (the subinertial flow and the near-inertial kinetic
energy) from the two-part archive, in one row-block pass over the 3D velocity.
Nothing is plotted here. fig02_plot.py draws from fig02/fig02_data.nc alone.

Panels served (plan.md, Section 3, Figure 2)

    a  subinertial zeta / f over the upper 100 m, T1 mean, with the OW = 0
       contour and the subinertial velocity arrows
    b  rho0 int K dz averaged over T1
    c  the same over T3, with the track
    d  daily area-weighted rho0 int K dz over h <= 1000 m by sector, through T4
    e  the same normalised by each sector's T1 mean

Also written, for Figures 3 and 4: the upper-layer subinertial velocity and
its gradient tensor (delta, Sn, Ss, zeta, Sigma, OW, f_eff) for every window,
and the K maps of every window.

Definitions (common_v3.py): u' is the Lanczos band pass [0.8, 1.2] f, ubar the
Lanczos low pass at 1.75 Tf, both latitude dependent, K = <|u'_h|^2> / 2 per unit
mass with the phase average over one local inertial period, quoted multiplied
by rho0 and depth integrated with the layer thickness of the time-mean free
surface. The subinertial flow of the maps is the upper 100 m depth average, and
its gradient tensor is formed in flux form on the C grid after a 3-cell boxcar.

The pass reads u and v level by level for every row block. On the contiguous
archive this is a strided read per level, about as long as the v2 pass.

    nohup python -u fig02_compute.py > fig02/log_fig02.txt 2>&1 &
    python fig02_compute.py --stride 2 --eta-chunk 40     # a quick trial at half resolution
"""

import argparse
import json
import os
import time as _time

import numpy as np
import netCDF4 as nc
import xarray as xr

import common_v3 as C

TAG = 'fig02'
FIG_DIR = C.fig_dir('fig02')
OUT_NC = os.path.join(FIG_DIR, 'fig02_data.nc')
OUT_JSON = os.path.join(FIG_DIR, 'fig02_stats.json')
OUT_REPORT = os.path.join(FIG_DIR, 'fig02_report.txt')


def mean_zeta(arc, stride, sel_z, nt):
    """Time-mean free surface, from the surface file when it exists."""
    if os.path.exists(C.SURFACE_FILE):
        sf = nc.Dataset(C.SURFACE_FILE)
        sf.set_auto_mask(False)
        if sf.dimensions['time'].size == nt:
            z = np.zeros(sf.variables['zeta'].shape[1:], np.float64)
            for i0 in range(0, nt, 48):
                z += np.nansum(sf.variables['zeta'][i0:i0 + 48].astype(np.float64), axis=0)
            sf.close()
            return (z / nt)[::stride, ::stride]
        sf.close()
    z = None
    for i0 in range(0, nt, 24):
        blk = arc.read('zeta', sel_z, it=np.arange(i0, min(i0 + 24, nt)))
        z = blk.sum(axis=0) if z is None else z + blk.sum(axis=0)
    return (z / nt)[::stride, ::stride]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--eta-chunk', type=int, default=C.ETA_CHUNK)
    ap.add_argument('--workers', type=int, default=-1)
    args = ap.parse_args()
    st = args.stride
    t_start = _time.time()

    arc = C.Archive()
    arc.describe(TAG)
    arc.assert_even(TAG)
    g = C.load_grid(arc, stride=st)
    lon, lat, h, mask = g['lon'], g['lat'], g['h'], g['mask']
    pm, pn, angle, land, area = g['pm'], g['pn'], g['angle'], g['land'], g['area']
    neta, nxi = lon.shape
    nt, nlev = arc.nt, arc.nlev
    f2d = g['f']
    C.report(TAG, f'grid {neta} x {nxi} x {nlev} (stride {st}), {nt} records')

    # ---- filters and windows -----------------------------------------------
    fb = C.FilterBank(lat, land, arc.dt_h, nt, workers=args.workers)
    fb.attach_time(arc.time)
    fb.describe(TAG)
    valid = fb.valid
    idx = C.window_indices(arc.time, valid)
    for k in C.WKEYS:
        if idx[k].size == 0:
            raise SystemExit(f'window {k} has no valid samples')
    wsl = {k: C.as_slice(v) for k, v in idx.items()}
    days = np.unique(arc.time[valid].astype('datetime64[D]'))
    day_of = arc.time.astype('datetime64[D]')
    dsl = [C.as_slice(np.where((day_of == d) & valid)[0]) for d in days]

    # ---- vertical grid --------------------------------------------------------
    sel_z = dict(eta_rho=C.ETA_SLICE, xi_rho=C.XI_SLICE)
    zeta_m = mean_zeta(arc, st, sel_z, nt)
    Hz = np.nan_to_num(C.compute_Hz(arc.ds0, zeta_m, st, h=h), nan=0.0)      # (nlev, neta, nxi)
    depth_from_top = np.cumsum(Hz[::-1], axis=0)[::-1]
    inlayer = depth_from_top <= C.UPPER_LAYER_M
    inlayer[-1] = True
    Hz_lay = Hz * inlayer
    Hz_u = 0.5 * (Hz_lay[:, :, :-1] + Hz_lay[:, :, 1:])
    Hz_v = 0.5 * (Hz_lay[:, :-1, :] + Hz_lay[:, 1:, :])
    Hsum_u, Hsum_v = Hz_u.sum(axis=0), Hz_v.sum(axis=0)
    C.report(TAG, f'upper layer {C.UPPER_LAYER_M:.0f} m, {inlayer.sum(axis=0)[~land].mean():.1f} '
                  f'levels on average')

    # ---- accumulators -----------------------------------------------------------
    K_win = {k: np.zeros((neta, nxi)) for k in C.WKEYS}
    K_daily = np.zeros((days.size, neta, nxi))
    Ksub_win = {k: np.zeros((neta, nxi)) for k in C.WKEYS}          # subinertial KE, depth integrated
    ub_u = {k: np.zeros((neta, nxi - 1)) for k in C.WKEYS}
    vb_v = {k: np.zeros((neta - 1, nxi)) for k in C.WKEYS}
    ub_u_d = np.zeros((days.size, neta, nxi - 1), np.float32)     # daily upper-layer subinertial flow
    vb_v_d = np.zeros((days.size, neta - 1, nxi), np.float32)

    def read_rows(var, de, dx, r0, r1, lev):
        a = arc.read(var, {de: slice(C.ETA0 + r0 * st, C.ETA0 + r1 * st, st),
                           dx: slice(C.XI0, None, st), 's_rho': lev})
        # at a stride above 1 the strided faces can outnumber the strided cells by one
        return a[:, :, :nxi - 1] if dx == 'xi_u' else a[:, :, :nxi]

    # ---- the pass -----------------------------------------------------------------
    C.report(TAG, 'row-block pass over the 3D velocity')
    nchunk = int(np.ceil(neta / args.eta_chunk))
    t0 = _time.time()
    for ic, j0 in enumerate(range(0, neta, args.eta_chunk)):
        j1 = min(j0 + args.eta_chunk, neta)
        nrow = j1 - j0
        vr0, vr1 = max(j0 - 1, 0), min(j1, neta - 1)      # v rows with a halo below
        off = j0 - vr0
        w0, w1 = j0, min(j1, neta - 1)                      # v rows this block owns
        lv = w0 - vr0
        Kc = np.zeros((nt, nrow, nxi))
        Ksc = np.zeros((nt, nrow, nxi))
        tb = _time.time()
        for kk in range(nlev):
            tl = _time.time()
            u = read_rows('u', 'eta_u', 'xi_u', j0, j1, kk)
            v = read_rows('v', 'eta_v', 'xi_v', vr0, vr1, kk)
            t_read = _time.time() - tl
            fu = fb.apply_both(u, j0)
            fv = fb.apply_both(v, vr0)
            del u, v
            ur = C.u2rho(fu['bp'])
            vr = C.v2rho(fv['bp'])[:, off:off + nrow, :]
            Kc += 0.5 * C.RHO0 * (ur ** 2 + vr ** 2) * Hz[kk, j0:j1, :][None]
            ul, vl = fu['lp'], fv['lp']
            ulr = C.u2rho(ul)
            vlr = C.v2rho(vl)[:, off:off + nrow, :]
            Ksc += 0.5 * C.RHO0 * (ulr ** 2 + vlr ** 2) * Hz[kk, j0:j1, :][None]
            del ur, vr, ulr, vlr
            if inlayer[kk, j0:j1, :].any():
                for k in C.WKEYS:
                    ub_u[k][j0:j1] += ul[wsl[k]].mean(axis=0) * Hz_u[kk, j0:j1, :]
                    if w1 > w0:
                        vb_v[k][w0:w1] += (vl[wsl[k]].mean(axis=0)[lv:lv + (w1 - w0)]
                                           * Hz_v[kk, w0:w1, :])
                for d, s_ in enumerate(dsl):
                    ub_u_d[d, j0:j1] += ul[s_].mean(axis=0) * Hz_u[kk, j0:j1, :]
                    if w1 > w0:
                        vb_v_d[d, w0:w1] += (vl[s_].mean(axis=0)[lv:lv + (w1 - w0)]
                                             * Hz_v[kk, w0:w1, :])
            del fu, fv, ul, vl
            if ic == 0 and (kk % 5 == 0 or kk == nlev - 1):
                C.report(TAG, f'    block 1, level {kk + 1:2d}/{nlev}, read {t_read:5.1f} s, '
                              f'level total {_time.time() - tl:5.1f} s, block so far '
                              f'{(_time.time() - tb) / 60:.1f} min, whole pass about '
                              f'{(_time.time() - tb) / (kk + 1) * nlev * nchunk / 3600:.1f} h')
        Kc = fb.phase_average(Kc, j0)
        for k in C.WKEYS:
            K_win[k][j0:j1] = Kc[wsl[k]].mean(axis=0)
            Ksub_win[k][j0:j1] = Ksc[wsl[k]].mean(axis=0)
        for d, s_ in enumerate(dsl):
            K_daily[d, j0:j1] = Kc[s_].mean(axis=0)
        del Kc, Ksc
        el = _time.time() - t0
        C.report(TAG, f'  block {ic + 1:2d}/{nchunk}, rows {j0:5d}-{j1:5d}, '
                      f'{(_time.time() - tb) / 60:.1f} min, elapsed {el / 60:.0f} min, about '
                      f'{el / (ic + 1) * (nchunk - ic - 1) / 60:.0f} min left, '
                      f'finish near {_time.strftime("%H:%M", _time.localtime(_time.time() + el / (ic + 1) * (nchunk - ic - 1)))}')

    # ---- subinertial flow, upper layer average, and its tensor ------------------------
    mask_u = (mask[:, :-1] * mask[:, 1:]) > 0.5
    mask_v = (mask[:-1, :] * mask[1:, :]) > 0.5
    tensor = {}
    for k in C.WKEYS:
        uu = np.where((Hsum_u > 0) & mask_u, ub_u[k] / np.maximum(Hsum_u, 1e-9), np.nan)
        vv = np.where((Hsum_v > 0) & mask_v, vb_v[k] / np.maximum(Hsum_v, 1e-9), np.nan)
        us = np.nan_to_num(C.nan_boxcar(uu, C.SMOOTH_CELLS), nan=0.0)
        vs = np.nan_to_num(C.nan_boxcar(vv, C.SMOOTH_CELLS), nan=0.0)
        delta, Sn_g, zeta, Ss_g = C.cgrid_gradient_tensor(us, vs, pm, pn)
        Sn, Ss = C.strain_to_east_north(Sn_g, Ss_g, angle)
        Sigma = np.hypot(Sn, Ss)
        OW = Sigma ** 2 - zeta ** 2
        ue, vn = C.grid_to_east_north(C.u2rho(np.nan_to_num(uu, nan=0.0)),
                                      C.v2rho(np.nan_to_num(vv, nan=0.0)), angle)
        for a in (delta, Sn, Ss, zeta, Sigma, OW, ue, vn):
            a[land] = np.nan
        tensor[k] = dict(delta=delta, Sn=Sn, Ss=Ss, zeta=zeta, Sigma=Sigma, OW=OW,
                         ue=ue, vn=vn, feff=f2d + 0.5 * zeta)
    for k in C.WKEYS:
        K_win[k][land] = np.nan
        Ksub_win[k][land] = np.nan
    K_daily[:, land] = np.nan
    # daily upper-layer subinertial velocity on rho points, east and north, and its vorticity
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
    del ub_u_d, vb_v_d

    # ---- statistics --------------------------------------------------------------------
    masks = C.sector_masks(g, stride=st)
    shelf = masks['shelf']
    series = np.full((len(C.SECTOR_NAMES), days.size), np.nan)
    for si, sec in enumerate(C.SECTOR_NAMES):
        for di in range(days.size):
            series[si, di] = C.area_mean(K_daily[di], shelf[sec], area)
    stats = {'sectors': {}, 'series': {}}
    for si, sec in enumerate(C.SECTOR_NAMES):
        m = shelf[sec]
        d = {}
        for k in C.WKEYS:
            d[f'K_{k}_J_m2'] = C.area_mean(K_win[k], m, area)
            d[f'Ksub_{k}_J_m2'] = C.area_mean(Ksub_win[k], m, area)
            t = tensor[k]
            d[f'Sigma_{k}_s-1'] = C.area_mean(t['Sigma'], m, area)
            d[f'absZeta_over_f_{k}'] = C.area_mean(np.abs(t['zeta'] / f2d), m, area)
            d[f'delta_rms_{k}_s-1'] = float(np.sqrt(C.area_mean(t['delta'] ** 2, m, area)))
            d[f'Sigma_rms_{k}_s-1'] = float(np.sqrt(C.area_mean(t['Sigma'] ** 2, m, area)))
            ow = t['OW']
            good = m & np.isfinite(ow)
            d[f'frac_strain_dominated_{k}'] = (float(np.nansum(area[good & (ow > 0)])
                                                     / np.nansum(area[good])) if good.any() else np.nan)
            d[f'frac_anticyclonic_{k}'] = (float(np.nansum(area[good & (t['zeta'] * f2d < 0)])
                                                 / np.nansum(area[good])) if good.any() else np.nan)
        b1 = d['K_T1_J_m2']
        for k in C.WKEYS[1:]:
            d[f'K_{k}_over_T1'] = d[f'K_{k}_J_m2'] / b1 if b1 > 0 else np.nan
        s = series[si]
        kpk = int(np.nanargmax(s))
        d['K_peak_J_m2'] = float(s[kpk])
        d['K_peak_day'] = str(days[kpk])[:10]
        d['K_peak_over_T1'] = float(s[kpk] / np.nanmean(s[C.window_mask(days, C.WINDOWS['T1'])]))
        stats['sectors'][sec] = d
    stats['filters'] = fb.config()
    stats['filter_bands'] = fb.diag
    stats['config'] = dict(stride=st, eta_chunk=args.eta_chunk, upper_layer_m=C.UPPER_LAYER_M,
                           smooth_cells=C.SMOOTH_CELLS, stat_max_depth_m=C.STAT_MAX_DEPTH_M,
                           windows={k: list(v) for k, v in C.WINDOWS.items()},
                           n_valid={k: int(v.size) for k, v in idx.items()})

    # ---- report ----------------------------------------------------------------------------
    L = ['FIGURE 2 REPORT, numbers for the text', f'run {_time.strftime("%Y-%m-%d %H:%M")}',
         f'valid record {str(fb.valid_span()[0])[:16]} to {str(fb.valid_span()[1])[:16]}, '
         f'stride {st}, shelf and slope h <= {C.STAT_MAX_DEPTH_M:.0f} m, area weighted', '']
    L.append('near-inertial kinetic energy rho0 int K dz, window means, J m-2, and the ratio to T1')
    L.append(f'  {"sector":8s} ' + ' '.join(f'{k:>8s}' for k in C.WKEYS) + '   ' +
             ' '.join(f'{k + "/T1":>7s}' for k in C.WKEYS[1:]) + f' {"peak":>7s} {"day":>11s}')
    for sec in C.SECTOR_NAMES:
        d = stats['sectors'][sec]
        L.append(f'  {sec:8s} ' + ' '.join(f'{d[f"K_{k}_J_m2"]:8.0f}' for k in C.WKEYS) + '   ' +
                 ' '.join(f'{d[f"K_{k}_over_T1"]:7.1f}' for k in C.WKEYS[1:]) +
                 f' {d["K_peak_J_m2"]:7.0f} {d["K_peak_day"]:>11s}')
    L += ['', 'subinertial kinetic energy rho0 int Kbar dz, window means, J m-2']
    for sec in C.SECTOR_NAMES:
        d = stats['sectors'][sec]
        L.append(f'  {sec:8s} ' + ' '.join(f'{d[f"Ksub_{k}_J_m2"]:8.0f}' for k in C.WKEYS))
    L += ['', f'subinertial flow of the upper {C.UPPER_LAYER_M:.0f} m, per window']
    L.append(f'  {"sector":8s} {"win":4s} {"Sigma (s-1)":>12s} {"|zeta|/f":>9s} {"delta_rms/Sigma_rms":>20s} '
             f'{"OW>0 area":>10s} {"anticyc area":>13s}')
    for sec in C.SECTOR_NAMES:
        d = stats['sectors'][sec]
        for k in C.WKEYS:
            L.append(f'  {sec:8s} {k:4s} {d[f"Sigma_{k}_s-1"]:12.2e} {d[f"absZeta_over_f_{k}"]:9.3f} '
                     f'{d[f"delta_rms_{k}_s-1"] / d[f"Sigma_rms_{k}_s-1"]:20.2f} '
                     f'{100 * d[f"frac_strain_dominated_{k}"]:9.0f}% {100 * d[f"frac_anticyclonic_{k}"]:12.0f}%')
    L += ['', 'reading. The T1 row of the tensor is the pre-storm state of Figure 2a. The ratio '
              'delta_rms / Sigma_rms says how far the subinertial flow is from nondivergent in each '
              'window, which decides whether the divergence work of the polar form can matter (Figure 3).',
          f'filters: {json.dumps(fb.config())}', f'wall time {(_time.time() - t_start) / 3600:.2f} h']
    C.write_report(OUT_REPORT, L)
    print('\n'.join(L))

    # ---- archive -------------------------------------------------------------------------
    data = {'h': (('eta', 'xi'), h), 'mask': (('eta', 'xi'), mask), 'f': (('eta', 'xi'), f2d),
            'area': (('eta', 'xi'), area), 'sector': (('eta', 'xi'), masks['sector_id']),
            'K_daily': (('day', 'eta', 'xi'), K_daily.astype(np.float32)),
            'zeta_daily': (('day', 'eta', 'xi'), zeta_d),
            'ue_daily': (('day', 'eta', 'xi'), ue_d), 'vn_daily': (('day', 'eta', 'xi'), vn_d),
            'K_series': (('sector_name', 'day'), series)}
    for k in C.WKEYS:
        data[f'K_{k}'] = (('eta', 'xi'), K_win[k])
        data[f'Ksub_{k}'] = (('eta', 'xi'), Ksub_win[k])
        for name, arr in tensor[k].items():
            data[f'{name}_{k}'] = (('eta', 'xi'), arr)
    out = xr.Dataset(data_vars=data,
                     coords=dict(lon=(('eta', 'xi'), lon), lat=(('eta', 'xi'), lat),
                                 day=days.astype('datetime64[ns]'), sector_name=list(C.SECTOR_NAMES)),
                     attrs=dict(source=' + '.join(arc.paths), rho0=C.RHO0, stride=st,
                                eta0=C.ETA0, xi0=C.XI0, **{k: list(v) for k, v in C.WINDOWS.items()},
                                landfall=str(C.LANDFALL_DEFAULT)[:16],
                                upper_layer_m=C.UPPER_LAYER_M, smooth_cells=C.SMOOTH_CELLS,
                                stat_max_depth_m=C.STAT_MAX_DEPTH_M,
                                south_top=masks['south_top'], central_top=masks['central_top'],
                                sector_key='0 land, 1 South, 2 Central, 3 North',
                                filters=json.dumps(fb.config()),
                                note=('K is the phase-averaged near-inertial kinetic energy per unit '
                                      'mass times rho0, depth integrated, J m-2. Ksub the same for the '
                                      'subinertial flow. ue, vn are the upper-layer subinertial velocity '
                                      'in east and north components, the tensor scalars are rotation '
                                      'invariant except Sn and Ss which are east-north. feff = f + zeta/2. '
                                      'zeta_daily, ue_daily and vn_daily are the daily upper-layer subinertial '
                                      'vorticity and velocity (Figure 2e).'),
                                created=_time.strftime('%Y-%m-%d %H:%M')))
    # the landfall of fig01, when available, replaces the fallback
    f1 = os.path.join(os.getcwd(), 'fig01', 'fig01_data.nc')
    if os.path.exists(f1):
        with xr.open_dataset(f1) as d1:
            out.attrs['landfall'] = str(d1.attrs.get('landfall', out.attrs['landfall']))
    for v in out.data_vars:
        if out[v].dtype.kind == 'f':
            out[v].encoding.update(zlib=True, complevel=4)
    out.to_netcdf(OUT_NC)
    with open(OUT_JSON, 'w') as fh:
        json.dump(stats, fh, indent=2, default=float)
    print(f'wrote {OUT_NC} and {OUT_JSON}')
    arc.close()


if __name__ == '__main__':
    main()