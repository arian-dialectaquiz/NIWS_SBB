"""
decomp_check.py

Table 1 of the Methods. Quantifies the velocity decomposition u = ubar + u' + u''
before any budget is read (comment 44 of the review, decision 12 of plan.md).

For every sector and window, depth integrated over the shelf and slope
(h <= 1000 m) and area weighted, the script reports

    rho0 int 1/2 |u_h|^2 dz          the total horizontal kinetic energy
    fraction in ubar                 subinertial
    fraction in u'                   near-inertial
    fraction in u''                  residual band (what neither filter keeps)
    fraction in the cross terms      ubar.u' + ubar.u'' + u'.u''
    <|u''|^2> / <|u'|^2>             the residual against the wave kinetic energy

The archive is read record by record (one 3D chunk per record), averaged to rho
points and subsampled by --stride in both horizontal directions, and the full
time series of every level is held in memory, so the Lanczos filters run once.
At stride 4 the two components take about 15 GB. Use --stride 6 on a smaller
machine (7 GB), --levels 2 to keep every second level (half of that).

Writes decomp/decomp_check.nc, decomp/decomp_check_report.txt and decomp/table1_decomp.tex.

    nohup python -u decomp_check.py --stride 4 > log_decomp.txt 2>&1 &
"""

import argparse
import json
import os
import time as _time

import numpy as np
import xarray as xr

import common_v3 as C

TAG = 'decomp'
OUT_DIR = None   # set in main to <cwd>/decomp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--levels', type=int, default=1, help='keep every n-th level')
    ap.add_argument('--max-gb', type=float, default=40.0)
    ap.add_argument('--workers', type=int, default=-1)
    args = ap.parse_args()
    S, L = args.stride, args.levels
    t_start = _time.time()
    out_dir = C.fig_dir('decomp')

    arc = C.Archive()
    arc.describe(TAG)
    arc.assert_even(TAG)
    g = C.load_grid(arc)                      # full resolution subdomain
    gs = C.load_grid(arc, stride=S)           # strided grid for the statistics
    neta, nxi = g['lon'].shape
    nt, nlev = arc.nt, arc.nlev
    levels = np.arange(nlev)[::L]
    nl = levels.size
    lon_s, lat_s, land_s = gs['lon'], gs['lat'], gs['land']
    neta_s, nxi_s = lon_s.shape

    gb = 2 * nt * nl * neta_s * nxi_s * 4 / 1e9
    C.report(TAG, f'strided grid {neta_s} x {nxi_s}, {nl} levels, {nt} records, '
                  f'{gb:.1f} GB for u and v in float32')
    if gb > args.max_gb:
        raise SystemExit(f'{gb:.1f} GB exceeds --max-gb {args.max_gb}, raise --stride or --levels')

    U = np.empty((nt, nl, neta_s, nxi_s), np.float32)
    V = np.empty((nt, nl, neta_s, nxi_s), np.float32)
    zeta_sum = np.zeros((neta_s, nxi_s))

    sel_u = dict(eta_u=slice(C.ETA0, C.ETA_SLICE.stop), xi_u=slice(C.XI0, None))
    sel_v = dict(eta_v=slice(C.ETA0, C.ETA_SLICE.stop - 1), xi_v=slice(C.XI0, None))
    sel_z = dict(eta_rho=C.ETA_SLICE, xi_rho=C.XI_SLICE)

    C.report(TAG, 'reading the archive record by record')
    t0 = _time.time()
    for it in range(nt):
        u = arc.read_record('u', it, sel_u)          # (nlev, neta, nxi-1)
        v = arc.read_record('v', it, sel_v)          # (nlev, neta-1, nxi)
        ur = C.u2rho(u)[levels][:, ::S, ::S]
        vr = C.v2rho(v)[levels][:, ::S, ::S]
        U[it], V[it] = ur, vr
        zeta_sum += arc.read_record('zeta', it, sel_z)[::S, ::S]
        if it % 48 == 0 or it == nt - 1:
            el = _time.time() - t0
            C.report(TAG, f'{it + 1:4d}/{nt} {str(arc.time[it])[:16]}  {el / 60:.1f} min, '
                          f'about {el / (it + 1) * (nt - it - 1) / 60:.1f} min left')
    U[:, :, land_s] = 0.0
    V[:, :, land_s] = 0.0

    # ---- vertical grid (time-mean free surface) -----------------------------
    Hz_full = C.compute_Hz(arc.ds0, zeta_sum / nt, stride=S, h=gs['h'])   # (nlev, neta_s, nxi_s)
    Hz = Hz_full[levels] * L                      # every n-th level stands for n levels
    Hz = np.nan_to_num(Hz, nan=0.0)
    Hz[:, land_s] = 0.0

    # ---- filters -----------------------------------------------------------
    fb = C.FilterBank(lat_s, land_s, arc.dt_h, nt, need_pa=False, workers=args.workers)
    fb.attach_time(arc.time)
    fb.describe(TAG)
    valid = fb.valid
    idx = C.window_indices(arc.time, valid)
    days = np.unique(arc.time[valid].astype('datetime64[D]'))
    day_of = arc.time.astype('datetime64[D]')

    # depth-integrated kinetic energies per record, J m-2
    names = ('total', 'sub', 'ni', 'res', 'cross')
    KE = {n: np.zeros((nt, neta_s, nxi_s)) for n in names}
    C.report(TAG, 'filtering level by level')
    for k in range(nl):
        u = U[:, k].astype(np.float64)
        v = V[:, k].astype(np.float64)
        fu = fb.apply_both(u, 0)
        fv = fb.apply_both(v, 0)
        ub, un = fu['lp'], fu['bp']
        vb, vn = fv['lp'], fv['bp']
        ur, vr = u - ub - un, v - vb - vn
        hz = Hz[k][None]
        tot = 0.5 * (u ** 2 + v ** 2) * hz
        sub = 0.5 * (ub ** 2 + vb ** 2) * hz
        ni = 0.5 * (un ** 2 + vn ** 2) * hz
        res = 0.5 * (ur ** 2 + vr ** 2) * hz
        KE['total'] += tot
        KE['sub'] += sub
        KE['ni'] += ni
        KE['res'] += res
        KE['cross'] += tot - sub - ni - res
        del u, v, fu, fv, ub, un, vb, vn, ur, vr, tot, sub, ni, res
        if k % 5 == 0:
            C.report(TAG, f'  level {k + 1}/{nl}')
    del U, V
    for n in names:
        KE[n] *= C.RHO0

    # ---- statistics ----------------------------------------------------------
    masks = C.sector_masks(gs, stride=S)['shelf']
    area = gs['area']
    rows, stats = [], {}
    for sec in C.SECTOR_NAMES:
        m = masks[sec]
        stats[sec] = {}
        for w in C.WKEYS:
            ii = idx[w]
            if ii.size == 0:
                continue
            mean = {n: C.area_mean(KE[n][ii].mean(axis=0), m, area) for n in names}
            tot = mean['total']
            d = dict(KE_total_J_m2=tot,
                     frac_sub=mean['sub'] / tot, frac_ni=mean['ni'] / tot,
                     frac_res=mean['res'] / tot, frac_cross=mean['cross'] / tot,
                     res_over_ni=mean['res'] / mean['ni'] if mean['ni'] > 0 else np.nan,
                     KE_sub_J_m2=mean['sub'], KE_ni_J_m2=mean['ni'], KE_res_J_m2=mean['res'],
                     n_samples=int(ii.size))
            stats[sec][w] = d
            rows.append((sec, w, d))

    # daily series of the parts per sector
    series = np.full((len(C.SECTOR_NAMES), len(names), days.size), np.nan)
    for si, sec in enumerate(C.SECTOR_NAMES):
        for ni_, n in enumerate(names):
            for di, d in enumerate(days):
                ii = np.where((day_of == d) & valid)[0]
                if ii.size:
                    series[si, ni_, di] = C.area_mean(KE[n][ii].mean(axis=0), masks[sec], area)

    # ---- report ----------------------------------------------------------------
    lines = ['DECOMPOSITION CHECK, Table 1 of the Methods',
             f'stride {S}, levels every {L}, shelf and slope h <= {C.STAT_MAX_DEPTH_M:.0f} m, '
             f'depth integrated, area weighted, rho0 = {C.RHO0:g}',
             f'valid record {str(fb.valid_span()[0])[:16]} to {str(fb.valid_span()[1])[:16]}', '',
             f'{"sector":8s} {"win":4s} {"rho0 int K dz":>14s} {"ubar":>7s} {"u_ni":>7s} '
             f'{"u_res":>7s} {"cross":>7s} {"res/ni":>7s}',
             f'{"":8s} {"":4s} {"(J m-2)":>14s} {"(%)":>7s} {"(%)":>7s} {"(%)":>7s} {"(%)":>7s}']
    for sec, w, d in rows:
        lines.append(f'{sec:8s} {w:4s} {d["KE_total_J_m2"]:14.0f} {100 * d["frac_sub"]:7.1f} '
                     f'{100 * d["frac_ni"]:7.1f} {100 * d["frac_res"]:7.1f} '
                     f'{100 * d["frac_cross"]:7.1f} {d["res_over_ni"]:7.2f}')
    lines += ['', 'reading: the cross terms vanish for orthogonal parts over a long record, so',
              'their window mean measures the leakage between the filters over eight days.',
              'The residual band over the near-inertial band is the ratio asked for in',
              'comment 44, and the largest value in the table is the one quoted in the text.',
              f'wall time {(_time.time() - t_start) / 60:.1f} min']
    C.write_report(os.path.join(out_dir, 'decomp_check_report.txt'), lines)
    print('\n'.join(lines))

    # LaTeX table
    tex = [r'\begin{table}[H]', r'\centering',
           r'\caption{Partition of the depth-integrated horizontal kinetic energy among the '
           r'subinertial, near-inertial and residual parts of the velocity decomposition '
           r'(Equation~\ref{eq:decomp}), area weighted over the shelf and slope shallower than '
           r'1000~m of each sector and averaged over each window. The cross terms are the '
           r'remainder of the total, and the last column is the residual-band kinetic energy '
           r'over the near-inertial kinetic energy.}',
           r'\label{tab:decomp}', r'\small',
           r'\begin{tabular}{llrrrrrr}', r'\toprule',
           r'Sector & Window & $\rho_0\int\frac{1}{2}|\mathbf{u}_h|^2\,\mathrm{d}z$ & '
           r'$\bar{\mathbf{u}}$ & $\mathbf{u}^{\prime}$ & $\mathbf{u}^{\prime\prime}$ & cross & '
           r'$\langle|\mathbf{u}^{\prime\prime}|^2\rangle/\langle|\mathbf{u}^{\prime}|^2\rangle$ \\',
           r' & & (kJ~m$^{-2}$) & (\%) & (\%) & (\%) & (\%) & \\', r'\midrule']
    for sec, w, d in rows:
        if sec == 'Domain':
            continue
        tex.append(f'{sec} & ${w}$ & {d["KE_total_J_m2"] / 1e3:.1f} & {100 * d["frac_sub"]:.1f} & '
                   f'{100 * d["frac_ni"]:.1f} & {100 * d["frac_res"]:.1f} & '
                   f'{100 * d["frac_cross"]:.1f} & {d["res_over_ni"]:.2f} \\\\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    with open(os.path.join(out_dir, 'table1_decomp.tex'), 'w') as fh:
        fh.write('\n'.join(tex) + '\n')
    print(f'wrote {out_dir}/table1_decomp.tex')

    # ---- archive ---------------------------------------------------------------
    out = xr.Dataset(
        data_vars={**{f'KE_{n}_{w}': (('eta', 'xi'), np.where(land_s, np.nan,
                                                               KE[n][idx[w]].mean(axis=0)))
                      for n in names for w in C.WKEYS if idx[w].size},
                   'series': (('sector', 'part', 'day'), series),
                   'h': (('eta', 'xi'), gs['h']), 'mask': (('eta', 'xi'), gs['mask']),
                   'area': (('eta', 'xi'), area)},
        coords=dict(lon=(('eta', 'xi'), lon_s), lat=(('eta', 'xi'), lat_s), day=days,
                    sector=list(C.SECTOR_NAMES), part=list(names)),
        attrs=dict(source=' + '.join(arc.paths), stride=S, levels_every=L, rho0=C.RHO0,
                   stat_max_depth_m=C.STAT_MAX_DEPTH_M, **{k: list(v) for k, v in C.WINDOWS.items()},
                   filters=json.dumps(fb.config()), created=_time.strftime('%Y-%m-%d %H:%M')))
    for v in out.data_vars:
        out[v].encoding.update(zlib=True, complevel=4)
    out.to_netcdf(os.path.join(out_dir, 'decomp_check.nc'))
    with open(os.path.join(out_dir, 'decomp_check_stats.json'), 'w') as fh:
        json.dump(dict(stats=stats, filters=fb.config(), filter_bands=fb.diag), fh, indent=2)
    print(f'wrote {out_dir}/decomp_check.nc and decomp_check_stats.json')
    arc.close()


if __name__ == '__main__':
    main()