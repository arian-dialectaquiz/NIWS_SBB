"""
fig04_compute.py

Where the exchange occurs in the subinertial flow (Figure 4, Question 3). No pass
over the archive: everything is formed from the window maps that Figures 2 and 3
wrote, the subinertial tensor of the upper 100 m (fig02/fig02_data.nc) and the
depth-integrated exchange (fig03/fig03_data.nc), on the same grid.

For each window, sector mask and route (Pi, Pi_H, Pi_V) the area-weighted mean
exchange is binned on two properties of the subinertial flow, the Okubo and Weiss
parameter scaled by its standard deviation over the mask and the effective
inertial frequency f_eff / f = 1 + zeta / 2f, with 95 % intervals from a
moving-block bootstrap over square blocks (BLOCK_KM), which keeps the spatial
correlation of the fields inside the resamples. The tails of each property are
folded into the outermost bins. The transfer per unit of near-inertial kinetic
energy, lambda = sum(Pi dA) / sum(K dA) per bin (Equation lambda of the text), is
bootstrapped the same way, and the pointwise median of Pi / K per bin is reported
beside it (comment 74). The class integrals (OW > 0 against OW < 0, anticyclonic
against cyclonic) and their area fractions give the proportionality sentence of
the Results. When fig03/fig03_anisotropy.nc exists its table is copied into the
report (M / K, the alignment) beside the strain-forced value Sigma / 2|f|.

Also written for panel (d): the T3 exchange, subinertial velocity and zeta / f
over the zoom box of the southern slope.

    python fig04_compute.py            # a few minutes
"""

import argparse
import json
import os
import time as _time

import numpy as np
import xarray as xr

import common_v3 as C

TAG = 'fig04'
FIG_DIR = C.fig_dir('fig04')
FIG02_NC = os.path.join(os.getcwd(), 'fig02', 'fig02_data.nc')
FIG03_NC = os.path.join(os.getcwd(), 'fig03', 'fig03_data.nc')
ANISO_NC = os.path.join(os.getcwd(), 'fig03', 'fig03_anisotropy.nc')
OUT_NC = os.path.join(FIG_DIR, 'fig04_data.nc')
OUT_JSON = os.path.join(FIG_DIR, 'fig04_stats.json')
OUT_REPORT = os.path.join(FIG_DIR, 'fig04_report.txt')

WKEYS = ('T1', 'T2', 'T3', 'T4')
ROUTES = ('Pi', 'PiH', 'PiV')
CMASKS = ('South', 'Domain')
OW_SIG_EDGES = np.linspace(-4.0, 4.0, 17)        # OW / sigma_OW, 0.5 wide, tails folded
FEFF_EDGES = np.linspace(0.86, 1.14, 15)         # f_eff / f, 0.02 wide, tails folded
BLOCK_KM = 30.0
NBOOT = 500
K_FLOOR_FRAC = 0.1                               # pointwise lambda only where K exceeds this fraction of the sector median
ZOOM = dict(lon=(-49.8, -46.0), lat=(-31.2, -28.0))
ZOOM_WINDOW = 'T3'
SEED = 0


def fold_digitize(x, edges):
    return np.clip(np.digitize(x, edges) - 1, 0, edges.size - 2)


def binned_bootstrap(x, y, area, m, edges, blockid, nblk, nboot, rng, denom=None):
    """Area-weighted mean of y (or the ratio of the sums of y and denom) in bins
    of x over the mask m, with a moving-block bootstrap over the spatial blocks.
    Returns mean, lo, hi, counts."""
    nb = edges.size - 1
    good = m & np.isfinite(x) & np.isfinite(y) & np.isfinite(area)
    if denom is not None:
        good &= np.isfinite(denom)
    z = np.full(nb, np.nan)
    if not good.any():
        return z, z, z, np.zeros(nb, int)
    bi = fold_digitize(x[good], edges)
    bl = blockid[good]
    w = area[good]
    Syw = np.zeros((nblk, nb))
    Sw = np.zeros((nblk, nb))
    np.add.at(Syw, (bl, bi), w * y[good])
    np.add.at(Sw, (bl, bi), w * (denom[good] if denom is not None else 1.0))
    counts = np.zeros(nb, int)
    np.add.at(counts, bi, 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = Syw.sum(0) / Sw.sum(0)
    present = np.where(Sw.sum(1) != 0)[0]
    boots = np.full((nboot, nb), np.nan)
    for b in range(nboot):
        pick = rng.choice(present, size=present.size, replace=True)
        c = np.bincount(pick, minlength=nblk).astype(float)
        with np.errstate(invalid='ignore', divide='ignore'):
            boots[b] = (c @ Syw) / (c @ Sw)
    lo = np.nanpercentile(boots, 2.5, axis=0)
    hi = np.nanpercentile(boots, 97.5, axis=0)
    return mean, lo, hi, counts


def pointwise_median(x, y, m, edges):
    nb = edges.size - 1
    good = m & np.isfinite(x) & np.isfinite(y)
    out = np.full(nb, np.nan)
    if good.any():
        bi = fold_digitize(x[good], edges)
        yy = y[good]
        for b in range(nb):
            if (bi == b).any():
                out[b] = np.median(yy[bi == b])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nboot', type=int, default=NBOOT)
    args = ap.parse_args()
    t_start = _time.time()
    rng = np.random.default_rng(SEED)

    d2 = xr.open_dataset(FIG02_NC)
    d3 = xr.open_dataset(FIG03_NC)
    if d2['h'].shape != d3['h'].shape or int(d2.attrs['stride']) != int(d3.attrs['stride']):
        raise SystemExit(f'fig02 grid {d2["h"].shape} stride {d2.attrs["stride"]} and fig03 grid '
                         f'{d3["h"].shape} stride {d3.attrs["stride"]} differ')
    st = int(d2.attrs['stride'])
    lon, lat = d2['lon'].values, d2['lat'].values
    h, mask, f2d, area = d2['h'].values, d2['mask'].values, d2['f'].values, d2['area'].values
    sector = d2['sector'].values
    land = mask < 0.5
    neta, nxi = h.shape
    shelf = (~land) & (h <= float(d2.attrs['stat_max_depth_m']))
    cmasks = {'South': shelf & (sector == 1), 'Domain': shelf}
    wkeys = [k for k in WKEYS if f'PiH_{k}' in d3 and f'OW_{k}' in d2]
    C.report(TAG, f'grid {neta} x {nxi} (stride {st}), windows {wkeys}')

    routes = {k: {'PiH': d3[f'PiH_{k}'].values, 'PiV': d3[f'PiV_{k}'].values} for k in wkeys}
    for k in wkeys:
        routes[k]['Pi'] = routes[k]['PiH'] + routes[k]['PiV']
    K = {k: d2[f'K_{k}'].values for k in wkeys}
    OW = {k: d2[f'OW_{k}'].values for k in wkeys}
    FEFF = {k: d2[f'feff_{k}'].values / f2d for k in wkeys}       # 1 + zeta / 2f (f < 0 in the south, feff / f > 0)
    ZOF = {k: d2[f'zeta_{k}'].values / f2d for k in wkeys}            # signed f, anticyclonic negative in both hemispheres
    SIG = {k: d2[f'Sigma_{k}'].values for k in wkeys}

    blk = max(int(round(BLOCK_KM / (1.0 * st))), 1)
    bj, bi = np.arange(neta) // blk, np.arange(nxi) // blk
    blockid = (bj[:, None] * (bi.max() + 1) + bi[None, :]).astype(int)
    nblk = int(blockid.max()) + 1
    C.report(TAG, f'bootstrap blocks of {blk} cells ({blk * st} km), {nblk} blocks, {args.nboot} resamples')

    nb_ow, nb_f = OW_SIG_EDGES.size - 1, FEFF_EDGES.size - 1
    cen_ow = 0.5 * (OW_SIG_EDGES[:-1] + OW_SIG_EDGES[1:])
    cen_f = 0.5 * (FEFF_EDGES[:-1] + FEFF_EDGES[1:])
    shape = (len(ROUTES), len(wkeys), len(CMASKS))
    C_ow, L_ow, H_ow = (np.full(shape + (nb_ow,), np.nan) for _ in range(3))
    C_f, L_f, H_f = (np.full(shape + (nb_f,), np.nan) for _ in range(3))
    N_ow = np.zeros((len(wkeys), len(CMASKS), nb_ow), int)
    N_f = np.zeros((len(wkeys), len(CMASKS), nb_f), int)
    lam, lam_lo, lam_hi, lam_med = (np.full((len(wkeys), len(CMASKS), nb_f), np.nan) for _ in range(4))
    sig_ow = np.full((len(wkeys), len(CMASKS)), np.nan)

    C.report(TAG, 'conditioning')
    for iw, k in enumerate(wkeys):
        for im, mn in enumerate(CMASKS):
            m = cmasks[mn]
            s = float(np.nanstd(OW[k][m]))
            sig_ow[iw, im] = s
            xow = OW[k] / s
            for ir, rn in enumerate(ROUTES):
                y = routes[k][rn]
                C_ow[ir, iw, im], L_ow[ir, iw, im], H_ow[ir, iw, im], N_ow[iw, im] = \
                    binned_bootstrap(xow, y, area, m, OW_SIG_EDGES, blockid, nblk, args.nboot, rng)
                C_f[ir, iw, im], L_f[ir, iw, im], H_f[ir, iw, im], N_f[iw, im] = \
                    binned_bootstrap(FEFF[k], y, area, m, FEFF_EDGES, blockid, nblk, args.nboot, rng)
            lam[iw, im], lam_lo[iw, im], lam_hi[iw, im], _ = \
                binned_bootstrap(FEFF[k], routes[k]['Pi'], area, m, FEFF_EDGES, blockid, nblk, args.nboot, rng,
                                 denom=K[k])
            kfloor = K_FLOOR_FRAC * float(np.nanmedian(K[k][m]))
            with np.errstate(invalid='ignore', divide='ignore'):
                lam_pt = np.where(K[k] > kfloor, routes[k]['Pi'] / K[k], np.nan)
            lam_med[iw, im] = pointwise_median(FEFF[k], lam_pt, m, FEFF_EDGES)
        C.report(TAG, f'  {k} done, {(_time.time() - t_start) / 60:.1f} min')

    # ---- class integrals and shares ------------------------------------------------
    stats = {'config': dict(ow_sig_edges=OW_SIG_EDGES.tolist(), feff_edges=FEFF_EDGES.tolist(),
                            block_km=BLOCK_KM, block_cells=blk, nblocks=nblk, nboot=args.nboot,
                            k_floor_frac=K_FLOOR_FRAC, zoom=ZOOM, zoom_window=ZOOM_WINDOW),
             'windows': {}}
    for iw, k in enumerate(wkeys):
        m = cmasks['South']
        good = m & np.isfinite(routes[k]['Pi']) & np.isfinite(OW[k]) & np.isfinite(area)
        tot = C.area_sum(routes[k]['Pi'], good, area)
        pos, neg = good & (OW[k] > 0), good & (OW[k] <= 0)
        anti, cyc = good & (FEFF[k] < 1.0), good & (FEFF[k] >= 1.0)
        atot = float(np.nansum(area[good]))
        d = dict(sigma_OW_south=sig_ow[iw, 0],
                 int_MW_total=tot * 1e-6, int_MW_OWpos=C.area_sum(routes[k]['Pi'], pos, area) * 1e-6,
                 int_MW_OWneg=C.area_sum(routes[k]['Pi'], neg, area) * 1e-6,
                 area_frac_OWpos=float(np.nansum(area[pos])) / atot,
                 int_MW_anti=C.area_sum(routes[k]['Pi'], anti, area) * 1e-6,
                 int_MW_cyc=C.area_sum(routes[k]['Pi'], cyc, area) * 1e-6,
                 area_frac_anti=float(np.nansum(area[anti])) / atot,
                 K_frac_anti=C.area_sum(K[k], anti, area) / C.area_sum(K[k], good, area),
                 PiV_frac_anti=(C.area_sum(routes[k]['PiV'], anti, area) / C.area_sum(routes[k]['PiV'], good, area)
                                if C.area_sum(routes[k]['PiV'], good, area) else np.nan),
                 Sigma_over_2f=C.area_mean(SIG[k] / (2.0 * np.abs(f2d)), m, area),
                 lambda_south_per_day=(C.area_sum(routes[k]['Pi'], good, area)
                                       / C.area_sum(K[k], good, area) * 86400.0),
                 most_anti_bin=dict(feff=float(cen_f[0]), Pi_mW=C_f[0, iw, 0, 0] * 1e3,
                                    lo=L_f[0, iw, 0, 0] * 1e3, hi=H_f[0, iw, 0, 0] * 1e3,
                                    lambda_per_day=lam[iw, 0, 0] * 86400.0),
                 most_cyc_bin=dict(feff=float(cen_f[-1]), Pi_mW=C_f[0, iw, 0, -1] * 1e3,
                                   lo=L_f[0, iw, 0, -1] * 1e3, hi=H_f[0, iw, 0, -1] * 1e3,
                                   lambda_per_day=lam[iw, 0, -1] * 86400.0))
        j = int(np.nanargmax(np.abs(C_f[0, iw, 0])))
        d['extreme_bin'] = dict(feff=float(cen_f[j]), Pi_mW=C_f[0, iw, 0, j] * 1e3, lo=L_f[0, iw, 0, j] * 1e3,
                                hi=H_f[0, iw, 0, j] * 1e3)
        d['ow_bins_interval_includes_zero'] = int(np.sum((L_ow[0, iw, 0] <= 0) & (H_ow[0, iw, 0] >= 0)))
        d['feff_bins_interval_includes_zero'] = int(np.sum((L_f[0, iw, 0] <= 0) & (H_f[0, iw, 0] >= 0)))
        stats['windows'][k] = d
        d['share_anti'] = d['int_MW_anti'] / d['int_MW_total'] if d['int_MW_total'] else np.nan
        d['share_OWpos'] = d['int_MW_OWpos'] / d['int_MW_total'] if d['int_MW_total'] else np.nan

    aniso = None
    if os.path.exists(ANISO_NC):
        with xr.open_dataset(ANISO_NC) as da:
            aniso = json.loads(da.attrs['table'])
        stats['anisotropy'] = aniso

    # ---- zoom fields for panel (d) ------------------------------------------------------
    zk = ZOOM_WINDOW if ZOOM_WINDOW in wkeys else wkeys[-1]
    inz = ((lon >= ZOOM['lon'][0]) & (lon <= ZOOM['lon'][1]) & (lat >= ZOOM['lat'][0]) & (lat <= ZOOM['lat'][1]))
    rows = np.where(inz.any(axis=1))[0]
    cols = np.where(inz.any(axis=0))[0]
    zs = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    C.report(TAG, f'zoom {zk}: rows {rows[0]} to {rows[-1]}, cols {cols[0]} to {cols[-1]}')

    # ---- report ------------------------------------------------------------------------------
    L = ['FIGURE 4 REPORT, numbers for the text', f'run {_time.strftime("%Y-%m-%d %H:%M")}',
         f'South shelf and slope h <= {d2.attrs["stat_max_depth_m"]:.0f} m, area weighted, blocks {blk * st} km, '
         f'{args.nboot} resamples, positive from the subinertial flow to the waves', '']
    L.append('class integrals of the depth-integrated exchange over the South sector, MW')
    L.append(f'  {"win":4s} {"total":>8s} {"OW>0":>8s} {"OW<0":>8s} {"area OW>0":>10s} {"share OW>0":>11s} '
             f'{"anticyc":>8s} {"cyc":>8s} {"area anti":>10s} {"share anti":>11s} {"K anti":>8s} {"PiV anti":>9s}')
    for k in wkeys:
        d = stats['windows'][k]
        L.append(f'  {k:4s} {d["int_MW_total"]:+8.2f} {d["int_MW_OWpos"]:+8.2f} {d["int_MW_OWneg"]:+8.2f} '
                 f'{d["area_frac_OWpos"]:10.2f} {d["share_OWpos"]:11.2f} {d["int_MW_anti"]:+8.2f} {d["int_MW_cyc"]:+8.2f} '
                 f'{d["area_frac_anti"]:10.2f} {d["share_anti"]:11.2f} {d["K_frac_anti"]:8.2f} {d["PiV_frac_anti"]:9.2f}')
    L.append('  (share = class integral over the total, K anti = share of the near-inertial kinetic energy in '
             'anticyclonic water, PiV anti = share of the vertical shear production there)')
    L.append('')
    L.append('binned on f_eff / f (South, Pi, mW m-2, with the 95 % interval), the extreme bin and the two end bins')
    for k in wkeys:
        d = stats['windows'][k]
        e, a, c = d['extreme_bin'], d['most_anti_bin'], d['most_cyc_bin']
        L.append(f'  {k}: extreme {e["Pi_mW"]:+.3f} [{e["lo"]:+.3f}, {e["hi"]:+.3f}] at f_eff/f {e["feff"]:.2f}, '
                 f'most anticyclonic bin {a["Pi_mW"]:+.3f} [{a["lo"]:+.3f}, {a["hi"]:+.3f}], '
                 f'most cyclonic bin {c["Pi_mW"]:+.3f} [{c["lo"]:+.3f}, {c["hi"]:+.3f}], '
                 f'bins whose interval includes zero {d["feff_bins_interval_includes_zero"]} of {nb_f} '
                 f'(OW {d["ow_bins_interval_includes_zero"]} of {nb_ow})')
    L.append('')
    L.append('lambda = sum(Pi) / sum(K) per bin of f_eff / f, South, d-1 (bin ratio of sums, then the pointwise median)')
    L.append('  ' + f'{"win":4s} ' + ' '.join(f'{c:6.2f}' for c in cen_f))
    for iw, k in enumerate(wkeys):
        L.append('  ' + f'{k:4s} ' + ' '.join(f'{v * 86400:6.2f}' for v in lam[iw, 0]))
        L.append('  ' + f'{"med":4s} ' + ' '.join(f'{v * 86400:6.2f}' for v in lam_med[iw, 0]))
    L.append('  sector-wide lambda per window, d-1: ' + ', '.join(
        f'{k} {stats["windows"][k]["lambda_south_per_day"]:+.3f}' for k in wkeys))
    L.append('')
    L.append('strain-forced anisotropy Sigma / 2|f| of the upper 100 m, South: ' + ', '.join(
        f'{k} {stats["windows"][k]["Sigma_over_2f"]:.3f}' for k in wkeys))
    if aniso:
        L.append('diagnosed anisotropy M / K and alignment cos 2(phi_w - phi_s), South (fig03_anisotropy): ' + ', '.join(
            f'{k} {aniso["South"][k]["M_over_K"]:.2f} ({aniso["South"][k]["cos2"]:+.2f})' for k in wkeys
            if k in aniso['South']))
    L += ['', f'wall time {(_time.time() - t_start) / 60:.1f} min']
    C.write_report(OUT_REPORT, L)
    print('\n'.join(L))

    # ---- archive ----------------------------------------------------------------------------
    out = xr.Dataset(
        data_vars={
            'C_ow': (('route', 'window', 'cmask', 'bin_ow'), C_ow), 'L_ow': (('route', 'window', 'cmask', 'bin_ow'), L_ow),
            'H_ow': (('route', 'window', 'cmask', 'bin_ow'), H_ow), 'N_ow': (('window', 'cmask', 'bin_ow'), N_ow),
            'C_f': (('route', 'window', 'cmask', 'bin_f'), C_f), 'L_f': (('route', 'window', 'cmask', 'bin_f'), L_f),
            'H_f': (('route', 'window', 'cmask', 'bin_f'), H_f), 'N_f': (('window', 'cmask', 'bin_f'), N_f),
            'lam': (('window', 'cmask', 'bin_f'), lam), 'lam_lo': (('window', 'cmask', 'bin_f'), lam_lo),
            'lam_hi': (('window', 'cmask', 'bin_f'), lam_hi), 'lam_med': (('window', 'cmask', 'bin_f'), lam_med),
            'sigma_OW': (('window', 'cmask'), sig_ow),
            'zoom_Pi': (('zeta', 'zxi'), routes[zk]['Pi'][zs]), 'zoom_ue': (('zeta', 'zxi'), d2[f'ue_{zk}'].values[zs]),
            'zoom_vn': (('zeta', 'zxi'), d2[f'vn_{zk}'].values[zs]), 'zoom_zof': (('zeta', 'zxi'), ZOF[zk][zs]),
            'zoom_h': (('zeta', 'zxi'), h[zs]), 'zoom_mask': (('zeta', 'zxi'), mask[zs]),
            'zoom_lon': (('zeta', 'zxi'), lon[zs]), 'zoom_lat': (('zeta', 'zxi'), lat[zs])},
        coords=dict(route=list(ROUTES), window=wkeys, cmask=list(CMASKS), bin_ow=cen_ow, bin_f=cen_f),
        attrs=dict(source=f'{FIG02_NC} + {FIG03_NC}', stride=st, block_km=BLOCK_KM, nboot=args.nboot,
                   ow_sig_edges=OW_SIG_EDGES.tolist(), feff_edges=FEFF_EDGES.tolist(),
                   zoom_window=zk, zoom_lon=list(ZOOM['lon']), zoom_lat=list(ZOOM['lat']),
                   zoom_rows=[int(rows[0]), int(rows[-1])], zoom_cols=[int(cols[0]), int(cols[-1])],
                   rho0=C.RHO0, stats=json.dumps(stats, default=float),
                   note=('Conditional means of the depth-integrated exchange (W m-2) on OW / sigma_OW and on '
                         'f_eff / f with moving-block bootstrap 95 % intervals, lambda in s-1, tails folded '
                         'into the end bins. Zoom fields of panel (d) in W m-2, m s-1 and zeta / |f|.'),
                   created=_time.strftime('%Y-%m-%d %H:%M')))
    out.to_netcdf(OUT_NC)
    with open(OUT_JSON, 'w') as fh:
        json.dump(stats, fh, indent=2, default=float)
    print(f'wrote {OUT_NC} and {OUT_JSON}')


if __name__ == '__main__':
    main()