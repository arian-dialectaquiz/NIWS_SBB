"""
fig03_compute.py

Every quantity of Figure 3 (the exchange through the storm and the wake) and of
Table 2, from the two-part archive, in one row-block pass over the 3D velocity.
Nothing is plotted here. fig03_plot.py draws from fig03/fig03_data.nc alone.

The exchange is the shear production of the near-inertial kinetic energy by the
subinertial flow, positive from the subinertial flow to the waves,

    Pi_H = -<u'u'> d_x ubar - <u'v'> (d_y ubar + d_x vbar) - <v'v'> d_y vbar
         = -M Sigma cos[2(phi_w - phi_s)]  -  K delta        (deformation work + divergence work)
    Pi_V = -<u'w'> d_z ubar - <v'w'> d_z vbar

per unit mass, quoted multiplied by rho0 and depth integrated (W m-2). The
discretisation is that of v2: the subinertial gradient tensor in flux form on the
C grid after a 3-cell boxcar of the subinertial velocity, the wave stresses on rho
points, the vertical stress and shear on the w levels, the phase average over one
local inertial period, and the window and daily means over the valid record.
The deformation and divergence works are the two terms of the polar form, which
in the component form are the deviatoric part and the trace part of the stress
contracted with the strain and the divergence.

Written: window maps (T1 to T4) and daily maps of Pi_H, Pi_V, the deformation
work and the divergence work, the sector series, the window means per sector
(Table 2), the reversal day of the South sector with its lag after landfall in
local inertial periods, and the e-folding of the T2 drain.

    nohup python -u fig03_compute.py > fig03/log_fig03.txt 2>&1 &
    python fig03_compute.py --stride 2 --eta-chunk 32      # half-resolution trial
"""

import argparse
import json
import os
import time as _time

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter

import common_v3 as C
from fig02_compute import mean_zeta

TAG = 'fig03'
FIG_DIR = C.fig_dir('fig03')
OUT_NC = os.path.join(FIG_DIR, 'fig03_data.nc')
OUT_JSON = os.path.join(FIG_DIR, 'fig03_stats.json')
OUT_REPORT = os.path.join(FIG_DIR, 'fig03_report.txt')
OUT_TABLE = os.path.join(FIG_DIR, 'table2_exchange.tex')
FIG02_JSON = os.path.join(os.getcwd(), 'fig02', 'fig02_stats.json')
FIG01_NC = os.path.join(os.getcwd(), 'fig01', 'fig01_data.nc')
HALO = 2


def psi2rho_t(a):
    nt = a.shape[0]
    neta, nxi = a.shape[1] + 1, a.shape[2] + 1
    out = np.full((nt, neta, nxi), np.nan)
    out[:, 1:-1, 1:-1] = 0.25 * (a[:, :-1, :-1] + a[:, :-1, 1:] + a[:, 1:, :-1] + a[:, 1:, 1:])
    return out


def smooth_h(arr, size):
    """NaN-tolerant boxcar over the two trailing axes of a (time, eta, xi) array."""
    if size <= 1:
        return arr
    valid = np.isfinite(arr).astype(np.float64)
    num = uniform_filter(np.where(np.isfinite(arr), arr, 0.0), size=(1, size, size), mode='nearest')
    den = uniform_filter(valid, size=(1, size, size), mode='nearest')
    return np.where(den > 0.5, num / np.maximum(den, 1e-12), np.nan)


def cgrid_tensor_t(u_u, v_v, pm, pn):
    """delta, Sn (grid), zeta, Ss (grid) on rho points, vectorised over time."""
    pm_u = 0.5 * (pm[:, :-1] + pm[:, 1:])
    pn_u = 0.5 * (pn[:, :-1] + pn[:, 1:])
    pm_v = 0.5 * (pm[:-1, :] + pm[1:, :])
    pn_v = 0.5 * (pn[:-1, :] + pn[1:, :])
    pm_psi = 0.25 * (pm[:-1, :-1] + pm[:-1, 1:] + pm[1:, :-1] + pm[1:, 1:])
    pn_psi = 0.25 * (pn[:-1, :-1] + pn[:-1, 1:] + pn[1:, :-1] + pn[1:, 1:])
    Uflux, Vflux = u_u / pn_u[None], v_v / pm_v[None]
    nt, R, Cc = u_u.shape[0], pm.shape[0], pm.shape[1]
    dUdxi = np.full((nt, R, Cc), np.nan)
    dVdeta = np.full((nt, R, Cc), np.nan)
    dUdxi[:, :, 1:-1] = Uflux[:, :, 1:] - Uflux[:, :, :-1]
    dVdeta[:, 1:-1, :] = Vflux[:, 1:, :] - Vflux[:, :-1, :]
    delta = pm[None] * pn[None] * (dUdxi + dVdeta)
    Sn_g = pm[None] * pn[None] * (dUdxi - dVdeta)
    Vf, Uf = v_v / pn_v[None], u_u / pm_u[None]
    dVdxi = Vf[:, :, 1:] - Vf[:, :, :-1]
    dUdeta = Uf[:, 1:, :] - Uf[:, :-1, :]
    zeta = psi2rho_t(pm_psi[None] * pn_psi[None] * (dVdxi - dUdeta))
    Ss_g = psi2rho_t(pm_psi[None] * pn_psi[None] * (dVdxi + dUdeta))
    return delta, Sn_g, zeta, Ss_g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--eta-chunk', type=int, default=24)
    ap.add_argument('--workers', type=int, default=-1)
    ap.add_argument('--no-preload', action='store_true',
                    help='read level by level instead of one 4D block per row block')
    args = ap.parse_args()
    st = args.stride
    t_start = _time.time()

    arc = C.Archive()
    arc.describe(TAG)
    arc.assert_even(TAG)
    have_w = arc.has('w')
    g = C.load_grid(arc, stride=st)
    lon, lat, h, mask = g['lon'], g['lat'], g['h'], g['mask']
    pm, pn, angle, land, area = g['pm'], g['pn'], g['angle'], g['land'], g['area']
    neta, nxi = lon.shape
    nt, nlev = arc.nt, arc.nlev
    C.report(TAG, f'grid {neta} x {nxi} x {nlev} (stride {st}), {nt} records, w {"present" if have_w else "ABSENT"}')

    fb = C.FilterBank(lat, land, arc.dt_h, nt, workers=args.workers)
    fb.attach_time(arc.time)
    fb.describe(TAG)
    valid = fb.valid
    idx = C.window_indices(arc.time, valid)
    wsl = {k: C.as_slice(v) for k, v in idx.items()}
    days = np.unique(arc.time[valid].astype('datetime64[D]'))
    day_of = arc.time.astype('datetime64[D]')
    dsl = [C.as_slice(np.where((day_of == d) & valid)[0]) for d in days]

    C.report(TAG, 'time-mean free surface and layer thicknesses')
    sel_z = dict(eta_rho=C.ETA_SLICE, xi_rho=C.XI_SLICE)
    zeta_m = mean_zeta(arc, st, sel_z, nt)
    Hz = np.nan_to_num(C.compute_Hz(arc.ds0, zeta_m, st, h=h), nan=0.0)
    C.report(TAG, f'  done after {(_time.time() - t_start) / 60:.1f} min')

    # ---- accumulators ----------------------------------------------------------
    shp = (neta, nxi)
    terms = ('PiH', 'PiV', 'PiHdef', 'PiHdiv')
    W = {t: {k: np.zeros(shp) for k in C.WKEYS} for t in terms}
    D = {t: np.zeros((days.size,) + shp, np.float32) for t in terms}

    def read_block(var, de, dx, r0, r1, lev=None):
        sel = {de: slice(C.ETA0 + r0 * st, C.ETA0 + r1 * st, st), dx: slice(C.XI0, None, st)}
        if lev is not None:
            sel['s_rho' if 's_rho' in arc.ds0[var].dims else 's_w'] = lev
        a = arc.read(var, sel, dtype=np.float32 if lev is None else np.float64)
        ncol = nxi - 1 if dx == 'xi_u' else nxi
        return a[..., :ncol]

    nchunk = int(np.ceil(neta / args.eta_chunk))
    gb = 4 * nt * nlev * (args.eta_chunk + 2 * HALO) * nxi * 3 / 1e9
    C.report(TAG, f'{nchunk} row blocks of {args.eta_chunk} rows, about {gb:.0f} GB for the u v w of one block'
                  + (' (not preloaded)' if args.no_preload else '') + ', pass starts')
    t0 = _time.time()
    for ic, j0 in enumerate(range(0, neta, args.eta_chunk)):
        tb = _time.time()
        j1 = min(j0 + args.eta_chunk, neta)
        nrow = j1 - j0
        a0, a1 = max(j0 - HALO, 0), min(j1 + HALO, neta)          # rho rows with halo
        b0, b1 = max(j0 - HALO, 0), min(j1 + 1, neta - 1)         # v rows
        pmL, pnL = pm[a0:a1, :], pn[a0:a1, :]
        s0, sv = j0 - a0, j0 - b0
        if not args.no_preload:
            # the block is read level by level into float32 (nt, nlev, rows, cols) so the
            # log shows the read advancing; the read is the larger part of each block
            U = np.empty((nt, nlev, a1 - a0, nxi - 1), np.float32)
            V = np.empty((nt, nlev, b1 - b0, nxi), np.float32)
            Wf = np.empty((nt, nlev, nrow, nxi), np.float32) if have_w else None
            tr = _time.time()
            for kk in range(nlev):
                U[:, kk] = read_block('u', 'eta_u', 'xi_u', a0, a1, kk)
                V[:, kk] = read_block('v', 'eta_v', 'xi_v', b0, b1, kk)
                if have_w:
                    Wf[:, kk] = read_block('w', 'eta_rho', 'xi_rho', j0, j1, kk)
                if ic == 0 and (kk % 5 == 4 or kk == nlev - 1):
                    per = (_time.time() - tr) / (kk + 1)
                    C.report(TAG, f'    block 1, read level {kk + 1:2d}/{nlev}, {per:5.1f} s per level, '
                                  f'read of one block about {per * nlev / 60:.1f} min, '
                                  f'reads of the whole pass about {per * nlev * nchunk / 3600:.1f} h')
            if ic == 0:
                C.report(TAG, f'    block 1 read in {(_time.time() - tr) / 60:.1f} min, filtering starts')
        t_read = _time.time() - tb
        tc = _time.time()
        acc_w = {t: {k: np.zeros((nrow, nxi)) for k in C.WKEYS} for t in terms}
        acc_d = {t: np.zeros((days.size, nrow, nxi)) for t in terms}
        up_prev = vp_prev = ub_prev = vb_prev = None
        for kk in range(nlev):
            tl = _time.time()
            if args.no_preload:
                u = read_block('u', 'eta_u', 'xi_u', a0, a1, kk)
                v = read_block('v', 'eta_v', 'xi_v', b0, b1, kk)
            else:
                u = U[:, kk].astype(np.float64)
                v = V[:, kk].astype(np.float64)
            fu = fb.apply_both(u, a0)
            fv = fb.apply_both(v, b0)
            del u, v
            dlt, Sng, _, Ssg = cgrid_tensor_t(
                np.nan_to_num(smooth_h(fu['lp'], C.SMOOTH_CELLS), nan=0.0),
                np.nan_to_num(smooth_h(fv['lp'], C.SMOOTH_CELLS), nan=0.0), pmL, pnL)
            dlt, Sng, Ssg = (a[:, s0:s0 + nrow, :] for a in (dlt, Sng, Ssg))
            upr = C.u2rho(fu['bp'])[:, s0:s0 + nrow, :]
            vpr = C.v2rho(fv['bp'])[:, sv:sv + nrow, :]
            ulr = C.u2rho(fu['lp'])[:, s0:s0 + nrow, :]
            vlr = C.v2rho(fv['lp'])[:, sv:sv + nrow, :]
            del fu, fv
            pa = lambda a: fb.phase_average(a, j0)
            dev = pa(upr * upr - vpr * vpr)          # <u'u'> - <v'v'>
            tra = pa(upr * upr + vpr * vpr)          # <u'u'> + <v'v'> = 2 K
            uv = pa(upr * vpr)
            PiHdef = -C.RHO0 * (0.5 * dev * Sng + uv * Ssg)      # deformation work
            PiHdiv = -C.RHO0 * (0.5 * tra * dlt)                 # divergence work, -rho0 K delta
            del dev, tra, uv, dlt, Sng, Ssg
            hz = Hz[kk, j0:j1, :]
            for k in C.WKEYS:
                s = wsl[k]
                acc_w['PiHdef'][k] += PiHdef[s].mean(axis=0) * hz
                acc_w['PiHdiv'][k] += PiHdiv[s].mean(axis=0) * hz
            for d, s in enumerate(dsl):
                acc_d['PiHdef'][d] += PiHdef[s].mean(axis=0) * hz
                acc_d['PiHdiv'][d] += PiHdiv[s].mean(axis=0) * hz
            del PiHdef, PiHdiv
            if have_w and kk > 0:
                if args.no_preload:
                    wint = read_block('w', 'eta_rho', 'xi_rho', j0, j1, kk)
                else:
                    wint = Wf[:, kk].astype(np.float64)
                wp = fb.apply(wint, j0, 'bp')
                del wint
                uw = pa(0.5 * (upr + up_prev) * wp)
                vw = pa(0.5 * (vpr + vp_prev) * wp)
                del wp
                dz = np.maximum(0.5 * (Hz[kk, j0:j1, :] + Hz[kk - 1, j0:j1, :]), 1e-3)
                PiV = -C.RHO0 * (uw * (ulr - ub_prev) / dz[None] + vw * (vlr - vb_prev) / dz[None])
                del uw, vw
                for k in C.WKEYS:
                    acc_w['PiV'][k] += PiV[wsl[k]].mean(axis=0) * dz
                for d, s in enumerate(dsl):
                    acc_d['PiV'][d] += PiV[s].mean(axis=0) * dz
                del PiV
            up_prev, vp_prev, ub_prev, vb_prev = upr, vpr, ulr, vlr
            if ic == 0 and (kk % 5 == 0 or kk == nlev - 1):
                per_lev = (_time.time() - tc) / (kk + 1)
                blk = t_read + per_lev * nlev
                C.report(TAG, f'    block 1, level {kk + 1:2d}/{nlev}, {_time.time() - tl:5.1f} s, '
                              f'block so far {(_time.time() - tb) / 60:.1f} min, one block about '
                              f'{blk / 60:.0f} min, whole pass about {blk * nchunk / 3600:.1f} h, finish near '
                              f'{_time.strftime("%d %b %H:%M", _time.localtime(tb + blk * nchunk))}')
        if not args.no_preload:
            del U, V, Wf
        for k in C.WKEYS:
            W['PiHdef'][k][j0:j1] = acc_w['PiHdef'][k]
            W['PiHdiv'][k][j0:j1] = acc_w['PiHdiv'][k]
            W['PiV'][k][j0:j1] = acc_w['PiV'][k]
            W['PiH'][k][j0:j1] = acc_w['PiHdef'][k] + acc_w['PiHdiv'][k]
        D['PiHdef'][:, j0:j1] = acc_d['PiHdef']
        D['PiHdiv'][:, j0:j1] = acc_d['PiHdiv']
        D['PiV'][:, j0:j1] = acc_d['PiV']
        D['PiH'][:, j0:j1] = acc_d['PiHdef'] + acc_d['PiHdiv']
        del acc_w, acc_d
        el = _time.time() - t0
        C.report(TAG, f'  block {ic + 1:2d}/{nchunk}, rows {j0:5d}-{j1:5d}, '
                      f'{(_time.time() - tb) / 60:.1f} min, elapsed {el / 60:.0f} min, about '
                      f'{el / (ic + 1) * (nchunk - ic - 1) / 60:.0f} min left, finish near '
                      f'{_time.strftime("%d %b %H:%M", _time.localtime(_time.time() + el / (ic + 1) * (nchunk - ic - 1)))}')

    for t in terms:
        for k in C.WKEYS:
            W[t][k][land] = np.nan
        D[t][:, land] = np.nan

    # ---- sector series and window means --------------------------------------------
    masks = C.sector_masks(g, stride=st)
    shelf = masks['shelf']
    ser = {t: np.full((len(C.SECTOR_NAMES), days.size), np.nan) for t in terms}
    for t in terms:
        for si, sec in enumerate(C.SECTOR_NAMES):
            for di in range(days.size):
                ser[t][si, di] = C.area_mean(D[t][di], shelf[sec], area)
    ser['Pi'] = ser['PiH'] + ser['PiV']
    table = {}
    for sec in C.SECTOR_NAMES:
        table[sec] = {}
        for k in C.WKEYS:
            row = {t: C.area_mean(W[t][k], shelf[sec], area) for t in terms}
            row['Pi'] = row['PiH'] + row['PiV']
            table[sec][k] = row

    # ---- reversal, lag and e-folding of the South drain ------------------------------
    landfall = C.LANDFALL_DEFAULT
    if os.path.exists(FIG01_NC):
        with xr.open_dataset(FIG01_NC) as d1:
            landfall = np.datetime64(str(d1.attrs.get('landfall', landfall)))
    Tf_south = C.inertial_period_hours(float(np.nanmean(lat[shelf['South']])))
    s_south = ser['Pi'][C.SECTOR_NAMES.index('South')]
    after = days + np.timedelta64(12, 'h') > landfall
    stats_rev = {}
    imin = int(np.nanargmin(np.where(after, s_south, np.nan)))
    pos = np.where(after & (np.arange(days.size) > imin) & (s_south > 0))[0]
    if pos.size:
        irev = int(pos[0])
        t_rev = days[irev] + np.timedelta64(12, 'h')
        lag_h = (t_rev - landfall).astype('timedelta64[s]').astype(float) / 3600.0
        stats_rev = dict(reversal_day=str(days[irev])[:10], lag_hours=float(lag_h),
                         lag_inertial_periods=float(lag_h / Tf_south),
                         minimum_day=str(days[imin])[:10], minimum_mW_m2=float(s_south[imin]) * 1e3)
        seg = s_south[imin:irev]
        if seg.size >= 3 and np.all(seg < 0):
            tt = np.arange(seg.size, dtype=float)
            p = np.polyfit(tt, np.log(-seg), 1)
            stats_rev['drain_efold_days'] = float(-1.0 / p[0]) if p[0] < 0 else np.nan
    C.report(TAG, f'South reversal {stats_rev}')

    sub = {}
    if os.path.exists(FIG02_JSON):
        with open(FIG02_JSON) as fh:
            s2 = json.load(fh)['sectors']
        for sec in C.SECTOR_NAMES:
            sub[sec] = {k: dict(delta_rms=s2[sec][f'delta_rms_{k}_s-1'], Sigma_rms=s2[sec][f'Sigma_rms_{k}_s-1'],
                                ratio=s2[sec][f'delta_rms_{k}_s-1'] / s2[sec][f'Sigma_rms_{k}_s-1'],
                                delta_rms_over_f=s2[sec][f'delta_rms_{k}_s-1'] / abs(C.coriolis(float(np.nanmean(lat[shelf[sec]])))))
                        for k in C.WKEYS}

    # ---- report and Table 2 ----------------------------------------------------------
    L = ['FIGURE 3 REPORT, numbers for the text', f'run {_time.strftime("%Y-%m-%d %H:%M")}',
         f'valid record {str(fb.valid_span()[0])[:16]} to {str(fb.valid_span()[1])[:16]}, stride {st}, '
         f'shelf and slope h <= {C.STAT_MAX_DEPTH_M:.0f} m, area weighted, positive from the subinertial flow to the waves', '']
    L.append('window-mean exchange rho0 int Pi dz, mW m-2 (Table 2)')
    L.append(f'  {"sector":8s} {"term":12s} ' + ' '.join(f'{k:>9s}' for k in C.WKEYS))
    names = {'Pi': 'Pi', 'PiH': 'Pi_H', 'PiHdef': 'deformation', 'PiHdiv': 'divergence', 'PiV': 'Pi_V'}
    for sec in C.SECTOR_NAMES:
        for t in ('Pi', 'PiH', 'PiHdef', 'PiHdiv', 'PiV'):
            L.append(f'  {sec:8s} {names[t]:12s} ' + ' '.join(f'{table[sec][k][t] * 1e3:+9.4f}' for k in C.WKEYS))
    L += ['', 'South, share of the vertical shear production in the total per window: ' +
          ', '.join(f'{k} {table["South"][k]["PiV"] / table["South"][k]["Pi"]:.2f}' if table['South'][k]['Pi'] else f'{k} nan'
                    for k in C.WKEYS)]
    L += ['', f'South sector reversal: {json.dumps(stats_rev)}',
          f'  landfall {str(landfall)[:16]}, South inertial period {Tf_south:.1f} h']
    if sub:
        L += ['', 'subinertial flow of the upper 100 m (from fig02): delta_rms / Sigma_rms and delta_rms / |f| per window']
        for sec in C.SECTOR_NAMES:
            L.append(f'  {sec:8s} ' + ' '.join(f'{k} {sub[sec][k]["ratio"]:.2f} ({sub[sec][k]["delta_rms_over_f"]:.3f})'
                                                for k in C.WKEYS))
    L += ['', f'filters: {json.dumps(fb.config())}', f'wall time {(_time.time() - t_start) / 3600:.2f} h']
    C.write_report(OUT_REPORT, L)
    print('\n'.join(L))

    tex = [r'\begin{table}[H]', r'\centering',
           r'\caption{Window-mean exchange over the shelf and slope shallower than 1000~m, in mW~m$^{-2}$, '
           r'positive from the subinertial flow to the waves. The deformation work and the divergence work '
           r'are the two terms of the polar form (Equation~\ref{eq:polar}).}',
           r'\label{tab:exchange}', r'\begin{tabular}{llrrrr}', r'\toprule',
           r'Sector & Term & $T1$ & $T2$ & $T3$ & $T4$ \\', r'\midrule']
    tex_names = {'Pi': r'$\Pi$', 'PiH': r'$\PiH$', 'PiHdef': 'deformation work', 'PiHdiv': 'divergence work', 'PiV': r'$\PiV$'}
    for t in ('Pi', 'PiH', 'PiHdef', 'PiHdiv', 'PiV'):
        tex.append(f'South & {tex_names[t]} & ' + ' & '.join(f'{table["South"][k][t] * 1e3:+.3f}' for k in C.WKEYS) + r' \\')
    for sec in ('Central', 'North'):
        tex.append(f'{sec} & $\\Pi$ & ' + ' & '.join(f'{table[sec][k]["Pi"] * 1e3:+.3f}' for k in C.WKEYS) + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    with open(OUT_TABLE, 'w') as fh:
        fh.write('\n'.join(tex) + '\n')
    print(f'wrote {OUT_TABLE}')

    # ---- archive ----------------------------------------------------------------------------
    data = {'h': (('eta', 'xi'), h), 'mask': (('eta', 'xi'), mask), 'area': (('eta', 'xi'), area),
            'sector': (('eta', 'xi'), masks['sector_id'])}
    for t in terms:
        for k in C.WKEYS:
            data[f'{t}_{k}'] = (('eta', 'xi'), W[t][k])
        data[f'{t}_daily'] = (('day', 'eta', 'xi'), D[t])
        data[f'{t}_series'] = (('sector_name', 'day'), ser[t])
    data['Pi_series'] = (('sector_name', 'day'), ser['Pi'])
    out = xr.Dataset(data_vars=data,
                     coords=dict(lon=(('eta', 'xi'), lon), lat=(('eta', 'xi'), lat),
                                 day=days.astype('datetime64[ns]'), sector_name=list(C.SECTOR_NAMES)),
                     attrs=dict(source=' + '.join(arc.paths), rho0=C.RHO0, stride=st, have_w=int(have_w),
                                **{k: list(v) for k, v in C.WINDOWS.items()}, landfall=str(landfall)[:16],
                                Tf_south_h=Tf_south, stat_max_depth_m=C.STAT_MAX_DEPTH_M,
                                south_top=masks['south_top'], central_top=masks['central_top'],
                                smooth_cells=C.SMOOTH_CELLS, filters=json.dumps(fb.config()),
                                reversal=json.dumps(stats_rev),
                                note=('Depth-integrated exchange terms times rho0, W m-2, window means and '
                                      'daily means over the valid record, phase averaged over one local '
                                      'inertial period. PiH = PiHdef + PiHdiv.'),
                                created=_time.strftime('%Y-%m-%d %H:%M')))
    for v in out.data_vars:
        if out[v].dtype.kind == 'f':
            out[v].encoding.update(zlib=True, complevel=4)
    out.to_netcdf(OUT_NC)
    with open(OUT_JSON, 'w') as fh:
        json.dump(dict(table={s: {k: {t: float(v) for t, v in r.items()} for k, r in d.items()}
                              for s, d in table.items()},
                       reversal=stats_rev, subinertial=sub, filters=fb.config(), Tf_south_h=Tf_south),
                  fh, indent=2, default=float)
    print(f'wrote {OUT_NC} and {OUT_JSON}')
    arc.close()


if __name__ == '__main__':
    main()