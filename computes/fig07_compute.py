"""
fig07_compute.py

The wake decay and the kinetic energy pathway (Figure 7, second part of
Question 4). Two sources.

1. A light pass over the South rows at a reduced stride (default 3, about a
   tenth of the Figure 6 pass) that writes the DAILY sector series of every
   term of the near-inertial kinetic energy budget (W_NI, Pi_H, Pi_V, <w'b'>,
   eps_NI, bottom drag, the advective and radiative transport divergences) and
   the depth-integrated K at the start of every day, so the cumulative curves
   of panel (b) are read from one consistent set and their residual closes.
   The discretisation is that of fig06_compute.py.

2. The Figure 2 and 3 archives (daily South K and Pi at full resolution) for
   the decay fits of panel (a), and the Figure 6 statistics for the window
   magnitudes of the pathway of panel (c).

The decay is fitted as K = K_peak exp(-(t - t_peak) / tau) by least squares
on log K, from the peak day through the end of T3 and separately through T4,
and the e-folding of each fit is reported.

    nohup python -u fig07_compute.py --stride 3 > fig07/log_fig07.txt 2>&1 &   # about 1.5 h
"""

import argparse
import json
import os
import time as _time

import netCDF4 as nc
import numpy as np
import xarray as xr

import common_v3 as C
from fig02_compute import mean_zeta
from fig03_compute import HALO, cgrid_tensor_t, smooth_h
from fig06_compute import flux_divergence

TAG = 'fig07'
FIG_DIR = C.fig_dir('fig07')
FIG02_NC = os.path.join(os.getcwd(), 'fig02', 'fig02_data.nc')
FIG03_NC = os.path.join(os.getcwd(), 'fig03', 'fig03_data.nc')
FIG06_JSON = os.path.join(os.getcwd(), 'fig06', 'fig06_stats.json')
OUT_NC = os.path.join(FIG_DIR, 'fig07_data.nc')
OUT_JSON = os.path.join(FIG_DIR, 'fig07_stats.json')
OUT_REPORT = os.path.join(FIG_DIR, 'fig07_report.txt')

SECTOR = 'South'
CUM_START = 'T2'                   # the cumulative curves start with this window
TERMS = ('W_NI', 'PiH', 'PiV', 'wb', 'eps_NI', 'drag_NI', 'div_adv', 'div_rad')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--eta-chunk', type=int, default=36)
    ap.add_argument('--workers', type=int, default=-1)
    args = ap.parse_args()
    st = args.stride
    t_start = _time.time()
    smooth = max(1, int(round(C.SMOOTH_CELLS / st)))

    arc = C.Archive()
    arc.describe(TAG)
    arc.assert_even(TAG)
    have_akv = arc.has('AKv')
    g = C.load_grid(arc, stride=st)
    lon, lat, h, mask = g['lon'], g['lat'], g['h'], g['mask']
    pm, pn, land, area = g['pm'], g['pn'], g['land'], g['area']
    neta_all, nxi = lon.shape
    nt, nlev = arc.nt, arc.nlev
    masks = C.sector_masks(g, stride=st)
    shelf = masks['shelf'][SECTOR]
    neta = int(np.where(shelf.any(axis=1))[0].max()) + 1
    C.report(TAG, f'{SECTOR} rows 0 to {neta} of {neta_all} x {nxi} x {nlev} (stride {st}), {nt} records, '
                  f'AKv {"present" if have_akv else "ABSENT"}')

    sf = nc.Dataset(C.SURFACE_FILE)
    sf.set_auto_mask(False)
    if sf.dimensions['time'].size != nt:
        raise SystemExit(f'{C.SURFACE_FILE} holds {sf.dimensions["time"].size} records, the archive {nt}')

    fb = C.FilterBank(lat, land, arc.dt_h, nt, workers=args.workers)
    fb.attach_time(arc.time)
    fb.describe(TAG)
    valid = fb.valid
    idx = C.window_indices(arc.time, valid)
    days = np.unique(arc.time[valid].astype('datetime64[D]'))
    day_of = arc.time.astype('datetime64[D]')
    dsl = [C.as_slice(np.where((day_of == d) & valid)[0]) for d in days]
    dfirst = [int(np.where((day_of == d) & valid)[0][0]) for d in days]
    sel_z = dict(eta_rho=C.ETA_SLICE, xi_rho=C.XI_SLICE)
    Hz = np.nan_to_num(C.compute_Hz(arc.ds0, mean_zeta(arc, st, sel_z, nt), st, h=h), nan=0.0)

    nd = days.size
    D = {t: np.zeros((nd, neta, nxi)) for t in TERMS}
    K0 = np.zeros((nd, neta, nxi))                     # int K dz at the first record of each day
    Kd = np.zeros((nd, neta, nxi))                     # daily mean int K dz
    Fadv_u = np.zeros((nd, neta, nxi - 1)); Frad_u = np.zeros((nd, neta, nxi - 1))
    Fadv_v = np.zeros((nd, neta - 1, nxi)); Frad_v = np.zeros((nd, neta - 1, nxi))

    def read_block(var, de, dx, r0, r1, lev=None):
        sel = {de: slice(C.ETA0 + r0 * st, C.ETA0 + r1 * st, st), dx: slice(C.XI0, None)}
        if lev is not None:
            sel['s_rho' if 's_rho' in arc.ds0[var].dims else 's_w'] = lev
        a = arc.read(var, sel)[..., ::st]
        return a[..., :nxi - 1] if dx == 'xi_u' else a[..., :nxi]

    def read_surface(name, r0, r1, ncol):
        return sf.variables[name][:, r0 * st:r1 * st:st, ::st].astype(np.float64)[..., :ncol]

    nchunk = int(np.ceil(neta / args.eta_chunk))
    t0 = _time.time()
    for ic, j0 in enumerate(range(0, neta, args.eta_chunk)):
        tb = _time.time()
        j1 = min(j0 + args.eta_chunk, neta)
        nrow = j1 - j0
        a0, a1 = max(j0 - HALO, 0), min(j1 + HALO, neta_all)
        b0, b1 = a0, a1 - 1
        na = a1 - a0
        s0 = j0 - a0
        w0, w1 = j0, min(j1, neta - 1)
        pmL, pnL = pm[a0:a1, :], pn[a0:a1, :]
        hz_all = Hz[:, a0:a1, :]
        su, sv = read_surface('sustr', a0, a1, nxi - 1), read_surface('svstr', b0, b1, nxi)
        us, vs = read_surface('u_s', a0, a1, nxi - 1), read_surface('v_s', b0, b1, nxi)
        zt = read_surface('zeta', a0, a1, nxi)
        wni = fb.phase_average(C.u2rho(fb.apply(su, a0, 'bp') * fb.apply(us, a0, 'bp'))
                               + C.v2rho(fb.apply(sv, b0, 'bp') * fb.apply(vs, b0, 'bp')), a0)[:, s0:s0 + nrow]
        del su, sv, us, vs
        for d, s in enumerate(dsl):
            D['W_NI'][d, j0:j1] = wni[s].mean(axis=0)
        del wni
        zp = fb.apply(zt, a0, 'bp')
        del zt
        p_acc = np.zeros((nt, na, nxi))
        K_int = np.zeros((nt, nrow, nxi))
        prev = None
        for kk in range(nlev - 1, -1, -1):
            tl = _time.time()
            u_raw = read_block('u', 'eta_u', 'xi_u', a0, a1, kk)
            v_raw = read_block('v', 'eta_v', 'xi_v', b0, b1, kk)
            fu, fv = fb.apply_both(u_raw, a0), fb.apply_both(v_raw, b0)
            rp = fb.apply(read_block('rho', 'eta_rho', 'xi_rho', a0, a1, kk), a0, 'bp')
            hz = hz_all[kk]
            p_k = C.RHO0 * C.G * zp + C.G * (p_acc + 0.5 * rp * hz[None])
            p_acc += rp * hz[None]
            bp = -(C.G / C.RHO0) * rp[:, s0:s0 + nrow]
            del rp
            upr, vpr = C.u2rho(fu['bp']), C.v2rho(fv['bp'])
            ulr, vlr = C.u2rho(fu['lp']), C.v2rho(fv['lp'])
            uu, vv, uv = (fb.phase_average(upr * upr, a0), fb.phase_average(vpr * vpr, a0),
                          fb.phase_average(upr * vpr, a0))
            K_k = 0.5 * C.RHO0 * (uu + vv)
            dlt, Sng, _, Ssg = cgrid_tensor_t(np.nan_to_num(smooth_h(fu['lp'], smooth), nan=0.0),
                                              np.nan_to_num(smooth_h(fv['lp'], smooth), nan=0.0), pmL, pnL)
            PiH = -C.RHO0 * (0.5 * (uu - vv) * Sng + uv * Ssg + 0.5 * (uu + vv) * dlt)[:, s0:s0 + nrow]
            del dlt, Sng, Ssg, uu, vv, uv
            E_u, E_v = 0.5 * (K_k[:, :, :-1] + K_k[:, :, 1:]), 0.5 * (K_k[:, :-1, :] + K_k[:, 1:, :])
            p_u, p_v = 0.5 * (p_k[:, :, :-1] + p_k[:, :, 1:]), 0.5 * (p_k[:, :-1, :] + p_k[:, 1:, :])
            fa_u, fa_v = fu['lp'] * E_u, fv['lp'] * E_v
            fr_u, fr_v = fb.phase_average(fu['bp'] * p_u, a0), fb.phase_average(fv['bp'] * p_v, b0)
            del E_u, E_v, p_u, p_v, p_k
            hz_u, hz_v = 0.5 * (hz[:, :-1] + hz[:, 1:]), 0.5 * (hz[:-1, :] + hz[1:, :])
            K_int += K_k[:, s0:s0 + nrow] * hz[None, s0:s0 + nrow]
            for d, s in enumerate(dsl):
                D['PiH'][d, j0:j1] += PiH[s].mean(axis=0) * hz[s0:s0 + nrow]
                Fadv_u[d, j0:j1] += (fa_u[s].mean(axis=0) * hz_u)[s0:s0 + nrow]
                Frad_u[d, j0:j1] += (fr_u[s].mean(axis=0) * hz_u)[s0:s0 + nrow]
                if w1 > w0:
                    Fadv_v[d, w0:w1] += (fa_v[s].mean(axis=0) * hz_v)[s0:s0 + (w1 - w0)]
                    Frad_v[d, w0:w1] += (fr_v[s].mean(axis=0) * hz_v)[s0:s0 + (w1 - w0)]
            del fa_u, fa_v, fr_u, fr_v, PiH
            if prev is not None:
                up_p, vp_p, ul_p, vl_p, bp_p = prev
                wp = fb.apply(read_block('w', 'eta_rho', 'xi_rho', j0, j1, kk + 1), j0, 'bp')
                dz = np.maximum(0.5 * (Hz[kk, j0:j1, :] + Hz[kk + 1, j0:j1, :]), 1e-3)[None]
                upo, vpo = upr[:, s0:s0 + nrow], vpr[:, s0:s0 + nrow]
                ulo, vlo = ulr[:, s0:s0 + nrow], vlr[:, s0:s0 + nrow]
                uw, vw = fb.phase_average(0.5 * (upo + up_p) * wp, j0), fb.phase_average(0.5 * (vpo + vp_p) * wp, j0)
                PiV = -C.RHO0 * (uw * (ul_p - ulo) / dz + vw * (vl_p - vlo) / dz)
                wb = C.RHO0 * fb.phase_average(wp * 0.5 * (bp + bp_p), j0)
                sh2 = fb.phase_average(((up_p - upo) / dz) ** 2 + ((vp_p - vpo) / dz) ** 2, j0)
                if have_akv:
                    eps = C.RHO0 * fb.phase_average(read_block('AKv', 'eta_rho', 'xi_rho', j0, j1, kk + 1) * sh2, j0)
                else:
                    eps = np.zeros_like(sh2)
                for d, s in enumerate(dsl):
                    D['PiV'][d, j0:j1] += PiV[s].mean(axis=0) * dz[0]
                    D['wb'][d, j0:j1] += wb[s].mean(axis=0) * dz[0]
                    D['eps_NI'][d, j0:j1] += eps[s].mean(axis=0) * dz[0]
                del wp, uw, vw, PiV, wb, sh2, eps
            if kk == 0:
                ub, vb = C.u2rho(u_raw)[:, s0:s0 + nrow], C.v2rho(v_raw)[:, s0:s0 + nrow]
                spd = np.hypot(ub, vb)
                dni = C.RHO0 * C.CD_BOT * fb.phase_average(spd * (ub * upr[:, s0:s0 + nrow] + vb * vpr[:, s0:s0 + nrow]), j0)
                for d, s in enumerate(dsl):
                    D['drag_NI'][d, j0:j1] = dni[s].mean(axis=0)
                del ub, vb, spd, dni
            prev = (upr[:, s0:s0 + nrow], vpr[:, s0:s0 + nrow], ulr[:, s0:s0 + nrow], vlr[:, s0:s0 + nrow], bp)
            del u_raw, v_raw, fu, fv, upr, vpr, ulr, vlr, K_k
            if ic == 0 and (kk % 5 == 0 or kk == nlev - 1):
                per = (_time.time() - tb) / (nlev - kk)
                C.report(TAG, f'    block 1, level {kk + 1:2d}/{nlev} (from the top), {_time.time() - tl:5.1f} s, '
                              f'whole pass about {per * nlev * nchunk / 60:.0f} min, finish near '
                              f'{_time.strftime("%d %b %H:%M", _time.localtime(tb + per * nlev * nchunk))}')
        del prev, p_acc, zp
        for d, s in enumerate(dsl):
            K0[d, j0:j1] = K_int[dfirst[d]]
            Kd[d, j0:j1] = K_int[s].mean(axis=0)
        del K_int
        el = _time.time() - t0
        C.report(TAG, f'  block {ic + 1:2d}/{nchunk}, elapsed {el / 60:.0f} min, about '
                      f'{el / (ic + 1) * (nchunk - ic - 1) / 60:.0f} min left')
    sf.close()

    mask_u = (mask[:neta, :-1] * mask[:neta, 1:]) > 0.5
    mask_v = (mask[:neta - 1, :] * mask[1:neta, :]) > 0.5
    pmS, pnS = pm[:neta], pn[:neta]
    for d in range(nd):
        for a, m in ((Fadv_u[d], mask_u), (Frad_u[d], mask_u), (Fadv_v[d], mask_v), (Frad_v[d], mask_v)):
            a[~m] = 0.0
        D['div_adv'][d] = flux_divergence(Fadv_u[d], Fadv_v[d], pmS, pnS)
        D['div_rad'][d] = flux_divergence(Frad_u[d], Frad_v[d], pmS, pnS)
    m = shelf[:neta]
    ar = area[:neta]
    for t in TERMS:
        D[t][:, land[:neta]] = np.nan
    ser = {t: np.array([C.area_mean(D[t][d], m, ar) for d in range(nd)]) for t in TERMS}
    K0s = np.array([C.area_mean(K0[d], m, ar) for d in range(nd)])
    Kds = np.array([C.area_mean(Kd[d], m, ar) for d in range(nd)])

    # ---- cumulative curves from the start of CUM_START -------------------------------------------
    i0 = int(np.where(days >= np.datetime64(C.WINDOWS[CUM_START][0]))[0][0])
    sec_per_day = 86400.0
    cum = {t: np.concatenate([[0.0], np.cumsum(ser[t][i0:] * sec_per_day)]) for t in TERMS}
    nK = nd - i0 + 1
    K_edge = np.append(K0s[i0:], Kds[-1] * 2 - K0s[-1])       # K at the day edges, the last edge extrapolated
    dK = K_edge - K_edge[0]
    cum['dK'] = dK
    cum['R'] = dK - (cum['W_NI'] + cum['PiH'] + cum['PiV'] + cum['wb'] - cum['eps_NI'] - cum['drag_NI']
                     - cum['div_adv'] - cum['div_rad'])
    edges = np.append(days[i0:], days[-1] + np.timedelta64(1, 'D'))

    # ---- decay fits from the Figure 2 archive --------------------------------------------------------
    fits = {}
    d2 = xr.open_dataset(FIG02_NC)
    secs = [str(x) for x in d2['sector_name'].values]
    Kfull = d2['K_series'].values[secs.index(SECTOR)]
    kdays = d2['day'].values.astype('datetime64[D]')
    ip = int(np.nanargmax(Kfull))
    fits['peak_day'] = str(kdays[ip])[:10]
    fits['peak_J_m2'] = float(Kfull[ip])
    for name, end in (('T3', C.WINDOWS['T3'][1]), ('T4', C.WINDOWS['T4'][1])):
        ie = int(np.where(kdays <= np.datetime64(end))[0][-1])
        tt = np.arange(ie - ip + 1, dtype=float)
        yy = np.log(Kfull[ip:ie + 1])
        p = np.polyfit(tt, yy, 1)
        pred = np.exp(np.polyval(p, tt))
        fits[f'efold_{name}_d'] = float(-1.0 / p[0]) if p[0] < 0 else np.nan
        fits[f'fit_{name}_end'] = str(kdays[ie])[:10]
        fits[f'fit_{name}_r2'] = float(1 - np.sum((yy - np.polyval(p, tt)) ** 2) / np.sum((yy - yy.mean()) ** 2))
    # the e-folding from the T3 fit alone and from the T4 segment alone
    i3 = int(np.where(kdays <= np.datetime64(C.WINDOWS['T3'][1]))[0][-1])
    i4 = int(np.where(kdays <= np.datetime64(C.WINDOWS['T4'][1]))[0][-1])
    tt = np.arange(i4 - i3, dtype=float)
    p4 = np.polyfit(tt, np.log(Kfull[i3 + 1:i4 + 1]), 1)
    fits['efold_T4_segment_d'] = float(-1.0 / p4[0]) if p4[0] < 0 else np.nan
    fits['K_end_J_m2'] = float(Kfull[-1])
    fits['K_end_over_peak'] = float(Kfull[-1] / Kfull[ip])
    d2.close()
    d3 = xr.open_dataset(FIG03_NC)
    Pifull = d3['Pi_series'].values[[str(x) for x in d3['sector_name'].values].index(SECTOR)]
    pdays = d3['day'].values.astype('datetime64[D]')
    d3.close()
    icum = int(np.nanargmin(np.cumsum(Pifull) * sec_per_day))
    fits['cum_Pi_min_kJ'] = float(np.min(np.cumsum(Pifull)) * sec_per_day * 1e-3)
    fits['cum_Pi_min_day'] = str(pdays[icum])[:10]
    fits['cum_Pi_end_kJ'] = float(np.sum(Pifull) * sec_per_day * 1e-3)
    fits['return_over_drain'] = float((fits['cum_Pi_end_kJ'] - fits['cum_Pi_min_kJ']) / -fits['cum_Pi_min_kJ'])

    # ---- the pathway magnitudes from Figure 6 -----------------------------------------------------------
    path = {}
    if os.path.exists(FIG06_JSON):
        with open(FIG06_JSON) as fh:
            B = json.load(fh)['budgets'][SECTOR]
        for k in C.WKEYS:
            r = B[k]
            path[k] = dict(W_NI=r['W_NI'] * 1e-3, Pi=(r['PiH'] + r['PiV']) * 1e-3, eps=r['eps_NI'] * 1e-3,
                           drag=r['drag_NI'] * 1e-3, transport=(r['div_adv'] + r['div_rad']) * 1e-3,
                           wb=r['wb'] * 1e-3, tend=r['tend'] * 1e-3, R=r['R'] * 1e-3)
    inj = cum['W_NI'][-1]
    fate = dict(injected_kJ=inj * 1e-3, exchange_pct=100 * (cum['PiH'][-1] + cum['PiV'][-1]) / inj,
                dissipated_pct=100 * cum['eps_NI'][-1] / inj, drag_pct=100 * cum['drag_NI'][-1] / inj,
                exported_pct=100 * (cum['div_adv'][-1] + cum['div_rad'][-1]) / inj,
                buoyancy_pct=100 * cum['wb'][-1] / inj, remaining_pct=100 * dK[-1] / inj,
                residual_pct=100 * cum['R'][-1] / inj)

    L = ['FIGURE 7 REPORT, the wake decay and the pathway', f'run {_time.strftime("%Y-%m-%d %H:%M")}',
         f'{SECTOR} shelf and slope h <= {C.STAT_MAX_DEPTH_M:.0f} m, daily series at stride {st}, cumulative from '
         f'{str(edges[0])[:10]} to {str(edges[-1])[:10]}', '',
         f'decay of K (Figure 2 series, full resolution): peak {fits["peak_J_m2"]:.0f} J m-2 on {fits["peak_day"]}, '
         f'e-folding {fits["efold_T3_d"]:.1f} d through {fits["fit_T3_end"]} (r2 {fits["fit_T3_r2"]:.2f}), '
         f'{fits["efold_T4_d"]:.1f} d through {fits["fit_T4_end"]} (r2 {fits["fit_T4_r2"]:.2f}), '
         f'{fits["efold_T4_segment_d"]:.1f} d over T4 alone, K at the end {fits["K_end_J_m2"]:.0f} J m-2 '
         f'({100 * fits["K_end_over_peak"]:.0f} % of the peak)',
         f'cumulative Pi (Figure 3 series): minimum {fits["cum_Pi_min_kJ"]:+.3f} kJ m-2 on {fits["cum_Pi_min_day"]}, '
         f'{fits["cum_Pi_end_kJ"]:+.3f} at the end, return {100 * fits["return_over_drain"]:.0f} % of the drain', '',
         'fate of the near-inertial injection from the start of T2 to the end of the record (this pass)']
    L.append(f'  injected {fate["injected_kJ"]:.2f} kJ m-2, exchange {fate["exchange_pct"]:+.1f} %, shear dissipation '
             f'{fate["dissipated_pct"]:.1f} %, bed drag {fate["drag_pct"]:.1f} %, lateral export {fate["exported_pct"]:+.1f} %, '
             f'buoyancy flux {fate["buoyancy_pct"]:+.1f} %, remaining in K {fate["remaining_pct"]:+.1f} %, residual {fate["residual_pct"]:+.1f} %')
    if path:
        L.append('')
        L.append('window magnitudes of the pathway (Figure 6, kJ m-2): W_NI, Pi, eps_NI, drag, transport, tendency, residual')
        for k in C.WKEYS:
            r = path[k]
            L.append(f'  {k}: {r["W_NI"]:.2f}, {r["Pi"]:+.3f}, {r["eps"]:.2f}, {r["drag"]:.2f}, {r["transport"]:+.3f}, {r["tend"]:+.2f}, {r["R"]:+.2f}')
        chk = {k: ser[t][np.where((days >= np.datetime64(C.WINDOWS[k][0])) & (days <= np.datetime64(C.WINDOWS[k][1])))[0]].sum() * sec_per_day * 1e-3
               for k, t in (('T2', 'W_NI'), ('T3', 'eps_NI'))}
        L.append(f'  check, this pass against Figure 6: W_NI T2 {chk["T2"] / path["T2"]["W_NI"]:.2f}, eps_NI T3 {chk["T3"] / path["T3"]["eps"]:.2f}')
    L += ['', f'wall time {(_time.time() - t_start) / 60:.0f} min']
    C.write_report(OUT_REPORT, L)
    print('\n'.join(L))

    data = {f'{t}_daily': (('day',), ser[t]) for t in TERMS}
    data['K0_daily'] = (('day',), K0s)
    data['K_daily'] = (('day',), Kds)
    for t, v in cum.items():
        data[f'cum_{t}'] = (('edge',), v)
    out = xr.Dataset(data, coords=dict(day=days.astype('datetime64[ns]'), edge=edges.astype('datetime64[ns]')),
                     attrs=dict(sector=SECTOR, stride=st, cum_start=CUM_START, have_akv=int(have_akv),
                                **{k: list(v) for k, v in C.WINDOWS.items()}, landfall=str(C.LANDFALL_DEFAULT)[:16],
                                fits=json.dumps(fits), fate=json.dumps(fate), pathway=json.dumps(path),
                                note=('Daily area-weighted depth-integrated terms of the near-inertial budget (W m-2) '
                                      'and the cumulative integrals from the start of the first storm window (J m-2) '
                                      'on the day edges, with dK the change of int K dz and R the residual.'),
                                created=_time.strftime('%Y-%m-%d %H:%M')))
    out.to_netcdf(OUT_NC)
    with open(OUT_JSON, 'w') as fh:
        json.dump(dict(fits=fits, fate=fate, pathway=path, config=dict(stride=st, cum_start=CUM_START)), fh, indent=2, default=float)
    print(f'wrote {OUT_NC} and {OUT_JSON}')
    arc.close()


if __name__ == '__main__':
    main()