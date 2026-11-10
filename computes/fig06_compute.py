"""
fig06_compute.py

The two kinetic energy budgets (Figure 6, Question 4, Table 4). One row-block
pass over the whole analysis domain, level by level from the surface down so
the hydrostatic wave pressure accumulates, reading u, v, w, rho and AKv from
the archive and the surface stress and free surface from surface_v3.nc.

Every term is a window mean, depth integrated, times rho0, in W m-2 on the
grid, and the sector values are area weighted over the shelf and slope (h <=
1000 m). The report multiplies them by the window length, J m-2, which is how
Figure 6 and Table 4 quote them.

Near-inertial budget (Equation nike of the text), positive terms feed K
    tend     rho0 [int K dz (end) - int K dz (start)] / window length
    W_NI     <tau' . u'_s>, band-passed stress and surface velocity
    PiH, PiV the two shear productions (as in fig03_compute.py)
    wb       rho0 <w' b'>, b' = -g rho' / rho0 (band-passed density)
    eps_NI   rho0 <A_v |d_z u'_h|^2> from the vertical viscosity of the closure
    drag_NI  rho0 C_d <|u_b| u_b . u'_b> at the bottom level, the viscous flux at the bed
    div_adv  div_h of int ubar_h K dz (flux form on the C grid)
    div_rad  div_h of int <u'_h p'> dz, p' = rho0 g eta' + g int_z^0 rho' dz'
    R        tend + div_adv + div_rad - W_NI - PiH - PiV - wb + eps_NI + drag_NI
Subinertial budget (Equation lfke), sinks and sources of Kbar = 0.5 |ubar_h|^2
    tend_b   rho0 [int Kbar dz (end) - (start)] / window length
    W_S      taubar . ubar_s, low-passed stress and surface velocity
    -Pi      the exchange with the opposite sign
    drag_b   rho0 C_d |u_b| u_b . ubar_b at the bottom level
    div_ws   div_h of int ubar_h . <u' u'> dz, the wave-stress work (comment 50)
    R_b      tend_b + div_ws - W_S + Pi + drag_b (holds the advective and pressure
             transports of Kbar, the mean conversion and the closure dissipation)

Also written: daily sector series of int Kbar dz and of W_NI, the window maps
of the transports for panels (e) and (f), alpha_Pi per sector and stage with
moving-block bootstrap intervals (unresolved where the T2 injection is below
three times the T1 injection), and Table 4 for the South sector.

    nohup python -u fig06_compute.py --stride 2 > fig06/log_fig06.txt 2>&1 &   # about 4 h
    nohup python -u fig06_compute.py > fig06/log_fig06.txt 2>&1 &              # stride 1, about 15 h
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

TAG = 'fig06'
FIG_DIR = C.fig_dir('fig06')
OUT_NC = os.path.join(FIG_DIR, 'fig06_data.nc')
OUT_JSON = os.path.join(FIG_DIR, 'fig06_stats.json')
OUT_REPORT = os.path.join(FIG_DIR, 'fig06_report.txt')
OUT_TABLE = os.path.join(FIG_DIR, 'table4_fractions.tex')

NI_TERMS = ('tend', 'W_NI', 'PiH', 'PiV', 'wb', 'eps_NI', 'drag_NI', 'div_adv', 'div_rad')
SUB_TERMS = ('tend_b', 'W_S', 'Pi_b', 'drag_b', 'div_ws')
ALPHA_REF = 'T2'
ALPHA_MIN_RATIO = 3.0          # T2 injection over T1 injection below which a sector is unresolved
BLOCK_KM = 30.0
NBOOT = 500
SEED = 0


def flux_divergence(Fu, Fv, pm, pn):
    """Horizontal divergence on rho points of a flux given on the u and v faces,
    flux form with the ROMS metrics."""
    pn_u = 0.5 * (pn[:, :-1] + pn[:, 1:])
    pm_v = 0.5 * (pm[:-1, :] + pm[1:, :])
    div = np.zeros(pm.shape)
    Uf, Vf = Fu / pn_u, Fv / pm_v
    div[:, 1:-1] += Uf[:, 1:] - Uf[:, :-1]
    div[1:-1, :] += Vf[1:, :] - Vf[:-1, :]
    return pm * pn * div


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--eta-chunk', type=int, default=30)
    ap.add_argument('--workers', type=int, default=-1)
    ap.add_argument('--nboot', type=int, default=NBOOT)
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
    have_akv = arc.has('AKv')
    g = C.load_grid(arc, stride=st)
    lon, lat, h, mask = g['lon'], g['lat'], g['h'], g['mask']
    pm, pn, land, area, f2d = g['pm'], g['pn'], g['land'], g['area'], g['f']
    neta, nxi = lon.shape
    nt, nlev = arc.nt, arc.nlev
    C.report(TAG, f'grid {neta} x {nxi} x {nlev} (stride {st}), {nt} records, AKv {"present" if have_akv else "ABSENT (eps_NI zero)"}')

    sf = nc.Dataset(C.SURFACE_FILE)
    sf.set_auto_mask(False)
    if sf.dimensions['time'].size != nt:
        raise SystemExit(f'{C.SURFACE_FILE} holds {sf.dimensions["time"].size} records, the archive {nt}')

    fb = C.FilterBank(lat, land, arc.dt_h, nt, workers=args.workers)
    fb.attach_time(arc.time)
    fb.describe(TAG)
    valid = fb.valid
    idx = C.window_indices(arc.time, valid)
    wsl = {k: C.as_slice(v) for k, v in idx.items()}
    wdur = {k: float(v.size * arc.dt_h * 3600.0) for k, v in idx.items()}      # window length, s
    days = np.unique(arc.time[valid].astype('datetime64[D]'))
    day_of = arc.time.astype('datetime64[D]')
    dsl = [C.as_slice(np.where((day_of == d) & valid)[0]) for d in days]

    C.report(TAG, 'time-mean free surface and the vertical grid')
    sel_z = dict(eta_rho=C.ETA_SLICE, xi_rho=C.XI_SLICE)
    Hz = np.nan_to_num(C.compute_Hz(arc.ds0, mean_zeta(arc, st, sel_z, nt), st, h=h), nan=0.0)

    # ---- accumulators, window means on the grid (W m-2 after rho0) ----------------
    M = {t: {k: np.zeros((neta, nxi)) for k in C.WKEYS} for t in NI_TERMS + SUB_TERMS}
    Fadv_u = {k: np.zeros((neta, nxi - 1)) for k in C.WKEYS}
    Frad_u = {k: np.zeros((neta, nxi - 1)) for k in C.WKEYS}
    Fadv_v = {k: np.zeros((neta - 1, nxi)) for k in C.WKEYS}
    Frad_v = {k: np.zeros((neta - 1, nxi)) for k in C.WKEYS}
    Gx = {k: np.zeros((neta, nxi)) for k in C.WKEYS}          # int ubar_h . <u' u'> dz, x component on rho
    Gy = {k: np.zeros((neta, nxi)) for k in C.WKEYS}
    Kb_daily = np.zeros((days.size, neta, nxi))
    WNI_daily = np.zeros((days.size, neta, nxi))
    K_win = {k: np.zeros((neta, nxi)) for k in C.WKEYS}        # window-mean int K dz, for Table 4
    Kb_win = {k: np.zeros((neta, nxi)) for k in C.WKEYS}

    def read_block(var, de, dx, r0, r1, lev=None):
        sel = {de: slice(C.ETA0 + r0 * st, C.ETA0 + r1 * st, st), dx: slice(C.XI0, None)}
        if lev is not None:
            sel['s_rho' if 's_rho' in arc.ds0[var].dims else 's_w'] = lev
        a = arc.read(var, sel)[..., ::st]
        return a[..., :nxi - 1] if dx == 'xi_u' else a[..., :nxi]

    def read_surface(name, r0, r1, ncol):
        a = sf.variables[name][:, r0 * st:r1 * st:st, ::st].astype(np.float64)
        return a[..., :ncol]

    nchunk = int(np.ceil(neta / args.eta_chunk))
    t0 = _time.time()
    for ic, j0 in enumerate(range(0, neta, args.eta_chunk)):
        tb = _time.time()
        j1 = min(j0 + args.eta_chunk, neta)
        nrow = j1 - j0
        a0, a1 = max(j0 - HALO, 0), min(j1 + HALO, neta)      # rho rows with the halo
        b0, b1 = a0, a1 - 1                                     # v rows between them
        na, nb = a1 - a0, b1 - b0
        s0 = j0 - a0
        w0, w1 = j0, min(j1, neta - 1)                          # v faces this block owns
        pmL, pnL = pm[a0:a1, :], pn[a0:a1, :]
        hz_all = Hz[:, a0:a1, :]

        # surface forcing of the block (stress on the faces, velocity on the faces)
        su = read_surface('sustr', a0, a1, nxi - 1)
        sv = read_surface('svstr', b0, b1, nxi)
        us = read_surface('u_s', a0, a1, nxi - 1)
        vs = read_surface('v_s', b0, b1, nxi)
        zt = read_surface('zeta', a0, a1, nxi)
        fsu, fsv = fb.apply_both(su, a0), fb.apply_both(sv, b0)
        fus, fvs = fb.apply_both(us, a0), fb.apply_both(vs, b0)
        del su, sv, us, vs
        wni = fb.phase_average(C.u2rho(fsu['bp'] * fus['bp']) + C.v2rho(fsv['bp'] * fvs['bp']), a0)[:, s0:s0 + nrow]
        ws = (C.u2rho(fsu['lp'] * fus['lp']) + C.v2rho(fsv['lp'] * fvs['lp']))[:, s0:s0 + nrow]
        del fsu, fsv, fus, fvs
        for k in C.WKEYS:
            M['W_NI'][k][j0:j1] = wni[wsl[k]].mean(axis=0)
            M['W_S'][k][j0:j1] = ws[wsl[k]].mean(axis=0)
        for d, s in enumerate(dsl):
            WNI_daily[d, j0:j1] = wni[s].mean(axis=0)
        del wni, ws
        zp = fb.apply(zt, a0, 'bp')
        del zt
        p_acc = np.zeros((nt, na, nxi))
        K_int = np.zeros((nt, nrow, nxi))
        Kb_int = np.zeros((nt, nrow, nxi))
        prev = None

        for kk in range(nlev - 1, -1, -1):
            tl = _time.time()
            u_raw = read_block('u', 'eta_u', 'xi_u', a0, a1, kk)
            v_raw = read_block('v', 'eta_v', 'xi_v', b0, b1, kk)
            fu = fb.apply_both(u_raw, a0)
            fv = fb.apply_both(v_raw, b0)
            rho = read_block('rho', 'eta_rho', 'xi_rho', a0, a1, kk)
            rp = fb.apply(rho, a0, 'bp')
            del rho
            hz = hz_all[kk]
            p_k = C.RHO0 * C.G * zp + C.G * (p_acc + 0.5 * rp * hz[None])
            p_acc += rp * hz[None]
            bp = -(C.G / C.RHO0) * rp[:, s0:s0 + nrow]
            del rp
            pa = lambda a, r0=j0: fb.phase_average(a, r0)
            upr, vpr = C.u2rho(fu['bp']), C.v2rho(fv['bp'])        # rho rows a0:a1
            ulr, vlr = C.u2rho(fu['lp']), C.v2rho(fv['lp'])
            uu, vv, uv = (fb.phase_average(upr * upr, a0), fb.phase_average(vpr * vpr, a0),
                          fb.phase_average(upr * vpr, a0))
            K_k = 0.5 * C.RHO0 * (uu + vv)                           # J m-3, rows a0:a1
            # the tensor and the horizontal shear production on the owned rows
            dlt, Sng, _, Ssg = cgrid_tensor_t(np.nan_to_num(smooth_h(fu['lp'], smooth), nan=0.0),
                                              np.nan_to_num(smooth_h(fv['lp'], smooth), nan=0.0), pmL, pnL)
            PiH = -C.RHO0 * (0.5 * (uu - vv) * Sng + uv * Ssg + 0.5 * (uu + vv) * dlt)[:, s0:s0 + nrow]
            del dlt, Sng, Ssg
            # wave-stress work vector on rho, int ubar . <u'u'> dz
            gx = C.RHO0 * (ulr * uu + vlr * uv)
            gy = C.RHO0 * (ulr * uv + vlr * vv)
            # transports on the faces
            E_u = 0.5 * (K_k[:, :, :-1] + K_k[:, :, 1:])
            E_v = 0.5 * (K_k[:, :-1, :] + K_k[:, 1:, :])
            p_u = 0.5 * (p_k[:, :, :-1] + p_k[:, :, 1:])
            p_v = 0.5 * (p_k[:, :-1, :] + p_k[:, 1:, :])
            fa_u, fa_v = fu['lp'] * E_u, fv['lp'] * E_v
            fr_u = fb.phase_average(fu['bp'] * p_u, a0)             # <u' p'> with p' in Pa, W m-2 per metre
            fr_v = fb.phase_average(fv['bp'] * p_v, b0)
            del E_u, E_v, p_u, p_v, p_k
            hz_u = 0.5 * (hz[:, :-1] + hz[:, 1:])
            hz_v = 0.5 * (hz[:-1, :] + hz[1:, :])
            Kb_k = 0.5 * C.RHO0 * (ulr * ulr + vlr * vlr)
            K_int += K_k[:, s0:s0 + nrow] * hz[None, s0:s0 + nrow]
            Kb_int += Kb_k[:, s0:s0 + nrow] * hz[None, s0:s0 + nrow]
            for k in C.WKEYS:
                s = wsl[k]
                M['PiH'][k][j0:j1] += PiH[s].mean(axis=0) * hz[s0:s0 + nrow]
                Gx[k][j0:j1] += gx[s, s0:s0 + nrow].mean(axis=0) * hz[s0:s0 + nrow]
                Gy[k][j0:j1] += gy[s, s0:s0 + nrow].mean(axis=0) * hz[s0:s0 + nrow]
                Fadv_u[k][j0:j1] += (fa_u[s].mean(axis=0) * hz_u)[s0:s0 + nrow]
                Frad_u[k][j0:j1] += (fr_u[s].mean(axis=0) * hz_u)[s0:s0 + nrow]
                if w1 > w0:
                    Fadv_v[k][w0:w1] += (fa_v[s].mean(axis=0) * hz_v)[s0:s0 + (w1 - w0)]
                    Frad_v[k][w0:w1] += (fr_v[s].mean(axis=0) * hz_v)[s0:s0 + (w1 - w0)]
            del fa_u, fa_v, fr_u, fr_v, gx, gy, PiH, uu, vv, uv
            # the interface above this level, index kk + 1 on the w grid
            if prev is not None:
                up_p, vp_p, ul_p, vl_p, bp_p = prev
                wp = fb.apply(read_block('w', 'eta_rho', 'xi_rho', j0, j1, kk + 1), j0, 'bp')
                dz = np.maximum(0.5 * (Hz[kk, j0:j1, :] + Hz[kk + 1, j0:j1, :]), 1e-3)[None]
                upo, vpo = upr[:, s0:s0 + nrow], vpr[:, s0:s0 + nrow]
                ulo, vlo = ulr[:, s0:s0 + nrow], vlr[:, s0:s0 + nrow]
                uw, vw = pa(0.5 * (upo + up_p) * wp), pa(0.5 * (vpo + vp_p) * wp)
                PiV = -C.RHO0 * (uw * (ul_p - ulo) / dz + vw * (vl_p - vlo) / dz)
                wb = C.RHO0 * pa(wp * 0.5 * (bp + bp_p))
                sh2 = pa(((up_p - upo) / dz) ** 2 + ((vp_p - vpo) / dz) ** 2)
                if have_akv:
                    akv = read_block('AKv', 'eta_rho', 'xi_rho', j0, j1, kk + 1)
                    eps = C.RHO0 * pa(akv * sh2)
                    del akv
                else:
                    eps = np.zeros_like(sh2)
                for k in C.WKEYS:
                    s = wsl[k]
                    M['PiV'][k][j0:j1] += PiV[s].mean(axis=0) * dz[0]
                    M['wb'][k][j0:j1] += wb[s].mean(axis=0) * dz[0]
                    M['eps_NI'][k][j0:j1] += eps[s].mean(axis=0) * dz[0]
                del wp, uw, vw, PiV, wb, sh2, eps
            if kk == 0:
                # the bed, quadratic drag with the full bottom velocity
                ub = C.u2rho(u_raw)[:, s0:s0 + nrow]
                vb = C.v2rho(v_raw)[:, s0:s0 + nrow]
                spd = np.hypot(ub, vb)
                dni = C.RHO0 * C.CD_BOT * pa(spd * (ub * upr[:, s0:s0 + nrow] + vb * vpr[:, s0:s0 + nrow]))
                dsub = C.RHO0 * C.CD_BOT * spd * (ub * ulr[:, s0:s0 + nrow] + vb * vlr[:, s0:s0 + nrow])
                for k in C.WKEYS:
                    M['drag_NI'][k][j0:j1] = dni[wsl[k]].mean(axis=0)
                    M['drag_b'][k][j0:j1] = dsub[wsl[k]].mean(axis=0)
                del ub, vb, spd, dni, dsub
            prev = (upr[:, s0:s0 + nrow], vpr[:, s0:s0 + nrow], ulr[:, s0:s0 + nrow], vlr[:, s0:s0 + nrow], bp)
            del u_raw, v_raw, fu, fv, upr, vpr, ulr, vlr, K_k, Kb_k
            if ic == 0 and (kk % 5 == 0 or kk == nlev - 1):
                per = (_time.time() - tb) / (nlev - kk)
                C.report(TAG, f'    block 1, level {kk + 1:2d}/{nlev} (from the top), {_time.time() - tl:5.1f} s, '
                              f'whole pass about {per * nlev * nchunk / 3600:.1f} h, finish near '
                              f'{_time.strftime("%d %b %H:%M", _time.localtime(tb + per * nlev * nchunk))}')
        del prev, p_acc, zp
        # tendencies from the first and last valid record of each window, window means of the energies
        for k in C.WKEYS:
            i0, i1 = int(idx[k][0]), int(idx[k][-1])
            span = (i1 - i0) * arc.dt_h * 3600.0
            M['tend'][k][j0:j1] = (K_int[i1] - K_int[i0]) / span
            M['tend_b'][k][j0:j1] = (Kb_int[i1] - Kb_int[i0]) / span
            K_win[k][j0:j1] = K_int[wsl[k]].mean(axis=0)
            Kb_win[k][j0:j1] = Kb_int[wsl[k]].mean(axis=0)
        for d, s in enumerate(dsl):
            Kb_daily[d, j0:j1] = Kb_int[s].mean(axis=0)
        del K_int, Kb_int
        el = _time.time() - t0
        C.report(TAG, f'  block {ic + 1:2d}/{nchunk}, rows {j0:4d}-{j1:4d}, {(_time.time() - tb) / 60:.1f} min, '
                      f'elapsed {el / 60:.0f} min, about {el / (ic + 1) * (nchunk - ic - 1) / 60:.0f} min left, '
                      f'finish near {_time.strftime("%d %b %H:%M", _time.localtime(_time.time() + el / (ic + 1) * (nchunk - ic - 1)))}')
    sf.close()

    # ---- divergences and the residuals ---------------------------------------------------------
    mask_u = (mask[:, :-1] * mask[:, 1:]) > 0.5
    mask_v = (mask[:-1, :] * mask[1:, :]) > 0.5
    for k in C.WKEYS:
        for a, m in ((Fadv_u[k], mask_u), (Frad_u[k], mask_u), (Fadv_v[k], mask_v), (Frad_v[k], mask_v)):
            a[~m] = 0.0
        M['div_adv'][k] = flux_divergence(Fadv_u[k], Fadv_v[k], pm, pn)
        M['div_rad'][k] = flux_divergence(Frad_u[k], Frad_v[k], pm, pn)
        gxu = np.where(mask_u, 0.5 * (Gx[k][:, :-1] + Gx[k][:, 1:]), 0.0)
        gyv = np.where(mask_v, 0.5 * (Gy[k][:-1, :] + Gy[k][1:, :]), 0.0)
        M['div_ws'][k] = flux_divergence(gxu, gyv, pm, pn)
        M['Pi_b'][k] = -(M['PiH'][k] + M['PiV'][k])
    R = {k: (M['tend'][k] + M['div_adv'][k] + M['div_rad'][k] - M['W_NI'][k] - M['PiH'][k] - M['PiV'][k]
             - M['wb'][k] + M['eps_NI'][k] + M['drag_NI'][k]) for k in C.WKEYS}
    Rb = {k: (M['tend_b'][k] + M['div_ws'][k] - M['W_S'][k] - M['Pi_b'][k] + M['drag_b'][k]) for k in C.WKEYS}
    for k in C.WKEYS:
        for t in NI_TERMS + SUB_TERMS:
            M[t][k][land] = np.nan
        R[k][land], Rb[k][land] = np.nan, np.nan
        K_win[k][land], Kb_win[k][land] = np.nan, np.nan
    Kb_daily[:, land] = np.nan
    WNI_daily[:, land] = np.nan

    # ---- sector budgets, J m-2 per window -----------------------------------------------------------
    masks = C.sector_masks(g, stride=st)
    shelf = masks['shelf']
    B = {}
    for sec in C.SECTOR_NAMES:
        B[sec] = {}
        for k in C.WKEYS:
            row = {t: C.area_mean(M[t][k], shelf[sec], area) * wdur[k] for t in NI_TERMS + SUB_TERMS}
            row['R'] = C.area_mean(R[k], shelf[sec], area) * wdur[k]
            row['R_b'] = C.area_mean(Rb[k], shelf[sec], area) * wdur[k]
            row['Pi'] = row['PiH'] + row['PiV']
            row['K_mean'] = C.area_mean(K_win[k], shelf[sec], area)
            row['Kb_mean'] = C.area_mean(Kb_win[k], shelf[sec], area)
            B[sec][k] = row
    Kb_series = np.array([[C.area_mean(Kb_daily[d], shelf[sec], area) for d in range(days.size)] for sec in C.SECTOR_NAMES])
    WNI_series = np.array([[C.area_mean(WNI_daily[d], shelf[sec], area) for d in range(days.size)] for sec in C.SECTOR_NAMES])

    # ---- alpha_Pi with a moving-block bootstrap --------------------------------------------------------
    rng = np.random.default_rng(SEED)
    blk = max(int(round(BLOCK_KM / (1.0 * st))), 1)
    bj, bi = np.arange(neta) // blk, np.arange(nxi) // blk
    blockid = (bj[:, None] * (bi.max() + 1) + bi[None, :]).astype(int)
    nblk = int(blockid.max()) + 1
    alpha = {}
    for sec in C.SECTOR_NAMES:
        m = shelf[sec] & np.isfinite(M['W_NI'][ALPHA_REF]) & np.isfinite(M['W_NI']['T1'])
        for k in C.WKEYS:
            m &= np.isfinite(M['PiH'][k] + M['PiV'][k])      # the tensor is undefined on the outer row and column
        wref = np.where(m, M['W_NI'][ALPHA_REF] * wdur[ALPHA_REF] * area, 0.0)
        w1 = np.where(m, M['W_NI']['T1'] * wdur['T1'] * area, 0.0)
        ratio_ref = float(wref.sum() / w1.sum()) if w1.sum() else np.inf
        resolved = bool(ratio_ref >= ALPHA_MIN_RATIO)
        alpha[sec] = {'resolved': resolved, 'injection_ratio_T2_over_T1': ratio_ref}
        Sw = np.bincount(blockid[m], weights=wref[m], minlength=nblk)
        for k in C.WKEYS:
            num = np.where(m, (M['PiH'][k] + M['PiV'][k]) * wdur[k] * area, 0.0)
            Sn = np.bincount(blockid[m], weights=num[m], minlength=nblk)
            a_hat = float(Sn.sum() / Sw.sum()) if Sw.sum() else np.nan
            present = np.where(Sw != 0)[0]
            boots = np.empty(args.nboot)
            for b in range(args.nboot):
                pick = rng.choice(present, size=present.size, replace=True)
                c = np.bincount(pick, minlength=nblk).astype(float)
                boots[b] = (c @ Sn) / (c @ Sw)
            alpha[sec][k] = dict(alpha=a_hat, lo=float(np.nanpercentile(boots, 2.5)), hi=float(np.nanpercentile(boots, 97.5)))

    # ---- report, Table 4 ------------------------------------------------------------------------------------
    L = ['FIGURE 6 REPORT, the two kinetic energy budgets', f'run {_time.strftime("%Y-%m-%d %H:%M")}',
         f'stride {st}, shelf and slope h <= {C.STAT_MAX_DEPTH_M:.0f} m, area weighted, every term integrated over '
         f'its window, J m-2, positive feeds the kinetic energy named', '']
    for sec in C.SECTOR_NAMES:
        L.append(f'near-inertial kinetic energy budget, {sec} (tend = W_NI + PiH + PiV + wb - eps_NI - drag_NI - div_adv - div_rad + R)')
        L.append('  ' + f'{"win":4s}' + ''.join(f'{t:>9s}' for t in NI_TERMS + ('R',)))
        for k in C.WKEYS:
            r = B[sec][k]
            L.append('  ' + f'{k:4s}' + ''.join(f'{r[t]:9.1f}' for t in NI_TERMS + ('R',)))
        L.append(f'subinertial kinetic energy budget, {sec} (tend_b = W_S - Pi - drag_b - div_ws + R_b)')
        L.append('  ' + f'{"win":4s}' + ''.join(f'{t:>9s}' for t in SUB_TERMS + ('R_b',)))
        for k in C.WKEYS:
            r = B[sec][k]
            L.append('  ' + f'{k:4s}' + ''.join(f'{r[t]:9.1f}' for t in SUB_TERMS + ('R_b',)))
        L.append('')
    L.append('alpha_Pi = window-integrated Pi over the T2 injection, per sector and window [95 % block bootstrap]')
    for sec in C.SECTOR_NAMES:
        a = alpha[sec]
        L.append(f'  {sec:8s} T2/T1 injection {a["injection_ratio_T2_over_T1"]:6.1f} ({"resolved" if a["resolved"] else "UNRESOLVED"})  '
                 + '  '.join(f'{k} {a[k]["alpha"]:+.4f} [{a[k]["lo"]:+.4f}, {a[k]["hi"]:+.4f}]' for k in C.WKEYS))
    L.append('')
    L.append('Table 4, South sector')
    tab4 = {}
    for k in C.WKEYS:
        r = B['South'][k]
        tab4[k] = dict(Pi_kJ=r['Pi'] * 1e-3, alpha=alpha['South'][k]['alpha'], alpha_lo=alpha['South'][k]['lo'],
                       alpha_hi=alpha['South'][k]['hi'],
                       frac_K=r['Pi'] / r['K_mean'] if r['K_mean'] else np.nan,
                       frac_Kb=r['Pi'] / r['Kb_mean'] if r['Kb_mean'] else np.nan,
                       frac_drag=r['Pi'] / r['drag_b'] if r['drag_b'] else np.nan,
                       W_NI_kJ=r['W_NI'] * 1e-3, eps_NI_kJ=r['eps_NI'] * 1e-3, R_kJ=r['R'] * 1e-3,
                       K_mean=r['K_mean'], Kb_mean=r['Kb_mean'])
        L.append(f'  {k}: Pi {tab4[k]["Pi_kJ"]:+.3f} kJ m-2, alpha {tab4[k]["alpha"]:+.4f} [{tab4[k]["alpha_lo"]:+.4f}, {tab4[k]["alpha_hi"]:+.4f}], '
                 f'of K {100 * tab4[k]["frac_K"]:+.1f} %, of Kbar {100 * tab4[k]["frac_Kb"]:+.2f} %, of the drag {100 * tab4[k]["frac_drag"]:+.1f} %, '
                 f'W_NI {tab4[k]["W_NI_kJ"]:.3f}, eps_NI {tab4[k]["eps_NI_kJ"]:.3f}, R {tab4[k]["R_kJ"]:+.3f} kJ m-2')
    L.append('')
    L.append('South, K and Kbar window means (J m-2): ' + ', '.join(f'{k} {B["South"][k]["K_mean"]:.0f} / {B["South"][k]["Kb_mean"]:.0f}' for k in C.WKEYS))
    L.append(f'South, K-weighted e-folding of the T2 drain, K_mean / |Pi rate|: '
             f'{B["South"]["T2"]["K_mean"] / abs(B["South"]["T2"]["Pi"] / wdur["T2"]) / 86400:.1f} d')
    L += ['', f'filters {json.dumps(fb.config())}', f'wall time {(_time.time() - t_start) / 3600:.2f} h']
    C.write_report(OUT_REPORT, L)
    print('\n'.join(L))

    tex = [r'\begin{table}[H]', r'\centering',
           r'\caption{The exchange of the South sector against the injection and the energies, per window. '
           r'$\alpha_\Pi$ (Equation~\ref{eq:alpha}) with its 95~\% moving-block bootstrap interval, the window-integrated '
           r'exchange as a fraction of the window-mean near-inertial and subinertial kinetic energies and of the '
           r'window-integrated bottom drag on the subinertial flow, and the window-integrated wind work, shear '
           r'dissipation and residual of the near-inertial budget.}', r'\label{tab:fractions}',
           r'\begin{tabular}{lrrr}', r'\toprule', r' & $T2$ & $T3$ & $T4$ \\', r'\midrule']
    ks = ('T2', 'T3', 'T4')
    tex.append(r'$\rho_0\iint\Pi\,\mathrm{d}z\,\mathrm{d}t$ (kJ~m$^{-2}$) & ' + ' & '.join(f'{tab4[k]["Pi_kJ"]:+.3f}' for k in ks) + r' \\')
    tex.append(r'$\alpha_\Pi$ [95~\% interval] & ' + ' & '.join(f'{tab4[k]["alpha"]:+.3f} [{tab4[k]["alpha_lo"]:+.3f}, {tab4[k]["alpha_hi"]:+.3f}]' for k in ks) + r' \\')
    tex.append(r'fraction of $\rho_0\int\K\,\mathrm{d}z$ (\%) & ' + ' & '.join(f'{100 * tab4[k]["frac_K"]:+.0f}' for k in ks) + r' \\')
    tex.append(r'fraction of $\rho_0\int\Kb\,\mathrm{d}z$ (\%) & ' + ' & '.join(f'{100 * tab4[k]["frac_Kb"]:+.1f}' for k in ks) + r' \\')
    tex.append(r'fraction of the bottom drag (\%) & ' + ' & '.join(f'{100 * tab4[k]["frac_drag"]:+.0f}' for k in ks) + r' \\')
    tex.append(r'$\rho_0\int\WNI\,\mathrm{d}t$ (kJ~m$^{-2}$) & ' + ' & '.join(f'{tab4[k]["W_NI_kJ"]:.3f}' for k in ks) + r' \\')
    tex.append(r'$\rho_0\iint\epsNI\,\mathrm{d}z\,\mathrm{d}t$ (kJ~m$^{-2}$) & ' + ' & '.join(f'{tab4[k]["eps_NI_kJ"]:.3f}' for k in ks) + r' \\')
    tex.append(r'residual (kJ~m$^{-2}$) & ' + ' & '.join(f'{tab4[k]["R_kJ"]:+.3f}' for k in ks) + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    with open(OUT_TABLE, 'w') as fh:
        fh.write('\n'.join(tex) + '\n')

    # ---- archive -------------------------------------------------------------------------------------------------
    data = {'h': (('eta', 'xi'), h), 'mask': (('eta', 'xi'), mask), 'area': (('eta', 'xi'), area),
            'sector': (('eta', 'xi'), masks['sector_id']),
            'Kb_series': (('sector_name', 'day'), Kb_series), 'WNI_series': (('sector_name', 'day'), WNI_series)}
    for k in C.WKEYS:
        for t in NI_TERMS + SUB_TERMS:
            data[f'{t}_{k}'] = (('eta', 'xi'), M[t][k].astype(np.float32))
        data[f'R_{k}'] = (('eta', 'xi'), R[k].astype(np.float32))
        data[f'Rb_{k}'] = (('eta', 'xi'), Rb[k].astype(np.float32))
        data[f'Fx_{k}'] = (('eta', 'xi'), (C.u2rho(Fadv_u[k] + Frad_u[k])).astype(np.float32))
        data[f'Fy_{k}'] = (('eta', 'xi'), (C.v2rho(Fadv_v[k] + Frad_v[k])).astype(np.float32))
        data[f'Fadv_x_{k}'] = (('eta', 'xi'), C.u2rho(Fadv_u[k]).astype(np.float32))
        data[f'Fadv_y_{k}'] = (('eta', 'xi'), C.v2rho(Fadv_v[k]).astype(np.float32))
        data[f'K_{k}'] = (('eta', 'xi'), K_win[k].astype(np.float32))
        data[f'Kb_{k}'] = (('eta', 'xi'), Kb_win[k].astype(np.float32))
    bud = {sec: {k: {t: float(v) for t, v in r.items()} for k, r in d.items()} for sec, d in B.items()}
    stats = dict(budgets=bud, alpha=alpha, table4=tab4, window_length_s=wdur,
                 config=dict(stride=st, cd_bot=C.CD_BOT, have_akv=bool(have_akv), block_km=BLOCK_KM, nboot=args.nboot,
                             alpha_ref=ALPHA_REF, alpha_min_ratio=ALPHA_MIN_RATIO, filters=fb.config()))
    out = xr.Dataset(data, coords=dict(lon=(('eta', 'xi'), lon), lat=(('eta', 'xi'), lat),
                                       day=days.astype('datetime64[ns]'), sector_name=list(C.SECTOR_NAMES)),
                     attrs=dict(source=' + '.join(arc.paths), stride=st, rho0=C.RHO0, cd_bot=C.CD_BOT,
                                have_akv=int(have_akv), **{k: list(v) for k, v in C.WINDOWS.items()},
                                **{f'dur_{k}': v for k, v in wdur.items()}, landfall=str(C.LANDFALL_DEFAULT)[:16],
                                stat_max_depth_m=C.STAT_MAX_DEPTH_M, south_top=masks['south_top'],
                                central_top=masks['central_top'], stats=json.dumps(stats, default=float),
                                note=('Window-mean depth-integrated budget terms times rho0, W m-2 (multiply by the '
                                      'window length dur_Tk for J m-2), the residuals R and Rb, the transport vectors '
                                      'Fx Fy (W m-1, advective plus radiative, on rho points), the window-mean '
                                      'energies K and Kb (J m-2), the daily sector series of Kb (J m-2) and W_NI (W m-2).'),
                                created=_time.strftime('%Y-%m-%d %H:%M')))
    for v in out.data_vars:
        if out[v].dtype.kind == 'f':
            out[v].encoding.update(zlib=True, complevel=4)
    out.to_netcdf(OUT_NC)
    with open(OUT_JSON, 'w') as fh:
        json.dump(stats, fh, indent=2, default=float)
    print(f'wrote {OUT_NC}, {OUT_JSON} and {OUT_TABLE}')
    arc.close()


if __name__ == '__main__':
    main()