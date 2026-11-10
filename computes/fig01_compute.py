"""
fig01_compute.py

Every quantity of Figure 1 (the storm, the wind forcing and the shelf), from
surface_v3.nc written by extract_surface_v3.py. Nothing is plotted here.
fig01_plot.py draws from fig01_data.nc alone.

Panels served (plan.md, Section 3, Figure 1)

    a  bathymetry, isobaths, landmarks, stations, sector dividers, the track
       with its 6-hourly fixes, grid outlines for the inset
    b  near-inertial wind work integrated over T2, rho0 int W_NI dt, kJ m-2
    c  wind stress magnitude and direction at A to D, N m-2
    d  wind-work cospectrum against omega / f_local, averaged over a 30 km box
       around each station, Welch segments of 8 d, 50 % overlap
    e  surface u and v at D with their subinertial parts
    f  complex-demodulated near-inertial amplitude at D and A, diurnal at D

Numbers for the text go to fig01_report.txt and fig01_stats.json.

Conventions (identical to v2)
    * velocity at a station from the two u faces and the two v faces of the
      rho point, then rotated to east and north with the grid angle
    * stress in N m-2 (scaled in the surface file)
    * near-inertial band [0.8 f, 1.2 f] with the local f, Lanczos filters of
      common_v3.py, phase average over one local inertial period
    * demodulation at +f (anticlockwise rotation, the inertial sense of the
      Southern Hemisphere in the rotary convention of v2) and at +1 cpd

    python fig01_compute.py
"""

import json
import os
import time as _time

import numpy as np
import netCDF4 as nc
import xarray as xr

import common_v3 as C

TAG = 'fig01'
FIG_DIR = C.fig_dir('fig01')
OUT_NC = os.path.join(FIG_DIR, 'fig01_data.nc')
OUT_JSON = os.path.join(FIG_DIR, 'fig01_stats.json')
OUT_REPORT = os.path.join(FIG_DIR, 'fig01_report.txt')

DEMOD_STATIONS = ('D', 'A')
FIELD_STRIDE = 2            # map fields stored at half resolution


def main():
    t_start = _time.time()
    sf = nc.Dataset(C.SURFACE_FILE)
    sf.set_auto_mask(False)
    time = (np.datetime64('2004-03-13T00:00:00') +
            sf.variables['time'][:].astype('timedelta64[s]'))
    nt = time.size
    dt_h = C.sampling_interval_hours(time)
    fs = 1.0 / dt_h
    g = {k: sf.variables[k][:] for k in ('lon', 'lat', 'h', 'mask', 'angle', 'pm', 'pn', 'f')}
    g['land'] = g['mask'] < 0.5
    g['area'] = np.where(g['land'], np.nan, 1.0 / (g['pm'] * g['pn']))
    neta, nxi = g['lon'].shape
    C.report(TAG, f'{nt} records, dt {dt_h:.2f} h, grid {neta} x {nxi}')

    # ---- filters ----------------------------------------------------------
    fb = C.FilterBank(g['lat'], g['land'], dt_h, nt)
    fb.attach_time(time)
    fb.describe(TAG)
    valid = fb.valid
    idx = C.window_indices(time, valid)
    days = np.unique(time[valid].astype('datetime64[D]'))
    day_of = time.astype('datetime64[D]')
    dur = {k: idx[k].size * dt_h * 3600.0 for k in C.WKEYS}

    # ---- track and landfall ------------------------------------------------
    track = C.read_track(C.TRACK_CSV)
    landfall, lf_lon, lf_lat = C.landfall_from_track(track, g['lon'], g['lat'], g['mask'])
    if landfall is None:
        landfall = C.LANDFALL_DEFAULT
        C.report(TAG, f'landfall not found on the mask, using {landfall}')
    else:
        C.report(TAG, f'landfall {str(landfall)[:16]} at {lf_lon:.2f}, {lf_lat:.2f}')

    # ---- stations ------------------------------------------------------------
    st = {}
    for name in C.STATION_ORDER:
        xi, eta = C.STATIONS[name]['xi'], C.STATIONS[name]['eta']
        a = float(g['angle'][eta, xi])
        lat_st = float(g['lat'][eta, xi])
        f_cph = abs(C.coriolis(lat_st)) / (2 * np.pi) * 3600.0
        u_g = 0.5 * (sf.variables['u_s'][:, eta, xi - 1] + sf.variables['u_s'][:, eta, xi])
        v_g = 0.5 * (sf.variables['v_s'][:, eta - 1, xi] + sf.variables['v_s'][:, eta, xi])
        tx_g = 0.5 * (sf.variables['sustr'][:, eta, xi - 1] + sf.variables['sustr'][:, eta, xi])
        ty_g = 0.5 * (sf.variables['svstr'][:, eta - 1, xi] + sf.variables['svstr'][:, eta, xi])
        u_e, v_n = C.grid_to_east_north(u_g.astype(float), v_g.astype(float), a)
        tau_e, tau_n = C.grid_to_east_north(tx_g.astype(float), ty_g.astype(float), a)
        tf, Tf = C.station_filters(lat_st, dt_h, nt)
        parts = {}
        for key, ser in (('u', u_e), ('v', v_n), ('tx', tau_e), ('ty', tau_n)):
            F = tf.transform(ser[:, None, None])
            parts[key + '_lp'] = tf.apply(F, 'lp')[:, 0, 0]
            parts[key + '_bp'] = tf.apply(F, 'bp')[:, 0, 0]
        wni = parts['tx_bp'] * parts['u_bp'] + parts['ty_bp'] * parts['v_bp']
        n_pa = int(round(C.PHASE_AVG_PERIODS * Tf / dt_h)) | 1
        wni_pa = C.boxcar_time(wni[:, None, None], n_pa)[:, 0, 0]
        tau = np.hypot(tau_e, tau_n)
        d = dict(lon=float(g['lon'][eta, xi]), lat=lat_st, depth=float(g['h'][eta, xi]),
                 f_cph=f_cph, Tf_h=Tf, angle=a, u_e=u_e, v_n=v_n, tau_e=tau_e, tau_n=tau_n,
                 tau=tau, w_ni=wni_pa, **parts)
        for k in C.WKEYS:
            ii = idx[k]
            d[f'W_NI_{k}'] = float(np.sum(wni[ii]) * dt_h * 3600.0)       # J m-2
            d[f'taumax_{k}'] = float(np.max(tau[C.window_mask(time, C.WINDOWS[k])]))
        d['tau_median_T1'] = float(np.median(tau[C.window_mask(time, C.WINDOWS['T1'])]))
        d['tau_max'] = float(tau.max())
        d['tau_max_time'] = str(time[int(np.argmax(tau))])[:16]
        st[name] = d
        C.report(TAG, f'station {name}: lat {lat_st:.2f}, h {d["depth"]:.0f} m, Tf {Tf:.1f} h, '
                      f'tau max {d["tau_max"]:.2f} N m-2 at {d["tau_max_time"]}, '
                      f'T1 median {d["tau_median_T1"]:.3f} N m-2')

    # ---- demodulation ------------------------------------------------------------
    diurnal_cph = 1.0 / C.DIURNAL_PERIOD_H
    for name in DEMOD_STATIONS:
        d = st[name]
        env_ni, _ = C.complex_demodulate(d['u_e'], d['v_n'], d['f_cph'], fs)
        env_di, _ = C.complex_demodulate(d['u_e'], d['v_n'], diurnal_cph, fs)
        d['amp_ni'], d['amp_diurnal'] = np.abs(env_ni), np.abs(env_di)
        k = int(np.argmax(np.where(valid, d['amp_ni'], -1)))
        d['amp_ni_peak'] = float(d['amp_ni'][k])
        d['amp_ni_peak_time'] = str(time[k])[:16]
        d['amp_ni_T1_median'] = float(np.median(d['amp_ni'][idx['T1']]))
        d['amp_diurnal_max_T2'] = float(d['amp_diurnal'][idx['T2']].max())
        d['amp_diurnal_T1_median'] = float(np.median(d['amp_diurnal'][idx['T1']]))
        C.report(TAG, f'demodulation at {name}: near-inertial peak {d["amp_ni_peak"]:.3f} m s-1 '
                      f'at {d["amp_ni_peak_time"]}, T1 median {d["amp_ni_T1_median"]:.3f}, '
                      f'diurnal T2 max {d["amp_diurnal_max_T2"]:.3f}')

    # ---- box cospectra ------------------------------------------------------------
    nperseg = int(round(C.COSPEC_NPERSEG_D * 24.0 / dt_h))
    if C.COSPEC_METHOD == 'welch':
        edof, nseg = C.welch_edof(nt, nperseg, C.COSPEC_OVERLAP)
        C.report(TAG, f'cospectra, Welch, {nperseg} samples per segment ({C.COSPEC_NPERSEG_D:g} d), '
                      f'{nseg} segments, edof {edof:.1f}, df {fs / nperseg:.5f} cph')
    else:
        edof, nseg = 2.0 * C.COSPEC_SMOOTH_BINS, 1
        C.report(TAG, f'cospectra, periodogram of the whole record ({nt} samples, boxcar), '
                      f'df {fs / nt:.5f} cph, smoothed over {C.COSPEC_SMOOTH_BINS} bins for display')
    for name in C.STATION_ORDER:
        d = st[name]
        es, xs = C.box_slices(name)
        e0, e1, x0, x1 = es.start, es.stop, xs.start, xs.stop
        u_u = sf.variables['u_s'][:, e0:e1, x0 - 1:x1].astype(float)       # faces x0-1 .. x1-1
        v_v = sf.variables['v_s'][:, e0 - 1:e1, x0:x1].astype(float)
        tx_u = sf.variables['sustr'][:, e0:e1, x0 - 1:x1].astype(float)
        ty_v = sf.variables['svstr'][:, e0 - 1:e1, x0:x1].astype(float)
        u_r = 0.5 * (u_u[:, :, :-1] + u_u[:, :, 1:])
        v_r = 0.5 * (v_v[:, :-1, :] + v_v[:, 1:, :])
        tx_r = 0.5 * (tx_u[:, :, :-1] + tx_u[:, :, 1:])
        ty_r = 0.5 * (ty_v[:, :-1, :] + ty_v[:, 1:, :])
        ang = g['angle'][e0:e1, x0:x1][None]
        wet = ~g['land'][e0:e1, x0:x1]
        u_e, v_n = C.grid_to_east_north(u_r, v_r, ang)
        tau_e, tau_n = C.grid_to_east_north(tx_r, ty_r, ang)
        sel = wet.ravel()
        flat = lambda a: a.reshape(nt, -1)[:, sel]
        if C.COSPEC_METHOD == 'welch':
            freq, Co = C.welch_cospectrum(flat(tau_e), flat(tau_n), flat(u_e), flat(v_n),
                                          fs, nperseg)
            Co_box = Co.mean(axis=1)                    # W m-2 cph-1
            Co_show = Co_box
        else:
            freq, Co_box, Co_show = C.periodogram_cospectrum(
                flat(tau_e), flat(tau_n), flat(u_e), flat(v_n), fs, C.COSPEC_SMOOTH_BINS)
        d['cospec_freq'], d['cospec'], d['cospec_smooth'] = freq, Co_box, Co_show
        d['cospec_omega_over_f'] = freq / d['f_cph']
        band = (freq >= C.NI_BAND[0] * d['f_cph']) & (freq <= C.NI_BAND[1] * d['f_cph'])
        d['W_NI_cospec_box'] = float(np.trapezoid(Co_box[band], freq[band]))
        # direct box mean of the band-passed products over the valid record
        tf, Tf = C.station_filters(d['lat'], dt_h, nt)
        w_direct = np.zeros(nt)
        for a_, b_ in ((tau_e, u_e), (tau_n, v_n)):
            w_direct += (tf.filt(flat(a_), 'bp') * tf.filt(flat(b_), 'bp')).mean(axis=1)
        d['W_NI_direct_box'] = float(w_direct[valid].mean())
        d['W_NI_direct_box_full'] = float(np.nanmean(w_direct))      # same span as the periodogram
        dm = lambda a: a - a.mean(axis=0, keepdims=True)
        d['W_total_box_full'] = float((dm(flat(tau_e)) * dm(flat(u_e))
                                       + dm(flat(tau_n)) * dm(flat(v_n))).mean())   # covariance, as the cospectrum
        d['cospec_total'] = float(np.trapezoid(Co_box, freq))
        d['W_NI_direct_box_T2'] = float(np.sum(w_direct[idx['T2']]) * dt_h * 3600.0)
        d['cospec_ratio'] = d['W_NI_cospec_box'] / d['W_NI_direct_box_full']
        d['n_box_cells'] = int(sel.sum())
        d['diurnal_over_f'] = diurnal_cph / d['f_cph']
        C.report(TAG, f'  {name}: box {d["n_box_cells"]} cells, cospectrum band integral '
                      f'{d["W_NI_cospec_box"]:.3e} W m-2, direct {d["W_NI_direct_box"]:.3e}, '
                      f'ratio {d["cospec_ratio"]:.2f}, omega_diurnal/f {d["diurnal_over_f"]:.2f}')

    # ---- wind work maps -----------------------------------------------------------
    C.report(TAG, 'near-inertial wind work over the domain, row blocks')
    W = {k: np.zeros((neta, nxi)) for k in C.WKEYS}             # window integrals, J m-2
    W_daily = np.zeros((days.size, neta, nxi))
    dsl = [C.as_slice(np.where((day_of == d) & valid)[0]) for d in days]
    t0 = _time.time()
    nchunk = int(np.ceil(neta / C.ETA_CHUNK))
    for ic, j0 in enumerate(range(0, neta, C.ETA_CHUNK)):
        j1 = min(j0 + C.ETA_CHUNK, neta)
        nrow = j1 - j0
        vr0, vr1 = max(j0 - 1, 0), min(j1, neta - 1)
        off = j0 - vr0
        su = sf.variables['sustr'][:, j0:j1, :].astype(float)
        us = sf.variables['u_s'][:, j0:j1, :].astype(float)
        sv = sf.variables['svstr'][:, vr0:vr1, :].astype(float)
        vs = sf.variables['v_s'][:, vr0:vr1, :].astype(float)
        wni_u = fb.apply(su, j0, 'bp') * fb.apply(us, j0, 'bp')
        wni_v = fb.apply(sv, vr0, 'bp') * fb.apply(vs, vr0, 'bp')
        wni = fb.phase_average(C.u2rho(wni_u) + C.v2rho(wni_v)[:, off:off + nrow, :], j0)
        del su, us, sv, vs, wni_u, wni_v
        for k in C.WKEYS:
            W[k][j0:j1] = wni[idx[k]].sum(axis=0) * dt_h * 3600.0
        for di, s_ in enumerate(dsl):
            W_daily[di, j0:j1] = wni[s_].mean(axis=0)
        del wni
        if ic % 10 == 0 or ic == nchunk - 1:
            el = _time.time() - t0
            C.report(TAG, f'  rows {j0:5d}-{j1:5d}  {el / 60:.1f} min, about '
                          f'{el / (ic + 1) * (nchunk - ic - 1) / 60:.1f} min left')
    for k in C.WKEYS:
        W[k][g['land']] = np.nan
    W_daily[:, g['land']] = np.nan

    masks = C.sector_masks(g)
    area = g['area']
    inj = {}
    for k in C.WKEYS:
        inj[k] = {}
        for sec in C.SECTOR_NAMES:
            for which in ('shelf', 'full'):
                m = masks[which][sec]
                inj[k][f'{sec}_{which}_mean_J_m2'] = C.area_mean(W[k], m, area)
                inj[k][f'{sec}_{which}_total_J'] = C.area_sum(W[k], m, area)
        inj[k]['South_fraction_of_domain_shelf'] = (
            inj[k]['South_shelf_total_J'] / inj[k]['Domain_shelf_total_J'])
        inj[k]['South_fraction_of_domain_full'] = (
            inj[k]['South_full_total_J'] / inj[k]['Domain_full_total_J'])
    series = np.full((len(C.SECTOR_NAMES), days.size), np.nan)
    for si, sec in enumerate(C.SECTOR_NAMES):
        for di in range(days.size):
            series[si, di] = C.area_mean(W_daily[di], masks['shelf'][sec], area)

    for name in C.STATION_ORDER:
        xi, eta = C.STATIONS[name]['xi'], C.STATIONS[name]['eta']
        for k in C.WKEYS:
            st[name][f'W_NI_map_{k}'] = float(W[k][eta, xi])

    # ---- grids for the inset ------------------------------------------------------
    perim = {}
    for key, path in (('child', C.GRID_CHILD), ('parent', C.GRID_PARENT)):
        if os.path.exists(path):
            gg = xr.open_dataset(path)
            perim[key] = C.grid_perimeter(gg['lon_rho'].values, gg['lat_rho'].values, 4)
            gg.close()
    rows = (C.SOUTH_TOP, C.CENTRAL_TOP)

    # ---- report --------------------------------------------------------------------
    L = ['FIGURE 1 REPORT, numbers for the text', f'run {_time.strftime("%Y-%m-%d %H:%M")}',
         f'record {str(time[0])[:16]} to {str(time[-1])[:16]}, valid '
         f'{str(fb.valid_span()[0])[:16]} to {str(fb.valid_span()[1])[:16]}',
         f'landfall {str(landfall)[:16]} UTC (track crosses the land mask)', '']
    L.append('windows, valid samples: ' + ', '.join(
        f'{k} {idx[k].size} ({dur[k] / 86400:.2f} d)' for k in C.WKEYS))
    L += ['', 'stations (lon, lat, depth, inertial period)']
    for n in C.STATION_ORDER:
        d = st[n]
        L.append(f'  {n}: {d["lon"]:.3f}, {d["lat"]:.3f}, {d["depth"]:.0f} m, Tf {d["Tf_h"]:.1f} h, '
                 f'omega_diurnal/f = {d["diurnal_over_f"]:.2f}')
    L += ['', 'wind stress, N m-2']
    L.append(f'  {"st":3s} {"max":>7s} {"at":>17s} {"T1 median":>10s} ' +
             ' '.join(f'{"max " + k:>8s}' for k in C.WKEYS))
    for n in C.STATION_ORDER:
        d = st[n]
        L.append(f'  {n:3s} {d["tau_max"]:7.2f} {d["tau_max_time"]:>17s} {d["tau_median_T1"]:10.3f} ' +
                 ' '.join(f'{d["taumax_" + k]:8.2f}' for k in C.WKEYS))
    L += ['', 'near-inertial wind work integrated per window, rho0 int W_NI dt, kJ m-2 '
              '(station point, band-passed products)']
    L.append(f'  {"st":3s} ' + ' '.join(f'{k:>8s}' for k in C.WKEYS))
    for n in C.STATION_ORDER:
        L.append(f'  {n:3s} ' + ' '.join(f'{st[n]["W_NI_" + k] / 1e3:8.2f}' for k in C.WKEYS))
    L.append(f'  D to C ratio of the T2 injection {st["D"]["W_NI_T2"] / st["C"]["W_NI_T2"]:.1f}, '
             f'D to A {st["D"]["W_NI_T2"] / st["A"]["W_NI_T2"]:.1f}')
    L += ['', 'box-averaged cospectrum check (30 km box)']
    for n in C.STATION_ORDER:
        d = st[n]
        L.append(f'  {n}: band integral {d["W_NI_cospec_box"] * 1e3:.3f} mW m-2, direct over the '
                 f'same record {d["W_NI_direct_box_full"] * 1e3:.3f} (valid part {d["W_NI_direct_box"] * 1e3:.3f}) '
                 f'mW m-2, ratio {d["cospec_ratio"]:.2f}, Parseval check total {d["cospec_total"] * 1e3:.3f} '
                 f'against {d["W_total_box_full"] * 1e3:.3f} mW m-2, T2 box injection '
                 f'{d["W_NI_direct_box_T2"] / 1e3:.2f} kJ m-2')
    L += ['', 'demodulated amplitudes, m s-1']
    for n in DEMOD_STATIONS:
        d = st[n]
        L.append(f'  {n}: near-inertial peak {d["amp_ni_peak"]:.3f} at {d["amp_ni_peak_time"]}, '
                 f'T1 median {d["amp_ni_T1_median"]:.3f} (x{d["amp_ni_peak"] / d["amp_ni_T1_median"]:.0f}), '
                 f'diurnal T1 median {d["amp_diurnal_T1_median"]:.3f}, T2 max {d["amp_diurnal_max_T2"]:.3f}')
    L += ['', 'sector injection, rho0 int W_NI dt over the window, shelf and slope h <= 1000 m']
    L.append(f'  {"win":4s} ' + ' '.join(f'{s + " mean kJ/m2":>18s}' for s in C.SECTOR_NAMES) +
             f' {"South frac":>11s}')
    for k in C.WKEYS:
        L.append(f'  {k:4s} ' + ' '.join(f'{inj[k][s + "_shelf_mean_J_m2"] / 1e3:18.2f}'
                                         for s in C.SECTOR_NAMES) +
                 f' {100 * inj[k]["South_fraction_of_domain_shelf"]:10.0f}%')
    L.append('  the same over the whole sector (no depth limit), South fraction: ' +
             ', '.join(f'{k} {100 * inj[k]["South_fraction_of_domain_full"]:.0f}%' for k in C.WKEYS))
    L += ['', f'filters: {json.dumps(fb.config())}', f'wall time {(_time.time() - t_start) / 60:.1f} min']
    C.write_report(OUT_REPORT, L)
    print('\n'.join(L))

    # ---- archive -----------------------------------------------------------------------
    s = FIELD_STRIDE
    data = {
        'h': (('eta', 'xi'), g['h'][::s, ::s]), 'mask': (('eta', 'xi'), g['mask'][::s, ::s]),
        **{f'W_NI_{k}': (('eta', 'xi'), W[k][::s, ::s]) for k in C.WKEYS},
        'W_NI_series': (('sector', 'day'), series),
        'divider_south_central_lon': (('xi_full',), g['lon'][rows[0], :]),
        'divider_south_central_lat': (('xi_full',), g['lat'][rows[0], :]),
        'divider_central_north_lon': (('xi_full',), g['lon'][rows[1], :]),
        'divider_central_north_lat': (('xi_full',), g['lat'][rows[1], :]),
        'track_lon': (('fix',), track.lon.values), 'track_lat': (('fix',), track.lat.values),
        'track_vmax_ms': (('fix',), track.vmax_ms.values.astype(float)),
        'track_type': (('fix',), track.type.values.astype('U8')),
    }
    for key, (plon, plat) in perim.items():
        data[f'perim_{key}_lon'] = ((f'p_{key}',), plon)
        data[f'perim_{key}_lat'] = ((f'p_{key}',), plat)
    series_keys = ['u_e', 'v_n', 'tau_e', 'tau_n', 'tau', 'w_ni', 'u_lp', 'v_lp', 'u_bp', 'v_bp']
    for n in C.STATION_ORDER:
        d = st[n]
        for key in series_keys:
            data[f'{n}_{key}'] = (('time',), np.asarray(d[key], float))
        data[f'{n}_cospec'] = ((f'freq_{n}',), d['cospec'])
        data[f'{n}_cospec_smooth'] = ((f'freq_{n}',), d['cospec_smooth'])
        data[f'{n}_cospec_omega_over_f'] = ((f'freq_{n}',), d['cospec_omega_over_f'])
    for n in DEMOD_STATIONS:
        data[f'{n}_amp_ni'] = (('time',), st[n]['amp_ni'])
        data[f'{n}_amp_diurnal'] = (('time',), st[n]['amp_diurnal'])
    coords = dict(lon=(('eta', 'xi'), g['lon'][::s, ::s]), lat=(('eta', 'xi'), g['lat'][::s, ::s]),
                  time=time, day=days, sector=list(C.SECTOR_NAMES),
                  track_time=(('fix',), track.time.values))
    stn_attrs = {}
    for n in C.STATION_ORDER:
        for key in ('lon', 'lat', 'depth', 'f_cph', 'Tf_h', 'tau_max', 'tau_median_T1',
                    'W_NI_cospec_box', 'W_NI_direct_box', 'cospec_ratio', 'diurnal_over_f'):
            stn_attrs[f'{n}_{key}'] = float(st[n][key])
        for k in C.WKEYS:
            stn_attrs[f'{n}_W_NI_{k}'] = float(st[n][f'W_NI_{k}'])
    out = xr.Dataset(data_vars=data, coords=coords, attrs=dict(
        source=C.SURFACE_FILE, rho0=C.RHO0, stride=s, dt_hours=dt_h, fs_cph=fs,
        landfall=str(landfall)[:16], landfall_lon=lf_lon, landfall_lat=lf_lat,
        **{k: list(v) for k, v in C.WINDOWS.items()},
        south_top=rows[0], central_top=rows[1], stat_max_depth_m=C.STAT_MAX_DEPTH_M,
        ni_band=list(C.NI_BAND), cospec_method=C.COSPEC_METHOD, cospec_nperseg=nperseg, cospec_edof=edof,
        box_half_cells=C.BOX_HALF_CELLS, demod_stations=list(DEMOD_STATIONS),
        valid_start=str(fb.valid_span()[0])[:16], valid_end=str(fb.valid_span()[1])[:16],
        **stn_attrs, created=_time.strftime('%Y-%m-%d %H:%M')))
    for v in out.data_vars:
        if out[v].dtype.kind == 'f':
            out[v].encoding.update(zlib=True, complevel=4)
    out.to_netcdf(OUT_NC)
    stats = {n: {k: (float(v) if np.isscalar(v) and not isinstance(v, str) else v)
                 for k, v in st[n].items() if np.isscalar(v)} for n in C.STATION_ORDER}
    stats['injection'] = inj
    stats['landfall'] = str(landfall)[:16]
    stats['filters'] = fb.config()
    stats['filter_bands'] = fb.diag
    with open(OUT_JSON, 'w') as fh:
        json.dump(stats, fh, indent=2, default=float)
    print(f'wrote {OUT_NC} and {OUT_JSON}')
    sf.close()


if __name__ == '__main__':
    main()