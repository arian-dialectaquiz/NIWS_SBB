"""
extract_surface_v3.py

One pass over the two archive parts that writes the surface fields of the
analysis subdomain to surface_v3.nc, record by record, in chunks laid out for
time-series access. Every later surface diagnostic (Figure 1, the wind work
maps and series of Figure 6, the station spectra) reads this file instead of
the 1.2 TB archive, whose one-chunk-per-record layout makes a row-block pass
over all times prohibitively slow.

Fields written, native staggering kept so the downstream code is identical to v2

    u_s, v_s        top sigma level velocity          (time, eta_u, xi_u) (time, eta_v, xi_v)
    sustr, svstr    surface stress, scaled to N m-2   same staggering
    zeta            free surface                      (time, eta_rho, xi_rho)

plus the static grid of the subdomain. About 15 GB, uncompressed.

    python extract_surface_v3.py            # full pass
    python extract_surface_v3.py --resume   # continue an interrupted pass, or append the
                                            # records part 2 wrote after the first pass
"""

import argparse
import os
import time as _time

import numpy as np
import netCDF4 as nc

import common_v3 as C

TAG = 'extract'
TCHUNK, RCHUNK, CCHUNK = 48, 64, 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=C.SURFACE_FILE)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    arc = C.Archive()
    arc.describe(TAG)
    arc.assert_even(TAG)
    g = C.load_grid(arc)
    neta, nxi = g['lon'].shape
    nt = arc.nt
    ds0 = arc.ds0
    neta_u, nxi_u = neta, nxi - 1
    neta_v, nxi_v = neta - 1, nxi

    sel_u = dict(eta_u=slice(C.ETA0, C.ETA_SLICE.stop), xi_u=slice(C.XI0, None), s_rho=arc.nlev - 1)
    sel_v = dict(eta_v=slice(C.ETA0, C.ETA_SLICE.stop - 1), xi_v=slice(C.XI0, None), s_rho=arc.nlev - 1)
    sel_su = dict(eta_u=slice(C.ETA0, C.ETA_SLICE.stop), xi_u=slice(C.XI0, None))
    sel_sv = dict(eta_v=slice(C.ETA0, C.ETA_SLICE.stop - 1), xi_v=slice(C.XI0, None))
    sel_z = dict(eta_rho=C.ETA_SLICE, xi_rho=C.XI_SLICE)

    # check the staggered sizes against the archive
    a_u = arc.ds0['u'].isel(sel_u).shape
    a_v = arc.ds0['v'].isel(sel_v).shape
    assert a_u[-2:] == (neta_u, nxi_u), (a_u, neta_u, nxi_u)
    assert a_v[-2:] == (neta_v, nxi_v), (a_v, neta_v, nxi_v)

    scales = C.stress_scale_per_part(arc, TAG)
    scale = scales[0]
    part_of = np.zeros(nt, int)
    for k in range(1, len(arc.offsets) - 1):
        part_of[arc.offsets[k]:arc.offsets[k + 1]] = k

    mode = 'a' if (args.resume and os.path.exists(args.out)) else 'w'
    out = nc.Dataset(args.out, mode, format='NETCDF4')
    if mode == 'w':
        out.createDimension('time', None)          # unlimited, so a resume can append
        for name, n in (('eta_rho', neta), ('xi_rho', nxi), ('eta_u', neta_u),
                        ('xi_u', nxi_u), ('eta_v', neta_v), ('xi_v', nxi_v)):
            out.createDimension(name, n)
        tv = out.createVariable('time', 'f8', ('time',))
        tv.units = 'seconds since 2004-03-13 00:00:00'
        tv.calendar = 'standard'
        tv[:nt] = (arc.time.astype('datetime64[s]') - np.datetime64('2004-03-13T00:00:00')
                   ).astype('timedelta64[s]').astype(float)
        for name in ('lon', 'lat', 'h', 'mask', 'angle', 'pm', 'pn', 'f'):
            v = out.createVariable(name, 'f8', ('eta_rho', 'xi_rho'))
            v[:] = g[name] if name != 'mask' else g['mask']
        done = out.createVariable('done', 'i1', ('time',))
        done[:nt] = np.zeros(nt, np.int8)
        for name, (var, lev, dims) in C.SURFACE_VARS.items():
            shape = tuple(out.dimensions[d].size for d in dims)
            v = out.createVariable(name, 'f4', dims,
                                   chunksizes=(min(TCHUNK, nt), min(RCHUNK, shape[1]),
                                               min(CCHUNK, shape[2])),
                                   zlib=False, fill_value=np.float32(np.nan))
            v.source = var
        out.setncatts(dict(
            source=' + '.join(arc.paths), stress_scale=' '.join(f'{x:g}' for x in scales), rho0=C.RHO0,
            eta0=C.ETA0, xi0=C.XI0, eta_stop=C.ETA_SLICE.stop, top_level=arc.nlev - 1,
            note=('Surface fields of the analysis subdomain, native staggering. u_s and v_s '
                  'are the top sigma level. sustr and svstr are in N m-2. Written record by '
                  'record by extract_surface_v3.py from the two archive parts joined at the '
                  'restart.'), created=_time.strftime('%Y-%m-%d %H:%M')))
    done = out.variables['done']
    n_have = out.dimensions['time'].size
    if n_have < nt:
        # the archive grew since the file was created (part 2 still running then)
        tv = out.variables['time']
        tv[n_have:nt] = (arc.time[n_have:].astype('datetime64[s]')
                         - np.datetime64('2004-03-13T00:00:00')).astype('timedelta64[s]').astype(float)
        done[n_have:nt] = np.zeros(nt - n_have, np.int8)
        C.report(TAG, f'archive grew from {n_have} to {nt} records, appending')
    elif n_have > nt:
        raise SystemExit(f'{args.out} holds {n_have} records but the archive has {nt}, '
                         f'remove the file and extract again')
    todo = np.where(done[:nt] == 0)[0]
    C.report(TAG, f'{todo.size} of {nt} records to write')

    t0 = _time.time()
    for n, it in enumerate(todo):
        out.variables['u_s'][it] = arc.read_record('u', it, sel_u)
        out.variables['v_s'][it] = arc.read_record('v', it, sel_v)
        sc = scales[part_of[it]]
        out.variables['sustr'][it] = arc.read_record('sustr', it, sel_su) * sc
        out.variables['svstr'][it] = arc.read_record('svstr', it, sel_sv) * sc
        out.variables['zeta'][it] = arc.read_record('zeta', it, sel_z)
        done[it] = 1
        if n % 24 == 0 or n == todo.size - 1:
            out.sync()
            el = _time.time() - t0
            C.report(TAG, f'{n + 1:4d}/{todo.size} {str(arc.time[it])[:16]}  '
                          f'{el / 60:.1f} min elapsed, about '
                          f'{el / (n + 1) * (todo.size - n - 1) / 60:.1f} min left')
    out.close()
    arc.close()
    C.report(TAG, f'wrote {args.out} ({os.path.getsize(args.out) / 1e9:.1f} GB)')


if __name__ == '__main__':
    main()