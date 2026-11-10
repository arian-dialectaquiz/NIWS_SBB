"""
diag_stress.py

Where does the part 2 stress go missing, in the archive or in surface_v3.nc?
Prints, for a few records of each part, the maximum and the NaN count of
sustr straight from the archive files and the same from the surface file, with
the surface velocity alongside. Takes a minute.

    python diag_stress.py
"""

import numpy as np
import netCDF4 as nc
import xarray as xr

import common_v3 as C

TAG = 'diag'


def main():
    arc = C.Archive()
    arc.describe(TAG)
    sel_u = dict(eta_u=slice(C.ETA0, C.ETA_SLICE.stop), xi_u=slice(C.XI0, None))
    print('\narchive, raw (no scale, no alignment)')
    for k, d in enumerate(arc.ds):
        n = d.sizes[arc.tname]
        v = d['sustr']
        print(f'  part {k + 1}: sustr dims {v.dims}, dtype {v.dtype}, attrs {dict(v.attrs)}')
        for i in (0, n // 2, n - 1):
            a = v.isel({arc.tname: i, **sel_u}).values.astype(float)
            us = d['u'].isel({arc.tname: i, 's_rho': arc.nlev - 1, **sel_u}).values.astype(float)
            t = d[arc.tname].values[i]
            print(f'    record {i:4d} {str(t)[:16]}  |sustr| max {np.nanmax(np.abs(a)):.4g} '
                  f'median {np.nanmedian(np.abs(a)):.4g} NaN {np.isnan(a).sum()} zeros '
                  f'{(a == 0).sum()} of {a.size}   |u_s| max {np.nanmax(np.abs(us)):.3f}')
        for name in ('svstr', 'ubar', 'zeta'):
            if name in d:
                a = d[name].isel({arc.tname: n // 2}).values.astype(float)
                print(f'    record {n // 2:4d} {name}: max |.| {np.nanmax(np.abs(a)):.4g}, NaN {np.isnan(a).sum()}')

    print('\nthrough Archive.read_record (aligned, joined index)')
    for it in (0, arc.offsets[1] - 1, arc.offsets[1], arc.offsets[1] + 5, arc.nt - 1):
        a = arc.read_record('sustr', it, sel_u).astype(float)
        print(f'    joined {it:4d} {str(arc.time[it])[:16]}  |sustr| max {np.nanmax(np.abs(a)):.4g} '
              f'NaN {np.isnan(a).sum()} zeros {(a == 0).sum()}')

    print(f'\nsurface file {C.SURFACE_FILE}')
    sf = nc.Dataset(C.SURFACE_FILE)
    sf.set_auto_mask(False)
    print(f'  stress_scale attribute "{sf.getncattr("stress_scale")}", {sf.dimensions["time"].size} records, '
          f'done flags {int(sf.variables["done"][:].sum())}')
    nt = sf.dimensions['time'].size
    for it in (0, arc.offsets[1] - 1, arc.offsets[1], arc.offsets[1] + 5, nt - 1):
        a = sf.variables['sustr'][it].astype(float)
        u = sf.variables['u_s'][it].astype(float)
        print(f'    record {it:4d}  |sustr| max {np.nanmax(np.abs(a)):.4g} NaN {np.isnan(a).sum()} '
              f'zeros {(a == 0).sum()}   |u_s| max {np.nanmax(np.abs(u)):.3f}')
    sf.close()
    arc.close()


if __name__ == '__main__':
    main()