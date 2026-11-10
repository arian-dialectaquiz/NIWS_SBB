"""
common_v3.py

Configuration and routines shared by every compute script of the v3 near-inertial
wave manuscript (Hurricane Catarina, South Brazil Bight, 1 km ROMS hindcast).
Every script imports this module, so no two scripts can disagree on the archive,
the windows, the sectors, the stations, the filters or the constants.

The hindcast is archived in two parts because the run was restarted after a power
loss. Part 1 starts 13 March 2004 11:30 (the initial condition time) and runs to
10 April 03:00, part 2 restarts from the 9 April 18:00 restart record and runs to
17 April 15:30. Both stamp their hourly averages at the centre of the interval,
which puts part 1 on the hour and part 2 on the half hour. The Archive class
brings part 2 onto the part 1 phase by averaging adjacent records (SEAM_ALIGN),
keeps part 1 up to the first record of part 2 and part 2 from there on, and
archive_check.py reports the seam.

Velocity decomposition, identical to v2

    u = ubar + u' + u''

    ubar  Lanczos low pass, cutoff 1.4 Tf / 0.8 = 1.75 Tf (latitude dependent,
          Tf the local inertial period), half-width 72 h
    u'    Lanczos band pass [0.8 f, 1.2 f], half-width 72 h
    u''   the rest
    <.>   phase average, centred running mean over one local inertial period

Windows (inclusive calendar days, eight days each)

    T1 16 to 23 March   pre-storm
    T2 24 to 31 March   storm
    T3  1 to  8 April   wake
    T4  9 to 16 April   late wake
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal
from scipy.fft import rfft, irfft, next_fast_len
from scipy.ndimage import uniform_filter

# =====================================================================
# 1. PATHS
# =====================================================================

ROMS_DIR = '/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km'
ARCHIVE_PARTS = [
    os.path.join(ROMS_DIR, 'avg_1km_NIWS_HC_extended.nc'),
    os.path.join(ROMS_DIR, 'avg_1km_NIWS_HC_extended_part2.nc'),
]
GRID_CHILD = os.path.join(ROMS_DIR, 'inputs/paper_2_1km_closed_cropped_smooth_sponge.nc')
GRID_PARENT = ('/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/3km/inputs/'
               'deproas_spongebob_grd_cropped_3km.nc')
FORCING_FILE = os.path.join(ROMS_DIR, 'inputs/forcing_paper2_hc.nc')
IC_FILE = os.path.join(ROMS_DIR, 'inputs/IC_paper_2_1km.nc')
RST_FILES = [os.path.join(ROMS_DIR, 'rst_paper_2_HC_1km_extended.nc'),
             os.path.join(ROMS_DIR, 'rst_paper_2_HC_1km_extended_part2.nc')]

# Time stamp correction per archive part, seconds added to ocean_time. The two
# parts were written by two builds of ROMS and archive_check.py found the part 1
# stamps on the hour (xx:59:59.99, a decoding artefact of 'days since') and the
# part 2 stamps on the half hour (xx:29:59.99), so the two conventions differ by
# 30 min across the seam. The time stamp audit of the check report (initial
# condition time, restart record times) decides which part moves and in which
# direction. An average stamped at the end of its interval moves by -1800 s to
# the centre. Set here once the audit is read. [0.0, 0.0] reads the files as they are.
TIME_SHIFT_S = [0.0, 0.0]

# The audit of 22 October showed both parts stamped at the centre of their
# hourly interval, part 1 started at 11:30 on 13 March (its intervals run from
# HH:30 to HH+1:30, stamps on the hour) and part 2 restarted at 18:00 on 9 April
# (intervals HH:00 to HH+1:00, stamps on the half hour). With SEAM_ALIGN True a
# part whose stamps sit half a step off the first part is brought onto the first
# part's phase by averaging each pair of adjacent records, which is the linear
# interpolation at the midpoint. The gain of that average is cos(pi dt / T), that
# is 0.991 at 24 h and 0.98 at the 22.8 h inertial period of the South sector, a
# 1 to 2 % attenuation of the near-inertial amplitude in part 2 (T4) only.
SEAM_ALIGN = True
TRACK_CSV = '/home/arian/dd_waves/NIWS/FIGURES_V4/catarina_2004_full.csv'

WORK_DIR = '/home/arian/dd_waves/NIWS_v3'
SURFACE_FILE = os.path.join(WORK_DIR, 'surface_v3.nc')     # written by extract_surface_v3.py


def fig_dir(name):
    """Each figure keeps its archive, report and image in its own subfolder of
    the working directory (fig01, fig02, ..., decomp)."""
    d = os.path.join(os.getcwd(), name)
    os.makedirs(d, exist_ok=True)
    return d

# =====================================================================
# 2. CONSTANTS AND CONVENTIONS
# =====================================================================

RHO0 = 1025.0          # kg m-3, the model RHO0
OMEGA = 7.2921e-5      # rad s-1
G = 9.81               # m s-2
KT2MS = 0.5144444444

# Analysis subdomain of the child grid (rho frame). Identical to v2, so the
# station indices, the sector rows and every field keep their meaning.
ETA_SLICE = slice(0, 1358)
XI_SLICE = slice(40, None)
ETA0 = ETA_SLICE.start or 0
XI0 = XI_SLICE.start or 0

# Sectors, as eta_rho rows of the subset frame. South is eta < SOUTH_TOP,
# Central is SOUTH_TOP <= eta < CENTRAL_TOP, North is eta >= CENTRAL_TOP.
SOUTH_TOP = 420
CENTRAL_TOP = 900
SECTOR_NAMES = ('South', 'Central', 'North', 'Domain')
STAT_MAX_DEPTH_M = 1000.0        # area statistics over the shelf and slope
UPPER_LAYER_M = 100.0            # layer of the subinertial flow maps

# Stations, (xi_rho, eta_rho) in the subset frame. One table only.
STATIONS = {
    'A': dict(xi=380, eta=1233),
    'B': dict(xi=276, eta=896),
    'C': dict(xi=333, eta=557),
    'D': dict(xi=453, eta=323),
}
STATION_ORDER = ['A', 'B', 'C', 'D']
BOX_HALF_CELLS = 15              # 31 x 31 cells, a 30 km square at 1 km

# Windows, inclusive calendar days, seven days each. The archive starts on
# 13 March 12:00 and ends on 19 April 00:00, and the filter edge is 3.6 d, so
# the valid record runs from 17 March 03:00 to 15 April 08:00 and four complete
# equal windows fit only at seven days. T2 holds the passage and the landfall,
# T3 opens on the reversal day (31 March).
WINDOWS = {
    'T1': ('2004-03-17', '2004-03-23'),
    'T2': ('2004-03-24', '2004-03-30'),
    'T3': ('2004-03-31', '2004-04-06'),
    'T4': ('2004-04-07', '2004-04-13'),
}
WINDOW_DAYS = 7
WKEYS = tuple(WINDOWS)
MAP_WINDOWS = ('T1', 'T2', 'T3')

# Reference time. The hour is derived from the best track and the land mask by
# landfall_from_track() and written to every archive. This value is the fallback.
LANDFALL_DEFAULT = np.datetime64('2004-03-28T06:00')

# Filters
NI_BAND = (0.8, 1.2)             # multiples of the local f
LP_BAND_MARGIN = 1.4             # low-pass cutoff = margin x Tf / NI_BAND[0]
LP_HALF_WIDTH_H = 72.0
BP_HALF_WIDTH_H = 72.0
F_BIN_TOL = 0.02                 # relative change of f that opens a new band
LP_LEAK_TOL = 0.10
PHASE_AVG_PERIODS = 1.0          # phase average length in local inertial periods
DIURNAL_PERIOD_H = 24.0

# Station diagnostics
DEMOD_LOWPASS_H = 40.0           # low pass of the back-rotated series
COSPEC_METHOD = 'periodogram'    # 'periodogram' (whole record, boxcar, Parseval exact) or 'welch'
COSPEC_SMOOTH_BINS = 5           # running mean of the periodogram for display
COSPEC_NPERSEG_D = 8.0           # Welch segment length, days
COSPEC_OVERLAP = 0.5
COSPEC_WINDOW = 'hann'

# Horizontal smoothing of the subinertial velocity before the gradient tensor
SMOOTH_CELLS = 3

# Row block of the chunked passes over the archive
ETA_CHUNK = 30

# Bottom drag of the model (RDRG2 in the .in file) and the reference level
CD_BOT = 3.0e-3
BOT_LEVEL_UP = 1


# =====================================================================
# 3. SMALL HELPERS
# =====================================================================

def report(tag, msg):
    print(f'[{tag}] {msg}', flush=True)


def sampling_interval_hours(t):
    dt = np.diff(np.asarray(t).astype('datetime64[s]').astype(float))
    if dt.size and (dt.max() - dt.min()) > 1.0:
        warnings.warn(f'time axis is uneven, spread {(dt.max() - dt.min()) / 3600:.3f} h')
    return float(np.median(dt)) / 3600.0


def window_mask(t, bounds):
    """Boolean mask of the samples inside an inclusive-day window."""
    a, b = bounds
    t = np.asarray(t)
    return (t >= np.datetime64(a)) & (t < np.datetime64(b) + np.timedelta64(1, 'D'))


def as_slice(ii):
    """A slice when the indices are evenly spaced (strided reads are far faster
    than fancy indexing on netCDF), else the index array itself."""
    ii = np.asarray(ii)
    if ii.size == 1:
        return slice(int(ii[0]), int(ii[0]) + 1)
    if ii.size > 1:
        step = int(ii[1] - ii[0])
        if step > 0 and np.all(np.diff(ii) == step):
            return slice(int(ii[0]), int(ii[-1]) + 1, step)
    return ii


def window_indices(t, valid=None):
    """Index arrays of every window, restricted to the valid samples."""
    valid = np.ones(len(t), bool) if valid is None else valid
    return {k: np.where(window_mask(t, b) & valid)[0] for k, b in WINDOWS.items()}


def inertial_period_hours(lat_deg):
    f = 2.0 * OMEGA * np.sin(np.deg2rad(lat_deg))
    return 2.0 * np.pi / np.abs(f) / 3600.0


def coriolis(lat_deg):
    return 2.0 * OMEGA * np.sin(np.deg2rad(lat_deg))


def write_report(path, lines):
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'wrote {path}')


# =====================================================================
# 4. THE TWO-PART ARCHIVE
# =====================================================================

class Archive:
    """The hindcast as one record, read from the two ROMS average files.

    Part 1 is kept for every record earlier than the first record of part 2,
    part 2 supplies the rest. Static grid variables come from part 1. Reads
    return numpy arrays with time first, concatenated across the seam, and are
    lazy on the file side, so a read of one level over one row block touches
    only those records.
    """

    def __init__(self, parts=None, time_name='ocean_time'):
        parts = ARCHIVE_PARTS if parts is None else parts
        self.paths = [p for p in parts if os.path.exists(p)]
        if not self.paths:
            sys.exit(f'no archive part found among {parts}')
        self.tname = time_name
        self.ds = [xr.open_dataset(p, decode_times=True) for p in self.paths]
        self.raw = [xr.open_dataset(p, decode_times=False)[time_name] for p in self.paths]
        shifts = list(TIME_SHIFT_S) + [0.0] * len(self.paths)
        self.shifts = shifts[:len(self.paths)]
        # decoded stamps carry a sub-second decoding artefact (xx:59:59.99), so
        # they are rounded to the second before anything is compared
        times = [((d[time_name].values.astype('datetime64[ms]') + np.timedelta64(500, 'ms'))
                  .astype('datetime64[s]').astype('datetime64[ns]')
                  + np.timedelta64(int(round(sh * 1e3)), 'ms'))
                 for d, sh in zip(self.ds, self.shifts)]
        # a later part whose stamps sit half a step off the first part is
        # interpolated onto the phase of the first part (adjacent-pair average)
        dt0 = np.median(np.diff(times[0]).astype('timedelta64[s]').astype(float))
        self.interp = [False] * len(times)
        for k in range(1, len(times)):
            off = (times[k][0] - times[0][0]).astype('timedelta64[s]').astype(float) % dt0
            off = min(off, dt0 - off)
            if SEAM_ALIGN and off > 1.0:
                self.interp[k] = True
                times[k] = 0.5 * (times[k][:-1] - times[k][0]) + 0.5 * (times[k][1:] - times[k][0]) + times[k][0]
                times[k] = times[k].astype('datetime64[s]').astype('datetime64[ns]')
        # keep part k for t < first time of part k+1
        self.sel = []
        for k, t in enumerate(times):
            if k + 1 < len(times):
                keep = t < times[k + 1][0]
            else:
                keep = np.ones(t.size, bool)
            self.sel.append(np.where(keep)[0])
        self.time = np.concatenate([t[s] for t, s in zip(times, self.sel)])
        self.nt = self.time.size
        self.dt_h = sampling_interval_hours(self.time)
        self.offsets = np.cumsum([0] + [s.size for s in self.sel])
        self.ds0 = self.ds[0]
        self.nlev = int(self.ds0.sizes['s_rho'])
        self.overlap = [int((t.size - s.size)) for t, s in zip(times, self.sel)]
        self.part_times = times

    # ---- description ----------------------------------------------------
    def audit_lines(self):
        """Raw time stamps of each part, for the convention check."""
        out = []
        for k, raw in enumerate(self.raw):
            v = np.asarray(raw.values, float)
            units = raw.attrs.get('units', '?')
            sec = np.mod(v, 3600.0)
            out.append(f'  part {k + 1}: units "{units}", {v.size} records, first raw values '
                       f'{v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f} s, last {v[-1]:.1f} s, '
                       f'stamp within the hour {sec.min():.1f} to {sec.max():.1f} s, '
                       f'shift applied {self.shifts[k]:+.0f} s')
        return out

    def describe(self, tag='archive'):
        for k, (p, d) in enumerate(zip(self.paths, self.ds)):
            t = self.part_times[k]
            report(tag, f'part {k + 1}: {os.path.basename(p)}, {d.sizes[self.tname]} records, '
                        f'{str(t[0])[:16]} to {str(t[-1])[:16]}'
                        + (' after the adjacent-pair average onto the part 1 phase'
                           if self.interp[k] else '')
                        + f', {self.sel[k].size} used, {self.overlap[k]} dropped at the seam')
        report(tag, f'joined record {self.nt} steps, dt = {self.dt_h:.3f} h, '
                    f'{str(self.time[0])[:16]} to {str(self.time[-1])[:16]}')
        gaps = np.diff(self.time).astype('timedelta64[s]').astype(float) / 3600.0
        if gaps.size and (abs(gaps - self.dt_h) > 1e-6).any():
            bad = np.where(abs(gaps - self.dt_h) > 1e-6)[0]
            report(tag, f'WARNING, {bad.size} uneven step(s), first at '
                        f'{str(self.time[bad[0]])[:16]} with {gaps[bad[0]]:.2f} h')
        else:
            report(tag, 'time axis even across the seam')

    def is_even(self, tol_h=1e-3):
        gaps = np.diff(self.time).astype('timedelta64[s]').astype(float) / 3600.0
        return bool(gaps.size == 0 or (np.abs(gaps - self.dt_h) < tol_h).all())

    def assert_even(self, tag='archive'):
        if not self.is_even():
            sys.exit(f'[{tag}] the joined time axis is uneven at the seam. Read the time '
                     f'stamp audit of archive_check_report.txt and set TIME_SHIFT_S in '
                     f'common_v3.py so both parts share the centre-of-interval convention.')

    # ---- reading ---------------------------------------------------------
    def part_slices(self, it=None):
        """List of (part index, record indices in that part) covering the
        joined record indices `it` (None for all)."""
        it = np.arange(self.nt) if it is None else np.atleast_1d(it)
        out = []
        for k in range(len(self.ds)):
            lo, hi = self.offsets[k], self.offsets[k + 1]
            m = (it >= lo) & (it < hi)
            if m.any():
                out.append((k, self.sel[k][it[m] - lo]))
        return out

    def read(self, var, sel=None, it=None, dtype=np.float64):
        """Read `var` with the index selection `sel` (dict of dim: slice or
        int, without the time dimension) over the joined record indices `it`.
        Returns an array with time first."""
        sel = {} if sel is None else dict(sel)
        blocks = []
        for k, rec in self.part_slices(it):
            da = self.ds[k][var]
            if self.tname not in da.dims:
                return da.isel(**sel).values.astype(dtype, copy=False)
            if self.interp[k]:
                rec = np.asarray(rec)
                sl = as_slice(rec)
                if isinstance(sl, slice) and (sl.step or 1) == 1:
                    a = da.isel({self.tname: slice(sl.start, sl.stop + 1), **sel}).values
                    blocks.append((0.5 * (a[:-1] + a[1:])).astype(dtype, copy=False))
                elif isinstance(sl, slice):
                    a = da.isel({self.tname: sl, **sel}).values
                    b = da.isel({self.tname: slice(sl.start + 1, sl.stop + 1, sl.step), **sel}).values
                    blocks.append((0.5 * (a + b)).astype(dtype, copy=False))
                else:
                    a = da.isel({self.tname: rec, **sel}).values
                    b = da.isel({self.tname: rec + 1, **sel}).values
                    blocks.append((0.5 * (a + b)).astype(dtype, copy=False))
            else:
                blocks.append(da.isel({self.tname: as_slice(rec), **sel})
                              .values.astype(dtype, copy=False))
        return np.concatenate(blocks, axis=0) if len(blocks) > 1 else blocks[0]

    def read_record(self, var, it, sel=None, dtype=np.float32):
        """One joined record of `var`, time dimension dropped."""
        (k, rec), = self.part_slices(it)
        sel = {} if sel is None else dict(sel)
        da = self.ds[k][var]
        if self.interp[k]:
            a = da.isel({self.tname: slice(int(rec[0]), int(rec[0]) + 2), **sel}).values
            return (0.5 * (a[0] + a[1])).astype(dtype, copy=False)
        return da.isel({self.tname: int(rec[0]), **sel}).values.astype(dtype, copy=False)

    def static(self, name, stride=1):
        """A 2D grid variable on the rho subdomain."""
        src = self.ds0 if name in self.ds0 else None
        if src is None:
            raise KeyError(f'{name} is not in the archive, read it from the grid file')
        return src[name].isel(eta_rho=ETA_SLICE, xi_rho=XI_SLICE).values[::stride, ::stride]

    def has(self, name):
        return name in self.ds0

    def chunking(self, name):
        enc = self.ds0[name].encoding
        return enc.get('chunksizes'), enc.get('zlib'), enc.get('complevel')

    def close(self):
        for d in self.ds:
            d.close()


# =====================================================================
# 5. GRID
# =====================================================================

def load_grid(arc=None, grid_file=None, stride=1):
    """The static fields of the analysis subdomain. Prefers the archive, falls
    back to the grid file for anything the archive does not hold."""
    grid_file = GRID_CHILD if grid_file is None else grid_file
    g = xr.open_dataset(grid_file) if os.path.exists(grid_file) else None

    def get(name):
        if arc is not None and arc.has(name):
            return arc.static(name, stride)
        if g is not None and name in g:
            return g[name].isel(eta_rho=ETA_SLICE, xi_rho=XI_SLICE).values[::stride, ::stride]
        raise KeyError(name)

    out = dict(lon=get('lon_rho'), lat=get('lat_rho'), h=get('h'),
               mask=get('mask_rho'), pm=get('pm') / stride, pn=get('pn') / stride)
    try:
        out['angle'] = get('angle')
    except KeyError:
        out['angle'] = np.zeros_like(out['lon'])
    out['land'] = out['mask'] < 0.5
    out['f'] = coriolis(out['lat'])
    out['area'] = np.where(out['land'], np.nan, 1.0 / (out['pm'] * out['pn']))
    if g is not None:
        g.close()
    return out


def compute_z_w(ds0, zeta2d, stride=1, h=None):
    """Depths of the w levels (N+1, eta, xi) from the vertical grid parameters
    of the archive and a 2D free surface."""
    if h is None:
        h = ds0['h'].isel(eta_rho=ETA_SLICE, xi_rho=XI_SLICE).values[::stride, ::stride]
    hc = float(ds0['hc'].values)
    s_w, Cs_w = ds0['s_w'].values, ds0['Cs_w'].values
    vtrans = int(ds0['Vtransform'].values) if 'Vtransform' in ds0 else 2
    z0 = np.nan_to_num(np.asarray(zeta2d, float), nan=0.0)
    N = s_w.size - 1
    z_w = np.empty((N + 1,) + h.shape, dtype=np.float64)
    for k in range(N + 1):
        if vtrans == 2:
            S = (hc * s_w[k] + h * Cs_w[k]) / (hc + h)
            z_w[k] = z0 + (z0 + h) * S
        else:
            S = hc * (s_w[k] - Cs_w[k]) + h * Cs_w[k]
            z_w[k] = S + z0 * (1.0 + S / np.maximum(h, 1e-3))
    return z_w


def compute_Hz(ds0, zeta2d, stride=1, h=None):
    return np.diff(compute_z_w(ds0, zeta2d, stride, h), axis=0)


def compute_z_rho(ds0, zeta2d, stride=1, h=None):
    zw = compute_z_w(ds0, zeta2d, stride, h)
    return 0.5 * (zw[:-1] + zw[1:])


def sector_masks(grid, stride=1, max_depth=STAT_MAX_DEPTH_M):
    """Sector masks on the subset grid. 'shelf' restricts to h <= max_depth."""
    neta, nxi = grid['lon'].shape
    eta = np.arange(neta)[:, None] * np.ones((1, nxi), int)
    s_top, c_top = int(round(SOUTH_TOP / stride)), int(round(CENTRAL_TOP / stride))
    wet = ~grid['land']
    shelf = wet & (grid['h'] <= max_depth)
    full = {'South': wet & (eta < s_top),
            'Central': wet & (eta >= s_top) & (eta < c_top),
            'North': wet & (eta >= c_top), 'Domain': wet}
    return {'full': full, 'shelf': {k: v & shelf for k, v in full.items()},
            'sector_id': np.where(full['South'], 1, np.where(full['Central'], 2,
                                  np.where(full['North'], 3, 0))).astype(np.int8),
            'south_top': s_top, 'central_top': c_top}


def area_mean(field, mask, area):
    m = mask & np.isfinite(field) & np.isfinite(area)
    if not m.any():
        return float('nan')
    return float(np.nansum(field[m] * area[m]) / np.nansum(area[m]))


def area_sum(field, mask, area):
    m = mask & np.isfinite(field) & np.isfinite(area)
    return float(np.nansum(field[m] * area[m])) if m.any() else float('nan')


# =====================================================================
# 6. STAGGERING AND ROTATION
# =====================================================================

def u2rho(a):
    out = np.empty(a.shape[:-1] + (a.shape[-1] + 1,), dtype=np.float64)
    out[..., 1:-1] = 0.5 * (a[..., :-1] + a[..., 1:])
    out[..., 0], out[..., -1] = a[..., 0], a[..., -1]
    return out


def v2rho(a):
    out = np.empty(a.shape[:-2] + (a.shape[-2] + 1, a.shape[-1]), dtype=np.float64)
    out[..., 1:-1, :] = 0.5 * (a[..., :-1, :] + a[..., 1:, :])
    out[..., 0, :], out[..., -1, :] = a[..., 0, :], a[..., -1, :]
    return out


def psi2rho(a):
    neta, nxi = a.shape[0] + 1, a.shape[1] + 1
    out = np.full((neta, nxi), np.nan)
    out[1:-1, 1:-1] = 0.25 * (a[:-1, :-1] + a[:-1, 1:] + a[1:, :-1] + a[1:, 1:])
    out[0, 1:-1] = 0.5 * (a[0, :-1] + a[0, 1:])
    out[-1, 1:-1] = 0.5 * (a[-1, :-1] + a[-1, 1:])
    out[1:-1, 0] = 0.5 * (a[:-1, 0] + a[1:, 0])
    out[1:-1, -1] = 0.5 * (a[:-1, -1] + a[1:, -1])
    out[0, 0], out[0, -1] = a[0, 0], a[0, -1]
    out[-1, 0], out[-1, -1] = a[-1, 0], a[-1, -1]
    return out


def grid_to_east_north(u_xi, v_eta, angle):
    """u_east = u_xi cos A - v_eta sin A, v_north = u_xi sin A + v_eta cos A."""
    ca, sa = np.cos(angle), np.sin(angle)
    return u_xi * ca - v_eta * sa, u_xi * sa + v_eta * ca


def strain_to_east_north(Sn_grid, Ss_grid, angle):
    rot = (Sn_grid + 1j * Ss_grid) * np.exp(2j * angle)
    return np.real(rot), np.imag(rot)


def rho_point_from_faces(u_u, v_v, eta, xi):
    """Velocity at a rho point from its two u faces and two v faces. Arrays are
    (time, eta_u, xi_u) and (time, eta_v, xi_v) in the subset frame."""
    u = 0.5 * (u_u[:, eta, xi - 1] + u_u[:, eta, xi])
    v = 0.5 * (v_v[:, eta - 1, xi] + v_v[:, eta, xi])
    return u, v


def nan_boxcar(field, size):
    if size <= 1:
        return field
    valid = np.isfinite(field).astype(np.float64)
    num = uniform_filter(np.where(np.isfinite(field), field, 0.0), size=size,
                         mode='nearest')
    den = uniform_filter(valid, size=size, mode='nearest')
    return np.where(den > 0.5, num / np.maximum(den, 1e-12), np.nan)


def cgrid_gradient_tensor(u_u, v_v, pm, pn):
    """delta, Sn (grid), zeta, Ss (grid) on rho points, flux form on the C grid."""
    pm_u = 0.5 * (pm[:, :-1] + pm[:, 1:])
    pn_u = 0.5 * (pn[:, :-1] + pn[:, 1:])
    pm_v = 0.5 * (pm[:-1, :] + pm[1:, :])
    pn_v = 0.5 * (pn[:-1, :] + pn[1:, :])
    pm_psi = 0.25 * (pm[:-1, :-1] + pm[:-1, 1:] + pm[1:, :-1] + pm[1:, 1:])
    pn_psi = 0.25 * (pn[:-1, :-1] + pn[:-1, 1:] + pn[1:, :-1] + pn[1:, 1:])

    Uflux, Vflux = u_u / pn_u, v_v / pm_v
    dUdxi = np.full(pm.shape, np.nan)
    dVdeta = np.full(pm.shape, np.nan)
    dUdxi[:, 1:-1] = Uflux[:, 1:] - Uflux[:, :-1]
    dVdeta[1:-1, :] = Vflux[1:, :] - Vflux[:-1, :]
    delta = pm * pn * (dUdxi + dVdeta)
    Sn_g = pm * pn * (dUdxi - dVdeta)

    Vf, Uf = v_v / pn_v, u_u / pm_u
    dVdxi = Vf[:, 1:] - Vf[:, :-1]
    dUdeta = Uf[1:, :] - Uf[:-1, :]
    zeta = psi2rho(pm_psi * pn_psi * (dVdxi - dUdeta))
    Ss_g = psi2rho(pm_psi * pn_psi * (dVdxi + dUdeta))
    return delta, Sn_g, zeta, Ss_g


# =====================================================================
# 7. FILTERS
# =====================================================================

def ntaps_for(half_width_h, dt_hours):
    return max(int(round(half_width_h / dt_hours)), 3)


def lanczos_lowpass_weights(cutoff_hours, dt_hours, ntaps):
    """Symmetric Lanczos-windowed sinc low pass (Duchon 1979), zero phase."""
    fc = dt_hours / cutoff_hours
    if not 0.0 < fc < 0.5:
        raise ValueError(f'cutoff {cutoff_hours} h unresolvable at dt = {dt_hours} h')
    k = np.arange(1, ntaps + 1)
    wk = (np.sin(2 * np.pi * fc * k) / (np.pi * k)
          * np.sin(np.pi * k / ntaps) / (np.pi * k / ntaps))
    w = np.concatenate([wk[::-1], [2 * fc], wk])
    return w / w.sum()


def lanczos_bandpass_weights(period_long_h, period_short_h, dt_hours, ntaps):
    return (lanczos_lowpass_weights(period_short_h, dt_hours, ntaps)
            - lanczos_lowpass_weights(period_long_h, dt_hours, ntaps))


def filter_gain(weights, period_h, dt_hours):
    n = (weights.size - 1) // 2
    lag = np.arange(-n, n + 1) * dt_hours
    return float(np.sum(weights * np.cos(2 * np.pi * lag / period_h)))


class TimeFilter:
    """Zero-phase FIR filtering along axis 0 in the frequency domain. One forward
    transform of a block serves every filter of the band."""

    def __init__(self, nt, weights, workers=-1):
        self.nt, self.workers = nt, workers
        maxlen = max(w.size for w in weights.values())
        self.npad = next_fast_len(nt + maxlen)
        self.H = {}
        for name, w in weights.items():
            n = (w.size - 1) // 2
            k = np.zeros(self.npad)
            k[:n + 1] = w[n:]
            k[self.npad - n:] = w[:n]
            self.H[name] = rfft(k)

    def transform(self, arr):
        return rfft(arr, n=self.npad, axis=0, workers=self.workers)

    def apply(self, F, name):
        shape = (slice(None),) + (None,) * (F.ndim - 1)
        return irfft(F * self.H[name][shape], n=self.npad, axis=0,
                     workers=self.workers)[:self.nt]

    def filt(self, arr, name):
        return self.apply(self.transform(arr), name)


def boxcar_time(arr, n):
    """Centred running mean of odd length n along axis 0, edge replicated."""
    n = int(n) | 1
    if n < 3:
        return arr.copy()
    half = n // 2
    pad = np.concatenate([np.repeat(arr[:1], half, axis=0), arr,
                          np.repeat(arr[-1:], half, axis=0)], axis=0)
    c = np.empty((pad.shape[0] + 1,) + pad.shape[1:], dtype=np.float64)
    c[0] = 0.0
    np.cumsum(pad, axis=0, out=c[1:])
    del pad
    return (c[n:] - c[:-n]) / n


class FilterBank:
    """Latitude-dependent Lanczos filters for a (time, rows, cols) block.

    The rows are grouped into bands inside which f changes by less than
    F_BIN_TOL. Each band has its own low pass (cutoff 1.75 Tf), band pass
    ([0.8, 1.2] f) and phase-average length (one local inertial period).
    """

    def __init__(self, lat2d, land, dt_h, nt, need_bp=True, need_pa=True,
                 workers=-1):
        self.dt_h, self.nt = dt_h, nt
        lat_ma = np.where(land, np.nan, lat2d)
        f_row = np.abs(2.0 * OMEGA * np.sin(np.deg2rad(np.nanmean(lat_ma, axis=1))))
        f_row = np.where(np.isfinite(f_row), f_row, np.nanmedian(f_row))
        self.f_row = f_row
        neta = f_row.size
        bands, start = [], 0
        for j in range(1, neta + 1):
            if j == neta or abs(f_row[j] - f_row[start]) / f_row[start] > F_BIN_TOL:
                bands.append((start, j, float(np.mean(f_row[start:j]))))
                start = j
        self.lp_taps = ntaps_for(LP_HALF_WIDTH_H, dt_h)
        self.bp_taps = ntaps_for(BP_HALF_WIDTH_H, dt_h)
        self.filters, self.pa_len, self.diag = {}, {}, []
        pa_half = 0
        for (j0, j1, fb) in bands:
            Tf = 2 * np.pi / fb / 3600.0
            T_lo, T_hi = Tf / NI_BAND[1], Tf / NI_BAND[0]
            T_lp = LP_BAND_MARGIN * Tf / NI_BAND[0]
            wts = dict(lp=lanczos_lowpass_weights(T_lp, dt_h, self.lp_taps))
            if need_bp:
                wts['bp'] = lanczos_bandpass_weights(T_hi, T_lo, dt_h, self.bp_taps)
            self.filters[(j0, j1)] = TimeFilter(nt, wts, workers=workers)
            n_pa = int(round(PHASE_AVG_PERIODS * Tf / dt_h)) | 1
            if need_pa:
                self.pa_len[(j0, j1)] = n_pa
                pa_half = max(pa_half, n_pa // 2)
            d = dict(eta0=j0, eta1=j1, lat=float(np.nanmean(lat_ma[j0:j1])),
                     Tf_h=float(Tf), ni_band_h=[float(T_lo), float(T_hi)],
                     lp_cutoff_h=float(T_lp),
                     lp_ni_leak=filter_gain(wts['lp'], T_hi, dt_h))
            if need_bp:
                d['inertial_gain'] = filter_gain(wts['bp'], Tf, dt_h)
                d['diurnal_gain'] = filter_gain(wts['bp'], DIURNAL_PERIOD_H, dt_h)
            self.diag.append(d)
        self.pa_half = pa_half
        self.edge = max(self.lp_taps, self.bp_taps if need_bp else 0) + pa_half
        self.valid = np.zeros(nt, bool)
        self.valid[self.edge:nt - self.edge] = True
        self.lp_lo = min(d['lp_cutoff_h'] for d in self.diag)
        self.lp_hi = max(d['lp_cutoff_h'] for d in self.diag)
        self.worst_lp_leak = max(abs(d['lp_ni_leak']) for d in self.diag)

    def describe(self, tag):
        report(tag, f'inertial period {2*np.pi/self.f_row.max()/3600:.1f} to '
                    f'{2*np.pi/self.f_row.min()/3600:.1f} h, {len(self.diag)} filter bands')
        report(tag, f'low pass cutoff {LP_BAND_MARGIN:g} Tf / {NI_BAND[0]:g} = '
                    f'{self.lp_lo:.0f} to {self.lp_hi:.0f} h, {self.lp_taps} taps '
                    f'({self.lp_taps*self.dt_h/24:.1f} d half-width), largest leak '
                    f'into the near-inertial band {self.worst_lp_leak:.3f}')
        if self.worst_lp_leak > LP_LEAK_TOL:
            report(tag, f'NOTE, leak above {LP_LEAK_TOL:.2f}, raise LP_BAND_MARGIN')
        report(tag, f'band pass [{NI_BAND[0]}, {NI_BAND[1]}] f, {self.bp_taps} taps, '
                    f'phase average {PHASE_AVG_PERIODS:g} inertial period, '
                    f'edge {self.edge} samples = {self.edge*self.dt_h/24:.1f} d each side')
        report(tag, f'valid record {str(self.valid_span()[0])[:16]} to '
                    f'{str(self.valid_span()[1])[:16]}' if hasattr(self, 'time') else
                    f'valid samples {self.edge} to {self.nt - self.edge - 1}')

    def attach_time(self, time):
        self.time = np.asarray(time)

    def valid_span(self):
        return self.time[self.edge], self.time[self.nt - self.edge - 1]

    def apply(self, arr, row0, name):
        """Filter a (time, rows, cols) block whose first row is `row0` of the
        subset grid."""
        out = np.empty_like(arr, dtype=np.float64)
        nrow = arr.shape[1]
        for (b0, b1), tf in self.filters.items():
            a = max(b0, row0) - row0
            b = min(b1, row0 + nrow) - row0
            if b > a:
                out[:, a:b, :] = tf.filt(arr[:, a:b, :], name)
        return out

    def apply_both(self, arr, row0, names=('lp', 'bp')):
        outs = {n: np.empty_like(arr, dtype=np.float64) for n in names}
        nrow = arr.shape[1]
        for (b0, b1), tf in self.filters.items():
            a = max(b0, row0) - row0
            b = min(b1, row0 + nrow) - row0
            if b <= a:
                continue
            F = tf.transform(arr[:, a:b, :])
            for n in names:
                outs[n][:, a:b, :] = tf.apply(F, n)
            del F
        return outs

    def phase_average(self, arr, row0):
        out = np.empty_like(arr, dtype=np.float64)
        nrow = arr.shape[1]
        for (b0, b1), n in self.pa_len.items():
            a = max(b0, row0) - row0
            b = min(b1, row0 + nrow) - row0
            if b > a:
                out[:, a:b, :] = boxcar_time(arr[:, a:b, :], n)
        return out

    def band_of_row(self, row):
        for (b0, b1) in self.filters:
            if b0 <= row < b1:
                return (b0, b1)
        raise IndexError(row)

    def config(self):
        return dict(ni_band=list(NI_BAND), lp_band_margin=LP_BAND_MARGIN,
                    lp_cutoff_range_h=[self.lp_lo, self.lp_hi], lp_taps=self.lp_taps,
                    bp_taps=self.bp_taps, lp_max_ni_leak=self.worst_lp_leak,
                    phase_average_periods=PHASE_AVG_PERIODS, edge_samples=int(self.edge),
                    edge_days=float(self.edge * self.dt_h / 24), n_bands=len(self.diag))


def station_filters(lat_deg, dt_h, nt):
    """Low pass and band pass weights for one station, same rules as the bank."""
    Tf = inertial_period_hours(lat_deg)
    lp = lanczos_lowpass_weights(LP_BAND_MARGIN * Tf / NI_BAND[0], dt_h,
                                 ntaps_for(LP_HALF_WIDTH_H, dt_h))
    bp = lanczos_bandpass_weights(Tf / NI_BAND[0], Tf / NI_BAND[1], dt_h,
                                  ntaps_for(BP_HALF_WIDTH_H, dt_h))
    return TimeFilter(nt, dict(lp=lp, bp=bp)), Tf


# =====================================================================
# 8. SURFACE STRESS, SPECTRA, DEMODULATION
# =====================================================================

def stress_scale_per_part(arc, tag='stress'):
    """One conversion factor to N m-2 per archive part. The two parts were
    written by two ROMS builds, and a part that stores the kinematic stress
    (tau / rho0, peaks below 0.05 m2 s-2 even under a hurricane) is scaled by
    rho0 while a part in N m-2 is left alone."""
    scales = []
    for k, d in enumerate(arc.ds):
        n = d.sizes[arc.tname]
        sample = d['sustr'].isel({arc.tname: slice(0, n, max(1, n // 40)),
                                  'eta_u': slice(ETA0, ETA0 + 400),
                                  'xi_u': slice(XI0, None)}).values
        peak = float(np.nanmax(np.abs(sample)))
        units = d['sustr'].attrs.get('units', '?')
        sc = RHO0 if peak < 0.05 else 1.0
        report(tag, f'part {k + 1} stress: peak {peak:.4g}, units "{units}", scale {sc:g}')
        scales.append(sc)
    return scales


def detect_stress_scale(sample, override=None, tag='stress'):
    """Factor that converts the stored surface stress to N m-2. ROMS writes
    sustr and svstr multiplied by rho0 (N m-2); a kinematic archive (m2 s-2)
    peaks below 0.05 even in a hurricane core."""
    if override is not None:
        return float(override)
    peak = float(np.nanmax(np.abs(sample)))
    if peak < 0.05:
        report(tag, f'stored stress peaks at {peak:.3e}, kinematic, multiplied by rho0')
        return RHO0
    report(tag, f'stored stress peaks at {peak:.3f}, taken as N m-2')
    return 1.0


def welch_cospectrum(tx, ty, u, v, fs, nperseg, overlap=COSPEC_OVERLAP,
                     window=COSPEC_WINDOW):
    """One-sided Welch cospectrum Re{T U*} of the stress and velocity vectors,
    density per cycle per hour, so that its integral over a band is the mean
    wind work of that band. Arrays may be (time,) or (time, n)."""
    tx, ty, u, v = (np.asarray(a, float) for a in (tx, ty, u, v))
    kw = dict(fs=fs, window=window, nperseg=nperseg, noverlap=int(nperseg * overlap),
              detrend='constant', scaling='density', axis=0)
    f, cx = signal.csd(tx, u, **kw)
    _, cy = signal.csd(ty, v, **kw)
    return f, np.real(cx + cy)


def periodogram_cospectrum(tx, ty, u, v, fs, smooth_bins=5):
    """One-sided cospectrum of the whole record with a boxcar window, so that
    the integral over all frequencies equals the record mean of the wind work
    exactly (Parseval), smoothed by a running mean of smooth_bins for display.
    Arrays may be (time,) or (time, n). Returns the frequency in cph and the
    density in W m-2 cph-1 (the mean over the cells when 2D)."""
    tx, ty, u, v = (np.asarray(a, float) for a in (tx, ty, u, v))
    n = tx.shape[0]
    kw = dict(fs=fs, window='boxcar', nperseg=n, noverlap=0, detrend='constant',
              scaling='density', axis=0)
    f, cx = signal.csd(tx, u, **kw)
    _, cy = signal.csd(ty, v, **kw)
    co = np.real(cx + cy)
    if co.ndim == 2:
        co = co.mean(axis=1)
    if smooth_bins > 1:
        k = int(smooth_bins) | 1
        co_s = np.convolve(co, np.ones(k) / k, mode='same')
        co_s[:k // 2], co_s[-(k // 2):] = co[:k // 2], co[-(k // 2):]
        return f, co, co_s
    return f, co, co


def welch_edof(n_total, nperseg, overlap):
    step = int(nperseg * (1 - overlap))
    n_seg = 1 + (n_total - nperseg) // max(step, 1)
    if overlap == 0.5:
        return (36.0 / 19.0) * (2.0 * n_total / nperseg - 1.0), n_seg
    return 2.0 * n_seg, n_seg


def complex_demodulate(u, v, target_cph, fs, lowpass_h=DEMOD_LOWPASS_H, order=4):
    """Complex demodulation about target_cph (positive for anticlockwise
    rotation, the inertial sense in the Southern Hemisphere is clockwise, so
    the inertial envelope is demodulated at -f)."""
    w = np.asarray(u, float) + 1j * np.asarray(v, float)
    t = np.arange(w.size) / fs
    w_back = w * np.exp(-2j * np.pi * target_cph * t)
    b, a = signal.butter(order, 1.0 / lowpass_h, btype='low', fs=fs)
    env = signal.filtfilt(b, a, w_back)
    return env, t


# =====================================================================
# 9. BEST TRACK AND LANDFALL
# =====================================================================

COLUMN_ALIASES = {
    'time': ['time', 'iso_time', 'date', 'datetime'],
    'lat': ['lat', 'latitude', 'usa_lat'],
    'lon': ['lon', 'longitude', 'usa_lon'],
    'vmax': ['vmax', 'wmo_wind', 'usa_wind', 'wind'],
    'mslp': ['mslp', 'wmo_pres', 'usa_pres', 'pres'],
    'type': ['type', 'nature', 'usa_status', 'status'],
}


def read_track(path=TRACK_CSV, wind_units='kt'):
    df = pd.read_csv(path)
    lower = {c.lower().strip(): c for c in df.columns}
    found = {}
    for key, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in lower:
                found[key] = lower[a]
                break
    missing = {'time', 'lat', 'lon'} - set(found)
    if missing:
        raise KeyError(f'track CSV lacks {sorted(missing)}, columns {list(df.columns)}')
    out = pd.DataFrame({'time': pd.to_datetime(df[found['time']]),
                        'lat': pd.to_numeric(df[found['lat']], errors='coerce'),
                        'lon': pd.to_numeric(df[found['lon']], errors='coerce')})
    out['vmax'] = (pd.to_numeric(df[found['vmax']], errors='coerce')
                   if 'vmax' in found else np.nan)
    out['mslp'] = (pd.to_numeric(df[found['mslp']], errors='coerce')
                   if 'mslp' in found else np.nan)
    out['type'] = (df[found['type']].astype(str).str.strip()
                   if 'type' in found else 'NA')
    out = out.dropna(subset=['lat', 'lon']).reset_index(drop=True)
    out['vmax_ms'] = out['vmax'] * (KT2MS if wind_units.lower().startswith('k') else 1.0)
    return out


def landfall_from_track(track, lon, lat, mask, step_minutes=10):
    """First time the interpolated storm centre lies on a land cell of the
    model grid. Returns (time, lon, lat) or (None, nan, nan)."""
    t = track.time.values.astype('datetime64[s]').astype(float)
    ti = np.arange(t[0], t[-1], step_minutes * 60.0)
    lo = np.interp(ti, t, track.lon.values)
    la = np.interp(ti, t, track.lat.values)
    land = mask < 0.5
    lonf, latf = lon.ravel(), lat.ravel()
    at_sea_before = False
    for k in range(ti.size):
        d2 = (lonf - lo[k]) ** 2 * np.cos(np.deg2rad(la[k])) ** 2 + (latf - la[k]) ** 2
        i = int(np.argmin(d2))
        if d2[i] > 0.05 ** 2:          # outside the grid
            continue
        is_land = land.ravel()[i]
        if at_sea_before and is_land:
            return np.datetime64(int(ti[k]), 's'), float(lo[k]), float(la[k])
        at_sea_before = at_sea_before or (not is_land)
    return None, float('nan'), float('nan')


def grid_perimeter(lon, lat, stride=1):
    lo, la = lon[::stride, ::stride], lat[::stride, ::stride]
    plon = np.concatenate([lo[0, :], lo[:, -1], lo[-1, ::-1], lo[::-1, 0]])
    plat = np.concatenate([la[0, :], la[:, -1], la[-1, ::-1], la[::-1, 0]])
    return plon, plat


# =====================================================================
# 10. THE SURFACE ARCHIVE written by extract_surface_v3.py
# =====================================================================

SURFACE_VARS = {
    # name: (archive variable, level selection, dims in the surface file)
    'u_s': ('u', 'top', ('time', 'eta_u', 'xi_u')),
    'v_s': ('v', 'top', ('time', 'eta_v', 'xi_v')),
    'sustr': ('sustr', None, ('time', 'eta_u', 'xi_u')),
    'svstr': ('svstr', None, ('time', 'eta_v', 'xi_v')),
    'zeta': ('zeta', None, ('time', 'eta_rho', 'xi_rho')),
}


def open_surface(path=SURFACE_FILE):
    if not os.path.exists(path):
        sys.exit(f'{path} not found, run extract_surface_v3.py first')
    return xr.open_dataset(path, decode_times=True)


def box_slices(station, half=None):
    """Slices of the rho frame covering a square box around a station."""
    half = BOX_HALF_CELLS if half is None else half
    xi, eta = STATIONS[station]['xi'], STATIONS[station]['eta']
    return slice(eta - half, eta + half + 1), slice(xi - half, xi + half + 1)


# =====================================================================
# 11. OVERRIDES FOR TESTS AND REDUCED RUNS
# =====================================================================
# A JSON file named by the environment variable NIWS_V3_CONFIG replaces any of
# the module constants above (paths, slices as two-element lists, STATIONS,
# SOUTH_TOP, ...). Used by the synthetic test and by reduced-resolution trials.

_cfg = os.environ.get('NIWS_V3_CONFIG')
if _cfg:
    with open(_cfg) as _fh:
        _over = json.load(_fh)
    for _k, _v in _over.items():
        if _k in ('ETA_SLICE', 'XI_SLICE'):
            _v = slice(*_v)
        globals()[_k] = _v
    ETA0 = ETA_SLICE.start or 0
    XI0 = XI_SLICE.start or 0
    WKEYS = tuple(WINDOWS)
    SURFACE_FILE = _over.get('SURFACE_FILE', os.path.join(WORK_DIR, 'surface_v3.nc'))
    print(f'[common_v3] overrides from {_cfg}: {sorted(_over)}')