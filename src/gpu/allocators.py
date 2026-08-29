"""SoA allocation against the frozen cell struct. [Shrestha]

Everything the frame loop will ever touch is allocated here, once, at startup.
Nothing below allocates after `allocate()` returns -- the compile-time memory
bound is a headline claim in the report, and one allocation in the loop makes
it false.

Layout. One flat array per field, spanning all rings, with per-ring offsets.
Structure-of-arrays, never array-of-structs, so a kernel reading only
`ground_height` touches contiguous memory.

Storage shape. Each ring is an ANNULUS -- the square of half-width R_L minus
the hole covered by ring L-1 -- because that is what the 745,000-cell headline
counts. Storing four full squares instead would cost 910,000 cells and quietly
break every ratio in the report. The annulus is laid out as four rectangular
bands (above the hole, below it, left of it, right of it), which keeps
addressing O(1) arithmetic with no lookup table:

        +-----------------+
        |       A         |   A: rows above the hole, full width
        +-----+-----+-----+
        |  C  |/////|  D  |   C, D: rows beside the hole
        +-----+-----+-----+
        |       B         |   B: rows below the hole, full width
        +-----------------+

This file owns storage, not lattice semantics: `annulus_index()` takes cell
coordinates that are already on ring L's integer lattice (math §2, Aakash).
"""

import ctypes
import mmap
import sys
from dataclasses import dataclass, field

import numpy as np
from vrgrid.cell import CELL_BYTES, CELL_FIELDS
from vrgrid.gpu.kernels import (
    CEILING_NONE,
    new_dense_scratch,
    new_sorted_scratch,
    scatter_scratch_bytes,
)
from vrgrid.gpu.pyramid import NODE_BYTES, allocate_pyramid, pyramid_bytes
from vrgrid.gpu.pyramid import scratch_bytes as pyramid_scratch_bytes

# --- the state of a cell nothing has been written into yet ---------------------
#
# `np.zeros` is the right empty value for nine of the ten fields: obs_count 0,
# log_odds 0 (§10.1 decides unknown by observation count, not by log-odds near
# zero), and variance code 0, which the codec deliberately maps to MAXIMUM
# variance so a fresh cell claims no certainty it has not earned.
#
# `ceiling_height` is the exception and it is not a small one. Zero decodes as
# "something solid at the ground datum", so `ceiling - ground < h_vehicle`
# holds for every cell in the map and TRAV_CLEARANCE marks the entire world
# untraversable -- permanently, because `fuse()` only ever lowers a ceiling and
# nothing raises one back up. The empty value here is a sentinel, not a zero.
#
# Two places produce empty cells and they have to agree: `allocate()` at
# startup, and the strip `shift()` clears as the window scrolls. That is why
# this is one dict rather than two literals in two files -- getting it right in
# only one of them yields a map that is correct until the vehicle moves.
EMPTY_CELL = {"ceiling_height": CEILING_NONE}


def initialise_cells(soa: dict, slots=None) -> None:
    """Put cells into the state a never-observed map is supposed to be in.

    `slots` selects which cells to reset; None means the whole array. Fields
    absent from `EMPTY_CELL` are left alone: the caller has already zeroed
    them and zero is their empty value. Passing a dict with none of the named
    fields -- the transient layer, say -- is a no-op rather than an error.
    """
    for name, value in EMPTY_CELL.items():
        arr = soa.get(name)
        if arr is None:
            continue
        if slots is None:
            arr[:] = value
        else:
            arr[slots] = value


# --- how much memory we are ACTUALLY costing the machine -----------------------
#
# `np.zeros` does not allocate memory. It asks for a mapping and the kernel
# hands back copy-on-write zero pages that cost nothing until written to:
# measured, `np.zeros(2_560_000_000, np.uint8)` moves RSS by 0.0 MB. Two
# consequences, and both matter more than they look.
#
# The demo one: a dense-3D baseline built the obvious way would show 0 MB on
# screen beside our counter, and we would be claiming a 286x reduction over
# something visibly free, in front of judges. See gpu/baseline.py.
#
# The one that is ours: if our own grid is never faulted in either, the first
# frames pay the page faults instead -- which is a latency spike in exactly
# the p99 the 10 Hz claim rests on, and it lands during the demo rather than
# during a benchmark. So `allocate()` commits what it allocates, and the
# preallocation is real rather than promised.

# `os.sysconf` and /proc are POSIX; two of the three devs are on Windows and
# CI is ubuntu, so a Linux-only import here is invisible in CI and fatal
# locally. `mmap.PAGESIZE` is the same number from the stdlib on every
# platform. -- portability fix, Aakash
PAGE_BYTES = mmap.PAGESIZE

# Refuse to allocate past this share of what the OS says is available. An OOM
# kill halfway through the demo is a worse outcome than a baseline that
# declines to run and says why.
SAFETY_FRACTION = 0.6


def resident_bytes() -> int:
    """This process's resident set size, from the OS rather than from us.

    The counter on the dashboard should read this, not `nbytes`. `nbytes` is
    what we asked for; this is what we are actually costing the machine, and
    the difference between them is the entire subject of this file.
    """
    if sys.platform != "win32":
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * PAGE_BYTES
    return _windows_working_set()


def available_bytes() -> int:
    """MemAvailable, the kernel's own estimate of what can be had without
    swapping. Deliberately not MemFree, which excludes reclaimable cache and
    would refuse allocations that would have been fine."""
    if sys.platform != "win32":
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        raise RuntimeError("MemAvailable missing from /proc/meminfo")
    return _windows_available()


# --- Windows equivalents of the two /proc reads above ------------------------
# Same quantities, from the Win32 API. Kept together and out of the way so the
# Linux path above still reads as the primary one -- the Jetson is the target
# and these exist so the thing can be developed on the machines we have.


def _windows_working_set() -> int:
    """WorkingSetSize from GetProcessMemoryInfo -- the Windows name for RSS."""
    import ctypes

    class _Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    # argtypes are not optional here: a HANDLE is 64-bit and ctypes defaults
    # to c_int, so the pseudo-handle is truncated and the call just fails.
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p,
                                           ctypes.POINTER(_Counters),
                                           ctypes.c_ulong]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int

    counters = _Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                      ctypes.byref(counters), counters.cb):
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return int(counters.WorkingSetSize)


def _windows_available() -> int:
    """ullAvailPhys from GlobalMemoryStatusEx. Not the same estimate as
    MemAvailable -- it does not count reclaimable cache -- so it is the more
    conservative of the two, which is the right direction for a check whose
    job is to refuse an allocation that would OOM mid-demo."""
    import ctypes

    class _Status(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_Status)]
    kernel32.GlobalMemoryStatusEx.restype = ctypes.c_int

    status = _Status()
    status.dwLength = ctypes.sizeof(status)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError(ctypes.get_last_error(), "GlobalMemoryStatusEx failed")
    return int(status.ullAvailPhys)


def commit(array: np.ndarray) -> np.ndarray:
    """Fault in every page so the allocation is real and the counter is honest.

    Any *store* to a copy-on-write zero page faults it in, so an OR with zero
    commits the page while leaving the byte exactly as it was. Assigning zero
    would commit it just as well and would quietly blank one byte per page of
    whatever it was handed -- harmless on the fresh `np.zeros` this is called
    on today, and a corruption bug the first time someone reuses it.
    """
    flat = array.reshape(-1).view(np.uint8)
    flat[::PAGE_BYTES] |= 0
    return array


def resident_fraction(array: np.ndarray) -> float | None:
    """What share of THIS array's own pages the OS holds in core, via
    `mincore(2)`. Returns None where mincore is unavailable.

    Why this exists alongside `resident_bytes()`. The obvious way to ask
    whether an allocation is real is to read process RSS before and after and
    subtract, and that measurement is wrong in a way that only shows up once a
    process has been running for a while: glibc raises its mmap threshold when
    it sees a large block freed, so a later allocation of about that size comes
    off the heap and reuses pages the process ALREADY has resident. RSS barely
    moves, the delta reads as a few tens of per cent of what was claimed, and
    the honest conclusion "these pages are faulted in" is reported as a
    failure. Measured here: a 64 MB baseline showed a 42 MB delta when it ran
    after the allocator tests and the full 64 MB when it ran alone.

    So the process delta is the right instrument for "what did this cost the
    machine" -- which is the dashboard counter, and it stays -- and the wrong
    one for "are these particular pages in core". This is the second question,
    asked of the pages themselves, and it does not care what else the process
    has done.
    """
    if _MINCORE is None or array.nbytes == 0:
        return None

    base = array.__array_interface__["data"][0]
    start = base - (base % PAGE_BYTES)
    length = base + array.nbytes - start
    pages = (length + PAGE_BYTES - 1) // PAGE_BYTES

    vec = (ctypes.c_ubyte * pages)()
    if _MINCORE(ctypes.c_void_p(start), ctypes.c_size_t(length), vec) != 0:
        return None
    # Bit 0 is the residency bit; the rest are reserved and are not always 0.
    return float(np.count_nonzero(np.frombuffer(vec, np.uint8) & 1) / pages)


def _load_mincore():
    """`mincore` if the platform has it, else None. POSIX only -- two of the
    three devs are on Windows, where this stays None and callers fall back."""
    import ctypes.util

    name = ctypes.util.find_library("c")
    if name is None:
        return None
    try:
        libc = ctypes.CDLL(name, use_errno=True)
        fn = libc.mincore
    except (OSError, AttributeError):
        return None
    fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
    fn.restype = ctypes.c_int
    return fn


_MINCORE = _load_mincore()


# The transient layer shares the foveated grid geometry (master v4 §3.7) but
# not the full 12-byte cell: a dynamic-obstacle hit needs a height, an
# occupancy value and a flag, nothing else.
TRANSIENT_FIELDS = [
    ("ground_height", np.int16),  # 2 B
    ("log_odds", np.int8),        # 1 B
    ("flags", np.uint8),          # 1 B
]
TRANSIENT_BYTES = 4

# Tracked objects persist ~1 s with constant-velocity prediction, so a
# pedestrian briefly hidden by a parked car does not vanish. Capped, with the
# same priority eviction as the refinement pool.
TRACK_DTYPE = np.dtype([
    ("x_m", np.float32), ("y_m", np.float32),
    ("vx_ms", np.float32), ("vy_ms", np.float32),
    ("semantic_class", np.uint8), ("frames_since_seen", np.uint8),
    ("track_id", np.uint16),
])


def array_module(device: str = "cpu"):
    """numpy on cpu, cupy on gpu. Every allocation below goes through this so
    the same code path serves both and the CPU run stays byte-identical."""
    if device == "cpu":
        return np
    import cupy

    return cupy


@dataclass
class RingLayout:
    """Where ring L lives in the flat arrays, and how its annulus is banded."""

    ring: int
    cell_m: float
    side: int          # W: cells across the full square
    hole: int          # w: cells across the hole covered by the inner ring
    count: int         # W^2 - w^2, the LOGICAL cells -- what every report ratio counts
    offset: int        # start index in the flat per-field arrays

    @property
    def slots(self) -> int:
        """Allocated slots. Toroidal storage keeps the full square so the
        ego-motion shift stays O(perimeter); see gpu/shift.py for the
        measurement that decided it. `slots - count` is the padding."""
        return self.side * self.side

    @property
    def lo(self) -> int:
        return (self.side - self.hole) // 2

    @property
    def hi(self) -> int:
        return self.lo + self.hole

    @property
    def band_offsets(self) -> tuple:
        """Start index of bands A, B, C, D within this ring."""
        a = self.lo * self.side
        b = (self.side - self.hi) * self.side
        c = self.hole * self.lo
        return 0, a, a + b, a + b + c


def derive_ring_layouts(schedule, storage: str = "toroidal") -> list:
    """Ring geometry from the schedule, cross-checked against its stated counts.

    The check matters: if someone edits a half-width in the YAML without
    recomputing `cells`, the config and the memory table silently disagree.
    This is where that gets caught, at startup, not in the report.
    """
    layouts, offset = [], 0
    for i, r in enumerate(schedule.rings):
        side = round(2 * r.half_width_m / r.cell_m)
        inner_r = schedule.rings[i - 1].half_width_m if i else 0.0
        hole = round(2 * inner_r / r.cell_m)
        if side % 2 or hole % 2:
            raise ValueError(f"ring {r.ring}: half-width {r.half_width_m} m is not an "
                             f"even number of {r.cell_m} m cells")
        count = side * side - hole * hole
        if count != r.cells:
            raise ValueError(f"ring {r.ring}: geometry gives {count:,} cells but the "
                             f"config says {r.cells:,} -- one of them is wrong")
        layouts.append(RingLayout(r.ring, r.cell_m, side, hole, count, offset))
        offset += side * side if storage == "toroidal" else count
    logical = sum(r.count for r in layouts)
    if logical != schedule.total_cells:
        raise ValueError(f"rings sum to {logical:,}, config says {schedule.total_cells:,}")
    return layouts


def annulus_index(layout: RingLayout, ix, iy, xp=np):
    """Flat index within ring L for cell coordinates already on ring L's lattice.

    `ix`, `iy` are 0-based from the ring's top-left corner. Returns -1 for a
    cell inside the hole, which belongs to a finer ring -- callers must treat
    -1 as "not mine", never as index 0.

    Vectorised: pass arrays, get an array. This is on the scatter hot path.
    Pass `xp=cupy` when the coordinates already live on the device.
    """
    ix = xp.asarray(ix)
    iy = xp.asarray(iy)
    W, lo, hi = layout.side, layout.lo, layout.hi
    off_a, off_b, off_c, off_d = layout.band_offsets

    out = xp.full(ix.shape, -1, dtype=xp.int64)
    inside = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < W)

    above = inside & (iy < lo)
    below = inside & (iy >= hi)
    beside = inside & ~above & ~below
    left = beside & (ix < lo)
    right = beside & (ix >= hi)

    out = xp.where(above, off_a + iy * W + ix, out)
    out = xp.where(below, off_b + (iy - hi) * W + ix, out)
    out = xp.where(left, off_c + (iy - lo) * lo + ix, out)
    out = xp.where(right, off_d + (iy - lo) * (W - hi) + (ix - hi), out)
    return out


@dataclass
class Allocation:
    """Everything the frame loop touches. Allocated once; never grows."""

    schedule_name: str
    rings: list
    grid: dict                 # field -> flat array over all rings
    transient: dict            # field -> flat array, same geometry, 4 B/cell
    pool: dict                 # field -> flat array over pool blocks
    tracks: np.ndarray
    scratch: dict
    scatter_mode: str
    storage: str
    pool_blocks: int
    pool_cells_per_block: int
    max_tracks: int
    device: str = "cpu"
    pyramid: object = None     # None unless allocate(with_pyramid=True); §7.2
    _budget: dict = field(default_factory=dict)
    _resident_delta: int = 0

    def ring(self, index: int) -> RingLayout:
        return self.rings[index]

    def view(self, field_name: str, ring_index: int):
        """A ring's slice of one field. A view, not a copy -- writing to it
        writes through to the allocation. Spans allocated slots, which under
        toroidal storage exceed the ring's logical cell count."""
        r = self.rings[ring_index]
        n = r.slots if self.storage == "toroidal" else r.count
        return self.grid[field_name][r.offset:r.offset + n]

    @property
    def logical_cells(self) -> int:
        """What the report's ratios count: 745,000 for the default schedule."""
        return sum(r.count for r in self.rings)

    @property
    def allocated_slots(self) -> int:
        return sum(r.slots if self.storage == "toroidal" else r.count for r in self.rings)

    @property
    def budget(self) -> dict:
        return dict(self._budget)

    def total_bytes(self) -> int:
        """What we claim. Compare against `resident_delta` -- the gap between
        the two is the difference between a bound asserted and a bound paid."""
        return sum(self._budget.values())

    @property
    def resident_delta(self) -> int:
        """What allocating this actually cost the machine, per the OS."""
        return self._resident_delta

    def report(self) -> str:
        header = (f"schedule {self.schedule_name}, device {self.device}, "
                  f"storage {self.storage}")
        lines = [header, ""]
        for k, v in self._budget.items():
            lines.append(f"  {k:<34} {v / 1e6:>8.2f} MB")
        lines.append(f"  {'-' * 34} {'-' * 8}")
        lines.append(f"  {'TOTAL (preallocated, fixed)':<34} {self.total_bytes() / 1e6:>8.2f} MB")
        if self._resident_delta:
            lines.append(f"  {'  of which resident, per the OS':<34} "
                         f"{self._resident_delta / 1e6:>8.2f} MB")
        return "\n".join(lines)


def allocate(schedule, thresholds: dict | None = None, device: str = "cpu",
             transient_rings: int | None = None, max_tracks: int = 256,
             storage: str = "toroidal", commit_pages: bool = True,
             with_pyramid: bool = False) -> Allocation:
    """Preallocate the grid, the transient layer, the refinement pool and the
    tracked-object list. Called once at startup.

    `transient_rings` limits the transient layer to the innermost N rings.
    Default is every ring. See the note in `docs/` and the budget printout --
    this is the one line item whose size is a team decision, not a derivation.

    `with_pyramid` adds the conservative pyramid (§7.2). **Off by default, on
    purpose.** It is a stretch item and it costs 3.11 MB on the default
    schedule -- 2.73 MB of nodes plus 0.38 MB of reduction scratch -- which
    moves the preallocated total from 29.06 MB to 32.17 MB and therefore moves
    a number that is already on a slide. Switching it on is one argument and
    the budget line appears with it; doing that by default would change the
    headline from inside my directory, which is the thing the Day-0 gate
    review said not to do. §7.2's own figure of 1.24 MB is low by about half
    and is wrong for a second reason too -- see the note in gpu/pyramid.py.
    """
    xp = array_module(device)
    rings = derive_ring_layouts(schedule, storage)
    n_logical = sum(r.count for r in rings)
    n_cells = sum(r.slots for r in rings) if storage == "toroidal" else n_logical

    pool_cfg = (thresholds or {}).get("refinement_pool", {})
    blocks = pool_cfg.get("blocks", 512)
    cells_per_block = pool_cfg.get("cells_per_block", 16)

    scatter_cfg = (thresholds or {}).get("scatter", {})
    scatter_mode = scatter_cfg.get("mode", "sorted")
    max_points = scatter_cfg.get("max_points_per_frame", 150_000)

    def _size(rs):
        return sum(r.slots if storage == "toroidal" else r.count for r in rs)

    n_transient = n_cells if transient_rings is None else _size(rings[:transient_rings])

    resident_before = resident_bytes() if device == "cpu" else 0

    grid = {name: xp.zeros(n_cells, dtype=dt) for name, dt in CELL_FIELDS}
    transient = {name: xp.zeros(n_transient, dtype=dt) for name, dt in TRANSIENT_FIELDS}
    pool = {name: xp.zeros(blocks * cells_per_block, dtype=dt) for name, dt in CELL_FIELDS}
    tracks = xp.zeros(max_tracks, dtype=TRACK_DTYPE)
    scratch = (new_sorted_scratch(max_points, n_cells) if scatter_mode == "sorted"
               else new_dense_scratch(n_cells))
    pyramid = allocate_pyramid(rings) if with_pyramid else None

    # The grid and the pool hold cells, so they get the empty-cell state rather
    # than raw zeros. The pool matters as much as the grid: a block handed out
    # by `pool.acquire()` is a set of brand-new cells, and one that boots with
    # ceiling 0 is untraversable from the moment it is refined.
    initialise_cells(grid)
    initialise_cells(pool)

    # Fault every page in now rather than during frame 1. Only meaningful on
    # host memory; a cupy allocation is device-side and this does not apply.
    if commit_pages and device == "cpu":
        for group in (grid, transient, pool, scratch):
            for arr in group.values():
                commit(arr)
        commit(tracks)

    alloc = Allocation(
        schedule_name=schedule.name, rings=rings, grid=grid, transient=transient,
        pool=pool, tracks=tracks, scratch=scratch, scatter_mode=scatter_mode,
        storage=storage, pool_blocks=blocks,
        pool_cells_per_block=cells_per_block, max_tracks=max_tracks, device=device,
        pyramid=pyramid,
    )
    alloc._resident_delta = (resident_bytes() - resident_before) if device == "cpu" else 0
    alloc._budget = {
        f"grid ({n_logical:,} logical cells x {CELL_BYTES} B)": n_logical * CELL_BYTES,
        f"toroidal padding ({n_cells - n_logical:,} slots)":
            (n_cells - n_logical) * CELL_BYTES,
        f"scatter scratch ({scatter_mode})": scatter_scratch_bytes(
            scatter_mode, n_cells, max_points),
        f"transient ({n_transient:,} x {TRANSIENT_BYTES} B)": n_transient * TRANSIENT_BYTES,
        f"refinement pool ({blocks} x {cells_per_block})": blocks * cells_per_block * CELL_BYTES,
        f"tracked objects (cap {max_tracks})": max_tracks * TRACK_DTYPE.itemsize,
    }
    if pyramid is not None:
        alloc._budget[f"conservative pyramid ({NODE_BYTES} B/node)"] = \
            pyramid_bytes(rings)
        alloc._budget["pyramid reduction scratch"] = pyramid_scratch_bytes(rings)
    return alloc


def bytes_allocated(handle: Allocation) -> int:
    """Must agree with src/eval/metrics.memory_bytes() and the report table."""
    return handle.total_bytes()


def measured_bytes(handle: Allocation) -> int:
    """What the arrays actually occupy, read back from the arrays themselves.

    Kept separate from `bytes_allocated()` on purpose: the budget is what we
    claim, this is what we allocated, and the test that they match is the
    difference between a bound we assert and a bound we can demonstrate.
    """
    total = sum(a.nbytes for a in handle.grid.values())
    total += sum(a.nbytes for a in handle.transient.values())
    total += sum(a.nbytes for a in handle.pool.values())
    total += sum(a.nbytes for a in handle.scratch.values())
    return total + handle.tracks.nbytes
