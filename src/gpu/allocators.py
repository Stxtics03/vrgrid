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

from dataclasses import dataclass, field

import numpy as np
from vrgrid.cell import CELL_BYTES, CELL_FIELDS
from vrgrid.gpu.kernels import (
    new_dense_scratch,
    new_sorted_scratch,
    scatter_scratch_bytes,
)

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
    _budget: dict = field(default_factory=dict)

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
        return sum(self._budget.values())

    def report(self) -> str:
        header = (f"schedule {self.schedule_name}, device {self.device}, "
                  f"storage {self.storage}")
        lines = [header, ""]
        for k, v in self._budget.items():
            lines.append(f"  {k:<34} {v / 1e6:>8.2f} MB")
        lines.append(f"  {'-' * 34} {'-' * 8}")
        lines.append(f"  {'TOTAL (preallocated, fixed)':<34} {self.total_bytes() / 1e6:>8.2f} MB")
        return "\n".join(lines)


def allocate(schedule, thresholds: dict | None = None, device: str = "cpu",
             transient_rings: int | None = None, max_tracks: int = 256,
             storage: str = "toroidal") -> Allocation:
    """Preallocate the grid, the transient layer, the refinement pool and the
    tracked-object list. Called once at startup.

    `transient_rings` limits the transient layer to the innermost N rings.
    Default is every ring. See the note in `docs/` and the budget printout --
    this is the one line item whose size is a team decision, not a derivation.
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

    grid = {name: xp.zeros(n_cells, dtype=dt) for name, dt in CELL_FIELDS}
    transient = {name: xp.zeros(n_transient, dtype=dt) for name, dt in TRANSIENT_FIELDS}
    pool = {name: xp.zeros(blocks * cells_per_block, dtype=dt) for name, dt in CELL_FIELDS}
    tracks = xp.zeros(max_tracks, dtype=TRACK_DTYPE)
    scratch = (new_sorted_scratch(max_points) if scatter_mode == "sorted"
               else new_dense_scratch(n_cells))

    alloc = Allocation(
        schedule_name=schedule.name, rings=rings, grid=grid, transient=transient,
        pool=pool, tracks=tracks, scratch=scratch, scatter_mode=scatter_mode,
        storage=storage, pool_blocks=blocks,
        pool_cells_per_block=cells_per_block, max_tracks=max_tracks, device=device,
    )
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
