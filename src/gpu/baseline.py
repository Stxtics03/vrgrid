"""Baselines that actually allocate. [Shrestha]

The problem statement asks us to **demonstrate** the memory reduction, not to
calculate it. A number in a table is an assertion; a dense voxel grid sitting
in RAM beside ours, both counters ticking on the same screen, is a
demonstration. That is the whole reason this file exists rather than a row in
`scripts/memory_table.py`, which computes the same figures and proves nothing.

**The trap, and it is the entire point of the file.** `np.zeros` does not
allocate memory. It asks the kernel for a mapping, and the kernel hands back
copy-on-write zero pages that cost nothing until something writes to them.
Measured here: `np.zeros(2_560_000_000, np.uint8)` moves resident set size by
**0.0 MB**. A baseline built the obvious way would put 0 MB on the dashboard
next to our 27.86 MB and the demo would be worse than no demo -- we would be
claiming a 286x reduction over something visibly costing nothing, in front of
judges, with the evidence on screen. So every allocation here is TOUCHED, one
write per page, and `resident_bytes()` reads back what the OS thinks we own.
Touching 2.56 GB costs 0.23 s, once, at startup.

What these are: honest *stubs*. They allocate the storage a dense 3D voxel
grid or a uniform 5 cm 2.5D grid would need for the same 200 x 200 m footprint,
and they ingest a scan into it. They do not fuse, filter, or maintain
occupancy over time. That is a fair comparison for the memory claim, which is
about storage, and it would not be a fair comparison for accuracy -- do not let
the demo imply otherwise. The ratios are pure cell-count ratios and are
invariant to bytes-per-cell (math §11).

Numbers, at 5 cm over 200 x 200 x 8 m:

    dense 3D voxel   2.56e9 voxels x 1 B   = 2.56 GB
    uniform 2.5D     16.0e6 cells  x 12 B  = 192 MB
    ours (4-ring)    745,000 cells x 12 B  = 8.94 MB
"""

from dataclasses import dataclass

import numpy as np
from vrgrid.cell import CELL_BYTES, CELL_FIELDS

# Allocation mechanics live in allocators.py, which owns allocation; this file
# owns the comparison. Re-exported so the baseline reads as one story.
from vrgrid.gpu.allocators import (  # noqa: F401
    PAGE_BYTES,
    SAFETY_FRACTION,
    available_bytes,
    commit,
    resident_bytes,
)

# 200 x 200 m footprint, -2 to +6 m vertical extent, 5 cm base cell. These
# match scripts/memory_table.py and math §11 -- if they ever disagree, the
# table is quoting a baseline we do not allocate, which is the one thing the
# gate review said we must never do.
FOOTPRINT_M = 200.0
VERTICAL_M = 8.0
Z_MIN_M = -2.0
BASE_CELL_M = 0.05

def dense3d_voxels(footprint_m: float = FOOTPRINT_M, vertical_m: float = VERTICAL_M,
                   cell_m: float = BASE_CELL_M) -> int:
    """math §11: (200/0.05)^2 x (8/0.05) = 2.56e9."""
    side = round(footprint_m / cell_m)
    return side * side * round(vertical_m / cell_m)


def uniform25d_cells(footprint_m: float = FOOTPRINT_M,
                     cell_m: float = BASE_CELL_M) -> int:
    """math §11: (200/0.05)^2 = 16.0e6."""
    side = round(footprint_m / cell_m)
    return side * side


@dataclass
class Counters:
    """One row of the live memory display. `claimed` is what the slide says,
    `resident` is what the machine says, and showing both is the point."""

    name: str
    units: int          # voxels or cells
    claimed_bytes: int  # what the report table quotes
    resident_bytes: int # process RSS attributable to this allocation

    def __str__(self) -> str:
        def mb(b):
            return f"{b / 1e9:.2f} GB" if b >= 1e9 else f"{b / 1e6:.2f} MB"
        return (f"{self.name:<28} {self.units:>14,}  claimed {mb(self.claimed_bytes):>9}"
                f"  resident {mb(self.resident_bytes):>9}")


class Baseline:
    """A baseline map that occupies the memory it claims to occupy."""

    def __init__(self, name: str, arrays: dict, units: int, side: int,
                 cell_m: float, layers: int = 1):
        self.name = name
        self.arrays = arrays
        self.units = units
        self.side = side
        self.cell_m = cell_m
        self.layers = layers
        self._resident_delta = 0

    @property
    def claimed_bytes(self) -> int:
        return sum(a.nbytes for a in self.arrays.values())

    def counters(self) -> Counters:
        return Counters(self.name, self.units, self.claimed_bytes, self._resident_delta)

    def ratio_against(self, ours_bytes: int) -> float:
        return self.claimed_bytes / ours_bytes

    def _xy_index(self, x_m, y_m):
        """Footprint-centred cell coordinates. Returns -1 outside, on the same
        never-use--1-as-an-index rule as `annulus_index()`."""
        half = self.side // 2
        ix = np.floor(np.asarray(x_m) / self.cell_m).astype(np.int64) + half
        iy = np.floor(np.asarray(y_m) / self.cell_m).astype(np.int64) + half
        inside = (ix >= 0) & (ix < self.side) & (iy >= 0) & (iy < self.side)
        return np.where(inside, iy * self.side + ix, -1)

    def ingest(self, x_m, y_m, z_m) -> int:
        raise NotImplementedError


class DenseVoxelBaseline(Baseline):
    """Dense 3D occupancy voxels, 1 B each, over the full vertical extent.

    This is the comparison the problem statement asks for, and the one that
    makes the foveation argument: the grid is 99.87% empty at 50 m (math §1.3),
    and it pays for every empty voxel anyway.
    """

    def ingest(self, x_m, y_m, z_m) -> int:
        """Mark occupied voxels. Returns how many returns landed in the map."""
        flat_xy = self._xy_index(x_m, y_m)
        iz = np.floor((np.asarray(z_m) - Z_MIN_M) / self.cell_m).astype(np.int64)
        keep = (flat_xy >= 0) & (iz >= 0) & (iz < self.layers)
        idx = flat_xy[keep] * self.layers + iz[keep]
        self.arrays["occupied"][idx] = 1
        return int(keep.sum())


class UniformGridBaseline(Baseline):
    """Uniform 5 cm 2.5D grid using the same frozen 12 B cell as ours.

    The honest headline comparison: it isolates our contribution from the
    3D -> 2.5D reduction, which anyone could have made. Sharing `CELL_FIELDS`
    is deliberate -- if the cell struct changes, this baseline changes with it
    and the ratio cannot drift away from the code.
    """

    def ingest(self, x_m, y_m, z_m) -> int:
        """Last return wins on height -- no fusion, this is a storage stub.

        The observation count is NOT last-wins: `count[idx] += 1` through fancy
        indexing is buffered, so two returns in one cell increment it once and
        the baseline silently under-counts. Same trap as a float atomic, same
        fix -- an unbuffered accumulate. Cheap here because it is sized by
        touched cells, and unlike `scatter_sorted` this is not on our frame
        loop, so allocating in it costs the baseline and not our bound.
        """
        flat_xy = self._xy_index(x_m, y_m)
        keep = flat_xy >= 0
        idx = flat_xy[keep]
        self.arrays["ground_height"][idx] = np.clip(
            np.rint(np.asarray(z_m)[keep] * 100.0), -32768, 32767).astype(np.int16)

        cnt = self.arrays["obs_count"]
        touched, added = np.unique(idx, return_counts=True)
        cnt[touched] = np.minimum(cnt[touched].astype(np.int32) + added, 255)
        return int(keep.sum())


def _guard(nbytes: int, name: str, allow_unsafe: bool) -> None:
    avail = available_bytes()
    if not allow_unsafe and nbytes > SAFETY_FRACTION * avail:
        raise MemoryError(
            f"{name} needs {nbytes / 1e9:.2f} GB and only {avail / 1e9:.2f} GB is "
            f"available. Refusing rather than risking an OOM kill mid-demo. Pass "
            f"allow_unsafe=True, or shrink footprint_m, if you know better.")


def allocate_dense3d(footprint_m: float = FOOTPRINT_M, vertical_m: float = VERTICAL_M,
                     cell_m: float = BASE_CELL_M, allow_unsafe: bool = False
                     ) -> DenseVoxelBaseline:
    """Really allocate, and really commit, the dense 3D voxel baseline.

    Default is 2.56 GB and takes about a quarter of a second to fault in.
    """
    voxels = dense3d_voxels(footprint_m, vertical_m, cell_m)
    _guard(voxels, "dense 3D voxel baseline", allow_unsafe)

    before = resident_bytes()
    arrays = {"occupied": commit(np.zeros(voxels, np.uint8))}
    b = DenseVoxelBaseline("dense 3D voxel, 1 B", arrays, voxels,
                           round(footprint_m / cell_m), cell_m,
                           layers=round(vertical_m / cell_m))
    b._resident_delta = resident_bytes() - before
    return b


def allocate_uniform25d(footprint_m: float = FOOTPRINT_M, cell_m: float = BASE_CELL_M,
                        allow_unsafe: bool = False) -> UniformGridBaseline:
    """Really allocate the uniform 5 cm 2.5D baseline, same 12 B cell as ours."""
    cells = uniform25d_cells(footprint_m, cell_m)
    _guard(cells * CELL_BYTES, "uniform 2.5D baseline", allow_unsafe)

    before = resident_bytes()
    arrays = {name: commit(np.zeros(cells, dtype=dt)) for name, dt in CELL_FIELDS}
    b = UniformGridBaseline(f"uniform 5 cm 2.5D, {CELL_BYTES} B", arrays, cells,
                            round(footprint_m / cell_m), cell_m)
    b._resident_delta = resident_bytes() - before
    return b
