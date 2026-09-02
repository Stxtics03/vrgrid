"""Reverse index -> (x, y, z) for `gpu.baseline.DenseVoxelBaseline`. [JP]

`DenseVoxelBaseline` only offers the forward path (points -> occupied voxel
indices). To *draw* the baseline's occupied voxels we need the inverse: a flat
index back to the cell centre it stands for. This module derives that inverse
from the forward logic in `src/gpu/baseline.py` and nothing else -- it does not
import or modify that file beyond reading its constants.

Verification (100% round-trip on real KITTI, edge cases, out-of-range guard):
run this module as a script -- `python -m vrgrid.dash.dense3d_debug`.

FORWARD  (verbatim from `Baseline._xy_index` + `DenseVoxelBaseline.ingest`):

    half    = side // 2                       # side is even -- allocate() asserts it
    ix      = floor(x / cell_m) + half
    iy      = floor(y / cell_m) + half
    inside  = 0 <= ix < side and 0 <= iy < side       # else _xy_index() -> -1
    iz      = floor((z - Z_MIN_M) / cell_m)
    keep    = inside and 0 <= iz < layers             # else ingest() drops the point
    flat3d  = (iy * side + ix) * layers + iz          # only kept indices are stored

REVERSE  (this module):

    iz      = flat3d %  layers
    flat_xy = flat3d // layers
    ix      = flat_xy %  side
    iy      = flat_xy // side
    x_centre = (ix - half + 0.5) * cell_m
    y_centre = (iy - half + 0.5) * cell_m
    z_centre = Z_MIN_M + (iz + 0.5) * cell_m
"""

import numpy as np
from vrgrid.gpu.baseline import Z_MIN_M


def forward_dense3d(x, y, z, side: int, layers: int, cell_m: float,
                    z_min_m: float = Z_MIN_M):
    """Re-implementation of the forward indexing, self-contained so the round
    trip can be checked without a real allocation. Returns `(flat3d, keep)`
    where `flat3d` is -1 wherever `keep` is False."""
    half = side // 2
    ix = np.floor(np.asarray(x) / cell_m).astype(np.int64) + half
    iy = np.floor(np.asarray(y) / cell_m).astype(np.int64) + half
    inside = (ix >= 0) & (ix < side) & (iy >= 0) & (iy < side)
    flat_xy = np.where(inside, iy * side + ix, -1)
    iz = np.floor((np.asarray(z) - z_min_m) / cell_m).astype(np.int64)
    keep = (flat_xy >= 0) & (iz >= 0) & (iz < layers)
    flat3d = np.where(keep, flat_xy * layers + iz, -1)
    return flat3d, keep


def reverse_dense3d(flat3d, side: int, layers: int, cell_m: float,
                    z_min_m: float = Z_MIN_M):
    """Flat occupied-voxel index -> `(x, y, z, ix, iy, iz)`, cell centre in metres.

    Raises `ValueError` on an index outside `[0, side*side*layers)` -- a genuine
    occupied voxel is always in range, so this only fires on caller error, and
    it must never silently return an out-of-footprint coordinate.
    """
    flat3d = np.asarray(flat3d, dtype=np.int64)
    units = side * side * layers
    bad = (flat3d < 0) | (flat3d >= units)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} index(es) outside [0, {units}); "
                         "not occupied voxels")
    half = side // 2
    iz = flat3d % layers
    flat_xy = flat3d // layers
    ix = flat_xy % side
    iy = flat_xy // side
    x = (ix - half + 0.5) * cell_m
    y = (iy - half + 0.5) * cell_m
    z = z_min_m + (iz + 0.5) * cell_m
    return x, y, z, ix, iy, iz


def occupied_voxel_centres(baseline):
    """`(x, y, z)` cell centres for every occupied voxel of a `DenseVoxelBaseline`.

    Convenience wrapper: pulls `side / layers / cell_m` off the instance so a
    caller cannot pass mismatched parameters."""
    occ = np.flatnonzero(baseline.arrays["occupied"])
    x, y, z, *_ = reverse_dense3d(occ, baseline.side, baseline.layers, baseline.cell_m)
    return np.stack([x, y, z], axis=1)


# --------------------------------------------------------------------------
# self-test -- `python -m vrgrid.dash.dense3d_debug`
# --------------------------------------------------------------------------
def _selftest():
    import os

    from vrgrid.gpu.baseline import allocate_dense3d
    from vrgrid.perception import loader

    os.environ.setdefault("VRGRID_DATA_ROOT", "C:/KITTI/dataset")
    d = allocate_dense3d(footprint_m=20.0, allow_unsafe=True)
    S, L, C = d.side, d.layers, d.cell_m
    print(f"side={S} layers={L} cell_m={C} units={d.units:,}  half={S // 2} (even: {S % 2 == 0})")

    scan = loader.load_velodyne_scan(loader._velodyne_path("00", 10))
    X, Y, Zc = (scan[:, 0].astype(np.float64), scan[:, 1].astype(np.float64),
                scan[:, 2].astype(np.float64))
    f, k = forward_dense3d(X, Y, Zc, S, L, C)
    rx, ry, rz, *_ = reverse_dense3d(f[k], S, L, C)
    err = np.maximum.reduce([np.abs(rx - X[k]), np.abs(ry - Y[k]), np.abs(rz - Zc[k])])
    f2, _ = forward_dense3d(rx, ry, rz, S, L, C)
    print(f"round trip on {int(k.sum()):,} real pts: max per-axis err {err.max() * 100:.3f} cm "
          f"(cell/2 = {C * 50:.1f}); forward(reverse(idx))==idx: {np.array_equal(f2, f[k])}")

    n = d.ingest(X, Y, Zc)
    occ = np.flatnonzero(d.arrays["occupied"])
    gx = np.floor(X[k] / C).astype(np.int64) + S // 2
    gy = np.floor(Y[k] / C).astype(np.int64) + S // 2
    gz = np.floor((Zc[k] - Z_MIN_M) / C).astype(np.int64)
    gt = np.unique((gy * S + gx) * L + gz)
    print(f"ingested {n:,} pts -> {len(occ):,} voxels; reverse-set == independent binning: "
          f"{np.array_equal(np.sort(occ), gt)}")

    for bad in (d.units, -1):
        try:
            reverse_dense3d(np.array([bad]), S, L, C)
            print(f"  idx {bad}: NO RAISE (bug)")
        except ValueError:
            print(f"  idx {bad}: ValueError (correct)")


if __name__ == "__main__":
    _selftest()
