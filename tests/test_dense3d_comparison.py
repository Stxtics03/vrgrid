"""Dense-3D vs variable-grid comparison + the reverse index->(x,y,z) map. [JP]

`dense3d_debug.reverse_dense3d` is the load-bearing piece: it inverts
`gpu.baseline.DenseVoxelBaseline`'s forward indexing so the baseline's occupied
voxels can be drawn. The comparison renderer (`dense3d_comparison`) is a
standalone artifact, not part of the live dashboard.
"""

import numpy as np
import pytest

pytest.importorskip("rerun")

from vrgrid.dash.dense3d_debug import (
    forward_dense3d,
    occupied_voxel_centres,
    reverse_dense3d,
)
from vrgrid.gpu.baseline import Z_MIN_M, allocate_dense3d
from vrgrid.perception.loader import _velodyne_path, verify_sequence_exists

_HAS_00 = verify_sequence_exists("00") and _velodyne_path("00", 10).exists()
needs_data = pytest.mark.skipif(not _HAS_00, reason="KITTI seq 00 not present")

SIDE, LAYERS, CELL = 400, 160, 0.05          # footprint_m = 20


# --------------------------------------------------------------------------
# reverse mapping
# --------------------------------------------------------------------------


def test_round_trip_is_within_half_a_cell():
    rng = np.random.default_rng(0)
    x = rng.uniform(-9.5, 9.5, 50_000)
    y = rng.uniform(-9.5, 9.5, 50_000)
    z = rng.uniform(Z_MIN_M + 0.1, Z_MIN_M + LAYERS * CELL - 0.1, 50_000)

    flat, keep = forward_dense3d(x, y, z, SIDE, LAYERS, CELL)
    assert keep.all()
    rx, ry, rz, *_ = reverse_dense3d(flat, SIDE, LAYERS, CELL)

    assert np.abs(rx - x).max() <= CELL / 2 + 1e-9
    assert np.abs(ry - y).max() <= CELL / 2 + 1e-9
    assert np.abs(rz - z).max() <= CELL / 2 + 1e-9
    # exact integer round trip
    assert np.array_equal(forward_dense3d(rx, ry, rz, SIDE, LAYERS, CELL)[0], flat)


def test_forward_drops_outside_the_footprint_without_wrapping():
    b = (SIDE // 2) * CELL                    # +boundary, 10.0 m
    x = np.array([b, b + 0.01, 150.0, -1e6, -b])
    y = np.zeros(5)
    z = np.zeros(5)
    flat, keep = forward_dense3d(x, y, z, SIDE, LAYERS, CELL)
    # exact +b is out (half-open), beyond is out, exact -b is in
    assert keep.tolist() == [False, False, False, False, True]
    assert (flat[~keep] == -1).all()          # -1, never a wrapped index


def test_reverse_raises_on_out_of_range_index():
    units = SIDE * SIDE * LAYERS
    for bad in (units, units + 10, -1):
        with pytest.raises(ValueError, match="not occupied voxels"):
            reverse_dense3d(np.array([bad]), SIDE, LAYERS, CELL)


@needs_data
def test_every_occupied_voxel_reverses_to_where_a_real_point_was():
    from vrgrid.perception.loader import load_velodyne_scan

    scan = load_velodyne_scan(_velodyne_path("00", 10))
    x, y, z = scan[:, 0], scan[:, 1], scan[:, 2]

    d = allocate_dense3d(footprint_m=20.0, allow_unsafe=True)
    d.ingest(x, y, z)
    centres = occupied_voxel_centres(d)
    assert len(centres) > 1000

    # independent binning of the same points must give the identical cell set
    keep = forward_dense3d(x, y, z, SIDE, LAYERS, CELL)[1]
    gx = np.floor(x[keep] / CELL).astype(np.int64) + SIDE // 2
    gy = np.floor(y[keep] / CELL).astype(np.int64) + SIDE // 2
    gz = np.floor((z[keep] - Z_MIN_M) / CELL).astype(np.int64)
    gt = np.unique((gy * SIDE + gx) * LAYERS + gz)
    occ = np.flatnonzero(d.arrays["occupied"])
    assert np.array_equal(np.sort(occ), gt)

    # centres finite and inside the footprint
    assert np.isfinite(centres).all()
    assert (np.abs(centres[:, :2]) <= (SIDE // 2) * CELL).all()


# --------------------------------------------------------------------------
# the comparison renderer
# --------------------------------------------------------------------------


@needs_data
def test_comparison_renders_both_trees_and_dense_needs_more_boxes(tmp_path):
    from vrgrid.dash.dense3d_comparison import COMPARE_OFFSET_M, run

    last = run("00", n_frames=8, footprint_m=20.0,
               save_path=str(tmp_path / "cmp.rrd"))

    assert last["dense_boxes"] > last["var_boxes"] > 0
    assert last["frame"] == 7
    # a real .rrd was written with both entity trees
    blob = (tmp_path / "cmp.rrd").read_bytes()
    assert b"world/map/occupied" in blob and b"dense3d/occupied" in blob
    assert COMPARE_OFFSET_M == 50.0
