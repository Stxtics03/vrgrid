"""Instance clustering of moving points -- range-image connected components. [JP]

The synthetic tests pin the algorithm (two blobs -> two instances, a range gap
splits, azimuth wraps). The `@needs_data` tests check it separates real
SemanticKITTI moving objects into the right count at roughly the right place,
against the ground-truth instance ids in the `.label` upper bits.
"""

import numpy as np
import pytest
from vrgrid.perception.instances import Instance, cluster_moving
from vrgrid.perception.loader import _label_path, _velodyne_path, verify_sequence_exists

H, W = 64, 512

_HAS_00 = verify_sequence_exists("00") and _velodyne_path("00", 10).exists()
_HAS_07 = verify_sequence_exists("07") and _velodyne_path("07", 30).exists()


def _blank_ri():
    ri = np.full((H, W, 5), np.nan, dtype=np.float32)
    inv = np.full((H, W), -1, dtype=np.int32)
    return ri, inv


def _put_blob(ri, inv, moving, v0, u0, dv, du, rng_m, next_idx):
    """Fill a (dv x du) patch at (v0, u0) with returns at range `rng_m`."""
    i = next_idx
    for v in range(v0, v0 + dv):
        for u in range(u0, u0 + du):
            uu = u % W
            ri[v, uu, 0] = rng_m
            ri[v, uu, 1:4] = (rng_m, 0.1 * (v - v0), 0.1 * (u - u0))  # arbitrary xyz
            inv[v, uu] = i
            moving[i] = True
            i += 1
    return i


# --------------------------------------------------------------------------
# synthetic
# --------------------------------------------------------------------------


def test_two_separated_blobs_become_two_instances():
    ri, inv = _blank_ri()
    moving = np.zeros(4096, dtype=bool)
    n = _put_blob(ri, inv, moving, 20, 100, 6, 6, 10.0, 0)
    _put_blob(ri, inv, moving, 20, 300, 6, 6, 25.0, n)

    insts = cluster_moving(ri, inv, moving)
    assert len(insts) == 2
    assert all(isinstance(c, Instance) for c in insts)
    assert [c.label for c in insts] == [0, 1]
    # largest first, and the two are at their two ranges
    ranges = sorted(round(c.range_m) for c in insts)
    assert ranges == [10, 25]
    # point sets are disjoint and cover every moving pixel
    idx = np.concatenate([c.point_indices for c in insts])
    assert len(idx) == len(set(idx.tolist())) == int(moving.sum())


def test_a_range_gap_splits_one_patch_in_two():
    ri, inv = _blank_ri()
    moving = np.zeros(4096, dtype=bool)
    # one contiguous patch in the image, but the right half is 3 m farther
    n = _put_blob(ri, inv, moving, 30, 200, 5, 5, 12.0, 0)
    _put_blob(ri, inv, moving, 30, 205, 5, 5, 15.0, n)   # touches, +3 m

    insts = cluster_moving(ri, inv, moving)
    assert len(insts) == 2, "the 3 m step exceeds RANGE_TOL_M -> two instances"


def test_one_patch_within_tolerance_stays_one_instance():
    ri, inv = _blank_ri()
    moving = np.zeros(4096, dtype=bool)
    n = _put_blob(ri, inv, moving, 30, 200, 5, 5, 12.0, 0)
    _put_blob(ri, inv, moving, 30, 205, 5, 5, 12.3, n)   # +0.3 m, under tol

    assert len(cluster_moving(ri, inv, moving)) == 1


def test_clusters_wrap_across_the_azimuth_seam():
    ri, inv = _blank_ri()
    moving = np.zeros(4096, dtype=bool)
    # a blob straddling column 0 / column W-1
    _put_blob(ri, inv, moving, 25, W - 3, 4, 6, 8.0, 0)  # u = 509,510,511,0,1,2

    insts = cluster_moving(ri, inv, moving)
    assert len(insts) == 1, "column 0 and column W-1 are neighbours"


def test_no_moving_points_gives_no_instances():
    ri, inv = _blank_ri()
    assert cluster_moving(ri, inv, np.zeros(10, dtype=bool)) == []


def test_sub_threshold_specks_are_dropped():
    ri, inv = _blank_ri()
    moving = np.zeros(4096, dtype=bool)
    _put_blob(ri, inv, moving, 10, 10, 1, 2, 5.0, 0)      # 2 px -- noise
    _put_blob(ri, inv, moving, 30, 300, 6, 6, 20.0, 100)  # 36 px -- real

    insts = cluster_moving(ri, inv, moving)
    assert len(insts) == 1 and insts[0].n_points == 36


# --------------------------------------------------------------------------
# real SemanticKITTI frames
# --------------------------------------------------------------------------


def _real(seq, frame):
    from vrgrid.perception import range_image, semantics
    from vrgrid.perception.loader import load_labels, load_velodyne_scan

    pts = load_velodyne_scan(_velodyne_path(seq, frame))
    raw = load_labels(_label_path(seq, frame))
    moving = semantics.is_moving(raw)
    ri, inv = range_image.project(pts)
    return pts, raw, moving, ri, inv


@pytest.mark.skipif(not _HAS_07, reason="KITTI seq 07 not present -- set VRGRID_DATA_ROOT")
def test_seq07_frame30_splits_the_two_pedestrians():
    """Two GT moving-person instances (181 pts near-left, 22 pts far-right).
    The clusterer must return exactly two, at the two GT centroids -- not one
    blob spanning 30 m of the scene."""
    pts, raw, moving, ri, inv = _real("07", 30)
    insts = cluster_moving(ri, inv, moving)

    # GT: raw id 254 (person), two distinct instance ids in the upper bits
    gt_inst = (raw >> 16)[moving & ((raw & 0xFFFF) == 254)]
    assert len(set(gt_inst.tolist())) == 2

    assert len(insts) == 2, f"expected 2 instances, got {len(insts)}"
    got = sorted((round(c.centroid[0], 1), round(c.centroid[1], 1)) for c in insts)
    # near pedestrian ~ (2.4, -13.2), far pedestrian ~ (27.2, -30.1)
    assert got[0] == pytest.approx((2.4, -13.2), abs=1.0)
    assert got[1] == pytest.approx((27.2, -30.1), abs=1.5)
    # the two instances are well apart, not a split of one object
    c0, c1 = insts[0].centroid, insts[1].centroid
    assert np.linalg.norm(c0[:2] - c1[:2]) > 20.0


@pytest.mark.skipif(not _HAS_07, reason="KITTI seq 07 not present -- set VRGRID_DATA_ROOT")
def test_seq07_frame628_splits_pedestrian_from_the_large_vehicle():
    """A moving lorry (~1500 pts, right) and a pedestrian (~38 pts, left-behind).
    Different sizes, ~30 m apart -- must not merge."""
    pts, raw, moving, ri, inv = _real("07", 628)
    insts = cluster_moving(ri, inv, moving)
    assert len(insts) == 2

    big, small = insts[0], insts[1]
    assert big.n_points > 5 * small.n_points                 # the lorry dominates
    assert big.centroid[0] > 5 and big.centroid[1] > 3       # ahead-right
    assert small.centroid[0] < -8 and small.centroid[1] < -5  # behind-left
    assert big.extent.max() > 2.0 and small.extent.max() < 1.5


@pytest.mark.skipif(not _HAS_00, reason="KITTI seq 00 not present -- set VRGRID_DATA_ROOT")
def test_seq00_frame10_recovers_the_motorcyclist():
    """The textbook ghost-toggle frame. GT has a motorcyclist (64 pts, ahead)
    and a pedestrian -- but the pedestrian is only *2* points at 20 m, below
    what a 64x512 range image can resolve as an instance, so the honest result
    is one instance (the motorcyclist) at ~(29, 1.4)."""
    pts, raw, moving, ri, inv = _real("00", 10)
    insts = cluster_moving(ri, inv, moving)

    assert len(insts) == 1
    m = insts[0]
    assert m.centroid[:2] == pytest.approx((29.0, 1.4), abs=1.5)
    assert 25 < m.range_m < 33
    # bounding box is object-sized, not scene-sized
    assert m.extent.max() < 2.5
