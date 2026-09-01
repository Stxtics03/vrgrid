"""Residual-image motion segmentation. [JP]

The synthetic test pins the mechanism (a block that moves between two otherwise
identical frames must raise a residual spike where it was and where it is). The
`@needs_data` tests report the measured agreement with GT `is_moving()` on two
named SemanticKITTI frames -- the actual overlap numbers are in the docstrings,
not tuned thresholds, because a geometric single-pair MOS is a weak-but-real
signal and the point of the test is to keep it from regressing, not to claim it
is good.
"""

import numpy as np
import pytest
from vrgrid.perception.loader import _label_path, _velodyne_path, verify_sequence_exists
from vrgrid.perception.residual_mos import (
    RESIDUAL_THRESHOLD_M,
    ResidualMOS,
    residual_motion_from_poses,
    residual_motion_mask,
)

_HAS_00 = verify_sequence_exists("00") and _velodyne_path("00", 10).exists()
_HAS_07 = verify_sequence_exists("07") and _velodyne_path("07", 674).exists()


# --------------------------------------------------------------------------
# synthetic -- identity ego-motion, one block moves
# --------------------------------------------------------------------------


# geometry kept near the horizon so nothing clamps to an edge ring: the HDL-64E
# FOV is [-24.8, +2.0] deg, and a block at x=15, z in [-1.6, -0.6] sits at
# ~ -3 deg elevation.
def _wall(n=6000, x=30.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([
        np.full(n, x), rng.uniform(-10, 10, n), rng.uniform(-2.0, 0.0, n),
        np.full(n, 0.3),
    ])


def _block(cx, cy, n=1500, seed=1):
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(cx - 0.6, cx + 0.6, n),
        rng.uniform(cy - 0.6, cy + 0.6, n),
        rng.uniform(-1.6, -0.6, n),
        np.full(n, 0.3),
    ])


def test_a_moving_block_raises_a_residual_spike_where_it_moved():
    prev = np.vstack([_wall(), _block(15.0, 0.0)])
    curr = np.vstack([_wall(), _block(15.0, 4.0)])          # block slid +4 m in y

    res = residual_motion_mask(prev, curr, np.eye(4))
    assert isinstance(res, ResidualMOS)
    assert res.threshold_m == RESIDUAL_THRESHOLD_M

    # the wall is unchanged -> almost no residual there
    wall_only = residual_motion_mask(_wall(), _wall(seed=2), np.eye(4))
    assert wall_only.pixel_mask.sum() < 20

    # with the block, motion pixels light up and stay LOCALISED -- well under
    # half the covered pixels, not the whole wall
    assert 20 < res.pixel_mask.sum() < 0.5 * res.valid.sum()
    flagged = curr[res.point_mask]
    assert 100 < len(flagged) < 0.5 * len(curr)
    # the block moved through +y, and the disocclusion it leaves is also at +y
    # (the wall behind it) -- so the flagged returns sit on the +y side, never
    # out on the far -y wall the block never touched
    assert (flagged[:, 1] > -2.0).mean() > 0.9
    # both the block's positions are represented (new at y~4, revealed gap at y~0)
    assert flagged[:, 1].max() > 2.5


def test_no_motion_no_mask():
    w = _wall()
    res = residual_motion_mask(w, w.copy(), np.eye(4))
    assert res.pixel_mask.sum() < 20
    assert res.point_mask.sum() < 50


def test_residual_is_nan_where_not_comparable_and_finite_where_it_is():
    res = residual_motion_mask(np.vstack([_wall(), _block(10, 0)]),
                               np.vstack([_wall(), _block(10, 4)]), np.eye(4))
    assert np.isnan(res.residual[~res.valid]).all()
    assert np.isfinite(res.residual[res.valid]).all()
    assert (res.residual[res.valid] >= 0).all()


# --------------------------------------------------------------------------
# real SemanticKITTI frames -- measured agreement with GT is_moving()
# --------------------------------------------------------------------------


def _overlap(seq, frame):
    from vrgrid.perception import range_image, semantics
    from vrgrid.perception.loader import load_gt_poses, load_labels, load_velodyne_scan

    poses = load_gt_poses(seq)
    prev = load_velodyne_scan(_velodyne_path(seq, frame - 1))
    curr = load_velodyne_scan(_velodyne_path(seq, frame))
    gt = semantics.is_moving(load_labels(_label_path(seq, frame)))
    res = residual_motion_from_poses(prev, poses[frame - 1], curr, poses[frame], seq)

    m = res.point_mask
    tp, fp, fn = int((m & gt).sum()), int((m & ~gt).sum()), int((~m & gt).sum())
    point = dict(
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
        iou=tp / (tp + fp + fn) if tp + fp + fn else 0.0,
    )

    _, inv = range_image.project(curr)
    gt_px = np.zeros(res.pixel_mask.shape, bool)
    filled = inv >= 0
    gt_px[filled] = gt[inv[filled]]
    pm = res.pixel_mask
    tpx, fpx, fnx = int((pm & gt_px).sum()), int((pm & ~gt_px).sum()), int((~pm & gt_px).sum())
    pixel = dict(
        precision=tpx / (tpx + fpx) if tpx + fpx else 0.0,
        recall=tpx / (tpx + fnx) if tpx + fnx else 0.0,
        iou=tpx / (tpx + fpx + fnx) if tpx + fpx + fnx else 0.0,
    )
    return point, pixel


@pytest.mark.skipif(not _HAS_07, reason="KITTI seq 07 not present -- set VRGRID_DATA_ROOT")
def test_seq07_frame674_recovers_the_fast_vehicle():
    """Frame 674 has a fast-approaching vehicle (~3300 GT moving points).
    Measured this run: pixel IoU ~0.24 (P~0.40, R~0.39), point IoU ~0.20
    (P~0.32, R~0.35). A geometric single-pair MOS on a 64x512 image -- a real
    signal, well short of a learned MOS head. The bar here is only that it does
    not regress below IoU 0.15."""
    point, pixel = _overlap("07", 674)
    assert pixel["iou"] > 0.15, pixel
    assert point["iou"] > 0.12, point
    assert point["recall"] > 0.25 and pixel["recall"] > 0.25


@pytest.mark.skipif(not _HAS_00, reason="KITTI seq 00 not present -- set VRGRID_DATA_ROOT")
def test_seq00_frame10_finds_the_motorcyclist_but_at_poor_precision():
    """Frame 10's only movers are a motorcyclist at ~29 m and a 2-point
    pedestrian, while the sensor drives fast through dense urban structure.
    Measured this run: point recall ~0.89 (it does flag the motorcyclist) but
    precision ~0.005 -- ~11,500 false positives at dis-occlusion boundaries the
    sensor's own motion reveals. This is the documented failure mode of
    single-pair residual MOS; the test asserts the signal is present (recall),
    not that it is clean."""
    point, _pixel = _overlap("00", 10)
    assert point["recall"] > 0.5, point          # the motorcyclist is found
    assert point["precision"] < 0.1, point        # but the frame is full of FPs
