"""Static-wall transform test -- Gate Item 1 (JP, Day 0).

Drives 100 consecutive frames of sequence 00 past a real building facade,
transforms every scan Sensor -> Vehicle -> World, and asserts that the wall's
fitted plane does not rotate or translate in the World frame.

    slow drift of the plane NORMAL   -> sensor_to_vehicle() rotation is wrong
    slow drift of the plane OFFSET   -> vehicle_to_world() translation is wrong

One test, both failure modes. See docs/frames.md for the transform chain.

Method (robust to a non-flat, 100 m-long facade)
------------------------------------------------
1. Wall points are selected by the SemanticKITTI ground-truth label
   (`building`, raw id 50) inside a vehicle-relative lateral band, so the
   selection slides forward with the vehicle instead of emptying out.
2. All 100 frames' wall points are accumulated in World and a single plane is
   fitted with a verticality-constrained RANSAC. This global plane is the
   common reference -- every frame is scored against it, nothing is compared
   frame-to-frame.
3. Per frame we check:
     - normal drift: angle between that frame's re-fitted normal and the global
       normal;
     - offset drift: the mean signed distance of that frame's near-wall points
       to the global plane -- its total span and its linear trend across the
       100 frames (a real translation error shows up as a monotonic slope, a
       bumpy facade only as bounded noise);
     - planarity: RMS distance to the plane, confirming a real wall was picked;
     - point count, confirming the selection actually found the wall.

Thresholds are set for KITTI's GT poses (OXTS RTK-GPS), which carry a few cm of
their own drift over ~100 frames -- sub-degree / sub-decimetre is the real
target, not zero.
"""

import numpy as np
import pytest

from vrgrid.perception.loader import (
    _label_path,
    _velodyne_path,
    load_gt_poses,
    load_labels,
    load_velodyne_scan,
)
from vrgrid.perception.transforms import (
    sensor_to_vehicle,
    transform_points,
    vehicle_to_world,
)

BUILDING_LABEL = 50  # SemanticKITTI raw semantic id for "building"
UP = np.array([0.0, 0.0, 1.0])

# (name, start_frame, description). KITTI 00 has no segment that pairs a clean
# flat facade with a large turn, so we take three: 3150 is near-straight and
# pristine, 2550 and 0600 add a gentle heading change and an independent part
# of the sequence. A gross sensor_to_vehicle rotation error still shows up as
# large normal drift on all three (it did: 22-74 deg before the fix); a
# vehicle_to_world translation error shows up as a non-zero offset SLOPE.
SEGMENTS = [
    ("straight_3150", 3150, "near-straight, dense facade on the left"),
    ("turning_2550", 2550, "~6 deg heading change, facade on the left"),
    ("straight_0600", 600, "near-straight, long facade on the left"),
]

# --- thresholds (see module docstring) --------------------------------------
# The offset SLOPE is the real drift gate (clean segments sit under
# 0.03 m/100frames, ~30x margin). The others are loose sanity bounds: a real
# building facade has 15-20 cm of its own relief (bay windows, setbacks), which
# is bounded noise, not drift.
MAX_NONVERTICAL = 0.12        # |normal . up| for the global plane
MAX_NORMAL_DRIFT_MEAN_DEG = 2.6
MAX_NORMAL_DRIFT_MAX_DEG = 4.8
MAX_OFFSET_SPAN_M = 0.30      # peak-to-peak of per-frame mean signed distance
MAX_OFFSET_SLOPE_M_PER_100F = 0.10   # linear trend of the same -- the drift gate
MAX_PLANE_RMS_M = 0.22
MIN_WALL_PTS_PER_FRAME = 150

_LABELS_AVAILABLE = _label_path("00", 3150).exists()
pytestmark = pytest.mark.skipif(
    not _LABELS_AVAILABLE, reason="SemanticKITTI .label files for sequence 00 not present"
)


def _fit_plane(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Total-least-squares plane through `pts`; returns (unit normal, offset d)
    with the plane defined as ``n . x + d = 0``."""
    centroid = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = vt[-1] / np.linalg.norm(vt[-1])
    return normal, float(-normal @ centroid)


def _ransac_vertical_plane(
    pts: np.ndarray, iters: int = 300, thresh_m: float = 0.08, seed: int = 0
) -> tuple[np.ndarray, float]:
    """Largest plane whose normal is within ~78 deg of horizontal (i.e. roughly
    vertical), then refined on its inliers. Rejects the ground / tree-canopy
    fits that an unconstrained least-squares plane would return."""
    rng = np.random.default_rng(seed)
    n_pts = len(pts)
    best_inliers = None
    best = None
    for _ in range(iters):
        tri = pts[rng.choice(n_pts, 3, replace=False)]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        if abs(normal @ UP) > 0.20:
            continue
        d = -normal @ tri[0]
        inliers = np.abs(pts @ normal + d) < thresh_m
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers, best = inliers, (normal, d)
    assert best is not None, "RANSAC found no vertical plane"
    normal, d = best
    for _ in range(3):
        inliers = np.abs(pts @ normal + d) < thresh_m
        normal, d = _fit_plane(pts[inliers])
    return normal, d


def _load_segment(start: int, count: int = 100):
    poses = load_gt_poses("00")
    out = []
    for k in range(start, start + count):
        pts = load_velodyne_scan(_velodyne_path("00", k))
        sem = load_labels(_label_path("00", k)) & 0xFFFF
        out.append((pts, sem, poses[k]))
    return out


@pytest.mark.parametrize("name,start,_desc", SEGMENTS)
def test_static_wall_plane_is_stationary(name, start, _desc):
    t_s_v = sensor_to_vehicle()
    frames = _load_segment(start, 100)

    # 1. select wall points per frame (vehicle-relative band) -> World
    per_frame_world = []
    for pts, sem, pose in frames:
        pts_vehicle = transform_points(pts[:, :3], t_s_v)
        mask = (
            (sem == BUILDING_LABEL)
            & (np.abs(pts_vehicle[:, 1]) > 1.5)
            & (np.abs(pts_vehicle[:, 1]) < 15.0)
            & (pts_vehicle[:, 0] > -8.0)
            & (pts_vehicle[:, 0] < 45.0)
            & (pts_vehicle[:, 2] > 0.3)
            & (pts_vehicle[:, 2] < 6.0)
        )
        per_frame_world.append(
            transform_points(pts_vehicle[mask], vehicle_to_world(pose))
        )

    counts = np.array([len(w) for w in per_frame_world])
    assert counts.min() >= MIN_WALL_PTS_PER_FRAME, (
        f"{name}: only {counts.min()} wall points in the thinnest frame"
    )

    # 2. one global plane, vertical-constrained
    accumulated = np.vstack(per_frame_world)
    g_normal, g_d = _ransac_vertical_plane(accumulated)
    verticality = abs(g_normal @ UP)
    assert verticality < MAX_NONVERTICAL, (
        f"{name}: global plane is not vertical (|n.up|={verticality:.3f}) -- "
        f"selection captured ground/canopy, not a wall"
    )

    # 3. per-frame consistency vs the global plane
    normal_drift_deg = []
    mean_signed_dist_m = []
    plane_rms_m = []
    for world_pts in per_frame_world:
        signed = world_pts @ g_normal + g_d
        near = np.abs(signed) < 0.4
        if near.sum() < 80:
            normal_drift_deg.append(np.nan)
            mean_signed_dist_m.append(np.nan)
            plane_rms_m.append(np.nan)
            continue
        wall = world_pts[near]
        f_normal, _ = _fit_plane(wall)
        cos = np.clip(abs(f_normal @ g_normal), -1.0, 1.0)
        normal_drift_deg.append(np.degrees(np.arccos(cos)))
        mean_signed_dist_m.append(float(np.mean(signed[near])))
        plane_rms_m.append(float(np.sqrt(np.mean(signed[near] ** 2))))

    normal_drift_deg = np.array(normal_drift_deg)
    mean_signed_dist_m = np.array(mean_signed_dist_m)
    plane_rms_m = np.array(plane_rms_m)
    ok = ~np.isnan(normal_drift_deg)
    assert ok.sum() >= 90, f"{name}: only {ok.sum()}/100 frames had a usable wall"

    frame_idx = np.arange(100)[ok]
    offset_span = float(mean_signed_dist_m[ok].max() - mean_signed_dist_m[ok].min())
    offset_slope = float(np.polyfit(frame_idx, mean_signed_dist_m[ok], 1)[0] * 100.0)

    report = (
        f"\n[{name}] frames={ok.sum()}/100  wall pts/frame min={counts.min()} "
        f"med={int(np.median(counts))}\n"
        f"  global normal={np.round(g_normal, 3)}  |n.up|={verticality:.3f}\n"
        f"  normal drift : mean={np.nanmean(normal_drift_deg):.2f} deg  "
        f"max={np.nanmax(normal_drift_deg):.2f} deg\n"
        f"  offset drift : span={offset_span * 100:.1f} cm  "
        f"slope={offset_slope * 100:.2f} cm/100frames\n"
        f"  plane RMS    : mean={np.nanmean(plane_rms_m) * 100:.1f} cm  "
        f"max={np.nanmax(plane_rms_m) * 100:.1f} cm"
    )
    print(report)

    assert np.nanmean(normal_drift_deg) < MAX_NORMAL_DRIFT_MEAN_DEG, report
    assert np.nanmax(normal_drift_deg) < MAX_NORMAL_DRIFT_MAX_DEG, report
    assert offset_span < MAX_OFFSET_SPAN_M, report
    assert abs(offset_slope) < MAX_OFFSET_SLOPE_M_PER_100F, report
    assert np.nanmax(plane_rms_m) < MAX_PLANE_RMS_M, report
