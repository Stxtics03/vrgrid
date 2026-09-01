"""Residual-image motion segmentation (MOS). [JP]

A ground-truth-label-free alternative to `semantics.is_moving()`: flag moving
returns from the disagreement between the current range image and the previous
one *warped into the current viewpoint* by ego-motion.

    1. compose the relative sensor transform between the two GT poses
       (`transforms.sensor_to_world` and its inverse)
    2. warp frame N-1's returns into frame N's sensor frame and re-project them
       through the same spherical projection as `range_image.project`
    3. residual = |current range - predicted range| at every pixel both cover
    4. threshold the residual -> a candidate-motion mask, mapped back to the
       current points by re-projecting each point

**This is a geometric baseline, not a replacement for the GT motion flag.** On
a 64x512 range image with single-frame differencing it recovers fast, large
movers at modest precision and misses slow ones. Measured against GT
`is_moving()` on SemanticKITTI (see `tests/test_residual_mos.py` for the
numbers): pixel-level IoU ~0.25 on a frame with a fast vehicle (seq 07 f674),
and near-zero precision on a frame whose only movers are far and slow while the
sensor drives fast through dense structure (seq 00 f10). The disagreements are
mostly (a) false positives at dis-occlusion boundaries -- where the sensor's
own motion reveals background the previous frame could not see -- and (b)
missed slow pedestrians, whose per-frame displacement is below the sensor's
range-quantisation noise. Both are inherent to the single-pair geometric
approach; a learned MOS head or multi-frame differencing closes the gap.
"""

from dataclasses import dataclass

import numpy as np

from . import range_image, transforms

# Residual above this many metres marks a pixel as candidate-motion.
#
# Justification, not a round guess: on frames with little ego-motion the static
# |residual| distribution sits at p50 < 0.05 m and p90 < 0.1 m -- it is
# dominated by the 64x512 image's azimuth quantisation at range (a ~0.7 deg
# column is ~0.6 m wide at 50 m) plus GT-pose registration error, both a few cm
# to a few tens of cm. A vehicle at 10 m/s displaces ~1.0 m between 10 Hz
# frames. 0.75 m sits above the static bulk and below one frame of typical
# vehicle motion. On frames with large ego-motion through dense structure the
# static tail (dis-occlusion) climbs past this and precision drops -- a
# documented failure mode, not a threshold to tune away.
RESIDUAL_THRESHOLD_M = 0.75

# Pixels closer than this are ignored: the blind cone and sensor-mount returns
# have unstable geometry and dominate the residual with noise.
MIN_RANGE_M = 2.0


@dataclass
class ResidualMOS:
    """Output of `residual_motion_mask`.

    `point_mask` is aligned to the current point array; `pixel_mask`,
    `residual` and `valid` are (H, W) in the range-image frame."""

    point_mask: np.ndarray     # (N,) bool -- current points in a motion pixel
    pixel_mask: np.ndarray     # (H, W) bool
    residual: np.ndarray       # (H, W) float, metres; NaN where not comparable
    valid: np.ndarray          # (H, W) bool -- both frames have a return here
    threshold_m: float


def relative_transform(pose_prev: np.ndarray, pose_curr: np.ndarray,
                       sequence: str) -> np.ndarray:
    """4x4 taking a point in frame N-1's sensor frame to frame N's sensor frame.

    `inv(T_world<-curr) @ T_world<-prev`, both from `transforms.sensor_to_world`
    so the Velodyne->Camera-0 extrinsic and the world convention match the rest
    of the pipeline exactly."""
    t_wp = transforms.sensor_to_world(pose_prev, sequence=sequence)
    t_wc = transforms.sensor_to_world(pose_curr, sequence=sequence)
    return np.linalg.inv(t_wc) @ t_wp


def _pixel_index(xyz: np.ndarray, cfg: dict):
    """(v, u) column/row per point -- the exact formula `range_image.project`
    uses, so a forward lookup lines up with its output pixel-for-pixel."""
    h = cfg["num_rings"]
    w = cfg["num_azimuth"]
    phi_max = np.deg2rad(cfg["phi_max_deg"])
    d_theta, d_phi = range_image.bin_widths(cfg)

    r = np.linalg.norm(xyz, axis=1)
    finite = r > 1e-6
    az = np.arctan2(xyz[:, 1], xyz[:, 0])
    z_over_r = np.divide(xyz[:, 2], r, out=np.zeros_like(r), where=finite)
    el = np.arcsin(np.clip(z_over_r, -1.0, 1.0))
    u = np.floor((az + np.pi) / d_theta).astype(np.int64) % w
    v = np.clip(np.floor((phi_max - el) / d_phi).astype(np.int64), 0, h - 1)
    return v, u, r, finite


def residual_motion_mask(
    pts_prev: np.ndarray,
    pts_curr: np.ndarray,
    t_rel: np.ndarray,
    *,
    threshold_m: float = RESIDUAL_THRESHOLD_M,
    min_range_m: float = MIN_RANGE_M,
) -> ResidualMOS:
    """Candidate moving returns of the current scan from a warped-residual.

    Args:
        pts_prev: (M, 3+) frame N-1 points, sensor frame.
        pts_curr: (N, 3+) frame N points, sensor frame.
        t_rel: 4x4 prev-sensor -> curr-sensor (see `relative_transform`).
        threshold_m: residual above this marks a pixel as moving.
        min_range_m: ignore returns closer than this.

    Returns:
        ResidualMOS.
    """
    cfg = range_image.load_sensor_config()

    prev_xyz = np.asarray(pts_prev)[:, :3]
    curr = np.asarray(pts_curr)
    curr_xyz = curr[:, :3]

    warped = transforms.transform_points(prev_xyz, t_rel)
    intensity_col = pts_prev[:, 3:4] if np.asarray(pts_prev).shape[1] >= 4 else \
        np.zeros((len(prev_xyz), 1), np.float32)
    pred_img, _ = range_image.project(np.hstack([warped, intensity_col]).astype(np.float32))
    curr_img, _ = range_image.project(curr)

    pred_r = pred_img[:, :, 0]
    curr_r = curr_img[:, :, 0]
    valid = np.isfinite(pred_r) & np.isfinite(curr_r) & (curr_r >= min_range_m)
    residual = np.where(valid, np.abs(curr_r - pred_r), np.nan)
    pixel_mask = valid & (residual > threshold_m)

    # per-point: re-project every current point, flag those in a motion pixel
    v, u, r, finite = _pixel_index(curr_xyz, cfg)
    point_mask = np.zeros(len(curr_xyz), dtype=bool)
    inb = finite & (r >= min_range_m)
    point_mask[inb] = pixel_mask[v[inb], u[inb]]

    return ResidualMOS(point_mask=point_mask, pixel_mask=pixel_mask,
                       residual=residual, valid=valid, threshold_m=threshold_m)


def residual_motion_from_poses(
    pts_prev: np.ndarray,
    pose_prev: np.ndarray,
    pts_curr: np.ndarray,
    pose_curr: np.ndarray,
    sequence: str,
    **kw,
) -> ResidualMOS:
    """Convenience wrapper: compose `t_rel` from the two poses, then run."""
    t_rel = relative_transform(pose_prev, pose_curr, sequence)
    return residual_motion_mask(pts_prev, pts_curr, t_rel, **kw)
