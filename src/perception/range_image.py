"""Spherical range image. [JP]

Feeds two things: FRNet, and the O(1)-per-cell visibility cleanup (math §10.4)
that replaces ray casting entirely.

Projection: 64 rings x 512 azimuth bins (HDL-64E on KITTI).
Inverse index: every filled pixel maps back to its exact source point index, so
`range_image[v, u, 1:4] == points[inverse_index[v, u], :3]` byte-for-byte. Tested
in `tests/test_range_image.py`.

Binning
-------
The azimuth and elevation bin widths are DERIVED from the image size and the
sensor's vertical FOV, not read as fixed degree values:

    d_theta = 2*pi / num_azimuth                     (full 360 deg across W cols)
    d_phi   = (phi_max - phi_min) / num_rings         (full FOV across H rows)

(The older `d_theta_deg` / `d_phi_deg` config keys were inconsistent with
`num_azimuth = 512` -- 0.2 deg implies 1800 columns -- and silently dropped ~72%
of every scan on the azimuth axis. They are ignored now.)

Out-of-FOV elevation: CLAMP, not drop
-------------------------------------
A point whose spherical elevation lands outside [phi_min, phi_max] is assigned
to the nearest edge ring (row 0 or row H-1) rather than discarded. Reasoning:

  * Every KITTI point is a real return from one of the 64 physical HDL-64E laser
    channels, so it belongs to some ring. An elevation just outside the nominal
    FOV is a calibration / quantisation effect (per-ring elevations vary ~+-0.5
    deg; near-range points have ill-conditioned elevation), not proof the return
    is spurious.
  * Dropping it would leave an empty pixel, and the visibility-cleanup kernel
    (math §10.4) reads an empty pixel as "no return along this ray" -- a false
    gap there would let it clear a cell that actually had a return, breaking the
    "never clear a cell with a current return" guard.
  * Clamping the row index matches the SemanticKITTI community convention
    (RangeNet++, SalsaNext, FRNet all `np.clip` proj_y), so our range image
    lines up with FRNet's training-time preprocessing.

The clamp counts are returned in `stats` (call with `return_stats=True`) so a
spike -- a wrong FOV in the config, or points in the wrong frame -- is visible.
On KITTI 00 the clamp-above fraction is ~4-7%; a warning fires above 15%.
"""

import warnings
from pathlib import Path

import numpy as np
import yaml

OUT_OF_FOV_WARN_FRAC = 0.15

# Resolved from this file, not the CWD -- matches src/grid/schedule.py.
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_sensor_config(config_path: str | Path = CONFIG_DIR / "frnet.yaml") -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["sensor"]


def bin_widths(sensor_cfg: dict) -> tuple[float, float]:
    """(d_theta, d_phi) in radians, derived from image size and vertical FOV."""
    w = sensor_cfg["num_azimuth"]
    h = sensor_cfg["num_rings"]
    d_theta = 2.0 * np.pi / w
    d_phi = np.deg2rad(sensor_cfg["phi_max_deg"] - sensor_cfg["phi_min_deg"]) / h
    return d_theta, d_phi


# Row convention: row 0 is the TOP of the image (highest elevation, phi_max),
# row H-1 is the bottom (phi_min). This matches the SemanticKITTI / RangeNet++ /
# FRNet convention. Column 0 is azimuth -pi, increasing with atan2(y, x).


def project(
    points: np.ndarray, sensor_cfg: dict | None = None, return_stats: bool = False
):
    """Project 3D points to an (H, W) spherical range image.

    Args:
        points: (N, 4) or (N, 3) array [x, y, z, (intensity)], in the frame the
            range image should be built in (sensor frame for FRNet).
        sensor_cfg: sensor parameter dict; loaded from configs/frnet.yaml if None.
        return_stats: if True, also return a stats dict (see below).

    Returns:
        range_image: (H, W, 5) float32 -- [range_m, x, y, z, intensity], NaN where
            no point projected.
        inverse_index: (H, W) int32 -- source point index per pixel, -1 where empty.
        stats (only if return_stats): dict with n_input, n_zero_range,
            n_clamped_below, n_clamped_above, n_pixels_filled, fill_fraction,
            out_of_fov_fraction.
    """
    if sensor_cfg is None:
        sensor_cfg = load_sensor_config()

    h = sensor_cfg["num_rings"]
    w = sensor_cfg["num_azimuth"]
    phi_max = np.deg2rad(sensor_cfg["phi_max_deg"])
    d_theta, d_phi = bin_widths(sensor_cfg)

    pts = np.asarray(points)
    xyz = pts[:, :3]
    intensity = (
        pts[:, 3].astype(np.float32)
        if pts.shape[1] >= 4
        else np.ones(pts.shape[0], dtype=np.float32)
    )
    n_input = pts.shape[0]

    r = np.linalg.norm(xyz, axis=1)
    finite = r > 1e-6
    n_zero_range = int((~finite).sum())

    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    z_over_r = np.divide(xyz[:, 2], r, out=np.zeros_like(r), where=finite)
    elevation = np.arcsin(np.clip(z_over_r, -1.0, 1.0))

    # azimuth wraps -- every point has a valid column
    u = np.floor((azimuth + np.pi) / d_theta).astype(np.int64) % w

    # elevation -- row 0 = phi_max (top). Clamp to the edge ring, count how many
    # were out of FOV. "above" = above phi_max (elevation too high, row < 0);
    # "below" = below phi_min (row >= h).
    v_raw = np.floor((phi_max - elevation) / d_phi).astype(np.int64)
    n_clamped_above = int((finite & (v_raw < 0)).sum())
    n_clamped_below = int((finite & (v_raw >= h)).sum())
    v = np.clip(v_raw, 0, h - 1)

    out_of_fov_fraction = (
        (n_clamped_below + n_clamped_above) / n_input if n_input else 0.0
    )
    if out_of_fov_fraction > OUT_OF_FOV_WARN_FRAC:
        warnings.warn(
            f"range_image.project: {out_of_fov_fraction:.1%} of points fell "
            f"outside the vertical FOV [{sensor_cfg['phi_min_deg']}, "
            f"{sensor_cfg['phi_max_deg']}] deg and were clamped to an edge ring "
            f"-- check the sensor config and the point frame",
            stacklevel=2,
        )

    keep = finite
    u, v, r = u[keep], v[keep], r[keep]
    xyz, intensity = xyz[keep], intensity[keep]
    src_idx = np.flatnonzero(keep)

    range_image = np.full((h, w, 5), np.nan, dtype=np.float32)
    inverse_index = np.full((h, w), -1, dtype=np.int32)

    # closest point wins: sort by range ascending, take first per pixel
    order = np.argsort(r, kind="stable")
    u, v, r = u[order], v[order], r[order]
    xyz, intensity, src_idx = xyz[order], intensity[order], src_idx[order]

    _, first = np.unique(v * w + u, return_index=True)
    u, v = u[first], v[first]
    range_image[v, u, 0] = r[first]
    range_image[v, u, 1:4] = xyz[first]
    range_image[v, u, 4] = intensity[first]
    inverse_index[v, u] = src_idx[first]

    if not return_stats:
        return range_image, inverse_index

    n_filled = int((inverse_index >= 0).sum())
    stats = {
        "n_input": n_input,
        "n_zero_range": n_zero_range,
        "n_clamped_below": n_clamped_below,
        "n_clamped_above": n_clamped_above,
        "n_pixels_filled": n_filled,
        "fill_fraction": n_filled / (h * w),
        "out_of_fov_fraction": out_of_fov_fraction,
    }
    return range_image, inverse_index, stats


def project_with_inverse(points: np.ndarray, sensor_cfg: dict | None = None):
    """Alias for project() -- explicit about the inverse index being returned."""
    return project(points, sensor_cfg)


def range_image_to_frnet_input(range_image: np.ndarray) -> np.ndarray:
    """
    Convert range image to FRNet input format.

    FRNet expects: (1, 5, 64, 512) — [range, x, y, z, intensity]
    Normalized: range in [0, 1], xyz in [-1, 1], intensity in [0, 1]
    """
    H, W, C = range_image.shape
    assert H == 64 and W == 512 and C == 5

    # Normalize
    img = range_image.copy()

    # Range: normalize to [0, 1] using max range ~100m
    img[:, :, 0] = np.clip(img[:, :, 0] / 100.0, 0, 1)

    # XYZ: normalize to [-1, 1] using ±100m bounds
    img[:, :, 1:4] = np.clip(img[:, :, 1:4] / 100.0, -1, 1)

    # Intensity: normalize to [0, 1] (assuming 0-255 or similar)
    img[:, :, 4] = np.clip(img[:, :, 4] / 255.0, 0, 1)

    # Replace NaN with 0
    img = np.nan_to_num(img, nan=0.0)

    # Transpose to (C, H, W) and add batch dim
    img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]  # (1, 5, 64, 512)

    return img.astype(np.float32)
