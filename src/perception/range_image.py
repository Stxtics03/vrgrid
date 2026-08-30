"""Spherical range image. [JP]

Feeds two things: FRNet, and the O(1)-per-cell visibility cleanup (math §10.4)
that replaces ray casting entirely.

Projection: 64 rings × 512 azimuth bins (HDL-64E on KITTI).
Inverse index: every pixel maps back to its exact source point index.
"""

import numpy as np
import yaml


def load_sensor_config(config_path: str = "configs/frnet.yaml") -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["sensor"]


def project(points: np.ndarray, sensor_cfg: dict = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 64×512 spherical range image.

    Args:
        points: (N, 4) or (N, 3) array [x, y, z, (intensity)]
        sensor_cfg: dict with sensor parameters. If None, loads from configs/frnet.yaml

    Returns:
        range_image: (64, 512, 5) float32 — [range_m, x, y, z, intensity]
        inverse_index: (64, 512) int32 — source point index, -1 for invalid
    """
    if sensor_cfg is None:
        sensor_cfg = load_sensor_config()

    H = sensor_cfg["num_rings"]      # 64
    W = sensor_cfg["num_azimuth"]    # 512
    phi_min = np.deg2rad(sensor_cfg["phi_min_deg"])
    d_phi = np.deg2rad(sensor_cfg["d_phi_deg"])
    d_theta = np.deg2rad(sensor_cfg["d_theta_deg"])

    xyz = points[:, :3]
    intensity = points[:, 3] if points.shape[1] >= 4 else np.ones(points.shape[0], dtype=np.float32)

    # Spherical coordinates
    r = np.linalg.norm(xyz, axis=1)
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])  # y left, x forward → atan2(y, x)
    elevation = np.arcsin(np.clip(xyz[:, 2] / (r + 1e-6), -1.0, 1.0))

    # Bin indices
    u = ((azimuth + np.pi) / d_theta).astype(np.int32)  # 0..511
    v = ((elevation - phi_min) / d_phi).astype(np.int32)  # 0..63

    # Clamp to valid range
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (r > 0)
    u = u[valid]
    v = v[valid]
    r = r[valid]
    xyz = xyz[valid]
    intensity = intensity[valid]
    src_idx = np.where(valid)[0]

    # For each pixel, keep closest point (smallest range)
    range_image = np.full((H, W, 5), np.nan, dtype=np.float32)
    inverse_index = np.full((H, W), -1, dtype=np.int32)

    # Sort by range so first write wins (closest point)
    order = np.argsort(r)
    u = u[order]
    v = v[order]
    r = r[order]
    xyz = xyz[order]
    intensity = intensity[order]
    src_idx = src_idx[order]

    # Unique pixels: first occurrence is closest
    _, unique_idx = np.unique(v * W + u, return_index=True)
    u = u[unique_idx]
    v = v[unique_idx]
    r = r[unique_idx]
    xyz = xyz[unique_idx]
    intensity = intensity[unique_idx]
    src_idx = src_idx[unique_idx]

    range_image[v, u, 0] = r
    range_image[v, u, 1:4] = xyz
    range_image[v, u, 4] = intensity
    inverse_index[v, u] = src_idx

    return range_image, inverse_index


def project_with_inverse(points: np.ndarray, sensor_cfg: dict = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Alias for project() — explicit about inverse index being returned.
    """
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