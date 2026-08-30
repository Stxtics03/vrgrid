"""Reflectivity normalisation. [JP]

Math appendix §10.3, eq (31). Raw LiDAR intensity confounds surface reflectance
with geometry -- the LiDAR equation gives ``I proportional to rho * cos(theta_inc) / r^2``
for a Lambertian surface -- so recover the intrinsic reflectance:

    rho_hat = I * r^2 / max(cos(theta_inc), COS_INC_MIN)      then -> one byte

`r` is the per-pixel range from the range image; `theta_inc` is the angle
between the beam and the local surface normal, estimated from the range image's
own neighbour structure by central differences (the "trivial on a grid,
effectively impossible on a raw cloud" operation the appendix leans on -- §7.1,
§10.3). `cos(theta_inc)` is clamped at 0.1 to avoid the singularity at pure
grazing (§3.2), and pixels that hit that clamp -- or have no usable normal --
are flagged so the caller knows the value is unreliable there.

Use: lane paint has rho ~= 0.5, dry asphalt ~= 0.1; wet asphalt reflects
specularly and returns almost nothing, so rho_hat ~= 0 on a cell classified
`road` is a wet-surface indicator. One byte, no extra sensor.

NOTE on KITTI: the recorded Velodyne intensity is already partly range-rolloff
compensated by the sensor firmware, so the `* r^2` term over-corrects far
returns. The median rho_hat still separates lane-marking from plain road
clearly (see tests/test_reflectivity.py); the per-point spread on plain road is
wide and washes out in per-cell aggregation.
"""

from dataclasses import dataclass

import numpy as np

# cos(theta_inc) clamp -- math appendix §3.2 line 241 / §10.3 eq (31).
COS_INC_MIN = 0.1

# rho_hat value that saturates the output byte. Linear map, clipped:
# byte = clip(round(rho_hat / RHO_SATURATION * 255), 0, 255). This is the single
# per-sensor calibration constant; the default keeps plain-road medians in the
# low-mid byte range on KITTI and lets the r^2-inflated tail clip.
RHO_SATURATION = 255.0

# reliability flags (bitfield, 0 = clean)
FLAG_GRAZING = 1  # cos(theta_inc) hit the COS_INC_MIN clamp
FLAG_NO_NORMAL = 2  # no usable surface normal at this pixel (edge / empty neighbour / degenerate)


@dataclass
class Reflectivity:
    """Per-pixel result, all arrays (H, W)."""

    rho8: np.ndarray  # uint8  -- normalised reflectivity, 0 where invalid
    cos_inc: np.ndarray  # float64 -- cos(theta_inc), NaN where no normal
    flags: np.ndarray  # uint8  -- FLAG_* bitfield, 0 = clean
    rho_hat: np.ndarray  # float64 -- pre-normalisation rho_hat, NaN where invalid

    @property
    def valid(self) -> np.ndarray:
        """(H, W) bool -- a normal was found and incidence was not pure grazing."""
        return self.flags == 0


def incidence_cos(range_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cos(theta_inc) per pixel from range-image neighbour normals.

    Args:
        range_image: (H, W, 5) [range, x, y, z, intensity], sensor frame, NaN empty.

    Returns:
        cos_inc: (H, W) float64 -- |n . beam|, NaN where no usable normal.
        has_normal: (H, W) bool.
    """
    xyz = range_image[:, :, 1:4].astype(np.float64)
    rng = np.linalg.norm(xyz, axis=2)

    with np.errstate(invalid="ignore", divide="ignore"):
        beam = xyz / rng[:, :, None]  # unit sensor -> surface

        # central differences: azimuth wraps (columns), rings do not (rows)
        d_az = np.roll(xyz, -1, axis=1) - np.roll(xyz, 1, axis=1)
        d_ring = np.full_like(xyz, np.nan)
        d_ring[1:-1] = xyz[2:] - xyz[:-2]

        normal = np.cross(d_az, d_ring)
        nlen = np.linalg.norm(normal, axis=2)
        normal = normal / nlen[:, :, None]

        cos_inc = np.abs(np.sum(normal * beam, axis=2))

    has_normal = np.isfinite(cos_inc) & (nlen > 1e-6) & (rng > 1e-6)
    cos_inc = np.where(has_normal, cos_inc, np.nan)
    return cos_inc, has_normal


def normalise(range_image: np.ndarray) -> Reflectivity:
    """Per-pixel normalised reflectivity from a range image (math §10.3 eq 31).

    Args:
        range_image: (H, W, 5) [range, x, y, z, intensity], sensor frame.

    Returns:
        Reflectivity -- see the dataclass. Invalid pixels (no return, no normal)
        get rho8 = 0 and FLAG_NO_NORMAL; grazing-clamped pixels get a real rho8
        and FLAG_GRAZING.
    """
    rng = range_image[:, :, 0].astype(np.float64)
    intensity = range_image[:, :, 4].astype(np.float64)
    cos_inc, has_normal = incidence_cos(range_image)

    flags = np.zeros(rng.shape, dtype=np.uint8)
    flags[~has_normal] |= FLAG_NO_NORMAL
    # grazing: cos below the clamp (only meaningful where we have a normal)
    grazing = has_normal & (cos_inc < COS_INC_MIN)
    flags[grazing] |= FLAG_GRAZING

    cos_eff = np.where(has_normal, np.maximum(cos_inc, COS_INC_MIN), np.nan)
    with np.errstate(invalid="ignore"):
        rho_hat = intensity * rng**2 / cos_eff
    rho_hat[~np.isfinite(rho_hat)] = np.nan

    rho8 = np.zeros(rng.shape, dtype=np.uint8)
    good = np.isfinite(rho_hat)
    scaled = np.clip(np.round(rho_hat[good] / RHO_SATURATION * 255.0), 0, 255)
    rho8[good] = scaled.astype(np.uint8)

    return Reflectivity(rho8=rho8, cos_inc=cos_inc, flags=flags, rho_hat=rho_hat)


def scatter_to_points(result: Reflectivity, inverse_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spread the per-pixel result back onto the source point cloud.

    Args:
        result: from normalise().
        inverse_index: (H, W) int32 from range_image.project() -- source point
            index per pixel, -1 where empty.

    Returns:
        rho8: (N,) uint8 -- 0 for points that did not project this frame.
        flags: (N,) uint8 -- FLAG_* per point; FLAG_NO_NORMAL for non-projected.
    """
    n = int(inverse_index.max()) + 1 if inverse_index.max() >= 0 else 0
    rho8 = np.zeros(n, dtype=np.uint8)
    flags = np.full(n, FLAG_NO_NORMAL, dtype=np.uint8)
    filled = inverse_index >= 0
    src = inverse_index[filled]
    rho8[src] = result.rho8[filled]
    flags[src] = result.flags[filled]
    return rho8, flags
