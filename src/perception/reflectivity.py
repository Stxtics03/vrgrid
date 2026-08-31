"""Reflectivity normalisation. [JP]

Math appendix §10.3 eq (31). The LiDAR equation for a Lambertian surface is
``I_received  proportional to  rho * cos(theta_inc) / r^2``, so the intrinsic
reflectance is recovered by

    rho_hat = I * r^2 / max(cos(theta_inc), 0.1)                       (31)

`theta_inc` is the angle between the beam and the local surface normal,
estimated from the range image's own neighbour structure by central differences
(the "trivial on a grid" operation the appendix leans on -- §7.1, §10.3),
implemented in `incidence_cos()` and checked against analytic plane normals to
1e-4.

Eq (31) assumes the sensor reports RAW received power. KITTI's Velodyne does
not -- its firmware already delivers a range- and incidence-normalised
reflectance-like quantity in [0, 1]. Measured on flat asphalt (label `road`)
across sequence 00:

    log(I) vs log(r) slope = 0.01           -> I has no range trend  (r^2 already out)
    median I flat at ~0.25 while cos(theta_inc)
        falls from 0.32 at 6 m to 0.12 at 17 m -> I has no incidence trend either

Applying eq (31)'s geometric terms to KITTI therefore re-injects a range trend
that is not in the data and destroys cross-range comparability:

    * `* r^2`    -> ring-1 road pixels 62% saturated at byte 255 (ring 0: 0%)
    * `/ cos`    -> ring-1 road reads ~2x ring-0 road, purely from geometry

So for KITTI both `range_compensated` and `incidence_compensated` default to
True and `rho_hat = I`; the byte is `round(clip(I, 0, 1) * 255)` -- a direct
8-bit quantisation of the reflectance the firmware already gives us. A raw-power
sensor passes `range_compensated=False, incidence_compensated=False` and gets
the full eq (31). `incidence_cos()` is always computed and returned regardless,
because the elevation-variance model (§3.2) and the caller may want it.

Use (math §10.3): lane paint reads ~1.3-1.5x brighter than plain asphalt
(per-point, range-stable); wet asphalt reflects specularly and returns almost
nothing, so `rho8 ~= 0` on a cell classified `road` is a wet-surface indicator.
One byte, no extra sensor.
"""

from dataclasses import dataclass

import numpy as np

# cos(theta_inc) clamp for the raw-power eq-(31) path -- math §3.2 / §10.3.
COS_INC_MIN = 0.1

# rho_hat value mapped to byte 255 on the raw-power path (I * r^2 / cos, which
# is unbounded). Unused on the KITTI path, where rho_hat = I in [0, 1] and the
# map to a byte is a plain * 255.
RHO_SATURATION = 4000.0

# reliability flags (bitfield, 0 = clean)
FLAG_GRAZING = 1    # cos(theta_inc) below COS_INC_MIN -- advisory; on the raw-power
#                     path the value used the clamp, not the true cos
FLAG_NO_NORMAL = 2  # no usable surface normal (image edge / empty neighbour / degenerate)


@dataclass
class Reflectivity:
    """Per-pixel result, all arrays (H, W)."""

    rho8: np.ndarray     # uint8   -- normalised reflectivity, 0 where no return
    cos_inc: np.ndarray  # float64 -- cos(theta_inc), NaN where no normal
    flags: np.ndarray    # uint8   -- FLAG_* bitfield, 0 = clean
    rho_hat: np.ndarray  # float64 -- pre-byte reflectivity estimate, NaN where none

    @property
    def valid(self) -> np.ndarray:
        """(H, W) bool -- a reflectivity value was produced for this pixel."""
        return np.isfinite(self.rho_hat)


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


def normalise(
    range_image: np.ndarray,
    *,
    range_compensated: bool = True,
    incidence_compensated: bool = True,
    rho_full_scale: float | None = None,
) -> Reflectivity:
    """Per-pixel reflectivity byte from a range image (math §10.3 eq 31).

    Args:
        range_image: (H, W, 5) [range, x, y, z, intensity], sensor frame.
        range_compensated: True (KITTI default) -> skip the `* r^2` term because
            the sensor firmware already removed the range roll-off. False -> a
            raw-power sensor, apply `* r^2` (eq 31).
        incidence_compensated: True (KITTI default) -> skip the `/ cos` term.
            False -> apply `/ max(cos, COS_INC_MIN)` (eq 31).
        rho_full_scale: rho_hat value mapped to byte 255. Default: 1.0 when both
            compensations are on (rho_hat = I in [0, 1]), else RHO_SATURATION.

    Returns:
        Reflectivity. Pixels with no return get rho8 = 0, rho_hat = NaN,
        FLAG_NO_NORMAL. On the raw-power path, pixels with no surface normal also
        get rho8 = 0 (the `/cos` term needs one).
    """
    rng = range_image[:, :, 0].astype(np.float64)
    intensity = range_image[:, :, 4].astype(np.float64)
    cos_inc, has_normal = incidence_cos(range_image)

    flags = np.zeros(rng.shape, dtype=np.uint8)
    flags[~has_normal] |= FLAG_NO_NORMAL
    flags[has_normal & (cos_inc < COS_INC_MIN)] |= FLAG_GRAZING

    rho_hat = intensity.copy()
    if not range_compensated:
        rho_hat = rho_hat * rng**2
    if not incidence_compensated:
        cos_eff = np.where(has_normal, np.maximum(cos_inc, COS_INC_MIN), np.nan)
        with np.errstate(invalid="ignore"):
            rho_hat = rho_hat / cos_eff

    rho_hat[~np.isfinite(rng) | (rng <= 0) | ~np.isfinite(intensity)] = np.nan
    rho_hat[~np.isfinite(rho_hat)] = np.nan

    if rho_full_scale is None:
        rho_full_scale = 1.0 if (range_compensated and incidence_compensated) else RHO_SATURATION

    rho8 = np.zeros(rng.shape, dtype=np.uint8)
    good = np.isfinite(rho_hat)
    rho8[good] = np.clip(
        np.round(rho_hat[good] / rho_full_scale * 255.0), 0, 255
    ).astype(np.uint8)

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
