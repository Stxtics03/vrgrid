"""Instance clustering of moving-object points. [JP]

Connected components **on the range image**, not DBSCAN on the raw cloud. The
range image already encodes sensor adjacency -- two returns in neighbouring
(row, column) pixels came from adjacent laser shots -- so a 4-neighbour flood
fill with a range-continuity break is O(pixels), needs no spatial index, and
naturally separates a near object from a far one seen "through" the same
bearing. DBSCAN on 120k raw points would rebuild that neighbourhood structure
from scratch with an epsilon that is wrong at both 5 m and 50 m.

This groups the `is_moving()` mask into discrete instances, so the dashboard and
anything downstream can refer to "the motorcyclist" and "the pedestrian" rather
than one undifferentiated blob.

SemanticKITTI ships ground-truth instance ids in the upper 16 bits of the
`.label` word. This module deliberately does **not** read them: the motion mask
is what the pipeline carries, and clustering it is the operation a real,
label-free system would have to do. The GT instance ids are used only in the
test, to check the split is correct.

Note on counts: `inverse_index` keeps one point per filled pixel, so
`Instance.n_points` counts moving *pixels*, a slight undercount of the moving
points in the raw scan (several raw points can fall in one pixel). It is the
right quantity for separating instances; it is not a point-exact object size.
"""

from dataclasses import dataclass

import numpy as np

# Two adjacent moving pixels join the same instance only if their ranges agree
# to within this. 0.75 m spans a vehicle's own depth curvature across a pixel
# edge (0.5 m over-splits a car seen obliquely into two or three fragments)
# while still breaking the metres-wide seam between a pedestrian and a vehicle
# one bearing behind them. Verified on seq 00 f3899 (bike + vehicle, adjacent ->
# 2) and seq 07 f30 / f628 (2 objects -> 2).
RANGE_TOL_M = 0.75

# Instances smaller than this many moving pixels are dropped as noise -- a
# couple of returns clipped off a distant object, an isolated mislabel. At the
# HDL-64E's 64x512 sampling a real object below ~4 pixels is already
# indistinguishable from a stray return (a 2-point pedestrian at 20 m simply
# cannot be recovered as an instance).
MIN_PIXELS = 4


@dataclass
class Instance:
    """One connected moving object.

    `point_indices` index the original point array that `range_image` /
    `inverse_index` were built from, so `points[inst.point_indices]` are its
    returns. All geometry is in the sensor frame (the range image's frame)."""

    label: int                     # 0 .. K-1, largest instance first
    n_points: int                  # moving pixels in the instance
    point_indices: np.ndarray      # (n_points,) int, into the source point array
    centroid: np.ndarray           # (3,) mean x, y, z
    bbox_min: np.ndarray           # (3,)
    bbox_max: np.ndarray           # (3,)
    range_m: float                 # Euclidean range of the centroid

    @property
    def extent(self) -> np.ndarray:
        """(3,) bounding-box size."""
        return self.bbox_max - self.bbox_min


def _moving_pixel_mask(inverse_index: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """(H, W) bool -- pixels whose source point is flagged moving."""
    filled = inverse_index >= 0
    out = np.zeros(inverse_index.shape, dtype=bool)
    out[filled] = moving[inverse_index[filled]]
    return out


def _connected_labels(seed: np.ndarray, rng: np.ndarray, range_tol_m: float) -> np.ndarray:
    """Flat connected-component labels over `seed` pixels (int, -1 elsewhere).

    4-neighbour. Azimuth (columns) wraps -- column 0 borders column W-1 -- rows
    do not, because the top and bottom laser rings are not adjacent. An edge
    exists only where both pixels are seeds and their ranges differ by
    <= range_tol_m. Iterative label-minimisation: each sweep pulls every pixel
    down to the smallest label among itself and its connected neighbours;
    converges in as many sweeps as the widest instance is pixels across.
    """
    h, w = seed.shape
    labels = np.where(seed, np.arange(h * w).reshape(h, w), -1)
    big = h * w  # "no label" sentinel during the min; restored to -1 after

    for _ in range(h * w):  # cap; the real exit is convergence
        lab = np.where(labels < 0, big, labels)
        best = lab.copy()

        # vertical neighbours -- explicit slices, no wrap across the ring poles
        vclose = np.abs(rng[1:] - rng[:-1]) <= range_tol_m
        vconn = seed[1:] & seed[:-1] & vclose
        best[1:] = np.where(vconn, np.minimum(best[1:], lab[:-1]), best[1:])   # look up
        best[:-1] = np.where(vconn, np.minimum(best[:-1], lab[1:]), best[:-1])  # look down

        # horizontal neighbours -- azimuth wraps
        for shift in (1, -1):
            lab_n = np.roll(lab, shift, axis=1)
            rng_n = np.roll(rng, shift, axis=1)
            seed_n = np.roll(seed, shift, axis=1)
            hconn = seed & seed_n & (np.abs(rng - rng_n) <= range_tol_m)
            best = np.where(hconn, np.minimum(best, lab_n), best)

        new = np.where(best < big, best, -1)
        if np.array_equal(new, labels):
            break
        labels = new

    return labels


def cluster_moving(
    range_image: np.ndarray,
    inverse_index: np.ndarray,
    moving: np.ndarray,
    *,
    range_tol_m: float = RANGE_TOL_M,
    min_points: int = MIN_PIXELS,
) -> list[Instance]:
    """Split the moving returns of one frame into discrete instances.

    Args:
        range_image: (H, W, 5) from `range_image.project` -- [range, x, y, z, i].
        inverse_index: (H, W) int -- source point index per pixel, -1 if empty.
        moving: (N,) bool -- `semantics.is_moving`, aligned to the source points.
        range_tol_m: max range jump across a pixel edge within one instance.
        min_points: drop instances with fewer moving pixels than this.

    Returns:
        list[Instance], largest first. Empty if the frame has no moving returns.
    """
    rng = np.ascontiguousarray(range_image[:, :, 0], dtype=np.float64)
    xyz = range_image[:, :, 1:4]
    seed = _moving_pixel_mask(inverse_index, moving)
    if not seed.any():
        return []

    # empty pixels carry NaN range; force them far from any real range so a
    # roll that lands one beside a seed never reads as "close"
    rng = np.where(np.isfinite(rng), rng, -1e9)

    flat = _connected_labels(seed, rng, range_tol_m)

    out: list[Instance] = []
    for lab in np.unique(flat[seed]):
        px = flat == lab
        n = int(px.sum())
        if n < min_points:
            continue
        pts = xyz[px].astype(np.float64)
        centroid = pts.mean(axis=0)
        out.append(Instance(
            label=0,                       # renumbered after the size sort
            n_points=n,
            point_indices=np.sort(inverse_index[px]),
            centroid=centroid,
            bbox_min=pts.min(axis=0),
            bbox_max=pts.max(axis=0),
            range_m=float(np.linalg.norm(centroid)),
        ))

    out.sort(key=lambda c: c.n_points, reverse=True)
    for i, inst in enumerate(out):
        inst.label = i
    return out
