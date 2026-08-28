"""The traversability bitfield. Math §7.1. [Aakash — Day 4]

Six independent conditions, not a scalar. They fail for different reasons and
a planner should be able to tell them apart: "I cannot fit under it" and "I
have never seen it" are both untraversable and nothing else about them is
alike.

    bit 0  clearance   ceiling - ground  <  h_vehicle
    bit 1  slope       ||grad z||        >  tan(theta_max)
    bit 2  step        max|z_c - z_nbr|  >  s_max
    bit 3  roughness   sigma^2           >  sigma^2_max
    bit 4  class       class not in drivable_set
    bit 5  confidence  n                 <  n_min          (fail safe)

⚑ Geometry decides, semantics filters. A road with a 40 cm pothole has class
  `road` and is not drivable; a packed grass verge has class `vegetation` and
  usually is. Class is one bit of six, never the decision. It is also the
  evidence for the grid: slope and step are finite differences over
  neighbours, trivial here and effectively impossible on a raw point cloud
  without first building something like this.

--- what the section leaves out, and what this file does about it ---------

**Neighbours must have been OBSERVED, not merely exist.** An unobserved cell
holds ground_height 0, which is a default and not a measurement at the datum.
Differencing against it fabricates obstacles -- on a 30 cm rise, an observed
cell beside an unobserved one reads as a 30 cm step and becomes impassable.
At ring 0's 11.6% single-frame fill rate that is most of the map. Bits 1 and 2
are therefore computed only where the cell and its four neighbours have all
been seen; everything else carries bit 5 instead, which says "I have not
looked" rather than "there is a step here". Both are untraversable and only
one is true.

**Neighbours stop at the ring window.** (22) is a central difference over the
four neighbours, which needs both of them. A ring is stored as a side x side
toroidal square, so rolling the array wraps the far edge of the map onto the
near one -- a cell on the north edge would take its gradient against ground
100 m to the south. The border ring of cells therefore gets bit 5 set
(confidence) rather than a fabricated gradient: fail safe is already the rule
for "not enough evidence", and a made-up slope at the map edge is exactly the
kind of plausible number that survives review.

**Cross-ring neighbours are not computed.** A cell on the inner edge of ring 2
has neighbours in ring 1, at half the cell size. Resolving that means
resampling across a ring boundary for a two-cell strip, and it interacts with
the toroidal offsets of both rings. Not done here; the strip is marked
low-confidence by the border rule above, which is conservative in the right
direction. Worth an explicit line in the report rather than silence: the ring
seams are the one place the traversability layer is coarser than the map.

**The class table is SemanticKITTI's 19-class learning map, and it does not
fit the cell.** `configs/thresholds.yaml` names the drivable set in words;
turning words into ids needs the label map, which is JP's `semantics.py` when
it lands. Until then it is below, and it makes the §10.2 class-width conflict
concrete: **`terrain` is learning id 17 and does not fit in 4 bits.** One of
the five drivable classes cannot be stored in the byte the map has. Confirm
the ids against the real `semantic-kitti.yaml` when the download lands --
that is Hriday's R2 item -- and treat this table as provisional until then.
"""

import numpy as np
from vrgrid.cell import (
    TRAV_CLASS,
    TRAV_CLEARANCE,
    TRAV_CONFIDENCE,
    TRAV_ROUGHNESS,
    TRAV_SLOPE,
    TRAV_STEP,
)
from vrgrid.grid.quantise import dequantise_variance_cm2
from vrgrid.grid.schedule import load_thresholds

# SemanticKITTI learning ids (semantic-kitti.yaml `learning_map`), provisional
# until the download confirms them. 0 is `unlabeled`, which is why an unset
# class byte must never be read as drivable.
CLASS_IDS = {
    "unlabeled": 0, "car": 1, "bicycle": 2, "motorcycle": 3, "truck": 4,
    "other-vehicle": 5, "person": 6, "bicyclist": 7, "motorcyclist": 8,
    "road": 9, "parking": 10, "sidewalk": 11, "other-ground": 12,
    "building": 13, "fence": 14, "vegetation": 15, "trunk": 16,
    "terrain": 17, "pole": 18, "traffic-sign": 19,
}


def drivable_ids(thresholds=None) -> np.ndarray:
    """The drivable set, by id. Names come from config; ids from the table."""
    th = thresholds if thresholds is not None else load_thresholds()
    names = th["traversability"]["drivable_classes"]
    unknown = [n for n in names if n not in CLASS_IDS]
    if unknown:
        raise ValueError(f"drivable_classes names no class in the label map: {unknown}")
    return np.array(sorted(CLASS_IDS[n] for n in names), dtype=np.int32)


def gradient(ground_cm, side: int, cell_m: float):
    """Central differences over the four neighbours. Math §7.1 eq. (22).

        dz/dx = (z[i+1,j] - z[i-1,j]) / (2 c_L)

    In and out in the ring's flat slot order. Dimensionless (m/m): heights are
    centimetres and `cell_m` is metres, so the 100 is the unit conversion and
    not a fudge -- suffix discipline, CLAUDE.md.

    The values on the window border are computed by wrapping and are WRONG;
    `bitfield()` masks them. They are returned rather than nan-ed so the array
    stays int-clean and the caller decides.
    """
    z = np.asarray(ground_cm, dtype=np.float64).reshape(side, side) / 100.0
    dzdx = (np.roll(z, -1, axis=1) - np.roll(z, 1, axis=1)) / (2.0 * cell_m)
    dzdy = (np.roll(z, -1, axis=0) - np.roll(z, 1, axis=0)) / (2.0 * cell_m)
    return dzdx.reshape(-1), dzdy.reshape(-1)


def max_step_cm(ground_cm, side: int):
    """max|z_c - z_nbr| over the 4-neighbourhood, in centimetres. Bit 2.

    The maximum, not the mean: a cell with three flat neighbours and one 20 cm
    kerb is a kerb, and averaging it away is how a step disappears from a map
    that still looks correct.
    """
    z = np.asarray(ground_cm, dtype=np.int32).reshape(side, side)
    diffs = [np.abs(np.roll(z, s, axis=a) - z) for a in (0, 1) for s in (-1, 1)]
    return np.maximum.reduce(diffs).reshape(-1)


def border_mask(side: int) -> np.ndarray:
    """The one-cell border of a ring window, where a central difference would
    wrap onto the opposite edge of the map."""
    m = np.zeros((side, side), dtype=bool)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
    return m.reshape(-1)


def bitfield(soa, ring_slice: slice, side: int, cell_m: float, thresholds=None):
    """The six bits, for one ring. Math §7.1. Returns uint8, 0 = traversable.

    `ring_slice` is the ring's span in the flat arrays and `side` its window
    extent, so this works against either storage layout as long as the caller
    knows the shape of what it is passing.

    Every threshold comes from `configs/thresholds.yaml`; nothing here is
    inline, because these are frozen before schedules are compared and a
    constant living in the source cannot be frozen (flaw E6).
    """
    th = thresholds if thresholds is not None else load_thresholds()
    t = th["traversability"]

    ground = soa["ground_height"][ring_slice].astype(np.int32)
    ceiling = soa["ceiling_height"][ring_slice].astype(np.int32)
    var_code = soa["height_variance"][ring_slice]
    n = soa["obs_count"][ring_slice].astype(np.int32)
    cls = (soa["semantic_class"][ring_slice] >> 4).astype(np.int32)  # §10.2 candidate

    out = np.zeros(ground.size, dtype=np.uint8)

    # ⚑ An unobserved cell holds ground_height 0, which is a DEFAULT, not a
    # measurement at the datum. Differencing against it fabricates obstacles:
    # on a 30 cm rise, an observed cell beside an unobserved one reads as a
    # 30 cm step, sets bit 2, and becomes impassable. At ring 0's 11.6%
    # single-frame fill rate (§1.3) that is most of the map, so the far field
    # would come out walled off by cells that were never looked at -- and it
    # looks like terrain, not like a bug.
    #
    # So the geometric bits are only computed where the cell AND its four
    # neighbours have been seen. Everything else already carries bit 5
    # (confidence), which is the honest statement: not "there is a step here"
    # but "I have not looked". Both are untraversable; only one is true, and
    # only one lets a planner tell an obstacle from a hole in the data.
    seen = (n >= 1).reshape(side, side)
    geometric = seen.copy()
    for axis in (0, 1):
        for shift in (-1, 1):
            geometric &= np.roll(seen, shift, axis=axis)
    geometric = geometric.reshape(-1)

    # bit 0 -- clearance
    h_vehicle_cm = t["h_vehicle_m"] * 100.0
    out |= np.where(ceiling - ground < h_vehicle_cm, TRAV_CLEARANCE, 0).astype(np.uint8)

    # bit 1 -- slope, compared as tan(theta_max) so no arctan on the hot path
    dzdx, dzdy = gradient(ground, side, cell_m)
    slope = np.hypot(dzdx, dzdy)
    out |= np.where(geometric & (slope > np.tan(np.radians(t["theta_max_deg"]))),
                    TRAV_SLOPE, 0).astype(np.uint8)

    # bit 2 -- step
    out |= np.where(geometric & (max_step_cm(ground, side) > t["s_max_m"] * 100.0),
                    TRAV_STEP, 0).astype(np.uint8)

    # bit 3 -- roughness. The stored variance is a log code, in cm^2 once
    # decoded; sigma2_max is quoted in m^2.
    sigma2_m2 = dequantise_variance_cm2(var_code) * 1e-4
    out |= np.where(sigma2_m2 > t["sigma2_max_m2"], TRAV_ROUGHNESS, 0).astype(np.uint8)

    # bit 4 -- class. An unobserved cell has class byte 0, i.e. `unlabeled`,
    # which is not in the drivable set -- so this bit fails safe on its own,
    # before bit 5 gets to.
    out |= np.where(np.isin(cls, drivable_ids(th)), 0, TRAV_CLASS).astype(np.uint8)

    # bit 5 -- confidence. Fail safe: unobserved is not traversable. The window
    # border joins it, because a central difference there wrapped around the
    # map and bits 1 and 2 above cannot be trusted for those cells.
    thin = (n < t["n_min"]) | border_mask(side)
    out |= np.where(thin, TRAV_CONFIDENCE, 0).astype(np.uint8)

    return out


def update(soa, schedule, rings, thresholds=None) -> None:
    """Recompute the bitfield for every ring, in place into `soa`.

    `rings` is a sequence of (slice, side) -- from `gpu.allocators.RingLayout`
    or `lattice.ring_slice`/`ring_extent`, whichever the caller allocated with.
    """
    th = thresholds if thresholds is not None else load_thresholds()
    for level, (sl, side) in enumerate(rings):
        soa["traversability"][sl] = bitfield(
            soa, sl, side, schedule.rings[level].cell_m, th)


def is_traversable_bits(bits) -> np.ndarray:
    """0 means traversable. Spelled out because `not bits` reads as the
    opposite of what it means and this is a safety predicate."""
    return np.asarray(bits, dtype=np.uint8) == 0
