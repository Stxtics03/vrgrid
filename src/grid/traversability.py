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

**The class table is SemanticKITTI's 19-class learning map.**
`configs/thresholds.yaml` names the drivable set in words; turning words into
ids needs the label ordering, and that is read from `configs/frnet.yaml` via
`schedule.load_class_names` rather than written out again here.

⚑ It WAS written out again here, and it was off by one for every class. The
  hand-written table began `unlabeled: 0, car: 1, ...` where the real learning
  map begins `car: 0` and puts `unlabeled` at 19. So the five names in
  `drivable_classes` resolved to the ids of
  {parking, sidewalk, other-ground, **building**, **pole**}: `road` and
  `terrain` were not drivable, and a building wall and a lamp post were. Bit 4
  is a cost of `w_class` rather than impassable, so nothing crashed and no
  path was blocked -- the whole road surface just quietly carried a penalty
  and the planner preferred to drive along the kerb line.

  It survived because the synthetic scene wrote learning ids 9/10/11 directly,
  which are inside the wrong table's drivable set by coincidence. Correcting
  that scene to raw ids (1 Sep) put `road` = 8 into the map and made the bug
  reachable. Two errors that cancelled, which is the argument for one table
  and not three.
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
from vrgrid.grid.fusion import unpack_class
from vrgrid.grid.quantise import dequantise_variance_cm2
from vrgrid.grid.schedule import load_class_names, load_thresholds


def class_ids() -> dict:
    """name -> 19-class learning id, from `configs/frnet.yaml`.

    Index in the config's `class_names` IS the learning id, which is what
    `perception.semantics.semantic_labels` produces. `unlabeled` is 19, not 0 --
    an unset class byte reads as `car`, so bit 4 must never be allowed to
    treat 0 as "no information". It does not: `car` is not drivable either.
    """
    return {name: i for i, name in enumerate(load_class_names())}


def drivable_ids(thresholds=None) -> np.ndarray:
    """The drivable set, by id. Names come from config; ids come from config."""
    th = thresholds if thresholds is not None else load_thresholds()
    ids = class_ids()
    names = th["traversability"]["drivable_classes"]
    unknown = [n for n in names if n not in ids]
    if unknown:
        raise ValueError(f"drivable_classes names no class in the label map: {unknown}")
    return np.array(sorted(ids[n] for n in names), dtype=np.int32)


def baseline_k(cell_m: float, baseline_m=None) -> int:
    """Half-width, in cells, of the §7.1 finite-difference stencil.

    Eq. (22) differenced over ONE cell measures height change per metre *at
    the cell scale*, so a step discontinuity reads steeper the finer the
    lattice. §4.1's 12 cm kerb is a gradient of 1.200 at 5 cm, 0.600 at 10 cm
    and 0.240 at 25 cm -- against one frozen `tan(theta_max) = 0.364`. The
    same physical kerb is therefore a wall on the fine rings and flat ground
    on the coarse ones and on the 25 cm reference, which is what put the two
    sides of eq. (23) on different geometry.

    A fixed physical baseline removes the scale: difference over +/-k cells
    spanning `baseline_m`, and divide by the distance actually spanned. k is
    at least 1, so a ring whose cells are already coarser than the baseline is
    left exactly as it was -- it cannot resolve the baseline, and falling back
    to one cell says so instead of inventing a sub-cell sample. That also
    makes this a no-op wherever `2 * cell_m >= baseline_m`, which is why the
    coarse rings, M*, and every existing caller passing no baseline are
    unaffected.
    """
    if baseline_m is None:
        return 1
    return max(1, round(float(baseline_m) / (2.0 * cell_m)))


def _stencil(side: int, k: int):
    """Clipped +/-k index arrays and the distance each pair actually spans.

    Clipped, not wrapped: rolling compares the north edge of the ring against
    ground `side * cell_m` to the south. Near the border the stencil shortens
    and the divisor shortens with it, so the quotient stays a real gradient in
    m/m rather than one scaled by a distance that was never spanned. The
    outermost cell of each edge is still handed bit 5 by `border_mask()` --
    a one-sided difference at the map edge is exactly the plausible number
    §7.1's note refuses to fabricate.
    """
    idx = np.arange(side)
    ip = np.minimum(idx + k, side - 1)
    im = np.maximum(idx - k, 0)
    span_cells = (ip - im).astype(np.float64)
    return ip, im, np.where(span_cells > 0, span_cells, np.inf)


def gradient(ground_cm, side: int, cell_m: float, baseline_m=None):
    """Central differences over a fixed physical baseline. Math §7.1 eq. (22).

        dz/dx = (z[i+k,j] - z[i-k,j]) / (2 k c_L),   k = baseline_m / (2 c_L)

    With `baseline_m=None` this is k=1, the one-cell form the section was
    written with. In and out in the ring's flat slot order. Dimensionless
    (m/m): heights are centimetres and `cell_m` is metres, so the 100 is the
    unit conversion and not a fudge -- suffix discipline, CLAUDE.md.

    The values on the window border are one-sided; `bitfield()` masks them.
    They are returned rather than nan-ed so the array stays int-clean and the
    caller decides.
    """
    z = np.asarray(ground_cm, dtype=np.float64).reshape(side, side) / 100.0
    ip, im, span_cells = _stencil(side, baseline_k(cell_m, baseline_m))
    dzdx = (z[:, ip] - z[:, im]) / (span_cells * cell_m)[None, :]
    dzdy = (z[ip, :] - z[im, :]) / (span_cells * cell_m)[:, None]
    return dzdx.reshape(-1), dzdy.reshape(-1)


def max_step_cm(ground_cm, side: int, cell_m: float | None = None,
                baseline_m=None):
    """max|z_c - z_nbr| over the 4-neighbourhood, in centimetres. Bit 2.

    The maximum, not the mean: a cell with three flat neighbours and one 20 cm
    kerb is a kerb, and averaging it away is how a step disappears from a map
    that still looks correct.

    The neighbour is taken at the same +/-k as the gradient, so the step is
    read over a fixed physical distance too. Without that, bit 2 scales the
    other way from bit 1: a 1.5% ramp steps 1.5 mm across a 10 cm cell and
    12 mm across an 80 cm one, and the coarse map calls a ramp a kerb. `cell_m`
    is optional so that callers testing the one-cell form need not supply it.
    """
    z = np.asarray(ground_cm, dtype=np.int32).reshape(side, side)
    k = 1 if cell_m is None else baseline_k(cell_m, baseline_m)
    ip, im, _ = _stencil(side, k)
    diffs = [np.abs(z[:, ip] - z), np.abs(z[:, im] - z),
             np.abs(z[ip, :] - z), np.abs(z[im, :] - z)]
    return np.maximum.reduce(diffs).reshape(-1)


def border_mask(side: int) -> np.ndarray:
    """The one-cell border of a ring window, where the stencil is one-sided."""
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
    # §10.2 candidate, through `unpack_class` and never a literal shift.
    # The field is 5 bits since 1 Sep, and a stale `>> 4` here reads the id
    # with the counter's top bit welded on: every drivable class then fails
    # the drivable_set test and the entire road reads untraversable.
    cls = unpack_class(soa["semantic_class"][ring_slice])[0].astype(np.int32)

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
    baseline_m = t.get("baseline_m")
    k = baseline_k(cell_m, baseline_m)
    ip, im, _ = _stencil(side, k)
    seen = (n >= 1).reshape(side, side)
    geometric = (seen & seen[:, ip] & seen[:, im] & seen[ip, :] & seen[im, :])
    geometric = geometric.reshape(-1)

    # bit 0 -- clearance
    h_vehicle_cm = t["h_vehicle_m"] * 100.0
    out |= np.where(ceiling - ground < h_vehicle_cm, TRAV_CLEARANCE, 0).astype(np.uint8)

    # bit 1 -- slope, compared as tan(theta_max) so no arctan on the hot path
    dzdx, dzdy = gradient(ground, side, cell_m, baseline_m)
    slope = np.hypot(dzdx, dzdy)
    out |= np.where(geometric & (slope > np.tan(np.radians(t["theta_max_deg"]))),
                    TRAV_SLOPE, 0).astype(np.uint8)

    # bit 2 -- step
    out |= np.where(geometric & (max_step_cm(ground, side, cell_m, baseline_m)
                                 > t["s_max_m"] * 100.0),
                    TRAV_STEP, 0).astype(np.uint8)

    # bit 3 -- roughness. The stored variance is a log code, in cm^2 once
    # decoded; sigma2_max is quoted in m^2.
    sigma2_m2 = dequantise_variance_cm2(var_code) * 1e-4
    out |= np.where(sigma2_m2 > t["sigma2_max_m2"], TRAV_ROUGHNESS, 0).astype(np.uint8)

    # bit 4 -- class. An unobserved cell has class byte 0, which in the
    # learning map is `car`, not `unlabeled` -- but `car` is not in the
    # drivable set either, so this bit still fails safe on its own, before
    # bit 5 gets to. It fails safe by arithmetic rather than by meaning, which
    # is worth knowing if the drivable set is ever widened.
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
