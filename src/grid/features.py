"""Curb and pothole detection on the 2.5D grid. [Shrestha]

The problem statement names these two by name -- it says a 2D occupancy grid
"loses critical height information necessary for detecting curbs, potholes".
That sentence is the argument for the whole project, so it is worth answering
literally rather than leaving it implied by the traversability bitfield.

WHY §7.1 DOES NOT ALREADY DO THIS
    §7.1 answers "can the vehicle be here", as six bits, and after eq. (22a)
    it answers it at one physical scale. A 12 cm kerb comes out PASSABLE --
    13.5 deg over the 0.50 m baseline, under theta_max -- and that is correct
    for a predicate about whether a wheel can climb it. It is not the whole
    answer for a planner: a kerb still bounds the drivable corridor, still
    marks where the road ends, and a route that rides along one is wrong even
    though every cell on it is traversable. So this module reports curbs as
    FEATURES with a height and an orientation, not as impassable cells.

    Potholes are the opposite case. A 40 cm hole does set bit 1, but as an
    anonymous "slope" -- indistinguishable from a kerb, a ditch or the side of
    a parked car. Depth and extent are what a planner needs to tell "slow
    down" from "go around", and neither survives the bitfield.

WHAT THIS DOES NOT DO
    It does not touch the cell struct. `CELL_FIELDS` is frozen at 12 bytes and
    adding a field means recomputing every memory figure in the report, so
    detections are returned as separate records sized by what was found -- a
    few hundred cells on a real frame, not one byte on all 745,000. The memory
    bound is unchanged and stays a compile-time claim.

    It does not run inside the frame loop. These allocate, deliberately: they
    are called on demand by the dashboard, the report and the evaluation, at
    their own rate. `no allocation inside the frame loop` (CLAUDE.md) is about
    the mapping path, and this is not on it.

THE FAILURE MODE BOTH DETECTORS ARE BUILT AROUND
    An unobserved cell holds `ground_height` 0, which is a DEFAULT and not a
    measurement at the datum. Differenced against a real neighbour it invents
    a step, and at ring 0's fill rate that is most of the map. §7.1 solves it
    with the `geometric` mask; the same rule is enforced here, and a hole in
    the data can never be reported as a pothole. `test_a_hole_in_the_data_is_
    not_a_pothole` is the test that says so.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from vrgrid.cell import FLAG_BLIND
from vrgrid.grid.schedule import load_thresholds
from vrgrid.grid.traversability import baseline_k

# The eight compass offsets, in (di, dj), ordered so that _DIRS[o] is the
# direction at angle o * 45 degrees under `arctan2(di, dj)` -- the same
# convention the curb's rise angle is measured in. Getting this ordering wrong
# does not crash: it rotates the run test 90 degrees, so a kerb is asked to
# continue ACROSS itself, finds nothing, and the detector silently reports no
# kerbs anywhere. _DIRS[(o + 2) % 8] is then the perpendicular, and
# _DIRS[(o + 4) % 8] the opposite.
_DIRS = np.array([(0, 1), (1, 1), (1, 0), (1, -1),
                  (0, -1), (-1, -1), (-1, 0), (-1, 1)], dtype=np.int64)


@dataclass(frozen=True)
class Curbs:
    """Curb edges found in one ring, addressed by flat slot within it.

    `height_cm` is the rise across the edge and `normal_deg` the compass
    direction it rises in, measured from +x toward +y -- so a planner can tell
    a kerb it is driving along from one it is about to cross.
    """
    slot: np.ndarray        # int64, index within the ring window
    height_cm: np.ndarray   # float32, rise across the edge
    normal_deg: np.ndarray  # float32, direction of the rise
    cell_m: float

    def __len__(self) -> int:
        return int(self.slot.size)


@dataclass(frozen=True)
class Potholes:
    """Local depressions: a cell sitting below the surface that surrounds it.

    `depth_cm` is measured against the MEDIAN of the rim annulus, not its mean
    -- a mean rim is dragged down by the hole's own far side once the hole is
    wider than the annulus, which reads a deep pothole as a shallow one
    exactly when it matters most.
    """
    slot: np.ndarray        # int64, index within the ring window
    depth_cm: np.ndarray    # float32, below the rim median
    rim_pairs: np.ndarray   # uint8, how many of the 4 OPPOSED rim axes were seen
    cell_m: float

    def __len__(self) -> int:
        return int(self.slot.size)


def _observed(soa, ring_slice, side, min_obs):
    """Cells holding a MEASUREMENT rather than a default, and not blind.

    ⚑ `min_obs`, not `traversability.n_min`, and the difference is the whole
      reason this function exists separately. n_min = 3 is the fail-safe
      threshold for §7.1 bit 5: below it a cell is not to be *driven on*. That
      is the right rule for a predicate about the vehicle and the wrong one
      here, and using it silently emptied this module -- on the synthetic
      scene ring 0 holds 80,719 cells with a return and only 17,516 with three
      of them, and requiring all five cells of a stencil to clear n_min left
      936 cells, 1.2% of the observed map, none of which straddled the kerb.
      The detector reported zero curbs on a scene built around a kerb.

      The failure this actually has to prevent is different: an UNOBSERVED
      cell holds `ground_height` 0 as a default, and differencing against it
      fabricates the exact feature being looked for. One return is enough to
      rule that out, because one return is a measurement. `bitfield()`'s own
      `geometric` mask draws the line in the same place, at `n >= 1`, and this
      had no business drawing it anywhere else.

      Sparse-cell noise is handled where it belongs instead -- by the curb's
      run test, which wants three aligned cells, and by the pothole's rim
      pairs. A false negative here is safe; a false positive is not.

    The blind-cone test is not redundant with the count: those cells are
    unknown by construction (master v4 §3.6) and a 3.74 m disc of default
    zeros centred on the vehicle is exactly the shape that would otherwise be
    reported as one enormous pothole under the car.
    """
    n = soa["obs_count"][ring_slice].astype(np.int32)
    blind = (soa["flags"][ring_slice] & FLAG_BLIND) != 0
    return ((n >= min_obs) & ~blind).reshape(side, side)


def _shift(a, di, dj):
    """`a` translated by (di, dj) TOROIDALLY, which is how a ring window is
    addressed.

    ⚑ A ring is a `gpu.shift.RingBuffer`: world lattice index `ix` lives at
      memory index `ix % side`, and the window's origin moves with the
      vehicle. So a physically contiguous feature is contiguous in the WORLD
      and generally split in MEMORY -- the synthetic scene's pothole came out
      at memory rows 0-4 and 395-398 of a 400-row ring, one hole in two
      pieces at opposite ends of the array.

      A linear shift with edge fill, which is what this used to be, therefore
      falls off both pieces and finds nothing: the detector reported zero
      potholes on a scene built around a pothole, and reported it without
      error. `np.roll` follows world adjacency correctly everywhere except
      across the window's single world edge, and `_seam_mask` is what removes
      that one place -- see there for why it is not memory index 0.
    """
    return np.roll(a, (di, dj), axis=(0, 1))


def _seam_mask(side: int, origin, k: int) -> np.ndarray:
    """Cells within `k` of the window's world edge, in memory coordinates.

    The seam is NOT at memory index 0. `RingBuffer.x0`/`y0` are the world
    lattice coordinates of the window's low corner, so the low edge sits at
    memory index `y0 % side` along axis 0 and `x0 % side` along axis 1 -- and
    it moves every time the vehicle recentres. Rows on either side of it are
    `side` cells apart in the world, 20 m at ring 0, and differencing across
    them fabricates a step out of two unrelated places.

    With no origin the whole array is treated as one window and the mask falls
    back to the outer k cells, which is the right answer for a plain array.
    """
    m = np.zeros((side, side), dtype=bool)
    if origin is None:
        m[:k, :] = m[-k:, :] = m[:, :k] = m[:, -k:] = True
        return m
    x0, y0 = origin                      # axis 1 is x, axis 0 is y -- as in
    i_edge, j_edge = int(y0) % side, int(x0) % side   # traversability.gradient
    for d in range(k):
        m[(i_edge + d) % side, :] = True
        m[(i_edge - 1 - d) % side, :] = True
        m[:, (j_edge + d) % side] = True
        m[:, (j_edge - 1 - d) % side] = True
    return m


def detect_curbs(soa, ring_slice, side: int, cell_m: float,
                 thresholds=None, origin=None) -> Curbs:
    """Sustained height discontinuities of curb size, on one ring.

    A curb is three things at once, and all three are required because any one
    of them alone fires on noise:

      1. A rise in the curb band -- `min_height_m` to `max_height_m`. Below the
         band is road texture, above it is a wall or a vehicle, and neither is
         a kerb.
      2. Read over a SHORT baseline. §7.1's 0.50 m baseline deliberately smears
         a kerb until it reads as a gentle slope, which is the right answer for
         "can I drive over it" and the wrong one for "where is the edge". This
         uses `features.curb.baseline_m`, which is short on purpose.
      3. Linear. A kerb is a line; a lone cell that rises 12 cm is a rock, a
         bin or a bad return. The run test requires support from the two
         neighbours PERPENDICULAR to the rise, which is where a kerb's
         continuation is and where a speckle's is not.
    """
    th = thresholds if thresholds is not None else load_thresholds()
    c = th["features"]["curb"]
    min_obs = int(th["features"]["min_obs"])

    z = soa["ground_height"][ring_slice].astype(np.float64).reshape(side, side)
    seen = _observed(soa, ring_slice, side, min_obs)

    k = baseline_k(cell_m, c["baseline_m"])
    edge = _seam_mask(side, origin, k)
    # Rise over the stencil in each axis, only where both ends were observed
    # and neither the cell nor its stencil straddles the window's world edge.
    ok = (seen & _shift(seen, 0, k) & _shift(seen, 0, -k)
          & _shift(seen, k, 0) & _shift(seen, -k, 0) & ~edge)
    dzdj = np.where(ok, _shift(z, 0, -k) - _shift(z, 0, k), 0.0)
    dzdi = np.where(ok, _shift(z, -k, 0) - _shift(z, k, 0), 0.0)
    rise_cm = np.hypot(dzdi, dzdj)

    lo, hi = c["min_height_m"] * 100.0, c["max_height_m"] * 100.0
    band = ok & (rise_cm >= lo) & (rise_cm <= hi)

    # The run test. The edge runs perpendicular to the rise, so quantise the
    # rise direction to one of eight compass points and step along the two
    # perpendicular ones. Requiring BOTH is what makes this a line test rather
    # than a blob test: a 2x2 patch of noise has neighbours, but not opposed
    # ones along a single direction.
    octant = np.rint(np.arctan2(dzdi, dzdj) / (np.pi / 4.0)).astype(np.int64) % 8
    # Orientation must agree along the run, not just membership of the band.
    # Without that, dense noise passes: in a random field some cell always has
    # two in-band neighbours by chance, and the test degenerates into "is
    # anything nearby". A kerb's neighbours rise the SAME WAY it does.
    support = np.zeros_like(band, dtype=np.int64)
    for o in range(8):
        di, dj = _DIRS[(o + 2) % 8]      # +90 degrees from the rise
        here = band & (octant == o)
        aligned = band & (np.abs(((octant - o + 4) % 8) - 4) <= 1)
        fwd = _shift(aligned, -di, -dj)
        bwd = _shift(aligned, di, dj)
        support += (here & fwd & bwd).astype(np.int64)
    keep = band & (support > 0)

    if int(c.get("min_run_cells", 3)) > 3:
        raise ValueError("min_run_cells > 3 needs a connected-run pass, not "
                         "the two-neighbour test implemented here")

    slot = np.flatnonzero(keep.reshape(-1))
    return Curbs(slot=slot,
                 height_cm=rise_cm.reshape(-1)[slot].astype(np.float32),
                 normal_deg=np.degrees(np.arctan2(dzdi, dzdj)
                                       ).reshape(-1)[slot].astype(np.float32),
                 cell_m=cell_m)


def detect_potholes(soa, ring_slice, side: int, cell_m: float,
                    thresholds=None, origin=None) -> Potholes:
    """Local depressions below the surrounding ground surface, on one ring.

    The surface a pothole is measured against is the MEDIAN of eight samples
    on a ring of radius `rim_baseline_m` around the cell. Median, not mean, for
    the reason in `Potholes`; eight samples, because four cannot outvote a hole
    that swallows two of them.

    Locality is required as well as depth: if most of the rim is itself
    depressed, this is a dip in the road, a ditch or the bottom of a hill, and
    calling it a pothole would flag every downgrade in the sequence. That test
    is what separates this from "cells below their neighbours", which is not a
    pothole detector.
    """
    th = thresholds if thresholds is not None else load_thresholds()
    p = th["features"]["pothole"]
    min_obs = int(th["features"]["min_obs"])

    z = soa["ground_height"][ring_slice].astype(np.float64).reshape(side, side)
    seen = _observed(soa, ring_slice, side, min_obs)

    r = max(1, round(p["rim_baseline_m"] / cell_m))
    rim_z, rim_ok = [], []
    for di, dj in _DIRS:
        rim_z.append(_shift(z, di * r, dj * r))
        rim_ok.append(_shift(seen, di * r, dj * r))
    rim_z = np.stack(rim_z).astype(np.float64)
    rim_ok = np.stack(rim_ok) & ~_seam_mask(side, origin, r)

    # ⚑ Only OPPOSED PAIRS count. A rim sampled on one side only is not a
    #   surface, it is a slope reading: on a 15% grade at the edge of the
    #   window the up-slope samples survive and the down-slope ones fall off
    #   the array, the median jumps by the whole rise, and a smooth downgrade
    #   reports as a field of potholes along the window border. Requiring both
    #   ends of an axis makes the estimate symmetric, so a constant grade
    #   cancels exactly -- which is the property that separates a depression
    #   from a slope.
    paired = rim_ok[:4] & rim_ok[4:]
    rim_ok = np.concatenate([paired, paired])
    support = paired.sum(axis=0)

    # Unobserved rim samples are nan and drop out of the median rather than
    # voting 0 cm, which would put the rim at the datum and read the whole
    # far field as a pothole.
    rim = np.where(rim_ok, rim_z, np.nan)
    # An all-nan rim -- every sample unobserved -- is an all-NaN slice, which
    # `nanmedian` warns about and answers nan. nan fails the `isfinite` test
    # below, which is the honest outcome: no rim, no verdict.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rim_med = np.nanmedian(rim, axis=0)

    depth_cm = rim_med - z
    min_depth_cm = p["min_depth_m"] * 100.0
    max_depth_cm = p["max_depth_m"] * 100.0

    # Locality: how much of the rim is itself down in the hole.
    depressed = np.where(rim_ok, (rim_med - rim_z) > (0.5 * min_depth_cm), False)
    local = depressed.sum(axis=0) <= (p["max_depressed_rim"] * support)

    # ⚑ The rim must agree with ITSELF. A cell just inside a kerb has half its
    #   rim on the road and half on the 12 cm sidewalk; the median then sits
    #   above the road, the road reads as a depression, and 26 cells of
    #   perfectly good carriageway came out as 10 cm potholes on the synthetic
    #   scene. A pothole sits in a locally flat surface -- if the rim spans
    #   more than `max_rim_spread_m` there is no single surface to be a hole
    #   in, and the cell declines to answer.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        spread = np.nanmax(rim, axis=0) - np.nanmin(rim, axis=0)
    flat_rim = np.nan_to_num(spread, nan=np.inf) <= p["max_rim_spread_m"] * 100.0

    keep = (seen
            & (support >= int(p["min_rim_pairs"]))
            & np.isfinite(depth_cm)
            & (depth_cm >= min_depth_cm)
            # ⚑ Bounded ABOVE as well, the same way the curb band is. On
            #   sequence 08 ring 2 this reported 156 "potholes" at a median
            #   71.5 cm and a p90 of 200 cm. A two-metre depression is not a
            #   road defect -- it is a ditch, a drop-off at the kerb line, or
            #   the cell under a parked car -- and the problem statement asks
            #   for potholes. Deeper negative obstacles are real and dangerous
            #   and belong to a separate detector, not to this one silently.
            & (depth_cm <= max_depth_cm)
            & flat_rim
            & local)

    slot = np.flatnonzero(keep.reshape(-1))
    return Potholes(slot=slot,
                    depth_cm=depth_cm.reshape(-1)[slot].astype(np.float32),
                    rim_pairs=support.reshape(-1)[slot].astype(np.uint8),
                    cell_m=cell_m)


def detect(soa, schedule, rings, thresholds=None, buffers=None):
    """Run both detectors over every ring. Mirrors `traversability.update`.

    `buffers` is `gm.buffers`, one `gpu.shift.RingBuffer` per ring: the window
    origins, without which the world edge cannot be told from a memory edge.

    Returns `(curbs, potholes)`, one entry per ring, in ring order -- not one
    merged list, because `slot` is an index WITHIN a ring window and merging
    them would silently alias ring 0's slot 5 onto ring 3's.
    """
    th = thresholds if thresholds is not None else load_thresholds()
    curbs, holes = [], []
    for level, (sl, side) in enumerate(rings):
        cell_m = schedule.rings[level].cell_m
        # Without `buffers` the window origin is unknown and the seam cannot be
        # located, so the outer k cells are masked instead -- correct for a
        # plain array, and WRONG for a live ring that has recentred. Callers
        # holding a GridMap should pass `gm.buffers`.
        origin = None if buffers is None else (buffers[level].x0, buffers[level].y0)
        curbs.append(detect_curbs(soa, sl, side, cell_m, th, origin))
        holes.append(detect_potholes(soa, sl, side, cell_m, th, origin))
    return curbs, holes
