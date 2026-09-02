"""Per-cell height and occupancy fusion. Math §3, §10. [Aakash — Day 1]

Kalman update with a range-dependent measurement model: a return at 50 m is
not evidence of the same strength as a return at 5 m.

Accumulation is fixed-point int32 in 1 cm units. Float atomic adds are
non-associative, so two runs over identical input produce different maps and
bugs move when you look at them. `make test-determinism` is CI-blocking for
exactly this reason. See math §3.4.

Class fusion is Boyer-Moore streaming majority in one byte (4-bit candidate,
4-bit counter): match -> increment, mismatch -> decrement, zero -> adopt.
Never average softmax vectors across frames.

--- what this file owns, and what it does not ----------------------------

`gpu/kernels.py` turns a scan into per-cell sufficient statistics and says so:
it owns making the accumulation fast and repeatable, not what the numbers
mean. This file is the other half — the filter (§3.3), the three-state
occupancy rule (§10.1), the majority vote (§10.2) and the reflectivity
normalisation (§10.3). The split is worth keeping: everything here is
elementwise over touched cells, with no reduction and therefore no ordering,
so determinism is settled upstream and not re-argued per rule.

⚑ The filter runs in float64 and that does NOT reintroduce the §3.4 problem.
  What is non-associative is *summation order*, and every sum has already
  happened in integers by the time a CellAggregate exists. The update below
  is elementwise on already-reduced inputs, so identical inputs give
  bit-identical outputs; the results land back in int16 cm and one uint8
  code, both rounded by fixed rules.

--- two defects at the seam with gpu/, both now closed from that side ------

Kept as a record of what the seam is for, because both produced maps that
looked entirely plausible and neither was findable from this file alone.

(1) `allocate()` zeroed every field, so `ceiling_height` started at 0, which
    reads as "something solid at the ground datum" rather than "nothing
    overhead seen" — the sentinel for that is `CEILING_NONE` = 32767. Left
    alone, `ceiling - ground < h_vehicle` was true for every cell in the map
    and TRAV_CLEARANCE marked the entire world untraversable, forever,
    because nothing ever raises a ceiling back up. Fixed in `allocate()` and
    in the shift's strip clear, which now share one `EMPTY_CELL` definition;
    `initialise()` below delegates to it and is kept for callers.

(2) The aggregate's height sums were taken over ALL returns, ground and not,
    so a cell holding a road return at 0 cm and a canopy return at 200 cm
    reported a ground height of 100 cm — a metre of tree averaged into the
    road. The kernel now masks the height sums by `is_ground`, which makes
    `wz_sum`/`w_sum` the GROUND evidence §3 says they are, and makes
    `w_sum == 0` mean "no ground return this frame". `fuse()` leaves such a
    cell's height and variance alone — see `has_ground_evidence()` below.
"""

import numpy as np
from vrgrid.cell import (
    FLAG_BLIND,
    OCC_FREE,
    OCC_OCCUPIED,
    OCC_UNKNOWN,
)
from vrgrid.gpu.allocators import initialise_cells
from vrgrid.gpu.kernels import WEIGHT_SCALE
from vrgrid.grid.quantise import dequantise_variance_cm2, quantise_variance_cm2
from vrgrid.grid.schedule import load_thresholds

# --- §10.2: the one byte -----------------------------------------------------
# 5-bit candidate | 3-bit counter. Ratified 1 Sep (Gate 3, item 3).
#
# §10.2 as written specifies 4 | 4, and that is wrong for this project: the
# label set is 20 learning ids (0-19) and a 4-bit candidate holds 16. The
# shortfall was not "a few rare classes missing" -- it landed in three
# independent places, each of which read as working:
#
#   * `pole` (18) and `traffic-sign` (19) are the two classes the semantic
#     refinement gate exists for -- thin structures smaller than the cell they
#     land in past 25 m. No cell could report either, so the gate's most
#     important criterion matched nothing and never fired (`grid/gate.py`).
#   * `terrain` (17) is one of the five `drivable_classes` the traversability
#     predicate consults on every cell (§7.1), so one of the five was
#     unstorable.
#   * The first real 19-class frame raised rather than degrading, and two
#     different stand-ins had grown around it -- `% 16` in the eval harness and
#     `clip(0, 15)` in the frame loop -- which disagreed with each other.
#     `% 16` mapped `terrain` onto `car`, turning drivable ground into blocked
#     cells; `clip` mapped everything above 15 onto `vegetation`.
#
# 5 | 3 holds all 20 with room to 31, and costs only the top of the counter's
# range: it saturates at 7 rather than 15. Boyer-Moore's majority guarantee is
# unaffected by where the counter saturates -- the invariant is that a true
# majority can never be fully decremented away, which holds for any cap >= 1 --
# so only the confidence readout gets coarser. The byte is still one byte, so
# the frozen 12-byte cell struct does not move.
CLASS_BITS = 5
COUNTER_BITS = 8 - CLASS_BITS
CLASS_MAX = (1 << CLASS_BITS) - 1      # 31, and the label set stops at 19
COUNTER_MAX = (1 << COUNTER_BITS) - 1  # 7

# Learning ids run 0-19, so 20-31 are spare inside the 5-bit field. 31 is the
# one reserved meaning: the point carried no usable label. Real SemanticKITTI
# scans contain `unlabeled` (raw 0) and ids outside the scheme, which
# `semantic_labels()` reports as -1 -- and -1 through `astype(uint8)` is 255,
# which does not fit the field and cannot be packed. It is NOT in any drivable
# set, so a cell that lands here fails safe by §7.1 bit 4, which is the same
# answer an unknown class should get anyway.
CLASS_UNLABELLED = CLASS_MAX  # 31

OBS_COUNT_MAX = 255                    # uint8, saturating (cell.py)
FRAMES_SEEN_MAX = 255


def initialise(soa) -> None:
    """Put the grid into the state a never-observed map is supposed to be in.

    `allocate()` and `shift()` now do this themselves, so this is no longer
    required after either — it is kept because it is idempotent, because a
    grid built by hand in a test still needs it, and because deleting a public
    name mid-week costs more than keeping one that forwards.

    Only `ceiling_height` needs setting: every other field's "no information"
    value really is zero — obs_count 0, log_odds 0 (§10.1 decides unknown by
    count, not by log-odds near zero), and variance code 0, which this
    project's codec deliberately maps to MAXIMUM variance for exactly this
    reason. Ceiling is the one field whose empty value is a sentinel rather
    than a zero. See defect (1) in the module docstring.
    """
    initialise_cells(soa)


# --- §3.3: the scalar update -------------------------------------------------


def fuse(soa, aggregate, thresholds=None, dt_s: float | None = None) -> None:
    """Fold one frame's aggregate into the persistent map, in place. Math §3.3.

        K   = sigma2_prior / (sigma2_prior + sigma2_z)
        mu <- mu + K (z - mu)
        sigma2 <- (1 - K) sigma2_prior

    `aggregate` is a `gpu.kernels.CellAggregate`: sufficient statistics per
    touched cell, already reduced in integers. Its `wz_sum`/`w_sum` are the
    frame's GROUND evidence — see defect (2) in the module docstring.

    The frame's measurement variance falls out of the weights rather than
    being recomputed: the kernel quantises w = WEIGHT_SCALE/sigma2_i, so a
    cell's combined precision is `w_sum` and `sigma2_z = WEIGHT_SCALE/w_sum`.
    That is inverse-variance combination, and it is the RIGHT rule here for
    the same reason it is the wrong one in §4: several returns in one cell in
    one frame are repeated measurements of one quantity. Across a footprint
    they are not, which is what §4.1 is about.

    Process noise (`sigma2 <- sigma2 + q dt`) is added to the prior of every
    cell updated here. Ageing the cells NOT in this frame — process noise plus
    `frames_since_seen` — belongs with the visibility pass (§10.4, Day 3), so
    that the whole grid is walked once per frame rather than twice.
    """
    if len(aggregate) == 0:
        return
    th = thresholds if thresholds is not None else load_thresholds()
    fus = th.get("fusion", {})
    occ = th["occupancy"]
    dt = fus.get("frame_dt_s", 0.1) if dt_s is None else dt_s

    slots = np.asarray(aggregate.cells, dtype=np.int64)
    if np.any(slots < 0):
        raise ValueError(
            "aggregate contains a negative cell index. annulus_index() returns "
            "-1 for the ring's hole and scatter drops those; a -1 arriving here "
            "would be written into cell 0 and pile the far field onto the origin"
        )

    # --- prior, with process noise (§3.3) -----------------------------------
    q_cm2 = fus.get("process_noise_m2_per_s", 1e-4) * 1e4
    prior_var = dequantise_variance_cm2(soa["height_variance"][slots]) + q_cm2 * dt
    prior_mu = soa["ground_height"][slots].astype(np.float64)

    # --- measurement --------------------------------------------------------
    z_cm = aggregate.mean_height_cm().astype(np.float64)
    meas_var = WEIGHT_SCALE / np.maximum(aggregate.w_sum.astype(np.float64), 1.0)

    # --- update -------------------------------------------------------------
    gain = prior_var / (prior_var + meas_var)
    post_mu = prior_mu + gain * (z_cm - prior_mu)
    post_var = (1.0 - gain) * prior_var

    # A cell with no observations has no prior to update. Initialising on the
    # first measurement rather than filtering against a nominal one keeps the
    # cold start independent of where the codec's ceiling happens to sit.
    first = soa["obs_count"][slots] == 0
    post_mu = np.where(first, z_cm, post_mu)
    post_var = np.where(first, meas_var, post_var)

    # ...and a cell whose returns this frame were ALL non-ground has no
    # measurement to update it with. The kernel masks the height sums by
    # `is_ground`, so `w_sum == 0` says exactly that, and `mean_height_cm()`
    # is 0 there for want of anything better to return. Writing that 0 would
    # put the top of every wall and vehicle at the datum -- 1.73 m above the
    # road in the vehicle frame -- with `meas_var` of 1024 cm^2 behind it,
    # which is confident enough to hold against the next few real returns.
    #
    # The prior goes back untouched except for the process noise already added
    # to it, which is the right answer: the cell was seen, so its occupancy,
    # class and ceiling all update below; its ground height simply was not
    # measured this frame, and it ages exactly as an unseen cell's would.
    # This overrides `first` deliberately -- a cell whose first-ever return is
    # a wall is the case that matters most.
    ground = aggregate.has_ground_evidence()
    post_mu = np.where(ground, post_mu, prior_mu)
    post_var = np.where(ground, post_var, prior_var)

    soa["ground_height"][slots] = np.clip(np.rint(post_mu), -32768, 32767).astype(np.int16)
    soa["height_variance"][slots] = quantise_variance_cm2(post_var)

    # --- ceiling: the lowest thing overhead (§7.1's clearance bit) -----------
    soa["ceiling_height"][slots] = np.minimum(
        soa["ceiling_height"][slots], aggregate.ceiling_cm
    )

    # --- occupancy (§10.1): hits only ---------------------------------------
    # A return is evidence of occupancy. Evidence of FREE comes from beams that
    # passed through, which is the visibility pass (§10.4) -- not from the
    # absence of a return here, which is mostly just the 1-2% fill rate of §1.3.
    lo, hi = occ["log_odds_clamp"]
    soa["log_odds"][slots] = np.clip(
        soa["log_odds"][slots].astype(np.int32) + occ["log_odds_hit"], lo, hi
    ).astype(np.int8)

    # --- class (§10.2) ------------------------------------------------------
    soa["semantic_class"][slots] = boyer_moore_update(
        soa["semantic_class"][slots], aggregate.class_id
    )

    # --- reflectivity (§10.3) -----------------------------------------------
    # This frame's mean, not a running one. rho-hat is already normalised for
    # range and incidence, so what is left is a property of the surface NOW --
    # and the use §10.3 puts it to, wet asphalt returning almost nothing on a
    # cell classified road, is a statement about the current surface, which an
    # average over the whole sequence would wash out.
    n = np.maximum(aggregate.n.astype(np.int32), 1)
    soa["reflectivity"][slots] = np.clip(aggregate.refl_sum // n, 0, 255).astype(np.uint8)

    # --- counts -------------------------------------------------------------
    soa["obs_count"][slots] = np.minimum(
        soa["obs_count"][slots].astype(np.int32) + aggregate.n, OBS_COUNT_MAX
    ).astype(np.uint8)
    soa["frames_since_seen"][slots] = 0


# --- §10.2: Boyer-Moore majority in one byte ---------------------------------


def unpack_class(packed):
    """One byte -> (candidate, counter)."""
    p = np.asarray(packed, dtype=np.uint8)
    return (p >> COUNTER_BITS).astype(np.uint8), (p & COUNTER_MAX).astype(np.uint8)


def pack_class(candidate, counter):
    """(candidate, counter) -> one byte."""
    c = np.asarray(candidate, dtype=np.uint8)
    k = np.asarray(counter, dtype=np.uint8)
    if np.any(c > CLASS_MAX) or np.any(k > COUNTER_MAX):
        raise ValueError(f"candidate and counter must both fit in {CLASS_BITS} bits")
    return ((c << COUNTER_BITS) | k).astype(np.uint8)


def boyer_moore_update(packed, observed):
    """One streaming-majority step per cell, vectorised. Math §10.2.

        counter == 0      -> candidate <- y, counter <- 1
        y == candidate    -> counter <- min(counter + 1, COUNTER_MAX)
        otherwise         -> counter <- counter - 1

    Returns the new packed bytes; does not write. Guaranteed to hold the true
    majority class whenever one exists, in constant memory, and the counter
    doubles as a confidence readout.

    The candidate is 5 bits since 1 Sep, so the whole 0-19 label set fits and
    ids up to 31 are accepted. Anything above that is still rejected rather
    than wrapped: a silent `% 32` would relabel class 32 as 0, and 0 is
    `unlabeled` -- a chunk of the map quietly becoming unlabelled ground is
    plausible-looking and undetectable without the reference map. Loud failure
    at the seam is the safer of the two. See the header for why 5 | 3.
    """
    observed = np.asarray(observed, dtype=np.uint8)
    if np.any(observed > CLASS_MAX):
        raise ValueError(
            f"class id {int(observed.max())} does not fit in {CLASS_BITS} bits "
            f"(max {CLASS_MAX}). The SemanticKITTI label set stops at 19, so an "
            "id above 31 means the caller is passing RAW label ids (10, 11, 40, "
            "252, ...) where learning ids (0-19) are expected -- see "
            "`perception.semantics.semantic_labels`."
        )
    cand, cnt = unpack_class(packed)
    cnt = cnt.astype(np.int16)

    empty = cnt == 0
    match = (~empty) & (cand == observed)

    new_cand = np.where(empty, observed, cand).astype(np.uint8)
    new_cnt = np.where(empty, 1,
                       np.where(match, np.minimum(cnt + 1, COUNTER_MAX), cnt - 1))
    return pack_class(new_cand, new_cnt.astype(np.uint8))


# --- §10.1: three-state occupancy --------------------------------------------


def new_occupancy_scratch(n_slots: int) -> dict:
    """Working set for `occupancy_state(out=...)`. 2 B per slot -- 1.82 MB over
    the 910,000 allocated slots, against 8.19 MB of churn per call."""
    return {"mask": np.zeros(n_slots, np.bool_),
            "byte": np.zeros(n_slots, np.uint8),
            "n_slots": int(n_slots)}


def occupancy_state(soa, thresholds=None, slots=None, out=None, scratch=None):
    """UNKNOWN / FREE / OCCUPIED for every cell, or for `slots`. Math §10.1.

        UNKNOWN   if n < n_min          <- observation count, NOT log-odds
        OCCUPIED  if l > l_occ
        FREE      otherwise

    "I looked and it's empty" and "I couldn't see" are different facts and a
    planner must treat them differently; a log-odds value near zero conflates
    them, which is why the count is consulted first.

    FLAG_BLIND short-circuits to UNKNOWN whatever the log-odds say. The blind
    cone is 3.74 m of ground the sensor cannot see in any single frame (§1.4),
    and a cell there accumulating free-space evidence from a later frame's
    geometry must not be allowed to report FREE on the strength of it.

    ⚑ Pass `out` and `scratch` on the frame path. Without them this allocates
      **8.19 MB per call** over 910,000 slots, and the frame loop calls it
      every frame: the visibility cleanup's candidate set is the currently
      OCCUPIED cells, and this is the only thing that computes them.

      The cost is not obvious from reading it. `np.where(cond, OCC, FREE)`
      picks between two Python ints, so numpy chooses int64 -- one int64 array
      across the grid is 7.28 MB on its own -- and the chain built three of
      them plus a bool and two uint8 casts, for a uint8 answer. Writing
      through `out` keeps every intermediate at its natural width.

      `new_occupancy_scratch(n_slots)` sizes the scratch; `out` is uint8 of
      the same length as the selection. Both optional, because hundreds of
      unit tests call this once on a throwaway grid and should not have to.
    """
    th = thresholds if thresholds is not None else load_thresholds()
    occ = th["occupancy"]
    l_occ = occ.get("log_odds_occupied", 0)

    sel = slice(None) if slots is None else np.asarray(slots, dtype=np.int64)
    n = soa["obs_count"][sel]
    log_odds = soa["log_odds"][sel]
    flags = soa["flags"][sel]

    size = len(n)
    if out is None:
        out = np.empty(size, np.uint8)
    elif len(out) < size:
        raise ValueError(f"out holds {len(out)} of {size} slots")
    out = out[:size]

    if scratch is None or scratch["n_slots"] < size:
        mask = np.empty(size, np.bool_)
        byte = np.empty(size, np.uint8)
    else:
        mask, byte = scratch["mask"][:size], scratch["byte"][:size]

    # Order matters and is the section's, not an optimisation: the count wins
    # over the log-odds, and the blind flag wins over both.
    out[:] = OCC_FREE
    np.greater(log_odds, l_occ, out=mask)
    np.copyto(out, OCC_OCCUPIED, where=mask)

    np.less(n, occ["unknown_below_obs"], out=mask)
    np.copyto(out, OCC_UNKNOWN, where=mask)

    np.bitwise_and(flags, FLAG_BLIND, out=byte)
    np.not_equal(byte, 0, out=mask)
    np.copyto(out, OCC_UNKNOWN, where=mask)
    return out


def is_blind(x_m, y_m, thresholds=None):
    """Inside the blind cone? Math §1.4 eq. (5): r_blind = h_s/tan|phi_min|.

    3.74 m for the HDL-64E at 1.73 m, which is 11% of ring 0 and unobservable
    in any single frame. Ego-motion fills most of it, so a cell here is not
    permanently blind -- it is blind until driven over, which is why this is a
    flag on a cell rather than a hole in the map.
    """
    th = thresholds if thresholds is not None else load_thresholds()
    r = th["sensor"]["blind_cone_m"]
    return np.hypot(np.asarray(x_m, dtype=np.float64),
                    np.asarray(y_m, dtype=np.float64)) < r


def mark_blind(soa, slots) -> None:
    """Set FLAG_BLIND on `slots`. Unknown, never free (CLAUDE.md, §10.1)."""
    soa["flags"][np.asarray(slots, dtype=np.int64)] |= FLAG_BLIND


def scatter(gm, points_m, class_id, is_ground, reflectivity=None,
            points_world_m=None):
    """One scan into the variable-resolution grid. Math §3, master v4 §3.5.

    ⚑ TWO frames, and they are not interchangeable. This is the single most
      confusable thing in the file, so it is spelled out rather than implied:

      `points_m`       VEHICLE frame (x forward, y left, z up). Decides the
                       RING, because foveation follows the vehicle, and the
                       measurement variance, because that depends on range
                       from the sensor.
      `points_world_m` WORLD frame. Decides the CELL, because cell identity is
                       world-anchored -- that is the whole reason the toroidal
                       shift exists (§2.4). Defaults to `points_m`, which is
                       the stationary case and the one the unit tests use.

      Get this backwards and the map still builds, still looks plausible, and
      smears six frames of a moving vehicle onto one patch of ground. Ring
      membership is relative; cell identity is absolute.

    The pose composition that produces `points_world_m` is
    `perception.transforms`, JP's; this function starts where the frames are
    already right, because ring assignment and lattice indexing are the parts
    that are mine and they do not depend on how the points arrived.

    Composition only. Ring from §6, cell index from the one base lattice of
    §2, slot from the ring's toroidal window, weights from §3.2, reduction by
    the kernel that guarantees §3.4. Nothing here recomputes any of those --
    that is the rule the lattice file exists to enforce.

    Returns the CellAggregate, so a caller can fuse it, hash it, or throw it
    away without this function deciding. ⚑ Its arrays are VIEWS into the
    allocation's scatter scratch and are valid only until the next scatter on
    the same scratch -- `fuse()` consumes it inside the frame, and anything
    that must outlive the frame has to copy it.
    """
    from vrgrid.gpu.kernels import (
        measurement_variance_cm2,
        quantise_height,
        quantise_weight,
        scatter_atomic,
        scatter_sorted,
    )
    from vrgrid.grid.lattice import bin_points

    pts = np.asarray(points_m, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3) in vehicle frame, got {pts.shape}")
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    world = pts if points_world_m is None else np.asarray(points_world_m, dtype=np.float64)
    if world.shape != pts.shape:
        raise ValueError(f"world points {world.shape} do not match {pts.shape}")
    wx, wy = world[:, 0], world[:, 1]

    # One binning function, in the lattice, where the semantics live. This
    # was composed by hand here -- ring_of, then i_ring per ring, then
    # flat_slot -- and in three other places, and the four spellings agreeing
    # was luck rather than design. See `lattice.bin_points`.
    scratch, out = gm.bin_scratch(pts.shape[0])
    slots = bin_points(x, y, wx, wy, gm.schedule, gm.buffers, out, scratch,
                       gm.speed_ms).copy()

    # OUTSIDE and out-of-window both come through as -1, which the kernel
    # drops. It must be -1 and not 0: numpy would write cell 0 and pile the
    # far field onto the origin. `bin_points` returns -1 for both cases, so
    # the separate OUTSIDE mask this used to apply is gone with it.

    ranges = np.linalg.norm(pts, axis=1)
    w_q = quantise_weight(measurement_variance_cm2(ranges))
    refl = (np.zeros(pts.shape[0], dtype=np.int32) if reflectivity is None
            else np.asarray(reflectivity, dtype=np.int32))

    # The scratch from allocate(), not a private one. Omitting it is legal and
    # allocates per call -- fine in a test, and 19 MB a frame in the loop,
    # which is more than the whole grid. See gpu/CLAUDE.md.
    scratch = getattr(getattr(gm, "allocation", None), "scratch", None)
    args = (slots, quantise_height(z), w_q, refl,
            np.asarray(class_id, dtype=np.uint8), np.asarray(is_ground, dtype=bool))
    if gm.scatter_mode == "sorted":
        return scatter_sorted(*args, scratch=scratch)
    return scatter_atomic(*args, n_cells=gm.soa["ground_height"].size,
                          scratch=scratch)


# §10.4 visibility cleanup does NOT live here. It is
# `vrgrid.gpu.visibility.visibility_cleanup` — implemented, tested, benchmarked
# and wired into the frame loop by `run/engine.py`.
#
# It was declared here as well, as a stub with a Day-3 owner on it, and the two
# survived side by side long enough for a second reader to find this one first
# and conclude the ghost gate was unbuilt. The section is a range-image
# comparison, so it belongs in the module that owns range-image kernels; what
# is fusion's, and is below, is what a miss MEANS — `apply_miss` walks the
# log-odds and `occupancy_state` (§10.1) turns them into a three-state answer.
#
# Deleted rather than left raising: a NotImplementedError with a name on it
# reads as work outstanding, and this was work already done.
