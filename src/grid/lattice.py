"""The integer lattice. Math §2. [Aakash — Day 0/1, first task]

Everything downstream indexes through this file, which is why it is the first
thing built and the partition test is CI-blocking.

The rule, and it has no exceptions: there is ONE lattice, at the base
resolution c0 = 5 cm. Coarser ring indices are derived from it by integer
division, never recomputed in floating point.

    i_fine(x) = floor(x / c0)
    i_L(x)    = floor(i_fine(x) / k_L),   k_L = c_L / c0 in Z+

Theorem (nested floor, math §2.2): floor(floor(x/c0)/k) == floor(x/(k*c0)).
So the ring-L lattice IS the direct lattice of size k*c0 -- the rings partition
the plane exactly. There is no tolerance to tune and no epsilon.

Computing floor(x/0.20) directly instead is the bug this file exists to
prevent: 0.2 is not representable in binary, the two lattices drift apart, and
near a boundary a point falls in both cells or neither.
"""

import numpy as np
from vrgrid.cell import CELL_FIELDS, alloc_soa
from vrgrid.gpu.allocators import EMPTY_CELL

# A point beyond the last ring is not in the map. It is not ring 0 either.
OUTSIDE = -1

# The rear resolution floor applies within this range behind the vehicle.
# Math §6.2: "c_L <= 0.20 m whenever x < 0 and |x| < 50".
REAR_FLOOR_RANGE_M = 50.0


def stretch_factors(schedule, speed_ms: float):
    """(a_f, a_s, a_r) of math §6.2 eq. (20).

    a_f stretches the map forward with speed (clamped at 2x), a_s squeezes it
    laterally, a_r never stretches. At v = 0 all three are 1 and (20) collapses
    to the plain Chebyshev norm of (18) -- which is why there is no separate
    isotropic code path to keep in sync.
    """
    a = schedule.anisotropy
    t = speed_ms / a.v_ref_ms
    a_f = min(max(1.0 + a.kappa_forward * t, 1.0), 2.0)
    a_s = 1.0 / (1.0 + a.kappa_side * t)
    return a_f, a_s, a.rear_stretch


def d_aniso(x, y, schedule, speed_ms: float = 0.0):
    """Scaled L-infinity distance. Math §6.2 eq. (20).

        d = max( x+/a_f,  x-/a_r,  |y|/a_s )

    Note the direction of the lateral term: a_s < 1, so |y|/a_s > |y|. Points
    to the side are pushed OUT to a coarser ring -- the resolution is taken
    from the sides and spent forward, which is the whole idea.
    """
    a_f, a_s, a_r = stretch_factors(schedule, speed_ms)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    forward = np.maximum(x, 0.0) / a_f
    rear = np.maximum(-x, 0.0) / a_r
    side = np.abs(y) / a_s
    return np.maximum(np.maximum(forward, rear), side)


def _rear_floor_ring(schedule):
    """Coarsest ring whose cell still satisfies the rear floor, or None if the
    floor never binds for this schedule (the ablation's 50 cm ring is its own
    floor, so nothing is clamped)."""
    floor_m = schedule.anisotropy.rear_floor_cell_m
    allowed = [r.ring for r in schedule.rings if r.cell_m <= floor_m + 1e-9]
    if not allowed or len(allowed) == len(schedule.rings):
        return None
    return max(allowed)


def i_fine(x: float, base_cell_m: float) -> int:
    """Base-lattice index. Math §2.1 eq. (8).

    Floor, not truncation: `int(x / c0)` rounds toward zero, so -0.02 and
    +0.02 both land in cell 0 and the cell straddling the origin comes out
    twice the size of every other cell. Floor division is correct on both
    sides of zero.

    Scalar in -> int out; ndarray in -> int64 array out. Both paths are the
    same floor-division operator, so the vectorised path (scatter) and the
    scalar path (query) cannot answer differently.
    """
    q = x // base_cell_m
    return int(q) if np.ndim(q) == 0 else q.astype(np.int64)


def i_ring(x: float, base_cell_m: float, k: int) -> int:
    """Ring index, by integer division from the base lattice. Math §2.1 eq. (9).

    Deliberately NOT floor(x / (k * base_cell_m)). Theorem §2.2 proves the two
    agree in exact arithmetic; in IEEE-754 they are two lattices that drift
    apart, and near a boundary a point falls in both cells or in neither.
    Derive, never recompute.

    `k` must be a positive integer -- the same hard rule `schedule.validate()`
    enforces on the config. Powers of two are a convenience (the divide is a
    bit shift), not a requirement.
    """
    if k != int(k) or k < 1:
        raise ValueError(f"k must be a positive integer (math §2.1), got {k!r}")
    return i_fine(x, base_cell_m) // int(k)


def ring_of(x: float, y: float, schedule, speed_ms: float = 0.0) -> int:
    """Which ring a point falls in, after anisotropic stretch (master v4 §3.2).

    Anisotropy changes ring MEMBERSHIP only. Every cell stays on the same base
    5 cm lattice, so nesting and alignment are untouched -- say this explicitly
    in the report, because it looks like it should break alignment.

    Math §6.1 eq. (18) with the scaled L-infinity norm of §6.2 eq. (20):

        L = min { L : d_aniso(x, y) < R_L }

    Returns OUTSIDE (-1) beyond the last ring. Callers must check: an
    out-of-map point is not ring 0, and silently clamping it there is how a
    100 m return ends up written into a 5 cm cell at the origin.

    ⚑ CONTAINMENT, and it constrains §6.2 more than the section admits. Ring L
    is a fixed square buffer of half-width R_L, so a point may only be assigned
    to a ring that physically contains it. Equation (20) is free to push a
    point OUTWARD to a coarser ring -- coarser rings are larger, so the result
    still fits -- but it cannot pull one inward past its geometric ring. At
    15 m/s the forward stretch a_f = 2 maps a return 58 m ahead to d = 29 and
    would file it under ring 2, whose buffer stops at 50 m; the index then
    wraps toroidally onto a cell on the far side of the map.

    So the lateral squeeze survives contact with fixed buffers and the forward
    stretch does not. Buying back the forward half means allocating each ring
    for its maximum stretch (a_f <= 2, so ~1.5x the cells), which is a memory
    decision, not a lattice one. Anisotropy is stretch item 12 and this is why
    it is last on the list.

    Scalar in -> int out; ndarray in -> int64 array out, so per-frame ring
    migration runs over a whole scan without a Python loop.
    """
    radii = np.array([r.half_width_m for r in schedule.rings], dtype=np.float64)
    x_a, y_a = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)

    # First ring whose half-width strictly exceeds the distance. searchsorted
    # with side="right" returns len(radii) for a point past the last ring.
    d = d_aniso(x, y, schedule, speed_ms)
    chebyshev = np.maximum(np.abs(x_a), np.abs(y_a))          # eq. (18), unstretched
    ring_aniso = np.searchsorted(radii, d, side="right")
    ring_geom = np.searchsorted(radii, chebyshev, side="right")

    # The containment rule above: never finer than geometry allows.
    ring = np.maximum(ring_aniso, ring_geom)

    # Hard rear floor (§6.2): never coarser than rear_floor_cell_m within 50 m
    # behind. Closing traffic is exactly where a coarse cell hurts, so the
    # stretch is taken from the sides, never from the back.
    #
    # Applied only where the point fits in the floor ring, for the same reason.
    # §6.2 states the condition as "x < 0 and |x| < 50", which a point at
    # (-10, -70) satisfies -- behind the vehicle, within 50 m longitudinally,
    # 70 m to the side -- and forcing that into ring 2 wraps it onto the cell
    # at +30 m, on the far side of the vehicle.
    floor_ring = _rear_floor_ring(schedule)
    if floor_ring is not None:
        rear = (x_a < 0.0) & (np.abs(x_a) < REAR_FLOOR_RANGE_M)
        ring = np.where(rear & (ring_geom <= floor_ring),
                        np.minimum(ring, floor_ring), ring)

    # A point that physically fits stays in the map even when the stretched
    # norm overflows past the last ring: at 15 m/s the squeeze sends (0, 70)
    # to d = 105, and dropping a return ring 3 has a cell for is pure loss.
    # Otherwise the map's lateral reach would silently shrink from 100 m to
    # 66.7 m as the vehicle speeds up.
    ring = np.minimum(ring, len(radii) - 1)
    ring = np.where(ring_geom >= len(radii), OUTSIDE, ring)
    return int(ring) if np.ndim(ring) == 0 else ring.astype(np.int64)


def migrate_ring(x, y, schedule, current_ring, speed_ms: float = 0.0):
    """Ring assignment WITH hysteresis, for per-frame migration. Math §6.3.

    A cell sitting exactly on a ring boundary while speed fluctuates would
    otherwise split and merge every frame: refinement-pool thrash, and by
    §5.4 unbounded variance inflation if the derived flag is ever cleared
    mid-cycle. So the thresholds are asymmetric (eq. 21):

        split (go finer)   when  d < R_L
        merge (go coarser) when  d > R_L (1 + eps)

    Between the two the cell stays where it is. `ring_of` is the eps = 0 case
    and is the right function for a fresh point with no history; this one is
    for a cell that already has a ring.
    """
    eps = schedule.hysteresis_eps
    d = d_aniso(x, y, schedule, speed_ms)
    radii = [r.half_width_m for r in schedule.rings]
    cur = int(current_ring)

    if cur == OUTSIDE:
        return ring_of(x, y, schedule, speed_ms)

    # Coarser only once past the OUTER edge of the current ring, widened by
    # eps. Finer as soon as the inner boundary is genuinely crossed.
    if d > radii[cur] * (1.0 + eps):
        target = ring_of(x, y, schedule, speed_ms)
        return target if target == OUTSIDE else max(target, cur + 1)
    if cur > 0 and d < radii[cur - 1]:
        return ring_of(x, y, schedule, speed_ms)
    return cur


# --- the frame path: zero-allocation binning, math §2.1 + §6.1 --------------
#
# `ring_of`, `d_aniso` and `i_ring` above are the reference implementations:
# scalars or arrays in, allocate freely, and they are what every test compares
# against. Everything below is their frame-loop twin -- the same arithmetic in
# the same order, with every intermediate preallocated. Bit-identity is pinned
# by `test_ring_of_into_matches_ring_of` and
# `test_bin_points_matches_the_reference_path` over both frozen schedules,
# several speeds, and both ring parities.
#
# ⚑ Why the twin exists, and it is not tidiness. `ring_of` allocates roughly
#   seven full-length float64 temporaries per call -- 6.96 MB per
#   120,000-point sweep, measured by `scripts/timing_table.py --alloc` --
#   against CLAUDE.md's "no allocation inside the frame loop", which is a hard
#   invariant and a sentence in the report. Binning is also the largest single
#   stage in the frame, larger than `fuse` and larger than `cleanup`.


def _bin_geometry(schedule):
    """Per-schedule constants `bin_points` would otherwise rebuild every frame:
    ring radii as one array, the integer k per ring, and the rear-floor ring.

    Small allocations, but they are allocations in the frame loop, and
    `_rear_floor_ring` builds a list comprehension per call. Baked into the
    scratch at startup instead.
    """
    radii = np.array([r.half_width_m for r in schedule.rings], dtype=np.float64)
    ks = []
    for r in schedule.rings:
        k = round(r.cell_m / schedule.base_cell_m)
        if abs(k * schedule.base_cell_m - r.cell_m) > 1e-9 or k < 1:
            raise ValueError(
                f"ring {r.ring}: cell {r.cell_m} m is not a positive integer "
                f"multiple of the base {schedule.base_cell_m} m (math §2.1). "
                "schedule.validate() should have rejected this."
            )
        ks.append(int(k))
    return radii, ks, _rear_floor_ring(schedule)


def new_bin_scratch(max_points: int, schedule) -> dict:
    """Working set for `bin_points`, sized at startup like every other
    frame-path buffer.

    Five int64 lanes, one float64 and two bool: **50 B per point**, 7.50 MB at
    the 150,000-point cap in
    `configs/thresholds.yaml: scatter.max_points_per_frame`. Of that, 3.75 MB
    is the `new_slot_scratch` the frame loop already allocates and which this
    replaces, so the new declared footprint is 3.75 MB.

    That is the trade, stated plainly: ~4 MB of declared startup footprint to
    remove 6.96 MB of undeclared per-frame churn. It is the trade
    `scatter_sorted` already made, and it is the right way round -- churn is
    invisible until someone profiles it, footprint is a number on a slide.

    The lanes are reused aggressively and the comments in `bin_points` say
    where, because five buffers doing eleven jobs is only safe if the handover
    points are written down. The caller's `out` array is used as a sixth lane
    until the final write, for the same reason.
    """
    radii, ks, floor_ring = _bin_geometry(schedule)
    n_rings = len(schedule.rings)
    return {
        "f0": np.zeros(max_points, np.float64),   # d_aniso temp, then xw/yw scaling
        "f1": np.zeros(max_points, np.float64),   # cheb, then d_aniso
        "level": np.zeros(max_points, np.int64),  # ring per point, whole pass
        "a": np.zeros(max_points, np.int64),      # clipped ring, then gathers
        "b": np.zeros(max_points, np.int64),      # k, then side
        "c": np.zeros(max_points, np.int64),      # ix, then col
        "d": np.zeros(max_points, np.int64),      # iy, then row, then the slot
        "live": np.zeros(max_points, np.bool_),
        "tmp": np.zeros(max_points, np.bool_),
        # per-ring tables, gathered per point. x0/y0 move with the vehicle
        # every frame, so they are refilled per call -- into these arrays,
        # never rebuilt.
        "t_k": np.array(ks, dtype=np.int64),
        "t_side": np.zeros(n_rings, np.int64),
        "t_x0": np.zeros(n_rings, np.int64),
        "t_y0": np.zeros(n_rings, np.int64),
        "t_off": np.zeros(n_rings, np.int64),
        "t_dx": np.zeros(n_rings, np.int64),      # x0 mod side, the wrap offset
        "t_dy": np.zeros(n_rings, np.int64),
        # per-schedule constants, so the frame loop never rebuilds them
        "radii": radii,
        "ks": ks,
        "floor_ring": floor_ring,
        "max_points": int(max_points),
    }


def _count_at_or_below(values, radii, out, mask) -> None:
    """`np.searchsorted(radii, values, side="right")` without the allocation.

    searchsorted has no `out=`, and it runs twice per frame over a full-length
    array -- 1.92 MB at 120,000 points. Its result is the count of radii that
    are `<= v`, so over four rings a branch-free accumulate is allocation-free
    and, at this length, no slower than a binary search per element.
    """
    out[:] = 0
    for r in radii:
        np.greater_equal(values, r, out=mask)
        # `np.add(out, mask, out=out)` reads more naturally and costs 64 kB a
        # call: adding a bool array to an int64 one is a mixed-dtype ufunc, so
        # numpy casts through its fixed internal buffer. Bounded rather than
        # per-point, but it is the last allocation on this path. A masked
        # scalar increment does the same arithmetic with no cast at all.
        np.add(out, 1, out=out, where=mask)


def d_aniso_into(x, y, schedule, speed_ms, out, tmp):
    """`d_aniso` with both temporaries supplied. Math §6.2 eq. (20).

    The three terms are combined in the same order as the reference, so the
    float64 result is bit-identical rather than merely close. That matters:
    the value is compared against a ring radius, and a point exactly on a
    boundary has to land in the same ring on both paths or the partition test
    is measuring two different maps.
    """
    a_f, a_s, a_r = stretch_factors(schedule, speed_ms)
    np.maximum(x, 0.0, out=out)                  # forward
    np.divide(out, a_f, out=out)
    np.negative(x, out=tmp)                      # rear
    np.maximum(tmp, 0.0, out=tmp)
    np.divide(tmp, a_r, out=tmp)
    np.maximum(out, tmp, out=out)
    np.abs(y, out=tmp)                           # side
    np.divide(tmp, a_s, out=tmp)
    np.maximum(out, tmp, out=out)
    return out


def ring_of_into(x, y, schedule, speed_ms, out, scratch):
    """`ring_of` over a whole sweep, allocation-free. Math §6.1 eq. (18).

    `x`, `y` are VEHICLE frame -- ring membership is a question about distance
    from the sensor. `out` is int64 and receives OUTSIDE (-1) beyond the last
    ring, exactly as the reference does; callers must check it.

    Every rule the reference applies is applied here, in the same order and for
    the same reason: containment (never finer than geometry allows), the rear
    floor (§6.2, and only where the point fits in the floor ring), and the
    keep-what-fits clamp. `ring_of`'s docstring is the one that explains why
    each exists; this is the one that runs.
    """
    n = len(x)
    radii, floor_ring = scratch["radii"], scratch["floor_ring"]
    f0, f1 = scratch["f0"][:n], scratch["f1"][:n]
    # Lane "a" holds the geometric ring here; `bin_points` reuses it for the
    # clamped ring index only after this function has returned.
    geom, mask, aux = scratch["a"][:n], scratch["tmp"][:n], scratch["live"][:n]

    # Geometry first and consumed into an integer, which frees f1 for d_aniso.
    np.abs(x, out=f1)
    np.abs(y, out=f0)
    np.maximum(f1, f0, out=f1)                       # eq. (18), unstretched
    _count_at_or_below(f1, radii, geom, mask)

    d_aniso_into(x, y, schedule, speed_ms, f1, f0)   # eq. (20)
    _count_at_or_below(f1, radii, out, mask)

    np.maximum(out, geom, out=out)                   # containment

    if floor_ring is not None:
        np.less(x, 0.0, out=mask)
        np.abs(x, out=f0)
        np.less(f0, REAR_FLOOR_RANGE_M, out=aux)
        np.logical_and(mask, aux, out=mask)
        np.less_equal(geom, floor_ring, out=aux)
        np.logical_and(mask, aux, out=mask)
        np.minimum(out, floor_ring, out=out, where=mask)

    np.minimum(out, len(radii) - 1, out=out)
    np.greater_equal(geom, len(radii), out=mask)
    np.copyto(out, OUTSIDE, where=mask)
    return out


def _fill_ring_tables(buffers, scratch) -> None:
    """Per-ring constants as arrays indexed by ring, refreshed every call.

    `x0`/`y0` are the only state a toroidal shift changes (§2.4), so they move
    under us every frame and cannot be baked at startup like `k`. Written into
    preallocated arrays element by element -- a four-element Python loop, no
    array construction.
    """
    for L, buf in enumerate(buffers):
        W = buf.side
        scratch["t_side"][L] = W
        scratch["t_x0"][L] = buf.x0
        scratch["t_y0"][L] = buf.y0
        scratch["t_off"][L] = buf.offset
        scratch["t_dx"][L] = buf.x0 % W
        scratch["t_dy"][L] = buf.y0 % W


def bin_points(xv, yv, xw, yw, schedule, buffers, out, scratch,
               speed_ms: float = 0.0):
    """Points -> flat storage slots, in one place. Math §2.1, §2.4, §6.1.

    The stage between perception and the grid: ring membership, then the
    lattice index, then the slot in that ring's toroidal window. It runs every
    frame on every return, it is the largest single stage in the frame, and
    until now no module exported it -- it was composed by hand in
    `fusion.scatter`, `grid/transient.py`, `run/engine.py` and
    `scripts/timing_table.py`. Four spellings of one step in three
    directories, which will disagree eventually, and a binning bug does not
    crash: it produces a plausible map.

    ⚑ TWO frames, and mixing them is the whole difficulty -- the same trap
      `scatter` warns about, arriving one layer earlier.

      `xv, yv`  VEHICLE frame. Decides the RING, because foveation follows the
                vehicle and ring membership is distance from the sensor (§6.1).
      `xw, yw`  WORLD frame. Decides the CELL, because cell identity is
                absolute -- that is the entire reason the toroidal shift
                exists (§2.4).

      Feed world coordinates to `ring_of` and every point reads as OUTSIDE once
      the vehicle has driven past the last ring's half-width. Feed vehicle
      coordinates to `i_ring` and the map slides along under the vehicle
      instead of staying put. Both look correct for the first few seconds, and
      the first makes the latency table read BETTER -- everything bins to -1,
      so scatter and fuse post sub-millisecond medians for doing nothing.

    **No ring loop.** The obvious shape is a pass per ring over the points that
    fall in it, and that is what all four hand-rolled copies did. It needs the
    selected world coordinates compacted into a buffer, and numpy will not do
    that without allocating: `np.compress(..., out=)` still built 1.54 MB of
    internal index at 96,000 selected points. So every per-ring constant --
    `k`, `side`, `x0`, `y0`, `offset` -- is instead gathered per POINT by ring
    index, and the sweep is binned in one pass of full-length ufuncs. That is
    allocation-free, and it is also less arithmetic than four iterations plus
    four compacting copies.

    Bit-identical to `ring_of` + `i_ring` + `RingBuffer.flat_slot`, pinned by
    `test_bin_points_matches_the_reference_path` over both frozen schedules and
    four speeds. It has to be: those are what the partition test proves things
    about, and a second lattice is what this module's header forbids.

    Returns a view of `out` of length len(xv): the flat slot per point, and -1
    for anything outside the map or outside its ring's window. Allocates
    nothing; `scratch` comes from `new_bin_scratch`.
    """
    n = len(xv)
    if n > scratch["max_points"]:
        raise ValueError(
            f"{n} points exceeds the {scratch['max_points']} this scratch was "
            "built for (configs/thresholds.yaml: scatter.max_points_per_frame). "
            "Sizing the frame path at startup is the point -- growing it here "
            "would allocate inside the frame loop."
        )
    _fill_ring_tables(buffers, scratch)

    level = scratch["level"][:n]
    ring_of_into(xv, yv, schedule, speed_ms, level, scratch)

    c0 = schedule.base_cell_m
    f = scratch["f0"][:n]
    lv, kk = scratch["a"][:n], scratch["b"][:n]
    ix, iy = scratch["c"][:n], scratch["d"][:n]
    live, aux = scratch["live"][:n], scratch["tmp"][:n]
    gather = out[:n]                 # the sixth lane, free until the last write

    # OUTSIDE (-1) would index the tables from the back, so clamp for the
    # gather and mask those points out at the end instead.
    #
    # ⚑ Every `np.take` below passes `mode="clip"`, and not for the clipping.
    #   numpy's default `mode="raise"` performs its bounds check by building a
    #   full-length index array -- 0.96 MB per call, six calls a frame, which
    #   is most of what this rewrite exists to remove. `clip` skips that and is
    #   also 5x faster. It is safe only because `lv` is clamped to [0, n_rings)
    #   on the line above, so no index is ever out of range and clipping never
    #   actually clips; if that clamp is ever removed, this silently reads the
    #   wrong ring instead of raising.
    np.maximum(level, 0, out=lv)
    np.take(scratch["t_k"], lv, out=kk, mode="clip")

    # §2.1 eq. (9): floor to the ONE base lattice, then integer-divide by k.
    # Never floor(x / (k*c0)) -- see this module's header.
    np.floor_divide(xw, c0, out=f)
    np.copyto(ix, f, casting="unsafe")          # integer-valued float -> int64
    np.floor_divide(ix, kk, out=ix)
    np.floor_divide(yw, c0, out=f)
    np.copyto(iy, f, casting="unsafe")
    np.floor_divide(iy, kk, out=iy)

    # k is spent; the lane becomes the ring side, which is needed to the end.
    W = kk
    np.take(scratch["t_side"], lv, out=W, mode="clip")

    # in_view: x0 <= ix < x0 + W, and the same for iy (§2.4). Computed as
    # col = ix - x0 in [0, W), which is also what the wrap below needs.
    np.take(scratch["t_x0"], lv, out=gather, mode="clip")
    np.subtract(ix, gather, out=ix)
    np.greater_equal(ix, 0, out=live)
    np.less(ix, W, out=aux)
    np.logical_and(live, aux, out=live)

    np.take(scratch["t_y0"], lv, out=gather, mode="clip")
    np.subtract(iy, gather, out=iy)
    np.greater_equal(iy, 0, out=aux)
    np.logical_and(live, aux, out=live)
    np.less(iy, W, out=aux)
    np.logical_and(live, aux, out=live)

    # The wrap, division-free, exactly as `flat_slot_into` argues it: in view,
    # (x0 + col) mod W == (x0 mod W) + col, minus W once if that overflowed,
    # since both terms are in [0, W) and their sum is in [0, 2W).
    np.take(scratch["t_dx"], lv, out=gather, mode="clip")
    np.add(ix, gather, out=ix)
    np.greater_equal(ix, W, out=aux)
    np.subtract(ix, W, out=ix, where=aux)

    np.take(scratch["t_dy"], lv, out=gather, mode="clip")
    np.add(iy, gather, out=iy)
    np.greater_equal(iy, W, out=aux)
    np.subtract(iy, W, out=iy, where=aux)

    np.multiply(iy, W, out=iy)
    np.add(iy, ix, out=iy)
    np.take(scratch["t_off"], lv, out=gather, mode="clip")
    np.add(iy, gather, out=iy)

    # A point past the last ring is not in ring 0, and must not take ring 0's
    # slot -- silently clamping it there is how a 100 m return ends up written
    # into a 5 cm cell at the origin.
    np.greater_equal(level, 0, out=aux)
    np.logical_and(live, aux, out=live)

    idx = out[:n]
    np.copyto(idx, iy, where=live)
    np.logical_not(live, out=aux)
    np.copyto(idx, -1, where=aux)
    return idx


# --- toroidal ring buffers, math §2.4 ---------------------------------------


def ring_extent(schedule, ring: int) -> int:
    """N_L: cells per side of ring L's square buffer.

    Careful with the notation: math §2.4 uses N_L for this LINEAR extent
    (Ring 3 -> 500), while §6.1 eq. (19) uses N_L for an annulus cell COUNT
    (Ring 3 -> 187,500). They are different numbers and the docs reuse the
    symbol. This function is the §2.4 one.
    """
    r = schedule.rings[ring]
    n = 2.0 * r.half_width_m / r.cell_m
    if abs(n - round(n)) > 1e-9:
        raise ValueError(
            f"ring {ring}: extent {2 * r.half_width_m} m / {r.cell_m} m = {n} "
            "is not a whole number of cells"
        )
    return round(n)


def ring_slice(schedule, ring: int) -> slice:
    """Where ring L's square buffer lives in the flat SoA arrays."""
    start = sum(ring_extent(schedule, level) ** 2 for level in range(ring))
    return slice(start, start + ring_extent(schedule, ring) ** 2)


def buffer_cells(schedule) -> int:
    """Total cells actually ALLOCATED: sum of N_L^2 over rings.

    This is NOT schedule.total_cells. That figure counts square annuli
    (eq. 19), which is the map's logical extent; a toroidal ring buffer has
    to store the full square per ring because the hole moves through the
    buffer as the vehicle drives. See the note in tests/test_lattice.py --
    the difference is a headline memory number and it is Shrestha's call.
    """
    return sum(ring_extent(schedule, level) ** 2 for level in range(len(schedule.rings)))


def alloc_ring_buffers(schedule) -> dict:
    """Preallocate every ring buffer plus its origin bookkeeping, once.

    A stand-in for gpu.allocators.allocate() so the grid is not blocked on
    it; it wraps the frozen alloc_soa() and adds nothing to the cell struct.
    `ring_origin[L]` is the world index of the first row/column of ring L's
    window -- the offsets o_L of eq. (10), kept as state instead of moving
    data.
    """
    soa = alloc_soa(buffer_cells(schedule))
    soa["ring_origin"] = np.zeros((len(schedule.rings), 2), dtype=np.int64)
    return soa


def toroidal_shift(soa, schedule, delta_cells) -> int:
    """Ego-motion shift of the ring buffers, O(perimeter) clear, in place.

    Math §2.4. Nothing is copied: the window slides by advancing the offset
    of eq. (10), and only the newly exposed strip is cleared. Ring 3 clears
    2N = 1,000 cells per unit step instead of N^2 = 250,000 -- the difference
    between a sub-millisecond shift and a 40 ms stall.

    `delta_cells` is (dx, dy) in COARSEST cells, which is the §2.4 constraint:
    the origin may only move in whole coarsest-cell steps (40 cm), or every
    ring boundary shifts by a fraction of a cell and you have to resample --
    precisely the "data loss during projection" the brief warns about. Finer
    rings therefore shift by a whole multiple, k_coarsest / k_L, which is an
    integer because the schedule validator demands integer ratios.

    Newly exposed cells get `allocators.EMPTY_CELL`, the same state
    `allocate()` starts the map in -- NOT raw zeros. Nine of the ten fields
    really are zero when empty (obs_count = 0, and §10.1 decides unknown by
    observation count, never by log-odds near zero); `ceiling_height` is the
    one whose empty value is a sentinel. Zeroing it says "something solid at
    the ground datum", so `ceiling - ground < h_vehicle` holds across the
    whole strip and §7.1 bit 0 marks it untraversable forever -- nothing ever
    raises a ceiling back up. `gpu.shift.shift()` has always filled this way;
    this function zeroed instead, so the two spellings of one operation
    disagreed about what an empty cell is.

    Returns the number of cells cleared, so the O(perimeter) claim is
    measurable rather than asserted.
    """
    dx, dy = (int(delta_cells[0]), int(delta_cells[1]))
    k_coarsest = schedule.k(len(schedule.rings) - 1)
    origins = soa["ring_origin"]
    cleared = 0

    for level in range(len(schedule.rings)):
        n = ring_extent(schedule, level)
        scale = k_coarsest // schedule.k(level)  # integer by validate()
        step = (dx * scale, dy * scale)

        # `axis` here is a WORLD axis: 0 is x, 1 is y, matching the (dx, dy) of
        # `delta_cells` and the (x0, y0) columns of `ring_origin`. It is not a
        # numpy axis -- see `_clear_strip`, which is where the two meet.
        for axis in (0, 1):
            s = step[axis]
            if s == 0:
                continue
            # Newly exposed slots are [origin + min(s,0), +|s|) mod N, for
            # either sign -- the window's leading edge in the direction of
            # travel maps onto the slots the trailing edge just vacated.
            start = int(origins[level, axis]) + min(s, 0)
            cleared += _clear_strip(soa, schedule, level, n, start, abs(s), axis)
            origins[level, axis] += s

    return cleared


def _clear_strip(soa, schedule, level, n, start, width, axis) -> int:
    """Empty `width` wrapped lattice lines of one ring, world `axis` 0 = x.

    ⚑ The world axis and the numpy axis are TRANSPOSED, and getting it wrong
      is silent. A slot is `iy * W + ix` (`bin_points`, `RingBuffer.slot`), so
      `reshape(n, n)[a, b]` is `[iy, ix]`: moving in **x** exposes a strip of
      constant `ix`, which is a numpy **column**, `view[:, idx]`. This
      function ran the two the other way round, so a pure +x shift cleared a
      row of constant y -- it kept stale cells in the strip the window had
      actually just uncovered and wiped live ones on an edge that had not
      moved. Every test shifted and then un-shifted by the same vector, which
      is symmetric in the bug, so nothing caught it.

    Filled with `EMPTY_CELL`, not zeros: see `toroidal_shift`.
    """
    if width <= 0:
        return 0
    sl = ring_slice(schedule, level)
    if width >= n:
        for name, _ in CELL_FIELDS:
            soa[name][sl] = EMPTY_CELL.get(name, 0)
        return n * n

    idx = np.arange(start, start + width) % n
    for name, _ in CELL_FIELDS:
        view = soa[name][sl].reshape(n, n)
        if axis == 0:
            view[:, idx] = EMPTY_CELL.get(name, 0)   # x moved -> a column strip
        else:
            view[idx, :] = EMPTY_CELL.get(name, 0)   # y moved -> a row strip
        soa[name][sl] = view.reshape(-1)
    return width * n
