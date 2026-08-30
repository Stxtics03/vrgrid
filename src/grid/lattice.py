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

    Newly exposed cells are zeroed, which is the correct UNKNOWN state rather
    than a convenient one: obs_count = 0, and §10.1 decides unknown by
    observation count, never by log-odds near zero.

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
    """Zero `width` wrapped rows (axis 0) or columns (axis 1) of one ring."""
    if width <= 0:
        return 0
    if width >= n:
        for name, _ in CELL_FIELDS:
            soa[name][ring_slice(schedule, level)] = 0
        return n * n

    idx = np.arange(start, start + width) % n
    sl = ring_slice(schedule, level)
    for name, _ in CELL_FIELDS:
        view = soa[name][sl].reshape(n, n)
        if axis == 0:
            view[idx, :] = 0
        else:
            view[:, idx] = 0
        soa[name][sl] = view.reshape(-1)
    return width * n
