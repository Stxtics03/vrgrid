"""The conservative pyramid. Math §7.2-7.3. [Shrestha]

A 4-ary pyramid over each ring's window, so a planner can ask about a whole
block of the map at once and get an answer that is *provably* not optimistic.
Coarse queries become cheap, and fine resolution is paid for only where the
coarse answer is genuinely ambiguous.

Per node, over the block B it covers:

    H_max(B) = max ground        H_min(B) = min ground
    C_min(B) = min ceiling       n_min(B) = min observation count
    AND_mask(B) = AND of traversability bitfields
    OR_mask(B)  = OR  of traversability bitfields

**Not means.** Averaging heights hides hazards: a coarse cell straddling a
kerb reports the mean and looks flat. Max/min preserves the worst case, which
is the entire reason this structure can carry a safety claim.

Every reduction is an integer max, min, AND or OR. All four are exactly
associative, so the pyramid is bit-identical run to run for the same input
without any of the care §3.4 needs -- there is no float anywhere on this path
and there must never be one.

--- three predicates, and they do NOT mean the same thing -------------------

This is the part to get right before using any of it, because all three
sound like "safe" in English and one of them is a much weaker claim.

`theorem3_safe(B)`  Math §7.3 exactly as written: bits 0 (clearance), 2 (step)
                    and 5 (confidence) are clear for EVERY cell in B. It says
                    nothing about slope, roughness or class. A planner that
                    reads this as "drivable" will drive onto a 30 degree bank.
                    It is proved from the raw quantities rather than from the
                    per-cell bitfield, which is what makes it re-evaluable
                    against a different vehicle without rebuilding anything.

`all_clear(B)`      OR_mask == 0: every cell in B is traversable on all SIX
                    conditions. This is the one that answers "may I drive
                    here". Strictly stronger than the above.

`certainly_blocked(B)`  AND_mask != 0: every cell in B fails the same
                    condition, so no route through B exists at any resolution.

`classify()` returns the frozen `QueryLOD` and uses `all_clear` for SAFE,
because `api.QueryLOD.SAFE` says "every cell in the block is traversable" and
that is the reading a caller will take. The two cannot collide: OR_mask == 0
implies AND_mask == 0.

--- note, 29 Aug: two things §7.2 does not say ------------------------------

**OR_mask is not in §7.2's list; it is added here.** Without it the only
available notion of SAFE is Theorem 3's, which covers three of the six bits,
and every consumer would have to remember that. One byte per node buys a
predicate that means what its name says. §7.3's theorem is untouched and is
still tested exactly as stated.

**⚑ §7.2's memory figure is low by about half.** It says "5 bytes of the 12"
and computes 745,000 x 5 / 3 ~= 1.24 MB. Two things are off. A node does not
store the source fields, it stores the reductions: ground contributes BOTH
H_max and H_min, so it is 4 bytes and not 2, and n_min adds another -- 8 bytes
by §7.2's own list, 9 with OR_mask. And the pyramid is built over the ring
WINDOWS, which are the 910,000 allocated slots, not the 745,000 logical cells.
Measured by `pyramid_bytes()` on the default schedule: **2.73 MB** of nodes
plus 0.38 MB of reduction scratch, taking the preallocated total from 29.06 MB
to 32.17 MB. That is why `allocate()` does not switch it on by itself -- see
`with_pyramid` there. Raise the corrected figure at a gate
review before it reaches a slide.
"""

from dataclasses import dataclass

import numpy as np
from vrgrid.api import QueryLOD
from vrgrid.grid.schedule import load_thresholds

# dst field -> (level-0 source field, reduction, dtype).
#
# From level 1 upward a field reduces into itself: max of maxima is a maximum,
# min of minima a minimum, AND of ANDs an AND. That is exactly the property
# that makes a pyramid a pyramid, and it is why one table serves both cases.
REDUCTIONS = (
    ("h_max", "ground_height", np.maximum, np.int16),
    ("h_min", "ground_height", np.minimum, np.int16),
    ("c_min", "ceiling_height", np.minimum, np.int16),
    ("n_min", "obs_count", np.minimum, np.uint8),
    ("and_mask", "traversability", np.bitwise_and, np.uint8),
    ("or_mask", "traversability", np.bitwise_or, np.uint8),
)

NODE_BYTES = sum(np.dtype(dt).itemsize for _, _, _, dt in REDUCTIONS)


def level_sides(side: int) -> list:
    """Window extent at each level, level 0 first, down to a single node.

    Halved with a CEILING, not a floor. Ring windows are 400 and 500 cells
    across and neither is a power of two, so a floor would drop the last row
    and column at every odd level -- silently, and at the map edge, where
    nothing would look wrong. Ceiling means the edge blocks are 1 wide instead
    of 2, which the reduction handles explicitly.
    """
    sides = [int(side)]
    while sides[-1] > 1:
        sides.append((sides[-1] + 1) // 2)
    return sides


def pyramid_bytes(rings) -> int:
    """Nodes only, excluding the reduction scratch. The number for the budget."""
    return NODE_BYTES * sum(
        sum(s * s for s in level_sides(r.side)[1:]) for r in rings)


def scratch_bytes(rings) -> int:
    """The two-pass reduction's intermediate, shared across rings and levels."""
    widest = max(r.side for r in rings)
    elems = widest * ((widest + 1) // 2)
    return elems * (np.dtype(np.int16).itemsize + np.dtype(np.uint8).itemsize)


@dataclass
class Pyramid:
    """Preallocated pyramid over a set of ring windows.

    `levels[ring][k]` is the dict of node arrays for level k+1 -- level 0 is
    NOT stored. It is the grid itself, and duplicating it would cost 8.2 MB to
    hold a copy of something already in memory. `level_arrays()` hands back
    either, with the same keys, so callers never special-case the base.
    """

    sides: list      # per ring: level_sides(ring.side)
    levels: list     # per ring: list of {field: array} for levels 1..L
    tmp: dict        # dtype -> flat scratch for the two-pass reduction

    def depth(self, ring: int) -> int:
        """Number of levels including level 0, so the coarsest is depth-1."""
        return len(self.sides[ring])

    def side(self, ring: int, level: int) -> int:
        return self.sides[ring][level]

    def nodes(self, ring: int, level: int) -> int:
        s = self.sides[ring][level]
        return s * s


def allocate_pyramid(rings) -> Pyramid:
    """Allocate every level for every ring, once, at startup.

    Called from `allocate()`; nothing here runs in the frame loop. `build()`
    then rewrites these buffers in place every frame and allocates nothing.
    """
    sides = [level_sides(r.side) for r in rings]
    levels = [
        [{name: np.zeros(s * s, dtype=dt) for name, _, _, dt in REDUCTIONS}
         for s in ring_sides[1:]]
        for ring_sides in sides
    ]
    widest = max(r.side for r in rings)
    elems = widest * ((widest + 1) // 2)
    tmp = {np.int16: np.zeros(elems, np.int16), np.uint8: np.zeros(elems, np.uint8)}
    return Pyramid(sides=sides, levels=levels, tmp=tmp)


def _reduce_2x2(op, src, side: int, dst, tmp) -> int:
    """One 4-ary reduction step, `side` x `side` -> ceil(side/2) squared.

    Two pairwise passes -- columns, then rows -- rather than the obvious
    `reshape(h, 2, h, 2).max(axis=(1, 3))`. The reshape spelling needs an even
    side, so it would need the array padded to even with each op's identity
    element, and `np.pad` allocates: one pad per field per level per ring,
    every frame, on the path whose whole contract is that it allocates
    nothing. Two passes handle an odd side by copying the leftover row or
    column through, which is the correct reduction over a block of one.

    Everything writes through an `out=`; `tmp` is the only intermediate and it
    is preallocated and shared.
    """
    h = (side + 1) // 2
    e = side // 2                      # whole 2-wide blocks; e == h when even
    s = src[:side * side].reshape(side, side)
    t = tmp[:side * h].reshape(side, h)

    op(s[:, 0:2 * e:2], s[:, 1:2 * e:2], out=t[:, :e])
    if side & 1:
        t[:, e] = s[:, side - 1]       # a block one column wide

    d = dst[:h * h].reshape(h, h)
    op(t[0:2 * e:2, :], t[1:2 * e:2, :], out=d[:e, :])
    if side & 1:
        d[e, :] = t[side - 1, :]       # a block one row tall
    return h


def build(pyr: Pyramid, soa: dict, rings) -> None:
    """Rebuild every level from the grid, in place. Allocates nothing.

    Level 1 reduces the grid itself; every level above reduces the one below,
    which is sound because max, min, AND and OR are all associative and
    idempotent over nesting. Call it after `traversability.update()` -- the
    AND and OR masks reduce the bitfield that pass writes, so a pyramid built
    before it describes the previous frame's traversability with this frame's
    heights, which is the kind of skew that produces a map that is wrong only
    while moving.
    """
    if len(rings) != len(pyr.levels):
        raise ValueError(
            f"pyramid was allocated over {len(pyr.levels)} rings, got {len(rings)}; "
            "it is sized by the schedule and cannot be reused across schedules")

    for r, layout in enumerate(rings):
        sides = pyr.sides[r]
        if sides[0] != layout.side:
            raise ValueError(
                f"ring {r} is {layout.side} cells across, pyramid was built for "
                f"{sides[0]}")
        span = slice(layout.offset, layout.offset + layout.slots)

        for name, source, op, dt in REDUCTIONS:
            # Level 0 is the grid, in the source field's own width; every level
            # above reads back the node field it just wrote.
            src = np.asarray(soa[source][span], dtype=dt)
            for level in range(1, len(sides)):
                dst = pyr.levels[r][level - 1][name]
                _reduce_2x2(op, src, sides[level - 1], dst, pyr.tmp[dt])
                src = dst


def level_arrays(pyr: Pyramid, soa: dict, rings, ring: int, level: int) -> dict:
    """The six node arrays at any level, level 0 included.

    Level 0 is served as views onto the grid -- `ground_height` answers to both
    `h_max` and `h_min` there, because a block of one cell is its own maximum
    and its own minimum. Callers therefore never branch on level, which is
    what stops a descent loop having a special last step.
    """
    if level == 0:
        layout = rings[ring]
        span = slice(layout.offset, layout.offset + layout.slots)
        return {name: soa[source][span] for name, source, _, _ in REDUCTIONS}
    return pyr.levels[ring][level - 1]


# --- §7.3, the theorem ------------------------------------------------------


def theorem3_safe(nodes: dict, thresholds=None) -> np.ndarray:
    """Math §7.3, exactly as stated. Bits 0, 2 and 5 only.

        SAFE(B) <=> H_max - H_min < s_max
                  & C_min - H_max > h_vehicle
                  & n_min >= n_min_threshold

    True means every cell in the block is traversable on clearance, step and
    confidence. It says NOTHING about slope, roughness or class -- see
    `all_clear()` for the predicate that does.

    Computed in int32. `C_min` is 32767 where nothing overhead has been seen,
    and `C_min - H_max` at int16 width would wrap that into a large negative
    clearance -- turning "the sky is clear" into "there is a ceiling at the
    ground", on exactly the cells with the least evidence.
    """
    th = thresholds if thresholds is not None else load_thresholds()
    t = th["traversability"]

    h_max = nodes["h_max"].astype(np.int32)
    h_min = nodes["h_min"].astype(np.int32)
    c_min = nodes["c_min"].astype(np.int32)
    n_min = nodes["n_min"].astype(np.int32)

    return ((h_max - h_min < t["s_max_m"] * 100.0)
            & (c_min - h_max > t["h_vehicle_m"] * 100.0)
            & (n_min >= t["n_min"]))


def all_clear(nodes: dict) -> np.ndarray:
    """Every cell in the block is traversable on all six conditions.

    The OR of the bitfields is zero exactly when no cell has any bit set. This
    is the predicate a planner may read as "drivable"; Theorem 3's is not.
    """
    return nodes["or_mask"] == 0


def certainly_blocked(nodes: dict) -> np.ndarray:
    """Every cell in the block fails the same condition, so there is no route
    through it at any resolution and a descent can stop here.

    Note the asymmetry with `all_clear`: a block where every cell is blocked
    for a DIFFERENT reason has AND_mask == 0 and is reported MIXED. That is
    conservative in the direction that costs time rather than safety.
    """
    return nodes["and_mask"] != 0


def classify(nodes: dict, thresholds=None) -> np.ndarray:
    """SAFE / BLOCKED / MIXED per node, as the frozen `api.QueryLOD`.

    SAFE is `all_clear`, not Theorem 3's predicate: `QueryLOD.SAFE` is
    documented as "every cell in the block is traversable", and that is the
    reading a caller will act on. The two cannot both fire -- OR_mask == 0
    implies AND_mask == 0.
    """
    del thresholds  # kept in the signature: descent policy may yet want it
    out = np.full(nodes["or_mask"].size, QueryLOD.MIXED, dtype=np.uint8)
    out[certainly_blocked(nodes)] = QueryLOD.BLOCKED
    out[all_clear(nodes)] = QueryLOD.SAFE
    return out


# --- geometry, for descent and for the tests --------------------------------


def block_extent(sides: list, level: int, row: int, col: int) -> tuple:
    """Level-0 rows and columns a node covers: (row0, row1, col0, col1).

    Clamped at each step, because ceiling-halved levels have edge nodes that
    cover one row or column rather than two. A version that just doubled would
    claim cells past the window edge and quietly index into the next ring.
    """
    r0, r1, c0, c1 = row, row + 1, col, col + 1
    for k in range(level, 0, -1):
        below = sides[k - 1]
        r0, r1 = 2 * r0, min(2 * r1, below)
        c0, c1 = 2 * c0, min(2 * c1, below)
    return r0, r1, c0, c1


def block_slots(sides: list, level: int, index: int) -> np.ndarray:
    """Flat level-0 slots, within the ring, that a node covers."""
    side = sides[level]
    r0, r1, c0, c1 = block_extent(sides, level, index // side, index % side)
    base = sides[0]
    rows = np.arange(r0, r1)[:, None]
    cols = np.arange(c0, c1)[None, :]
    return (rows * base + cols).ravel()
