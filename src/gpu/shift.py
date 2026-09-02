"""Toroidal ego-motion shift. [Shrestha]

The map is ego-centric: as the vehicle drives, the window of the world each
ring covers slides with it. Doing that by moving data costs O(area) every
frame. Doing it by moving the *origin* costs O(perimeter) -- Ring 3 clears
about 1,000 cells per shift rather than 250,000.

Addressing. Each ring is a square buffer of side W, and absolute lattice cell
(ix, iy) lives permanently at slot (iy mod W, ix mod W). Nothing is ever
copied. A cell scrolling out of view aliases to the slot a newly-visible cell
needs, so the only work per shift is clearing the columns and rows that just
came into view.

Why squares and not annuli. Storing each ring as an annulus -- the square
minus the hole the finer ring covers -- is 745,000 cells against 910,000, a
real 1.98 MB saving. But the hole is centred on the vehicle, so under a shift
cells migrate between the annulus bands and the shift becomes a gather over
the whole ring. Measured on this machine: 15.2 ms p50 for the annulus gather
against 0.04 ms for the toroidal clear, on a 100 ms frame budget. The padding
buys back 15% of the frame, every frame, and it does not grow with the map.

The 745,000 figure is unaffected as a *logical* cell count, which is what
every ratio in the report is computed from. Allocation is 910,000 slots.
"""

from dataclasses import dataclass

import numpy as np
from vrgrid.gpu.allocators import EMPTY_CELL
from vrgrid.gpu.kernels import CEILING_NONE, Z_MAX_CM, Z_MIN_CM


@dataclass
class RingBuffer:
    """One ring's toroidal window. `x0`, `y0` are the absolute lattice
    coordinates of the window's low corner; they are the only state a shift
    changes."""

    side: int
    offset: int      # start of this ring in the flat SoA arrays
    x0: int = 0
    y0: int = 0

    @property
    def slots(self) -> int:
        return self.side * self.side

    def in_view(self, ix, iy):
        ix, iy = np.asarray(ix), np.asarray(iy)
        return ((ix >= self.x0) & (ix < self.x0 + self.side)
                & (iy >= self.y0) & (iy < self.y0 + self.side))

    def slot(self, ix, iy):
        """Flat slot within the ring for an absolute lattice cell.

        Returns -1 outside the current window. A cell that is out of view has
        no slot of its own -- its slot belongs to whatever is in view there
        now, and writing to it would corrupt a live cell.
        """
        ix, iy = np.asarray(ix), np.asarray(iy)
        s = (np.mod(iy, self.side) * self.side + np.mod(ix, self.side)).astype(np.int64)
        return np.where(self.in_view(ix, iy), s, -1)

    def flat_slot(self, ix, iy):
        """Slot in the whole-grid flat arrays. -1 stays -1."""
        s = self.slot(ix, iy)
        return np.where(s < 0, -1, s + self.offset)


def columns_to_clear(buf: RingBuffer, dx: int) -> np.ndarray:
    """Slots of the columns that come into view when the window moves by dx.

    Their slots are exactly those of the columns leaving on the other side, so
    the stale data is overwritten in place. |dx| >= side means the whole ring
    is new, which is the one case that degenerates to O(area).
    """
    if dx == 0:
        return np.zeros(0, dtype=np.int64)
    W = buf.side
    if abs(dx) >= W:
        return np.arange(buf.slots, dtype=np.int64)
    cols = (np.arange(buf.x0, buf.x0 + dx) if dx > 0
            else np.arange(buf.x0 + W + dx, buf.x0 + W))
    rows = np.arange(W, dtype=np.int64)
    return (rows[:, None] * W + np.mod(cols, W)[None, :]).ravel()


def rows_to_clear(buf: RingBuffer, dy: int) -> np.ndarray:
    if dy == 0:
        return np.zeros(0, dtype=np.int64)
    W = buf.side
    if abs(dy) >= W:
        return np.arange(buf.slots, dtype=np.int64)
    rws = (np.arange(buf.y0, buf.y0 + dy) if dy > 0
           else np.arange(buf.y0 + W + dy, buf.y0 + W))
    cols = np.arange(W, dtype=np.int64)
    return (np.mod(rws, W)[:, None] * W + cols[None, :]).ravel()


def shift(buf: RingBuffer, dx: int, dy: int, soa: dict | None = None,
          fill: dict | None = None) -> np.ndarray:
    """Move the window by (dx, dy) whole cells and clear what just came into
    view. Returns the cleared slots, in flat-array coordinates.

    Nothing is copied. The cost is |dx|*W + |dy|*W slot writes -- O(perimeter)
    for the small per-frame shifts ego-motion actually produces.

    A newly visible cell is a never-observed cell, so by default the strip is
    written with `allocators.EMPTY_CELL` -- the same state `allocate()` starts
    the map in -- and not with raw zeros. The difference is `ceiling_height`:
    zeroing it says "solid ground at the datum" and drives TRAV_CLEARANCE
    across the whole strip, so a map that boots correct would degrade as soon
    as the vehicle moved. Pass `fill={}` for a literal zero clear.
    """
    cleared = np.unique(np.concatenate(
        [columns_to_clear(buf, dx), rows_to_clear(buf, dy)]))
    buf.x0 += dx
    buf.y0 += dy

    fill = EMPTY_CELL if fill is None else fill

    if soa is not None and cleared.size:
        flat = cleared + buf.offset
        for name, arr in soa.items():
            arr[flat] = fill.get(name, 0)
    return cleared + buf.offset


def cells_per_shift(buf: RingBuffer, dx: int, dy: int) -> int:
    """How many cells a shift touches. The O(perimeter) claim, as a number the
    dashboard and the report can both read."""
    W = buf.side
    if abs(dx) >= W or abs(dy) >= W:
        return buf.slots
    return abs(dx) * W + abs(dy) * W - abs(dx) * abs(dy)


def new_slot_scratch(max_points: int) -> dict:
    """Working set for `flat_slot_into`, sized at startup like every other
    frame-path buffer. 25 B per point: 8.4 MB at the 150,000-point cap in
    `configs/thresholds.yaml: scatter.max_points_per_frame`."""
    return {"col": np.zeros(max_points, np.int64),
            "row": np.zeros(max_points, np.int64),
            "live": np.zeros(max_points, np.bool_),
            "tmp": np.zeros(max_points, np.bool_)}


def flat_slot_into(buf: RingBuffer, ix, iy, out, scratch: dict):
    """`RingBuffer.flat_slot` with no allocation and no integer division.

    Bit-identical to `flat_slot` -- `test_flat_slot_into_matches_flat_slot`
    pins that across ring sides, offsets and shifted windows -- and about 28%
    faster on the frame path, because it drops both costs the method version
    pays:

    **No `np.mod`.** Integer division is the expensive operation in the
    method, and it is avoidable. In view means `x0 <= ix < x0 + W`, so
    `c = ix - x0` is already in `[0, W)`, and

        (x0 + c) mod W  ==  (x0 mod W) + c,  minus W once if that overflowed

    since both terms are in `[0, W)` and their sum is therefore in `[0, 2W)`.
    One subtract under a mask replaces a division per point per axis.

    **No temporaries.** The method builds seven intermediate arrays per call
    -- two `asarray`, four comparisons inside `in_view`, and the `np.where`
    that selects against them -- and at 120,000 points on four rings that is
    the kind of per-frame cost that does not show up until someone profiles
    it. Everything here writes through `out=` into `scratch`.

    Out-of-window points are -1, exactly as `flat_slot` returns them, so
    scatter drops them. Their `col`/`row` intermediates are meaningless rather
    than merely out of range -- they are never used as an index, only
    overwritten.
    """
    n, W = len(ix), buf.side
    col, row = scratch["col"][:n], scratch["row"][:n]
    live, tmp = scratch["live"][:n], scratch["tmp"][:n]

    np.subtract(ix, buf.x0, out=col)
    np.subtract(iy, buf.y0, out=row)

    # live = (0 <= col < W) & (0 <= row < W). Every comparison lands in `tmp`
    # and is folded straight into `live`, so the four of them share one
    # temporary instead of building four arrays -- which is the whole point of
    # this function and the reason the pairs used to sit on single lines.
    np.greater_equal(col, 0, out=live)
    np.less(col, W, out=tmp)
    np.logical_and(live, tmp, out=live)
    np.greater_equal(row, 0, out=tmp)
    np.logical_and(live, tmp, out=live)
    np.less(row, W, out=tmp)
    np.logical_and(live, tmp, out=live)

    # (x0 + c) mod W, by the identity in the docstring: add `x0 mod W`, then
    # take W back off only the ones that overflowed past it.
    np.add(col, buf.x0 % W, out=col)
    np.greater_equal(col, W, out=tmp)
    np.subtract(col, W, out=col, where=tmp)
    np.add(row, buf.y0 % W, out=row)
    np.greater_equal(row, W, out=tmp)
    np.subtract(row, W, out=row, where=tmp)

    np.multiply(row, W, out=row)
    np.add(row, col, out=out)
    np.add(out, buf.offset, out=out)
    np.copyto(out, -1, where=np.logical_not(live, out=tmp))
    return out


# The vertical band slides in whole steps of this, the way `_track_vehicle`
# slides the horizontal window in whole cells. Continuous tracking would
# re-base on every frame; whole steps make it rare.
Z_DATUM_STEP_M = 1.0


def track_datum(grid, datum_m, ego_z_m: float) -> float:
    """Slide the 8 m vertical band to keep the vehicle inside it. Returns the
    new datum, and RE-BASES every height already in `grid` to match.

    The vertical counterpart of `_track_vehicle`, and the reason it exists:
    `kernels.quantise_height` clamps to an 8 m band, and clamped world-absolute
    that band is a slab from -2 to +6 m above sea level rather than 8 m around
    the vehicle. On seq 08's 45.7 m climb every nearby cell saturated at the
    ceiling, `visibility_cleanup` computed viewing angles past the sensor's FOV
    limit, and ghost removal went inert for 57% of frames. Widening the clamp
    would also have worked and would have cost the report its memory claim --
    the dense-3D baseline counts voxels over exactly this extent, so a taller
    band inflates our own headline ratio. Moving an 8 m band leaves that
    arithmetic untouched.

    ⚑ Heights are stored RELATIVE to the datum, so moving it means re-basing
      every height already in the map. That is what makes a moving datum safe
      for the feature detectors: after re-basing, every cell is relative to the
      SAME current datum, so differences -- slope, step, curb height, pothole
      depth -- are unaffected. A datum that moved without re-basing would put a
      spurious step between any two cells last seen at different elevations.

    ⚑ This allocates (the `seen` mask, 0.91 MB against a 10.92 MB grid) and is
      the one path in the frame loop that does. It is freed immediately and it
      is rare: seq 08 crosses a 1 m step 46 times in 4,071 frames. Measured at
      4.57 ms on a frame that re-bases and 0.23 us on one that does not.

    `datum_m` of None means "first frame": the map is empty, so there is
    nothing to re-base and the band starts where the vehicle is. That also
    keeps the first step from being a jump of the sequence's whole elevation.
    """
    want = float(np.floor(ego_z_m / Z_DATUM_STEP_M) * Z_DATUM_STEP_M)
    if datum_m is None:
        return want
    if want == datum_m:
        return datum_m

    delta_cm = round((want - datum_m) * 100.0)
    ground = grid["ground_height"]
    ceiling = grid["ceiling_height"]
    seen = ceiling != CEILING_NONE
    if abs(delta_cm) >= Z_MAX_CM - Z_MIN_CM:
        # The band moved further than it is wide, so nothing already in the map
        # is inside it any more. Saturating in one fill is what the clamp would
        # do cell by cell, and it avoids widening every height to int32 to hold
        # an intermediate that is about to be clipped anyway.
        end = Z_MIN_CM if delta_cm > 0 else Z_MAX_CM
        ground[:] = end
        ceiling[seen] = end
        return want

    # In range, so the subtraction cannot leave int16 on the way: stored heights
    # are already inside the band and |delta| is below its span. CEILING_NONE is
    # a sentinel, not a height -- re-basing it would turn "nothing overhead was
    # ever seen" into a ceiling at the band's edge, and §7.1 would read that as
    # a clearance failure over the whole map.
    np.subtract(ground, delta_cm, out=ground)
    np.clip(ground, Z_MIN_CM, Z_MAX_CM, out=ground)
    np.subtract(ceiling, delta_cm, out=ceiling, where=seen)
    np.clip(ceiling, Z_MIN_CM, Z_MAX_CM, out=ceiling, where=seen)
    return want
