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
    """
    cleared = np.unique(np.concatenate(
        [columns_to_clear(buf, dx), rows_to_clear(buf, dy)]))
    buf.x0 += dx
    buf.y0 += dy

    if soa is not None and cleared.size:
        flat = cleared + buf.offset
        for name, arr in soa.items():
            arr[flat] = (fill or {}).get(name, 0)
    return cleared + buf.offset


def cells_per_shift(buf: RingBuffer, dx: int, dy: int) -> int:
    """How many cells a shift touches. The O(perimeter) claim, as a number the
    dashboard and the report can both read."""
    W = buf.side
    if abs(dx) >= W or abs(dy) >= W:
        return buf.slots
    return abs(dx) * W + abs(dy) * W - abs(dx) * abs(dy)
