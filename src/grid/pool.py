"""Refinement pool — 512 blocks x 16 cells x 12 B = 98 KB, fixed. [Aakash]

Master v4 §3.4. Semantics can force local refinement below what range alone
would give, but only into this preallocated pool. When it is full, evict by
priority = closeness x dynamism x time-to-collision.

Nothing here allocates after startup. The compile-time memory bound is a
headline claim in the report and an allocation in the frame loop makes it
false. That rules out the obvious implementation: a dict from cell to block
grows, rehashes and allocates, so the owner table is a fixed pair of arrays
and lookup is a 512-wide vectorised compare. At 512 entries that is faster
than a hash anyway.

--- flaw E1, which is the whole reason this file is subtle ----------------

v2's pool and the ring schedule fought each other. Fix, from master v4 §3.4:
**refinement is "levels finer than the current ring", never an absolute cell
size**, and a block is released automatically when the schedule overtakes it.

Concretely: a block refining a ring-3 cell by one level is holding 20 cm
resolution. Drive toward it and the cell migrates to ring 2, which *is* 20 cm
— the schedule now provides for free exactly what the block is paying for.
Stored as an absolute size, the block looks like it is still doing its job, is
priority-protected because the cell is now close, and never leaves. The pool
fills with blocks that buy nothing, and it degrades to useless precisely as
you approach the things you cared about. `release_overtaken()` is the fix, and
it must be called every frame after ring migration.

--- ⚑ 16 cells per block does not hold one level of the ablation ---------

`cells_per_block: 16` holds a 4x4 subdivision: two levels at ratio 2, which is
every boundary of 5/10/20/40. The 5/10/50 ablation refines **5x** between
rings 1 and 2, so one level there is 25 children — larger than an entire
block. `levels_available()` returns 0 for that boundary and `acquire()`
refuses rather than silently truncating a 5x5 refinement into 16 cells, which
would drop 9 children and leave them reading as whatever the block held
before.

Options, all cheap, none mine to pick: 32 cells per block (196 KB, still
trivial), or refine the ablation's ring 2 in two steps through ring 1, or
state that semantic refinement is unavailable on the ablation schedule. It
only matters if a semantic gate fires on the ablation, which is a Day-3
question — but the number is wrong now and it will not announce itself.
"""

import numpy as np
from vrgrid.cell import CELL_FIELDS, alloc_soa

FREE = -1

# Priority reference scales. Not thresholds -- they set where each factor is
# half-weight, and the ranking is what matters, not the units.
CLOSENESS_REF_M = 10.0
TTC_REF_S = 3.0
DYNAMIC_GAIN = 3.0

# No known collision course is BASELINE priority, not zero priority. Without a
# floor the urgency factor goes to 0 as ttc -> inf and the product collapses,
# so the pool would refuse to refine anything not on a collision course --
# which is almost the whole map, almost all of the time.
URGENCY_FLOOR = 0.05


def priority(range_m, is_dynamic=False, ttc_s=np.inf) -> float:
    """closeness x dynamism x time-to-collision. Master v4 §3.4.

    ⚑ Each factor is INVERTED from the quantity it is named after, and the
      literal product is backwards. Range, "dynamism" read as a raw flag, and
      time-to-collision all get *larger* for things that matter *less*: a
      static kerb 90 m away with no collision in sight would score highest and
      evict the pedestrian stepping off the pavement in front of you. So:

        closeness = 1 / (1 + r/10 m)      near is urgent
        dynamism  = 1 + 3 if moving       moving is urgent
        urgency   = 1 / (1 + ttc/3 s)     imminent is urgent

    Higher is kept. `ttc = inf` -- not on a collision course, which is most of
    the map -- floors the urgency factor rather than zeroing it, so such a
    cell still ranks by closeness and motion. Zeroing it makes the whole
    product zero and the pool stops refining anything that is not about to be
    hit, which is not what "priority" was supposed to mean.
    """
    closeness = 1.0 / (1.0 + max(float(range_m), 0.0) / CLOSENESS_REF_M)
    dynamism = 1.0 + (DYNAMIC_GAIN if is_dynamic else 0.0)
    ttc = float(ttc_s)
    urgency = max(1.0 / (1.0 + ttc / TTC_REF_S), URGENCY_FLOOR) if ttc >= 0 else 1.0
    return closeness * dynamism * urgency


class RefinementPool:
    """A fixed set of blocks, each holding one cell's children.

    A block records the ring it was taken against and how many levels FINER it
    refines — never a cell size. See E1 above.
    """

    def __init__(self, blocks: int = 512, cells_per_block: int = 16, arrays=None):
        if cells_per_block < 4:
            raise ValueError("a block must hold at least one 2x2 refinement")
        self.blocks = int(blocks)
        self.cells_per_block = int(cells_per_block)

        # The cells themselves. Same 12-byte struct as the grid, so a refined
        # cell is a cell and query() does not need a second code path.
        self.cells = arrays if arrays is not None else alloc_soa(
            self.blocks * self.cells_per_block)

        # Owner table. Fixed size, allocated here, never resized.
        self.owner_ring = np.full(self.blocks, FREE, dtype=np.int16)
        self.owner_slot = np.full(self.blocks, FREE, dtype=np.int64)
        self.levels = np.zeros(self.blocks, dtype=np.uint8)
        self.score = np.zeros(self.blocks, dtype=np.float64)

    # --- capacity, per schedule boundary ------------------------------------

    def children_per_level(self, schedule, ring: int) -> int:
        """m^2 children when a ring-`ring` cell is refined one level."""
        if ring < 1:
            raise ValueError(f"ring {ring} is the base lattice; nothing is finer")
        m = schedule.k(ring) // schedule.k(ring - 1)
        return m * m

    def levels_available(self, schedule, ring: int) -> int:
        """How many levels of refinement a single block can actually hold.

        Returns 0 when even one level does not fit -- the ablation's 5x
        boundary. Callers must check; `acquire()` does.
        """
        levels, cells = 0, 1
        r = ring
        while r >= 1:
            cells *= self.children_per_level(schedule, r)
            if cells > self.cells_per_block:
                break
            levels += 1
            r -= 1
        return levels

    # --- acquire / release ---------------------------------------------------

    def find(self, ring: int, slot: int) -> int:
        """Block refining this cell, or FREE. Vectorised compare over the fixed
        owner table -- no dict, so no allocation in the frame loop."""
        hit = np.flatnonzero((self.owner_ring == ring) & (self.owner_slot == slot))
        return int(hit[0]) if hit.size else FREE

    @property
    def free_blocks(self) -> int:
        return int(np.count_nonzero(self.owner_ring == FREE))

    def acquire(self, schedule, ring: int, slot: int, levels: int, score: float) -> int:
        """Take a block for cell `slot` of `ring`, refining `levels` finer.

        Returns the block index, or FREE if the request was refused. Refused
        for two reasons, and they are different failures:

        * the refinement does not FIT in a block (the ablation's 5x boundary),
          which is a configuration error and raises;
        * the pool is full of more important blocks, which is the designed
          degradation -- "bounded, degrading gracefully by dropping the least
          relevant" -- and returns FREE.

        Eviction takes the lowest-scoring block, and only if the incoming
        request outscores it. Evicting something more urgent to make room for
        something less urgent is worse than refusing.
        """
        if levels < 1:
            raise ValueError("acquire() with levels < 1 refines nothing")
        available = self.levels_available(schedule, ring)
        if levels > available:
            need = 1
            r = ring
            for _ in range(levels):
                need *= self.children_per_level(schedule, r)
                r -= 1
            raise ValueError(
                f"ring {ring} refined {levels} level(s) needs {need} cells; a block "
                f"holds {self.cells_per_block}. On {schedule.name} the ratio at this "
                f"boundary is {schedule.k(ring) // schedule.k(ring - 1)}x. See the "
                "note at the top of pool.py -- this is a config decision, not a bug "
                "to work around here"
            )

        existing = self.find(ring, slot)
        if existing != FREE:
            self.score[existing] = score
            return existing

        free = np.flatnonzero(self.owner_ring == FREE)
        block = int(free[0]) if free.size else self._evict_for(score)
        if block == FREE:
            return FREE

        self.owner_ring[block] = ring
        self.owner_slot[block] = slot
        self.levels[block] = levels
        self.score[block] = score
        self._clear(block)
        return block

    def _evict_for(self, score: float) -> int:
        victim = int(np.argmin(self.score))
        if self.score[victim] >= score:
            return FREE
        self.release(victim)
        return victim

    def release(self, block: int) -> None:
        """Give a block back. The cells are cleared on the next acquire, not
        here -- releasing is on the migration path and clearing 16 cells x 10
        fields for a block nobody has asked for yet is work done early."""
        self.owner_ring[block] = FREE
        self.owner_slot[block] = FREE
        self.levels[block] = 0
        self.score[block] = 0.0

    def _clear(self, block: int) -> None:
        lo = block * self.cells_per_block
        sl = slice(lo, lo + self.cells_per_block)
        for name, _ in CELL_FIELDS:
            self.cells[name][sl] = 0

    def block_cells(self, block: int) -> slice:
        """Where block `block`'s cells live in the pool arrays."""
        lo = block * self.cells_per_block
        return slice(lo, lo + self.cells_per_block)

    # --- E1: release what the schedule now provides free ---------------------

    def release_overtaken(self, current_ring) -> int:
        """Release every block whose refinement the schedule has overtaken.

        `current_ring(ring, slot) -> int` answers where that cell lives NOW,
        after this frame's migration. A block taken against ring L to give
        `levels` finer is buying ring L - levels; if the cell has since
        migrated to ring L' <= L - levels, the schedule provides that or
        better and the block is dead weight. Returns the number released.

        Call this every frame, after migration and before any acquire. This is
        the whole of the E1 fix -- without it the pool degrades to useless
        exactly as you approach things, which is the opposite of what a
        refinement pool is for.
        """
        released = 0
        for block in np.flatnonzero(self.owner_ring != FREE):
            ring = int(self.owner_ring[block])
            slot = int(self.owner_slot[block])
            now = current_ring(ring, slot)
            if now is None:
                continue
            if int(now) <= ring - int(self.levels[block]):
                self.release(int(block))
                released += 1
        return released

    def bytes_used(self) -> int:
        """Fixed, and the number the memory table quotes: 512 x 16 x 12 B."""
        return sum(a.nbytes for a in self.cells.values())
