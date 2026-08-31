"""The resolution-agnostic query API. Master v4 §3.7, frozen day one. [Aakash]

    query(x, y) -> CellQuery      the caller never learns which ring served it
    is_traversable(x, y) -> bool

That promise is the point. A planner asks about a place, not about a ring, and
gets one answer whatever resolution happens to be there — which is also one of
the three answers to the hardest question this project will be asked ("standard
planners want uniform grids, so you give the savings back in resampling").

--- the union rule, defined once, here -----------------------------------

Master v4 §3.7 is explicit that the transient layer shares the grid geometry
and that `query()` returns the union:

    occupancy = OCCUPIED if persistent OR transient
    dynamic   = True when the transient layer supplied it

Defined in one place on purpose. Every consumer that has to merge two layers
itself inherits the merge problem, and they will not all solve it the same way.

--- the frozen signature takes no map, and that is deliberate -------------

`include/vrgrid/api.py` freezes `query(x, y)`. A map handle has to come from
somewhere, so the working functions here take a `GridMap` and the frozen
module-level pair binds one via `bind()`. Keeping the state explicit here
means the tests never depend on import order, and the singleton is one line
of wiring in the frame loop rather than a hidden global everything reads.
"""

from dataclasses import dataclass, field

import numpy as np
from vrgrid.api import CellQuery
from vrgrid.cell import (
    FLAG_DYNAMIC,
    OCC_FREE,
    OCC_OCCUPIED,
    OCC_UNKNOWN,
    TRAV_CONFIDENCE,
)
from vrgrid.grid.fusion import occupancy_state, unpack_class
from vrgrid.grid.lattice import OUTSIDE, i_ring, ring_of
from vrgrid.grid.schedule import load_thresholds

# What a query outside the map returns. Not ring 0, not zeros: an out-of-map
# point is unknown, and unknown is not traversable (fail safe, §7.1 bit 5).
OUT_OF_MAP = CellQuery(
    ground_height=float("nan"),
    ceiling_height=float("nan"),
    semantic_class=0,
    traversability=TRAV_CONFIDENCE,
    confidence=0,
    occupancy=OCC_UNKNOWN,
    dynamic=False,
)


@dataclass
class GridMap:
    """Everything a query needs: the cells, the geometry, and the windows.

    `buffers` is one `gpu.shift.RingBuffer` per ring — the toroidal window
    state, which is the only thing that says where a ring's absolute lattice
    coordinates currently live in memory.
    """

    soa: dict
    schedule: object
    buffers: list
    thresholds: dict = field(default_factory=load_thresholds)
    transient: dict | None = None
    pool: object | None = None
    speed_ms: float = 0.0
    scatter_mode: str = "sorted"

    # Where the vehicle is in the WORLD, in metres. Queries arrive in vehicle
    # frame (the frozen API says so) but cells are world-anchored, so this is
    # what converts between them. `harness.recenter()` maintains it; it stays
    # (0, 0) for a stationary map, which is what the unit tests use.
    vehicle_xy_m: tuple = (0.0, 0.0)

    # Working set for `lattice.bin_points`, built on first use and then reused
    # for the life of the map. Deliberately NOT part of `allocate()`: a GridMap
    # is built by hundreds of unit tests that never bin a point, and 7.5 MB
    # each would be paid by all of them. The frame loop pays it once, on its
    # first sweep; `run/engine.py` allocates its own up front instead.
    _bin: object = None

    def bin_scratch(self, n: int):
        """(scratch, out) sized for at least `n` points.

        Growing means rebuilding, which allocates -- so it happens on frame 0
        and then never again, which is the shape "no allocation in the frame
        loop" actually asks for.
        """
        from vrgrid.grid.lattice import new_bin_scratch

        if self._bin is None or self._bin[0]["max_points"] < n:
            self._bin = (new_bin_scratch(n, self.schedule),
                         np.zeros(n, dtype=np.int64))
        return self._bin


def slot_of(gm: GridMap, x_m: float, y_m: float):
    """(ring, flat slot) for a point in vehicle frame, or (OUTSIDE, -1).

    Ring from `ring_of` (§6), cell index by integer division from the ONE base
    lattice (§2), slot from the ring's toroidal window. Three separate
    concerns and each one owned by the file that proved it — this function is
    only allowed to compose them, never to recompute a lattice index itself.

    ⚑ The same two-frame split as `fusion.scatter()`, and it has to be: the
      RING is decided in the vehicle frame, because foveation follows the
      vehicle, and the CELL is decided in the world frame, because cell
      identity is world-anchored and the toroidal window is addressed in world
      lattice coordinates.

      Index the lattice in the vehicle frame instead and nothing raises: the
      window has moved with the vehicle, so the computed slot is simply some
      other place's cell, or falls outside the window and reads as out-of-map.
      A map holding 143,000 observed cells answers "never seen" five metres
      ahead, and every metric built on `query()` quietly measures an empty
      map. Found by the plan-regret harness reporting a path 100% unknown.
    """
    ring = ring_of(x_m, y_m, gm.schedule, gm.speed_ms)
    if ring == OUTSIDE:
        return OUTSIDE, -1

    c0 = gm.schedule.base_cell_m
    k = gm.schedule.k(ring)
    ix = i_ring(x_m + gm.vehicle_xy_m[0], c0, k)
    iy = i_ring(y_m + gm.vehicle_xy_m[1], c0, k)

    slot = int(gm.buffers[ring].flat_slot(ix, iy))
    return (ring, slot) if slot >= 0 else (OUTSIDE, -1)


def window_cells(buf):
    """Absolute lattice (ix, iy) for every slot of a ring window, in slot order.

    The inverse of `RingBuffer.slot()`, which is `(iy mod W)*W + (ix mod W)`.
    A slot does not carry its own coordinates -- that is the whole point of
    toroidal addressing, and it is why a shift is O(perimeter) -- so the
    window's origin is what turns one back into a place. Needed by anything
    that has to walk the map rather than ask about a point: the metrics, the
    reference-map comparison, the dashboard.
    """
    W = buf.side
    sx, sy = np.meshgrid(np.arange(W), np.arange(W), indexing="xy")
    ix = buf.x0 + np.mod(sx - buf.x0, W)
    iy = buf.y0 + np.mod(sy - buf.y0, W)
    return ix.reshape(-1), iy.reshape(-1)


def query(gm: GridMap, x_m: float, y_m: float) -> CellQuery:
    """Point query in vehicle frame (x forward, y left, z up). Master v4 §3.7.

    Heights come back in METRES. They are stored as int16 centimetres and the
    conversion happens exactly here, at the boundary, so that no consumer ever
    has to know the storage unit — CLAUDE.md's suffix rule applied to an
    interface rather than a variable.
    """
    ring, slot = slot_of(gm, x_m, y_m)
    if ring == OUTSIDE:
        return OUT_OF_MAP

    soa, slot = _refined(gm, ring, slot, x_m, y_m)

    ground = float(soa["ground_height"][slot]) / 100.0
    ceiling = float(soa["ceiling_height"][slot]) / 100.0
    occ = int(occupancy_state(soa, gm.thresholds, [slot])[0])
    trav = int(soa["traversability"][slot])
    cls = int(unpack_class(soa["semantic_class"][slot])[0])   # §10.2 candidate
    n = int(soa["obs_count"][slot])

    dynamic = False
    if gm.transient is not None:
        t_occ, t_ground = _transient(gm, ring, slot)
        if t_occ == OCC_OCCUPIED:
            # The union rule. OCCUPIED wins, and the height it wins with is the
            # transient one -- a pedestrian's height is not the road's.
            occ = OCC_OCCUPIED
            dynamic = True
            ground = t_ground

    return CellQuery(
        ground_height=ground,
        ceiling_height=ceiling,
        semantic_class=cls,
        traversability=trav,
        confidence=n,
        occupancy=occ,
        dynamic=dynamic,
    )


def _refined(gm: GridMap, ring: int, slot: int, x_m: float, y_m: float):
    """Serve from the refinement pool when a block covers this cell.

    This is what "resolution-agnostic" costs: one lookup. The caller asked
    about a place and gets the finest answer the map holds for it, without
    being told that a semantic gate refined it three frames ago.
    """
    if gm.pool is None:
        return gm.soa, slot
    block = gm.pool.find(ring, slot)
    if block < 0:
        return gm.soa, slot

    levels = int(gm.pool.levels[block])
    child_ring = ring - levels
    c0 = gm.schedule.base_cell_m
    k_child = gm.schedule.k(child_ring)
    k_parent = gm.schedule.k(ring)
    m = k_parent // k_child                       # children per side

    # Where inside the parent the point falls, in child cells. World frame,
    # for the same reason slot_of() uses it.
    wx = x_m + gm.vehicle_xy_m[0]
    wy = y_m + gm.vehicle_xy_m[1]
    ox = i_ring(wx, c0, k_child) - i_ring(wx, c0, k_parent) * m
    oy = i_ring(wy, c0, k_child) - i_ring(wy, c0, k_parent) * m
    inner = int(oy) * m + int(ox)
    return gm.pool.cells, gm.pool.block_cells(block).start + inner


def _transient(gm: GridMap, ring: int, slot: int):
    """(occupancy, ground in metres) from the transient layer for one slot.

    The transient layer is 4 bytes per cell — height, log-odds, flags — and
    shares the grid's geometry, so the slot index is the same one. It is
    frame-fresh: whatever is in it was put there this frame, which is why
    occupancy here is a flag test and not a log-odds history.
    """
    if slot >= gm.transient["log_odds"].size:
        return OCC_UNKNOWN, 0.0
    flags = int(gm.transient["flags"][slot])
    if not flags & FLAG_DYNAMIC:
        return OCC_UNKNOWN, 0.0
    return OCC_OCCUPIED, float(gm.transient["ground_height"][slot]) / 100.0


def is_traversable(gm: GridMap, x_m: float, y_m: float) -> bool:
    """True only if every one of §7.1's six conditions passes.

    Fail safe in both directions that matter: out of map is not traversable,
    and neither is a cell the transient layer says something is standing in.
    A dynamic obstacle is not a property of the ground, so it cannot be a bit
    in the ground's bitfield -- it is folded in here, at the predicate.
    """
    q = query(gm, x_m, y_m)
    if q.occupancy == OCC_UNKNOWN or q.dynamic:
        return False
    return q.traversability == 0


def free_space(gm: GridMap, x_m: float, y_m: float) -> bool:
    """Explicitly FREE, as opposed to "not occupied". Unknown is not free --
    the distinction §10.1 exists to preserve, offered as its own function so
    that no caller has to spell `!= OCC_OCCUPIED` and get it wrong."""
    return query(gm, x_m, y_m).occupancy == OCC_FREE


# --- binding for the frozen module-level signatures ---------------------------

_BOUND: GridMap | None = None


def bind(gm: GridMap) -> None:
    """Make `gm` the map that `api.query(x, y)` answers from."""
    global _BOUND
    _BOUND = gm


def bound() -> GridMap:
    if _BOUND is None:
        raise RuntimeError("no GridMap bound -- call grid.query.bind(gm) at startup")
    return _BOUND


def query_bound(x: float, y: float) -> CellQuery:
    return query(bound(), x, y)


def is_traversable_bound(x: float, y: float) -> bool:
    return is_traversable(bound(), x, y)


def query_region(gm: GridMap, xs, ys):
    """Many points, one call. The scalar path above is the definition; this
    exists because the dashboard and the metrics both want a whole raster and
    a Python loop over 10^5 points at 10 Hz is not a thing you can do.

    Deliberately a loop over `query()` rather than a second implementation:
    two query paths that can disagree is precisely the bug this project keeps
    designing out (see lattice.i_fine on the scalar/vector split).
    """
    xs = np.atleast_1d(np.asarray(xs, dtype=np.float64))
    ys = np.atleast_1d(np.asarray(ys, dtype=np.float64))
    return [query(gm, float(x), float(y)) for x, y in zip(xs, ys)]
