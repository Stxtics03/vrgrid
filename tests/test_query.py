"""The resolution-agnostic query API. Master v4 §3.7. [Aakash]"""

import numpy as np
import pytest
from vrgrid.cell import (
    FLAG_DYNAMIC,
    OCC_FREE,
    OCC_OCCUPIED,
    OCC_UNKNOWN,
    TRAV_CONFIDENCE,
)
from vrgrid.eval.harness import build_gridmap
from vrgrid.grid.lattice import OUTSIDE
from vrgrid.grid.pool import priority
from vrgrid.grid.query import (
    _refined,
    bind,
    bound,
    free_space,
    is_traversable,
    is_traversable_bound,
    query,
    query_bound,
    slot_of,
    window_cells,
)
from vrgrid.grid.schedule import load


@pytest.fixture
def gm():
    return build_gridmap(load("5/10/20/40"))


def _write(gm, x, y, ground_cm=0, ceiling_cm=32767, n=9, trav=0, cls=9):
    ring, slot = slot_of(gm, x, y)
    assert ring != OUTSIDE
    gm.soa["ground_height"][slot] = ground_cm
    gm.soa["ceiling_height"][slot] = ceiling_cm
    gm.soa["obs_count"][slot] = n
    gm.soa["traversability"][slot] = trav
    gm.soa["semantic_class"][slot] = (cls << 4) | 5
    gm.soa["log_odds"][slot] = 20
    return ring, slot


def test_window_cells_inverts_the_toroidal_slot(gm):
    """Every slot's absolute lattice coordinates must map back to that slot.
    Everything that walks the map rather than asking about a point -- the
    metrics, the dashboard -- depends on this being an exact inverse."""
    for ring, buf in enumerate(gm.buffers):
        if buf.side > 200:
            continue                       # the big rings are covered by ring 0
        ix, iy = window_cells(buf)
        assert np.array_equal(buf.slot(ix, iy), np.arange(buf.slots))


def test_the_caller_never_learns_which_ring_served_it(gm):
    """The §3.7 promise, as a property: `CellQuery` carries no ring, and a
    point either side of a ring boundary answers in the same units and the
    same shape. A planner asks about a place, not about a resolution."""
    inner = _write(gm, 5.0, 0.0, ground_cm=-7)
    outer = _write(gm, 30.0, 0.0, ground_cm=-7)
    assert inner[0] != outer[0], "test points are not on opposite sides of a boundary"

    a, b = query(gm, 5.0, 0.0), query(gm, 30.0, 0.0)
    assert a.ground_height == b.ground_height == pytest.approx(-0.07)
    assert not hasattr(a, "ring")
    assert set(vars(a)) == set(vars(b))


def test_heights_come_back_in_metres(gm):
    """Stored as int16 centimetres, returned as metres, converted exactly at
    the boundary so no consumer has to know the storage unit."""
    _write(gm, 3.0, 1.0, ground_cm=-12, ceiling_cm=350)
    q = query(gm, 3.0, 1.0)
    assert q.ground_height == pytest.approx(-0.12)
    assert q.ceiling_height == pytest.approx(3.50)


def test_out_of_map_is_unknown_and_not_traversable(gm):
    """An out-of-map point is not ring 0 and not zeros. Silently clamping it
    to the origin is how a 150 m return ends up in a 5 cm cell at the
    vehicle."""
    far = 500.0
    assert slot_of(gm, far, far)[0] == OUTSIDE

    q = query(gm, far, far)
    assert q.occupancy == OCC_UNKNOWN
    assert q.traversability & TRAV_CONFIDENCE
    assert not is_traversable(gm, far, far)
    assert not free_space(gm, far, far)


def test_unknown_is_not_free_and_not_traversable(gm):
    """Three states, and the two that are not OCCUPIED are not the same. A
    never-observed cell must fail both predicates -- "I couldn't see" is not
    "I looked and it's empty"."""
    q = query(gm, 2.0, 2.0)                    # untouched cell
    assert q.occupancy == OCC_UNKNOWN
    assert not free_space(gm, 2.0, 2.0)
    assert not is_traversable(gm, 2.0, 2.0)


def test_is_traversable_requires_every_bit_clear(gm):
    _write(gm, 4.0, 0.0, trav=0)
    assert is_traversable(gm, 4.0, 0.0)

    for bit in (1, 2, 4, 8, 16, 32):
        _write(gm, 4.0, 0.0, trav=bit)
        assert not is_traversable(gm, 4.0, 0.0), f"bit {bit} did not block"


# --- the union rule, master v4 §3.7 ------------------------------------------


def test_transient_layer_wins_occupancy_and_marks_it_dynamic(gm):
    """occupancy = OCCUPIED if persistent OR transient, dynamic = True when
    the transient layer supplied it. Defined once, here, so no consumer has to
    merge two layers itself and get it slightly differently."""
    _ring, slot = _write(gm, 6.0, 0.0, ground_cm=-5)
    assert query(gm, 6.0, 0.0).dynamic is False

    gm.transient["flags"][slot] = FLAG_DYNAMIC
    gm.transient["ground_height"][slot] = 170          # a person, not the road

    q = query(gm, 6.0, 0.0)
    assert q.occupancy == OCC_OCCUPIED
    assert q.dynamic is True
    assert q.ground_height == pytest.approx(1.70), "returned the road under the person"


def test_a_dynamic_cell_is_not_traversable_even_with_a_clear_bitfield(gm):
    """A moving obstacle is not a property of the ground, so it cannot be a
    bit in the ground's bitfield. It is folded in at the predicate instead --
    which means a cell whose ground is perfectly drivable still refuses while
    something is standing in it."""
    _ring, slot = _write(gm, 6.0, 0.0, trav=0)
    assert is_traversable(gm, 6.0, 0.0)

    gm.transient["flags"][slot] = FLAG_DYNAMIC
    assert query(gm, 6.0, 0.0).traversability == 0
    assert not is_traversable(gm, 6.0, 0.0)


# --- the refinement pool, served transparently -------------------------------


def test_a_refined_cell_is_served_from_the_pool(gm):
    """What "resolution-agnostic" costs: one lookup. The caller asked about a
    place and gets the finest answer the map holds for it, without being told
    a semantic gate refined it three frames ago."""
    x, y = 30.0, 0.0
    ring, slot = _write(gm, x, y, ground_cm=-5)
    assert ring >= 1

    block = gm.pool.acquire(gm.schedule, ring, slot, levels=1,
                            score=priority(30.0))
    assert block >= 0
    cells = gm.pool.block_cells(block)
    gm.pool.cells["ground_height"][cells] = -5
    gm.pool.cells["obs_count"][cells] = 9
    gm.pool.cells["log_odds"][cells] = 20

    # one child carries a kerb the parent cell averaged away
    child = _refined(gm, ring, slot, x, y)[1]
    gm.pool.cells["ground_height"][child] = 7

    assert query(gm, x, y).ground_height == pytest.approx(0.07)
    # a neighbouring parent cell, not refined, still answers from the grid
    assert query(gm, x + 5.0, y).ground_height != pytest.approx(0.07)


def test_queries_inside_one_parent_reach_different_children(gm):
    """The point of refining at all: two places inside one coarse cell must be
    able to answer differently."""
    x, y = 30.0, 0.0
    ring, slot = _write(gm, x, y)
    cell_m = gm.schedule.rings[ring].cell_m

    block = gm.pool.acquire(gm.schedule, ring, slot, 1, 0.5)
    gm.pool.cells["obs_count"][gm.pool.block_cells(block)] = 5

    seen = set()
    for dx in (0.05, cell_m - 0.05):
        for dy in (0.05, cell_m - 0.05):
            _, s = _refined(gm, ring, slot, x + dx, y + dy)
            seen.add(s)
    assert len(seen) == 4, "four quadrants of the parent hit one child"


# --- the frozen module-level pair --------------------------------------------


def test_bind_serves_the_frozen_signature(gm):
    """`api.query(x, y)` takes no map, so one has to be bound. Keeping the
    state explicit here means a test never depends on import order."""
    _write(gm, 1.0, 1.0, ground_cm=-3, trav=0)
    bind(gm)
    assert bound() is gm
    assert query_bound(1.0, 1.0).ground_height == pytest.approx(-0.03)
    assert is_traversable_bound(1.0, 1.0)


def test_query_is_free_of_the_frozen_api_s_state_by_default():
    """A GridMap built fresh must not see another test's bound map."""
    gm2 = build_gridmap(load("5/10/50"))
    assert query(gm2, 1.0, 1.0).occupancy == OCC_UNKNOWN


def test_occupied_and_free_are_distinguished(gm):
    _ring, slot = _write(gm, 7.0, 2.0)
    gm.soa["log_odds"][slot] = 30
    assert query(gm, 7.0, 2.0).occupancy == OCC_OCCUPIED

    gm.soa["log_odds"][slot] = -30
    assert query(gm, 7.0, 2.0).occupancy == OCC_FREE
    assert free_space(gm, 7.0, 2.0)
