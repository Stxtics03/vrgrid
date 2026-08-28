"""Refinement pool. Master v4 §3.4, flaw E1. [Aakash]"""

import numpy as np
import pytest
from vrgrid.grid.pool import FREE, RefinementPool, priority
from vrgrid.grid.schedule import load


def _pool(blocks=8, cells=16):
    return RefinementPool(blocks=blocks, cells_per_block=cells)


def test_priority_is_not_the_literal_product_of_its_names():
    """closeness x dynamism x time-to-collision, with every factor inverted
    from the quantity it is named after.

    The trap this asserts away: range, a dynamism flag and TTC all get LARGER
    for things that matter LESS. A literal product ranks a static kerb 90 m
    away with no collision in sight above a pedestrian stepping off the kerb
    in front of you, and the pool then evicts exactly the wrong block.
    """
    near_moving_soon = priority(3.0, is_dynamic=True, ttc_s=1.0)
    far_static_never = priority(90.0, is_dynamic=False, ttc_s=np.inf)
    assert near_moving_soon > far_static_never

    # and "no collision course" must not zero the product: it is most of the
    # map, and a pool that only refines imminent collisions refines nothing
    assert far_static_never > 0.0

    # each factor on its own, holding the others fixed
    assert priority(3.0) > priority(30.0)                       # closeness
    assert priority(10.0, is_dynamic=True) > priority(10.0)     # dynamism
    assert priority(10.0, ttc_s=0.5) > priority(10.0, ttc_s=20.0)   # urgency


def test_acquire_release_and_lookup():
    p = _pool()
    s = load("5/10/20/40")
    assert p.free_blocks == 8

    b = p.acquire(s, ring=3, slot=1234, levels=1, score=0.5)
    assert b != FREE
    assert p.find(3, 1234) == b
    assert p.find(3, 9999) == FREE
    assert p.free_blocks == 7

    # asking again for the same cell returns the same block, not a second one
    assert p.acquire(s, 3, 1234, 1, 0.9) == b
    assert p.free_blocks == 7

    p.release(b)
    assert p.find(3, 1234) == FREE
    assert p.free_blocks == 8


def test_eviction_takes_the_least_important_block_and_only_if_it_loses():
    """"Bounded, degrading gracefully by dropping the least relevant" -- the
    slide phrasing from master v4 §3.4, as a property.

    And the other half, which is easy to leave out: a full pool must REFUSE a
    request that is less important than everything in it. Evicting something
    urgent to make room for something idle is worse than not refining at all.
    """
    p = _pool(blocks=3)
    s = load("5/10/20/40")
    for i, score in enumerate((0.9, 0.1, 0.5)):
        assert p.acquire(s, 3, i, 1, score) != FREE
    assert p.free_blocks == 0

    # a more important request evicts the 0.1, not the 0.9
    b = p.acquire(s, 3, 99, 1, score=0.7)
    assert b != FREE
    assert p.find(3, 1) == FREE, "evicted the wrong block"
    assert p.find(3, 0) != FREE and p.find(3, 2) != FREE

    # a less important one is refused rather than served at someone's expense
    assert p.acquire(s, 3, 100, 1, score=0.05) == FREE
    assert p.find(3, 99) != FREE


def test_a_reused_block_does_not_inherit_the_last_tenant_s_cells():
    """Stale children are worse than no children: they are plausible heights
    in the right place for the wrong cell."""
    p = _pool(blocks=1)
    s = load("5/10/20/40")
    b = p.acquire(s, 3, 7, 1, 0.2)
    p.cells["ground_height"][p.block_cells(b)] = 123
    p.cells["obs_count"][p.block_cells(b)] = 9

    p.release(b)
    b2 = p.acquire(s, 3, 8, 1, 0.2)
    assert np.all(p.cells["ground_height"][p.block_cells(b2)] == 0)
    assert np.all(p.cells["obs_count"][p.block_cells(b2)] == 0)


# --- flaw E1 -----------------------------------------------------------------


def test_release_overtaken_frees_what_the_schedule_now_provides():
    """⚑ Flaw E1, and the reason refinement is defined in LEVELS.

    A block refining a ring-3 cell one level is buying 20 cm. Drive toward it,
    the cell migrates to ring 2, and ring 2 *is* 20 cm -- the schedule now
    provides for free exactly what the block is paying for. Held as an
    absolute cell size the block still looks busy, is priority-protected
    because the cell is now close, and never leaves; the pool fills with
    blocks that buy nothing, precisely as you approach the things you cared
    about.
    """
    p = _pool(blocks=4)
    s = load("5/10/20/40")
    p.acquire(s, ring=3, slot=10, levels=1, score=0.9)   # buying 20 cm
    p.acquire(s, ring=3, slot=11, levels=2, score=0.9)   # buying 10 cm
    p.acquire(s, ring=2, slot=12, levels=1, score=0.9)   # buying 10 cm

    # every one of those cells has migrated inward to ring 2 (20 cm)
    assert p.release_overtaken(lambda ring, slot: 2) == 1
    assert p.find(3, 10) == FREE, "20 cm block survived the schedule reaching 20 cm"
    assert p.find(3, 11) != FREE, "10 cm block released while still buying 10 cm"
    assert p.find(2, 12) != FREE

    # migrate further: ring 1 is 10 cm, which overtakes both survivors
    assert p.release_overtaken(lambda ring, slot: 1) == 2
    assert p.free_blocks == 4


def test_release_overtaken_keeps_blocks_that_still_buy_something():
    p = _pool(blocks=4)
    s = load("5/10/20/40")
    p.acquire(s, ring=3, slot=1, levels=2, score=0.5)
    # the cell has not moved; the block is still buying two levels
    assert p.release_overtaken(lambda ring, slot: 3) == 0
    assert p.find(3, 1) != FREE


# --- ⚑ the ablation does not fit ---------------------------------------------


def test_sixteen_cells_per_block_cannot_hold_one_level_of_the_ablation():
    """⚑ `cells_per_block: 16` holds a 4x4 subdivision -- two levels at ratio
    2, which is every boundary of 5/10/20/40. The 5/10/50 ablation refines 5x
    between rings 1 and 2, so ONE level there is 25 children, larger than an
    entire block.

    Refused loudly. Truncating a 5x5 refinement into 16 cells drops nine
    children and leaves them reading as whatever the block held before, which
    is a plausible map with nine wrong cells per refined cell.

    Only bites if a semantic gate fires on the ablation, which is a Day-3
    question -- but the number is wrong now and it will not announce itself.
    """
    p = _pool()
    default, ablation = load("5/10/20/40"), load("5/10/50")

    assert p.levels_available(default, 3) == 2       # 40 -> 20 -> 10 cm, 16 cells
    assert p.levels_available(ablation, 1) == 1      # 10 -> 5 cm; nothing below c0
    assert p.levels_available(ablation, 2) == 0      # 50 -> 10 cm, ratio 5, 25 cells

    assert p.children_per_level(ablation, 2) == 25
    with pytest.raises(ValueError, match="needs 25 cells"):
        p.acquire(ablation, ring=2, slot=0, levels=1, score=1.0)

    # 32 would fit it, which is what makes this a config decision
    assert RefinementPool(8, 32).levels_available(ablation, 2) == 1


def test_ring_zero_cannot_be_refined():
    p = _pool()
    with pytest.raises(ValueError, match="base lattice"):
        p.children_per_level(load("5/10/20/40"), 0)


# --- the memory bound --------------------------------------------------------


def test_pool_is_fixed_at_98_kb_and_never_grows():
    """512 x 16 x 12 B = 98,304 B, master v4 §3.4. A headline figure, so it is
    asserted rather than assumed -- and asserted again after churn, because
    the failure mode is a pool that grows quietly under load."""
    p = RefinementPool(512, 16)
    assert p.bytes_used() == 512 * 16 * 12

    s = load("5/10/20/40")
    before = {k: v.nbytes for k, v in p.cells.items()}
    rng = np.random.default_rng(3)
    for i in range(4000):
        slot = int(rng.integers(0, 100_000))
        block = p.acquire(s, 3, slot, 1, float(rng.random()))
        if block != FREE and i % 7 == 0:
            p.release(block)
        p.release_overtaken(lambda ring, slot: 3)

    assert {k: v.nbytes for k, v in p.cells.items()} == before
    assert p.bytes_used() == 98_304
    assert len(p.owner_ring) == 512
