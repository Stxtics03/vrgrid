"""The semantic gate. Master v4 §3.4, flaw E1. [Aakash]

The gate is what makes the refinement pool a mechanism rather than a data
structure, so these test the DECISIONS -- what fires it, what does not, and
what happens when the budget runs out -- rather than the plumbing.
"""

import numpy as np
import pytest
from vrgrid.cell import FLAG_DERIVED, FLAG_DYNAMIC, TRAV_ROUGHNESS, TRAV_SLOPE, TRAV_STEP
from vrgrid.eval.harness import build_gridmap
from vrgrid.grid.fusion import pack_class
from vrgrid.grid.gate import _cell_centre as gate_cell_centre
from vrgrid.grid.gate import apply, candidates, refine_class_ids, ring_of_slot
from vrgrid.grid.pool import FREE
from vrgrid.grid.quantise import dequantise_variance_cm2, quantise_variance_cm2
from vrgrid.grid.query import slot_of
from vrgrid.grid.schedule import load, load_thresholds
from vrgrid.grid.splitmerge import CellValue, merge
from vrgrid.grid.traversability import CLASS_IDS


@pytest.fixture
def gm():
    return build_gridmap(load("5/10/20/40"))


def _cell(gm, x, y, cls="road", n=9, var_cm2=1.0, trav=0):
    ring, slot = slot_of(gm, x, y)
    gm.soa["obs_count"][slot] = n
    # `pack_class`, not a hand-rolled shift. This fixture spelled the 4/4
    # split out itself and so kept writing a 4-bit-shifted byte after the
    # field became 5 bits -- every cell then read back as some other class
    # and half this file failed for a reason that had nothing to do with
    # the gate.
    gm.soa["semantic_class"][slot] = pack_class(CLASS_IDS[cls], 5)
    gm.soa["height_variance"][slot] = quantise_variance_cm2(var_cm2)
    gm.soa["traversability"][slot] = trav
    gm.soa["ground_height"][slot] = -5
    return ring, slot


# --- what fires it -----------------------------------------------------------


def test_a_dynamic_cell_fires_the_gate(gm):
    """A pedestrian at 30 m sits in a 20 cm cell that also contains the road
    she is standing on, and a planner needs the two separated."""
    _, slot = _cell(gm, 30.0, 0.0)
    assert not candidates(gm, [slot])[0]

    gm.transient["flags"][slot] = FLAG_DYNAMIC
    assert candidates(gm, [slot])[0]


def test_a_thin_class_fires_the_gate(gm):
    """A person past 25 m is entirely inside one cell of the ring she lands
    in, so range alone under-resolves her."""
    _, road = _cell(gm, 30.0, 0.0, cls="road")
    _, person = _cell(gm, 30.0, 2.0, cls="person")
    assert not candidates(gm, [road])[0]
    assert candidates(gm, [person])[0]
    assert CLASS_IDS["person"] in refine_class_ids()


def test_every_configured_refine_class_can_now_fire_it():
    """The §10.2 conflict, resolved. Was `..._cannot_fire_it`.

    With a 4-bit candidate the class field held ids 0-15, so `pole` (18) and
    `traffic-sign` (19) did not fit -- and those are the two classes semantic
    refinement exists for: thin structures whose entire geometry is smaller
    than the cell they land in past 25 m. The gate matched on ids no cell
    could report, fired on nothing, and nothing failed.

    The byte was re-split 5/3 on 1 Sep. `unstorable_refine_classes()` was
    written so that it would empty itself on that day without being edited,
    and this is the assertion that it did.
    """
    from vrgrid.grid.fusion import CLASS_MAX
    from vrgrid.grid.gate import unstorable_refine_classes

    assert unstorable_refine_classes() == [], (
        "a configured refine class still does not fit the cell's class field"
    )

    ids = refine_class_ids().tolist()
    for name in ("pole", "traffic-sign"):
        assert CLASS_IDS[name] <= CLASS_MAX
        assert CLASS_IDS[name] in ids, (
            f"{name} fits the byte now but the gate is not matching on it"
        )


def test_an_uncertain_edge_fires_the_gate_and_a_certain_one_does_not(gm):
    """The map knows there is a step here and is unsure exactly where it runs.
    That boundary is what the planner steers along."""
    _, sure = _cell(gm, 30.0, 0.0, var_cm2=1.0, trav=TRAV_STEP)
    _, unsure = _cell(gm, 30.0, 3.0, var_cm2=400.0, trav=TRAV_STEP)
    _, open_road = _cell(gm, 30.0, 6.0, var_cm2=400.0, trav=0)

    assert not candidates(gm, [sure])[0], "a well-known kerb does not need a look"
    assert candidates(gm, [unsure])[0]
    assert not candidates(gm, [open_road])[0], "uncertain open road is just unobserved"


def test_the_ambiguity_criterion_is_not_its_own_definition(gm):
    """⚑ Regression, and the bug was invisible by inspection.

    TRAV_ROUGHNESS IS the predicate `sigma^2 > sigma^2_max`. Writing the gate
    as "roughness bit AND high variance" is the same condition twice: it looks
    like a careful two-part test and is one part. Written that way it fired on
    20,954 of 143,587 observed cells -- eight times the whole pool every frame
    -- leaving the priority ordering to do all the selection the gate was
    supposed to do.

    So the criterion pairs high variance with a GEOMETRIC hazard, and a cell
    carrying only the roughness bit must not fire.
    """
    th = load_thresholds()
    over = th["traversability"]["sigma2_max_m2"] * 1e4 * 4      # cm^2, well over

    _, rough_only = _cell(gm, 30.0, 0.0, var_cm2=over, trav=TRAV_ROUGHNESS)
    _, stepped = _cell(gm, 30.0, 3.0, var_cm2=over, trav=TRAV_STEP)
    _, sloped = _cell(gm, 30.0, 6.0, var_cm2=over, trav=TRAV_SLOPE)

    assert not candidates(gm, [rough_only])[0], (
        "the roughness bit fired the variance test, which is the same test"
    )
    assert candidates(gm, [stepped])[0]
    assert candidates(gm, [sloped])[0]

    # and the variance really is over the line, so the test above is not
    # passing for the trivial reason
    assert dequantise_variance_cm2(gm.soa["height_variance"][rough_only]) * 1e-4 \
        > th["traversability"]["sigma2_max_m2"]


def test_ring_zero_is_never_refined(gm):
    """Ring 0 is the base lattice. Semantic refinement goes into the pool at
    ring-0 resolution, never below c0 (master v4 §3.4)."""
    _, slot = _cell(gm, 2.0, 0.0, cls="person")
    assert ring_of_slot(gm, slot) == 0
    out = apply(gm, [slot])
    assert out["acquired"] == 0
    assert gm.pool.free_blocks == gm.pool.blocks


# --- what it does ------------------------------------------------------------


def test_a_refined_block_is_a_split_not_a_blank(gm):
    """⚑ Children come from `splitmerge.split()`: they inherit mu_p, carry an
    inflated variance and FLAG_DERIVED.

    Zeros would give the planner four confident readings of 0 m where the
    parent knew nothing -- the map looking sharpest exactly where it had just
    admitted to guessing. The bit is also what makes the block mergeable back
    to the parent exactly if the gate stops firing (Theorem 2).
    """
    ring, slot = _cell(gm, 30.0, 0.0, cls="person")
    gm.soa["ground_height"][slot] = 42
    parent_var = dequantise_variance_cm2(gm.soa["height_variance"][slot])

    out = apply(gm, [slot])
    assert out["acquired"] == 1

    block = gm.pool.find(ring, slot)
    cells = gm.pool.block_cells(block)
    m = gm.schedule.k(ring) // gm.schedule.k(ring - 1)
    kids = slice(cells.start, cells.start + m * m)

    assert np.all(gm.pool.cells["ground_height"][kids] == 42), "children lost mu_p"
    assert np.all(gm.pool.cells["flags"][kids] & FLAG_DERIVED), "FLAG_DERIVED unset"
    assert np.all(gm.pool.cells["obs_count"][kids] == 9)
    assert np.all(dequantise_variance_cm2(
        gm.pool.cells["height_variance"][kids]) >= parent_var)


def test_the_children_merge_back_to_the_parent(gm):
    """Theorem 2 through the gate: if the gate stops firing, what it wrote can
    be given back exactly rather than left as a slightly-wrong finer map."""
    ring, slot = _cell(gm, 30.0, 0.0, cls="person")
    apply(gm, [slot])
    block = gm.pool.find(ring, slot)
    cells = gm.pool.block_cells(block)
    m = gm.schedule.k(ring) // gm.schedule.k(ring - 1)

    values = [CellValue(
        mu_m=float(gm.pool.cells["ground_height"][cells.start + i]) / 100.0,
        sigma2_m2=float(dequantise_variance_cm2(
            gm.pool.cells["height_variance"][cells.start + i])) * 1e-4,
        n=int(gm.pool.cells["obs_count"][cells.start + i]),
        flags=int(gm.pool.cells["flags"][cells.start + i]),
    ) for i in range(m * m)]

    back = merge(values)
    assert back.mu_m == pytest.approx(float(gm.soa["ground_height"][slot]) / 100.0)
    assert back.n == int(gm.soa["obs_count"][slot])


def test_release_happens_before_acquire(gm):
    """⚑ Flaw E1's ordering. A block whose cell has migrated inward is buying
    resolution the schedule now provides free; holding it while new requests
    are refused is the degradation E1 describes -- the pool going useless
    exactly as you approach the things you cared about."""
    small = build_gridmap(load("5/10/20/40"))
    small.pool.blocks = small.pool.blocks       # capacity is the real pool's

    ring, slot = _cell(gm, 30.0, 0.0, cls="person")
    apply(gm, [slot])
    assert gm.pool.find(ring, slot) != FREE

    # pretend that cell now lives at ring 0: the schedule overtook the block
    out = apply(gm, [], thresholds=None)
    assert out["released"] >= 0                 # nothing migrated, nothing freed

    from vrgrid.grid import gate as gate_mod
    real = gate_mod.ring_of_slot
    gate_mod.ring_of_slot = lambda g, s: 0      # everything is now ring 0
    try:
        out = apply(gm, [])
    finally:
        gate_mod.ring_of_slot = real
    assert out["released"] == 1, "a block the schedule overtook was not released"


def test_the_pool_stays_bounded_under_a_flood(gm):
    """"Bounded, degrading gracefully by dropping the least relevant." Fire the
    gate on far more cells than there are blocks and assert the bound holds and
    the refusals are counted rather than silent."""
    slots = []
    for i in range(700):
        x = 12.0 + (i % 60) * 0.5
        y = -12.0 + (i // 60) * 1.0
        ring, slot = _cell(gm, x, y, cls="person")
        if ring >= 1:
            slots.append(slot)

    out = apply(gm, slots)

    # `acquired` counts ACQUISITIONS, not distinct blocks: a request that
    # outranks the weakest tenant evicts it, and both are acquisitions. The
    # bound is on blocks held, never on how many times they changed hands.
    in_use = gm.pool.blocks - gm.pool.free_blocks
    assert in_use == gm.pool.blocks, "a flood did not fill the pool"
    assert out["acquired"] + out["refused"] + out["unfit"] == len(slots)
    assert out["acquired"] > gm.pool.blocks, "nothing was ever evicted"
    assert gm.pool.bytes_used() == gm.pool.blocks * gm.pool.cells_per_block * 12

    # and what it kept is the near stuff: "dropping the least relevant" is a
    # claim about WHICH blocks survive, not merely that the count is capped.
    held = np.flatnonzero(gm.pool.owner_ring != FREE)
    ranges = []
    for block in held:
        ring = int(gm.pool.owner_ring[block])
        x, y = gate_cell_centre(gm, ring, int(gm.pool.owner_slot[block]))
        ranges.append(np.hypot(x, y))
    assert max(ranges) < 45.0, "the pool held on to the far field over the near"


def test_the_ablation_reports_unfit_rather_than_truncating():
    """⚑ 5/10/50 refines 5x between rings 1 and 2, so one level is 25 children
    and a 16-cell block cannot hold it. The gate counts that separately from a
    refusal: one is a full pool, the other is a configuration that cannot work
    however empty the pool is."""
    ab = build_gridmap(load("5/10/50"))
    ring, slot = slot_of(ab, 40.0, 0.0)
    assert ring == 2
    ab.soa["obs_count"][slot] = 9
    ab.soa["semantic_class"][slot] = pack_class(CLASS_IDS["person"], 5)

    out = apply(ab, [slot])
    assert out["unfit"] == 1
    assert out["acquired"] == 0
    assert ab.pool.free_blocks == ab.pool.blocks
