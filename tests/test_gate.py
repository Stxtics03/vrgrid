"""The semantic gate. Master v4 §3.4, flaw E1. [Aakash]

The gate is what makes the refinement pool a mechanism rather than a data
structure, so these test the DECISIONS -- what fires it, what does not, and
what happens when the budget runs out -- rather than the plumbing.
"""

import numpy as np
import pytest
from vrgrid.cell import FLAG_DERIVED, FLAG_DYNAMIC, TRAV_ROUGHNESS, TRAV_SLOPE, TRAV_STEP
from vrgrid.eval.harness import build_gridmap, recenter
from vrgrid.grid.fusion import pack_class
from vrgrid.grid.gate import _cell_centre as gate_cell_centre
from vrgrid.grid.gate import apply, candidates, refine_class_ids, ring_of_slot
from vrgrid.grid.pool import FREE
from vrgrid.grid.quantise import dequantise_variance_cm2, quantise_variance_cm2
from vrgrid.grid.query import slot_of
from vrgrid.grid.schedule import load, load_thresholds
from vrgrid.grid.splitmerge import CellValue, merge
from vrgrid.grid.traversability import class_ids


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
    gm.soa["semantic_class"][slot] = pack_class(class_ids()[cls], 5)
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
    assert class_ids()["person"] in refine_class_ids()


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
        assert class_ids()[name] <= CLASS_MAX
        assert class_ids()[name] in ids, (
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


def test_release_happens_when_the_vehicle_drives_up_to_the_cell(gm):
    """⚑ Flaw E1's ordering, exercised by DRIVING rather than by pretending.

    A block whose cell has migrated inward is buying resolution the schedule
    now provides free; holding it while new requests are refused is the
    degradation E1 describes -- the pool going useless exactly as you approach
    the things you cared about.

    ⚑ This test used to monkeypatch `gate.ring_of_slot` to return 0 for every
      slot, and that monkeypatch was the tell. `ring_of_slot` answers which
      ring a flat SLOT is STORED in, which the allocation fixes at startup; it
      cannot change because the vehicle moved. So the release condition
      `now <= ring - levels` was unsatisfiable on the real path, nothing was
      ever released, and the only way to see a release was to replace the
      function with one that lies. Twelve frames of the synthetic sequence
      ended `released 0 ... pool 512/512 blocks` with 15,791 refusals.

      `apply` now asks `migrate_ring` where the cell IS, from its
      vehicle-relative centre. So this drives the vehicle at it instead, which
      is the thing that actually happens.
    """
    ring, slot = _cell(gm, 30.0, 0.0, cls="person")
    assert ring >= 2, "the cell must start coarse enough to be worth refining"
    apply(gm, [slot])
    block = gm.pool.find(ring, slot)
    assert block != FREE, "nothing was acquired, so the release proves nothing"

    # Parked: the cell has not moved, so the block is still buying something.
    assert apply(gm, [])["released"] == 0

    # Drive up to it. `_cell_centre` is vehicle-relative, so moving the
    # vehicle to the cell puts it in ring 0 -- which the schedule already
    # resolves at 5 cm, finer than the block was bought to provide.
    recenter(gm, 30.0, 0.0)
    out = apply(gm, [])
    assert out["released"] == 1, "a block the schedule overtook was not released"
    assert gm.pool.find(ring, slot) == FREE


def test_hysteresis_keeps_a_boundary_cell_from_thrashing_the_pool(gm):
    """§6.3's specified unit test, on the path that actually splits and merges.

    "Drive a synthetic trajectory with sinusoidal speed across a ring boundary;
    assert the number of split/merge events per cell is bounded." A cell
    sitting on a boundary while `v` fluctuates would otherwise acquire and
    release every frame: pool thrash, and by §5.4 variance inflation with no
    physical cause.

    The anisotropy is what makes speed move the boundary at all. Note it is
    the LATERAL edge that moves here, not the forward one: eq. (20) divides
    |y| by `a_s < 1`, so a point to the side is pushed OUT to a coarser ring
    as the vehicle speeds up -- resolution taken from the sides and spent
    forward, which is the whole idea. At this schedule's `kappa_forward = 1.0`
    a cell dead ahead at the ring 0/1 edge does not change ring at all, so a
    forward cell would have made this test vacuous. A cell parked on the
    moving lateral edge is the worst case, and `migrate_ring`'s asymmetric
    thresholds (eq. 21) are what bound it.

    ⚑ `migrate_ring` had no caller outside its own unit test until 2 Sep, so
      this could not have been asserted before: there was nothing on the frame
      path for the hysteresis to protect.
    """
    from vrgrid.grid.lattice import d_aniso, ring_of

    sched = gm.schedule
    # Lateral, and between the ring 1/2 edges at rest and at speed: ring 1 at
    # 0 m/s, ring 2 at 30 m/s, so the boundary sweeps over it every cycle.
    y = 14.0
    assert ring_of(0.0, y, sched, 0.0) != ring_of(0.0, y, sched, 30.0), (
        "this y is not on a speed-sensitive boundary, so the test is vacuous")

    _ring, slot = _cell(gm, 0.0, y, cls="person")
    var_before = int(gm.soa["height_variance"][slot])

    # ⚑ Count pool OCCUPANCY changes, not `acquired`. `pool.acquire` is
    #   idempotent for a (ring, slot) it already holds, so `acquired` counts
    #   the gate re-affirming a block it already has -- 1 every frame, on a
    #   cell that never moved. Occupancy is the thing §6.3 bounds: a split
    #   takes a block, a merge gives it back.
    events, held = 0, gm.pool.free_blocks
    for frame in range(1000):
        speed = 15.0 + 15.0 * np.sin(frame * 0.35)     # 0 .. 30 m/s
        apply(gm, [slot], vehicle_speed_ms=speed)
        if gm.pool.free_blocks != held:
            events += 1
            held = gm.pool.free_blocks

    # One acquisition on the first frame, and then nothing: the band holds it
    # across every crossing. Generous on purpose -- the claim is "bounded",
    # not a tuned count.
    assert events <= 4, (
        f"{events} pool occupancy changes in 1000 frames on one boundary "
        "cell -- the hysteresis band is not holding it")

    # §5.4's half of the same argument: a cell oscillating across a boundary
    # must not inflate its variance every cycle with no physical cause. The
    # `derived` bit is what makes `merge(split(c)) == c` exact, and this is
    # the 1000-frame check §6.3 asks for.
    assert int(gm.soa["height_variance"][slot]) == var_before, (
        "the boundary cell's variance moved over 1000 frames of speed "
        "oscillation, with no new evidence")

    # And the band is genuinely doing the holding, rather than the cell never
    # being near the edge: the anisotropy really does move it, outward, as
    # eq. (20)'s lateral term says.
    assert d_aniso(0.0, y, sched, 30.0) > d_aniso(0.0, y, sched, 0.0)


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
    ab.soa["semantic_class"][slot] = pack_class(class_ids()["person"], 5)

    out = apply(ab, [slot])
    assert out["unfit"] == 1
    assert out["acquired"] == 0
    assert ab.pool.free_blocks == ab.pool.blocks
