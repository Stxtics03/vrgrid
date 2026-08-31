"""The conservative pyramid. Math §7.2-7.3. [Shrestha]

Theorem 3 is a proof, not a tuning target. `test_theorem3_has_no_false_
negatives` is the exhaustive check §7.3 asks for, and a counterexample there
means the implementation is wrong -- never that the threshold should move.
"""

import numpy as np
import pytest
import yaml
from vrgrid.api import QueryLOD
from vrgrid.cell import (
    TRAV_CLEARANCE,
    TRAV_CONFIDENCE,
    TRAV_SLOPE,
    TRAV_STEP,
)
from vrgrid.gpu.allocators import allocate, derive_ring_layouts
from vrgrid.gpu.kernels import CEILING_NONE
from vrgrid.gpu.pyramid import (
    NODE_BYTES,
    Pyramid,
    all_clear,
    allocate_pyramid,
    block_extent,
    block_slots,
    build,
    certainly_blocked,
    classify,
    level_arrays,
    level_sides,
    pyramid_bytes,
    theorem3_safe,
)
from vrgrid.grid.schedule import load


@pytest.fixture(scope="module")
def thresholds():
    with open("configs/thresholds.yaml") as f:
        return yaml.safe_load(f)


# --- a small stand-in schedule, so the exhaustive tests stay quick -----------
#
# The reductions do not care how big a ring is, and a 910,000-cell pyramid
# rebuilt a few hundred times is a slow test rather than a stronger one. Sides
# 8 and 5 are chosen deliberately: 8 halves cleanly to 1 and 5 does not, so
# both the even path and the ceiling-halved odd path are exercised everywhere.


class FakeRing:
    def __init__(self, ring, side, offset, cell_m=0.05):
        self.ring, self.side, self.offset, self.cell_m = ring, side, offset, cell_m
        self.count = side * side

    @property
    def slots(self):
        return self.side * self.side


def small_rings():
    return [FakeRing(0, 8, 0), FakeRing(1, 5, 64)]


def empty_soa(rings):
    n = sum(r.slots for r in rings)
    return {
        "ground_height": np.zeros(n, np.int16),
        "ceiling_height": np.full(n, CEILING_NONE, np.int16),
        "obs_count": np.zeros(n, np.uint8),
        "traversability": np.zeros(n, np.uint8),
    }


def random_soa(rng, rings):
    """Adversarial: every field independent and uniform. No structure at all,
    so the reductions get no help from locality."""
    n = sum(r.slots for r in rings)
    return {
        "ground_height": rng.integers(-200, 200, n).astype(np.int16),
        "ceiling_height": rng.choice(
            [CEILING_NONE, 150, 300, 600], size=n).astype(np.int16),
        "obs_count": rng.integers(0, 12, n).astype(np.uint8),
        "traversability": rng.integers(0, 64, n).astype(np.uint8),
    }


def terrain_soa(rng, rings):
    """A map shaped like ground: a gently sloping, mostly-observed surface with
    kerbs, overhangs and holes in the data punched into it.

    The adversarial map above is the harder test of the reductions and the
    weaker test of the THEOREM: with heights uniform over 4 m, almost no block
    has a spread under the 15 cm step threshold, so SAFE fires a handful of
    times and a theorem test that never sees its antecedent proves nothing.
    Real terrain is locally smooth, which is the case where SAFE is both common
    and worth getting right.
    """
    n = sum(r.slots for r in rings)
    ground = np.empty(n, np.int16)
    for layout in rings:
        side = layout.side
        iy, ix = np.mgrid[0:side, 0:side]
        # ~2 cm per cell of tilt: a real grade, well inside every threshold
        plane = -173.0 + 2.0 * ix + 1.4 * iy
        z = plane + rng.integers(-2, 3, (side, side))
        hazard = rng.random((side, side)) < 0.08          # kerbs and potholes
        z[hazard] += rng.integers(20, 60, int(hazard.sum()))
        ground[layout.offset: layout.offset + layout.slots] = z.ravel()

    ceiling = np.full(n, CEILING_NONE, np.int16)
    low = rng.random(n) < 0.06                            # overhangs
    ceiling[low] = rng.integers(-100, 500, int(low.sum()))

    obs = rng.integers(3, 20, n).astype(np.uint8)
    thin = rng.random(n) < 0.10                           # holes in the data
    obs[thin] = rng.integers(0, 3, int(thin.sum()))
    return {
        "ground_height": ground,
        "ceiling_height": ceiling,
        "obs_count": obs,
        "traversability": rng.integers(0, 64, n).astype(np.uint8),
    }


# --- level geometry ----------------------------------------------------------


def test_levels_halve_by_ceiling_all_the_way_to_one_node():
    assert level_sides(8) == [8, 4, 2, 1]
    assert level_sides(5) == [5, 3, 2, 1]
    assert level_sides(1) == [1]


def test_a_floor_would_have_dropped_the_map_edge():
    """The regression this pins. Ring windows are 400 and 500 across, neither
    a power of two, so floor-halving 500 gives 250, 125, 62 -- and that 62
    silently drops the last row and column of a 125-wide level. At the map
    edge, where nothing looks wrong."""
    assert level_sides(500) == [500, 250, 125, 63, 32, 16, 8, 4, 2, 1]
    assert level_sides(400) == [400, 200, 100, 50, 25, 13, 7, 4, 2, 1]

    # every level covers its whole child level, edge included
    for side in (400, 500, 5, 7):
        sides = level_sides(side)
        for level in range(1, len(sides)):
            covered = sum(
                (r1 - r0) * (c1 - c0)
                for i in range(sides[level] ** 2)
                for r0, r1, c0, c1 in [block_extent(
                    sides, level, i // sides[level], i % sides[level])]
            )
            assert covered == side * side, f"side {side}, level {level}"


def test_blocks_at_a_level_partition_the_window():
    """Exactly one block per cell, per level -- the same property the lattice
    partition test asserts, one dimension up. Overlap would double-count a
    hazard into two blocks and a gap would hide one from both."""
    for side in (8, 5, 13):
        sides = level_sides(side)
        for level in range(len(sides)):
            seen = np.concatenate(
                [block_slots(sides, level, i) for i in range(sides[level] ** 2)])
            assert np.array_equal(np.sort(seen), np.arange(side * side))


# --- the reductions ----------------------------------------------------------


def test_the_reductions_are_max_min_and_or_over_the_block():
    """Checked against a direct reduction over each block's own cells, at
    every level, on both an even and an odd side."""
    rng = np.random.default_rng(20260829)
    rings = small_rings()
    soa = random_soa(rng, rings)
    pyr = allocate_pyramid(rings)
    build(pyr, soa, rings)

    for r, layout in enumerate(rings):
        sides = pyr.sides[r]
        base = slice(layout.offset, layout.offset + layout.slots)
        ground = soa["ground_height"][base]
        ceiling = soa["ceiling_height"][base]
        obs = soa["obs_count"][base]
        trav = soa["traversability"][base]

        for level in range(1, len(sides)):
            nodes = level_arrays(pyr, soa, rings, r, level)
            for i in range(sides[level] ** 2):
                cells = block_slots(sides, level, i)
                assert nodes["h_max"][i] == ground[cells].max()
                assert nodes["h_min"][i] == ground[cells].min()
                assert nodes["c_min"][i] == ceiling[cells].min()
                assert nodes["n_min"][i] == obs[cells].min()
                assert nodes["and_mask"][i] == np.bitwise_and.reduce(trav[cells])
                assert nodes["or_mask"][i] == np.bitwise_or.reduce(trav[cells])


def test_means_are_not_used_because_a_kerb_would_vanish():
    """§7.2's reason for max/min in one case. Four cells straddling a 30 cm
    kerb: the mean is flat and drivable, the max/min pair is a 30 cm step and
    is not. If this ever reports a mean, the safety claim is gone."""
    rings = [FakeRing(0, 2, 0)]
    soa = empty_soa(rings)
    soa["ground_height"][:] = [0, 0, 30, 30]
    soa["obs_count"][:] = 9

    pyr = allocate_pyramid(rings)
    build(pyr, soa, rings)
    top = level_arrays(pyr, soa, rings, 0, 1)

    assert top["h_max"][0] == 30 and top["h_min"][0] == 0
    assert top["h_max"][0] - top["h_min"][0] == 30      # a kerb, not 15 cm of nothing


def test_level_zero_is_the_grid_itself_not_a_copy():
    """Storing it would cost 8.2 MB to duplicate what is already in memory, so
    `level_arrays` serves views -- and a block of one cell really is its own
    maximum and its own minimum."""
    rings = small_rings()
    soa = empty_soa(rings)
    base = level_arrays(pyr_of(rings), soa, rings, 0, 0)

    assert base["h_max"].base is soa["ground_height"]
    assert np.shares_memory(base["h_min"], soa["ground_height"])
    soa["ground_height"][3] = 77
    assert base["h_max"][3] == 77 and base["h_min"][3] == 77


def pyr_of(rings):
    return allocate_pyramid(rings)


def test_a_pyramid_cannot_be_reused_across_schedules():
    """It is sized by the ring geometry, and quietly reducing a 500-wide ring
    into buffers cut for 400 would read past the level and produce a plausible
    map of the wrong place."""
    rings = small_rings()
    pyr = allocate_pyramid(rings)
    soa = empty_soa(rings)

    with pytest.raises(ValueError, match="cannot be reused"):
        build(pyr, soa, rings[:1])
    with pytest.raises(ValueError, match="cells across"):
        build(pyr, empty_soa([FakeRing(0, 4, 0), FakeRing(1, 5, 16)]),
              [FakeRing(0, 4, 0), FakeRing(1, 5, 16)])


# --- §7.3, Theorem 3 ---------------------------------------------------------


def test_theorem3_has_no_false_negatives(thresholds):
    """§7.3's exhaustive unit test, and the reason this structure is allowed
    to carry a safety claim.

        If SAFE(B) then every cell in B is traversable on conditions 0, 2
        and 5.

    Checked cell by cell against the raw quantities for every block at every
    level, over many random maps -- not against the per-cell bitfield, which
    would only prove the two computations agree. A single counterexample is a
    failed proof, not a threshold to tune.
    """
    t = thresholds["traversability"]
    s_max_cm = t["s_max_m"] * 100.0
    h_vehicle_cm = t["h_vehicle_m"] * 100.0

    rng = np.random.default_rng(731)
    rings = small_rings()
    pyr = allocate_pyramid(rings)

    # Both map shapes: the adversarial one exercises the reductions hardest,
    # the terrain one is where SAFE actually fires often enough to be a test.
    maps = ([terrain_soa(rng, rings) for _ in range(250)]
            + [random_soa(rng, rings) for _ in range(60)])

    checked = 0
    for soa in maps:
        build(pyr, soa, rings)

        for r, layout in enumerate(rings):
            sides = pyr.sides[r]
            base = slice(layout.offset, layout.offset + layout.slots)
            ground = soa["ground_height"][base].astype(np.int32)
            ceiling = soa["ceiling_height"][base].astype(np.int32)
            obs = soa["obs_count"][base].astype(np.int32)

            for level in range(1, len(sides)):
                nodes = level_arrays(pyr, soa, rings, r, level)
                safe = theorem3_safe(nodes, thresholds)
                for i in np.flatnonzero(safe):
                    cells = block_slots(sides, level, i)
                    z, c, n = ground[cells], ceiling[cells], obs[cells]

                    # bit 0, clearance -- for every cell, not on average
                    assert np.all(c - z > h_vehicle_cm), (
                        f"ring {r} level {level} block {i}: SAFE with a cell "
                        f"whose clearance is {int((c - z).min())} cm")
                    # bit 2, step -- over every PAIR in the block
                    assert z.max() - z.min() < s_max_cm
                    # bit 5, confidence
                    assert np.all(n >= t["n_min"])
                    checked += 1

    assert checked > 2000, f"only {checked} SAFE blocks seen; test is too weak"


def test_theorem3_is_not_vacuous(thresholds):
    """Negative control. The theorem test above passes trivially on an
    implementation where SAFE is never true, which is exactly the shape a
    broken reduction takes -- one `min` written as a `max` and n_min is 0
    everywhere. So: a map that IS safe must come back SAFE."""
    rings = [FakeRing(0, 4, 0)]
    soa = empty_soa(rings)
    soa["ground_height"][:] = 5
    soa["ceiling_height"][:] = 400          # 3.95 m of clearance
    soa["obs_count"][:] = 9

    pyr = allocate_pyramid(rings)
    build(pyr, soa, rings)
    for level in range(1, len(pyr.sides[0])):
        nodes = level_arrays(pyr, soa, rings, 0, level)
        assert np.all(theorem3_safe(nodes, thresholds)), f"level {level}"


@pytest.mark.parametrize("field, value, why", [
    ("obs_count", 1, "n below n_min must fail bit 5"),
    ("ceiling_height", 100, "0.95 m of clearance must fail bit 0"),
    ("ground_height", 100, "a 95 cm step must fail bit 2"),
])
def test_one_bad_cell_denies_the_whole_block(thresholds, field, value, why):
    """The property that makes it conservative: SAFE is about EVERY cell, so a
    single cell out of sixteen has to take the block, and every block above it
    on the way to the root."""
    rings = [FakeRing(0, 4, 0)]
    soa = empty_soa(rings)
    soa["ground_height"][:] = 5
    soa["ceiling_height"][:] = 400
    soa["obs_count"][:] = 9
    soa[field][10] = value                  # one cell, in the interior

    pyr = allocate_pyramid(rings)
    build(pyr, soa, rings)

    sides = pyr.sides[0]
    for level in range(1, len(sides)):
        nodes = level_arrays(pyr, soa, rings, 0, level)
        safe = theorem3_safe(nodes, thresholds)
        for i in np.flatnonzero(~safe):
            pass
        containing = [i for i in range(sides[level] ** 2)
                      if 10 in block_slots(sides, level, i)]
        assert not safe[containing].any(), f"{why}, level {level}"


def test_an_unseen_ceiling_does_not_wrap_into_a_low_one(thresholds):
    """C_min is 32767 where nothing overhead has been seen. `C_min - H_max` at
    int16 width wraps that to a large negative clearance -- turning "the sky
    is clear" into "there is a ceiling at the ground", on precisely the cells
    with the least evidence. Computed in int32 for that reason."""
    rings = [FakeRing(0, 2, 0)]
    soa = empty_soa(rings)
    soa["ground_height"][:] = -173
    soa["obs_count"][:] = 9
    assert soa["ceiling_height"][0] == CEILING_NONE

    pyr = allocate_pyramid(rings)
    build(pyr, soa, rings)
    nodes = level_arrays(pyr, soa, rings, 0, 1)
    assert theorem3_safe(nodes, thresholds)[0]


# --- the other two predicates ------------------------------------------------


def test_all_clear_is_strictly_stronger_than_theorem_three(thresholds):
    """The trap this exists to close. A uniformly steep bank has every cell
    clear on bits 0, 2 and 5 -- the heights differ by little over a small
    block -- so Theorem 3 reports SAFE while every cell fails bit 1. A planner
    reading Theorem 3 as "drivable" drives onto the bank."""
    rings = [FakeRing(0, 4, 0)]
    soa = empty_soa(rings)
    soa["ground_height"][:] = 5
    soa["ceiling_height"][:] = 400
    soa["obs_count"][:] = 9
    soa["traversability"][:] = TRAV_SLOPE       # every cell, slope only

    pyr = allocate_pyramid(rings)
    build(pyr, soa, rings)
    nodes = level_arrays(pyr, soa, rings, 0, 2)

    assert theorem3_safe(nodes, thresholds)[0], "bits 0/2/5 really are clear"
    assert not all_clear(nodes)[0], "but it is not drivable"
    assert certainly_blocked(nodes)[0]
    assert classify(nodes)[0] == QueryLOD.BLOCKED


def test_safe_and_blocked_cannot_both_fire():
    """OR_mask == 0 implies AND_mask == 0, so the two are mutually exclusive
    by construction rather than by the order of the assignments in
    `classify()`. Asserted over random masks so a future edit cannot make the
    ordering load-bearing without noticing."""
    rng = np.random.default_rng(11)
    rings = small_rings()
    pyr = allocate_pyramid(rings)

    for _ in range(50):
        soa = random_soa(rng, rings)
        build(pyr, soa, rings)
        for r in range(len(rings)):
            for level in range(1, len(pyr.sides[r])):
                nodes = level_arrays(pyr, soa, rings, r, level)
                assert not np.any(all_clear(nodes) & certainly_blocked(nodes))


def test_certainly_blocked_needs_a_COMMON_reason():
    """A block where every cell is blocked for a different reason has
    AND_mask == 0 and comes back MIXED, not BLOCKED. That costs a descent and
    saves nothing unsafe, which is the right direction to be wrong in."""
    rings = [FakeRing(0, 2, 0)]
    soa = empty_soa(rings)
    soa["traversability"][:] = [TRAV_SLOPE, TRAV_STEP, TRAV_CLEARANCE,
                                TRAV_CONFIDENCE]

    pyr = allocate_pyramid(rings)
    build(pyr, soa, rings)
    nodes = level_arrays(pyr, soa, rings, 0, 1)

    assert not certainly_blocked(nodes)[0]
    assert not all_clear(nodes)[0]
    assert classify(nodes)[0] == QueryLOD.MIXED


def test_classify_returns_the_frozen_enum():
    """Three states, and the values are `api.QueryLOD`'s -- a consumer reading
    a bare int must get the same answer as one reading the enum."""
    rings = [FakeRing(0, 2, 0)]
    soa = empty_soa(rings)
    soa["ground_height"][:] = 5
    soa["ceiling_height"][:] = 400
    soa["obs_count"][:] = 9
    pyr = allocate_pyramid(rings)
    build(pyr, soa, rings)

    nodes = level_arrays(pyr, soa, rings, 0, 1)
    assert classify(nodes)[0] == QueryLOD.SAFE
    assert set(np.unique(classify(nodes))) <= {int(q) for q in QueryLOD}


# --- the cost ----------------------------------------------------------------


def test_the_pyramid_costs_a_third_of_the_map_per_node_byte():
    """§7.2's N/3, which is the whole argument that a pyramid is affordable.
    Ceiling-halving overshoots it slightly -- edge blocks of one -- and the
    test pins that the overshoot is small rather than assuming it away."""
    rings = derive_ring_layouts(load("5/10/20/40"))
    slots = sum(r.slots for r in rings)
    nodes = pyramid_bytes(rings) / NODE_BYTES

    assert slots / 3 < nodes < slots / 3 * 1.02


def test_the_measured_cost_is_the_one_in_the_budget():
    """§7.2 says 1.24 MB from `745,000 x 5 / 3`. Both halves are wrong: a node
    stores the REDUCTIONS, so ground contributes H_max and H_min and it is 4
    bytes not 2, and the pyramid covers the 910,000 allocated slots rather
    than the 745,000 logical cells. Pinned so the corrected figure and the
    allocator cannot drift apart before it reaches a slide."""
    rings = derive_ring_layouts(load("5/10/20/40"))
    assert NODE_BYTES == 9
    assert pyramid_bytes(rings) / 1e6 == pytest.approx(2.73, abs=0.01)

    alloc = allocate(load("5/10/20/40"), with_pyramid=True)
    line = next(v for k, v in alloc.budget.items() if k.startswith("conservative"))
    assert line == pyramid_bytes(rings)


def test_the_pyramid_is_off_unless_asked_for():
    """It is a stretch item and it moves the preallocated total by 3.11 MB. A
    number already on a slide does not get to change because a default did."""
    off = allocate(load("5/10/20/40"))
    on = allocate(load("5/10/20/40"), with_pyramid=True)

    assert off.pyramid is None
    assert isinstance(on.pyramid, Pyramid)
    assert not any(k.startswith("conservative") for k in off.budget)
    assert on.total_bytes() - off.total_bytes() == pytest.approx(3.11e6, abs=0.01e6)


# --- determinism and the frame loop -----------------------------------------


def test_rebuilding_gives_a_bit_identical_pyramid():
    """Max, min, AND and OR are exactly associative, so unlike §3.4 this needs
    no care -- but there is no float on this path and there must never be one,
    so the property is asserted rather than assumed."""
    rng = np.random.default_rng(4)
    rings = small_rings()
    soa = random_soa(rng, rings)

    outs = []
    for _ in range(2):
        pyr = allocate_pyramid(rings)
        build(pyr, soa, rings)
        outs.append([{k: v.copy() for k, v in lvl.items()}
                     for r in pyr.levels for lvl in r])

    for a, b in zip(outs[0], outs[1], strict=True):
        for name in a:
            assert np.array_equal(a[name], b[name]), name


def test_a_stale_buffer_is_fully_overwritten():
    """`build()` writes every node without clearing first, which is only sound
    if every node is written. A previous frame's values surviving into a
    quieter one is the shape of bug that shows a hazard that is no longer
    there -- or worse, hides one that is."""
    rings = small_rings()
    pyr = allocate_pyramid(rings)

    busy = random_soa(np.random.default_rng(1), rings)
    build(pyr, busy, rings)

    calm = empty_soa(rings)
    calm["obs_count"][:] = 9
    build(pyr, calm, rings)

    fresh = allocate_pyramid(rings)
    build(fresh, calm, rings)
    for r in range(len(rings)):
        for level in range(len(pyr.levels[r])):
            for name in pyr.levels[r][level]:
                assert np.array_equal(pyr.levels[r][level][name],
                                      fresh.levels[r][level][name]), name


def test_build_allocates_nothing_per_frame():
    """The Day-2 invariant, on the newest path. Measured with a profiler, not
    read: the two-pass reduction exists precisely because the natural
    `reshape(h, 2, h, 2)` spelling needs an even side and would pad -- one
    allocation per field per level per ring, every frame."""
    import tracemalloc

    rings = derive_ring_layouts(load("5/10/20/40"))
    alloc = allocate(load("5/10/20/40"), with_pyramid=True, commit_pages=False)
    build(alloc.pyramid, alloc.grid, rings)      # warm

    tracemalloc.start()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    for _ in range(3):
        build(alloc.pyramid, alloc.grid, rings)
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()

    assert peak < 64 * 1024, f"{peak:,} B per frame"
