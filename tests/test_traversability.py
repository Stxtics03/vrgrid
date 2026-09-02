"""The traversability bitfield. Math §7.1. [Aakash]

Each bit is tested in isolation, on a grid where only that condition can
fire. A bitfield whose bits are only ever tested together is a scalar with
extra steps.
"""

import numpy as np
import pytest
from vrgrid.cell import (
    TRAV_CLASS,
    TRAV_CLEARANCE,
    TRAV_CONFIDENCE,
    TRAV_ROUGHNESS,
    TRAV_SLOPE,
    TRAV_STEP,
    alloc_soa,
)
from vrgrid.grid.fusion import CLASS_MAX, initialise, pack_class
from vrgrid.grid.quantise import quantise_variance_cm2
from vrgrid.grid.schedule import load_thresholds
from vrgrid.grid.traversability import (
    baseline_k,
    bitfield,
    border_mask,
    class_ids,
    drivable_ids,
    gradient,
    max_step_cm,
)

SIDE = 8
CELL_M = 0.20


def _flat_grid(ground_cm=0, class_name="road", n=9):
    """A grid on which every condition passes, so any bit that fires in a test
    was fired by that test and not by the background."""
    soa = alloc_soa(SIDE * SIDE)
    initialise(soa)
    soa["ground_height"][:] = ground_cm
    soa["obs_count"][:] = n
    soa["semantic_class"][:] = pack_class(class_ids()[class_name], 5)
    soa["height_variance"][:] = quantise_variance_cm2(1.0)   # sigma = 1 cm
    return soa


def _bits(soa, th=None):
    return bitfield(soa, slice(None), SIDE, CELL_M,
                    th if th is not None else load_thresholds())


def _interior(a):
    """Drop the window border, which carries the confidence bit by design."""
    return np.asarray(a).reshape(SIDE, SIDE)[1:-1, 1:-1].reshape(-1)


def test_flat_well_observed_road_is_traversable():
    """The background case. If this fires a bit, every other test below is
    measuring the wrong thing."""
    assert np.all(_interior(_bits(_flat_grid())) == 0)


def test_clearance_bit():
    soa = _flat_grid()
    soa["ceiling_height"][20] = 150          # 1.5 m over 1.8 m of vehicle
    bits = _bits(soa)
    assert bits[20] & TRAV_CLEARANCE
    assert not bits[21] & TRAV_CLEARANCE


def test_slope_bit_and_the_gradient_it_uses():
    """Central differences, eq. (22), against a slope whose value is known
    exactly: 30% over 20 cm cells is 6 cm per cell."""
    soa = _flat_grid()
    z = (np.arange(SIDE) * 6.0)[None, :] * np.ones((SIDE, 1))   # 30% in +x
    soa["ground_height"][:] = z.reshape(-1).astype(np.int16)

    dzdx, dzdy = gradient(soa["ground_height"], SIDE, CELL_M)
    assert np.allclose(_interior(dzdx), 0.30)
    assert np.allclose(_interior(dzdy), 0.0)

    # theta_max is 20 deg, tan = 0.364, so 30% passes and 45% does not
    th = load_thresholds()
    assert np.tan(np.radians(th["traversability"]["theta_max_deg"])) > 0.30
    assert not np.any(_interior(_bits(soa)) & TRAV_SLOPE)

    soa["ground_height"][:] = (z * 1.5).reshape(-1).astype(np.int16)
    assert np.all(_interior(_bits(soa)) & TRAV_SLOPE)


def test_step_bit_uses_the_maximum_not_the_mean():
    """A cell with three flat neighbours and one 20 cm kerb is a kerb.
    Averaging it away is how a step disappears from a map that still looks
    correct -- so this asserts the max, on exactly that arrangement."""
    soa = _flat_grid()
    soa["ground_height"][SIDE * 4 + 4] = 20          # one 20 cm neighbour
    bits = _bits(soa)

    th = load_thresholds()
    assert th["traversability"]["s_max_m"] == 0.15   # the step is over it

    steps = max_step_cm(soa["ground_height"], SIDE)
    assert steps[SIDE * 4 + 3] == 20                 # neighbour sees the full step
    assert steps[SIDE * 4 + 3] != 5                  # not the mean of {20,0,0,0}
    assert bits[SIDE * 4 + 3] & TRAV_STEP


def test_roughness_bit_reads_through_the_variance_codec():
    """sigma2_max is 0.0025 m^2, i.e. (5 cm)^2. The stored value is a log code
    in cm^2, so this also asserts the two units meet correctly -- the kind of
    seam where a factor of 10^4 hides for a week."""
    soa = _flat_grid()
    soa["height_variance"][30] = quantise_variance_cm2(100.0)   # sigma = 10 cm
    bits = _bits(soa)
    assert bits[30] & TRAV_ROUGHNESS
    assert not bits[31] & TRAV_ROUGHNESS


def test_class_bit_filters_but_does_not_decide():
    """⚑ Geometry decides, semantics filters. A `road` cell with a 40 cm
    pothole is not drivable and a `vegetation` verge often is, so this asserts
    the two are independent bits rather than one overriding the other."""
    pothole_on_road = _flat_grid(class_name="road")
    pothole_on_road["ground_height"][SIDE * 4 + 4] = -40
    bits = _bits(pothole_on_road)
    assert bits[SIDE * 4 + 3] & TRAV_STEP
    assert not bits[SIDE * 4 + 3] & TRAV_CLASS, "class cleared a geometric hazard"

    verge = _flat_grid(class_name="vegetation")
    bits = _bits(verge)
    assert np.all(_interior(bits) & TRAV_CLASS)
    assert not np.any(_interior(bits) & (TRAV_SLOPE | TRAV_STEP))


def test_unobserved_fails_safe_twice_over():
    """A zeroed cell is unobserved AND unlabelled: n = 0 trips confidence, and
    class byte 0 is `unlabeled`, which is not in the drivable set. Both bits
    firing is not redundancy -- it is the two independent reasons a fresh cell
    must not be driven into."""
    soa = _flat_grid()
    soa["obs_count"][25] = 0
    soa["semantic_class"][25] = 0
    bits = _bits(soa)
    assert bits[25] & TRAV_CONFIDENCE
    assert bits[25] & TRAV_CLASS


def test_the_window_border_is_marked_rather_than_fabricated():
    """A central difference on the window edge wraps onto the far side of the
    map -- a cell on the north edge would take its gradient against ground
    100 m south. The border carries the confidence bit instead of a
    fabricated slope: fail safe is already the rule for "not enough
    evidence", and an invented gradient at the map edge is exactly the kind of
    plausible number that survives review."""
    bits = _bits(_flat_grid())
    assert np.all(bits[border_mask(SIDE)] & TRAV_CONFIDENCE)
    assert not np.any(_interior(bits) & TRAV_CONFIDENCE)


def test_drivable_set_comes_from_config_by_name():
    th = load_thresholds()
    ids = drivable_ids(th)
    assert set(ids) == {class_ids()[n] for n in th["traversability"]["drivable_classes"]}

    bad = {"traversability": dict(th["traversability"], drivable_classes=["tarmac"])}
    with pytest.raises(ValueError, match="names no class"):
        drivable_ids(bad)


def test_the_class_table_is_the_one_the_labels_use():
    """⚑ The bug this test exists for, and it was live for five days.

    There were three copies of the 19-class learning order: `configs/frnet.yaml`,
    `perception.semantics.FRNET_CLASS_NAMES`, and a hand-written `CLASS_IDS`
    dict in `grid/traversability.py`. The third was off by one for every class
    -- it began `unlabeled: 0, car: 1, ...` where the real map begins `car: 0`
    and puts `unlabeled` at 19.

    So `drivable_classes: [road, parking, sidewalk, other-ground, terrain]`
    resolved to the ids of {parking, sidewalk, other-ground, **building**,
    **pole**}. The road was not drivable and a building wall was. Bit 4 costs
    `w_class` rather than blocking, so nothing crashed and no path failed --
    the whole road surface just quietly carried a penalty.

    It stayed hidden because the synthetic scene wrote learning ids 9/10/11
    directly, which fall inside the WRONG table's drivable set by coincidence.
    Two errors that cancelled. Correcting that scene to raw ids on 1 Sep put
    `road` = 8 into the map, made this one reachable, and moved plan regret
    from 0.000 to 2.389 -- which was briefly and wrongly attributed to the
    pothole fix landing in the same commit.

    So this asserts against the label producer, not against a literal list.
    """
    from vrgrid.perception.semantics import (
        FRNET_CLASS_NAMES,
        SEMANTIC_KITTI_LABEL_MAP,
        semantic_labels,
    )

    ids = class_ids()
    assert [n for n, _ in sorted(ids.items(), key=lambda kv: kv[1])] == \
        list(FRNET_CLASS_NAMES), "the config and semantics.py disagree"

    # The end-to-end statement: a raw `.label` word for a road surface must
    # come out of `semantic_labels` as the id this module calls `road`.
    for raw, name in ((40, "road"), (44, "parking"), (48, "sidewalk"),
                      (50, "building"), (80, "pole"), (81, "traffic-sign")):
        assert SEMANTIC_KITTI_LABEL_MAP[raw] == ids[name]
        assert int(semantic_labels(np.array([raw], np.uint32))[0]) == ids[name]


def test_the_road_is_drivable_and_a_building_is_not():
    """The predicate the off-by-one inverted, stated in words rather than ids.

    Worth a test of its own: reading `drivable_ids() == [8, 9, 10, 11, 16]` and
    checking it is what a reviewer does once. Reading it back as names is what
    catches the next table shift.
    """
    names = [n for n, _ in sorted(class_ids().items(), key=lambda kv: kv[1])]
    drivable = {names[i] for i in drivable_ids(load_thresholds())}
    assert drivable == {"road", "parking", "sidewalk", "other-ground", "terrain"}
    for blocked in ("building", "pole", "car", "person", "vegetation"):
        assert blocked not in drivable


def test_terrain_is_drivable_and_now_fits_the_cell():
    """⚑ The §10.2 class-width conflict, met where it bit, and closed.

    `terrain` is in the drivable set in `configs/thresholds.yaml` and is
    learning id 16. The cell's class candidate was 4 bits and held 0-15, so
    one of the five classes the config calls drivable could not be stored in
    the map at all -- not an edge case in a class nobody uses, a class the
    §7.1 predicate consults on every cell.

    The byte was re-split 5 | 3 on 1 Sep and it fits now. This asserts the
    resolved state and keeps the boundary visible: `terrain` is still above
    what four bits would hold, so a revert would fail here rather than
    silently mark drivable verges blocked.
    """
    th = load_thresholds()
    assert "terrain" in th["traversability"]["drivable_classes"]
    assert class_ids()["terrain"] == 16
    assert class_ids()["terrain"] > 15, "a 4-bit candidate would lose this"
    assert class_ids()["terrain"] <= CLASS_MAX, "the 5-bit candidate must hold it"

    assert set(drivable_ids(th)) <= set(range(CLASS_MAX + 1))


def test_geometry_is_not_fabricated_against_unobserved_neighbours():
    """⚑ An unobserved cell holds ground_height 0 -- a default, not a
    measurement at the datum. Differencing against it invents obstacles.

    On a 30 cm rise, an observed cell beside an unobserved one reads as a
    30 cm step, sets bit 2 and becomes IMPASSABLE. At ring 0's 11.6%
    single-frame fill rate (§1.3) that is most of the map, so the far field
    comes out walled off by cells nobody ever looked at -- and it looks like
    terrain rather than like a bug. Found by plan regret: the map under test
    blocked 15% of a planning window where the reference blocked 4.5%, and no
    path existed at all.

    The cell is still untraversable either way. The difference is whether it
    says "there is a step here" or "I have not looked", and only one of those
    is true -- and only one lets a planner tell an obstacle from a hole in
    the data.
    """
    soa = _flat_grid(ground_cm=30)          # a patch of ground 30 cm up
    soa["obs_count"][:] = 9
    hole = SIDE * 4 + 4
    soa["obs_count"][hole] = 0              # one neighbour never observed
    soa["ground_height"][hole] = 0          # ... so it still holds the default

    bits = _bits(soa)
    neighbour = SIDE * 4 + 3
    assert not bits[neighbour] & TRAV_STEP, "invented a 30 cm step out of a hole"
    assert not bits[neighbour] & TRAV_SLOPE
    assert bits[hole] & TRAV_CONFIDENCE, "the hole must still fail safe"

    # and a real step, between two observed cells, still fires
    soa["obs_count"][hole] = 9
    assert _bits(soa)[neighbour] & TRAV_STEP


# --- the fixed physical baseline, §7.1 eq. (22) ------------------------------
#
# Eq. (22) differenced over ONE cell measures height change per metre at the
# cell scale, so a step discontinuity reads steeper the finer the lattice.
# Against one frozen tan(theta_max) that makes the same physical feature a wall
# on the fine rings and flat ground on the coarse ones -- which is how the two
# sides of eq. (23) ended up on different geometry. These tests pin the
# invariant that fixes it: one physical feature, one verdict, every lattice.

KERB_M = 0.12          # §4.1's worked example, and the synthetic scene's kerb
POTHOLE_M = 0.40       # the synthetic scene's pothole, the real hazard
LATTICES = [0.05, 0.10, 0.20, 0.25, 0.40, 0.80]


def _step_grid(side, cell_m, height_m, ground_cm=0):
    """A grid split by one straight step of `height_m` down its middle."""
    soa = alloc_soa(side * side)
    initialise(soa)
    z = np.full((side, side), float(ground_cm))
    z[:, side // 2:] += height_m * 100.0
    soa["ground_height"][:] = z.reshape(-1).astype(np.int16)
    soa["obs_count"][:] = 9
    soa["semantic_class"][:] = pack_class(class_ids()["road"], 5)
    soa["height_variance"][:] = quantise_variance_cm2(1.0)
    return soa


def _peak_slope(soa, side, cell_m, baseline_m):
    dzdx, dzdy = gradient(soa["ground_height"], side, cell_m, baseline_m)
    interior = np.hypot(dzdx, dzdy).reshape(side, side)[1:-1, 1:-1]
    return float(interior.max())


def _side_for(cell_m, extent_m=8.0):
    """A window of fixed PHYSICAL extent, so the stencil always fits."""
    return max(8, round(extent_m / cell_m))


def test_baseline_k_falls_back_to_one_cell_when_the_ring_is_coarser():
    """A ring whose cells already exceed the baseline cannot resolve it, and
    says so by differencing over one cell rather than inventing a sub-cell
    sample. That is also what makes this change a no-op at 40 cm and 80 cm."""
    assert baseline_k(0.05, None) == 1, "no baseline configured is the old form"
    assert baseline_k(0.05, 0.50) == 5
    assert baseline_k(0.25, 0.50) == 1
    assert baseline_k(0.40, 0.50) == 1
    assert baseline_k(0.80, 0.50) == 1


@pytest.mark.parametrize("cell_m", LATTICES)
def test_the_stencil_spans_the_baseline_to_within_one_cell(cell_m):
    """k is an integer, so the span it buys is the baseline rounded to the
    lattice -- not the baseline exactly. The bound is what the invariant below
    actually rests on, so it is asserted rather than assumed."""
    th = load_thresholds()["traversability"]
    baseline_m = th["baseline_m"]
    span_m = 2.0 * baseline_k(cell_m, baseline_m) * cell_m
    if 2.0 * cell_m <= baseline_m:
        assert abs(span_m - baseline_m) <= cell_m
    else:
        assert span_m == 2.0 * cell_m, "coarser than the baseline: one cell"


def test_the_kerb_gets_one_slope_verdict_across_every_lattice():
    """THE invariant. A 12 cm kerb is 13.5 deg over the 0.50 m baseline, under
    theta_max, so it is passable -- and it must be passable at 5 cm too, or a
    fine schedule is charged regret for RESOLVING a feature the 25 cm reference
    cannot see."""
    th = load_thresholds()
    baseline_m = th["traversability"]["baseline_m"]
    tan_max = np.tan(np.radians(th["traversability"]["theta_max_deg"]))

    verdicts = {}
    for cell_m in LATTICES:
        side = _side_for(cell_m)
        soa = _step_grid(side, cell_m, KERB_M)
        verdicts[cell_m] = _peak_slope(soa, side, cell_m, baseline_m) > tan_max

    assert set(verdicts.values()) == {False}, (
        f"the kerb is a wall on some lattices and not others: {verdicts}")


def test_without_the_baseline_the_kerb_verdict_flips_with_cell_size():
    """The regression this guards. Documented as a test so that removing
    `baseline_m` from the config fails here loudly rather than quietly putting
    eq. (23) back onto two lattices."""
    tan_max = np.tan(np.radians(
        load_thresholds()["traversability"]["theta_max_deg"]))

    one_cell = {}
    for cell_m in LATTICES:
        side = _side_for(cell_m)
        soa = _step_grid(side, cell_m, KERB_M)
        one_cell[cell_m] = _peak_slope(soa, side, cell_m, None) > tan_max

    assert one_cell[0.05] and one_cell[0.10], "the fine rings called it a wall"
    assert not one_cell[0.25] and not one_cell[0.80], "the coarse ones did not"


def test_the_pothole_rim_still_fails_wherever_the_lattice_can_resolve_it():
    """The baseline must not buy scale-invariance by blinding the predicate to
    a real hazard. 40 cm over the 0.50 m baseline is 38.7 deg, still over
    theta_max -- which is why `baseline_m` is bounded above by
    0.40/tan(theta_max) = 1.10 m and not chosen freely.

    The bound is on the SPAN, not on the baseline, and a ring coarser than the
    baseline falls back to one cell and spans 2c. At 80 cm that is 1.6 m, past
    the bound, and the hazard stops firing -- a real limit of the coarse rings
    and not a regression here. `uniform_80cm` already carried 0 impassable
    cells in the survey for the same reason: a 60 cm hole does not survive an
    80 cm cell. Asserted rather than skipped so the boundary is on the record.
    """
    th = load_thresholds()
    baseline_m = th["traversability"]["baseline_m"]
    tan_max = np.tan(np.radians(th["traversability"]["theta_max_deg"]))

    fires = {}
    for cell_m in LATTICES:
        side = _side_for(cell_m)
        soa = _step_grid(side, cell_m, POTHOLE_M)
        fires[cell_m] = _peak_slope(soa, side, cell_m, baseline_m) > tan_max

    for cell_m in (0.05, 0.10, 0.20, 0.25, 0.40):
        assert fires[cell_m], (
            f"the {POTHOLE_M} m hazard stopped firing at {cell_m} m cells")
    assert not fires[0.80], (
        "80 cm cells span 1.6 m, past the 1.10 m bound -- if this starts "
        "passing, the span bound moved and the kerb invariant needs rechecking")


def test_the_step_bit_reads_over_the_baseline_not_the_cell():
    """Bit 2 scales the OTHER way from bit 1: on a constant grade the step per
    neighbour grows with the cell, so a coarse map calls a ramp a kerb. Read
    over a fixed baseline, one grade gives one step."""
    th = load_thresholds()
    baseline_m = th["traversability"]["baseline_m"]
    grade = 0.25                                     # 25%, well under theta_max

    steps_m, one_cell_m = {}, {}
    for cell_m in (0.05, 0.10, 0.25):
        side = _side_for(cell_m)
        soa = alloc_soa(side * side)
        initialise(soa)
        ramp = (np.arange(side) * cell_m * grade * 100.0)[None, :]
        soa["ground_height"][:] = (ramp * np.ones((side, 1))).reshape(-1).astype(np.int16)
        soa["obs_count"][:] = 9
        interior = slice(1, -1)
        steps_m[cell_m] = float(np.median(
            max_step_cm(soa["ground_height"], side, cell_m, baseline_m)
            .reshape(side, side)[interior, interior])) / 100.0
        one_cell_m[cell_m] = float(np.median(
            max_step_cm(soa["ground_height"], side)
            .reshape(side, side)[interior, interior])) / 100.0

    assert max(steps_m.values()) - min(steps_m.values()) <= 0.03, steps_m
    assert one_cell_m[0.25] > 4 * one_cell_m[0.05], (
        f"one-cell steps must scale with the cell: {one_cell_m}")
