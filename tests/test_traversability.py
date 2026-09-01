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
