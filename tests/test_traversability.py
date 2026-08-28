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
from vrgrid.grid.fusion import initialise, pack_class
from vrgrid.grid.quantise import quantise_variance_cm2
from vrgrid.grid.schedule import load_thresholds
from vrgrid.grid.traversability import (
    CLASS_IDS,
    bitfield,
    border_mask,
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
    soa["semantic_class"][:] = pack_class(CLASS_IDS[class_name], 5)
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
    assert set(ids) == {CLASS_IDS[n] for n in th["traversability"]["drivable_classes"]}

    bad = {"traversability": dict(th["traversability"], drivable_classes=["tarmac"])}
    with pytest.raises(ValueError, match="names no class"):
        drivable_ids(bad)


def test_terrain_is_drivable_and_does_not_fit_the_cell():
    """⚑ The §10.2 class-width conflict, met where it actually bites.

    `terrain` is in the drivable set in `configs/thresholds.yaml` and is
    learning id 17. The cell's class nibble is 4 bits and holds 0-15. So one
    of the five classes the config calls drivable cannot be stored in the map
    at all -- it is not an edge case in a class nobody uses, it is a class the
    traversability predicate consults on every cell.

    Documents the conflict rather than working around it; the fix (5-bit
    candidate, 3-bit counter) is a whole-team call on a frozen struct.
    """
    th = load_thresholds()
    assert "terrain" in th["traversability"]["drivable_classes"]
    assert CLASS_IDS["terrain"] == 17
    assert CLASS_IDS["terrain"] > 15

    fits = [n for n in th["traversability"]["drivable_classes"] if CLASS_IDS[n] <= 15]
    assert len(fits) == 4, "the class table changed -- recheck the 4-bit conflict"
