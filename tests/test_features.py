"""Curb and pothole detection. Math §7.4. [Shrestha]

The problem statement names curbs and potholes as the reason a 2D grid is not
enough, so these are tested against the geometry it names: a 12 cm kerb and a
40 cm hole, on grids where nothing else can fire.

The tests that matter most here are the negative ones. A detector that finds
curbs is easy; one that does not invent them out of unobserved cells is the
whole job, because an unobserved cell holds `ground_height` 0 as a DEFAULT and
differencing against it fabricates exactly the feature being looked for.
"""

import numpy as np
from vrgrid.cell import FLAG_BLIND, alloc_soa
from vrgrid.grid.features import detect_curbs, detect_potholes
from vrgrid.grid.fusion import initialise

SIDE = 40
CELL_M = 0.10
KERB_CM = 12.0          # §4.1's worked example
POTHOLE_CM = 40.0       # §1.4's negative obstacle


def _grid(n=9):
    soa = alloc_soa(SIDE * SIDE)
    initialise(soa)
    soa["obs_count"][:] = n
    soa["ground_height"][:] = 0
    soa["flags"][:] = 0
    return soa


def _z(soa):
    return soa["ground_height"].reshape(SIDE, SIDE)


def _mask(hits):
    m = np.zeros(SIDE * SIDE, dtype=bool)
    m[hits.slot] = True
    return m.reshape(SIDE, SIDE)


# --- curbs -------------------------------------------------------------------

def test_a_straight_kerb_is_found_along_its_whole_run():
    """The positive case. A 12 cm step down column 20, found as a LINE."""
    soa = _grid()
    _z(soa)[:, 20:] = KERB_CM
    curbs = detect_curbs(soa, slice(None), SIDE, CELL_M)

    assert len(curbs) > 0, "found no kerb at all"
    found = _mask(curbs)
    # every row should carry a detection, and they should sit at the step
    assert found.any(axis=1).sum() >= SIDE - 4, "the kerb is not a continuous run"
    cols = np.flatnonzero(found.any(axis=0))
    assert cols.min() >= 17 and cols.max() <= 23, f"kerb located at {cols}"
    assert np.allclose(curbs.height_cm, KERB_CM, atol=1.0)


def test_a_kerb_is_reported_as_a_feature_not_as_impassable():
    """§7.1 after eq. (22a) calls a 12 cm kerb PASSABLE -- 13.5 deg over the
    0.50 m baseline. That is correct and it is not the whole answer, which is
    the reason this module exists. Asserted so the two cannot silently drift
    into disagreeing about what a kerb is."""
    from vrgrid.cell import TRAV_SLOPE, TRAV_STEP
    from vrgrid.grid.traversability import bitfield

    soa = _grid()
    _z(soa)[:, 20:] = KERB_CM
    bits = bitfield(soa, slice(None), SIDE, CELL_M).reshape(SIDE, SIDE)
    geometric = bits[1:-1, 1:-1] & (TRAV_SLOPE | TRAV_STEP)

    assert not geometric.any(), "§7.1 called a 12 cm kerb impassable"
    assert len(detect_curbs(soa, slice(None), SIDE, CELL_M)) > 0, (
        "the kerb §7.1 passes must still be reported as a feature")


def test_a_wall_is_not_a_kerb():
    """Above `max_height_m` it is a wall, a vehicle or a fence. The band has an
    upper edge for a reason and this is it."""
    soa = _grid()
    _z(soa)[:, 20:] = 90.0                      # 90 cm, well over the band
    assert len(detect_curbs(soa, slice(None), SIDE, CELL_M)) == 0


def test_road_texture_is_not_a_kerb():
    """Below `min_height_m`. 2 cm of camber and seams must not light up."""
    soa = _grid()
    rng = np.random.default_rng(7)
    _z(soa)[:] = rng.integers(-2, 3, size=(SIDE, SIDE))
    assert len(detect_curbs(soa, slice(None), SIDE, CELL_M)) == 0


def test_a_lone_raised_cell_is_a_rock_not_a_kerb():
    """The run test. One cell at kerb height, with no linear continuation --
    a rock, a bin, a bad return. A detector without this fires on every
    speckle in the map."""
    soa = _grid()
    _z(soa)[20, 20] = KERB_CM
    assert len(detect_curbs(soa, slice(None), SIDE, CELL_M)) == 0


def test_an_unobserved_strip_is_not_a_kerb():
    """THE failure mode. Unobserved cells hold ground_height 0 as a default;
    beside real ground at 12 cm they look exactly like a kerb. They must not
    be reported as one."""
    soa = _grid()
    _z(soa)[:] = KERB_CM
    soa["obs_count"].reshape(SIDE, SIDE)[:, 20:] = 0     # never looked
    assert len(detect_curbs(soa, slice(None), SIDE, CELL_M)) == 0, (
        "invented a kerb out of the edge of the observed region")


# --- potholes ----------------------------------------------------------------

def test_a_pothole_is_found_with_its_depth():
    """The positive case: a 40 cm hole, 60 cm across, on flat road."""
    soa = _grid()
    i, j = np.indices((SIDE, SIDE))
    hole = (i - 20) ** 2 + (j - 20) ** 2 <= 3 ** 2        # ~60 cm at 10 cm cells
    _z(soa)[hole] = -POTHOLE_CM
    holes = detect_potholes(soa, slice(None), SIDE, CELL_M)

    assert len(holes) > 0, "found no pothole"
    found = _mask(holes)
    assert found[hole].any(), "the detection is not on the hole"
    assert not found[~hole].any(), "reported intact road as a pothole"
    assert np.allclose(holes.depth_cm.max(), POTHOLE_CM, atol=2.0)


def test_a_hole_in_the_data_is_not_a_pothole():
    """THE failure mode, and the reason `_observed` exists. An unobserved
    patch reads ground_height 0. Put the road at 50 cm and the unobserved
    patch is a 50 cm depression that was never looked at."""
    soa = _grid()
    _z(soa)[:] = 50
    i, j = np.indices((SIDE, SIDE))
    gap = (i - 20) ** 2 + (j - 20) ** 2 <= 3 ** 2
    soa["obs_count"].reshape(SIDE, SIDE)[gap] = 0
    soa["ground_height"].reshape(SIDE, SIDE)[gap] = 0     # the default, not a measurement

    holes = detect_potholes(soa, slice(None), SIDE, CELL_M)
    assert len(holes) == 0, "reported a hole in the DATA as a hole in the ROAD"


def test_the_blind_cone_is_not_one_enormous_pothole():
    """FLAG_BLIND cells are unknown by construction (master v4 §3.6) and sit in
    a 3.74 m disc centred on the vehicle. Without the blind test they are a
    pothole the size of the car."""
    soa = _grid()
    _z(soa)[:] = 40
    i, j = np.indices((SIDE, SIDE))
    cone = (i - 20) ** 2 + (j - 20) ** 2 <= 10 ** 2
    soa["flags"].reshape(SIDE, SIDE)[cone] = FLAG_BLIND
    soa["ground_height"].reshape(SIDE, SIDE)[cone] = 0

    assert len(detect_potholes(soa, slice(None), SIDE, CELL_M)) == 0


def test_a_downgrade_is_not_a_pothole():
    """Locality. On a constant 15% grade every cell is below the ones behind
    it; without the depressed-rim test the whole slope reports as potholes."""
    soa = _grid()
    _z(soa)[:] = (np.arange(SIDE) * CELL_M * 0.15 * 100.0)[None, :]
    assert len(detect_potholes(soa, slice(None), SIDE, CELL_M)) == 0


def test_a_shallow_patch_is_not_a_pothole():
    """Under `min_depth_m`. A 4 cm settled patch is not a hazard."""
    soa = _grid()
    i, j = np.indices((SIDE, SIDE))
    _z(soa)[(i - 20) ** 2 + (j - 20) ** 2 <= 3 ** 2] = -4
    assert len(detect_potholes(soa, slice(None), SIDE, CELL_M)) == 0


def test_the_two_detectors_do_not_claim_the_same_cells():
    """A kerb is a rise and a pothole is a depression. On a scene with one of
    each they must not overlap -- if they do, one of them is firing on the
    other's geometry and the report would double-count."""
    soa = _grid()
    _z(soa)[:, 28:] = KERB_CM
    i, j = np.indices((SIDE, SIDE))
    _z(soa)[(i - 12) ** 2 + (j - 12) ** 2 <= 3 ** 2] = -POTHOLE_CM

    curbs = _mask(detect_curbs(soa, slice(None), SIDE, CELL_M))
    holes = _mask(detect_potholes(soa, slice(None), SIDE, CELL_M))
    assert curbs.any() and holes.any(), "the scene should contain both"
    assert not (curbs & holes).any(), "a cell was reported as kerb AND pothole"


# --- the ring buffer ---------------------------------------------------------
#
# A ring is a `gpu.shift.RingBuffer`: world lattice index `ix` lives at memory
# index `ix % side`, and the origin moves with the vehicle. So a feature that
# is contiguous in the WORLD is generally split in MEMORY, and a detector that
# walks the array linearly falls off both halves of it.
#
# This is the bug these two tests exist for, and it is worth stating how it
# presented: on the synthetic scene the pothole landed at memory rows 0-4 and
# 395-398 of a 400-row ring. detect_potholes returned zero detections, raised
# nothing, and every unit test above still passed -- because they all build
# plain arrays where memory order and world order agree.

SEAM = 20          # where the window's world edge sits in memory, for these tests
ORIGIN = (SEAM, SEAM)


def _wrapped_hole(soa, depth_cm=POTHOLE_CM):
    """A round depression centred on memory row 0, so it wraps the array."""
    i, j = np.indices((SIDE, SIDE))
    di = np.minimum(i, SIDE - i)                      # toroidal distance to row 0
    hole = (di ** 2 + (j - 30) ** 2) <= 3 ** 2
    _z(soa)[hole] = -depth_cm
    return hole


def test_a_pothole_split_across_the_memory_wrap_is_still_found():
    """World-contiguous, memory-split. The detector must follow the torus."""
    soa = _grid()
    hole = _wrapped_hole(soa)
    holes = detect_potholes(soa, slice(None), SIDE, CELL_M, origin=ORIGIN)

    assert len(holes) > 0, "lost the pothole at the memory wrap"
    found = _mask(holes)
    assert found[hole].any(), "detection is not on the hole"
    assert not found[~hole].any(), "reported intact road as a pothole"
    assert np.allclose(holes.depth_cm.max(), POTHOLE_CM, atol=2.0)


def test_nothing_is_measured_across_the_windows_world_edge():
    """The other half of the same rule, and the reason the seam is not just
    "memory index 0".

    Memory columns either side of the seam are `side` cells apart in the
    WORLD -- 20 m at ring 0 -- so a height difference between them is two
    unrelated places, not a feature. The scene here is a gentle ramp in world
    coordinates: no real kerb anywhere, every world-adjacent pair differing by
    0.5 cm. Laid into a ring buffer it puts a 19.5 cm cliff across the seam,
    which is precisely the artifact a linear scan would report as a kerb.
    """
    ramp = _grid()
    x0 = ORIGIN[0]
    world_j = x0 + ((np.arange(SIDE) - x0) % SIDE)
    _z(ramp)[:] = (world_j * 0.5)[None, :]

    seam_step = abs(_z(ramp)[0, (x0 - 1) % SIDE] - _z(ramp)[0, x0 % SIDE])
    assert seam_step > KERB_CM, "the test scene does not exercise the seam"

    assert len(detect_curbs(ramp, slice(None), SIDE, CELL_M,
                            origin=ORIGIN)) == 0, "differenced across the world edge"

    away = _grid()
    _z(away)[:, SEAM + 8:] = KERB_CM
    assert len(detect_curbs(away, slice(None), SIDE, CELL_M, origin=ORIGIN)) > 0, (
        "a kerb away from the seam must still be found")


def test_a_kerb_adjacent_cell_is_not_a_pothole():
    """The rim-consistency rule. A cell just inside a 12 cm kerb has half its
    rim on the road and half on the sidewalk, so the rim median sits above the
    road and the carriageway reads as a depression. 26 cells of good road came
    out as 10 cm potholes this way before `max_rim_spread_m` existed."""
    soa = _grid()
    _z(soa)[:, 25:] = KERB_CM
    holes = detect_potholes(soa, slice(None), SIDE, CELL_M, origin=ORIGIN)
    assert len(holes) == 0, (
        f"invented {len(holes)} potholes out of the road beside a kerb")
