"""Reference map and the synthetic sequence. Math §9.1. [Aakash]

M* is the thing every other number is measured against, so an error here is
invisible everywhere and fatal everywhere. The two properties that matter most
-- that it strips moving points, and that it is schedule-independent -- are
asserted directly rather than inferred from a metric coming out plausible.
"""

import numpy as np
import pytest
from vrgrid.eval.reference_map import (
    ReferenceMap,
    build_from_scans,
    is_moving,
    load,
)
from vrgrid.eval.synthetic import (
    KERB_HEIGHT_M,
    KERB_Y_M,
    MOVING_CAR,
    POTHOLE_DEPTH_M,
    POTHOLE_XY_M,
    read_sequence,
    scan,
    terrain_height_m,
    write_sequence,
)


@pytest.fixture(scope="module")
def sequence(tmp_path_factory):
    root = tmp_path_factory.mktemp("seq")
    write_sequence(root, "99", n_frames=4)
    return root


@pytest.fixture(scope="module")
def reference(sequence):
    return build_from_scans(read_sequence(sequence, "99"))


# --- the synthetic surface ---------------------------------------------------


def test_the_surface_is_a_surface_not_a_sample():
    """The same place must give the same height on every frame, or the
    reference map is averaging a moving world and means nothing."""
    x = np.array([1.0, 12.0, 40.0])
    y = np.array([0.0, 3.5, -8.0])
    assert np.array_equal(terrain_height_m(x, y), terrain_height_m(x, y))


def test_the_surface_has_the_features_the_metrics_are_meant_to_catch():
    # The step is across the kerb, not from the crown of the road: the road is
    # cambered, so z(centre) - z(edge) is 6 cm of drainage camber on top of the
    # 12 cm kerb. Measuring from the centre would report half a kerb and look
    # like a fusion bug later.
    inside = terrain_height_m(5.0, KERB_Y_M - 0.01)
    outside = terrain_height_m(5.0, KERB_Y_M + 0.01)
    assert outside - inside == pytest.approx(KERB_HEIGHT_M, abs=0.01)
    assert terrain_height_m(5.0, 0.0) - inside == pytest.approx(0.06, abs=0.01)

    pot = terrain_height_m(*POTHOLE_XY_M)
    beside = terrain_height_m(POTHOLE_XY_M[0] + 2.0, POTHOLE_XY_M[1])
    assert beside - pot == pytest.approx(POTHOLE_DEPTH_M, abs=0.02)

    flat = terrain_height_m(20.0, 0.0)
    uphill = terrain_height_m(50.0, 0.0)
    assert uphill > flat + 1.0, "the ramp is not a slope worth testing bit 1 on"


def test_the_scan_has_the_sampling_geometry_the_rings_are_derived_from():
    """§1.2's quadratic radial spacing is the entire justification for the
    ring schedule. If the synthetic scan does not reproduce it, every fill
    rate measured on it is meaningless."""
    pts, _, _ = scan(moving_car=False)
    r = np.hypot(pts[:, 0], pts[:, 1])

    near = np.count_nonzero((r > 5) & (r < 10))
    far = np.count_nonzero((r > 50) & (r < 55))
    assert near > 5 * far, "returns are not thinning out quadratically with range"

    # and the blind cone: nothing lands inside 3.74 m (§1.4 eq. 5)
    assert np.count_nonzero(r < 3.0) == 0


# --- §9.1: what M* is and is not ---------------------------------------------


def test_moving_points_are_stripped():
    """§9.1. Ids 250-259 from the raw `.label` file, nothing retrained."""
    assert is_moving(MOVING_CAR)
    assert is_moving(np.array([250, 255, 259])).all()
    assert not is_moving(np.array([9, 11, 249, 260])).any()

    pts = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 5.0]])
    labels = np.array([9, MOVING_CAR])
    ref = build_from_scans([(pts, labels, np.eye(4))])
    assert ref.observed.sum() == 1
    assert ref.height_cm[ref.observed][0] == pytest.approx(0.0)


def test_reference_is_on_the_base_lattice_and_matches_the_surface(reference):
    """The reference must recover the analytic surface it was sampled from.

    Checked on cells that actually hold a return rather than on a chosen
    coordinate: at 6 m the beams land ~15 cm apart radially (§1.2), so most
    5 cm cells there are empty and picking one by hand tests the sampling
    pattern rather than the rasteriser.
    """
    obs = np.argwhere(reference.observed)
    rng = np.random.default_rng(1)
    for r, c in obs[rng.choice(len(obs), 300, replace=False)]:
        x = (r + reference.i0 + 0.5) * reference.cell_m
        y = (c + reference.j0 + 0.5) * reference.cell_m
        assert reference.height_cm[r, c] / 100.0 == pytest.approx(
            terrain_height_m(x, y), abs=0.06)


def test_a_uniform_5cm_grid_over_this_scene_is_almost_entirely_empty():
    """§1.3's headline claim, reproduced by the synthetic sampler: a uniform
    5 cm raster of what the sensor actually returned is >95% empty. That is
    the argument for the whole representation, and it is worth having the
    scaffold demonstrate it rather than assert it from the doc."""
    pts, _, _ = scan(moving_car=False)
    r = np.hypot(pts[:, 0], pts[:, 1])
    near = pts[r < 25.0]
    cells = {(int(np.floor(px / 0.05)), int(np.floor(py / 0.05)))
             for px, py in near[:, :2]}
    footprint = (2 * 25.0 / 0.05) ** 2
    assert len(cells) / footprint < 0.05


def test_reference_knows_nothing_about_schedules(reference):
    """Schedule-independence is what makes cross-schedule comparison valid. It
    is a structural property, so it is asserted structurally: there is no
    schedule anywhere in the object."""
    assert reference.cell_m == 0.05
    assert not any("ring" in s or "schedule" in s for s in ReferenceMap.__slots__)


def test_block_stats_matches_a_brute_force_reduction(reference):
    """The summed-area tables are an optimisation, and an optimisation of the
    thing every metric is built on. Checked against the obvious slow version
    on random blocks -- if these disagree, every number in the harness is
    wrong in a way no other test would catch."""
    rng = np.random.default_rng(5)
    H, W = reference.shape
    for k in (1, 2, 4, 8, 10):
        i_lo = rng.integers(reference.i0, reference.i0 + H - k, 40)
        j_lo = rng.integers(reference.j0, reference.j0 + W - k, 40)
        n, mean, var = reference.block_stats(i_lo, j_lo, k)

        for idx in range(len(i_lo)):
            r0 = i_lo[idx] - reference.i0
            c0 = j_lo[idx] - reference.j0
            block_obs = reference.observed[r0:r0 + k, c0:c0 + k]
            block_h = reference.height_cm[r0:r0 + k, c0:c0 + k][block_obs]

            assert n[idx] == block_obs.sum()
            if block_h.size:
                assert mean[idx] == pytest.approx(block_h.mean(), abs=1e-6)
                assert var[idx] == pytest.approx(block_h.var(), abs=1e-6)


def test_block_stats_clips_at_the_edge_rather_than_wrapping(reference):
    """A ring extends far past where a short sequence drove, so most blocks
    fall outside the reference entirely. They must come back with n = 0 so the
    caller drops them -- not wrap, and not score as agreement."""
    n, _, _ = reference.block_stats(np.array([reference.i0 - 10_000]),
                                    np.array([reference.j0 - 10_000]), 8)
    assert n[0] == 0


def test_save_and_load_round_trip(reference, tmp_path):
    path = tmp_path / "ref.npz"
    reference.save(path)
    back = load(path)

    assert back.cell_m == reference.cell_m
    assert (back.i0, back.j0) == (reference.i0, reference.j0)
    assert np.array_equal(back.count, reference.count)
    assert np.allclose(back.height_cm, reference.height_cm, atol=1e-2)

    # and the loaded one answers block queries identically
    i = np.array([reference.i0 + 10, reference.i0 + 50])
    j = np.array([reference.j0 + 10, reference.j0 + 50])
    for a, b in zip(reference.block_stats(i, j, 4), back.block_stats(i, j, 4)):
        assert np.allclose(a, b, atol=1e-2)


def test_an_empty_sequence_is_an_error_not_an_empty_map():
    """A reference map with no static points would silently score every
    schedule as perfect, because every comparison would be dropped."""
    pts = np.array([[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="no static points"):
        build_from_scans([(pts, np.array([MOVING_CAR]), np.eye(4))])


def test_poses_are_applied_so_the_map_is_world_anchored(sequence):
    """Frames matter more here than anywhere else: the whole sequence has to
    land in one world frame, or M* is a smear of every pose at once."""
    ref = build_from_scans(read_sequence(sequence, "99"))
    extent_m = ref.shape[0] * ref.cell_m
    assert extent_m > 100.0, "the map did not grow as the vehicle drove"

    # the surface at a place only the later frames could see
    x = 3 * 2.0 + 20.0                        # 4 frames, 2 m apart
    i = int(np.floor(x / 0.05)) - ref.i0
    j = int(np.floor(0.0 / 0.05)) - ref.j0
    if 0 <= i < ref.shape[0] and 0 <= j < ref.shape[1] and ref.count[i, j]:
        assert ref.height_cm[i, j] / 100.0 == pytest.approx(
            terrain_height_m(x, 0.0), abs=0.05)


def test_one_definition_of_moving():
    """⚑ `is_moving` was written out three times: `perception/semantics.py`,
    `grid/transient.py` and here. `perception/loader.py` states the id range a
    fourth time as a bare constant. JP found them while wiring the front end.

    They agree today, which is the only reason nothing has failed. They are
    also the predicate that decides which returns never enter the persistent
    map (§3.6) AND the predicate the reference map removes by (§9.1) -- so if
    two of them ever drift, the ghost-removal metric is scored against a
    reference that still contains the ghosts, and the number moves in the
    direction that looks like success.

    This file's copy now imports the transient layer's. The perception copies
    are a cross-directory change and stay pinned here rather than edited.
    """
    import numpy as np
    from vrgrid.eval import reference_map
    from vrgrid.grid import transient

    ids = np.arange(0, 400, dtype=np.uint32)
    core = transient.is_moving(ids)

    assert reference_map.is_moving is transient.is_moving, (
        "eval re-declared is_moving instead of importing it"
    )
    assert ids[core].tolist() == list(range(250, 260))

    semantics = pytest.importorskip("vrgrid.perception.semantics")
    assert np.array_equal(semantics.is_moving(ids), core), (
        "perception/semantics.py and grid/transient.py disagree about which "
        "raw label ids are moving -- consolidate before this reaches a metric"
    )

    loader = pytest.importorskip("vrgrid.perception.loader")
    assert list(loader.MOVING_LABEL_IDS) == list(range(250, 260)), (
        "perception/loader.py's MOVING_LABEL_IDS has drifted from the predicate"
    )
