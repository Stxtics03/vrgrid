"""Transient layer and tracked objects. Master v4 §3.6-3.7. [Aakash]"""

import numpy as np
import pytest
from vrgrid.cell import FLAG_DYNAMIC, OCC_OCCUPIED
from vrgrid.eval.harness import build_gridmap, run_sequence
from vrgrid.eval.metrics import height_rmse_per_ring
from vrgrid.eval.reference_map import build_from_scans
from vrgrid.eval.synthetic import MOVING_CAR, read_sequence, write_sequence
from vrgrid.grid.query import query, slot_of
from vrgrid.grid.schedule import load
from vrgrid.grid.transient import (
    TRACK_TTL_FRAMES,
    TrackList,
    clear_grid,
    cluster,
    ingest,
    is_moving,
    separate,
)


@pytest.fixture
def gm():
    return build_gridmap(load("5/10/20/40"))


# --- §3.6: where the motion labels come from ---------------------------------


def test_separate_reads_raw_moving_ids():
    """Master v4 §3.6: `moving-*` is 250-259 in the RAW .label files. Nothing
    is retrained, and the mapping contribution is evaluated independently of
    segmentation quality -- which is a feature, and only true if the ids are
    read rather than inferred."""
    labels = np.array([9, 11, MOVING_CAR, 250, 259, 249, 260])
    static, moving = separate(labels)
    assert moving.tolist() == [False, False, True, True, True, False, False]
    assert (static == ~moving).all()
    assert is_moving(MOVING_CAR)


def test_the_learning_map_destroys_the_information_separate_needs():
    """⚑ Ordering, and it is the kind that produces a plausible wrong map.

    The 19-class collapse maps every `moving-*` id onto its static
    counterpart, so a scan already through the learning map cannot be
    separated at all -- every car that ever drove past ends up welded into the
    elevation map, and nothing raises.
    """
    raw = np.array([MOVING_CAR] * 5)
    assert separate(raw)[1].all()

    collapsed = raw % 16                    # what a learning map does to it
    assert not separate(collapsed)[1].any(), (
        "moving points survived a learning-map collapse -- if this ever passes, "
        "the pipeline is separating on the wrong array"
    )


# --- §3.7: what persists and what does not -----------------------------------


def test_clear_wipes_the_grid_and_not_the_tracks(gm):
    """Master v4 §3.7, and the sentence is easy to read past: "do not wipe the
    transient layer's memory, only its grid."

    The grid is frame-fresh -- where a thing WAS is not where it is. The track
    list is the memory that lets a pedestrian briefly hidden by a parked car
    survive, which master v4 calls the failure mode that matters most.
    """
    tracks = TrackList(8)
    tracks.update(np.array([[5.0, 0.0, 1.0, 6]]))
    gm.transient["flags"][:10] = FLAG_DYNAMIC
    gm.transient["ground_height"][:10] = 170

    clear_grid(gm.transient)
    assert not gm.transient["flags"].any()
    assert not gm.transient["ground_height"].any()
    assert tracks.count == 1, "clearing the grid took the memory with it"


def test_ingest_keeps_the_top_of_the_obstacle_not_its_mean(gm):
    """A pedestrian is 1.7 m of person over 0 m of road, and the mean is a
    metre of neither. A transient cell exists to say "something is standing
    here", so the planner needs the top."""
    pts = np.array([[5.0, 0.0, 0.0], [5.0, 0.0, 0.8], [5.0, 0.0, 1.7]])
    n = ingest(gm, pts, None, np.ones(3, dtype=bool))
    assert n == 3

    _, slot = slot_of(gm, 5.0, 0.0)
    assert gm.transient["ground_height"][slot] == 170
    assert gm.transient["flags"][slot] & FLAG_DYNAMIC


def test_both_layers_measure_height_from_the_same_datum(gm):
    """`query()` returns the persistent and the transient height through ONE
    field, so they have to be on one vertical origin.

    The persistent layer stores `world_z - z_datum` (`engine.step`,
    `harness.run_sequence`). `ingest` quantised the VEHICLE-frame z instead,
    which is `world_z - ego_z` -- the same point, a different origin, and the
    gap is `frac(ego_z)`, up to a metre. It moved every time the band stepped,
    so a pedestrian's reported height drifted against the road they stood on
    while the vehicle climbed.

    Nothing raises when the two disagree; the transient cell simply reports a
    height that is wrong relative to everything around it.
    """
    ego_z, world_z = 3.4, 4.60          # datum floors to 3.0, so frac is 0.40
    gm.z_datum_m = 3.0

    ingest(gm, np.array([[5.0, 0.0, world_z - ego_z]]), None, np.ones(1, bool),
           points_world_m=np.array([[5.0, 0.0, world_z]]))

    _, slot = slot_of(gm, 5.0, 0.0)
    stored_cm = int(gm.transient["ground_height"][slot])
    assert stored_cm == round((world_z - gm.z_datum_m) * 100.0)
    assert query(gm, 5.0, 0.0).ground_height == pytest.approx(world_z - gm.z_datum_m)

    # and the vehicle-frame reading is the wrong answer it used to give
    assert stored_cm != round((world_z - ego_z) * 100.0)


def test_a_stationary_map_is_unchanged_by_the_datum_rule(gm):
    """With no world points and no datum the two are the same array and the
    same origin, so every existing caller -- and every test above -- is
    bit-identical. Stated as an assertion so the compatibility claim in
    `ingest`'s docstring is not just a claim."""
    assert gm.z_datum_m == 0.0
    pts = np.array([[5.0, 0.0, 0.0], [5.0, 0.0, 0.8], [5.0, 0.0, 1.7]])
    ingest(gm, pts, None, np.ones(3, dtype=bool))
    _, slot = slot_of(gm, 5.0, 0.0)
    assert gm.transient["ground_height"][slot] == 170


def test_ingest_does_not_depend_on_point_order(gm):
    """§3.4's argument applied to a different array: an unordered scatter makes
    the map depend on the order points arrived in, and two runs then differ."""
    rng = np.random.default_rng(3)
    pts = np.column_stack([rng.uniform(3, 9, 400), rng.uniform(-3, 3, 400),
                           rng.uniform(0, 2, 400)])
    mask = np.ones(400, dtype=bool)

    a = build_gridmap(load("5/10/20/40"))
    ingest(a, pts, None, mask)
    b = build_gridmap(load("5/10/20/40"))
    order = rng.permutation(400)
    ingest(b, pts[order], None, mask)

    assert np.array_equal(a.transient["ground_height"], b.transient["ground_height"])


def test_query_returns_the_union(gm):
    """The §3.7 merge rule, end to end through the public API."""
    ingest(gm, np.array([[6.0, 1.0, 1.5]]), None, np.ones(1, dtype=bool))
    q = query(gm, 6.0, 1.0)
    assert q.occupancy == OCC_OCCUPIED
    assert q.dynamic is True
    assert q.ground_height == pytest.approx(1.5)


# --- the tracked-object list -------------------------------------------------


def test_tracks_persist_through_a_gap_then_expire():
    """~1 s of constant-velocity prediction, so a pedestrian briefly hidden by
    a parked car does not vanish -- and does not persist forever either."""
    tracks = TrackList(8)
    tracks.update(np.array([[10.0, 0.0, 1.0, 6]]))
    tracks.update(np.array([[11.0, 0.0, 1.0, 6]]))       # moved 1 m in 0.1 s

    assert tracks.alive()["vx_ms"][0] == pytest.approx(10.0)

    for _ in range(TRACK_TTL_FRAMES):                    # occluded
        tracks.predict(0.1)
    assert tracks.count == 1, "the track vanished while it was merely hidden"
    assert tracks.alive()["x_m"][0] > 11.0, "constant velocity did not predict"

    tracks.predict(0.1)
    assert tracks.count == 0, "a track that was never seen again lived forever"


def test_track_list_is_capped_and_evicts_the_stalest():
    """The memory bound is compile-time. A list that grows with traffic makes
    it false, so a full list drops the stalest rather than growing."""
    tracks = TrackList(4)
    for i in range(4):
        tracks.update(np.array([[float(i) * 10, 0.0, 1.0, 6]]))
    assert tracks.count == 4

    for _ in range(3):
        tracks.predict(0.1)
    tracks.update(np.array([[400.0, 0.0, 1.0, 6]]))      # a fifth, far away
    assert tracks.count == 4, "the cap did not hold"
    assert 400.0 in tracks.alive()["x_m"], "the new detection was dropped instead"


def test_cluster_groups_a_vehicle_into_one_object():
    pts = np.column_stack([
        np.random.default_rng(1).uniform(12.0, 13.5, 300),
        np.random.default_rng(2).uniform(-0.9, 0.9, 300),
        np.random.default_rng(3).uniform(0.0, 1.5, 300)])
    out = cluster(pts, cell_m=2.0)
    assert 1 <= len(out) <= 4
    assert out[:, 2].max() == pytest.approx(1.5, abs=0.05), "kept the top"


# --- the measured claim ------------------------------------------------------


def test_the_transient_layer_keeps_the_car_out_of_the_elevation_map(tmp_path):
    """⚑ The number this whole file exists for.

    Before it, one moving car 12 m ahead moved ring 1's height RMSE from
    0.48 cm to 11.71 cm -- the entire error budget of that ring, from one
    object -- and `scripts/eval_synthetic.py` had to strip moving points by
    hand to get an interpretable figure. That hand-holding stops being
    possible the moment a real sequence lands.

    Run the pipeline with the moving points fed straight in, as the real
    loader will feed them, and assert ring 1 is clean anyway.
    """
    write_sequence(tmp_path, "99", n_frames=6)
    reference = build_from_scans(read_sequence(tmp_path, "99"))

    def scans():
        for pts, labels, pose in read_sequence(tmp_path, "99"):
            yield pts, labels, np.ones(len(pts), dtype=bool), pose

    gm = build_gridmap(load("5/10/20/40"))
    stats = run_sequence(gm, scans())

    assert stats.dynamic_points > 0, "the sequence has no moving points to route"
    assert stats.dynamic_to_transient == stats.dynamic_points

    rmse = height_rmse_per_ring(gm, reference)
    assert rmse[1] < 3.0, (
        f"ring 1 RMSE {rmse[1]:.2f} cm -- the car is in the elevation map again"
    )

    # and §9.4 scores both directions, not just removal
    assert stats.removal["DR"] == 1.0
    assert stats.removal["SP"] == 1.0
    assert stats.removal["F"] == 1.0
