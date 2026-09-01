"""The synthetic sequence's on-disk layout and its sampling geometry. [Aakash]

`eval/synthetic.py` claims two things that nothing used to check, because
`read_sequence` was its only reader and shared every one of its assumptions:

  1. what it writes is the layout `perception.loader` reads, in the
     conventions `docs/frames.md` fixes;
  2. what it writes is an HDL-64E firing on the analytic surface -- §1.2's
     beam model, which is what makes the ring schedule derivable from it.

Both were false. The files were in the wrong places, the points were in the
wrong frame, the labels were in the wrong id space, the poses were in the
wrong convention, and the beam intersection had a sign error that put every
return on the wrong side of every feature the surface has. None of it crashed;
that is the point of this file.
"""

import numpy as np
import pytest
from vrgrid.eval.synthetic import (
    PHI_MAX_DEG,
    PHI_MIN_DEG,
    POTHOLE_DEPTH_M,
    POTHOLE_RADIUS_M,
    POTHOLE_XY_M,
    ROAD,
    SIDEWALK,
    read_sequence,
    scan,
    terrain_height_m,
    write_sequence,
)

SENSOR_H_M = 1.73
SEQ = "99"


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn")
    write_sequence(root, SEQ, n_frames=4, structure=True)
    return root


# --- 1. the layout -----------------------------------------------------------

def test_the_files_land_where_the_loader_looks_for_them(written):
    """`loader.py`'s header says in as many words that it uses the official GT
    poses at `poses/<seq>.txt` and NOT the SemanticKITTI SLAM poses at
    `sequences/<seq>/poses.txt`. This module used to write only the second,
    which is precisely the file the loader is built to ignore."""
    assert (written / "poses" / f"{SEQ}.txt").exists()
    assert (written / "sequences" / SEQ / "calib.txt").exists()
    assert (written / "sequences" / SEQ / "velodyne" / "000000.bin").exists()
    assert (written / "sequences" / SEQ / "labels" / "000000.label").exists()
    assert not (written / "sequences" / SEQ / "poses.txt").exists()


def test_the_bin_is_in_the_sensor_frame(written):
    """A `.bin` holds SENSOR-frame points and the vehicle origin is 1.73 m
    below the laser. Writing vehicle-frame points into one puts every road
    return that far underground, and the map still looks entirely plausible --
    which is why this is asserted on the bytes rather than inferred."""
    raw = np.fromfile(written / "sequences" / SEQ / "velodyne" / "000000.bin",
                      dtype=np.float32).reshape(-1, 4)
    road_ish = np.abs(raw[:, 1]) < 2.0
    assert np.median(raw[road_ish, 2]) == pytest.approx(-SENSOR_H_M, abs=0.05)


def test_the_label_file_holds_raw_semantickitti_ids(written):
    """`.label` words are raw ids, which `semantics.semantic_labels` maps.

    Learning ids written into one are read back as raw, and the collision is
    silent and wrong rather than out of range: this scene used 9/10/11, and as
    raw ids 9 is unmapped, 10 is `car` and 11 is `bicycle`. The road came back
    as ignore and the parking as car.
    """
    from vrgrid.perception.semantics import semantic_labels

    lbl = np.fromfile(written / "sequences" / SEQ / "labels" / "000000.label",
                      dtype=np.uint32) & 0xFFFF
    assert ROAD in lbl and SIDEWALK in lbl
    learning = semantic_labels(lbl)
    assert set(np.unique(learning[lbl == ROAD]).tolist()) == {8}         # road
    assert set(np.unique(learning[lbl == SIDEWALK]).tolist()) == {10}    # sidewalk
    assert learning.max() > 15, "structure=True must reach the 5-bit range (§10.2)"


def test_the_pose_row_is_camera_convention_and_composes_back(written):
    """A `poses/<seq>.txt` row is Camera-0 -> World_cam, so it is NOT the
    vehicle -> world matrix; `transforms.vehicle_to_world` is the composition
    between them. The rows here are derived by inverting that composition, so
    this is the round trip.

    The translation is what tells the two conventions apart, and it is worth
    asserting on rather than the rotation. This sequence drives straight, and
    its `Tr` is the exact inverse axis permutation, so the two rotations
    cancel and the row on disk has an identity R either way. The translation
    does not cancel: 2 m of world forward motion appears on the camera's +z,
    and the vehicle origin sitting 1.73 m below the laser appears as +1.73 on
    the camera's +y (which points down). A vehicle->world row written into
    this file by mistake would read (2i, 0, 0).
    """
    from vrgrid.perception.loader import read_calib
    from vrgrid.perception.transforms import vehicle_to_world

    tr = read_calib(written / "sequences" / SEQ / "calib.txt")["Tr_velo_to_cam0"]
    rows = np.loadtxt(written / "poses" / f"{SEQ}.txt").reshape(-1, 3, 4)

    for i, row in enumerate(rows):
        assert np.allclose(row[:3, 3], [0.0, -1.73, i * 2.0], atol=1e-9), (
            "the pose row is not in the camera convention -- forward motion "
            "must land on camera z, not camera x")
        t_vw = vehicle_to_world(row, tr=tr)
        assert np.allclose(t_vw[:3, :3], np.eye(3), atol=1e-9)
        assert np.allclose(t_vw[:3, 3], [i * 2.0, 0.0, 0.0], atol=1e-9)


def test_the_round_trip_puts_the_ground_back_on_the_surface(written):
    """`read_sequence` returns vehicle-frame points and a vehicle -> world
    transform, so composing them must land on `terrain_height_m` -- the whole
    scene's contract. Tolerance is float32 storage, not a fudge factor."""
    for i, (pts, lbl, T) in enumerate(read_sequence(written, SEQ)):
        world = pts @ T[:3, :3].T + T[:3, 3]
        g = lbl == ROAD
        err = world[g, 2] - terrain_height_m(world[g, 0], world[g, 1], i)
        assert np.abs(err).max() < 1e-5


# --- 2. the sampling geometry ------------------------------------------------

def test_every_return_is_at_the_elevation_of_the_beam_that_fired_it():
    """⚑ The test the sign error would have failed.

    `scan` solves `f(r) = h_s + r tan(phi) - z(r) = 0` for each beam, so every
    return must come back at exactly the elevation it was fired at. The old
    sampler took one correction step using `(h_s + z)` where the sensor's
    height above a surface at z is `(h_s - z)`; on flat ground the two agree,
    which is why it survived, and on the 12 cm kerb and the 40 cm pothole it
    put the return roughly `2z/tan|phi|` out radially -- 1.7 m at the steepest
    beam.

    This also subsumes the FOV property. A return at exactly its beam's
    elevation is inside [-24.8, +2] because that is the interval the beams
    were drawn from, and `range_image.project` has nothing to clamp.
    """
    pts, _, _ = scan(structure=True, seed=0)
    horiz = np.hypot(pts[:, 0], pts[:, 1])
    phi = np.degrees(np.arctan2(pts[:, 2] - SENSOR_H_M, np.maximum(horiz, 1e-9)))
    assert phi.min() >= PHI_MIN_DEG
    assert phi.max() <= PHI_MAX_DEG


def test_ground_returns_lie_exactly_on_the_analytic_surface():
    """Not approximately. M* is built from these scans and the metrics compare
    against `terrain_height_m`, so a return that is merely near the surface is
    an error budget nobody declared."""
    pts, _, ground = scan(structure=True, seed=3)
    g = pts[ground]
    assert np.abs(g[:, 2] - terrain_height_m(g[:, 0], g[:, 1], 3)).max() == 0.0


def test_the_blind_cone_is_where_section_1_2_says_it_is():
    """r_blind = h_s / tan|phi_min| = 1.73 / tan(24.8 deg) = 3.74 m, math eq (5).
    It falls out of the beam model rather than being imposed, so it is the
    cheapest check that the model is the one the schedule was derived from."""
    pts, _, ground = scan(seed=0)
    forward = ground & (pts[:, 0] > 0) & (np.abs(pts[:, 1]) < 0.5)
    assert np.hypot(pts[forward, 0], pts[forward, 1]).min() == pytest.approx(3.74,
                                                                            abs=0.02)


def test_the_pothole_is_resolved_near_and_invisible_far():
    """§1.4's named check, on the scene that is supposed to demonstrate it.

    A negative obstacle of width W is sampled only where W > s_rad(r), giving
    r_max between 10.8 m (50 cm) and 15.2 m (1 m) for this 60 cm pothole.

    The far case is the interesting one and it is asserted the strict way:
    beams DO land inside the pothole's footprint from 14 m and 16 m, and every
    one of them comes back at rim height. The hole is narrower than the radial
    spacing there, so the beam that would fall in it falls across it instead.
    Asserting merely that no point is returned would also pass on a range
    where no beam lands at all, which proves nothing.

    ⚑ The old sampler returned NOTHING at depth from any range, in any frame.
      The scene's only negative obstacle had never once been observed as a
      hole, and the §8.2 money plot was reading R(S) off a lane with no hazard
      in it. That is the expensive kind of wrong: a benchmark quietly
      measuring nothing.
    """
    def counts(ahead_m):
        vx = POTHOLE_XY_M[0] - ahead_m
        pts, _, _ = scan(pose_x_m=vx, seed=5)
        inside = np.hypot(pts[:, 0] + vx - POTHOLE_XY_M[0],
                          pts[:, 1] - POTHOLE_XY_M[1]) < POTHOLE_RADIUS_M
        deep = pts[inside, 2] < -0.5 * POTHOLE_DEPTH_M
        return int(inside.sum()), int(deep.sum())

    inside, deep = counts(8.0)
    assert deep > 0, "the pothole is not resolved even from 8 m"

    for ahead in (14.0, 16.0):
        inside, deep = counts(ahead)
        assert inside > 0, f"no beam lands in the footprint from {ahead} m"
        assert deep == 0, f"the pothole is resolved from {ahead} m, past r_max"


def test_structure_is_off_by_default():
    """Every reference number in `docs/` was measured on the ground scene, so
    the facade and the pole are opt-in. A default that quietly changed them
    would invalidate the ablation table without touching the ablation."""
    plain, cls_plain, ground = scan(seed=0)
    assert ground[cls_plain != 252].all(), "only the moving car is off-ground"
    assert len(scan(structure=True, seed=0)[0]) > len(plain)
