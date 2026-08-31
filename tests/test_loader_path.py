"""The real-data path, exercised before the real data. [Shrestha]

`loader.scans` -> `transforms` -> `range_image` -> `semantics` -> `MapEngine`,
over a sequence written in the layout the loader actually reads. Every other
test of this path skips with "KITTI sequence 00 not present" -- twelve of the
suite's twenty skips say exactly that -- so until now nothing had run it at
all, and the first time it ran would have been Day 5 with the download in.

What this cannot check is KITTI's own bytes: real intensity distributions,
real occlusion, real pose drift. What it does check is that every interface
between the six modules fits, which is the failure that costs a day rather
than an hour.
"""

import warnings

# No tests/__init__.py in this repo, so pytest puts the test directory on
# sys.path and a sibling module is a plain import.
import kitti_layout as fixture
import numpy as np
import pytest

pytest.importorskip("vrgrid.perception.loader")

from vrgrid.grid.schedule import load
from vrgrid.perception import loader, transforms
from vrgrid.run.engine import MapEngine

SEQ = "99"


@pytest.fixture
def sequence(tmp_path, monkeypatch):
    """A written sequence, with the loader pointed at it.

    `loader.DATA_ROOT` is resolved from `$VRGRID_DATA_ROOT` at IMPORT time, so
    setting the environment variable inside a test does nothing once the module
    is loaded. The module attributes have to be patched instead -- a trap worth
    knowing about before someone spends an afternoon on it.
    """
    fixture.write_sequence(tmp_path, SEQ, n_frames=5)
    monkeypatch.setattr(loader, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(loader, "GT_POSES_DIR", tmp_path / "poses")
    monkeypatch.setattr(loader, "VELODYNE_DIR", tmp_path / "sequences")
    monkeypatch.setattr(loader, "LABELS_DIR", tmp_path / "sequences")
    transforms._TR_CACHE.pop(SEQ, None)
    yield tmp_path
    transforms._TR_CACHE.pop(SEQ, None)


def test_the_real_loader_reads_the_fixture(sequence):
    """The point of the fixture. `eval/synthetic.py` writes its poses to
    `sequences/<seq>/poses.txt`, which is the file loader.py's header says it
    deliberately ignores in favour of `poses/<seq>.txt`."""
    frames = list(loader.scans(SEQ, max_frames=5))
    assert len(frames) == 5
    points, labels, pose = frames[0]
    assert points.dtype == np.float32 and points.shape[1] == 4
    assert labels.dtype == np.uint32 and len(labels) == len(points)
    assert np.asarray(pose).shape == (3, 4)


def test_the_pose_puts_the_vehicle_where_it_was_asked_to(sequence):
    """The fixture derives its poses by inverting `vehicle_to_world`, so this
    is the round trip. If it fails, the fixture and `docs/frames.md` disagree
    and every world coordinate below is meaningless."""
    for i, (_, _, pose) in enumerate(loader.scans(SEQ, max_frames=5)):
        t_vw = transforms.vehicle_to_world(pose, sequence=SEQ)
        assert np.allclose(t_vw[:3, 3], [i * 2.0, 0.0, 0.0], atol=1e-9)
        assert np.allclose(t_vw[:3, :3], np.eye(3), atol=1e-9)


def test_the_ground_lands_at_world_zero(sequence):
    """The vehicle origin sits on the road, so road returns must come out at
    world z = 0 -- not at -1.73, which is what dropping `T_S_V` would give.
    The whole sensor/vehicle/world chain is wrong by 1.73 m if this fails, and
    the map would still look entirely plausible."""
    points, labels, pose = next(iter(loader.scans(SEQ, max_frames=1)))
    world = transforms.transform_points(
        points[:, :3], transforms.sensor_to_world(pose, sequence=SEQ))
    road = world[(np.asarray(labels) & 0xFFFF) == fixture.RAW_ROAD]
    assert len(road) > 100
    assert np.allclose(road[:, 2], 0.0, atol=1e-6)


def test_every_return_is_inside_the_sensor_fov(sequence):
    """`range_image.project` warns when returns fall outside [-24.8, +2] deg
    and clamps them to an edge ring. A fixture that trips a real warning
    teaches everyone to ignore it -- the first version of this one clamped
    26.1% of every sweep."""
    ri_mod = pytest.importorskip("vrgrid.perception.range_image")
    points, _, _ = next(iter(loader.scans(SEQ, max_frames=1)))
    _, _, stats = ri_mod.project(points, return_stats=True)
    assert stats["n_clamped_above"] == 0
    assert stats["n_clamped_below"] == 0


def test_the_whole_pipeline_runs_a_sequence(sequence):
    """The end-to-end fit: loader, transforms, range image, semantics, ground,
    reflectivity, then bin/scatter/fuse/cleanup/shift. Warnings are errors,
    because a warning here is an interface disagreeing quietly."""
    from vrgrid.run.__main__ import iter_pipeline

    engine = MapEngine(load("5/10/20/40"), max_points=40_000,
                       max_candidates=80_000, clip_class_ids=True)
    counters = []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for frame in iter_pipeline(SEQ, 5, use_patchworkpp=False):
            counters.append(engine.step(frame))

    assert len(counters) == 5
    assert all(c.binned == c.points for c in counters), "points fell outside every ring"
    assert counters[-1].occupied > 1000
    assert sum(c.cleared for c in counters) > 0, "the cleanup never fired on real frames"


def test_a_realistic_label_set_hits_the_four_bit_class_limit(sequence):
    """⚑ The blocker, on the real path rather than in the abstract.

    A street scene contains poles (raw 80 -> class 17) and buildings
    (50 -> 12). `fusion.boyer_moore_update` packs the class candidate into 4
    bits, so anything above 15 raises. This is every real frame, not an edge
    case, and it is why `clip_class_ids=True` appears in the test above.

    When the room ratifies the 5-bit/3-bit split this test should start
    failing, and that is the signal to drop the clip.
    """
    from vrgrid.run.__main__ import iter_pipeline

    engine = MapEngine(load("5/10/20/40"), max_points=40_000, max_candidates=80_000)
    frame = next(iter(iter_pipeline(SEQ, 1, use_patchworkpp=False)))
    assert np.asarray(frame.semantic).max() > 15, "the fixture stopped being realistic"
    with pytest.raises(ValueError, match="4-bit candidate"):
        engine.step(frame)


def test_one_timer_covers_the_whole_frame(sequence):
    """⚑ `total` must span perception AND the map, or the table lies.

    The perception half runs inside the generator, during `next()`. Timing
    only `engine.step` produced a FRAME row *smaller than several of its own
    stages* and shares that summed to 156% -- a table that looks authoritative
    and is arithmetically impossible. This asserts the containment directly:
    the frame cannot be quicker than its slowest part.
    """
    import time

    from vrgrid.gpu.timing import STAGES, Timer
    from vrgrid.run.__main__ import iter_pipeline

    t = Timer(stages=STAGES)
    engine = MapEngine(load("5/10/20/40"), max_points=40_000,
                       max_candidates=80_000, clip_class_ids=True, timer=t)

    frames = iter(iter_pipeline(SEQ, 5, use_patchworkpp=False, timer=t))
    while True:
        t0 = time.perf_counter()
        frame = next(frames, None)
        if frame is None:
            break
        engine.step(frame)
        t.record("total", (time.perf_counter() - t0) * 1e3)

    summary = t.summary()
    # Both halves of the pipeline reported, under the frozen STAGES spellings.
    for name in ("load", "transform", "range_image", "semantics", "motion",
                 "ground", "reflectivity", "bin", "scatter", "fuse",
                 "cleanup", "shift", "total"):
        assert name in summary, f"{name} recorded no samples"

    # One sample per stage per frame -- a stage timed twice in a frame would
    # mix two different operations into one p50. `ground` and `reflectivity`
    # were folded under `semantics` in the first version of this, which did
    # exactly that.
    n = summary["total"]["n"]
    for name, s in summary.items():
        assert s["n"] == n, f"{name} recorded {s['n']} samples across {n} frames"

    slowest = max(s["max_ms"] for name, s in summary.items() if name != "total")
    assert summary["total"]["max_ms"] >= slowest, (
        "the frame is quicker than its slowest stage, so `total` is not "
        "wrapping the whole frame")


def test_the_moving_car_actually_moves_in_the_world(sequence):
    """⚑ A `moving-car` label on a parked car is worse than no car at all.

    The fixture generates the scene in the SENSOR frame, so a car placed at
    `14 - i * step_m` cancels the vehicle's own motion exactly and sits at
    world x 13-15 m for the whole sequence: labelled moving, never moves.
    Ghost removal then has nothing to remove, and
    `ghost_removal_figure.py --seq` reported clearing 1.0% of the trail --
    correct behaviour, measured against a fixture that lied. Fixing the car's
    world speed took the same number to 69.1%.

    A benchmark that quietly measures nothing is the expensive kind of wrong,
    so this asserts the car is somewhere else every frame.
    """
    from vrgrid.run.__main__ import iter_pipeline

    seen = []
    for frame in iter_pipeline(SEQ, 5, use_patchworkpp=False):
        moving = np.asarray(frame.moving)
        assert moving.any(), f"frame {frame.index} has no moving returns"
        seen.append(float(np.median(frame.points_world[moving, 0])))

    steps = np.diff(seen)
    assert (steps > 1.0).all(), (
        f"the moving car advanced {steps} m/frame in the world -- a label "
        "that says moving on geometry that does not")
