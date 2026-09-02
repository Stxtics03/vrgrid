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

⚑ This ran against `tests/kitti_layout.py`, a second sequence writer that
  existed because `eval/synthetic.py` wrote its poses to the file `loader.py`
  is built to ignore. It also wrote vehicle-frame points into sensor-frame
  `.bin`s, learning ids into raw `.label`s, and vehicle->world rows into
  Camera-0 poses -- so the gap was four conventions, not two paths. Fixing
  them there made the second writer redundant and it is gone: one scene
  generator, and the only frame convention in the repo is JP's. The scene
  here is the analytic one from `docs/sih-math.md` §9.1 with `structure=True`,
  which is a beam-model sweep rather than uniform samples, so the sampling
  density is the one the ring schedule was derived from.
"""

import warnings

# No tests/__init__.py in this repo, so pytest puts the test directory on
# sys.path and a sibling module is a plain import.
import numpy as np
import pytest

pytest.importorskip("vrgrid.perception.loader")

from vrgrid.eval import synthetic as fixture
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
    fixture.write_sequence(tmp_path, SEQ, n_frames=5, structure=True)
    monkeypatch.setattr(loader, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(loader, "GT_POSES_DIR", tmp_path / "poses")
    monkeypatch.setattr(loader, "VELODYNE_DIR", tmp_path / "sequences")
    monkeypatch.setattr(loader, "LABELS_DIR", tmp_path / "sequences")
    transforms._TR_CACHE.pop(SEQ, None)
    yield tmp_path
    transforms._TR_CACHE.pop(SEQ, None)


def test_the_real_loader_reads_the_fixture(sequence):
    """The loader reads what the writer wrote, through no adapter.

    `poses/<seq>.txt` and `sequences/<seq>/calib.txt` are the two files that
    used to be in the wrong place or absent, and they are the two the loader
    opens before it opens a single scan."""
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


def test_the_ground_lands_on_the_surface_it_was_generated_from(sequence):
    """The vehicle origin sits on the road, so road returns must come back at
    the analytic terrain height -- not 1.73 m under it, which is what dropping
    `T_S_V` anywhere in the chain gives. The map would still look entirely
    plausible if this were wrong, which is why it is asserted here.

    Against `terrain_height_m` rather than against zero: the synthetic road is
    crowned (§9.1), so z = 0 is only true on the centreline. A tolerance loose
    enough to absorb the crown would be loose enough to absorb a good part of
    the 1.73 m this exists to catch.
    """
    points, labels, pose = next(iter(loader.scans(SEQ, max_frames=1)))
    world = transforms.transform_points(
        points[:, :3], transforms.sensor_to_world(pose, sequence=SEQ))
    road = world[(np.asarray(labels) & 0xFFFF) == fixture.ROAD]
    assert len(road) > 100
    surface = fixture.terrain_height_m(road[:, 0], road[:, 1], 0)
    assert np.abs(road[:, 2] - surface).max() < 1e-5


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

    # ⚑ max_points must exceed the sweep. The fixture is ~48k returns and
    #   this said 40_000, which silently truncated the tail -- and the tail is
    #   where `structure` appends the facade, the pole and the sign, so the
    #   whole reason this sequence has classes above 15 was being cut off
    #   before it reached a cell. `binned == points` below is what caught it.
    engine = MapEngine(load("5/10/20/40"), max_points=60_000,
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


def test_a_realistic_label_set_fuses_without_clipping(sequence):
    """⚑ The blocker, cleared, on the real path rather than in the abstract.

    A street scene contains poles and buildings, whose learning ids are above
    15. With a 4-bit candidate `fusion.boyer_moore_update` raised on every
    real frame, and `clip_class_ids=True` existed to get past it at the cost
    of corrupting the class layer.

    This test used to assert the raise. Its docstring said that ratifying the
    5/3 split should make it fail and that this would be the signal to drop
    the clip. That happened on 1 Sep, so it now asserts the other side: the
    frame fuses, nothing is clipped, and the ids above 15 survive into cells.
    """
    from vrgrid.grid.fusion import unpack_class
    from vrgrid.run.__main__ import iter_pipeline

    engine = MapEngine(load("5/10/20/40"), max_points=60_000, max_candidates=80_000)
    assert not engine.clip_class_ids, "the clip must be off by default now"

    frame = next(iter(iter_pipeline(SEQ, 1, use_patchworkpp=False)))
    semantic = np.asarray(frame.semantic)
    assert semantic.max() > 15, "the fixture stopped being realistic"

    engine.step(frame)                                   # no raise

    slots = engine.occupied_slots()
    assert slots.size, "the frame fused nothing, so this asserts nothing"
    cand, _ = unpack_class(engine.handle.grid["semantic_class"][slots])
    assert int(cand.max()) > 15, (
        "no cell came back with a class above 15, so either the fixture has no "
        "such returns in occupied cells or the byte is being truncated again"
    )


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
    engine = MapEngine(load("5/10/20/40"), max_points=60_000,
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

    The scene is generated in the vehicle frame, so a car placed at a fixed
    offset ahead cancels the vehicle's own motion exactly and sits at the same
    world x for the whole sequence: labelled moving, never moves. Ghost
    removal then has nothing to remove, and `ghost_removal_figure.py --seq`
    reported clearing 1.0% of the trail -- correct behaviour, measured against
    a fixture that lied. Fixing the car's world speed took the same number to
    69.1%.

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


# --- which pose file a sequence is read with ---------------------------------

def test_pose_source_defaults_per_sequence():
    """08 reads SemanticKITTI's SLAM poses; everything else the official GT.

    Not a preference. Measured as the median absolute ground-height
    disagreement between consecutive frames in 20 cm cells both frames saw:
    seq 07 is 0.49 cm on GT and 0.66 on SLAM, so GT wins; seq 08 is 16.63 cm on
    GT and 1.04 on SLAM. 08's GT poses put the same patch of road 16.6 cm apart
    frame to frame, consistently, which accumulates into M* itself and put its
    per-ring RMSE at 162 cm.
    """
    from vrgrid.perception import loader

    assert loader.pose_source("08") == "slam"
    for seq in ("00", "01", "07", "09", "10"):
        assert loader.pose_source(seq) == "gt", seq


def test_the_pose_source_can_be_forced(monkeypatch):
    """So the comparison above stays reproducible, and so a reviewer can ask
    'what does 08 look like on GT poses' without editing the source."""
    from vrgrid.perception import loader

    monkeypatch.setenv("VRGRID_POSE_SOURCE", "gt")
    assert loader.pose_source("08") == "gt"
    monkeypatch.setenv("VRGRID_POSE_SOURCE", "slam")
    assert loader.pose_source("07") == "slam"

    monkeypatch.setenv("VRGRID_POSE_SOURCE", "sideways")
    with pytest.raises(ValueError, match="gt.*slam"):
        loader.pose_source("07")
