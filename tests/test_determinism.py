"""Determinism. CI-blocking. [Shrestha]

Fixed-point int32 accumulation in 1 cm units is the whole reason this passes.
Float atomic adds are non-associative: the map differs run to run, and bugs
move when you look at them. Math §3.4.

The Day-1 gate is "same sequence twice -> byte-identical map hash". Until JP's
loader lands, the sequence is synthetic; the property being tested is the same
one, and swapping in real scans changes only where the points come from.
"""

import numpy as np
import pytest
from vrgrid.gpu.kernels import map_hash, scatter_atomic, scatter_sorted

N_CELLS = 8192


def synthetic_scan(seed=0, n=50_000, n_cells=N_CELLS):
    rng = np.random.default_rng(seed)
    idx = rng.integers(-1, n_cells, n).astype(np.int64)
    return {
        "idx": idx,
        "z_cm": rng.integers(-200, 600, n).astype(np.int16),
        "w_q": rng.integers(1, 2000, n).astype(np.int32),
        "refl": rng.integers(0, 256, n).astype(np.uint8),
        "class_id": rng.integers(0, 19, n).astype(np.uint8),
        "is_ground": rng.random(n) < 0.6,
    }


def shuffled(scan, seed):
    """Same points, different arrival order -- what parallel execution does."""
    order = np.random.default_rng(seed).permutation(len(scan["idx"]))
    out = {k: v[order] for k, v in scan.items()}
    # point_id travels with the point, so the class tiebreak is a property of
    # the scan rather than of the order the points happen to be processed in.
    out["point_id"] = order
    return out


@pytest.mark.determinism
@pytest.mark.parametrize("scatter", [scatter_sorted, scatter_atomic])
def test_same_input_twice_gives_identical_map_hash(scatter):
    kwargs = {} if scatter is scatter_sorted else {"n_cells": N_CELLS}
    scan = synthetic_scan(seed=8)
    first = map_hash(scatter(**scan, **kwargs).as_dict())
    second = map_hash(scatter(**scan, **kwargs).as_dict())
    assert first == second


@pytest.mark.determinism
@pytest.mark.parametrize("scatter", [scatter_sorted, scatter_atomic])
def test_point_order_does_not_change_the_map(scatter):
    """The test that actually catches a float atomic sneaking in.

    Integer addition is exactly associative, so any interleaving of the atomic
    adds must produce the identical map. Swap the accumulators to float and
    this test starts failing intermittently -- which is the whole argument for
    fixed point.
    """
    kwargs = {} if scatter is scatter_sorted else {"n_cells": N_CELLS}
    scan = synthetic_scan(seed=11)
    scan["point_id"] = np.arange(len(scan["idx"]))
    baseline = map_hash(scatter(**scan, **kwargs).as_dict())
    for seed in range(5):
        assert map_hash(scatter(**shuffled(scan, seed), **kwargs).as_dict()) == baseline


@pytest.mark.determinism
def test_the_two_scatter_paths_produce_the_same_hash():
    scan = synthetic_scan(seed=3)
    assert (map_hash(scatter_sorted(**scan).as_dict())
            == map_hash(scatter_atomic(**scan, n_cells=N_CELLS).as_dict()))


def test_float_accumulation_would_not_survive_this():
    """Demonstrates the failure fixed point avoids, so the invariant in
    CLAUDE.md is backed by a number rather than an assertion.

    Same 50,000 addends, two orders. Integers agree exactly; float64 does not.
    """
    rng = np.random.default_rng(1)
    vals = rng.uniform(-1e3, 1e3, 50_000)
    order = rng.permutation(len(vals))
    assert float(vals.sum()) != float(vals[order].sum())
    ints = np.rint(vals * 100).astype(np.int64)  # the same values in 1 cm units
    assert int(ints.sum()) == int(ints[order].sum())


def test_map_hash_is_sensitive_to_every_field():
    """A hash that ignored a field would let the determinism gate pass while
    the map differed."""
    agg = scatter_sorted(**synthetic_scan(seed=5))
    base = map_hash(agg.as_dict())
    for field in agg.as_dict():
        d = agg.as_dict()
        d[field] = d[field].copy()
        d[field][0] += 1
        assert map_hash(d) != base, f"{field} does not affect the hash"


def test_map_hash_distinguishes_fields_with_equal_contents():
    """Swapping two same-typed arrays must change the hash -- otherwise the
    field name is not really part of the digest."""
    a = {"ground_height": np.arange(4, dtype=np.int16),
         "ceiling_height": np.zeros(4, dtype=np.int16)}
    b = {"ground_height": np.zeros(4, dtype=np.int16),
         "ceiling_height": np.arange(4, dtype=np.int16)}
    assert map_hash(a) != map_hash(b)


# --- the whole-pipeline gate -------------------------------------------------
#
# These two ran as `@pytest.mark.skip` + `raise NotImplementedError` until
# 4 Sep, on reasons that had both expired: "needs src/perception/loader.py --
# JP" (`loader.py` is 314 lines and every other test in the suite gates on its
# data) and "needs the full frame loop -- Day 2" (`run/engine.py` is 435 lines
# and `tests/test_engine.py` drives it).
#
# A skip is invisible in a green run, so `make test-determinism` reported a
# passing CI-blocking gate while testing only `scatter` -- the stage where the
# integers already made the answer obvious -- and never the fuse, shift,
# datum-rebase and cleanup stages composed, which is where an ordering bug
# would actually come from.


def _engine_sequence(n_frames, speed_ms, seed=0):
    """A synthetic drive: static scene, vehicle moving, so the toroidal shift
    and the datum tracker are both on the path being hashed.

    Deliberately not a still vehicle. A stationary engine never shifts a ring
    window, and the shift is the one stage that mutates cells it was not handed
    -- exactly the kind of thing that has an ordering to get wrong.
    """
    from types import SimpleNamespace

    ri = pytest.importorskip("vrgrid.perception.range_image")
    rng = np.random.default_rng(seed)
    n_g, n_w = 6000, 9000
    r = rng.uniform(4.0, 12.0, n_g)
    a = rng.uniform(-np.pi, np.pi, n_g)
    ground = np.column_stack([r * np.cos(a), r * np.sin(a), np.full(n_g, -1.73)])
    wall = np.column_stack([np.full(n_w, 30.0), rng.uniform(-8.0, 8.0, n_w),
                            rng.uniform(-4.0, 2.0, n_w)])
    pts = np.vstack([ground, wall])
    is_ground = np.zeros(len(pts), bool)
    is_ground[:n_g] = True

    p4 = np.column_stack([pts, np.full(len(pts), 0.5)])
    image, inverse = ri.project(p4)
    for i in range(n_frames):
        ego = np.array([i * speed_ms, 0.0, i * 0.25])
        yield SimpleNamespace(
            index=i, points_sensor=p4,
            points_world=pts + np.array([ego[0], ego[1], 1.73 + ego[2]]),
            pose=np.eye(4)[:3], vehicle_xyz_world=ego,
            semantic=np.zeros(len(pts), np.int8),
            moving=np.zeros(len(pts), bool), ground=is_ground,
            reflectivity8=np.full(len(pts), 100, np.uint8),
            range_image=image, inverse_index=inverse)


def _run_engine(n_frames=12, speed_ms=2.0, seed=0):
    from vrgrid.grid.schedule import load
    from vrgrid.run.engine import MapEngine

    engine = MapEngine(load("5/10/20/40"), max_points=40_000,
                       max_candidates=80_000, ghost_removal=True)
    for frame in _engine_sequence(n_frames, speed_ms, seed):
        engine.step(frame)
    return engine


@pytest.mark.determinism
def test_the_whole_frame_loop_replays_identically():
    """Every stage composed -- shift, datum re-base, bin, scatter, fuse,
    cleanup -- twice, byte-identical.

    This is the gate `make test-determinism` is supposed to be enforcing. The
    scatter-level tests above cannot see a stage that is not scatter.
    """
    a, b = _run_engine(), _run_engine()
    assert map_hash(a.handle.grid) == map_hash(b.handle.grid)
    assert a.z_datum == b.z_datum


@pytest.mark.determinism
def test_the_frame_loop_is_not_hashing_an_empty_map():
    """The negative control the gate needs and did not have.

    Two identical runs of a loop that silently did nothing also hash equal.
    Every earlier version of this gate would have passed on a map where
    binning sent every point to -1 -- which is a real failure mode, and one
    that makes the latency table read BETTER (see `lattice.bin_points`).
    """
    engine = _run_engine()
    assert int((engine.handle.grid["obs_count"] > 0).sum()) > 10_000
    assert map_hash(engine.handle.grid) != map_hash(_run_engine(n_frames=6).handle.grid)


@pytest.mark.determinism
def test_real_sequence_replay_is_identical():
    """The gate as stated: 50 frames of sequence 08, twice, byte-identical.

    Skipped where the sequence is absent, the same way every other data-backed
    test in this suite is -- NOT skipped unconditionally, which is how this one
    spent its life reporting nothing.
    """
    loader = pytest.importorskip("vrgrid.perception.loader")
    if not (loader.verify_sequence_exists("08")
            and loader._velodyne_path("08", 0).exists()):
        pytest.skip("KITTI seq 08 not present -- set VRGRID_DATA_ROOT")

    from vrgrid.grid.schedule import load
    from vrgrid.run.__main__ import iter_pipeline
    from vrgrid.run.engine import MapEngine

    def replay():
        engine = MapEngine(load("5/10/20/40"), ghost_removal=True)
        for frame in iter_pipeline("08", max_frames=50):
            engine.step(frame)
        return map_hash(engine.handle.grid)

    assert replay() == replay()


def test_no_allocation_inside_the_frame_loop():
    """The Day-2 gate, measured rather than read.

    `tests/test_allocators.py` asserts this for a hand-rolled frame; this is
    the real `MapEngine.step`, which is the thing the memory bound is a claim
    about. The first frames are excluded on purpose: `GridMap.bin_scratch`
    and the visibility scratch size themselves on first use, which is what
    "sized at startup" means for a buffer whose extent depends on the first
    sweep.

    ⚑ This measures RETAINED growth, not per-frame churn. A temporary that is
      allocated and freed inside one frame does not move
      `get_traced_memory()[0]` and will not fail here -- churn is what
      `scripts/timing_table.py --alloc` reports, and what
      `lattice.new_bin_scratch` exists to remove. What this catches is the
      other failure: a buffer that grows with frame count, which is the one
      that makes the compile-time bound false.
    """
    import tracemalloc

    engine = MapEngine_for_alloc = _run_engine(n_frames=2)      # warm every scratch
    frames = list(_engine_sequence(6, 2.0))

    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    for frame in frames:
        engine.step(frame)
    after = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()

    grew = after - before
    assert grew < 64 * 1024, (
        f"the frame loop grew the heap by {grew:,} B over {len(frames)} frames; "
        "every buffer it touches is supposed to be sized at startup")
    assert engine is MapEngine_for_alloc
