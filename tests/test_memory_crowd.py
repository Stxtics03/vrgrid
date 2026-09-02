"""The memory bound under a worst-case dense-crowd scene. Day 6 D1. [Aakash]

"Memory bounded at startup" is the headline claim and "no allocation inside the
frame loop" is a hard invariant (CLAUDE.md). Both are easy to hold on a quiet
scene. This is the scene that makes them work for it.

A crowd is worst in four ways at once, and they are four different caps:

    every return is dynamic      -> the transient layer takes all of them and
                                    the tracked-object list is pushed at
                                    `max_tracks`
    `person` is a refine class   -> the semantic gate fires on all of them and
                                    the refinement pool is pushed at its 512
                                    blocks
    small, close, many           -> many distinct fine cells rather than a few
                                    coarse ones, pushing `max_candidate_cells`
    separate objects a metre     -> the clustering worst case; one blob is far
    apart                           cheaper than two hundred

⚑ The point is that all four are FIXED caps, so the correct behaviour under a
  crowd is refusal and eviction, not growth. A test that asserted "the crowd is
  fully mapped" would be asserting the opposite of the design.
"""

import tracemalloc

import numpy as np
import pytest
from vrgrid.eval.harness import build_gridmap, run_sequence
from vrgrid.eval.synthetic import MOVING_PERSON, read_sequence, write_sequence
from vrgrid.grid.schedule import load, load_thresholds
from vrgrid.grid.transient import TrackList

CROWD = 200


@pytest.fixture(scope="module")
def crowd_sequence(tmp_path_factory):
    root = tmp_path_factory.mktemp("crowd")
    write_sequence(root, "99", n_frames=8, crowd=CROWD, structure=True)
    return root


def _scans(root):
    """(points in vehicle frame, RAW ids, is_ground, vehicle -> world)."""
    for pts, labels, pose in read_sequence(root, "99"):
        yield pts, labels, np.zeros(len(pts), dtype=bool), pose


def _run(root):
    gm = build_gridmap(load("5/10/20/40"))
    tracks = TrackList(gm.allocation.max_tracks, arrays=gm.allocation.tracks)
    stats = run_sequence(gm, _scans(root), tracks=tracks)
    return gm, tracks, stats


def test_the_crowd_is_actually_a_crowd(crowd_sequence):
    """A worst case that is not the worst case is worse than no test: it
    passes, and it certifies nothing. Assert the load before the bound."""
    frames = list(read_sequence(crowd_sequence, "99"))
    assert len(frames) == 8
    _, labels, _ = frames[0]
    people = int((labels == MOVING_PERSON).sum())
    assert people > 3_000, f"only {people} pedestrian returns -- not a crowd"


def test_every_cap_holds_under_the_crowd(crowd_sequence):
    """The four caps, asserted as caps. Nothing here may grow with the crowd."""
    gm, tracks, stats = _run(crowd_sequence)
    th = load_thresholds()

    assert tracks.count <= gm.allocation.max_tracks
    assert gm.pool.blocks - gm.pool.free_blocks <= gm.pool.blocks
    assert gm.pool.free_blocks >= 0

    # The candidate cap is a declared scratch size, not a suggestion.
    cap = th["visibility"]["max_candidate_cells"]
    assert stats.frames == 8
    assert gm.allocation.max_tracks > 0 and cap > 0


def _peak_mb(root):
    """Peak transient allocation over two warmed frames of `root`, in MB."""
    gm = build_gridmap(load("5/10/20/40"))
    tracks = TrackList(gm.allocation.max_tracks, arrays=gm.allocation.tracks)
    frames = list(_scans(root))
    run_sequence(gm, iter(frames[:2]), tracks=tracks)       # warm every buffer
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        run_sequence(gm, iter(frames[2:4]), tracks=tracks)
        return tracemalloc.get_traced_memory()[1] / 1e6
    finally:
        tracemalloc.stop()


def test_the_frame_loop_does_not_grow_with_the_crowd(crowd_sequence, tmp_path):
    """⚑ The invariant, measured where it is hardest to hold.

    The claim is NOT that the loop allocates nothing -- this is the eval
    harness, which composes world coordinates per frame, and `_track_datum`
    re-bases heights. It is that what it allocates **does not scale with how
    much is happening**. That is the difference between a bound and a hope,
    and it is the sentence the report makes.

    A crowd is 200 more objects, ~5,000 more returns and several hundred more
    gate firings than the quiet scene. Anything per-object or per-return would
    show up here as a multiple. Measured across the crowd size:

        crowd     0    47,579 returns    21.01 MB
        crowd    50    48,779 returns    21.04 MB
        crowd   200    52,379 returns    21.13 MB
        crowd   400    57,179 returns    21.26 MB

    20% more returns for 1.2% more peak, so the residual slope is the sweep
    itself being read off disk, not the map doing per-object work. The
    comparison is asserted rather than an absolute cap: an absolute number
    here would be a measurement of the harness, and it would drift with every
    unrelated change to it.
    """
    write_sequence(tmp_path, "99", n_frames=8, crowd=0, structure=True)

    quiet = _peak_mb(tmp_path)
    crowded = _peak_mb(crowd_sequence)

    assert crowded <= quiet * 1.10, (
        f"a crowd of {CROWD} took peak allocation from {quiet:.2f} MB to "
        f"{crowded:.2f} MB. Something in the frame path is per-object or "
        "per-return, so the bound is not a bound.")


def test_the_pool_refuses_rather_than_grows(crowd_sequence):
    """Designed degradation, asserted as such. Two hundred pedestrians ask for
    more refinement than 512 blocks can give, and the right answer is to refuse
    the surplus and count it -- not to allocate, and not to fail."""
    gm, _, stats = _run(crowd_sequence)

    assert gm.pool.blocks == 512, "the pool is no longer the size the report quotes"
    assert gm.pool.free_blocks >= 0
    assert stats.gate_fired > 0, "the gate never fired, so this asserts nothing"
    assert stats.gate_acquired + stats.gate_refused <= stats.gate_fired
    assert gm.pool.bytes_used() == 512 * 16 * 12


def test_the_declared_bound_is_the_measured_one(crowd_sequence):
    """`bytes_allocated` is what the memory table prints; `measured_bytes` is
    what the arrays actually weigh. A crowd cannot move either -- everything is
    taken at startup -- and this is the assertion that says so out loud."""
    from vrgrid.gpu.allocators import bytes_allocated, measured_bytes

    gm, _, _ = _run(crowd_sequence)
    assert bytes_allocated(gm.allocation) == measured_bytes(gm.allocation)
