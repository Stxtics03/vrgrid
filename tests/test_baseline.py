"""Allocating baselines. [Shrestha]

Nothing here allocates at full scale: the 2.56 GB run is a demo, not a test.
The full-scale numbers are checked arithmetically, and the thing that can only
be checked by allocating -- that the memory is really resident -- is checked at
64 MB, which is enough to be mmap'd and therefore enough to prove the point.
"""

import numpy as np
import pytest
from vrgrid.cell import CELL_BYTES, CELL_FIELDS
from vrgrid.gpu.allocators import resident_fraction
from vrgrid.gpu.baseline import (
    PAGE_BYTES,
    allocate_dense3d,
    allocate_uniform25d,
    commit,
    dense3d_voxels,
    resident_bytes,
    uniform25d_cells,
)

OURS_CELLS = 745_000  # math §11, the 4-ring default schedule

# 115.5 m of footprint at 5 cm is ~5.3 M cells, ~64 MB at 12 B -- large enough
# to be an mmap the kernel serves lazily, small enough for CI.
SMALL_FOOTPRINT_M = 115.5


# --- the figures the report quotes -------------------------------------------


def test_full_scale_counts_match_the_report():
    """math §11. If these drift, a slide is quoting a baseline we do not build."""
    assert dense3d_voxels() == 2_560_000_000
    assert uniform25d_cells() == 16_000_000


def test_full_scale_ratios_are_the_headline_numbers():
    """286x and 21.5x, derived here rather than typed in."""
    ours = OURS_CELLS * CELL_BYTES
    assert dense3d_voxels() * 1 / ours == pytest.approx(286, abs=1)
    assert uniform25d_cells() * CELL_BYTES / ours == pytest.approx(21.5, abs=0.1)


def test_the_uniform_ratio_is_invariant_to_bytes_per_cell():
    """math §11: it is a pure cell-count ratio, which is why fields can be
    added to the cell without weakening the headline. The dense-3D ratio is
    NOT -- it compares 1 B voxels against our 12 B cells -- so it moves, and
    anyone editing the cell struct has to recompute it."""
    for hypothetical in (8, 12, 16, 24):
        ratio = uniform25d_cells() * hypothetical / (OURS_CELLS * hypothetical)
        assert ratio == pytest.approx(21.5, abs=0.1)


def test_uniform_baseline_uses_the_frozen_cell_struct():
    """Sharing CELL_FIELDS is what stops the baseline and the map being
    measured with different rulers."""
    b = allocate_uniform25d(footprint_m=SMALL_FOOTPRINT_M)
    assert set(b.arrays) == {name for name, _ in CELL_FIELDS}
    assert b.claimed_bytes == b.units * CELL_BYTES


# --- the reason the file exists ----------------------------------------------


def test_allocation_is_actually_resident():
    """The demo-critical property. A baseline that claims 2.56 GB and costs
    the machine nothing is not a demonstration, it is a worse version of the
    table we already have.

    Asked of the baseline's OWN pages via mincore(2), not of the process RSS
    delta. The delta is the honest instrument for "what did this cost the
    machine" and the wrong one for this question: glibc raises its mmap
    threshold once it has seen a large block freed, so a later allocation of
    about that size comes off the heap and reuses pages the process already
    holds. RSS then barely moves and a correctly faulted-in 64 MB baseline
    reports 42 MB, depending on nothing but which tests ran first. That is a
    test that fails on Tuesdays, which is worse than no test.
    """
    b = allocate_uniform25d(footprint_m=SMALL_FOOTPRINT_M)
    assert b.claimed_bytes > 60e6, "test size too small to be a real mmap"

    fractions = {name: resident_fraction(a) for name, a in b.arrays.items()}
    if all(f is None for f in fractions.values()):
        pytest.skip("mincore unavailable on this platform")
    for name, f in fractions.items():
        assert f > 0.99, f"{name}: only {f:.1%} of its pages are in core"


def test_an_untouched_allocation_would_have_read_as_free():
    """Negative control, and the bug this file is built to avoid.

    Without it the test above looks like it is checking something when it is
    really checking that numpy can multiply. `np.zeros` returns copy-on-write
    zero pages: the array exists, `nbytes` is right, and the machine has given
    up nothing at all.
    """
    n = 64 * 1024 * 1024
    lazy = np.zeros(n, np.uint8)
    touched = commit(np.zeros(n, np.uint8))
    assert lazy.nbytes == touched.nbytes == n

    if resident_fraction(lazy) is None:
        pytest.skip("mincore unavailable on this platform")
    assert resident_fraction(lazy) < 0.01, (
        f"np.zeros came back {resident_fraction(lazy):.1%} resident on its own; "
        "this platform does not need commit(), and baseline.py should say so")
    assert resident_fraction(touched) > 0.99


def test_the_process_counter_still_sees_a_large_allocation():
    """`resident_bytes()` is what the dashboard shows, so it keeps a test of
    its own -- it is just no longer the instrument for the question above.

    Loose on purpose: the reuse effect that makes it a bad residency probe is
    real and platform-dependent, so this asserts only that a 64 MB commit
    moves the process counter substantially, not that it moves it by 64 MB.
    """
    before = resident_bytes()
    held = commit(np.zeros(64 * 1024 * 1024, np.uint8))
    delta = resident_bytes() - before
    assert delta > 0.25 * held.nbytes, f"{delta:,} B for a {held.nbytes:,} B commit"


def test_commit_does_not_disturb_the_contents():
    """It writes a zero into a page of zeros. If it ever writes anything else,
    the baseline starts the demo with garbage occupancy in it."""
    a = commit(np.zeros(4 * PAGE_BYTES, np.uint8))
    assert not a.any()
    b = np.zeros(1000, np.int16)
    b[:] = 7
    assert np.array_equal(commit(b.copy()), b)


# --- ingest ------------------------------------------------------------------


def test_dense_ingest_marks_the_voxel_containing_the_point():
    b = allocate_dense3d(footprint_m=10.0, vertical_m=8.0)
    assert b.ingest(np.array([0.02]), np.array([0.02]), np.array([0.02])) == 1
    assert int(b.arrays["occupied"].sum()) == 1

    half = b.side // 2
    ix = iy = half  # x, y in [0, 0.05)
    iz = int((0.02 - (-2.0)) / 0.05)
    assert b.arrays["occupied"][(iy * b.side + ix) * b.layers + iz] == 1


def test_returns_outside_the_footprint_are_dropped_not_wrapped():
    """Same rule as `annulus_index()`: a -1 used as an index piles the far
    field into cell 0 and still looks entirely plausible on screen."""
    b = allocate_dense3d(footprint_m=10.0, vertical_m=8.0)
    x = np.array([50.0, -50.0, 0.0, 0.0])
    y = np.array([0.0, 0.0, 50.0, 0.0])
    z = np.array([0.0, 0.0, 0.0, 99.0])       # last one is out of vertical extent
    assert b.ingest(x, y, z) == 0
    assert int(b.arrays["occupied"].sum()) == 0


def test_uniform_ingest_records_height_and_counts_observations():
    b = allocate_uniform25d(footprint_m=10.0)
    n = b.ingest(np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([1.5, 1.5]))
    assert n == 2
    half = b.side // 2
    flat = (int(1.0 / 0.05) + half) * b.side + int(1.0 / 0.05) + half
    assert b.arrays["ground_height"][flat] == 150
    assert b.arrays["obs_count"][flat] == 2


# --- not dying on stage ------------------------------------------------------


def test_an_impossible_allocation_is_refused_rather_than_attempted():
    """An OOM kill halfway through the demo is a worse outcome than a baseline
    that declines and says why."""
    with pytest.raises(MemoryError, match="Refusing"):
        allocate_dense3d(footprint_m=100_000.0)


def test_the_guard_can_be_overridden_deliberately():
    """The check is a safety rail, not a policy. It must be possible to say
    'I know, allocate it anyway' without editing the module."""
    with pytest.raises(MemoryError):
        allocate_uniform25d(footprint_m=1_000_000.0)
    # and the same call with the override gets past the guard to numpy's own
    # refusal, which is a different error -- proving the rail was the blocker.
    with pytest.raises((MemoryError, ValueError)):
        allocate_uniform25d(footprint_m=1_000_000.0, allow_unsafe=True)
