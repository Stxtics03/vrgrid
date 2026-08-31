"""Toroidal ego-motion shift. [Shrestha]

Day-1 gate: shift by +d then -d restores every cell that never left the map,
bit-exactly.
"""

import numpy as np
import pytest
from vrgrid.cell import CELL_FIELDS
from vrgrid.gpu.allocators import EMPTY_CELL
from vrgrid.gpu.kernels import CEILING_NONE
from vrgrid.gpu.shift import (
    RingBuffer,
    cells_per_shift,
    columns_to_clear,
    flat_slot_into,
    new_slot_scratch,
    rows_to_clear,
    shift,
)

W = 32


@pytest.fixture
def buf():
    return RingBuffer(side=W, offset=0)


def filled_soa(n=W * W):
    """Distinct value per slot, so any corruption is visible."""
    return {"ground_height": (np.arange(n) % 30000).astype(np.int16),
            "obs_count": (np.arange(n) % 251).astype(np.uint8)}


# --- addressing --------------------------------------------------------------


def test_absolute_cell_keeps_its_slot_across_shifts(buf):
    """A cell that stays in view must not move. The whole design rests on it."""
    before = buf.slot(5, 7)
    shift(buf, 3, 2)
    assert buf.slot(5, 7) == before


def test_out_of_view_cells_have_no_slot(buf):
    """Their slot belongs to whatever is in view there now; writing to it
    would corrupt a live cell, so it must not be handed out."""
    assert buf.slot(-1, 0) == -1
    assert buf.slot(W, 0) == -1
    shift(buf, 10, 0)
    assert buf.slot(5, 0) == -1          # scrolled off the low edge
    assert buf.slot(W + 5, 0) >= 0       # scrolled into view


def test_every_visible_cell_has_a_unique_slot(buf):
    ix, iy = np.meshgrid(np.arange(W), np.arange(W), indexing="xy")
    slots = buf.slot(ix.ravel(), iy.ravel())
    assert len(np.unique(slots)) == W * W
    assert slots.min() == 0 and slots.max() == W * W - 1


def test_flat_slot_applies_the_ring_offset():
    b = RingBuffer(side=W, offset=1000)
    assert b.flat_slot(0, 0) == 1000
    assert b.flat_slot(-1, 0) == -1      # -1 must survive the offset


# --- the gate ----------------------------------------------------------------


@pytest.mark.parametrize("d", [1, 3, 7])
def test_round_trip_is_bit_exact_for_cells_that_never_left(buf, d):
    """+d then -d. Cells that left the map are legitimately gone -- they were
    cleared on the way out, and re-entering they are unknown, not remembered.
    Everything else must be untouched, bit for bit."""
    soa = filled_soa()
    before = {k: v.copy() for k, v in soa.items()}

    shift(buf, d, d, soa)
    shift(buf, -d, -d, soa)

    ix, iy = np.meshgrid(np.arange(d, W - d), np.arange(d, W - d), indexing="xy")
    interior = buf.slot(ix.ravel(), iy.ravel())
    for name, arr in soa.items():
        np.testing.assert_array_equal(arr[interior], before[name][interior], err_msg=name)


def test_newly_visible_cells_are_cleared_not_stale(buf):
    """The failure this prevents: a cell scrolls out, its slot is reused, and
    the map reports year-old evidence for ground it has never seen."""
    soa = filled_soa()
    cleared = shift(buf, 4, 0, soa)
    assert np.all(soa["ground_height"][cleared] == 0)
    assert np.all(soa["obs_count"][cleared] == 0)

    new_cells = buf.slot(np.arange(W, W + 4), np.zeros(4, int))
    assert np.all(soa["obs_count"][new_cells] == 0)


def test_fill_values_override_zero(buf):
    """Some fields have a non-zero 'unknown'. A ceiling of 0 cm would be a
    ceiling at ground level."""
    soa = {"ceiling_height": np.full(W * W, 123, np.int16)}
    cleared = shift(buf, 2, 0, soa, fill={"ceiling_height": 32767})
    assert np.all(soa["ceiling_height"][cleared] == 32767)


# --- the O(perimeter) claim --------------------------------------------------


def test_shift_touches_a_perimeter_not_an_area(buf):
    """Ring 3 clears about 1,000 cells per shift, not 250,000."""
    ring3 = RingBuffer(side=500, offset=0)
    assert cells_per_shift(ring3, 1, 0) == 500
    assert cells_per_shift(ring3, 1, 1) == 999
    assert cells_per_shift(ring3, 1, 1) < 0.005 * ring3.slots


def test_cleared_count_matches_the_prediction(buf):
    for dx, dy in ((1, 0), (0, 1), (3, 2), (-4, 5), (0, 0)):
        b = RingBuffer(side=W, offset=0)
        assert len(shift(b, dx, dy)) == cells_per_shift(b, dx, dy)


def test_a_shift_past_the_whole_window_clears_everything(buf):
    """Teleporting the map is O(area) and there is no way around it -- but it
    must be correct, not merely fast."""
    assert len(shift(buf, W + 1, 0)) == W * W
    assert cells_per_shift(RingBuffer(side=W, offset=0), W, 0) == W * W


def test_zero_shift_clears_nothing(buf):
    soa = filled_soa()
    before = soa["ground_height"].copy()
    assert len(shift(buf, 0, 0, soa)) == 0
    np.testing.assert_array_equal(soa["ground_height"], before)


def test_columns_and_rows_are_disjoint_from_the_retained_interior(buf):
    """A clear that ate an interior cell would silently erase live map."""
    cols = columns_to_clear(buf, 3)
    rows = rows_to_clear(buf, 2)
    ix, iy = np.meshgrid(np.arange(3, W), np.arange(2, W), indexing="xy")
    interior = buf.slot(ix.ravel(), iy.ravel())
    assert not set(cols.tolist()) & set(interior.tolist())
    assert not set(rows.tolist()) & set(interior.tolist())


def test_negative_shift_clears_the_other_edge(buf):
    soa = filled_soa()
    cleared = shift(buf, -3, 0, soa)
    assert len(cleared) == 3 * W
    assert np.all(soa["ground_height"][cleared] == 0)


def test_a_newly_visible_strip_has_no_ceiling(buf):
    """The strip a shift exposes is unobserved ground, and unobserved means
    CEILING_NONE, not a ceiling at the datum.

    Zeroing it instead makes `ceiling - ground < h_vehicle` true across the
    whole strip, so TRAV_CLEARANCE marks it untraversable and nothing ever
    raises it back up -- `fuse()` only lowers a ceiling. A map that booted
    correct would rot along its leading edge as the vehicle drove, which is
    the version of this bug that survives a static test.
    """
    soa = {"ceiling_height": np.full(W * W, 123, np.int16),
           "obs_count": np.full(W * W, 7, np.uint8)}
    cleared = shift(buf, 3, 0, soa)

    assert np.all(soa["ceiling_height"][cleared] == CEILING_NONE)
    assert np.all(soa["obs_count"][cleared] == 0)   # zero really is empty for this one

    retained = np.setdiff1d(np.arange(W * W), cleared)
    assert np.all(soa["ceiling_height"][retained] == 123), "the interior was touched"


def test_the_strip_clear_agrees_with_what_allocate_starts_the_map_in(buf):
    """One definition, two callers. Getting this right in `allocate()` alone
    yields a map that is correct exactly until the vehicle moves, so the test
    is that the two use the same dict rather than that each looks sensible."""
    soa = {name: np.full(W * W, 99, dtype=dt) for name, dt in CELL_FIELDS}
    cleared = shift(buf, 1, 1, soa)
    for name, _ in CELL_FIELDS:
        expected = EMPTY_CELL.get(name, 0)
        assert np.all(soa[name][cleared] == expected), name


def test_a_literal_zero_clear_is_still_available(buf):
    """`fill={}` is not the same as `fill=None`. The transient layer and the
    scratch buffers want raw zeros, and an empty dict has to keep meaning
    that rather than falling through to the cell default."""
    soa = {"ceiling_height": np.full(W * W, 123, np.int16)}
    cleared = shift(buf, 2, 0, soa, fill={})
    assert np.all(soa["ceiling_height"][cleared] == 0)


# --- flat_slot_into: the frame-path spelling of flat_slot ---------------------

@pytest.mark.parametrize("side,offset", [(400, 0), (500, 160_000), (500, 660_000),
                                         (32, 7), (33, 11)])
@pytest.mark.parametrize("dx,dy", [(0, 0), (37, -13), (1234, 998), (-900, 401)])
def test_flat_slot_into_matches_flat_slot(side, offset, dx, dy):
    """The optimisation is only worth having if it is bit-identical.

    `flat_slot_into` replaces `np.mod` with a subtract under a mask, which is
    valid only because an in-view point's window offset is already in [0, W).
    An odd side (33) is in the parametrisation on purpose: the identity has
    nothing to do with powers of two, and a version that quietly assumed one
    would pass on 400 and 500 and fail here.
    """
    rng = np.random.default_rng(abs(side + offset + dx + dy))
    buf = RingBuffer(side=side, offset=offset, x0=-(side // 2) + dx,
                     y0=-(side // 2) + dy)
    # Sampled around the WINDOW, not around the lattice origin: with a small
    # side and a large shift the window has driven clear of the origin, and a
    # sample centred there would be entirely out of view -- which agrees
    # trivially and tests nothing.
    n = 4096
    ix = rng.integers(buf.x0 - side, buf.x0 + 2 * side, n).astype(np.int64)
    iy = rng.integers(buf.y0 - side, buf.y0 + 2 * side, n).astype(np.int64)

    scratch = new_slot_scratch(n)
    got = flat_slot_into(buf, ix, iy, np.zeros(n, np.int64), scratch)
    assert np.array_equal(got, buf.flat_slot(ix, iy))

    # The parametrisation has to actually exercise both branches, or this
    # asserts that two functions agree about nothing.
    assert (got >= 0).any() and (got < 0).any()


def test_flat_slot_into_allocates_nothing_per_call():
    """The whole point: this runs on the frame path, and the frame path does
    not allocate. Anything above a few KB means a temporary crept back in."""
    import tracemalloc

    buf = RingBuffer(side=500, offset=0, x0=-250, y0=-250)
    n = 60_000
    rng = np.random.default_rng(0)
    ix = rng.integers(-600, 600, n).astype(np.int64)
    iy = rng.integers(-600, 600, n).astype(np.int64)
    scratch, out = new_slot_scratch(n), np.zeros(n, np.int64)

    flat_slot_into(buf, ix, iy, out, scratch)        # warm up
    tracemalloc.start()
    try:
        peak = 0
        for _ in range(5):
            tracemalloc.reset_peak()
            before = tracemalloc.get_traced_memory()[0]
            flat_slot_into(buf, ix, iy, out, scratch)
            peak = max(peak, tracemalloc.get_traced_memory()[1] - before)
    finally:
        tracemalloc.stop()
    assert peak < 64_000, f"{peak:,} B per call -- a temporary is back"


def test_flat_slot_into_accepts_a_scratch_larger_than_the_batch():
    """Sized once at the point cap, used every frame at whatever the sweep
    actually returned."""
    buf = RingBuffer(side=64, offset=0, x0=-32, y0=-32)
    scratch = new_slot_scratch(10_000)
    ix = np.array([0, 5, -31, 40], np.int64)
    iy = np.array([0, -5, 31, -40], np.int64)
    got = flat_slot_into(buf, ix, iy, np.zeros(4, np.int64), scratch)
    assert np.array_equal(got, buf.flat_slot(ix, iy))
