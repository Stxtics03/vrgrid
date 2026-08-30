"""Toroidal ego-motion shift. [Shrestha]

Day-1 gate: shift by +d then -d restores every cell that never left the map,
bit-exactly.
"""

import numpy as np
import pytest
from vrgrid.gpu.shift import (
    RingBuffer,
    cells_per_shift,
    columns_to_clear,
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
