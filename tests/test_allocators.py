"""SoA allocation and the memory bound. [Shrestha]

The bound is a headline claim, so these are not documentation tests: they are
the demonstration that the claim holds.
"""

from itertools import pairwise

import numpy as np
import pytest
import yaml
from vrgrid.cell import CELL_BYTES, CELL_FIELDS
from vrgrid.gpu.allocators import (
    EMPTY_CELL,
    TRANSIENT_BYTES,
    Allocation,
    allocate,
    annulus_index,
    bytes_allocated,
    derive_ring_layouts,
    measured_bytes,
)
from vrgrid.gpu.kernels import CEILING_NONE
from vrgrid.gpu.visibility import visibility_scratch_bytes
from vrgrid.grid.schedule import load


@pytest.fixture(scope="module")
def thresholds():
    with open("configs/thresholds.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def alloc(thresholds):
    return allocate(load("5/10/20/40"), thresholds)


# --- geometry ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [("5/10/20/40", [160_000, 210_000, 187_500, 187_500]),
     ("5/10/50", [160_000, 210_000, 150_000])],
)
def test_ring_geometry_reproduces_the_published_cell_counts(name, expected):
    """Derived from half-widths and cell sizes, not copied from the config.
    If these drift, every ratio in the report drifts with them."""
    layouts = derive_ring_layouts(load(name))
    assert [r.count for r in layouts] == expected
    assert sum(expected) == load(name).total_cells


def test_rings_tile_the_flat_array_without_gap_or_overlap():
    layouts = derive_ring_layouts(load("5/10/20/40"))
    assert layouts[0].offset == 0
    for a, b in pairwise(layouts):
        assert a.offset + a.slots == b.offset


def test_config_that_disagrees_with_its_own_geometry_is_rejected():
    """The failure this guards: someone edits a half-width in the YAML and
    forgets to recompute `cells`, so the code and the memory table diverge."""
    s = load("5/10/20/40")
    s.rings[2].cells = 999
    with pytest.raises(ValueError, match="geometry gives"):
        derive_ring_layouts(s)


# --- annulus addressing ------------------------------------------------------


def test_annulus_addressing_is_a_bijection():
    """Exhaustive over a small ring: every cell outside the hole gets exactly
    one index in [0, count), and every index is hit exactly once.

    This is the storage-layer analogue of the lattice partition test. A
    collision here silently double-counts returns; a gap silently drops them.
    """
    layouts = derive_ring_layouts(load("5/10/20/40"))
    for layout in layouts:
        # Shrink to a proportionally identical toy ring to keep it exhaustive.
        scale = max(1, layout.side // 40)
        toy = type(layout)(layout.ring, layout.cell_m, layout.side // scale,
                           layout.hole // scale, 0, 0)
        toy.count = toy.side ** 2 - toy.hole ** 2
        ix, iy = np.meshgrid(np.arange(toy.side), np.arange(toy.side), indexing="xy")
        idx = annulus_index(toy, ix.ravel(), iy.ravel())

        in_hole = ((ix >= toy.lo) & (ix < toy.hi) & (iy >= toy.lo) & (iy < toy.hi)).ravel()
        assert np.all(idx[in_hole] == -1), "hole cells must report -1, not index 0"
        outside = idx[~in_hole]
        assert outside.min() == 0
        assert outside.max() == toy.count - 1
        assert len(np.unique(outside)) == toy.count, "collision or gap in the banding"


def test_hole_cells_are_not_index_zero():
    """A -1 misread as an index would dump every far-field return into the
    first cell of the ring, which looks like a plausible map."""
    layout = derive_ring_layouts(load("5/10/20/40"))[1]
    centre = layout.side // 2
    assert annulus_index(layout, centre, centre) == -1


# --- structure of arrays -----------------------------------------------------


def test_one_array_per_field_not_array_of_structs(alloc):
    assert set(alloc.grid) == {name for name, _ in CELL_FIELDS}
    for name, dt in CELL_FIELDS:
        assert alloc.grid[name].dtype == dt
        assert alloc.grid[name].flags["C_CONTIGUOUS"]


def test_ring_view_writes_through_to_the_allocation(alloc):
    v = alloc.view("ground_height", 2)
    r = alloc.ring(2)
    assert v.shape == (r.slots,)
    v[0] = 1234
    assert alloc.grid["ground_height"][r.offset] == 1234
    v[0] = 0


# --- the bound ---------------------------------------------------------------


def test_claimed_bound_equals_measured_bytes(alloc):
    """The budget is what we claim; measured_bytes reads it back off the
    arrays. Equal means the bound is demonstrated, not asserted."""
    assert bytes_allocated(alloc) == measured_bytes(alloc)


def test_logical_cell_count_is_the_headline_figure(alloc):
    """745,000 logical cells is what every ratio in the report counts. It is
    unchanged by toroidal padding, which adds slots, not cells."""
    assert alloc.logical_cells == 745_000
    assert round(alloc.logical_cells * CELL_BYTES / 1e6, 2) == 8.94


def test_toroidal_padding_is_reported_separately(alloc):
    """Storing full squares so the shift stays O(perimeter) costs 165,000
    slots. It belongs on its own budget line, not folded into the grid."""
    assert alloc.allocated_slots == 910_000
    padding = alloc.allocated_slots - alloc.logical_cells
    assert padding == 165_000
    assert round(padding * CELL_BYTES / 1e6, 2) == 1.98
    assert any("padding" in k for k in alloc.budget)


def test_annulus_storage_drops_the_padding(thresholds):
    """The memory-optimal layout is still available, and costs 15 ms a frame
    on the shift -- see gpu/shift.py."""
    a = allocate(load("5/10/20/40"), thresholds, storage="annulus")
    assert a.allocated_slots == a.logical_cells == 745_000


def test_refinement_pool_is_ninety_eight_kilobytes(alloc):
    pool_bytes = sum(a.nbytes for a in alloc.pool.values())
    assert pool_bytes == 512 * 16 * CELL_BYTES == 98_304


def test_transient_layer_shares_the_grid_geometry(alloc):
    n = alloc.allocated_slots
    assert all(a.shape == (n,) for a in alloc.transient.values())
    assert sum(a.nbytes for a in alloc.transient.values()) == n * TRANSIENT_BYTES


def test_transient_layer_can_be_limited_to_the_inner_rings(thresholds):
    """Pedestrian motion is undetectable beyond ~25 m (math §1.4), so a
    transient layer over rings 0-1 is defensible and saves 1.5 MB. Team call,
    not mine -- but the allocator supports either answer."""
    a = allocate(load("5/10/20/40"), thresholds, transient_rings=2)
    n = 160_000 + 250_000   # allocated slots for rings 0-1
    assert all(arr.shape == (n,) for arr in a.transient.values())
    assert a.total_bytes() < allocate(load("5/10/20/40"), thresholds).total_bytes()


def test_ablation_schedule_allocates_the_smaller_bound(thresholds):
    a = allocate(load("5/10/50"), thresholds)
    assert a.logical_cells == 520_000
    assert round(a.logical_cells * CELL_BYTES / 1e6, 2) == 6.24


def test_tracked_object_list_is_capped(alloc):
    """Capped with priority eviction. Degrading by dropping the least relevant
    track is what makes the bound one we can actually hold."""
    assert alloc.tracks.shape == (alloc.max_tracks,)


def test_budget_lines_sum_to_the_total(alloc: Allocation):
    assert sum(alloc.budget.values()) == alloc.total_bytes()


# --- the invariant everything else rests on ---------------------------------


def test_frame_loop_does_not_allocate(alloc):
    """Simulate 100 frames of read-modify-write over the arrays and assert the
    process allocates nothing on the steady-state path.

    Day 2's gate says verify with a profiler rather than by reading the code.
    This is the cheap always-on version of that check.
    """
    import tracemalloc

    def frame():
        for name in ("ground_height", "obs_count"):
            arr = alloc.grid[name]
            arr[:1000] += 1
            arr[:1000] -= 1

    frame()  # warm up: first touch may allocate scratch
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    for _ in range(100):
        frame()
    after = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    assert after - before < 4096, f"frame loop allocated {after - before} bytes"


def test_our_own_allocation_is_resident_not_just_promised(alloc):
    """The baselines are committed page by page so their counters are honest.
    If ours were not, we would be comparing a real 2.56 GB against a promise,
    and the comparison would flatter us on the one number the demo shows.

    It is also a latency claim: pages faulted in at startup are pages not
    faulted in during frame 1, and that spike would land in the p99 the 10 Hz
    figure rests on -- during the demo, not during a benchmark.
    """
    assert alloc.resident_delta > 0.9 * alloc.total_bytes(), (
        f"claimed {alloc.total_bytes():,} B, resident {alloc.resident_delta:,} B")


def test_commit_can_be_declined(thresholds):
    """Committing costs a moment at startup and is not always wanted -- a unit
    test allocating a hundred grids does not need them all faulted in."""
    schedule = load("5/10/20/40")
    lazy = allocate(schedule, thresholds, commit_pages=False)
    assert lazy.total_bytes() == allocate(schedule, thresholds).total_bytes()
    assert lazy.resident_delta < 0.5 * lazy.total_bytes()


# --- the state a fresh cell is in --------------------------------------------


def test_a_fresh_grid_reports_nothing_overhead(alloc):
    """`np.zeros` is the right empty value for nine of the ten fields and the
    wrong one for `ceiling_height`: 0 cm reads as solid ground at the datum.

    Left zeroed, `ceiling - ground < h_vehicle` holds for every cell in the
    map, TRAV_CLEARANCE marks the whole world untraversable, and it stays that
    way forever because `fuse()` only ever lowers a ceiling. The symptom is a
    planner that refuses to move on a map that renders perfectly.
    """
    assert np.all(alloc.grid["ceiling_height"] == CEILING_NONE)


def test_the_refinement_pool_starts_in_the_same_state(alloc):
    """A block handed out by the pool is a set of brand-new cells, so it needs
    the empty-cell state as much as the grid does. Missing it here is the
    nastier half: the map is fine until semantics force a refinement, and then
    exactly the cells we chose to look at more closely go untraversable."""
    assert np.all(alloc.pool["ceiling_height"] == CEILING_NONE)


def test_every_other_field_really_is_empty_at_zero(alloc):
    """The counterpart claim, asserted rather than assumed -- obs_count 0,
    log_odds 0 (§10.1 decides unknown by observation count, not by log-odds
    near zero) and variance code 0, which the codec maps to MAXIMUM variance
    so a fresh cell claims no certainty it has not earned."""
    for name, _ in CELL_FIELDS:
        if name in EMPTY_CELL:
            continue
        assert not np.any(alloc.grid[name]), name


# --- the §10.4 cleanup scratch, as a declared line item ----------------------

def test_visibility_scratch_is_off_by_default():
    """Same rule as the pyramid: switching it on moves the preallocated total,
    and a number already on a slide does not get to move because a default
    did. The cap is a team decision; the mechanism is not."""
    a = allocate(load("5/10/20/40"))
    assert a.visibility is None
    assert not any("visibility" in k for k in a.budget)


def test_visibility_scratch_is_declared_when_switched_on():
    a = allocate(load("5/10/20/40"), with_visibility=True)
    assert a.visibility is not None
    line = [k for k in a.budget if k.startswith("visibility scratch")]
    assert len(line) == 1
    assert a.budget[line[0]] == sum(x.nbytes for x in a.visibility.values())


@pytest.mark.parametrize("cap", [50_000, 150_000, 200_000])
def test_the_cap_sizes_the_scratch_and_the_budget_together(cap):
    """The declared line and the array that is actually handed to the frame
    loop must be the same object's size, or the bound describes a scratch
    nobody uses."""
    th = {"visibility": {"max_candidate_cells": cap}}
    a = allocate(load("5/10/20/40"), th, with_visibility=True)
    assert len(a.visibility["u"]) == cap
    assert a.budget[f"visibility scratch (cap {cap:,})"] == \
        visibility_scratch_bytes(cap, np.float32)


def test_the_gather_buffer_matches_the_range_image_dtype():
    """float32 because that is what `perception/range_image.project()`
    produces. `np.take` does not widen into `out`, so a float64 buffer against
    a float32 image raises rather than converting -- and converting the image
    per frame would copy the whole thing, which is what the scratch exists to
    avoid."""
    a = allocate(load("5/10/20/40"), with_visibility=True)
    assert a.visibility["observed"].dtype == np.float32


@pytest.mark.parametrize("kw", [
    {},
    {"with_pyramid": True},
    {"with_visibility": True},
    {"with_pyramid": True, "with_visibility": True},
])
def test_claimed_bound_equals_measured_bytes_in_every_configuration(kw):
    """The budget is what we claim; `measured_bytes` is what we allocated.

    Parametrised over the optional groups because it was not: `measured_bytes`
    counted only the always-on groups, so the pyramid carried a budget line
    that nothing ever weighed and this equality had never been asserted with
    it switched on.
    """
    a = allocate(load("5/10/20/40"), with_pyramid=kw.get("with_pyramid", False),
                 with_visibility=kw.get("with_visibility", False))
    assert bytes_allocated(a) == measured_bytes(a)
