"""Scatter kernel. [Shrestha]

The determinism tests live in tests/test_determinism.py; these cover the
arithmetic and the equivalence of the two paths.
"""

import numpy as np
import pytest
from vrgrid.gpu.kernels import (
    CEILING_NONE,
    CLASS_RADIX,
    SORTED_SCRATCH_POINT_FIELDS,
    WEIGHT_MAX,
    Z_MAX_CM,
    CellAggregate,
    grid_bytes,
    measurement_variance_cm2,
    new_sorted_scratch,
    quantise_height,
    quantise_weight,
    scatter_atomic,
    scatter_scratch_bytes,
    scatter_sorted,
)

N_CELLS = 4096


def random_scan(rng, n=20_000, n_cells=N_CELLS, hole_fraction=0.05):
    idx = rng.integers(0, n_cells, n).astype(np.int64)
    idx[rng.random(n) < hole_fraction] = -1        # hole / off-map returns
    return {
        "idx": idx,
        "z_cm": rng.integers(-200, 600, n).astype(np.int16),
        "w_q": rng.integers(1, 2000, n).astype(np.int32),
        "refl": rng.integers(0, 256, n).astype(np.uint8),
        "class_id": rng.integers(0, 19, n).astype(np.uint8),
        "is_ground": rng.random(n) < 0.6,
    }


# --- the measurement model ---------------------------------------------------


def test_variance_reproduces_the_published_sigmas():
    """Math §3.2: 8.7 cm at 50 m, 17.5 cm at 100 m."""
    assert np.sqrt(measurement_variance_cm2(50.0)) == pytest.approx(8.7, abs=0.1)
    assert np.sqrt(measurement_variance_cm2(100.0)) == pytest.approx(17.5, abs=0.1)


def test_variance_grows_quadratically_at_range():
    """The r^2 term dominates: doubling range roughly quadruples sigma^2."""
    ratio = measurement_variance_cm2(100.0) / measurement_variance_cm2(50.0)
    assert 3.8 < ratio < 4.2


def test_grazing_incidence_inflates_variance():
    """A lateral error on a near-grazing surface maps into a large height
    error. Clamped at cos = 0.1 so it cannot become a singularity."""
    head_on = measurement_variance_cm2(50.0, cos_incidence=1.0)
    grazing = measurement_variance_cm2(50.0, cos_incidence=0.2)
    assert grazing == pytest.approx(head_on / 0.04)
    assert measurement_variance_cm2(50.0, cos_incidence=0.0) == pytest.approx(head_on / 0.01)


def test_weights_are_integers_and_never_zero():
    """A 100 m return is weak evidence, not absent evidence. A zero weight
    would leave a far-only cell with no height at all."""
    w = quantise_weight(measurement_variance_cm2(np.array([5.0, 50.0, 100.0, 1e6])))
    assert w.dtype == np.int32
    assert np.all(w >= 1)
    assert np.all(w <= WEIGHT_MAX)
    assert w[0] > w[1] > w[2]  # nearer returns weigh more


def test_heights_clamp_to_the_vertical_extent():
    """-2 to +6 m. Overpasses are out of scope, and an unclamped value would
    silently wrap in int16."""
    z = quantise_height(np.array([-9.0, -2.0, 0.0, 6.0, 9.0]))
    assert z.dtype == np.int16
    assert z.tolist() == [-200, -200, 0, 600, 600]


# --- aggregation -------------------------------------------------------------


def test_hand_worked_aggregate():
    """Three returns in cell 7, one in cell 9, one dropped.

    Cell 7's middle return is non-ground, which is what makes this worth
    working by hand: it counts toward `n`, `refl_sum` and the ceiling, and it
    is absent from both height sums.
    """
    agg = scatter_sorted(
        idx=np.array([7, 9, 7, -1, 7]),
        z_cm=np.array([100, 50, 200, 999, 300], np.int16),
        w_q=np.array([2, 5, 3, 7, 5], np.int32),
        refl=np.array([10, 20, 30, 40, 50], np.uint8),
        class_id=np.array([1, 2, 3, 4, 5], np.uint8),
        is_ground=np.array([True, True, False, True, True]),
    )
    assert agg.cells.tolist() == [7, 9]
    assert agg.wz_sum.tolist() == [2 * 100 + 5 * 300, 5 * 50]   # w=3 at 200 cm excluded
    assert agg.w_sum.tolist() == [2 + 5, 5]                     # ground weight only
    assert agg.n.tolist() == [3, 1]                             # all returns
    assert agg.refl_sum.tolist() == [90, 20]                    # all returns
    assert agg.ceiling_cm.tolist() == [200, CEILING_NONE]   # only the non-ground return
    assert agg.class_id.tolist() == [1, 2]                  # lowest-indexed point
    assert agg.mean_height_cm().tolist() == [243, 50]       # 1700/7, rounded away from 0


def test_negative_indices_are_dropped_not_folded_into_cell_zero():
    """annulus_index() returns -1 for the ring's hole. Folding those into
    cell 0 would pile the far field onto one cell and still look plausible."""
    agg = scatter_sorted(
        idx=np.array([-1, -1, -1]), z_cm=np.zeros(3, np.int16),
        w_q=np.ones(3, np.int32), refl=np.zeros(3, np.uint8),
        class_id=np.zeros(3, np.uint8), is_ground=np.ones(3, bool),
    )
    assert len(agg) == 0


def test_ceiling_is_the_lowest_non_ground_return():
    """'Lowest thing overhead' -- a cell with only ground returns has no
    ceiling, and must not report 0 as one."""
    agg = scatter_sorted(
        idx=np.array([3, 3, 3]), z_cm=np.array([500, 250, 10], np.int16),
        w_q=np.ones(3, np.int32), refl=np.zeros(3, np.uint8),
        class_id=np.zeros(3, np.uint8), is_ground=np.array([False, False, True]),
    )
    assert agg.ceiling_cm.tolist() == [250]


def test_weighted_mean_favours_the_near_return():
    """A 5 m return and a 100 m return disagreeing by 1 m: the mean must sit
    near the close one, which is the whole point of the variance model."""
    w = quantise_weight(measurement_variance_cm2(np.array([5.0, 100.0])))
    agg = scatter_sorted(
        idx=np.array([0, 0]), z_cm=np.array([0, 100], np.int16), w_q=w,
        refl=np.zeros(2, np.uint8), class_id=np.zeros(2, np.uint8),
        is_ground=np.ones(2, bool),
    )
    assert agg.mean_height_cm()[0] < 5   # cm, i.e. within 5 cm of the near return


def test_mean_of_empty_scan_does_not_divide_by_zero():
    agg = scatter_sorted(np.array([], np.int64), np.array([], np.int16),
                         np.array([], np.int32), np.array([], np.uint8),
                         np.array([], np.uint8), np.array([], bool))
    assert len(agg) == 0
    assert agg.mean_height_cm().size == 0


def test_class_ids_must_fit_the_packed_key():
    with pytest.raises(ValueError, match="class ids must be"):
        scatter_sorted(np.array([0]), np.array([0], np.int16), np.array([1], np.int32),
                       np.array([0], np.uint8), np.array([CLASS_RADIX], np.uint8),
                       np.array([True]))


def test_mismatched_input_lengths_are_rejected():
    with pytest.raises(ValueError, match="z_cm has"):
        scatter_sorted(np.array([0, 1]), np.array([0], np.int16), np.array([1, 1], np.int32),
                       np.array([0, 0], np.uint8), np.array([0, 0], np.uint8),
                       np.array([True, True]))


# --- the gather contract -----------------------------------------------------
#
# These two exist because the kernel was written against numpy 2.5 and CI runs
# 2.4, where the difference is not a warning but 30 red tests. Before 2.5,
# `np.take`/`np.compress` wrap `out` in the SOURCE dtype, so gathering an int32
# column into an int64 buffer is refused outright: "cannot cast int64 to int32
# under rule 'safe'". Widening has to happen after the gather, not through it.


@pytest.mark.parametrize("column, dtype", [
    ("z_cm", np.int16), ("w_q", np.int32), ("refl", np.int32),
    ("is_ground", np.bool_),
])
def test_gather_buffers_match_the_width_of_the_column_they_gather(column, dtype):
    """Every payload buffer is exactly as wide as the input column named in
    `fusion.scatter()`. A buffer wider than its source is a gather np.take
    cannot perform; narrower, and it truncates silently."""
    widths = dict(SORTED_SCRATCH_POINT_FIELDS)
    assert np.dtype(widths[column]) == np.dtype(dtype)


def test_the_weight_height_product_is_accumulated_in_64_bits():
    """`w_q` gathers at int32 now, so nothing about the buffer widths forces
    w*z to 64 bits any more -- only the explicit `dtype=` on the multiply does.

    `quantise_height()` clamps to Z_MAX_CM and at that clamp the int32 product
    still fits, with the 3x margin the weight scale was chosen for. The kernel
    does not take a clamped column though, it takes an int16 one, and a height
    an order of magnitude past the clamp wraps the product NEGATIVE -- which
    fuses into a map that looks entirely plausible. 64 bits or the clamp has to
    move into the kernel; a 3x margin held by a caller is not a guarantee.
    """
    n = 64
    z_cm = 20_000                                   # 200 m: absurd, and int16
    assert WEIGHT_MAX * z_cm > np.iinfo(np.int32).max
    agg = scatter_sorted(
        idx=np.zeros(n, np.int64),
        z_cm=np.full(n, z_cm, np.int16), w_q=np.full(n, WEIGHT_MAX, np.int32),
        refl=np.zeros(n, np.int32), class_id=np.zeros(n, np.uint8),
        is_ground=np.ones(n, bool),
    )
    assert agg.wz_sum.dtype == np.int64
    assert int(agg.wz_sum[0]) == n * WEIGHT_MAX * z_cm      # exact, in python ints
    assert agg.mean_height_cm()[0] == z_cm

    # The clamp the production caller does observe, for contrast: this one fits
    # int32 either way, which is exactly why the bug above is invisible here.
    assert WEIGHT_MAX * Z_MAX_CM < np.iinfo(np.int32).max


def test_a_column_in_the_wrong_width_is_converted_not_refused():
    """The contract is int32 reflectivity, but a uint8 column is a reasonable
    thing for a caller to hold and must not crash the kernel -- it costs one
    copy, which is why `fusion.scatter()` does the conversion at the boundary
    instead."""
    agg = scatter_sorted(
        idx=np.array([4, 4]), z_cm=np.array([10, 20], np.int16),
        w_q=np.ones(2, np.int32), refl=np.array([200, 250], np.uint8),
        class_id=np.zeros(2, np.uint8), is_ground=np.ones(2, bool),
    )
    assert agg.refl_sum.tolist() == [450]     # not wrapped at 255


# --- the two paths must be the same map -------------------------------------


def test_sorted_and_atomic_paths_agree_field_for_field():
    """The sorted path is an optimisation of the atomic one specified in
    master v4 §3.5. If they ever diverge, the optimisation is a bug."""
    rng = np.random.default_rng(20260828)
    scan = random_scan(rng)
    a = scatter_sorted(**scan)
    b = scatter_atomic(**scan, n_cells=N_CELLS)
    for field in a.as_dict():
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field), err_msg=field)


def test_scratch_cost_is_reported_for_both_paths():
    """The sorted path's scratch is independent of grid size; the atomic
    path's is not. That is the entire argument for the default."""
    sorted_b = scatter_scratch_bytes("sorted", 745_000, 150_000)
    atomic_b = scatter_scratch_bytes("atomic", 745_000, 150_000)
    assert sorted_b < atomic_b
    assert scatter_scratch_bytes("sorted", 10 * 745_000, 150_000) == sorted_b
    assert scatter_scratch_bytes("atomic", 2 * 745_000, 150_000) == 2 * atomic_b
    with pytest.raises(ValueError, match="unknown scatter mode"):
        scatter_scratch_bytes("magic", 1, 1)


# --- the Day-2 gate: no allocation inside the frame loop ---------------------
#
# "Verify with a profiler, not by reading the code." Reading the code was
# exactly how this was got wrong the first time: `scatter_sorted` was handed a
# preallocated scratch by `allocate()`, never touched it, and allocated 19 MB a
# frame -- more than twice the whole 8.94 MB grid -- behind a docstring that
# said it didn't. These measure instead.


def _per_frame_bytes(fn, frames=3):
    """Peak transient allocation over a few steady-state frames."""
    import tracemalloc

    fn()  # warm: first call may touch lazily-initialised numpy internals
    tracemalloc.start()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    for _ in range(frames):
        fn()
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    return peak


def _scan_of(rng, n, n_cells):
    scan = random_scan(rng, n=n, n_cells=n_cells)
    return scan


def test_scatter_with_scratch_allocates_a_small_bounded_amount():
    """The shipping path runs in preallocated buffers.

    Not literally zero on the numpy stand-in: `np.compress` builds one index
    temporary of 8 bytes per touched cell that it gives no `out=` to reach. That
    residual is bounded by the cells one scan can touch, and on the CUDA port it
    is `cub::DeviceSelect::Flagged`'s temp storage, which IS preallocated. Every
    other step below writes through an `out=`.
    """
    rng = np.random.default_rng(4)
    n_cells = 200_000
    scan = _scan_of(rng, 60_000, n_cells)
    scratch = new_sorted_scratch(60_000, n_cells)

    touched = len(scatter_sorted(**scan, scratch=scratch))
    per_frame = _per_frame_bytes(lambda: scatter_sorted(**scan, scratch=scratch))

    # Generous factor over the one temporary; tight enough that reintroducing a
    # single per-point copy (8 bytes x 60,000 points) would break it.
    assert per_frame < 4 * 8 * touched, (
        f"{per_frame:,} B per frame against {touched:,} touched cells")
    assert per_frame < grid_bytes(n_cells) / 2


def test_dropping_the_scratch_is_caught_by_the_profiler():
    """Negative control. Without this the test above passes on an
    implementation that ignores its scratch entirely -- which is the bug it
    exists to catch, and the one that was actually there."""
    rng = np.random.default_rng(4)
    n_cells = 200_000
    scan = _scan_of(rng, 60_000, n_cells)
    scratch = new_sorted_scratch(60_000, n_cells)

    with_scratch = _per_frame_bytes(lambda: scatter_sorted(**scan, scratch=scratch))
    without = _per_frame_bytes(lambda: scatter_sorted(**scan))
    assert without > 5 * with_scratch, (
        f"private scratch cost {without:,} B, preallocated {with_scratch:,} B -- "
        "the profiler can no longer tell the two apart")


def test_per_frame_allocation_does_not_grow_with_the_grid():
    """The claim the memory bound rests on: the same scan into a grid ten times
    larger must not cost ten times the transient memory. A per-cell temporary
    anywhere on this path would show up here and nowhere else."""
    rng = np.random.default_rng(5)
    scan = _scan_of(rng, 40_000, 100_000)

    small = new_sorted_scratch(40_000, 100_000)
    large = new_sorted_scratch(40_000, 1_000_000)
    a = _per_frame_bytes(lambda: scatter_sorted(**scan, scratch=small))
    b = _per_frame_bytes(lambda: scatter_sorted(**scan, scratch=large))

    assert b < 1.5 * a + 4096, f"10x the grid cost {b:,} B against {a:,} B"
    assert scatter_scratch_bytes("sorted", 1_000_000, 40_000) == \
           scatter_scratch_bytes("sorted", 100_000, 40_000)


def test_scratch_arrays_are_reused_not_replaced():
    """Steady state: after many frames the buffers must be the same objects at
    the same sizes. A path that quietly swapped in a bigger array would keep
    the profiler happy for one frame and blow the bound on a busy one."""
    scratch = new_sorted_scratch(30_000, 50_000)
    before = {k: (id(v), v.nbytes) for k, v in scratch.items()}
    for seed in range(20):
        scatter_sorted(**_scan_of(np.random.default_rng(seed), 30_000, 50_000),
                       scratch=scratch)
    assert {k: (id(v), v.nbytes) for k, v in scratch.items()} == before


def test_too_many_points_is_refused_rather_than_reallocated():
    """The one place growth would be tempting. Growing the buffer here is an
    allocation in the frame loop, so it raises and names the config knob."""
    rng = np.random.default_rng(7)
    scratch = new_sorted_scratch(1_000, 4_096)
    with pytest.raises(ValueError, match="max_points_per_frame"):
        scatter_sorted(**_scan_of(rng, 2_000, 4_096), scratch=scratch)


def test_aggregate_is_a_view_into_the_scratch():
    """Documented contract, asserted so nobody relies on the opposite: the
    aggregate is valid only until the next scatter on the same scratch."""
    rng = np.random.default_rng(8)
    scratch = new_sorted_scratch(5_000, 4_096)
    agg = scatter_sorted(**_scan_of(rng, 5_000, 4_096), scratch=scratch)
    assert agg.cells.base is scratch["cells"]
    first = agg.wz_sum.copy()
    scatter_sorted(**_scan_of(np.random.default_rng(9), 5_000, 4_096), scratch=scratch)
    assert not np.array_equal(agg.wz_sum, first), (
        "the second scatter did not write through the first aggregate's buffers, "
        "so this contract is stale and the docstring is wrong")


# --- the height sums are GROUND evidence -------------------------------------


def test_a_cell_with_no_ground_return_has_no_height_evidence():
    """A wall, a car flank, a tree trunk: returns, but none of them measure
    the elevation of the ground under the cell.

    `w_sum == 0` is the signal, and it is unambiguous because
    `quantise_weight()` clips every point to >= 1 -- a zero sum cannot mean
    "ground returns that happened to weigh nothing". `mean_height_cm()` is 0
    here for want of anything better to return, which is precisely why the
    predicate exists rather than callers testing the mean against 0: in the
    vehicle frame the road sits near -173 cm, so 0 is a plausible-looking
    height rather than an obviously absent one.
    """
    agg = scatter_sorted(
        idx=np.array([2, 2, 2]), z_cm=np.array([100, 150, 200], np.int16),
        w_q=np.array([500, 500, 500], np.int32), refl=np.zeros(3, np.uint8),
        class_id=np.zeros(3, np.uint8), is_ground=np.zeros(3, bool),
    )
    assert agg.w_sum.tolist() == [0]
    assert agg.wz_sum.tolist() == [0]
    assert not agg.has_ground_evidence()[0]

    assert agg.n.tolist() == [3]                 # still three observations
    assert agg.ceiling_cm.tolist() == [100]      # and the lowest is the ceiling


def test_ground_evidence_is_per_cell_not_per_frame():
    """Mixed frame: one cell all ground, one all canopy, one both. The
    predicate has to separate them, because a frame is almost never uniform
    and a per-frame answer would be useless."""
    agg = scatter_sorted(
        idx=np.array([0, 1, 2, 2]), z_cm=np.array([10, 20, 30, 40], np.int16),
        w_q=np.full(4, 100, np.int32), refl=np.zeros(4, np.uint8),
        class_id=np.zeros(4, np.uint8),
        is_ground=np.array([True, False, True, False]),
    )
    assert agg.has_ground_evidence().tolist() == [True, False, True]
    assert agg.mean_height_cm().tolist() == [10, 0, 30]   # cell 2 excludes the 40


def test_the_ground_mask_does_not_split_the_two_paths():
    """The two scatter paths must stay bit-identical, and the mask is the
    newest chance for them to drift. This scan is built so that some cells are
    all-ground, some all-canopy and some mixed -- an all-ground scan would
    agree whether or not either path masked anything."""
    rng = np.random.default_rng(20260829)
    scan = random_scan(rng, n=3_000, n_cells=2048)   # ~1.5 returns per cell
    a = scatter_sorted(**scan)
    b = scatter_atomic(**scan, n_cells=2048)

    assert not np.all(a.has_ground_evidence()), "no all-canopy cell: test is vacuous"
    assert np.any(a.has_ground_evidence()), "no ground at all: test is vacuous"
    for field in a.as_dict():
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field), err_msg=field)


def test_the_weighted_mean_rounds_symmetrically_about_zero():
    """Half-away-from-zero, and the same rule on both sides of zero.

    The regression this pins: `(2*wz + sign(wz)*w) // (2*w)` floors, so after
    the away-from-zero nudge every negative quotient took one extra step down
    and came out 1 cm low -- exact values included. It looked right because
    the positive half is right, and because 1 cm is a plausible height error.

    It is not a curiosity at this sign. Vehicle frame is z up with the sensor
    at 1.73 m, so the road is near -173 cm and almost every ground cell in the
    map has a negative mean: the defect was a systematic 1 cm sag across the
    entire ground plane, against a §3.2 noise floor of 0.8 cm at 5 m.
    """
    w = 4000
    z_cm = np.arange(-200, 601)
    agg = CellAggregate(
        np.arange(z_cm.size, dtype=np.int64), (z_cm * w).astype(np.int64),
        np.full(z_cm.size, w, np.int64), np.ones(z_cm.size, np.int32),
        np.zeros(z_cm.size, np.int16), np.zeros(z_cm.size, np.int32),
        np.zeros(z_cm.size, np.uint8),
    )
    np.testing.assert_array_equal(agg.mean_height_cm(), z_cm,
                                  err_msg="an exact mean did not survive the round")


@pytest.mark.parametrize(("wz_sum", "expected"), [
    (250, 3), (150, 2), (100, 1), (50, 1), (49, 0),        # +0.5 rounds up
    (-250, -3), (-150, -2), (-100, -1), (-50, -1), (-49, 0),  # -0.5 rounds down
])
def test_the_mean_is_a_mirror_image_across_zero(wz_sum, expected):
    """Half-away-from-zero means |mean(-x)| == |mean(x)| for every input. A
    rule that is not symmetric puts a sign-dependent bias into the ground
    plane, which is exactly the shape of error a map hides best."""
    agg = CellAggregate(
        np.array([0]), np.array([wz_sum], np.int64), np.array([100], np.int64),
        np.ones(1, np.int32), np.zeros(1, np.int16), np.zeros(1, np.int32),
        np.zeros(1, np.uint8),
    )
    assert int(agg.mean_height_cm()[0]) == expected
