"""Scatter kernel. [Shrestha]

The determinism tests live in tests/test_determinism.py; these cover the
arithmetic and the equivalence of the two paths.
"""

import numpy as np
import pytest
from vrgrid.gpu.kernels import (
    CEILING_NONE,
    CLASS_RADIX,
    WEIGHT_MAX,
    measurement_variance_cm2,
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
    """Three returns in cell 7, one in cell 9, one dropped."""
    agg = scatter_sorted(
        idx=np.array([7, 9, 7, -1, 7]),
        z_cm=np.array([100, 50, 200, 999, 300], np.int16),
        w_q=np.array([2, 5, 3, 7, 5], np.int32),
        refl=np.array([10, 20, 30, 40, 50], np.uint8),
        class_id=np.array([1, 2, 3, 4, 5], np.uint8),
        is_ground=np.array([True, True, False, True, True]),
    )
    assert agg.cells.tolist() == [7, 9]
    assert agg.wz_sum.tolist() == [2 * 100 + 3 * 200 + 5 * 300, 5 * 50]
    assert agg.w_sum.tolist() == [10, 5]
    assert agg.n.tolist() == [3, 1]
    assert agg.refl_sum.tolist() == [90, 20]
    assert agg.ceiling_cm.tolist() == [200, CEILING_NONE]   # only the non-ground return
    assert agg.class_id.tolist() == [1, 2]                  # lowest-indexed point
    assert agg.mean_height_cm().tolist() == [230, 50]       # 2300/10


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
