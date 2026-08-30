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


@pytest.mark.skip(reason="needs src/perception/loader.py — JP")
@pytest.mark.determinism
def test_real_sequence_replay_is_identical():
    """The gate as stated: 50 frames of sequence 08, twice, byte-identical."""
    raise NotImplementedError


@pytest.mark.skip(reason="needs the full frame loop — Day 2")
def test_no_allocation_inside_the_frame_loop():
    """Day-2 gate: verify with a profiler, not by reading the code. The
    array-level version of this check is in tests/test_allocators.py."""
    raise NotImplementedError
