"""The cost -> bitfield decomposition behind the query survey. [Shrestha]

The survey itself builds a sequence and six maps and takes ~40 s, which
belongs in `scripts/`, not in CI. What is worth pinning here is the one piece
of arithmetic in it that could silently lie: `CostMap` stores the weight, not
the §7.1 bitfield, so `bits_from_cost` reads the population back out of the
weights. If that is wrong, the survey reports a wall set that is not the one
the planner would see -- and it reports it confidently, in a table, in a memo.

The failure mode being pinned is specific and it was real: a cell OBSERVED but
below `n_min` carries `w_unknown` through bit 5 while `CostMap.unknown` stays
False, so an implementation that subtracted `w_unknown` only where `unknown`
was set dropped those cells out of every bucket and the counts quietly stopped
summing to the window.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture(scope="module")
def decompose():
    pytest.importorskip("vrgrid.eval.plan_regret")
    from plan_query_survey import bits_from_cost
    return bits_from_cost


W = {"w_base": 1.0, "w_roughness": 2.0, "w_class": 3.0, "w_unknown": 4.0}


def _costmap(cost, unknown=None):
    from vrgrid.eval.plan_regret import CostMap
    cost = np.asarray(cost, dtype=float)
    if unknown is None:
        unknown = np.zeros(cost.shape, dtype=bool)
    return CostMap(0.25, 0.0, 0.0, cost, np.asarray(unknown, dtype=bool))


def test_each_soft_weight_lands_in_its_own_bucket(decompose):
    """w_base + one soft weight is that bit and nothing else."""
    cm = _costmap([[1.0, 3.0], [4.0, 5.0]])
    b = decompose(cm, W)
    assert b == {"plain": 1, "roughness": 1, "class": 1, "low_confidence": 1,
                 "impassable": 0, "unknown": 0}


def test_two_bits_on_one_cell_are_counted_under_both(decompose):
    """Roughness and class together are 1 + 2 + 3, and that cell is in each
    count -- the buckets overlap by construction, which is why the totals are
    checked against the window rather than against their own sum."""
    cm = _costmap([[6.0]])
    b = decompose(cm, W)
    assert b["roughness"] == 1
    assert b["class"] == 1
    assert b["plain"] == 0


def test_infinite_cost_is_impassable_and_never_a_soft_bit(decompose):
    cm = _costmap([[np.inf, np.inf], [1.0, 1.0]])
    b = decompose(cm, W)
    assert b["impassable"] == 2
    assert b["plain"] == 2
    assert b["roughness"] == b["class"] == b["low_confidence"] == 0


def test_observed_but_below_n_min_is_counted_though_unknown_is_false(decompose):
    """The regression. Bit 5 charges `w_unknown` on a cell that IS observed,
    so `CostMap.unknown` is False and the weight is still there. It must be
    counted, and it must not be counted as unobserved."""
    cm = _costmap([[5.0]], unknown=[[False]])
    b = decompose(cm, W)
    assert b["low_confidence"] == 1
    assert b["unknown"] == 0
    assert b["plain"] == 0


def test_unobserved_cells_are_reported_separately_from_the_weight(decompose):
    """`unknown` is the flag; `low_confidence` is the weight. An unobserved
    cell has both, and the two counts answer different questions."""
    cm = _costmap([[5.0]], unknown=[[True]])
    b = decompose(cm, W)
    assert b["unknown"] == 1
    assert b["low_confidence"] == 1


def test_a_cost_that_is_not_a_subset_sum_raises_rather_than_miscounts(decompose):
    """The whole point of the assertion. 1 + 1.5 is not w_base plus any subset
    of the soft weights, so the decomposition is not valid and must say so
    instead of returning counts that do not add up."""
    cm = _costmap([[2.5]])
    with pytest.raises(AssertionError, match="fell into no bucket"):
        decompose(cm, W)


def test_weights_without_distinct_subset_sums_are_refused(decompose):
    """With w_roughness == w_class the sums collide and a cost can no longer
    be attributed to a bit. Refuse rather than guess."""
    cm = _costmap([[1.0]])
    with pytest.raises(AssertionError, match="distinct subset sums"):
        decompose(cm, {"w_base": 1.0, "w_roughness": 2.0, "w_class": 2.0,
                       "w_unknown": 4.0})
