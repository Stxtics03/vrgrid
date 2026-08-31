"""The monotonicity check behind the Day-4 figure. [Shrestha]

The sweep itself takes ~40 s and belongs in `scripts/`, not in CI. What is
worth testing here is the one piece of judgement in `regret_plot.py`: whether
it notices that the rows contradict the claim the figure makes. A plotting
script that will happily draw a rising line through falling data is how a
figure ends up on a slide arguing against its own caption.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture(scope="module")
def check():
    pytest.importorskip("vrgrid.eval.harness")
    from regret_plot import monotonicity
    return monotonicity


def _row(name, mb, regret):
    return {"schedule": name, "megabytes": mb, "regret": regret}


def test_a_curve_that_matches_the_claim_is_reported_monotone(check):
    """§8.2's claim: cheaper maps make worse decisions, so regret does not
    fall as memory falls."""
    rows = [_row("a", 29.0, 0.0), _row("b", 18.0, 0.4), _row("c", 7.0, 1.9)]
    assert check(rows).startswith("monotone")


def test_flat_zero_is_monotone_but_says_nothing(check):
    """All-zero regret is not a violation — it is a curve with no knee, which
    the caller has to notice for itself. Recorded here so nobody later
    'fixes' the check into flagging it."""
    assert check([_row("a", 29.0, 0.0), _row("b", 7.0, 0.0)]).startswith("monotone")


def test_a_coarser_map_scoring_better_is_flagged(check):
    """The case actually observed on the synthetic sequence: uniform 10 cm
    scores 1.389 while the coarser 20/40/80 cm all score 0."""
    rows = [_row("5_10_20_40", 29.06, 0.0), _row("uniform_10cm", 18.19, 1.389),
            _row("uniform_20cm", 10.71, 0.0), _row("uniform_80cm", 7.09, 0.0)]
    out = check(rows)
    assert "NOT MONOTONE" in out
    assert "uniform_10cm -> uniform_20cm" in out


def test_every_violating_step_is_named(check):
    rows = [_row("a", 30.0, 0.0), _row("b", 20.0, 2.0), _row("c", 10.0, 1.0),
            _row("d", 5.0, 0.5)]
    out = check(rows)
    assert "2 step(s)" in out and "b -> c" in out and "c -> d" in out


def test_missing_regret_is_skipped_not_crashed(check):
    """`memory_vs_regret_row` returns regret=None when no Regret was passed,
    and a half-populated sweep must not take the figure down with it."""
    rows = [_row("a", 29.0, None), _row("b", 18.0, 0.4), _row("c", 7.0, None)]
    assert isinstance(check(rows), str)
