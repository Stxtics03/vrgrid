"""Split/merge theorems. Math §4–5. [Aakash]

These are proofs, not tuning targets. If one fails, the implementation is
wrong — do not weaken the test to make it pass.

Two of them carry a negative control, because a split/merge suite is unusually
easy to make worthless: the correct rule and the wrong rule agree on flat
ground, on identical children, and on any test whose numbers were chosen for
convenience. Every assertion below is on a case where they disagree.
"""

import itertools

import numpy as np
import pytest
from vrgrid.cell import FLAG_BLIND, FLAG_DERIVED, FLAG_DYNAMIC
from vrgrid.grid.lattice import migrate_ring, ring_of
from vrgrid.grid.schedule import load
from vrgrid.grid.splitmerge import (
    KAPPA_FROM_GEOMETRY,
    CellValue,
    SplitParams,
    clear_derived,
    inflate,
    law_of_total_variance,
    load_params,
    merge,
    split,
)

# The §4.1 worked example, in metres: four children straddling a 12 cm kerb,
# each measured to sigma = 2 cm.
KERB_M = 0.12
CHILD_SIGMA_M = 0.02


def _kerb_children():
    s2 = CHILD_SIGMA_M**2
    return tuple(CellValue(mu_m=h, sigma2_m2=s2, n=4)
                 for h in (0.0, 0.0, KERB_M, KERB_M))


def _inverse_variance(children):
    """The WRONG rule, implemented so the tests can show it is wrong rather
    than assert it. 1/sigma2 = sum(1/sigma2_i), the ML combination of repeated
    measurements of ONE quantity. Math §4.1."""
    return 1.0 / sum(1.0 / c.sigma2_m2 for c in children)


# --- §4: merge ---------------------------------------------------------------


@pytest.mark.theorem
def test_merge_uses_law_of_total_variance():
    """sigma2_p = sum(w_i sigma_i^2) + sum(w_i (mu_i - mu_p)^2).

    Constructed case: four children with identical tiny variance but means
    straddling a 12 cm kerb. Inverse-variance fusion returns a *smaller*
    variance than any child — confidently wrong exactly at the kerb. The
    correct rule returns a variance dominated by the between-cell spread.
    """
    children = _kerb_children()
    parent = merge(children)

    # eq. (15): the mean is the plain weighted mean either way.
    assert parent.mu_m == pytest.approx(KERB_M / 2)

    # eq. (16), the number §4.2 quotes: 0.0004 within + 0.0036 between.
    assert parent.sigma2_m2 == pytest.approx(0.0004 + 0.0036)
    assert parent.sigma_cm == pytest.approx(6.32, abs=0.01)

    # The point of the section: the merged cell is LESS certain than any
    # child, because it now spans a step it did not previously span.
    assert parent.sigma2_m2 > max(c.sigma2_m2 for c in children)

    # Negative control. The rule we are not using claims twice the certainty
    # of any child, sitting on top of the step it just erased.
    naive = _inverse_variance(children)
    assert naive < min(c.sigma2_m2 for c in children)
    assert np.sqrt(naive) == pytest.approx(0.01, abs=1e-6)      # 1 cm, §4.1
    assert parent.sigma2_m2 / naive == pytest.approx(40.0)


@pytest.mark.theorem
def test_between_term_vanishes_for_identical_means():
    """§4.3 unit test, first half: identical means -> sigma2_p = sum(w sigma2_i)
    exactly. This is the case where the correct and the naive rule are most
    tempting to confuse, because the between-term contributes nothing."""
    s2 = (0.0001, 0.0004, 0.0009, 0.0016)
    children = tuple(CellValue(mu_m=0.31, sigma2_m2=v, n=1) for v in s2)
    parent = merge(children)

    assert parent.mu_m == 0.31
    assert parent.sigma2_m2 == pytest.approx(sum(s2) / 4, rel=0, abs=0)
    # §4.3's corollary, the weak inequality, holds with equality here.
    assert parent.sigma2_m2 >= min(s2)


@pytest.mark.theorem
def test_merge_on_a_step_exceeds_delta_squared_over_four():
    """§4.3 unit test, second half: four children on a synthetic step of height
    Delta -> sigma2_p >= Delta^2/4, for any Delta and any child variance."""
    for delta in (0.02, 0.12, 0.35, 1.0):
        children = tuple(CellValue(mu_m=h, sigma2_m2=1e-6, n=1)
                         for h in (0.0, 0.0, delta, delta))
        assert merge(children).sigma2_m2 >= delta**2 / 4


@pytest.mark.theorem
def test_merge_never_falls_below_the_average_child_variance():
    """§4.3 corollary: sigma2_p >= sum(w_i sigma2_i) >= min_i sigma2_i, on
    random children, weights and counts. The one-line summary of why merging
    can never manufacture confidence."""
    rng = np.random.default_rng(20260829)
    for _ in range(2000):
        mus = rng.uniform(-2.0, 2.0, 4)
        s2s = rng.uniform(1e-6, 1e-2, 4)
        counts = rng.integers(0, 40, 4)
        children = tuple(CellValue(float(m), float(v), int(n))
                         for m, v, n in zip(mus, s2s, counts))
        parent = merge(children)
        w = counts / counts.sum() if counts.sum() else np.full(4, 0.25)
        assert parent.sigma2_m2 >= float((w * s2s).sum()) - 1e-15
        assert parent.sigma2_m2 >= float(s2s.min()) - 1e-15


def test_merged_confidence_is_the_least_confident_child():
    """obs_count merges by MIN, not by sum.

    This is what keeps §7.2 Theorem 3 (no false negatives) true through a
    merge. n is the field TRAV_CONFIDENCE fails on, so if four children with
    n = 30, 30, 30, 1 merged to n = 91, a parent containing an effectively
    unobserved cell would report SAFE — the exact failure Theorem 3 forbids,
    arriving through the merge rule rather than through the pyramid.
    """
    children = tuple(CellValue(0.0, 1e-4, n) for n in (30, 30, 30, 1))
    assert merge(children).n == 1


def test_merge_flags_are_unanimous_where_the_footprint_demands_it():
    """DERIVED and BLIND describe the whole footprint, so one measured or one
    seen child clears them. DYNAMIC is safety-relevant upward, so any child
    carries it up."""
    base = [CellValue(0.0, 1e-4, 2, FLAG_DERIVED | FLAG_BLIND) for _ in range(4)]
    assert merge(base).flags & FLAG_BLIND

    seen = list(base)
    seen[2] = CellValue(0.0, 1e-4, 2, FLAG_DERIVED)   # one child not blind
    assert not merge(seen).flags & FLAG_BLIND

    moving = list(base)
    moving[1] = CellValue(0.0, 1e-4, 2, FLAG_DERIVED | FLAG_BLIND | FLAG_DYNAMIC)
    assert merge(moving).flags & FLAG_DYNAMIC


def test_merge_rejects_a_child_count_that_is_not_a_square():
    """m^2 children of one parent, m the refinement ratio. Three children is a
    caller bug, and silently averaging them would produce a plausible map."""
    with pytest.raises(ValueError, match="m\\^2 children"):
        merge([CellValue(0.0, 1e-4)] * 3)


# --- §5: split ---------------------------------------------------------------


@pytest.mark.theorem
def test_split_strictly_inflates_variance():
    """Children inherit mu_p with strictly larger variance, and FLAG_DERIVED set."""
    s = load("5/10/20/40")
    parent = CellValue(mu_m=0.42, sigma2_m2=CHILD_SIGMA_M**2, n=7)

    for ring in (1, 2, 3):
        children = split(parent, s, ring, grad_z=0.2)
        assert len(children) == 4
        for c in children:
            assert c.mu_m == parent.mu_m            # §5.1: the mean is forced
            assert c.sigma2_m2 > parent.sigma2_m2   # Theorem 1, strictly
            assert c.derived
            assert c.n == parent.n

    # Theorem 1 is about the geometry, so a coarser parent must cost more:
    # (c_p^2 - c_c^2) grows with the ring.
    grew = [split(parent, s, r, 0.2)[0].sigma2_m2 for r in (1, 2, 3)]
    assert grew[0] < grew[1] < grew[2]


@pytest.mark.theorem
def test_split_on_flat_ground_is_free():
    """§5.3's limiting behaviour and §5.4 unit test (c): grad z = 0 -> the
    child variance equals the parent's exactly.

    Right, and not merely convenient: splitting a flat surface genuinely loses
    no information, so a formula that charged for it would be inventing
    uncertainty to be conservative — the same sin as inverse-variance fusion
    in the other direction.
    """
    s = load("5/10/20/40")
    parent = CellValue(mu_m=-0.03, sigma2_m2=7.5e-5, n=3)
    for c in split(parent, s, 3, grad_z=0.0):
        assert c.sigma2_m2 == parent.sigma2_m2      # exactly, not approx
        assert c.derived


def test_split_refuses_the_base_lattice():
    """Ring 0 is c0 = 5 cm and there is nothing finer. Semantic refinement
    goes into the pool at ring-0 resolution, never below it (master v4 §3.4)."""
    s = load("5/10/20/40")
    with pytest.raises(ValueError, match="base lattice"):
        split(CellValue(0.0, 1e-4), s, 0)


def test_split_follows_the_schedule_not_the_number_four():
    """The 5/10/50 ablation refines 5x between rings 1 and 2, so that split
    produces 25 children, not 4.

    §5.2 is written for c_c = c_p/2 throughout and it is easy to hardwire the
    four. The schedule validator permits any integer ratio — 5/10/50 is legal
    precisely because 10/5 = 2 and 50/10 = 5 — and it is the schedule the
    memory claim is quoted on, so this path is not hypothetical.
    """
    ab = load("5/10/50")
    parent = CellValue(0.11, 1e-4, n=5)

    assert len(split(parent, ab, 1)) == 4       # 10 -> 5 cm, m = 2
    assert len(split(parent, ab, 2)) == 25      # 50 -> 10 cm, m = 5

    # and the 25 merge back through the same rule
    assert merge(split(parent, ab, 2, grad_z=0.3)) is parent


# --- §5.4: Theorem 2, the round trip ----------------------------------------


@pytest.mark.theorem
def test_round_trip_idempotence():
    """merge(split(c)) == c exactly, in mean AND variance, when no measurement
    intervenes. Math §5, Theorem 2.

    This is what the `derived` bit buys. Without it a cell oscillating across a
    ring boundary as the vehicle changes speed inflates variance every frame
    with no physical cause, and the map drifts toward uncertainty.

    Exactly, not approximately: `==` on the floats, not pytest.approx. A round
    trip that is right to 1e-12 per cycle is still unbounded drift over a
    30-second sequence at 10 Hz, and the whole point of §5.4 is that the bit
    makes the drift zero rather than small.
    """
    rng = np.random.default_rng(20260829)
    for name in ("5/10/20/40", "5/10/50"):
        s = load(name)
        for ring in range(1, len(s.rings)):
            for _ in range(200):
                parent = CellValue(mu_m=float(rng.uniform(-3.0, 3.0)),
                                   sigma2_m2=float(rng.uniform(1e-8, 1e-2)),
                                   n=int(rng.integers(0, 200)))
                grad = float(rng.uniform(0.0, 0.6))
                back = merge(split(parent, s, ring, grad))
                assert back.mu_m == parent.mu_m
                assert back.sigma2_m2 == parent.sigma2_m2
                assert back.n == parent.n
                assert back.flags == parent.flags


@pytest.mark.theorem
def test_round_trip_without_the_bit_drifts_every_cycle():
    """The negative control for Theorem 2, and the reason the bit exists.

    Merge the same children by (16) alone — the branch a correct
    implementation takes for genuinely measured children — and the split's
    inflation is never given back: the between-term is zero because all four
    children carry mu_p, so sigma2_merged = sigma2_child = sigma2_p + Delta.
    Variance then grows by Delta on every cycle, with nothing physical
    happening. 100 cycles on a 20% slope is a 5 cm cell claiming 3 cm of
    uncertainty it invented.
    """
    s = load("5/10/20/40")
    start = CellValue(mu_m=0.0, sigma2_m2=1e-6, n=4)

    with_bit = start
    without_bit = start
    for _ in range(100):
        with_bit = merge(split(with_bit, s, 3, grad_z=0.2))
        # same split, but strip what §5.4 added, i.e. pretend the bit is not there
        stripped = [clear_derived(c) for c in split(without_bit, s, 3, grad_z=0.2)]
        without_bit = merge(stripped)

    delta = inflate(0.0, 0.2, s.rings[3].cell_m, s.rings[2].cell_m)
    assert with_bit.sigma2_m2 == start.sigma2_m2
    assert without_bit.sigma2_m2 == pytest.approx(start.sigma2_m2 + 100 * delta)
    assert without_bit.sigma_cm > 3.0


@pytest.mark.theorem
def test_one_measurement_takes_the_merge_off_the_restore_path():
    """§5.4 unit test (d): split, inject a measurement into one child, merge —
    the result must follow (16), not the restore.

    The restore branch is the inverse of split only while nothing has been
    measured. One new return in one child makes the four children a genuine
    observation of four different places, and the parent must pay the
    between-cell term for the disagreement.
    """
    s = load("5/10/20/40")
    parent = CellValue(mu_m=0.0, sigma2_m2=1e-6, n=4)
    children = list(split(parent, s, 3, grad_z=0.1))

    # fusion.py's job; here, one child measured 8 cm higher and clears the bit
    measured = clear_derived(children[0])
    children[0] = CellValue(0.08, 4e-4, measured.n + 1, measured.flags)

    back = merge(children)
    assert back is not parent
    assert not back.derived
    assert back.mu_m != parent.mu_m
    assert back.sigma2_m2 > parent.sigma2_m2

    # and it is (16) that produced it, not something else
    mu = [c.mu_m for c in children]
    s2 = [c.sigma2_m2 for c in children]
    n = [c.n for c in children]
    expect_mu, expect_s2 = law_of_total_variance(mu, s2, n)
    assert back.mu_m == pytest.approx(float(expect_mu))
    assert back.sigma2_m2 == pytest.approx(float(expect_s2))


def test_a_count_that_moved_behind_the_bit_does_not_restore():
    """Defensive: if fusion ever writes a measurement without calling
    clear_derived(), the count moves while the bit still says `derived`.
    Restoring then would hand back a value that is no longer the marginal of
    what is stored. Fall through to (16) instead — wrong-ish, but not silently
    wrong, and the assertion documents which failure we chose."""
    s = load("5/10/20/40")
    parent = CellValue(0.0, 1e-6, n=4)
    children = list(split(parent, s, 3, grad_z=0.1))
    children[2] = CellValue(children[2].mu_m, children[2].sigma2_m2,
                            children[2].n + 1, children[2].flags,
                            children[2].derived_from)
    assert merge(children) is not parent


# --- §6.3 hysteresis, the reason §5.4 is load-bearing ------------------------


def test_hysteresis_prevents_boundary_thrash():
    """Split at R_L, merge only at R_L(1+eps). A cell parked on the boundary
    must not split and merge on consecutive frames.

    Same parked cell as tests/test_lattice.py's hysteresis test, driven
    through the actual split/merge pair rather than through ring indices, so
    what is measured is the thing §6.3 exists to protect: refinement-pool
    churn, and the variance inflation of §5.4 that rides on it.
    """
    s = load("5/10/20/40")
    x, y = 0.0, 9.9                     # on R_0 = 10 m, laterally
    speeds = 1.5 + 1.5 * np.sin(np.linspace(0.0, 200.0 * np.pi, 1_000))

    naive = [ring_of(x, y, s, float(v)) for v in speeds]
    naive_events = sum(a != b for a, b in itertools.pairwise(naive))
    assert naive_events > 100, "test point is not actually sitting on a boundary"

    cell = CellValue(mu_m=0.05, sigma2_m2=1e-6, n=9)
    ring = ring_of(x, y, s, float(speeds[0]))
    events = 0
    for v in speeds[1:]:
        nxt = migrate_ring(x, y, s, ring, float(v))
        if nxt == ring:
            continue
        events += 1
        if nxt < ring:                       # finer: split into ring nxt
            children = split(cell, s, ring, grad_z=0.15)
            cell = children[0]
        else:                                # coarser: merge back
            cell = merge((cell,) * 4)
        ring = nxt

    assert events <= 4, f"hysteresis let {events} split/merge events through"
    # and because the bit survived every one of them, nothing was invented
    assert cell.sigma2_m2 <= inflate(1e-6, 0.15, s.rings[1].cell_m,
                                     s.rings[0].cell_m)


# --- the two documented discrepancies, kept visible --------------------------


def test_config_carries_kappa_and_alpha():
    """No threshold is inline. CLAUDE.md, and flaw E6: a constant that lives in
    the source cannot be frozen before the schedule comparison."""
    p = load_params()
    assert p.kappa == 0.0625
    assert p.alpha_m2 == 0.0


@pytest.mark.theorem
def test_kappa_from_geometry_is_one_twelfth_at_every_ratio():
    """§5.2 says "kappa = 1/16 from the offset geometry (d^2 = c_p^2/16)".

    The geometry is right and the constant does not follow from it, because
    (17) multiplies kappa by (c_p^2 - c_c^2) rather than by c_p^2. The
    mean-square child-centre offset per axis for an m x m split is

        c_p^2 (m^2 - 1) / (12 m^2)   ==   (c_p^2 - c_c^2) / 12

    so the geometry gives kappa = 1/12 — the same value at every m, which is
    what lets the ablation's 5x refinement use the same formula. 1/16
    under-inflates by a factor 3/4, uniformly.

    Not changed here: a frozen constant is a room decision. This test holds
    both numbers so the decision cannot be forgotten, and fails the day
    someone edits the config, which is when the document must be edited too.
    """
    c_p, g = 0.40, 0.25
    geometric = SplitParams(kappa=KAPPA_FROM_GEOMETRY, alpha_m2=0.0)
    as_written = SplitParams(kappa=load_params().kappa, alpha_m2=0.0)

    for m in (2, 3, 4, 5, 8):
        c_c = c_p / m
        mean_square_offset = c_p**2 * (m**2 - 1) / (12 * m**2)
        assert inflate(0.0, g, c_p, c_c, geometric) == pytest.approx(
            g**2 * mean_square_offset
        )
        # and the shortfall is the same 3/4 at every ratio, so this is one
        # constant to decide, not one per schedule
        assert (inflate(0.0, g, c_p, c_c, as_written)
                / inflate(0.0, g, c_p, c_c, geometric)) == pytest.approx(0.75)


def test_alpha_would_break_the_flat_ground_remark():
    """(17) adds alpha unconditionally, so any alpha > 0 falsifies §5.3's
    "splitting a flat surface costs nothing" and §5.4's unit test (c).

    Theorem 1 survives — alpha > 0 only strengthens the strict inequality —
    which is why this is a documentation defect rather than a maths one, and
    why it is easy to walk past. alpha is 0 today only because §9 cannot
    calibrate it without the reference map. This test is the tripwire: it
    passes now, and the day someone sets alpha it fails, in the same commit
    where §5.3 and §5.4(c) have to be rewritten.
    """
    s = load("5/10/20/40")
    parent = CellValue(0.0, 1e-6, n=2)
    rough = SplitParams(kappa=load_params().kappa, alpha_m2=1e-4)

    flat_free = split(parent, s, 3, 0.0)[0]
    flat_charged = split(parent, s, 3, 0.0, rough)[0]

    assert flat_free.sigma2_m2 == parent.sigma2_m2       # §5.4(c), as written
    assert flat_charged.sigma2_m2 > parent.sigma2_m2     # what alpha > 0 means
    assert load_params().alpha_m2 == 0.0, (
        "alpha is now non-zero: §5.3's flat-ground remark and §5.4 unit test "
        "(c) have to be restated in this commit"
    )
