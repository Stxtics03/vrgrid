"""Lattice and partition. Math §2. [Aakash]

Delete the skip decorators as the implementations land. The partition test is
CI-blocking: it is the proof that there is no epsilon to tune.
"""

import itertools
import math
import random
from fractions import Fraction

import numpy as np
import pytest
from vrgrid.cell import CELL_FIELDS
from vrgrid.grid.lattice import (
    OUTSIDE,
    alloc_ring_buffers,
    buffer_cells,
    i_fine,
    i_ring,
    migrate_ring,
    ring_extent,
    ring_of,
    toroidal_shift,
)
from vrgrid.grid.schedule import load

C0 = 0.05
N_POINTS = 1_000_000
N_EXACT = 20_000  # subsample checked against exact rational arithmetic

# Fixed seed on purpose. A CI-blocking gate must fail reproducibly or not at
# all -- a partition test that fails one run in a thousand teaches the team to
# re-run CI instead of reading it.
SEED = 20260828

SCHEDULES = ["5/10/20/40", "5/10/50"]


def _ks(schedule_name):
    """Integer divisors k_L = c_L / c0 for every ring of a frozen schedule."""
    s = load(schedule_name)
    return [s.k(r.ring) for r in s.rings]


def _true_index(x, k):
    """Which cell of the size-(k*c0) lattice x really falls in, in exact
    rational arithmetic. c0 is the double 0.05, not the decimal 0.05 -- the
    lattice the code implements is the one built on that double."""
    return math.floor(Fraction(x) / (Fraction(C0) * k))


@pytest.mark.theorem
def test_nested_floor_identity():
    """floor(floor(x/c0)/k) == floor(x/(k*c0)) for all x, integer k >= 1.

    This is why ring lattices nest exactly rather than approximately.
    Graham, Knuth & Patashnik, Concrete Mathematics eq. 3.11. Math §2.2.

    The right-hand side is evaluated in EXACT rational arithmetic rather than
    as x // (k*c0). The theorem is a statement about real numbers, and k * c0
    is itself a rounded double: 10 * fl(0.05) rounds down to exactly 0.5 while
    ten fine cells truly span 0.5000000000000000277. Evaluating the right-hand
    side in floats measures that rounding, not the theorem -- see
    test_direct_float_lattice_disagrees_at_ring_boundaries.
    """
    rng = random.Random(SEED)
    for _ in range(100_000):
        x = rng.uniform(-100.0, 100.0)
        for k in (1, 2, 4, 8, 10):
            assert i_ring(x, C0, k) == _true_index(x, k)
            assert i_ring(x, C0, k) == i_fine(x, C0) // k


@pytest.mark.partition
@pytest.mark.parametrize("schedule_name", SCHEDULES)
def test_partition_one_cell_per_ring_per_point(schedule_name):
    """10^6 random points: every point lands in exactly one cell of each ring.
    Never zero, never two. CI-blocking. Math §2.3.

    Both frozen schedules, because the ablation's k=10 ring is the non-power-
    of-two case: if anything in here quietly assumes a bit shift, 5/10/50 is
    what catches it.
    """
    rng = np.random.default_rng(SEED)
    xy = rng.uniform(-100.0, 100.0, size=(2, N_POINTS))

    for axis in xy:  # separable lattice: the cell is (i_L(x), i_L(y))
        f = i_fine(axis, C0)

        # The base cell really does contain the point: [f*c0, (f+1)*c0).
        assert np.all(f * C0 <= axis)
        assert np.all(axis < (f + 1) * C0)

        for k in _ks(schedule_name):
            i = i_ring(axis, C0, k)

            # Exactly one, anchored at the index actually returned. Counting
            # how many of {i-1, i, i+1} contain the point is NOT enough: the
            # cells are disjoint by construction, so that count is 1 even when
            # i is off by one. Existence has to be asserted at i itself.
            def contains(j, _f=f, _k=k):
                return ((j * _k) <= _f) & (_f < (j + 1) * _k)

            missed = int((~contains(i)).sum())
            assert missed == 0, f"k={k}: {missed} points outside the cell returned for them"

            doubled = int((contains(i - 1) | contains(i + 1)).sum())
            assert doubled == 0, f"k={k}: {doubled} points also claimed by a neighbour"

            # And the derived index is the true cell of the size-(k*c0)
            # lattice, checked in exact rational arithmetic on a subsample.
            # NOT against axis // (k*c0): that is a second float lattice, and
            # it is the one that is wrong -- see the disagreement test below.
            for x, idx in zip(axis[:N_EXACT], i[:N_EXACT]):
                assert idx == _true_index(x, k)


@pytest.mark.partition
def test_direct_float_lattice_disagrees_at_ring_boundaries():
    """Why (9) derives the index instead of recomputing it -- with the
    counterexample, not just the warning.

    docs/sih-math.md §2.4 item (b) asks for i_L == floor(x/(k_L*c0)) computed
    directly, bit-exact for all rings. That holds for k in {2,4,8} and is
    FALSE for the ablation schedule's k=10, at every boundary of the direct
    lattice.

    fl(0.05) = 3602879701896397/2^56 is slightly greater than 1/20, so ten
    fine cells span 0.5000000000000000277..., but fl(10*0.05) rounds to
    exactly 0.5. The double 0.5 therefore lies inside fine cell 9 -- ring
    cell 0 -- while the direct lattice calls it ring cell 1. The point is in
    two cells or in neither depending on which line of code you ask. This is
    exactly the failure §2.3 removes, and the derived index is the right one.

    Every claim §2.3 now makes about this is asserted below, so the document
    and the code cannot drift apart either.
    """
    assert i_fine(0.5, C0) == 9
    assert i_ring(0.5, C0, 10) == 0
    assert _true_index(0.5, 10) == 0     # exact arithmetic agrees with derived
    assert int(0.5 // (10 * C0)) == 1    # the direct float lattice, disagreeing

    def direct(x, k):
        return int(x // (k * C0))

    # Powers of two are the case where the shortcut is accidentally safe:
    # k*fl(c0) is representable, so the two lattices are the same lattice.
    # The default schedule is all powers of two and cannot catch this.
    for k in (1, 2, 4, 8):
        assert all(direct(m * k * C0, k) == _true_index(m * k * C0, k)
                   for m in range(-4000, 4001))

    # Non-power-of-two k: wrong at every positive boundary, and the derived
    # index is right at every one of them.
    for k in (5, 10, 20):
        pos = [m * (k * C0) for m in range(1, 4001)]
        neg = [-m * (k * C0) for m in range(1, 4001)]
        assert all(i_ring(x, C0, k) == _true_index(x, k) for x in pos + neg)
        assert sum(direct(x, k) != _true_index(x, k) for x in pos) == 4000
        # One-sided: the naive cell is narrower than the k fine cells it
        # should contain, so below zero the flooring absorbs the shortfall.
        assert sum(direct(x, k) != _true_index(x, k) for x in neg) == 0

    # And the defect has measure zero -- it is AT the boundary doubles, not
    # in a neighbourhood of them. This is why random sampling never finds it,
    # and why §2.4(b) has to compare against exact arithmetic instead.
    probes = off_boundary = at_boundary = 0
    for m in range(-4000, 4001):
        b = m * (10 * C0)
        neighbours = []
        for direction in (math.inf, -math.inf):
            y = b
            for _ in range(4):
                y = math.nextafter(y, direction)
                neighbours.append(y)
        for x in [b] + neighbours:
            probes += 1
            if direct(x, 10) != _true_index(x, 10):
                if x == b:
                    at_boundary += 1
                else:
                    off_boundary += 1
    assert (probes, at_boundary, off_boundary) == (72_009, 4000, 0)


@pytest.mark.partition
def test_no_gap_at_ring_boundary():
    """The failure the integer lattice exists to prevent: computing
    floor(x/0.20) directly puts points near a boundary in both cells or
    neither, because 0.2 is not representable in binary.

    Probed from the fine lattice so the test itself has no float boundary
    case of its own -- a fine-cell midpoint sits 2.5 cm from any edge, some
    10^13 ulps clear. The fine cell just below a ring boundary must belong to
    ring cell m-1 and the one just above it to ring cell m: consecutive, no
    gap between them and no cell claimed twice.
    """
    for k in _ks("5/10/20/40")[1:] + _ks("5/10/50")[1:]:  # k=1 has no boundary
        for m in range(-500, 500):
            for f, expected in ((m * k - 1, m - 1), (m * k, m)):
                x = (f + 0.5) * C0
                assert i_fine(x, C0) == f
                assert i_ring(x, C0, k) == expected


@pytest.mark.partition
def test_floor_not_truncation_at_the_origin():
    """int(x/c0) truncates toward zero, so -0.02 and +0.02 both land in cell 0
    and the cell straddling the origin silently comes out twice the size of
    every other cell -- a partition violation exactly where the vehicle is.
    """
    assert i_fine(-0.02, C0) == -1
    assert i_fine(0.02, C0) == 0
    assert i_fine(-0.05, C0) == -1  # half-open: cell -1 is [-0.05, 0)
    assert i_fine(0.0, C0) == 0

    for k in _ks("5/10/20/40"):
        assert i_ring(-0.001, C0, k) == -1
        assert i_ring(0.001, C0, k) == 0


def test_scalar_and_vectorised_paths_agree():
    """query() indexes one point, scatter() indexes a whole scan. If the two
    paths disagree, the map is built on one lattice and read on another."""
    rng = np.random.default_rng(SEED + 1)
    xs = rng.uniform(-100.0, 100.0, 10_000)
    for k in _ks("5/10/20/40"):
        vec = i_ring(xs, C0, k)
        assert all(int(v) == i_ring(float(x), C0, k) for x, v in zip(xs, vec))


def test_i_ring_rejects_non_integer_k():
    """Same hard rule as schedule.validate(): integer ratios, or the lattices
    drift apart. Rejected at the index, not only at config load."""
    for bad_k in (2.5, 0, -2):
        with pytest.raises(ValueError):
            i_ring(1.0, C0, bad_k)


# --- ring assignment, math §6 ------------------------------------------------


def test_ring_of_isotropic_boundaries():
    """At v=0 eq. (20) collapses to the Chebyshev norm of (18): L is the first
    ring whose half-width strictly exceeds max(|x|,|y|)."""
    s = load("5/10/20/40")
    assert ring_of(0.0, 0.0, s) == 0
    assert ring_of(9.99, 0.0, s) == 0
    assert ring_of(10.01, 0.0, s) == 1
    assert ring_of(0.0, -24.0, s) == 1
    assert ring_of(30.0, 0.0, s) == 2
    assert ring_of(70.0, 70.0, s) == 3   # square annuli, not circular
    assert ring_of(99.0, 0.0, s) == 3


def test_point_beyond_the_last_ring_is_outside_not_ring_zero():
    """A 101 m return is not in the map. Clamping it to ring 0 would write a
    point past the map edge into a 5 cm cell at the origin."""
    s = load("5/10/20/40")
    assert ring_of(101.0, 0.0, s) == OUTSIDE
    assert ring_of(0.0, 250.0, s) == OUTSIDE
    assert list(ring_of(np.array([5.0, 101.0]), np.array([0.0, 0.0]), s)) == [0, OUTSIDE]


def test_anisotropy_changes_ring_membership_only():
    """Cells stay on the base 5 cm lattice under anisotropic stretch, so
    nesting and alignment are untouched. Master v4 §3.2.

    The claim that needs proving is a negative one: speed changes which ring a
    point belongs to, and changes nothing whatever about its index on any
    lattice. Ring membership is bookkeeping; the lattice is geometry.
    """
    s = load("5/10/20/40")
    rng = np.random.default_rng(SEED)
    xs = rng.uniform(-99.0, 99.0, 5_000)
    ys = rng.uniform(-99.0, 99.0, 5_000)

    r_slow = ring_of(xs, ys, s, 0.0)
    r_fast = ring_of(xs, ys, s, 15.0)
    assert int((r_slow != r_fast).sum()) > 0, "anisotropy moved nothing -- check (20)"

    # whichever ring a point now belongs to, its index on every lattice is
    # unchanged: computed from position alone, speed is never an input
    for k in _ks("5/10/20/40"):
        assert np.array_equal(i_ring(xs, C0, k), i_fine(xs, C0) // k)

    # and the map does not change size with speed -- the same points are in it
    assert np.array_equal(r_slow == OUTSIDE, r_fast == OUTSIDE)

    # The sides are squeezed (coarser sooner), the rear is untouched.
    assert ring_of(0.0, 7.0, s, 15.0) > ring_of(0.0, 7.0, s, 0.0)
    assert ring_of(-19.0, 0.0, s, 15.0) == ring_of(-19.0, 0.0, s, 0.0)

    # Forward is NOT stretched, and that is the containment rule biting, not
    # a bug: ring 0 spans 10 m, so a return at 19 m ahead cannot be filed
    # there however fast the vehicle is going -- the buffer has no cell for
    # it. Under fixed square ring buffers eq. (20) reduces to its lateral
    # half. Recovering the forward half means allocating each ring for its
    # maximum stretch, roughly 1.5x the cells, which is a memory decision.
    assert ring_of(19.0, 0.0, s, 15.0) == ring_of(19.0, 0.0, s, 0.0)


def test_rear_resolution_floor():
    """Never coarser than 20 cm within 50 m behind — closing traffic is exactly
    where coarse cells hurt. Anisotropy comes from the sides, not the back.
    """
    s = load("5/10/20/40")
    floor_m = s.anisotropy.rear_floor_cell_m
    for x in np.arange(-49.5, 0.0, 0.5):
        for v in (0.0, 5.0, 15.0, 30.0):
            ring = ring_of(float(x), 0.0, s, v)
            assert ring != OUTSIDE
            assert s.rings[ring].cell_m <= floor_m


def test_rear_floor_never_pulls_a_point_into_a_ring_too_small_for_it():
    """Math §6.2 states the floor as: x < 0 and |x| < 50. Read literally, a
    point at (-10, -70) qualifies -- behind the vehicle, within 50 m
    longitudinally, 70 m to the side -- and forcing it into ring 2, whose
    buffer spans 50 m, wraps its index toroidally onto the cell at +30 m, on
    the far side of the vehicle.

    Silent, and it corrupts the side of the map the point is not even on.
    """
    s = load("5/10/20/40")
    assert ring_of(-10.0, -70.0, s) == 3   # not clamped to ring 2
    assert ring_of(-10.0, -70.0, s, 15.0) == 3   # nor dropped at speed

    # the floor still applies where it can, i.e. where the point really fits
    assert s.rings[ring_of(-10.0, -30.0, s)].cell_m <= s.anisotropy.rear_floor_cell_m

    # no ring ever receives a point outside its own half-width, at any speed
    rng = np.random.default_rng(SEED + 7)
    xs = rng.uniform(-99.0, 99.0, 20_000)
    ys = rng.uniform(-99.0, 99.0, 20_000)
    for x, y in zip(xs, ys):
        for v in (0.0, 15.0):
            ring = ring_of(x, y, s, v)
            if ring != OUTSIDE:
                assert max(abs(x), abs(y)) < s.rings[ring].half_width_m


def test_hysteresis_bounds_ring_thrash():
    """Math §6.3. Sinusoidal speed across a ring boundary: assert the number of
    split/merge events per cell is bounded.

    Without eq. (21) a cell parked on a boundary changes ring every frame,
    which thrashes the refinement pool and, by §5.4, inflates variance every
    cycle for no physical cause.
    """
    s = load("5/10/20/40")
    # Parked on the R_0 = 10 m boundary, laterally, where the squeeze acts:
    # d ranges over 9.90 .. 10.89 m as speed cycles 0 .. 3 m/s. It crosses
    # R_0 every cycle and never reaches R_0(1+eps) = 11 m, which is exactly
    # the band eq. (21) exists to absorb.
    x, y = 0.0, 9.9
    speeds = 1.5 + 1.5 * np.sin(np.linspace(0.0, 200.0 * np.pi, 1_000))

    naive = [ring_of(x, y, s, float(v)) for v in speeds]
    naive_changes = sum(a != b for a, b in itertools.pairwise(naive))

    ring = ring_of(x, y, s, float(speeds[0]))
    hyst_changes = 0
    for v in speeds[1:]:
        nxt = migrate_ring(x, y, s, ring, float(v))
        hyst_changes += int(nxt != ring)
        ring = nxt

    assert naive_changes > 100, "test point is not actually sitting on a boundary"
    assert hyst_changes <= 4, f"hysteresis let {hyst_changes} transitions through"


# --- toroidal ring buffers, math §2.4 ---------------------------------------


def _fill(soa, seed):
    """Fill every field with non-zero data, so a cleared cell is detectable."""
    rng = np.random.default_rng(seed)
    for name, dt in CELL_FIELDS:
        info = np.iinfo(dt)
        lo, hi = max(info.min, 1), min(info.max, 100)
        soa[name][:] = rng.integers(lo, hi + 1, size=soa[name].size).astype(dt)


def test_toroidal_shift_round_trip_is_bit_exact():
    """Shift by +d then -d returns an identical map. Gate 1.

    Exactly as §2.4(c) states it, with the caveat that section omits: a shift
    CLEARS the newly exposed strip, so it destroys information by design.
    Shifting +d then -d exposes the same slots twice, so the map is identical
    everywhere provided that strip started out empty -- which is the only
    reading under which (c) is literally true. The general case is the test
    below.
    """
    s = load("5/10/20/40")
    zeros = np.zeros((len(s.rings), 2), dtype=np.int64)

    for delta in ((1, 0), (0, 1), (3, 2), (-2, 5)):
        soa = alloc_ring_buffers(s)
        _fill(soa, SEED)

        # empty exactly what this round trip will expose, up front
        probe = alloc_ring_buffers(s)
        _fill(probe, SEED)
        toroidal_shift(probe, s, delta)
        for name, _ in CELL_FIELDS:
            soa[name][probe[name] == 0] = 0
        before = {name: soa[name].copy() for name, _ in CELL_FIELDS}

        toroidal_shift(soa, s, delta)
        toroidal_shift(soa, s, (-delta[0], -delta[1]))

        assert np.array_equal(soa["ring_origin"], zeros)
        for name, _ in CELL_FIELDS:
            assert np.array_equal(soa[name], before[name]), f"{name} lost, d={delta}"


def test_toroidal_shift_only_ever_loses_the_exposed_strip():
    """The general case, on a fully populated map: after +d then -d every cell
    outside the exposed strip is bit-identical and the strip is zeroed.
    Nothing is permuted -- a permutation is what a wrong modulo produces, and
    it looks plausible on a dashboard for a long time."""
    s = load("5/10/20/40")
    soa = alloc_ring_buffers(s)
    _fill(soa, SEED)
    before = {name: soa[name].copy() for name, _ in CELL_FIELDS}

    toroidal_shift(soa, s, (1, 0))
    toroidal_shift(soa, s, (-1, 0))

    changed = soa["ground_height"] != before["ground_height"]
    assert changed.any()
    assert (soa["ground_height"][changed] == 0).all()
    assert int(changed.sum()) <= 6_700  # the strip, and nothing beyond it


def test_toroidal_shift_is_o_perimeter_not_o_area():
    """Ring 3 clears 2N = 1,000 cells per step, not N^2 = 250,000 -- the
    difference between a sub-millisecond shift and a 40 ms stall. Measured,
    not asserted in a comment."""
    s = load("5/10/20/40")
    soa = alloc_ring_buffers(s)
    total = buffer_cells(s)

    cleared = toroidal_shift(soa, s, (1, 0))
    coarsest = s.k(len(s.rings) - 1)
    analytic = sum(ring_extent(s, L) * (coarsest // s.k(L)) for L in range(len(s.rings)))
    assert cleared == analytic == 6_700
    assert cleared < total / 100

    n3 = ring_extent(s, len(s.rings) - 1)     # the numbers quoted in §2.4
    assert n3 == 500
    assert 2 * n3 == 1_000
    assert n3 * n3 == 250_000


def test_finer_rings_shift_by_whole_cells():
    """The §2.4 constraint: the origin moves in whole COARSEST cells (40 cm),
    so every finer ring moves a whole number of its own cells -- 8 for ring 0
    at 5 cm. A fractional step would move every ring boundary by part of a
    cell and force a resample, which is the data loss the brief warns about."""
    s = load("5/10/20/40")
    soa = alloc_ring_buffers(s)
    toroidal_shift(soa, s, (1, 0))
    assert list(soa["ring_origin"][:, 0]) == [8, 4, 2, 1]
    assert list(soa["ring_origin"][:, 1]) == [0, 0, 0, 0]


def test_allocated_cells_exceed_the_annulus_count():
    """Records a real discrepancy rather than papering over it.

    schedule.total_cells counts square ANNULI (eq. 19) and is where 8.94 MB
    comes from. A toroidal ring buffer has to store the full N_L x N_L square
    per ring, because the hole is centred on the vehicle and therefore travels
    through the buffer as it drives. Allocation is 910,000 cells, not 745,000
    -- 10.92 MB, and 17.6x against a uniform 5 cm grid rather than 21.5x.

    Shrestha owns the bound. This test pins the arithmetic so the decision
    gets made with the real number in front of it.
    """
    s = load("5/10/20/40")
    assert buffer_cells(s) == 910_000
    assert s.total_cells == 745_000

    uniform = (200.0 / C0) ** 2
    assert uniform / buffer_cells(s) == pytest.approx(17.6, abs=0.1)
    assert uniform / s.total_cells == pytest.approx(21.5, abs=0.1)
