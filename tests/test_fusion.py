"""Fusion, occupancy and visibility. Math §3, §10. [Aakash]"""

import itertools

import numpy as np
import pytest
from vrgrid.cell import (
    FLAG_BLIND,
    OCC_FREE,
    OCC_OCCUPIED,
    OCC_UNKNOWN,
    alloc_soa,
)
from vrgrid.gpu.kernels import (
    CEILING_NONE,
    WEIGHT_SCALE,
    CellAggregate,
    measurement_variance_cm2,
    quantise_weight,
    scatter_sorted,
)
from vrgrid.grid.fusion import (
    CLASS_BITS,
    CLASS_MAX,
    COUNTER_MAX,
    boyer_moore_update,
    fuse,
    initialise,
    is_blind,
    mark_blind,
    occupancy_state,
    pack_class,
    unpack_class,
)
from vrgrid.grid.quantise import (
    CODE_RATIO,
    SIGMA2_MAX_CM2,
    SIGMA2_MIN_CM2,
    dequantise_variance_cm2,
    quantise_variance_cm2,
)
from vrgrid.grid.schedule import load, load_thresholds
from vrgrid.grid.splitmerge import inflate

N_CELLS = 64


def _grid(n=N_CELLS):
    """A freshly allocated grid, in the state allocate() leaves it in."""
    soa = alloc_soa(n)
    initialise(soa)
    return soa


def _agg(cells, z_cm, w_q, n=None, ceiling=None, refl=None, class_id=None):
    """A CellAggregate by hand, so a test can state one frame's evidence
    without going through a scan."""
    k = len(cells)
    z = np.asarray(z_cm, dtype=np.int64)
    w = np.asarray(w_q, dtype=np.int64)
    return CellAggregate(
        np.asarray(cells, dtype=np.int64),
        (w * z).astype(np.int64),
        w.astype(np.int64),
        np.asarray(n if n is not None else [1] * k, dtype=np.int32),
        np.asarray(ceiling if ceiling is not None else [CEILING_NONE] * k, dtype=np.int16),
        np.asarray(refl if refl is not None else [0] * k, dtype=np.int32),
        np.asarray(class_id if class_id is not None else [0] * k, dtype=np.uint8),
    )


# --- §3.3: the filter --------------------------------------------------------


def test_first_measurement_initialises_rather_than_filters():
    """A cell with obs_count 0 has no prior. It must take the measurement, not
    filter against whatever the zeroed bytes happen to decode to."""
    soa = _grid()
    fuse(soa, _agg([7], [123], [quantise_weight(measurement_variance_cm2(10.0))]))

    assert soa["ground_height"][7] == 123
    assert soa["obs_count"][7] == 1
    assert soa["frames_since_seen"][7] == 0
    # and it did not touch anybody else
    assert soa["obs_count"][6] == 0 and soa["ground_height"][6] == 0


def test_variance_decreases_under_repeated_consistent_measurements():
    """§3.4's unit test, first half. Twenty consistent returns must make the
    cell more certain, monotonically -- never less certain on any single step,
    which is what the codec's round-toward-larger rule is there to guarantee."""
    soa = _grid()
    w = quantise_weight(measurement_variance_cm2(20.0))

    seen = []
    for _ in range(20):
        fuse(soa, _agg([3], [50], [w]))
        seen.append(dequantise_variance_cm2(soa["height_variance"][3]))

    assert all(b <= a for a, b in itertools.pairwise(seen)), "variance went back up"
    assert seen[-1] < seen[0]
    assert soa["ground_height"][3] == 50


def test_process_noise_stops_the_filter_locking():
    """§3.3's second half: without `sigma2 <- sigma2 + q dt` the variance
    collapses toward zero, the gain goes to zero with it, and the cell stops
    responding to evidence. The map then cannot register a change that really
    happened -- a kerb rebuilt, a car parked -- which is a failure a demo shows
    and a unit test usually does not.

    Run a cell to steady state, then move the world 30 cm and assert it still
    follows.
    """
    soa = _grid()
    w = quantise_weight(measurement_variance_cm2(8.0))
    for _ in range(200):
        fuse(soa, _agg([1], [0], [w]))

    settled = dequantise_variance_cm2(soa["height_variance"][1])
    assert settled >= SIGMA2_MIN_CM2, "variance fell below the storage floor"

    for _ in range(30):
        fuse(soa, _agg([1], [30], [w]))
    assert soa["ground_height"][1] > 25, "cell locked and ignored the new evidence"


def test_a_far_return_moves_the_cell_less_than_a_near_one():
    """§3.1, the reason this is a Kalman filter and not a running mean. Same
    prior, same measured height; the only difference is where the return came
    from. 5 m must dominate 80 m."""
    near, far = _grid(), _grid()
    prior_w = quantise_weight(measurement_variance_cm2(20.0))
    for soa in (near, far):
        fuse(soa, _agg([0], [0], [prior_w]))

    fuse(near, _agg([0], [100], [quantise_weight(measurement_variance_cm2(5.0))]))
    fuse(far, _agg([0], [100], [quantise_weight(measurement_variance_cm2(80.0))]))

    assert near["ground_height"][0] > far["ground_height"][0]
    # §3.2's numbers: sigma_z is ~8.7 cm at 50 m and ~17.5 cm at 100 m, so an
    # 80 m return is weak evidence -- but it is not zero evidence.
    assert far["ground_height"][0] > 0


def test_fuse_is_bit_identical_run_to_run():
    """§3.4. The filter is float64; what §3.4 forbids is nondeterministic
    SUMMATION order, and every sum has already happened in integers by the time
    an aggregate exists. This asserts the float half does not reintroduce it."""
    rng = np.random.default_rng(20260829)
    cells = np.unique(rng.integers(0, N_CELLS, 40))
    z = rng.integers(-150, 400, cells.size)
    w = quantise_weight(measurement_variance_cm2(rng.uniform(3.0, 90.0, cells.size)))

    outs = []
    for _ in range(2):
        soa = _grid()
        for _ in range(5):
            fuse(soa, _agg(cells, z, w))
        outs.append({k: v.copy() for k, v in soa.items()})

    for name in outs[0]:
        assert np.array_equal(outs[0][name], outs[1][name]), f"{name} differs"


def test_fuse_rejects_the_hole_sentinel():
    """annulus_index() returns -1 for the ring's hole. A -1 reaching the map
    would be written into cell 0 by numpy's negative indexing and pile the far
    field onto the origin -- plausible-looking, and wrong."""
    soa = _grid()
    with pytest.raises(ValueError, match="negative cell index"):
        fuse(soa, _agg([-1], [10], [1000]))


# --- the height-variance codec (grid/quantise.py) ----------------------------


def test_zeroed_memory_decodes_to_no_information():
    """Code 0 must be MAXIMUM variance.

    `allocate()` zeros every field and the ego-motion shift zeros each newly
    exposed strip, so this is the state of every cell that has never been
    looked at and every cell that just scrolled into view. If 0 decoded to the
    smallest variance the map would boot claiming millimetre certainty about
    ground it has never seen, and -- worse -- the first Kalman gain would be
    ~0, so the cell would never recover.
    """
    assert dequantise_variance_cm2(0) == pytest.approx(SIGMA2_MAX_CM2)
    assert dequantise_variance_cm2(255) == pytest.approx(SIGMA2_MIN_CM2)
    assert dequantise_variance_cm2(0) > dequantise_variance_cm2(255)


def test_codec_never_understates_uncertainty():
    """Rounding is toward the larger variance, so a stored value is always at
    least the true one. An overstated variance costs sharpness; an understated
    one makes the filter ignore real evidence."""
    rng = np.random.default_rng(7)
    v = np.exp(rng.uniform(np.log(SIGMA2_MIN_CM2), np.log(SIGMA2_MAX_CM2), 5000))
    stored = dequantise_variance_cm2(quantise_variance_cm2(v))
    assert np.all(stored >= v - 1e-9)
    assert np.all(stored <= v * CODE_RATIO + 1e-9)


def test_codec_is_monotone_so_the_filter_cannot_round_backwards():
    """A decreasing variance must never come back with a larger code than it
    started with -- otherwise §3.4's monotonicity test fails through the
    storage rather than through the filter."""
    codes = quantise_variance_cm2(np.geomspace(SIGMA2_MAX_CM2, SIGMA2_MIN_CM2, 3000))
    assert np.all(np.diff(codes.astype(np.int32)) >= 0)


def test_theorem_1_is_not_always_visible_through_the_codec():
    """One code is a factor of 1.064, so §5's Theorem 1 -- split strictly
    inflates variance -- is only OBSERVABLE in the stored map when the slope
    term clears 6.4% of the parent variance.

    This is not a defect in §5 and not one in the codec: the theorem is about
    reals, and one byte cannot hold reals. It is a limit on what the map can
    be asked to demonstrate, and it is better measured than discovered.

    Measured, for a cell settled to sigma = 3 cm on a 20% slope:

        ring 1  10 ->  5 cm   +2.1%   invisible, the codec swallows it
        ring 2  20 -> 10 cm   +8.3%   visible
        ring 3  40 -> 20 cm  +33.3%   visible

    So the theorem disappears exactly where the cells are finest — which is
    where it matters least, since a 5 cm child of a 10 cm parent is barely a
    coarser claim. Worth knowing before someone builds a demo slide on
    "watch the variance rise when we refine" and picks ring 1 to show it on.
    """
    s = load("5/10/20/40")
    settled_cm2 = 9.0                                     # sigma = 3 cm
    visible = {}
    for ring in (1, 2, 3):
        c_p, c_c = s.rings[ring].cell_m, s.rings[ring - 1].cell_m
        inflated = inflate(settled_cm2 * 1e-4, 0.2, c_p, c_c) * 1e4   # m^2 -> cm^2
        assert inflated > settled_cm2                     # Theorem 1, in reals
        visible[ring] = (quantise_variance_cm2(inflated)
                         != quantise_variance_cm2(settled_cm2))

    assert visible == {1: False, 2: True, 3: True}, "codec resolution has changed"

    # the line is CODE_RATIO, not anything to do with §5
    c_p, c_c = s.rings[1].cell_m, s.rings[0].cell_m
    gentle = inflate(settled_cm2 * 1e-4, 0.2, c_p, c_c) * 1e4
    steep = inflate(settled_cm2 * 1e-4, 1.0, c_p, c_c) * 1e4
    assert gentle / settled_cm2 < CODE_RATIO < steep / settled_cm2
    assert quantise_variance_cm2(steep) != quantise_variance_cm2(settled_cm2)


# --- §10.2: class fusion in one byte -----------------------------------------


def _unbounded_majority(obs):
    """Textbook Boyer-Moore, counter unbounded. The reference the one-byte
    version is supposed to agree with, and the only version the theorem in
    §10.2 actually holds for."""
    cand, cnt = None, 0
    for y in obs:
        if cnt == 0:
            cand, cnt = y, 1
        elif y == cand:
            cnt += 1
        else:
            cnt -= 1
    return cand


def test_boyer_moore_majority_in_one_byte():
    """The majority guarantee, over sequences that stay inside the counter.

    ⚑ Read `test_saturation_is_what_bounds_the_guarantee` before changing the
      bound here. `n <= COUNTER_MAX` is not an arbitrary test parameter -- it
      is the condition under which the one-byte version and the textbook
      version are the same algorithm, because a counter that never reaches its
      cap never discards anything.
    """
    rng = np.random.default_rng(0)
    for _ in range(400):
        n = int(rng.integers(3, COUNTER_MAX + 1))
        true_majority = int(rng.integers(0, CLASS_MAX + 1))
        obs = [true_majority] * (n // 2 + 1)
        obs += list(rng.integers(0, CLASS_MAX + 1, n - len(obs)))
        rng.shuffle(obs)

        packed = np.zeros(1, dtype=np.uint8)
        for y in obs:
            packed = boyer_moore_update(packed, np.array([y], dtype=np.uint8))

        cand, _ = unpack_class(packed)
        assert int(cand[0]) == true_majority
        assert int(cand[0]) == _unbounded_majority(obs)


def test_boyer_moore_matches_the_textbook_until_the_counter_saturates():
    """The real equivalence, stated as it holds: the one-byte version IS
    textbook Boyer-Moore on any sequence whose running excess stays within the
    counter. Long random sequences included -- most never reach the cap."""
    rng = np.random.default_rng(1)
    agreed = 0
    for _ in range(400):
        n = int(rng.integers(4, 60))
        obs = list(rng.integers(0, CLASS_MAX + 1, n))

        cand, cnt = 0, 0
        saturated = False
        for y in obs:
            if cnt == 0:
                cand, cnt = int(y), 1
            elif int(y) == cand:
                if cnt == COUNTER_MAX:
                    saturated = True
                cnt = min(cnt + 1, COUNTER_MAX)
            else:
                cnt -= 1
        if saturated:
            continue

        packed = np.zeros(1, dtype=np.uint8)
        for y in obs:
            packed = boyer_moore_update(packed, np.array([y], dtype=np.uint8))
        assert int(unpack_class(packed)[0][0]) == _unbounded_majority(obs)
        agreed += 1

    assert agreed > 200, "the sample stopped exercising the unsaturated path"


def test_saturation_is_what_bounds_the_guarantee():
    """⚑ §10.2 says the majority guarantee "does not depend on where the
    counter saturates". That is not true, and it was not true at 4/4 either.

    Boyer-Moore's proof assumes an unbounded counter. A saturating counter
    throws away exactly the evidence the proof relies on: once the counter is
    pinned at C, further sightings of the majority class are not recorded, so
    C+1 contradicting observations can unseat a class that holds a genuine
    strict majority.

    Demonstrated at BOTH caps, which is the point -- 5/3 did not introduce
    this, it halved the headroom:

        cap 15 (4/4):  1 x 32 then 2 x 31  -> candidate 2, and 1 had 32 of 63
        cap  7 (5/3):  1 x 16 then 2 x 15  -> candidate 2, and 1 had 16 of 31

    So what the cell actually provides is not a proof but a **time constant**:
    a cell changes its mind after more than C net contradicting observations.
    C = 7 makes the map adapt about twice as fast as C = 15, which for a
    rolling local map with dynamics in it is arguably the better default --
    but it is a tuning claim, not a theorem, and the report must not say
    "guaranteed majority" without the condition attached.

    This test exists so nobody restores the stronger sentence.
    """
    def run(obs, cap):
        cand, cnt = None, 0
        for y in obs:
            if cnt == 0:
                cand, cnt = y, 1
            elif y == cand:
                cnt = min(cnt + 1, cap)
            else:
                cnt -= 1
        return cand

    for cap, run_len in ((15, 32), (COUNTER_MAX, 2 * (COUNTER_MAX + 1))):
        obs = [1] * run_len + [2] * (run_len - 1)
        assert obs.count(1) > obs.count(2), "1 must hold a strict majority"
        assert _unbounded_majority(obs) == 1, "the textbook version finds it"
        assert run(obs, cap) == 2, (
            f"cap {cap} no longer loses the majority -- if this is a real "
            "improvement, update §10.2 and the report sentence with it"
        )

    # and the shipped implementation is the cap-7 case of exactly that
    obs = [1] * (2 * (COUNTER_MAX + 1)) + [2] * (2 * (COUNTER_MAX + 1) - 1)
    packed = np.zeros(1, dtype=np.uint8)
    for y in obs:
        packed = boyer_moore_update(packed, np.array([y], dtype=np.uint8))
    assert int(unpack_class(packed)[0][0]) == 2


def test_class_byte_packs_and_survives_a_round_trip():
    """Every (candidate, counter) the 5/3 byte can hold. Bounds come from the
    constants, so the day the split moves again this covers the new space
    instead of silently testing a corner of it."""
    for c in range(CLASS_MAX + 1):
        for k in range(COUNTER_MAX + 1):
            cand, cnt = unpack_class(pack_class(c, k))
            assert (int(cand), int(cnt)) == (c, k)


def test_counter_saturates_without_losing_the_candidate():
    """min(counter + 1, COUNTER_MAX). A cell seen 300 times as road must not
    overflow the counter into the candidate -- which would silently relabel it.

    The cap is 7 since the byte was re-split 5/3 (it was 15). Asserted against
    the constant rather than a literal, because the literal is what made this
    a two-line edit instead of a no-op when the split landed.
    """
    packed = np.zeros(1, dtype=np.uint8)
    for _ in range(300):
        packed = boyer_moore_update(packed, np.array([5], dtype=np.uint8))
    cand, cnt = unpack_class(packed)
    assert int(cand[0]) == 5
    assert int(cnt[0]) == COUNTER_MAX


def test_the_whole_label_set_fits():
    """⚑ Was `test_nineteen_classes_do_not_fit`, and it is the same theorem
    read from the other side. Gate 3, item 3, ratified 1 Sep.

    A 4-bit candidate held 16 ids and the label set is 20 (0-19). That
    shortfall reached three independent places, each of which read as working:
    the semantic gate could never match `pole` or `traffic-sign`, the
    traversability predicate could never store `terrain`, and the first real
    frame raised rather than degrading -- around which two disagreeing
    stand-ins had grown, `% 16` in the eval harness and `clip(0, 15)` in the
    frame loop.

    5 bits holds the set with room to 31. Anything above 31 is still refused
    loudly rather than wrapped, because `% 32` would relabel class 32 as 0 and
    0 is `unlabeled` -- a chunk of the map quietly becoming unlabelled ground
    is plausible-looking and invisible without the reference map.
    """
    from vrgrid.grid.traversability import CLASS_IDS

    packed = np.zeros(1, dtype=np.uint8)

    # every id the project actually uses, including the three that did not fit
    for name, cid in CLASS_IDS.items():
        assert cid <= CLASS_MAX, f"{name} ({cid}) does not fit in {CLASS_BITS} bits"
        out = boyer_moore_update(packed, np.array([cid], dtype=np.uint8))
        cand, cnt = unpack_class(out)
        assert int(cand[0]) == cid, f"{name} did not survive a fuse"
        assert int(cnt[0]) == 1

    # the three that used to be silently corrupted, named explicitly
    for name in ("terrain", "pole", "traffic-sign"):
        assert CLASS_IDS[name] > 15, "this test is no longer testing anything"
        assert CLASS_IDS[name] <= CLASS_MAX

    # and the seam is still loud
    with pytest.raises(ValueError, match="does not fit"):
        boyer_moore_update(packed, np.array([32], dtype=np.uint8))
    with pytest.raises(ValueError, match="does not fit"):
        boyer_moore_update(packed, np.array([252], dtype=np.uint8))   # a RAW id


def test_fuse_votes_the_class_it_was_given():
    soa = _grid()
    for _ in range(5):
        fuse(soa, _agg([2], [0], [1000], class_id=[9]))
    cand, cnt = unpack_class(soa["semantic_class"][2:3])
    assert int(cand[0]) == 9
    assert int(cnt[0]) == 5


# --- §10.1: three-state occupancy --------------------------------------------


def test_unknown_is_decided_by_observation_count():
    """Three occupancy states. Unknown is NOT log-odds near zero — a cell
    observed twice with conflicting evidence is not the same as a cell never
    observed, and a planner must treat them differently."""
    soa = _grid()
    th = load_thresholds()

    # never observed: log_odds is 0, and so is a cell whose hits and misses
    # cancelled. Only the count separates them.
    assert occupancy_state(soa, th)[0] == OCC_UNKNOWN

    soa["obs_count"][0] = 4
    soa["log_odds"][0] = 0
    assert occupancy_state(soa, th)[0] == OCC_FREE, "conflicting evidence is not unknown"

    soa["log_odds"][0] = 20
    assert occupancy_state(soa, th)[0] == OCC_OCCUPIED


def test_blind_cone_is_unknown_never_free():
    """3.74 m radius, 11% of Ring 0, unobservable in any single frame."""
    th = load_thresholds()
    r = th["sensor"]["blind_cone_m"]
    assert r == pytest.approx(3.74)

    assert is_blind(0.0, 0.0, th)
    assert is_blind(2.0, 2.0, th)              # 2.83 m
    assert not is_blind(3.0, 3.0, th)          # 4.24 m
    assert not is_blind(r + 0.01, 0.0, th)

    # and the flag outranks the evidence: a blind cell with a pile of
    # free-space observations still reports UNKNOWN, never FREE.
    soa = _grid()
    mark_blind(soa, [0])
    soa["obs_count"][0] = 200
    soa["log_odds"][0] = -60
    assert soa["flags"][0] & FLAG_BLIND
    assert occupancy_state(soa, th)[0] == OCC_UNKNOWN


def test_a_return_is_evidence_of_occupancy_and_the_clamp_holds():
    """§10.1's clamp. An unclamped cell that has seen 500 free observations
    needs 500 occupied ones to change its mind, which is why unclamped maps
    fail to register a newly-appeared obstacle."""
    soa = _grid()
    th = load_thresholds()
    lo, hi = th["occupancy"]["log_odds_clamp"]
    for _ in range(400):
        fuse(soa, _agg([0], [0], [1000]), th)
    assert soa["log_odds"][0] == hi
    assert lo <= soa["log_odds"][0] <= hi


# --- the ceiling sentinel ----------------------------------------------------


def test_a_zeroed_grid_would_report_the_whole_world_untraversable():
    """⚑ `allocate()` zeros every field, so ceiling_height starts at 0 — "a
    solid thing at the ground datum", not "nothing overhead seen", whose
    sentinel is CEILING_NONE. Left alone, `ceiling - ground < h_vehicle` holds
    for every cell in the map and TRAV_CLEARANCE blocks the entire world,
    permanently, because a minimum never raises a ceiling back up.

    initialise() is the fix from this side. The real one is in allocate() and
    in the shift's strip clear, both of which are gpu/.
    """
    th = load_thresholds()
    h_vehicle_cm = th["traversability"]["h_vehicle_m"] * 100

    raw = alloc_soa(N_CELLS)                       # exactly what allocate() gives
    clearance = raw["ceiling_height"].astype(np.int32) - raw["ground_height"]
    assert np.all(clearance < h_vehicle_cm), "the premise of this test has changed"

    initialise(raw)
    clearance = raw["ceiling_height"].astype(np.int32) - raw["ground_height"]
    assert np.all(clearance > h_vehicle_cm)


def test_ceiling_keeps_the_lowest_thing_overhead():
    soa = _grid()
    fuse(soa, _agg([4], [0], [1000], ceiling=[400]))
    assert soa["ceiling_height"][4] == 400
    fuse(soa, _agg([4], [0], [1000], ceiling=[250]))
    assert soa["ceiling_height"][4] == 250
    fuse(soa, _agg([4], [0], [1000], ceiling=[600]))
    assert soa["ceiling_height"][4] == 250, "a higher ceiling must not raise it"


# --- the seam with gpu/kernels.py --------------------------------------------


def test_ground_height_must_not_average_in_the_canopy():
    """⚑ A cell holding a road return at 0 cm and a canopy return at 200 cm
    must not report a ground height of 100 cm — a metre of tree averaged into
    the road. The resulting map looks entirely plausible, which is what makes
    it expensive to find later.

    §3 is about the height of the GROUND, so the kernel masks the height sums
    by `is_ground`; the canopy return still supplies the ceiling. Was an xfail
    until the mask landed in `gpu/kernels.py`.
    """
    agg = scatter_sorted(
        idx=np.array([0, 0]),
        z_cm=np.array([0, 200], dtype=np.int16),
        w_q=np.array([1000, 1000], dtype=np.int32),
        refl=np.array([10, 10], dtype=np.int32),
        class_id=np.array([1, 2], dtype=np.uint8),
        is_ground=np.array([True, False]),
    )
    assert agg.ceiling_cm[0] == 200
    assert agg.mean_height_cm()[0] == 0
    assert agg.w_sum[0] == 1000              # the road return alone
    assert agg.n[0] == 2                     # but both are still observations


def test_measurement_variance_matches_the_weights_fuse_infers():
    """fuse() recovers sigma2_z as WEIGHT_SCALE/w_sum rather than recomputing
    it. That identity is the contract between the two files, so it is asserted
    rather than assumed — within the weight's own quantisation step, which is
    where the interesting part is.

    ⚑ WEIGHT_SCALE = 1024 leaves the FAR field almost no resolution. A single
      return's weight is rint(1024/sigma2_cm2), so:

          r =   5 m   sigma_z =  1.1 cm   w = 826   error  0.06%
          r =  25 m   sigma_z =  4.4 cm   w =  54   error  0.50%
          r =  50 m   sigma_z =  8.7 cm   w =  13   error  3.43%
          r = 100 m   sigma_z = 17.5 cm   w =   3   error 12.05%

      Three levels at 100 m. The kernel's comment reads "~3 decades of dynamic
      range ... with the smallest weight still >= 1", which is true and is
      about the wrong end: the constraint that bites is resolution at the
      coarse end, not underflow. It matters least where it is worst — a cell
      at 100 m is filled by ego-motion over many frames (§1.3), and weights
      sum, so w_sum recovers — but a single far return's variance is off by
      12% and that is worth knowing before it is inferred from a plot.
    """
    for r in (5.0, 25.0, 50.0, 100.0):
        var = measurement_variance_cm2(r)
        w = float(quantise_weight(var))
        implied = WEIGHT_SCALE / w
        assert abs(implied - var) / var <= 0.5 / w + 1e-9

    assert int(quantise_weight(measurement_variance_cm2(100.0))) == 3


def test_visibility_cleanup_is_not_declared_here():
    """§10.4 is `gpu.visibility.visibility_cleanup`, and only there.

    This file used to declare a second `visibility_cleanup` — same name, same
    section, same guard in the docstring, a different signature and a
    `NotImplementedError`. Both were reachable, and the stub was the one a
    reader looking for §10.4 in `src/grid` found first, which is how the ghost
    gate came to be reported as unbuilt after it had been wired.

    The guard itself is tested where the code is:
    `tests/test_visibility.py::test_a_cell_with_a_return_now_is_never_cleared`.
    This asserts only that the duplicate has not come back.
    """
    import vrgrid.grid.fusion as fusion_mod
    from vrgrid.gpu.visibility import visibility_cleanup

    assert not hasattr(fusion_mod, "visibility_cleanup"), (
        "grid/fusion.py has re-declared visibility_cleanup — §10.4 has one "
        "implementation, in gpu/visibility.py"
    )
    assert callable(visibility_cleanup)


def test_a_cell_with_only_non_ground_returns_keeps_its_height():
    """The other half of the canopy fix, and the half that bites hardest.

    Masking the height sums by `is_ground` leaves a wall, a car flank or a
    tree trunk with `w_sum == 0`, and `mean_height_cm()` returns 0 there for
    want of anything better. 0 is not a neutral value: in the vehicle frame
    the road sits near -173 cm, so writing it would stand every wall in the
    scene 1.7 m above the road — with `meas_var` of WEIGHT_SCALE/1 = 1024 cm²
    behind it, which is confident enough to hold against the next few real
    ground returns. `fuse()` must leave the height alone instead.
    """
    soa = _grid()
    fuse(soa, _agg([5], [-173], [4000]))            # a real road return first
    settled_mu = int(soa["ground_height"][5])
    settled_var = dequantise_variance_cm2(soa["height_variance"][5])
    assert settled_mu == -173

    wall = _agg([5], [0], [0], n=[8], ceiling=[120])   # eight returns, none ground
    assert not wall.has_ground_evidence()[0]
    fuse(soa, wall)

    assert soa["ground_height"][5] == settled_mu, "the wall was averaged into the road"
    # The cell ages rather than sharpening: it gets the process noise every
    # updated cell gets, and no measurement to pay for a smaller variance.
    assert dequantise_variance_cm2(soa["height_variance"][5]) >= settled_var, \
        "certainty grew on no evidence"


def test_a_wall_seen_before_any_ground_does_not_plant_itself_at_the_datum():
    """The first-observation path is the dangerous one: `first` would take the
    measurement outright rather than filtering it, so a cell whose first-ever
    return is a wall would be initialised to 0 cm at high confidence. The
    ground guard has to override `first`, not the other way round."""
    soa = _grid()
    assert soa["obs_count"][9] == 0

    fuse(soa, _agg([9], [0], [0], n=[4], ceiling=[300]))

    assert soa["ground_height"][9] == 0        # unchanged, because never measured
    assert soa["height_variance"][9] == 0      # code 0 is MAXIMUM variance
    assert soa["obs_count"][9] == 4            # but it WAS observed
    assert soa["ceiling_height"][9] == 300     # and the ceiling did land


def test_the_rest_of_the_cell_still_updates_without_ground_evidence():
    """A cell with no ground return is not an unobserved cell. Occupancy,
    class, reflectivity, the ceiling and the counters all take evidence from
    every return; only the height does not."""
    soa = _grid()
    before_odds = int(soa["log_odds"][3])

    fuse(soa, _agg([3], [0], [0], n=[6], ceiling=[250], refl=[600], class_id=[2]))

    assert soa["log_odds"][3] > before_odds
    assert soa["ceiling_height"][3] == 250
    assert soa["reflectivity"][3] == 100       # 600 // 6
    assert soa["frames_since_seen"][3] == 0
    assert soa["obs_count"][3] == 6
