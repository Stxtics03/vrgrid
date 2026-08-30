"""Plan regret. Math §8. [Aakash]

The research claim, so these get the same treatment as the §4-5 theorems:
non-negativity is proved by construction and asserted on random input, and
the two ways the metric can be quietly wrong -- scoring on the wrong map, and
a detour that costs the same but goes somewhere else -- each get a test that
fails if the guard is removed.
"""

import numpy as np
import pytest
from vrgrid.eval.harness import build_gridmap, run_sequence
from vrgrid.eval.plan_regret import (
    BLOCKED,
    CostMap,
    corridor,
    costmap_from_gridmap,
    costmap_from_reference,
    dijkstra,
    path_cost,
    plan,
    regret,
    weights,
)
from vrgrid.eval.reference_map import build_from_scans
from vrgrid.eval.synthetic import read_sequence, write_sequence
from vrgrid.grid.schedule import load, load_thresholds

NX = NY = 40
CELL = 0.25
START, GOAL = (2, 10), (37, 10)


def _map(cost, unknown=None):
    return CostMap(CELL, 0.0, 0.0, cost,
                   np.zeros(cost.shape, bool) if unknown is None else unknown)


def _wall(gap_j=10, gap=3, wall_to=32):
    """A wall across most of the lattice with a gap in it, and a way round the
    end beyond `wall_to` -- so losing the gap forces a longer route rather than
    making the problem infeasible. Both outcomes matter and they are not the
    same outcome; see `blocked_on_reference`."""
    cost = np.ones((NX, NY))
    cost[NX // 2, :wall_to] = BLOCKED
    if gap:
        cost[NX // 2, gap_j:gap_j + gap] = 1.0
    return _map(cost)


def _coarsen(costmap, factor):
    """Conservative block-reduce: a coarse cell is blocked if ANY fine cell in
    it is blocked, and takes the worst weight otherwise.

    This is what coarsening actually does to an obstacle field -- and what
    §7.2's pyramid does deliberately -- so it models the schedule's effect on
    the plan without needing to run a whole sequence to produce it.
    """
    n = costmap.shape[0] // factor * factor
    block = costmap.cost[:n, :n].reshape(n // factor, factor, n // factor, factor)
    coarse = block.max(axis=(1, 3))
    out = np.repeat(np.repeat(coarse, factor, axis=0), factor, axis=1)
    full = costmap.cost.copy()
    full[:n, :n] = out
    return _map(full, costmap.unknown)


# --- eq. (23) ----------------------------------------------------------------


@pytest.mark.theorem
def test_regret_is_never_negative():
    """R(S) >= 0 by construction: pi* minimises J_{M*}, so nothing scored on
    M* can beat it. Asserted on random maps because a negative value is the
    signature of the two costmaps being on different lattices, or of a planner
    that is not actually optimal -- both of which look like a bug in the
    metric rather than in the setup.
    """
    rng = np.random.default_rng(20260829)
    for _ in range(60):
        cost = rng.uniform(1.0, 5.0, (NX, NY))
        cost[rng.random((NX, NY)) < 0.15] = BLOCKED
        cost[START] = cost[GOAL] = 1.0
        star = _map(cost)

        mine = _map(np.where(np.isfinite(cost),
                             cost * rng.uniform(0.5, 2.0, cost.shape), BLOCKED))
        out = regret(star, mine, START, GOAL)
        if out.found:
            assert out.regret >= -1e-9, f"negative regret {out.regret}"


@pytest.mark.theorem
def test_identical_maps_have_exactly_zero_regret():
    """The other end of the same statement, and it must be exact rather than
    small: the money plot's headline is "below 8.9 MB the plan is unchanged --
    regret is exactly zero", and a metric with float noise in it cannot say
    'exactly'."""
    m = _wall()
    out = regret(m, _map(m.cost.copy()), START, GOAL)
    assert out.regret == 0.0
    assert out.frechet_m == 0.0


def test_a_gap_narrower_than_the_cell_costs_the_plan():
    """§8.3's specified unit test: a synthetic map with a narrow gap, R(S) = 0
    for schedules fine enough to resolve it and R(S) > 0 for schedules coarser
    than the gap width.

    This is the money plot in miniature -- the knee is exactly the coarsening
    at which the gap stops being resolvable -- and it is the clearest statement
    of what the whole project is measuring: not how wrong the heights are, but
    whether the vehicle still goes the same way.
    """
    star = _wall(gap=3)                       # a 75 cm gap at 25 cm cells

    fine = regret(star, _coarsen(star, 1), START, GOAL)
    assert fine.regret == 0.0, "an uncoarsened map changed the plan"

    mid = regret(star, _coarsen(star, 2), START, GOAL)
    assert mid.regret == 0.0, "a 50 cm cell should still resolve a 75 cm gap"

    coarse = regret(star, _coarsen(star, 8), START, GOAL)
    assert coarse.found
    assert coarse.regret > 0.0, "a 2 m cell swallowed a 75 cm gap and cost nothing"
    assert coarse.frechet_m > 1.0, "the plan changed but went nowhere different"


@pytest.mark.theorem
def test_scoring_on_the_compressed_map_hides_the_damage():
    """⚑ The critical detail of §8.1, as a negative control.

    A coarsened map that has averaged a rough patch into smooth ground will
    plan straight across it and report that plan as CHEAP -- on its own map,
    which no longer knows the patch is there. Scored on M*, where the patch
    still is, the same path is expensive. The wrong metric is not merely noisy:
    it is smallest exactly when the map is worst, which is the failure mode
    that would survive every other test in this repo.
    """
    truth = np.ones((NX, NY))
    truth[NX // 2 - 1:NX // 2 + 2, :] = 30.0      # a band that really hurts
    truth[NX // 2 - 1:NX // 2 + 2, 28:31] = 1.0   # one cheap gate through it
    star = _map(truth)

    blind = _map(np.ones((NX, NY)))               # coarsening lost the band

    out = regret(star, blind, START, GOAL)
    assert out.found

    right = out.regret
    wrong = out.self_scored_cost - out.reference_cost
    assert right > 1.0, "the correct metric did not notice the lost band"
    assert wrong < right - 1.0, "the wrong metric was not fooled -- rebuild the case"
    assert wrong < 0.0, (
        "scoring on its own map made the coarsened plan look CHEAPER than the "
        "optimum, which is the self-consistency trap §8.1 warns about"
    )


def test_frechet_catches_a_detour_that_costs_the_same():
    """§8.1 asks for the Frechet distance alongside R(S) precisely for this:
    two routes around opposite sides of an obstacle can cost the same to
    within rounding and are not the same decision. Cost alone scores it zero.
    """
    # A wall with two gaps placed symmetrically about the start/goal line, so
    # the two routes cost exactly the same and differ only in where they go.
    cost = np.ones((NX, NY))
    cost[NX // 2, :] = BLOCKED
    cost[NX // 2, 4:7] = 1.0                   # lower gap
    cost[NX // 2, 14:17] = 1.0                 # upper gap, same distance
    star = _map(cost)

    taken = plan(star, START, GOAL).path
    used_lower = min(c[1] for c in taken) < NY // 4
    closed = cost.copy()
    lost = slice(4, 7) if used_lower else slice(14, 17)
    closed[NX // 2, lost] = BLOCKED            # M_S lost the gap pi* used
    out = regret(star, _map(closed), START, GOAL)

    assert out.found
    assert out.regret == pytest.approx(0.0, abs=0.6), "pick a truly equal-cost detour"
    assert out.frechet_m > 1.0, "the two routes are geometrically identical"


def test_planning_through_a_wall_is_flagged_not_just_expensive():
    """Two different failures live in one number. A map that routes AROUND a
    lost gap has planned a worse route; a map that routes THROUGH something
    M* calls impassable has planned into a wall. The second is a safety
    failure and infinite cost is the honest score, but the money plot needs
    them separable or it shows a cliff of infinities instead of a knee."""
    star = _wall(gap_j=10, gap=3)
    wrong_place = _wall(gap_j=25, gap=3)

    out = regret(star, wrong_place, START, GOAL)
    assert out.blocked_on_reference
    assert not np.isfinite(out.regret)

    detour = regret(star, _coarsen(star, 8), START, GOAL)
    assert not detour.blocked_on_reference
    assert np.isfinite(detour.regret)


# --- the planner itself ------------------------------------------------------


def test_astar_is_optimal_because_the_theorem_depends_on_it():
    """Non-negativity of eq. (23) IS the statement that pi* is optimal. A
    greedy or inflated-heuristic planner produces negative regrets, and they
    read as a broken metric. Checked against Dijkstra, which has no heuristic
    to get wrong."""
    rng = np.random.default_rng(4)
    for _ in range(15):
        cost = rng.uniform(1.0, 4.0, (NX, NY))
        cost[rng.random((NX, NY)) < 0.1] = BLOCKED
        cost[START] = cost[GOAL] = 1.0
        m = _map(cost)

        a = plan(m, START, GOAL)
        d = dijkstra(m, START)[GOAL]
        if a.found:
            assert a.cost == pytest.approx(float(d), rel=1e-9)


def test_path_cost_reproduces_the_planner_on_its_own_map():
    """eq. (23) scores pi_S on a map that did not produce it, so `path_cost`
    has to be the same functional the planner minimised -- otherwise regret
    carries a constant offset that no test would otherwise show."""
    m = _wall()
    p = plan(m, START, GOAL)
    assert path_cost(m, p.path) == pytest.approx(p.cost, rel=1e-12)


def test_the_planner_is_deterministic():
    """A metric that compares PATHS cannot have the path depend on heap order.
    Same cost twice is not enough; it must be the same path."""
    m = _wall()
    assert plan(m, START, GOAL).path == plan(m, START, GOAL).path


def test_impassable_is_geometry_only():
    """§7.1's split, in the cost model: clearance, slope and step say the
    vehicle cannot. Roughness and class say it would rather not -- a packed
    verge is drivable and undesirable, which is a weight, not a wall."""
    from vrgrid.cell import (
        TRAV_CLASS,
        TRAV_CLEARANCE,
        TRAV_ROUGHNESS,
        TRAV_SLOPE,
        TRAV_STEP,
    )
    from vrgrid.eval.plan_regret import _cost_from_bits

    w = weights()
    bits = np.array([[0, TRAV_ROUGHNESS, TRAV_CLASS,
                      TRAV_CLEARANCE, TRAV_SLOPE, TRAV_STEP]], dtype=np.uint8)
    cost = _cost_from_bits(bits, np.zeros(bits.shape, bool), w)

    assert np.isfinite(cost[0, :3]).all(), "a preference became a wall"
    assert not np.isfinite(cost[0, 3:]).any(), "geometry did not block"
    assert cost[0, 1] > cost[0, 0] and cost[0, 2] > cost[0, 0]


def test_unknown_is_passable_here_and_the_fraction_is_reported():
    """⚑ The concession this metric makes. Everywhere else in this project
    unknown fails safe; for a planner that rule makes R(S) undefined, because
    at P_fill < 2% per frame most of the far field has never been observed and
    no path exists at all.

    So unknown is passable at a price, and the price of that choice is that
    the fraction must be reported with the number: zero regret along a path
    that is mostly unknown says the sequence was too short, not that the
    coarsening was free.
    """
    cost = np.ones((NX, NY))
    unknown = np.zeros((NX, NY), bool)
    unknown[NX // 3:2 * NX // 3, :] = True     # spans the lattice: unavoidable
    cost[unknown] += weights()["w_unknown"]

    p = plan(_map(cost, unknown), START, GOAL)
    assert p.found, "unknown was treated as impassable and the metric died"
    assert p.unknown_fraction > 0.25
    assert plan(_map(np.ones((NX, NY))), START, GOAL).unknown_fraction == 0.0


def test_costmaps_must_share_a_lattice():
    """Subtracting two integrals taken over different domains is how a
    negative regret appears, and it would be read as a bug in the theorem."""
    a = _wall()
    b = CostMap(0.5, 0.0, 0.0, a.cost, a.unknown)
    with pytest.raises(ValueError, match="same lattice"):
        regret(a, b, START, GOAL)


# --- §8.3: the corridor ------------------------------------------------------


def test_corridor_is_a_connected_band_containing_the_optimal_path():
    """§8.3's specified test. T(c) = f(c) + g(c) is the best cost of any path
    THROUGH c, so every cell of pi* has T = J(pi*) exactly and must be in the
    band for any tau > 0.

    ⚑ The weights VARY on purpose. With a uniform-weight map the forward and
      reverse endpoint conventions differ only by a constant and the identity
      holds either way -- so a uniform test passes against a cost-to-go that
      is simply the forward search run from the goal, which is not the same
      function. That version was written first and this case is what caught
      it.
    """
    rng = np.random.default_rng(17)
    m = _wall()
    varied = m.cost.copy()
    finite = np.isfinite(varied)
    varied[finite] = rng.uniform(1.0, 6.0, int(finite.sum()))
    m = _map(varied)
    star = plan(m, START, GOAL)
    mask, through, j_star = corridor(m, START, GOAL, tau=2.0)

    assert j_star == pytest.approx(star.cost)
    for cell in star.path:
        assert mask[cell], f"the optimal path left its own corridor at {cell}"
        assert through[cell] == pytest.approx(j_star, abs=1e-9)

    # connected, by flood fill from the start
    seen, stack = set(), [START]
    while stack:
        c = stack.pop()
        if c in seen or not mask[c]:
            continue
        seen.add(c)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                n = (c[0] + di, c[1] + dj)
                if 0 <= n[0] < NX and 0 <= n[1] < NY and n not in seen:
                    stack.append(n)
    assert len(seen) == int(mask.sum()), "the corridor is in disconnected pieces"


def test_a_wider_tau_can_only_widen_the_corridor():
    """tau is the refinement budget, so it has to behave like one: paying more
    can never buy fewer cells."""
    m = _wall()
    sizes = [corridor(m, START, GOAL, tau=t)[0].sum() for t in (0.5, 2.0, 8.0)]
    assert sizes[0] <= sizes[1] <= sizes[2]


def test_cells_outside_the_corridor_cannot_change_the_plan():
    """⚑ The claim §8.3 makes, tested as stated: refining outside the band is
    provably wasted compute. Take cells well outside it, make them free, and
    assert the optimal cost does not move -- no amount of resolution there
    could have helped."""
    m = _wall()
    mask, through, j_star = corridor(m, START, GOAL, tau=1.0)

    outside = np.isfinite(m.cost) & ~mask & (through > j_star + 8.0)
    assert outside.sum() > 50, "not enough cells outside the band to test on"

    improved = m.cost.copy()
    improved[outside] = 0.01
    assert plan(_map(improved), START, GOAL).cost == pytest.approx(j_star, rel=1e-9)


# --- end to end, on the synthetic scene --------------------------------------


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    root = tmp_path_factory.mktemp("regret")
    write_sequence(root, "99", n_frames=6)
    reference = build_from_scans(read_sequence(root, "99"))

    def scans():
        for pts, labels, pose in read_sequence(root, "99"):
            moving = (labels >= 250) & (labels <= 259)
            yield (pts[~moving], (labels[~moving] % 16).astype("uint8"),
                   np.ones(int((~moving).sum()), dtype=bool), pose)

    gm = build_gridmap(load("5/10/20/40"))
    frames = run_sequence(gm, scans()).frames
    return gm, reference, frames


def test_regret_runs_end_to_end_through_the_query_api(scene):
    """The whole chain on a real map: M* -> costmap, M_S -> costmap through
    query() only, plan on both, score both on M*.

    The planner never learns the map is variable-resolution -- that is §3.7
    being exercised rather than asserted, and it is one of the three answers
    to "planners want uniform grids".
    """
    gm, reference, _ = scene
    th = load_thresholds()
    x0, y0, n = 4.0, -4.0, 32          # 8 x 8 m of road ahead of the vehicle

    star = costmap_from_reference(reference, x0, y0, n, n, thresholds=th)
    mine = costmap_from_gridmap(gm, x0, y0, n, n, vehicle_xy_m=(10.0, 0.0),
                                thresholds=th)
    assert star.same_lattice(mine)

    out = regret(star, mine, (1, n // 2), (n - 2, n // 2))
    assert out.found, "no path across 8 m of road"
    assert out.regret >= -1e-9
    assert 0.0 <= out.unknown_fraction <= 1.0
