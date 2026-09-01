"""Plan regret — coarsening measured in units of decision. Math §8. [Aakash]

This is the research claim. Everything else in the project is engineering.

Plan on the reference map, plan on the compressed map, compare the DECISIONS,
not the reconstructions. If the planner reaches the same waypoint sequence,
the compression was free in the only sense a robot cares about.

It is also the answer to the hardest question you will be asked — "standard
planners want uniform grids, so you give the savings back in resampling" —
alongside the resolution-agnostic query API (§3.7) and the conservative
pyramid (§7.2). Three independent answers, which is why this is worth its days.

--- the one detail that decides whether the metric means anything ---------

    R(S) = J_{M*}(pi_S) - J_{M*}(pi*)                              eq. (23)

**Both paths are scored on M*.** Scoring pi_S on M_S measures self-consistency,
not quality: a badly coarsened map will happily report that its own bad plan is
cheap, and the metric comes out beautiful precisely when the map is worst.
`test_scoring_on_the_compressed_map_hides_the_damage` builds that case and
shows the wrong metric reporting ~0 where the right one reports a real detour.

Non-negativity is by construction, not by luck: pi* minimises J_{M*}, so
nothing scored on M* can beat it. A negative regret means the two costmaps are
not on the same lattice, and that is the first thing to check if one appears.

--- planning through query(), on purpose ---------------------------------

The planner runs on a uniform lattice at its OWN resolution and gets every
cell through `query()`, never by reaching into a ring. That is the §3.7 claim
being exercised rather than asserted: the planner does not know the map is
variable-resolution, and a planner that has to know is a planner that has to
be rewritten. The planning lattice is coarser than the map's finest ring by
design -- a vehicle plans in 25 cm steps, not 5 cm ones -- so this is not the
resampling the objection is about.

--- ⚑⚑ the confound that decides whether the ablation is valid at all -----

When this note was written the scene produced, at 14 frames and one
11 x 11 m planning window:

    schedule        R(S)     cells low-confidence in the window
    5/10/20/40      5.803    65%
    uniform 20 cm   0.146     4%

Read naively that says our schedule is forty times worse than a uniform grid.
It says nothing of the kind. Nothing is impassable in either map. The 5 cm
ring holds few returns per cell -- P_fill is 11.6% per frame at ring 0 and the
far field fills only by ego-motion (§1.3) -- so most of its cells are below
`n_min`, pay `w_unknown` plus the class penalty for an unset class byte, and
the planner routes around a map that is merely SPARSE rather than wrong.

⚑ **Those magnitudes no longer reproduce, and the reason is not that the
  confound went away.** Re-measured 2026-09-01 with
  `python scripts/eval_synthetic.py --frames 14 --confound`:

    schedule        R(S)     cells low-confidence in the window
    5/10/20/40      2.389     1%
    uniform 20 cm   3.354     0%

  The window is now 99.1% common support, so restricting it changes almost
  nothing -- unrestricted and restricted R(S) agree to three decimals. Two
  things moved underneath the old numbers: the planning window was repositioned
  behind the final pose rather than ahead of it (see `PLAN_BEHIND_M` in
  `scripts/eval_synthetic.py`, which records why), and the synthetic sampler's
  beam-surface intersection was wrong and is fixed, which raised coverage in
  every ring.

  The restriction stays, and the numbers are printed side by side, because how
  large this effect is depends on the scene, the window placement and the frame
  count -- none of which are frozen. It was 65% against 4% on one arrangement
  of them. An ablation that quotes R(S) across cell sizes without the
  restriction is not interpretable whatever the current gap happens to be.

**So R(S) compared across schedules with different cell sizes measures fill
rate, not coarsening, unless the comparison is restricted to ground all of
them have actually observed.** A finer schedule is penalised for resolving
finely, which is precisely backwards, and the effect is large enough to
reverse the headline.

`common_support()` builds that restriction: cells observed by every map in
the comparison, everything else excluded from both paths' cost. Use it for
any cross-schedule number. `unknown_fraction` is what makes the confound
visible when it is not used, which is why it is returned next to R(S) and not
buried -- an ablation quoting R(S) without it is not interpretable.

--- ⚑ unknown cannot be impassable here, and that is a real concession ----

The project's rule is "unknown is not free" and §7.1 bit 5 fails safe. For a
PLANNER that rule makes the metric undefined: at P_fill < 2% per frame (§1.3)
most of the far field has never been observed, so with unknown impassable no
path exists at all and R(S) cannot be computed for any schedule.

So unknown is passable at a price, `w_unknown` in `configs/thresholds.yaml`,
and `PlanResult.unknown_fraction` reports how much of each path went through
it. Read the two together: a regret of zero along a path that is 80% unknown
is not evidence that the coarsening was free, it is evidence that the sequence
was too short to fill the map. The fraction is not decoration -- it is the
condition under which the headline number means anything.
"""

import heapq
import itertools
from dataclasses import dataclass, field

import numpy as np
from vrgrid.cell import (
    OCC_UNKNOWN,
    TRAV_CLASS,
    TRAV_CLEARANCE,
    TRAV_CONFIDENCE,
    TRAV_ROUGHNESS,
    TRAV_SLOPE,
    TRAV_STEP,
)
from vrgrid.grid.query import query
from vrgrid.grid.schedule import load_thresholds

# Geometry decides, semantics filters (§7.1). These three say the vehicle
# physically cannot: no weight makes them passable.
IMPASSABLE_BITS = TRAV_CLEARANCE | TRAV_SLOPE | TRAV_STEP

BLOCKED = np.inf
DIAG = np.sqrt(2.0)


def weights(thresholds=None) -> dict:
    """Cost weights per §8.1's `w`, from config. §8.1 says "w derived from the
    traversability bitfield" and stops there, so the derivation is here and
    the numbers are in `configs/thresholds.yaml` with everything else frozen
    before the ablation (flaw E6)."""
    th = thresholds if thresholds is not None else load_thresholds()
    return th.get("plan", {})


@dataclass
class CostMap:
    """A uniform planning lattice in WORLD metres. `cost` is the per-cell
    weight `w`; `inf` is impassable.

    Both maps in a regret comparison must share this lattice exactly --
    same cell size, same origin, same shape -- or eq. (23) is subtracting
    two different integrals and can come out negative.
    """

    cell_m: float
    x0_m: float
    y0_m: float
    cost: np.ndarray            # (nx, ny)
    unknown: np.ndarray         # (nx, ny) bool, for the fraction report

    @property
    def shape(self):
        return self.cost.shape

    def index_of(self, x_m, y_m):
        return (int(np.floor((x_m - self.x0_m) / self.cell_m)),
                int(np.floor((y_m - self.y0_m) / self.cell_m)))

    def centre_of(self, i, j):
        return (self.x0_m + (i + 0.5) * self.cell_m,
                self.y0_m + (j + 0.5) * self.cell_m)

    def same_lattice(self, other) -> bool:
        return (self.cell_m == other.cell_m and self.x0_m == other.x0_m
                and self.y0_m == other.y0_m and self.shape == other.shape)


def _cost_from_bits(trav, unknown, w) -> np.ndarray:
    """Bitfield -> weight. The `w` of §8.1, spelled out.

    Impassable is reserved for the three geometric bits -- clearance, slope,
    step -- because those are statements that the vehicle cannot fit, climb or
    mount. Roughness and class are preferences: a packed verge is drivable and
    undesirable, which is a weight, not a wall. That split is §7.1's "geometry
    decides, semantics filters" turned into numbers.
    """
    cost = np.full(trav.shape, float(w.get("w_base", 1.0)))
    cost += np.where(trav & TRAV_ROUGHNESS, w.get("w_roughness", 2.0), 0.0)
    cost += np.where(trav & TRAV_CLASS, w.get("w_class", 3.0), 0.0)
    cost += np.where(unknown | (trav & TRAV_CONFIDENCE).astype(bool),
                     w.get("w_unknown", 4.0), 0.0)
    return np.where(trav & IMPASSABLE_BITS, BLOCKED, cost)


def costmap_from_gridmap(gm, x0_m, y0_m, nx, ny, cell_m=None,
                         vehicle_xy_m=(0.0, 0.0), thresholds=None,
                         samples: int = 5) -> CostMap:
    """M_S -> a planning costmap, entirely through `query()`.

    Every planning cell is `samples x samples` `query()` calls over its
    footprint, combined conservatively: the bitfield is OR-ed, so a hazard
    anywhere in the cell blocks it, and the cell counts as observed if ANY
    sample was observed.

    ⚑ A single query at the cell centre is wrong, and wrong in a way that
      looks fine. A 25 cm planning cell over a 5 cm ring covers 25 map cells;
      at ring 0's fill rate most of them are the gaps between beam tracks, so
      centre-sampling picks an unobserved cell about four times in five and
      the planner sees a map that is mostly holes. The finer the schedule,
      the worse it looks -- exactly backwards. It also makes the common
      support of several schedules disconnected, so no path exists at all,
      which is how this was found.

      OR-ing the bits is the same rule as §7.2's conservative pyramid: a block
      is safe only if every cell in it is. Doing it by sampling rather than by
      a pyramid is slower and needs no new structure; when
      `query_conservative()` lands this should call it instead.

    Slow, and deliberately so: the claim being demonstrated is that a planner
    can treat this map as uniform, and reaching into the rings to go faster
    here would assume away the thing under test.
    """
    th = thresholds if thresholds is not None else gm.thresholds
    w = weights(th)
    cell_m = float(w.get("cell_m", 0.25)) if cell_m is None else cell_m
    samples = max(1, int(samples))
    offsets = (np.arange(samples) + 0.5) / samples

    trav = np.zeros((nx, ny), dtype=np.uint8)
    unknown = np.ones((nx, ny), dtype=bool)
    for i in range(nx):
        for j in range(ny):
            bits = 0
            seen = False
            for du in offsets:
                wx = x0_m + (i + du) * cell_m - vehicle_xy_m[0]
                for dv in offsets:
                    wy = y0_m + (j + dv) * cell_m - vehicle_xy_m[1]
                    q = query(gm, wx, wy)
                    if q.occupancy == OCC_UNKNOWN:
                        continue
                    seen = True
                    bits |= q.traversability
            trav[i, j] = bits
            unknown[i, j] = not seen
    return CostMap(cell_m, x0_m, y0_m, _cost_from_bits(trav, unknown, w), unknown)


def costmap_from_reference(reference, x0_m, y0_m, nx, ny, cell_m=None,
                           thresholds=None) -> CostMap:
    """M* -> a planning costmap, by applying §7.1 to the reference heights.

    The same predicate as the map under test, on the same lattice, so eq. (23)
    subtracts like from like. Clearance is absent -- M* is 2.5D ground and has
    no ceiling -- which is stated rather than hidden: regret measures what
    COARSENING costs, and a condition the reference cannot evaluate would
    otherwise be scored as a difference between the two maps.

    Slope and step come from the block means, roughness from the block
    variance -- all three straight out of `block_stats`, which is the same
    summed-area machinery §9.2 uses.
    """
    th = thresholds if thresholds is not None else load_thresholds()
    w = weights(th)
    t = th["traversability"]
    cell_m = float(w.get("cell_m", 0.25)) if cell_m is None else cell_m

    k = round(cell_m / reference.cell_m)
    if abs(cell_m / reference.cell_m - k) > 1e-9:
        raise ValueError(
            f"planning cell {cell_m} m is not a whole number of {reference.cell_m} m "
            "reference cells; the footprint would not be a rectangle of integers"
        )

    i_lo = (np.floor(x0_m / reference.cell_m).astype(np.int64)
            + np.arange(nx)[:, None] * k)
    j_lo = (np.floor(y0_m / reference.cell_m).astype(np.int64)
            + np.arange(ny)[None, :] * k)
    i_lo, j_lo = np.broadcast_arrays(i_lo, j_lo)

    n, mean, var = reference.block_stats(i_lo, j_lo, k)
    unknown = n == 0
    z = np.where(unknown, np.nan, mean / 100.0)      # cm -> m

    trav = np.zeros(z.shape, dtype=np.uint8)
    trav |= np.where(_slope(z, cell_m) > np.tan(np.radians(t["theta_max_deg"])),
                     TRAV_SLOPE, 0).astype(np.uint8)
    trav |= np.where(_max_step(z) > t["s_max_m"], TRAV_STEP, 0).astype(np.uint8)
    trav |= np.where(var * 1e-4 > t["sigma2_max_m2"], TRAV_ROUGHNESS, 0).astype(np.uint8)
    trav |= np.where(n < t["n_min"], TRAV_CONFIDENCE, 0).astype(np.uint8)

    return CostMap(cell_m, x0_m, y0_m, _cost_from_bits(trav, unknown, w), unknown)


def _neighbour_diffs(z):
    """|z - z_nbr| over the 4-neighbourhood, nan where either side is unknown.

    Edges are excluded rather than wrapped: rolling would compare the north
    edge against the south one, which is the same mistake the ring windows
    have to avoid in `traversability.gradient`."""
    out = []
    for axis in (0, 1):
        for shift in (-1, 1):
            rolled = np.roll(z, shift, axis=axis)
            sl = [slice(None), slice(None)]
            sl[axis] = 0 if shift == -1 else -1
            rolled[tuple(sl)] = np.nan          # the wrapped row/column
            out.append(np.abs(rolled - z))
    return out


def _max_step(z):
    with np.errstate(invalid="ignore"):
        stacked = np.stack(_neighbour_diffs(z))
        return np.nan_to_num(np.nanmax(stacked, axis=0), nan=0.0)


def _slope(z, cell_m):
    """Central differences, eq. (22), on the planning lattice."""
    with np.errstate(invalid="ignore"):
        dzdx = (np.roll(z, -1, axis=0) - np.roll(z, 1, axis=0)) / (2 * cell_m)
        dzdy = (np.roll(z, -1, axis=1) - np.roll(z, 1, axis=1)) / (2 * cell_m)
        dzdx[[0, -1], :] = 0.0
        dzdy[:, [0, -1]] = 0.0
        return np.nan_to_num(np.hypot(dzdx, dzdy), nan=0.0)


# --- the planner -------------------------------------------------------------


@dataclass
class PlanResult:
    path: list = field(default_factory=list)      # [(i, j), ...] in lattice cells
    cost: float = float("inf")
    unknown_fraction: float = float("nan")
    expanded: int = 0

    @property
    def found(self) -> bool:
        return bool(self.path)


_STEPS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
          (-1, -1, DIAG), (-1, 1, DIAG), (1, -1, DIAG), (1, 1, DIAG)]


def plan(costmap: CostMap, start, goal) -> PlanResult:
    """Grid A* on the costmap. `start` and `goal` are (i, j) lattice cells.

    Eight-connected with the octile heuristic, which is admissible for
    eight-connected movement and therefore leaves A* optimal -- and optimality
    is not a nicety here: eq. (23)'s non-negativity is exactly the statement
    that pi* is optimal on M*. A greedy or inflated-heuristic planner would
    produce negative regrets and they would look like a bug in the metric.

    Ties are broken by the cell index so two runs on one map give the same
    path, not merely the same cost. A metric that compares paths cannot have
    the path depend on heap order.
    """
    cost = costmap.cost
    nx, ny = cost.shape
    if not _inside(start, nx, ny) or not _inside(goal, nx, ny):
        raise ValueError(f"start {start} or goal {goal} outside {nx}x{ny}")
    if not np.isfinite(cost[goal]) or not np.isfinite(cost[start]):
        return PlanResult()

    w_min = float(np.min(cost[np.isfinite(cost)])) if np.any(np.isfinite(cost)) else 1.0

    def h(node):
        di, dj = abs(node[0] - goal[0]), abs(node[1] - goal[1])
        return w_min * costmap.cell_m * ((DIAG - 1.0) * min(di, dj) + max(di, dj))

    g = {start: 0.0}
    came = {}
    heap = [(h(start), 0.0, start)]
    seen = set()
    expanded = 0

    while heap:
        _, gc, node = heapq.heappop(heap)
        if node in seen:
            continue
        seen.add(node)
        expanded += 1
        if node == goal:
            return _result(costmap, _unwind(came, node), gc, expanded)

        for di, dj, step in _STEPS:
            nxt = (node[0] + di, node[1] + dj)
            if not _inside(nxt, nx, ny) or nxt in seen:
                continue
            w = cost[nxt]
            if not np.isfinite(w):
                continue
            # J = sum over cells of w(c) * dl: the weight of the cell ENTERED,
            # times how far this step travelled. eq. (23)'s functional, and
            # `path_cost` below must agree with it exactly or a path scored on
            # its own map would not reproduce the planner's own number.
            cand = gc + float(w) * step * costmap.cell_m
            if cand < g.get(nxt, np.inf) - 1e-12:
                g[nxt] = cand
                came[nxt] = node
                heapq.heappush(heap, (cand + h(nxt), cand, nxt))

    return PlanResult(expanded=expanded)


def _inside(node, nx, ny):
    return 0 <= node[0] < nx and 0 <= node[1] < ny


def _unwind(came, node):
    path = [node]
    while node in came:
        node = came[node]
        path.append(node)
    return path[::-1]


def _result(costmap, path, cost, expanded) -> PlanResult:
    unknown = np.mean([costmap.unknown[c] for c in path]) if path else float("nan")
    return PlanResult(path, float(cost), float(unknown), expanded)


def path_cost(costmap: CostMap, path) -> float:
    """J_M(pi) for a path that may have been planned on a different map.

    The whole of eq. (23) rests on this: pi_S is scored on M*, so this has to
    apply M*'s weights to a path M* never produced. Returns inf if the path
    crosses a cell M* calls impassable -- which is the honest answer, and it
    is what makes a schedule that plans through a wall score badly rather than
    cheaply.
    """
    if not path:
        return float("inf")
    total = 0.0
    for prev, cur in itertools.pairwise(path):
        w = costmap.cost[cur]
        if not np.isfinite(w):
            return float("inf")
        step = DIAG if (prev[0] != cur[0] and prev[1] != cur[1]) else 1.0
        total += float(w) * step * costmap.cell_m
    return total


# --- eq. (23) ----------------------------------------------------------------


@dataclass
class Regret:
    regret: float
    reference_cost: float
    scored_cost: float
    frechet_m: float
    unknown_fraction: float
    self_scored_cost: float      # the WRONG metric, kept for the comparison
    found: bool
    blocked_on_reference: bool = False


def regret(reference_map: CostMap, compressed_map: CostMap, start, goal) -> Regret:
    """R(S) = J_{M*}(pi_S) - J_{M*}(pi*). Math §8.1 eq. (23).

    `reference_map` is M* as a costmap, `compressed_map` is M_S. Both paths
    are scored on M*; `self_scored_cost` is what you would get by scoring pi_S
    on M_S instead, and it is returned only so the two can be printed side by
    side. It is not the metric and must never be reported as one.

    Also returns the discrete Frechet distance to pi*, per §8.1: a detour that
    costs the same but goes somewhere quite different is a real difference in
    decision that a cost difference alone scores as zero.
    """
    if not reference_map.same_lattice(compressed_map):
        raise ValueError(
            "the two costmaps are not on the same lattice, so eq. (23) would "
            "subtract two different integrals -- and would sometimes come out "
            "negative, which reads as a bug in the metric rather than in the setup"
        )

    star = plan(reference_map, start, goal)
    mine = plan(compressed_map, start, goal)
    if not (star.found and mine.found):
        return Regret(float("nan"), star.cost, float("inf"), float("nan"),
                      mine.unknown_fraction, mine.cost, False)

    # ⚑ Two failures live in this one number and they are not the same failure.
    # A coarse map that routes AROUND a lost gap costs more: finite regret, a
    # worse plan. A coarse map that routes THROUGH something M* calls
    # impassable has not planned a worse route, it has planned into a wall --
    # a safety failure, and infinite cost is the honest score for it. The flag
    # separates them so the money plot can show a knee rather than a cliff of
    # infinities, and so nobody reads "inf" as "very expensive detour".
    scored = path_cost(reference_map, mine.path)
    return Regret(
        blocked_on_reference=not np.isfinite(scored),
        regret=scored - star.cost,
        reference_cost=star.cost,
        scored_cost=scored,
        frechet_m=frechet(reference_map, mine.path, star.path),
        unknown_fraction=mine.unknown_fraction,
        self_scored_cost=mine.cost,
        found=True,
    )


def common_support(*costmaps) -> np.ndarray:
    """Cells every map in the comparison has actually observed. Math §8.2.

    The ablation is only a comparison of SCHEDULES if every schedule is scored
    on the same ground. A 5 cm ring and a 40 cm ring see very different
    fractions of the same sequence, and the difference is fill rate rather
    than information loss -- see the confound note at the top of this file.

    Returns a boolean mask to pass to `restrict()`.
    """
    mask = ~costmaps[0].unknown
    for c in costmaps[1:]:
        if not costmaps[0].same_lattice(c):
            raise ValueError("common support needs one lattice for every map")
        mask &= ~c.unknown
    return mask


def restrict(costmap: CostMap, mask) -> CostMap:
    """A costmap that exists only where `mask` holds; elsewhere impassable.

    Impassable rather than merely expensive on purpose: a cell outside the
    common support must not be routed through at ANY price, or the restriction
    leaks back in as a weight and the comparison is confounded again by how
    much each map declined to look at.
    """
    cost = np.where(mask, costmap.cost, BLOCKED)
    return CostMap(costmap.cell_m, costmap.x0_m, costmap.y0_m, cost,
                   costmap.unknown & mask)


def frechet(costmap: CostMap, path_a, path_b) -> float:
    """Discrete Frechet distance between two paths, in metres. §8.1.

    The geometric companion to the cost difference: two routes around opposite
    sides of a building can cost the same to within rounding and are not the
    same decision. Reported alongside R(S), never instead of it.

    Iterative rather than the usual recursion -- a path over a 200x200 lattice
    is a few hundred cells and Python's recursion limit is 1000.
    """
    if not path_a or not path_b:
        return float("nan")
    pa = np.array([costmap.centre_of(*c) for c in path_a])
    pb = np.array([costmap.centre_of(*c) for c in path_b])

    d = np.hypot(pa[:, None, 0] - pb[None, :, 0], pa[:, None, 1] - pb[None, :, 1])
    ca = np.empty_like(d)
    ca[0, 0] = d[0, 0]
    for i in range(1, len(pa)):
        ca[i, 0] = max(ca[i - 1, 0], d[i, 0])
    for j in range(1, len(pb)):
        ca[0, j] = max(ca[0, j - 1], d[0, j])
    for i in range(1, len(pa)):
        for j in range(1, len(pb)):
            ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), d[i, j])
    return float(ca[-1, -1])


# --- §8.3: the online policy -------------------------------------------------


def dijkstra(costmap: CostMap, source, reverse: bool = False) -> np.ndarray:
    """Cost from `source` over the whole lattice. Unreachable is inf.

    `reverse=False` gives cost-to-come: the cheapest cost of a path from
    `source` to c, paying for every cell ENTERED -- so it includes w(c) and
    excludes w(source). That is the functional `plan()` minimises.

    `reverse=True` gives cost-to-go: the cheapest cost from c to `source`,
    which excludes w(c) and includes w(source). It is not the same walk
    backwards -- the endpoint conventions differ at both ends -- so it pays
    the weight of the cell it LEAVES instead of the one it enters.

    ⚑ Getting this wrong does not break anything visibly. f + g comes out
      offset by a constant, the corridor of §8.3 is still a connected band in
      roughly the right place, and only `T(c) == J(pi*)` along the optimal
      path -- which is the identity eq. (24) actually asserts -- catches it.
    """
    cost = costmap.cost
    nx, ny = cost.shape
    dist = np.full((nx, ny), np.inf)
    if not np.isfinite(cost[source]):
        return dist
    dist[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node] + 1e-12:
            continue
        for di, dj, step in _STEPS:
            nxt = (node[0] + di, node[1] + dj)
            if not _inside(nxt, nx, ny):
                continue
            w = cost[nxt] if not reverse else cost[node]
            if not np.isfinite(cost[nxt]):
                continue
            cand = d + float(w) * step * costmap.cell_m
            if cand < dist[nxt] - 1e-12:
                dist[nxt] = cand
                heapq.heappush(heap, (cand, nxt))
    return dist


def corridor(costmap: CostMap, start, goal, tau: float):
    """Cells that could still change the plan. Math §8.3, eqs. (24)-(25).

        T(c) = f(c) + g(c)          best path cost THROUGH c
        refine where T(c) - J(pi*) < tau

    ⚑ Cells outside that band cannot change the plan however finely they are
      resolved, so refining them is provably wasted compute. That is the whole
      claim, and it is what connects this metric to the refinement pool: tau
      sets the budget and maps onto the pool size.

    Two Dijkstras: cost-to-come from the start, cost-to-go to the goal. The
    second is `reverse=True` and that is not a detail -- see the note there.
    With the right endpoint conventions f + g needs no correction term, and
    T(c) equals J(pi*) exactly on every cell of the optimal path, which is
    what makes tau a cost budget rather than an arbitrary knob.

    Returns (mask, T, J*), so a caller can plot the band as well as use it.
    """
    f = dijkstra(costmap, start)
    g = dijkstra(costmap, goal, reverse=True)
    through = f + g

    star = plan(costmap, start, goal)
    if not star.found:
        return np.zeros(costmap.shape, dtype=bool), through, float("inf")
    return (through - star.cost) < tau, through, star.cost
