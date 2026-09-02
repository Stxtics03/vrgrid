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

⚑⚑ **The confound was real, it was 100% against 4.2%, and the diagnostic
  built to catch it reported 0.0%.** Both are fixed as of 2026-09-02, and the
  history matters because the reassuring number is the dangerous one.

  `_cost_from_bits` charges `w_unknown` for `unknown | TRAV_CONFIDENCE`. The
  `--confound` diagnostic read `CostMap.unknown` alone -- the smaller term by
  two orders of magnitude, because a cell can be observed and still sit below
  `n_min`. So the tool added specifically to make this visible said the
  problem was absent. `CostMap.low_confidence()` is now the one place that
  answers "who pays", and it is what `--confound` prints.

  The confound itself was in `costmap_from_gridmap`, which OR-ed the
  confidence bit over the sub-cells of each planning cell. A 25 cm planning
  cell covers 25 map cells of a 5 cm ring; at ring 0's fill rate most are
  thin, so the OR fired essentially always and **the handicap grew with the
  resolution**. Measured at 14 frames, cells paying `w_unknown` inside the
  common support:

    schedule        before    after
    5/10/20/40      100.0%     0.9%
    uniform 20 cm     4.2%     0.0%
    M* reference        --      0.0%

  Confidence is evidence and evidence adds up, so the observation counts over
  the footprint's distinct cells are summed and compared against `n_min` once
  -- which is what this file's own reference side had always done with
  `block_stats`. The two sides now use one rule.

  **The restriction to common support stays** even though it currently changes
  almost nothing (the window is 99.1% common). How large this effect is
  depends on the scene, the window placement and the frame count, none of
  which are frozen; it was 65% against 4% on one earlier arrangement. An
  ablation quoting R(S) across cell sizes without the restriction is not
  interpretable whatever the current gap happens to be.

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
import warnings
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
from vrgrid.grid.lattice import OUTSIDE
from vrgrid.grid.query import query, slot_of
from vrgrid.grid.schedule import load_thresholds
from vrgrid.grid.traversability import drivable_ids

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
    unknown: np.ndarray         # (nx, ny) bool, nothing observed at all
    trav: np.ndarray = None     # (nx, ny) uint8, the §7.1 bitfield

    # ⚑ `unknown` alone is not who pays `w_unknown`. The cost function charges
    #   it for `unknown | TRAV_CONFIDENCE`, and the second term is almost all
    #   of it: a cell CAN have been observed and still be below `n_min`. The
    #   confound diagnostic read `.unknown` and reported 0.0% where the real
    #   figure was 100.0%, which is worse than having no diagnostic -- it was
    #   built to catch exactly this and said the problem was absent. `trav` is
    #   here so `low_confidence()` can answer the question that was asked.

    @property
    def shape(self):
        return self.cost.shape

    def index_of(self, x_m, y_m):
        return (int(np.floor((x_m - self.x0_m) / self.cell_m)),
                int(np.floor((y_m - self.y0_m) / self.cell_m)))

    def centre_of(self, i, j):
        return (self.x0_m + (i + 0.5) * self.cell_m,
                self.y0_m + (j + 0.5) * self.cell_m)

    def low_confidence(self) -> np.ndarray:
        """(nx, ny) bool -- cells actually paying `w_unknown`. Not `.unknown`.

        This is the fill-rate confound made visible. Read it next to R(S) for
        any cross-schedule comparison: a schedule whose fine rings hold few
        returns per cell pays this on most of its window and is penalised for
        resolving finely, which is precisely backwards.
        """
        if self.trav is None:
            return np.asarray(self.unknown, dtype=bool)
        return np.asarray(self.unknown, dtype=bool) | (
            np.asarray(self.trav) & TRAV_CONFIDENCE).astype(bool)

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

    Every planning cell gathers `samples x samples` `query()` calls over its
    footprint and then **the §7.1 predicate is evaluated once, on the planning
    lattice** -- the same lattice, from the same statistics, as
    `costmap_from_reference`. Eq. (23) subtracts the two, so they have to be
    the same measurement of the same thing at the same scale.

    ⚑ **This used to OR the stored bitfields, and eq. (23) was not comparing
      like with like.** The bits in a cell were computed on its RING lattice --
      a step over a 5 cm neighbourhood in ring 0, over 40 cm in ring 3 -- while
      the reference side computed them at the 25 cm planning cell from block
      means. The 12 cm kerb is a step at 5 cm and is smoothed at 25 cm, so the
      fine schedule reported walls the reference structurally could not have.
      Measured on 14 frames: 5/10/20/40 invented 148 impassable cells the
      reference did not have and missed 8 that it did, against 12 real ones.
      A path planned round 148 phantom walls and scored against a map without
      them is not a regret, it is two different problems.

    ⚑ **And OR-ing `TRAV_CONFIDENCE` penalised a schedule for being fine.**
      A 25 cm planning cell covers 25 map cells of a 5 cm ring; at ring 0's
      fill rate most are thin, so the OR set the confidence bit essentially
      always. 5/10/20/40 paid `w_unknown` on 100% of the surviving window
      against uniform 20 cm's 4.2% -- a 4-unit handicap on every cell, for
      resolving finely. Confidence is EVIDENCE and evidence adds up: the
      footprint's observation counts are summed over the distinct cells it
      covers and compared against `n_min` once, which is exactly what the
      reference side does with `block_stats`.

      Note "distinct". Twenty-five samples over a 40 cm ring cell all land in
      the same cell, and summing per sample would multiply its evidence by 25.
      `slot_of` is used for identity only -- never to read a cell -- so the
      claim that a planner needs no knowledge of the ring layout still holds.

    ⚑ **Clearance is not evaluated here**, because `costmap_from_reference`
      cannot evaluate it: M* is 2.5D ground with no ceiling. Keeping it on
      this side alone would mean M_S blocking cells M* is structurally unable
      to block, which is a difference between the two maps' CONTENTS being
      scored as a difference in coarsening. Stated rather than hidden, as the
      reference side already stated it. The clearance bit is still in the map
      and still in `query()`; it is this metric that must not use it.

    Heights are combined by observation count and variances by the law of
    total variance (§4.2) -- the children measure different *places*, so
    dropping the between-cell term would make a planning cell most confident
    exactly where it straddles a kerb.

    Slow, and deliberately so: the claim being demonstrated is that a planner
    can treat this map as uniform, and reaching into the rings to go faster
    here would assume away the thing under test.
    """
    th = thresholds if thresholds is not None else gm.thresholds
    w = weights(th)
    t = th["traversability"]
    cell_m = float(w.get("cell_m", 0.25)) if cell_m is None else cell_m
    samples = max(1, int(samples))
    offsets = (np.arange(samples) + 0.5) / samples

    n_tot = np.zeros((nx, ny), dtype=np.int64)
    z = np.full((nx, ny), np.nan)
    var = np.zeros((nx, ny))
    cls = np.zeros((nx, ny), dtype=np.int64)
    soft = np.zeros((nx, ny), dtype=np.uint8)   # bits that stay per-cell

    for i in range(nx):
        for j in range(ny):
            seen = {}
            for du in offsets:
                wx = x0_m + (i + du) * cell_m - vehicle_xy_m[0]
                for dv in offsets:
                    wy = y0_m + (j + dv) * cell_m - vehicle_xy_m[1]
                    ring, slot = slot_of(gm, wx, wy)
                    if ring == OUTSIDE or (ring, slot) in seen:
                        continue
                    q = query(gm, wx, wy)
                    if q.occupancy == OCC_UNKNOWN:
                        continue
                    seen[(ring, slot)] = q

            if not seen:
                continue
            qs = list(seen.values())
            counts = np.array([max(q.confidence, 1) for q in qs], dtype=np.float64)
            mus = np.array([q.ground_height for q in qs], dtype=np.float64)
            wts = counts / counts.sum()

            n_tot[i, j] = int(sum(q.confidence for q in qs))
            mu = float((wts * mus).sum())
            z[i, j] = mu
            # §4.2: between-cell spread is part of the block's variance. The
            # within-cell term is not available through `query()`, so this is
            # the between term alone and therefore a LOWER bound -- which is
            # the conservative direction for a roughness threshold.
            var[i, j] = float((wts * (mus - mu) ** 2).sum())
            cls[i, j] = int(qs[int(np.argmax(counts))].semantic_class)
            # Roughness and class are per-cell properties, not neighbourhood
            # ones, so OR is the right combiner for them: a rough patch
            # anywhere in the footprint makes the footprint rough.
            for q in qs:
                soft[i, j] |= q.traversability & (TRAV_ROUGHNESS | TRAV_CLASS)

    unknown = n_tot == 0

    trav = np.zeros((nx, ny), dtype=np.uint8)
    trav |= np.where(_slope(z, cell_m) > np.tan(np.radians(t["theta_max_deg"])),
                     TRAV_SLOPE, 0).astype(np.uint8)
    trav |= np.where(_max_step(z) > t["s_max_m"], TRAV_STEP, 0).astype(np.uint8)
    trav |= np.where(var > t["sigma2_max_m2"], TRAV_ROUGHNESS, 0).astype(np.uint8)
    trav |= soft & TRAV_ROUGHNESS
    trav |= np.where(np.isin(cls, drivable_ids(th)) & ~unknown, 0,
                     TRAV_CLASS).astype(np.uint8)
    trav |= np.where(n_tot < t["n_min"], TRAV_CONFIDENCE, 0).astype(np.uint8)

    return CostMap(cell_m, x0_m, y0_m, _cost_from_bits(trav, unknown, w),
                   unknown, trav)


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

    return CostMap(cell_m, x0_m, y0_m, _cost_from_bits(trav, unknown, w),
                   unknown, trav)


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
    """max|z - z_nbr| over the 4-neighbourhood, 0 where nothing is comparable.

    A cell whose four neighbours are all unknown gives an all-NaN slice, and
    `np.nanmax` warns on those. The warning is not informative here -- an
    unobserved neighbourhood is the ordinary case at the window edge and the
    answer is "no step evidence", which is what 0.0 says -- and one test runs
    the pipeline with warnings as errors, so it is suppressed at the source
    rather than left to rattle through every eval run.
    """
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
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
