#!/usr/bin/env python3
"""What question can the plan-regret window actually pose? Math §7.1, §8.1.
[Shrestha]

    python scripts/plan_query_survey.py [--frames 12] [--map]

This exists because of one open item. `regret_plot.py`'s curve has no knee,
the diagnosis in its docstring is that `eval_synthetic.PLAN_LANE_CELLS` runs
the path down a lane the scene's hazards are not on, and the agreed next step
is to reposition the query so it has to decide about one of them. Aakash's
note is that changing the experiment after seeing which answer it gives is how
a figure stops being evidence, and he is right, so this script is deliberately
built so that it *cannot* be used to tune:

    ⚑ It never computes R(S). Not once, not for a candidate query, not as a
      diagnostic. It reports what the costmaps CONTAIN -- which cells M* calls
      impassable and why, what the passable cost field looks like, and which
      of that survives into each schedule's M_S. Those are properties of the
      scene and of the maps, and they are knowable before a query is posed.

The point of the separation is that a query chosen from this output is chosen
from the terrain, and a query chosen from an R(S) table is chosen from the
answer. Only the first one is evidence.

Window and lattice come from `eval_synthetic` by import, not by a second copy
of the constants, so what is surveyed here is the window the figure uses --
including the frame-count dependence, which is why `--frames` is the first
argument and is printed in the header.

--- what to read in the output ------------------------------------------

**IMPASSABLE, on M*.** The three geometric bits are the only ones that make a
cell a wall (§7.1, and `_cost_from_bits`'s "geometry decides, semantics
filters"). A query that does not have at least one of these between its start
and its goal is not asking the planner to decide anything, and R(S) along it
is measuring tie-breaking between equal-cost paths.

**PASSABLE COST SPREAD.** A wall is not the only kind of decision. A graded
cost field poses one too -- the cheapest route bends around expense rather
than around a hole -- and it is the kind that produces a *curve* rather than a
step. If every passable cell in the window costs `w_base`, the only decision
available is binary and the money plot cannot have a knee whatever the query.

**SURVIVES INTO M_S.** The claim the figure exists to make is that a coarse
map loses a feature a fine map keeps. That is visible here directly, per
schedule, without a planner: if every schedule marks the same cells
impassable, coarsening cost nothing on this scene and the flat curve is the
true answer rather than a badly-posed one.

**THE TWO SIDES OF eq. (23) ARE NOT ON THE SAME LATTICE.** This is what the
survey was built to check and it is what it found. `costmap_from_reference`
blocks M* down to `plan.cell_m` and applies §7.1 to the block means;
`costmap_from_gridmap` applies §7.1 at the MAP's cell size and samples the
result. §7.1's thresholds are not scale-invariant -- a step of `h` reads as a
gradient of `h / 2c` at cell size `c` -- so the same kerb is a wall on a 5 cm
map and flat ground on M*. Section 6 prints the crossing point across every
cell size in the sweep; section 3 prints what it does to the wall sets.

**CLASS IS ON ONLY ONE SIDE.** `costmap_from_reference` states that clearance
is absent from M* and why. Bit 4 (class) is absent too and that is not stated:
`ReferenceMap` carries `class_id`, but the reference costmap is built from
`block_stats` alone, which is heights. So M_S pays `w_class` on ground it
calls non-drivable and M* pays nothing anywhere. Any query whose alternative
routes differ in ground class -- road against verge, which on this scene means
any LATERAL query -- reads that asymmetry as regret. It is reported as its own
line because it decides whether a lateral reposition is available at all.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_synthetic as sweep
from vrgrid.cell import (
    TRAV_CLASS,
    TRAV_CLEARANCE,
    TRAV_CONFIDENCE,
    TRAV_ROUGHNESS,
    TRAV_SLOPE,
    TRAV_STEP,
)
from vrgrid.eval.harness import build_gridmap, run_sequence, uniform_schedule
from vrgrid.eval.plan_regret import (
    common_support,
    costmap_from_gridmap,
    costmap_from_reference,
    weights,
)
from vrgrid.eval.reference_map import build_from_scans
from vrgrid.eval.synthetic import (
    KERB_HEIGHT_M,
    KERB_Y_M,
    POTHOLE_DEPTH_M,
    POTHOLE_XY_M,
    RAMP_SLOPE,
    RAMP_START_X_M,
    read_sequence,
    write_sequence,
)
from vrgrid.grid.schedule import load, load_thresholds
from vrgrid.grid.transient import TrackList
from vrgrid.grid.traversability import baseline_k


# The scene's three features, with the quantity §7.1 actually tests against
# each one. `predicate` is evaluated at the PLANNING cell size, because that is
# where both costmaps are built -- M* is blocked down from 5 cm to `plan.cell_m`
# by `costmap_from_reference` before any bit is set, so a feature that fires at
# 5 cm and not at 25 cm is not a hazard to this metric.
def features(cell_m: float, t: dict) -> list:
    """(name, where, bit tested, value, threshold, fires) at one lattice."""
    tan_max = np.tan(np.radians(t["theta_max_deg"]))
    s_max = t["s_max_m"]
    return [
        (f"kerb  |y|={KERB_Y_M:.0f} m", "step",
         KERB_HEIGHT_M, s_max),
        (f"kerb  |y|={KERB_Y_M:.0f} m", "slope",
         KERB_HEIGHT_M / (2.0 * cell_m), tan_max),
        (f"pothole ({POTHOLE_XY_M[0]:.0f},{POTHOLE_XY_M[1]:.0f})", "step",
         POTHOLE_DEPTH_M, s_max),
        (f"pothole ({POTHOLE_XY_M[0]:.0f},{POTHOLE_XY_M[1]:.0f})", "slope",
         POTHOLE_DEPTH_M / (2.0 * cell_m), tan_max),
        (f"ramp  x>{RAMP_START_X_M:.0f} m", "step",
         RAMP_SLOPE * cell_m, s_max),
        (f"ramp  x>{RAMP_START_X_M:.0f} m", "slope",
         RAMP_SLOPE, tan_max),
    ]


def bit_breakdown(trav, unknown) -> dict:
    """Cell counts per §7.1 bit. `trav` is the bitfield, not the cost."""
    return {
        "clearance": int(np.count_nonzero(trav & TRAV_CLEARANCE)),
        "slope": int(np.count_nonzero(trav & TRAV_SLOPE)),
        "step": int(np.count_nonzero(trav & TRAV_STEP)),
        "roughness": int(np.count_nonzero(trav & TRAV_ROUGHNESS)),
        "class": int(np.count_nonzero(trav & TRAV_CLASS)),
        "confidence": int(np.count_nonzero(trav & TRAV_CONFIDENCE)),
        "unknown": int(np.count_nonzero(unknown)),
    }


def soft_masks(cm, w) -> dict:
    """Recover the §7.1 population from a costmap, as boolean masks.

    `CostMap` keeps the weight, not the bitfield, so the population is read
    back out of the weight arithmetic in `_cost_from_bits`: impassable is the
    three geometric bits, and each soft bit adds its own weight on top of
    `w_base`. Exact only while every SUBSET SUM of the soft weights is
    distinct, which is asserted rather than assumed -- with `w_roughness` 2,
    `w_class` 3 and `w_unknown` 4 the eight sums are 0,2,3,4,5,6,7,9.

    ⚑ The decomposition is over subsets, not over single weights, and the
      buckets are asserted exhaustive. A cell that is OBSERVED but below
      `n_min` carries `w_unknown` through bit 5 while `CostMap.unknown` stays
      False -- so a version of this that subtracted `w_unknown` only where
      `unknown` was set put those cells in no bucket at all and reported a
      total that quietly did not add up. That is the same confusion the two
      arrays cause everywhere else in this metric, which is section 5.
    """
    base = float(w.get("w_base", 1.0))
    soft_w = {"roughness": float(w.get("w_roughness", 2.0)),
              "class": float(w.get("w_class", 3.0)),
              "low_confidence": float(w.get("w_unknown", 4.0))}
    names = sorted(soft_w)
    sums = {}
    for m in range(1 << len(names)):
        picked = tuple(n for k, n in enumerate(names) if m >> k & 1)
        sums[round(sum(soft_w[n] for n in picked), 9)] = picked
    assert len(sums) == 1 << len(names), (
        "the soft weights no longer have distinct subset sums, so a cost "
        "cannot be decomposed back into bits -- read the bitfield instead")

    blocked = ~np.isfinite(cm.cost)
    soft = np.round(np.where(blocked, 0.0, cm.cost - base), 9)
    out = {n: np.zeros(cm.cost.shape, dtype=bool) for n in names}
    out["impassable"] = blocked
    out["unknown"] = np.asarray(cm.unknown, dtype=bool)
    out["plain"] = np.zeros(cm.cost.shape, dtype=bool)
    accounted = int(np.count_nonzero(blocked))
    for total, picked in sums.items():
        hit = ~blocked & (soft == total)
        accounted += int(np.count_nonzero(hit))
        if not picked:
            out["plain"] = hit
        for name in picked:
            out[name] |= hit
    assert accounted == cm.cost.size, (
        f"{cm.cost.size - accounted} cells fell into no bucket -- a cost in "
        "this window is not w_base plus a subset of the soft weights")
    return out


def bits_from_cost(cm, w) -> dict:
    """`soft_masks` as counts."""
    return {k: int(np.count_nonzero(v)) for k, v in soft_masks(cm, w).items()}


def ascii_map(cm) -> str:
    """The window, x down the page and y across, vehicle at the bottom.

    `#` impassable, `?` unknown, digits are the passable cost above `w_base`
    rounded to whole weight units, `.` is `w_base`. Small enough to paste into
    a memo, which is the point -- an argument about where to put a query is an
    argument about a picture.
    """
    base = float(weights().get("w_base", 1.0))
    rows = []
    for i in range(cm.shape[0]):
        line = []
        for j in range(cm.shape[1]):
            c = cm.cost[i, j]
            if not np.isfinite(c):
                line.append("#")
            elif cm.unknown[i, j]:
                line.append("?")
            elif abs(c - base) < 1e-9:
                line.append(".")
            else:
                line.append(str(min(9, round(c - base))))
        rows.append("".join(line))
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--map", action="store_true",
                    help="print the M* window, and each M_S, as ASCII")
    ap.add_argument("--out", default=None, help="keep the sequence here")
    args = ap.parse_args()

    th = load_thresholds()
    t = th["traversability"]
    w = weights(th)
    cell_m = float(w.get("cell_m", 0.25))

    root = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="vrgrid-syn-"))
    write_sequence(root, "99", n_frames=args.frames)
    reference = build_from_scans(read_sequence(root, "99"))

    vehicle_x = (args.frames - 1) * 2.0
    x0 = vehicle_x + sweep.PLAN_BEHIND_M
    y0 = 0.0 + sweep.PLAN_Y0_M
    x1 = x0 + sweep.PLAN_N * cell_m
    y1 = y0 + sweep.PLAN_N * cell_m
    lane_y = y0 + (sweep.PLAN_N // 2 - sweep.PLAN_LANE_CELLS) * cell_m

    print(f"frames {args.frames}  ->  vehicle at x = {vehicle_x:.1f} m")
    print(f"planning window  x [{x0:.2f}, {x1:.2f}]  y [{y0:.2f}, {y1:.2f}]  "
          f"{sweep.PLAN_N}x{sweep.PLAN_N} cells of {cell_m:.2f} m")
    print(f"current query    lane y = {lane_y:+.2f} m, straight along x "
          f"(PLAN_LANE_CELLS = {sweep.PLAN_LANE_CELLS})")
    print()

    # 1. Is each feature inside the window at all, and does it fire at the
    #    planning lattice? Both are questions about the scene, answerable from
    #    constants -- no map is consulted and no path is planned.
    print(f"1. THE SCENE'S FEATURES AT THE PLANNING LATTICE ({cell_m:.2f} m cells)")
    print("   'in window' is geometry; 'fires' is §7.1 against "
          "configs/thresholds.yaml.")
    print(f"   {'feature':<20} {'in window':>10} {'bit':>6} {'value':>9} "
          f"{'threshold':>10} {'fires':>7}")
    inside = {
        "kerb": (x0 < x1) and (y0 < -KERB_Y_M < y1 or y0 < KERB_Y_M < y1),
        "pothole": (x0 <= POTHOLE_XY_M[0] < x1) and (y0 <= POTHOLE_XY_M[1] < y1),
        "ramp": x1 > RAMP_START_X_M,
    }
    for name, bit, value, thr in features(cell_m, t):
        key = name.split()[0]
        fires = value > thr
        print(f"   {name:<20} {inside[key]!s:>10} {bit:>6} {value:>9.3f} "
              f"{thr:>10.3f} {('YES' if fires else 'no'):>7}")
    print()

    # 2. M* itself. This is the ceiling on what any query can ask: a wall that
    #    is not here cannot be decided about, whatever the start and goal.
    star = costmap_from_reference(reference, x0, y0, sweep.PLAN_N, sweep.PLAN_N)
    b = bits_from_cost(star, w)
    n_cells = star.cost.size
    print("2. M*, THE REFERENCE COSTMAP OVER THAT WINDOW")
    print(f"   {n_cells} cells:  impassable {b['impassable']}  "
          f"unknown {b['unknown']}  below n_min {b['low_confidence']}  "
          f"roughness {b['roughness']}  class {b['class']}  "
          f"plain w_base {b['plain']}")
    finite = star.cost[np.isfinite(star.cost)]
    vals, counts = np.unique(np.round(finite, 6), return_counts=True)
    print("   passable cost spread: "
          + "  ".join(f"{v:.2f}x{c}" for v, c in zip(vals, counts)))
    if len(vals) == 1:
        print("   ⚑ ONE cost value. Every passable route through this window "
              "costs the same per")
        print("     metre, so the only decision available is around a wall. A "
              "graded curve needs")
        print("     a graded cost field and this window does not have one.")
    print()

    # 3. What each schedule keeps. The claim is that coarsening loses a feature
    #    the fine map holds; if it does not, the flat curve is the true answer.
    schedules = ([load(n) for n in sweep.SCHEDULES]
                 + [uniform_schedule(c, half_width_m=24.0)
                    for c in sweep.UNIFORM_CELLS_M])
    print("3. WHAT SURVIVES INTO EACH M_S")
    print(f"   {'schedule':<14} {'impassable':>11} {'<n_min':>8} "
          f"{'class':>7} {'rough':>7}   agreement with M* on the wall set")
    mines = []
    for schedule in schedules:
        gm = build_gridmap(schedule)
        tracks = TrackList(gm.allocation.max_tracks, arrays=gm.allocation.tracks)
        run_sequence(gm, sweep.vehicle_frame_scans(root, "99"), tracks=tracks)
        mine = costmap_from_gridmap(gm, x0, y0, sweep.PLAN_N, sweep.PLAN_N,
                                    vehicle_xy_m=(vehicle_x, 0.0))
        mines.append((schedule.name, mine))
        mb = bits_from_cost(mine, w)
        wall_star = ~np.isfinite(star.cost)
        wall_mine = ~np.isfinite(mine.cost)
        both = int(np.count_nonzero(wall_star & wall_mine))
        only_star = int(np.count_nonzero(wall_star & ~wall_mine))
        only_mine = int(np.count_nonzero(~wall_star & wall_mine))
        print(f"   {schedule.name:<14} {mb['impassable']:>11} "
              f"{mb['low_confidence']:>8} {mb['class']:>7} {mb['roughness']:>7}   "
              f"both {both}, M* only {only_star}, M_S only {only_mine}")
    print()

    # 4. The asymmetry that decides whether a lateral query is available.
    print("4. BIT 4 (CLASS) IS ON ONE SIDE OF eq. (23) ONLY")
    star_class = bits_from_cost(star, w)["class"]
    mine_class = max(bits_from_cost(m, w)["class"] for _, m in mines)
    print(f"   M* cells carrying a class penalty:  {star_class}")
    print(f"   M_S cells carrying one (worst schedule): {mine_class}")
    if star_class == 0 and mine_class > 0:
        print("   ⚑ Both sides now evaluate bit 4: `costmap_from_reference` "
              "reads")
        print("     `ReferenceMap.block_class` and maps its RAW ids through "
              "`learning_ids`")
        print("     before comparing against the drivable set. Until 2 Sep it "
              "read heights")
        print("     only, so M* charged nothing and every labelled cell in a "
              "schedule was")
        print("     w_class of pure regret. A residual difference here is a "
              "real disagreement")
        print("     about semantics, not the metric asking two different "
              "questions.")
    print()

    # 5. Common support. A query is only posable where every schedule looked.
    mask = common_support(*[m for _, m in mines])
    print("5. COMMON SUPPORT, AND WHAT IT DOES NOT RESTRICT")
    print(f"   {mask.mean():.1%} of the window was observed by every schedule")
    print()
    print("   ⚑ `common_support()` masks on `CostMap.unknown`, which is "
          "NEVER-OBSERVED -- in")
    print("     `costmap_from_gridmap` it is `not seen` over 25 samples, so "
          "one observed")
    print("     sub-cell in twenty-five clears it. Bit 5 (confidence, n < "
          "n_min) is a")
    print("     different set, it is OR-ed across the same 25 samples so one "
          "thin sub-cell")
    print("     sets it, and it charges the same `w_unknown`. The mask does "
          "not remove it.")
    print(f"   {'schedule':<14} {'below n_min':>12} "
          f"{'still inside the mask':>22} {'of the mask':>12}")
    kept = max(1, int(np.count_nonzero(mask)))
    for name, mine in mines:
        low = soft_masks(mine, w)["low_confidence"]
        inside = int(np.count_nonzero(low & mask))
        print(f"   {name:<14} {low.mean():>11.1%} {inside:>22,} "
              f"{inside / kept:>11.1%}")
    print("   A schedule paying `w_unknown` on most of the restricted window "
          "is not being")
    print("   measured for coarsening. `PlanResult.unknown_fraction` reads "
          "`unknown`, not")
    print("   this, so it reports the first column of section 3 rather than "
          "this one.")
    lane_j = sweep.PLAN_N // 2 - sweep.PLAN_LANE_CELLS
    print(f"   current query's lane (j={lane_j}, y={lane_y:+.2f} m): "
          f"{mask[:, lane_j].mean():.1%} supported")
    centre_j = sweep.PLAN_N // 2
    print(f"   centreline          (j={centre_j}, y={y0 + centre_j * cell_m:+.2f} m): "
          f"{mask[:, centre_j].mean():.1%} supported")
    print()

    # 6. The mechanism behind section 3, in one table. A step of h reads as a
    #    gradient of h/2c at cell size c, and §7.1 compares that against a
    #    single tan(theta_max) whatever c is. So there is a cell size at which
    #    every step in the scene stops being a wall, and it is not the same one
    #    for the two sides of eq. (23).
    print("6. WHERE A STEP STOPS BEING A WALL, BY CELL SIZE")
    print(f"   §7.1 bit 1 fires when |grad z| > tan({t['theta_max_deg']:.0f} deg) "
          f"= {np.tan(np.radians(t['theta_max_deg'])):.3f}; bit 2 when a "
          f"4-neighbour step > {t['s_max_m']:.2f} m.")
    baseline_m = t.get("baseline_m")
    print(f"   The kerb is {KERB_HEIGHT_M:.2f} m and its step is "
          f"{KERB_HEIGHT_M:.2f} at every c. Its gradient is "
          f"{KERB_HEIGHT_M:.2f}/(2kc), where k comes from "
          f"traversability.baseline_k -- NOT from this script, so what is "
          f"printed is what §7.1 will do.")
    print(f"   {'cell size':>12}  {'k':>3} {'span':>7}  {'|grad| at kerb':>15}  "
          f"{'wall?':>6}   which map evaluates §7.1 there")
    tan_max = np.tan(np.radians(t["theta_max_deg"]))
    lattices = [(0.05, "M_S rings 0 of 5/10/20/40 and 5/10/50"),
                (0.10, "M_S uniform_10cm, ring 1"),
                (0.20, "M_S uniform_20cm, ring 2"),
                (cell_m, "⚑ M* -- costmap_from_reference blocks to plan.cell_m"),
                (0.40, "M_S uniform_40cm, ring 3"),
                (0.80, "M_S uniform_80cm")]
    walls = set()
    for c, who in sorted(lattices):
        k = baseline_k(c, baseline_m)
        span = 2.0 * k * c
        g = KERB_HEIGHT_M / span
        is_wall = g > tan_max
        walls.add(is_wall)
        print(f"   {c:>11.2f}m  {k:>3} {span:>6.2f}m  {g:>15.3f}  "
              f"{('WALL' if is_wall else '-'):>6}   {who}")
    print(f"   The kerb's STEP never fires: {KERB_HEIGHT_M:.2f} m < "
          f"{t['s_max_m']:.2f} m at every cell size.")
    if len(walls) == 1:
        verdict = "WALL on every lattice" if walls.pop() else "passable on every lattice"
        print(f"   ✔ The kerb now reads {verdict}, so eq. (23) is not charging a")
        print("     schedule for RESOLVING it against a reference that cannot see it.")
        print(f"     baseline_m = {baseline_m:.2f} m is what removes the scale: every "
              f"lattice at or")
        print("     below half of it differences over the same physical distance.")
    else:
        print("   ⚑ The kerb is still a wall on some lattices and not on others, so a")
        print("     schedule is charged regret for RESOLVING it against a reference")
        print("     that cannot see it. That is not the fill-rate confound in")
        print("     eval/plan_regret.py -- `common_support()` does not touch it.")
    print()

    if args.map:
        print("x increases DOWN, y increases RIGHT; the vehicle is at the "
              "bottom edge.")
        print("'#' impassable, '?' unknown, '.' w_base, digits are added weight.")
        print()
        print("M*:")
        print(ascii_map(star))
        for name, mine in mines:
            print()
            print(f"M_S {name}:")
            print(ascii_map(mine))
        print()

    print("No R(S) was computed anywhere above, deliberately. See the head of "
          "this file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
