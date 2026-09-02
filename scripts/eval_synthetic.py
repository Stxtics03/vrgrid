"""End-to-end evaluation on a synthetic sequence. [Aakash]

    python scripts/eval_synthetic.py [--frames 12] [--keep]

Gate 6 says every number on a slide comes from a script. This is the script
for the per-ring table, running the whole chain with no data and no network:

    synthetic sequence on disk  ->  reference map M* (§9.1)
                                ->  scatter + fuse per schedule (§3)
                                ->  traversability bitfield (§7.1)
                                ->  per-ring RMSE / rho / IoU / fill (§9.2-9.3)

⚑ The numbers it prints are NOT reportable. The terrain is analytic, so there
  is no sensor noise, no occlusion and no registration error; what is measured
  is the pipeline against a surface it can in principle recover exactly. It is
  the right thing to develop against and the wrong thing to put on a slide.
  Swap `read_sequence` for `perception.loader` and sequence 07 when the
  download lands, and the same script prints reportable numbers.
"""

import argparse
import dataclasses
import shutil
import tempfile
from pathlib import Path

import numpy as np
from vrgrid.eval.harness import (
    build_gridmap,
    evaluate,
    format_result,
    memory_vs_regret_row,
    run_sequence,
    uniform_schedule,
)
from vrgrid.eval.plan_regret import (
    common_support,
    costmap_from_gridmap,
    costmap_from_reference,
    regret,
    restrict,
)
from vrgrid.eval.reference_map import build_from_scans
from vrgrid.eval.synthetic import read_sequence, write_sequence
from vrgrid.grid.schedule import load
from vrgrid.grid.transient import TrackList

SCHEDULES = ["5/10/20/40", "5/10/50"]

# §8.2 sweeps "5/10/20/40, 5/10/50, uniform 5, uniform 10, uniform 20, ...".
# The uniform points are what give the curve a knee: the two frozen schedules
# share rings 0 and 1 and differ only past 25 m, so a near-field planning
# problem reports the same regret for both.
UNIFORM_CELLS_M = [0.10, 0.20, 0.40, 0.80]


def vehicle_frame_scans(root, sequence, keep_moving=False):
    """(points, RAW label ids, is_ground, pose) per frame, in vehicle frame.

    The synthetic writer stores points already in vehicle frame with the pose
    separately, which is what a real loader gives you too -- so this is the
    seam `perception.loader` slots into unchanged.

    RAW ids, not learning ids: `moving-*` (250-259) is what the transient
    layer separates on, and the 19-class collapse destroys exactly that.
    The pipeline routes dynamic returns away from the persistent map itself
    now -- this used to strip them by hand, which was a stand-in for the
    transient layer and stopped being possible the moment real data landed.

    `--keep-moving` bypasses the separation to show what it is worth: on this
    sequence ring 1's RMSE goes from 0.48 cm to 11.71 cm, its entire error
    budget, from one car 12 m ahead.
    """
    for pts, labels, pose in read_sequence(root, sequence):
        # Raw `car` (10), not raw `outlier` (1). `--keep-moving` is meant to
        # show what the transient layer is worth, so the car has to survive as
        # the thing it is; folding it onto an id the 19-class map sends to
        # ignore would answer a different question.
        out = np.where(labels >= 250, 10, labels) if keep_moving else labels
        yield pts, out, np.ones(len(pts), dtype=bool), pose


# The planning problem the regret is measured on. Placed RELATIVE to the
# vehicle's final pose, and behind it: that is the road the sequence has
# actually driven over, so the map has been filled by ego-motion (§1.3) rather
# than by one sparse sweep.
#
# ⚑ This placement is a measurement decision, not a detail. Put the window
#   ahead of the final pose and most of it has been seen once, at range, at
#   P_fill < 2%; the fine rings are then mostly UNKNOWN, they pay w_unknown,
#   and the regret measures how sparse the map is rather than what the
#   coarsening cost. It inverts the result -- a coarse grid whose big cells
#   each caught a return scores BETTER than a fine one full of holes. The
#   `unknown` column is what makes that visible, which is why it is printed
#   next to R(S) and not buried.
PLAN_BEHIND_M, PLAN_N = -11.0, 44
PLAN_Y0_M = -5.5

# Start and goal sit in a LANE, not on the centreline. The synthetic car drives
# down the middle of a 6 m road, so the ground under its track is never observed
# statically in a short sequence -- it drops out of the common support and the
# centreline corridor is severed. That is not an artefact to hide: it is what a
# vehicle in front of you does to a map, and planning in the free lane is what a
# real planner does about it.
PLAN_LANE_CELLS = 6


def costmaps_for(gm, reference, vehicle_xy_m):
    """(M*, M_S) on one shared planning lattice. M_S through query() only.

    `vehicle_xy_m` is the vehicle's final world position, `(x, y)`. A float is
    accepted and read as `(x, 0.0)` -- the synthetic sequence drives straight
    down y = 0, and every caller here predates real data.

    ⚑ The y was hardcoded to `PLAN_Y0_M` about the world origin, which is only
      the vehicle's lane while the trajectory is a straight line along +x. On
      a real KITTI sequence the car turns, so the window stayed near the
      origin while the vehicle drove away from it: the regret would have been
      measured over ground the map never saw, on both maps equally, and come
      out as a confident zero. Worth being explicit about, because that
      failure produces a *better-looking* number than the truth.
    """
    vx, vy = ((float(vehicle_xy_m), 0.0) if np.isscalar(vehicle_xy_m)
              else (float(vehicle_xy_m[0]), float(vehicle_xy_m[1])))
    x0 = vx + PLAN_BEHIND_M
    y0 = vy + PLAN_Y0_M
    return (costmap_from_reference(reference, x0, y0, PLAN_N, PLAN_N),
            costmap_from_gridmap(gm, x0, y0, PLAN_N, PLAN_N,
                                 vehicle_xy_m=(vx, vy)))


#: How many planning queries R(S) is averaged over. ONE was not an estimator.
PLAN_QUERIES = 64


PLAN_QUERY_FAMILIES = ("longitudinal", "lateral")


def plan_queries(n_queries: int, seed: int = 0, family: str = "longitudinal"):
    """Start/goal pairs spanning the planning window, deterministically.

    ⚑ R(S) FROM A SINGLE QUERY IS NOT AN ESTIMATE. A plan is discrete: one
      §7.1 bit flips and the path jumps a whole cell, so the statistic has no
      continuity and a single sample has enormous variance. Measured on real
      sequence 08, the same schedule by window length --

          5_10_20_40    2.207 -> 0.207 -> 0.000 -> 0.414   (20/40/80/160 frames)
          uniform_80cm  3.293 ->   inf -> 0.207 -> 0.000

      -- not monotone, not stable, and the ordering BETWEEN schedules inverts
      with the frame count. A figure whose conclusion you can choose by picking
      a window length is not measuring what it claims.

      Averaging over many queries is what turns it into an estimator with a
      spread you can quote. It does not make the underlying quantity less
      discrete; it makes the REPORTED number a mean of many discrete draws
      rather than one of them.

    ⚑ This does NOT touch PLAN_LANE_CELLS or the single-lane query, which are
      Pratyushi's parked design decision (docs/decisions-2026-09-02.md,
      Decision 4). The lane query is still query 0 of the set, so every
      previous single-query number remains reproducible as
      `plan_queries(1)[0]`.

    Seeded, because the determinism test is CI-blocking and a regret that
    moves between runs of the same map is worse than one that is noisy.
    """
    if family not in PLAN_QUERY_FAMILIES:
        raise ValueError(f"family must be one of {PLAN_QUERY_FAMILIES}, "
                         f"not {family!r}")
    rng = np.random.default_rng(seed)
    edge = 1
    out = []

    if family == "longitudinal":
        # Along the lane. Query 0 is the historical one, unchanged, so every
        # single-query number ever reported stays reproducible.
        lane = PLAN_N // 2 - PLAN_LANE_CELLS
        out.append(((1, lane), (PLAN_N - 2, lane)))
        while len(out) < n_queries:
            a = int(rng.integers(edge, PLAN_N - edge))
            b = int(rng.integers(edge, PLAN_N - edge))
            out.append(((edge, a), (PLAN_N - 1 - edge, b)))
        return out

    # ⚑ ACROSS the lane -- road to verge, the direction that crosses the kerb.
    #   The longitudinal family cannot discriminate between resolutions and it
    #   is structural, not bad luck: a query that runs the length of one lane
    #   rewards a map that is uniformly adequate along a line, and a foveated
    #   map's advantage is that it is SHARP WHERE THE VEHICLE IS LOOKING, which
    #   a line down the middle never tests. Measured on seq 08 at matched
    #   extent, every schedule from 10 cm to 40 cm scored between 0.231 and
    #   0.488 on the lane query with the ordering inverted against cell size.
    #
    #   A lateral query crosses the kerb, which is the one feature in the scene
    #   whose representation actually depends on cell size (§7.4: a 12 cm kerb
    #   at 5 cm resolves, at 40 cm averages away). It is also only a FAIR test
    #   since 2 Sep, when §7.1 bit 4 was put on both sides of eq. (23) -- before
    #   that a lateral query was measured almost entirely through an asymmetry,
    #   because crossing the kerb is exactly where the class penalty lives.
    while len(out) < n_queries:
        a = int(rng.integers(edge, PLAN_N - edge))
        b = int(rng.integers(edge, PLAN_N - edge))
        out.append(((a, edge), (b, PLAN_N - 1 - edge)))
    return out


def plan_regret_for(gm, reference, vehicle_xy_m, mask=None,
                    n_queries: int = PLAN_QUERIES,
                    family: str = "longitudinal"):
    """R(S) for one map, averaged over `n_queries` planning problems. §8.1.

    `mask` restricts both maps to the common support -- ground every schedule
    in the comparison observed, at comparable evidence. Without it the number
    measures fill rate rather than coarsening.

    Returns the median-regret query's `Regret`, with `regret` replaced by the
    mean over all queries that found a path and `n_queries`/`n_found`/`spread`
    attached. Median rather than mean for the representative query, because an
    `inf` -- a plan into a wall -- must not be averaged away, and it is counted
    separately in `blocked_fraction`.
    """
    star, mine = costmaps_for(gm, reference, vehicle_xy_m)
    if mask is not None:
        star, mine = restrict(star, mask), restrict(mine, mask)

    results = [regret(star, mine, s, g)
               for s, g in plan_queries(n_queries, family=family)]
    found = [r for r in results if r.found]
    finite = [r.regret for r in found if np.isfinite(r.regret)]
    blocked = sum(1 for r in found if not np.isfinite(r.regret))

    rep = found[len(found) // 2] if found else results[0]
    rep = dataclasses.replace(
        rep,
        regret=float(np.mean(finite)) if finite else float("inf"),
        frechet_m=float(np.mean([r.frechet_m for r in found])) if found else float("nan"),
    )
    rep.n_queries = len(results)
    rep.n_found = len(found)
    rep.n_blocked = blocked
    rep.regret_sd = float(np.std(finite)) if len(finite) > 1 else 0.0
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--seq", default=None,
                    help="a REAL SemanticKITTI sequence (needs "
                         "$VRGRID_DATA_ROOT). Without it, the synthetic "
                         "writer -- whose numbers are NOT reportable.")
    ap.add_argument("--out", default=None, help="keep the sequence here")
    ap.add_argument("--keep-moving", action="store_true",
                    help="do not strip moving-* before scatter; shows what the "
                         "missing transient layer costs")
    ap.add_argument("--confound", action="store_true",
                    help="also print R(S) WITHOUT the common-support "
                         "restriction, next to how much of each planning "
                         "window was low-confidence -- the table in "
                         "eval/plan_regret.py's confound note")
    args = ap.parse_args()

    root = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="vrgrid-syn-"))
    try:
        if args.seq:
            # The swap this file's docstring has described since Day 0. Same
            # `build_from_scans` and same `run_sequence` -- only the source of
            # the scans changes, which is the whole point of that seam.
            from vrgrid.eval.harness import final_vehicle_xy, real_scans
            print(f"sequence {args.seq}: {args.frames} frames, real data")
            # 4-tuple: M* gets the SAME ground mask the map is built with.
            # Without it M* averages building facades into the road surface --
            # +139.86 cm of bias on seq 07, against a coarsening error that
            # should be sub-centimetre.
            reference = build_from_scans(real_scans(args.seq, args.frames))
            print(f"reference map:      {reference}\n")
            # ⚑ NOT `(frames - 1) * 2.0`. That is the synthetic car driving
            #   straight down y = 0; a real one turns, and `costmaps_for`'s own
            #   note says a window placed about the origin then measures ground
            #   the map never saw and reports a confident zero.
            vehicle_x = final_vehicle_xy(args.seq, args.frames)
        else:
            write_sequence(root, "99", n_frames=args.frames)
            print(f"synthetic sequence: {args.frames} frames in {root}")
            reference = build_from_scans(read_sequence(root, "99"))
            print(f"reference map:      {reference}\n")
            vehicle_x = (args.frames - 1) * 2.0
        schedules = ([load(n) for n in SCHEDULES]
                     + [uniform_schedule(c, half_width_m=24.0)
                        for c in UNIFORM_CELLS_M])

        # Two passes. Every map is built first so the common support -- ground
        # EVERY schedule observed -- is known before anything is scored. A
        # cross-schedule regret computed without it measures fill rate.
        built = []
        for schedule in schedules:
            gm = build_gridmap(schedule)
            tracks = TrackList(gm.allocation.max_tracks,
                               arrays=gm.allocation.tracks)
            scans = (real_scans(args.seq, args.frames) if args.seq
                     else vehicle_frame_scans(root, "99", args.keep_moving))
            stats = run_sequence(gm, scans, tracks=tracks)
            built.append((schedule, gm, evaluate(gm, reference, stats.frames), stats))

        mask = common_support(*[costmaps_for(gm, reference, vehicle_x)[1]
                                for _, gm, _, _ in built])
        print(f"common support: {mask.mean():.1%} of the planning window was "
              f"observed by every schedule")
        print()

        rows, unrestricted = [], []
        for schedule, gm, result, stats in built:
            if args.confound:
                _, mine = costmaps_for(gm, reference, vehicle_x)
                raw_u = plan_regret_for(gm, reference, vehicle_x)
                # `low_confidence()`, NOT `.unknown`. The cost function
                # charges w_unknown for `unknown | TRAV_CONFIDENCE`, and the
                # second term is almost all of it -- this printed 0.0% where
                # the real figure was 100.0%, which is worse than no
                # diagnostic: it was added to catch this and said it was
                # absent.
                unrestricted.append((schedule.name, raw_u,
                                     float(np.mean(mine.low_confidence()))))
            if schedule.name.startswith("uniform"):
                rows.append((result, plan_regret_for(gm, reference, vehicle_x, mask)))
                continue
            print(format_result(result, schedule))
            print(f"  transient: {stats.dynamic_points:,} dynamic returns routed "
                  f"out of the persistent map, {stats.tracks} tracks alive; "
                  f"§9.4 DR={stats.removal['DR']:.2f} SP={stats.removal['SP']:.2f} "
                  f"F={stats.removal['F']:.2f}")
            print(f"  gate:      fired {stats.gate_fired}, acquired "
                  f"{stats.gate_acquired}, refused {stats.gate_refused}, "
                  f"released {stats.gate_released}; pool "
                  f"{gm.pool.blocks - gm.pool.free_blocks}/{gm.pool.blocks} blocks")
            reg = plan_regret_for(gm, reference, vehicle_x, mask)
            raw = plan_regret_for(gm, reference, vehicle_x)
            print(f"  plan regret R(S) = {reg.regret:.3f} on the common support   "
                  f"({raw.regret:.3f} unrestricted, path {raw.unknown_fraction:.0%} "
                  f"unknown -- that number measures fill rate, not coarsening)")
            print()
            rows.append((result, reg))

        print("§8.2, the money plot: memory on x, plan regret on y.")
        print("  R(S) = J_M*(pi_S) - J_M*(pi*), BOTH paths scored on M*.")
        print(f"  {'schedule':<12} {'MB':>7} {'cells':>10} {'RMSE':>7} {'rho':>6} "
              f"{'R(S)':>8} {'frechet':>8} {'unknown':>8}")
        for r, reg in rows:
            m = memory_vs_regret_row(r, reg)
            blocked = " BLOCKED" if m["blocked_on_reference"] else ""
            print(f"  {m['schedule']:<12} {m['megabytes']:>7.2f} "
                  f"{m['logical_cells']:>10,} {m['worst_ring_rmse_cm']:>6.2f}c "
                  f"{m['mean_rho']:>6.2f} {m['regret']:>8.3f} "
                  f"{m['frechet_m']:>7.2f}m {m['unknown_fraction']:>7.1%}{blocked}")
        if unrestricted:
            print()
            print("The confound, unrestricted. Math §8.2, and the note in "
                  "eval/plan_regret.py.")
            print("  R(S) here is NOT on the common support, so it is not "
                  "comparable across")
            print("  cell sizes -- that is the whole point of printing it.")
            print(f"  {'schedule':<14} {'R(S)':>8} {'window low-confidence':>22}")
            for name, raw_u, low in unrestricted:
                print(f"  {name:<14} {raw_u.regret:>8.3f} {low:>21.0%}")
            print()
            print("  A finer schedule holds fewer returns per cell, so more of "
                  "its window is")
            print("  below n_min; it pays w_unknown and the planner routes "
                  "around a map that")
            print("  is merely SPARSE. Read across this table and finer looks "
                  "worse, which is")
            print("  precisely backwards. `common_support()` is the fix and the "
                  "money plot")
            print("  above uses it.")

        print()
        print("  R(S) = 0 means the coarsening did not change the decision -- the")
        print("  only sense in which a saving is free. Read it WITH `unknown`:")
        print("  zero regret along a mostly-unknown path says the sequence was too")
        print("  short to fill the map, not that the schedule cost nothing.")
    finally:
        if args.out is None:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
