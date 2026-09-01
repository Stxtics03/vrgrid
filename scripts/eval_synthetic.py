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


def costmaps_for(gm, reference, vehicle_x_m):
    """(M*, M_S) on one shared planning lattice. M_S through query() only."""
    x0 = vehicle_x_m + PLAN_BEHIND_M
    return (costmap_from_reference(reference, x0, PLAN_Y0_M, PLAN_N, PLAN_N),
            costmap_from_gridmap(gm, x0, PLAN_Y0_M, PLAN_N, PLAN_N,
                                 vehicle_xy_m=(vehicle_x_m, 0.0)))


def plan_regret_for(gm, reference, vehicle_x_m, mask=None):
    """R(S) for one map, through query() only. Math §8.1.

    `mask` restricts both maps to the common support -- ground every schedule
    in the comparison observed. Without it the number measures fill rate
    rather than coarsening; see the confound note in eval/plan_regret.py.
    """
    star, mine = costmaps_for(gm, reference, vehicle_x_m)
    if mask is not None:
        star, mine = restrict(star, mask), restrict(mine, mask)
    lane = PLAN_N // 2 - PLAN_LANE_CELLS
    return regret(star, mine, (1, lane), (PLAN_N - 2, lane))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=12)
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
            stats = run_sequence(
                gm, vehicle_frame_scans(root, "99", args.keep_moving),
                tracks=tracks)
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
                unrestricted.append((schedule.name, raw_u,
                                     float(np.mean(mine.unknown))))
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
