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

from vrgrid.eval.harness import build_gridmap, evaluate, format_result, run_sequence
from vrgrid.eval.reference_map import build_from_scans
from vrgrid.eval.synthetic import read_sequence, write_sequence
from vrgrid.grid.schedule import load

SCHEDULES = ["5/10/20/40", "5/10/50"]


def vehicle_frame_scans(root, sequence, keep_moving=False):
    """(points, class ids, is_ground, pose) per frame; points in vehicle frame.

    The synthetic writer stores points already in vehicle frame and the pose
    separately, which is what a real loader gives you too -- so this is the
    seam `perception.loader` slots into unchanged.

    Moving points are dropped by default, and that is a STAND-IN, not a
    result: §9.1 strips `moving-*` from the reference, so anything left in the
    map that the reference does not have scores as error. The transient layer
    is what removes them for real (master v4 §3.7, and §9.4 measures how well)
    and it is not wired up. `--keep-moving` shows the difference, which on
    this sequence is ring 1's entire error budget -- 0.48 cm RMSE becomes
    11.71 cm, because the car sits 12 m ahead and ring 1 owns 10-25 m.
    """
    for pts, labels, pose in read_sequence(root, sequence):
        moving = (labels >= 250) & (labels <= 259)
        if not keep_moving:
            pts, labels, moving = pts[~moving], labels[~moving], moving[~moving]
        yield pts, (labels % 16).astype("uint8"), ~moving, pose


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--out", default=None, help="keep the sequence here")
    ap.add_argument("--keep-moving", action="store_true",
                    help="do not strip moving-* before scatter; shows what the "
                         "missing transient layer costs")
    args = ap.parse_args()

    root = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="vrgrid-syn-"))
    try:
        write_sequence(root, "99", n_frames=args.frames)
        print(f"synthetic sequence: {args.frames} frames in {root}")

        reference = build_from_scans(read_sequence(root, "99"))
        print(f"reference map:      {reference}\n")

        rows = []
        for name in SCHEDULES:
            schedule = load(name)
            gm = build_gridmap(schedule)
            frames = run_sequence(
                gm, vehicle_frame_scans(root, "99", args.keep_moving))
            result = evaluate(gm, reference, frames)
            print(format_result(result, schedule))
            print()
            rows.append(result)

        print("memory vs accuracy, one row per schedule (the Day-4 curve's axes;")
        print("the regret column joins it when eval/plan_regret lands):")
        print(f"  {'schedule':<12} {'MB':>7} {'cells':>10} {'worst RMSE':>11} {'mean rho':>9}")
        for r in rows:
            from vrgrid.eval.harness import memory_vs_regret_row
            m = memory_vs_regret_row(r)
            print(f"  {m['schedule']:<12} {m['megabytes']:>7.2f} "
                  f"{m['logical_cells']:>10,} {m['worst_ring_rmse_cm']:>10.2f}c "
                  f"{m['mean_rho']:>9.2f}")
    finally:
        if args.out is None:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
