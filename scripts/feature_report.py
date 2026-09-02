#!/usr/bin/env python3
"""Curbs and potholes found by the map, per ring. Math §7.4. [Shrestha]

    python scripts/feature_report.py                 # synthetic scene
    python scripts/feature_report.py --seq 08        # real, needs VRGRID_DATA_ROOT

The problem statement names curbs and potholes as the reason a 2D occupancy
grid is not enough -- it says a 2D grid "loses critical height information
necessary for detecting curbs, potholes". This is the script that answers that
sentence with numbers.

Both paths run the SAME pipeline: `build_gridmap` -> `run_sequence` ->
`traversability.update` -> `features.detect`. The only thing `--seq` changes is
where the scans come from, which is the seam `perception.loader` slots into --
`vehicle_frame_scans` already yields exactly what `loader.scans` yields.

⚑ On the synthetic scene the answer is KNOWN, which is the point of running it
  there first: `eval/synthetic.py` builds kerbs of 12 cm at |y| = 3 m and one
  pothole 40 cm deep and 60 cm across at (18, 0). A detector that reports curb
  heights clustered anywhere but 12 cm, or a depth anywhere but 40 cm, is
  wrong in a way no count of detections would show.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_synthetic as sweep
from vrgrid.eval.harness import (
    _update_traversability,
    build_gridmap,
    load,
    run_sequence,
)
from vrgrid.eval.synthetic import (
    KERB_HEIGHT_M,
    POTHOLE_DEPTH_M,
    write_sequence,
)
from vrgrid.grid.features import detect
from vrgrid.grid.transient import TrackList


def real_scans(sequence, max_frames):
    """The real path. `loader.scans` already yields (points, raw labels, pose)
    in the loader's own frame, so this mirrors `vehicle_frame_scans` exactly --
    same tuple, same raw ids, same `is_ground` convention."""
    from vrgrid.perception import loader
    for pts, labels, pose in loader.scans(sequence, max_frames=max_frames):
        yield pts, labels, np.ones(len(pts), dtype=bool), pose


def summarise(name, hits, value_cm, expect_cm=None):
    if not len(hits):
        print(f"  {name:<10} none")
        return
    v = np.asarray(value_cm, dtype=np.float64)
    line = (f"  {name:<10} {len(hits):>6,} cells   "
            f"median {np.median(v):>6.1f} cm   "
            f"p10 {np.percentile(v, 10):>5.1f}   p90 {np.percentile(v, 90):>5.1f}")
    if expect_cm is not None:
        err = abs(np.median(v) - expect_cm)
        line += f"   vs {expect_cm:.0f} cm expected  ({'OK' if err <= 3 else 'OFF'})"
    print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", default="99",
                    help="'99' = synthetic (default); 07/08 = real data")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--schedule", default="5/10/20/40")
    args = ap.parse_args()

    synthetic = args.seq == "99"
    if not synthetic and not os.environ.get("VRGRID_DATA_ROOT"):
        print("VRGRID_DATA_ROOT is not set; real sequences cannot be located.",
              file=sys.stderr)
        return 2

    schedule = load(args.schedule)
    gm = build_gridmap(schedule)
    tracks = TrackList(gm.allocation.max_tracks, arrays=gm.allocation.tracks)

    if synthetic:
        root = Path(tempfile.mkdtemp(prefix="vrgrid-feat-"))
        write_sequence(root, "99", n_frames=args.frames)
        scans = sweep.vehicle_frame_scans(root, "99")
        print(f"synthetic scene, {args.frames} frames "
              f"(kerbs {KERB_HEIGHT_M*100:.0f} cm at |y|=3 m, "
              f"pothole {POTHOLE_DEPTH_M*100:.0f} cm at (18, 0))")
    else:
        scans = real_scans(args.seq, args.frames)
        print(f"sequence {args.seq}, {args.frames} frames, "
              f"root {os.environ['VRGRID_DATA_ROOT']}")

    stats = run_sequence(gm, scans, tracks=tracks)
    _update_traversability(gm)
    rings = [(slice(r.offset, r.offset + r.side * r.side), r.side)
             for r in gm.allocation.rings]
    curbs, holes = detect(gm.soa, gm.schedule, rings, gm.thresholds,
                          buffers=gm.buffers)

    print(f"schedule {args.schedule}, {stats.frames} frames ingested\n")
    print(f"{'ring':>4} {'cell':>6}   curbs / potholes, by ring")
    print("-" * 74)
    for level, (c, h) in enumerate(zip(curbs, holes)):
        cell = schedule.rings[level].cell_m
        print(f"{level:>4} {cell*100:>5.0f}c")
        summarise("curb", c, c.height_cm,
                  KERB_HEIGHT_M * 100 if synthetic and len(c) else None)
        summarise("pothole", h, h.depth_cm,
                  POTHOLE_DEPTH_M * 100 if synthetic and len(h) else None)

    tot_c = sum(len(c) for c in curbs)
    tot_h = sum(len(h) for h in holes)
    print("-" * 74)
    print(f"total: {tot_c:,} curb cells, {tot_h:,} pothole cells")
    print()
    print("Curbs are FEATURES, not traversability bits: §7.1 after eq. (22a)")
    print("calls a 12 cm kerb passable, and it is -- a wheel can climb it. It")
    print("still bounds the drivable corridor, which is what this reports.")
    if not synthetic:
        print()
        print("⚑ Real data: there is no ground truth for curb geometry in")
        print("  SemanticKITTI, so the `road`/`sidewalk` label boundary is the")
        print("  only cross-check available. Report the counts WITH that")
        print("  caveat rather than as a measured detection rate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
