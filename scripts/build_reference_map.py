#!/usr/bin/env python3
"""Build and cache the reference map M* for a real sequence. Math §9.1. [Aakash]

    python scripts/build_reference_map.py 08 --out docs/cache/mstar-08.npz
    python scripts/build_reference_map.py 07 --max-frames 200        # a first pass

M* is the artefact every real number in this project is measured against, and
until now there were none: `reference_map.build()` -- the only path from
SemanticKITTI to M* -- had never been executed by anything, and raised
`ValueError: too many values to unpack` on its own first line. So the
"plan regret has never run on real data" item is not really about plan regret.
Nothing downstream of M* could run, because M* could not be built.

What this does NOT do is decide anything about the schedules. It writes the
cache; `scripts/eval_synthetic.py` and `scripts/regret_plot.py` consume it.
Building is the expensive step and the ablation runs the metrics once per
schedule, so it is deliberately a separate command with a file in between.

--- read the frame line before you trust the numbers ------------------------

The chain from a KITTI `.bin` to a world coordinate is four matrices
(`docs/frames.md`), and getting it wrong produces a complete, plausible map in
the wrong cells -- there is no downstream check that can tell. So this prints
the vehicle's start and end position and the median ground height, and
`reference_map.build` runs `harness.FrameGuard` over the first frame and one
frame after 10 m of travel. **The second look is the one that matters**: a
`poses.txt` starts at the identity, where the right composition and the wrong
one agree to well inside the tolerance.

--- how long, and how big ---------------------------------------------------

At 5 cm, seq 08's 4,071 frames cover roughly 3.2 km of trajectory. Use
`--max-frames` for a first pass: it answers "is the frame convention right"
in a minute rather than an hour, which is the whole reason it exists.

⚑ This needs the download. With no data it exits 2 and says which path it
  looked in, rather than failing somewhere deep in the loader.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vrgrid.eval.reference_map import build, load
from vrgrid.perception import loader


def _describe(sequence: str, max_frames: int | None) -> str:
    """Where the vehicle went, in world metres. Cheap -- poses only."""
    from vrgrid.perception.transforms import vehicle_to_world

    poses = loader.poses(sequence)
    if max_frames is not None:
        poses = poses[:max_frames]
    t = np.array([vehicle_to_world(p, sequence)[:3, 3] for p in poses])
    span = t.max(axis=0) - t.min(axis=0)
    climb = float(t[:, 2].max() - t[:, 2].min())
    return (f"  trajectory: {len(t)} frames, "
            f"start ({t[0, 0]:.1f}, {t[0, 1]:.1f}, {t[0, 2]:.1f}) -> "
            f"end ({t[-1, 0]:.1f}, {t[-1, 1]:.1f}, {t[-1, 2]:.1f}) m\n"
            f"  extent:     {span[0]:.0f} x {span[1]:.0f} m, "
            f"{climb:.1f} m of elevation change")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence", help='"00", "07" or "08"')
    ap.add_argument("--out", default=None,
                    help="cache path (default docs/cache/mstar-<seq>.npz)")
    ap.add_argument("--cell-m", type=float, default=0.05,
                    help="M* lattice, metres. The base lattice, not a schedule.")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="stop after N frames -- use for a first pass")
    ap.add_argument("--no-frame-check", action="store_true",
                    help="skip the world-frame guard. Do not.")
    args = ap.parse_args()

    if not loader.verify_sequence_exists(args.sequence):
        print(f"sequence {args.sequence} is not present under "
              f"{loader.DATA_ROOT.resolve()}\n"
              f"  expected {loader._gt_poses_path(args.sequence)}\n"
              f"       and {loader.VELODYNE_DIR / args.sequence / 'velodyne'}\n"
              f"  set $VRGRID_DATA_ROOT if the download lives elsewhere "
              f"(see data/README.md).", file=sys.stderr)
        return 2

    out = Path(args.out) if args.out else Path("docs/cache") / f"mstar-{args.sequence}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"sequence {args.sequence}, M* at {args.cell_m * 100:.0f} cm")
    print(_describe(args.sequence, args.max_frames))

    t0 = time.perf_counter()
    ref = build(args.sequence, out_path=out, cell_m=args.cell_m,
                max_frames=args.max_frames,
                check_frame=not args.no_frame_check)
    dt = time.perf_counter() - t0

    observed = ref.count > 0
    heights = ref.height_cm[observed] / 100.0
    print(f"\n  {ref}")
    print(f"  observed:   {int(observed.sum()):,} cells "
          f"({observed.mean():.1%} of the bounding box)")
    print(f"  returns:    {int(ref.count.sum()):,}, "
          f"median {np.median(ref.count[observed]):.0f} per observed cell")
    print(f"  height:     median {np.median(heights):.2f} m, "
          f"p1 {np.percentile(heights, 1):.2f} m, "
          f"p99 {np.percentile(heights, 99):.2f} m")
    print(f"  built in    {dt:.1f} s -> {out}  "
          f"({out.stat().st_size / 1e6:.1f} MB)")

    # Cheap proof the file is readable, before anything depends on it being so.
    back = load(out)
    assert back.cell_m == ref.cell_m and back.count.shape == ref.count.shape
    print("  reloaded OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
