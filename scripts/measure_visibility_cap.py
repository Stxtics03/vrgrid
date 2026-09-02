#!/usr/bin/env python3
"""What should `visibility.max_candidate_cells` be? Measure, do not guess.

    python scripts/measure_visibility_cap.py --seq 07 --frames 200

The §10.4 cleanup tests the currently-OCCUPIED cells against this frame's range
image, and `visibility.max_candidate_cells` bounds how many it may test. The
cap has been a placeholder since Gate 3 for one reason: nobody could pick it
honestly without knowing how many cells are occupied at once on real data.

⚑ Getting it WRONG IS SILENT in the dangerous direction. `engine._cleanup`
  truncates with `occupied[:max_candidates]` -- deterministically, so the
  determinism test still passes -- and the cells past the cap are simply never
  tested. They keep their occupancy, so a ghost that lands beyond the cut stays
  in the map forever and the ghost counter still looks healthy. Too high only
  costs memory, at 64 B per cell, and says so in the budget line.

  So the cap is read off the MAXIMUM over the sequence with headroom, not off
  the mean, and this prints the whole distribution so the choice is visible.

Tuning sequence is 07, per the header of configs/thresholds.yaml -- tune on 07,
report on 08, never both on one. `counters.occupied` is recorded BEFORE the
truncation, so the true occupied count is measured even when the running cap is
already too small to hold it.
"""

import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", default="07", help="tuning sequence (07)")
    # ⚑ The whole sequence by default. A frame limit is a footgun here: this
    #   script exists to find a MAXIMUM, and a default that stops early
    #   reports a smaller one with no indication that it did.
    ap.add_argument("--frames", type=int, default=None,
                    help="stop after N frames (default: the whole sequence)")
    ap.add_argument("--schedule", default="5/10/20/40")
    ap.add_argument("--headroom", type=float, default=1.30,
                    help="multiplier on the observed max (default 1.30)")
    args = ap.parse_args()

    if not os.environ.get("VRGRID_DATA_ROOT"):
        print("VRGRID_DATA_ROOT is not set.", file=sys.stderr)
        return 2

    from vrgrid.eval.harness import load
    from vrgrid.gpu.visibility import visibility_scratch_bytes
    from vrgrid.grid.schedule import load_thresholds
    from vrgrid.run.__main__ import iter_pipeline
    from vrgrid.run.engine import MapEngine

    th = load_thresholds()
    schedule = load(args.schedule)
    # A cap far above anything plausible, so nothing is truncated while the
    # true occupied set is being measured. This is the one run that must not
    # be bounded by the number it is trying to choose.
    engine = MapEngine(schedule, thresholds=th, max_candidates=4_000_000)

    occupied, tested, cleared = [], [], []
    for frame in iter_pipeline(args.seq, args.frames):
        c = engine.step(frame)
        occupied.append(c.occupied)
        tested.append(c.tested)
        cleared.append(c.cleared)
        if len(occupied) % 25 == 0:
            print(f"  frame {len(occupied):>4}  occupied {c.occupied:>9,}",
                  flush=True)

    if not occupied:
        print("no frames ingested", file=sys.stderr)
        return 1

    occ = np.array(occupied)
    from vrgrid.perception import loader
    total = loader.get_frame_count(args.seq)
    span = f"{len(occ)} of {total}" if len(occ) < total else f"all {total}"
    print(f"\nsequence {args.seq}, {span} frames, schedule {args.schedule}")
    print("occupied cells offered to the cleanup, per frame:")
    for name, v in (("min", occ.min()), ("median", int(np.median(occ))),
                    ("p95", int(np.percentile(occ, 95))),
                    ("p99", int(np.percentile(occ, 99))), ("max", occ.max())):
        print(f"  {name:>7} {int(v):>10,}")
    print(f"  cleared total {sum(cleared):,} over {sum(tested):,} tested")

    # Resolved, not raw: since 2 Sep the shipped value is `null`, meaning the
    # grid's own slot count. A raw read gets None and every format below fails.
    from vrgrid.gpu.allocators import resolve_candidate_cap
    n_slots = engine.handle.grid["log_odds"].size
    configured = th["visibility"]["max_candidate_cells"]
    current = resolve_candidate_cap(configured, n_slots)
    if configured is None:
        print(f"\nconfigured cap is `null` -> the grid itself, "
              f"{n_slots:,} slots. Truncation is impossible by construction;")
        print("the numbers below say how much headroom that leaves.")
    proposed = int(np.ceil(occ.max() * args.headroom / 10_000) * 10_000)
    print(f"\ncurrent cap   {current:>10,}   "
          f"{visibility_scratch_bytes(current, np.float32)/1e6:>6.2f} MB")
    print(f"proposed      {proposed:>10,}   "
          f"{visibility_scratch_bytes(proposed, np.float32)/1e6:>6.2f} MB"
          f"   (max x {args.headroom:g}, rounded up to 10k)")
    if occ.max() > current:
        short = (occ.max() - current)
        print(f"\n⚑ The current cap TRUNCATES: {short:,} occupied cells "
              f"({short/occ.max():.1%} of the peak) are never tested by the")
        print("  cleanup on this sequence, and any ghost among them is "
              "permanent and invisible to the counter.")
    else:
        print(f"\nThe current cap holds: peak {occ.max():,} is "
              f"{occ.max()/current:.0%} of it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
