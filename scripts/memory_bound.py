#!/usr/bin/env python3
"""Print the itemised, preallocated memory bound. [Shrestha]

Gate 6: every number on a slide comes from a script in here. This one reads the
budget off the real allocation, so the slide cannot claim a bound the code does
not actually hold.

    python scripts/memory_bound.py
    python scripts/memory_bound.py --schedule 5/10/50 --transient-rings 2
"""

import argparse

import yaml
from vrgrid.gpu.allocators import allocate, bytes_allocated, measured_bytes
from vrgrid.grid.schedule import load

PYRAMID_FRACTION = 5 / 12 / 3  # 5 of 12 bytes, 4-ary pyramid adds 1/3. Math §7.3.


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", default="5/10/20/40")
    ap.add_argument("--transient-rings", type=int, default=None)
    ap.add_argument("--max-tracks", type=int, default=256)
    ap.add_argument("--storage", default="toroidal", choices=["toroidal", "annulus"])
    args = ap.parse_args()

    with open("configs/thresholds.yaml") as f:
        thresholds = yaml.safe_load(f)

    sched = load(args.schedule)
    a = allocate(sched, thresholds, transient_rings=args.transient_rings,
                 max_tracks=args.max_tracks, storage=args.storage)

    print(a.report())
    assert bytes_allocated(a) == measured_bytes(a), "claimed bound != measured bytes"

    pyramid = sched.total_cells * 12 * PYRAMID_FRACTION
    print(f"\n  {'+ conservative pyramid (stretch)':<34} {pyramid / 1e6:>8.2f} MB")
    print(f"  {'TOTAL with pyramid':<34} {(a.total_bytes() + pyramid) / 1e6:>8.2f} MB")

    print(f"\n{a.logical_cells:,} logical cells, {a.allocated_slots:,} allocated slots.")
    print("Ratios in the report are cell-count ratios over the logical count, so")
    print("toroidal padding changes the absolute MB figure and no ratio.")
    print("\nEvery line is preallocated at startup and fixed for the run.")
    print("Nothing here scales with point count, dynamic-object count or frame index.")


if __name__ == "__main__":
    main()
