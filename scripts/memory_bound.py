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

    # Allocated for real rather than estimated. It used to be estimated as
    # `total_cells * 12 * 5/12/3`, straight from §7.2, and that printed
    # 1.24 MB -- low by more than half. A node stores the REDUCTIONS, not the
    # source fields (ground is both H_max and H_min), and the pyramid covers
    # the allocated SLOTS, not the logical cells. See docs/sih-math.md §7.2.
    def variant(**kw):
        return allocate(sched, thresholds, transient_rings=args.transient_rings,
                        max_tracks=args.max_tracks, storage=args.storage, **kw)

    with_p = variant(with_pyramid=True)
    with_v = variant(with_visibility=True)
    both = variant(with_pyramid=True, with_visibility=True)
    for handle in (with_p, with_v, both):
        assert bytes_allocated(handle) == measured_bytes(handle), \
            "claimed bound != measured bytes"

    # Resolved, not raw: the shipped value is `null` since 2 Sep, meaning the
    # grid's own slot count. Every consumer goes through the same resolver so
    # the printed bound and the allocated scratch cannot disagree.
    from vrgrid.gpu.allocators import resolve_candidate_cap
    configured = thresholds.get("visibility", {}).get("max_candidate_cells")
    cap = resolve_candidate_cap(configured, a.allocated_slots)
    print(f"\n  {'+ conservative pyramid (stretch)':<34} "
          f"{(with_p.total_bytes() - a.total_bytes()) / 1e6:>8.2f} MB")
    print(f"  {'+ visibility scratch (§10.4)':<34} "
          f"{(with_v.total_bytes() - a.total_bytes()) / 1e6:>8.2f} MB")
    print(f"  {'-' * 34} {'-' * 8}")
    print(f"  {'TOTAL with both':<34} {both.total_bytes() / 1e6:>8.2f} MB")

    print(f"\nBoth are OFF by default and neither is in the {a.total_bytes() / 1e6:.2f} MB "
          f"above. The pyramid is a\nstretch item.")
    if configured is None:
        print(f"The visibility scratch is sized to the GRID -- {cap:,} slots, the\n"
              "structural bound. The occupied set cannot exceed the grid, so the\n"
              "cleanup can never silently drop a candidate. Measured peaks over whole\n"
              "sequences: 314,442 on 07 and 455,714 on 08, against the retired\n"
              "placeholder of 150,000 which dropped 52.3% and 67.1% of them untested.")
    else:
        print(f"The visibility scratch is sized to an EXPLICIT cap of {cap:,} candidate\n"
              "cells, which is below the grid's slot count -- so the cleanup can drop\n"
              "candidates. `StepCounters.truncated` counts them and `vrgrid.run` prints\n"
              "it; a run that reports truncation has ghost numbers that are a floor.")

    print(f"\n{a.logical_cells:,} logical cells, {a.allocated_slots:,} allocated slots.")
    print("Ratios in the report are cell-count ratios over the logical count, so")
    print("toroidal padding changes the absolute MB figure and no ratio.")
    print("\nEvery line is preallocated at startup and fixed for the run.")
    print("Nothing here scales with point count, dynamic-object count or frame index.")


if __name__ == "__main__":
    main()
