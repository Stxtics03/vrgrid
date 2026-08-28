#!/usr/bin/env python3
"""Run our map beside the baselines with all three counters ticking. [Shrestha]

The Day-4 gate: "allocating 3D baseline running, both counters ticking on
screen." This is that, on a terminal; JP's dashboard reads the same
`Counters` objects out of `vrgrid.gpu.baseline` so the two cannot disagree.

    python scripts/baseline_demo.py                  # 192 MB uniform baseline
    python scripts/baseline_demo.py --dense          # + the 2.56 GB voxel grid
    python scripts/baseline_demo.py --dense --frames 100

The column that matters is RESIDENT, not CLAIMED. Claimed is what the slide
says; resident is what the machine gave up. They agree here because
`baseline.commit()` faults every page in -- see the module docstring for why
that is the whole point and not an implementation detail.
"""

import argparse
import time

import numpy as np
import yaml
from vrgrid.cell import CELL_BYTES
from vrgrid.gpu.allocators import allocate, annulus_index
from vrgrid.gpu.baseline import (
    allocate_dense3d,
    allocate_uniform25d,
    available_bytes,
    resident_bytes,
)
from vrgrid.gpu.kernels import quantise_height, quantise_weight, scatter_sorted
from vrgrid.grid.schedule import load

SCHEDULE = "configs/schedule_5_10_20_40.yaml"
THRESHOLDS = "configs/thresholds.yaml"


def synthetic_sweep(rng, n):
    """A sweep shaped like a LiDAR scan rather than a uniform cube: returns
    thin out with range, which is the effect the whole design exists for."""
    r = 3.74 + rng.gamma(2.0, 12.0, n)          # blind cone out, long tail
    r = np.clip(r, 3.74, 99.0)
    az = rng.uniform(-np.pi, np.pi, n)
    el = rng.uniform(np.radians(-24.8), np.radians(2.0), n)
    return (r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az),
            r * np.sin(el) + 1.73)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--points", type=int, default=120_000, help="returns per sweep")
    ap.add_argument("--dense", action="store_true",
                    help="also allocate the 2.56 GB dense 3D voxel grid")
    ap.add_argument("--every", type=int, default=10, help="print every N frames")
    args = ap.parse_args()

    schedule = load(SCHEDULE)
    with open(THRESHOLDS) as f:
        thresholds = yaml.safe_load(f)

    print(f"MemAvailable {available_bytes() / 1e9:.1f} GB, RSS at start "
          f"{resident_bytes() / 1e6:.0f} MB\n")

    t0 = time.perf_counter()
    ours = allocate(schedule, thresholds)
    print(ours.report())
    print(f"\n  allocated in {time.perf_counter() - t0:.2f} s\n")

    baselines = [allocate_uniform25d()]
    if args.dense:
        t0 = time.perf_counter()
        baselines.append(allocate_dense3d())
        print(f"  dense 3D voxel grid faulted in in {time.perf_counter() - t0:.2f} s\n")

    ours_bytes = ours.total_bytes()
    ours_map_bytes = ours.logical_cells * CELL_BYTES

    rng = np.random.default_rng(0)
    header = f"{'frame':>6}  {'ours':>10}  " + "  ".join(
        f"{b.name.split(',')[0]:>22}" for b in baselines)
    print(header)
    print("-" * len(header))

    for f in range(args.frames):
        x, y, z = synthetic_sweep(rng, args.points)

        # Ours: ring 0 only, which is all this demo needs -- it exercises the
        # scatter path so the frame loop is really running, and the memory
        # claim is about what is allocated, not about what is filled.
        r0 = ours.ring(0)
        half = r0.side // 2
        ix = np.floor(x / r0.cell_m).astype(np.int64) + half
        iy = np.floor(y / r0.cell_m).astype(np.int64) + half
        idx = annulus_index(r0, ix, iy)
        scatter_sorted(idx, quantise_height(z), quantise_weight(np.full(len(z), 4.0)),
                       np.zeros(len(z), np.uint8), np.zeros(len(z), np.uint8),
                       z < 0.2, scratch=ours.scratch)

        for b in baselines:
            b.ingest(x, y, z)

        if f % args.every == 0 or f == args.frames - 1:
            cells = "  ".join(f"{b.counters().resident_bytes / 1e6:>19,.0f} MB"
                              for b in baselines)
            print(f"{f:>6}  {ours.resident_delta / 1e6:>7,.1f} MB  {cells}")

    print()
    for b in baselines:
        print("  " + str(b.counters()))
    print(f"  {'ours (map only)':<28} {ours.logical_cells:>14,}  "
          f"claimed {ours_map_bytes / 1e6:>6.2f} MB")
    print(f"  {'ours (total preallocated)':<28} {'':>14}  "
          f"claimed {ours_bytes / 1e6:>6.2f} MB  resident "
          f"{ours.resident_delta / 1e6:>6.2f} MB")
    print(f"\n  whole process RSS {resident_bytes() / 1e6:,.0f} MB "
          f"(all three maps plus the interpreter)")

    print("\nRatios against our MAP memory, which is what the report compares:")
    for b in baselines:
        print(f"  {b.name:<28} {b.claimed_bytes / ours_map_bytes:>8.1f}x")
    print("\nAgainst our TOTAL preallocated footprint, scratch included -- the number")
    print("the dashboard counter shows, and the one a hostile question will ask for:")
    for b in baselines:
        print(f"  {b.name:<28} {b.claimed_bytes / ours_bytes:>8.1f}x")


if __name__ == "__main__":
    main()
