#!/usr/bin/env python3
"""Benchmark the two scatter paths against the 10 Hz sensor rate. [Shrestha]

Gate 6: the timing numbers on a slide come from here.

    python scripts/bench_scatter.py [--points 120000] [--frames 200]
"""

import argparse

import numpy as np
from vrgrid.gpu.kernels import map_hash, scatter_atomic, scatter_scratch_bytes, scatter_sorted
from vrgrid.gpu.timing import Timer

N_CELLS = 745_000


def make_scan(rng, n):
    return {
        "idx": rng.integers(-1, N_CELLS, n).astype(np.int64),
        "z_cm": rng.integers(-200, 600, n).astype(np.int16),
        "w_q": rng.integers(1, 2000, n).astype(np.int32),
        "refl": rng.integers(0, 256, n).astype(np.uint8),
        "class_id": rng.integers(0, 19, n).astype(np.uint8),
        "is_ground": rng.random(n) < 0.6,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=120_000, help="returns per sweep")
    ap.add_argument("--frames", type=int, default=200)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    scans = [make_scan(rng, args.points) for _ in range(10)]

    t = Timer(stages=("scatter",))
    hashes = {}
    for label, fn, kw in (("sorted", scatter_sorted, {}),
                          ("atomic", scatter_atomic, {"n_cells": N_CELLS})):
        fn(**scans[0], **kw)          # warm up
        t.reset()
        for i in range(args.frames):
            scan = scans[i % len(scans)]
            with t.stage("scatter"):
                agg = fn(**scan, **kw)
        hashes[label] = map_hash(agg.as_dict())
        s = t.summary()["scatter"]
        scratch = scatter_scratch_bytes(label, N_CELLS, args.points) / 1e6
        print(f"{label:7s} p50 {s['p50_ms']:6.2f} ms   p99 {s['p99_ms']:6.2f} ms   "
              f"max {s['max_ms']:6.2f} ms   scratch {scratch:5.2f} MB   "
              f"{1e3 / s['p99_ms'] / 10:.1f}x headroom at p99")

    print(f"\nboth paths agree: {hashes['sorted'] == hashes['atomic']}  ({hashes['sorted']})")
    print(f"{args.points:,} returns/sweep, {args.frames} frames, {N_CELLS:,} cells")
    print("\nCPU reference path (numpy). These orderings need not hold on the GPU:\n"
          "hardware atomics are far cheaper than np.add.at, so the atomic path may\n"
          "win on latency there. Its scratch cost does not change either way.")


if __name__ == "__main__":
    main()
