#!/usr/bin/env python3
"""Benchmark the conservative pyramid against the 10 Hz sensor rate. [Shrestha]

Gate 6: the timing and memory numbers on a slide come from here, not from
memory. Also prints the corrected §7.2 memory figure -- the section's own
1.24 MB is low by about half, for two separate reasons (see gpu/pyramid.py).

    python scripts/bench_pyramid.py [--schedule 5/10/20/40] [--frames 200]
"""

import argparse

import numpy as np
from vrgrid.gpu.allocators import allocate, bytes_allocated
from vrgrid.gpu.pyramid import (
    NODE_BYTES,
    build,
    classify,
    level_arrays,
    level_sides,
    pyramid_bytes,
    scratch_bytes,
)
from vrgrid.gpu.timing import Timer
from vrgrid.grid.schedule import load


def fill(grid, rings, rng):
    """Terrain-shaped, so the SAFE/BLOCKED/MIXED split is representative.

    A uniform-random map is ~100% MIXED above the first level, which makes the
    descent look far more expensive than it is: with heights spread over 4 m,
    no block of any size has a spread under the 15 cm step threshold. Real
    ground is locally smooth, and the whole value of a pyramid is that smooth
    ground resolves at a coarse level. The timing does not depend on this --
    the reductions are branch-free -- but the class breakdown does, and that
    is the number a slide would quote.
    """
    for layout in rings:
        side, span = layout.side, slice(layout.offset, layout.offset + layout.slots)
        iy, ix = np.mgrid[0:side, 0:side]
        z = -173.0 + 2.0 * ix + 1.4 * iy + rng.integers(-2, 3, (side, side))
        hazard = rng.random((side, side)) < 0.02
        z[hazard] += rng.integers(20, 60, int(hazard.sum()))
        grid["ground_height"][span] = z.ravel()

        obs = rng.integers(3, 20, layout.slots).astype(np.uint8)
        thin = rng.random(layout.slots) < 0.05
        obs[thin] = rng.integers(0, 3, int(thin.sum()))
        grid["obs_count"][span] = obs

        # bit 5 is the only one a synthetic map can set honestly here; the
        # rest come from traversability.update() against a real scan.
        grid["traversability"][span] = np.where(obs < 3, 1 << 5, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", default="5/10/20/40")
    ap.add_argument("--frames", type=int, default=200)
    args = ap.parse_args()

    sched = load(args.schedule)
    with_p = allocate(sched, with_pyramid=True)
    without = allocate(sched)
    fill(with_p.grid, with_p.rings, np.random.default_rng(0))

    t = Timer(stages=("pyramid",))
    build(with_p.pyramid, with_p.grid, with_p.rings)      # warm up
    for _ in range(args.frames):
        with t.stage("pyramid"):
            build(with_p.pyramid, with_p.grid, with_p.rings)

    s = t.summary()["pyramid"]
    slots = sum(r.slots for r in with_p.rings)
    print(f"schedule {args.schedule}, {slots:,} slots, {args.frames} frames\n")
    print(f"build   p50 {s['p50_ms']:6.2f} ms   p99 {s['p99_ms']:6.2f} ms   "
          f"max {s['max_ms']:6.2f} ms   {1e3 / s['p99_ms'] / 10:.0f}x headroom at p99")

    nodes = pyramid_bytes(with_p.rings)
    print(f"\nnodes   {nodes / 1e6:5.2f} MB  ({nodes // NODE_BYTES:,} nodes x "
          f"{NODE_BYTES} B)   = slots/{slots / (nodes / NODE_BYTES):.2f}")
    print(f"scratch {scratch_bytes(with_p.rings) / 1e6:5.2f} MB  (shared, all rings)")
    print(f"total   {bytes_allocated(without) / 1e6:5.2f} MB without  ->  "
          f"{bytes_allocated(with_p) / 1e6:5.2f} MB with")
    print(f"\n⚡ math §7.2 quotes 1.24 MB (745,000 x 5 / 3). A node stores the "
          f"REDUCTIONS,\n  so ground is H_max AND H_min -- {NODE_BYTES} B, not 5 -- "
          f"and the pyramid covers the\n  {slots:,} allocated slots, not 745,000 "
          f"logical cells. Corrected: {nodes / 1e6:.2f} MB.")

    print("\nnodes and query classes per level, ring 1:")
    sides = level_sides(with_p.rings[1].side)
    for level in range(1, len(sides)):
        nd = level_arrays(with_p.pyramid, with_p.grid, with_p.rings, 1, level)
        c = classify(nd)
        n = c.size
        print(f"  level {level:<2} {sides[level]:>4} x {sides[level]:<4} "
              f"{n:>7,} nodes   SAFE {np.count_nonzero(c == 0) / n:5.1%}   "
              f"BLOCKED {np.count_nonzero(c == 1) / n:5.1%}   "
              f"MIXED {np.count_nonzero(c == 2) / n:5.1%}")


if __name__ == "__main__":
    main()
