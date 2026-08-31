#!/usr/bin/env python3
"""The Day-6 per-stage latency table. [Shrestha]

Gate 6: the latency numbers on a slide come from here, not from memory.

    python scripts/timing_table.py [--schedule 5/10/20/40] [--frames 200]

⚑ **This is a lower bound on frame latency, not the frame total.** It times
  the back half of the frame -- the four `src/gpu` stages and the `fuse()`
  that consumes the scatter's aggregate. The perception front end (load,
  transform, range_image, semantics, motion) needs SemanticKITTI on disk, and
  `src/run/__main__.py` still has `scatter`/`fuse` stubbed against Aakash's
  grid, so there is no end-to-end loop to time yet. The unmeasured stages are
  printed as rows with their owner and why, rather than omitted: a table
  showing four green rows and a 16x headroom, with the missing half silently
  dropped, is exactly the shape of a number that gets called on stage.

Percentiles come from `gpu.timing.Timer`, which uses nearest-rank -- the whole
point of the harness is that no reported latency is one that never occurred.
Headroom is FPS / 10 against the 10 Hz sensor rate, at p99 and not at p50: a
pipeline that clears 10 Hz on the median and misses one frame in a hundred has
dropped a frame of obstacles.
"""

import argparse
import platform
import subprocess

import numpy as np
from vrgrid.gpu.allocators import allocate
from vrgrid.gpu.kernels import scatter_sorted
from vrgrid.gpu.pyramid import build
from vrgrid.gpu.shift import RingBuffer, cells_per_shift, shift
from vrgrid.gpu.timing import SENSOR_HZ, Timer
from vrgrid.gpu.visibility import new_visibility_scratch, visibility_cleanup
from vrgrid.grid.fusion import fuse
from vrgrid.grid.schedule import load

# Range image geometry is locked to 64x512 sub-clouds (FLARES; research log
# 2026-09-01), so the gather in visibility_cleanup is sized off that and not
# off a full 64x2048 sweep.
IMAGE_SHAPE = (64, 512)

# Who to ask when a row moves, and -- for the unmeasured rows -- what is in
# the way. Ownership is root CLAUDE.md's, not the execution plan's.
STAGE_OWNER = {
    "load": "JP", "transform": "JP", "range_image": "JP",
    "semantics": "JP", "motion": "JP",
    "scatter": "Shrestha", "fuse": "Aakash", "split_merge": "Aakash",
    "cleanup": "Shrestha", "pyramid": "Shrestha", "shift": "Shrestha",
}

BLOCKED = {
    "load": "needs SemanticKITTI on disk",
    "transform": "needs SemanticKITTI on disk",
    "range_image": "needs SemanticKITTI on disk",
    "semantics": "needs SemanticKITTI on disk",
    "motion": "needs SemanticKITTI on disk",
    "split_merge": "per-cell API; driven by the refinement pool, no frame batch",
}

MEASURED = ("scatter", "fuse", "cleanup", "pyramid", "shift")


def cpu_name() -> str:
    """The machine, named. A latency table without one is not reproducible."""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def gpu_name() -> str:
    """Named if present. This is the CPU reference path either way -- every
    kernel here is numpy -- so an absent GPU is a fact to print, not to hide
    behind a blank cell."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5,
                             check=False)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return "none -- numpy CPU reference path"


def make_scan(rng, n, n_slots):
    """One sweep's worth of already-binned returns, as `scatter()` receives
    them. -1 is the annulus hole, which scatter drops; keeping it in the mix
    means the timing includes the drop it really does.

    ⚑ `class_id` is drawn over 0..15, not the 19 classes the project actually
      uses. `fusion.boyer_moore_update` rejects anything above 15 -- §10.2
      specifies a 4-bit candidate -- so a 19-class scan cannot be fused today
      and this script could not run against one. That is the pinned room
      decision in `fusion.boyer_moore_update`, and it costs nothing here
      (the kernels are branch-free in the class field), but it does mean the
      first real frame from JP's GT labels will raise rather than fuse.
    """
    return {
        "idx": rng.integers(-1, n_slots, n).astype(np.int64),
        "z_cm": rng.integers(-200, 600, n).astype(np.int16),
        "w_q": rng.integers(1, 2000, n).astype(np.int32),
        "refl": rng.integers(0, 256, n).astype(np.uint8),
        "class_id": rng.integers(0, 16, n).astype(np.uint8),
        "is_ground": rng.random(n) < 0.6,
    }


def fill_terrain(grid, rings, rng):
    """Terrain-shaped rather than uniform-random, matching bench_pyramid.py.

    The reductions are branch-free so the pyramid's timing does not depend on
    this, but running both scripts against different maps invites the question
    of why their numbers differ, and the answer should not be the fill.
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
        grid["traversability"][span] = np.where(obs < 3, 1 << 5, 0)


def make_range_image(rng, shape):
    """A sweep with a horizon, some sky and some absorbed beams.

    The NO_RETURN pixels are the point: eq (32) must not clear through them,
    and a benchmark image without any would time a branch the real one takes.
    """
    v = np.arange(shape[0])[:, None]
    horizon = 8.0 + 70.0 * (v / shape[0])
    img = horizon + rng.normal(0.0, 3.0, shape)
    img[rng.random(shape) < 0.08] = np.inf          # sky and absorbed beams
    return np.maximum(img, 1.0)


def make_candidates(rng, n):
    """Occupied cell centres in the vehicle frame -- what cleanup is handed.

    A free cell has nothing to clear, so the candidate set is the occupied
    cells, not the whole map; `--cells` is the cap that ⚑ sizing the
    visibility scratch into `allocate()` would have to fix.
    """
    theta = rng.uniform(-np.pi, np.pi, n)
    r = np.sqrt(rng.uniform(1.0, 100.0 ** 2, n))
    return (r * np.cos(theta), r * np.sin(theta),
            rng.uniform(-2.0, 6.0, n))


def build_frame(handle, args, rng):
    """Everything the loop needs, allocated before it starts. Nothing below
    this line may allocate -- that is the property the table is measuring."""
    n_slots = handle.allocated_slots
    scans = [make_scan(rng, args.points, n_slots) for _ in range(10)]

    cand_x, cand_y, cand_z = make_candidates(rng, args.cells)
    image = make_range_image(rng, IMAGE_SHAPE)
    vis_scratch = new_visibility_scratch(args.cells, image.dtype)
    has_return = rng.random(args.cells) < 0.35

    # One RingBuffer per ring. Ego-motion at `--speed-mps` over one 10 Hz
    # frame, in whole cells of that ring: 1.5 m at 15 m/s is 30 cells of the
    # 5 cm ring and 4 of the 40 cm ring, which is why the clear is per ring
    # and not one number for the map.
    per_frame_m = args.speed_mps / SENSOR_HZ
    buffers = [(RingBuffer(side=r.side, offset=r.offset),
                max(1, round(per_frame_m / r.cell_m)))
               for r in handle.rings]
    return scans, (cand_x, cand_y, cand_z, image, vis_scratch, has_return), buffers


def run(handle, args, rng):
    scans, vis, buffers = build_frame(handle, args, rng)
    cand_x, cand_y, cand_z, image, vis_scratch, has_return = vis
    t = Timer(stages=MEASURED + ("measured",))

    def one_frame(scan, timed: bool):
        """The frame body. `timed=False` is the warm-up pass -- numpy's first
        touch of a buffer faults its pages in, and that cost belongs to
        startup, not to the p99 a 10 Hz claim rests on."""
        ctx = t.stage if timed else _untimed
        with ctx("measured"):
            with ctx("scatter"):
                agg = scatter_sorted(**scan, scratch=handle.scratch)
            with ctx("fuse"):
                fuse(handle.grid, agg)
            with ctx("cleanup"):
                visibility_cleanup(cand_x, cand_y, cand_z, image,
                                   has_return_now=has_return,
                                   scratch=vis_scratch)
            if handle.pyramid is not None:
                with ctx("pyramid"):
                    build(handle.pyramid, handle.grid, handle.rings)
            with ctx("shift"):
                for buf, dx in buffers:
                    shift(buf, dx, 0, handle.grid)

    one_frame(scans[0], timed=False)
    for i in range(args.frames):
        one_frame(scans[i % len(scans)], timed=True)
    return t


class _untimed:
    """A no-op stand-in for `Timer.stage`, so the warm-up runs the identical
    code path rather than a second copy of it that can drift."""

    def __init__(self, name):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def print_table(t, handle, args):
    summary = t.summary()
    budget_ms = 1e3 / SENSOR_HZ

    print(f"{'stage':<13}{'owner':<10}{'p50 ms':>9}{'p99 ms':>9}"
          f"{'max ms':>9}{'x10Hz p99':>11}")
    print("-" * 61)
    for name in MEASURED:
        if name not in summary:
            continue
        s = summary[name]
        head = budget_ms / s["p99_ms"]
        print(f"{name:<13}{STAGE_OWNER[name]:<10}{s['p50_ms']:>9.2f}"
              f"{s['p99_ms']:>9.2f}{s['max_ms']:>9.2f}{head:>10.1f}x")

    print("-" * 61)
    m = summary["measured"]
    h = t.headroom("measured")
    print(f"{'MEASURED':<13}{'':<10}{m['p50_ms']:>9.2f}{m['p99_ms']:>9.2f}"
          f"{m['max_ms']:>9.2f}{budget_ms / m['p99_ms']:>10.1f}x")
    print(f"\n{h['fps_p50']:.1f} FPS p50 ({h['headroom_p50']:.1f}x), "
          f"{h['fps_p99']:.1f} FPS p99 ({h['headroom_p99']:.1f}x) "
          f"-- against the {budget_ms:.0f} ms budget at {SENSOR_HZ:.0f} Hz")

    print("\nnot in the subtotal above:")
    for name, why in BLOCKED.items():
        print(f"  {name:<13}{STAGE_OWNER[name]:<10}{why}")

    print(f"\n⚑ MEASURED is a LOWER BOUND on frame latency, not the frame "
          f"total. Six stages\n  above are unmeasured; the front end is the "
          f"whole of perception. The honest\n  sentence is \"the mapping back "
          f"end costs {m['p99_ms']:.1f} ms at p99\", never \"we run at "
          f"{h['fps_p99']:.0f} FPS\".")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schedule", default="5/10/20/40")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--points", type=int, default=120_000,
                    help="returns per sweep; a real HDL-64E sweep is ~120,000")
    ap.add_argument("--cells", type=int, default=200_000,
                    help="candidate (occupied) cells handed to visibility cleanup")
    ap.add_argument("--speed-mps", type=float, default=None,
                    help="ego speed for the shift row; default is the "
                         "schedule's anisotropy v_ref")
    ap.add_argument("--no-pyramid", action="store_true",
                    help="drop the stretch-item pyramid row (§7.2)")
    args = ap.parse_args()

    sched = load(args.schedule)
    if args.speed_mps is None:
        args.speed_mps = sched.anisotropy.v_ref_ms

    handle = allocate(sched, with_pyramid=not args.no_pyramid)
    rng = np.random.default_rng(0)
    fill_terrain(handle.grid, handle.rings, rng)

    per_frame_m = args.speed_mps / SENSOR_HZ
    shifted = sum(cells_per_shift(RingBuffer(side=r.side, offset=r.offset),
                                  max(1, round(per_frame_m / r.cell_m)), 0)
                  for r in handle.rings)

    print(f"CPU  {cpu_name()}")
    print(f"GPU  {gpu_name()}")
    print(f"numpy {np.__version__}, python {platform.python_version()}, "
          f"{platform.system()}\n")
    print(f"schedule {args.schedule}, {handle.allocated_slots:,} slots "
          f"({handle.logical_cells:,} logical), {args.frames} frames")
    print(f"{args.points:,} returns/sweep, {args.cells:,} candidate cells, "
          f"range image {IMAGE_SHAPE[0]}x{IMAGE_SHAPE[1]}")
    print(f"shift {args.speed_mps:.0f} m/s -> {per_frame_m:.2f} m/frame, "
          f"{shifted:,} cells cleared across {len(handle.rings)} rings\n")

    t = run(handle, args, rng)
    print_table(t, handle, args)


if __name__ == "__main__":
    main()
