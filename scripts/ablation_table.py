#!/usr/bin/env python3
"""What the ring schedule costs and buys, against uniform grids. [Shrestha]

    python scripts/ablation_table.py [--frames 40] [--out docs/figures]
    python scripts/ablation_table.py --expect-thresholds <sha256>

One row per schedule: cells, memory, and frame latency, with **one frozen set
of thresholds across every row**. That is the whole discipline of an ablation
and it is the easy thing to lose -- a schedule compared under thresholds tuned
for it is not being compared. CLAUDE.md says thresholds live in `configs/` and
are frozen before schedules are compared, so this hashes
`configs/thresholds.yaml`, prints the digest with the table, and re-reads and
re-hashes it at the end. `--expect-thresholds` turns that into an assertion, so
a number in the report can be pinned to the config it was produced under.

**This is the memory-and-latency half of the ablation, not the whole thing.**
Whether a coarser schedule makes a WORSE DECISION is plan regret, and that is
`scripts/regret_plot.py` over `src/eval`. Reading this table alone would say
uniform 80 cm is the best schedule in the project, because it is the cheapest
and the fastest and this table cannot see what it costs you. The two tables are
meant to be read side by side and the footer says so.

The uniform baselines are built through `harness.uniform_schedule`, so they go
through the same `validate()`, the same lattice, the same allocator and the
same kernels we do. A baseline evaluated by a second code path is not a
baseline. Uniform 5 cm at 100 m is the 192 MB figure the 21.5x claim is
measured against, and it is reproduced here rather than quoted.

⚑ Latency is over a synthetic HDL-64E sweep, on the numpy CPU reference path.
  The comparison BETWEEN rows is the point and it is fair -- one machine, one
  sweep, one threshold set. The absolute milliseconds are not reportable.
"""

import argparse
import csv
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import timing_table as tt
import yaml
from vrgrid.eval.harness import uniform_schedule
from vrgrid.gpu.allocators import allocate, bytes_allocated
from vrgrid.gpu.baseline import dense3d_voxels, uniform25d_cells
from vrgrid.grid.schedule import load

CONFIG = Path("configs/thresholds.yaml")
FROZEN = ["5/10/20/40", "5/10/50"]
UNIFORM_CELLS_M = [0.05, 0.10, 0.20, 0.40, 0.80]

# The stages this table compares. `pyramid` is absent because it is a stretch
# item that `allocate()` leaves off, and a row that silently included it would
# be comparing a different system.
STAGES = ("bin", "scatter", "fuse", "cleanup", "shift", "measured")


def thresholds_digest(path: Path = CONFIG):
    """SHA-256 of the config as bytes, not of the parsed dict.

    The bytes are what a reader can reproduce with `sha256sum`; a dict hash
    would depend on Python's repr and could not be checked from a shell.
    """
    raw = path.read_bytes()
    return hashlib.sha256(raw).hexdigest(), yaml.safe_load(raw)


def schedules():
    for name in FROZEN:
        yield load(name), False
    for cell in UNIFORM_CELLS_M:
        yield uniform_schedule(cell), True


def measure(sched, thresholds, args, repeats):
    """Memory off the real allocation, latency off `timing_table`'s harness.

    Both are imported rather than reimplemented: a second copy of the frame
    loop is how two scripts start disagreeing about what a millisecond in this
    system means.

    **Repeated, and the spread is reported.** One run of this comparison put
    our schedule 18% ahead of uniform 5 cm; the next put it 1% ahead. The
    DIRECTION is stable -- across three repeats the two ranges do not overlap
    -- but a point estimate from a single run swings about 10% on a laptop,
    and quoting one would be quoting noise. The median across repeats goes in
    the table and the spread goes beside it.
    """
    per_run = []
    for _ in range(repeats):
        rng = np.random.default_rng(0)          # same sweep every time
        handle = allocate(sched, thresholds=thresholds)
        tt.fill_terrain(handle.grid, handle.rings, rng)
        per_run.append(tt.run(tt.Frame(handle, sched, args, rng), args).summary())

    def across(stage, key):
        vals = [r[stage][key] for r in per_run if stage in r]
        return float(np.median(vals)) if vals else float("nan")

    row = {
        "schedule": sched.name,
        "logical_cells": handle.logical_cells,
        "allocated_slots": handle.allocated_slots,
        "map_mb": handle.logical_cells * 12 / 1e6,
        "total_mb": bytes_allocated(handle) / 1e6,
        "repeats": repeats,
    }
    for stage in STAGES:
        row[f"{stage}_p50_ms"] = across(stage, "p50_ms")
        row[f"{stage}_p99_ms"] = across(stage, "p99_ms")
    totals = [r["measured"]["p50_ms"] for r in per_run if "measured" in r]
    row["measured_p50_lo"], row["measured_p50_hi"] = min(totals), max(totals)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=3,
                    help="timing runs per schedule; the table reports the median "
                         "and the spread, because one run swings ~10%%")
    ap.add_argument("--points", type=int, default=120_000)
    ap.add_argument("--cells", type=int, default=150_000,
                    help="candidate cells for the §10.4 cleanup row")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--expect-thresholds", default=None,
                    help="fail unless configs/thresholds.yaml has this sha256")
    args = ap.parse_args()

    digest, thresholds = thresholds_digest()
    if args.expect_thresholds and args.expect_thresholds != digest:
        raise SystemExit(
            f"thresholds have moved: configs/thresholds.yaml is {digest}, "
            f"expected {args.expect_thresholds}. Schedules compared under "
            f"different thresholds are not compared.")

    run_args = SimpleNamespace(frames=args.frames, points=args.points,
                               cells=args.cells, speed_mps=15.0)
    rows = []
    for sched, is_uniform in schedules():
        row = measure(sched, thresholds, run_args, args.repeats)
        row["uniform"] = is_uniform
        rows.append(row)

    ours = next(r for r in rows if r["schedule"] == "5_10_20_40")
    uni5 = next((r for r in rows if r["schedule"].startswith("uniform_5cm")), None)

    print(f"CPU  {tt.cpu_name()}")
    print(f"thresholds  sha256 {digest[:16]}...  ({CONFIG})")
    print(f"{args.frames} frames x {args.repeats} repeats, {args.points:,} "
          f"returns/sweep, {args.cells:,} candidate cells,\none sweep shared by "
          f"every row\n")

    head = (f"{'schedule':<14}{'cells':>11}{'map MB':>9}{'total MB':>10}"
            f"{'x ours':>8}{'p50 ms':>9}{'range':>15}")
    print(head)
    print("-" * len(head))
    for r in rows:
        mark = " " if r["uniform"] else "*"
        ratio = r["map_mb"] / ours["map_mb"]
        span = f"{r['measured_p50_lo']:.1f}-{r['measured_p50_hi']:.1f}"
        print(f"{mark}{r['schedule']:<13}{r['logical_cells']:>11,}"
              f"{r['map_mb']:>9.2f}{r['total_mb']:>10.2f}{ratio:>8.2f}"
              f"{r['measured_p50_ms']:>9.2f}{span:>15}")
    print("\n* = a frozen ring schedule. The rest are uniform grids at the same "
          "100 m half-width.")

    print("\nper-stage p50 ms")
    sub = [s for s in STAGES if s != "measured"]
    print(f"{'schedule':<14}" + "".join(f"{s:>10}" for s in sub))
    print("-" * (14 + 10 * len(sub)))
    for r in rows:
        print(f"{r['schedule']:<14}"
              + "".join(f"{r.get(f'{s}_p50_ms', float('nan')):>10.2f}" for s in sub))

    # The two things in the per-stage table that a reader should not have to
    # spot for themselves.
    one_ring = [r for r in rows if r["uniform"]]
    if one_ring and "bin_p50_ms" in ours:
        flat = sum(r["bin_p50_ms"] for r in one_ring) / len(one_ring)
        print(f"\n⚑ `bin` scales with RING COUNT, not map size: ~{flat:.1f} ms for every "
              f"one-ring\n  row here, from 62,500 cells to 16,000,000, against "
              f"{ours['bin_p50_ms']:.1f} ms for our four. The\n  foveated schedule pays "
              f"~{ours['bin_p50_ms'] - flat:.0f} ms of latency purely for having rings, and "
              f"all of it is in the\n  per-ring mask-and-gather loop -- the step no module "
              f"owns. One vectorised\n  binning function over the whole sweep removes the "
              f"penalty; the foveation\n  itself does not cost it.")

    if uni5 is not None:
        # The report's headline ratio, reproduced through the allocator rather
        # than quoted from a slide. baseline.py computes the same two counts
        # from geometry; if these disagree, one of them is wrong.
        print(f"\nuniform 5 cm is the 21.5x baseline, rebuilt here: "
              f"{uni5['logical_cells']:,} cells, {uni5['map_mb']:.2f} MB, "
              f"{uni5['map_mb'] / ours['map_mb']:.1f}x our map memory.")
        expect = uniform25d_cells()
        if uni5["logical_cells"] != expect:
            print(f"  ⚑ but baseline.uniform25d_cells() says {expect:,} -- "
                  f"the two disagree and one of them is on a slide.")
        print(f"  dense 3D for scale: {dense3d_voxels():,} voxels, "
              f"{dense3d_voxels() / 1e6:.0f} MB at 1 B each.")
        # The honest headline this table supports. Uniform 5 cm is the only
        # baseline that resolves what our finest ring resolves; the coarser
        # uniforms buy their speed by throwing away the near field, which is
        # a trade this table cannot price and regret_plot can.
        disjoint = ours["measured_p50_hi"] < uni5["measured_p50_lo"]
        print(f"\nAgainst uniform 5 cm -- the only baseline that resolves what our finest "
              f"ring\ndoes -- we are {uni5['map_mb'] / ours['map_mb']:.1f}x smaller on map "
              f"memory, which is exact and comes from cell\ncounts. On latency we are "
              f"{ours['measured_p50_ms']:.1f} ms against {uni5['measured_p50_ms']:.1f} ms, "
              f"ranges {ours['measured_p50_lo']:.1f}-{ours['measured_p50_hi']:.1f} "
              f"and\n{uni5['measured_p50_lo']:.1f}-{uni5['measured_p50_hi']:.1f} across "
              f"{ours['repeats']} repeats -- "
              f"{'disjoint, so the direction holds' if disjoint else 'OVERLAPPING, so quote the direction at most'}. "
              f"Quote the memory\nratio; treat the latency margin as a direction, not a "
              f"percentage.\n\nThe structural latency result is `shift`: "
              f"{ours.get('shift_p50_ms', float('nan')):.1f} ms against "
              f"{uni5.get('shift_p50_ms', float('nan')):.1f} ms, because the\nclear is "
              f"O(perimeter) and uniform 5 cm has a 4,000-cell window where our finest "
              f"has\n400. That one is geometry, not noise.")

    after, _ = thresholds_digest()
    if after != digest:
        raise SystemExit("configs/thresholds.yaml changed DURING the run; "
                         "this table compares nothing.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "ablation.csv"
    fields = ["schedule", "uniform", "logical_cells", "allocated_slots",
              "map_mb", "total_mb"] + \
             [f"{s}_p50_ms" for s in STAGES] + [f"{s}_p99_ms" for s in STAGES]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields + ["thresholds_sha256"],
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "thresholds_sha256": digest})
    print(f"\nwrote {path}  (thresholds sha256 {digest})")

    print("\n⚑ This is the cost half of the ablation only. On these columns "
          "uniform 80 cm\n  is the best schedule in the project -- cheapest, "
          "fastest -- because nothing\n  here can see what coarsening does to "
          "the decision. Read it beside\n  scripts/regret_plot.py, which is "
          "the half that can.")


if __name__ == "__main__":
    main()
