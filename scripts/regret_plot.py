#!/usr/bin/env python3
"""The Day-4 headline figure: memory on x, plan regret on y. Math §8.2.

    python scripts/regret_plot.py [--frames 16] [--out docs/figures]

Gate 4 asks for this curve with at least four schedules on it. Gate 6 asks
that every number on a slide come from a script. This is that script: it runs
`eval_synthetic.py`'s sweep -- the same one, imported, not a second copy -- and
writes both the CSV of rows and the figure drawn from it, so the figure and
the table can never disagree.

The CSV is written whether or not matplotlib is installed. Numbers are the
deliverable; the picture is a rendering of them, and a missing plotting
library should not cost you the sweep.

⚑ **The curve this currently produces has no knee, and that is a finding
  rather than a bug in the plotting.** On the synthetic sequence R(S) is 0.000
  almost everywhere -- for both frozen schedules and for uniform 20, 40 and
  80 cm alike -- with a single non-monotone spike at uniform 10 cm. It is also
  frame-count dependent in a way a real result should not be: at 24 frames the
  two frozen schedules move to 0.207 while the three coarse uniforms stay at
  0.000, i.e. coarser maps scoring BETTER than ours, the opposite of the claim
  the figure exists to make. The diagnosis is in the planning
  query rather than in the metric: `eval_synthetic.PLAN_LANE_CELLS` runs the
  path down a fixed lane six cells off centre, and the hazards the terrain was
  built to contain -- the 12 cm kerbs at |y| = 3 m, the 40 cm pothole at
  (18, 0), the ramp from x = 30 m -- are not on it. Every schedule plans the
  same straight line, so R(S) is measuring tie-breaking between equal-cost
  paths rather than the cost of coarsening, and a fine map that resolves the
  pothole can be penalised for routing around something the coarse maps
  smooth away entirely.

  **I have deliberately not adjusted the query.** Changing the experiment
  after seeing which answer it gives is how a figure stops being evidence.
  Posing it properly -- a start and goal that require a decision about the
  kerb or the pothole -- is a change to `eval_synthetic.py`, which is Aakash's,
  and it should be agreed before it is made rather than tuned until the curve
  looks right.

  ---- 2026-09-01, Aakash: the numbers above have moved, and not by tuning ----

  The block above is left as written because its diagnosis is still right and
  still open. What changed underneath it is `eval/synthetic.py`'s beam-surface
  intersection, which had a sign error: the sensor's height above a surface at
  elevation z is `(h_s - z)` and the sampler used `(h_s + z)`. On flat ground
  they agree, so it survived every test the scene had. On the features it did
  not, and the consequence for this figure is specific: **across a whole
  sequence the old sampler returned not one point below -30 cm.** The 40 cm
  pothole -- the scene's only negative obstacle -- had never once been
  observed as a hole at any range, so "the hazards are not on the lane" was
  understating it. One of them was not in the map at all.

  I found this from a FOV assertion in `tests/test_loader_path.py`, before
  looking at any regret number, and the fix is forced by the geometry rather
  than chosen: `_beam_range` now solves the intersection instead of taking one
  step towards it. `PLAN_LANE_CELLS` and the rest of the query are untouched.

  What the curve does now, `--frames 14`:

      5_10_20_40    2.389      uniform_20cm   3.354
      5_10_50       2.389      uniform_40cm   3.854
      uniform_10cm  0.000      uniform_80cm   3.854

  The all-zeros are gone, the non-monotone spike at uniform 10 cm is gone --
  it is now the best of the six, which is the direction a finer map should
  move -- and the four uniforms are identical at 12, 14 and 24 frames where
  they used to swing. The two frozen schedules still move with frame count
  (2.389 at 12 and 14, 1.450 at 24), so that half of your finding stands.

  It is not a flattering curve: on this scene ours costs 29.06 MB to score
  2.389 where uniform 10 cm costs 18.19 MB to score 0.000. That is a real
  reading of a synthetic scene whose hazards sit off the planned lane, and it
  is an argument for posing the query properly -- your point, unchanged --
  rather than for a different sampler.

Everything here is synthetic and none of it is reportable: the terrain is
analytic, so there is no sensor noise, no occlusion and no registration error.
The figure is stamped with that, in the figure itself, so a screenshot of it
cannot lose the caveat.
"""

import argparse
import csv
import shutil
import sys
import tempfile
from itertools import pairwise
from pathlib import Path

# `scripts/` is sys.path[0] when this is run as a script, so the sweep is
# imported from the one place it already lives rather than reimplemented. A
# second copy of the orchestration is how the figure and the table drift.
import eval_synthetic as sweep
from vrgrid.eval.harness import build_gridmap, evaluate, memory_vs_regret_row, run_sequence
from vrgrid.eval.plan_regret import common_support
from vrgrid.eval.reference_map import build_from_scans
from vrgrid.eval.synthetic import read_sequence, write_sequence
from vrgrid.grid.schedule import load
from vrgrid.grid.transient import TrackList

FIELDS = ("schedule", "megabytes", "logical_cells", "worst_ring_rmse_cm",
          "mean_rho", "regret", "frechet_m", "unknown_fraction",
          "blocked_on_reference")

CAVEAT = ("SYNTHETIC SEQUENCE - NOT REPORTABLE: analytic terrain, "
          "no sensor noise, occlusion or registration error")


def collect(root, frames: int) -> list:
    """Run every schedule over one sequence and return the §8.2 rows.

    The two-pass structure is `eval_synthetic`'s and it matters: every map is
    built before any is scored, because the regret has to be restricted to the
    common support -- ground EVERY schedule observed. Scored without it the
    number measures fill rate rather than coarsening, and the coarse maps win
    for the wrong reason.
    """
    reference = build_from_scans(read_sequence(root, "99"))
    vehicle_x = (frames - 1) * 2.0
    schedules = ([load(n) for n in sweep.SCHEDULES]
                 + [sweep.uniform_schedule(c, half_width_m=24.0)
                    for c in sweep.UNIFORM_CELLS_M])

    built = []
    for schedule in schedules:
        gm = build_gridmap(schedule)
        tracks = TrackList(gm.allocation.max_tracks, arrays=gm.allocation.tracks)
        stats = run_sequence(gm, sweep.vehicle_frame_scans(root, "99"),
                             tracks=tracks)
        built.append((schedule, gm, evaluate(gm, reference, stats.frames)))

    mask = common_support(*[sweep.costmaps_for(gm, reference, vehicle_x)[1]
                            for _, gm, _ in built])
    rows = []
    for _, gm, result in built:
        reg = sweep.plan_regret_for(gm, reference, vehicle_x, mask)
        rows.append(memory_vs_regret_row(result, reg))
    return rows, float(mask.mean())


def write_csv(rows, path: Path) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in FIELDS})


def monotonicity(rows) -> str:
    """Whether the curve does what §8.2 says it should.

    The claim is that regret rises as memory falls -- cheap maps make worse
    decisions. Sorted by memory descending, regret should be non-increasing.
    Reporting the violations is the point: a figure that contradicts its own
    caption is worse than no figure, and this is the check that catches it
    before it reaches a slide.
    """
    ordered = sorted(rows, key=lambda r: -r["megabytes"])
    bad = [(a["schedule"], b["schedule"])
           for a, b in pairwise(ordered)
           if b["regret"] is not None and a["regret"] is not None
           and b["regret"] < a["regret"] - 1e-9]
    if not bad:
        return "monotone: regret does not fall as memory falls, as §8.2 claims"
    pairs = ", ".join(f"{a} -> {b}" for a, b in bad)
    return (f"⚑ NOT MONOTONE in {len(bad)} step(s): {pairs}. A coarser map "
            "scoring lower regret than a finer one is the opposite of the "
            "claim this figure makes -- see this script's docstring.")


def draw(rows, path: Path, mask_frac: float, frames: int) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    ours = [r for r in rows if not r["schedule"].startswith("uniform")]
    base = [r for r in rows if r["schedule"].startswith("uniform")]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    curve = sorted(rows, key=lambda r: r["megabytes"])
    ax.plot([r["megabytes"] for r in curve], [r["regret"] for r in curve],
            "-", color="0.75", zorder=1, linewidth=1.2)
    ax.scatter([r["megabytes"] for r in base], [r["regret"] for r in base],
               s=46, facecolor="white", edgecolor="#4C72B0", zorder=3,
               label="uniform grid (baseline)")
    ax.scatter([r["megabytes"] for r in ours], [r["regret"] for r in ours],
               s=64, color="#C44E52", zorder=4, marker="D",
               label="ring schedule (ours)")
    # Points bunch up at the cheap end of the axis, so the labels are
    # staggered by rank rather than all placed at the same offset. Overlapping
    # text is how a figure that is correct becomes a figure nobody trusts.
    for rank, r in enumerate(sorted(rows, key=lambda r: r["megabytes"])):
        dy = 9 if rank % 2 == 0 else -14
        ax.annotate(r["schedule"].replace("_", "/"),
                    (r["megabytes"], r["regret"]),
                    textcoords="offset points", xytext=(0, dy), fontsize=7.5,
                    ha="center", color="0.35")

    ax.set_xlabel("preallocated memory (MB)")
    ax.set_ylabel(r"plan regret  $R(S) = J_{M^*}(\pi_S) - J_{M^*}(\pi^*)$")
    ax.set_title("Memory against plan regret (math §8.2)", fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.margins(x=0.10, y=0.18)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.text(0.995, 0.985, f"{frames} frames, common support {mask_frac:.0%}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            color="0.5")
    fig.text(0.5, 0.012, CAVEAT, ha="center", fontsize=7.5, color="#B4342F")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--out", default="docs/figures",
                    help="directory for regret.csv and regret.svg/.png")
    ap.add_argument("--keep", default=None,
                    help="keep the generated sequence here instead of a tempdir")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="vrgrid-syn-"))
    try:
        write_sequence(root, "99", n_frames=args.frames)
        rows, mask_frac = collect(root, args.frames)
    finally:
        if args.keep is None:
            shutil.rmtree(root, ignore_errors=True)

    write_csv(rows, out / "regret.csv")

    print(f"{'schedule':<14}{'MB':>8}{'cells':>10}{'R(S)':>9}"
          f"{'frechet':>9}{'unknown':>9}")
    print("-" * 59)
    for r in sorted(rows, key=lambda r: -r["megabytes"]):
        print(f"{r['schedule']:<14}{r['megabytes']:>8.2f}{r['logical_cells']:>10,}"
              f"{r['regret']:>9.3f}{r['frechet_m']:>8.2f}m"
              f"{r['unknown_fraction']:>8.1%}")
    print(f"\ncommon support: {mask_frac:.1%} of the planning window was observed "
          f"by every schedule")
    print(monotonicity(rows))

    if draw(rows, out / "regret.svg", mask_frac, args.frames):
        print(f"\nwrote {out / 'regret.csv'}, {out / 'regret.svg'} "
              f"and {out / 'regret.png'}")
    else:
        print(f"\nwrote {out / 'regret.csv'}. No figure: matplotlib is not "
              f"installed -- `pip install -e \".[report]\"`. The numbers above "
              f"are the deliverable; the plot only renders them.")
    print(f"\n{CAVEAT}")


if __name__ == "__main__":
    sys.exit(main())
