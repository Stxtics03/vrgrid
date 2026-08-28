"""The evaluation harness. Master v4 §3.8. [Aakash]

The harness is the product — judges see the numbers and the demo, nothing
else. So this is the one place that owns the whole loop:

    sequence -> reference map (M*)          math §9.1, schedule-independent
             -> map under test, per schedule
             -> per-ring metrics             §9.2, §9.3
             -> one table                    Gate 6: every number from a script

Two things it is built around, both of which are constraints rather than
preferences:

**The reference is built once and reused across schedules.** M* is on the base
lattice and knows nothing about rings, which is exactly what makes comparing
5/10/20/40 against 5/10/50 a comparison of schedules rather than of two
separately-tuned pipelines (flaw E6, and `src/eval/CLAUDE.md`).

**Nothing here tunes anything.** Thresholds come from the frozen config. A
harness that can adjust a threshold to improve its own number is not a
harness, and the moment one exists somebody will use it the night before the
deadline.
"""

from dataclasses import dataclass

import numpy as np
from vrgrid.eval import metrics
from vrgrid.eval.reference_map import ReferenceMap
from vrgrid.gpu.allocators import allocate, bytes_allocated
from vrgrid.gpu.kernels import CEILING_NONE
from vrgrid.gpu.shift import RingBuffer, shift
from vrgrid.grid import traversability
from vrgrid.grid.fusion import fuse, initialise, scatter
from vrgrid.grid.pool import RefinementPool
from vrgrid.grid.query import GridMap
from vrgrid.grid.schedule import load, load_thresholds


def build_gridmap(schedule, thresholds=None, with_pool: bool = True) -> GridMap:
    """A ready-to-use map: allocation, ring windows, pool, initialised fields.

    The windows are centred on the vehicle, so ring L's absolute lattice runs
    from -N_L/2 to +N_L/2 and the vehicle sits at index 0. Centring is a
    convention rather than a requirement -- the toroidal addressing does not
    care -- but it is the convention the whole project reads x forward, y left
    in, so it is set here once instead of in every caller.
    """
    th = thresholds if thresholds is not None else load_thresholds()
    alloc = allocate(schedule, th)

    buffers = [RingBuffer(side=r.side, offset=r.offset,
                          x0=-r.side // 2, y0=-r.side // 2)
               for r in alloc.rings]

    initialise(alloc.grid)   # ceiling sentinel; see fusion.initialise
    pool_cfg = th.get("refinement_pool", {})
    pool = RefinementPool(pool_cfg.get("blocks", 512),
                          pool_cfg.get("cells_per_block", 16),
                          arrays=alloc.pool) if with_pool else None

    gm = GridMap(soa=alloc.grid, schedule=schedule, buffers=buffers,
                 thresholds=th, transient=alloc.transient, pool=pool,
                 scatter_mode=alloc.scatter_mode)
    gm.allocation = alloc
    return gm


def recenter(gm: GridMap, x_m: float, y_m: float) -> int:
    """Slide every ring window so it is centred on the vehicle. Math §2.4.

    Returns the number of slots cleared, which is the O(perimeter) claim as a
    measurement rather than an assertion.

    The origin moves in whole COARSEST cells and every finer ring moves a whole
    multiple of that -- §2.4's constraint, and the reason it exists: a fractional
    step would shift each ring boundary by a fraction of a cell and force a
    resample, which is precisely the "data loss during projection" the problem
    statement warns about. Expected side effect: the nominal ring boundary
    wobbles by up to 40 cm. That is correct behaviour, not a bug.
    """
    coarsest = gm.schedule.rings[-1]
    k_coarsest = gm.schedule.k(len(gm.schedule.rings) - 1)

    # Where the vehicle sits, in coarsest cells, and where the window wants
    # its low corner to be so the vehicle stays in the middle.
    want_x = int(np.floor(x_m / coarsest.cell_m)) - gm.buffers[-1].side // 2
    step = want_x - gm.buffers[-1].x0
    want_y = int(np.floor(y_m / coarsest.cell_m)) - gm.buffers[-1].side // 2
    step_y = want_y - gm.buffers[-1].y0
    if step == 0 and step_y == 0:
        return 0

    cleared = 0
    for level, buf in enumerate(gm.buffers):
        scale = k_coarsest // gm.schedule.k(level)   # integer, by validate()
        slots = shift(buf, step * scale, step_y * scale, gm.soa,
                      fill={"ceiling_height": CEILING_NONE})
        cleared += int(slots.size)
    return cleared


def run_sequence(gm: GridMap, scans, recentre: bool = True) -> int:
    """Drive the map through a sequence. Returns the number of frames.

    `scans` yields (points in VEHICLE frame, class ids, is_ground, pose 4x4).

    ⚑ Both frames are needed and they do different jobs -- see the note on
      `fusion.scatter()`. The ring a point lands in is decided in the vehicle
      frame, because foveation follows the vehicle; the CELL it lands in is
      decided in the world frame, because cell identity is world-anchored and
      that is the entire reason the toroidal shift exists. Scatter every frame
      at the vehicle origin instead and the map still builds, still looks
      plausible, and smears the whole sequence onto one patch of ground.

    The pose is applied here rather than in `scatter()` so there is one place
    that knows the frame convention. When `perception.transforms` lands this
    should call `transform_points()` instead of composing the matrix itself --
    two implementations of one convention is how a map ends up slowly rotating.
    """
    frames = 0
    for pts, cls, ground, pose in scans:
        pose = np.asarray(pose, dtype=np.float64)
        world = np.asarray(pts, dtype=np.float64) @ pose[:3, :3].T + pose[:3, 3]
        if recentre:
            recenter(gm, float(pose[0, 3]), float(pose[1, 3]))
        agg = scatter(gm, pts, cls, ground, points_world_m=world)
        fuse(gm.soa, agg, gm.thresholds)
        frames += 1

    rings = [(slice(r.offset, r.offset + r.side * r.side), r.side)
             for r in gm.allocation.rings]
    traversability.update(gm.soa, gm.schedule, rings, gm.thresholds)
    return frames


@dataclass
class Result:
    """One schedule's scorecard. Deliberately plain data: the thing that
    formats a table must not be the thing that computes it, or a formatting
    change becomes a number change."""

    schedule_name: str
    frames: int
    bytes_allocated: int
    logical_cells: int
    rmse_cm: dict
    coarsening: dict
    iou: dict
    fill: dict

    def rows(self):
        for ring in sorted(self.rmse_cm):
            c = self.coarsening[ring]
            yield {"ring": ring, "rmse_cm": self.rmse_cm[ring], "rho": c["rho"],
               "il_cm": c["il_cm"], "bias_cm": c["bias_cm"],
               "spread_cm": c["spread_cm"], "n": c["n"],
               "iou": self.iou[ring], "fill": self.fill[ring]}


def evaluate(gm: GridMap, reference: ReferenceMap, frames: int = 0) -> Result:
    """Every §9 metric for one map, per ring."""
    return Result(
        schedule_name=gm.schedule.name,
        frames=frames,
        bytes_allocated=bytes_allocated(gm.allocation),
        logical_cells=gm.allocation.logical_cells,
        rmse_cm=metrics.height_rmse_per_ring(gm, reference),
        coarsening=metrics.coarsening_ratio_per_ring(gm, reference),
        iou=metrics.occupancy_iou_per_ring(gm, reference),
        fill=metrics.fill_rate_per_ring(gm, reference),
    )


def format_result(result: Result, schedule) -> str:
    """The per-ring table. One row per ring, because a single aggregate number
    hides the entire claim: error is SUPPOSED to grow with range."""
    head = (f"{result.schedule_name}  --  {result.frames} frames, "
            f"{result.logical_cells:,} logical cells, "
            f"{result.bytes_allocated / 1e6:.2f} MB allocated")
    cols = (f"{'ring':>4} {'cell':>6} {'reach':>7} {'cells':>8} {'RMSE':>8} "
            f"{'bias':>7} {'spread':>7} {'IL':>7} {'rho':>6} {'IoU':>6} {'fill':>6}")
    lines = [head, "", cols, "-" * len(cols)]

    def fmt(v, w, p=2):
        return f"{'--':>{w}}" if np.isnan(v) else f"{v:>{w}.{p}f}"   # v != v is nan

    for r in result.rows():
        ring = schedule.rings[r["ring"]]
        lines.append(
            f"{r['ring']:>4} {ring.cell_m * 100:>5.0f}c {ring.half_width_m:>6.0f}m "
            f"{r['n']:>8,} {fmt(r['rmse_cm'], 8)} {fmt(r['bias_cm'], 7)} "
            f"{fmt(r['spread_cm'], 7)} {fmt(r['il_cm'], 7)} {fmt(r['rho'], 6)} "
            f"{fmt(r['iou'], 6)} {fmt(r['fill'], 6)}"
        )
    lines += [
        "",
        "RMSE, bias, spread, IL in cm against M*. rho = IL/spread (§9.3):",
        "  rho ~ 1  coarsening cost only the terrain's own sub-cell variability",
        "  rho >> 1 the estimate is biased beyond that -- schedule too aggressive",
        "cells = ring cells with an observed reference footprint AND >1 reference",
        "        return; everything else is dropped rather than scored as agreement.",
    ]
    return "\n".join(lines)


def compare(schedules, scans_factory, reference: ReferenceMap,
            thresholds=None) -> list:
    """Run several schedules against one reference. The ablation, in one call.

    `scans_factory()` must return a FRESH iterator each time -- the same scans
    in the same order for every schedule, or the comparison measures which
    schedule got the better half of the sequence.
    """
    results = []
    for name in schedules:
        s = load(name) if isinstance(name, str) else name
        gm = build_gridmap(s, thresholds)
        frames = run_sequence(gm, scans_factory())
        results.append((s, gm, evaluate(gm, reference, frames)))
    return results


def _nanmean(values) -> float:
    """np.nanmean over an all-nan list warns and returns nan; a ring nobody
    drove through is the ordinary case here, not an anomaly worth a warning."""
    finite = [v for v in values if not np.isnan(v)]
    return float(np.mean(finite)) if finite else float("nan")


def memory_vs_regret_row(result: Result) -> dict:
    """One point of the Day-4 headline curve. Plan regret is not wired in yet
    (§8, and it needs the planner), so this emits the memory axis and the
    accuracy proxy, and the regret column joins it when `plan_regret` lands.

    Kept here rather than invented later so the curve's shape is fixed before
    the number that goes on it exists -- which is the order that stops a plot
    being designed around the result it got.
    """
    rings = sorted(result.rmse_cm)
    finite = [result.rmse_cm[r] for r in rings if not np.isnan(result.rmse_cm[r])]
    return {
        "schedule": result.schedule_name,
        "megabytes": result.bytes_allocated / 1e6,
        "logical_cells": result.logical_cells,
        "worst_ring_rmse_cm": max(finite) if finite else float("nan"),
        "mean_rho": _nanmean([result.coarsening[r]["rho"] for r in rings]),
        "regret": None,
    }
