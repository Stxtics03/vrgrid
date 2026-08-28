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
from vrgrid.grid import gate, traversability
from vrgrid.grid.fusion import fuse, initialise, scatter
from vrgrid.grid.pool import RefinementPool
from vrgrid.grid.query import GridMap
from vrgrid.grid.schedule import load, load_thresholds
from vrgrid.grid.transient import TrackList, separate
from vrgrid.grid.transient import step as transient_step


def uniform_schedule(cell_m: float, half_width_m: float = 100.0,
                     base_cell_m: float = 0.05, hysteresis_eps: float = 0.1):
    """A single-ring schedule: the uniform-grid baselines of §8.2's sweep.

    The money plot needs points that are not our own schedule, and a uniform
    grid is the honest comparison -- it is what everyone else builds and what
    the 21.5x claim is measured against. Built through the same `Schedule` the
    frozen configs load into, so it goes through `validate()` and through
    exactly the same lattice, fusion and metric code. A baseline evaluated by
    a second code path is not a baseline.

    ⚑ The two FROZEN schedules share rings 0 and 1 (5 cm to 10 m, 10 cm to
      25 m) and differ only beyond 25 m. So a planning problem inside 25 m
      cannot tell them apart and will report identical regret for both -- not
      a bug, and not evidence that the ablation is free. Either plan into the
      far field or read the curve against these uniform points, which differ
      from us everywhere.
    """
    from vrgrid.grid.schedule import Anisotropy, Ring, Schedule, validate

    side = 2 * half_width_m / cell_m
    if abs(side - round(side)) > 1e-9 or round(side) % 2:
        raise ValueError(
            f"{half_width_m} m at {cell_m} m is {side} cells; the allocator "
            "needs an even whole number so the ring has a centre"
        )
    cells = round(side) ** 2
    name = f"uniform_{cell_m * 100:.0f}cm"
    s = Schedule(
        name=name, base_cell_m=base_cell_m,
        rings=[Ring(0, half_width_m, cell_m, cells, 0.0)],
        total_cells=cells, vertical_extent_m=(-2.0, 6.0),
        hysteresis_eps=hysteresis_eps, anisotropy=Anisotropy(),
    )
    validate(s)
    return s


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

    # query() converts vehicle frame to world with this, so it has to move
    # even when the window does not: the vehicle drifts within a coarsest cell
    # between shifts, and 40 cm of unrecorded drift is a whole ring-0 cell.
    gm.vehicle_xy_m = (x_m, y_m)
    if step == 0 and step_y == 0:
        return 0

    cleared = 0
    for level, buf in enumerate(gm.buffers):
        scale = k_coarsest // gm.schedule.k(level)   # integer, by validate()
        slots = shift(buf, step * scale, step_y * scale, gm.soa,
                      fill={"ceiling_height": CEILING_NONE})
        cleared += int(slots.size)
    return cleared


@dataclass
class RunStats:
    """What the frame loop did, so the dashboard and §9.4 can both read it."""

    frames: int = 0
    static_points: int = 0
    dynamic_points: int = 0
    dynamic_to_transient: int = 0
    tracks: int = 0
    gate_fired: int = 0
    gate_acquired: int = 0
    gate_refused: int = 0
    gate_released: int = 0

    @property
    def removal(self) -> dict:
        """§9.4's DR / SP / F, from the counters the loop already keeps.

        DR is the fraction of dynamic returns kept OUT of the persistent map;
        SP the fraction of static returns kept IN. Both directions, always:
        DR alone is gameable -- delete the whole map and score 100%.
        """
        from vrgrid.eval.metrics import dynamic_removal

        return dynamic_removal(self.dynamic_points, self.dynamic_points,
                               self.static_points, self.static_points)


def run_sequence(gm: GridMap, scans, recentre: bool = True,
                 tracks: TrackList | None = None) -> RunStats:
    """Drive the map through a sequence. Returns what it did.

    `scans` yields (points in VEHICLE frame, RAW label ids, is_ground, pose).

    ⚑ RAW label ids, not learning ids. `moving-*` (250-259) is what separates
      dynamic from static, and the 19-class learning map collapses every
      `moving-*` onto its static counterpart -- so a scan already through the
      learning map cannot be separated at all, and every car that ever drove
      past ends up welded into the elevation map. See grid/transient.py.

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
    stats = RunStats()
    speed = 0.0
    last_xy = None

    for pts, labels, ground, pose in scans:
        pose = np.asarray(pose, dtype=np.float64)
        pts = np.asarray(pts, dtype=np.float64)
        world = pts @ pose[:3, :3].T + pose[:3, 3]
        xy = (float(pose[0, 3]), float(pose[1, 3]))
        if last_xy is not None:
            dt = gm.thresholds.get("fusion", {}).get("frame_dt_s", 0.1)
            speed = float(np.hypot(xy[0] - last_xy[0], xy[1] - last_xy[1]) / dt)
        last_xy = xy
        if recentre:
            recenter(gm, *xy)

        # Dynamic returns never reach the persistent map. Before this existed,
        # one car 12 m ahead moved ring 1's RMSE from 0.48 cm to 11.71 cm.
        static, moving = separate(labels)
        stats.static_points += int(static.sum())
        stats.dynamic_points += int(moving.sum())

        written, n_tracks = transient_step(gm, pts, labels, world, tracks=tracks)
        stats.dynamic_to_transient += written
        stats.tracks = n_tracks

        agg = scatter(gm, pts[static], np.asarray(labels)[static] % 16,
                      np.asarray(ground, dtype=bool)[static],
                      points_world_m=world[static])
        fuse(gm.soa, agg, gm.thresholds)

        # Traversability before the gate: the gate consults the hazard bits,
        # so computing it after would gate on the PREVIOUS frame's geometry.
        _update_traversability(gm)
        fired = gate.apply(gm, agg.cells, vehicle_speed_ms=speed,
                           thresholds=gm.thresholds)
        stats.gate_fired += fired["fired"]
        stats.gate_acquired += fired["acquired"]
        stats.gate_refused += fired["refused"]
        stats.gate_released += fired["released"]
        stats.frames += 1

    _update_traversability(gm)
    return stats


def _update_traversability(gm) -> None:
    rings = [(slice(r.offset, r.offset + r.side * r.side), r.side)
             for r in gm.allocation.rings]
    traversability.update(gm.soa, gm.schedule, rings, gm.thresholds)


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
        frames = run_sequence(gm, scans_factory()).frames
        results.append((s, gm, evaluate(gm, reference, frames)))
    return results


def _nanmean(values) -> float:
    """np.nanmean over an all-nan list warns and returns nan; a ring nobody
    drove through is the ordinary case here, not an anomaly worth a warning."""
    finite = [v for v in values if not np.isnan(v)]
    return float(np.mean(finite)) if finite else float("nan")


def memory_vs_regret_row(result: Result, regret=None) -> dict:
    """One point of the Day-4 headline curve. Math §8.2.

    x is memory, y is R(S). The curve has a knee and the schedule should sit
    at it: "below 8.9 MB the plan is unchanged -- regret is exactly zero --
    and above the knee it degrades measurably."

    `regret` is a `plan_regret.Regret`. Its `unknown_fraction` travels with it
    on purpose: zero regret along a mostly-unknown path says the sequence was
    too short to fill the map, not that the coarsening was free, and the two
    numbers are only meaningful side by side.
    """
    rings = sorted(result.rmse_cm)
    finite = [result.rmse_cm[r] for r in rings if not np.isnan(result.rmse_cm[r])]
    return {
        "schedule": result.schedule_name,
        "megabytes": result.bytes_allocated / 1e6,
        "logical_cells": result.logical_cells,
        "worst_ring_rmse_cm": max(finite) if finite else float("nan"),
        "mean_rho": _nanmean([result.coarsening[r]["rho"] for r in rings]),
        "regret": None if regret is None else regret.regret,
        "frechet_m": None if regret is None else regret.frechet_m,
        "unknown_fraction": None if regret is None else regret.unknown_fraction,
        "blocked_on_reference": None if regret is None else regret.blocked_on_reference,
    }
