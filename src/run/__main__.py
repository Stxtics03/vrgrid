"""Pipeline entry point -- `python -m vrgrid.run --seq 00 --frames 50`.

Thin wiring only: parse arguments, pull frames from the loader, run each
perception stage in order, hand the result to the dashboard. No algorithm
lives here -- it belongs to nobody and everybody. Keep it that way.

Stages (all JP's, `src/perception/`):

    loader.scans()          raw points + raw .label + GT pose, per frame
    transforms              sensor -> vehicle -> world  (docs/frames.md)
    range_image.project()   64x512 spherical image + inverse index (sensor frame)
    semantics               semantic_labels() 19-class + is_moving()  (GT .label)
    ground.segment_ground() Patchwork++ ground / non-ground mask
    reflectivity.normalise() rho_hat -> one byte  (KITTI: rho_hat = I; the
                             eq-31 r^2/cos terms are firmware-redundant here)

then the map back end, in `engine.MapEngine` (see that file for the order):

    bin -> scatter -> fuse -> visibility cleanup -> shift

The dashboard (`--viz` / `--save`) renders the real per-frame output, replacing
the Day-0 synthetic plane/boxes/slope one layer at a time via `--color-by`.

⚑ `--show-ghosts` is the Gate 3 toggle and it now drives BOTH halves: the
  viewer keeps the moving returns in the main cloud, AND the map stops running
  §10.4, so the ghost trails stay in the cells. Until the engine existed it
  drove only the first, which filters the input cloud on the ground-truth
  `moving-*` label and demonstrates nothing about the mapping engine.
"""

import argparse
import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
from vrgrid.grid import schedule as schedule_mod
from vrgrid.run.engine import MapEngine


@dataclass
class PerceptionFrame:
    """One frame after every perception stage. Arrays are point-aligned to
    `points_sensor` unless noted."""

    index: int
    points_sensor: np.ndarray      # (N, 4) raw x,y,z,intensity
    points_world: np.ndarray       # (N, 3) after sensor->world
    pose: np.ndarray               # (3, 4) GT
    vehicle_xyz_world: np.ndarray  # (3,) vehicle origin in world
    semantic: np.ndarray           # (N,) 19-class, -1 ignore
    moving: np.ndarray             # (N,) bool
    ground: np.ndarray             # (N,) bool
    reflectivity8: np.ndarray      # (N,) uint8, 0 where not projected this frame
    range_image: np.ndarray        # (H, W, 5)
    inverse_index: np.ndarray      # (H, W) int32


def iter_pipeline(seq: str, max_frames: int | None, use_patchworkpp: bool = True,
                  timer=None):
    """Yield a PerceptionFrame per scan of `seq`.

    `timer` is an optional `gpu.timing.Timer`. Passing one names each stage
    with the spelling in `timing.STAGES`, which is what lets
    `scripts/timing_table.py --seq` print a whole-frame latency table instead
    of the back end alone. The stage names were fixed in `timing.py` on Day 0
    precisely so the front end and the map would not invent two spellings of
    "range image"; this is the other half of that.

    `loader.scans` is a generator, so the `load` stage times the pull of one
    scan off it rather than the whole sequence -- which is the per-frame cost
    the 10 Hz budget is about.
    """
    from vrgrid.perception import ground, loader, range_image, reflectivity, semantics, transforms

    def stage(name):
        return timer.stage(name) if timer is not None else nullcontext()

    scans = loader.scans(seq, max_frames=max_frames)
    i = 0
    while True:
        # Timed by hand rather than with `stage("load")`, because the pull that
        # EXHAUSTS the generator must not be recorded: it is not a frame, and
        # counting it gave `load` one more sample than there were frames and
        # dragged its p99 down with a near-zero reading.
        t0 = time.perf_counter()
        item = next(scans, None)
        if item is None:
            break
        if timer is not None:
            timer.record("load", (time.perf_counter() - t0) * 1e3)
        points, raw_labels, pose = item

        with stage("transform"):
            t_s_w = transforms.sensor_to_world(pose, sequence=seq)
            points_world = transforms.transform_points(points[:, :3], t_s_w)
            vehicle_xyz = transforms.vehicle_to_world(pose, sequence=seq)[:3, 3]

        with stage("range_image"):
            ri, inv = range_image.project(points)
        with stage("semantics"):
            semantic = semantics.semantic_labels(raw_labels)
        with stage("motion"):
            moving = semantics.is_moving(raw_labels)

        with stage("ground"):
            if use_patchworkpp and ground._HAVE_PATCHWORKPP:
                gmask = ground.segment_ground(points)
            else:
                gmask = ground.ground_from_semantics(semantic)

        with stage("reflectivity"):
            refl = reflectivity.normalise(ri)
            rho8, _ = reflectivity.scatter_to_points(refl, inv)
            if len(rho8) < len(points):  # pad points that never projected
                rho8 = np.concatenate([rho8, np.zeros(len(points) - len(rho8), np.uint8)])

        # grid.scatter(soa, points, semantic, pose, schedule)  -- STUB, see module docstring

        yield PerceptionFrame(
            index=i,
            points_sensor=points,
            points_world=points_world,
            pose=pose,
            vehicle_xyz_world=vehicle_xyz,
            semantic=semantic,
            moving=moving,
            ground=gmask,
            reflectivity8=rho8,
            range_image=ri,
            inverse_index=inv,
        )
        i += 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vrgrid.run")
    p.add_argument("--seq", default="00", help="SemanticKITTI sequence")
    p.add_argument("--schedule", default="5/10/20/40", help="ring schedule name")
    p.add_argument("--thresholds", default="configs/thresholds.yaml")
    p.add_argument("--frames", type=int, default=None, help="stop after N frames")
    p.add_argument("--viz", action="store_true", help="open the Rerun dashboard")
    p.add_argument("--save", default=None, help="write a Rerun .rrd recording here")
    p.add_argument(
        "--color-by",
        default="class",
        choices=["intensity", "class", "motion", "ground", "reflectivity"],
        help="how the dashboard colours the point cloud",
    )
    p.add_argument("--show-ghosts", action="store_true",
                   help="Gate 3 toggle OFF: keep moving points in the main cloud "
                        "and stop running the map's visibility cleanup, so ghost "
                        "trails stay in the cells (default: both on)")
    p.add_argument("--no-map", action="store_true",
                   help="perception only; skip the map back end entirely")
    p.add_argument("--clip-class-ids", action="store_true",
                   help="clip semantic ids to 15 so fusion's 4-bit candidate "
                        "accepts them (math §10.2). Corrupts the class layer; "
                        "the real fix is the 5/3 split, a room decision")
    p.add_argument("--palette", default="semantickitti", choices=["semantickitti", "groups"],
                   help="class colours: the 19-class standard, or 7 colourblind-safe groups")
    p.add_argument("--no-patchworkpp", action="store_true", help="use the semantic-class ground proxy")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    sched = schedule_mod.load(args.schedule)
    print(
        f"schedule {sched.name}: {len(sched.rings)} rings, "
        f"{sched.total_cells:,} cells, {sched.total_cells * 12 / 1e6:.2f} MB"
    )

    view = None
    if args.viz or args.save:
        from vrgrid.dash.pipeline_view import PipelineView

        view = PipelineView(sched, spawn=args.viz, save_path=args.save,
                            color_by=args.color_by, ghost_removal=not args.show_ghosts,
                            palette=args.palette)

    engine = None
    if not args.no_map:
        engine = MapEngine(sched, ghost_removal=not args.show_ghosts,
                           clip_class_ids=args.clip_class_ids)
        print(f"map: {engine.handle.allocated_slots:,} slots preallocated, "
              f"ghost removal {'OFF' if args.show_ghosts else 'ON'}")

    n, cleared, protected = 0, 0, 0
    for frame in iter_pipeline(args.seq, args.frames, use_patchworkpp=not args.no_patchworkpp):
        counters = engine.step(frame) if engine is not None else None
        if counters is not None:
            cleared += counters.cleared
            protected += counters.protected
        if view is not None:
            view.log_frame(frame)
        n += 1
        if n % 20 == 0:
            msg = f"  frame {frame.index}: {len(frame.points_sensor):,} pts"
            if counters is not None:
                msg += (f", {counters.occupied:,} occupied cells, "
                        f"{counters.cleared:,} cleared, {counters.protected:,} protected")
            print(msg)

    print(f"done: {n} frames, sequence {args.seq}")
    if engine is not None:
        # The number the Gate 3 demo is actually about. With --show-ghosts it
        # is zero by construction, which is the point of printing it.
        print(f"ghost removal: {cleared:,} cells cleared, {protected:,} spared by "
              f"the current-return guard")
    if args.save:
        print(f"recording written to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
