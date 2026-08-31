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
    grid.scatter()          << STUB on this branch -- Aakash's grid is not here yet

The dashboard (`--viz` / `--save`) renders the real per-frame output, replacing
the Day-0 synthetic plane/boxes/slope one layer at a time via `--color-by`.
"""

import argparse
from dataclasses import dataclass

import numpy as np

from vrgrid.grid import schedule as schedule_mod


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


def iter_pipeline(seq: str, max_frames: int | None, use_patchworkpp: bool = True):
    """Yield a PerceptionFrame per scan of `seq`."""
    from vrgrid.perception import ground, loader, range_image, reflectivity, semantics, transforms

    for i, (points, raw_labels, pose) in enumerate(loader.scans(seq, max_frames=max_frames)):
        t_s_w = transforms.sensor_to_world(pose, sequence=seq)
        points_world = transforms.transform_points(points[:, :3], t_s_w)
        vehicle_xyz = transforms.vehicle_to_world(pose, sequence=seq)[:3, 3]

        ri, inv = range_image.project(points)
        semantic = semantics.semantic_labels(raw_labels)
        moving = semantics.is_moving(raw_labels)

        if use_patchworkpp and ground._HAVE_PATCHWORKPP:
            gmask = ground.segment_ground(points)
        else:
            gmask = ground.ground_from_semantics(semantic)

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
                   help="keep moving points in the main cloud (default: split to world/ghosts)")
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

    n = 0
    for frame in iter_pipeline(args.seq, args.frames, use_patchworkpp=not args.no_patchworkpp):
        if view is not None:
            view.log_frame(frame)
        n += 1
        if n % 20 == 0:
            print(f"  frame {frame.index}: {len(frame.points_sensor):,} pts")

    print(f"done: {n} frames, sequence {args.seq}")
    if args.save:
        print(f"recording written to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
