"""Rerun dashboard -- `python -m vrgrid.dash`. [JP]

Runs as a SEPARATE PROCESS from the pipeline. If the dashboard falls over two
days before submission the framework must still run and still produce numbers.

    python -m vrgrid.dash                       synthetic mock (Day 0, no data)
    python -m vrgrid.dash --seq 00              real pipeline, colour by class
    python -m vrgrid.dash --seq 00 --color-by ground --frames 60
    python -m vrgrid.dash --seq 07 --start-frame 660 --frames 30 --save shot.rrd
    python -m vrgrid.dash --seq 00 --save run.rrd      headless -> open with `rerun run.rrd`

Shows: the point cloud coloured by the chosen layer, ring boundaries and the
blind cone tracking the vehicle, and the vehicle pose per frame on the timeline.

Ghost toggle: the moving points are logged to `world/ghosts`; toggle that
entity's visibility in the viewer. `--show-ghosts` puts them back in the main
cloud instead (for a "raw pipeline output" view with nothing to toggle).
"""

import argparse


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="vrgrid.dash")
    p.add_argument("--seq", default=None, help="run the real pipeline on this sequence")
    p.add_argument("--frames", type=int, default=None, help="stop after N frames")
    p.add_argument("--start-frame", type=int, default=0,
                   help="start from this frame index (default 0); --frames counts from here")
    p.add_argument("--schedule", default="5/10/20/40")
    p.add_argument(
        "--color-by",
        default="class",
        choices=["intensity", "class", "motion", "ground", "reflectivity"],
    )
    p.add_argument("--palette", default="semantickitti", choices=["semantickitti", "groups"],
                   help="class colours: the 19-class standard, or 7 colourblind-safe groups")
    p.add_argument("--save", default=None, help="write a .rrd recording instead of spawning a viewer")
    p.add_argument("--show-ghosts", action="store_true",
                   help="keep moving points in world/points AND stop the map's "
                        "visibility cleanup, so ghost trails stay in the cells "
                        "(default: both on)")
    p.add_argument("--no-map", action="store_true",
                   help="perception only; skip the map back end and its occupied-cell surface")
    p.add_argument("--no-patchworkpp", action="store_true")
    p.add_argument("--features", action="store_true",
                   help="draw the curb/pothole (math 7.4) and confidence (7.5) "
                        "layers. Recomputed every 20 frames and once at the end, "
                        "not every frame: the detector is a full-window pass and "
                        "costs ~1.1 s")
    args = p.parse_args(argv)

    if args.seq is None:
        from .demo_synthetic import main as mock_main

        mock_main()
        return

    from vrgrid.grid import schedule as schedule_mod
    from vrgrid.run.__main__ import iter_pipeline
    from vrgrid.run.engine import MapEngine

    from .pipeline_view import PipelineView

    sched = schedule_mod.load(args.schedule)
    engine = None if args.no_map else MapEngine(sched, ghost_removal=not args.show_ghosts)
    view = PipelineView(sched, spawn=args.save is None, save_path=args.save,
                        color_by=args.color_by, ghost_removal=not args.show_ghosts,
                        palette=args.palette, engine=engine, features=args.features)
    n = 0
    for frame in iter_pipeline(args.seq, args.frames, use_patchworkpp=not args.no_patchworkpp,
                               start_frame=args.start_frame):
        if engine is not None:
            engine.step(frame)
        view.log_frame(frame)
        n += 1
    view.log_features()   # final state; no-op unless --features
    start = f" from frame {args.start_frame}" if args.start_frame else ""
    print(f"{n} frames from sequence {args.seq}{start}"
          + (f" -> {args.save}" if args.save else ""))


if __name__ == "__main__":
    main()
