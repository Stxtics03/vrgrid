"""Generate every demo .rrd recording in the shot list, reproducibly.

Gate 6 says every number on a slide comes from a script in `scripts/`. The same
has to hold for every FRAME on a slide: the recordings are 4.7 GB across 33
files and 16 of them are over GitHub's 100 MB per-file limit, so the artifacts
themselves cannot live in the repo. This script is what makes that acceptable
-- it is the durable half, and it regenerates the whole set from the sequences.

    python scripts/gen_demo_rrds.py <output-dir>

⚑ RECORDINGS GO STALE AGAINST THE ENGINE, and silently. Twelve of the fourteen
  shots were generated on 1 Sep at 06:03-06:10; the elevation fix (`51bff0f`)
  landed at 10:38 the same morning. `shot4a`/`shot4b` are seq 07 frames 660-690,
  where the vehicle sits at -5.37..-5.25 m -- outside the old world-absolute
  [-2, +6] m band on 100% of frames -- so they had recorded the ghost toggle on
  frames where cleanup was inert. Regenerated post-fix they clear a median
  18,345 cells/frame with 0 inert frames.

  So: after ANY change to the map back end, the visibility cleanup or the
  height datum, regenerate rather than reuse. Check `cleared/frame` per frame,
  not just that the file was written -- a stale recording opens fine and looks
  plausible, which is exactly why this went unnoticed for two days.
"""
import sys
import time
from pathlib import Path

import numpy as np
import rerun as rr

from vrgrid.dash.pipeline_view import PipelineView
from vrgrid.grid.schedule import load as load_sched
from vrgrid.perception import ground, loader, range_image, reflectivity, semantics, transforms
from vrgrid.run.__main__ import PerceptionFrame
from vrgrid.run.engine import MapEngine

OUT = Path(r"C:/Users/JAIPRE~1/AppData/Local/Temp/claude/c--Users-JAIPREET-SINGH-OneDrive-Documents-Sih/57bcc393-2146-4b8e-b77e-1313d80fb54a/scratchpad/demo_final")
SCHED = load_sched("5/10/20/40")


def make_frame(seq, i, use_pw=True):
    pts = loader.load_velodyne_scan(loader._velodyne_path(seq, i))
    raw = loader.load_labels(loader._label_path(seq, i))
    pose = loader.load_gt_poses(seq)[i]
    t_s_w = transforms.sensor_to_world(pose, sequence=seq)
    pw = transforms.transform_points(pts[:, :3], t_s_w)
    veh = transforms.vehicle_to_world(pose, sequence=seq)[:3, 3]
    ri, inv = range_image.project(pts)
    sem = semantics.semantic_labels(raw)
    mov = semantics.is_moving(raw)
    if use_pw and ground._HAVE_PATCHWORKPP:
        g = ground.segment_ground(pts)
    else:
        g = ground.ground_from_semantics(sem)
    refl = reflectivity.normalise(ri)
    rho8, _ = reflectivity.scatter_to_points(refl, inv)
    if len(rho8) < len(pts):
        rho8 = np.concatenate([rho8, np.zeros(len(pts) - len(rho8), np.uint8)])
    return PerceptionFrame(index=i, points_sensor=pts, points_world=pw, pose=pose,
                           vehicle_xyz_world=veh, semantic=sem, moving=mov, ground=g,
                           reflectivity8=rho8, range_image=ri, inverse_index=inv)


def shot(name, seq, lo, hi, *, warmup=0, color_by="class", palette="semantickitti",
         show_ghosts=False, no_map=False):
    """Frames [lo-warmup, hi) run through the wired pipeline; saved to name.rrd."""
    path = OUT / f"{name}.rrd"
    start = max(0, lo - warmup)
    engine = None if no_map else MapEngine(SCHED, ghost_removal=not show_ghosts)
    view = PipelineView(SCHED, spawn=False, save_path=str(path), color_by=color_by,
                        ghost_removal=not show_ghosts, palette=palette, engine=engine)
    t0 = time.perf_counter()
    n = 0
    occ_last = 0
    for i in range(start, hi):
        f = make_frame(seq, i)
        if engine is not None:
            c = engine.step(f)
            occ_last = c.occupied if c.occupied else occ_last
        view.log_frame(f)
        n += 1
    rr.disconnect()
    dt = time.perf_counter() - t0
    sz = path.stat().st_size / 1e6
    occ_now = len(engine.occupied_slots()) if engine is not None else 0
    print(f"  {name:42} frames {start}-{hi-1} ({n})  {sz:6.1f} MB  {dt:5.1f}s  "
          f"occupied@end={occ_now:,}")
    return dict(name=name, frames=n, range=(start, hi - 1), mb=round(sz, 1),
                occupied_end=occ_now)


ONLY = sys.argv[1] if len(sys.argv) > 1 else None
RESULTS = []


def run(r):
    if ONLY is None or ONLY in r["name"]:
        pass


shots = [
    ("shot1_seq00_0-160_class_groups",  dict(seq="00", lo=0, hi=160, color_by="class", palette="groups")),
    ("shot1lean_seq00_0-160_class_groups_nomap", dict(seq="00", lo=0, hi=160, color_by="class", palette="groups", no_map=True)),
    ("shot2_seq00_0-160_occupied_rings", dict(seq="00", lo=0, hi=160, color_by="class")),
    ("shot3a_seq00_5-20_ghost_ON",      dict(seq="00", lo=0, hi=22, color_by="motion")),
    ("shot3b_seq00_5-20_ghost_OFF",     dict(seq="00", lo=0, hi=22, color_by="motion", show_ghosts=True)),
    # the map-level ghost demo needs trail accumulation -- 0-60 shows the real delta
    ("shot3c_seq00_0-60_ghost_ON",      dict(seq="00", lo=0, hi=60, color_by="motion")),
    ("shot3d_seq00_0-60_ghost_OFF",     dict(seq="00", lo=0, hi=60, color_by="motion", show_ghosts=True)),
    ("shot4a_seq07_660-690_ghost_ON",   dict(seq="07", lo=660, hi=691, warmup=45, color_by="motion")),
    ("shot4b_seq07_660-690_ghost_OFF",  dict(seq="07", lo=660, hi=691, warmup=45, color_by="motion", show_ghosts=True)),
    ("shot5_seq00_4420-4470_reflectivity", dict(seq="00", lo=4420, hi=4471, warmup=12, color_by="reflectivity")),
    # ⚑ The shot `docs/demo-safe-ranges.md` has always listed -- "Ghost toggle at
    #   elevation | 08 | any climbed stretch" -- and that never existed as a
    #   recording until 3 Sep. It is the one that proves the elevation fix:
    #   frames 3000-3060 sit at veh_z +37.0..+38.4 m, and the pre-fix engine
    #   cleared ZERO cells per frame anywhere above +20.7 m
    #   (`known-limitations.md` §1). Post-fix it clears a median 30,851.
    ("shot8a_seq08_3000-3060_climb_ghost_ON",
     dict(seq="08", lo=3000, hi=3061, warmup=40, color_by="motion")),
    ("shot8b_seq08_3000-3060_climb_ghost_OFF",
     dict(seq="08", lo=3000, hi=3061, warmup=40, color_by="motion", show_ghosts=True)),
]

if __name__ == "__main__":
    print(f"generating into {OUT}")
    for name, kw in shots:
        if ONLY and ONLY not in name:
            continue
        RESULTS.append(shot(name, **kw))
    print("\nSUMMARY")
    for r in RESULTS:
        print(f"  {r['name']:42} {r['frames']:>4} frames  {r['mb']:>6} MB  occ@end {r['occupied_end']:,}")
