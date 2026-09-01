"""Side-by-side: the variable-resolution grid vs a dense 5 cm 3-D voxel grid. [JP]

Standalone comparison artifact -- NOT part of the live dashboard, not imported
by `pipeline_view.py` or `run/__main__.py`. Run it directly:

    python -m vrgrid.dash.dense3d_comparison --seq 00 --frames 20 --save cmp.rrd

It renders two entity trees over the same seq-00 frames, offset in Y so they sit
next to each other in one view:

    world/map/occupied   the real MapEngine occupied cells (5/10/20/40 cm),
                         cropped to the comparison footprint
    dense3d/occupied     gpu.baseline.DenseVoxelBaseline's occupied voxels
                         (uniform 5 cm), placed via the verified reverse map
                         in dense3d_debug.py, shifted +COMPARE_OFFSET_M in Y

**Scale disclaimer.** This uses a reduced `footprint_m` (20 m, ~25 MB) because
that is what a dev machine can actually allocate -- the real dense baseline is
200 x 200 x 8 m = 2.56 GB. The box-count ratio here is a *local* illustration of
foveation, NOT the report's 286x figure. The 286x is the full-grid byte ratio
and is shown by the live memory panel (`_config.memory_overlay_markdown`) /
`scripts/memory_table.py`. This render is honest about being small.

Reads only from `gpu.baseline` (no modification) and `dense3d_debug`.
"""

import argparse

import numpy as np
import rerun as rr
from vrgrid.gpu.baseline import Z_MIN_M, allocate_dense3d, dense3d_voxels
from vrgrid.grid.schedule import load as load_schedule
from vrgrid.run.engine import MapEngine

from .dense3d_debug import occupied_voxel_centres

# how far apart, in +Y metres, to place the dense tree from the variable one
COMPARE_OFFSET_M = 50.0

# steel-blue ramp by cell size -- lets the ring boundaries read in the variable
# grid; the dense grid is a single amber so the "one uniform size" is obvious
_VAR_RAMP = {0.05: (70, 130, 180), 0.10: (110, 160, 195),
             0.20: (150, 185, 210), 0.40: (190, 210, 225)}
_DENSE_RGB = (230, 159, 0)


def _cell_m_per_slot(engine, slots):
    out = np.full(len(slots), engine.sched.base_cell_m, np.float64)
    for layout in engine.handle.rings:
        m = (slots >= layout.offset) & (slots < layout.offset + layout.slots)
        out[m] = layout.cell_m
    return out


def run(seq: str, n_frames: int, footprint_m: float, save_path: str,
        spawn: bool = False):
    from vrgrid.run.__main__ import iter_pipeline

    sched = load_schedule("5/10/20/40")
    engine = MapEngine(sched, ghost_removal=True)
    dense = allocate_dense3d(footprint_m=footprint_m, allow_unsafe=True)
    half_fp = (dense.side // 2) * dense.cell_m       # +/- extent of the footprint

    rr.init("vrgrid_dense3d_comparison", spawn=spawn)
    if save_path:
        rr.save(save_path)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    full_voxels = dense3d_voxels()                   # 2.56e9, the real baseline
    origin = None
    last = {}
    for i, frame in enumerate(iter_pipeline(seq, n_frames, use_patchworkpp=True)):
        engine.step(frame)
        if origin is None:
            origin = np.asarray(frame.vehicle_xyz_world, float).copy()

        # dense baseline: ingest this frame's world points, recentred on the
        # frame-0 vehicle position so the fixed footprint stays over the scene
        pw = frame.points_world - origin
        dense.ingest(pw[:, 0], pw[:, 1], pw[:, 2])

        rr.set_time("frame", sequence=i)

        # --- variable grid, cropped to the same footprint --------------------
        slots, x, y, z = engine.occupied_cells()
        vx, vy, vz = x - origin[0], y - origin[1], z - origin[2]
        crop = (np.abs(vx) <= half_fp) & (np.abs(vy) <= half_fp) & \
               (vz >= Z_MIN_M) & (vz <= Z_MIN_M + dense.layers * dense.cell_m)
        cm = _cell_m_per_slot(engine, slots)[crop]
        vcent = np.stack([vx[crop], vy[crop], vz[crop]], axis=1).astype(np.float32)
        vhalf = np.stack([cm / 2, cm / 2, np.full_like(cm, 0.02)], axis=1).astype(np.float32)
        vcol = np.array([_VAR_RAMP.get(round(float(c), 2), (120, 120, 120)) for c in cm],
                        np.uint8) if len(cm) else np.zeros((0, 3), np.uint8)
        rr.log("world/map/occupied",
               rr.Boxes3D(centers=vcent, half_sizes=vhalf, colors=vcol, fill_mode="solid"))

        # --- dense baseline, same footprint, shifted +Y for side-by-side -----
        dcent = occupied_voxel_centres(dense)
        dcent = dcent + np.array([0.0, COMPARE_OFFSET_M, 0.0])
        dhalf = np.full((len(dcent), 3), dense.cell_m / 2, np.float32)
        rr.log("dense3d/occupied",
               rr.Boxes3D(centers=dcent.astype(np.float32), half_sizes=dhalf,
                          colors=[_DENSE_RGB], fill_mode="solid"))

        last = dict(frame=i, var_boxes=int(crop.sum()),
                    dense_boxes=len(dcent),
                    var_total_occupied=len(slots))

        rr.log("comparison", rr.TextDocument(
            _overlay(footprint_m, half_fp, dense, full_voxels, last),
            media_type=rr.MediaType.MARKDOWN), static=(i == 0))

    rr.disconnect()
    return last


def _overlay(footprint_m, half_fp, dense, full_voxels, last):
    ratio = last["dense_boxes"] / last["var_boxes"] if last["var_boxes"] else float("nan")
    return "\n".join([
        "**Variable grid vs dense 5 cm 3-D voxel grid**",
        "",
        f"_Reduced-footprint illustration: {footprint_m:g} x {footprint_m:g} x "
        f"{dense.layers * dense.cell_m:g} m ({dense.units / 1e6:.1f} M voxels, "
        f"{dense.claimed_bytes / 1e6:.1f} MB). The **real** dense baseline is "
        f"200 x 200 x 8 m = {full_voxels / 1e9:.2f} G voxels = "
        f"{full_voxels / 1e9:.2f} GB -- that byte ratio (286x) is on the live "
        f"memory panel, NOT this number._",
        "",
        "| | boxes drawn (same {0:g} m footprint, centred on the frame-0 "
        "vehicle position) |".format(2 * half_fp),
        "|---|---|",
        f"| `world/map/occupied` (variable, 5/10/20/40 cm) | **{last['var_boxes']:,}** |",
        f"| `dense3d/occupied` (uniform 5 cm) | **{last['dense_boxes']:,}** |",
        f"| local box-count ratio | **{ratio:.1f}x** |",
        "",
        "_The ratio grows over the clip: early on the vehicle is in the footprint "
        "and both grids see it at ~5 cm; as the vehicle drives on, the variable "
        "grid coarsens this now-distant region (5 -> 10 -> 20 -> 40 cm) and the "
        "toroidal window releases the far part, while the dense grid keeps every "
        "voxel at 5 cm forever. So the late ratio is foveation + bounded memory, "
        "not foveation alone._",
        "",
        f"frame {last['frame']}   ·   variable grid total occupied cells "
        f"(full 100 m map): {last['var_total_occupied']:,}",
    ])


def main(argv=None):
    p = argparse.ArgumentParser(prog="vrgrid.dash.dense3d_comparison")
    p.add_argument("--seq", default="00")
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--footprint", type=float, default=20.0,
                   help="dense-grid footprint in m (kept small: 20 m ~= 25 MB)")
    p.add_argument("--save", default=None, help="write a .rrd here")
    p.add_argument("--spawn", action="store_true")
    a = p.parse_args(argv)
    last = run(a.seq, a.frames, a.footprint, a.save, a.spawn)
    print(f"frame {last['frame']}: variable {last['var_boxes']:,} boxes  "
          f"dense {last['dense_boxes']:,} boxes  "
          f"ratio {last['dense_boxes'] / max(last['var_boxes'], 1):.1f}x  "
          f"(footprint {a.footprint:g} m)")
    if a.save:
        print(f"wrote {a.save}")


if __name__ == "__main__":
    main()
