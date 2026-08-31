"""Rerun view of the real perception pipeline. [JP]

Replaces the Day-0 synthetic plane/boxes/slope (`demo_synthetic.py`) with the
actual per-frame output of `vrgrid.run`, one interpretation layer at a time via
`color_by`:

    intensity     raw Velodyne intensity, greyscale  (the "no semantics" baseline)
    class         semantic_labels() 19-class colours
    motion        is_moving(): static dim, moving bright red
    ground        segment_ground(): ground tan, non-ground steel blue
    reflectivity  reflectivity.normalise(): rho_hat byte, greyscale

Ghost toggle
------------
`world/points` holds the ghost-free cloud; the moving points are logged
separately to `world/ghosts`. Toggling `world/ghosts` visibility in the viewer
(the eye icon in the entity panel) is the ghost toggle -- ON shows the trails
behind moving objects, OFF removes them, static points untouched either way.

`get_display_points(frame, ghost_removal)` is the single swap point: it decides
which points land in `world/points`, currently by branching on the raw motion
mask `frame.moving`. When Aakash's `scatter()`/`fuse()` exist, only that
function's body changes -- it queries the grid's transient layer instead -- and
none of the logging wiring moves.

Ring boundaries and the blind cone are logged under the vehicle transform, so
they track the vehicle. Points are world-frame and accumulate on the timeline.
"""

import numpy as np
import rerun as rr

from .demo_synthetic import load_schedule

# Every colour below is defined in `palettes.py`, which imports no rerun: the
# CVD audit (`cvd.py`) and tests/test_cvd.py check these numbers in CI, where
# rerun-sdk (the optional `[dash]` extra) is not installed. Imported rather than
# defined here so the dashboard and the audit cannot drift apart -- and so this
# module stays the import path the rest of the dashboard already uses.
from .palettes import (
    _CLASS_LUT,
    _GROUP_LUT,
    GHOST_RGB,
    GROUP_MEMBERS,
    GROUP_NAMES,
    GROUP_RGB,
)

PALETTES = ("semantickitti", "groups")


def legend_markdown(palette: str) -> str:
    """Which raw classes fall into each colour, so nothing is lost in grouping."""
    if palette == "groups":
        lines = ["**Palette: groups** (colourblind-safe)", "", "| group | raw classes |", "|---|---|"]
        lines += [f"| {g} | {', '.join(GROUP_MEMBERS[g])} |" for g in GROUP_NAMES]
        return "\n".join(lines)
    return "**Palette: semantickitti** -- the standard 19-class map (not colourblind-safe)"


def _frame_colors(frame, color_by: str, palette: str = "semantickitti") -> np.ndarray:
    """(N, 3) uint8 colour per point of `frame`, for the chosen layer.

    `palette` only affects the `class` layer: "semantickitti" (default) or
    "groups" (19 classes -> 7 colourblind-safe super-groups).
    """
    if color_by == "intensity":
        g = np.clip(frame.points_sensor[:, 3] * 255.0 * 1.5, 0, 255).astype(np.uint8)
        return np.repeat(g[:, None], 3, axis=1)
    if color_by == "class":
        idx = np.clip(frame.semantic + 1, 0, 19)
        if palette == "groups":
            return GROUP_RGB[_GROUP_LUT[idx]]
        return _CLASS_LUT[idx]
    if color_by == "motion":
        c = np.full((len(frame.moving), 3), 90, dtype=np.uint8)  # static: neutral grey
        c[frame.moving] = GHOST_RGB
        return c
    if color_by == "ground":
        # tan vs steel-blue -- an orange/blue pair, the axis all three CVD types
        # preserve; min Delta-E 55 under every simulation (dashboard/cvd.py).
        c = np.empty((len(frame.ground), 3), dtype=np.uint8)
        c[frame.ground] = (170, 130, 90)
        c[~frame.ground] = (70, 130, 180)
        return c
    if color_by == "reflectivity":
        return np.repeat(frame.reflectivity8[:, None], 3, axis=1)
    raise ValueError(f"unknown color_by {color_by!r}")


COLOR_BY = ("intensity", "class", "motion", "ground", "reflectivity")


def get_display_points(frame, ghost_removal: bool, color_by: str = "class",
                       palette: str = "semantickitti"):
    """Points + colours for the `world/points` entity.

    Args:
        frame: a run.PerceptionFrame.
        ghost_removal: True drops the points flagged by `frame.moving`.
        color_by: which colour layer (see COLOR_BY).
        palette: "semantickitti" (default) or "groups" -- only affects `class`.

    Returns:
        (xyz (M, 3) float32, colors (M, 3) uint8).

    PLACEHOLDER: the mask is `frame.moving` (raw per-frame motion labels). When
    the grid's transient layer exists this becomes a grid query -- e.g.
    ``keep = ~grid.is_transient(frame.points_world)`` -- and callers do not change.
    """
    keep = ~frame.moving if ghost_removal else np.ones(len(frame.moving), dtype=bool)
    xyz = frame.points_world[keep].astype(np.float32)
    colors = _frame_colors(frame, color_by, palette)[keep]
    return xyz, colors


class PipelineView:
    def __init__(self, schedule, spawn: bool = False, save_path: str | None = None,
                 color_by: str = "class", ghost_removal: bool = True,
                 palette: str = "semantickitti", schedule_yaml: str | None = None):
        self.color_by = color_by
        self.ghost_removal = ghost_removal
        self.palette = palette
        rr.init("vrgrid_pipeline", spawn=spawn)
        if save_path:
            rr.save(save_path)

        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log(
            "instructions",
            rr.TextDocument(
                "Ghost toggle: show/hide the `world/ghosts` entity (eye icon in the "
                "entity panel).\nvisible = ghost removal OFF (trails behind moving "
                "objects)\nhidden  = ghost removal ON",
                media_type=rr.MediaType.MARKDOWN,
            ),
            static=True,
        )
        rr.log("legend", rr.TextDocument(legend_markdown(palette),
                                         media_type=rr.MediaType.MARKDOWN), static=True)
        self._log_rings(load_schedule() if schedule_yaml is None else load_schedule(schedule_yaml))
        self._log_blind_cone()

    def _log_rings(self, sched: dict):
        for ring in sched.get("rings", []):
            hw = ring["half_width_m"]
            th = np.linspace(0, 2 * np.pi, 129)
            strip = np.stack([hw * np.cos(th), hw * np.sin(th), np.zeros_like(th)], axis=1)
            rr.log(
                f"world/vehicle/rings/ring_{ring['ring']}",
                rr.LineStrips3D([strip.astype(np.float32)], colors=[220, 220, 220], radii=0.04),
                static=True,
            )

    def _log_blind_cone(self, radius_m: float = 3.74):
        th = np.linspace(0, 2 * np.pi, 65)
        strip = np.stack([radius_m * np.cos(th), radius_m * np.sin(th), np.zeros_like(th)], axis=1)
        rr.log(
            "world/vehicle/blind_cone",
            rr.LineStrips3D([strip.astype(np.float32)], colors=[230, 60, 60], radii=0.05),
            static=True,
        )

    def log_frame(self, frame):
        rr.set_time("frame", sequence=frame.index)

        xyz, colors = get_display_points(frame, self.ghost_removal, self.color_by, self.palette)
        rr.log("world/points", rr.Points3D(xyz, colors=colors, radii=0.03))

        # The removed set, on its own entity -- this is what the demo toggles.
        ghosts = frame.points_world[frame.moving].astype(np.float32)
        rr.log("world/ghosts", rr.Points3D(ghosts, colors=list(GHOST_RGB), radii=0.09))

        # vehicle transform: origin + heading from the GT pose (world-frame yaw)
        fwd_world = frame.pose[:3, :3] @ np.array([0.0, 0.0, 1.0])  # camera z = forward
        yaw = np.arctan2(-fwd_world[0], fwd_world[2])  # into the z-up world convention
        rr.log("world/vehicle", rr.Transform3D(
            translation=frame.vehicle_xyz_world.astype(np.float32),
            rotation=rr.RotationAxisAngle(axis=[0, 0, 1], angle=float(yaw)),
        ))
        rr.log("world/vehicle/marker", rr.Points3D([[0, 0, 0]], colors=[40, 220, 40], radii=0.4))
