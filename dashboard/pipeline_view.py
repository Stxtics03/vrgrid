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
from vrgrid.cell import OCC_FREE, OCC_UNKNOWN

from ._config import (
    blind_cone_radius_m,
    memory_overlay_markdown,
    schedule_legend_markdown,
)

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

# Elevation ramp for the occupied-cell surface: blue -> green -> yellow ->
# vermillion over a FIXED [-3, 15] m band (z-up world), so a cell's colour does
# not change frame to frame as the visible height range moves. Okabe-Ito stops,
# an ordered light->dark progression that survives all three CVD types.
_HEIGHT_STOPS = np.array(
    [[0, 114, 178], [0, 158, 115], [240, 228, 66], [213, 94, 0]], dtype=np.float32
)


def _height_ramp(z: np.ndarray, lo: float = -3.0, hi: float = 15.0) -> np.ndarray:
    """(N, 3) uint8 colour per cell from its world-z, clipped to [lo, hi]."""
    t = np.clip((np.asarray(z, np.float32) - lo) / (hi - lo), 0.0, 1.0) * 3.0
    i = np.clip(t.astype(np.int64), 0, 2)
    f = (t - i)[:, None]
    return (_HEIGHT_STOPS[i] * (1.0 - f) + _HEIGHT_STOPS[i + 1] * f).astype(np.uint8)


# Occupancy layers are drawn as three visually distinct things -- "unknown is
# not free" is a hard invariant (CLAUDE.md, math §10.1) and the view has to keep
# them apart:
#   OCCUPIED  elevation-ramped solid boxes at the cell's height (`_log_occupied`)
#   FREE      flat translucent slate tiles at the ground datum (`_log_free`) --
#             "the sensor looked here and it is clear"
#   UNKNOWN   the blind cone, plus any cell the map still calls UNKNOWN despite
#             having been observed (`_log_unknown`); never-observed allocation
#             slots are left undrawn, they are not information
_FREE_RGBA = (110, 125, 140, 70)      # slate, ~27% opacity -- recedes behind occupied
_UNKNOWN_RGBA = (150, 90, 160, 90)    # muted violet, matches the blind-cone "unknown" hue family


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
                 palette: str = "semantickitti", engine=None):
        self.color_by = color_by
        self.ghost_removal = ghost_removal
        self.palette = palette
        self.schedule = schedule
        # The map back end (`run.engine.MapEngine`), or None for perception-only
        # runs. When present, `log_frame` draws its occupied cells as the real
        # 2.5D surface -- see `_log_occupied`.
        self.engine = engine
        self._last_occupied_n = 0   # updated by _log_occupied, read by _log_memory
        rr.init("vrgrid_pipeline", spawn=spawn)
        if save_path:
            rr.save(save_path)

        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log(
            "instructions",
            rr.TextDocument(
                "Ghost toggle: show/hide the `world/ghosts` entity (eye icon in the "
                "entity panel).\nvisible = ghost removal OFF (trails behind moving "
                "objects)\nhidden  = ghost removal ON\n\n"
                "Map occupancy (when a MapEngine is attached), math §10.1:\n"
                "  `world/map/occupied`  raised solid boxes, coloured by height\n"
                "  `world/map/free`      flat translucent slate tiles -- looked, clear\n"
                "  `world/map/unknown`   observed-but-still-unknown cells + the blind "
                "cone; never-observed cells are not drawn.\n"
                "Unknown is not free -- they are separate entities on purpose.",
                media_type=rr.MediaType.MARKDOWN,
            ),
            static=True,
        )
        rr.log("legend", rr.TextDocument(legend_markdown(palette),
                                         media_type=rr.MediaType.MARKDOWN), static=True)
        rr.log("schedules", rr.TextDocument(schedule_legend_markdown(schedule.name),
                                            media_type=rr.MediaType.MARKDOWN), static=True)
        self._log_rings(schedule)
        self._log_blind_cone(blind_cone_radius_m())

    def _log_rings(self, schedule):
        """Ring-boundary circles, straight from the passed `Schedule`. The ring
        half-widths and cell sizes come from `configs/schedule_*.yaml` via
        `grid.schedule.load` -- nothing here is hardcoded, and this draws the
        same rings the engine bins into."""
        for ring in schedule.rings:
            hw = ring.half_width_m
            th = np.linspace(0, 2 * np.pi, 129)
            strip = np.stack([hw * np.cos(th), hw * np.sin(th), np.zeros_like(th)], axis=1)
            rr.log(
                f"world/vehicle/rings/ring_{ring.ring}_{ring.cell_m * 100:g}cm",
                rr.LineStrips3D([strip.astype(np.float32)], colors=[220, 220, 220], radii=0.04),
                static=True,
            )

    def _log_blind_cone(self, radius_m: float):
        th = np.linspace(0, 2 * np.pi, 65)
        strip = np.stack([radius_m * np.cos(th), radius_m * np.sin(th), np.zeros_like(th)], axis=1)
        rr.log(
            "world/vehicle/blind_cone",
            rr.LineStrips3D([strip.astype(np.float32)], colors=[230, 60, 60], radii=0.05,
                            labels=[f"blind cone {radius_m:.2f} m (unknown, never free)"]),
            static=True,
        )

    def _cell_m_per_slot(self, slots: np.ndarray) -> np.ndarray:
        """Cell edge length (m) for each occupied slot, from the ring it lives
        in. This is what makes the foveation visible: a box drawn at a cell's
        own size grows from 5 cm near the vehicle to 40 cm at 100 m."""
        out = np.full(len(slots), self.engine.sched.base_cell_m, dtype=np.float32)
        for layout in self.engine.handle.rings:
            sel = (slots >= layout.offset) & (slots < layout.offset + layout.slots)
            out[sel] = layout.cell_m
        return out

    def _centres_world(self, slots: np.ndarray):
        """World-frame `(x, y, z)` for arbitrary slots, via the engine's own
        inverse of `flat_slot`. ego (0, 0) leaves the centres in the world
        frame, exactly as `MapEngine.occupied_cells` does it for the occupied
        set -- this just reuses it for the free / unknown sets."""
        n = len(slots)
        x, y, z = np.zeros(n), np.zeros(n), np.zeros(n)
        if n:
            self.engine._centres(slots, np.zeros(2), x, y, z)
        return x, y, z

    def _log_occupied(self):
        """The real 2.5D occupied surface: every cell the map currently calls
        OCCUPIED, drawn as a box at its world xy, at its visibility height z
        (ceiling where one was seen, ground otherwise -- the same height §10.4
        tests), sized to its ring's cell. `--show-ghosts` keeps the moving
        car's cells here; the default clears them via §10.4, so this entity is
        where the toggle actually shows on screen.

        Calling `occupied_cells()` also refreshes `engine.occ_state`, which
        `_log_free` / `_log_unknown` then read -- so this runs first."""
        slots, x, y, z = self.engine.occupied_cells()
        self._last_occupied_n = len(slots)   # for _log_memory, no second pass
        if len(slots) == 0:
            rr.log("world/map/occupied", rr.Clear(recursive=True))
            return
        cell_m = self._cell_m_per_slot(slots)
        centres = np.stack([x, y, z], axis=1).astype(np.float32)
        half = np.stack(
            [cell_m / 2.0, cell_m / 2.0, np.full_like(cell_m, 0.02)], axis=1
        ).astype(np.float32)
        rr.log(
            "world/map/occupied",
            rr.Boxes3D(centers=centres, half_sizes=half,
                       colors=_height_ramp(z), fill_mode="solid"),
        )

    def _log_free(self):
        """FREE cells -- observed and clear -- as flat translucent tiles at the
        ground datum, sized to their ring's cell so the foveation still reads.
        Distinct from OCCUPIED (which is solid and raised) and from UNKNOWN
        (undrawn / blind cone): "looked and clear" is not "did not look"."""
        free = np.flatnonzero(self.engine.occ_state == OCC_FREE)
        if len(free) == 0:
            rr.log("world/map/free", rr.Clear(recursive=True))
            return
        x, y, z = self._centres_world(free)
        cell_m = self._cell_m_per_slot(free)
        centres = np.stack([x, y, z], axis=1).astype(np.float32)
        half = np.stack(
            [cell_m / 2.0, cell_m / 2.0, np.full_like(cell_m, 0.01)], axis=1
        ).astype(np.float32)
        rr.log(
            "world/map/free",
            rr.Boxes3D(centers=centres, half_sizes=half,
                       colors=[_FREE_RGBA], fill_mode="solid"),
        )

    def _log_unknown(self):
        """UNKNOWN cells that were nonetheless OBSERVED at least once (blind-cone
        cells, cells whose evidence never cleared `n_min`) -- the planner-
        relevant unknown, drawn in the blind-cone hue. Never-observed allocation
        slots (the bulk of the grid) are deliberately not drawn: they carry no
        information, and 700k boxes would bury the ones that matter. The blind
        cone itself is always drawn as a circle under the vehicle transform."""
        obs = self.engine.handle.grid["obs_count"]
        seen_unknown = np.flatnonzero((self.engine.occ_state == OCC_UNKNOWN) & (obs > 0))
        if len(seen_unknown) == 0:
            rr.log("world/map/unknown", rr.Clear(recursive=True))
            return
        x, y, z = self._centres_world(seen_unknown)
        cell_m = self._cell_m_per_slot(seen_unknown)
        centres = np.stack([x, y, z], axis=1).astype(np.float32)
        half = np.stack(
            [cell_m / 2.0, cell_m / 2.0, np.full_like(cell_m, 0.01)], axis=1
        ).astype(np.float32)
        rr.log(
            "world/map/unknown",
            rr.Boxes3D(centers=centres, half_sizes=half,
                       colors=[_UNKNOWN_RGBA], fill_mode="solid"),
        )

    def _log_memory(self):
        """Per-frame live memory overlay: the real occupied-cell storage now
        (`occupied count * CELL_BYTES`), the dense-3D baseline derived from the
        schedule for the same covered volume, and the live ratio. Logged
        alongside the static `schedules` panel, but updated every frame."""
        rr.log(
            "memory",
            rr.TextDocument(
                memory_overlay_markdown(self._last_occupied_n, self.schedule),
                media_type=rr.MediaType.MARKDOWN,
            ),
        )

    def log_frame(self, frame):
        rr.set_time("frame", sequence=frame.index)

        xyz, colors = get_display_points(frame, self.ghost_removal, self.color_by, self.palette)
        rr.log("world/points", rr.Points3D(xyz, colors=colors, radii=0.03))

        if self.engine is not None:
            self._log_occupied()   # also refreshes engine.occ_state
            self._log_free()
            self._log_unknown()
            self._log_memory()

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
