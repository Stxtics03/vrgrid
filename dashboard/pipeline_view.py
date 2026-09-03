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
from vrgrid.grid.confidence import drivable_confidence
from vrgrid.grid.features import detect

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

# --- §7.4 features and §7.5 confidence -------------------------------------
#
# Three more read-only layers over fields `MapEngine.step` already fills. They
# are OFF by default and OFF the per-frame path, and that is a measurement
# rather than caution: `features.detect` is a neighbourhood pass over all
# 910,000 window slots and costs **1,137 ms p50** on this machine, eleven times
# the whole 100 ms frame budget (`drivable_confidence` over four rings is a
# further 89 ms). Logging them every frame would make the viewer unusable and
# would quietly triple the cost of every `--save` recording, so `--features`
# recomputes them every `FEATURE_INTERVAL` frames and once more at the end.
# What is drawn is therefore the map as of the last recompute, which is why
# the final call after the loop exists: the last frame is the one a still gets
# taken from.
FEATURE_INTERVAL = 20

_CURB_RGBA = (230, 159, 0, 235)       # Okabe-Ito orange -- a positive step, drawn standing up
_POTHOLE_RGBA = (213, 94, 0, 245)     # Okabe-Ito vermillion -- a negative one, drawn sunken

# Confidence is a scalar field, so it needs a ramp rather than a flat colour,
# and the ramp is ordered in LIGHTNESS (dark = no confidence -> light = full)
# so it survives all three CVD types without relying on hue. Deliberately not
# red-to-green, which is the one ramp a deuteranope cannot read at all.
_CONFIDENCE_STOPS = np.array(
    [[38, 54, 92], [0, 114, 178], [86, 180, 233], [240, 228, 66]], dtype=np.float32
)


def _confidence_ramp(c: np.ndarray) -> np.ndarray:
    """(N, 3) uint8 from a drivable-confidence in [0, 1]. §7.5."""
    t = np.clip(np.asarray(c, np.float32), 0.0, 1.0) * 3.0
    i = np.clip(t.astype(np.int64), 0, 2)
    f = (t - i)[:, None]
    return (_CONFIDENCE_STOPS[i] * (1.0 - f) + _CONFIDENCE_STOPS[i + 1] * f).astype(np.uint8)


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
                 palette: str = "semantickitti", engine=None, features: bool = False):
        self.color_by = color_by
        self.ghost_removal = ghost_removal
        self.palette = palette
        self.schedule = schedule
        # The map back end (`run.engine.MapEngine`), or None for perception-only
        # runs. When present, `log_frame` draws its occupied cells as the real
        # 2.5D surface -- see `_log_occupied`.
        self.engine = engine
        self._last_occupied_n = 0   # updated by _log_occupied, read by _log_memory
        # §7.4 curbs / potholes and §7.5 confidence. Off by default because
        # `features.detect` costs 1,137 ms -- see FEATURE_INTERVAL.
        self.features = bool(features) and engine is not None
        self._frames_logged = 0
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
                "Unknown is not free -- they are separate entities on purpose.\n\n"
                "With `--features` (math §7.4, §7.5):\n"
                "  `world/map/curbs`      orange boxes standing at the measured rise\n"
                "  `world/map/potholes`   vermillion boxes sunk to the measured depth\n"
                "  `world/map/confidence` flat tiles above the surface, dark = no "
                "confidence in drivability, light = full.\n"
                "These recompute every 20 frames, not every frame: the detector is a "
                "full-window pass and costs ~1.1 s.\n\n"
                "⚑ KNOWN BUG, ring 3 confidence: every ring-3 tile reads a FALSE "
                "0.000. Beyond ~50 m SemanticKITTI has no labels (100% of returns "
                "in the 50-100 m band on seq 00), and `run/engine.py` maps "
                "unlabelled to class 0 = `car`, which is not drivable. So ring 3 "
                "is dark because it is UNLABELLED, not because it is hazardous. "
                "Live-map path only -- the published per-ring / rho tables use "
                "`eval/harness.py`, which maps unlabelled to CLASS_UNLABELLED and "
                "is correct.",
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

    def _ring_slices(self):
        """`[(slice, side), ...]` per ring -- the shape `features.detect` and
        `confidence.summarise` both take. Built from the engine's allocation so
        it cannot drift from the storage layout."""
        return [(slice(r.offset, r.offset + r.slots), r.side)
                for r in self.engine.handle.rings]

    def _flat(self, level: int, slot: np.ndarray) -> np.ndarray:
        """`Curbs.slot` / `Potholes.slot` are indices WITHIN a ring window --
        the dataclasses say so, and merging rings without this offset silently
        aliases ring 0's slot 5 onto ring 3's. Lift them to flat SoA slots."""
        return slot.astype(np.int64) + self.engine.handle.rings[level].offset

    def _log_curbs(self, curbs):
        """§7.4 curb edges, drawn STANDING UP at their measured rise.

        A curb is a step, so the box is drawn from the cell's surface up by
        `height_cm` rather than as a flat marker: the thing the problem
        statement says a 2D grid loses is exactly this height, and drawing it
        at its real magnitude is the difference between a detection and a
        measurement. Colour is flat -- height is already carried by the shape.
        """
        cent, half = [], []
        for level, c in enumerate(curbs):
            if not len(c):
                continue
            x, y, z = self._centres_world(self._flat(level, c.slot))
            h = np.maximum(c.height_cm.astype(np.float32), 1.0) / 100.0  # cm -> m
            cent.append(np.stack([x, y, z + h / 2.0], axis=1))
            half.append(np.stack([np.full_like(h, c.cell_m / 2.0),
                                  np.full_like(h, c.cell_m / 2.0), h / 2.0], axis=1))
        if not cent:
            rr.log("world/map/curbs", rr.Clear(recursive=True))
            return
        rr.log("world/map/curbs",
               rr.Boxes3D(centers=np.concatenate(cent).astype(np.float32),
                          half_sizes=np.concatenate(half).astype(np.float32),
                          colors=[_CURB_RGBA], fill_mode="solid"))

    def _log_potholes(self, holes):
        """§7.4 potholes, drawn SUNK to their measured depth below the rim.

        The mirror of `_log_curbs` and for the same reason: a negative obstacle
        is the other half of the sentence a 2D grid cannot answer, and a marker
        floating at the surface would show the detection while hiding the one
        number that says whether it matters.
        """
        cent, half = [], []
        for level, h in enumerate(holes):
            if not len(h):
                continue
            x, y, z = self._centres_world(self._flat(level, h.slot))
            d = np.maximum(h.depth_cm.astype(np.float32), 1.0) / 100.0   # cm -> m
            cent.append(np.stack([x, y, z - d / 2.0], axis=1))
            half.append(np.stack([np.full_like(d, h.cell_m / 2.0),
                                  np.full_like(d, h.cell_m / 2.0), d / 2.0], axis=1))
        if not cent:
            rr.log("world/map/potholes", rr.Clear(recursive=True))
            return
        rr.log("world/map/potholes",
               rr.Boxes3D(centers=np.concatenate(cent).astype(np.float32),
                          half_sizes=np.concatenate(half).astype(np.float32),
                          colors=[_POTHOLE_RGBA], fill_mode="solid"))

    def _log_confidence(self):
        """§7.5 per-cell confidence in the DRIVABILITY verdict, over observed
        cells, as flat tiles floating 15 cm above the surface.

        Floated deliberately: this is a scalar field over the same cells
        `_log_free` and `_log_occupied` already draw, and at the surface it
        would z-fight with both. Read it WITH the occupancy layers, never
        instead of them -- `drivable_confidence`'s own docstring makes the
        point that a cell can be traversable under §7.1 and still carry 0.1,
        which is the case the bitfield alone cannot express.

        ⚑ RING 3 CURRENTLY READS A FALSE 0.000, AND IT IS NOT THIS LAYER.
          Every ring-3 tile reports zero confidence with `binding` =
          "not-drivable", and the honest reason is a bug upstream, not a
          vegetation verge:

            * SemanticKITTI's annotation stops at roughly 50 m. Measured on
              seq 00 frames 0-59, **100.0%** of returns in the 50-100 m band
              (104,758 of 104,758) carry no label -- `semantic_labels` gives
              them -1. Ring 2 is 8.3% unlabelled, rings 0-1 under 1%.
            * `run/engine.py` maps `semantic < 0` to class **0**, and learning
              id 0 is `car`. So every ring-3 cell stores "car", `car` is not in
              `drivable_classes`, and the class gate zeroes the cell outright.
            * `eval/harness.py`'s `learning_ids()` does the same conversion
              CORRECTLY -- `-1 -> CLASS_UNLABELLED (31)`, which is in no
              drivable set and so fails safe as *unknown* rather than as a
              parked car. The two paths disagree about one conversion.

          Scope: this is the LIVE MapEngine path only (this dashboard,
          `vrgrid.run`, `timing_table`, `ghost_removal_figure`). The published
          §2b per-ring and rho tables go through `harness.run_sequence` and are
          NOT affected.

          Magnitude, so nobody over-reads the fix: bypassing the class gate,
          ring 3's worst-of-four margin is mean **0.008**, median 0.000, above
          zero on 3.6% of tiles -- `surface` is exactly zero on 91.8% of them
          and `geometry` on 60.5%. Corrected, ring 3 stays dark. What changes
          is that it would be dark for a true reason and `binding` would say
          `surface` / `geometry` instead of falsely saying "not-drivable" --
          low confidence honestly labelled unknown, rather than a phantom car.

          `run/engine.py` is not this lane, so this is documented here and
          flagged rather than fixed. A dark ring-3 tile in a recording made
          before that fix means "unlabelled beyond 50 m", not "hazard".
        """
        cent, cols, half = [], [], []
        for level, (sl, side) in enumerate(self._ring_slices()):
            cell_m = self.engine.sched.rings[level].cell_m
            conf = drivable_confidence(self.engine.handle.grid, sl, side, cell_m,
                                       self.engine.thresholds)
            seen = np.flatnonzero(self.engine.handle.grid["obs_count"][sl] >= 1)
            if not seen.size:
                continue
            x, y, z = self._centres_world(seen + self.engine.handle.rings[level].offset)
            cent.append(np.stack([x, y, z + 0.15], axis=1))
            cols.append(_confidence_ramp(conf[seen]))
            half.append(np.stack([np.full(seen.size, cell_m / 2.0),
                                  np.full(seen.size, cell_m / 2.0),
                                  np.full(seen.size, 0.01)], axis=1))
        if not cent:
            rr.log("world/map/confidence", rr.Clear(recursive=True))
            return
        rr.log("world/map/confidence",
               rr.Boxes3D(centers=np.concatenate(cent).astype(np.float32),
                          half_sizes=np.concatenate(half).astype(np.float32),
                          colors=np.concatenate(cols), fill_mode="solid"))

    def log_features(self):
        """Recompute and draw the §7.4 / §7.5 layers. ~1.2 s -- see
        FEATURE_INTERVAL. Public so a caller can force one final pass after the
        loop, which is the state a still gets taken from."""
        if not self.features:
            return
        rings = self._ring_slices()
        curbs, holes = detect(self.engine.handle.grid, self.engine.sched, rings,
                              self.engine.thresholds, buffers=self.engine.buffers)
        self._log_curbs(curbs)
        self._log_potholes(holes)
        self._log_confidence()

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
            # Not every frame: `features.detect` is 1,137 ms. The caller runs
            # one more pass after the loop so the final state is complete.
            if self.features and self._frames_logged % FEATURE_INTERVAL == 0:
                self.log_features()
        self._frames_logged += 1

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
