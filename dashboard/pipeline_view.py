"""Rerun view of the real perception pipeline. [JP]

Replaces the Day-0 synthetic plane/boxes/slope (`demo_synthetic.py`) with the
actual per-frame output of `vrgrid.run`, one interpretation layer at a time via
`color_by`:

    intensity     raw Velodyne intensity, greyscale  (the "no semantics" baseline)
    class         semantic_labels() 19-class colours
    motion        is_moving(): static dim, moving bright red
    ground        segment_ground(): ground tan, non-ground steel blue
    reflectivity  reflectivity.normalise(): rho_hat byte, greyscale

Ring boundaries and the blind cone are logged under the vehicle transform, so
they track the vehicle as it drives. Points are in the world frame and
accumulate over time (Rerun's timeline scrubs frame by frame).
"""

import numpy as np
import rerun as rr

from .demo_synthetic import class_to_color, load_schedule

_CLASS_LUT = np.array([class_to_color(c) for c in range(-1, 19)], dtype=np.uint8)  # index c+1


def _colors_intensity(frame) -> np.ndarray:
    g = np.clip(frame.points_sensor[:, 3] * 255.0 * 1.5, 0, 255).astype(np.uint8)
    return np.repeat(g[:, None], 3, axis=1)


def _colors_class(frame) -> np.ndarray:
    return _CLASS_LUT[np.clip(frame.semantic + 1, 0, 19)]


def _colors_motion(frame) -> np.ndarray:
    c = np.full((len(frame.moving), 3), 90, dtype=np.uint8)  # static: dim grey
    c[frame.moving] = (255, 40, 40)                          # moving: red
    return c


def _colors_ground(frame) -> np.ndarray:
    c = np.empty((len(frame.ground), 3), dtype=np.uint8)
    c[frame.ground] = (170, 130, 90)      # ground: tan
    c[~frame.ground] = (70, 130, 180)     # non-ground: steel blue
    return c


def _colors_reflectivity(frame) -> np.ndarray:
    v = frame.reflectivity8
    return np.repeat(v[:, None], 3, axis=1)


_COLORERS = {
    "intensity": _colors_intensity,
    "class": _colors_class,
    "motion": _colors_motion,
    "ground": _colors_ground,
    "reflectivity": _colors_reflectivity,
}


class PipelineView:
    def __init__(self, schedule, spawn: bool = False, save_path: str | None = None,
                 color_by: str = "class", schedule_yaml: str | None = None):
        self.color_by = color_by
        rr.init("vrgrid_pipeline", spawn=spawn)
        if save_path:
            rr.save(save_path)

        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
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

        colors = _COLORERS[self.color_by](frame)
        rr.log("world/points", rr.Points3D(frame.points_world.astype(np.float32),
                                           colors=colors, radii=0.03))

        # vehicle transform: origin + heading from the GT pose (world-frame yaw)
        fwd_world = (frame.pose[:3, :3] @ np.array([0.0, 0.0, 1.0]))  # camera z = forward
        yaw = np.arctan2(-fwd_world[0], fwd_world[2])  # into the z-up world convention
        rr.log("world/vehicle", rr.Transform3D(
            translation=frame.vehicle_xyz_world.astype(np.float32),
            rotation=rr.RotationAxisAngle(axis=[0, 0, 1], angle=float(yaw)),
        ))
        rr.log("world/vehicle/marker", rr.Points3D([[0, 0, 0]], colors=[40, 220, 40], radii=0.4))
