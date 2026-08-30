"""Coordinate transforms. [JP — Day 0, hour 0-2 with the whole team]

EVERY transform in this file must also be written down in `docs/frames.md`,
with origin, axes, handedness and units. Frame confusion is the most common
silent bug in this project: the map looks entirely plausible and slowly
rotates. It costs three days if found on Day 4 and minutes if found now.

Vehicle frame: x forward, y left, z up.

Do the static-wall test before anything else — drive a sequence past a flat
wall and check the wall stays flat and stationary in the map. It catches
sensor-to-vehicle and vehicle-to-world errors in one shot.
"""

import numpy as np

# Sensor → Vehicle: identity rotation, translation [0, 0, 1.73] (sensor height)
# From calib.txt and KITTI spec: Velodyne frame aligns with vehicle frame.
T_S_V = np.eye(4, dtype=np.float64)
T_S_V[2, 3] = 1.73  # sensor height in metres


def sensor_to_vehicle() -> np.ndarray:
    """Constant transform: Sensor (Velodyne) → Vehicle frame.

    Returns:
        4×4 homogeneous matrix. Sensor axes already match vehicle axes (x forward, y left, z up).
        Only translation is sensor height (1.73 m).
    """
    return T_S_V.copy()


def vehicle_to_world(pose: np.ndarray) -> np.ndarray:
    """Convert KITTI 3×4 pose to 4×4 Vehicle → World transform.

    Args:
        pose: (3, 4) or (12,) array from poses.txt — row-major [R | t]

    Returns:
        4×4 homogeneous matrix: point_V → point_W
    """
    pose = np.asarray(pose, dtype=np.float64).reshape(3, 4)
    T = np.eye(4, dtype=np.float64)
    T[:3, :4] = pose
    return T


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Transform 3D points by 4×4 homogeneous matrix.

    Args:
        points: (N, 3) or (N, 4) array — if 4th column present, treated as homogeneous (w=1)
        T: 4×4 transform matrix

    Returns:
        (N, 3) transformed points
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[1] == 3:
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])
    elif pts.shape[1] == 4:
        pts_h = pts.copy()
        pts_h[:, 3] = 1.0
    else:
        raise ValueError(f"points must be (N, 3) or (N, 4), got {pts.shape}")

    out = (T @ pts_h.T).T
    return out[:, :3]