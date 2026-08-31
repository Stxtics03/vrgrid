"""Coordinate transforms. [JP — Day 0, hour 0-2 with the whole team]

EVERY transform in this file is also written down in `docs/frames.md`, with
origin, axes, handedness and units. Frame confusion is the most common silent
bug in this project: the map looks entirely plausible and slowly rotates. It
costs three days if found on Day 4 and minutes if found now.

Three frames
------------
  Sensor  (Velodyne HDL-64E):   x forward, y left, z up.  Origin at the laser,
                                ~1.73 m above the ground.
  Vehicle:                      x forward, y left, z up.  Same axes as the
                                sensor; origin dropped to the ground plane
                                directly below the laser (z = 0 at the road).
  World:                        x forward, y left, z up.  Coincides with the
                                Vehicle frame of the FIRST frame of the
                                sequence and never moves after that.

KITTI odometry gives us two things and neither is Velodyne -> Vehicle directly:

  * `calib.txt` line `Tr:`  ->  Velodyne -> Camera-0    (4x4, measured extrinsic)
  * `poses.txt`  line i     ->  Camera-0 -> World_cam   (3x4, GT trajectory,
                                row-major [R | t], frame 0 is identity)

`World_cam` is in the camera convention (x right, y down, z forward). We rotate
it once, with a constant axis permutation, into the z-up World frame above so
that every downstream consumer (grid, reference map, dashboard) sees x-forward,
y-left, z-up.

The static-wall test (`tests/test_static_wall.py`, Gate Item 1) is the check
that this file is right: drive 100 frames past a building facade, transform
every scan into World, and confirm the wall's fitted plane does not rotate or
translate.
"""

import numpy as np

# --- Sensor -> Vehicle -------------------------------------------------------
#
# KITTI's odometry benchmark publishes no Velodyne -> Vehicle extrinsic. By the
# universal KITTI convention the Velodyne axes are already aligned with the
# vehicle axes (x forward, y left, z up), so the rotation is identity and the
# only difference is height: the laser sits 1.73 m above the road (KITTI spec
# sheet / HDL-64E mounting documentation, used throughout the literature).
#
# We put the Vehicle origin on the ground, so a point at the sensor origin is at
# vehicle-frame (0, 0, 1.73): the translation is +1.73 m in z.
SENSOR_HEIGHT_M = 1.73

T_S_V = np.eye(4, dtype=np.float64)
T_S_V[2, 3] = SENSOR_HEIGHT_M


def sensor_to_vehicle() -> np.ndarray:
    """Constant transform: Sensor (Velodyne) -> Vehicle frame.

    Identity rotation (KITTI convention: Velodyne axes align with the vehicle)
    and +1.73 m in z so the Vehicle origin sits on the road surface.

    Returns:
        (4, 4) float64 homogeneous matrix.
    """
    return T_S_V.copy()


# --- Camera-0 -> Vehicle-convention rotation --------------------------------
#
# Camera-0 frame:  x right, y down, z forward.
# Vehicle frame:   x forward, y left, z up.
#
#   vehicle_x =  camera_z
#   vehicle_y = -camera_x
#   vehicle_z = -camera_y
R_CAM0_TO_VEH = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)

# Vehicle -> Velodyne: inverse of sensor_to_vehicle (drop the ground origin
# back up to the laser). Used inside vehicle_to_world so the composed transform
# can be applied to points already lifted into the Vehicle frame.
_T_V_S = np.eye(4, dtype=np.float64)
_T_V_S[2, 3] = -SENSOR_HEIGHT_M


_TR_CACHE: dict[str, np.ndarray] = {}


def velo_to_cam0(sequence: str = "00") -> np.ndarray:
    """Velodyne -> Camera-0 (the `Tr:` line of `sequences/<seq>/calib.txt`).

    Cached per sequence. The matrix is near-identical across sequences 00/07/08
    (same rig), but we read the real one so a re-calibrated sequence is handled.

    Returns:
        (4, 4) float64 homogeneous matrix.
    """
    if sequence not in _TR_CACHE:
        from .loader import load_calib

        _TR_CACHE[sequence] = load_calib(sequence)["Tr_velo_to_cam0"]
    return _TR_CACHE[sequence].copy()


def _pose_to_4x4(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(3, 4)
    T = np.eye(4, dtype=np.float64)
    T[:3, :4] = pose
    return T


def vehicle_to_world(pose: np.ndarray, sequence: str = "00") -> np.ndarray:
    """4x4 Vehicle -> World transform for one frame.

    Composition, applied right to left to a Vehicle-frame point:

        Vehicle -> Velodyne          (_T_V_S, undo the 1.73 m ground drop)
        Velodyne -> Camera-0         (Tr, from calib.txt)
        Camera-0 -> World_cam        (pose, from poses.txt)
        World_cam -> World (z-up)    (R_CAM0_TO_VEH, constant axis permutation)

    Args:
        pose: (3, 4) or (12,) row-major [R | t] from `poses.txt`, Camera-0 -> World_cam.
        sequence: which calib.txt to read `Tr` from. Default "00".

    Returns:
        (4, 4) float64 homogeneous matrix: point_Vehicle -> point_World.
    """
    T_pose = _pose_to_4x4(pose)
    T_tr = velo_to_cam0(sequence)

    R_flip = np.eye(4, dtype=np.float64)
    R_flip[:3, :3] = R_CAM0_TO_VEH

    return R_flip @ T_pose @ T_tr @ _T_V_S


def sensor_to_world(pose: np.ndarray, sequence: str = "00") -> np.ndarray:
    """4x4 Sensor -> World transform for one frame (convenience).

    Equivalent to ``vehicle_to_world(pose) @ sensor_to_vehicle()`` and, because
    the 1.73 m ground drop cancels, to the textbook KITTI chain
    ``R_flip @ pose @ Tr`` acting on raw Velodyne points.
    """
    return vehicle_to_world(pose, sequence) @ T_S_V


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to 3D points.

    Args:
        points: (N, 3) or (N, 4) array. A 4th column (intensity) is ignored and
            not returned.
        T: (4, 4) transform matrix.

    Returns:
        (N, 3) float64 transformed points.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[1] not in (3, 4):
        raise ValueError(f"points must be (N, 3) or (N, 4), got {pts.shape}")

    xyz = pts[:, :3]
    pts_h = np.hstack([xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)])
    return (T @ pts_h.T).T[:, :3]
