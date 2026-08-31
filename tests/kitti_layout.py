"""A synthetic sequence in the layout `perception.loader` actually reads.
[Shrestha]

`eval/synthetic.py` already writes a sequence, and its docstring promises that
swapping `read_sequence` for `perception.loader` is the only edit the real-data
path needs. It is not, today. The two disagree about where two files live:

    loader.py wants                       synthetic.write_sequence writes
      poses/<seq>.txt                       sequences/<seq>/poses.txt
      sequences/<seq>/calib.txt             (absent)
      sequences/<seq>/velodyne/*.bin        sequences/<seq>/velodyne/*.bin
      sequences/<seq>/labels/*.label        sequences/<seq>/labels/*.label

And the pose mismatch is deliberate on the loader's side -- its header says it
uses the OFFICIAL KITTI poses from `poses/<seq>.txt`, "not the SemanticKITTI
internal SLAM poses at `sequences/<seq>/poses.txt`". So the file the synthetic
writer produces is precisely the one the loader is built to ignore. Discovered
by reading, not by running: with no data on disk, every test of that path skips,
so nothing has ever exercised it.

This writes the layout the loader wants, so the whole real path --
`loader.scans` -> `transforms` -> `range_image` -> `semantics` -> `MapEngine` --
can be run before the download lands. The only variable left is the KITTI bytes.

**The poses are derived from the real transform chain, not hand-written.**
`vehicle_to_world` composes `R_flip @ pose @ Tr @ T_V_S`, so this inverts that
composition to produce the `pose` matrix that puts the vehicle where we want it.
A hand-written pose would be a second opinion about the frame convention, and
the whole point of `docs/frames.md` is that there is only one.
"""

from pathlib import Path

import numpy as np
from vrgrid.perception.transforms import (
    _T_V_S,
    R_CAM0_TO_VEH,
    T_S_V,
)

# Raw SemanticKITTI ids, so `semantics.semantic_labels` maps them for real.
# ROAD and SIDEWALK are the ground; BUILDING and VEGETATION are the structure.
#
# ⚑ BUILDING maps to class 12 and VEGETATION to 14, and a realistic scene also
#   carries pole (80 -> 17) and traffic-sign (81 -> 18). Anything above 15 does
#   not fit fusion's 4-bit candidate (math §10.2), so a faithful fixture hits
#   the same wall the first real frame will. That is deliberate: a fixture that
#   quietly stayed under 16 would let the pipeline pass here and fail on KITTI.
RAW_ROAD, RAW_SIDEWALK, RAW_BUILDING, RAW_VEGETATION = 40, 48, 50, 70
RAW_POLE, RAW_MOVING_CAR = 80, 252

SENSOR_HEIGHT_M = 1.73


def _tr_velo_to_cam0() -> np.ndarray:
    """Velodyne -> Camera-0, as the inverse of the axis permutation.

    KITTI's real `Tr` also carries a few centimetres of translation and a
    fraction of a degree of rotation. Using the exact inverse instead keeps the
    fixture's world coordinates equal to the ones it was asked for, so a test
    can assert where a wall ended up rather than only that nothing crashed. The
    chain being exercised is identical either way -- it is the same four
    matrices multiplied in the same order.
    """
    tr = np.eye(4, dtype=np.float64)
    tr[:3, :3] = R_CAM0_TO_VEH.T          # cam0 <- world_cam is R; invert it
    return tr


def pose_for(vehicle_xyz_world, sequence_tr=None) -> np.ndarray:
    """The `poses/<seq>.txt` row that puts the vehicle at `vehicle_xyz_world`.

    Inverts `vehicle_to_world`'s composition:

        T_VW = R_flip @ T_pose @ Tr @ T_V_S
        =>  T_pose = R_flip^-1 @ T_VW @ T_V_S^-1 @ Tr^-1
    """
    tr = _tr_velo_to_cam0() if sequence_tr is None else sequence_tr
    r_flip = np.eye(4, dtype=np.float64)
    r_flip[:3, :3] = R_CAM0_TO_VEH

    t_vw = np.eye(4, dtype=np.float64)
    t_vw[:3, 3] = np.asarray(vehicle_xyz_world, dtype=np.float64)

    t_pose = np.linalg.inv(r_flip) @ t_vw @ np.linalg.inv(_T_V_S) @ np.linalg.inv(tr)
    return t_pose[:3, :4]


PHI_MIN_DEG, PHI_MAX_DEG = -24.8, 2.0     # HDL-64E vertical FOV, configs/frnet.yaml


def _in_fov(xyz):
    """Drop returns the sensor could not physically have made.

    A real HDL-64E fires 64 beams between -24.8 and +2 degrees; nothing outside
    that comes back. The first version of this fixture put a 4 m building wall
    2 m from the sensor, which is 63 degrees up, and `range_image.project`
    warned that 26.1% of the sweep was being clamped to an edge ring. That
    warning was right and the fixture was wrong -- and a fixture that trips a
    real warning teaches everyone to ignore it.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    horiz = np.hypot(xyz[:, 0], xyz[:, 1])
    phi = np.degrees(np.arctan2(xyz[:, 2], np.maximum(horiz, 1e-9)))
    return (phi >= PHI_MIN_DEG) & (phi <= PHI_MAX_DEG) & (horiz > 1.0)


def street_scan(rng, n_ground=6000, n_wall=9000, n_pole=1200, car_x=None):
    """One sweep in the SENSOR frame, with raw SemanticKITTI label words.

    Geometry only has to be plausible; what matters is that every downstream
    stage receives the shapes, dtypes and FOV it will receive from KITTI.
    """
    parts, labels = [], []

    r = rng.uniform(3.0, 40.0, n_ground)
    a = rng.uniform(-np.pi, np.pi, n_ground)
    parts.append(np.column_stack([r * np.cos(a), r * np.sin(a),
                                  np.full(n_ground, -SENSOR_HEIGHT_M)]))
    labels.append(np.where(np.abs(r * np.sin(a)) > 4.0, RAW_SIDEWALK, RAW_ROAD))

    y = np.where(rng.random(n_wall) < 0.5, -7.0, 7.0)
    parts.append(np.column_stack([rng.uniform(6.0, 45.0, n_wall), y,
                                  rng.uniform(-SENSOR_HEIGHT_M, 1.5, n_wall)]))
    labels.append(np.full(n_wall, RAW_BUILDING))

    parts.append(np.column_stack([rng.uniform(12.5, 13.5, n_pole),
                                  rng.uniform(4.4, 4.6, n_pole),
                                  rng.uniform(-SENSOR_HEIGHT_M, 0.4, n_pole)]))
    labels.append(np.full(n_pole, RAW_POLE))

    if car_x is not None:
        n_car = 1200
        parts.append(np.column_stack([rng.uniform(car_x - 1.0, car_x + 1.0, n_car),
                                      rng.uniform(-1.0, 1.0, n_car),
                                      rng.uniform(-1.5, -0.2, n_car)]))
        labels.append(np.full(n_car, RAW_MOVING_CAR))

    xyz = np.vstack(parts)
    lab = np.concatenate(labels)
    keep = _in_fov(xyz)
    xyz, lab = xyz[keep], lab[keep]
    intensity = rng.uniform(0.0, 1.0, len(xyz))
    points = np.column_stack([xyz, intensity]).astype(np.float32)
    return points, lab.astype(np.uint32)


def write_sequence(root, sequence: str = "99", n_frames: int = 6,
                   step_m: float = 2.0, seed: int = 0, moving_car: bool = True,
                   car_step_m: float = 4.0):
    """Write `n_frames` in the layout `perception.loader` reads.

    The vehicle drives along +x in the world at `step_m` per frame, and the
    scene is regenerated in the SENSOR frame each frame -- which is what a real
    sensor gives you, and what makes the poses do any work.

    ⚑ The car has to move in the WORLD, not just in the sensor frame. The first
      version put it at sensor-frame `14 - i * step_m`, which exactly cancels
      the vehicle's own motion: a car labelled `moving-car` that sat at world
      x 13-15 m for the whole sequence. It was a parked car wearing a moving
      label, so ghost removal had nothing to remove and
      `ghost_removal_figure.py --seq` cleared 1.0% of it -- correct behaviour,
      measured against a fixture that lied. `car_step_m` is the car's world
      speed; it pulls away at `car_step_m - step_m` in the sensor frame.
    """
    root = Path(root)
    seq_dir = root / "sequences" / sequence
    (seq_dir / "velodyne").mkdir(parents=True, exist_ok=True)
    (seq_dir / "labels").mkdir(parents=True, exist_ok=True)
    (root / "poses").mkdir(parents=True, exist_ok=True)

    tr = _tr_velo_to_cam0()
    lines = " ".join(f"{v:.12e}" for v in tr[:3, :4].reshape(-1))
    (seq_dir / "calib.txt").write_text(
        "P0: " + " ".join(["0.0"] * 12) + "\n"
        f"Tr: {lines}\n")

    rng = np.random.default_rng(seed)
    poses = []
    for i in range(n_frames):
        # sensor-frame x, so that world x = 14 + i * car_step_m
        car_x = (14.0 + i * (car_step_m - step_m)) if moving_car else None
        points, labels = street_scan(rng, car_x=car_x)
        points.tofile(seq_dir / "velodyne" / f"{i:06d}.bin")
        labels.tofile(seq_dir / "labels" / f"{i:06d}.label")
        poses.append(pose_for((i * step_m, 0.0, 0.0), tr).reshape(-1))

    np.savetxt(root / "poses" / f"{sequence}.txt", np.array(poses), fmt="%.12e")
    return root


def point_in_world(points_sensor, pose, tr=None):
    """The chain the pipeline uses, for a test to check the fixture against."""
    r_flip = np.eye(4, dtype=np.float64)
    r_flip[:3, :3] = R_CAM0_TO_VEH
    t_pose = np.eye(4, dtype=np.float64)
    t_pose[:3, :4] = np.asarray(pose, dtype=np.float64).reshape(3, 4)
    t = r_flip @ t_pose @ (_tr_velo_to_cam0() if tr is None else tr) @ _T_V_S @ T_S_V
    xyz = np.asarray(points_sensor, dtype=np.float64)[:, :3]
    h = np.hstack([xyz, np.ones((len(xyz), 1))])
    return (t @ h.T).T[:, :3]
