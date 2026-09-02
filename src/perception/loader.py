"""SemanticKITTI loader. [JP — Day 0, first task]

Sequences 00, 07, 08 only — about 40 GB, not the full 200 GB. Start the
download before anything else on Day 0; it is the one item on the critical
path that neither cleverness nor effort can accelerate.

Motion labels come straight out of the raw `.label` files (`moving-*`,
IDs 250-259). Nothing is retrained. Disclose it plainly in the report: motion
labels are ground truth, so the mapping contribution is evaluated independently
of segmentation quality. That is a feature — it isolates the contribution from
segmentation error.

Data root is set by the `VRGRID_DATA_ROOT` environment variable and must hold
the KITTI odometry layout:

    $VRGRID_DATA_ROOT/poses/<seq>.txt                    official GT poses
    $VRGRID_DATA_ROOT/sequences/<seq>/velodyne/*.bin
    $VRGRID_DATA_ROOT/sequences/<seq>/labels/*.label     SemanticKITTI labels
    $VRGRID_DATA_ROOT/sequences/<seq>/calib.txt

We use the OFFICIAL KITTI GT poses from `poses/<seq>.txt`, not the SemanticKITTI
internal SLAM poses at `sequences/<seq>/poses.txt`. If the variable is unset it
defaults to `./data` (the gitignored data dir at the repo root); set it
explicitly if your download lives elsewhere -- see the README.
"""

import os
import warnings
from pathlib import Path

import numpy as np

# Data root: $VRGRID_DATA_ROOT, or ./data relative to the working directory.
# Resolved once at import; nothing here touches the filesystem, so an unset
# variable is not an error until a scan is actually requested (callers get a
# FileNotFoundError naming the missing path).
DATA_ROOT = Path(os.environ.get("VRGRID_DATA_ROOT", "data")).expanduser()

# Official KITTI ground-truth poses (top-level)
GT_POSES_DIR = DATA_ROOT / "poses"

# SemanticKITTI velodyne scans and labels
VELODYNE_DIR = DATA_ROOT / "sequences"
LABELS_DIR = DATA_ROOT / "sequences"

MOVING_LABEL_IDS = range(250, 260)  # verify against raw files — Hriday, hour 4


def _velodyne_path(sequence: str, frame: int) -> Path:
    return VELODYNE_DIR / sequence / "velodyne" / f"{frame:06d}.bin"


def _label_path(sequence: str, frame: int) -> Path:
    # SemanticKITTI labels (if downloaded)
    return LABELS_DIR / sequence / "labels" / f"{frame:06d}.label"


def _gt_poses_path(sequence: str) -> Path:
    """Official KITTI ground-truth poses for sequence."""
    return GT_POSES_DIR / f"{sequence}.txt"


def _calib_path(sequence: str) -> Path:
    """Calibration file for sequence (contains Tr_velo_to_cam0)."""
    return VELODYNE_DIR / sequence / "calib.txt"


def read_calib(path) -> dict:
    """Parse one `calib.txt`. Same as `load_calib`, addressed by path.

    `load_calib` resolves the path from the module-level DATA_ROOT, which is
    fixed at import. Anything holding a sequence somewhere else -- a test
    fixture, `eval/synthetic.py`'s writer -- needs the parser without the
    lookup, and the alternative is a second parser.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Calib not found: {path}")

    calib = {}
    with open(path) as f:
        for line in f:
            if line.startswith('Tr:'):
                vals = list(map(float, line.split()[1:]))
                Tr = np.eye(4, dtype=np.float64)
                Tr[:3, :4] = np.array(vals, dtype=np.float64).reshape(3, 4)
                calib['Tr_velo_to_cam0'] = Tr
                break
    if 'Tr_velo_to_cam0' not in calib:
        raise ValueError(f"Tr not found in calib.txt: {path}")
    return calib


def load_calib(sequence: str) -> dict:
    """Load calibration from `sequences/<seq>/calib.txt`.

    Returns dict with 'Tr_velo_to_cam0' as 4x4 matrix (Velodyne → Camera-0).
    By KITTI convention, Velodyne frame = Vehicle frame.
    """
    return read_calib(_calib_path(sequence))


# --- which pose file a sequence is read with ---------------------------------
#
# Two exist per sequence and they are not interchangeable:
#
#   poses/<seq>.txt              official KITTI ground truth, a GPS/IMU
#                                solution optimised for TRAJECTORY evaluation
#   sequences/<seq>/poses.txt    SemanticKITTI's own SLAM poses, optimised so
#                                scans REGISTER into a consistent map
#
# README.md:21 chose the official ones on Day 0, which is right for most
# sequences and wrong for 08. Measured as the median absolute ground-height
# disagreement between consecutive frames, in 20 cm cells both frames saw:
#
#       seq        GT poses     SLAM poses
#        07          0.49 cm       0.66 cm
#        08         16.63 cm       1.04 cm     <-- 16x better
#
# 08's GT poses put the same patch of road 16.6 cm apart from one frame to the
# next, consistently -- 16.1 to 17.6 cm across every pair, so a systematic
# offset rather than drift. A cell seen over N frames accumulates about
# N x 16.6 cm, which made M* itself carry a 64.5 cm median standard deviation
# INSIDE a 10 cm footprint and put seq 08's per-ring RMSE at 162 cm.
#
# ⚑ THE PER-FRAME MEASURE ABOVE IS A WEAK PREDICTOR, and choosing this list
#   from it alone was wrong. It catches 08 because 08 is catastrophic, and it
#   misses everything subtler: seq 00 disagrees by only 2.27 cm per frame yet
#   ACCUMULATES a mean height bias of -13.95 cm by ring 3, while seq 03 at a
#   comparable 1.97 cm/frame accumulates -0.80. What matters is the
#   accumulated bias, measured per ring against M*:
#
#       seq   per-frame   mean_b r1   mean_b r2   mean_b r3
#        00      2.27cm       -2.86       -9.85      -13.95   <-- SLAM
#        06      1.32cm       -0.34       -3.32       -5.80
#        03      1.97cm       -1.91       +2.45       -0.80
#        others  1.0-1.4cm    < |0.8|     < |2.3|     < |2.9|
#
#   Switching 00 to SLAM takes ring 2's bias from -9.85 to -0.44 cm and ring
#   3's from -13.95 to -1.18, with ring 3 RMSE 26.03 -> 13.57. Sequence 06,
#   the next worst, was tested the same way and is a WASH -- GT better at ring
#   1, SLAM marginally better at rings 2-3 -- so it stays on GT. Only the two
#   sequences with a measured win are overridden.
#
# So 08 defaults to SLAM and everything else to GT: per sequence, measured, and
# overridable rather than assumed. `VRGRID_POSE_SOURCE` forces one globally,
# which is how you reproduce the table above.
POSE_SOURCE_BY_SEQUENCE = {"00": "slam", "08": "slam"}
POSE_SOURCE_DEFAULT = "gt"


def pose_source(sequence: str) -> str:
    """"gt" or "slam" for this sequence. `VRGRID_POSE_SOURCE` overrides."""
    forced = os.environ.get("VRGRID_POSE_SOURCE")
    if forced:
        if forced not in ("gt", "slam"):
            raise ValueError(
                f"VRGRID_POSE_SOURCE must be 'gt' or 'slam', not {forced!r}")
        return forced
    return POSE_SOURCE_BY_SEQUENCE.get(sequence, POSE_SOURCE_DEFAULT)


def load_slam_poses(sequence: str) -> np.ndarray:
    """SemanticKITTI's SLAM poses, at `sequences/<seq>/poses.txt`.

    Same convention as the GT file -- Camera-0 -> World_cam, 3x4 row-major --
    so it goes through exactly the same `transforms` composition. Only the
    numbers differ, and on 08 they differ by enough to decide whether the
    sequence is usable.
    """
    path = VELODYNE_DIR / sequence / "poses.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"SemanticKITTI SLAM poses not found: {path}. They ship with the "
            "label archive; extract it over the same root.")
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, 12)
    return data.reshape(-1, 3, 4)


def load_gt_poses(sequence: str) -> np.ndarray:
    """Load official KITTI ground-truth poses for a sequence.

    Args:
        sequence: "00", "07", or "08"

    Returns:
        (N, 3, 4) array — N frames, each a 3×4 pose [R | t] in row-major order
    """
    path = _gt_poses_path(sequence)
    if not path.exists():
        raise FileNotFoundError(f"GT poses not found: {path}")

    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, 12)
    return data.reshape(-1, 3, 4)


def load_velodyne_scan(path: Path) -> np.ndarray:
    """Load .bin velodyne scan as (N, 4) float32 — x, y, z, intensity."""
    if not path.exists():
        raise FileNotFoundError(f"Velodyne scan not found: {path}")
    data = np.fromfile(path, dtype=np.float32)
    return data.reshape(-1, 4)


def load_labels(path: Path) -> np.ndarray:
    """Load .label file as (N,) uint32 — SemanticKITTI raw labels."""
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")
    data = np.fromfile(path, dtype=np.uint32)
    return data


def _available_frames(sequence: str) -> list[int]:
    """Return sorted list of available frame indices from velodyne directory."""
    velo_dir = VELODYNE_DIR / sequence / "velodyne"
    if not velo_dir.exists():
        return []
    frames = []
    for f in velo_dir.glob("*.bin"):
        try:
            frames.append(int(f.stem))
        except ValueError:
            pass
    return sorted(frames)


def scans(sequence: str, max_frames: int | None = None, start_frame: int = 0):
    """Yield (points, labels, pose) per frame for a sequence.

    Args:
        sequence: "00", "07", or "08"
        max_frames: optional limit -- yield at most this many frames
        start_frame: skip to this frame index before yielding (default 0);
            `max_frames` then counts from here. Lets a demo recording start
            partway through a sequence without streaming the frames before it.

    Yields:
        points: (N, 4) float32 — x, y, z, intensity in SENSOR frame
        labels: (N,) uint32 — raw SemanticKITTI labels (0-259)
        pose: (3, 4) float64 — official KITTI GT pose Vehicle → World
    """
    gt_poses = poses(sequence)
    available = _available_frames(sequence)

    if not available:
        raise FileNotFoundError(f"No velodyne frames found for sequence {sequence}")

    if start_frame:
        available = [f for f in available if f >= start_frame]
    if max_frames is not None:
        available = available[:max_frames]

    for frame_idx in available:
        # Load velodyne scan
        bin_path = _velodyne_path(sequence, frame_idx)
        points = load_velodyne_scan(bin_path)

        # Load labels if available
        label_path = _label_path(sequence, frame_idx)
        if label_path.exists():
            labels = load_labels(label_path)
        else:
            labels = np.zeros(points.shape[0], dtype=np.uint32)  # placeholder

        # Official GT pose: Vehicle → World (use frame_idx as pose index)
        if frame_idx >= gt_poses.shape[0]:
            raise IndexError(f"Frame {frame_idx} exceeds GT poses ({gt_poses.shape[0]})")
        pose = gt_poses[frame_idx]

        yield points, labels, pose


def poses(sequence: str, source: str | None = None) -> np.ndarray:
    """All poses for a sequence as (N, 3, 4), from whichever file it uses.

    `source` is "gt", "slam", or None for the per-sequence default -- see
    `pose_source` and `POSE_SOURCE_BY_SEQUENCE` for why 08 differs. Both files
    carry the same convention (Camera-0 -> World_cam), so callers downstream of
    this need no change.
    """
    src = pose_source(sequence) if source is None else source
    if src != "slam":
        return load_gt_poses(sequence)
    # ⚑ Fall back rather than fail when the SLAM file is absent. It ships with
    #   the label archive, so a root without it is an incomplete extract -- and
    #   also every synthetic fixture that writes a sequence called "08" without
    #   one. Failing hard there would make the default for ONE sequence break
    #   test data that has nothing to do with it. The fallback is announced,
    #   because on real 08 the GT poses are the 16.6 cm problem.
    if not (VELODYNE_DIR / sequence / "poses.txt").exists():
        warnings.warn(
            f"sequence {sequence} defaults to SemanticKITTI SLAM poses but "
            f"{VELODYNE_DIR / sequence / 'poses.txt'} is missing; falling back "
            "to official GT poses. On real sequence 08 those disagree between "
            "consecutive frames by 16.6 cm and the accumulated map is not "
            "reportable -- extract the label archive over this root.",
            RuntimeWarning, stacklevel=2)
        return load_gt_poses(sequence)
    return load_slam_poses(sequence)


def get_frame_count(sequence: str) -> int:
    """Number of frames in a sequence, from whichever pose file it uses."""
    return poses(sequence).shape[0]


def verify_sequence_exists(sequence: str) -> bool:
    """Check if sequence data is available."""
    gt_path = _gt_poses_path(sequence)
    velo_dir = VELODYNE_DIR / sequence / "velodyne"
    return gt_path.exists() and velo_dir.exists()