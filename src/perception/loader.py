"""SemanticKITTI loader. [JP — Day 0, first task]

Sequences 00, 07, 08 only — about 40 GB, not the full 200 GB. Start the
download before anything else on Day 0; it is the one item on the critical
path that neither cleverness nor effort can accelerate.

Motion labels come straight out of the raw `.label` files (`moving-*`,
IDs 250-259). Nothing is retrained. Disclose it plainly in the report: motion
labels are ground truth, so the mapping contribution is evaluated independently
of segmentation quality. That is a feature — it isolates the contribution from
segmentation error.

USES OFFICIAL KITTI GROUND-TRUTH POSES from `C:/KITTI/dataset/poses/<seq>.txt`
NOT the SemanticKITTI internal SLAM poses at `data/sequences/<seq>/poses.txt`.
VELODYNE SCANS AND LABELS from `C:/KITTI/dataset/sequences/<seq>/velodyne/` and `labels/`.
"""

import numpy as np
from pathlib import Path

# Data root — use the REAL KITTI dataset at C:/KITTI/dataset
DATA_ROOT = Path(r"C:/KITTI/dataset")

# Official KITTI ground-truth poses (top-level)
GT_POSES_DIR = DATA_ROOT / "poses"

# SemanticKITTI velodyne scans and labels (full dataset)
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


def scans(sequence: str, max_frames: int = None):
    """Yield (points, labels, pose) per frame for a sequence.

    Args:
        sequence: "00", "07", or "08"
        max_frames: optional limit for testing (e.g., 10 frames)

    Yields:
        points: (N, 4) float32 — x, y, z, intensity in SENSOR frame
        labels: (N,) uint32 — raw SemanticKITTI labels (0-259)
        pose: (3, 4) float64 — official KITTI GT pose Vehicle → World
    """
    gt_poses = load_gt_poses(sequence)
    available = _available_frames(sequence)

    if not available:
        raise FileNotFoundError(f"No velodyne frames found for sequence {sequence}")

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


def poses(sequence: str) -> np.ndarray:
    """Return all official GT poses for a sequence as (N, 3, 4)."""
    return load_gt_poses(sequence)


def get_frame_count(sequence: str) -> int:
    """Return number of frames in sequence (from GT poses)."""
    return load_gt_poses(sequence).shape[0]


def verify_sequence_exists(sequence: str) -> bool:
    """Check if sequence data is available."""
    gt_path = _gt_poses_path(sequence)
    velo_dir = VELODYNE_DIR / sequence / "velodyne"
    return gt_path.exists() and velo_dir.exists()