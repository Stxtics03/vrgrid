"""Semantic labels. [JP]

Semantic class per point comes STRAIGHT FROM the SemanticKITTI `.label` files,
mapped to the 19-class scheme. Nothing is inferred and nothing is retrained.

Disclose it plainly in the report: like the motion labels (see loader.py),
the semantic labels are ground truth. This is deliberate -- it isolates the
mapping contribution from segmentation error, which is what a careful evaluator
wants. The variable-resolution grid is the contribution; how the points got
their class is not.

Why not FRNet: the pretrained checkpoint is available, but the only standalone
implementation we have (src/perception/frnet/) does not reproduce the trained
network -- wrong backbone activation (LeakyReLU vs HSwish), wrong FOV params,
missing RangeInterpolation -- and collapses to ~15% point accuracy on every
frame. Running FRNet properly needs its full mmengine/mmcv/mmdet/mmdet3d stack.
That is a possible later swap (Option B); the port code is kept, flagged
non-functional, for that. The module-level segment*/get_frnet helpers below
raise rather than silently return that garbage.

The 19-class scheme and the raw-id -> class map are the canonical SemanticKITTI
ones (FRNet repo configs/_base_/datasets/semantickitti_seg.py labels_map), so a
future FRNet swap lines up with no relabelling.
"""

import os
from pathlib import Path

import numpy as np
import yaml

# torch and the FRNet port are imported lazily inside FRNetInference only. The
# ground-truth label path (semantic_labels / is_moving) is pure numpy, so this
# module must import with neither torch nor the frnet package present.

# Resolved from this file, not the CWD -- matches src/grid/schedule.py.
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
_FRNET_YAML = CONFIG_DIR / "frnet.yaml"

# Raw SemanticKITTI id (lower 16 bits of the .label word) -> 19-class index.
# 19 = unlabeled/ignore. moving-* ids (252-259) fold onto their static class
# for the SEMANTIC label; motion is a separate signal (is_moving / loader.py).
SEMANTIC_KITTI_LABEL_MAP = {
    0: 19, 1: 19, 10: 0, 11: 1, 13: 4, 15: 2, 16: 4, 18: 3, 20: 4, 30: 5,
    31: 6, 32: 7, 40: 8, 44: 9, 48: 10, 49: 11, 50: 12, 51: 13, 52: 19,
    60: 8, 70: 14, 71: 15, 72: 16, 80: 17, 81: 18, 99: 19,
    252: 0, 253: 6, 254: 5, 255: 7, 256: 4, 257: 4, 258: 3, 259: 4,
}
MOVING_LABEL_IDS = range(250, 260)  # SemanticKITTI moving-* raw ids

_MAX_RAW = max(SEMANTIC_KITTI_LABEL_MAP) + 1
_LUT = np.full(_MAX_RAW, -1, dtype=np.int32)  # unmapped raw id -> -1 (ignore)
for _raw, _cls in SEMANTIC_KITTI_LABEL_MAP.items():
    _LUT[_raw] = -1 if _cls == 19 else _cls


def semantic_labels(raw_labels: np.ndarray) -> np.ndarray:
    """Map raw SemanticKITTI `.label` words to 19-class indices.

    Args:
        raw_labels: (N,) uint32 -- as returned by loader.load_labels().

    Returns:
        (N,) int32 -- class index 0-18, or -1 for unlabeled / ignore / any raw
        id outside the SemanticKITTI scheme.
    """
    sem = np.asarray(raw_labels, dtype=np.uint32) & 0xFFFF
    out = np.full(sem.shape, -1, dtype=np.int32)
    in_range = sem < _MAX_RAW
    out[in_range] = _LUT[sem[in_range]]
    return out


def is_moving(raw_labels: np.ndarray) -> np.ndarray:
    """(N,) bool -- True where the raw label is a SemanticKITTI moving-* id."""
    sem = np.asarray(raw_labels, dtype=np.uint32) & 0xFFFF
    return (sem >= MOVING_LABEL_IDS.start) & (sem < MOVING_LABEL_IDS.stop)


# The projection the SemanticKITTI checkpoint was TRAINED with. Not the
# sensor's field of view -- see the note at the FRNet construction below.
FRNET_TRAIN_FOV_UP_DEG = 3.0
FRNET_TRAIN_FOV_DOWN_DEG = -25.0

_PORT_BROKEN = (
    "FRNet inference is disabled: the standalone port in src/perception/frnet/ "
    "does not reproduce the trained network (see its header). Use "
    "semantic_labels(raw_labels) -- semantic class comes from the SemanticKITTI "
    ".label files. To re-enable FRNet, install the real mmdet3d stack and "
    "replace this module."
)


# FRNet model class names (from configs/_base_/datasets/semantickitti_seg.py)
# Model outputs 0-18 = semantic classes, 19 = unlabeled/ignore
FRNET_CLASS_NAMES = [
    "car",             # 0
    "bicycle",         # 1
    "motorcycle",      # 2
    "truck",           # 3
    "other-vehicle",   # 4
    "person",          # 5
    "bicyclist",       # 6
    "motorcyclist",    # 7
    "road",            # 8
    "parking",         # 9
    "sidewalk",        # 10
    "other-ground",    # 11
    "building",        # 12
    "fence",           # 13
    "vegetation",      # 14
    "trunk",           # 15
    "terrain",         # 16
    "pole",            # 17
    "traffic-sign",    # 18
    "unlabeled",       # 19 (ignore)
]

# Mapping: model outputs 0-19, where 0-18 = semantic classes, 19 = ignore
MODEL_TO_SEMANTIC = {i: i for i in range(19)}  # 0->0, 1->1, ..., 18->18
# Model class 19 (unlabeled) maps to -1 (ignore)


class FRNetInference:
    """FRNet 20-class (19 semantic) semantic segmentation inference."""

    def __init__(self, config_path: str | Path = _FRNET_YAML):
        import torch  # lazy: the GT-label path must import without torch

        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.num_classes = self.cfg["model"]["num_classes"]  # 20
        self.class_names = self.cfg["model"]["class_names"]
        self.checkpoint_path = Path(
            os.environ.get("VRGRID_FRNET_CHECKPOINT", self.cfg["model"]["checkpoint"])
        ).expanduser()
        self.sensor_cfg = self.cfg["sensor"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load FRNet model from checkpoint."""
        import torch

        from .frnet import FRNet

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"FRNet checkpoint not found at {self.checkpoint_path}. "
                f"Expected path from configs/frnet.yaml"
            )

        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        # Instantiate our standalone FRNet
        self.model = FRNet(
            num_classes=self.num_classes,
            ignore_index=19,
            output_shape=(self.sensor_cfg["num_rings"], self.sensor_cfg["num_azimuth"]),
            # ⚑ TRAINING FOV, NOT THE SENSOR'S. These were being passed from
            #   `sensor_cfg` -- 2.0 / -24.8, the HDL-64E's real vertical field
            #   of view -- and that is a different quantity. The checkpoint was
            #   trained with a FIXED spherical projection at 3.0 / -25.0, so
            #   every point must be projected into the same grid the weights
            #   learned, whatever sensor the points came from. Feeding the
            #   physical FOV shifts every row of the range image against the
            #   filters and is one of the three reasons inference collapsed to
            #   ~15% point accuracy.
            fov_up=FRNET_TRAIN_FOV_UP_DEG,
            fov_down=FRNET_TRAIN_FOV_DOWN_DEG,
        )

        # Map checkpoint keys to our model
        # Checkpoint uses: backbone.stem, backbone.point_stem, backbone.fusion_stem,
        # backbone.point_fusion_layers, backbone.pixel_fusion_layers, backbone.attention_layers,
        # backbone.layer1-4, backbone.fuse_layer1-2, backbone.point_fuse_layer1-2,
        # decode_head.mlps, decode_head.cls_seg, voxel_encoder.pre_norm, voxel_encoder.ffe_layers,
        # voxel_encoder.compression_layers
        # Our model uses: voxel_encoder, backbone, decode_head
        self._load_state_dict_mapped(state_dict)

        self.model.to(self.device)
        self.model.eval()

    def _load_state_dict_mapped(self, state_dict: dict):
        """Map checkpoint state_dict keys to our model structure."""
        model_dict = self.model.state_dict()
        mapped = {}

        for k, v in state_dict.items():
            # Voxel encoder
            if k.startswith(("voxel_encoder.", "backbone.", "decode_head.")):
                new_k = k  # Same structure
                if new_k in model_dict:
                    mapped[new_k] = v

            # Auxiliary heads - skip (not in our inference model)
            elif k.startswith("auxiliary_head."):
                continue

        # Load with strict=False to see what's missing
        missing, unexpected = self.model.load_state_dict(mapped, strict=False)

        if missing:
            print(f"[FRNet] Missing keys ({len(missing)}): {missing[:10]}...")
        if unexpected:
            print(f"[FRNet] Unexpected keys ({len(unexpected)}): {unexpected[:10]}...")

        print(f"[FRNet] Loaded {len(mapped)} / {len(model_dict)} parameters from checkpoint")

    def infer_points(self, points: np.ndarray) -> np.ndarray:
        """
        Run FRNet inference on raw 3D points.

        Args:
            points: (N, 4) float32 — [x, y, z, intensity] in sensor frame

        Returns:
            per_point_labels: (N,) int32 — class indices 0-18 (19 semantic classes), -1=ignore
        """
        import torch

        with torch.no_grad():
            pts_tensor = torch.from_numpy(points).float().to(self.device)
            if pts_tensor.ndim == 2:
                pts_tensor = pts_tensor.unsqueeze(0)  # (1, N, 4)

            # Model expects List[Tensor]
            pts_list = (
                [pts_tensor[0]] if pts_tensor.shape[0] == 1 else list(pts_tensor.unbind(0))
            )

            # Forward pass
            pred_list = self.model.predict(pts_list)
            per_point_labels = pred_list[0].cpu().numpy()  # (N,)

        # Map from model classes (0-18 = semantic, 19 = ignore) to semantic (0-18)
        semantic_labels = np.full_like(per_point_labels, -1, dtype=np.int32)
        for model_cls, sem_cls in MODEL_TO_SEMANTIC.items():
            semantic_labels[per_point_labels == model_cls] = sem_cls
        # model class 19 (unlabeled) stays -1

        return semantic_labels

    def infer_range_image(self, range_image: np.ndarray, inverse_index: np.ndarray) -> np.ndarray:
        """
        Run FRNet inference and project results to range image.

        Args:
            range_image: (64, 512, 5) — [range, x, y, z, intensity]
            inverse_index: (64, 512) — source point index for each pixel

        Returns:
            range_image_labels: (64, 512) int32 — class indices 0-18, -1 for invalid
        """
        # Extract valid points from range image
        valid_mask = inverse_index >= 0

        # Reconstruct points from range_image (x, y, z are already there)
        points_xyz = range_image[valid_mask, 1:4]  # (M, 3)
        intensity = range_image[valid_mask, 4:5]   # (M, 1)
        points = np.hstack([points_xyz, intensity]).astype(np.float32)

        # Run inference on points
        point_labels = self.infer_points(points)  # (M,)

        # Scatter back to range image
        range_labels = np.full((64, 512), -1, dtype=np.int32)
        range_labels[valid_mask] = point_labels

        return range_labels


def get_frnet(config_path: str | Path = _FRNET_YAML) -> FRNetInference:
    """Disabled -- the standalone FRNet port is non-functional. See _PORT_BROKEN."""
    raise RuntimeError(_PORT_BROKEN)


def segment(points: np.ndarray, config_path: str | Path = _FRNET_YAML) -> np.ndarray:
    """Disabled -- use semantic_labels(raw_labels). See _PORT_BROKEN."""
    raise RuntimeError(_PORT_BROKEN)


def segment_range_image(
    range_image: np.ndarray,
    inverse_index: np.ndarray,
    config_path: str | Path = _FRNET_YAML,
) -> np.ndarray:
    """Disabled -- use semantic_labels(raw_labels). See _PORT_BROKEN."""
    raise RuntimeError(_PORT_BROKEN)


def segment_with_probs(
    points: np.ndarray, config_path: str | Path = _FRNET_YAML
) -> tuple[np.ndarray, np.ndarray]:
    """Disabled -- ground-truth .label files carry no probabilities. See _PORT_BROKEN."""
    raise RuntimeError(_PORT_BROKEN)