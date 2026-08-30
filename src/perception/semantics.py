"""Semantic segmentation. [JP]

Pretrained FRNet, 20-class (19 semantic + 1 ignore at index 19), off the shelf, Apache 2.0.
ZERO training. Wire it in; do not reimplement it and do not fine-tune it.
The checkpoint path lives in the config, never inline.

Uses the FULL FRNet architecture: frustum (range-image) + point fusion.
Input: raw 3D points (x, y, z, intensity) from loader.py
Output: per-point class predictions (mapped to range image via inverse index)
"""

# TODO(Day 1): FRNet class distribution on frame 000043 looked skewed
# (50% other-ground, minimal road/vegetation) - sanity-check projection/
# intensity pipeline before trusting outputs further.

import torch
import numpy as np
import yaml
from pathlib import Path

from .frnet import FRNet
from .range_image import project, load_sensor_config


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

    def __init__(self, config_path: str = "configs/frnet.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.num_classes = self.cfg["model"]["num_classes"]  # 20
        self.class_names = self.cfg["model"]["class_names"]
        self.checkpoint_path = Path(self.cfg["model"]["checkpoint"])
        self.sensor_cfg = self.cfg["sensor"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load FRNet model from checkpoint."""
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
            fov_up=self.sensor_cfg.get("phi_max_deg", 2.0),
            fov_down=self.sensor_cfg["phi_min_deg"],
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
            if k.startswith("voxel_encoder."):
                new_k = k  # Same structure
                if new_k in model_dict:
                    mapped[new_k] = v

            # Backbone
            elif k.startswith("backbone."):
                new_k = k  # Same structure
                if new_k in model_dict:
                    mapped[new_k] = v

            # Decode head
            elif k.startswith("decode_head."):
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

    @torch.no_grad()
    def infer_points(self, points: np.ndarray) -> np.ndarray:
        """
        Run FRNet inference on raw 3D points.

        Args:
            points: (N, 4) float32 — [x, y, z, intensity] in sensor frame

        Returns:
            per_point_labels: (N,) int32 — class indices 0-18 (19 semantic classes), -1=ignore
        """
        # Convert to tensor
        pts_tensor = torch.from_numpy(points).float().to(self.device)
        if pts_tensor.ndim == 2:
            pts_tensor = pts_tensor.unsqueeze(0)  # (1, N, 4)

        # Model expects List[Tensor]
        pts_list = [pts_tensor[0]] if pts_tensor.shape[0] == 1 else list(pts_tensor.unbind(0))

        # Forward pass
        pred_list = self.model.predict(pts_list)
        per_point_labels = pred_list[0].cpu().numpy()  # (N,)

        # Map from model classes (0-18 = semantic, 19 = ignore) to semantic (0-18)
        semantic_labels = np.full_like(per_point_labels, -1, dtype=np.int32)
        for model_cls, sem_cls in MODEL_TO_SEMANTIC.items():
            semantic_labels[per_point_labels == model_cls] = sem_cls
        # model class 19 (unlabeled) stays -1

        return semantic_labels

    @torch.no_grad()
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
        valid_pixels = np.where(valid_mask)
        point_indices = inverse_index[valid_mask]

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


# Global instance for reuse
_frnet_instance = None


def get_frnet(config_path: str = "configs/frnet.yaml") -> FRNetInference:
    """Get or create FRNet inference instance."""
    global _frnet_instance
    if _frnet_instance is None:
        _frnet_instance = FRNetInference(config_path)
    return _frnet_instance


def segment(points: np.ndarray, config_path: str = "configs/frnet.yaml") -> np.ndarray:
    """
    Segment raw 3D points using pretrained FRNet.

    Args:
        points: (N, 4) float32 — [x, y, z, intensity] in sensor frame
        config_path: Path to FRNet config YAML

    Returns:
        semantic_labels: (N,) int32 — SemanticKITTI 19-class labels (0-18), -1=unlabeled
    """
    frnet = get_frnet(config_path)
    return frnet.infer_points(points)


def segment_range_image(
    range_image: np.ndarray,
    inverse_index: np.ndarray,
    config_path: str = "configs/frnet.yaml"
) -> np.ndarray:
    """
    Segment range image using pretrained FRNet (projects to points, infers, projects back).

    Args:
        range_image: (64, 512, 5) float32 — [range_m, x, y, z, intensity]
        inverse_index: (64, 512) int32 — source point index per pixel
        config_path: Path to FRNet config YAML

    Returns:
        semantic_labels: (64, 512) int32 — class indices 0-18, -1 for invalid
    """
    frnet = get_frnet(config_path)
    return frnet.infer_range_image(range_image, inverse_index)


def segment_with_probs(points: np.ndarray, config_path: str = "configs/frnet.yaml") -> tuple[np.ndarray, np.ndarray]:
    """
    Segment points and return both labels and probabilities.

    Returns:
        labels: (N,) int32 — class indices 0-18, -1=unlabeled
        probs: (N, 19) float32 — softmax probabilities for 19 semantic classes
    """
    frnet = get_frnet(config_path)
    device = frnet.device

    pts_tensor = torch.from_numpy(points).float().to(device)
    if pts_tensor.ndim == 2:
        pts_tensor = pts_tensor.unsqueeze(0)

    pts_list = [pts_tensor[0]] if pts_tensor.shape[0] == 1 else list(pts_tensor.unbind(0))

    frnet.model.eval()
    with torch.no_grad():
        voxel_dict = frnet.model.forward(pts_list)
        logits = voxel_dict['seg_logit']  # (N_total, 20)
        probs_20 = torch.softmax(logits, dim=1).cpu().numpy()  # (N, 20)
        pred_20 = logits.argmax(dim=1).cpu().numpy()  # (N,)

    # Map to 19 semantic classes (model 0-18 -> semantic 0-18, model 19 -> ignore)
    labels = np.full_like(pred_20, -1, dtype=np.int32)
    probs_19 = np.zeros((len(pred_20), 19), dtype=np.float32)
    for model_cls, sem_cls in MODEL_TO_SEMANTIC.items():
        mask = pred_20 == model_cls
        labels[mask] = sem_cls
        probs_19[:, sem_cls] = probs_20[:, model_cls]

    return labels, probs_19