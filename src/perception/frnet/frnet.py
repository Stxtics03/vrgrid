"""Full FRNet model — standalone inference, no mmcv/mmdet3d deps."""

import torch
import torch.nn as nn
from typing import List, Dict, Optional

from .frustum_encoder import FrustumFeatureEncoder
from .frnet_backbone import FRNetBackbone
from .frnet_head import FRHead


class FRNet(nn.Module):
    """
    FRNet: Frustum-Range Network for LiDAR Segmentation.

    Full model with:
    - FrustumFeatureEncoder (voxel encoder): raw points -> frustum features
    - FRNetBackbone: frustum-point fusion backbone
    - FRHead: per-point semantic segmentation

    Expects input dict with:
    - 'points': List[Tensor] — batch of (N, 4) point clouds [x, y, z, intensity]
    """

    def __init__(
        self,
        num_classes: int = 20,
        ignore_index: int = 19,
        output_shape: tuple = (64, 512),
        fov_up: float = 3.0,
        fov_down: float = -25.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.output_shape = output_shape
        self.fov_up = fov_up
        self.fov_down = fov_down
        self.H, self.W = output_shape

        # Voxel encoder (frustum feature encoder)
        self.voxel_encoder = FrustumFeatureEncoder(
            in_channels=4,
            feat_channels=(64, 128, 256, 256),
            with_distance=True,
            with_cluster_center=True,
            feat_compression=16,
        )

        # Backbone with frustum-point fusion
        self.backbone = FRNetBackbone(
            in_channels=16,
            point_in_channels=384,
            output_shape=output_shape,
            depth=34,
            stem_channels=128,
            num_stages=4,
            out_channels=(128, 128, 128, 128),
            strides=(1, 2, 2, 2),
            dilations=(1, 1, 1, 1),
            fuse_channels=(256, 128),
        )

        # Decode head
        self.decode_head = FRHead(
            in_channels=128,
            middle_channels=(128, 256, 128, 64),
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    def frustum_region_group(self, points: List[torch.Tensor]) -> Dict:
        """
        Project points to frustum/range image coordinates.
        Matches FrustumRangePreprocessor.frustum_region_group.
        """
        voxel_dict = {}
        coors_list = []
        voxels_list = []

        fov_up_rad = self.fov_up / 180 * 3.14159265359
        fov_down_rad = self.fov_down / 180 * 3.14159265359
        fov = abs(fov_down_rad) + abs(fov_up_rad)

        for i, res in enumerate(points):
            # res: (N, 4) — x, y, z, intensity
            depth = torch.linalg.norm(res[:, :3], 2, dim=1)
            yaw = -torch.atan2(res[:, 1], res[:, 0])
            pitch = torch.arcsin(res[:, 2] / (depth + 1e-6))

            coors_x = 0.5 * (yaw / 3.14159265359 + 1.0)
            coors_y = 1.0 - (pitch + abs(fov_down_rad)) / fov

            coors_x *= self.W
            coors_y *= self.H

            coors_x = torch.floor(coors_x)
            coors_x = torch.clamp(coors_x, min=0, max=self.W - 1).type(torch.int64)

            coors_y = torch.floor(coors_y)
            coors_y = torch.clamp(coors_y, min=0, max=self.H - 1).type(torch.int64)

            res_coors = torch.stack([coors_y, coors_x], dim=1)
            res_coors = torch.nn.functional.pad(res_coors, (1, 0), mode='constant', value=i)
            coors_list.append(res_coors)
            voxels_list.append(res)

        voxels = torch.cat(voxels_list, dim=0)
        coors = torch.cat(coors_list, dim=0)

        voxel_dict['voxels'] = voxels
        voxel_dict['coors'] = coors
        return voxel_dict

    def forward(self, points: List[torch.Tensor]) -> Dict:
        """
        Full forward pass.

        Args:
            points: List of (N_i, 4) tensors — x, y, z, intensity

        Returns:
            dict with 'seg_logit': (N_total, num_classes) per-point logits
        """
        # Project points to frustum coordinates
        voxel_dict = self.frustum_region_group(points)

        # Voxel encoder: point features -> frustum features
        voxel_dict = self.voxel_encoder(voxel_dict)

        # Backbone: frustum-point fusion
        voxel_dict = self.backbone(voxel_dict)

        # Decode head: per-point predictions
        voxel_dict = self.decode_head(voxel_dict)

        return voxel_dict

    @torch.no_grad()
    def predict(self, points: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Inference: returns per-point class predictions for each batch item.

        Returns:
            List of (N_i,) tensors with class indices 0-19
        """
        self.eval()
        voxel_dict = self.forward(points)
        seg_logit = voxel_dict['seg_logit']  # (N_total, 20)
        seg_pred = seg_logit.argmax(dim=1)   # (N_total,)

        # Split by batch
        pts_coors = voxel_dict['coors']
        pred_list = []
        for batch_idx in range(len(points)):
            batch_mask = pts_coors[:, 0] == batch_idx
            pred_list.append(seg_pred[batch_mask])
        return pred_list