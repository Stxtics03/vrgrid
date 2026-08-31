# NON-FUNCTIONAL -- see the header of frnet.py. This standalone port does not
# reproduce the trained network (wrong backbone activation, wrong FOV params,
# missing RangeInterpolation). Kept, not deleted, for a possible real FRNet
# install later. semantics.py uses ground-truth .label files instead.

"""FRNet Decode Head — standalone, no mmcv/mmdet3d deps. NON-FUNCTIONAL."""

import torch
import torch.nn as nn
from typing import Sequence


class FRHead(nn.Module):
    """
    FRNet Decode Head.

    Matches config: in_channels=128, middle_channels=(128, 256, 128, 64),
    channels=64, num_classes=20, ignore_index=19.
    """

    def __init__(
        self,
        in_channels: int = 128,
        middle_channels: Sequence[int] = (128, 256, 128, 64),
        num_classes: int = 20,
        ignore_index: int = 19,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.mlps = nn.ModuleList()
        for i in range(len(middle_channels)):
            out_channels = middle_channels[i]
            self.mlps.append(nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)))
            in_channels = out_channels

        self.conv_seg = nn.Linear(middle_channels[-1], num_classes)

    def forward(self, voxel_dict: dict) -> dict:
        point_feats_backbone = voxel_dict['point_feats_backbone'][0]  # (N, C)
        point_feats = voxel_dict['point_feats'][:-1]                  # List[(N, C_i)] from FFE
        voxel_feats = voxel_dict['voxel_feats']                       # (B, C, H, W)
        pts_coors = voxel_dict['coors']                               # (N, 3)

        # Frustum features to point features
        voxel_feats = voxel_feats.permute(0, 2, 3, 1)                # (B, H, W, C)
        map_point_feats = voxel_feats[pts_coors[:, 0], pts_coors[:, 1], pts_coors[:, 2]]

        for i, mlp in enumerate(self.mlps):
            map_point_feats = mlp(map_point_feats)
            if i == 0:
                map_point_feats = map_point_feats + point_feats_backbone
            else:
                map_point_feats = map_point_feats + point_feats[-i]

        seg_logit = self.conv_seg(map_point_feats)  # (N, 20)
        voxel_dict['seg_logit'] = seg_logit
        return voxel_dict

    def predict(self, voxel_dict: dict) -> torch.Tensor:
        """Returns per-point class predictions (argmax)."""
        self.forward(voxel_dict)
        return voxel_dict['seg_logit'].argmax(dim=1)