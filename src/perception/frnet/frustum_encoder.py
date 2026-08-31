# NON-FUNCTIONAL -- see the header of frnet.py. This standalone port does not
# reproduce the trained network (wrong backbone activation, wrong FOV params,
# missing RangeInterpolation; the manual scatter_max/scatter_mean here are
# unaudited against the real source). Kept, not deleted, for a possible real
# FRNet install later. semantics.py uses ground-truth .label files instead.

"""Frustum Feature Encoder — standalone, no torch_scatter deps. NON-FUNCTIONAL."""

import torch
import torch.nn as nn
from typing import Optional, Sequence


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Manual scatter_max - reliable but slower."""
    if dim_size is None:
        dim_size = index.max().item() + 1
    out_shape = (dim_size,) + src.shape[1:]
    out = torch.full(out_shape, float('-inf'), dtype=src.dtype, device=src.device)
    argmax = torch.zeros(out_shape, dtype=torch.long, device=src.device)

    # Manual implementation
    for i in range(dim_size):
        mask = (index == i)
        if mask.any():
            vals, idxs = src[mask].max(dim=0)
            out[i] = vals
            argmax[i] = idxs
    return out, argmax


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: Optional[int] = None) -> torch.Tensor:
    """Manual scatter_mean - reliable but slower."""
    if dim_size is None:
        dim_size = index.max().item() + 1
    out_shape = (dim_size,) + src.shape[1:]
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)

    for i in range(dim_size):
        mask = (index == i)
        if mask.any():
            out[i] = src[mask].mean(dim=0)
    return out


class FrustumFeatureEncoder(nn.Module):
    """
    Frustum Feature Encoder (Voxel Encoder in FRNet).

    Encodes raw 3D points (x, y, z, intensity) into frustum/voxel features.
    Matches FRNet config: in_channels=4, feat_channels=(64, 128, 256, 256),
    with_distance=True, with_cluster_center=True, feat_compression=16.
    """

    def __init__(
        self,
        in_channels: int = 4,
        feat_channels: Sequence[int] = (64, 128, 256, 256),
        with_distance: bool = True,
        with_cluster_center: bool = True,
        feat_compression: Optional[int] = 16,
    ) -> None:
        super().__init__()

        _in_channels = in_channels
        if with_distance:
            _in_channels += 1
        if with_cluster_center:
            _in_channels += 3
        self.in_channels = _in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center

        feat_channels = [self.in_channels] + list(feat_channels)

        self.pre_norm = nn.BatchNorm1d(self.in_channels, eps=1e-3, momentum=0.01)

        ffe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i == len(feat_channels) - 2:
                ffe_layers.append(nn.Linear(in_filters, out_filters))
            else:
                ffe_layers.append(nn.Sequential(
                    nn.Linear(in_filters, out_filters, bias=False),
                    nn.BatchNorm1d(out_filters, eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True)))
        self.ffe_layers = nn.ModuleList(ffe_layers)

        self.compression_layers = None
        if feat_compression is not None:
            self.compression_layers = nn.Sequential(
                nn.Linear(feat_channels[-1], feat_compression),
                nn.ReLU(inplace=True))

    def forward(self, voxel_dict: dict) -> dict:
        features = voxel_dict['voxels']          # (N, 4) — x, y, z, intensity
        coors = voxel_dict['coors']              # (N, 3) — batch_idx, y, x (frustum coords)

        features_ls = [features]

        # Unique voxel coordinates (frustum pixels)
        voxel_coors, inverse_map = torch.unique(coors, return_inverse=True, dim=0)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)

        if self._with_cluster_center:
            voxel_mean = scatter_mean(features, inverse_map, dim=0)
            points_mean = voxel_mean[inverse_map]
            f_cluster = features[:, :3] - points_mean[:, :3]
            features_ls.append(f_cluster)

        features = torch.cat(features_ls, dim=-1)
        features = self.pre_norm(features)

        point_feats = []
        for ffe in self.ffe_layers:
            features = ffe(features)
            point_feats.append(features)

        # Max pool to voxel/frustum level
        voxel_feats, _ = scatter_max(features, inverse_map, dim=0)

        if self.compression_layers is not None:
            voxel_feats = self.compression_layers(voxel_feats)

        voxel_dict['voxel_feats'] = voxel_feats          # (M, 16) — M unique frustum pixels
        voxel_dict['voxel_coors'] = voxel_coors          # (M, 3)
        voxel_dict['point_feats'] = point_feats          # List of (N, C_i)

        return voxel_dict