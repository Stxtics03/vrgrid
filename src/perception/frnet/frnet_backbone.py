"""FRNet Backbone — standalone, no mmcv/mmdet3d/torch_scatter deps."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Tuple, List

from .frustum_encoder import scatter_max


class BasicBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv1 = nn.Conv2d(
            inplanes, planes, 3, stride=stride, padding=dilation,
            dilation=dilation, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.relu = nn.LeakyReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ConvModule(nn.Module):
    """mmcv ConvModule compatible: has conv, bn, act attributes."""
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=False, act_cfg=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(inplace=True) if act_cfg is None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class FRNetBackbone(nn.Module):
    """
    FRNet Backbone with frustum-point fusion.

    Matches config: in_channels=16, point_in_channels=384, depth=34,
    stem_channels=128, num_stages=4, out_channels=(128,128,128,128),
    strides=(1,2,2,2), fuse_channels=(256,128).
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
    }

    def __init__(
        self,
        in_channels: int = 16,
        point_in_channels: int = 384,
        output_shape: Sequence[int] = (64, 512),
        depth: int = 34,
        stem_channels: int = 128,
        num_stages: int = 4,
        out_channels: Sequence[int] = (128, 128, 128, 128),
        strides: Sequence[int] = (1, 2, 2, 2),
        dilations: Sequence[int] = (1, 1, 1, 1),
        fuse_channels: Sequence[int] = (256, 128),
    ) -> None:
        super().__init__()

        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for FRNetBackbone.')

        self.block, stage_blocks = self.arch_settings[depth]
        self.output_shape = output_shape
        self.ny = output_shape[0]
        self.nx = output_shape[1]
        assert len(stage_blocks) == len(out_channels) == len(strides) == len(dilations) == num_stages

        self.stem = self._make_stem_layer(in_channels, stem_channels)
        self.point_stem = self._make_point_layer(point_in_channels, stem_channels)
        self.fusion_stem = self._make_fusion_layer(stem_channels * 2, stem_channels)

        inplanes = stem_channels
        self.res_layers = []
        self.point_fusion_layers = nn.ModuleList()
        self.pixel_fusion_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.strides = []
        overall_stride = 1

        for i, num_blocks in enumerate(stage_blocks):
            stride = strides[i]
            overall_stride = stride * overall_stride
            self.strides.append(overall_stride)
            dilation = dilations[i]
            planes = out_channels[i]
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation)
            self.point_fusion_layers.append(
                self._make_point_layer(inplanes + planes, planes))
            self.pixel_fusion_layers.append(
                self._make_fusion_layer(planes * 2, planes))
            self.attention_layers.append(self._make_attention_layer(planes))
            inplanes = planes
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        in_channels = stem_channels + sum(out_channels)
        self.fuse_layers = []
        self.point_fuse_layers = []
        for i, fuse_channel in enumerate(fuse_channels):
            fuse_layer = ConvModule(in_channels, fuse_channel, 3, padding=1, bias=False)
            point_fuse_layer = self._make_point_layer(in_channels, fuse_channel)
            in_channels = fuse_channel
            layer_name = f'fuse_layer{i + 1}'
            point_layer_name = f'point_fuse_layer{i + 1}'
            self.add_module(layer_name, fuse_layer)
            self.add_module(point_layer_name, point_fuse_layer)
            self.fuse_layers.append(layer_name)
            self.point_fuse_layers.append(point_layer_name)

    def _make_stem_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True))

    def _make_point_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True))

    def _make_fusion_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True))

    def _make_attention_layer(self, channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid())

    def make_res_layer(
        self,
        block: nn.Module,
        inplanes: int,
        planes: int,
        num_blocks: int,
        stride: int,
        dilation: int,
    ) -> nn.Module:
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, dilation, downsample))
        inplanes = planes
        for _ in range(1, num_blocks):
            layers.append(block(inplanes, planes, 1, dilation))
        return nn.Sequential(*layers)

    def forward(self, voxel_dict: dict) -> dict:
        point_feats = voxel_dict['point_feats'][-1]       # (N, 256) — last FFE layer
        voxel_feats = voxel_dict['voxel_feats']           # (M, 16) — compressed frustum feats
        voxel_coors = voxel_dict['voxel_coors']           # (M, 3) — batch, y, x
        pts_coors = voxel_dict['coors']                   # (N, 3) — batch, y, x per point
        batch_size = pts_coors[-1, 0].item() + 1

        # Frustum (voxel) to pixel (range image grid)
        x = self.frustum2pixel(voxel_feats, voxel_coors, batch_size, stride=1)
        x = self.stem(x)                                  # (B, 128, H, W)

        # Pixel to point
        map_point_feats = self.pixel2point(x, pts_coors, stride=1)  # (N, 128)
        fusion_point_feats = torch.cat((map_point_feats, point_feats), dim=1)  # (N, 128+256)
        point_feats = self.point_stem(fusion_point_feats)  # (N, 128)

        # Point to frustum
        stride_voxel_coors, frustum_feats = self.point2frustum(point_feats, pts_coors, stride=1)
        pixel_feats = self.frustum2pixel(frustum_feats, stride_voxel_coors, batch_size, stride=1)
        fusion_pixel_feats = torch.cat((pixel_feats, x), dim=1)  # (B, 256, H, W)
        x = self.fusion_stem(fusion_pixel_feats)                # (B, 128, H, W)

        outs = [x]
        out_points = [point_feats]

        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)

            # Frustum-to-point fusion
            map_point_feats = self.pixel2point(x, pts_coors, stride=self.strides[i])
            fusion_point_feats = torch.cat((map_point_feats, point_feats), dim=1)
            point_feats = self.point_fusion_layers[i](fusion_point_feats)

            # Point-to-frustum fusion
            stride_voxel_coors, frustum_feats = self.point2frustum(point_feats, pts_coors, stride=self.strides[i])
            pixel_feats = self.frustum2pixel(frustum_feats, stride_voxel_coors, batch_size, stride=self.strides[i])
            fusion_pixel_feats = torch.cat((pixel_feats, x), dim=1)
            fuse_out = self.pixel_fusion_layers[i](fusion_pixel_feats)

            # Residual attention
            attention_map = self.attention_layers[i](fuse_out)
            x = fuse_out * attention_map + x

            outs.append(x)
            out_points.append(point_feats)

        # Upsample all outputs to same resolution
        for i in range(len(outs)):
            if outs[i].shape != outs[0].shape:
                outs[i] = F.interpolate(outs[i], size=outs[0].size()[2:], mode='bilinear', align_corners=True)

        outs[0] = torch.cat(outs, dim=1)
        out_points[0] = torch.cat(out_points, dim=1)

        for layer_name, point_layer_name in zip(self.fuse_layers, self.point_fuse_layers):
            fuse_layer = getattr(self, layer_name)
            outs[0] = fuse_layer(outs[0])
            point_fuse_layer = getattr(self, point_layer_name)
            out_points[0] = point_fuse_layer(out_points[0])

        voxel_dict['voxel_feats'] = outs[0]              # (B, C, H, W) — fused frustum feats
        voxel_dict['point_feats_backbone'] = out_points  # List[(N, C)]

        return voxel_dict

    def frustum2pixel(self, frustum_features: torch.Tensor, coors: torch.Tensor, batch_size: int, stride: int = 1) -> torch.Tensor:
        nx = self.nx // stride
        ny = self.ny // stride
        pixel_features = torch.zeros(
            (batch_size, ny, nx, frustum_features.shape[-1]),
            dtype=frustum_features.dtype, device=frustum_features.device)
        pixel_features[coors[:, 0], coors[:, 1], coors[:, 2]] = frustum_features
        pixel_features = pixel_features.permute(0, 3, 1, 2).contiguous()
        return pixel_features

    def pixel2point(self, pixel_features: torch.Tensor, coors: torch.Tensor, stride: int = 1) -> torch.Tensor:
        pixel_features = pixel_features.permute(0, 2, 3, 1).contiguous()
        point_feats = pixel_features[coors[:, 0], coors[:, 1] // stride, coors[:, 2] // stride]
        return point_feats

    def point2frustum(self, point_features: torch.Tensor, pts_coors: torch.Tensor, stride: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        coors = pts_coors.clone()
        coors[:, 1] = pts_coors[:, 1] // stride
        coors[:, 2] = pts_coors[:, 2] // stride
        voxel_coors, inverse_map = torch.unique(coors, return_inverse=True, dim=0)
        frustum_features, _ = scatter_max(point_features, inverse_map, dim=0)
        return voxel_coors, frustum_features