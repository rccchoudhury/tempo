# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmpose.models import BACKBONES
from .base_backbone import BaseBackbone


# 2d conv blocks
class Basic2DBlock(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d')):
        super(Basic2DBlock, self).__init__()

        self.block = ConvModule(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=1,
            padding=((kernel_size - 1) // 2),
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=True)

    def forward(self, x):
        return self.block(x)


class Res2DBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d')):
        super(Res2DBlock, self).__init__()

        self.res_branch = nn.Sequential(
            ConvModule(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                bias=True),
            ConvModule(
                out_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
                bias=True))

        if in_channels == out_channels:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = ConvModule(
                in_channels,
                out_channels,
                1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
                bias=True)

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)  # skip connection
        return F.relu(res + skip, True)


class Pool2DBlock(nn.Module):

    def __init__(self, pool_size):
        super(Pool2DBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        return F.max_pool2d(
            x, kernel_size=self.pool_size, stride=self.pool_size)


class Upsample2DBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample2DBlock, self).__init__()
        assert (kernel_size == 2)
        assert (stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                output_padding=0), nn.BatchNorm2d(out_planes), nn.ReLU(True))

    def forward(self, x):
        return self.block(x)


class EncoderDecorder(nn.Module):

    def __init__(self):
        super(EncoderDecorder, self).__init__()

        self.encoder_pool1 = Pool2DBlock(2)
        self.encoder_res1 = Res2DBlock(32, 64)
        self.encoder_pool2 = Pool2DBlock(2)
        self.encoder_res2 = Res2DBlock(64, 128)

        self.mid_res = Res2DBlock(128, 128)

        self.decoder_res2 = Res2DBlock(128, 128)
        self.decoder_upsample2 = Upsample2DBlock(128, 64, 2, 2)
        self.decoder_res1 = Res2DBlock(64, 64)
        self.decoder_upsample1 = Upsample2DBlock(64, 32, 2, 2)

        self.skip_res1 = Res2DBlock(32, 32)
        self.skip_res2 = Res2DBlock(64, 64)

    def forward(self, x):
        skip_x1 = self.skip_res1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)

        skip_x2 = self.skip_res2(x)
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)

        x = self.mid_res(x)

        x = self.decoder_res2(x)
        x = self.decoder_upsample2(x)
        x = x + skip_x2

        x = self.decoder_res1(x)
        x = self.decoder_upsample1(x)
        x = x + skip_x1

        return x


@BACKBONES.register_module()
class P2PNet(BaseBackbone):

    def __init__(self, input_channels, output_channels):
        super(P2PNet, self).__init__()
        self.output_channels = output_channels

        self.front_layers = nn.Sequential(
            Basic2DBlock(input_channels, 16, 7),
            Res2DBlock(16, 32),
        )

        self.encoder_decoder = EncoderDecorder()

        self.output_layer = nn.Conv2d(
            32, output_channels, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def forward(self, x):
        x = self.front_layers(x)
        x = self.encoder_decoder(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)


@BACKBONES.register_module()
class CenterNet(BaseBackbone):

    def __init__(self,
                 input_channels,
                 output_channels,
                 mid_channels=32,
                 head_conv=32):
        super(CenterNet, self).__init__()
        self.output_channels = output_channels

        self.front_layers = nn.Sequential(
            Basic2DBlock(input_channels, mid_channels // 2, 7),
            Res2DBlock(mid_channels // 2, mid_channels),
        )

        self.encoder_decoder = EncoderDecorder()

        self.output_hm = nn.Sequential(
            nn.Conv2d(32, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, output_channels, kernel_size=1, padding=0))

        self.output_size = nn.Sequential(
            nn.Conv2d(32, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, kernel_size=1, padding=0, bias=True))

        self._initialize_weights()

    def forward(self, x):
        x, _ = torch.max(x, dim=4)  # max-pooling along z-axis
        x = self.front_layers(x)
        x = self.encoder_decoder(x)
        hm, size = self.output_hm(x), self.output_size(x)
        return hm, size

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)


@BACKBONES.register_module()
class WeightNet(BaseBackbone):

    def __init__(self, voxels_per_axis, num_joints):
        super(WeightNet, self).__init__()
        self.voxels_per_axis = voxels_per_axis
        self.num_joints = num_joints
        self.channel_num_per_joint = 64
        self.heatmap_feature_net = nn.Sequential(
            nn.Conv2d(1, self.channel_num_per_joint, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel_num_per_joint), nn.MaxPool2d(2),
            nn.ReLU(inplace=True))

        self.output = nn.Sequential(
            nn.Linear(self.channel_num_per_joint, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        # x: [3, num_people, num_channels, height, width]
        # Modifying to x: [batch_size, 3, num_channels, height, width]
        x = torch.flatten(x, 0, 1)
        batch_size = x.shape[0]
        num_joints = self.num_joints
        x = x.view(batch_size * num_joints, 1, self.voxels_per_axis[0],
                   self.voxels_per_axis[1])

        x = self.heatmap_feature_net(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(batch_size * num_joints, -1)
        x = self.output(x)
        x = x.view(batch_size, num_joints, 1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
