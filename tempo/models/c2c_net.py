# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmpose.models import BACKBONES
from .base_backbone import BaseBackbone


class Basic1DBlock(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d')):
        super(Basic1DBlock, self).__init__()

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


class Res1DBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d')):
        super(Res1DBlock, self).__init__()

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


class Pool1DBlock(nn.Module):

    def __init__(self, pool_size):
        super(Pool1DBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        return F.max_pool1d(
            x, kernel_size=self.pool_size, stride=self.pool_size)


class Upsample1DBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample1DBlock, self).__init__()
        assert (kernel_size == 2)
        assert (stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose1d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                output_padding=0), nn.BatchNorm1d(out_planes), nn.ReLU(True))

    def forward(self, x):
        return self.block(x)


class EncoderDecorder(nn.Module):

    def __init__(self):
        super(EncoderDecorder, self).__init__()

        self.encoder_pool1 = Pool1DBlock(2)
        self.encoder_res1 = Res1DBlock(32, 64)
        self.encoder_pool2 = Pool1DBlock(2)
        self.encoder_res2 = Res1DBlock(64, 128)

        self.mid_res = Res1DBlock(128, 128)

        self.decoder_res2 = Res1DBlock(128, 128)
        self.decoder_upsample2 = Upsample1DBlock(128, 64, 2, 2)
        self.decoder_res1 = Res1DBlock(64, 64)
        self.decoder_upsample1 = Upsample1DBlock(64, 32, 2, 2)

        self.skip_res1 = Res1DBlock(32, 32)
        self.skip_res2 = Res1DBlock(64, 64)

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
class C2CNet(BaseBackbone):

    def __init__(self, input_channels, output_channels, head_conv=32):
        super(C2CNet, self).__init__()
        self.output_channels = output_channels

        self.front_layers = nn.Sequential(
            Basic1DBlock(input_channels, 16, 7),
            Res1DBlock(16, head_conv),
        )

        self.encoder_decoder = EncoderDecorder()

        self.output_hm = nn.Conv1d(
            head_conv, output_channels, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def forward(self, x):
        x = self.front_layers(x)
        x = self.encoder_decoder(x)
        hm = self.output_hm(x)
        return hm

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose1d):
                # nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
