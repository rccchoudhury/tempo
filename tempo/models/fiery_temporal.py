# Copyright (c) OpenMMLab. All rights reserved.
# Code taken from FIERY:
# https://github.com/wayveai/fiery

import pdb
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.models.builder import BACKBONES

from .base_backbone import BaseBackbone
from .center_net import P2PNet


class ConvBlock(nn.Module):
    """2D convolution followed by.

    - an optional normalisation (batch norm or instance norm)
    - an optional activation (ReLU, LeakyReLU, or tanh)
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=3,
        stride=1,
        norm='bn',
        activation='relu',
        bias=False,
        transpose=False,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d if not transpose else partial(
            nn.ConvTranspose2d, output_padding=1)
        self.conv = self.conv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding,
            bias=bias)

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            raise ValueError('Invalid norm {}'.format(norm))

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh(inplace=True)
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError('Invalid activation {}'.format(activation))

    def forward(self, x):
        x = self.conv(x)

        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LayerNorm2d(nn.LayerNorm):
    r""" LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        if x.is_contiguous():
            # still faster than going to alternate implementation
            # call contiguous at the end, because otherwise the rest of the model is computed in channels-last
            return F.layer_norm(
                x.permute(0, 2, 3, 1), self.normalized_shape, self.weight,
                self.bias, self.eps).permute(0, 3, 1, 2).contiguous()
        elif x.is_contiguous(memory_format=torch.channels_last):
            x = x.permute(0, 2, 3, 1)
            # trick nvfuser into picking up layer norm, even though it's a single op
            # it's a slight pessimization (~.2%) if nvfuser is not enabled
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                             self.eps) * 1.
            return x.permute(0, 3, 1, 2)
        else:
            s, u = torch.var_mean(x, dim=1, unbiased=False, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x


class SpatialGRU(nn.Module):
    """A GRU cell that takes an input tensor [BxTxCxHxW] and an optional
    previous state and passes a convolutional gated recurrent unit over the
    data."""

    def __init__(self,
                 input_size,
                 hidden_size,
                 gru_bias_init=0.0,
                 norm='bn',
                 activation='relu'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_bias_init = gru_bias_init
        # Specify normalized shape as (H, W, C)since the layer norm
        # takes care of it ... i thiink..?
        #self.layer_norm2d = LayerNorm2d((64, 64, hidden_size), eps=1e-6)
        self.conv_update = nn.Conv2d(
            input_size + hidden_size,
            hidden_size,
            kernel_size=3,
            bias=True,
            padding=1)
        self.conv_reset = nn.Conv2d(
            input_size + hidden_size,
            hidden_size,
            kernel_size=3,
            bias=True,
            padding=1)

        self.conv_state_tilde = ConvBlock(
            input_size + hidden_size,
            hidden_size,
            kernel_size=3,
            bias=False,
            norm=norm,
            activation=activation)

    def forward(self, x, state=None, mode='bilinear'):
        # pylint: disable=unused-argument, arguments-differ
        # Check size
        assert len(x.size()) == 5, 'Input tensor must be BxTxCxHxW.'
        b, timesteps, c, h, w = x.size()
        assert c == self.input_size, f'feature sizes must match, got input {c} for layer with size {self.input_size}'

        # recurrent layers
        rnn_output = []
        rnn_state = torch.zeros(
            b, self.hidden_size, h, w,
            device=x.device) if state is None else state
        for t in range(timesteps):
            x_t = x[:, t]
            #rnn_state = self.layer_norm2d(rnn_state)
            # propagate rnn state
            rnn_state = self.gru_cell(x_t, rnn_state)
            rnn_output.append(rnn_state)

        # reshape rnn output to batch tensor
        return torch.stack(rnn_output, dim=1)

    def gru_cell(self, x, state):
        # Compute gates
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = self.conv_update(x_and_state)
        reset_gate = self.conv_reset(x_and_state)
        # Add bias to initialise gate as close to identity function
        update_gate = torch.sigmoid(update_gate + self.gru_bias_init)
        reset_gate = torch.sigmoid(reset_gate + self.gru_bias_init)

        # Compute proposal state, activation is defined in norm_act_config (can be tanh, ReLU etc)
        state_tilde = self.conv_state_tilde(
            torch.cat([x, (1.0 - reset_gate) * state], dim=1))

        output = (1.0 - update_gate) * state + update_gate * state_tilde
        return output


class CausalConv3d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(2, 3, 3),
                 dilation=(1, 1, 1),
                 bias=False):
        super().__init__()
        assert len(kernel_size) == 3, 'kernel_size must be a 3-tuple.'
        time_pad = (kernel_size[0] - 1) * dilation[0]
        height_pad = ((kernel_size[1] - 1) * dilation[1]) // 2
        width_pad = ((kernel_size[2] - 1) * dilation[2]) // 2

        # Pad temporally on the left
        self.pad = nn.ConstantPad3d(
            padding=(width_pad, width_pad, height_pad, height_pad, time_pad,
                     0),
            value=0)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            stride=1,
            padding=0,
            bias=bias)
        self.norm = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, *inputs):
        (x, ) = inputs
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class CausalMaxPool3d(nn.Module):

    def __init__(self, kernel_size=(2, 3, 3)):
        super().__init__()
        assert len(kernel_size) == 3, 'kernel_size must be a 3-tuple.'
        time_pad = kernel_size[0] - 1
        height_pad = (kernel_size[1] - 1) // 2
        width_pad = (kernel_size[2] - 1) // 2

        # Pad temporally on the left
        self.pad = nn.ConstantPad3d(
            padding=(width_pad, width_pad, height_pad, height_pad, time_pad,
                     0),
            value=0)
        self.max_pool = nn.MaxPool3d(kernel_size, stride=1)

    def forward(self, *inputs):
        (x, ) = inputs
        x = self.pad(x)
        x = self.max_pool(x)
        return x


def conv_1x1x1_norm_activated(in_channels, out_channels):
    """1x1x1 3D convolution, normalization and activation layer."""
    return nn.Sequential(
        OrderedDict([
            ('conv',
             nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)),
            ('norm', nn.BatchNorm3d(out_channels)),
            ('activation', nn.ReLU(inplace=True)),
        ]))


class Bottleneck3D(nn.Module):
    """Defines a bottleneck module with a residual connection."""

    def __init__(self,
                 in_channels,
                 out_channels=None,
                 kernel_size=(2, 3, 3),
                 dilation=(1, 1, 1)):
        super().__init__()
        bottleneck_channels = in_channels // 2
        out_channels = out_channels or in_channels

        self.layers = nn.Sequential(
            OrderedDict([
                # First projection with 1x1 kernel
                ('conv_down_project',
                 conv_1x1x1_norm_activated(in_channels, bottleneck_channels)),
                # Second conv block
                (
                    'conv',
                    CausalConv3d(
                        bottleneck_channels,
                        bottleneck_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        bias=False,
                    ),
                ),
                # Final projection with 1x1 kernel
                ('conv_up_project',
                 conv_1x1x1_norm_activated(bottleneck_channels, out_channels)),
            ]))

        if out_channels != in_channels:
            self.projection = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.projection = None

    def forward(self, *args):
        (x, ) = args
        x_residual = self.layers(x)
        x_features = self.projection(x) if self.projection is not None else x
        return x_residual + x_features


class PyramidSpatioTemporalPooling(nn.Module):
    """Spatio-temporal pyramid pooling.

    Performs 3D average pooling followed by 1x1x1 convolution to reduce the
    number of channels and upsampling. Setting contains a list of kernel_size:
    usually it is [(2, h, w), (2, h//2, w//2), (2, h//4, w//4)]
    """

    def __init__(self, in_channels, reduction_channels, pool_sizes):
        super().__init__()
        self.features = []
        for pool_size in pool_sizes:
            assert pool_size[0] == 2, (
                'Time kernel should be 2 as PyTorch raises an error when'
                'padding with more than half the kernel size')
            stride = (1, *pool_size[1:])
            padding = (pool_size[0] - 1, 0, 0)
            self.features.append(
                nn.Sequential(
                    OrderedDict([
                        # Pad the input tensor but do not take into account zero padding into the average.
                        (
                            'avgpool',
                            torch.nn.AvgPool3d(
                                kernel_size=pool_size,
                                stride=stride,
                                padding=padding,
                                count_include_pad=False),
                        ),
                        ('conv_bn_relu',
                         conv_1x1x1_norm_activated(in_channels,
                                                   reduction_channels)),
                    ])))
        self.features = nn.ModuleList(self.features)

    def forward(self, *inputs):
        (x, ) = inputs
        b, _, t, h, w = x.shape
        # Do not include current tensor when concatenating
        out = []
        for f in self.features:
            # Remove unnecessary padded values (time dimension) on the right
            x_pool = f(x)[:, :, :-1].contiguous()
            c = x_pool.shape[1]
            x_pool = nn.functional.interpolate(
                x_pool.view(b * t, c, *x_pool.shape[-2:]), (h, w),
                mode='bilinear',
                align_corners=False)
            x_pool = x_pool.view(b, c, t, h, w)
            out.append(x_pool)
        out = torch.cat(out, 1)
        return out


class TemporalBlock(nn.Module):
    """Temporal block with the following layers:

    - 2x3x3, 1x3x3, spatio-temporal pyramid pooling
    - dropout
    - skip connection.
    """

    def __init__(self,
                 in_channels,
                 out_channels=None,
                 use_pyramid_pooling=False,
                 pool_sizes=None):
        super().__init__()
        self.in_channels = in_channels
        self.half_channels = in_channels // 2
        self.out_channels = out_channels or self.in_channels
        self.kernels = [(2, 3, 3), (1, 3, 3)]

        # Flag for spatio-temporal pyramid pooling
        self.use_pyramid_pooling = use_pyramid_pooling

        # 3 convolution paths: 2x3x3, 1x3x3, 1x1x1
        self.convolution_paths = []
        for kernel_size in self.kernels:
            self.convolution_paths.append(
                nn.Sequential(
                    conv_1x1x1_norm_activated(self.in_channels,
                                              self.half_channels),
                    CausalConv3d(
                        self.half_channels,
                        self.half_channels,
                        kernel_size=kernel_size),
                ))
        self.convolution_paths.append(
            conv_1x1x1_norm_activated(self.in_channels, self.half_channels))
        self.convolution_paths = nn.ModuleList(self.convolution_paths)

        agg_in_channels = len(self.convolution_paths) * self.half_channels

        if self.use_pyramid_pooling:
            assert pool_sizes is not None, 'setting must contain the list of kernel_size, but is None.'
            reduction_channels = self.in_channels // 3
            self.pyramid_pooling = PyramidSpatioTemporalPooling(
                self.in_channels, reduction_channels, pool_sizes)
            agg_in_channels += len(pool_sizes) * reduction_channels

        # Feature aggregation
        self.aggregation = nn.Sequential(
            conv_1x1x1_norm_activated(agg_in_channels, self.out_channels), )

        if self.out_channels != self.in_channels:
            self.projection = nn.Sequential(
                nn.Conv3d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=1,
                    bias=False),
                nn.BatchNorm3d(self.out_channels),
            )
        else:
            self.projection = None

    def forward(self, *inputs):
        (x, ) = inputs
        x_paths = []
        for conv in self.convolution_paths:
            x_paths.append(conv(x))
        x_residual = torch.cat(x_paths, dim=1)
        if self.use_pyramid_pooling:
            x_pool = self.pyramid_pooling(x)
            x_residual = torch.cat([x_residual, x_pool], dim=1)
        x_residual = self.aggregation(x_residual)

        if self.out_channels != self.in_channels:
            x = self.projection(x)
        x = x + x_residual
        return x


class Interpolate(nn.Module):

    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self._interpolate = nn.functional.interpolate
        self._scale_factor = scale_factor

    # pylint: disable=arguments-differ
    def forward(self, x):
        return self._interpolate(
            x,
            scale_factor=self._scale_factor,
            mode='bilinear',
            align_corners=False)


class Bottleneck(nn.Module):
    """Defines a bottleneck module with a residual connection."""

    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=3,
        dilation=1,
        groups=1,
        upsample=False,
        downsample=False,
        dropout=0.0,
    ):
        super().__init__()
        self._downsample = downsample
        bottleneck_channels = int(in_channels / 2)
        out_channels = out_channels or in_channels
        padding_size = ((kernel_size - 1) * dilation + 1) // 2

        # Define the main conv operation
        assert dilation == 1
        if upsample:
            assert not downsample, 'downsample and upsample not possible simultaneously.'
            bottleneck_conv = nn.ConvTranspose2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=kernel_size,
                bias=False,
                dilation=1,
                stride=2,
                output_padding=padding_size,
                padding=padding_size,
                groups=groups,
            )
        elif downsample:
            bottleneck_conv = nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=kernel_size,
                bias=False,
                dilation=dilation,
                stride=2,
                padding=padding_size,
                groups=groups,
            )
        else:
            bottleneck_conv = nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=kernel_size,
                bias=False,
                dilation=dilation,
                padding=padding_size,
                groups=groups,
            )

        self.layers = nn.Sequential(
            OrderedDict([
                # First projection with 1x1 kernel
                ('conv_down_project',
                 nn.Conv2d(
                     in_channels,
                     bottleneck_channels,
                     kernel_size=1,
                     bias=False)),
                ('abn_down_project',
                 nn.Sequential(
                     nn.BatchNorm2d(bottleneck_channels),
                     nn.ReLU(inplace=True))),
                # Second conv block
                ('conv', bottleneck_conv),
                ('abn',
                 nn.Sequential(
                     nn.BatchNorm2d(bottleneck_channels),
                     nn.ReLU(inplace=True))),
                # Final projection with 1x1 kernel
                ('conv_up_project',
                 nn.Conv2d(
                     bottleneck_channels,
                     out_channels,
                     kernel_size=1,
                     bias=False)),
                ('abn_up_project',
                 nn.Sequential(
                     nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))),
                # Regulariser
                ('dropout', nn.Dropout2d(p=dropout)),
            ]))

        if out_channels == in_channels and not downsample and not upsample:
            self.projection = None
        else:
            projection = OrderedDict()
            if upsample:
                projection.update(
                    {'upsample_skip_proj': Interpolate(scale_factor=2)})
            elif downsample:
                projection.update({
                    'upsample_skip_proj':
                    nn.MaxPool2d(kernel_size=2, stride=2)
                })
            projection.update({
                'conv_skip_proj':
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, bias=False),
                'bn_skip_proj':
                nn.BatchNorm2d(out_channels),
            })
            self.projection = nn.Sequential(projection)

    # pylint: disable=arguments-differ
    def forward(self, *args):
        (x, ) = args
        x_residual = self.layers(x)
        if self.projection is not None:
            if self._downsample:
                # pad h/w dimensions if they are odd to prevent shape mismatch with residual layer
                x = nn.functional.pad(
                    x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), value=0)
            return x_residual + self.projection(x)
        return x_residual + x


class FieryTemporalModel(nn.Module):

    def __init__(self,
                 in_channels,
                 receptive_field,
                 input_shape,
                 start_out_channels=64,
                 extra_in_channels=0,
                 n_spatial_layers_between_temporal_layers=0,
                 use_pyramid_pooling=True):
        """
        :param in_channels: number of input channels
        :param receptive_field: # of timesteps taken in.
        """
        super().__init__()
        self.receptive_field = receptive_field
        self.input_shape = input_shape
        n_temporal_layers = receptive_field - 1

        h, w = input_shape
        modules = []

        block_in_channels = in_channels
        block_out_channels = start_out_channels

        for _ in range(n_temporal_layers):
            if use_pyramid_pooling:
                use_pyramid_pooling = True
                pool_sizes = [(2, h, w)]
            else:
                use_pyramid_pooling = False
                pool_sizes = None
            temporal = TemporalBlock(
                block_in_channels,
                block_out_channels,
                use_pyramid_pooling=use_pyramid_pooling,
                pool_sizes=pool_sizes,
            )
            spatial = [
                Bottleneck3D(
                    block_out_channels,
                    block_out_channels,
                    kernel_size=(1, 3, 3))
                for _ in range(n_spatial_layers_between_temporal_layers)
            ]
            temporal_spatial_layers = nn.Sequential(temporal, *spatial)
            modules.extend(temporal_spatial_layers)

            block_in_channels = block_out_channels
            block_out_channels += extra_in_channels

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        #assert len(x.shape) == 6
        assert x.shape[-2:] == torch.Size(self.input_shape)
        # Input is shape (batch, time, C, X, Y, Z)
        #x, _ = torch.max(x, dim=5)  # max-pooling along z-axis
        # Reshape input tensor to (batch, C, time, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.model(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x_init_shape = x.shape
        # X shape is [b, T,  C, H, H]
        x = x.view(-1, *x_init_shape[2:])

        x = x.view(x_init_shape)

        return x


class TemporalModelIdentity(nn.Module):

    def __init__(self, in_channels, receptive_field):
        super().__init__()
        self.receptive_field = receptive_field
        self.out_channels = in_channels

    def forward(self, x):
        return x[:, (self.receptive_field - 1):]


@BACKBONES.register_module()
class TemporalPoseModel(BaseBackbone):

    def __init__(self, in_channels, receptive_field, out_channels,
                 input_shape):
        super().__init__()
        self.receptive_field = receptive_field
        self.fiery_temporal = FieryTemporalModel(
            in_channels,
            receptive_field,
            input_shape,
            start_out_channels=in_channels,
        )
        self.output_head = P2PNet(
            input_channels=in_channels,
            output_channels=out_channels,
        )

    def forward(self, x):
        """Takes input of dimension.

        (batch * 3, time, C, X, Y, Z)
        """
        assert len(x.shape) == 5
        batch_size, input_time, channels, X, Y = x.shape
        temporal_rep = self.fiery_temporal(x)
        temporal_rep = temporal_rep.view(-1, channels, X, Y)
        output = self.output_head(temporal_rep)
        _, num_joints, X, Y = output.shape
        output = output.view(batch_size, input_time, num_joints, X, Y)
        temporal_rep = temporal_rep.view(batch_size, input_time, channels, X,
                                         Y)
        temporal_rep = temporal_rep[:, (self.receptive_field - 1):]
        temporal_rep = temporal_rep.squeeze(1)

        return output, temporal_rep


@BACKBONES.register_module()
class FuturePrediction(BaseBackbone):

    def __init__(self,
                 in_channels,
                 out_channels,
                 latent_dim,
                 n_gru_blocks=3,
                 n_res_layers=3):
        super().__init__()
        self.n_gru_blocks = n_gru_blocks
        self.out_channels = out_channels
        # Convolutional recurrent model with z_t as an initial hidden state and inputs the sample
        # from the probabilistic model. The architecture of the model is:
        # [Spatial GRU - [Bottleneck] x n_res_layers] x n_gru_blocks
        self.spatial_grus = []
        self.res_blocks = []

        for i in range(self.n_gru_blocks):
            gru_in_channels = latent_dim if i == 0 else in_channels
            self.spatial_grus.append(SpatialGRU(gru_in_channels, in_channels))
            self.res_blocks.append(
                torch.nn.Sequential(
                    *[Bottleneck(in_channels) for _ in range(n_res_layers)]))

        self.spatial_grus = torch.nn.ModuleList(self.spatial_grus)
        self.res_blocks = torch.nn.ModuleList(self.res_blocks)
        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, hidden_state):
        # x has shape (b, n_future, c, h, w), hidden_state (b, c, h, w)
        for i in range(self.n_gru_blocks):
            x = self.spatial_grus[i](x, hidden_state)
            b, n_future, c, h, w = x.shape

            x = self.res_blocks[i](x.view(b * n_future, c, h, w))
            x = x.view(b, n_future, c, h, w)
        x = x.view(-1, c, h, w)
        x = self.conv_out(x)
        x = x.view(b, n_future, self.out_channels, h, w)

        return x
