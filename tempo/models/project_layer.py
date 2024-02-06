# Copyright (c) OpenMMLab. All rights reserved.
import pdb
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
from scipy.optimize import linear_sum_assignment

from mmpose.core.camera import SimpleCameraTorch
from mmpose.core.post_processing.post_transforms import (
    affine_transform_torch, get_affine_transform)
from mmpose.models.utils.misc import torch_meshgrid_ij


class ProjectLayer(nn.Module):

    def __init__(self, image_size, heatmap_size, agg_method):
        """Project layer to get voxel feature. Adapted from
        https://github.com/microsoft/voxelpose-
        pytorch/blob/main/lib/models/project_layer.py.

        Args:
            image_size (int or list): input size of the 2D model
            heatmap_size (int or list): output size of the 2D model
        """
        super(ProjectLayer, self).__init__()
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.agg_method = agg_method
        if self.agg_method == 'learned':
            self.agg_layer = nn.Sequential(
                nn.Linear(5 * 32, 3 * 32), nn.ReLU(inplace=True),
                nn.Linear(3 * 32, 32), nn.Softmax(dim=-1))

        if isinstance(self.image_size, int):
            self.image_size = [self.image_size, self.image_size]
        if isinstance(self.heatmap_size, int):
            self.heatmap_size = [self.heatmap_size, self.heatmap_size]

    def compute_grid(self, box_size, box_center, num_bins, device=None):
        if isinstance(box_size, int) or isinstance(box_size, float):
            box_size = [box_size, box_size, box_size]
        if isinstance(num_bins, int):
            num_bins = [num_bins, num_bins, num_bins]

        grid_1D_x = torch.linspace(
            -box_size[0] / 2, box_size[0] / 2, num_bins[0], device=device)
        grid_1D_y = torch.linspace(
            -box_size[1] / 2, box_size[1] / 2, num_bins[1], device=device)
        grid_1D_z = torch.linspace(
            -box_size[2] / 2, box_size[2] / 2, num_bins[2], device=device)
        grid_x, grid_y, grid_z = torch_meshgrid_ij(grid_1D_x + box_center[0],
                                                   grid_1D_y + box_center[1],
                                                   grid_1D_z + box_center[2])
        grid_x = grid_x.contiguous().view(-1, 1)
        grid_y = grid_y.contiguous().view(-1, 1)
        grid_z = grid_z.contiguous().view(-1, 1)
        grid = torch.cat([grid_x, grid_y, grid_z], dim=1)

        return grid

    @force_fp32(apply_to=('feature_maps', 'centers'))
    def get_voxel(self, feature_maps, grid_size, grid_center, cube_size,
                  centers, scales, cameras):
        device = feature_maps[0].device
        batch_size = feature_maps[0].shape[0]
        num_channels = feature_maps[0].shape[1]
        num_bins = cube_size[0] * cube_size[1] * cube_size[2]
        num_views = len(feature_maps)
        cubes = torch.zeros(
            batch_size, num_channels, 1, num_bins, num_views, device=device)
        w, h = self.heatmap_size
        grids = torch.zeros(batch_size, num_bins, 3, device=device)
        bounding = torch.zeros(
            batch_size, 1, 1, num_bins, num_views, device=device)

        for batch_idx in range(batch_size):
            if len(grid_center[0]) == 3 or grid_center[batch_idx][3] >= 0:
                if len(grid_center) == 1:
                    grid = self.compute_grid(
                        grid_size, grid_center[0], cube_size, device=device)
                else:
                    grid = self.compute_grid(
                        grid_size,
                        grid_center[batch_idx],
                        cube_size,
                        device=device)
                grids[batch_idx:batch_idx + 1] = grid

                for cam_idx in range(num_views):
                    center = centers[batch_idx][cam_idx]
                    scale = scales[batch_idx][cam_idx]

                    width, height = center * 2
                    trans = torch.as_tensor(
                        get_affine_transform(center, scale / 200.0, 0,
                                             self.image_size),
                        dtype=torch.float,
                        device=device)

                    cam_param = cameras[batch_idx][cam_idx].copy()
                    if 'intrinsics' in cam_param:
                        single_view_camera = FisheyeCameraTorch(
                            param=cam_param, device=device)
                    else:
                        single_view_camera = SimpleCameraTorch(
                            param=cam_param, device=device)
                    xy = single_view_camera.world_to_pixel(grid)

                    bounding[batch_idx, 0, 0, :, cam_idx] = (xy[:, 0] >= 0) & (
                        xy[:, 1] >= 0) & (xy[:, 0] < width) & (
                            xy[:, 1] < height)
                    xy = torch.clamp(xy, -1.0, max(width, height))
                    xy = affine_transform_torch(xy, trans)
                    xy = xy * torch.tensor(
                        [w, h], dtype=torch.float,
                        device=device) / torch.tensor(
                            self.image_size, dtype=torch.float, device=device)
                    sample_grid = xy / torch.tensor([w - 1, h - 1],
                                                    dtype=torch.float,
                                                    device=device) * 2.0 - 1.0
                    sample_grid = torch.clamp(
                        sample_grid.view(1, 1, num_bins, 2), -1.1, 1.1)

                    cubes[batch_idx:batch_idx + 1, :, :, :,
                          cam_idx] += F.grid_sample(
                              feature_maps[cam_idx][batch_idx:batch_idx +
                                                    1, :, :, :],
                              sample_grid,
                              align_corners=False)

        if self.agg_method == 'learned':
            cubes = torch.mul(cubes, bounding)
            cubes = cubes.squeeze(2)
            chunks = torch.chunk(cubes, chunks=32, dim=1)
            concatted_tensors = torch.cat(chunks, dim=-1)
            cubes = self.agg_layer(concatted_tensors)
            cubes = torch.transpose(cubes, 1, 2)
        elif self.agg_method == 'mean':
            cubes = torch.sum(
                torch.mul(cubes, bounding), dim=-1) / (
                    torch.sum(bounding, dim=-1) + 1e-6)
        else:
            raise NotImplementedError

        cubes = cubes.clamp(0.0, 1.0)
        cubes = cubes.view(batch_size, num_channels, cube_size[0],
                           cube_size[1], cube_size[2])
        return cubes, grids

    def forward(self, feature_maps, grid_size, grid_center, cube_size, centers,
                scales, cameras):
        cubes, grids = self.get_voxel(feature_maps, grid_size, grid_center,
                                      cube_size, centers, scales, cameras)
        return cubes, grids

class ProjectLayerWithMask(nn.Module):

    def __init__(self, image_size, heatmap_size, agg_method):
        """Project layer to get voxel feature. Adapted from
        https://github.com/microsoft/voxelpose-
        pytorch/blob/main/lib/models/project_layer.py.

        Args:
            image_size (int or list): input size of the 2D model
            heatmap_size (int or list): output size of the 2D model
        """
        super(ProjectLayerWithMask, self).__init__()
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.agg_method = agg_method

        if isinstance(self.image_size, int):
            self.image_size = [self.image_size, self.image_size]
        if isinstance(self.heatmap_size, int):
            self.heatmap_size = [self.heatmap_size, self.heatmap_size]

        self.fine_grid = None
        self.center_grid = None

    def set_space_size(self, space_center, space_size, sub_space_size,
                       sub_cube_size, device):
        self.whole_space_center = torch.tensor(space_center, device=device)
        self.whole_space_size = torch.tensor(space_size, device=device)
        self.ind_space_size = torch.tensor(sub_space_size, device=device)
        self.voxels_per_axis = torch.tensor(
            sub_cube_size, device=device, dtype=torch.int32)
        self.fine_voxels_per_axis = (self.whole_space_size /
                                     self.ind_space_size *
                                     (self.voxels_per_axis - 1)).int() + 1
        self.scale = (self.fine_voxels_per_axis.float() -
                      1) / self.whole_space_size
        self.bias = - self.ind_space_size / 2.0 / self.whole_space_size * (self.fine_voxels_per_axis - 1)\
                    - self.scale * (self.whole_space_center - self.whole_space_size / 2.0)
        self.num_bins = self.voxels_per_axis[0] * self.voxels_per_axis[
            1] * self.voxels_per_axis[2]
        self.num_fine_bins = self.fine_voxels_per_axis[
            0] * self.fine_voxels_per_axis[1] * self.fine_voxels_per_axis[2]

    def compute_grid(self, box_size, box_center, num_bins, device=None):
        if isinstance(box_size, int) or isinstance(box_size, float):
            box_size = [box_size, box_size, box_size]
        if isinstance(num_bins, int):
            num_bins = [num_bins, num_bins, num_bins]

        grid_1D_x = torch.linspace(
            -box_size[0] / 2, box_size[0] / 2, num_bins[0], device=device)
        grid_1D_y = torch.linspace(
            -box_size[1] / 2, box_size[1] / 2, num_bins[1], device=device)
        grid_1D_z = torch.linspace(
            -box_size[2] / 2, box_size[2] / 2, num_bins[2], device=device)
        grid_x, grid_y, grid_z = torch_meshgrid_ij(grid_1D_x + box_center[0],
                                                   grid_1D_y + box_center[1],
                                                   grid_1D_z + box_center[2])
        grid_x = grid_x.contiguous().view(-1, 1)
        grid_y = grid_y.contiguous().view(-1, 1)
        grid_z = grid_z.contiguous().view(-1, 1)
        grid = torch.cat([grid_x, grid_y, grid_z], dim=1)

        return grid

    def project_grid(self, feature_maps, grids, grid_center, cubes, start, end,
                     centers_tl, centers, scales, cameras):
        batch_size, num_channels = feature_maps[0].shape[:2]
        device = feature_maps[0].device
        w, h = self.heatmap_size

        for i in range(batch_size):
            if len(grid_center[0]) == 3 or grid_center[i][3] >= 0:
                grid = self.fine_grid
                grids[i:i + 1] = grid

                for c in range(len(feature_maps)):
                    center = centers[i][c]
                    scale = scales[i][c]

                    width, height = center * 2
                    trans = torch.as_tensor(
                        get_affine_transform(center, scale / 200.0, 0,
                                             self.image_size),
                        dtype=torch.float,
                        device=device)

                    cam_param = cameras[i][c].copy()
                    #pdb.set_trace()
                    if 'intrinsics' in cam_param:
                        single_view_camera = FisheyeCameraTorch(
                            param=cam_param, device=device)
                    else:
                        single_view_camera = SimpleCameraTorch(
                            param=cam_param, device=device)
                    xy = single_view_camera.world_to_pixel(grid)

                    xy = torch.clamp(xy, -1.0, max(width, height))
                    xy = affine_transform_torch(xy, trans)
                    xy = xy * torch.tensor(
                        [w, h], dtype=torch.float,
                        device=device) / torch.tensor(
                            self.image_size, dtype=torch.float, device=device)
                    sample_grid = xy / torch.tensor([w - 1, h - 1],
                                                    dtype=torch.float,
                                                    device=device) * 2.0 - 1.0
                    sample_grid = torch.clamp(
                        sample_grid.view(1, 1, self.num_fine_bins, 2), -1.1,
                        1.1)
                    sample_grid = sample_grid.view(
                        self.fine_voxels_per_axis[0],
                        self.fine_voxels_per_axis[1],
                        self.fine_voxels_per_axis[2], 2)
                    sample_grid = sample_grid[start[i, 0]:end[i, 0],
                                              start[i, 1]:end[i, 1],
                                              start[i, 2]:end[i, 2]].reshape(
                                                  1, 1, -1, 2)
                    try:
                        output = F.grid_sample(
                            feature_maps[c][i:i + 1, :, :, :],
                            sample_grid,
                            align_corners=False).view(num_channels, 1,
                                                      end[i, 0] - start[i, 0],
                                                      end[i, 1] - start[i, 1],
                                                      end[i, 2] - start[i, 2])
                    except RuntimeError:
                        pdb.set_trace()
                    cubes[i, :, :, start[i, 0] - centers_tl[i, 0]:end[i, 0] -
                          centers_tl[i, 0], start[i, 1] -
                          centers_tl[i, 1]:end[i, 1] - centers_tl[i, 1],
                          start[i, 2] - centers_tl[i, 2]:end[i, 2] -
                          centers_tl[i, 2], c] += output

        return cubes, grids

    def get_voxel(self, feature_maps, grid_center, centers, scales, cameras):
        device = feature_maps[0].device
        batch_size = feature_maps[0].shape[0]

        centers_tl = torch.round(grid_center[:, :3].float() * self.scale +
                                 self.bias).int()
        offset = centers_tl.float() / (
            self.fine_voxels_per_axis - 1
        ) * self.whole_space_size - self.whole_space_size / 2.0 + self.ind_space_size / 2.0

        # mask the feature volume outside the bounding box
        mask = ((1 - grid_center[:, 5:7]) / 2 *
                (self.voxels_per_axis[0:2] - 1)).int()
        mask[mask < 0] = 0
        # the vertical length of the bounding box is kept fixed as 2000mm
        mask = torch.cat([
            mask,
            torch.zeros((batch_size, 1), device=device, dtype=torch.int32)
        ],
                         dim=1)
        # compute the valid range to filter the outsider
        start = torch.where(centers_tl + mask >= 0, centers_tl + mask,
                            torch.zeros_like(centers_tl))
        end = torch.where(
            centers_tl + self.voxels_per_axis - mask <=
            self.fine_voxels_per_axis,
            centers_tl + self.voxels_per_axis - mask,
            self.fine_voxels_per_axis)
        #pdb.set_trace()
        end = torch.maximum(end, start + 1)

        if self.fine_grid is None:
            self.fine_grid = self.compute_grid(
                self.whole_space_size,
                self.whole_space_center,
                self.fine_voxels_per_axis,
                device=device)
            grid = self.compute_grid(
                self.ind_space_size,
                self.whole_space_center,
                self.voxels_per_axis,
                device=device)
            grid = grid.view(self.voxels_per_axis[0], self.voxels_per_axis[1],
                             self.voxels_per_axis[2], 3)
            self.center_grid = torch.stack([grid[:, :, 0, :2].reshape(-1, 2), grid[:, 0, :, ::2].reshape(-1, 2), \
                                            grid[0, :, :, 1:].reshape(-1, 2)])

        cube_size = self.voxels_per_axis
        num_channels = feature_maps[0].shape[1]

        cubes = torch.zeros(
            batch_size,
            num_channels,
            1,
            self.voxels_per_axis[0],
            self.voxels_per_axis[1],
            self.voxels_per_axis[2],
            len(feature_maps),
            device=device)

        grids = torch.zeros(batch_size, self.num_fine_bins, 3, device=device)
        cubes, grids = self.project_grid(feature_maps, grids, grid_center,
                                         cubes, start, end, centers_tl,
                                         centers, scales, cameras)

        if self.agg_method == 'learned':
            cubes = torch.mul(cubes, bounding)
            cubes = cubes.squeeze(2)
            chunks = torch.chunk(cubes, chunks=32, dim=1)
            concatted_tensors = torch.cat(chunks, dim=-1)
            cubes = self.agg_layer(concatted_tensors)
            cubes = torch.transpose(cubes, 1, 2)
        elif self.agg_method == 'mean':
            bounding = torch.ones_like(cubes)
            cubes = torch.sum(
                torch.mul(cubes, bounding), dim=-1) / (
                    torch.sum(bounding, dim=-1) + 1e-6)
        else:
            raise NotImplementedError

        cubes = cubes.clamp(0.0, 1.0)
        cubes = cubes.view(batch_size, num_channels, cube_size[0],
                           cube_size[1], cube_size[2])
        return cubes, offset

    def forward(self, feature_maps, grid_center, space_center, space_size,
                sub_space_size, sub_cube_size, centers, scales, cameras):
        device = feature_maps[0].device
        self.set_space_size(space_center, space_size, sub_space_size,
                            sub_cube_size, device)
        cubes, offset = self.get_voxel(feature_maps, grid_center, centers,
                                       scales, cameras)

        return cubes, offset