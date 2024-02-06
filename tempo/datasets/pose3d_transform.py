# Copyright (c) OpenMMLab. All rights reserved.
import copy
import pdb
import random

import mmcv
import numpy as np
import torch
from mmcv.utils import build_from_cfg

from mmpose.core.camera import CAMERAS
from mmpose.core.post_processing import (affine_transform, fliplr_regression,
                                         get_affine_transform)
from mmpose.datasets.builder import PIPELINES


@PIPELINES.register_module()
class GenerateVoxelMultiDHeatmapTarget:
    """Generate the target 3d heatmap.

    Required keys: 'joints_3d', 'joints_3d_visible', 'ann_info_3d'.
    Modified keys: 'target', and 'target_weight'.

    Can also generate 2d and 1d heatmaps. Will separate the two
    after I confirm the detector works.

    Args:
        sigma: Sigma of heatmap gaussian (mm).
        joint_indices (list): Indices of joints used for heatmap generation.
            If None (default) is given, all joints will be used.
    """

    def __init__(self, sigma=200.0, joint_indices=None, max_num_people=10):
        # joint indices is the root idx in usage.
        self.sigma = sigma  # mm
        self.joint_indices = joint_indices
        self.max_num_people = max_num_people

    def __call__(self, results):
        """Generate the target heatmap."""
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        cfg = results['ann_info']

        num_people = len(joints_3d)
        num_joints = joints_3d[0].shape[0]

        if self.joint_indices is not None:
            num_joints = len(self.joint_indices)
            joint_indices = self.joint_indices
        else:
            joint_indices = list(range(num_joints))

        space_size = cfg['space_size']
        space_center = cfg['space_center']
        cube_size = cfg['cube_size']
        grids_x = np.linspace(-space_size[0] / 2, space_size[0] / 2,
                              cube_size[0]) + space_center[0]
        grids_y = np.linspace(-space_size[1] / 2, space_size[1] / 2,
                              cube_size[1]) + space_center[1]
        grids_z = np.linspace(-space_size[2] / 2, space_size[2] / 2,
                              cube_size[2]) + space_center[2]
        #voxel_size = space_size / (cube_size - 1)

        target = np.zeros(
            (num_joints, cube_size[0], cube_size[1], cube_size[2]),
            dtype=np.float32)
        target_2d = np.zeros((num_joints, cube_size[0], cube_size[1]),
                             dtype=np.float32)
        target_1d = np.zeros((num_joints, self.max_num_people, cube_size[2]),
                             dtype=np.float32)

        for n in range(num_people):
            for idx, joint_id in enumerate(joint_indices):
                assert joints_3d.shape[2] == 3
                #pdb.set_trace()

                mu_x = np.mean(joints_3d[n][joint_id, 0])
                mu_y = np.mean(joints_3d[n][joint_id, 1])
                mu_z = np.mean(joints_3d[n][joint_id, 2])
                vis = np.mean(joints_3d_visible[n][joint_id, 0])
                if vis < 1:
                    continue
                i_x = [
                    np.searchsorted(grids_x, mu_x - 3 * self.sigma),
                    np.searchsorted(grids_x, mu_x + 3 * self.sigma, 'right')
                ]
                i_y = [
                    np.searchsorted(grids_y, mu_y - 3 * self.sigma),
                    np.searchsorted(grids_y, mu_y + 3 * self.sigma, 'right')
                ]
                i_z = [
                    np.searchsorted(grids_z, mu_z - 3 * self.sigma),
                    np.searchsorted(grids_z, mu_z + 3 * self.sigma, 'right')
                ]
                if i_x[0] >= i_x[1] or i_y[0] >= i_y[1] or i_z[0] >= i_z[1]:
                    continue
                kernel_xs, kernel_ys, kernel_zs = np.meshgrid(
                    grids_x[i_x[0]:i_x[1]],
                    grids_y[i_y[0]:i_y[1]],
                    grids_z[i_z[0]:i_z[1]],
                    indexing='ij')
                g = np.exp(-((kernel_xs - mu_x)**2 + (kernel_ys - mu_y)**2 +
                             (kernel_zs - mu_z)**2) / (2 * self.sigma**2))
                target[idx, i_x[0]:i_x[1], i_y[0]:i_y[1], i_z[0]:i_z[1]] \
                    = np.maximum(target[idx, i_x[0]:i_x[1],
                                 i_y[0]:i_y[1], i_z[0]:i_z[1]], g)
                # Below is lifted from F-VP.
                # generate 2D target
                kernel_xs, kernel_ys = np.meshgrid(
                    grids_x[i_x[0]:i_x[1]],
                    grids_y[i_y[0]:i_y[1]],
                    indexing='ij')
                g = np.exp(-((kernel_xs - mu_x)**2 + (kernel_ys - mu_y)**2) /
                           (2 * self.sigma**2))
                target_2d[idx, i_x[0]:i_x[1], i_y[0]:i_y[1]] = np.maximum(
                    target_2d[idx, i_x[0]:i_x[1], i_y[0]:i_y[1]], g)

                # generate 1D target
                kernel_zs = grids_z[i_z[0]:i_z[1]]
                g = np.exp(-(kernel_zs - mu_z)**2 / (2 * self.sigma**2))
                target_1d[idx, n, i_z[0]:i_z[1]] = np.maximum(
                    target_1d[idx, n, i_z[0]:i_z[1]], g)

        target = np.clip(target, 0, 1)
        target_2d = np.clip(target_2d, 0, 1)
        target_1d = np.clip(target_1d, 0, 1)

        if target.shape[0] == 1:
            target = target[0]
            target_2d = target_2d[0]
            target_1d = target_1d[0]

        results['targets_3d'] = target
        results['targets_2d'] = target_2d
        results['targets_1d'] = target_1d

        return results

@PIPELINES.register_module()
class GenerateBBoxTarget:
    """Generate target bounding box.

    Required keys: 'joints_3d', 'joints_3d_vis', 'ann_info'
    Modified keys: 'bbox3d_index', 'bbox3d_offset', 'bbox3d'

    Args:
        slack: amount of slack to provide for bounding box
        joint_indices: root index for joint.
        max_num_people: maximum number of people.
    """

    def __init__(self, slack=200.0, joint_indices=None, max_num_people=10):
        self.slack = slack
        self.joint_indices = joint_indices
        self.max_num_people = max_num_people

    def __call__(self, results):
        joints_3d = results['joints_3d']
        joints_3d_vis = results['joints_3d_visible']
        #pdb.set_trace()
        cfg = results['ann_info']

        space_size = np.array(cfg['space_size'])
        space_center = np.array(cfg['space_center'])
        cube_size = np.array(cfg['cube_size'])

        voxel_size = space_size / (cube_size - 1)
        individual_space_size = np.array(cfg['sub_space_size'])

        if self.joint_indices is not None:
            num_joints = len(self.joint_indices)
            joint_indices = self.joint_indices
        else:
            joint_indices = list(range(num_joints))
        #pdb.set_trace()
        num_people = results['num_persons']
        target_index = np.zeros((num_joints, self.max_num_people))
        target_bbox = np.zeros((num_joints, self.max_num_people, 2))
        target_offset = np.zeros((num_joints, self.max_num_people, 2))
        #pdb.set_trace()
        for n in range(num_people):
            for i, joint_id in enumerate(joint_indices):
                idx = joints_3d_vis > 0.1
                idx_n = idx[n].min(axis=-1)
                center_pos = joints_3d[n][joint_id]
                loc = (center_pos - space_center +
                       0.5 * space_size) / voxel_size
                target_index[i,
                             n] = (loc // 1)[0] * cube_size[0] + (loc // 1)[1]
                target_offset[i, n] = (loc % 1)[:2]
                target_bbox[i, n] = (
                    (2 * np.abs(center_pos - joints_3d[n][idx_n]).max(axis=0) +
                     self.slack) / individual_space_size)[:2]

        if target_index.shape[0] == 1:
            target_index = target_index[0]
            target_bbox = target_bbox[0]
            target_offset = target_offset[0]

        results['bbox3d_index'] = target_index
        results['bbox3d_offset'] = target_offset
        results['bbox3d'] = target_bbox

        return results