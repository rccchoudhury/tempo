import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from mmpose.models import builder, POSENETS
from mmpose.models.detectors.base import BasePose

from .model_utils import nms_by_max_pool
from .project_layer import ProjectLayer
from ..tracker.sort import Sort

@POSENETS.register_module()
class RNNVoxelCenterDetector(BasePose):
    """Detect human center by 3D CNN on voxels.

    Please refer to the
    `paper <https://arxiv.org/abs/2004.06239>` for details.
    Args:
        image_size (list): input size of the 2D model.
        heatmap_size (list): output size of the 2D model.
        space_size (list): Size of the 3D space.
        cube_size (list): Size of the input volume to the 3D CNN.
        space_center (list): Coordinate of the center of the 3D space.
        center_net (ConfigDict): Dictionary to construct the center net.
        center_head (ConfigDict): Dictionary to construct the center head.
        train_cfg (ConfigDict): Config for training. Default: None.
        test_cfg (ConfigDict): Config for testing. Default: None.
    """

    def __init__(self,
                 image_size,
                 heatmap_size,
                 center_net,
                 center_net_1d,
                 center_head,
                 num_features,
                 max_num_people=10,
                 horizon=1,
                 input_time=1,
                 train_cfg=None,
                 test_cfg=None,
                 use_gt=False,
                 do_forecasting=False,
                 agg_method='mean'):
        super(RNNVoxelCenterDetector, self).__init__()
        self.project_layer = ProjectLayer(image_size, heatmap_size, agg_method)
        self.center_net = builder.build_backbone(center_net)
        self.center_net1d = builder.build_backbone(center_net_1d)
        self.center_head = builder.build_head(center_head)

        self.horizon = horizon
        self.input_time = input_time
        self.nms_thresh = 0.1
        self.num_features = num_features
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.use_gt = use_gt
        self.max_num_people = max_num_people
        self.do_forecasting = do_forecasting
        self.tracker = Sort(max_age=100, min_hits=5, iou_threshold=0.1)

    def create_tracker_gt(self, prev_candidates, cur_candidates, test=False):
        """Uses the center candidates to create GT assignment matrix for the
        tracker.

        Args:
            center_candidates (torch.Tensor): [N, T, max_num_people, 7]
        Output:
            gt_assignment (torch.Tensor): [N, T-1, max_num_people, max_num_people]

        would love to write a test for this somewhere sadgely.
        """
        batch_size = cur_candidates.shape[0]
        gt_assignment = cur_candidates.new_zeros(cur_candidates.shape[0],
                                                 self.max_num_people,
                                                 self.max_num_people + 1)

        cost_mat = torch.cdist(
            prev_candidates[:, :, :3],
            cur_candidates[:, :, :3]).detach().cpu().numpy()

        new_cands = torch.zeros_like(cur_candidates)
        # Only unweight these if we have access to GT.
        for batch_idx in range(batch_size):
            row_inds, col_inds = linear_sum_assignment(cost_mat[batch_idx])
            for row_idx, col_idx in zip(row_inds, col_inds):
                gt_assignment[batch_idx, row_idx, col_idx] = 1.0
                # Swap the matching ind in the cur candidates the ind of the prev.
                new_cands[batch_idx, row_idx, :] = cur_candidates[batch_idx,
                                                                  col_idx, :]
                #new_cands[batch_idx, col_idx, :] = cur_candidates[batch_idx, row_idx, :]
        return new_cands, gt_assignment

    def assign2gt(self, center_candidates, bbox_preds, gt_centers, gt_bbox,
                  gt_num_persons):
        """"Assign gt id to each valid human center candidate."""
        det_centers = center_candidates[..., :3]
        batch_size = center_candidates.shape[0]
        cand_num = center_candidates.shape[1]
        cand2gt = torch.zeros(batch_size, cand_num)
        for i in range(batch_size):
            cand = det_centers[i].view(cand_num, 1, -1)
            gt = gt_centers[None, i, :gt_num_persons[i]]

            dist = torch.sqrt(torch.sum((cand - gt)**2, dim=-1))
            min_dist, min_gt = torch.min(dist, dim=-1)

            cand2gt[i] = min_gt
            cand2gt[i][min_dist > self.train_cfg['dist_threshold']] = -1.0
            for k in range(self.max_num_people):
                threshold = 0.1
                if cand2gt[i, k] < 0:
                    continue
                if torch.sum(
                        bbox_preds[i, k] < gt_bbox[i, cand2gt[i, k].long()] -
                        threshold):
                    bbox_preds[i, k] = gt_bbox[i, cand2gt[i, k].long()]

        center_candidates[:, :, 3] = cand2gt

        return center_candidates

    def forward(self,
                img,
                img_metas,
                return_loss=True,
                feature_maps=None,
                bbox3d=None,
                bbox3d_offset=None,
                bbox3d_index=None,
                targets_1d=None,
                targets_2d=None):
        """
        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C
            img_width: imgW
            img_height: imgH
            heatmaps width: W
            heatmaps height: H
        Args:
            img (list(torch.Tensor[NxCximgHximgW])):
                Multi-camera input images to the 2D model.
            img_metas (list(dict)):
                Information about image, 3D groundtruth and camera parameters.
            return_loss: Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.
            targets_3d (torch.Tensor[NxcubeLxcubeWxcubeH]):
                Ground-truth 3D heatmap of human centers.
            feature_maps (list(torch.Tensor[NxKxHxW])):
                Multi-camera feature_maps.
        Returns:
            dict: if 'return_loss' is true, then return losses.
                Otherwise, return predicted poses
        """
        if return_loss:
            return self.forward_train(img, img_metas, feature_maps, bbox3d,
                                      bbox3d_offset, bbox3d_index, targets_2d,
                                      targets_1d)
        else:
            return self.forward_test(img, img_metas, feature_maps)

    def _get_center_candidates(self, feature_maps, img_metas, time_horizon):
        batch_size, num_channels = feature_maps[0][0].shape[:2]
        center_candidates = torch.zeros(
            batch_size * time_horizon,
            self.max_num_people,
            7,
            device=feature_maps[0][0].device)
        device = feature_maps[0][0].device
        # Run projection for each time instance separately, until we can batch it properly.
        all_initial_cubes = []
        for time_idx in range(time_horizon):
            space_size = img_metas[0]['ann_info'][time_idx]['space_size']
            space_center = img_metas[0]['ann_info'][time_idx]['space_center']
            cube_size = img_metas[0]['ann_info'][time_idx]['cube_size']
            scale = (torch.tensor(space_size) /
                     (torch.tensor(cube_size) - 1)).to(device)
            bias = (torch.tensor(space_center) -
                    torch.tensor(space_size) / 2.0).to(device)
            cameras = [img_meta['camera'][time_idx] for img_meta in img_metas]
            scales = [img_meta['scale'][time_idx] for img_meta in img_metas]
            centers = [img_meta['center'][time_idx] for img_meta in img_metas]

            initial_cubes, _ = self.project_layer(feature_maps[time_idx],
                                                  space_size, [space_center],
                                                  cube_size, centers, scales,
                                                  cameras)
            all_initial_cubes.append(initial_cubes)

        all_initial_cubes = torch.cat(all_initial_cubes, dim=0)
        center_heatmaps_2d, bbox_preds = self.center_net(all_initial_cubes)
        center_heatmaps_2d = center_heatmaps_2d.squeeze(1)
        topk_2d_confs, topk_2d_index, topk_2d_flatten_index = nms_by_max_pool(
            center_heatmaps_2d.detach(), self.max_num_people)
        bbox_preds = torch.flatten(
            bbox_preds, start_dim=2, end_dim=3).permute(0, 2, 1)
        match_bbox_preds = torch.gather(
            bbox_preds,
            dim=1,
            index=topk_2d_flatten_index.unsqueeze(2).repeat(1, 1, 2))
        feature_1d = torch.gather(torch.flatten(all_initial_cubes, 2, 3).permute(0, 2, 1, 3), dim=1,\
                                  index=topk_2d_flatten_index.view(batch_size * time_horizon, -1, 1, 1)\
                                    .repeat(1, 1, num_channels, all_initial_cubes.shape[4]))
        center_heatmaps_1d = self.center_net1d(
            torch.flatten(feature_1d, 0, 1)).view(batch_size * time_horizon,
                                                  self.max_num_people, -1)
        topk_1d_confs, topk_1d_index = center_heatmaps_1d.detach().topk(1)

        topk_index = torch.cat([topk_2d_index, topk_1d_index], dim=2)
        topk_confs = topk_2d_confs * topk_1d_confs.squeeze(2)

        topk_index = topk_index.float() * scale + bias
        center_candidates[:, :, 0:3] = topk_index
        center_candidates[:, :, 4] = topk_confs
        center_candidates[:, :, 5:7] = match_bbox_preds

        output_dict = {
            'center_candidates': center_candidates,
            'center_heatmaps_2d': center_heatmaps_2d,
            'center_heatmaps_1d': center_heatmaps_1d,
            'match_bbox_preds': match_bbox_preds,
            'bbox_preds': bbox_preds,
            'topk_2d_confs': topk_2d_confs,
            'topk_index': topk_2d_index,
        }
        return output_dict

    def forward_train(self,
                      img,
                      img_metas,
                      feature_maps=None,
                      targets_2d=None,
                      targets_1d=None,
                      bbox3d=None,
                      bbox3d_index=None,
                      return_preds=False):
        """
        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C
            img_width: imgW
            img_height: imgH
            heatmaps width: W
            heatmaps height: H
        Args:
            img (list(torch.Tensor[NxCximgHximgW])):
                Multi-camera input images to the 2D model.
            img_metas (list(dict)):
                Information about image, 3D groundtruth and camera parameters.
            targets_3d (torch.Tensor[NxcubeLxcubeWxcubeH]):
                Ground-truth 3D heatmap of human centers.
            feature_maps (list(torch.Tensor[NxKxHxW])):
                Multi-camera feature_maps.
            return_preds (bool): Whether to return prediction results
        Returns:
            dict: if 'return_pred' is true, then return losses
                and human centers. Otherwise, return losses only
        """
        assert isinstance(feature_maps, list)
        assert isinstance(feature_maps[0], list)
        assert len(feature_maps) == self.input_time

        if self.use_gt:
            assert return_preds == True
            assert isinstance(targets_2d, list)
            gt_cands = self._get_gt_candidates(img_metas)
            gt_cands = gt_cands.to(targets_2d[0].device)
            return gt_cands, None

        batch_size = feature_maps[0][0].shape[0]
        cc_dict = self._get_center_candidates(feature_maps, img_metas, time_horizon=self.input_time)
        center_candidates = cc_dict['center_candidates']
        match_bbox_preds = cc_dict['match_bbox_preds']
        bbox_preds = cc_dict['bbox_preds']
    
        device = center_candidates.device
        gt_centers = torch.cat([
            torch.stack([
                torch.tensor(img_meta['roots_3d'][time_idx], device=device)
                for img_meta in img_metas
            ]) for time_idx in range(self.input_time)
        ])

        gt_num_persons = torch.cat([
            torch.stack([
                torch.tensor(img_meta['num_persons'][time_idx], device=device)
                for img_meta in img_metas
            ]) for time_idx in range(self.input_time)
        ])
        
        bbox3d = torch.cat(bbox3d[:self.input_time], dim=0)
        center_candidates = self.assign2gt(center_candidates, match_bbox_preds,
                                           gt_centers, bbox3d, gt_num_persons)
        
        center_candidates = torch.stack([
            center_candidates[batch_size * time_idx:batch_size * (time_idx + 1)]
            for time_idx in range(self.input_time)
        ], dim=1)

        for time_idx in range(0, self.input_time - 1):
            # Just assocaite so they always on the same row.
            # shape is [B, T, max_num_people, 7]
            permuted_cands, assignment = self.create_tracker_gt(
                center_candidates[:, time_idx],
                center_candidates[:, time_idx + 1])
            center_candidates[:, time_idx + 1] = permuted_cands

        # Split center candidates into the correct shape.
        losses = dict()
        losses.update(
            self.center_head.get_loss(
                cc_dict['center_heatmaps_2d'],
                torch.cat(targets_2d[:self.input_time], dim=0),
                cc_dict['center_heatmaps_1d'],
                torch.cat(targets_1d[:self.input_time], dim=0)))
        gt_bbox_indices = torch.cat(bbox3d_index[:self.input_time], dim=0)
        flattened_bbox3d_index = gt_bbox_indices.long().view(
            batch_size * self.input_time, -1, 1)
        bbox_preds = torch.gather(bbox_preds, 1, flattened_bbox3d_index.repeat(1, 1, 2))

        # Cat along dim 0 to make time a sort of batch dimension too.
        loss_bbox = F.l1_loss(bbox_preds, bbox3d, reduction='mean')
        losses.update(dict(loss_bbox=loss_bbox))

        tracker_input = {
            #'flattened_feature_1d': flattened_feature_1d,
            'topk_2d_confs': cc_dict['topk_2d_confs'],
            'topk_index': cc_dict['topk_index'],
        }

        if return_preds:
            return center_candidates, cc_dict['center_heatmaps_2d'], cc_dict['center_heatmaps_1d'], bbox_preds, losses, tracker_input
        else:
            return losses

    def forward_test(self, img, img_metas, feature_maps=None):
        """
        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C
            img_width: imgW
            img_height: imgH
            heatmaps width: W
            heatmaps height: H
        Args:
            img (list(torch.Tensor[NxCximgHximgW])):
                Multi-camera input images to the 2D model.
            img_metas (list(dict)):
                Information about image, 3D groundtruth and camera parameters.
            feature_maps (list(torch.Tensor[NxKxHxW])):
                Multi-camera feature_maps.
        Returns:
            human centers
        """
        assert isinstance(feature_maps, list)
        assert isinstance(feature_maps[0], list)
        assert len(feature_maps) == 1

        batch_size = feature_maps[0][0].shape[0]
        cc_dict = self._get_center_candidates(feature_maps, img_metas, time_horizon=1)
        center_candidates = cc_dict['center_candidates']
        center_candidates = torch.stack([
            center_candidates[batch_size * time_idx:batch_size *
                              (time_idx + 1)] for time_idx in range(1)
        ], dim=1)

        dets = []
        valid_idxs = []
        assert center_candidates.shape[1] == 1 and center_candidates.shape[0] == 1
        for root_idx, root in enumerate(center_candidates[0][0].cpu().numpy()):
            det = [root[0] - root[4] * 1000, root[1] - root[5] * 1000, root[0] + root[4] * 1000, root[1] + root[5] * 1000, root[4] * root[5]]
            if root[4] * root[5] > 0.025:
                dets.append(det)
                valid_idxs.append(root_idx)
        dets = np.array(dets)
        # Don't bother tracking if no valid detections.
        if len(dets) == 0:
            return center_candidates
        
        _, match_mat = self.tracker.update(dets)
        # Label the track with it's id.
        for det_idx, matched_track_idx in match_mat:
            if matched_track_idx < len(self.tracker.trackers):
                center_candidates[0][0][valid_idxs[det_idx]][3]  = 1 + self.tracker.trackers[matched_track_idx].id

        return center_candidates

    def show_result(self, **kwargs):
        """Visualize the results."""
        raise NotImplementedError

    def forward_dummy(self, feature_maps):
        """Used for computing network FLOPs."""
        batch_size, num_channels, _, _ = feature_maps[0].shape
        initial_cubes = feature_maps[0].new_zeros(batch_size, num_channels,
                                                  *self.cube_size)
        _ = self.center_net(initial_cubes)
