import torch

from mmpose.models import builder, POSENETS
from mmpose.models.detectors.base import BasePose

from .model_utils import RNNFeatureEncoder
from .project_layer import ProjectLayer, ProjectLayerWithMask

@POSENETS.register_module()
class RNNVoxelSinglePose(BasePose):
    """

    Args:
        image_size (list): input size of the 2D model.
        heatmap_size (list): output size of the 2D model.
        sub_space_size (list): Size of the cuboid human proposal.
        sub_cube_size (list): Size of the input volume to the pose net.
        pose_net (ConfigDict): Dictionary to construct the pose net.
        pose_head (ConfigDict): Dictionary to construct the pose head.
        train_cfg (ConfigDict): Config for training. Default: None.
        test_cfg (ConfigDict): Config for testing. Default: None.
    """

    def __init__(self,
                 image_size,
                 heatmap_size,
                 num_joints,
                 num_features,
                 pose_net,
                 weight_net,
                 pose_head,
                 horizon=1,
                 input_time=1,
                 use_masks=True,
                 train_cfg=None,
                 test_cfg=None,
                 agg_method='mean'):
        super(RNNVoxelSinglePose, self).__init__()

        self.use_masks = use_masks
        if self.use_masks:
            self.project_layer = ProjectLayerWithMask(image_size, heatmap_size,
                                                      agg_method)
        else:
            self.project_layer = ProjectLayer(image_size, heatmap_size,
                                              agg_method)
        self.weight_net = builder.build_backbone(weight_net)
        self.pose_head = builder.build_head(pose_head)
        self.pose_rnn = RNNFeatureEncoder(
            in_channels=num_features, out_channels=num_joints)

        # Map from int -> tensor
        # for a track id, keep the IDs and embeddings
        self.track_id_to_embedding = dict()
        self.num_joints = num_joints
        self.num_features = num_features
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_candidates = 10
        self.horizon = horizon
        self.input_time = input_time
        self.time = 0
        self.prev_features = torch.zeros(self.num_candidates, 3, num_features, 64, 64)

    def forward(self,
                img,
                img_metas,
                return_loss=True,
                return_preds=False,
                feature_maps=None,
                tracker_idxs=None,
                human_candidates=None,
                **kwargs):
        """
        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C
            img_width: imgW
            img_height: imgH
            feature_maps width: W
            feature_maps height: H
            volume_length: cubeL
            volume_width: cubeW
            volume_height: cubeH

        Args:
            img (list(torch.Tensor[NxCximgHximgW])):
                Multi-camera input images to the 2D model.
            feature_maps (list(torch.Tensor[NxCxHxW])):
                Multi-camera input feature_maps.
            img_metas (list(dict)):
                Information about image, 3D groundtruth and camera parameters.
            human_candidates (torch.Tensor[NxPx5]):
                Human candidates.
            return_loss: Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.

        """
        if return_loss:
            return self.forward_train(img, img_metas, feature_maps,
                                      human_candidates, tracker_idxs,
                                      return_preds)
        else:
            return self.forward_test(img, img_metas, feature_maps,
                                     human_candidates, tracker_idxs)

    def fuse_pose_preds(self, pose_preds, weights):
        weights = torch.chunk(weights, 3)
        xy_weight, xz_weight, yz_weight = weights[0], weights[1], weights[2]
        xy_pred, xz_pred, yz_pred = pose_preds[0], pose_preds[1], pose_preds[2]

        # normalize
        x_weight = torch.cat([xy_weight, xz_weight], dim=2)
        y_weight = torch.cat([xy_weight, yz_weight], dim=2)
        z_weight = torch.cat([xz_weight, yz_weight], dim=2)
        x_weight = x_weight / torch.sum(x_weight, dim=2).unsqueeze(2)
        y_weight = y_weight / torch.sum(y_weight, dim=2).unsqueeze(2)
        z_weight = z_weight / torch.sum(z_weight, dim=2).unsqueeze(2)

        x_pred = x_weight[:, :, :
                          1] * xy_pred[:, :, :
                                       1] + x_weight[:, :,
                                                     1:] * xz_pred[:, :, :1]
        y_pred = y_weight[:, :, :
                          1] * xy_pred[:, :,
                                       1:] + y_weight[:, :,
                                                      1:] * yz_pred[:, :, :1]
        z_pred = z_weight[:, :, :
                          1] * xz_pred[:, :, 1:] + z_weight[:, :,
                                                            1:] * yz_pred[:, :,
                                                                          1:]

        pred = torch.cat([x_pred, y_pred, z_pred], dim=2)
        return pred

    def compute_person_feature_maps(self, img_metas, feature_maps, human_candidates, cand_idx):
        batch_size = human_candidates.shape[0]
        sub_space_size = img_metas[0]['ann_info'][0]['sub_space_size']
        sub_cube_size = img_metas[0]['ann_info'][0]['sub_cube_size']
        pose_input_cubes = []
        stacked_grids = []
        all_offsets = []

        for time_idx in range(len(img_metas[0]['ann_info'])):
            space_center = img_metas[0]['ann_info'][time_idx]['space_center']
            space_size = img_metas[0]['ann_info'][time_idx]['space_size']
            cameras = [img_meta['camera'][time_idx] for img_meta in img_metas]
            scales = [img_meta['scale'][time_idx] for img_meta in img_metas]
            centers = [img_meta['center'][time_idx] for img_meta in img_metas]

            pose_input_cube, offset \
                = self.project_layer(feature_maps[time_idx],
                                    human_candidates[:, -1, cand_idx],
                                    space_center, space_size, sub_space_size, sub_cube_size,
                                    centers, scales, cameras)

            assert self.project_layer.center_grid is not None
            stacked_grid = torch.stack([
                self.project_layer.center_grid
                for _ in range(batch_size)
            ], dim=0)

            all_offsets.append(offset)
            pose_input_cubes.append(pose_input_cube)
            stacked_grids.append(stacked_grid)

        pose_input_cubes = torch.stack(pose_input_cubes, dim=1)
        stacked_grids = torch.stack(stacked_grids, dim=1)
        offsets = torch.stack(all_offsets, dim=1)

        input = torch.cat([
            torch.max(pose_input_cubes, dim=5)[0],
            torch.max(pose_input_cubes, dim=4)[0],
            torch.max(pose_input_cubes, dim=3)[0]
        ])

        return input, offsets, stacked_grids
    
    def produce_pose_outputs(self, output, index, offsets, stacked_grids):
        joint_features = torch.chunk(output, 3)
        joint_features = torch.stack(joint_features, dim=2)
        
        joint_features = joint_features[index]
        num_valid = joint_features.shape[0]
        joint_features = joint_features.flatten(end_dim=1)
        stacked_grids = stacked_grids[index].flatten(end_dim=1)
        pose_preds, confs = self.pose_head(joint_features, stacked_grids)

        offsets = offsets[index].reshape(-1, 1, 3)
        pose_preds[:, 0] += offsets[:, :, :2]
        pose_preds[:, 1] += offsets[:, :, ::2]
        pose_preds[:, 2] += offsets[:, :, 1:]
        # Go from [batch_size, 3, num_channels, height, width]
        # to [3, batch_size, num_channels, height, width]
        joint_features = joint_features.transpose(0, 1)
        pose_preds = pose_preds.transpose(0, 1)
        weights = self.weight_net(joint_features)
        fused_pose_preds = self.fuse_pose_preds(pose_preds, weights)
        # Only save the last index for eval purposes.
        final_pred = fused_pose_preds.detach().view(num_valid, -1, self.num_joints, 3)[:, -1]
        
        return pose_preds, fused_pose_preds, final_pred

    def forward_train(self,
                      img,
                      img_metas,
                      feature_maps=None,
                      human_candidates=None,
                      tracker_idxs=None,
                      return_preds=False,
                      **kwargs):
        """Defines the computation performed at training.
        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C
            img_width: imgW
            img_height: imgH
            feature_maps width: W
            feature_maps height: H
            volume_length: cubeL
            volume_width: cubeW
            volume_height: cubeH

        Args:
            img (list(torch.Tensor[NxCximgHximgW])):
                Multi-camera input images to the 2D model.
            feature_maps (list(torch.Tensor[NxCxHxW])):
                Multi-camera input feature_maps.
            img_metas (list(dict)):
                Information about image, 3D groundtruth and camera parameters.
            human_candidates (torch.Tensor[NxPx5]):
                Human candidates.
            return_preds (bool): Whether to return prediction results

        Returns:
            dict: losses.

        """
        batch_size, _, num_candidates, _ = human_candidates.shape
        pred = human_candidates.new_zeros(batch_size, num_candidates, self.num_joints, 5)
        pred[:, :, :, 3:] = human_candidates[:, self.input_time - 1, :, None, 3:5]

        device = feature_maps[0][0].device
        gt_3d = torch.stack([
            torch.stack([
                torch.tensor(img_meta['joints_3d'][time_idx], device=device)
                for img_meta in img_metas
            ]) for time_idx in range(self.input_time)
        ], dim=1)

        gt_3d_vis = torch.stack([
            torch.stack([
                torch.tensor(
                    img_meta['joints_3d_visible'][time_idx], device=device)
                for img_meta in img_metas
            ]) for time_idx in range(self.input_time)
        ], dim=1)

        valid_preds = []
        valid_preds_2d = []
        valid_targets = []
        valid_weights = []

        for n in range(num_candidates):
            index = pred[:, n, 0, 3] >= 0
            num_valid = index.sum()
            if num_valid > 0:
                input, offsets, stacked_grids = self.compute_person_feature_maps(
                    img_metas, feature_maps, human_candidates, n)

                output, _ = self.pose_rnn(input)               
                pose_preds, fused_pose_preds, final_pred = \
                    self.produce_pose_outputs(output, index, offsets, stacked_grids)
                # Only save the last index for eval purposes.
                pred[index, n, :, :3] = final_pred
                pose_preds = pose_preds.transpose(0, 1)

                valid_preds.append(fused_pose_preds)
                valid_preds_2d.append(pose_preds)
                valid_targets.append(gt_3d[index, :,
                                           pred[index, n, 0,
                                                3].long()].flatten(end_dim=1))
                valid_weights.append(gt_3d_vis[index, :, pred[index, n, 0,
                                                              3].long(), :,
                                               0:1].float().flatten(end_dim=1))

        losses = dict()
        if len(valid_preds) > 0:
            valid_targets = torch.cat(valid_targets, dim=0)
            valid_weights = torch.cat(valid_weights, dim=0)
            valid_preds = torch.cat(valid_preds, dim=0)
            valid_preds2d = torch.cat(valid_preds_2d, dim=0)
            losses.update(
                self.pose_head.get_loss(valid_preds, valid_preds2d,
                                        valid_targets, valid_weights))
        else:
            sub_cube_size = img_metas[0]['ann_info'][0]['sub_cube_size']
            pose_input_cube = feature_maps[0][0].new_zeros(
                batch_size, self.input_time, self.num_features, *sub_cube_size)
            input = torch.cat([
                torch.max(pose_input_cube, dim=5)[0],
                torch.max(pose_input_cube, dim=4)[0],
                torch.max(pose_input_cube, dim=3)[0]
            ])
            #joint_features, temporal_rep = self.pose_net(input)
            joint_features, _ = self.pose_rnn(input)
            joint_features = torch.chunk(joint_features, 3)
            # Output is [batch_size * 3, 1, out_channels, height, width]
            joint_features = torch.stack(
                joint_features, dim=2).flatten(end_dim=1)
            coordinates = feature_maps[0][0].new_zeros(
                batch_size, self.input_time, 3, *sub_cube_size[:2],
                2).view(batch_size * self.input_time, 3, -1, 2)
            # Stack the time and batch dims to make a single dim, run the pose head.
            pose_preds, _ = self.pose_head(joint_features, coordinates)
            joint_features = joint_features.transpose(0, 1)
            pose_preds = pose_preds.transpose(0, 1)
            weights = self.weight_net(joint_features)
            fused_pose_preds = self.fuse_pose_preds(pose_preds, weights)
            pseudo_targets = feature_maps[0][0].new_zeros(
                batch_size * self.input_time, self.num_joints, 3)
            pseudo_weights = feature_maps[0][0].new_zeros(
                batch_size * self.input_time, self.num_joints, 1)
            pose_preds = pose_preds.transpose(0, 1)
            pose_loss = self.pose_head.get_loss(fused_pose_preds, pose_preds,
                                                pseudo_targets, pseudo_weights)
            losses.update(pose_loss)

        if return_preds:
            return pred, losses
        else:
            return losses

    def forward_test(self,
                     img,
                     img_metas,
                     feature_maps=None,
                     human_candidates=None,
                     tracker_idxs=None,
                     **kwargs):
        """Defines the computation performed at training.
        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C
            img_width: imgW
            img_height: imgH
            feature_maps width: W
            feature_maps height: H
            volume_length: cubeL
            volume_width: cubeW
            volume_height: cubeH

        Args:
            img (list(torch.Tensor[NxCximgHximgW])):
                Multi-camera input images to the 2D model.
            feature_maps (list(torch.Tensor[NxCxHxW])):
                Multi-camera input feature_maps.
            img_metas (list(dict)):
                Information about image, 3D groundtruth and camera parameters.
            human_candidates (torch.Tensor[NxPx5]):
                Human candidates.

        Returns:
            dict: predicted poses, human centers and sample_id

        """
        input_time = 1
        batch_size, _, num_candidates, _ = human_candidates.shape
        #batch_size = batch_size // self.horizon
        pred = human_candidates.new_zeros(batch_size, num_candidates,
                                          self.num_joints, 5)
        self.prev_features = self.prev_features.to(human_candidates.device)
        pred[:, :, :, 3:] = human_candidates[:, input_time - 1, :, None, 3:5]
        for n in range(num_candidates):
            index = pred[:, n, 0, 3] >= 1
            num_valid = index.sum()
            if num_valid > 0:
                input, offsets, stacked_grids = self.compute_person_feature_maps(
                    img_metas, feature_maps, human_candidates, n)
                # Update the embedding used.
                track_id = int(human_candidates[0, 0, n, 3].cpu())
                assert input.shape[1] == 1

                if track_id in self.track_id_to_embedding:
                    new_embed = torch.zeros_like(self.track_id_to_embedding[track_id], device=input.device)
                    new_embed[:,:-1] = self.track_id_to_embedding[track_id][:, 1:].clone()
                    new_embed[:, -1] = input.squeeze(1)
                    output, _ = self.pose_rnn(new_embed)
                    # Only output pose at current TS.
                    output = output[:, -1:]
                    self.track_id_to_embedding[track_id] = new_embed
                else:
                    output, _ = self.pose_rnn(input, self.prev_features[n])
                    self.track_id_to_embedding[track_id] = torch.zeros(input.shape[0], self.horizon, *input.shape[2:])
                    self.track_id_to_embedding[track_id][:, -1] = input.squeeze(1)

                _, _, final_pred = \
                    self.produce_pose_outputs(output, index, offsets, stacked_grids)
                # Only save the last index for eval purposes.
                pred[index, n, :, :3] = final_pred
        self.time += 1
        return pred

    def show_result(self, **kwargs):
        """Visualize the results."""
        raise NotImplementedError

    def forward_dummy(self, feature_maps, num_candidates=5):
        """Used for computing network FLOPs."""
        batch_size, num_channels = feature_maps[0].shape
        default_cube_sze = [80, 80, 20]
        pose_input_cube = feature_maps[0].new_zeros(batch_size, num_channels,
                                                    *default_cube_sze)
        for n in range(num_candidates):
            _ = self.pose_net(pose_input_cube)
