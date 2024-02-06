# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile
import mmcv
import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from mmpose.core.camera import SimpleCamera
from mmpose.models import builder, POSENETS
from mmpose.models.detectors.base import BasePose

@POSENETS.register_module()
class RNNDetectAndRegress(BasePose):
    """DetectAndRegress approach for multiview human pose detection.

    Args:
        backbone (ConfigDict): Dictionary to construct the 2D pose detector
        human_detector (ConfigDict): dictionary to construct human detector
        pose_regressor (ConfigDict): dictionary to construct pose regressor
        train_cfg (ConfigDict): Config for training. Default: None.
        test_cfg (ConfigDict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained 2D model. Default: None.
        freeze_2d (bool): Whether to freeze the 2D model in training.
            Default: True.
    """

    def __init__(self,
                 backbone,
                 human_detector,
                 pose_regressor,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 freeze_2d=True,
                 num_features=32,
                 horizon=1,
                 input_time=1,
                 use_2d_hmaps=False):
        super(RNNDetectAndRegress, self).__init__()
        self.freeze_2d = freeze_2d
        if backbone is not None:
            self.backbone = builder.build_posenet(backbone)
            if self.training and pretrained is not None:
                load_checkpoint(self.backbone, pretrained)
            if self.freeze_2d:
                self._freeze(self.backbone)
        else:
            self.backbone = None

        self.horizon = horizon
        self.input_time = input_time

        self.num_features = num_features
        
        self.use_2d_hmaps = use_2d_hmaps
        self.human_detector = builder.MODELS.build(human_detector)
        self.pose_regressor = builder.MODELS.build(pose_regressor)
        if not self.use_2d_hmaps:
            self.process_layer = nn.Conv2d(
                in_channels=256,
                out_channels=num_features,
                kernel_size=1,
                stride=1,
                padding=0)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.use_gt_detections = self.human_detector.use_gt

    @staticmethod
    def _freeze(model):
        """Freeze parameters."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        """Sets the module in training mode.
        Args:
            mode (bool): whether to set training mode (``True``)
                or evaluation mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        super().train(mode)
        if mode and self.freeze_2d and self.backbone is not None:
            self._freeze(self.backbone)

        return self

    @staticmethod
    def coco2campus3D(coco_pose):
        """transform coco order(our method output) 3d pose to campus dataset
        order with interpolation.

        Args:
            coco_pose: np.array with shape 17x3

        Returns: 3D pose in campus order with shape 14x3
        """
        campus_pose = np.zeros((14, 3))
        coco2campus = np.array([16, 14, 12, 11, 13, 15, 10, 8, 6, 5, 7, 9])
        campus_pose[0:12] += coco_pose[coco2campus]

        # L and R shoulder
        mid_sho = (coco_pose[5] + coco_pose[6]) / 2
        # middle of two ear
        head_center = (coco_pose[3] + coco_pose[4]) / 2

        # nose and head center
        head_bottom = (mid_sho + head_center) / 2
        head_top = head_bottom + (head_center - head_bottom) * 2
        campus_pose[12] += head_bottom
        campus_pose[13] += head_top

        return campus_pose

    @staticmethod
    def coco2shelf3D(coco_pose, alpha=0.75):
        """transform coco order(our method output) 3d pose to shelf dataset
        order with interpolation.

        Args:
            coco_pose: np.array with shape 17x3

        Returns: 3D pose in shelf order with shape 14x3
        """
        shelf_pose = np.zeros((14, 3))
        coco2shelf = np.array([16, 14, 12, 11, 13, 15, 10, 8, 6, 5, 7, 9])
        shelf_pose[0:12] += coco_pose[coco2shelf]

        # L and R shoulder
        mid_sho = (coco_pose[5] + coco_pose[6]) / 2
        # middle of two ear
        head_center = (coco_pose[3] + coco_pose[4]) / 2

        # nose and head center
        head_bottom = (mid_sho + head_center) / 2
        head_top = head_bottom + (head_center - head_bottom) * 2

        # Use middle of shoulder to init
        shelf_pose[12] = (shelf_pose[8] + shelf_pose[9]) / 2
        # use nose to init
        shelf_pose[13] = coco_pose[0]

        shelf_pose[13] = shelf_pose[12] + (
            shelf_pose[13] - shelf_pose[12]) * np.array([0.75, 0.75, 1.5])
        shelf_pose[12] = shelf_pose[12] + (
            coco_pose[0] - shelf_pose[12]) * np.array([0.5, 0.5, 0.5])

        shelf_pose[13] = shelf_pose[13] * alpha + head_top * (1 - alpha)
        shelf_pose[12] = shelf_pose[12] * alpha + head_bottom * (1 - alpha)

        return shelf_pose

    def forward(self,
                img=None,
                img_metas=None,
                return_loss=True,
                targets_1d=None,
                targets_2d=None,
                input_heatmaps=None,
                bbox3d_index=None,
                bbox3d_offset=None,
                bbox3d=None,
                joints=None,
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
            img_metas (list(dict)):
                Information about image, 3D groundtruth and camera parameters.
            return_loss: Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.
            targets (list(torch.Tensor[NxKxHxW])):
                Multi-camera target feature_maps of the 2D model.
            masks (list(torch.Tensor[NxHxW])):
                Multi-camera masks of the input to the 2D model.
            targets_3d (torch.Tensor[NxcubeLxcubeWxcubeH]):
                Ground-truth 3D heatmap of human centers.
            input_heatmaps (list(torch.Tensor[NxKxHxW])):
                Multi-camera feature_maps when the 2D model is not available.
                 Default: None.
            **kwargs:

        Returns:
            dict: if 'return_loss' is true, then return losses.
              Otherwise, return predicted poses, human centers and sample_id

        """
        if return_loss:
            return self.forward_train(img, img_metas, bbox3d_index,
                                      bbox3d_offset, bbox3d, targets_2d,
                                      targets_1d, joints, input_heatmaps)
        else:
            return self.forward_test(img, img_metas, input_heatmaps)

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses, log_imgs = self.forward(**data_batch)
        loss, log_vars = self._parse_losses(losses)
        if 'img' in data_batch:
            batch_size = data_batch['img'][0][0].shape[0]
        else:
            assert 'input_heatmaps' in data_batch
            batch_size = data_batch['input_heatmaps'][0][0].shape[0]

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=batch_size,
            log_imgs=log_imgs)

        return outputs

    def forward_train(self,
                      img,
                      img_metas,
                      bbox3d_index,
                      bbox3d_offset,
                      bbox3d,
                      targets_2d=None,
                      targets_1d=None,
                      joints=None,
                      input_heatmaps=None):
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
            img_metas (list(dict)):
                Information about image, 3D groundtruth and camera parameters.
            targets (list(torch.Tensor[NxKxHxW])):
                Multi-camera target feature_maps of the 2D model.
            masks (list(torch.Tensor[NxHxW])):
                Multi-camera masks of the input to the 2D model.
            targets_3d (torch.Tensor[NxcubeLxcubeWxcubeH]):
                Ground-truth 3D heatmap of human centers.
            input_heatmaps (list(torch.Tensor[NxKxHxW])):
                Multi-camera feature_maps when the 2D model is not available.
                 Default: None.

        Returns:
            dict: losses.

        """
        #pdb.set_trace()
        if input_heatmaps is not None:
            feature_maps = []
            for input_heatmap in input_heatmaps:
                feature_maps.append(input_heatmap[0])
        else:
            feature_maps = []
            assert isinstance(img, list) and len(img) == self.horizon
            assert isinstance(img[0], list)
            num_views = len(img[0])
            # Dont' compute backbone
            # with torch.no_grad():
            #     img_stack_pre = torch.cat(
            #         [torch.cat(img_t, dim=0) for img_t in img[:self.input_time-1]],
            #         dim=0)
            #     backbone_map = self.backbone.forward_dummy(img_stack_pre)[0]
            #     if not self.use_2d_hmaps:
            #         backbone_map_pre = self.process_layer(backbone_map)
            img_stack = torch.cat(
                    [torch.cat(img_t, dim=0) for img_t in img[:self.input_time]],
                    dim=0)
            backbone_map = self.backbone.forward_dummy(img_stack)[0]
            if not self.use_2d_hmaps:
                backbone_map = self.process_layer(backbone_map)

            # Re-organize feature maps into the list format as expected by the rest of the code.
            feature_maps = list(
                list(torch.chunk(bmap_t, num_views))
                for bmap_t in torch.chunk(backbone_map, self.input_time))

        losses = dict()
        human_candidates, heatmaps_2d, heatmaps_1d, bbox_preds, human_loss, tracker_input = self.human_detector.forward_train(
            img,
            img_metas,
            feature_maps=feature_maps,
            targets_2d=targets_2d,
            targets_1d=targets_1d,
            bbox3d_index=bbox3d_index,
            bbox3d=bbox3d,
            return_preds=True)
        if not self.use_gt_detections:
            losses.update(human_loss)

        # Only train on the
        preds, pose_loss = self.pose_regressor(
            None,
            img_metas,
            return_preds=True,
            return_loss=True,
            feature_maps=feature_maps,
            tracker_idxs=None,
            human_candidates=human_candidates)
        losses.update(pose_loss)

        log_imgs = dict()
        log_imgs['bbox3d'] = bbox3d
        log_imgs['bbox_preds'] = bbox_preds
        log_imgs['targets_2d'] = targets_2d
        log_imgs['heatmaps_2d'] = heatmaps_2d
        log_imgs['targets_1d'] = targets_1d
        log_imgs['heatmaps_1d'] = heatmaps_1d
        #log_imgs['heatmaps_pred'] = feature_maps
        log_imgs['roots_3d_pred'] = human_candidates
        log_imgs['tracker_idxs'] = None
        # only visualize batch_idx = 0
        log_imgs['joints_3d'] = [
            img_meta['joints_3d'] for img_meta in img_metas
        ]
        log_imgs['joints_2d_gt'] = [
            img_meta['joints'] for img_meta in img_metas
        ]
        # #log_imgs['joints_pred'] = preds
        log_imgs['camera'] = img_metas[0]['camera']
        log_imgs['center'] = img_metas[0]['center']
        log_imgs['scale'] = img_metas[0]['scale']
        log_imgs['img'] = img
        log_imgs['roots_3d_gt'] = [
            img_meta['roots_3d'] for img_meta in img_metas
        ]
        log_imgs['num_persons'] = [
            img_meta['num_persons'] for img_meta in img_metas
        ]
        # log_imgs['space_center'] = img_metas[0]['ann_info']['space_center']
        # log_imgs['space_size'] = img_metas[0]['ann_info']['space_size']

        return losses, log_imgs

    def forward_test(
        self,
        img,
        img_metas,
        input_heatmaps=None,
    ):
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
            img_metas (list(dict)):
                Information about image, 3D groundtruth and camera parameters.
            input_heatmaps (list(torch.Tensor[NxKxHxW])):
                Multi-camera feature_maps when the 2D model is not available.
                 Default: None.

        Returns:
            dict: predicted poses, human centers and sample_id

        """
        input_time = 1
        if input_heatmaps is not None:
            feature_maps = []
            for input_heatmap in input_heatmaps:
                feature_maps.append(input_heatmap[0])
        else:
            feature_maps = []
            # instantaneous infernece means that we only have 1 timestep!
            assert isinstance(img, list) and len(img) == 1
            assert isinstance(img[0], list)

            num_views = len(img[0])
            img_stack = torch.cat(
                [torch.cat(img_t, dim=0) for img_t in img[:input_time]], dim=0)
            backbone_map = self.backbone.forward_dummy(img_stack)[0]
            if not self.use_2d_hmaps:
                backbone_map = self.process_layer(backbone_map)
            # Re-organize feature maps into the list format as expected by the rest of the code.
            feature_maps = list(
                list(torch.chunk(bmap_t, num_views))
                for bmap_t in torch.chunk(backbone_map, input_time))

        human_candidates = self.human_detector.forward_test(
            None, img_metas, feature_maps)

        human_poses = self.pose_regressor.forward_test(
            None,
            img_metas,
            feature_maps=feature_maps,
            human_candidates=human_candidates.clone())

        result = {}
        result['pose_3d'] = human_poses.cpu().numpy()
        result['human_detection_3d'] = human_candidates.cpu().numpy()
        result['sample_id'] = [
            img_meta['sample_id'][input_time - 1] for img_meta in img_metas
        ]
        result['image_file'] = img_metas[0]['image_file']
        result['camera'] = img_metas[0]['camera']

        return result

    def show_result(self,
                    img,
                    img_metas,
                    visualize_2d=False,
                    input_heatmaps=None,
                    dataset_info=None,
                    radius=4,
                    thickness=2,
                    out_dir=None,
                    show=False):
        """Visualize the results."""
        result = self.forward_test(
            img, img_metas, input_heatmaps=input_heatmaps)
        pose_3d = result['pose_3d']
        sample_id = result['sample_id']
        root_idx = 11
        hmaps = result['heatmaps_2d']
        root_heatmaps = [hmap[:, root_idx] for hmap in hmaps]
        batch_size = pose_3d.shape[0]

        # get kpts and skeleton structure
        for i in range(batch_size):
            # visualize 3d results
            img_meta = img_metas[i]

            num_cameras = len(img_meta['camera'])
            pose_3d_i = pose_3d[i]
            pose_3d_i = pose_3d_i[pose_3d_i[:, 0, 3] >= 0]
            new_poses = []
            for pose in pose_3d_i:
                new_pose = self.coco2shelf3D(pose[:, :3])
                # Add column of ones to new_pose
                new_poses.append(new_pose)
            if len(new_poses) == 0:
                continue
            #print("Nonzero poses, visualizing...")
            pose_3d_i = np.array(new_poses)
            num_persons, num_keypoints, _ = pose_3d_i.shape
            poses_3d = np.expand_dims(pose_3d_i, 0)
            #print("Running visualization...")
            img_3d = imshow_multiview_keypoints_3d_simple(
                joints_pred=poses_3d,
                cameras=[camera for camera in img_meta['camera']],
            )
            assert isinstance(img_3d, torch.Tensor)
            img_3d = img_3d.cpu().numpy().transpose(1, 2, 0)
            img_2ds = []
            hmaps_2d = []
            for j in range(num_cameras):
                single_camera = SimpleCamera(img_meta['camera'][j])
                # img = mmcv.imread(img)
                if num_persons > 0:
                    pose_2d = np.ones_like(pose_3d_i[..., :3])
                    pose_2d_flat = single_camera.world_to_pixel(
                        pose_3d_i[..., :3].reshape((-1, 3)))
                    pose_2d[..., :2] = pose_2d_flat.reshape(
                        (num_persons, -1, 2))
                    pose_2d_list = [pose for pose in pose_2d]
                else:
                    pose_2d_list = []

                with tempfile.TemporaryDirectory() as tmpdir:
                    if 'image_file' in img_meta:
                        img_file = img_meta['image_file'][j]
                    else:
                        img_size = img_meta['center'][j] * 2
                        img = np.zeros([int(img_size[1]),
                                        int(img_size[0]), 3],
                                       dtype=np.uint8)
                        img.fill(255)  # or img[:] = 255
                        img_file = os.path.join(tmpdir, 'tmp.jpg')
                        mmcv.image.imwrite(img, img_file)
                    img = imshow_keypoints(
                        img_file, pose_2d_list, dataset_info.skeleton, 0.0,
                        dataset_info.pose_kpt_color[:num_keypoints],
                        dataset_info.pose_link_color, radius, thickness)

                    # Resize 2D image to fit on top of the 3D one.
                    new_size = img_3d.shape[1] // num_cameras
                    img = mmcv.imresize(img, (new_size, new_size))
                    hmap = root_heatmaps[j]
                    # hmap = (hmap - hmap.min())
                    # hmap = hmap / hmap.max()
                    hmap = hmap.cpu().numpy().transpose(1, 2, 0)
                    hmap = mmcv.imresize(hmap, (new_size, new_size))
                    hmaps_2d.append(hmap)
                    img_2ds.append(img)
            full_img_2d = np.concatenate(img_2ds, axis=1)
            full_img_2d = mmcv.imresize(full_img_2d, (img_3d.shape[1],
                                                      full_img_2d.shape[0]))
            full_hmap_2d = np.concatenate(hmaps_2d, axis=1)
            full_hmap_2d = mmcv.gray2rgb(
                mmcv.imresize(full_hmap_2d,
                              (img_3d.shape[1], full_hmap_2d.shape[0]))) * 255
            full_img = np.concatenate([img_3d, full_img_2d, full_hmap_2d],
                                      axis=0)
            if out_dir is not None:
                #print("Saving visualization to {}".format(out_dir))
                mmcv.image.imwrite(
                    full_img, os.path.join(out_dir, f'{sample_id[i]}.jpg'))

    def forward_dummy(self, img, input_heatmaps=None, num_candidates=5):
        """Used for computing network FLOPs."""
        if self.backbone is None:
            assert input_heatmaps is not None
            feature_maps = []
            for input_heatmap in input_heatmaps:
                feature_maps.append(input_heatmap[0])
        else:
            feature_maps = []
            assert isinstance(img, list)
            for img_ in img:
                backbone_map = self.backbone.forward_dummy(img_)[0]
                if not self.use_2d_hmaps:
                    backbone_map = self.process_layer(backbone_map)
                feature_maps.append(backbone_map)

        _ = self.human_detector.forward_dummy(feature_maps)

        _ = self.pose_regressor.forward_dummy(feature_maps, num_candidates)