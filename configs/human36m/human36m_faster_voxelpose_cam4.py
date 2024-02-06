_base_ = [
    '../../third_party/mmpose/configs/_base_/datasets/h36m.py',
]

dist_params = dict(backend='nccl')
log_level = 'INFO'
find_unused_parameters = True
workflow = [('train', 1)]
resume_from = None
load_from = None
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric=['mAP', 'mpjpe'], save_best='mAP')

optimizer = dict(
    type='Adam',
    lr=0.0003,
    weight_decay=0.0001,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    gamma=0.7,
    warmup_iters=500,
    warmup_ratio=0.001,
    step=2)
total_epochs = 20

log_config = dict(
    interval=15,
    hooks=[
        dict(type='TextLoggerHook'),
    ],
)

space_size = [8000, 8000, 2000]
space_center = [300, 300, 800]
cube_size = [80, 80, 20]
sub_space_size = [2000, 2000, 2000]
sub_cube_size = [64, 64, 64]
image_size = [384, 384]
heatmap_size = [384, 384]
num_joints = 17
num_features = 32
find_unused_parameters = True
root_idx = 11
max_num_people = 1
horizon = 4
input_time = 3

train_data_cfg = dict(
    image_size=image_size,
    heatmap_size=[heatmap_size],
    num_joints=num_joints,
    num_cameras=4,
    seq_frame_interval=5,
    seq_list=[
       'Miscellaneous',
        'Directions',
        'Discussion',
         'Eating',
        #'Greeting',
        'Phoning',
        'Posing',
        'Purchases',
        'Sitting',
        # 'SittingDown',
        'Smoking',
        'TakingPhoto',
        # 'Waiting',
        'Walking',
        'WalkingDog',
        'WalkingTogether'
    ],
    subset='train',
    root_id=[root_idx],
    max_num=max_num_people,
    space_size=space_size,
    space_center=space_center,
    sub_space_size=[2000, 2000, 2000],
    sub_cube_size=[64, 64, 64],
    cube_size=cube_size,
    cam_range=(0, 4),
    joints_format='coco')

test_data_cfg = train_data_cfg.copy()
test_data_cfg.update(subset='validation')

# model settings
backbone = dict(
    type='AssociativeEmbedding',
    pretrained='./checkpoints/hrnet_w32_coco_512x512-bcb8c247_20200816.pth',
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
    ),
    keypoint_head=dict(
        type='DeconvPreHead',
        in_channels=32,
        out_channels=num_joints,
        return_prefinal=True,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=num_joints,
            num_stages=1,
            ae_loss_type='exp',
            with_ae_loss=[False],
            push_loss_factor=[0.001],
            pull_loss_factor=[0.001],
            with_heatmaps_loss=[True],
            heatmaps_loss_factor=[1.0],
        )),
    train_cfg=dict(),
    test_cfg=dict(
        num_joints=num_joints,
        nms_kernel=None,
        nms_padding=None,
        tag_per_joint=None,
        max_num_people=None,
        detection_threshold=None,
        tag_threshold=None,
        use_detection_val=None,
        ignore_too_much=None,
    ))

model = dict(
    type='RNNDetectAndRegress',
    backbone=backbone,
    pretrained=None,
    freeze_2d=False,
    use_2d_hmaps=False,
    num_features=num_features,
    horizon=horizon,
    input_time=input_time,
    human_detector=dict(
        type='RNNVoxelCenterDetector',
        image_size=image_size,
        heatmap_size=heatmap_size,
        use_gt=False,
        num_features=32,
        input_time=input_time,
        horizon=horizon,
        do_forecasting=False,
        center_net=dict(
            type='CenterNet', input_channels=num_features, output_channels=1),
        forecast_net=dict(
            type='FuturePrediction',
            in_channels=num_features,
            out_channels=1,
            latent_dim=1,
        ),
        center_net_1d=dict(
            type='C2CNet', input_channels=num_features, output_channels=1),
        center_head=dict(
            type='BEVCenterHead',
            space_size=space_size,
            space_center=space_center,
            cube_size=cube_size,
            max_num=10,
            max_pool_kernel=3),
        train_cfg=dict(dist_threshold=500.0),
        test_cfg=dict(center_threshold=0.3),
        agg_method='mean',
    ),
    pose_regressor=dict(
        type='RNNVoxelSinglePose',
        image_size=image_size,
        heatmap_size=heatmap_size,
        num_joints=num_joints,
        use_masks=True,
        num_features=32,
        horizon=horizon,
        input_time=input_time,
        agg_method='mean',
        pose_net=dict(
            type='TemporalPoseModel',
            in_channels=num_features,
            out_channels=num_joints,
            receptive_field=input_time,
            input_shape=sub_cube_size[:2]),
        weight_net=dict(
            type='WeightNet',
            voxels_per_axis=sub_cube_size,
            num_joints=num_joints,
        ),
        pose_head=dict(type='FasterCuboidPoseHead', beta=100.0)))

train_pipeline = [
    dict(
        type='MultiItemProcess',
        pipeline=[
            dict(
                type='MultiItemProcess',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(
                        type='BottomUpRandomAffine',
                        rot_factor=0,
                        scale_factor=[1.0, 1.0],
                        scale_type='long',
                        trans_factor=0),
                    dict(type='ToTensor'),
                    dict(
                        type='NormalizeTensor',
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
                ]),
            dict(
                type='DiscardDuplicatedItems',
                keys_list=[
                    'joints_3d', 'joints_3d_visible', 'ann_info', 'roots_3d',
                    'num_persons', 'sample_id'
                ]),
            dict(
                type='GenerateVoxelMultiDHeatmapTarget',
                sigma=200.0,
                joint_indices=[root_idx]),
            dict(
                type='GenerateBBoxTarget',
                slack=200.0,
                max_num_people=max_num_people,
                joint_indices=[root_idx]),
        ]),
    dict(
        type='CollectTemporal',
        keys=[
            'img',
            'targets_2d',
            'targets_1d',
            'bbox3d_index',
            'bbox3d_offset',
            'bbox3d',
        ],
        meta_keys=[
            'num_persons', 'joints_3d', 'camera', 'center', 'scale',
            'joints_3d_visible', 'roots_3d', 'ann_info', 'sample_id', 'joints'
        ]),
]

val_pipeline = [
    dict(
        type='MultiItemProcess',
        pipeline=[
            dict(
                type='MultiItemProcess',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(
                        type='BottomUpRandomAffine',
                        rot_factor=0,
                        scale_factor=[1.0, 1.0],
                        scale_type='long',
                        trans_factor=0),
                    dict(type='ToTensor'),
                    dict(
                        type='NormalizeTensor',
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
                ]),
            dict(
                type='DiscardDuplicatedItems',
                keys_list=[
                    'joints_3d', 'joints_3d_visible', 'ann_info', 'roots_3d',
                    'num_persons', 'sample_id', 'image_file'
                ]),
        ]),
    dict(
        type='CollectTemporal',
        keys=['img',],
        meta_keys=[
            'sample_id', 'camera', 'center', 'scale', 'roots_3d',
            'num_persons', 'ann_info', 'joints'
        ]),
]

test_pipeline = val_pipeline

data_root = '/edrive2/rchoudhu/data/H36M-Toolbox/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=12,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='Body3DMviewTemporalH36MDataset',
        horizon=horizon,
        metadata_path='/edrive2/rchoudhu/data/H36M-Toolbox/metadata.xml',
        ann_file=None,
        input_time=input_time,
        img_prefix=data_root,
        data_cfg=train_data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='Body3DMviewTemporalH36MDataset',
        horizon=1,
        input_time=1,
        metadata_path='/edrive2/rchoudhu/data/H36M-Toolbox/metadata.xml',
        ann_file=None,
        img_prefix=data_root,
        data_cfg=test_data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='Body3DMviewTemporalH36MDataset',
        horizon=1,
        input_time=1,
        metadata_path='/edrive2/rchoudhu/data/H36M-Toolbox/metadata.xml',
        ann_file=None,
        img_prefix=data_root,
        data_cfg=test_data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
