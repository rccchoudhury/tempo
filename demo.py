import argparse
import cv2
import os
import os.path as osp
import warnings
import tempfile
from tqdm import tqdm
import ipdb
from functools import partial
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmpose.apis import multi_gpu_test, single_gpu_test
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.models import build_posenet
from mmpose.utils import setup_multi_processes
from mmpose.core.camera import SimpleCamera

from tempo import *
from tempo.vis_utils import draw_3d_skeleton, draw_2d_skeleton
from tempo.tracker.sort import Sort

def parse_args():
    parser = argparse.ArgumentParser(description='mmpose test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--work-dir', help='the dir to save evaluation results')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--eval',
        default=None,
        nargs='+',
        help='evaluation metric, which depends on the dataset,'
        ' e.g., "mAP" for MSCOCO')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_configs(cfg1, cfg2):
    # Merge cfg2 into cfg1
    # Overwrite cfg1 if repeated, ignore if value is None.
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1


def process_output(colors, tmpdir, i_output):
    i, output = i_output
    centers = output['human_detection_3d'][0].squeeze(0)
    skeletons = np.array(output['pose_3d'][0])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(-3500, 3000)
    ax.set_ylim(-3500, 3000)
    ax.set_zlim(0, 2000)
    for root_idx, root in enumerate(centers):
        if root[3] > 0:
            draw_3d_skeleton(ax, skeletons[root_idx], color=colors[int(root[3] % 10)])
        
    file_name = os.path.join(tmpdir, "{:04d}.png".format(i))
    plt.savefig(file_name)
    plt.close(fig)
    return file_name

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    # step 1: give default values and override (if exist) from cfg.data
    loader_cfg = {
        **dict(seed=cfg.get('seed'), drop_last=False, dist=distributed),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               pin_memory=False,
           )),
        **dict((k, cfg.data[k]) for k in [
                   'seed',
                   'prefetch_num',
                   'pin_memory',
                   'persistent_workers',
               ] if k in cfg.data)
    }
    # step2: cfg.data.test_dataloader has higher priority
    test_loader_cfg = {
        **loader_cfg,
        **dict(shuffle=False, drop_last=False),
        **dict(workers_per_gpu=cfg.data.get('workers_per_gpu', 1)),
        **dict(samples_per_gpu=cfg.data.get('samples_per_gpu', 1)),
        **cfg.data.get('test_dataloader', {})
    }
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    model = build_posenet(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        model = MMDataParallel(model, device_ids=[args.gpu_id])
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    eval_config = cfg.get('evaluation', {})
    eval_config = merge_configs(eval_config, dict(metric=args.eval))

    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)

        results = dataset.evaluate(outputs, cfg.work_dir, **eval_config)
        for k, v in sorted(results.items()):
            print(f'{k}: {v}')

    print("Running visualization...")
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    colors = ['b', 'r', 'g', 'y', 'c', 'm', 'k', 'purple', 'orange', 'brown']
    colors_cv2 = [
        (255, 0, 0),      # Blue
        (0, 0, 255),      # Red
        (0, 128, 0),      # Green
        (0, 255, 255),    # Yellow
        (204, 204, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 0, 0),        # Black
        (128, 0, 128),    # Purple
        (0, 165, 255),    # Orange
        (42, 42, 165)     # Brown
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        print("Plotting results...")
        # Do plotting with MP.
        cameras = [SimpleCamera(cam) for cam in dataset[0]['img_metas'].data['camera'][0]]
        func = partial(process_output, colors, tmpdir)
        with Pool(processes=12) as pool:
            image_paths = list(tqdm(pool.imap(func, enumerate(outputs)), total=len(outputs)))
        # Make video from frames
        print("Making video...")
        output_path = "./vis_2d_bboxes.avi"
        video_writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*'DIVX'), 10, (1152, 648))
        images = sorted(os.listdir(tmpdir))
        for idx, img_path in tqdm(enumerate(images), total=len(images)):
            img_path = os.path.join(tmpdir, img_path)
            image3d = cv2.imread(img_path)
            # 2D images.
            input_image_files = dataset[idx]['img_metas'].data['image_file']
            assert len(input_image_files) == 1
            input_image_files = input_image_files[0]
            skeletons = np.array(outputs[idx]['pose_3d'][0])
            roots = outputs[idx]['human_detection_3d'][0].squeeze(0)
            # Load all the image files with CV2, concat with image. 
            cam_images = []
            for cam_idx, image_file in enumerate(input_image_files):
                image = cv2.imread(image_file)
                pose2d = cameras[cam_idx].world_to_pixel(skeletons[..., :3])
                for pose_idx, pose in enumerate(pose2d):
                    image = draw_2d_skeleton(image, pose, color=colors_cv2[int(roots[pose_idx][3] % 10)])
                image = cv2.resize(image, (384, 216))
                cam_images.append(image)
            image2d_right = np.concatenate(cam_images[:3], axis=0)
            image2d_bottom = np.concatenate(cam_images[3:], axis=1)
            image3d = cv2.resize(image3d, (image2d_bottom.shape[1], image2d_bottom.shape[0] * 2))
            final_image = np.concatenate([image3d, image2d_bottom], axis=0)
            final_image = np.concatenate([final_image, image2d_right], axis=1)
            video_writer.write(final_image)
        video_writer.release()
        print("Done! Video written to %s" % output_path)


if __name__ == '__main__':
    main()
