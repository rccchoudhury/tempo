# TEMPO: Efficient Multi-View Pose Estimation, Tracking, and Forecasting

## Setup and installation
To set up the environment, run the provided setup script:
```
./setup.sh
```

This script should create a conda environment with the correct dependencies, and set up the mmpose subdirectory that the code uses for many basic functions.

Next, you need to download the ResNet backbone pre-trained on the Panoptic studio dataset. You can do so from this link. Place it in a directory ```checkpoints``` so that
the directory structure looks like:
```
${ROOT}
|-- checkpoints
    | -- resnet_50_deconv.pth.tar
```

## Data 
We provide initial support for the CMU Panoptic and Human3.6M datasets. To set up the Panoptic dataset, please refer to [VoxelPose](https://github.com/microsoft/voxelpose-pytorch#data-preparation) for detailed instructions.

The directory tree should look like this:
```
${ROOT}
|-- data
    |-- panoptic
        |-- 16060224_haggling1
        |   |-- hdImgs
        |   |-- hdvideos
        |   |-- hdPose3d_stage1_coco19
        |   |-- calibration_160224_haggling1.json
        |-- 160226_haggling1  
        |-- ...
```

## Training
To train a model, use a command like below:
```
NCCL_P2P_DISABLE=1 tools/dist_train.sh ./configs/panoptic/resnet_rnn_panoptic_cam5.py <NUM_GPUS>
```

You can modify the config as needed.

## Evaluation
To evaluate a trained model, you can use the below command:

```
NCCL_P2P_DISABLE=1 tools/dist_test.sh ./configs/panoptic/resnet_rnn_panoptic_cam5.py <path/to/checkpoint> 1
```

You can download a checkpoint for the Panoptic dataset below: from the following [google drive link](https://drive.google.com/file/d/1cKKpvXVnUrdWwVBcttJjBzhE2HWT_a0P/view?usp=sharing).


## Demo

To generate demo visualization for the Panoptic dataset, run the following:
```
python3 demo.py ./configs/panoptic/demo_config.py </path/to/checkpoint> --gpu-id 0
```

You can modify the demo config to run on any chosen sequence from the dataset.

## Citation
If you use our code or models in your research, please cite our work with:
```
@inproceedings{choudhury2023tempo,
  title={TEMPO: Efficient multi-view pose estimation, tracking, and forecasting},
  author={Choudhury, Rohan and Kitani, Kris M and Jeni, L{\'a}szl{\'o} A},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={14750--14760},
  year={2023}
}