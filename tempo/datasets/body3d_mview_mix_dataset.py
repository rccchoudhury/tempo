# Copyright (c) OpenMMLab. All rights reserved.
import pdb
import sys
from abc import ABCMeta

import numpy as np
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler

from mmpose.datasets.builder import DATASETS
# from .body3d_mview_temporal_egoexo_dataset import \
#     Body3DMviewTemporalEgoExoDataset
from .body3d_mview_temporal_human36m import Body3DMviewTemporalH36MDataset
from .body3d_mview_temporal_panoptic_dataset import Body3DMviewTemporalPanopticDataset


@DATASETS.register_module()
class Body3DMviewMixDataset(Dataset, metaclass=ABCMeta):
    """Mix Dataset for multi-view human keypoint estimation.

    The dataset combines data from multiple datasets (Kpt3dMviewRgbImgDirectDataset) and
    sample the data from different datasets with the provided proportions.
    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    Component datasets can provide their input in different formats, but
    will be internally converted to the same format.

    Args:
        configs (list): List of configs for multiple datasets.
        partition (list): Sample proportion of multiple datasets. The length
            of partition should be same with that of configs. The elements
            of it should be non-negative and is not necessary summing up to
            one.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.

    Example:
    """

    def __init__(self, configs, partition, test_mode=False):
        """Load data from multiple datasets."""
        assert min(partition) >= 0
        self.all_datasets = []
        # TODO: do this with base classes as in MeshMixDataset.
        for cfg in configs:
            dataset_name = cfg.pop('type')
            print('loading dataset: {}'.format(dataset_name))
            self.all_datasets.append(eval(dataset_name + '(**cfg)'))
        self.dataset = ConcatDataset(self.all_datasets)

        weights = [
            np.ones(len(ds)) * p / len(ds)
            for (p, ds) in zip(partition, self.all_datasets)
        ]
        weights = np.concatenate(weights, axis=0)

        self.length = max(len(ds) for ds in self.all_datasets)
        self.sampler = WeightedRandomSampler(weights, 1)

    def __len__(self):
        """Get the size of the dataset."""
        return self.length

    def __getitem__(self, idx):
        """Given index, sample the data from multiple datasets with the given
        proportion."""
        idx_new = list(self.sampler)[0]
        
        return self.dataset[idx_new]

    # Need to write evaluate methods.
    # Extend this to support multiple datasets.
    def evaluate(self, results, *args, **kwargs):
        eval_output = self.dataset.datasets[0].evaluate(
            results, *args, **kwargs)

        return eval_output
