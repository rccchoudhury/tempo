
import torch
import torch.nn as nn
import torch.nn.functional as F

from .center_net import P2PNet
from .center_net import P2PNet
from .fiery_temporal import SpatialGRU

class RNNFeatureEncoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.hidden_state = None
        self.encoder_net = P2PNet(in_channels, in_channels)
        self.spatial_gru = SpatialGRU(in_channels, in_channels)
        self.decode_block = P2PNet(in_channels, out_channels)

    def forward(self, x, prev_feats=None):
        assert len(x.shape) == 5
        time = x.shape[1]
        #outputs = []
        outputs = []
        self.hidden_state = prev_feats
        ###################################################################
        # HARDCODED
        if prev_feats is not None:
            time = 1

        ###################################################################
        for i in range(time):
            encoding_t = self.encoder_net(x[:, i, :, :, :])
            encoding_t = encoding_t.unsqueeze(1)
            self.hidden_state = self.spatial_gru(encoding_t,
                                                 self.hidden_state).squeeze(1)
            output = self.decode_block(self.hidden_state)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)
        return outputs, self.hidden_state


def max_pool(inputs, kernel=3):
    padding = (kernel - 1) // 2
    max = F.max_pool2d(inputs, kernel_size=kernel, stride=1, padding=padding)
    keep = (inputs == max).float()
    return keep * inputs


def get_2d_indices(indices, shape):
    """Get indices in the 2-D tensor.

    Args:
        indices (torch.Tensor(NXp)): Indices of points in the 1D tensor
        shape (torch.Size(2)): The shape of the original 3D tensor

    Returns:
        indices: Indices of points in the original 3D tensor
    """
    batch_size = indices.shape[0]
    num_people = indices.shape[1]
    indices_x = torch.div(
        indices, (shape[1]),
        rounding_mode='floor').reshape(batch_size, num_people, -1)
    indices_y = (indices % (shape[1])).reshape(batch_size, num_people, -1)
    indices = torch.cat([indices_x, indices_y], dim=2)
    return indices


def nms_by_max_pool(heatmap_volumes, max_num=10):
    batch_size = heatmap_volumes.shape[0]
    root_cubes_nms = max_pool(heatmap_volumes)
    root_cubes_nms_reshape = root_cubes_nms.reshape(batch_size, -1)
    topk_values, topk_flatten_index = root_cubes_nms_reshape.topk(max_num)
    topk_unravel_index = get_2d_indices(topk_flatten_index,
                                        heatmap_volumes[0].shape)
    return topk_values, topk_unravel_index, topk_flatten_index


def nms_3d(center_candidates, confidence_thresh, distance_thresh):
    batch_size = center_candidates.shape[0]
    num_people = center_candidates.shape[1]
    # Set all invalid ones to -1.
    center_candidates[...,
                      3] = (center_candidates[..., 4] < confidence_thresh) * -1
    # Compute distances between each other.
    mutual_dists = torch.cdist(center_candidates[..., :3],
                               center_candidates[..., :3])
    for batch_idx in range(batch_size):
        for person_idx in range(num_people):
            if center_candidates[batch_idx, person_idx, 3] == -1:
                continue
            close = torch.logical_and(
                mutual_dists[batch_idx, person_idx] <= distance_thresh,
                mutual_dists[batch_idx, person_idx] > 0)
            # valid / invalid.
            center_candidates[batch_idx, :, 3][close] = -1

    return center_candidates
