from itertools import permutations
import numpy as np
import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from typing import Tuple


class SELLoss(_Loss):
    """Custom sound event localization (SEL) loss function, which returns the sum of the binary cross-entropy loss
    regarding the estimated number of sources at each time-step and the minimum direction-of-arrival mean squared error
    loss, calculated according to all possible combinations of active sources."""

    __constants__ = ['reduction']

    def __init__(self,
                 max_num_sources: int,
                 alpha: float = 1.0,
                 size_average=None,
                 reduce=None,
                 reduction='mean') -> None:
        super(SELLoss, self).__init__(size_average, reduce, reduction)

        if (alpha < 0) or (alpha > 1):
            assert ValueError('The weighting parameter must be a number between 0 and 1.')

        self.alpha = alpha
        self.permutations = torch.from_numpy(np.array(list(permutations(range(max_num_sources)))))
        self.num_permutations = self.permutations.shape[0]

    @staticmethod
    def compute_spherical_distance(y_pred: torch.Tensor,
                                   y_true: torch.Tensor) -> torch.Tensor:
        if (y_pred.shape[-1] != 2) or (y_true.shape[-1] != 2):
            assert RuntimeError('Input tensors require a dimension of two.')

        sine_term = torch.sin(y_pred[:, 0]) * torch.sin(y_true[:, 0])
        cosine_term = torch.cos(y_pred[:, 0]) * torch.cos(y_true[:, 0]) * torch.cos(y_true[:, 1] - y_pred[:, 1])

        return torch.acos(F.hardtanh(sine_term + cosine_term, min_val=-1, max_val=1))

    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        source_activity_pred, direction_of_arrival_pred, _ = predictions
        source_activity_target, direction_of_arrival_target = targets

        source_activity_bce_loss = F.binary_cross_entropy_with_logits(source_activity_pred, source_activity_target)

        source_activity_mask = source_activity_target.bool()

        spherical_distance = self.compute_spherical_distance(
            direction_of_arrival_pred[source_activity_mask], direction_of_arrival_target[source_activity_mask])
        direction_of_arrival_loss = self.alpha * torch.mean(spherical_distance)

        loss = source_activity_bce_loss + direction_of_arrival_loss

        meta_data = {
            'source_activity_loss': source_activity_bce_loss,
            'direction_of_arrival_loss': direction_of_arrival_loss
        }

        return loss, meta_data


def compute_angular_distance(x, y):
    """Computes the angle between two spherical direction-of-arrival points.

    :param x: single direction-of-arrival, where the first column is the azimuth and second column is elevation
    :param y: single or multiple DoAs, where the first column is the azimuth and second column is elevation
    :return: angular distance
    """
    if np.ndim(x) != 1:
        raise ValueError('First DoA must be a single value.')

    return np.arccos(np.sin(x[0]) * np.sin(y[0]) + np.cos(x[0]) * np.cos(y[0]) * np.cos(y[1] - x[1]))


def get_num_params(model):
    """Returns the number of trainable parameters of a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
