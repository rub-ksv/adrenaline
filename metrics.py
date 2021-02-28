import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from utils import compute_angular_distance


def frame_recall(predictions: torch.Tensor,
                 targets: torch.Tensor) -> torch.Tensor:
    """Frame-recall metric, describing the percentage of frames where the number of predicted sources matches the number
    of sources provided in the ground-truth data. For additional information, refer to e.g.

        Adavanne et al.: "A multi-room reverberant dataset for sound event localization and detection" (2019)

    :param predictions: predicted source activities and doas
    :param targets: ground-truth source activities and doas
    :return: frame recall
    """
    predicted_source_activity = predictions[0].cpu()
    target_source_activity = targets[0].cpu()

    predicted_num_active_sources = torch.sum(torch.sigmoid(predicted_source_activity) > 0.5, dim=-1)
    target_num_active_sources = torch.sum(target_source_activity, dim=-1)

    frame_recall = torch.mean((predicted_num_active_sources == target_num_active_sources).float())

    return frame_recall


def doa_error(predictions: torch.Tensor,
              targets: torch.Tensor) -> torch.Tensor:
    batch_size, num_time_steps, _ = predictions[0].shape

    doa_error_matrix = np.zeros((batch_size, num_time_steps))

    for batch_idx in range(batch_size):
        for step_idx in range(num_time_steps):
            predicted_source_activity = (torch.sigmoid(predictions[0][batch_idx, step_idx, :]) > 0.5).detach().cpu().numpy()
            predicted_direction_of_arrival = predictions[1][batch_idx, step_idx, :, :].detach().cpu().numpy()
            target_source_activity = targets[0][batch_idx, step_idx, :].bool().detach().cpu().numpy()
            target_direction_of_arrival = targets[1][batch_idx, step_idx, :, :].detach().cpu().numpy()

            predicted_sources = predicted_direction_of_arrival[predicted_source_activity, :]
            num_predicted_sources = predicted_sources.shape[0]
            target_sources = target_direction_of_arrival[target_source_activity, :]
            num_target_sources = target_sources.shape[0]

            if (num_predicted_sources > 0) and (num_target_sources > 0):
                cost_matrix = np.zeros((num_predicted_sources, num_target_sources))

                for pred_idx in range(num_predicted_sources):
                    for target_idx in range(num_target_sources):
                        cost_matrix[pred_idx, target_idx] = compute_angular_distance(
                            predicted_sources[pred_idx, :], target_sources[target_idx, :])

                row_idx, col_idx = linear_sum_assignment(cost_matrix)
                doa_error_matrix[batch_idx, step_idx] = np.rad2deg(cost_matrix[row_idx, col_idx].mean())
            else:
                doa_error_matrix[batch_idx, step_idx] = np.nan

    return torch.tensor(np.nanmean(doa_error_matrix, dtype=np.float32))
