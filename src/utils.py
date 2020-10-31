import numpy as np
import torch
from torch import Tensor
from l5kit.geometry import transform_points


def pytorch_neg_multi_log_likelihood_batch(
    gt: Tensor,
    pred: Tensor,
    confidences: Tensor,
    avails: Tensor,
) -> Tensor:
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (
        batch_size,
        future_len,
        num_coords,
    ), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (
        batch_size,
        num_modes,
    ), f"expected 1D (Modes) array for confidences, got {confidences.shape}"
    assert torch.allclose(
        torch.sum(confidences, dim=1), confidences.new_ones((batch_size,))
    ), "confidences should sum to 1, got: " + torch.sum(confidences, dim=1) + confidences 
    assert avails.shape == (
        batch_size,
        future_len,
    ), f"expected 1D (Time) array for avails, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(gt).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(
        ((gt - pred) * avails) ** 2, dim=-1
    )  # reduce coords and use availability

    with np.errstate(
        divide="ignore"
    ):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value, _ = error.max(
        dim=1, keepdim=True
    )  # error are negative at this point, so max() gives the minimum one
    error = (
        -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True))
        - max_value
    )  # reduce modes
    # print("error", error)
    return torch.mean(error)


def convert_agent_coordinates_to_world_offsets(
    agents_coords: np.ndarray,
    world_from_agents: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    coords_offset = []
    for agent_coords, world_from_agent, centroid in zip(
        agents_coords, world_from_agents, centroids
    ):
        predition_offset = []
        for agent_coord in agent_coords:
            predition_offset.append(
                transform_points(agent_coord, world_from_agent) - centroid[:2]
            )
        predition_offset = np.stack(predition_offset)
        coords_offset.append(predition_offset)
    return np.stack(coords_offset)
