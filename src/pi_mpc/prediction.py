"""
Probabilistic obstacle trajectory prediction for DRA-MPPI.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


def _diag_process_cov(
    std_xy: torch.Tensor, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    std_xy = std_xy.to(device=device, dtype=dtype)
    if std_xy.shape != (2,):
        raise ValueError("process_noise_std must have shape (2,).")
    return torch.diag(std_xy**2)


class ConstantVelocityPredictor:
    """
    Predict pedestrian trajectories with a constant-velocity model.
    """

    def __init__(
        self,
        process_noise_std: torch.Tensor = torch.tensor([0.2, 0.2]),
        velocity_decay: float = 1.0,
    ) -> None:
        self._process_noise_std = process_noise_std
        self._velocity_decay = velocity_decay

    def predict_gaussian(
        self, pedestrian_state: torch.Tensor, horizon: int, dt: float
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pedestrian_state: (N, 4), [x, y, vx, vy]
            horizon: prediction horizon
            dt: time interval

        Returns:
            dict with keys:
                - mode: "gaussian"
                - means: (N, T, 2)
                - covariances: (N, T, 2, 2)
        """
        if pedestrian_state.ndim != 2 or pedestrian_state.shape[1] != 4:
            raise ValueError("pedestrian_state must have shape (N, 4).")

        device = pedestrian_state.device
        dtype = pedestrian_state.dtype
        num_obs = pedestrian_state.shape[0]

        pos0 = pedestrian_state[:, :2]
        vel0 = pedestrian_state[:, 2:]

        means = torch.zeros(num_obs, horizon, 2, device=device, dtype=dtype)
        covariances = torch.zeros(num_obs, horizon, 2, 2, device=device, dtype=dtype)

        process_cov = _diag_process_cov(
            self._process_noise_std, device=device, dtype=dtype
        )

        for t in range(horizon):
            step = float(t + 1)
            decay = self._velocity_decay**step
            means[:, t, :] = pos0 + vel0 * (dt * step * decay)
            covariances[:, t, :, :] = process_cov * step

        return {
            "mode": "gaussian",
            "means": means,
            "covariances": covariances,
        }

    def predict_mog(
        self,
        pedestrian_state: torch.Tensor,
        horizon: int,
        dt: float,
        mode_weights: torch.Tensor,
        mode_velocities: torch.Tensor,
        mode_covariances: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-modal prediction with mixture of Gaussians.

        Args:
            pedestrian_state: (N, 4), [x, y, vx, vy]
            mode_weights: (N, M), per-obstacle intent probabilities
            mode_velocities: (N, M, 2), per-mode velocity hypotheses
            mode_covariances: (N, M, 2, 2), optional base covariance
        Returns:
            dict with keys:
                - mode: "mog"
                - weights: (N, M)
                - means: (N, M, T, 2)
                - covariances: (N, M, T, 2, 2)
        """
        if pedestrian_state.ndim != 2 or pedestrian_state.shape[1] != 4:
            raise ValueError("pedestrian_state must have shape (N, 4).")
        if mode_weights.ndim != 2:
            raise ValueError("mode_weights must have shape (N, M).")
        if mode_velocities.ndim != 3 or mode_velocities.shape[-1] != 2:
            raise ValueError("mode_velocities must have shape (N, M, 2).")

        num_obs = pedestrian_state.shape[0]
        num_modes = mode_weights.shape[1]
        device = pedestrian_state.device
        dtype = pedestrian_state.dtype

        if mode_weights.shape[0] != num_obs:
            raise ValueError("mode_weights first dimension must match N.")
        if mode_velocities.shape[0] != num_obs or mode_velocities.shape[1] != num_modes:
            raise ValueError("mode_velocities must match mode_weights dimensions.")

        weights = mode_weights.to(device=device, dtype=dtype)
        weights = weights / torch.sum(weights, dim=1, keepdim=True).clamp_min(1e-6)

        pos0 = pedestrian_state[:, :2]
        process_cov = _diag_process_cov(
            self._process_noise_std, device=device, dtype=dtype
        )

        if mode_covariances is None:
            base_cov = process_cov.repeat(num_obs, num_modes, 1, 1)
        else:
            base_cov = mode_covariances.to(device=device, dtype=dtype)
            if base_cov.shape != (num_obs, num_modes, 2, 2):
                raise ValueError("mode_covariances must have shape (N, M, 2, 2).")

        means = torch.zeros(
            num_obs, num_modes, horizon, 2, device=device, dtype=dtype
        )
        covariances = torch.zeros(
            num_obs, num_modes, horizon, 2, 2, device=device, dtype=dtype
        )

        vel_hyp = mode_velocities.to(device=device, dtype=dtype)
        for t in range(horizon):
            step = float(t + 1)
            decay = self._velocity_decay**step
            means[:, :, t, :] = pos0.unsqueeze(1) + vel_hyp * (dt * step * decay)
            covariances[:, :, t, :, :] = base_cov * step

        return {
            "mode": "mog",
            "weights": weights,
            "means": means,
            "covariances": covariances,
        }
