"""
Distributionally Risk-Aware MPPI (DRA-MPPI).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from scipy.optimize import brentq, minimize_scalar

from pi_mpc.mppi import MPPI


class DRAMPPI(MPPI):
    def __init__(
        self,
        *args,
        num_mc_samples: int = 192,
        collision_radius: float = 0.6,
        w_soft: float = 40.0,
        w_hard: float = 5000.0,
        risk_threshold: float = 0.2,
        cov_epsilon: float = 1e-5,
        mc_seed: int = 123,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._num_mc_samples = num_mc_samples
        self._collision_radius = collision_radius
        self._collision_area = math.pi * (collision_radius**2)
        self._w_soft = w_soft
        self._w_hard = w_hard
        self._risk_threshold = risk_threshold
        self._cov_epsilon = cov_epsilon

        self._predicted_obstacles: Optional[Dict[str, torch.Tensor]] = None
        self._latest_rollout_collision_prob: Optional[torch.Tensor] = None
        self._latest_best_collision_profile: Optional[torch.Tensor] = None

        self._mc_offsets = self._sample_mc_offsets(seed=mc_seed)

    def _sample_mc_offsets(self, seed: int) -> torch.Tensor:
        generator = torch.Generator(device=self._device)
        generator.manual_seed(seed)

        radius = torch.sqrt(
            torch.rand(
                self._num_mc_samples,
                device=self._device,
                dtype=self._dtype,
                generator=generator,
            )
        )
        angle = 2.0 * torch.pi * torch.rand(
            self._num_mc_samples,
            device=self._device,
            dtype=self._dtype,
            generator=generator,
        )
        offset_x = self._collision_radius * radius * torch.cos(angle)
        offset_y = self._collision_radius * radius * torch.sin(angle)
        return torch.stack([offset_x, offset_y], dim=-1)

    def set_predicted_obstacles(self, predicted_obstacles: Dict[str, torch.Tensor]) -> None:
        self._predicted_obstacles = predicted_obstacles

    def get_collision_prob_profile(self) -> Optional[torch.Tensor]:
        return self._latest_best_collision_profile

    def get_mc_points_for_trajectory(self, state_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_seq: (B, T+1, state_dim) or (T+1, state_dim)
        Returns:
            (B, T, N_mc, 2) MC points around each robot state.
        """
        if state_seq.ndim == 2:
            state_seq = state_seq.unsqueeze(0)
        robot_xy = state_seq[:, 1:, :2]
        return robot_xy.unsqueeze(2) + self._mc_offsets.view(1, 1, self._num_mc_samples, 2)

    def _gaussian_point_probability(
        self, points: torch.Tensor, means: torch.Tensor, covariances: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            points: (K, N_mc, 2)
            means: (N_obs, 2)
            covariances: (N_obs, 2, 2)
        Returns:
            (K, N_mc, N_obs), per-obstacle collision probabilities.
        """
        num_obs = means.shape[0]
        eye = torch.eye(2, device=self._device, dtype=self._dtype).unsqueeze(0)
        cov = covariances + self._cov_epsilon * eye
        inv_cov = torch.linalg.inv(cov)
        det_cov = torch.linalg.det(cov).clamp_min(self._cov_epsilon)

        diff = points.unsqueeze(2) - means.view(1, 1, num_obs, 2)
        mahal = torch.einsum("kmoi,oij,kmoj->kmo", diff, inv_cov, diff)
        norm_const = 2.0 * torch.pi * torch.sqrt(det_cov)
        pdf = torch.exp(-0.5 * mahal) / norm_const.view(1, 1, num_obs)

        return torch.clamp(pdf * self._collision_area, 0.0, 1.0)

    def _estimate_collision_prob_gaussian(
        self, points: torch.Tensor, means_t: torch.Tensor, covariances_t: torch.Tensor
    ) -> torch.Tensor:
        p_obs = self._gaussian_point_probability(points, means_t, covariances_t)
        p_joint = 1.0 - torch.prod(1.0 - p_obs, dim=2)
        return torch.mean(p_joint, dim=1)

    def _estimate_collision_prob_mog(
        self,
        points: torch.Tensor,
        weights: torch.Tensor,
        means_t: torch.Tensor,
        covariances_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            points: (K, N_mc, 2)
            weights: (N_obs, M)
            means_t: (N_obs, M, 2)
            covariances_t: (N_obs, M, 2, 2)
        """
        num_obs, num_modes = weights.shape
        p_obs = torch.zeros(
            points.shape[0], points.shape[1], num_obs, device=self._device, dtype=self._dtype
        )
        for mode_idx in range(num_modes):
            p_mode = self._gaussian_point_probability(
                points, means_t[:, mode_idx, :], covariances_t[:, mode_idx, :, :]
            )
            p_obs += weights[:, mode_idx].view(1, 1, num_obs) * p_mode

        p_obs = torch.clamp(p_obs, 0.0, 1.0)
        p_joint = 1.0 - torch.prod(1.0 - p_obs, dim=2)
        return torch.mean(p_joint, dim=1)

    def _compute_rollout_risk_costs(
        self, rollout_state_batch: torch.Tensor, predicted_obstacles: Optional[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Args:
            rollout_state_batch: (K, T, state_dim)
            predicted_obstacles: dict from prediction.py
        Returns:
            (K,) risk-aware cost.
        """
        num_samples, horizon, _ = rollout_state_batch.shape
        total_risk_cost = torch.zeros(num_samples, device=self._device, dtype=self._dtype)
        collision_profile = torch.zeros(
            num_samples, horizon, device=self._device, dtype=self._dtype
        )

        if predicted_obstacles is None:
            self._latest_rollout_collision_prob = collision_profile
            return total_risk_cost

        mode = predicted_obstacles.get("mode", "gaussian")

        if mode == "gaussian":
            means = predicted_obstacles["means"].to(self._device, self._dtype)
            covariances = predicted_obstacles["covariances"].to(self._device, self._dtype)
            max_t = min(horizon, means.shape[1])
            for t in range(max_t):
                robot_xy = rollout_state_batch[:, t, :2]
                points = robot_xy.unsqueeze(1) + self._mc_offsets.unsqueeze(0)
                collision_prob = self._estimate_collision_prob_gaussian(
                    points, means[:, t, :], covariances[:, t, :, :]
                )

                step_risk = self._w_soft * collision_prob
                step_risk += self._w_hard * (collision_prob > self._risk_threshold).float()
                total_risk_cost += step_risk
                collision_profile[:, t] = collision_prob

        elif mode == "mog":
            weights = predicted_obstacles["weights"].to(self._device, self._dtype)
            means = predicted_obstacles["means"].to(self._device, self._dtype)
            covariances = predicted_obstacles["covariances"].to(self._device, self._dtype)
            max_t = min(horizon, means.shape[2])
            for t in range(max_t):
                robot_xy = rollout_state_batch[:, t, :2]
                points = robot_xy.unsqueeze(1) + self._mc_offsets.unsqueeze(0)
                collision_prob = self._estimate_collision_prob_mog(
                    points, weights, means[:, :, t, :], covariances[:, :, t, :, :]
                )

                step_risk = self._w_soft * collision_prob
                step_risk += self._w_hard * (collision_prob > self._risk_threshold).float()
                total_risk_cost += step_risk
                collision_profile[:, t] = collision_prob

        else:
            raise ValueError(f"Unsupported predicted obstacle mode: {mode}")

        self._latest_rollout_collision_prob = collision_profile
        return total_risk_cost

    def forward(
        self, state: torch.Tensor, info: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert state.shape == (self._dim_state,)

        if not torch.is_tensor(state):
            state = torch.tensor(state, device=self._device, dtype=self._dtype)
        else:
            if state.device != self._device or state.dtype != self._dtype:
                state = state.to(self._device, self._dtype)

        if info is None:
            info = {}
        else:
            info = dict(info)

        if "predicted_obstacles" not in info and self._predicted_obstacles is not None:
            info["predicted_obstacles"] = self._predicted_obstacles

        mean_action_seq = self._previous_action_seq.clone().detach()

        self._action_noises = self._noise_distribution.rsample(sample_shape=self._sample_shape)

        threshold = int(self._num_samples * (1 - self._exploration))
        inherited_samples = mean_action_seq + self._action_noises[:threshold]
        self._perturbed_action_seqs = torch.cat(
            [inherited_samples, self._action_noises[threshold:]]
        )
        self._perturbed_action_seqs = torch.clamp(
            self._perturbed_action_seqs, self._u_min, self._u_max
        )

        self._state_seq_batch[:, 0, :] = state.repeat(self._num_samples, 1)
        for t in range(self._horizon):
            self._state_seq_batch[:, t + 1, :] = self._dynamics(
                self._state_seq_batch[:, t, :], self._perturbed_action_seqs[:, t, :]
            )

        stage_costs = torch.zeros(
            self._num_samples, self._horizon, device=self._device, dtype=self._dtype
        )
        initial_state = self._state_seq_batch[:, 0, :]
        for t in range(self._horizon):
            prev_index = t - 1 if t > 0 else 0
            prev_state = self._state_seq_batch[:, prev_index, :]
            prev_action = self._perturbed_action_seqs[:, prev_index, :]
            info["prev_state"] = prev_state
            info["prev_action"] = prev_action
            info["initial_state"] = initial_state
            info["t"] = t

            stage_costs[:, t] = self._cost_func(
                self._state_seq_batch[:, t, :], self._perturbed_action_seqs[:, t, :], info
            )

        info["prev_state"] = self._state_seq_batch[:, -2, :]
        zero_action = torch.zeros(
            self._num_samples, self._dim_control, device=self._device, dtype=self._dtype
        )
        terminal_costs = self._cost_func(self._state_seq_batch[:, -1, :], zero_action, info)

        risk_costs = self._compute_rollout_risk_costs(
            self._state_seq_batch[:, 1:, :], info.get("predicted_obstacles")
        )

        costs = (
            torch.sum(stage_costs, dim=1)
            + terminal_costs
            + risk_costs
            # + torch.sum(self._lambda * action_costs, dim=1)
        )

        if self._auto_lambda == "LBPS":
            result = minimize_scalar(
                lambda lambda_: self._lbps_objective(lambda_, costs.detach()),
                bounds=(self._lambda_min, self._lambda_max),
                method="bounded",
            )
            self._lambda = result.x

        elif self._auto_lambda == "ESSPS":
            ess_at_min = self._compute_ess(
                torch.softmax(-costs.detach() / self._lambda_min, dim=0)
            )
            ess_at_max = self._compute_ess(
                torch.softmax(-costs.detach() / self._lambda_max, dim=0)
            )

            if self._essps_target_ess <= ess_at_min:
                self._lambda = self._lambda_min
            elif self._essps_target_ess >= ess_at_max:
                self._lambda = self._lambda_max
            else:
                self._lambda = brentq(
                    lambda lambda_: self._essps_objective(lambda_, costs.detach()),
                    self._lambda_min,
                    self._lambda_max,
                )

        self._weights = torch.softmax(-costs / self._lambda, dim=0)
        optimal_action_seq = torch.sum(
            self._weights.view(self._num_samples, 1, 1) * self._perturbed_action_seqs, dim=0
        )

        if self._auto_lambda == "MPO":
            for _ in range(1):
                self.optimizer.zero_grad()
                temperature = torch.nn.functional.softplus(self.log_temperature)
                cost_logsumexp = torch.logsumexp(-costs / temperature, dim=0)
                loss = temperature * (self._mpo_epsilon + torch.mean(cost_logsumexp))
                loss.backward()
                self.optimizer.step()
            self._lambda = torch.exp(self.log_temperature).item()

        if self._use_sg_filter:
            prolonged_action_seq = torch.cat(
                [self._actions_history_for_sg, optimal_action_seq], dim=0
            )
            filtered_action_seq = torch.zeros_like(
                prolonged_action_seq, device=self._device, dtype=self._dtype
            )
            for i in range(self._dim_control):
                filtered_action_seq[:, i] = self._apply_savitzky_golay(
                    prolonged_action_seq[:, i], self._coeffs
                )
            optimal_action_seq = filtered_action_seq[-self._horizon :]

        optimal_state_seq = self._states_prediction(
            state, optimal_action_seq.repeat(1, 1, 1)
        )

        self._previous_action_seq = optimal_action_seq
        self._actions_history_for_sg = torch.cat(
            [self._actions_history_for_sg[1:], optimal_action_seq[0].view(1, -1)]
        )

        if self._latest_rollout_collision_prob is not None:
            best_idx = torch.argmax(self._weights)
            self._latest_best_collision_profile = self._latest_rollout_collision_prob[
                best_idx
            ].detach()
        else:
            self._latest_best_collision_profile = None

        return optimal_action_seq, optimal_state_seq
