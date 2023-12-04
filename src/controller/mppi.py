"""
Kohei Honda, 2023.
"""

from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal


class MPPI(nn.Module):
    """
    Model Predictive Path Integral Control,
    J. Williams et al., T-RO, 2017.
    """

    def __init__(
        self,
        horizon: int,
        num_samples: int,
        dim_state: int,
        dim_control: int,
        dynamics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        stage_cost: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        terminal_cost: Callable[[torch.Tensor], torch.Tensor],
        u_min: torch.Tensor,
        u_max: torch.Tensor,
        sigmas: torch.Tensor,
        lambda_: float,
        device=torch.device("cuda"),
        dtype=torch.float32,
        seed: int = 42,
    ) -> None:
        """
        :param horizon: Predictive horizon length.
        :param delta: predictive horizon step size (seconds).
        :param num_samples: Number of samples.
        :param dim_state: Dimension of state.
        :param dim_control: Dimension of control.
        :param dynamics: Dynamics model.
        :param stage_cost: Stage cost.
        :param terminal_cost: Terminal cost.
        :param u_min: Minimum control.
        :param u_max: Maximum control.
        :param sigmas: Noise standard deviation for each control dimension.
        :param lambda_: temperature parameter.
        :param device: Device to run the solver.
        :param dtype: Data type to run the solver.
        :param seed: Seed for torch.
        """

        super().__init__()

        # torch seed
        torch.manual_seed(seed)

        # check dimensions
        assert u_min.shape == (dim_control,)
        assert u_max.shape == (dim_control,)
        assert sigmas.shape == (dim_control,)
        # assert num_samples % batch_size == 0 and num_samples >= batch_size

        # device and dtype
        if torch.cuda.is_available() and device == torch.device("cuda"):
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        self._dtype = dtype

        # set parameters
        self._horizon = horizon
        self._num_samples = num_samples
        self._dim_state = dim_state
        self._dim_control = dim_control
        self._dynamics = dynamics
        self._stage_cost = stage_cost
        self._terminal_cost = terminal_cost
        self._u_min = u_min.clone().detach().to(self._device, self._dtype)
        self._u_max = u_max.clone().detach().to(self._device, self._dtype)
        self._sigmas = sigmas.clone().detach().to(self._device, self._dtype)
        self._lambda = lambda_

        # noise distribution
        zero_mean = torch.zeros(dim_control, device=self._device, dtype=self._dtype)
        initial_covariance = torch.diag(sigmas**2).to(self._device, self._dtype)
        self._inv_covariance = torch.inverse(initial_covariance).to(
            self._device, self._dtype
        )

        self._noise_distribution = MultivariateNormal(
            loc=zero_mean, covariance_matrix=initial_covariance
        )
        self._sample_shape = torch.Size([self._num_samples, self._horizon])

        # sampling with reparameting trick
        self._action_noises = self._noise_distribution.rsample(
            sample_shape=self._sample_shape
        )

        zero_mean_seq = torch.zeros(
            self._horizon, self._dim_control, device=self._device, dtype=self._dtype
        )
        self._perturbed_action_seqs = torch.clamp(
            zero_mean_seq + self._action_noises, self._u_min, self._u_max
        )

        self._previous_action_seq = zero_mean_seq

        # inner variables
        self._state_seq_batch = torch.zeros(
            self._num_samples,
            self._horizon + 1,
            self._dim_state,
            device=self._device,
            dtype=self._dtype,
        )
        self._weights = torch.zeros(
            self._num_samples, device=self._device, dtype=self._dtype
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve the optimal control problem.
        Args:
            state (torch.Tensor): Current state.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of predictive control and state sequence.
        """
        assert state.shape == (self._dim_state,)

        if not torch.is_tensor(state):
            state = torch.tensor(state, device=self._device, dtype=self._dtype)
        else:
            if state.device != self._device or state.dtype != self._dtype:
                state = state.to(self._device, self._dtype)

        mean_action_seq = self._previous_action_seq.clone().detach()

        # random sampling with reparametrization trick
        self._action_noises = self._noise_distribution.rsample(
            sample_shape=self._sample_shape
        )
        self._perturbed_action_seqs = mean_action_seq + self._action_noises

        # clamp actions
        self._perturbed_action_seqs = torch.clamp(
            self._perturbed_action_seqs, self._u_min, self._u_max
        )

        # rollout samples in parallel
        self._state_seq_batch[:, 0, :] = state.repeat(self._num_samples, 1)

        for t in range(self._horizon):
            self._state_seq_batch[:, t + 1, :] = self._dynamics(
                self._state_seq_batch[:, t, :], self._perturbed_action_seqs[:, t, :]
            )

        # compute sample costs
        stage_costs = torch.zeros(
            self._num_samples, self._horizon, device=self._device, dtype=self._dtype
        )
        action_costs = torch.zeros(
            self._num_samples, self._horizon, device=self._device, dtype=self._dtype
        )
        for t in range(self._horizon):
            stage_costs[:, t] = self._stage_cost(
                self._state_seq_batch[:, t, :], self._perturbed_action_seqs[:, t, :]
            )
            action_costs[:, t] = (
                mean_action_seq[t]
                @ self._inv_covariance
                @ self._perturbed_action_seqs[:, t].T
            )

        terminal_costs = self._terminal_cost(self._state_seq_batch[:, -1, :])

        costs = (
            torch.sum(stage_costs, dim=1)
            + terminal_costs
            + torch.sum(self._lambda * action_costs, dim=1)
        )

        # calculate weights
        self._weights = torch.softmax(-costs / self._lambda, dim=0)

        # find optimal control by weighted average
        optimal_action_seq = torch.sum(
            self._weights.view(self._num_samples, 1, 1) * self._perturbed_action_seqs,
            dim=0,
        )

        # predivtive state seq
        optimal_state_seq = torch.zeros(
            1,
            self._horizon + 1,
            self._dim_state,
            device=self._device,
            dtype=self._dtype,
        )
        optimal_state_seq[:, 0, :] = state
        expanded_optimal_action_seq = optimal_action_seq.repeat(1, 1, 1)
        for t in range(self._horizon):
            optimal_state_seq[:, t + 1, :] = self._dynamics(
                optimal_state_seq[:, t, :], expanded_optimal_action_seq[:, t, :]
            )

        # update previous actions
        self._previous_action_seq = optimal_action_seq

        return optimal_action_seq, optimal_state_seq

    def get_top_samples(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top samples.
        Args:
            num_samples (int): Number of state samples to get.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of top samples and their weights.
        """
        assert num_samples <= self._num_samples

        # large weights are better
        top_indices = torch.topk(self._weights, num_samples).indices

        top_samples = self._state_seq_batch[top_indices]
        top_weights = self._weights[top_indices]

        top_samples = top_samples[torch.argsort(top_weights, descending=True)]
        top_weights = top_weights[torch.argsort(top_weights, descending=True)]

        return top_samples, top_weights
