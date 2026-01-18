"""
Kohei Honda, 2023.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from scipy.optimize import brentq, minimize_scalar
from torch.distributions.multivariate_normal import MultivariateNormal


class MPPI(nn.Module):
    """Model Predictive Path Integral Control (MPPI) solver.

    Reference:
        Williams et al., "Information Theoretic MPC for Model-Based
        Reinforcement Learning", IEEE T-RO, 2017.
    """

    def __init__(
        self,
        horizon: int,
        num_samples: int,
        dim_state: int,
        dim_control: int,
        dynamics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        cost_func: Callable[[torch.Tensor, torch.Tensor, Dict], torch.Tensor],
        u_min: torch.Tensor,
        u_max: torch.Tensor,
        sigmas: torch.Tensor,
        lambda_: float | str,
        lbps_delta: float = 0.01,
        essps_target_ess: Optional[float] = None,
        lambda_min: float = 0.01,
        lambda_max: float = 10.0,
        exploration: float = 0.0,
        use_sg_filter: bool = False,
        sg_window_size: int = 5,
        sg_poly_order: int = 3,
        device=torch.device("cuda"),
        dtype=torch.float32,
        seed: int = 42,
    ) -> None:
        """Initialize MPPI controller.

        Args:
            horizon: Predictive horizon length (number of timesteps).
            num_samples: Number of trajectory samples to generate.
            dim_state: Dimension of state vector.
            dim_control: Dimension of control input vector.
            dynamics: Dynamics model function: (state, action) -> next_state.
            cost_func: Cost function: (state, action, info) -> cost.
            u_min: Minimum control bounds, shape (dim_control,).
            u_max: Maximum control bounds, shape (dim_control,).
            sigmas: Noise standard deviation for each control dimension,
                shape (dim_control,).
            lambda_: Temperature parameter (float) or auto-tuning method
                ('MPO', 'LBPS', 'ESSPS').

        Auto-lambda args:
            lbps_delta: Confidence parameter for LBPS (0 < delta < 1).
                Higher values give tighter bounds. Default: 0.01.
            essps_target_ess: Target effective sample size for ESSPS.
                Default: num_samples / 10.
            lambda_min: Minimum lambda for optimization bounds. Default: 0.01.
            lambda_max: Maximum lambda for optimization bounds. Default: 10.0.

        Sampling args:
            exploration: Fraction of purely random samples (0 to 1).
                Default: 0.0 (all samples inherit from previous solution).

        Filtering args:
            use_sg_filter: Apply Savitzky-Golay filter for smoothing.
                Default: False.
            sg_window_size: Window size for SG filter (must be odd).
                Larger values give smoother output. Default: 5.
            sg_poly_order: Polynomial order for SG filter.
                Smaller values give smoother output. Default: 3.

        Device args:
            device: Torch device ('cuda' or 'cpu'). Default: 'cuda'.
            dtype: Torch data type. Default: torch.float32.
            seed: Random seed for reproducibility. Default: 42.
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
        self._cost_func = cost_func
        self._u_min = u_min.clone().detach().to(self._device, self._dtype)
        self._u_max = u_max.clone().detach().to(self._device, self._dtype)
        self._sigmas = sigmas.clone().detach().to(self._device, self._dtype)
        self._exploration = exploration
        self._use_sg_filter = use_sg_filter
        self._sg_window_size = sg_window_size
        self._sg_poly_order = sg_poly_order

        # noise distribution
        self._covariance = torch.zeros(
            self._horizon,
            self._dim_control,
            self._dim_control,
            device=self._device,
            dtype=self._dtype,
        )
        self._covariance[:, :, :] = torch.diag(sigmas**2).to(self._device, self._dtype)
        self._inv_covariance = torch.zeros_like(
            self._covariance, device=self._device, dtype=self._dtype
        )
        for t in range(1, self._horizon):
            self._inv_covariance[t] = torch.inverse(self._covariance[t])

        zero_mean = torch.zeros(dim_control, device=self._device, dtype=self._dtype)
        self._noise_distribution = MultivariateNormal(
            loc=zero_mean, covariance_matrix=self._covariance
        )

        self._sample_shape = torch.Size([self._num_samples])

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

        # Initialize Savitzky-Golay filter
        self._coeffs = self._savitzky_golay_coeffs(
            self._sg_window_size, self._sg_poly_order
        )
        self._actions_history_for_sg = torch.zeros(
            self._horizon - 1, self._dim_control, device=self._device, dtype=self._dtype
        )  # previous inputted actions for sg filter

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
        self._optimal_state_seq = torch.zeros(
            self._horizon + 1, self._dim_state, device=self._device, dtype=self._dtype
        )

        # auto lambda tuning
        self._lambda: float | str = lambda_
        self._lbps_delta = lbps_delta
        self._essps_target_ess = (
            essps_target_ess if essps_target_ess is not None else num_samples / 10
        )
        self._lambda_min = lambda_min
        self._lambda_max = lambda_max

        if self._lambda == "MPO":
            self._auto_lambda = "MPO"
            self._lambda = 1.0  # initial value
            self._mpo_epsilon = 0.1
            self.log_temperature = torch.nn.Parameter(
                torch.log(
                    torch.tensor([self._lambda], device=self._device, dtype=self._dtype)
                )
            )
            self.optimizer = torch.optim.Adam([self.log_temperature], lr=0.2)
        elif self._lambda == "LBPS":
            self._auto_lambda = "LBPS"
        elif self._lambda == "ESSPS":
            self._auto_lambda = "ESSPS"
        elif isinstance(self._lambda, float):
            self._auto_lambda = None
        else:
            raise ValueError(
                "lambda_ must be 'MPO', 'LBPS', 'ESSPS', or a float value."
            )

    def reset(self):
        """
        Reset the previous action sequence.
        """
        self._previous_action_seq = torch.zeros(
            self._horizon, self._dim_control, device=self._device, dtype=self._dtype
        )
        self._actions_history_for_sg = torch.zeros(
            self._horizon - 1, self._dim_control, device=self._device, dtype=self._dtype
        )  # previous inputted actions for sg filter

    def forward(
        self, state: torch.Tensor, info: Dict = {}
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Solve the optimal control problem for one timestep.

        Algorithm steps:
            1. Sample action sequences around previous solution
            2. Rollout dynamics for all samples in parallel
            3. Compute trajectory costs (stage + terminal)
            4. Update temperature parameter (if auto-lambda enabled)
            5. Compute importance weights using softmax
            6. Compute optimal action as weighted average
            7. Apply smoothing filter (if enabled)

        Args:
            state: Current state vector, shape (dim_state,).
            info: Optional dictionary for passing additional context
                to the cost function (e.g., reference trajectory).

        Returns:
            Tuple of (optimal_action_seq, optimal_state_seq):
                - optimal_action_seq: Shape (horizon, dim_control)
                - optimal_state_seq: Shape (horizon + 1, dim_state)
        """
        assert state.shape == (self._dim_state,)

        if not torch.is_tensor(state):
            state = torch.tensor(state, device=self._device, dtype=self._dtype)
        else:
            if state.device != self._device or state.dtype != self._dtype:
                state = state.to(self._device, self._dtype)

        mean_action_seq = self._previous_action_seq.clone().detach()

        # =================================================================
        # Step 1: Sample action sequences
        # =================================================================
        # Generate noise samples using reparametrization trick
        self._action_noises = self._noise_distribution.rsample(
            sample_shape=self._sample_shape
        )

        # Split samples: (1-exploration) inherit from previous, rest are random
        threshold = int(self._num_samples * (1 - self._exploration))
        inherited_samples = mean_action_seq + self._action_noises[:threshold]
        self._perturbed_action_seqs = torch.cat(
            [inherited_samples, self._action_noises[threshold:]]
        )

        # Enforce control limits
        self._perturbed_action_seqs = torch.clamp(
            self._perturbed_action_seqs, self._u_min, self._u_max
        )

        # =================================================================
        # Step 2: Rollout dynamics for all samples
        # =================================================================
        self._state_seq_batch[:, 0, :] = state.repeat(self._num_samples, 1)

        for t in range(self._horizon):
            self._state_seq_batch[:, t + 1, :] = self._dynamics(
                self._state_seq_batch[:, t, :],
                self._perturbed_action_seqs[:, t, :],
            )

        # =================================================================
        # Step 3: Compute trajectory costs
        # =================================================================
        costs = torch.zeros(
            self._num_samples, self._horizon, device=self._device, dtype=self._dtype
        )
        action_costs = torch.zeros(
            self._num_samples, self._horizon, device=self._device, dtype=self._dtype
        )
        initial_state = self._state_seq_batch[:, 0, :]
        for t in range(self._horizon):
            prev_index = t - 1 if t > 0 else 0
            prev_state = self._state_seq_batch[:, prev_index, :]
            prev_action = self._perturbed_action_seqs[:, prev_index, :]
            # info update
            info["prev_state"] = prev_state
            info["prev_action"] = prev_action
            info["initial_state"] = initial_state
            info["t"] = t
            costs[:, t] = self._cost_func(
                self._state_seq_batch[:, t, :],
                self._perturbed_action_seqs[:, t, :],
                info,
            )
            action_costs[:, t] = (
                mean_action_seq[t]
                @ self._inv_covariance[t]
                @ self._perturbed_action_seqs[:, t].T
            )

        prev_state = self._state_seq_batch[:, -2, :]
        info["prev_state"] = prev_state
        zero_action = torch.zeros(
            self._num_samples,
            self._dim_control,
            device=self._device,
            dtype=self._dtype,
        )
        terminal_costs = self._cost_func(
            self._state_seq_batch[:, -1, :], zero_action, info
        )

        # In the original paper, the action cost is added to consider KL div. penalty,
        # but it is easier to tune without it
        # please see: https://arxiv.org/abs/2511.08019
        costs = (
            torch.sum(costs, dim=1) + terminal_costs
            # + torch.sum(self._lambda * action_costs, dim=1)
        )

        # =================================================================
        # Step 4: Update temperature (auto-lambda)
        # =================================================================
        if self._auto_lambda == "LBPS":
            # Lower-Bound Policy Search
            # ref: https://openreview.net/forum?id=HbGgF93Ppoy
            result = minimize_scalar(
                lambda lambda_: self._lbps_objective(lambda_, costs.detach()),
                bounds=(self._lambda_min, self._lambda_max),
                method="bounded",
            )
            self._lambda = result.x

        elif self._auto_lambda == "ESSPS":
            # Effective Sample Size Policy Search
            # ref: https://openreview.net/forum?id=HbGgF93Ppoy
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

        # =================================================================
        # Step 5: Compute importance weights
        # =================================================================
        # w_i = softmax(-cost_i / lambda) = exp(-cost_i/lambda) / sum(exp(-cost_j/lambda))
        self._weights = torch.softmax(-costs / self._lambda, dim=0)

        # =================================================================
        # Step 6: Compute optimal action as weighted average
        # =================================================================
        optimal_action_seq = torch.sum(
            self._weights.view(self._num_samples, 1, 1) * self._perturbed_action_seqs,
            dim=0,
        )
        mean_action_seq = optimal_action_seq

        if self._auto_lambda == "MPO":
            # auto-tune temperature parameter
            # Refer E step of MPO algorithm:
            # https://arxiv.org/pdf/1806.06920
            for _ in range(1):
                self.optimizer.zero_grad()
                temperature = torch.nn.functional.softplus(self.log_temperature)
                cost_logsumexp = torch.logsumexp(-costs / temperature, dim=0)
                loss = temperature * (self._mpo_epsilon + torch.mean(cost_logsumexp))
                loss.backward()
                self.optimizer.step()
            self._lambda = torch.exp(self.log_temperature).item()

        # calculate new covariance
        # https://arxiv.org/pdf/2104.00241
        # covariance = torch.sum(
        #     self._weights.view(self._num_samples, 1, 1)
        #     * (self._perturbed_action_seqs - optimal_action_seq) ** 2,
        #     dim=0,
        # )  # T x dim_control

        # small_cov = 1e-6 * torch.eye(
        #     self._dim_control, device=self._device, dtype=self._dtype
        # )
        # self._covariance = torch.diag_embed(covariance) + small_cov

        # for t in range(1, self._horizon):
        #     self._inv_covariance[t] = torch.inverse(self._covariance[t])
        # zero_mean = torch.zeros(self._dim_control, device=self._device, dtype=self._dtype)
        # self._noise_distribution = MultivariateNormal(
        #     loc=zero_mean, covariance_matrix=self._covariance
        # )

        # =================================================================
        # Step 7: Apply smoothing filter (optional)
        # =================================================================
        if self._use_sg_filter:
            # Concatenate history with new optimal sequence for filtering
            prolonged_action_seq = torch.cat(
                [
                    self._actions_history_for_sg,
                    optimal_action_seq,
                ],
                dim=0,
            )

            # Apply SG filter for each control dimension
            filtered_action_seq = torch.zeros_like(
                prolonged_action_seq, device=self._device, dtype=self._dtype
            )
            for i in range(self._dim_control):
                filtered_action_seq[:, i] = self._apply_savitzky_golay(
                    prolonged_action_seq[:, i], self._coeffs
                )

            # Extract filtered horizon-length sequence
            optimal_action_seq = filtered_action_seq[-self._horizon :]

        # =================================================================
        # Step 8: Predict state sequence and update history
        # =================================================================
        expanded_optimal_action_seq = optimal_action_seq.repeat(1, 1, 1)
        optimal_state_seq = self._states_prediction(state, expanded_optimal_action_seq)

        # Store for warm-starting next iteration
        self._previous_action_seq = optimal_action_seq

        # Update action history for Savitzky-Golay filter
        optimal_action = optimal_action_seq[0]
        self._actions_history_for_sg = torch.cat(
            [self._actions_history_for_sg[1:], optimal_action.view(1, -1)]
        )

        return optimal_action_seq, optimal_state_seq

    def get_top_samples(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the top-weighted trajectory samples.

        Retrieves the trajectories with highest importance weights from the
        most recent forward pass. Useful for visualization or debugging.

        Args:
            num_samples: Number of top samples to retrieve.

        Returns:
            Tuple of (top_samples, top_weights):
                - top_samples: Shape (num_samples, horizon + 1, dim_state)
                - top_weights: Shape (num_samples,), sorted descending
        """
        assert num_samples <= self._num_samples

        # large weights are better
        top_indices = torch.topk(self._weights, num_samples).indices

        top_samples = self._state_seq_batch[top_indices]
        top_weights = self._weights[top_indices]

        top_samples = top_samples[torch.argsort(top_weights, descending=True)]
        top_weights = top_weights[torch.argsort(top_weights, descending=True)]

        return top_samples, top_weights

    def get_samples_from_posterior(
        self, optimal_solution: torch.Tensor, state: torch.Tensor, num_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert num_samples <= self._num_samples

        # Posterior distribution of MPPI
        # Covariance is the same as noise distribution
        posterior_distribution = MultivariateNormal(
            loc=optimal_solution, covariance_matrix=self._covariance
        )

        # sampling control input sequence from posterior
        samples = posterior_distribution.sample(sample_shape=torch.Size([num_samples]))

        # get state sequence from sampled control input sequence
        predictive_states = self._states_prediction(state, samples)

        return samples, predictive_states

    def _states_prediction(
        self, state: torch.Tensor, action_seqs: torch.Tensor
    ) -> torch.Tensor:
        state_seqs = torch.zeros(
            action_seqs.shape[0],
            self._horizon + 1,
            self._dim_state,
            device=self._device,
            dtype=self._dtype,
        )
        state_seqs[:, 0, :] = state
        # expanded_optimal_action_seq = action_seq.repeat(1, 1, 1)
        for t in range(self._horizon):
            state_seqs[:, t + 1, :] = self._dynamics(
                state_seqs[:, t, :], action_seqs[:, t, :]
            )
        return state_seqs

    def _compute_ess(self, weights: torch.Tensor) -> float:
        """Compute Effective Sample Size (ESS).

        ESS = 1 / sum(weights^2)
        Range: 1 <= ESS <= N
        """
        return 1.0 / torch.sum(weights**2).item()

    def _lbps_objective(self, lambda_: float, costs: torch.Tensor) -> float:
        """LBPS objective function (Lower-Bound Policy Search).

        Maximizes the lower bound of expected return:
        J_LB(λ) = E[R] - penalty
        where penalty = R_range * sqrt((1-δ)/δ) / sqrt(ESS)

        Returns negative value for minimization.
        """
        weights = torch.softmax(-costs / lambda_, dim=0)
        ess = self._compute_ess(weights)

        # Expected return (negative cost)
        expected_return = -torch.sum(weights * costs).item()

        # Penalty term
        cost_range = (costs.max() - costs.min()).item()
        penalty = (
            cost_range
            * math.sqrt((1 - self._lbps_delta) / self._lbps_delta)
            / math.sqrt(ess)
        )

        return -(expected_return - penalty)  # Negative for minimization

    def _essps_objective(self, lambda_: float, costs: torch.Tensor) -> float:
        """ESSPS objective function (Effective Sample Size Policy Search).

        Returns ESS - target_ess for root finding.
        """
        weights = torch.softmax(-costs / lambda_, dim=0)
        ess = self._compute_ess(weights)
        return ess - self._essps_target_ess

    def _savitzky_golay_coeffs(self, window_size: int, poly_order: int) -> torch.Tensor:
        """
        Compute the Savitzky-Golay filter coefficients using PyTorch.

        Parameters:
        - window_size: The size of the window (must be odd).
        - poly_order: The order of the polynomial to fit.

        Returns:
        - coeffs: The filter coefficients as a PyTorch tensor.
        """
        # Ensure the window size is odd and greater than the polynomial order
        if window_size % 2 == 0 or window_size <= poly_order:
            raise ValueError("window_size must be odd and greater than poly_order.")

        # Generate the Vandermonde matrix of powers for the polynomial fit
        half_window = (window_size - 1) // 2
        indices = torch.arange(
            -half_window, half_window + 1, dtype=self._dtype, device=self._device
        )
        A = torch.vander(indices, N=poly_order + 1, increasing=True)

        # Compute the pseudo-inverse of the matrix
        pseudo_inverse = torch.linalg.pinv(A)

        # The filter coefficients are given by the first row of the pseudo-inverse
        coeffs = pseudo_inverse[0]

        return coeffs

    def _apply_savitzky_golay(
        self, y: torch.Tensor, coeffs: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply the Savitzky-Golay filter to a 1D signal using the provided coefficients.

        Parameters:
        - y: The input signal as a PyTorch tensor.
        - coeffs: The filter coefficients as a PyTorch tensor.

        Returns:
        - y_filtered: The filtered signal.
        """
        # Pad the signal at both ends to handle the borders
        pad_size = len(coeffs) // 2
        y_padded = torch.cat([y[:pad_size].flip(0), y, y[-pad_size:].flip(0)])

        # Apply convolution
        y_filtered = torch.conv1d(
            y_padded.view(1, 1, -1), coeffs.view(1, 1, -1), padding="valid"
        )

        return y_filtered.view(-1)
