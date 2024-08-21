"""
Kohei Honda, 2023.
"""

from __future__ import annotations

from typing import Callable, Tuple, Dict
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
        cost_func: Callable[[torch.Tensor, torch.Tensor, Dict], torch.Tensor],
        u_min: torch.Tensor,
        u_max: torch.Tensor,
        sigmas: torch.Tensor,
        lambda_: float,
        auto_lambda: bool = False,
        exploration: float = 0.0,
        use_sg_filter: bool = False,
        sg_window_size: int = 5,
        sg_poly_order: int = 3,
        device=torch.device("cuda"),
        dtype=torch.float32,
        seed: int = 42,
    ) -> None:
        """
        :param horizon: Predictive horizon length.
        :param predictive_interval: Predictive interval (seconds).
        :param delta: predictive horizon step size (seconds).
        :param num_samples: Number of samples.
        :param dim_state: Dimension of state.
        :param dim_control: Dimension of control.
        :param dynamics: Dynamics model.
        :param cost_func: Cost function.
        :param u_min: Minimum control.
        :param u_max: Maximum control.
        :param sigmas: Noise standard deviation for each control dimension.
        :param lambda_: temperature parameter.
        :param exploration: Exploration rate when sampling.
        :param use_sg_filter: Use Savitzky-Golay filter.
        :param sg_window_size: Window size for Savitzky-Golay filter. larger is smoother. Must be odd.
        :param sg_poly_order: Polynomial order for Savitzky-Golay filter. Smaller is smoother.
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
        print(f"Device: {self._device}")
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
        self._lambda = lambda_
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

        # init satitzky-golay filter
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
        self._auto_lambda = auto_lambda
        if auto_lambda:
            self.log_tempature = torch.nn.Parameter(
                torch.log(
                    torch.tensor([self._lambda], device=self._device, dtype=self._dtype)
                )
            )
            self.optimizer = torch.optim.Adam([self.log_tempature], lr=1e-2)

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

        # noise injection with exploration
        threshold = int(self._num_samples * (1 - self._exploration))
        inherited_samples = mean_action_seq + self._action_noises[:threshold]
        self._perturbed_action_seqs = torch.cat(
            [inherited_samples, self._action_noises[threshold:]]
        )

        # clamp actions
        self._perturbed_action_seqs = torch.clamp(
            self._perturbed_action_seqs, self._u_min, self._u_max
        )

        # rollout samples in parallel
        self._state_seq_batch[:, 0, :] = state.repeat(self._num_samples, 1)

        for t in range(self._horizon):
            self._state_seq_batch[:, t + 1, :] = self._dynamics(
                self._state_seq_batch[:, t, :],
                self._perturbed_action_seqs[:, t, :],
            )

        # compute sample costs
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
        costs = (
            torch.sum(costs, dim=1)
            + terminal_costs
            # + torch.sum(self._lambda * action_costs, dim=1)
        )

        # calculate weights
        self._weights = torch.softmax(-costs / self._lambda, dim=0)

        # find optimal control by weighted average
        optimal_action_seq = torch.sum(
            self._weights.view(self._num_samples, 1, 1) * self._perturbed_action_seqs,
            dim=0,
        )

        mean_action_seq = optimal_action_seq

        # auto-tune temperature parameter
        # Refer E step of MPO algorithm:
        # https://arxiv.org/pdf/1806.06920
        if self._auto_lambda:
            for _ in range(1):
                self.optimizer.zero_grad()
                tempature = torch.nn.functional.softplus(self.log_tempature)
                cost_logsumexp = torch.logsumexp(-costs / tempature, dim=0)
                epsilon = 0.1  # tolerance hyperparameter for KL divergence
                loss = tempature * (epsilon + torch.mean(cost_logsumexp))
                loss.backward()
                self.optimizer.step()
            self._lambda = torch.exp(self.log_tempature).item()

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

        if self._use_sg_filter:
            # apply savitzky-golay filter to N-1 previous action history + N optimal action seq
            prolonged_action_seq = torch.cat(
                [
                    self._actions_history_for_sg,
                    optimal_action_seq,
                ],
                dim=0,
            )

            # appply sg filter for each control dimension
            filtered_action_seq = torch.zeros_like(
                prolonged_action_seq, device=self._device, dtype=self._dtype
            )
            for i in range(self._dim_control):
                filtered_action_seq[:, i] = self._apply_savitzky_golay(
                    prolonged_action_seq[:, i], self._coeffs
                )

            # use only N step optimal action seq
            optimal_action_seq = filtered_action_seq[-self._horizon :]

        # predivtive state seq
        expanded_optimal_action_seq = optimal_action_seq.repeat(1, 1, 1)
        optimal_state_seq = self._states_prediction(state, expanded_optimal_action_seq)

        # update previous actions
        self._previous_action_seq = optimal_action_seq

        # stuck previous actions for sg filter
        optimal_action = optimal_action_seq[0]
        self._actions_history_for_sg = torch.cat(
            [self._actions_history_for_sg[1:], optimal_action.view(1, -1)]
        )

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

    def get_samples_from_posterior(
        self, optimal_solution: torch.Tensor, state: torch.Tensor, num_samples: int
    ) -> Tuple[torch.Tensor]:
        assert num_samples <= self._num_samples

        # posterior distribution of MPPI
        # covaraince is the same as noise distribution
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
