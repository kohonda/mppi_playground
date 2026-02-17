"""
Dynamic crowd navigation environment for DRA-MPPI experiments.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from envs.navigation_2d import Navigation2DEnv


class CrowdNavigationEnv(Navigation2DEnv, gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        num_pedestrians: int = 8,
        pedestrian_radius: float = 0.3,
        robot_radius: float = 0.4,
        pedestrian_model: str = "social_force",
        dt: float = 0.1,
        max_episode_steps: int = 500,
        render_mode: str = "human",
        device=torch.device("cuda"),
        dtype=torch.float32,
        seed: int = 42,
    ) -> None:
        super().__init__(device=device, dtype=dtype, seed=seed)

        self._num_pedestrians = num_pedestrians
        self._pedestrian_radius = pedestrian_radius
        self._robot_radius = robot_radius
        self._pedestrian_model = pedestrian_model
        self._dt = dt
        self._render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        self._pedestrian_state = torch.zeros(
            self._num_pedestrians, 4, device=self._device, dtype=self._dtype
        )  # [x, y, vx, vy]
        self._pedestrian_goal = torch.zeros(
            self._num_pedestrians, 2, device=self._device, dtype=self._dtype
        )
        self._step_count = 0

        self._pedestrian_speed_min = 0.2
        self._pedestrian_speed_max = 1.0
        self._pedestrian_goal_threshold = 0.8
        self._social_force_gain = 1.2
        self._social_force_decay = 1.5
        self._robot_repulsion_gain = 3.0

        obs_dim = 3 + 4 * self._num_pedestrians
        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=self.u_min.detach().cpu().numpy(),
            high=self.u_max.detach().cpu().numpy(),
            dtype=np.float32,
        )

        self._render_info: Dict = {}
        self._rendered_frames = []

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def num_pedestrians(self) -> int:
        return self._num_pedestrians

    def _uniform_random_position(self) -> torch.Tensor:
        x = torch.empty(1, device=self._device, dtype=self._dtype).uniform_(
            float(self._obstacle_map.x_lim[0]) + 0.5,
            float(self._obstacle_map.x_lim[1]) - 0.5,
        )
        y = torch.empty(1, device=self._device, dtype=self._dtype).uniform_(
            float(self._obstacle_map.y_lim[0]) + 0.5,
            float(self._obstacle_map.y_lim[1]) - 0.5,
        )
        return torch.cat([x, y], dim=0)

    def _is_position_valid(
        self, position: torch.Tensor, margin: float = 0.35, allow_robot_overlap: bool = False
    ) -> bool:
        pos_batch = position.view(1, 1, 2)
        is_occ = self._obstacle_map.compute_cost(pos_batch).item() > 0.5
        if is_occ:
            return False

        if not allow_robot_overlap:
            if torch.norm(position - self._robot_state[:2]).item() < (
                self._robot_radius + margin
            ):
                return False

        if torch.norm(position - self._start_pos).item() < 1.0:
            return False
        if torch.norm(position - self._goal_pos).item() < 1.0:
            return False
        return True

    def _sample_free_position(
        self, margin: float = 0.35, allow_robot_overlap: bool = False
    ) -> torch.Tensor:
        for _ in range(1000):
            p = self._uniform_random_position()
            if self._is_position_valid(
                p, margin=margin, allow_robot_overlap=allow_robot_overlap
            ):
                return p
        raise RuntimeError("Failed to sample a valid pedestrian position.")

    def _reset_pedestrians(self) -> None:
        for i in range(self._num_pedestrians):
            pos = self._sample_free_position(margin=self._pedestrian_radius + 0.15)
            goal = self._sample_free_position(
                margin=self._pedestrian_radius + 0.15, allow_robot_overlap=True
            )
            direction = goal - pos
            direction_norm = torch.norm(direction).clamp_min(1e-6)
            direction = direction / direction_norm
            speed = torch.empty(1, device=self._device, dtype=self._dtype).uniform_(
                self._pedestrian_speed_min, self._pedestrian_speed_max
            )[0]

            self._pedestrian_state[i, :2] = pos
            self._pedestrian_state[i, 2:] = direction * speed
            self._pedestrian_goal[i, :] = goal

    def _get_observation(self) -> torch.Tensor:
        return torch.cat([self._robot_state, self._pedestrian_state.flatten()], dim=0)

    def extract_robot_state(self, observation: torch.Tensor) -> torch.Tensor:
        return observation[:3]

    def extract_pedestrian_state(self, observation: torch.Tensor) -> torch.Tensor:
        return observation[3:].view(self._num_pedestrians, 4)

    def _update_pedestrians_constant_velocity(self) -> None:
        self._pedestrian_state[:, :2] += self._pedestrian_state[:, 2:] * self._dt

    def _update_pedestrians_social_force(self) -> None:
        pos = self._pedestrian_state[:, :2]
        vel = self._pedestrian_state[:, 2:]

        goal_vec = self._pedestrian_goal - pos
        goal_dist = torch.norm(goal_vec, dim=1, keepdim=True).clamp_min(1e-6)
        desired_dir = goal_vec / goal_dist
        desired_speed = torch.clamp(
            torch.norm(vel, dim=1, keepdim=True), self._pedestrian_speed_min, self._pedestrian_speed_max
        )
        desired_vel = desired_dir * desired_speed

        pairwise_delta = pos.unsqueeze(1) - pos.unsqueeze(0)
        pairwise_dist = torch.norm(pairwise_delta, dim=2, keepdim=True).clamp_min(1e-4)
        pairwise_dir = pairwise_delta / pairwise_dist

        eye = torch.eye(self._num_pedestrians, device=self._device, dtype=self._dtype)
        mask = (1.0 - eye).unsqueeze(-1)
        repulsion_mag = self._social_force_gain * torch.exp(
            -pairwise_dist / self._social_force_decay
        )
        social_force = torch.sum(mask * repulsion_mag * pairwise_dir, dim=1)

        robot_delta = pos - self._robot_state[:2].unsqueeze(0)
        robot_dist = torch.norm(robot_delta, dim=1, keepdim=True).clamp_min(1e-4)
        robot_repulsion = (
            self._robot_repulsion_gain
            * torch.exp(-robot_dist / (self._robot_radius + self._pedestrian_radius))
            * (robot_delta / robot_dist)
        )

        accel = (desired_vel - vel) + social_force + robot_repulsion
        new_vel = vel + accel * self._dt
        speed = torch.norm(new_vel, dim=1, keepdim=True).clamp_min(1e-6)
        new_vel = new_vel * torch.clamp(speed, max=self._pedestrian_speed_max) / speed

        self._pedestrian_state[:, 2:] = new_vel
        self._pedestrian_state[:, :2] += new_vel * self._dt

    def _clamp_pedestrians(self) -> None:
        x_lim = torch.tensor(
            self._obstacle_map.x_lim, device=self._device, dtype=self._dtype
        )
        y_lim = torch.tensor(
            self._obstacle_map.y_lim, device=self._device, dtype=self._dtype
        )
        self._pedestrian_state[:, 0] = torch.clamp(
            self._pedestrian_state[:, 0], x_lim[0], x_lim[1]
        )
        self._pedestrian_state[:, 1] = torch.clamp(
            self._pedestrian_state[:, 1], y_lim[0], y_lim[1]
        )

        is_occ = (
            self._obstacle_map.compute_cost(self._pedestrian_state[:, :2].view(-1, 1, 2))
            .squeeze(1)
            .bool()
        )
        self._pedestrian_state[is_occ, 2:] = -0.5 * self._pedestrian_state[is_occ, 2:]
        self._pedestrian_state[is_occ, :2] += self._pedestrian_state[is_occ, 2:] * self._dt

    def _refresh_pedestrian_goals(self) -> None:
        dist_to_goal = torch.norm(self._pedestrian_goal - self._pedestrian_state[:, :2], dim=1)
        reached = dist_to_goal < self._pedestrian_goal_threshold
        for idx in torch.where(reached)[0]:
            self._pedestrian_goal[idx] = self._sample_free_position(
                margin=self._pedestrian_radius + 0.15, allow_robot_overlap=True
            )

    def _update_pedestrians(self) -> None:
        if self._pedestrian_model == "constant_velocity":
            self._update_pedestrians_constant_velocity()
        elif self._pedestrian_model == "social_force":
            self._update_pedestrians_social_force()
        else:
            raise ValueError(f"Unsupported pedestrian model: {self._pedestrian_model}")

        self._clamp_pedestrians()
        self._refresh_pedestrian_goals()

    def _check_robot_pedestrian_collision(self) -> bool:
        dist = torch.norm(self._pedestrian_state[:, :2] - self._robot_state[:2], dim=1)
        return bool(torch.any(dist < (self._robot_radius + self._pedestrian_radius)))

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self._seed = seed
            torch.manual_seed(seed)
            np.random.seed(seed)

        super().reset()
        self._reset_pedestrians()
        self._step_count = 0
        self._rendered_frames = []
        self._render_info = {}

        obs = self._get_observation().detach().cpu().numpy().astype(np.float32)
        info = {"is_collision": False, "is_goal_reached": False}
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        prev_dist_to_goal = torch.norm(self._robot_state[:2] - self._goal_pos).item()

        action_th = torch.as_tensor(action, device=self._device, dtype=self._dtype)
        _, is_goal_reached = super().step(action_th)
        self._update_pedestrians()

        is_collision = self._check_robot_pedestrian_collision()
        curr_dist_to_goal = torch.norm(self._robot_state[:2] - self._goal_pos).item()

        progress_reward = prev_dist_to_goal - curr_dist_to_goal
        collision_penalty = 50.0 if is_collision else 0.0
        reward = float(progress_reward - collision_penalty)

        terminated = bool(is_goal_reached or is_collision)
        self._step_count += 1
        truncated = bool(self._step_count >= self.max_episode_steps)

        obs = self._get_observation().detach().cpu().numpy().astype(np.float32)
        info = {
            "is_collision": is_collision,
            "is_goal_reached": bool(is_goal_reached),
            "robot_state": self._robot_state.detach().cpu().numpy(),
            "pedestrian_state": self._pedestrian_state.detach().cpu().numpy(),
        }
        return obs, reward, terminated, truncated, info

    def set_render_info(
        self,
        predicted_trajectory: Optional[np.ndarray] = None,
        top_samples: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        predicted_obstacles: Optional[Dict[str, torch.Tensor]] = None,
        mc_points: Optional[np.ndarray] = None,
    ) -> None:
        self._render_info = {
            "predicted_trajectory": predicted_trajectory,
            "top_samples": top_samples,
            "predicted_obstacles": predicted_obstacles,
            "mc_points": mc_points,
        }

    def _draw_distribution(self, means: np.ndarray, covariances: np.ndarray) -> None:
        num_obs = means.shape[0]
        for i in range(num_obs):
            mean = means[i]
            cov = covariances[i]
            eigvals, eigvecs = np.linalg.eigh(cov + 1e-6 * np.eye(2))
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            width = 2.0 * np.sqrt(max(eigvals[0], 1e-6))
            height = 2.0 * np.sqrt(max(eigvals[1], 1e-6))
            patch = Ellipse(
                xy=mean,
                width=width,
                height=height,
                angle=angle,
                edgecolor="tab:orange",
                facecolor="none",
                alpha=0.6,
                linewidth=1.2,
                zorder=12,
            )
            self._ax.add_patch(patch)

    def render(self) -> Optional[np.ndarray]:
        self._ax.set_xlabel("x [m]")
        self._ax.set_ylabel("y [m]")
        self._ax.set_xlim(self._obstacle_map.x_lim)
        self._ax.set_ylim(self._obstacle_map.y_lim)
        self._ax.set_aspect("equal")

        self._obstacle_map.render(self._ax, zorder=0)

        self._ax.scatter(
            self._start_pos[0].item(),
            self._start_pos[1].item(),
            marker="o",
            color="red",
            zorder=10,
            label="start",
        )
        self._ax.scatter(
            self._goal_pos[0].item(),
            self._goal_pos[1].item(),
            marker="o",
            color="orange",
            zorder=10,
            label="goal",
        )

        self._ax.scatter(
            self._robot_state[0].item(),
            self._robot_state[1].item(),
            marker="o",
            color="green",
            s=45,
            zorder=20,
            label="robot",
        )

        ped_pos = self._pedestrian_state[:, :2].detach().cpu().numpy()
        ped_vel = self._pedestrian_state[:, 2:].detach().cpu().numpy()
        self._ax.scatter(
            ped_pos[:, 0], ped_pos[:, 1], marker="x", color="black", zorder=20, label="ped"
        )
        self._ax.quiver(
            ped_pos[:, 0],
            ped_pos[:, 1],
            ped_vel[:, 0],
            ped_vel[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="black",
            alpha=0.5,
            zorder=15,
        )

        predicted_trajectory = self._render_info.get("predicted_trajectory")
        if predicted_trajectory is not None:
            self._ax.plot(
                predicted_trajectory[0, :, 0],
                predicted_trajectory[0, :, 1],
                color="tab:blue",
                linewidth=1.8,
                zorder=11,
                label="pred traj",
            )

        top_samples = self._render_info.get("top_samples")
        if top_samples is not None:
            top_seq, top_weights = top_samples
            top_weights = 0.7 * top_weights / max(np.max(top_weights), 1e-6)
            top_weights = np.clip(top_weights, 0.08, 0.7)
            for i in range(top_seq.shape[0]):
                self._ax.plot(
                    top_seq[i, :, 0],
                    top_seq[i, :, 1],
                    color="lightblue",
                    alpha=float(top_weights[i]),
                    zorder=3,
                )

        predicted_obstacles = self._render_info.get("predicted_obstacles")
        if predicted_obstacles is not None and predicted_obstacles.get("mode") == "gaussian":
            means = predicted_obstacles["means"][:, 0, :].detach().cpu().numpy()
            covariances = predicted_obstacles["covariances"][:, 0, :, :].detach().cpu().numpy()
            self._draw_distribution(means, covariances)

        mc_points = self._render_info.get("mc_points")
        if mc_points is not None:
            self._ax.scatter(
                mc_points[:, 0],
                mc_points[:, 1],
                color="tab:purple",
                s=5,
                alpha=0.2,
                zorder=2,
                label="mc",
            )

        if self._render_mode == "human":
            plt.pause(0.001)
            plt.cla()
            return None
        if self._render_mode == "rgb_array":
            self._fig.canvas.draw()
            image = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
            self._rendered_frames.append(image)
            plt.cla()
            return image
        raise ValueError(f"Unsupported render mode: {self._render_mode}")

    def close(self, path: Optional[str] = None) -> None:
        if path is None:
            if not os.path.exists("video"):
                os.mkdir("video")
            path = "video/crowd_navigation.gif"

        if len(self._rendered_frames) > 0:
            clip = ImageSequenceClip(self._rendered_frames, fps=10)
            clip.write_gif(path, fps=10)
