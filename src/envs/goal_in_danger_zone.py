"""
Kohei Honda, 2024
"""

from __future__ import annotations

from typing import Tuple, Optional
from matplotlib import pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch


class DangerZone:
    def __init__(self, shape: str = "circle", cfg: dict = {}):
        self._shape = shape
        self._cfg = cfg

        if self._shape == "circle":
            self._generate_circle()
        else:
            raise ValueError(f"Invalid shape: {self._shape}")

    def _generate_circle(self):
        self.radius = self._cfg["radius"]
        self.center = self._cfg["center"]

    def get_random_inside_point(self):
        if self._shape == "circle":
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, self.radius)
            x = radius * np.cos(angle) + self.center[0]
            y = radius * np.sin(angle) + self.center[1]
            return np.array([x, y])

    def get_random_outside_point(self):
        if self._shape == "circle":
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(self.radius, 2 * self.radius)
            x = radius * np.cos(angle) + self.center[0]
            y = radius * np.sin(angle) + self.center[1]
            return np.array([x, y])

    def is_inside(self, pos: np.ndarray):
        if self._shape == "circle":
            return np.linalg.norm(pos - self.center) < self.radius

    def render(self, ax: plt.Axes):
        if self._shape == "circle":
            ax.set_xlim(-self.radius * 2, self.radius * 2)
            ax.set_ylim(-self.radius * 2, self.radius * 2)
            circle = plt.Circle(self.center, self.radius, color="gray", alpha=0.5)
            ax.add_artist(circle)
            ax.zorder = 0


# Normalize angle to [-pi, pi]
def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class GoalInDangerZoneEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        render_mode: str = "human",
        seed: int = 42,
        cfg: dict = {"shape": "circle", "radius": 10.0, "center": [0.0, 0.0]},
    ):
        self.render_mode = render_mode
        self._danger_zone = DangerZone(cfg=cfg)

        self._v_max = 1.0
        self._omega_max = 1.0
        self._v_min = -1.0
        self._omega_min = -1.0

        self._dt = 0.1

        self.max_episode_steps = 100

        self.action_space = spaces.Box(
            low=np.array([self._v_min, self._omega_min]),
            high=np.array([self._v_max, self._omega_max]),
            dtype=np.float32,
        )

        high = np.inf * np.ones(7)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self._fig = plt.figure(layout="tight")
        self._ax = self._fig.add_subplot()
        self._ax.set_aspect("equal")

    def _set_goal(self, danger_zone: DangerZone):
        # set goal inside the danger zone
        self._goal = danger_zone.get_random_inside_point()

    def _set_initial_state(self, danger_zone: DangerZone):
        # set initial position outside the danger zone
        self._pos = danger_zone.get_random_outside_point()
        self._angle = np.random.uniform(-np.pi, np.pi)

        self._v = 0.0
        self._omega = 0.0

    def parallel_step(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = obs[:, 0].view(-1, 1)
        y = obs[:, 1].view(-1, 1)
        theta = obs[:, 2].view(-1, 1)

        v = torch.clamp(action[:, 0].view(-1, 1), self._v_min, self._v_max)
        omega = torch.clamp(action[:, 1].view(-1, 1), self._omega_min, self._omega_max)

        def th_angle_normalize(x):
            return ((x + torch.pi) % (2 * torch.pi)) - torch.pi

        theta = th_angle_normalize(theta + omega * self._dt)

        new_x = x + v * torch.cos(theta) * self._dt
        new_y = y + v * torch.sin(theta) * self._dt

        th_goal = torch.tensor(self._goal, device=obs.device, dtype=obs.dtype)
        vec_to_goal = th_goal - torch.cat((new_x, new_y), dim=-1)
        th_center = torch.tensor(
            self._danger_zone.center, device=obs.device, dtype=obs.dtype
        )
        vec_to_center = th_center - torch.cat((new_x, new_y), dim=-1)

        return torch.cat((new_x, new_y, theta, vec_to_goal, vec_to_center), dim=-1)

    def parallel_cost(
        self, obs: torch.Tensor, action: torch.Tensor, info: dict
    ) -> torch.Tensor:
        prev_vec_to_goal = info["prev_state"][:, 3:5]
        vec_to_goal = obs[:, 3:5]
        vec_to_center = obs[:, 5:7]

        dist_to_goal = torch.norm(vec_to_goal, dim=-1)
        # minimize the distance to the goal
        cost = dist_to_goal

        # maximize the difference between the distance to the goal
        # prev_dist_to_goal = torch.norm(prev_vec_to_goal, dim=-1)
        # cost = dist_to_goal - prev_dist_to_goal

        is_collided = torch.norm(vec_to_center, dim=-1) < self._danger_zone.radius
        cost += is_collided.float() * 1000

        return cost

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        prev_pos = self._pos.copy()
        self._v = np.clip(action[0], self._v_min, self._v_max)
        self._omega = np.clip(action[1], self._omega_min, self._omega_max)

        self._angle = angle_normalize(self._angle + self._omega * self._dt)
        self._pos[0] += self._v * np.cos(self._angle) * self._dt
        self._pos[1] += self._v * np.sin(self._angle) * self._dt

        vec_to_goal = self._goal - self._pos
        vec_to_center = self._danger_zone.center - self._pos

        np_angle = np.array([self._angle])
        obs = np.concatenate([self._pos, np_angle, vec_to_goal, vec_to_center])

        prev_distance_to_goal = np.linalg.norm(prev_pos - self._goal)
        distance_to_goal = np.linalg.norm(self._pos - self._goal)

        is_collided = self._danger_zone.is_inside(self._pos)
        reward = float(prev_distance_to_goal - distance_to_goal)
        # reward = -distance_to_goal
        cost = float(is_collided)

        # is_goal_reached = distance_to_goal < 0.1
        # terminated = bool(is_goal_reached)
        terminated = False

        truncated = False
        if self._step >= self.max_episode_steps:
            truncated = True

        info = {"cost": cost}

        self._step += 1

        return obs, reward, terminated, truncated, info

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._set_initial_state(self._danger_zone)
        self._set_goal(self._danger_zone)
        self.set_render_info()
        self._step = 0

        vec_to_goal = self._goal - self._pos
        vec_to_center = self._danger_zone.center - self._pos

        cost = 0.0

        np_angle = np.array([self._angle])

        obs = np.concatenate([self._pos, np_angle, vec_to_goal, vec_to_center])

        return obs, {"cost": cost}

    def set_render_info(
        self,
        is_colllision: bool = None,
        predicted_trajectory: np.ndarray = None,
        top_samples: Tuple[np.ndarray, np.ndarray] = None,
    ):
        self._is_collision = is_colllision
        self._predicted_trajectory = predicted_trajectory
        self._top_samples = top_samples

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        self._danger_zone.render(self._ax)

        is_collision = None
        predicted_trajectory = None
        top_samples = None
        if self._is_collision is not None:
            is_collision = self._is_collision
        if self._predicted_trajectory is not None:
            predicted_trajectory = self._predicted_trajectory
        if self._top_samples is not None:
            top_samples = self._top_samples

        self._ax.scatter(
            self._goal[0],
            self._goal[1],
            marker="o",
            color="orange",
            zorder=10,
        )

        if is_collision is not None:
            if is_collision:
                self._ax.scatter(
                    self._pos[0],
                    self._pos[1],
                    marker="o",
                    color="red",
                    zorder=100,
                )
            else:
                self._ax.scatter(
                    self._pos[0],
                    self._pos[1],
                    marker="o",
                    color="green",
                    zorder=100,
                )

        if predicted_trajectory is not None:
            self._ax.scatter(
                predicted_trajectory[:, 0],
                predicted_trajectory[:, 1],
                color="darkblue",
                marker="o",
                s=3,
                zorder=2,
            )

        if top_samples is not None:
            top_samples, top_weights = top_samples
            top_samples = top_samples
            top_weights = top_weights
            top_weights = 0.7 * top_weights / np.max(top_weights)
            top_weights = np.clip(top_weights, 0.1, 0.7)
            for i in range(top_samples.shape[0]):
                self._ax.plot(
                    top_samples[i, :, 0],
                    top_samples[i, :, 1],
                    color="lightblue",
                    alpha=top_weights[i],
                    zorder=1,
                )

        if self.render_mode == "human":
            plt.pause(0.01)
            plt.cla()
        elif self.render_mode == "rgb_array":
            self._fig.canvas.draw()
            image = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
            plt.cla()
            return image

    def close(self):
        pass
