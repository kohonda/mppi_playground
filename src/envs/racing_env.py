"""
Michikuni Eguchi, 2024.
"""

from __future__ import annotations

from typing import Tuple, Union
from matplotlib import pyplot as plt

import torch
import numpy as np
import os


from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from envs.obstacle_map_2d import ObstacleMap, generate_random_obstacles
from envs.circuit_generator.path_generate import make_track, make_side_lane



@torch.jit.script
def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi


class RacingEnv:
    def __init__(
        self, device=torch.device("cuda"), dtype=torch.float32, seed: int = 42
    ) -> None:
        # device and dtype
        if torch.cuda.is_available() and device == torch.device("cuda"):
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        self._dtype = dtype

        # generate reference path
        self.circle_radius = 15
        self.linelength = 20.0
        self.dl = 0.1
        self.line_width = 3
        self.line_width_gpu = torch.tensor(self.line_width*0.6, device=self._device, dtype=self._dtype)
        self._reference_path = make_track(circle_radius=self.circle_radius, linelength=self.linelength, dl=self.dl)
        self.right_lane, self.left_lane = make_side_lane(self._reference_path, lane_width=self.line_width)

        # generate obstacles
        self._obstacle_map = ObstacleMap(
            map_size=(60, 40), cell_size=0.1, device=self._device, dtype=self._dtype
        )
        self._seed = seed

        generate_random_obstacles(
            obstacle_map=self._obstacle_map,
            random_x_range=(-30, 30),
            random_y_range=(-17, 17),
            num_circle_obs=30,
            radius_range=(1.2, 1.5),
            num_rectangle_obs=10,
            width_range=(1.5, 2.0),
            height_range=(1.5, 2.0),
            max_iteration=1000,
            seed=seed,
        )
        self._obstacle_map.convert_to_torch()

        self._start_pos = torch.tensor(
            [self._reference_path[0][0], self._reference_path[0][1]], device=self._device, dtype=self._dtype
        )
        self._goal_pos = torch.tensor(
            [self._reference_path[-1][0], self._reference_path[-1][1]], device=self._device, dtype=self._dtype
        )

        self._robot_state = torch.zeros(4, device=self._device, dtype=self._dtype)
        self._robot_state[:2] = self._start_pos
        self._robot_state[2] = angle_normalize(
            torch.atan2(
                self._reference_path[1][1] - self._start_pos[1],
                self._reference_path[1][0] - self._start_pos[0],
            )
        )
        self._robot_state[3] = 0.0

        # u: [accel, steer] (m/s2, rad)
        self.u_min = torch.tensor([-2.0, -0.5], device=self._device, dtype=self._dtype)
        self.u_max = torch.tensor([2.0, 0.5], device=self._device, dtype=self._dtype)
        
        # model parameters
        self.L = torch.tensor(0.5, device=self._device, dtype=self._dtype)
        self.V_MAX = torch.tensor(8.0, device=self._device, dtype=self._dtype)

    def reset(self) -> torch.Tensor:
        """
        Reset robot state.
        Returns:
            torch.Tensor: shape (3,) [x, y, theta]
        """
        self._robot_state[:2] = self._start_pos
        self._robot_state[2] = angle_normalize(
            torch.atan2(
                self._reference_path[1][1] - self._start_pos[1],
                self._reference_path[1][0] - self._start_pos[0],
            )
        )
        self._robot_state[3] = 0.0

        self._fig = plt.figure(layout="tight")
        self._ax = self._fig.add_subplot()
        self._ax.set_xlim(self._obstacle_map.x_lim)
        self._ax.set_ylim(self._obstacle_map.y_lim)
        self._ax.set_aspect("equal")

        self._rendered_frames = []

        return self._robot_state

    def step(self, u: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Update robot state based on differential drive dynamics.
        Args:
            u (torch.Tensor): control batch tensor, shape (2) [v, omega]
        Returns:
            Tuple[torch.Tensor, bool]: Tuple of robot state and is goal reached.
        """

        u = torch.clamp(u, self.u_min, self.u_max)

        self._robot_state = self.dynamics(
            state=self._robot_state.unsqueeze(0), action=u.unsqueeze(0)
        ).squeeze(0)

        # goal check
        # goal_threshold = 0.5
        # is_goal_reached = (
        #     torch.norm(self._robot_state[:2] - self._goal_pos) < goal_threshold
        # )
        is_goal_reached = False

        return self._robot_state, is_goal_reached

    def render(
        self,
        action: torch.Tensor = None,
        predicted_trajectory: torch.Tensor = None,
        is_collisions: torch.Tensor = None,
        top_samples: Tuple[torch.Tensor, torch.Tensor] = None,
        reference_trajectory: np.ndarray = None,
        mode: str = "human",
    ) -> None:
        self._ax.set_xlabel("x [m]")
        self._ax.set_ylabel("y [m]")

        # obstacle map
        self._obstacle_map.render(self._ax, zorder=10)

        # reference path
        self._ax.plot(
            self._reference_path[:, 0],
            self._reference_path[:, 1],
            color="gray",
            linestyle="--",
            zorder=5,
        )
        self._ax.plot(
            self.right_lane[:, 0],
            self.right_lane[:, 1],
            color="green",
            linestyle="--",
            zorder=5,
        )
        self._ax.plot(
            self.left_lane[:, 0],
            self.left_lane[:, 1],
            color="green",
            linestyle="--",
            zorder=5,
        )
    
        # reference trajectory
        if reference_trajectory is not None:
            self._ax.plot(
                reference_trajectory[:, 0],
                reference_trajectory[:, 1],
                color="red",
                linestyle="dotted",
                zorder=5,
            )



        # robot
        robot_x = self._robot_state[0].item()
        robot_y = self._robot_state[1].item()
        robot_theta = self._robot_state[2].item()
        robot_v = self._robot_state[3].item()
        accel = action[0].item()
        steer = action[1].item()

        self._ax.scatter(
            robot_x,
            robot_y,
            marker="o",
            color="green",
            zorder=100,
        )
        # robot direction
        self._ax.quiver(
            robot_x,
            robot_y,
            robot_v * np.cos(robot_theta),
            robot_v * np.sin(robot_theta),
            color="green",
            zorder=100,
        )
        # steering direction
        self._ax.quiver(
            robot_x,
            robot_y,
            self.L.cpu() * np.cos(robot_theta + steer),
            self.L.cpu() * np.sin(robot_theta + steer),
            color="blue",
            zorder=100,
        )

        # visualize top samples with different alpha based on weights
        if top_samples is not None:
            top_samples, top_weights = top_samples
            top_samples = top_samples.cpu().numpy()
            top_weights = top_weights.cpu().numpy()
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

        # predicted trajectory
        if predicted_trajectory is not None:
            # if is collision color is red
            colors = np.array(["darkblue"] * predicted_trajectory.shape[1])
            if is_collisions is not None:
                is_collisions = is_collisions.cpu().numpy()
                is_collisions = np.any(is_collisions, axis=0)
                colors[is_collisions] = "red"

            self._ax.scatter(
                predicted_trajectory[0, :, 0].cpu().numpy(),
                predicted_trajectory[0, :, 1].cpu().numpy(),
                color=colors,
                marker="o",
                s=3,
                zorder=2,
            )

        if mode == "human":
            # online rendering
            plt.pause(0.001)
            plt.cla()
        elif mode == "rgb_array":
            # offline rendering for video
            # TODO: high resolution rendering
            self._fig.canvas.draw()
            data = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
            plt.cla()
            self._rendered_frames.append(data)

    def close(self, path: str = None) -> None:
        if path is None:
            # mkdir video if not exists

            if not os.path.exists("video"):
                os.mkdir("video")
            path = "video/" + "navigation_2d_" + str(self._seed) + ".gif"

        if len(self._rendered_frames) > 0:
            # save animation
            clip = ImageSequenceClip(self._rendered_frames, fps=10)
            # clip.write_videofile(path, fps=10)
            clip.write_gif(path, fps=10)

    def dynamics(
        self, state: torch.Tensor, action: torch.Tensor, delta_t: float = 0.1
    ) -> torch.Tensor:
        """
        Update robot state based on differential drive dynamics.
        Args:
            state (torch.Tensor): state batch tensor, shape (batch_size, 3) [x, y, theta, v]
            action (torch.Tensor): control batch tensor, shape (batch_size, 2) [accel, steer]
            delta_t (float): time step interval [s]
        Returns:
            torch.Tensor: shape (batch_size, 3) [x, y, theta]
        """

        # Perform calculations as before
        x = state[:, 0].view(-1, 1)
        y = state[:, 1].view(-1, 1)
        theta = state[:, 2].view(-1, 1)
        v = state[:, 3].view(-1, 1)
        accel = torch.clamp(action[:, 0].view(-1, 1), self.u_min[0], self.u_max[0])
        steer = torch.clamp(action[:, 1].view(-1, 1), self.u_min[1], self.u_max[1])
        theta = angle_normalize(theta)

        dx = v * torch.cos(theta)
        dy = v * torch.sin(theta)
        dv = accel
        dtheta = v * torch.tan(steer) / self.L

        new_x = x + dx * delta_t
        new_y = y + dy * delta_t
        new_theta = angle_normalize(theta + dtheta * delta_t)
        new_v = v + dv * delta_t

        # Clamp x and y to the map boundary
        x_lim = torch.tensor(
            self._obstacle_map.x_lim, device=self._device, dtype=self._dtype
        )
        y_lim = torch.tensor(
            self._obstacle_map.y_lim, device=self._device, dtype=self._dtype
        )
        clamped_x = torch.clamp(new_x, x_lim[0], x_lim[1])
        clamped_y = torch.clamp(new_y, y_lim[0], y_lim[1])
        clamped_v = torch.clamp(new_v, -self.V_MAX, self.V_MAX)


        result = torch.cat([clamped_x, clamped_y, new_theta, clamped_v], dim=1)

        return result

    def cost_function(self, state: torch.Tensor, action: torch.Tensor, info: dict) -> torch.Tensor:
        """
        Calculate cost function
        Args:
            state (torch.Tensor): state batch tensor, shape (batch_size, 4) [x, y, theta, v]
            action (torch.Tensor): control batch tensor, shape (batch_size, 2) [accel, steer]
        Returns:
            torch.Tensor: shape (batch_size,)
        """

        reference_path = torch.from_numpy(info["reference_path"]).to(self._device, self._dtype)
        prev_action = info["prev_action"]
        t = info["t"] # horizon number

        # contouring error of path
        ec = torch.sin(reference_path[t, 2]) * (state[:, 0] - reference_path[t, 0]) - torch.cos(reference_path[t, 2]) * (state[:, 1] - reference_path[t, 1])
        # lag error of path
        el = -torch.cos(reference_path[t, 2]) * (state[:, 0] - reference_path[t, 0]) - torch.sin(reference_path[t, 2]) * (state[:, 1] - reference_path[t, 1])

        path_cost = 0.4 * ec ** 2 + 0.5 * el ** 2
        # 経路の範囲外に出た場合のコスト
        path_cost += 10000 * (ec.abs() > self.line_width_gpu).float()

        pos_batch = state[:, :2].unsqueeze(1)  # (batch_size, 1, 2)
        obstacle_cost = self._obstacle_map.compute_cost(pos_batch).squeeze(
            1
        )  # (batch_size,)

        # input cost
        input_cost = 0.01 * action.pow(2).sum(dim=1)
        input_cost += 0.01 * (action - prev_action).pow(2).sum(dim=1)

        cost = path_cost + 10000 * obstacle_cost + input_cost

        return cost


    def collision_check(self, state: torch.Tensor) -> torch.Tensor:
        """

        Args:
            state (torch.Tensor): state batch tensor, shape (batch_size, traj_size , 3) [x, y, theta]
        Returns:
            torch.Tensor: shape (batch_size,)
        """
        pos_batch = state[:, :, :2]
        is_collisions = self._obstacle_map.compute_cost(pos_batch).squeeze(1)
        return is_collisions
