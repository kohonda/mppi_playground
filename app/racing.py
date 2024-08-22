import torch
import numpy as np
from typing import Tuple

import time

# import gymnasium
import fire
import tqdm

from controller.mppi import MPPI
from envs.racing_env import RacingEnv
from envs.obstacle_map_2d import ObstacleMap
from envs.lane_map_2d import LaneMap

class racing_controller:
    def __init__(self, env, debug=False, device=torch.device("cuda"), dtype=torch.float32) -> None:
        
        self.debug = debug
        self.current_path_index = 0

        # solver
        self.solver = MPPI(
            horizon=25,
            num_samples=4000,
            dim_state=4,
            dim_control=2,
            dynamics=env.dynamics,
            cost_func=self.cost_function,
            u_min=env.u_min,
            u_max=env.u_max,
            sigmas=torch.tensor([0.5, 0.1]),
            lambda_=1.0,
            auto_lambda=False,
        )

        # config
        self.env = env

        # cost weights
        self.Qc = 2.0  # contouring error cost
        self.Ql = 3.0  # lag error cost
        self.Qv = 2.0  # velocity cost
        self.Qo = 10000.0  # obstacle cost
        self.Qin = 0.01  # input cost
        self.Qdin = 0.5  # differential input cost

        # device and dtype
        if torch.cuda.is_available() and device == torch.device("cuda"):
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        self._dtype = dtype

        # reference indformation (tensor)
        self.reference_path: torch.Tensor = None
        self.obstacle_map: ObstacleMap = None
        self.lane_map: LaneMap = None

    def update(self, state: torch.Tensor, racing_center_path: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the controller with the current state and reference path.
        Args:
            state (torch.Tensor): current state of the vehicle, shape (4,) [x, y, yaw, v]
            racing_center_path (torch.Tensor): racing center path, shape (N, 3) [x, y, yaw]
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: action sequence tensor, shape (horizon, 2) [accel, steer], state sequence tensor, shape (horizon + 1, 4) [x, y, yaw, v]
        """

        # reference
        self.reference_path, self.current_path_index = self.calc_ref_trajectory(
            state, racing_center_path, self.current_path_index, self.solver._horizon, DL=0.1, lookahead_distance=3, reference_path_interval=0.85
        )

        if self.reference_path is None and self.obstacle_map is None and self.lane_map is None:
            raise ValueError("reference path, obstacle map, and lane map must be set before calling solve method.")

        # solve        
        start = time.time()
        action_seq, state_seq = self.solver.forward(state=state)
        end = time.time()
        solve_time = end - start

        if self.debug:
            print("solve time: {}".format(round(solve_time * 1000, 2)), " [ms]")

        return action_seq, state_seq
    
    def get_top_samples(self, num_samples = 300) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.solver.get_top_samples(num_samples=num_samples)
    
    def set_cost_map(self, obstacle_map: ObstacleMap, lane_map: LaneMap) -> None:
        self.obstacle_map = obstacle_map
        self.lane_map = lane_map

    def cost_function(self, state: torch.Tensor, action: torch.Tensor, info: dict) -> torch.Tensor:
        """
        Calculate cost function
        Args:
            state (torch.Tensor): state batch tensor, shape (batch_size, 4) [x, y, theta, v]
            action (torch.Tensor): control batch tensor, shape (batch_size, 2) [accel, steer]
        Returns:
            torch.Tensor: shape (batch_size,)
        """
        # info
        prev_action = info["prev_action"]
        t = info["t"] # horizon number

        # path cost
        # contouring and lag error of path
        ec = torch.sin(self.reference_path[t, 2]) * (state[:, 0] - self.reference_path[t, 0]) \
            -torch.cos(self.reference_path[t, 2]) * (state[:, 1] - self.reference_path[t, 1])
        el = -torch.cos(self.reference_path[t, 2]) * (state[:, 0] - self.reference_path[t, 0]) \
             -torch.sin(self.reference_path[t, 2]) * (state[:, 1] - self.reference_path[t, 1])

        path_cost = self.Qc * ec.pow(2) + self.Ql * el.pow(2)

        # velocity cost
        v = state[:, 3]
        v_target = self.reference_path[t, 3]
        velocity_cost = self.Qv * (v - v_target).pow(2)

        # compute obstacle cost from cost map
        pos_batch = state[:, :2].unsqueeze(1)  # (batch_size, 1, 2)
        obstacle_cost = self.obstacle_map.compute_cost(pos_batch).squeeze(1)  # (batch_size,)
        obstacle_cost += self.lane_map.compute_cost(pos_batch).squeeze(1)
        obstacle_cost = self.Qo * obstacle_cost

        # input cost
        input_cost = self.Qin * action.pow(2).sum(dim=1)
        input_cost += self.Qdin * (action - prev_action).pow(2).sum(dim=1)

        cost = path_cost + velocity_cost + obstacle_cost + input_cost

        return cost
    
    def calc_ref_trajectory(self, state: torch.Tensor, path: torch.Tensor, 
                            cind: int, horizon: int, DL=0.1, lookahead_distance=1.0, reference_path_interval=0.5
                            ) -> Tuple[torch.Tensor, int]:
        """
        Calculate the reference trajectory for the vehicle.

        Args:
            state (torch.Tensor): current state of the vehicle, shape (4,) [x, y, yaw, v]
            path (torch.Tensor): reference path, shape (N, 3) [x, y, yaw]
            cind (int): current index of the vehicle on the path
            horizon (int): prediction horizon
            DL (float): resolution of the path
            lookahead_distance (float): distance to look ahead
            reference_path_interval (float): interval of the reference path

        Returns:
            Tuple[torch.Tensor, int]: reference trajectory tensor, shape (horizon + 1, 4) [x, y, yaw, target_v], index of the vehicle on the path
        """

        ncourse = len(path)
        xref = torch.zeros((horizon + 1, state.shape[0]), dtype=state.dtype, device=state.device)

        # Calculate the nearest index to the vehicle
        ind = min(range(len(path)), key=lambda i: np.hypot(path[i, 0] - state[0].item(), path[i, 1] - state[1].item()))
        # Ensure the index is not less than the current index
        ind = max(cind, ind)

        # Generate the rest of the reference trajectory
        travel = lookahead_distance

        for i in range(horizon + 1):
            travel += reference_path_interval
            dind = int(round(travel / DL))

            if (ind + dind) < ncourse:
                xref[i, :3] = path[ind + dind]
                xref[i, 3] = self.env.V_MAX
            else:
                xref[i, :3] = path[-1]
                # set the target velocity to zero if the vehicle reaches the end of the path
                xref[:, 3] = 0.0

        return xref, ind


def main(save_mode: bool = False):
    env = RacingEnv()

    # controller
    controller = racing_controller(env, debug=True)
    controller.set_cost_map(env._obstacle_map, env._lane_map)

    state = env.reset()
    max_steps = 500
    average_time = 0
    for i in range(max_steps):
        action_seq, state_seq = controller.update(state, env.racing_center_path)

        state, is_goal_reached = env.step(action_seq[0, :])

        is_collisions = env.collision_check(state=state_seq)

        top_samples, top_weights = controller.get_top_samples(num_samples=300)

        if save_mode:
            env.render(
                action=action_seq[0, :],
                predicted_trajectory=state_seq,
                is_collisions=is_collisions,
                top_samples=(top_samples, top_weights),
                reference_trajectory=controller.reference_path,
                mode="rgb_array",
            )
            # progress bar
            if i == 0:
                pbar = tqdm.tqdm(total=max_steps, desc="recording video")
            pbar.update(1)

        else:
            env.render(
                action=action_seq[0, :],
                predicted_trajectory=state_seq,
                is_collisions=is_collisions,
                top_samples=(top_samples, top_weights),
                reference_trajectory=controller.reference_path,
                mode="human",
            )
        if is_goal_reached:
            print("Goal Reached!")
            break

    print("average solve time: {}".format(average_time * 1000), " [ms]")
    env.close()  # close window and save video if save_mode is True


if __name__ == "__main__":
    fire.Fire(main)
