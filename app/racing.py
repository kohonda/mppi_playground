import torch
import numpy as np
from scipy.spatial import KDTree

import time

# import gymnasium
import fire
import tqdm

from controller.mppi import MPPI
from envs.racing_env import RacingEnv

class racing_controller:
    def __init__(self, env, debug=False, device=torch.device("cuda"), dtype=torch.float32):
        self.debug = debug
        self.current_path_index = 0

        # solver
        self.solver = MPPI(
            horizon=25,
            num_samples=3000,
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

        # device and dtype
        if torch.cuda.is_available() and device == torch.device("cuda"):
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        self._dtype = dtype

        # reference indformation (tensor)
        self.reference_path: torch.Tensor = None
        self.obstacle_map: torch.Tensor = None
        self.lane_map: torch.Tensor = None

    def update(self, state, racing_center_path):
        # reference
        reference_path_np, self.current_path_index = self.calc_ref_trajectory(
            state, racing_center_path, self.current_path_index, self.solver._horizon, DL=0.1, lookahead_distance=3, reference_path_interval=0.85
        )
        self.reference_path = torch.from_numpy(reference_path_np).to(self._device, self._dtype)

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
    
    def get_top_samples(self, num_samples=300):
        return self.solver.get_top_samples(num_samples=num_samples)
    
    def set_cost_map(self, obstacle_map, lane_map):
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
        # reference_path = torch.from_numpy(info["reference_path"]).to(self._device, self._dtype)
        prev_action = info["prev_action"]
        t = info["t"] # horizon number

        # path cost
        # contouring and lag error of path
        ec = torch.sin(self.reference_path[t, 2]) * (state[:, 0] - self.reference_path[t, 0]) - torch.cos(self.reference_path[t, 2]) * (state[:, 1] - self.reference_path[t, 1])
        el = -torch.cos(self.reference_path[t, 2]) * (state[:, 0] - self.reference_path[t, 0]) - torch.sin(self.reference_path[t, 2]) * (state[:, 1] - self.reference_path[t, 1])

        path_cost = 5 * ec ** 2 + 10 * el ** 2

        # compute obstacle cost from cost map
        pos_batch = state[:, :2].unsqueeze(1)  # (batch_size, 1, 2)
        obstacle_cost = self.obstacle_map.compute_cost(pos_batch).squeeze(1)  # (batch_size,)
        obstacle_cost += self.lane_map.compute_cost(pos_batch).squeeze(1)

        # input cost
        input_cost = 0.01 * action.pow(2).sum(dim=1)
        input_cost += 0.5 * (action - prev_action).pow(2).sum(dim=1)

        cost = path_cost + 10000 * obstacle_cost + input_cost

        return cost
    
    def calc_ref_trajectory(self, state, path, cind, horizon, DL=0.1, lookahead_distance=1.0, reference_path_interval=0.5):
        """
        Calculate the reference trajectory for the vehicle.

        Parameters:
        state : array-like
            The current state of the vehicle [x, y, yaw].
        path : array-like
            The global path [x, y, yaw] for the vehicle to follow.
        cind : int
            The current index on the path.
        horizon : int
            The number of points in the horizon.
        DL : float, optional
            The distance between points on the path.
        lookahead_distance : float, optional
            The initial lookahead distance.
        reference_path_interval : float, optional
            The distance between reference points on the path.

        Returns:
        xref : array-like
            The reference trajectory [x, y, yaw] for the vehicle.
        """

        ncourse = len(path)
        npath_state = path.shape[1]
        xref = np.zeros((horizon + 1, npath_state))

        # Calculate the nearest index using KD-Tree
        tree = KDTree(path[:, :2])
        _, ind = tree.query(state[:2])

        # Ensure the index is not less than the current index
        ind = max(cind, ind)

        # Generate the rest of the reference trajectory
        travel = lookahead_distance

        for i in range(horizon + 1):
            travel += reference_path_interval
            dind = int(round(travel / DL))

            if (ind + dind) < ncourse:
                xref[i] = path[ind + dind]
            else:
                xref[i] = path[-1]

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
