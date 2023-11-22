import torch

import time
import gymnasium
import fire
import numpy as np

from controller.mppi import MPPI


@torch.jit.script
def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi


def main(save_mode: bool = False):
    # dynamics and cost
    @torch.jit.script
    def dynamics(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # dynamics from gymnasium
        th = state[:, 0].view(-1, 1)
        thdot = state[:, 1].view(-1, 1)
        g = 10
        m = 1
        l = 1
        dt = 0.05
        u = action
        u = torch.clamp(u, -2, 2)
        newthdot = (
            thdot
            + (-3 * g / (2 * l) * torch.sin(th + torch.pi) + 3.0 / (m * l**2) * u)
            * dt
        )
        newth = th + newthdot * dt
        newthdot = torch.clamp(newthdot, -8, 8)

        state = torch.cat((newth, newthdot), dim=1)
        return state

    @torch.jit.script
    def stage_cost(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        theta = state[:, 0]
        theta_dt = state[:, 1]
        # u = action[:, 0]
        cost = angle_normalize(theta) ** 2 + 0.1 * theta_dt**2
        return cost

    @torch.jit.script
    def terminal_cost(state: torch.Tensor) -> torch.Tensor:
        theta = state[:, 0]
        theta_dt = state[:, 1]
        cost = angle_normalize(theta) ** 2 + 0.1 * theta_dt**2
        return cost

    # simulator
    if save_mode:
        env = gymnasium.make("Pendulum-v1", render_mode="rgb_array")
        env = gymnasium.wrappers.RecordVideo(env=env, video_folder="video")
    else:
        env = gymnasium.make("Pendulum-v1", render_mode="human")
    observation, _ = env.reset(seed=42)

    # solver
    solver = MPPI(
        horizon=15,
        delta=0.05,
        num_samples=1000,
        dim_state=2,
        dim_control=1,
        dynamics=dynamics,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        u_min=torch.tensor([-2.0]),
        u_max=torch.tensor([2.0]),
        sigmas=torch.tensor([1.0]),
        lambda_=1.0,
    )

    average_time = 0
    for i in range(200):
        state = env.unwrapped.state.copy()

        # solve
        start = time.time()
        action_seq, state_seq = solver.forward(state=state)
        elipsed_time = time.time() - start
        average_time = i / (i + 1) * average_time + elipsed_time / (i + 1)

        action_seq_np = action_seq.cpu().numpy()
        state_seq_np = state_seq.cpu().numpy()

        # update simulator
        observation, reward, terminated, truncated, info = env.step(action_seq_np[0, :])
        env.render()

    print("average solve time: {}".format(average_time * 1000), " [ms]")
    env.close()


if __name__ == "__main__":
    fire.Fire(main)
