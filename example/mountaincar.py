import time

import fire
import gymnasium
import torch

from pi_mpc.mppi import MPPI


@torch.jit.script
def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi


def main(save_mode: bool = False):
    # dynamics and cost
    @torch.jit.script
    def dynamics(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # dynamics from gymnasium
        min_action = -1.0
        max_action = 1.0
        min_position = -1.2
        max_position = 0.6
        max_speed = 0.07
        # goal_position = 0.45
        # goal_velocity = 0.0
        power = 0.0015

        position = state[:, 0].view(-1, 1)
        velocity = state[:, 1].view(-1, 1)

        force = torch.clamp(action[:, 0].view(-1, 1), min_action, max_action)

        velocity += force * power - 0.0025 * torch.cos(3 * position)
        velocity = torch.clamp(velocity, -max_speed, max_speed)
        position += velocity
        position = torch.clamp(position, min_position, max_position)
        # if (position == min_position and velocity < 0):
        #     velocity = torch.zeros_like(velocity)

        new_state = torch.cat((position, velocity), dim=1)

        return new_state

    def cost_func(state: torch.Tensor, action: torch.Tensor, info) -> torch.Tensor:
        goal_position = 0.45
        # goal_velocity = 0.0

        position = state[:, 0]
        # velocity = state[:, 1]

        cost = (goal_position - position) ** 2
        # + 0.01 * (velocity-goal_velocity)**2

        return cost

    # simulator
    if save_mode:
        env = gymnasium.make("MountainCarContinuous-v0", render_mode="rgb_array")
        env = gymnasium.wrappers.RecordVideo(env=env, video_folder="video")
    else:
        env = gymnasium.make("MountainCarContinuous-v0", render_mode="human")
    observation, _ = env.reset(seed=42)

    # solver
    solver = MPPI(
        horizon=100,
        num_samples=1000,
        dim_state=2,
        dim_control=1,
        dynamics=dynamics,
        cost_func=cost_func,
        u_min=torch.tensor([-1.0]),
        u_max=torch.tensor([1.0]),
        sigmas=torch.tensor([1.0]),
        lambda_=0.1,
    )

    average_time = 0
    for i in range(300):
        state = env.unwrapped.state.copy()

        # solve
        start = time.time()
        action_seq, state_seq = solver.forward(state=state)
        elipsed_time = time.time() - start
        average_time = i / (i + 1) * average_time + elipsed_time / (i + 1)

        action_seq_np = action_seq.cpu().numpy()
        # state_seq_np = state_seq.cpu().numpy()

        # update simulator
        observation, reward, terminated, truncated, info = env.step(action_seq_np[0, :])
        env.render()

    print("average solve time: {}".format(average_time * 1000), " [ms]")
    env.close()


if __name__ == "__main__":
    fire.Fire(main)
