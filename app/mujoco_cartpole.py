import time

import fire
import gymnasium as gym
import torch

from mppi_playground import MPPI


@torch.jit.script
def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi


# Not work well because of the difference of dynamics
# I should use the true dynamics from mujoco like:
# https://github.com/mohakbhardwaj/mjmpc/blob/master/examples/example_mpc.py#L112
def main(save_mode: bool = False):
    # dynamics and cost
    @torch.jit.script
    def dynamics(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state (torch.Tensor): [x, x_dt, theta, theta_dt]
            action (torch.Tensor): [-1, 1]
        """
        # dynamics not from mujoco
        # https://github.com/openai/gym/blob/master/gym/envs/mujoco/assets/inverted_pendulum.xml
        x = state[:, 0].view(-1, 1)
        x_dt = state[:, 1].view(-1, 1)
        theta = state[:, 2].view(-1, 1)
        theta_dt = state[:, 3].view(-1, 1)

        force = action[:, 0].view(-1, 1)

        gravity = 9.8
        masscart = 1.0
        masspole = 1.0
        total_mass = masspole + masscart
        length = 0.5  # actually half the pole's length
        polemass_length = masspole * length
        tau = 0.02  # seconds between state updates

        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        temp = (force + polemass_length * theta_dt**2 * sintheta) / total_mass
        thetaacc = (gravity * sintheta - costheta * temp) / (
            length * (4.0 / 3.0 - masspole * costheta**2 / total_mass)
        )
        xacc = temp - polemass_length * thetaacc * costheta / total_mass

        newx = x + tau * x_dt
        newx_dt = x_dt + tau * xacc
        newtheta = theta + tau * theta_dt
        newtheta_dt = theta_dt + tau * thetaacc

        x_threshold = 1.0
        theta_threshold_radians = 12 * 2 * torch.pi / 360
        newx = torch.clamp(newx, -x_threshold, x_threshold)
        newtheta = torch.clamp(
            newtheta, -theta_threshold_radians, theta_threshold_radians
        )

        new_state = torch.cat((newx, newx_dt, newtheta, newtheta_dt), dim=1)

        return new_state

    def cost_func(state: torch.Tensor, action: torch.Tensor, info) -> torch.Tensor:
        x = state[:, 0]
        # x_dt = state[:, 1]
        theta = state[:, 2]
        theta_dt = state[:, 3]

        normlized_theta = angle_normalize(theta)

        cost = normlized_theta**2 + 0.1 * theta_dt**2 + 0.1 * x**2

        return cost

    # simulator
    if save_mode:
        env = gym.make("InvertedPendulum-v4", render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env=env, video_folder="video")
    else:
        env = gym.make("InvertedPendulum-v4", render_mode="human")

    observation, _ = env.reset(seed=42)

    # start from the inverted position
    # env.unwrapped.state = np.array([0.0, 0.0, np.pi / 8, 0.0])
    # observation, _, _, _, _ = env.step(0)

    # solver
    solver = MPPI(
        horizon=50,
        num_samples=1000,
        dim_state=4,
        dim_control=1,
        dynamics=dynamics,
        cost_func=cost_func,
        u_min=torch.tensor([-3.0]),
        u_max=torch.tensor([3.0]),
        sigmas=torch.tensor([1.0]),
        lambda_=1.0,
    )

    average_time = 0
    for i in range(500):
        # solve
        start = time.time()
        action_seq, state_seq = solver.forward(state=observation)

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
