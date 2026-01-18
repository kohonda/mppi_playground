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
        """
        Args:
            state (torch.Tensor): [x, x_dt, theta, theta_dt]
            action (torch.Tensor): [-1, 1]
        """
        # dynamics from gymnasium
        x = state[:, 0].view(-1, 1)
        x_dt = state[:, 1].view(-1, 1)
        theta = state[:, 2].view(-1, 1)
        theta_dt = state[:, 3].view(-1, 1)

        gravity = 9.8
        masscart = 1.0
        masspole = 0.1
        total_mass = masspole + masscart
        length = 0.5  # actually half the pole's length
        polemass_length = masspole * length
        force_mag = 10.0
        tau = 0.02  # seconds between state updates

        # convert continuous action to discrete action
        # because MPPI only can handle continuous action
        continuous_action = action[:, 0].view(-1, 1)
        force = torch.zeros_like(continuous_action)
        force[continuous_action >= 0] = force_mag
        force[continuous_action < 0] = -force_mag

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

        x_threshold = 2.4
        theta_threshold_radians = 12 * 2 * torch.pi / 360
        newx = torch.clamp(newx, -x_threshold, x_threshold)
        newtheta = torch.clamp(
            newtheta, -theta_threshold_radians, theta_threshold_radians
        )

        new_state = torch.cat((newx, newx_dt, newtheta, newtheta_dt), dim=1)

        return new_state

    def stage_cost(state: torch.Tensor, action: torch.Tensor, info) -> torch.Tensor:
        x = state[:, 0]
        # x_dt = state[:, 1]
        theta = state[:, 2]
        theta_dt = state[:, 3]

        normlized_theta = angle_normalize(theta)

        cost = normlized_theta**2 + 0.1 * theta_dt**2 + 0.1 * x**2

        return cost

    # simulator
    if save_mode:
        env = gymnasium.make("CartPole-v1", render_mode="rgb_array")
        env = gymnasium.wrappers.RecordVideo(env=env, video_folder="video")
    else:
        env = gymnasium.make("CartPole-v1", render_mode="human")
    observation, _ = env.reset(seed=42)

    # start from the inverted position
    # env.unwrapped.state = np.array([0.0, 0.0, np.pi, 0.0])
    # observation, _, _, _, _ = env.step(0)

    # solver
    solver = MPPI(
        horizon=10,
        num_samples=100,
        dim_state=4,
        dim_control=1,
        dynamics=dynamics,
        cost_func=stage_cost,
        u_min=torch.tensor([-3.0]),
        u_max=torch.tensor([3.0]),
        sigmas=torch.tensor([1.0]),
        lambda_=0.001,
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

        # convert continuous action to discrete action
        discrete_action = 0 if action_seq_np[0, 0] < 0 else 1

        # update simulator
        observation, reward, terminated, truncated, info = env.step(discrete_action)
        env.render()

    print("average solve time: {}".format(average_time * 1000), " [ms]")
    env.close()


if __name__ == "__main__":
    fire.Fire(main)
