import os
import random

import fire
import gymnasium as gym
import numpy as np
import torch
import tqdm

from envs.goal_in_danger_zone import GoalInDangerZoneEnv
from pi_mpc.mppi import MPPI


def main(save_mode: bool = False):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if save_mode:
        video_dir = "videos"
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        env = GoalInDangerZoneEnv(render_mode="rgb_array", seed=seed)
        env = gym.wrappers.RecordVideo(env, video_dir)
    else:
        env = GoalInDangerZoneEnv(render_mode="human", seed=seed)

    # solver
    solver = MPPI(
        horizon=30,
        num_samples=3000,
        dim_state=7,
        dim_control=2,
        dynamics=env.parallel_step,
        cost_func=env.parallel_cost,
        u_min=torch.tensor([-1.0, -1.0]),
        u_max=torch.tensor([1.0, 1.0]),
        sigmas=torch.tensor([0.5, 0.5]),
        lambda_=1.0,
    )

    obs, info = env.reset(seed=seed)

    if save_mode:
        env.start_video_recorder()

    max_steps = env.max_episode_steps
    episodic_reward = 0
    episodic_cost = 0
    for i in range(max_steps):
        obs = torch.tensor(obs, dtype=torch.float32)
        action_seq, predicted_traj = solver.forward(state=obs)

        action = action_seq[0, :].cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action)
        episodic_reward += reward
        episodic_cost += info["cost"]

        predicted_traj_np = predicted_traj[0, :, :2].cpu().numpy()
        top_samples, top_weights = solver.get_top_samples(num_samples=100)
        is_collision = info["cost"] > 0.0

        top_samples_np = top_samples.cpu().numpy()
        top_weights_np = top_weights.cpu().numpy()
        env.set_render_info(
            is_colllision=is_collision,
            predicted_trajectory=predicted_traj_np,
            top_samples=(top_samples_np, top_weights_np),
        )
        env.render()

        # progress bar
        if save_mode:
            if i == 0:
                pbar = tqdm.tqdm(total=max_steps, desc="recording video")
            pbar.update(1)

        if truncated or terminated:
            obs, info = env.reset()

    if save_mode:
        env.close_video_recorder()

    print("episodic reward: ", episodic_reward)
    print("episodic cost: ", episodic_cost)
    env.close()


if __name__ == "__main__":
    fire.Fire(main)
