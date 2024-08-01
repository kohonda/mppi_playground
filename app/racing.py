import torch

import time

# import gymnasium
import fire
import tqdm

from controller.mppi import MPPI
from envs.racing_env import RacingEnv

from envs.reference_trajectory import calc_ref_trajectory


def main(save_mode: bool = False):
    env = RacingEnv()

    # solver
    solver = MPPI(
        horizon=20,
        num_samples=4000,
        dim_state=4,
        dim_control=2,
        dynamics=env.dynamics,
        cost_func=env.cost_function,
        u_min=env.u_min,
        u_max=env.u_max,
        sigmas=torch.tensor([0.5, 0.1]),
        lambda_=1.0,
        auto_lambda=False,
    )

    state = env.reset()
    max_steps = 500
    average_time = 0
    current_path_index = 0
    for i in range(max_steps):
        reference_path, current_path_index = calc_ref_trajectory(
            state, env._reference_path, current_path_index, solver._horizon, DL=0.1, lookahed_distance=1.0, reference_path_interval=0.85
        )
        start = time.time()
        # reference_pathをdict形式に変換
        ref = {
            "reference_path": reference_path,
        }
        action_seq, state_seq = solver.forward(state=state, info=ref)
        end = time.time()
        average_time += (end - start) / max_steps

        state, is_goal_reached = env.step(action_seq[0, :])

        is_collisions = env.collision_check(state=state_seq)

        top_samples, top_weights = solver.get_top_samples(num_samples=300)

        if save_mode:
            env.render(
                action=action_seq[0, :],
                predicted_trajectory=state_seq,
                is_collisions=is_collisions,
                top_samples=(top_samples, top_weights),
                reference_trajectory=reference_path,
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
                reference_trajectory=reference_path,
                mode="human",
            )
        if is_goal_reached:
            print("Goal Reached!")
            break

    print("average solve time: {}".format(average_time * 1000), " [ms]")
    env.close()  # close window and save video if save_mode is True


if __name__ == "__main__":
    fire.Fire(main)
