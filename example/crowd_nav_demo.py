import time

import fire
import torch
import tqdm

from envs.crowd_navigation import CrowdNavigationEnv
from pi_mpc.dra_mppi import DRAMPPI
from pi_mpc.prediction import ConstantVelocityPredictor


def main(
    save_mode: bool = False,
    seed: int = 42,
    num_pedestrians: int = 8,
    horizon: int = 30,
    num_samples: int = 2500,
):
    render_mode = "rgb_array" if save_mode else "human"
    env = CrowdNavigationEnv(
        num_pedestrians=num_pedestrians,
        pedestrian_model="social_force",
        render_mode=render_mode,
        seed=seed,
    )
    predictor = ConstantVelocityPredictor(
        process_noise_std=torch.tensor([0.3, 0.3]), velocity_decay=1.0
    )

    solver = DRAMPPI(
        horizon=horizon,
        num_samples=num_samples,
        dim_state=3,
        dim_control=2,
        dynamics=env.dynamics,
        cost_func=env.cost_function,
        u_min=env.u_min,
        u_max=env.u_max,
        sigmas=torch.tensor([0.45, 0.5]),
        lambda_="ESSPS",
        num_mc_samples=256,
        collision_radius=env._robot_radius + env._pedestrian_radius,
        w_soft=45.0,
        w_hard=8000.0,
        risk_threshold=0.25,
        seed=seed,
    )

    obs, _ = env.reset(seed=seed)
    max_steps = env.max_episode_steps
    total_time = 0.0
    step_count = 0
    cumulative_reward = 0.0

    pbar = tqdm.tqdm(total=max_steps, disable=not save_mode, desc="recording video")
    for _ in range(max_steps):
        obs_th = torch.tensor(obs, device=solver._device, dtype=solver._dtype)
        robot_state = env.extract_robot_state(obs_th)
        ped_state = env.extract_pedestrian_state(obs_th)
        predicted_obstacles = predictor.predict_gaussian(
            pedestrian_state=ped_state, horizon=horizon, dt=env.dt
        )

        start = time.time()
        action_seq, state_seq = solver.forward(
            state=robot_state, info={"predicted_obstacles": predicted_obstacles}
        )
        total_time += time.time() - start
        step_count += 1

        action = action_seq[0].detach().cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward

        top_samples, top_weights = solver.get_top_samples(num_samples=200)
        mc_points = solver.get_mc_points_for_trajectory(state_seq)[0, 0].detach().cpu().numpy()

        env.set_render_info(
            predicted_trajectory=state_seq.detach().cpu().numpy(),
            top_samples=(
                top_samples.detach().cpu().numpy(),
                top_weights.detach().cpu().numpy(),
            ),
            predicted_obstacles=predicted_obstacles,
            mc_points=mc_points,
        )
        env.render()

        pbar.update(1)
        if terminated or truncated:
            print(
                f"terminated={terminated}, truncated={truncated}, "
                f"is_collision={info['is_collision']}, is_goal={info['is_goal_reached']}"
            )
            break

    pbar.close()
    avg_ms = 1000.0 * total_time / max(step_count, 1)
    print(f"average solve time: {avg_ms:.3f} ms")
    print(f"cumulative reward: {cumulative_reward:.3f}")
    env.close()


if __name__ == "__main__":
    fire.Fire(main)
