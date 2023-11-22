import gymnasium
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

# Run gymnasium with real-time rendering
env = gymnasium.make("Pendulum-v1", render_mode="human")
_, _ = env.reset(seed=42)
for _ in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        observation, info = env.reset()
env.close()

# video recording mode
env = gymnasium.make("Pendulum-v1", render_mode="rgb_array")
env = gymnasium.wrappers.RecordVideo(env=env, video_folder="video")
_, _ = env.reset(seed=42)
env.start_video_recorder()
for _ in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        observation, info = env.reset()
env.close()

# Run matplotlib
plt.style.use("ggplot")
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1000), np.random.randn(1000))
plt.show()
