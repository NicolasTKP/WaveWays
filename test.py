# Install library first (run in terminal)
# pip install stable-baselines3 gymnasium[box2d]

import gymnasium as gym
from stable_baselines3 import PPO

# 1. Create environment
env = gym.make("CartPole-v1")

# 2. Initialize the agent
model = PPO("MlpPolicy", env, verbose=1)

# 3. Train the model
model.learn(total_timesteps=100_000)

# 4. Save and test the trained model
model.save("ppo_cartpole")
env = gym.make("CartPole-v1", render_mode="human")

obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, _ = env.reset()

env.close()
