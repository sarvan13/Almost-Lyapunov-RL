import matplotlib
matplotlib.use('Agg')
import matplotlib
matplotlib.use('Agg')
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Create the environment
env = gym.make('Pendulum-v1')

# Define a callback to log rewards and steps
class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.steps = []
        self.current_episode_reward = 0

    def _on_step(self) -> bool:
        # Accumulate rewards for the current episode
        self.current_episode_reward += self.locals['rewards'][0]
        
        # Check if the episode is done
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.steps.append(self.num_timesteps)
            self.current_episode_reward = 0  # Reset for the next episode
        
        return True

# Instantiate the callback
reward_callback = RewardCallback()

# Instantiate the PPO model
model = PPO('MlpPolicy', env, verbose=1)

# Train the model
num_episodes = 500
num_steps = 200 * num_episodes
model.learn(total_timesteps=num_steps, callback=reward_callback)

# Close the environment
env.close()

# Plot the rewards
plt.figure(figsize=(10, 5))
plt.plot(reward_callback.steps, reward_callback.episode_rewards)
plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("Training Rewards for PPO on Pendulum-v1")
plt.grid(True)

# Save the plot to a file
plt.savefig('training_rewards.png')

# Save the rewards and steps to a file
np.savez('training_data.npz', rewards=reward_callback.episode_rewards, steps=reward_callback.steps)