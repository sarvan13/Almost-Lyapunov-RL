import numpy as np
import matplotlib.pyplot as plt

# # Load the data from the npz file
# data = np.load('training_data.npz')
# rewards = np.array(data['rewards']).flatten()
# steps = np.array(data['steps']).flatten()

# # Calculate the moving average of the last 20 rewards for each entry
# moving_avg_rewards = [np.mean(rewards[max(0, i-20):i+1]) for i in range(len(rewards))]

ppo_rewards =np.load('ppo-rew-batch.npy')
moving_avg_rewards = [np.mean(ppo_rewards[max(0, i-20):i+1]) for i in range(len(ppo_rewards))]

ly_rewards = np.load('ly2-reward-batch.npy')
moving_avg_rewards2 = [np.mean(ly_rewards[max(0, i-20):i+1]) for i in range(len(ly_rewards))]

# Plot the rewards
plt.figure(figsize=(10, 5))
plt.plot(moving_avg_rewards)
plt.plot(moving_avg_rewards2)
plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.title("Training Rewards for PPO on Pendulum-v1")
plt.grid(True)

# Save the plot to a file
plt.show()