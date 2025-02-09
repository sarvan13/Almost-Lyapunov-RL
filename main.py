from agent import PPOAgent
import numpy as np
import matplotlib.pyplot as plt

agent = PPOAgent("InvertedPendulum-v5", 3e-4, 3e-4, 5, 20, 4, 0.2, render=False)

reward_arr, p_loss_arr, v_loss_arr = agent.train()

plt.plot(np.arange(len(reward_arr)), reward_arr)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.show()

plt.plot(np.arange(len(p_loss_arr)), p_loss_arr)
plt.xlabel("Episodes")
plt.ylabel("Policy Loss")
plt.show()

plt.plot(np.arange(len(v_loss_arr)), v_loss_arr)
plt.xlabel("Episodes")
plt.ylabel("Value Loss")
plt.show()

np.save("ppo_rewards.npy", reward_arr)