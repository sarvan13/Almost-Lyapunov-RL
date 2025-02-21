import gymnasium as gym
import numpy as np
from lsac.agent import LSACAgent
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

env = gym.make('Pendulum-v1')
agent = LSACAgent(state_dims=env.observation_space.shape[0], action_dims=env.action_space.shape[0],
                max_action=env.action_space.high)
agent.load()

theta_arr = np.linspace(-np.pi, np.pi, 100)
theta_dot_arr = np.linspace(-7, 7, 100)

theta_grid, theta_dot_grid = np.meshgrid(theta_arr, theta_dot_arr)
lyapunov_vals = np.zeros_like(theta_grid)

eq_state = torch.tensor([np.zeros(agent.state_dims)], dtype=torch.float).to(agent.actor.device)
eq_action, _ = agent.actor.sample(eq_state, False)
equilibrium_lyapunov = agent.lyapunov(eq_state, eq_action)
print(equilibrium_lyapunov)

for _ in range(100):
    for i, theta in enumerate(theta_arr):
        for j, theta_dot in enumerate(theta_dot_arr):
            state = np.array([np.cos(theta), np.sin(theta), theta_dot])
            action = agent.choose_action(state)
            lyapunov_val = agent.lyapunov(torch.tensor([state], dtype=torch.float).to(agent.actor.device), torch.tensor([action], dtype=torch.float).to(agent.actor.device))
            lyapunov_vals[j, i] += lyapunov_val

for i in range(100):
    for j in range(100):
        lyapunov_vals[j, i] /= 100 # Average over 100 runs

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta_grid, theta_dot_grid, lyapunov_vals, cmap='viridis')

ax.set_xlabel('Theta')
ax.set_ylabel('Theta Dot')
ax.set_zlabel('Lyapunov Value')
ax.set_title('3D Plot of Theta, Theta Dot, and Lyapunov Value')

plt.show()