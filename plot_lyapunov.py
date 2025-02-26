import gymnasium as gym
import numpy as np
from lsac.agent import LSACAgent
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import torch

env = gym.make('Pendulum-v1')
agent = LSACAgent(state_dims=env.observation_space.shape[0], action_dims=env.action_space.shape[0],
                max_action=env.action_space.high)

def plot3D(agent):
    with torch.no_grad():
        agent.load()
        theta_arr = np.linspace(-np.pi, np.pi, 100)
        theta_dot_arr = np.linspace(-7, 7, 100)

        theta_grid, theta_dot_grid = np.meshgrid(theta_arr, theta_dot_arr)
        lyapunov_vals = np.zeros_like(theta_grid)

        eq_state = torch.tensor([np.zeros(agent.state_dims)], dtype=torch.float).to(agent.actor.device)
        eq_action, _ = agent.actor.forward(eq_state)
        equilibrium_lyapunov = agent.lyapunov.forward(eq_state, eq_action)
        print(equilibrium_lyapunov)

        for i, theta in enumerate(theta_arr):
            for j, theta_dot in enumerate(theta_dot_arr):
                state = np.array([np.cos(theta), np.sin(theta), theta_dot])
                action, _ = agent.actor.forward(torch.tensor([state], dtype=torch.float).to(agent.actor.device))
                lyapunov_val = agent.lyapunov(torch.tensor([state], dtype=torch.float).to(agent.actor.device), action)
                lyapunov_vals[j, i] += lyapunov_val.item()

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(theta_grid, theta_dot_grid, lyapunov_vals, cmap='viridis')

        ax.set_xlabel('Theta')
        ax.set_ylabel('Theta Dot')
        ax.set_zlabel('Lyapunov Value')
        ax.set_title('LSAC Lyapunov Function')

        plt.show()

def plot2D(agent):
    with torch.no_grad():
        agent.load()
        theta_arr = np.linspace(-np.pi, np.pi, 100)
        theta_dot_arr = np.linspace(-7, 7, 100)

        theta_grid, theta_dot_grid = np.meshgrid(theta_arr, theta_dot_arr)
        lyapunov_vals = np.zeros_like(theta_grid)

        eq_state = torch.tensor([np.zeros(agent.state_dims)], dtype=torch.float).to(agent.actor.device)
        eq_action, _ = agent.actor.forward(eq_state)
        equilibrium_lyapunov = agent.lyapunov.forward(eq_state, eq_action)
        print(equilibrium_lyapunov)

        for i, theta in enumerate(theta_arr):
            for j, theta_dot in enumerate(theta_dot_arr):
                state = np.array([np.cos(theta), np.sin(theta), theta_dot])
                action, _ = agent.actor.forward(torch.tensor([state], dtype=torch.float).to(agent.actor.device))
                lyapunov_val = agent.lyapunov(torch.tensor([state], dtype=torch.float).to(agent.actor.device), action)
                lyapunov_vals[j, i] += lyapunov_val.item()

        # Define custom contour levels
        # levels = np.concatenate([
        #     np.linspace(0, 0.1, 10),  # More contours between 0 and 0.1
        #     np.linspace(0.11, 1.2, 5)  # Fewer contours between 0.1 and 1.2
        # ])
        # Define custom contour levels
        levels = np.linspace(0, 1.5, 25)  # More contours between 0 and 0.1

        # Create a custom colormap
        colors = [(0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]  # Blue, Green, Yellow, Red
        n_bins = 100  # Discretizes the interpolation into bins
        cmap_name = 'custom_cmap'
        cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

        # Create a 2D contour plot
        fig, ax = plt.subplots()
        contour_filled = ax.contourf(theta_grid, theta_dot_grid, lyapunov_vals, levels=levels, cmap=cm)
        contour_lines = ax.contour(theta_grid, theta_dot_grid, lyapunov_vals, levels=levels, colors='black')


        ax.set_xlabel('Theta')
        ax.set_ylabel('Theta Dot')
        ax.set_title('LSAC Lyapunov Function')

        # Add a color bar
        cbar = fig.colorbar(contour_filled)
        cbar.set_label('Lyapunov Value')

        plt.show()


plot2D(agent)