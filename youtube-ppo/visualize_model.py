import gymnasium as gym
import numpy as np
from ppo import Agent
import matplotlib.pyplot as plt
import torch as T
import copy

env = gym.make('Pendulum-v1', render_mode='human')
N = 20
batch_size = 5
n_epochs = 4
alpha = 0.0003
agent = Agent(n_actions=env.action_space.shape[0], batch_size=batch_size, 
                alpha=alpha, n_epochs=n_epochs, 
                input_dims=env.observation_space.shape[0],
                max_action=env.action_space.high)
agent.load_models()
n_games = 2

for i in range(n_games):
    print('Game:', i)
    observation, _ = env.reset()
    done = False
    while not done:
        action, prob, val = agent.choose_action(observation)
        observation_, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        observation = observation_