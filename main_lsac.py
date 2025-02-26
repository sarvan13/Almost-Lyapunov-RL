# Run Cart Pole env
import torch

import gymnasium as gym
import env
from lsac.agent import LSACAgent
from tqdm import tqdm
import numpy as np

environment = gym.make('Pendulum-v1')
print(environment.action_space.high)
agent = LSACAgent(environment.observation_space.shape[0], environment.action_space.shape[0], environment.action_space.high)

state, info = environment.reset(seed=42)
max_num_episodes = 1000
max_episode_length = 200
cost_arr = []
step_arr = []
steps_per_episode = []
total_steps = 0
longest_episode = 0

lyapunov_loss = []
q_loss = []
v_loss = []
actor_loss = []
beta_arr = []

for k in (range(max_num_episodes)):
    episode_cost = 0
    episode_steps = 0
    for i in range(max_episode_length):
        action = agent.choose_action(state, reparameterize=False)
        next_state, cost, terminated, truncated, _ = environment.step(action)

        agent.remember((state, action, cost, next_state, terminated))

        state = next_state

        episode_cost += cost
        episode_steps += 1
        total_steps += 1

        if terminated:
            break

    state, _ = environment.reset()
    
    for j in range(episode_steps):
        l_loss = agent.learn_lyapunov()
        losses = agent.train()
        if losses is not None:
            v, a, q = losses
            lyapunov_loss.append(l_loss.item())
            v_loss.append(v.item())
            actor_loss.append(a.item())
            q_loss.append(q.item())
    
    beta_arr.append(agent.beta.item())

    print(f"Episode {k} - Cost: {episode_cost} - Beta: {agent.beta.item()}")
    steps_per_episode.append(episode_steps)
    cost_arr.append(episode_cost)
    step_arr.append(total_steps)

np.save("lsac-pendulum-cost2-arr.npy", np.array(cost_arr))
np.save("lsac-pendulum-step2-arr.npy", np.array(step_arr))
np.save("lsac-pendulum-lyapunov-loss2-arr.npy", np.array(lyapunov_loss))
np.save("lsac-pendulum-v-loss2-arr.npy", np.array(v_loss))
np.save("lsac-pendulum-q-loss2-arr.npy", np.array(q_loss))
np.save("lsac-pendulum-actor-loss2-arr.npy", np.array(actor_loss))
np.save("lsac-pendulum-beta-arr.npy", np.array(actor_loss))
agent.save()