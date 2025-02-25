import gymnasium as gym
import numpy as np
from ppo import PPOAgent
import matplotlib.pyplot as plt
import torch as T
import copy

if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    N = 2048
    batch_size = 64
    n_epochs = 10
    alpha = 0.0003
    agent = PPOAgent(n_actions=env.action_space.shape[0], batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, dt=env.unwrapped.dt,
                    input_dims=env.observation_space.shape[0],
                    max_action=env.action_space.high)
    n_games = 1000

    score_history = []
    actor_loss = []
    critic_loss = []
    std = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    best_score = -2000

    for i in range(n_games):
        observation, _ = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, observation_, done)
            if n_steps % N == 0:
                a_loss, c_loss = agent.learn()
                actor_loss.append(a_loss)
                critic_loss.append(c_loss)
                learn_iters += 10
            observation = observation_
        agent.actor.decay_covariance(n_games)
        score_history.append(score)
        std.append(agent.actor.cov_var[0].item())
        avg_score = np.mean(score_history[-100:])

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
        
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
    
    plt.plot(np.arange(len(score_history)), score_history)
    plt.plot(np.arange(len(score_history)), 1000*np.array(std))
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.show()

    loss_steps = N*np.arange(len(actor_loss))
    plt.plot(loss_steps, actor_loss)
    plt.xlabel("Time Steps")
    plt.ylabel("Policy Loss")
    plt.show()

    plt.plot(loss_steps, critic_loss)
    plt.xlabel("Time Steps")
    plt.ylabel("Value Loss")
    plt.show()

    np.save('ppo-reward.npy', score_history)
