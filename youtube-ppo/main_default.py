import gymnasium as gym
import numpy as np
from ppo_torch import Agent
import matplotlib.pyplot as plt
import torch as T
import copy

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape)
    n_games = 300

    figure_file = 'plots/cartpole.png'
    score_history = []
    actor_loss = []
    critic_loss = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    # initial_actor_weights = copy.deepcopy(agent.actor.state_dict())
    # initial_value_weights = copy.deepcopy(agent.critic.state_dict())

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
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                a_loss, c_loss = agent.learn()
                actor_loss.append(a_loss)
                critic_loss.append(c_loss)
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    

    # current_actor_weights = agent.actor.state_dict()
    # current_value_weights = agent.critic.state_dict()

    # # Compare the initial and current weights
    # actor_weights_changed = any(not T.equal(initial_actor_weights[key], current_actor_weights[key]) for key in initial_actor_weights)
    # value_weights_changed = any(not T.equal(initial_value_weights[key], current_value_weights[key]) for key in initial_value_weights)

    # if actor_weights_changed:
    #     print("The weights of the 'actor' layer have changed.")
    # else:
    #     print("The weights of the 'actor' layer have not changed.")    
    # if value_weights_changed:
    #     print("The weights of the value layer have changed.")
    # else:
    #     print("The weights of the value layer have not changed.")
    x = [i+1 for i in range(len(score_history))]
    
    plt.plot(np.arange(len(score_history)), score_history)
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