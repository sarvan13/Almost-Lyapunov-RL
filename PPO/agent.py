import torch
import gymnasium as gym
import numpy as np
from memory import Memory
from networks import ActorNet, CriticNet

class PPOAgent():
    def __init__(self, env_name, a_lr, c_lr, batch_size, 
                 memory_size, num_grad_updates, epsilon, gamma = 0.99, max_episodes=500, max_steps=1e6,
                 render = False):
        self.env_name = env_name
        self.render = render
        self.a_lr = a_lr
        self.c_lr = c_lr
        self.batch_size = batch_size
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.memory = Memory(memory_size, gamma)
        self.num_grad_updates = num_grad_updates

        if self.render:
            self.env = gym.make(self.env_name, render_mode="Human")
        else:
            self.env = gym.make(self.env_name)

        self.state_dims = self.env.observation_space.shape[0]
        self.action_dims = self.env.action_space.shape[0]
        self.max_action = self.env.action_space.high

        self.policy = ActorNet(self.state_dims, self.action_dims, self.max_action, a_lr)
        self.critic = CriticNet(self.state_dims, c_lr)

    def train(self):
        curr_ep = 0
        step = 0
        episode_rewards = []
        p_loss_arr = []
        v_loss_arr = []
        ep_rew = 0

        state, _ = self.env.reset()
        done = False
        while step < self.max_steps and curr_ep < self.max_episodes:
            while self.memory.full == False:
                action, log_prob = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                value = self.critic(torch.tensor([state], dtype=torch.float))
                self.memory.store((state, action, reward, next_state, done, log_prob, value), self.critic)
                ep_rew += reward

                if done:
                    state, _ = self.env.reset()
                    curr_ep += 1
                    episode_rewards.append(ep_rew)
                    ep_rew = 0
                
                step += 1
                state = next_state
        
            for i in range(self.num_grad_updates):
                policy_loss, value_loss = self.learn()
                p_loss_arr.append(policy_loss)
                v_loss_arr.append(value_loss)
            
            self.memory.clear()

        return episode_rewards, p_loss_arr, v_loss_arr
    
    def choose_action(self, state, reparameterize=False):
        state = torch.tensor([state], dtype=torch.float).to(self.policy.device)
        action, log_prob = self.policy.sample(state, reparameterize)
        return action.cpu().detach().numpy()[0], log_prob

    def learn(self):
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones, log_probs, values, advantages = zip(*batch)

        states = torch.tensor(states, dtype=torch.float).to(self.policy.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.policy.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.policy.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.policy.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.policy.device)
        old_log_probs = torch.tensor(log_probs, dtype=torch.float).to(self.policy.device)
        values = torch.tensor(values, dtype=torch.float).to(self.policy.device)
        advantages = torch.tensor(advantages, dtype=torch.float).to(self.policy.device)

        # Calculate Advantages
        returns = advantages + values

        # Calculate Policy Loss
        new_log_probs = self.policy.get_log_prob(states, actions)
        ratio = (new_log_probs - old_log_probs).exp()
        policy_loss1 = ratio * advantages
        policy_loss2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -torch.min(policy_loss1, policy_loss2).mean()

        # Calculate Value Loss
        value_loss = (0.5*(self.critic(states) - returns) ** 2).mean()

        # Update NNs
        self.policy.optimizer.zero_grad()
        policy_loss.backward()
        self.policy.optimizer.step()

        self.critic.optimizer.zero_grad()
        value_loss.backward()
        self.critic.optimizer.step()

        return policy_loss.item(), value_loss.item()