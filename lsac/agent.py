import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from .networks import ActorNet, QNet, ValueNet, LyapunovNet
import gymnasium as gym
from collections import deque, namedtuple
import random

class LSACAgent():
    def __init__(self, state_dims, action_dims, max_action, alr=1e-4, qlr=3e-4, vlr=3e-4, llr=3e-4, clr=3e-4, batch_size=256,
                 rewards_scale = 2, alpha = 0.2, gamma=1, tau=0.005, mem_length=1e5):
        self.actor = ActorNet(alr,state_dims, action_dims, max_action)
        self.q = QNet(qlr, state_dims, action_dims)
        self.value = ValueNet(vlr, state_dims)
        self.value_target = ValueNet(vlr, state_dims)
        self.lyapunov = LyapunovNet(llr, state_dims, action_dims)
        self.value_target.load_state_dict(self.value.state_dict())
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.dt = 0.05
        self.equilibrium_state = torch.tensor([np.array([np.cos(0), np.sin(0), 0])], dtype=torch.float).to(self.actor.device)
        # self.equilibrium_action = torch.tensor(torch.zeros(action_dims), dtype=torch.float, requires_grad=True).to(self.actor.device)

        self.max_action = max_action
        
        self.mem_length = mem_length
        self.replay_buffer = []
        # self.data_point = namedtuple()
        # self.replay_buffer = deque()

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.rewards_scale = rewards_scale
        beta = torch.tensor([10.0]).to(self.actor.device)
        self.log_beta = nn.Parameter(torch.tensor(torch.log(beta)))
        self.beta_optimizer = optim.Adam([self.log_beta], lr=clr)
        self.beta = torch.exp(self.log_beta.detach())

        self.loss = nn.MSELoss()

    def save(self):
        self.actor.save()
        self.value.save()
        self.value_target.save()
        self.q.save()
        self.lyapunov.save()
    def load(self):
        self.actor.load()
        self.value.load()
        self.value_target.load()
        self.q.load()
        self.lyapunov.load()

    def remember(self, data_point):
        self.replay_buffer.append(data_point)
        if len(self.replay_buffer) > self.mem_length:
            self.replay_buffer.pop(0)

    def choose_action(self, state, reparameterize=False):
        state = torch.tensor([state], dtype=torch.float).to(self.actor.device)
        action, _ = self.actor.sample(state, reparameterize)

        return action.cpu().detach().numpy()[0]
    
    def learn_lyapunov(self, batch_size=256):
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = random.sample(self.replay_buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float).to(self.lyapunov.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.lyapunov.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.lyapunov.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.lyapunov.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.lyapunov.device)

        
        next_actions, _ = self.actor.sample(next_states, False)
        eq_action, _ = self.actor.forward(self.equilibrium_state)

        lyapunov_values = self.lyapunov(states, actions)
        lie_derivative = (self.lyapunov(next_states, next_actions) - lyapunov_values)
        equilibrium_lyapunov = self.lyapunov(self.equilibrium_state, eq_action)

        loss = torch.max(torch.tensor(0), -lyapunov_values).mean() + torch.max(torch.tensor(0), lie_derivative/self.dt + 0.1).mean() + equilibrium_lyapunov**2

        self.lyapunov.optimizer.zero_grad()
        loss.backward()
        self.lyapunov.optimizer.step()

        return loss



    def train(self, batch_size=256):
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = random.sample(self.replay_buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float).to(self.actor.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.actor.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.actor.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.actor.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.actor.device)

        # Train Value Network
        # error = V(s) - E(Q(s,a) - log(pi(a|s)))
        self.value.optimizer.zero_grad()
        v = self.value.forward(states).view(-1)
        sampled_actions, log_probs = self.actor.sample(states, reparameterize=False)
        q_v = self.q.forward(states, sampled_actions).view(-1)
        target_v = q_v - log_probs.view(-1)
        v_loss = 0.5*self.loss(v, target_v.detach())
        v_loss.backward()
        self.value.optimizer.step()

        #Train Actor Network
        # error = log(pi(a|s)) - q(s,a) + max(0, L' - L)
        self.actor.optimizer.zero_grad()
        sampled_actions, log_probs = self.actor.sample(states, reparameterize=True)
        q_actor = self.q.forward(states, sampled_actions).view(-1)

        next_actions, _ = self.actor.sample(next_states, reparameterize=True)
        eq_action, _ = self.actor.forward(self.equilibrium_state)
        lie_derivative = (self.lyapunov.forward(next_states, next_actions) - self.lyapunov.forward(states, actions))/self.dt + 0.1
        l_equi = self.lyapunov.forward(self.equilibrium_state, eq_action)
        lyapunov_error = self.beta*torch.max(torch.tensor(0), lie_derivative).mean() #+ l_equi**2
        
        actor_loss = (log_probs.view(-1) - q_actor).mean()
        loss = actor_loss + lyapunov_error
        actor_loss.backward()
        lyapunov_error.backward()
        self.actor.optimizer.step()

        # Train Q Network
        # error = Q(s,a) - (r(s,a) + gamma* E(V'(s)))
        self.q.optimizer.zero_grad()
        q = self.q.forward(states, actions).view(-1)
        next_value = self.value_target.forward(next_states).view(-1)
        next_value = (1 - dones).view(-1) * next_value
        q_target = self.rewards_scale*rewards.view(-1) + self.gamma * next_value
        q_loss = 0.5*self.loss(q,q_target.detach())

        q_loss.backward()
        self.q.optimizer.step()

        # beta_loss = -self.log_beta*(lyapunov_error.detach())
        # self.beta_optimizer.zero_grad()
        # beta_loss.backward()
        # self.beta_optimizer.step()

        self.beta = torch.exp(self.log_beta) 

        # Update V Target Network
        # Update the target value network
        for target_param, param in zip(self.value_target.parameters(), self.value.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return v_loss, loss, q_loss



