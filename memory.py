from collections import deque
import random
import numpy as np
import torch

class Memory():
    def __init__(self, mem_length, gamma):
        self.mem_length = int(mem_length)
        self.gamma = gamma
        self.memory = deque(maxlen=self.mem_length)
        self.current_path = deque(maxlen=self.mem_length)
        self.full = False
    
    def length(self):
        return len(self.memory)
    
    def store(self, data, critic):
        if self.full:
            return
        self.current_path.append(data)
        state, action, reward, next_state, done, log_prob, value = data
        next_state = torch.tensor([next_state], dtype=torch.float).to(critic.device)
       
        if done or (len(self.current_path) + len(self.memory) >= self.mem_length):
            #mc_returns = np.zeros(len(self.current_path))
            advantages = np.zeros(len(self.current_path))
            rewards = [data[2] for data in self.current_path]
            rewards = np.array(rewards)
            
            # curr_return = 0
            # for i in range(1, len(rewards) + 1):
            #     if i == 1 and not done:
            #         curr_return = reward + self.gamma*critic(next_state).detach()
            #     else:
            #         curr_return = rewards[-i] + self.gamma*curr_return
                    
            #     mc_returns[-i] = curr_return
            for i in range(len(rewards)):
                a_t = 0
                discount = 1
                for j in range(i, len(rewards)):
                    a_t += discount * (rewards[j] + self.gamma*self.current_path[j+1][6] * (1 - self.current_path[j][4]) \
                        - self.current_path[j][6])
                    discount *= self.gamma
                
                advantages[i] = a_t

            for tup, value in zip(self.current_path, advantages):
                advantages = (advantages - advantages.mean() / 
                              (advantages.std() + 1e-8))
                for tup, adv in zip(self.current_path, advantages):
                    self.memory.append((*tup, adv))
            
            self.current_path.clear()

            if len(self.memory) >= self.mem_length:
                self.full = True


    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def clear(self):
        self.memory.clear()
        self.current_path.clear()
        self.full = False

