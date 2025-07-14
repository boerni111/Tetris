import numpy as np
import torch
from torch.optim import Adam
from collections import deque
import random
from torch import nn
from src.model import DQN_model

class DQNAgent:
    def __init__(self,state_dim,action_dim,learning_rate,gamma,epsilon,epsilon_decay,buffer_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=buffer_size)
        self.model = DQN_model(state_dim,action_dim)
        self.optimizer = Adam(self.model.parameters(),lr=learning_rate)
        self.target_model = DQN_model(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.loss = nn.SmoothL1Loss()
        self.target_model.eval()
        self.steps_done = 0
        self.target_update_freq = 200
    def act(self,state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            with torch.no_grad():
                q_values = self.model(state.clone().detach().float())
            return np.argmax(q_values.detach().numpy())
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
    def replay(self,batch_size):
        self.model.train()
        if len(self.memory) < batch_size:
            return
        #Select minibatch
        batch = random.sample(self.memory,batch_size)
        state, action, reward, next_state, done = zip(*batch)
        state = torch.stack(state, dim=0).float()
        action = torch.tensor(action, dtype=torch.int64)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_states = torch.stack(next_state, dim=0).float()
        done = torch.tensor(done, dtype=torch.float32)
        target = reward
        
        #Calculating Q-value
        with torch.no_grad():
            next_q_values = self.target_model(next_states) 
            max_next_q, _ = next_q_values.max(dim=1, keepdim=True)
            max_next_q = max_next_q.squeeze()
            target = reward + self.gamma * max_next_q * (1 - done)

        action = action.unsqueeze(1)
        current_q_values = self.model(state).gather(1,action).squeeze()

        #Perform gradient descent
        self.optimizer.zero_grad()
        loss = self.loss(current_q_values,target)

        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            self.steps_done = 0

    def soft_update(self,TAU=0.05):
        target_net_state_dict = self.model.state_dict()
        policy_net_state_dict = self.target_model.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.target_model.load_state_dict(target_net_state_dict)