import torch
import copy
import numpy as np
from torch import nn
from trainer.base_trainer import BaseTrainer
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, act, rew, next_obs, done):
        self.buffer.append((obs, act, rew, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, act, rew, next_obs, done = zip(*batch)
        return obs, act, rew, next_obs, done

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

class DQNTrainer(BaseTrainer):
    def __init__(self,
                 model,
                 gamma=0.99,
                 target_update_freq=100,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 lr=1e-3,
                 optimizer=torch.optim.Adam,
                 epsilon=0.1,  
                 epsilon_decay=0.995,
                 final_epsilon=0.01,    
                 grad_clip_norm=None):
        super().__init__(model,gamma,target_update_freq,device,lr,optimizer,epsilon,grad_clip_norm)
        self.target_model = copy.deepcopy(model)
        self.target_model.eval()
        self.step = 0
        self.act_dim = model.act_dim
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

    def train_step(self,obs,act,rew,next_obs,done):
        self.step += 1
        self.optimizer.zero_grad()
        q_values = self.model(obs)
        next_q_values = self.target_model(next_obs)
        q_value = q_values.gather(1,act.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = rew + self.gamma * next_q_value * (1 - done)
        loss = nn.functional.mse_loss(q_value,expected_q_value.detach())
        loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(),self.grad_clip_norm)
        self.optimizer.step()
        if self.step % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        return loss.item()
    
    def get_action(self,obs):
        with torch.no_grad():
            if np.random.rand() < self.epsilon:
                act = np.random.randint(0,self.act_dim)
            else:
                obs = torch.from_numpy(obs).float().to(self.device)
                act = self.model(obs).argmax(-1).item()

        return act
    
    def experience_replay(self,memory:ReplayBuffer,batch_size=32):
        obs,act,rew,next_obs,done = memory.sample(batch_size)
        obs = torch.from_numpy(np.array(obs)).float().to(self.device)
        act = torch.from_numpy(np.array(act)).long().to(self.device)
        rew = torch.from_numpy(np.array(rew)).float().to(self.device)
        next_obs = torch.from_numpy(np.array(next_obs)).float().to(self.device)
        done = torch.from_numpy(np.array(done)).float().to(self.device)
        loss = self.train_step(obs,act,rew,next_obs,done)
        return loss
    
    def epsilon_decay_func(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon,self.final_epsilon)
