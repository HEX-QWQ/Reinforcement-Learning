import torch
import random
from torch import nn

class ActionValueNetwork(nn.Module):
    def __init__(self,obs_dim,act_dim,hidden_dim=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,act_dim)
        )

    def forward(self,obs):
        return self.net(obs)
    
class DQN(nn.Module):
    def __init__(self,obs_dim,act_dim,hidden_dim=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,act_dim)
        )
        
    def forward(self,obs):
        return self.net(obs)