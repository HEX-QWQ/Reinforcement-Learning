import torch
import torch.nn as nn
import torch.optim as optim
import copy

class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )

    def forward(self, obs):
        return self.net(obs)

class DQN:
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_dim=128,
        lr=1e-3,
        gamma=0.99,
        target_update_freq=100,
        grad_clip_norm=None,
        device="cpu"
    ):
        self.device = device
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.grad_clip_norm = grad_clip_norm
        
        self.q_net = QNetwork(obs_dim, act_dim, hidden_dim).to(device)
        self.target_net = copy.deepcopy(self.q_net).to(device)
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.train_steps = 0

    def update(self, batch):
        """
        batch: tuple of (obs, act, rew, next_obs, done) as tensors
        """
        obs, act, rew, next_obs, done = batch
        
        self.optimizer.zero_grad()
        
        # Current Q values
        q_values = self.q_net(obs)
        q_value = q_values.gather(1, act.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_obs)
            next_q_value = next_q_values.max(1)[0]
            expected_q_value = rew + self.gamma * next_q_value * (1 - done)
            
        loss = nn.functional.mse_loss(q_value, expected_q_value)
        loss.backward()
        
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip_norm)
            
        self.optimizer.step()
        self.train_steps += 1
        
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            
        return loss.item()

    def get_q_values(self, obs):
        with torch.no_grad():
            return self.q_net(obs)
