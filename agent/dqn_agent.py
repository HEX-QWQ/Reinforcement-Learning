import numpy as np
import torch
from .base_agent import BaseAgent

class DQNAgent(BaseAgent):
    def __init__(
        self,
        algorithm,
        action_space,
        initial_epsilon=1.0,
        epsilon_decay=0.995,
        final_epsilon=0.01,
        device="cpu"
    ):
        super().__init__(action_space)
        self.algo = algorithm
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.device = device

    def select_action(self, obs, deterministic=False):
        if not deterministic and np.random.random() < self.epsilon:
            return self.action_space.sample()
        
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.algo.get_q_values(obs_t)
        return q_values.argmax(1).item()

    def update(self, batch):
        return self.algo.update(batch)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)
