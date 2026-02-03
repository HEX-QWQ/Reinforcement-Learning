import torch
import numpy as np
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
        return (
            np.array(obs),
            np.array(act),
            np.array(rew),
            np.array(next_obs),
            np.array(done)
        )

    def __len__(self):
        return len(self.buffer)

class Trainer:
    def __init__(
        self,
        env,
        agent,
        memory,
        batch_size=32,
        warmup_steps=1000,
        train_every=1,
        device="cpu"
    ):
        self.env = env
        self.agent = agent
        self.memory = memory
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.train_every = train_every
        self.device = device
        self.global_step = 0

    def train_one_step(self, obs):
        action = self.agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        self.memory.push(obs, action, reward, next_obs, done)
        
        loss = None
        if len(self.memory) >= self.warmup_steps and self.global_step % self.train_every == 0:
            batch = self.memory.sample(self.batch_size)
            # Convert to tensors
            batch_t = [
                torch.tensor(x, dtype=torch.float32, device=self.device) if i != 1 
                else torch.tensor(x, dtype=torch.long, device=self.device)
                for i, x in enumerate(batch)
            ]
            loss = self.agent.update(batch_t)
            
        self.global_step += 1
        return next_obs, reward, done, info, loss
