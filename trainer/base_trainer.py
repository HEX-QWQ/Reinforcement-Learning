import torch
from torch import nn
import numpy as np

class BaseTrainer:
    def __init__(self,
                 model,
                 gamma=0.99,
                 target_update_freq=100,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 lr=1e-3,
                 optimizer=torch.optim.Adam,
                 epsilon=0.1,
                 grad_clip_norm=None):
        self.lr = lr
        self.optimizer = optimizer(model.parameters(),lr=self.lr)
        self.model = model
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.device = device
        self.grad_clip_norm = grad_clip_norm
        self.epsilon = epsilon

    
