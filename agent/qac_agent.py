import torch
from torch.distributions import Categorical
from algorithm.qac import PolicyNetwork, ValueNetwork

class QACAgent:
    def __init__(self, 
                action_space,
                obs_dim,
                act_dim,
                hidden_dim=128,
                device="cuda" if torch.cuda.is_available() else "cpu"):
        
        self.action_space = action_space
        self.device = device
        self.policy_net = PolicyNetwork(obs_dim, act_dim, hidden_dim).to(self.device)
        self.value_net = ValueNetwork(obs_dim, hidden_dim).to(self.device)

    def select_action(self, obs, deterministic=False):
        ## sample action from policy network
        if isinstance(obs, torch.Tensor):
            if obs.device != torch.device(self.device):
                obs = obs.to(self.device)
        else:
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        with torch.no_grad():
            logits = self.policy_net(obs)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                dist = Categorical(logits=logits)
                action = dist.sample()
        
        return action.squeeze().item()
