import torch
import torch.nn.functional as F
import numpy as np

from trainer.base_trainer import BaseTrainer


class QACTrainer(BaseTrainer):
    def __init__(
        self,
        agent,
        env,
        num_episodes,
        train_steps_per_episode,
        max_steps_per_episode=1000,
        display_every_episodes=200,
        print_every_episodes=10,
        batch_size=64,
        gamma=0.99,
        lr=1e-3,
        entropy_coef=0.01,
        advantage_norm=False,
        rollout_steps=128,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(env, agent, None, batch_size, 0, 1, device)

        self.agent = agent
        self.env = env
        self.policy_optimizer = torch.optim.Adam(self.agent.policy_net.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.agent.value_net.parameters(), lr=lr)
        self.max_steps_per_episode = max_steps_per_episode
        self.train_steps_per_episode = train_steps_per_episode
        self.batch_size = batch_size
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.advantage_norm = advantage_norm
        self.rollout_steps = rollout_steps
        self.train_steps = 0
        self.num_episodes = num_episodes
        self.display_every_episodes = display_every_episodes
        self.print_every_episodes = print_every_episodes

    def _compute_returns(self, rewards, dones, last_value):
        returns = []
        R = last_value
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                R = 0.0
            R = r + self.gamma * R
            returns.append(R)
        returns.reverse()
        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    def update(self, batch, last_obs):
        """
        batch: list of (obs, act, rew, done)
        last_obs: observation after the final step in rollout
        """
        obs = torch.tensor(np.array([s[0] for s in batch]), dtype=torch.float32, device=self.device)
        act = torch.tensor(np.array([s[1] for s in batch]), dtype=torch.int64, device=self.device)
        rew = [s[2] for s in batch]
        done = [s[3] for s in batch]

        with torch.no_grad():
            if done[-1]:
                last_value = 0.0
            else:
                last_obs_t = torch.tensor(last_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                last_value = self.agent.value_net(last_obs_t).item()

        returns = self._compute_returns(rew, done, last_value)

        values = self.agent.value_net(obs)
        advantages = returns - values.detach()
        if self.advantage_norm and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logits = self.agent.policy_net(obs)
        logp = F.log_softmax(logits, dim=-1)
        pi = logp.exp()
        logp_act = logp.gather(1, act.unsqueeze(1)).squeeze(1)
        entropy = -(pi * logp).sum(dim=1)

        policy_loss = -(logp_act * advantages).mean() - self.entropy_coef * entropy.mean()
        value_loss = F.mse_loss(values, returns)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.policy_net.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.value_net.parameters(), max_norm=1.0)
        self.value_optimizer.step()

        return value_loss.item(), policy_loss.item()

    def display_episode(self, agent):
        import gymnasium as gym

        display_env = gym.make("LunarLander-v3", render_mode="human")
        obs, info = display_env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        done = False
        while not done:
            action = agent.select_action(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = display_env.step(action)
            done = terminated or truncated
            obs = next_obs
        display_env.close()

    def train(self):
        obs, _ = self.env.reset()
        for episode in range(self.num_episodes):
            episode_reward = None
            done = False
            while not done:
                batch = []
                for _ in range(self.rollout_steps):
                    act = self.agent.select_action(obs, deterministic=False)
                    action_val = act.item() if hasattr(act, "item") else act
                    next_obs, rew, terminated, truncated, info = self.env.step(action_val)
                    done = terminated or truncated
                    batch.append((obs, action_val, rew, done))
                    obs = next_obs

                    if done:
                        break
                v_loss, p_loss = self.update(batch, obs)

            episode_reward = self.env.return_queue[-1]
            obs, info = self.env.reset()

            if episode % self.display_every_episodes == 0:
                self.display_episode(self.agent)
            if episode % self.print_every_episodes == 0:
                print(f"Episode {episode}: Average Reward: {episode_reward:.2f}")
