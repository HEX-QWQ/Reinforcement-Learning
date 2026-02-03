import time
from collections import deque
import torch
import gymnasium as gym
import numpy as np

from algorithm.dqn import DQN
from agent.dqn_agent import DQNAgent
from trainer.base_trainer import Trainer, ReplayBuffer

# =====================
# Hyperparameters
# =====================
learning_rate = 1e-3
n_episodes = 100_000

start_epsilon = 1.0
epsilon_decay = 0.995
final_epsilon = 0.01

gamma = 0.99
target_update_freq = 100
grad_clip_norm = 10.0

replay_capacity = 100_000
warmup_steps = 1000
train_every = 1
batch_size = 64
display_every_episodes = 100

print_every_episodes = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =====================
# Environment
# =====================
env = gym.make("LunarLander-v3")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=10_000)

def display_episode(agent):
    display_env = gym.make("LunarLander-v3", render_mode="human")
    obs, info = display_env.reset()
    done = False
    while not done:
        action = agent.select_action(obs, deterministic=True)
        next_obs, reward, terminated, truncated, info = display_env.step(action)
        done = terminated or truncated
        obs = next_obs
    display_env.close()

# =====================
# Initialize Components
# =====================
algo = DQN(
    obs_dim=env.observation_space.shape[0],
    act_dim=env.action_space.n,
    hidden_dim=128,
    lr=learning_rate,
    gamma=gamma,
    target_update_freq=target_update_freq,
    grad_clip_norm=grad_clip_norm,
    device=device
)

agent = DQNAgent(
    algorithm=algo,
    action_space=env.action_space,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    device=device
)

memory = ReplayBuffer(capacity=replay_capacity)

trainer = Trainer(
    env=env,
    agent=agent,
    memory=memory,
    batch_size=batch_size,
    warmup_steps=warmup_steps,
    train_every=train_every,
    device=device
)

# =====================
# Logging
# =====================
recent_returns = deque(maxlen=100)
recent_losses = deque(maxlen=100)
start_time = time.time()

# =====================
# Training loop
# =====================
for episode in range(1, n_episodes + 1):
    obs, info = env.reset()
    done = False
    ep_reward = 0
    
    if episode % display_every_episodes == 0:
        display_episode(agent)
        
    while not done:
        obs, reward, done, info, loss = trainer.train_one_step(obs)
        ep_reward += reward
        if loss is not None:
            recent_losses.append(loss)

    if "episode" in info:
        recent_returns.append(float(info["episode"]["r"]))
    else:
        recent_returns.append(ep_reward)

    agent.decay_epsilon()

    # Print logs
    if episode % print_every_episodes == 0:
        elapsed = time.time() - start_time
        avg_return = np.mean(recent_returns) if recent_returns else 0
        avg_loss = np.mean(recent_losses) if recent_losses else 0
        print(f"Episode {episode} | Return: {avg_return:.2f} | Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon:.3f} | Time: {elapsed:.1f}s")
