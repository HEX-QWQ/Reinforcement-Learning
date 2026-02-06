import torch 
import gymnasium as gym
from torch import nn
from agent.qac_agent import QACAgent
from trainer.qac_trainer import QACTrainer

# =====================
# Hyperparameters
# =====================
learning_rate = 1e-4
num_episodes = 100_000

start_epsilon = 1.0
epsilon_decay = 0.995
final_epsilon = 0.01

gamma = 0.99
grad_clip_norm = 10.0

replay_capacity = 5000
warmup_steps = 1000
train_steps_per_episode = 1
batch_size = 64
display_every_episodes = 500
rollout_steps = 128

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
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    done = False
    while not done:
        action = agent.select_action(obs, deterministic=True)
        next_obs, reward, terminated, truncated, info = display_env.step(action)
        done = terminated or truncated
        obs = next_obs
    display_env.close()

agent = QACAgent(
    action_space=env.action_space,
    obs_dim=env.observation_space.shape[0],
    act_dim=env.action_space.n,
    hidden_dim=128,
    device=device
)

trainer = QACTrainer(
    agent=agent,
    env=env,
    num_episodes=num_episodes,
    batch_size=batch_size,
    gamma=gamma,
    lr=learning_rate,
    train_steps_per_episode=train_steps_per_episode,
    rollout_steps=rollout_steps,
    device=device
)

if __name__ == "__main__":
    
    # obs, info = env.reset()
    # obs = torch.tensor(obs, dtype=torch.float32, device=device)
    
    # for episode in range(num_episodes):
    #     trainer.train_one_episode(obs,env)
    #     if episode % display_every_episodes == 0:
    #         display_episode(trainer.agent)
    #     if episode % print_every_episodes == 0:
    #         print(f"Episode {episode}: Average Reward: {env.return_queue[-1]:.2f}")
    trainer.train()

