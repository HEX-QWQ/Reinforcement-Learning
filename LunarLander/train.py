import torch
from tqdm import tqdm
import gymnasium as gym
from algorithm.dqn import ActionValueNetwork
from trainer.dqn_trainer import DQNTrainer, ReplayBuffer

# Training hyperparameters
learning_rate = 1e-3        # How fast to learn (higher = faster but less stable)
n_episodes = 100_000        # Number of hands to practice
start_epsilon = 1.0         # Start with 100% random actions
epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
final_epsilon = 0.01
final_epsilon = 0.1         # Always keep some exploration

# Create environment and agent
env = gym.make("LunarLander-v3")
display_env = gym.make("LunarLander-v3",render_mode="human")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = ActionValueNetwork(
    obs_dim=env.observation_space.shape[0],
    act_dim=env.action_space.n,
    hidden_dim=128
)

trainer = DQNTrainer(
    model=agent,
    gamma=0.99,
    target_update_freq=100,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    lr=learning_rate,
    optimizer=torch.optim.Adam,
    epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    grad_clip_norm=10.0
)

memory = ReplayBuffer(capacity=100000)
for episode in tqdm(range(n_episodes)):
    # Start a new hand
    obs, info = env.reset()
    if episode % 100 == 0:
        obs,info = display_env.reset()
    done = False

    done = False

    
    while not done:
        
        action = trainer.get_action(obs)  

        # Render the environment
        if episode % 100 == 0:
            display_env.step(action)
            display_env.render()

        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        obs = next_obs

        memory.push(obs, action, reward, next_obs, done)

        if len(memory) >= 1000:
            trainer.experience_replay(memory)
            # memory.clear()

        trainer.epsilon_decay_func()
