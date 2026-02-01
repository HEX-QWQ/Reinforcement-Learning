import time
from collections import deque

import torch
import gymnasium as gym

from algorithm.dqn import ActionValueNetwork
from trainer.dqn_trainer import DQNTrainer, ReplayBuffer


# =====================
# Helper functions
# =====================
def to_float(x):
    try:
        if x is None:
            return None
        if torch.is_tensor(x):
            return float(x.detach().cpu().item())
        return float(x)
    except Exception:
        return None


def unpack_train_result(out):
    """
    Compatible with different trainer implementations.
    """
    if out is None:
        return {}
    if isinstance(out, dict):
        return {k: to_float(v) for k, v in out.items()}
    if isinstance(out, (float, int)) or torch.is_tensor(out):
        return {"loss": to_float(out)}
    if isinstance(out, (tuple, list)) and len(out) > 0:
        return {"loss": to_float(out[0])}
    return {}


# =====================
# Hyperparameters
# =====================
learning_rate = 1e-3
n_episodes = 100_000

start_epsilon = 1.0
epsilon_decay = 0.99
final_epsilon = 0.01

gamma = 0.99
target_update_freq = 100
grad_clip_norm = 10.0

replay_capacity = 100_000
warmup_steps = 10_000
train_every = 1

print_every_episodes = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# =====================
# Environment
# =====================
env = gym.make("LunarLander-v3")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=10_000)


# =====================
# Agent & Trainer
# =====================
agent = ActionValueNetwork(
    obs_dim=env.observation_space.shape[0],
    act_dim=env.action_space.n,
    hidden_dim=128
).to(device)

trainer = DQNTrainer(
    model=agent,
    gamma=gamma,
    target_update_freq=target_update_freq,
    device=device,
    lr=learning_rate,
    optimizer=torch.optim.Adam,
    epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    grad_clip_norm=grad_clip_norm
)

memory = ReplayBuffer(capacity=replay_capacity)


# =====================
# Logging buffers
# =====================
window = 100
recent_returns = deque(maxlen=window)
recent_lengths = deque(maxlen=window)
recent_losses = deque(maxlen=window)
recent_q_taken = deque(maxlen=window)
recent_q_max = deque(maxlen=window)
recent_rewards = deque(maxlen=window)

global_step = 0
start_time = time.time()


# =====================
# Training loop
# =====================
for episode in range(1, n_episodes + 1):
    obs, info = env.reset()
    done = False

    ep_reward_sum = 0.0
    ep_len = 0

    q_taken_sum = 0.0
    q_max_sum = 0.0
    q_count = 0

    while not done:
        # Q prediction (model output, NOT real reward)
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            q_values = agent(obs_t)
            q_max = q_values.max().item()

        action = trainer.get_action(obs)

        with torch.no_grad():
            q_taken = q_values[action].item()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        memory.push(obs, action, reward, next_obs, done)

        obs = next_obs
        ep_reward_sum += reward
        ep_len += 1
        global_step += 1

        q_taken_sum += q_taken
        q_max_sum += q_max
        q_count += 1
        recent_rewards.append(reward)

        # training
        if len(memory) >= warmup_steps and global_step % train_every == 0:
            out = trainer.experience_replay(memory)
            metrics = unpack_train_result(out)
            if "loss" in metrics and metrics["loss"] is not None:
                recent_losses.append(metrics["loss"])

    # real episode return (ground-truth)
    if "episode" in info:
        ep_return = float(info["episode"]["r"])
        ep_length = int(info["episode"]["l"])
    else:
        ep_return = ep_reward_sum
        ep_length = ep_len

    recent_returns.append(ep_return)
    recent_lengths.append(ep_length)

    if q_count > 0:
        recent_q_taken.append(q_taken_sum / q_count)
        recent_q_max.append(q_max_sum / q_count)

    trainer.epsilon_decay_func()

    # =====================
    # Print logs
    # =====================
    if episode % print_every_episodes == 0:
        elapsed = time.time() - start_time
        sps = global_step / elapsed if elapsed > 0 else 0.0

        avg_return = sum(recent_returns) / len(recent_returns)
        avg_len = sum(recent_lengths) / len(recent_lengths)
        avg_loss = (
            sum(recent_losses) / len(recent_losses)
            if len(recent_losses) > 0 else float("nan")
        )
        avg_q_taken = (
            sum(recent_q_taken) / len(recent_q_taken)
            if len(recent_q_taken) > 0 else float("nan")
        )
        avg_q_max = (
            sum(recent_q_max) / len(recent_q_max)
            if len(recent_q_max) > 0 else float("nan")
        )
        avg_r_step = sum(recent_rewards) / len(recent_rewards)

        print(
            f"[Ep {episode:6d}] "
            f"Ret {ep_return:7.1f} | "
            f"AvgRet100 {avg_return:7.1f} | "
            f"Len {ep_length:4d} | "
            f"Eps {trainer.epsilon:5.3f} | "
            f"Loss {avg_loss:8.4f} | "
            f"Q(a) {avg_q_taken:7.2f} | "
            f"Qmax {avg_q_max:7.2f} | "
            f"R/step {avg_r_step:6.3f} | "
            f"SPS {sps:7.1f}"
        )

env.close()
