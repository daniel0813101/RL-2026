#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 4(b): DDPG for HalfCheetah-v5 (MuJoCo).

"""

import os
import random
from collections import namedtuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import wandb
from tqdm import tqdm


# ─────────── helpers ────────────────────────────────────────────────────────────
def soft_update(target, source, tau):
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tp.data * (1.0 - tau) + p.data * tau)


def hard_update(target, source):
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(p.data)


Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward')
)


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# ─────────── networks ───────────────────────────────────────────────────────────
class Actor(nn.Module):
    """Deterministic policy μ(s) → action ∈ [-max_action, +max_action]^d."""

    def __init__(self, hidden_size, num_inputs, action_space):
        super().__init__()
        num_outputs = action_space.shape[0]
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_outputs)
        self.max_action = float(action_space.high[0])

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return torch.tanh(self.linear3(x)) * self.max_action


class Critic(nn.Module):
    """Q(s,a) — state and action concatenated at the input layer."""

    def __init__(self, hidden_size, num_inputs, action_space):
        super().__init__()
        num_outputs = action_space.shape[0]
        self.linear1 = nn.Linear(num_inputs + num_outputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)


# ─────────── DDPG agent ─────────────────────────────────────────────────────────
class DDPG:
    def __init__(self, num_inputs, action_space,
                 gamma=0.99, tau=0.005, hidden_size=256,
                 lr_a=1e-3, lr_c=1e-3):
        self.action_space = action_space
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(hidden_size, num_inputs, action_space)
        self.actor_target = Actor(hidden_size, num_inputs, action_space)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_a)

        self.critic = Critic(hidden_size, num_inputs, action_space)
        self.critic_target = Critic(hidden_size, num_inputs, action_space)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr_c)

        # Targets start as exact copies; they then track via Polyak averaging.
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def select_action(self, state, noise_std=0.0):
        """Greedy μ(s) plus optional Gaussian exploration noise."""
        self.actor.eval()
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(np.asarray(state, dtype=np.float32))
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            mu = self.actor(state).squeeze(0)
        mu_np = mu.numpy()
        if noise_std > 0.0:
            # Noise scaled by the action range so std=0.1 means "10% of max".
            mu_np = mu_np + np.random.randn(*mu_np.shape) * noise_std * self.actor.max_action
        mu_np = np.clip(mu_np, self.action_space.low, self.action_space.high)
        self.actor.train()
        return torch.FloatTensor(mu_np)

    def update_parameters(self, batch):
        state_batch      = torch.cat(batch.state)
        action_batch     = torch.cat(batch.action)
        reward_batch     = torch.cat(batch.reward).unsqueeze(1)
        mask_batch       = torch.cat(batch.mask).unsqueeze(1)
        next_state_batch = torch.cat(batch.next_state)

        # --- Critic update: TD target r + γ · mask · Q_target(s', μ_target(s')) ---
        with torch.no_grad():
            next_action = self.actor_target(next_state_batch)
            next_q = self.critic_target(next_state_batch, next_action)
            target_q = reward_batch + self.gamma * mask_batch * next_q

        predicted_q = self.critic(state_batch, action_batch)
        value_loss = F.mse_loss(predicted_q, target_q)
        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()

        # --- Actor update: maximize Q(s, μ(s)) by minimizing its negative ---
        policy_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        # --- Polyak averaging of target networks ---
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def save_model(self, env_name, suffix=""):
        os.makedirs('preTrained/', exist_ok=True)
        actor_path  = f"preTrained/ddpg_actor_{env_name}_{suffix}"
        critic_path = f"preTrained/ddpg_critic_{env_name}_{suffix}"
        print(f'Saving models to {actor_path} and {critic_path}')
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        return actor_path, critic_path

    def load_model(self, actor_path, critic_path):
        print(f'Loading models from {actor_path} and {critic_path}')
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))


# ─────────── evaluation ─────────────────────────────────────────────────────────
def evaluate_agent(agent, env_name, n_episodes=20, seed=0):
    """Greedy rollout (no exploration noise) over n_episodes; returns (mean, std)."""
    test_env = gym.make(env_name)
    rewards = []
    for ep in range(n_episodes):
        state, _ = test_env.reset(seed=seed + ep)
        total = 0.0
        while True:
            action = agent.select_action(state, noise_std=0.0)
            next_state, reward, term, trunc, _ = test_env.step(action.numpy())
            total += reward
            state = next_state
            if term or trunc:
                rewards.append(total)
                break
    test_env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


# ─────────── training loop ──────────────────────────────────────────────────────
def train():
    # ── Hyperparameters ───────────────────────────────────────────────────────
    total_steps      = 500_000     # env-step budget per problem statement
    gamma            = 0.99
    tau              = 0.005       # Polyak averaging coefficient
    hidden_size      = 256
    batch_size       = 256
    replay_size      = 1_000_000
    lr_a             = 1e-3
    lr_c             = 1e-3
    warmup_steps     = 10_000      # uniform-random actions to seed the buffer
    expl_noise_std   = 0.1         # Gaussian std as fraction of |max_action|
    updates_per_step = 1
    eval_freq        = 25_000      # evaluate every N env steps
    eval_episodes    = 20          # spec: "average over 20 evaluation episodes"

    wandb.init(
        project="ddpg",
        name=f"{env_name}_seed{random_seed}",
        config={
            "env":              env_name,
            "seed":             random_seed,
            "total_steps":      total_steps,
            "gamma":            gamma,
            "tau":              tau,
            "hidden_size":      hidden_size,
            "batch_size":       batch_size,
            "replay_size":      replay_size,
            "lr_actor":         lr_a,
            "lr_critic":        lr_c,
            "warmup_steps":     warmup_steps,
            "expl_noise_std":   expl_noise_std,
            "updates_per_step": updates_per_step,
            "eval_freq":        eval_freq,
            "eval_episodes":    eval_episodes,
        },
    )

    agent = DDPG(env.observation_space.shape[0], env.action_space,
                 gamma=gamma, tau=tau, hidden_size=hidden_size,
                 lr_a=lr_a, lr_c=lr_c)
    memory = ReplayMemory(replay_size)

    wandb.watch(agent.actor,  log="all", log_freq=1000, idx=0)
    wandb.watch(agent.critic, log="all", log_freq=1000, idx=1)

    best_eval_reward = -float('inf')
    best_actor_path = best_critic_path = None
    ewma_reward = 0.0

    state, _ = env.reset(seed=random_seed)
    state = torch.FloatTensor(np.array(state)).unsqueeze(0)

    episode_reward, episode_steps, ep_idx = 0.0, 0, 0
    ep_v_loss, ep_p_loss, ep_updates = 0.0, 0.0, 0
    last_eval_step = 0

    pbar = tqdm(total=total_steps, desc="train")
    for step in range(1, total_steps + 1):
        # 1. Action: uniform-random during warmup, policy + Gaussian noise after.
        if step <= warmup_steps:
            action = torch.FloatTensor(env.action_space.sample())
        else:
            action = agent.select_action(state, noise_std=expl_noise_std)

        # 2. Step environment.
        next_state, reward, terminated, truncated, _ = env.step(action.numpy())
        done = terminated or truncated
        episode_reward += reward
        episode_steps  += 1

        next_state_tensor = torch.FloatTensor(np.array(next_state)).unsqueeze(0)
        # Mask zeros only on TRUE terminal — truncation (time limit) keeps bootstrap.
        # HalfCheetah never terminates naturally, so mask is essentially always 1.
        mask = torch.FloatTensor([0.0 if terminated else 1.0])
        reward_t = torch.FloatTensor([reward])

        # 3. Store transition; action stored as [1, action_dim] so torch.cat works.
        memory.push(state, action.unsqueeze(0), mask, next_state_tensor, reward_t)
        state = next_state_tensor

        # 4. Gradient updates after warmup, once buffer has enough samples.
        if len(memory) >= batch_size and step > warmup_steps:
            for _ in range(updates_per_step):
                batch = Transition(*zip(*memory.sample(batch_size)))
                v_l, p_l = agent.update_parameters(batch)
                ep_v_loss   += v_l
                ep_p_loss   += p_l
                ep_updates  += 1

        # 5. End-of-episode book-keeping and reset.
        if done:
            ep_idx += 1
            ewma_reward = 0.05 * episode_reward + 0.95 * ewma_reward
            log = {
                "episode":              ep_idx,
                "train/total_steps":    step,
                "train/episode_return": episode_reward,
                "train/episode_steps":  episode_steps,
                "train/ewma_reward":    ewma_reward,
                "replay_buffer/size":   len(memory),
            }
            if ep_updates > 0:
                log["train/value_loss"]  = ep_v_loss / ep_updates
                log["train/policy_loss"] = ep_p_loss / ep_updates
            wandb.log(log, step=step)

            state, _ = env.reset()
            state = torch.FloatTensor(np.array(state)).unsqueeze(0)
            episode_reward, episode_steps = 0.0, 0
            ep_v_loss, ep_p_loss, ep_updates = 0.0, 0.0, 0

        # 6. Periodic evaluation (no exploration noise) over 20 episodes.
        if step - last_eval_step >= eval_freq:
            last_eval_step = step
            mean_r, std_r = evaluate_agent(
                agent, env_name,
                n_episodes=eval_episodes,
                seed=random_seed + step,
            )
            wandb.log({
                "eval/mean_reward":  mean_r,
                "eval/std_reward":   std_r,
                "train/total_steps": step,
            }, step=step)
            pbar.set_postfix(eval=f"{mean_r:.0f}±{std_r:.0f}")
            print(f"\n[step {step:>7d}] eval mean={mean_r:.2f} std={std_r:.2f}")

            # Save only if strictly improving; evict prior best to avoid clutter.
            if mean_r > best_eval_reward:
                if best_actor_path  and os.path.exists(best_actor_path):
                    os.remove(best_actor_path)
                if best_critic_path and os.path.exists(best_critic_path):
                    os.remove(best_critic_path)
                best_eval_reward = mean_r
                tag = f"step{step}_score{mean_r:.0f}.pth"
                best_actor_path, best_critic_path = agent.save_model(env_name, tag)
                print(f"  → New best eval reward: {mean_r:.2f}")

                artifact = wandb.Artifact(
                    name=f"ddpg-{env_name}-best", type="model",
                    description=f"Best @ step {step}, eval={mean_r:.1f}",
                )
                artifact.add_file(best_actor_path)
                artifact.add_file(best_critic_path)
                wandb.log_artifact(artifact)

        pbar.update(1)
    pbar.close()

    print(f"\nTraining done. Best eval reward during training: {best_eval_reward:.2f}")
    if best_actor_path:
        print("Running final test on best checkpoint...")
        final = test(best_actor_path, best_critic_path, hidden_size=hidden_size)
        wandb.log({"eval/final_reward": final})
    wandb.finish()


def test(actor_path, critic_path, hidden_size=256, n_episodes=20):
    """Final evaluation entry point — mirrors test() in ddpg.py for consistency."""
    test_env = gym.make(env_name)
    model = DDPG(test_env.observation_space.shape[0], test_env.action_space,
                 hidden_size=hidden_size)
    model.load_model(actor_path, critic_path)

    rewards = []
    for ep in range(1, n_episodes + 1):
        state, _ = test_env.reset()
        total, t = 0.0, 0
        while True:
            action = model.select_action(state, noise_std=0.0)
            next_state, reward, term, trunc, _ = test_env.step(action.numpy())
            total += reward
            state = next_state
            t += 1
            if term or trunc:
                rewards.append(total)
                print(f"Eval Episode {ep}: length={t}, reward={total:.2f}")
                break

    mean_r = float(np.mean(rewards))
    print(f"Number of Eval Episodes: {n_episodes}\t; Evaluation Reward: {mean_r:.2f}")
    test_env.close()
    return mean_r


def set_seed(env, seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    env.action_space.seed(seed)


if __name__ == '__main__':
    random_seed = 42
    env_name = 'HalfCheetah-v5'
    env = gym.make(env_name)
    set_seed(env, seed=random_seed)
    train()
