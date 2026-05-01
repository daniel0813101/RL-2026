#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 4(c): DDPG + Clipped Double Q (CDQ) for HalfCheetah-v5.

Built on top of ddpg_cheetah.py (b). Only the critic side changes — everything
else (actor, replay, exploration, eval, schedule, hyperparameters) is held
fixed so the comparison against (b) is apples-to-apples.

CDQ changes vs. (b), per the spec:
  • Two critics Qw1, Qw2 with their own targets and optimizers.
  • Target action is smoothed: a' = π_θ_target(s') + ε,
    ε ~ clip(N(0, σ²), -ε_max, ε_max), then clipped to the action bounds.
  • Per-critic TD target uses the *min* over the two target critics:
       y = r + γ · min_{i'=1,2} Q_{w'_i'}(s', a')
  • Each critic minimizes (Q_{w_i}(s,a) − y)².
  • Actor still does the deterministic policy gradient using critic-1, as is
    standard (TD3 paper Sec. 4.2). Critic-2 exists only to debias the target.

Note on faithfulness: the spec asks only for CDQ + target-policy smoothing,
NOT for TD3's delayed policy updates. I keep actor/critic updates synchronous
so the only difference from (b) is the new TD target.
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


# ─────────── networks (unchanged from b) ────────────────────────────────────────
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


# ─────────── DDPG + CDQ agent ───────────────────────────────────────────────────
class DDPG_CDQ:
    def __init__(self, num_inputs, action_space,
                 gamma=0.99, tau=0.005, hidden_size=256,
                 lr_a=1e-3, lr_c=1e-3,
                 policy_noise=0.2, noise_clip=0.5):
        """
        policy_noise : σ of the smoothing noise ε added to the target action,
                       expressed as a fraction of |max_action|.
        noise_clip   : ε_max — the noise is clipped to [-ε_max, +ε_max],
                       expressed as a fraction of |max_action|.
        """
        self.action_space = action_space
        self.gamma = gamma
        self.tau = tau

        # Smoothing noise parameters scaled to absolute action units.
        self.max_action   = float(action_space.high[0])
        self.policy_noise = policy_noise * self.max_action
        self.noise_clip   = noise_clip   * self.max_action
        # Action bounds as tensors for clamping the smoothed target action.
        self._act_low_t  = torch.FloatTensor(action_space.low)
        self._act_high_t = torch.FloatTensor(action_space.high)

        # Actor (single, as in vanilla DDPG).
        self.actor        = Actor(hidden_size, num_inputs, action_space)
        self.actor_target = Actor(hidden_size, num_inputs, action_space)
        self.actor_optim  = Adam(self.actor.parameters(), lr=lr_a)

        # Twin critics (the "double" part of clipped double Q).
        self.critic1        = Critic(hidden_size, num_inputs, action_space)
        self.critic2        = Critic(hidden_size, num_inputs, action_space)
        self.critic1_target = Critic(hidden_size, num_inputs, action_space)
        self.critic2_target = Critic(hidden_size, num_inputs, action_space)
        self.critic1_optim  = Adam(self.critic1.parameters(), lr=lr_c)
        self.critic2_optim  = Adam(self.critic2.parameters(), lr=lr_c)

        hard_update(self.actor_target,   self.actor)
        hard_update(self.critic1_target, self.critic1)
        hard_update(self.critic2_target, self.critic2)

    def select_action(self, state, noise_std=0.0):
        """Greedy μ(s) plus optional Gaussian *exploration* noise (env-side)."""
        self.actor.eval()
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(np.asarray(state, dtype=np.float32))
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            mu = self.actor(state).squeeze(0)
        mu_np = mu.numpy()
        if noise_std > 0.0:
            mu_np = mu_np + np.random.randn(*mu_np.shape) * noise_std * self.max_action
        mu_np = np.clip(mu_np, self.action_space.low, self.action_space.high)
        self.actor.train()
        return torch.FloatTensor(mu_np)

    def update_parameters(self, batch):
        state_batch      = torch.cat(batch.state)
        action_batch     = torch.cat(batch.action)
        reward_batch     = torch.cat(batch.reward).unsqueeze(1)
        mask_batch       = torch.cat(batch.mask).unsqueeze(1)
        next_state_batch = torch.cat(batch.next_state)

        # ── Build the CDQ target ──────────────────────────────────────────────
        with torch.no_grad():
            # 1. Target policy smoothing: a' = clip( μ_target(s') + clip(ε), bounds )
            #    ε ~ N(0, policy_noise²), clipped to ±noise_clip.
            noise = torch.randn_like(action_batch) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = self.actor_target(next_state_batch) + noise
            next_action = torch.max(torch.min(next_action, self._act_high_t),
                                    self._act_low_t)

            # 2. Take the MIN over the two target critics (curbs over-estimation).
            next_q1 = self.critic1_target(next_state_batch, next_action)
            next_q2 = self.critic2_target(next_state_batch, next_action)
            next_q  = torch.min(next_q1, next_q2)

            target_q = reward_batch + self.gamma * mask_batch * next_q

        # ── Critic-1 update: MSE against the shared CDQ target ────────────────
        pred_q1 = self.critic1(state_batch, action_batch)
        critic1_loss = F.mse_loss(pred_q1, target_q)
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        # ── Critic-2 update: same target, independent network ─────────────────
        pred_q2 = self.critic2(state_batch, action_batch)
        critic2_loss = F.mse_loss(pred_q2, target_q)
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # ── Actor update: deterministic PG using critic-1 (TD3 convention) ────
        policy_loss = -self.critic1(state_batch, self.actor(state_batch)).mean()
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        # ── Polyak averaging for all targets ──────────────────────────────────
        soft_update(self.actor_target,   self.actor,   self.tau)
        soft_update(self.critic1_target, self.critic1, self.tau)
        soft_update(self.critic2_target, self.critic2, self.tau)

        # Report mean critic loss for logging parity with (b).
        value_loss = 0.5 * (critic1_loss.item() + critic2_loss.item())
        return value_loss, policy_loss.item()

    def save_model(self, env_name, suffix=""):
        os.makedirs('preTrained/', exist_ok=True)
        actor_path   = f"preTrained/ddpg_cdq_actor_{env_name}_{suffix}"
        critic1_path = f"preTrained/ddpg_cdq_critic1_{env_name}_{suffix}"
        critic2_path = f"preTrained/ddpg_cdq_critic2_{env_name}_{suffix}"
        print(f'Saving models to {actor_path}, {critic1_path}, {critic2_path}')
        torch.save(self.actor.state_dict(),   actor_path)
        torch.save(self.critic1.state_dict(), critic1_path)
        torch.save(self.critic2.state_dict(), critic2_path)
        return actor_path, critic1_path, critic2_path

    def load_model(self, actor_path, critic1_path, critic2_path):
        print(f'Loading models from {actor_path}, {critic1_path}, {critic2_path}')
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic1_path is not None:
            self.critic1.load_state_dict(torch.load(critic1_path))
        if critic2_path is not None:
            self.critic2.load_state_dict(torch.load(critic2_path))


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
    # ── Hyperparameters (mirror (b) so the comparison is apples-to-apples) ───
    total_steps      = 500_000
    gamma            = 0.99
    tau              = 0.005
    hidden_size      = 256
    batch_size       = 256
    replay_size      = 1_000_000
    lr_a             = 1e-3
    lr_c             = 1e-3
    warmup_steps     = 10_000
    expl_noise_std   = 0.1            # env-side exploration noise
    updates_per_step = 1
    eval_freq        = 25_000
    eval_episodes    = 20
    # ── New CDQ-only knobs (TD3 defaults) ────────────────────────────────────
    policy_noise     = 0.2            # σ of target-policy smoothing noise
    noise_clip       = 0.5            # ε_max for the smoothing noise

    wandb.init(
        project="ddpg",
        name=f"cdq_{env_name}_seed{random_seed}",
        config={
            "algo":             "DDPG+CDQ",
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
            "policy_noise":     policy_noise,
            "noise_clip":       noise_clip,
        },
    )

    agent = DDPG_CDQ(env.observation_space.shape[0], env.action_space,
                     gamma=gamma, tau=tau, hidden_size=hidden_size,
                     lr_a=lr_a, lr_c=lr_c,
                     policy_noise=policy_noise, noise_clip=noise_clip)
    memory = ReplayMemory(replay_size)

    wandb.watch(agent.actor,   log="all", log_freq=1000, idx=0)
    wandb.watch(agent.critic1, log="all", log_freq=1000, idx=1)
    wandb.watch(agent.critic2, log="all", log_freq=1000, idx=2)

    best_eval_reward = -float('inf')
    best_actor_path = best_critic1_path = best_critic2_path = None
    ewma_reward = 0.0

    state, _ = env.reset(seed=random_seed)
    state = torch.FloatTensor(np.array(state)).unsqueeze(0)

    episode_reward, episode_steps, ep_idx = 0.0, 0, 0
    ep_v_loss, ep_p_loss, ep_updates = 0.0, 0.0, 0
    last_eval_step = 0

    pbar = tqdm(total=total_steps, desc="train_cdq")
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
        # Bootstrap mask uses `terminated` only (time-limit is not a true terminal).
        mask = torch.FloatTensor([0.0 if terminated else 1.0])
        reward_t = torch.FloatTensor([reward])

        memory.push(state, action.unsqueeze(0), mask, next_state_tensor, reward_t)
        state = next_state_tensor

        # 3. Gradient updates (after warmup).
        if len(memory) >= batch_size and step > warmup_steps:
            for _ in range(updates_per_step):
                batch = Transition(*zip(*memory.sample(batch_size)))
                v_l, p_l = agent.update_parameters(batch)
                ep_v_loss  += v_l
                ep_p_loss  += p_l
                ep_updates += 1

        # 4. End-of-episode book-keeping and reset.
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

        # 5. Periodic evaluation (no exploration noise) over 20 episodes.
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

            if mean_r > best_eval_reward:
                # Evict prior best to keep disk clean.
                for p in (best_actor_path, best_critic1_path, best_critic2_path):
                    if p and os.path.exists(p):
                        os.remove(p)
                best_eval_reward = mean_r
                tag = f"step{step}_score{mean_r:.0f}.pth"
                (best_actor_path, best_critic1_path,
                 best_critic2_path) = agent.save_model(env_name, tag)
                print(f"  → New best eval reward: {mean_r:.2f}")

                artifact = wandb.Artifact(
                    name=f"ddpg-cdq-{env_name}-best", type="model",
                    description=f"Best @ step {step}, eval={mean_r:.1f}",
                )
                artifact.add_file(best_actor_path)
                artifact.add_file(best_critic1_path)
                artifact.add_file(best_critic2_path)
                wandb.log_artifact(artifact)

        pbar.update(1)
    pbar.close()

    print(f"\nTraining done. Best eval reward during training: {best_eval_reward:.2f}")
    if best_actor_path:
        print("Running final test on best checkpoint...")
        final = test(best_actor_path, best_critic1_path, best_critic2_path,
                     hidden_size=hidden_size)
        wandb.log({"eval/final_reward": final})
    wandb.finish()


def test(actor_path, critic1_path, critic2_path,
         hidden_size=256, n_episodes=20):
    """Final evaluation on the best saved checkpoint."""
    test_env = gym.make(env_name)
    model = DDPG_CDQ(test_env.observation_space.shape[0], test_env.action_space,
                     hidden_size=hidden_size)
    model.load_model(actor_path, critic1_path, critic2_path)

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
