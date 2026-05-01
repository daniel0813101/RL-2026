#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pinghsieh
"""

import sys
import gymnasium as gym
import numpy as np
import os
import random
import time
from collections import namedtuple
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
import wandb
from tqdm import tqdm

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

class ReplayMemory(object):

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

class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.8, sigma=1.0):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale    

class SeedWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, seed: int):
        super().__init__(env)
        self._seed = seed
        self._used = False  # only apply fixed seed on the very first reset

    def reset(self, seed=None):
        if seed is not None:
            return self.env.reset(seed=seed)
        if not self._used:
            self._used = True
            return self.env.reset(seed=self._seed)
        return self.env.reset()  # subsequent resets: random starts

class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        ########## YOUR CODE HERE (5~10 lines) ##########
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_outputs)
        self.max_action = float(action_space.high[0])
        ########## END OF YOUR CODE ##########

    def forward(self, inputs):
        ########## YOUR CODE HERE (5~10 lines) ##########
        x = F.relu(self.linear1(inputs))
        x = F.relu(self.linear2(x))
        # tanh squashes to [-1,1], scaled to action bounds
        return torch.tanh(self.linear3(x)) * self.max_action
        ########## END OF YOUR CODE ##########

class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        ########## YOUR CODE HERE (5~10 lines) ##########
        # State and action are concatenated at the input layer
        self.linear1 = nn.Linear(num_inputs + num_outputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        ########## END OF YOUR CODE ##########

    def forward(self, inputs, actions):
        ########## YOUR CODE HERE (5~10 lines) ##########
        x = torch.cat([inputs, actions], dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)  # linear output — Q-value is unbounded
        ########## END OF YOUR CODE ##########

class DDPG(object):
    def __init__(self, num_inputs, action_space, gamma=0.99, tau=0.001, hidden_size=256, lr_a=3e-4, lr_c=1e-3):

        self.num_inputs = num_inputs
        self.action_space = action_space

        self.actor = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_perturbed = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_a)

        self.critic = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_target = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr_c)

        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)


    def select_action(self, state, action_noise=None):
        self.actor.eval()
        # Ensure state is a float tensor with shape [1, obs_dim]
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            mu = self.actor(state)          # [1, act_dim]
            mu = mu.squeeze(0)             # [act_dim]
 
        ########## YOUR CODE HERE (3~5 lines) ##########
        mu_np = mu.data.numpy()
        if action_noise is not None:
            mu_np = mu_np + action_noise.noise()
        mu_np = np.clip(mu_np, self.action_space.low, self.action_space.high)
        self.actor.train()
        return torch.FloatTensor(mu_np)
        ########## END OF YOUR CODE ##########

    def update_parameters(self, batch):
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        mask_batch = Variable(torch.cat(batch.mask))
        next_state_batch = Variable(torch.cat(batch.next_state))
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)
        reward_batch = reward_batch.unsqueeze(1)
        mask_batch = mask_batch.unsqueeze(1)

        ########## YOUR CODE HERE (10~20 lines) ##########
        # --- Critic update ---
        # TD target: r + γ * Q_target(s', μ_target(s'))  (zero out terminal states via mask)
        expected_q = reward_batch + self.gamma * mask_batch * next_state_action_values.detach()
        self.critic.train()
        predicted_q = self.critic(state_batch, action_batch)
        value_loss = F.mse_loss(predicted_q, expected_q)
        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()

        # --- Actor update ---
        # Maximize Q(s, μ(s)) by minimizing its negative
        self.actor.train()
        policy_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()
        ########## END OF YOUR CODE ##########

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()


    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('preTrained/'):
            os.makedirs('preTrained/')
 
        if actor_path is None:
            actor_path = "preTrained/ddpg_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "preTrained/ddpg_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        return actor_path, critic_path
 
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None: 
            self.critic.load_state_dict(torch.load(critic_path))

def evaluate_agent(agent, n_episodes=20):
    """Evaluate current policy in-memory (no noise) and return mean reward."""
    test_env = gym.make(env_name)
    rewards = []
    for _ in range(n_episodes):
        state, _ = test_env.reset()
        total = 0
        while True:
            action = agent.select_action(torch.FloatTensor(np.array(state)).unsqueeze(0))
            next_state, reward, term, trunc, _ = test_env.step(action.numpy())
            total += reward
            state = next_state
            if term or trunc:
                rewards.append(total)
                break
    test_env.close()
    return float(np.mean(rewards))

def train():
    num_episodes = 500        # 500k steps budget
    gamma = 0.99
    tau = 0.005               # Target tracking
    hidden_size = 256         # network hidden size
    noise_scale_start = 0.2   # exploration noise at episode 0
    noise_scale_final = 0.05  # anneal down to this by the last episode
    replay_size = 1000000     # buffer size
    batch_size = 256          # batch size
    lr_a = 3e-4               # actor lr — bumped 3x for faster policy improvement on Pendulum's short horizon
    lr_c = 1e-3               # critic lr — DDPG-paper standard; faster Q convergence stabilizes the actor target
    updates_per_step = 1      # gradient steps per env step
    warmup_steps = 1000       # 5 episodes of random play (was 10k = ~50 eps wasted out of a 300-ep budget)
    print_freq = 1
    save_freq = 50
    best_eval_reward = -float('inf')
    best_actor_path = None
    best_critic_path = None
    ewma_reward = 0
    rewards = []
    ewma_reward_history = []
    total_numsteps = 0
    updates = 0

    # ────────── W&B: initialize run ───────────────────────────────────────────────
    wandb.init(
        project="ddpg",
        name=f"{env_name}_seed{random_seed}",
        config={
            "env":                env_name,
            "seed":               random_seed,
            "num_episodes":       num_episodes,
            "gamma":              gamma,
            "tau":                tau,
            "hidden_size":        hidden_size,
            "noise_scale_start":  noise_scale_start,
            "noise_scale_final":  noise_scale_final,
            "replay_size":        replay_size,
            "batch_size":         batch_size,
            "lr_actor":           lr_a,
            "lr_critic":          lr_c,
            "warmup_steps":       warmup_steps,
            "updates_per_step":   updates_per_step,
        },
    )
    # ──────────────────────────────────────────────────────────────────────────────────

    agent = DDPG(env.observation_space.shape[0], env.action_space, gamma, tau, hidden_size, lr_a=lr_a, lr_c=lr_c)
    ounoise = OUNoise(env.action_space.shape[0])
    memory = ReplayMemory(replay_size)

    # ──────────── W&B: watch actor & critic to log gradients + weights automatically ──
    wandb.watch(agent.actor,  log="all", log_freq=100, idx=0)
    wandb.watch(agent.critic, log="all", log_freq=100, idx=1)
    # ──────────────────────────────────────────────────────────────────────────────────
    
    for i_episode in tqdm(range(num_episodes)):
        
        # Linearly anneal exploration noise: high early, low late
        frac = i_episode / max(num_episodes - 1, 1)
        noise_scale = noise_scale_start + frac * (noise_scale_final - noise_scale_start)
        ounoise.scale = noise_scale
        ounoise.reset()
        
        state, info = env.reset()
        state = torch.FloatTensor(np.array(state)).unsqueeze(0)  # convert immediately

        episode_reward = 0
        episode_value_loss  = 0.0   # W&B: accumulate losses for the episode
        episode_policy_loss = 0.0
        episode_updates     = 0

        while True:
            ########## YOUR CODE HERE (15~30 lines) ##########
            # 1. Select action (random during warmup, policy+noise after)
            if total_numsteps < warmup_steps:
                action = torch.FloatTensor(env.action_space.sample())
            else:
                action = agent.select_action(state, ounoise)

            # 2. Step environment
            next_state, reward, terminated, truncated, _ = env.step(action.numpy())
            done = terminated or truncated
            total_numsteps += 1
            episode_reward += reward

            next_state_tensor = torch.FloatTensor(np.array(next_state)).unsqueeze(0)
            mask   = torch.FloatTensor([0.0 if done else 1.0])
            reward_t = torch.FloatTensor([reward])

            # 3. Push transition — action stored as [1, action_dim] so torch.cat works in update
            memory.push(state, action.unsqueeze(0), mask, next_state_tensor, reward_t)
            state = next_state_tensor

            # 4. Update networks once the buffer has enough data
            if len(memory) >= batch_size:
                for _ in range(updates_per_step):
                    transitions = memory.sample(batch_size)
                    batch = Transition(*zip(*transitions))
                    v_loss, p_loss = agent.update_parameters(batch)
                    episode_value_loss  += v_loss
                    episode_policy_loss += p_loss
                    updates += 1
                    episode_updates += 1

            if done:
                break
            ########## END OF YOUR CODE ########## 
      

        rewards.append(episode_reward)
        t = 0
        if i_episode % print_freq == 0:
            state, _ = env.reset()
            state = torch.FloatTensor(np.array(state)).unsqueeze(0)

            episode_reward = 0
            while True:
                action = agent.select_action(state)

                next_state, reward, terminated, truncated, _  = env.step(action.numpy())
                done = terminated or truncated
                #env.render()
                
                episode_reward += reward

                next_state = torch.FloatTensor(np.array(next_state)).unsqueeze(0)
                state = next_state
                
                t += 1
                if done:
                    break

            rewards.append(episode_reward)
            # update EWMA reward and log the results
            ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward
            ewma_reward_history.append(ewma_reward)        
            print("Episode: {}, length: {}, reward: {:.2f}, ewma reward: {:.2f}".format(i_episode, t, rewards[-1], ewma_reward))

            # ────────── W&B: log training metrics every print_freq episode ──────────────
            log_dict = {
                "episode":              i_episode,
                "train/total_steps":    total_numsteps,
                "train/total_updates":  updates,
                "train/ewma_reward":    ewma_reward,
                "train/noise_scale":    noise_scale,
                "replay_buffer/size":   len(memory),
            }
            # Average losses over the updates performed this episode
            if episode_updates > 0:
                log_dict["train/value_loss"]  = episode_value_loss  / episode_updates
                log_dict["train/policy_loss"] = episode_policy_loss / episode_updates
            wandb.log(log_dict, step=i_episode)
            # ────────────────────────────────────────────────────────────────────────────
    
        if (i_episode+1) % save_freq == 0:
            mean_eval_reward = evaluate_agent(agent)

            # ──────────── W&B: log evaluation score at save_freq intervals ────────────
            wandb.log({
                "episode":              i_episode,
                "eval/mean_reward":     mean_eval_reward,
                "eval/ewma_reward":     ewma_reward,
            }, step=i_episode)
            # ─────────────────────────────────────────────────────────────────────────

            # Save only if this is the best checkpoint seen so far
            if mean_eval_reward > best_eval_reward:
                # Remove previous best from disk to avoid accumulating stale files
                if best_actor_path and os.path.exists(best_actor_path):
                    os.remove(best_actor_path)
                if best_critic_path and os.path.exists(best_critic_path):
                    os.remove(best_critic_path)

                best_eval_reward = mean_eval_reward
                filename = 'ep{}_score{:.1f}.pth'.format(i_episode, mean_eval_reward)
                best_actor_path, best_critic_path = agent.save_model(env_name, filename)
                print(f"  → New best eval reward: {mean_eval_reward:.2f} — checkpoint saved.")

                # ── W&B: upload only the new best checkpoint as an artifact ──────────
                artifact = wandb.Artifact(
                    name=f"ddpg-{env_name}-best",
                    type="model",
                    description=f"Best checkpoint at episode {i_episode}, eval={mean_eval_reward:.1f}",
                )
                artifact.add_dir("preTrained/")
                wandb.log_artifact(artifact)
                # ─────────────────────────────────────────────────────────────────────

    # ── Final evaluation on the best saved checkpoint ────────────────────────────
    if best_actor_path:
        print(f"\nTraining done. Best eval reward during training: {best_eval_reward:.2f}")
        print("Running final test on best checkpoint...")
        final_reward = test(best_actor_path, best_critic_path)
        wandb.log({"eval/final_reward": final_reward})
    else:
        print("\nTraining done. No checkpoint met the save criterion.")

    wandb.finish()   # W&B: cleanly close the run
 
def test(actor_path, critic_path, hidden_size=256, n_episodes=20):
    '''
        Test the learned model (no change needed)
    '''      
    test_env = gym.make(env_name)
    model = DDPG(test_env.observation_space.shape[0], test_env.action_space, hidden_size=hidden_size)
    
    model.load_model(actor_path, critic_path)
    
    render = False
    eval_reward_history = []
 
    for i_episode in range(1, n_episodes+1):
        state, info = test_env.reset()
        running_reward = 0
        t = 0
        while True:
            action = model.select_action(state)
            next_state, reward, terminated, truncated, _  = test_env.step(action.numpy())
            done = terminated or truncated
            running_reward += reward
            next_state = torch.FloatTensor(np.array(next_state)).unsqueeze(0)
            state = next_state
            t += 1
            if render:
                 test_env.render()
            if done:
                eval_reward_history.append(running_reward)
                print("Eval Episode: {}, length: {}, reward: {:.2f}".format(i_episode, t, running_reward))
                break
 
    mean_reward = np.mean(eval_reward_history)
    print('Number of Eval Episodes: {}\t; Evaluation Reward: {}'.format(n_episodes, mean_reward))
    test_env.close()
    return mean_reward

def set_seed(env, seed):
    # For reproducibility, fix the random seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    env = SeedWrapper(env=env,seed=random_seed)
    env.action_space.seed(seed)


if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 0
    env_name = 'Pendulum-v1'
    #env_name = 'HalfCheetah-v5'
    env = gym.make(env_name)
    set_seed(env,seed=random_seed)
    train()