# Spring 2026, 535510 Reinforcement Learning
# HW1: REINFORCE with GAE

import os
import argparse
from itertools import count
from collections import namedtuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import wandb

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

wandb_run = None


class GAE:
    def __init__(self, gamma, lambda_):
        self.gamma = gamma
        self.lambda_ = lambda_

    def __call__(self, rewards, values, dones):
        """
        Compute GAE advantages from trajectory rewards/value predictions.
        values should have length len(rewards) + 1 for bootstrap at t+1.
        """
        advantages = torch.zeros(len(rewards), dtype=torch.double)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            gae = delta + self.gamma * self.lambda_ * mask * gae
            advantages[t] = gae

        return advantages


class Policy(nn.Module):
    """
    REINFORCE + value baseline + GAE advantage estimation.
    """

    def __init__(self, gae_lambda=0.95):
        super().__init__()

        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]

        ########## YOUR CODE HERE (5~10 lines) ##########
        self.hidden_size = 128
        self.gamma = 0.99
        self.gae_lambda = gae_lambda
        self.entropy_coef = 0.001
        self.value_coef = 0.5

        # Shared feature extractor + actor/value heads.
        self.shared_layer = nn.Linear(self.observation_dim, self.hidden_size)
        self.action_head = nn.Linear(self.hidden_size, self.action_dim)
        self.value_head = nn.Linear(self.hidden_size, 1)

        nn.init.xavier_uniform_(self.shared_layer.weight)
        nn.init.zeros_(self.shared_layer.bias)
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)
        nn.init.xavier_uniform_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

        self.double()

        ########## END OF YOUR CODE ##########

        self.gae = GAE(gamma=self.gamma, lambda_=self.gae_lambda)

        self.saved_actions = []
        self.rewards = []
        self.dones = []
        self.entropies = []

    def forward(self, state):
        ########## YOUR CODE HERE (3~5 lines) ##########
        # Build hidden representation, then output policy and state-value.
        x = F.relu(self.shared_layer(state))
        action_prob = F.softmax(self.action_head(x), dim=-1)
        state_value = self.value_head(x)

        ########## END OF YOUR CODE ##########
        return action_prob, state_value

    def select_action(self, state):
        ########## YOUR CODE HERE (3~5 lines) ##########
        # Convert state and sample an action from stochastic policy.
        state = torch.from_numpy(state).double()
        probs, state_value = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()

        ########## END OF YOUR CODE ##########

        self.saved_actions.append(SavedAction(dist.log_prob(action), state_value))
        self.entropies.append(dist.entropy())
        return action.item()

    def calculate_loss(self):
        ########## YOUR CODE HERE (8-15 lines) ##########
        policy_losses = []
        value_losses = []

        # Current value predictions for all visited states in this episode.
        values = torch.stack([sa.value.squeeze(-1) for sa in self.saved_actions])
        values_detached = values.detach()

        # Add final bootstrap value. For terminal episode end, bootstrap is 0.
        values_for_gae = torch.cat([values_detached, torch.zeros(1, dtype=torch.double)])

        # Compute raw GAE first; use it for value targets before normalization.
        raw_advantages = self.gae(self.rewards, values_for_gae, self.dones)
        returns = raw_advantages + values_detached
        advantages = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std() + 1e-8)

        entropy_bonus = torch.stack(self.entropies).sum()

        for (log_prob, value), adv, ret in zip(self.saved_actions, advantages, returns):
            policy_losses.append(-log_prob * adv.detach())
            value_losses.append(F.smooth_l1_loss(value.squeeze(-1), ret.detach()))

        policy_loss = torch.stack(policy_losses).sum()
        value_loss = torch.stack(value_losses).sum()
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

        ########## END OF YOUR CODE ##########
        return loss

    def clear_memory(self):
        del self.rewards[:]
        del self.saved_actions[:]
        del self.dones[:]
        del self.entropies[:]


def train(lr=0.001, gae_lambda=0.95, max_episodes=5000):
    model = Policy(gae_lambda=gae_lambda)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    global wandb_run
    ########## YOUR CODE HERE (4-5 lines) ##########
    # Track each lambda run in WandB for easy comparison.
    wandb_run = wandb.init(
        project='rl-hw1-reinforce-gae',
        name=f'lunarlander_gae_lambda_{gae_lambda}',
        config={
            'learning_rate': lr,
            'env_name': env.spec.id,
            'hidden_size': model.hidden_size,
            'gamma': model.gamma,
            'gae_lambda': model.gae_lambda,
            'entropy_coef': model.entropy_coef,
            'value_coef': model.value_coef,
        },
    )

    ########## END OF YOUR CODE ##########

    ewma_reward = 0

    for i_episode in count(1):
        state, _ = env.reset()
        ep_reward = 0
        t = 0

        ########## YOUR CODE HERE (10-15 lines) ##########
        # Roll out one complete episode and collect rewards + done flags.
        for t in range(1, 10000):
            action = model.select_action(state)
            state, reward, terminations, truncations, _ = env.step(action)
            done = np.logical_or(terminations, truncations)

            model.rewards.append(reward)
            model.dones.append(done)
            ep_reward += reward

            if done:
                break

        # Single update at episode end using GAE advantages.
        optimizer.zero_grad()
        loss = model.calculate_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        model.clear_memory()

        ########## END OF YOUR CODE ##########

        ewma_reward = 0.05 * ep_reward + 0.95 * ewma_reward
        print(f'Episode {i_episode}\tlength: {t}\treward: {ep_reward}\t ewma reward: {ewma_reward}')

        ########## YOUR CODE HERE (4-5 lines) ##########
        wandb.log(
            {
                'Train/EpisodeReward': ep_reward,
                'Train/EpisodeLength': t,
                'Train/EWMAReward': ewma_reward,
                'Train/Loss': loss.item(),
                'Train/LearningRate': optimizer.param_groups[0]['lr'],
                'Train/GAELambda': gae_lambda,
            },
            step=i_episode,
        )

        ########## END OF YOUR CODE ##########

        if ewma_reward > env.spec.reward_threshold:
            if not os.path.isdir('./preTrained'):
                os.mkdir('./preTrained')
            torch.save(model.state_dict(), f'./preTrained/LunarLander_gae_lam_{gae_lambda}_lr_{lr}.pth')
            print(
                f'Solved! Running reward is now {ewma_reward} and '
                f'the last episode runs to {t} time steps!'
            )
            if wandb_run is not None:
                wandb.finish()
            break

        if i_episode >= max_episodes:
            if wandb_run is not None:
                wandb.finish()
            break


def test(name, env_name, gae_lambda=0.95, n_episodes=10):
    model = Policy(gae_lambda=gae_lambda)
    model.load_state_dict(torch.load(f'./preTrained/{name}'))
    env = gym.make(env_name, render_mode='human')
    max_episode_len = 10000

    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        running_reward = 0
        for _ in range(max_episode_len + 1):
            action = model.select_action(state)
            state, reward, terminations, truncations, _ = env.step(action)
            done = np.logical_or(terminations, truncations)
            running_reward += reward
            if done:
                break
        print(f'Episode {i_episode}\tReward: {running_reward}')
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='REINFORCE + GAE on LunarLander-v3')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda value (e.g. 0.90, 0.95, 0.99)')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--max-episodes', type=int, default=5000, help='Maximum training episodes')
    parser.add_argument('--test-episodes', type=int, default=10, help='Number of testing episodes')
    parser.add_argument('--seed', type=int, default=10, help='Random seed')
    parser.add_argument('--env-name', type=str, default='LunarLander-v3', help='Gymnasium environment name')
    args = parser.parse_args()

    random_seed = args.seed
    lr = args.lr
    env_name = args.env_name
    gae_lambda = args.gae_lambda

    env = gym.make(env_name)
    obs, _ = env.reset(seed=random_seed)
    torch.manual_seed(random_seed)

    train(lr=lr, gae_lambda=gae_lambda, max_episodes=args.max_episodes)
    test(
        name=f'LunarLander_gae_lam_{gae_lambda}_lr_{lr}.pth',
        env_name=env_name,
        gae_lambda=gae_lambda,
        n_episodes=args.test_episodes,
    )

    env.close()
