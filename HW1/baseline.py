# Spring 2026, 535510 Reinforcement Learning
# HW1: REINFORCE with baseline

import os
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


class Policy(nn.Module):
    """
    Policy network + value baseline for REINFORCE with baseline.
    The policy learns a stochastic action distribution, while the value head
    estimates V(s) to reduce the variance of the policy-gradient update.
    """

    def __init__(self):
        super().__init__()

        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]

        ########## YOUR CODE HERE (5~10 lines) ##########
        self.hidden_size = 128
        # Discount factor used when turning episode rewards into returns.
        self.gamma = 0.99

        # Shared feature extractor followed by policy and value heads.
        self.shared_layer = nn.Linear(self.observation_dim, self.hidden_size)
        self.action_head = nn.Linear(self.hidden_size, self.action_dim)
        self.value_head = nn.Linear(self.hidden_size, 1)

        # Xavier init keeps the initial signal scale reasonable.
        nn.init.xavier_uniform_(self.shared_layer.weight)
        nn.init.zeros_(self.shared_layer.bias)
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)
        nn.init.xavier_uniform_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

        self.double()

        ########## END OF YOUR CODE ##########

        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        ########## YOUR CODE HERE (3~5 lines) ##########
        # Transform the state into a hidden representation.
        x = F.relu(self.shared_layer(state))
        # Policy head gives a probability for each action.
        action_prob = F.softmax(self.action_head(x), dim=-1)
        # Value head estimates how good this state is.
        state_value = self.value_head(x)

        ########## END OF YOUR CODE ##########
        return action_prob, state_value

    def select_action(self, state):
        ########## YOUR CODE HERE (3~5 lines) ##########
        # Convert the environment state into a torch tensor.
        state = torch.from_numpy(state).double()
        probs, state_value = self.forward(state)
        # Sample an action from the policy distribution.
        dist = Categorical(probs)
        action = dist.sample()

        ########## END OF YOUR CODE ##########
        self.saved_actions.append(SavedAction(dist.log_prob(action), state_value))
        return action.item()

    def calculate_loss(self, gamma=0.99):
        ########## YOUR CODE HERE (8-15 lines) ##########
        # Compute reward-to-go by walking backward through the episode.
        returns = []
        running_return = 0

        for reward in reversed(self.rewards):
            running_return = reward + gamma * running_return
            returns.insert(0, running_return)

        # Normalize returns so the gradient update is less noisy.
        returns = torch.tensor(returns, dtype=torch.double)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_losses = []
        value_losses = []

        # Advantage = actual return minus the baseline prediction.
        for (log_prob, value), ret in zip(self.saved_actions, returns):
            advantage = ret - value.squeeze(-1)
            # Policy learns from advantage; value head learns to predict return.
            policy_losses.append(-log_prob * advantage.detach())
            value_losses.append(F.smooth_l1_loss(value.squeeze(-1), ret))

        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        ########## END OF YOUR CODE ##########
        return loss

    def clear_memory(self):
        del self.rewards[:]
        del self.saved_actions[:]


def train(lr=0.001, max_episodes=5000):
    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    global wandb_run
    ########## YOUR CODE HERE (4-5 lines) ##########
    # Start a WandB run so we can track training curves online.
    wandb_run = wandb.init(
        project='rl-hw1-reinforce-baseline',
        config={
            'learning_rate': lr,
            'env_name': env.spec.id,
            'hidden_size': model.hidden_size,
            'gamma': model.gamma,
        },
    )

    ########## END OF YOUR CODE ##########

    ewma_reward = 0

    for i_episode in count(1):
        state, _ = env.reset()
        ep_reward = 0
        t = 0

        ########## YOUR CODE HERE (10-15 lines) ##########
        # Roll out one complete episode and store rewards and log-probs.
        for t in range(1, 10000):
            action = model.select_action(state)
            state, reward, terminations, truncations, _ = env.step(action)
            done = np.logical_or(terminations, truncations)

            model.rewards.append(reward)
            ep_reward += reward

            if done:
                break

        # Update both the policy and the baseline once per episode.
        optimizer.zero_grad()
        loss = model.calculate_loss(gamma=model.gamma)
        loss.backward()
        optimizer.step()
        model.clear_memory()

        ########## END OF YOUR CODE ##########

        ewma_reward = 0.05 * ep_reward + 0.95 * ewma_reward
        print(f'Episode {i_episode}\tlength: {t}\treward: {ep_reward}\t ewma reward: {ewma_reward}')

        ########## YOUR CODE HERE (4-5 lines) ##########
        # Log the main training metrics for later analysis.
        wandb.log(
            {
                'Train/EpisodeReward': ep_reward,
                'Train/EpisodeLength': t,
                'Train/EWMAReward': ewma_reward,
                'Train/Loss': loss.item(),
                'Train/LearningRate': optimizer.param_groups[0]['lr'],
            },
            step=i_episode,
        )
        
        ########## END OF YOUR CODE ##########

        if ewma_reward > env.spec.reward_threshold:
            if not os.path.isdir('./preTrained'):
                os.mkdir('./preTrained')
            torch.save(model.state_dict(), f'./preTrained/LunarLander_baseline_{lr}.pth')
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


def test(name, env_name, n_episodes=10):
    model = Policy()
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
    random_seed = 10
    lr = 0.001
    env_name = 'LunarLander-v3'
    env = gym.make(env_name)
    obs, _ = env.reset(seed=random_seed)
    torch.manual_seed(random_seed)
    train(lr)
    test(f'LunarLander_baseline_{lr}.pth', env_name)
