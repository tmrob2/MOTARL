# http://www.wang.works/portfolio/code/Reinforcement-Learning-Policy-Gradient-Methods.html
# https://github.com/lcswillems/rl-starter-files/blob/master/scripts/train.py
# https://github.com/lcswillems/torch-ac/blob/master/torch_ac/algos/a2c.py

# TODO the plan here is to build a minimal working case on the teamgrid environment
#  first using REINFORCE and then upgrading to A2C
# Rollout Buffer
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from a2c_team_tf.nets.pytorch_nets import ActorNetwork
import torch.optim as optim
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

class RolloutStorage():
    def __init__(self, rollout_size, obs_size):
        self.rollout_size = rollout_size
        self.obs_size = obs_size
        self.reset()

    def insert(self, step, done, action , log_prob, reward, obs):
        self.done[step].copy_(done)
        self.actions[step].copy_(action)
        self.log_probs[step].copy_(log_prob)
        self.rewards[step].copy_(reward)
        self.obs[step].copy_(obs)

    def reset(self):
        self.done = torch.zeros(self.rollout_size, 1)
        self.returns = torch.zeros(self.rollout_size + 1, 1, requires_grad=False)
        self.actions = torch.zeros(self.rollout_size, 1, dtype=torch.int64)
        self.log_probs = torch.zeros(self.rollout_size, 1)
        self.rewards = torch.zeros(self.rollout_size, 1)
        self.obs = torch.zeros(self.rollout_size, self.obs_size)

    def compute_returns(self, gamma):
        # compute the returns until the last finished episode
        self.last_done = (self.done == 1).nonzero().max()
        self.returns[self.last_done + 1] = 0
        # accumulate discounted returns
        for step in reversed(range(self.last_done)):
            self.returns[step] = self.returns[step + 1] * gamma * (1- self.done[step]) + \
                                 self.rewards

    def batch_sampler(self, batch_size, get_old_log_probs=False):
        sampler = BatchSampler(
            SubsetRandomSampler(range(self.last_done)),
            batch_size,
            drop_last=True
        )
        for indices in sampler:
            if get_old_log_probs:
                yield self.actions[indices], self.returns[indices], self.obs[indices], self.log_probs[indices]
            else:
                yield self.actions[indices], self.returns[indices], self.obs[indices]

class Policy():
    def __init__(
            self,
            num_inputs,
            num_actions,
            hidden_dim,
            learning_rate,
            batch_size,
            policy_epochs,
            entropy_coef=0.001):
        self.actor = ActorNetwork(num_inputs, num_actions, hidden_dim)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.policy_epochs = policy_epochs
        self.entropy_coef = entropy_coef

    def act(self, state):
        # Run the actor network on the current state to retrieve the action logits
        # Build a categorical distribution instance from the logis
        # Sample an action from the distribution
        logits = self.actor.forward(state)
        dist = Categorical(logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_actions(self, state, action):
        logits = self.actor.forward(state)
        dist = Categorical(logits)
        log_prob = dist.log_prob(action.squeeze(-1)).view(-1, 1)
        entropy = dist.entropy().view(-1, 1)
        return log_prob, entropy

    def update(self, rollouts):
        for epoch in range(self.policy_epochs):
            data = rollouts.batch_sampler(self.batch_size)

            for sample in data:
                actions_batch, returns_batch, obs_batch = sample
                log_probs_batch, entropy_batch = self.evaluate_actions(obs_batch, actions_batch)
                # Compute the mean loss for the policy update using action log probabilities and policy returns
                # Compute the mean entropy for the policy update
                policy_loss = torch.mean(torch.sum(torch.mul(log_probs_batch, returns_batch).mul(-1), -1))
                entropy_loss = torch.mean(entropy_batch).mul(-1)
                loss = policy_loss + self.entropy_coef * entropy_loss
                self.optimizer.zero_grad()
                loss.backward(retain_graph=False)
                self.optimizer.step()


