import collections
import copy

import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm
from a2c_team_tf.utils.dfa import *
from a2c_team_tf.nets.base import ActorCritic
from a2c_team_tf.lib import motaplib
from typing import Any, List, Sequence, Tuple
from abc import ABC


class MoveToPos(DFAStates, ABC):
    def __init__(self):
        self.init = "I"
        self.just_finish = "J"
        self.finish = "F"
        self.failed = "N"


def get_reached(env: gym.Env):
    if env is not None:
        if env.env.state[0] > cart_pos:
            return "F"
        else:
            return "I"
    else:
        return "I"


def finished(data):
    return "F"


def make_move_to_pos_dfa():
    dfa = DFA(start_state="I", acc=["F"], rej=[])
    dfa.states = MoveToPos()
    dfa.add_state(dfa.states.init, get_reached)
    dfa.add_state(dfa.states.finish, finished)
    dfa.start()
    return dfa


# Create the environment
env1 = gym.make("CartPole-v0")
env2 = gym.make("CartPole-v0")
envs = [env1, env2]

# Set seed for experiment reproducibility
seed = 42
env1.seed(seed)
env2.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

print(env1.observation_space)

step_rew0 = 15  # step reward threshold

cart_pos = 2.0  # the position for the task

one_off_reward = 10.0  # one-off reward
task_prob0 = 0.8  # the probability threhold of archieving the above task

task = make_move_to_pos_dfa()  # make_move_to_pos_dfa()

num_agents = 2
num_tasks = 1
cpdfa1 = CrossProductDFA(num_tasks=num_tasks, dfas=[task])
dfas = [copy.deepcopy(cpdfa1)] * num_agents

num_actions = env1.action_space.n  # 2
num_hidden_units = 128

models = [ActorCritic(num_actions, num_hidden_units, num_tasks, name="AC{}".format(i)) for i in range(num_agents)]
## Some auxiliary functions for defining the "compute_loss" function.
# mu = 1.0 / num_agents  # fixed even probability of allocating each task to each agent
lam = 1.0
chi = 1.0
c = step_rew0
e = task_prob0 * one_off_reward  # task reward threshold

min_episodes_criterion = 100
max_episodes = 1000  # 10000
max_steps_per_episode = 50  # 1000

# Cartpole-v0 is considered solved if average reward is >= 195 over 100
# consecutive trials
reward_threshold = 195
running_reward = 0

## No discount
gamma = 1.00
alpha1 = 0.001
alpha2 = 0.001

# Keep last episodes reward
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)
render_env, print_rewards = False, False
motap = motaplib.TfObsEnv(envs=envs, models=models, dfas=dfas, one_off_reward=one_off_reward,
                          num_tasks=num_tasks, num_agents=num_agents, render=render_env, debug=print_rewards)
## Have to use a smaller learning_rate to make the training convergent
optimizer = tf.keras.optimizers.Adam(learning_rate=alpha1)  # 0.01

kappa = tf.Variable(np.random.rand(num_tasks * num_agents), dtype=tf.float32)
# print(f"kappa:{kappa}")

with tqdm.trange(max_episodes) as t:
    for i in t:
        mu = tf.nn.softmax(tf.reshape(kappa, shape=[num_agents, num_tasks]), axis=0)
        # initial_state = tf.constant(env.reset(), dtype=tf.float32)
        # episode_reward = int(train_step(
        #    initial_state, models, optimizer, gamma, max_steps_per_episode))
        episode_reward, ini_values = motap.train_step(optimizer, gamma, max_steps_per_episode, lam, chi, mu, e, c)
        with tf.GradientTape() as tape:
            mu = tf.nn.softmax(tf.reshape(kappa, shape=[num_agents, num_tasks]), axis=0)
            allocator_loss = motap.compute_alloc_loss(ini_values, chi, mu, e)

        # compute the gradient from the allocator loss vector
        grads_kappa = tape.gradient(allocator_loss, kappa)
        # print(f"grads kappa: {grads_kappa}")
        processed_grads = [-alpha2 * g for g in grads_kappa]
        kappa.assign_add(processed_grads)
        # print(f"kappa: {kappa}")

        episode_reward = int(episode_reward)

        episodes_reward.append(episode_reward)
        running_reward = statistics.mean(episodes_reward)

        t.set_description(f'Episode {i}')
        t.set_postfix(
            episode_reward=episode_reward, running_reward=running_reward)

        # Show the learned values, and learned allocation matrix every 20 steps
        if i % 20 == 0:
            for k in range(num_agents):
                print(f'values at the initial state for model#{k}: {ini_values[k]}')
            print(f"allocation matrix mu: \n{mu}")

        if running_reward > reward_threshold and i >= min_episodes_criterion:
            break

print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')