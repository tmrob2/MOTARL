import collections
import copy
import math
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm
from a2c_team_tf.utils.dfa import *
from a2c_team_tf.nets.base import ActorCritic
from a2c_team_tf.lib.lib_mult_env import Agent
from a2c_team_tf.utils.env_utils import make_env
from a2c_team_tf.utils.data_capture import AsyncWriter
from typing import Any, List, Sequence, Tuple
from abc import ABC

gym.logger.set_level(40)
seed = 42
cart_pos = 2.0  # the position for the task

class MoveToPos(DFAStates, ABC):
    def __init__(self):
        self.init = "I"
        self.position = "P"
        self.fail = "F"

def go_left_to_pos(data, _):
    if data['state'][0] < -0.5 and not data['done']:
        return "P"
    elif data['done']:
        return "F"
    else:
        return "I"

def go_right(data, _):
    if data['state'][0] > 1.0 and not data['done']:
        return "P"
    elif data['done']:
        return "F"
    else:
        return "I"

def finished_move(data, _):
    return "P"

def failed(data, _):
    return "F"

def make_move_to_pos_dfa():
    dfa = DFA(start_state="I", acc=["P"], rej=["F"])
    dfa.states = MoveToPos()
    dfa.add_state(dfa.states.init, go_right)
    dfa.add_state(dfa.states.position, finished_move)
    dfa.add_state(dfa.states.fail, failed)
    return dfa

def make_move_left_to_pos():
    dfa = DFA(start_state="I", acc=["P"], rej=["F"])
    dfa.states = MoveToPos()
    dfa.add_state(dfa.states.init, go_left_to_pos)
    dfa.add_state(dfa.states.position, finished_move)
    dfa.add_state(dfa.states.fail, failed)
    return dfa


# Create the environment
envs = []
envs.append(make_env('CartPole-default-v0', 0, seed, False))
envs.append(make_env('CartPole-heavy-long-v0', 0, seed + 10000, False))

# Set seed for experiment reproducibility

tf.random.set_seed(seed)
np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

print(envs[0].observation_space)

one_off_reward = 100.0  # one-off reward
task_prob0 = 0.8  # the probability threhold of archieving the above task

right = make_move_to_pos_dfa()
left = make_move_left_to_pos()

num_agents = 2
num_tasks = 2
dfas = [CrossProductDFA(num_tasks=num_tasks, dfas=[right, left], agent=agent) for agent in range(num_agents)]

num_actions = envs[0].action_space.n  # 2
num_hidden_units = 128

models = [ActorCritic(num_actions, num_hidden_units, num_tasks, name="AC{}".format(i)) for i in range(num_agents)]
lam = 1.0
chi = 1.0
c = 300
e = task_prob0 * one_off_reward  # task reward threshold

min_episodes_criterion = 100
max_episodes = 30000  # 10000
max_steps_per_episode = 1000  # 1000

# Cartpole-v0 is considered solved if average reward is >= 195 over 100
# consecutive trials
running_reward = 0

## No discount
gamma = 1.00
alpha1 = 0.001
alpha2 = 0.001

agent = Agent(envs=envs, dfas=dfas, e=e, c=c, chi=chi, lam=lam, gamma=gamma,
              one_off_reward=one_off_reward, num_tasks=num_tasks, num_agents=num_agents)

data_writer = AsyncWriter('data-cartpole', num_agents, num_tasks)

############################################################################
# TRAIN AGENT SCRIPT
#############################################################################
episodes_reward = collections.deque(maxlen=min_episodes_criterion)
kappa = tf.Variable(np.full(num_agents * num_tasks, 1.0 / num_agents), dtype=tf.float32)
mu = tf.nn.softmax(tf.reshape(kappa, shape=[num_agents, num_tasks]), axis=0)

agent_threshold = 195.

with tqdm.trange(max_episodes) as t:
    for i in t:
        initial_states = agent.get_initial_states()
        rewards, ini_values = \
            agent.train_step(initial_states, max_steps_per_episode, mu, *models)
        # Calculate the episode rewards
        episode_reward = np.around(
            tf.reshape(tf.reduce_sum(rewards, 1), [-1]).numpy(), decimals=2)
        episodes_reward.append(episode_reward)
        running_reward = np.around(np.mean(episodes_reward, 0), decimals=2)
        data_writer.write(episode_reward)
        if i % 500 == 0:
            agent.render_episode(max_steps_per_episode, *models)
        t.set_description(f"Epsiode: {i}")
        t.set_postfix(running_reward=running_reward)

        if all(x > 195 for x in running_reward[::3]) and all(x > e for x in running_reward[1::3]) \
                and all(x > e for x in running_reward[2::3]) and i > min_episodes_criterion:
            break

