import collections
import copy
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm
from a2c_team_tf.utils.dfa import *
from a2c_team_tf.nets.base import ActorCritic
from a2c_team_tf.lib import lib_mult_env
from a2c_team_tf.utils.env_utils import make_env
from typing import Any, List, Sequence, Tuple
from abc import ABC

seed = 42

class LearnEnv(DFAStates, ABC):
    def __init__(self):
        self.init = "I"
        self.success = "S"
        self.fail = "F"

def learn_env(data, _):
    if data['reward'] > 190:
        return "S"
    else:
        return "I"

class MoveToPos(DFAStates, ABC):
    def __init__(self):
        self.init = "I"
        self.position = "P"

def get_reached(data, _):
    if data['state'][0] > cart_pos:
        return "P"
    else:
        return "I"


def finished(data, _):
    return "P"


def make_move_to_pos_dfa():
    dfa = DFA(start_state="I", acc=["P"], rej=[])
    dfa.states = MoveToPos()
    dfa.add_state(dfa.states.init, get_reached)
    dfa.add_state(dfa.states.position, finished)
    return dfa


# Create the environment
envs = []
for i in range(2):
    envs.append(make_env('CartPole-v0', 0, seed, False))

# Set seed for experiment reproducibility

tf.random.set_seed(seed)
np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

print(envs[0].observation_space)

step_rew0 = 15  # step reward threshold

cart_pos = 2.0  # the position for the task

one_off_reward = 10.0  # one-off reward
task_prob0 = 0.8  # the probability threhold of archieving the above task

task = make_move_to_pos_dfa()  # make_move_to_pos_dfa()

num_agents = 2
num_tasks = 1
dfas = [CrossProductDFA(num_tasks=num_tasks, dfas=[task], agent=agent) for agent in range(num_agents)]

num_actions = envs[0].action_space.n  # 2
num_hidden_units = 128

models = np.array([ActorCritic(num_actions, num_hidden_units, num_tasks, name="AC{}".format(i)) for i in range(num_agents)])
## Some auxiliary functions for defining the "compute_loss" function.
# mu = 1.0 / num_agents  # fixed even probability of allocating each task to each agent
lam = 1.0
chi = 1.0
c = 195
e = task_prob0 * one_off_reward  # task reward threshold

min_episodes_criterion = 100
max_episodes = 1000  # 10000
max_steps_per_episode = 50  # 1000

# Cartpole-v0 is considered solved if average reward is >= 195 over 100
# consecutive trials
running_reward = 0

## No discount
gamma = 1.00
alpha1 = 0.001
alpha2 = 0.001

modelset = tf.data.Dataset.from_tensor_slices(models)