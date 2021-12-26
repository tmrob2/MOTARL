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
from typing import Any, List, Sequence, Tuple
from abc import ABC
from a2c_team_tf.utils.data_capture import AsyncWriter

seed = 42
cart_pos = 0.1  # the position for the task

class MoveToPos(DFAStates, ABC):
    def __init__(self):
        self.init = "I"
        self.position = "P"
        self.fail = "F"

def go_left_to_pos(data, _):
    if data['state'][0] < - 0.1 / 2 and not data['done']:
        return "P"
    elif data['done']:
        return "F"
    else:
        return "I"

def get_reached(data, _):
    if data['state'][0] > 0.3 and not data['done']:
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
    dfa.add_state(dfa.states.init, get_reached)
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
for i in range(2):
    envs.append(make_env('CartPole-v0', 0, seed, False))

# Set seed for experiment reproducibility

#tf.random.set_seed(seed)
#np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

print(envs[0].observation_space)

step_rew0 = 15  # step reward threshold

one_off_reward = 10.0  # one-off reward
task_prob0 = 0.8  # the probability threhold of archieving the above task

right = make_move_to_pos_dfa()
left = make_move_left_to_pos()

num_agents = 2
num_tasks = 2
dfas = [CrossProductDFA(num_tasks=num_tasks, dfas=[right, left], agent=agent) for agent in range(num_agents)]

num_actions = envs[0].action_space.n  # 2
num_hidden_units = 128

models = [ActorCritic(num_actions, num_hidden_units, num_tasks, name="AC{}".format(i)) for i in range(num_agents)]
## Some auxiliary functions for defining the "compute_loss" function.
# mu = 1.0 / num_agents  # fixed even probability of allocating each task to each agent
lam = 1.0
chi = 1.0
c = 195
e = task_prob0 * one_off_reward  # task reward threshold

min_episodes_criterion = 100
max_episodes = 200  # 10000
max_steps_per_episode = 200  # 1000

# Cartpole-v0 is considered solved if average reward is >= 195 over 100
# consecutive trials
running_reward = 0

## No discount
gamma = 1.00
alpha1 = 0.001
alpha2 = 0.001

agent = Agent(envs=envs, dfas=dfas, e=e, c=c, chi=chi, lam=lam, gamma=gamma,
              one_off_reward=one_off_reward, num_tasks=num_tasks, num_agents=num_agents)

initial_state = agent.tf_reset(0)  # initial state of one of the environments (agent 0)
print("initial state shape ", initial_state.shape)  # the state shape will be the cartpole state
                                                    # shape + the DFA progress state
initial_state = tf.expand_dims(initial_state, 0)  # the model expects a batch size (batch size=1 is a batch size)
action_logits, value = models[0](initial_state)
# There are two action in the cartpole environment action logits shape should be (1, 2)
# The critic values will be the number of tasks given to the model + 1
print(f"action logits {action_logits.shape}, values: {value.shape}")
# Create the models probability distribution from the action logits and sample an action
action = tf.random.categorical(action_logits, 1)[0, 0]
action_probs = tf.nn.softmax(action_logits)
selected_action = action_probs[0, action]
state, reward, done = agent.tf_env_step(action, 0)
# running an episode
agent_idx = tf.constant(0, dtype=tf.int32)
initial_state = agent.tf_reset(0)
print("init state ", initial_state.shape)
#action_probs, values, rewards, mask, padd1d, padd2d = \
action_probs, values, rewards, mask = agent.run_episode(initial_state, agent_idx, max_steps_per_episode, models[0])
# print(f"action probs shape: {action_probs.shape}, values shape: {values.shape}, rewards shape: {rewards.shape}, mask shape: {mask.shape}")
# # _, r_ = tf.dynamic_partition(rewards, mask, 2)
# # print("r shape: ", r_.shape)
# kappa is the allocation parameter tensor
kappa = tf.Variable(np.full(num_agents * num_tasks, 1.0 / num_agents), dtype=tf.float32)
# mu is the allocation probability based on kappa
mu = tf.nn.softmax(tf.reshape(kappa, shape=[num_agents, num_tasks]), axis=0)
# episodes reward is a deque FILO deque structure which
episodes_reward = collections.deque(maxlen=min_episodes_criterion)
# data writer, to store the data
data_writer = AsyncWriter('data-cartpole', num_agents, num_tasks)
with tqdm.trange(100) as t:
    for i in t:
        initial_states = agent.get_initial_states()
        rewards_l, ini_values = agent.train_step(initial_states, max_steps_per_episode, mu, *models)
        #if i % 5 == 0:
        #    agent.render_episode(max_steps_per_episode, *models)
        #print("episode rewards ", tf.reshape(episode_rewards, [-1]).numpy())
        #with tf.GradientTape() as tape:
        #    mu = tf.nn.softmax(tf.reshape(kappa, shape=[num_agents, num_tasks]), axis=0)
        #    # print("mu: ", mu)
        #    alloc_loss = agent.update_alloc_loss(ini_values, mu)  # alloc loss
        #kappa_grads = tape.gradient(alloc_loss, kappa)
        #processed_grads = [-0.001 * g for g in kappa_grads]
        #kappa.assign_add(processed_grads)
        summed_rewards = tf.reduce_sum(rewards_l, 1)
        print("summed rewards ", tf.reshape(summed_rewards, [-1]).numpy())
        episode_reward = np.around(tf.reshape(summed_rewards, [-1]).numpy(), decimals=2)
        episodes_reward.append(episode_reward)
        running_reward = np.around(np.mean(episodes_reward, 0), decimals=2)
        data_writer.write(episode_reward)
        t.set_description(f"Epsiode: {i}")
        t.set_postfix(running_reward=running_reward)