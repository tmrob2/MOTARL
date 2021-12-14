# thoughts: we can engineer the reward further by constructing a reward function
# rewarding steps in multitask completion, i.e. if there are two tasks then the
# reward function looks like 0.5 - 0.9 * steps / (max_steps / 2). This is a one off
# reward for completing one task and two task completion will obviously be
# 1 - 0.9 * steps / max_steps, i.e. a reward of one for accomplishing both tasks -
# a penalty for the number of steps taken to get here.

"""
Consecutive task DFA environment
"""

import collections
import copy
import numpy as np
import tensorflow as tf
import gym
import tqdm
from a2c_team_tf.utils import obs_wrapper
from a2c_team_tf.nets.base import Actor, Critic
from a2c_team_tf.lib.tf2_a2c_base import Agent
from a2c_team_tf.utils.dfa import DFAStates, DFA, CrossProductDFA
from abc import ABC
from a2c_team_tf.envs.minigrid_fetch_mult import MultObjNoGoal


env = gym.make('Mult-obj-4x4-v0')
seed = 44
env.seed(seed)
env.reset()
max_steps_per_episode = 50
env = obs_wrapper.FlatObsWrapper(env, max_steps_per_episode)
np.random.seed(seed)
tf.random.set_seed(seed)
min_episode_criterion = 50
max_episodes = 50000

num_tasks = 2

reward_threshold = 0.7
running_reward = 0

episodes_reward: collections.deque = collections.deque(maxlen=min_episode_criterion)
actor = Actor(env.action_space.n)
critic = Critic(num_tasks=num_tasks)
# construct a DFA which says get to the goal square\
class PickupObj(DFAStates, ABC):
    def __init__(self):
        self.init = "I"
        self.carrying = "C"

def pickup_ball(env: MultObjNoGoal, _):
    if env.carrying:
        if env.carrying.type == "ball":
            return "C"
        else:
            return "I"
    else:
        return "I"

def pickup_key(env: MultObjNoGoal, _):
    if env.carrying:
        if env.carrying.type == "key":
            return "C"
        else:
            return "I"
    else:
        return "I"

def finished(a, b):
    return "C"

def make_pickup_ball_dfa():
    dfa = DFA(start_state="I", acc=["C"], rej=[])
    states = PickupObj()
    dfa.add_state(states.init, pickup_ball)
    dfa.add_state(states.carrying, finished)
    return dfa

def make_pickup_key_dfa():
    dfa = DFA(start_state="I", acc=["C"], rej=[])
    states = PickupObj()
    dfa.add_state(states.init, pickup_key)
    dfa.add_state(states.carrying, finished)
    return dfa


ball = make_pickup_ball_dfa()
key = make_pickup_key_dfa()
xdfa = CrossProductDFA(num_tasks=num_tasks, dfas=[copy.deepcopy(obj) for obj in [key, ball]], agent=0)
e, c, mu, chi, lam = 0.8, 0.85, 1.0, 1.0, 1.0
agent = Agent(env, actor, critic, num_tasks=num_tasks, xdfa=xdfa, one_off_reward=1.0,
              e=e, c=c, mu=mu, chi=chi, lam=lam, gamma=1.0, alr=1e-4, clr=1e-4)
#############################################################################
#  DFA TEST
#############################################################################
# construct a random policy to make sure that the DFA is constructed correctly
run_dfa_test = False
if run_dfa_test:
    max_resets = 2
    for _ in range(max_resets):
       initial_state = agent.tf_reset()
       agent.random_policy(initial_state, 1000)

#############################################################################
# TRAIN AGENT SCRIPT
#############################################################################
with tqdm.trange(max_episodes) as t:
    for i in t:
        initial_state = agent.tf_reset()
        episode_reward = agent.train(initial_state, max_steps_per_episode)
        # print(f"episode reward: {episode_reward.numpy()}")
        episodes_reward.append(episode_reward.numpy())
        running_reward = np.around(np.mean(episodes_reward, 0), decimals=2)
        # print(f"episodes reward: {episodes_reward}")

        t.set_description(f"Episode: {i}")
        t.set_postfix(episode_reward=np.around(episode_reward.numpy(), decimals=2), running_reward=running_reward)

        #if running_reward[0] > reward_threshold and i >= min_episode_criterion:
        #    break
# Save the model(s)
tf.saved_model.save(agent.actor, "/home/tmrob2/PycharmProjects/MORLTAP/saved_models/a_4x4_mult_obj_room")
tf.saved_model.save(agent.critic, "/home/tmrob2/PycharmProjects/MORLTAP/saved_models/c_4x4_mult_obj_room")