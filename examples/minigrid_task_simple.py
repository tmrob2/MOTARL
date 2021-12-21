"""
This is an example of a single agent environment (minigrid) where the tasks are
presented to an agent in the form of a DFA. There is only one task, to find the
goal square. This example is meant to serve as the most basic DFA based environment
to demonstrate that the loss function in our paper does allow, and achieve learning.
"""

import collections
import copy
import numpy as np
import tensorflow as tf
import gym
import tqdm
from a2c_team_tf.utils import obs_wrapper
from a2c_team_tf.nets.base import Actor, Critic
from a2c_team_tf.lib.tf2_a2c_base import MORLTAP
from a2c_team_tf.utils.dfa import DFAStates, DFA, CrossProductDFA
from abc import ABC
from a2c_team_tf.envs.minigrid_empty_mult import EmptyMultiTask


env = gym.make('Empty-multi-5x5-v0')
seed = 44
env.seed(seed)
env.reset()
env = obs_wrapper.FlatObsWrapper(env, 10)
np.random.seed(seed)
tf.random.set_seed(seed)
min_episode_criterion = 50
max_episodes = 20000
max_steps_per_episode = 10

reward_threshold = 0.7
running_reward = 0

episodes_reward: collections.deque = collections.deque(maxlen=min_episode_criterion)
actor = Actor(env.action_space.n)
critic = Critic(num_tasks=1)
# construct a DFA which says get to the goal square\
class GetToGoal(DFAStates, ABC):
    def __init__(self):
        self.init = "I"
        self.goal = "G"

def at_goal_state(env: EmptyMultiTask, _):
    cell = env.grid.get(*env.agent_pos)
    if cell != None:
        next_state = "G" if cell.type == 'goal' else "I"
    else:
        next_state = "I"
    return next_state

def finished(a, b):
    return "G"

def make_dfa():
    dfa = DFA(start_state="I", acc=["G"], rej=[])
    states = GetToGoal()
    dfa.add_state(states.init, at_goal_state)
    dfa.add_state(states.goal, finished)
    return dfa

task = make_dfa()
xdfa = CrossProductDFA(num_tasks=1, dfas=[copy.deepcopy(task)], agent=0)
e, c, mu, chi, lam = 0.8, 0.85, 1.0, 1.0, 1.0
agent = MORLTAP(env, actor, critic, num_tasks=1, xdfa=xdfa, one_off_reward=1.0,
                e=e, c=c, mu=mu, chi=chi, lam=lam, gamma=1.0, alr=5e-5, clr=5e-5)
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

        if running_reward[0] > reward_threshold and i >= min_episode_criterion:
            break
# Save the model(s)
tf.saved_model.save(agent.actor, "/saved_models/actor_mult_task_model")
tf.saved_model.save(agent.critic, "/saved_models/critic_multi_task_model")

