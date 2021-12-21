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
import tqdm
from a2c_team_tf.nets.base import ActorCrticLSTM
from a2c_team_tf.lib.tf2_a2c_base import MORLTAP
from a2c_team_tf.utils.dfa import DFAStates, DFA, CrossProductDFA
from abc import ABC
from a2c_team_tf.envs.minigrid_fetch_mult import MultObjNoGoal
from a2c_team_tf.utils.env_utils import make_env
from a2c_team_tf.utils.data_capture import AsyncWriter

# Parameters
env_key = 'Mult-obj-4x4-v0'
seed = 123
max_steps_per_update = 10
np.random.seed(seed)
tf.random.set_seed(seed)
min_episode_criterion = 100
max_epsiode_steps = 50
max_episodes = 10000
num_tasks = 2
reward_threshold = 0.9
running_reward = 0
num_procs = 30
recurrence = 10
recurrent = recurrence > 1

# construct DFAs
class PickupObj(DFAStates, ABC):
    def __init__(self):
        self.init = "I"
        self.carrying = "C"
        self.drop = "D"

def pickup_ball(env: MultObjNoGoal, _):
    if env.carrying:
        if env.carrying.type == "ball":
            return "C"
        else:
            return "I"
    else:
        return "I"

def drop_ball(env: MultObjNoGoal, _):
    if env.carrying:
        if env.carrying.type == "ball":
            return "C"
        else:
            return "D"
    else:
        return "D"

def pickup_key(env: MultObjNoGoal, _):
    if env.carrying:
        if env.carrying.type == "key":
            return "C"
        else:
            return "I"
    else:
        return "I"

def finished_key(a, b):
    return "C"

def finished_ball(a, b):
    return "D"

def make_pickupanddrop_ball_dfa():
    dfa = DFA(start_state="I", acc=["D"], rej=[])
    states = PickupObj()
    dfa.add_state(states.init, pickup_ball)
    dfa.add_state(states.carrying, drop_ball)
    dfa.add_state(states.drop, finished_ball)
    return dfa

def make_pickup_key_dfa():
    dfa = DFA(start_state="I", acc=["C"], rej=[])
    states = PickupObj()
    dfa.add_state(states.init, pickup_key)
    dfa.add_state(states.carrying, finished_key)
    return dfa
#############################################################################
#  Construct Environments
#############################################################################
envs = []
for i in range(num_procs):
    # seed + 1000 * i
    eseed = seed
    envs.append(make_env(env_key=env_key, max_steps_per_episode=max_epsiode_steps, seed=eseed, apply_flat_wrapper=True))
#############################################################################
#  Initialise data structures
#############################################################################
episodes_reward: collections.deque = collections.deque(maxlen=min_episode_criterion)
model = ActorCrticLSTM(num_actions=envs[0].action_space.n, num_tasks=num_tasks, recurrent=True)
ball = make_pickupanddrop_ball_dfa()
key = make_pickup_key_dfa()
xdfa = CrossProductDFA(num_tasks=num_tasks, dfas=[copy.deepcopy(obj) for obj in [key, ball]], agent=0)
e, c, mu, chi, lam = 0.8, 0.85, 1.0, 1.0, 1.0
agent = MORLTAP(envs, model, num_tasks=num_tasks, xdfa=xdfa, one_off_reward=1.0,
                e=e, c=c, mu=mu, chi=chi, lam=lam, gamma=1.0, lr=5e-5, seed=seed,
                num_procs=num_procs, num_frames_per_proc=max_steps_per_update,
                recurrence=recurrence, max_eps_steps=max_epsiode_steps, env_key=env_key)

data_writer = AsyncWriter('data-4x4-lstm')

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
episodes_reward = collections.deque(maxlen=min_episode_criterion)
with tqdm.trange(max_episodes) as t:
    # get the initial state
    state = agent.tf_reset()
    log_reward = tf.zeros([num_procs, num_tasks + 1], dtype=tf.float32)
    indices = agent.tf_starting_indexes()
    for i in t:
        state, log_reward, running_reward, loss = agent.train(state, log_reward, indices)
        t.set_description(f"Batch: {i}")
        for reward in running_reward:
            episodes_reward.append(reward.numpy())
        if episodes_reward:
            episode_reward = episodes_reward[-1]
            running_reward = np.around(np.mean(episodes_reward, 0), decimals=2)
            data_writer.write(running_reward)
            t.set_postfix(eps=episode_reward, running_r=running_reward, loss=loss.numpy())

        if i % 200 == 0:
            # render an episode
            r_init_state = agent.render_reset()
            agent.render_episode(r_init_state, max_epsiode_steps)
            # agent.renv.window.close()
        if episodes_reward:
            if running_reward[0] > reward_threshold and i >= min_episode_criterion:
                break

# Save the model(s)
tf.saved_model.save(agent.model, "/home/tmrob2/PycharmProjects/MORLTAP/saved_models/agent_lstm_4x4")

