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
from a2c_team_tf.envs.team_grid_mult import TestEnv
from a2c_team_tf.utils.env_utils import make_env
from a2c_team_tf.utils.data_capture import AsyncWriter
import multiprocessing

# Parameters
env_key = 'Team-obj-5x5-v0'
seed = 123
max_steps_per_update = 10
np.random.seed(seed)
tf.random.set_seed(seed)
min_episode_criterion = 100
max_epsiode_steps = 50
max_episodes = 10000
num_tasks = 2
num_agents = 2
reward_threshold = 0.9
running_reward = 0
# the number of CPUs to run in parallel when generating environments
num_procs = min(multiprocessing.cpu_count(), 30)
recurrence = 10
recurrent = recurrence > 1

# construct DFAs
class PickupObj(DFAStates, ABC):
    def __init__(self):
        self.init = "I"
        self.carrying = "C"
        self.drop = "D"

def pickup_ball(env: TestEnv, agent):
    if env.agents[agent].carrying:
        if env.agents[agent].carrying.type == "ball":
            return "C"
        else:
            return "I"
    else:
        return "I"

def drop_ball(env: TestEnv, agent):
    if env.agents[agent].carrying:
        if env.agents[agent].carrying.type == "ball":
            return "C"
        else:
            return "D"
    else:
        return "D"

def pickup_key(env: TestEnv, agent):
    if env.agents[agent].carrying:
        if env.agents[agent].carrying.type == "key":
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
    eseed = seed
    envs.append(make_env(env_key=env_key, max_steps_per_episode=max_epsiode_steps, apply_flat_wrapper=False))
#############################################################################
#  Initialise data structures
#############################################################################
ball = make_pickupanddrop_ball_dfa()
key = make_pickup_key_dfa()
xdfas = [[
    CrossProductDFA(
        num_tasks=num_tasks,
        dfas=[copy.deepcopy(obj) for obj in [key, ball]],
        agent=agent) for agent in range(num_agents)] for _ in range(num_procs)]
e, c, chi, lam = 0.8, 0.85, 1.0, 1.0
models = [ActorCrticLSTM(envs[0].action_space.n, num_tasks, recurrent) for _ in range(num_agents)]
agent = MORLTAP(envs, models, num_tasks=num_tasks, num_agents=num_agents, xdfas=xdfas, one_off_reward=1.0,
                e=e, c=c, chi=chi, lam=lam, gamma=1.0, lr=5e-5, lr2=0.001, seed=seed,
                num_procs=num_procs, num_frames_per_proc=max_steps_per_update,
                recurrence=recurrence, max_eps_steps=max_epsiode_steps, env_key=env_key)

data_writer = AsyncWriter('data-4x4-lstm-ma', num_agents, num_tasks)

checkpoint = tf.train.Checkpoint()

#############################################################################
# TRAIN AGENT SCRIPT
#############################################################################
episodes_reward = collections.deque(maxlen=min_episode_criterion)
kappa = tf.Variable(np.full(num_agents * num_tasks, 1.0 / num_agents), dtype=tf.float32)
kopt = tf.keras.optimizers.SGD(learning_rate=0.1)
with tqdm.trange(max_episodes) as t:
    # get the initial state
    state = agent.tf_reset2()
    state = tf.squeeze(state)
    state = tf.expand_dims(tf.transpose(state, perm=[1, 0, 2]), 2)
    log_reward = tf.zeros([num_agents, num_procs, num_tasks + 1], dtype=tf.float32)
    indices = agent.tf_1d_indices()
    mu = tf.nn.softmax(tf.reshape(kappa, shape=[num_agents, num_tasks]), axis=0)
    for i in t:
        state, log_reward, running_reward, loss, ini_values = agent.train(state, log_reward, indices, mu, *models)
        if i % 10 == 0:
            with tf.GradientTape() as tape:
                mu = tf.nn.softmax(tf.reshape(kappa, shape=[num_agents, num_tasks]), axis=0)
                alloc_loss = agent.update_alloc_loss(ini_values, mu)
            kappa_grads = tape.gradient(alloc_loss, kappa)
            processed_grads = [-agent.lr2 * g for g in kappa_grads]
            kappa.assign_add(processed_grads)
        if i % 200 == 0:
            tf.print("mu\n", mu)
        t.set_description(f"Batch: {i}")
        for reward in running_reward:
            episodes_reward.append(reward.numpy().flatten())
        if episodes_reward:
            running_reward = np.around(np.mean(episodes_reward, 0), decimals=2)
            data_writer.write(running_reward)
            t.set_postfix(running_r=running_reward, loss=loss.numpy())

        if i % 200 == 0:
            # render an episode
            r_init_state = agent.render_reset()
            r_init_state = tf.expand_dims(tf.expand_dims(r_init_state, 1), 2)
            agent.render_episode(r_init_state, max_epsiode_steps, *models)
            # agent.renv.window.close()
        if episodes_reward:
            if running_reward[0] > reward_threshold and i >= min_episode_criterion:
                break

# Save the model(s)
ix = 0
for model in models:
    r_init_state = agent.render_reset()
    r_init_state = tf.expand_dims(tf.expand_dims(r_init_state, 1), 2)
    _ = model(r_init_state[ix])
    tf.saved_model.save(model, f"/home/tmrob2/PycharmProjects/MORLTAP/saved_models/agent{ix}_lstm_4x4_ma")
    ix += 1

