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
from a2c_team_tf.envs.minigrid_fetch_mult import MultObjNoGoal4x4
from a2c_team_tf.utils.env_utils import make_env
from a2c_team_tf.utils.data_capture import AsyncWriter
import multiprocessing

# Parameters
env_key = 'Mult-obj-4x4-v0'
seed = 123
max_steps_per_update = 10
np.random.seed(seed)
tf.random.set_seed(seed)
min_episode_criterion = 100
max_steps_per_episode = 50
max_episodes = 20000
num_tasks = 2
num_agents = 2
# the number of CPUs to run in parallel when generating environments
num_procs = min(multiprocessing.cpu_count(), 30)
recurrence = 10
recurrent = recurrence > 1
alpha1 = 0.001
alpha2 = 0.001
one_off_reward = 10.0

# construct DFAs
class PickupObj(DFAStates, ABC):
    def __init__(self):
        self.init = "I"
        self.carrying = "C"
        self.drop = "D"
        self.fail = "F"

def pickup_ball(env: MultObjNoGoal4x4, _):
    if env.carrying:
        if env.carrying.type == "ball":
            return "C"
        else:
            return "I"
    else:
        return "I"

def drop_ball(env: MultObjNoGoal4x4, _):
    if env.carrying:
        if env.carrying.type == "ball":
            return "C"
        else:
            return "D"
    else:
        return "D"

def pickup_key(env: MultObjNoGoal4x4, _):
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
for j in range(num_agents):
    agent_envs = []
    for i in range(num_procs):
        eseed = seed
        agent_envs.append(
            make_env(
                env_key=env_key,
                max_steps_per_episode=max_steps_per_episode,
                seed=seed + 100 * i + 1000 * (j + 1),
                apply_flat_wrapper=True))
    envs.append(agent_envs)
#############################################################################
#  Initialise data structures
#############################################################################
ball = make_pickupanddrop_ball_dfa()
key = make_pickup_key_dfa()
xdfas = [[
    CrossProductDFA(
        num_tasks=num_tasks,
        dfas=[copy.deepcopy(obj) for obj in [key, ball]],
        agent=agent) for _ in range(num_procs)] for agent in range(num_agents)]
observation_space = envs[0][0].observation_space
action_space = envs[0][0].action_space.n
e, c, chi, lam = 0.8 * one_off_reward, -10., 1.0, 1.0

# reward capture queues for tensorflow graph api
q1 = tf.queue.FIFOQueue(capacity=max_steps_per_update * num_procs * num_tasks * num_agents + 1, dtypes=[tf.float32])
q2 = tf.queue.FIFOQueue(capacity=max_steps_per_update * num_procs * num_tasks * num_agents + 1, dtypes=[tf.int32])

models = [ActorCrticLSTM(action_space, num_tasks, recurrent) for _ in range(num_agents)]
log_rewards = tf.Variable(tf.zeros([num_agents, num_procs, num_tasks + 1], dtype=tf.float32))
agent = MORLTAP(envs, num_tasks=num_tasks, num_agents=num_agents, xdfas=xdfas, one_off_reward=10.0,
                e=e, c=c, chi=chi, lam=lam, gamma=1.0, lr=alpha1, seed=seed,
                num_procs=num_procs, num_frames_per_proc=max_steps_per_update,
                recurrence=recurrence, max_eps_steps=max_episodes, env_key=env_key,
                observation_space=observation_space, action_space=action_space, flatten_env=True,
                q1=q1, q2=q2, log_reward=log_rewards)
data_writer = AsyncWriter(
    fname_learning='data-4x4-lstm-ma-learning',
    fname_alloc='data-4x4-lstm-ma-alloc',
    num_agents=num_agents,
    num_tasks=num_tasks)

############################################################################
# TRAIN AGENT SCRIPT
#############################################################################
episodes_reward = collections.deque(maxlen=min_episode_criterion)
running_rewards = [collections.deque(maxlen=100) for _ in range(num_agents)]
initial_states = agent.tf_reset2()
initial_states = tf.expand_dims(initial_states, 2)
state = initial_states
indices = agent.tf_1d_indices()
kappa = tf.Variable(np.full([num_agents, num_tasks], 1.0 / num_agents), dtype=tf.float32)
mu = tf.nn.softmax(kappa, axis=1)

with tqdm.trange(max_episodes) as t:
    for i in t:
        state, loss, ini_values = agent.train(state, indices, mu, *models)
        # calculate queue length
        for _ in range(agent.q1.size()):
            index = agent.q2.dequeue()
            value = agent.q1.dequeue()
            running_rewards[index].append(value.numpy())
        if all(len(x) >= 1 for x in running_rewards):
            running_rewards_x_agent = np.around(
                np.array([np.mean(running_rewards[j], 0) for j in range(num_agents)]).flatten(), decimals=2)
            t.set_description(f"Episode {i}")
            t.set_postfix(running_reward=running_rewards_x_agent)
            data_writer.write({'learn': running_rewards_x_agent, 'alloc': mu.numpy()})
        # with tf.GradientTape() as tape:
        #     mu = tf.nn.softmax(tf.reshape(kappa, shape=[num_agents, num_tasks]), axis=0)
        #     # print("mu: ", mu)
        #     alloc_loss = agent.update_alloc_loss(ini_values, mu)  # alloc loss
        # kappa_grads = tape.gradient(alloc_loss, kappa)
        # kappa.assign_add(alpha2 * kappa_grads)
        if i % 2000 == 0 and i > 0:
            r_init_state = agent.render_reset()
            r_init_state = [tf.expand_dims(tf.expand_dims(r_init_state[i], 0), 1) for i in range(num_agents)]
            agent.render_episode(r_init_state, max_steps_per_episode, *models)

# Save the model(s)
ix = 0
for model in models:
    r_init_state = agent.render_reset()
    r_init_state = [tf.expand_dims(tf.expand_dims(r_init_state[i], 0), 1) for i in range(num_agents)]
    _ = model(r_init_state[ix])  # calling the model makes tells tensorflow the size of the model
    tf.saved_model.save(model, f"/home/tmrob2/PycharmProjects/MORLTAP/saved_models/agent{ix}_lstm_4x4_ma")
    ix += 1

