import collections
import copy
import numpy as np
import tensorflow as tf
import tqdm
from a2c_team_tf.nets.base import DeepActorCritic
from a2c_team_tf.lib.tf2_a2c_base_v2 import MORLTAP
from a2c_team_tf.utils.dfa import DFAStates, DFA, CrossProductDFA
from abc import ABC
from a2c_team_tf.envs.team_grid_mult import TestEnv
from a2c_team_tf.utils.env_utils import make_env
from a2c_team_tf.utils.data_capture import AsyncWriter
import multiprocessing

# Parameters
env_key = 'Team-obj-5x5-v0'
seed = 321
max_steps_per_update = 10
np.random.seed(seed)
tf.random.set_seed(seed)
min_episode_criterion = 100
max_epsiode_steps = 50
max_episodes = 120000
num_tasks = 2
num_agents = 2
# the number of CPUs to run in parallel when generating environments
num_procs = min(multiprocessing.cpu_count(), 30)
recurrence = 5
recurrent = recurrence > 1
one_off_reward = 100.0
normalisation_coeff = 1.
entropy_coef = 3.
alpha1 = 0.00001
alpha2 = 0.0001
e, c, chi, lam = 85, 0.8, 1.0, 1.0

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

def drop_key(env: TestEnv, agent):
    if env.agents[agent].carrying:
        if env.agents[agent].carrying.type == "key":
            return "C"
        else:
            return "D"
    else:
        return "D"

def finished_key(a, b):
    return "D"

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
    dfa = DFA(start_state="I", acc=["D"], rej=[])
    states = PickupObj()
    dfa.add_state(states.init, pickup_key)
    dfa.add_state(states.carrying, drop_key)
    dfa.add_state(states.drop, finished_key)
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
agent = MORLTAP(envs, num_tasks=num_tasks, num_agents=num_agents, xdfas=xdfas,
                one_off_reward=one_off_reward,
                e=e, c=c, chi=chi, lam=lam, gamma=1.0, lr=alpha1, lr2=alpha2, seed=seed,
                num_procs=num_procs, num_frames_per_proc=max_steps_per_update,
                recurrence=recurrence, max_eps_steps=max_epsiode_steps, env_key=env_key,
                normalisation_coef=normalisation_coeff, use_entropy=True, entropy_coef=entropy_coef)
i_s_shape = agent.tf_reset2().shape[-1]
models = [DeepActorCritic(envs[0].action_space.n, 64, num_tasks, name=f"agent{i}", activation="tanh", feature_set=i_s_shape) for i in range(num_agents)]

data_writer = AsyncWriter(
    fname_learning='data-4x4-ma-learning',
    fname_alloc='data-4x4-ma-alloc',
    num_agents=num_agents,
    num_tasks=num_tasks)

############################################################################
# TRAIN AGENT SCRIPT
#############################################################################
episodes_reward = collections.deque(maxlen=min_episode_criterion)
kappa = tf.Variable(np.full([num_agents, num_tasks], 1.0 / num_agents), dtype=tf.float32)
mu_thresh = np.ones([num_agents, num_tasks]) - np.ones([num_agents, num_tasks]) * 0.03

with tqdm.trange(max_episodes) as t:
    # get the initial state
    state = agent.tf_reset2()
    state = tf.squeeze(state)
    state = tf.expand_dims(tf.transpose(state, perm=[1, 0, 2]), 2)
    log_reward = tf.zeros([num_agents, num_procs, num_tasks + 1], dtype=tf.float32)
    indices = agent.tf_1d_indices()
    mu = tf.nn.softmax(kappa, axis=0)
    print("mu ", mu)
    for i in t:
        state, log_reward, running_reward, loss, ini_values = agent.train(state, log_reward, indices, mu, *models)
        if i % 50 == 0:
             with tf.GradientTape() as tape:
                 mu = tf.nn.softmax(kappa, axis=0)
                 alloc_loss = agent.update_alloc_loss(ini_values, mu)
             kappa_grads = tape.gradient(alloc_loss, kappa)
             #processed_grads = [-agent.lr2 * g for g in kappa_grads]
             kappa.assign_add(-alpha2 * kappa_grads)
             print("mu\n", mu)
        t.set_description(f"Batch: {i}")
        for reward in running_reward:
            episodes_reward.append(reward.numpy().flatten())
        if episodes_reward:
            running_reward = np.around(np.mean(episodes_reward, 0), decimals=2)
            #data_writer.write(running_reward)
            data_writer.write({'learn': running_reward, 'alloc': mu.numpy()})
            t.set_postfix(running_r=running_reward)
        if i % 200 == 0:
            # render an episode
            r_init_state = agent.render_reset()
            r_init_state = tf.expand_dims(tf.expand_dims(r_init_state, 1), 2)
            agent.render_episode(r_init_state, max_epsiode_steps, *models)
            # agent.renv.window.close()
        ### Define break clause
        if episodes_reward:
            running_tasks = np.reshape(running_reward, [num_agents, num_tasks + 1])
            running_tasks_ = running_tasks[:, 1:]
            mu_term = (mu > mu_thresh).numpy().astype(np.float32)
            allocated_task_rewards = mu_term * running_tasks_
            task_term = all([np.any(np.greater_equal(allocated_task_rewards[:, i], e)) for i in range(num_tasks)])
            if all(x > c for x in running_reward[::3]) and task_term:
                break
print("mu ", mu)

# Save the model(s)
ix = 0
for model in models:
    r_init_state = agent.render_reset()
    r_init_state = tf.expand_dims(tf.expand_dims(r_init_state, 1), 2)
    _ = model(r_init_state[ix])  # calling the model makes tells tensorflow the size of the model
    tf.saved_model.save(model, f"/home/tmrob2/PycharmProjects/MORLTAP/saved_models/agent{ix}_lstm_4x4_ma")
    ix += 1