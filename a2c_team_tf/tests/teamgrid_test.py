import collections
import copy

import gym
import numpy as np
import tensorflow as tf
import tqdm

from a2c_team_tf.nets.base import ActorCrticLSTM
from a2c_team_tf.lib.tf2_a2c_base import MORLTAP
from a2c_team_tf.utils.dfa import DFAStates, DFA, CrossProductDFA
from abc import ABC
from a2c_team_tf.envs.team_grid_mult import TestEnv
from a2c_team_tf.utils.env_utils import make_env

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

num_tasks = 2
num_agents = 2
recurrent = True
num_procs = 30
seed = 123
env_key = 'Team-obj-5x5-v0'
max_steps_per_episode = 50
max_steps_per_update = 10
recurrence = 10
max_episode_steps = 2
e, c, mu, chi, lam = tf.constant([0.8], dtype=tf.float32), -5.0, 0.5, 1.0, 1.0
envs = []
ball = make_pickupanddrop_ball_dfa()
key = make_pickup_key_dfa()
xdfas = [[
    CrossProductDFA(
        num_tasks=num_tasks,
        dfas=[copy.deepcopy(obj) for obj in [key, ball]],
        agent=agent) for agent in range(num_agents)] for _ in range(num_procs)]
for i in range(num_procs):
    eseed = seed
    envs.append(make_env(env_key=env_key, max_steps_per_episode=max_steps_per_episode, apply_flat_wrapper=False))
# generate a list of input samples for each agent and input this into the model
models = [ActorCrticLSTM(envs[0].action_space.n, num_tasks, recurrent) for _ in range(num_agents)]
agent = MORLTAP(envs, models, num_tasks=num_tasks, num_agents=num_agents, xdfas=xdfas, one_off_reward=1.0,
                e=e, c=c, chi=chi, lam=lam, gamma=1.0, lr=5e-5, seed=seed,
                num_procs=num_procs, num_frames_per_proc=max_steps_per_update,
                recurrence=recurrence, max_eps_steps=max_episode_steps, env_key=env_key)

kappa = tf.Variable(np.full(num_agents * num_tasks, 1.0 / num_agents), dtype=tf.float32)
mu = tf.nn.softmax(tf.reshape(kappa, shape=[num_agents, num_tasks]), axis=0)
initial_states = agent.tf_reset2()
initial_states = tf.squeeze(initial_states)
initial_states = tf.expand_dims(tf.transpose(initial_states, perm=[1, 0, 2]), 2)
action_logits_x_agents, value_x_agents = agent.call_models(initial_states, *models)
# # # process the action logits
print("action logits shape ", action_logits_x_agents.shape, "values shape ", value_x_agents.shape)
action_logits_x_agents = tf.squeeze(action_logits_x_agents)
print("action logits shape: ", action_logits_x_agents.shape)
actions = agent.collect_actions(action_logits_x_agents)
print("actions: ", actions)
print("actions: ", actions.shape)
print("actions transpose: ", tf.transpose(actions))
state, reward, done = agent.tf_env_step(actions)
print(f"state shape: {state.shape}, rewards: {reward.shape}, done: {done.shape}")
r_init_state = agent.render_reset()
r_init_state = tf.expand_dims(tf.expand_dims(r_init_state, 1), 2)
print("render reset state shape: ", r_init_state.shape)
render_action_logits, _ = agent.call_models(r_init_state, *models)
print("render action logits shape: ", render_action_logits.shape)
render_action_logits = tf.squeeze(render_action_logits, 1)
actions = agent.collect_actions(render_action_logits)
print("action shape: ", actions.shape)
state, reward, done = agent.tf_render_env_step(actions)
#agent.render_episode(r_init_state, 500)

log_rewards = tf.zeros([num_agents, num_procs, num_tasks + 1], dtype=tf.float32)
actions, observations, values, rewards, masks, state_, running_rewards, log_rewards = \
     agent.collect_batch(initial_states, log_rewards, *models)
print("action shape ", actions.shape)
print("observations shape ", observations.shape)
print("values shape ", values.shape)
print("rewards shape ", rewards.shape)
print("masks shape ", masks.shape)
print("state shape ", state_.shape)
print("running rewards shape ", running_rewards.shape)
print("log rewards shape ", log_rewards.shape)
indices = agent.tf_1d_indices()
state = initial_states
# state, log_rewards, running_rewards, loss, ini_values = agent.train(state, log_rewards, indices, mu, *models)

with tqdm.trange(10) as t:
    for i in t:
        state, log_rewards, running_rewards, loss, ini_values = agent.train(state, log_rewards, indices, mu, *models)
        with tf.GradientTape() as tape:
            mu = tf.nn.softmax(tf.reshape(kappa, shape=[num_agents, num_tasks]), axis=0)
            # print("mu: ", mu)
            alloc_loss = agent.update_alloc_loss(ini_values, mu)  # alloc loss
        kappa_grads = tape.gradient(alloc_loss, kappa)
        processed_grads = [-0.001 * g for g in kappa_grads]
        kappa.assign_add(processed_grads)










