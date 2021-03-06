import collections
import copy
import gym
import numpy as np
import tensorflow as tf
import tqdm
from a2c_team_tf.nets.base import DeepActorCritic
from a2c_team_tf.lib.tf2_a2c_base_v2 import MTARL
from a2c_team_tf.utils.dfa import DFAStates, DFA, CrossProductDFA, RewardMachines, RewardMachine
from abc import ABC
from a2c_team_tf.envs.team_grid_mult import TestEnv
from a2c_team_tf.utils.env_utils import make_env

class PickupObj(DFAStates, ABC):
    def __init__(self):
        self.init = "I"
        self.carrying = "C"
        self.drop = "D"

def pickup_ball(data, agent):
    if data['env'].agents[agent].carrying:
        if data['env'].agents[agent].carrying.type == "ball":
            return "C"
        else:
            return "I"
    else:
        return "I"

def drop_ball(data, agent):
    if data['env'].agents[agent].carrying:
        if data['env'].agents[agent].carrying.type == "ball":
            return "C"
        else:
            return "D"
    else:
        return "D"

def pickup_key(data, agent):
    if data['env'].agents[agent].carrying:
        if data['env'].agents[agent].carrying.type == "key":
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

## Reward machines
def pickup_ball_rm(data, agent):
    if data['word'] == "ball":
        return "C"
    else:
        return "I"

def drop_ball_rm(data, agent):
    if data['word'] == "ball":
        return "C"
    else:
        return "D"

def pickup_key_rm(data, agent):
    if data['word'] == "key":
        return "C"
    else:
        return "I"

def finished_key_rm(a, b):
    return "C"

def finished_ball_rm(a, b):
    return "D"

def make_pickup_ball_rm():
    rm = RewardMachine(start_state="I", acc=["D"], rej=[], words=["ball", ""])
    states = PickupObj()
    rm.add_state(states.init, pickup_ball_rm)
    rm.add_state(states.carrying, drop_ball_rm)
    rm.add_state(states.drop, finished_ball_rm)
    return rm

def make_pickup_key_rm():
    rm = RewardMachine(start_state="I", acc=["C"], rej=[], words=["key", ""])
    states = PickupObj()
    rm.add_state(states.init, pickup_key_rm)
    rm.add_state(states.carrying, finished_key_rm)
    return rm

num_tasks = 2
num_agents = 2
num_procs = 2
seed = 123
env_key = 'Team-obj-5x5-v0'
max_steps_per_episode = 50
max_steps_per_update = 10
one_off_reward = 1.0
recurrence = 1
max_episode_steps = 2
e, c, mu, chi, lam = 10, -1.5, 0.5, 1.0, 1.0
envs = []
ball = make_pickupanddrop_ball_dfa()
key = make_pickup_key_dfa()
xdfa = CrossProductDFA(
        num_tasks=num_tasks,
        dfas=[copy.deepcopy(obj) for obj in [key, ball]],
        agent=0)

ball_rm = make_pickup_ball_rm()
key_rm = make_pickup_key_rm()

### Caution! The Reward machine must have the same RM ordering as the xDFA sub DFA ordering
reward_machine = RewardMachines(
    dfas=[copy.deepcopy(obj) for obj in [key_rm, ball_rm]],
    one_off_reward=1.0,
    num_tasks=num_tasks
)

def f(xdfa: CrossProductDFA, agent):
    xdfa.agent = agent
    return xdfa

### Compute the state space of the rewrd machine 1:1 correspondence with DFA
reward_machine.compute_state_space()
v = reward_machine.value_iteration(0.9)
Phi = -1. * v
print(v)
xdfa.assign_shaped_rewards(Phi)
xdfa.assign_reward_machine_mappings(reward_machine.state_space, reward_machine.statespace_mapping)

print("Cross product DFA state space \n", xdfa.state_space)
xdfas = [[f(copy.deepcopy(xdfa), agent) for agent in range(num_agents)] for _ in range(num_procs)]
for i in range(num_procs):
    eseed = seed
    envs.append(make_env(env_key=env_key, max_steps_per_episode=max_steps_per_episode, apply_flat_wrapper=False))
# generate a list of input samples for each agent and input this into the model
agent = MTARL(envs, num_tasks=num_tasks, num_agents=num_agents, xdfas=xdfas, one_off_reward=1.0,
              e=e, c=c, chi=chi, lam=lam, gamma=1.0, lr=5e-5, seed=seed,
              num_procs=num_procs, num_frames_per_proc=max_steps_per_update,
              recurrence=recurrence, max_eps_steps=max_episode_steps, env_key=env_key)

kappa = tf.Variable(np.full([num_agents, num_tasks], 1.0 / num_agents), dtype=tf.float32)
mu = tf.nn.softmax(kappa, axis=0)
initial_states = agent.tf_reset2()
models = [DeepActorCritic(envs[0].action_space.n, 64, num_tasks, name=f"agent{i}", activation="tanh", feature_set=initial_states.shape[-1]) for i in range(num_agents)]

print("raw init state ", initial_states.shape)
initial_states = tf.squeeze(initial_states)
print("squeeze init state ", initial_states.shape)
initial_states = tf.transpose(initial_states, perm=[1, 0, 2])
print("transpose init state ", initial_states.shape)
initial_states = tf.expand_dims(initial_states, 2)
action_logits_x_agents, value_x_agents = agent.call_models(initial_states, *models)
# # # process the action logits
print("action logits shape ", action_logits_x_agents.shape, "values shape ", value_x_agents.shape)
action_logits_x_agents = tf.squeeze(action_logits_x_agents)
actions = agent.collect_actions(action_logits_x_agents)
print("actions: ", actions)
print("actions: ", actions.shape)
print("actions transpose: ", tf.transpose(actions))
state, reward, done = agent.tf_env_step(actions)
print("rewards ", reward)
print(f"state shape: {state.shape}, rewards: {reward.shape}, done: {done.shape}")
r_init_state = agent.render_reset()
print("r init shape ", r_init_state.shape)
r_init_state = tf.expand_dims(tf.expand_dims(r_init_state, 1), 1)
print("render reset state shape: ", r_init_state.shape)
render_action_logits, _ = agent.call_models(r_init_state, *models)
print("render action logits shape: ", render_action_logits.shape)
render_action_logits = tf.squeeze(render_action_logits, 1)
actions = agent.collect_actions(render_action_logits)
print("action shape: ", actions.shape)
state, reward, done = agent.tf_render_env_step(actions)
#agent.render_episode(r_init_state, 500, *models)
##
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
state, log_rewards, running_rewards, loss, ini_values = agent.train(state, log_rewards, indices, mu, *models)

episodes_reward = collections.deque(maxlen=100)

with tqdm.trange(1000) as t:
    for i in t:
        state, log_rewards, running_reward, loss, ini_values = agent.train(state, log_rewards, indices, mu, *models)
        with tf.GradientTape() as tape:
            mu = tf.nn.softmax(kappa, axis=0)
            # print("mu: ", mu)
            alloc_loss = agent.update_alloc_loss(ini_values, mu)  # alloc loss
        kappa_grads = tape.gradient(alloc_loss, kappa)
        processed_grads = [-0.001 * g for g in kappa_grads]
        kappa.assign_add(processed_grads)
        for reward in running_reward:
            episodes_reward.append(reward.numpy().flatten())
        if episodes_reward:
            running_reward = np.around(np.mean(episodes_reward, 0), decimals=2)
            t.set_postfix(running_r=running_reward)