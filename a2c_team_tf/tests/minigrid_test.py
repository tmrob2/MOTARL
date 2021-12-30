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

# what are the conceptual changes that need to be made
# * the environments no longer contain a set of states for each of the agents
# * the model will no longer need to contain a set of models for each agent
# the big change is with how the parallel environments work. We kind of want to set
# up a matrix of environments and then loop over these environments

class PickupObj(DFAStates, ABC):
    def __init__(self):
        self.init = "I"
        self.carrying = "C"
        self.drop = "D"

def pickup_ball(env: MultObjNoGoal4x4, agent):
    if env.carrying:
        if env.carrying.type == "ball":
            return "C"
        else:
            return "I"
    else:
        return "I"

def drop_ball(env: MultObjNoGoal4x4, agent):
    if env.carrying:
        if env.carrying.type == "ball":
            return "C"
        else:
            return "D"
    else:
        return "D"

def pickup_key(env: MultObjNoGoal4x4, agent):
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

num_tasks = 2
num_agents = 2
recurrent = True
num_procs = 30
seed = 123
env_key = 'Mult-obj-4x4-v0'
max_steps_per_episode = 50
max_steps_per_update = 10
recurrence = 10
max_episode_steps = 2
e, c, mu, chi, lam = tf.constant([0.8], dtype=tf.float32), -5.0, 0.5, 1.0, 1.0
envs = []
ball = make_pickupanddrop_ball_dfa()
key = make_pickup_key_dfa()
# generate a cross product DFA for each agent and each proc
xdfas = [[
    CrossProductDFA(
        num_tasks=num_tasks,
        dfas=[copy.deepcopy(obj) for obj in [key, ball]],
        agent=agent) for _ in range(num_procs)] for agent in range(num_agents)]
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
observation_space = envs[0][0].observation_space
action_space = envs[0][0].action_space.n
# generate a list of input samples for each agent and input this into the model
models = [ActorCrticLSTM(action_space, num_tasks, recurrent) for _ in range(num_agents)]
agent = MORLTAP(envs, models, num_tasks=num_tasks, num_agents=num_agents, xdfas=xdfas, one_off_reward=1.0,
                e=e, c=c, chi=chi, lam=lam, gamma=1.0, lr=5e-5, seed=seed,
                num_procs=num_procs, num_frames_per_proc=max_steps_per_update,
                recurrence=recurrence, max_eps_steps=max_episode_steps, env_key=env_key,
                observation_space=observation_space, action_space=action_space, flatten_env=True)

# kappa = tf.Variable(np.full(num_agents * num_tasks, 1.0 / num_agents), dtype=tf.float32)
# mu = tf.nn.softmax(tf.reshape(kappa, shape=[num_agents, num_tasks]), axis=0)
initial_states = agent.tf_reset2()
print("len initial states ", len(initial_states), " state shape ", initial_states[0].shape)
initial_states_i = tf.expand_dims(initial_states[0], 1)
#
action_logits, value = models[0](initial_states_i)
print("action logits shape ", action_logits.shape, "value shape ", value.shape)
# # # # process the action logits
action_logits = tf.squeeze(action_logits)
actions = tf.random.categorical(action_logits, 1, dtype=tf.int32)
actions = tf.squeeze(actions)
print("actions shape ", actions.shape)
action_probs = tf.nn.softmax(action_logits)
indices = tf.transpose([tf.range(action_probs.shape[0], dtype=tf.int32), actions])
action_probs_selected = tf.gather_nd(action_probs, indices=indices)
print("action probs shape ", action_probs_selected.shape)
state, reward, done = agent.tf_env_step(actions, 0)
print(f"state shape: {state.shape}, rewards: {reward.shape}, done: {done.shape}")
r_init_state = agent.render_reset()
print("render init state len ", len(r_init_state), "render init state shape ", r_init_state[0].shape)
r_init_state = [tf.expand_dims(tf.expand_dims(r_init_state[i], 0), 1) for i in range(num_agents)]
print("render reset state shape: ", r_init_state[0].shape)
render_action_logits, _ = models[0](r_init_state[0])
print("render action logits shape: ", render_action_logits.shape)
render_action_logits = tf.squeeze(render_action_logits, 1)
action = tf.random.categorical(render_action_logits, 1, dtype=tf.int32)[0, 0].numpy()
print("action: ", action)
state, _, _, _ = agent.renv[0].step(action)
state, reward, done = agent.tf_render_env_step(action, 0)
agent.render_episode(r_init_state, 500, *models)
#
log_rewards = tf.zeros([num_procs, num_tasks + 1], dtype=tf.float32)
actions, observations, values, rewards, masks, state_, running_rewards, log_rewards = \
      agent.collect_batch(initial_states, log_rewards, *models)
# print("action shape ", actions.shape)
# print("observations shape ", observations.shape)
# print("values shape ", values.shape)
# print("rewards shape ", rewards.shape)
# print("masks shape ", masks.shape)
# print("state shape ", state_.shape)
# print("running rewards shape ", running_rewards.shape)
# print("log rewards shape ", log_rewards.shape)
# indices = agent.tf_1d_indices()
# state = initial_states
# # state, log_rewards, running_rewards, loss, ini_values = agent.train(state, log_rewards, indices, mu, *models)
#
# with tqdm.trange(10) as t:
#     for i in t:
#         state, log_rewards, running_rewards, loss, ini_values = agent.train(state, log_rewards, indices, mu, *models)
#         with tf.GradientTape() as tape:
#             mu = tf.nn.softmax(tf.reshape(kappa, shape=[num_agents, num_tasks]), axis=0)
#             # print("mu: ", mu)
#             alloc_loss = agent.update_alloc_loss(ini_values, mu)  # alloc loss
#         kappa_grads = tape.gradient(alloc_loss, kappa)
#         processed_grads = [-0.001 * g for g in kappa_grads]
#         kappa.assign_add(processed_grads)










