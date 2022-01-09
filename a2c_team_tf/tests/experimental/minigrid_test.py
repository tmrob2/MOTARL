import collections
import copy
import numpy as np
import tensorflow as tf
import tqdm
from a2c_team_tf.nets.base import DeepActorCritic
from a2c_team_tf.lib.experimental.tf2_a2c_base import MTARL
from a2c_team_tf.utils.dfa import DFAStates, DFA, CrossProductDFA
from abc import ABC
from a2c_team_tf.envs.experimental.minigrid_fetch_mult import MultObjNoGoal4x4
from a2c_team_tf.utils.env_utils import make_env
from a2c_team_tf.utils.data_capture import AsyncWriter
import multiprocessing

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
        self.fail = "F"

def pickup_ball(data, agent):
    env = data['env'].unwrapped
    if env.carrying:
        if env.carrying.type == "ball":
            return "C"
        else:
            return "I"
    else:
        return "I"

def drop_ball(data, agent):
    env = data['env'].unwrapped
    if env.carrying:
        if env.unwrapped.carrying.type == "ball":
            return "C"
        else:
            fwd_pos = env.front_pos
            fwd_cell = env.grid.get(*fwd_pos)
            if fwd_cell is not None:
                if fwd_cell.type == "box":
                    return "D"
                else:
                    return "F"
            else:
                "F"
    else:
        return "D"

def pickup_key(data, agent):
    env = data['env'].unwrapped
    if env.carrying:
        if env.carrying.type == "key":
            return "C"
        else:
            return "I"
    else:
        return "I"

def drop_key(data, agent):
    env = data['env'].unwrapped
    if env.carrying:
        if env.carrying.type == "key":
            return "C"
        else:
            fwd_pos = env.front_pos
            fwd_cell = env.grid.get(*fwd_pos)
            if fwd_cell is not None:
                if fwd_cell.type == "box":
                    return "D"
                else:
                    return "F"
            else:
                "F"
    else:
        return "D"

def finished(a, b):
    return "D"

def fail(a, b):
    return "F"

def make_pickupanddrop_ball_dfa():
    dfa = DFA(start_state="I", acc=["D"], rej=["F"])
    states = PickupObj()
    dfa.add_state(states.init, pickup_ball)
    dfa.add_state(states.carrying, drop_ball)
    dfa.add_state(states.drop, finished)
    dfa.add_state(states.fail, fail)
    return dfa

def make_pickup_key_dfa():
    dfa = DFA(start_state="I", acc=["D"], rej=["F"])
    states = PickupObj()
    dfa.add_state(states.init, pickup_key)
    dfa.add_state(states.carrying, drop_key)
    dfa.add_state(states.drop, finished)
    dfa.add_state(states.fail, fail)
    return dfa

num_tasks = 2
num_agents = 2
num_procs = min(multiprocessing.cpu_count(), 30)
seed = 123
env_key = 'Mult-obj-5x5-v0'
max_steps_per_episode = 50
max_steps_per_update = 5
recurrence = 1
max_episode_steps = 2
alpha1 = 0.0001
alpha2 = 0.001
one_off_reward = 10.0
e, c, mu, chi, lam = tf.constant([0.8 * one_off_reward], dtype=tf.float32), -5.0, 0.5, 1.0, 1.0
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
q1 = tf.queue.FIFOQueue(capacity=max_steps_per_update * num_procs * num_tasks * num_agents + 1, dtypes=[tf.float32])
q2 = tf.queue.FIFOQueue(capacity=max_steps_per_update * num_procs * num_tasks * num_agents + 1, dtypes=[tf.int32])
# generate a list of input samples for each agent and input this into the model
log_rewards = tf.zeros([num_agents, num_procs, num_tasks + 1], dtype=tf.float32)
agent = MTARL(envs, num_tasks=num_tasks, num_agents=num_agents, xdfas=xdfas, one_off_reward=10.0,
              e=e, c=c, chi=chi, lam=lam, gamma=1.0, lr=alpha1, seed=seed,
              num_procs=num_procs, num_frames_per_proc=max_steps_per_update,
              recurrence=recurrence, max_eps_steps=max_episode_steps, env_key=env_key,
              observation_space=observation_space, action_space=action_space, flatten_env=True,
              q1=q1, q2=q2)

kappa = tf.Variable(np.full([num_agents, num_tasks], 1.0 / num_agents), dtype=tf.float32)
mu = tf.nn.softmax(kappa, axis=0)
initial_states = agent.tf_reset2()
models = [DeepActorCritic(action_space, 64, num_tasks, f"agent{i}", initial_states.shape[-1]) for i in range(num_agents)]
initial_states = tf.expand_dims(initial_states, 2)
print("len initial states ", initial_states.shape, " state shape ", initial_states[0].shape)
#
action_logits, value = models[0](initial_states[0])
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
#agent.render_episode(r_init_state, 5, *models)
#
initial_states = agent.tf_reset2()
initial_states = tf.expand_dims(initial_states, 2)
print("init state shape batch input ", initial_states.shape)
all_actions, all_obss, all_values, all_rewards, all_masks, all_states, log_rewards = \
            agent.collect_batch(initial_states, log_rewards, *models)
print("action shape ", actions.shape)
print("observations shape ", all_obss.shape)
print("values shape ", all_values.shape)
print("rewards shape ", all_rewards.shape)
print("masks shape ", all_masks.shape)
print("state shape ", all_states.shape)
print("log rewards shape ", log_rewards.shape)
#
indices = agent.tf_1d_indices()
# agent.train_preprocess(initial_states, log_rewards, mu, *models)
# #
# state = initial_states
# # agent.train(state, log_rewards, indices, mu, *models)
# state, log_rewards, running_rewards, loss, ini_values = \
#     agent.train(state, log_rewards, indices, mu, *models)
# print(f"state {state.shape}, log_rewards {log_rewards.shape}, running_rewards: {running_rewards.shape}"
#       f"loss: {loss.shape}, ini_values: {ini_values.shape}")
# print(f"running rewards {running_rewards}")
# #
initial_states = agent.tf_reset2()
initial_states = tf.expand_dims(initial_states, 2)
state = initial_states
log_rewards = tf.zeros([num_agents, num_procs, num_tasks + 1], dtype=tf.float32)
running_rewards = [collections.deque(maxlen=100) for _ in range(num_agents)]
#data_writer = AsyncWriter('minigrid-learning', 'minigrid-alloc', num_agents, num_tasks)
#
with tqdm.trange(10000) as t:
    for i in t:
        state, loss, ini_values, log_rewards = agent.train(state, indices, mu, log_rewards, *models)
        # calculate queue length
        for _ in range(agent.q1.size()):
            index = agent.q2.dequeue()
            value = agent.q1.dequeue()
            running_rewards[index].append(value.numpy())
        if all(len(x) >= 1 for x in running_rewards):
            running_rewards_x_agent = np.around(np.array([np.mean(running_rewards[j], 0) for j in range(num_agents)]).flatten(), decimals=2)
            t.set_description(f"Episode {i}")
            t.set_postfix(running_reward=running_rewards_x_agent)
            #data_writer.write({'learn': running_rewards_x_agent, 'alloc': mu.numpy()})
        # with tf.GradientTape() as tape:
        #     mu = tf.nn.softmax(tf.reshape(kappa, shape=[num_agents, num_tasks]), axis=0)
        #     # print("mu: ", mu)
        #     alloc_loss = agent.update_alloc_loss(ini_values, mu)  # alloc loss
        # kappa_grads = tape.gradient(alloc_loss, kappa)
        # kappa.assign_add(alpha2 * kappa_grads)
        if i % 500 == 0 and i > 0:
            r_init_state = agent.render_reset()
            r_init_state = [tf.expand_dims(tf.expand_dims(r_init_state[i], 0), 1) for i in range(num_agents)]
            agent.render_episode(r_init_state, max_steps_per_episode, *models)










