import collections
import copy
import numpy as np
import tensorflow as tf
import tqdm
from a2c_team_tf.nets.base import DeepActorCritic
from a2c_team_tf.lib.tf2_a2c_base_v2 import MTARL
from a2c_team_tf.utils.dfa import DFAStates, DFA, CrossProductDFA, RewardMachines, RewardMachine, Graph
from abc import ABC
from a2c_team_tf.envs.team_grid_mult import TestEnv
from a2c_team_tf.utils.env_utils import make_env
from a2c_team_tf.utils.data_capture import AsyncWriter
import multiprocessing

# Parameters
width, height = 12, 7
blue_door_pos = ((3 * width) // 4, height // 2)
env_key = 'Team-obj-4x4-door-v0'
seed = 321
np.random.seed(seed)
tf.random.set_seed(seed)
max_steps_per_update = 4
np.random.seed(seed)
tf.random.set_seed(seed)
min_episode_criterion = 100
max_epsiode_steps = 30
max_episodes = 120000
num_tasks = 1
num_agents = 2
# the number of CPUs to run in parallel when generating environments
num_procs = min(multiprocessing.cpu_count(), 30)
recurrence = 1
recurrent = recurrence > 1
one_off_reward = 100.0
normalisation_coeff = 10.
normalisation_coeff2 = 1.0
alpha1 = 0.000005
alpha2 = 0.0001
e, c, chi, lam = 100, 0.8, 1.0, 1.0

def manhatten_dist(p1, p2):
    distance = 0
    #for x, q_i in zip(p1, p2):
    #    distance += min([abs(p_i - q_i) for p_i in x])
    for p_i, q_i in zip(p1, p2):
        distance += abs(p_i - q_i)
    return distance

class KeyDoorSwitch(DFAStates, ABC):
    def __init__(self):
        self.init = "I"
        self.carrying = "C"
        self.dist1 = "C1"
        self.dist2 = "C2"
        self.dist3 = "C3"
        self.dist4 = "C4"
        self.unlock = "U"
        #self.fail = "F"

def pickup_blue_key(data, agent):
    env = data['env'].unwrapped
    if env.agents[agent].carrying:
        if env.agents[agent].carrying.type == "key" and \
            env.agents[agent].carrying.color == "blue":
            return "C"
        else:
            return "I"
    else:
        return "I"

def pickup_blue_key_rm(data, agent):
    if data['word'] == "blue_key":
        return "C"
    else:
        return "I"

def goto_door_dist1(data, agent):
    env = data['env'].unwrapped
    agent_pos = env.agents[agent].cur_pos
    dist = manhatten_dist(blue_door_pos, agent_pos)
    if env.agents[agent].carrying:
        if dist < 9:
            #print(f"C1: agent {agent}, dist {dist}, blue door: {blue_door_pos}, agent pos: {agent_pos}")
            return "C1"
        else:
            return "C"
    else:
        return "I"

def goto_door_dist2(data, agent):
    env = data['env'].unwrapped
    agent_pos = env.agents[agent].cur_pos
    dist = manhatten_dist(blue_door_pos, agent_pos)
    if env.agents[agent].carrying:
        if dist < 7:
            #print(f"C1: agent {agent}, dist {dist}, blue door: {blue_door_pos}, agent pos: {agent_pos}")
            return "C2"
        else:
            return "C1"
    else:
        return "I"
#
#def goto_door_dist1_rm(data, agent):
#    if data['word'] == "key_less4":
#        return "C1"
#    elif data['word'] == "key_greater4":
#        return "C"
#    else:
#        return "I"
#
def goto_door_dist3(data, agent):
    env = data['env'].unwrapped
    agent_pos = env.agents[agent].cur_pos
    dist = manhatten_dist(blue_door_pos, agent_pos)
    if env.agents[agent].carrying:
        if dist < 4:
            #print(f"C2: agent {agent}, dist {dist}, blue door: {blue_door_pos}, agent pos: {agent_pos}")
            return "C3"
        else:
            return "C2"
    else:
        return "I"
#
#def goto_door_dist2_rm(data, agent):
#    if data['word'] == "key_less3":
#        return "C2"
#    elif data['word'] == "key_greater3":
#        return "C1"
#    else:
#        return "I"
#
def goto_door_dist4(data, agent):
    env = data['env'].unwrapped
    agent_pos = env.agents[agent].cur_pos
    dist = manhatten_dist(blue_door_pos, agent_pos)
    if env.agents[agent].carrying:
        if dist < 2:
            #print(f"C3: agent {agent}, dist {dist}, blue door: {blue_door_pos}, agent pos: {agent_pos}")
            return "C4"
        else:
            return "C3"
    else:
        return "I"
#
#def goto_door_dist3_rm(data, agent):
#    if data['word'] == "key_less2":
#        return "C3"
#    elif data['word'] == "key_greater2":
#        return "C2"
#    else:
#        return "I"

def unlock_door(data, agent):
    env = data['env'].unwrapped
    front_pos = env.agents[agent].front_pos
    fwd_cell = env.agents[agent].grid.get(*front_pos)
    if env.agents[agent].carrying:
        if env.agents[agent].carrying.type == "key":
            if fwd_cell is not None:
                if fwd_cell.type == "door" and fwd_cell.is_open and data['action'][agent] == 3:
                    #print(f"agent: {agent} opened door")
                    return "U"
                else:
                    #return "C3"
                    return "C4"
            else:
                #return "C3"
                return "C4"
        else:
            return "I"
    else:
        return "I"

def unlock_door_rm(data, agent):
    if data['word'] == "is_open":
        return "U"
    elif data['word'] == "no_key":
        return "F"
    else:
        #return "C3"
        return "C"

def finished_door(a, b):
    return "U"

def fail(a, b):
    return "F"


##############################################################################
#                   Make DFA and Reward Machines
##############################################################################
def make_key_to_switch_dfa():
    dfa = DFA(start_state="I", acc=["U"], rej=["F"])
    states = KeyDoorSwitch()
    dfa.add_state(states.init, pickup_blue_key)
    #dfa.add_state(states.carrying, unlock_door)
    dfa.add_state(states.carrying, goto_door_dist1)
    dfa.add_state(states.dist1, goto_door_dist2)
    dfa.add_state(states.dist2, goto_door_dist3)
    dfa.add_state(states.dist3, goto_door_dist4)
    dfa.add_state(states.dist4, unlock_door)
    #dfa.add_state(states.unlock, turn_on_red_switch)
    dfa.add_state(states.unlock, finished_door)
    #dfa.add_state(states.fail, fail)
    state_map = {'I': 0, "C": 1, "C1": 2, "C2": 3, "C3": 4, "C4": 5, "U": 6}
    g = Graph(7)
    g.graph = [
        [ 0, 1, 0, 0, 0, 0, 0],
        [-1, 0, 1, 0, 0, 0, 0],
        [-2,-1, 0, 1, 0, 0, 0],
        [-3, 0,-1, 0, 1, 0, 0],
        [-4, 0, 0,-1, 0, 1, 0],
        [-5, 0, 0, 0,-1, 0, 1],
        [-6, 0, 0, 0,0, -1, 0]
    ]
    dist = g.dijkstra(0)
    dfa.distance_from_root(dist, state_map)
    dfa.assign_max_value(max_epsiode_steps)
    return dfa

def make_key_to_switch_dfa_rm():
    dfa = RewardMachine(start_state="I", acc=["U"], rej=["F"],
                        words=["blue_key",
                               "is_open",
                               "no_key",
                               "key_not_open",
                               "key_less4",
                               "key_greater4",
                               "key_less3",
                               "key_greater3",
                               "key_less2",
                               "key_greater2",
                               "key_less1",
                               "key_greater1"
                               ])
    states = KeyDoorSwitch()
    dfa.add_state(states.init, pickup_blue_key_rm)
    dfa.add_state(states.carrying, unlock_door_rm)
    #dfa.add_state(states.carrying, goto_door_dist1_rm)
    #dfa.add_state(states.dist1, goto_door_dist2_rm)
    #dfa.add_state(states.dist2, goto_door_dist3_rm)
    #dfa.add_state(states.dist3, unlock_door_rm)
    dfa.add_state(states.unlock, finished_door)
    #dfa.add_state(states.fail, fail)
    return dfa

#def make_pickup_purple_ball_dfa():
#    dfa = DFA(start_state="I", acc=["D"], rej=["F"])
#    states = PickupObj()
#    dfa.add_state(states.init, pickup_purple_ball)
#    dfa.add_state(states.carrying, drop_purple_ball)
#    dfa.add_state(states.drop, finished)
#    dfa.add_state(states.fail, fail)
#    return dfa
#
#def make_pickup_purple_ball_rm():
#    dfa = RewardMachine(start_state="I", acc=["D"], rej=["F"], words=["purple_ball", "not_box", "box", ""])
#    states = PickupObj()
#    dfa.add_state(states.init, pickup_purple_ball_rm)
#    dfa.add_state(states.carrying, drop_purple_ball_rm)
#    dfa.add_state(states.drop, finished)
#    dfa.add_state(states.fail, fail)
#    return dfa

task1 = make_key_to_switch_dfa()
task1_rm = make_key_to_switch_dfa_rm()

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

xdfa = CrossProductDFA(
        num_tasks=num_tasks,
        dfas=[copy.deepcopy(obj) for obj in [task1]],
        agent=0)

### Caution! The Reward machine must have the same RM ordering as the xDFA sub DFA ordering
#reward_machine = RewardMachines(
#    dfas=[copy.deepcopy(obj) for obj in [task1_rm]],
#    one_off_reward=1.,
#    num_tasks=num_tasks
#)

def f(xdfa: CrossProductDFA, agent):
    xdfa.agent = agent
    return xdfa

### Compute the state space of the rewrd machine 1:1 correspondence with DFA
#reward_machine.compute_state_space()
#v = reward_machine.value_iteration(0.9)
#Phi = -1. * v
#for q, v_ in zip(reward_machine.state_space, v):
#    print(q, v_)
#xdfa.assign_shaped_rewards(Phi)
#xdfa.assign_reward_machine_mappings(reward_machine.state_space, reward_machine.statespace_mapping)

xdfas = [[f(copy.deepcopy(xdfa), agent) for agent in range(num_agents)] for _ in range(num_procs)]

agent = MTARL(envs, num_tasks=num_tasks, num_agents=num_agents, xdfas=xdfas,
              one_off_reward=one_off_reward,
              e=e, c=c, chi=chi, lam=lam, gamma=1.0, lr=alpha1, lr2=alpha2, seed=seed,
              num_procs=num_procs, num_frames_per_proc=max_steps_per_update,
              max_eps_steps=max_epsiode_steps, env_key=env_key,
              normalisation_coef=normalisation_coeff, normalisation_coef2=normalisation_coeff2,
              reward_machine=False, shaped_rewards=True)
i_s_shape = agent.tf_reset2().shape[-1]
models = [DeepActorCritic(envs[0].action_space.n, 64, num_tasks, name=f"agent{i}", activation="tanh", feature_set=i_s_shape) for i in range(num_agents)]

data_writer = AsyncWriter(
    fname_learning='data-4x4-ma-learning_2',
    fname_alloc='data-4x4-ma-alloc_2',
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
        #if i % 10 == 0:
        #     with tf.GradientTape() as tape:
        #         mu = tf.nn.softmax(kappa, axis=0)
        #         alloc_loss = agent.update_alloc_loss(ini_values, mu)
        #     kappa_grads = tape.gradient(alloc_loss, kappa)
        #     #processed_grads = [-agent.lr2 * g for g in kappa_grads]
        #     kappa.assign_add(-alpha2 * kappa_grads)
        #     print("mu\n", mu)
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