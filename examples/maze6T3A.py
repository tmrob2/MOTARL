import collections
import copy
import numpy as np
import tensorflow as tf
import tqdm
from a2c_team_tf.nets.base import DeepActorCritic
from a2c_team_tf.lib.tf2_a2c_base_v2 import MTARL
from a2c_team_tf.utils.dfa import DFAStates, DFA, CrossProductDFA, RewardMachines, RewardMachine
from abc import ABC
from a2c_team_tf.utils.env_utils import make_env
from a2c_team_tf.utils.data_capture import AsyncWriter
import multiprocessing

env_key = 'DualDoors-v0'
seed = 123
#env = make_env(env_key, 50, seed, False)
#env.reset()
#for _ in range(10000):
#    env.render('human')
max_steps_per_update = 5
np.random.seed(seed)
tf.random.set_seed(seed)
min_episode_criterion = 100
max_epsiode_steps = 250
max_episodes = 120000
num_tasks = 2
num_agents = 2
# the number of CPUs to run in parallel when generating environments
num_procs = min(multiprocessing.cpu_count(), 30)
recurrence = 1
recurrent = recurrence > 1
one_off_reward = 100.0
normalisation_coeff = 10.
alpha1 = 0.000001
alpha2 = 0.0001
e, c, chi, lam = 100, 0.8, 1.0, 1.0
width, height = 19, 11
blue_door_poss = (2 * width) // 3, (2 * height) // 3
red_door_pos = (width // 3, height // 4)

def manhatten_dist(p1, p2):
    distance = 0
    for x, q_i in zip(p1, p2):
        distance += min([abs(p_i - q_i) for p_i in x])
    return distance

# construct DFAs
class PickupObj(DFAStates, ABC):
    def __init__(self):
        self.init = "I"
        self.carrying = "C"
        self.drop = "D"
        self.fail = "F"

class Goal(DFAStates, ABC):
    def __init__(self):
        self.init = "I"
        self.goal = "G"

class ActivateSwitch(DFAStates, ABC):
    def __init__(self):
        self.init = "I"
        self.switch = "S"

class KeyDoorSwitch(DFAStates, ABC):
    def __init__(self):
        self.init = "I"
        self.carrying = "C"
        self.dist1 = "C1"
        self.dist2 = "C2"
        self.dist3 = "C3"
        self.unlock = "U"
        self.fail = "F"

def pickup_purple_ball(data, agent):
    if data['env'].agents[agent].carrying:
        if data['env'].agents[agent].carrying.type == "ball" and \
                data['env'].agents[agent].carrying.color == "purple":
            #print("picked up purple ball")
            return "C"
        else:
            return "I"
    else:
        return "I"

def drop_purple_ball(data, agent):
    if data['env'].agents[agent].carrying:
        return "C"
    else:
        fwd_pos = data['env'].agents[agent].front_pos
        fwd_cell = data['env'].grid.get(*fwd_pos)
        if fwd_cell is not None:
            if fwd_cell.type == "box":
                return "D"
            else:
                return "F"
        else:
            return "I"

def pickup_purple_ball_rm(data, agent):
    if data['word'] == "purple_ball":
        return "C"
    else:
        return "I"

def drop_purple_ball_rm(data, agent):
    if data['word'] == "purple_ball":
        return "C"
    elif data['word'] == "not_box":
        return "F"
    elif data['word'] == "box":
        return "D"
    else:
        return "I"

def finished(a, b):
    return "D"

def finished_goal(a, b):
    return "G"

def finished_door(a, b):
    return "U"

def finished_switch(a, b):
    return "S"

def fail(a, b):
    return "F"

def gotogoal(data, agent):
    env = data['env'].unwrapped
    agent_pos = env.agents[agent].cur_pos
    current_grid_obj = env.grid.get(*agent_pos)
    if current_grid_obj is not None:
        if current_grid_obj.type == "goal":
            return "G"
        else:
            return "I"
    else:
        return "I"

def gotogoal_rm(data, agent):
    if data['word'] == "goal":
        return "G"
    else:
        return "I"

def pickup_yellow_ball(data, agent):
    env = data['env'].unwrapped
    if env.agents[agent].carrying:
        if env.agents[agent].carrying.type == "ball" \
                and env.agents[agent].carrying.color == "yellow":
            return "C"
        else:
            return "I"
    else:
        return "I"

def pickup_yellow_ball_rm(data, agent):
    if data['word'] == "yellow_key":
        return "C"
    else:
        return "I"

def drop_yellow_ball(data, agent):
    env = data['env'].unwrapped
    if env.agents[agent].carrying:
        if env.agents[agent].carrying.type == "ball":
            return "C"
        else:
            fwd_pos = env.agents[agent].front_pos
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

def drop_yellow_ball_rm(data, agent):
    if data['word'] == "yellow_ball":
        return "C"
    elif data['word'] == "not_box":
        return "F"
    else:
        return "D"

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

def pickup_red_key(data, agent):
    env = data['env'].unwrapped
    if env.agents[agent].carrying:
        if env.agents[agent].carrying.type == "key" and \
            env.agents[agent].carrying.color == "red":
            return "C"
        else:
            return "I"
    else:
        return "I"

def pickup_red_key_rm(data, agent):
    if data['word'] == "red_key":
        return "C"
    else:
        return "I"

def goto_door_dist1_red(data, agent):
    env = data['env'].unwrapped
    agent_pos = env.agents[agent].cur_pos
    dist = manhatten_dist(red_door_pos, agent_pos)
    if env.agents[agent].carrying:
        if dist < 5:
            return "C1"
        else:
            return "C"
    else:
        return "I"

def goto_door_dist1_blue(data, agent):
    env = data['env'].unwrapped
    agent_pos = env.agents[agent].cur_pos
    dist = manhatten_dist(blue_door_poss, agent_pos)
    if env.agents[agent].carrying:
        if dist < 5:
            return "C1"
        else:
            return "C"
    else:
        return "I"

def goto_door_dist1_rm(data, agent):
    if data['word'] == "key_less4":
        return "C1"
    elif data['word'] == "key_greater4":
        return "C"
    else:
        return "I"

def goto_door_dist2_red(data, agent):
    env = data['env'].unwrapped
    agent_pos = env.agents[agent].cur_pos
    dist = manhatten_dist(red_door_pos, agent_pos)
    if env.agents[agent].carrying:
        if dist < 5:
            return "C2"
        else:
            return "C1"
    else:
        return "I"

def goto_door_dist2_blue(data, agent):
    env = data['env'].unwrapped
    agent_pos = env.agents[agent].cur_pos
    dist = manhatten_dist(blue_door_poss, agent_pos)
    if env.agents[agent].carrying:
        if dist < 5:
            return "C2"
        else:
            return "C1"
    else:
        return "I"

def goto_door_dist2_rm(data, agent):
    if data['word'] == "key_less3":
        return "C2"
    elif data['word'] == "key_greater3":
        return "C1"
    else:
        return "I"

def goto_door_dist3_red(data, agent):
    env = data['env'].unwrapped
    agent_pos = env.agents[agent].cur_pos
    dist = manhatten_dist(red_door_pos, agent_pos)
    if env.agents[agent].carrying:
        if dist < 2:
            return "C3"
        else:
            return "C2"
    else:
        return "I"

def goto_door_dist3_blue(data, agent):
    env = data['env'].unwrapped
    agent_pos = env.agents[agent].cur_pos
    dist = manhatten_dist(blue_door_poss, agent_pos)
    if env.agents[agent].carrying:
        if dist < 2:
            return "C3"
        else:
            return "C2"
    else:
        return "I"

def goto_door_dist3_rm(data, agent):
    if data['word'] == "key_less2":
        return "C3"
    elif data['word'] == "key_greater2":
        return "C2"
    else:
        return "I"

def unlock_door(data, agent):
    env = data['env'].unwrapped
    front_pos = env.agents[agent].front_pos
    fwd_cell = env.grid.get(*front_pos)
    if env.agents[agent].carrying:
        if env.agents[agent].carrying.type == "key":
            if fwd_cell is not None:
                if fwd_cell.type == "door" and fwd_cell.is_open and data['action'] == 3:
                    return "U"
                else:
                    return "C3"
            else:
                return "C3"
        else:
            return "F"
    else:
        return "F"

def unlock_door_rm(data, agent):
    if data['word'] == "is_open":
        return "U"
    elif data['word'] == "no_key":
        return "F"
    else:
        return "C3"

def turn_on_red_switch(data, agent):
    env = data['env'].unwrapped
    front_pos = env.agents[agent].front_pos
    fwd_cell = env.grid.get(*front_pos)
    if fwd_cell is not None:
        if fwd_cell.type == "switch" and fwd_cell.color == "red" and \
                fwd_cell.is_on and data['action'] == 3:
            return "S"
        else:
            return "U"
    else:
        return "U"

def turn_on_red_switch_rm(data, agent):
    if data['word'] == "red_on":
        return "S"
    else:
        return "U"

def turn_on_blue_switch(data, agent):
    env = data['env'].unwrapped
    front_pos = env.agents[agent].front_pos
    fwd_cell = env.grid.get(*front_pos)
    if fwd_cell is not None:
        if fwd_cell.type == "switch" and fwd_cell.color == "blue" and \
                fwd_cell.is_on and data['action'] == 3:
            return "S"
        else:
            return "I"
    else:
        return "I"

def turn_on_blue_switch_rm(data, agent):
    if data['word'] == "blue_on":
        return "S"
    else:
        return "I"

# Task 1: find a blue key to open a blue door and toggle a switch
def make_key_to_switch_dfa():
    dfa = DFA(start_state="I", acc=["U"], rej=["F"])
    states = KeyDoorSwitch()
    dfa.add_state(states.init, pickup_blue_key)
    dfa.add_state(states.carrying, goto_door_dist1_blue)
    dfa.add_state(states.dist1, goto_door_dist2_blue)
    dfa.add_state(states.dist2, goto_door_dist3_blue)
    dfa.add_state(states.dist3, unlock_door)
    #dfa.add_state(states.unlock, turn_on_red_switch)
    dfa.add_state(states.unlock, finished_door)
    dfa.add_state(states.fail, fail)
    return dfa

def make_key_to_switch_dfa_rm():
    dfa = RewardMachine(start_state="I", acc=["U"], rej=["F"],
                        words=["blue_key",
                               "is_open",
                               "no_key",
                               "key_not_open",
                               #"red_on",
                               #"red_off",
                               "key_less4",
                               "key_greater4",
                               "key_less3",
                               "key_greater3",
                               "key_less2",
                               "key_greater2"])
    states = KeyDoorSwitch()
    dfa.add_state(states.init, pickup_blue_key_rm)
    dfa.add_state(states.carrying, goto_door_dist1_rm)
    dfa.add_state(states.dist1, goto_door_dist2_rm)
    dfa.add_state(states.dist2, goto_door_dist3_rm)
    dfa.add_state(states.dist3, unlock_door_rm)
    #dfa.add_state(states.unlock, turn_on_red_switch_rm)
    dfa.add_state(states.unlock, finished_door)
    dfa.add_state(states.fail, fail)
    return dfa

# Task 2: activate a blue switch
def make_activate_blue_switch():
    dfa = DFA(start_state="I", acc=["S"], rej=[])
    states = ActivateSwitch()
    dfa.add_state(states.init, turn_on_blue_switch)
    dfa.add_state(states.switch, finished_switch)
    return dfa


# Task 3: pickup a purple ball and then place it in a box
def make_pickup_purple_ball_dfa():
    dfa = DFA(start_state="I", acc=["D"], rej=["F"])
    states = PickupObj()
    dfa.add_state(states.init, pickup_purple_ball)
    dfa.add_state(states.carrying, drop_purple_ball)
    dfa.add_state(states.drop, finished)
    dfa.add_state(states.fail, fail)
    return dfa

def make_pickup_purple_ball_rm():
    dfa = RewardMachine(start_state="I", acc=["D"], rej=["F"], words=["purple_ball", "not_box", "box", ""])
    states = PickupObj()
    dfa.add_state(states.init, pickup_purple_ball_rm)
    dfa.add_state(states.carrying, drop_purple_ball_rm)
    dfa.add_state(states.drop, finished)
    dfa.add_state(states.fail, fail)
    return dfa

# Task 4: pickup a yellow ball and place it in a box
def make_pickup_yellow_ball_dfa():
    dfa = DFA(start_state="I", acc=["S"], rej=["F"])
    states = PickupObj()
    dfa.add_state(states.init, pickup_yellow_ball)
    dfa.add_state(states.carrying, pickup_yellow_ball)
    dfa.add_state(states.drop, finished)
    dfa.add_state(states.fail, fail)
    return dfa

# Task 5: Find and go to the green square
def make_goto_goal_dfa():
    dfa = DFA(start_state="I", acc=["S"], rej=[])
    states = Goal()
    dfa.add_state(states.init, gotogoal)
    dfa.add_state(states.goal, finished_goal)
    return dfa

# Task 6: Pick up a red key and unlock a red door
def make_redkey_door_dfa():
    dfa = DFA(start_state="I", acc=["U"], rej=["F"])
    states = KeyDoorSwitch()
    dfa.add_state(states.init, pickup_red_key)
    dfa.add_state(states.carrying, goto_door_dist1)
    dfa.add_state(states.dist1, goto_door_dist2)
    dfa.add_state(states.dist2, goto_door_dist3)
    dfa.add_state(states.dist3, unlock_door_rm)
    dfa.add_state(states.unlock, finished_door)
    dfa.add_state(states.fail, fail)
    return dfa

def make_redkey_door_rm():
    dfa = RewardMachine(start_state="I", acc=["U"], rej=["F"], words=[
        "red_key",
        "is_open",
        "no_key",
        "key_less4",
        "key_greater4",
        "key_less3",
        "key_greater3",
        "key_less2",
        "key_greater2"])
    states = KeyDoorSwitch()
    dfa.add_state(states.init, pickup_red_key_rm)
    dfa.add_state(states.carrying, goto_door_dist1_rm)
    dfa.add_state(states.dist1, goto_door_dist2_rm)
    dfa.add_state(states.dist2, goto_door_dist3_rm)
    dfa.add_state(states.dist3, unlock_door_rm)
    dfa.add_state(states.unlock, finished_door)
    dfa.add_state(states.fail, fail)
    return dfa

task1 = make_key_to_switch_dfa()
task2 = make_activate_blue_switch()
task3 = make_pickup_purple_ball_dfa()
task4 = make_pickup_yellow_ball_dfa()
task5 = make_goto_goal_dfa()
task6 = make_redkey_door_dfa()

## Reward machines
task1_rm = make_key_to_switch_dfa_rm()
task3_rm = make_pickup_purple_ball_rm()
task6_rm = make_redkey_door_rm()


#############################################################################
#  Construct Environments
#############################################################################
envs = []
for i in range(num_procs):
    eseed = seed
    envs.append(make_env(
        env_key=env_key,
        max_steps_per_episode=max_epsiode_steps,
        apply_flat_wrapper=False))
#############################################################################
#  Initialise data structures
#############################################################################
xdfa = CrossProductDFA(
        num_tasks=num_tasks,
        dfas=[copy.deepcopy(obj) for obj in [task1, task6]],  # , task2, task3, task4, task5, task6
        agent=0)

def f(xdfa: CrossProductDFA, agent):
    xdfa.agent = agent
    return xdfa

reward_machine = RewardMachines(
    dfas=[copy.deepcopy(obj) for obj in [task1_rm, task6_rm]],
    one_off_reward=1.0,
    num_tasks=num_tasks
)
reward_machine.compute_state_space()
v = reward_machine.value_iteration(0.9)
for q, v_ in zip(reward_machine.state_space, v):
    print(q, v_)
Phi = -1. * v
xdfa.assign_shaped_rewards(Phi)
xdfa.assign_reward_machine_mappings(reward_machine.state_space, reward_machine.statespace_mapping)

xdfas = [[f(copy.deepcopy(xdfa), agent) for agent in range(num_agents)] for _ in range(num_procs)]

agent = MTARL(envs, num_tasks=num_tasks, num_agents=num_agents, xdfas=xdfas,
              one_off_reward=one_off_reward,
              e=e, c=c, chi=chi, lam=lam, gamma=1.0, lr=alpha1, lr2=alpha2, seed=seed,
              num_procs=num_procs, num_frames_per_proc=max_steps_per_update,
              recurrence=recurrence, max_eps_steps=max_epsiode_steps, env_key=env_key,
              normalisation_coef=normalisation_coeff, reward_machine=True)
i_s_shape = agent.tf_reset2().shape[-1]
models = [DeepActorCritic(envs[0].action_space.n, 64, num_tasks, name=f"agent{i}", activation="tanh", feature_set=i_s_shape) for i in range(num_agents)]

data_writer = AsyncWriter(
    fname_learning='data-maze-ma-learning',
    fname_alloc='data-maze-ma-alloc',
    num_agents=num_agents,
    num_tasks=num_tasks)


#############################################################################
## TRAIN AGENT SCRIPT
##############################################################################
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


