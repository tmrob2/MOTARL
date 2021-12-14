import gym
import tensorflow as tf
from a2c_team_tf.nets.base import Actor, Critic
from a2c_team_tf.lib.tf2_a2c_base import Agent
import numpy as np
from a2c_team_tf.utils import obs_wrapper
from a2c_team_tf.utils.dfa import DFAStates, DFA, CrossProductDFA
from abc import ABC
from a2c_team_tf.envs.minigrid_empty_mult import EmptyMultiTask
from a2c_team_tf.envs.minigrid_fetch_mult import MultObjNoGoal
import copy

# Load saved models
class PickupObj(DFAStates, ABC):
    def __init__(self):
        self.init = "I"
        self.carrying = "C"

def pickup_ball(env: MultObjNoGoal, _):
    if env.carrying:
        if env.carrying.type == "ball":
            return "C"
        else:
            return "I"
    else:
        return "I"

def pickup_key(env: MultObjNoGoal, _):
    if env.carrying:
        if env.carrying.type == "key":
            return "C"
        else:
            return "I"
    else:
        return "I"

def finished(a, b):
    return "C"

def make_pickup_ball_dfa():
    dfa = DFA(start_state="I", acc=["C"], rej=[])
    states = PickupObj()
    dfa.add_state(states.init, pickup_ball)
    dfa.add_state(states.carrying, finished)
    return dfa

def make_pickup_key_dfa():
    dfa = DFA(start_state="I", acc=["C"], rej=[])
    states = PickupObj()
    dfa.add_state(states.init, pickup_key)
    dfa.add_state(states.carrying, finished)
    return dfa

env: gym.Env = gym.make('Mult-obj-4x4-v0')
seed = 44
num_tasks = 2
env.seed(seed)
env = obs_wrapper.FlatObsWrapper(env, 50)
np.random.seed(seed)
tf.random.set_seed(seed)
actor = tf.saved_model.load('/home/tmrob2/PycharmProjects/MORLTAP/saved_models/a_4x4_mult_obj_room')
critic = tf.saved_model.load('/home/tmrob2/PycharmProjects/MORLTAP/saved_models/c_4x4_mult_obj_room')
e, c, mu, chi, lam = 0.8, 0.85, 1.0, 1.0, 1.0
ball = make_pickup_ball_dfa()
key = make_pickup_key_dfa()
xdfa = CrossProductDFA(num_tasks=num_tasks, dfas=[copy.deepcopy(obj) for obj in [key, ball]], agent=0)
agent = Agent(env, actor, critic, num_tasks=num_tasks, xdfa=xdfa, one_off_reward=1.0,
              e=e, c=c, mu=mu, chi=chi, lam=lam)
agent.env.render('human')
max_steps = 50
for episode in range(1):
    initial_state = agent.tf_reset()
    agent.render_episode(initial_state, max_steps)
    if agent.env.window.closed:
        break
