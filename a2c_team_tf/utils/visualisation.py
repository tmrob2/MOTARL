import gym
import tensorflow as tf
from a2c_team_tf.nets.base import Actor, Critic
from a2c_team_tf.lib.tf2_a2c_base import Agent
import numpy as np
from a2c_team_tf.utils import obs_wrapper
from a2c_team_tf.utils.dfa import DFAStates, DFA, CrossProductDFA
from abc import ABC
from a2c_team_tf.envs.minigrid_empty_mult import EmptyMultiTask
import copy

# Load saved models
class GetToGoal(DFAStates, ABC):
    def __init__(self):
        self.init = "I"
        self.goal = "G"

def at_goal_state(env: EmptyMultiTask, _):
    cell = env.grid.get(*env.agent_pos)
    if cell != None:
        next_state = "G" if cell.type == 'goal' else "I"
    else:
        next_state = "I"
    return next_state

def finished(a, b):
    return "G"

def make_dfa():
    dfa = DFA(start_state="I", acc=["G"], rej=[])
    states = GetToGoal()
    dfa.add_state(states.init, at_goal_state)
    dfa.add_state(states.goal, finished)
    return dfa

env: gym.Env = gym.make('Empty-multi-5x5-v0')
seed = 44
env.seed(seed)
env = obs_wrapper.FlatObsWrapper(env, 7)
np.random.seed(seed)
tf.random.set_seed(seed)
actor = tf.saved_model.load('/home/tmrob2/PycharmProjects/MORLTAP/saved_models/actor_mult_task_model')
critic = tf.saved_model.load('/home/tmrob2/PycharmProjects/MORLTAP/saved_models/critic_multi_task_model')
e, c, mu, chi, lam = 0.8, 0.85, 1.0, 1.0, 1.0
task = make_dfa()
xdfa = CrossProductDFA(num_tasks=1, dfas=[copy.deepcopy(task)], agent=0)
agent = Agent(env, actor, critic, num_tasks=1, xdfa=xdfa, one_off_reward=1.0,
              e=e, c=c, mu=mu, chi=chi, lam=lam)
agent.env.render('human')
max_steps = 50
for episode in range(10000):
    initial_state = agent.tf_reset()
    agent.render_episode(initial_state, max_steps)
    if agent.env.window.closed:
        break
