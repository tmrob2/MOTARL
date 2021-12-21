import gym
from gym.envs.registration import register
from a2c_team_tf.envs.team_grid_mult import TestEnv
import tensorflow as tf
from a2c_team_tf.lib.single_env_mult_agent import MAS
from a2c_team_tf.nets.base import ActorCritic
from a2c_team_tf.utils.dfa import DFA, DFAStates, CrossProductDFA
from abc import ABC
from teamgrid.minigrid import *
import copy
from enum import Enum
"""
The goal of this test is for two agents to learn to pick up a ball in a 3x3 grid 
and move it to a particular square. Even in this small simple environment the agents
have to deal with avoiding collisions, avoiding conflicting pickups, and learn how
to perform a complex task with sparse reward. 
"""

max_episode_steps = 100
max_episodes = 10000
min_episodes_criterion = 100
num_agents = 2
num_tasks = 1
one_off_reward = 10.0
step_rew0 = 10
gamma = 1.0
seed = 42
lam = 1.0
chi = 1.0
task_prob0 = 0.8
e = tf.constant([task_prob0 * one_off_reward], dtype=tf.float32)
c = tf.constant([step_rew0] * num_agents, dtype=tf.float32)
# Register the teamgrid environment
register(
    id="empty-room-5x5-v0",
    entry_point='a2c_team_tf.envs.team_grid_wrapper:TestEnv',
    max_episode_steps=max_episode_steps
)

class MoveObjStates(DFAStates, ABC):
    def __init__(self):
        self.init = "I"
        self.carrying = "C"

    def define_states(self):
        states = {}
        for (i, (_, v)) in enumerate(self.__dict__.items()):
            states[v] = i
        return states

def pickup_obj(env: TestEnv, agent: int):
    # print(f"running pickup object, agent carrying: {env.agents[agent].carrying}")
    next_state = "C" if isinstance(env.agents[agent].carrying, Ball) else "I"
    return next_state

def finished(*args):
    return "C"

def make_carry_obj_dfa():
    dfa = DFA(start_state="I", acc=["C"], rej=[])
    states = MoveObjStates()
    dfa.states = states.define_states()
    dfa.add_state(states.init, pickup_obj)
    dfa.add_state(states.carrying, finished)
    return dfa


if __name__ == '__main__':
    # instantiate the teamgrid environment
    env: gym.Env = gym.make('empty-room-5x5-v0')
    env.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    models = [ActorCritic(env.action_space.n, 128, 1, f'test{i}') for i in range(num_agents)]
    tf.print(f"actions: {env.action_space}")
    task1 = make_carry_obj_dfa()
    dfas = [CrossProductDFA(
        num_tasks=num_tasks,
        dfas=[copy.deepcopy(task1)],
        agent=idx) for idx in range(num_agents)]
    # mas = MAS(seed=seed, env=env, models=models, dfas=dfas, one_off_reward=one_off_reward,
    #                num_tasks=num_tasks, num_agents=num_agents, render=True, e=e, c=c, lr=5e-4, lr2=1e-4, direction=MAS.Direction.MINIMISE)
    # initial_state = mas.tf_reset()
    # mas.learn(max_episodes=max_episodes, max_steps_per_episode=max_episode_steps, min_episodes_criterion=min_episodes_criterion)

    

