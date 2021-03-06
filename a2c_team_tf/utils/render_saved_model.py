import os
import tensorflow as tf
from a2c_team_tf.lib.experimental.tf2_a2c_base import MTARL
import numpy as np
from a2c_team_tf.utils.dfa import DFAStates, DFA, CrossProductDFA
from abc import ABC
from a2c_team_tf.envs.team_grid_mult import TestEnv
import copy
import click

# TODO this script is very clunky, it needs to know the DFAs used, the library used,
#  the saved tensorflow models, the openAI-gym environments, data filenames etc
#  - make selecting all of these options easier

# path
path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'saved_models'))

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

seed = 123
env_key = 'Team-obj-5x5-v0'
num_tasks = 2
num_agents = 2
np.random.seed(seed)
tf.random.set_seed(seed)
models = [tf.saved_model.load(f"{path}/agent{i}_lstm_4x4_ma") for i in range(num_agents)]
e, c, mu, chi, lam = 0.8, 0.85, 1.0, 1.0, 1.0
ball = make_pickupanddrop_ball_dfa()
key = make_pickup_key_dfa()
num_procs = 4
xdfas = [[
    CrossProductDFA(
        num_tasks=num_tasks,
        dfas=[copy.deepcopy(obj) for obj in [key, ball]],
        agent=agent) for agent in range(num_agents)] for _ in range(num_procs)]
agent = MTARL(envs=[], models=models, num_agents=num_agents, num_tasks=num_tasks, xdfas=xdfas,
              one_off_reward=1.0,
              e=e, c=c, chi=chi, lam=lam, env_key=env_key, seed=seed, recurrence=4, flatten_env=False)

@click.command()
@click.option('--render/--no-render', default=True)
@click.option('--plot/--no-plot', default=True)
@click.option('--frames', type=int, default=10)
@click.option('--fname', default=None)
def render(render, frames, plot, fname):
    if render:
        max_steps = 50
        for _ in range(frames):
            r_init_state = agent.render_reset()
            r_init_state = tf.expand_dims(tf.expand_dims(r_init_state, 1), 2)
            agent.render_episode(r_init_state, max_steps, *models)

if __name__ == '__main__':
    render()




