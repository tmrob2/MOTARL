import os.path
import pandas as pd
import tensorflow as tf
from a2c_team_tf.lib.tf2_a2c_base import MORLTAP
import numpy as np
from a2c_team_tf.utils.dfa import DFAStates, DFA, CrossProductDFA
from abc import ABC
from a2c_team_tf.envs.minigrid_fetch_mult import MultObjNoGoal
import copy
import click
import matplotlib.pyplot as plt

# path
models = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))

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

seed = 123
env_key = 'Mult-obj-4x4-v0'
num_tasks = 2
np.random.seed(seed)
tf.random.set_seed(seed)
model = tf.saved_model.load('/home/tmrob2/PycharmProjects/MORLTAP/saved_models/agent_lstm_4x4')
e, c, mu, chi, lam = 0.8, 0.85, 1.0, 1.0, 1.0
ball = make_pickup_ball_dfa()
key = make_pickup_key_dfa()
xdfa = CrossProductDFA(num_tasks=num_tasks, dfas=[copy.deepcopy(obj) for obj in [key, ball]], agent=0)
agent = MORLTAP(envs=[], model=model, num_tasks=num_tasks, xdfa=xdfa, one_off_reward=1.0,
                e=e, c=c, mu=mu, chi=chi, lam=lam, env_key=env_key, seed=seed, recurrence=4)


@click.command()
@click.option('--render/--no-render', default=True)
@click.option('--plot/--no-plot', default=True)
@click.option('--frames', type=int, default=10)
def render(render, frames, plot):
    if render:
        max_steps = 50
        for episode in range(frames):
            initial_state = agent.render_reset()
            agent.render_episode(initial_state, max_steps)
            if agent.renv.window.closed:
                break
    if plot:
        data = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
        df = pd.read_csv(f'{data}/data-4x4-lstm.csv', delimiter=' ')
        # plt.style.use('_mpl-gallery')
        x = np.arange(0, df.shape[0], 1)
        y1 = df.iloc[:, 0].to_numpy()
        y2 = df.iloc[:, 1].to_numpy()
        y3 = df.iloc[:, 2].to_numpy()
        fig, ax = plt.subplots()
        ax.plot(x, y2, x, y3, x, y1, linewidth=0.5)
        ax.set(ylim=(0, 1.0), yticks=np.arange(0, 1.1, 0.1))
        plt.show()

if __name__ == '__main__':
    render()




