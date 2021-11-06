from abc import ABC
import copy
import unittest
from a2c_team_tf.nets.base import ActorCritic
import tensorflow as tf
import tqdm
from a2c_team_tf.lib import motaplib
from gym_minigrid.minigrid import *
from gym.envs.registration import register
from a2c_team_tf.utils.dfa import DFA, DFAStates, CrossProductDFA
from a2c_team_tf.environments.minigrid_wrapper import convert_to_flat_and_full, ObjRoom
import collections
import statistics

max_steps_per_episode = 1000
seed = 103
render_env = False
print_rewards = False
# tf.random.set_seed(seed)
# np.random.seed(seed)
register(
    id="empty-room-5x5-v0",
    entry_point='a2c_team_tf.environments.minigrid_wrapper:EmptyRoom5x5',
    max_episode_steps=max_steps_per_episode
)

register(
    id="obj1-room-3x3-v0",
    entry_point='a2c_team_tf.environments.minigrid_wrapper:OneKeyRoom3x3',
    max_episode_steps=max_steps_per_episode
)

register(
    id="obj1-room-2x2-v0",
    entry_point='a2c_team_tf.environments.minigrid_wrapper:OneKeyRoom2x2',
    max_episode_steps=max_steps_per_episode
)
env1 = gym.make('obj1-room-2x2-v0')
env1_ = convert_to_flat_and_full(env1)
env2 = gym.make('obj1-room-2x2-v0')
env2_ = convert_to_flat_and_full(env2)
envs = [env1_, env2_]


# Very simple task to pick up a key
class PickupObjStates(DFAStates, ABC):
    def __init__(self):
        self.init = "I"
        self.carrying = "C"


class MoveKeyStates(DFAStates, ABC):
    def __init__(self):
        self.init = "I"
        self.carrying = "C"
        self.deliver = "D"
        self.fail = "F"


def pickup_key(env: ObjRoom):
    """If the agent is not carrying a key, then picks up a key"""
    if env is not None:
        next_state = "C" if isinstance(env.carrying, Key) else "I"
    else:
        next_state = "I"
    return next_state


def pickup_ball(env: ObjRoom):
    """If the agent is not carrying a ball, then it picks up a ball"""
    if env is not None:
        next_state = "C" if isinstance(env.carrying, Ball) else "I"
    else:
        next_state = "I"
    return next_state


def carrying(env: ObjRoom):
    """If the agent is carrying a key then an agent must continue to carry the key, unless it is
    at the drop off coordinate"""
    if env.carrying is None:
        if np.array_equal(env.agent_pos, np.ndarray([1, 1])):
            return "D"
    else:
        return "C"


def finish(_):
    return "C"


def deliver(_):
    return "D"


def fail(_):
    return "F"


def make_key_dfa():
    """Task: Pick up a key move it to 1,1, then go to the goal state"""
    dfa = DFA(start_state="I", acc=["D"], rej=["F"])
    dfa.states = MoveKeyStates()
    dfa.add_state(dfa.states.init, pickup_key)
    dfa.add_state(dfa.states.carrying, carrying)
    dfa.add_state(dfa.states.deliver, deliver)
    dfa.add_state(dfa.states.fail, fail)
    dfa.start()
    return dfa


def make_pickup_key_dfa():
    """Task pick up a key"""
    dfa = DFA(start_state="I", acc=["C"], rej=[])
    dfa.states = PickupObjStates()
    dfa.add_state(dfa.states.init, pickup_key)
    dfa.add_state(dfa.states.carrying, finish)
    dfa.start()
    return dfa


def make_pickup_ball_dfa():
    """Task: pick up a ball"""
    dfa = DFA(start_state="I", acc=["C"], rej=[])
    dfa.states = PickupObjStates()
    dfa.add_state(dfa.states.init, pickup_ball)
    dfa.add_state(dfa.states.carrying, finish)
    dfa.start()
    return dfa


# Parameters
step_rew0 = 10
task_prob0 = 0.8
N_AGENTS, N_TASKS, N_OBJS = 2, 2, 2
gamma = 1.0
mu = 1.0 / N_AGENTS  # fixed even probability of allocating each task to each agent
lam = 1.0
chi = 1.0
c = step_rew0
e = task_prob0  # task reward threshold
num_actions = env1.action_space.n
num_hidden_units = 128
models = [ActorCritic(num_actions, num_hidden_units, N_TASKS, name="AC{}".format(i)) for i in range(N_AGENTS)]
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

"""
Tests can be run as is, or can be modified to render environments, or print certain characteristics
of the test. 
"""


class TestModelMethods(unittest.TestCase):
    # implement a test for converting training data to correct tensor shapes
    # two environment step test, and extract v_pi,ini
    def test_env_goal(self):



if __name__ == '__main__':
    unittest.main()
