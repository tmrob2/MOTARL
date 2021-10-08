from abc import ABC

import gym
import unittest

import numpy as np

from model import ActorCritic
import tensorflow as tf
import motaplib
from gym_minigrid.wrappers import *
from gym_minigrid.minigrid import *
from gym.envs.registration import register
from dfa import DFA, DFAStates
from environment import convert_to_flat_and_full, OneObjRoom
from gym_minigrid.minigrid import OBJECT_TO_IDX

seed = 103
# tf.random.set_seed(seed)
# np.random.seed(seed)
register(
    id="empty-room-5x5-v0",
    entry_point='environment:EmptyRoom5x5',
    max_episode_steps=2000
)

register(
    id="obj1-room-5x5-v0",
    entry_point='environment:OneObjRoom',
    max_episode_steps=2000
)
env1 = gym.make('obj1-room-5x5-v0')
env1_ = convert_to_flat_and_full(env1)
env2 = gym.make('obj1-room-5x5-v0')
env2_ = convert_to_flat_and_full(env2)
envs = [env1_, env2_]


class MoveKeyStates(DFAStates, ABC):
    def __init__(self):
        self.init = "I"
        self.carrying = "C"
        self.deliver = "D"
        self.fail = "F"


# make a task (DFA), simple (no colours), but colours can be used
def pickup_key(env: OneObjRoom):
    """If the agent is not carrying a key, then picks up a key move to the carrying state"""
    if env is not None:
        next_state = "C" if isinstance(env.carrying, Key) else "I"
    else:
        next_state = "I"
    return next_state


def carrying(env: OneObjRoom):
    """If the agent is carrying a key then an agent must continue to carry the key, unless it is
    at the drop off coordinate"""
    next_state = ""
    if env.carrying is None:
        if np.array_equal(env.agent_pos, np.ndarray([1, 1])):
            return "D"
        else:
            return "F"
    else:
        return "C"


def deliver(_):
    return "D"


def fail(_):
    return "F"


def make_key_dfa():
    dfa = DFA(start_state="I", acc=["D"], rej=["F"])
    dfa.states = MoveKeyStates()
    dfa.add_state(dfa.states.init, pickup_key)
    dfa.add_state(dfa.states.carrying, carrying)
    dfa.add_state(dfa.states.deliver, deliver)
    dfa.add_state(dfa.states.fail, fail)
    dfa.start()
    return dfa


# Parameters
step_rew0 = 10
task_prob0 = 0.8
NUMAGENTS, TASKS, GRIDSIZE, N_OBJS = 2, 1, (5, 5), 3
max_steps_per_episode = 1000
gamma = 1.0
mu = 1.0 / NUMAGENTS  # fixed even probability of allocating each task to each agent
lam = 1.0
chi = 1.0
c = step_rew0
e = task_prob0  # task reward threshold
num_actions = env1.action_space.n
num_hidden_units = 128
models = [ActorCritic(num_actions, num_hidden_units, TASKS, name="AC{}".format(i)) for i in range(NUMAGENTS)]
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


class TestModelMethods(unittest.TestCase):
    # implement a test for converting training data to correct tensor shapes
    # todo implement a test for an instance of calculating a loss function
    # todo test that there is no flow between A and Advantage grad_theta π(a_t|s_t)*(G - A)
    # two environment step test, and extract v_pi,ini
    # todo test tf function implementation of KL divergence function
    def test_environment_init(self):
        assert num_actions == 7
        # env = FullyObsWrapper(env1)
        # env_fobs = FlatObsWrapper(env)
        state = env1_.reset()
        print("state", state)
        # envs = [env1, env2]
        # envs = [FlatObsWrapper(e) for e in envs]
        #
        return True

    def test_dfa(self):
        assert make_key_dfa().handlers['C'] == carrying

    def test_action_sample_from_model(self):
        # randomly select an initial action
        state = env1_.reset()
        dfas = [make_key_dfa()]
        motap = motaplib.TfObsEnv(envs, models, dfas)
        # select a random action from the model
        done = False
        while not done:
            env1_.render('human')
            state = tf.expand_dims(state, 0)
            action_logits_t, values = models[0](state)
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            state, reward, done = motap.tf_env_step(action, 0)
            # print("state", state['image'])
        return True

    def _test_rendering_multi_env(self):
        pass

    def _test_episode(self):
        num_models = len(models)
        motap = motaplib.TfObsEnv(envs, models)
        for i in range(num_models):
            i_state, _, _ = envs[i].reset()
            print(i_state)
            initial_state = tf.constant([i_state['image'], i_state['direction']], dtype=tf.int32)
            action_probs, values, rewards = motap.run_episode(initial_state, i, max_steps_per_episode)

            print("action probs: \n{}".format(action_probs))
            print("values: \n{}".format(values))
            print("rewards: \{}".format(rewards))
        return True

    def test_compute_h(self):
        pass

    def test_compute_f(self):
        pass

    def _expected_rewards(self):
        """function to test what the output of the expected rewards for an episode is"""
        num_models = len(models)
        motap = motaplib.TfObsEnv(envs, models)
        for i in range(num_models):
            initial_state = tf.constant(envs[0].reset(), dtype=tf.float32)
            action_probs, values, rewards = motap.run_episode(initial_state, i, max_steps_per_episode)
            returns = motap.get_expected_returns(rewards, gamma, TASKS, False)
            print("values: \n".format(values))
            print("returns: \n{}".format(returns))
        return True

    def _compute_loss(self):
        return True

    def test_train_step(self):
        # envs = [env1, env2]
        # motap = motaplib.TfObsEnv(envs, models)
        # mean = tf.Variable(0.0)
        # with tqdm.trange(100) as t:
        #    for _ in t:
        #        motap.train_step(optimizer, gamma, max_steps_per_episode, TASKS, lam, chi, mu, e)
        #        tf.print(motap.mean.numpy())
        return True

    def _train(self):
        # todo remove the reference to environments from a list
        # envs = [env1, env2]
        # motap = motaplib.TfObsEnv(envs, models)
        # max_episodes = 100
        # motap.train(max_episodes, optimizer, gamma, max_steps_per_episode, TASKS, lam, chi, mu, e)
        pass


if __name__ == '__main__':
    unittest.main()
