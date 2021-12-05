# todo complete the cartpole unit tests, to check if the implementation achieves the
#  same results as GS.
import collections
from a2c_team_tf.utils.dfa import *
from abc import ABC
from a2c_team_tf.nets.base import ActorCritic
from a2c_team_tf.lib import lib_mult_env
from a2c_team_tf.lib.lib_mult_env import LossObjective
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm
import unittest
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple

# Create the environment
env1 = gym.make("CartPole-v0")
env2 = gym.make("CartPole-v0")

# Set seed for experiment reproducibility
seed = 42
env1.seed(seed)
env2.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

step_rew0 = 10  # step reward threshold
cart_pos = 0.5
num_tasks = 1
num_agents = 2
num_actions = env1.action_space.n
num_hidden_units = 128
models = [
    ActorCritic(num_actions, num_hidden_units, num_tasks, name="AC{}".format(i))
    for i in range(num_agents)
]


class MoveToPos(DFAStates, ABC):
    def __init__(self):
        self.init = "I"
        self.finish = "F"


def get_reached(env: gym.Env):
    if env is not None:
        if env.env.state[0] > cart_pos:
            return "F"
        else:
            return "I"
    else:
        return "I"


def finished(data):
    return "F"


def make_move_to_pos_dfa():
    dfa = DFA(start_state="I", acc=["F"], rej=[])
    dfa.states = MoveToPos()
    dfa.add_state(dfa.states.init, get_reached)
    dfa.add_state(dfa.states.finish, finished)
    dfa.start()
    return dfa

# Parameters
task_prob0 = 0.8
N_AGENTS, N_TASKS = num_agents, num_tasks
gamma = 1.0
mu = 1.0 / N_AGENTS  # fixed even probability of allocating each task to each agent
lam = 1.0
chi = 1.0
c = step_rew0
e = task_prob0  # task reward threshold
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
envs = [env1, env2]
render_env = False
print_rewards = False
max_steps_per_episode = 100


class TestModelMethods(unittest.TestCase):
    def test_run_episide(self):
        print()
        print("-----------------------------------")
        print("           testing episode         ")
        print("-----------------------------------")
        # Set a single environment
        num_models = 1

        # Construct a set of tasks for the agent to complete
        task1 = make_move_to_pos_dfa()
        prod_dfa = CrossProductDFA(num_tasks=N_TASKS, dfas=[task1])
        dfas = [prod_dfa] * num_models

        # Set the configuration of the test

        # Construct the MOTAP environment
        motap = motaplib.TfObsEnv(envs, models, dfas, N_TASKS, N_AGENTS, render_env, print_rewards)

        # Set the initial state of the environment
        i_state = motap.envs[0].reset()
        action_probs, values, rewards = motap.run_episode(i_state, 0, max_steps_per_episode)

        # Print the returns for the episode
        # print("values: \n{}".format(values))
        # print("rewards: \n{}".format(rewards))
        return True

    def test_expected_returns(self):
        """function to test the expected returns for an episode, an important step before
        calculating the loss"""
        print()
        print("-----------------------------------")
        print("           testing returns         ")
        print("-----------------------------------")
        num_models = 2
        # Construct a set of tasks for the agent to complete
        task1 = make_move_to_pos_dfa()
        prod_dfa = CrossProductDFA(num_tasks=N_TASKS, dfas=[task1])
        dfas = [prod_dfa] * num_models

        # Construct the MOTAP environment
        motap = motaplib.TfObsEnv(envs, models, dfas, N_TASKS, N_AGENTS, render_env, print_rewards)

        for i in range(num_models):
            initial_state = tf.constant(envs[i].reset(), dtype=tf.float32)
            action_probs, values, rewards = motap.run_episode(initial_state, i, max_steps_per_episode)
            returns = motap.get_expected_returns(rewards, gamma, N_TASKS, False)
            #print("values: \n".format(values))
            print("returns: \n{}".format(returns[0]))
        return True

    def test_compute_loss(self):
        print()
        print("-----------------------------------")
        print("           testing loss            ")
        print("-----------------------------------")
        num_models = 2
        # Construct a set of tasks for the agent to complete
        task1 = make_move_to_pos_dfa()
        prod_dfa = CrossProductDFA(num_tasks=N_TASKS, dfas=[task1])
        dfas = [prod_dfa] * num_models

        # Set the configuration of the test

        # Construct the MOTAP environment
        motap = motaplib.TfObsEnv(envs, models, dfas, N_TASKS, N_AGENTS, render_env, print_rewards)

        # storage per model - the issue with using lists is that the don't perform the way they
        #                     are expected to in tf.function. We should look at some optimisations when the algo
        #                     is working as expected
        action_probs_l = []
        values_l = []
        rewards_l = []
        returns_l = []
        loss_l = []  # the loss storage for an agent

        for i in range(num_models):
            initial_state = tf.constant(envs[i].reset(), dtype=tf.float32)
            action_probs, values, rewards = motap.run_episode(initial_state, i, max_steps_per_episode)
            returns = motap.get_expected_returns(rewards, gamma, N_TASKS, False)
            print("returns: \n{}".format(returns[0]))
            print(f"Agent: {i} returns shape: {returns.shape}")
            print(f"Agent: {i}: values shape: {values.shape}")
            # Append tensors to respective lists
            action_probs_l.append(action_probs)
            values_l.append(values)
            rewards_l.append(rewards)
            returns_l.append(returns)

        ini_values = tf.convert_to_tensor([x[0, :] for x in values_l])

        for i in range(num_models):
            # get loss
            values = values_l[i]
            returns = returns_l[i]
            ini_values_i = ini_values[i]
            loss = motap.compute_loss(action_probs_l[i], values, returns, ini_values, ini_values_i, lam, chi, mu, e, c)
            loss_l.append(loss)
        print(f"loss: {loss_l}")
        return True

    def test_train_step(self):
        print()
        print("-----------------------------------")
        print("           testing train step      ")
        print("-----------------------------------")
        num_models = 2
        # Construct a set of tasks for the agent to complete
        task1 = make_move_to_pos_dfa()
        prod_dfa = CrossProductDFA(num_tasks=N_TASKS, dfas=[task1])
        dfas = [prod_dfa] * num_models

        # Set the configuration of the test

        # Construct the MOTAP environment
        motap = motaplib.TfObsEnv(envs, models, dfas, N_TASKS, N_AGENTS, render_env, print_rewards)
        motap.train_step(optimizer, gamma, max_steps_per_episode, N_TASKS, lam, chi, mu, e, c)
        return True

    def test_train(self):
        print()
        print("-----------------------------------")
        print("           testing train           ")
        print(""" 
                trains a 2 x 2 environment with a 
                key and a ball, there a two agents and 
                two tasks. An agents mission is to pick 
                up both the key and the ball        """)
        print("-----------------------------------")
        num_models = 2
        # Construct a set of tasks for the agent to complete
        task1 = make_move_to_pos_dfa()
        prod_dfa = CrossProductDFA(num_tasks=N_TASKS, dfas=[task1])
        dfas = [prod_dfa] * num_models

        # Set the configuration of the test
        max_episodes = 1000
        min_episodes_criterion = 10
        reward_threshold = 195

        # Construct the MOTAP environment
        motap = motaplib.TfObsEnv(envs, models, dfas, N_TASKS, N_AGENTS, render_env, print_rewards, LossObjective.MAXIMISE)
        # Keep last episodes reward
        episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

        with tqdm.trange(max_episodes) as t:
            for tt in t:
                episode_reward = int(
                    motap.train_step(optimizer, gamma, max_steps_per_episode, N_TASKS, lam, chi, mu, e, c))
                episodes_reward.append(episode_reward)
                running_reward = statistics.mean(episodes_reward)
                t.set_description(f"Episode {tt}")
                t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)
                # Show average episode reward every 10 episodes
                # if tt % 10 == 0:
                #    print(f'Episode {tt}: average reward: {running_reward}')

                if running_reward > reward_threshold and tt >= min_episodes_criterion:
                    break
            print(f'\nSolved at episode {tt}: average reward: {running_reward:.2f}!')


if __name__ == '__main__':
    unittest.main()
