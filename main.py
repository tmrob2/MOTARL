

# todo convert this file to unit_tests.py, this will form the template for the implementation
#  of main.py
import random
from abc import ABC

from utils import Coord, Action1
import numpy as np
from grid import Grid
from environment import Environment
import visualisation
import tensorflow as tf
import statistics
import tqdm
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple
from model import ActorCritic
import unittest


seed = 103
tf.random.set_seed(seed)
np.random.seed(seed)
eps = np.finfo(np.float32).eps.item()
tasks, grid_size, n_objs = 2, (10, 10), 3


class Env1(Environment, ABC):
    # todo convert the output of these into probability distributions
    # todo act should also return a word
    def act(self, action):
        x, y = self.position.x, self.position.y
        if action == Action1.FORWARD:
            self.position.set(x, y + 1)
        elif action == Action1.LEFT:
            self.position.set(x - 1, y)
        elif action == Action1.DOWN:
            self.position.set(x, y - 1)
        elif action == Action1.RIGHT:
            self.position.set(x + 1, y)
        self.energy += self.STEP_VALUE


env = Env1(grid_size=grid_size, n_objs=n_objs, start_energy=1.0, num_tasks=tasks)


def env_step(actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns state, reward and done flag given an action
    Actions are an array because there are multiple agents
    Rewards (output) is an array because there are multiple rewards, agent costs, task rewards
    """
    # the return from env.step is an nd_array
    state, reward, done = env.step(actions)  # todo this argument must be a tuple of environments?
    return state.astype(np.float32), reward.astype(np.float32), np.array(done, np.int32)


def tf_env_step(actions: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(env_step, [actions], [tf.float32, tf.float32, tf.int32])


def get_expected_returns(
        rewards: tf.Tensor, standardise: bool = True) -> tf.Tensor:
    """Compute expected returns per timestep"""
    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    # start from the end of rewards and accumulate sums into the returns array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    sum_val = tf.constant(0.0)
    sum_shape = sum_val.shape
    for i in tf.range(n):
        reward = rewards[i]
        sum_val = reward + sum_val
        sum_val.set_shape(sum_shape)
        returns = returns.write(i, sum_val)
    returns = returns.stack()[::-1]
    if standardise:
        returns = ((returns - tf.math.reduce_mean(returns)) /
                   (tf.math.reduce_std(returns) + eps))
    return returns


def run_episode(
        initial_state: tf.Tensor,
        model: tf.keras.Model,
        max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Runs a single episode to collect training data"""

    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
        # Convert state into a batched tensor (batch size=1)
        state = tf.expand_dims(state, 0)
        # Run the model and to get action probabilities and critic value
        action_logits_t, value = model(state)
        # Sample the next action from the action probability distribution
        # Tensor values need to be restricted to the enabled actions at the current state
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)
        # Store critic values
        values = values.write(t, tf.squeeze(value))
        # Store log probability of action chosen
        action_probs = action_probs.write(t, action_probs_t[0, action])
        # Apply actions to the environment to get the next state and reward
        state, reward, done = tf_env_step(action)
        state.set_shape(initial_state_shape)
        # Store rewards
        rewards = rewards.write(t, reward)
        if tf.cast(done, tf.bool):
            break
    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()

    return action_probs, values, rewards


def compute_loss(action_probs: tf.Tensor, v_values: tf.Tensor, v_returns: tf.Tensor) -> tf.Tensor:
    advantages = v_returns - v_values
    action_log_probs = tf.math.log(action_probs)
    ini_values = make_ini_values(v_values)


def allocation_probs(allocation_logits: tf.Tensor) -> tf.Tensor:
    """µ(i,j) is the allocation probability for task j to agent i. This can be determined at
    t=0 because the allocation is fixed throughout the duration of the episode"""
    return tf.nn.softmax(allocation_logits)


def make_ini_values(v_values: tf.Tensor) -> tf.Tensor:
    """We construct this function to break the flow of tensors for values that are
    considered constant at time t, and not propagated"""
    return tf.convert_to_tensor(v_values[:, 0].numpy())


def compute_f(ini_values: tf.Tensor, c: np.float32) -> tf.Tensor:
    """Cost function to determine if an agent has exceeded it cost threshold and applies a
    corresponding penalty"""
    return tf.math.square(tf.math.maximum(0, ini_values - c))


def compute_h(ini_values: tf.Tensor, e: np.float32, mu_ij: tf.Tensor) -> tf.Tensor:
    """h is a divergence function which determines how far away the reward frequency is from the
    required task probability threshold
    Parameters:
    -----------
    ini_values: is a vector of critic from t=0 to t_max, this will actually be a tensor slice
    because we are only looking at the jth

    mu_ij: is the probability distribution of allocating task j across all agents

    intuitively this requires that all environments must be 'stepped' at each time step so that
    we can record all of the v_{pi,ini} values
    """
    assert e > 0.0, "The value of task frequency requirement must be greater than zero"
    kl = tf.matmul(ini_values, tf.math.log(ini_values) - tf.math.log(e)) + \
         tf.matmul((1.0 - ini_values), tf.math.log(1.0 - ini_values))
    tf.tensordot(mu_ij, kl)


class TestModelMethods(unittest.TestCase):
    # todo implement a test for converting training data to correct tensor shapes
    # todo implement a test for an instance of calculating a loss function
    # todo test that there is no flow between A and Advantage grad_theta π(a_t|s_t)*(G - A)
    # todo two environment step test, and extract v_pi,ini
    # todo test KL divergence function
    def test_environment(self):
        agents, tasks, grid_size, n_objs = 2, 2, (10, 10), 3
        env.reset()
        done = False
        for i in range(10):
            #print("Episode {}".format(i))
            while not done:
                action = random.choice(list(Action1))
                state, _, done = env.step(action=action)
                # print([s.__str__() for s in state])
            env.reset()
            done = False
        return True

    def test_tf_action_sampling(self):
        print("------------------------------------------\n")
        print("         Testing action selection           ")
        print("--------------------------------------------")
        model = ActorCritic(6, 2, 2, 128)
        initial_state = tf.constant(env.reset(), dtype=tf.float32)
        state = tf.expand_dims(initial_state, 0)
        # Run the model and to get action probabilities and critic value
        action_logits_t, allocator_logits_t, value = model(state)
        print("action_logits: {}".format(action_logits_t))
        print("Allocator logits: {}".format(allocator_logits_t))
        print("value: {}", value)
        # sample the next action from the action probability distribution
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        print("sampled action: {}".format(action))
        # get the next state,reward, done signal
        state, reward, done = tf_env_step(action)
        print("state: {}, reward: {}, done: {}".format(state, reward, done))
        print(model.summary())
        return True

    def test_episode(self):
        print("------------------------------------------\n")
        print("              Testing episode               ")
        print("--------------------------------------------")
        model = ActorCritic(6, 2, 2, 128)
        initial_state = tf.constant(env.reset(), dtype=tf.float32)
        initial_state_shape = initial_state.shape
        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        state = initial_state
        max_steps = 10
        for t in tf.range(max_steps):
            state = tf.expand_dims(state, 0)
            action_logits_t, _, value = model(state)
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            action_probs_t = tf.nn.softmax(action_logits_t)

            # Store critic Value (V)
            values = values.write(t, tf.squeeze(value))

            # store log prob of the action chosen
            action_probs = action_probs.write(t, action_probs_t[0, action])

            # apply action to the environment to get the next state and reward
            state, reward, done = tf_env_step(action)
            print("reward: {}".format(reward[0]))
            state.set_shape(initial_state_shape)

            #store rewards
            #rewards = rewards.write(t, reward)
            if tf.cast(done, tf.bool):
                break

        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()
        print("action, p: {}".format(action_probs))
        print("values, V: {}".format(values))
        print("rewards, R_T: {}".format(rewards))
        return True


    def test_tf_rewards(self):
        """function to test what the output of the expected rewards for an episode is"""
        # todo we require an instance of run episode, which will return rewards
        # todo input the rewards into the tf function expected rewards, corresponding to v_ini under
        #  some policy π
        return True


if __name__ == '__main__':
    unittest.main()
