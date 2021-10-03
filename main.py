

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


env1 = Env1(grid_size=grid_size, n_objs=n_objs, start_energy=1.0, num_tasks=tasks)
env2 = Env1(grid_size=grid_size, n_objs=n_objs, start_energy=2.0, num_tasks=tasks)


def env_step(action: np.ndarray, env: List[Environment]) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Returns state, reward and done flag given an action
    Actions are an array because there are multiple agents
    Rewards (output) is an array because there are multiple rewards, agent costs, task rewards
    """
    state, reward, done = env[0].step(action)
    return state.astype(np.float32), reward.astype(np.float32), np.array(done, np.int32)


def tf_env_step(action: tf.Tensor, env) -> List[tf.Tensor]:
    return tf.numpy_function(env_step, [action, env], [tf.float32, tf.float32, tf.int32])


def get_expected_returns(rewards: tf.Tensor, standardise: bool = True) -> tf.Tensor:
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
    envs = [env1, env2]

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
        # Convert state into a batched tensor (batch size=1)
        state = tf.expand_dims(state, 0)
        # Run the model and to get action probabilities and critic value
        action_logits_t, value = model(state)
        print("action logits: {}", action_logits_t)
        # Sample the next action from the action probability distribution
        # Tensor values need to be restricted to the enabled actions at the current state
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)
        # Store critic values
        values = values.write(t, tf.squeeze(value))
        # Store log probability of action chosen
        action_probs = action_probs.write(t, action_probs_t[0, action])
        # Apply actions to the environment to get the next state and reward
        state, reward, done = tf_env_step(action, envs)
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


def compute_sliced_softmax(output, n_actions, agent) -> tf.Tensor:
    """The dense layer of the NN for actions returns multiple action sets. To get the action
    sets for a particular agent i, we need to slice this array. This is instead of doing a reshape
    operation on the network itself"""
    a = output[0][n_actions * agent:n_actions * (agent + 1)]
    return tf.exp(a) / tf.reduce_sum(tf.exp(a), 0)


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
    # todo check if matmul is actually the right function, the intention is to do element wise
    #  multiplication?
    kl = tf.matmul(ini_values, tf.math.log(ini_values) - tf.math.log(e)) + \
         tf.matmul((1.0 - ini_values), tf.math.log(1.0 - ini_values))
    tf.tensordot(mu_ij, kl)


def compute_allocation_func(allocation_logits, n_agents, m_tasks, test=False):
    a = tf.reshape(allocation_logits, [m_tasks, n_agents])
    r = tf.exp(a) / tf.reduce_sum(tf.exp(a), axis=0)
    if test:
        print("allocator logit matrix: \n{}".format(a))
        print("agent logits: \n{}".format(a))
        print("softmax evaluation: \n{}".format(r))
    return r


class TestModelMethods(unittest.TestCase):
    # todo implement a test for converting training data to correct tensor shapes
    # todo implement a test for an instance of calculating a loss function
    # todo test that there is no flow between A and Advantage grad_theta π(a_t|s_t)*(G - A)
    # todo two environment step test, and extract v_pi,ini
    # todo test KL divergence function
    def test_environment(self):
        agents, tasks, grid_size, n_objs = 2, 2, (10, 10), 3
        env1.reset()
        done = False
        for i in range(10):
            #print("Episode {}".format(i))
            while not done:
                action = random.choice(list(Action1))
                state, _, done = env1.step(action=action)
                # print([s.__str__() for s in state])
            env1.reset()
            done = False
        return True

    def test_tf_action_sampling(self):
        print("------------------------------------------\n")
        print("         Testing action selection           ")
        print("--------------------------------------------")
        actions, n_agents, m_tasks = 6, 2, 2
        model = ActorCritic(6, n_agents, m_tasks, 128)
        initial_state = tf.constant(env1.reset(), dtype=tf.float32)
        state = tf.expand_dims(initial_state, 0)
        envs = [env1, env2]
        # Run the model and to get action, and allocation probabilities as well as the critic value
        action_logits_t, allocator_logits_t, value = model(state)
        print("action_logits: {}".format(action_logits_t))
        print("Allocator logits: {}".format(allocator_logits_t))
        mu = compute_allocation_func(allocator_logits_t, n_agents, m_tasks)
        print("probability of allocation: \n{}".format(mu))
        print("total probability of mu: {}".format(tf.reduce_sum(mu[:, 1])))
        assert tf.reduce_sum(mu[:, 1]) == 1.0
        print("value: {}", value)
        # sample the next action from the action probability distribution
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        print("sliced action: {}".format(compute_sliced_softmax(action_logits_t, len(Action1), 0)))
        print("sampled action: {}".format(action))
        # get the next state,reward, done signal
        state, reward, done = tf_env_step(action, envs)
        print("state: {}, reward: {}, done: {}".format(state, reward, done))
        print(model.summary())
        return True

    def test_episode(self):
        print("------------------------------------------\n")
        print("              Testing episode               ")
        print("--------------------------------------------")
        n_actions, n_agents, m_tasks = 6, 2, 2
        model = ActorCritic(n_actions, 2, 2, 128)
        initial_state = tf.constant(env1.reset(), dtype=tf.float32)
        initial_state_shape = initial_state.shape
        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        envs = [env1, env2]
        _, allocation_logits, _ = model(tf.expand_dims(initial_state, 0))
        # mu is the allocation function, which is the probability of allocating a task to an
        # agent.
        # mu is a matrix with dimensions (m_tasks, n_agents) and when we want the distribution of allocation
        # for a particular task we just take the corresponding row in this matrix, which will sum to one.
        mu = compute_allocation_func(allocation_logits, n_agents, m_tasks)

        state = initial_state
        max_steps = 10
        # test environment 2
        for t in tf.range(max_steps):
            state = tf.expand_dims(state, 0)
            action_logits_t, _, value = model(state)
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            action_probs_t = compute_sliced_softmax(action_logits_t, n_actions, 1)

            # Store critic Value (V)
            values = values.write(t, tf.squeeze(value))

            # store log prob of the action chosen
            action_probs = action_probs.write(t, action_probs_t[action])

            # apply action to the environment to get the next state and reward
            for e in envs:
                # now we have to be quite careful here, the state is the state
                # relative to which environment is being observed, the state should be the position
                # of all of the agents in a system, plus the energy of all of the agents at time t
                # therefore we must no take the updated
                state, reward, done = tf_env_step(action, [envs[0]])
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

    def test_compute_h(self):
        pass

    def test_compute_f(self):
        pass

    def test_tf_rewards(self):
        """function to test what the output of the expected rewards for an episode is"""
        # todo we require an instance of run episode, which will return rewards
        # todo input the rewards into the tf function expected rewards, corresponding to v_ini under
        #  some policy π
        return True


if __name__ == '__main__':
    unittest.main()
