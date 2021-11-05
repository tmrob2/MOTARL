import collections
import copy

import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm
from a2c_team_tf.utils.dfa import *
from a2c_team_tf.nets.base import ActorCritic
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple
from abc import ABC


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


# Create the environment
env1 = gym.make("CartPole-v0")
env2 = gym.make("CartPole-v0")
envs = [env1, env2]

# Set seed for experiment reproducibility
seed = 42
env1.seed(seed)
env2.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

print(env1.observation_space)

step_rew0 = 15  # step reward threshold

cart_pos = 0.10  # the position for the task

one_off_reward = 10.0  # one-off reward
task_prob0 = 0.8  # the probability threhold of archieving the above task

task = make_move_to_pos_dfa()  # make_move_to_pos_dfa()

num_agents = 2
num_tasks = 1
cpdfa1 = CrossProductDFA(num_tasks=num_tasks, dfas=[task])
dfas = [copy.deepcopy(cpdfa1)] * num_agents

num_actions = env1.action_space.n  # 2
num_hidden_units = 128

models = [ActorCritic(num_actions, num_hidden_units, num_tasks, name="AC{}".format(i)) for i in range(num_agents)]


def env_step(state: np.ndarray, action: np.ndarray, env_index: np.int32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns state, reward and done flag given an action."""

    state_new, step_reward, done, _ = envs[env_index].step(action)

    ## Get a one-off reward when reaching the position threshold for the first time.

    # update the task xDFA
    # task.update(state_new[0])
    dfas[env_index].next(envs[env_index])

    # check if the DFA accepting state set has become non-reachable from the current
    # DFA state
    dfas[env_index].non_reachable()

    # agent-task rewards
    task_rewards = dfas[env_index].rewards(one_off_reward)
    state_task_rewards = [step_reward] + task_rewards

    return (state.astype(np.float32),
            # np.array(step_reward, np.int32),
            np.array(state_task_rewards, np.int32),
            np.array(done, np.int32))


def tf_env_step(state: tf.Tensor, action: tf.Tensor, env_index: tf.int32) -> List[tf.Tensor]:
    # return tf.numpy_function(env_step, [action],
    return tf.numpy_function(env_step, [state, action, env_index],
                             [tf.float32, tf.int32, tf.int32])


def run_episode(
        initial_state: tf.Tensor,
        model: tf.keras.Model,
        env_index: tf.int32,
        max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Runs a single episode to collect training data."""

    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
        # Convert state into a batched tensor (batch size = 1)
        state1 = tf.expand_dims(state, 0)

        # Run the model and to get action probabilities and critic value
        action_logits_t, value = model(state1)

        # Sample next action from the action probability distribution
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)

        # Store critic values
        values = values.write(t, tf.squeeze(value))

        # Store log probability of the action chosen
        action_probs = action_probs.write(t, action_probs_t[0, action])

        # Apply action to the environment to get next state and reward
        state, reward, done = tf_env_step(state, action, env_index)
        state.set_shape(initial_state_shape)

        # Store reward
        rewards = rewards.write(t, reward)

        if tf.cast(done, tf.bool):
            break

    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()

    ## Reset the task score at the end of each episode.
    for ii in range(num_agents):
        env_reset(ii)

    return action_probs, values, rewards


def env_reset(env_index):
    state = envs[env_index].reset()
    dfas[env_index].reset()
    return state


## Change "standardize" to "False", and
## Instantiate "discounted_sum" to a list "[0.0]*(num_tasks+1)".

def get_expected_return(
        rewards: tf.Tensor,
        gamma: float,
        standardize: bool = False) -> tf.Tensor:
    """Compute expected returns per timestep."""

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    # discounted_sum = tf.constant(0.0)
    discounted_sum = tf.constant([0.0] * (num_tasks + 1))
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
        returns = ((returns - tf.math.reduce_mean(returns)) /
                   (tf.math.reduce_std(returns) + eps))

    return returns


## Some auxiliary functions for defining the "compute_loss" function.
mu = 1.0 / num_agents  # fixed even probability of allocating each task to each agent
lam = 1.0
chi = 1.0
c = step_rew0
e = task_prob0 * one_off_reward  # task reward threshold


def df(x: tf.Tensor) -> tf.Tensor:
    """Threshold '<=c' is used as running rewards (not costs) are considered."""
    if x <= c:
        return 2 * (x - c)
    else:
        return tf.convert_to_tensor(0.0)


def dh(x: tf.Tensor) -> tf.Tensor:
    if x <= e:
        return 2 * (x - e)
    else:
        return tf.convert_to_tensor(0.0)


"""
[TO-FIX] Intend to implement the following derivative for the KL loss but get an error.
def dh(x: tf.Tensor) -> tf.Tensor:
  if x <= e and x > 0:
    return tf.math.log(x/e) - tf.math.log((1-x)/(1-e))
  else:
    return tf.convert_to_tensor(0.0)
"""


def compute_H(X: tf.Tensor, Xi: tf.Tensor) -> tf.Tensor:
    _, y = X.get_shape()
    ###Try to use tf.TensorArray to implement H but get an error.!!!
    H = [lam * df(Xi[0])]
    for j in range(1, y):
        H.append(chi * dh(tf.math.reduce_sum(mu * X[:, j])) * mu)
    return tf.expand_dims(tf.convert_to_tensor(H), 1)


## The compute_loss function (with the above aux definitions) implements our loss function
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)


def compute_loss(
        action_probs: tf.Tensor,
        values: tf.Tensor,
        returns: tf.Tensor,
        ini_value: tf.Tensor,
        ini_values_i: tf.Tensor) -> tf.Tensor:
    """Computes the combined actor-critic loss."""

    # advantage = returns - values
    H = compute_H(ini_value, ini_values_i)
    advantage = tf.matmul(returns - values, H)
    action_log_probs = tf.math.log(action_probs)
    actor_loss = tf.math.reduce_sum(action_log_probs * advantage)

    critic_loss = huber_loss(values, returns)

    # print(f'shape of action_log_probs:, {action_log_probs.get_shape()}')
    # print(f'shape of H:, {H.get_shape()}')
    # print(f'shape of advantage:, {advantage.get_shape()}')
    # print(f'shape of actor_loss:, {actor_loss.get_shape()}')
    # print(f'shape of critic_loss:, {critic_loss.get_shape()}')

    return actor_loss + critic_loss


## Have to use a smaller learning_rate to make the training convergent
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # 0.01


## Comment out '@tf.function'.
## This annotation gives me an incorrect (non-reasonable) result.
## Need to figure out whether it is possible to
##  change our customerised implementation in order to use this feature.
##@tf.function
def train_step0(
        models: List[tf.keras.Model],
        optimizer: tf.keras.optimizers.Optimizer,
        gamma: float,
        max_steps_per_episode: int) -> [tf.Tensor, tf.Tensor]:
    """Runs a model training step."""

    num_model = len(models)
    action_probs_l = []
    values_l = []
    rewards_l = []
    returns_l = []

    with tf.GradientTape() as tape:

        for i in range(num_model):
            ## add the location into the state
            initial_state = envs[i].reset()
            np.append(initial_state, 0.0)
            initial_state = tf.constant(envs[i].reset(), dtype=tf.float32)

            # Run the model for one episode to collect training data
            action_probs, values, rewards = run_episode(
                initial_state, models[i], i, max_steps_per_episode)

            # Calculate expected returns
            returns = get_expected_return(rewards, gamma)

            action_probs_l.append(action_probs)
            values_l.append(values)
            rewards_l.append(rewards)
            returns_l.append(returns)

        ini_values = tf.convert_to_tensor([x[0, :] for x in values_l])
        loss_l = []
        for i in range(num_model):
            # Convert training data to appropriate TF tensor shapes
            action_probs = tf.expand_dims(action_probs_l[i], 1)
            ## Don't need to convert the shapes of values and returns from our networks
            # action_probs, values, returns = [
            #  tf.expand_dims(x, 1) for x in [action_probs_l[i], values_l[i], returns_l[i]]]

            values = values_l[i]
            returns = returns_l[i]

            # Calculating loss values to update our network
            ini_values_i = ini_values[i, :]
            loss = compute_loss(action_probs, values, returns, ini_values, ini_values_i)
            loss_l.append(loss)

            # print(f'ini_values for model#{i}: {ini_values_i}')
            # print(f'loss value for model#{i}: {loss}')
            # print(f'returns for model#{i}: {returns[0]}')

    # Compute the gradients from the loss vector
    vars_l = [m.trainable_variables for m in models]
    grads_l = tape.gradient(loss_l, vars_l)

    # Apply the gradients to the model's parameters
    grads_l_f = [x for y in grads_l for x in y]
    vars_l_f = [x for y in vars_l for x in y]
    optimizer.apply_gradients(zip(grads_l_f, vars_l_f))

    episode_reward_l = [tf.math.reduce_sum(rewards_l[i]) for i in range(num_agents)]

    ## For convenience, just return the first episode_reward to the console.
    ## To improve the 'tqdm.trange' code (below) in future.
    return episode_reward_l[0], ini_values


min_episodes_criterion = 100
max_episodes = 10000  # 10000
max_steps_per_episode = 50  # 1000

# Cartpole-v0 is considered solved if average reward is >= 195 over 100
# consecutive trials
reward_threshold = 195
running_reward = 0

## No discount
# Discount factor for future rewards
gamma = 1.00  # 0.99

# Keep last episodes reward
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

with tqdm.trange(max_episodes) as t:
    for i in t:
        # initial_state = tf.constant(env.reset(), dtype=tf.float32)
        # episode_reward = int(train_step(
        #    initial_state, models, optimizer, gamma, max_steps_per_episode))
        episode_reward, ini_values = train_step0(
            models, optimizer, gamma, max_steps_per_episode)

        episode_reward = int(episode_reward)

        episodes_reward.append(episode_reward)
        running_reward = statistics.mean(episodes_reward)

        t.set_description(f'Episode {i}')
        t.set_postfix(
            episode_reward=episode_reward, running_reward=running_reward)

        # Show average episode reward every 10 episodes
        if i % 20 == 0:
            for k in range(num_agents):
                print(f'values at the initial state for model#{k}: {ini_values[k]}')
                # pass # print(f'Episode {i}: average reward: {avg_reward}')

        if running_reward > reward_threshold and i >= min_episodes_criterion:
            break

print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')
