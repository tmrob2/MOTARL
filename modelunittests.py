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


#seed = 103
#tf.random.set_seed(seed)
#np.random.seed(seed)
eps = np.finfo(np.float32).eps.item()
TASKS, GRIDSIZE, N_OBJS = 3, (10, 10), 3


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
        elif action == Action1.PICK:
            self.position.set(x, y)
        elif action == Action1.PLACE:
            self.position.set(x, y)
        self.energy += self.STEP_VALUE


env1 = Env1(grid_size=GRIDSIZE, n_objs=N_OBJS, start_energy=1.0, start_location=Coord(4, 4), num_tasks=TASKS)
env2 = Env1(grid_size=GRIDSIZE, n_objs=N_OBJS, start_energy=2.0, start_location=Coord(7, 3), num_tasks=TASKS)
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

def compute_total_state(acc: tf.Tensor, new: tf.Tensor) -> tf.Tensor:
    """Compute a state which acknowledges the position, and energy, of all environments in a system"""
    return tf.concat([acc, new], 0)


def compute_action(action_logits_t: tf.Tensor, n_actions, agent) -> tf.int32:
    """Randomly select 1 action according to the parametrised action distribution of the model"""
    act = tf.random.categorical([action_logits_t[0][n_actions * agent:n_actions * (agent + 1)]], 1)[0, 0]
    return tf.cast(act, tf.int32)


def concatenate_rewards(acc: tf.Tensor, new: tf.Tensor) -> tf.Tensor:
    """extends rewards returned from env(i).step"""
    return tf.concat([acc, new], 0)


def concatenate_costs(acc: tf.Tensor, new: tf.Tensor) -> tf.Tensor:
    """extends the costs returned from env(i).step for an agents expenditure"""
    return tf.concat([acc, new], 0)


def env_step(action: np.ndarray, env: List[Environment], index: List[np.int32]) \
        -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns state, reward and done flag given an action
    Actions are an array because there are multiple agents
    Rewards (output) is an array because there are multiple rewards, agent costs, task rewards
    """
    # here we always select env[0] because the input to this function is always one env object
    state, reward, cost, done = env[0].step(action, index[0])
    return state.astype(np.float32), reward.astype(np.float32), np.array(cost, np.float32), np.array(done, np.int32)


def tf_env_step(action: tf.Tensor, env, index) -> List[tf.Tensor]:
    """
    tensorflow function for wrapping the environment step function of the env object
    returns model parameters defined in model.py of a tf.keras.model
    """
    return tf.numpy_function(env_step, [action, env, index], [tf.float32, tf.float32, tf.float32, tf.int32])


def get_expected_returns(rewards: tf.Tensor, costs: tf.Tensor, n_agents: tf.int32, m_tasks: tf.int32) -> Tuple[tf.Tensor, tf.Tensor]:
    """Compute expected returns per timestep"""
    n = tf.shape(rewards)[0]
    v_returns = tf.TensorArray(dtype=tf.float32, size=n)
    i_returns = tf.TensorArray(dtype=tf.float32, size=n)

    # Start from the end of rewards and accumulate reward sums into the returns array
    v_rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    i_rewards = tf.cast(costs[::-1], dtype=tf.float32)

    sum_val = tf.zeros([n_agents * m_tasks], tf.float32)
    sum_val_shape = sum_val.shape
    sum_val_i = tf.zeros([n_agents], tf.float32)
    sum_val_i_shape = sum_val_i.shape

    for i in tf.range(n):
        # task rewards
        v_reward = v_rewards[i]
        sum_val = v_reward + sum_val
        sum_val.set_shape(sum_val_shape)
        v_returns = v_returns.write(i, sum_val)
        # costs
        i_reward = i_rewards[i]
        sum_val_i = i_reward + sum_val_i
        sum_val_i.set_shape(sum_val_i_shape)
        i_returns = i_returns.write(i, sum_val_i)

    v_returns = v_returns.stack()[::-1]
    i_returns = i_returns.stack()[::-1]
    return v_returns, i_returns


# todo rewrite run episode according to the test function test_run_episode
def run_episode(
        initial_state: tf.Tensor,
        model: tf.keras.Model,
        max_steps: int,
        envs: List[Environment],
        n_agents: int,
        n_actions: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Runs a single episode to collect training data"""

    # todo, we basically know the size of all of these so they shouldn't be dynamic
    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    costs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    total_state_tf = initial_state
    initial_state_shape = total_state_tf.shape

    for t in tf.range(max_steps):
        # Run the model and to get action probabilities and critic value
        action_logits_t, _, value = model(total_state_tf)
        # Sample the next action from the action probability distribution
        # Tensor values need to be restricted to the enabled actions at the current state
        actions_chosen = tf.convert_to_tensor(np.array([], dtype=np.int32))
        action_probs_t = tf.convert_to_tensor(np.array([], dtype=np.float32))
        for i in tf.range(n_agents):
            action = compute_action(action_logits_t, n_actions, i)
            actions_chosen = tf.concat([actions_chosen, tf.expand_dims(action, 0)], 0)
            action_probs_agent = compute_sliced_softmax(action_logits_t, n_actions, i)
            action_probs_t = tf.concat([action_probs_t, tf.expand_dims(action_probs_agent[0], 0)], 0)

        # Store critic values
        #print("value: {}".format(value))
        values = values.write(t, value[0])

        # Store log probabilities of actions chosen
        action_probs = action_probs.write(t, action_probs_t)

        # apply action to the environment to get the next state and reward
        acc = tf.constant(np.array([], dtype=np.float32))
        rewards_acc = tf.convert_to_tensor(np.array([], dtype=np.float32))
        costs_acc = tf.convert_to_tensor(np.array([], dtype=np.float32))
        for (i, e) in enumerate(envs):
            # now we have to be quite careful here, the state is the state
            # relative to which environment is being observed, the state should be the position
            # of all of the agents in a system, plus the energy of all of the agents at time t
            # therefore we must no take the updated
            state, reward, cost, done = tf_env_step(actions_chosen[i], [e], [i])
            acc = compute_total_state(acc, state)
            rewards_acc = concatenate_rewards(rewards_acc, reward)
            costs_acc = concatenate_costs(costs_acc, tf.expand_dims(tf.constant(cost), 0))

        total_state = acc
        total_state_tf = tf.expand_dims(total_state, 0)
        total_state_tf.set_shape(initial_state_shape)

        # Store rewards
        rewards = rewards.write(t, rewards_acc)
        costs = costs.write(t, costs_acc)
        if tf.cast(done, tf.bool):
            break
    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()
    costs = costs.stack()

    return action_probs, values, rewards, costs


def allocation_probs(allocation_logits: tf.Tensor) -> tf.Tensor:
    """µ(i,j) is the allocation probability for task j to agent i. This can be determined at
    t=0 because the allocation is fixed throughout the duration of the episode"""
    return tf.nn.softmax(allocation_logits)


def make_ini_values(v_returns: tf.Tensor) -> tf.Tensor:
    """We construct this function to break the flow of tensors for values that are
    considered constant at time t, and not propagated"""
    return tf.convert_to_tensor(v_returns.numpy())


def compute_f(cost_returns: tf.Tensor, c: tf.Tensor) -> tf.Tensor:
    """Cost function to determine if an agent has exceeded it cost threshold and applies a
    corresponding penalty"""
    f_val = tf.TensorArray(dtype=tf.float32, size=tf.shape(cost_returns)[0])
    for i in range(tf.shape(c)[0]):
        val = tf.math.square(tf.math.maximum(0, cost_returns[i] - c[i]))
        f_val.write(i, val)
    f_val = f_val.stack()
    return f_val


def compute_sliced_softmax(output, n_actions, agent) -> tf.Tensor:
    """The dense layer of the NN for actions returns multiple action sets. To get the action
    sets for a particular agent i, we need to slice this array. This is instead of doing a reshape
    operation on the network itself"""
    a = output[0][n_actions * agent:n_actions * (agent + 1)]
    return tf.exp(a) / tf.reduce_sum(tf.exp(a), 0)


def compute_h(v_returns: tf.Tensor, e: np.ndarray, mu: tf.Tensor, m_tasks: tf.int32, n_agents: tf.int32) -> tf.Tensor:
    """h is a divergence function which determines how far away the reward frequency is from the
    required task probability threshold
    Parameters:
    -----------
    ini_values: is a vector of Expected returns E[G^{i,j} | S0 = s] from t=0 to t_max, this will
    actually be a tensor slice because we are only looking at the jth task

    mu_ij: is the probability distribution of allocating task j across all agents

    intuitively this requires that all environments must be 'stepped' at each time step so that
    we can record all of the v_{pi,ini} values
    """
    hreturn = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    for j in tf.range(m_tasks, dtype=tf.int32):
        e_val = tf.fill(v_returns.shape[0], e[j])
        indices = tf.constant(np.array([i * TASKS + j for i in range(n_agents)]))
        r = tf.gather(tf.transpose(v_returns), indices=indices)
        v_ij_ini = tf.squeeze(r)
        i_values = tf.TensorArray(dtype=tf.float32, size=v_returns.shape[0])
        for i in range(n_agents):
            i_values.write(i, tf.math.scalar_mul(mu[j, i], v_ij_ini[i]))
        i_values = i_values.stack()
        y_value = tf.reduce_sum(i_values, axis=0)
        kl = (y_value + eps) * (tf.math.log(y_value + eps) - tf.math.log(e_val)) + \
             (1.0 - y_value + eps) * (tf.math.log(1.0 - y_value) - tf.math.log(1.0 - e_val))
        hreturn = hreturn.write(j, kl)
    hreturn = hreturn.stack()
    return hreturn


def compute_allocation_func(allocation_logits, n_agents, m_tasks, test=False):
    """
    computes an allocation function mu: I x J -> [0,1], which naturally is a tensor (dim 2) of probabilities
    where each row sums to one.
    """
    a = tf.reshape(allocation_logits, [m_tasks, n_agents])
    r = tf.exp(a) / tf.reduce_sum(tf.exp(a), axis=0)
    return r


def compute_loss(
        action_probs: tf.Tensor,
        v_values: tf.Tensor,
        v_returns: tf.Tensor,
        c_returns: tf.Tensor,
        n_agents: tf.int32,
        m_tasks: tf.int32,
        mu: tf.Tensor,
        c: tf.Tensor,
        e: np.ndarray) -> tf.Tensor:
    advantages = v_returns - v_values
    action_log_probs = tf.math.log(action_probs)
    print("advantages shape", advantages.shape)
    f_consts = compute_f(c_returns, c)
    h_consts = compute_h(v_returns, e, mu, m_tasks, n_agents)
    # todo I think we actually have to add the loss up for each i in I and j in J
    # look at equations 7, 8, 9 and try and understand them thoroughly
    loss = -tf.math.reduce_sum(action_log_probs[:, 0] * advantages[:, 0] * (f_consts[:, 0] + h_consts[0, :]))
    for i in tf.range(1, n_agents):
        loss += -tf.math.reduce_sum(action_log_probs[:, i] * advantages[:, i] * (f_consts[:, i] + h_consts[i, :]))
    critic_loss = huber_loss(v_values, v_returns)
    return loss

class TestModelMethods(unittest.TestCase):
    # implement a test for converting training data to correct tensor shapes
    # todo implement a test for an instance of calculating a loss function
    # todo test that there is no flow between A and Advantage grad_theta π(a_t|s_t)*(G - A)
    # two environment step test, and extract v_pi,ini
    # todo test KL divergence function
    def test_environment(self):
        print("\n------------------------------------------")
        print(  "           Env action selection           ")
        print(  "------------------------------------------")
        agents, tasks, grid_size, n_objs = 2, 2, (10, 10), 3
        env1.reset()
        done = False
        for i in range(10):
            #print("Episode {}".format(i))
            while not done:
                action = random.choice(list(Action1))
                state, _, cost, done = env1.step(action=action, index=0)
                #print([s.__str__() for s in state])
            env1.reset()
            done = False
            print("env: e: {}, start: ({},{}), position: ({},{})".format(env1.energy, env1.start_position.x, env1.start_position.y, env1.position.x, env1.position.y))
        return True

    def test_tf_action_sampling(self):
        print("\n------------------------------------------")
        print(  "       Testing action selection           ")
        print(  "------------------------------------------")
        actions, n_agents, m_tasks = 6, 2, 3
        model = ActorCritic(6, n_agents, m_tasks, 128)
        envs = [env1, env2]

        acc = tf.constant(np.array([], dtype=np.float32))
        for e in envs:
            new = tf.constant(e.reset(), dtype=tf.float32)
            compute_total_state(acc, new)

        total_init_state = acc
        state = tf.expand_dims(total_init_state, 0)

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
        action = tf.random.categorical([action_logits_t[0][len(Action1)*0:1 * len(Action1)]], 1)[0, 0]
        print("sliced action: {}".format(compute_sliced_softmax(action_logits_t, len(Action1), 0)))
        print("sampled action: {}".format(action))
        # get the next state,reward, done signal
        for (i, e) in enumerate(envs):
            state, reward, cost, done = tf_env_step(action, [e], [i])
            print("state: {}, reward: {}, done: {}".format(state, reward, done))
        print(model.summary())
        return True

    def test_episode(self):
        print("\n------------------------------------------")
        print(  "            Testing episode               ")
        print(  "------------------------------------------")
        n_actions, n_agents, m_tasks = 7, 2, 3
        model = ActorCritic(n_actions, n_agents, m_tasks, 128)
        envs = [env1, env2]
        # we construct a state which records all of the agent positions, but we suggest that
        # the action only affects that particular environment
        acc = tf.constant(np.array([], dtype=np.float32))
        for e in envs:
            new = tf.constant(e.reset(), dtype=tf.float32)
            acc = compute_total_state(acc, new)
        print("acc: {}".format(acc))
        total_init_state = acc
        total_state_tf = tf.expand_dims(total_init_state, 0)
        initial_state_shape = total_state_tf.shape
        print("init state shape: {}".format(initial_state_shape))
        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        costs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        _, allocation_logits, _ = model(total_state_tf)
        # mu is the allocation function, which is the probability of allocating a task to an
        # agent.
        # mu is a matrix with dimensions (m_tasks, n_agents) and when we want the distribution of allocation
        # for a particular task we just take the corresponding row in this matrix, which will sum to one.
        mu = compute_allocation_func(allocation_logits, n_agents, m_tasks)
        max_steps = 1000
        # test environment 2
        for t in tf.range(max_steps):
            action_logits_t, _, value = model(total_state_tf)
            actions_chosen = tf.convert_to_tensor(np.array([], dtype=np.int32))
            action_probs_t = tf.convert_to_tensor(np.array([], dtype=np.float32))
            # todo (optimisation) convert the action selection operation to a tensor
            # todo (optimsiation) do a matrix transformation function which projects the action probabilities under the selected actions
            for i in tf.range(n_agents):
                action = compute_action(action_logits_t, n_actions, i)
                actions_chosen = tf.concat([actions_chosen, tf.expand_dims(action, 0)], 0)
                action_probs_agent = compute_sliced_softmax(action_logits_t, n_actions, i)
                action_probs_t = tf.concat([action_probs_t, tf.expand_dims(action_probs_agent[action], 0)], 0)

            # Store critic Value (V)
            values = values.write(t, value[0])

            # Store log probabilities of actions chosen
            action_probs = action_probs.write(t, action_probs_t)

            # apply action to the environment to get the next state and reward
            acc = tf.convert_to_tensor(np.array([], dtype=np.float32))
            rewards_acc = tf.convert_to_tensor(np.array([], dtype=np.float32))
            costs_acc = tf.convert_to_tensor(np.array([], dtype=np.float32))
            for (i, e) in enumerate(envs):
                # now we have to be quite careful here, the state is the state
                # relative to which environment is being observed, the state should be the position
                # of all of the agents in a system, plus the energy of all of the agents at time t
                # therefore we must no take the updated
                state, reward, cost, done = tf_env_step(actions_chosen[i], [e], [i])
                acc = compute_total_state(acc, state)
                rewards_acc = concatenate_rewards(rewards_acc, reward)
                costs_acc = concatenate_costs(costs_acc, tf.expand_dims(tf.constant(cost), 0))
            total_state = acc
            total_state_tf = tf.expand_dims(total_state, 0)
            total_state_tf.set_shape(initial_state_shape)

            #store rewards
            # todo this is incorrect there though be j rewards returned at each time step for each agent
            rewards = rewards.write(t, rewards_acc)
            costs = costs.write(t, costs_acc)
            #print("done: {}".format(done))
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
        print("\n------------------------------------------")
        print(  "         Testing task divergence          ")
        print(  "------------------------------------------")
        n_actions, n_agents, m_tasks = 7, 2, TASKS
        max_steps_per_episode = 1000
        model = ActorCritic(n_actions, n_agents, m_tasks, 128)
        envs = [env1, env2]
        acc = tf.constant(np.array([], dtype=np.float32))
        for e in envs:
            new = tf.constant(e.reset(), dtype=tf.float32)
            acc = compute_total_state(acc, new)
        total_init_state = acc
        total_init_state_tf = tf.expand_dims(total_init_state, 0)
        _, allocation_logits, _ = model(total_init_state_tf)
        # mu is the allocation function, which is the probability of allocating a task to an
        # agent.
        # mu is a matrix with dimensions (m_tasks, n_agents) and when we want the distribution of allocation
        # for a particular task we just take the corresponding row in this matrix, which will sum to one.
        mu = compute_allocation_func(allocation_logits, n_agents, m_tasks)
        action_probs, values, rewards, costs = run_episode(total_init_state_tf, model, max_steps_per_episode, envs,
                                                           n_agents,
                                                           n_actions)
        # returns are v_ij_ini, costs are v_i_ini
        returns, cost_returns = get_expected_returns(rewards, costs, n_agents, m_tasks)

        # test the input value of h
        # for a given task we want u_ij * v_ij_ini i.e. all of the jth components of the values in returns
        # The size of v_ij is t * (i*j), we want to take slices for a fixed j for each i
        j = 0
        indices = tf.constant([i * TASKS + j for i in range(n_agents)])
        print("finding v_ij_ini values for indices: {}".format(indices))
        r = tf.gather(tf.transpose(returns), indices=[[0], [3]])
        print("r: \n", tf.squeeze(r), "\nr shape: ", tf.squeeze(r).shape)
        action_log_probs = tf.math.log(action_probs)
        print("action log(pr): \n", action_log_probs, "\nshape: ", action_log_probs.shape)
        v_ij_ini = tf.squeeze(tf.transpose(r))
        actor_loss = action_log_probs * v_ij_ini
        print("actor loss: ", actor_loss)
        expected_values = tf.reduce_sum(actor_loss, axis=0)
        print("E[G | S, a]: {}".format(expected_values))
        e_thresh = tf.constant([0.6, 0.5, 0.4], dtype=tf.float32)
        y = tf.reduce_sum(mu[0, :] * expected_values)
        ini_values = make_ini_values(v_returns=returns)
        #h_vals = compute_h(ini_values, e_thresh, mu, m_tasks, n_agents)
        print("y: ", y)
        #print("h_vals: ", h_vals)

    def test_compute_f(self):
        pass

    def test_expected_rewards(self):
        """function to test what the output of the expected rewards for an episode is"""
        # todo we require an instance of run episode, which will return rewards
        # todo input the rewards into the tf function expected rewards, corresponding to v_ini under
        #  some policy π
        print("\n------------------------------------------")
        print(  "            Testing returns               ")
        print(  "------------------------------------------")
        n_actions, n_agents, m_tasks = 7, 2, 3
        max_steps_per_episode = 1000
        model = ActorCritic(n_actions, n_agents, m_tasks, 128)
        envs = [env1, env2]
        acc = tf.constant(np.array([], dtype=np.float32))
        for e in envs:
            new = tf.constant(e.reset(), dtype=tf.float32)
            acc = compute_total_state(acc, new)
        total_init_state = acc
        total_init_state_tf = tf.expand_dims(total_init_state, 0)
        action_probs, values, rewards, costs = run_episode(total_init_state_tf, model, max_steps_per_episode, envs, n_agents, n_actions)
        print("Rewards shape: {}".format(tf.shape(rewards)))
        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)
        # Start from the end of rewards and accumulate reward sums into the returns array
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        print(rewards)

        sum_val = tf.zeros([n_agents * m_tasks], tf.float32)
        sum_val_shape = sum_val.shape
        print("sum val shape: {}".format(sum_val_shape))

        for i in tf.range(n):
            reward = rewards[i]
            #print("reward: {}".format(reward))
            sum_val = reward + sum_val
            #print("sum val: {}".format(sum_val))
            sum_val.set_shape(sum_val_shape)
            returns = returns.write(i, sum_val)
        returns = returns.stack()[::-1]
        print("returns")
        return True

    def test_compute_loss(self):
        print("\n------------------------------------------")
        print(  "            Testing loss                  ")
        print(  "------------------------------------------")
        n_actions, n_agents, m_tasks = 7, 2, 3
        max_steps_per_episode = 1000
        model = ActorCritic(n_actions, n_agents, m_tasks, 128)
        envs = [env1, env2]
        acc = tf.constant(np.array([], dtype=np.float32))
        c_thresh = tf.constant([0.5, 0.5], dtype=tf.float32)
        e_thresh = np.array([0.6, 0.6, 0.6], dtype=np.float32)

        for e in envs:
            new = tf.constant(e.reset(), dtype=tf.float32)
            acc = compute_total_state(acc, new)
        total_init_state = acc
        total_init_state_tf = tf.expand_dims(total_init_state, 0)
        _, allocation_logits, _ = model(total_init_state_tf)
        mu = compute_allocation_func(allocation_logits, n_agents, m_tasks)
        action_probs, values, rewards, costs = run_episode(total_init_state_tf, model, max_steps_per_episode, envs, n_agents, n_actions)
        returns, cost_returns = get_expected_returns(rewards, costs, n_agents, m_tasks)
        print("returns shape: {}, values shape: {}".format(returns.shape, values.shape))
        # this is all of the advantages in the format (i1,j1),(i1,j2),...(i1,jm),(i2,j1),...,(in,jm)
        advantages = returns - values
        print("Advantages: \n{}".format(advantages))
        action_log_probs = tf.math.log(action_probs)
        print("action log(pr) \n{}".format(action_log_probs))
        v_returns = make_ini_values(returns)
        v_values = make_ini_values(values)
        compute_loss(action_probs, v_values, v_returns, cost_returns, n_agents, m_tasks, mu, c_thresh, e_thresh)
        return True


if __name__ == '__main__':
    unittest.main()
