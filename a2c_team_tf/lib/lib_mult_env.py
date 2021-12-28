import numpy as np
import gym
import tensorflow as tf
from tensorflow import Variable

from a2c_team_tf.utils.dfa import CrossProductDFA
from typing import List, Tuple, Union, Any
from enum import Enum

eps = np.finfo(np.float32).eps.item()
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

class Agent:
    def __init__(
            self,
            envs,
            dfas: List[CrossProductDFA],
            c, e, chi, lam, gamma,
            one_off_reward,
            num_tasks, num_agents, lr1=1e-4, lr2=1e-4):
        self.envs = envs
        self.e, self.c, self.chi, self.lam = e, c, chi, lam
        self.gamma = gamma
        self.lr1 = lr1
        self.lr2 = lr2
        self.dfas: List[CrossProductDFA] = dfas
        self.num_tasks = num_tasks
        self.num_agents = num_agents
        self.mean: tf.Variable = tf.Variable(0.0, trainable=False)
        self.episode_reward: tf.Variable = tf.Variable(0.0, trainable=False)
        self.one_off_reward = one_off_reward
        self.opt = tf.keras.optimizers.Adam(learning_rate=lr1)
        self.huber = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

    def env_step(self, action: np.ndarray, agent: np.int32) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """Returns state, reward and done flag given an action."""

        state_new, step_reward, done, _ = self.envs[agent].step(action)

        ## Get a one-off reward when reaching the position threshold for the first time.
        # update the task xDFA
        data = {"state": state_new, "reward": step_reward, "done": done}
        self.dfas[agent].next(data)
        # print("state", state_new)
        #if agent == 1:
        #    print(f"agent ({agent}) ", self.dfas[agent].progress)

        # agent-task rewards
        task_rewards = self.dfas[agent].rewards(self.one_off_reward)
        state_task_rewards = [step_reward] + task_rewards
        # append the dfa state to the agent state
        state_ = np.append(state_new, np.array(self.dfas[agent].progress))

        if self.dfas[agent].done():
            done = True
        else:
            done = False

        return (state_.astype(np.float32),
                # np.array(step_reward, np.int32),
                np.array(state_task_rewards, np.float32),
                np.array(done, np.int32))

    def render_episode(self, max_steps, *models):
        initial_state = self.get_initial_states()
        state = [initial_state[i] for i in range(self.num_agents)]
        # state_shape = initial_states.shape
        for _ in tf.range(max_steps):
            dones = []
            for i, model in enumerate(models):
                # generate a policy according to the model and act out the trajectory
                self.envs[i].render('human')
                state_ = tf.expand_dims(state[i], 0)
                action_logits, _ = model(state_)
                action = tf.random.categorical(action_logits, 1)[0, 0]
                state_i, _, done = self.tf_env_step(action, i)
                dones.append(tf.cast(done, tf.bool).numpy())
                state[i] = state_i
            if all(dones):
                break

    def env_reset(self, agent):
        state = self.envs[agent].reset()
        self.dfas[agent].reset()
        initial_state = np.append(state, np.array(self.dfas[agent].progress, dtype=np.float32))
        return initial_state

    def tf_reset(self, agent: tf.int32):
        return tf.numpy_function(self.env_reset, [agent], [tf.float32])

    def get_initial_states(self):
        initial_states = tf.TensorArray(dtype=tf.float32, size=self.num_agents)
        for agent in tf.range(self.num_agents):
            init_state_i = self.tf_reset(agent)
            initial_states = initial_states.write(agent, init_state_i)
        initial_states = initial_states.stack()
        return initial_states

    def tf_env_step(self, action: tf.Tensor, agent: tf.int32) -> List[tf.Tensor]:
        """
        tensorflow function for wrapping the environment step function of the env object
        returns model parameters defined in base.py of a tf.keras.model
        """
        return tf.numpy_function(self.env_step, [action, agent], [tf.float32, tf.float32, tf.int32])

    def get_expected_returns(
            self,
            rewards: tf.Tensor) -> tf.Tensor:
        """Compute expected returns per timestep for a given agent (implicit in the rewards input)"""
        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        # Start from the end of rewards and accumulate reward sums into the returns array
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)

        discounted_sum = tf.constant([0.0] * (self.num_tasks + 1), dtype=tf.float32)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + self.gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]
        return returns

    def df(self, x: tf.Tensor) -> tf.Tensor:
        """derivative of mean squared error"""
        if tf.less_equal(x, self.c):
            return 2 * (x - self.c)
        else:
            return tf.convert_to_tensor(0.0)

    def dh(self, x: tf.Tensor) -> tf.Tensor:
        # print(f"x: {x}, e: {e}")
        if tf.less_equal(x, self.e):
            return 2 * (x - self.e)
        else:
            return tf.convert_to_tensor(0.0)

    #@staticmethod
    #def dh(x: tf.float32, e: tf.float32) -> tf.Tensor:
    #    if tf.greater_equal(e, x) and tf.greater(x, 0.0):
    #        return tf.math.log(x / e) - tf.math.log((1.0 - x) / (1.0 - e))
    #    else:
    #        return tf.convert_to_tensor(0.0)

    def compute_H(self, X: tf.Tensor, Xi: tf.Tensor, agent: tf.int32, mu: tf.Tensor) -> tf.Tensor:
        """
        :param X: values (non-participant in gradient)
        :param Xi: initial_values (non-participant in gradient)
        :param lam: weigting assigned to the agent performance loss
        :param chi: weighting assigned to the task performance loss
        :param mu: probability of allocation, learned parameter
        :param e: task threshold [0,1]
        :return:
        """
        _, y = X.get_shape()
        H = tf.TensorArray(dtype=tf.float32, size=self.num_tasks + 1)
        H = H.write(0, self.lam * self.df(Xi[0]))
        for j in range(1, y):
            H = H.write(j, self.chi * self.dh(tf.math.reduce_sum(mu[:, j - 1] * X[:, j])) * mu[agent, j - 1])
        H = H.stack()
        return tf.expand_dims(tf.convert_to_tensor(H), 1)

    def compute_alloc_H(self, X: tf.Tensor, mu: tf.Tensor, task: tf.int32):
        return self.chi * self.dh(tf.math.reduce_sum(mu[:, task - 1] * X[:, task]))

    @tf.function
    def compute_alloc_loss(self, ini_values: tf.Tensor, mu: tf.Tensor):
        loss = tf.constant(0.0, dtype=tf.float32)
        for task in tf.range(1, self.num_tasks + 1):
            h = self.compute_alloc_H(ini_values, mu, task)
            agent_ini_val_x_alloc = tf.reduce_sum(tf.transpose(ini_values[:, task]) * mu[:, task - 1])
            alloc_loss_task_j = h * agent_ini_val_x_alloc
            loss += alloc_loss_task_j
        return loss

    def compute_loss(
            self,
            action_probs: tf.Tensor,
            values: tf.Tensor,
            returns: tf.Tensor,
            ini_value: tf.Tensor,
            ini_values_i: tf.Tensor,
            agent: tf.int32,
            mu: tf.Tensor) -> tf.Tensor:
        """Computes the combined actor-critic loss."""

        H = self.compute_H(ini_value, ini_values_i, agent, mu)
        advantage = tf.matmul(returns - values, H)
        action_log_probs = tf.math.log(action_probs)
        actor_loss = tf.math.reduce_sum(action_log_probs * advantage)

        critic_loss = huber_loss(values, returns)
        return actor_loss + critic_loss

    def run_episode(
            self,
            initial_state: tf.Tensor,
            env_index: tf.int32,
            max_steps: tf.int32,
            model: tf.keras.Model):
        """Runs a single episode to collect training data."""

        action_probs = tf.TensorArray(dtype=tf.float32, size=max_steps)
        values = tf.TensorArray(dtype=tf.float32, size=max_steps)
        rewards = tf.TensorArray(dtype=tf.float32, size=max_steps)
        mask = tf.TensorArray(dtype=tf.int32, size=max_steps)
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
            state, reward, done = self.tf_env_step(action, env_index)

            state.set_shape(initial_state_shape)

            rewards = rewards.write(t, reward)
            mask = mask.write(t, 1)

            if tf.cast(done, tf.bool):
                break

        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()
        mask = mask.stack()
        return action_probs, values, rewards, mask

    @tf.function
    def train_step(
            self,
            initial_states: tf.Tensor,
            max_steps_per_episode: tf.int32,
            mu: tf.Tensor,
            *models) -> [tf.Tensor, tf.Tensor]:

        action_probs_l = tf.TensorArray(dtype=tf.float32, size=self.num_agents)
        values_l = tf.TensorArray(dtype=tf.float32, size=self.num_agents)
        rewards_l = tf.TensorArray(dtype=tf.float32, size=self.num_agents)
        returns_l = tf.TensorArray(dtype=tf.float32, size=self.num_agents)
        masks_l = tf.TensorArray(dtype=tf.int32, size=self.num_agents)

        idx = tf.constant(0, dtype=tf.int32)
        with tf.GradientTape() as tape:
            for model in models:
                # Run an episode
                action_probs, values, rewards, mask = self.run_episode(
                    initial_states[idx], idx, max_steps_per_episode, model)

                # Get expected rewards
                returns = self.get_expected_returns(rewards)

                # Append tensors to respective lists
                action_probs_l = action_probs_l.write(idx, action_probs)
                values_l = values_l.write(idx, values)
                rewards_l = rewards_l.write(idx, rewards)
                returns_l = returns_l.write(idx, returns)
                masks_l = masks_l.write(idx, mask)
                idx += tf.constant(1, dtype=tf.int32)

            action_probs_l = action_probs_l.stack()
            values_l = values_l.stack()
            rewards_l = rewards_l.stack()
            returns_l = returns_l.stack()
            masks_l = masks_l.stack()
            ini_values = values_l[:, 0, :]

            loss_l = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

            for i in tf.range(self.num_agents):
                # Get loss
                mask = masks_l[i]
                _, values = tf.dynamic_partition(values_l[i], mask, 2)
                _, returns = tf.dynamic_partition(returns_l[i], mask, 2)
                _, probs = tf.dynamic_partition(action_probs_l[i], mask, 2)
                ini_values_i = ini_values[i]
                loss = self.compute_loss(probs, values, returns, ini_values, ini_values_i, i, mu)
                loss_l = loss_l.write(i, loss)
            loss_l = loss_l.stack()
        # compute the gradient from the loss vector
        vars_l = [m.trainable_variables for m in models]
        grads_l = tape.gradient(loss_l, vars_l)

        # Apply the gradients to the model's parameters
        grads_l_f = [x for y in grads_l for x in y]
        vars_l_f = [x for y in vars_l for x in y]
        self.opt.apply_gradients(zip(grads_l_f, vars_l_f))

        return rewards_l, ini_values




