import gym_minigrid.minigrid
import numpy as np
from typing import Tuple, List
import tensorflow as tf


class Agent:
    def __init__(self, env, actor: tf.keras.Model, critic: tf.keras.Model,
                 num_tasks, xdfa, one_off_reward,
                 e, c, mu, chi, lam,
                 gamma=1.0, alr=5e-4, clr=5e-4, entropy_coef=0.001, seed=None):
        self.env: gym_minigrid.minigrid.MiniGridEnv = env
        self.actor = actor
        self.critic = critic
        self.num_tasks = num_tasks
        self.dfa = xdfa
        self.one_off_reward = one_off_reward
        self.e, self.c, self.mu, self.chi, self.lam = e, c, mu, chi, lam
        self.alr = alr
        self.clr = clr
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.seed = seed
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=self.alr)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=self.clr)
        self.huber = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

    def env_step2(self, action: np.array) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns state, reward, done flag given an action"""
        state, reward, done, info = self.env.step(action)
        # where we define the DFA steps
        self.dfa.next(self.env)
        task_rewards = self.dfa.rewards(self.one_off_reward)
        if self.dfa.done():
            # Assign the agent reward
            agent_reward = 1 - 0.9 * self.env.step_count / self.env.max_steps
            done = True
        else:
            agent_reward = 0.0
            done = False
        rewards_ = np.array([agent_reward] + task_rewards)
        state_ = np.append(state, np.array(self.dfa.progress))
        return (
            state_.astype(np.float32),
            np.array(rewards_, np.float32),
            np.array(done, np.int32))

    def reset(self):
        if self.seed:
            self.env.seed(self.seed)
            state = self.env.reset()
        else:
            state = self.env.reset()
        # reset the product DFA
        self.dfa.reset()
        initial_state = np.append(state, np.array(self.dfa.progress, dtype=np.float32))
        return initial_state

    def tf_reset(self):
        return tf.numpy_function(self.reset, [], [tf.float32])

    def tf_env_step2(self, action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self.env_step2, [action], [tf.float32, tf.float32, tf.int32])

    def render_episode(self, initial_state: tf.Tensor, max_steps: tf.int32):
        state = initial_state
        initial_state_shape = initial_state.shape
        for _ in tf.range(max_steps):
            self.env.render('human')
            state = tf.expand_dims(state, 0)
            # Run the model to get an action probability distribution
            action_logits_t = self.actor(state)
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            # Apply the action to the environment to get the next state and reward
            state, reward, done = self.tf_env_step2(action)
            state.set_shape(initial_state_shape)
            if tf.cast(done, tf.bool):
                break

    def run_episode2(self, initial_state: tf.Tensor, max_steps: tf.int32):
        """Runs a single episode to collect the training data"""
        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state
        for t in tf.range(max_steps):
            state = tf.expand_dims(state, 0)
            # Run the model to get an action probability distribution
            action_logits_t = self.actor(state)
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            action_probs_t = tf.nn.softmax(action_logits_t)
            # Store the critic values
            value = self.critic(state)
            values = values.write(t, value)

            # Store the probabilities of the action chosen
            action_probs = action_probs.write(t, action_probs_t[0, action])

            # Apply the action to the environment to get the next state and reward
            state, reward, done = self.tf_env_step2(action)
            state.set_shape(initial_state_shape)

            # Store the reward
            rewards = rewards.write(t, reward)

            if tf.cast(done, tf.bool):
                break

        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()

        return action_probs, values, rewards

    def get_expected_return(self, rewards: tf.Tensor) -> tf.Tensor:
        # Compute the expected returns per timestep
        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)
        # Start from the end of the rewards and accumulate reward sums into
        # the returns array
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.zeros([self.num_tasks + 1], dtype=tf.float32)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + self.gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]
        return returns

    def df(self, x: tf.Tensor, c: tf.float32) -> tf.Tensor:
        """derivative mean squared error"""
        if tf.less_equal(x, c):
            return 2 * (x - c)
        else:
            return tf.convert_to_tensor(0.0)

    def dh(self, x: tf.Tensor, e: tf.Tensor) -> tf.Tensor:
        if tf.less_equal(x, e):
            return 2 * (x - e)
        else:
            return tf.convert_to_tensor(0.0)

    def compute_H(
            self, X: tf.Tensor, Xi: tf.Tensor, c: tf.Tensor, e: tf.Tensor,
            lam: tf.float32, chi: tf.float32, mu: tf.float32) -> tf.Tensor:
        # _, y = X.get_shape()
        H = [lam * self.df(Xi[0], c)]
        for j in range(1, self.num_tasks + 1):
            # H.append(chi * self.dh(mu * X[:, j], e) * mu)
            H.append(chi * self.dh(mu * X[j], e) * mu)
        return tf.convert_to_tensor(H)

    def compute_actor_loss2(
            self, action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor,
            ini_values: tf.Tensor, ini_values_i: tf.Tensor):
        H = self.compute_H(ini_values, ini_values_i, self.c, self.e, self.lam, self.chi, self.mu)
        H = tf.expand_dims(H, 1)  # add this while there is only one agent, will need to be taken out this library becomes multiagent
        advantage = tf.matmul(returns - values, H)
        action_log_probs = tf.math.log(action_probs)
        actor_loss = tf.math.reduce_sum(action_log_probs * advantage)
        return actor_loss

    def compute_actor_loss(
            self,
            action_probs: tf.Tensor,
            returns: tf.Tensor,
            values: tf.Tensor) -> tf.Tensor:
        advantage = returns - values
        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)
        # entropy_loss = tf.keras.losses.categorical_crossentropy(action_probs, action_probs)
        return actor_loss  # - self.entropy_coef * entropy_loss

    @tf.function
    def train(
            self,
            initial_state: tf.Tensor,
            max_steps: tf.int32) -> tf.Tensor:
        """Runs a model training step"""
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            # Runs the model for one epsidoe to collect the training data
            action_probs, values, rewards = self.run_episode2(initial_state, max_steps)
            # Calculate the expected returns
            values = tf.squeeze(values)  # TODO add this while we are only dealing with one agent
            rewards = tf.squeeze(rewards)  # TODO add this while we are only dealing with one agent
            returns = self.get_expected_return(rewards)
            # Convert the training data to appropriate TF shapes
            # action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

            # Calculate the loss function to update the network
            critic_loss = self.huber(values, returns)
            ini_values = tf.convert_to_tensor(values[0])
            actor_loss = self.compute_actor_loss2(
                action_probs, values, returns, ini_values, ini_values)
        grads1 = tape1.gradient(actor_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(critic_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        episode_reward = tf.math.reduce_sum(rewards, 0)
        return episode_reward



