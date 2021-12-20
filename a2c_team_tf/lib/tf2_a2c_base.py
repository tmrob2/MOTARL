import collections
import copy
import gym_minigrid.minigrid
import gym
import numpy as np
from typing import Tuple, List
import tensorflow as tf
from a2c_team_tf.utils.dfa import DFA
from a2c_team_tf.utils.parallel_envs import ParallelEnv
from a2c_team_tf.utils.env_utils import make_env


class Agent:
    def __init__(self, envs, model,
                 num_tasks, xdfa, one_off_reward,
                 e, c, mu, chi, lam,
                 gamma=1.0, lr=1e-4, entropy_coef=0.001,
                 seed=None, num_procs=10, num_frames_per_proc=100, recurrence=1, max_eps_steps=100, env_key=None):
        self.recurrent = recurrence > 1
        self.recurrence = recurrence
        self.env: ParallelEnv = \
            ParallelEnv(envs, [copy.deepcopy(xdfa) for _ in range(num_procs)], one_off_reward, num_tasks, [float(max_eps_steps) for _ in range(num_tasks)], seed)
        if env_key:
            self.renv = make_env(env_key=env_key, max_steps_per_episode=max_eps_steps, seed=seed, apply_flat_wrapper=True)
        # self.actor = actor
        # self.critic = critic
        self.model = model
        self.num_tasks = num_tasks
        self.dfa = xdfa
        self.one_off_reward = one_off_reward
        self.e, self.c, self.mu, self.chi, self.lam = e, c, mu, chi, lam
        self.lr = lr
        #self.clr = clr
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.seed = seed
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.huber = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
        self.num_frames_per_proc = num_frames_per_proc
        self.num_procs = num_procs
        self.log_episode_reward = tf.zeros([self.num_procs, self.num_tasks + 1], dtype=tf.float32)
        assert self.recurrent or self.recurrence == 1
        assert num_frames_per_proc % recurrence == 0

    def random_policy(self, initial_state, max_steps):
        """Method that is useful for checking DFAs, generate a random policy
        to move around the environment and take random actions"""
        state = initial_state
        initial_state_shape = initial_state.shape
        cached_dfa_state = None
        for _ in tf.range(max_steps):
            self.renv.render('human')
            state = tf.expand_dims(state, 0)
            # Run the model to get an action probability distribution
            if self.recurrent:
                state = tf.expand_dims(state, 0)
            action_logits_t, _ = self.model(state)
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            # Apply the action to the environment to get the next state and reward
            state, reward, done = self.tf_render_env_step(action)
            if self.dfa.product_state != cached_dfa_state or done:
                print(f"DFA state: {self.dfa.product_state}")
                print(f"DFA complete: {self.dfa.done()}")
                print(f"Reward: {reward}")
                cached_dfa_state = self.dfa.product_state
            state.set_shape(initial_state_shape)
            if tf.cast(done, tf.bool):
                break

    def render_reset(self):
        if self.seed:
            self.renv.seed(self.seed)
        state = self.renv.reset()
        # reset the product DFA
        self.dfa.reset()
        initial_state = np.append(state, np.array(self.dfa.progress, dtype=np.float32))
        return initial_state

    def tf_render_reset(self):
        return tf.numpy_function(self.render_reset, [], [tf.float32])

    def tf_reset2(self):
        return tf.numpy_function(self.env.reset, [], [tf.float32])

    def render_env_step(self, action: np.array) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns state, reward, done flag given an action"""
        state, reward, done, info = self.renv.step(action)
        # where we define the DFA steps
        self.dfa.next(self.renv)
        task_rewards = self.dfa.rewards(self.one_off_reward)
        agent_reward = 0
        if self.dfa.done():
            # Assign the agent reward
            done = True
        else:
            # agent_reward = reward
            done = False
        rewards_ = np.array([agent_reward] + task_rewards)
        state_ = np.append(state, np.array(self.dfa.progress))
        return (
            state_.astype(np.float32),
            np.array(rewards_, np.float32),
            np.array(done, np.int32))

    def tf_render_env_step(self, action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self.render_env_step, [action], [tf.float32, tf.float32, tf.int32])

    def env_step3(self, actions: np.array):
        state, reward, done = self.env.step(actions)
        return state.astype(np.float32), reward.astype(np.float32), done.astype(np.int32)

    def tf_env_step3(self, actions: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self.env_step3, [actions], [tf.float32, tf.float32, tf.int32])

    def render_episode(self, initial_state: tf.Tensor, max_steps: tf.int32):
        state = initial_state
        initial_state_shape = initial_state.shape
        for _ in tf.range(max_steps):
            self.renv.render('human')
            state = tf.expand_dims(state, 0)
            if self.recurrent:
                state = tf.expand_dims(state, 0)
            # Run the model to get an action probability distribution
            action_logits_t, _ = self.model(state)
            if self.recurrent:
                action_logits_t = tf.squeeze(action_logits_t, axis=0)
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            # Apply the action to the environment to get the next state and reward
            state, _, done = self.tf_render_env_step(action)
            state.set_shape(initial_state_shape)
            if tf.cast(done, tf.bool):
                break

    def collect_batch2(self, initial_obs: tf.Tensor, log_reward: tf.Tensor):
        """Collects rollouts and computes advantages
        Runs several environments concurrently, the next actions are computed in a batch
        for all environments at the same time. The rollouts and the advantages from all
        environments are concatenated together
        ** There will be no tensor flow from the tensors returned in this collection to the
           model.
        """
        observations = tf.TensorArray(dtype=tf.float32, size=self.num_frames_per_proc)
        selected_actions = tf.TensorArray(dtype=tf.int32, size=self.num_frames_per_proc)
        values = tf.TensorArray(dtype=tf.float32, size=self.num_frames_per_proc)
        rewards = tf.TensorArray(dtype=tf.float32, size=self.num_frames_per_proc)
        masks = tf.TensorArray(dtype=tf.float32, size=self.num_frames_per_proc)
        running_rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        log_reward_shape = log_reward.shape
        state = initial_obs
        state_shape = initial_obs.shape
        log_reward_counter = 0
        for i in tf.range(self.num_frames_per_proc):
            if self.recurrent:
                state = tf.expand_dims(state, 1)
            observations = observations.write(i, state)
            action_logits_t, value = self.model(state)
            # value = self.critic(state)
            values = values.write(i, value)
            # action_logits [samples, 1, actions] -> [samples, actions]
            action_logits_t = tf.squeeze(action_logits_t)
            actions = tf.random.categorical(action_logits_t, num_samples=1, dtype=tf.int32)
            actions = tf.squeeze(actions)
            selected_actions = selected_actions.write(i, actions)
            # indices = tf.transpose([z, actions])
            # action_probs_t = tf.nn.softmax(action_logits_t)
            state, reward_, done_ = self.tf_env_step3(actions)
            mask = tf.constant(1.0, dtype=tf.float32) - tf.cast(done_, dtype=tf.float32)
            masks = masks.write(i, mask)
            log_reward += reward_
            log_reward.set_shape(log_reward_shape)
            for idx in tf.range(self.num_procs):
                if tf.cast(done_[idx], tf.bool):
                    running_rewards = running_rewards.write(log_reward_counter, log_reward[idx])
                    log_reward_counter += 1
            log_reward = tf.expand_dims(mask, 1) * log_reward
            log_reward.set_shape(log_reward_shape)
            rewards = rewards.write(i, reward_)
            state.set_shape(state_shape)
        # dim labels:
        #   T - timesteps (the number of experiences recorded)
        #   S - samples (generated by parallel env procs)
        #   D - tasks + 1  (agent)
        # action_probs = action_probs.stack() #
        values = values.stack()
        rewards = rewards.stack()
        masks = masks.stack()
        selected_actions = selected_actions.stack()
        observations = observations.stack()
        running_rewards = running_rewards.stack()
        # observations = tf.squeeze(observations)
        values = tf.squeeze(values)
        return selected_actions, observations, values, rewards, masks, state, running_rewards, log_reward

    def _indices(self):
        # because we take n_proc samples from the environment, we map the starting index of the
        # sample trajectory back to the concatenated tensor
        # If the model is not recurrent then recurrence will be set to 1, and the NN model will be updated at
        # every time-step
        sample_starting_indices = \
            np.arange(0, self.num_frames_per_proc * self.num_procs, self.recurrence, dtype=np.int32)
        return sample_starting_indices

    def tf_starting_indexes(self):
        return tf.numpy_function(self._indices, [], [tf.int32])

    def run_episode2(self, initial_state: tf.Tensor, max_steps: tf.int32):
        """Runs a single episode to collect the training data"""
        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state
        for t in tf.range(max_steps):
            state = tf.expand_dims(state, 0)
            if self.recurrent:
                state = tf.expand_dims(state, 0)
            # Run the model to get an action probability distribution
            action_logits_t, value = self.model(state)
            if self.recurrent:
                action_logits_t = tf.squeeze(action_logits_t, axis=0)
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            action_probs_t = tf.nn.softmax(action_logits_t)
            # Store the critic values
            if self.recurrent:
                value = tf.squeeze(value)
            values = values.write(t, value)

            # Store the probabilities of the action chosen
            action_probs = action_probs.write(t, action_probs_t[0, action])

            # Apply the action to the environment to get the next state and reward
            state, reward, done = self.tf_render_env_step(action)
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
        # Compute the expected returns per time-step
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

    def compute_advantages(self, returns, values, ini_values, ini_values_i):
        H = self.compute_H2(ini_values, ini_values_i)
        advantages = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        for p in tf.range(self.num_procs):
            sample_advantage = tf.matmul(returns[:, p] - values[:, p], tf.expand_dims(H[p], 1))
            advantages = advantages.write(p, sample_advantage)
        advantages = advantages.stack()
        return advantages

    def get_expected_return2(self, rewards: tf.Tensor) -> tf.Tensor:
        """Expects the shape of rewards to be (steps, proc_sample, tasks + 1)"""
        # Compute the expected returns per time-step
        n = tf.shape(rewards)[0]  # index 1 is the number of experiences collected
        returns = tf.TensorArray(dtype=tf.float32, size=n)
        # Start from the end of the rewards and accumulate reward sums into
        # the returns array
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.zeros([self.num_procs, self.num_tasks + 1], dtype=tf.float32)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i, :, :]
            discounted_sum = reward + self.gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]
        return returns

    def df(self, x: tf.Tensor) -> tf.Tensor:
        """derivative mean squared error"""
        if tf.less_equal(x, self.c):
            return 2 * (x - self.c)
        else:
            return tf.convert_to_tensor(0.0)

    def dh(self, x: tf.Tensor) -> tf.Tensor:
        if tf.less_equal(x, self.e):
            return 2 * (x - self.e)
        else:
            return tf.convert_to_tensor(0.0)

    def compute_H(
            self, X: tf.Tensor, Xi: tf.Tensor) -> tf.Tensor:
        # _, y = X.get_shape()
        H = [self.lam * self.df(Xi[0])]
        for j in range(1, self.num_tasks + 1):
            # H.append(chi * self.dh(mu * X[:, j], e) * mu)
            H.append(self.chi * self.dh(self.mu * X[j]) * self.mu)
        return tf.convert_to_tensor(H)

    def computeH2_proc_i(self, X: tf.Tensor, Xi: tf.Tensor, proc: tf.int32) -> tf.Tensor:
        # should be a relatively small calculation
        # computation with list comprehension should add some efficiency
        H = tf.TensorArray(dtype=tf.float32, size=self.num_tasks + 1)
        agent_value = self.lam * self.df(Xi[proc, 0])
        H = H.write(0, agent_value)
        for j in tf.range(1, self.num_tasks + 1):
            task_j_value = self.chi * self.dh(self.mu * X[proc, j])
            H = H.write(j, task_j_value)
        H = H.stack()
        return H

    def compute_H2(self, X: tf.Tensor, Xi: tf.Tensor):
        H_ = tf.TensorArray(dtype=tf.float32, size=self.num_procs)
        for proc in tf.range(self.num_procs):
            H = self.computeH2_proc_i(X, Xi, proc)
            H_ = H_.write(proc, H)
        H_ = H_.stack()
        return H_

    def compute_actor_loss2(
            self, action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor,
            ini_values: tf.Tensor, ini_values_i: tf.Tensor):
        H = self.compute_H(ini_values, ini_values_i)
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

    def update_loss(self, observations, actions, masks, returns, advantages, ii, batch_shape, mask_shape):
        loss = 0
        for t in tf.range(self.recurrence):
            ix = ii + t
            # Construct a sub batch of experiences for the timestep t across all of the samples
            sub_batch_obs = tf.gather(observations, indices=ix)
            #sub_batch_obs.set_shape([40, 1, 50])
            # print(f"sub batch shape: {sub_batch_obs.shape}, indices: {ix.shape}")
            # Construct the sub batch mask from experiences
            mask = tf.expand_dims(tf.cast(tf.gather(masks, indices=ix), tf.bool), 1)
            #mask.set_shape([40, 1])
            # Construct the sub batch of advantages and returns from experience samples
            sb_advantage = tf.gather(advantages, indices=ix)
            sb_returns = tf.gather(returns, indices=ix)
            # Compute the actor loss value
            action_logits_t, value = self.model(sub_batch_obs, mask=mask)
            action_logits_t = tf.squeeze(action_logits_t)
            sb_actions = tf.gather(actions, indices=ix)
            action_probs_t = tf.nn.softmax(action_logits_t)
            # helper index
            z = tf.range(action_probs_t.shape[0])
            # construct the slice indices from the helper index and sub batch actions selected
            jj = tf.transpose([z, sb_actions])
            action_probs = tf.gather_nd(action_probs_t, indices=jj)
            # finally, calculate the actor loss
            actor_loss = tf.math.reduce_mean(tf.math.log(action_probs) * tf.squeeze(sb_advantage))
            # compute the critic loss value
            critic_sb_losses = self.huber(tf.squeeze(value), sb_returns)
            # update the loss values
            loss += actor_loss + tf.nn.compute_average_loss(critic_sb_losses)
        loss /= self.recurrence
        return loss

    # @tf.function
    def train(
            self,
            initial_state: tf.Tensor,
            max_steps: tf.int32) -> tf.Tensor:
        """Runs a model training step"""
        with tf.GradientTape() as tape:
            # Runs the model for one epsidoe to collect the training data
            action_probs, values, rewards = self.run_episode2(initial_state, max_steps)
            # Calculate the expected returns
            values = tf.squeeze(values)  # TODO add this while we are only dealing with one agent
            rewards = tf.squeeze(rewards)  # TODO add this while we are only dealing with one agent
            returns = self.get_expected_return(rewards)
            # Convert the training data to appropriate TF shapes
            # action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

            # Calculate the loss function to update the network
            critic_loss = tf.reduce_sum(self.huber(values, returns))
            ini_values = tf.convert_to_tensor(values[0])
            actor_loss = self.compute_actor_loss2(
                action_probs, values, returns, ini_values, ini_values)
        grads1 = tape.gradient(actor_loss + critic_loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads1, self.model.trainable_variables))
        episode_reward = tf.math.reduce_sum(rewards, 0)
        return episode_reward

    # @tf.function
    def train2(self, initial_state: tf.Tensor, log_reward: tf.Tensor):
        # We require masks to know which tensor elements the model should ignore
        # We require values and returns for huber loss
        # We require advantages for actor loss
        # collect the batch of experiences for all environments over the range of time-steps
        acts, obss, values, rewards, masks, state, running_rewards, log_reward = \
            self.collect_batch2(initial_state, log_reward)

        returns = self.get_expected_return2(rewards)
        ini_values = values[0, :, :]
        advantages = self.compute_advantages(returns, values, ini_values, ini_values)

        # concatenate masks together => masks reshape: T x S -> S x T -> S * T
        masks = tf.reshape(tf.transpose(masks), [-1])
        acts = tf.reshape(tf.transpose(acts), [-1])
        # Concatenate the samples together =>
        #   values/rewards/returns reshape: T x S x D -> S x T x D -> (S * T) x D
        values = tf.reshape(tf.transpose(values, perm=[1, 0, 2]), [-1, values.shape[-1]])
        returns = tf.reshape(tf.transpose(returns, perm=[1, 0, 2]), [-1, values.shape[-1]])
        observations = tf.reshape(tf.transpose(obss, perm=[1, 0, 2, 3]), [-1, obss.shape[-2], obss.shape[-1]])
        advantages = tf.reshape(advantages, [-1, advantages.shape[-1]])
        # destroy flow
        advantages = tf.convert_to_tensor(advantages)
        returns = tf.convert_to_tensor(returns)
        observations = tf.convert_to_tensor(observations)
        acts = tf.convert_to_tensor(acts)
        ii = self.tf_starting_indexes()
        batch_shape = tf.gather(observations, ii).shape
        mask_shape = tf.expand_dims(tf.cast(tf.gather(masks, indices=ii), tf.bool), 1).shape
        with tf.GradientTape() as tape:
            loss = self.update_loss(observations, acts, masks, returns, advantages, ii, batch_shape, mask_shape)
            grads1 = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads1, self.model.trainable_variables))

        return state, log_reward, running_rewards, loss




