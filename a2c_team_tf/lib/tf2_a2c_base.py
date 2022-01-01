import copy

import gym
import numpy as np
from typing import Tuple, List
import tensorflow as tf
from a2c_team_tf.utils.parallel_envs import ParallelEnv
from a2c_team_tf.utils.env_utils import make_env


class MORLTAP:
    def __init__(self, envs: List[List[gym.Env]], num_agents,
                 num_tasks, xdfas, one_off_reward,
                 e, c, chi, lam,
                 observation_space, action_space,
                 gamma=1.0, lr=1e-4, lr2=1e-4, entropy_coef=0.001,
                 seed=None, num_procs=10, num_frames_per_proc=100,
                 recurrence=1, max_eps_steps=100, env_key=None, flatten_env=False,
                 q1: tf.queue.FIFOQueue=None, q2: tf.queue.FIFOQueue=None, log_reward: tf.Variable=None):
        self.recurrent = recurrence > 1
        self.recurrence = recurrence
        self.num_agents = num_agents
        self.envs: ParallelEnv = ParallelEnv(
                envs,
                xdfas,
                observation_space,
                action_space,
                one_off_reward,
                num_agents)
        if env_key:
            self.renv = [make_env(
                env_key=env_key,
                max_steps_per_episode=max_eps_steps,
                seed=seed,
                apply_flat_wrapper=flatten_env) for _ in range(num_agents)]
        self.num_tasks = num_tasks
        self.dfas = [copy.deepcopy(xdfas[0][i]) for i in range(self.num_agents)]
        self.one_off_reward = one_off_reward
        self.e, self.c, self.chi, self.lam = e, c, chi, lam
        self.lr = lr
        self.lr2 = lr2
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.seed = seed
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.huber = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
        self.num_frames_per_proc = num_frames_per_proc
        self.num_procs = num_procs
        self.log_episode_reward = tf.zeros([self.num_procs, self.num_tasks + 1], dtype=tf.float32)
        self.q1 = q1
        self.q2 = q2
        self.log_reward: tf.Variable = log_reward
        assert self.recurrent or self.recurrence == 1
        assert num_frames_per_proc % recurrence == 0

    def render_reset(self):
        initial_states = []
        for agent in range(self.num_agents):
            if self.seed:
                self.renv[agent].seed(self.seed)
            state = self.renv[agent].reset()
            # reset the product DFA
            [d.reset() for d in self.dfas]
            # initial_state = np.append(state, np.array(self.dfas.progress, dtype=np.float32))
            initial_states.append(np.append(state, np.array(self.dfas[agent].progress, dtype=np.float32)))
        return initial_states

    def tf_render_reset(self):
        return tf.numpy_function(self.render_reset, [], [tf.float32])

    def reset(self):
        return np.array(self.envs.reset(), dtype=np.float32)

    def tf_reset2(self):
        return tf.numpy_function(self.reset, [], [tf.float32])

    def render_env_step(self, action: np.array, agent: np.int32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns state, reward, done flag given an action"""
        state, reward, done, info = self.renv[agent].step(action)
        # where we define the DFA steps
        self.dfas[agent].next(self.renv[agent])
        task_rewards = self.dfas[agent].rewards(self.one_off_reward)
        agent_reward = 0
        if self.dfas[agent].done():
            done = True
        else:
            done = False
        rewards_ = np.array([agent_reward] + task_rewards)
        state_ = np.append(state, np.array(self.dfas[agent].progress, dtype=np.float32))
        state_ = np.expand_dims(state_, 1)
        state_ = state_.transpose()
        return (
            state_.astype(np.float32),
            np.array(rewards_, np.float32),
            np.array(done, np.int32))

    def tf_render_env_step(self, action: tf.int32, agent: tf.int32) -> List[tf.Tensor]:
        return tf.numpy_function(self.render_env_step, [action, agent], [tf.float32, tf.float32, tf.int32])

    def env_step(self, actions: np.array, agent: np.int32):
        state, reward, done = self.envs.step(actions, agent)
        if self.recurrent:
            state = np.expand_dims(state, 1)
        return state.astype(np.float32), reward.astype(np.float32), done.astype(np.int32)

    def tf_env_step(self, actions: tf.Tensor, agent: tf.int32) -> List[tf.Tensor]:
        return tf.numpy_function(self.env_step, [actions, agent], [tf.float32, tf.float32, tf.int32])

    def render_episode(self, initial_state: List[tf.Tensor], max_steps: tf.int32, *models):
        states = initial_state
        # initial_state_shape = initial_state.shape
        for _ in tf.range(max_steps):
            states_ = []
            dones = [False] * self.num_agents
            for agent in range(self.num_agents):
                self.renv[agent].render('human')
                # Run the model to get an action probability distribution
                action_logits_t, _ = models[agent](states[agent])
                if self.recurrent:
                    action_logits_t = tf.squeeze(action_logits_t, axis=1)
                actions = tf.random.categorical(action_logits_t, 1, dtype=tf.int32)
                # Apply the action to the environment to get the next state and reward
                state, rewards, done = self.tf_render_env_step(actions, agent)
                state = tf.expand_dims(state, 1)
                states_.append(state)
                if tf.cast(done, tf.bool):
                    dones.append(True)
            states = states_
            if all(dones):
                break

    # @tf.function # - is graph compatible
    def collect_batch(self, initial_obs: tf.Tensor, model: tf.keras.Model, agent: tf.int32) \
            -> [tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Collects rollouts and computes advantages
        The expected shape of log_rewards is (samples, num_agents, tasks + 1)
        Runs several environments concurrently, the next actions are computed in a batch
        for all environments at the same time. The rollouts and the advantages from all
        environments are concatenated together.

        Expected shapes:
        T - timesteps, A - agents, S - samples, F - features
        actions - (T, S)
        observations - (T, S, 1, F)
        values - (T, S, tasks + 1)
        rewards - (T, S, tasks + 1)
        mask - (T, S)
        initial state - (S, F)
        log rewards - (S, tasks + 1)
        """
        #print("initial state shape ", initial_obs.shape)
        #print("log reward shape ", log_reward.shape)
        observations = tf.TensorArray(dtype=tf.float32, size=self.num_frames_per_proc)
        selected_actions = tf.TensorArray(dtype=tf.int32, size=self.num_frames_per_proc)
        values = tf.TensorArray(dtype=tf.float32, size=self.num_frames_per_proc)
        rewards = tf.TensorArray(dtype=tf.float32, size=self.num_frames_per_proc)
        masks = tf.TensorArray(dtype=tf.float32, size=self.num_frames_per_proc)
        # log_reward_shape = log_reward.shape
        state = initial_obs
        state_shape = initial_obs.shape
        for i in tf.range(self.num_frames_per_proc):
            observations = observations.write(i, state)
            action_logits_t, value = model(state)
            # value = self.critic(state)
            values = values.write(i, value)
            # action_logits [samples, 1, actions] -> [samples, actions]
            action_logits_t = tf.squeeze(action_logits_t)
            actions = tf.random.categorical(action_logits_t, 1, dtype=tf.int32)
            actions = tf.squeeze(actions)
            selected_actions = selected_actions.write(i, actions)
            state, reward_, done_ = self.tf_env_step(actions, agent)
            state.set_shape(state_shape)
            mask = tf.constant(1.0, dtype=tf.float32) - tf.cast(done_, dtype=tf.float32)
            masks = masks.write(i, mask)

            #print(f"mask shape: {mask.shape}, state shape: {state.shape}, log reward shape: {log_reward.shape}, reward shape: {reward_.shape}")
            padded_reward = tf.repeat(tf.expand_dims(reward_, 0), self.num_agents, 0)
            #print("padded reward ", padded_reward)
            mask_padded_reward = tf.expand_dims(tf.expand_dims(tf.constant([1.0 if x == agent else 0.0 for x in range(self.num_agents)]), 1), 1)
            #print("masked padded reward ", mask_padded_reward)
            masked_rewards = mask_padded_reward * padded_reward
            self.log_reward.assign_add(masked_rewards)
            # log_reward.set_shape(log_reward_shape)
            for idx in tf.range(self.num_procs):
                if tf.cast(done_[idx], tf.bool):
                    self.q1.enqueue(self.log_reward[agent, idx, :])
                    self.log_reward[agent, idx, :].assign(tf.zeros([self.num_tasks + 1]))
                    self.q2.enqueue(agent)
            # log_reward.set_shape(log_reward_shape)
            rewards = rewards.write(i, reward_)
        ### dim labels:
        ###   T - timesteps (the number of experiences recorded)
        ###   S - samples (generated by parallel env procs)
        ###   D - tasks + 1  (agent)
        ## action_probs = action_probs.stack() #
        values = values.stack()
        rewards = rewards.stack()
        masks = masks.stack()
        selected_actions = selected_actions.stack()
        observations = observations.stack()
        # running_rewards = running_rewards.stack()
        values = tf.squeeze(values)
        return selected_actions, observations, values, rewards, masks, state

    def tf_2d_indices(self, agent: tf.int32, size: tf.int32, indices=tf.Tensor):
        x = tf.repeat(agent, size)
        return tf.transpose([x, indices])

    def tf_1d_indices(self):
        return tf.experimental.numpy.arange(0, self.num_frames_per_proc * self.num_procs, self.recurrence, dtype=tf.int32)

    def compute_advantages(self, returns, values, ini_values, mu):
        H_agent_vals = tf.TensorArray(dtype=tf.float32, size=self.num_agents)
        for agent in tf.range(self.num_agents):
            H = self.compute_H(ini_values, ini_values[agent], agent, mu)
            H_agent_vals = H_agent_vals.write(agent, H)
        H_agent_vals = H_agent_vals.stack()
        advantages = tf.TensorArray(dtype=tf.float32, size=self.num_agents)
        for agent in tf.range(self.num_agents):
            adv_proc = tf.TensorArray(dtype=tf.float32, size=self.num_procs)
            for p in tf.range(self.num_procs):
                sample_advantage = tf.matmul(returns[agent, :, p] - values[agent, :, p], tf.expand_dims(H_agent_vals[agent, p], 1))
                adv_proc = adv_proc.write(p, sample_advantage)
            adv_proc = adv_proc.stack()
            advantages = advantages.write(agent, adv_proc)
        advantages = advantages.stack()
        return advantages

    def get_expected_return(self, rewards: tf.Tensor) -> tf.Tensor:
        """Expects the shape of rewards to be (agents, steps, proc_sample, tasks + 1)"""
        # Compute the expected returns per time-step
        # index 1 is the number of experiences collected
        returns = tf.TensorArray(dtype=tf.float32, size=self.num_frames_per_proc)
        # Start from the end of the rewards and accumulate reward sums into
        # the returns array
        rewards = tf.transpose(rewards, perm=[1, 0, 2, 3])
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.zeros([self.num_agents, self.num_procs, self.num_tasks + 1], dtype=tf.float32)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(self.num_frames_per_proc):
            reward = rewards[i]
            discounted_sum = reward + self.gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]
        returns = tf.transpose(returns, perm=[1, 0, 2, 3])
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

    def computeH_proc_i(self, X: tf.Tensor, Xi: tf.Tensor, proc: tf.int32, agent: tf.int32, mu: tf.Tensor) -> tf.Tensor:
        # should be a relatively small calculation
        # computation with list comprehension should add some efficiency
        H_proc_i_agent_j = tf.TensorArray(dtype=tf.float32, size=self.num_tasks + 1)
        agent_value = self.lam * self.df(Xi[proc, 0])
        H_proc_i_agent_j = H_proc_i_agent_j.write(0, agent_value)
        for j in tf.range(1, self.num_tasks + 1):
            task_j_value = self.chi * self.dh(tf.math.reduce_sum(mu[:, j - 1] * X[agent, proc, j])) * mu[agent, j - 1]
            H_proc_i_agent_j = H_proc_i_agent_j.write(j, tf.squeeze(task_j_value))
        H_proc_i_agent_j = H_proc_i_agent_j.stack()
        return H_proc_i_agent_j

    def compute_H(self, X: tf.Tensor, Xi: tf.Tensor, agent: tf.int32, mu: tf.Tensor):
        H_ = tf.TensorArray(dtype=tf.float32, size=self.num_procs)
        for proc in tf.range(self.num_procs):
            H = self.computeH_proc_i(X, Xi, proc, agent, mu)
            H_ = H_.write(proc, H)
        H_ = H_.stack()
        return H_

    def compute_alloc_H_proc_i(self, X: tf.Tensor, mu: tf.Tensor, proc: tf.int32, task: tf.int32):
        return_val = self.chi * self.dh(tf.math.reduce_sum(mu[:, task - 1] * X[:, proc, task]))
        return return_val

    def compute_alloc_H(self, X: tf.Tensor, mu: tf.Tensor, task: tf.int32):
        H_ = tf.TensorArray(dtype=tf.float32, size=self.num_procs)
        for proc in tf.range(self.num_procs):
            H_ = H_.write(proc, self.compute_alloc_H_proc_i(X, mu, proc, task))
        H_ = H_.stack()
        return H_

    @tf.function
    def update_alloc_loss(self, X: tf.Tensor, mu: tf.Tensor):
        loss = tf.constant(0.0, dtype=tf.float32)
        for task in tf.range(1, self.num_tasks + 1):
            h = self.chi * self.compute_alloc_H(X, mu, task)  # return shape: (S, )
            # destroy flow
            h = tf.convert_to_tensor(h)
            agent_ini_val_x_alloc = tf.math.reduce_sum(tf.transpose(X[:, :, task]) * mu[:, task - 1])
            alloc_proc_loss_task_j = h * agent_ini_val_x_alloc
            loss += tf.reduce_mean(alloc_proc_loss_task_j)
        return loss

    def update_loss(self,
                    observations: tf.Tensor,
                    actions: tf.Tensor,
                    masks: tf.Tensor, returns: tf.Tensor,
                    advantages: tf.Tensor,
                    ii: tf.Tensor, *args):
        loss = tf.constant([0.0] * self.num_agents, dtype=tf.float32)
        for t in tf.range(self.recurrence):
            ix = ii + t
            # Construct a sub batch of experiences for the timestep t across all of the samples
            sb_obss_x_agents = tf.TensorArray(dtype=tf.float32, size=self.num_agents)
            sb_advs_x_agents = tf.TensorArray(dtype=tf.float32, size=self.num_agents)
            sb_rets_x_agents = tf.TensorArray(dtype=tf.float32, size=self.num_agents)
            sb_masks_x_agents = tf.TensorArray(dtype=tf.bool, size=self.num_agents)
            # compute the sub batch values for each of the agents
            for agent in tf.range(self.num_agents):
                iz = self.tf_2d_indices(agent, ii.shape[0], indices=ix)
                sb_obss_x_agents = sb_obss_x_agents.write(agent, tf.gather_nd(observations, indices=iz))
                sb_advs_x_agents = sb_advs_x_agents.write(agent, tf.gather_nd(advantages, indices=iz))
                sb_rets_x_agents = sb_rets_x_agents.write(agent, tf.gather_nd(returns, indices=iz))
                sb_masks_x_agents = sb_masks_x_agents.write(agent, tf.expand_dims(tf.cast(tf.gather_nd(masks, indices=iz), tf.bool), 1))
            sb_obss_x_agents = sb_obss_x_agents.stack()
            sb_advs_x_agents = sb_advs_x_agents.stack()
            sb_rets_x_agents = sb_rets_x_agents.stack()
            sb_masks_x_agents = sb_masks_x_agents.stack()
            ## Construct the sub batch mask from experiences
            #mask = tf.expand_dims(tf.cast(tf.gather(masks, indices=ix), tf.bool), 1)
            sb_obss_x_agents = tf.expand_dims(sb_obss_x_agents, 2)
            # print("mask ", sb_masks_x_agents.shape, "observations ", sb_obss_x_agents.shape)
            # Compute the actor and critic loss value for the respective agents
            actions_logits_x_agents, values_x_agents = [], []
            agent = tf.constant(0)
            for model in args:
                action_logits, values = model.call(sb_obss_x_agents[agent], sb_masks_x_agents[agent])
                actions_logits_x_agents.append(action_logits)
                values_x_agents.append(values)
                agent += tf.constant(1)
            value = tf.squeeze(values_x_agents)
            action_logits_t = tf.squeeze(actions_logits_x_agents)
            critic_loss = self.huber(value, sb_rets_x_agents)
            loss_update = tf.TensorArray(dtype=tf.float32, size=self.num_agents)
            for agent in tf.range(self.num_agents):
                iz = self.tf_2d_indices(agent, ii.shape[0], indices=ix)
                actions_selected = tf.gather_nd(actions, indices=iz)
                action_probs_t = tf.nn.softmax(action_logits_t[agent])
                x = tf.range(ii.shape[0])
                y = tf.transpose([x, actions_selected])
                action_probs = tf.gather_nd(action_probs_t, indices=y)
                actor_loss_agent = tf.math.reduce_mean(tf.math.log(action_probs) * sb_advs_x_agents[agent])
                critic_loss_agent = tf.nn.compute_average_loss(critic_loss[agent])
                loss_update = loss_update.write(agent, actor_loss_agent + critic_loss_agent)
            loss_update = loss_update.stack()
            loss += loss_update
        loss /= self.recurrence
        return loss


    # @tf.function
    def train_preprocess(self, initial_state: tf.Tensor, mu: tf.Tensor, *models) \
            -> [tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Call batch experiences for each of the agents and transform the data so that we don't mix
        experiences from each of the agent-environment interactions.
            * We require masks to know which tensor elements the model should ignore
            * We require values and returns for huber loss
            * We require advantages for actor loss
            * collect the batch of experiences for all environments over the range of time-steps
        """
        # collect batches for each of the agents
        all_actions = tf.TensorArray(dtype=tf.int32, size=self.num_agents)
        all_obss = tf.TensorArray(dtype=tf.float32, size=self.num_agents)
        all_values = tf.TensorArray(dtype=tf.float32, size=self.num_agents)
        all_rewards = tf.TensorArray(dtype=tf.float32, size=self.num_agents)
        all_masks = tf.TensorArray(dtype=tf.float32, size=self.num_agents)
        all_states = tf.TensorArray(dtype=tf.float32, size=self.num_agents)

        for agent, model in enumerate(models):
            # Collect the batch of experiences for an agent
            acts, obss, values, rewards, masks, state = \
                self.collect_batch(initial_state[agent], model, agent)
            all_actions = all_actions.write(agent, acts)
            all_obss = all_obss.write(agent, obss)
            all_values = all_values.write(agent, values)
            all_rewards = all_rewards.write(agent, rewards)
            all_masks = all_masks.write(agent, masks)
            all_states = all_states.write(agent, state)

        # Shape transformations:
        # T - timesteps, A - agents, S - samples, F - features
        # actions - (T, S) -> (A, T, S)
        # observations - (T, S, 1, F) -> (A, T, S, 1, F)
        # values - (T, S, tasks + 1) -> (A, T, S, tasks + 1)
        # rewards - (T, S, tasks + 1) -> (A, T, S, tasks + 1)
        # mask - (T, S) -> (A, T, S)
        # initial state - (S, F) -> (A, S, F)
        # log rewards - (S, tasks + 1) -> (A, S, tasks + 1)

        all_actions = all_actions.stack()
        all_obss = all_obss.stack()
        all_values = all_values.stack()
        all_rewards = all_rewards.stack()
        all_masks = all_masks.stack()
        all_states = all_states.stack()  # all state is the rolling initial state for each agent
        ### compute returns
        returns = self.get_expected_return(all_rewards)
        ini_values = all_values[:, 0, :, :]
        # shape of ini_values: (A, S, tasks + 1)
        advantages = self.compute_advantages(returns, all_values, ini_values, mu)
        ### NOTE: it is very important that this transpose is done so that we don't mix up any data
        # advantages shape: (A, S, T, 1) -> (A, S, T) -> (A, S * T)
        advantages = tf.reshape(tf.squeeze(advantages), [self.num_agents, self.num_procs * self.num_frames_per_proc])
        # print("advantages after transpose ", advantages.shape)
        ### concatenate actions together => actions reshape: T x A x S -> A x S x T -> A * S * T
        all_masks = tf.reshape(tf.transpose(all_masks, perm=[0, 2, 1]), [self.num_agents, self.num_procs * self.num_frames_per_proc])
        all_actions = tf.reshape(tf.transpose(all_actions, perm=[0, 2, 1]), [self.num_agents, self.num_procs * self.num_frames_per_proc])
        ### Concatenate the samples together => A x T x S x D -> A x S x T x D -> (A x (S * T) x D)
        returns = tf.reshape(tf.transpose(returns, perm=[0, 2, 1, 3]), [self.num_agents, self.num_procs * self.num_frames_per_proc, self.num_tasks + 1])
        all_obss = tf.reshape(tf.transpose(all_obss, perm=[0, 2, 1, 3, 4]), [self.num_agents, self.num_procs * self.num_frames_per_proc, initial_state.shape[-1]])
        all_values = tf.reshape(tf.transpose(all_values, perm=[0, 2, 1, 3]), [self.num_agents, self.num_procs * self.num_frames_per_proc, self.num_tasks + 1])
        ### destroy flow
        ini_values = tf.convert_to_tensor(ini_values)
        advantages = tf.convert_to_tensor(advantages)
        returns = tf.convert_to_tensor(returns)
        all_obss = tf.convert_to_tensor(all_obss)
        all_actions = tf.convert_to_tensor(all_actions)
        return all_obss, all_actions, all_masks, returns, all_values, advantages, all_states, ini_values

    @tf.function
    def train(self, initial_state: tf.Tensor, ii: tf.Tensor,
              mu: tf.Tensor, *models) -> [tf.Tensor, tf.Variable, tf.Tensor, tf.Tensor]:
        """
        :param initial_state: The initial states across the sample environments, shape: (samples x features)
        :param log_reward: A logging helper tensor which captures the current rewards
            across samples per episode, shape: (timesteps, samples, agents, (tasks + 1))
        :param ii: starting indices used in recurrent calculations
        """

        observations, acts, masks, returns, values, advantages, state, ini_values = \
            self.train_preprocess(initial_state, mu, *models)
        with tf.GradientTape() as tape:
            loss = self.update_loss(observations, acts, masks, returns, advantages, ii, *models)
        vars_l = [m.trainable_variables for m in models]
        grads_l = tape.gradient(loss, vars_l)
        grads_l_ = [x for y in grads_l for x in y]
        vars_l_ = [x for y in vars_l for x in y]
        self.opt.apply_gradients(zip(grads_l_, vars_l_))
        return state, loss, ini_values



