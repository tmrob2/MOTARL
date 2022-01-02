import copy
import numpy as np
from typing import Tuple, List
import tensorflow as tf
from a2c_team_tf.utils.parallel_envs_team import ParallelEnv
from a2c_team_tf.utils.env_utils import make_env


class MORLTAP:
    def __init__(self, envs, models, num_agents,
                 num_tasks, xdfas, one_off_reward,
                 e, c, chi, lam,
                 gamma=1.0, lr=1e-4, lr2=1e-4, entropy_coef=0.001,
                 seed=None, num_procs=10, num_frames_per_proc=100,
                 recurrence=1, max_eps_steps=100, env_key=None, flatten_env=False):
        self.recurrent = recurrence > 1
        self.recurrence = recurrence
        self.num_agents = num_agents
        self.envs: ParallelEnv = ParallelEnv(
                envs,
                xdfas,
                one_off_reward,
                num_agents,
                seed)
        if env_key:
            self.renv = make_env(
                env_key=env_key,
                max_steps_per_episode=max_eps_steps,
                seed=seed,
                apply_flat_wrapper=flatten_env)
        self.models = models
        self.num_tasks = num_tasks
        self.dfas = xdfas
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
        assert self.recurrent or self.recurrence == 1
        assert num_frames_per_proc % recurrence == 0

    def render_reset(self):
        if self.seed:
            self.renv.seed(self.seed)
        state = self.renv.reset()
        # reset the product DFA
        [d.reset() for d in self.dfas[0]]
        # initial_state = np.append(state, np.array(self.dfas.progress, dtype=np.float32))
        initial_state = np.array([np.append(state[k], self.dfas[0][k].progress) for k in range(self.num_agents)], dtype=np.float32)
        return initial_state

    def tf_render_reset(self):
        return tf.numpy_function(self.render_reset, [], [tf.float32])

    def call_models(self, state: tf.Tensor, *args):
        actions_logits_x_agents = tf.TensorArray(dtype=tf.float32, size=self.num_agents)
        values_x_agents = tf.TensorArray(dtype=tf.float32, size=self.num_agents)
        ix = tf.constant(0, dtype=tf.int32)
        for model in args:
            actions_logits, values = model(state[ix])
            actions_logits_x_agents = actions_logits_x_agents.write(ix, actions_logits)
            values_x_agents = values_x_agents.write(ix, values)
            ix += tf.constant(1, dtype=tf.int32)
        actions_logits_x_agents = actions_logits_x_agents.stack()
        values_x_agents = values_x_agents.stack()
        return actions_logits_x_agents, values_x_agents

    def reset(self):
        states = []
        states.append(self.envs.reset())
        return np.array(states, dtype=np.float32)

    def tf_reset2(self):
        return tf.numpy_function(self.reset, [], [tf.float32])

    def render_env_step(self, action: np.array) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns state, reward, done flag given an action"""
        state, reward, done, info = self.renv.step(action)
        # where we define the DFA steps
        [self.dfas[0][k].next(self.renv) for k in range(self.num_agents)]
        task_rewards = [d_.rewards(self.one_off_reward) for d_ in self.dfas[0]]
        agent_reward = [0] * self.num_agents
        if all(d.done() for d in self.dfas[0]):
            # Assign the agent reward
            done = True
        else:
            # agent_reward = reward
            done = False
        rewards_ = np.array([agent_reward] + task_rewards)
        state_ = np.array([np.append(state[k], self.dfas[0][k].progress) for k in range(self.num_agents)])
        state_ = np.expand_dims(state_, 1)
        return (
            state_.astype(np.float32),
            np.array(rewards_, np.float32),
            np.array(done, np.int32))

    def tf_render_env_step(self, action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self.render_env_step, [action], [tf.float32, tf.float32, tf.int32])

    def env_step(self, actions: np.array):
        actions = actions.transpose()
        state, reward, done = self.envs.step(actions)
        state = state.transpose(1, 0, 2)
        if self.recurrent:
            state = np.expand_dims(state, 2)
        return state.astype(np.float32), reward.astype(np.float32), done.astype(np.int32)

    def tf_env_step(self, actions: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self.env_step, [actions], [tf.float32, tf.float32, tf.int32])

    def render_episode(self, initial_state: tf.Tensor, max_steps: tf.int32, *args):
        state = initial_state
        # initial_state_shape = initial_state.shape
        for _ in tf.range(max_steps):
            self.renv.render('human')
            # Run the model to get an action probability distribution
            action_logits_t, _ = self.call_models(state, *args)
            if self.recurrent:
                action_logits_t = tf.squeeze(action_logits_t, axis=1)
            actions = self.collect_actions(action_logits_t)
            # Apply the action to the environment to get the next state and reward
            state, rewards, done = self.tf_render_env_step(actions)
            state = tf.expand_dims(state, 1)
            if tf.cast(done, tf.bool):
                break

    def collect_actions(self, action_logits: tf.Tensor) -> tf.Tensor:
        """
        :param action_logits: actions logits is a tensor of shape (agents, samples, env_action_space)
        """
        actions = tf.TensorArray(dtype=tf.int32, size=self.num_agents)
        for idx in tf.range(self.num_agents):
            action_set = tf.random.categorical(action_logits[idx, :, :], num_samples=1, dtype=tf.int32)
            action_set = tf.squeeze(action_set)
            actions = actions.write(idx, action_set)
        actions = actions.stack()
        return actions


    @tf.function
    def collect_batch(self, initial_obs: tf.Tensor, log_reward: tf.Tensor, *args):
        """Collects rollouts and computes advantages
        The expected shape of log_rewards is (samples, num_agents, tasks + 1)
        Runs several environments concurrently, the next actions are computed in a batch
        for all environments at the same time. The rollouts and the advantages from all
        environments are concatenated together
        Expected shapes:
        T - timesteps, A - agents, S - samples, F - features
        actions - (T, A, S)
        observations - (T, S, A, F)
        values - (T, A, S, tasks + 1)
        rewards - (T, S, A, tasks + 1)
        mask - (T, S)
        initial state - (S, A, F)
        log rewards - (S, A, tasks + 1)
        In this way we keep the model inputs for a batch discrete.
        """
        #print("initial state shape ", initial_obs.shape)
        #print("log reward shape ", log_reward.shape)
        observations = tf.TensorArray(dtype=tf.float32, size=self.num_frames_per_proc)
        selected_actions = tf.TensorArray(dtype=tf.int32, size=self.num_frames_per_proc)
        values = tf.TensorArray(dtype=tf.float32, size=self.num_frames_per_proc)
        rewards = tf.TensorArray(dtype=tf.float32, size=self.num_frames_per_proc)
        masks = tf.TensorArray(dtype=tf.float32, size=self.num_frames_per_proc)
        running_rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        log_reward_shape = log_reward.shape
        state = initial_obs
        state_shape = initial_obs.shape
        log_reward_counter = tf.constant(0, dtype=tf.int32)
        for i in tf.range(self.num_frames_per_proc):
            observations = observations.write(i, state)
            action_logits_t_x_agents, value_x_agents = self.call_models(state, *args)
            # value = self.critic(state)
            values = values.write(i, value_x_agents)
            # action_logits [samples, 1, actions] -> [samples, actions]
            action_logits_t_x_agents = tf.squeeze(action_logits_t_x_agents)
            # actions = tf.random.categorical(action_logits_t, num_samples=1, dtype=tf.int32)
            actions = self.collect_actions(action_logits_t_x_agents)
            selected_actions = selected_actions.write(i, actions)
            state, reward_, done_ = self.tf_env_step(actions)
            state.set_shape(state_shape)
            mask = tf.constant(1.0, dtype=tf.float32) - tf.cast(done_, dtype=tf.float32)
            masks = masks.write(i, mask)
            # print(f"mask shape: {mask.shape}, state shape: {state.shape}, log reward shape: {log_reward.shape}, reward shape: {reward_.shape}")
            reward_ = tf.transpose(reward_, perm=[1, 0, 2])
            log_reward += reward_
            log_reward.set_shape(log_reward_shape)
            for idx in tf.range(self.num_procs):
                if tf.cast(done_[idx], tf.bool):
                    running_rewards = running_rewards.write(log_reward_counter, log_reward[:, idx])
                    log_reward_counter += tf.constant(1, dtype=tf.int32)
            log_reward = mask * tf.transpose(log_reward, perm=[0, 2, 1])
            log_reward = tf.transpose(log_reward, perm=[0, 2, 1])
            log_reward.set_shape(log_reward_shape)
            rewards = rewards.write(i, reward_)
        ## dim labels:
        ##   T - timesteps (the number of experiences recorded)
        ##   S - samples (generated by parallel env procs)
        ##   D - tasks + 1  (agent)
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
                sample_advantage = tf.matmul(returns[:, agent, p] - values[:, agent, p], tf.expand_dims(H_agent_vals[agent, p], 1))
                adv_proc = adv_proc.write(p, sample_advantage)
            adv_proc = adv_proc.stack()
            advantages = advantages.write(agent, adv_proc)
        advantages = advantages.stack()
        return advantages

    def get_expected_return(self, rewards: tf.Tensor) -> tf.Tensor:
        """Expects the shape of rewards to be (steps, proc_sample, tasks + 1)"""
        # Compute the expected returns per time-step
        n = tf.shape(rewards)[0]  # index 1 is the number of experiences collected
        returns = tf.TensorArray(dtype=tf.float32, size=n)
        # Start from the end of the rewards and accumulate reward sums into
        # the returns array
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.zeros([self.num_agents, self.num_procs, self.num_tasks + 1], dtype=tf.float32)
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
            return 2 * (self.c - x)
        else:
            return tf.convert_to_tensor(0.0)

    def dh(self, x: tf.Tensor) -> tf.Tensor:
        if tf.less_equal(x, self.e):
            return 2 * (self.e - x)
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

    @tf.function
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
            # compute the sub batch values for each of the agents
            for agent in tf.range(self.num_agents):
                iz = self.tf_2d_indices(agent, ii.shape[0], indices=ix)
                sb_obss_x_agents = sb_obss_x_agents.write(agent, tf.gather_nd(observations, indices=iz))
                sb_advs_x_agents = sb_advs_x_agents.write(agent, tf.gather_nd(advantages, indices=iz))
                sb_rets_x_agents = sb_rets_x_agents.write(agent, tf.gather_nd(returns, indices=iz))
            sb_obss_x_agents = sb_obss_x_agents.stack()
            sb_advs_x_agents = sb_advs_x_agents.stack()
            sb_rets_x_agents = sb_rets_x_agents.stack()
            # Construct the sub batch mask from experiences
            mask = tf.expand_dims(tf.cast(tf.gather(masks, indices=ix), tf.bool), 1)
            # Compute the actor and critic loss value for the respective agents
            actions_logits_x_agents, values_x_agents = [], []
            agent = tf.constant(0)
            for model in args:
                action_logits, values = model.call(sb_obss_x_agents[agent], mask)
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
                actor_loss_agent = -tf.math.reduce_mean(tf.math.log(action_probs) * sb_advs_x_agents[agent])
                critic_loss_agent = tf.nn.compute_average_loss(critic_loss[agent])
                loss_update = loss_update.write(agent, actor_loss_agent - critic_loss_agent)
            loss_update = loss_update.stack()
            loss += loss_update
        loss /= self.recurrence
        return loss

    @tf.function
    def train_preprocess(self, initial_state: tf.Tensor, log_reward: tf.Tensor, ii: tf.Tensor, mu: tf.Tensor, *args):
        # We require masks to know which tensor elements the model should ignore
        # We require values and returns for huber loss
        # We require advantages for actor loss
        # collect the batch of experiences for all environments over the range of time-steps
        acts, obss, values, rewards, masks, state, running_rewards, log_reward = \
            self.collect_batch(initial_state, log_reward, *args)

        returns = self.get_expected_return(rewards)
        ini_values = values[0, :, :]
        advantages = self.compute_advantages(returns, values, ini_values, mu)
        ## concatenate actions together => actions reshape: T x A x S -> A x S x T -> A * S * T
        masks = tf.reshape(tf.transpose(masks), [-1])
        acts = tf.transpose(acts, perm=[1, 2, 0])
        acts = tf.reshape(acts, [self.num_agents, self.num_frames_per_proc * self.num_procs])
        # # Concatenate the samples together =>
        returns = tf.transpose(returns, perm=[2, 1, 0, 3])
        obss = tf.transpose(obss, perm=[1, 2, 0, 3, 4])
        # # values/rewards/returns reshape: T x S x A x D -> A x S x T x D -> (A * S * T) x D
        # # values = tf.reshape(values, [values.shape[0] * values.shape[1] * values.shape[2], values.shape[3]])
        returns = tf.reshape(returns, [self.num_agents, ii.shape[0] * self.num_frames_per_proc, self.num_tasks + 1])
        observations = tf.reshape(obss, [self.num_agents, self.num_procs * self.num_frames_per_proc, 1,
                                         initial_state.shape[-1]])
        advantages = tf.reshape(tf.squeeze(advantages), [self.num_agents, self.num_frames_per_proc * self.num_procs])
        # print()
        # print("post transpose and reshape")
        # print(f"returns shape: {returns.shape}, advantages shape {advantages.shape}")
        # print(f"masks: {masks.shape}, actions: {acts.shape}, obss: {observations.shape}")

        ## # destroy flow
        ini_values = tf.convert_to_tensor(ini_values)
        advantages = tf.convert_to_tensor(advantages)
        returns = tf.convert_to_tensor(returns)
        observations = tf.convert_to_tensor(observations)
        acts = tf.convert_to_tensor(acts)
        return observations, acts, masks, returns, values, advantages, state, \
               log_reward, running_rewards, ini_values

    @tf.function
    def train(self, initial_state: tf.Tensor, log_reward: tf.Tensor, ii: tf.Tensor, mu: tf.Tensor, *args):
        """
        :param initial_state: The initial states across the sample environments, shape: (samples x features)
        :param log_reward: A logging helper tensor which captures the current rewards
            across samples per episode, shape: (timesteps, samples, agents, (tasks + 1))
        :param ii: starting indices used in recurrent calculations
        """

        observations, acts, masks, returns, values, advantages, state, log_reward, \
            running_rewards, ini_values = self.train_preprocess(initial_state, log_reward, ii, mu, *args)
        with tf.GradientTape() as tape:
            loss = self.update_loss(observations, acts, masks, returns, advantages, ii, *args)
        vars_l = [m.trainable_variables for m in self.models]
        grads_l = tape.gradient(loss, vars_l)
        grads_l_ = [x for y in grads_l for x in y]
        vars_l_ = [x for y in vars_l for x in y]
        self.opt.apply_gradients(zip(grads_l_, vars_l_))
        return state, log_reward, running_rewards, loss, ini_values