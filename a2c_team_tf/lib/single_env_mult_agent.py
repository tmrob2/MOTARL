import collections
import numpy as np
import tensorflow as tf
from typing import List, Tuple
import tqdm
from a2c_team_tf.utils.dfa import CrossProductDFA
from enum import Enum

class MAS:
    class Direction(Enum):
        MAXIMISE = 1
        MINIMISE = 2

    def __init__(
            self,
            seed,
            env,
            models: List[tf.keras.Model],
            dfas: List[CrossProductDFA],
            one_off_reward,
            num_tasks,
            num_agents,
            e: tf.Tensor,
            c: tf.Tensor,
            gamma:tf.float32=tf.constant(1.0, dtype=tf.float32),
            lam: tf.float32=tf.constant(1.0, dtype=tf.float32),
            chi: tf.float32=tf.constant(1.0, dtype=tf.float32),
            lr=5e-4,
            lr2=0.01,
            direction=Direction.MAXIMISE,
            render=False,
            debug=False,
            loss_debug=False):
        self.env = env
        self.seed = seed
        self.gamma = gamma
        self.lam = lam
        self.chi = chi
        self.e = e
        self.c = c
        self.models = models
        self.dfas = dfas
        self.num_agents = num_agents
        self.num_tasks = num_tasks
        self.render: bool = render
        self.debug = debug
        self.lr = lr
        self.lr2 = lr2
        self.direction = direction
        self.loss_debug = loss_debug
        self.one_off_reward = one_off_reward
        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    """def get_action(self, state: tf.Tensor, agent: tf.int32) -> [tf.Tensor, tf.Tensor]:
        action_probabilities, _ = self.models[agent](state)
        action_probabilities = tf.nn.softmax(action_probabilities)
        dist = tfp.distributions.Categorical(probs=action_probabilities, dtype=tf.int32)
        return dist.sample(), action_probabilities
    """

    def env_reset(self):
        self.env.seed(self.seed)
        state = self.env.reset()
        rtn_state = []
        for agent in range(self.num_agents):
            self.dfas[agent].reset()
            grid_state_component = state[agent].flatten()
            dfa_state_component = np.array(self.dfas[agent].progress)
            rtn_state.append(np.append(grid_state_component, dfa_state_component))
        return np.array(rtn_state)

    def tf_reset(self):
        return tf.numpy_function(self.env_reset, [], [tf.float32])

    def env_step(
            self,
            state: np.ndarray,
            actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # In this multiagent setting, the step reward will be a vector of rewards
        # for the number of agents in the system
        # the state will not be in the correct shape though because it doesn't have the
        state_new, step_reward, env_done, _ = self.env.step(actions)
        done = False
        # update the DFA
        rewards_ = np.zeros(shape=(self.num_agents, self.num_tasks + 1))
        dfa_finished = [False] * self.num_agents
        state_new_ = []
        for agent in range(self.num_agents):
            self.dfas[agent].next(self.env)
            # print(f"agent: {agent}, xprod state: {self.dfas[agent].product_state}")
            # agent task rewards
            task_rewards = self.dfas[agent].rewards(self.one_off_reward)
            state_task_rewards = [step_reward[agent]] + task_rewards
            rewards_[agent, :] = state_task_rewards
            dfa_finished[agent] = self.dfas[agent].done()
            state_new_.append(np.append(state_new[agent], np.array(self.dfas[agent].progress)))
        state_new_ = np.array(state_new_)
        if all(dfa_finished):
            if env_done:
                done = True

        return (state_new_.astype(np.float32),
                np.array(rewards_, np.float32),
                np.array(done, np.int32))

    def tf_env_step(self, state: tf.Tensor, actions: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self.env_step, [state, actions], [tf.float32, tf.float32, tf.int32])

    def get_expected_returns(
            self,
            rewards: tf.Tensor) -> tf.Tensor:
        """Compute expected returns per timestep"""
        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        # Start from the end of rewards and accumulate reward sums into the returns array
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)

        # discounted_sum = tf.constant(0.0)
        discounted_sum = tf.constant(tf.zeros([self.num_agents, self.num_tasks + 1]), dtype=tf.float32, name='discounted_sum')
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + self.gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]
        return returns

    def actor_critic_loss(
            self,
            action_probs: tf.Tensor,
            values: tf.Tensor,
            returns: tf.Tensor,
            ini_values: tf.Tensor,
            ini_values_i: tf.Tensor,
            agent: tf.int32,
            mu: tf.Tensor) -> [tf.Tensor, tf.Tensor, tf.Tensor]:
        H = self.compute_H(ini_values, ini_values_i, agent, self.lam, self.chi, mu)
        advantage = tf.matmul(returns - values, H)
        action_log_probs = tf.math.log(action_probs)
        actor_loss = tf.math.reduce_sum(action_log_probs * advantage)
        critic_loss = self.huber_loss(values, returns)

        return actor_loss + critic_loss, actor_loss, critic_loss

    def alloc_loss(self, ini_values: tf.Tensor, mu: tf.Tensor):
        H = self.compute_alloc_H(ini_values, mu)
        alloc_loss = tf.math.reduce_sum(H * tf.math.reduce_sum(mu * ini_values))
        return alloc_loss

    def compute_H(
            self,
            X: tf.Tensor,
            Xi: tf.Tensor,
            agent: tf.int32,
            lam: tf.float32,
            chi: tf.float32,
            mu: tf.Tensor) -> tf.Tensor:
        _, y = X.get_shape()
        H = [lam * self.df(Xi[0], self.c[agent])]
        for j in range(1, y):
            H.append(chi * self.dh(tf.math.reduce_sum(mu[:, j - 1] * X[:, j]), self.e[j - 1]) * mu[agent, j - 1])
        return tf.expand_dims(tf.convert_to_tensor(H), 1)

    def compute_alloc_H(self, X: tf.Tensor, mu):
        H = []
        for j in tf.range(1, self.num_tasks + 1):
            H.append(self.chi * self.dh(tf.math.reduce_sum(mu[:, j - 1] * X[:, j]), self.e[j - 1]))
        return tf.expand_dims(tf.convert_to_tensor(H), 1)

    def df(self, x: tf.Tensor, c: tf.float32) -> tf.Tensor:
        """derivative of mean squared error"""
        if self.direction.MAXIMISE:
            if tf.less_equal(x, c):
                return 2 * (x - c)
            else:
                return tf.convert_to_tensor(0.0)
        else:
            if tf.less_equal(c, x):
                return 2 * (c - x)
            else:
                return tf.convert_to_tensor(0.0)

    @staticmethod
    def dh(x: tf.Tensor, e) -> tf.Tensor:
        # print(f"x: {x}, e: {e}")
        if tf.less_equal(x, e): #  and tf.greater_equal(x, 0.0):
            return 2 * (x - e)
        else:
            return tf.convert_to_tensor(0.0)

    def run_episode(
            self,
            initial_state: tf.Tensor,
            max_steps: tf.int32) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        action_probabilities = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        initial_state_shape = initial_state.shape
        state = initial_state
        for t in tf.range(max_steps):
            # Convert stat into a batched tensor (batch size = 1)
            # Run the model and get the chosen action from the distribution generated from the
            # AC network for a particular agent
            actions = []
            values_t = []
            probs_t_agent = []
            for agent in tf.range(self.num_agents):
                state_tf = tf.expand_dims(state[agent], 0)  # TODO must expand the state with DFA progress, DFA status

                action_logits_t, values_ = self.models[agent](state_tf)
                # Sample the next action from the action probability distribution
                action = tf.random.categorical(action_logits_t, 1, dtype=tf.int32)[0, 0]
                action_probs_t = tf.nn.softmax(action_logits_t)
                # Store the actions to apply to the environment at the end of the agent loop
                actions.append(action)
                # Store the log probability of the chosen action
                values_t.append(tf.squeeze(values_))
                # log_prob_p = tf.math.log(action_probs_t[0, action_])
                probs_t_agent.append(action_probs_t[0, action])
            state_, reward, done = self.tf_env_step(state, actions)
            state = state_
            state.set_shape(initial_state_shape)
            rewards = rewards.write(t, reward)
            values = values.write(t, values_t)
            action_probabilities = action_probabilities.write(t, probs_t_agent)
            if tf.cast(done, tf.bool):
                break

            if self.render:
                self.env.render('human')

        action_probabilities = action_probabilities.stack()
        values = values.stack()
        rewards = rewards.stack()
        return action_probabilities, values, rewards

    def train_step(
            self,
            initial_state: tf.Tensor,
            max_steps_per_episode: tf.int32,
            mu: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            # Run an episode:
            action_probs, values, rewards = self.run_episode(initial_state, max_steps=max_steps_per_episode)
            # Calculate the expected returns
            returns = self.get_expected_returns(rewards)
            # Append the tensors to the respective lists

            # non-differentiated values
            ini_values = tf.convert_to_tensor(values[0])  # this will include the network's critic of all of the agents
            #loss_l = tf.Variable(tf.zeros([2], dtype=tf.float32))
            loss_l = []
            actor_l = []
            critic_l = []
            for agent in tf.range(self.num_agents):
                values_agent = values[:, agent, :]
                returns_agent = returns[:, agent, :]
                ini_values_agent = ini_values[agent, :]
                loss, a_loss, c_loss = self.actor_critic_loss(
                    action_probs[:, agent], values_agent, returns_agent,
                    ini_values, ini_values_agent, agent, mu)
                #loss_l[agent].assign(loss)
                loss_l.append(loss)
                actor_l.append(a_loss)
                critic_l.append(c_loss)

        vars_l = [m.trainable_variables for m in self.models]
        # tf.print("loss_l: ", loss_l)
        grads_l = tape.gradient(loss_l, vars_l)

        # Apply the gradients to the model's parameters
        grads_l_f = [x for y in grads_l for x in y]
        vars_l_f = [x for y in vars_l for x in y]
        self.optimizer.apply_gradients(zip(grads_l_f, vars_l_f))
        episode_reward = tf.math.reduce_sum(rewards, axis=0)
        # tf.print(f"Episode rewards: {episode_reward}")
        return episode_reward, ini_values, loss_l, actor_l, critic_l

    def learn(self, max_episodes, max_steps_per_episode, min_episodes_criterion):
        episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)
        kappa = tf.Variable(np.random.rand(self.num_tasks * self.num_agents), dtype=tf.float32)
        with tqdm.trange(max_episodes) as t:
            for i in t:
                initial_state = self.tf_reset()
                mu = tf.nn.softmax(tf.reshape(kappa, shape=[self.num_agents, self.num_tasks]), axis=0)
                episode_reward, ini_values, loss, aloss, closs = self.train_step(initial_state, max_steps_per_episode, mu)
                if i % 1000 == 0:
                    with tf.GradientTape() as tape:
                        mu = tf.nn.softmax(tf.reshape(kappa, shape=[self.num_agents, self.num_tasks]), axis=0)
                        allocator_loss = self.alloc_loss(ini_values, mu)
                    # Compute the gradient from the allocator loss
                    # tf.print(f"Allocator loss: {allocator_loss}")
                    # tf.print(f"kappa: {kappa}")
                    grads_kappa = tape.gradient(allocator_loss, kappa)
                    # tf.print(f"grads kappa: {grads_kappa}")
                    processed_grads = [-self.lr2 * g for g in grads_kappa]
                    kappa.assign_add(processed_grads)
                print_f_epsisode_reward = episode_reward.numpy().flatten()
                agents_mean_rewards = tf.math.reduce_mean(episode_reward, axis=0).numpy()
                episodes_reward.append(agents_mean_rewards)
                running_reward = np.mean(episodes_reward, axis=0)
                t.set_description(f"Episode: {i}")
                t.set_postfix(episode_reward=print_f_epsisode_reward, running_reward=running_reward, aloss=[x.numpy() for x in aloss], closs=[x.numpy() for x in closs])

                # Show the learned values, and learned allocation matrix every 20 steps
                if i % 50 == 0:
                    print()
                    for k in range(self.num_agents):
                        print(f'values at the initial state for model#{k}: {ini_values[k]}')
                    print(f"allocation matrix mu: \n{mu}")









