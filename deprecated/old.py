 todo rewrite run episode according to the test function test_run_episode
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