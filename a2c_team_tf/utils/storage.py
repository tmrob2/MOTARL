def collect_offline_batch():
    """Not in use at this point"""
    # states = []
    # max_steps = 100
    # mask = tf.ones([max_steps], dtype=tf.int32)
    # calc_mask = tf.Variable(tf.zeros(max_steps, tf.int32), dtype=tf.int32)
    # for i in tf.range(max_steps):
    #     action_logits_t = self.actor(state)
    #     action = tf.random.categorical(action_logits_t, 1)[0, 0]
    #     state, reward, done = self.agent.tf_env_step2(action)
    #     if done:
    #         states.append(prev_obs)
    #         calc_mask[i].assign(1)
    #     else:
    #         calc_mask[i].assign(0)
    #         states.append(state)
    #     prev_obs = state
    #     state = tf.expand_dims(state, 0)
    # dataset = tf.data.Dataset.from_tensor_slices(states)
    # dataset = dataset.batch(batch_size=5)
    # mask_ = mask - calc_mask
    pass