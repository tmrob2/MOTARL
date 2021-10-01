import tensorflow as tf
from tensorflow.keras import layers
from typing import Tuple, List
from main import tf_env_step


class ActorCritic(tf.keras.Model):
    """Actor-critic Neural Network"""

    def __init__(self, n_actions: int, hidden_units: int):
        """
        :param n_actions: The number of actions in a model
        :param hidden_units: The number of hidden units
        """
        super().__init__()

        self.common = layers.Dense(hidden_units, activation="relu")
        self.actor = layers.Dense(n_actions)
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.actor(x), self.critic(x)


def run_episode(
        initial_state: tf.Tensor,
        model: tf.keras.Model,
        max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Runs a single episode to collect training data"""

    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
        # Convert state into a batched tensor (batch size=1)
        state = tf.expand_dims(state, 0)

        # Run the model and to get action probabilities and critic value
        action_logits_t, value = model(state)

        # Sample the next action from the action probability distribution
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)

        # Store critic values
        values = values.write(t, tf.squeeze(value))

        # Store log probability of action chosen
        action_probs = action_probs.write(t, action_probs_t[0, action])

        # Apply action to the environemtn to get the next state and reward
        state, reward, done = tf_env_step(action)
        state.set_shape(initial_state_shape)

        # Store rewards
        rewards = rewards.write(t, reward)

        if tf.cast(done, tf.bool):
            break

    # todo what does stack do?
    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()

    return action_probs, values, rewards

