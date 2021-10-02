import tensorflow as tf
from tensorflow.keras import layers
from typing import Tuple, List


class ActorCritic(tf.keras.Model):
    """Actor-critic Neural Network"""

    def __init__(self, n_actions: int, n_agents: int, m_tasks: int, hidden_units: int):
        """
        :param n_actions: The number of actions in a model
        :param hidden_units: The number of hidden units
        """
        super().__init__()

        self.common = layers.Dense(hidden_units, activation="relu")
        self.actor = layers.Dense(n_actions)
        # We add an extra output layer which will be used to derive the parameters τ for the
        # deterministic allocator
        self.allocator = layers.Dense(n_agents)
        self.critic = layers.Dense(m_tasks)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.actor(x), self.allocator(x), self.critic(x)
