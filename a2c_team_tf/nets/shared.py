import tensorflow as tf
from tensorflow.keras import layers
from typing import Tuple, List


# Todo we can also try separating out the shared network and experiment with stability
class ActorCritic(tf.keras.Model):
    """Actor-critic Neural Network"""

    def __init__(self, n_actions: int, hidden_units: int, m_tasks: int, name: str):
        """
        :param n_actions: The number of actions in a model
        :param hidden_units: The number of hidden units
        :param name
        """
        super().__init__()

        self.common = layers.Dense(hidden_units, activation="relu")
        self.actor = layers.Dense(n_actions)
        self.critic = layers.Dense(m_tasks + 1)  # tasks + the agent
        self.model_name = name

    def __call__(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.actor(x), self.critic(x)


class AllocatorNetwork(tf.keras.Model):
    """A network which learns the parameters of the allocator"""

    def __init__(self, m_tasks: int, n_agents: int):
        pass
