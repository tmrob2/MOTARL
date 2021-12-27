import tensorflow as tf
from tensorflow.keras import layers
from typing import Tuple, List


class ActorCritic(tf.keras.Model):
    """Actor-critic Neural Network"""

    def __init__(self, n_actions: int, hidden_units: int, num_tasks: int, name: str):
        """
        :param n_actions: The number of actions in a model
        :param hidden_units: The number of hidden units
        :param name
        """
        super().__init__()
        self.fc1 = layers.Dense(hidden_units, activation="relu")
        self.actor = layers.Dense(n_actions)
        self.critic = layers.Dense(num_tasks + 1)  # tasks + the agent
        self.model_name = name

    def __call__(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.fc1(inputs)
        return self.actor(x), self.critic(x)


class Actor(tf.keras.Model):
    def __init__(self, num_actions, recurrent=False):
        super().__init__()
        self.recurrent = recurrent
        if recurrent:
            self.lstm = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(64))
        self.fc1 = tf.keras.layers.Dense(64, activation='tanh')
        self.fc2 = tf.keras.layers.Dense(64, activation='tanh')
        # self.fc3 = tf.keras.layers.Dense(32, activation='tanh')
        self.a = tf.keras.layers.Dense(num_actions, activation=None)

    def call(self, input, mask=None):
        if self.recurrent:
            x = self.lstm(input, mask=mask)
            x = self.fc1(x)
        else:
            x = self.fc1(input)
        x = self.fc2(x)
        # x = self.fc3(x)
        x = self.a(x)
        return x


class Critic(tf.keras.Model):
    def __init__(self, num_tasks=0, recurrent=False):
        super().__init__()
        self.recurrent = recurrent
        if recurrent:
            self.lstm = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(64), return_sequences=True)
        self.fc1 = tf.keras.layers.Dense(64, activation='tanh')
        self.fc2 = tf.keras.layers.Dense(64, activation='tanh')
        # self.fc3 = tf.keras.layers.Dense(32, activation='tanh')
        self.c = tf.keras.layers.Dense(num_tasks + 1, activation=None)

    def call(self, input, mask=None):
        if self.recurrent:
            x = self.lstm(input, mask=mask)
            x = self.fc1(x)
        else:
            x = self.fc1(input)
        x = self.fc2(x)
        # x = self.fc3(x)
        x = self.c(x)
        return x

class ActorCrticLSTM(tf.keras.Model):
    def __init__(self, num_actions, num_tasks=0, recurrent=False):
        super().__init__()
        self.recurrent = recurrent
        if recurrent:
            self.lstm = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(64), return_sequences=True)
        self.afc1 = tf.keras.layers.Dense(64, activation='tanh')
        # self.afc2 = tf.keras.layers.Dense(64, activation='tanh')
        self.a = tf.keras.layers.Dense(num_actions, activation=None)

        self.cfc1 = tf.keras.layers.Dense(64, activation='tanh')
        # self.cfc2 = tf.keras.layers.Dense(64, activation='tanh')
        self.c = tf.keras.layers.Dense(num_tasks + 1, activation=None)

    def call(self, input, mask=None):
        if self.recurrent:
            x = self.lstm(input, mask=mask)
        else:
            x = input
        a = self.afc1(x)
        a = self.a(a)

        c = self.cfc1(x)
        c = self.c(c)
        return a, c


