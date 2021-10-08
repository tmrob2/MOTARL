import gym
import tqdm
from model import ActorCritic
import tensorflow as tf
import motaplib
from gym.envs.registration import register
import numpy as np

seed = 103
# tf.random.set_seed(seed)
# np.random.seed(seed)

env = gym.make('Empty-grid-6x6-v0')
step_rew0 = 10
task_prob0 = 0.8
NUMAGENTS, TASKS, GRIDSIZE, N_OBJS = 2, 1, (5, 5), 3
max_steps_per_episode = 1000
gamma = 1.0
mu = 1.0 / NUMAGENTS # fixed even probability of allocating each task to each agent
lam = 1.0
chi = 1.0
c = step_rew0
e = task_prob0 # task reward threshold
num_actions = env.action_space.n
num_hidden_units = 128
models = [ActorCritic(num_actions, num_hidden_units, TASKS, name="AC{}".format(i)) for i in range(NUMAGENTS)]
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

