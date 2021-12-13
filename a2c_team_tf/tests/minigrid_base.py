import collections
import statistics
import numpy as np
import tensorflow as tf
import gym
import tqdm
from a2c_team_tf.utils import obs_wrapper
from a2c_team_tf.nets.base import Actor, Critic
from a2c_team_tf.lib.tf2_a2c_base import Agent
from tensorflow.keras.callbacks import ModelCheckpoint
from pyvirtualdisplay import Display


env = gym.make('Empty-5x5-v0')
seed = 123
env.seed(seed)
env.reset()
env = utils.FlatObsWrapper(env)
np.random.seed(seed)
tf.random.set_seed(seed)
min_episode_criterion = 50
max_episodes = 10000
max_steps_per_episode = 100

reward_threshold = 0.8
running_reward = 0

episodes_reward: collections.deque = collections.deque(maxlen=min_episode_criterion)
actor = Actor(env.action_space.n)
critic = Critic()
agent = Agent(env, actor, critic, gamma=1.0, alr=1e-4, clr=1e-4)
with tqdm.trange(max_episodes) as t:
    for i in t:
        initial_state = tf.constant(env.reset(), dtype=tf.float32)
        episode_reward = agent.train(initial_state, max_steps_per_episode)
        episodes_reward.append(episode_reward.numpy())
        running_reward = statistics.mean(episodes_reward)

        t.set_description(f"Episode: {i}")
        t.set_postfix(episode_reward=episode_reward.numpy(), running_reward=running_reward)

        if running_reward > reward_threshold and i >= min_episode_criterion:
            break

# Save the model(s)
tf.saved_model.save(actor, "/home/tmrob2/PycharmProjects/MORLTAP/saved_models/actor_model")
tf.saved_model.save(critic, "/home/tmrob2/PycharmProjects/MORLTAP/saved_models/critic_model")


