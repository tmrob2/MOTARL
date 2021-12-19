import gym
from a2c_team_tf.utils.obs_wrapper import FlatObsWrapper

def make_env(env_key, max_steps_per_episode, seed=None):
    env = gym.make(env_key)
    env.seed(seed)
    env.reset()
    env = FlatObsWrapper(env, max_steps_per_episode)
    return env