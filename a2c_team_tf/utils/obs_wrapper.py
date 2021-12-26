import gym.core
from gym import spaces
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX
import numpy as np

class FlatObsWrapper(gym.core.ObservationWrapper):
    """Compatible with gym-minigrid, this wrapper returns a flat fully observable
    state representation of the environment"""
    def __init__(self, env, max_steps):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width * self.env.height * 3, ),
            dtype='uint8'
        )
        self.unwrapped.max_steps = max_steps

    def observation(self, obs):
        # observation is called in the step function
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])
        flattened_grid = full_grid.flatten()
        return flattened_grid

    def render(self, *args, **kwargs):
        """This removes the default visualization of the partially observable field of view."""
        kwargs['highlight'] = False
        return self.unwrapped.render(*args, **kwargs)
