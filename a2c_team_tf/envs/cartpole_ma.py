"""
***This is a copy of the cartpole environment. The purpose of re-implementing this environment
    is to change the parameters of the cartpole for different agents. In this way we have a
    truely multiagent environment with different capabilities.
"""
import math
from gym.envs.classic_control.cartpole import CartPoleEnv


class CartPoleDefault(CartPoleEnv):
    def __init__(self):
        super().__init__()
        self.theta_threshold_radians = 30 * 2 * math.pi / 360
        self.masscart = 1.0
        self.length = 0.5


class CartPoleHeavyLong(CartPoleEnv):
    def __init__(self):
        super().__init__()
        self.theta_threshold_radians = 35 * 2 * math.pi / 360
        self.masscart = 1.2
        self.length = 0.6