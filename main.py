import random
from abc import ABC
from agent import AbstractAgent, AbstractAction
from utils import Coord
import numpy as np
from grid import Grid
from environment import Environment
import visualisation
import tensorflow as tf
import statistics
import tqdm
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple


seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
eps = np.finfo(np.float32).eps.item()


class Action(AbstractAction):
    NOOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    DOWN = 4
    PICK = 5
    PLACE = 6


class Agent1(AbstractAgent, ABC):
    def __init__(self, position):
        super().__init__(position=position)

    def act(self, action: Action):
        """
        The outcomes of an agent acting
        """
        x, y = self.pos.x, self.pos.y
        if action == Action.FORWARD:
            self.pos = Coord(x, y + 1)
        elif action == Action.LEFT:
            self.pos = Coord(x - 1, y)
        elif action == Action.DOWN:
            self.pos = Coord(x, y - 1)
        elif action == Action.RIGHT:
            self.pos = Coord(x + 1, y)
        self.energy += self.STEP_VALUE

    def set_visibility(self, grid):
        self.visible = np.array(grid.visible(self.pos))


def env_step(actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns state, reward and done flag given an action
    Actions are an array because there are multiple agents
    Rewards (output) is an array because there are multiple rewards, agent costs, task rewards
    """
    # the return from env.step is an nd_array
    state, reward, done = env.step(actions.tolist())
    return state.astype(np.float32), reward.astype(np.float32), np.array(done, np.int32)


def tf_env_step(actions: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(env_step, [actions], [tf.float32, tf.float32, tf.int32])


if __name__ == '__main__':
    agents, tasks, grid_size, n_objs = 2, 2, (10, 10), 3
    env = Environment([Agent1(Coord(4, 2)), Agent1(Coord(5, 7))], tasks, grid_size, n_objs)
    env.reset()

    done = False
    for i in range(10):
        print("Episode {}".format(i))
        while not done:
            actions = [random.choice(list(Action)) for _ in range(agents)]  # todo check how seed works with random without np being called
            state, _, done = env.step(actions=actions)
            # print([s.__str__() for s in state])
        env.reset()
        done = False
