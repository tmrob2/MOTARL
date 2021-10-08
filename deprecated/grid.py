import numpy as np
import random
from utils import Coord

GOAL_VALUE = 1000
WALL_VALUE = -10
VISIBLE_RADIUS = 1
MAX_REWARD = 0.5
MIN_REWARD = -1


class Grid:
    def __init__(self, grid_size: (int, int) = (10, 10), n_obj=5):
        self.grid_size = grid_size
        self.n_obj = n_obj
        self.grid = None

    def reset(self):
        padded_size_x = self.grid_size[0] + 2 * VISIBLE_RADIUS
        padded_size_y = self.grid_size[1] + 2 * VISIBLE_RADIUS
        self.grid = np.zeros((padded_size_x, padded_size_y))

        # Edges
        self.grid[0:VISIBLE_RADIUS, :] = WALL_VALUE
        self.grid[-1*VISIBLE_RADIUS, :] = WALL_VALUE
        self.grid[:, 0:VISIBLE_RADIUS] = WALL_VALUE
        self.grid[:, -1*VISIBLE_RADIUS:] = WALL_VALUE

        # randomly place objects
        for _ in range(self.n_obj):
            rx = random.randint(0, self.grid_size[0] - 1) + VISIBLE_RADIUS
            ry = random.randint(0, self.grid_size[1] - 1) + VISIBLE_RADIUS
            self.grid[rx, ry] = random.random() * (MAX_REWARD - MIN_REWARD) + MIN_REWARD

        # Return point:
        S = VISIBLE_RADIUS
        E = self.grid_size[0] + VISIBLE_RADIUS - 1
        corners = [(E, E), (S, E), (E, S), (S, S)]
        goal = corners[random.randint(0, len(corners) - 1)]
        self.grid[goal] = GOAL_VALUE

    def visible(self, pos: Coord):
        x, y = pos.x, pos.y
        return self.grid[x - VISIBLE_RADIUS: x + VISIBLE_RADIUS + 1, y - VISIBLE_RADIUS: y + VISIBLE_RADIUS + 1]



