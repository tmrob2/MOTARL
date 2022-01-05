from collections import defaultdict
from teamgrid.minigrid import *


class Point():
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

class TestEnv(MiniGridEnv):
    """The major difference in this environment to the original
    teamgrid environment is that world objects are a 'fountain' of
    objects in that they infinitely resupply, rewards are negative, done is
    always false as it is up to the DFAs to decide when the epsiode is over.

    Note: This environment is deceptively difficult to learn becuase it is actually dynamic"""

    def __init__(self, num_agents=2, gridsize=6, numKeys=2, numBalls=2, numBoxes=2):
        self.num_agents = num_agents
        self.num_keys = numKeys
        self.num_balls = numBalls
        self.num_boxes = numBoxes

        super().__init__(
            grid_size=gridsize,
            agent_view_size=gridsize,
            see_through_walls=True,
            max_steps=100
        )

    def _gen_grid(self, width, height):
        # instantiate the grid
        self.grid = Grid(width, height)
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.grid.set(1, 1, Key('red'))
        self.grid.set(4, 4, Key('red'))
        self.grid.set(4, 1, Ball('blue'))
        self.grid.set(1, 4, Ball('blue'))

        self.grid.set(1, 3, Box('grey'))
        self.grid.set(4, 3, Box('grey'))

        self.place_agent(top=(0, 0), color='purple')
        self.place_agent(top=(0, 0), color='green')

        self.toggled = False

    def place_agent(
        self,
        top=None,
        size=None,
        dir=None,
        color=None,
        max_tries=math.inf,
        agent_can_overlap=True
    ):
        """
        Set the agent's starting point at an empty position in the grid
        """

        free_colors = COLOR_NAMES[:]
        for agent in self.agents:
            if agent.color in free_colors:
                free_colors.remove(agent.color)

        if color is None:
            # Pick random agent color
            if len(free_colors) == 0:
                free_colors = COLOR_NAMES[:]
            color = self._rand_elem(free_colors)

        assert color in free_colors
        ### An additional line so that agents can overlap and environments are not competitive
        agent = Agent(color=color)

        pos = self.place_obj(agent, top, size, max_tries=max_tries)

        if dir is None:
            dir = self._rand_int(0, 4)
        agent.dir = dir

        self.agents.append(agent)

        return pos

    def step(self, actions):
        obs, rewards, done, _ = MiniGridEnv.step(self, actions)

        obs_ = []
        for img in obs:
            obs_.append(img.flatten())
        return obs_, rewards, False, {}

    def reset(self):
        obs = MiniGridEnv.reset(self)
        obs_ = []
        for img in obs:
            obs_.append(img.flatten())
        return obs_



