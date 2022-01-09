import copy
from collections import defaultdict
from teamgrid.minigrid import *


class Point():
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

class BaseEnv(MiniGridEnv):
    """The major difference in this environment to the original
    teamgrid environment is that world objects are a 'fountain' of
    objects in that they infinitely resupply, rewards are negative, done is
    always false as it is up to the DFAs to decide when the epsiode is over.

    Note: This environment is deceptively difficult to learn becuase it is actually dynamic"""

    def __init__(self, num_agents=2, gridsize=None, max_steps=100, width=None, height=None):
        self.num_agents = num_agents

        super().__init__(
            grid_size=gridsize,
            agent_view_size=gridsize if gridsize else max(height, width),
            see_through_walls=True,
            max_steps=max_steps,
            width=width,
            height=height
        )

    def place_agent(
        self,
        grid,
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
        agent.grid = grid

        pos = self.place_obj(agent, grid, top, size, max_tries=max_tries)

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


class TestEnv(BaseEnv):

    def __init__(self, numKeys=2, numBalls=2, numBoxes=1, max_steps=15):
        self.num_keys = numKeys
        self.num_balls = numBalls
        self.num_boxes = numBoxes
        super(TestEnv, self).__init__(max_steps=max_steps, gridsize=6)

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

        grid1 = copy.deepcopy(self.grid)
        grid2 = copy.deepcopy(self.grid)
        self.place_agent(grid1, top=(0, 0), color='purple')
        self.place_agent(grid2, top=(0, 0), color='green')

        self.toggled = False

class TestEnv2(BaseEnv):

    def __init__(self, numKeys=2, numBalls=2, numBoxes=1, max_steps=30):
        self.num_keys = numKeys
        self.num_balls = numBalls
        self.num_boxes = numBoxes
        super(TestEnv2, self).__init__(max_steps=max_steps, width=12, height=7)

    def _gen_grid(self, width, height):
        # instantiate the grid
        self.grid = Grid(width, height)
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.grid.vert_wall((3 * width) // 4, 0, height)

        #place a bridge and a door
        self.blue_door = Door('blue', is_locked=True)
        self.grid.set((3 * width) // 4, height // 2, None)
        self.place_obj(self.blue_door, self.grid, top=((3 * width) // 4, height // 2), size=(1, 1))

        self.grid.set(1, 1, Key('blue'))
        self.grid.set(2, 3, Key('blue'))
        self.grid.set(1, 5, Key('blue'))
        #self.grid.set(4, 4, Key('red'))
        #self.grid.set(4, 1, Ball('blue'))
        #self.grid.set(1, 4, Ball('blue'))

        #self.grid.set(1, 3, Box('grey'))
        #self.grid.set(4, 3, Box('grey'))

        grid1 = copy.deepcopy(self.grid)
        grid2 = copy.deepcopy(self.grid)

        self.place_agent(grid1, top=(1, 2), size=(1, 1), color='purple')
        self.place_agent(grid2, top=(1, 4), size=(1, 1), color='green')

        self.toggled = False

class DualDoors(BaseEnv):
    def __init__(self):
        super(DualDoors, self).__init__(width=19, height=11, max_steps=250, gridsize=None)


    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Leftmost wall
        self.grid.vert_wall(width // 3, 0, height)

        # Rightmost wall
        self.grid.vert_wall((2 * width) // 3, 0, height)

        # upper wall
        self.grid.horz_wall(0, height // 2, width)

        # lower wall
        #self.grid.horz_wall(0, (2 * height) // 3, (3 * width) // 4 - 1)

        ## Switch in the left room
        #self.grid.set(
        #    1, (3 * height) // 4,
        #    Switch(
        #        'blue',
        #        is_on=False,
        #        env=self,
        #        left=0,
        #        top=0,
        #        width=0,
        #        height=0,
        #    )
        #)
        #
        ## Switch in the right room
        #self.grid.set(
        #    (2 * width) // 3 + 2, (3 * height) // 4,
        #    Switch(
        #        'red',
        #        is_on=False,
        #        env=self,
        #        left=0,
        #        top=0,
        #        width=0,
        #        height=0,
        #    )
        #)

        # bridges
        self.grid.set(width // 2, height // 2, None)
        self.grid.set(width // 2, height // 3, None)
        self.grid.set(width // 2, height // 2, None)
        self.grid.set(width // 3, (2 * height) // 3, None)
        self.grid.set((2 * width) // 3, height // 3, None)

        # Door in the left room
        self.red_door = Door('red', is_locked=True)
        self.grid.set(width // 3, height // 4, None)
        self.place_obj(self.red_door, self.grid, top=(width // 3, height // 4), size=(1, 1))

        # Door in the right room
        self.blue_door = Door('blue', is_locked=True)
        self.grid.set((2 * width) // 3, (2 * height) // 3, None)
        self.place_obj(self.blue_door, self.grid, top=((2 * width) // 3, (2 * height) // 3), size=(1, 1))

        # place some red keys
        self.grid.set(width - 4, height // 4, Key('red'))
        self.grid.set(width // 2 - 2, (3 * height) // 4, Key('red'))

        # place some blue keys
        self.grid.set(width // 2, 2, Key('blue'))
        self.grid.set(width // 2, (3 * height) // 4 - 1, Key('blue'))

        # place some balls
        self.grid.set(2, height - 2, Ball('yellow'))
        self.grid.set(width // 2 + 1, height // 2 - 3, Ball('purple'))

        # place a green square
        self.grid.set(width - 2, 1, Goal())

        # place some boxes
        self.grid.set(4, height // 2 + 2, Box('yellow'))
        self.grid.set(width // 2 + 2, height // 4, Box('purple'))

        # Place one agent in each room
        grid1 = copy.deepcopy(self.grid)
        grid2 = copy.deepcopy(self.grid)
        self.place_agent(grid1, top=(width // 2, (3 * height) // 4), size=(1, 1), color='red')
        self.place_agent(grid2, top=(width // 2 - 2, height // 3 + 1), size=(1, 1), color='blue')
        #self.place_agent(self.grid, top=(width // 2 + 2, height // 3 + 2), size=(width // 3, height // 3), color='green')

        self.toggled = [False] * 2



