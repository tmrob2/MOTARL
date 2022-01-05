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

    def __init__(self, num_agents=2, gridsize=6, max_steps=100):
        self.num_agents = num_agents

        super().__init__(
            grid_size=gridsize,
            agent_view_size=gridsize,
            see_through_walls=True,
            max_steps=max_steps
        )

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


class TestEnv(BaseEnv):

    def __init__(self, numKeys=2, numBalls=2, numBoxes=1, max_steps=30):
        self.num_keys = numKeys
        self.num_balls = numBalls
        self.num_boxes = numBoxes
        super(TestEnv, self).__init__(max_steps=max_steps)

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


class FourRooms(BaseEnv):

    def __init__(self):
        super(FourRooms, self).__init__(gridsize=16, num_agents=4)

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, Door('red', is_locked=True))

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, Door('green', is_locked=True))

        # Randomize the player start position and orientation
        for agent in range(self.num_agents):
            self.place_agent()



