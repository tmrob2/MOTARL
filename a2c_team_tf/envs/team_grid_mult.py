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

    Note: An important point in using this environment in the context of object
    fountains is that objects can contaminate the environment, and an agent can
    essentially block other agents with objects maliciously"""
    def __init__(self, num_agents=2, width=4, height=4, numKeys=1, numBalls=1):
        self.num_agents = num_agents
        self.num_keys = numKeys
        self.num_balls = numBalls

        super().__init__(
            width=width,
            height=height,
            max_steps=50,
            agent_view_size=max(width, height),
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # instantiate the grid
        self.grid = Grid(width, height)
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        for key in range(self.num_keys):
            obj = Key('red')
            self.place_obj(obj)

        for ball in range(self.num_balls):
            obj = Ball('blue')
            self.place_obj(obj)

        for i in range(self.num_agents):
            self.place_agent()

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
        agent = Agent(color=color, can_overlap=agent_can_overlap)

        pos = self.place_obj(agent, top, size, max_tries=max_tries)

        if dir is None:
            dir = self._rand_int(0, 4)
        agent.dir = dir

        self.agents.append(agent)

        return pos

    def step(self, actions):
        # Each agent needs to produce an action
        assert len(actions) == len(self.agents)

        self.step_count += 1
        rewards = [0] * len(self.agents)

        # For each agent
        for agent_idx, agent in enumerate(self.agents):
            # Get the position in front of the agent
            fwd_pos = agent.front_pos

            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)

            # Get the action for this agent
            action = actions[agent_idx]

            # Rotate left
            if action == self.actions.left:
                rewards[agent_idx] -= 1
                agent.dir -= 1
                if agent.dir < 0:
                    agent.dir += 4

            # Rotate right
            elif action == self.actions.right:
                rewards[agent_idx] -= 1
                agent.dir = (agent.dir + 1) % 4

            # Move forward
            elif action == self.actions.forward:
                rewards[agent_idx] -= 1
                if fwd_cell == None or fwd_cell.can_overlap():
                    self.grid.set(*agent.cur_pos, None)
                    self.grid.set(*fwd_pos, agent)
                    agent.cur_pos = fwd_pos

            # Done action (not used by default)
            elif action == self.actions.wait:
                pass

            # Toggle/activate an object
            elif action == self.actions.toggle:
                rewards[agent_idx] -= 1
                if fwd_cell:
                    fwd_cell.toggle(self, fwd_pos)

            # Pick up an object
            elif action == self.actions.pickup:
                rewards[agent_idx] -= 1
                if fwd_cell and fwd_cell.can_pickup():
                    if agent.carrying is None:
                        agent.carrying = fwd_cell
                        agent.carrying.cur_pos = np.array([-1, -1])
                        self.grid.set(*fwd_pos, None)

            # Drop an object
            elif action == self.actions.drop:
                rewards[agent_idx] -= 1
                if not fwd_cell and agent.carrying:
                    self.grid.set(*fwd_pos, agent.carrying)
                    agent.carrying.cur_pos = fwd_pos
                    agent.carrying = None

            elif action == self.actions.wait:
                rewards[agent_idx] -= 0

            else:
                assert False, "unknown action"

        obss = self.gen_obss()

        obs_ = []
        for img in obss:
            obs_.append(img.flatten())
        return obs_, rewards, False, {}

    def reset(self):
        obs = MiniGridEnv.reset(self)
        obs_ = []
        for img in obs:
            obs_.append(img.flatten())
        return obs_



