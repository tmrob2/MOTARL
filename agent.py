from utils import Coord
from abc import abstractmethod
from grid import Grid
from enum import Enum


class AbstractAction(Enum):
    ...


class AbstractAgent:
    """
    An abstract agent which acts upon an environment. Used for constructing a specific agent in main
    must contain:
    Methods
    -------
    reset():
        A function which resets the agent attributes
    act(action):
        An action which causes a change in the environment
    """
    AGENT_ENERGY = 1.0
    STEP_VALUE = -0.02

    def __init__(self, position: Coord):
        self.pos: Coord = position
        self.starting_pos: Coord = position
        self.energy = self.AGENT_ENERGY
        self.visible = None
        self.dead = False
        self.success = False

    def reset(self):
        self.energy = self.AGENT_ENERGY
        self.pos = self.starting_pos
        self.visible = None
        self.dead = False
        self.success = False

    @abstractmethod
    def act(self, action):
        """
        Class method: takes an input action (non-deterministic agent) and results in a distribution of states

        Parameters
        ----------
        action: A non-deterministic action in the set of actions
        """
        ...

    @abstractmethod
    def set_visibility(self, grid):
        ...

