from grid import Grid, GOAL_VALUE
from agent import AbstractAction
import numpy as np
from typing import List, Tuple, Optional, Dict
from utils import Coord


VISIBLE_RADIUS = 1


class EnvHistory:
    def __init__(self, grid, agents):
        self.grid = grid
        self.energy = [x.energy for x in agents]
        self.visibles = [x.visible for x in agents]


class AgentState:
    def __init__(self, energy, position: Coord):
        self.energy = energy
        self.position = position

    def to_ndarray(self):
        np.array([self.energy, self.position.x, self.position.y])

    def __str__(self):
        return "pos: ({},{}), e: {}".format(self.position.x, self.position.y, self.energy)


class Environment:
    def __init__(self, agents: List, m_tasks, grid_size, n_objs):
        self.grid = Grid(grid_size=grid_size, n_obj=n_objs)
        self.agents = agents
        self.t = 0
        self.history = []
        self.n_agents = len(agents)
        self.m_tasks = m_tasks

    def reset(self):
        """Start a new episode by resetting the grid and a agents"""
        self.grid.reset()
        list(map(lambda x: x.reset(), self.agents))
        #for i in self.agents:
        #    i.reset()
        self.t = 0
        self.history = []
        return self.visible_state

    @property
    def visible_state(self):
        """State of the system"""
        state_ls = [AgentState(energy=x.energy, position=x.pos).to_ndarray() for x in self.agents]
        state = np.concatenate(state_ls).ravel()
        return state

    def record_step(self):
        """Add the current state to history for display later"""
        grid = np.array(self.grid.grid)
        for agent in self.agents:
            grid[agent.pos.x, agent.pos.y] = agent.AGENT_ENERGY * 0.5
            agent.set_visibility(self.grid)
        self.history.append(EnvHistory(grid, self.agents))

    def step(self, actions: List):
        """Because this is a MAS we need to update the state of the system based on all of the agents"""
        task_rewards = [0] * self.m_tasks  # will be size j

        # old_positions = [x.pos for x in self.agents]

        # Todo at this point we also have to generate a word, or each agent must generate a
        #  word but we will leave this for now and work on the agents moving around

        for (i, (a, action)) in enumerate(zip(self.agents, actions)):
            if not a.dead and not a.success:
                a.act(action)
                value = self.grid.grid[a.pos.x, a.pos.y]
                a.energy += value
                if a.energy <= 0:
                    a.dead = True
                if value == GOAL_VALUE:
                    task_rewards[i] = 1
                    a.success = True
                # print("agent {}, action: {}, pos: ({}, {}), pos': ({}, {})".format(i, action, old_positions[i].x, old_positions[i].y, a.pos.x, a.pos.y))

        done = True if all(x.success or x.dead for x in self.agents) else False
        self.record_step()
        return self.visible_state, task_rewards, done



