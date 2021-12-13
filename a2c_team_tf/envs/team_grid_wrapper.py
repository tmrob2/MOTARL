from collections import defaultdict

import teamgrid.minigrid
from teamgrid.minigrid import *

class Point():
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

class TestEnv(MiniGridEnv):
    def __init__(self, num_agents=2, penalty=30, width=5, height=5):
        self.num_agents = num_agents
        self.penalty = penalty
        super().__init__(
            width=width,
            height=height,
            max_steps=100,
            agent_view_size=max(width, height)
        )

    def _gen_grid(self, width, height):
        # instantiate the grid
        self.grid = Grid(width, height)
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        # randomise the player start positions
        for i in range(1):
            obj = Ball(color='green')
            self.grid.set(1, 1, obj)

        for i in range(self.num_agents):
            self.place_agent()

    def detect_conflict(self, conflicting_positions):
        tally = defaultdict(list)
        for i, item in enumerate(conflicting_positions):
            if item is not None:
                tally[item].append(i)
        return ((key, locs) for key, locs in tally.items() if len(locs) > 1)

    def step(self, actions):
        # I guess actions are a vector now
        conflicting_positions = [None] * self.num_agents
        # conflict positions looks like: {(x,y): [a_1,a_2,...,a_k]}
        object_pickup_conflict = [None] * self.num_agents
        # conflict when objects are pickup up in the same time-step by multiple agents
        object_drop_conflict = [None] * self.num_agents
        rewards = [0] * self.num_agents
        # conflict when an agent attempts to drop a non-overlapping object
        for agent_idx, agent in enumerate(self.agents):
            # Get the contents of the cell in front of the agent, do not actually move the agents
            # just get a preview of what they are going to do.
            fwd_pos = agent.front_pos
            fwd_cell = self.grid.get(*fwd_pos)
            if actions[agent_idx] == self.actions.forward:
                if fwd_cell is None or fwd_cell.can_overlap():
                    conflicting_positions[agent_idx] = Point(fwd_pos[0], fwd_pos[1])
                elif fwd_cell.type == 'wall':
                    rewards[agent_idx] += 10
            elif actions[agent_idx] == self.actions.pickup:
                if fwd_cell and fwd_cell.can_pickup():
                    if agent.carrying is None:
                        object_pickup_conflict[agent_idx] = Point(fwd_pos[0], fwd_pos[1])
            elif actions[agent_idx] == self.actions.drop:
                if fwd_cell is None or fwd_cell.can_overlap():
                    if agent.carrying:
                        object_drop_conflict[agent_idx] = Point(fwd_pos[0], fwd_pos[1])
        collisions = self.detect_conflict(conflicting_positions)
        drop_violations = self.detect_conflict(object_drop_conflict)
        pickup_violations = self.detect_conflict(object_pickup_conflict)
        # collections is a generator, so we basically want to add some penalty to the penalty vector
        # in every position that is returned from the k, v in collisions
        info = {'collisions': [], 'drop_violations': [], 'pickup_violations': []}
        for k, v in list(collisions):
            # v will be a list of agents that collided with each other
            # if v: print(f"colliding agents: {v}")
            for agent in v:
                rewards[agent] += self.penalty
                info['collisions'].append(agent)
        for k, v in list(drop_violations):
            # if v: print(f"conflicting drops: {v}")
            for agent in v:
                rewards[agent] += self.penalty
                info['drop_violations'].append(agent)
        for k,v in list(pickup_violations):
            # if v: print(f"conflicting pickups")
            for agent in v:
                rewards[agent] += self.penalty
                info['pickup_violations'].append(agent)
        # We assume in this test environment that there are no natural end points
        # and there are no goals so done is always true
        observation, step_rewards, _, _ = MiniGridEnv.step(self, actions)
        rewards = [rewards[idx] + r_ for idx, r_ in enumerate(step_rewards)]
        return observation, rewards, True, info



