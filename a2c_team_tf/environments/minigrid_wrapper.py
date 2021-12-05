from gym_minigrid.envs import RoomGrid, EmptyEnv
from gym_minigrid.minigrid import *
from gym.envs.registration import register
from gym_minigrid.wrappers import *
import random


def convert_to_flat_and_full(env):
    """A function which manipulates the environment to give outputs which our A2C algo expects"""
    env_fobs = FullyObsWrapper(env)
    flat_env = FlatObsWrapper(env_fobs)
    return flat_env


class ObjRoom(MiniGridEnv):
    def __init__(
            self,
            size=4,
            num_keys=1,
            num_balls=1,
            door_pos=None,
            types=None
    ):
        self.num_objs = num_keys + num_balls
        self.num_keys = num_keys
        self.num_balls = num_balls
        self.types = ['ball', 'key'] if types is None else types
        self.door_pos = door_pos
        super().__init__(
            grid_size=size,
            max_steps=2000,
            see_through_walls=True
        )

    def _reward(self):
        return 10.0

    def _gen_grid(self, height, width):
        self.grid = Grid(height, width)
        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # Place a goal square in the bottom-right corner
        #self.put_obj(Goal(), width - 2, height - 2)
        if self.door_pos is not None:
            self.grid.set(*self.door_pos, Door('red'))

        types = self.types

        objs = []

        # For each object to be generated,
        # todo cannot be random, must be statically set because we sample a number of environments
        for key in range(self.num_keys):
            obj = Key('red')
            self.place_obj(obj)
            objs.append(obj)

        for ball in range(self.num_balls):
            obj = Ball('blue')
            self.place_obj(obj)
            objs.append(obj)

        # Randomize the player start position and orientation
        self.place_agent()
        self.mission = 'Complete the tasks'

    def step(self, action):
        obs, reward, done, info = super().step(action)

        reward = self.step_count
        # fwd_pos = self.front_pos
        # fwd_cell = self.grid.get(*fwd_pos)
        # obj_type = "" if fwd_cell is None else fwd_cell.type

        # print("obj in front of agent: {}, action: {}, position: {}".format(obj_type, self.actions(action), self.agent_pos))
        return obs, reward, done, info


class OneKeyRoom3x3(ObjRoom):
    def __init__(self):
        super(OneKeyRoom3x3, self).__init__(size=5, types=['key'], num_keys=1, num_balls=1)


class OneKeyRoom2x2(ObjRoom):
    def __init__(self):
        super(OneKeyRoom2x2, self).__init__(size=4, types=['key'], num_keys=1, num_balls=1)


class EmptyRoom5x5(EmptyEnv):
    def __init__(self):
        super().__init__(size=5, agent_start_pos=(1,1))

    def step(self, action):
        # todo adjust this with DFA acceptance
        obs, reward, done, info = super().step(action)
        print("position: {}, done: {}".format(self.agent_pos, done))
        if done:
            output_reward = np.array([reward, 0.0])
        else:
            output_reward = np.array([reward, 1.0])
        return obs, output_reward, done, info


class FaultyObjRoom(MiniGridEnv):
    def __init__(
            self,
            size=4,
            num_keys=1,
            num_balls=1,
            drop_prob=0.01,
            move_prob=0.05,
            door_pos=None,
            lava=None,
            types=None,
            goal=False,
            seed=543  # including a seed means that we can generate the environment randomly,
                      # and it will be the same for each agent
    ):
        """
        :param size: The size of the grid
        :param num_keys: The number of keys to place on the map
        :param num_balls: The number of balls to place on the map
        :param drop_prob: The probability of dropping an object while carrying
        :param move_prob: The probability of malfunctioning while moving
        :param door_pos: The position of a door, default is None
        :param lava: The number of lava squares to place on a map
        :param types: The types of objects that a DFA can use
        :param goal(bool): Whether or not a goal state is on the map
        :param seed: The random seed used to generate an environment
        """
        random.seed(seed)
        self.seed(seed)
        self.p1 = drop_prob
        self.place_goal = goal
        self.p2 = move_prob
        self.num_objs = num_keys + num_balls
        self.num_keys = num_keys
        self.num_balls = num_balls
        self.num_lava_squares = lava
        self.types = ['ball', 'key'] if types is None else types
        if lava and 'lava' not in self.types:
            self.types + ['lava']
        self.door_pos = door_pos
        super().__init__(
            grid_size=size,
            max_steps=2000,
            see_through_walls=True
        )

    def _reward(self):
        return 10.0

    def _gen_grid(self, height, width):
        self.grid = Grid(height, width)
        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # Place a goal square in the bottom-right corner
        if self.place_goal:
            self.put_obj(Goal(), width - 2, height - 2)
        if self.door_pos is not None:
            self.grid.set(*self.door_pos, Door('red'))

        # types = self.types

        objs = []

        # For each object to be generated,
        for lava in range(self.num_lava_squares):
            obj = Lava()
            self.place_obj(obj)
            objs.append(obj)

        for key in range(self.num_keys):
            obj = Key('red')
            self.place_obj(obj)
            objs.append(obj)

        for ball in range(self.num_balls):
            obj = Ball('blue')
            self.place_obj(obj)
            objs.append(obj)

        # Randomize the player start position and orientation
        self.place_agent()
        self.mission = 'Complete the tasks'

    @property
    def left_vec(self):
        """
        Get the vector pointing to the left of the agent
        """
        dx, dy = self.dir_vec
        return np.array((dy, dx))

    @property
    def random_pos(self):
        # sample a uniform random number
        r = random.random()
        if r < self.p2:
            # agent can move to a position that rotates left or right
            right = self.right_vec
            left = self.left_vec
            choice = [left, right]
            return self.agent_pos + choice[random.randint(0, 1)]
        else:
            return self.front_pos

    def step(self, action):
        self.step_count += 1

        reward = 1.0
        done = False

        # Get the position in front of the agent, this includes some noise
        # if the agent fails the random noise check then the fwd position
        # will be either a rotation left or right.
        fwd_pos = self.random_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # todo introduce noise,
        # [ ] - movement noise
        # [X] - carrying noise

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            # if fwd_cell != None and fwd_cell.type == 'goal':
            #    done = True
            #    reward = self._reward()
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)

        # Drop an object, if carrying with probability p1

        if action != self.actions.drop:
            # sample a uniform random number, if the random number is higher than p1 drop the item
            r = random.random()
            if r < self.p1:
                # If there is nothing in the fwd cell and the agent is carrying then drop the object
                if not fwd_cell and self.carrying:
                    self.grid.set(*fwd_pos, self.carrying)
                    self.carrying.cur_pos = fwd_pos
                    self.carrying = None
                elif fwd_cell is not None and self.carrying:
                    # find an empty cell
                    for dir in range(0, 4):
                        new_pos = self.agent_pos + DIR_TO_VEC[dir]
                        # check to the if the new position is empty
                        cell = self.grid.get(*new_pos)
                        if not cell:
                            # drop the object here
                            self.grid.set(*cell, self.carrying)
                            self.carrying.cur_pos = new_pos
                            self.carrying = None

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}


class Room5x5RandomOneKey(FaultyObjRoom):
    def __init__(self):
        super(Room5x5RandomOneKey, self).__init__(
            size=7, num_keys=1, num_balls=0, drop_prob=0.01, move_prob=0.05, lava=3, goal=True
        )




