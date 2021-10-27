from gym_minigrid.envs import RoomGrid, EmptyEnv
from gym_minigrid.minigrid import *
from gym.envs.registration import register
from gym_minigrid.wrappers import *


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


class PuzzleRoom(RoomGrid):
    """Find the goal"""
    def __init__(self, seed=None):
        room_size = 6
        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=8 * room_size ** 2,
            seed=seed
        )

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Add a box to the room on the right
        obj, _ = self.add_object(1, 0, kind='box')
        # Make sure that the rooms are directly connected by a locked door
        door, _ = self.add_door(0, 0, 0, locked=True)
        # Add a key to unlock the door
        self.add_object(0, 0, 'key', door.color)

        self.place_agent(0, 0)
        self.obj = obj

    def step(self, action):
        # todo adjust this with DFA acceptance
        obs, reward, done, info = super().step(action)
        print("position: {}".format(self.agent_pos))
        return obs, reward, done, info




