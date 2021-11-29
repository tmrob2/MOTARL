import teamgrid as tg
from teamgrid.minigrid import MiniGridEnv, Grid

class TestEnv(MiniGridEnv):
    def __init__(
            self,
            num_agents=2
    ):
        super(TestEnv, self).__init__(
            width=5,
            height=5,
            max_steps=100
        )
        self.num_agents = num_agents

    def _gen_grid(self, width, height):
        # instantiate the grid
        self.grid = Grid(width, height)
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        # randomise the player start positions
        for i in range(self.num_agents):
            self.place_agent()


    def step(self, actions):
        # I guess actions are a vector now

        rewards = [0] * len(self.agents)



