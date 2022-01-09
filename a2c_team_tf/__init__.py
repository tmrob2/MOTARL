from gym.envs.registration import register

register(
    id='Empty-multi-v0',
    entry_point='a2c_team_tf.envs.minigrid_empty_mult:EmptyMultiTask'
)

register(
    id='Empty-multi-4x4-v0',
    entry_point='a2c_team_tf.envs.minigrid_empty_mult:EmptyMultiEnv4x4'
)

register(
    id='Empty-multi-5x5-v0',
    entry_point='a2c_team_tf.envs.minigrid_empty_mult:EmptyMultiEnv5x5'
)

register(
    id='Mult-obj-5x5-v0',
    entry_point='a2c_team_tf.envs.experimental.minigrid_fetch_mult:MultObjNoGoal5x5'
)

register(
    id='Mult-obj-4x4-v0',
    entry_point='a2c_team_tf.envs.experimental.minigrid_fetch_mult:MultObjNoGoal4x4'
)

register(
    id='Team-obj-5x5-v0',
    entry_point='a2c_team_tf.envs.team_grid_mult:TestEnv'
)

register(
    id='Team-obj-4x4-door-v0',
    entry_point='a2c_team_tf.envs.team_grid_mult:TestEnv2'
)

register(
    id='CartPole-default-v0',
    entry_point='a2c_team_tf.envs.cartpole_ma:CartPoleDefault'
)

register(
    id='CartPole-heavy-long-v0',
    entry_point='a2c_team_tf.envs.cartpole_ma:CartPoleHeavyLong'
)

register(
    id='DualDoors-v0',
    entry_point='a2c_team_tf.envs.team_grid_mult:DualDoors'
)