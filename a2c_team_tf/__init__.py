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
    entry_point='a2c_team_tf.envs.minigrid_fetch_mult:MultObjNoGoal5x5'
)

register(
    id='Mult-obj-4x4-v0',
    entry_point='a2c_team_tf.envs.minigrid_fetch_mult:MultObjNoGoal4x4'
)

register(
    id='Mult-obj-4x4-bonus-v0',
    entry_point='a2c_team_tf.envs.minigrid_fetch_mult:MultObj4x4ActBonus'
)

register(
    id='Team-obj-5x5-v0',
    entry_point='a2c_team_tf.envs.team_grid_mult:TestEnv'
)