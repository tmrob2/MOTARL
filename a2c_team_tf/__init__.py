from gym.envs.registration import register

register(
    id='Empty-5x5-v0',
    entry_point='a2c_team_tf.envs.minigrid_wrapper:EmptyEnv5x5'
)

register(
    id='Empty-S-Bonus-5x5-v0',
    entry_point='a2c_team_tf.envs.minigrid_wrapper:EmptyEnv5x5StateBonus'
)

register(
    id='Empty-multi-v0',
    entry_point='a2c_team_tf.envs.minigrid_multitask_wrapper:EmptyMultiTask'
)

register(
    id='Empty-multi-4x4-v0',
    entry_point='a2c_team_tf.envs.minigrid_multitask_wrapper:EmptyMultiEnv4x4'
)

register(
    id='Empty-multi-5x5-v0',
    entry_point='a2c_team_tf.envs.minigrid_multitask_wrapper:EmptyMultiEnv5x5'
)