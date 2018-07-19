from gym.envs.registration import register

register(
    id='ArmBall-v0',
    entry_point='gym_flowers.envs:ArmBall',
    max_episode_steps=50,
    reward_threshold=1.0,
)

register(
    id='ArmBall-v1',
    entry_point='gym_flowers.envs:ArmBallDense',
    max_episode_steps=50,
)