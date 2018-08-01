from gym.envs.registration import register

# register(
#     id='ArmBall-v0',
#     entry_point='gym_flowers.envs:ArmBall',
#     max_episode_steps=50,
#     reward_threshold=1.0,
# )
#
# register(
#     id='ArmBall-v1',
#     entry_point='gym_flowers.envs:ArmBallDense',
#     max_episode_steps=50,
# )

register(
    id='ModularArm-v0',
    entry_point='gym_flowers.envs:ModularArmV0',
    max_episode_steps=75,
    reward_threshold=1.0,
)


for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
    }

    # ArmBall

    register(
        id='ArmBall{}-v0'.format(suffix),
        entry_point='gym_flowers.envs:ArmBall',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
            id='ArmBalls{}-v0'.format(suffix),
            entry_point='gym_flowers.envs:ArmBalls',
            kwargs=kwargs,
            max_episode_steps=50,
    )
