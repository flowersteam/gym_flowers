import itertools
from gym.envs.registration import register

register(
    id='ModularArm-v0',
    entry_point='gym_flowers.envs:ModularArmV0',
    max_episode_steps=75,
    reward_threshold=1.0,
)

reward_types = ['sparse', 'dense']
obs_types = ['xyz', 'rgb', 'vae', 'betavae']
params_iterator = list(itertools.product(reward_types, obs_types))

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

for (reward_type, obs_type) in params_iterator:
    suffix = obs_type if obs_type is not 'xyz' else ''
    suffix += 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'obs_type': obs_type,
        'reward_type': reward_type,
    }

    # ArmBalls

    register(
            id='ArmBalls{}-v0'.format(suffix),
            entry_point='gym_flowers.envs:ArmBalls',
            kwargs=kwargs,
            max_episode_steps=50,
    )
