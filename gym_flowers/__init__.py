import itertools
from gym.envs.registration import register
import numpy as np


register(id='Reacher{}-v0'.format(2),
         entry_point='gym_flowers.envs.mujoco:ReacherEnv2',
         max_episode_steps=50,
         reward_threshold=1.0
         )

modules = ['0','1','2','02','01','012']
random_objects = [True, False]
for mod in modules:
    for ro in random_objects:
        if ro:
            suffix=mod+'ro'
        else:
            suffix=mod
        kwargs = dict(modules=[int(s) for s in mod],
                      random_objects=ro)
        register(
            id='ModularArm{}-v0'.format(suffix),
            entry_point='gym_flowers.envs:ModularArmV0',
            max_episode_steps=50,
            kwargs=kwargs,
            reward_threshold=1.0,
        )


random_objects = [True, False]
for ro in random_objects:
    if ro:
        suffix='ro'
    else:
        suffix=''
    kwargs = dict(random_objects=ro)
    register(
        id='ArmStickBall{}-v0'.format(suffix),
        entry_point='gym_flowers.envs:ArmStickBallV0',
        max_episode_steps=50,
        kwargs=kwargs,
        reward_threshold=1.0,
    )

arm_lengths = [[0.5, 0.5], [0.5, 0.3, 0.2], [0.4, 0.3, 0.2, 0.1], [0.35, 0.25, 0.2, 0.1, 0.1],  [0.3, 0.2, 0.2, 0.1, 0.1, 0.05], [0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]]
for i in range(2,7):
    suffix=str(i)
    register(
        id='ArmBall{}joints-v0'.format(suffix),
        entry_point='gym_flowers.envs:ArmBall',
        kwargs=dict(arm_lengths=np.array(arm_lengths[i-2])),
        max_episode_steps=50,
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

reward_types = ['sparse', 'dense']
obs_types = ['xyz', 'RGB', 'Vae', 'Betavae']
params_iterator = list(itertools.product(reward_types, obs_types))

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
