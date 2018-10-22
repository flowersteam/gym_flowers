import itertools
from gym.envs.registration import register
import numpy as np



for n_dist in range(5):
    n_tasks = 4 + n_dist
    suffix = str(n_tasks)
    tasks = list(range(n_tasks))
    kwargs = {
        'tasks': tasks,
        'n_distractors': n_dist
    }
    register(
        id='MultiTaskFetchArm{}-v0'.format(suffix),
        entry_point='gym_flowers.envs.robotics:MultiTaskFetchArmV0',
        kwargs=kwargs,
        max_episode_steps=50,
    )

for n_dist in range(5):
    n_tasks = 4 + n_dist
    suffix = str(n_tasks)
    tasks = list(range(n_tasks))
    kwargs = {
        'tasks': tasks,
        'n_distractors': n_dist
    }
    register(
        id='MultiTaskFetchArm{}-v1'.format(suffix),
        entry_point='gym_flowers.envs.robotics:MultiTaskFetchArmV1',
        kwargs=kwargs,
        max_episode_steps=50,
    )


tasks = ['0','1','2','02','01','012']
random_objects = [True, False]
dist = [True, False]
for task in tasks:
    for ro in random_objects:
        for d in dist:
            suffix = task
            if ro:
                suffix += 'ro'
            if d:
                suffix += 'dist'
            kwargs = dict(tasks=[int(s) for s in task],
                          random_objects=ro,
                          distractor=d)
            register(
                id='ModularArm{}-v0'.format(suffix),
                entry_point='gym_flowers.envs.armball:ModularArmV0',
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
        entry_point='gym_flowers.envs.armball:ArmStickBallV0',
        max_episode_steps=50,
        kwargs=kwargs,
        reward_threshold=1.0,
    )



arm_lengths = [[0.5, 0.5], [0.5, 0.3, 0.2], [0.4, 0.3, 0.2, 0.1], [0.35, 0.25, 0.2, 0.1, 0.1],  [0.3, 0.2, 0.2, 0.1, 0.1, 0.05], [0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]]
for i in range(2,7):
    suffix=str(i)
    register(
        id='ArmBall{}joints-v0'.format(suffix),
        entry_point='gym_flowers.envs.armball:ArmBall',
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
        entry_point='gym_flowers.envs.armball:ArmBall',
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
            entry_point='gym_flowers.envs.armball:ArmBalls',
            kwargs=kwargs,
            max_episode_steps=50,
    )

grid_sizes = [10, 20, 40, 80, 160]
randomness = [False, True]
params_iterator = list(itertools.product(grid_sizes, randomness))

for (grid_size, stochastic) in params_iterator:
    suffix = 'v1' if stochastic else 'v0'
    kwargs = {
        'grid_size': grid_size,
        'stochastic': stochastic,
    }

    # SquareDistractor

    register(id='SquareDistractor'+str(grid_size)+'-'+suffix,
             entry_point='gym_flowers.envs.flokoban:SquareDistractor',
             max_episode_steps=50,
             reward_threshold=1.0,
             kwargs=kwargs,
             )
