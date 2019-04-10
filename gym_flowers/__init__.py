import itertools
from gym.envs.registration import register
import numpy as np



register(
    id='MultiTaskFetchArm4-v3',
    entry_point='gym_flowers.envs.robotics:MultiTaskFetchArmV3',
    kwargs=dict(tasks=[0,1,2,3], n_distractors=0),
    max_episode_steps=50,
)


for target_range in [1, 2, 3, 5, 10]:
    for n_tasks in range(1, 12):
        suffix = str(n_tasks)
        kwargs = {'tasks': range(n_tasks),
                  'target_range': target_range}
        if target_range == 1:
            id = 'MultiTaskFetchArm{}-v5'.format(suffix)
        else:
            id = 'MultiTaskFetchArm{}_{}-v5'.format(suffix, str(target_range))
        register(
            id=id,
            entry_point='gym_flowers.envs.robotics:MultiTaskFetchArmV5',
            kwargs=kwargs,
            max_episode_steps=50,
        )

register(
    id='MultiTaskFetchArm-v6',
    entry_point='gym_flowers.envs.robotics:MultiTaskFetchArmV6',
    max_episode_steps=50,
)



