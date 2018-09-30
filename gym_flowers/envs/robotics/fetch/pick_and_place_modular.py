from gym import utils
from gym_flowers.envs.robotics import fetch_env_modular


class ModularFetchPickAndPlaceEnv(fetch_env_modular.ModularFetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', tasks=[0,1,2,3,4,5,6], n_distractors=1):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        for i in range(n_distractors + 2):
            initial_qpos['object'+str(i)+':joint'] = [1.25, 0.53, 0.45, 1., 0., 0., 0.]
        fetch_env_modular.ModularFetchEnv.__init__(
            self, 'fetch/pick_and_place_modular.xml', has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, tasks=tasks, n_distractors=n_distractors)
        utils.EzPickle.__init__(self)




