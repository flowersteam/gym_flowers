from gym import utils
from gym_flowers.envs.robotics import multi_task_fetch_arm_v3, multi_task_fetch_arm_v5

class MultiTaskFetchArmV3(multi_task_fetch_arm_v3.MultiTaskFetchArmV3, utils.EzPickle):
    def __init__(self, reward_type='sparse', tasks=[0,1,2,3,4,5,6], n_distractors=1):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        for i in range(2):
            initial_qpos['object'+str(i)+':joint'] = [1.25, 0.53, 0.45, 1., 0., 0., 0.]
        multi_task_fetch_arm_v3.MultiTaskFetchArmV3.__init__(
            self, 'fetch/multi_task_fetch_arm.xml', has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, tasks=tasks)
        utils.EzPickle.__init__(self)



class MultiTaskFetchArmV5(multi_task_fetch_arm_v5.MultiTaskFetchArmV5, utils.EzPickle):
    def __init__(self, reward_type='sparse', tasks=[0,1,2,3,4,5,6,7], target_range=1):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        for i in range(2):
            initial_qpos['object'+str(i)+':joint'] = [1.25, 0.53, 0.45, 1., 0., 0., 0.]
        multi_task_fetch_arm_v5.MultiTaskFetchArmV5.__init__(
            self, 'fetch/multi_task_fetch_arm.xml', has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15*target_range, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, tasks=tasks)
        utils.EzPickle.__init__(self)

