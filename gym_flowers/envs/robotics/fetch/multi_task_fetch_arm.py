from gym import utils
from gym_flowers.envs.robotics import multi_task_fetch_arm_v0, multi_task_fetch_arm_v1, multi_task_fetch_arm_v2, \
    multi_task_fetch_arm_v3, multi_task_fetch_arm_v4, multi_task_fetch_arm_v5, multi_task_fetch_arm_nlp, multi_task_fetch_arm_v6, \
multi_task_fetch_arm_nlp_1_cube


class MultiTaskFetchArmV0(multi_task_fetch_arm_v0.MultiTaskFetchArmV0, utils.EzPickle):
    def __init__(self, reward_type='sparse', tasks=[0,1,2,3,4,5,6], n_distractors=1):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        for i in range(2):
            initial_qpos['object'+str(i)+':joint'] = [1.25, 0.53, 0.45, 1., 0., 0., 0.]
        multi_task_fetch_arm_v0.MultiTaskFetchArmV0.__init__(
            self, 'fetch/multi_task_fetch_arm.xml', has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, tasks=tasks)
        utils.EzPickle.__init__(self)


class MultiTaskFetchArmV1(multi_task_fetch_arm_v1.MultiTaskFetchArmV1, utils.EzPickle):
    def __init__(self, reward_type='sparse', tasks=[0,1,2,3,4,5,6], n_distractors=1):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        for i in range(2):
            initial_qpos['object'+str(i)+':joint'] = [1.25, 0.53, 0.45, 1., 0., 0., 0.]
        multi_task_fetch_arm_v1.MultiTaskFetchArmV1.__init__(
            self, 'fetch/multi_task_fetch_arm.xml', has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, tasks=tasks)
        utils.EzPickle.__init__(self)

class MultiTaskFetchArmV2(multi_task_fetch_arm_v2.MultiTaskFetchArmV2, utils.EzPickle):
    def __init__(self, reward_type='sparse', tasks=[0,1,2,3,4,5,6], n_distractors=1):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        for i in range(2):
            initial_qpos['object'+str(i)+':joint'] = [1.25, 0.53, 0.45, 1., 0., 0., 0.]
        multi_task_fetch_arm_v2.MultiTaskFetchArmV2.__init__(
            self, 'fetch/multi_task_fetch_arm.xml', has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, tasks=tasks)
        utils.EzPickle.__init__(self)


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


class MultiTaskFetchArmV4(multi_task_fetch_arm_v4.MultiTaskFetchArmV4, utils.EzPickle):
    def __init__(self, reward_type='sparse', tasks=[0,1,2,3,4,5,6], target_range=1):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        for i in range(2):
            initial_qpos['object'+str(i)+':joint'] = [1.25, 0.53, 0.45, 1., 0., 0., 0.]
        multi_task_fetch_arm_v4.MultiTaskFetchArmV4.__init__(
            self, 'fetch/multi_task_fetch_arm.xml', has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15 * target_range, distance_threshold=0.05,
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

class MultiTaskFetchArmV6(multi_task_fetch_arm_v6.MultiTaskFetchArmV6, utils.EzPickle):
    def __init__(self, reward_type='sparse', tasks=[0,1,2,3,4,5,6,7,8,9], target_range=1):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        for i in range(2):
            initial_qpos['object'+str(i)+':joint'] = [1.25, 0.53, 0.45, 1., 0., 0., 0.]
        multi_task_fetch_arm_v6.MultiTaskFetchArmV6.__init__(
            self, 'fetch/multi_task_fetch_arm.xml', has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15*target_range, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, tasks=tasks)
        utils.EzPickle.__init__(self)

class MultiTaskFetchArmNLP(multi_task_fetch_arm_nlp.MultiTaskFetchArmNLP, utils.EzPickle):
    def __init__(self):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        for i in range(2):
            initial_qpos['object'+str(i)+':joint'] = [1.25, 0.53, 0.45, 1., 0., 0., 0.]
        multi_task_fetch_arm_nlp.MultiTaskFetchArmNLP.__init__(
            self, 'fetch/multi_task_fetch_arm.xml', n_substeps=20, gripper_extra_height=0.15,
            obj_range=0.15, initial_qpos=initial_qpos)
        utils.EzPickle.__init__(self)

class MultiTaskFetchArmNLP1(multi_task_fetch_arm_nlp_1_cube.MultiTaskFetchArmNLP1, utils.EzPickle):
    def __init__(self):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        for i in range(1):
            initial_qpos['object'+str(i)+':joint'] = [1.25, 0.53, 0.45, 1., 0., 0., 0.]
        multi_task_fetch_arm_nlp_1_cube.MultiTaskFetchArmNLP1.__init__(
            self, 'fetch/multi_task_fetch_arm.xml', n_substeps=20, gripper_extra_height=0.15,
            obj_range=0.15, initial_qpos=initial_qpos)
        utils.EzPickle.__init__(self)


