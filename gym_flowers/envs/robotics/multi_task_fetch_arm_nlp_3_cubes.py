import numpy as np
import os
from gym.envs.robotics import rotations, utils
from gym_flowers.envs.robotics import simple_multi_task_robot_env
import numpy.linalg as la


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class MultiTaskFetchArmNLP(simple_multi_task_robot_env.SimpleMultiTaskRobotEnv):
    """Superclass for all Fetch environments.
    Note that the addition of more than 3 cubes in the mujoco simulation involve some weird behaviors of the simulation.
    Because of this, we do not add more than 3 cubes in the simulation, but simulate the 2 extra distractor cubes ourselves.
    This does not influence the complexity of the task.
    """

    def __init__(self, model_path, n_substeps, gripper_extra_height, obj_range, initial_qpos):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            obj_range (float): range of a uniform distribution for sampling initial object positions
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
        """
        self.gripper_extra_height = gripper_extra_height
        self.obj_range = obj_range
        self.n_timesteps = 50

        if model_path.startswith('/'):
            model_path = model_path
        else:
            model_path = os.path.join(os.path.dirname(__file__), 'assets', model_path)
        super(MultiTaskFetchArmNLP, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)


    # RobotEnv methods
    # ----------------------------

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)


    def _get_obs(self):

        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        # object 0
        object0_pos = self.sim.data.get_site_xpos('object0')
        # rotations
        object0_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
        # velocities
        object0_velp = self.sim.data.get_site_xvelp('object0') * dt
        object0_velr = self.sim.data.get_site_xvelr('object0') * dt
        # gripper state
        object0_rel_pos = object0_pos - grip_pos
        object0_velp -= grip_velp

        # object 1
        object1_pos = self.sim.data.get_site_xpos('object1')
        # rotations
        object1_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object1'))
        # velocities
        object1_velp = self.sim.data.get_site_xvelp('object1') * dt
        object1_velr = self.sim.data.get_site_xvelr('object1') * dt
        # gripper state
        object1_rel_pos = object1_pos - grip_pos
        object1_velp -= grip_velp

        # object 2
        object2_pos = self.sim.data.get_site_xpos('object2')
        # rotations
        object2_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object2'))
        # velocities
        object2_velp = self.sim.data.get_site_xvelp('object2') * dt
        object2_velr = self.sim.data.get_site_xvelr('object2') * dt
        # gripper state
        object2_rel_pos = object2_pos - grip_pos
        object2_velp -= grip_velp

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric
        obs = np.concatenate([grip_pos,
                              object0_pos.ravel(), object1_pos.ravel(), object2_pos.ravel(),
                              object0_rel_pos.ravel(), object1_rel_pos.ravel(), object2_rel_pos.ravel(),
                              object0_rot.ravel(), object1_rot.ravel(), object2_rot.ravel(),
                              object0_velp.ravel(), object1_velp.ravel(), object2_velp.ravel(),
                              object0_velr.ravel(), object1_velr.ravel(), object2_velr.ravel(),
                              grip_velp, gripper_vel, gripper_state])

        if self.first_obs is None:
            self.first_obs = obs.copy()

        return np.concatenate([obs.copy(), obs.copy() - self.first_obs], axis=0)

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = np.zeros([3]) - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        object0_xpos = self.initial_gripper_xpos[:2]
        object1_xpos = self.initial_gripper_xpos[:2]
        object2_xpos = self.initial_gripper_xpos[:2]

        while np.linalg.norm(object0_xpos - self.initial_gripper_xpos[:2]) < 0.1:
            object0_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)

        while np.linalg.norm(object1_xpos - self.initial_gripper_xpos[:2]) < 0.1 or np.linalg.norm(object1_xpos - object0_xpos) < 0.1:
            object1_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)

        while np.linalg.norm(object2_xpos - self.initial_gripper_xpos[:2]) < 0.1 or np.linalg.norm(object2_xpos - object0_xpos) < 0.1 or \
            np.linalg.norm(object1_xpos - object2_xpos) < 0.1:
            object2_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)

        object0_qpos = self.sim.data.get_joint_qpos('object0:joint')
        object1_qpos = self.sim.data.get_joint_qpos('object1:joint')
        object2_qpos = self.sim.data.get_joint_qpos('object2:joint')

        assert object0_qpos.shape == (7,)
        assert object1_qpos.shape == (7,)
        assert object2_qpos.shape == (7,)
        object0_qpos[:2] = object0_xpos
        object1_qpos[:2] = object1_xpos
        object2_qpos[:2] = object2_xpos
        object0_qpos[-3:] = 0
        object1_qpos[-3:] = 0
        object2_qpos[-3:] = 0
        object0_qpos[2] = self.height_offset
        object1_qpos[2] = self.height_offset
        object2_qpos[2] = self.height_offset

        self.sim.data.set_joint_qpos('object0:joint', object0_qpos)
        self.sim.data.set_joint_qpos('object1:joint', object1_qpos)
        self.sim.data.set_joint_qpos('object2:joint', object2_qpos)

        self.sim.forward()
        return True


    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        self.height_offset = 0.42599082 #height of table
