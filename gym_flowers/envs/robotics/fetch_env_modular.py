import numpy as np
import os
from gym.envs.robotics import rotations, utils
from gym_flowers.envs.robotics import robot_env_modular


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class ModularFetchEnv(robot_env_modular.ModularRobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type, tasks
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type

        # define the different tasks
        # task 0: Hand position (3D)
        # task 1: Cube0 position (2D)
        # task 2: Cube1 position (2D)
        # task 3: Cube0 position (3D)
        # task 4: Cube1 position (3D)
        # task 5: Stack one Cube over other in given position (2D)
        self.tasks = tasks
        self.n_tasks = len(self.tasks)
        all_tasks_id = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [3, 4, 5], [6, 7, 8], [3, 4, 5, 6, 7, 8]]
        ind = 0
        self.tasks_id = []
        for i in range(0,self.n_tasks):
            self.tasks_id.append(list(range(ind, ind + len(all_tasks_id[i]))))
            ind += len(all_tasks_id[i])
        self.tasks_obs_id = [all_tasks_id[i] for i in self.tasks]
        self.flat = False
        self.dim_g = sum([len(self.tasks_id[i]) for i in range(self.n_tasks)])
        self.goal = np.zeros([self.dim_g])
        self.mask = np.zeros([self.n_tasks])

        self.info = dict(is_success=0)


        if model_path.startswith('/'):
            model_path = model_path
        else:
            model_path = os.path.join(os.path.dirname(__file__), 'assets', model_path)
        super(ModularFetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info=None):

        if goal.ndim == 1:
            # find current task
            ind = np.argwhere(goal!=0).squeeze().tolist()
            good_task = []
            for i_t in range(self.n_tasks):
                if ind == self.tasks_id[i_t]:
                    good_task.append(i_t)
            assert len(good_task) == 1

            task = good_task[0]

            if task in [0, 1, 2, 3, 4]:
                # Compute distance between goal and the achieved goal.
                d = goal_distance(achieved_goal[self.tasks_id[task]], goal[self.tasks_id[task]])
                if self.reward_type == 'sparse':
                    return -(d > self.distance_threshold).astype(np.float32)
                else:
                    return -d
            elif task == 5:
                d0 = goal_distance(achieved_goal[self.tasks_id[task][:2]], goal[self.tasks_id[task][:2]])
                d1 = goal_distance(achieved_goal[ self.tasks_id[task][3:5]], goal[self.tasks_id[task][:2]])
                dh = np.abs(achieved_goal[self.tasks_id[task][2]] - achieved_goal[self.tasks_id[task][5]])
                if self.reward_type == 'sparse':
                    return - ((d0 > self.distance_threshold).astype(np.float32) or (d1 > self.distance_threshold).astype(np.float32) or (dh < 0.04).astype(np.float32) or (dh >
                                                                                                                                                                            0.055).astype(np.float32))
                else:
                    return - 1
        else:
            r = np.zeros([goal.shape[0]])
            for i_g in range(goal.shape[0]):
                # find current task
                ind = np.argwhere(goal[i_g] != 0).squeeze().tolist()
                good_task = []
                for i_t in range(self.n_tasks):
                    if ind == self.tasks_id[i_t]:
                        good_task.append(i_t)
                assert len(good_task) == 1


                task = good_task[0]
                if task in [0, 1, 2, 3, 4]:
                    # Compute distance between goal and the achieved goal.
                    d = goal_distance(achieved_goal[i_g, self.tasks_id[task]], goal[i_g,self.tasks_id[task]])
                    if self.reward_type == 'sparse':
                        r[i_g] =  -(d > self.distance_threshold).astype(np.float32)
                    else:
                        r[i_g] = -d
                elif task == 5:
                    d0 = goal_distance(achieved_goal[i_g,self.tasks_id[task][:2]], goal[i_g,self.tasks_id[task][:2]])
                    d1 = goal_distance(achieved_goal[i_g,self.tasks_id[task][3:5]], goal[i_g,self.tasks_id[task][:2]])
                    dh = np.abs(achieved_goal[i_g, self.tasks_id[task][2]] - achieved_goal[i_g, self.tasks_id[task][5]])
                    if self.reward_type == 'sparse':
                        r[i_g] = - ((d0 > self.distance_threshold).astype(np.float32) or (d1 > self.distance_threshold).astype(np.float32) or (dh < 0.045).astype(np.float32) or
                                    (dh >0.051).astype(np.float32))
                    else:
                        r[i_g] = - 1

            return r.reshape([r.size, 1])


    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def reset_task_goal(self, goal, task=None):
        self._set_task(task)
        self._set_goal(goal)
        obs = self._get_obs()
        return obs

    def _set_task(self, t):
        if not self.flat:
            self.task = t

    def set_flat_env(self):
        self.flat = True

    def _set_goal(self, goal):

        desired_goal = np.zeros([self.dim_g])
        if self.task==0: # 3D coordinates for the hand
            tmp_goal = self.initial_gripper_xpos[:3] + goal * 0.15
            desired_goal[self.tasks_id[self.task]] = tmp_goal.copy()
            self.goal_to_render = tmp_goal.copy()

        elif self.task in [1, 2]: # 3D coordinates for the object in 2D plane
            tmp_goal = self.initial_gripper_xpos[:3] + goal * self.target_range + self.target_offset
            tmp_goal[2] = self.height_offset
            desired_goal[self.tasks_id[self.task]] = tmp_goal.copy()
            self.goal_to_render = tmp_goal.copy()

        elif self.task == 5: # 3D coordinates for the two objects
            tmp_goal = self.initial_gripper_xpos[:3] + goal[:3] * self.target_range + self.target_offset
            tmp_goal[2] = self.height_offset
            desired_goal[self.tasks_id[self.task][0:3]] = tmp_goal.copy()
            desired_goal[self.tasks_id[self.task][3:6]] = tmp_goal.copy()
            self.goal_to_render = tmp_goal.copy()

        elif self.task in [3, 4]: # 3D coordinates for the object
            tmp_goal = self.initial_gripper_xpos[:3] + goal * self.target_range + self.target_offset
            tmp_goal[2] = self.height_offset + (goal[2] + 1) * 0.45 / 2 # mapping in -1,1 to 0,0.45 #self.np_random.uniform(0, 0.45)
            desired_goal[self.tasks_id[self.task]] = tmp_goal.copy()
            self.goal_to_render = tmp_goal.copy()

        self.goal = desired_goal
        self.mask = np.zeros([self.n_tasks])
        self.mask[self.task] = 1

    def _compute_achieved_goal(self, obs):

        achieved_goal = np.zeros([self.dim_g])
        for i_t in range(self.n_tasks):
            achieved_goal[self.tasks_id[i_t]] = obs[self.tasks_obs_id[i_t]]
        return achieved_goal

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            #object 0
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
        else:
            object1_pos = object0_rot = object0_velp = object0_velr = object0_rel_pos = np.zeros(0)
            object1_pos = object1_rot = object1_velp = object1_velr = object1_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        obs = np.concatenate([
            grip_pos, object0_pos.ravel(), object1_pos.ravel(), object0_rel_pos.ravel(), object1_rel_pos.ravel(), gripper_state, object0_rot.ravel(),
            object1_rot.ravel(), object0_velp.ravel(), object1_velp.ravel(), object0_velr.ravel(), object1_velr.ravel(), grip_velp, gripper_vel,
        ])

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = self._compute_achieved_goal(obs)

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
            'mask': self.mask
        }

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
        self.sim.model.site_pos[site_id] = self.goal_to_render - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object0_xpos = self.initial_gripper_xpos[:2]
            object1_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object0_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object0_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            # set second object's position
            while np.linalg.norm(object1_xpos - self.initial_gripper_xpos[:2]) < 0.1 or np.linalg.norm(object1_xpos - object0_xpos) < 0.1:
                object1_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            # object1_xpos = object0_xpos.copy()
            object0_qpos = self.sim.data.get_joint_qpos('object0:joint')
            object1_qpos = self.sim.data.get_joint_qpos('object1:joint')
            assert object0_qpos.shape == (7,)
            assert object1_qpos.shape == (7,)
            object0_qpos[:2] = object0_xpos
            object1_qpos[:2] = object1_xpos
            # object0_qpos[2] = 1
            self.sim.data.set_joint_qpos('object0:joint', object0_qpos)
            self.sim.data.set_joint_qpos('object1:joint', object1_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        return float(not self.compute_reward(achieved_goal, desired_goal, {}))

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
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    @property
    def nb_tasks(self):
        return self.n_tasks