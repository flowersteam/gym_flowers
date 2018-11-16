import numpy as np
import os
from gym.envs.robotics import rotations, utils
from gym_flowers.envs.robotics import multi_task_robot_env
import numpy.linalg as la


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class MultiTaskFetchArmV4(multi_task_robot_env.MultiTaskRobotEnv):
    """Superclass for all Fetch environments.
    Note that the addition of more than 3 cubes in the mujoco simulation involve some weird behaviors of the simulation.
    Because of this, we do not add more than 3 cubes in the simulation, but simulate the 2 extra distractor cubes ourselves.
    This does not influence the complexity of the task.
    """

    def __init__(self, model_path, n_substeps, gripper_extra_height, block_gripper, has_object, target_in_the_air, target_offset,
                 obj_range, target_range, distance_threshold, initial_qpos, reward_type, tasks):
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
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense,
            tasks (list of int): list of the tasks id
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
        self.n_timesteps = 50

        # define the different tasks
        # task 0: Hand position (3D)
        # task 1: Cube0 position (2D)
        # task 2: Cube0 position (3D)
        # task 3-4-5-6: Distractor cubes (1-2-3-4) position (2D)

        self.tasks = tasks
        self.n_tasks = len(self.tasks)
        self.n_distractors = self.n_tasks - 3

        # indices of relevant object position (achieved_goal)
        # the achieved goal for the stacking task (T3) contains the gripper coordinate, as it is necessary to compute the reward (has to be far from the goal)
        self.tasks_obs_id = [[0, 1, 2], [3, 4, 5], [3, 4, 5],  [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17]]

        dim_tasks_g = [3] * self.n_tasks
        ind_g = 0
        ind_ag = 0
        self.tasks_g_id = [] # indices of goal referring to the different tasks
        self.tasks_ag_id = [] # indices of achieved_goal referring to the different tasks
        for i in range(self.n_tasks):
            self.tasks_ag_id.append(list(range(ind_ag, ind_ag + len(self.tasks_obs_id[i]))))
            ind_ag += len(self.tasks_obs_id[i])
            self.tasks_g_id.append(list(range(ind_g, ind_g + 3)))
            ind_g += 3

        self.flat = False # flat architecture ?
        self.dim_ag = sum([len(self.tasks_ag_id[i]) for i in range(self.n_tasks)])
        self.dim_g = sum(dim_tasks_g)
        self.goal = np.zeros([self.dim_g])
        self.mask = np.zeros([self.n_tasks])
        self.task = 0

        self.info = dict(is_success=0, dense_reward=None)
        # initialization of object pos variables, position are initialized in self.reset_sim()
        self.object3_xpos = np.array([1,1])
        self.object4_xpos = np.array([1,1])


        self.bias = False

        if model_path.startswith('/'):
            model_path = model_path
        else:
            model_path = os.path.join(os.path.dirname(__file__), 'assets', model_path)
        super(MultiTaskFetchArmV4, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, task_descr, info=None, dense=False):

        if dense:
            reward_type = 'dense'
        else:
            reward_type = self.reward_type

        if goal.ndim == 1:
            goal = goal.reshape([1, goal.size])
            achieved_goal = achieved_goal.reshape([1, achieved_goal.size])
            task_descr = task_descr.reshape([1, task_descr.size])

        if self.flat:
            # When the goal is set in the union of the tasks goal spaces, the reward is the euclidean distance in that space.
            r = np.zeros([goal.shape[0]])
            ag = np.zeros([goal.shape[0], goal.shape[1]])
            for i_g in range(goal.shape[0]):
                ind = 0
                for t in range(self.n_tasks):
                    len_t = len(self.tasks_g_id[t])
                    ag[i_g, ind: ind + len_t] = achieved_goal[i_g, self.tasks_ag_id[t][:len_t]]
                    ind += len_t
                d = goal_distance(ag[i_g, :], goal[i_g, :])
                if reward_type == 'sparse':
                    r[i_g] = -(d > self.distance_threshold).astype(np.float32)
                else:
                    r[i_g] = - d
        else:
            r = np.zeros([goal.shape[0]])
            for i_g in range(goal.shape[0]):
                # find current task
                task = np.argwhere(task_descr[i_g] == 1)[0][0]

                if task in [0, 1, 2, 3, 4, 5, 6]:

                    # Compute distance between goal and the achieved goal.
                    d = goal_distance(achieved_goal[i_g, self.tasks_ag_id[task]], goal[i_g, self.tasks_g_id[task]])
                    if reward_type == 'sparse':
                        r[i_g] = -(d > self.distance_threshold).astype(np.float32)
                    else:
                        r[i_g] = -d

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

    def reset_task_goal(self, goal, task=None, directly=False, eval=False):
        self._set_task(task)
        if directly:
            self.goal_to_render = goal.copy()
            self.goal = np.zeros([self.dim_g])
            self.goal[self.tasks_g_id[task]] = goal.copy()
            self.mask = np.zeros([self.nb_tasks])
            self.mask[task] = 1
        else:
            self.goal, self.mask, self.goal_to_render = self._compute_goal(goal, task, eval)
        obs = self._get_obs()
        return obs

    def _set_task(self, t):
        if not self.flat:
            self.task = t

    def set_flat_env(self):
        self.flat = True

    def _compute_goal(self, full_goal, task, eval):
        if self.flat:
            task = self.tasks
        else:
            task = [task]
        desired_goal = np.zeros([self.dim_g])
        if eval:
            target_range = 0.15
        else:
            target_range = self.target_range
        for t in task:
            if self.flat:
                goal = full_goal[self.tasks_g_id[t]]
            else:
                goal = full_goal
            if t == 0:  # 3D coordinates for the hand
                tmp_goal = self.initial_gripper_xpos[:3] + goal * target_range
                tmp_goal[2] = self.height_offset + (goal[2] + 1) * 3 * target_range / 2  # mapping in -1,1 to 0,0.45
                desired_goal[self.tasks_g_id[t]] = tmp_goal.copy()
                goal_to_render = tmp_goal.copy()

            elif t in [1, 3, 4, 5, 6]:  # 3D coordinates for object in 2D plane
                tmp_goal = self.initial_gripper_xpos[:3] + goal * target_range + self.target_offset
                tmp_goal[2] = self.height_offset
                desired_goal[self.tasks_g_id[t]] = tmp_goal.copy()
                goal_to_render = tmp_goal.copy()

            elif t == 2:  # 3D coordinates for the object
                tmp_goal = self.initial_gripper_xpos[:3] + goal * target_range + self.target_offset
                tmp_goal[2] = self.height_offset + (goal[2] + 1) * 3 * target_range / 2  # mapping in -1,1 to 0.0,0.45
                desired_goal[self.tasks_g_id[t]] = tmp_goal.copy()
                goal_to_render = tmp_goal.copy()

        mask = np.zeros([self.n_tasks])
        if not self.flat:
            mask[task[0]] = 1
        return desired_goal, mask, goal_to_render

    def _compute_achieved_goal(self, obs):

        achieved_goal = np.zeros([self.dim_ag])
        for i_t in range(self.n_tasks):
            achieved_goal[self.tasks_ag_id[i_t]] = obs[self.tasks_obs_id[i_t]]
        return achieved_goal


    def _get_obs(self):
        # add noise to distractor objects positions
        object0_qpos = self.sim.data.get_joint_qpos('object0:joint')

        if self.n_distractors > 0:

            object1_qpos = self.sim.data.get_joint_qpos('object1:joint')
            object2_qpos = self.sim.data.get_joint_qpos('object2:joint')

            tmp = object2_qpos[:2].copy() + np.random.randn(2) * 0.005
            i = 1
            while la.norm(tmp - object1_qpos[:2]) < 0.05 or la.norm(tmp - object0_qpos[:2]) < 0.05 or la.norm(tmp - self.object3_xpos[:2]) < 0.05 or la.norm(tmp - self.object4_xpos[:2]) < \
                    0.05:
                tmp = object2_qpos[:2].copy() + np.random.randn(2) * 0.005
                i += 1
                if i == 100:
                    tmp = object2_qpos[:2].copy()
                    break
            object2_qpos[:2] = tmp.copy()



            tmp = object1_qpos[:2].copy() + np.random.randn(2) * 0.005
            i = 1
            while la.norm(tmp - self.object3_xpos[:2]) < 0.05 or la.norm(tmp - object0_qpos[:2]) < 0.05 or la.norm(tmp - object2_qpos[:2]) < 0.05 or la.norm(tmp - self.object4_xpos[:2]) < \
                    0.05:
                tmp = object1_qpos[:2].copy() + np.random.randn(2) * 0.005
                i += 1
                if i == 100:
                    tmp = object1_qpos[:2].copy()
                    break
            object1_qpos[:2] = tmp.copy()



            tmp = object1_qpos[:2].copy() + np.random.randn(2) * 0.005
            i=1
            while la.norm(tmp - object1_qpos[:2]) < 0.05 or la.norm(tmp - object0_qpos[:2]) < 0.05 or la.norm(tmp - object2_qpos[:2]) < 0.05 or la.norm(tmp - self.object4_xpos[:2]) < \
                    0.05:
                tmp = self.object3_xpos[:2].copy() + np.random.randn(2) * 0.005
                i += 1
                if i == 100:
                    tmp = self.object3_xpos[:2].copy()
                    break
            self.object3_xpos[:2] = tmp.copy()

            tmp = self.object4_xpos[:2].copy() + np.random.randn(2) * 0.005
            i = 1
            while la.norm(tmp - object1_qpos[:2]) < 0.05 or la.norm(tmp - object0_qpos[:2]) < 0.05 or la.norm(tmp - object2_qpos[:2]) < 0.05 or la.norm(tmp - self.object3_xpos[:2]) < \
                    0.05:
                tmp = self.object4_xpos[:2].copy() + np.random.randn(2) * 0.005
                i += 1
                if i == 100:
                    tmp = self.object4_xpos[:2].copy()
                    break
            self.object4_xpos[:2] = tmp.copy()


            # for video
            object3_qpos = self.sim.data.get_joint_qpos('object3:joint')
            object4_qpos = self.sim.data.get_joint_qpos('object4:joint')
            object3_qpos[:3] = self.object3_xpos
            object4_qpos[:3] = self.object4_xpos
            self.sim.data.set_joint_qpos('object3:joint', object3_qpos)
            self.sim.data.set_joint_qpos('object4:joint', object4_qpos)
            self.sim.data.set_joint_qpos('object1:joint', object1_qpos)
            self.sim.data.set_joint_qpos('object2:joint', object2_qpos)


        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        objects_pos = []
        objects_rot = []
        objects_velp = []
        objects_velr = []
        objects_rel_pos = []
        if self.has_object:
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

            if self.n_distractors > 0:

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

        else:
            object0_pos = object0_rot = object0_velp = object0_velr = object0_rel_pos = np.zeros(0)
            if self.n_distractors > 0:
                object1_pos = object1_rot = object1_velp = object1_velr = object1_rel_pos = np.zeros(0)
                object2_pos = object2_rot = object2_velp = object2_velr = object2_rel_pos = np.zeros(0)



        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric
        if self.n_distractors > 0:

            obs = np.concatenate([grip_pos,
                                  object0_pos.ravel(), object1_pos.ravel(), object2_pos.ravel(), self.object3_xpos, self.object4_xpos,
                                  object0_rel_pos.ravel(), object1_rel_pos.ravel(), object2_rel_pos.ravel(),
                                  object0_rot.ravel(), object1_rot.ravel(), object2_rot.ravel(),
                                  object0_velp.ravel(), object1_velp.ravel(), object2_velp.ravel(),
                                  object0_velr.ravel(), object1_velr.ravel(), object2_velr.ravel(),
                                  grip_velp, gripper_vel, gripper_state])
        else:
            obs = np.concatenate([grip_pos, object0_pos.ravel(), object0_rel_pos.ravel(), object0_rot.ravel(), object0_velp.ravel(), object0_velr.ravel(),
                                  grip_velp, gripper_vel, gripper_state])



        self.last_obs = obs.copy()
        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = self._compute_achieved_goal(obs.copy())




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

            if self.n_distractors > 0:
                object1_xpos = np.array([1.9, 0.7, self.height_offset])
                object2_xpos = np.array([1.8, 1., self.height_offset])
                object3_xpos_init = np.array([1.8, 0.8, self.height_offset])
                object4_xpos_init = np.array([1.9, 0.9,  self.height_offset])

                object1_xpos = object1_xpos.copy() + np.array([np.random.uniform(-0.01, 0.01), np.random.uniform(-0.02, 0.02), 0])
                object2_xpos = object2_xpos.copy() + np.array([np.random.uniform(-0.01, 0.01), np.random.uniform(-0.02, 0.02), 0])
                self.object3_xpos = object3_xpos_init.copy() + np.array([np.random.uniform(-0.01, 0.01), np.random.uniform(-0.02, 0.02), 0])
                self.object4_xpos = object4_xpos_init.copy() + np.array([np.random.uniform(-0.01, 0.01), np.random.uniform(-0.02, 0.02), 0])


            while np.linalg.norm(object0_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object0_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)


            object0_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object0_qpos.shape == (7,)
            object0_qpos[:2] = object0_xpos
            object0_qpos[-3:] = 0
            object0_qpos[2] = self.height_offset
            self.sim.data.set_joint_qpos('object0:joint', object0_qpos)

            if self.n_distractors > 0:
                # for video
                object3_qpos = self.sim.data.get_joint_qpos('object3:joint')
                object4_qpos = self.sim.data.get_joint_qpos('object4:joint')
                object1_qpos = self.sim.data.get_joint_qpos('object1:joint')
                object2_qpos = self.sim.data.get_joint_qpos('object2:joint')
                object3_qpos[:3] = self.object3_xpos
                object4_qpos[:3] = self.object4_xpos
                object1_qpos[:3] = object1_xpos
                object2_qpos[:3] = object2_xpos
                self.sim.data.set_joint_qpos('object3:joint', object3_qpos)
                self.sim.data.set_joint_qpos('object4:joint', object4_qpos)
                self.sim.data.set_joint_qpos('object4:joint', object1_qpos)
                self.sim.data.set_joint_qpos('object4:joint', object2_qpos)

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

    def _is_success(self, achieved_goal, desired_goal, task_descr):
        return float(not self.compute_reward(achieved_goal, desired_goal, task_descr, {}))

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
            self.height_offset = 0.42599082 #height of table


    @property
    def nb_tasks(self):
        return self.n_tasks