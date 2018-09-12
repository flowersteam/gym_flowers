from __future__ import division
import gym
from gym import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl


class ModularArmV0(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self,
                 size=(3,3), # environment size
                 initial_angles=(0.,0.,0.), # initial angular position of the arm's joints
                 obj=(0, -1.4), # object location (square)
                 stick=(-0.6,0.6), # stick location
                 dist=(0.6,-0.6),
                 len_stick=0.5, # stick length
                 len_arm=(0.5,0.3,0.2), # length of the arm parts
                 action_scaling=10, # action are in +/- 180/action_scaling
                 epsilon_grasping=0.1, # precision for goal reach
                 n_timesteps=50, # number of timesteps
                 random_objects=True, # whether objects are located at random
                 distractor=False,
                 tasks = [0]
                 ):

        self.tasks = tasks.copy() # goal task, 0 is gripper pos, 1 is end stick pos, 2 is object pos
        self.distractor = distractor
        self.random_objects = random_objects
        self.action_scaling = action_scaling
        self.n_timesteps = n_timesteps
        self.len_arm = np.array(len_arm)
        self.len_stick = len_stick

        if self.distractor:
            self.tasks.append(tasks[-1] + 1)
        self.n_tasks = len(self.tasks)

        all_tasks_id = [[0,1],[2,3],[4,5],[5,6]]
        self.tasks_id = [all_tasks_id[i] for i in self.tasks]


        self.default_stick_pos_0 = np.array(stick)
        self.default_obj_pos = np.array(obj)
        self.default_dist_pos = np.array(dist)

        self.n_act = 4
        if self.distractor:
            self.n_obs = 14
        else:
            self.n_obs = 12 #3 angular position of arm, stick end, object, stick beginning and gripper open or not + stick grabbed + object grabbed

        self.gripper = -1 # open
        self.stick_grabbed = False
        self.object_grabbed = False
        self.reward_range = (-1, 0)

        # We define the spaces
        self.action_space = spaces.Box(low=-np.ones(self.n_act),
                                       high=np.ones(self.n_act),
                                       dtype=np.float32)

        self.observation_space = spaces.Dict(dict(desired_goal=spaces.Box(low=-np.ones(self.n_tasks*2)*1.5,
                                                                          high=np.ones(self.n_tasks*2)*1.5,
                                                                          dtype='float32'),
                                                  achieved_goal=spaces.Box(low=-np.ones(self.n_tasks*2)*1.5,
                                                                           high=np.ones(self.n_tasks*2)*1.5,
                                                                           dtype='float32'),
                                                  observation=spaces.Box(low=-np.ones(self.n_obs)*1.5,
                                                                         high=np.ones(self.n_obs)*1.5,
                                                                         dtype='float32'),
                                                  ))


        self.epsilon = epsilon_grasping # precision to decide whether a goal is fulfilled or not
        self.flat = False
        self.viewer = None


        # We set to None to rush error if reset not called
        self.reward = None
        self.observation = None
        self.done = None
        self.desired_goal = np.zeros([self.n_tasks*2])
        self.task = 0
        self.achieved_goal = None
        self.info = dict(is_success=0)

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        return seed


    def compute_reward(self, achieved_goal, goal, info=None):
        if achieved_goal.ndim > 1:
            d = np.zeros([goal.shape[0]])
            for i in range(goal.shape[0]):
                ind = np.argwhere(goal[i, :] != 0).squeeze()
                d[i] = np.linalg.norm(achieved_goal[i, ind] - goal[i, ind], ord=2)
            return -(d > self.epsilon).astype(np.int).reshape([d.shape[0], 1])
        else:
            ind = np.argwhere(goal != 0)
            d = np.linalg.norm(achieved_goal[ind] - goal[ind], ord=2)
            return -(d > self.epsilon).astype(np.int)


    def _set_task(self, t):
        if not self.flat:
            self.task = t

    def set_flat_env(self):
        self.flat = True

    def _set_desired_goal(self, g):

        self.desired_goal = np.zeros([self.n_tasks*2])
        if not self.flat:
            coeff = 1 if self.task==0 else 1.5
            self.desired_goal[self.tasks_id[self.task]] = g.copy()*coeff
        else:
            self.desired_goal[0:2] = g[0:2]
            self.desired_goal[2:] = g[2:] * 1.5


    def compute_achieved_goal(self, obs, task):
        achieved_goal = np.zeros([self.n_tasks*2])
        if 0 in task:
            # grip position is the achieved goal
            angles = np.cumsum(obs[:3])
            angles_rads = np.pi * angles
            achieved_goal[:2] = np.array([np.sum(np.cos(angles_rads) * self.len_arm),
                                         np.sum(np.sin(angles_rads) * self.len_arm)])
        if 1 in task:
            # end_stick_pos is the achieved goal
            achieved_goal[2:4] = obs[3:5]
        if 2 in task:
            # obj_pos is the achieved goal
            achieved_goal[4:6] = obs[5:7]
        if 3 in task:
            #distractor pos is the achieved goal
            achieved_goal[6:8] = obs[12:14]

        return achieved_goal


    def reset(self):
        # We reset the simulation
        if self.random_objects:
            while True:
                self.stick_pos_0 = (np.random.uniform(-1, 1, 2))
                if self.stick_pos_0[0]**2 + self.stick_pos_0[1]**2 < 1 and self.stick_pos_0[0]**2 + self.stick_pos_0[1]**2 > 0.8**2 and self.stick_pos_0[0]<0:
                    break
            while True:
                self.object_pos = (np.random.uniform(-1.5, 1.5, 2))
                if self.object_pos[0]**2 + self.object_pos[1]**2 < 1.5**2 and self.object_pos[0]**2 + self.object_pos[1]**2 > 1.1:
                    break
            if self.distractor:
                while True:
                    self.distractor_pos = (np.random.uniform(-1.5, 1.5, 2))
                    if self.distractor_pos[0] ** 2 + self.distractor_pos[1] ** 2 < 1.5 ** 2:
                        break
        else:
            self.stick_pos_0 = np.copy(self.default_stick_pos_0) + np.random.uniform(-0.1, 0.1, 2)
            self.object_pos = np.copy(self.default_obj_pos) + np.random.uniform(-0.1, 0.1, 2)
            if self.distractor:
                self.distractor_pos = np.copy(self.default_dist_pos) + np.random.uniform(-0.1, 0.1, 2)

        self.stick_pos = np.array([self.stick_pos_0[0] + self.len_stick * np.cos(3*np.pi / 4),
                                   self.stick_pos_0[1] + self.len_stick * np.sin(3*np.pi / 4)])

        self.gripper = -1 #open
        self.stick_grabbed = False
        self.object_grabbed = False
        self.reward = 0

        # Initialize angular arm_pos and compute gripper cartesian position
        self.arm_pos = np.random.uniform(-0.1,0.1,3)
        angles = np.cumsum(self.arm_pos)
        angles_rads = np.pi * angles
        self.grip_pos = np.array([np.sum(np.cos(angles_rads) * self.len_arm),
                                  np.sum(np.sin(angles_rads) * self.len_arm)])

        # check whether stick is grabbed
        if not self.stick_grabbed:
            if np.linalg.norm(self.stick_pos_0 - self.grip_pos, ord=2) < self.epsilon and self.gripper == -1:
                self.stick_grabbed = True
                self.stick_pos_0 = self.grip_pos

        if self.stick_grabbed:
            self.stick_pos_0 = self.grip_pos
            # place stick in the continuity of the arm
            self.stick_pos = np.copy(self.stick_pos_0 + self.len_stick * np.array([np.cos(angles_rads[-1]), -np.sin(angles_rads[-1])]))
            # check whether object is grabbed
            if not self.object_grabbed:
                if np.linalg.norm(self.object_pos - self.stick_pos, ord=2) < self.epsilon:
                    self.object_grabbed = True
            if self.object_grabbed:
                self.object_pos = self.stick_pos

        stick_grabbed = 1 if self.stick_grabbed else -1
        object_grabbed = 1 if self.object_grabbed else -1
        # construct vector of observations
        self.observation = np.concatenate([self.arm_pos, self.stick_pos, self.object_pos, self.stick_pos_0, np.array([self.gripper, stick_grabbed, object_grabbed])])
        if self.distractor:
            self.observation = np.concatenate([self.observation, self.distractor_pos])
        self.steps = 0
        self.done = False
        return self.observation

    def reset_task_goal(self, goal, task=None):

        self._set_task(task)
        self._set_desired_goal(goal)

        # fill achieved_goal depending on goal task
        self.achieved_goal = self.compute_achieved_goal(self.observation, self.tasks)
        self.mask = np.zeros([self.n_tasks])
        self.mask[self.task] = 1
        self.obs_out = dict(observation=self.observation, achieved_goal=self.achieved_goal, desired_goal=self.desired_goal, mask=self.mask)

        return self.obs_out


    def step(self, action):
        """Run one timestep of the environment's dynamics.
        """
        action = np.copy(action.clip(-1,1))
        grip = np.copy(action[-1])
        
        # We compute the position of the end effector
        self.arm_pos = np.clip((self.arm_pos + action[:-1] / self.action_scaling + 1) % 2 -1,
                               a_min=-np.ones(self.n_act-1),
                               a_max=np.ones(self.n_act-1))
        angles = np.cumsum(self.arm_pos)
        angles_rads = np.pi * angles
        self.grip_pos = np.array([np.sum(np.cos(angles_rads) * self.len_arm),
                                  np.sum(np.sin(angles_rads) * self.len_arm)])

        if self.distractor:
            self.distractor_pos = np.clip(self.distractor_pos + np.random.uniform(-0.1,0.1,2), -1, 1)

        if grip>0:
            self.gripper = 1
        else:
            self.gripper = -1

        # check whether stick is grabbed
        if not self.stick_grabbed:
            if np.linalg.norm(self.stick_pos_0 - self.grip_pos, ord=2) < self.epsilon and self.gripper == -1:
                self.stick_grabbed = True
                self.stick_pos_0 = self.grip_pos

        if self.stick_grabbed:
            self.stick_pos_0 = self.grip_pos
            # place stick in the continuity of the arm
            self.stick_pos = np.array([np.sum(np.cos(angles_rads) * self.len_arm) + np.cos(angles_rads[-1]) *self.len_stick,
                                       np.sum(np.sin(angles_rads) * self.len_arm) + np.sin(angles_rads[-1]) *self.len_stick])
            # check whether object is grabbed
            if not self.object_grabbed:
                if np.linalg.norm(self.object_pos - self.stick_pos, ord=2) < self.epsilon:
                    self.object_grabbed = True
            if self.object_grabbed:
                self.object_pos = self.stick_pos

        # We update observation and reward
        stick_grabbed = 1 if self.stick_grabbed else -1
        object_grabbed = 1 if self.object_grabbed else -1
        # construct vector of observations
        self.observation = np.concatenate([self.arm_pos, self.stick_pos, self.object_pos, self.stick_pos_0, np.array([self.gripper, stick_grabbed, object_grabbed])])
        if self.distractor:
            self.observation = np.concatenate([self.observation, self.distractor_pos])
        self.achieved_goal = self.compute_achieved_goal(self.observation, self.tasks)
        self.mask = np.zeros([self.n_tasks])
        self.mask[self.task] = 1
        self.obs_out = dict(observation=self.observation, achieved_goal=self.achieved_goal, desired_goal=self.desired_goal, mask=self.mask)
        self.reward = self.compute_reward(self.achieved_goal, self.desired_goal)

        self.info = dict(is_success= self.reward == 0)
        self.steps += 1
        if self.steps == self.n_timesteps:
            self.done = True

        return self.obs_out, float(self.reward), self.done, self.info

    def render(self, mode='human', close=False):
        """Renders the environment.

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with
            close (bool): close all open renderings
        """

        if self.viewer is None:
            self.start_viewer()
        plt.clf()
        plt.xlim([-1.5, 1.5])
        plt.ylim([1.5, -1.5])
        plt.xticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
        plt.yticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
        plt.gca().invert_yaxis()

        small_circle = 0.03
        large_circle = 0.05

        arm_angles = np.cumsum(self.arm_pos)
        arm_angles = np.pi * arm_angles
        arm_points = np.array([np.cumsum(np.cos(arm_angles) * self.len_arm),
                               np.cumsum(np.sin(arm_angles) * self.len_arm)]).transpose()

        # draw arm parts
        l = plt.Line2D([arm_points[0,0],0], [arm_points[0,1], 0], color=(0.5,0.5,0.5),linewidth=2)
        plt.gca().add_line(l)
        l = plt.Line2D([arm_points[0, 0],  arm_points[1, 0]],
                          [arm_points[0, 1],  arm_points[1, 1]], color=(0.5,0.5,0.5), linewidth=2)
        plt.gca().add_line(l)
        l = plt.Line2D([arm_points[1, 0], arm_points[2, 0]],
                          [arm_points[1, 1],arm_points[2, 1]], color=(0.5,0.5,0.5), linewidth=2)
        plt.gca().add_line(l)
        #draw arm joints
        j = mpl.patches.Circle((0, 0), radius=small_circle, fc=(0, 0, 0), zorder=10)
        plt.gca().add_patch(j)
        j = mpl.patches.Circle(tuple(arm_points[0,:]), radius=small_circle, fc=(0, 0, 0), zorder=10)
        plt.gca().add_patch(j)
        j = mpl.patches.Circle(tuple(arm_points[1, :]), radius=small_circle, fc=(0, 0, 0), zorder=10)
        plt.gca().add_patch(j)
        # add gripper
        j = mpl.patches.Circle(tuple(arm_points[2, :]), radius=large_circle, fc=(1, 128/255, 0), zorder=10)
        plt.gca().add_patch(j)

        # draw goal
        j = mpl.patches.Circle(tuple(self.desired_goal[self.task*2: 2*(self.task+1)]), radius=self.epsilon, fc=(1, 0, 0), zorder=2)
        plt.gca().add_patch(j)

        # draw stick
        if self.stick_grabbed:
            l = plt.Line2D([arm_points[2, 0], self.stick_pos[0]],
                           [arm_points[2, 1], self.stick_pos[1]], linewidth=2)
            plt.gca().add_line(l)
            j = mpl.patches.Circle(tuple(self.stick_pos), radius=small_circle   , fc=(102 / 255, 0, 204/255), zorder=20)
            plt.gca().add_patch(j)
        else:
            l = plt.Line2D(
                [self.stick_pos_0[0], self.stick_pos[0]], [self.stick_pos_0[1], self.stick_pos[1]], linewidth=2)
            plt.gca().add_line(l)
            j = mpl.patches.Circle(tuple(self.stick_pos), radius=small_circle, fc=(102 / 255, 0, 204/255), zorder=15)
            plt.gca().add_patch(j)
            j = mpl.patches.Circle(tuple(self.stick_pos_0), radius=small_circle, fc=(1, 128/255, 0), zorder=15)
            plt.gca().add_patch(j)

        # draw object
        if self.object_grabbed:
            obj = mpl.patches.Rectangle(tuple(self.object_pos-0.05), 0.1, 0.1, fc=(102 / 255, 0, 204/255), zorder=20)
            plt.gca().add_patch(obj)
        else:
            obj = mpl.patches.Rectangle(tuple(self.object_pos-0.05), 0.1, 0.1, fc=(102 / 255, 0, 204/255), zorder=20)
            plt.gca().add_patch(obj)

        # draw distractor
        if self.distractor:
            obj = mpl.patches.Rectangle(tuple(self.distractor_pos - 0.05), 0.1, 0.1, fc=(0,153/255, 0), zorder=25)
            plt.gca().add_patch(obj)

        if mode == 'rgb_array':
            return self.rendering  # return RGB frame suitable for video
        elif mode is 'human':
            plt.pause(0.01)
            plt.draw()

    def start_viewer(self):
        plt.ion()
        self.viewer = plt.figure(figsize=(5, 5), frameon=False)



    def close(self):
        if self.viewer is not None:
            plt.close(self.viewer)
            self.viewer = None

    @property
    def dim_goal(self):
        return 6

    @property
    def nb_tasks(self):
        return self.n_tasks

    @property
    def current_task(self):
        return self.task

