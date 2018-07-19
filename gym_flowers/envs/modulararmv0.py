from __future__ import division
import gym2
from gym2 import spaces
import numpy as np
import random
import gizeh
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl
class ModularArmV0(gym2.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self,
                 size=(3,3),
                 initial_angles=(0.,0.,0.),
                 obj=(-0.3, 1.1),
                 stick=(-0.75,0.25),
                 len_stick=0.5,
                 len_arm=(0.5,0.3,0.2),
                 render=True,
                 action_scaling=5,
                 goal_strategy='random',
                 epsilon_grasping=0.1,
                 n_timesteps=75,
                 random_objects=False
                 ):

        self.random_objects = random_objects
        self.action_scaling = action_scaling
        self.goal_strategy = goal_strategy
        self.n_act = 4
        self.n_obs = 9 #2D loc of gripper, stick end, object, stick beginning and gripper open or not
        self._n_timesteps = n_timesteps
        self.initial_angles = np.array(initial_angles)

        if self.random_objects:
            while True:
                self.initial_stick_pos_0 = (np.random.random(2)-0.5)*2
                if self.initial_stick_pos_0[0]**1 + self.initial_stick_pos_0[1]**2 < 1:
                    break
            while True:
                self.initial_obj_pos = (np.random.random(2)-0.5)*2
                if self.initial_obj_pos[0]**1 + self.initial_obj_pos[1]**2 < 1:
                    break
        else:
            self.initial_stick_pos_0 = np.array(stick)
            self.initial_obj_pos = np.array(obj)
        self.initial_stick_pos = np.array([self.initial_stick_pos_0[0] - len_stick * np.sin(np.pi / 4),
                                           self.initial_stick_pos_0[1] + len_stick * np.cos(np.pi / 4)])
        self.initial_grip_pos = np.array([np.dot(-np.sin(initial_angles), len_arm).sum(),np.dot(np.cos(initial_angles), len_arm).sum()])
        self.len_arm = len_arm
        self.len_stick = len_stick
        self.gripper = -1 # open
        self.stick_grabbed = False
        self.object_grabbed = False


        # We set the space
        self.action_space = spaces.Box(low=-np.ones(self.n_act) / action_scaling,
                                       high=np.ones(self.n_act) / action_scaling,
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.ones(self.n_obs)*1.5,
                                            high=np.ones(self.n_obs)*1.5,
                                            dtype=np.float32)

        self.module = 0 # goal module, 0 is gripper pos, 1 is stick pos, 2 is object pos
        self.ind_goal = [[0,1], [2,3], [4,5]] # indexes of observation for each goal module
        self.epsilon = epsilon_grasping # precision to decide whether a goal is fulfilled or not

        self.viewer = None

        # We set to None to rush error if reset not called
        self.reward = None
        self.observation = None
        self.done = None
        self.desired_goal = None
        self.achieved_goal = None
        self.module = 0
        self.n_modules = 3


    def reset(self, goal=None):
        # We reset the simulation
        if self.random_objects:
            while True:
                self.initial_stick_pos_0 = (np.random.random(2)-0.5)*2
                if self.initial_stick_pos_0[0]**1 + self.initial_stick_pos_0[1]**2 < 1:
                    break
            while True:
                self.initial_obj_pos = (np.random.random(2)-0.5)*2
                if self.initial_obj_pos[0]**1 + self.initial_obj_pos[1]**2 < 1:
                    break
        self.initial_stick_pos = np.array([self.initial_stick_pos_0[0] - self.len_stick * np.sin(np.pi / 4),
                                           self.initial_stick_pos_0[1] + self.len_stick * np.cos(np.pi / 4)])

        self.stick_pos = np.copy(self.initial_stick_pos)
        self.stick_pos_0 = np.copy(self.initial_stick_pos_0)
        self.object_pos = np.copy(self.initial_obj_pos)
        self.arm_angles = np.copy(self.initial_angles)

        self.gripper = -1 #open
        self.stick_grabbed = False
        self.object_grabbed = False
        self.reward = 0
        joint_1 = self.len_arm[0] * np.array([-np.sin(self.arm_angles[0]), np.cos(self.arm_angles[0])]).reshape(1,-1)
        joint_2 = joint_1 + self.len_arm[1] * np.array([-np.sin(self.arm_angles[1]), np.cos(self.arm_angles[1])]).reshape(1,-1)
        joint_3 = joint_2 + self.len_arm[2] * np.array([-np.sin(self.arm_angles[2]), np.cos(self.arm_angles[2])]).reshape(1,-1)
        self.arm_pos = np.concatenate([joint_1, joint_2,joint_3], axis=0)
        self.grip_pos = joint_3.squeeze()

        self.observation = np.concatenate([self.grip_pos, self.stick_pos, self.object_pos, self.stick_pos_0, np.array([self.gripper])])

        self.ind = self.ind_goal[self.module]
        self.desired_goal = np.zeros([6])
        self.achieved_goal = np.zeros([6])
        self.desired_goal[self.ind] = self._sample_goal()
        self.achieved_goal[self.ind] = np.copy(self.observation[self.ind])
        self.obs = dict(observation=self.observation, achieved_goal=self.achieved_goal, desired_goal=self.desired_goal)
        self.steps = 0
        self.done = False

        return self.obs

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        return seed

    def _sample_goal(self):
        if self.module == 0:
            while True:
                goal = (np.random.random(2)-0.5)*2
                if goal[0]**2 + goal[1]**2 < 1:
                    break
        elif self.module == 2 or self.module==1:
            while True:
                goal = (np.random.random(2)-0.5)*3
                if goal[0]**2 + goal[1]**2 < 1.5**2:
                    break
        return goal

    def compute_reward(self, achieved_goal, goal, info=None):
        if achieved_goal.ndim > 1:
            d = np.linalg.norm(achieved_goal - goal, ord=2, axis=1)
        else:
            d = np.linalg.norm(achieved_goal - goal, ord=2)
        return -(d > self.epsilon).astype(np.int)

    def sample_module(self):
        # we sample a goal module
        if self.goal_strategy == 'random':
            self.module = np.random.randint(0, 3)
            print('Module #', self.module)
            # self.module=2

    def set_module(self, module):
        # we sample a goal module
        self.module = module
        print('Module #', self.module)
        # self.module=2

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        """

        angles = action[:-1]
        grip = action[-1]
        # update arm pos
        self.arm_angles += (angles/self.action_scaling) * np.pi
        # self.arm_angles = np.clip(self.arm_angles, -np.pi, np.pi)
        joint_1 = self.len_arm[0] * np.array([-np.sin(self.arm_angles[0]), np.cos(self.arm_angles[0])]).reshape(1, -1)
        joint_2 = joint_1 + self.len_arm[1] * np.array([-np.sin(self.arm_angles[1]), np.cos(self.arm_angles[1])]).reshape(1, -1)
        joint_3 = joint_2 + self.len_arm[2] * np.array([-np.sin(self.arm_angles[2]), np.cos(self.arm_angles[2])]).reshape(1, -1)
        self.arm_pos = np.concatenate([joint_1, joint_2, joint_3], axis=0)
        self.grip_pos = joint_3.squeeze()

        if grip>0:
            self.gripper = 1
        else:
            self.gripper = -1

        # check whether stick is grabbed
        if not self.stick_grabbed:
            if np.linalg.norm(self.initial_stick_pos_0-self.grip_pos, ord=2) < self.epsilon and self.gripper==-1:
                self.stick_grabbed = True
                self.stick_pos_0 = self.grip_pos

        if self.stick_grabbed:
            self.stick_pos = joint_3.squeeze() + self.len_stick * np.array([-np.sin(self.arm_angles[2]), np.cos(self.arm_angles[2])])
            if not self.object_grabbed:
                if np.linalg.norm(self.initial_obj_pos - self.stick_pos, ord=2) < self.epsilon:
                    self.object_grabbed = True
            if self.object_grabbed:
                self.object_pos = self.stick_pos

        # We update observation and reward
        self.observation = np.concatenate([self.grip_pos, self.stick_pos, self.object_pos, self.stick_pos_0, np.array([self.gripper])])

        self.achieved_goal = np.zeros([6])
        self.achieved_goal[self.ind] = np.copy(self.observation[self.ind])
        self.obs = dict(observation=self.observation, achieved_goal=self.achieved_goal, desired_goal=self.desired_goal)
        self.steps += 1


        info = {}
        info['is_success'] = self.compute_reward(self.achieved_goal, self.desired_goal) == 0
        if info['is_success']:
            print('Success !')
        if self.reward == 0:
            self._done = True
        return self.obs, self.reward, self._done, info

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

        small_circle = 0.03
        large_circle = 0.05
        # draw arm parts
        l = plt.Line2D([self.arm_pos[0,0],0], [self.arm_pos[0,1], 0], color=(0.5,0.5,0.5),linewidth=2)
        plt.gca().add_line(l)
        l = plt.Line2D([self.arm_pos[0, 0],  self.arm_pos[1, 0]],
                          [self.arm_pos[0, 1],  self.arm_pos[1, 1]], color=(0.5,0.5,0.5), linewidth=2)
        plt.gca().add_line(l)
        l = plt.Line2D([self.arm_pos[1, 0], self.arm_pos[2, 0]],
                          [self.arm_pos[1, 1],self.arm_pos[2, 1]], color=(0.5,0.5,0.5), linewidth=2)
        plt.gca().add_line(l)
        #draw arm joints
        j = mpl.patches.Circle((0, 0), radius=small_circle, fc=(0, 0, 0), zorder=10)
        plt.gca().add_patch(j)
        j = mpl.patches.Circle(tuple(self.arm_pos[0,:]), radius=small_circle, fc=(0, 0, 0), zorder=10)
        plt.gca().add_patch(j)
        j = mpl.patches.Circle(tuple(self.arm_pos[1, :]), radius=small_circle, fc=(0, 0, 0), zorder=10)
        plt.gca().add_patch(j)
        # add gripper
        j = mpl.patches.Circle(tuple(self.arm_pos[2, :]), radius=large_circle, fc=(1, 128/255, 0), zorder=10)
        plt.gca().add_patch(j)

        # draw goal
        j = mpl.patches.Circle(tuple(self.desired_goal[self.ind]), radius=self.epsilon, fc=(1, 0, 0), zorder=2)
        plt.gca().add_patch(j)

        # draw stick
        if self.stick_grabbed:
            l = plt.Line2D([self.arm_pos[2, 0], self.stick_pos[0]],
                           [self.arm_pos[2, 1], self.stick_pos[1]], linewidth=2)
            plt.gca().add_line(l)
            j = mpl.patches.Circle(tuple(self.stick_pos), radius=small_circle   , fc=(102 / 255, 0, 204/255), zorder=20)
            plt.gca().add_patch(j)
        else:
            l = plt.Line2D(
                [self.initial_stick_pos_0[0], self.stick_pos[0]], [self.initial_stick_pos_0[1], self.stick_pos[1]], linewidth=2)
            plt.gca().add_line(l)
            j = mpl.patches.Circle(tuple(self.stick_pos), radius=small_circle, fc=(102 / 255, 0, 204/255), zorder=15)
            plt.gca().add_patch(j)
            j = mpl.patches.Circle(tuple(self.initial_stick_pos_0), radius=small_circle, fc=(1, 128/255, 0), zorder=15)
            plt.gca().add_patch(j)

        # draw object
        if self.object_grabbed:
            obj = mpl.patches.Rectangle(tuple(self.stick_pos-0.05), 0.1, 0.1, fc=(102 / 255, 0, 204/255), zorder=20)
            plt.gca().add_patch(obj)
        else:
            obj = mpl.patches.Rectangle(tuple(self.initial_obj_pos-0.05), 0.1, 0.1, fc=(102 / 255, 0, 204/255), zorder=20)
            plt.gca().add_patch(obj)

        if mode == 'rgb_array':
            return self.rendering  # return RGB frame suitable for video
        elif mode is 'human':
            plt.gca().invert_yaxis()
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
    def nb_modules(self):
        return self.n_modules

    @property
    def current_module(self):
        return self.module


