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
                 obj=(-0.5, 1.1), # object location (square)
                 stick=(0.75,0.75), # stick location
                 len_stick=0.5, # stick length
                 len_arm=(0.5,0.3,0.2), # length of the arm parts
                 action_scaling=10, # action are in +/- 180/action_scaling
                 goal_strategy='random', # strategy for goal selection
                 epsilon_grasping=0.1, # precision for goal reach
                 n_timesteps=75, # number of timesteps
                 random_objects=False # whether objects are located at random
                 ):
        
        self.random_objects = random_objects
        self.action_scaling = action_scaling
        self.goal_strategy = goal_strategy
        self.n_act = 4
        self.n_obs = 10 #3 angular position of arm, stick end, object, stick beginning and gripper open or not
        self._n_timesteps = n_timesteps
        self.initial_angles = np.random.random(3)*0.1

        # determine location of objects (stick and object (square))
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
        self.len_arm = np.array(len_arm)
        self.len_stick = len_stick
        self.gripper = -1 # open
        self.stick_grabbed = False
        self.object_grabbed = False


        # We define the spaces
        self.action_space = spaces.Box(low=-np.ones(self.n_act),
                                       high=np.ones(self.n_act),
                                       dtype=np.float32)

        self.observation_space = spaces.Dict(dict(desired_goal=spaces.Box(low=-np.ones(6)*1.5,
                                                                          high=np.ones(6)*1.5,
                                                                          dtype='float32'),
                                                  achieved_goal=spaces.Box(low=-np.ones(6)*1.5,
                                                                           high=np.ones(6)*1.5,
                                                                           dtype='float32'),
                                                  observation=spaces.Box(low=-np.ones(self.n_obs)*1.5,
                                                                         high=np.ones(self.n_obs)*1.5,
                                                                         dtype='float32'),
                                                  ))


        self.module = 2 # goal module, 0 is gripper pos, 1 is end stick pos, 2 is object pos
        self.ind_goal = [[0,1], [3,4], [5,6]] # indexes of observation for each goal module
        self.epsilon = epsilon_grasping # precision to decide whether a goal is fulfilled or not

        self.viewer = None

        # We set to None to rush error if reset not called
        self.reward = None
        self.observation = None
        self.done = None
        self.desired_goal = None
        self.achieved_goal = None
        self.n_modules = 3


    def reset(self, goal=None):
        self.set_module(0)
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

        self.gripper = -1 #open
        self.stick_grabbed = False
        self.object_grabbed = False
        self.reward = 0

        # Initialize angular arm_pos and compute gripper cartesian position
        self.arm_pos = np.zeros([3]) #np.random.uniform(-0.1,0.1,3)
        angles = np.cumsum(self.arm_pos)
        angles_rads = np.pi * angles
        self.grip_pos = np.array([np.sum(np.cos(angles_rads) * self.len_arm),
                                  np.sum(np.sin(angles_rads) * self.len_arm)])

        # check whether stick is grabbed
        if not self.stick_grabbed:
            if np.linalg.norm(self.initial_stick_pos_0 - self.grip_pos, ord=2) < self.epsilon and self.gripper == -1:
                self.stick_grabbed = True
                self.stick_pos_0 = self.grip_pos

        if self.stick_grabbed:
            self.stick_pos = np.copy(self.stick_pos_0 + self.len_stick * np.array([np.cos(angles_rads[-1]), -np.sin(angles_rads[-1])]))
            if not self.object_grabbed:
                if np.linalg.norm(self.initial_obj_pos - self.stick_pos, ord=2) < self.epsilon:
                    self.object_grabbed = True
            if self.object_grabbed:
                self.object_pos = self.stick_pos
                
        # construct vector of observations
        self.observation = np.concatenate([self.arm_pos, self.stick_pos, self.object_pos, self.stick_pos_0, np.array([self.gripper])])

        self.ind = self.ind_goal[self.module]
        
        # Sample desired_goal and fill achieved_goal depending on goal module
        self.desired_goal = np.zeros([6])
        self.achieved_goal = np.zeros([6])
        self.desired_goal[self.module*2: 2*(self.module+1)] = self._sample_goal()
        if self.module == 0:
            self.achieved_goal[self.ind] = self.grip_pos
        else:
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

    def set_module(self, module):
        # we sample a goal module
        self.module = module


    def step(self, action):
        """Run one timestep of the environment's dynamics.
        """
        action = np.copy(action.clip(-1,1))
        grip = np.copy(action[-1])
        
        # We compute the position of the end effector
        self.arm_pos = np.clip(self.arm_pos + action[:-1] / self.action_scaling,
                                a_min=-np.ones(self.n_act-1),
                                a_max=np.ones(self.n_act-1))
        angles = np.cumsum(self.arm_pos)
        angles_rads = np.pi * angles
        self.grip_pos = np.array([np.sum(np.cos(angles_rads) * self.len_arm),
                                   np.sum(np.sin(angles_rads) * self.len_arm)])

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
            self.stick_pos = np.copy(self.grip_pos + self.len_stick * np.array([np.cos(angles_rads[-1]), np.sin(angles_rads[-1])]))
            if not self.object_grabbed:
                if np.linalg.norm(self.initial_obj_pos - self.stick_pos, ord=2) < self.epsilon:
                    self.object_grabbed = True
            if self.object_grabbed:
                self.object_pos = self.stick_pos

        # We update observation and reward
        self.observation = np.concatenate([self.arm_pos, self.stick_pos, self.object_pos, self.stick_pos_0, np.array([self.gripper])])

        self.achieved_goal = np.zeros([6])
        if self.module == 0:
            self.achieved_goal[self.ind] = self.grip_pos
        else:
            self.achieved_goal[self.ind] = np.copy(self.observation[self.ind])
        self.obs = dict(observation=self.observation, achieved_goal=self.achieved_goal, desired_goal=self.desired_goal)

        self.reward = self.compute_reward(self.achieved_goal, self.desired_goal)


        info = {}
        info['is_success'] = self.reward == 0
        self.steps += 1
        if self.steps == self._n_timesteps:
            self.done = True

        return self.obs, float(self.reward), self.done, info

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
        j = mpl.patches.Circle(tuple(self.desired_goal[self.ind]), radius=self.epsilon, fc=(1, 0, 0), zorder=2)
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
            # plt.gca().invert_yaxis()
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


