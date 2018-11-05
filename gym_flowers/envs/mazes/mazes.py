import os
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import collections as mc
import random
import pickle
import time

class Maze(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, maze_id='maze_pic_0', reward_type='sparse'):
        """
        Initializes a new maze environment from descriptor.
        """
        self.reward_type = reward_type
        loading_path = './mazes/' + maze_id
        assert os.path.exists(loading_path + '_descriptor.txt'), "No maze descriptor file found at " + loading_path
        self.descriptor = np.loadtxt(loading_path + '_descriptor.txt')

        with open(loading_path + '_params.pkl', 'rb') as f:
            params = pickle.load(f)

        self.agent_pos = np.array(params['agent_position'])
        self.initial_agent_pos = self.agent_pos.copy()
        self.agent_size = params['scale'] * 2
        self.direction = 180
        self.rangefinder_dir = np.array([-90, -45, 0, 45, 90, 180]) + self.direction
        self.rangefinder_max = 3 * self.agent_size
        self.rangefinder = self.rangefinder_max * np.ones([len(self.rangefinder_dir)])
        self.show_range_finder = True
        self.lines = []

        self.nb_radar = 4
        self.angle_limits = (np.arange(0, 360, 360 / self.nb_radar) + self.direction ) % 360

        self.nb_act = 2
        self.nb_obs = 12
        self.max_direction_shift = 45
        self.max_step_size = self.agent_size // 4
        self.max_timesteps = 500
        self.t = 0

        if self.reward_type == 'sparse':
            self.reward_range = (-1, 0)
        else:
            self.reward_range = (-2, 0)

        # We set the space
        self.action_space = spaces.Box(low=-np.ones(self.nb_act),
                                       high=np.ones(self.nb_act),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.ones(self.nb_obs),
                                            high=np.ones(self.nb_obs),
                                            dtype=np.float32)

        self.width = self.descriptor.shape[1]
        self.height = self.descriptor.shape[0]

        self.renderer = None

        # We set to None to rush error if reset not called
        self.reward = None
        self.obs = None
        self.done = None

    def my_init(self, maze_id):
        self.__init__(maze_id)


    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        return seed

    def _sample_goal(self):
        # draw goals at random until it doesn't collide any wall
        while True:
            goal = np.array([np.random.uniform(0, self.height), np.random.uniform(0, self.width)]).astype(np.int)
            try:
                square = self.descriptor[goal[0] - self.agent_size // 2: goal[0] + self.agent_size // 2, goal[1] - self.agent_size // 2: goal[1] + self.agent_size // 2]
                if square.sum() == 0:
                    break
            except:
                pass
        self.goal = goal

    def compute_reward(self):
        if self.reward_type == 'sparse':
            if np.linalg.norm(self.agent_pos - self.goal) < self.agent_size:
                return 1
            else:
                return 0
        else:
            return np.linalg.norm(self.agent_pos - self.goal)

    def reset(self, goal=None):
        self._sample_goal()
        self.agent_pos = self.initial_agent_pos.copy()
        self.obs = self.agent_pos.copy()
        self.done = False
        self.t = 0
        self.compute_obs()


        return self.obs

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        """
        step_size = action[0] * self.max_step_size
        self.direction = self.direction + action[1] * self.max_direction_shift % 360
        dx = np.int(step_size * np.cos(self.direction * np.pi / 180))
        dy = - np.int(step_size * np.sin(self.direction * np.pi / 180))

        y, x = self.agent_pos
        s = self.agent_size // 2

        # check collisions with border
        square_of_concern = self.descriptor[y + dy - s: y + dy + s, x + dx - s: x + dx + s]
        if square_of_concern.sum() != 0:
            delta_x = 0
            delta_y = 0
            sum_tmp = 0
            for i in range(min(dx, 0), max(dx, 0)):
                for j in range(min(dy, 0), max(dy, 0)):
                    if self.descriptor[y + j - s: y + j + s, x + i - s: x + i + s].sum() == 0:
                        if np.abs(i) + np.abs(j) > sum_tmp:
                            sum_tmp = np.abs(i) + np.abs(j)
                            delta_x = i
                            delta_y = j
        else:
            delta_x = dx
            delta_y = dy

        self.agent_pos[0] += delta_y
        self.agent_pos[1] += delta_x
        self.compute_obs()

        self.reward = self.compute_reward()

        self.t += 1
        if self.t == self.max_timesteps:
            self.done = True

        return self.obs, self.reward, self.done, {}

    def compute_obs(self):

        # compute rangefinders
        self.rangefinder_dir = np.array([-90, -45, 0, 45, 90, 180]) + self.direction
        self.rangefinder = [self.rangefinder_max for _ in range(self.rangefinder_dir.size)]
        self.lines = []
        for i in range(self.rangefinder_dir.size):
            found = False
            dx = np.int(self.rangefinder_max * np.cos(self.rangefinder_dir[i] * np.pi / 180))
            dy = - np.int(self.rangefinder_max * np.sin(self.rangefinder_dir[i] * np.pi / 180))
            d = np.array([dy, dx])
            ind_main = np.argmax(np.abs(d))  # main axis
            ind_sec = int(ind_main == 0)  # secondary axis
            last_sec = 0
            new_last_sec = 0
            if ind_main == 1:
                dirs = self.rangefinder_dir * np.pi / 180
            else:
                dirs = (self.rangefinder_dir - 90) * np.pi / 180

            m_new = 2 * np.abs(d[ind_sec])
            slope_error_new = m_new - np.abs(d[ind_main])
            y = 0
            for x in range(np.abs(d[ind_main] + 1)):
                signed_y = np.sign(d[ind_sec]) * y
                signed_x = np.sign(d[ind_main]) * x
                x_test = self.agent_pos[0] + ind_main * signed_y + (1 - ind_main) * signed_x
                y_test = self.agent_pos[1] + (1 - ind_main) * signed_y + ind_main * signed_x
                if self.descriptor[x_test, y_test] == 1:
                    self.rangefinder[i] = np.sqrt(x ** 2 + y ** 2)
                    found = True
                    break
                else:
                    slope_error_new = slope_error_new + m_new
                    if slope_error_new >= 0:
                        y += 1
                        slope_error_new = slope_error_new - 2 * np.abs(d[ind_main])

            range_x = self.rangefinder[i] * np.cos(self.rangefinder_dir[i] * np.pi / 180)
            range_y = - self.rangefinder[i] * np.sin(self.rangefinder_dir[i] * np.pi / 180)
            self.lines.append([(self.agent_pos[1], self.agent_pos[0]), (self.agent_pos[1] + range_x, self.agent_pos[0] + range_y)])

        # compute pie slide radars
        # compute angle between agent and goal
        self.angle_limits = (np.arange(-90, 270, 360 / self.nb_radar) + self.direction ) % 360
        dx = self.goal[1] - self.agent_pos[1]
        dy = - (self.goal[0] - self.agent_pos[0])
        theta = np.arctan(dy / dx) * 180 / np.pi
        if dx < 0:
            theta = (theta + 180) % 360
        elif dy < 0:
            theta = theta + 360
        ind_sort = np.argsort(self.angle_limits)
        angle_limits = self.angle_limits[ind_sort]
        radar_obs = np.zeros([self.nb_radar])
        for i in range(self.nb_radar - 1):
            if theta < angle_limits[i+1] and theta > angle_limits[i]:
                radar_obs[ind_sort[i]] = 1
                break
        if radar_obs.sum() == 0:
            radar_obs[ind_sort[self.nb_radar - 1]] = 1

        self.obs = np.concatenate([self.agent_pos, radar_obs, self.rangefinder])


    def render(self, mode='human', close=False):

        try:
            for p in self.renderer.patches:
                p.remove()
            for p in self.renderer.patches:
                p.remove()
            for p in self.renderer.collections:
                p.remove()
        except:
            pass

        if self.renderer is None:
            fig, self.renderer = plt.subplots(1)
            self.window = self.renderer.imshow(self.descriptor)

        self.window.set_data(self.descriptor)
        goal_circle = Circle([self.goal[1], self.goal[0]], self.agent_size // 2, color=[204/255, 0, 0])
        self.renderer.add_patch(goal_circle)
        agent_circle = Circle([self.agent_pos[1], self.agent_pos[0]], self.agent_size // 2, color=[0, 76/255, 153/255])
        self.renderer.add_patch(agent_circle)

        if self.show_range_finder:
            lc = mc.LineCollection(self.lines, colors='r', linewidths=1)
            self.renderer.add_collection(lc)
        plt.pause(0.05)
        plt.draw()

    def close(self):
        if self.renderer is not None:
            plt.close()

    @property
    def dim_goal(self):
        return 2