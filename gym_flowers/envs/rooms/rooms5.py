#! /usr/bin/env python

import os
import random
import pygame

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import collections as mc
from gym import spaces


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


 # Class for the orange dude
class Player(object):
    def __init__(self, scale):
        self.rect = pygame.Rect(10 * scale, 10 * scale, scale, scale)

    def move(self, dx, dy, walls):

        # Move each axis separately. Note that this checks for collisions both times.
        if dx != 0:
            self.move_single_axis(dx, 0, walls)
        if dy != 0:
            self.move_single_axis(0, dy, walls)

    def move_single_axis(self, dx, dy, walls):

        # Move the rect
        self.rect.x += round(dx)
        self.rect.y += round(dy)

        # If you collide with a wall, move out based on velocity
        for wall in walls:
            if self.rect.colliderect(wall.rect):
                if dx > 0:  # Moving right; Hit the left side of the wall
                    self.rect.right = wall.rect.left
                if dx < 0:  # Moving left; Hit the right side of the wall
                    self.rect.left = wall.rect.right
                if dy > 0:  # Moving down; Hit the top side of the wall
                    self.rect.bottom = wall.rect.top
                if dy < 0:  # Moving up; Hit the bottom side of the wall
                    self.rect.top = wall.rect.bottom

# Nice class to hold a wall rect
class Wall(object):
    def __init__(self, pos, walls, scale):
        walls.append(self)
        self.rect = pygame.Rect(pos[0], pos[1], scale, scale)

class Rooms5():
    def __init__(self, reward_type='sparse', nb_rooms=5, distance_threshold=16, debug=False):

        self.debug = debug
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.in_rooms = [False] * nb_rooms
        self.nb_rooms = nb_rooms

        self.scale = 16

        # Initialise pygame
        os.environ["SDL_VIDEO_CENTERED"] = "1"
        pygame.init()

        self.renderer = None

        self.walls = []
        self.player = Player(self.scale)  # Create the player
        self.button_rect = []
        self.button_pos = []
        self.button_ind = [4, 0, 2, 1, 3]
        # Holds the level layout in a list of strings

        # init_level = [
        #     "1111111111111111111111111",
        #     "1000000000000000100000001",
        #     "1000000000000000100000001",
        #     "1000000000000000100000001",
        #     "1000000000000000100000001",
        #     "1000000000000000100000001",
        #     "1000000000000000100000001",
        #     "1000000000000000100000001",
        #     "1000000000000000111111111",
        #     "1000000000000000100000001",
        #     "1000000000000000100000201",
        #     "1000000000000000100000001",
        #     "1000000000000000100000001",
        #     "1000000000000000100000001",
        #     "1020000000000000100000001",
        #     "1000000000000000100000001",
        #     "1111111111111111111111111",
        #     "1000000010000000100000001",
        #     "1000000010000020100000001",
        #     "1000000010000000100000001",
        #     "1000000010000000100000001",
        #     "1000000010000000100000001",
        #     "1000002010000000100000201",
        #     "1000000010000000100000001",
        #     "1111111111111111111111111"]
        init_level = [
            "1111111111111111111",
            "1000000000001000001",
            "1000000000001000001",
            "1000000000001000001",
            "1000000000001000001",
            "1000000000001000001",
            "1000000000001111111",
            "1000000000001000001",
            "1000000000001000201",
            "1000000000001000001",
            "1020000000001000001",
            "1000000000001000001",
            "1111111111111111111",
            "1000001000001000001",
            "1000001000201000001",
            "1000001000001000001",
            "1000201000001000201",
            "1000001000001000001",
            "1111111111111111111"]
        self.init_level = np.zeros([len(init_level), len(init_level[0])])
        for i, row in enumerate(init_level):
            for j, col in enumerate(row):
                self.init_level[i, j] = int(col)
        self.level = self.init_level.copy()
        self.door_coordinates = [[12, 12, 1, 6], [13, 18, 6, 6], [13, 18, 12, 12], [12, 12, 13, 18], [6, 6, 13, 18]]
        # self.door_coordinates = [[16, 16, 1, 8], [17, 24, 8, 8], [17, 24, 16, 16], [16, 16, 17, 24], [8, 8, 17, 24]]
        self.door_orientation = ['h', 'v', 'v', 'h', 'h']
        # self.room_coordinates = [(17, 1), (17, 9), (17, 17), (9, 17), (1, 17)]
        self.room_coordinates = [(13, 1), (13, 7), (13, 13), (7, 13), (1, 13)]
        self.update_doors(self.init_level)

        self.n_timesteps = 500
        self.t = 0


        self.grid_size = 18 * self.scale #24
        self.size_big_room = 10 * self.scale # 14
        self.size_small_room = 5 * self.scale #7
        self.max_step_size = 10
        self.max_door_step = 6

        self.tasks = list(range(nb_rooms * 2 + 1))
        self.n_tasks = len(self.tasks)

        self.tasks_obs_id = [[0, 1]]
        dim_tasks_g = [2]
        for i in range(self.nb_rooms):
            self.tasks_obs_id.extend([[4 * (i + 1)], [0, 1, 4 * (i + 1) + 1]])
            dim_tasks_g.extend([1, 2])

        ind_g = 0
        ind_ag = 0
        self.tasks_g_id = []  # indices of goal referring to the different tasks
        self.tasks_ag_id = []  # indices of achieved_goal referring to the different tasks
        for i in range(self.n_tasks):
            self.tasks_ag_id.append(list(range(ind_ag, ind_ag + len(self.tasks_obs_id[i]))))
            ind_ag += len(self.tasks_obs_id[i])
            self.tasks_g_id.append(list(range(ind_g, ind_g + dim_tasks_g[i])))
            ind_g += dim_tasks_g[i]

        self.flat = False  # flat architecture ?
        self.dim_ag = sum([len(self.tasks_ag_id[i]) for i in range(self.n_tasks)])
        self.dim_g = sum(dim_tasks_g)
        self.goal = np.zeros([self.dim_g])
        self.mask = np.zeros([self.n_tasks])
        self.task = 0
        self.agent_pos = np.array([self.player.rect.centerx, self.player.rect.centery])
        self.doors = np.zeros([self.nb_rooms])

        self.action_space = spaces.Box(-1., 1., shape=(3,), dtype='float32')
        obs = self._get_obs()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['desired_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))
        self.reward_range = range(-1, 0)
        self.metadata = {}

        self.info = dict(is_success=0)
        self.spec = None

    def update_doors(self, level):
        # Parse the level string above. W = wall, E = exit
        self.button_rect = [None] * 5
        self.button_pos = [None] * 5
        self.walls = []
        x = y = 0
        ind = 0
        for row in level:
            for col in row:
                if col == 1:
                    Wall((x, y), self.walls, self.scale)
                if col == 2:
                    rec = pygame.Rect(x, y, self.scale, self.scale)
                    self.button_rect[self.button_ind[ind]] = rec
                    self.button_pos[self.button_ind[ind]] = np.array((rec.centerx, rec.centery))
                    ind += 1
                x += self.scale
            y += self.scale
            x = 0


    # GoalEnv methods
    # ----------------------------
    @property
    def unwrapped(self):
        return self

    def seed(self, seed):
        np.random.seed(seed)
        return seed

    def reset(self):
        self.t = 0
        self.update_doors(self.init_level)
        self.level = self.init_level.copy()
        self.player = Player(self.scale)
        self.doors = np.zeros([self.nb_rooms])
        self.in_rooms = [False] * self.nb_rooms
        obs = self._get_obs()

        return obs

    def compute_reward(self, achieved_goal, goal, task_descr, info=None):

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
                if self.reward_type == 'sparse':
                    r[i_g] = -(d > self.distance_threshold).astype(np.float32)
                else:
                    r[i_g] = - d
        else:
            r = np.zeros([goal.shape[0]])
            for i_g in range(goal.shape[0]):
                # find current task
                task = np.argwhere(task_descr[i_g] == 1)[0][0]

                if task == 0:
                    ag = self.grid_size * (achieved_goal[i_g, self.tasks_ag_id[task]] + 1) / 2
                    g = self.grid_size * (goal[i_g, self.tasks_g_id[task]] + 1) / 2
                    # Compute distance between goal and the achieved goal.
                    g += self.scale // 2 #get center coordinates of g
                    d = goal_distance(ag, g)
                    if self.reward_type == 'sparse':
                        r[i_g] = -(d > (self.distance_threshold)).astype(np.float32)
                    else:
                        r[i_g] = -d

                elif task in [1, 3, 5, 7, 9]:
                    ag = self.size_small_room * (achieved_goal[i_g, self.tasks_ag_id[task]] + 1) / 2
                    g = self.size_small_room * (goal[i_g, self.tasks_g_id[task]] + 1) / 2
                    # Compute distance between goal and the achieved goal.
                    d = goal_distance(ag, g)
                    if self.reward_type == 'sparse':
                        r[i_g] = -(d > (self.distance_threshold//2)).astype(np.float32)
                    else:
                        r[i_g] = -d

                elif task in [2, 4, 6, 8, 10]:
                    in_room = achieved_goal[i_g, self.tasks_ag_id[task][2]]
                    if in_room == 1:
                        ag = self.grid_size * (achieved_goal[i_g, self.tasks_ag_id[task][:2]] + 1) / 2
                        g = self.grid_size * (goal[i_g, self.tasks_g_id[task]] + 1) / 2
                        g += self.scale // 2
                        d = goal_distance(ag, g)

                        if self.reward_type == 'sparse':
                            r[i_g] = -(d > self.distance_threshold).astype(np.float32)
                        else:
                            r[i_g] = -d
                    else:
                        r[i_g] = - 1
                else:
                    raise NotImplementedError

        return r.reshape([r.size, 1])

    def _get_obs(self):

        agent_pos = 2 * self.agent_pos.copy() / self.grid_size - 1
        self.in_rooms = np.zeros([5])
        if self.agent_pos[1] >= 12 * self.scale: #16
            if self.agent_pos[0] < 6 * self.scale: #8
                self.in_rooms[0] = 1
                # print('room0')
            elif self.agent_pos[0] < 12 * self.scale:#16
                self.in_rooms[1] = 1
                # print('room1')
            else:
                self.in_rooms[2] = 1
                # print('room2')
        elif self.agent_pos[0] > 13 * self.scale: #17
            if self.agent_pos[1] < 6 * self.scale:#8
                self.in_rooms[4] = 1
                # print('room4')
            else:
                self.in_rooms[3] = 1
                # print('room3')



        obs = agent_pos
        for i in range(self.nb_rooms):
            button_pos = 2 * self.button_pos[i].copy() / self.grid_size - 1
            door = np.array([2 * self.doors[i] / self.size_small_room - 1])
            in_room = np.array([1 if self.in_rooms[i] else -1])
            obs = np.concatenate([obs, button_pos, door, in_room])
        # print(obs)

        achieved_goal = self._compute_achieved_goal(obs)

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
            'mask': self.mask
        }

    def step(self, action):
        assert np.all(action <= 1)
        assert np.all(action >= -1)

        if not self.debug:
            dx = action[0] * self.max_step_size
            dy = action[1] * self.max_step_size
            door_act = action[2] * self.max_door_step
            self.player.move(dx, dy, self.walls)

        else:
            # Move the player if an arrow key is pressed
            key_pressed = False
            while not key_pressed:
                pygame.event.get()
                key = pygame.key.get_pressed()
                if key[pygame.K_LEFT]:
                    self.player.move(-self.max_step_size, 0, self.walls)
                    key_pressed = True
                if key[pygame.K_RIGHT]:
                    self.player.move(self.max_step_size, 0, self.walls)
                    key_pressed = True
                if key[pygame.K_UP]:
                    self.player.move(0, -self.max_step_size, self.walls)
                    key_pressed = True
                if key[pygame.K_DOWN]:
                    self.player.move(0, self.max_step_size, self.walls)
                    key_pressed = True
                if key[pygame.K_SPACE]:
                    door_act = 1 * self.max_door_step
                    key_pressed = True
                elif key[pygame.K_BACKSPACE]:
                    door_act = -1 * self.max_door_step
                    key_pressed = True
                else:
                    door_act = 0


        for i in range(self.nb_rooms):
            if self.player.rect.colliderect(self.button_rect[i]):
                self.doors[i] = np.clip(self.doors[i] + door_act, 0, self.size_small_room)
                nb_rect = int(self.doors[i] // self.scale)
                new_wall = np.concatenate([np.zeros([nb_rect]), np.ones([self.size_small_room // self.scale - nb_rect])])
                ind = 0
                if self.door_orientation[i] == 'h':
                    for x in range(self.door_coordinates[i][2], self.door_coordinates[i][3]):
                        self.level[self.door_coordinates[i][0]][x] = new_wall[ind]
                        ind += 1
                else:
                    for y in range(self.door_coordinates[i][0], self.door_coordinates[i][1]):
                        self.level[y][self.door_coordinates[i][2]] = new_wall[ind]
                        ind += 1

        self.agent_pos = np.array([self.player.rect.centerx, self.player.rect.centery])

        self.update_doors(self.level)

        obs = self._get_obs()
        reward = self.compute_reward(obs['achieved_goal'], self.goal, obs['mask'])
        info = dict(is_success=reward == 0)
        if self.t == self.n_timesteps:
            done = True
        else:
            done = False

        return obs, reward, done, info

    def reset_task_goal(self, goal, task=None, directly=None, eval=None):
        self._set_task(task)
        self.goal, self.mask, self.goal_to_render = self._compute_goal(goal, task)
        obs = self._get_obs()
        return obs

    def _set_task(self, t):
        if not self.flat:
            self.task = t

    def set_flat_env(self):
        self.flat = True

    def _compute_goal(self, full_goal, task, eval=None):
        if self.flat:
            task = self.tasks
        else:
            task = [task]
        desired_goal = np.zeros([self.dim_g])
        for t in task:
            if self.flat:
                goal = full_goal[self.tasks_g_id[t]]
            else:
                goal = full_goal
            if t == 0:  # coordinates in the big room
                tmp_goal = np.array([self.scale, self.scale]) + (goal + 1) / 2 * self.size_big_room
                desired_goal[self.tasks_g_id[t]] = 2 * (tmp_goal.copy() ) / self.grid_size - 1  # normalize to grid_size
                goal_to_render = pygame.Rect(tmp_goal[0], tmp_goal[1], self.scale, self.scale)

            elif t in [1, 3, 5, 7, 9]:  # coordinates 1D for the door
                i = t//2
                tmp_goal = self.scale * np.array([self.door_coordinates[i][2], self.door_coordinates[i][0]])
                if self.door_orientation[i] == 'h':
                    tmp_goal += np.array([int((goal[0] + 1) / 2 * self.size_small_room), 0])
                else:
                    tmp_goal += np.array([0, int((goal[0] + 1) / 2 * self.size_small_room)])
                desired_goal[self.tasks_g_id[t]] = goal[0]
                goal_to_render = pygame.Rect(tmp_goal[0], tmp_goal[1], self.scale, self.scale)

            elif t in [2, 4, 6, 8, 10]:  # coordinate in the small room 1
                i = (t - 1) // 2
                tmp_goal = self.scale * np.array([self.room_coordinates[i][1], self.room_coordinates[i][0]]) + (goal + 1) / 2 * (self.size_small_room - self.scale)
                desired_goal[self.tasks_g_id[t]] = 2 * (tmp_goal.copy() ) / self.grid_size - 1  # normalize to grid_size
                goal_to_render = pygame.Rect(tmp_goal[0], tmp_goal[1], self.scale, self.scale)

        mask = np.zeros([self.n_tasks])
        if not self.flat:
            mask[task[0]] = 1
        return desired_goal, mask, goal_to_render

    def _compute_achieved_goal(self, obs):

        achieved_goal = np.zeros([self.dim_ag])
        for i_t in range(self.n_tasks):
            achieved_goal[self.tasks_ag_id[i_t]] = obs[self.tasks_obs_id[i_t]]
        return achieved_goal

    def render(self, mode='human', close=False):

        if not self.renderer:
            # Set up the display
            # self.screen = pygame.display.set_mode((497, 497))
            self.screen = pygame.display.set_mode((300, 300))

            self.clock = pygame.time.Clock()

        # Draw the scene
        self.screen.fill((0, 0, 0))
        for wall in self.walls:
            pygame.draw.rect(self.screen, (255, 255, 255), wall.rect)
        for i in range(self.nb_rooms):
            pygame.draw.rect(self.screen, (255, 0, 0), self.button_rect[i])
        pygame.draw.rect(self.screen, (0, 0, 255), self.goal_to_render)
        pygame.draw.rect(self.screen, (255, 200, 0), self.player.rect)

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pass

    @property
    def nb_tasks(self):
        return self.n_tasks


