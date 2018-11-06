import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import collections as mc
from gym import spaces


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class Rooms1():
    # def __init__(self, size_agent=5, size_grid=200, size_room=100, button_pos=(180,20), max_step_size=10, max_door_step=10, reward_type='sparse', distance_threshold=10):

    def __init__(self, size_agent=5, size_grid=100, size_room=50, button_pos=(90,10), max_step_size=10, max_door_step=10, reward_type='sparse', distance_threshold=10):
        """Initializes
        """

        self.size_agent = size_agent
        self.size_grid = size_grid
        self.size_room = size_room
        self.max_step_size = max_step_size
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.max_door_step = max_door_step
        self.button_pos = np.array(button_pos)
        self.n_timesteps = 50
        self.t = 0

        self.agent_pos_init = np.array([4 * self.size_agent, self.size_grid - 4 * self.size_agent])
        self.door_init = 10
        self.in_room = False

        self.renderer = None


        self.tasks = [0, 1, 2]
        self.n_tasks = len(self.tasks)
        # indices of relevant object position (achieved_goal)
        # the achieved goal for the stacking task (T3) contains the gripper coordinate, as it is necessary to compute the reward (has to be far from the goal)
        self.tasks_obs_id = [[0, 1], [4], [0, 1, 5]]

        dim_tasks_g = [2, 1, 2]
        ind_g = 0
        ind_ag = 0
        self.tasks_g_id = [] # indices of goal referring to the different tasks
        self.tasks_ag_id = [] # indices of achieved_goal referring to the different tasks
        for i in range(self.n_tasks):
            self.tasks_ag_id.append(list(range(ind_ag, ind_ag + len(self.tasks_obs_id[i]))))
            ind_ag += len(self.tasks_obs_id[i])
            self.tasks_g_id.append(list(range(ind_g, ind_g + dim_tasks_g[i])))
            ind_g += dim_tasks_g[i]

        self.flat = False # flat architecture ?
        self.dim_ag = sum([len(self.tasks_ag_id[i]) for i in range(self.n_tasks)])
        self.dim_g = sum(dim_tasks_g)
        self.goal = np.zeros([self.dim_g])
        self.mask = np.zeros([self.n_tasks])
        self.goal_to_render = np.zeros([self.n_tasks])
        self.task = 0
        self.agent_pos = self.agent_pos_init.copy()
        self.door = self.door_init
        self.grid_min = np.array([0, -size_room])
        self.grid_max = np.array([size_grid, size_grid])

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
        done = False
        self.agent_pos = self.agent_pos_init.copy()
        self.door = self.door_init
        self.in_room = False
        obs = self._get_obs()

        return obs

    def compute_reward(self, achieved_goal, goal, info=None):

        if goal.ndim == 1:
            goal = goal.reshape([1, goal.size])
            achieved_goal = achieved_goal.reshape([1, achieved_goal.size])

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
                ind = np.argwhere(goal[i_g] != 0).squeeze().tolist()
                if type(ind) == int:
                    ind = [ind]
                good_task = []
                for i_t in range(self.n_tasks):
                    if ind == self.tasks_g_id[i_t]:
                        good_task.append(i_t)
                assert len(good_task) == 1
                task = good_task[0]

                if task in [0, 1]:
                    if task == 0:
                        ag = (self.grid_max - self.grid_min) * (achieved_goal[i_g, self.tasks_ag_id[task]] + 1) / 2 + self.grid_min
                        g = (self.grid_max - self.grid_min) * (goal[i_g, self.tasks_g_id[task]] + 1) / 2 + self.grid_min
                    else:
                        ag = self.size_room * (achieved_goal[i_g, self.tasks_ag_id[task]] + 1) / 2 + 10
                        g = self.size_room * (goal[i_g, self.tasks_ag_id[task]] + 1) / 2 + 10
                    # Compute distance between goal and the achieved goal.
                    d = goal_distance(ag, g)
                    if self.reward_type == 'sparse':
                        r[i_g] = -(d > (self.distance_threshold // 2)).astype(np.float32)
                    else:
                        r[i_g] = -d

                elif task == 2:
                    in_room = achieved_goal[i_g, self.tasks_ag_id[task][2]]
                    if in_room == 1:
                        ag = (self.grid_max - self.grid_min) * (achieved_goal[i_g, self.tasks_ag_id[task][:2]] + 1) / 2 + self.grid_min
                        g = (self.grid_max - self.grid_min) * (goal[i_g, self.tasks_g_id[task]] + 1) / 2 + self.grid_min
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

        agent_pos = 2 * (self.agent_pos.copy() - self.grid_min) / (self.grid_max - self.grid_min) - 1
        button_pos = 2 * (self.button_pos.copy() - self.grid_min) / (self.grid_max - self.grid_min) - 1
        door = 2 * (self.door - 10) / self.size_room - 1
        in_room = 1 if self.in_room else -1

        obs = np.concatenate([agent_pos, button_pos, np.array([door]), np.array([in_room])])
        achieved_goal = self._compute_achieved_goal(obs)

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
            'mask': self.mask
        }

    def step(self, action):
        assert np.all(action<=1)
        assert np.all(action>=-1)

        dx = action[0] * self.max_step_size
        dy = action[1] * self.max_step_size
        door_act = action[2] * self.max_door_step

        # check collisions
        agent_pos = self.agent_pos.copy() + np.array([dx, dy])
        if not self.in_room:
            # check bottom
            if agent_pos[1] + self.size_agent > self.size_grid:
                agent_pos[1] = self.size_grid - self.size_agent
            # check left
            if agent_pos[0] - self.size_agent < 0:
                agent_pos[0] = self.size_agent
            # check right
            if agent_pos[0] + self.size_agent > self.size_grid:
                agent_pos[0] = self.size_grid - self.size_agent
            # check top before room (left) and after beginning of door
            if agent_pos[1] - self.size_agent < 0 and (agent_pos[0] - self.size_agent < self.size_grid - self.size_room or \
                                    agent_pos[0] + self.size_agent > self.size_grid - self.size_room + self.door - 10):
                agent_pos[1] = self.size_agent
            if agent_pos[1] - self.size_agent < 0:
                self.in_room = True

        if self.in_room:
            if self.agent_pos[1] + self.size_agent > 0:
                # check left
                if agent_pos[0] - self.size_agent < self.size_grid - self.size_room:
                    agent_pos[0] = self.size_grid - self.size_agent
                # check right
                if agent_pos[0] + self.size_agent > self.size_grid - self.size_room + self.door - 10:
                    agent_pos[0] = self.size_grid - self.size_room + self.door - 10 - self.size_agent
            else:
                # check left
                if agent_pos[0] - self.size_agent < self.size_grid - self.size_room:
                    agent_pos[0] = self.size_grid - self.size_room + self.size_agent
                # check right
                if agent_pos[0] + self.size_agent > self.size_grid:
                    agent_pos[0] = self.size_grid - self.size_agent
                if agent_pos[1] - self.size_agent < - self.size_room:
                    agent_pos[1] = -self.size_room + self.size_agent
                if agent_pos[1] + self.size_agent > 0 and agent_pos[0] + self.size_agent > self.size_grid - self.size_room + self.door - 10:
                    agent_pos[1] = - self.size_agent
                if agent_pos[1] + self.size_agent > 0:
                    self.in_room = False

        self.agent_pos = agent_pos.copy()

        # door act
        if np.linalg.norm(self.agent_pos - self.button_pos) < self.distance_threshold:
            self.door = np.clip(self.door + door_act, 10, self.size_room + 10)

        obs = self._get_obs()
        reward = self.compute_reward(obs['achieved_goal'], self.goal)
        info = dict(is_success= reward == 0)
        if self.t == self.n_timesteps:
            done = True
        else:
            done = False

        return obs, reward, done, info

    def reset_task_goal(self, goal, task=None):
        self._set_task(task)
        self.goal, self.mask, self.goal_to_render = self._compute_goal(goal, task)
        obs = self._get_obs()
        return obs

    def _set_task(self, t):
        if not self.flat:
            self.task = t

    def set_flat_env(self):
        self.flat = True

    def _compute_goal(self, full_goal, task):
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
                tmp_goal = (goal + 1) * (self.size_grid // 2 - 2 * self.size_agent)
                desired_goal[self.tasks_g_id[t]] = 2 * (tmp_goal.copy() - self.grid_min) / (self.grid_max - self.grid_min) - 1  # normalize to room size
                goal_to_render = tmp_goal.copy()

            elif t == 1:  # coordinates 1D for the door
                tmp_goal = ((goal[0] + 1) / 2) * self.size_room + 10
                desired_goal[self.tasks_g_id[t]] = 2 * (tmp_goal.copy() - 10) / self.size_room - 1
                goal_to_render = tmp_goal.copy()

            elif t == 2:  # coordinate in the small room
                tmp_goal = np.zeros([2])
                tmp_goal[0] = ((goal[0] + 1) / 2) * (self.size_room - 2 * self.size_agent) + self.size_grid - self.size_room
                tmp_goal[1] = ((goal[1] + 1) / 2) * (self.size_room - 2 * self.size_agent) - self.size_room
                desired_goal[self.tasks_g_id[t]] = 2 * (tmp_goal.copy() - self.grid_min) / (self.grid_max - self.grid_min) - 1  # normalize to room size
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

    def render(self, mode='human', close=False):

        try:
            for p in self.renderer.patches:
                p.remove()
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
        plt.gca().invert_yaxis()
        plt.axis('equal')
        plt.xlim([0, self.size_grid])
        plt.ylim([- self.size_room, self.size_grid])
        lines = [[(0, self.size_grid), (self.size_grid, self.size_grid)], [(0, 0), (0, self.size_grid)],
                 [(self.size_grid, - self.size_room), (self.size_grid, self.size_grid)], [(0, 0), (self.size_grid - self.size_room, 0)],
                 [(self.size_grid - self.size_room + self.door - 10, 0), (self.size_grid, 0)], [(self.size_grid - self.size_room, -self.size_room), (self.size_grid,
                                                                                                                                                   -self.size_room)],
                 [(self.size_grid - self.size_room, -self.size_room), (self.size_grid - self.size_room, 0)]]
        lc = mc.LineCollection(lines, colors='k', linewidths=2)
        self.renderer.add_collection(lc)

        if self.task in [0, 2]:
            goal_circle = Circle(self.goal_to_render, self.size_agent, color='k')
        else:
            goal_circle = Circle(np.array([self.size_grid - self.size_room + self.goal_to_render - 10, 0]), self.size_agent, color='k')
        self.renderer.add_patch(goal_circle)

        button_circle = Circle(self.button_pos, self.size_agent, color=[204 / 255, 0, 0])
        self.renderer.add_patch(button_circle)
        agent_circle = Circle(self.agent_pos, self.size_agent, color=[0, 76 / 255, 153 / 255])
        self.renderer.add_patch(agent_circle)



        plt.pause(0.05)
        plt.draw()

    def close(self):
        if self.renderer is not None:
            plt.close()

    @property
    def nb_tasks(self):
        return self.n_tasks