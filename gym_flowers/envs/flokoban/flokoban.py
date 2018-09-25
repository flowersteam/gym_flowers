import itertools
import random
import numpy as np
import gym
from gym import spaces
import matplotlib.cm as cm
from matplotlib import pyplot as plt


class SquareDistractor(gym.Env):
    """The Sokoban environment.
    """

    def __init__(self, grid_size=10):

        self.grid_size = grid_size

        self._noop = np.array([0, 0])
        self._move_down = np.array([1, 0])
        self._move_up = np.array([-1, 0])
        self._move_right = np.array([0, 1])
        self._move_left = np.array([0, -1])
        self._distract = np.array([1, 1])
        self._moves = [self._noop, self._move_down, self._move_up, self._move_right, self._move_left, self._distract]

        # We set the space
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Discrete(grid_size * grid_size)

        self.reward = None
        self.observation = None
        self.goal = None

        self.viewer = None

    def env_seed(self, seed):
        # TODO: fix it
        np.random.seed(seed)

    def reset(self):

        self.agent_pos = np.random.randint(0, self.grid_size, size=(2,))

        # create a list of all available cells in the grid
        cells = list(itertools.product(range(self.grid_size), range(self.grid_size)))
        available_cells = list(itertools.product(range(self.grid_size), range(self.grid_size)))

        available_cells.remove(tuple(self.agent_pos))

        self.goal = random.sample(cells, 1)[0]
        self.goal = np.array(self.goal)

        self._compute_grid()
        self._compute_observation()

        # We compute the initial reward.
        self.compute_reward()
        if self.reward == self._object_count:
            self.done = True
        else:
            self.done = False

        info = {}

        return self.observation, self.reward, self.done, info

    def compute_reward(self):
        self.reward = 0
        for obj in self._objects:
            for goal in self._goals:
                if (obj == goal).all():
                    self.reward += 1
        return self.reward

    def step(self, action=np.array([0])):
        """Perform an agent action in the Environment
        """

        assert action.shape == (1,)

        # We compute the new positions
        move = self._moves[action[0]]
        if self._in_grid(self.agent_pos + move):
            occupied, object_pose = self._is_occupied(self.agent_pos + move)
            if occupied and self._in_grid(object_pose + move):
                blocked, _ = self._is_occupied(object_pose + move)
                if not blocked:
                    object_pose += move
                    self.agent_pos += move
            if not occupied:
                self.agent_pos += move

        self.compute_reward()
        if self._reward == self._object_count:
            self._done = True

        info = {}

        return self.grid, self.reward, self._done, info

    def _is_occupied(self, pos):
        for obj in self._objects:
            if (pos == obj).all():
                return True, obj
        return False, False

    def _in_grid(self, pose):
        return (pose >= 0).all() and (pose < self.grid_size).all()

    def _compute_grid(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.grid[self.agent_pos[0], self.agent_pos[1]] = 1

        self.grid[self.goal[0], self.goal[1]] = 3

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

        if mode == 'rgb_array':
            return self.grid  # return RGB frame suitable for video
        elif mode is 'human':
            if self.viewer is None:
                self._start_viewer()
                # Retrieve image of corresponding to observation
                img = self.grid
                self._img_artist = self._ax.imshow(img, cmap=cm.jet, interpolation='nearest')
            else:
                # Retrieve image of corresponding to observation
                img = self.grid
                self._img_artist.set_data(img)
                plt.draw()
                plt.pause(0.05)

    def _start_viewer(self):
        plt.ion()
        self.viewer = plt.figure()
        self._ax = self.viewer.add_subplot(111)


class PushingObjects(gym.Env):
    """The Sokoban environment.
    """

    def __init__(self, *args, grid_size=10, object_count=2, **kwargs):

        self._grid_size = grid_size
        self._object_count = object_count

        self._noop = np.array([0, 0])
        self._move_down = np.array([1, 0])
        self._move_up = np.array([-1, 0])
        self._move_right = np.array([0, 1])
        self._move_left = np.array([0, -1])
        self._distract = np.array([1, 1])
        self._moves = [self._noop, self._move_down, self._move_up, self._move_right, self._move_left, self._distract]

        # We set the space
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Discrete(5)

        self._action_space = np.array([[0, 4]])

        self.reward = None
        self.observation = None
        self._objects = None
        self._goals = None
        self._grid = None
        self.agent_pos = None

        self.viewer = None

    def env_seed(self, seed):
        # TODO: fix it
        np.random.seed(seed)

    def reset(self):

        self.agent_pos = np.random.randint(0, self._grid_size, size=(2,))

        # create a list of all available cells in the grid
        cells = list(itertools.product(range(self._grid_size), range(self._grid_size)))
        available_cells = list(itertools.product(range(self._grid_size), range(self._grid_size)))

        available_cells.remove(tuple(self.agent_pos))

        self._objects = []
        self._goals = []
        for obj_idx in range(self._object_count):
            obj_pos = random.sample(available_cells, 1)[0]
            available_cells.remove(obj_pos)
            self._objects.append(obj_pos)
        self._objects = np.array(self._objects)
        for goal_idx in range(self._object_count):
            goal = random.sample(cells, 1)[0]
            self._goals.append(goal)
        self._goals = np.array(self._goals)

        obs = np.array([self.agent_pos, self.goal])

        # We compute the initial reward.
        self.compute_reward()
        if self.reward == self._object_count:
            self._done = True
        else:
            self._done = False

        info = {}

        return obs, self.reward, self._done, info

    def compute_reward(self):
        self.reward = 0
        for obj in self._objects:
            for goal in self._goals:
                if (obj == goal).all():
                    self.reward += 1
        return self.reward

    def step(self, action=np.array([0])):
        """Perform an agent action in the Environment
        """

        assert action.shape == (1,)

        # We compute the new positions
        move = self._moves[action[0]]
        if self._in_grid(self.agent_pos + move):
            occupied, object_pose = self._is_occupied(self.agent_pos + move)
            if occupied and self._in_grid(object_pose + move):
                blocked, _ = self._is_occupied(object_pose + move)
                if not blocked:
                    object_pose += move
                    self.agent_pos += move
            if not occupied:
                self.agent_pos += move

        self.compute_reward()
        if self._reward == self._object_count:
            self._done = True

        info = {}

        return self._grid, self.reward, self._done, info

    def _is_occupied(self, pos):
        for obj in self._objects:
            if (pos == obj).all():
                return True, obj
        return False, False

    def _in_grid(self, pose):
        return (pose >= 0).all() and (pose < self._grid_size).all()

    def compute_grid(self):
        self._grid = np.zeros((self._grid_size, self._grid_size), dtype=np.uint8)
        self._grid[self.agent_pos[0], self.agent_pos[1]] = 1

        for obj_pos in self._objects:
            self._grid[obj_pos[0], obj_pos[1]] = 2
        for goal_pos in self._goals:
            self._grid[goal_pos[0], goal_pos[1]] = 3

    def compute_observation(self):
        obs_agent = np.zeros((self._grid_size, self._grid_size), dtype=np.float32)
        obs_objects = np.zeros((self._grid_size, self._grid_size), dtype=np.float32)
        obs_goals = np.zeros((self._grid_size, self._grid_size), dtype=np.float32)

        obs_agent[self.agent_pos[0], self.agent_pos[1]] = 1
        for obj_pos in self._objects:
            obs_objects[obj_pos[0], obj_pos[1]] = 1
        for goal_pos in self._goals:
            obs_goals[goal_pos[0], goal_pos[1]] = 1
        self._observation = np.array([obs_agent, obs_objects, obs_goals])

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

        if mode == 'rgb_array':
            return self._grid  # return RGB frame suitable for video
        elif mode is 'human':
            if self.viewer is None:
                self._start_viewer()
                # Retrieve image of corresponding to observation
                img = self._grid
                self._img_artist = self._ax.imshow(img, cmap=cm.jet, interpolation='nearest')
            else:
                # Retrieve image of corresponding to observation
                img = self._grid
                self._img_artist.set_data(img)
                plt.draw()
                plt.pause(0.05)


    def _start_viewer(self):
        plt.ion()
        self.viewer = plt.figure()
        self._ax = self.viewer.add_subplot(111)
        self._observation = None


if __name__ == '__main__':
    env = SquareDistractor()
    env.env_seed(0)
    env.reset()
    env._compute_grid()
    print(env.grid)

    actor = PushingObjects(object_count=2)
    actor.env_seed(0)
    actor.reset()
    actor.compute_grid()
    print(actor._grid)
    actor.compute_reward()
    actor.agent_pos = np.array([7, 0])
    actor._objects[0] = np.array([7, 1])
    actor._objects[1] = np.array([7, 2])
    actor._goals[1] = np.array([7, 2])
    actor.compute_grid()
    print(actor._grid)
    actor.render()
    for _ in range(50):
        actor.step(np.random.randint(5, size=1))
        actor.compute_grid()
        actor.render()
    actor.compute_grid()
    print(actor._grid)
    actor.compute_reward()
    print(actor.reward)
    actor.compute_observation()
    actor.step(np.array([0]))
    print(actor.observation)
