import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import random
import gizeh


class ArmBall(gym.GoalEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, object_initial_pos=np.array([0.6, 0.6]),
                 arm_lengths=np.array([0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]), object_size=0.1, n_timesteps=50,
                 epsilon=0.05, action_scaling=10, reward_type='sparse', one_goal=False, env_noise=0):
        """Initializes a new ArmBall environment.

                Args:
                    object_initial_pos (np_array): initial pose for the ball
                    arm_lengths (np_array): lengths of the robotic arm between each joint
                    object_size (float): ball size, maximum distance to catch the ball
                    n_timesteps (int): maximum number of timesteps in the environment before reset
                    epsilon (float): the threshold after which a goal is considered achieved
                    action_scaling (float): the scaling for action (actions are between -1 and 1 before scaling)
                    one_goal (np_array): if True then the goal is always the same (useful for debug and test)
                    env_noise (float): amount of gaussian noise for rendering
                    reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """

        assert arm_lengths.size < 8, "The number of joints must be inferior to 8"
        assert arm_lengths.sum() == 1., "The arm length must sum to 1."

        # We set the parameters
        self._n_joints = arm_lengths.size
        self._arm_lengths = arm_lengths
        self._object_initial_pos = object_initial_pos
        self._object_size = object_size
        self.achieved_goal = self._object_initial_pos
        self._arm_pos = np.zeros(self._arm_lengths.size)
        self._hand_pos = np.zeros(2)
        self._object_handled = False
        self.desired_goal = None  # goal position
        self._n_timesteps = n_timesteps
        self.reward_type = reward_type
        self._epsilon = epsilon  # precision for sparse reward
        self._one_goal = one_goal
        self._action_scaling = action_scaling

        if self.reward_type == 'sparse':
            self.reward_range = (-1, 0)
        else:
            self.reward_range = (-2, 0)

        # We set the space
        self.action_space = spaces.Box(low=-np.ones(self._n_joints),
                                       high=np.ones(self._n_joints),
                                       dtype=np.float32)
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(low=-np.ones(2), high=np.ones(2), dtype=np.float32),  # position of ball
            achieved_goal=spaces.Box(low=-np.ones(2), high=np.ones(2), dtype=np.float32),
            observation=spaces.Box(low=-np.ones(self._n_joints + 2),  # joints + position of ball
                                   high=np.ones(self._n_joints + 2),
                                   dtype=np.float32),
        ))

        self._env_noise = env_noise
        self._width = 500
        self._height = 500
        self._rendering = np.zeros([self._height, self._width, 3])
        self._rendering[0] = 1
        self._render_arm = True
        self._render_goal = True
        self._render_obj = True
        self._render_hand = True
        self._rgb = True

        self.viewer = None

        # We set to None to rush error if reset not called
        self.reward = None
        self._observation = None
        self._steps = None
        self._done = None

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        return seed

    def _sample_goal(self):
        goal = np.random.uniform(-1, 1, 2)
        return goal

    def compute_reward(self, achieved_goal, goal, info=None):
        if achieved_goal.ndim > 1:
            d = np.linalg.norm(achieved_goal - goal, ord=2, axis=1)
        else:
            d = np.linalg.norm(achieved_goal - goal, ord=2)
        if self.reward_type == 'sparse':
            return -(d > self._epsilon).astype(np.int)
        else:
            return -d

    def _is_success(self, achieved_goal, goal):
        if achieved_goal.ndim > 1:
            d = np.linalg.norm(achieved_goal - goal, ord=2, axis=1)
        else:
            d = np.linalg.norm(achieved_goal - goal, ord=2)
        return -(d > self._epsilon).astype(np.int)

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        """

        # We compute the position of the end effector
        self._arm_pos = np.clip(self._arm_pos + action / self._action_scaling,
                                a_min=-np.ones(self._n_joints),
                                a_max=np.ones(self._n_joints))
        angles = np.cumsum(self._arm_pos)
        angles_rads = np.pi * angles
        self._hand_pos = np.array([np.sum(np.cos(angles_rads) * self._arm_lengths),
                                   np.sum(np.sin(angles_rads) * self._arm_lengths)])

        # We check if the object is handled and we move it.
        if np.linalg.norm(self._hand_pos - self.achieved_goal, ord=2) < self._object_size:
            self._object_handled = True
        if self._object_handled:
            self.achieved_goal = self._hand_pos

        # We update observation and reward
        self._observation = np.concatenate([self._arm_pos, self.achieved_goal])
        self.reward = self.compute_reward(self.achieved_goal, self.desired_goal)
        self._steps += 1
        if self._steps == self._n_timesteps:
            self._done = True

        self.obs = dict(observation=self._observation, desired_goal=self.desired_goal, achieved_goal=self.achieved_goal)

        info = {}
        info['is_success'] = self._is_success(self.achieved_goal, self.desired_goal) == 0

        return self.obs, self.reward, self._done, info


    def reset(self, goal=None):
        # We reset the simulation
        if self._one_goal:
            self.desired_goal = np.array([0.3, 0.5])
        else:
            if goal is not None:
                self.desired_goal = goal
            else:
                self.desired_goal = self._sample_goal()
        self.achieved_goal = self._object_initial_pos
        self._arm_pos = np.zeros(self._arm_lengths.shape)
        self._object_handled = False
        self._observation = np.concatenate([self._arm_pos, self.achieved_goal])
        self._steps = 0
        self._done = False

        # We compute the initial reward.
        self.reward = self.compute_reward(self.achieved_goal, self.desired_goal)

        self.obs = dict(observation=self._observation, desired_goal=self.desired_goal, achieved_goal=self.achieved_goal)

        return self.obs

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

        # We retrieve arm and object pose
        arm_pos = self._arm_pos
        object_pos = self.achieved_goal

        # World parameters
        world_size = 2.
        arm_angles = np.cumsum(arm_pos)
        arm_angles = np.pi * arm_angles
        arm_points = np.array([np.cumsum(np.cos(arm_angles) * self._arm_lengths),
                               np.cumsum(np.sin(arm_angles) * self._arm_lengths)])
        hand_pos = np.array([np.sum(np.cos(arm_angles) * self._arm_lengths),
                             np.sum(np.sin(arm_angles) * self._arm_lengths)])

        # Screen parameters
        screen_center_w = np.ceil(self._width / 2)
        screen_center_h = np.ceil(self._height / 2)

        # Ratios
        world2screen = min(self._width / world_size, self._height / world_size)

        # Instantiating surface
        surface = gizeh.Surface(width=self._width, height=self._height)

        # Drawing Background
        background = gizeh.rectangle(lx=self._width, ly=self._height,
                                     xy=(screen_center_w, screen_center_h), fill=(1, 1, 1))
        background.draw(surface)

        # Drawing object
        if self._render_obj:
            objt = gizeh.circle(r=self._object_size * world2screen,
                                xy=(screen_center_w + object_pos[0] * world2screen,
                                    screen_center_h + object_pos[1] * world2screen),
                                fill=(0, 1, 1))
            objt.draw(surface)

        # Drawing goal
        if self._render_goal:
            objt = gizeh.circle(r=self._epsilon * world2screen,
                                xy=(screen_center_w + self.desired_goal[0] * world2screen,
                                    screen_center_h + self.desired_goal[1] * world2screen),
                                fill=(1, 0, 0))
            objt.draw(surface)

        # Drawing hand
        if self._render_hand:
            objt = gizeh.circle(r=self._object_size * world2screen / 4,
                                xy=(screen_center_w + hand_pos[0] * world2screen,
                                    screen_center_h + hand_pos[1] * world2screen),
                                fill=(1, 0, 1))
            objt.draw(surface)

        # Drawing arm
        if self._render_arm:
            screen_arm_points = arm_points * world2screen
            screen_arm_points = np.concatenate([[[0., 0.]], screen_arm_points.T], axis=0) + \
                                np.array([screen_center_w, screen_center_h])
            arm = gizeh.polyline(screen_arm_points, stroke=(0, 1, 0), stroke_width=3.)
            arm.draw(surface)

        if self._rgb:
            self._rendering = surface.get_npimage().astype(np.float32)
            self._rendering -= self._rendering.min()
            self._rendering /= self._rendering.max()
            if self._env_noise > 0:
                self._rendering = np.random.normal(self._rendering, self._env_noise)
                self._rendering -= self._rendering.min()
                self._rendering /= self._rendering.max()
        else:
            self._rendering = surface.get_npimage().astype(np.float32).sum(axis=-1)
            self._rendering -= self._rendering.min()
            self._rendering /= self._rendering.max()
            if self._env_noise > 0:
                self._rendering = np.random.normal(self._rendering, self._env_noise)
                self._rendering = np.clip(self._rendering, 0, 1)
                # self._rendering -= self._rendering.min()
                # self._rendering /= self._rendering.max()
            # Added by Adrien, makes training easier
            # self._rendering = -self._rendering + 1

        if mode == 'rgb_array':
            return (self._rendering * 255).astype(np.uint8)  # return RGB frame suitable for video
        elif mode is 'human':

            if self.viewer is None:
                self._start_viewer()
                # Retrieve image of corresponding to observation
                img = self._rendering
                self._img_artist = self._ax.imshow(img)
            else:
                # Retrieve image of corresponding to observation
                img = self._rendering
                self._img_artist.set_data(img)
                plt.draw()
                plt.pause(0.05)


    def _start_viewer(self):
        plt.ion()
        self.viewer = plt.figure()
        self._ax = self.viewer.add_subplot(111)


    def close(self):
        if self.viewer is not None:
            plt.close(self.viewer)
            self.viewer = None

    @property
    def dim_goal(self):
        return 2


class ArmBalls(gym.GoalEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, arm_lengths=np.array([0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]),
                 object_initial_pos=np.array([0.6, 0.6]), object_size=0.1,
                 distract_initial_pose=np.array([0.7, -0.45]), distract_size=0.15, distract_noise=0.2, n_timesteps=50,
                 epsilon=0.05, action_scaling=30, reward_type='sparse', one_goal=False, env_noise=0, obs_type='xyz'):
        """Initializes a new ArmBalls environment.

            This environment is similar to ArmBall except that there is a another ball (distractor) that cannot be
            controlled and moves randomly in the scene.

                Args:
                    arm_lengths (np_array): lengths of the robotic arm between each joint
                    object_initial_pos (np_array): initial pose for the ball
                    object_size (float): ball size, maximum distance to catch the ball
                    distrack_initial_pos (np_array): initial pose for the distractor
                    distrack_size (float): distractor size (impacts only rendering)
                    n_timesteps (int): maximum number of timesteps in the environment before reset
                    epsilon (float): the threshold after which a goal is considered achieved
                    action_scaling (float): the scaling for action (actions are between -1 and 1 before scaling)
                    env_noise (float): amount of gaussian noise for rendering
                    one_goal (np_array): if True then the goal is always the same (useful for debug and test)
                    reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
                    obs_type ('xyz', 'RGB', 'Vae' or 'Betavae'): the type of observation type, i.e. coordinates, images
                        or entangled/disentangled latent representation
        """

        assert arm_lengths.size < 8, "The number of joints must be inferior to 8"
        assert arm_lengths.sum() == 1., "The arm length must sum to 1."

        # We set the parameters
        self._n_joints = arm_lengths.size
        self._arm_lengths = arm_lengths
        self._object_initial_pos = object_initial_pos
        self._object_size = object_size
        self._distract_initial_pose = distract_initial_pose
        self._distract_size = distract_size
        self._distract_noise = distract_noise
        self.achieved_goal = self._object_initial_pos
        self._arm_pos = np.zeros(self._arm_lengths.size)
        self._hand_pos = np.zeros(2)
        self._object_handled = False
        self.desired_goal = None  # goal position
        self._n_timesteps = n_timesteps
        self.reward_type = reward_type
        self._epsilon = epsilon  # precision for sparse reward
        self._one_goal = one_goal
        self._action_scaling = action_scaling
        self._obs_type = obs_type
        
        if self.reward_type == 'sparse':
            self.reward_range = (-1, 0)
        else:
            self.reward_range = (-2, 0)


        # We set the space
        self.action_space = spaces.Box(low=-np.ones(self._n_joints),
                                       high=np.ones(self._n_joints),
                                       dtype=np.float32)
        if self._obs_type == 'xyz':
            observation = spaces.Box(low=-np.ones(self._n_joints + 4),  # joints + position of ball and distractor
                                     high=np.ones(self._n_joints + 4),
                                     dtype=np.float32)
        elif self._obs_type == 'RGB':
            observation = spaces.Box(low=0, high=1, shape=(self._n_joints + 64 * 64, ), dtype=np.float32)  # joints + 64 * 64 image

            self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(low=-np.ones(2), high=np.ones(2), dtype=np.float32),  # position of ball
                achieved_goal=spaces.Box(low=-np.ones(2), high=np.ones(2), dtype=np.float32),
                observation=observation,
            ))
        elif self._obs_type == 'Vae':
            from latentgoalexplo.representation.representation_pytorch import ArmBallsVAE
            self.ArmBallsVAE = ArmBallsVAE
            observation = spaces.Box(low=np.concatenate([-np.ones(self._n_joints), -3 * np.ones(10)]),  # joints + 10 latent values
                                     high=np.concatenate([np.ones(self._n_joints), 3 * np.ones(10)]),
                                     dtype=np.float32)
            # observation = spaces.Box(low=-3, high=3, shape=(self._n_joints + 10, ), dtype=np.float32)

            self.observation_space = spaces.Dict(dict(
                    desired_goal=spaces.Box(low=-np.ones(2), high=np.ones(2), dtype=np.float32),  # position of ball
                    achieved_goal=spaces.Box(low=-np.ones(2), high=np.ones(2), dtype=np.float32),
                    observation=observation,
            ))
        elif self._obs_type == 'Betavae':
            from latentgoalexplo.representation.representation_pytorch import ArmBallsBetaVAE
            self.ArmBallsBetaVAE = ArmBallsBetaVAE
            observation = spaces.Box(low=np.concatenate([-np.ones(self._n_joints), -3 * np.ones(10)]),  # joints + 10 latent values
                                     high=np.concatenate([np.ones(self._n_joints), 3 * np.ones(10)]),
                                     dtype=np.float32)
            # observation = spaces.Box(low=-3, high=3, shape=(self._n_joints + 10, ), dtype=np.float32)

            self.observation_space = spaces.Dict(dict(
                    desired_goal=spaces.Box(low=-np.ones(2), high=np.ones(2), dtype=np.float32),  # position of ball
                    achieved_goal=spaces.Box(low=-np.ones(2), high=np.ones(2), dtype=np.float32),
                    observation=observation,
            ))

        self._env_noise = env_noise
        self._width = 500
        self._height = 500
        self._rendering = np.zeros([self._height, self._width, 3])
        self._rendering[0] = 1

        self.viewer = None

        # We set to None to rush error if reset not called
        self.reward = None
        self._observation = None
        self._steps = None
        self._done = None

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        return seed

    def _sample_goal(self):
        goal = np.random.uniform(-1, 1, 2)
        return goal

    def compute_reward(self, achieved_goal, goal, info=None):
        if achieved_goal.ndim > 1:
            d = np.linalg.norm(achieved_goal - goal, ord=2, axis=1)
        else:
            d = np.linalg.norm(achieved_goal - goal, ord=2)
        if self.reward_type == 'sparse':
            return -(d > self._epsilon).astype(np.int)
        else:
            return -d

    def _is_success(self, achieved_goal, goal):
        if achieved_goal.ndim > 1:
            d = np.linalg.norm(achieved_goal - goal, ord=2, axis=1)
        else:
            d = np.linalg.norm(achieved_goal - goal, ord=2)
        return -(d > self._epsilon).astype(np.int)

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        """

        # We compute the position of the end effector
        self._arm_pos = np.clip(self._arm_pos + action / self._action_scaling,
                                a_min=-np.ones(self._n_joints),
                                a_max=np.ones(self._n_joints))
        angles = np.cumsum(self._arm_pos)
        angles_rads = np.pi * angles
        self._hand_pos = np.array([np.sum(np.cos(angles_rads) * self._arm_lengths),
                                   np.sum(np.sin(angles_rads) * self._arm_lengths)])

        # We check if the object is handled and we move it.
        if np.linalg.norm(self._hand_pos - self.achieved_goal, ord=2) < self._object_size:
            self._object_handled = True
        if self._object_handled:
            self.achieved_goal = self._hand_pos

        # We move the distractor
        self._actual_distract_pose = self._actual_distract_pose + np.random.randn(2) * self._distract_noise
        self._actual_distract_pose = np.clip(self._actual_distract_pose, -.95, 0.95)

        # We update observation and reward
        if self._obs_type == 'xyz':
            self._observation = np.concatenate([self._arm_pos, self._actual_distract_pose, self.achieved_goal])
        elif self._obs_type == 'RGB':
            self._calc_rendering(width=64, height=64)
            self._observation = np.concatenate([self._arm_pos, self._rendering.sum(axis=-1).flatten()])
        elif self._obs_type == 'Vae':
            self._calc_rendering(width=64, height=64)
            self.ArmBallsVAE.act(X_pred=self._rendering)
            self._observation = np.clip(self.ArmBallsVAE.representation.squeeze(), -3, 3)
            self._observation = np.concatenate([self._arm_pos, self._observation])
        elif self._obs_type == 'Betavae':
            self._calc_rendering(width=64, height=64)
            self.ArmBallsBetaVAE.act(X_pred=self._rendering)
            self._observation = np.clip(self.ArmBallsBetaVAE.representation.squeeze(), -3, 3)
            self._observation = np.concatenate([self._arm_pos, self._observation])

        self.reward = self.compute_reward(self.achieved_goal, self.desired_goal)
        self._steps += 1
        if self._steps == self._n_timesteps:
            self._done = True

        self.obs = dict(observation=self._observation, desired_goal=self.desired_goal, achieved_goal=self.achieved_goal)

        info = {}
        info['is_success'] = self._is_success(self.achieved_goal, self.desired_goal) == 0

        return self.obs, self.reward, self._done, info

    def reset(self, goal=None):
        # We reset the simulation
        if self._one_goal:
            self.desired_goal = np.array([0.3, 0.5])
        else:
            if goal is not None:
                self.desired_goal = goal
            else:
                self.desired_goal = self._sample_goal()
        self.achieved_goal = self._object_initial_pos
        self._actual_distract_pose = self._distract_initial_pose.copy()
        self._arm_pos = np.zeros(self._arm_lengths.shape)
        self._object_handled = False
        self._steps = 0
        self._done = False

        if self._obs_type == 'xyz':
            self._observation = np.concatenate([self._arm_pos, self._actual_distract_pose, self.achieved_goal])
        elif self._obs_type == 'RGB':
            self._calc_rendering(width=64, height=64)
            self._observation = np.concatenate([self._arm_pos, self._rendering.sum(axis=-1).flatten()])
        elif self._obs_type == 'Vae':
            self._calc_rendering(width=64, height=64)
            self.ArmBallsVAE.act(X_pred=self._rendering)
            self._observation = np.clip(self.ArmBallsVAE.representation.squeeze(), -3, 3)
            self._observation = np.concatenate([self._arm_pos, self._observation])
        elif self._obs_type == 'Betavae':
            self._calc_rendering(width=64, height=64)
            self.ArmBallsBetaVAE.act(X_pred=self._rendering)
            self._observation = np.clip(self.ArmBallsBetaVAE.representation.squeeze(), -3, 3)
            self._observation = np.concatenate([self._arm_pos, self._observation])

        # We compute the initial reward.
        self.reward = self.compute_reward(self.achieved_goal, self.desired_goal)

        self.obs = dict(observation=self._observation, desired_goal=self.desired_goal, achieved_goal=self.achieved_goal)

        return self.obs

    def _calc_rendering(self, width, height):
        # We retrieve arm and object pose
        arm_pos = self._arm_pos
        object_pos = self.achieved_goal
        distract_pos = self._actual_distract_pose

        # World parameters
        world_size = 2.
        arm_angles = np.cumsum(arm_pos)
        arm_angles = np.pi * arm_angles
        arm_points = np.array([np.cumsum(np.cos(arm_angles) * self._arm_lengths),
                               np.cumsum(np.sin(arm_angles) * self._arm_lengths)])
        hand_pos = np.array([np.sum(np.cos(arm_angles) * self._arm_lengths),
                             np.sum(np.sin(arm_angles) * self._arm_lengths)])

        # Screen parameters
        screen_center_w = np.ceil(width / 2)
        screen_center_h = np.ceil(height / 2)

        # Ratios
        world2screen = min(width / world_size, height / world_size)

        # Instantiating surface
        surface = gizeh.Surface(width=width, height=height)

        # Drawing Background
        background = gizeh.rectangle(lx=width, ly=height,
                                     xy=(screen_center_w, screen_center_h), fill=(1, 1, 1))
        background.draw(surface)

        # Drawing object
        objt = gizeh.circle(r=self._object_size * world2screen,
                            xy=(screen_center_w + object_pos[0] * world2screen,
                                screen_center_h + object_pos[1] * world2screen),
                            fill=(1, 1, 0))
        objt.draw(surface)

        # Drawing distractor
        distractor = gizeh.circle(r=self._distract_size * world2screen,
                            xy=(screen_center_w + distract_pos[0] * world2screen,
                                screen_center_h + distract_pos[1] * world2screen),
                            fill=(0, 0, 1))
        distractor.draw(surface)

        # Drawing goal
        goal = gizeh.circle(r=self._epsilon * world2screen,
                            xy=(screen_center_w + self.desired_goal[0] * world2screen,
                                screen_center_h + self.desired_goal[1] * world2screen),
                            fill=(1, 0, 0))
        goal.draw(surface)

        # Drawing hand
        hand = gizeh.circle(r=self._object_size * world2screen / 4,
                            xy=(screen_center_w + hand_pos[0] * world2screen,
                                screen_center_h + hand_pos[1] * world2screen),
                            fill=(1, 0, 1))
        hand.draw(surface)

        # Drawing arm
        screen_arm_points = arm_points * world2screen
        screen_arm_points = np.concatenate([[[0., 0.]], screen_arm_points.T], axis=0) + \
                            np.array([screen_center_w, screen_center_h])
        arm = gizeh.polyline(screen_arm_points, stroke=(0, 1, 0), stroke_width=3.)
        arm.draw(surface)

        self._rendering = surface.get_npimage().astype(np.float32)
        self._rendering -= self._rendering.min()
        self._rendering /= self._rendering.max()
        if self._env_noise > 0:
            self._rendering = np.random.normal(self._rendering, self._env_noise)
            self._rendering -= self._rendering.min()
            self._rendering /= self._rendering.max()

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

        self._calc_rendering(width=self._width, height=self._height)
        if mode == 'rgb_array':
            return (self._rendering * 255).astype(np.uint8)  # return RGB frame suitable for video
        elif mode is 'human':

            if self.viewer is None:
                self._start_viewer()
                # Retrieve image of corresponding to observation
                img = self._rendering
                self._img_artist = self._ax.imshow(img)
            else:
                # Retrieve image of corresponding to observation
                img = self._rendering
                self._img_artist.set_data(img)
                plt.draw()
                plt.pause(0.05)


    def _start_viewer(self):
        plt.ion()
        self.viewer = plt.figure()
        self._ax = self.viewer.add_subplot(111)


    def close(self):
        if self.viewer is not None:
            plt.close(self.viewer)
            self.viewer = None

    @property
    def dim_goal(self):
        return 2


if __name__ == '__main__':
    a = ArmBall()
    a.reset()
    for i in range(50):
        a.step(np.random.uniform(-1, 1, 7))
        a.render()
    a = ArmBalls()
    a.reset()
    for i in range(50):
        a.step(np.random.uniform(-1, 1, 7))
        a.render()
