import gym
import gym_flowers
import numpy as np
import os

os.environ['LD_LIBRARY_PATH']=':'+os.environ['HOME']+'/.mujoco/mjpro150/bin:'


env = gym.make('MultiTaskFetchArm8-v5')
env.unwrapped.reset_task_goal(goal=np.zeros([3]), task=1)
obs_0 = env.reset()
print(obs_0)
for _ in range(2000):
    obs, rew, done, info = env.step(np.array([-1, -1, 1, 0]))
    print(obs)
    env.render()
obs = env.reset()
obs, _, _, _ = env.step(np.array([-1, 1, 0, 0]))
