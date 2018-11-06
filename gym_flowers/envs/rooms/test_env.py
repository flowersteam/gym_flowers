import numpy as np
import gym
import gym_flowers

env = gym.make('Rooms-v0')
env.reset()
task = 2
goal = np.random.uniform(-1,1,2)
env.unwrapped.reset_task_goal(task=task, goal=goal)
env.render()
for i in range(500):
    act = np.array([1,-1,1])
    obs = env.step(act)
    # print(obs[1])
    env.render()