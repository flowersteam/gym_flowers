import numpy as np
import gym
import gym_flowers

env = gym.make('Room-v0')
env.reset()
task = 0
goal = np.random.uniform(-1,1,2)
env.unwrapped.reset_task_goal(task=task, goal=goal)

for i in range(500):
    act = np.random.uniform(-1,1,3)
    env.step(act)
    env.render()