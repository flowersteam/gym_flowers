import numpy as np
import gym
import gym_flowers

env = gym.make('Rooms5-v0')
env.reset()
task = 1
goal = np.random.uniform(-1,1,2)
goal = np.array([0,1])
env.unwrapped.reset_task_goal(task=task, goal=goal)
env.render()
while True:
    act = np.array([0,0,0])
    obs = env.step(act)
    print(obs[1])
    env.render()