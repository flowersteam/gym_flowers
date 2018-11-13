import numpy as np
import gym
import gym_flowers

env = gym.make('Rooms5-v0')
env.reset()
task = 5
# goal = np.random.uniform(-1,1,2)
goal = np.array([1,1])
env.unwrapped.reset_task_goal(task=task, goal=goal)
g_id = env.unwrapped.tasks_g_id
dim_goal = env.unwrapped.goal.size
env.render()
n_tasks = 10
while True:
    act = np.array([0,0,0])
    obs = env.step(act)
    print(obs[1])
    env.render()