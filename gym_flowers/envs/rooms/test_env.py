import numpy as np
import gym
import gym_flowers

env = gym.make('Rooms3-v0')
env.reset()
task = 0
goal = np.random.uniform(-1,1,2)
# goal = np.array([1,1])
env.unwrapped.reset_task_goal(task=task, goal=goal)
g_id = env.unwrapped.tasks_g_id
dim_goal = env.unwrapped.goal.size
env.render()
n_tasks = 10
for i in range(100):
    act = np.random.uniform(-1,1,3)
    obs = env.step(act)
    print(obs[1])
    print(obs[0]['desired_goal'])
    env.render()