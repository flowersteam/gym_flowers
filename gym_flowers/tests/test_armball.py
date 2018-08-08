import gym
import gym_flowers


env = gym.make('ArmBallsrgb-v0')
obs = env.reset()
assert obs['observation'].shape == (64, 64, 3), 'Observation not of good shape. Expected (64, 64), got ' + ", ".join(str(x) for x in obs['observation'].shape)

env = gym.make('ArmBallsvae-v0')
obs = env.reset()
assert obs['observation'].shape == (10, ), 'Observation not of good shape. Expected 10, got ' + ", ".join(str(x) for x in obs['observation'].shape)