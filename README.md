## Installation

clone repository
cd into gym_flowers
pip install -e .

Then use it as follows:

import gym
import gym_flowers
env = gym.make('ArmBall-v0')

## List of environments

* ArmBall-v0
ArmBall environment with a sparse reward. 7 joint robotic arm. Each action correspond to setting the joints angles.

* ArmBall-v1
ArmBall environment with a dense reward. 7 joint robotic arm. Each action correspond to setting the joints angles.
