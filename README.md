Set of custom environments used in the Inria Flowers team

## Installation

```
git clone https://github.com/qdrn/gym_flowers.git
cd gym_flowers
pip install -e .
```

Then use it as follows:

```
import gym
import gym_flowers
env = gym.make('ArmBall-v0')
```

## List of supported environments

* MultiTaskFetchArm-v0
Modification of the Fetch Arm environments from [OpenAI Gym](https://github.com/openai/gym). A 7-DoF robotic arm faces 5 cubes. It has access to a set of 7 tasks. T_0: Place the gripper at the target location; T1: Place cube 1 at the 2D target location on the table; T_2: Place cube 1 at the 3D target location above cube 0; T3: Stack cube 0 above cube 1; T4-6: Place moving and out-of-reach cube 2-4 (respectively) to 2D target location on the table (distracting tasks).
Observations: list of all objects position and velocities (49 obs). Actions: 3D cartesian actions and gripper action (4 in total). Reward 0 when goal is met (sparse reward), -1 otherwise.

## List of environments in development

* ArmBall-v0
ArmBall environment with a sparse reward. 7 joint robotic arm. Each action correspond to setting the joints angles.

* ArmBall-v1
ArmBall environment with a dense reward. 7 joint robotic arm. Each action correspond to setting the joints angles.
