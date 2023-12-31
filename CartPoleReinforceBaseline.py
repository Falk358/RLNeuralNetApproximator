#!/usr/bin/env python
import gymnasium as gym
from torch import nn
import ReinforceBaselineAgent as rl


env = gym.make('CartPole-v1', max_episode_steps=100)
dimObs = env.observation_space.shape[0]

class H(nn.Module):
    def __init__(self):
        super(H, self).__init__()
        # BEGIN YOUR CODE HERE

        # END YOUR CODE HERE
        
    def forward(self, x):
        return self.nn(x)


class V(nn.Module):
    def __init__(self):
        super(V, self).__init__()
        # BEGIN YOUR CODE HERE

        # END YOUR CODE HERE
        
    def forward(self, x):
        return self.nn(x)


import sys
run = int(sys.argv[1]) if len(sys.argv) == 2 else None

# Play with gamma, alpha, and perhaps other pararameters:
agent = rl.Agent(H, V, env.action_space.n, gamma=1,
                 alpha=0.00001, alphaw = 0.0001)
agent.train(env)
if run is not None:
    agent.save(f"CartPoleReinforceBaseline-{run:02d}.npy")
