#!/usr/bin/env python
import gymnasium as gym
from torch import nn
import ReinforceAgent as rl


env = gym.make('CartPole-v1', max_episode_steps=100)
dimObs = env.observation_space.shape[0]

class H(nn.Module):
    def __init__(self):
        super(H, self).__init__()
        # BEGIN YOUR CODE HERE
        self.fc_input = nn.Linear(dimObs, 8)
        self.relu = nn.ReLU()
        self.fc_middle = nn.Linear(8, 1)
        self.softmax = nn.Softmax()

        # END YOUR CODE HERE

    def forward(self, x):
        x = self.fc_input(x)
        x = self.relu(x)
        x = self.fc_middle(x)
        x = self.softmax(x)
        return x


import sys
run = int(sys.argv[1]) if len(sys.argv) == 2 else None

# Play with gamma, alpha, and perhaps other pararameters:
agent = rl.Agent(H, env.action_space.n, gamma=1, alpha=0.00004)
agent.train(env)
if run is not None:
    agent.save(f"CartPoleReinforce-{run:02d}.npy")
