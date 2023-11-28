# Base class for REINFORCE RL agents
# - with separate or one joint h(s) preference neural net(s) for discrete actions
# - using a policy parametrized as soft-max in action preferences.
#
# The first argument of its constructor is the class (not instance)
# derived from torch.nn.Module that implements the h(s) neural net(s).
# Its constructor must not take any arguments.

import torch
import numpy as np
from random import random


class Agent():
    def __init__(self, H, nActions, alpha=0.000001, gamma=1,
                 nEpisodes=25000, jointNN=False):
        self.gamma = gamma
        self.jointNN = jointNN
        if jointNN:
            self.h = H()
            self.optim = torch.optim.SGD(self.h.parameters(), alpha)
        else:
            self.h = [H() for _ in range(nActions)]
            # One common optimizer for all nActions' h functions:
            self.optim = torch.optim.SGD([p for ha in self.h
                                          for p in ha.parameters()], alpha)
        self.episodes = np.zeros((nEpisodes, 2))

    # Implements a policy parametrized as soft-max in action preferences.
    def chooseAction(self, obs):
        with torch.no_grad():
            ha = self.h(torch.tensor(obs)).exp().numpy() if self.jointNN \
                else np.array([ha(torch.tensor(obs)).exp().item()
                               for ha in self.h])
            actions = ha.cumsum()
            choice = random() * actions[-1]
            for action in range(len(actions)):
                if choice < actions[action]:
                    # print(f"{ha=} {actions=} {choice=} {action=}")
                    return action
            print(f"{ha=} {actions=} {choice=}")
            assert False

    # Perform a gradient-ascent REINFORCE parameter update.
    # t is the time step of the current episode.
    # action is A_t, observation is S_t, and target is G_t.
    # self.optim.zero_grad() and self.optim.step() should be called
    # either here or in trainEpisode() below.
    def update(self, t, action, observation, target):
        # BEGIN YOUR CODE HERE

        # END YOUR CODE HERE

    # This method trains the agent on the given gymnasium environment.
    # The method for training one episode, trainEpisode(env), is defined below.
    def train(self, env):
        for episode in range(len(self.episodes)):
            self.episodes[episode,:] = self.trainEpisode(env)
            print(f"{episode=:5d}, t={self.episodes[episode,0]:3.0f}: G={self.episodes[episode,1]:6.1f}")

    # Call this to save data collected during training for further analysis.
    def save(self, file):
        np.save(file, self.episodes)        

    # This method trains the agent for one episode on the given
    # gymnasium environment, and is called by train() above.
    # This method should repeatedly call chooseAction() and update().
    # This method must return
    #   T, the length of this episode in time steps, and
    #   G, the (discounted) return earned during this episode.
    # This code will be very similar to DiscreteMonteCarloAgent.trainEpisode().
    def trainEpisode(self, env):
        # BEGIN YOUR CODE HERE

        # END YOUR CODE HERE
        return T, G
