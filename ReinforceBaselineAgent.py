# Base class for RL agents using REINFORCE with baseline
# - with a v(s) neural net
# The rest is inherited from its base class.
#
# The second argument of its constructor is the class (not instance)
# derived from torch.nn.Module that implements the v(s) neural net.
# Its constructor must not take any arguments.

import torch
import numpy as np
import ReinforceAgent as Reinforce


class Agent(Reinforce.Agent):
    def __init__(self, H, V, nActions, alphaw=0.0001, **kwargs):
        super(Agent, self).__init__(H, nActions, **kwargs)
        self.v = V()
        self.optimv = torch.optim.SGD(self.v.parameters(), alphaw)

    # This method does almost the same as its base-class counterpart, and
    # also computes the gradient-based parameter update of the v(s) network.
    def update(self, t, action, observation, target):
        # BEGIN YOUR CODE HERE

        # END YOUR CODE HERE
