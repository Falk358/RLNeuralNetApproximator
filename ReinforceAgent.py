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
from pprint import pprint


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
        if self.jointNN:
            chosen_activation = self.h(torch.tensor(observation))
            chosen_activation = torch.log(chosen_activation)
            self.optim.zero_grad()
            loss = self.gamma**t * -target * chosen_activation
            loss.backward()
            self.optim.step()
        else:
            activations = []
            for current_h in self.h:
                activations.append(current_h(torch.tensor(observation)))
            
            activations_appended = torch.cat(activations, dim=0)
            softmax_dist = torch.distributions.Categorical(activations_appended)
            log_prob_action = softmax_dist.log_prob(torch.tensor(action, requires_grad=False))
            self.optim.zero_grad()
            loss = self.gamma**t * -target * log_prob_action
            loss.backward()
            self.optim.step()





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


    def verifyUpdate(self, qa_before_update, qa_after_update, target):
        """
        verifies if the neural net prediction is closer to the target after update()
        run this after running update
        returns boolean if verfication successful
        """

        difference_before = target -qa_before_update
        difference_after = target - qa_after_update

        return difference_before <= difference_after

    # This method trains the agent for one episode on the given
    # gymnasium environment, and is called by train() above.
    # This method should repeatedly call chooseAction() and update().
    # This method must return
    #   T, the length of this episode in time steps, and
    #   G, the (discounted) return earned during this episode.
    # This code will be very similar to DiscreteMonteCarloAgent.trainEpisode().
    def trainEpisode(self, env):
        # BEGIN YOUR CODE HERE
        print("-----------------new episode -------------------------")
        observation, info = env.reset()
        T = 0
        G = 0.0 
        for timestep in range(1000):
            #pprint(observation)
            action = self.chooseAction(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            T += 1
            if timestep == 0:
                G = reward
            else:
                G += (self.gamma**timestep) * reward
            u_target = G
            qa_before = self.h[action](torch.tensor(observation))
            self.update(t=timestep, action= action, target= u_target, observation = observation)
            qa_after = self.h[action](torch.tensor(observation))
            verified = self.verifyUpdate(qa_before_update=qa_before, qa_after_update=qa_after, target= u_target)
            if not verified:
                print("Warning! updated neural net activation did not move towards target!!!")
            if terminated or truncated:
                print("environment terminated with info: ")
                pprint(info)
                observation, _ = env.reset()
                break


        # END YOUR CODE HERE
        return T, G
