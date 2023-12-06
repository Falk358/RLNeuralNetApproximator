# Base class for RL agents for episodic tasks
# - derived from DiscreteAgent

import numpy as np
import torch
from pprint import pprint
import DiscreteAgent as Discrete


class Agent(Discrete.Agent):
    def __init__(self, Q, nActions, gamma=1, **kwargs):
        super(Agent, self).__init__(Q, nActions, **kwargs)
        self.gamma = gamma

    # This method trains the agent for one episode on the given
    # gymnasium environment, and is called by the base class' train() method.
    # This method should repeatedly call chooseAction() and update().
    # This method must return
    #   T, the length of this episode in time steps, and
    #   G, the (discounted) return earned during this episode.
    def trainEpisode(self, env):
        # BEGIN YOUR CODE HERE
        print("-----------------new episode -------------------------")
        action_space = env.action_space
        observation, info = env.reset()
        T = 0
        G = 0.0 
        rewards = []
        states = []
        actions = []
        while True:
            action, _ = self.chooseAction(observation, action_space)
            observation, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            states.append(torch.tensor(observation).detach())
            actions.append(action)  
            T += 1
            if terminated or truncated:
                print("environment terminated with info: ")
                pprint(info)
                observation, _ = env.reset()
                break
        
        discounted_rewards = np.zeros_like(rewards)
        for i in reversed(range(len(rewards))):
            G = G * self.gamma + rewards[i]
            discounted_rewards[i] = G

        for i in range(len(discounted_rewards)):
            self.update(action=actions[i], target=discounted_rewards[i], qa= self.q[actions[i]](states[i]))


        


        # END YOUR CODE HERE
        return T, G
