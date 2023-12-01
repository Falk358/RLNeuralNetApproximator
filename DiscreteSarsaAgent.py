# Base class for RL agents for episodic tasks
# - derived from DiscreteAgent

from pprint import pprint
import sys
import torch
import DiscreteAgent as Discrete
from gymnasium import Env


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
    def trainEpisode(self, env: Env):
        # BEGIN YOUR CODE HERE
        print("-----------------new episode -------------------------")
        action_space = env.action_space
        observation, info = env.reset()
        T = 0
        G = 0.0 
        for timestep in range(1000):
            action, action_value = self.chooseAction(observation, action_space)
            observation, reward, terminated, truncated, info = env.step(action)
            qa_before = self.q[action](torch.tensor(observation))
            u_target = qa_before  * self.gamma + reward
            self.update(action= action, target= u_target, qa = qa_before)

            qa_after = self.q[action](torch.tensor(observation))
            verified = self.verifyUpdate(qa_before_update= qa_before, qa_after_update= qa_after, target=u_target)
            if not verified:
                print("Warning! neural net activation did not move towards target!!")

            T += 1
            if timestep == 0:
                G = reward
            else:
                G += (self.gamma**timestep) * reward
            if terminated or truncated:
                print("environment terminated with info: ")
                pprint(info)
                observation, _ = env.reset()
                break


        # END YOUR CODE HERE
        return T, G
