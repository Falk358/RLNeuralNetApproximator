# Base class for RL agents for episodic tasks
# - derived from DiscreteAgent

from pprint import pprint
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
        action, _ = self.chooseAction(observation, action_space)
        T = 0
        G = 0.0 
        while True:
            qa_action = self.q[action](torch.tensor(observation))
            observation_new, reward, terminated, truncated, info = env.step(action)
            action_new, _ = self.chooseAction(observation_new, action_space)
            qa_action_new = self.q[action_new](torch.tensor(observation_new))
            u_target = qa_action_new  * self.gamma + reward
            if terminated or truncated:
                print("environment terminated with info: ")
                pprint(info)
                self.update(action = action, target= reward, qa=qa_action)
                env.reset()
                T += 1
                G += (self.gamma**T) * reward
                break
            self.update(action= action, target= u_target, qa = qa_action)
            T += 1
            G += (self.gamma**T) * reward
            observation = observation_new
            action = action_new




        # END YOUR CODE HERE
        return T, G
