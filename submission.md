# My Observations and Comments

I used the same Neural Network configuration for Sarsa and Monte Carlo, but changed it for REINFORCE.

## CartPoleSarsa

Here, the results are a bit underwhelming, even though I believe my Implementation to be correct. It seems that Sarsa performs well periodically (large spikes at the beginning and around episodes 2500), but falls of after.
See CartPoleMonteCarlo section for speculation on why this could be.

Parameters: *gamma* = 1, *alpha* = 0.0001

![](CartPoleSarsa-stats.svg)

## CartPoleMonteCarlo

Curiously, this algorithm doesn't perform as well, even though the implementation is most likely correct since it is almost identical to **CartPoleReinforce**'s `trainEpisode()` method (both are Monte Carlo Methods at heart after all.) One possible explanation for this could be the use of a Neural network for each Action in Action space: to my knowledge this is rather unusual, since examples i came across dont use a separate network for each action: [REINFORCE Implementation from pytorch documentation](https://pytorch.org/docs/stable/distributions.html). Since the networks I used are quite small, and not all networks are updated at each `update()` call as per the task given, it is possible that this caused my model to underfit. 




Parameters: *gamma* = 1, *alpha* = 0.0001

![](CartPoleMonteCarlo-stats.svg)

## CartPoleReinforce

**Important** Neural Network used only supports single Neural Network implementation, please run with `jointNN=True` 

We were able to get this run to work quite well, this possibly because we switched to a joint Neural Network instead of using a separate Neural Network for each action. We ran the experiment with default parameters: *gamma* = 1, *alpha* = 0.00001.


![](CartPoleReinforce-stats.svg)



## CartPoleReinforce with Baseline

not implemented due to time constraints
