import numpy as np
import torch
import gym
from gym import ObservationWrapper

ENV_NAME = "MiniGrid-Empty-Random-6x6-v0"

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Variable(torch.autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

class TransformObservation(ObservationWrapper):
    """
    observationが(H,W,C)なので(C,H,W)に変換する
    """
    def __init__(self,env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3,56,56))

    def observation(self,observation):
        return np.reshape(observation,(3,56,56))
