import torch
import torch.nn as nn
import torchvision

from utils import DEVICE


class SimpleQNet(torch.nn.Module):
    def __init__(self,observation_space,num_actions):
        """
        arguments:
            observation_space: int 入力の次元数=observationのshape
            num_actions: int 行動の次元数(outputの次元数)
        """
        super(SimpleQNet,self).__init__()
        self.layers = torch.nn.Sequential(
              torch.nn.Linear(observation_space,50),
              torch.nn.ReLU(),
              torch.nn.Linear(50,50),
              torch.nn.ReLU(),
              torch.nn.Linear(50,num_actions),
        )
        ### TODO kernel initialize

    def forward(self,x):
        return self.layers(x)

class CNNQNet(torch.nn.Module):
    def __init__(self,observation_space,num_actions):
        super(CNNQNet,self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(observation_space[0],32,kernel_size=8,stride=4),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=4,stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,stride=1)
        )
        convw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(56,8,4),4,2),3,1)
        convh = convw # 正方形なので
        linear_input_size = convw*convh*64
        self.layers = nn.Sequential(
            nn.Linear(linear_input_size,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,num_actions)
        )

    def forward(self,x):
        x = x.to(DEVICE)
        x = self.feature_extractor(x)
        x = x.view(x.size(0),-1)
        x = self.layers(x)
        return x

    def conv2d_size_out(self,size,kernel_size,stride):
        return (size - (kernel_size - 1) - 1) // stride  + 1
