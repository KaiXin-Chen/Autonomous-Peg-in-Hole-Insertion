"""
This file will include the networks for imitation learning
"""
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor,get_graph_node_names
import torch
from torch import nn

class Imi_networks(nn.Module):
    def __init__(self):
        super.__init__()

        '''
        天恒 我VISION ENCODER用的是ResNet 18 做的transfer learning, output shape 应该是512，你用512作为input shape就好
        '''

        self.model = nn.Sequential(
            nn.Linear(512, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 60),
            nn.ReLU(),
            nn.Linear(60, 3),
        )

    def forward(self, x):
        return self.model(x).flatten()

    raise NotImplementedError


