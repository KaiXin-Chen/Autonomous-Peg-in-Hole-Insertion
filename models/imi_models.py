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
        self.model = nn.Sequential(
            nn.Linear(1024, 3)
        )

    def forward(self, x):
        return self.model(x)

    raise NotImplementedError


