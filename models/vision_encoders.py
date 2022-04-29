"""
This file will include networks for visual inputs
"""
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor,get_graph_node_names
import torch
from torch import nn

class Encoder(nn.Module):
    # TODO: Implement the encoder template for visual inputs
    def __init__(self, feature_extractor):
        super().__init__()
        layers = list(feature_extractor.children())[:-1] #THis is specific to ResNet 18 and should be changed for other models
        self.feat=nn.Sequential(*layers)

    def forward(self, x):
        feat = self.feat(x)
        return feat.flatten()

def make_vision_encoder():
    # TODO: make vision encoders using the above template
    fea = resnet18()
    return Encoder(fea)
