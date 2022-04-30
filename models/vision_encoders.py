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
        self.feature_extractor = feature_extractor

    def forward(self, x):
        feats = self.feature_extractor(x)["avgpool"]
        feats = feats.squeeze(3).squeeze(2)
        # final dim is (..., 512), may add another fc layer to change the output dim
        return feats

def make_vision_encoder():
    # define a visual encoder using resnet 18, with pretrained parameters
    vision_extractor = resnet18(pretrained=True)
    # take the output after pooling layer and before final linear layer
    # this layer contains the visual features we want
    vision_extractor = create_feature_extractor(vision_extractor, ["avgpool"])
    return Encoder(vision_extractor)
