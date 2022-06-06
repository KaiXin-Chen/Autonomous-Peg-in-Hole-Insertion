"""
This file will include networks for visual inputs
"""
from torchvision.models import resnet18
from torchvision.models import convnext_tiny
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
        # here is just adjust the shape of features to be (..., 512), delete redundant dim
        feats = feats.squeeze(3).squeeze(2)
        # final dim is (..., 512), may add another fc layer to change the output dim
        return feats

def make_vision_encoder():
    # define a visual encoder using resnet 18, with pretrained parameters
    vision_extractor =  resnet18(pretrained=True)
    # take the output after pooling layer and before final linear layer
    # this layer contains the visual features we want
    vision_extractor = create_feature_extractor(vision_extractor, ["avgpool"])
    return Encoder(vision_extractor)


class pos_feature_extactor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
        )

    def forward(self, x):
        return self.model(x)

def make_pos_encoder():
    return pos_feature_extactor()

# def make_pos_encoder():
#     '''
#     pos_extractor = resnet18(pretrained=True)
#     pos_extractor.conv1 = nn.Conv2d(1, 64, kernel_size=7, padding=3, bias=True)
#     # pos_extractor.conv1.weight.data.fill_(0.01)
#     # pos_extractor.conv1.bias.data.fill_(0.01)
#     pos_extractor = create_feature_extractor(pos_extractor, ["avgpool"])
#     return Encoder(pos_extractor)
#     '''

#     return  nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(3, 64),
#             nn.ReLU(),
#             nn.Linear(64, 512),

#         )