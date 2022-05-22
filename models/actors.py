"""
This file will include actors for the entire task
"""
from torch.nn.modules.activation import MultiheadAttention
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor
import torch
from torch import nn
import cv2
import numpy as np

class robotActor(torch.nn.Module):
    def __init__(self, vision_encoder, pos_encoder, imi_models, args):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.pos_encoder = pos_encoder
        self.imi_models = imi_models

    def forward(self, visual_input, pos_input, freeze):
        if freeze:
            with torch.no_grad():
                visual_feats = self.vision_encoder(visual_input).detach()
                pos_feats = self.pos_encoder(pos_input).detach()
        else:
            visual_feats = self.vision_encoder(visual_input)
            pos_feats = self.pos_encoder(pos_input)
        embeds = torch.cat((visual_feats, pos_feats), dim=-1)
        action_logits = self.imi_models(embeds)
        return action_logits