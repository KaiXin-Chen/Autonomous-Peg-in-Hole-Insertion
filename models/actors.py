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
        self.bottleneck = nn.Linear(1024, 512)

    def forward(self, visual_input, pos_input, freeze):
        
        batch, channel, H, W = visual_input.shape
        visual_input = visual_input.view(-1, 3, H, W) # (2*N / N, 3/6, H, W)

        if freeze:
            with torch.no_grad():
                visual_feats = self.vision_encoder(visual_input).detach()
                pos_feats = self.pos_encoder(pos_input).detach()
        else:
            visual_feats = self.vision_encoder(visual_input)
            pos_feats = self.pos_encoder(pos_input)
        #convert back to (batch, 2*embeds)    
        visual_feats = visual_feats.view(batch, -1)
        if channel == 6:
            visual_feats = self.bottleneck(visual_feats) #(batch, 512)
        embeds = torch.cat((visual_feats, pos_feats), dim=-1)
        action_logits = self.imi_models(embeds)
        return action_logits