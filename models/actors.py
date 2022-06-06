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
        self.mha = MultiheadAttention(128, 8) # 128 if for one single image
        self.bottleneck = nn.Linear(768, 512)
        self.two_cam_bottleneck = nn.Linear(1024, 512)
        self.use_two_cam = args.num_camera == 2
        self.use_convnext = args.use_convnext

    def forward(self, visual_input, pos_input, freeze):
                
        batch, frames, channel, H, W = visual_input.shape
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
        if self.use_two_cam:
            visual_feats = self.two_cam_bottleneck(visual_feats) #(batch, 512)
        elif self.use_convnext:
            visual_feats = self.bottleneck(visual_feats)
        
        #0 attention in one cam with history
        #dataset can input history
        #mha_input = visual_feats.view(4, -1, 128)
        #mha_out, weights = self.mha(mha_input, mha_input, mha_input)
        #mha_out += mha_input
        #embeds = torch.concat([mha_out[i] for i in range(mha_out.shape[0])], 1)
        
        #1 attention in one single image
        #visual_feats -> (batch, 512) -> (4, batch, 128)
        '''
        mha_input = torch.split(visual_feats,128, dim=-1)
        mha_input=torch.stack(mha_input)
        mha_out, weights = self.mha(mha_input, mha_input, mha_input)
        mha_out += mha_input
        embeds = torch.concat([mha_out[i] for i in range(mha_out.shape[0])], 1)
        '''
        # #2 attention across different embeds
        # #stack two cam visual feat and pos feat, attention on these two feature
        # mha_input = torch.stack((visual_feats, pos_feats, second_can_feats), 0) # (2, batch, 512)
        # mha_out, weights = self.mha(mha_input, mha_input, mha_input) #self attention, mhaout(2, batch, 512)
        # # [[0.1, 0.8], [0.9, 0.2]]
        # mha_out += mha_input
        # embeds = torch.concat([mha_out[i] for i in range(mha_out.shape[0])], 1) #(batch, 1024)
        


        embeds = visual_feats#torch.cat((visual_feats, pos_feats), dim=-1)
        action_logits = self.imi_models(embeds)
        return action_logits

class TransformerRobotActor(torch.nn.Module):
    def __init__(self, vision_encoder, pos_encoder, imi_models, args):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.pos_encoder = pos_encoder
        self.imi_models = imi_models
        self.mha = MultiheadAttention(512, 8) # 128 if for one single image
        self.bottleneck = nn.Linear(2048, 512)
        self.use_two_cam = args.num_camera == 2
        self.use_convnext = args.use_convnext

    def forward(self, visual_input, pos_input, freeze):
                
        batch, frames, channel, H, W = visual_input.shape
        visual_input = visual_input.view(-1, 3, H, W) # (batch*frames, 3, H, W)

        if freeze:
            with torch.no_grad():
                visual_feats = self.vision_encoder(visual_input).detach()
                pos_feats = self.pos_encoder(pos_input).detach()
        else:
            visual_feats = self.vision_encoder(visual_input)
            pos_feats = self.pos_encoder(pos_input)
        #convert back to (batch, frames, 512)    
        visual_feats = visual_feats.view(batch, frames, -1)

        # #2 attention across different embeds
        # #stack two cam visual feat and pos feat, attention on these two feature
        mha_input = torch.stack([visual_feats[:, i, :] for i in range(frames)], 0) # (frames, batch, 512)
        mha_out, weights = self.mha(mha_input, mha_input, mha_input) #self attention, mhaout(frames, batch, 512)
        # [[0.1, 0.8], [0.9, 0.2]]
        mha_out += mha_input
        embeds = torch.concat([mha_out[i] for i in range(mha_out.shape[0])], 1) #(batch, 512*frames)
        embeds = self.bottleneck(embeds)
        action_logits = self.imi_models(embeds)
        return action_logits


class LSTMRobotActor(torch.nn.Module):
    def __init__(self, vision_encoder, pos_encoder, imi_models, args):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.pos_encoder = pos_encoder
        self.imi_models = imi_models
        self.bottleneck = nn.Linear(3072, 512)
        self.lstm = torch.nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)

    def forward(self, visual_input, pos_input, freeze, prev_hidden=None):
                
        batch, frames, channel, H, W = visual_input.shape
        visual_input = visual_input.view(-1, 3, H, W) # (batch*frames, 3, H, W)

        if freeze:
            with torch.no_grad():
                visual_feats = self.vision_encoder(visual_input).detach()
                pos_feats = self.pos_encoder(pos_input).detach()
        else:
            visual_feats = self.vision_encoder(visual_input)
            pos_feats = self.pos_encoder(pos_input)
        #convert back to (batch, frames, 512)    
        visual_feats = visual_feats.view(batch, frames, -1)

        lstm_inp = visual_feats
        mlp_inp, next_hidden = self.lstm(lstm_inp, prev_hidden)

        action_logits = self.imi_models(mlp_inp)
        return action_logits, next_hidden

class rnn_robotActor(torch.nn.Module):
    def __init__(self, vision_encoder, pos_encoder, imi_models, args):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.pos_encoder = pos_encoder
        self.imi_models = imi_models
        # self.mha = MultiheadAttention(128, 8) # 128 if for one single image
        self.bottleneck = nn.Linear(768, 512)
        self.two_cam_bottleneck = nn.Linear(1024, 512)
        self.use_two_cam = args.num_camera == 2
        self.use_convnext = args.use_convnext

    def forward(self, visual_input, pos_input, freeze):
        
        batch, T, channel, H, W = visual_input.shape
        visual_input = visual_input.view(-1, 3, H, W) # (2*N / N, 3/6, H, W)
        pos_input = pos_input.view(-1,3)
        if freeze:
            with torch.no_grad():
                visual_feats = self.vision_encoder(visual_input).detach()
                pos_feats = self.pos_encoder(pos_input).detach()
        else:
            visual_feats = self.vision_encoder(visual_input)
            pos_feats = self.pos_encoder(pos_input)

        visual_feats.view(batch, T, -1)
        pos_feats.view(batch, T, -1)
        #convert back to (batch, 2*embeds)    
        # visual_feats = visual_feats.view(batch, -1)
        if self.use_two_cam:
            visual_feats = self.two_cam_bottleneck(visual_feats) #(batch, 512)
        elif self.use_convnext:
            visual_feats = self.bottleneck(visual_feats)
        
        #0 attention in one cam with history
        #dataset can input history
        #mha_input = visual_feats.view(4, -1, 128)
        #mha_out, weights = self.mha(mha_input, mha_input, mha_input)
        #mha_out += mha_input
        #embeds = torch.concat([mha_out[i] for i in range(mha_out.shape[0])], 1)
        
        #1 attention in one single image
        #visual_feats -> (batch, 512) -> (4, batch, 128)
        # mha_input = torch.split(visual_feats,128, dim=-1)
        # mha_input=torch.stack(mha_input)
        # mha_out, weights = self.mha(mha_input, mha_input, mha_input)
        # mha_out += mha_input
        # embeds = torch.concat([mha_out[i] for i in range(mha_out.shape[0])], 1)
        
        # #2 attention across different embeds
        # #stack two cam visual feat and pos feat, attention on these two feature
        # mha_input = torch.stack((visual_feats, pos_feats, second_can_feats), 0) # (2, batch, 512)
        # mha_out, weights = self.mha(mha_input, mha_input, mha_input) #self attention, mhaout(2, batch, 512)
        # # [[0.1, 0.8], [0.9, 0.2]]
        # mha_out += mha_input
        # embeds = torch.concat([mha_out[i] for i in range(mha_out.shape[0])], 1) #(batch, 1024)
        


        embeds = torch.cat((visual_feats, pos_feats), dim=-1)
        action_logits = self.imi_models(embeds)
        return action_logits