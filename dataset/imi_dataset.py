import json
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import pandas as pd
from torchvision import transforms as T
from dataset.DatasetPlate import BaseDataset

class ImiDatasetLabelCount(BaseDataset):
    def __init__(self, log_file, args, dataset_idx, data_folder=None):
        super().__init__(log_file, data_folder)
        self.trial, self.timestamps, self.num_frames = self.get_episode(
            dataset_idx)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        action = self.timestamps["action_history"][idx]
        xy_space = {-.0005: 0, 0: 1, .0005: 2}
        z_space = {-.0005: 0, 0: 1, .0005: 2}
        action = xy_space[action[0]] * 9 + xy_space[action[1]] * 3 + z_space[action[2]]
        return action

class ImiDataset(BaseDataset):
    def __init__(self, log_file, args, dataset_idx, data_folder="data/test_recordings_0214", train=True):
        super().__init__(log_file, data_folder)
        self.train = train
        self.num_cam = args.num_camera
        self.resized_height = args.resized_height
        self.resized_width = args.resized_width
        self._croped_height = int(args.resized_height * (1 - args.crop_per))
        self._croped_width = int(args.resized_width * (1 - args.crop_per))
        self.trial, self.timestamps, self.num_frames = self.get_episode(
            dataset_idx)
        self.pose_mean = torch.Tensor([0.47, -0.0195, 0.097])
        self.pose_var = torch.Tensor([0.005, 0.004, 0.004]) 

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        # timestep equals to the sampled idx in one episode
        timestep = idx
        
        if self.train:
            # resize the img data to specific size for memory efficiency
            transform = T.Compose([
                T.Resize((self.resized_height, self.resized_width)),
                T.RandomCrop((self._croped_height, self._croped_width)),
                T.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.2)
            ])
            # load third view cam image
            fixed_cam = transform(self.load_image(self.trial, "cam_fixed_color", timestep))
            # if use two cam, load gripper view image
            if self.num_cam == 2:
                gripper_cam = transform(self.load_image(self.trial, "cam_fixed_color", timestep))
        else:
            # during testing, the processing is the same for now, but later might be different
            # resize the img data to specific size for memory efficiency
            transform = T.Compose([
                T.Resize((self.resized_height, self.resized_width)),
                T.CenterCrop((self._croped_height, self._croped_width))                
            ])
            # load third view cam image
            fixed_cam = transform(self.load_image(self.trial, "cam_fixed_color", timestep))
            # if use two cam, load gripper view image
            if self.num_cam == 2:
                gripper_cam = transform(self.load_image(self.trial, "cam_fixed_color", timestep))

        if self.num_cam == 2:
            # if use two cam, concat them in the rgb channel
            v_input = torch.cat((fixed_cam, gripper_cam), dim=0) # (6, H, W)
        else:
            v_input = fixed_cam

        # load the action at this step
        action = self.timestamps["action_history"][timestep]
        # load position history at this step
        pos = self.timestamps["pose_history"][timestep][:3]
        # map the action from robot coordinate steps to 0, 1, 2 discrete actions
        # 0 means increase one unit distance
        # 1 means stay still
        # 0 means decrease one unit distance
        xy_space = {-.0005: 0, 0: 1, .0005: 2}
        z_space = {-.0005: 0, 0: 1, .0005: 2}
        action = torch.as_tensor(
            [xy_space[action[0]], xy_space[action[1]], z_space[action[2]]])
        pose = torch.as_tensor([pos[0], pos[1], pos[2]]) - self.pose_mean
        pose /= self.pose_var
        # finally return visual image of [..., 3, H, W] and action of size [..., 3]
        return v_input, action, pose