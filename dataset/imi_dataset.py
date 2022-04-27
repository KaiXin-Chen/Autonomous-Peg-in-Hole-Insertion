import json
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import pandas as pd
from torchvision import transforms as T
from DatasetPlate import BaseDataset

class ImiDataset(BaseDataset):
    def __init__(self):
        super(self).__init__()
        self.trial, self.timestamps, self.episode_len = self.get_episode(0)

    def __len__(self):
        return self.episode_len

    def __getitem__(self, item):
        trans = T.Resize((100, 100))
        img = self.load_image(self.trial, "cam_fixed_color", 0)
        img = trans(img)
        action = self.timestamps["action_history"]
        action = action[:3]
        return img, action[0]