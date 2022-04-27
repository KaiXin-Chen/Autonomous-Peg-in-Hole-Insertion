import json
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import pandas as pd

class BaseDataset(Dataset):
    def __init__(self, log_file, data_folder="data/test_recordings_0214"):
        """
        This is a template for all other data sets
        """
        super().__init__()
        self.logs = pd.read_csv(log_file)
        self.data_folder = data_folder
        self.streams = ["cam_gripper_color", "cam_fixed_color"]

    def get_episode(self, idx):
        """
        Return:
            Get the episodes for corresponding index
        """
        format_time = self.logs.iloc[idx].Time.replace(":", "_")
        trial = os.path.join(self.data_folder, format_time)
        with open(os.path.join(trial, "timestamps.json")) as ts:
            self.timestamps = json.load(ts)
        return trial, self.timestamps,  len(self.timestamps["action_history"])

    def __len__(self):
        """
        :return:
            Should return the length of the current episode
        """
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    @staticmethod
    def load_image(trial, stream, timestep):
        """
        Args:
            trial: the folder of the current episode
            stream: ["cam_gripper_color", "cam_fixed_color"]
            timestep: the timestep of frame you want to extract
        """
        img_path = os.path.join(trial, stream, str(timestep) + ".png")
        image = torch.as_tensor(np.array(Image.open(img_path))).float().permute(2, 0, 1) / 255
        return image

