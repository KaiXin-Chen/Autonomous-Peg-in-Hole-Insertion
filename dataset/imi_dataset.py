import json
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import pandas as pd
from DatasetPlate import BaseDataset

class ImiDataset(BaseDataset):
    def __init__(self):
        super(self).__init__()

    def __len__(self):
        return None

    def __getitem__(self, item):
        return None