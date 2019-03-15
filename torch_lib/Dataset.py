import cv2
import numpy as np
import os
import random

import torch
from torchvision import transforms
from torch.utils import data


"""
This class is meant to hold a large amount of data to iterate through when
training. It subclasses pytorch's dataset in order to take advantage of a bunch
of cool multithreading
"""
class Dataset(data.Dataset):
    def __init__(self):
        pass
    """
    returns length
    """
    def __len__(self):
        pass
    """
    Returns (input, label) to feed to net
    """
    def __getitem__(self, index):
        pass
