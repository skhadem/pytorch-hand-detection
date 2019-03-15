"""
Main training script
"""

import os
import numpy as np
import cv2
import argparse

import torch
from torch.autograd import Variable
from torchvision.models import vgg

from torch_lib.Dataset import Dataset
from torch_lib.Model import Model

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Expected a bool')

parser = argparse.ArgumentParser()

def main():
    print("Hello")

if __name__ == '__main__':
    main()
