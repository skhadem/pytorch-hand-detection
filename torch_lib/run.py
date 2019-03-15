"""
Main file for running
"""
import os
import numpy as np
import cv2
import argparse

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
