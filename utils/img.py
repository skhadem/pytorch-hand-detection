import numpy as np

def get_size(img):
    if len(img.shape) == 3:
        return np.array(img.shape[:-1][::-1])
    else:
        return np.array(img.shape[::-1])
