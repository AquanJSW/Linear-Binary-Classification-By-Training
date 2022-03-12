import numpy as np
import random

def randfloat(left, right):
    if left >= right:
        raise ValueError
    return random.random() * (right - left) + left


def onehot2scalar(onehot: np.ndarray):
    return np.nonzero(onehot)[0]
