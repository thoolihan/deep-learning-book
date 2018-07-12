import numpy as np

def flatten(data):
    i, h, w = data.shape
    return data.reshape(i, h * w)

def scale(data):
    return data.astype('float32') / 255.

def add_channel(data):
    return np.expand_dims(data, -1)