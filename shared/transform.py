def flatten(data):
    i, h, w = data.shape
    return data.reshape(i, h * w)

def scale(data):
    return data.astype('float32') / 255.