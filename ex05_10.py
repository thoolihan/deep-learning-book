import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from shared.utility import get_model_file
from shared.metrics import f1_score

PROJECT_NAME="dogs_cats"
INPUT_DIR="data/{}".format(PROJECT_NAME)
OUTPUT_DIR="output/{}".format(PROJECT_NAME)
TEST_DIR = os.path.join(INPUT_DIR, "test")

MODEL_FILE = get_model_file(OUTPUT_DIR, fname="ex05_02", ts="2019.01.27.14.36.31")
model = load_model(MODEL_FILE, custom_objects={'f1_score': f1_score})
model.summary()

IMG_PATH = os.path.join(TEST_DIR, 'cats', 'cat.1700.jpg')

img = image.load_img(IMG_PATH, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

print(img_tensor.shape)
plt.imshow(img_tensor[0])
plt.show()

