import os
import numpy as np
from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import TensorBoard
from shared.logger import get_logger, get_filename, get_start_time, get_curr_time
from shared.metrics import f1_score
from shared.utility import open_plot, ensure_directory, get_tensorboard_directory, get_model_file

logger = get_logger()

model = VGG16(weights='imagenet')

# Constants and Config for index, features, and label
PROJECT_NAME="lion"
INPUT_DIR="data/{}".format(PROJECT_NAME)
OUTPUT_DIR="output/{}".format(PROJECT_NAME)
ensure_directory(OUTPUT_DIR, logger)
MODEL_FILE = get_model_file(OUTPUT_DIR)
TBLOGDIR=get_tensorboard_directory(PROJECT_NAME)
logger.info("Tensorboard is at: {}".format(TBLOGDIR))

img_path = '{}/white_lion.jpg'.format(INPUT_DIR)
img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])