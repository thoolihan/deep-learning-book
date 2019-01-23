### This version hydrates 5_07, uses it on the test set

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.models import load_model
from shared.logger import get_logger, get_filename, get_start_time, get_curr_time
from shared.metrics import f1_score
from shared.utility import open_plot, ensure_directory, get_tensorboard_directory, get_model_file

logger = get_logger()

# Constants and Config for index, features, and label
IMG_BATCH_SIZE=20
MODEL_TYPE="ex05_07"
MODEL_VERSION="2019.01.23.13.12.52"

PROJECT_NAME="dogs_cats"
INPUT_DIR="data/{}".format(PROJECT_NAME)
OUTPUT_DIR="output/{}".format(PROJECT_NAME)
ensure_directory(OUTPUT_DIR, logger)

MODEL_FILE = get_model_file(OUTPUT_DIR, fname=MODEL_TYPE, ts=MODEL_VERSION)

test_dir = os.path.join(INPUT_DIR, "test")

# load data
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=IMG_BATCH_SIZE,
    class_mode='binary')

model = load_model(MODEL_FILE, custom_objects={'f1_score': f1_score})

model.summary()

test_loss, test_acc, test_f1 = model.evaluate_generator(test_generator, steps=50)
print("test accuracy: {}".format(test_acc))
print("test loss: {}".format(test_loss))
print("test f1: {}".format(test_f1))