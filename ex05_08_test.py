### This version takes 5_06, and after training, unfreezes the 
### last layer in the conv_base in order to fine-tune

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras import models, layers, optimizers, load_model
from shared.logger import get_logger, get_filename, get_start_time, get_curr_time
from shared.metrics import f1_score
from shared.utility import open_plot, ensure_directory, get_tensorboard_directory, get_model_file

logger = get_logger()

# Constants and Config for index, features, and label
PROJECT_NAME="dogs_cats"
INPUT_DIR="data/{}".format(PROJECT_NAME)
OUTPUT_DIR="output/{}".format(PROJECT_NAME)
ensure_directory(OUTPUT_DIR, logger)

MODEL_FILE = get_model_file(OUTPUT_DIR, fname="ex05_08", ts="")

test_dir = os.path.join(INPUT_DIR, "test")

# load data
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=IMG_BATCH_SIZE,
    class_mode='binary')

model.load(MODEL_FILE)

model.summary()

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print("test accuracy: {}".format(test_acc))
print("test loss: {}".format(test_loss))