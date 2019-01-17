import os
import numpy as np
from functools import reduce
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import TensorBoard
from shared.logger import get_logger, get_filename, get_start_time, get_curr_time
from shared.metrics import f1_score
from shared.utility import open_plot, ensure_directory, get_tensorboard_directory, get_model_file

logger = get_logger()

# Constants and Config for index, features, and label
PROJECT_NAME="dogs_cats"
INPUT_DIR="data/{}".format(PROJECT_NAME)
OUTPUT_DIR="output/{}".format(PROJECT_NAME)
ensure_directory(OUTPUT_DIR, logger)
MODEL_FILE = get_model_file(OUTPUT_DIR)
TBLOGDIR=get_tensorboard_directory(PROJECT_NAME)
logger.info("Tensorboard is at: {}".format(TBLOGDIR))

DRO = 0.5
LR = 2e-5
HLAYER = 256
EPOCHS = 30
IMG_BATCH_SIZE=20
TRAIN_SIZE = 2000
VAL_SIZE = 1000
TEST_SIZE = 1000
VGG_SHAPE = (4, 4, 512)
VGG_FLAT = reduce(lambda x, y: x * y, VGG_SHAPE)
SAVE_MODEL = False

train_dir = os.path.join(INPUT_DIR, "train")
validation_dir = os.path.join(INPUT_DIR, "validation")
test_dir = os.path.join(INPUT_DIR, "test")

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

logger.debug("early exit, to be removed later...")
quit()





model.compile(optimizer=optimizers.RMSprop(LR),
              loss='binary_crossentropy',
              metrics=['acc', f1_score])

logger.info("Fitting network...")
history = model.fit(train_features, train_labels,
                    epochs=EPOCHS,
                    batch_size=IMG_BATCH_SIZE,
                    validation_data=(validation_features, validation_labels),
                    callbacks=[TensorBoard(log_dir=TBLOGDIR)])

if SAVE_MODEL:
    model.save(MODEL_FILE)
    logger.info("model saved at: {}".format(MODEL_FILE))
else:
    logger.info("did NOT save {}, change flag if you meant to".format(MODEL_FILE))
