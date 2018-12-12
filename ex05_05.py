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

datagen = ImageDataGenerator(rescale=1./255)

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=IMG_BATCH_SIZE,
        class_mode='binary')
    i = 0
    for input_batch, labels_batch in generator:
        features_batch = conv_base.predict(input_batch)
        features[i * IMG_BATCH_SIZE : (i + 1) * IMG_BATCH_SIZE] = features_batch
        labels[i * IMG_BATCH_SIZE : (i + 1) * IMG_BATCH_SIZE] = labels_batch
        i += 1
        if i * IMG_BATCH_SIZE >= sample_count:
            break
    return features, labels

logger.info("Extracting Features (using prebuilt VGG16)...")
train_features, train_labels = extract_features(train_dir, TRAIN_SIZE)
validation_features, validation_labels = extract_features(validation_dir, VAL_SIZE)
test_features, test_labels = extract_features(test_dir, TEST_SIZE)

logger.info("Reshaping Tensors (flatten extracted output)...")
train_features = np.reshape(train_features, (TRAIN_SIZE, VGG_FLAT))
validation_features = np.reshape(validation_features, (VAL_SIZE, VGG_FLAT))
test_features = np.reshape(test_features, (TEST_SIZE, VGG_FLAT))

logger.info("Creating NN...")
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=VGG_FLAT))
model.add(layers.Dropout(DRO))
model.add(layers.Dense(1, activation='sigmoid'))

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
