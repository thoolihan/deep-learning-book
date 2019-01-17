### From scratch CNN on dog and cat pics, plus augmentation to avoid overfitting

from keras import models, layers, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from shared.logger import get_logger, get_filename, get_start_time, get_curr_time
from shared.metrics import f1_score
from shared.utility import open_plot, ensure_directory, get_tensorboard_directory, get_model_file
import os

logger = get_logger()

# Constants and Config for index, features, and label
PROJECT_NAME="dogs_cats"
INPUT_DIR="data/{}".format(PROJECT_NAME)
OUTPUT_DIR="output/{}".format(PROJECT_NAME)
ensure_directory(OUTPUT_DIR, logger)
MODEL_FILE = get_model_file(OUTPUT_DIR)
TBLOGDIR=get_tensorboard_directory(PROJECT_NAME)

DRO = 0.5
LR = 1e-4
HLAYER = 64
EPOCHS = 30
IMG_BATCH_SIZE=20
img_shape = (150, 150, 3)
SAVE_MODEL = False
train_dir = "{}/train".format(INPUT_DIR)
validation_dir = "{}/validation".format(INPUT_DIR)

# load data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=IMG_BATCH_SIZE,
    class_mode='binary')

validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=IMG_BATCH_SIZE,
    class_mode='binary')


# build model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=img_shape))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dropout(DRO))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=LR),
              metrics=['acc', f1_score])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=50,
    callbacks=[TensorBoard(log_dir=TBLOGDIR)])

if SAVE_MODEL:
    model.save(MODEL_FILE)
    logger.info("model saved at: {}".format(MODEL_FILE))
else:
    logger.info("did NOT save {}, change flag if you meant to".format(MODEL_FILE))
