### This version takes 5_06, and after training, unfreezes the 
### last layer in the conv_base in order to fine-tune

import os
import numpy as np
from functools import reduce
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
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

LR = 2e-5
HLAYER = 256
EPOCHS = 30
EPOCHS_FINETUNE = 100
IMG_BATCH_SIZE=20
SAVE_MODEL = True

train_dir = os.path.join(INPUT_DIR, "train")
validation_dir = os.path.join(INPUT_DIR, "validation")
test_dir = os.path.join(INPUT_DIR, "test")

# load data
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=IMG_BATCH_SIZE,
    class_mode='binary')

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=IMG_BATCH_SIZE,
    class_mode='binary')

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

logger.info("Trainable weights before freeze: {}".format(len(model.trainable_weights)))
conv_base.trainable = False
logger.info("Trainable weights after freeze: {}".format(len(model.trainable_weights)))

model.summary()

model.compile(optimizer=optimizers.RMSprop(lr=LR),
              loss='binary_crossentropy',
              metrics=['acc', f1_score])

logger.info("Fitting network...")
history = model.fit_generator(train_generator,
                              steps_per_epoch=int(train_generator.n / IMG_BATCH_SIZE),
                              epochs=EPOCHS,
                              validation_data=validation_generator,
                              validation_steps=int(validation_generator.n / IMG_BATCH_SIZE),
                              callbacks=[TensorBoard(log_dir=TBLOGDIR)])

logger.info("Unfreezing the last layer of the conv_base (VGG16)...")
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
        
logger.info("Fitting for another 100 epochs with a lower LR...")
model.compile(optimizer=optimizers.RMSprop(lr=LR/2.0),
              loss='binary_crossentropy',
              metrics=['acc', f1_score])

history = model.fit_generator(train_generator,
                              steps_per_epoch=int(train_generator.n / IMG_BATCH_SIZE),
                              epochs=EPOCHS_FINETUNE,
                              validation_data=validation_generator,
                              validation_steps=int(validation_generator.n / IMG_BATCH_SIZE),
                              callbacks=[TensorBoard(log_dir=TBLOGDIR)])

    
if SAVE_MODEL:
    model.save(MODEL_FILE)
    logger.info("model saved at: {}".format(MODEL_FILE))
else:
    logger.info("did NOT save {}, change flag if you meant to".format(MODEL_FILE))
