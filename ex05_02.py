from keras import models, layers, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from shared.logger import get_logger, get_filename, get_start_time, get_curr_time
from shared.transform import flatten, scale
from shared.metrics import f1_score
from shared.plot_history import plot_all
from shared.utility import open_plot, ensure_directory
import os

logger = get_logger()

# Constants and Config for index, features, and label
PROJECT_NAME="dogs_cats"
INPUT_DIR="data/{}".format(PROJECT_NAME)
OUTPUT_DIR="output/{}".format(PROJECT_NAME)
TBLOGDIR='/tmp/tensorboard'
ensure_directory(OUTPUT_DIR, logger)
DRO = 0.25
LR = 1e-4
HLAYER = 64
EPOCHS = 1
IMG_BATCH_SIZE=20
BATCH_SIZE = 64
img_shape = (150, 150, 3)
train_dir = "{}/train".format(INPUT_DIR)
validation_dir = "{}/validation".format(INPUT_DIR)

# load data
train_datagen = ImageDataGenerator(rescale=1./255)
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
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=LR),
              metrics=['acc', f1_score])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=10,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=50,
    callbacks=[TensorBoard(log_dir=TBLOGDIR)])

model.save(os.path.join(OUTPUT_DIR, "model-{}.h5".format(get_start_time())))

