from keras.datasets import mnist
from keras import models
from keras import layers
from keras import regularizers
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from shared.logger import get_logger, get_filename, get_start_time, get_curr_time
from shared.transform import flatten, scale, add_channel
from shared.utility import ensure_directory
import numpy as np

logger = get_logger()

OUTPUT_DIR="output/ae02"
ensure_directory(OUTPUT_DIR, logger)
ENCODING_SHAPE = (128,)
EPOCHS = 100
BATCH_SIZE = 128
ACTIVITY_REG = 10e-7
TBLOGDIR='/tmp/autoencoder'

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# preprocess
train_images = scale(add_channel(train_images))
test_images = scale(add_channel(test_images))

input_img = layers.Input(shape=train_images[0].shape)

l1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
l1 = layers.MaxPooling2D((2, 2), padding='same')(l1)

l2 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(l1)
l2 = layers.MaxPooling2D((2, 2), padding='same')(l2)


l3 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(l2)
encoded = layers.MaxPooling2D((2, 2), padding='same')(l3)


l4 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
l4 = layers.UpSampling2D((2, 2))(l4)

l5 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(l4)
l5 = layers.UpSampling2D((2, 2))(l5)

l6 = layers.Conv2D(16, (3, 3), activation='sigmoid')(l5)
l6 = layers.UpSampling2D((2, 2))(l6)

decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(l6)

# whole autoencoder
autoencoder = models.Model(input_img, decoded)

# build
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# look at model architecture
logger.debug("Autoencoder: {}\n".format(autoencoder.summary()))

#fit
logger.info("Fitting autoencoder...")
history = autoencoder.fit(train_images,
                train_images,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                validation_data=(test_images, test_images),
                callbacks=[TensorBoard(log_dir=TBLOGDIR)])

# look at summaries
logger.info("Autoencoder loss: {}".format(history.history['loss'][-1]))
