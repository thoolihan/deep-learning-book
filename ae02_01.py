from keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from shared.logger import get_logger, get_filename, get_start_time, get_curr_time
from shared.transform import flatten, scale
from shared.metrics import f1_score
from shared.plot_history import plot_all
from shared.utility import open_plot, ensure_directory, limit_gpu_memory
import numpy as np
import os

logger = get_logger()
limit_gpu_memory()

OUTPUT_DIR=os.path.join("output", "ae02")
ensure_directory(OUTPUT_DIR, logger)
ENCODING_DIM = 36
ENCODING_SHAPE = (ENCODING_DIM,)
INPUT_DIM = 784
INPUT_SHAPE = (INPUT_DIM,)
EPOCHS = 25
BATCH_SIZE = 256

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# preprocess
train_images = scale(flatten(train_images))
test_images = scale(flatten(test_images))

input_img = layers.Input(shape=INPUT_SHAPE)
encoded = layers.Dense(ENCODING_DIM, activation='relu')(input_img)
decoded = layers.Dense(INPUT_DIM, activation='sigmoid')(encoded)

# whole autoencoder
autoencoder = models.Model(input_img, decoded)

# encoder
encoder = models.Model(input_img, encoded)

# decoder
encoded_input = layers.Input(shape = ENCODING_SHAPE)

# build
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#fit
logger.info("Fitting autoencoder...")
history = autoencoder.fit(train_images,
                train_images,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                validation_data=(test_images, test_images))

# look at summaries
logger.debug("Encoder: {}\n".format(encoder.summary()))
logger.debug("Autoencoder: {}\n".format(autoencoder.summary()))

logger.info("Autoencoder loss: {}".format(history.history['loss'][-1]))

encoded_imgs = encoder.predict(test_images)
decoded_imgs = autoencoder.predict(test_images)

logger.info("encoded_imgs.mean() is {}".format(encoded_imgs.mean()))

import matplotlib.pyplot as plt

DIGITS = 10  # how many digits we will display
ROWS = 3
plt.figure(figsize=(20, 4))
for i in range(DIGITS):
    # display original
    ax = plt.subplot(ROWS, DIGITS, i + 1)
    plt.imshow(test_images[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display compressed
    ax = plt.subplot(ROWS, DIGITS, i + 1 + DIGITS)
    side_size = int(np.sqrt(ENCODING_DIM))
    plt.imshow(encoded_imgs[i].reshape(side_size, side_size))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(ROWS, DIGITS, i + 1 + (2 * DIGITS))
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

pname = "{}/{}-{}.png".format(OUTPUT_DIR, get_filename(), get_start_time())
plt.savefig(pname)
logger.info("Created {}".format(pname))

open_plot(pname)
