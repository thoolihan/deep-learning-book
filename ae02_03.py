from keras.datasets import mnist
from keras import models
from keras import layers
from keras import regularizers
from keras.utils import to_categorical
from shared.logger import get_logger, get_filename, get_start_time, get_curr_time
from shared.transform import flatten, scale
from shared.metrics import f1_score
from shared.plot_history import plot_all
from shared.utility import open_plot, ensure_directory
import numpy as np

logger = get_logger()

OUTPUT_DIR="output/ae02"
ensure_directory(OUTPUT_DIR, logger)
ENCODING_DIM = [128, 64, 36]
ENCODING_SHAPE = (ENCODING_DIM[2],)
INPUT_DIM = 784
INPUT_SHAPE = (INPUT_DIM,)
EPOCHS = 100
BATCH_SIZE = 256
ACTIVITY_REG = 10e-7

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# preprocess
train_images = scale(flatten(train_images))
test_images = scale(flatten(test_images))

input_img = layers.Input(shape=INPUT_SHAPE)
encoded = layers.Dense(ENCODING_DIM[0], activation='relu')(input_img)
encoded = layers.Dense(ENCODING_DIM[1], activation='relu')(encoded)
encoded = layers.Dense(ENCODING_DIM[2], activation='relu')(encoded)

decoded = layers.Dense(ENCODING_DIM[1], activation='relu')(encoded)
decoded = layers.Dense(ENCODING_DIM[0], activation='relu')(decoded)
decoded = layers.Dense(INPUT_DIM, activation='sigmoid')(decoded)

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
    side_size = int(np.sqrt(ENCODING_DIM[-1]))
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
