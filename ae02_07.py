from keras.datasets import mnist
from keras import models
from keras import layers
from keras import regularizers
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from shared.logger import get_logger, get_filename, get_start_time, get_curr_time
from shared.transform import flatten, scale, add_channel
from shared.metrics import f1_score
from shared.plot_history import plot_all
from shared.utility import open_plot, ensure_directory
import numpy as np

logger = get_logger()

OUTPUT_DIR="output/ae02"
ensure_directory(OUTPUT_DIR, logger)
ENCODING_SHAPE = (128,)
EPOCHS = 100
BATCH_SIZE = 128
ACTIVITY_REG = 10e-7
TBLOGDIR="/tmp/autoencoder/{}".format(get_start_time())

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# preprocess
train_images = scale(add_channel(train_images))
test_images = scale(add_channel(test_images))

NOISE_FACTOR = .5
train_images_noise = train_images + NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=train_images.shape)
test_images_noise = test_images + NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=test_images.shape)
train_images_noise = np.clip(train_images_noise, 0.0, 1.0)
test_images_noise = np.clip(test_images_noise, 0.0, 1.0)

logger.info("training data shape: {}".format(train_images.shape))

input_img = layers.Input(shape=train_images[0].shape)

d1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
d1 = layers.MaxPooling2D((2, 2), padding='same')(d1)

d2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(d1)
encoded = layers.MaxPooling2D((2, 2), padding='same')(d2)


u1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
u1 = layers.UpSampling2D((2, 2))(u1)

u2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u1)
u2 = layers.UpSampling2D((2, 2))(u2)

decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(u2)

# whole autoencoder
autoencoder = models.Model(input_img, decoded)
encoder = models.Model(input_img, encoded)

# build
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# look at model architecture
logger.debug("Autoencoder: {}\n".format(autoencoder.summary()))

#fit
logger.info("Fitting autoencoder...")
logger.info("Tensorboard Directory is {}".format(TBLOGDIR))
history = autoencoder.fit(train_images_noise,
                train_images,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                validation_data=(test_images_noise, test_images),
                callbacks=[TensorBoard(log_dir=TBLOGDIR)])

# look at summaries
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
    plt.imshow(test_images_noise[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display convolved version
    ax = plt.subplot(ROWS, DIGITS, i + 1 + DIGITS)
    plt.imshow(encoded_imgs[i].reshape(7, 7 * 32).T)
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
