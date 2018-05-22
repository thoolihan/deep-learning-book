from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from shared.logger import get_logger, get_filename, get_start_time, get_curr_time
from shared.transform import flatten, scale
from shared.metrics import f1_score
from shared.plot_history import plot_all
from shared.utility import open_plot, ensure_directory

OUTPUT_DIR="output/ae02"
ensure_directory(OUTPUT_DIR)
ENCODING_DIM = 32
ENCODING_SHAPE = (ENCODING_DIM,)
INPUT_DIM = 784
INPUT_SHAPE = (INPUT_DIM,)
EPOCHS = 50
BATCH_SIZE = 256

logger = get_logger()

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# preprocess
train_images = scale(flatten(train_images))
test_images = scale(flatten(test_images))

input_img = layers.Input(shape=INPUT_SHAPE)
encoded = layers.Dense(ENCODING_DIM, activation='relu')(input_img)
_decoded = layers.Dense(INPUT_DIM, activation='sigmoid')
decoded = _decoded(encoded)

# whole autoencoder
autoencoder = models.Model(input_img, decoded)

# encoder
encoder = models.Model(input_img, encoded)

# decoder
encoded_input = layers.Input(shape = ENCODING_SHAPE)
decoder = models.Model(encoded_input, _decoded(encoded_input))

# build
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#fit
autoencoder.fit(train_images,
                train_images,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                validation_data=(test_images, test_images))

encoded_imgs = encoder.predict(test_images)
decoded_imgs = decoder.predict(encoded_imgs)

import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test_images[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


