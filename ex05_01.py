from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from shared.logger import get_logger, get_filename, get_start_time, get_curr_time
from shared.transform import flatten, scale
from shared.metrics import f1_score
from shared.plot_history import plot_all
from shared.utility import open_plot, ensure_directory

logger = get_logger()

# Constants and Config for index, features, and label
PROJECT_NAME="ex05"
INPUT_DIR="data/{}".format(PROJECT_NAME)
OUTPUT_DIR="output/{}".format(PROJECT_NAME)
ensure_directory(OUTPUT_DIR, logger)
DRO = 0.25
HLAYER = 64
EPOCHS = 5
BATCH_SIZE = 64

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# preprocess
train_images = scale(train_images.reshape(train_images.shape[0], 28, 28, 1))
test_images = scale(test_images.reshape(test_images.shape[0], 28, 28, 1))

# to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# nn
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(HLAYER, activation='relu'))
model.add(layers.Dropout(DRO))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer = 'rmsprop',
               loss = 'categorical_crossentropy',
               metrics = ['accuracy', f1_score])

# fit
history = model.fit(x=train_images, y=train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE)

# summary
logger.info(model.summary())

# test
test_loss, test_acc, test_f1 = model.evaluate(x=test_images, y=test_labels)
logger.info("Test loss: {} Test accuracy: {} Test F1: {}".format(test_loss, test_acc, test_f1))

logger.info("saving plot of loss and accuracy")
plots = plot_all(history, {'acc': 'Accuracy', 'f1_score': 'F-score'})
pname = "{}/{}-{}.png".format(OUTPUT_DIR, get_filename(), get_start_time())
plots.savefig(pname)
logger.info("created {}".format(pname))
logger.info("finished at {}".format(get_curr_time()))

open_plot(pname)