from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from shared.logger import get_logger, get_filename, get_start_time, get_curr_time
from shared.transform import flatten, scale
from shared.metrics import f1_score
from shared.plot_history import plot_all
from shared.utility import open_plot, ensure_directory

OUTPUT_DIR="output/ex02"
logger = get_logger()
ensure_directory(OUTPUT_DIR, logger)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# preprocess
train_images = scale(flatten(train_images))
test_images = scale(flatten(test_images))

# to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# nn
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer = 'rmsprop',
               loss = 'categorical_crossentropy',
               metrics = ['accuracy', f1_score])

# fit
history = model.fit(x=train_images, y=train_labels, epochs=5, batch_size=128)

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