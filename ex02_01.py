from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from sklearn import metrics
import numpy as np
from shared.logger import get_logger, get_filename, get_start_time, get_curr_time
from shared.transform import flatten, scale
from shared.metrics import f1_score
from shared.plot_history import plot_all
from shared.utility import open_plot, ensure_directory, get_tensorboard_directory, get_model_file

PROJECT_NAME="mnist"
OUTPUT_DIR="output/ex02"
TBLOGDIR=get_tensorboard_directory(PROJECT_NAME)
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
history = model.fit(x=train_images, 
                    y=train_labels, 
                    epochs=5, 
                    batch_size=128,
                    callbacks=[TensorBoard(log_dir=TBLOGDIR)])

# test
logger.info("model evaluate")
test_loss, test_acc, test_f1 = model.evaluate(x=test_images, y=test_labels)
logger.info("Test loss: {} Test accuracy: {} Test F1: {}".format(test_loss, test_acc, test_f1))

logger.info("sklearn evaluate")
y_test_pred = np.round(model.predict(test_images))
logger.info("sklearn f1 scores: {}".format(metrics.f1_score(test_labels, y_test_pred, average=None)))
logger.info("classification report:\n{}".format(metrics.classification_report(test_labels, y_test_pred)))

logger.info("finished at {}".format(get_curr_time()))
