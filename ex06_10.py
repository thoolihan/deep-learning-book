import os
import tensorflow as tf
import numpy as np
from scipy import stats
from sklearn import metrics
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Flatten, Dense
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb

from shared.logger import get_logger
from shared.metrics import f1_score
from shared.utility import ensure_directory, get_tensorboard_directory, get_model_file, limit_gpu_memory

limit_gpu_memory()
logger = get_logger()

# Constants and Config for index, features, and label
PROJECT_NAME = "imdb-lstm"
INPUT_DIR = os.path.join("data", PROJECT_NAME)
OUTPUT_DIR = os.path.join("output", PROJECT_NAME)
GLOVE_HOME = os.path.join(os.path.expanduser("~"), "workspace", "Embeddings", "glove")

EMBEDDINGS_DIMENSIONS = 100
EMBEDDINGS_FILE_NAME = "glove.6B.{}d.txt".format(EMBEDDINGS_DIMENSIONS)
EMBEDDINGS_PATH = os.path.join(GLOVE_HOME, EMBEDDINGS_FILE_NAME)
ensure_directory(OUTPUT_DIR, logger)
MODEL_FILE = get_model_file(OUTPUT_DIR)
SAVE_MODEL = True

EPOCHS = 10
BATCH_SIZE = 128
NUM_WORDS = 10000
DIM = 32
MAX_LEN = 500
TRAIN_SIZE = 200
VAL_SIZE = 10000

TBLOGDIR=get_tensorboard_directory(PROJECT_NAME)
logger.info("Tensorboard is at: {}".format(TBLOGDIR))

logger.info("Loading data...")
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words = NUM_WORDS)
logger.debug("input_train shape: {}".format(input_train.shape))
logger.debug("stats of original input_train lengths:\n{}".format(stats.describe([len(x) for x in input_train])))

input_train = sequence.pad_sequences(input_train, maxlen=MAX_LEN)
input_test = sequence.pad_sequences(input_test, maxlen=MAX_LEN)
logger.debug("stats of original input_train lengths:\n{}".format(stats.describe([len(x) for x in input_train])))

model = Sequential()
model.add(Embedding(NUM_WORDS, DIM))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
logger.debug(model.summary())

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc', f1_score])
history = model.fit(input_train,
                   y_train,
                   epochs=EPOCHS,
                   batch_size=BATCH_SIZE,
                   validation_split=0.2,
                   callbacks=[TensorBoard(log_dir=TBLOGDIR)])

logger.info("Running model.evaluate on test set...")
model.evaluate(input_test, y_test)

if SAVE_MODEL:
    model.save_weights(MODEL_FILE)
    logger.info("model weights saved at: {}".format(MODEL_FILE))
else:
    logger.info("did NOT save model weights at {}, change flag if you meant to".format(MODEL_FILE))
    