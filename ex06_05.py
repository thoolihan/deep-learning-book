import os
import numpy as np
from tensorflow.keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from shared.logger import get_logger
from shared.metrics import f1_score
from tensorflow.keras.callbacks import TensorBoard
from shared.utility import ensure_directory, get_tensorboard_directory, get_model_file

logger = get_logger()

# Constants and Config for index, features, and label
PROJECT_NAME="imdb"
INPUT_DIR = os.path.join("data", PROJECT_NAME)
OUTPUT_DIR = os.path.join("output", PROJECT_NAME)
ensure_directory(OUTPUT_DIR, logger)
MODEL_FILE = get_model_file(OUTPUT_DIR)
EPOCHS = 30
BATCH_SIZE=64
NUM_WORDS = 10000
MAX_LEN = 100
TRAIN_SIZE = 200
TBLOGDIR=get_tensorboard_directory(PROJECT_NAME)
logger.info("Tensorboard is at: {}".format(TBLOGDIR))

TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(TRAIN_DIR, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            with open(os.path.join(dir_name, fname)) as f:
                texts.append(f.read())
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)
            
tkn = Tokenizer(num_words = NUM_WORDS)
tkn.fit_on_texts(texts)
sequences = tkn.texts_to_sequences(texts)

word_index = tkn.word_index
logger.info("Found {} unique tokens".format(len(word_index)))
