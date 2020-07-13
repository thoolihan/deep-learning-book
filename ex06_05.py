import os
import numpy as np
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from shared.logger import get_logger
from shared.metrics import f1_score
from tensorflow.keras.callbacks import TensorBoard
from shared.utility import ensure_directory, get_tensorboard_directory, get_model_file, get_config_value

logger = get_logger()

# Constants and Config for index, features, and label
PROJECT_NAME = "imdb"
INPUT_DIR = os.path.join("data", PROJECT_NAME)
OUTPUT_DIR = os.path.join("output", PROJECT_NAME)
GLOVE_HOME = os.path.join(*get_config_value("glove_dir"))
EMBEDDINGS_DIMENSIONS = 100
EMBEDDINGS_FILE_NAME = "glove.6B.{}d.txt".format(EMBEDDINGS_DIMENSIONS)
EMBEDDINGS_PATH = os.path.join(GLOVE_HOME, EMBEDDINGS_FILE_NAME)
if os.path.exists(EMBEDDINGS_PATH):
    logger.info("Confirmed embeddings exist at: {}".format(EMBEDDINGS_PATH))
    logger.info("File Size: {:.1f} MB".format(os.path.getsize(EMBEDDINGS_PATH) / 1024 ** 2))
else:
    logger.error("Embeddings do not exist at: {}".format(EMBEDDINGS_PATH))
    exit(1)
ensure_directory(OUTPUT_DIR, logger)
MODEL_FILE = get_model_file(OUTPUT_DIR)
EPOCHS = 10
BATCH_SIZE=32
NUM_WORDS = 10000
MAX_LEN = 100
TRAIN_SIZE = 200
VAL_SIZE = 10000
TBLOGDIR = get_tensorboard_directory(PROJECT_NAME)
SAVE_MODEL = True
logger.info("Tensorboard is at: {}".format(TBLOGDIR))

TRAIN_DIR = os.path.join(INPUT_DIR, "train")
ensure_directory(TRAIN_DIR, logger)
TEST_DIR = os.path.join(INPUT_DIR, "test")
ensure_directory(TEST_DIR, logger)

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(TRAIN_DIR, label_type)
    ensure_directory(dir_name, logger)
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

data = pad_sequences(sequences, maxlen=MAX_LEN)

labels = np.asarray(labels)
logger.info('Shape of data tensor: {}'.format(data.shape))
logger.info('Shape of label tensor: {}'.format(labels.shape))

logger.info("Shuffling records...")
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

logger.info("Creating training and validation sets")
x_train = data[:TRAIN_SIZE]
y_train = labels[:TRAIN_SIZE]
x_val = data[TRAIN_SIZE:(TRAIN_SIZE + VAL_SIZE)]
y_val = labels[TRAIN_SIZE:(TRAIN_SIZE + VAL_SIZE)]

logger.info("Loading Embeddings...")
logger.info("Embeddings File: {}".format(EMBEDDINGS_PATH))
embeddings_index = {}
i = 0



with open(EMBEDDINGS_PATH) as fh:
    for line in fh:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        if i % 100000 == 0:
            logger.info("loaded {} embeddings...".format(i))
        i += 1
        
logger.info('Found {} word vectors'.format(len(embeddings_index)))

logger.info("Converting tokerizer index to embeddings matrix")
embedding_matrix = np.zeros((NUM_WORDS, EMBEDDINGS_DIMENSIONS))
for word, i in word_index.items():
    if i < NUM_WORDS:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        
logger.info("Defining Model")
model = Sequential()
model.add(Embedding(NUM_WORDS, EMBEDDINGS_DIMENSIONS, input_length = MAX_LEN))
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

logger.info("Loading embeddings matrix into first model layer (Embedding Layer)")
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

logger.info("Fitting model...")
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc', f1_score])
history = model.fit(x_train, y_train,
                   epochs=EPOCHS,
                   batch_size=BATCH_SIZE,
                   validation_data=(x_val, y_val))

if SAVE_MODEL:
    model.save_weights(MODEL_FILE)
    logger.info("model weights saved at: {}".format(MODEL_FILE))
else:
    logger.info("did NOT save model weights at {}, change flag if you meant to".format(MODEL_FILE))