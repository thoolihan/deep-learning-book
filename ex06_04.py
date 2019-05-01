from tensorflow.keras.layers import Embedding
from tensorflow.keras.datasets import imdb
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
INPUT_DIR="data/{}".format(PROJECT_NAME)
OUTPUT_DIR="output/{}".format(PROJECT_NAME)
ensure_directory(OUTPUT_DIR, logger)
MODEL_FILE = get_model_file(OUTPUT_DIR)
EPOCHS = 30
BATCH_SIZE=64
NUM_WORDS = 1000
DIMS = 8 #64
MAX_FEATURES = 10000
MAX_LEN = 25
TBLOGDIR=get_tensorboard_directory(PROJECT_NAME)
logger.info("Tensorboard is at: {}".format(TBLOGDIR))

def show_shapes(msg=""):
    logger.info(msg)
    logger.info("x_train.shape: {}".format(x_train.shape))
    logger.info("y_train.shape: {}".format(y_train.shape))
    logger.info("x_test.shape: {}".format(x_test.shape))
    logger.info("y_test.shape: {}".format(y_test.shape))    

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = MAX_FEATURES)

show_shapes("after imdb load")

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_LEN)

show_shapes("after padding sequences")

model = Sequential()
model.add(Embedding(NUM_WORDS, DIMS, input_length=MAX_LEN))

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc', f1_score])
model.summary()

history = model.fit(x_train, 
                    y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_split=0.2,
                    callbacks=[TensorBoard(log_dir=TBLOGDIR)])

