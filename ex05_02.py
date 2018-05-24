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
PROJECT_NAME="dogs_cats"
INPUT_DIR="data/{}".format(PROJECT_NAME)
OUTPUT_DIR="output/{}".format(PROJECT_NAME)
ensure_directory(INPUT_DIR, logger)
ensure_directory(OUTPUT_DIR, logger)
DRO = 0.25
L2R = 0.0001
HLAYER = 64
EPOCHS = 5
BATCH_SIZE = 64

# load data
