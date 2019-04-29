import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from shared.logger import get_logger, get_filename, get_start_time, get_curr_time
from shared.utility import open_plot, ensure_directory, get_tensorboard_directory, get_model_file, limit_gpu_memory

logger = get_logger()
limit_gpu_memory()
with tf.Session() as sess:

    # Constants and Config for index, features, and label
    PROJECT_NAME="scratch"

    x = np.random.rand(3,2)

    print("x is {}".format(x))
    x_ = tf.constant(x)
    print("x_ is {}".format(sess.run([x_])))
    print("K.mean(x_) is {}".format(sess.run([K.mean(x_)])))